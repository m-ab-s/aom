/*
 * Copyright (c) 2017, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include "av1/encoder/encodetxb.h"

#include "aom_ports/mem.h"
#include "av1/common/blockd.h"
#include "av1/common/idct.h"
#include "av1/common/pred_common.h"
#include "av1/common/scan.h"
#include "av1/encoder/bitstream.h"
#include "av1/encoder/cost.h"
#include "av1/encoder/encodeframe.h"
#include "av1/encoder/hash.h"
#include "av1/encoder/rdopt.h"
#include "av1/encoder/tokenize.h"

static int hbt_needs_init = 1;
static CRC_CALCULATOR crc_calculator;
static const int HBT_EOB = 16;            // also the length in opt_qcoeff
static const int HBT_TABLE_SIZE = 65536;  // 16 bit: holds 65536 'arrays'
static const int HBT_ARRAY_LENGTH = 256;  // 8 bit: 256 entries
// If removed in hbt_create_hashes or increased beyond int8_t, widen deltas type
static const int HBT_KICKOUT = 3;

typedef struct OptTxbQcoeff {
  // Use larger type if larger/no kickout value is used in hbt_create_hashes
  int8_t deltas[16];
  uint32_t hbt_qc_hash;
  uint32_t hbt_ctx_hash;
  int init;
  int rate_cost;
} OptTxbQcoeff;

OptTxbQcoeff *hbt_hash_table;

typedef struct LevelDownStats {
  int update;
  tran_low_t low_qc;
  tran_low_t low_dqc;
  int64_t dist0;
  int rate;
  int rate_low;
  int64_t dist;
  int64_t dist_low;
  int64_t rd;
  int64_t rd_low;
  int64_t nz_rd;
  int64_t rd_diff;
  int cost_diff;
  int64_t dist_diff;
  int new_eob;
} LevelDownStats;

void av1_alloc_txb_buf(AV1_COMP *cpi) {
  AV1_COMMON *cm = &cpi->common;
  int size = ((cm->mi_rows >> cm->seq_params.mib_size_log2) + 1) *
             ((cm->mi_cols >> cm->seq_params.mib_size_log2) + 1);

  av1_free_txb_buf(cpi);
  // TODO(jingning): This should be further reduced.
  CHECK_MEM_ERROR(cm, cpi->coeff_buffer_base,
                  aom_malloc(sizeof(*cpi->coeff_buffer_base) * size));
}

void av1_free_txb_buf(AV1_COMP *cpi) { aom_free(cpi->coeff_buffer_base); }

void av1_set_coeff_buffer(const AV1_COMP *const cpi, MACROBLOCK *const x,
                          int mi_row, int mi_col) {
  const AV1_COMMON *const cm = &cpi->common;
  const int num_planes = av1_num_planes(cm);
  int mib_size_log2 = cm->seq_params.mib_size_log2;
  int stride = (cm->mi_cols >> mib_size_log2) + 1;
  int offset = (mi_row >> mib_size_log2) * stride + (mi_col >> mib_size_log2);
  CB_COEFF_BUFFER *coeff_buf = &cpi->coeff_buffer_base[offset];
  const int txb_offset = x->cb_offset / (TX_SIZE_W_MIN * TX_SIZE_H_MIN);
  for (int plane = 0; plane < num_planes; ++plane) {
    x->mbmi_ext->tcoeff[plane] = coeff_buf->tcoeff[plane] + x->cb_offset;
    x->mbmi_ext->eobs[plane] = coeff_buf->eobs[plane] + txb_offset;
    x->mbmi_ext->txb_skip_ctx[plane] =
        coeff_buf->txb_skip_ctx[plane] + txb_offset;
    x->mbmi_ext->dc_sign_ctx[plane] =
        coeff_buf->dc_sign_ctx[plane] + txb_offset;
  }
}

static void write_golomb(aom_writer *w, int level) {
  int x = level + 1;
  int i = x;
  int length = 0;

  while (i) {
    i >>= 1;
    ++length;
  }
  assert(length > 0);

  for (i = 0; i < length - 1; ++i) aom_write_bit(w, 0);

  for (i = length - 1; i >= 0; --i) aom_write_bit(w, (x >> i) & 0x01);
}

static INLINE tran_low_t get_lower_coeff(tran_low_t qc) {
  if (qc == 0) {
    return 0;
  }
  return qc > 0 ? qc - 1 : qc + 1;
}

static INLINE tran_low_t qcoeff_to_dqcoeff(tran_low_t qc, int coeff_idx,
                                           int dqv, int shift,
                                           const qm_val_t *iqmatrix) {
  int sgn = qc < 0 ? -1 : 1;
  if (iqmatrix != NULL)
    dqv =
        ((iqmatrix[coeff_idx] * dqv) + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS;
  return sgn * ((abs(qc) * dqv) >> shift);
}

static INLINE int64_t get_coeff_dist(tran_low_t tcoeff, tran_low_t dqcoeff,
                                     int shift) {
  const int64_t diff = (tcoeff - dqcoeff) * (1 << shift);
  const int64_t error = diff * diff;
  return error;
}

void av1_update_eob_context(int eob, int seg_eob, TX_SIZE tx_size,
                            TX_TYPE tx_type, PLANE_TYPE plane,
                            FRAME_CONTEXT *ec_ctx, FRAME_COUNTS *counts,
                            uint8_t allow_update_cdf) {
  int eob_extra, dummy;
  const int eob_pt = get_eob_pos_token(eob, &eob_extra);
  const int max_eob_pt = get_eob_pos_token(seg_eob, &dummy);
  TX_SIZE txs_ctx = get_txsize_entropy_ctx(tx_size);

  (void)max_eob_pt;
  const int eob_multi_size = txsize_log2_minus4[tx_size];
  const int eob_multi_ctx = (tx_type_to_class[tx_type] == TX_CLASS_2D) ? 0 : 1;

  switch (eob_multi_size) {
    case 0:
      ++counts->eob_multi16[plane][eob_multi_ctx][eob_pt - 1];
      if (allow_update_cdf)
        update_cdf(ec_ctx->eob_flag_cdf16[plane][eob_multi_ctx], eob_pt - 1, 5);
      break;
    case 1:
      ++counts->eob_multi32[plane][eob_multi_ctx][eob_pt - 1];
      if (allow_update_cdf)
        update_cdf(ec_ctx->eob_flag_cdf32[plane][eob_multi_ctx], eob_pt - 1, 6);
      break;
    case 2:
      ++counts->eob_multi64[plane][eob_multi_ctx][eob_pt - 1];
      if (allow_update_cdf)
        update_cdf(ec_ctx->eob_flag_cdf64[plane][eob_multi_ctx], eob_pt - 1, 7);
      break;
    case 3:
      ++counts->eob_multi128[plane][eob_multi_ctx][eob_pt - 1];
      if (allow_update_cdf) {
        update_cdf(ec_ctx->eob_flag_cdf128[plane][eob_multi_ctx], eob_pt - 1,
                   8);
      }
      break;
    case 4:
      ++counts->eob_multi256[plane][eob_multi_ctx][eob_pt - 1];
      if (allow_update_cdf) {
        update_cdf(ec_ctx->eob_flag_cdf256[plane][eob_multi_ctx], eob_pt - 1,
                   9);
      }
      break;
    case 5:
      ++counts->eob_multi512[plane][eob_multi_ctx][eob_pt - 1];
      if (allow_update_cdf) {
        update_cdf(ec_ctx->eob_flag_cdf512[plane][eob_multi_ctx], eob_pt - 1,
                   10);
      }
      break;
    case 6:
    default:
      ++counts->eob_multi1024[plane][eob_multi_ctx][eob_pt - 1];
      if (allow_update_cdf) {
        update_cdf(ec_ctx->eob_flag_cdf1024[plane][eob_multi_ctx], eob_pt - 1,
                   11);
      }
      break;
  }

  if (k_eob_offset_bits[eob_pt] > 0) {
    int eob_shift = k_eob_offset_bits[eob_pt] - 1;
    int bit = (eob_extra & (1 << eob_shift)) ? 1 : 0;
#if CONFIG_ENTROPY_STATS
    counts->eob_extra[txs_ctx][plane][eob_pt][bit]++;
#endif  // CONFIG_ENTROPY_STATS
    if (allow_update_cdf)
      update_cdf(ec_ctx->eob_extra_cdf[txs_ctx][plane][eob_pt], bit, 2);
  }
}

static int get_eob_cost(int eob, int seg_eob,
                        const LV_MAP_EOB_COST *txb_eob_costs,
                        const LV_MAP_COEFF_COST *txb_costs, TX_TYPE tx_type) {
  int eob_extra, dummy;
  const int eob_pt = get_eob_pos_token(eob, &eob_extra);
  const int max_eob_pt = get_eob_pos_token(seg_eob, &dummy);
  int eob_cost = 0;
  (void)max_eob_pt;
  const int eob_multi_ctx = (tx_type_to_class[tx_type] == TX_CLASS_2D) ? 0 : 1;
  eob_cost = txb_eob_costs->eob_cost[eob_multi_ctx][eob_pt - 1];

  if (k_eob_offset_bits[eob_pt] > 0) {
    const int eob_shift = k_eob_offset_bits[eob_pt] - 1;
    const int bit = (eob_extra & (1 << eob_shift)) ? 1 : 0;
    eob_cost += txb_costs->eob_extra_cost[eob_pt][bit];
    const int offset_bits = k_eob_offset_bits[eob_pt];
    if (offset_bits > 1) eob_cost += av1_cost_literal(offset_bits - 1);
  }
  return eob_cost;
}

static int get_coeff_cost(const tran_low_t qc, const int scan_idx,
                          const int is_eob, const TxbInfo *const txb_info,
                          const LV_MAP_COEFF_COST *const txb_costs,
                          const int coeff_ctx);

static void get_dist_cost_stats(LevelDownStats *const stats, const int scan_idx,
                                const int is_eob,
                                const LV_MAP_COEFF_COST *const txb_costs,
                                const TxbInfo *const txb_info,
                                int has_nz_tail) {
  const int16_t *const scan = txb_info->scan_order->scan;
  const int coeff_idx = scan[scan_idx];
  const tran_low_t qc = txb_info->qcoeff[coeff_idx];
  const uint8_t *const levels = txb_info->levels;
  stats->new_eob = -1;
  stats->update = 0;
  stats->rd_low = 0;
  stats->rd = 0;
  stats->nz_rd = 0;
  stats->dist_low = 0;
  stats->rate_low = 0;
  stats->low_qc = 0;

  const tran_low_t tqc = txb_info->tcoeff[coeff_idx];
  const int dqv = txb_info->dequant[coeff_idx != 0];

  const int coeff_ctx =
      get_nz_map_ctx(levels, coeff_idx, txb_info->bwl, txb_info->height,
                     scan_idx, is_eob, txb_info->tx_size, txb_info->tx_type);
  const int qc_cost =
      get_coeff_cost(qc, scan_idx, is_eob, txb_info, txb_costs, coeff_ctx);
  if (qc == 0) {
    stats->dist = 0;
    stats->rate = qc_cost;
    return;
  } else {
    const tran_low_t dqc = qcoeff_to_dqcoeff(
        qc, coeff_idx, dqv, txb_info->shift, txb_info->iqmatrix);
    const int64_t dqc_dist = get_coeff_dist(tqc, dqc, txb_info->shift);

    // distortion difference when coefficient is quantized to 0
    const tran_low_t dqc0 = qcoeff_to_dqcoeff(
        0, coeff_idx, dqv, txb_info->shift, txb_info->iqmatrix);

    stats->dist0 = get_coeff_dist(tqc, dqc0, txb_info->shift);
    stats->dist = dqc_dist - stats->dist0;
    stats->rate = qc_cost;
  }

  stats->rd = RDCOST(txb_info->rdmult, stats->rate, stats->dist);

  stats->low_qc = get_lower_coeff(qc);

  if (is_eob && stats->low_qc == 0) {
    stats->rd_low = stats->rd;  // disable selection of low_qc in this case.
  } else {
    if (stats->low_qc == 0) {
      stats->dist_low = 0;
    } else {
      stats->low_dqc = qcoeff_to_dqcoeff(stats->low_qc, coeff_idx, dqv,
                                         txb_info->shift, txb_info->iqmatrix);
      const int64_t low_dqc_dist =
          get_coeff_dist(tqc, stats->low_dqc, txb_info->shift);
      stats->dist_low = low_dqc_dist - stats->dist0;
    }
    const int low_qc_cost = get_coeff_cost(stats->low_qc, scan_idx, is_eob,
                                           txb_info, txb_costs, coeff_ctx);
    stats->rate_low = low_qc_cost;
    stats->rd_low = RDCOST(txb_info->rdmult, stats->rate_low, stats->dist_low);
  }
  if ((has_nz_tail < 2) && !is_eob) {
    (void)levels;
    const int coeff_ctx_temp =
        get_nz_map_ctx(levels, coeff_idx, txb_info->bwl, txb_info->height,
                       scan_idx, 1, txb_info->tx_size, txb_info->tx_type);
    const int qc_eob_cost =
        get_coeff_cost(qc, scan_idx, 1, txb_info, txb_costs, coeff_ctx_temp);
    int64_t rd_eob = RDCOST(txb_info->rdmult, qc_eob_cost, stats->dist);
    if (stats->low_qc != 0) {
      const int low_qc_eob_cost = get_coeff_cost(
          stats->low_qc, scan_idx, 1, txb_info, txb_costs, coeff_ctx_temp);
      int64_t rd_eob_low =
          RDCOST(txb_info->rdmult, low_qc_eob_cost, stats->dist_low);
      rd_eob = (rd_eob > rd_eob_low) ? rd_eob_low : rd_eob;
    }

    stats->nz_rd = AOMMIN(stats->rd_low, stats->rd) - rd_eob;
  }
}

static INLINE void update_qcoeff(const int coeff_idx, const tran_low_t qc,
                                 const TxbInfo *const txb_info) {
  txb_info->qcoeff[coeff_idx] = qc;
  txb_info->levels[get_padded_idx(coeff_idx, txb_info->bwl)] =
      (uint8_t)clamp(abs(qc), 0, INT8_MAX);
}

static INLINE void update_coeff(const int coeff_idx, const tran_low_t qc,
                                const TxbInfo *const txb_info) {
  update_qcoeff(coeff_idx, qc, txb_info);
  const int dqv = txb_info->dequant[coeff_idx != 0];
  txb_info->dqcoeff[coeff_idx] = qcoeff_to_dqcoeff(
      qc, coeff_idx, dqv, txb_info->shift, txb_info->iqmatrix);
}

static INLINE void av1_txb_init_levels(const tran_low_t *const coeff,
                                       const int width, const int height,
                                       uint8_t *const levels) {
  const int stride = width + TX_PAD_HOR;
  uint8_t *ls = levels;

  memset(levels - TX_PAD_TOP * stride, 0,
         sizeof(*levels) * TX_PAD_TOP * stride);
  memset(levels + stride * height, 0,
         sizeof(*levels) * (TX_PAD_BOTTOM * stride + TX_PAD_END));

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      *ls++ = (uint8_t)clamp(abs(coeff[i * width + j]), 0, INT8_MAX);
    }
    for (int j = 0; j < TX_PAD_HOR; j++) {
      *ls++ = 0;
    }
  }
}

void av1_get_nz_map_contexts_c(const uint8_t *const levels,
                               const int16_t *const scan, const uint16_t eob,
                               const TX_SIZE tx_size, const TX_TYPE tx_type,
                               int8_t *const coeff_contexts) {
  const int bwl = get_txb_bwl(tx_size);
  const int height = get_txb_high(tx_size);
  for (int i = 0; i < eob; ++i) {
    const int pos = scan[i];
    coeff_contexts[pos] = get_nz_map_ctx(levels, pos, bwl, height, i,
                                         i == eob - 1, tx_size, tx_type);
  }
}

void av1_write_coeffs_txb(const AV1_COMMON *const cm, MACROBLOCKD *xd,
                          aom_writer *w, int blk_row, int blk_col, int plane,
                          TX_SIZE tx_size, const tran_low_t *tcoeff,
                          uint16_t eob, TXB_CTX *txb_ctx) {
  const PLANE_TYPE plane_type = get_plane_type(plane);
  const TX_SIZE txs_ctx = get_txsize_entropy_ctx(tx_size);
  const TX_TYPE tx_type = av1_get_tx_type(plane_type, xd, blk_row, blk_col,
                                          tx_size, cm->reduced_tx_set_used);
  const SCAN_ORDER *const scan_order = get_scan(tx_size, tx_type);
  const int16_t *const scan = scan_order->scan;
  const int seg_eob = av1_get_max_eob(tx_size);
  int c;
  const int bwl = get_txb_bwl(tx_size);
  const int width = get_txb_wide(tx_size);
  const int height = get_txb_high(tx_size);
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
  uint8_t levels_buf[TX_PAD_2D];
  uint8_t *const levels = set_levels(levels_buf, width);
  DECLARE_ALIGNED(16, int8_t, coeff_contexts[MAX_TX_SQUARE]);

  aom_write_symbol(w, eob == 0,
                   ec_ctx->txb_skip_cdf[txs_ctx][txb_ctx->txb_skip_ctx], 2);
  if (plane == 0 && eob == 0) {
    assert(tx_type == DCT_DCT);
  }
  if (eob == 0) return;

  av1_txb_init_levels(tcoeff, width, height, levels);

  av1_write_tx_type(cm, xd, blk_row, blk_col, plane, tx_size, w);

  int eob_extra, dummy;
  const int eob_pt = get_eob_pos_token(eob, &eob_extra);
  const int max_eob_pt = get_eob_pos_token(seg_eob, &dummy);

  (void)max_eob_pt;
  const int eob_multi_size = txsize_log2_minus4[tx_size];
  const int eob_multi_ctx = (tx_type_to_class[tx_type] == TX_CLASS_2D) ? 0 : 1;
  switch (eob_multi_size) {
    case 0:
      aom_write_symbol(w, eob_pt - 1,
                       ec_ctx->eob_flag_cdf16[plane_type][eob_multi_ctx], 5);
      break;
    case 1:
      aom_write_symbol(w, eob_pt - 1,
                       ec_ctx->eob_flag_cdf32[plane_type][eob_multi_ctx], 6);
      break;
    case 2:
      aom_write_symbol(w, eob_pt - 1,
                       ec_ctx->eob_flag_cdf64[plane_type][eob_multi_ctx], 7);
      break;
    case 3:
      aom_write_symbol(w, eob_pt - 1,
                       ec_ctx->eob_flag_cdf128[plane_type][eob_multi_ctx], 8);
      break;
    case 4:
      aom_write_symbol(w, eob_pt - 1,
                       ec_ctx->eob_flag_cdf256[plane_type][eob_multi_ctx], 9);
      break;
    case 5:
      aom_write_symbol(w, eob_pt - 1,
                       ec_ctx->eob_flag_cdf512[plane_type][eob_multi_ctx], 10);
      break;
    default:
      aom_write_symbol(w, eob_pt - 1,
                       ec_ctx->eob_flag_cdf1024[plane_type][eob_multi_ctx], 11);
      break;
  }

  if (k_eob_offset_bits[eob_pt] > 0) {
    int eob_shift = k_eob_offset_bits[eob_pt] - 1;
    int bit = (eob_extra & (1 << eob_shift)) ? 1 : 0;
    aom_write_symbol(w, bit, ec_ctx->eob_extra_cdf[txs_ctx][plane_type][eob_pt],
                     2);
    for (int i = 1; i < k_eob_offset_bits[eob_pt]; i++) {
      eob_shift = k_eob_offset_bits[eob_pt] - 1 - i;
      bit = (eob_extra & (1 << eob_shift)) ? 1 : 0;
      aom_write_bit(w, bit);
    }
  }

  av1_get_nz_map_contexts(levels, scan, eob, tx_size, tx_type, coeff_contexts);

  for (c = eob - 1; c >= 0; --c) {
    const int pos = scan[c];
    const int coeff_ctx = coeff_contexts[pos];
    const tran_low_t v = tcoeff[pos];
    const tran_low_t level = abs(v);

    if (c == eob - 1) {
      aom_write_symbol(
          w, AOMMIN(level, 3) - 1,
          ec_ctx->coeff_base_eob_cdf[txs_ctx][plane_type][coeff_ctx], 3);
    } else {
      aom_write_symbol(w, AOMMIN(level, 3),
                       ec_ctx->coeff_base_cdf[txs_ctx][plane_type][coeff_ctx],
                       4);
    }
    if (level > NUM_BASE_LEVELS) {
      // level is above 1.
      const int base_range = level - 1 - NUM_BASE_LEVELS;
      const int br_ctx = get_br_ctx(levels, pos, bwl, tx_type);
      for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
        const int k = AOMMIN(base_range - idx, BR_CDF_SIZE - 1);
        aom_write_symbol(
            w, k,
            ec_ctx->coeff_br_cdf[AOMMIN(txs_ctx, TX_32X32)][plane_type][br_ctx],
            BR_CDF_SIZE);
        if (k < BR_CDF_SIZE - 1) break;
      }
    }
  }

  // Loop to code all signs in the transform block,
  // starting with the sign of DC (if applicable)
  for (c = 0; c < eob; ++c) {
    const tran_low_t v = tcoeff[scan[c]];
    const tran_low_t level = abs(v);
    const int sign = (v < 0) ? 1 : 0;
    if (level) {
      if (c == 0) {
        aom_write_symbol(
            w, sign, ec_ctx->dc_sign_cdf[plane_type][txb_ctx->dc_sign_ctx], 2);
      } else {
        aom_write_bit(w, sign);
      }
      if (level > COEFF_BASE_RANGE + NUM_BASE_LEVELS) {
        const int pos = scan[c];
        write_golomb(w,
                     abs(tcoeff[pos]) - COEFF_BASE_RANGE - 1 - NUM_BASE_LEVELS);
      }
    }
  }
}

typedef struct encode_txb_args {
  const AV1_COMMON *cm;
  MACROBLOCK *x;
  aom_writer *w;
} ENCODE_TXB_ARGS;

static void write_coeffs_txb_wrap(const AV1_COMMON *cm, MACROBLOCK *x,
                                  aom_writer *w, int plane, int block,
                                  int blk_row, int blk_col, TX_SIZE tx_size) {
  MACROBLOCKD *xd = &x->e_mbd;
  tran_low_t *tcoeff = BLOCK_OFFSET(x->mbmi_ext->tcoeff[plane], block);
  uint16_t eob = x->mbmi_ext->eobs[plane][block];
  TXB_CTX txb_ctx = { x->mbmi_ext->txb_skip_ctx[plane][block],
                      x->mbmi_ext->dc_sign_ctx[plane][block] };
  av1_write_coeffs_txb(cm, xd, w, blk_row, blk_col, plane, tx_size, tcoeff, eob,
                       &txb_ctx);
}

void av1_write_coeffs_mb(const AV1_COMMON *const cm, MACROBLOCK *x, int mi_row,
                         int mi_col, aom_writer *w, BLOCK_SIZE bsize) {
  MACROBLOCKD *xd = &x->e_mbd;
  const int num_planes = av1_num_planes(cm);
  int block[MAX_MB_PLANE] = { 0 };
  int row, col;
  const struct macroblockd_plane *const y_pd = &xd->plane[0];
  const BLOCK_SIZE plane_bsize = get_plane_block_size(bsize, y_pd);
  const int max_blocks_wide = max_block_wide(xd, plane_bsize, 0);
  const int max_blocks_high = max_block_high(xd, plane_bsize, 0);
  const BLOCK_SIZE max_unit_bsize = get_plane_block_size(BLOCK_64X64, y_pd);
  int mu_blocks_wide = block_size_wide[max_unit_bsize] >> tx_size_wide_log2[0];
  int mu_blocks_high = block_size_high[max_unit_bsize] >> tx_size_high_log2[0];
  mu_blocks_wide = AOMMIN(max_blocks_wide, mu_blocks_wide);
  mu_blocks_high = AOMMIN(max_blocks_high, mu_blocks_high);

  for (row = 0; row < max_blocks_high; row += mu_blocks_high) {
    for (col = 0; col < max_blocks_wide; col += mu_blocks_wide) {
      for (int plane = 0; plane < num_planes; ++plane) {
        const struct macroblockd_plane *const pd = &xd->plane[plane];
        if (!is_chroma_reference(mi_row, mi_col, bsize, pd->subsampling_x,
                                 pd->subsampling_y))
          continue;
        const TX_SIZE tx_size = av1_get_tx_size(plane, xd);
        const int stepr = tx_size_high_unit[tx_size];
        const int stepc = tx_size_wide_unit[tx_size];
        const int step = stepr * stepc;

        const int unit_height = ROUND_POWER_OF_TWO(
            AOMMIN(mu_blocks_high + row, max_blocks_high), pd->subsampling_y);
        const int unit_width = ROUND_POWER_OF_TWO(
            AOMMIN(mu_blocks_wide + col, max_blocks_wide), pd->subsampling_x);
        for (int blk_row = row >> pd->subsampling_y; blk_row < unit_height;
             blk_row += stepr) {
          for (int blk_col = col >> pd->subsampling_x; blk_col < unit_width;
               blk_col += stepc) {
            write_coeffs_txb_wrap(cm, x, w, plane, block[plane], blk_row,
                                  blk_col, tx_size);
            block[plane] += step;
          }
        }
      }
    }
  }
}

static INLINE int get_br_cost(tran_low_t abs_qc, int ctx,
                              const int *coeff_lps) {
  const tran_low_t min_level = 1 + NUM_BASE_LEVELS;
  const tran_low_t max_level = 1 + NUM_BASE_LEVELS + COEFF_BASE_RANGE;
  (void)ctx;
  if (abs_qc >= min_level) {
    if (abs_qc >= max_level)
      return coeff_lps[COEFF_BASE_RANGE];  // COEFF_BASE_RANGE * cost0;
    else
      return coeff_lps[(abs_qc - min_level)];  //  * cost0 + cost1;
  } else {
    return 0;
  }
}

// Note: don't call this function when eob is 0.
int av1_cost_coeffs_txb(const AV1_COMMON *const cm, const MACROBLOCK *x,
                        const int plane, const int blk_row, const int blk_col,
                        const int block, const TX_SIZE tx_size,
                        const TXB_CTX *const txb_ctx) {
  const MACROBLOCKD *const xd = &x->e_mbd;
  const TX_SIZE txs_ctx = get_txsize_entropy_ctx(tx_size);
  const PLANE_TYPE plane_type = get_plane_type(plane);
  const TX_TYPE tx_type = av1_get_tx_type(plane_type, xd, blk_row, blk_col,
                                          tx_size, cm->reduced_tx_set_used);
  const MB_MODE_INFO *const mbmi = &xd->mi[0]->mbmi;
  const struct macroblock_plane *p = &x->plane[plane];
  const int eob = p->eobs[block];
  const tran_low_t *const qcoeff = BLOCK_OFFSET(p->qcoeff, block);
  int c, cost;
  const int txb_skip_ctx = txb_ctx->txb_skip_ctx;
  const int bwl = get_txb_bwl(tx_size);
  const int width = get_txb_wide(tx_size);
  const int height = get_txb_high(tx_size);
  const SCAN_ORDER *const scan_order = get_scan(tx_size, tx_type);
  const int16_t *const scan = scan_order->scan;
  uint8_t levels_buf[TX_PAD_2D];
  uint8_t *const levels = set_levels(levels_buf, width);
  DECLARE_ALIGNED(16, int8_t, coeff_contexts[MAX_TX_SQUARE]);
  const LV_MAP_COEFF_COST *const coeff_costs =
      &x->coeff_costs[txs_ctx][plane_type];

  const int eob_multi_size = txsize_log2_minus4[tx_size];
  const LV_MAP_EOB_COST *const eob_costs =
      &x->eob_costs[eob_multi_size][plane_type];
  // eob must be greater than 0 here.
  assert(eob > 0);
  cost = coeff_costs->txb_skip_cost[txb_skip_ctx][0];

  av1_txb_init_levels(qcoeff, width, height, levels);

  cost += av1_tx_type_cost(cm, x, xd, mbmi->sb_type, plane, tx_size, tx_type);

  const int seg_eob = av1_get_max_eob(tx_size);
  int eob_cost = get_eob_cost(eob, seg_eob, eob_costs, coeff_costs, tx_type);
  cost += eob_cost;

  av1_get_nz_map_contexts(levels, scan, eob, tx_size, tx_type, coeff_contexts);

  for (c = eob - 1; c >= 0; --c) {
    const int pos = scan[c];
    const tran_low_t v = qcoeff[pos];
    const int is_nz = (v != 0);
    const int level = abs(v);
    const int coeff_ctx = coeff_contexts[pos];
    if (c == eob - 1) {
      cost += coeff_costs->base_eob_cost[coeff_ctx][AOMMIN(level, 3) - 1];
    } else {
      cost += coeff_costs->base_cost[coeff_ctx][AOMMIN(level, 3)];
    }

    if (is_nz) {
      const int sign = (v < 0) ? 1 : 0;

      // sign bit cost
      if (c == 0) {
        int dc_sign_ctx = txb_ctx->dc_sign_ctx;
        cost += coeff_costs->dc_sign_cost[dc_sign_ctx][sign];
      } else {
        cost += av1_cost_literal(1);
      }
      if (level > NUM_BASE_LEVELS) {
        int ctx;
        ctx = get_br_ctx(levels, pos, bwl, tx_type);
        const int base_range = level - 1 - NUM_BASE_LEVELS;
        if (base_range < COEFF_BASE_RANGE) {
          cost += coeff_costs->lps_cost[ctx][base_range];
        } else {
          cost += coeff_costs->lps_cost[ctx][COEFF_BASE_RANGE];
        }

        if (level >= 1 + NUM_BASE_LEVELS + COEFF_BASE_RANGE) {
          // residual cost
          const int ri = level - COEFF_BASE_RANGE - NUM_BASE_LEVELS;
          const int length = get_msb(ri) + 1;
          cost += av1_cost_literal(2 * length);
        }
      }
    }
  }
  return cost;
}

static INLINE int has_base(tran_low_t qc, int base_idx) {
  const int level = base_idx + 1;
  return abs(qc) >= level;
}

static INLINE int has_br(tran_low_t qc) {
  return abs(qc) >= 1 + NUM_BASE_LEVELS;
}

static INLINE int get_sign_bit_cost(tran_low_t qc, int coeff_idx,
                                    const int (*dc_sign_cost)[2],
                                    int dc_sign_ctx) {
  const int sign = (qc < 0) ? 1 : 0;
  // sign bit cost
  if (coeff_idx == 0) {
    return dc_sign_cost[dc_sign_ctx][sign];
  } else {
    return av1_cost_literal(1);
  }
}
static INLINE int get_golomb_cost(int abs_qc) {
  if (abs_qc >= 1 + NUM_BASE_LEVELS + COEFF_BASE_RANGE) {
    // residual cost
    int r = abs_qc - COEFF_BASE_RANGE - NUM_BASE_LEVELS;
    int ri = r;
    int length = 0;

    while (ri) {
      ri >>= 1;
      ++length;
    }

    return av1_cost_literal(2 * length - 1);
  } else {
    return 0;
  }
}

static INLINE void set_eob(TxbInfo *txb_info, int eob) {
  txb_info->eob = eob;
  txb_info->seg_eob = av1_get_max_eob(txb_info->tx_size);
}

static int get_coeff_cost(const tran_low_t qc, const int scan_idx,
                          const int is_eob, const TxbInfo *const txb_info,
                          const LV_MAP_COEFF_COST *const txb_costs,
                          const int coeff_ctx) {
  const TXB_CTX *txb_ctx = txb_info->txb_ctx;
  const int is_nz = (qc != 0);
  const tran_low_t abs_qc = abs(qc);
  int cost = 0;
  const int16_t *const scan = txb_info->scan_order->scan;
  const int pos = scan[scan_idx];

  if (is_eob) {
    cost += txb_costs->base_eob_cost[coeff_ctx][AOMMIN(abs_qc, 3) - 1];
  } else {
    cost += txb_costs->base_cost[coeff_ctx][AOMMIN(abs_qc, 3)];
  }
  if (is_nz) {
    cost += get_sign_bit_cost(qc, scan_idx, txb_costs->dc_sign_cost,
                              txb_ctx->dc_sign_ctx);

    if (abs_qc > NUM_BASE_LEVELS) {
      const int ctx =
          get_br_ctx(txb_info->levels, pos, txb_info->bwl, txb_info->tx_type);
      cost += get_br_cost(abs_qc, ctx, txb_costs->lps_cost[ctx]);
      cost += get_golomb_cost(abs_qc);
    }
  }
  return cost;
}

static int optimize_txb(TxbInfo *txb_info, const LV_MAP_COEFF_COST *txb_costs,
                        const LV_MAP_EOB_COST *txb_eob_costs,
                        TxbCache *txb_cache, int dry_run, int fast_mode,
                        int *rate_cost) {
  (void)fast_mode;
  (void)txb_cache;
  int update = 0;
  // return update;  // TODO(DKHE): training only.
  if (txb_info->eob == 0) return update;
  const int max_eob = av1_get_max_eob(txb_info->tx_size);

  tran_low_t *org_qcoeff = txb_info->qcoeff;
  tran_low_t *org_dqcoeff = txb_info->dqcoeff;
  uint8_t *const org_levels = txb_info->levels;

  tran_low_t tmp_qcoeff[MAX_TX_SQUARE];
  tran_low_t tmp_dqcoeff[MAX_TX_SQUARE];
  uint8_t tmp_levels_buf[TX_PAD_2D];
  uint8_t *const tmp_levels = set_levels(tmp_levels_buf, txb_info->width);
  const int org_eob = txb_info->eob;
  if (dry_run) {
    const int stride = txb_info->width + TX_PAD_HOR;
    const int levels_size =

        (stride * (txb_info->height + TX_PAD_VER) + TX_PAD_END);
    memcpy(tmp_qcoeff, org_qcoeff, sizeof(org_qcoeff[0]) * max_eob);
    memcpy(tmp_dqcoeff, org_dqcoeff, sizeof(org_dqcoeff[0]) * max_eob);
    memcpy(tmp_levels, org_levels - TX_PAD_TOP * stride,
           sizeof(org_levels[0]) * levels_size);
    txb_info->qcoeff = tmp_qcoeff;
    txb_info->dqcoeff = tmp_dqcoeff;
    txb_info->levels = tmp_levels;
  }

  const int16_t *const scan = txb_info->scan_order->scan;

  // forward optimize the nz_map`
  const int init_eob = txb_info->eob;
  const int seg_eob = txb_info->seg_eob;
  const int eob_cost = get_eob_cost(init_eob, seg_eob, txb_eob_costs, txb_costs,
                                    txb_info->tx_type);

  // backward optimize the level-k map
  int accu_rate = eob_cost;
  int64_t accu_dist = 0;
  int64_t prev_eob_rd_cost = INT64_MAX;
  int64_t cur_eob_rd_cost = 0;
  int8_t has_nz_tail = 0;

  for (int si = init_eob - 1; si >= 0; --si) {
    const int coeff_idx = scan[si];
    tran_low_t qc = txb_info->qcoeff[coeff_idx];

    LevelDownStats stats;
    get_dist_cost_stats(&stats, si, si == init_eob - 1, txb_costs, txb_info,
                        has_nz_tail);

    if (qc == 0) {
      accu_rate += stats.rate;
    } else {
      if (has_nz_tail < 2) {
        if (si == init_eob - 1) {
          if ((stats.rd_low < stats.rd) && (stats.low_qc != 0)) {
            update = 1;
            update_coeff(coeff_idx, stats.low_qc, txb_info);
            accu_rate += stats.rate_low;
            accu_dist += stats.dist_low;
          } else {
            accu_rate += stats.rate;
            accu_dist += stats.dist;
          }
          continue;
        } else {
          // check if it is better to make this the last significant coefficient
          int cur_eob_rate = get_eob_cost(si + 1, seg_eob, txb_eob_costs,
                                          txb_costs, txb_info->tx_type);
          cur_eob_rd_cost = RDCOST(txb_info->rdmult, cur_eob_rate, 0);
          prev_eob_rd_cost =
              RDCOST(txb_info->rdmult, accu_rate, accu_dist) + stats.nz_rd;
          if (cur_eob_rd_cost <= prev_eob_rd_cost) {
            update = 1;
            for (int j = si + 1; j < txb_info->eob; j++) {
              const int coeff_pos_j = scan[j];
              update_coeff(coeff_pos_j, 0, txb_info);
            }
            txb_info->eob = si + 1;

            // rerun cost calculation due to change of eob
            accu_rate = cur_eob_rate;
            accu_dist = 0;
            get_dist_cost_stats(&stats, si, 1, txb_costs, txb_info,
                                has_nz_tail);
            if ((stats.rd_low < stats.rd) && (stats.low_qc != 0)) {
              update = 1;
              update_coeff(coeff_idx, stats.low_qc, txb_info);
              accu_rate += stats.rate_low;
              accu_dist += stats.dist_low;
            } else {
              accu_rate += stats.rate;
              accu_dist += stats.dist;
            }

            // reset non zero tail when new eob is found
            has_nz_tail = 0;
            continue;
          }
        }
      }

      int bUpdCoeff = 0;
      if (stats.rd_low < stats.rd) {
        if ((si < txb_info->eob - 1)) {
          bUpdCoeff = 1;
          update = 1;
        }
      } else {
        ++has_nz_tail;
      }

      if (bUpdCoeff) {
        update_coeff(coeff_idx, stats.low_qc, txb_info);
        accu_rate += stats.rate_low;
        accu_dist += stats.dist_low;
      } else {
        accu_rate += stats.rate;
        accu_dist += stats.dist;
      }
    }
  }  // for (si)

  int non_zero_blk_rate =
      txb_costs->txb_skip_cost[txb_info->txb_ctx->txb_skip_ctx][0];
  prev_eob_rd_cost =
      RDCOST(txb_info->rdmult, accu_rate + non_zero_blk_rate, accu_dist);

  int zero_blk_rate =
      txb_costs->txb_skip_cost[txb_info->txb_ctx->txb_skip_ctx][1];
  int64_t zero_blk_rd_cost = RDCOST(txb_info->rdmult, zero_blk_rate, 0);
  if (zero_blk_rd_cost <= prev_eob_rd_cost) {
    update = 1;
    for (int j = 0; j < txb_info->eob; j++) {
      const int coeff_pos_j = scan[j];
      update_coeff(coeff_pos_j, 0, txb_info);
    }
    txb_info->eob = 0;
  }

  // record total rate cost
  *rate_cost = zero_blk_rd_cost <= prev_eob_rd_cost
                   ? zero_blk_rate
                   : accu_rate + non_zero_blk_rate;

  if (dry_run) {
    txb_info->qcoeff = org_qcoeff;
    txb_info->dqcoeff = org_dqcoeff;
    txb_info->levels = org_levels;
    set_eob(txb_info, org_eob);
  }
  return update;
}

// These numbers are empirically obtained.
static const int plane_rd_mult[REF_TYPES][PLANE_TYPES] = {
  { 17, 13 },
  { 16, 10 },
};

void hbt_init() {
  hbt_hash_table =
      aom_malloc(sizeof(OptTxbQcoeff) * HBT_TABLE_SIZE * HBT_ARRAY_LENGTH);
  memset(hbt_hash_table, 0,
         sizeof(OptTxbQcoeff) * HBT_TABLE_SIZE * HBT_ARRAY_LENGTH);
  av1_crc_calculator_init(&crc_calculator, 31, 0x5D6DCB);  // 31 bit: qc & ctx

  hbt_needs_init = 0;
}

void hbt_destroy() { aom_free(hbt_hash_table); }

int hbt_hash_miss(uint32_t hbt_ctx_hash, uint32_t hbt_qc_hash,
                  TxbInfo *txb_info, const LV_MAP_COEFF_COST *txb_costs,
                  const LV_MAP_EOB_COST *txb_eob_costs,
                  const struct macroblock_plane *p, int block, int fast_mode,
                  int *rate_cost) {
  const int16_t *scan = txb_info->scan_order->scan;
  int prev_eob = txb_info->eob;
  assert(HBT_EOB <= 16);  // Lengthen array if allowing longer eob.
  int32_t prev_coeff[16];
  for (int i = 0; i < prev_eob; i++) {
    prev_coeff[i] = txb_info->qcoeff[scan[i]];
  }
  for (int i = prev_eob; i < HBT_EOB; i++) {
    prev_coeff[i] = 0;  // For compiler piece of mind.
  }

  av1_txb_init_levels(txb_info->qcoeff, txb_info->width, txb_info->height,
                      txb_info->levels);

  const int update = optimize_txb(txb_info, txb_costs, txb_eob_costs, NULL, 0,
                                  fast_mode, rate_cost);

  // Overwrite old entry
  uint16_t hbt_table_index = hbt_ctx_hash % HBT_TABLE_SIZE;
  uint16_t hbt_array_index = hbt_qc_hash % HBT_ARRAY_LENGTH;
  hbt_hash_table[hbt_table_index * HBT_ARRAY_LENGTH + hbt_array_index]
      .rate_cost = *rate_cost;
  hbt_hash_table[hbt_table_index * HBT_ARRAY_LENGTH + hbt_array_index].init = 1;
  hbt_hash_table[hbt_table_index * HBT_ARRAY_LENGTH + hbt_array_index]
      .hbt_qc_hash = hbt_qc_hash;
  hbt_hash_table[hbt_table_index * HBT_ARRAY_LENGTH + hbt_array_index]
      .hbt_ctx_hash = hbt_ctx_hash;
  assert(prev_eob >= txb_info->eob);  // eob can't get longer
  for (int i = 0; i < txb_info->eob; i++) {
    // Record how coeff changed. Convention: towards zero is negative.
    if (txb_info->qcoeff[scan[i]] > 0)
      hbt_hash_table[hbt_table_index * HBT_ARRAY_LENGTH + hbt_array_index]
          .deltas[i] = txb_info->qcoeff[scan[i]] - prev_coeff[i];
    else
      hbt_hash_table[hbt_table_index * HBT_ARRAY_LENGTH + hbt_array_index]
          .deltas[i] = prev_coeff[i] - txb_info->qcoeff[scan[i]];
  }
  for (int i = txb_info->eob; i < prev_eob; i++) {
    // If eob got shorter, record that all after it changed to zero.
    if (prev_coeff[i] > 0)
      hbt_hash_table[hbt_table_index * HBT_ARRAY_LENGTH + hbt_array_index]
          .deltas[i] = -prev_coeff[i];
    else
      hbt_hash_table[hbt_table_index * HBT_ARRAY_LENGTH + hbt_array_index]
          .deltas[i] = prev_coeff[i];
  }
  for (int i = prev_eob; i < HBT_EOB; i++) {
    // Record 'no change' after optimized coefficients run out.
    hbt_hash_table[hbt_table_index * HBT_ARRAY_LENGTH + hbt_array_index]
        .deltas[i] = 0;
  }

  if (update) {
    p->eobs[block] = txb_info->eob;
    p->txb_entropy_ctx[block] = av1_get_txb_entropy_context(
        txb_info->qcoeff, txb_info->scan_order, txb_info->eob);
  }
  return txb_info->eob;
}

int hbt_hash_hit(uint32_t hbt_table_index, int hbt_array_index,
                 TxbInfo *txb_info, const struct macroblock_plane *p, int block,
                 int *rate_cost) {
  const int16_t *scan = txb_info->scan_order->scan;
  int new_eob = 0;
  int update = 0;

  for (int i = 0; i < txb_info->eob; i++) {
    // Delta convention is negatives go towards zero, so only apply those ones.
    if (hbt_hash_table[hbt_table_index * HBT_ARRAY_LENGTH + hbt_array_index]
            .deltas[i] < 0) {
      if (txb_info->qcoeff[scan[i]] > 0)
        txb_info->qcoeff[scan[i]] +=
            hbt_hash_table[hbt_table_index * HBT_ARRAY_LENGTH + hbt_array_index]
                .deltas[i];
      else
        txb_info->qcoeff[scan[i]] -=
            hbt_hash_table[hbt_table_index * HBT_ARRAY_LENGTH + hbt_array_index]
                .deltas[i];

      update = 1;
      update_coeff(scan[i], txb_info->qcoeff[scan[i]], txb_info);
    }
    if (txb_info->qcoeff[scan[i]]) new_eob = i + 1;
  }

  // Rate_cost can be calculated here instead (av1_cost_coeffs_txb), but
  // it is expensive and gives little benefit as long as qc_hash is high bit
  *rate_cost =
      hbt_hash_table[hbt_table_index * HBT_ARRAY_LENGTH + hbt_array_index]
          .rate_cost;

  if (update) {
    txb_info->eob = new_eob;
    p->eobs[block] = txb_info->eob;
    p->txb_entropy_ctx[block] = av1_get_txb_entropy_context(
        txb_info->qcoeff, txb_info->scan_order, txb_info->eob);
  }

  return txb_info->eob;
}

int hbt_search_match(uint32_t hbt_ctx_hash, uint32_t hbt_qc_hash,
                     TxbInfo *txb_info, const LV_MAP_COEFF_COST *txb_costs,
                     const LV_MAP_EOB_COST *txb_eob_costs,
                     const struct macroblock_plane *p, int block, int fast_mode,
                     int *rate_cost) {
  // Check for qcoeff match
  int hbt_array_index = hbt_qc_hash % HBT_ARRAY_LENGTH;
  int hbt_table_index = hbt_ctx_hash % HBT_TABLE_SIZE;

  if (hbt_hash_table[hbt_table_index * HBT_ARRAY_LENGTH + hbt_array_index]
              .hbt_qc_hash == hbt_qc_hash &&
      hbt_hash_table[hbt_table_index * HBT_ARRAY_LENGTH + hbt_array_index]
              .hbt_ctx_hash == hbt_ctx_hash &&
      hbt_hash_table[hbt_table_index * HBT_ARRAY_LENGTH + hbt_array_index]
          .init) {
    return hbt_hash_hit(hbt_table_index, hbt_array_index, txb_info, p, block,
                        rate_cost);
  } else {
    return hbt_hash_miss(hbt_ctx_hash, hbt_qc_hash, txb_info, txb_costs,
                         txb_eob_costs, p, block, fast_mode, rate_cost);
  }
}

int hbt_create_hashes(TxbInfo *txb_info, const LV_MAP_COEFF_COST *txb_costs,
                      const LV_MAP_EOB_COST *txb_eob_costs,
                      const struct macroblock_plane *p, int block,
                      int fast_mode, int *rate_cost) {
  // Initialize hash table if needed.
  if (hbt_needs_init) {
    hbt_init();
  }

  //// Hash creation
  uint8_t txb_hash_data[256];  // Asserts below to ensure enough space.
  const int16_t *scan = txb_info->scan_order->scan;
  uint8_t chunk = 0;
  int hash_data_index = 0;

  // Make qc_hash.
  int packing_index = 0;  // needed for packing.
  for (int i = 0; i < txb_info->eob; i++) {
    tran_low_t prechunk = txb_info->qcoeff[scan[i]];

    // Softening: Improves speed. Aligns with signed deltas.
    if (prechunk < 0) prechunk *= -1;

    // Early kick out: Don't apply feature if there are large coeffs:
    // If this kickout value is removed or raised beyond int8_t,
    // widen deltas type in OptTxbQcoeff struct.
    assert((int8_t)HBT_KICKOUT == HBT_KICKOUT);  // If not, widen types.
    if (prechunk > HBT_KICKOUT) {
      av1_txb_init_levels(txb_info->qcoeff, txb_info->width, txb_info->height,
                          txb_info->levels);

      const int update = optimize_txb(txb_info, txb_costs, txb_eob_costs, NULL,
                                      0, fast_mode, rate_cost);

      if (update) {
        p->eobs[block] = txb_info->eob;
        p->txb_entropy_ctx[block] = av1_get_txb_entropy_context(
            txb_info->qcoeff, txb_info->scan_order, txb_info->eob);
      }
      return txb_info->eob;
    }

    // Since coeffs are 0 to 3, only 2 bits are needed: pack into bytes
    if (packing_index == 0) txb_hash_data[hash_data_index] = 0;
    chunk = prechunk << packing_index;
    packing_index += 2;
    txb_hash_data[hash_data_index] |= chunk;

    // Full byte:
    if (packing_index == 8) {
      packing_index = 0;
      hash_data_index++;
    }
  }
  // Needed when packing_index != 0, to include final byte.
  hash_data_index++;
  assert(hash_data_index <= 64);
  // 31 bit qc_hash: index to array
  uint32_t hbt_qc_hash =
      av1_get_crc_value(&crc_calculator, txb_hash_data, hash_data_index);

  // Make ctx_hash.
  hash_data_index = 0;
  tran_low_t prechunk;

  for (int i = 0; i < txb_info->eob; i++) {
    // Save as magnitudes towards or away from zero.
    if (txb_info->tcoeff[scan[i]] >= 0)
      prechunk = txb_info->tcoeff[scan[i]] - txb_info->dqcoeff[scan[i]];
    else
      prechunk = txb_info->dqcoeff[scan[i]] - txb_info->tcoeff[scan[i]];

    chunk = prechunk & 0xff;
    txb_hash_data[hash_data_index++] = chunk;
  }

  // Extra ctx data:
  // Include dequants.
  txb_hash_data[hash_data_index++] = txb_info->dequant[0] & 0xff;
  txb_hash_data[hash_data_index++] = txb_info->dequant[1] & 0xff;
  chunk = txb_info->txb_ctx->txb_skip_ctx & 0xff;
  txb_hash_data[hash_data_index++] = chunk;
  chunk = txb_info->txb_ctx->dc_sign_ctx & 0xff;
  txb_hash_data[hash_data_index++] = chunk;
  // eob
  chunk = txb_info->eob & 0xff;
  txb_hash_data[hash_data_index++] = chunk;
  // rdmult (int64)
  chunk = txb_info->rdmult & 0xff;
  txb_hash_data[hash_data_index++] = chunk;
  // tx_type
  chunk = txb_info->tx_type & 0xff;
  txb_hash_data[hash_data_index++] = chunk;
  // base_eob_cost
  for (int i = 1; i < 3; i++) {  // i = 0 are softened away
    for (int j = 0; j < SIG_COEF_CONTEXTS_EOB; j++) {
      chunk = (txb_costs->base_eob_cost[j][i] & 0xff00) >> 8;
      txb_hash_data[hash_data_index++] = chunk;
    }
  }
  // eob_cost
  for (int i = 0; i < 11; i++) {
    for (int j = 0; j < 2; j++) {
      chunk = (txb_eob_costs->eob_cost[j][i] & 0xff00) >> 8;
      txb_hash_data[hash_data_index++] = chunk;
    }
  }
  // dc_sign_cost
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < DC_SIGN_CONTEXTS; j++) {
      chunk = (txb_costs->dc_sign_cost[j][i] & 0xff00) >> 8;
      txb_hash_data[hash_data_index++] = chunk;
    }
  }

  assert(hash_data_index <= 256);
  // 31 bit ctx_hash: used to index table
  uint32_t hbt_ctx_hash =
      av1_get_crc_value(&crc_calculator, txb_hash_data, hash_data_index);
  //// End hash creation

  return hbt_search_match(hbt_ctx_hash, hbt_qc_hash, txb_info, txb_costs,
                          txb_eob_costs, p, block, fast_mode, rate_cost);
}

int av1_optimize_txb(const struct AV1_COMP *cpi, MACROBLOCK *x, int plane,
                     int blk_row, int blk_col, int block, TX_SIZE tx_size,
                     TXB_CTX *txb_ctx, int fast_mode, int *rate_cost) {
  const AV1_COMMON *cm = &cpi->common;
  MACROBLOCKD *const xd = &x->e_mbd;
  const PLANE_TYPE plane_type = get_plane_type(plane);
  const TX_SIZE txs_ctx = get_txsize_entropy_ctx(tx_size);
  const TX_TYPE tx_type = av1_get_tx_type(plane_type, xd, blk_row, blk_col,
                                          tx_size, cm->reduced_tx_set_used);
  const MB_MODE_INFO *mbmi = &xd->mi[0]->mbmi;
  const struct macroblock_plane *p = &x->plane[plane];
  struct macroblockd_plane *pd = &xd->plane[plane];
  const int eob = p->eobs[block];
  tran_low_t *qcoeff = BLOCK_OFFSET(p->qcoeff, block);
  tran_low_t *dqcoeff = BLOCK_OFFSET(pd->dqcoeff, block);
  const tran_low_t *tcoeff = BLOCK_OFFSET(p->coeff, block);
  const int16_t *dequant = p->dequant_QTX;
  const int seg_eob = av1_get_max_eob(tx_size);
  const int bwl = get_txb_bwl(tx_size);
  const int width = get_txb_wide(tx_size);
  const int height = get_txb_high(tx_size);
  const int is_inter = is_inter_block(mbmi);
  const SCAN_ORDER *const scan_order = get_scan(tx_size, tx_type);
  const LV_MAP_COEFF_COST txb_costs = x->coeff_costs[txs_ctx][plane_type];
  const int eob_multi_size = txsize_log2_minus4[tx_size];
  const LV_MAP_EOB_COST txb_eob_costs =
      x->eob_costs[eob_multi_size][plane_type];

  const int shift = av1_get_tx_scale(tx_size);
  const int64_t rdmult =
      ((x->rdmult * plane_rd_mult[is_inter][plane_type] << (2 * (xd->bd - 8))) +
       2) >>
      2;
  uint8_t levels_buf[TX_PAD_2D];
  uint8_t *const levels = set_levels(levels_buf, width);
  const TX_SIZE qm_tx_size = av1_get_adjusted_tx_size(tx_size);
  const qm_val_t *iqmatrix =
      IS_2D_TRANSFORM(tx_type)
          ? pd->seg_iqmatrix[mbmi->segment_id][qm_tx_size]
          : cm->giqmatrix[NUM_QM_LEVELS - 1][0][qm_tx_size];
  assert(width == (1 << bwl));
  TxbInfo txb_info = {
    qcoeff,   levels,  dqcoeff,    tcoeff,  dequant, shift,
    tx_size,  txs_ctx, tx_type,    bwl,     width,   height,
    eob,      seg_eob, scan_order, txb_ctx, rdmult,  &cm->coeff_ctx_table,
    iqmatrix,
  };

  // Hash based trellis (hbt) speed feature: avoid expensive optimize_txb calls
  // by storing the coefficient deltas in a hash table.
  // Currently disabled in speedfeatures.c
  if (eob <= HBT_EOB && eob > 0 && cpi->sf.use_hash_based_trellis) {
    return hbt_create_hashes(&txb_info, &txb_costs, &txb_eob_costs, p, block,
                             fast_mode, rate_cost);
  }

  av1_txb_init_levels(qcoeff, width, height, levels);

  const int update = optimize_txb(&txb_info, &txb_costs, &txb_eob_costs, NULL,
                                  0, fast_mode, rate_cost);

  if (update) {
    p->eobs[block] = txb_info.eob;
    p->txb_entropy_ctx[block] =
        av1_get_txb_entropy_context(qcoeff, scan_order, txb_info.eob);
  }
  return txb_info.eob;
}

int av1_get_txb_entropy_context(const tran_low_t *qcoeff,
                                const SCAN_ORDER *scan_order, int eob) {
  const int16_t *const scan = scan_order->scan;
  int cul_level = 0;
  int c;

  if (eob == 0) return 0;
  for (c = 0; c < eob; ++c) {
    cul_level += abs(qcoeff[scan[c]]);
  }

  cul_level = AOMMIN(COEFF_CONTEXT_MASK, cul_level);
  set_dc_sign(&cul_level, qcoeff[0]);

  return cul_level;
}

void av1_update_txb_context_b(int plane, int block, int blk_row, int blk_col,
                              BLOCK_SIZE plane_bsize, TX_SIZE tx_size,
                              void *arg) {
  struct tokenize_b_args *const args = arg;
  const AV1_COMP *cpi = args->cpi;
  const AV1_COMMON *cm = &cpi->common;
  ThreadData *const td = args->td;
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  struct macroblock_plane *p = &x->plane[plane];
  struct macroblockd_plane *pd = &xd->plane[plane];
  const uint16_t eob = p->eobs[block];
  const tran_low_t *qcoeff = BLOCK_OFFSET(p->qcoeff, block);
  const PLANE_TYPE plane_type = pd->plane_type;
  const TX_TYPE tx_type = av1_get_tx_type(plane_type, xd, blk_row, blk_col,
                                          tx_size, cm->reduced_tx_set_used);
  const SCAN_ORDER *const scan_order = get_scan(tx_size, tx_type);
  (void)plane_bsize;

  int cul_level = av1_get_txb_entropy_context(qcoeff, scan_order, eob);
  av1_set_contexts(xd, pd, plane, tx_size, cul_level, blk_col, blk_row);
}

void av1_update_and_record_txb_context(int plane, int block, int blk_row,
                                       int blk_col, BLOCK_SIZE plane_bsize,
                                       TX_SIZE tx_size, void *arg) {
  struct tokenize_b_args *const args = arg;
  const AV1_COMP *cpi = args->cpi;
  const AV1_COMMON *cm = &cpi->common;
  ThreadData *const td = args->td;
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  struct macroblock_plane *p = &x->plane[plane];
  struct macroblockd_plane *pd = &xd->plane[plane];
  MB_MODE_INFO *mbmi = &xd->mi[0]->mbmi;
  int eob = p->eobs[block];
  const PLANE_TYPE plane_type = pd->plane_type;
  const tran_low_t *qcoeff = BLOCK_OFFSET(p->qcoeff, block);
  tran_low_t *tcoeff = BLOCK_OFFSET(x->mbmi_ext->tcoeff[plane], block);
  const int segment_id = mbmi->segment_id;
  const TX_TYPE tx_type = av1_get_tx_type(plane_type, xd, blk_row, blk_col,
                                          tx_size, cm->reduced_tx_set_used);
  const SCAN_ORDER *const scan_order = get_scan(tx_size, tx_type);
  const int16_t *const scan = scan_order->scan;
  const int seg_eob = av1_get_tx_eob(&cpi->common.seg, segment_id, tx_size);
  int c;
  TXB_CTX txb_ctx;
  get_txb_ctx(plane_bsize, tx_size, plane, pd->above_context + blk_col,
              pd->left_context + blk_row, &txb_ctx);
  const int bwl = get_txb_bwl(tx_size);
  const int width = get_txb_wide(tx_size);
  const int height = get_txb_high(tx_size);
  uint8_t levels_buf[TX_PAD_2D];
  uint8_t *const levels = set_levels(levels_buf, width);
  const uint8_t allow_update_cdf = args->allow_update_cdf;

  TX_SIZE txsize_ctx = get_txsize_entropy_ctx(tx_size);
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
  DECLARE_ALIGNED(16, int8_t, coeff_contexts[MAX_TX_SQUARE]);

  memcpy(tcoeff, qcoeff, sizeof(*tcoeff) * seg_eob);
#if CONFIG_ENTROPY_STATS
  ++td->counts->txb_skip[txsize_ctx][txb_ctx.txb_skip_ctx][eob == 0];
#endif  // CONFIG_ENTROPY_STATS
  if (allow_update_cdf) {
    update_cdf(ec_ctx->txb_skip_cdf[txsize_ctx][txb_ctx.txb_skip_ctx], eob == 0,
               2);
  }
  x->mbmi_ext->txb_skip_ctx[plane][block] = txb_ctx.txb_skip_ctx;

  x->mbmi_ext->eobs[plane][block] = eob;

  if (eob == 0) {
    av1_set_contexts(xd, pd, plane, tx_size, 0, blk_col, blk_row);
    return;
  }

  av1_txb_init_levels(tcoeff, width, height, levels);
  av1_update_tx_type_count(cm, xd, blk_row, blk_col, plane, mbmi->sb_type,
                           tx_size, td->counts, allow_update_cdf);
  av1_update_eob_context(eob, seg_eob, tx_size, tx_type, plane_type, ec_ctx,
                         td->counts, allow_update_cdf);
  av1_get_nz_map_contexts(levels, scan, eob, tx_size, tx_type, coeff_contexts);

  for (c = eob - 1; c >= 0; --c) {
    const int pos = scan[c];
    const int coeff_ctx = coeff_contexts[pos];
    const tran_low_t v = qcoeff[pos];
    const tran_low_t level = abs(v);
    const int is_nz = (v != 0);

    (void)is_nz;
    if (allow_update_cdf) {
      if (c == eob - 1) {
        assert(coeff_ctx < 4);
        update_cdf(
            ec_ctx->coeff_base_eob_cdf[txsize_ctx][plane_type][coeff_ctx],
            AOMMIN(level, 3) - 1, 3);
      } else {
        update_cdf(ec_ctx->coeff_base_cdf[txsize_ctx][plane_type][coeff_ctx],
                   AOMMIN(level, 3), 4);
      }
    }
    {
      if (c == eob - 1) {
        assert(coeff_ctx < 4);
        ++td->counts->coeff_base_eob_multi[txsize_ctx][plane_type][coeff_ctx]
                                          [AOMMIN(level, 3) - 1];
      } else {
        ++td->counts->coeff_base_multi[txsize_ctx][plane_type][coeff_ctx]
                                      [AOMMIN(level, 3)];
      }
    }
    if (level > NUM_BASE_LEVELS) {
      const int base_range = level - 1 - NUM_BASE_LEVELS;
      const int br_ctx = get_br_ctx(levels, pos, bwl, tx_type);
      for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
        const int k = AOMMIN(base_range - idx, BR_CDF_SIZE - 1);
        if (allow_update_cdf) {
          update_cdf(ec_ctx->coeff_br_cdf[AOMMIN(txsize_ctx, TX_32X32)]
                                         [plane_type][br_ctx],
                     k, BR_CDF_SIZE);
        }
        for (int lps = 0; lps < BR_CDF_SIZE - 1; lps++) {
#if CONFIG_ENTROPY_STATS
          ++td->counts->coeff_lps[AOMMIN(txsize_ctx, TX_32X32)][plane_type][lps]
                                 [br_ctx][lps == k];
#endif  // CONFIG_ENTROPY_STATS
          if (lps == k) break;
        }
        ++td->counts->coeff_lps_multi[AOMMIN(txsize_ctx, TX_32X32)][plane_type]
                                     [br_ctx][k];
        if (k < BR_CDF_SIZE - 1) break;
      }
    }
  }

  // Update the context needed to code the DC sign (if applicable)
  const int sign = (tcoeff[0] < 0) ? 1 : 0;
  if (tcoeff[0] != 0) {
    int dc_sign_ctx = txb_ctx.dc_sign_ctx;
#if CONFIG_ENTROPY_STATS
    ++td->counts->dc_sign[plane_type][dc_sign_ctx][sign];
#endif  // CONFIG_ENTROPY_STATS
    if (allow_update_cdf)
      update_cdf(ec_ctx->dc_sign_cdf[plane_type][dc_sign_ctx], sign, 2);
    x->mbmi_ext->dc_sign_ctx[plane][block] = dc_sign_ctx;
  }

  int cul_level = av1_get_txb_entropy_context(tcoeff, scan_order, eob);
  av1_set_contexts(xd, pd, plane, tx_size, cul_level, blk_col, blk_row);
}

void av1_update_txb_context(const AV1_COMP *cpi, ThreadData *td,
                            RUN_TYPE dry_run, BLOCK_SIZE bsize, int *rate,
                            int mi_row, int mi_col, uint8_t allow_update_cdf) {
  const AV1_COMMON *const cm = &cpi->common;
  const int num_planes = av1_num_planes(cm);
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  MB_MODE_INFO *const mbmi = &xd->mi[0]->mbmi;
  struct tokenize_b_args arg = { cpi, td, NULL, 0, allow_update_cdf };
  (void)rate;
  (void)mi_row;
  (void)mi_col;
  if (mbmi->skip) {
    av1_reset_skip_context(xd, mi_row, mi_col, bsize, num_planes);
    return;
  }

  if (!dry_run) {
    av1_foreach_transformed_block(xd, bsize, mi_row, mi_col,
                                  av1_update_and_record_txb_context, &arg,
                                  num_planes);
  } else if (dry_run == DRY_RUN_NORMAL) {
    av1_foreach_transformed_block(xd, bsize, mi_row, mi_col,
                                  av1_update_txb_context_b, &arg, num_planes);
  } else {
    printf("DRY_RUN_COSTCOEFFS is not supported yet\n");
    assert(0);
  }
}
