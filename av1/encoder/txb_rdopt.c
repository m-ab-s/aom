/*
 * Copyright (c) 2021, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include "av1/encoder/txb_rdopt.h"
#include "av1/encoder/txb_rdopt_utils.h"
#include "av1/encoder/rdopt_utils.h"

#include "aom_ports/mem.h"
#include "av1/common/idct.h"

static AOM_FORCE_INLINE void update_coeff_general(
    int *accu_rate, int64_t *accu_dist, int si, int eob, TX_SIZE tx_size,
    TX_CLASS tx_class, int bhl, int width, int64_t rdmult, int shift,
    int dc_sign_ctx, const int16_t *dequant, const int16_t *scan,
    const LV_MAP_COEFF_COST *txb_costs, const tran_low_t *tcoeff,
    tran_low_t *qcoeff, tran_low_t *dqcoeff, uint8_t *levels,
    const qm_val_t *iqmatrix, const qm_val_t *qmatrix) {
  const int ci = scan[si];
  const tran_low_t qc = qcoeff[ci];
  const int is_last = si == (eob - 1);
  const int coeff_ctx = get_lower_levels_ctx_general(
      is_last, si, bhl, width, levels, ci, tx_size, tx_class);
  if (qc == 0) {
    *accu_rate += txb_costs->base_cost[coeff_ctx][0];
  } else {
    const int sign = (qc < 0) ? 1 : 0;
    const tran_low_t abs_qc = abs(qc);
    const tran_low_t tqc = tcoeff[ci];
    const tran_low_t dqc = dqcoeff[ci];
    const int rate =
        get_coeff_cost_general(is_last, ci, abs_qc, sign, coeff_ctx,
                               dc_sign_ctx, txb_costs, bhl, tx_class, levels);

    tran_low_t qc_low = 0, dqc_low = 0;
    tran_low_t abs_qc_low = 0;
    int rate_low;
    if (abs_qc == 1) {
      rate_low = txb_costs->base_cost[coeff_ctx][0];
    } else {
      const int dqv = get_dqv(dequant, ci, iqmatrix);
      get_qc_dqc_low(abs_qc, sign, dqv, shift, &qc_low, &dqc_low);
      abs_qc_low = abs_qc - 1;
      rate_low =
          get_coeff_cost_general(is_last, ci, abs_qc_low, sign, coeff_ctx,
                                 dc_sign_ctx, txb_costs, bhl, tx_class, levels);
    }

    int64_t dist_diff_0, dist_diff_low_0;
    if (qmatrix == NULL) {
      const int64_t tqc2 = (int64_t)tqc * 2;
      dist_diff_0 = ((int64_t)dqc * (dqc - tqc2)) * (1LL << (2 * shift));
      dist_diff_low_0 =
          (abs_qc == 1)
              ? 0
              : (((int64_t)dqc_low * (dqc_low - tqc2)) * (1LL << (2 * shift)));
    } else {
      const int64_t qm_val = qmatrix[ci];
      const int64_t dist0 = get_coeff_dist_qm(tqc, 0, shift, qm_val);
      dist_diff_0 = get_coeff_dist_qm(tqc, dqc, shift, qm_val) - dist0;
      dist_diff_low_0 =
          (abs_qc == 1)
              ? 0
              : (get_coeff_dist_qm(tqc, dqc_low, shift, qm_val) - dist0);
    }

    const int64_t rd = RDCOST(rdmult, rate, dist_diff_0);
    const int64_t rd_low = RDCOST(rdmult, rate_low, dist_diff_low_0);

    if (rd_low < rd) {
      qcoeff[ci] = qc_low;
      dqcoeff[ci] = dqc_low;
      levels[get_padded_idx(ci, bhl)] = AOMMIN(abs_qc_low, INT8_MAX);
      *accu_rate += rate_low;
      *accu_dist += dist_diff_low_0;
    } else {
      *accu_rate += rate;
      *accu_dist += dist_diff_0;
    }
  }
}

static AOM_FORCE_INLINE int update_coeff_simple(
    int si, int eob, TX_SIZE tx_size, TX_CLASS tx_class, int bhl,
    int64_t rdmult, int shift, const int16_t *dequant, const int16_t *scan,
    const LV_MAP_COEFF_COST *txb_costs, const tran_low_t *tcoeff,
    tran_low_t *qcoeff, tran_low_t *dqcoeff, uint8_t *levels, int sharpness,
    const qm_val_t *iqmatrix, const qm_val_t *qmatrix) {
  (void)eob;
  // this simple version assumes the coeff's scan_idx is not DC (scan_idx != 0)
  // and not the last (scan_idx != eob - 1)
  assert(si != eob - 1);
  assert(si > 0);
  const int ci = scan[si];
  assert(ci > 0);
  const tran_low_t qc = qcoeff[ci];
  const int coeff_ctx =
      get_lower_levels_ctx_simple(levels, ci, bhl, tx_size, tx_class);
  if (qc == 0) {
    return txb_costs->base_cost[coeff_ctx][0];
  } else {
    const tran_low_t abs_qc = abs(qc);
    const tran_low_t abs_tqc = abs(tcoeff[ci]);
    const tran_low_t abs_dqc = abs(dqcoeff[ci]);
    if (abs_qc == 1) {
      const int *base_cost = txb_costs->base_cost[coeff_ctx];
      const int rate = base_cost[1] + av1_cost_literal(1);
      if (abs_dqc < abs_tqc) {
        return rate;
      }

      const int allow_lower_qc = sharpness ? 0 : 1;
      if (allow_lower_qc) {
        int64_t dist_diff_0;
        if (qmatrix == NULL) {
          dist_diff_0 =
              ((int64_t)abs_dqc * (abs_dqc - ((int64_t)abs_tqc * 2))) *
              (1LL << (2 * shift));
        } else {
          const int64_t qm_val = qmatrix[ci];
          const int64_t dist0 = get_coeff_dist_qm(abs_tqc, 0, shift, qm_val);
          dist_diff_0 =
              get_coeff_dist_qm(abs_tqc, abs_dqc, shift, qm_val) - dist0;
        }
        const int rate_low = rate - base_cost[5];
        const int64_t rd = RDCOST(rdmult, rate, dist_diff_0);
        const int64_t rd_low = RDCOST(rdmult, rate_low, 0);
        if (rd_low < rd) {
          qcoeff[ci] = 0;
          dqcoeff[ci] = 0;
          levels[get_padded_idx(ci, bhl)] = 0;
          return rate_low;
        }
      }
      return rate;
    } else {
      if (abs_dqc < abs_tqc) {
        int rate = txb_costs->base_cost[coeff_ctx][AOMMIN(abs_qc, 3)] +
                   av1_cost_literal(1);
        if (abs_qc > NUM_BASE_LEVELS) {
          const int br_ctx = get_br_ctx_simple(levels, ci, bhl, tx_class);
          rate += get_br_cost(abs_qc, txb_costs->lps_cost[br_ctx]);
        }
        return rate;
      }

      int rate_low = 0;
      const int rate = get_two_coeff_cost_simple(
          ci, abs_qc, coeff_ctx, txb_costs, bhl, tx_class, levels, &rate_low);
      // update_coeff_simple is only invoked for AC coefficients (si > 0), so
      // dequant[1] can be used directly without a (ci > 0) branch.
      int dqv = dequant[1];
      if (iqmatrix != NULL) {
        dqv = ((iqmatrix[ci] * dqv) + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS;
      }
      const tran_low_t abs_qc_low = abs_qc - 1;
      const tran_low_t abs_dqc_low = (abs_qc_low * dqv) >> shift;
      int64_t dist_diff_0, dist_diff_low_0;
      if (qmatrix == NULL) {
        const int64_t abs_tqc2 = (int64_t)abs_tqc << 1;
        dist_diff_0 =
            ((int64_t)abs_dqc * (abs_dqc - abs_tqc2)) * (1LL << (2 * shift));
        dist_diff_low_0 = ((int64_t)abs_dqc_low * (abs_dqc_low - abs_tqc2)) *
                          (1LL << (2 * shift));
      } else {
        const int64_t qm_val = qmatrix[ci];
        const int64_t dist0 = get_coeff_dist_qm(abs_tqc, 0, shift, qm_val);
        dist_diff_0 =
            get_coeff_dist_qm(abs_tqc, abs_dqc, shift, qm_val) - dist0;
        dist_diff_low_0 =
            get_coeff_dist_qm(abs_tqc, abs_dqc_low, shift, qm_val) - dist0;
      }
      const int64_t rd = RDCOST(rdmult, rate, dist_diff_0);
      const int64_t rd_low = RDCOST(rdmult, rate_low, dist_diff_low_0);
      if (rd_low < rd) {
        const int sign = (qc < 0) ? 1 : 0;
        qcoeff[ci] = (-sign ^ abs_qc_low) + sign;
        dqcoeff[ci] = (-sign ^ abs_dqc_low) + sign;
        levels[get_padded_idx(ci, bhl)] = AOMMIN(abs_qc_low, INT8_MAX);
        return rate_low;
      }
      return rate;
    }
  }
}

static inline void update_skip(int *accu_rate, int64_t accu_dist, int *eob,
                               int nz_num, int *nz_ci, int64_t rdmult,
                               int skip_cost, int non_skip_cost,
                               tran_low_t *qcoeff, tran_low_t *dqcoeff) {
  const int64_t rd = RDCOST(rdmult, *accu_rate + non_skip_cost, accu_dist);
  const int64_t rd_new_eob = RDCOST(rdmult, skip_cost, 0);
  if (rd_new_eob < rd) {
    for (int i = 0; i < nz_num; ++i) {
      const int ci = nz_ci[i];
      qcoeff[ci] = 0;
      dqcoeff[ci] = 0;
      // no need to set up levels because this is the last step
      // levels[get_padded_idx(ci, bhl)] = 0;
    }
    *accu_rate = 0;
    *eob = 0;
  }
}

// TODO(angiebird): use this function whenever it's possible
static int get_tx_type_cost(const MACROBLOCK *x, const MACROBLOCKD *xd,
                            int plane, TX_SIZE tx_size, TX_TYPE tx_type,
                            int reduced_tx_set_used) {
  if (plane > 0) return 0;

  const TX_SIZE square_tx_size = txsize_sqr_map[tx_size];

  const MB_MODE_INFO *mbmi = xd->mi[0];
  const int is_inter = is_inter_block(mbmi);
  const TxSetType set_type =
      av1_get_ext_tx_set_type(tx_size, is_inter, reduced_tx_set_used);
  if (av1_num_ext_tx_set[set_type] > 1 &&
      !xd->lossless[xd->mi[0]->segment_id]) {
    const int ext_tx_set = ext_tx_set_index[is_inter][set_type];
    if (is_inter) {
      if (ext_tx_set > 0)
        return x->mode_costs
            .inter_tx_type_costs[ext_tx_set][square_tx_size][tx_type];
    } else {
      if (ext_tx_set > 0) {
        PREDICTION_MODE intra_dir;
        if (mbmi->filter_intra_mode_info.use_filter_intra)
          intra_dir = fimode_to_intradir[mbmi->filter_intra_mode_info
                                             .filter_intra_mode];
        else
          intra_dir = mbmi->mode;
        return x->mode_costs.intra_tx_type_costs[ext_tx_set][square_tx_size]
                                                [intra_dir][tx_type];
      }
    }
  }
  return 0;
}

static AOM_FORCE_INLINE void update_coeff_eob_facade(
    int *accu_rate, int64_t *accu_dist, int *eob, int *nz_num, int *nz_ci,
    int *si, TX_SIZE tx_size, TX_CLASS tx_class, int bhl, int width,
    int dc_sign_ctx, int64_t rdmult, int shift, const int16_t *dequant,
    const int16_t *__restrict scan,
    const LV_MAP_EOB_COST *__restrict txb_eob_costs,
    const LV_MAP_COEFF_COST *__restrict txb_costs,
    const tran_low_t *__restrict tcoeff, tran_low_t *__restrict qcoeff,
    tran_low_t *__restrict dqcoeff, uint8_t *__restrict levels, int sharpness,
    const qm_val_t *iqmatrix, const qm_val_t *qmatrix, int max_nz_num,
    int is_noise_pattern) {
  int local_rate = *accu_rate;
  int64_t local_dist = *accu_dist;
  int local_eob = *eob;
  int local_nz_num = *nz_num;
  int local_si = *si;
  for (; local_si >= 0 && local_nz_num <= max_nz_num; --local_si) {
    assert(local_si != local_eob - 1);
    const int ci = scan[local_si];
    const tran_low_t qc = qcoeff[ci];
    const int coeff_ctx =
        get_lower_levels_ctx(levels, ci, bhl, tx_size, tx_class);
    if (qc == 0) {
      local_rate += txb_costs->base_cost[coeff_ctx][0];
    } else {
      int lower_level = 0;
      const tran_low_t abs_qc = abs(qc);
      const tran_low_t tqc = tcoeff[ci];
      const tran_low_t dqc = dqcoeff[ci];
      const int sign = (qc < 0) ? 1 : 0;

      tran_low_t qc_low = 0, dqc_low = 0;
      tran_low_t abs_qc_low = 0;
      int rate_low;

      if (abs_qc == 1) {
        rate_low = txb_costs->base_cost[coeff_ctx][0];
      } else {
        const int dqv = get_dqv(dequant, ci, iqmatrix);
        get_qc_dqc_low(abs_qc, sign, dqv, shift, &qc_low, &dqc_low);
        abs_qc_low = abs_qc - 1;
        rate_low = get_coeff_cost_general(0, ci, abs_qc_low, sign, coeff_ctx,
                                          dc_sign_ctx, txb_costs, bhl, tx_class,
                                          levels);
      }

      int64_t dist, dist_low;
      if (qmatrix == NULL) {
        const int64_t tqc2 = (int64_t)tqc * 2;
        dist = ((int64_t)dqc * (dqc - tqc2)) * (1LL << (2 * shift));
        dist_low = (abs_qc == 1) ? 0
                                 : (((int64_t)dqc_low * (dqc_low - tqc2)) *
                                    (1LL << (2 * shift)));
      } else {
        const int64_t qm_val = qmatrix[ci];
        const int64_t dist0 = get_coeff_dist_qm(tqc, 0, shift, qm_val);
        dist = get_coeff_dist_qm(tqc, dqc, shift, qm_val) - dist0;
        dist_low =
            (abs_qc == 1)
                ? 0
                : (get_coeff_dist_qm(tqc, dqc_low, shift, qm_val) - dist0);
      }

      int rate =
          get_coeff_cost_general(0, ci, abs_qc, sign, coeff_ctx, dc_sign_ctx,
                                 txb_costs, bhl, tx_class, levels);
      int64_t rd = RDCOST(rdmult, local_rate + rate, local_dist + dist);
      int64_t rd_low =
          RDCOST(rdmult, local_rate + rate_low, local_dist + dist_low);

      int lower_level_new_eob = 0;
      const int new_eob = local_si + 1;
      const int coeff_ctx_new_eob =
          get_lower_levels_ctx_eob(bhl, width, local_si);
      const int new_eob_cost =
          get_eob_cost(new_eob, txb_eob_costs, txb_costs, tx_class);
      int rate_coeff_eob =
          new_eob_cost + get_coeff_cost_eob(ci, abs_qc, sign, coeff_ctx_new_eob,
                                            dc_sign_ctx, txb_costs, bhl,
                                            tx_class);
      int64_t dist_new_eob = dist;
      int64_t rd_new_eob = RDCOST(rdmult, rate_coeff_eob, dist_new_eob);

      if (abs_qc_low > 0) {
        const int rate_coeff_eob_low =
            new_eob_cost + get_coeff_cost_eob(ci, abs_qc_low, sign,
                                              coeff_ctx_new_eob, dc_sign_ctx,
                                              txb_costs, bhl, tx_class);
        const int64_t dist_new_eob_low = dist_low;
        const int64_t rd_new_eob_low =
            RDCOST(rdmult, rate_coeff_eob_low, dist_new_eob_low);
        if (rd_new_eob_low < rd_new_eob) {
          lower_level_new_eob = 1;
          rd_new_eob = rd_new_eob_low;
          rate_coeff_eob = rate_coeff_eob_low;
          dist_new_eob = dist_new_eob_low;
        }
      }

      const int min_eob_cutoff = is_noise_pattern ? 8 : 5;
      const int qc_threshold = (local_si <= min_eob_cutoff) ? 2 : 1;
      const int allow_lower_qc = sharpness ? abs_qc > qc_threshold : 1;

      if (allow_lower_qc) {
        if (rd_low < rd) {
          lower_level = 1;
          rd = rd_low;
          rate = rate_low;
          dist = dist_low;
        }
      }

      if ((sharpness == 0 || new_eob >= min_eob_cutoff) && rd_new_eob < rd) {
        for (int ni = 0; ni < local_nz_num; ++ni) {
          int last_ci = nz_ci[ni];
          levels[get_padded_idx(last_ci, bhl)] = 0;
          qcoeff[last_ci] = 0;
          dqcoeff[last_ci] = 0;
        }
        local_eob = new_eob;
        local_nz_num = 0;
        local_rate = rate_coeff_eob;
        local_dist = dist_new_eob;
        lower_level = lower_level_new_eob;
      } else {
        local_rate += rate;
        local_dist += dist;
      }

      if (lower_level) {
        qcoeff[ci] = qc_low;
        dqcoeff[ci] = dqc_low;
        levels[get_padded_idx(ci, bhl)] = AOMMIN(abs_qc_low, INT8_MAX);
      }
      if (qcoeff[ci]) {
        nz_ci[local_nz_num] = ci;
        ++local_nz_num;
      }
    }
  }
  *accu_rate = local_rate;
  *accu_dist = local_dist;
  *eob = local_eob;
  *nz_num = local_nz_num;
  *si = local_si;
}

static AOM_FORCE_INLINE void update_coeff_simple_facade(
    int *accu_rate, int *si, int eob, TX_SIZE tx_size, TX_CLASS tx_class,
    int bhl, int64_t rdmult, int shift, const int16_t *dequant,
    const int16_t *__restrict scan,
    const LV_MAP_COEFF_COST *__restrict txb_costs,
    const tran_low_t *__restrict tcoeff, tran_low_t *__restrict qcoeff,
    tran_low_t *__restrict dqcoeff, uint8_t *__restrict levels, int sharpness,
    const qm_val_t *iqmatrix, const qm_val_t *qmatrix) {
  int local_rate = *accu_rate;
  int local_si = *si;
  for (; local_si >= 1; --local_si) {
    local_rate +=
        update_coeff_simple(local_si, eob, tx_size, tx_class, bhl, rdmult,
                            shift, dequant, scan, txb_costs, tcoeff, qcoeff,
                            dqcoeff, levels, sharpness, iqmatrix, qmatrix);
  }
  *accu_rate = local_rate;
  *si = local_si;
}

int av1_optimize_txb(const struct AV1_COMP *cpi, MACROBLOCK *x, int plane,
                     int block, TX_SIZE tx_size, TX_TYPE tx_type,
                     const TXB_CTX *const txb_ctx, int *rate_cost,
                     int sharpness) {
  MACROBLOCKD *xd = &x->e_mbd;
  const struct macroblock_plane *p = &x->plane[plane];
  const SCAN_ORDER *scan_order = get_scan(tx_size, tx_type);
  const int16_t *scan = scan_order->scan;
  const int shift = av1_get_tx_scale(tx_size);
  int eob = p->eobs[block];
  const int16_t *dequant = p->dequant_QTX;
  const qm_val_t *iqmatrix =
      tx_type < IDTX
          ? xd->plane[plane].seg_iqmatrix[xd->mi[0]->segment_id][tx_size]
          : NULL;
  const qm_val_t *qmatrix =
      (cpi->oxcf.tune_cfg.dist_metric == AOM_DIST_METRIC_QM_PSNR &&
       tx_type < IDTX)
          ? xd->plane[plane].seg_qmatrix[xd->mi[0]->segment_id][tx_size]
          : NULL;
  const int block_offset = BLOCK_OFFSET(block);
  tran_low_t *qcoeff = p->qcoeff + block_offset;
  tran_low_t *dqcoeff = p->dqcoeff + block_offset;
  const tran_low_t *tcoeff = p->coeff + block_offset;
  const CoeffCosts *coeff_costs = &x->coeff_costs;

  // This function is not called if eob = 0.
  assert(eob > 0);

  const AV1_COMMON *cm = &cpi->common;
  const PLANE_TYPE plane_type = get_plane_type(plane);
  const TX_SIZE txs_ctx = get_txsize_entropy_ctx(tx_size);
  const TX_CLASS tx_class = tx_type_to_class[tx_type];
  const MB_MODE_INFO *mbmi = xd->mi[0];
  const int bhl = get_txb_bhl(tx_size);
  const int width = get_txb_wide(tx_size);
  const int height = get_txb_high(tx_size);
  assert(height == (1 << bhl));
  const int is_inter = is_inter_block(mbmi);
  const LV_MAP_COEFF_COST *txb_costs =
      &coeff_costs->coeff_costs[txs_ctx][plane_type];
  const int eob_multi_size = txsize_log2_minus4[tx_size];
  const LV_MAP_EOB_COST *txb_eob_costs =
      &coeff_costs->eob_costs[eob_multi_size][plane_type];

  int is_noise_pattern = 0;
#if CONFIG_AV1_HIGHBITDEPTH
  if (xd->bd > 8 && sharpness == 3 && plane == AOM_PLANE_Y) {
    int64_t src_var, rec_var;
    av1_get_variance_stats(x, &src_var, &rec_var);
    if (src_var > rec_var) {
      const int num_pixels =
          block_size_wide[mbmi->bsize] * block_size_high[mbmi->bsize];
      const int64_t src_var_per_px = src_var / num_pixels;
      if (src_var_per_px < 64) {
        is_noise_pattern = 1;
      }
    }
  }
#endif

  // For the IQ and SSIMULACRA 2 tunings, increase rshift from 2 to 4.
  // This biases trellis quantization towards keeping more coefficients, and
  // together with the IQ and SSIMULACRA2 rdmult adjustment in
  // av1_compute_rd_mult_based_on_qindex(), this helps preserve image
  // features (like repeating patterns and camera noise/film grain), which
  // improves SSIMULACRA 2 scores.
  const int rshift = (cpi->oxcf.tune_cfg.tuning == AOM_TUNE_IQ ||
                      cpi->oxcf.tune_cfg.tuning == AOM_TUNE_SSIMULACRA2)
                         ? 7
                         : (is_noise_pattern ? 6 : 5);

  const int (*trellis_rd_mult)[2] = cpi->sf.tx_sf.use_chroma_trellis_rd_mult
                                        ? plane_rd_mult_chroma
                                        : plane_rd_mult;
  const int64_t rdmult = ROUND_POWER_OF_TWO(
      (int64_t)x->rdmult * (8 - sharpness) *
          (trellis_rd_mult[is_inter][plane_type] << (2 * (xd->bd - 8))),
      rshift);

  uint8_t levels_buf[TX_PAD_2D];
  uint8_t *const levels = set_levels(levels_buf, height);

  if (eob > 1) av1_txb_init_levels(qcoeff, width, height, levels);

  // TODO(angirbird): check iqmatrix

  const int non_skip_cost = txb_costs->txb_skip_cost[txb_ctx->txb_skip_ctx][0];
  const int skip_cost = txb_costs->txb_skip_cost[txb_ctx->txb_skip_ctx][1];
  const int eob_cost = get_eob_cost(eob, txb_eob_costs, txb_costs, tx_class);
  int accu_rate = eob_cost;
  int64_t accu_dist = 0;
  int si = eob - 1;
  const int ci = scan[si];
  const tran_low_t qc = qcoeff[ci];
  const tran_low_t abs_qc = abs(qc);
  const int sign = qc < 0;
  const int max_nz_num = 2;
  int nz_num = 1;
  int nz_ci[3] = { ci, 0, 0 };
  if (abs_qc >= 2) {
    update_coeff_general(&accu_rate, &accu_dist, si, eob, tx_size, tx_class,
                         bhl, width, rdmult, shift, txb_ctx->dc_sign_ctx,
                         dequant, scan, txb_costs, tcoeff, qcoeff, dqcoeff,
                         levels, iqmatrix, qmatrix);
    --si;
  } else {
    assert(abs_qc == 1);
    const int coeff_ctx = get_lower_levels_ctx_eob(bhl, width, si);
    accu_rate +=
        get_coeff_cost_eob(ci, abs_qc, sign, coeff_ctx, txb_ctx->dc_sign_ctx,
                           txb_costs, bhl, tx_class);
    const tran_low_t tqc = tcoeff[ci];
    const tran_low_t dqc = dqcoeff[ci];
    if (qmatrix == NULL) {
      accu_dist +=
          ((int64_t)dqc * (dqc - (int64_t)tqc * 2)) * (1LL << (2 * shift));
    } else {
      const int64_t dist = get_coeff_dist(tqc, dqc, shift, qmatrix, ci);
      const int64_t dist0 = get_coeff_dist(tqc, 0, shift, qmatrix, ci);
      accu_dist += dist - dist0;
    }
    --si;
  }

#define UPDATE_COEFF_EOB_CASE(tx_class_literal)                              \
  case tx_class_literal:                                                     \
    update_coeff_eob_facade(                                                 \
        &accu_rate, &accu_dist, &eob, &nz_num, nz_ci, &si, tx_size,          \
        tx_class_literal, bhl, width, txb_ctx->dc_sign_ctx, rdmult, shift,   \
        dequant, scan, txb_eob_costs, txb_costs, tcoeff, qcoeff, dqcoeff,    \
        levels, sharpness, iqmatrix, qmatrix, max_nz_num, is_noise_pattern); \
    break
  if (si >= 0) {
    switch (tx_class) {
      UPDATE_COEFF_EOB_CASE(TX_CLASS_2D);
      UPDATE_COEFF_EOB_CASE(TX_CLASS_HORIZ);
      UPDATE_COEFF_EOB_CASE(TX_CLASS_VERT);
#undef UPDATE_COEFF_EOB_CASE
      default: assert(false);
    }
  }

  if (si == -1 && nz_num <= max_nz_num && sharpness == 0) {
    update_skip(&accu_rate, accu_dist, &eob, nz_num, nz_ci, rdmult, skip_cost,
                non_skip_cost, qcoeff, dqcoeff);
  } else if (si >= 1) {
#define UPDATE_COEFF_SIMPLE_CASE(tx_class_literal)                            \
  case tx_class_literal:                                                      \
    update_coeff_simple_facade(&accu_rate, &si, eob, tx_size,                 \
                               tx_class_literal, bhl, rdmult, shift, dequant, \
                               scan, txb_costs, tcoeff, qcoeff, dqcoeff,      \
                               levels, sharpness, iqmatrix, qmatrix);         \
    break
    switch (tx_class) {
      UPDATE_COEFF_SIMPLE_CASE(TX_CLASS_2D);
      UPDATE_COEFF_SIMPLE_CASE(TX_CLASS_HORIZ);
      UPDATE_COEFF_SIMPLE_CASE(TX_CLASS_VERT);
#undef UPDATE_COEFF_SIMPLE_CASE
      default: assert(false);
    }
  }

  // DC position
  if (si == 0) {
    // no need to update accu_dist because it's not used after this point
    int64_t dummy_dist = 0;
    update_coeff_general(&accu_rate, &dummy_dist, si, eob, tx_size, tx_class,
                         bhl, width, rdmult, shift, txb_ctx->dc_sign_ctx,
                         dequant, scan, txb_costs, tcoeff, qcoeff, dqcoeff,
                         levels, iqmatrix, qmatrix);
  }

  p->eobs[block] = eob;
  if (eob == 0) {
    accu_rate += skip_cost;
    p->txb_entropy_ctx[block] = 0;
  } else {
    const int tx_type_cost = get_tx_type_cost(x, xd, plane, tx_size, tx_type,
                                              cm->features.reduced_tx_set_used);
    accu_rate += non_skip_cost + tx_type_cost;
    p->txb_entropy_ctx[block] =
        av1_get_txb_entropy_context(qcoeff, scan_order, eob);
  }

  *rate_cost = accu_rate;
  return eob;
}

static AOM_FORCE_INLINE int warehouse_efficients_txb(
    const MACROBLOCK *x, const int plane, const int block,
    const TX_SIZE tx_size, const TXB_CTX *const txb_ctx,
    const struct macroblock_plane *p, const int eob,
    const PLANE_TYPE plane_type, const LV_MAP_COEFF_COST *const coeff_costs,
    const MACROBLOCKD *const xd, const TX_TYPE tx_type, const TX_CLASS tx_class,
    int reduced_tx_set_used) {
  const tran_low_t *const __restrict qcoeff = p->qcoeff + BLOCK_OFFSET(block);
  const int txb_skip_ctx = txb_ctx->txb_skip_ctx;
  const int eob_multi_size = txsize_log2_minus4[tx_size];
  const LV_MAP_EOB_COST *const eob_costs =
      &x->coeff_costs.eob_costs[eob_multi_size][plane_type];
  int cost = coeff_costs->txb_skip_cost[txb_skip_ctx][0];
  cost += get_tx_type_cost(x, xd, plane, tx_size, tx_type, reduced_tx_set_used);
  cost += get_eob_cost(eob, eob_costs, coeff_costs, tx_class);

  const int (*lps_cost)[COEFF_BASE_RANGE + 1 + COEFF_BASE_RANGE + 1] =
      coeff_costs->lps_cost;

  if (eob == 1) {
    const tran_low_t v = qcoeff[0];
    if (v) {
      const int sign = AOMSIGN(v);
      const int level = (v ^ sign) - sign;
      cost += coeff_costs->base_eob_cost[0][AOMMIN(level, 3) - 1];
      if (level > NUM_BASE_LEVELS) {
        cost += get_br_cost(level, lps_cost[0]);
      }
      const int sign01 = -sign;
      cost += coeff_costs->dc_sign_cost[txb_ctx->dc_sign_ctx][sign01];
    }
    return cost;
  }

  const int bhl = get_txb_bhl(tx_size);
  const int width = get_txb_wide(tx_size);
  const SCAN_ORDER *const scan_order = get_scan(tx_size, tx_type);
  const int16_t *const __restrict scan = scan_order->scan;
  const int (*base_cost)[8] = coeff_costs->base_cost;

  if (eob == 2 && tx_class == TX_CLASS_2D) {
    const int pos1 = scan[1];
    const tran_low_t v1 = qcoeff[pos1];
    int level1 = 0;
    if (v1) {
      const int sign1 = AOMSIGN(v1);
      level1 = (v1 ^ sign1) - sign1;
      const int coeff_ctx1 = get_lower_levels_ctx_eob(bhl, width, 1);
      cost += coeff_costs->base_eob_cost[coeff_ctx1][AOMMIN(level1, 3) - 1];
      if (level1 > NUM_BASE_LEVELS) {
        const int ctx1 = get_br_ctx_eob(pos1, bhl, TX_CLASS_2D);
        cost += get_br_cost(level1, lps_cost[ctx1]);
      }
      cost += av1_cost_literal(1);
    }
    const tran_low_t v0 = qcoeff[0];
    if (!v0) {
      cost += base_cost[0][0];
    } else {
      const int sign0 = AOMSIGN(v0);
      const int level0 = (v0 ^ sign0) - sign0;
      cost += base_cost[0][AOMMIN(level0, 3)];
      const int sign01 = (sign0 ^ sign0) - sign0;
      const int dc_sign_ctx = txb_ctx->dc_sign_ctx;
      cost += coeff_costs->dc_sign_cost[dc_sign_ctx][sign01];
      if (level0 > NUM_BASE_LEVELS) {
        const int mag =
            (pos1 == 1 || pos1 == (1 << bhl) || pos1 == (1 << bhl) + 1)
                ? AOMMIN(level1, INT8_MAX)
                : 0;
        const int ctx0 = AOMMIN((mag + 1) >> 1, 6);
        cost += get_br_cost(level0, lps_cost[ctx0]);
      }
    }
    return cost;
  }

  const int height = get_txb_high(tx_size);
  uint8_t levels_buf[TX_PAD_2D];
  uint8_t *const __restrict levels = set_levels(levels_buf, height);

  av1_txb_init_levels(qcoeff, width, height, levels);

  int c = eob - 1;
  {
    const int pos = scan[c];
    const tran_low_t v = qcoeff[pos];

    if (v) {
      const int sign = AOMSIGN(v);
      const int level = (v ^ sign) - sign;
      const int coeff_ctx = get_lower_levels_ctx_eob(bhl, width, c);
      cost += coeff_costs->base_eob_cost[coeff_ctx][AOMMIN(level, 3) - 1];
      if (level > NUM_BASE_LEVELS) {
        const int ctx = get_br_ctx_eob(pos, bhl, tx_class);
        cost += get_br_cost(level, lps_cost[ctx]);
      }
      cost += av1_cost_literal(1);
    }
  }

  if (eob <= ((width << bhl) >> 4) + 3) {
    for (c = eob - 2; c >= 1; --c) {
      const int pos = scan[c];
      const int coeff_ctx =
          get_lower_levels_ctx(levels, pos, bhl, tx_size, tx_class);
      const int level = abs(qcoeff[pos]);
      if (level <= NUM_BASE_LEVELS) {
        cost +=
            base_cost[coeff_ctx][level] + ((level > 0) << AV1_PROB_COST_SHIFT);
      } else {
        cost += base_cost[coeff_ctx][3] + av1_cost_literal(1);
        const int ctx = get_br_ctx(levels, pos, bhl, tx_class);
        cost += get_br_cost(level, lps_cost[ctx]);
      }
    }
    const tran_low_t v = qcoeff[0];
    const int coeff_ctx =
        (tx_class == TX_CLASS_2D)
            ? 0
            : get_lower_levels_ctx(levels, 0, bhl, tx_size, tx_class);
    if (!v) {
      cost += base_cost[coeff_ctx][0];
    } else {
      const int sign = AOMSIGN(v);
      const int level = (v ^ sign) - sign;
      cost += base_cost[coeff_ctx][AOMMIN(level, 3)];
      const int sign01 = (sign ^ sign) - sign;
      const int dc_sign_ctx = txb_ctx->dc_sign_ctx;
      cost += coeff_costs->dc_sign_cost[dc_sign_ctx][sign01];
      if (level > NUM_BASE_LEVELS) {
        const int ctx = get_br_ctx(levels, 0, bhl, tx_class);
        cost += get_br_cost(level, lps_cost[ctx]);
      }
    }
    return cost;
  }

  DECLARE_ALIGNED(16, int8_t, coeff_contexts[MAX_TX_SQUARE]);
  av1_get_nz_map_contexts(levels, scan, eob, tx_size, tx_class, coeff_contexts);
  for (c = eob - 2; c >= 1; --c) {
    const int pos = scan[c];
    const int coeff_ctx = coeff_contexts[pos];
    const int level = abs(qcoeff[pos]);
    if (level <= NUM_BASE_LEVELS) {
      cost +=
          base_cost[coeff_ctx][level] + ((level > 0) << AV1_PROB_COST_SHIFT);
    } else {
      cost += base_cost[coeff_ctx][3] + av1_cost_literal(1);
      const int ctx = get_br_ctx(levels, pos, bhl, tx_class);
      cost += get_br_cost(level, lps_cost[ctx]);
    }
  }
  // c == 0 after previous loop
  {
    const int pos = scan[c];
    const tran_low_t v = qcoeff[pos];
    const int coeff_ctx = coeff_contexts[pos];
    if (!v) {
      cost += base_cost[coeff_ctx][0];
    } else {
      const int sign = AOMSIGN(v);
      const int level = (v ^ sign) - sign;
      cost += base_cost[coeff_ctx][AOMMIN(level, 3)];
      // sign bit cost
      const int sign01 = (sign ^ sign) - sign;
      const int dc_sign_ctx = txb_ctx->dc_sign_ctx;
      cost += coeff_costs->dc_sign_cost[dc_sign_ctx][sign01];
      if (level > NUM_BASE_LEVELS) {
        const int ctx = get_br_ctx(levels, pos, bhl, tx_class);
        cost += get_br_cost(level, lps_cost[ctx]);
      }
    }
  }
  return cost;
}

/*!\brief Estimate the entropy cost of transform coefficients using Laplacian
 * distribution.
 *
 * \ingroup coefficient_coding
 *
 * This function assumes each transform coefficient is of its own Laplacian
 * distribution and the coefficient is the only observation of the Laplacian
 * distribution.
 *
 * Based on that, each coefficient's coding cost can be estimated by computing
 * the entropy of the corresponding Laplacian distribution.
 *
 * This function then return the sum of the estimated entropy cost for all
 * coefficients in the transform block.
 *
 * Note that the entropy cost of end of block (eob) and transform type (tx_type)
 * are not included.
 *
 * \param[in]    x              Pointer to structure holding the data for the
                                current encoding macroblock
 * \param[in]    plane          The index of the current plane
 * \param[in]    block          The index of the current transform block in the
 * macroblock. It's defined by number of 4x4 units that have been coded before
 * the currernt transform block
 * \param[in]    tx_size        The transform size
 * \param[in]    tx_type        The transform type
 * \return       int            Estimated entropy cost of coefficients in the
 * transform block.
 */
static int av1_cost_coeffs_txb_estimate(const MACROBLOCK *x, const int plane,
                                        const int block, const TX_SIZE tx_size,
                                        const TX_TYPE tx_type) {
  assert(plane == 0);

  int cost = 0;
  const struct macroblock_plane *p = &x->plane[plane];
  const SCAN_ORDER *scan_order = get_scan(tx_size, tx_type);
  const int16_t *scan = scan_order->scan;
  tran_low_t *qcoeff = p->qcoeff + BLOCK_OFFSET(block);

  int eob = p->eobs[block];

  // coeffs
  int c = eob - 1;
  // eob
  {
    const int pos = scan[c];
    const tran_low_t v = abs(qcoeff[pos]) - 1;
    cost += (v << (AV1_PROB_COST_SHIFT + 2));
  }
  // other coeffs
  for (c = eob - 2; c >= 0; c--) {
    const int pos = scan[c];
    const tran_low_t v = abs(qcoeff[pos]);
    const int idx = AOMMIN(v, 14);

    cost += costLUT[idx];
  }

  // const_term does not contain DC, and log(e) does not contain eob, so both
  // (eob-1)
  cost += (const_term + loge_par) * (eob - 1);

  return cost;
}

static AOM_FORCE_INLINE int warehouse_efficients_txb_laplacian(
    const MACROBLOCK *x, const int plane, const int block,
    const TX_SIZE tx_size, const TXB_CTX *const txb_ctx, const int eob,
    const PLANE_TYPE plane_type, const LV_MAP_COEFF_COST *const coeff_costs,
    const MACROBLOCKD *const xd, const TX_TYPE tx_type, const TX_CLASS tx_class,
    int reduced_tx_set_used) {
  const int txb_skip_ctx = txb_ctx->txb_skip_ctx;

  const int eob_multi_size = txsize_log2_minus4[tx_size];
  const LV_MAP_EOB_COST *const eob_costs =
      &x->coeff_costs.eob_costs[eob_multi_size][plane_type];
  int cost = coeff_costs->txb_skip_cost[txb_skip_ctx][0];

  cost += get_tx_type_cost(x, xd, plane, tx_size, tx_type, reduced_tx_set_used);

  cost += get_eob_cost(eob, eob_costs, coeff_costs, tx_class);

  cost += av1_cost_coeffs_txb_estimate(x, plane, block, tx_size, tx_type);
  return cost;
}

int av1_cost_coeffs_txb(const MACROBLOCK *x, const int plane, const int block,
                        const TX_SIZE tx_size, const TX_TYPE tx_type,
                        const TXB_CTX *const txb_ctx, int reduced_tx_set_used) {
  const struct macroblock_plane *p = &x->plane[plane];
  const int eob = p->eobs[block];
  const TX_SIZE txs_ctx = get_txsize_entropy_ctx(tx_size);
  const PLANE_TYPE plane_type = get_plane_type(plane);
  const LV_MAP_COEFF_COST *const coeff_costs =
      &x->coeff_costs.coeff_costs[txs_ctx][plane_type];
  if (eob == 0) {
    return coeff_costs->txb_skip_cost[txb_ctx->txb_skip_ctx][1];
  }

  const MACROBLOCKD *const xd = &x->e_mbd;
  const TX_CLASS tx_class = tx_type_to_class[tx_type];

  switch (tx_class) {
    case TX_CLASS_2D:
      return warehouse_efficients_txb(x, plane, block, tx_size, txb_ctx, p, eob,
                                      plane_type, coeff_costs, xd, tx_type,
                                      TX_CLASS_2D, reduced_tx_set_used);

    case TX_CLASS_VERT:
      return warehouse_efficients_txb(x, plane, block, tx_size, txb_ctx, p, eob,
                                      plane_type, coeff_costs, xd, tx_type,
                                      TX_CLASS_VERT, reduced_tx_set_used);

    case TX_CLASS_HORIZ:
      return warehouse_efficients_txb(x, plane, block, tx_size, txb_ctx, p, eob,
                                      plane_type, coeff_costs, xd, tx_type,
                                      TX_CLASS_HORIZ, reduced_tx_set_used);

    default: assert(0 && "Invalid TX_CLASS"); return 0;
  }
}

int av1_cost_coeffs_txb_laplacian(const MACROBLOCK *x, const int plane,
                                  const int block, const TX_SIZE tx_size,
                                  const TX_TYPE tx_type,
                                  const TXB_CTX *const txb_ctx,
                                  const int reduced_tx_set_used,
                                  const int adjust_eob) {
  const struct macroblock_plane *p = &x->plane[plane];
  int eob = p->eobs[block];

  if (adjust_eob) {
    const SCAN_ORDER *scan_order = get_scan(tx_size, tx_type);
    const int16_t *scan = scan_order->scan;
    tran_low_t *tcoeff = p->coeff + BLOCK_OFFSET(block);
    tran_low_t *qcoeff = p->qcoeff + BLOCK_OFFSET(block);
    tran_low_t *dqcoeff = p->dqcoeff + BLOCK_OFFSET(block);
    update_coeff_eob_fast(&eob, av1_get_tx_scale(tx_size), p->dequant_QTX, scan,
                          tcoeff, qcoeff, dqcoeff);
    p->eobs[block] = eob;
  }

  const TX_SIZE txs_ctx = get_txsize_entropy_ctx(tx_size);
  const PLANE_TYPE plane_type = get_plane_type(plane);
  const LV_MAP_COEFF_COST *const coeff_costs =
      &x->coeff_costs.coeff_costs[txs_ctx][plane_type];
  if (eob == 0) {
    return coeff_costs->txb_skip_cost[txb_ctx->txb_skip_ctx][1];
  }

  const MACROBLOCKD *const xd = &x->e_mbd;
  const TX_CLASS tx_class = tx_type_to_class[tx_type];

  return warehouse_efficients_txb_laplacian(
      x, plane, block, tx_size, txb_ctx, eob, plane_type, coeff_costs, xd,
      tx_type, tx_class, reduced_tx_set_used);
}
