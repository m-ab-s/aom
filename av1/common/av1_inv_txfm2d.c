/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include "config/aom_dsp_rtcd.h"
#include "config/av1_rtcd.h"

#include "av1/common/enums.h"
#include "av1/common/av1_txfm.h"
#include "av1/common/av1_inv_txfm1d.h"
#include "av1/common/av1_inv_txfm1d_cfg.h"
#include "av1/common/resize.h"

void av1_highbd_iwht4x4_16_add_c(const tran_low_t *input, uint8_t *dest8,
                                 int stride, int bd) {
  /* 4-point reversible, orthonormal inverse Walsh-Hadamard in 3.5 adds,
     0.5 shifts per pixel. */
  int i;
  tran_low_t output[16];
  tran_low_t a1, b1, c1, d1, e1;
  const tran_low_t *ip = input;
  tran_low_t *op = output;
  uint16_t *dest = CONVERT_TO_SHORTPTR(dest8);

  for (i = 0; i < 4; i++) {
    a1 = ip[0] >> UNIT_QUANT_SHIFT;
    c1 = ip[1] >> UNIT_QUANT_SHIFT;
    d1 = ip[2] >> UNIT_QUANT_SHIFT;
    b1 = ip[3] >> UNIT_QUANT_SHIFT;
    a1 += c1;
    d1 -= b1;
    e1 = (a1 - d1) >> 1;
    b1 = e1 - b1;
    c1 = e1 - c1;
    a1 -= b1;
    d1 += c1;

    op[0] = a1;
    op[1] = b1;
    op[2] = c1;
    op[3] = d1;
    ip += 4;
    op += 4;
  }

  ip = output;
  for (i = 0; i < 4; i++) {
    a1 = ip[4 * 0];
    c1 = ip[4 * 1];
    d1 = ip[4 * 2];
    b1 = ip[4 * 3];
    a1 += c1;
    d1 -= b1;
    e1 = (a1 - d1) >> 1;
    b1 = e1 - b1;
    c1 = e1 - c1;
    a1 -= b1;
    d1 += c1;

    range_check_value(a1, bd + 1);
    range_check_value(b1, bd + 1);
    range_check_value(c1, bd + 1);
    range_check_value(d1, bd + 1);

    dest[stride * 0] = highbd_clip_pixel_add(dest[stride * 0], a1, bd);
    dest[stride * 1] = highbd_clip_pixel_add(dest[stride * 1], b1, bd);
    dest[stride * 2] = highbd_clip_pixel_add(dest[stride * 2], c1, bd);
    dest[stride * 3] = highbd_clip_pixel_add(dest[stride * 3], d1, bd);

    ip++;
    dest++;
  }
}

void av1_highbd_iwht4x4_1_add_c(const tran_low_t *in, uint8_t *dest8,
                                int dest_stride, int bd) {
  int i;
  tran_low_t a1, e1;
  tran_low_t tmp[4];
  const tran_low_t *ip = in;
  tran_low_t *op = tmp;
  uint16_t *dest = CONVERT_TO_SHORTPTR(dest8);
  (void)bd;

  a1 = ip[0] >> UNIT_QUANT_SHIFT;
  e1 = a1 >> 1;
  a1 -= e1;
  op[0] = a1;
  op[1] = op[2] = op[3] = e1;

  ip = tmp;
  for (i = 0; i < 4; i++) {
    e1 = ip[0] >> 1;
    a1 = ip[0] - e1;
    dest[dest_stride * 0] =
        highbd_clip_pixel_add(dest[dest_stride * 0], a1, bd);
    dest[dest_stride * 1] =
        highbd_clip_pixel_add(dest[dest_stride * 1], e1, bd);
    dest[dest_stride * 2] =
        highbd_clip_pixel_add(dest[dest_stride * 2], e1, bd);
    dest[dest_stride * 3] =
        highbd_clip_pixel_add(dest[dest_stride * 3], e1, bd);
    ip++;
    dest++;
  }
}

static INLINE TxfmFunc inv_txfm_type_to_func(int mode, TXFM_TYPE txfm_type) {
  (void)mode;
  switch (txfm_type) {
    case TXFM_TYPE_DCT4: return av1_idct4_new;
    case TXFM_TYPE_DCT8: return av1_idct8_new;
    case TXFM_TYPE_DCT16: return av1_idct16_new;
    case TXFM_TYPE_DCT32: return av1_idct32_new;
    case TXFM_TYPE_DCT64: return av1_idct64_new;
#if CONFIG_LGT
    case TXFM_TYPE_ADST4:
      return (mode >= INTER_MODE_START && mode < INTER_MODE_END)
                 ? av1_iadst4_lgt_inter
                 : av1_iadst4_lgt_intra;
    case TXFM_TYPE_ADST8:
      return (mode >= INTER_MODE_START && mode < INTER_MODE_END)
                 ? av1_iadst8_lgt_inter
                 : av1_iadst8_lgt_intra;
    case TXFM_TYPE_ADST16:
      return (mode >= INTER_MODE_START && mode < INTER_MODE_END)
                 ? av1_iadst16_lgt_inter
                 : av1_iadst16_lgt_intra;
#else
    case TXFM_TYPE_ADST4: return av1_iadst4_new;
    case TXFM_TYPE_ADST8: return av1_iadst8_new;
    case TXFM_TYPE_ADST16: return av1_iadst16_new;
#endif  // CONFIG_LGT
#if CONFIG_MODE_DEP_INTRA_TX || CONFIG_MODE_DEP_INTER_TX
    case TXFM_TYPE_MDTX4: return av1_imdt4;
    case TXFM_TYPE_MDTX8: return av1_imdt8;
    case TXFM_TYPE_MDTX16: return av1_imdt16;
#endif
#if CONFIG_DST7_32x32
    case TXFM_TYPE_ADST32: return av1_iadst32_new;
#endif
    case TXFM_TYPE_IDENTITY4: return av1_iidentity4_c;
    case TXFM_TYPE_IDENTITY8: return av1_iidentity8_c;
    case TXFM_TYPE_IDENTITY16: return av1_iidentity16_c;
    case TXFM_TYPE_IDENTITY32: return av1_iidentity32_c;
    default: assert(0); return NULL;
  }
}

static const int8_t inv_shift_4x4[2] = { 0, -4 };
static const int8_t inv_shift_8x8[2] = { -1, -4 };
static const int8_t inv_shift_16x16[2] = { -2, -4 };
static const int8_t inv_shift_32x32[2] = { -2, -4 };
static const int8_t inv_shift_64x64[2] = { -2, -4 };
static const int8_t inv_shift_4x8[2] = { 0, -4 };
static const int8_t inv_shift_8x4[2] = { 0, -4 };
static const int8_t inv_shift_8x16[2] = { -1, -4 };
static const int8_t inv_shift_16x8[2] = { -1, -4 };
static const int8_t inv_shift_16x32[2] = { -1, -4 };
static const int8_t inv_shift_32x16[2] = { -1, -4 };
static const int8_t inv_shift_32x64[2] = { -1, -4 };
static const int8_t inv_shift_64x32[2] = { -1, -4 };
static const int8_t inv_shift_4x16[2] = { -1, -4 };
static const int8_t inv_shift_16x4[2] = { -1, -4 };
static const int8_t inv_shift_8x32[2] = { -2, -4 };
static const int8_t inv_shift_32x8[2] = { -2, -4 };
static const int8_t inv_shift_16x64[2] = { -2, -4 };
static const int8_t inv_shift_64x16[2] = { -2, -4 };
#if CONFIG_FLEX_PARTITION
static const int8_t inv_shift_4x32[2] = { -1, -4 };
static const int8_t inv_shift_32x4[2] = { -1, -4 };
static const int8_t inv_shift_8x64[2] = { -1, -4 };
static const int8_t inv_shift_64x8[2] = { -1, -4 };
static const int8_t inv_shift_4x64[2] = { -2, -4 };
static const int8_t inv_shift_64x4[2] = { -2, -4 };
#endif  // CONFIG_FLEX_PARTITION

const int8_t *av1_inv_txfm_shift_ls[TX_SIZES_ALL] = {
  inv_shift_4x4,   inv_shift_8x8,   inv_shift_16x16, inv_shift_32x32,
  inv_shift_64x64, inv_shift_4x8,   inv_shift_8x4,   inv_shift_8x16,
  inv_shift_16x8,  inv_shift_16x32, inv_shift_32x16, inv_shift_32x64,
  inv_shift_64x32, inv_shift_4x16,  inv_shift_16x4,  inv_shift_8x32,
  inv_shift_32x8,  inv_shift_16x64, inv_shift_64x16,
#if CONFIG_FLEX_PARTITION
  inv_shift_4x32,  inv_shift_32x4,  inv_shift_8x64,  inv_shift_64x8,
  inv_shift_4x64,  inv_shift_64x4,
#endif  // CONFIG_FLEX_PARTITION
};

/* clang-format off */
const int8_t av1_inv_cos_bit_col[MAX_TXWH_IDX]      // txw_idx
                            [MAX_TXWH_IDX] = {  // txh_idx
    { INV_COS_BIT, INV_COS_BIT, INV_COS_BIT, INV_COS_BIT, INV_COS_BIT },
    { INV_COS_BIT, INV_COS_BIT, INV_COS_BIT, INV_COS_BIT, INV_COS_BIT },
    { INV_COS_BIT, INV_COS_BIT, INV_COS_BIT, INV_COS_BIT, INV_COS_BIT },
    { INV_COS_BIT, INV_COS_BIT, INV_COS_BIT, INV_COS_BIT, INV_COS_BIT },
    { INV_COS_BIT, INV_COS_BIT, INV_COS_BIT, INV_COS_BIT, INV_COS_BIT }
  };

const int8_t av1_inv_cos_bit_row[MAX_TXWH_IDX]      // txw_idx
                            [MAX_TXWH_IDX] = {  // txh_idx
    { INV_COS_BIT, INV_COS_BIT, INV_COS_BIT, INV_COS_BIT, INV_COS_BIT },
    { INV_COS_BIT, INV_COS_BIT, INV_COS_BIT, INV_COS_BIT, INV_COS_BIT },
    { INV_COS_BIT, INV_COS_BIT, INV_COS_BIT, INV_COS_BIT, INV_COS_BIT },
    { INV_COS_BIT, INV_COS_BIT, INV_COS_BIT, INV_COS_BIT, INV_COS_BIT },
    { INV_COS_BIT, INV_COS_BIT, INV_COS_BIT, INV_COS_BIT, INV_COS_BIT }
  };
/* clang-format on */

static const int8_t iadst4_range[7] = { 0, 1, 0, 0, 0, 0, 0 };

void av1_get_inv_txfm_cfg(TX_TYPE tx_type, TX_SIZE tx_size,
                          PREDICTION_MODE mode, TXFM_2D_FLIP_CFG *cfg) {
  assert(cfg != NULL);
  cfg->tx_size = tx_size;
  av1_zero(cfg->stage_range_col);
  av1_zero(cfg->stage_range_row);
  set_flip_cfg(tx_type, cfg);
  const TX_TYPE_1D tx_type_1d_col = vtx_tab[tx_type];
  const TX_TYPE_1D tx_type_1d_row = htx_tab[tx_type];
  cfg->shift = av1_inv_txfm_shift_ls[tx_size];
  const int txw_idx = get_txw_idx(tx_size);
  const int txh_idx = get_txh_idx(tx_size);
  cfg->cos_bit_col = av1_inv_cos_bit_col[txw_idx][txh_idx];
  cfg->cos_bit_row = av1_inv_cos_bit_row[txw_idx][txh_idx];
  cfg->txfm_type_col = av1_txfm_type_ls[txh_idx][tx_type_1d_col];
  if (cfg->txfm_type_col == TXFM_TYPE_ADST4) {
    memcpy(cfg->stage_range_col, iadst4_range, sizeof(iadst4_range));
  }
  cfg->txfm_type_row = av1_txfm_type_ls[txw_idx][tx_type_1d_row];
  if (cfg->txfm_type_row == TXFM_TYPE_ADST4) {
    memcpy(cfg->stage_range_row, iadst4_range, sizeof(iadst4_range));
  }
  cfg->mode = mode;
#if CONFIG_MODE_DEP_INTRA_TX && CONFIG_MODE_DEP_NONSEP_INTRA_TX
  if (use_nstx(tx_type, tx_size, mode)) {
    cfg->nstx_mtx_ptr = nstx_arr(tx_size, mode);
  } else if (use_nsst(tx_type, tx_size, mode)) {
    // For secondary transforms, use DCT_DCT as primary transform
    cfg->nstx_mtx_ptr = nstx_arr(tx_size, mode);
    cfg->txfm_type_col = av1_txfm_type_ls[txh_idx][DCT_1D];
    cfg->txfm_type_row = av1_txfm_type_ls[txw_idx][DCT_1D];
  } else {
    cfg->nstx_mtx_ptr = NULL;
  }
#endif  // CONFIG_MODE_DEP_INTRA_TX &&
        // CONFIG_MODE_DEP_NONSEP_INTRA_TX
  cfg->stage_num_col = av1_txfm_stage_num_list[cfg->txfm_type_col];
  cfg->stage_num_row = av1_txfm_stage_num_list[cfg->txfm_type_row];
}

void av1_gen_inv_stage_range(int8_t *stage_range_col, int8_t *stage_range_row,
                             const TXFM_2D_FLIP_CFG *cfg, TX_SIZE tx_size,
                             int bd) {
  const int fwd_shift = inv_start_range[tx_size];
  const int8_t *shift = cfg->shift;
  int8_t opt_range_row, opt_range_col;
  if (bd == 8) {
    opt_range_row = 16;
    opt_range_col = 16;
  } else if (bd == 10) {
    opt_range_row = 18;
    opt_range_col = 16;
  } else {
    assert(bd == 12);
    opt_range_row = 20;
    opt_range_col = 18;
  }
  // i < MAX_TXFM_STAGE_NUM will mute above array bounds warning
  for (int i = 0; i < cfg->stage_num_row && i < MAX_TXFM_STAGE_NUM; ++i) {
    int real_range_row = cfg->stage_range_row[i] + fwd_shift + bd + 1;
    (void)real_range_row;
    if (cfg->txfm_type_row == TXFM_TYPE_ADST4 && i == 1) {
      // the adst4 may use 1 extra bit on top of opt_range_row at stage 1
      // so opt_range_row >= real_range_row will not hold
      stage_range_row[i] = opt_range_row;
    } else {
      assert(opt_range_row >= real_range_row);
      stage_range_row[i] = opt_range_row;
    }
  }
  // i < MAX_TXFM_STAGE_NUM will mute above array bounds warning
  for (int i = 0; i < cfg->stage_num_col && i < MAX_TXFM_STAGE_NUM; ++i) {
    int real_range_col =
        cfg->stage_range_col[i] + fwd_shift + shift[0] + bd + 1;
    (void)real_range_col;
    if (cfg->txfm_type_col == TXFM_TYPE_ADST4 && i == 1) {
      // the adst4 may use 1 extra bit on top of opt_range_col at stage 1
      // so opt_range_col >= real_range_col will not hold
      stage_range_col[i] = opt_range_col;
    } else {
      assert(opt_range_col >= real_range_col);
      stage_range_col[i] = opt_range_col;
    }
  }
}

#if CONFIG_MODE_DEP_INTRA_TX && CONFIG_MODE_DEP_NONSEP_INTRA_TX
// Apply ordinary inverse non-separable transform (inv_nonsep_txfm2d)
// on 4x4 blocks.
static INLINE void inv_nonsep_txfm2d_add(const int32_t *input, uint16_t *output,
                                         const int stride, int32_t *txfm_buf,
                                         const int32_t *nstx_mtx,
                                         const TX_SIZE tx_size, int bd) {
  int ud_flip = 0, lr_flip = 0;
  const int tx_stride = tx_size_wide[tx_size] * tx_size_high[tx_size];

  // column/row indices in pixel (p) or transform (t) domains
  int cp, rp, ct, rt, kp, kt, l;
  int txw = tx_size_wide[tx_size], txh = tx_size_high[tx_size];

  // Values of input[l]: transform coefficients * 2^3
  // Max possible bit depth: 19

  // initialize txfm_buf
  for (rt = 0; rt < txh; ++rt)
    for (ct = 0; ct < txw; ++ct) txfm_buf[rt * txw + ct] = 0;

  // 2D transform
  for (rp = 0; rp < txh; ++rp) {
    for (cp = 0; cp < txw; ++cp) {
      kt = idx_flip(txw, txh, rp, cp, ud_flip, lr_flip);
      for (rt = 0; rt < txh; ++rt) {
        for (ct = 0; ct < txw; ++ct) {
          l = rt * txw + ct;
          // Values of txfm_buf[kt] = residue value * 2*3 * 2^8 * 2^(-1)
          //                        = residue value * 2^(8+2)
          // Bit depth = 19 + 3 + bd - 1 = 29
          // Max possible bit depth = 19 + 3 + 16 - 1 = 37
          // However, the transform coefficients are supposed to come from
          // residues, and those operations are the reversal of the previous
          // forward transform. Thus, there should not be an overflow issue.
          txfm_buf[rp * txw + cp] +=
              round_shift(nstx_mtx[l * tx_stride + kt] * input[l], 1);
        }
      }
    }
  }

  for (rt = 0; rt < txh; ++rt) {
    for (ct = 0; ct < txw; ++ct) {
      kp = rt * stride + ct;
      kt = rt * txw + ct;
      // Values of txfm_buf[kt] = residue value
      txfm_buf[kt] = round_shift(txfm_buf[kt], 10);
      output[kp] = highbd_clip_pixel_add(output[kp], txfm_buf[kt], bd);
    }
  }
}

#if CONFIG_MODE_DEP_NONSEP_SEC_INTRA_TX
// Apply a simplified inverse non-separable transform--inverse secondary
// transform (inv_nonsep_secondary_txfm2d) for blocks larger than 4x4.
static INLINE void inv_nonsep_secondary_txfm2d(const int32_t *input,
                                               int32_t *nsst_buf,
                                               const int32_t *nsst_mtx,
                                               const TX_SIZE tx_size) {
  const int txw = tx_size_wide[tx_size], txh = tx_size_high[tx_size];
  const int txwh = txw / 2, txhh = txh / 2;
  const int tx_stride = txwh * txhh;
  int k, l, rt, ct, rp, cp;

#if MDTX_DEBUG
  fprintf(stderr, "INV-NSST: before NSST\n");
  for (rt = 0; rt < txh; ++rt) {
    for (ct = 0; ct < txw; ++ct) {
      fprintf(stderr, "%3d ", input[rt * txw + ct]);
    }
    fprintf(stderr, "\n");
  }
#endif

  // Initialize nsst_buf
  for (rp = 0; rp < txh; ++rp)
    for (cp = 0; cp < txw; ++cp) nsst_buf[rp * txw + cp] = 0;

  // Apply a 2D non-separable transform on the 1/4 block (only the 1/4
  // top-left part of nsst_buf will be used). Note: stride in input[] and
  // nsst_buf[] should be txw.
  for (rp = 0; rp < txhh; ++rp) {
    for (cp = 0; cp < txwh; ++cp) {
      k = rp * txwh + cp;
      for (rt = 0; rt < txhh; ++rt) {
        for (ct = 0; ct < txwh; ++ct) {
          l = rt * txwh + ct;
          // Values of nsst_buf[kt] = residue value * 2*3 * 2^8 * 2^(-1)
          //                        = residue value * 2^(8+2)
          // Bit depth = 19 + 3 + bd - 1 = 29
          // Max possible bit depth = 19 + 3 + 16 - 1 = 37
          // However, the transform coefficients are supposed to come from
          // residues, and those operations are the reversal of the previous
          // forward transform. Thus, there should not be an overflow issue.
          nsst_buf[rp * txw + cp] += round_shift(
              nsst_mtx[l * tx_stride + k] * input[rt * txw + ct], 1);
#if 0
          fprintf(stderr, "(%d,%d,%d)[%d,%d,%d]", l, tx_stride, k,
                  nsst_mtx[l * tx_stride + k], input[rt * txw + ct],
                  nsst_buf[rp * txw + cp]);
#endif
        }
#if 0
        fprintf(stderr, "\n");
#endif
      }
    }
  }

  for (rt = 0; rt < txhh; ++rt)
    for (ct = 0; ct < txwh; ++ct)
      nsst_buf[rt * txw + ct] = round_shift(nsst_buf[rt * txw + ct], 7);

#if MDTX_DEBUG
  fprintf(stderr, "INV-NSST: after NSST\n");
  for (rt = 0; rt < txh; ++rt) {
    for (ct = 0; ct < txw; ++ct) {
      fprintf(stderr, "%3d ", nsst_buf[rt * txw + ct]);
    }
    fprintf(stderr, "\n");
  }
#endif
}
#endif  // CONFIG_MODE_DEP_NONSEP_SEC_INTRA_TX
#endif  // CONFIG_MODE_DEP_INTRA_TX &&
        // CONFIG_MODE_DEP_NONSEP_INTRA_TX

static INLINE void inv_txfm2d_add_c(const int32_t *input, uint16_t *output,
                                    int stride, TXFM_2D_FLIP_CFG *cfg,
                                    int32_t *txfm_buf, TX_SIZE tx_size,
                                    int bd) {
#if CONFIG_MODE_DEP_INTRA_TX && CONFIG_MODE_DEP_NONSEP_INTRA_TX
#if CONFIG_MODE_DEP_NONSEP_SEC_INTRA_TX
  DECLARE_ALIGNED(32, int, nsst_buf[8 * 8 + 8 + 8]);
#endif  // CONFIG_MODE_DEP_NONSEP_SEC_INTRA_TX
  if (cfg->nstx_mtx_ptr) {
#if !CONFIG_MODE_DEP_NONSEP_SEC_INTRA_TX
    // 4x4 non-separable transform
    inv_nonsep_txfm2d_add(input, output, stride, txfm_buf, cfg->nstx_mtx_ptr,
                          cfg->tx_size, bd);
    return;
#else
    if (tx_size == TX_4X4) {
      // 4x4 non-separable transform
      inv_nonsep_txfm2d_add(input, output, stride, txfm_buf, cfg->nstx_mtx_ptr,
                            cfg->tx_size, bd);
      return;
    } else {
      // In-place inverse secondary transform
      inv_nonsep_secondary_txfm2d(input, nsst_buf, cfg->nstx_mtx_ptr,
                                  cfg->tx_size);
    }
#endif  // !CONFIG_MODE_DEP_NONSEP_SEC_INTRA_TX
  }
#endif

  // Note when assigning txfm_size_col, we use the txfm_size from the
  // row configuration and vice versa. This is intentionally done to
  // accurately perform rectangular transforms. When the transform is
  // rectangular, the number of columns will be the same as the
  // txfm_size stored in the row cfg struct. It will make no difference
  // for square transforms.
  const int txfm_size_col = tx_size_wide[cfg->tx_size];
  const int txfm_size_row = tx_size_high[cfg->tx_size];
  // Take the shift from the larger dimension in the rectangular case.
  const int8_t *shift = cfg->shift;
  const int rect_type = get_rect_tx_log_ratio(txfm_size_col, txfm_size_row);
  int8_t stage_range_row[MAX_TXFM_STAGE_NUM + 1];
  int8_t stage_range_col[MAX_TXFM_STAGE_NUM + 1];
  assert(cfg->stage_num_row <= MAX_TXFM_STAGE_NUM);
  assert(cfg->stage_num_col <= MAX_TXFM_STAGE_NUM);
  av1_gen_inv_stage_range(stage_range_col, stage_range_row, cfg, tx_size, bd);

  const int8_t cos_bit_col = cfg->cos_bit_col;
  const int8_t cos_bit_row = cfg->cos_bit_row;
  const TxfmFunc txfm_func_col =
      inv_txfm_type_to_func(cfg->mode, cfg->txfm_type_col);
  const TxfmFunc txfm_func_row =
      inv_txfm_type_to_func(cfg->mode, cfg->txfm_type_row);
  stage_range_col[MAX_TXFM_STAGE_NUM] = (int)cfg->mode;
  stage_range_row[MAX_TXFM_STAGE_NUM] = (int)cfg->mode;

  // txfm_buf's length is  txfm_size_row * txfm_size_col + 2 *
  // AOMMAX(txfm_size_row, txfm_size_col)
  // it is used for intermediate data buffering
  const int buf_offset = AOMMAX(txfm_size_row, txfm_size_col);
  int32_t *temp_in = txfm_buf;
  int32_t *temp_out = temp_in + buf_offset;
  int32_t *buf = temp_out + buf_offset;
  int32_t *buf_ptr = buf;
  int c, r;

#if CONFIG_MODE_DEP_INTRA_TX && CONFIG_MODE_DEP_NONSEP_INTRA_TX && MDTX_DEBUG
  if (txfm_size_col <= 8 && txfm_size_row <= 8 && cfg->nstx_mtx_ptr) {
#if 0
    fprintf(stderr, "INV: input block\n");
    for (r = 0; r < txfm_size_row; ++r) {
      for (c = 0; c < txfm_size_col; ++c) {
        fprintf(stderr, "%3d ", input[r * txfm_size_col + c]);
      }
      fprintf(stderr, "\n");
    }
#endif
    fprintf(stderr, "INV: original output block (predicted block)\n");
    for (r = 0; r < txfm_size_row; ++r) {
      for (c = 0; c < txfm_size_col; ++c) {
        fprintf(stderr, "%3d ", output[r * stride + c]);
      }
      fprintf(stderr, "\n");
    }
  }
#endif

  // Rows
  for (r = 0; r < txfm_size_row; ++r) {
    if (abs(rect_type) == 1) {
      for (c = 0; c < txfm_size_col; ++c) {
#if CONFIG_MODE_DEP_INTRA_TX && CONFIG_MODE_DEP_NONSEP_INTRA_TX && \
    CONFIG_MODE_DEP_NONSEP_SEC_INTRA_TX
        // when secondary transforms are used, replace the transform
        // coefficients in the top-left subblock by those after inverse
        // secondary transforms
        if (cfg->nstx_mtx_ptr && r < txfm_size_row / 2 && c < txfm_size_col / 2)
          temp_in[c] = round_shift(
              (int64_t)nsst_buf[r * txfm_size_col + c] * NewInvSqrt2,
              NewSqrt2Bits);
        else
#endif  // CONFIG_MODE_DEP_INTRA_TX &&
        // CONFIG_MODE_DEP_NONSEP_INTRA_TX &&
        // CONFIG_MODE_DEP_NONSEP_SEC_INTRA_TX
          temp_in[c] =
              round_shift((int64_t)input[c] * NewInvSqrt2, NewSqrt2Bits);
      }
    }
  }

  // Rows
  for (r = 0; r < txfm_size_row; ++r) {
    if ((abs(rect_type) % 2) == 1) {
      for (c = 0; c < txfm_size_col; ++c) {
        temp_in[c] = round_shift((int64_t)input[c] * NewInvSqrt2, NewSqrt2Bits);
      }
      clamp_buf(temp_in, txfm_size_col, bd + 8);
      txfm_func_row(temp_in, buf_ptr, cos_bit_row, stage_range_row);
    } else {
      for (c = 0; c < txfm_size_col; ++c) {
#if CONFIG_MODE_DEP_INTRA_TX && CONFIG_MODE_DEP_NONSEP_INTRA_TX && \
    CONFIG_MODE_DEP_NONSEP_SEC_INTRA_TX
        if (cfg->nstx_mtx_ptr && r < txfm_size_row / 2 && c < txfm_size_col / 2)
          temp_in[c] = nsst_buf[r * txfm_size_col + c];
        else
#endif  // CONFIG_MODE_DEP_INTRA_TX &&
        // CONFIG_MODE_DEP_NONSEP_INTRA_TX &&
        // CONFIG_MODE_DEP_NONSEP_SEC_INTRA_TX
          temp_in[c] = input[c];
      }
      clamp_buf(temp_in, txfm_size_col, bd + 8);
      txfm_func_row(temp_in, buf_ptr, cos_bit_row, stage_range_row);
    }
    av1_round_shift_array(buf_ptr, txfm_size_col, -shift[0]);
    input += txfm_size_col;
    buf_ptr += txfm_size_col;
  }

  // Columns
  for (c = 0; c < txfm_size_col; ++c) {
    if (cfg->lr_flip == 0) {
      for (r = 0; r < txfm_size_row; ++r)
        temp_in[r] = buf[r * txfm_size_col + c];
    } else {
      // flip left right
      for (r = 0; r < txfm_size_row; ++r)
        temp_in[r] = buf[r * txfm_size_col + (txfm_size_col - c - 1)];
    }
    clamp_buf(temp_in, txfm_size_row, AOMMAX(bd + 6, 16));
    txfm_func_col(temp_in, temp_out, cos_bit_col, stage_range_col);
    av1_round_shift_array(temp_out, txfm_size_row, -shift[1]);
    if (cfg->ud_flip == 0) {
      for (r = 0; r < txfm_size_row; ++r) {
        output[r * stride + c] =
            highbd_clip_pixel_add(output[r * stride + c], temp_out[r], bd);
      }
    } else {
      // flip upside down
      for (r = 0; r < txfm_size_row; ++r) {
        output[r * stride + c] = highbd_clip_pixel_add(
            output[r * stride + c], temp_out[txfm_size_row - r - 1], bd);
      }
    }
  }

#if CONFIG_MODE_DEP_INTRA_TX && CONFIG_MODE_DEP_NONSEP_INTRA_TX && MDTX_DEBUG
  if (txfm_size_col <= 8 && txfm_size_row <= 8 && cfg->nstx_mtx_ptr) {
    fprintf(stderr, "INV: output block (with residues added)\n");
    for (r = 0; r < txfm_size_row; ++r) {
      for (c = 0; c < txfm_size_col; ++c) {
        fprintf(stderr, "%3d ", output[r * stride + c]);
      }
      fprintf(stderr, "\n");
    }
  }
#endif  // CONFIG_MODE_DEP_INTRA_TX &&
        // CONFIG_MODE_DEP_NONSEP_INTRA_TX &&
        // MDTX_DEBUG
}

static INLINE void inv_txfm2d_add_facade(const int32_t *input, uint16_t *output,
                                         int stride, int32_t *txfm_buf,
                                         TX_TYPE tx_type, TX_SIZE tx_size,
                                         PREDICTION_MODE mode, int bd) {
  TXFM_2D_FLIP_CFG cfg;
  av1_get_inv_txfm_cfg(tx_type, tx_size, mode, &cfg);
  // Forward shift sum uses larger square size, to be consistent with what
  // av1_gen_inv_stage_range() does for inverse shifts.
  inv_txfm2d_add_c(input, output, stride, &cfg, txfm_buf, tx_size, bd);
}

#if CONFIG_NEW_TX64X64
static INLINE void inv_txfm2d_c(const int32_t *input, int16_t *output,
                                int stride, TXFM_2D_FLIP_CFG *cfg,
                                int32_t *txfm_buf, TX_SIZE tx_size, int bd) {
#if CONFIG_MODE_DEP_INTRA_TX && CONFIG_MODE_DEP_NONSEP_INTRA_TX
  if (cfg->nstx_mtx_ptr) {
    inv_nonsep_txfm2d(input, output, stride, txfm_buf, cfg->nstx_mtx_ptr,
                      cfg->tx_size, bd);
    return;
  }
#endif  // CONFIG_MODE_DEP_INTRA_TX &&
        // CONFIG_MODE_DEP_NONSEP_INTRA_TX
  // Note when assigning txfm_size_col, we use the txfm_size from the
  // row configuration and vice versa. This is intentionally done to
  // accurately perform rectangular transforms. When the transform is
  // rectangular, the number of columns will be the same as the
  // txfm_size stored in the row cfg struct. It will make no difference
  // for square transforms.
  const int txfm_size_col = tx_size_wide[cfg->tx_size];
  const int txfm_size_row = tx_size_high[cfg->tx_size];
  // Take the shift from the larger dimension in the rectangular case.
  const int8_t *shift = cfg->shift;
  const int rect_type = get_rect_tx_log_ratio(txfm_size_col, txfm_size_row);
  int8_t stage_range_row[MAX_TXFM_STAGE_NUM + 1];
  int8_t stage_range_col[MAX_TXFM_STAGE_NUM + 1];
  assert(cfg->stage_num_row <= MAX_TXFM_STAGE_NUM);
  assert(cfg->stage_num_col <= MAX_TXFM_STAGE_NUM);
  av1_gen_inv_stage_range(stage_range_col, stage_range_row, cfg, tx_size, bd);

  const int8_t cos_bit_col = cfg->cos_bit_col;
  const int8_t cos_bit_row = cfg->cos_bit_row;
  const TxfmFunc txfm_func_col =
      inv_txfm_type_to_func(cfg->mode, cfg->txfm_type_col);
  const TxfmFunc txfm_func_row =
      inv_txfm_type_to_func(cfg->mode, cfg->txfm_type_row);
  stage_range_col[MAX_TXFM_STAGE_NUM] = (int)cfg->mode;
  stage_range_row[MAX_TXFM_STAGE_NUM] = (int)cfg->mode;

  // txfm_buf's length is  txfm_size_row * txfm_size_col + 2 *
  // AOMMAX(txfm_size_row, txfm_size_col)
  // it is used for intermediate data buffering
  const int buf_offset = AOMMAX(txfm_size_row, txfm_size_col);
  int32_t *temp_in = txfm_buf;
  int32_t *temp_out = temp_in + buf_offset;
  int32_t *buf = temp_out + buf_offset;
  int32_t *buf_ptr = buf;
  int c, r;

  // Rows
  for (r = 0; r < txfm_size_row; ++r) {
    if (abs(rect_type) == 1) {
      for (c = 0; c < txfm_size_col; ++c) {
        temp_in[c] = round_shift((int64_t)input[c] * NewInvSqrt2, NewSqrt2Bits);
      }
    }
  }

  // Rows
  for (r = 0; r < txfm_size_row; ++r) {
    if ((abs(rect_type) % 2) == 1) {
      for (c = 0; c < txfm_size_col; ++c) {
        temp_in[c] = round_shift((int64_t)input[c] * NewInvSqrt2, NewSqrt2Bits);
      }
      clamp_buf(temp_in, txfm_size_col, bd + 8);
      txfm_func_row(temp_in, buf_ptr, cos_bit_row, stage_range_row);
    } else {
      for (c = 0; c < txfm_size_col; ++c) {
        temp_in[c] = input[c];
      }
      clamp_buf(temp_in, txfm_size_col, bd + 8);
      txfm_func_row(temp_in, buf_ptr, cos_bit_row, stage_range_row);
    }
    av1_round_shift_array(buf_ptr, txfm_size_col, -shift[0]);
    input += txfm_size_col;
    buf_ptr += txfm_size_col;
  }

  // Columns
  for (c = 0; c < txfm_size_col; ++c) {
    if (cfg->lr_flip == 0) {
      for (r = 0; r < txfm_size_row; ++r)
        temp_in[r] = buf[r * txfm_size_col + c];
    } else {
      // flip left right
      for (r = 0; r < txfm_size_row; ++r)
        temp_in[r] = buf[r * txfm_size_col + (txfm_size_col - c - 1)];
    }
    clamp_buf(temp_in, txfm_size_row, AOMMAX(bd + 6, 16));
    txfm_func_col(temp_in, temp_out, cos_bit_col, stage_range_col);
    av1_round_shift_array(temp_out, txfm_size_row, -shift[1]);
    if (cfg->ud_flip == 0) {
      for (r = 0; r < txfm_size_row; ++r) {
        output[r * stride + c] = clip_pixel_signed(temp_out[r], bd);
      }
    } else {
      // flip upside down
      for (r = 0; r < txfm_size_row; ++r) {
        output[r * stride + c] =
            clip_pixel_signed(temp_out[txfm_size_row - r - 1], bd);
      }
    }
  }
}

static INLINE void inv_txfm2d_facade(const int32_t *input, int16_t *output,
                                     int stride, int32_t *txfm_buf,
                                     TX_TYPE tx_type, TX_SIZE tx_size,
                                     PREDICTION_MODE mode, int bd) {
  TXFM_2D_FLIP_CFG cfg;
  av1_get_inv_txfm_cfg(tx_type, tx_size, mode, &cfg);
  // Forward shift sum uses larger square size, to be consistent with what
  // av1_gen_inv_stage_range() does for inverse shifts.
  inv_txfm2d_c(input, output, stride, &cfg, txfm_buf, tx_size, bd);
}
#endif  // CONFIG_NEW_TX64X64

void av1_inv_txfm2d_add_4x8_c(const int32_t *input, uint16_t *output,
                              int stride, TX_TYPE tx_type, PREDICTION_MODE mode,
                              int bd) {
  DECLARE_ALIGNED(32, int, txfm_buf[4 * 8 + 8 + 8]);
  inv_txfm2d_add_facade(input, output, stride, txfm_buf, tx_type, TX_4X8, mode,
                        bd);
}

void av1_inv_txfm2d_add_8x4_c(const int32_t *input, uint16_t *output,
                              int stride, TX_TYPE tx_type, PREDICTION_MODE mode,
                              int bd) {
  DECLARE_ALIGNED(32, int, txfm_buf[8 * 4 + 8 + 8]);
  inv_txfm2d_add_facade(input, output, stride, txfm_buf, tx_type, TX_8X4, mode,
                        bd);
}

void av1_inv_txfm2d_add_8x16_c(const int32_t *input, uint16_t *output,
                               int stride, TX_TYPE tx_type,
                               PREDICTION_MODE mode, int bd) {
  DECLARE_ALIGNED(32, int, txfm_buf[8 * 16 + 16 + 16]);
  inv_txfm2d_add_facade(input, output, stride, txfm_buf, tx_type, TX_8X16, mode,
                        bd);
}

void av1_inv_txfm2d_add_16x8_c(const int32_t *input, uint16_t *output,
                               int stride, TX_TYPE tx_type,
                               PREDICTION_MODE mode, int bd) {
  DECLARE_ALIGNED(32, int, txfm_buf[16 * 8 + 16 + 16]);
  inv_txfm2d_add_facade(input, output, stride, txfm_buf, tx_type, TX_16X8, mode,
                        bd);
}

void av1_inv_txfm2d_add_16x32_c(const int32_t *input, uint16_t *output,
                                int stride, TX_TYPE tx_type,
                                PREDICTION_MODE mode, int bd) {
  DECLARE_ALIGNED(32, int, txfm_buf[16 * 32 + 32 + 32]);
  inv_txfm2d_add_facade(input, output, stride, txfm_buf, tx_type, TX_16X32,
                        mode, bd);
}

void av1_inv_txfm2d_add_32x16_c(const int32_t *input, uint16_t *output,
                                int stride, TX_TYPE tx_type,
                                PREDICTION_MODE mode, int bd) {
  DECLARE_ALIGNED(32, int, txfm_buf[32 * 16 + 32 + 32]);
  inv_txfm2d_add_facade(input, output, stride, txfm_buf, tx_type, TX_32X16,
                        mode, bd);
}

void av1_inv_txfm2d_add_4x4_c(const int32_t *input, uint16_t *output,
                              int stride, TX_TYPE tx_type, PREDICTION_MODE mode,
                              int bd) {
  DECLARE_ALIGNED(32, int, txfm_buf[4 * 4 + 4 + 4]);
  inv_txfm2d_add_facade(input, output, stride, txfm_buf, tx_type, TX_4X4, mode,
                        bd);
}

void av1_inv_txfm2d_add_8x8_c(const int32_t *input, uint16_t *output,
                              int stride, TX_TYPE tx_type, PREDICTION_MODE mode,
                              int bd) {
  DECLARE_ALIGNED(32, int, txfm_buf[8 * 8 + 8 + 8]);
  inv_txfm2d_add_facade(input, output, stride, txfm_buf, tx_type, TX_8X8, mode,
                        bd);
}

void av1_inv_txfm2d_add_16x16_c(const int32_t *input, uint16_t *output,
                                int stride, TX_TYPE tx_type,
                                PREDICTION_MODE mode, int bd) {
  DECLARE_ALIGNED(32, int, txfm_buf[16 * 16 + 16 + 16]);
  inv_txfm2d_add_facade(input, output, stride, txfm_buf, tx_type, TX_16X16,
                        mode, bd);
}

void av1_inv_txfm2d_add_32x32_c(const int32_t *input, uint16_t *output,
                                int stride, TX_TYPE tx_type,
                                PREDICTION_MODE mode, int bd) {
  DECLARE_ALIGNED(32, int, txfm_buf[32 * 32 + 32 + 32]);
  inv_txfm2d_add_facade(input, output, stride, txfm_buf, tx_type, TX_32X32,
                        mode, bd);
}

#if CONFIG_NEW_TX64X64
static int is_constant_buffer(int16_t *buf, int w, int h, int stride) {
  const int16_t topleftcorner = buf[0];
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      if (buf[i * stride + j] != topleftcorner) return 0;
    }
  }
  return 1;
}

#if USE_SUPERRES_FILTER_TX64

#define STX64X64_FILTER_TAPS 32
#define STX64XN_FILTER_TAPS 20
#define STXNX64_FILTER_TAPS 20

#define STX64X64_FILTER_COEFFS 16
#define STX64XN_FILTER_COEFFS 10
#define STXNX64_FILTER_COEFFS 10

// Number of 1D edge bands
#define NUM_EDGE_BANDS 4

// Number of 2D edge classes
// Note: This value is 25 for NUM_EDGE_BANDS = 4
#define NUM_EDGE_CLASSES (2 * NUM_EDGE_BANDS * (NUM_EDGE_BANDS - 1) + 1)

static uint8_t get_class_from_grad(int gx, int gy, int bd) {
  const int thresh[NUM_EDGE_BANDS - 1] = { 2 << (bd - 8), 5 << (bd - 8),
                                           20 << (bd - 8) };
  const int samesign = (gx * gy > 0);
  int cx = 0, cy = 0;
  if (gx > thresh[2])
    cx = 3;
  else if (gx > thresh[1])
    cx = 2;
  else if (gx > thresh[0])
    cx = 1;
  else
    cx = 0;

  if (gy > thresh[2])
    cy = 3;
  else if (gy > thresh[1])
    cy = 2;
  else if (gy > thresh[0])
    cy = 1;
  else
    cy = 0;

  uint8_t c = cx * 4 + cy;
  uint8_t d = (cx - 1) * (NUM_EDGE_BANDS - 1) + cy - 1;
  if (cx > 0 && cy > 0 && samesign) c = NUM_EDGE_BANDS * NUM_EDGE_BANDS + d;
  assert(c >= 0 && c < NUM_EDGE_CLASSES);
  return c;
}

static void edge_based_classify(const uint16_t *dgd, int width, int height,
                                int stride, uint8_t *cls, int cls_stride,
                                int bit_depth) {
  {
    int i = 0, j = 0;
    const int id = i * stride + j;
    const int gx = 6 * (dgd[id + 1] - dgd[id]) +
                   2 * (dgd[id + stride + 1] - dgd[id + stride]);
    const int gy = 6 * (dgd[id + stride] - dgd[id]) +
                   2 * (dgd[id + stride + 1] - dgd[id + 1]);
    cls[i * cls_stride + j] = get_class_from_grad(gx, gy, bit_depth);
  }
  {
    int i = 0, j = width - 1;
    const int id = i * stride + j;
    const int gx = 6 * (dgd[id] - dgd[id - 1]) +
                   2 * (dgd[id + stride] - dgd[id + stride - 1]);
    const int gy = 6 * (dgd[id + stride] - dgd[id]) +
                   2 * (dgd[id + stride - 1] - dgd[id - 1]);
    cls[i * cls_stride + j] = get_class_from_grad(gx, gy, bit_depth);
  }
  {
    int i = height - 1, j = 0;
    const int id = i * stride + j;
    const int gx = 6 * (dgd[id + 1] - dgd[id]) +
                   2 * (dgd[id - stride + 1] - dgd[id - stride]);
    const int gy = 6 * (dgd[id] - dgd[id - stride]) +
                   2 * (dgd[id + 1] - dgd[id - stride + 1]);
    cls[i * cls_stride + j] = get_class_from_grad(gx, gy, bit_depth);
  }
  {
    int i = height - 1, j = width - 1;
    const int id = i * stride + j;
    const int gx = 6 * (dgd[id] - dgd[id - 1]) +
                   2 * (dgd[id - stride] - dgd[id - stride - 1]);
    const int gy = 6 * (dgd[id] - dgd[id - stride]) +
                   2 * (dgd[id - 1] - dgd[id - stride - 1]);
    cls[i * cls_stride + j] = get_class_from_grad(gx, gy, bit_depth);
  }
  for (int i = 0, j = 1; j < width - 1; ++j) {
    const int id = i * stride + j;
    const int gx = 3 * (dgd[id + 1] - dgd[id - 1]) +
                   1 * (dgd[id + stride + 1] - dgd[id + stride - 1]);
    const int gy = 4 * (dgd[id + stride] - dgd[id]) +
                   2 * (dgd[id + stride + 1] - dgd[id + 1]) +
                   2 * (dgd[id + stride - 1] - dgd[id - 1]);
    cls[i * cls_stride + j] = get_class_from_grad(gx, gy, bit_depth);
  }
  for (int i = height - 1, j = 1; j < width - 1; ++j) {
    const int id = i * stride + j;
    const int gx = 3 * (dgd[id + 1] - dgd[id - 1]) +
                   1 * (dgd[id - stride + 1] - dgd[id - stride - 1]);
    const int gy = 4 * (dgd[id] - dgd[id - stride]) +
                   2 * (dgd[id + 1] - dgd[id - stride + 1]) +
                   2 * (dgd[id - 1] - dgd[id - stride - 1]);
    cls[i * cls_stride + j] = get_class_from_grad(gx, gy, bit_depth);
  }
  for (int i = 1, j = 0; i < height - 1; ++i) {
    const int id = i * stride + j;
    const int gx = 4 * (dgd[id + 1] - dgd[id]) +
                   2 * (dgd[id + stride + 1] - dgd[id + stride]) +
                   2 * (dgd[id - stride + 1] - dgd[id - stride]);
    const int gy = 3 * (dgd[id + stride] - dgd[id - stride]) +
                   1 * (dgd[id + stride + 1] - dgd[id - stride + 1]);
    cls[i * cls_stride + j] = get_class_from_grad(gx, gy, bit_depth);
  }
  for (int i = 1, j = width - 1; i < height - 1; ++i) {
    const int id = i * stride + j;
    const int gx = 4 * (dgd[id] - dgd[id - 1]) +
                   2 * (dgd[id + stride] - dgd[id + stride - 1]) +
                   2 * (dgd[id - stride] - dgd[id - stride - 1]);
    const int gy = 3 * (dgd[id + stride] - dgd[id - stride]) +
                   1 * (dgd[id + stride - 1] - dgd[id - stride - 1]);
    cls[i * cls_stride + j] = get_class_from_grad(gx, gy, bit_depth);
  }
  for (int i = 1; i < height - 1; ++i) {
    for (int j = 1; j < width - 1; ++j) {
      const int id = i * stride + j;
      const int gx = 2 * (dgd[id + 1] - dgd[id - 1]) +
                     1 * (dgd[id + stride + 1] - dgd[id + stride - 1]) +
                     1 * (dgd[id - stride + 1] - dgd[id - stride - 1]);
      const int gy = 2 * (dgd[id + stride] - dgd[id - stride]) +
                     1 * (dgd[id + stride + 1] - dgd[id - stride + 1]) +
                     1 * (dgd[id + stride - 1] - dgd[id - stride - 1]);
      cls[i * cls_stride + j] = get_class_from_grad(gx, gy, bit_depth);
    }
  }
}

const int stx64x64_config[STX64X64_FILTER_TAPS][3] = {
  { 1, 0, 0 },    { -1, 0, 0 },   { 0, 1, 1 },   { 0, -1, 1 },  { 2, 0, 2 },
  { -2, 0, 2 },   { 0, 2, 3 },    { 0, -2, 3 },  { 1, 1, 4 },   { -1, -1, 4 },
  { 1, -1, 5 },   { -1, 1, 5 },   { 2, 2, 6 },   { -2, -2, 6 }, { 2, -2, 7 },
  { -2, 2, 7 },   { 3, 0, 8 },    { -3, 0, 8 },  { 0, 3, 9 },   { 0, -3, 9 },
  { 3, 3, 10 },   { -3, -3, 10 }, { 3, -3, 11 }, { -3, 3, 11 }, { 1, 2, 12 },
  { -1, -2, 12 }, { 1, -2, 13 },  { -1, 2, 13 }, { 2, 1, 14 },  { -2, -1, 14 },
  { 2, -1, 15 },  { -2, 1, 15 },
};

const int16_t stx64x64_filters[STX64X64_FILTER_COEFFS * NUM_EDGE_CLASSES] = {
  30,  -4,  7,   3,   91,  -12, 12, 1,   4,  11, -1, 3,  -15, 11, -5,  -14,
  86,  -14, -4,  -8,  110, -4,  23, -2,  6,  8,  -2, 4,  -35, 17, -10, -12,
  12,  31,  -1,  -10, 65,  -13, 14, -5,  4,  9,  2,  3,  -27, 16, -7,  -7,
  5,   27,  -14, -12, 4,   -13, 5,  -9,  3,  9,  2,  -1, -8,  10, 3,   11,
  -10, 77,  3,   -20, 130, -4,  11, -3,  5,  8,  0,  3,  -41, 19, -1,  -11,
  60,  41,  2,   -11, 117, -25, 21, -6,  7,  9,  -1, 4,  -43, 18, -14, -8,
  6,   42,  0,   -8,  82,  -14, 15, -3,  5,  8,  2,  3,  -34, 13, -12, -7,
  9,   25,  -8,  -14, 0,   -19, 9,  -4,  2,  11, 1,  0,  -4,  9,  -4,  5,
  42,  9,   3,   -8,  67,  -27, 12, -5,  11, 3,  0,  2,  -28, 18, -14, -4,
  48,  4,   -1,  1,   79,  -29, 12, -5,  10, 3,  0,  3,  -34, 14, -11, -1,
  17,  9,   0,   5,   85,  -17, 13, -4,  7,  2,  3,  3,  -35, 8,  -15, -2,
  -5,  18,  2,   -6,  27,  -10, 12, -4,  3,  9,  1,  1,  -16, 1,  -15, 1,
  12,  17,  -1,  -20, 1,   -20, 2,  -5,  12, 9,  -2, -2, -4,  14, -2,  3,
  26,  12,  -5,  -13, 11,  -29, 6,  -7,  12, 7,  -3, -2, -9,  17, -5,  4,
  17,  -4,  -2,  3,   25,  -19, 9,  -3,  6,  4,  0,  -1, -22, 8,  -7,  1,
  2,   1,   2,   0,   21,  -15, 7,  -3,  1,  4,  1,  0,  -15, 6,  -8,  0,
  57,  54,  -3,  -22, 104, -18, 22, -8,  8,  9,  -2, 3,  -42, 22, -11, -4,
  13,  53,  -4,  -20, 52,  -16, 15, -9,  6,  12, 0,  4,  -27, 16, -7,  -1,
  19,  43,  -14, -17, -8,  -21, 9,  -12, 5,  12, 0,  2,  -9,  16, 1,   10,
  43,  19,  -3,  -13, 54,  -26, 13, -5,  11, 4,  0,  2,  -26, 19, -9,  -2,
  11,  28,  -3,  -22, 23,  -14, 7,  -7,  6,  5,  1,  3,  -11, 17, -4,  2,
  5,   32,  -10, -19, -14, -10, 1,  -10, 4,  10, -1, 2,  -1,  17, 10,  4,
  30,  13,  -11, -15, -4,  -23, 6,  -7,  8,  4,  -2, 0,  -4,  15, -1,  11,
  10,  11,  -4,  -17, -10, -15, 3,  -3,  6,  2,  -2, -1, 3,   13, -1,  9,
  9,   16,  -4,  -17, -22, -4,  -6, -4,  2,  5,  0,  -1, 9,   15, 9,   -1,
};

const int stx64xn_config[STX64XN_FILTER_TAPS][3] = {
  { 1, 0, 0 },  { -1, 0, 0 }, { 0, 1, 1 },   { 0, -1, 1 },  { 0, 2, 2 },
  { 0, -2, 2 }, { 1, 1, 3 },  { -1, -1, 3 }, { 1, -1, 4 },  { -1, 1, 4 },
  { 0, 3, 5 },  { 0, -3, 5 }, { 1, 2, 6 },   { -1, -2, 6 }, { 1, -2, 7 },
  { -1, 2, 7 }, { 1, 3, 8 },  { -1, -3, 8 }, { 1, -3, 9 },  { -1, 3, 9 },
};

const int16_t stx64xn_filters[STX64XN_FILTER_COEFFS * NUM_EDGE_CLASSES] = {
  20, 21,  17,  84,  -22, 8,  -9,  6,   5,   6,   50,  46,  7,   103, -36,
  19, -23, 7,   6,   2,   10, 36,  -3,  72,  -23, 22,  -19, -1,  7,   4,
  12, 53,  -29, 6,   -13, 47, -11, -2,  2,   -5,  -32, 109, -3,  113, -37,
  3,  -21, 20,  13,  2,   19, 90,  -1,  110, -45, 12,  -36, 13,  15,  0,
  9,  68,  -8,  82,  -34, 16, -37, 5,   20,  0,   5,   59,  -24, 16,  -18,
  36, -19, 1,   18,  -11, 8,  41,  -11, 66,  -15, 10,  -25, 10,  10,  -1,
  8,  30,  0,   79,  -26, 9,  -32, 7,   10,  2,   -5,  28,  -3,  84,  -28,
  5,  -33, 7,   20,  3,   -4, 35,  -20, 29,  -29, 29,  -13, 6,   18,  -10,
  30, 3,   -6,  -1,  -2,  0,  -14, 12,  8,   -2,  22,  -9,  -12, 15,  -10,
  0,  -2,  9,   4,   -3,  15, -9,  -3,  26,  -14, 2,   -15, 9,   9,   -3,
  -1, 1,   -9,  19,  -18, 7,  -9,  12,  14,  -14, 33,  108, -7,  90,  -52,
  18, -36, 17,  11,  -1,  22, 74,  -23, 44,  -22, 25,  -26, 4,   15,  1,
  19, 47,  -14, 1,   -14, 33, -28, 8,   13,  -5,  16,  36,  -17, 49,  -12,
  12, -25, 11,  9,   3,   13, 39,  -17, 26,  -8,  11,  -16, 10,  10,  2,
  3,  44,  -22, -14, -13, 19, -1,  8,   2,   8,   35,  26,  -30, -22, -26,
  9,  12,  18,  7,   -12, 32, 1,   -12, -19, -7,  1,   4,   12,  5,   -3,
  2,  10,  -16, -20, 6,   8,  4,   5,   0,   0,
};

const int stxnx64_config[STXNX64_FILTER_TAPS][3] = {
  { 0, 1, 0 },  { 0, -1, 0 }, { 1, 0, 1 },   { -1, 0, 1 },  { 2, 0, 2 },
  { -2, 0, 2 }, { 1, 1, 3 },  { -1, -1, 3 }, { -1, 1, 4 },  { 1, -1, 4 },
  { 3, 0, 5 },  { -3, 0, 5 }, { 2, 1, 6 },   { -2, -1, 6 }, { -2, 1, 7 },
  { 2, -1, 7 }, { 3, 1, 8 },  { -3, -1, 8 }, { -3, 1, 9 },  { 3, -1, 9 },
};

const int16_t stxnx64_filters[STXNX64_FILTER_COEFFS * NUM_EDGE_CLASSES] = {
  20, 21,  17,  84,  -22, 8,  -9,  6,   5,   6,   -32, 109, -3,  113, -37,
  3,  -21, 20,  13,  2,   8,  41,  -11, 66,  -15, 10,  -25, 10,  10,  -1,
  30, 3,   -6,  -1,  -2,  0,  -14, 12,  8,   -2,  50,  46,  7,   103, -36,
  19, -23, 7,   6,   2,   19, 90,  -1,  110, -45, 12,  -36, 13,  15,  0,
  8,  30,  0,   79,  -26, 9,  -32, 7,   10,  2,   22,  -9,  -12, 15,  -10,
  0,  -2,  9,   4,   -3,  10, 36,  -3,  72,  -23, 22,  -19, -1,  7,   4,
  9,  68,  -8,  82,  -34, 16, -37, 5,   20,  0,   -5,  28,  -3,  84,  -28,
  5,  -33, 7,   20,  3,   15, -9,  -3,  26,  -14, 2,   -15, 9,   9,   -3,
  12, 53,  -29, 6,   -13, 47, -11, -2,  2,   -5,  5,   59,  -24, 16,  -18,
  36, -19, 1,   18,  -11, -4, 35,  -20, 29,  -29, 29,  -13, 6,   18,  -10,
  -1, 1,   -9,  19,  -18, 7,  -9,  12,  14,  -14, 33,  108, -7,  90,  -52,
  18, -36, 17,  11,  -1,  16, 36,  -17, 49,  -12, 12,  -25, 11,  9,   3,
  35, 26,  -30, -22, -26, 9,  12,  18,  7,   -12, 22,  74,  -23, 44,  -22,
  25, -26, 4,   15,  1,   13, 39,  -17, 26,  -8,  11,  -16, 10,  10,  2,
  32, 1,   -12, -19, -7,  1,  4,   12,  5,   -3,  19,  47,  -14, 1,   -14,
  33, -28, 8,   13,  -5,  3,  44,  -22, -14, -13, 19,  -1,  8,   2,   8,
  2,  10,  -16, -20, 6,   8,  4,   5,   0,   0,
};
#endif  // USE_SUPERRES_FILTER_TX64

void av1_inv_txfm2d_add_64x64_c(const int32_t *input, uint16_t *output,
                                int stride, TX_TYPE tx_type,
                                PREDICTION_MODE mode, int bd) {
  // Inverse 32x32 transform.
  DECLARE_ALIGNED(32, int, txfm_buf[32 * 32 + 32 + 32]);

  DECLARE_ALIGNED(32, int16_t, output_32x32[32 * 32]);
  memset(output_32x32, 0, 32 * 32 * sizeof(output_32x32[0]));
  inv_txfm2d_facade(input, output_32x32, 32, txfm_buf, tx_type, TX_32X32, mode,
                    bd);
  if (is_constant_buffer(output_32x32, 32, 32, 32)) {
    const tran_low_t residue = output_32x32[0];
    for (int r = 0; r < 64; ++r) {
      for (int c = 0; c < 64; ++c) {
        output[r * stride + c] =
            highbd_clip_pixel_add(output[r * stride + c], residue, bd);
      }
    }
  } else {
    // Upsample to 64x64.
    DECLARE_ALIGNED(32, int16_t, output_up[64 * 64]);
    av1_signed_up2(output_32x32, 32, 32, 32, output_up, 64, 1, 1, bd);
#if USE_SUPERRES_FILTER_TX64
    for (int r = 0; r < 64; ++r) {
      for (int c = 0; c < 64; ++c) {
        const tran_low_t residue = (tran_low_t)output_up[64 * r + c];
        output_up[r * 64 + c] =
            highbd_clip_pixel_add(output[r * stride + c], residue, bd);
      }
    }
    DECLARE_ALIGNED(32, uint8_t, cls[64 * 64]);
    NonsepFilterConfig nsfilter = { 10, STX64X64_FILTER_TAPS, 0,
                                    stx64x64_config, 1 };
    edge_based_classify((const uint16_t *)output_up, 64, 64, 64, cls, 64, bd);
    av1_convolve_nonsep_cls_highbd((const uint16_t *)output_up, 64, 64, 64,
                                   &nsfilter, cls, 64, stx64x64_filters, 16,
                                   output, stride, bd);
#else
    for (int r = 0; r < 64; ++r) {
      for (int c = 0; c < 64; ++c) {
        const tran_low_t residue = (tran_low_t)output_up[64 * r + c];
        output[r * stride + c] =
            highbd_clip_pixel_add(output[r * stride + c], residue, bd);
      }
    }
#endif  // USE_SUPERRES_FILTER_TX64
  }
}

void av1_inv_txfm2d_add_32x64_c(const int32_t *input, uint16_t *output,
                                int stride, TX_TYPE tx_type,
                                PREDICTION_MODE mode, int bd) {
  // Inverse 32x32 transform.
  DECLARE_ALIGNED(32, int, txfm_buf[32 * 32 + 32 + 32]);

  DECLARE_ALIGNED(32, int16_t, output_32x32[32 * 32]);
  memset(output_32x32, 0, 32 * 32 * sizeof(output_32x32[0]));
  inv_txfm2d_facade(input, output_32x32, 32, txfm_buf, tx_type, TX_32X32, mode,
                    bd);
  // Scale.
  for (int r = 0; r < 32; ++r) {
    for (int c = 0; c < 32; ++c) {
      output_32x32[r * 32 + c] = (int16_t)round_shift(
          (int64_t)output_32x32[r * 32 + c] * NewSqrt2, NewSqrt2Bits);
    }
  }

  if (is_constant_buffer(output_32x32, 32, 32, 32)) {
    const tran_low_t residue = output_32x32[0];
    for (int r = 0; r < 64; ++r) {
      for (int c = 0; c < 32; ++c) {
        output[r * stride + c] =
            highbd_clip_pixel_add(output[r * stride + c], residue, bd);
      }
    }
  } else {
    DECLARE_ALIGNED(32, int16_t, output_up[32 * 64]);
    // Upsample to 32x64.
    av1_signed_up2(output_32x32, 32, 32, 32, output_up, 32, 1, 0, bd);
#if USE_SUPERRES_FILTER_TX64
    for (int r = 0; r < 64; ++r) {
      for (int c = 0; c < 32; ++c) {
        const tran_low_t residue = (tran_low_t)output_up[32 * r + c];
        output_up[r * 32 + c] =
            highbd_clip_pixel_add(output[r * stride + c], residue, bd);
      }
    }
    DECLARE_ALIGNED(32, uint8_t, cls[32 * 64]);
    NonsepFilterConfig nsfilter = { 10, STXNX64_FILTER_TAPS, 0, stxnx64_config,
                                    1 };
    edge_based_classify((const uint16_t *)output_up, 32, 64, 32, cls, 32, bd);
    av1_convolve_nonsep_cls_highbd((const uint16_t *)output_up, 32, 64, 32,
                                   &nsfilter, cls, 32, stxnx64_filters, 10,
                                   output, stride, bd);
#else
    for (int r = 0; r < 64; ++r) {
      for (int c = 0; c < 32; ++c) {
        const tran_low_t residue = (tran_low_t)output_up[32 * r + c];
        output[r * stride + c] =
            highbd_clip_pixel_add(output[r * stride + c], residue, bd);
      }
    }
#endif  // USE_SUPERRES_FILTER_TX64
  }
}

void av1_inv_txfm2d_add_64x32_c(const int32_t *input, uint16_t *output,
                                int stride, TX_TYPE tx_type,
                                PREDICTION_MODE mode, int bd) {
  // Inverse 32x32 transform.
  DECLARE_ALIGNED(32, int, txfm_buf[32 * 32 + 32 + 32]);

  DECLARE_ALIGNED(32, int16_t, output_32x32[32 * 32]);
  memset(output_32x32, 0, 32 * 32 * sizeof(output_32x32[0]));
  inv_txfm2d_facade(input, output_32x32, 32, txfm_buf, tx_type, TX_32X32, mode,
                    bd);

  // Scale.
  for (int r = 0; r < 32; ++r) {
    for (int c = 0; c < 32; ++c) {
      output_32x32[r * 32 + c] = (int16_t)round_shift(
          (int64_t)output_32x32[r * 32 + c] * NewSqrt2, NewSqrt2Bits);
    }
  }

  if (is_constant_buffer(output_32x32, 32, 32, 32)) {
    const tran_low_t residue = output_32x32[0];
    for (int r = 0; r < 32; ++r) {
      for (int c = 0; c < 64; ++c) {
        output[r * stride + c] =
            highbd_clip_pixel_add(output[r * stride + c], residue, bd);
      }
    }
  } else {
    DECLARE_ALIGNED(32, int16_t, output_up[64 * 32]);
    // Upsample to 64x32.
    av1_signed_up2(output_32x32, 32, 32, 32, output_up, 64, 0, 1, bd);
#if USE_SUPERRES_FILTER_TX64
    for (int r = 0; r < 32; ++r) {
      for (int c = 0; c < 64; ++c) {
        const tran_low_t residue = (tran_low_t)output_up[64 * r + c];
        output_up[r * 64 + c] =
            highbd_clip_pixel_add(output[r * stride + c], residue, bd);
      }
    }
    DECLARE_ALIGNED(32, uint8_t, cls[32 * 64]);
    NonsepFilterConfig nsfilter = { 10, STX64XN_FILTER_TAPS, 0, stx64xn_config,
                                    1 };
    edge_based_classify((const uint16_t *)output_up, 64, 32, 64, cls, 64, bd);
    av1_convolve_nonsep_cls_highbd((const uint16_t *)output_up, 64, 32, 64,
                                   &nsfilter, cls, 64, stx64xn_filters, 10,
                                   output, stride, bd);
#else
    for (int r = 0; r < 32; ++r) {
      for (int c = 0; c < 64; ++c) {
        const tran_low_t residue = (tran_low_t)output_up[64 * r + c];
        output[r * stride + c] =
            highbd_clip_pixel_add(output[r * stride + c], residue, bd);
      }
    }
#endif  // USE_SUPERRES_FILTER_TX64
  }
}

void av1_inv_txfm2d_add_16x64_c(const int32_t *input, uint16_t *output,
                                int stride, TX_TYPE tx_type,
                                PREDICTION_MODE mode, int bd) {
  // Inverse 16x32 transform.
  DECLARE_ALIGNED(32, int, txfm_buf[16 * 32 + 32 + 32]);

  DECLARE_ALIGNED(32, int16_t, output_16x32[16 * 32]);
  memset(output_16x32, 0, 16 * 32 * sizeof(output_16x32[0]));
  inv_txfm2d_facade(input, output_16x32, 16, txfm_buf, tx_type, TX_16X32, mode,
                    bd);

  // Scale.
  for (int r = 0; r < 32; ++r) {
    for (int c = 0; c < 16; ++c) {
      output_16x32[r * 16 + c] = (int16_t)round_shift(
          (int64_t)output_16x32[r * 16 + c] * NewInvSqrt2, NewSqrt2Bits);
    }
  }

  if (is_constant_buffer(output_16x32, 16, 32, 16)) {
    const tran_low_t residue = output_16x32[0];
    for (int r = 0; r < 64; ++r) {
      for (int c = 0; c < 16; ++c) {
        output[r * stride + c] =
            highbd_clip_pixel_add(output[r * stride + c], residue, bd);
      }
    }
  } else {
    DECLARE_ALIGNED(32, int16_t, output_up[16 * 64]);
    // Upsample to 16x64.
    av1_signed_up2(output_16x32, 32, 16, 16, output_up, 16, 1, 0, bd);
#if USE_SUPERRES_FILTER_TX64
    for (int r = 0; r < 64; ++r) {
      for (int c = 0; c < 16; ++c) {
        const tran_low_t residue = (tran_low_t)output_up[16 * r + c];
        output_up[r * 16 + c] =
            highbd_clip_pixel_add(output[r * stride + c], residue, bd);
      }
    }
    DECLARE_ALIGNED(32, uint8_t, cls[16 * 64]);
    NonsepFilterConfig nsfilter = { 10, STXNX64_FILTER_TAPS, 0, stxnx64_config,
                                    1 };
    edge_based_classify((const uint16_t *)output_up, 16, 64, 16, cls, 16, bd);
    av1_convolve_nonsep_cls_highbd((const uint16_t *)output_up, 16, 64, 16,
                                   &nsfilter, cls, 16, stxnx64_filters, 10,
                                   output, stride, bd);
#else
    for (int r = 0; r < 64; ++r) {
      for (int c = 0; c < 16; ++c) {
        const tran_low_t residue = (tran_low_t)output_up[16 * r + c];
        output[r * stride + c] =
            highbd_clip_pixel_add(output[r * stride + c], residue, bd);
      }
    }
#endif  // USE_SUPERRES_FILTER_TX64
  }
}

void av1_inv_txfm2d_add_64x16_c(const int32_t *input, uint16_t *output,
                                int stride, TX_TYPE tx_type,
                                PREDICTION_MODE mode, int bd) {
  // Inverse 32x16 transform.
  DECLARE_ALIGNED(32, int, txfm_buf[32 * 16 + 32 + 32]);

  DECLARE_ALIGNED(32, int16_t, output_32x16[32 * 16]);
  memset(output_32x16, 0, 32 * 16 * sizeof(output_32x16[0]));
  inv_txfm2d_facade(input, output_32x16, 32, txfm_buf, tx_type, TX_32X16, mode,
                    bd);

  // Scale.
  for (int r = 0; r < 16; ++r) {
    for (int c = 0; c < 32; ++c) {
      output_32x16[r * 32 + c] = (int16_t)round_shift(
          (int64_t)output_32x16[r * 32 + c] * NewInvSqrt2, NewSqrt2Bits);
    }
  }

  if (is_constant_buffer(output_32x16, 32, 16, 32)) {
    const tran_low_t residue = output_32x16[0];
    for (int r = 0; r < 16; ++r) {
      for (int c = 0; c < 64; ++c) {
        output[r * stride + c] =
            highbd_clip_pixel_add(output[r * stride + c], residue, bd);
      }
    }
  } else {
    DECLARE_ALIGNED(32, int16_t, output_up[64 * 16]);
    // Upsample to 64x16.
    av1_signed_up2(output_32x16, 16, 32, 32, output_up, 64, 0, 1, bd);
#if USE_SUPERRES_FILTER_TX64
    for (int r = 0; r < 16; ++r) {
      for (int c = 0; c < 64; ++c) {
        const tran_low_t residue = (tran_low_t)output_up[64 * r + c];
        output_up[r * 64 + c] =
            highbd_clip_pixel_add(output[r * stride + c], residue, bd);
      }
    }
    DECLARE_ALIGNED(32, uint8_t, cls[16 * 64]);
    NonsepFilterConfig nsfilter = { 10, STX64XN_FILTER_TAPS, 0, stx64xn_config,
                                    1 };
    edge_based_classify((const uint16_t *)output_up, 64, 16, 64, cls, 64, bd);
    av1_convolve_nonsep_cls_highbd((const uint16_t *)output_up, 64, 16, 64,
                                   &nsfilter, cls, 64, stx64xn_filters, 10,
                                   output, stride, bd);
#else
    for (int r = 0; r < 16; ++r) {
      for (int c = 0; c < 64; ++c) {
        const tran_low_t residue = (tran_low_t)output_up[64 * r + c];
        output[r * stride + c] =
            highbd_clip_pixel_add(output[r * stride + c], residue, bd);
      }
    }
#endif  // USE_SUPERRES_FILTER_TX64
  }
}

#else

void av1_inv_txfm2d_add_64x64_c(const int32_t *input, uint16_t *output,
                                int stride, TX_TYPE tx_type,
                                PREDICTION_MODE mode, int bd) {
  // TODO(urvang): Can the same array be reused, instead of using a new array?
  // Remap 32x32 input into a modified 64x64 by:
  // - Copying over these values in top-left 32x32 locations.
  // - Setting the rest of the locations to 0.
  int32_t mod_input[64 * 64];
  for (int row = 0; row < 32; ++row) {
    memcpy(mod_input + row * 64, input + row * 32, 32 * sizeof(*mod_input));
    memset(mod_input + row * 64 + 32, 0, 32 * sizeof(*mod_input));
  }
  memset(mod_input + 32 * 64, 0, 32 * 64 * sizeof(*mod_input));
  DECLARE_ALIGNED(32, int, txfm_buf[64 * 64 + 64 + 64]);
  inv_txfm2d_add_facade(mod_input, output, stride, txfm_buf, tx_type, TX_64X64,
                        mode, bd);
}

void av1_inv_txfm2d_add_64x32_c(const int32_t *input, uint16_t *output,
                                int stride, TX_TYPE tx_type,
                                PREDICTION_MODE mode, int bd) {
  // Remap 32x32 input into a modified 64x32 by:
  // - Copying over these values in top-left 32x32 locations.
  // - Setting the rest of the locations to 0.
  int32_t mod_input[64 * 32];
  for (int row = 0; row < 32; ++row) {
    memcpy(mod_input + row * 64, input + row * 32, 32 * sizeof(*mod_input));
    memset(mod_input + row * 64 + 32, 0, 32 * sizeof(*mod_input));
  }
  DECLARE_ALIGNED(32, int, txfm_buf[64 * 32 + 64 + 64]);
  inv_txfm2d_add_facade(mod_input, output, stride, txfm_buf, tx_type, TX_64X32,
                        mode, bd);
}

void av1_inv_txfm2d_add_32x64_c(const int32_t *input, uint16_t *output,
                                int stride, TX_TYPE tx_type,
                                PREDICTION_MODE mode, int bd) {
  // Remap 32x32 input into a modified 32x64 input by:
  // - Copying over these values in top-left 32x32 locations.
  // - Setting the rest of the locations to 0.
  int32_t mod_input[32 * 64];
  memcpy(mod_input, input, 32 * 32 * sizeof(*mod_input));
  memset(mod_input + 32 * 32, 0, 32 * 32 * sizeof(*mod_input));
  DECLARE_ALIGNED(32, int, txfm_buf[64 * 32 + 64 + 64]);
  inv_txfm2d_add_facade(mod_input, output, stride, txfm_buf, tx_type, TX_32X64,
                        mode, bd);
}

void av1_inv_txfm2d_add_16x64_c(const int32_t *input, uint16_t *output,
                                int stride, TX_TYPE tx_type,
                                PREDICTION_MODE mode, int bd) {
  // Remap 16x32 input into a modified 16x64 input by:
  // - Copying over these values in top-left 16x32 locations.
  // - Setting the rest of the locations to 0.
  int32_t mod_input[16 * 64];
  memcpy(mod_input, input, 16 * 32 * sizeof(*mod_input));
  memset(mod_input + 16 * 32, 0, 16 * 32 * sizeof(*mod_input));
  DECLARE_ALIGNED(32, int, txfm_buf[16 * 64 + 64 + 64]);
  inv_txfm2d_add_facade(mod_input, output, stride, txfm_buf, tx_type, TX_16X64,
                        mode, bd);
}

void av1_inv_txfm2d_add_64x16_c(const int32_t *input, uint16_t *output,
                                int stride, TX_TYPE tx_type,
                                PREDICTION_MODE mode, int bd) {
  // Remap 32x16 input into a modified 64x16 by:
  // - Copying over these values in top-left 32x16 locations.
  // - Setting the rest of the locations to 0.
  int32_t mod_input[64 * 16];
  for (int row = 0; row < 16; ++row) {
    memcpy(mod_input + row * 64, input + row * 32, 32 * sizeof(*mod_input));
    memset(mod_input + row * 64 + 32, 0, 32 * sizeof(*mod_input));
  }
  DECLARE_ALIGNED(32, int, txfm_buf[16 * 64 + 64 + 64]);
  inv_txfm2d_add_facade(mod_input, output, stride, txfm_buf, tx_type, TX_64X16,
                        mode, bd);
}

#endif  // CONFIG_NEW_TX64X64

void av1_inv_txfm2d_add_4x16_c(const int32_t *input, uint16_t *output,
                               int stride, TX_TYPE tx_type,
                               PREDICTION_MODE mode, int bd) {
  DECLARE_ALIGNED(32, int, txfm_buf[4 * 16 + 16 + 16]);
  inv_txfm2d_add_facade(input, output, stride, txfm_buf, tx_type, TX_4X16, mode,
                        bd);
}

void av1_inv_txfm2d_add_16x4_c(const int32_t *input, uint16_t *output,
                               int stride, TX_TYPE tx_type,
                               PREDICTION_MODE mode, int bd) {
  DECLARE_ALIGNED(32, int, txfm_buf[4 * 16 + 16 + 16]);
  inv_txfm2d_add_facade(input, output, stride, txfm_buf, tx_type, TX_16X4, mode,
                        bd);
}

void av1_inv_txfm2d_add_8x32_c(const int32_t *input, uint16_t *output,
                               int stride, TX_TYPE tx_type,
                               PREDICTION_MODE mode, int bd) {
  DECLARE_ALIGNED(32, int, txfm_buf[8 * 32 + 32 + 32]);
  inv_txfm2d_add_facade(input, output, stride, txfm_buf, tx_type, TX_8X32, mode,
                        bd);
}

void av1_inv_txfm2d_add_32x8_c(const int32_t *input, uint16_t *output,
                               int stride, TX_TYPE tx_type,
                               PREDICTION_MODE mode, int bd) {
  DECLARE_ALIGNED(32, int, txfm_buf[8 * 32 + 32 + 32]);
  inv_txfm2d_add_facade(input, output, stride, txfm_buf, tx_type, TX_32X8, mode,
                        bd);
}

#if CONFIG_FLEX_PARTITION
void av1_inv_txfm2d_add_4x32_c(const int32_t *input, uint16_t *output,
                               int stride, TX_TYPE tx_type,
                               PREDICTION_MODE mode, int bd) {
  DECLARE_ALIGNED(32, int, txfm_buf[4 * 32 + 32 + 32]);
  inv_txfm2d_add_facade(input, output, stride, txfm_buf, tx_type, TX_4X32, mode,
                        bd);
}

void av1_inv_txfm2d_add_32x4_c(const int32_t *input, uint16_t *output,
                               int stride, TX_TYPE tx_type,
                               PREDICTION_MODE mode, int bd) {
  DECLARE_ALIGNED(32, int, txfm_buf[4 * 32 + 32 + 32]);
  inv_txfm2d_add_facade(input, output, stride, txfm_buf, tx_type, TX_32X4, mode,
                        bd);
}

void av1_inv_txfm2d_add_8x64_c(const int32_t *input, uint16_t *output,
                               int stride, TX_TYPE tx_type,
                               PREDICTION_MODE mode, int bd) {
  // Remap 8x32 input into a modified 8x64 input by:
  // - Copying over these values in top-left 8x32 locations.
  // - Setting the rest of the locations to 0.
  int32_t mod_input[8 * 64];
  memcpy(mod_input, input, 8 * 32 * sizeof(*mod_input));
  memset(mod_input + 8 * 32, 0, 8 * 32 * sizeof(*mod_input));
  DECLARE_ALIGNED(32, int, txfm_buf[8 * 64 + 64 + 64]);
  inv_txfm2d_add_facade(mod_input, output, stride, txfm_buf, tx_type, TX_8X64,
                        mode, bd);
}

void av1_inv_txfm2d_add_64x8_c(const int32_t *input, uint16_t *output,
                               int stride, TX_TYPE tx_type,
                               PREDICTION_MODE mode, int bd) {
  // Remap 32x8 input into a modified 64x8 by:
  // - Copying over these values in top-left 32x8 locations.
  // - Setting the rest of the locations to 0.
  int32_t mod_input[64 * 8];
  for (int row = 0; row < 8; ++row) {
    memcpy(mod_input + row * 64, input + row * 32, 32 * sizeof(*mod_input));
    memset(mod_input + row * 64 + 32, 0, 32 * sizeof(*mod_input));
  }
  DECLARE_ALIGNED(32, int, txfm_buf[8 * 64 + 64 + 64]);
  inv_txfm2d_add_facade(mod_input, output, stride, txfm_buf, tx_type, TX_64X8,
                        mode, bd);
}

void av1_inv_txfm2d_add_4x64_c(const int32_t *input, uint16_t *output,
                               int stride, TX_TYPE tx_type,
                               PREDICTION_MODE mode, int bd) {
  // Remap 4x32 input into a modified 4x64 input by:
  // - Copying over these values in top-left 4x32 locations.
  // - Setting the rest of the locations to 0.
  int32_t mod_input[4 * 64];
  memcpy(mod_input, input, 4 * 32 * sizeof(*mod_input));
  memset(mod_input + 4 * 32, 0, 4 * 32 * sizeof(*mod_input));
  DECLARE_ALIGNED(32, int, txfm_buf[4 * 64 + 64 + 64]);
  inv_txfm2d_add_facade(mod_input, output, stride, txfm_buf, tx_type, TX_4X64,
                        mode, bd);
}

void av1_inv_txfm2d_add_64x4_c(const int32_t *input, uint16_t *output,
                               int stride, TX_TYPE tx_type,
                               PREDICTION_MODE mode, int bd) {
  // Remap 32x4 input into a modified 64x8 by:
  // - Copying over these values in top-left 32x4 locations.
  // - Setting the rest of the locations to 0.
  int32_t mod_input[64 * 4];
  for (int row = 0; row < 4; ++row) {
    memcpy(mod_input + row * 64, input + row * 32, 32 * sizeof(*mod_input));
    memset(mod_input + row * 64 + 32, 0, 32 * sizeof(*mod_input));
  }
  DECLARE_ALIGNED(32, int, txfm_buf[4 * 64 + 64 + 64]);
  inv_txfm2d_add_facade(mod_input, output, stride, txfm_buf, tx_type, TX_64X4,
                        mode, bd);
}
#endif  // CONFIG_FLEX_PARTITION
