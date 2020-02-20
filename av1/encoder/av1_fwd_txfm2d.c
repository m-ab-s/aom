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

#include <assert.h>

#include "config/aom_dsp_rtcd.h"
#include "config/av1_rtcd.h"

#include "aom_dsp/txfm_common.h"
#include "av1/common/enums.h"
#include "av1/common/av1_txfm.h"
#include "av1/encoder/av1_fwd_txfm1d.h"
#include "av1/encoder/av1_fwd_txfm1d_cfg.h"

#if CONFIG_NEW_TX64X64
#define USE_SIMPLE_DOWNSCALE 0
#if USE_SIMPLE_DOWNSCALE == 0
#include "av1/common/resize.h"
#endif  // USE_SIMPLE_DOWNSCALE == 0
#endif  // CONFIG_NEW_TX64X64

static INLINE TxfmFunc fwd_txfm_type_to_func(int mode, TXFM_TYPE txfm_type) {
  (void)mode;
  switch (txfm_type) {
    case TXFM_TYPE_DCT4: return av1_fdct4_new;
    case TXFM_TYPE_DCT8: return av1_fdct8_new;
    case TXFM_TYPE_DCT16: return av1_fdct16_new;
    case TXFM_TYPE_DCT32: return av1_fdct32_new;
    case TXFM_TYPE_DCT64: return av1_fdct64_new;
#if CONFIG_LGT
    case TXFM_TYPE_ADST4:
      return (mode >= INTER_MODE_START && mode < INTER_MODE_END)
                 ? av1_fadst4_lgt_inter
                 : av1_fadst4_lgt_intra;
    case TXFM_TYPE_ADST8:
      return (mode >= INTER_MODE_START && mode < INTER_MODE_END)
                 ? av1_fadst8_lgt_inter
                 : av1_fadst8_lgt_intra;
    case TXFM_TYPE_ADST16:
      return (mode >= INTER_MODE_START && mode < INTER_MODE_END)
                 ? av1_fadst16_lgt_inter
                 : av1_fadst16_lgt_intra;
#else
    case TXFM_TYPE_ADST4: return av1_fadst4_new;
    case TXFM_TYPE_ADST8: return av1_fadst8_new;
    case TXFM_TYPE_ADST16: return av1_fadst16_new;
#endif  // CONFIG_LGT
#if CONFIG_MODE_DEP_INTRA_TX || CONFIG_MODE_DEP_INTER_TX
    case TXFM_TYPE_MDTX4: return av1_fmdt4;
    case TXFM_TYPE_MDTX8: return av1_fmdt8;
    case TXFM_TYPE_MDTX16: return av1_fmdt16;
#endif
#if CONFIG_DST7_32x32
    case TXFM_TYPE_ADST32: return av1_fadst32_new;
#endif
    case TXFM_TYPE_IDENTITY4: return av1_fidentity4_c;
    case TXFM_TYPE_IDENTITY8: return av1_fidentity8_c;
    case TXFM_TYPE_IDENTITY16: return av1_fidentity16_c;
    case TXFM_TYPE_IDENTITY32: return av1_fidentity32_c;
    default: assert(0); return NULL;
  }
}

void av1_gen_fwd_stage_range(int8_t *stage_range_col, int8_t *stage_range_row,
                             const TXFM_2D_FLIP_CFG *cfg, int bd) {
  // Take the shift from the larger dimension in the rectangular case.
  const int8_t *shift = cfg->shift;
  // i < MAX_TXFM_STAGE_NUM will mute above array bounds warning
  for (int i = 0; i < cfg->stage_num_col && i < MAX_TXFM_STAGE_NUM; ++i) {
    stage_range_col[i] = cfg->stage_range_col[i] + shift[0] + bd + 1;
  }

  // i < MAX_TXFM_STAGE_NUM will mute above array bounds warning
  for (int i = 0; i < cfg->stage_num_row && i < MAX_TXFM_STAGE_NUM; ++i) {
    stage_range_row[i] = cfg->stage_range_row[i] + shift[0] + shift[1] + bd + 1;
  }
}

#if CONFIG_MODE_DEP_INTRA_TX && CONFIG_MODE_DEP_NONSEP_INTRA_TX
static INLINE void fwd_nonsep_txfm2d(const int16_t *input, int32_t *output,
                                     const int stride, int32_t *buf,
                                     const int32_t *nstx_mtx,
                                     const TX_SIZE tx_size) {
  int ud_flip = 0, lr_flip = 0;
  const int tx_stride = tx_size_wide[tx_size] * tx_size_high[tx_size];

  // column/row indices in pixel (p) or transform (t) domains
  int cp, rp, ct, rt, kp, kt, l;
  int txw = tx_size_wide[tx_size], txh = tx_size_high[tx_size];

  for (rt = 0; rt < txh; ++rt)
    for (ct = 0; ct < txw; ++ct) buf[rt * txw + ct] = 0;

  // 2D transform
  for (rt = 0; rt < txh; ++rt) {
    for (ct = 0; ct < txw; ++ct) {
      l = rt * txw + ct;
      for (rp = 0; rp < txh; ++rp) {
        for (cp = 0; cp < txw; ++cp) {
          kp = rp * stride + cp;
          kt = idx_flip(txw, txh, rp, cp, ud_flip, lr_flip);
          // Values of buf[l] are transform coefficients * 2^(8-1)
          // Bit depth of buf[l] = 8 + 1 (nstx) + 9 (input) + 6 (64 coeffs) - 1
          //                     = 23
          // (8 for magnitude, and 1 for sign of tx. matrix's elements)
          // Max possible bit depth = 9 + 9 + 6 - 1 = 23
          buf[l] += round_shift(nstx_mtx[l * tx_stride + kt] * input[kp], 1);
        }
      }
    }
  }

  for (ct = 0; ct < txw; ++ct) {
    for (rt = 0; rt < txh; ++rt) {
      l = rt * txw + ct;
      // Values of output[l] are transform coefficients * 2^3
      // Max possible bit depth = 8 + 15 - (8 - 4) = 19
      output[l] = round_shift(buf[l], 4);
    }
  }
}

#if CONFIG_MODE_DEP_NONSEP_SEC_INTRA_TX
static INLINE void fwd_nonsep_secondary_txfm2d(int32_t *input, int32_t *buf,
                                               const int32_t *nsst_mtx,
                                               const TX_SIZE tx_size) {
  const int txw = tx_size_wide[tx_size], txh = tx_size_high[tx_size];
  const int txwh = txw / 2, txhh = txh / 2;
  const int tx_stride = txwh * txhh;
  int cp, rp, ct, rt, k, l;

#if MDTX_DEBUG && 0
  fprintf(stderr, "FWD-NSST: before NSST\n");
  for (rt = 0; rt < txh; ++rt) {
    for (ct = 0; ct < txw; ++ct) {
      fprintf(stderr, "%3d ", input[rt * txw + ct]);
    }
    fprintf(stderr, "\n");
  }
#endif

  for (rt = 0; rt < txh; ++rt)
    for (ct = 0; ct < txw; ++ct) buf[rt * txw + ct] = 0;

  // Apply a 2D non-separable transform on the 1/4 block (only the 1/4
  // top-left part of txfm_buf will be used). Note: stride in input[] and
  // txfm_buf[] should be txw.
  for (rt = 0; rt < txhh; ++rt) {
    for (ct = 0; ct < txwh; ++ct) {
      l = rt * txwh + ct;
      for (rp = 0; rp < txhh; ++rp) {
        for (cp = 0; cp < txwh; ++cp) {
          k = rp * txwh + cp;
          // Values of buf[l] are transform coefficients * 2^(8-1)
          // Bit depth of buf[l] = 8 + 1 (nsst) + 9 (input) + 6 (64 coeffs) - 1
          //                     = 23
          // (8 for magnitude, and 1 for sign of tx. matrix's elements)
          // Max possible bit depth = 9 + 9 + 6 - 1 = 23
          buf[rt * txw + ct] += round_shift(
              nsst_mtx[l * tx_stride + k] * input[rp * txw + cp], 1);
#if 0
          fprintf(stderr, "(%d,%d,%d)[%d,%d,%d]", l, tx_stride, k,
                  nsst_mtx[l * tx_stride + k], input[rp * txw + cp],
                  buf[rt * txw + ct]);
#endif
        }
#if 0
        fprintf(stderr, "\n");
#endif
      }
    }
  }

  for (ct = 0; ct < txwh; ++ct)
    for (rt = 0; rt < txhh; ++rt)
      input[rt * txw + ct] = round_shift(buf[rt * txw + ct], 7);

#if MDTX_DEBUG
  fprintf(stderr, "FWD-NSST: after NSST\n");
  for (rt = 0; rt < txh; ++rt) {
    for (ct = 0; ct < txw; ++ct) {
      fprintf(stderr, "%3d ", input[rt * txw + ct]);
    }
    fprintf(stderr, "\n");
  }
#endif
}
#endif  // CONFIG_MODE_DEP_NONSEP_SEC_INTRA_TX
#endif  // CONFIG_MODE_DEP_INTRA_TX &&
        // CONFIG_MODE_DEP_NONSEP_INTRA_TX

static INLINE void fwd_txfm2d_c(const int16_t *input, int32_t *output,
                                const int stride, const TXFM_2D_FLIP_CFG *cfg,
                                int32_t *buf, int bd) {
#if CONFIG_MODE_DEP_INTRA_TX && CONFIG_MODE_DEP_NONSEP_INTRA_TX
  if (cfg->nstx_mtx_ptr
#if CONFIG_MODE_DEP_NONSEP_SEC_INTRA_TX
      && cfg->tx_size == TX_4X4
#endif  // CONFIG_MODE_DEP_NONSEP_SEC_INTRA_TX
  ) {
    // 4x4 non-separable transform
    fwd_nonsep_txfm2d(input, output, stride, buf, cfg->nstx_mtx_ptr,
                      cfg->tx_size);
    return;
  }
#endif  // CONFIG_MODE_DEP_INTRA_TX &&
        // CONFIG_MODE_DEP_NONSEP_INTRA_TX
  int c, r;
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
  int8_t stage_range_col[MAX_TXFM_STAGE_NUM + 1];
  int8_t stage_range_row[MAX_TXFM_STAGE_NUM + 1];
  assert(cfg->stage_num_col <= MAX_TXFM_STAGE_NUM);
  assert(cfg->stage_num_row <= MAX_TXFM_STAGE_NUM);
  av1_gen_fwd_stage_range(stage_range_col, stage_range_row, cfg, bd);

  const int8_t cos_bit_col = cfg->cos_bit_col;
  const int8_t cos_bit_row = cfg->cos_bit_row;
  const TxfmFunc txfm_func_col =
      fwd_txfm_type_to_func(cfg->mode, cfg->txfm_type_col);
  const TxfmFunc txfm_func_row =
      fwd_txfm_type_to_func(cfg->mode, cfg->txfm_type_row);
  stage_range_col[MAX_TXFM_STAGE_NUM] = (int)cfg->mode;
  stage_range_row[MAX_TXFM_STAGE_NUM] = (int)cfg->mode;

  // use output buffer as temp buffer
  int32_t *temp_in = output;
  int32_t *temp_out = output + txfm_size_row;

#if CONFIG_MODE_DEP_INTRA_TX && CONFIG_MODE_DEP_NONSEP_INTRA_TX && MDTX_DEBUG
  // debug
  if (txfm_size_col <= 8 && txfm_size_row <= 8 && cfg->nstx_mtx_ptr) {
    fprintf(stderr, "FWD: input block, %dx%d, mode %d, ctx %d, rtx %d\n",
            txfm_size_col, txfm_size_row, (int)cfg->mode, cfg->txfm_type_col,
            cfg->txfm_type_row);
    for (r = 0; r < txfm_size_row; ++r) {
      for (c = 0; c < txfm_size_col; ++c) {
        fprintf(stderr, "%3d ", input[r * stride + c]);
      }
      fprintf(stderr, "\n");
    }
  }
#endif  // CONFIG_MODE_DEP_INTRA_TX &&
        // CONFIG_MODE_DEP_NONSEP_INTRA_TX && MDTX_DEBUG

  // Columns
  for (c = 0; c < txfm_size_col; ++c) {
    if (cfg->ud_flip == 0) {
      for (r = 0; r < txfm_size_row; ++r) temp_in[r] = input[r * stride + c];
    } else {
      for (r = 0; r < txfm_size_row; ++r)
        // flip upside down
        temp_in[r] = input[(txfm_size_row - r - 1) * stride + c];
    }
    av1_round_shift_array(temp_in, txfm_size_row, -shift[0]);
    txfm_func_col(temp_in, temp_out, cos_bit_col, stage_range_col);
    av1_round_shift_array(temp_out, txfm_size_row, -shift[1]);
    if (cfg->lr_flip == 0) {
      for (r = 0; r < txfm_size_row; ++r)
        buf[r * txfm_size_col + c] = temp_out[r];
    } else {
      for (r = 0; r < txfm_size_row; ++r)
        // flip from left to right
        buf[r * txfm_size_col + (txfm_size_col - c - 1)] = temp_out[r];
    }
  }

  // Rows
  for (r = 0; r < txfm_size_row; ++r) {
    txfm_func_row(buf + r * txfm_size_col, output + r * txfm_size_col,
                  cos_bit_row, stage_range_row);
    av1_round_shift_array(output + r * txfm_size_col, txfm_size_col, -shift[2]);
    if ((abs(rect_type) % 2) == 1) {
      // Multiply everything by Sqrt2 if the transform is rectangular and the
      // size difference is a factor of 2 or 8.
      for (c = 0; c < txfm_size_col; ++c) {
        output[r * txfm_size_col + c] = round_shift(
            (int64_t)output[r * txfm_size_col + c] * NewSqrt2, NewSqrt2Bits);
      }
    }
  }

#if CONFIG_MODE_DEP_INTRA_TX && CONFIG_MODE_DEP_NONSEP_INTRA_TX && \
    CONFIG_MODE_DEP_NONSEP_SEC_INTRA_TX
#if MDTX_DEBUG
  if (txfm_size_col <= 8 && txfm_size_row <= 8 && cfg->nstx_mtx_ptr) {
    fprintf(stderr, "FWD: output block\n");
    for (r = 0; r < txfm_size_row; ++r) {
      for (c = 0; c < txfm_size_col; ++c) {
        fprintf(stderr, "%3d ", output[r * txfm_size_col + c]);
      }
      fprintf(stderr, "\n");
    }
  }
#endif  // MDTX_DEBUG
  // Apply non-separable secondary transform after separable transforms
  if (cfg->nstx_mtx_ptr)
    fwd_nonsep_secondary_txfm2d(output, buf, cfg->nstx_mtx_ptr, cfg->tx_size);
#endif  // CONFIG_MODE_DEP_INTRA_TX &&
        // CONFIG_MODE_DEP_NONSEP_INTRA_TX &&
        // CONFIG_MODE_DEP_NONSEP_SEC_INTRA_TX
}

void av1_fwd_txfm2d_4x8_c(const int16_t *input, int32_t *output, int stride,
                          TX_TYPE tx_type, PREDICTION_MODE mode, int bd) {
  DECLARE_ALIGNED(32, int32_t, txfm_buf[4 * 8]);
  TXFM_2D_FLIP_CFG cfg;
  av1_get_fwd_txfm_cfg(tx_type, TX_4X8, mode, &cfg);
  fwd_txfm2d_c(input, output, stride, &cfg, txfm_buf, bd);
}

void av1_fwd_txfm2d_8x4_c(const int16_t *input, int32_t *output, int stride,
                          TX_TYPE tx_type, PREDICTION_MODE mode, int bd) {
  int32_t txfm_buf[8 * 4];
  TXFM_2D_FLIP_CFG cfg;
  av1_get_fwd_txfm_cfg(tx_type, TX_8X4, mode, &cfg);
  fwd_txfm2d_c(input, output, stride, &cfg, txfm_buf, bd);
}

void av1_fwd_txfm2d_8x16_c(const int16_t *input, int32_t *output, int stride,
                           TX_TYPE tx_type, PREDICTION_MODE mode, int bd) {
  DECLARE_ALIGNED(32, int32_t, txfm_buf[8 * 16]);
  TXFM_2D_FLIP_CFG cfg;
  av1_get_fwd_txfm_cfg(tx_type, TX_8X16, mode, &cfg);
  fwd_txfm2d_c(input, output, stride, &cfg, txfm_buf, bd);
}

void av1_fwd_txfm2d_16x8_c(const int16_t *input, int32_t *output, int stride,
                           TX_TYPE tx_type, PREDICTION_MODE mode, int bd) {
  int32_t txfm_buf[16 * 8];
  TXFM_2D_FLIP_CFG cfg;
  av1_get_fwd_txfm_cfg(tx_type, TX_16X8, mode, &cfg);
  fwd_txfm2d_c(input, output, stride, &cfg, txfm_buf, bd);
}

void av1_fwd_txfm2d_16x32_c(const int16_t *input, int32_t *output, int stride,
                            TX_TYPE tx_type, PREDICTION_MODE mode, int bd) {
  DECLARE_ALIGNED(32, int32_t, txfm_buf[16 * 32]);
  TXFM_2D_FLIP_CFG cfg;
  av1_get_fwd_txfm_cfg(tx_type, TX_16X32, mode, &cfg);
  fwd_txfm2d_c(input, output, stride, &cfg, txfm_buf, bd);
}

void av1_fwd_txfm2d_32x16_c(const int16_t *input, int32_t *output, int stride,
                            TX_TYPE tx_type, PREDICTION_MODE mode, int bd) {
  int32_t txfm_buf[32 * 16];
  TXFM_2D_FLIP_CFG cfg;
  av1_get_fwd_txfm_cfg(tx_type, TX_32X16, mode, &cfg);
  fwd_txfm2d_c(input, output, stride, &cfg, txfm_buf, bd);
}

void av1_fwd_txfm2d_4x16_c(const int16_t *input, int32_t *output, int stride,
                           TX_TYPE tx_type, PREDICTION_MODE mode, int bd) {
  DECLARE_ALIGNED(32, int32_t, txfm_buf[4 * 16]);
  TXFM_2D_FLIP_CFG cfg;
  av1_get_fwd_txfm_cfg(tx_type, TX_4X16, mode, &cfg);
  fwd_txfm2d_c(input, output, stride, &cfg, txfm_buf, bd);
}

void av1_fwd_txfm2d_16x4_c(const int16_t *input, int32_t *output, int stride,
                           TX_TYPE tx_type, PREDICTION_MODE mode, int bd) {
  int32_t txfm_buf[16 * 4];
  TXFM_2D_FLIP_CFG cfg;
  av1_get_fwd_txfm_cfg(tx_type, TX_16X4, mode, &cfg);
  fwd_txfm2d_c(input, output, stride, &cfg, txfm_buf, bd);
}

void av1_fwd_txfm2d_8x32_c(const int16_t *input, int32_t *output, int stride,
                           TX_TYPE tx_type, PREDICTION_MODE mode, int bd) {
  DECLARE_ALIGNED(32, int32_t, txfm_buf[32 * 8]);
  TXFM_2D_FLIP_CFG cfg;
  av1_get_fwd_txfm_cfg(tx_type, TX_8X32, mode, &cfg);
  fwd_txfm2d_c(input, output, stride, &cfg, txfm_buf, bd);
}

void av1_fwd_txfm2d_32x8_c(const int16_t *input, int32_t *output, int stride,
                           TX_TYPE tx_type, PREDICTION_MODE mode, int bd) {
  int32_t txfm_buf[32 * 8];
  TXFM_2D_FLIP_CFG cfg;
  av1_get_fwd_txfm_cfg(tx_type, TX_32X8, mode, &cfg);
  fwd_txfm2d_c(input, output, stride, &cfg, txfm_buf, bd);
}

void av1_fwd_txfm2d_4x4_c(const int16_t *input, int32_t *output, int stride,
                          TX_TYPE tx_type, PREDICTION_MODE mode, int bd) {
  int32_t txfm_buf[4 * 4];
  TXFM_2D_FLIP_CFG cfg;
  av1_get_fwd_txfm_cfg(tx_type, TX_4X4, mode, &cfg);
  fwd_txfm2d_c(input, output, stride, &cfg, txfm_buf, bd);
}

void av1_fwd_txfm2d_8x8_c(const int16_t *input, int32_t *output, int stride,
                          TX_TYPE tx_type, PREDICTION_MODE mode, int bd) {
  int32_t txfm_buf[8 * 8];
  TXFM_2D_FLIP_CFG cfg;
  av1_get_fwd_txfm_cfg(tx_type, TX_8X8, mode, &cfg);
  fwd_txfm2d_c(input, output, stride, &cfg, txfm_buf, bd);
}

void av1_fwd_txfm2d_16x16_c(const int16_t *input, int32_t *output, int stride,
                            TX_TYPE tx_type, PREDICTION_MODE mode, int bd) {
  int32_t txfm_buf[16 * 16];
  TXFM_2D_FLIP_CFG cfg;
  av1_get_fwd_txfm_cfg(tx_type, TX_16X16, mode, &cfg);
  fwd_txfm2d_c(input, output, stride, &cfg, txfm_buf, bd);
}

void av1_fwd_txfm2d_32x32_c(const int16_t *input, int32_t *output, int stride,
                            TX_TYPE tx_type, PREDICTION_MODE mode, int bd) {
  int32_t txfm_buf[32 * 32];
  TXFM_2D_FLIP_CFG cfg;
  av1_get_fwd_txfm_cfg(tx_type, TX_32X32, mode, &cfg);
  fwd_txfm2d_c(input, output, stride, &cfg, txfm_buf, bd);
}

#if CONFIG_NEW_TX64X64
void av1_fwd_txfm2d_64x64_c(const int16_t *input, int32_t *output, int stride,
                            TX_TYPE tx_type, PREDICTION_MODE mode, int bd) {
  // Downsample to 32x32
  DECLARE_ALIGNED(16, int16_t, input_32[32 * 32]);
#if USE_SIMPLE_DOWNSCALE
  for (int r = 0; r < 32; ++r) {
    for (int c = 0; c < 32; ++c) {
      const int16_t *const in = &input[2 * r * stride + 2 * c];
      input_32[r * 32 + c] = ROUND_POWER_OF_TWO_SIGNED(
          in[0] + in[1] + in[stride] + in[stride + 1], 2);
    }
  }
#else
  av1_signed_down2(input, 64, 64, stride, input_32, 32, 1, 1, bd);
#endif  // USE_SIMPLE_DOWNSCALE

  // Initialize output to all-zero.
  memset(output, 0, 64 * 64 * sizeof(*output));

  // Perform 32x32 transform and output to the top-left quadrant.
  av1_fwd_txfm2d_32x32(input_32, output, 32, tx_type, mode, bd);
}

void av1_fwd_txfm2d_32x64_c(const int16_t *input, int32_t *output, int stride,
                            TX_TYPE tx_type, PREDICTION_MODE mode, int bd) {
  // Downsample to 32x32
  DECLARE_ALIGNED(16, int16_t, input_32[32 * 32]);
#if USE_SIMPLE_DOWNSCALE
  for (int r = 0; r < 32; ++r) {
    for (int c = 0; c < 32; ++c) {
      const int16_t *const in = &input[2 * r * stride + c];
      const int32_t avg = ROUND_POWER_OF_TWO_SIGNED(in[0] + in[stride], 1);
      input_32[r * 32 + c] = avg;
    }
  }
#else
  av1_signed_down2(input, 64, 32, stride, input_32, 32, 1, 0, bd);
#endif  // USE_SIMPLE_DOWNSCALE

  // Initialize output to all-zero.
  memset(output, 0, 32 * 64 * sizeof(*output));

  // Perform 32x32 transform and output to the top-left quadrant.
  av1_fwd_txfm2d_32x32(input_32, output, 32, tx_type, mode, bd);
  for (int r = 0; r < 32; ++r) {
    for (int c = 0; c < 32; ++c) {
      output[r * 32 + c] =
          round_shift((int64_t)output[r * 32 + c] * NewInvSqrt2, NewSqrt2Bits);
    }
  }
}

void av1_fwd_txfm2d_64x32_c(const int16_t *input, int32_t *output, int stride,
                            TX_TYPE tx_type, PREDICTION_MODE mode, int bd) {
  // Downsample to 32x32
  DECLARE_ALIGNED(16, int16_t, input_32[32 * 32]);
#if USE_SIMPLE_DOWNSCALE
  for (int r = 0; r < 32; ++r) {
    for (int c = 0; c < 32; ++c) {
      const int16_t *const in = &input[r * stride + 2 * c];
      const int32_t avg = ROUND_POWER_OF_TWO_SIGNED(in[0] + in[1], 1);
      input_32[r * 32 + c] = avg;
    }
  }
#else
  av1_signed_down2(input, 32, 64, stride, input_32, 32, 0, 1, bd);
#endif  // USE_SIMPLE_DOWNSCALE

  // Initialize output to all-zero.
  memset(output, 0, 64 * 32 * sizeof(*output));

  // Perform 32x32 transform and output to the top-left quadrant.
  av1_fwd_txfm2d_32x32(input_32, output, 32, tx_type, mode, bd);
  for (int r = 0; r < 32; ++r) {
    for (int c = 0; c < 32; ++c) {
      output[r * 32 + c] =
          round_shift((int64_t)output[r * 32 + c] * NewInvSqrt2, NewSqrt2Bits);
    }
  }
}

void av1_fwd_txfm2d_16x64_c(const int16_t *input, int32_t *output, int stride,
                            TX_TYPE tx_type, PREDICTION_MODE mode, int bd) {
  // Downsample to 16x32
  DECLARE_ALIGNED(16, int16_t, input_32[16 * 32]);
#if USE_SIMPLE_DOWNSCALE
  for (int r = 0; r < 32; ++r) {
    for (int c = 0; c < 16; ++c) {
      const int16_t *const in = &input[2 * r * stride + c];
      const int32_t avg = ROUND_POWER_OF_TWO_SIGNED(in[0] + in[stride], 1);
      input_32[r * 16 + c] = avg;
    }
  }
#else
  av1_signed_down2(input, 64, 16, stride, input_32, 16, 1, 0, bd);
#endif  // USE_SIMPLE_DOWNSCALE

  // Initialize output to all-zero.
  memset(output, 0, 16 * 64 * sizeof(*output));

  // Perform 16x32 transform and output to the top-left quadrant.
  av1_fwd_txfm2d_16x32(input_32, output, 16, tx_type, mode, bd);
  for (int r = 0; r < 32; ++r) {
    for (int c = 0; c < 16; ++c) {
      output[r * 16 + c] =
          round_shift((int64_t)output[r * 16 + c] * NewSqrt2, NewSqrt2Bits);
    }
  }
}

void av1_fwd_txfm2d_64x16_c(const int16_t *input, int32_t *output, int stride,
                            TX_TYPE tx_type, PREDICTION_MODE mode, int bd) {
  // Downsample to 32x16
  DECLARE_ALIGNED(16, int16_t, input_32[32 * 16]);
#if USE_SIMPLE_DOWNSCALE
  for (int r = 0; r < 16; ++r) {
    for (int c = 0; c < 32; ++c) {
      const int16_t *const in = &input[r * stride + 2 * c];
      const int32_t avg = ROUND_POWER_OF_TWO_SIGNED(in[0] + in[1], 1);
      input_32[r * 32 + c] = avg;
    }
  }
#else
  av1_signed_down2(input, 16, 64, stride, input_32, 32, 0, 1, bd);
#endif  // USE_SIMPLE_DOWNSCALE

  // Initialize output to all-zero.
  memset(output, 0, 64 * 16 * sizeof(*output));

  // Perform 32x16 transform and output to the top-left quadrant.
  av1_fwd_txfm2d_32x16(input_32, output, 32, tx_type, mode, bd);
  for (int r = 0; r < 16; ++r) {
    for (int c = 0; c < 32; ++c) {
      output[r * 32 + c] =
          round_shift((int64_t)output[r * 32 + c] * NewSqrt2, NewSqrt2Bits);
    }
  }
}

#else
void av1_fwd_txfm2d_64x64_c(const int16_t *input, int32_t *output, int stride,
                            TX_TYPE tx_type, PREDICTION_MODE mode, int bd) {
  int32_t txfm_buf[64 * 64];
  TXFM_2D_FLIP_CFG cfg;
  av1_get_fwd_txfm_cfg(tx_type, TX_64X64, mode, &cfg);
  fwd_txfm2d_c(input, output, stride, &cfg, txfm_buf, bd);

  // Zero out top-right 32x32 area.
  for (int row = 0; row < 32; ++row) {
    memset(output + row * 64 + 32, 0, 32 * sizeof(*output));
  }
  // Zero out the bottom 64x32 area.
  memset(output + 32 * 64, 0, 32 * 64 * sizeof(*output));
  // Re-pack non-zero coeffs in the first 32x32 indices.
  for (int row = 1; row < 32; ++row) {
    memcpy(output + row * 32, output + row * 64, 32 * sizeof(*output));
  }
}

void av1_fwd_txfm2d_32x64_c(const int16_t *input, int32_t *output, int stride,
                            TX_TYPE tx_type, PREDICTION_MODE mode, int bd) {
  DECLARE_ALIGNED(32, int32_t, txfm_buf[32 * 64]);
  TXFM_2D_FLIP_CFG cfg;
  av1_get_fwd_txfm_cfg(tx_type, TX_32X64, mode, &cfg);
  fwd_txfm2d_c(input, output, stride, &cfg, txfm_buf, bd);
  // Zero out the bottom 32x32 area.
  memset(output + 32 * 32, 0, 32 * 32 * sizeof(*output));
  // Note: no repacking needed here.
}

void av1_fwd_txfm2d_64x32_c(const int16_t *input, int32_t *output, int stride,
                            TX_TYPE tx_type, PREDICTION_MODE mode, int bd) {
  int32_t txfm_buf[64 * 32];
  TXFM_2D_FLIP_CFG cfg;
  av1_get_fwd_txfm_cfg(tx_type, TX_64X32, mode, &cfg);
  fwd_txfm2d_c(input, output, stride, &cfg, txfm_buf, bd);

  // Zero out right 32x32 area.
  for (int row = 0; row < 32; ++row) {
    memset(output + row * 64 + 32, 0, 32 * sizeof(*output));
  }
  // Re-pack non-zero coeffs in the first 32x32 indices.
  for (int row = 1; row < 32; ++row) {
    memcpy(output + row * 32, output + row * 64, 32 * sizeof(*output));
  }
}

void av1_fwd_txfm2d_16x64_c(const int16_t *input, int32_t *output, int stride,
                            TX_TYPE tx_type, PREDICTION_MODE mode, int bd) {
  DECLARE_ALIGNED(32, int32_t, txfm_buf[64 * 16]);
  TXFM_2D_FLIP_CFG cfg;
  av1_get_fwd_txfm_cfg(tx_type, TX_16X64, mode, &cfg);
  fwd_txfm2d_c(input, output, stride, &cfg, txfm_buf, bd);
  // Zero out the bottom 16x32 area.
  memset(output + 16 * 32, 0, 16 * 32 * sizeof(*output));
  // Note: no repacking needed here.
}

void av1_fwd_txfm2d_64x16_c(const int16_t *input, int32_t *output, int stride,
                            TX_TYPE tx_type, PREDICTION_MODE mode, int bd) {
  int32_t txfm_buf[64 * 16];
  TXFM_2D_FLIP_CFG cfg;
  av1_get_fwd_txfm_cfg(tx_type, TX_64X16, mode, &cfg);
  fwd_txfm2d_c(input, output, stride, &cfg, txfm_buf, bd);
  // Zero out right 32x16 area.
  for (int row = 0; row < 16; ++row) {
    memset(output + row * 64 + 32, 0, 32 * sizeof(*output));
  }
  // Re-pack non-zero coeffs in the first 32x16 indices.
  for (int row = 1; row < 16; ++row) {
    memcpy(output + row * 32, output + row * 64, 32 * sizeof(*output));
  }
}

#endif  // CONFIG_NEW_TX64X64

#if CONFIG_FLEX_PARTITION
void av1_fwd_txfm2d_4x32_c(const int16_t *input, int32_t *output, int stride,
                           TX_TYPE tx_type, PREDICTION_MODE mode, int bd) {
  DECLARE_ALIGNED(32, int32_t, txfm_buf[32 * 4]);
  TXFM_2D_FLIP_CFG cfg;
  av1_get_fwd_txfm_cfg(tx_type, TX_4X32, mode, &cfg);
  fwd_txfm2d_c(input, output, stride, &cfg, txfm_buf, bd);
}

void av1_fwd_txfm2d_32x4_c(const int16_t *input, int32_t *output, int stride,
                           TX_TYPE tx_type, PREDICTION_MODE mode, int bd) {
  int32_t txfm_buf[32 * 4];
  TXFM_2D_FLIP_CFG cfg;
  av1_get_fwd_txfm_cfg(tx_type, TX_32X4, mode, &cfg);
  fwd_txfm2d_c(input, output, stride, &cfg, txfm_buf, bd);
}

void av1_fwd_txfm2d_8x64_c(const int16_t *input, int32_t *output, int stride,
                           TX_TYPE tx_type, PREDICTION_MODE mode, int bd) {
  DECLARE_ALIGNED(32, int32_t, txfm_buf[64 * 8]);
  TXFM_2D_FLIP_CFG cfg;
  av1_get_fwd_txfm_cfg(tx_type, TX_8X64, mode, &cfg);
  fwd_txfm2d_c(input, output, stride, &cfg, txfm_buf, bd);
  // Zero out the bottom 8x32 area.
  memset(output + 8 * 32, 0, 8 * 32 * sizeof(*output));
  // Note: no repacking needed here.
}

void av1_fwd_txfm2d_64x8_c(const int16_t *input, int32_t *output, int stride,
                           TX_TYPE tx_type, PREDICTION_MODE mode, int bd) {
  int32_t txfm_buf[64 * 8];
  TXFM_2D_FLIP_CFG cfg;
  av1_get_fwd_txfm_cfg(tx_type, TX_64X8, mode, &cfg);
  fwd_txfm2d_c(input, output, stride, &cfg, txfm_buf, bd);
  // Zero out right 32x8 area.
  for (int row = 0; row < 8; ++row) {
    memset(output + row * 64 + 32, 0, 32 * sizeof(*output));
  }
  // Re-pack non-zero coeffs in the first 32x16 indices.
  for (int row = 1; row < 8; ++row) {
    memcpy(output + row * 32, output + row * 64, 32 * sizeof(*output));
  }
}

void av1_fwd_txfm2d_4x64_c(const int16_t *input, int32_t *output, int stride,
                           TX_TYPE tx_type, PREDICTION_MODE mode, int bd) {
  DECLARE_ALIGNED(32, int32_t, txfm_buf[64 * 4]);
  TXFM_2D_FLIP_CFG cfg;
  av1_get_fwd_txfm_cfg(tx_type, TX_4X64, mode, &cfg);
  fwd_txfm2d_c(input, output, stride, &cfg, txfm_buf, bd);
  // Zero out the bottom 4x32 area.
  memset(output + 4 * 32, 0, 4 * 32 * sizeof(*output));
  // Note: no repacking needed here.
}

void av1_fwd_txfm2d_64x4_c(const int16_t *input, int32_t *output, int stride,
                           TX_TYPE tx_type, PREDICTION_MODE mode, int bd) {
  int32_t txfm_buf[64 * 4];
  TXFM_2D_FLIP_CFG cfg;
  av1_get_fwd_txfm_cfg(tx_type, TX_64X4, mode, &cfg);
  fwd_txfm2d_c(input, output, stride, &cfg, txfm_buf, bd);
  // Zero out right 32x4 area.
  for (int row = 0; row < 4; ++row) {
    memset(output + row * 64 + 32, 0, 32 * sizeof(*output));
  }
  // Re-pack non-zero coeffs in the first 32x4 indices.
  for (int row = 1; row < 4; ++row) {
    memcpy(output + row * 32, output + row * 64, 32 * sizeof(*output));
  }
}
#endif  // CONFIG_FLEX_PARTITION

static const int8_t fwd_shift_4x4[3] = { 2, 0, 0 };
static const int8_t fwd_shift_8x8[3] = { 2, -1, 0 };
static const int8_t fwd_shift_16x16[3] = { 2, -2, 0 };
static const int8_t fwd_shift_32x32[3] = { 2, -4, 0 };
static const int8_t fwd_shift_64x64[3] = { 0, -2, -2 };
static const int8_t fwd_shift_4x8[3] = { 2, -1, 0 };
static const int8_t fwd_shift_8x4[3] = { 2, -1, 0 };
static const int8_t fwd_shift_8x16[3] = { 2, -2, 0 };
static const int8_t fwd_shift_16x8[3] = { 2, -2, 0 };
static const int8_t fwd_shift_16x32[3] = { 2, -4, 0 };
static const int8_t fwd_shift_32x16[3] = { 2, -4, 0 };
static const int8_t fwd_shift_32x64[3] = { 0, -2, -2 };
static const int8_t fwd_shift_64x32[3] = { 2, -4, -2 };
static const int8_t fwd_shift_4x16[3] = { 2, -1, 0 };
static const int8_t fwd_shift_16x4[3] = { 2, -1, 0 };
static const int8_t fwd_shift_8x32[3] = { 2, -2, 0 };
static const int8_t fwd_shift_32x8[3] = { 2, -2, 0 };
static const int8_t fwd_shift_16x64[3] = { 0, -2, 0 };
static const int8_t fwd_shift_64x16[3] = { 2, -4, 0 };
#if CONFIG_FLEX_PARTITION
static const int8_t fwd_shift_4x32[3] = { 2, -2, 0 };
static const int8_t fwd_shift_32x4[3] = { 2, -2, 0 };
static const int8_t fwd_shift_8x64[3] = { 0, -2, 0 };
static const int8_t fwd_shift_64x8[3] = { 2, -4, 0 };
static const int8_t fwd_shift_4x64[3] = { 0, 0, 0 };
static const int8_t fwd_shift_64x4[3] = { 2, -2, 0 };
#endif  // CONFIG_FLEX_PARTITION

const int8_t *av1_fwd_txfm_shift_ls[TX_SIZES_ALL] = {
  fwd_shift_4x4,   fwd_shift_8x8,   fwd_shift_16x16, fwd_shift_32x32,
  fwd_shift_64x64, fwd_shift_4x8,   fwd_shift_8x4,   fwd_shift_8x16,
  fwd_shift_16x8,  fwd_shift_16x32, fwd_shift_32x16, fwd_shift_32x64,
  fwd_shift_64x32, fwd_shift_4x16,  fwd_shift_16x4,  fwd_shift_8x32,
  fwd_shift_32x8,  fwd_shift_16x64, fwd_shift_64x16,
#if CONFIG_FLEX_PARTITION
  fwd_shift_4x32,  fwd_shift_32x4,  fwd_shift_8x64,  fwd_shift_64x8,
  fwd_shift_4x64,  fwd_shift_64x4,
#endif  // CONFIG_FLEX_PARTITION
};

const int8_t av1_fwd_cos_bit_col[MAX_TXWH_IDX /*txw_idx*/]
                                [MAX_TXWH_IDX /*txh_idx*/] = {
                                  { 13, 13, 13, 12, 13 },
                                  { 13, 13, 13, 12, 13 },
                                  { 13, 13, 13, 12, 13 },
                                  { 13, 13, 13, 12, 13 },
                                  { 13, 13, 13, 12, 13 }
                                };

const int8_t av1_fwd_cos_bit_row[MAX_TXWH_IDX /*txw_idx*/]
                                [MAX_TXWH_IDX /*txh_idx*/] = {
                                  { 13, 13, 12, 12, 12 },
                                  { 13, 13, 13, 12, 12 },
                                  { 13, 13, 12, 13, 12 },
                                  { 12, 12, 13, 12, 11 },
                                  { 12, 12, 12, 11, 10 }
                                };

static const int8_t fdct4_range_mult2[4] = { 0, 2, 3, 3 };
static const int8_t fdct8_range_mult2[6] = { 0, 2, 4, 5, 5, 5 };
static const int8_t fdct16_range_mult2[8] = { 0, 2, 4, 6, 7, 7, 7, 7 };
static const int8_t fdct32_range_mult2[10] = { 0, 2, 4, 6, 8, 9, 9, 9, 9, 9 };
static const int8_t fdct64_range_mult2[12] = { 0,  2,  4,  6,  8,  10,
                                               11, 11, 11, 11, 11, 11 };

static const int8_t fadst4_range_mult2[7] = { 0, 2, 4, 3, 3, 3, 3 };
static const int8_t fadst8_range_mult2[8] = { 0, 0, 1, 3, 3, 5, 5, 5 };
static const int8_t fadst16_range_mult2[10] = { 0, 0, 1, 3, 3, 5, 5, 7, 7, 7 };
#if CONFIG_DST7_32x32
static const int8_t fadst32_range_mult2[1] = { 9 };
#endif
static const int8_t fidtx4_range_mult2[1] = { 1 };
static const int8_t fidtx8_range_mult2[1] = { 2 };
static const int8_t fidtx16_range_mult2[1] = { 3 };
static const int8_t fidtx32_range_mult2[1] = { 4 };

#if 0
const int8_t fwd_idtx_range_row[MAX_TXWH_IDX /*txw_idx*/]
                               [MAX_TXWH_IDX /*txh_idx*/] = { { 2, 4, 5, 0, 0 },
                                                              { 3, 4, 5, 6, 0 },
                                                              { 4, 5, 6, 7, 8 },
                                                              { 0, 5, 6, 7, 8 },
                                                              { 0, 0, 7, 8,
                                                                9 } };
#endif

static const int8_t *fwd_txfm_range_mult2_list[TXFM_TYPES] = {
  fdct4_range_mult2,  fdct8_range_mult2,   fdct16_range_mult2,
  fdct32_range_mult2, fdct64_range_mult2,  fadst4_range_mult2,
  fadst8_range_mult2, fadst16_range_mult2, fidtx4_range_mult2,
  fidtx8_range_mult2, fidtx16_range_mult2, fidtx32_range_mult2,
#if CONFIG_MODE_DEP_INTRA_TX || CONFIG_MODE_DEP_INTER_TX
  fadst4_range_mult2, fadst8_range_mult2,  fadst16_range_mult2,
#endif
#if CONFIG_DST7_32x32
  fadst32_range_mult2
#endif
};

static INLINE void set_fwd_txfm_non_scale_range(TXFM_2D_FLIP_CFG *cfg) {
  av1_zero(cfg->stage_range_col);
  av1_zero(cfg->stage_range_row);

  const int8_t *range_mult2_col = fwd_txfm_range_mult2_list[cfg->txfm_type_col];
  if (cfg->txfm_type_col != TXFM_TYPE_INVALID) {
    int stage_num_col = cfg->stage_num_col;
    for (int i = 0; i < stage_num_col; ++i)
      cfg->stage_range_col[i] = (range_mult2_col[i] + 1) >> 1;
  }

  if (cfg->txfm_type_row != TXFM_TYPE_INVALID) {
    int stage_num_row = cfg->stage_num_row;
    const int8_t *range_mult2_row =
        fwd_txfm_range_mult2_list[cfg->txfm_type_row];
    for (int i = 0; i < stage_num_row; ++i) {
      cfg->stage_range_row[i] =
          (range_mult2_col[cfg->stage_num_col - 1] + range_mult2_row[i] + 1) >>
          1;
    }
  }
}

void av1_get_fwd_txfm_cfg(TX_TYPE tx_type, TX_SIZE tx_size,
                          PREDICTION_MODE mode, TXFM_2D_FLIP_CFG *cfg) {
  assert(cfg != NULL);
  cfg->tx_size = tx_size;
  set_flip_cfg(tx_type, cfg);
  const TX_TYPE_1D tx_type_1d_col = vtx_tab[tx_type];
  const TX_TYPE_1D tx_type_1d_row = htx_tab[tx_type];
  const int txw_idx = tx_size_wide_log2[tx_size] - tx_size_wide_log2[0];
  const int txh_idx = tx_size_high_log2[tx_size] - tx_size_high_log2[0];
  cfg->shift = av1_fwd_txfm_shift_ls[tx_size];
  cfg->cos_bit_col = av1_fwd_cos_bit_col[txw_idx][txh_idx];
  cfg->cos_bit_row = av1_fwd_cos_bit_row[txw_idx][txh_idx];
  cfg->txfm_type_col = av1_txfm_type_ls[txh_idx][tx_type_1d_col];
  cfg->txfm_type_row = av1_txfm_type_ls[txw_idx][tx_type_1d_row];
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
  set_fwd_txfm_non_scale_range(cfg);
}
