/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <assert.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"
#include "config/aom_dsp_rtcd.h"

#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/bitwriter.h"
#include "aom_dsp/quantize.h"
#include "aom_mem/aom_mem.h"
#include "aom_ports/mem.h"

#if CONFIG_BITSTREAM_DEBUG || CONFIG_MISMATCH_DEBUG
#include "aom_util/debug_util.h"
#endif  // CONFIG_BITSTREAM_DEBUG || CONFIG_MISMATCH_DEBUG

#include "av1/common/cfl.h"
#include "av1/common/idct.h"
#include "av1/common/reconinter.h"
#include "av1/common/reconintra.h"
#include "av1/common/scan.h"

#include "av1/encoder/av1_quantize.h"
#include "av1/encoder/encodemb.h"
#include "av1/encoder/hybrid_fwd_txfm.h"
#include "av1/encoder/txb_rdopt.h"
#include "av1/encoder/rd.h"
#include "av1/encoder/rdopt.h"

// Compute the average value of the wxh block.
static inline int16_t avg_wxh_block_c(int16_t *diff, ptrdiff_t diff_stride,
                                      int w, int h) {
  assert(w > 0 && h > 0);
  int32_t sum = 0;
  for (int row = 0; row < h; ++row) {
    for (int col = 0; col < w; ++col) {
      sum += diff[row * diff_stride + col];
    }
  }
  return (int16_t)DIVIDE_AND_ROUND_SIGNED(sum, w * h);
}

// Compute the row average value of the wxh block.
static inline void avg_wxh_block_horiz_c(int16_t *diff, ptrdiff_t diff_stride,
                                         int w, int h, int16_t *out) {
  assert(w > 0 && h > 0);
  for (int row = 0; row < h; ++row) {
    int32_t sum = 0;
    for (int col = 0; col < w; ++col) {
      sum += diff[row * diff_stride + col];
    }
    out[row] = (int16_t)DIVIDE_AND_ROUND_SIGNED(sum, w);
  }
}

// Compute the column average value of the wxh block.
static inline void avg_wxh_block_vert_c(int16_t *diff, ptrdiff_t diff_stride,
                                        int w, int h, int16_t *out) {
  assert(w > 0 && h > 0);
  for (int col = 0; col < w; ++col) {
    int32_t sum = 0;
    for (int row = 0; row < h; ++row) {
      sum += diff[row * diff_stride + col];
    }
    out[col] = (int16_t)DIVIDE_AND_ROUND_SIGNED(sum, h);
  }
}

// Fill the outside-frame part's residues with values derived from the in-frame
// part's residues.
static inline void fill_residue_outside_frame(
    int16_t *diff, ptrdiff_t diff_stride, int tx_cols, int tx_rows,
    int visible_tx_cols, int visible_tx_rows, TX_TYPE tx_type) {
  const int complete_block_outside =
      (visible_tx_cols == 0 || visible_tx_rows == 0);

  if (tx_type <= IDTX) {
    int16_t avg = 0;
    if (tx_type != IDTX && !complete_block_outside)
      avg =
          avg_wxh_block_c(diff, diff_stride, visible_tx_cols, visible_tx_rows);

    // Fill the remaining parts of the block with the average value
    const int right_pixels = tx_cols - visible_tx_cols;
    for (int i = 0; i < tx_rows; ++i) {
      aom_memset16(diff + i * diff_stride + visible_tx_cols, avg, right_pixels);
    }

    for (int i = visible_tx_rows; i < tx_rows; ++i) {
      aom_memset16(diff + i * diff_stride, avg, visible_tx_cols);
    }
  } else if (htx_tab[tx_type] == IDTX_1D) {
    if (visible_tx_rows < tx_rows) {
      int16_t out[64] = { 0 };
      if (!complete_block_outside)
        avg_wxh_block_vert_c(diff, diff_stride, visible_tx_cols,
                             visible_tx_rows, out);

      for (int j = 0; j < visible_tx_cols; j++) {
        for (int i = visible_tx_rows; i < tx_rows; ++i) {
          *(diff + i * diff_stride + j) = out[j];
        }
      }
    }

    const int right_pixels = tx_cols - visible_tx_cols;
    if (right_pixels) {
      for (int i = 0; i < tx_rows; ++i) {
        memset(diff + i * diff_stride + visible_tx_cols, 0,
               right_pixels * sizeof(*diff));
      }
    }
  } else {
    assert(vtx_tab[tx_type] == IDTX_1D);

    const int right_pixels = tx_cols - visible_tx_cols;
    if (right_pixels) {
      int16_t out[64] = { 0 };
      if (!complete_block_outside)
        avg_wxh_block_horiz_c(diff, diff_stride, visible_tx_cols,
                              visible_tx_rows, out);

      for (int i = 0; i < visible_tx_rows; ++i) {
        aom_memset16(diff + i * diff_stride + visible_tx_cols, out[i],
                     right_pixels);
      }
    }

    for (int i = visible_tx_rows; i < tx_rows; ++i) {
      memset(diff + i * diff_stride, 0, tx_cols * sizeof(*diff));
    }
  }
}

void av1_pad_block_border(const MACROBLOCK *x, int rows, int cols,
                          int16_t *diff, ptrdiff_t diff_stride, int plane,
                          BLOCK_SIZE plane_bsize, int blk_col, int blk_row,
                          TX_TYPE tx_type) {
  int visible_cols, visible_rows;
  const int is_border_block = get_visible_dimensions(
      x, plane, plane_bsize, blk_col, blk_row, cols, rows,
      /*clip_dims=*/true, &visible_cols, &visible_rows);
  if (is_border_block)
    fill_residue_outside_frame(diff, diff_stride, cols, rows, visible_cols,
                               visible_rows, tx_type);
}

// Hyper-parameters for dropout optimization, based on following logics.
// TODO(yjshen): These settings are tuned by experiments. They may still be
// optimized for better performance.
// (1) Coefficients which are large enough will ALWAYS be kept.
static const tran_low_t DROPOUT_COEFF_MAX = 2;  // Max dropout-able coefficient.
// (2) Continuous coefficients will ALWAYS be kept. Here rigorous continuity is
//     NOT required. For example, `5 0 0 0 7` is treated as two continuous
//     coefficients if three zeros do not fulfill the dropout condition.
static const int DROPOUT_CONTINUITY_MAX =
    2;  // Max dropout-able continuous coeff.
// (3) Dropout operation is NOT applicable to blocks with large or small
//     quantization index.
static const int DROPOUT_Q_MAX = 128;
static const int DROPOUT_Q_MIN = 16;
// (4) Recall that dropout optimization will forcibly set some quantized
//     coefficients to zero. The key logic on determining whether a coefficient
//     should be dropped is to check the number of continuous zeros before AND
//     after this coefficient. The exact number of zeros for judgement depends
//     on block size and quantization index. More concretely, block size
//     determines the base number of zeros, while quantization index determines
//     the multiplier. Intuitively, larger block requires more zeros and larger
//     quantization index also requires more zeros (more information is lost
//     when using larger quantization index).
static const int DROPOUT_BEFORE_BASE_MAX =
    32;  // Max base number for leading zeros.
static const int DROPOUT_BEFORE_BASE_MIN =
    16;  // Min base number for leading zeros.
static const int DROPOUT_AFTER_BASE_MAX =
    32;  // Max base number for trailing zeros.
static const int DROPOUT_AFTER_BASE_MIN =
    16;  // Min base number for trailing zeros.
static const int DROPOUT_MULTIPLIER_MAX =
    8;  // Max multiplier on number of zeros.
static const int DROPOUT_MULTIPLIER_MIN =
    2;  // Min multiplier on number of zeros.
static const int DROPOUT_MULTIPLIER_Q_BASE =
    32;  // Base Q to compute multiplier.

void av1_dropout_qcoeff(MACROBLOCK *mb, int plane, int block, TX_SIZE tx_size,
                        TX_TYPE tx_type, int qindex) {
  const int tx_width = tx_size_wide[tx_size];
  const int tx_height = tx_size_high[tx_size];

  // Early return if `qindex` is out of range.
  if (qindex > DROPOUT_Q_MAX || qindex < DROPOUT_Q_MIN) {
    return;
  }

  // Compute number of zeros used for dropout judgement.
  const int base_size = AOMMAX(tx_width, tx_height);
  const int multiplier = CLIP(qindex / DROPOUT_MULTIPLIER_Q_BASE,
                              DROPOUT_MULTIPLIER_MIN, DROPOUT_MULTIPLIER_MAX);
  const int dropout_num_before =
      multiplier *
      CLIP(base_size, DROPOUT_BEFORE_BASE_MIN, DROPOUT_BEFORE_BASE_MAX);
  const int dropout_num_after =
      multiplier *
      CLIP(base_size, DROPOUT_AFTER_BASE_MIN, DROPOUT_AFTER_BASE_MAX);

  av1_dropout_qcoeff_num(mb, plane, block, tx_size, tx_type, dropout_num_before,
                         dropout_num_after);
}

void av1_dropout_qcoeff_num(MACROBLOCK *mb, int plane, int block,
                            TX_SIZE tx_size, TX_TYPE tx_type,
                            int dropout_num_before, int dropout_num_after) {
  const struct macroblock_plane *const p = &mb->plane[plane];
  tran_low_t *const qcoeff = p->qcoeff + BLOCK_OFFSET(block);
  tran_low_t *const dqcoeff = p->dqcoeff + BLOCK_OFFSET(block);
  const int max_eob = av1_get_max_eob(tx_size);
  const SCAN_ORDER *const scan_order = get_scan(tx_size, tx_type);

  // Early return if there are not enough non-zero coefficients.
  if (p->eobs[block] == 0 || p->eobs[block] <= dropout_num_before ||
      max_eob <= dropout_num_before + dropout_num_after) {
    return;
  }

  int count_zeros_before = 0;
  int count_zeros_after = 0;
  int count_nonzeros = 0;
  // Index of the first non-zero coefficient after sufficient number of
  // continuous zeros. If equals to `-1`, it means number of leading zeros
  // hasn't reach `dropout_num_before`.
  int idx = -1;
  int eob = 0;  // New end of block.

  for (int i = 0; i < p->eobs[block]; ++i) {
    const int scan_idx = scan_order->scan[i];
    if (abs(qcoeff[scan_idx]) > DROPOUT_COEFF_MAX) {
      // Keep large coefficients.
      count_zeros_before = 0;
      count_zeros_after = 0;
      idx = -1;
      eob = i + 1;
    } else if (qcoeff[scan_idx] == 0) {  // Count zeros.
      if (idx == -1) {
        ++count_zeros_before;
      } else {
        ++count_zeros_after;
      }
    } else {  // Count non-zeros.
      if (count_zeros_before >= dropout_num_before) {
        idx = (idx == -1) ? i : idx;
        ++count_nonzeros;
      } else {
        count_zeros_before = 0;
        eob = i + 1;
      }
    }

    // Handle continuity.
    if (count_nonzeros > DROPOUT_CONTINUITY_MAX) {
      count_zeros_before = 0;
      count_zeros_after = 0;
      count_nonzeros = 0;
      idx = -1;
      eob = i + 1;
    }

    // Handle the trailing zeros after original end of block.
    if (idx != -1 && i == p->eobs[block] - 1) {
      count_zeros_after += (max_eob - p->eobs[block]);
    }

    // Set redundant coefficients to zeros if needed.
    if (count_zeros_after >= dropout_num_after) {
      for (int j = idx; j <= i; ++j) {
        qcoeff[scan_order->scan[j]] = 0;
        dqcoeff[scan_order->scan[j]] = 0;
      }
      count_zeros_before += (i - idx + 1);
      count_zeros_after = 0;
      count_nonzeros = 0;
    } else if (i == p->eobs[block] - 1) {
      eob = i + 1;
    }
  }

  if (eob != p->eobs[block]) {
    p->eobs[block] = eob;
    p->txb_entropy_ctx[block] =
        av1_get_txb_entropy_context(qcoeff, scan_order, eob);
  }
}

// Computes the transform for DC only blocks
void av1_xform_dc_only(MACROBLOCK *x, int plane, int block,
                       TxfmParam *txfm_param, int64_t per_px_mean) {
  assert(per_px_mean != INT64_MAX);
  const struct macroblock_plane *const p = &x->plane[plane];
  const int block_offset = BLOCK_OFFSET(block);
  tran_low_t *const coeff = p->coeff + block_offset;
  const int n_coeffs = av1_get_max_eob(txfm_param->tx_size);
  memset(coeff, 0, sizeof(*coeff) * n_coeffs);
  coeff[0] =
      (tran_low_t)((per_px_mean * dc_coeff_scale[txfm_param->tx_size]) >> 12);
}

static void encode_block(int plane, int block, int blk_row, int blk_col,
                         BLOCK_SIZE plane_bsize, TX_SIZE tx_size, void *arg,
                         RUN_TYPE dry_run) {
  (void)dry_run;
  struct encode_b_args *const args = arg;
  const AV1_COMP *const cpi = args->cpi;
  const AV1_COMMON *const cm = &cpi->common;
  MACROBLOCK *const x = args->x;
  MACROBLOCKD *const xd = &x->e_mbd;
  MB_MODE_INFO *mbmi = xd->mi[0];
  struct macroblock_plane *const p = &x->plane[plane];
  struct macroblockd_plane *const pd = &xd->plane[plane];
  tran_low_t *const dqcoeff = p->dqcoeff + BLOCK_OFFSET(block);
  uint8_t *dst;
  ENTROPY_CONTEXT *a, *l;
  int dummy_rate_cost = 0;

  dst = &pd->dst.buf[(blk_row * pd->dst.stride + blk_col) << MI_SIZE_LOG2];

  a = &args->ta[blk_col];
  l = &args->tl[blk_row];

  TX_TYPE tx_type = DCT_DCT;
  if (!mbmi->skip_mode) {
    tx_type = av1_get_tx_type(xd, pd->plane_type, blk_row, blk_col, tx_size,
                              cm->features.reduced_tx_set_used);

    av1_subtract_txb(x, plane, plane_bsize, blk_col, blk_row, tx_size, tx_type,
                     cpi->do_border_pad);
    TxfmParam txfm_param;
    QUANT_PARAM quant_param;
    const int use_trellis = is_trellis_used(args->enable_optimize_b, dry_run);
    int quant_idx;
    if (use_trellis)
      quant_idx = AV1_XFORM_QUANT_FP;
    else
      quant_idx =
          USE_B_QUANT_NO_TRELLIS ? AV1_XFORM_QUANT_B : AV1_XFORM_QUANT_FP;
    av1_setup_xform(cm, x, tx_size, tx_type, &txfm_param);
    av1_setup_quant(tx_size, use_trellis, quant_idx,
                    cpi->oxcf.q_cfg.quant_b_adapt, &quant_param);
    av1_setup_qmatrix(&cm->quant_params, xd, plane, tx_size, tx_type,
                      &quant_param);
    av1_xform_quant(x, plane, block, blk_row, blk_col, plane_bsize, &txfm_param,
                    &quant_param);
    if (use_trellis) {
      TXB_CTX txb_ctx;
      get_txb_ctx(plane_bsize, tx_size, plane, a, l, &txb_ctx);
      av1_optimize_b(args->cpi, x, plane, block, tx_size, tx_type, &txb_ctx,
                     &dummy_rate_cost);
    }
  } else {
    p->eobs[block] = 0;
    p->txb_entropy_ctx[block] = 0;
  }

  av1_set_txb_context(x, plane, block, tx_size, a, l);

  if (p->eobs[block]) {
    // As long as any YUV plane has non-zero quantized transform coefficients,
    // mbmi->skip_txfm flag is set to 0.
    mbmi->skip_txfm = 0;
    av1_inverse_transform_block(xd, dqcoeff, plane, tx_type, tx_size, dst,
                                pd->dst.stride, p->eobs[block],
                                cm->features.reduced_tx_set_used);
  } else {
    // Only when YUV planes all have zero quantized transform coefficients,
    // mbmi->skip_txfm flag is set to 1.
    mbmi->skip_txfm &= 1;
  }

  // TODO(debargha, jingning): Temporarily disable txk_type check for eob=0
  // case. It is possible that certain collision in hash index would cause
  // the assertion failure. To further optimize the rate-distortion
  // performance, we need to re-visit this part and enable this assert
  // again.
  if (p->eobs[block] == 0 && plane == 0) {
#if 0
    if (args->cpi->oxcf.q_cfg.aq_mode == NO_AQ &&
        args->cpi->oxcf.q_cfg.deltaq_mode == NO_DELTA_Q) {
      // TODO(jingning,angiebird,huisu@google.com): enable txk_check when
      // enable_optimize_b is true to detect potential RD bug.
      const uint8_t disable_txk_check = args->enable_optimize_b;
      if (!disable_txk_check) {
        assert(xd->tx_type_map[blk_row * xd->tx_type_map_stride + blk_col)] ==
            DCT_DCT);
      }
    }
#endif
    update_txk_array(xd, blk_row, blk_col, tx_size, DCT_DCT);
  }

#if CONFIG_MISMATCH_DEBUG
  if (dry_run == OUTPUT_ENABLED) {
    int pixel_c, pixel_r;
    BLOCK_SIZE bsize = txsize_to_bsize[tx_size];
    int blk_w = block_size_wide[bsize];
    int blk_h = block_size_high[bsize];
    mi_to_pixel_loc(&pixel_c, &pixel_r, xd->mi_col, xd->mi_row, blk_col,
                    blk_row, pd->subsampling_x, pd->subsampling_y);
    mismatch_record_block_tx(dst, pd->dst.stride, cm->current_frame.order_hint,
                             plane, pixel_c, pixel_r, blk_w, blk_h,
                             xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH);
  }
#endif
}

static void encode_block_inter(int plane, int block, int blk_row, int blk_col,
                               BLOCK_SIZE plane_bsize, TX_SIZE tx_size,
                               void *arg, RUN_TYPE dry_run) {
  struct encode_b_args *const args = arg;
  MACROBLOCK *const x = args->x;
  MACROBLOCKD *const xd = &x->e_mbd;
  MB_MODE_INFO *const mbmi = xd->mi[0];
  const struct macroblockd_plane *const pd = &xd->plane[plane];
  const int max_blocks_high = max_block_high(xd, plane_bsize, plane);
  const int max_blocks_wide = max_block_wide(xd, plane_bsize, plane);

  if (blk_row >= max_blocks_high || blk_col >= max_blocks_wide) return;

  const TX_SIZE plane_tx_size =
      plane ? av1_get_max_uv_txsize(mbmi->bsize, pd->subsampling_x,
                                    pd->subsampling_y)
            : mbmi->inter_tx_size[av1_get_txb_size_index(plane_bsize, blk_row,
                                                         blk_col)];
  if (!plane) {
    assert(tx_size_wide[tx_size] >= tx_size_wide[plane_tx_size] &&
           tx_size_high[tx_size] >= tx_size_high[plane_tx_size]);
  }

  if (tx_size == plane_tx_size || plane) {
    encode_block(plane, block, blk_row, blk_col, plane_bsize, tx_size, arg,
                 dry_run);
  } else {
    assert(tx_size < TX_SIZES_ALL);
    const TX_SIZE sub_txs = sub_tx_size_map[tx_size];
    assert(IMPLIES(tx_size <= TX_4X4, sub_txs == tx_size));
    assert(IMPLIES(tx_size > TX_4X4, sub_txs < tx_size));
    // This is the square transform block partition entry point.
    const int bsw = tx_size_wide_unit[sub_txs];
    const int bsh = tx_size_high_unit[sub_txs];
    const int step = bsh * bsw;
    const int row_end =
        AOMMIN(tx_size_high_unit[tx_size], max_blocks_high - blk_row);
    const int col_end =
        AOMMIN(tx_size_wide_unit[tx_size], max_blocks_wide - blk_col);
    assert(bsw > 0 && bsh > 0);

    for (int row = 0; row < row_end; row += bsh) {
      const int offsetr = blk_row + row;
      for (int col = 0; col < col_end; col += bsw) {
        const int offsetc = blk_col + col;

        encode_block_inter(plane, block, offsetr, offsetc, plane_bsize, sub_txs,
                           arg, dry_run);
        block += step;
      }
    }
  }
}

void av1_foreach_transformed_block_in_plane(
    const MACROBLOCKD *const xd, BLOCK_SIZE plane_bsize, int plane,
    foreach_transformed_block_visitor visit, void *arg) {
  const struct macroblockd_plane *const pd = &xd->plane[plane];
  // block and transform sizes, in number of 4x4 blocks log 2 ("*_b")
  // 4x4=0, 8x8=2, 16x16=4, 32x32=6, 64x64=8
  // transform size varies per plane, look it up in a common way.
  const TX_SIZE tx_size = av1_get_tx_size(plane, xd);
  const BLOCK_SIZE tx_bsize = txsize_to_bsize[tx_size];
  // Call visit() directly with zero offsets if the current block size is the
  // same as the transform block size.
  if (plane_bsize == tx_bsize) {
    visit(plane, 0, 0, 0, plane_bsize, tx_size, arg);
    return;
  }
  const uint8_t txw_unit = tx_size_wide_unit[tx_size];
  const uint8_t txh_unit = tx_size_high_unit[tx_size];
  const int step = txw_unit * txh_unit;

  // If mb_to_right_edge is < 0 we are in a situation in which
  // the current block size extends into the UMV and we won't
  // visit the sub blocks that are wholly within the UMV.
  const int max_blocks_wide = max_block_wide(xd, plane_bsize, plane);
  const int max_blocks_high = max_block_high(xd, plane_bsize, plane);
  const BLOCK_SIZE max_unit_bsize =
      get_plane_block_size(BLOCK_64X64, pd->subsampling_x, pd->subsampling_y);
  const int mu_blocks_wide =
      AOMMIN(mi_size_wide[max_unit_bsize], max_blocks_wide);
  const int mu_blocks_high =
      AOMMIN(mi_size_high[max_unit_bsize], max_blocks_high);

  // Keep track of the row and column of the blocks we use so that we know
  // if we are in the unrestricted motion border.
  int i = 0;
  for (int r = 0; r < max_blocks_high; r += mu_blocks_high) {
    const int unit_height = AOMMIN(mu_blocks_high + r, max_blocks_high);
    // Skip visiting the sub blocks that are wholly within the UMV.
    for (int c = 0; c < max_blocks_wide; c += mu_blocks_wide) {
      const int unit_width = AOMMIN(mu_blocks_wide + c, max_blocks_wide);
      for (int blk_row = r; blk_row < unit_height; blk_row += txh_unit) {
        for (int blk_col = c; blk_col < unit_width; blk_col += txw_unit) {
          visit(plane, i, blk_row, blk_col, plane_bsize, tx_size, arg);
          i += step;
        }
      }
    }
  }
  // Check if visit() is invoked at least once.
  assert(i >= 1);
}

typedef struct encode_block_pass1_args {
  AV1_COMP *cpi;
  MACROBLOCK *x;
} encode_block_pass1_args;

static void encode_block_pass1(int plane, int block, int blk_row, int blk_col,
                               BLOCK_SIZE plane_bsize, TX_SIZE tx_size,
                               void *arg) {
  encode_block_pass1_args *args = (encode_block_pass1_args *)arg;
  AV1_COMP *cpi = args->cpi;
  AV1_COMMON *cm = &cpi->common;
  MACROBLOCK *const x = args->x;
  MACROBLOCKD *const xd = &x->e_mbd;
  struct macroblock_plane *const p = &x->plane[plane];
  struct macroblockd_plane *const pd = &xd->plane[plane];
  tran_low_t *const dqcoeff = p->dqcoeff + BLOCK_OFFSET(block);

  uint8_t *dst;
  dst = &pd->dst.buf[(blk_row * pd->dst.stride + blk_col) << MI_SIZE_LOG2];

  TxfmParam txfm_param;
  QUANT_PARAM quant_param;

  av1_setup_xform(cm, x, tx_size, DCT_DCT, &txfm_param);
  av1_setup_quant(tx_size, 0, AV1_XFORM_QUANT_B, cpi->oxcf.q_cfg.quant_b_adapt,
                  &quant_param);
  av1_setup_qmatrix(&cm->quant_params, xd, plane, tx_size, DCT_DCT,
                    &quant_param);

  av1_xform_quant(x, plane, block, blk_row, blk_col, plane_bsize, &txfm_param,
                  &quant_param);

  if (p->eobs[block] > 0) {
    txfm_param.eob = p->eobs[block];
    if (txfm_param.is_hbd) {
      av1_highbd_inv_txfm_add(dqcoeff, dst, pd->dst.stride, &txfm_param);
      return;
    }
    av1_inv_txfm_add(dqcoeff, dst, pd->dst.stride, &txfm_param);
  }
}

void av1_encode_sby_pass1(AV1_COMP *cpi, MACROBLOCK *x, BLOCK_SIZE bsize) {
  encode_block_pass1_args args = { cpi, x };
  av1_subtract_plane(x, bsize, PLANE_TYPE_Y, cpi->do_border_pad);
  av1_foreach_transformed_block_in_plane(&x->e_mbd, bsize, 0,
                                         encode_block_pass1, &args);
}

void av1_encode_sb(const struct AV1_COMP *cpi, MACROBLOCK *x, BLOCK_SIZE bsize,
                   RUN_TYPE dry_run) {
  assert(bsize < BLOCK_SIZES_ALL);
  MACROBLOCKD *const xd = &x->e_mbd;
  MB_MODE_INFO *mbmi = xd->mi[0];
  // In the current encoder implementation, for inter blocks,
  // only when YUV planes all have zero quantized transform coefficients,
  // mbmi->skip_txfm flag is set to 1.
  // For intra blocks, this flag is set to 0 since skipped blocks are so rare
  // that transmitting skip_txfm = 1 is very expensive.
  // mbmi->skip_txfm is init to 1, and will be modified in encode_block() based
  // on transform, quantization, and (if exists) trellis optimization.
  mbmi->skip_txfm = 1;
  if (x->txfm_search_info.skip_txfm) return;

  struct optimize_ctx ctx;
  struct encode_b_args arg = {
    cpi, x, &ctx, NULL, NULL, dry_run, cpi->optimize_seg_arr[mbmi->segment_id]
  };
  const AV1_COMMON *const cm = &cpi->common;
  const int num_planes = av1_num_planes(cm);
  for (int plane = 0; plane < num_planes; ++plane) {
    const struct macroblockd_plane *const pd = &xd->plane[plane];
    const int subsampling_x = pd->subsampling_x;
    const int subsampling_y = pd->subsampling_y;
    if (plane && !xd->is_chroma_ref) break;
    const BLOCK_SIZE plane_bsize =
        get_plane_block_size(bsize, subsampling_x, subsampling_y);
    assert(plane_bsize < BLOCK_SIZES_ALL);
    const int mi_width = mi_size_wide[plane_bsize];
    const int mi_height = mi_size_high[plane_bsize];
    const TX_SIZE max_tx_size = get_vartx_max_txsize(xd, plane_bsize, plane);
    const BLOCK_SIZE txb_size = txsize_to_bsize[max_tx_size];
    const int bw = mi_size_wide[txb_size];
    const int bh = mi_size_high[txb_size];
    int block = 0;
    const int step =
        tx_size_wide_unit[max_tx_size] * tx_size_high_unit[max_tx_size];
    av1_get_entropy_contexts(plane_bsize, pd, ctx.ta[plane], ctx.tl[plane]);
    arg.ta = ctx.ta[plane];
    arg.tl = ctx.tl[plane];
    const BLOCK_SIZE max_unit_bsize =
        get_plane_block_size(BLOCK_64X64, subsampling_x, subsampling_y);
    int mu_blocks_wide = mi_size_wide[max_unit_bsize];
    int mu_blocks_high = mi_size_high[max_unit_bsize];
    mu_blocks_wide = AOMMIN(mi_width, mu_blocks_wide);
    mu_blocks_high = AOMMIN(mi_height, mu_blocks_high);

    for (int idy = 0; idy < mi_height; idy += mu_blocks_high) {
      for (int idx = 0; idx < mi_width; idx += mu_blocks_wide) {
        int blk_row, blk_col;
        const int unit_height = AOMMIN(mu_blocks_high + idy, mi_height);
        const int unit_width = AOMMIN(mu_blocks_wide + idx, mi_width);
        for (blk_row = idy; blk_row < unit_height; blk_row += bh) {
          for (blk_col = idx; blk_col < unit_width; blk_col += bw) {
            encode_block_inter(plane, block, blk_row, blk_col, plane_bsize,
                               max_tx_size, &arg, dry_run);
            block += step;
          }
        }
      }
    }
  }
}

static AOM_FORCE_INLINE void quantize_b_lowbd(
    const struct macroblock_plane *const p, const SCAN_ORDER *const scan_order,
    int log_scale, int n_coeffs, const tran_low_t *const coeff,
    tran_low_t *const qcoeff, tran_low_t *const dqcoeff, uint16_t *const eob) {
  if (log_scale == 0) {
    aom_quantize_b(coeff, n_coeffs, p->zbin_QTX, p->round_QTX, p->quant_QTX,
                   p->quant_shift_QTX, qcoeff, dqcoeff, p->dequant_QTX, eob,
                   scan_order->scan, scan_order->iscan);
  } else if (log_scale == 1) {
    aom_quantize_b_32x32(coeff, n_coeffs, p->zbin_QTX, p->round_QTX,
                         p->quant_QTX, p->quant_shift_QTX, qcoeff, dqcoeff,
                         p->dequant_QTX, eob, scan_order->scan,
                         scan_order->iscan);
  } else {
    assert(log_scale == 2);
    aom_quantize_b_64x64(coeff, n_coeffs, p->zbin_QTX, p->round_QTX,
                         p->quant_QTX, p->quant_shift_QTX, qcoeff, dqcoeff,
                         p->dequant_QTX, eob, scan_order->scan,
                         scan_order->iscan);
  }
}

static AOM_FORCE_INLINE void recon_intra_tx_block(
    MACROBLOCKD *xd, const tran_low_t *dqcoeff, uint8_t *dst, int dst_stride,
    int plane, int blk_row, int blk_col, BLOCK_SIZE plane_bsize,
    TX_SIZE tx_size, uint16_t eob, int is_hbd, TxfmParam *txfm_param) {
  if (eob) {
    assert(eob <= av1_get_max_eob(tx_size));
    assert(av1_ext_tx_used[txfm_param->tx_set_type][txfm_param->tx_type]);
    txfm_param->eob = eob;
#if CONFIG_AV1_HIGHBITDEPTH
    if (is_hbd) {
      av1_highbd_inv_txfm_add(dqcoeff, dst, dst_stride, txfm_param);
    } else {
      av1_inv_txfm_add(dqcoeff, dst, dst_stride, txfm_param);
    }
#else
    (void)is_hbd;
    av1_inv_txfm_add(dqcoeff, dst, dst_stride, txfm_param);
#endif
  } else if (plane == AOM_PLANE_Y) {
    update_txk_array(xd, blk_row, blk_col, tx_size, DCT_DCT);
  }
#if !CONFIG_REALTIME_ONLY
  if (plane == AOM_PLANE_Y && xd->cfl.store_y) {
    cfl_store_tx(xd, blk_row, blk_col, tx_size, plane_bsize);
  }
#else
  (void)plane_bsize;
#endif
}

// Note on p->txb_entropy_ctx[block] across the intra encode helpers below:
// every reader (av1_set_txb_context() in multi-block trellis here, plus
// tx_search.c) writes p->txb_entropy_ctx[block] earlier on its own path, so
// skip, single-block trellis, and non-trellis intra paths omit writing it.

typedef struct encode_intra_skip_args {
  const AV1_COMP *cpi;
  MACROBLOCK *x;
} encode_intra_skip_args;

static void encode_block_intra_skip(int plane, int block, int blk_row,
                                    int blk_col, BLOCK_SIZE plane_bsize,
                                    TX_SIZE tx_size, void *arg) {
  const encode_intra_skip_args *const args =
      (const encode_intra_skip_args *)arg;
  const AV1_COMP *const cpi = args->cpi;
  MACROBLOCK *const x = args->x;
  MACROBLOCKD *const xd = &x->e_mbd;
  struct macroblock_plane *const p = &x->plane[plane];
  av1_predict_intra_block_facade(&cpi->common, xd, plane, blk_col, blk_row,
                                 tx_size);
  p->eobs[block] = 0;
  if (plane == AOM_PLANE_Y)
    update_txk_array(xd, blk_row, blk_col, tx_size, DCT_DCT);
#if !CONFIG_REALTIME_ONLY
  if (plane == AOM_PLANE_Y && xd->cfl.store_y) {
    cfl_store_tx(xd, blk_row, blk_col, tx_size, plane_bsize);
  }
#else
  (void)plane_bsize;
#endif
}

// Outlined so skip_txfm state stays out of av1_encode_intra_block_plane().
static AOM_NOINLINE void encode_intra_block_plane_skip(const AV1_COMP *cpi,
                                                       MACROBLOCK *x,
                                                       BLOCK_SIZE plane_bsize,
                                                       int plane) {
  encode_intra_skip_args args = { cpi, x };
  av1_foreach_transformed_block_in_plane(&x->e_mbd, plane_bsize, plane,
                                         encode_block_intra_skip, &args);
}

// !using_qmatrix uses qm level 15, leaving seg_qmatrix/seg_iqmatrix NULL.
static inline int has_qmatrix(const CommonQuantParams *quant_params,
                              const struct macroblockd_plane *pd, int seg_id,
                              TX_SIZE tx_size) {
  return quant_params->using_qmatrix &&
         (pd->seg_qmatrix[seg_id][tx_size] != NULL ||
          pd->seg_iqmatrix[seg_id][tx_size] != NULL);
}

typedef struct encode_intra_trellis_args {
  const AV1_COMP *cpi;
  MACROBLOCK *x;
  ENTROPY_CONTEXT *ta;
  ENTROPY_CONTEXT *tl;
  TxfmParam txfm_param;
  QUANT_PARAM quant_param;
  int has_qm;
  int is_single_block;
} encode_intra_trellis_args;

static void encode_block_intra_trellis(int plane, int block, int blk_row,
                                       int blk_col, BLOCK_SIZE plane_bsize,
                                       TX_SIZE tx_size, void *arg) {
  encode_intra_trellis_args *const args = (encode_intra_trellis_args *)arg;
  const AV1_COMP *const cpi = args->cpi;
  const AV1_COMMON *const cm = &cpi->common;
  MACROBLOCK *const x = args->x;
  MACROBLOCKD *const xd = &x->e_mbd;
  struct macroblock_plane *const p = &x->plane[plane];
  const struct macroblockd_plane *const pd = &xd->plane[plane];
  const int block_offset = BLOCK_OFFSET(block);
  tran_low_t *const qcoeff = p->qcoeff + block_offset;
  tran_low_t *const dqcoeff = p->dqcoeff + block_offset;
  const PLANE_TYPE plane_type = get_plane_type(plane);
  const int dst_stride = pd->dst.stride;
  uint8_t *const dst =
      &pd->dst.buf[(blk_row * dst_stride + blk_col) << MI_SIZE_LOG2];
  ENTROPY_CONTEXT *const a = &args->ta[blk_col];
  ENTROPY_CONTEXT *const l = &args->tl[blk_row];

  av1_predict_intra_block_facade(cm, xd, plane, blk_col, blk_row, tx_size);

  const TX_TYPE tx_type =
      av1_get_tx_type(xd, plane_type, blk_row, blk_col, tx_size,
                      cm->features.reduced_tx_set_used);
  assert(!is_stat_generation_stage(cpi));
  av1_subtract_txb(x, plane, plane_bsize, blk_col, blk_row, tx_size, tx_type,
                   cpi->do_border_pad);
  args->txfm_param.tx_type = tx_type;
  if (args->has_qm) {
    av1_setup_qmatrix(&cm->quant_params, xd, plane, tx_size, tx_type,
                      &args->quant_param);
  }
  av1_xform(x, plane, block, blk_row, blk_col, plane_bsize, &args->txfm_param);
  assert(!x->seg_skip_block);
  const int n_coeffs = av1_get_max_eob(tx_size);
  const SCAN_ORDER *const scan_order = get_scan(tx_size, tx_type);
  const tran_low_t *const coeff = p->coeff + block_offset;
  uint16_t *const eob_ptr = &p->eobs[block];
#if CONFIG_AV1_HIGHBITDEPTH
  if (!args->txfm_param.is_hbd) {
    av1_quantize_fp_facade(coeff, n_coeffs, p, qcoeff, dqcoeff, eob_ptr,
                           scan_order, &args->quant_param);
  } else {
    av1_highbd_quantize_fp_facade(coeff, n_coeffs, p, qcoeff, dqcoeff, eob_ptr,
                                  scan_order, &args->quant_param);
  }
#else
  av1_quantize_fp_facade(coeff, n_coeffs, p, qcoeff, dqcoeff, eob_ptr,
                         scan_order, &args->quant_param);
#endif
  uint16_t eob = *eob_ptr;
  if (eob) {
    TXB_CTX txb_ctx;
    int dummy_rate_cost = 0;
    get_txb_ctx(plane_bsize, tx_size, plane, a, l, &txb_ctx);
    const int opt_eob =
        av1_optimize_txb(cpi, x, plane, block, tx_size, tx_type, &txb_ctx,
                         &dummy_rate_cost, cpi->oxcf.algo_cfg.sharpness);
    assert(opt_eob == p->eobs[block]);
    eob = (uint16_t)opt_eob;
  } else if (!args->is_single_block) {
    p->txb_entropy_ctx[block] = 0;
  }
  if (!args->is_single_block) {
    av1_set_txb_context(x, plane, block, tx_size, a, l);
  }
  recon_intra_tx_block(xd, dqcoeff, dst, dst_stride, plane, blk_row, blk_col,
                       plane_bsize, tx_size, eob, args->txfm_param.is_hbd,
                       &args->txfm_param);
}

// Outlined so ta/tl buffers and trellis state stay out of
// av1_encode_intra_block_plane().
static AOM_NOINLINE void encode_intra_block_plane_trellis(
    const AV1_COMP *cpi, MACROBLOCK *x, BLOCK_SIZE plane_bsize, int plane,
    TX_SIZE tx_size) {
  const AV1_COMMON *const cm = &cpi->common;
  MACROBLOCKD *const xd = &x->e_mbd;
  const struct macroblockd_plane *const pd = &xd->plane[plane];
  const int seg_id = xd->mi[0]->segment_id;

  ENTROPY_CONTEXT ta[MAX_MIB_SIZE];
  ENTROPY_CONTEXT tl[MAX_MIB_SIZE];
  encode_intra_trellis_args args;
  args.cpi = cpi;
  args.x = x;
  args.has_qm = has_qmatrix(&cm->quant_params, pd, seg_id, tx_size);
  const int is_single_block = (plane_bsize == txsize_to_bsize[tx_size]);
  args.is_single_block = is_single_block;
  if (is_single_block) {
    args.ta = pd->above_entropy_context;
    args.tl = pd->left_entropy_context;
  } else {
    av1_get_entropy_contexts(plane_bsize, pd, ta, tl);
    args.ta = ta;
    args.tl = tl;
  }
  av1_setup_xform(cm, x, tx_size, DCT_DCT, &args.txfm_param);
  av1_setup_quant(tx_size, /*use_optimize_b=*/1, AV1_XFORM_QUANT_FP,
                  cpi->oxcf.q_cfg.quant_b_adapt, &args.quant_param);

  if (is_single_block) {
    encode_block_intra_trellis(plane, 0, 0, 0, plane_bsize, tx_size, &args);
  } else {
    av1_foreach_transformed_block_in_plane(xd, plane_bsize, plane,
                                           encode_block_intra_trellis, &args);
  }
}

typedef struct encode_intra_generic_args {
  const AV1_COMP *cpi;
  MACROBLOCK *x;
  TxfmParam txfm_param;
  QUANT_PARAM quant_param;
  int has_qm;
  bool do_border_pad;
  bool is_stat_gen;
} encode_intra_generic_args;

static void encode_block_intra_generic(int plane, int block, int blk_row,
                                       int blk_col, BLOCK_SIZE plane_bsize,
                                       TX_SIZE tx_size, void *arg) {
  encode_intra_generic_args *const args = (encode_intra_generic_args *)arg;
  const AV1_COMP *const cpi = args->cpi;
  const AV1_COMMON *const cm = &cpi->common;
  MACROBLOCK *const x = args->x;
  MACROBLOCKD *const xd = &x->e_mbd;
  struct macroblock_plane *const p = &x->plane[plane];
  const struct macroblockd_plane *const pd = &xd->plane[plane];
  const PLANE_TYPE plane_type = get_plane_type(plane);
  const int dst_stride = pd->dst.stride;
  const int reduced_tx_set = cm->features.reduced_tx_set_used;

  av1_predict_intra_block_facade(cm, xd, plane, blk_col, blk_row, tx_size);
  const TX_TYPE tx_type = av1_get_tx_type(xd, plane_type, blk_row, blk_col,
                                          tx_size, reduced_tx_set);
  const TX_TYPE primary_tx_type = args->is_stat_gen ? DCT_DCT : tx_type;
  av1_subtract_txb(x, plane, plane_bsize, blk_col, blk_row, tx_size,
                   primary_tx_type, args->do_border_pad);
  args->txfm_param.tx_type = tx_type;
  if (args->has_qm) {
    av1_setup_qmatrix(&cm->quant_params, xd, plane, tx_size, tx_type,
                      &args->quant_param);
  }
  av1_xform(x, plane, block, blk_row, blk_col, plane_bsize, &args->txfm_param);
  av1_quant_coeffs(x, plane, block, &args->txfm_param, &args->quant_param);
  uint8_t *const dst =
      &pd->dst.buf[(blk_row * dst_stride + blk_col) << MI_SIZE_LOG2];
  recon_intra_tx_block(xd, p->dqcoeff + BLOCK_OFFSET(block), dst, dst_stride,
                       plane, blk_row, blk_col, plane_bsize, tx_size,
                       p->eobs[block], args->txfm_param.is_hbd,
                       &args->txfm_param);
}

// Outlined fallback for non-trellis HBD, qmatrix, right/bottom-edge border
// padding, quant_b_adapt, seg_skip_block, non-B quant, or trellis disabled by
// segment/lossless.
static AOM_NOINLINE void encode_intra_block_plane_generic(
    const AV1_COMP *cpi, MACROBLOCK *x, BLOCK_SIZE plane_bsize, int plane,
    TX_SIZE tx_size, int quant_idx) {
  const AV1_COMMON *const cm = &cpi->common;
  MACROBLOCKD *const xd = &x->e_mbd;
  const struct macroblockd_plane *const pd = &xd->plane[plane];
  const int seg_id = xd->mi[0]->segment_id;
  const bool do_border_pad = cpi->do_border_pad;
  encode_intra_generic_args args;
  args.cpi = cpi;
  args.x = x;
  args.has_qm = has_qmatrix(&cm->quant_params, pd, seg_id, tx_size);
  args.do_border_pad = do_border_pad;
  // primary_tx_type is only read inside av1_subtract_txb()'s border-pad path.
  args.is_stat_gen = do_border_pad && is_stat_generation_stage(cpi);
  av1_setup_xform(cm, x, tx_size, DCT_DCT, &args.txfm_param);
  av1_setup_quant(tx_size, /*use_optimize_b=*/0, quant_idx,
                  cpi->oxcf.q_cfg.quant_b_adapt, &args.quant_param);

  if (plane_bsize == txsize_to_bsize[tx_size]) {
    encode_block_intra_generic(plane, 0, 0, 0, plane_bsize, tx_size, &args);
  } else {
    av1_foreach_transformed_block_in_plane(xd, plane_bsize, plane,
                                           encode_block_intra_generic, &args);
  }
}

// In av1_ext_tx_set_lookup, intra square-up >= TX_32X32 is EXT_TX_SET_DCTONLY,
// while smaller intra sets always include DTT4 (all intra_mode_to_tx_type()
// outputs).
static inline TX_TYPE get_intra_uv_tx_type(const MB_MODE_INFO *mbmi,
                                           TX_SIZE tx_size) {
  return txsize_sqr_up_map[tx_size] > TX_16X16
             ? DCT_DCT
             : intra_mode_to_tx_type(mbmi, PLANE_TYPE_UV);
}

// Outlined so multi-block loop state stays out of
// av1_encode_intra_block_plane().
static AOM_NOINLINE void encode_intra_block_plane_multi(const AV1_COMP *cpi,
                                                        MACROBLOCK *x,
                                                        BLOCK_SIZE plane_bsize,
                                                        int plane,
                                                        TX_SIZE tx_size) {
  const AV1_COMMON *const cm = &cpi->common;
  MACROBLOCKD *const xd = &x->e_mbd;
  const struct macroblockd_plane *const pd = &xd->plane[plane];
  struct macroblock_plane *const p = &x->plane[plane];
  const MB_MODE_INFO *const mbmi = xd->mi[0];
  const int seg_id = mbmi->segment_id;
  const int is_lossless = xd->lossless[seg_id];
  const int force_dct = is_lossless || txsize_sqr_up_map[tx_size] > TX_32X32;
  // TX_TYPES sentinel means look up tx_type per luma block in tx_type_map;
  // intra chroma shares one tx_type across the plane.
  const TX_TYPE invariant_tx_type =
      force_dct ? DCT_DCT
                : (plane != AOM_PLANE_Y ? get_intra_uv_tx_type(mbmi, tx_size)
                                        : TX_TYPES);
  const int diff_stride = block_size_wide[plane_bsize];
  const int src_stride = p->src.stride;
  const int dst_stride = pd->dst.stride;
  const int tx1d_width = tx_size_wide[tx_size];
  const int tx1d_height = tx_size_high[tx_size];
  const int n_coeffs = av1_get_max_eob(tx_size);
  const int log_scale = av1_get_tx_scale(tx_size);
  const uint8_t txw_unit = tx_size_wide_unit[tx_size];
  const uint8_t txh_unit = tx_size_high_unit[tx_size];
  const int step = txw_unit * txh_unit;
  const int max_blocks_wide = max_block_wide(xd, plane_bsize, plane);
  const int max_blocks_high = max_block_high(xd, plane_bsize, plane);
  const int mu_blocks_wide =
      AOMMIN(MI_SIZE_64X64 >> pd->subsampling_x, max_blocks_wide);
  const int mu_blocks_high =
      AOMMIN(MI_SIZE_64X64 >> pd->subsampling_y, max_blocks_high);

  uint8_t *const dst_base = pd->dst.buf;
  const uint8_t *const src_base = p->src.buf;
  int16_t *const src_diff_base = p->src_diff;
  tran_low_t *const coeff_base = p->coeff;
  tran_low_t *const qcoeff_base = p->qcoeff;
  tran_low_t *const dqcoeff_base = p->dqcoeff;
  uint16_t *const eobs = p->eobs;
  const uint8_t *const tx_type_map = xd->tx_type_map;
  const int tx_type_map_stride = xd->tx_type_map_stride;

  TxfmParam txfm_param;
  txfm_param.tx_size = tx_size;
  txfm_param.lossless = is_lossless;
#if !defined(NDEBUG)
  txfm_param.tx_set_type = av1_get_ext_tx_set_type(
      tx_size, /*is_inter=*/0, cm->features.reduced_tx_set_used);
#endif
  txfm_param.bd = 8;
  txfm_param.is_hbd = 0;

  assert(max_blocks_high > 0 && max_blocks_wide > 0);
  assert(txh_unit > 0 && txw_unit > 0);
  int i = 0;
  int r = 0;
  do {
    const int unit_height = AOMMIN(mu_blocks_high + r, max_blocks_high);
    int c = 0;
    do {
      const int unit_width = AOMMIN(mu_blocks_wide + c, max_blocks_wide);
      int blk_row = r;
      do {
        int blk_col = c;
        do {
          av1_predict_intra_block_facade(cm, xd, plane, blk_col, blk_row,
                                         tx_size);
          const TX_TYPE tx_type =
              (invariant_tx_type < TX_TYPES)
                  ? invariant_tx_type
                  : (TX_TYPE)
                        tx_type_map[blk_row * tx_type_map_stride + blk_col];
          assert(tx_type < TX_TYPES);
          assert(av1_ext_tx_used[txfm_param.tx_set_type][tx_type]);
          uint8_t *const dst =
              &dst_base[(blk_row * dst_stride + blk_col) << MI_SIZE_LOG2];
          const uint8_t *const src =
              &src_base[(blk_row * src_stride + blk_col) << MI_SIZE_LOG2];
          int16_t *const src_diff =
              &src_diff_base[(blk_row * diff_stride + blk_col) << MI_SIZE_LOG2];
          aom_subtract_block(tx1d_height, tx1d_width, src_diff, diff_stride,
                             src, src_stride, dst, dst_stride);
          const int block_offset = BLOCK_OFFSET(i);
          tran_low_t *const coeff = coeff_base + block_offset;
          tran_low_t *const qcoeff = qcoeff_base + block_offset;
          tran_low_t *const dqcoeff = dqcoeff_base + block_offset;
          txfm_param.tx_type = tx_type;
          av1_lowbd_fwd_txfm(src_diff, coeff, diff_stride, &txfm_param);
          const SCAN_ORDER *const scan_order = get_scan(tx_size, tx_type);
          quantize_b_lowbd(p, scan_order, log_scale, n_coeffs, coeff, qcoeff,
                           dqcoeff, &eobs[i]);
          recon_intra_tx_block(xd, dqcoeff, dst, dst_stride, plane, blk_row,
                               blk_col, plane_bsize, tx_size, eobs[i], 0,
                               &txfm_param);
          i += step;
          blk_col += txw_unit;
        } while (blk_col < unit_width);
        blk_row += txh_unit;
      } while (blk_row < unit_height);
      c += mu_blocks_wide;
    } while (c < max_blocks_wide);
    r += mu_blocks_high;
  } while (r < max_blocks_high);
}

// Outlined 8-bit single-block path so av1_encode_intra_block_plane() keeps
// per-path locals out of the top-level dispatcher.
static AOM_NOINLINE void encode_intra_block_plane_single(const AV1_COMP *cpi,
                                                         MACROBLOCK *x,
                                                         BLOCK_SIZE plane_bsize,
                                                         int plane,
                                                         TX_SIZE tx_size) {
  const AV1_COMMON *const cm = &cpi->common;
  MACROBLOCKD *const xd = &x->e_mbd;

  av1_predict_intra_block_facade(cm, xd, plane, 0, 0, tx_size);

  const MB_MODE_INFO *const mbmi = xd->mi[0];
  struct macroblock_plane *const p = &x->plane[plane];
  const struct macroblockd_plane *const pd = &xd->plane[plane];
  const int is_lossless = xd->lossless[mbmi->segment_id];
  const int force_dct = is_lossless || txsize_sqr_up_map[tx_size] > TX_32X32;
  const TX_TYPE tx_type =
      force_dct ? DCT_DCT
                : (plane == AOM_PLANE_Y ? (TX_TYPE)xd->tx_type_map[0]
                                        : get_intra_uv_tx_type(mbmi, tx_size));
  assert(tx_type < TX_TYPES);

  TxfmParam txfm_param;
  txfm_param.tx_type = tx_type;
  txfm_param.tx_size = tx_size;
  txfm_param.lossless = is_lossless;
#if !defined(NDEBUG)
  txfm_param.tx_set_type = av1_get_ext_tx_set_type(
      tx_size, /*is_inter=*/0, cm->features.reduced_tx_set_used);
  assert(av1_ext_tx_used[txfm_param.tx_set_type][tx_type]);
#endif
  txfm_param.bd = 8;
  txfm_param.is_hbd = 0;

  const int diff_stride = block_size_wide[plane_bsize];
  const int dst_stride = pd->dst.stride;
  uint8_t *const dst = pd->dst.buf;
  int16_t *const src_diff = p->src_diff;
  tran_low_t *const coeff = p->coeff;
  tran_low_t *const dqcoeff = p->dqcoeff;
  uint16_t *const eobs = p->eobs;
  aom_subtract_block(tx_size_high[tx_size], tx_size_wide[tx_size], src_diff,
                     diff_stride, p->src.buf, p->src.stride, dst, dst_stride);
  av1_lowbd_fwd_txfm(src_diff, coeff, diff_stride, &txfm_param);
  const SCAN_ORDER *const scan_order = get_scan(tx_size, tx_type);
  const int n_coeffs = av1_get_max_eob(tx_size);
  const int log_scale = av1_get_tx_scale(tx_size);
  quantize_b_lowbd(p, scan_order, log_scale, n_coeffs, coeff, p->qcoeff,
                   dqcoeff, eobs);
  recon_intra_tx_block(xd, dqcoeff, dst, dst_stride, plane, 0, 0, plane_bsize,
                       tx_size, eobs[0], 0, &txfm_param);
}

void av1_encode_intra_block_plane(const struct AV1_COMP *cpi, MACROBLOCK *x,
                                  BLOCK_SIZE bsize, int plane, RUN_TYPE dry_run,
                                  TRELLIS_OPT_TYPE enable_optimize_b) {
  assert(bsize < BLOCK_SIZES_ALL);
  assert(plane >= 0 && plane < MAX_MB_PLANE);
  MACROBLOCKD *const xd = &x->e_mbd;
  const MB_MODE_INFO *const mbmi = xd->mi[0];
  assert(!is_inter_block(mbmi));
  assert(bsize == mbmi->bsize);
  if (plane && !xd->is_chroma_ref) return;

  const struct macroblockd_plane *const pd = &xd->plane[plane];
  const BLOCK_SIZE plane_bsize =
      plane ? get_plane_block_size(bsize, pd->subsampling_x, pd->subsampling_y)
            : bsize;
  assert(plane_bsize ==
         get_plane_block_size(bsize, pd->subsampling_x, pd->subsampling_y));
  assert(plane_bsize < BLOCK_SIZES_ALL);
  if (UNLIKELY(mbmi->skip_txfm)) {
    encode_intra_block_plane_skip(cpi, x, plane_bsize, plane);
    return;
  }

  const int seg_id = mbmi->segment_id;
  const TX_SIZE tx_size =
      xd->lossless[seg_id]
          ? TX_4X4
          : (plane
                 ? av1_get_adjusted_tx_size(max_txsize_rect_lookup[plane_bsize])
                 : mbmi->tx_size);
  assert(tx_size == av1_get_tx_size(plane, xd));
  assert(tx_size < TX_SIZES_ALL);
  if (is_trellis_used(enable_optimize_b, dry_run)) {
    if (cpi->optimize_seg_arr[seg_id] && !xd->lossless[seg_id] &&
        !x->seg_skip_block) {
      encode_intra_block_plane_trellis(cpi, x, plane_bsize, plane, tx_size);
    } else {
      encode_intra_block_plane_generic(cpi, x, plane_bsize, plane, tx_size,
                                       AV1_XFORM_QUANT_FP);
    }
    return;
  }

  if (!USE_B_QUANT_NO_TRELLIS || is_cur_buf_hbd(xd) ||
#if !CONFIG_REALTIME_ONLY
      cpi->oxcf.q_cfg.quant_b_adapt ||
#endif
      x->seg_skip_block ||
      (cpi->do_border_pad &&
       (x->pix_to_right_edge < 0 || x->pix_to_bottom_edge < 0)) ||
      has_qmatrix(&cpi->common.quant_params, pd, seg_id, tx_size)) {
    const int quant_idx =
        USE_B_QUANT_NO_TRELLIS ? AV1_XFORM_QUANT_B : AV1_XFORM_QUANT_FP;
    encode_intra_block_plane_generic(cpi, x, plane_bsize, plane, tx_size,
                                     quant_idx);
    return;
  }
  assert(xd->bd == 8);

  if (plane_bsize != txsize_to_bsize[tx_size]) {
    encode_intra_block_plane_multi(cpi, x, plane_bsize, plane, tx_size);
    return;
  }

  encode_intra_block_plane_single(cpi, x, plane_bsize, plane, tx_size);
}
