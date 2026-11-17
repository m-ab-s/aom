/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"
#include "config/av1_rtcd.h"

#include "aom_dsp/aom_dsp_common.h"
#include "aom_ports/mem.h"

#if HAVE_NEON
#include "av1/common/enums.h"
#include "av1/common/reconintra.h"
#include "av1/encoder/encodemb.h"
#include "av1/encoder/hybrid_fwd_txfm.h"
#include "av1/encoder/intra_mode_search_utils.h"
#endif

int av1_hadamard_satd_4x4_sub_c(const uint8_t *src, ptrdiff_t src_stride,
                                const uint8_t *pred, ptrdiff_t pred_stride) {
  DECLARE_ALIGNED(16, int16_t, src_diff[16]);
  DECLARE_ALIGNED(16, tran_low_t, coeff[16]);
  aom_subtract_block_c(4, 4, src_diff, 4, src, src_stride, pred, pred_stride);
  aom_hadamard_4x4_c(src_diff, 4, coeff);
  return aom_satd_c(coeff, 16);
}

int av1_hadamard_satd_8x8_sub_c(const uint8_t *src, ptrdiff_t src_stride,
                                const uint8_t *pred, ptrdiff_t pred_stride) {
  DECLARE_ALIGNED(16, int16_t, src_diff[64]);
  DECLARE_ALIGNED(16, int16_t, lp_coeff[64]);
  aom_subtract_block_c(8, 8, src_diff, 8, src, src_stride, pred, pred_stride);
  aom_hadamard_lp_8x8_c(src_diff, 8, lp_coeff);
  return aom_satd_lp_c(lp_coeff, 64);
}

int av1_hadamard_satd_16x16_sub_c(const uint8_t *src, ptrdiff_t src_stride,
                                  const uint8_t *pred, ptrdiff_t pred_stride) {
  DECLARE_ALIGNED(16, int16_t, src_diff[256]);
  DECLARE_ALIGNED(16, int16_t, lp_coeff[256]);
  aom_subtract_block_c(16, 16, src_diff, 16, src, src_stride, pred,
                       pred_stride);
  aom_hadamard_lp_16x16_c(src_diff, 16, lp_coeff);
  return aom_satd_lp_c(lp_coeff, 256);
}

int av1_hadamard_satd_32x32_sub_c(const uint8_t *src, ptrdiff_t src_stride,
                                  const uint8_t *pred, ptrdiff_t pred_stride) {
  DECLARE_ALIGNED(16, int16_t, src_diff[1024]);
  DECLARE_ALIGNED(16, tran_low_t, coeff[1024]);
  aom_subtract_block_c(32, 32, src_diff, 32, src, src_stride, pred,
                       pred_stride);
  aom_hadamard_32x32_c(src_diff, 32, coeff);
  return aom_satd_c(coeff, 1024);
}

#if HAVE_NEON
int64_t av1_intra_model_rd_slow(const AV1_COMMON *cm, MACROBLOCK *const x,
                                int plane, BLOCK_SIZE plane_bsize,
                                TX_SIZE tx_size, int use_hadamard,
                                bool need_border_pad) {
  assert(plane >= 0 && plane < MAX_MB_PLANE);
  assert(plane_bsize < BLOCK_SIZES_ALL);
  assert(tx_size < TX_SIZES_ALL);
  assert(IMPLIES(use_hadamard, tx_size <= TX_32X32));
  MACROBLOCKD *const xd = &x->e_mbd;
  const struct macroblock_plane *const p = &x->plane[plane];
  const struct macroblockd_plane *const pd = &xd->plane[plane];
  const uint8_t *const src_buf = p->src.buf;
  const ptrdiff_t src_stride = p->src.stride;
  const uint8_t *const dst_buf = pd->dst.buf;
  const ptrdiff_t dst_stride = pd->dst.stride;
  const int stepr = tx_size_high_unit[tx_size];
  const int stepc = tx_size_wide_unit[tx_size];
  const int max_blocks_wide = max_block_wide(xd, plane_bsize, plane);
  const int max_blocks_high = max_block_high(xd, plane_bsize, plane);
  const bool is_hbd = is_cur_buf_hbd(xd);
  int64_t satd_cost = 0;

  if (use_hadamard && !need_border_pad && !is_hbd) {
    for (int row = 0; row < max_blocks_high; row += stepr) {
      for (int col = 0; col < max_blocks_wide; col += stepc) {
        av1_predict_intra_block_facade(cm, xd, plane, col, row, tx_size);
        satd_cost += quick_sub_hadamard_satd_lowbd(
            src_buf + (((row * src_stride) + col) << 2), src_stride,
            dst_buf + (((row * dst_stride) + col) << 2), dst_stride, tx_size);
      }
    }
    return satd_cost;
  }

  // Here we use p->src_diff and p->coeff as temporary buffers for
  // prediction residue and transform coefficients. The buffers are only
  // used in this for loop, therefore we don't need to properly add offset
  // to the buffers.
  const BitDepthInfo bd_info = get_bit_depth_info(xd);
  const int txbw = tx_size_wide[tx_size];
  const int txbh = tx_size_high[tx_size];
  const int diff_stride = block_size_wide[plane_bsize];
  int16_t *const src_diff = p->src_diff;
  tran_low_t *const coeff = p->coeff;
  const int num_coeffs = tx_size_2d[tx_size];
  for (int row = 0; row < max_blocks_high; row += stepr) {
    for (int col = 0; col < max_blocks_wide; col += stepc) {
      av1_predict_intra_block_facade(cm, xd, plane, col, row, tx_size);
      av1_subtract_block(
          x, txbh, txbw, src_diff, diff_stride,
          src_buf + (((row * src_stride) + col) << 2), src_stride,
          dst_buf + (((row * dst_stride) + col) << 2), dst_stride, plane,
          plane_bsize, col, row, DCT_DCT, need_border_pad);
      av1_quick_txfm(use_hadamard, tx_size, bd_info, src_diff, diff_stride,
                     coeff);
      satd_cost += aom_satd(coeff, num_coeffs);
    }
  }
  return satd_cost;
}
#endif  // HAVE_NEON
