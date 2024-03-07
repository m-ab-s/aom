/*
 * Copyright (c) 2024, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <assert.h>
#include <arm_neon.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"

#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/arm/aom_neon_sve_bridge.h"
#include "aom_dsp/arm/aom_neon_sve2_bridge.h"
#include "aom_dsp/arm/mem_neon.h"
#include "aom_ports/mem.h"
#include "av1/common/convolve.h"
#include "av1/common/filter.h"
#include "av1/common/filter.h"
#include "av1/common/arm/highbd_compound_convolve_neon.h"
#include "av1/common/arm/highbd_convolve_neon.h"

static INLINE uint16x8_t convolve8_8_x(int16x8_t s0[8], int16x8_t filter,
                                       int64x2_t offset, int32x4_t shift) {
  int64x2_t sum[8];
  sum[0] = aom_sdotq_s16(offset, s0[0], filter);
  sum[1] = aom_sdotq_s16(offset, s0[1], filter);
  sum[2] = aom_sdotq_s16(offset, s0[2], filter);
  sum[3] = aom_sdotq_s16(offset, s0[3], filter);
  sum[4] = aom_sdotq_s16(offset, s0[4], filter);
  sum[5] = aom_sdotq_s16(offset, s0[5], filter);
  sum[6] = aom_sdotq_s16(offset, s0[6], filter);
  sum[7] = aom_sdotq_s16(offset, s0[7], filter);

  sum[0] = vpaddq_s64(sum[0], sum[1]);
  sum[2] = vpaddq_s64(sum[2], sum[3]);
  sum[4] = vpaddq_s64(sum[4], sum[5]);
  sum[6] = vpaddq_s64(sum[6], sum[7]);

  int32x4_t sum0123 = vcombine_s32(vmovn_s64(sum[0]), vmovn_s64(sum[2]));
  int32x4_t sum4567 = vcombine_s32(vmovn_s64(sum[4]), vmovn_s64(sum[6]));

  sum0123 = vshlq_s32(sum0123, shift);
  sum4567 = vshlq_s32(sum4567, shift);

  return vcombine_u16(vqmovun_s32(sum0123), vqmovun_s32(sum4567));
}

static INLINE void highbd_dist_wtd_convolve_x_sve2(
    const uint16_t *src, int src_stride, uint16_t *dst, int dst_stride,
    int width, int height, const int16_t *x_filter_ptr,
    ConvolveParams *conv_params, const int offset) {
  const int32x4_t shift = vdupq_n_s32(-conv_params->round_0);
  const int64x2_t offset_vec = vdupq_n_s64(offset);

  const int64x2_t offset_lo =
      vcombine_s64(vget_low_s64(offset_vec), vdup_n_s64(0));
  const int16x8_t filter = vld1q_s16(x_filter_ptr);
  do {
    const int16_t *s = (const int16_t *)src;
    uint16_t *d = dst;
    int w = width;

    do {
      int16x8_t s0[8], s1[8], s2[8], s3[8];
      load_s16_8x8(s + 0 * src_stride, 1, &s0[0], &s0[1], &s0[2], &s0[3],
                   &s0[4], &s0[5], &s0[6], &s0[7]);
      load_s16_8x8(s + 1 * src_stride, 1, &s1[0], &s1[1], &s1[2], &s1[3],
                   &s1[4], &s1[5], &s1[6], &s1[7]);
      load_s16_8x8(s + 2 * src_stride, 1, &s2[0], &s2[1], &s2[2], &s2[3],
                   &s2[4], &s2[5], &s2[6], &s2[7]);
      load_s16_8x8(s + 3 * src_stride, 1, &s3[0], &s3[1], &s3[2], &s3[3],
                   &s3[4], &s3[5], &s3[6], &s3[7]);

      uint16x8_t d0 = convolve8_8_x(s0, filter, offset_lo, shift);
      uint16x8_t d1 = convolve8_8_x(s1, filter, offset_lo, shift);
      uint16x8_t d2 = convolve8_8_x(s2, filter, offset_lo, shift);
      uint16x8_t d3 = convolve8_8_x(s3, filter, offset_lo, shift);

      store_u16_8x4(d, dst_stride, d0, d1, d2, d3);

      s += 8;
      d += 8;
      w -= 8;
    } while (w != 0);
    src += 4 * src_stride;
    dst += 4 * dst_stride;
    height -= 4;
  } while (height != 0);
}

void av1_highbd_dist_wtd_convolve_x_sve2(
    const uint16_t *src, int src_stride, uint16_t *dst, int dst_stride, int w,
    int h, const InterpFilterParams *filter_params_x, const int subpel_x_qn,
    ConvolveParams *conv_params, int bd) {
  DECLARE_ALIGNED(16, uint16_t,
                  im_block[(MAX_SB_SIZE + MAX_FILTER_TAP) * MAX_SB_SIZE]);
  CONV_BUF_TYPE *dst16 = conv_params->dst;
  const int x_filter_taps = get_filter_tap(filter_params_x, subpel_x_qn);

  if (x_filter_taps != 8) {
    av1_highbd_dist_wtd_convolve_x_neon(src, src_stride, dst, dst_stride, w, h,
                                        filter_params_x, subpel_x_qn,
                                        conv_params, bd);
    return;
  }

  int dst16_stride = conv_params->dst_stride;
  const int im_stride = MAX_SB_SIZE;
  const int horiz_offset = filter_params_x->taps / 2 - 1;
  assert(FILTER_BITS == COMPOUND_ROUND1_BITS);
  const int offset_bits = bd + 2 * FILTER_BITS - conv_params->round_0;
  const int offset_avg = (1 << (offset_bits - conv_params->round_1)) +
                         (1 << (offset_bits - conv_params->round_1 - 1));
  const int offset_convolve = (1 << (conv_params->round_0 - 1)) +
                              (1 << (bd + FILTER_BITS)) +
                              (1 << (bd + FILTER_BITS - 1));

  const int16_t *x_filter_ptr = av1_get_interp_filter_subpel_kernel(
      filter_params_x, subpel_x_qn & SUBPEL_MASK);

  src -= horiz_offset;

  if (conv_params->do_average) {
    highbd_dist_wtd_convolve_x_sve2(src, src_stride, im_block, im_stride, w, h,
                                    x_filter_ptr, conv_params, offset_convolve);

    if (conv_params->use_dist_wtd_comp_avg) {
      if (bd == 12) {
        highbd_12_dist_wtd_comp_avg_neon(im_block, im_stride, dst, dst_stride,
                                         w, h, conv_params, offset_avg, bd);

      } else {
        highbd_dist_wtd_comp_avg_neon(im_block, im_stride, dst, dst_stride, w,
                                      h, conv_params, offset_avg, bd);
      }

    } else {
      if (bd == 12) {
        highbd_12_comp_avg_neon(im_block, im_stride, dst, dst_stride, w, h,
                                conv_params, offset_avg, bd);

      } else {
        highbd_comp_avg_neon(im_block, im_stride, dst, dst_stride, w, h,
                             conv_params, offset_avg, bd);
      }
    }
  } else {
    highbd_dist_wtd_convolve_x_sve2(src, src_stride, dst16, dst16_stride, w, h,
                                    x_filter_ptr, conv_params, offset_convolve);
  }
}
