/*
 * Copyright (c) 2023, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#ifndef AOM_AV1_COMMON_ARM_HIGHBD_CONVOLVE_NEON_H_
#define AOM_AV1_COMMON_ARM_HIGHBD_CONVOLVE_NEON_H_

#include <arm_neon.h>

#include "aom_dsp/arm/mem_neon.h"
#include "aom_dsp/arm/transpose_neon.h"
#include "av1/common/convolve.h"

static INLINE int32x4_t highbd_convolve8_4_s32(
    const int16x4_t s0, const int16x4_t s1, const int16x4_t s2,
    const int16x4_t s3, const int16x4_t s4, const int16x4_t s5,
    const int16x4_t s6, const int16x4_t s7, const int16x8_t y_filter,
    const int32x4_t offset) {
  const int16x4_t y_filter_lo = vget_low_s16(y_filter);
  const int16x4_t y_filter_hi = vget_high_s16(y_filter);

  int32x4_t sum = vmlal_lane_s16(offset, s0, y_filter_lo, 0);
  sum = vmlal_lane_s16(sum, s1, y_filter_lo, 1);
  sum = vmlal_lane_s16(sum, s2, y_filter_lo, 2);
  sum = vmlal_lane_s16(sum, s3, y_filter_lo, 3);
  sum = vmlal_lane_s16(sum, s4, y_filter_hi, 0);
  sum = vmlal_lane_s16(sum, s5, y_filter_hi, 1);
  sum = vmlal_lane_s16(sum, s6, y_filter_hi, 2);
  sum = vmlal_lane_s16(sum, s7, y_filter_hi, 3);

  return sum;
}

static INLINE uint16x4_t highbd_convolve8_4_sr_s32_s16(
    const int16x4_t s0, const int16x4_t s1, const int16x4_t s2,
    const int16x4_t s3, const int16x4_t s4, const int16x4_t s5,
    const int16x4_t s6, const int16x4_t s7, const int16x8_t y_filter,
    const int32x4_t shift_s32, const int32x4_t offset) {
  int32x4_t sum =
      highbd_convolve8_4_s32(s0, s1, s2, s3, s4, s5, s6, s7, y_filter, offset);

  sum = vqrshlq_s32(sum, shift_s32);
  return vqmovun_s32(sum);
}

// Like above but also perform round shifting and subtract correction term
static INLINE uint16x4_t highbd_convolve8_4_srsub_s32_s16(
    const int16x4_t s0, const int16x4_t s1, const int16x4_t s2,
    const int16x4_t s3, const int16x4_t s4, const int16x4_t s5,
    const int16x4_t s6, const int16x4_t s7, const int16x8_t y_filter,
    const int32x4_t round_shift, const int32x4_t offset,
    const int32x4_t correction) {
  int32x4_t sum =
      highbd_convolve8_4_s32(s0, s1, s2, s3, s4, s5, s6, s7, y_filter, offset);

  sum = vsubq_s32(vqrshlq_s32(sum, round_shift), correction);
  return vqmovun_s32(sum);
}

static INLINE void highbd_convolve8_8_s32(
    const int16x8_t s0, const int16x8_t s1, const int16x8_t s2,
    const int16x8_t s3, const int16x8_t s4, const int16x8_t s5,
    const int16x8_t s6, const int16x8_t s7, const int16x8_t y_filter,
    const int32x4_t offset, int32x4_t *sum0, int32x4_t *sum1) {
  const int16x4_t y_filter_lo = vget_low_s16(y_filter);
  const int16x4_t y_filter_hi = vget_high_s16(y_filter);

  *sum0 = vmlal_lane_s16(offset, vget_low_s16(s0), y_filter_lo, 0);
  *sum0 = vmlal_lane_s16(*sum0, vget_low_s16(s1), y_filter_lo, 1);
  *sum0 = vmlal_lane_s16(*sum0, vget_low_s16(s2), y_filter_lo, 2);
  *sum0 = vmlal_lane_s16(*sum0, vget_low_s16(s3), y_filter_lo, 3);
  *sum0 = vmlal_lane_s16(*sum0, vget_low_s16(s4), y_filter_hi, 0);
  *sum0 = vmlal_lane_s16(*sum0, vget_low_s16(s5), y_filter_hi, 1);
  *sum0 = vmlal_lane_s16(*sum0, vget_low_s16(s6), y_filter_hi, 2);
  *sum0 = vmlal_lane_s16(*sum0, vget_low_s16(s7), y_filter_hi, 3);

  *sum1 = vmlal_lane_s16(offset, vget_high_s16(s0), y_filter_lo, 0);
  *sum1 = vmlal_lane_s16(*sum1, vget_high_s16(s1), y_filter_lo, 1);
  *sum1 = vmlal_lane_s16(*sum1, vget_high_s16(s2), y_filter_lo, 2);
  *sum1 = vmlal_lane_s16(*sum1, vget_high_s16(s3), y_filter_lo, 3);
  *sum1 = vmlal_lane_s16(*sum1, vget_high_s16(s4), y_filter_hi, 0);
  *sum1 = vmlal_lane_s16(*sum1, vget_high_s16(s5), y_filter_hi, 1);
  *sum1 = vmlal_lane_s16(*sum1, vget_high_s16(s6), y_filter_hi, 2);
  *sum1 = vmlal_lane_s16(*sum1, vget_high_s16(s7), y_filter_hi, 3);
}

// Like above but also perform round shifting and subtract correction term
static INLINE uint16x8_t highbd_convolve8_8_srsub_s32_s16(
    const int16x8_t s0, const int16x8_t s1, const int16x8_t s2,
    const int16x8_t s3, const int16x8_t s4, const int16x8_t s5,
    const int16x8_t s6, const int16x8_t s7, const int16x8_t y_filter,
    const int32x4_t round_shift, const int32x4_t offset,
    const int32x4_t correction) {
  int32x4_t sum0;
  int32x4_t sum1;
  highbd_convolve8_8_s32(s0, s1, s2, s3, s4, s5, s6, s7, y_filter, offset,
                         &sum0, &sum1);

  sum0 = vsubq_s32(vqrshlq_s32(sum0, round_shift), correction);
  sum1 = vsubq_s32(vqrshlq_s32(sum1, round_shift), correction);

  return vcombine_u16(vqmovun_s32(sum0), vqmovun_s32(sum1));
}

static INLINE int32x4_t highbd_convolve8_2d_scale_horiz4x8_s32(
    const int16x8_t s0, const int16x8_t s1, const int16x8_t s2,
    const int16x8_t s3, const int16x4_t *filters_lo,
    const int16x4_t *filters_hi, const int32x4_t offset) {
  int16x4_t s_lo[] = { vget_low_s16(s0), vget_low_s16(s1), vget_low_s16(s2),
                       vget_low_s16(s3) };
  int16x4_t s_hi[] = { vget_high_s16(s0), vget_high_s16(s1), vget_high_s16(s2),
                       vget_high_s16(s3) };

  transpose_array_inplace_u16_4x4((uint16x4_t *)s_lo);
  transpose_array_inplace_u16_4x4((uint16x4_t *)s_hi);

  int32x4_t sum = vmlal_s16(offset, s_lo[0], filters_lo[0]);
  sum = vmlal_s16(sum, s_lo[1], filters_lo[1]);
  sum = vmlal_s16(sum, s_lo[2], filters_lo[2]);
  sum = vmlal_s16(sum, s_lo[3], filters_lo[3]);
  sum = vmlal_s16(sum, s_hi[0], filters_hi[0]);
  sum = vmlal_s16(sum, s_hi[1], filters_hi[1]);
  sum = vmlal_s16(sum, s_hi[2], filters_hi[2]);
  sum = vmlal_s16(sum, s_hi[3], filters_hi[3]);

  return sum;
}

static INLINE uint16x4_t highbd_convolve8_2d_scale_horiz4x8_s32_s16(
    const int16x8_t s0, const int16x8_t s1, const int16x8_t s2,
    const int16x8_t s3, const int16x4_t *filters_lo,
    const int16x4_t *filters_hi, const int32x4_t shift_s32,
    const int32x4_t offset) {
  int32x4_t sum = highbd_convolve8_2d_scale_horiz4x8_s32(
      s0, s1, s2, s3, filters_lo, filters_hi, offset);

  sum = vqrshlq_s32(sum, shift_s32);
  return vqmovun_s32(sum);
}

static INLINE int32x4_t highbd_convolve8_horiz4x8_s32(
    const int16x8_t s0, const int16x8_t s1, const int16x8_t s2,
    const int16x8_t s3, const int16x8_t x_filter_0_7, const int32x4_t offset) {
  int16x4_t s_lo[] = { vget_low_s16(s0), vget_low_s16(s1), vget_low_s16(s2),
                       vget_low_s16(s3) };
  int16x4_t s_hi[] = { vget_high_s16(s0), vget_high_s16(s1), vget_high_s16(s2),
                       vget_high_s16(s3) };

  transpose_array_inplace_u16_4x4((uint16x4_t *)s_lo);
  transpose_array_inplace_u16_4x4((uint16x4_t *)s_hi);
  const int16x4_t x_filter_0_3 = vget_low_s16(x_filter_0_7);
  const int16x4_t x_filter_4_7 = vget_high_s16(x_filter_0_7);

  int32x4_t sum = vmlal_lane_s16(offset, s_lo[0], x_filter_0_3, 0);
  sum = vmlal_lane_s16(sum, s_lo[1], x_filter_0_3, 1);
  sum = vmlal_lane_s16(sum, s_lo[2], x_filter_0_3, 2);
  sum = vmlal_lane_s16(sum, s_lo[3], x_filter_0_3, 3);
  sum = vmlal_lane_s16(sum, s_hi[0], x_filter_4_7, 0);
  sum = vmlal_lane_s16(sum, s_hi[1], x_filter_4_7, 1);
  sum = vmlal_lane_s16(sum, s_hi[2], x_filter_4_7, 2);
  sum = vmlal_lane_s16(sum, s_hi[3], x_filter_4_7, 3);

  return sum;
}

static INLINE uint16x4_t highbd_convolve8_horiz4x8_s32_s16(
    const int16x8_t s0, const int16x8_t s1, const int16x8_t s2,
    const int16x8_t s3, const int16x8_t x_filters_0_7,
    const int32x4_t shift_s32, const int32x4_t offset) {
  int32x4_t sum =
      highbd_convolve8_horiz4x8_s32(s0, s1, s2, s3, x_filters_0_7, offset);

  sum = vqrshlq_s32(sum, shift_s32);
  return vqmovun_s32(sum);
}

static INLINE void highbd_comp_avg_neon(const uint16_t *src_ptr, int src_stride,
                                        uint16_t *dst_ptr, int dst_stride,
                                        int w, int h,
                                        ConvolveParams *conv_params,
                                        const int round_bits, const int offset,
                                        const int bd) {
  CONV_BUF_TYPE *dst16 = conv_params->dst;
  const int dst16_stride = conv_params->dst_stride;
  const int32x4_t round_shift_s32 = vdupq_n_s32(-round_bits);
  const int16x4_t offset_s16 = vdup_n_s16(offset);
  const uint16x8_t max = vdupq_n_u16((1 << bd) - 1);

  if (w <= 4) {
    for (int y = 0; y < h; ++y) {
      const uint16x4_t s = vld1_u16(src_ptr + y * src_stride);
      const uint16x4_t d16 = vld1_u16(dst16 + y * dst16_stride);
      int32x4_t s_s32 = vreinterpretq_s32_u32(vmovl_u16(s));
      int32x4_t d16_s32 = vreinterpretq_s32_u32(vmovl_u16(d16));
      int32x4_t d0 = vhaddq_s32(s_s32, d16_s32);
      d0 = vsubw_s16(d0, offset_s16);
      d0 = vqrshlq_s32(d0, round_shift_s32);
      uint16x4_t d = vqmovun_s32(d0);
      d = vmin_u16(d, vget_low_u16(max));
      if (w == 2) {
        store_u16_2x1(dst_ptr + y * dst_stride, d, 0);
      } else {
        vst1_u16(dst_ptr + y * dst_stride, d);
      }
    }
  } else {
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; x += 8) {
        const uint16x8_t s = vld1q_u16(src_ptr + y * src_stride + x);
        const uint16x8_t d16 = vld1q_u16(dst16 + y * dst16_stride + x);
        int32x4_t s_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(s)));
        int32x4_t s_hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(s)));
        int32x4_t d16_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(d16)));
        int32x4_t d16_hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(d16)));
        int32x4_t d0 = vhaddq_s32(s_lo, d16_lo);
        int32x4_t d1 = vhaddq_s32(s_hi, d16_hi);
        d0 = vsubw_s16(d0, offset_s16);
        d1 = vsubw_s16(d1, offset_s16);
        d0 = vqrshlq_s32(d0, round_shift_s32);
        d1 = vqrshlq_s32(d1, round_shift_s32);
        uint16x8_t d01 = vcombine_u16(vqmovun_s32(d0), vqmovun_s32(d1));
        d01 = vminq_u16(d01, max);
        vst1q_u16(dst_ptr + y * dst_stride + x, d01);
      }
    }
  }
}

#endif  // AOM_AV1_COMMON_ARM_HIGHBD_CONVOLVE_NEON_H_
