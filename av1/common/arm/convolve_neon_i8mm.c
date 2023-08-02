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

#include <arm_neon.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"

#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/arm/mem_neon.h"
#include "aom_ports/mem.h"
#include "av1/common/convolve.h"
#include "av1/common/filter.h"

DECLARE_ALIGNED(16, static const uint8_t, dot_prod_permute_tbl[48]) = {
  0, 1, 2,  3,  1, 2,  3,  4,  2,  3,  4,  5,  3,  4,  5,  6,
  4, 5, 6,  7,  5, 6,  7,  8,  6,  7,  8,  9,  7,  8,  9,  10,
  8, 9, 10, 11, 9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14
};

static INLINE int16x4_t convolve12_4_x(uint8x16_t samples,
                                       const int8x16_t filter,
                                       const uint8x16x3_t permute_tbl,
                                       const int32x4_t horiz_const) {
  uint8x16_t permuted_samples[3];
  int32x4_t sum;

  // Permute samples ready for dot product.
  // { 0,  1,  2,  3,  1,  2,  3,  4,  2,  3,  4,  5,  3,  4,  5,  6 }
  permuted_samples[0] = vqtbl1q_u8(samples, permute_tbl.val[0]);
  // { 4,  5,  6,  7,  5,  6,  7,  8,  6,  7,  8,  9,  7,  8,  9, 10 }
  permuted_samples[1] = vqtbl1q_u8(samples, permute_tbl.val[1]);
  // { 8,  9, 10, 11,  9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14 }
  permuted_samples[2] = vqtbl1q_u8(samples, permute_tbl.val[2]);

  // First 4 output values.
  sum = vusdotq_laneq_s32(horiz_const, permuted_samples[0], filter, 0);
  sum = vusdotq_laneq_s32(sum, permuted_samples[1], filter, 1);
  sum = vusdotq_laneq_s32(sum, permuted_samples[2], filter, 2);

  return vqrshrn_n_s32(sum, FILTER_BITS);
}

static INLINE uint8x8_t convolve12_8_x(uint8x16_t samples[2],
                                       const int8x16_t filter,
                                       const uint8x16x3_t permute_tbl,
                                       const int32x4_t horiz_const) {
  uint8x16_t permuted_samples[4];
  int32x4_t sum[2];

  // Permute samples ready for dot product.
  // { 0,  1,  2,  3,  1,  2,  3,  4,  2,  3,  4,  5,  3,  4,  5,  6 }
  permuted_samples[0] = vqtbl1q_u8(samples[0], permute_tbl.val[0]);
  // { 4,  5,  6,  7,  5,  6,  7,  8,  6,  7,  8,  9,  7,  8,  9, 10 }
  permuted_samples[1] = vqtbl1q_u8(samples[0], permute_tbl.val[1]);
  // { 8,  9, 10, 11,  9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14 }
  permuted_samples[2] = vqtbl1q_u8(samples[0], permute_tbl.val[2]);
  // {12, 13, 14, 15, 13, 14, 15, 16, 14, 15, 16, 17, 15, 16, 17, 18 }
  permuted_samples[3] = vqtbl1q_u8(samples[1], permute_tbl.val[2]);

  // First 4 output values.
  sum[0] = vusdotq_laneq_s32(horiz_const, permuted_samples[0], filter, 0);
  sum[0] = vusdotq_laneq_s32(sum[0], permuted_samples[1], filter, 1);
  sum[0] = vusdotq_laneq_s32(sum[0], permuted_samples[2], filter, 2);
  // Second 4 output values.
  sum[1] = vusdotq_laneq_s32(horiz_const, permuted_samples[1], filter, 0);
  sum[1] = vusdotq_laneq_s32(sum[1], permuted_samples[2], filter, 1);
  sum[1] = vusdotq_laneq_s32(sum[1], permuted_samples[3], filter, 2);

  // Narrow and re-pack.
  int16x8_t sum_s16 = vcombine_s16(vqrshrn_n_s32(sum[0], FILTER_BITS),
                                   vqrshrn_n_s32(sum[1], FILTER_BITS));
  return vqmovun_s16(sum_s16);
}

static INLINE void convolve_x_sr_12tap_neon_i8mm(const uint8_t *src,
                                                 int src_stride, uint8_t *dst,
                                                 int dst_stride, int w, int h,
                                                 const int16_t *x_filter_ptr) {
  const int16x8_t filter_0_7 = vld1q_s16(x_filter_ptr);
  const int16x4_t filter_8_11 = vld1_s16(x_filter_ptr + 8);
  const int16x8_t filter_8_15 = vcombine_s16(filter_8_11, vdup_n_s16(0));
  const int8x16_t filter =
      vcombine_s8(vmovn_s16(filter_0_7), vmovn_s16(filter_8_15));

  // Special case the following no-op filter as 128 won't fit into the
  // 8-bit signed dot-product instruction:
  // { 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0 }
  if (vgetq_lane_s16(filter_0_7, 5) == 128) {
    // Undo the horizontal offset in the calling function.
    src += 5;

    do {
      const uint8_t *s = src;
      uint8_t *d = dst;
      int width = w;

      do {
        uint8x8_t d0 = vld1_u8(s);
        if (w == 2) {
          store_u8_2x1(d, d0, 0);
        } else if (w == 4) {
          store_u8_4x1(d, d0, 0);
        } else {
          vst1_u8(d, d0);
        }

        s += 8;
        d += 8;
        width -= 8;
      } while (width > 0);
      src += src_stride;
      dst += dst_stride;
    } while (--h != 0);
  } else {
    const uint8x16x3_t permute_tbl = vld1q_u8_x3(dot_prod_permute_tbl);
    // This shim of 1 << (ROUND0_BITS - 1) enables us to use a single rounding
    // right shift by FILTER_BITS - instead of a first rounding right shift by
    // ROUND0_BITS, followed by second rounding right shift by FILTER_BITS -
    // ROUND0_BITS.
    const int32x4_t horiz_const = vdupq_n_s32(1 << (ROUND0_BITS - 1));

    if (w <= 4) {
      do {
        uint8x16_t s0, s1, s2, s3;
        load_u8_16x4(src, src_stride, &s0, &s1, &s2, &s3);

        int16x4_t d0 = convolve12_4_x(s0, filter, permute_tbl, horiz_const);
        int16x4_t d1 = convolve12_4_x(s1, filter, permute_tbl, horiz_const);
        int16x4_t d2 = convolve12_4_x(s2, filter, permute_tbl, horiz_const);
        int16x4_t d3 = convolve12_4_x(s3, filter, permute_tbl, horiz_const);

        uint8x8_t d01 = vqmovun_s16(vcombine_s16(d0, d1));
        uint8x8_t d23 = vqmovun_s16(vcombine_s16(d2, d3));

        if (w == 2) {
          store_u8_2x1(dst + 0 * dst_stride, d01, 0);
          store_u8_2x1(dst + 1 * dst_stride, d01, 2);
          if (h != 2) {
            store_u8_2x1(dst + 2 * dst_stride, d23, 0);
            store_u8_2x1(dst + 3 * dst_stride, d23, 2);
          }
        } else {
          store_u8_4x1(dst + 0 * dst_stride, d01, 0);
          store_u8_4x1(dst + 1 * dst_stride, d01, 1);
          if (h != 2) {
            store_u8_4x1(dst + 2 * dst_stride, d23, 0);
            store_u8_4x1(dst + 3 * dst_stride, d23, 1);
          }
        }

        dst += 4 * dst_stride;
        src += 4 * src_stride;
        h -= 4;
      } while (h > 0);
    } else {
      do {
        const uint8_t *s = src;
        uint8_t *d = dst;
        int width = w;

        do {
          uint8x16_t s0[2], s1[2], s2[2], s3[2];
          load_u8_16x4(s, src_stride, &s0[0], &s1[0], &s2[0], &s3[0]);
          load_u8_16x4(s + 4, src_stride, &s0[1], &s1[1], &s2[1], &s3[1]);

          uint8x8_t d0 = convolve12_8_x(s0, filter, permute_tbl, horiz_const);
          uint8x8_t d1 = convolve12_8_x(s1, filter, permute_tbl, horiz_const);
          uint8x8_t d2 = convolve12_8_x(s2, filter, permute_tbl, horiz_const);
          uint8x8_t d3 = convolve12_8_x(s3, filter, permute_tbl, horiz_const);

          store_u8_8x2(d + 0 * dst_stride, dst_stride, d0, d1);
          if (h != 2) {
            store_u8_8x2(d + 2 * dst_stride, dst_stride, d2, d3);
          }

          s += 8;
          d += 8;
          width -= 8;
        } while (width > 0);
        src += 4 * src_stride;
        dst += 4 * dst_stride;
        h -= 4;
      } while (h > 0);
    }
  }
}

static INLINE int16x4_t convolve4_4_x(uint8x16_t samples, const int8x8_t filter,
                                      const uint8x16_t permute_tbl,
                                      const int32x4_t horiz_const) {
  // Permute samples ready for dot product.
  // { 0,  1,  2,  3,  1,  2,  3,  4,  2,  3,  4,  5,  3,  4,  5,  6 }
  uint8x16_t permuted_samples = vqtbl1q_u8(samples, permute_tbl);

  // First 4 output values.
  int32x4_t sum = vusdotq_lane_s32(horiz_const, permuted_samples, filter, 0);

  // Packing is performed by the caller.
  return vmovn_s32(sum);
}

static INLINE uint8x8_t convolve8_8_x(uint8x16_t samples, const int8x8_t filter,
                                      const uint8x16x3_t permute_tbl,
                                      const int32x4_t horiz_const) {
  uint8x16_t permuted_samples[3];
  int32x4_t sum[2];

  // Permute samples ready for dot product.
  // { 0,  1,  2,  3,  1,  2,  3,  4,  2,  3,  4,  5,  3,  4,  5,  6 }
  permuted_samples[0] = vqtbl1q_u8(samples, permute_tbl.val[0]);
  // { 4,  5,  6,  7,  5,  6,  7,  8,  6,  7,  8,  9,  7,  8,  9, 10 }
  permuted_samples[1] = vqtbl1q_u8(samples, permute_tbl.val[1]);
  // { 8,  9, 10, 11,  9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14 }
  permuted_samples[2] = vqtbl1q_u8(samples, permute_tbl.val[2]);

  // First 4 output values.
  sum[0] = vusdotq_lane_s32(horiz_const, permuted_samples[0], filter, 0);
  sum[0] = vusdotq_lane_s32(sum[0], permuted_samples[1], filter, 1);
  // Second 4 output values.
  sum[1] = vusdotq_lane_s32(horiz_const, permuted_samples[1], filter, 0);
  sum[1] = vusdotq_lane_s32(sum[1], permuted_samples[2], filter, 1);

  int16x8_t sum_s16 = vcombine_s16(vmovn_s32(sum[0]), vmovn_s32(sum[1]));
  // We halved the convolution filter values so - 1 from the right shift.
  return vqrshrun_n_s16(sum_s16, FILTER_BITS - 1);
}

void av1_convolve_x_sr_neon_i8mm(const uint8_t *src, int src_stride,
                                 uint8_t *dst, int dst_stride, int w, int h,
                                 const InterpFilterParams *filter_params_x,
                                 const int subpel_x_qn,
                                 ConvolveParams *conv_params) {
  (void)conv_params;
  const uint8_t horiz_offset = filter_params_x->taps / 2 - 1;
  src -= horiz_offset;

  const int16_t *x_filter_ptr = av1_get_interp_filter_subpel_kernel(
      filter_params_x, subpel_x_qn & SUBPEL_MASK);

  if (filter_params_x->taps > 8) {
    convolve_x_sr_12tap_neon_i8mm(src, src_stride, dst, dst_stride, w, h,
                                  x_filter_ptr);
    return;
  }

  // This shim of 1 << ((ROUND0_BITS - 1) - 1) enables us to use a single
  // rounding right shift by FILTER_BITS - instead of a first rounding right
  // shift by ROUND0_BITS, followed by second rounding right shift by
  // FILTER_BITS - ROUND0_BITS.
  // The outermost -1 is needed because we will halve the filter values.
  const int32x4_t horiz_const = vdupq_n_s32(1 << ((ROUND0_BITS - 1) - 1));

  if (w <= 4) {
    const uint8x16_t permute_tbl = vld1q_u8(dot_prod_permute_tbl);
    // 4-tap filters are used for blocks having width <= 4.
    // Filter values are even, so halve to reduce intermediate precision reqs.
    const int8x8_t x_filter =
        vshrn_n_s16(vcombine_s16(vld1_s16(x_filter_ptr + 2), vdup_n_s16(0)), 1);

    src += 2;

    do {
      uint8x16_t s0, s1, s2, s3;
      load_u8_16x4(src, src_stride, &s0, &s1, &s2, &s3);

      int16x4_t d0 = convolve4_4_x(s0, x_filter, permute_tbl, horiz_const);
      int16x4_t d1 = convolve4_4_x(s1, x_filter, permute_tbl, horiz_const);
      int16x4_t d2 = convolve4_4_x(s2, x_filter, permute_tbl, horiz_const);
      int16x4_t d3 = convolve4_4_x(s3, x_filter, permute_tbl, horiz_const);

      // We halved the convolution filter values so - 1 from the right shift.
      uint8x8_t d01 = vqrshrun_n_s16(vcombine_s16(d0, d1), FILTER_BITS - 1);
      uint8x8_t d23 = vqrshrun_n_s16(vcombine_s16(d2, d3), FILTER_BITS - 1);

      if (w == 2) {
        store_u8_2x1(dst + 0 * dst_stride, d01, 0);
        store_u8_2x1(dst + 1 * dst_stride, d01, 2);
        if (h != 2) {
          store_u8_2x1(dst + 2 * dst_stride, d23, 0);
          store_u8_2x1(dst + 3 * dst_stride, d23, 2);
        }
      } else {
        store_u8_4x1(dst + 0 * dst_stride, d01, 0);
        store_u8_4x1(dst + 1 * dst_stride, d01, 1);
        if (h != 2) {
          store_u8_4x1(dst + 2 * dst_stride, d23, 0);
          store_u8_4x1(dst + 3 * dst_stride, d23, 1);
        }
      }

      src += 4 * src_stride;
      dst += 4 * dst_stride;
      h -= 4;
    } while (h > 0);

  } else {
    const uint8x16x3_t permute_tbl = vld1q_u8_x3(dot_prod_permute_tbl);
    // Filter values are even, so halve to reduce intermediate precision reqs.
    const int8x8_t x_filter = vshrn_n_s16(vld1q_s16(x_filter_ptr), 1);

    do {
      const uint8_t *s = src;
      uint8_t *d = dst;
      int width = w;

      do {
        uint8x16_t s0, s1, s2, s3;
        load_u8_16x4(s, src_stride, &s0, &s1, &s2, &s3);

        uint8x8_t d0 = convolve8_8_x(s0, x_filter, permute_tbl, horiz_const);
        uint8x8_t d1 = convolve8_8_x(s1, x_filter, permute_tbl, horiz_const);
        uint8x8_t d2 = convolve8_8_x(s2, x_filter, permute_tbl, horiz_const);
        uint8x8_t d3 = convolve8_8_x(s3, x_filter, permute_tbl, horiz_const);

        vst1_u8(d + 0 * dst_stride, d0);
        vst1_u8(d + 1 * dst_stride, d1);
        if (h != 2) {
          vst1_u8(d + 2 * dst_stride, d2);
          vst1_u8(d + 3 * dst_stride, d3);
        }

        s += 8;
        d += 8;
        width -= 8;
      } while (width > 0);
      src += 4 * src_stride;
      dst += 4 * dst_stride;
      h -= 4;
    } while (h > 0);
  }
}
