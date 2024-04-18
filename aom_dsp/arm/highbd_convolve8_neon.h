/*
 *  Copyright (c) 2024, Alliance for Open Media. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef AOM_AOM_DSP_ARM_HIGHBD_CONVOLVE8_NEON_H_
#define AOM_AOM_DSP_ARM_HIGHBD_CONVOLVE8_NEON_H_

#include <arm_neon.h>

#include "config/aom_config.h"
#include "aom_dsp/arm/mem_neon.h"

static INLINE void highbd_convolve8_horiz_2tap_neon(
    const uint16_t *src_ptr, ptrdiff_t src_stride, uint16_t *dst_ptr,
    ptrdiff_t dst_stride, const int16_t *x_filter_ptr, int w, int h, int bd) {
  // Bilinear filter values are all positive and multiples of 8. Divide by 8 to
  // reduce intermediate precision requirements and allow the use of non
  // widening multiply.
  const uint16x8_t f0 = vdupq_n_u16((uint16_t)x_filter_ptr[3] / 8);
  const uint16x8_t f1 = vdupq_n_u16((uint16_t)x_filter_ptr[4] / 8);

  const uint16x8_t max = vdupq_n_u16((1 << bd) - 1);

  if (w == 4) {
    do {
      uint16x8_t s0 =
          load_unaligned_u16_4x2(src_ptr + 0 * src_stride + 0, (int)src_stride);
      uint16x8_t s1 =
          load_unaligned_u16_4x2(src_ptr + 0 * src_stride + 1, (int)src_stride);
      uint16x8_t s2 =
          load_unaligned_u16_4x2(src_ptr + 2 * src_stride + 0, (int)src_stride);
      uint16x8_t s3 =
          load_unaligned_u16_4x2(src_ptr + 2 * src_stride + 1, (int)src_stride);

      uint16x8_t sum01 = vmulq_u16(s0, f0);
      sum01 = vmlaq_u16(sum01, s1, f1);
      uint16x8_t sum23 = vmulq_u16(s2, f0);
      sum23 = vmlaq_u16(sum23, s3, f1);

      // We divided filter taps by 8 so subtract 3 from right shift.
      sum01 = vrshrq_n_u16(sum01, FILTER_BITS - 3);
      sum23 = vrshrq_n_u16(sum23, FILTER_BITS - 3);

      sum01 = vminq_u16(sum01, max);
      sum23 = vminq_u16(sum23, max);

      store_u16x4_strided_x2(dst_ptr + 0 * dst_stride, (int)dst_stride, sum01);
      store_u16x4_strided_x2(dst_ptr + 2 * dst_stride, (int)dst_stride, sum23);

      src_ptr += 4 * src_stride;
      dst_ptr += 4 * dst_stride;
      h -= 4;
    } while (h > 0);
  } else {
    do {
      int width = w;
      const uint16_t *s = src_ptr;
      uint16_t *d = dst_ptr;

      do {
        uint16x8_t s0 = vld1q_u16(s + 0 * src_stride + 0);
        uint16x8_t s1 = vld1q_u16(s + 0 * src_stride + 1);
        uint16x8_t s2 = vld1q_u16(s + 1 * src_stride + 0);
        uint16x8_t s3 = vld1q_u16(s + 1 * src_stride + 1);

        uint16x8_t sum01 = vmulq_u16(s0, f0);
        sum01 = vmlaq_u16(sum01, s1, f1);
        uint16x8_t sum23 = vmulq_u16(s2, f0);
        sum23 = vmlaq_u16(sum23, s3, f1);

        // We divided filter taps by 8 so subtract 3 from right shift.
        sum01 = vrshrq_n_u16(sum01, FILTER_BITS - 3);
        sum23 = vrshrq_n_u16(sum23, FILTER_BITS - 3);

        sum01 = vminq_u16(sum01, max);
        sum23 = vminq_u16(sum23, max);

        vst1q_u16(d + 0 * dst_stride, sum01);
        vst1q_u16(d + 1 * dst_stride, sum23);

        s += 8;
        d += 8;
        width -= 8;
      } while (width != 0);
      src_ptr += 2 * src_stride;
      dst_ptr += 2 * dst_stride;
      h -= 2;
    } while (h > 0);
  }
}

#endif  // AOM_AOM_DSP_ARM_HIGHBD_CONVOLVE8_NEON_H_
