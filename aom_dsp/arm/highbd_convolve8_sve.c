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

#include <arm_neon.h>
#include <assert.h>
#include <stdint.h>

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"

#include "aom_dsp/arm/dot_sve.h"
#include "aom_dsp/arm/mem_neon.h"

static INLINE uint16x4_t highbd_convolve8_4_h(int16x8_t s[4],
                                              int16x8_t filter) {
  int64x2_t sum[4];

  sum[0] = aom_sdotq_s16(vdupq_n_s64(0), s[0], filter);
  sum[1] = aom_sdotq_s16(vdupq_n_s64(0), s[1], filter);
  sum[2] = aom_sdotq_s16(vdupq_n_s64(0), s[2], filter);
  sum[3] = aom_sdotq_s16(vdupq_n_s64(0), s[3], filter);

  int64x2_t sum01 = vpaddq_s64(sum[0], sum[1]);
  int64x2_t sum23 = vpaddq_s64(sum[2], sum[3]);

  int32x4_t res = vcombine_s32(vmovn_s64(sum01), vmovn_s64(sum23));

  return vqrshrun_n_s32(res, FILTER_BITS);
}

static INLINE uint16x8_t highbd_convolve8_8_h(int16x8_t s[8],
                                              int16x8_t filter) {
  int64x2_t sum[8];

  sum[0] = aom_sdotq_s16(vdupq_n_s64(0), s[0], filter);
  sum[1] = aom_sdotq_s16(vdupq_n_s64(0), s[1], filter);
  sum[2] = aom_sdotq_s16(vdupq_n_s64(0), s[2], filter);
  sum[3] = aom_sdotq_s16(vdupq_n_s64(0), s[3], filter);
  sum[4] = aom_sdotq_s16(vdupq_n_s64(0), s[4], filter);
  sum[5] = aom_sdotq_s16(vdupq_n_s64(0), s[5], filter);
  sum[6] = aom_sdotq_s16(vdupq_n_s64(0), s[6], filter);
  sum[7] = aom_sdotq_s16(vdupq_n_s64(0), s[7], filter);

  int64x2_t sum01 = vpaddq_s64(sum[0], sum[1]);
  int64x2_t sum23 = vpaddq_s64(sum[2], sum[3]);
  int64x2_t sum45 = vpaddq_s64(sum[4], sum[5]);
  int64x2_t sum67 = vpaddq_s64(sum[6], sum[7]);

  int32x4_t res0 = vcombine_s32(vmovn_s64(sum01), vmovn_s64(sum23));
  int32x4_t res1 = vcombine_s32(vmovn_s64(sum45), vmovn_s64(sum67));

  return vcombine_u16(vqrshrun_n_s32(res0, FILTER_BITS),
                      vqrshrun_n_s32(res1, FILTER_BITS));
}

void aom_highbd_convolve8_horiz_sve(const uint8_t *src8, ptrdiff_t src_stride,
                                    uint8_t *dst8, ptrdiff_t dst_stride,
                                    const int16_t *filter_x, int x_step_q4,
                                    const int16_t *filter_y, int y_step_q4,
                                    int width, int height, int bd) {
  assert(x_step_q4 == 16);
  assert(width >= 4 && height >= 4);
  (void)filter_y;
  (void)x_step_q4;
  (void)y_step_q4;

  uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  uint16_t *dst = CONVERT_TO_SHORTPTR(dst8);

  src -= SUBPEL_TAPS / 2 - 1;
  const uint16x8_t max = vdupq_n_u16((1 << bd) - 1);
  const int16x8_t filter = vld1q_s16(filter_x);

  if (width == 4) {
    const int16_t *s = (const int16_t *)src;
    uint16_t *d = dst;

    do {
      int16x8_t s0[4], s1[4], s2[4], s3[4];
      load_s16_8x4(s + 0 * src_stride, 1, &s0[0], &s0[1], &s0[2], &s0[3]);
      load_s16_8x4(s + 1 * src_stride, 1, &s1[0], &s1[1], &s1[2], &s1[3]);
      load_s16_8x4(s + 2 * src_stride, 1, &s2[0], &s2[1], &s2[2], &s2[3]);
      load_s16_8x4(s + 3 * src_stride, 1, &s3[0], &s3[1], &s3[2], &s3[3]);

      uint16x4_t d0 = highbd_convolve8_4_h(s0, filter);
      uint16x4_t d1 = highbd_convolve8_4_h(s1, filter);
      uint16x4_t d2 = highbd_convolve8_4_h(s2, filter);
      uint16x4_t d3 = highbd_convolve8_4_h(s3, filter);

      d0 = vmin_u16(d0, vget_low_u16(max));
      d1 = vmin_u16(d1, vget_low_u16(max));
      d2 = vmin_u16(d2, vget_low_u16(max));
      d3 = vmin_u16(d3, vget_low_u16(max));

      store_u16_4x4(d, dst_stride, d0, d1, d2, d3);

      s += 4 * src_stride;
      d += 4 * dst_stride;
      height -= 4;
    } while (height > 0);
  } else {
    do {
      int w = width;
      const int16_t *s = (const int16_t *)src;
      uint16_t *d = dst;

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

        uint16x8_t d0 = highbd_convolve8_8_h(s0, filter);
        uint16x8_t d1 = highbd_convolve8_8_h(s1, filter);
        uint16x8_t d2 = highbd_convolve8_8_h(s2, filter);
        uint16x8_t d3 = highbd_convolve8_8_h(s3, filter);

        d0 = vminq_u16(d0, max);
        d1 = vminq_u16(d1, max);
        d2 = vminq_u16(d2, max);
        d3 = vminq_u16(d3, max);

        store_u16_8x4(d, dst_stride, d0, d1, d2, d3);

        s += 8;
        d += 8;
        w -= 8;
      } while (w != 0);
      src += 4 * src_stride;
      dst += 4 * dst_stride;
      height -= 4;
    } while (height > 0);
  }
}
