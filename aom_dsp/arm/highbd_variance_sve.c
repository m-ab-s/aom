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
#include <assert.h>

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"

#include "aom_dsp/aom_filter.h"
#include "aom_dsp/arm/dot_sve.h"
#include "aom_dsp/arm/mem_neon.h"
#include "aom_dsp/variance.h"

static INLINE uint32_t highbd_mse_wxh_sve(const uint16_t *src_ptr,
                                          int src_stride,
                                          const uint16_t *ref_ptr,
                                          int ref_stride, int w, int h,
                                          unsigned int *sse) {
  uint64x2_t sse_u64 = vdupq_n_u64(0);

  do {
    int j = 0;
    do {
      uint16x8_t s = vld1q_u16(src_ptr + j);
      uint16x8_t r = vld1q_u16(ref_ptr + j);

      uint16x8_t diff = vabdq_u16(s, r);

      sse_u64 = aom_udotq_u16(sse_u64, diff, diff);

      j += 8;
    } while (j < w);

    src_ptr += src_stride;
    ref_ptr += ref_stride;
  } while (--h != 0);

  *sse = (uint32_t)vaddvq_u64(sse_u64);
  return *sse;
}

#define HIGHBD_MSE_WXH_SVE(w, h)                                      \
  uint32_t aom_highbd_8_mse##w##x##h##_sve(                           \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, \
      int ref_stride, uint32_t *sse) {                                \
    uint16_t *src = CONVERT_TO_SHORTPTR(src_ptr);                     \
    uint16_t *ref = CONVERT_TO_SHORTPTR(ref_ptr);                     \
    highbd_mse_wxh_sve(src, src_stride, ref, ref_stride, w, h, sse);  \
    return *sse;                                                      \
  }                                                                   \
                                                                      \
  uint32_t aom_highbd_10_mse##w##x##h##_sve(                          \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, \
      int ref_stride, uint32_t *sse) {                                \
    uint16_t *src = CONVERT_TO_SHORTPTR(src_ptr);                     \
    uint16_t *ref = CONVERT_TO_SHORTPTR(ref_ptr);                     \
    highbd_mse_wxh_sve(src, src_stride, ref, ref_stride, w, h, sse);  \
    *sse = ROUND_POWER_OF_TWO(*sse, 4);                               \
    return *sse;                                                      \
  }                                                                   \
                                                                      \
  uint32_t aom_highbd_12_mse##w##x##h##_sve(                          \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, \
      int ref_stride, uint32_t *sse) {                                \
    uint16_t *src = CONVERT_TO_SHORTPTR(src_ptr);                     \
    uint16_t *ref = CONVERT_TO_SHORTPTR(ref_ptr);                     \
    highbd_mse_wxh_sve(src, src_stride, ref, ref_stride, w, h, sse);  \
    *sse = ROUND_POWER_OF_TWO(*sse, 8);                               \
    return *sse;                                                      \
  }

HIGHBD_MSE_WXH_SVE(16, 16)
HIGHBD_MSE_WXH_SVE(16, 8)
HIGHBD_MSE_WXH_SVE(8, 16)
HIGHBD_MSE_WXH_SVE(8, 8)

#undef HIGHBD_MSE_WXH_SVE

uint64_t aom_mse_wxh_16bit_highbd_sve(uint16_t *dst, int dstride, uint16_t *src,
                                      int sstride, int w, int h) {
  assert((w == 8 || w == 4) && (h == 8 || h == 4));

  uint64x2_t sum = vdupq_n_u64(0);

  if (w == 8) {
    do {
      uint16x8_t d0 = vld1q_u16(dst + 0 * dstride);
      uint16x8_t d1 = vld1q_u16(dst + 1 * dstride);
      uint16x8_t s0 = vld1q_u16(src + 0 * sstride);
      uint16x8_t s1 = vld1q_u16(src + 1 * sstride);

      uint16x8_t abs_diff0 = vabdq_u16(s0, d0);
      uint16x8_t abs_diff1 = vabdq_u16(s1, d1);

      sum = aom_udotq_u16(sum, abs_diff0, abs_diff0);
      sum = aom_udotq_u16(sum, abs_diff1, abs_diff1);

      dst += 2 * dstride;
      src += 2 * sstride;
      h -= 2;
    } while (h != 0);
  } else {  // w == 4
    do {
      uint16x8_t d0 = load_unaligned_u16_4x2(dst + 0 * dstride, dstride);
      uint16x8_t d1 = load_unaligned_u16_4x2(dst + 2 * dstride, dstride);
      uint16x8_t s0 = load_unaligned_u16_4x2(src + 0 * sstride, sstride);
      uint16x8_t s1 = load_unaligned_u16_4x2(src + 2 * sstride, sstride);

      uint16x8_t abs_diff0 = vabdq_u16(s0, d0);
      uint16x8_t abs_diff1 = vabdq_u16(s1, d1);

      sum = aom_udotq_u16(sum, abs_diff0, abs_diff0);
      sum = aom_udotq_u16(sum, abs_diff1, abs_diff1);

      dst += 4 * dstride;
      src += 4 * sstride;
      h -= 4;
    } while (h != 0);
  }

  return vaddvq_u64(sum);
}
