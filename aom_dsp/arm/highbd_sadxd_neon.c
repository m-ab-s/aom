/*
 * Copyright (c) 2023 The WebM project authors. All rights reserved.
 * Copyright (c) 2023, Alliance for Open Media. All rights reserved.
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

#include "aom/aom_integer.h"
#include "aom_dsp/arm/mem_neon.h"
#include "aom_dsp/arm/sum_neon.h"

static inline void highbd_sad4xhx4d_neon(const uint8_t *src_ptr, int src_stride,
                                         const uint8_t *const ref_ptr[4],
                                         int ref_stride, uint32_t res[4],
                                         int h) {
  const uint16_t *src16_ptr = CONVERT_TO_SHORTPTR(src_ptr);
  const uint16_t *ref16_ptr0 = CONVERT_TO_SHORTPTR(ref_ptr[0]);
  const uint16_t *ref16_ptr1 = CONVERT_TO_SHORTPTR(ref_ptr[1]);
  const uint16_t *ref16_ptr2 = CONVERT_TO_SHORTPTR(ref_ptr[2]);
  const uint16_t *ref16_ptr3 = CONVERT_TO_SHORTPTR(ref_ptr[3]);
  const int src_stride_2 = src_stride << 1;
  const int ref_stride_2 = ref_stride << 1;

  // 'h_overflow' is the number of 4-wide rows packed in pairs into uint16x8_t
  // before 16-bit accumulators overflow: 65535 / 4095 ~= 16 pairs = 32 rows.
  const int h_overflow = 32;
  const int h_limit = h < h_overflow ? h : h_overflow;
  assert(h % 2 == 0);
  assert(h % h_limit == 0);

  uint32x4_t sum_u32[4] = { vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0),
                            vdupq_n_u32(0) };

  do {
    uint16x8_t sum_u16[4] = { vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0),
                              vdupq_n_u16(0) };
    int i = h_limit;
    do {
      uint16x8_t s =
          vcombine_u16(vld1_u16(src16_ptr), vld1_u16(src16_ptr + src_stride));
      uint16x8_t r0 =
          vcombine_u16(vld1_u16(ref16_ptr0), vld1_u16(ref16_ptr0 + ref_stride));
      uint16x8_t r1 =
          vcombine_u16(vld1_u16(ref16_ptr1), vld1_u16(ref16_ptr1 + ref_stride));
      uint16x8_t r2 =
          vcombine_u16(vld1_u16(ref16_ptr2), vld1_u16(ref16_ptr2 + ref_stride));
      uint16x8_t r3 =
          vcombine_u16(vld1_u16(ref16_ptr3), vld1_u16(ref16_ptr3 + ref_stride));

      sum_u16[0] = vabaq_u16(sum_u16[0], s, r0);
      sum_u16[1] = vabaq_u16(sum_u16[1], s, r1);
      sum_u16[2] = vabaq_u16(sum_u16[2], s, r2);
      sum_u16[3] = vabaq_u16(sum_u16[3], s, r3);

      src16_ptr += src_stride_2;
      ref16_ptr0 += ref_stride_2;
      ref16_ptr1 += ref_stride_2;
      ref16_ptr2 += ref_stride_2;
      ref16_ptr3 += ref_stride_2;
      i -= 2;
    } while (i != 0);

    sum_u32[0] = vpadalq_u16(sum_u32[0], sum_u16[0]);
    sum_u32[1] = vpadalq_u16(sum_u32[1], sum_u16[1]);
    sum_u32[2] = vpadalq_u16(sum_u32[2], sum_u16[2]);
    sum_u32[3] = vpadalq_u16(sum_u32[3], sum_u16[3]);
    h -= h_limit;
  } while (h != 0);

  vst1q_u32(res, horizontal_add_4d_u32x4(sum_u32));
}

static inline void highbd_sad8xhx4d_neon(const uint8_t *src_ptr, int src_stride,
                                         const uint8_t *const ref_ptr[4],
                                         int ref_stride, uint32_t res[4],
                                         int h) {
  const uint16_t *src16_ptr = CONVERT_TO_SHORTPTR(src_ptr);
  const uint16_t *ref16_ptr0 = CONVERT_TO_SHORTPTR(ref_ptr[0]);
  const uint16_t *ref16_ptr1 = CONVERT_TO_SHORTPTR(ref_ptr[1]);
  const uint16_t *ref16_ptr2 = CONVERT_TO_SHORTPTR(ref_ptr[2]);
  const uint16_t *ref16_ptr3 = CONVERT_TO_SHORTPTR(ref_ptr[3]);
  const int src_stride_2 = src_stride << 1;
  const int ref_stride_2 = ref_stride << 1;

  // Two 16-bit accumulators each hold up to 16 rows: 32 rows total.
  const int h_overflow = 32;
  const int h_limit = h < h_overflow ? h : h_overflow;
  assert(h % 2 == 0);
  assert(h % h_limit == 0);

  uint32x4_t sum_u32[4] = { vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0),
                            vdupq_n_u32(0) };

  do {
    uint16x8_t sum0_u16[4] = { vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0),
                               vdupq_n_u16(0) };
    uint16x8_t sum1_u16[4] = { vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0),
                               vdupq_n_u16(0) };

    int i = h_limit;
    do {
      uint16x8_t s0 = vld1q_u16(src16_ptr);
      uint16x8_t s1 = vld1q_u16(src16_ptr + src_stride);

      sum0_u16[0] = vabaq_u16(sum0_u16[0], s0, vld1q_u16(ref16_ptr0));
      sum0_u16[1] = vabaq_u16(sum0_u16[1], s0, vld1q_u16(ref16_ptr1));
      sum0_u16[2] = vabaq_u16(sum0_u16[2], s0, vld1q_u16(ref16_ptr2));
      sum0_u16[3] = vabaq_u16(sum0_u16[3], s0, vld1q_u16(ref16_ptr3));

      sum1_u16[0] =
          vabaq_u16(sum1_u16[0], s1, vld1q_u16(ref16_ptr0 + ref_stride));
      sum1_u16[1] =
          vabaq_u16(sum1_u16[1], s1, vld1q_u16(ref16_ptr1 + ref_stride));
      sum1_u16[2] =
          vabaq_u16(sum1_u16[2], s1, vld1q_u16(ref16_ptr2 + ref_stride));
      sum1_u16[3] =
          vabaq_u16(sum1_u16[3], s1, vld1q_u16(ref16_ptr3 + ref_stride));

      src16_ptr += src_stride_2;
      ref16_ptr0 += ref_stride_2;
      ref16_ptr1 += ref_stride_2;
      ref16_ptr2 += ref_stride_2;
      ref16_ptr3 += ref_stride_2;
      i -= 2;
    } while (i != 0);

    sum_u32[0] = vpadalq_u16(sum_u32[0], sum0_u16[0]);
    sum_u32[1] = vpadalq_u16(sum_u32[1], sum0_u16[1]);
    sum_u32[2] = vpadalq_u16(sum_u32[2], sum0_u16[2]);
    sum_u32[3] = vpadalq_u16(sum_u32[3], sum0_u16[3]);

    sum_u32[0] = vpadalq_u16(sum_u32[0], sum1_u16[0]);
    sum_u32[1] = vpadalq_u16(sum_u32[1], sum1_u16[1]);
    sum_u32[2] = vpadalq_u16(sum_u32[2], sum1_u16[2]);
    sum_u32[3] = vpadalq_u16(sum_u32[3], sum1_u16[3]);

    h -= h_limit;
  } while (h != 0);

  vst1q_u32(res, horizontal_add_4d_u32x4(sum_u32));
}

static inline void highbd_sadwxhx4d_neon(const uint8_t *src_ptr, int src_stride,
                                         const uint8_t *const ref_ptr[4],
                                         int ref_stride, uint32_t res[4], int w,
                                         int h, const int h_overflow) {
  const uint16_t *src16_ptr = CONVERT_TO_SHORTPTR(src_ptr);
  const uint16_t *ref16_ptr0 = CONVERT_TO_SHORTPTR(ref_ptr[0]);
  const uint16_t *ref16_ptr1 = CONVERT_TO_SHORTPTR(ref_ptr[1]);
  const uint16_t *ref16_ptr2 = CONVERT_TO_SHORTPTR(ref_ptr[2]);
  const uint16_t *ref16_ptr3 = CONVERT_TO_SHORTPTR(ref_ptr[3]);

  const int h_limit = h < h_overflow ? h : h_overflow;
  assert(h % h_limit == 0);

  uint32x4_t sum_u32[4] = { vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0),
                            vdupq_n_u32(0) };

  do {
    uint16x8_t sum0_u16[4] = { vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0),
                               vdupq_n_u16(0) };
    uint16x8_t sum1_u16[4] = { vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0),
                               vdupq_n_u16(0) };

    int i = h_limit;
    do {
      int j = 0;
      do {
        uint16x8_t s0 = vld1q_u16(src16_ptr + j);

        sum0_u16[0] = vabaq_u16(sum0_u16[0], s0, vld1q_u16(ref16_ptr0 + j));
        sum0_u16[1] = vabaq_u16(sum0_u16[1], s0, vld1q_u16(ref16_ptr1 + j));
        sum0_u16[2] = vabaq_u16(sum0_u16[2], s0, vld1q_u16(ref16_ptr2 + j));
        sum0_u16[3] = vabaq_u16(sum0_u16[3], s0, vld1q_u16(ref16_ptr3 + j));

        uint16x8_t s1 = vld1q_u16(src16_ptr + j + 8);
        sum1_u16[0] = vabaq_u16(sum1_u16[0], s1, vld1q_u16(ref16_ptr0 + j + 8));
        sum1_u16[1] = vabaq_u16(sum1_u16[1], s1, vld1q_u16(ref16_ptr1 + j + 8));
        sum1_u16[2] = vabaq_u16(sum1_u16[2], s1, vld1q_u16(ref16_ptr2 + j + 8));
        sum1_u16[3] = vabaq_u16(sum1_u16[3], s1, vld1q_u16(ref16_ptr3 + j + 8));

        j += 16;
      } while (j < w);

      src16_ptr += src_stride;
      ref16_ptr0 += ref_stride;
      ref16_ptr1 += ref_stride;
      ref16_ptr2 += ref_stride;
      ref16_ptr3 += ref_stride;
    } while (--i != 0);

    sum_u32[0] = vpadalq_u16(sum_u32[0], sum0_u16[0]);
    sum_u32[1] = vpadalq_u16(sum_u32[1], sum0_u16[1]);
    sum_u32[2] = vpadalq_u16(sum_u32[2], sum0_u16[2]);
    sum_u32[3] = vpadalq_u16(sum_u32[3], sum0_u16[3]);

    sum_u32[0] = vpadalq_u16(sum_u32[0], sum1_u16[0]);
    sum_u32[1] = vpadalq_u16(sum_u32[1], sum1_u16[1]);
    sum_u32[2] = vpadalq_u16(sum_u32[2], sum1_u16[2]);
    sum_u32[3] = vpadalq_u16(sum_u32[3], sum1_u16[3]);

    h -= h_limit;
  } while (h != 0);

  vst1q_u32(res, horizontal_add_4d_u32x4(sum_u32));
}

static inline void highbd_sad16xhx4d_neon(const uint8_t *src_ptr,
                                          int src_stride,
                                          const uint8_t *const ref_ptr[4],
                                          int ref_stride, uint32_t res[4],
                                          int h) {
  // Two 16-bit accumulators hold up to 16 16-wide rows before overflow.
  const int h_overflow = 16;
  highbd_sadwxhx4d_neon(src_ptr, src_stride, ref_ptr, ref_stride, res, 16, h,
                        h_overflow);
}

static inline void highbd_sad32xhx4d_neon(const uint8_t *src_ptr,
                                          int src_stride,
                                          const uint8_t *const ref_ptr[4],
                                          int ref_stride, uint32_t res[4],
                                          int h) {
  // Two 16-bit accumulators hold up to 8 32-wide rows before overflow.
  const int h_overflow = 8;
  highbd_sadwxhx4d_neon(src_ptr, src_stride, ref_ptr, ref_stride, res, 32, h,
                        h_overflow);
}

static inline void highbd_sad64xhx4d_neon(const uint8_t *src_ptr,
                                          int src_stride,
                                          const uint8_t *const ref_ptr[4],
                                          int ref_stride, uint32_t res[4],
                                          int h) {
  // Two 16-bit accumulators hold up to 4 64-wide rows before overflow.
  const int h_overflow = 4;
  highbd_sadwxhx4d_neon(src_ptr, src_stride, ref_ptr, ref_stride, res, 64, h,
                        h_overflow);
}

static inline void highbd_sad128xhx4d_neon(const uint8_t *src_ptr,
                                           int src_stride,
                                           const uint8_t *const ref_ptr[4],
                                           int ref_stride, uint32_t res[4],
                                           int h) {
  // Two 16-bit accumulators hold up to 2 128-wide rows before overflow.
  const int h_overflow = 2;
  highbd_sadwxhx4d_neon(src_ptr, src_stride, ref_ptr, ref_stride, res, 128, h,
                        h_overflow);
}

#define HBD_SAD_WXH_4D_NEON(w, h)                                            \
  void aom_highbd_sad##w##x##h##x4d_neon(                                    \
      const uint8_t *src, int src_stride, const uint8_t *const ref_array[4], \
      int ref_stride, uint32_t sad_array[4]) {                               \
    highbd_sad##w##xhx4d_neon(src, src_stride, ref_array, ref_stride,        \
                              sad_array, (h));                               \
  }

HBD_SAD_WXH_4D_NEON(4, 4)
HBD_SAD_WXH_4D_NEON(4, 8)

HBD_SAD_WXH_4D_NEON(8, 4)
HBD_SAD_WXH_4D_NEON(8, 8)
HBD_SAD_WXH_4D_NEON(8, 16)

HBD_SAD_WXH_4D_NEON(16, 8)
HBD_SAD_WXH_4D_NEON(16, 16)
HBD_SAD_WXH_4D_NEON(16, 32)

HBD_SAD_WXH_4D_NEON(32, 16)
HBD_SAD_WXH_4D_NEON(32, 32)
HBD_SAD_WXH_4D_NEON(32, 64)

HBD_SAD_WXH_4D_NEON(64, 32)
HBD_SAD_WXH_4D_NEON(64, 64)
HBD_SAD_WXH_4D_NEON(64, 128)

HBD_SAD_WXH_4D_NEON(128, 64)
HBD_SAD_WXH_4D_NEON(128, 128)

#if !CONFIG_REALTIME_ONLY
HBD_SAD_WXH_4D_NEON(4, 16)

HBD_SAD_WXH_4D_NEON(8, 32)

HBD_SAD_WXH_4D_NEON(16, 4)
HBD_SAD_WXH_4D_NEON(16, 64)

HBD_SAD_WXH_4D_NEON(32, 8)

HBD_SAD_WXH_4D_NEON(64, 16)
#endif  // !CONFIG_REALTIME_ONLY

#undef HBD_SAD_WXH_4D_NEON

#define HBD_SAD_SKIP_WXH_4D_NEON(w, h)                                        \
  void aom_highbd_sad_skip_##w##x##h##x4d_neon(                               \
      const uint8_t *src, int src_stride, const uint8_t *const ref_array[4],  \
      int ref_stride, uint32_t sad_array[4]) {                                \
    highbd_sad##w##xhx4d_neon(src, 2 * src_stride, ref_array, 2 * ref_stride, \
                              sad_array, ((h) >> 1));                         \
    sad_array[0] <<= 1;                                                       \
    sad_array[1] <<= 1;                                                       \
    sad_array[2] <<= 1;                                                       \
    sad_array[3] <<= 1;                                                       \
  }

HBD_SAD_SKIP_WXH_4D_NEON(8, 16)

HBD_SAD_SKIP_WXH_4D_NEON(16, 16)
HBD_SAD_SKIP_WXH_4D_NEON(16, 32)

HBD_SAD_SKIP_WXH_4D_NEON(32, 16)
HBD_SAD_SKIP_WXH_4D_NEON(32, 32)
HBD_SAD_SKIP_WXH_4D_NEON(32, 64)

HBD_SAD_SKIP_WXH_4D_NEON(64, 32)
HBD_SAD_SKIP_WXH_4D_NEON(64, 64)
HBD_SAD_SKIP_WXH_4D_NEON(64, 128)

HBD_SAD_SKIP_WXH_4D_NEON(128, 64)
HBD_SAD_SKIP_WXH_4D_NEON(128, 128)

#if !CONFIG_REALTIME_ONLY
HBD_SAD_SKIP_WXH_4D_NEON(4, 16)

HBD_SAD_SKIP_WXH_4D_NEON(8, 32)

HBD_SAD_SKIP_WXH_4D_NEON(16, 64)

HBD_SAD_SKIP_WXH_4D_NEON(64, 16)
#endif  // !CONFIG_REALTIME_ONLY

#undef HBD_SAD_SKIP_WXH_4D_NEON

static inline void highbd_sad4xhx3d_neon(const uint8_t *src_ptr, int src_stride,
                                         const uint8_t *const ref_ptr[4],
                                         int ref_stride, uint32_t res[4],
                                         int h) {
  const uint16_t *src16_ptr = CONVERT_TO_SHORTPTR(src_ptr);
  const uint16_t *ref16_ptr0 = CONVERT_TO_SHORTPTR(ref_ptr[0]);
  const uint16_t *ref16_ptr1 = CONVERT_TO_SHORTPTR(ref_ptr[1]);
  const uint16_t *ref16_ptr2 = CONVERT_TO_SHORTPTR(ref_ptr[2]);
  const int src_stride_2 = src_stride << 1;
  const int ref_stride_2 = ref_stride << 1;

  // 'h_overflow' is the number of 4-wide rows packed in pairs into uint16x8_t
  // before 16-bit accumulators overflow: 65535 / 4095 ~= 16 pairs = 32 rows.
  const int h_overflow = 32;
  const int h_limit = h < h_overflow ? h : h_overflow;
  assert(h % 2 == 0);
  assert(h % h_limit == 0);

  uint32x4_t sum_u32[3] = { vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0) };

  do {
    uint16x8_t sum_u16[3] = { vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0) };
    int i = h_limit;
    do {
      uint16x8_t s =
          vcombine_u16(vld1_u16(src16_ptr), vld1_u16(src16_ptr + src_stride));
      uint16x8_t r0 =
          vcombine_u16(vld1_u16(ref16_ptr0), vld1_u16(ref16_ptr0 + ref_stride));
      uint16x8_t r1 =
          vcombine_u16(vld1_u16(ref16_ptr1), vld1_u16(ref16_ptr1 + ref_stride));
      uint16x8_t r2 =
          vcombine_u16(vld1_u16(ref16_ptr2), vld1_u16(ref16_ptr2 + ref_stride));

      sum_u16[0] = vabaq_u16(sum_u16[0], s, r0);
      sum_u16[1] = vabaq_u16(sum_u16[1], s, r1);
      sum_u16[2] = vabaq_u16(sum_u16[2], s, r2);

      src16_ptr += src_stride_2;
      ref16_ptr0 += ref_stride_2;
      ref16_ptr1 += ref_stride_2;
      ref16_ptr2 += ref_stride_2;
      i -= 2;
    } while (i != 0);

    sum_u32[0] = vpadalq_u16(sum_u32[0], sum_u16[0]);
    sum_u32[1] = vpadalq_u16(sum_u32[1], sum_u16[1]);
    sum_u32[2] = vpadalq_u16(sum_u32[2], sum_u16[2]);
    h -= h_limit;
  } while (h != 0);

  res[0] = horizontal_add_u32x4(sum_u32[0]);
  res[1] = horizontal_add_u32x4(sum_u32[1]);
  res[2] = horizontal_add_u32x4(sum_u32[2]);
}

static inline void highbd_sad8xhx3d_neon(const uint8_t *src_ptr, int src_stride,
                                         const uint8_t *const ref_ptr[4],
                                         int ref_stride, uint32_t res[4],
                                         int h) {
  const uint16_t *src16_ptr = CONVERT_TO_SHORTPTR(src_ptr);
  const uint16_t *ref16_ptr0 = CONVERT_TO_SHORTPTR(ref_ptr[0]);
  const uint16_t *ref16_ptr1 = CONVERT_TO_SHORTPTR(ref_ptr[1]);
  const uint16_t *ref16_ptr2 = CONVERT_TO_SHORTPTR(ref_ptr[2]);
  const int src_stride_2 = src_stride << 1;
  const int ref_stride_2 = ref_stride << 1;

  // Two 16-bit accumulators each hold up to 16 rows: 32 rows total.
  const int h_overflow = 32;
  const int h_limit = h < h_overflow ? h : h_overflow;
  assert(h % 2 == 0);
  assert(h % h_limit == 0);

  uint32x4_t sum_u32[3] = { vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0) };

  do {
    uint16x8_t sum0_u16[3] = { vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0) };
    uint16x8_t sum1_u16[3] = { vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0) };
    int i = h_limit;
    do {
      uint16x8_t s0 = vld1q_u16(src16_ptr);
      uint16x8_t s1 = vld1q_u16(src16_ptr + src_stride);

      sum0_u16[0] = vabaq_u16(sum0_u16[0], s0, vld1q_u16(ref16_ptr0));
      sum0_u16[1] = vabaq_u16(sum0_u16[1], s0, vld1q_u16(ref16_ptr1));
      sum0_u16[2] = vabaq_u16(sum0_u16[2], s0, vld1q_u16(ref16_ptr2));

      sum1_u16[0] =
          vabaq_u16(sum1_u16[0], s1, vld1q_u16(ref16_ptr0 + ref_stride));
      sum1_u16[1] =
          vabaq_u16(sum1_u16[1], s1, vld1q_u16(ref16_ptr1 + ref_stride));
      sum1_u16[2] =
          vabaq_u16(sum1_u16[2], s1, vld1q_u16(ref16_ptr2 + ref_stride));

      src16_ptr += src_stride_2;
      ref16_ptr0 += ref_stride_2;
      ref16_ptr1 += ref_stride_2;
      ref16_ptr2 += ref_stride_2;
      i -= 2;
    } while (i != 0);

    sum_u32[0] = vpadalq_u16(sum_u32[0], sum0_u16[0]);
    sum_u32[1] = vpadalq_u16(sum_u32[1], sum0_u16[1]);
    sum_u32[2] = vpadalq_u16(sum_u32[2], sum0_u16[2]);

    sum_u32[0] = vpadalq_u16(sum_u32[0], sum1_u16[0]);
    sum_u32[1] = vpadalq_u16(sum_u32[1], sum1_u16[1]);
    sum_u32[2] = vpadalq_u16(sum_u32[2], sum1_u16[2]);

    h -= h_limit;
  } while (h != 0);

  res[0] = horizontal_add_u32x4(sum_u32[0]);
  res[1] = horizontal_add_u32x4(sum_u32[1]);
  res[2] = horizontal_add_u32x4(sum_u32[2]);
}

static inline void highbd_sadwxhx3d_neon(const uint8_t *src_ptr, int src_stride,
                                         const uint8_t *const ref_ptr[4],
                                         int ref_stride, uint32_t res[4], int w,
                                         int h, const int h_overflow) {
  const uint16_t *src16_ptr = CONVERT_TO_SHORTPTR(src_ptr);
  const uint16_t *ref16_ptr0 = CONVERT_TO_SHORTPTR(ref_ptr[0]);
  const uint16_t *ref16_ptr1 = CONVERT_TO_SHORTPTR(ref_ptr[1]);
  const uint16_t *ref16_ptr2 = CONVERT_TO_SHORTPTR(ref_ptr[2]);
  uint32x4_t sum_u32[3] = { vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0) };

  const int h_limit = h < h_overflow ? h : h_overflow;
  assert(h % h_limit == 0);

  do {
    uint16x8_t sum0_u16[3] = { vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0) };
    uint16x8_t sum1_u16[3] = { vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0) };

    int i = h_limit;
    do {
      int j = 0;
      do {
        uint16x8_t s0 = vld1q_u16(src16_ptr + j);

        sum0_u16[0] = vabaq_u16(sum0_u16[0], s0, vld1q_u16(ref16_ptr0 + j));
        sum0_u16[1] = vabaq_u16(sum0_u16[1], s0, vld1q_u16(ref16_ptr1 + j));
        sum0_u16[2] = vabaq_u16(sum0_u16[2], s0, vld1q_u16(ref16_ptr2 + j));

        uint16x8_t s1 = vld1q_u16(src16_ptr + j + 8);
        sum1_u16[0] = vabaq_u16(sum1_u16[0], s1, vld1q_u16(ref16_ptr0 + j + 8));
        sum1_u16[1] = vabaq_u16(sum1_u16[1], s1, vld1q_u16(ref16_ptr1 + j + 8));
        sum1_u16[2] = vabaq_u16(sum1_u16[2], s1, vld1q_u16(ref16_ptr2 + j + 8));

        j += 16;
      } while (j < w);

      src16_ptr += src_stride;
      ref16_ptr0 += ref_stride;
      ref16_ptr1 += ref_stride;
      ref16_ptr2 += ref_stride;
    } while (--i != 0);

    sum_u32[0] = vpadalq_u16(sum_u32[0], sum0_u16[0]);
    sum_u32[1] = vpadalq_u16(sum_u32[1], sum0_u16[1]);
    sum_u32[2] = vpadalq_u16(sum_u32[2], sum0_u16[2]);

    sum_u32[0] = vpadalq_u16(sum_u32[0], sum1_u16[0]);
    sum_u32[1] = vpadalq_u16(sum_u32[1], sum1_u16[1]);
    sum_u32[2] = vpadalq_u16(sum_u32[2], sum1_u16[2]);

    h -= h_limit;
  } while (h != 0);

  res[0] = horizontal_add_u32x4(sum_u32[0]);
  res[1] = horizontal_add_u32x4(sum_u32[1]);
  res[2] = horizontal_add_u32x4(sum_u32[2]);
}

static inline void highbd_sad16xhx3d_neon(const uint8_t *src_ptr,
                                          int src_stride,
                                          const uint8_t *const ref_ptr[4],
                                          int ref_stride, uint32_t res[4],
                                          int h) {
  // Two 16-bit accumulators hold up to 16 16-wide rows before overflow.
  const int h_overflow = 16;
  highbd_sadwxhx3d_neon(src_ptr, src_stride, ref_ptr, ref_stride, res, 16, h,
                        h_overflow);
}

static inline void highbd_sad32xhx3d_neon(const uint8_t *src_ptr,
                                          int src_stride,
                                          const uint8_t *const ref_ptr[4],
                                          int ref_stride, uint32_t res[4],
                                          int h) {
  // Two 16-bit accumulators hold up to 8 32-wide rows before overflow.
  const int h_overflow = 8;
  highbd_sadwxhx3d_neon(src_ptr, src_stride, ref_ptr, ref_stride, res, 32, h,
                        h_overflow);
}

static inline void highbd_sad64xhx3d_neon(const uint8_t *src_ptr,
                                          int src_stride,
                                          const uint8_t *const ref_ptr[4],
                                          int ref_stride, uint32_t res[4],
                                          int h) {
  // Two 16-bit accumulators hold up to 4 64-wide rows before overflow.
  const int h_overflow = 4;
  highbd_sadwxhx3d_neon(src_ptr, src_stride, ref_ptr, ref_stride, res, 64, h,
                        h_overflow);
}

static inline void highbd_sad128xhx3d_neon(const uint8_t *src_ptr,
                                           int src_stride,
                                           const uint8_t *const ref_ptr[4],
                                           int ref_stride, uint32_t res[4],
                                           int h) {
  // Two 16-bit accumulators hold up to 2 128-wide rows before overflow.
  const int h_overflow = 2;
  highbd_sadwxhx3d_neon(src_ptr, src_stride, ref_ptr, ref_stride, res, 128, h,
                        h_overflow);
}

#define HBD_SAD_WXH_3D_NEON(w, h)                                            \
  void aom_highbd_sad##w##x##h##x3d_neon(                                    \
      const uint8_t *src, int src_stride, const uint8_t *const ref_array[4], \
      int ref_stride, uint32_t sad_array[4]) {                               \
    highbd_sad##w##xhx3d_neon(src, src_stride, ref_array, ref_stride,        \
                              sad_array, (h));                               \
  }

HBD_SAD_WXH_3D_NEON(4, 4)
HBD_SAD_WXH_3D_NEON(4, 8)

HBD_SAD_WXH_3D_NEON(8, 4)
HBD_SAD_WXH_3D_NEON(8, 8)
HBD_SAD_WXH_3D_NEON(8, 16)

HBD_SAD_WXH_3D_NEON(16, 8)
HBD_SAD_WXH_3D_NEON(16, 16)
HBD_SAD_WXH_3D_NEON(16, 32)

HBD_SAD_WXH_3D_NEON(32, 16)
HBD_SAD_WXH_3D_NEON(32, 32)
HBD_SAD_WXH_3D_NEON(32, 64)

HBD_SAD_WXH_3D_NEON(64, 32)
HBD_SAD_WXH_3D_NEON(64, 64)
HBD_SAD_WXH_3D_NEON(64, 128)

HBD_SAD_WXH_3D_NEON(128, 64)
HBD_SAD_WXH_3D_NEON(128, 128)

#if !CONFIG_REALTIME_ONLY
HBD_SAD_WXH_3D_NEON(4, 16)

HBD_SAD_WXH_3D_NEON(8, 32)

HBD_SAD_WXH_3D_NEON(16, 4)
HBD_SAD_WXH_3D_NEON(16, 64)

HBD_SAD_WXH_3D_NEON(32, 8)

HBD_SAD_WXH_3D_NEON(64, 16)
#endif  // !CONFIG_REALTIME_ONLY

#undef HBD_SAD_WXH_3D_NEON
