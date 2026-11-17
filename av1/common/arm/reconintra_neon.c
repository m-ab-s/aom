/*
 * Copyright (c) 2020, Alliance for Open Media. All rights reserved.
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
#include <string.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"

#include "aom/aom_integer.h"
#include "aom_dsp/arm/mem_neon.h"
#include "aom_dsp/arm/sum_neon.h"

#define FILTER_INTRA_SCALE_BITS 4

// These kernels are a transposed version of those defined in reconintra.c,
// with the absolute value of the negatives taken in the top row.
DECLARE_ALIGNED(16, static const uint8_t,
                av1_filter_intra_taps_neon[FILTER_INTRA_MODES][7][8]) = {
  // clang-format off
  {
      {  6,  5,  3,  3,  4,  3,  3,  3 },
      { 10,  2,  1,  1,  6,  2,  2,  1 },
      {  0, 10,  1,  1,  0,  6,  2,  2 },
      {  0,  0, 10,  2,  0,  0,  6,  2 },
      {  0,  0,  0, 10,  0,  0,  0,  6 },
      { 12,  9,  7,  5,  2,  2,  2,  3 },
      {  0,  0,  0,  0, 12,  9,  7,  5 }
  },
  {
      { 10,  6,  4,  2, 10,  6,  4,  2 },
      { 16,  0,  0,  0, 16,  0,  0,  0 },
      {  0, 16,  0,  0,  0, 16,  0,  0 },
      {  0,  0, 16,  0,  0,  0, 16,  0 },
      {  0,  0,  0, 16,  0,  0,  0, 16 },
      { 10,  6,  4,  2,  0,  0,  0,  0 },
      {  0,  0,  0,  0, 10,  6,  4,  2 }
  },
  {
      {  8,  8,  8,  8,  4,  4,  4,  4 },
      {  8,  0,  0,  0,  4,  0,  0,  0 },
      {  0,  8,  0,  0,  0,  4,  0,  0 },
      {  0,  0,  8,  0,  0,  0,  4,  0 },
      {  0,  0,  0,  8,  0,  0,  0,  4 },
      { 16, 16, 16, 16,  0,  0,  0,  0 },
      {  0,  0,  0,  0, 16, 16, 16, 16 }
  },
  {
      {  2,  1,  1,  0,  1,  1,  1,  1 },
      {  8,  3,  2,  1,  4,  3,  2,  2 },
      {  0,  8,  3,  2,  0,  4,  3,  2 },
      {  0,  0,  8,  3,  0,  0,  4,  3 },
      {  0,  0,  0,  8,  0,  0,  0,  4 },
      { 10,  6,  4,  2,  3,  4,  4,  3 },
      {  0,  0,  0,  0, 10,  6,  4,  3 }
  },
  {
      { 12, 10,  9,  8, 10,  9,  8,  7 },
      { 14,  0,  0,  0, 12,  1,  0,  0 },
      {  0, 14,  0,  0,  0, 12,  0,  0 },
      {  0,  0, 14,  0,  0,  0, 12,  1 },
      {  0,  0,  0, 14,  0,  0,  0, 12 },
      { 14, 12, 11, 10,  0,  0,  1,  1 },
      {  0,  0,  0,  0, 14, 12, 11,  9 }
  }
  // clang-format on
};

static inline uint8x8_t filter_intra_predictor(
    uint8x8_t s0, uint8x8_t s1, uint8x8_t s2, uint8x8_t s3, uint8x8_t s4,
    uint8x8_t s5, uint8x8_t s6, const uint8x8_t f0, const uint8x8_t f1,
    const uint8x8_t f2, const uint8x8_t f3, const uint8x8_t f4,
    const uint8x8_t f5, const uint8x8_t f6) {
  uint16x8_t acc = vmull_u8(s1, f1);
  // First row of each filter has all negative values so subtract.
  acc = vmlsl_u8(acc, s0, f0);
  acc = vmlal_u8(acc, s2, f2);
  acc = vmlal_u8(acc, s3, f3);
  acc = vmlal_u8(acc, s4, f4);
  acc = vmlal_u8(acc, s5, f5);
  acc = vmlal_u8(acc, s6, f6);

  return vqrshrun_n_s16(vreinterpretq_s16_u16(acc), FILTER_INTRA_SCALE_BITS);
}

void av1_filter_intra_predictor_neon(uint8_t *dst, ptrdiff_t stride,
                                     TX_SIZE tx_size, const uint8_t *above,
                                     const uint8_t *left, int mode) {
  assert(tx_size < TX_SIZES_ALL);
  assert(mode >= 0 && mode < FILTER_INTRA_MODES);
  const int width = tx_size_wide[tx_size];
  const int height = tx_size_high[tx_size];
  assert(width >= 4 && width <= 32 && height >= 4 && height <= 32);
  assert((height & 1) == 0);

  const uint8x8_t f0 = vld1_u8(av1_filter_intra_taps_neon[mode][0]);
  const uint8x8_t f1 = vld1_u8(av1_filter_intra_taps_neon[mode][1]);
  const uint8x8_t f2 = vld1_u8(av1_filter_intra_taps_neon[mode][2]);
  const uint8x8_t f3 = vld1_u8(av1_filter_intra_taps_neon[mode][3]);
  const uint8x8_t f4 = vld1_u8(av1_filter_intra_taps_neon[mode][4]);
  const uint8x8_t f5 = vld1_u8(av1_filter_intra_taps_neon[mode][5]);
  const uint8x8_t f6 = vld1_u8(av1_filter_intra_taps_neon[mode][6]);

  if (width == 4) {
    uint8x8_t s0 = vdup_n_u8(above[-1]);
    uint8x8_t s5 = vdup_n_u8(left[0]);
    uint8x8_t s6 = vdup_n_u8(left[1]);

    uint8x8_t s1234 = load_unaligned_dup_u8_4x2(above);
    uint8x8_t s1 = vdup_lane_u8(s1234, 0);
    uint8x8_t s2 = vdup_lane_u8(s1234, 1);
    uint8x8_t s3 = vdup_lane_u8(s1234, 2);
    uint8x8_t s4 = vdup_lane_u8(s1234, 3);

    uint8x8_t res = filter_intra_predictor(s0, s1, s2, s3, s4, s5, s6, f0, f1,
                                           f2, f3, f4, f5, f6);

    store_u8x4_strided_x2(dst, stride, res);

    left += 1;
    int h = height - 2;
    dst += 2 * stride;
    do {
      s0 = vdup_n_u8(left[0]);
      s5 = vdup_n_u8(left[1]);
      s6 = vdup_n_u8(left[2]);

      s1 = vdup_lane_u8(res, 4);
      s2 = vdup_lane_u8(res, 5);
      s3 = vdup_lane_u8(res, 6);
      s4 = vdup_lane_u8(res, 7);

      res = filter_intra_predictor(s0, s1, s2, s3, s4, s5, s6, f0, f1, f2, f3,
                                   f4, f5, f6);

      store_u8x4_strided_x2(dst, stride, res);

      left += 2;
      dst += 2 * stride;
      h -= 2;
    } while (h != 0);
  } else if (width == 8) {
    uint8x8_t s0_lo = vdup_n_u8(above[-1]);
    uint8x8_t s5_lo = vdup_n_u8(left[0]);
    uint8x8_t s6_lo = vdup_n_u8(left[1]);
    const uint8x8_t abv = vld1_u8(above);

    uint8x8_t s1_lo = vdup_lane_u8(abv, 0);
    uint8x8_t s2_lo = vdup_lane_u8(abv, 1);
    uint8x8_t s3_lo = vdup_lane_u8(abv, 2);
    uint8x8_t s4_lo = vdup_lane_u8(abv, 3);

    uint8x8_t res_lo =
        filter_intra_predictor(s0_lo, s1_lo, s2_lo, s3_lo, s4_lo, s5_lo, s6_lo,
                               f0, f1, f2, f3, f4, f5, f6);

    uint8x8_t s0_hi = s4_lo;
    uint8x8_t s1_hi = vdup_lane_u8(abv, 4);
    uint8x8_t s2_hi = vdup_lane_u8(abv, 5);
    uint8x8_t s3_hi = vdup_lane_u8(abv, 6);
    uint8x8_t s4_hi = vdup_lane_u8(abv, 7);
    uint8x8_t s5_hi = vdup_lane_u8(res_lo, 3);
    uint8x8_t s6_hi = vdup_lane_u8(res_lo, 7);

    uint8x8_t res_hi =
        filter_intra_predictor(s0_hi, s1_hi, s2_hi, s3_hi, s4_hi, s5_hi, s6_hi,
                               f0, f1, f2, f3, f4, f5, f6);

    uint32x2x2_t res_zip =
        vzip_u32(vreinterpret_u32_u8(res_lo), vreinterpret_u32_u8(res_hi));

    vst1_u8(dst + 0 * stride, vreinterpret_u8_u32(res_zip.val[0]));
    vst1_u8(dst + 1 * stride, vreinterpret_u8_u32(res_zip.val[1]));

    left += 1;
    int h = height - 2;
    dst += 2 * stride;
    do {
      s0_lo = vdup_n_u8(left[0]);
      s5_lo = vdup_n_u8(left[1]);
      s6_lo = vdup_n_u8(left[2]);

      s1_lo = vdup_lane_u8(res_lo, 4);
      s2_lo = vdup_lane_u8(res_lo, 5);
      s3_lo = vdup_lane_u8(res_lo, 6);
      s4_lo = vdup_lane_u8(res_lo, 7);

      res_lo = filter_intra_predictor(s0_lo, s1_lo, s2_lo, s3_lo, s4_lo, s5_lo,
                                      s6_lo, f0, f1, f2, f3, f4, f5, f6);

      s0_hi = s4_lo;
      s1_hi = vdup_lane_u8(res_hi, 4);
      s2_hi = vdup_lane_u8(res_hi, 5);
      s3_hi = vdup_lane_u8(res_hi, 6);
      s4_hi = vdup_lane_u8(res_hi, 7);
      s5_hi = vdup_lane_u8(res_lo, 3);
      s6_hi = vdup_lane_u8(res_lo, 7);

      res_hi = filter_intra_predictor(s0_hi, s1_hi, s2_hi, s3_hi, s4_hi, s5_hi,
                                      s6_hi, f0, f1, f2, f3, f4, f5, f6);

      res_zip =
          vzip_u32(vreinterpret_u32_u8(res_lo), vreinterpret_u32_u8(res_hi));

      vst1_u8(dst + 0 * stride, vreinterpret_u8_u32(res_zip.val[0]));
      vst1_u8(dst + 1 * stride, vreinterpret_u8_u32(res_zip.val[1]));

      left += 2;
      dst += 2 * stride;
      h -= 2;
    } while (h != 0);
  } else {
    assert(width >= 16 && (width & 7) == 0);
    uint8x8_t s0_lo = vdup_n_u8(above[-1]);
    uint8x8_t s5_lo = vdup_n_u8(left[0]);
    uint8x8_t s6_lo = vdup_n_u8(left[1]);

    int c = 0;
    do {
      uint8x8_t s1234 = vld1_u8(above + c);
      uint8x8_t s1_lo = vdup_lane_u8(s1234, 0);
      uint8x8_t s2_lo = vdup_lane_u8(s1234, 1);
      uint8x8_t s3_lo = vdup_lane_u8(s1234, 2);
      uint8x8_t s4_lo = vdup_lane_u8(s1234, 3);

      uint8x8_t res_lo =
          filter_intra_predictor(s0_lo, s1_lo, s2_lo, s3_lo, s4_lo, s5_lo,
                                 s6_lo, f0, f1, f2, f3, f4, f5, f6);

      uint8x8_t s0_hi = s4_lo;
      uint8x8_t s1_hi = vdup_lane_u8(s1234, 4);
      uint8x8_t s2_hi = vdup_lane_u8(s1234, 5);
      uint8x8_t s3_hi = vdup_lane_u8(s1234, 6);
      uint8x8_t s4_hi = vdup_lane_u8(s1234, 7);
      uint8x8_t s5_hi = vdup_lane_u8(res_lo, 3);
      uint8x8_t s6_hi = vdup_lane_u8(res_lo, 7);

      uint8x8_t res_hi =
          filter_intra_predictor(s0_hi, s1_hi, s2_hi, s3_hi, s4_hi, s5_hi,
                                 s6_hi, f0, f1, f2, f3, f4, f5, f6);

      uint32x2x2_t res =
          vzip_u32(vreinterpret_u32_u8(res_lo), vreinterpret_u32_u8(res_hi));

      vst1_u8(dst + 0 * stride + c, vreinterpret_u8_u32(res.val[0]));
      vst1_u8(dst + 1 * stride + c, vreinterpret_u8_u32(res.val[1]));

      s0_lo = s4_hi;
      s5_lo = vdup_lane_u8(res_hi, 3);
      s6_lo = vdup_lane_u8(res_hi, 7);

      c += 8;
    } while (c != width);

    left += 1;
    int h = height - 2;
    dst += 2 * stride;
    do {
      s0_lo = vdup_n_u8(left[0]);
      s5_lo = vdup_n_u8(left[1]);
      s6_lo = vdup_n_u8(left[2]);

      c = 0;
      do {
        uint8x8_t s1234 = vld1_u8(dst - stride + c);
        uint8x8_t s1_lo = vdup_lane_u8(s1234, 0);
        uint8x8_t s2_lo = vdup_lane_u8(s1234, 1);
        uint8x8_t s3_lo = vdup_lane_u8(s1234, 2);
        uint8x8_t s4_lo = vdup_lane_u8(s1234, 3);

        uint8x8_t res_lo =
            filter_intra_predictor(s0_lo, s1_lo, s2_lo, s3_lo, s4_lo, s5_lo,
                                   s6_lo, f0, f1, f2, f3, f4, f5, f6);

        uint8x8_t s0_hi = s4_lo;
        uint8x8_t s1_hi = vdup_lane_u8(s1234, 4);
        uint8x8_t s2_hi = vdup_lane_u8(s1234, 5);
        uint8x8_t s3_hi = vdup_lane_u8(s1234, 6);
        uint8x8_t s4_hi = vdup_lane_u8(s1234, 7);
        uint8x8_t s5_hi = vdup_lane_u8(res_lo, 3);
        uint8x8_t s6_hi = vdup_lane_u8(res_lo, 7);

        uint8x8_t res_hi =
            filter_intra_predictor(s0_hi, s1_hi, s2_hi, s3_hi, s4_hi, s5_hi,
                                   s6_hi, f0, f1, f2, f3, f4, f5, f6);

        uint32x2x2_t res =
            vzip_u32(vreinterpret_u32_u8(res_lo), vreinterpret_u32_u8(res_hi));

        vst1_u8(dst + 0 * stride + c, vreinterpret_u8_u32(res.val[0]));
        vst1_u8(dst + 1 * stride + c, vreinterpret_u8_u32(res.val[1]));

        s0_lo = s4_hi;
        s5_lo = vdup_lane_u8(res_hi, 3);
        s6_lo = vdup_lane_u8(res_hi, 7);

        c += 8;
      } while (c != width);

      left += 2;
      dst += 2 * stride;
      h -= 2;
    } while (h != 0);
  }
}

DECLARE_ALIGNED(16, static const uint8_t,
                kEdgeTailIdx[16]) = { 0, 1, 2,  3,  4,  5,  6,  7,
                                      8, 9, 10, 11, 12, 13, 14, 15 };

static inline uint8x8_t load_padded_u8x8(const uint8_t *p, int n, uint8_t pad) {
  assert(n >= 1 && n <= 8);
  if (n == 8) return vld1_u8(p);
  uint64_t val = 0;
  int shift = 0;
  if (n & 4) {
    uint32_t w;
    memcpy(&w, p, 4);
    val = (uint64_t)w;
    p += 4;
    shift = 32;
  }
  if (n & 2) {
    uint16_t h;
    memcpy(&h, p, 2);
    val |= (uint64_t)h << shift;
    p += 2;
    shift += 16;
  }
  if (n & 1) {
    val |= (uint64_t)(*p) << shift;
  }
  const uint64_t pad64 = (uint64_t)pad * 0x0101010101010101ULL;
  const uint64_t mask = (1ULL << (n * 8)) - 1ULL;
  return vcreate_u8(val | (pad64 & ~mask));
}

static inline uint8x8_t load_tail_u8x8(const uint8_t *end_ptr, int k) {
  assert(k >= 1 && k <= 8);
  const uint8x8_t raw = vld1_u8(end_ptr - 8);
  const uint8x8_t shift = vdup_n_u8((uint8_t)(8 - k));
  const uint8x8_t idx =
      vmin_u8(vadd_u8(vld1_u8(kEdgeTailIdx), shift), vdup_n_u8(7));
  return vtbl1_u8(raw, idx);
}

static inline void store_partial_u8x8(uint8_t *dst, uint8x8_t res, int n) {
  assert(n >= 1 && n <= 8);
  if (n == 8) {
    vst1_u8(dst, res);
    return;
  }
  uint64_t val = vget_lane_u64(vreinterpret_u64_u8(res), 0);
  if (n & 4) {
    uint32_t w = (uint32_t)val;
    memcpy(dst, &w, 4);
    dst += 4;
    val >>= 32;
  }
  if (n & 2) {
    uint16_t h = (uint16_t)val;
    memcpy(dst, &h, 2);
    dst += 2;
    val >>= 16;
  }
  if (n & 1) {
    *dst = (uint8_t)val;
  }
}

static inline void store_partial_u8x16(uint8_t *dst, uint8x16_t res, int n) {
  assert(n >= 1 && n <= 16);
  if (n == 16) {
    vst1q_u8(dst, res);
    return;
  }
  uint8x8_t lo = vget_low_u8(res);
  if (n >= 8) {
    vst1_u8(dst, lo);
    dst += 8;
    n -= 8;
    if (!n) return;
    lo = vget_high_u8(res);
  }
  store_partial_u8x8(dst, lo, n);
}

static inline uint8x16_t load_tail_u8x16(const uint8_t *end_ptr, int k) {
  assert(k >= 1 && k <= 15);
  const uint8x16_t raw = vld1q_u8(end_ptr - 16);
  const uint8x16_t shift = vdupq_n_u8((uint8_t)(16 - k));
  const uint8x16_t idx =
      vminq_u8(vaddq_u8(vld1q_u8(kEdgeTailIdx), shift), vdupq_n_u8(15));
#if AOM_ARCH_AARCH64
  return vqtbl1q_u8(raw, idx);
#else
  const uint8x8x2_t tbl = { { vget_low_u8(raw), vget_high_u8(raw) } };
  return vcombine_u8(vtbl2_u8(tbl, vget_low_u8(idx)),
                     vtbl2_u8(tbl, vget_high_u8(idx)));
#endif
}

static inline uint8x8_t filter_edge_s1_x8(uint8x8_t in0, uint8x8_t in1) {
  const uint8x8_t s1 = vext_u8(in0, in1, 1);
  const uint8x8_t s2 = vext_u8(in0, in1, 2);
  return vrhadd_u8(vhadd_u8(in0, s2), s1);
}

static inline uint8x16_t filter_edge_s1_x16(uint8x16_t in0, uint8x16_t in1) {
  const uint8x16_t s1 = vextq_u8(in0, in1, 1);
  const uint8x16_t s2 = vextq_u8(in0, in1, 2);
  return vrhaddq_u8(vhaddq_u8(in0, s2), s1);
}

static inline uint8x8_t filter_edge_s2_x8(uint8x8_t in0, uint8x8_t in1,
                                          uint8x8_t f5, uint8x8_t f6) {
  const uint8x8_t s0 = in0;
  const uint8x8_t s1 = vext_u8(in0, in1, 1);
  const uint8x8_t s2 = vext_u8(in0, in1, 2);
  uint16x8_t accum = vmull_u8(s0, f5);
  accum = vmlal_u8(accum, s1, f6);
  accum = vmlal_u8(accum, s2, f5);
  return vrshrn_n_u16(accum, 4);
}

static inline uint8x16_t filter_edge_s2_x16(uint8x16_t in0, uint8x16_t in1,
                                            uint8x16_t f5q, uint8x16_t f6q) {
  const uint8x16_t s0 = in0;
  const uint8x16_t s1 = vextq_u8(in0, in1, 1);
  const uint8x16_t s2 = vextq_u8(in0, in1, 2);

  uint16x8_t acc_lo = vmull_u8(vget_low_u8(s0), vget_low_u8(f5q));
  acc_lo = vmlal_u8(acc_lo, vget_low_u8(s1), vget_low_u8(f6q));
  acc_lo = vmlal_u8(acc_lo, vget_low_u8(s2), vget_low_u8(f5q));
  const uint8x8_t res_lo = vrshrn_n_u16(acc_lo, 4);

#if AOM_ARCH_AARCH64
  uint16x8_t acc_hi = vmull_high_u8(s0, f5q);
  acc_hi = vmlal_high_u8(acc_hi, s1, f6q);
  acc_hi = vmlal_high_u8(acc_hi, s2, f5q);
#else
  uint16x8_t acc_hi = vmull_u8(vget_high_u8(s0), vget_high_u8(f5q));
  acc_hi = vmlal_u8(acc_hi, vget_high_u8(s1), vget_high_u8(f6q));
  acc_hi = vmlal_u8(acc_hi, vget_high_u8(s2), vget_high_u8(f5q));
#endif
  const uint8x8_t res_hi = vrshrn_n_u16(acc_hi, 4);

  return vcombine_u8(res_lo, res_hi);
}

static inline uint8x8_t filter_edge_s3_x8(uint8x8_t prev, uint8x8_t in0,
                                          uint8x8_t in1) {
  const uint8x8_t s0 = vext_u8(prev, in0, 7);
  const uint8x8_t s1 = in0;
  const uint8x8_t s2 = vext_u8(in0, in1, 1);
  const uint8x8_t s3 = vext_u8(in0, in1, 2);
  const uint8x8_t s4 = vext_u8(in0, in1, 3);

  const uint8x8_t h04 = vhadd_u8(s0, s4);
  uint16x8_t sum = vaddl_u8(s1, s2);
  sum = vaddw_u8(sum, s3);
  sum = vaddw_u8(sum, h04);
  return vrshrn_n_u16(sum, 2);
}

static inline uint8x16_t filter_edge_s3_x16(uint8x16_t prev, uint8x16_t in0,
                                            uint8x16_t in1) {
  const uint8x16_t s0 = vextq_u8(prev, in0, 15);
  const uint8x16_t s1 = in0;
  const uint8x16_t s2 = vextq_u8(in0, in1, 1);
  const uint8x16_t s3 = vextq_u8(in0, in1, 2);
  const uint8x16_t s4 = vextq_u8(in0, in1, 3);

  const uint8x16_t h04 = vhaddq_u8(s0, s4);
  uint16x8_t sum_lo = vaddl_u8(vget_low_u8(s1), vget_low_u8(s2));
  sum_lo = vaddw_u8(sum_lo, vget_low_u8(s3));
  sum_lo = vaddw_u8(sum_lo, vget_low_u8(h04));
  const uint8x8_t res_lo = vrshrn_n_u16(sum_lo, 2);

#if AOM_ARCH_AARCH64
  uint16x8_t sum_hi = vaddl_high_u8(s1, s2);
  sum_hi = vaddw_high_u8(sum_hi, s3);
  sum_hi = vaddw_high_u8(sum_hi, h04);
#else
  uint16x8_t sum_hi = vaddl_u8(vget_high_u8(s1), vget_high_u8(s2));
  sum_hi = vaddw_u8(sum_hi, vget_high_u8(s3));
  sum_hi = vaddw_u8(sum_hi, vget_high_u8(h04));
#endif
  const uint8x8_t res_hi = vrshrn_n_u16(sum_hi, 2);

  return vcombine_u8(res_lo, res_hi);
}

void av1_filter_intra_edge_neon(uint8_t *p, int sz, int strength) {
  assert(sz >= 0 && sz <= 129);
  assert(strength >= 0 && strength <= 3);
  if (!strength || sz <= 1) return;

  uint8_t *dst = p + 1;
  int rem = sz - 1;

  if (sz <= 9) {
    const uint8_t last = p[sz - 1];
    const uint8x8_t in0 =
        (sz == 9) ? vld1_u8(p) : load_padded_u8x8(p, sz, last);
    const uint8x8_t in1 = vdup_n_u8(last);
    uint8x8_t res;
    if (strength == 1) {
      res = filter_edge_s1_x8(in0, in1);
    } else if (strength == 2) {
      res = filter_edge_s2_x8(in0, in1, vdup_n_u8(5), vdup_n_u8(6));
    } else {
      res = filter_edge_s3_x8(vdup_n_u8(p[0]), in0, in1);
    }
    store_partial_u8x8(dst, res, rem);
    return;
  }

  const uint8_t *const end_ptr = p + sz;
  const uint8_t last = end_ptr[-1];
  const uint8_t *src = p;
  uint8x16_t in0;
  if (sz < 16) {
    in0 = vcombine_u8(vld1_u8(p), load_tail_u8x8(end_ptr, sz - 8));
  } else {
    in0 = vld1q_u8(src);
  }

  uint8x16_t tail_res;
  if (strength == 1) {
    while (rem >= 31) {
      const uint8x16_t in1 = vld1q_u8(src + 16);
      vst1q_u8(dst, filter_edge_s1_x16(in0, in1));
      in0 = in1;
      src += 16;
      dst += 16;
      rem -= 16;
    }
    if (rem > 16) {
      const uint8x16_t in1 = load_tail_u8x16(end_ptr, rem - 15);
      vst1q_u8(dst, filter_edge_s1_x16(in0, in1));
      in0 = in1;
      dst += 16;
      rem -= 16;
    }
    tail_res = filter_edge_s1_x16(in0, vdupq_n_u8(last));
  } else if (strength == 2) {
    const uint8x16_t f5q = vdupq_n_u8(5);
    const uint8x16_t f6q = vdupq_n_u8(6);
    while (rem >= 31) {
      const uint8x16_t in1 = vld1q_u8(src + 16);
      vst1q_u8(dst, filter_edge_s2_x16(in0, in1, f5q, f6q));
      in0 = in1;
      src += 16;
      dst += 16;
      rem -= 16;
    }
    if (rem > 16) {
      const uint8x16_t in1 = load_tail_u8x16(end_ptr, rem - 15);
      vst1q_u8(dst, filter_edge_s2_x16(in0, in1, f5q, f6q));
      in0 = in1;
      dst += 16;
      rem -= 16;
    }
    tail_res = filter_edge_s2_x16(in0, vdupq_n_u8(last), f5q, f6q);
  } else {
    uint8x16_t prev = vdupq_lane_u8(vget_low_u8(in0), 0);
    while (rem >= 31) {
      const uint8x16_t in1 = vld1q_u8(src + 16);
      vst1q_u8(dst, filter_edge_s3_x16(prev, in0, in1));
      prev = in0;
      in0 = in1;
      src += 16;
      dst += 16;
      rem -= 16;
    }
    if (rem > 16) {
      const uint8x16_t in1 = load_tail_u8x16(end_ptr, rem - 15);
      vst1q_u8(dst, filter_edge_s3_x16(prev, in0, in1));
      prev = in0;
      in0 = in1;
      dst += 16;
      rem -= 16;
    }
    tail_res = filter_edge_s3_x16(prev, in0, vdupq_n_u8(last));
  }
  store_partial_u8x16(dst, tail_res, rem);
}

static inline uint8x8_t upsample_edge_x8(uint8x8_t s0, uint8x8_t s1,
                                         uint8x8_t s2, uint8x8_t s3) {
  const int16x8_t t0 = vreinterpretq_s16_u16(vaddl_u8(s0, s3));
  int16x8_t t1 = vreinterpretq_s16_u16(vaddl_u8(s1, s2));
  t1 = vmulq_n_s16(t1, 9);
  t1 = vsubq_s16(t1, t0);
  return vqrshrun_n_s16(t1, 4);
}

void av1_upsample_intra_edge_neon(uint8_t *p, int sz) {
  if (!sz) return;

  assert(sz == 4 || sz == 8 || sz == 12 || sz == 16);

  const uint8_t pm1_val = p[-1];
  p[-2] = pm1_val;

  if (sz == 4) {
    const uint8x8_t p03 = load_unaligned_dup_u8_4x2(p);
    const uint8x8_t tail = vdup_lane_u8(p03, 3);
    const uint8x8_t s2 = vreinterpret_u8_u64(
        vsli_n_u64(vreinterpret_u64_u8(p03), vreinterpret_u64_u8(tail), 32));
    const uint8x8_t pm1 = vdup_n_u8(pm1_val);
    const uint8x8_t s0 = vext_u8(pm1, s2, 6);
    const uint8x8_t s1 = vext_u8(pm1, s2, 7);
    const uint8x8_t s3 = vext_u8(s2, s2, 1);
    const uint8x8_t interp = upsample_edge_x8(s0, s1, s2, s3);
    const uint8x8x2_t zipped = vzip_u8(interp, s2);
    vst1_u8(p - 1, zipped.val[0]);
    return;
  }

  if (sz == 8) {
    const uint8x8_t s2 = vld1_u8(p);
    const uint8x8_t pm1 = vdup_n_u8(pm1_val);
    const uint8x8_t tail = vdup_lane_u8(s2, 7);
    const uint8x8_t s0 = vext_u8(pm1, s2, 6);
    const uint8x8_t s1 = vext_u8(pm1, s2, 7);
    const uint8x8_t s3 = vext_u8(s2, tail, 1);
    const uint8x8x2_t res = { { upsample_edge_x8(s0, s1, s2, s3), s2 } };
    vst2_u8(p - 1, res);
    return;
  }

  assert(sz == 12 || sz == 16);
  uint8x16_t s2_q;
  if (sz == 16) {
    s2_q = vld1q_u8(p);
  } else {
    const uint8x8_t lo = vld1_u8(p);
    const uint8x8_t p8_11 = load_unaligned_dup_u8_4x2(p + 8);
    const uint8x8_t tail8 = vdup_lane_u8(p8_11, 3);
    const uint8x8_t hi = vreinterpret_u8_u64(
        vsli_n_u64(vreinterpret_u64_u8(p8_11), vreinterpret_u64_u8(tail8), 32));
    s2_q = vcombine_u8(lo, hi);
  }
  const uint8x16_t tail_q = vdupq_lane_u8(vget_high_u8(s2_q), 7);
  const uint8x16_t pm1_q = vdupq_n_u8(pm1_val);
  const uint8x16_t s0_q = vextq_u8(pm1_q, s2_q, 14);
  const uint8x16_t s1_q = vextq_u8(pm1_q, s2_q, 15);
  const uint8x16_t s3_q = vextq_u8(s2_q, tail_q, 1);

  const uint8x8_t res_lo =
      upsample_edge_x8(vget_low_u8(s0_q), vget_low_u8(s1_q), vget_low_u8(s2_q),
                       vget_low_u8(s3_q));

#if AOM_ARCH_AARCH64
  const int16x8_t t0_hi = vreinterpretq_s16_u16(vaddl_high_u8(s0_q, s3_q));
  int16x8_t t1_hi = vreinterpretq_s16_u16(vaddl_high_u8(s1_q, s2_q));
  t1_hi = vmulq_n_s16(t1_hi, 9);
  t1_hi = vsubq_s16(t1_hi, t0_hi);
  const uint8x8_t res_hi = vqrshrun_n_s16(t1_hi, 4);
#else
  const uint8x8_t res_hi =
      upsample_edge_x8(vget_high_u8(s0_q), vget_high_u8(s1_q),
                       vget_high_u8(s2_q), vget_high_u8(s3_q));
#endif

  if (sz == 16) {
    const uint8x16x2_t res = { { vcombine_u8(res_lo, res_hi), s2_q } };
    vst2q_u8(p - 1, res);
  } else {
    const uint8x8x2_t res_lo2 = { { res_lo, vget_low_u8(s2_q) } };
    vst2_u8(p - 1, res_lo2);
    const uint8x8x2_t zipped_hi = vzip_u8(res_hi, vget_high_u8(s2_q));
    vst1_u8(p + 15, zipped_hi.val[0]);
  }
}
