/*
 * Copyright (c) 2025, Alliance for Open Media. All rights reserved
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

#include "aom_dsp/arm/mem_neon.h"

#define FILTER_INTRA_SCALE_BITS 4

// Taps split into contiguous top-4 [t1, t2, t3, t4] (rows 0..3 and 4..7) and
// left-3 [t0, t5, t6, 0] (rows 0..3 and 4..7) so top pixels from the previous
// row pair require zero permutation and accumulate in parallel with left-col
// dependencies via SUDOT.
DECLARE_ALIGNED(16, static const int8_t,
                av1_filter_intra_taps_neon_i8mm[FILTER_INTRA_MODES][4][16]) = {
  {
      { 10, 0, 0, 0, 2, 10, 0, 0, 1, 1, 10, 0, 1, 1, 2, 10 },
      { 6, 0, 0, 0, 2, 6, 0, 0, 2, 2, 6, 0, 1, 2, 2, 6 },
      { -6, 12, 0, 0, -5, 9, 0, 0, -3, 7, 0, 0, -3, 5, 0, 0 },
      { -4, 2, 12, 0, -3, 2, 9, 0, -3, 2, 7, 0, -3, 3, 5, 0 },
  },
  {
      { 16, 0, 0, 0, 0, 16, 0, 0, 0, 0, 16, 0, 0, 0, 0, 16 },
      { 16, 0, 0, 0, 0, 16, 0, 0, 0, 0, 16, 0, 0, 0, 0, 16 },
      { -10, 10, 0, 0, -6, 6, 0, 0, -4, 4, 0, 0, -2, 2, 0, 0 },
      { -10, 0, 10, 0, -6, 0, 6, 0, -4, 0, 4, 0, -2, 0, 2, 0 },
  },
  {
      { 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8 },
      { 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4 },
      { -8, 16, 0, 0, -8, 16, 0, 0, -8, 16, 0, 0, -8, 16, 0, 0 },
      { -4, 0, 16, 0, -4, 0, 16, 0, -4, 0, 16, 0, -4, 0, 16, 0 },
  },
  {
      { 8, 0, 0, 0, 3, 8, 0, 0, 2, 3, 8, 0, 1, 2, 3, 8 },
      { 4, 0, 0, 0, 3, 4, 0, 0, 2, 3, 4, 0, 2, 2, 3, 4 },
      { -2, 10, 0, 0, -1, 6, 0, 0, -1, 4, 0, 0, 0, 2, 0, 0 },
      { -1, 3, 10, 0, -1, 4, 6, 0, -1, 4, 4, 0, -1, 3, 3, 0 },
  },
  {
      { 14, 0, 0, 0, 0, 14, 0, 0, 0, 0, 14, 0, 0, 0, 0, 14 },
      { 12, 0, 0, 0, 1, 12, 0, 0, 0, 0, 12, 0, 0, 0, 1, 12 },
      { -12, 14, 0, 0, -10, 12, 0, 0, -9, 11, 0, 0, -8, 10, 0, 0 },
      { -10, 0, 14, 0, -9, 0, 12, 0, -8, 1, 11, 0, -7, 1, 9, 0 },
  },
};

// Filter sums lie in [-3060, 7140], fitting in signed 16-bit lanes without
// saturation before vqrshrun_n_s16.
static inline uint8x8_t filter_intra_finish(int32x4_t acc0, int32x4_t acc1) {
  const int16x8_t acc =
      vuzp1q_s16(vreinterpretq_s16_s32(acc0), vreinterpretq_s16_s32(acc1));
  return vqrshrun_n_s16(acc, FILTER_INTRA_SCALE_BITS);
}

static inline uint8x8_t filter_intra_left_first_lane0(
    uint8x8_t p_left, uint8x8_t p_top, int8x16_t ft0, int8x16_t ft1,
    int8x16_t fl0, int8x16_t fl1) {
  int32x4_t acc0 = vsudotq_lane_s32(vdupq_n_s32(0), fl0, p_left, 0);
  int32x4_t acc1 = vsudotq_lane_s32(vdupq_n_s32(0), fl1, p_left, 0);
  acc0 = vsudotq_lane_s32(acc0, ft0, p_top, 0);
  acc1 = vsudotq_lane_s32(acc1, ft1, p_top, 0);
  return filter_intra_finish(acc0, acc1);
}

static inline uint8x8_t filter_intra_left_first_lane1(
    uint8x8_t p_left, uint8x8_t p_top, int8x16_t ft0, int8x16_t ft1,
    int8x16_t fl0, int8x16_t fl1) {
  int32x4_t acc0 = vsudotq_lane_s32(vdupq_n_s32(0), fl0, p_left, 0);
  int32x4_t acc1 = vsudotq_lane_s32(vdupq_n_s32(0), fl1, p_left, 0);
  acc0 = vsudotq_lane_s32(acc0, ft0, p_top, 1);
  acc1 = vsudotq_lane_s32(acc1, ft1, p_top, 1);
  return filter_intra_finish(acc0, acc1);
}

static inline uint8x8_t filter_intra_top_first_lane0(
    uint8x8_t p_top, uint8x8_t p_left, int8x16_t ft0, int8x16_t ft1,
    int8x16_t fl0, int8x16_t fl1) {
  int32x4_t acc0 = vsudotq_lane_s32(vdupq_n_s32(0), ft0, p_top, 0);
  int32x4_t acc1 = vsudotq_lane_s32(vdupq_n_s32(0), ft1, p_top, 0);
  acc0 = vsudotq_lane_s32(acc0, fl0, p_left, 0);
  acc1 = vsudotq_lane_s32(acc1, fl1, p_left, 0);
  return filter_intra_finish(acc0, acc1);
}

static inline uint8x8_t filter_intra_top_first_lane1(
    uint8x8_t p_top, uint8x8_t p_left, int8x16_t ft0, int8x16_t ft1,
    int8x16_t fl0, int8x16_t fl1) {
  int32x4_t acc0 = vsudotq_lane_s32(vdupq_n_s32(0), ft0, p_top, 1);
  int32x4_t acc1 = vsudotq_lane_s32(vdupq_n_s32(0), ft1, p_top, 1);
  acc0 = vsudotq_lane_s32(acc0, fl0, p_left, 0);
  acc1 = vsudotq_lane_s32(acc1, fl1, p_left, 0);
  return filter_intra_finish(acc0, acc1);
}

static inline uint8x8_t gather_p_left(uint8x8_t cur_left, uint8x8_t prev_top,
                                      uint8x16_t tbl_idx) {
  return vget_low_u8(vqtbl1q_u8(vcombine_u8(prev_top, cur_left), tbl_idx));
}

void av1_filter_intra_predictor_neon_i8mm(uint8_t *dst, ptrdiff_t stride,
                                          TX_SIZE tx_size, const uint8_t *above,
                                          const uint8_t *left, int mode) {
  assert(tx_size < TX_SIZES_ALL);
  assert(mode >= 0 && mode < FILTER_INTRA_MODES);
  const int bw = tx_size_wide[tx_size];
  const int bh = tx_size_high[tx_size];

  assert(bw >= 4 && bw <= 32 && bh >= 4 && bh <= 32);
  assert((bw & 3) == 0 && (bh & 1) == 0);

  const int8x16_t ft0 = vld1q_s8(av1_filter_intra_taps_neon_i8mm[mode][0]);
  const int8x16_t ft1 = vld1q_s8(av1_filter_intra_taps_neon_i8mm[mode][1]);
  const int8x16_t fl0 = vld1q_s8(av1_filter_intra_taps_neon_i8mm[mode][2]);
  const int8x16_t fl1 = vld1q_s8(av1_filter_intra_taps_neon_i8mm[mode][3]);

  if (bw == 4) {
    uint16_t l12;
    memcpy(&l12, left, 2);
    uint32_t l012 = (uint32_t)above[-1] | ((uint32_t)l12 << 8);
    uint8_t l_prev = (uint8_t)(l12 >> 8);
    uint8x8_t prev = filter_intra_left_first_lane0(
        vreinterpret_u8_u32(vdup_n_u32(l012)), load_unaligned_dup_u8_4x2(above),
        ft0, ft1, fl0, fl1);
    store_u8x4_strided_x2(dst, stride, prev);
    left += 2;
    dst += 2 * stride;
    int h = bh - 2;
    do {
      memcpy(&l12, left, 2);
      l012 = (uint32_t)l_prev | ((uint32_t)l12 << 8);
      l_prev = (uint8_t)(l12 >> 8);
      prev = filter_intra_left_first_lane1(
          vreinterpret_u8_u32(vdup_n_u32(l012)), prev, ft0, ft1, fl0, fl1);
      store_u8x4_strided_x2(dst, stride, prev);
      left += 2;
      dst += 2 * stride;
      h -= 2;
    } while (h != 0);
    return;
  }

  // Gather [p0=prev_top[3] (lane 3), p5=cur_left[3] (11), p6=cur_left[7] (15),
  // 0] or [p0=prev_top[7] (lane 7), p5=cur_left[3] (11), p6=cur_left[7] (15),
  // 0] for the next 4x2 sub-block in row pairs.
  const uint8x16_t idx_tl_lane3 = vreinterpretq_u8_u32(vdupq_n_u32(0xFF0F0B03));
  const uint8x16_t idx_tl_lane7 = vreinterpretq_u8_u32(vdupq_n_u32(0xFF0F0B07));

  if (bw == 8) {
    const uint8x8_t abv = vld1_u8(above);
    uint16_t l12;
    memcpy(&l12, left, 2);
    uint32_t l012 = (uint32_t)above[-1] | ((uint32_t)l12 << 8);
    uint8_t l_prev = (uint8_t)(l12 >> 8);

    uint8x8_t prev0, prev1;
    {
      int32x4_t acc1_0 = vsudotq_lane_s32(vdupq_n_s32(0), ft0, abv, 1);
      int32x4_t acc1_1 = vsudotq_lane_s32(vdupq_n_s32(0), ft1, abv, 1);

      prev0 = filter_intra_left_first_lane0(
          vreinterpret_u8_u32(vdup_n_u32(l012)), abv, ft0, ft1, fl0, fl1);

      const uint8x8_t p_left1 = gather_p_left(prev0, abv, idx_tl_lane3);
      acc1_0 = vsudotq_lane_s32(acc1_0, fl0, p_left1, 0);
      acc1_1 = vsudotq_lane_s32(acc1_1, fl1, p_left1, 0);
      prev1 = filter_intra_finish(acc1_0, acc1_1);

      const uint32x2x2_t res_zip =
          vzip_u32(vreinterpret_u32_u8(prev0), vreinterpret_u32_u8(prev1));
      vst1_u8(dst, vreinterpret_u8_u32(res_zip.val[0]));
      vst1_u8(dst + stride, vreinterpret_u8_u32(res_zip.val[1]));
    }

    left += 2;
    dst += 2 * stride;
    int h = bh - 2;
    do {
      memcpy(&l12, left, 2);
      l012 = (uint32_t)l_prev | ((uint32_t)l12 << 8);
      l_prev = (uint8_t)(l12 >> 8);

      int32x4_t acc1_0 = vsudotq_lane_s32(vdupq_n_s32(0), ft0, prev1, 1);
      int32x4_t acc1_1 = vsudotq_lane_s32(vdupq_n_s32(0), ft1, prev1, 1);

      const uint8x8_t next0 = filter_intra_left_first_lane1(
          vreinterpret_u8_u32(vdup_n_u32(l012)), prev0, ft0, ft1, fl0, fl1);

      const uint8x8_t p_left1 = gather_p_left(next0, prev0, idx_tl_lane7);
      acc1_0 = vsudotq_lane_s32(acc1_0, fl0, p_left1, 0);
      acc1_1 = vsudotq_lane_s32(acc1_1, fl1, p_left1, 0);
      const uint8x8_t next1 = filter_intra_finish(acc1_0, acc1_1);

      const uint32x2x2_t res_zip =
          vzip_u32(vreinterpret_u32_u8(next0), vreinterpret_u32_u8(next1));
      vst1_u8(dst, vreinterpret_u8_u32(res_zip.val[0]));
      vst1_u8(dst + stride, vreinterpret_u8_u32(res_zip.val[1]));
      prev0 = next0;
      prev1 = next1;
      left += 2;
      dst += 2 * stride;
      h -= 2;
    } while (h != 0);
    return;
  }

  if (bw == 16) {
    const uint8x8_t abv_lo = vld1_u8(above);
    const uint8x8_t abv_hi = vld1_u8(above + 8);
    uint16_t l12;
    memcpy(&l12, left, 2);
    uint32_t l012 = (uint32_t)above[-1] | ((uint32_t)l12 << 8);
    uint8_t l_prev = (uint8_t)(l12 >> 8);

    uint8x8_t prev0, prev1, prev2, prev3;
    {
      int32x4_t a1_0 = vsudotq_lane_s32(vdupq_n_s32(0), ft0, abv_lo, 1);
      int32x4_t a1_1 = vsudotq_lane_s32(vdupq_n_s32(0), ft1, abv_lo, 1);
      int32x4_t a2_0 = vsudotq_lane_s32(vdupq_n_s32(0), ft0, abv_hi, 0);
      int32x4_t a2_1 = vsudotq_lane_s32(vdupq_n_s32(0), ft1, abv_hi, 0);
      int32x4_t a3_0 = vsudotq_lane_s32(vdupq_n_s32(0), ft0, abv_hi, 1);
      int32x4_t a3_1 = vsudotq_lane_s32(vdupq_n_s32(0), ft1, abv_hi, 1);

      prev0 = filter_intra_left_first_lane0(
          vreinterpret_u8_u32(vdup_n_u32(l012)), abv_lo, ft0, ft1, fl0, fl1);

      const uint8x8_t pl1 = gather_p_left(prev0, abv_lo, idx_tl_lane3);
      a1_0 = vsudotq_lane_s32(a1_0, fl0, pl1, 0);
      a1_1 = vsudotq_lane_s32(a1_1, fl1, pl1, 0);
      prev1 = filter_intra_finish(a1_0, a1_1);

      const uint8x8_t pl2 = gather_p_left(prev1, abv_lo, idx_tl_lane7);
      a2_0 = vsudotq_lane_s32(a2_0, fl0, pl2, 0);
      a2_1 = vsudotq_lane_s32(a2_1, fl1, pl2, 0);
      prev2 = filter_intra_finish(a2_0, a2_1);

      const uint8x8_t pl3 = gather_p_left(prev2, abv_hi, idx_tl_lane3);
      a3_0 = vsudotq_lane_s32(a3_0, fl0, pl3, 0);
      a3_1 = vsudotq_lane_s32(a3_1, fl1, pl3, 0);
      prev3 = filter_intra_finish(a3_0, a3_1);

      const uint32x2x2_t z01 =
          vzip_u32(vreinterpret_u32_u8(prev0), vreinterpret_u32_u8(prev1));
      const uint32x2x2_t z23 =
          vzip_u32(vreinterpret_u32_u8(prev2), vreinterpret_u32_u8(prev3));
      vst1q_u8(dst, vcombine_u8(vreinterpret_u8_u32(z01.val[0]),
                                vreinterpret_u8_u32(z23.val[0])));
      vst1q_u8(dst + stride, vcombine_u8(vreinterpret_u8_u32(z01.val[1]),
                                         vreinterpret_u8_u32(z23.val[1])));
    }

    left += 2;
    dst += 2 * stride;
    int h = bh - 2;
    do {
      memcpy(&l12, left, 2);
      l012 = (uint32_t)l_prev | ((uint32_t)l12 << 8);
      l_prev = (uint8_t)(l12 >> 8);

      int32x4_t a1_0 = vsudotq_lane_s32(vdupq_n_s32(0), ft0, prev1, 1);
      int32x4_t a1_1 = vsudotq_lane_s32(vdupq_n_s32(0), ft1, prev1, 1);
      int32x4_t a2_0 = vsudotq_lane_s32(vdupq_n_s32(0), ft0, prev2, 1);
      int32x4_t a2_1 = vsudotq_lane_s32(vdupq_n_s32(0), ft1, prev2, 1);
      int32x4_t a3_0 = vsudotq_lane_s32(vdupq_n_s32(0), ft0, prev3, 1);
      int32x4_t a3_1 = vsudotq_lane_s32(vdupq_n_s32(0), ft1, prev3, 1);

      const uint8x8_t next0 = filter_intra_left_first_lane1(
          vreinterpret_u8_u32(vdup_n_u32(l012)), prev0, ft0, ft1, fl0, fl1);

      const uint8x8_t pl1 = gather_p_left(next0, prev0, idx_tl_lane7);
      a1_0 = vsudotq_lane_s32(a1_0, fl0, pl1, 0);
      a1_1 = vsudotq_lane_s32(a1_1, fl1, pl1, 0);
      const uint8x8_t next1 = filter_intra_finish(a1_0, a1_1);

      const uint8x8_t pl2 = gather_p_left(next1, prev1, idx_tl_lane7);
      a2_0 = vsudotq_lane_s32(a2_0, fl0, pl2, 0);
      a2_1 = vsudotq_lane_s32(a2_1, fl1, pl2, 0);
      const uint8x8_t next2 = filter_intra_finish(a2_0, a2_1);

      const uint8x8_t pl3 = gather_p_left(next2, prev2, idx_tl_lane7);
      a3_0 = vsudotq_lane_s32(a3_0, fl0, pl3, 0);
      a3_1 = vsudotq_lane_s32(a3_1, fl1, pl3, 0);
      const uint8x8_t next3 = filter_intra_finish(a3_0, a3_1);

      const uint32x2x2_t z01 =
          vzip_u32(vreinterpret_u32_u8(next0), vreinterpret_u32_u8(next1));
      const uint32x2x2_t z23 =
          vzip_u32(vreinterpret_u32_u8(next2), vreinterpret_u32_u8(next3));
      vst1q_u8(dst, vcombine_u8(vreinterpret_u8_u32(z01.val[0]),
                                vreinterpret_u8_u32(z23.val[0])));
      vst1q_u8(dst + stride, vcombine_u8(vreinterpret_u8_u32(z01.val[1]),
                                         vreinterpret_u8_u32(z23.val[1])));
      prev0 = next0;
      prev1 = next1;
      prev2 = next2;
      prev3 = next3;
      left += 2;
      dst += 2 * stride;
      h -= 2;
    } while (h != 0);
    return;
  }

  assert(bw == 32);
  {
    uint16_t l12;
    memcpy(&l12, left, 2);
    uint32_t l012 = (uint32_t)above[-1] | ((uint32_t)l12 << 8);
    uint8_t l_prev = (uint8_t)(l12 >> 8);
    {
      uint8x8_t abv8 = vld1_u8(above);
      uint8x8_t res = filter_intra_left_first_lane0(
          vreinterpret_u8_u32(vdup_n_u32(l012)), abv8, ft0, ft1, fl0, fl1);
      store_u8x4_strided_x2(dst, stride, res);
      res = filter_intra_top_first_lane1(
          abv8, gather_p_left(res, abv8, idx_tl_lane3), ft0, ft1, fl0, fl1);
      store_u8x4_strided_x2(dst + 4, stride, res);
      int c = 8;
      do {
        const uint8x8_t prev_abv8 = abv8;
        abv8 = vld1_u8(above + c);
        res = filter_intra_top_first_lane0(
            abv8, gather_p_left(res, prev_abv8, idx_tl_lane7), ft0, ft1, fl0,
            fl1);
        store_u8x4_strided_x2(dst + c, stride, res);
        res = filter_intra_top_first_lane1(
            abv8, gather_p_left(res, abv8, idx_tl_lane3), ft0, ft1, fl0, fl1);
        store_u8x4_strided_x2(dst + c + 4, stride, res);
        c += 8;
      } while (c != 32);
    }

    left += 2;
    dst += 2 * stride;
    int h = bh - 2;
    do {
      const uint8_t *ptr = dst - stride;
      memcpy(&l12, left, 2);
      l012 = (uint32_t)l_prev | ((uint32_t)l12 << 8);
      l_prev = (uint8_t)(l12 >> 8);
      uint8x8_t abv8 = vld1_u8(ptr);
      uint8x8_t res = filter_intra_left_first_lane0(
          vreinterpret_u8_u32(vdup_n_u32(l012)), abv8, ft0, ft1, fl0, fl1);
      store_u8x4_strided_x2(dst, stride, res);
      res = filter_intra_top_first_lane1(
          abv8, gather_p_left(res, abv8, idx_tl_lane3), ft0, ft1, fl0, fl1);
      store_u8x4_strided_x2(dst + 4, stride, res);
      int c = 8;
      do {
        const uint8x8_t prev_abv8 = abv8;
        abv8 = vld1_u8(ptr + c);
        res = filter_intra_top_first_lane0(
            abv8, gather_p_left(res, prev_abv8, idx_tl_lane7), ft0, ft1, fl0,
            fl1);
        store_u8x4_strided_x2(dst + c, stride, res);
        res = filter_intra_top_first_lane1(
            abv8, gather_p_left(res, abv8, idx_tl_lane3), ft0, ft1, fl0, fl1);
        store_u8x4_strided_x2(dst + c + 4, stride, res);
        c += 8;
      } while (c != 32);
      left += 2;
      dst += 2 * stride;
      h -= 2;
    } while (h != 0);
  }
}
