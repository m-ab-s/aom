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

#include "av1/common/convolve.h"
#include "av1/common/enums.h"
#include "av1/common/filter.h"

static INLINE void compute_dist_wtd_avg_4x4(
    uint16x4_t dd0, uint16x4_t dd1, uint16x4_t dd2, uint16x4_t dd3,
    uint16x4_t d0, uint16x4_t d1, uint16x4_t d2, uint16x4_t d3,
    const uint16_t fwd_offset, const uint16_t bck_offset,
    const int16x8_t round_offset, uint8x8_t *d01_u8, uint8x8_t *d23_u8) {
  uint32x4_t blend0 = vmull_n_u16(dd0, fwd_offset);
  blend0 = vmlal_n_u16(blend0, d0, bck_offset);
  uint32x4_t blend1 = vmull_n_u16(dd1, fwd_offset);
  blend1 = vmlal_n_u16(blend1, d1, bck_offset);
  uint32x4_t blend2 = vmull_n_u16(dd2, fwd_offset);
  blend2 = vmlal_n_u16(blend2, d2, bck_offset);
  uint32x4_t blend3 = vmull_n_u16(dd3, fwd_offset);
  blend3 = vmlal_n_u16(blend3, d3, bck_offset);

  uint16x4_t avg0 = vshrn_n_u32(blend0, DIST_PRECISION_BITS);
  uint16x4_t avg1 = vshrn_n_u32(blend1, DIST_PRECISION_BITS);
  uint16x4_t avg2 = vshrn_n_u32(blend2, DIST_PRECISION_BITS);
  uint16x4_t avg3 = vshrn_n_u32(blend3, DIST_PRECISION_BITS);

  int16x8_t dst_01 = vreinterpretq_s16_u16(vcombine_u16(avg0, avg1));
  int16x8_t dst_23 = vreinterpretq_s16_u16(vcombine_u16(avg2, avg3));

  dst_01 = vsubq_s16(dst_01, round_offset);
  dst_23 = vsubq_s16(dst_23, round_offset);

  *d01_u8 = vqrshrun_n_s16(dst_01, FILTER_BITS - ROUND0_BITS);
  *d23_u8 = vqrshrun_n_s16(dst_23, FILTER_BITS - ROUND0_BITS);
}

static INLINE void compute_basic_avg_4x4(uint16x4_t dd0, uint16x4_t dd1,
                                         uint16x4_t dd2, uint16x4_t dd3,
                                         uint16x4_t d0, uint16x4_t d1,
                                         uint16x4_t d2, uint16x4_t d3,
                                         const int16x8_t round_offset,
                                         uint8x8_t *d01_u8, uint8x8_t *d23_u8) {
  uint16x4_t avg0 = vhadd_u16(dd0, d0);
  uint16x4_t avg1 = vhadd_u16(dd1, d1);
  uint16x4_t avg2 = vhadd_u16(dd2, d2);
  uint16x4_t avg3 = vhadd_u16(dd3, d3);

  int16x8_t dst_01 = vreinterpretq_s16_u16(vcombine_u16(avg0, avg1));
  int16x8_t dst_23 = vreinterpretq_s16_u16(vcombine_u16(avg2, avg3));

  dst_01 = vsubq_s16(dst_01, round_offset);
  dst_23 = vsubq_s16(dst_23, round_offset);

  *d01_u8 = vqrshrun_n_s16(dst_01, FILTER_BITS - ROUND0_BITS);
  *d23_u8 = vqrshrun_n_s16(dst_23, FILTER_BITS - ROUND0_BITS);
}

static INLINE void compute_dist_wtd_avg_8x4(
    uint16x8_t dd0, uint16x8_t dd1, uint16x8_t dd2, uint16x8_t dd3,
    uint16x8_t d0, uint16x8_t d1, uint16x8_t d2, uint16x8_t d3,
    const uint16_t fwd_offset, const uint16_t bck_offset,
    const int16x8_t round_offset, uint8x8_t *d0_u8, uint8x8_t *d1_u8,
    uint8x8_t *d2_u8, uint8x8_t *d3_u8) {
  uint32x4_t blend0_lo = vmull_n_u16(vget_low_u16(dd0), fwd_offset);
  blend0_lo = vmlal_n_u16(blend0_lo, vget_low_u16(d0), bck_offset);
  uint32x4_t blend0_hi = vmull_n_u16(vget_high_u16(dd0), fwd_offset);
  blend0_hi = vmlal_n_u16(blend0_hi, vget_high_u16(d0), bck_offset);

  uint32x4_t blend1_lo = vmull_n_u16(vget_low_u16(dd1), fwd_offset);
  blend1_lo = vmlal_n_u16(blend1_lo, vget_low_u16(d1), bck_offset);
  uint32x4_t blend1_hi = vmull_n_u16(vget_high_u16(dd1), fwd_offset);
  blend1_hi = vmlal_n_u16(blend1_hi, vget_high_u16(d1), bck_offset);

  uint32x4_t blend2_lo = vmull_n_u16(vget_low_u16(dd2), fwd_offset);
  blend2_lo = vmlal_n_u16(blend2_lo, vget_low_u16(d2), bck_offset);
  uint32x4_t blend2_hi = vmull_n_u16(vget_high_u16(dd2), fwd_offset);
  blend2_hi = vmlal_n_u16(blend2_hi, vget_high_u16(d2), bck_offset);

  uint32x4_t blend3_lo = vmull_n_u16(vget_low_u16(dd3), fwd_offset);
  blend3_lo = vmlal_n_u16(blend3_lo, vget_low_u16(d3), bck_offset);
  uint32x4_t blend3_hi = vmull_n_u16(vget_high_u16(dd3), fwd_offset);
  blend3_hi = vmlal_n_u16(blend3_hi, vget_high_u16(d3), bck_offset);

  uint16x8_t avg0 = vcombine_u16(vshrn_n_u32(blend0_lo, DIST_PRECISION_BITS),
                                 vshrn_n_u32(blend0_hi, DIST_PRECISION_BITS));
  uint16x8_t avg1 = vcombine_u16(vshrn_n_u32(blend1_lo, DIST_PRECISION_BITS),
                                 vshrn_n_u32(blend1_hi, DIST_PRECISION_BITS));
  uint16x8_t avg2 = vcombine_u16(vshrn_n_u32(blend2_lo, DIST_PRECISION_BITS),
                                 vshrn_n_u32(blend2_hi, DIST_PRECISION_BITS));
  uint16x8_t avg3 = vcombine_u16(vshrn_n_u32(blend3_lo, DIST_PRECISION_BITS),
                                 vshrn_n_u32(blend3_hi, DIST_PRECISION_BITS));

  int16x8_t dst0 = vsubq_s16(vreinterpretq_s16_u16(avg0), round_offset);
  int16x8_t dst1 = vsubq_s16(vreinterpretq_s16_u16(avg1), round_offset);
  int16x8_t dst2 = vsubq_s16(vreinterpretq_s16_u16(avg2), round_offset);
  int16x8_t dst3 = vsubq_s16(vreinterpretq_s16_u16(avg3), round_offset);

  *d0_u8 = vqrshrun_n_s16(dst0, FILTER_BITS - ROUND0_BITS);
  *d1_u8 = vqrshrun_n_s16(dst1, FILTER_BITS - ROUND0_BITS);
  *d2_u8 = vqrshrun_n_s16(dst2, FILTER_BITS - ROUND0_BITS);
  *d3_u8 = vqrshrun_n_s16(dst3, FILTER_BITS - ROUND0_BITS);
}

static INLINE void compute_basic_avg_8x4(uint16x8_t dd0, uint16x8_t dd1,
                                         uint16x8_t dd2, uint16x8_t dd3,
                                         uint16x8_t d0, uint16x8_t d1,
                                         uint16x8_t d2, uint16x8_t d3,
                                         const int16x8_t round_offset,
                                         uint8x8_t *d0_u8, uint8x8_t *d1_u8,
                                         uint8x8_t *d2_u8, uint8x8_t *d3_u8) {
  uint16x8_t avg0 = vhaddq_u16(dd0, d0);
  uint16x8_t avg1 = vhaddq_u16(dd1, d1);
  uint16x8_t avg2 = vhaddq_u16(dd2, d2);
  uint16x8_t avg3 = vhaddq_u16(dd3, d3);

  int16x8_t dst0 = vsubq_s16(vreinterpretq_s16_u16(avg0), round_offset);
  int16x8_t dst1 = vsubq_s16(vreinterpretq_s16_u16(avg1), round_offset);
  int16x8_t dst2 = vsubq_s16(vreinterpretq_s16_u16(avg2), round_offset);
  int16x8_t dst3 = vsubq_s16(vreinterpretq_s16_u16(avg3), round_offset);

  *d0_u8 = vqrshrun_n_s16(dst0, FILTER_BITS - ROUND0_BITS);
  *d1_u8 = vqrshrun_n_s16(dst1, FILTER_BITS - ROUND0_BITS);
  *d2_u8 = vqrshrun_n_s16(dst2, FILTER_BITS - ROUND0_BITS);
  *d3_u8 = vqrshrun_n_s16(dst3, FILTER_BITS - ROUND0_BITS);
}
