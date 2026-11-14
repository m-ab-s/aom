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

#include <assert.h>

#include <arm_neon.h>

#include "aom_dsp/arm/sum_neon.h"
#include "av1/encoder/rdopt.h"
#include "config/aom_config.h"
#include "config/av1_rtcd.h"

void av1_get_horver_correlation_full_neon(const int16_t *diff, int stride,
                                          int width, int height, float *hcorr,
                                          float *vcorr) {
  assert(width >= 4 && height >= 4);
  assert(width == 4 || (width & 7) == 0);

  int64_t xy_sum = 0, xz_sum = 0;
  int64_t x_sum = 0, x2_sum = 0;
  int64_t x_firstrow = 0, x_finalrow = 0;
  int64_t x2_firstrow = 0, x2_finalrow = 0;
  int64_t x_firstcol = 0, x_finalcol = 0;
  int64_t x2_firstcol = 0, x2_finalcol = 0;

  if (width == 4) {
    const int16x4_t zero4 = vdup_n_s16(0);
    int16x4_t prev = vld1_s16(diff);
    const int16x4_t prev_sli = vext_s16(zero4, prev, 3);
    int32x4_t col_x = vmovl_s16(prev);
    int32x4_t col_x2 = vmull_s16(prev, prev);
    int32x4_t xy_v = vmull_s16(prev, prev_sli);
    int32x4_t xz_v = vdupq_n_s32(0);

    x_firstrow = horizontal_add_s32x4(col_x);
    x2_firstrow = horizontal_long_add_s32x4(col_x2);

    int64x2_t v_col_x2_lo = vdupq_n_s64(0);
    int64x2_t v_col_x2_hi = vdupq_n_s64(0);
    int64x2_t v_xy_sum = vdupq_n_s64(0);
    int64x2_t v_xz_sum = vdupq_n_s64(0);

    const int16_t *p = diff + stride;
    int i = 1;
    do {
      const int strip_end = (i + 32 < height) ? (i + 32) : height;
      do {
        const int16x4_t cur = vld1_s16(p);
        const int16x4_t cur_sli = vext_s16(zero4, cur, 3);
        col_x = vaddw_s16(col_x, cur);
        col_x2 = vmlal_s16(col_x2, cur, cur);
        xy_v = vmlal_s16(xy_v, cur, cur_sli);
        xz_v = vmlal_s16(xz_v, prev, cur);
        prev = cur;
        p += stride;
        ++i;
      } while (i != strip_end);
      v_col_x2_lo = vaddw_s32(v_col_x2_lo, vget_low_s32(col_x2));
      v_col_x2_hi = vaddw_s32(v_col_x2_hi, vget_high_s32(col_x2));
      v_xy_sum = vpadalq_s32(v_xy_sum, xy_v);
      v_xz_sum = vpadalq_s32(v_xz_sum, xz_v);
      col_x2 = vdupq_n_s32(0);
      xy_v = vdupq_n_s32(0);
      xz_v = vdupq_n_s32(0);
    } while (i != height);

    const int32x4_t last_x2 = vmull_s16(prev, prev);
    x_finalrow = horizontal_add_s32x4(vmovl_s16(prev));
    x2_finalrow = horizontal_long_add_s32x4(last_x2);

    x_sum = horizontal_long_add_s32x4(col_x);
    x_firstcol = vgetq_lane_s32(col_x, 0);
    x_finalcol = vgetq_lane_s32(col_x, 3);
    x2_sum = horizontal_add_s64x2(vaddq_s64(v_col_x2_lo, v_col_x2_hi));
    x2_firstcol = vgetq_lane_s64(v_col_x2_lo, 0);
    x2_finalcol = vgetq_lane_s64(v_col_x2_hi, 1);
    xy_sum = horizontal_add_s64x2(v_xy_sum);
    xz_sum = horizontal_add_s64x2(v_xz_sum);
  } else if (width == 8) {
    const int16x8_t zero8 = vdupq_n_s16(0);
    int16x8_t prev = vld1q_s16(diff);
    const int16x8_t prev_sli = vextq_s16(zero8, prev, 7);
    int32x4_t x_v = vpaddlq_s16(prev);
    int32x4_t x2_v = vmull_s16(vget_low_s16(prev), vget_low_s16(prev));
    x2_v = vmlal_s16(x2_v, vget_high_s16(prev), vget_high_s16(prev));
    int32x4_t xy_v = vmull_s16(vget_low_s16(prev), vget_low_s16(prev_sli));
    xy_v = vmlal_s16(xy_v, vget_high_s16(prev), vget_high_s16(prev_sli));
    int32x4_t xz_v = vdupq_n_s32(0);

    x_firstrow = horizontal_add_s32x4(x_v);
    x2_firstrow = horizontal_long_add_s32x4(x2_v);
    x_firstcol = diff[0];
    x_finalcol = diff[7];
    x2_firstcol = x_firstcol * x_firstcol;
    x2_finalcol = x_finalcol * x_finalcol;

    int64x2_t v_x_sum = vdupq_n_s64(0);
    int64x2_t v_x2_sum = vdupq_n_s64(0);
    int64x2_t v_xy_sum = vdupq_n_s64(0);
    int64x2_t v_xz_sum = vdupq_n_s64(0);

    const int16_t *p = diff + stride;
    int i = 1;
    do {
      const int strip_end = (i + 16 < height) ? (i + 16) : height;
      do {
        const int16x8_t cur = vld1q_s16(p);
        const int16x8_t cur_sli = vextq_s16(zero8, cur, 7);
        const int16x4_t cur_lo = vget_low_s16(cur);
        const int16x4_t cur_hi = vget_high_s16(cur);
        const int16x4_t sli_lo = vget_low_s16(cur_sli);
        const int16x4_t sli_hi = vget_high_s16(cur_sli);
        const int16x4_t prev_lo = vget_low_s16(prev);
        const int16x4_t prev_hi = vget_high_s16(prev);

        x_v = vpadalq_s16(x_v, cur);
        x2_v = vmlal_s16(x2_v, cur_lo, cur_lo);
        x2_v = vmlal_s16(x2_v, cur_hi, cur_hi);
        xy_v = vmlal_s16(xy_v, cur_lo, sli_lo);
        xy_v = vmlal_s16(xy_v, cur_hi, sli_hi);
        xz_v = vmlal_s16(xz_v, prev_lo, cur_lo);
        xz_v = vmlal_s16(xz_v, prev_hi, cur_hi);

        const int64_t fc = p[0];
        const int64_t lc = p[7];
        x_firstcol += fc;
        x_finalcol += lc;
        x2_firstcol += fc * fc;
        x2_finalcol += lc * lc;
        prev = cur;
        p += stride;
        ++i;
      } while (i != strip_end);
      v_x_sum = vpadalq_s32(v_x_sum, x_v);
      v_x2_sum = vpadalq_s32(v_x2_sum, x2_v);
      v_xy_sum = vpadalq_s32(v_xy_sum, xy_v);
      v_xz_sum = vpadalq_s32(v_xz_sum, xz_v);
      x_v = vdupq_n_s32(0);
      x2_v = vdupq_n_s32(0);
      xy_v = vdupq_n_s32(0);
      xz_v = vdupq_n_s32(0);
    } while (i != height);

    int32x4_t last_x2 = vmull_s16(vget_low_s16(prev), vget_low_s16(prev));
    last_x2 = vmlal_s16(last_x2, vget_high_s16(prev), vget_high_s16(prev));
    x_finalrow = horizontal_add_s16x8(prev);
    x2_finalrow = horizontal_long_add_s32x4(last_x2);

    x_sum = horizontal_add_s64x2(v_x_sum);
    x2_sum = horizontal_add_s64x2(v_x2_sum);
    xy_sum = horizontal_add_s64x2(v_xy_sum);
    xz_sum = horizontal_add_s64x2(v_xz_sum);
  } else {
    // Strip heights (32 for w==4, 16 for w==8) and per-row s32 accumulators
    // (up to 128 / 4 = 32 squares per lane) stay < INT32_MAX for 12-bit diffs.
    const int16x8_t zero8 = vdupq_n_s16(0);
    int64x2_t v_x_sum = vdupq_n_s64(0);
    int64x2_t v_x2_sum = vdupq_n_s64(0);
    int64x2_t v_xy_sum = vdupq_n_s64(0);
    int64x2_t v_xz_sum = vdupq_n_s64(0);

    // Row 0 (peel j = 0 to avoid zero-initializing row accumulators)
    {
      const int16x8_t cur0 = vld1q_s16(diff);
      const int16x8_t cur0_sli = vextq_s16(zero8, cur0, 7);
      const int16x4_t cur0_lo = vget_low_s16(cur0);
      const int16x4_t cur0_hi = vget_high_s16(cur0);
      int16x8_t prev_vec = cur0;
      int32x4_t x_32 = vpaddlq_s16(cur0);
      int32x4_t x2_32 =
          vmlal_s16(vmull_s16(cur0_lo, cur0_lo), cur0_hi, cur0_hi);
      int32x4_t xy_32 = vmlal_s16(vmull_s16(cur0_lo, vget_low_s16(cur0_sli)),
                                  cur0_hi, vget_high_s16(cur0_sli));
      int j = 8;
      do {
        const int16x8_t cur = vld1q_s16(diff + j);
        const int16x8_t cur_sli = vextq_s16(prev_vec, cur, 7);
        const int16x4_t cur_lo = vget_low_s16(cur);
        const int16x4_t cur_hi = vget_high_s16(cur);
        prev_vec = cur;
        x_32 = vpadalq_s16(x_32, cur);
        x2_32 = vmlal_s16(x2_32, cur_lo, cur_lo);
        x2_32 = vmlal_s16(x2_32, cur_hi, cur_hi);
        xy_32 = vmlal_s16(xy_32, cur_lo, vget_low_s16(cur_sli));
        xy_32 = vmlal_s16(xy_32, cur_hi, vget_high_s16(cur_sli));
        j += 8;
      } while (j != width);
      x_firstrow = horizontal_add_s32x4(x_32);
      x2_firstrow = horizontal_long_add_s32x4(x2_32);
      xy_sum += horizontal_long_add_s32x4(xy_32);
      x_sum += x_firstrow;
      x2_sum += x2_firstrow;
      const int64_t fc = diff[0];
      const int64_t lc = diff[width - 1];
      x_firstcol = fc;
      x_finalcol = lc;
      x2_firstcol = fc * fc;
      x2_finalcol = lc * lc;
    }

    // Rows 1 .. height - 1
    const int16_t *row_prev = diff;
    const int16_t *row_cur = diff + stride;
    int i = 1;
    do {
      const int16x8_t cur0 = vld1q_s16(row_cur);
      const int16x8_t above0 = vld1q_s16(row_prev);
      const int16x8_t cur0_sli = vextq_s16(zero8, cur0, 7);
      const int16x4_t cur0_lo = vget_low_s16(cur0);
      const int16x4_t cur0_hi = vget_high_s16(cur0);
      int16x8_t prev_vec = cur0;
      int32x4_t x_32 = vpaddlq_s16(cur0);
      int32x4_t x2_32 =
          vmlal_s16(vmull_s16(cur0_lo, cur0_lo), cur0_hi, cur0_hi);
      int32x4_t xy_32 = vmlal_s16(vmull_s16(cur0_lo, vget_low_s16(cur0_sli)),
                                  cur0_hi, vget_high_s16(cur0_sli));
      int32x4_t xz_32 = vmlal_s16(vmull_s16(vget_low_s16(above0), cur0_lo),
                                  vget_high_s16(above0), cur0_hi);

      int j = 8;
      do {
        const int16x8_t cur = vld1q_s16(row_cur + j);
        const int16x8_t above = vld1q_s16(row_prev + j);
        const int16x8_t cur_sli = vextq_s16(prev_vec, cur, 7);
        const int16x4_t cur_lo = vget_low_s16(cur);
        const int16x4_t cur_hi = vget_high_s16(cur);
        prev_vec = cur;

        x_32 = vpadalq_s16(x_32, cur);
        x2_32 = vmlal_s16(x2_32, cur_lo, cur_lo);
        x2_32 = vmlal_s16(x2_32, cur_hi, cur_hi);
        xy_32 = vmlal_s16(xy_32, cur_lo, vget_low_s16(cur_sli));
        xy_32 = vmlal_s16(xy_32, cur_hi, vget_high_s16(cur_sli));
        xz_32 = vmlal_s16(xz_32, vget_low_s16(above), cur_lo);
        xz_32 = vmlal_s16(xz_32, vget_high_s16(above), cur_hi);
        j += 8;
      } while (j != width);

      if (i == height - 1) {
        x_finalrow = horizontal_add_s32x4(x_32);
        x2_finalrow = horizontal_long_add_s32x4(x2_32);
        x_sum += x_finalrow;
        x2_sum += x2_finalrow;
      } else {
        v_x_sum = vpadalq_s32(v_x_sum, x_32);
        v_x2_sum = vpadalq_s32(v_x2_sum, x2_32);
      }
      v_xy_sum = vpadalq_s32(v_xy_sum, xy_32);
      v_xz_sum = vpadalq_s32(v_xz_sum, xz_32);

      const int64_t fc = row_cur[0];
      const int64_t lc = row_cur[width - 1];
      x_firstcol += fc;
      x_finalcol += lc;
      x2_firstcol += fc * fc;
      x2_finalcol += lc * lc;
      row_prev = row_cur;
      row_cur += stride;
      ++i;
    } while (i != height);

    x_sum += horizontal_add_s64x2(v_x_sum);
    x2_sum += horizontal_add_s64x2(v_x2_sum);
    xy_sum += horizontal_add_s64x2(v_xy_sum);
    xz_sum += horizontal_add_s64x2(v_xz_sum);
  }

  int64_t xhor_sum = x_sum - x_finalcol;
  int64_t xver_sum = x_sum - x_finalrow;
  int64_t y_sum = x_sum - x_firstcol;
  int64_t z_sum = x_sum - x_firstrow;
  int64_t x2hor_sum = x2_sum - x2_finalcol;
  int64_t x2ver_sum = x2_sum - x2_finalrow;
  int64_t y2_sum = x2_sum - x2_firstcol;
  int64_t z2_sum = x2_sum - x2_firstrow;

  const float num_hor = (float)(height * (width - 1));
  const float num_ver = (float)((height - 1) * width);

  const float xhor_var_n = x2hor_sum - (xhor_sum * xhor_sum) / num_hor;
  const float xver_var_n = x2ver_sum - (xver_sum * xver_sum) / num_ver;

  const float y_var_n = y2_sum - (y_sum * y_sum) / num_hor;
  const float z_var_n = z2_sum - (z_sum * z_sum) / num_ver;

  const float xy_var_n = xy_sum - (xhor_sum * y_sum) / num_hor;
  const float xz_var_n = xz_sum - (xver_sum * z_sum) / num_ver;

  if (xhor_var_n > 0 && y_var_n > 0) {
    *hcorr = xy_var_n / sqrtf(xhor_var_n * y_var_n);
    *hcorr = *hcorr < 0 ? 0 : *hcorr;
  } else {
    *hcorr = 1.0;
  }
  if (xver_var_n > 0 && z_var_n > 0) {
    *vcorr = xz_var_n / sqrtf(xver_var_n * z_var_n);
    *vcorr = *vcorr < 0 ? 0 : *vcorr;
  } else {
    *vcorr = 1.0;
  }
}

static inline uint32x4_t energy_sq_s16x4(int16x4_t a) {
  return vreinterpretq_u32_s32(vmull_s16(a, a));
}

static inline uint32x4_t energy_mla_sq_s16x4(uint32x4_t acc, int16x4_t a) {
  return vreinterpretq_u32_s32(vmlal_s16(vreinterpretq_s32_u32(acc), a, a));
}

static inline uint32x4_t energy_pair8_1row(const int16_t *r) {
  const int16x8_t d = vld1q_s16(r);
  const uint32x4_t sq_lo = energy_sq_s16x4(vget_low_s16(d));
  const uint32x4_t sq_hi = energy_sq_s16x4(vget_high_s16(d));
  return horizontal_add_2d_u32(sq_lo, sq_hi);
}

static inline uint32x4_t energy_pair8_2row(const int16_t *r0,
                                           const int16_t *r1) {
  const int16x8_t d0 = vld1q_s16(r0);
  const int16x8_t d1 = vld1q_s16(r1);
  uint32x4_t sq_lo = energy_sq_s16x4(vget_low_s16(d0));
  uint32x4_t sq_hi = energy_sq_s16x4(vget_high_s16(d0));
  sq_lo = energy_mla_sq_s16x4(sq_lo, vget_low_s16(d1));
  sq_hi = energy_mla_sq_s16x4(sq_hi, vget_high_s16(d1));
  return horizontal_add_2d_u32(sq_lo, sq_hi);
}

static inline uint64_t energy_dist_w4(const int16_t *diff, int stride,
                                      int esq_h, int h_shift, uint32_t *col_sum,
                                      uint32_t *row_sum) {
  uint32x4_t v_col;
  if (h_shift == 0) {
    const uint32x4_t sq0 = energy_sq_s16x4(vld1_s16(diff));
    const uint32x4_t sq1 = energy_sq_s16x4(vld1_s16(diff + stride));
    const uint32x4_t sq2 = energy_sq_s16x4(vld1_s16(diff + 2 * stride));
    const uint32x4_t sq3 = energy_sq_s16x4(vld1_s16(diff + 3 * stride));
    v_col = vaddq_u32(vaddq_u32(sq0, sq1), vaddq_u32(sq2, sq3));
    const uint32x4_t sum4[4] = { sq0, sq1, sq2, sq3 };
    vst1q_u32(row_sum, horizontal_add_4d_u32x4(sum4));
    for (esq_h -= 4; esq_h != 0; esq_h -= 4) {
      diff += 4 * stride;
      row_sum += 4;
      const uint32x4_t s0 = energy_sq_s16x4(vld1_s16(diff));
      const uint32x4_t s1 = energy_sq_s16x4(vld1_s16(diff + stride));
      const uint32x4_t s2 = energy_sq_s16x4(vld1_s16(diff + 2 * stride));
      const uint32x4_t s3 = energy_sq_s16x4(vld1_s16(diff + 3 * stride));
      v_col = vaddq_u32(v_col, vaddq_u32(vaddq_u32(s0, s1), vaddq_u32(s2, s3)));
      const uint32x4_t s4[4] = { s0, s1, s2, s3 };
      vst1q_u32(row_sum, horizontal_add_4d_u32x4(s4));
    }
  } else {
    uint32x4_t sum4[4];
    for (int k = 0; k < 4; ++k) {
      sum4[k] = energy_mla_sq_s16x4(energy_sq_s16x4(vld1_s16(diff)),
                                    vld1_s16(diff + stride));
      diff += 2 * stride;
    }
    v_col = vaddq_u32(vaddq_u32(sum4[0], sum4[1]), vaddq_u32(sum4[2], sum4[3]));
    vst1q_u32(row_sum, horizontal_add_4d_u32x4(sum4));
    for (esq_h -= 4; esq_h != 0; esq_h -= 4) {
      row_sum += 4;
      for (int k = 0; k < 4; ++k) {
        const uint32x4_t sq = energy_mla_sq_s16x4(
            energy_sq_s16x4(vld1_s16(diff)), vld1_s16(diff + stride));
        v_col = vaddq_u32(v_col, sq);
        sum4[k] = sq;
        diff += 2 * stride;
      }
      vst1q_u32(row_sum, horizontal_add_4d_u32x4(sum4));
    }
  }
  vst1q_u32(col_sum, v_col);
  return horizontal_long_add_u32x4(v_col);
}

static inline uint64_t energy_dist_w8(const int16_t *diff, int stride,
                                      int esq_h, int h_shift, uint32_t *col_sum,
                                      uint32_t *row_sum) {
  uint32x4_t v_col0, v_col1;
  int i = 0;
  if (h_shift == 0) {
    do {
      uint32x4_t sum4[4];
      const int16x8_t d0 = vld1q_s16(diff + i * stride);
      const uint32x4_t sq_lo0 = energy_sq_s16x4(vget_low_s16(d0));
      const uint32x4_t sq_hi0 = energy_sq_s16x4(vget_high_s16(d0));
      v_col0 = (i == 0) ? sq_lo0 : vaddq_u32(v_col0, sq_lo0);
      v_col1 = (i == 0) ? sq_hi0 : vaddq_u32(v_col1, sq_hi0);
      sum4[0] = vaddq_u32(sq_lo0, sq_hi0);
      for (int k = 1; k < 4; ++k) {
        const int16x8_t d = vld1q_s16(diff + (i + k) * stride);
        const uint32x4_t sq_lo = energy_sq_s16x4(vget_low_s16(d));
        const uint32x4_t sq_hi = energy_sq_s16x4(vget_high_s16(d));
        v_col0 = vaddq_u32(v_col0, sq_lo);
        v_col1 = vaddq_u32(v_col1, sq_hi);
        sum4[k] = vaddq_u32(sq_lo, sq_hi);
      }
      vst1q_u32(row_sum + i, horizontal_add_4d_u32x4(sum4));
      i += 4;
    } while (i != esq_h);
  } else {
    do {
      uint32x4_t sum4[4];
      for (int k = 0; k < 4; ++k) {
        const int16_t *r = diff + 2 * (i + k) * stride;
        const int16x8_t d0 = vld1q_s16(r);
        const int16x8_t d1 = vld1q_s16(r + stride);
        uint32x4_t sq_lo = energy_sq_s16x4(vget_low_s16(d0));
        uint32x4_t sq_hi = energy_sq_s16x4(vget_high_s16(d0));
        sq_lo = energy_mla_sq_s16x4(sq_lo, vget_low_s16(d1));
        sq_hi = energy_mla_sq_s16x4(sq_hi, vget_high_s16(d1));
        v_col0 = (i == 0 && k == 0) ? sq_lo : vaddq_u32(v_col0, sq_lo);
        v_col1 = (i == 0 && k == 0) ? sq_hi : vaddq_u32(v_col1, sq_hi);
        sum4[k] = vaddq_u32(sq_lo, sq_hi);
      }
      vst1q_u32(row_sum + i, horizontal_add_4d_u32x4(sum4));
      i += 4;
    } while (i != esq_h);
  }
  vst1q_u32(col_sum, v_col0);
  vst1q_u32(col_sum + 4, v_col1);
  const uint32x4_t v_cols[2] = { v_col0, v_col1 };
  return horizontal_long_add_u32x4_x2(v_cols);
}

static inline uint64_t energy_dist_w16(const int16_t *diff, int stride,
                                       int esq_h, int h_shift,
                                       uint32_t *col_sum, uint32_t *row_sum) {
  uint32x4_t v_col0, v_col1;
  int i = 0;
  if (h_shift == 0) {
    do {
      uint32x4_t sum4[4];
      for (int k = 0; k < 4; ++k) {
        const int16_t *r = diff + (i + k) * stride;
        const uint32x4_t p0 = energy_pair8_1row(r);
        const uint32x4_t p1 = energy_pair8_1row(r + 8);
        v_col0 = (i == 0 && k == 0) ? p0 : vaddq_u32(v_col0, p0);
        v_col1 = (i == 0 && k == 0) ? p1 : vaddq_u32(v_col1, p1);
        sum4[k] = vaddq_u32(p0, p1);
      }
      vst1q_u32(row_sum + i, horizontal_add_4d_u32x4(sum4));
      i += 4;
    } while (i != esq_h);
  } else {
    do {
      uint32x4_t sum4[4];
      for (int k = 0; k < 4; ++k) {
        const int16_t *r0 = diff + 2 * (i + k) * stride;
        const int16_t *r1 = r0 + stride;
        const uint32x4_t p0 = energy_pair8_2row(r0, r1);
        const uint32x4_t p1 = energy_pair8_2row(r0 + 8, r1 + 8);
        v_col0 = (i == 0 && k == 0) ? p0 : vaddq_u32(v_col0, p0);
        v_col1 = (i == 0 && k == 0) ? p1 : vaddq_u32(v_col1, p1);
        sum4[k] = vaddq_u32(p0, p1);
      }
      vst1q_u32(row_sum + i, horizontal_add_4d_u32x4(sum4));
      i += 4;
    } while (i != esq_h);
  }
  vst1q_u32(col_sum, v_col0);
  vst1q_u32(col_sum + 4, v_col1);
  const uint32x4_t v_cols[2] = { v_col0, v_col1 };
  return horizontal_long_add_u32x4_x2(v_cols);
}

static inline uint64_t energy_dist_w32(const int16_t *diff, int stride,
                                       int esq_h, int h_shift,
                                       uint32_t *col_sum, uint32_t *row_sum) {
  uint32x4_t v_col[4];
  int i = 0;
  if (h_shift == 0) {
    do {
      uint32x4_t sum4[4];
      for (int k = 0; k < 4; ++k) {
        const int16_t *r = diff + (i + k) * stride;
        const uint32x4_t p0 = energy_pair8_1row(r);
        const uint32x4_t p1 = energy_pair8_1row(r + 8);
        const uint32x4_t p2 = energy_pair8_1row(r + 16);
        const uint32x4_t p3 = energy_pair8_1row(r + 24);
        v_col[0] = (i == 0 && k == 0) ? p0 : vaddq_u32(v_col[0], p0);
        v_col[1] = (i == 0 && k == 0) ? p1 : vaddq_u32(v_col[1], p1);
        v_col[2] = (i == 0 && k == 0) ? p2 : vaddq_u32(v_col[2], p2);
        v_col[3] = (i == 0 && k == 0) ? p3 : vaddq_u32(v_col[3], p3);
        sum4[k] = vaddq_u32(vaddq_u32(p0, p1), vaddq_u32(p2, p3));
      }
      vst1q_u32(row_sum + i, horizontal_add_4d_u32x4(sum4));
      i += 4;
    } while (i != esq_h);
  } else {
    do {
      uint32x4_t sum4[4];
      for (int k = 0; k < 4; ++k) {
        const int16_t *r0 = diff + 2 * (i + k) * stride;
        const int16_t *r1 = r0 + stride;
        const uint32x4_t p0 = energy_pair8_2row(r0, r1);
        const uint32x4_t p1 = energy_pair8_2row(r0 + 8, r1 + 8);
        const uint32x4_t p2 = energy_pair8_2row(r0 + 16, r1 + 16);
        const uint32x4_t p3 = energy_pair8_2row(r0 + 24, r1 + 24);
        v_col[0] = (i == 0 && k == 0) ? p0 : vaddq_u32(v_col[0], p0);
        v_col[1] = (i == 0 && k == 0) ? p1 : vaddq_u32(v_col[1], p1);
        v_col[2] = (i == 0 && k == 0) ? p2 : vaddq_u32(v_col[2], p2);
        v_col[3] = (i == 0 && k == 0) ? p3 : vaddq_u32(v_col[3], p3);
        sum4[k] = vaddq_u32(vaddq_u32(p0, p1), vaddq_u32(p2, p3));
      }
      vst1q_u32(row_sum + i, horizontal_add_4d_u32x4(sum4));
      i += 4;
    } while (i != esq_h);
  }
  vst1q_u32(col_sum, v_col[0]);
  vst1q_u32(col_sum + 4, v_col[1]);
  vst1q_u32(col_sum + 8, v_col[2]);
  vst1q_u32(col_sum + 12, v_col[3]);
  return horizontal_long_add_u32x4_x4(v_col);
}

void av1_get_energy_distribution_finer_neon(const int16_t *diff, int stride,
                                            int bw, int bh, float *hordist,
                                            float *verdist) {
  const int w_shift = bw <= 8 ? 0 : 1;
  const int h_shift = bh <= 8 ? 0 : 1;
  const int esq_w = bw >> w_shift;
  const int esq_h = bh >> h_shift;
  assert(esq_w >= 4 && esq_w <= 16 && (esq_w & 3) == 0);
  assert(esq_h >= 4 && esq_h <= 16 && (esq_h & 3) == 0);
  uint32_t col_sum[16];
  uint32_t row_sum[16];
  uint64_t total;

  if (bw == 4) {
    total = energy_dist_w4(diff, stride, esq_h, h_shift, col_sum, row_sum);
  } else if (bw == 8) {
    total = energy_dist_w8(diff, stride, esq_h, h_shift, col_sum, row_sum);
  } else if (bw == 16) {
    total = energy_dist_w16(diff, stride, esq_h, h_shift, col_sum, row_sum);
  } else {
    assert(bw == 32);
    total = energy_dist_w32(diff, stride, esq_h, h_shift, col_sum, row_sum);
  }

  if (total == 0) {
    const float hor_val = 1.0f / esq_w;
    for (int j = 0; j < esq_w - 1; ++j) hordist[j] = hor_val;
    const float ver_val = 1.0f / esq_h;
    for (int i = 0; i < esq_h - 1; ++i) verdist[i] = ver_val;
    return;
  }

  const float e_recip = 1.0f / (float)total;
  for (int j = 0; j < esq_w - 1; ++j) hordist[j] = (float)col_sum[j] * e_recip;
  for (int i = 0; i < esq_h - 1; ++i) verdist[i] = (float)row_sum[i] * e_recip;
}
