/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved.
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
#include <stdlib.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"

#include "aom_dsp/arm/sum_neon.h"
#include "av1/encoder/intra_mode_search_utils.h"

#if AOM_ARCH_AARCH64
static AOM_FORCE_INLINE int32x4_t count_gt_32bins(int32x4_t v_r, int32x4_t t0,
                                                  int32x4_t t1, int32x4_t t2,
                                                  int32x4_t t3, int32x4_t t4,
                                                  int32x4_t t5, int32x4_t t6,
                                                  int32x4_t t7) {
  const uint32x4_t s01 = vaddq_u32(vcgtq_s32(v_r, t0), vcgtq_s32(v_r, t1));
  const uint32x4_t s23 = vaddq_u32(vcgtq_s32(v_r, t2), vcgtq_s32(v_r, t3));
  const uint32x4_t s45 = vaddq_u32(vcgtq_s32(v_r, t4), vcgtq_s32(v_r, t5));
  const uint32x4_t s67 = vaddq_u32(vcgtq_s32(v_r, t6), vcgtq_s32(v_r, t7));
  return vreinterpretq_s32_u32(
      vaddq_u32(vaddq_u32(s01, s23), vaddq_u32(s45, s67)));
}

static AOM_FORCE_INLINE int32x4_t
compute_4_bins_neon(int16x4_t v_safe_dx, int16x4_t v_dy, int32x4_t t0,
                    int32x4_t t1, int32x4_t t2, int32x4_t t3, int32x4_t t4,
                    int32x4_t t5, int32x4_t t6, int32x4_t t7) {
  const int32x4_t safe_dx = vmovl_s16(v_safe_dx);
  const int32x4_t dy32 = vmovl_s16(v_dy);
  const int32x4_t scaled_dy = vshlq_n_s32(dy32, HOG_FIX_PREC_BITS);

  // For <= 12-bit inputs ([0, 4095]), |dx|, |dy| <= 16380 (fitting in int16_t)
  // and |scaled_dy| < 2^31. Since non-integer quotients stay >= 1/16380 away
  // from an integer, float64 division + vcvtq_s64_f64 matches C truncation.
  const float64x2_t num0 = vcvtq_f64_s64(vmovl_s32(vget_low_s32(scaled_dy)));
  const float64x2_t den0 = vcvtq_f64_s64(vmovl_s32(vget_low_s32(safe_dx)));
  const float64x2_t num1 = vcvtq_f64_s64(vmovl_high_s32(scaled_dy));
  const float64x2_t den1 = vcvtq_f64_s64(vmovl_high_s32(safe_dx));

  const int64x2_t q0 = vcvtq_s64_f64(vdivq_f64(num0, den0));
  const int64x2_t q1 = vcvtq_s64_f64(vdivq_f64(num1, den1));
  const int32x4_t ratio = vcombine_s32(vmovn_s64(q0), vmovn_s64(q1));

  const int32x4_t b0 = count_gt_32bins(vdupq_laneq_s32(ratio, 0), t0, t1, t2,
                                       t3, t4, t5, t6, t7);
  const int32x4_t b1 = count_gt_32bins(vdupq_laneq_s32(ratio, 1), t0, t1, t2,
                                       t3, t4, t5, t6, t7);
  const int32x4_t b2 = count_gt_32bins(vdupq_laneq_s32(ratio, 2), t0, t1, t2,
                                       t3, t4, t5, t6, t7);
  const int32x4_t b3 = count_gt_32bins(vdupq_laneq_s32(ratio, 3), t0, t1, t2,
                                       t3, t4, t5, t6, t7);
  const int32x4_t p01 = vpaddq_s32(b0, b1);
  const int32x4_t p23 = vpaddq_s32(b2, b3);
  return vnegq_s32(vpaddq_s32(p01, p23));
}

static AOM_FORCE_INLINE int8x8_t compute_8_bins_neon(
    int16x8_t v_dx, int16x8_t v_dy, int32x4_t t0, int32x4_t t1, int32x4_t t2,
    int32x4_t t3, int32x4_t t4, int32x4_t t5, int32x4_t t6, int32x4_t t7) {
  const uint16x8_t dx_zero_16 = vceqzq_s16(v_dx);
  const int16x8_t safe_dx = vsubq_s16(v_dx, vreinterpretq_s16_u16(dx_zero_16));
  const int32x4_t bins_lo =
      compute_4_bins_neon(vget_low_s16(safe_dx), vget_low_s16(v_dy), t0, t1, t2,
                          t3, t4, t5, t6, t7);
  const int32x4_t bins_hi =
      compute_4_bins_neon(vget_high_s16(safe_dx), vget_high_s16(v_dy), t0, t1,
                          t2, t3, t4, t5, t6, t7);
  const int16x8_t bins_16 =
      vcombine_s16(vmovn_s32(bins_lo), vmovn_s32(bins_hi));
  const int8x8_t bins_8 = vmovn_s16(bins_16);
  return vbsl_s8(vmovn_u16(dx_zero_16), vdup_n_s8(HOG_BINS), bins_8);
}

static AOM_FORCE_INLINE void sobel_8_lowbd(const uint8_t *row_above,
                                           const uint8_t *row_cur,
                                           const uint8_t *row_below, int c,
                                           int16x8_t *out_dx, int16x8_t *out_dy,
                                           uint16x8_t *out_sum) {
  const uint8x8_t a_l = vld1_u8(row_above + c - 1);
  const uint8x8_t a_m = vld1_u8(row_above + c);
  const uint8x8_t a_r = vld1_u8(row_above + c + 1);
  const uint8x8_t c_l = vld1_u8(row_cur + c - 1);
  const uint8x8_t c_r = vld1_u8(row_cur + c + 1);
  const uint8x8_t b_l = vld1_u8(row_below + c - 1);
  const uint8x8_t b_m = vld1_u8(row_below + c);
  const uint8x8_t b_r = vld1_u8(row_below + c + 1);

  const int16x8_t da = vreinterpretq_s16_u16(vsubl_u8(a_r, a_l));
  const int16x8_t dc = vreinterpretq_s16_u16(vsubl_u8(c_r, c_l));
  const int16x8_t db = vreinterpretq_s16_u16(vsubl_u8(b_r, b_l));
  const int16x8_t v_dx = vaddq_s16(vaddq_s16(da, db), vshlq_n_s16(dc, 1));

  const int16x8_t vl = vreinterpretq_s16_u16(vsubl_u8(b_l, a_l));
  const int16x8_t vm = vreinterpretq_s16_u16(vsubl_u8(b_m, a_m));
  const int16x8_t vr = vreinterpretq_s16_u16(vsubl_u8(b_r, a_r));
  const int16x8_t v_dy = vaddq_s16(vaddq_s16(vl, vr), vshlq_n_s16(vm, 1));

  const uint16x8_t abs_dx = vreinterpretq_u16_s16(vabsq_s16(v_dx));
  const uint16x8_t abs_dy = vreinterpretq_u16_s16(vabsq_s16(v_dy));
  *out_dx = v_dx;
  *out_dy = v_dy;
  *out_sum = vaddq_u16(abs_dx, abs_dy);
}

#if CONFIG_AV1_HIGHBITDEPTH
static AOM_FORCE_INLINE void sobel_8_highbd(const uint16_t *row_above,
                                            const uint16_t *row_cur,
                                            const uint16_t *row_below, int c,
                                            int16x8_t *out_dx,
                                            int16x8_t *out_dy,
                                            uint16x8_t *out_sum) {
  const int16x8_t a_l = vreinterpretq_s16_u16(vld1q_u16(row_above + c - 1));
  const int16x8_t a_m = vreinterpretq_s16_u16(vld1q_u16(row_above + c));
  const int16x8_t a_r = vreinterpretq_s16_u16(vld1q_u16(row_above + c + 1));
  const int16x8_t c_l = vreinterpretq_s16_u16(vld1q_u16(row_cur + c - 1));
  const int16x8_t c_r = vreinterpretq_s16_u16(vld1q_u16(row_cur + c + 1));
  const int16x8_t b_l = vreinterpretq_s16_u16(vld1q_u16(row_below + c - 1));
  const int16x8_t b_m = vreinterpretq_s16_u16(vld1q_u16(row_below + c));
  const int16x8_t b_r = vreinterpretq_s16_u16(vld1q_u16(row_below + c + 1));

  const int16x8_t da = vsubq_s16(a_r, a_l);
  const int16x8_t dc = vsubq_s16(c_r, c_l);
  const int16x8_t db = vsubq_s16(b_r, b_l);
  const int16x8_t v_dx = vaddq_s16(vaddq_s16(da, db), vshlq_n_s16(dc, 1));

  const int16x8_t vl = vsubq_s16(b_l, a_l);
  const int16x8_t vm = vsubq_s16(b_m, a_m);
  const int16x8_t vr = vsubq_s16(b_r, a_r);
  const int16x8_t v_dy = vaddq_s16(vaddq_s16(vl, vr), vshlq_n_s16(vm, 1));

  const uint16x8_t abs_dx = vreinterpretq_u16_s16(vabsq_s16(v_dx));
  const uint16x8_t abs_dy = vreinterpretq_u16_s16(vabsq_s16(v_dy));
  *out_dx = v_dx;
  *out_dy = v_dy;
  *out_sum = vaddq_u16(abs_dx, abs_dy);
}
#endif  // CONFIG_AV1_HIGHBITDEPTH

static AOM_FORCE_INLINE void store_grad_info_8(PixelLevelGradientInfo *dst,
                                               uint16x8_t v_sum,
                                               int8x8_t bins_8) {
  uint16_t sum_buf[8];
  int8_t bin_buf[8];
  vst1q_u16(sum_buf, v_sum);
  vst1_s8(bin_buf, bins_8);
  for (int k = 0; k < 8; ++k) {
    dst[k].abs_dx_abs_dy_sum = sum_buf[k];
    dst[k].hist_bin_idx = bin_buf[k];
  }
}

static AOM_FORCE_INLINE void store_flat_grad_info_8(
    PixelLevelGradientInfo *dst) {
  for (int k = 0; k < 8; ++k) {
    dst[k].abs_dx_abs_dy_sum = 0;
    dst[k].hist_bin_idx = HOG_BINS;
  }
}

static AOM_FORCE_INLINE void accum_hog_8(uint16x8_t v_sum, int8x8_t bins_8,
                                         float *total, float *hist) {
  uint16_t sum_buf[8];
  int8_t bin_buf[8];
  vst1q_u16(sum_buf, v_sum);
  vst1_s8(bin_buf, bins_8);
  for (int k = 0; k < 8; ++k) {
    const int s = sum_buf[k];
    if (s == 0) continue;
    *total += s;
    const int8_t b = bin_buf[k];
    if (b == HOG_BINS) {
      hist[0] += s / 2;
      hist[HOG_BINS - 1] += s / 2;
    } else {
      assert(b >= 0 && b < HOG_BINS);
      hist[b] += s;
    }
  }
}

void av1_lowbd_generate_hog_neon(const uint8_t *src, int stride, int rows,
                                 int cols, float *hist) {
  float total = 0.1f;
  if (cols >= 3 && rows >= 3) {
    const int32_t *const thresholds = av1_get_hog_thresholds();
    const int32x4_t t0 = vld1q_s32(&thresholds[0]);
    const int32x4_t t1 = vld1q_s32(&thresholds[4]);
    const int32x4_t t2 = vld1q_s32(&thresholds[8]);
    const int32x4_t t3 = vld1q_s32(&thresholds[12]);
    const int32x4_t t4 = vld1q_s32(&thresholds[16]);
    const int32x4_t t5 = vld1q_s32(&thresholds[20]);
    const int32x4_t t6 = vld1q_s32(&thresholds[24]);
    const int32x4_t t7 = vld1q_s32(&thresholds[28]);

    src += stride;
    for (int r = 1; r < rows - 1; ++r) {
      const uint8_t *row_above = src - stride;
      const uint8_t *row_cur = src;
      const uint8_t *row_below = src + stride;

      int c = 1;
      for (; c + 8 <= cols - 1; c += 8) {
        int16x8_t v_dx, v_dy;
        uint16x8_t v_sum;
        sobel_8_lowbd(row_above, row_cur, row_below, c, &v_dx, &v_dy, &v_sum);
        if (vmaxvq_u16(v_sum) == 0) continue;

        const int8x8_t bins_8 =
            compute_8_bins_neon(v_dx, v_dy, t0, t1, t2, t3, t4, t5, t6, t7);
        accum_hog_8(v_sum, bins_8, &total, hist);
      }

      if (c < cols - 1) {
        int v0 = row_above[c - 1] + 2 * row_cur[c - 1] + row_below[c - 1];
        int v1 = row_above[c] + 2 * row_cur[c] + row_below[c];
        int d0 = row_below[c - 1] - row_above[c - 1];
        int d1 = row_below[c] - row_above[c];
        for (; c < cols - 1; ++c) {
          const int above_r = row_above[c + 1];
          const int cur_r = row_cur[c + 1];
          const int below_r = row_below[c + 1];
          const int v2 = above_r + 2 * cur_r + below_r;
          const int d2 = below_r - above_r;
          const int dx = v2 - v0;
          const int dy = d0 + 2 * d1 + d2;
          v0 = v1;
          v1 = v2;
          d0 = d1;
          d1 = d2;
          if ((dx | dy) == 0) continue;
          const int temp = abs(dx) + abs(dy);
          total += temp;
          if (dx == 0) {
            hist[0] += temp / 2;
            hist[HOG_BINS - 1] += temp / 2;
          } else {
            hist[get_hist_bin_idx(dx, dy)] += temp;
          }
        }
      }
      src += stride;
    }
  }
  normalize_hog(total, hist);
}

void av1_lowbd_compute_gradient_info_sb_neon(MACROBLOCK *const x,
                                             BLOCK_SIZE sb_size,
                                             PLANE_TYPE plane) {
  PixelLevelGradientInfo *const grad_info_sb =
      x->pixel_gradient_info + plane * MAX_SB_SQUARE;
  const uint8_t *src = x->plane[plane].src.buf;
  const int stride = x->plane[plane].src.stride;
  const int ss_x = x->e_mbd.plane[plane].subsampling_x;
  const int ss_y = x->e_mbd.plane[plane].subsampling_y;
  const int sb_height = block_size_high[sb_size] >> ss_y;
  const int sb_width = block_size_wide[sb_size] >> ss_x;
  if (sb_width < 3 || sb_height < 3) return;

  const int32_t *const thresholds = av1_get_hog_thresholds();
  const int32x4_t t0 = vld1q_s32(&thresholds[0]);
  const int32x4_t t1 = vld1q_s32(&thresholds[4]);
  const int32x4_t t2 = vld1q_s32(&thresholds[8]);
  const int32x4_t t3 = vld1q_s32(&thresholds[12]);
  const int32x4_t t4 = vld1q_s32(&thresholds[16]);
  const int32x4_t t5 = vld1q_s32(&thresholds[20]);
  const int32x4_t t6 = vld1q_s32(&thresholds[24]);
  const int32x4_t t7 = vld1q_s32(&thresholds[28]);

  src += stride;
  for (int r = 1; r < sb_height - 1; ++r) {
    const uint8_t *row_above = src - stride;
    const uint8_t *row_cur = src;
    const uint8_t *row_below = src + stride;
    PixelLevelGradientInfo *const row_info = grad_info_sb + r * sb_width;

    int c = 1;
    for (; c + 8 <= sb_width - 1; c += 8) {
      int16x8_t v_dx, v_dy;
      uint16x8_t v_sum;
      sobel_8_lowbd(row_above, row_cur, row_below, c, &v_dx, &v_dy, &v_sum);
      if (vmaxvq_u16(v_sum) == 0) {
        store_flat_grad_info_8(row_info + c);
        continue;
      }
      const int8x8_t bins_8 =
          compute_8_bins_neon(v_dx, v_dy, t0, t1, t2, t3, t4, t5, t6, t7);
      store_grad_info_8(row_info + c, v_sum, bins_8);
    }

    if (c < sb_width - 1) {
      int v0 = row_above[c - 1] + 2 * row_cur[c - 1] + row_below[c - 1];
      int v1 = row_above[c] + 2 * row_cur[c] + row_below[c];
      int d0 = row_below[c - 1] - row_above[c - 1];
      int d1 = row_below[c] - row_above[c];
      for (; c < sb_width - 1; ++c) {
        const int above_r = row_above[c + 1];
        const int cur_r = row_cur[c + 1];
        const int below_r = row_below[c + 1];
        const int v2 = above_r + 2 * cur_r + below_r;
        const int d2 = below_r - above_r;
        const int dx = v2 - v0;
        const int dy = d0 + 2 * d1 + d2;
        v0 = v1;
        v1 = v2;
        d0 = d1;
        d1 = d2;
        row_info[c].abs_dx_abs_dy_sum = (uint16_t)(abs(dx) + abs(dy));
        row_info[c].hist_bin_idx =
            (dx == 0) ? HOG_BINS : (int8_t)get_hist_bin_idx(dx, dy);
      }
    }
    src += stride;
  }
}

#if CONFIG_AV1_HIGHBITDEPTH
void av1_highbd_generate_hog_neon(const uint8_t *src8, int stride, int rows,
                                  int cols, float *hist) {
  float total = 0.1f;
  if (cols >= 3 && rows >= 3) {
    const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
    const int32_t *const thresholds = av1_get_hog_thresholds();
    const int32x4_t t0 = vld1q_s32(&thresholds[0]);
    const int32x4_t t1 = vld1q_s32(&thresholds[4]);
    const int32x4_t t2 = vld1q_s32(&thresholds[8]);
    const int32x4_t t3 = vld1q_s32(&thresholds[12]);
    const int32x4_t t4 = vld1q_s32(&thresholds[16]);
    const int32x4_t t5 = vld1q_s32(&thresholds[20]);
    const int32x4_t t6 = vld1q_s32(&thresholds[24]);
    const int32x4_t t7 = vld1q_s32(&thresholds[28]);

    src += stride;
    for (int r = 1; r < rows - 1; ++r) {
      const uint16_t *row_above = src - stride;
      const uint16_t *row_cur = src;
      const uint16_t *row_below = src + stride;

      int c = 1;
      for (; c + 8 <= cols - 1; c += 8) {
        int16x8_t v_dx, v_dy;
        uint16x8_t v_sum;
        sobel_8_highbd(row_above, row_cur, row_below, c, &v_dx, &v_dy, &v_sum);
        if (vmaxvq_u16(v_sum) == 0) continue;

        const int8x8_t bins_8 =
            compute_8_bins_neon(v_dx, v_dy, t0, t1, t2, t3, t4, t5, t6, t7);
        accum_hog_8(v_sum, bins_8, &total, hist);
      }

      if (c < cols - 1) {
        int v0 = row_above[c - 1] + 2 * row_cur[c - 1] + row_below[c - 1];
        int v1 = row_above[c] + 2 * row_cur[c] + row_below[c];
        int d0 = row_below[c - 1] - row_above[c - 1];
        int d1 = row_below[c] - row_above[c];
        for (; c < cols - 1; ++c) {
          const int above_r = row_above[c + 1];
          const int cur_r = row_cur[c + 1];
          const int below_r = row_below[c + 1];
          const int v2 = above_r + 2 * cur_r + below_r;
          const int d2 = below_r - above_r;
          const int dx = v2 - v0;
          const int dy = d0 + 2 * d1 + d2;
          v0 = v1;
          v1 = v2;
          d0 = d1;
          d1 = d2;
          if ((dx | dy) == 0) continue;
          const int temp = abs(dx) + abs(dy);
          total += temp;
          if (dx == 0) {
            hist[0] += temp / 2;
            hist[HOG_BINS - 1] += temp / 2;
          } else {
            hist[get_hist_bin_idx(dx, dy)] += temp;
          }
        }
      }
      src += stride;
    }
  }
  normalize_hog(total, hist);
}

void av1_highbd_compute_gradient_info_sb_neon(MACROBLOCK *const x,
                                              BLOCK_SIZE sb_size,
                                              PLANE_TYPE plane) {
  PixelLevelGradientInfo *const grad_info_sb =
      x->pixel_gradient_info + plane * MAX_SB_SQUARE;
  const uint16_t *src = CONVERT_TO_SHORTPTR(x->plane[plane].src.buf);
  const int stride = x->plane[plane].src.stride;
  const int ss_x = x->e_mbd.plane[plane].subsampling_x;
  const int ss_y = x->e_mbd.plane[plane].subsampling_y;
  const int sb_height = block_size_high[sb_size] >> ss_y;
  const int sb_width = block_size_wide[sb_size] >> ss_x;
  if (sb_width < 3 || sb_height < 3) return;

  const int32_t *const thresholds = av1_get_hog_thresholds();
  const int32x4_t t0 = vld1q_s32(&thresholds[0]);
  const int32x4_t t1 = vld1q_s32(&thresholds[4]);
  const int32x4_t t2 = vld1q_s32(&thresholds[8]);
  const int32x4_t t3 = vld1q_s32(&thresholds[12]);
  const int32x4_t t4 = vld1q_s32(&thresholds[16]);
  const int32x4_t t5 = vld1q_s32(&thresholds[20]);
  const int32x4_t t6 = vld1q_s32(&thresholds[24]);
  const int32x4_t t7 = vld1q_s32(&thresholds[28]);

  src += stride;
  for (int r = 1; r < sb_height - 1; ++r) {
    const uint16_t *row_above = src - stride;
    const uint16_t *row_cur = src;
    const uint16_t *row_below = src + stride;
    PixelLevelGradientInfo *const row_info = grad_info_sb + r * sb_width;

    int c = 1;
    for (; c + 8 <= sb_width - 1; c += 8) {
      int16x8_t v_dx, v_dy;
      uint16x8_t v_sum;
      sobel_8_highbd(row_above, row_cur, row_below, c, &v_dx, &v_dy, &v_sum);
      if (vmaxvq_u16(v_sum) == 0) {
        store_flat_grad_info_8(row_info + c);
        continue;
      }
      const int8x8_t bins_8 =
          compute_8_bins_neon(v_dx, v_dy, t0, t1, t2, t3, t4, t5, t6, t7);
      store_grad_info_8(row_info + c, v_sum, bins_8);
    }

    if (c < sb_width - 1) {
      int v0 = row_above[c - 1] + 2 * row_cur[c - 1] + row_below[c - 1];
      int v1 = row_above[c] + 2 * row_cur[c] + row_below[c];
      int d0 = row_below[c - 1] - row_above[c - 1];
      int d1 = row_below[c] - row_above[c];
      for (; c < sb_width - 1; ++c) {
        const int above_r = row_above[c + 1];
        const int cur_r = row_cur[c + 1];
        const int below_r = row_below[c + 1];
        const int v2 = above_r + 2 * cur_r + below_r;
        const int d2 = below_r - above_r;
        const int dx = v2 - v0;
        const int dy = d0 + 2 * d1 + d2;
        v0 = v1;
        v1 = v2;
        d0 = d1;
        d1 = d2;
        row_info[c].abs_dx_abs_dy_sum = (uint16_t)(abs(dx) + abs(dy));
        row_info[c].hist_bin_idx =
            (dx == 0) ? HOG_BINS : (int8_t)get_hist_bin_idx(dx, dy);
      }
    }
    src += stride;
  }
}
#endif  // CONFIG_AV1_HIGHBITDEPTH
#else   // !AOM_ARCH_AARCH64
static inline int get_hist_bin_idx_neon32(int dx, int dy) {
  if (dy == 0) return 16;
  const int32_t ratio = (dy * (1 << HOG_FIX_PREC_BITS)) / dx;
  const int32_t *const thresholds = av1_get_hog_thresholds();
  const int lo_idx = (ratio > thresholds[15]) << 4;
  const int32x4_t v_ratio = vdupq_n_s32(ratio);
  const uint32x4_t c0 = vcgtq_s32(v_ratio, vld1q_s32(&thresholds[lo_idx]));
  const uint32x4_t c1 = vcgtq_s32(v_ratio, vld1q_s32(&thresholds[lo_idx + 4]));
  const uint32x4_t c2 = vcgtq_s32(v_ratio, vld1q_s32(&thresholds[lo_idx + 8]));
  const uint32x4_t c3 = vcgtq_s32(v_ratio, vld1q_s32(&thresholds[lo_idx + 12]));
  const int32x4_t sum =
      vreinterpretq_s32_u32(vaddq_u32(vaddq_u32(c0, c1), vaddq_u32(c2, c3)));
  return lo_idx - horizontal_add_s32x4(sum);
}

void av1_lowbd_generate_hog_neon(const uint8_t *src, int stride, int rows,
                                 int cols, float *hist) {
  lowbd_generate_hog(src, stride, rows, cols, hist, get_hist_bin_idx_neon32);
}

void av1_lowbd_compute_gradient_info_sb_neon(MACROBLOCK *const x,
                                             BLOCK_SIZE sb_size,
                                             PLANE_TYPE plane) {
  lowbd_compute_gradient_info_sb(x, sb_size, plane, get_hist_bin_idx_neon32);
}

#if CONFIG_AV1_HIGHBITDEPTH
void av1_highbd_generate_hog_neon(const uint8_t *src8, int stride, int rows,
                                  int cols, float *hist) {
  highbd_generate_hog(src8, stride, rows, cols, hist, get_hist_bin_idx_neon32);
}

void av1_highbd_compute_gradient_info_sb_neon(MACROBLOCK *const x,
                                              BLOCK_SIZE sb_size,
                                              PLANE_TYPE plane) {
  highbd_compute_gradient_info_sb(x, sb_size, plane, get_hist_bin_idx_neon32);
}
#endif  // CONFIG_AV1_HIGHBITDEPTH
#endif  // AOM_ARCH_AARCH64
