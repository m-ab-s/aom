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
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"

#include "aom_dsp/arm/mem_neon.h"
#include "aom_dsp/arm/sum_neon.h"
#include "aom_dsp/arm/transpose_neon.h"
#include "aom_ports/mem.h"
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

int av1_hadamard_satd_4x4_sub_neon(const uint8_t *src, ptrdiff_t src_stride,
                                   const uint8_t *pred, ptrdiff_t pred_stride) {
  const uint8x8_t s02 = load_unaligned_u8_4x2(src, 2 * src_stride);
  const uint8x8_t s13 = load_unaligned_u8_4x2(src + src_stride, 2 * src_stride);
  const uint8x8_t d02 = load_unaligned_u8_4x2(pred, 2 * pred_stride);
  const uint8x8_t d13 =
      load_unaligned_u8_4x2(pred + pred_stride, 2 * pred_stride);
  const int16x8_t diff02 = vreinterpretq_s16_u16(vsubl_u8(s02, d02));
  const int16x8_t diff13 = vreinterpretq_s16_u16(vsubl_u8(s13, d13));

  // Pass 1 (columns): residuals in [-255, 255]; t02/t13 in [-255, 255];
  // y01/y23 in [-510, 510] with no 16-bit overflow.
  const int16x8_t t02 = vhaddq_s16(diff02, diff13);
  const int16x8_t t13 = vhsubq_s16(diff02, diff13);
  const int16x8_t p01 = vreinterpretq_s16_s64(
      aom_vtrn1q_s64(vreinterpretq_s64_s16(t02), vreinterpretq_s64_s16(t13)));
  const int16x8_t p23 = vreinterpretq_s16_s64(
      aom_vtrn2q_s64(vreinterpretq_s64_s16(t02), vreinterpretq_s64_s16(t13)));
  const int16x8_t y01 = vaddq_s16(p01, p23);
  const int16x8_t y23 = vsubq_s16(p01, p23);

  // Pass 2 (rows): deinterleave even/odd columns and fold the final butterfly
  // via |u + v| + |u - v| = 2 * max(|u|, |v|) on [-510, 510] pairs.
  const int16x8x2_t y_uzp = vuzpq_s16(y01, y23);
  const int16x8_t b02 = vhaddq_s16(y_uzp.val[0], y_uzp.val[1]);
  const int16x8_t b13 = vhsubq_s16(y_uzp.val[0], y_uzp.val[1]);
  const int16x8_t abs02 = vabsq_s16(b02);
  const int16x8_t abs13 = vabsq_s16(b13);
#if AOM_ARCH_AARCH64
  const int16x8_t m = vpmaxq_s16(abs02, abs13);
  return (int)(horizontal_add_u16x8(vreinterpretq_u16_s16(m)) << 1);
#else
  const int16x4_t m0 = vpmax_s16(vget_low_s16(abs02), vget_high_s16(abs02));
  const int16x4_t m1 = vpmax_s16(vget_low_s16(abs13), vget_high_s16(abs13));
  const uint16x4_t m =
      vadd_u16(vreinterpret_u16_s16(m0), vreinterpret_u16_s16(m1));
  return (int)(horizontal_add_u16x4(m) << 1);
#endif  // AOM_ARCH_AARCH64
}

static inline void load_u8_sub_s16_8x8(
    const uint8_t *src, ptrdiff_t src_stride, const uint8_t *pred,
    ptrdiff_t pred_stride, int16x8_t *a0, int16x8_t *a1, int16x8_t *a2,
    int16x8_t *a3, int16x8_t *a4, int16x8_t *a5, int16x8_t *a6, int16x8_t *a7) {
  uint8x8_t s0, s1, s2, s3, s4, s5, s6, s7;
  uint8x8_t d0, d1, d2, d3, d4, d5, d6, d7;
  load_u8_8x8(src, src_stride, &s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7);
  load_u8_8x8(pred, pred_stride, &d0, &d1, &d2, &d3, &d4, &d5, &d6, &d7);
  *a0 = vreinterpretq_s16_u16(vsubl_u8(s0, d0));
  *a1 = vreinterpretq_s16_u16(vsubl_u8(s1, d1));
  *a2 = vreinterpretq_s16_u16(vsubl_u8(s2, d2));
  *a3 = vreinterpretq_s16_u16(vsubl_u8(s3, d3));
  *a4 = vreinterpretq_s16_u16(vsubl_u8(s4, d4));
  *a5 = vreinterpretq_s16_u16(vsubl_u8(s5, d5));
  *a6 = vreinterpretq_s16_u16(vsubl_u8(s6, d6));
  *a7 = vreinterpretq_s16_u16(vsubl_u8(s7, d7));
}

static inline void hadamard_8x8_stage1_2_neon(
    int16x8_t a0, int16x8_t a1, int16x8_t a2, int16x8_t a3, int16x8_t a4,
    int16x8_t a5, int16x8_t a6, int16x8_t a7, int16x8_t *c0, int16x8_t *c1,
    int16x8_t *c2, int16x8_t *c3, int16x8_t *c4, int16x8_t *c5, int16x8_t *c6,
    int16x8_t *c7) {
  const int16x8_t b0 = vaddq_s16(a0, a1);
  const int16x8_t b1 = vsubq_s16(a0, a1);
  const int16x8_t b2 = vaddq_s16(a2, a3);
  const int16x8_t b3 = vsubq_s16(a2, a3);
  const int16x8_t b4 = vaddq_s16(a4, a5);
  const int16x8_t b5 = vsubq_s16(a4, a5);
  const int16x8_t b6 = vaddq_s16(a6, a7);
  const int16x8_t b7 = vsubq_s16(a6, a7);

  *c0 = vaddq_s16(b0, b2);
  *c1 = vaddq_s16(b1, b3);
  *c2 = vsubq_s16(b0, b2);
  *c3 = vsubq_s16(b1, b3);
  *c4 = vaddq_s16(b4, b6);
  *c5 = vaddq_s16(b5, b7);
  *c6 = vsubq_s16(b4, b6);
  *c7 = vsubq_s16(b5, b7);
}

static inline void hadamard_8x8_one_pass_neon(int16x8_t *a0, int16x8_t *a1,
                                              int16x8_t *a2, int16x8_t *a3,
                                              int16x8_t *a4, int16x8_t *a5,
                                              int16x8_t *a6, int16x8_t *a7) {
  int16x8_t c0, c1, c2, c3, c4, c5, c6, c7;
  hadamard_8x8_stage1_2_neon(*a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, &c0, &c1,
                             &c2, &c3, &c4, &c5, &c6, &c7);
  *a0 = vaddq_s16(c0, c4);
  *a1 = vsubq_s16(c2, c6);
  *a2 = vsubq_s16(c0, c4);
  *a3 = vaddq_s16(c2, c6);
  *a4 = vaddq_s16(c3, c7);
  *a5 = vsubq_s16(c3, c7);
  *a6 = vsubq_s16(c1, c5);
  *a7 = vaddq_s16(c1, c5);
}

int av1_hadamard_satd_8x8_sub_neon(const uint8_t *src, ptrdiff_t src_stride,
                                   const uint8_t *pred, ptrdiff_t pred_stride) {
  int16x8_t a0, a1, a2, a3, a4, a5, a6, a7;
  load_u8_sub_s16_8x8(src, src_stride, pred, pred_stride, &a0, &a1, &a2, &a3,
                      &a4, &a5, &a6, &a7);

  hadamard_8x8_one_pass_neon(&a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7);
  transpose_elems_inplace_s16_8x8(&a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7);

  // Pass 2 reaches c0..c7 in [-8160, 8160] (sum of 32 terms in [-255, 255]).
  // Folding the final add/sub via |u + v| + |u - v| = 2 * max(|u|, |v|) gives
  // m0..m3 <= 8160 and sum <= 32640 <= UINT16_MAX without signed 16-bit
  // overflow.
  int16x8_t c0, c1, c2, c3, c4, c5, c6, c7;
  hadamard_8x8_stage1_2_neon(a0, a1, a2, a3, a4, a5, a6, a7, &c0, &c1, &c2, &c3,
                             &c4, &c5, &c6, &c7);

  const uint16x8_t m0 =
      vreinterpretq_u16_s16(vmaxq_s16(vabsq_s16(c0), vabsq_s16(c4)));
  const uint16x8_t m1 =
      vreinterpretq_u16_s16(vmaxq_s16(vabsq_s16(c2), vabsq_s16(c6)));
  const uint16x8_t m2 =
      vreinterpretq_u16_s16(vmaxq_s16(vabsq_s16(c3), vabsq_s16(c7)));
  const uint16x8_t m3 =
      vreinterpretq_u16_s16(vmaxq_s16(vabsq_s16(c1), vabsq_s16(c5)));
  const uint16x8_t sum = vaddq_u16(vaddq_u16(m0, m1), vaddq_u16(m2, m3));
  return (int)(horizontal_add_u16x8(sum) << 1);
}

static inline void hadamard_lp_8x8_sub_neon(const uint8_t *src,
                                            ptrdiff_t src_stride,
                                            const uint8_t *pred,
                                            ptrdiff_t pred_stride,
                                            int16_t *coeff) {
  int16x8_t a0, a1, a2, a3, a4, a5, a6, a7;
  load_u8_sub_s16_8x8(src, src_stride, pred, pred_stride, &a0, &a1, &a2, &a3,
                      &a4, &a5, &a6, &a7);

  hadamard_8x8_one_pass_neon(&a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7);
  transpose_elems_inplace_s16_8x8(&a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7);
  hadamard_8x8_one_pass_neon(&a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7);

  vst1q_s16(coeff + 0, a0);
  vst1q_s16(coeff + 8, a1);
  vst1q_s16(coeff + 16, a2);
  vst1q_s16(coeff + 24, a3);
  vst1q_s16(coeff + 32, a4);
  vst1q_s16(coeff + 40, a5);
  vst1q_s16(coeff + 48, a6);
  vst1q_s16(coeff + 56, a7);
}

static inline void hadamard_lp_16x16_quadrants_sub_neon(const uint8_t *src,
                                                        ptrdiff_t src_stride,
                                                        const uint8_t *pred,
                                                        ptrdiff_t pred_stride,
                                                        int16_t *lp_coeff) {
  for (int row = 0; row < 2; ++row) {
    const uint8_t *const s_row = src + row * 8 * src_stride;
    const uint8_t *const p_row = pred + row * 8 * pred_stride;
    for (int col = 0; col < 2; ++col) {
      hadamard_lp_8x8_sub_neon(s_row + col * 8, src_stride, p_row + col * 8,
                               pred_stride, lp_coeff + (row * 2 + col) * 64);
    }
  }
}

static inline uint16x8_t hadamard_satd_16x16_pair_max_neon(
    const int16_t *lp_coeff) {
  // 8x8 LP coefficients fit in [-16320, 16320]; vhaddq/vhsubq halves into
  // [-16320, 16320], so |u + v| + |u - v| = 2 * max(|u|, |v|) avoids [-32640,
  // 32640] abs hazards and yields m0 + m1 <= 32640 (2 slices <= 65280 <= u16).
  const int16x8_t a0 = vld1q_s16(lp_coeff + 0);
  const int16x8_t a1 = vld1q_s16(lp_coeff + 64);
  const int16x8_t a2 = vld1q_s16(lp_coeff + 128);
  const int16x8_t a3 = vld1q_s16(lp_coeff + 192);
  const int16x8_t b0 = vhaddq_s16(a0, a1);
  const int16x8_t b1 = vhsubq_s16(a0, a1);
  const int16x8_t b2 = vhaddq_s16(a2, a3);
  const int16x8_t b3 = vhsubq_s16(a2, a3);
  const uint16x8_t m0 =
      vreinterpretq_u16_s16(vmaxq_s16(vabsq_s16(b0), vabsq_s16(b2)));
  const uint16x8_t m1 =
      vreinterpretq_u16_s16(vmaxq_s16(vabsq_s16(b1), vabsq_s16(b3)));
  return vaddq_u16(m0, m1);
}

int av1_hadamard_satd_16x16_sub_neon(const uint8_t *src, ptrdiff_t src_stride,
                                     const uint8_t *pred,
                                     ptrdiff_t pred_stride) {
  DECLARE_ALIGNED(16, int16_t, lp_coeff[256]);
  hadamard_lp_16x16_quadrants_sub_neon(src, src_stride, pred, pred_stride,
                                       lp_coeff);
  uint32x4_t acc =
      vpaddlq_u16(vaddq_u16(hadamard_satd_16x16_pair_max_neon(lp_coeff + 0),
                            hadamard_satd_16x16_pair_max_neon(lp_coeff + 8)));
  const int16_t *lp = lp_coeff + 16;
  const int16_t *const lp_end = lp_coeff + 64;
// Disabling unrolling on this 3-iteration loop avoids ARMv7 Q-register spills
// and keeps the AArch64 loop compact.
#if defined(__clang__)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__) && __GNUC__ >= 8
#pragma GCC unroll 1
#endif
  do {
    acc =
        vpadalq_u16(acc, vaddq_u16(hadamard_satd_16x16_pair_max_neon(lp + 0),
                                   hadamard_satd_16x16_pair_max_neon(lp + 8)));
    lp += 16;
  } while (lp != lp_end);
  return (int)(horizontal_add_u32x4(acc) << 1);
}

static inline uint16x8_t hadamard_satd_32x32_stage3_pair_max_neon(
    int16x8_t a0, int16x8_t a1, int16x8_t a2, int16x8_t a3) {
  // Stage 2 outputs fit in [-32640, 32640]; >> 2 keeps b0..b3 in
  // [-16320, 16320], so m0 + m1 <= 32640 and two slices sum to <= 65280.
  const int16x8_t b0 = vshrq_n_s16(vhaddq_s16(a0, a1), 1);
  const int16x8_t b1 = vshrq_n_s16(vhsubq_s16(a0, a1), 1);
  const int16x8_t b2 = vshrq_n_s16(vhaddq_s16(a2, a3), 1);
  const int16x8_t b3 = vshrq_n_s16(vhsubq_s16(a2, a3), 1);
  const uint16x8_t m0 =
      vreinterpretq_u16_s16(vmaxq_s16(vabsq_s16(b0), vabsq_s16(b2)));
  const uint16x8_t m1 =
      vreinterpretq_u16_s16(vmaxq_s16(vabsq_s16(b1), vabsq_s16(b3)));
  return vaddq_u16(m0, m1);
}

#if AOM_ARCH_AARCH64
static inline void hadamard_lp_16x16_stage2_col_neon(const int16_t *lp_coeff,
                                                     int16x8_t *c0,
                                                     int16x8_t *c1,
                                                     int16x8_t *c2,
                                                     int16x8_t *c3) {
  const int16x8_t a0 = vld1q_s16(lp_coeff + 0);
  const int16x8_t a1 = vld1q_s16(lp_coeff + 64);
  const int16x8_t a2 = vld1q_s16(lp_coeff + 128);
  const int16x8_t a3 = vld1q_s16(lp_coeff + 192);
  const int16x8_t b0 = vhaddq_s16(a0, a1);
  const int16x8_t b1 = vhsubq_s16(a0, a1);
  const int16x8_t b2 = vhaddq_s16(a2, a3);
  const int16x8_t b3 = vhsubq_s16(a2, a3);
  *c0 = vaddq_s16(b0, b2);
  *c1 = vaddq_s16(b1, b3);
  *c2 = vsubq_s16(b0, b2);
  *c3 = vsubq_s16(b1, b3);
}

static inline void hadamard_satd_32x32_stage3_block_neon(const int16_t *lp,
                                                         uint16x8_t *sum01,
                                                         uint16x8_t *sum23) {
  int16x8_t q0_0, q0_1, q0_2, q0_3;
  int16x8_t q1_0, q1_1, q1_2, q1_3;
  int16x8_t q2_0, q2_1, q2_2, q2_3;
  int16x8_t q3_0, q3_1, q3_2, q3_3;
  hadamard_lp_16x16_stage2_col_neon(lp + 0, &q0_0, &q0_1, &q0_2, &q0_3);
  hadamard_lp_16x16_stage2_col_neon(lp + 256, &q1_0, &q1_1, &q1_2, &q1_3);
  hadamard_lp_16x16_stage2_col_neon(lp + 512, &q2_0, &q2_1, &q2_2, &q2_3);
  hadamard_lp_16x16_stage2_col_neon(lp + 768, &q3_0, &q3_1, &q3_2, &q3_3);
  const uint16x8_t p0 =
      hadamard_satd_32x32_stage3_pair_max_neon(q0_0, q1_0, q2_0, q3_0);
  const uint16x8_t p1 =
      hadamard_satd_32x32_stage3_pair_max_neon(q0_1, q1_1, q2_1, q3_1);
  const uint16x8_t p2 =
      hadamard_satd_32x32_stage3_pair_max_neon(q0_2, q1_2, q2_2, q3_2);
  const uint16x8_t p3 =
      hadamard_satd_32x32_stage3_pair_max_neon(q0_3, q1_3, q2_3, q3_3);
  *sum01 = vaddq_u16(p0, p1);
  *sum23 = vaddq_u16(p2, p3);
}
#else
static inline void hadamard_lp_16x16_stage2_col_even_neon(
    const int16_t *lp_coeff, int16x8_t *c0, int16x8_t *c2) {
  const int16x8_t a0 = vld1q_s16(lp_coeff + 0);
  const int16x8_t a1 = vld1q_s16(lp_coeff + 64);
  const int16x8_t a2 = vld1q_s16(lp_coeff + 128);
  const int16x8_t a3 = vld1q_s16(lp_coeff + 192);
  const int16x8_t b0 = vhaddq_s16(a0, a1);
  const int16x8_t b2 = vhaddq_s16(a2, a3);
  *c0 = vaddq_s16(b0, b2);
  *c2 = vsubq_s16(b0, b2);
}

static inline void hadamard_lp_16x16_stage2_col_odd_neon(
    const int16_t *lp_coeff, int16x8_t *c1, int16x8_t *c3) {
  const int16x8_t a0 = vld1q_s16(lp_coeff + 0);
  const int16x8_t a1 = vld1q_s16(lp_coeff + 64);
  const int16x8_t a2 = vld1q_s16(lp_coeff + 128);
  const int16x8_t a3 = vld1q_s16(lp_coeff + 192);
  const int16x8_t b1 = vhsubq_s16(a0, a1);
  const int16x8_t b3 = vhsubq_s16(a2, a3);
  *c1 = vaddq_s16(b1, b3);
  *c3 = vsubq_s16(b1, b3);
}

static inline uint16x8_t hadamard_satd_32x32_stage3_half_even_neon(
    const int16_t *lp) {
  int16x8_t q0_0, q0_2, q1_0, q1_2, q2_0, q2_2, q3_0, q3_2;
  hadamard_lp_16x16_stage2_col_even_neon(lp + 0, &q0_0, &q0_2);
  hadamard_lp_16x16_stage2_col_even_neon(lp + 256, &q1_0, &q1_2);
  hadamard_lp_16x16_stage2_col_even_neon(lp + 512, &q2_0, &q2_2);
  hadamard_lp_16x16_stage2_col_even_neon(lp + 768, &q3_0, &q3_2);
  const uint16x8_t p0 =
      hadamard_satd_32x32_stage3_pair_max_neon(q0_0, q1_0, q2_0, q3_0);
  const uint16x8_t p2 =
      hadamard_satd_32x32_stage3_pair_max_neon(q0_2, q1_2, q2_2, q3_2);
  return vaddq_u16(p0, p2);
}

static inline uint16x8_t hadamard_satd_32x32_stage3_half_odd_neon(
    const int16_t *lp) {
  int16x8_t q0_1, q0_3, q1_1, q1_3, q2_1, q2_3, q3_1, q3_3;
  hadamard_lp_16x16_stage2_col_odd_neon(lp + 0, &q0_1, &q0_3);
  hadamard_lp_16x16_stage2_col_odd_neon(lp + 256, &q1_1, &q1_3);
  hadamard_lp_16x16_stage2_col_odd_neon(lp + 512, &q2_1, &q2_3);
  hadamard_lp_16x16_stage2_col_odd_neon(lp + 768, &q3_1, &q3_3);
  const uint16x8_t p1 =
      hadamard_satd_32x32_stage3_pair_max_neon(q0_1, q1_1, q2_1, q3_1);
  const uint16x8_t p3 =
      hadamard_satd_32x32_stage3_pair_max_neon(q0_3, q1_3, q2_3, q3_3);
  return vaddq_u16(p1, p3);
}
#endif  // AOM_ARCH_AARCH64

int av1_hadamard_satd_32x32_sub_neon(const uint8_t *src, ptrdiff_t src_stride,
                                     const uint8_t *pred,
                                     ptrdiff_t pred_stride) {
  DECLARE_ALIGNED(16, int16_t, lp_coeff[1024]);
  for (int q = 0; q < 4; ++q) {
    const ptrdiff_t row_off = (q >> 1) * 16;
    const ptrdiff_t col_off = (q & 1) * 16;
    hadamard_lp_16x16_quadrants_sub_neon(src + row_off * src_stride + col_off,
                                         src_stride,
                                         pred + row_off * pred_stride + col_off,
                                         pred_stride, lp_coeff + q * 256);
  }
#if AOM_ARCH_AARCH64
  uint16x8_t sum01, sum23;
  hadamard_satd_32x32_stage3_block_neon(lp_coeff, &sum01, &sum23);
  uint32x4_t acc0 = vpaddlq_u16(sum01);
  uint32x4_t acc1 = vpaddlq_u16(sum23);
  const int16_t *lp = lp_coeff + 8;
  const int16_t *const lp_end = lp_coeff + 64;
#if defined(__clang__)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__) && __GNUC__ >= 8
#pragma GCC unroll 1
#endif
  do {
    hadamard_satd_32x32_stage3_block_neon(lp, &sum01, &sum23);
    acc0 = vpadalq_u16(acc0, sum01);
    acc1 = vpadalq_u16(acc1, sum23);
    lp += 8;
  } while (lp != lp_end);
#else
  // Split Stage 2+3 into even (c0, c2) and odd (c1, c3) passes on ARMv7 so
  // live vectors stay within the 16 Q-register budget without stack spills.
  // Peel the final slice (lp_coeff + 56) for acc1 so the compiler does not CSE
  // lp_coeff's 16 loads across the even loop.
  uint32x4_t acc0 =
      vpaddlq_u16(hadamard_satd_32x32_stage3_half_even_neon(lp_coeff));
  const int16_t *const lp_end = lp_coeff + 64;
  const int16_t *lp = lp_coeff + 8;
#if defined(__clang__)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__) && __GNUC__ >= 8
#pragma GCC unroll 1
#endif
  do {
    acc0 = vpadalq_u16(acc0, hadamard_satd_32x32_stage3_half_even_neon(lp));
    lp += 8;
  } while (lp != lp_end);

  const int16_t *const lp_odd_end = lp_coeff + 56;
  uint32x4_t acc1 =
      vpaddlq_u16(hadamard_satd_32x32_stage3_half_odd_neon(lp_odd_end));
  lp = lp_coeff;
#if defined(__clang__)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__) && __GNUC__ >= 8
#pragma GCC unroll 1
#endif
  do {
    acc1 = vpadalq_u16(acc1, hadamard_satd_32x32_stage3_half_odd_neon(lp));
    lp += 8;
  } while (lp != lp_odd_end);
#endif  // AOM_ARCH_AARCH64
  return (int)(horizontal_add_u32x4(vaddq_u32(acc0, acc1)) << 1);
}
