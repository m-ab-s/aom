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

#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/arm/sum_neon.h"
#include "aom_dsp/arm/transpose_neon.h"
#include "aom_ports/mem.h"
#include "av1/common/scale.h"
#include "av1/common/warped_motion.h"
#include "config/av1_rtcd.h"

static INLINE int clamp_iy(int iy, int height) {
  if (iy < 0) {
    return 0;
  }
  if (iy > height - 1) {
    return height - 1;
  }
  return iy;
}

static INLINE void load_filters(int16x8_t out[], int ofs, int stride) {
  const int ofs0 = ROUND_POWER_OF_TWO(ofs + stride * 0, WARPEDDIFF_PREC_BITS);
  const int ofs1 = ROUND_POWER_OF_TWO(ofs + stride * 1, WARPEDDIFF_PREC_BITS);
  const int ofs2 = ROUND_POWER_OF_TWO(ofs + stride * 2, WARPEDDIFF_PREC_BITS);
  const int ofs3 = ROUND_POWER_OF_TWO(ofs + stride * 3, WARPEDDIFF_PREC_BITS);
  const int ofs4 = ROUND_POWER_OF_TWO(ofs + stride * 4, WARPEDDIFF_PREC_BITS);
  const int ofs5 = ROUND_POWER_OF_TWO(ofs + stride * 5, WARPEDDIFF_PREC_BITS);
  const int ofs6 = ROUND_POWER_OF_TWO(ofs + stride * 6, WARPEDDIFF_PREC_BITS);
  const int ofs7 = ROUND_POWER_OF_TWO(ofs + stride * 7, WARPEDDIFF_PREC_BITS);

  const int16_t *base =
      (int16_t *)av1_warped_filter + WARPEDPIXEL_PREC_SHIFTS * 8;
  out[0] = vld1q_s16(base + ofs0 * 8);
  out[1] = vld1q_s16(base + ofs1 * 8);
  out[2] = vld1q_s16(base + ofs2 * 8);
  out[3] = vld1q_s16(base + ofs3 * 8);
  out[4] = vld1q_s16(base + ofs4 * 8);
  out[5] = vld1q_s16(base + ofs5 * 8);
  out[6] = vld1q_s16(base + ofs6 * 8);
  out[7] = vld1q_s16(base + ofs7 * 8);
}

static INLINE void warp_affine_horizontal_neon(const uint16_t *ref, int width,
                                               int height, int stride,
                                               int16_t alpha, int16_t beta,
                                               int iy4, int sx4, int ix4,
                                               int32x4x2_t tmp[], int bd) {
  const int round0 = (bd == 12) ? ROUND0_BITS + 2 : ROUND0_BITS;
  const int offset_bits_horiz = bd + FILTER_BITS - 1;

  if (ix4 <= -7) {
    for (int k = -7; k < 8; ++k) {
      int iy = clamp_iy(iy4 + k, height);
      int32_t dup_val = (1 << (bd + FILTER_BITS - round0 - 1)) +
                        ref[iy * stride] * (1 << (FILTER_BITS - round0));
      tmp[k + 7].val[0] = vdupq_n_s32(dup_val);
      tmp[k + 7].val[1] = vdupq_n_s32(dup_val);
    }
    return;
  } else if (ix4 >= width + 6) {
    for (int k = -7; k < 8; ++k) {
      int iy = clamp_iy(iy4 + k, height);
      int32_t dup_val =
          (1 << (bd + FILTER_BITS - round0 - 1)) +
          ref[iy * stride + (width - 1)] * (1 << (FILTER_BITS - round0));
      tmp[k + 7].val[0] = vdupq_n_s32(dup_val);
      tmp[k + 7].val[1] = vdupq_n_s32(dup_val);
    }
    return;
  }

  for (int k = -7; k < 8; ++k) {
    const int iy = clamp(iy4 + k, 0, height - 1);
    const int sx = sx4 + beta * (k + 4);

    uint16x8_t ref_vec[] = { vld1q_u16(ref + iy * stride + ix4 - 7),
                             vld1q_u16(ref + iy * stride + ix4 + 1) };

    const int out_of_boundary_left = -(ix4 - 6);
    const int out_of_boundary_right = (ix4 + 8) - width;

    const uint16_t k0[16] = { 0, 1, 2,  3,  4,  5,  6,  7,
                              8, 9, 10, 11, 12, 13, 14, 15 };
    const uint16x8_t indx0 = vld1q_u16(&k0[0]);
    const uint16x8_t indx1 = vld1q_u16(&k0[8]);

    if (out_of_boundary_left >= 0) {
      uint16x8_t cmp_vec = vdupq_n_u16(out_of_boundary_left);
      uint16x8_t vec_dup = vdupq_n_u16(ref[iy * stride]);
      uint16x8_t mask0 = vcleq_u16(indx0, cmp_vec);
      uint16x8_t mask1 = vcleq_u16(indx1, cmp_vec);
      ref_vec[0] = vbslq_u16(mask0, vec_dup, ref_vec[0]);
      ref_vec[1] = vbslq_u16(mask1, vec_dup, ref_vec[1]);
    }
    if (out_of_boundary_right >= 0) {
      uint16x8_t cmp_vec = vdupq_n_u16(15 - out_of_boundary_right);
      uint16x8_t vec_dup = vdupq_n_u16(ref[iy * stride + width - 1]);
      uint16x8_t mask0 = vcgeq_u16(indx0, cmp_vec);
      uint16x8_t mask1 = vcgeq_u16(indx1, cmp_vec);
      ref_vec[0] = vbslq_u16(mask0, vec_dup, ref_vec[0]);
      ref_vec[1] = vbslq_u16(mask1, vec_dup, ref_vec[1]);
    }

    int16x8_t f[8];
    load_filters(f, sx, alpha);

    int16x8_t rv0 = vreinterpretq_s16_u16(vextq_u16(ref_vec[0], ref_vec[1], 0));
    int16x8_t rv1 = vreinterpretq_s16_u16(vextq_u16(ref_vec[0], ref_vec[1], 1));
    int16x8_t rv2 = vreinterpretq_s16_u16(vextq_u16(ref_vec[0], ref_vec[1], 2));
    int16x8_t rv3 = vreinterpretq_s16_u16(vextq_u16(ref_vec[0], ref_vec[1], 3));
    int16x8_t rv4 = vreinterpretq_s16_u16(vextq_u16(ref_vec[0], ref_vec[1], 4));
    int16x8_t rv5 = vreinterpretq_s16_u16(vextq_u16(ref_vec[0], ref_vec[1], 5));
    int16x8_t rv6 = vreinterpretq_s16_u16(vextq_u16(ref_vec[0], ref_vec[1], 6));
    int16x8_t rv7 = vreinterpretq_s16_u16(vextq_u16(ref_vec[0], ref_vec[1], 7));

    int32x4_t m0 = vmull_s16(vget_low_s16(f[0]), vget_low_s16(rv0));
    m0 = vmlal_s16(m0, vget_high_s16(f[0]), vget_high_s16(rv0));
    int32x4_t m1 = vmull_s16(vget_low_s16(f[1]), vget_low_s16(rv1));
    m1 = vmlal_s16(m1, vget_high_s16(f[1]), vget_high_s16(rv1));
    int32x4_t m2 = vmull_s16(vget_low_s16(f[2]), vget_low_s16(rv2));
    m2 = vmlal_s16(m2, vget_high_s16(f[2]), vget_high_s16(rv2));
    int32x4_t m3 = vmull_s16(vget_low_s16(f[3]), vget_low_s16(rv3));
    m3 = vmlal_s16(m3, vget_high_s16(f[3]), vget_high_s16(rv3));
    int32x4_t m4 = vmull_s16(vget_low_s16(f[4]), vget_low_s16(rv4));
    m4 = vmlal_s16(m4, vget_high_s16(f[4]), vget_high_s16(rv4));
    int32x4_t m5 = vmull_s16(vget_low_s16(f[5]), vget_low_s16(rv5));
    m5 = vmlal_s16(m5, vget_high_s16(f[5]), vget_high_s16(rv5));
    int32x4_t m6 = vmull_s16(vget_low_s16(f[6]), vget_low_s16(rv6));
    m6 = vmlal_s16(m6, vget_high_s16(f[6]), vget_high_s16(rv6));
    int32x4_t m7 = vmull_s16(vget_low_s16(f[7]), vget_low_s16(rv7));
    m7 = vmlal_s16(m7, vget_high_s16(f[7]), vget_high_s16(rv7));

    int32x4_t m0123[] = { m0, m1, m2, m3 };
    int32x4_t m4567[] = { m4, m5, m6, m7 };

    int32x4_t res0 = horizontal_add_4d_s32x4(m0123);
    int32x4_t res1 = horizontal_add_4d_s32x4(m4567);

    res0 = vaddq_s32(res0, vdupq_n_s32(1 << offset_bits_horiz));
    res1 = vaddq_s32(res1, vdupq_n_s32(1 << offset_bits_horiz));

    res0 = vrshlq_s32(res0, vdupq_n_s32(-round0));
    res1 = vrshlq_s32(res1, vdupq_n_s32(-round0));

    tmp[k + 7].val[0] = res0;
    tmp[k + 7].val[1] = res1;
  }
}

static INLINE uint16x4_t clip_pixel_highbd_vec(int32x4_t val, int bd) {
  const int limit = (1 << bd) - 1;
  return vqmovun_s32(vminq_s32(val, vdupq_n_s32(limit)));
}

static INLINE void warp_affine_vertical_step_neon(
    uint16_t *pred, int p_width, int p_stride, int bd, uint16_t *dst,
    int dst_stride, int is_compound, int do_average, int use_dist_wtd_comp_avg,
    int fwd, int bwd, int16_t gamma, const int32x4x2_t *tmp, int i, int sy,
    int j) {
  int32x4x2_t s0 = tmp[0];
  int32x4x2_t s1 = tmp[1];
  int32x4x2_t s2 = tmp[2];
  int32x4x2_t s3 = tmp[3];
  int32x4x2_t s4 = tmp[4];
  int32x4x2_t s5 = tmp[5];
  int32x4x2_t s6 = tmp[6];
  int32x4x2_t s7 = tmp[7];
  transpose_s32_8x8(&s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7);

  int16x8_t f[8];
  load_filters(f, sy, gamma);

  int32x4_t m0 = vmulq_s32(s0.val[0], vmovl_s16(vget_low_s16(f[0])));
  m0 = vmlaq_s32(m0, s0.val[1], vmovl_s16(vget_high_s16(f[0])));
  int32x4_t m1 = vmulq_s32(s1.val[0], vmovl_s16(vget_low_s16(f[1])));
  m1 = vmlaq_s32(m1, s1.val[1], vmovl_s16(vget_high_s16(f[1])));
  int32x4_t m2 = vmulq_s32(s2.val[0], vmovl_s16(vget_low_s16(f[2])));
  m2 = vmlaq_s32(m2, s2.val[1], vmovl_s16(vget_high_s16(f[2])));
  int32x4_t m3 = vmulq_s32(s3.val[0], vmovl_s16(vget_low_s16(f[3])));
  m3 = vmlaq_s32(m3, s3.val[1], vmovl_s16(vget_high_s16(f[3])));
  int32x4_t m4 = vmulq_s32(s4.val[0], vmovl_s16(vget_low_s16(f[4])));
  m4 = vmlaq_s32(m4, s4.val[1], vmovl_s16(vget_high_s16(f[4])));
  int32x4_t m5 = vmulq_s32(s5.val[0], vmovl_s16(vget_low_s16(f[5])));
  m5 = vmlaq_s32(m5, s5.val[1], vmovl_s16(vget_high_s16(f[5])));
  int32x4_t m6 = vmulq_s32(s6.val[0], vmovl_s16(vget_low_s16(f[6])));
  m6 = vmlaq_s32(m6, s6.val[1], vmovl_s16(vget_high_s16(f[6])));
  int32x4_t m7 = vmulq_s32(s7.val[0], vmovl_s16(vget_low_s16(f[7])));
  m7 = vmlaq_s32(m7, s7.val[1], vmovl_s16(vget_high_s16(f[7])));

  int32x4_t m0123[] = { m0, m1, m2, m3 };
  int32x4_t m4567[] = { m4, m5, m6, m7 };

  int32x4_t sums[2];
  sums[0] = horizontal_add_4d_s32x4(m0123);
  sums[1] = horizontal_add_4d_s32x4(m4567);

  const int round0 = (bd == 12) ? ROUND0_BITS + 2 : ROUND0_BITS;
  const int offset_bits_vert = bd + 2 * FILTER_BITS - round0;

  sums[0] = vaddq_s32(sums[0], vdupq_n_s32(1 << offset_bits_vert));
  sums[1] = vaddq_s32(sums[1], vdupq_n_s32(1 << offset_bits_vert));

  uint16_t *p = &dst[i * dst_stride + j];
  uint16_t *dst16 = &pred[i * p_stride + j];

  if (!is_compound) {
    const int reduce_bits_vert = 2 * FILTER_BITS - round0;
    sums[0] = vrshlq_s32(sums[0], vdupq_n_s32(-reduce_bits_vert));
    sums[1] = vrshlq_s32(sums[1], vdupq_n_s32(-reduce_bits_vert));

    const int res_sub_const = (1 << (bd - 1)) + (1 << bd);
    sums[0] = vsubq_s32(sums[0], vdupq_n_s32(res_sub_const));
    sums[1] = vsubq_s32(sums[1], vdupq_n_s32(res_sub_const));
    uint16x4_t res0 = clip_pixel_highbd_vec(sums[0], bd);
    uint16x4_t res1 = clip_pixel_highbd_vec(sums[1], bd);
    vst1_u16(dst16, res0);
    if (p_width > 4) {
      vst1_u16(dst16 + 4, res1);
    }
    return;
  }

  sums[0] = vrshrq_n_s32(sums[0], COMPOUND_ROUND1_BITS);
  sums[1] = vrshrq_n_s32(sums[1], COMPOUND_ROUND1_BITS);

  if (!do_average) {
    vst1_u16(p, vqmovun_s32(sums[0]));
    if (p_width > 4) {
      vst1_u16(p + 4, vqmovun_s32(sums[1]));
    }
    return;
  }

  uint16x8_t p0 = vld1q_u16(p);
  int32x4_t p_vec[] = { vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(p0))),
                        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(p0))) };
  if (use_dist_wtd_comp_avg) {
    p_vec[0] = vmulq_n_s32(p_vec[0], fwd);
    p_vec[1] = vmulq_n_s32(p_vec[1], fwd);
    p_vec[0] = vmlaq_n_s32(p_vec[0], sums[0], bwd);
    p_vec[1] = vmlaq_n_s32(p_vec[1], sums[1], bwd);
    p_vec[0] = vshrq_n_s32(p_vec[0], DIST_PRECISION_BITS);
    p_vec[1] = vshrq_n_s32(p_vec[1], DIST_PRECISION_BITS);
  } else {
    p_vec[0] = vhaddq_s32(p_vec[0], sums[0]);
    p_vec[1] = vhaddq_s32(p_vec[1], sums[1]);
  }

  const int offset_bits = bd + 2 * FILTER_BITS - round0;
  const int round1 = COMPOUND_ROUND1_BITS;
  const int res_sub_const =
      (1 << (offset_bits - round1)) + (1 << (offset_bits - round1 - 1));
  const int round_bits = 2 * FILTER_BITS - round0 - round1;

  p_vec[0] = vsubq_s32(p_vec[0], vdupq_n_s32(res_sub_const));
  p_vec[1] = vsubq_s32(p_vec[1], vdupq_n_s32(res_sub_const));

  p_vec[0] = vrshlq_s32(p_vec[0], vdupq_n_s32(-round_bits));
  p_vec[1] = vrshlq_s32(p_vec[1], vdupq_n_s32(-round_bits));
  uint16x4_t res0 = clip_pixel_highbd_vec(p_vec[0], bd);
  uint16x4_t res1 = clip_pixel_highbd_vec(p_vec[1], bd);
  vst1_u16(dst16, res0);
  if (p_width > 4) {
    vst1_u16(dst16 + 4, res1);
  }
}

static INLINE void warp_affine_vertical_neon(
    uint16_t *pred, int p_width, int p_height, int p_stride, int bd,
    uint16_t *dst, int dst_stride, int is_compound, int do_average,
    int use_dist_wtd_comp_avg, int fwd, int bwd, int16_t gamma, int16_t delta,
    const int32x4x2_t *tmp, int i, int sy4, int j) {
  int limit_height = p_height > 4 ? 8 : 4;
  if (p_width > 4) {
    for (int k = 0; k < limit_height; ++k) {
      int sy = sy4 + delta * k;
      warp_affine_vertical_step_neon(
          pred, 8, p_stride, bd, dst, dst_stride, is_compound, do_average,
          use_dist_wtd_comp_avg, fwd, bwd, gamma, tmp + k, i + k, sy, j);
    }
  } else {
    for (int k = 0; k < limit_height; ++k) {
      int sy = sy4 + delta * k;
      warp_affine_vertical_step_neon(
          pred, 4, p_stride, bd, dst, dst_stride, is_compound, do_average,
          use_dist_wtd_comp_avg, fwd, bwd, gamma, tmp + k, i + k, sy, j);
    }
  }
}

void av1_highbd_warp_affine_neon(const int32_t *mat, const uint16_t *ref,
                                 int width, int height, int stride,
                                 uint16_t *pred, int p_col, int p_row,
                                 int p_width, int p_height, int p_stride,
                                 int subsampling_x, int subsampling_y, int bd,
                                 ConvolveParams *conv_params, int16_t alpha,
                                 int16_t beta, int16_t gamma, int16_t delta) {
  uint16_t *const dst = conv_params->dst;
  const int dst_stride = conv_params->dst_stride;
  const int is_compound = conv_params->is_compound;
  const int do_average = conv_params->do_average;
  const int use_dist_wtd_comp_avg = conv_params->use_dist_wtd_comp_avg;
  const int fwd = conv_params->fwd_offset;
  const int bwd = conv_params->bck_offset;

  assert(IMPLIES(is_compound, dst != NULL));

  for (int i = 0; i < p_height; i += 8) {
    for (int j = 0; j < p_width; j += 8) {
      // Calculate the center of this 8x8 block,
      // project to luma coordinates (if in a subsampled chroma plane),
      // apply the affine transformation,
      // then convert back to the original coordinates (if necessary)
      const int32_t src_x = (j + 4 + p_col) << subsampling_x;
      const int32_t src_y = (i + 4 + p_row) << subsampling_y;
      const int64_t dst_x =
          (int64_t)mat[2] * src_x + (int64_t)mat[3] * src_y + (int64_t)mat[0];
      const int64_t dst_y =
          (int64_t)mat[4] * src_x + (int64_t)mat[5] * src_y + (int64_t)mat[1];
      const int64_t x4 = dst_x >> subsampling_x;
      const int64_t y4 = dst_y >> subsampling_y;

      const int32_t ix4 = (int32_t)(x4 >> WARPEDMODEL_PREC_BITS);
      int32_t sx4 = x4 & ((1 << WARPEDMODEL_PREC_BITS) - 1);
      const int32_t iy4 = (int32_t)(y4 >> WARPEDMODEL_PREC_BITS);
      int32_t sy4 = y4 & ((1 << WARPEDMODEL_PREC_BITS) - 1);

      sx4 += alpha * (-4) + beta * (-4);
      sy4 += gamma * (-4) + delta * (-4);

      sx4 &= ~((1 << WARP_PARAM_REDUCE_BITS) - 1);
      sy4 &= ~((1 << WARP_PARAM_REDUCE_BITS) - 1);

      int32x4x2_t tmp[15];
      warp_affine_horizontal_neon(ref, width, height, stride, alpha, beta, iy4,
                                  sx4, ix4, tmp, bd);
      warp_affine_vertical_neon(pred, p_width, p_height, p_stride, bd, dst,
                                dst_stride, is_compound, do_average,
                                use_dist_wtd_comp_avg, fwd, bwd, gamma, delta,
                                tmp, i, sy4, j);
    }
  }
}
