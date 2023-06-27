/*
 * Copyright (c) 2018, Alliance for Open Media. All rights reserved
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
#include <memory.h>
#include <math.h>

#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/arm/sum_neon.h"
#include "aom_dsp/arm/transpose_neon.h"
#include "aom_ports/mem.h"
#include "config/av1_rtcd.h"
#include "av1/common/warped_motion.h"
#include "av1/common/scale.h"

#ifndef __has_include
#define __has_include(path) 0
#endif

#if AOM_ARCH_AARCH64 && defined(__ARM_NEON_SVE_BRIDGE) && \
    defined(__ARM_FEATURE_SVE) && __has_include(<arm_neon_sve_bridge.h>)
#include <arm_neon_sve_bridge.h>
#define AOM_HAVE_NEON_SVE_BRIDGE 1
#else
#define AOM_HAVE_NEON_SVE_BRIDGE 0
#endif

#if AOM_HAVE_NEON_SVE_BRIDGE
static INLINE int64x2_t aom_sdotq_s16(int64x2_t acc, int16x8_t x, int16x8_t y) {
  // The 16-bit dot product instructions only exist in SVE and not Neon.
  // We can get away without rewriting the existing Neon code by making use of
  // the Neon-SVE bridge intrinsics to reinterpret a Neon vector as a SVE
  // vector with the high part of the vector being "don't care", and then
  // operating on that instead.
  // This is clearly suboptimal in machines with a SVE vector length above
  // 128-bits as the remainder of the vector is wasted, however this appears to
  // still be beneficial compared to not using the instruction.
  return svget_neonq_s64(svdot_s64(svset_neonq_s64(svundef_s64(), acc),
                                   svset_neonq_s16(svundef_s16(), x),
                                   svset_neonq_s16(svundef_s16(), y)));
}
#endif  // AOM_HAVE_NEON_SVE_BRIDGE

static INLINE void horizontal_filter_neon(const uint8x16_t in,
                                          int16x8_t *tmp_dst, int sx, int alpha,
                                          int k) {
  const int32x4_t add_const = vdupq_n_s32(1 << (8 + FILTER_BITS - 1));

  // Loading the 8 filter taps
  const int16x8_t f0 =
      vld1q_s16((int16_t *)(av1_warped_filter +
                            ((sx + 0 * alpha) >> WARPEDDIFF_PREC_BITS)));
  const int16x8_t f1 =
      vld1q_s16((int16_t *)(av1_warped_filter +
                            ((sx + 1 * alpha) >> WARPEDDIFF_PREC_BITS)));
  const int16x8_t f2 =
      vld1q_s16((int16_t *)(av1_warped_filter +
                            ((sx + 2 * alpha) >> WARPEDDIFF_PREC_BITS)));
  const int16x8_t f3 =
      vld1q_s16((int16_t *)(av1_warped_filter +
                            ((sx + 3 * alpha) >> WARPEDDIFF_PREC_BITS)));
  const int16x8_t f4 =
      vld1q_s16((int16_t *)(av1_warped_filter +
                            ((sx + 4 * alpha) >> WARPEDDIFF_PREC_BITS)));
  const int16x8_t f5 =
      vld1q_s16((int16_t *)(av1_warped_filter +
                            ((sx + 5 * alpha) >> WARPEDDIFF_PREC_BITS)));
  const int16x8_t f6 =
      vld1q_s16((int16_t *)(av1_warped_filter +
                            ((sx + 6 * alpha) >> WARPEDDIFF_PREC_BITS)));
  const int16x8_t f7 =
      vld1q_s16((int16_t *)(av1_warped_filter +
                            ((sx + 7 * alpha) >> WARPEDDIFF_PREC_BITS)));

#if AOM_ARCH_AARCH64 && defined(__ARM_FEATURE_MATMUL_INT8)
  int8x16_t f01_u8 = vcombine_s8(vmovn_s16(f0), vmovn_s16(f1));
  int8x16_t f23_u8 = vcombine_s8(vmovn_s16(f2), vmovn_s16(f3));
  int8x16_t f45_u8 = vcombine_s8(vmovn_s16(f4), vmovn_s16(f5));
  int8x16_t f67_u8 = vcombine_s8(vmovn_s16(f6), vmovn_s16(f7));

  uint8x8_t in0 = vget_low_u8(in);
  uint8x8_t in1 = vget_low_u8(vextq_u8(in, in, 1));
  uint8x8_t in2 = vget_low_u8(vextq_u8(in, in, 2));
  uint8x8_t in3 = vget_low_u8(vextq_u8(in, in, 3));
  uint8x8_t in4 = vget_low_u8(vextq_u8(in, in, 4));
  uint8x8_t in5 = vget_low_u8(vextq_u8(in, in, 5));
  uint8x8_t in6 = vget_low_u8(vextq_u8(in, in, 6));
  uint8x8_t in7 = vget_low_u8(vextq_u8(in, in, 7));

  int32x4_t m01 = vusdotq_s32(vdupq_n_s32(0), vcombine_u8(in0, in1), f01_u8);
  int32x4_t m23 = vusdotq_s32(vdupq_n_s32(0), vcombine_u8(in2, in3), f23_u8);
  int32x4_t m45 = vusdotq_s32(vdupq_n_s32(0), vcombine_u8(in4, in5), f45_u8);
  int32x4_t m67 = vusdotq_s32(vdupq_n_s32(0), vcombine_u8(in6, in7), f67_u8);

  int32x4_t tmp_res_low = vpaddq_s32(m01, m23);
  int32x4_t tmp_res_high = vpaddq_s32(m45, m67);
#else   // !(AOM_ARCH_AARCH64 && defined(__ARM_FEATURE_MATMUL_INT8))
  int16x8_t in16_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(in)));
  int16x8_t in16_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(in)));

  int16x8_t m0 = vmulq_s16(f0, in16_lo);
  int16x8_t m1 = vmulq_s16(f1, vextq_s16(in16_lo, in16_hi, 1));
  int16x8_t m2 = vmulq_s16(f2, vextq_s16(in16_lo, in16_hi, 2));
  int16x8_t m3 = vmulq_s16(f3, vextq_s16(in16_lo, in16_hi, 3));
  int16x8_t m4 = vmulq_s16(f4, vextq_s16(in16_lo, in16_hi, 4));
  int16x8_t m5 = vmulq_s16(f5, vextq_s16(in16_lo, in16_hi, 5));
  int16x8_t m6 = vmulq_s16(f6, vextq_s16(in16_lo, in16_hi, 6));
  int16x8_t m7 = vmulq_s16(f7, vextq_s16(in16_lo, in16_hi, 7));

  int32x4_t m0123_pairs[] = { vpaddlq_s16(m0), vpaddlq_s16(m1), vpaddlq_s16(m2),
                              vpaddlq_s16(m3) };
  int32x4_t m4567_pairs[] = { vpaddlq_s16(m4), vpaddlq_s16(m5), vpaddlq_s16(m6),
                              vpaddlq_s16(m7) };

  int32x4_t tmp_res_low = horizontal_add_4d_s32x4(m0123_pairs);
  int32x4_t tmp_res_high = horizontal_add_4d_s32x4(m4567_pairs);
#endif  // AOM_ARCH_AARCH64 && defined(__ARM_FEATURE_MATMUL_INT8)

  tmp_res_low = vaddq_s32(tmp_res_low, add_const);
  tmp_res_high = vaddq_s32(tmp_res_high, add_const);

  uint16x8_t res = vcombine_u16(vqrshrun_n_s32(tmp_res_low, ROUND0_BITS),
                                vqrshrun_n_s32(tmp_res_high, ROUND0_BITS));
  tmp_dst[k + 7] = vreinterpretq_s16_u16(res);
}

static INLINE void vertical_filter_neon(const int16x8_t *src,
                                        int32x4_t *res_low, int32x4_t *res_high,
                                        int sy, int gamma) {
  int16x8_t s0 = src[0];
  int16x8_t s1 = src[1];
  int16x8_t s2 = src[2];
  int16x8_t s3 = src[3];
  int16x8_t s4 = src[4];
  int16x8_t s5 = src[5];
  int16x8_t s6 = src[6];
  int16x8_t s7 = src[7];
  transpose_s16_8x8(&s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7);

  int16x8_t f0 =
      vld1q_s16((int16_t *)(av1_warped_filter +
                            ((sy + 0 * gamma) >> WARPEDDIFF_PREC_BITS)));
  int16x8_t f1 =
      vld1q_s16((int16_t *)(av1_warped_filter +
                            ((sy + 1 * gamma) >> WARPEDDIFF_PREC_BITS)));
  int16x8_t f2 =
      vld1q_s16((int16_t *)(av1_warped_filter +
                            ((sy + 2 * gamma) >> WARPEDDIFF_PREC_BITS)));
  int16x8_t f3 =
      vld1q_s16((int16_t *)(av1_warped_filter +
                            ((sy + 3 * gamma) >> WARPEDDIFF_PREC_BITS)));
  int16x8_t f4 =
      vld1q_s16((int16_t *)(av1_warped_filter +
                            ((sy + 4 * gamma) >> WARPEDDIFF_PREC_BITS)));
  int16x8_t f5 =
      vld1q_s16((int16_t *)(av1_warped_filter +
                            ((sy + 5 * gamma) >> WARPEDDIFF_PREC_BITS)));
  int16x8_t f6 =
      vld1q_s16((int16_t *)(av1_warped_filter +
                            ((sy + 6 * gamma) >> WARPEDDIFF_PREC_BITS)));
  int16x8_t f7 =
      vld1q_s16((int16_t *)(av1_warped_filter +
                            ((sy + 7 * gamma) >> WARPEDDIFF_PREC_BITS)));

#if AOM_HAVE_NEON_SVE_BRIDGE
  int64x2_t m0 = aom_sdotq_s16(vdupq_n_s64(0), s0, f0);
  int64x2_t m1 = aom_sdotq_s16(vdupq_n_s64(0), s1, f1);
  int64x2_t m2 = aom_sdotq_s16(vdupq_n_s64(0), s2, f2);
  int64x2_t m3 = aom_sdotq_s16(vdupq_n_s64(0), s3, f3);
  int64x2_t m4 = aom_sdotq_s16(vdupq_n_s64(0), s4, f4);
  int64x2_t m5 = aom_sdotq_s16(vdupq_n_s64(0), s5, f5);
  int64x2_t m6 = aom_sdotq_s16(vdupq_n_s64(0), s6, f6);
  int64x2_t m7 = aom_sdotq_s16(vdupq_n_s64(0), s7, f7);

  int64x2_t m01 = vpaddq_s64(m0, m1);
  int64x2_t m23 = vpaddq_s64(m2, m3);
  int64x2_t m45 = vpaddq_s64(m4, m5);
  int64x2_t m67 = vpaddq_s64(m6, m7);

  *res_low = vcombine_s32(vmovn_s64(m01), vmovn_s64(m23));
  *res_high = vcombine_s32(vmovn_s64(m45), vmovn_s64(m67));
#else   // !AOM_HAVE_NEON_SVE_BRIDGE
  int32x4_t m0 = vmull_s16(vget_low_s16(s0), vget_low_s16(f0));
  m0 = vmlal_s16(m0, vget_high_s16(s0), vget_high_s16(f0));
  int32x4_t m1 = vmull_s16(vget_low_s16(s1), vget_low_s16(f1));
  m1 = vmlal_s16(m1, vget_high_s16(s1), vget_high_s16(f1));
  int32x4_t m2 = vmull_s16(vget_low_s16(s2), vget_low_s16(f2));
  m2 = vmlal_s16(m2, vget_high_s16(s2), vget_high_s16(f2));
  int32x4_t m3 = vmull_s16(vget_low_s16(s3), vget_low_s16(f3));
  m3 = vmlal_s16(m3, vget_high_s16(s3), vget_high_s16(f3));
  int32x4_t m4 = vmull_s16(vget_low_s16(s4), vget_low_s16(f4));
  m4 = vmlal_s16(m4, vget_high_s16(s4), vget_high_s16(f4));
  int32x4_t m5 = vmull_s16(vget_low_s16(s5), vget_low_s16(f5));
  m5 = vmlal_s16(m5, vget_high_s16(s5), vget_high_s16(f5));
  int32x4_t m6 = vmull_s16(vget_low_s16(s6), vget_low_s16(f6));
  m6 = vmlal_s16(m6, vget_high_s16(s6), vget_high_s16(f6));
  int32x4_t m7 = vmull_s16(vget_low_s16(s7), vget_low_s16(f7));
  m7 = vmlal_s16(m7, vget_high_s16(s7), vget_high_s16(f7));

  int32x4_t m0123_pairs[] = { m0, m1, m2, m3 };
  int32x4_t m4567_pairs[] = { m4, m5, m6, m7 };

  *res_low = horizontal_add_4d_s32x4(m0123_pairs);
  *res_high = horizontal_add_4d_s32x4(m4567_pairs);
#endif  // AOM_HAVE_NEON_SVE_BRIDGE
}

static void warp_affine_horizontal_neon(const uint8_t *ref, int width,
                                        int height, int stride, int p_height,
                                        int16_t alpha, int16_t beta,
                                        const int64_t x4, const int64_t y4,
                                        const int i, int16x8_t tmp[],
                                        const uint8x16_t indx_vec) {
  const int reduce_bits_horiz = ROUND0_BITS;
  const int bd = 8;

  int32_t ix4 = (int32_t)(x4 >> WARPEDMODEL_PREC_BITS);
  int32_t iy4 = (int32_t)(y4 >> WARPEDMODEL_PREC_BITS);

  int32_t sx4 = x4 & ((1 << WARPEDMODEL_PREC_BITS) - 1);
  sx4 += alpha * (-4) + beta * (-4) + (1 << (WARPEDDIFF_PREC_BITS - 1)) +
         (WARPEDPIXEL_PREC_SHIFTS << WARPEDDIFF_PREC_BITS);
  sx4 &= ~((1 << WARP_PARAM_REDUCE_BITS) - 1);

  if (ix4 <= -7) {
    for (int k = -7; k < AOMMIN(8, p_height - i); ++k) {
      int iy = iy4 + k;
      if (iy < 0)
        iy = 0;
      else if (iy > height - 1)
        iy = height - 1;
      int16_t dup_val =
          (1 << (bd + FILTER_BITS - reduce_bits_horiz - 1)) +
          ref[iy * stride] * (1 << (FILTER_BITS - reduce_bits_horiz));

      tmp[k + 7] = vdupq_n_s16(dup_val);
    }
  } else if (ix4 >= width + 6) {
    for (int k = -7; k < AOMMIN(8, p_height - i); ++k) {
      int iy = iy4 + k;
      if (iy < 0)
        iy = 0;
      else if (iy > height - 1)
        iy = height - 1;
      int16_t dup_val = (1 << (bd + FILTER_BITS - reduce_bits_horiz - 1)) +
                        ref[iy * stride + (width - 1)] *
                            (1 << (FILTER_BITS - reduce_bits_horiz));
      tmp[k + 7] = vdupq_n_s16(dup_val);
    }
  } else if (((ix4 - 7) < 0) || ((ix4 + 9) > width)) {
    const int out_of_boundary_left = -(ix4 - 6);
    const int out_of_boundary_right = (ix4 + 8) - width;

    for (int k = -7; k < AOMMIN(8, p_height - i); ++k) {
      int iy = iy4 + k;
      if (iy < 0)
        iy = 0;
      else if (iy > height - 1)
        iy = height - 1;
      int sx = sx4 + beta * (k + 4);

      const uint8_t *src = ref + iy * stride + ix4 - 7;
      uint8x16_t src_1 = vld1q_u8(src);

      if (out_of_boundary_left >= 0) {
        int limit = out_of_boundary_left + 1;
        uint8x16_t cmp_vec = vdupq_n_u8(out_of_boundary_left);
        uint8x16_t vec_dup = vdupq_n_u8(*(src + limit));
        uint8x16_t mask_val = vcleq_u8(indx_vec, cmp_vec);
        src_1 = vbslq_u8(mask_val, vec_dup, src_1);
      }
      if (out_of_boundary_right >= 0) {
        int limit = 15 - (out_of_boundary_right + 1);
        uint8x16_t cmp_vec = vdupq_n_u8(15 - out_of_boundary_right);
        uint8x16_t vec_dup = vdupq_n_u8(*(src + limit));
        uint8x16_t mask_val = vcgeq_u8(indx_vec, cmp_vec);
        src_1 = vbslq_u8(mask_val, vec_dup, src_1);
      }
      horizontal_filter_neon(src_1, tmp, sx, alpha, k);
    }
  } else {
    for (int k = -7; k < AOMMIN(8, p_height - i); ++k) {
      int iy = iy4 + k;
      if (iy < 0)
        iy = 0;
      else if (iy > height - 1)
        iy = height - 1;
      int sx = sx4 + beta * (k + 4);

      const uint8_t *src = ref + iy * stride + ix4 - 7;
      uint8x16_t src_1 = vld1q_u8(src);
      horizontal_filter_neon(src_1, tmp, sx, alpha, k);
    }
  }
}

static void warp_affine_vertical_neon(
    uint8_t *pred, int p_width, int p_height, int p_stride,
    ConvolveParams *conv_params, int16_t gamma, int16_t delta, const int64_t y4,
    const int i, const int j, int16x8_t tmp[], const int16x4_t res_sub_const,
    const int32x4_t shift_vert, const int32x4_t fwd, const int32x4_t bwd,
    const int16x4_t round_bits_vec) {
  const int bd = 8;
  const int reduce_bits_horiz = ROUND0_BITS;
  const int offset_bits_vert = bd + 2 * FILTER_BITS - reduce_bits_horiz;
  const int32x4_t add_const_vert =
      vdupq_n_s32((int32_t)(1 << offset_bits_vert));
  const int16x8_t sub_constant = vdupq_n_s16((1 << (bd - 1)) + (1 << bd));

  int32_t sy4 = y4 & ((1 << WARPEDMODEL_PREC_BITS) - 1);
  sy4 += gamma * (-4) + delta * (-4) + (1 << (WARPEDDIFF_PREC_BITS - 1)) +
         (WARPEDPIXEL_PREC_SHIFTS << WARPEDDIFF_PREC_BITS);
  sy4 &= ~((1 << WARP_PARAM_REDUCE_BITS) - 1);

  for (int k = -4; k < AOMMIN(4, p_height - i - 4); ++k) {
    int sy = sy4 + delta * (k + 4);

    const int16x8_t *v_src = tmp + (k + 4);

    int32x4_t res_lo, res_hi;
    vertical_filter_neon(v_src, &res_lo, &res_hi, sy, gamma);

    res_lo = vaddq_s32(res_lo, add_const_vert);
    res_hi = vaddq_s32(res_hi, add_const_vert);

    if (conv_params->is_compound) {
      uint16_t *const p = (uint16_t *)&conv_params
                              ->dst[(i + k + 4) * conv_params->dst_stride + j];

      res_lo = vrshlq_s32(res_lo, shift_vert);
      if (conv_params->do_average) {
        uint8_t *const dst8 = &pred[(i + k + 4) * p_stride + j];
        uint16x4_t tmp16_lo = vld1_u16(p);
        int32x4_t tmp32_lo = vreinterpretq_s32_u32(vmovl_u16(tmp16_lo));
        int16x4_t tmp16_low;
        if (conv_params->use_dist_wtd_comp_avg) {
          res_lo = vmulq_s32(res_lo, bwd);
          tmp32_lo = vmulq_s32(tmp32_lo, fwd);
          tmp32_lo = vaddq_s32(tmp32_lo, res_lo);
          tmp16_low = vshrn_n_s32(tmp32_lo, DIST_PRECISION_BITS);
        } else {
          tmp32_lo = vaddq_s32(tmp32_lo, res_lo);
          tmp16_low = vshrn_n_s32(tmp32_lo, 1);
        }
        int16x4_t res_low = vadd_s16(tmp16_low, res_sub_const);
        res_low = vqrshl_s16(res_low, round_bits_vec);
        int16x8_t final_res_low = vcombine_s16(res_low, res_low);
        uint8x8_t res_8_low = vqmovun_s16(final_res_low);

        vst1_lane_u32((uint32_t *)dst8, vreinterpret_u32_u8(res_8_low), 0);
      } else {
        uint16x4_t res_u16_low = vqmovun_s32(res_lo);
        vst1_u16(p, res_u16_low);
      }
      if (p_width > 4) {
        uint16_t *const p4 =
            (uint16_t *)&conv_params
                ->dst[(i + k + 4) * conv_params->dst_stride + j + 4];

        res_hi = vrshlq_s32(res_hi, shift_vert);
        if (conv_params->do_average) {
          uint8_t *const dst8_4 = &pred[(i + k + 4) * p_stride + j + 4];

          uint16x4_t tmp16_hi = vld1_u16(p4);
          int32x4_t tmp32_hi = vreinterpretq_s32_u32(vmovl_u16(tmp16_hi));
          int16x4_t tmp16_high;
          if (conv_params->use_dist_wtd_comp_avg) {
            res_hi = vmulq_s32(res_hi, bwd);
            tmp32_hi = vmulq_s32(tmp32_hi, fwd);
            tmp32_hi = vaddq_s32(tmp32_hi, res_hi);
            tmp16_high = vshrn_n_s32(tmp32_hi, DIST_PRECISION_BITS);
          } else {
            tmp32_hi = vaddq_s32(tmp32_hi, res_hi);
            tmp16_high = vshrn_n_s32(tmp32_hi, 1);
          }
          int16x4_t res_high = vadd_s16(tmp16_high, res_sub_const);
          res_high = vqrshl_s16(res_high, round_bits_vec);
          int16x8_t final_res_high = vcombine_s16(res_high, res_high);
          uint8x8_t res_8_high = vqmovun_s16(final_res_high);

          vst1_lane_u32((uint32_t *)dst8_4, vreinterpret_u32_u8(res_8_high), 0);
        } else {
          uint16x4_t res_u16_high = vqmovun_s32(res_hi);
          vst1_u16(p4, res_u16_high);
        }
      }
    } else {
      res_lo = vrshlq_s32(res_lo, shift_vert);
      res_hi = vrshlq_s32(res_hi, shift_vert);

      int16x8_t result_final =
          vcombine_s16(vmovn_s32(res_lo), vmovn_s32(res_hi));
      result_final = vsubq_s16(result_final, sub_constant);

      uint8_t *const p = (uint8_t *)&pred[(i + k + 4) * p_stride + j];
      uint8x8_t val = vqmovun_s16(result_final);

      if (p_width == 4) {
        vst1_lane_u32((uint32_t *)p, vreinterpret_u32_u8(val), 0);
      } else {
        vst1_u8(p, val);
      }
    }
  }
}

void av1_warp_affine_neon(const int32_t *mat, const uint8_t *ref, int width,
                          int height, int stride, uint8_t *pred, int p_col,
                          int p_row, int p_width, int p_height, int p_stride,
                          int subsampling_x, int subsampling_y,
                          ConvolveParams *conv_params, int16_t alpha,
                          int16_t beta, int16_t gamma, int16_t delta) {
  const int bd = 8;
  const int w0 = conv_params->fwd_offset;
  const int w1 = conv_params->bck_offset;
  const int32x4_t fwd = vdupq_n_s32((int32_t)w0);
  const int32x4_t bwd = vdupq_n_s32((int32_t)w1);

  static const uint8_t k0To15[16] = { 0, 1, 2,  3,  4,  5,  6,  7,
                                      8, 9, 10, 11, 12, 13, 14, 15 };
  uint8x16_t indx_vec = vld1q_u8(k0To15);

  const int reduce_bits_horiz = ROUND0_BITS;
  const int reduce_bits_vert = conv_params->is_compound
                                   ? conv_params->round_1
                                   : 2 * FILTER_BITS - reduce_bits_horiz;
  const int32x4_t shift_vert = vdupq_n_s32(-(int32_t)reduce_bits_vert);

  assert(IMPLIES(conv_params->is_compound, conv_params->dst != NULL));

  const int round_bits =
      2 * FILTER_BITS - conv_params->round_0 - conv_params->round_1;
  const int16x4_t round_bits_vec = vdup_n_s16(-(int16_t)round_bits);
  const int offset_bits = bd + 2 * FILTER_BITS - conv_params->round_0;
  const int16x4_t res_sub_const =
      vdup_n_s16(-((1 << (offset_bits - conv_params->round_1)) +
                   (1 << (offset_bits - conv_params->round_1 - 1))));

  assert(IMPLIES(conv_params->do_average, conv_params->is_compound));

  int16x8_t tmp[15];
  for (int i = 0; i < p_height; i += 8) {
    for (int j = 0; j < p_width; j += 8) {
      const int32_t src_x = (p_col + j + 4) << subsampling_x;
      const int32_t src_y = (p_row + i + 4) << subsampling_y;
      const int64_t dst_x =
          (int64_t)mat[2] * src_x + (int64_t)mat[3] * src_y + (int64_t)mat[0];
      const int64_t dst_y =
          (int64_t)mat[4] * src_x + (int64_t)mat[5] * src_y + (int64_t)mat[1];

      const int64_t x4 = dst_x >> subsampling_x;
      const int64_t y4 = dst_y >> subsampling_y;

      warp_affine_horizontal_neon(ref, width, height, stride, p_height, alpha,
                                  beta, x4, y4, i, tmp, indx_vec);
      warp_affine_vertical_neon(pred, p_width, p_height, p_stride, conv_params,
                                gamma, delta, y4, i, j, tmp, res_sub_const,
                                shift_vert, fwd, bwd, round_bits_vec);
    }
  }
}
