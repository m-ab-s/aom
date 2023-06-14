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
#include "aom_ports/mem.h"
#include "config/av1_rtcd.h"
#include "av1/common/warped_motion.h"
#include "av1/common/scale.h"

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

  int16x8_t in16_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(in)));
  int16x8_t in16_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(in)));

  int16x8_t m0 = vmulq_s16(f0, in16_lo);
  int16x8_t m1 = vmulq_s16(f2, vextq_s16(in16_lo, in16_hi, 2));
  int16x8_t m2 = vmulq_s16(f4, vextq_s16(in16_lo, in16_hi, 4));
  int16x8_t m3 = vmulq_s16(f6, vextq_s16(in16_lo, in16_hi, 6));
  int16x8_t m4 = vmulq_s16(f1, vextq_s16(in16_lo, in16_hi, 1));
  int16x8_t m5 = vmulq_s16(f3, vextq_s16(in16_lo, in16_hi, 3));
  int16x8_t m6 = vmulq_s16(f5, vextq_s16(in16_lo, in16_hi, 5));
  int16x8_t m7 = vmulq_s16(f7, vextq_s16(in16_lo, in16_hi, 7));

  int32x4_t m0123_pairs[] = { vpaddlq_s16(m0), vpaddlq_s16(m1), vpaddlq_s16(m2),
                              vpaddlq_s16(m3) };
  int32x4_t m4567_pairs[] = { vpaddlq_s16(m4), vpaddlq_s16(m5), vpaddlq_s16(m6),
                              vpaddlq_s16(m7) };

  int32x4_t tmp_res_low = horizontal_add_4d_s32x4(m0123_pairs);
  int32x4_t tmp_res_high = horizontal_add_4d_s32x4(m4567_pairs);

  tmp_res_low = vaddq_s32(tmp_res_low, add_const);
  tmp_res_high = vaddq_s32(tmp_res_high, add_const);

  uint16x8_t res = vcombine_u16(vqrshrun_n_s32(tmp_res_low, ROUND0_BITS),
                                vqrshrun_n_s32(tmp_res_high, ROUND0_BITS));
  tmp_dst[k + 7] = vreinterpretq_s16_u16(res);
}

static INLINE void vertical_filter_neon(const int16x8_t *src,
                                        int32x4_t *res_low, int32x4_t *res_high,
                                        int sy, int gamma) {
  int16x4_t src_0, src_1, fltr_0, fltr_1;
  int32x4_t res_0, res_1;
  int32x2_t res_0_im, res_1_im;
  int32x4_t res_even, res_odd, im_res_0, im_res_1;

  int16x8_t f0, f1, f2, f3, f4, f5, f6, f7;
  int16x8x2_t b0, b1, b2, b3;
  int32x4x2_t c0, c1, c2, c3;
  int32x4x2_t d0, d1, d2, d3;

  b0 = vtrnq_s16(src[0], src[1]);
  b1 = vtrnq_s16(src[2], src[3]);
  b2 = vtrnq_s16(src[4], src[5]);
  b3 = vtrnq_s16(src[6], src[7]);

  c0 = vtrnq_s32(vreinterpretq_s32_s16(b0.val[0]),
                 vreinterpretq_s32_s16(b0.val[1]));
  c1 = vtrnq_s32(vreinterpretq_s32_s16(b1.val[0]),
                 vreinterpretq_s32_s16(b1.val[1]));
  c2 = vtrnq_s32(vreinterpretq_s32_s16(b2.val[0]),
                 vreinterpretq_s32_s16(b2.val[1]));
  c3 = vtrnq_s32(vreinterpretq_s32_s16(b3.val[0]),
                 vreinterpretq_s32_s16(b3.val[1]));

  f0 = vld1q_s16((int16_t *)(av1_warped_filter +
                             ((sy + 0 * gamma) >> WARPEDDIFF_PREC_BITS)));
  f1 = vld1q_s16((int16_t *)(av1_warped_filter +
                             ((sy + 1 * gamma) >> WARPEDDIFF_PREC_BITS)));
  f2 = vld1q_s16((int16_t *)(av1_warped_filter +
                             ((sy + 2 * gamma) >> WARPEDDIFF_PREC_BITS)));
  f3 = vld1q_s16((int16_t *)(av1_warped_filter +
                             ((sy + 3 * gamma) >> WARPEDDIFF_PREC_BITS)));
  f4 = vld1q_s16((int16_t *)(av1_warped_filter +
                             ((sy + 4 * gamma) >> WARPEDDIFF_PREC_BITS)));
  f5 = vld1q_s16((int16_t *)(av1_warped_filter +
                             ((sy + 5 * gamma) >> WARPEDDIFF_PREC_BITS)));
  f6 = vld1q_s16((int16_t *)(av1_warped_filter +
                             ((sy + 6 * gamma) >> WARPEDDIFF_PREC_BITS)));
  f7 = vld1q_s16((int16_t *)(av1_warped_filter +
                             ((sy + 7 * gamma) >> WARPEDDIFF_PREC_BITS)));

  d0 = vtrnq_s32(vreinterpretq_s32_s16(f0), vreinterpretq_s32_s16(f2));
  d1 = vtrnq_s32(vreinterpretq_s32_s16(f4), vreinterpretq_s32_s16(f6));
  d2 = vtrnq_s32(vreinterpretq_s32_s16(f1), vreinterpretq_s32_s16(f3));
  d3 = vtrnq_s32(vreinterpretq_s32_s16(f5), vreinterpretq_s32_s16(f7));

  // row:0,1 even_col:0,2
  src_0 = vget_low_s16(vreinterpretq_s16_s32(c0.val[0]));
  fltr_0 = vget_low_s16(vreinterpretq_s16_s32(d0.val[0]));
  res_0 = vmull_s16(src_0, fltr_0);

  // row:0,1,2,3 even_col:0,2
  src_0 = vget_low_s16(vreinterpretq_s16_s32(c1.val[0]));
  fltr_0 = vget_low_s16(vreinterpretq_s16_s32(d0.val[1]));
  res_0 = vmlal_s16(res_0, src_0, fltr_0);
  res_0_im = vpadd_s32(vget_low_s32(res_0), vget_high_s32(res_0));

  // row:0,1 even_col:4,6
  src_1 = vget_low_s16(vreinterpretq_s16_s32(c0.val[1]));
  fltr_1 = vget_low_s16(vreinterpretq_s16_s32(d1.val[0]));
  res_1 = vmull_s16(src_1, fltr_1);

  // row:0,1,2,3 even_col:4,6
  src_1 = vget_low_s16(vreinterpretq_s16_s32(c1.val[1]));
  fltr_1 = vget_low_s16(vreinterpretq_s16_s32(d1.val[1]));
  res_1 = vmlal_s16(res_1, src_1, fltr_1);
  res_1_im = vpadd_s32(vget_low_s32(res_1), vget_high_s32(res_1));

  // row:0,1,2,3 even_col:0,2,4,6
  im_res_0 = vcombine_s32(res_0_im, res_1_im);

  // row:4,5 even_col:0,2
  src_0 = vget_low_s16(vreinterpretq_s16_s32(c2.val[0]));
  fltr_0 = vget_high_s16(vreinterpretq_s16_s32(d0.val[0]));
  res_0 = vmull_s16(src_0, fltr_0);

  // row:4,5,6,7 even_col:0,2
  src_0 = vget_low_s16(vreinterpretq_s16_s32(c3.val[0]));
  fltr_0 = vget_high_s16(vreinterpretq_s16_s32(d0.val[1]));
  res_0 = vmlal_s16(res_0, src_0, fltr_0);
  res_0_im = vpadd_s32(vget_low_s32(res_0), vget_high_s32(res_0));

  // row:4,5 even_col:4,6
  src_1 = vget_low_s16(vreinterpretq_s16_s32(c2.val[1]));
  fltr_1 = vget_high_s16(vreinterpretq_s16_s32(d1.val[0]));
  res_1 = vmull_s16(src_1, fltr_1);

  // row:4,5,6,7 even_col:4,6
  src_1 = vget_low_s16(vreinterpretq_s16_s32(c3.val[1]));
  fltr_1 = vget_high_s16(vreinterpretq_s16_s32(d1.val[1]));
  res_1 = vmlal_s16(res_1, src_1, fltr_1);
  res_1_im = vpadd_s32(vget_low_s32(res_1), vget_high_s32(res_1));

  // row:4,5,6,7 even_col:0,2,4,6
  im_res_1 = vcombine_s32(res_0_im, res_1_im);

  // row:0-7 even_col:0,2,4,6
  res_even = vaddq_s32(im_res_0, im_res_1);

  // row:0,1 odd_col:1,3
  src_0 = vget_high_s16(vreinterpretq_s16_s32(c0.val[0]));
  fltr_0 = vget_low_s16(vreinterpretq_s16_s32(d2.val[0]));
  res_0 = vmull_s16(src_0, fltr_0);

  // row:0,1,2,3 odd_col:1,3
  src_0 = vget_high_s16(vreinterpretq_s16_s32(c1.val[0]));
  fltr_0 = vget_low_s16(vreinterpretq_s16_s32(d2.val[1]));
  res_0 = vmlal_s16(res_0, src_0, fltr_0);
  res_0_im = vpadd_s32(vget_low_s32(res_0), vget_high_s32(res_0));

  // row:0,1 odd_col:5,7
  src_1 = vget_high_s16(vreinterpretq_s16_s32(c0.val[1]));
  fltr_1 = vget_low_s16(vreinterpretq_s16_s32(d3.val[0]));
  res_1 = vmull_s16(src_1, fltr_1);

  // row:0,1,2,3 odd_col:5,7
  src_1 = vget_high_s16(vreinterpretq_s16_s32(c1.val[1]));
  fltr_1 = vget_low_s16(vreinterpretq_s16_s32(d3.val[1]));
  res_1 = vmlal_s16(res_1, src_1, fltr_1);
  res_1_im = vpadd_s32(vget_low_s32(res_1), vget_high_s32(res_1));

  // row:0,1,2,3 odd_col:1,3,5,7
  im_res_0 = vcombine_s32(res_0_im, res_1_im);

  // row:4,5 odd_col:1,3
  src_0 = vget_high_s16(vreinterpretq_s16_s32(c2.val[0]));
  fltr_0 = vget_high_s16(vreinterpretq_s16_s32(d2.val[0]));
  res_0 = vmull_s16(src_0, fltr_0);

  // row:4,5,6,7 odd_col:1,3
  src_0 = vget_high_s16(vreinterpretq_s16_s32(c3.val[0]));
  fltr_0 = vget_high_s16(vreinterpretq_s16_s32(d2.val[1]));
  res_0 = vmlal_s16(res_0, src_0, fltr_0);
  res_0_im = vpadd_s32(vget_low_s32(res_0), vget_high_s32(res_0));

  // row:4,5 odd_col:5,7
  src_1 = vget_high_s16(vreinterpretq_s16_s32(c2.val[1]));
  fltr_1 = vget_high_s16(vreinterpretq_s16_s32(d3.val[0]));
  res_1 = vmull_s16(src_1, fltr_1);

  // row:4,5,6,7 odd_col:5,7
  src_1 = vget_high_s16(vreinterpretq_s16_s32(c3.val[1]));
  fltr_1 = vget_high_s16(vreinterpretq_s16_s32(d3.val[1]));
  res_1 = vmlal_s16(res_1, src_1, fltr_1);
  res_1_im = vpadd_s32(vget_low_s32(res_1), vget_high_s32(res_1));

  // row:4,5,6,7 odd_col:1,3,5,7
  im_res_1 = vcombine_s32(res_0_im, res_1_im);

  // row:0-7 odd_col:1,3,5,7
  res_odd = vaddq_s32(im_res_0, im_res_1);

  // reordering as 0 1 2 3 | 4 5 6 7
  c0 = vtrnq_s32(res_even, res_odd);

  // Final store
  *res_low = vcombine_s32(vget_low_s32(c0.val[0]), vget_low_s32(c0.val[1]));
  *res_high = vcombine_s32(vget_high_s32(c0.val[0]), vget_high_s32(c0.val[1]));
}

void av1_warp_affine_neon(const int32_t *mat, const uint8_t *ref, int width,
                          int height, int stride, uint8_t *pred, int p_col,
                          int p_row, int p_width, int p_height, int p_stride,
                          int subsampling_x, int subsampling_y,
                          ConvolveParams *conv_params, int16_t alpha,
                          int16_t beta, int16_t gamma, int16_t delta) {
  int16x8_t tmp[15];
  const int bd = 8;
  const int w0 = conv_params->fwd_offset;
  const int w1 = conv_params->bck_offset;
  const int32x4_t fwd = vdupq_n_s32((int32_t)w0);
  const int32x4_t bwd = vdupq_n_s32((int32_t)w1);
  const int16x8_t sub_constant = vdupq_n_s16((1 << (bd - 1)) + (1 << bd));

  int limit = 0;
  uint8x16_t vec_dup, mask_val;
  int32x4_t res_lo, res_hi;
  int16x8_t result_final;
  uint8x16_t src_1;
  static const uint8_t k0To15[16] = { 0, 1, 2,  3,  4,  5,  6,  7,
                                      8, 9, 10, 11, 12, 13, 14, 15 };
  uint8x16_t indx_vec = vld1q_u8(k0To15);
  uint8x16_t cmp_vec;

  const int reduce_bits_horiz = ROUND0_BITS;
  const int reduce_bits_vert = conv_params->is_compound
                                   ? conv_params->round_1
                                   : 2 * FILTER_BITS - reduce_bits_horiz;
  const int32x4_t shift_vert = vdupq_n_s32(-(int32_t)reduce_bits_vert);

  assert(IMPLIES(conv_params->is_compound, conv_params->dst != NULL));

  const int offset_bits_vert = bd + 2 * FILTER_BITS - reduce_bits_horiz;
  int32x4_t add_const_vert = vdupq_n_s32((int32_t)(1 << offset_bits_vert));
  const int round_bits =
      2 * FILTER_BITS - conv_params->round_0 - conv_params->round_1;
  const int16x4_t round_bits_vec = vdup_n_s16(-(int16_t)round_bits);
  const int offset_bits = bd + 2 * FILTER_BITS - conv_params->round_0;
  const int16x4_t res_sub_const =
      vdup_n_s16(-((1 << (offset_bits - conv_params->round_1)) +
                   (1 << (offset_bits - conv_params->round_1 - 1))));
  int k;

  assert(IMPLIES(conv_params->do_average, conv_params->is_compound));

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

      int32_t ix4 = (int32_t)(x4 >> WARPEDMODEL_PREC_BITS);
      int32_t sx4 = x4 & ((1 << WARPEDMODEL_PREC_BITS) - 1);
      int32_t iy4 = (int32_t)(y4 >> WARPEDMODEL_PREC_BITS);
      int32_t sy4 = y4 & ((1 << WARPEDMODEL_PREC_BITS) - 1);

      sx4 += alpha * (-4) + beta * (-4) + (1 << (WARPEDDIFF_PREC_BITS - 1)) +
             (WARPEDPIXEL_PREC_SHIFTS << WARPEDDIFF_PREC_BITS);
      sy4 += gamma * (-4) + delta * (-4) + (1 << (WARPEDDIFF_PREC_BITS - 1)) +
             (WARPEDPIXEL_PREC_SHIFTS << WARPEDDIFF_PREC_BITS);

      sx4 &= ~((1 << WARP_PARAM_REDUCE_BITS) - 1);
      sy4 &= ~((1 << WARP_PARAM_REDUCE_BITS) - 1);
      // horizontal
      if (ix4 <= -7) {
        for (k = -7; k < AOMMIN(8, p_height - i); ++k) {
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
        for (k = -7; k < AOMMIN(8, p_height - i); ++k) {
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

        for (k = -7; k < AOMMIN(8, p_height - i); ++k) {
          int iy = iy4 + k;
          if (iy < 0)
            iy = 0;
          else if (iy > height - 1)
            iy = height - 1;
          int sx = sx4 + beta * (k + 4);

          const uint8_t *src = ref + iy * stride + ix4 - 7;
          src_1 = vld1q_u8(src);

          if (out_of_boundary_left >= 0) {
            limit = out_of_boundary_left + 1;
            cmp_vec = vdupq_n_u8(out_of_boundary_left);
            vec_dup = vdupq_n_u8(*(src + limit));
            mask_val = vcleq_u8(indx_vec, cmp_vec);
            src_1 = vbslq_u8(mask_val, vec_dup, src_1);
          }
          if (out_of_boundary_right >= 0) {
            limit = 15 - (out_of_boundary_right + 1);
            cmp_vec = vdupq_n_u8(15 - out_of_boundary_right);
            vec_dup = vdupq_n_u8(*(src + limit));
            mask_val = vcgeq_u8(indx_vec, cmp_vec);
            src_1 = vbslq_u8(mask_val, vec_dup, src_1);
          }
          horizontal_filter_neon(src_1, tmp, sx, alpha, k);
        }
      } else {
        for (k = -7; k < AOMMIN(8, p_height - i); ++k) {
          int iy = iy4 + k;
          if (iy < 0)
            iy = 0;
          else if (iy > height - 1)
            iy = height - 1;
          int sx = sx4 + beta * (k + 4);

          const uint8_t *src = ref + iy * stride + ix4 - 7;
          src_1 = vld1q_u8(src);
          horizontal_filter_neon(src_1, tmp, sx, alpha, k);
        }
      }

      // vertical
      for (k = -4; k < AOMMIN(4, p_height - i - 4); ++k) {
        int sy = sy4 + delta * (k + 4);

        const int16x8_t *v_src = tmp + (k + 4);

        vertical_filter_neon(v_src, &res_lo, &res_hi, sy, gamma);

        res_lo = vaddq_s32(res_lo, add_const_vert);
        res_hi = vaddq_s32(res_hi, add_const_vert);

        if (conv_params->is_compound) {
          uint16_t *const p =
              (uint16_t *)&conv_params
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

              vst1_lane_u32((uint32_t *)dst8_4, vreinterpret_u32_u8(res_8_high),
                            0);
            } else {
              uint16x4_t res_u16_high = vqmovun_s32(res_hi);
              vst1_u16(p4, res_u16_high);
            }
          }
        } else {
          res_lo = vrshlq_s32(res_lo, shift_vert);
          res_hi = vrshlq_s32(res_hi, shift_vert);

          result_final = vcombine_s16(vmovn_s32(res_lo), vmovn_s32(res_hi));
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
  }
}
