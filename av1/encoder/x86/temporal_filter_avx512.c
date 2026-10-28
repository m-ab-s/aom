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

#if !defined(AVX512_EMU_H_) && defined(__GNUC__) && !defined(__clang__)
#pragma GCC target("avx2,avx512f,avx512cd,avx512bw,avx512dq,avx512vl")
#endif

#include <assert.h>
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"
#include "aom_dsp/mathutils.h"
#include "aom_ports/mem.h"
#include "av1/encoder/encoder.h"
#include "av1/encoder/temporal_filter.h"

#define SSE_STRIDE_AVX512 (BW + 4)

double av1_estimate_noise_from_single_plane_avx512(const uint8_t *src,
                                                   int height, int width,
                                                   int stride,
                                                   int edge_thresh) {
  int count = 0;
  int64_t accum = 0;
  const int w32 = (width - 1) & ~0x1f;
  const __m512i edge_threshold = _mm512_set1_epi16((int16_t)edge_thresh);
  const __m512i one16 = _mm512_set1_epi16(1);
  __m512i sum_accumulator = _mm512_setzero_si512();

  for (int w = 1; w < w32; w += 32) {
    int h = 1;
    const int start_idx = h * stride + w;
    const int stride_0 = start_idx - stride;

    const __m512i A = _mm512_cvtepu8_epi16(
        _mm256_loadu_si256((const __m256i *)(&src[stride_0 - 1])));
    const __m512i C = _mm512_cvtepu8_epi16(
        _mm256_loadu_si256((const __m256i *)(&src[stride_0 + 1])));
    const __m512i D = _mm512_cvtepu8_epi16(
        _mm256_loadu_si256((const __m256i *)(&src[start_idx - 1])));
    const __m512i F = _mm512_cvtepu8_epi16(
        _mm256_loadu_si256((const __m256i *)(&src[start_idx + 1])));
    __m512i B = _mm512_cvtepu8_epi16(
        _mm256_loadu_si256((const __m256i *)(&src[stride_0])));
    __m512i E = _mm512_cvtepu8_epi16(
        _mm256_loadu_si256((const __m256i *)(&src[start_idx])));

    __m512i sub_AC = _mm512_sub_epi16(A, C);
    __m512i sum_AC = _mm512_add_epi16(A, C);
    __m512i sub_DF = _mm512_sub_epi16(D, F);
    __m512i sum_DF = _mm512_add_epi16(D, F);

    for (; h < height - 1; h++) {
      const int k = h * stride + w;
      const __m512i G = _mm512_cvtepu8_epi16(
          _mm256_loadu_si256((const __m256i *)(&src[k + stride - 1])));
      const __m512i H = _mm512_cvtepu8_epi16(
          _mm256_loadu_si256((const __m256i *)(&src[k + stride])));
      const __m512i I = _mm512_cvtepu8_epi16(
          _mm256_loadu_si256((const __m256i *)(&src[k + stride + 1])));

      const __m512i sub_GI = _mm512_sub_epi16(G, I);
      const __m512i sum_GI = _mm512_add_epi16(G, I);
      const __m512i sub_BH = _mm512_sub_epi16(B, H);

      const __m512i gx = _mm512_abs_epi16(_mm512_add_epi16(
          _mm512_add_epi16(sub_AC, sub_GI), _mm512_slli_epi16(sub_DF, 1)));
      const __m512i gy = _mm512_abs_epi16(_mm512_add_epi16(
          _mm512_sub_epi16(sum_AC, sum_GI), _mm512_slli_epi16(sub_BH, 1)));
      const __m512i ga = _mm512_add_epi16(gx, gy);

      const __mmask32 mask = _mm512_cmpgt_epi16_mask(edge_threshold, ga);
      if (mask != 0) {
        const __m512i sum_BH = _mm512_add_epi16(B, H);
        const __m512i sum_DFBH =
            _mm512_slli_epi16(_mm512_add_epi16(sum_DF, sum_BH), 1);
        const __m512i sum_ACGI = _mm512_add_epi16(sum_AC, sum_GI);
        const __m512i v = _mm512_add_epi16(
            _mm512_sub_epi16(_mm512_slli_epi16(E, 2), sum_DFBH), sum_ACGI);
        const __m512i var = _mm512_maskz_abs_epi16(mask, v);
        sum_accumulator =
            _mm512_add_epi32(sum_accumulator, _mm512_madd_epi16(var, one16));
        count += __builtin_popcount((uint32_t)mask);
      }

      sub_AC = sub_DF;
      sub_DF = sub_GI;
      sum_AC = sum_DF;
      sum_DF = sum_GI;
      B = E;
      E = H;
    }
  }

  for (int h = 1; h < height - 1; ++h) {
    for (int w = w32 + 1; w < width - 1; ++w) {
      const int k = h * stride + w;
      const int g_x = (src[k - stride - 1] - src[k - stride + 1]) +
                      (src[k + stride - 1] - src[k + stride + 1]) +
                      2 * (src[k - 1] - src[k + 1]);
      const int g_y = (src[k - stride - 1] - src[k + stride - 1]) +
                      (src[k - stride + 1] - src[k + stride + 1]) +
                      2 * (src[k - stride] - src[k + stride]);
      const int ga = abs(g_x) + abs(g_y);
      if (ga < edge_thresh) {
        const int v =
            4 * src[k] -
            2 * (src[k - 1] + src[k + 1] + src[k - stride] + src[k + stride]) +
            (src[k - stride - 1] + src[k - stride + 1] + src[k + stride - 1] +
             src[k + stride + 1]);
        accum += abs(v);
        ++count;
      }
    }
  }

  accum += _mm512_reduce_add_epi32(sum_accumulator);
  return (count < 16) ? -1.0 : (double)accum / (6 * count) * SQRT_PI_BY_2;
}

#if CONFIG_AV1_HIGHBITDEPTH
double av1_highbd_estimate_noise_from_single_plane_avx512(const uint16_t *src16,
                                                          int height, int width,
                                                          int stride,
                                                          int bit_depth,
                                                          int edge_thresh) {
  int count = 0;
  int64_t accum = 0;
  const int w16 = (width - 2) & ~0x0f;
  const int shift = bit_depth - 8;
  const int round_val = shift > 0 ? (1 << (shift - 1)) : 0;
  const __m512i v_round = _mm512_set1_epi32(round_val);
  const __m512i v_edge_thresh = _mm512_set1_epi32(edge_thresh);
  __m512i sum_accumulator = _mm512_setzero_si512();

  for (int w = 1; w < w16; w += 16) {
    int h = 1;
    const int start_idx = h * stride + w;
    const int stride_0 = start_idx - stride;

    const __m512i A = _mm512_cvtepu16_epi32(
        _mm256_loadu_si256((const __m256i *)(&src16[stride_0 - 1])));
    const __m512i C = _mm512_cvtepu16_epi32(
        _mm256_loadu_si256((const __m256i *)(&src16[stride_0 + 1])));
    const __m512i D = _mm512_cvtepu16_epi32(
        _mm256_loadu_si256((const __m256i *)(&src16[start_idx - 1])));
    const __m512i F = _mm512_cvtepu16_epi32(
        _mm256_loadu_si256((const __m256i *)(&src16[start_idx + 1])));
    __m512i B = _mm512_cvtepu16_epi32(
        _mm256_loadu_si256((const __m256i *)(&src16[stride_0])));
    __m512i E = _mm512_cvtepu16_epi32(
        _mm256_loadu_si256((const __m256i *)(&src16[start_idx])));

    __m512i sub_AC = _mm512_sub_epi32(A, C);
    __m512i sum_AC = _mm512_add_epi32(A, C);
    __m512i sub_DF = _mm512_sub_epi32(D, F);
    __m512i sum_DF = _mm512_add_epi32(D, F);

    for (; h < height - 1; h++) {
      const int k = h * stride + w;
      const __m512i G = _mm512_cvtepu16_epi32(
          _mm256_loadu_si256((const __m256i *)(&src16[k + stride - 1])));
      const __m512i H = _mm512_cvtepu16_epi32(
          _mm256_loadu_si256((const __m256i *)(&src16[k + stride])));
      const __m512i I = _mm512_cvtepu16_epi32(
          _mm256_loadu_si256((const __m256i *)(&src16[k + stride + 1])));

      const __m512i sub_GI = _mm512_sub_epi32(G, I);
      const __m512i sum_GI = _mm512_add_epi32(G, I);
      const __m512i sub_BH = _mm512_sub_epi32(B, H);

      const __m512i gx = _mm512_abs_epi32(_mm512_add_epi32(
          _mm512_add_epi32(sub_AC, sub_GI), _mm512_slli_epi32(sub_DF, 1)));
      const __m512i gy = _mm512_abs_epi32(_mm512_add_epi32(
          _mm512_sub_epi32(sum_AC, sum_GI), _mm512_slli_epi32(sub_BH, 1)));
      const __m512i ga =
          _mm512_srai_epi32(_mm512_add_epi32(_mm512_add_epi32(gx, gy), v_round),
                            (unsigned int)shift);

      const __mmask16 mask = _mm512_cmpgt_epi32_mask(v_edge_thresh, ga);
      if (mask != 0) {
        const __m512i sum_BH = _mm512_add_epi32(B, H);
        const __m512i sum_DFBH =
            _mm512_slli_epi32(_mm512_add_epi32(sum_DF, sum_BH), 1);
        const __m512i sum_ACGI = _mm512_add_epi32(sum_AC, sum_GI);
        const __m512i v = _mm512_add_epi32(
            _mm512_sub_epi32(_mm512_slli_epi32(E, 2), sum_DFBH), sum_ACGI);
        const __m512i abs_v = _mm512_abs_epi32(v);
        const __m512i scaled_v = _mm512_srai_epi32(
            _mm512_add_epi32(abs_v, v_round), (unsigned int)shift);
        sum_accumulator = _mm512_mask_add_epi32(sum_accumulator, mask,
                                                sum_accumulator, scaled_v);
        count += __builtin_popcount((uint32_t)mask);
      }

      sub_AC = sub_DF;
      sub_DF = sub_GI;
      sum_AC = sum_DF;
      sum_DF = sum_GI;
      B = E;
      E = H;
    }
  }

  for (int i = 1; i < height - 1; ++i) {
    for (int j = w16 + 1; j < width - 1; ++j) {
      const int k = i * stride + j;
      const int Gx = (src16[k - stride - 1] - src16[k - stride + 1]) +
                     (src16[k + stride - 1] - src16[k + stride + 1]) +
                     2 * (src16[k - 1] - src16[k + 1]);
      const int Gy = (src16[k - stride - 1] - src16[k + stride - 1]) +
                     (src16[k - stride + 1] - src16[k + stride + 1]) +
                     2 * (src16[k - stride] - src16[k + stride]);
      const int Ga = ROUND_POWER_OF_TWO(abs(Gx) + abs(Gy), shift);
      if (Ga < edge_thresh) {
        const int v = 4 * src16[k] -
                      2 * (src16[k - 1] + src16[k + 1] + src16[k - stride] +
                           src16[k + stride]) +
                      (src16[k - stride - 1] + src16[k - stride + 1] +
                       src16[k + stride - 1] + src16[k + stride + 1]);
        accum += ROUND_POWER_OF_TWO(abs(v), shift);
        ++count;
      }
    }
  }

  accum += _mm512_reduce_add_epi32(sum_accumulator);
  return (count < 16) ? -1.0 : (double)accum / (6 * count) * SQRT_PI_BY_2;
}
#endif  // CONFIG_AV1_HIGHBITDEPTH

static AOM_FORCE_INLINE void get_squared_error_avx512(
    const uint8_t *frame1, const unsigned int stride, const uint8_t *frame2,
    const unsigned int stride2, const int block_width, const int block_height,
    uint16_t *frame_sse, const unsigned int sse_stride) {
  const uint8_t *src1 = frame1;
  const uint8_t *src2 = frame2;
  uint16_t *dst = frame_sse + 2;
  for (int i = 0; i < block_height; i++) {
    for (int j = 0; j < block_width; j += 32) {
      const __m256i vsrc1 = _mm256_loadu_si256((const __m256i *)(src1 + j));
      const __m256i vsrc2 = _mm256_loadu_si256((const __m256i *)(src2 + j));
      const __m512i s1 = _mm512_cvtepu8_epi16(vsrc1);
      const __m512i s2 = _mm512_cvtepu8_epi16(vsrc2);
      const __m512i diff = _mm512_sub_epi16(s1, s2);
      const __m512i res = _mm512_mullo_epi16(diff, diff);
      _mm512_storeu_si512((__m512i *)(dst + j), res);
    }
    dst[-2] = dst[-1] = dst[0];
    dst[block_width] = dst[block_width + 1] = dst[block_width - 1];
    src1 += stride;
    src2 += stride2;
    dst += sse_stride;
  }
}

static AOM_FORCE_INLINE __m512i load_and_sum5_row_avx512(const uint16_t *src) {
  const __m512i c0 =
      _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(src - 2)));
  const __m512i c1 =
      _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(src - 1)));
  const __m512i c2 =
      _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(src)));
  const __m512i c3 =
      _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(src + 1)));
  const __m512i c4 =
      _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(src + 2)));
  return _mm512_add_epi32(_mm512_add_epi32(c0, c1),
                          _mm512_add_epi32(_mm512_add_epi32(c2, c3), c4));
}

static inline __m256 approx_exp_avx512(__m256 y) {
#define A ((1 << 23) / 0.69314718056f)
#define B 127
#define C 60801
  const __m256 multiplier = _mm256_set1_ps(A);
  const __m256i offset = _mm256_set1_epi32(B * (1 << 23) - C);

  y = _mm256_mul_ps(y, multiplier);
  y = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_cvttps_epi32(y), offset));
  return y;
#undef A
#undef B
#undef C
}

// Bit-exact 512-bit AVX-512 double-precision implementation of exp(-x) for 8
// lanes simultaneously on x in [0, 7].
static inline __m512d exact_exp_neg_pd_avx512(__m512d x) {
  x = _mm512_min_pd(x, _mm512_set1_pd(7.0));
  const __m512d inv_ln2 = _mm512_set1_pd(1.44269504088896340736);
  const __m512d ln2_hi = _mm512_set1_pd(0.6931471805596629565116);
  const __m512d ln2_lo = _mm512_set1_pd(0.2823529056303157712259e-12);
  const __m512d k = _mm512_roundscale_pd(
      _mm512_mul_pd(x, inv_ln2), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  const __m512d r = _mm512_add_pd(_mm512_sub_pd(_mm512_mul_pd(k, ln2_hi), x),
                                  _mm512_mul_pd(k, ln2_lo));
  const __m512d c1 = _mm512_set1_pd(1.0);
  const __m512d c2 = _mm512_set1_pd(0.5);
  const __m512d c3 = _mm512_set1_pd(1.0 / 6.0);
  const __m512d c4 = _mm512_set1_pd(1.0 / 24.0);
  const __m512d c5 = _mm512_set1_pd(1.0 / 120.0);
  const __m512d c6 = _mm512_set1_pd(1.0 / 720.0);
  const __m512d c7 = _mm512_set1_pd(1.0 / 5040.0);
  __m512d p = _mm512_add_pd(_mm512_mul_pd(c7, r), c6);
  p = _mm512_add_pd(_mm512_mul_pd(p, r), c5);
  p = _mm512_add_pd(_mm512_mul_pd(p, r), c4);
  p = _mm512_add_pd(_mm512_mul_pd(p, r), c3);
  p = _mm512_add_pd(_mm512_mul_pd(p, r), c2);
  p = _mm512_add_pd(_mm512_mul_pd(p, r), c1);
  p = _mm512_add_pd(_mm512_mul_pd(p, r), c1);
  const __m256i ki = _mm512_cvtpd_epi32(k);
  const __m512i ki64 =
      _mm512_cvtepi32_epi64(_mm256_sub_epi32(_mm256_set1_epi32(1023), ki));
  const __m512d pow2 = _mm512_castsi512_pd(_mm512_slli_epi64(ki64, 52));
  return _mm512_mul_pd(p, pow2);
}

static void apply_temporal_filter_avx512(
    const uint8_t *frame1, const unsigned int stride, const uint8_t *frame2,
    const unsigned int stride2, const int block_width, const int block_height,
    const int *subblock_mses, unsigned int *accumulator, uint16_t *count,
    uint16_t *frame_sse, uint32_t *luma_sse_sum,
    const double inv_num_ref_pixels, const double decay_factor,
    const double inv_factor, const double weight_factor, double *d_factor,
    int tf_wgt_calc_lvl) {
  assert(((block_width == 64) || (block_width == 32)) &&
         ((block_height == 64) || (block_height == 32)));

  DECLARE_ALIGNED(64, uint32_t, acc_5x5_sse[BH][BW]);

  get_squared_error_avx512(frame1, stride, frame2, stride2, block_width,
                           block_height, frame_sse, SSE_STRIDE_AVX512);

  __m512i vsrc[5];

  for (int col = 0; col < block_width; col += 16) {
    const uint16_t *src = frame_sse + 2 + col;

    for (int i = 2; i < 5; i++) {
      vsrc[i] = load_and_sum5_row_avx512(src);
      src += SSE_STRIDE_AVX512;
    }

    vsrc[0] = vsrc[2];
    vsrc[1] = vsrc[2];

    for (int row = 0; row < block_height; row++) {
      const __m512i vsum01 = _mm512_add_epi32(vsrc[0], vsrc[1]);
      const __m512i vsum23 = _mm512_add_epi32(vsrc[2], vsrc[3]);
      const __m512i vsum =
          _mm512_add_epi32(_mm512_add_epi32(vsum01, vsum23), vsrc[4]);

      _mm512_storeu_si512((__m512i *)&acc_5x5_sse[row][col], vsum);

      vsrc[0] = vsrc[1];
      vsrc[1] = vsrc[2];
      vsrc[2] = vsrc[3];
      vsrc[3] = vsrc[4];

      if (row <= block_height - 4) {
        vsrc[4] = load_and_sum5_row_avx512(src);
        src += SSE_STRIDE_AVX512;
      } else {
        vsrc[4] = vsrc[3];
      }
    }
  }

  __m512d subblock_mses_reg[NUM_16X16];
  __m512d d_factor_mul_n_decay_qr_invs[NUM_16X16];
  for (int idx = 0; idx < NUM_16X16; idx++) {
    subblock_mses_reg[idx] = _mm512_set1_pd(subblock_mses[idx] * inv_factor);
    d_factor_mul_n_decay_qr_invs[idx] =
        _mm512_set1_pd(d_factor[idx] * decay_factor);
  }

  const __m256 zero = _mm256_set1_ps(0.0f);
  const __m256 point_five = _mm256_set1_ps(0.5f);
  const __m256 seven = _mm256_set1_ps(7.0f);
  const __m512d inv_num_ref_pixel_512 = _mm512_set1_pd(inv_num_ref_pixels);
  const __m512d weight_factor_512 = _mm512_set1_pd(weight_factor);
  const __m256 tf_weight_scale = _mm256_set1_ps((float)TF_WEIGHT_SCALE);
  const __m512d tf_weight_scale_pd = _mm512_set1_pd((double)TF_WEIGHT_SCALE);

  for (int i = 0; i < block_height; i++) {
    const int y32_blk_raster_offset = (i >= (block_height >> 1)) << 1;
    const int y16_blk_raster_offset =
        ((i % (block_height >> 1)) >= (block_height >> 2)) << 1;
    uint32_t *luma_sse_sum_temp = luma_sse_sum + i * BW;
    for (int j = 0; j < block_width; j += 8) {
      const __m256i acc_sse =
          _mm256_loadu_si256((const __m256i *)(acc_5x5_sse[i] + j));
      const __m256i luma_sse =
          _mm256_loadu_si256((const __m256i *)(luma_sse_sum_temp + j));
      const __m256i diff_sse = _mm256_add_epi32(acc_sse, luma_sse);
      const __m512d diff_sse_pd = _mm512_cvtepi32_pd(diff_sse);
      const __m512d window_error =
          _mm512_mul_pd(diff_sse_pd, inv_num_ref_pixel_512);

      const int x32_blk_raster_offset = (j >= (block_width >> 1));
      const int x16_blk_raster_offset =
          ((j % (block_width >> 1)) >= (block_width >> 2));
      const int subblock_idx =
          ((y32_blk_raster_offset + x32_blk_raster_offset) << 2) +
          (y16_blk_raster_offset + x16_blk_raster_offset);

      const __m512d combined_error =
          _mm512_add_pd(_mm512_mul_pd(window_error, weight_factor_512),
                        subblock_mses_reg[subblock_idx]);
      const __m512d scaled_error = _mm512_mul_pd(
          combined_error, d_factor_mul_n_decay_qr_invs[subblock_idx]);

      __m256i weights_in_32bit;
      if (tf_wgt_calc_lvl == 0) {
        weights_in_32bit = _mm512_cvttpd_epi32(_mm512_mul_pd(
            exact_exp_neg_pd_avx512(scaled_error), tf_weight_scale_pd));
      } else {
        const __m256 scaled_error_ps = _mm512_cvtpd_ps(scaled_error);
        const __m256 scaled_diff_ps = _mm256_min_ps(scaled_error_ps, seven);
        const __m256 minus_scaled_diff_ps = _mm256_sub_ps(zero, scaled_diff_ps);
        const __m256 exp_result = approx_exp_avx512(minus_scaled_diff_ps);
        const __m256 scale_weight_exp_result =
            _mm256_mul_ps(exp_result, tf_weight_scale);
        const __m256 round_result =
            _mm256_add_ps(scale_weight_exp_result, point_five);
        weights_in_32bit = _mm256_cvttps_epi32(round_result);
      }

      const __m128i weights_in_16bit =
          _mm_packus_epi32(_mm256_castsi256_si128(weights_in_32bit),
                           _mm256_extracti128_si256(weights_in_32bit, 0x1));

      const int stride_idx = i * stride2 + j;
      const __m128i count_array =
          _mm_loadu_si128((const __m128i *)(count + stride_idx));
      _mm_storeu_si128((__m128i *)(count + stride_idx),
                       _mm_add_epi16(count_array, weights_in_16bit));

      const __m256i accumulator_array =
          _mm256_loadu_si256((const __m256i *)(accumulator + stride_idx));
      const __m128i pred_values =
          _mm_loadl_epi64((const __m128i *)(frame2 + stride_idx));
      const __m256i pred_values_u32 = _mm256_cvtepu8_epi32(pred_values);
      const __m256i mull_frame2_weight_u32 =
          _mm256_mullo_epi32(pred_values_u32, weights_in_32bit);
      _mm256_storeu_si256(
          (__m256i *)(accumulator + stride_idx),
          _mm256_add_epi32(accumulator_array, mull_frame2_weight_u32));
    }
  }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((force_align_arg_pointer))
#endif
void av1_apply_temporal_filter_avx512(
    const YV12_BUFFER_CONFIG *frame_to_filter, const MACROBLOCKD *mbd,
    const BLOCK_SIZE block_size, const int mb_row, const int mb_col,
    const int num_planes, const double *noise_levels, const MV *subblock_mvs,
    const int *subblock_mses, const int q_factor, const int filter_strength,
    int tf_wgt_calc_lvl, const uint8_t *pred, uint32_t *accum,
    uint16_t *count) {
  const int is_high_bitdepth = frame_to_filter->flags & YV12_FLAG_HIGHBITDEPTH;
  assert(block_size == BLOCK_64X64 && "Only support 64x64 block with avx512!");
  assert(TF_WINDOW_LENGTH == 5 && "Only support window length 5 with avx512!");
  assert(!is_high_bitdepth && "Only support low bit-depth with avx512!");
  assert(num_planes >= 1 && num_planes <= MAX_MB_PLANE);
  (void)is_high_bitdepth;

  const int mb_height = block_size_high[block_size];
  const int mb_width = block_size_wide[block_size];
  const int frame_height = frame_to_filter->y_crop_height;
  const int frame_width = frame_to_filter->y_crop_width;
  const int min_frame_size = AOMMIN(frame_height, frame_width);
  const double inv_factor = 1.0 / ((TF_WINDOW_BLOCK_BALANCE_WEIGHT + 1) *
                                   TF_SEARCH_ERROR_NORM_WEIGHT);
  const double weight_factor =
      (double)TF_WINDOW_BLOCK_BALANCE_WEIGHT * inv_factor;
  double q_decay = pow((double)q_factor / TF_Q_DECAY_THRESHOLD, 2);
  q_decay = CLIP(q_decay, 1e-5, 1);
  if (q_factor >= TF_QINDEX_CUTOFF) {
    q_decay = 0.5 * pow((double)q_factor / 64, 2);
  }
  double s_decay = pow((double)filter_strength / TF_STRENGTH_THRESHOLD, 2);
  s_decay = CLIP(s_decay, 1e-5, 1);
  double d_factor[NUM_16X16] = { 0 };
  uint16_t *frame_sse = (uint16_t *)aom_memalign(
      64, sizeof(frame_sse[0]) * SSE_STRIDE_AVX512 * BH);
  uint32_t *luma_sse_sum =
      (uint32_t *)aom_memalign(64, sizeof(luma_sse_sum[0]) * BW * BH);
  memset(frame_sse, 0, sizeof(frame_sse[0]) * SSE_STRIDE_AVX512 * BH);
  memset(luma_sse_sum, 0, sizeof(luma_sse_sum[0]) * BW * BH);

  double distance_threshold = min_frame_size * TF_SEARCH_DISTANCE_THRESHOLD;
  distance_threshold = AOMMAX(distance_threshold, 1);
  for (int subblock_idx = 0; subblock_idx < NUM_16X16; subblock_idx++) {
    const MV mv = subblock_mvs[subblock_idx];
    const double distance = sqrt(pow(mv.row, 2) + pow(mv.col, 2));
    d_factor[subblock_idx] = distance / distance_threshold;
    d_factor[subblock_idx] = AOMMAX(d_factor[subblock_idx], 1);
  }

  int plane_offset = 0;
  for (int plane = 0; plane < num_planes; ++plane) {
    const uint32_t plane_h = mb_height >> mbd->plane[plane].subsampling_y;
    const uint32_t plane_w = mb_width >> mbd->plane[plane].subsampling_x;
    const uint32_t frame_stride = frame_to_filter->strides[plane == 0 ? 0 : 1];
    const int frame_offset = mb_row * plane_h * frame_stride + mb_col * plane_w;

    const uint8_t *ref = frame_to_filter->buffers[plane] + frame_offset;
    const int ss_x_shift =
        mbd->plane[plane].subsampling_x - mbd->plane[AOM_PLANE_Y].subsampling_x;
    const int ss_y_shift =
        mbd->plane[plane].subsampling_y - mbd->plane[AOM_PLANE_Y].subsampling_y;
    const int num_ref_pixels = TF_WINDOW_LENGTH * TF_WINDOW_LENGTH +
                               ((plane) ? (1 << (ss_x_shift + ss_y_shift)) : 0);
    const double inv_num_ref_pixels = 1.0 / num_ref_pixels;
    const double n_decay = 0.5 + log(2 * noise_levels[plane] + 5.0);
    const double decay_factor = 1 / (n_decay * q_decay * s_decay);

    if (plane == AOM_PLANE_U) {
      if (ss_x_shift == 1 && ss_y_shift == 1) {
        const __m512i mask_lo16 = _mm512_set1_epi32(0x0000FFFF);
        for (unsigned int i = 0; i < plane_h; i++) {
          const uint16_t *src_0 = &frame_sse[2 * i * SSE_STRIDE_AVX512 + 2];
          const uint16_t *src_1 =
              &frame_sse[(2 * i + 1) * SSE_STRIDE_AVX512 + 2];
          for (unsigned int j = 0; j < plane_w; j += 16) {
            const __m512i r0 =
                _mm512_loadu_si512((const __m512i *)(src_0 + j * 2));
            const __m512i r1 =
                _mm512_loadu_si512((const __m512i *)(src_1 + j * 2));
            const __m512i sum0 = _mm512_add_epi32(
                _mm512_and_si512(r0, mask_lo16), _mm512_srli_epi32(r0, 16));
            const __m512i sum1 = _mm512_add_epi32(
                _mm512_and_si512(r1, mask_lo16), _mm512_srli_epi32(r1, 16));
            const __m512i res = _mm512_add_epi32(sum0, sum1);
            _mm512_storeu_si512((__m512i *)&luma_sse_sum[i * BW + j], res);
          }
        }
      } else {
        for (unsigned int i = 0; i < plane_h; i++) {
          for (unsigned int j = 0; j < plane_w; j++) {
            for (int ii = 0; ii < (1 << ss_y_shift); ++ii) {
              for (int jj = 0; jj < (1 << ss_x_shift); ++jj) {
                const int yy = (i << ss_y_shift) + ii;
                const int xx = (j << ss_x_shift) + jj;
                luma_sse_sum[i * BW + j] +=
                    frame_sse[yy * SSE_STRIDE_AVX512 + xx + 2];
              }
            }
          }
        }
      }
    }

    apply_temporal_filter_avx512(
        ref, frame_stride, pred + plane_offset, plane_w, plane_w, plane_h,
        subblock_mses, accum + plane_offset, count + plane_offset, frame_sse,
        luma_sse_sum, inv_num_ref_pixels, decay_factor, inv_factor,
        weight_factor, d_factor, tf_wgt_calc_lvl);
    plane_offset += plane_h * plane_w;
  }
  aom_free(frame_sse);
  aom_free(luma_sse_sum);
}

#if CONFIG_AV1_HIGHBITDEPTH
static AOM_FORCE_INLINE void highbd_get_squared_error_avx512(
    const uint16_t *frame1, const unsigned int stride, const uint16_t *frame2,
    const unsigned int stride2, const int block_width, const int block_height,
    uint32_t *frame_sse, const unsigned int sse_stride) {
  const uint16_t *src1 = frame1;
  const uint16_t *src2 = frame2;
  uint32_t *dst = frame_sse + 2;
  for (int i = 0; i < block_height; i++) {
    for (int j = 0; j < block_width; j += 16) {
      const __m512i s1 = _mm512_cvtepu16_epi32(
          _mm256_loadu_si256((const __m256i *)(src1 + j)));
      const __m512i s2 = _mm512_cvtepu16_epi32(
          _mm256_loadu_si256((const __m256i *)(src2 + j)));
      const __m512i diff = _mm512_sub_epi32(s1, s2);
      const __m512i res = _mm512_mullo_epi32(diff, diff);
      _mm512_storeu_si512((__m512i *)(dst + j), res);
    }
    dst[-2] = dst[-1] = dst[0];
    dst[block_width] = dst[block_width + 1] = dst[block_width - 1];
    src1 += stride;
    src2 += stride2;
    dst += sse_stride;
  }
}

static AOM_FORCE_INLINE __m512i
highbd_load_and_sum5_row_avx512(const uint32_t *src) {
  const __m512i c0 = _mm512_loadu_si512((const __m512i *)(src - 2));
  const __m512i c1 = _mm512_loadu_si512((const __m512i *)(src - 1));
  const __m512i c2 = _mm512_loadu_si512((const __m512i *)(src));
  const __m512i c3 = _mm512_loadu_si512((const __m512i *)(src + 1));
  const __m512i c4 = _mm512_loadu_si512((const __m512i *)(src + 2));
  return _mm512_add_epi32(_mm512_add_epi32(c0, c1),
                          _mm512_add_epi32(_mm512_add_epi32(c2, c3), c4));
}

static void highbd_apply_temporal_filter_avx512(
    const uint16_t *frame1, const unsigned int stride, const uint16_t *frame2,
    const unsigned int stride2, const int block_width, const int block_height,
    const int *subblock_mses, unsigned int *accumulator, uint16_t *count,
    uint32_t *frame_sse, uint32_t *luma_sse_sum, int bd,
    const double inv_num_ref_pixels, const double decay_factor,
    const double inv_factor, const double weight_factor, double *d_factor,
    int tf_wgt_calc_lvl) {
  assert(((block_width == 64) || (block_width == 32)) &&
         ((block_height == 64) || (block_height == 32)));

  DECLARE_ALIGNED(64, uint32_t, acc_5x5_sse[BH][BW]);

  highbd_get_squared_error_avx512(frame1, stride, frame2, stride2, block_width,
                                  block_height, frame_sse, SSE_STRIDE_AVX512);

  __m512i vsrc[5];

  for (int col = 0; col < block_width; col += 16) {
    const uint32_t *src = frame_sse + 2 + col;

    for (int i = 2; i < 5; i++) {
      vsrc[i] = highbd_load_and_sum5_row_avx512(src);
      src += SSE_STRIDE_AVX512;
    }

    vsrc[0] = vsrc[2];
    vsrc[1] = vsrc[2];

    for (int row = 0; row < block_height; row++) {
      const __m512i vsum01 = _mm512_add_epi32(vsrc[0], vsrc[1]);
      const __m512i vsum23 = _mm512_add_epi32(vsrc[2], vsrc[3]);
      const __m512i vsum =
          _mm512_add_epi32(_mm512_add_epi32(vsum01, vsum23), vsrc[4]);

      _mm512_storeu_si512((__m512i *)&acc_5x5_sse[row][col], vsum);

      vsrc[0] = vsrc[1];
      vsrc[1] = vsrc[2];
      vsrc[2] = vsrc[3];
      vsrc[3] = vsrc[4];

      if (row <= block_height - 4) {
        vsrc[4] = highbd_load_and_sum5_row_avx512(src);
        src += SSE_STRIDE_AVX512;
      } else {
        vsrc[4] = vsrc[3];
      }
    }
  }

  __m512d subblock_mses_reg[NUM_16X16];
  __m512d d_factor_mul_n_decay_qr_invs[NUM_16X16];
  for (int idx = 0; idx < NUM_16X16; idx++) {
    subblock_mses_reg[idx] = _mm512_set1_pd(subblock_mses[idx] * inv_factor);
    d_factor_mul_n_decay_qr_invs[idx] =
        _mm512_set1_pd(d_factor[idx] * decay_factor);
  }

  const unsigned int bd_shift = (unsigned int)((bd - 8) * 2);
  const __m256 zero = _mm256_set1_ps(0.0f);
  const __m256 point_five = _mm256_set1_ps(0.5f);
  const __m256 seven = _mm256_set1_ps(7.0f);
  const __m512d inv_num_ref_pixel_512 = _mm512_set1_pd(inv_num_ref_pixels);
  const __m512d weight_factor_512 = _mm512_set1_pd(weight_factor);
  const __m256 tf_weight_scale = _mm256_set1_ps((float)TF_WEIGHT_SCALE);
  const __m512d tf_weight_scale_pd = _mm512_set1_pd((double)TF_WEIGHT_SCALE);

  for (int i = 0; i < block_height; i++) {
    const int y32_blk_raster_offset = (i >= (block_height >> 1)) << 1;
    const int y16_blk_raster_offset =
        ((i % (block_height >> 1)) >= (block_height >> 2)) << 1;
    uint32_t *luma_sse_sum_temp = luma_sse_sum + i * BW;
    for (int j = 0; j < block_width; j += 8) {
      const __m256i acc_sse =
          _mm256_loadu_si256((const __m256i *)(acc_5x5_sse[i] + j));
      const __m256i luma_sse =
          _mm256_loadu_si256((const __m256i *)(luma_sse_sum_temp + j));
      const __m256i diff_sse =
          _mm256_srli_epi32(_mm256_add_epi32(acc_sse, luma_sse), bd_shift);
      const __m512d diff_sse_pd = _mm512_cvtepu32_pd(diff_sse);
      const __m512d window_error =
          _mm512_mul_pd(diff_sse_pd, inv_num_ref_pixel_512);

      const int x32_blk_raster_offset = (j >= (block_width >> 1));
      const int x16_blk_raster_offset =
          ((j % (block_width >> 1)) >= (block_width >> 2));
      const int subblock_idx =
          ((y32_blk_raster_offset + x32_blk_raster_offset) << 2) +
          (y16_blk_raster_offset + x16_blk_raster_offset);

      const __m512d combined_error =
          _mm512_add_pd(_mm512_mul_pd(window_error, weight_factor_512),
                        subblock_mses_reg[subblock_idx]);
      const __m512d scaled_error = _mm512_mul_pd(
          combined_error, d_factor_mul_n_decay_qr_invs[subblock_idx]);

      __m256i weights_in_32bit;
      if (tf_wgt_calc_lvl == 0) {
        weights_in_32bit = _mm512_cvttpd_epi32(_mm512_mul_pd(
            exact_exp_neg_pd_avx512(scaled_error), tf_weight_scale_pd));
      } else {
        const __m256 scaled_error_ps = _mm512_cvtpd_ps(scaled_error);
        const __m256 scaled_diff_ps = _mm256_min_ps(scaled_error_ps, seven);
        const __m256 minus_scaled_diff_ps = _mm256_sub_ps(zero, scaled_diff_ps);
        const __m256 exp_result = approx_exp_avx512(minus_scaled_diff_ps);
        const __m256 scale_weight_exp_result =
            _mm256_mul_ps(exp_result, tf_weight_scale);
        const __m256 round_result =
            _mm256_add_ps(scale_weight_exp_result, point_five);
        weights_in_32bit = _mm256_cvttps_epi32(round_result);
      }

      const __m128i weights_in_16bit =
          _mm_packus_epi32(_mm256_castsi256_si128(weights_in_32bit),
                           _mm256_extracti128_si256(weights_in_32bit, 0x1));

      const int stride_idx = i * stride2 + j;
      const __m128i count_array =
          _mm_loadu_si128((const __m128i *)(count + stride_idx));
      _mm_storeu_si128((__m128i *)(count + stride_idx),
                       _mm_add_epi16(count_array, weights_in_16bit));

      const __m256i accumulator_array =
          _mm256_loadu_si256((const __m256i *)(accumulator + stride_idx));
      const __m128i pred_values =
          _mm_loadu_si128((const __m128i *)(frame2 + stride_idx));
      const __m256i pred_values_u32 = _mm256_cvtepu16_epi32(pred_values);
      const __m256i mull_frame2_weight_u32 =
          _mm256_mullo_epi32(pred_values_u32, weights_in_32bit);
      _mm256_storeu_si256(
          (__m256i *)(accumulator + stride_idx),
          _mm256_add_epi32(accumulator_array, mull_frame2_weight_u32));
    }
  }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((force_align_arg_pointer))
#endif
void av1_highbd_apply_temporal_filter_avx512(
    const YV12_BUFFER_CONFIG *frame_to_filter, const MACROBLOCKD *mbd,
    const BLOCK_SIZE block_size, const int mb_row, const int mb_col,
    const int num_planes, const double *noise_levels, const MV *subblock_mvs,
    const int *subblock_mses, const int q_factor, const int filter_strength,
    int tf_wgt_calc_lvl, const uint8_t *pred, uint32_t *accum,
    uint16_t *count) {
  const int is_high_bitdepth = frame_to_filter->flags & YV12_FLAG_HIGHBITDEPTH;
  assert(block_size == BLOCK_64X64 && "Only support 64x64 block with avx512!");
  assert(TF_WINDOW_LENGTH == 5 && "Only support window length 5 with avx512!");
  assert(num_planes >= 1 && num_planes <= MAX_MB_PLANE);
  (void)is_high_bitdepth;

  const int mb_height = block_size_high[block_size];
  const int mb_width = block_size_wide[block_size];
  const int frame_height = frame_to_filter->y_crop_height;
  const int frame_width = frame_to_filter->y_crop_width;
  const int min_frame_size = AOMMIN(frame_height, frame_width);
  const double inv_factor = 1.0 / ((TF_WINDOW_BLOCK_BALANCE_WEIGHT + 1) *
                                   TF_SEARCH_ERROR_NORM_WEIGHT);
  const double weight_factor =
      (double)TF_WINDOW_BLOCK_BALANCE_WEIGHT * inv_factor;
  double q_decay = pow((double)q_factor / TF_Q_DECAY_THRESHOLD, 2);
  q_decay = CLIP(q_decay, 1e-5, 1);
  if (q_factor >= TF_QINDEX_CUTOFF) {
    q_decay = 0.5 * pow((double)q_factor / 64, 2);
  }
  double s_decay = pow((double)filter_strength / TF_STRENGTH_THRESHOLD, 2);
  s_decay = CLIP(s_decay, 1e-5, 1);
  double d_factor[NUM_16X16] = { 0 };
  uint32_t *frame_sse = (uint32_t *)aom_memalign(
      64, sizeof(frame_sse[0]) * SSE_STRIDE_AVX512 * BH);
  uint32_t *luma_sse_sum =
      (uint32_t *)aom_memalign(64, sizeof(luma_sse_sum[0]) * BW * BH);
  memset(frame_sse, 0, sizeof(frame_sse[0]) * SSE_STRIDE_AVX512 * BH);
  memset(luma_sse_sum, 0, sizeof(luma_sse_sum[0]) * BW * BH);
  uint16_t *pred1 = CONVERT_TO_SHORTPTR(pred);

  double distance_threshold = min_frame_size * TF_SEARCH_DISTANCE_THRESHOLD;
  distance_threshold = AOMMAX(distance_threshold, 1);
  for (int subblock_idx = 0; subblock_idx < NUM_16X16; subblock_idx++) {
    const MV mv = subblock_mvs[subblock_idx];
    const double distance = sqrt(pow(mv.row, 2) + pow(mv.col, 2));
    d_factor[subblock_idx] = distance / distance_threshold;
    d_factor[subblock_idx] = AOMMAX(d_factor[subblock_idx], 1);
  }

  int plane_offset = 0;
  for (int plane = 0; plane < num_planes; ++plane) {
    const uint32_t plane_h = mb_height >> mbd->plane[plane].subsampling_y;
    const uint32_t plane_w = mb_width >> mbd->plane[plane].subsampling_x;
    const uint32_t frame_stride = frame_to_filter->strides[plane == 0 ? 0 : 1];
    const int frame_offset = mb_row * plane_h * frame_stride + mb_col * plane_w;

    const uint16_t *ref =
        CONVERT_TO_SHORTPTR(frame_to_filter->buffers[plane]) + frame_offset;
    const int ss_x_shift =
        mbd->plane[plane].subsampling_x - mbd->plane[AOM_PLANE_Y].subsampling_x;
    const int ss_y_shift =
        mbd->plane[plane].subsampling_y - mbd->plane[AOM_PLANE_Y].subsampling_y;
    const int num_ref_pixels = TF_WINDOW_LENGTH * TF_WINDOW_LENGTH +
                               ((plane) ? (1 << (ss_x_shift + ss_y_shift)) : 0);
    const double inv_num_ref_pixels = 1.0 / num_ref_pixels;
    const double n_decay = 0.5 + log(2 * noise_levels[plane] + 5.0);
    const double decay_factor = 1 / (n_decay * q_decay * s_decay);

    if (plane == AOM_PLANE_U) {
      if (ss_x_shift == 1 && ss_y_shift == 1) {
        const __m512i even_idx = _mm512_setr_epi32(
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        const __m512i odd_idx = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17,
                                                  19, 21, 23, 25, 27, 29, 31);
        for (unsigned int i = 0; i < plane_h; i++) {
          const uint32_t *src_0 = &frame_sse[2 * i * SSE_STRIDE_AVX512 + 2];
          const uint32_t *src_1 =
              &frame_sse[(2 * i + 1) * SSE_STRIDE_AVX512 + 2];
          for (unsigned int j = 0; j < plane_w; j += 16) {
            const __m512i r0_lo =
                _mm512_loadu_si512((const __m512i *)(src_0 + 2 * j));
            const __m512i r0_hi =
                _mm512_loadu_si512((const __m512i *)(src_0 + 2 * j + 16));
            const __m512i r1_lo =
                _mm512_loadu_si512((const __m512i *)(src_1 + 2 * j));
            const __m512i r1_hi =
                _mm512_loadu_si512((const __m512i *)(src_1 + 2 * j + 16));
            const __m512i s_lo = _mm512_add_epi32(r0_lo, r1_lo);
            const __m512i s_hi = _mm512_add_epi32(r0_hi, r1_hi);
            const __m512i evens =
                _mm512_permutex2var_epi32(s_lo, even_idx, s_hi);
            const __m512i odds = _mm512_permutex2var_epi32(s_lo, odd_idx, s_hi);
            const __m512i res = _mm512_add_epi32(evens, odds);
            _mm512_storeu_si512((__m512i *)&luma_sse_sum[i * BW + j], res);
          }
        }
      } else {
        for (unsigned int i = 0; i < plane_h; i++) {
          for (unsigned int j = 0; j < plane_w; j++) {
            for (int ii = 0; ii < (1 << ss_y_shift); ++ii) {
              for (int jj = 0; jj < (1 << ss_x_shift); ++jj) {
                const int yy = (i << ss_y_shift) + ii;
                const int xx = (j << ss_x_shift) + jj;
                luma_sse_sum[i * BW + j] +=
                    frame_sse[yy * SSE_STRIDE_AVX512 + xx + 2];
              }
            }
          }
        }
      }
    }

    highbd_apply_temporal_filter_avx512(
        ref, frame_stride, pred1 + plane_offset, plane_w, plane_w, plane_h,
        subblock_mses, accum + plane_offset, count + plane_offset, frame_sse,
        luma_sse_sum, mbd->bd, inv_num_ref_pixels, decay_factor, inv_factor,
        weight_factor, d_factor, tf_wgt_calc_lvl);
    plane_offset += plane_h * plane_w;
  }
  aom_free(frame_sse);
  aom_free(luma_sse_sum);
}
#endif  // CONFIG_AV1_HIGHBITDEPTH
