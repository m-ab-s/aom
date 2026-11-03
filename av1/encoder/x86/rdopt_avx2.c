/*
 * Copyright (c) 2018, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <assert.h>
#include <immintrin.h>
#include "aom_dsp/x86/mem_sse2.h"
#include "aom_dsp/x86/synonyms_avx2.h"

#include "config/av1_rtcd.h"
#include "av1/encoder/rdopt.h"

// Process horizontal and vertical correlations in a 4x4 block of pixels.
// We actually use the 4x4 pixels to calculate correlations corresponding to
// the top-left 3x3 pixels, so this function must be called with 1x1 overlap,
// moving the window along/down by 3 pixels at a time.
static inline void horver_correlation_4x4(const int16_t *diff, int stride,
                                          __m256i *xy_sum_32,
                                          __m256i *xz_sum_32, __m256i *x_sum_32,
                                          __m256i *x2_sum_32) {
  // Pixels in this 4x4   [ a b c d ]
  // are referred to as:  [ e f g h ]
  //                      [ i j k l ]
  //                      [ m n o p ]

  const __m256i pixels = _mm256_set_epi64x(
      loadu_int64(&diff[0 * stride]), loadu_int64(&diff[1 * stride]),
      loadu_int64(&diff[2 * stride]), loadu_int64(&diff[3 * stride]));
  // pixels = [d c b a h g f e] [l k j i p o n m] as i16

  const __m256i slli = _mm256_slli_epi64(pixels, 16);
  // slli = [c b a 0 g f e 0] [k j i 0 o n m 0] as i16

  const __m256i madd_xy = _mm256_madd_epi16(pixels, slli);
  // madd_xy = [bc+cd ab fg+gh ef] [jk+kl ij no+op mn] as i32
  *xy_sum_32 = _mm256_add_epi32(*xy_sum_32, madd_xy);

  // Permute control [3 2] [1 0] => [2 1] [0 0], 0b10010000 = 0x90
  const __m256i perm = _mm256_permute4x64_epi64(slli, 0x90);
  // perm = [g f e 0 k j i 0] [o n m 0 o n m 0] as i16

  const __m256i madd_xz = _mm256_madd_epi16(slli, perm);
  // madd_xz = [cg+bf ae gk+fj ei] [ko+jn im oo+nn mm] as i32
  *xz_sum_32 = _mm256_add_epi32(*xz_sum_32, madd_xz);

  // Sum every element in slli (and then also their squares)
  const __m256i madd1_slli = _mm256_madd_epi16(slli, _mm256_set1_epi16(1));
  // madd1_slli = [c+b a g+f e] [k+j i o+n m] as i32
  *x_sum_32 = _mm256_add_epi32(*x_sum_32, madd1_slli);

  const __m256i madd_slli = _mm256_madd_epi16(slli, slli);
  // madd_slli = [cc+bb aa gg+ff ee] [kk+jj ii oo+nn mm] as i32
  *x2_sum_32 = _mm256_add_epi32(*x2_sum_32, madd_slli);
}

void av1_get_horver_correlation_full_avx2(const int16_t *diff, int stride,
                                          int width, int height, float *hcorr,
                                          float *vcorr) {
  // The following notation is used:
  // x - current pixel
  // y - right neighbour pixel
  // z - below neighbour pixel
  // w - down-right neighbour pixel
  int64_t xy_sum = 0, xz_sum = 0;
  int64_t x_sum = 0, x2_sum = 0;

  // Process horizontal and vertical correlations through the body in 4x4
  // blocks.  This excludes the final row and column and possibly one extra
  // column depending how 3 divides into width and height
  int32_t xy_xz_tmp[8] = { 0 }, x_x2_tmp[8] = { 0 };
  __m256i xy_sum_32 = _mm256_setzero_si256();
  __m256i xz_sum_32 = _mm256_setzero_si256();
  __m256i x_sum_32 = _mm256_setzero_si256();
  __m256i x2_sum_32 = _mm256_setzero_si256();
  for (int i = 0; i <= height - 4; i += 3) {
    for (int j = 0; j <= width - 4; j += 3) {
      horver_correlation_4x4(&diff[i * stride + j], stride, &xy_sum_32,
                             &xz_sum_32, &x_sum_32, &x2_sum_32);
    }
    const __m256i hadd_xy_xz = _mm256_hadd_epi32(xy_sum_32, xz_sum_32);
    // hadd_xy_xz = [ae+bf+cg ei+fj+gk ab+bc+cd ef+fg+gh]
    //              [im+jn+ko mm+nn+oo ij+jk+kl mn+no+op] as i32
    yy_storeu_256(xy_xz_tmp, hadd_xy_xz);
    xy_sum += (int64_t)xy_xz_tmp[5] + xy_xz_tmp[4] + xy_xz_tmp[1];
    xz_sum += (int64_t)xy_xz_tmp[7] + xy_xz_tmp[6] + xy_xz_tmp[3];

    const __m256i hadd_x_x2 = _mm256_hadd_epi32(x_sum_32, x2_sum_32);
    // hadd_x_x2 = [aa+bb+cc ee+ff+gg a+b+c e+f+g]
    //             [ii+jj+kk mm+nn+oo i+j+k m+n+o] as i32
    yy_storeu_256(x_x2_tmp, hadd_x_x2);
    x_sum += (int64_t)x_x2_tmp[5] + x_x2_tmp[4] + x_x2_tmp[1];
    x2_sum += (int64_t)x_x2_tmp[7] + x_x2_tmp[6] + x_x2_tmp[3];

    xy_sum_32 = _mm256_setzero_si256();
    xz_sum_32 = _mm256_setzero_si256();
    x_sum_32 = _mm256_setzero_si256();
    x2_sum_32 = _mm256_setzero_si256();
  }

  // x_sum now covers every pixel except the final 1-2 rows and 1-2 cols
  int64_t x_finalrow = 0, x_finalcol = 0, x2_finalrow = 0, x2_finalcol = 0;

  // Do we have 2 rows remaining or just the one?  Note that width and height
  // are powers of 2, so each modulo 3 must be 1 or 2.
  if (height % 3 == 1) {  // Just horiz corrs on the final row
    const int16_t x0 = diff[(height - 1) * stride];
    x_sum += x0;
    x_finalrow += x0;
    x2_sum += x0 * x0;
    x2_finalrow += x0 * x0;
    for (int j = 0; j < width - 1; ++j) {
      const int16_t x = diff[(height - 1) * stride + j];
      const int16_t y = diff[(height - 1) * stride + j + 1];
      xy_sum += x * y;
      x_sum += y;
      x2_sum += y * y;
      x_finalrow += y;
      x2_finalrow += y * y;
    }
  } else {  // Two rows remaining to do
    const int16_t x0 = diff[(height - 2) * stride];
    const int16_t z0 = diff[(height - 1) * stride];
    x_sum += x0 + z0;
    x2_sum += x0 * x0 + z0 * z0;
    x_finalrow += z0;
    x2_finalrow += z0 * z0;
    for (int j = 0; j < width - 1; ++j) {
      const int16_t x = diff[(height - 2) * stride + j];
      const int16_t y = diff[(height - 2) * stride + j + 1];
      const int16_t z = diff[(height - 1) * stride + j];
      const int16_t w = diff[(height - 1) * stride + j + 1];

      // Horizontal and vertical correlations for the penultimate row:
      xy_sum += x * y;
      xz_sum += x * z;

      // Now just horizontal correlations for the final row:
      xy_sum += z * w;

      x_sum += y + w;
      x2_sum += y * y + w * w;
      x_finalrow += w;
      x2_finalrow += w * w;
    }
  }

  // Do we have 2 columns remaining or just the one?
  if (width % 3 == 1) {  // Just vert corrs on the final col
    const int16_t x0 = diff[width - 1];
    x_sum += x0;
    x_finalcol += x0;
    x2_sum += x0 * x0;
    x2_finalcol += x0 * x0;
    for (int i = 0; i < height - 1; ++i) {
      const int16_t x = diff[i * stride + width - 1];
      const int16_t z = diff[(i + 1) * stride + width - 1];
      xz_sum += x * z;
      x_finalcol += z;
      x2_finalcol += z * z;
      // So the bottom-right elements don't get counted twice:
      if (i < height - (height % 3 == 1 ? 2 : 3)) {
        x_sum += z;
        x2_sum += z * z;
      }
    }
  } else {  // Two cols remaining
    const int16_t x0 = diff[width - 2];
    const int16_t y0 = diff[width - 1];
    x_sum += x0 + y0;
    x2_sum += x0 * x0 + y0 * y0;
    x_finalcol += y0;
    x2_finalcol += y0 * y0;
    for (int i = 0; i < height - 1; ++i) {
      const int16_t x = diff[i * stride + width - 2];
      const int16_t y = diff[i * stride + width - 1];
      const int16_t z = diff[(i + 1) * stride + width - 2];
      const int16_t w = diff[(i + 1) * stride + width - 1];

      // Horizontal and vertical correlations for the penultimate col:
      // Skip these on the last iteration of this loop if we also had two
      // rows remaining, otherwise the final horizontal and vertical correlation
      // get erroneously processed twice
      if (i < height - 2 || height % 3 == 1) {
        xy_sum += x * y;
        xz_sum += x * z;
      }

      x_finalcol += w;
      x2_finalcol += w * w;
      // So the bottom-right elements don't get counted twice:
      if (i < height - (height % 3 == 1 ? 2 : 3)) {
        x_sum += z + w;
        x2_sum += z * z + w * w;
      }

      // Now just vertical correlations for the final column:
      xz_sum += y * w;
    }
  }

  // Calculate the simple sums and squared-sums
  int64_t x_firstrow = 0, x_firstcol = 0;
  int64_t x2_firstrow = 0, x2_firstcol = 0;

  for (int j = 0; j < width; ++j) {
    x_firstrow += diff[j];
    x2_firstrow += diff[j] * diff[j];
  }
  for (int i = 0; i < height; ++i) {
    x_firstcol += diff[i * stride];
    x2_firstcol += diff[i * stride] * diff[i * stride];
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

static inline uint64_t xx_cvtsi128_si64(__m128i a) {
#if AOM_ARCH_X86_64
  return (uint64_t)_mm_cvtsi128_si64(a);
#else
  uint64_t tmp;
  _mm_storel_epi64((__m128i *)&tmp, a);
  return tmp;
#endif
}

static inline uint64_t sum_u32x4_to_u64(__m128i v) {
  const __m256i v64 = _mm256_cvtepu32_epi64(v);
  const __m128i sum2 = _mm_add_epi64(_mm256_castsi256_si128(v64),
                                     _mm256_extracti128_si256(v64, 1));
  const __m128i sum1 = _mm_add_epi64(sum2, _mm_srli_si128(sum2, 8));
  return xx_cvtsi128_si64(sum1);
}

static inline uint64_t sum_u32x8_to_u64(__m256i v) {
  const __m256i lo64 = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(v));
  const __m256i hi64 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(v, 1));
  const __m256i sum4 = _mm256_add_epi64(lo64, hi64);
  const __m128i sum2 = _mm_add_epi64(_mm256_castsi256_si128(sum4),
                                     _mm256_extracti128_si256(sum4, 1));
  const __m128i sum1 = _mm_add_epi64(sum2, _mm_srli_si128(sum2, 8));
  return xx_cvtsi128_si64(sum1);
}

void av1_get_energy_distribution_finer_avx2(const int16_t *diff, int stride,
                                            int bw, int bh, float *hordist,
                                            float *verdist) {
  const int w_shift = bw <= 8 ? 0 : 1;
  const int h_shift = bh <= 8 ? 0 : 1;
  const int esq_w = bw >> w_shift;
  const int esq_h = bh >> h_shift;
  assert(esq_w <= 16);
  assert(esq_h <= 16);
  uint32_t col_sum[16] = { 0 };
  uint32_t row_sum[16] = { 0 };
  uint64_t total = 0;

  if (bw == 4) {
    __m128i v_col = _mm_setzero_si128();
    if (h_shift == 0) {
      for (int i = 0; i < esq_h; ++i) {
        const __m128i d = _mm_cvtepi16_epi32(
            _mm_loadl_epi64((const __m128i *)(diff + i * stride)));
        const __m128i sq = _mm_mullo_epi32(d, d);
        v_col = _mm_add_epi32(v_col, sq);
        const __m128i s1 = _mm_hadd_epi32(sq, sq);
        const __m128i s2 = _mm_hadd_epi32(s1, s1);
        row_sum[i] = (uint32_t)_mm_cvtsi128_si32(s2);
      }
    } else {
      for (int i = 0; i < esq_h; ++i) {
        const __m128i d0 =
            _mm_loadl_epi64((const __m128i *)(diff + (2 * i) * stride));
        const __m128i d1 =
            _mm_loadl_epi64((const __m128i *)(diff + (2 * i + 1) * stride));
        const __m128i d01 = _mm_unpacklo_epi16(d0, d1);
        const __m128i sq = _mm_madd_epi16(d01, d01);
        v_col = _mm_add_epi32(v_col, sq);
        const __m128i s1 = _mm_hadd_epi32(sq, sq);
        const __m128i s2 = _mm_hadd_epi32(s1, s1);
        row_sum[i] = (uint32_t)_mm_cvtsi128_si32(s2);
      }
    }
    _mm_storeu_si128((__m128i *)col_sum, v_col);
    total = sum_u32x4_to_u64(v_col);
  } else if (bw == 8) {
    __m256i v_col = _mm256_setzero_si256();
    if (h_shift == 0) {
      for (int i = 0; i < esq_h; ++i) {
        const __m256i d = _mm256_cvtepi16_epi32(
            _mm_loadu_si128((const __m128i *)(diff + i * stride)));
        const __m256i sq = _mm256_mullo_epi32(d, d);
        v_col = _mm256_add_epi32(v_col, sq);
        const __m128i sq128 = _mm_add_epi32(_mm256_castsi256_si128(sq),
                                            _mm256_extracti128_si256(sq, 1));
        const __m128i s1 = _mm_hadd_epi32(sq128, sq128);
        const __m128i s2 = _mm_hadd_epi32(s1, s1);
        row_sum[i] = (uint32_t)_mm_cvtsi128_si32(s2);
      }
    } else {
      for (int i = 0; i < esq_h; ++i) {
        const __m128i d0 =
            _mm_loadu_si128((const __m128i *)(diff + (2 * i) * stride));
        const __m128i d1 =
            _mm_loadu_si128((const __m128i *)(diff + (2 * i + 1) * stride));
        const __m256i d0_256 = _mm256_cvtepi16_epi32(d0);
        const __m256i d1_256 = _mm256_cvtepi16_epi32(d1);
        const __m256i sq = _mm256_add_epi32(_mm256_mullo_epi32(d0_256, d0_256),
                                            _mm256_mullo_epi32(d1_256, d1_256));
        v_col = _mm256_add_epi32(v_col, sq);
        const __m128i sq128 = _mm_add_epi32(_mm256_castsi256_si128(sq),
                                            _mm256_extracti128_si256(sq, 1));
        const __m128i s1 = _mm_hadd_epi32(sq128, sq128);
        const __m128i s2 = _mm_hadd_epi32(s1, s1);
        row_sum[i] = (uint32_t)_mm_cvtsi128_si32(s2);
      }
    }
    _mm256_storeu_si256((__m256i *)col_sum, v_col);
    total = sum_u32x8_to_u64(v_col);
  } else if (bw == 16) {
    __m256i v_col = _mm256_setzero_si256();
    if (h_shift == 0) {
      for (int i = 0; i < esq_h; ++i) {
        const __m256i d =
            _mm256_loadu_si256((const __m256i *)(diff + i * stride));
        const __m256i sq = _mm256_madd_epi16(d, d);
        v_col = _mm256_add_epi32(v_col, sq);
        const __m128i sq128 = _mm_add_epi32(_mm256_castsi256_si128(sq),
                                            _mm256_extracti128_si256(sq, 1));
        const __m128i s1 = _mm_hadd_epi32(sq128, sq128);
        const __m128i s2 = _mm_hadd_epi32(s1, s1);
        row_sum[i] = (uint32_t)_mm_cvtsi128_si32(s2);
      }
    } else {
      for (int i = 0; i < esq_h; ++i) {
        const __m256i d0 =
            _mm256_loadu_si256((const __m256i *)(diff + (2 * i) * stride));
        const __m256i d1 =
            _mm256_loadu_si256((const __m256i *)(diff + (2 * i + 1) * stride));
        const __m256i sq = _mm256_add_epi32(_mm256_madd_epi16(d0, d0),
                                            _mm256_madd_epi16(d1, d1));
        v_col = _mm256_add_epi32(v_col, sq);
        const __m128i sq128 = _mm_add_epi32(_mm256_castsi256_si128(sq),
                                            _mm256_extracti128_si256(sq, 1));
        const __m128i s1 = _mm_hadd_epi32(sq128, sq128);
        const __m128i s2 = _mm_hadd_epi32(s1, s1);
        row_sum[i] = (uint32_t)_mm_cvtsi128_si32(s2);
      }
    }
    _mm256_storeu_si256((__m256i *)col_sum, v_col);
    total = sum_u32x8_to_u64(v_col);
  } else {
    assert(bw == 32);
    __m256i v_col_lo = _mm256_setzero_si256();
    __m256i v_col_hi = _mm256_setzero_si256();
    if (h_shift == 0) {
      for (int i = 0; i < esq_h; ++i) {
        const int16_t *r0 = diff + i * stride;
        const __m256i d0_lo = _mm256_loadu_si256((const __m256i *)r0);
        const __m256i d0_hi = _mm256_loadu_si256((const __m256i *)(r0 + 16));
        const __m256i sq_lo = _mm256_madd_epi16(d0_lo, d0_lo);
        const __m256i sq_hi = _mm256_madd_epi16(d0_hi, d0_hi);
        v_col_lo = _mm256_add_epi32(v_col_lo, sq_lo);
        v_col_hi = _mm256_add_epi32(v_col_hi, sq_hi);
        const __m256i sq_sum = _mm256_add_epi32(sq_lo, sq_hi);
        const __m128i sq128 =
            _mm_add_epi32(_mm256_castsi256_si128(sq_sum),
                          _mm256_extracti128_si256(sq_sum, 1));
        const __m128i s1 = _mm_hadd_epi32(sq128, sq128);
        const __m128i s2 = _mm_hadd_epi32(s1, s1);
        row_sum[i] = (uint32_t)_mm_cvtsi128_si32(s2);
      }
    } else {
      for (int i = 0; i < esq_h; ++i) {
        const int16_t *r0 = diff + (2 * i) * stride;
        const int16_t *r1 = diff + (2 * i + 1) * stride;
        const __m256i d0_lo = _mm256_loadu_si256((const __m256i *)r0);
        const __m256i d0_hi = _mm256_loadu_si256((const __m256i *)(r0 + 16));
        const __m256i d1_lo = _mm256_loadu_si256((const __m256i *)r1);
        const __m256i d1_hi = _mm256_loadu_si256((const __m256i *)(r1 + 16));
        const __m256i sq_lo = _mm256_add_epi32(_mm256_madd_epi16(d0_lo, d0_lo),
                                               _mm256_madd_epi16(d1_lo, d1_lo));
        const __m256i sq_hi = _mm256_add_epi32(_mm256_madd_epi16(d0_hi, d0_hi),
                                               _mm256_madd_epi16(d1_hi, d1_hi));
        v_col_lo = _mm256_add_epi32(v_col_lo, sq_lo);
        v_col_hi = _mm256_add_epi32(v_col_hi, sq_hi);
        const __m256i sq_sum = _mm256_add_epi32(sq_lo, sq_hi);
        const __m128i sq128 =
            _mm_add_epi32(_mm256_castsi256_si128(sq_sum),
                          _mm256_extracti128_si256(sq_sum, 1));
        const __m128i s1 = _mm_hadd_epi32(sq128, sq128);
        const __m128i s2 = _mm_hadd_epi32(s1, s1);
        row_sum[i] = (uint32_t)_mm_cvtsi128_si32(s2);
      }
    }
    _mm256_storeu_si256((__m256i *)col_sum, v_col_lo);
    _mm256_storeu_si256((__m256i *)(col_sum + 8), v_col_hi);
    total = sum_u32x8_to_u64(v_col_lo) + sum_u32x8_to_u64(v_col_hi);
  }

  // Output hordist and verdist arrays are normalized 1D projections of esq
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
