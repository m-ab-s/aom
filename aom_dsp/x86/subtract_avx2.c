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
#include <immintrin.h>

#include "config/aom_dsp_rtcd.h"

static inline void subtract32_avx2(int16_t *diff_ptr, const uint8_t *src_ptr,
                                   const uint8_t *pred_ptr) {
  __m256i s = _mm256_lddqu_si256((__m256i *)(src_ptr));
  __m256i p = _mm256_lddqu_si256((__m256i *)(pred_ptr));
  __m256i set_one_minusone = _mm256_set1_epi32((int)0xff01ff01);
  __m256i diff0 = _mm256_unpacklo_epi8(s, p);
  __m256i diff1 = _mm256_unpackhi_epi8(s, p);
  diff0 = _mm256_maddubs_epi16(diff0, set_one_minusone);
  diff1 = _mm256_maddubs_epi16(diff1, set_one_minusone);
  _mm256_store_si256((__m256i *)(diff_ptr),
                     _mm256_permute2x128_si256(diff0, diff1, 0x20));
  _mm256_store_si256((__m256i *)(diff_ptr + 16),
                     _mm256_permute2x128_si256(diff0, diff1, 0x31));
}

static inline void subtract_block_4xn_avx2(
    int rows, int16_t *diff_ptr, ptrdiff_t diff_stride, const uint8_t *src_ptr,
    ptrdiff_t src_stride, const uint8_t *pred_ptr, ptrdiff_t pred_stride) {
  int32_t j = 0;
  for (; j + 3 < rows; j += 4) {
    const __m128i s = _mm_setr_epi32(
        *(const int32_t *)(src_ptr), *(const int32_t *)(src_ptr + src_stride),
        *(const int32_t *)(src_ptr + src_stride * 2),
        *(const int32_t *)(src_ptr + src_stride * 3));
    const __m128i p =
        _mm_setr_epi32(*(const int32_t *)(pred_ptr),
                       *(const int32_t *)(pred_ptr + pred_stride),
                       *(const int32_t *)(pred_ptr + pred_stride * 2),
                       *(const int32_t *)(pred_ptr + pred_stride * 3));
    const __m256i d =
        _mm256_sub_epi16(_mm256_cvtepu8_epi16(s), _mm256_cvtepu8_epi16(p));
    const __m128i d_lo = _mm256_castsi256_si128(d);
    const __m128i d_hi = _mm256_extracti128_si256(d, 1);
    _mm_storel_epi64((__m128i *)(diff_ptr), d_lo);
    _mm_storeh_pi((__m64 *)(diff_ptr + diff_stride), _mm_castsi128_ps(d_lo));
    _mm_storel_epi64((__m128i *)(diff_ptr + diff_stride * 2), d_hi);
    _mm_storeh_pi((__m64 *)(diff_ptr + diff_stride * 3),
                  _mm_castsi128_ps(d_hi));
    src_ptr += src_stride * 4;
    pred_ptr += pred_stride * 4;
    diff_ptr += diff_stride * 4;
  }
  for (; j < rows; j += 2) {
    const __m128i s =
        _mm_setr_epi32(*(const int32_t *)(src_ptr),
                       *(const int32_t *)(src_ptr + src_stride), 0, 0);
    const __m128i p =
        _mm_setr_epi32(*(const int32_t *)(pred_ptr),
                       *(const int32_t *)(pred_ptr + pred_stride), 0, 0);
    const __m128i d = _mm_sub_epi16(_mm_cvtepu8_epi16(s), _mm_cvtepu8_epi16(p));
    _mm_storel_epi64((__m128i *)(diff_ptr), d);
    _mm_storeh_pi((__m64 *)(diff_ptr + diff_stride), _mm_castsi128_ps(d));
    src_ptr += src_stride * 2;
    pred_ptr += pred_stride * 2;
    diff_ptr += diff_stride * 2;
  }
}

static inline void subtract_block_8xn_avx2(
    int rows, int16_t *diff_ptr, ptrdiff_t diff_stride, const uint8_t *src_ptr,
    ptrdiff_t src_stride, const uint8_t *pred_ptr, ptrdiff_t pred_stride) {
  for (int32_t j = 0; j < rows; j += 2) {
    const __m128i s = _mm_unpacklo_epi64(
        _mm_loadl_epi64((const __m128i *)(src_ptr)),
        _mm_loadl_epi64((const __m128i *)(src_ptr + src_stride)));
    const __m128i p = _mm_unpacklo_epi64(
        _mm_loadl_epi64((const __m128i *)(pred_ptr)),
        _mm_loadl_epi64((const __m128i *)(pred_ptr + pred_stride)));
    const __m256i d =
        _mm256_sub_epi16(_mm256_cvtepu8_epi16(s), _mm256_cvtepu8_epi16(p));
    _mm_store_si128((__m128i *)(diff_ptr), _mm256_castsi256_si128(d));
    _mm_store_si128((__m128i *)(diff_ptr + diff_stride),
                    _mm256_extracti128_si256(d, 1));
    src_ptr += src_stride * 2;
    pred_ptr += pred_stride * 2;
    diff_ptr += diff_stride * 2;
  }
}

static inline void subtract_block_16xn_avx2(
    int rows, int16_t *diff_ptr, ptrdiff_t diff_stride, const uint8_t *src_ptr,
    ptrdiff_t src_stride, const uint8_t *pred_ptr, ptrdiff_t pred_stride) {
  for (int32_t j = 0; j < rows; j += 2) {
    const __m128i s0 = _mm_lddqu_si128((const __m128i *)(src_ptr));
    const __m128i p0 = _mm_lddqu_si128((const __m128i *)(pred_ptr));
    const __m128i s1 = _mm_lddqu_si128((const __m128i *)(src_ptr + src_stride));
    const __m128i p1 =
        _mm_lddqu_si128((const __m128i *)(pred_ptr + pred_stride));
    const __m256i d0 =
        _mm256_sub_epi16(_mm256_cvtepu8_epi16(s0), _mm256_cvtepu8_epi16(p0));
    const __m256i d1 =
        _mm256_sub_epi16(_mm256_cvtepu8_epi16(s1), _mm256_cvtepu8_epi16(p1));
    _mm256_store_si256((__m256i *)(diff_ptr), d0);
    _mm256_store_si256((__m256i *)(diff_ptr + diff_stride), d1);
    src_ptr += src_stride * 2;
    pred_ptr += pred_stride * 2;
    diff_ptr += diff_stride * 2;
  }
}

static inline void subtract_block_32xn_avx2(
    int rows, int16_t *diff_ptr, ptrdiff_t diff_stride, const uint8_t *src_ptr,
    ptrdiff_t src_stride, const uint8_t *pred_ptr, ptrdiff_t pred_stride) {
  for (int32_t j = 0; j < rows; ++j) {
    subtract32_avx2(diff_ptr, src_ptr, pred_ptr);
    src_ptr += src_stride;
    pred_ptr += pred_stride;
    diff_ptr += diff_stride;
  }
}

static inline void subtract_block_64xn_avx2(
    int rows, int16_t *diff_ptr, ptrdiff_t diff_stride, const uint8_t *src_ptr,
    ptrdiff_t src_stride, const uint8_t *pred_ptr, ptrdiff_t pred_stride) {
  for (int32_t j = 0; j < rows; ++j) {
    subtract32_avx2(diff_ptr, src_ptr, pred_ptr);
    subtract32_avx2(diff_ptr + 32, src_ptr + 32, pred_ptr + 32);
    src_ptr += src_stride;
    pred_ptr += pred_stride;
    diff_ptr += diff_stride;
  }
}

static inline void subtract_block_128xn_avx2(
    int rows, int16_t *diff_ptr, ptrdiff_t diff_stride, const uint8_t *src_ptr,
    ptrdiff_t src_stride, const uint8_t *pred_ptr, ptrdiff_t pred_stride) {
  for (int32_t j = 0; j < rows; ++j) {
    subtract32_avx2(diff_ptr, src_ptr, pred_ptr);
    subtract32_avx2(diff_ptr + 32, src_ptr + 32, pred_ptr + 32);
    subtract32_avx2(diff_ptr + 64, src_ptr + 64, pred_ptr + 64);
    subtract32_avx2(diff_ptr + 96, src_ptr + 96, pred_ptr + 96);
    src_ptr += src_stride;
    pred_ptr += pred_stride;
    diff_ptr += diff_stride;
  }
}

void aom_subtract_block_avx2(int rows, int cols, int16_t *diff_ptr,
                             ptrdiff_t diff_stride, const uint8_t *src_ptr,
                             ptrdiff_t src_stride, const uint8_t *pred_ptr,
                             ptrdiff_t pred_stride) {
  switch (cols) {
    case 4:
      subtract_block_4xn_avx2(rows, diff_ptr, diff_stride, src_ptr, src_stride,
                              pred_ptr, pred_stride);
      break;
    case 8:
      subtract_block_8xn_avx2(rows, diff_ptr, diff_stride, src_ptr, src_stride,
                              pred_ptr, pred_stride);
      break;
    case 16:
      subtract_block_16xn_avx2(rows, diff_ptr, diff_stride, src_ptr, src_stride,
                               pred_ptr, pred_stride);
      break;
    case 32:
      subtract_block_32xn_avx2(rows, diff_ptr, diff_stride, src_ptr, src_stride,
                               pred_ptr, pred_stride);
      break;
    case 64:
      subtract_block_64xn_avx2(rows, diff_ptr, diff_stride, src_ptr, src_stride,
                               pred_ptr, pred_stride);
      break;
    case 128:
      subtract_block_128xn_avx2(rows, diff_ptr, diff_stride, src_ptr,
                                src_stride, pred_ptr, pred_stride);
      break;
    default:
#if HAVE_X86_ASM
      aom_subtract_block_sse2(rows, cols, diff_ptr, diff_stride, src_ptr,
                              src_stride, pred_ptr, pred_stride);
#else
      aom_subtract_block_c(rows, cols, diff_ptr, diff_stride, src_ptr,
                           src_stride, pred_ptr, pred_stride);
#endif
      break;
  }
}
