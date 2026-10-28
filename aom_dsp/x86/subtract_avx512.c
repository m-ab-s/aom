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

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"
#include "aom/aom_integer.h"

static inline void subtract32_avx512(int16_t *diff_ptr, const uint8_t *src_ptr,
                                     const uint8_t *pred_ptr) {
  const __m256i s = _mm256_loadu_si256((const __m256i *)src_ptr);
  const __m256i p = _mm256_loadu_si256((const __m256i *)pred_ptr);
  const __m512i d =
      _mm512_sub_epi16(_mm512_cvtepu8_epi16(s), _mm512_cvtepu8_epi16(p));
  _mm512_storeu_si512((__m512i *)diff_ptr, d);
}

static inline void subtract_block_4xn_avx512(
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

static inline void subtract_block_8xn_avx512(
    int rows, int16_t *diff_ptr, ptrdiff_t diff_stride, const uint8_t *src_ptr,
    ptrdiff_t src_stride, const uint8_t *pred_ptr, ptrdiff_t pred_stride) {
  int32_t j = 0;
  for (; j + 3 < rows; j += 4) {
    const __m128i s01 = _mm_unpacklo_epi64(
        _mm_loadl_epi64((const __m128i *)(src_ptr)),
        _mm_loadl_epi64((const __m128i *)(src_ptr + src_stride)));
    const __m128i s23 = _mm_unpacklo_epi64(
        _mm_loadl_epi64((const __m128i *)(src_ptr + src_stride * 2)),
        _mm_loadl_epi64((const __m128i *)(src_ptr + src_stride * 3)));
    const __m128i p01 = _mm_unpacklo_epi64(
        _mm_loadl_epi64((const __m128i *)(pred_ptr)),
        _mm_loadl_epi64((const __m128i *)(pred_ptr + pred_stride)));
    const __m128i p23 = _mm_unpacklo_epi64(
        _mm_loadl_epi64((const __m128i *)(pred_ptr + pred_stride * 2)),
        _mm_loadl_epi64((const __m128i *)(pred_ptr + pred_stride * 3)));
    const __m256i s =
        _mm256_inserti128_si256(_mm256_castsi128_si256(s01), s23, 1);
    const __m256i p =
        _mm256_inserti128_si256(_mm256_castsi128_si256(p01), p23, 1);
    const __m512i d =
        _mm512_sub_epi16(_mm512_cvtepu8_epi16(s), _mm512_cvtepu8_epi16(p));
    _mm_storeu_si128((__m128i *)(diff_ptr), _mm512_castsi512_si128(d));
    _mm_storeu_si128((__m128i *)(diff_ptr + diff_stride),
                     _mm512_extracti32x4_epi32(d, 1));
    _mm_storeu_si128((__m128i *)(diff_ptr + diff_stride * 2),
                     _mm512_extracti32x4_epi32(d, 2));
    _mm_storeu_si128((__m128i *)(diff_ptr + diff_stride * 3),
                     _mm512_extracti32x4_epi32(d, 3));
    src_ptr += src_stride * 4;
    pred_ptr += pred_stride * 4;
    diff_ptr += diff_stride * 4;
  }
  for (; j < rows; j += 2) {
    const __m128i s = _mm_unpacklo_epi64(
        _mm_loadl_epi64((const __m128i *)(src_ptr)),
        _mm_loadl_epi64((const __m128i *)(src_ptr + src_stride)));
    const __m128i p = _mm_unpacklo_epi64(
        _mm_loadl_epi64((const __m128i *)(pred_ptr)),
        _mm_loadl_epi64((const __m128i *)(pred_ptr + pred_stride)));
    const __m256i d =
        _mm256_sub_epi16(_mm256_cvtepu8_epi16(s), _mm256_cvtepu8_epi16(p));
    _mm_storeu_si128((__m128i *)(diff_ptr), _mm256_castsi256_si128(d));
    _mm_storeu_si128((__m128i *)(diff_ptr + diff_stride),
                     _mm256_extracti128_si256(d, 1));
    src_ptr += src_stride * 2;
    pred_ptr += pred_stride * 2;
    diff_ptr += diff_stride * 2;
  }
}

static inline void subtract_block_16xn_avx512(
    int rows, int16_t *diff_ptr, ptrdiff_t diff_stride, const uint8_t *src_ptr,
    ptrdiff_t src_stride, const uint8_t *pred_ptr, ptrdiff_t pred_stride) {
  for (int32_t j = 0; j < rows; j += 2) {
    const __m128i s0 = _mm_loadu_si128((const __m128i *)(src_ptr));
    const __m128i p0 = _mm_loadu_si128((const __m128i *)(pred_ptr));
    const __m128i s1 = _mm_loadu_si128((const __m128i *)(src_ptr + src_stride));
    const __m128i p1 =
        _mm_loadu_si128((const __m128i *)(pred_ptr + pred_stride));
    const __m256i s =
        _mm256_inserti128_si256(_mm256_castsi128_si256(s0), s1, 1);
    const __m256i p =
        _mm256_inserti128_si256(_mm256_castsi128_si256(p0), p1, 1);
    const __m512i d =
        _mm512_sub_epi16(_mm512_cvtepu8_epi16(s), _mm512_cvtepu8_epi16(p));
    _mm256_storeu_si256((__m256i *)(diff_ptr), _mm512_castsi512_si256(d));
    _mm256_storeu_si256((__m256i *)(diff_ptr + diff_stride),
                        _mm512_extracti64x4_epi64(d, 1));
    src_ptr += src_stride * 2;
    pred_ptr += pred_stride * 2;
    diff_ptr += diff_stride * 2;
  }
}

static inline void subtract_block_32xn_avx512(
    int rows, int16_t *diff_ptr, ptrdiff_t diff_stride, const uint8_t *src_ptr,
    ptrdiff_t src_stride, const uint8_t *pred_ptr, ptrdiff_t pred_stride) {
  for (int32_t j = 0; j < rows; j += 2) {
    subtract32_avx512(diff_ptr, src_ptr, pred_ptr);
    subtract32_avx512(diff_ptr + diff_stride, src_ptr + src_stride,
                      pred_ptr + pred_stride);
    src_ptr += src_stride * 2;
    pred_ptr += pred_stride * 2;
    diff_ptr += diff_stride * 2;
  }
}

static inline void subtract_block_64xn_avx512(
    int rows, int16_t *diff_ptr, ptrdiff_t diff_stride, const uint8_t *src_ptr,
    ptrdiff_t src_stride, const uint8_t *pred_ptr, ptrdiff_t pred_stride) {
  for (int32_t j = 0; j < rows; ++j) {
    subtract32_avx512(diff_ptr, src_ptr, pred_ptr);
    subtract32_avx512(diff_ptr + 32, src_ptr + 32, pred_ptr + 32);
    src_ptr += src_stride;
    pred_ptr += pred_stride;
    diff_ptr += diff_stride;
  }
}

static inline void subtract_block_128xn_avx512(
    int rows, int16_t *diff_ptr, ptrdiff_t diff_stride, const uint8_t *src_ptr,
    ptrdiff_t src_stride, const uint8_t *pred_ptr, ptrdiff_t pred_stride) {
  for (int32_t j = 0; j < rows; ++j) {
    subtract32_avx512(diff_ptr, src_ptr, pred_ptr);
    subtract32_avx512(diff_ptr + 32, src_ptr + 32, pred_ptr + 32);
    subtract32_avx512(diff_ptr + 64, src_ptr + 64, pred_ptr + 64);
    subtract32_avx512(diff_ptr + 96, src_ptr + 96, pred_ptr + 96);
    src_ptr += src_stride;
    pred_ptr += pred_stride;
    diff_ptr += diff_stride;
  }
}

void aom_subtract_block_avx512(int rows, int cols, int16_t *diff_ptr,
                               ptrdiff_t diff_stride, const uint8_t *src_ptr,
                               ptrdiff_t src_stride, const uint8_t *pred_ptr,
                               ptrdiff_t pred_stride) {
  switch (cols) {
    case 4:
      subtract_block_4xn_avx512(rows, diff_ptr, diff_stride, src_ptr,
                                src_stride, pred_ptr, pred_stride);
      break;
    case 8:
      subtract_block_8xn_avx512(rows, diff_ptr, diff_stride, src_ptr,
                                src_stride, pred_ptr, pred_stride);
      break;
    case 16:
      subtract_block_16xn_avx512(rows, diff_ptr, diff_stride, src_ptr,
                                 src_stride, pred_ptr, pred_stride);
      break;
    case 32:
      subtract_block_32xn_avx512(rows, diff_ptr, diff_stride, src_ptr,
                                 src_stride, pred_ptr, pred_stride);
      break;
    case 64:
      subtract_block_64xn_avx512(rows, diff_ptr, diff_stride, src_ptr,
                                 src_stride, pred_ptr, pred_stride);
      break;
    case 128:
      subtract_block_128xn_avx512(rows, diff_ptr, diff_stride, src_ptr,
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
