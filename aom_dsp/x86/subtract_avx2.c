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
#include <stdint.h>

#include "config/aom_dsp_rtcd.h"
#include "aom_dsp/x86/synonyms.h"

static inline void subtract_block_4xn_avx2(
    int rows, int16_t *diff_ptr, ptrdiff_t diff_stride, const uint8_t *src_ptr,
    ptrdiff_t src_stride, const uint8_t *pred_ptr, ptrdiff_t pred_stride) {
  assert(rows >= 2 && (rows & 1) == 0);
  int32_t j = rows >> 1;
  // Process 2 rows of 4 pixels per iteration using 128-bit XMM registers
  // rather than 4 rows in a 256-bit YMM register:
  // 1. xx_loadl_32 + _mm_unpacklo_epi32 loads directly from memory into XMM
  //    without GPR-to-XMM inserts (_mm_set_epi32 / vpinsrd on Port 5).
  // 2. Staying in a single 128-bit lane avoids the cross-lane latency of
  //    _mm256_cvtepu8_epi16 and _mm256_extracti128_si256 needed for 8-byte
  //    per-row stores.
  // 3. Stepping pointers by stride each row avoids 3 * stride LEA addressing
  //    and eliminates a separate 2-row remainder loop.
  do {
    const __m128i s0 = xx_loadl_32(src_ptr);
    src_ptr += src_stride;
    const __m128i s1 = xx_loadl_32(src_ptr);
    src_ptr += src_stride;
    const __m128i p0 = xx_loadl_32(pred_ptr);
    pred_ptr += pred_stride;
    const __m128i p1 = xx_loadl_32(pred_ptr);
    pred_ptr += pred_stride;

    const __m128i d =
        _mm_sub_epi16(_mm_cvtepu8_epi16(_mm_unpacklo_epi32(s0, s1)),
                      _mm_cvtepu8_epi16(_mm_unpacklo_epi32(p0, p1)));

    _mm_storel_epi64((__m128i *)diff_ptr, d);
    diff_ptr += diff_stride;
    _mm_storeh_pi((__m64 *)diff_ptr, _mm_castsi128_ps(d));
    diff_ptr += diff_stride;
  } while (--j != 0);
}

static inline void subtract_block_8xn_avx2(
    int rows, int16_t *diff_ptr, ptrdiff_t diff_stride, const uint8_t *src_ptr,
    ptrdiff_t src_stride, const uint8_t *pred_ptr, ptrdiff_t pred_stride) {
  assert(rows >= 2 && (rows & 1) == 0);
  int32_t j = rows >> 1;
  // Process 2 rows of 8 pixels per iteration using a 256-bit register.
  do {
    const __m128i s0 = _mm_loadl_epi64((const __m128i *)src_ptr);
    src_ptr += src_stride;
    const __m128i s1 = _mm_loadl_epi64((const __m128i *)src_ptr);
    src_ptr += src_stride;
    const __m128i p0 = _mm_loadl_epi64((const __m128i *)pred_ptr);
    pred_ptr += pred_stride;
    const __m128i p1 = _mm_loadl_epi64((const __m128i *)pred_ptr);
    pred_ptr += pred_stride;

    const __m256i d01 =
        _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm_unpacklo_epi64(s0, s1)),
                         _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(p0, p1)));

    _mm_storeu_si128((__m128i *)diff_ptr, _mm256_castsi256_si128(d01));
    diff_ptr += diff_stride;
    _mm_storeu_si128((__m128i *)diff_ptr, _mm256_extracti128_si256(d01, 1));
    diff_ptr += diff_stride;
  } while (--j != 0);
}

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

static inline void subtract_block_16xn_avx2(
    int rows, int16_t *diff_ptr, ptrdiff_t diff_stride, const uint8_t *src_ptr,
    ptrdiff_t src_stride, const uint8_t *pred_ptr, ptrdiff_t pred_stride) {
  assert(rows >= 2 && (rows & 1) == 0);
  int32_t j = rows >> 1;
  do {
    const __m128i s0 = _mm_lddqu_si128((const __m128i *)src_ptr);
    src_ptr += src_stride;
    const __m128i s1 = _mm_lddqu_si128((const __m128i *)src_ptr);
    src_ptr += src_stride;
    const __m128i p0 = _mm_lddqu_si128((const __m128i *)pred_ptr);
    pred_ptr += pred_stride;
    const __m128i p1 = _mm_lddqu_si128((const __m128i *)pred_ptr);
    pred_ptr += pred_stride;
    const __m256i d0 =
        _mm256_sub_epi16(_mm256_cvtepu8_epi16(s0), _mm256_cvtepu8_epi16(p0));
    const __m256i d1 =
        _mm256_sub_epi16(_mm256_cvtepu8_epi16(s1), _mm256_cvtepu8_epi16(p1));
    _mm256_store_si256((__m256i *)diff_ptr, d0);
    diff_ptr += diff_stride;
    _mm256_store_si256((__m256i *)diff_ptr, d1);
    diff_ptr += diff_stride;
  } while (--j != 0);
}

static inline void subtract_block_32xn_avx2(
    int rows, int16_t *diff_ptr, ptrdiff_t diff_stride, const uint8_t *src_ptr,
    ptrdiff_t src_stride, const uint8_t *pred_ptr, ptrdiff_t pred_stride) {
  assert(rows > 0);
  int32_t j = rows;
  do {
    subtract32_avx2(diff_ptr, src_ptr, pred_ptr);
    src_ptr += src_stride;
    pred_ptr += pred_stride;
    diff_ptr += diff_stride;
  } while (--j != 0);
}

static inline void subtract_block_64xn_avx2(
    int rows, int16_t *diff_ptr, ptrdiff_t diff_stride, const uint8_t *src_ptr,
    ptrdiff_t src_stride, const uint8_t *pred_ptr, ptrdiff_t pred_stride) {
  assert(rows > 0);
  int32_t j = rows;
  do {
    subtract32_avx2(diff_ptr, src_ptr, pred_ptr);
    subtract32_avx2(diff_ptr + 32, src_ptr + 32, pred_ptr + 32);
    src_ptr += src_stride;
    pred_ptr += pred_stride;
    diff_ptr += diff_stride;
  } while (--j != 0);
}

static inline void subtract_block_128xn_avx2(
    int rows, int16_t *diff_ptr, ptrdiff_t diff_stride, const uint8_t *src_ptr,
    ptrdiff_t src_stride, const uint8_t *pred_ptr, ptrdiff_t pred_stride) {
  assert(rows > 0);
  int32_t j = rows;
  do {
    subtract32_avx2(diff_ptr, src_ptr, pred_ptr);
    subtract32_avx2(diff_ptr + 32, src_ptr + 32, pred_ptr + 32);
    subtract32_avx2(diff_ptr + 64, src_ptr + 64, pred_ptr + 64);
    subtract32_avx2(diff_ptr + 96, src_ptr + 96, pred_ptr + 96);
    src_ptr += src_stride;
    pred_ptr += pred_stride;
    diff_ptr += diff_stride;
  } while (--j != 0);
}

void aom_subtract_block_avx2(int rows, int cols, int16_t *diff_ptr,
                             ptrdiff_t diff_stride, const uint8_t *src_ptr,
                             ptrdiff_t src_stride, const uint8_t *pred_ptr,
                             ptrdiff_t pred_stride) {
  if (cols >= 16) {
    assert(((uintptr_t)diff_ptr & 31) == 0 &&
           ((diff_stride * sizeof(*diff_ptr)) & 31) == 0);
  }
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
    default:
      assert(cols == 128);
      subtract_block_128xn_avx2(rows, diff_ptr, diff_stride, src_ptr,
                                src_stride, pred_ptr, pred_stride);
      break;
  }
}
