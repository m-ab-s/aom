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

#include "config/aom_config.h"
#include "config/av1_rtcd.h"

#include <assert.h>
#include <emmintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "aom_dsp/aom_dsp_common.h"
#include "av1/common/enums.h"
#include "av1/common/reconintra.h"

// Computes the horizontal sum of txwpx 8-bit pixels in above_ref using SSE2
// _mm_sad_epu8 against zero.
static AOM_FORCE_INLINE uint32_t
sum_above_row_lbd_sse2(const uint8_t *above_ref, int txwpx) {
  const __m128i zero = _mm_setzero_si128();
  if (LIKELY(txwpx <= 16)) {
    if (txwpx == 4) {
      uint32_t v;
      memcpy(&v, above_ref, 4);
      const __m128i s = _mm_sad_epu8(_mm_cvtsi32_si128((int)v), zero);
      return (uint32_t)_mm_cvtsi128_si32(s);
    } else if (txwpx == 8) {
      const __m128i s =
          _mm_sad_epu8(_mm_loadl_epi64((const __m128i *)above_ref), zero);
      return (uint32_t)_mm_cvtsi128_si32(s);
    } else {
      assert(txwpx == 16);
      const __m128i v = _mm_loadu_si128((const __m128i *)above_ref);
      const __m128i s = _mm_sad_epu8(v, zero);
      return (uint32_t)_mm_cvtsi128_si32(s) + (uint32_t)_mm_extract_epi16(s, 4);
    }
  }
  // Widths 32/64 kept for full RTCD TX_SIZES_ALL coverage.
  if (txwpx == 32) {
    const __m128i v0 = _mm_loadu_si128((const __m128i *)above_ref);
    const __m128i v1 = _mm_loadu_si128((const __m128i *)(above_ref + 16));
    const __m128i s =
        _mm_add_epi64(_mm_sad_epu8(v0, zero), _mm_sad_epu8(v1, zero));
    return (uint32_t)_mm_cvtsi128_si32(s) + (uint32_t)_mm_extract_epi16(s, 4);
  }
  assert(txwpx == 64);
  const __m128i v0 = _mm_loadu_si128((const __m128i *)above_ref);
  const __m128i v1 = _mm_loadu_si128((const __m128i *)(above_ref + 16));
  const __m128i v2 = _mm_loadu_si128((const __m128i *)(above_ref + 32));
  const __m128i v3 = _mm_loadu_si128((const __m128i *)(above_ref + 48));

  const __m128i s01 =
      _mm_add_epi64(_mm_sad_epu8(v0, zero), _mm_sad_epu8(v1, zero));
  const __m128i s23 =
      _mm_add_epi64(_mm_sad_epu8(v2, zero), _mm_sad_epu8(v3, zero));
  const __m128i s = _mm_add_epi64(s01, s23);
  return (uint32_t)_mm_cvtsi128_si32(s) + (uint32_t)_mm_extract_epi16(s, 4);
}

static AOM_FORCE_INLINE void fill_block_lbd_sse2(uint8_t *dst,
                                                 ptrdiff_t dst_stride,
                                                 uint8_t val, int txwpx,
                                                 int txhpx) {
  assert(txhpx >= 4);
  if (txwpx == 4) {
    assert((txhpx & 3) == 0);
    const uint32_t v = (uint32_t)val * 0x01010101u;
    // Avoid GCC over-unrolling the 4-row fill loop.
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 8
#pragma GCC unroll 1
#endif
    do {
      memcpy(dst, &v, 4);
      memcpy(dst + dst_stride, &v, 4);
      dst += dst_stride * 2;
      memcpy(dst, &v, 4);
      memcpy(dst + dst_stride, &v, 4);
      dst += dst_stride * 2;
      txhpx -= 4;
    } while (txhpx != 0);
    return;
  } else if (txwpx == 8) {
    assert((txhpx & 3) == 0);
    const uint64_t v = (uint64_t)val * 0x0101010101010101ULL;
    // Avoid GCC over-unrolling the 4-row fill loop.
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 8
#pragma GCC unroll 1
#endif
    do {
      memcpy(dst, &v, 8);
      memcpy(dst + dst_stride, &v, 8);
      dst += dst_stride * 2;
      memcpy(dst, &v, 8);
      memcpy(dst + dst_stride, &v, 8);
      dst += dst_stride * 2;
      txhpx -= 4;
    } while (txhpx != 0);
    return;
  }
  const __m128i v = _mm_set1_epi8((char)val);
  if (LIKELY(txwpx == 16)) {
    assert((txhpx & 3) == 0);
    // Avoid GCC over-unrolling the 4-row fill loop.
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 8
#pragma GCC unroll 1
#endif
    do {
      _mm_storeu_si128((__m128i *)dst, v);
      _mm_storeu_si128((__m128i *)(dst + dst_stride), v);
      dst += dst_stride * 2;
      _mm_storeu_si128((__m128i *)dst, v);
      _mm_storeu_si128((__m128i *)(dst + dst_stride), v);
      dst += dst_stride * 2;
      txhpx -= 4;
    } while (txhpx != 0);
  } else if (txwpx == 32) {
    // Widths 32/64 kept for full RTCD TX_SIZES_ALL coverage.
    assert((txhpx & 1) == 0);
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 8
#pragma GCC unroll 1
#endif
    do {
      _mm_storeu_si128((__m128i *)dst, v);
      _mm_storeu_si128((__m128i *)(dst + 16), v);
      _mm_storeu_si128((__m128i *)(dst + dst_stride), v);
      _mm_storeu_si128((__m128i *)(dst + dst_stride + 16), v);
      dst += dst_stride * 2;
      txhpx -= 2;
    } while (txhpx != 0);
  } else {
    assert(txwpx == 64);
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 8
#pragma GCC unroll 1
#endif
    do {
      _mm_storeu_si128((__m128i *)dst, v);
      _mm_storeu_si128((__m128i *)(dst + 16), v);
      _mm_storeu_si128((__m128i *)(dst + 32), v);
      _mm_storeu_si128((__m128i *)(dst + 48), v);
      dst += dst_stride;
    } while (--txhpx != 0);
  }
}

// Computes 8-bit DC_PRED directly from the reference boundary pixels without
// staging above_row and left_col buffers on the stack.
void av1_predict_dc_block_lbd_sse2(const uint8_t *ref, int ref_stride,
                                   uint8_t *dst, int dst_stride,
                                   TX_SIZE tx_size, int txwpx, int txhpx,
                                   int n_top_px, int n_left_px) {
  assert((unsigned int)tx_size < TX_SIZES_ALL);
  assert(txwpx == tx_size_wide[tx_size] && txhpx == tx_size_high[tx_size]);
  assert(n_top_px >= 0 && n_top_px <= txwpx);
  assert(n_left_px >= 0 && n_left_px <= txhpx);
  uint32_t sum_top = 0;
  if (n_top_px > 0) {
    const uint8_t *const above_ref = ref - ref_stride;
    if (LIKELY(n_top_px == txwpx)) {
      sum_top = sum_above_row_lbd_sse2(above_ref, txwpx);
    } else {
      for (int i = 0; i < n_top_px; ++i) sum_top += above_ref[i];
      sum_top +=
          (uint32_t)above_ref[n_top_px - 1] * (uint32_t)(txwpx - n_top_px);
    }
  }

  uint32_t sum_left = 0;
  if (n_left_px > 0) {
    const uint8_t *const left_ref = ref - 1;
    if (LIKELY(n_left_px == txhpx)) {
      sum_left = av1_sum_left_col_lbd(left_ref, ref_stride, txhpx);
    } else {
      uint8_t last_val = 0;
      for (int i = 0; i < n_left_px; ++i) {
        last_val = left_ref[(ptrdiff_t)i * ref_stride];
        sum_left += last_val;
      }
      sum_left += (uint32_t)last_val * (uint32_t)(txhpx - n_left_px);
    }
  }

  const uint8_t dc_val = av1_dc_val_from_sums_lbd(
      tx_size, txwpx, txhpx, sum_top, sum_left, n_top_px, n_left_px);
  fill_block_lbd_sse2(dst, dst_stride, dc_val, txwpx, txhpx);
}
