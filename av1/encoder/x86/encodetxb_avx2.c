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
#include <emmintrin.h>  // SSE2
#include <smmintrin.h>  /* SSE4.1 */
#include <immintrin.h>  /* AVX2 */

#include "aom/aom_integer.h"
#include "aom_dsp/x86/mem_sse2.h"
#include "av1/common/av1_common_int.h"
#include "av1/common/txb_common.h"
#include "aom_dsp/x86/synonyms.h"
#include "aom_dsp/x86/synonyms_avx2.h"

void av1_txb_init_levels_avx2(const tran_low_t *const coeff, const int width,
                              const int height, uint8_t *const levels) {
  assert(TX_PAD_HOR == 4 && TX_PAD_BOTTOM == 4);
  const __m256i y_zeros = _mm256_setzero_si256();
  const __m128i zero = _mm256_castsi256_si128(y_zeros);

  uint8_t *ls = levels;
  const tran_low_t *cf = coeff;

  // Specialize each transform height with its constant padded stride
  // (height + TX_PAD_HOR) and zero the TX_PAD_BOTTOM (4) trailing rows directly
  // after the loop.
  if (height == 4) {
    assert(width > 0 && width % 4 == 0);
    // Padded stride = 4 + 4 = 8; 4 rows of 4 coeffs + 4 zero-pad bytes per
    // iter.
    uint8_t *const ls_end = levels + width * 8;
    do {
      const __m256i c0 = yy_loadu_256(cf);
      const __m256i c1 = yy_loadu_256(cf + 8);
      const __m256i abs01 = _mm256_abs_epi16(_mm256_packs_epi32(c0, c1));
      const __m256i abs01_8 = _mm256_packs_epi16(abs01, y_zeros);
      const __m256i res_ = _mm256_shuffle_epi32(abs01_8, 0xd8);
      const __m256i res = _mm256_permute4x64_epi64(res_, 0xd8);
      yy_storeu_256(ls, res);
      ls += 32;
      cf += 16;
    } while (ls < ls_end);
    // Zero 4 bottom padding rows (4 * 8 = 32 bytes).
    yy_storeu_256(ls, y_zeros);
  } else if (height == 8) {
    assert(width > 0 && width % 4 == 0);
    // Padded stride = 8 + 4 = 12; 4 rows per iteration (4 * 12 = 48 bytes).
    uint8_t *const ls_end = levels + width * 12;
    do {
      const __m256i coeffA = yy_loadu_256(cf);
      const __m256i coeffB = yy_loadu_256(cf + 8);
      const __m256i coeffC = yy_loadu_256(cf + 16);
      const __m256i coeffD = yy_loadu_256(cf + 24);
      const __m256i coeffAB = _mm256_packs_epi32(coeffA, coeffB);
      const __m256i coeffCD = _mm256_packs_epi32(coeffC, coeffD);
      const __m256i absAB = _mm256_abs_epi16(coeffAB);
      const __m256i absCD = _mm256_abs_epi16(coeffCD);
      const __m256i absABCD = _mm256_packs_epi16(absAB, absCD);
      const __m256i res_ = _mm256_permute4x64_epi64(absABCD, 0xd8);
      const __m256i res = _mm256_shuffle_epi32(res_, 0xd8);
      const __m128i res0 = _mm256_castsi256_si128(res);
      const __m128i res1 = _mm256_extracti128_si256(res, 1);

      xx_storel_64(ls, res0);
      xx_storel_32(ls + 8, zero);
      xx_storel_64(ls + 12, _mm_srli_si128(res0, 8));
      xx_storel_32(ls + 20, zero);
      xx_storel_64(ls + 24, res1);
      xx_storel_32(ls + 32, zero);
      xx_storel_64(ls + 36, _mm_srli_si128(res1, 8));
      xx_storel_32(ls + 44, zero);
      cf += 32;
      ls += 48;
    } while (ls < ls_end);
    // Zero 4 bottom padding rows (4 * 12 = 48 bytes).
    yy_storeu_256(ls, y_zeros);
    xx_storeu_128(ls + 32, zero);
  } else if (height == 16) {
    assert(width > 0 && width % 2 == 0);
    // Padded stride = 16 + 4 = 20; 2 rows per iteration (2 * 20 = 40 bytes).
    uint8_t *const ls_end = levels + width * 20;
    do {
      const __m256i coeffA = yy_loadu_256(cf);
      const __m256i coeffB = yy_loadu_256(cf + 8);
      const __m256i coeffC = yy_loadu_256(cf + 16);
      const __m256i coeffD = yy_loadu_256(cf + 24);
      const __m256i coeffAB = _mm256_packs_epi32(coeffA, coeffB);
      const __m256i coeffCD = _mm256_packs_epi32(coeffC, coeffD);
      const __m256i absAB = _mm256_abs_epi16(coeffAB);
      const __m256i absCD = _mm256_abs_epi16(coeffCD);
      const __m256i absABCD = _mm256_packs_epi16(absAB, absCD);
      const __m256i res_ = _mm256_permute4x64_epi64(absABCD, 0xd8);
      const __m256i res = _mm256_shuffle_epi32(res_, 0xd8);

      xx_storeu_128(ls, _mm256_castsi256_si128(res));
      xx_storeu_128(ls + 20, _mm256_extracti128_si256(res, 1));
      cf += 32;
      xx_storel_32(ls + 16, zero);
      xx_storel_32(ls + 36, zero);
      ls += 40;
    } while (ls < ls_end);
    // Zero 4 bottom padding rows (4 * 20 = 80 bytes).
    yy_storeu_256(ls, y_zeros);
    yy_storeu_256(ls + 32, y_zeros);
    xx_storeu_128(ls + 64, zero);
  } else {
    assert(height == 32 && width > 0);
    // Padded stride = 32 + 4 = 36; 1 row per iteration (36 bytes).
    uint8_t *const ls_end = levels + width * 36;
    do {
      const __m256i coeffA = yy_loadu_256(cf);
      const __m256i coeffB = yy_loadu_256(cf + 8);
      const __m256i coeffC = yy_loadu_256(cf + 16);
      const __m256i coeffD = yy_loadu_256(cf + 24);
      const __m256i coeffAB = _mm256_packs_epi32(coeffA, coeffB);
      const __m256i coeffCD = _mm256_packs_epi32(coeffC, coeffD);
      const __m256i absAB = _mm256_abs_epi16(coeffAB);
      const __m256i absCD = _mm256_abs_epi16(coeffCD);
      const __m256i absABCD = _mm256_packs_epi16(absAB, absCD);
      const __m256i res_ = _mm256_permute4x64_epi64(absABCD, 0xd8);
      const __m256i res = _mm256_shuffle_epi32(res_, 0xd8);

      yy_storeu_256(ls, res);
      cf += 32;
      xx_storel_32(ls + 32, zero);
      ls += 36;
    } while (ls < ls_end);
    // Zero 4 bottom padding rows (4 * 36 = 144 bytes).
    yy_storeu_256(ls, y_zeros);
    yy_storeu_256(ls + 32, y_zeros);
    yy_storeu_256(ls + 64, y_zeros);
    yy_storeu_256(ls + 96, y_zeros);
    xx_storeu_128(ls + 128, zero);
  }
}
