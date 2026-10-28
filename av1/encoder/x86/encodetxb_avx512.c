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
#include <string.h>

#include "config/av1_rtcd.h"

#include "aom/aom_integer.h"
#include "aom_dsp/x86/mem_sse2.h"
#include "av1/common/av1_common_int.h"
#include "av1/common/reconinter.h"
#include "av1/common/txb_common.h"
#include "av1/encoder/encodetxb.h"

// Packs 64 tran_low_t values into 64 uint8_t levels in sequential order in a
// single 512-bit ZMM register.
static inline __m512i pack_64_levels_avx512(const tran_low_t *coeff) {
  const __m512i c0 =
      _mm512_abs_epi32(_mm512_loadu_si512((const __m512i *)coeff));
  const __m512i c1 =
      _mm512_abs_epi32(_mm512_loadu_si512((const __m512i *)(coeff + 16)));
  const __m512i c2 =
      _mm512_abs_epi32(_mm512_loadu_si512((const __m512i *)(coeff + 32)));
  const __m512i c3 =
      _mm512_abs_epi32(_mm512_loadu_si512((const __m512i *)(coeff + 48)));

  const __m512i p01 = _mm512_packs_epi32(c0, c1);
  const __m512i p23 = _mm512_packs_epi32(c2, c3);
  const __m512i p0123 = _mm512_packs_epi16(p01, p23);
  const __m512i perm_idx =
      _mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
  return _mm512_permutexvar_epi32(perm_idx, p0123);
}

void av1_txb_init_levels_avx512(const tran_low_t *const coeff, const int width,
                                const int height, uint8_t *const levels) {
  const int stride = height + TX_PAD_HOR;
  if (width * height < 64 || height == 4) {
    av1_txb_init_levels_avx2(coeff, width, height, levels);
    return;
  }

  const __m256i y_zeros = _mm256_setzero_si256();
  const int32_t bottom_len = sizeof(*levels) * (TX_PAD_BOTTOM * stride);
  uint8_t *bottom_buf_end = levels + (width + TX_PAD_BOTTOM) * stride;
  uint8_t *bottom_buf = bottom_buf_end - ((bottom_len + 31) & (~31));
  do {
    _mm256_storeu_si256((__m256i *)bottom_buf, y_zeros);
    bottom_buf += 32;
  } while (bottom_buf < bottom_buf_end);

  uint8_t *ls = levels;
  const tran_low_t *cf = coeff;

  if (height == 32) {
    for (int i = 0; i < width; i += 2) {
      const __m512i lv = pack_64_levels_avx512(cf);
      _mm256_storeu_si256((__m256i *)ls, _mm512_castsi512_si256(lv));
      *(int32_t *)(ls + 32) = 0;
      ls += stride;
      _mm256_storeu_si256((__m256i *)ls, _mm512_extracti64x4_epi64(lv, 1));
      *(int32_t *)(ls + 32) = 0;
      ls += stride;
      cf += 64;
    }
  } else if (height == 16) {
    for (int i = 0; i < width; i += 4) {
      const __m512i lv = pack_64_levels_avx512(cf);
      const __m256i lv_lo = _mm512_castsi512_si256(lv);
      const __m256i lv_hi = _mm512_extracti64x4_epi64(lv, 1);
      _mm_storeu_si128((__m128i *)ls, _mm256_castsi256_si128(lv_lo));
      *(int32_t *)(ls + 16) = 0;
      ls += stride;
      _mm_storeu_si128((__m128i *)ls, _mm256_extracti128_si256(lv_lo, 1));
      *(int32_t *)(ls + 16) = 0;
      ls += stride;
      _mm_storeu_si128((__m128i *)ls, _mm256_castsi256_si128(lv_hi));
      *(int32_t *)(ls + 16) = 0;
      ls += stride;
      _mm_storeu_si128((__m128i *)ls, _mm256_extracti128_si256(lv_hi, 1));
      *(int32_t *)(ls + 16) = 0;
      ls += stride;
      cf += 64;
    }
  } else if (height == 8) {
    for (int i = 0; i < width; i += 8) {
      const __m512i lv = pack_64_levels_avx512(cf);
      const __m256i lv_lo = _mm512_castsi512_si256(lv);
      const __m256i lv_hi = _mm512_extracti64x4_epi64(lv, 1);
      const __m128i r01 = _mm256_castsi256_si128(lv_lo);
      const __m128i r23 = _mm256_extracti128_si256(lv_lo, 1);
      const __m128i r45 = _mm256_castsi256_si128(lv_hi);
      const __m128i r67 = _mm256_extracti128_si256(lv_hi, 1);
      _mm_storel_epi64((__m128i *)(ls + 0 * stride), r01);
      *(int32_t *)(ls + 0 * stride + 8) = 0;
      _mm_storeh_pi((__m64 *)(ls + 1 * stride), _mm_castsi128_ps(r01));
      *(int32_t *)(ls + 1 * stride + 8) = 0;
      _mm_storel_epi64((__m128i *)(ls + 2 * stride), r23);
      *(int32_t *)(ls + 2 * stride + 8) = 0;
      _mm_storeh_pi((__m64 *)(ls + 3 * stride), _mm_castsi128_ps(r23));
      *(int32_t *)(ls + 3 * stride + 8) = 0;
      _mm_storel_epi64((__m128i *)(ls + 4 * stride), r45);
      *(int32_t *)(ls + 4 * stride + 8) = 0;
      _mm_storeh_pi((__m64 *)(ls + 5 * stride), _mm_castsi128_ps(r45));
      *(int32_t *)(ls + 5 * stride + 8) = 0;
      _mm_storel_epi64((__m128i *)(ls + 6 * stride), r67);
      *(int32_t *)(ls + 6 * stride + 8) = 0;
      _mm_storeh_pi((__m64 *)(ls + 7 * stride), _mm_castsi128_ps(r67));
      *(int32_t *)(ls + 7 * stride + 8) = 0;
      ls += 8 * stride;
      cf += 64;
    }
  } else {
    av1_txb_init_levels_avx2(coeff, width, height, levels);
  }
}

static inline __m512i load_8bit_16x4_to_1_reg_avx512(const uint8_t *const src,
                                                     const int stride) {
  const __m128i r0 = _mm_loadu_si128((const __m128i *)(src + 0 * stride));
  const __m128i r1 = _mm_loadu_si128((const __m128i *)(src + 1 * stride));
  const __m128i r2 = _mm_loadu_si128((const __m128i *)(src + 2 * stride));
  const __m128i r3 = _mm_loadu_si128((const __m128i *)(src + 3 * stride));
  const __m256i r01 =
      _mm256_inserti128_si256(_mm256_castsi128_si256(r0), r1, 1);
  const __m256i r23 =
      _mm256_inserti128_si256(_mm256_castsi128_si256(r2), r3, 1);
  return _mm512_inserti64x4(_mm512_castsi256_si512(r01), r23, 1);
}

static inline __m512i load_8bit_32x2_to_1_reg_avx512(const uint8_t *const src,
                                                     const int stride) {
  const __m256i r0 = _mm256_loadu_si256((const __m256i *)(src + 0 * stride));
  const __m256i r1 = _mm256_loadu_si256((const __m256i *)(src + 1 * stride));
  return _mm512_inserti64x4(_mm512_castsi256_si512(r0), r1, 1);
}

static inline __m512i load_8bit_8x8_to_1_reg_avx512(const uint8_t *const src,
                                                    const int stride) {
  const __m128i r01 = load_8bit_8x2_to_1_reg_sse2(src + 0 * stride, stride);
  const __m128i r23 = load_8bit_8x2_to_1_reg_sse2(src + 2 * stride, stride);
  const __m128i r45 = load_8bit_8x2_to_1_reg_sse2(src + 4 * stride, stride);
  const __m128i r67 = load_8bit_8x2_to_1_reg_sse2(src + 6 * stride, stride);
  const __m256i r0123 =
      _mm256_inserti128_si256(_mm256_castsi128_si256(r01), r23, 1);
  const __m256i r4567 =
      _mm256_inserti128_si256(_mm256_castsi128_si256(r45), r67, 1);
  return _mm512_inserti64x4(_mm512_castsi256_si512(r0123), r4567, 1);
}

static inline void load_levels_16x4x5_avx512(const uint8_t *const src,
                                             const int stride,
                                             const ptrdiff_t *const offsets,
                                             __m512i *const level) {
  level[0] = load_8bit_16x4_to_1_reg_avx512(src + 1, stride);
  level[1] = load_8bit_16x4_to_1_reg_avx512(src + stride, stride);
  level[2] = load_8bit_16x4_to_1_reg_avx512(src + offsets[0], stride);
  level[3] = load_8bit_16x4_to_1_reg_avx512(src + offsets[1], stride);
  level[4] = load_8bit_16x4_to_1_reg_avx512(src + offsets[2], stride);
}

static inline void load_levels_32x2x5_avx512(const uint8_t *const src,
                                             const int stride,
                                             const ptrdiff_t *const offsets,
                                             __m512i *const level) {
  level[0] = load_8bit_32x2_to_1_reg_avx512(src + 1, stride);
  level[1] = load_8bit_32x2_to_1_reg_avx512(src + stride, stride);
  level[2] = load_8bit_32x2_to_1_reg_avx512(src + offsets[0], stride);
  level[3] = load_8bit_32x2_to_1_reg_avx512(src + offsets[1], stride);
  level[4] = load_8bit_32x2_to_1_reg_avx512(src + offsets[2], stride);
}

static inline void load_levels_8x8x5_avx512(const uint8_t *const src,
                                            const int stride,
                                            const ptrdiff_t *const offsets,
                                            __m512i *const level) {
  level[0] = load_8bit_8x8_to_1_reg_avx512(src + 1, stride);
  level[1] = load_8bit_8x8_to_1_reg_avx512(src + stride, stride);
  level[2] = load_8bit_8x8_to_1_reg_avx512(src + offsets[0], stride);
  level[3] = load_8bit_8x8_to_1_reg_avx512(src + offsets[1], stride);
  level[4] = load_8bit_8x8_to_1_reg_avx512(src + offsets[2], stride);
}

static inline __m512i get_coeff_contexts_kernel_avx512(__m512i *const level) {
  const __m512i const_3 = _mm512_set1_epi8(3);
  const __m512i const_4 = _mm512_set1_epi8(4);
  __m512i count = _mm512_min_epu8(level[0], const_3);
  level[1] = _mm512_min_epu8(level[1], const_3);
  level[2] = _mm512_min_epu8(level[2], const_3);
  level[3] = _mm512_min_epu8(level[3], const_3);
  level[4] = _mm512_min_epu8(level[4], const_3);
  count = _mm512_add_epi8(count, level[1]);
  count = _mm512_add_epi8(count, level[2]);
  count = _mm512_add_epi8(count, level[3]);
  count = _mm512_add_epi8(count, level[4]);
  count = _mm512_avg_epu8(count, _mm512_setzero_si512());
  count = _mm512_min_epu8(count, const_4);
  return count;
}

static inline __m512i pack_4x128_to_512(__m128i a, __m128i b, __m128i c,
                                        __m128i d) {
  const __m256i ab = _mm256_inserti128_si256(_mm256_castsi128_si256(a), b, 1);
  const __m256i cd = _mm256_inserti128_si256(_mm256_castsi128_si256(c), d, 1);
  return _mm512_inserti64x4(_mm512_castsi256_si512(ab), cd, 1);
}

static inline void get_8_coeff_contexts_2d_avx512(
    const uint8_t *levels, const int width, const ptrdiff_t *const offsets,
    int8_t *coeff_contexts) {
  const int stride = 8 + TX_PAD_HOR;
  int8_t *cc = coeff_contexts;
  int col = width;
  __m512i level[5];
  __m128i p0, p1;
  const __m128i p2 = _mm_set1_epi8(21);

  if (width == 8) {
    p0 = _mm_setr_epi8(0, 1, 6, 6, 21, 21, 21, 21, 1, 6, 6, 21, 21, 21, 21, 21);
    p1 = _mm_setr_epi8(6, 6, 21, 21, 21, 21, 21, 21, 6, 21, 21, 21, 21, 21, 21,
                       21);
  } else {
    p0 = _mm_setr_epi8(0, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
                       16, 16);
    p1 = _mm_setr_epi8(6, 6, 21, 21, 21, 21, 21, 21, 6, 21, 21, 21, 21, 21, 21,
                       21);
  }

  __m512i pos_to_offset = pack_4x128_to_512(p0, p1, p2, p2);
  const __m512i pos_to_offset_large = _mm512_set1_epi8(21);

  do {
    load_levels_8x8x5_avx512(levels, stride, offsets, level);
    __m512i count = get_coeff_contexts_kernel_avx512(level);
    count = _mm512_add_epi8(count, pos_to_offset);
    _mm512_storeu_si512((__m512i *)cc, count);
    pos_to_offset = pos_to_offset_large;
    levels += 8 * stride;
    cc += 64;
    col -= 8;
  } while (col > 0);

  coeff_contexts[0] = 0;
}

static inline void get_8_coeff_contexts_ver_avx512(
    const uint8_t *levels, const int width, const ptrdiff_t *const offsets,
    int8_t *coeff_contexts) {
  const int stride = 8 + TX_PAD_HOR;
  const __m128i p0 =
      _mm_setr_epi8(SIG_COEF_CONTEXTS_2D + 0, SIG_COEF_CONTEXTS_2D + 5,
                    SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
                    SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
                    SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
                    SIG_COEF_CONTEXTS_2D + 0, SIG_COEF_CONTEXTS_2D + 5,
                    SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
                    SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
                    SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10);
  const __m512i pos_to_offset = _mm512_broadcast_i32x4(p0);
  int col = width;
  __m512i level[5];

  do {
    load_levels_8x8x5_avx512(levels, stride, offsets, level);
    __m512i count = get_coeff_contexts_kernel_avx512(level);
    count = _mm512_add_epi8(count, pos_to_offset);
    _mm512_storeu_si512((__m512i *)coeff_contexts, count);
    levels += 8 * stride;
    coeff_contexts += 64;
    col -= 8;
  } while (col > 0);
}

static inline void get_8_coeff_contexts_hor_avx512(
    const uint8_t *levels, const int width, const ptrdiff_t *const offsets,
    int8_t *coeff_contexts) {
  const int stride = 8 + TX_PAD_HOR;
  const __m128i p0 =
      _mm_setr_epi8(SIG_COEF_CONTEXTS_2D + 0, SIG_COEF_CONTEXTS_2D + 0,
                    SIG_COEF_CONTEXTS_2D + 0, SIG_COEF_CONTEXTS_2D + 0,
                    SIG_COEF_CONTEXTS_2D + 0, SIG_COEF_CONTEXTS_2D + 0,
                    SIG_COEF_CONTEXTS_2D + 0, SIG_COEF_CONTEXTS_2D + 0,
                    SIG_COEF_CONTEXTS_2D + 5, SIG_COEF_CONTEXTS_2D + 5,
                    SIG_COEF_CONTEXTS_2D + 5, SIG_COEF_CONTEXTS_2D + 5,
                    SIG_COEF_CONTEXTS_2D + 5, SIG_COEF_CONTEXTS_2D + 5,
                    SIG_COEF_CONTEXTS_2D + 5, SIG_COEF_CONTEXTS_2D + 5);
  const __m128i p_large = _mm_set1_epi8(SIG_COEF_CONTEXTS_2D + 10);
  __m512i pos_to_offset = pack_4x128_to_512(p0, p_large, p_large, p_large);
  const __m512i pos_to_offset_large =
      _mm512_set1_epi8(SIG_COEF_CONTEXTS_2D + 10);
  int col = width;
  __m512i level[5];

  do {
    load_levels_8x8x5_avx512(levels, stride, offsets, level);
    __m512i count = get_coeff_contexts_kernel_avx512(level);
    count = _mm512_add_epi8(count, pos_to_offset);
    _mm512_storeu_si512((__m512i *)coeff_contexts, count);
    pos_to_offset = pos_to_offset_large;
    levels += 8 * stride;
    coeff_contexts += 64;
    col -= 8;
  } while (col > 0);
}

static inline void get_16_coeff_contexts_2d_avx512(
    const uint8_t *levels, const int real_width, const int real_height,
    const int width, const ptrdiff_t *const offsets, int8_t *coeff_contexts) {
  const int stride = 16 + TX_PAD_HOR;
  int8_t *cc = coeff_contexts;
  int col = width;
  __m512i level[5];
  __m512i pos_to_offset;
  __m512i pos_to_offset_large;

  if (real_width == real_height) {
    const __m128i p0 = _mm_setr_epi8(0, 1, 6, 6, 21, 21, 21, 21, 21, 21, 21, 21,
                                     21, 21, 21, 21);
    const __m128i p1 = _mm_setr_epi8(1, 6, 6, 21, 21, 21, 21, 21, 21, 21, 21,
                                     21, 21, 21, 21, 21);
    const __m128i p2 = _mm_setr_epi8(6, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21,
                                     21, 21, 21, 21, 21);
    const __m128i p3 = _mm_setr_epi8(6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
                                     21, 21, 21, 21, 21);
    pos_to_offset = pack_4x128_to_512(p0, p1, p2, p3);
    pos_to_offset_large = _mm512_set1_epi8(21);
  } else if (real_width < real_height) {
    const __m128i p0 = _mm_setr_epi8(0, 11, 6, 6, 21, 21, 21, 21, 21, 21, 21,
                                     21, 21, 21, 21, 21);
    const __m128i p1 = _mm_setr_epi8(11, 11, 6, 21, 21, 21, 21, 21, 21, 21, 21,
                                     21, 21, 21, 21, 21);
    const __m128i p2 = _mm_setr_epi8(11, 11, 21, 21, 21, 21, 21, 21, 21, 21, 21,
                                     21, 21, 21, 21, 21);
    pos_to_offset = pack_4x128_to_512(p0, p1, p2, p2);
    pos_to_offset_large = _mm512_broadcast_i32x4(p2);
  } else {
    const __m128i p0 = _mm_set1_epi8(16);
    const __m128i p2 = _mm_setr_epi8(6, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21,
                                     21, 21, 21, 21, 21);
    const __m128i p3 = _mm_setr_epi8(6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
                                     21, 21, 21, 21, 21);
    pos_to_offset = pack_4x128_to_512(p0, p0, p2, p3);
    pos_to_offset_large = _mm512_set1_epi8(21);
  }

  do {
    load_levels_16x4x5_avx512(levels, stride, offsets, level);
    __m512i count = get_coeff_contexts_kernel_avx512(level);
    count = _mm512_add_epi8(count, pos_to_offset);
    _mm512_storeu_si512((__m512i *)cc, count);
    pos_to_offset = pos_to_offset_large;
    levels += 4 * stride;
    cc += 64;
    col -= 4;
  } while (col > 0);

  coeff_contexts[0] = 0;
}

static inline void get_16_coeff_contexts_ver_avx512(
    const uint8_t *levels, const int width, const ptrdiff_t *const offsets,
    int8_t *coeff_contexts) {
  const int stride = 16 + TX_PAD_HOR;
  const __m128i p0 =
      _mm_setr_epi8(SIG_COEF_CONTEXTS_2D + 0, SIG_COEF_CONTEXTS_2D + 5,
                    SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
                    SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
                    SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
                    SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
                    SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
                    SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
                    SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10);
  const __m512i pos_to_offset = _mm512_broadcast_i32x4(p0);
  int col = width;
  __m512i level[5];

  do {
    load_levels_16x4x5_avx512(levels, stride, offsets, level);
    __m512i count = get_coeff_contexts_kernel_avx512(level);
    count = _mm512_add_epi8(count, pos_to_offset);
    _mm512_storeu_si512((__m512i *)coeff_contexts, count);
    levels += 4 * stride;
    coeff_contexts += 64;
    col -= 4;
  } while (col > 0);
}

static inline void get_16_coeff_contexts_hor_avx512(
    const uint8_t *levels, const int width, const ptrdiff_t *const offsets,
    int8_t *coeff_contexts) {
  const int stride = 16 + TX_PAD_HOR;
  const __m128i p0 = _mm_set1_epi8(SIG_COEF_CONTEXTS_2D + 0);
  const __m128i p1 = _mm_set1_epi8(SIG_COEF_CONTEXTS_2D + 5);
  const __m128i p2 = _mm_set1_epi8(SIG_COEF_CONTEXTS_2D + 10);
  __m512i pos_to_offset = pack_4x128_to_512(p0, p1, p2, p2);
  const __m512i pos_to_offset_large =
      _mm512_set1_epi8(SIG_COEF_CONTEXTS_2D + 10);
  int col = width;
  __m512i level[5];

  do {
    load_levels_16x4x5_avx512(levels, stride, offsets, level);
    __m512i count = get_coeff_contexts_kernel_avx512(level);
    count = _mm512_add_epi8(count, pos_to_offset);
    _mm512_storeu_si512((__m512i *)coeff_contexts, count);
    pos_to_offset = pos_to_offset_large;
    levels += 4 * stride;
    coeff_contexts += 64;
    col -= 4;
  } while (col > 0);
}

static inline void get_32_coeff_contexts_2d_avx512(
    const uint8_t *levels, const int real_width, const int real_height,
    const int width, const ptrdiff_t *const offsets, int8_t *coeff_contexts) {
  const int stride = 32 + TX_PAD_HOR;
  int8_t *cc = coeff_contexts;
  int col = width;
  __m512i level[5];
  __m512i pos_to_offset[3];
  const __m128i p_21 = _mm_set1_epi8(21);

  if (real_width == real_height) {
    const __m128i p0 = _mm_setr_epi8(0, 1, 6, 6, 21, 21, 21, 21, 21, 21, 21, 21,
                                     21, 21, 21, 21);
    const __m128i p1 = _mm_setr_epi8(1, 6, 6, 21, 21, 21, 21, 21, 21, 21, 21,
                                     21, 21, 21, 21, 21);
    const __m128i p2 = _mm_setr_epi8(6, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21,
                                     21, 21, 21, 21, 21);
    const __m128i p3 = _mm_setr_epi8(6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
                                     21, 21, 21, 21, 21);
    pos_to_offset[0] = pack_4x128_to_512(p0, p_21, p1, p_21);
    pos_to_offset[1] = pack_4x128_to_512(p2, p_21, p3, p_21);
    pos_to_offset[2] = _mm512_set1_epi8(21);
  } else if (real_width < real_height) {
    const __m128i p0 = _mm_setr_epi8(0, 11, 6, 6, 21, 21, 21, 21, 21, 21, 21,
                                     21, 21, 21, 21, 21);
    const __m128i p1 = _mm_setr_epi8(11, 11, 6, 21, 21, 21, 21, 21, 21, 21, 21,
                                     21, 21, 21, 21, 21);
    const __m128i p2 = _mm_setr_epi8(11, 11, 21, 21, 21, 21, 21, 21, 21, 21, 21,
                                     21, 21, 21, 21, 21);
    pos_to_offset[0] = pack_4x128_to_512(p0, p_21, p1, p_21);
    pos_to_offset[1] = pos_to_offset[2] = pack_4x128_to_512(p2, p_21, p2, p_21);
  } else {
    const __m128i p_16 = _mm_set1_epi8(16);
    const __m128i p2 = _mm_setr_epi8(6, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21,
                                     21, 21, 21, 21, 21);
    const __m128i p3 = _mm_setr_epi8(6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
                                     21, 21, 21, 21, 21);
    pos_to_offset[0] = pack_4x128_to_512(p_16, p_16, p_16, p_16);
    pos_to_offset[1] = pack_4x128_to_512(p2, p_21, p3, p_21);
    pos_to_offset[2] = _mm512_set1_epi8(21);
  }

  do {
    load_levels_32x2x5_avx512(levels, stride, offsets, level);
    __m512i count = get_coeff_contexts_kernel_avx512(level);
    count = _mm512_add_epi8(count, pos_to_offset[0]);
    _mm512_storeu_si512((__m512i *)cc, count);
    pos_to_offset[0] = pos_to_offset[1];
    pos_to_offset[1] = pos_to_offset[2];
    levels += 2 * stride;
    cc += 64;
    col -= 2;
  } while (col > 0);

  coeff_contexts[0] = 0;
}

static inline void get_32_coeff_contexts_ver_avx512(
    const uint8_t *levels, const int width, const ptrdiff_t *const offsets,
    int8_t *coeff_contexts) {
  const int stride = 32 + TX_PAD_HOR;
  const __m128i p0 =
      _mm_setr_epi8(SIG_COEF_CONTEXTS_2D + 0, SIG_COEF_CONTEXTS_2D + 5,
                    SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
                    SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
                    SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
                    SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
                    SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
                    SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
                    SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10);
  const __m128i p1 = _mm_set1_epi8(SIG_COEF_CONTEXTS_2D + 10);
  const __m512i pos_to_offset = pack_4x128_to_512(p0, p1, p0, p1);
  int col = width;
  __m512i level[5];

  do {
    load_levels_32x2x5_avx512(levels, stride, offsets, level);
    __m512i count = get_coeff_contexts_kernel_avx512(level);
    count = _mm512_add_epi8(count, pos_to_offset);
    _mm512_storeu_si512((__m512i *)coeff_contexts, count);
    levels += 2 * stride;
    coeff_contexts += 64;
    col -= 2;
  } while (col > 0);
}

static inline void get_32_coeff_contexts_hor_avx512(
    const uint8_t *levels, const int width, const ptrdiff_t *const offsets,
    int8_t *coeff_contexts) {
  const int stride = 32 + TX_PAD_HOR;
  const __m128i p0 = _mm_set1_epi8(SIG_COEF_CONTEXTS_2D + 0);
  const __m128i p1 = _mm_set1_epi8(SIG_COEF_CONTEXTS_2D + 5);
  __m512i pos_to_offset = pack_4x128_to_512(p0, p0, p1, p1);
  const __m512i pos_to_offset_large =
      _mm512_set1_epi8(SIG_COEF_CONTEXTS_2D + 10);
  int col = width;
  __m512i level[5];

  do {
    load_levels_32x2x5_avx512(levels, stride, offsets, level);
    __m512i count = get_coeff_contexts_kernel_avx512(level);
    count = _mm512_add_epi8(count, pos_to_offset);
    _mm512_storeu_si512((__m512i *)coeff_contexts, count);
    pos_to_offset = pos_to_offset_large;
    levels += 2 * stride;
    coeff_contexts += 64;
    col -= 2;
  } while (col > 0);
}

void av1_get_nz_map_contexts_avx512(const uint8_t *const levels,
                                    const int16_t *const scan, const int eob,
                                    const TX_SIZE tx_size,
                                    const TX_CLASS tx_class,
                                    int8_t *const coeff_contexts) {
  const int last_idx = eob - 1;
  if (!last_idx) {
    coeff_contexts[0] = 0;
    return;
  }

  const int real_width = tx_size_wide[tx_size];
  const int real_height = tx_size_high[tx_size];
  const int width = get_txb_wide(tx_size);
  const int height = get_txb_high(tx_size);
  const int stride = height + TX_PAD_HOR;
  ptrdiff_t offsets[3];

  /* coeff_contexts must be 16 byte aligned. */
  assert(!((intptr_t)coeff_contexts & 0xf));

  if (tx_class == TX_CLASS_2D) {
    offsets[0] = 0 * stride + 2;
    offsets[1] = 1 * stride + 1;
    offsets[2] = 2 * stride + 0;

    if (height == 8 && width >= 8) {
      get_8_coeff_contexts_2d_avx512(levels, width, offsets, coeff_contexts);
    } else if (height == 16) {
      get_16_coeff_contexts_2d_avx512(levels, real_width, real_height, width,
                                      offsets, coeff_contexts);
    } else if (height == 32) {
      get_32_coeff_contexts_2d_avx512(levels, real_width, real_height, width,
                                      offsets, coeff_contexts);
    } else {
      av1_get_nz_map_contexts_sse2(levels, scan, eob, tx_size, tx_class,
                                   coeff_contexts);
      return;
    }
  } else if (tx_class == TX_CLASS_HORIZ) {
    offsets[0] = 2 * stride;
    offsets[1] = 3 * stride;
    offsets[2] = 4 * stride;

    if (height == 8 && width >= 8) {
      get_8_coeff_contexts_hor_avx512(levels, width, offsets, coeff_contexts);
    } else if (height == 16) {
      get_16_coeff_contexts_hor_avx512(levels, width, offsets, coeff_contexts);
    } else if (height == 32) {
      get_32_coeff_contexts_hor_avx512(levels, width, offsets, coeff_contexts);
    } else {
      av1_get_nz_map_contexts_sse2(levels, scan, eob, tx_size, tx_class,
                                   coeff_contexts);
      return;
    }
  } else {
    offsets[0] = 2;
    offsets[1] = 3;
    offsets[2] = 4;

    if (height == 8 && width >= 8) {
      get_8_coeff_contexts_ver_avx512(levels, width, offsets, coeff_contexts);
    } else if (height == 16) {
      get_16_coeff_contexts_ver_avx512(levels, width, offsets, coeff_contexts);
    } else if (height == 32) {
      get_32_coeff_contexts_ver_avx512(levels, width, offsets, coeff_contexts);
    } else {
      av1_get_nz_map_contexts_sse2(levels, scan, eob, tx_size, tx_class,
                                   coeff_contexts);
      return;
    }
  }

  const int bhl = get_txb_bhl(tx_size);
  const int pos = scan[last_idx];
  if (last_idx <= (width << bhl) / 8)
    coeff_contexts[pos] = 1;
  else if (last_idx <= (width << bhl) / 4)
    coeff_contexts[pos] = 2;
  else
    coeff_contexts[pos] = 3;
}

uint64_t av1_wedge_sse_from_residuals_avx512(const int16_t *r1,
                                             const int16_t *d, const uint8_t *m,
                                             int N) {
  assert(N % 64 == 0);
  const __m512i v_max_mask = _mm512_set1_epi16(1 << WEDGE_WEIGHT_BITS);
  __m512i v_acc0 = _mm512_setzero_si512();
  __m512i v_acc1 = _mm512_setzero_si512();

  for (int i = 0; i < N; i += 64) {
    const __m512i r1_l = _mm512_loadu_si512((const __m512i *)(r1 + i));
    const __m512i r1_h = _mm512_loadu_si512((const __m512i *)(r1 + i + 32));
    const __m512i d_l = _mm512_loadu_si512((const __m512i *)(d + i));
    const __m512i d_h = _mm512_loadu_si512((const __m512i *)(d + i + 32));
    const __m512i m_l =
        _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *)(m + i)));
    const __m512i m_h =
        _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *)(m + i + 32)));

    const __m512i rd_ll = _mm512_unpacklo_epi16(d_l, r1_l);
    const __m512i rd_lh = _mm512_unpackhi_epi16(d_l, r1_l);
    const __m512i m_ll = _mm512_unpacklo_epi16(m_l, v_max_mask);
    const __m512i m_lh = _mm512_unpackhi_epi16(m_l, v_max_mask);
    const __m512i t_l = _mm512_packs_epi32(_mm512_madd_epi16(rd_ll, m_ll),
                                           _mm512_madd_epi16(rd_lh, m_lh));

    const __m512i rd_hl = _mm512_unpacklo_epi16(d_h, r1_h);
    const __m512i rd_hh = _mm512_unpackhi_epi16(d_h, r1_h);
    const __m512i m_hl = _mm512_unpacklo_epi16(m_h, v_max_mask);
    const __m512i m_hh = _mm512_unpackhi_epi16(m_h, v_max_mask);
    const __m512i t_h = _mm512_packs_epi32(_mm512_madd_epi16(rd_hl, m_hl),
                                           _mm512_madd_epi16(rd_hh, m_hh));

    const __m512i sq_l = _mm512_madd_epi16(t_l, t_l);
    const __m512i sq_h = _mm512_madd_epi16(t_h, t_h);

    v_acc0 = _mm512_add_epi64(
        v_acc0, _mm512_cvtepu32_epi64(_mm512_castsi512_si256(sq_l)));
    v_acc1 = _mm512_add_epi64(
        v_acc1, _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(sq_l, 1)));
    v_acc0 = _mm512_add_epi64(
        v_acc0, _mm512_cvtepu32_epi64(_mm512_castsi512_si256(sq_h)));
    v_acc1 = _mm512_add_epi64(
        v_acc1, _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(sq_h, 1)));
  }

  const uint64_t csse =
      (uint64_t)_mm512_reduce_add_epi64(_mm512_add_epi64(v_acc0, v_acc1));
  return ROUND_POWER_OF_TWO(csse, 2 * WEDGE_WEIGHT_BITS);
}

int8_t av1_wedge_sign_from_residuals_avx512(const int16_t *ds, const uint8_t *m,
                                            int N, int64_t limit) {
  assert(N % 64 == 0);
  __m512i acc0 = _mm512_setzero_si512();
  __m512i acc1 = _mm512_setzero_si512();

  for (int i = 0; i < N; i += 64) {
    const __m512i ds_l = _mm512_loadu_si512((const __m512i *)(ds + i));
    const __m512i ds_h = _mm512_loadu_si512((const __m512i *)(ds + i + 32));
    const __m512i m_l =
        _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *)(m + i)));
    const __m512i m_h =
        _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *)(m + i + 32)));

    acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(ds_l, m_l));
    acc1 = _mm512_add_epi32(acc1, _mm512_madd_epi16(ds_h, m_h));
  }

  const int64_t sign_sum =
      (int64_t)_mm512_reduce_add_epi32(_mm512_add_epi32(acc0, acc1));
  return sign_sum > limit;
}

void av1_wedge_compute_delta_squares_avx512(int16_t *d, const int16_t *a,
                                            const int16_t *b, int N) {
  assert(N % 64 == 0);
  const __m512i zero = _mm512_setzero_si512();
  for (int i = 0; i < N; i += 64) {
    const __m512i a_l = _mm512_loadu_si512((const __m512i *)(a + i));
    const __m512i a_h = _mm512_loadu_si512((const __m512i *)(a + i + 32));
    const __m512i b_l = _mm512_loadu_si512((const __m512i *)(b + i));
    const __m512i b_h = _mm512_loadu_si512((const __m512i *)(b + i + 32));

    const __m512i nb_l = _mm512_subs_epi16(zero, b_l);
    const __m512i nb_h = _mm512_subs_epi16(zero, b_h);

    const __m512i r0 = _mm512_madd_epi16(_mm512_unpacklo_epi16(a_l, b_l),
                                         _mm512_unpacklo_epi16(a_l, nb_l));
    const __m512i r1 = _mm512_madd_epi16(_mm512_unpackhi_epi16(a_l, b_l),
                                         _mm512_unpackhi_epi16(a_l, nb_l));
    const __m512i r2 = _mm512_madd_epi16(_mm512_unpacklo_epi16(a_h, b_h),
                                         _mm512_unpacklo_epi16(a_h, nb_h));
    const __m512i r3 = _mm512_madd_epi16(_mm512_unpackhi_epi16(a_h, b_h),
                                         _mm512_unpackhi_epi16(a_h, nb_h));

    const __m512i res_l = _mm512_packs_epi32(r0, r1);
    const __m512i res_h = _mm512_packs_epi32(r2, r3);

    _mm512_storeu_si512((__m512i *)(d + i), res_l);
    _mm512_storeu_si512((__m512i *)(d + i + 32), res_h);
  }
}
