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
#include <stdlib.h>

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"
#include "aom/aom_integer.h"
#include "aom_dsp/x86/bitdepth_conversion_avx2.h"
#include "aom_ports/mem.h"

// Stores 32 int16_t values from a 512-bit register into 32 tran_low_t (int32_t)
// slots with the exact same lane layout as two back-to-back AVX2
// store_tran_low() calls.
static inline void store_tran_low_avx512(__m512i a, tran_low_t *b) {
  const __m512i a_hi = _mm512_srai_epi16(a, 15);
  const __m512i a_1 = _mm512_unpacklo_epi16(a, a_hi);
  const __m512i a_2 = _mm512_unpackhi_epi16(a, a_hi);
  const __m512i out0 = _mm512_shuffle_i64x2(a_1, a_2, 0x44);
  const __m512i out1 = _mm512_shuffle_i64x2(a_1, a_2, 0xEE);
  _mm512_storeu_si512((__m512i *)b, out0);
  _mm512_storeu_si512((__m512i *)(b + 16), out1);
}

static inline void sign_extend_16bit_to_32bit_avx512(__m512i in,
                                                     __m512i *out_lo,
                                                     __m512i *out_hi) {
  const __m512i sign_bits = _mm512_srai_epi16(in, 15);
  *out_lo = _mm512_unpacklo_epi16(in, sign_bits);
  *out_hi = _mm512_unpackhi_epi16(in, sign_bits);
}

// Processes four 8x8 16-bit Hadamard blocks simultaneously across the four
// 128-bit lanes of in[0..7].
static inline void hadamard_col8x4_avx512(__m512i *in, int iter) {
  __m512i a0 = in[0];
  __m512i a1 = in[1];
  __m512i a2 = in[2];
  __m512i a3 = in[3];
  __m512i a4 = in[4];
  __m512i a5 = in[5];
  __m512i a6 = in[6];
  __m512i a7 = in[7];

  __m512i b0 = _mm512_add_epi16(a0, a1);
  __m512i b1 = _mm512_sub_epi16(a0, a1);
  __m512i b2 = _mm512_add_epi16(a2, a3);
  __m512i b3 = _mm512_sub_epi16(a2, a3);
  __m512i b4 = _mm512_add_epi16(a4, a5);
  __m512i b5 = _mm512_sub_epi16(a4, a5);
  __m512i b6 = _mm512_add_epi16(a6, a7);
  __m512i b7 = _mm512_sub_epi16(a6, a7);

  a0 = _mm512_add_epi16(b0, b2);
  a1 = _mm512_add_epi16(b1, b3);
  a2 = _mm512_sub_epi16(b0, b2);
  a3 = _mm512_sub_epi16(b1, b3);
  a4 = _mm512_add_epi16(b4, b6);
  a5 = _mm512_add_epi16(b5, b7);
  a6 = _mm512_sub_epi16(b4, b6);
  a7 = _mm512_sub_epi16(b5, b7);

  if (iter == 0) {
    b0 = _mm512_add_epi16(a0, a4);
    b7 = _mm512_add_epi16(a1, a5);
    b3 = _mm512_add_epi16(a2, a6);
    b4 = _mm512_add_epi16(a3, a7);
    b2 = _mm512_sub_epi16(a0, a4);
    b6 = _mm512_sub_epi16(a1, a5);
    b1 = _mm512_sub_epi16(a2, a6);
    b5 = _mm512_sub_epi16(a3, a7);

    a0 = _mm512_unpacklo_epi16(b0, b1);
    a1 = _mm512_unpacklo_epi16(b2, b3);
    a2 = _mm512_unpackhi_epi16(b0, b1);
    a3 = _mm512_unpackhi_epi16(b2, b3);
    a4 = _mm512_unpacklo_epi16(b4, b5);
    a5 = _mm512_unpacklo_epi16(b6, b7);
    a6 = _mm512_unpackhi_epi16(b4, b5);
    a7 = _mm512_unpackhi_epi16(b6, b7);

    b0 = _mm512_unpacklo_epi32(a0, a1);
    b1 = _mm512_unpacklo_epi32(a4, a5);
    b2 = _mm512_unpackhi_epi32(a0, a1);
    b3 = _mm512_unpackhi_epi32(a4, a5);
    b4 = _mm512_unpacklo_epi32(a2, a3);
    b5 = _mm512_unpacklo_epi32(a6, a7);
    b6 = _mm512_unpackhi_epi32(a2, a3);
    b7 = _mm512_unpackhi_epi32(a6, a7);

    in[0] = _mm512_unpacklo_epi64(b0, b1);
    in[1] = _mm512_unpackhi_epi64(b0, b1);
    in[2] = _mm512_unpacklo_epi64(b2, b3);
    in[3] = _mm512_unpackhi_epi64(b2, b3);
    in[4] = _mm512_unpacklo_epi64(b4, b5);
    in[5] = _mm512_unpackhi_epi64(b4, b5);
    in[6] = _mm512_unpacklo_epi64(b6, b7);
    in[7] = _mm512_unpackhi_epi64(b6, b7);
  } else {
    in[0] = _mm512_add_epi16(a0, a4);
    in[7] = _mm512_add_epi16(a1, a5);
    in[3] = _mm512_add_epi16(a2, a6);
    in[4] = _mm512_add_epi16(a3, a7);
    in[2] = _mm512_sub_epi16(a0, a4);
    in[6] = _mm512_sub_epi16(a1, a5);
    in[1] = _mm512_sub_epi16(a2, a6);
    in[5] = _mm512_sub_epi16(a3, a7);
  }
}

void aom_hadamard_lp_8x8_dual_avx512(const int16_t *src_diff,
                                     ptrdiff_t src_stride, int16_t *coeff) {
  aom_hadamard_lp_8x8_dual_avx2(src_diff, src_stride, coeff);
}

static inline void load_row_pair_16x16(const int16_t *r_top,
                                       const int16_t *r_bot, __m512i *out) {
  const __m256i top = _mm256_loadu_si256((const __m256i *)r_top);
  const __m256i bot = _mm256_loadu_si256((const __m256i *)r_bot);
  *out = _mm512_inserti64x4(_mm512_castsi256_si512(top), bot, 1);
}

// Transposes 4x4 128-bit lanes across 4 ZMM registers so c0..c3 hold 32
// contiguous int16_t coefficients from quadrants 0..3 respectively, then runs
// the 2x2 inter-quadrant Hadamard butterfly in-register.
static inline void hadamard_16x16_quadrant_butterfly_avx512(
    __m512i s0, __m512i s1, __m512i s2, __m512i s3, __m512i *out0,
    __m512i *out1, __m512i *out2, __m512i *out3) {
  const __m512i t0 = _mm512_shuffle_i64x2(s0, s1, 0x44);
  const __m512i t1 = _mm512_shuffle_i64x2(s2, s3, 0x44);
  const __m512i t2 = _mm512_shuffle_i64x2(s0, s1, 0xEE);
  const __m512i t3 = _mm512_shuffle_i64x2(s2, s3, 0xEE);

  const __m512i coeff0 = _mm512_shuffle_i64x2(t0, t1, 0x88);
  const __m512i coeff1 = _mm512_shuffle_i64x2(t0, t1, 0xDD);
  const __m512i coeff2 = _mm512_shuffle_i64x2(t2, t3, 0x88);
  const __m512i coeff3 = _mm512_shuffle_i64x2(t2, t3, 0xDD);

  __m512i b0 = _mm512_srai_epi16(_mm512_add_epi16(coeff0, coeff1), 1);
  __m512i b1 = _mm512_srai_epi16(_mm512_sub_epi16(coeff0, coeff1), 1);
  __m512i b2 = _mm512_srai_epi16(_mm512_add_epi16(coeff2, coeff3), 1);
  __m512i b3 = _mm512_srai_epi16(_mm512_sub_epi16(coeff2, coeff3), 1);

  *out0 = _mm512_add_epi16(b0, b2);
  *out1 = _mm512_add_epi16(b1, b3);
  *out2 = _mm512_sub_epi16(b0, b2);
  *out3 = _mm512_sub_epi16(b1, b3);
}

static inline void hadamard_16x16_avx512(const int16_t *src_diff,
                                         ptrdiff_t src_stride,
                                         tran_low_t *coeff, int is_final) {
  __m512i src[8];
  const int16_t *src_top = src_diff;
  const int16_t *src_bot = src_diff + 8 * src_stride;

  load_row_pair_16x16(src_top, src_bot, &src[0]);
  load_row_pair_16x16(src_top += src_stride, src_bot += src_stride, &src[1]);
  load_row_pair_16x16(src_top += src_stride, src_bot += src_stride, &src[2]);
  load_row_pair_16x16(src_top += src_stride, src_bot += src_stride, &src[3]);
  load_row_pair_16x16(src_top += src_stride, src_bot += src_stride, &src[4]);
  load_row_pair_16x16(src_top += src_stride, src_bot += src_stride, &src[5]);
  load_row_pair_16x16(src_top += src_stride, src_bot += src_stride, &src[6]);
  load_row_pair_16x16(src_top + src_stride, src_bot + src_stride, &src[7]);

  hadamard_col8x4_avx512(src, 0);
  hadamard_col8x4_avx512(src, 1);

  __m512i out0, out1, out2, out3;
  hadamard_16x16_quadrant_butterfly_avx512(src[0], src[1], src[2], src[3],
                                           &out0, &out1, &out2, &out3);
  if (is_final) {
    store_tran_low_avx512(out0, coeff);
    store_tran_low_avx512(out1, coeff + 64);
    store_tran_low_avx512(out2, coeff + 128);
    store_tran_low_avx512(out3, coeff + 192);
  } else {
    int16_t *coeff16 = (int16_t *)coeff;
    _mm512_storeu_si512((__m512i *)coeff16, out0);
    _mm512_storeu_si512((__m512i *)(coeff16 + 64), out1);
    _mm512_storeu_si512((__m512i *)(coeff16 + 128), out2);
    _mm512_storeu_si512((__m512i *)(coeff16 + 192), out3);
  }

  hadamard_16x16_quadrant_butterfly_avx512(src[4], src[5], src[6], src[7],
                                           &out0, &out1, &out2, &out3);
  if (is_final) {
    store_tran_low_avx512(out0, coeff + 32);
    store_tran_low_avx512(out1, coeff + 96);
    store_tran_low_avx512(out2, coeff + 160);
    store_tran_low_avx512(out3, coeff + 224);
  } else {
    int16_t *coeff16 = (int16_t *)coeff + 32;
    _mm512_storeu_si512((__m512i *)coeff16, out0);
    _mm512_storeu_si512((__m512i *)(coeff16 + 64), out1);
    _mm512_storeu_si512((__m512i *)(coeff16 + 128), out2);
    _mm512_storeu_si512((__m512i *)(coeff16 + 192), out3);
  }
}

void aom_hadamard_16x16_avx512(const int16_t *src_diff, ptrdiff_t src_stride,
                               tran_low_t *coeff) {
  hadamard_16x16_avx512(src_diff, src_stride, coeff, 1);
}

void aom_hadamard_lp_16x16_avx512(const int16_t *src_diff, ptrdiff_t src_stride,
                                  int16_t *coeff) {
  hadamard_16x16_avx512(src_diff, src_stride, (tran_low_t *)coeff, 0);
}

void aom_hadamard_32x32_avx512(const int16_t *src_diff, ptrdiff_t src_stride,
                               tran_low_t *coeff) {
  DECLARE_ALIGNED(64, int16_t, temp_coeff[32 * 32]);
  int16_t *t_coeff = temp_coeff;
  for (int idx = 0; idx < 4; ++idx) {
    const int16_t *src_ptr =
        src_diff + (idx >> 1) * 16 * src_stride + (idx & 0x01) * 16;
    hadamard_16x16_avx512(src_ptr, src_stride,
                          (tran_low_t *)(t_coeff + idx * 256), 0);
  }

  for (int idx = 0; idx < 256; idx += 32) {
    const __m512i coeff0 = _mm512_loadu_si512((const __m512i *)t_coeff);
    const __m512i coeff1 = _mm512_loadu_si512((const __m512i *)(t_coeff + 256));
    const __m512i coeff2 = _mm512_loadu_si512((const __m512i *)(t_coeff + 512));
    const __m512i coeff3 = _mm512_loadu_si512((const __m512i *)(t_coeff + 768));

    __m512i coeff0_lo, coeff0_hi, coeff1_lo, coeff1_hi;
    __m512i coeff2_lo, coeff2_hi, coeff3_lo, coeff3_hi;
    sign_extend_16bit_to_32bit_avx512(coeff0, &coeff0_lo, &coeff0_hi);
    sign_extend_16bit_to_32bit_avx512(coeff1, &coeff1_lo, &coeff1_hi);
    sign_extend_16bit_to_32bit_avx512(coeff2, &coeff2_lo, &coeff2_hi);
    sign_extend_16bit_to_32bit_avx512(coeff3, &coeff3_lo, &coeff3_hi);

    __m512i b0_lo =
        _mm512_srai_epi32(_mm512_add_epi32(coeff0_lo, coeff1_lo), 2);
    __m512i b0_hi =
        _mm512_srai_epi32(_mm512_add_epi32(coeff0_hi, coeff1_hi), 2);
    __m512i b1_lo =
        _mm512_srai_epi32(_mm512_sub_epi32(coeff0_lo, coeff1_lo), 2);
    __m512i b1_hi =
        _mm512_srai_epi32(_mm512_sub_epi32(coeff0_hi, coeff1_hi), 2);
    __m512i b2_lo =
        _mm512_srai_epi32(_mm512_add_epi32(coeff2_lo, coeff3_lo), 2);
    __m512i b2_hi =
        _mm512_srai_epi32(_mm512_add_epi32(coeff2_hi, coeff3_hi), 2);
    __m512i b3_lo =
        _mm512_srai_epi32(_mm512_sub_epi32(coeff2_lo, coeff3_lo), 2);
    __m512i b3_hi =
        _mm512_srai_epi32(_mm512_sub_epi32(coeff2_hi, coeff3_hi), 2);

    const __m512i b0 = _mm512_packs_epi32(b0_lo, b0_hi);
    const __m512i b1 = _mm512_packs_epi32(b1_lo, b1_hi);
    const __m512i b2 = _mm512_packs_epi32(b2_lo, b2_hi);
    const __m512i b3 = _mm512_packs_epi32(b3_lo, b3_hi);

    store_tran_low_avx512(_mm512_add_epi16(b0, b2), coeff);
    store_tran_low_avx512(_mm512_add_epi16(b1, b3), coeff + 256);
    store_tran_low_avx512(_mm512_sub_epi16(b0, b2), coeff + 512);
    store_tran_low_avx512(_mm512_sub_epi16(b1, b3), coeff + 768);

    coeff += 32;
    t_coeff += 32;
  }
}

#if CONFIG_AV1_HIGHBITDEPTH
// Processes two 8x8 32-bit Hadamard blocks simultaneously (block 0 in the low
// 256-bit half, block 1 in the high 256-bit half of in[0..7]).
static inline void highbd_hadamard_col8x2_avx512(__m512i *in, int iter) {
  __m512i a0 = in[0];
  __m512i a1 = in[1];
  __m512i a2 = in[2];
  __m512i a3 = in[3];
  __m512i a4 = in[4];
  __m512i a5 = in[5];
  __m512i a6 = in[6];
  __m512i a7 = in[7];

  __m512i b0 = _mm512_add_epi32(a0, a1);
  __m512i b1 = _mm512_sub_epi32(a0, a1);
  __m512i b2 = _mm512_add_epi32(a2, a3);
  __m512i b3 = _mm512_sub_epi32(a2, a3);
  __m512i b4 = _mm512_add_epi32(a4, a5);
  __m512i b5 = _mm512_sub_epi32(a4, a5);
  __m512i b6 = _mm512_add_epi32(a6, a7);
  __m512i b7 = _mm512_sub_epi32(a6, a7);

  a0 = _mm512_add_epi32(b0, b2);
  a1 = _mm512_add_epi32(b1, b3);
  a2 = _mm512_sub_epi32(b0, b2);
  a3 = _mm512_sub_epi32(b1, b3);
  a4 = _mm512_add_epi32(b4, b6);
  a5 = _mm512_add_epi32(b5, b7);
  a6 = _mm512_sub_epi32(b4, b6);
  a7 = _mm512_sub_epi32(b5, b7);

  if (iter == 0) {
    b0 = _mm512_add_epi32(a0, a4);
    b7 = _mm512_add_epi32(a1, a5);
    b3 = _mm512_add_epi32(a2, a6);
    b4 = _mm512_add_epi32(a3, a7);
    b2 = _mm512_sub_epi32(a0, a4);
    b6 = _mm512_sub_epi32(a1, a5);
    b1 = _mm512_sub_epi32(a2, a6);
    b5 = _mm512_sub_epi32(a3, a7);

    a0 = _mm512_unpacklo_epi32(b0, b1);
    a1 = _mm512_unpacklo_epi32(b2, b3);
    a2 = _mm512_unpackhi_epi32(b0, b1);
    a3 = _mm512_unpackhi_epi32(b2, b3);
    a4 = _mm512_unpacklo_epi32(b4, b5);
    a5 = _mm512_unpacklo_epi32(b6, b7);
    a6 = _mm512_unpackhi_epi32(b4, b5);
    a7 = _mm512_unpackhi_epi32(b6, b7);

    b0 = _mm512_unpacklo_epi64(a0, a1);
    b1 = _mm512_unpacklo_epi64(a4, a5);
    b2 = _mm512_unpackhi_epi64(a0, a1);
    b3 = _mm512_unpackhi_epi64(a4, a5);
    b4 = _mm512_unpacklo_epi64(a2, a3);
    b5 = _mm512_unpacklo_epi64(a6, a7);
    b6 = _mm512_unpackhi_epi64(a2, a3);
    b7 = _mm512_unpackhi_epi64(a6, a7);

    const __m512i idx_lo = _mm512_setr_epi64(0, 1, 8, 9, 4, 5, 12, 13);
    const __m512i idx_hi = _mm512_setr_epi64(2, 3, 10, 11, 6, 7, 14, 15);
    in[0] = _mm512_permutex2var_epi64(b0, idx_lo, b1);
    in[1] = _mm512_permutex2var_epi64(b0, idx_hi, b1);
    in[2] = _mm512_permutex2var_epi64(b2, idx_lo, b3);
    in[3] = _mm512_permutex2var_epi64(b2, idx_hi, b3);
    in[4] = _mm512_permutex2var_epi64(b4, idx_lo, b5);
    in[5] = _mm512_permutex2var_epi64(b4, idx_hi, b5);
    in[6] = _mm512_permutex2var_epi64(b6, idx_lo, b7);
    in[7] = _mm512_permutex2var_epi64(b6, idx_hi, b7);
  } else {
    in[0] = _mm512_add_epi32(a0, a4);
    in[7] = _mm512_add_epi32(a1, a5);
    in[3] = _mm512_add_epi32(a2, a6);
    in[4] = _mm512_add_epi32(a3, a7);
    in[2] = _mm512_sub_epi32(a0, a4);
    in[6] = _mm512_sub_epi32(a1, a5);
    in[1] = _mm512_sub_epi32(a2, a6);
    in[5] = _mm512_sub_epi32(a3, a7);
  }
}

static inline void highbd_hadamard_8x8_dual_avx512(const int16_t *src_diff,
                                                   ptrdiff_t src_stride,
                                                   tran_low_t *coeff) {
  __m512i src32[8];
  src32[0] =
      _mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i *)src_diff));
  src32[1] = _mm512_cvtepi16_epi32(
      _mm256_loadu_si256((const __m256i *)(src_diff += src_stride)));
  src32[2] = _mm512_cvtepi16_epi32(
      _mm256_loadu_si256((const __m256i *)(src_diff += src_stride)));
  src32[3] = _mm512_cvtepi16_epi32(
      _mm256_loadu_si256((const __m256i *)(src_diff += src_stride)));
  src32[4] = _mm512_cvtepi16_epi32(
      _mm256_loadu_si256((const __m256i *)(src_diff += src_stride)));
  src32[5] = _mm512_cvtepi16_epi32(
      _mm256_loadu_si256((const __m256i *)(src_diff += src_stride)));
  src32[6] = _mm512_cvtepi16_epi32(
      _mm256_loadu_si256((const __m256i *)(src_diff += src_stride)));
  src32[7] = _mm512_cvtepi16_epi32(
      _mm256_loadu_si256((const __m256i *)(src_diff + src_stride)));

  highbd_hadamard_col8x2_avx512(src32, 0);
  highbd_hadamard_col8x2_avx512(src32, 1);

  _mm512_storeu_si512((__m512i *)coeff,
                      _mm512_shuffle_i64x2(src32[0], src32[1], 0x44));
  _mm512_storeu_si512((__m512i *)(coeff + 16),
                      _mm512_shuffle_i64x2(src32[2], src32[3], 0x44));
  _mm512_storeu_si512((__m512i *)(coeff + 32),
                      _mm512_shuffle_i64x2(src32[4], src32[5], 0x44));
  _mm512_storeu_si512((__m512i *)(coeff + 48),
                      _mm512_shuffle_i64x2(src32[6], src32[7], 0x44));
  _mm512_storeu_si512((__m512i *)(coeff + 64),
                      _mm512_shuffle_i64x2(src32[0], src32[1], 0xEE));
  _mm512_storeu_si512((__m512i *)(coeff + 80),
                      _mm512_shuffle_i64x2(src32[2], src32[3], 0xEE));
  _mm512_storeu_si512((__m512i *)(coeff + 96),
                      _mm512_shuffle_i64x2(src32[4], src32[5], 0xEE));
  _mm512_storeu_si512((__m512i *)(coeff + 112),
                      _mm512_shuffle_i64x2(src32[6], src32[7], 0xEE));
}

void aom_highbd_hadamard_16x16_avx512(const int16_t *src_diff,
                                      ptrdiff_t src_stride, tran_low_t *coeff) {
  tran_low_t *t_coeff = coeff;
  highbd_hadamard_8x8_dual_avx512(src_diff, src_stride, t_coeff);
  highbd_hadamard_8x8_dual_avx512(src_diff + 8 * src_stride, src_stride,
                                  t_coeff + 128);

  for (int idx = 0; idx < 64; idx += 16) {
    __m512i coeff0 = _mm512_loadu_si512((const __m512i *)t_coeff);
    __m512i coeff1 = _mm512_loadu_si512((const __m512i *)(t_coeff + 64));
    __m512i coeff2 = _mm512_loadu_si512((const __m512i *)(t_coeff + 128));
    __m512i coeff3 = _mm512_loadu_si512((const __m512i *)(t_coeff + 192));

    __m512i b0 = _mm512_srai_epi32(_mm512_add_epi32(coeff0, coeff1), 1);
    __m512i b1 = _mm512_srai_epi32(_mm512_sub_epi32(coeff0, coeff1), 1);
    __m512i b2 = _mm512_srai_epi32(_mm512_add_epi32(coeff2, coeff3), 1);
    __m512i b3 = _mm512_srai_epi32(_mm512_sub_epi32(coeff2, coeff3), 1);

    _mm512_storeu_si512((__m512i *)coeff, _mm512_add_epi32(b0, b2));
    _mm512_storeu_si512((__m512i *)(coeff + 64), _mm512_add_epi32(b1, b3));
    _mm512_storeu_si512((__m512i *)(coeff + 128), _mm512_sub_epi32(b0, b2));
    _mm512_storeu_si512((__m512i *)(coeff + 192), _mm512_sub_epi32(b1, b3));

    coeff += 16;
    t_coeff += 16;
  }
}

void aom_highbd_hadamard_32x32_avx512(const int16_t *src_diff,
                                      ptrdiff_t src_stride, tran_low_t *coeff) {
  tran_low_t *t_coeff = coeff;
  for (int idx = 0; idx < 4; ++idx) {
    const int16_t *src_ptr =
        src_diff + (idx >> 1) * 16 * src_stride + (idx & 0x01) * 16;
    aom_highbd_hadamard_16x16_avx512(src_ptr, src_stride, t_coeff + idx * 256);
  }

  for (int idx = 0; idx < 256; idx += 16) {
    __m512i coeff0 = _mm512_loadu_si512((const __m512i *)t_coeff);
    __m512i coeff1 = _mm512_loadu_si512((const __m512i *)(t_coeff + 256));
    __m512i coeff2 = _mm512_loadu_si512((const __m512i *)(t_coeff + 512));
    __m512i coeff3 = _mm512_loadu_si512((const __m512i *)(t_coeff + 768));

    __m512i b0 = _mm512_srai_epi32(_mm512_add_epi32(coeff0, coeff1), 2);
    __m512i b1 = _mm512_srai_epi32(_mm512_sub_epi32(coeff0, coeff1), 2);
    __m512i b2 = _mm512_srai_epi32(_mm512_add_epi32(coeff2, coeff3), 2);
    __m512i b3 = _mm512_srai_epi32(_mm512_sub_epi32(coeff2, coeff3), 2);

    _mm512_storeu_si512((__m512i *)coeff, _mm512_add_epi32(b0, b2));
    _mm512_storeu_si512((__m512i *)(coeff + 256), _mm512_add_epi32(b1, b3));
    _mm512_storeu_si512((__m512i *)(coeff + 512), _mm512_sub_epi32(b0, b2));
    _mm512_storeu_si512((__m512i *)(coeff + 768), _mm512_sub_epi32(b1, b3));

    coeff += 16;
    t_coeff += 16;
  }
}
#endif  // CONFIG_AV1_HIGHBITDEPTH

int aom_satd_avx512(const tran_low_t *coeff, int length) {
  __m512i accum0 = _mm512_setzero_si512();
  __m512i accum1 = _mm512_setzero_si512();
  int i = 0;
  for (; i + 32 <= length; i += 32, coeff += 32) {
    const __m512i s0 = _mm512_loadu_si512((const __m512i *)coeff);
    const __m512i s1 = _mm512_loadu_si512((const __m512i *)(coeff + 16));
    accum0 = _mm512_add_epi32(accum0, _mm512_abs_epi32(s0));
    accum1 = _mm512_add_epi32(accum1, _mm512_abs_epi32(s1));
  }
  for (; i < length; i += 16, coeff += 16) {
    const __m512i s0 = _mm512_loadu_si512((const __m512i *)coeff);
    accum0 = _mm512_add_epi32(accum0, _mm512_abs_epi32(s0));
  }
  accum0 = _mm512_add_epi32(accum0, accum1);
  return _mm512_reduce_add_epi32(accum0);
}

int aom_satd_lp_avx512(const int16_t *coeff, int length) {
  if (length == 16) {
    return aom_satd_lp_avx2(coeff, length);
  }
  const __m512i one = _mm512_set1_epi16(1);
  __m512i accum0 = _mm512_setzero_si512();
  __m512i accum1 = _mm512_setzero_si512();
  int i = 0;
  for (; i + 64 <= length; i += 64, coeff += 64) {
    const __m512i s0 = _mm512_loadu_si512((const __m512i *)coeff);
    const __m512i s1 = _mm512_loadu_si512((const __m512i *)(coeff + 32));
    accum0 =
        _mm512_add_epi32(accum0, _mm512_madd_epi16(_mm512_abs_epi16(s0), one));
    accum1 =
        _mm512_add_epi32(accum1, _mm512_madd_epi16(_mm512_abs_epi16(s1), one));
  }
  for (; i < length; i += 32, coeff += 32) {
    const __m512i s0 = _mm512_loadu_si512((const __m512i *)coeff);
    accum0 =
        _mm512_add_epi32(accum0, _mm512_madd_epi16(_mm512_abs_epi16(s0), one));
  }
  accum0 = _mm512_add_epi32(accum0, accum1);
  return _mm512_reduce_add_epi32(accum0);
}

void aom_int_pro_row_avx512(int16_t *hbuf, const uint8_t *ref,
                            const int ref_stride, const int width,
                            const int height, int norm_factor) {
  assert(width % 16 == 0 && height % 2 == 0);
  if (width % 64 == 0) {
    for (int wd = 0; wd < width; wd += 64) {
      const uint8_t *ref_tmp = ref + wd;
      __m512i s0 = _mm512_setzero_si512();
      __m512i s1 = _mm512_setzero_si512();
      for (int idx = 0; idx < height; idx += 2) {
        const __m256i r0_lo = _mm256_loadu_si256((const __m256i *)ref_tmp);
        const __m256i r0_hi =
            _mm256_loadu_si256((const __m256i *)(ref_tmp + 32));
        ref_tmp += ref_stride;
        const __m256i r1_lo = _mm256_loadu_si256((const __m256i *)ref_tmp);
        const __m256i r1_hi =
            _mm256_loadu_si256((const __m256i *)(ref_tmp + 32));
        ref_tmp += ref_stride;

        s0 = _mm512_add_epi16(s0, _mm512_cvtepu8_epi16(r0_lo));
        s1 = _mm512_add_epi16(s1, _mm512_cvtepu8_epi16(r0_hi));
        s0 = _mm512_add_epi16(s0, _mm512_cvtepu8_epi16(r1_lo));
        s1 = _mm512_add_epi16(s1, _mm512_cvtepu8_epi16(r1_hi));
      }
      s0 = _mm512_srai_epi16(s0, norm_factor);
      s1 = _mm512_srai_epi16(s1, norm_factor);
      _mm512_storeu_si512((__m512i *)(hbuf + wd), s0);
      _mm512_storeu_si512((__m512i *)(hbuf + wd + 32), s1);
    }
  } else if (width == 32) {
    const uint8_t *ref_tmp = ref;
    __m512i s0 = _mm512_setzero_si512();
    for (int idx = 0; idx < height; idx += 2) {
      const __m256i r0 = _mm256_loadu_si256((const __m256i *)ref_tmp);
      ref_tmp += ref_stride;
      const __m256i r1 = _mm256_loadu_si256((const __m256i *)ref_tmp);
      ref_tmp += ref_stride;
      s0 = _mm512_add_epi16(s0, _mm512_cvtepu8_epi16(r0));
      s0 = _mm512_add_epi16(s0, _mm512_cvtepu8_epi16(r1));
    }
    s0 = _mm512_srai_epi16(s0, norm_factor);
    _mm512_storeu_si512((__m512i *)hbuf, s0);
  } else {
    aom_int_pro_row_sse2(hbuf, ref, ref_stride, width, height, norm_factor);
  }
}

#define CALC_TOT_SAD_AND_STORE_AVX512                                          \
  const __m512i r01 = _mm512_add_epi16(_mm512_bslli_epi128(r1, 2), r0);        \
  const __m512i r012 = _mm512_add_epi16(_mm512_bslli_epi128(r2, 4), r01);      \
  const __m512i result0 = _mm512_add_epi16(_mm512_bslli_epi128(r3, 6), r012);  \
  const __m256i res256 = _mm256_add_epi16(                                     \
      _mm512_castsi512_si256(result0), _mm512_extracti64x4_epi64(result0, 1)); \
  const __m128i res128 = _mm_add_epi16(_mm256_castsi256_si128(res256),         \
                                       _mm256_extracti128_si256(res256, 1));   \
  const __m128i results1 = _mm_add_epi16(res128, _mm_srli_si128(res128, 8));   \
  _mm_storel_epi64((__m128i *)vbuf, _mm_srli_epi16(results1, norm_factor));

void aom_int_pro_col_avx512(int16_t *vbuf, const uint8_t *ref,
                            const int ref_stride, const int width,
                            const int height, int norm_factor) {
  assert(width % 16 == 0);
  if (width == 128) {
    const __m512i zero = _mm512_setzero_si512();
    for (int ht = 0; ht < height; ht += 4) {
      const __m512i s0_0 = _mm512_loadu_si512((const __m512i *)ref);
      const __m512i s0_1 = _mm512_loadu_si512((const __m512i *)(ref + 64));
      const __m512i s1_0 =
          _mm512_loadu_si512((const __m512i *)(ref + ref_stride));
      const __m512i s1_1 =
          _mm512_loadu_si512((const __m512i *)(ref + ref_stride + 64));
      const __m512i s2_0 =
          _mm512_loadu_si512((const __m512i *)(ref + 2 * ref_stride));
      const __m512i s2_1 =
          _mm512_loadu_si512((const __m512i *)(ref + 2 * ref_stride + 64));
      const __m512i s3_0 =
          _mm512_loadu_si512((const __m512i *)(ref + 3 * ref_stride));
      const __m512i s3_1 =
          _mm512_loadu_si512((const __m512i *)(ref + 3 * ref_stride + 64));

      const __m512i r0 = _mm512_add_epi16(_mm512_sad_epu8(s0_0, zero),
                                          _mm512_sad_epu8(s0_1, zero));
      const __m512i r1 = _mm512_add_epi16(_mm512_sad_epu8(s1_0, zero),
                                          _mm512_sad_epu8(s1_1, zero));
      const __m512i r2 = _mm512_add_epi16(_mm512_sad_epu8(s2_0, zero),
                                          _mm512_sad_epu8(s2_1, zero));
      const __m512i r3 = _mm512_add_epi16(_mm512_sad_epu8(s3_0, zero),
                                          _mm512_sad_epu8(s3_1, zero));

      CALC_TOT_SAD_AND_STORE_AVX512
      vbuf += 4;
      ref += ref_stride << 2;
    }
  } else if (width == 64) {
    const __m512i zero = _mm512_setzero_si512();
    for (int ht = 0; ht < height; ht += 4) {
      const __m512i s0 = _mm512_loadu_si512((const __m512i *)ref);
      const __m512i s1 =
          _mm512_loadu_si512((const __m512i *)(ref + ref_stride));
      const __m512i s2 =
          _mm512_loadu_si512((const __m512i *)(ref + 2 * ref_stride));
      const __m512i s3 =
          _mm512_loadu_si512((const __m512i *)(ref + 3 * ref_stride));

      const __m512i r0 = _mm512_sad_epu8(s0, zero);
      const __m512i r1 = _mm512_sad_epu8(s1, zero);
      const __m512i r2 = _mm512_sad_epu8(s2, zero);
      const __m512i r3 = _mm512_sad_epu8(s3, zero);

      CALC_TOT_SAD_AND_STORE_AVX512
      vbuf += 4;
      ref += ref_stride << 2;
    }
  } else {
    aom_int_pro_col_avx2(vbuf, ref, ref_stride, width, height, norm_factor);
  }
}

int aom_vector_var_avx512(const int16_t *ref, const int16_t *src, int bwl) {
  const int width = 4 << bwl;
  assert(width % 16 == 0 && width <= 128);
  const __m512i one = _mm512_set1_epi16(1);

  if (width == 128) {
    const __m512i s0 = _mm512_loadu_si512((const __m512i *)src);
    const __m512i s1 = _mm512_loadu_si512((const __m512i *)(src + 32));
    const __m512i s2 = _mm512_loadu_si512((const __m512i *)(src + 64));
    const __m512i s3 = _mm512_loadu_si512((const __m512i *)(src + 96));
    const __m512i r0 = _mm512_loadu_si512((const __m512i *)ref);
    const __m512i r1 = _mm512_loadu_si512((const __m512i *)(ref + 32));
    const __m512i r2 = _mm512_loadu_si512((const __m512i *)(ref + 64));
    const __m512i r3 = _mm512_loadu_si512((const __m512i *)(ref + 96));

    const __m512i d0 = _mm512_sub_epi16(r0, s0);
    const __m512i d1 = _mm512_sub_epi16(r1, s1);
    const __m512i d2 = _mm512_sub_epi16(r2, s2);
    const __m512i d3 = _mm512_sub_epi16(r3, s3);

    const __m512i mean16 =
        _mm512_add_epi16(_mm512_add_epi16(d0, d1), _mm512_add_epi16(d2, d3));
    const __m512i sse0 =
        _mm512_add_epi32(_mm512_madd_epi16(d0, d0), _mm512_madd_epi16(d1, d1));
    const __m512i sse1 =
        _mm512_add_epi32(_mm512_madd_epi16(d2, d2), _mm512_madd_epi16(d3, d3));
    const __m512i sse32 = _mm512_add_epi32(sse0, sse1);
    const __m512i mean32 = _mm512_madd_epi16(mean16, one);

    const int mean_int = _mm512_reduce_add_epi32(mean32);
    const int sse_int = _mm512_reduce_add_epi32(sse32);
    const unsigned int mean_abs = (unsigned int)abs(mean_int);
    return sse_int - (int)((mean_abs * mean_abs) >> (bwl + 2));
  } else if (width == 64) {
    const __m512i s0 = _mm512_loadu_si512((const __m512i *)src);
    const __m512i s1 = _mm512_loadu_si512((const __m512i *)(src + 32));
    const __m512i r0 = _mm512_loadu_si512((const __m512i *)ref);
    const __m512i r1 = _mm512_loadu_si512((const __m512i *)(ref + 32));

    const __m512i d0 = _mm512_sub_epi16(r0, s0);
    const __m512i d1 = _mm512_sub_epi16(r1, s1);

    const __m512i mean16 = _mm512_add_epi16(d0, d1);
    const __m512i sse32 =
        _mm512_add_epi32(_mm512_madd_epi16(d0, d0), _mm512_madd_epi16(d1, d1));
    const __m512i mean32 = _mm512_madd_epi16(mean16, one);

    const int mean_int = _mm512_reduce_add_epi32(mean32);
    const int sse_int = _mm512_reduce_add_epi32(sse32);
    const unsigned int mean_abs = (unsigned int)abs(mean_int);
    return sse_int - (int)((mean_abs * mean_abs) >> (bwl + 2));
  } else if (width == 32) {
    const __m512i s0 = _mm512_loadu_si512((const __m512i *)src);
    const __m512i r0 = _mm512_loadu_si512((const __m512i *)ref);
    const __m512i d0 = _mm512_sub_epi16(r0, s0);
    const __m512i sse32 = _mm512_madd_epi16(d0, d0);
    const __m512i mean32 = _mm512_madd_epi16(d0, one);

    const int mean_int = _mm512_reduce_add_epi32(mean32);
    const int sse_int = _mm512_reduce_add_epi32(sse32);
    const unsigned int mean_abs = (unsigned int)abs(mean_int);
    return sse_int - (int)((mean_abs * mean_abs) >> (bwl + 2));
  } else {
    return aom_vector_var_avx2(ref, src, bwl);
  }
}

static inline void init_one_b_qp_avx512(__m128i v128, __m512i *v512) {
  const __m128i ac = _mm_unpackhi_epi64(v128, v128);
  const __m256i dc_ac =
      _mm256_inserti128_si256(_mm256_castsi128_si256(v128), ac, 1);
  const __m256i ac256 = _mm256_broadcastq_epi64(ac);
  *v512 = _mm512_inserti64x4(_mm512_castsi256_si512(dc_ac), ac256, 1);
}

static inline void load_b_values_avx512(
    const int16_t *zbin_ptr, __m512i *zbin, const int16_t *round_ptr,
    __m512i *round, const int16_t *quant_ptr, __m512i *quant,
    const int16_t *dequant_ptr, __m512i *dequant, const int16_t *shift_ptr,
    __m512i *shift, int log_scale) {
  __m128i zbin128 = _mm_loadu_si128((const __m128i *)zbin_ptr);
  __m128i round128 = _mm_loadu_si128((const __m128i *)round_ptr);
  const __m128i quant128 = _mm_loadu_si128((const __m128i *)quant_ptr);
  const __m128i dequant128 = _mm_loadu_si128((const __m128i *)dequant_ptr);
  const __m128i shift128 = _mm_loadu_si128((const __m128i *)shift_ptr);

  if (log_scale > 0) {
    const __m128i rnd = _mm_set1_epi16((int16_t)(1 << (log_scale - 1)));
    zbin128 = _mm_srai_epi16(_mm_add_epi16(zbin128, rnd), log_scale);
    round128 = _mm_srai_epi16(_mm_add_epi16(round128, rnd), log_scale);
  }
  zbin128 = _mm_sub_epi16(zbin128, _mm_set1_epi16(1));

  init_one_b_qp_avx512(zbin128, zbin);
  init_one_b_qp_avx512(round128, round);
  init_one_b_qp_avx512(quant128, quant);
  init_one_b_qp_avx512(dequant128, dequant);
  init_one_b_qp_avx512(shift128, shift);
}

static inline __m512i load_b_coefficients_avx512(const tran_low_t *coeff_ptr) {
  const __m512i coeff1 = _mm512_loadu_si512((const __m512i *)coeff_ptr);
  const __m512i coeff2 = _mm512_loadu_si512((const __m512i *)(coeff_ptr + 16));
  return _mm512_packs_epi32(coeff1, coeff2);
}

static inline void store_b_coefficients_avx512(__m512i coeff_vals,
                                               tran_low_t *coeff_ptr) {
  const __m512i coeff_sign = _mm512_srai_epi16(coeff_vals, 15);
  const __m512i coeff_lo = _mm512_unpacklo_epi16(coeff_vals, coeff_sign);
  const __m512i coeff_hi = _mm512_unpackhi_epi16(coeff_vals, coeff_sign);
  _mm512_storeu_si512((__m512i *)coeff_ptr, coeff_lo);
  _mm512_storeu_si512((__m512i *)(coeff_ptr + 16), coeff_hi);
}

static inline __m512i get_max_lane_eob_b_avx512(const int16_t *iscan,
                                                __m512i v_eobmax,
                                                __mmask32 nz_mask) {
  const __m512i v_iscan = _mm512_loadu_si512((const __m512i *)iscan);
  const __m512i perm_idx = _mm512_setr_epi64(0, 4, 1, 5, 2, 6, 3, 7);
  const __m512i v_iscan_perm = _mm512_permutexvar_epi64(perm_idx, v_iscan);
  const __m512i v_nz_iscan =
      _mm512_maskz_add_epi16(nz_mask, v_iscan_perm, _mm512_set1_epi16(1));
  return _mm512_max_epi16(v_eobmax, v_nz_iscan);
}

static inline uint16_t accumulate_eob512(__m512i eob512) {
  const __m256i eob256 = _mm256_max_epi16(_mm512_castsi512_si256(eob512),
                                          _mm512_extracti64x4_epi64(eob512, 1));
  const __m128i eob_lo = _mm256_castsi256_si128(eob256);
  const __m128i eob_hi = _mm256_extracti128_si256(eob256, 1);
  __m128i eob_s = _mm_max_epi16(eob_lo, eob_hi);
  eob_s = _mm_subs_epu16(_mm_set1_epi16(INT16_MAX), eob_s);
  eob_s = _mm_minpos_epu16(eob_s);
  return (uint16_t)(INT16_MAX - _mm_extract_epi16(eob_s, 0));
}

static inline __m512i mm512_sign_epi16(__m512i a, __m512i b) {
  const __m512i zero = _mm512_setzero_si512();
  const __mmask32 nz_mask = _mm512_cmpneq_epi16_mask(b, zero);
  const __mmask32 neg_mask = _mm512_cmpgt_epi16_mask(zero, b);
  const __m512i a_nz = _mm512_maskz_mov_epi16(nz_mask, a);
  return _mm512_mask_sub_epi16(a_nz, neg_mask, zero, a_nz);
}

static AOM_FORCE_INLINE __mmask32 quantize_b_logscale0_32_avx512(
    const tran_low_t *coeff_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, const __m512i *v_quant, const __m512i *v_dequant,
    const __m512i *v_round, const __m512i *v_zbin,
    const __m512i *v_quant_shift) {
  const __m512i v_coeff = load_b_coefficients_avx512(coeff_ptr);
  const __m512i v_abs_coeff = _mm512_abs_epi16(v_coeff);
  const __mmask32 zbin_mask = _mm512_cmpgt_epi16_mask(v_abs_coeff, *v_zbin);

  if (zbin_mask == 0) {
    const __m512i zero = _mm512_setzero_si512();
    _mm512_storeu_si512((__m512i *)qcoeff_ptr, zero);
    _mm512_storeu_si512((__m512i *)(qcoeff_ptr + 16), zero);
    _mm512_storeu_si512((__m512i *)dqcoeff_ptr, zero);
    _mm512_storeu_si512((__m512i *)(dqcoeff_ptr + 16), zero);
    return 0;
  }

  const __m512i v_tmp_rnd =
      _mm512_maskz_adds_epi16(zbin_mask, v_abs_coeff, *v_round);
  const __m512i v_tmp32_a = _mm512_mulhi_epi16(v_tmp_rnd, *v_quant);
  const __m512i v_tmp32_b = _mm512_add_epi16(v_tmp32_a, v_tmp_rnd);
  const __m512i v_tmp32 = _mm512_mulhi_epi16(v_tmp32_b, *v_quant_shift);
  const __mmask32 nz_mask =
      _mm512_cmpgt_epi16_mask(v_tmp32, _mm512_setzero_si512());
  const __m512i v_qcoeff = mm512_sign_epi16(v_tmp32, v_coeff);
  const __m512i v_dqcoeff = _mm512_mullo_epi16(v_qcoeff, *v_dequant);
  store_b_coefficients_avx512(v_qcoeff, qcoeff_ptr);
  store_b_coefficients_avx512(v_dqcoeff, dqcoeff_ptr);
  return nz_mask;
}

void aom_quantize_b_avx512(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                           const int16_t *zbin_ptr, const int16_t *round_ptr,
                           const int16_t *quant_ptr,
                           const int16_t *quant_shift_ptr,
                           tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                           const int16_t *dequant_ptr, uint16_t *eob_ptr,
                           const int16_t *scan, const int16_t *iscan) {
  if (n_coeffs < 32) {
    aom_quantize_b_avx2(coeff_ptr, n_coeffs, zbin_ptr, round_ptr, quant_ptr,
                        quant_shift_ptr, qcoeff_ptr, dqcoeff_ptr, dequant_ptr,
                        eob_ptr, scan, iscan);
    return;
  }
  (void)scan;
  __m512i v_zbin, v_round, v_quant, v_dequant, v_quant_shift;
  __m512i v_eobmax = _mm512_setzero_si512();

  load_b_values_avx512(zbin_ptr, &v_zbin, round_ptr, &v_round, quant_ptr,
                       &v_quant, dequant_ptr, &v_dequant, quant_shift_ptr,
                       &v_quant_shift, 0);

  __mmask32 nz_mask = quantize_b_logscale0_32_avx512(
      coeff_ptr, qcoeff_ptr, dqcoeff_ptr, &v_quant, &v_dequant, &v_round,
      &v_zbin, &v_quant_shift);
  if (nz_mask) v_eobmax = get_max_lane_eob_b_avx512(iscan, v_eobmax, nz_mask);

  v_round = _mm512_unpackhi_epi64(v_round, v_round);
  v_quant = _mm512_unpackhi_epi64(v_quant, v_quant);
  v_dequant = _mm512_unpackhi_epi64(v_dequant, v_dequant);
  v_quant_shift = _mm512_unpackhi_epi64(v_quant_shift, v_quant_shift);
  v_zbin = _mm512_unpackhi_epi64(v_zbin, v_zbin);

  for (intptr_t count = n_coeffs - 32; count > 0; count -= 32) {
    coeff_ptr += 32;
    qcoeff_ptr += 32;
    dqcoeff_ptr += 32;
    iscan += 32;
    nz_mask = quantize_b_logscale0_32_avx512(coeff_ptr, qcoeff_ptr, dqcoeff_ptr,
                                             &v_quant, &v_dequant, &v_round,
                                             &v_zbin, &v_quant_shift);
    if (nz_mask) v_eobmax = get_max_lane_eob_b_avx512(iscan, v_eobmax, nz_mask);
  }

  *eob_ptr = accumulate_eob512(v_eobmax);
}

static AOM_FORCE_INLINE __mmask32 quantize_b_logscale_32_avx512(
    const tran_low_t *coeff_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, const __m512i *v_quant, const __m512i *v_dequant,
    const __m512i *v_round, const __m512i *v_zbin, const __m512i *v_quant_shift,
    int log_scale) {
  const __m512i v_coeff = load_b_coefficients_avx512(coeff_ptr);
  const __m512i v_abs_coeff = _mm512_abs_epi16(v_coeff);
  const __mmask32 zbin_mask = _mm512_cmpgt_epi16_mask(v_abs_coeff, *v_zbin);

  if (zbin_mask == 0) {
    const __m512i zero = _mm512_setzero_si512();
    _mm512_storeu_si512((__m512i *)qcoeff_ptr, zero);
    _mm512_storeu_si512((__m512i *)(qcoeff_ptr + 16), zero);
    _mm512_storeu_si512((__m512i *)dqcoeff_ptr, zero);
    _mm512_storeu_si512((__m512i *)(dqcoeff_ptr + 16), zero);
    return 0;
  }

  const __m512i v_tmp_rnd =
      _mm512_maskz_adds_epi16(zbin_mask, v_abs_coeff, *v_round);
  const __m512i v_tmp32_a = _mm512_mulhi_epi16(v_tmp_rnd, *v_quant);
  const __m512i v_tmp32_b = _mm512_add_epi16(v_tmp32_a, v_tmp_rnd);
  const __m512i v_tmp32_hi = _mm512_slli_epi16(
      _mm512_mulhi_epi16(v_tmp32_b, *v_quant_shift), (unsigned int)log_scale);
  const __m512i v_tmp32_lo =
      _mm512_srli_epi16(_mm512_mullo_epi16(v_tmp32_b, *v_quant_shift),
                        (unsigned int)(16 - log_scale));
  const __m512i v_tmp32 = _mm512_or_si512(v_tmp32_hi, v_tmp32_lo);
  const __m512i v_dqcoeff_hi = _mm512_slli_epi16(
      _mm512_mulhi_epi16(v_tmp32, *v_dequant), (unsigned int)(16 - log_scale));
  const __m512i v_dqcoeff_lo = _mm512_srli_epi16(
      _mm512_mullo_epi16(v_tmp32, *v_dequant), (unsigned int)log_scale);
  const __m512i v_dqcoeff =
      mm512_sign_epi16(_mm512_or_si512(v_dqcoeff_hi, v_dqcoeff_lo), v_coeff);
  const __m512i v_qcoeff = mm512_sign_epi16(v_tmp32, v_coeff);
  const __mmask32 nz_mask =
      _mm512_cmpgt_epi16_mask(v_tmp32, _mm512_setzero_si512());
  store_b_coefficients_avx512(v_qcoeff, qcoeff_ptr);
  store_b_coefficients_avx512(v_dqcoeff, dqcoeff_ptr);
  return nz_mask;
}

static AOM_FORCE_INLINE void quantize_b_no_qmatrix_avx512(
    const tran_low_t *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr,
    const int16_t *round_ptr, const int16_t *quant_ptr,
    const int16_t *quant_shift_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr,
    const int16_t *iscan, int log_scale) {
  __m512i v_zbin, v_round, v_quant, v_dequant, v_quant_shift;
  __m512i v_eobmax = _mm512_setzero_si512();

  load_b_values_avx512(zbin_ptr, &v_zbin, round_ptr, &v_round, quant_ptr,
                       &v_quant, dequant_ptr, &v_dequant, quant_shift_ptr,
                       &v_quant_shift, log_scale);

  __mmask32 nz_mask = quantize_b_logscale_32_avx512(
      coeff_ptr, qcoeff_ptr, dqcoeff_ptr, &v_quant, &v_dequant, &v_round,
      &v_zbin, &v_quant_shift, log_scale);
  if (nz_mask) v_eobmax = get_max_lane_eob_b_avx512(iscan, v_eobmax, nz_mask);

  v_round = _mm512_unpackhi_epi64(v_round, v_round);
  v_quant = _mm512_unpackhi_epi64(v_quant, v_quant);
  v_dequant = _mm512_unpackhi_epi64(v_dequant, v_dequant);
  v_quant_shift = _mm512_unpackhi_epi64(v_quant_shift, v_quant_shift);
  v_zbin = _mm512_unpackhi_epi64(v_zbin, v_zbin);

  for (intptr_t count = n_coeffs - 32; count > 0; count -= 32) {
    coeff_ptr += 32;
    qcoeff_ptr += 32;
    dqcoeff_ptr += 32;
    iscan += 32;
    nz_mask = quantize_b_logscale_32_avx512(coeff_ptr, qcoeff_ptr, dqcoeff_ptr,
                                            &v_quant, &v_dequant, &v_round,
                                            &v_zbin, &v_quant_shift, log_scale);
    if (nz_mask) v_eobmax = get_max_lane_eob_b_avx512(iscan, v_eobmax, nz_mask);
  }

  *eob_ptr = accumulate_eob512(v_eobmax);
}

void aom_quantize_b_32x32_avx512(
    const tran_low_t *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr,
    const int16_t *round_ptr, const int16_t *quant_ptr,
    const int16_t *quant_shift_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr,
    const int16_t *scan, const int16_t *iscan) {
  (void)scan;
  quantize_b_no_qmatrix_avx512(coeff_ptr, n_coeffs, zbin_ptr, round_ptr,
                               quant_ptr, quant_shift_ptr, qcoeff_ptr,
                               dqcoeff_ptr, dequant_ptr, eob_ptr, iscan, 1);
}

void aom_quantize_b_64x64_avx512(
    const tran_low_t *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr,
    const int16_t *round_ptr, const int16_t *quant_ptr,
    const int16_t *quant_shift_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr,
    const int16_t *scan, const int16_t *iscan) {
  (void)scan;
  quantize_b_no_qmatrix_avx512(coeff_ptr, n_coeffs, zbin_ptr, round_ptr,
                               quant_ptr, quant_shift_ptr, qcoeff_ptr,
                               dqcoeff_ptr, dequant_ptr, eob_ptr, iscan, 2);
}
