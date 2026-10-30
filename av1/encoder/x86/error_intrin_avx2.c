/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <immintrin.h>  // AVX2

#include "config/av1_rtcd.h"

#include "aom/aom_integer.h"
#include "aom_dsp/aom_dsp_common.h"

static inline void read_coeff(const tran_low_t *coeff, intptr_t offset,
                              __m256i *c) {
  const tran_low_t *addr = coeff + offset;

  if (sizeof(tran_low_t) == 4) {
    const __m256i x0 = _mm256_loadu_si256((const __m256i *)addr);
    const __m256i x1 = _mm256_loadu_si256((const __m256i *)addr + 1);
    const __m256i y = _mm256_packs_epi32(x0, x1);
    *c = _mm256_permute4x64_epi64(y, 0xD8);
  } else {
    *c = _mm256_loadu_si256((const __m256i *)addr);
  }
}

static inline void av1_block_error_block_size16_avx2(const int16_t *coeff,
                                                     const int16_t *dqcoeff,
                                                     __m256i *sse_256) {
  const __m256i _coeff = _mm256_loadu_si256((const __m256i *)coeff);
  const __m256i _dqcoeff = _mm256_loadu_si256((const __m256i *)dqcoeff);
  // d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
  const __m256i diff = _mm256_sub_epi16(_dqcoeff, _coeff);
  // r0 r1 r2 r3 r4 r5 r6 r7
  const __m256i error = _mm256_madd_epi16(diff, diff);
  // r0+r1 r2+r3 | r0+r1 r2+r3 | r4+r5 r6+r7 | r4+r5 r6+r7
  const __m256i error_hi = _mm256_hadd_epi32(error, error);
  // r0+r1 | r2+r3 | r4+r5 | r6+r7
  *sse_256 = _mm256_unpacklo_epi32(error_hi, _mm256_setzero_si256());
}

static inline void av1_block_error_block_size32_avx2(const int16_t *coeff,
                                                     const int16_t *dqcoeff,
                                                     __m256i *sse_256) {
  const __m256i zero = _mm256_setzero_si256();
  const __m256i _coeff_0 = _mm256_loadu_si256((const __m256i *)coeff);
  const __m256i _dqcoeff_0 = _mm256_loadu_si256((const __m256i *)dqcoeff);
  const __m256i _coeff_1 = _mm256_loadu_si256((const __m256i *)(coeff + 16));
  const __m256i _dqcoeff_1 =
      _mm256_loadu_si256((const __m256i *)(dqcoeff + 16));

  // d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
  const __m256i diff_0 = _mm256_sub_epi16(_dqcoeff_0, _coeff_0);
  const __m256i diff_1 = _mm256_sub_epi16(_dqcoeff_1, _coeff_1);

  // r0 r1 r2 r3 r4 r5 r6 r7
  const __m256i error_0 = _mm256_madd_epi16(diff_0, diff_0);
  const __m256i error_1 = _mm256_madd_epi16(diff_1, diff_1);
  const __m256i err_final_0 = _mm256_add_epi32(error_0, error_1);

  // For extreme input values, the accumulation needs to happen in 64 bit
  // precision to avoid any overflow.
  const __m256i exp0_error_lo = _mm256_unpacklo_epi32(err_final_0, zero);
  const __m256i exp0_error_hi = _mm256_unpackhi_epi32(err_final_0, zero);
  const __m256i sum_temp_0 = _mm256_add_epi64(exp0_error_hi, exp0_error_lo);
  *sse_256 = _mm256_add_epi64(*sse_256, sum_temp_0);
}

static inline void av1_block_error_block_size64_avx2(const int16_t *coeff,
                                                     const int16_t *dqcoeff,
                                                     __m256i *sse_256,
                                                     intptr_t block_size) {
  const __m256i zero = _mm256_setzero_si256();
  for (int i = 0; i < block_size; i += 64) {
    // Load 64 elements for coeff and dqcoeff.
    const __m256i _coeff_0 = _mm256_loadu_si256((const __m256i *)coeff);
    const __m256i _dqcoeff_0 = _mm256_loadu_si256((const __m256i *)dqcoeff);
    const __m256i _coeff_1 = _mm256_loadu_si256((const __m256i *)(coeff + 16));
    const __m256i _dqcoeff_1 =
        _mm256_loadu_si256((const __m256i *)(dqcoeff + 16));
    const __m256i _coeff_2 = _mm256_loadu_si256((const __m256i *)(coeff + 32));
    const __m256i _dqcoeff_2 =
        _mm256_loadu_si256((const __m256i *)(dqcoeff + 32));
    const __m256i _coeff_3 = _mm256_loadu_si256((const __m256i *)(coeff + 48));
    const __m256i _dqcoeff_3 =
        _mm256_loadu_si256((const __m256i *)(dqcoeff + 48));

    // d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
    const __m256i diff_0 = _mm256_sub_epi16(_dqcoeff_0, _coeff_0);
    const __m256i diff_1 = _mm256_sub_epi16(_dqcoeff_1, _coeff_1);
    const __m256i diff_2 = _mm256_sub_epi16(_dqcoeff_2, _coeff_2);
    const __m256i diff_3 = _mm256_sub_epi16(_dqcoeff_3, _coeff_3);

    // r0 r1 r2 r3 r4 r5 r6 r7
    const __m256i error_0 = _mm256_madd_epi16(diff_0, diff_0);
    const __m256i error_1 = _mm256_madd_epi16(diff_1, diff_1);
    const __m256i error_2 = _mm256_madd_epi16(diff_2, diff_2);
    const __m256i error_3 = _mm256_madd_epi16(diff_3, diff_3);
    // r00 r01 r02 r03 r04 r05 r06 r07
    const __m256i err_final_0 = _mm256_add_epi32(error_0, error_1);
    // r10 r11 r12 r13 r14 r15 r16 r17
    const __m256i err_final_1 = _mm256_add_epi32(error_2, error_3);

    // For extreme input values, the accumulation needs to happen in 64 bit
    // precision to avoid any overflow. r00 r01 r04 r05
    const __m256i exp0_error_lo = _mm256_unpacklo_epi32(err_final_0, zero);
    // r02 r03 r06 r07
    const __m256i exp0_error_hi = _mm256_unpackhi_epi32(err_final_0, zero);
    // r10 r11 r14 r15
    const __m256i exp1_error_lo = _mm256_unpacklo_epi32(err_final_1, zero);
    // r12 r13 r16 r17
    const __m256i exp1_error_hi = _mm256_unpackhi_epi32(err_final_1, zero);

    const __m256i sum_temp_0 = _mm256_add_epi64(exp0_error_hi, exp0_error_lo);
    const __m256i sum_temp_1 = _mm256_add_epi64(exp1_error_hi, exp1_error_lo);
    const __m256i sse_256_temp = _mm256_add_epi64(sum_temp_1, sum_temp_0);
    *sse_256 = _mm256_add_epi64(*sse_256, sse_256_temp);
    coeff += 64;
    dqcoeff += 64;
  }
}

int64_t av1_block_error_lp_avx2(const int16_t *coeff, const int16_t *dqcoeff,
                                intptr_t block_size) {
  assert(block_size % 16 == 0);
  __m256i sse_256 = _mm256_setzero_si256();
  int64_t sse;

  if (block_size == 16)
    av1_block_error_block_size16_avx2(coeff, dqcoeff, &sse_256);
  else if (block_size == 32)
    av1_block_error_block_size32_avx2(coeff, dqcoeff, &sse_256);
  else
    av1_block_error_block_size64_avx2(coeff, dqcoeff, &sse_256, block_size);

  // Save the higher 64 bit of each 128 bit lane.
  const __m256i sse_hi = _mm256_srli_si256(sse_256, 8);
  // Add the higher 64 bit to the low 64 bit.
  sse_256 = _mm256_add_epi64(sse_256, sse_hi);
  // Accumulate the sse_256 register to get final sse
  const __m128i sse_128 = _mm_add_epi64(_mm256_castsi256_si128(sse_256),
                                        _mm256_extractf128_si256(sse_256, 1));

  // Store the results.
  _mm_storel_epi64((__m128i *)&sse, sse_128);
  return sse;
}

int64_t av1_block_error_avx2(const tran_low_t *coeff, const tran_low_t *dqcoeff,
                             intptr_t block_size, int64_t *ssz) {
  __m256i sse_reg, ssz_reg, coeff_reg, dqcoeff_reg;
  __m256i exp_dqcoeff_lo, exp_dqcoeff_hi, exp_coeff_lo, exp_coeff_hi;
  __m256i sse_reg_64hi, ssz_reg_64hi;
  __m128i sse_reg128, ssz_reg128;
  int64_t sse;
  int i;
  const __m256i zero_reg = _mm256_setzero_si256();

  // init sse and ssz registerd to zero
  sse_reg = _mm256_setzero_si256();
  ssz_reg = _mm256_setzero_si256();

  for (i = 0; i < block_size; i += 16) {
    // load 32 bytes from coeff and dqcoeff
    read_coeff(coeff, i, &coeff_reg);
    read_coeff(dqcoeff, i, &dqcoeff_reg);
    // dqcoeff - coeff
    dqcoeff_reg = _mm256_sub_epi16(dqcoeff_reg, coeff_reg);
    // madd (dqcoeff - coeff)
    dqcoeff_reg = _mm256_madd_epi16(dqcoeff_reg, dqcoeff_reg);
    // madd coeff
    coeff_reg = _mm256_madd_epi16(coeff_reg, coeff_reg);
    // expand each double word of madd (dqcoeff - coeff) to quad word
    exp_dqcoeff_lo = _mm256_unpacklo_epi32(dqcoeff_reg, zero_reg);
    exp_dqcoeff_hi = _mm256_unpackhi_epi32(dqcoeff_reg, zero_reg);
    // expand each double word of madd (coeff) to quad word
    exp_coeff_lo = _mm256_unpacklo_epi32(coeff_reg, zero_reg);
    exp_coeff_hi = _mm256_unpackhi_epi32(coeff_reg, zero_reg);
    // add each quad word of madd (dqcoeff - coeff) and madd (coeff)
    sse_reg = _mm256_add_epi64(sse_reg, exp_dqcoeff_lo);
    ssz_reg = _mm256_add_epi64(ssz_reg, exp_coeff_lo);
    sse_reg = _mm256_add_epi64(sse_reg, exp_dqcoeff_hi);
    ssz_reg = _mm256_add_epi64(ssz_reg, exp_coeff_hi);
  }
  // save the higher 64 bit of each 128 bit lane
  sse_reg_64hi = _mm256_srli_si256(sse_reg, 8);
  ssz_reg_64hi = _mm256_srli_si256(ssz_reg, 8);
  // add the higher 64 bit to the low 64 bit
  sse_reg = _mm256_add_epi64(sse_reg, sse_reg_64hi);
  ssz_reg = _mm256_add_epi64(ssz_reg, ssz_reg_64hi);

  // add each 64 bit from each of the 128 bit lane of the 256 bit
  sse_reg128 = _mm_add_epi64(_mm256_castsi256_si128(sse_reg),
                             _mm256_extractf128_si256(sse_reg, 1));

  ssz_reg128 = _mm_add_epi64(_mm256_castsi256_si128(ssz_reg),
                             _mm256_extractf128_si256(ssz_reg, 1));

  // store the results
  _mm_storel_epi64((__m128i *)(&sse), sse_reg128);

  _mm_storel_epi64((__m128i *)(ssz), ssz_reg128);
  _mm256_zeroupper();
  return sse;
}

int64_t av1_block_error_qm_avx2(const tran_low_t *coeff,
                                const tran_low_t *dqcoeff, intptr_t block_size,
                                const qm_val_t *qmatrix, const int16_t *scan,
                                int64_t *ssz, int bd) {
  assert(block_size % 8 == 0);
  assert(bd == 8 || bd == 10 || bd == 12);
  __m256i sse_reg = _mm256_setzero_si256();
  __m256i ssz_reg = _mm256_setzero_si256();
  const __m256i round_qm = _mm256_set1_epi64x(1 << (2 * AOM_QM_BITS - 1));

  for (intptr_t i = 0; i < block_size; i += 8) {
    // Load 8 transform coefficients and dequantized coefficients.
    const __m256i coeff_reg = _mm256_loadu_si256((const __m256i *)(coeff + i));
    const __m256i dqcoeff_reg =
        _mm256_loadu_si256((const __m256i *)(dqcoeff + i));

    // When all 8 dequantized coefficients are zero, sse and ssz accumulate the
    // same weighted squared coefficient energy.
    if (_mm256_testz_si256(dqcoeff_reg, dqcoeff_reg)) {
      if (_mm256_testz_si256(coeff_reg, coeff_reg)) continue;

      const __m256i wt_reg = _mm256_setr_epi32(
          qmatrix[scan[i + 0]], qmatrix[scan[i + 1]], qmatrix[scan[i + 2]],
          qmatrix[scan[i + 3]], qmatrix[scan[i + 4]], qmatrix[scan[i + 5]],
          qmatrix[scan[i + 6]], qmatrix[scan[i + 7]]);
      const __m256i cc = _mm256_mullo_epi32(coeff_reg, wt_reg);
      const __m256i cc_hi = _mm256_srli_epi64(cc, 32);
      const __m256i cc_lo_sq = _mm256_mul_epi32(cc, cc);
      const __m256i cc_hi_sq = _mm256_mul_epi32(cc_hi, cc_hi);

      const __m256i sq_lo = _mm256_srli_epi64(
          _mm256_add_epi64(cc_lo_sq, round_qm), 2 * AOM_QM_BITS);
      const __m256i sq_hi = _mm256_srli_epi64(
          _mm256_add_epi64(cc_hi_sq, round_qm), 2 * AOM_QM_BITS);
      const __m256i sq = _mm256_add_epi64(sq_lo, sq_hi);

      sse_reg = _mm256_add_epi64(sse_reg, sq);
      ssz_reg = _mm256_add_epi64(ssz_reg, sq);
    } else {
      const __m256i wt_reg = _mm256_setr_epi32(
          qmatrix[scan[i + 0]], qmatrix[scan[i + 1]], qmatrix[scan[i + 2]],
          qmatrix[scan[i + 3]], qmatrix[scan[i + 4]], qmatrix[scan[i + 5]],
          qmatrix[scan[i + 6]], qmatrix[scan[i + 7]]);
      const __m256i diff = _mm256_sub_epi32(coeff_reg, dqcoeff_reg);
      const __m256i dd = _mm256_mullo_epi32(diff, wt_reg);
      const __m256i cc = _mm256_mullo_epi32(coeff_reg, wt_reg);

      // Compute weighted squared error (coeff - dqcoeff)^2 in 64-bit lanes.
      const __m256i dd_hi = _mm256_srli_epi64(dd, 32);
      const __m256i dd_lo_sq = _mm256_mul_epi32(dd, dd);
      const __m256i dd_hi_sq = _mm256_mul_epi32(dd_hi, dd_hi);
      const __m256i err_lo = _mm256_srli_epi64(
          _mm256_add_epi64(dd_lo_sq, round_qm), 2 * AOM_QM_BITS);
      const __m256i err_hi = _mm256_srli_epi64(
          _mm256_add_epi64(dd_hi_sq, round_qm), 2 * AOM_QM_BITS);
      sse_reg = _mm256_add_epi64(sse_reg, _mm256_add_epi64(err_lo, err_hi));

      // Compute weighted squared coefficient energy coeff^2 in 64-bit lanes.
      const __m256i cc_hi = _mm256_srli_epi64(cc, 32);
      const __m256i cc_lo_sq = _mm256_mul_epi32(cc, cc);
      const __m256i cc_hi_sq = _mm256_mul_epi32(cc_hi, cc_hi);
      const __m256i sq_lo = _mm256_srli_epi64(
          _mm256_add_epi64(cc_lo_sq, round_qm), 2 * AOM_QM_BITS);
      const __m256i sq_hi = _mm256_srli_epi64(
          _mm256_add_epi64(cc_hi_sq, round_qm), 2 * AOM_QM_BITS);
      ssz_reg = _mm256_add_epi64(ssz_reg, _mm256_add_epi64(sq_lo, sq_hi));
    }
  }

  // Save the higher 64 bit of each 128 bit lane and add to the low 64 bit.
  const __m256i sse_reg_64hi = _mm256_srli_si256(sse_reg, 8);
  const __m256i ssz_reg_64hi = _mm256_srli_si256(ssz_reg, 8);
  sse_reg = _mm256_add_epi64(sse_reg, sse_reg_64hi);
  ssz_reg = _mm256_add_epi64(ssz_reg, ssz_reg_64hi);

  // Add each 64 bit from each of the 128 bit lanes of the 256 bit register.
  const __m128i sse_reg128 = _mm_add_epi64(
      _mm256_castsi256_si128(sse_reg), _mm256_extracti128_si256(sse_reg, 1));
  const __m128i ssz_reg128 = _mm_add_epi64(
      _mm256_castsi256_si128(ssz_reg), _mm256_extracti128_si256(ssz_reg, 1));

  // Scale down for high bit-depth and store the results.
  int64_t sse, sqcoeff;
  _mm_storel_epi64((__m128i *)&sse, sse_reg128);
  _mm_storel_epi64((__m128i *)&sqcoeff, ssz_reg128);
  const int shift = 2 * (bd - 8);
  const int rounding = (1 << shift) >> 1;
  sse = (sse + rounding) >> shift;
  sqcoeff = (sqcoeff + rounding) >> shift;

  *ssz = sqcoeff;
  return sse;
}
