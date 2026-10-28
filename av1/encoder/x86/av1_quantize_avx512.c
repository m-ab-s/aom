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

#include "config/av1_rtcd.h"

#include "aom/aom_integer.h"
#include "aom_dsp/aom_dsp_common.h"

static inline __m512i mm512_sign_epi16(__m512i a, __m512i b) {
  const __m512i zero = _mm512_setzero_si512();
  const __mmask32 nz_mask = _mm512_cmpneq_epi16_mask(b, zero);
  const __mmask32 neg_mask = _mm512_cmpgt_epi16_mask(zero, b);
  const __m512i a_nz = _mm512_maskz_mov_epi16(nz_mask, a);
  return _mm512_mask_sub_epi16(a_nz, neg_mask, zero, a_nz);
}

static inline void write_zero_32_avx512(tran_low_t *qcoeff) {
  const __m512i zero = _mm512_setzero_si512();
  _mm512_storeu_si512((__m512i *)qcoeff, zero);
  _mm512_storeu_si512((__m512i *)(qcoeff + 16), zero);
}

static inline void init_one_qp_avx512(const __m128i *p, __m512i *qp) {
  const __m128i ac = _mm_unpackhi_epi64(*p, *p);
  const __m256i dc_ac =
      _mm256_inserti128_si256(_mm256_castsi128_si256(*p), ac, 1);
  const __m256i ac256 = _mm256_broadcastq_epi64(ac);
  *qp = _mm512_inserti64x4(_mm512_castsi256_si512(dc_ac), ac256, 1);
}

static inline void init_qp_avx512(const int16_t *round_ptr,
                                  const int16_t *quant_ptr,
                                  const int16_t *dequant_ptr, int log_scale,
                                  __m512i *thr, __m512i *qp) {
  __m128i round = _mm_loadu_si128((const __m128i *)round_ptr);
  const __m128i quant = _mm_loadu_si128((const __m128i *)quant_ptr);
  const __m128i dequant = _mm_loadu_si128((const __m128i *)dequant_ptr);

  if (log_scale > 0) {
    const __m128i rnd = _mm_set1_epi16((int16_t)1 << (log_scale - 1));
    round = _mm_add_epi16(round, rnd);
    round = _mm_srai_epi16(round, log_scale);
  }

  init_one_qp_avx512(&round, &qp[0]);
  init_one_qp_avx512(&quant, &qp[1]);

  if (log_scale == 1) {
    qp[1] = _mm512_slli_epi16(qp[1], (unsigned int)log_scale);
  }

  init_one_qp_avx512(&dequant, &qp[2]);
  *thr = _mm512_srai_epi16(qp[2], (unsigned int)(1 + log_scale));
  *thr = _mm512_sub_epi16(*thr, _mm512_set1_epi16(1));
}

static inline void update_qp_avx512(__m512i *thr, __m512i *qp) {
  qp[0] = _mm512_shuffle_i64x2(qp[0], qp[0], 0xFF);
  qp[1] = _mm512_shuffle_i64x2(qp[1], qp[1], 0xFF);
  qp[2] = _mm512_shuffle_i64x2(qp[2], qp[2], 0xFF);
  *thr = _mm512_shuffle_i64x2(*thr, *thr, 0xFF);
}

static inline __m512i load_coefficients_avx512(const tran_low_t *coeff_ptr) {
  const __m512i coeff1 = _mm512_loadu_si512((const __m512i *)coeff_ptr);
  const __m512i coeff2 = _mm512_loadu_si512((const __m512i *)(coeff_ptr + 16));
  return _mm512_packs_epi32(coeff1, coeff2);
}

static inline void store_coefficients_avx512(__m512i coeff_vals,
                                             tran_low_t *coeff_ptr) {
  const __m512i coeff_sign = _mm512_srai_epi16(coeff_vals, 15);
  const __m512i coeff_vals_lo = _mm512_unpacklo_epi16(coeff_vals, coeff_sign);
  const __m512i coeff_vals_hi = _mm512_unpackhi_epi16(coeff_vals, coeff_sign);
  _mm512_storeu_si512((__m512i *)coeff_ptr, coeff_vals_lo);
  _mm512_storeu_si512((__m512i *)(coeff_ptr + 16), coeff_vals_hi);
}

static inline uint16_t quant_gather_eob_avx512(__m512i eob512) {
  const __m256i eob256 = _mm256_max_epi16(_mm512_castsi512_si256(eob512),
                                          _mm512_extracti64x4_epi64(eob512, 1));
  const __m128i eob_lo = _mm256_castsi256_si128(eob256);
  const __m128i eob_hi = _mm256_extracti128_si256(eob256, 1);
  __m128i eob_s = _mm_max_epi16(eob_lo, eob_hi);
  eob_s = _mm_subs_epu16(_mm_set1_epi16(INT16_MAX), eob_s);
  eob_s = _mm_minpos_epu16(eob_s);
  return (uint16_t)(INT16_MAX - _mm_extract_epi16(eob_s, 0));
}

static AOM_FORCE_INLINE void quantize_lp_32_avx512(
    const int16_t *coeff_ptr, const int16_t *iscan_ptr, int16_t *qcoeff_ptr,
    int16_t *dqcoeff_ptr, const __m512i *round512, const __m512i *quant512,
    const __m512i *dequant512, __m512i *eob) {
  const __m512i coeff = _mm512_loadu_si512((const __m512i *)coeff_ptr);
  const __m512i abs_coeff = _mm512_abs_epi16(coeff);
  const __m512i tmp_rnd = _mm512_adds_epi16(abs_coeff, *round512);
  const __m512i abs_qcoeff = _mm512_mulhi_epi16(tmp_rnd, *quant512);
  const __m512i qcoeff = mm512_sign_epi16(abs_qcoeff, coeff);
  const __m512i dqcoeff = _mm512_mullo_epi16(qcoeff, *dequant512);
  const __mmask32 nz_mask =
      _mm512_cmpgt_epi16_mask(abs_qcoeff, _mm512_setzero_si512());

  _mm512_storeu_si512((__m512i *)qcoeff_ptr, qcoeff);
  _mm512_storeu_si512((__m512i *)dqcoeff_ptr, dqcoeff);

  if (nz_mask) {
    const __m512i iscan = _mm512_loadu_si512((const __m512i *)iscan_ptr);
    const __m512i nz_iscan =
        _mm512_maskz_add_epi16(nz_mask, iscan, _mm512_set1_epi16(1));
    *eob = _mm512_max_epi16(*eob, nz_iscan);
  }
}

void av1_quantize_lp_avx512(const int16_t *coeff_ptr, intptr_t n_coeffs,
                            const int16_t *round_ptr, const int16_t *quant_ptr,
                            int16_t *qcoeff_ptr, int16_t *dqcoeff_ptr,
                            const int16_t *dequant_ptr, uint16_t *eob_ptr,
                            const int16_t *scan, const int16_t *iscan) {
  if (n_coeffs < 32) {
    av1_quantize_lp_avx2(coeff_ptr, n_coeffs, round_ptr, quant_ptr, qcoeff_ptr,
                         dqcoeff_ptr, dequant_ptr, eob_ptr, scan, iscan);
    return;
  }
  (void)scan;
  __m512i eob512 = _mm512_setzero_si512();
  __m512i round512, quant512, dequant512;

  const __m128i round128 = _mm_loadu_si128((const __m128i *)round_ptr);
  const __m128i quant128 = _mm_loadu_si128((const __m128i *)quant_ptr);
  const __m128i dequant128 = _mm_loadu_si128((const __m128i *)dequant_ptr);

  init_one_qp_avx512(&round128, &round512);
  init_one_qp_avx512(&quant128, &quant512);
  init_one_qp_avx512(&dequant128, &dequant512);

  quantize_lp_32_avx512(coeff_ptr, iscan, qcoeff_ptr, dqcoeff_ptr, &round512,
                        &quant512, &dequant512, &eob512);

  if (n_coeffs > 32) {
    round512 = _mm512_shuffle_i64x2(round512, round512, 0xFF);
    quant512 = _mm512_shuffle_i64x2(quant512, quant512, 0xFF);
    dequant512 = _mm512_shuffle_i64x2(dequant512, dequant512, 0xFF);

    for (intptr_t idx = 32; idx < n_coeffs; idx += 32) {
      quantize_lp_32_avx512(coeff_ptr + idx, iscan + idx, qcoeff_ptr + idx,
                            dqcoeff_ptr + idx, &round512, &quant512,
                            &dequant512, &eob512);
    }
  }

  *eob_ptr = quant_gather_eob_avx512(eob512);
}

static AOM_FORCE_INLINE __m512i get_max_lane_eob_avx512(const int16_t *iscan,
                                                        __m512i v_eobmax,
                                                        __mmask32 nz_mask) {
  const __m512i v_iscan = _mm512_loadu_si512((const __m512i *)iscan);
  const __m512i perm_idx = _mm512_setr_epi64(0, 4, 1, 5, 2, 6, 3, 7);
  const __m512i v_iscan_perm = _mm512_permutexvar_epi64(perm_idx, v_iscan);
  const __m512i v_nz_iscan =
      _mm512_maskz_add_epi16(nz_mask, v_iscan_perm, _mm512_set1_epi16(1));
  return _mm512_max_epi16(v_eobmax, v_nz_iscan);
}

static AOM_FORCE_INLINE void quantize_fp_32_avx512(
    const __m512i *thr, const __m512i *qp, const tran_low_t *coeff_ptr,
    const int16_t *iscan_ptr, tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
    __m512i *eob) {
  const __m512i coeff = load_coefficients_avx512(coeff_ptr);
  const __m512i abs_coeff = _mm512_abs_epi16(coeff);
  const __mmask32 mask = _mm512_cmpgt_epi16_mask(abs_coeff, *thr);

  if (mask) {
    const __m512i tmp_rnd = _mm512_adds_epi16(abs_coeff, qp[0]);
    const __m512i abs_q = _mm512_mulhi_epi16(tmp_rnd, qp[1]);
    const __m512i q = mm512_sign_epi16(abs_q, coeff);
    const __m512i dq = _mm512_mullo_epi16(q, qp[2]);
    const __mmask32 nz_mask =
        _mm512_cmpgt_epi16_mask(abs_q, _mm512_setzero_si512());

    store_coefficients_avx512(q, qcoeff_ptr);
    store_coefficients_avx512(dq, dqcoeff_ptr);

    *eob = get_max_lane_eob_avx512(iscan_ptr, *eob, nz_mask);
  } else {
    write_zero_32_avx512(qcoeff_ptr);
    write_zero_32_avx512(dqcoeff_ptr);
  }
}

void av1_quantize_fp_avx512(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                            const int16_t *zbin_ptr, const int16_t *round_ptr,
                            const int16_t *quant_ptr,
                            const int16_t *quant_shift_ptr,
                            tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                            const int16_t *dequant_ptr, uint16_t *eob_ptr,
                            const int16_t *scan_ptr, const int16_t *iscan_ptr) {
  if (n_coeffs < 32) {
    av1_quantize_fp_avx2(coeff_ptr, n_coeffs, zbin_ptr, round_ptr, quant_ptr,
                         quant_shift_ptr, qcoeff_ptr, dqcoeff_ptr, dequant_ptr,
                         eob_ptr, scan_ptr, iscan_ptr);
    return;
  }
  (void)scan_ptr;
  (void)zbin_ptr;
  (void)quant_shift_ptr;

  const int log_scale = 0;
  const int step = 32;
  __m512i qp[3], thr;
  __m512i eob = _mm512_setzero_si512();

  init_qp_avx512(round_ptr, quant_ptr, dequant_ptr, log_scale, &thr, qp);

  quantize_fp_32_avx512(&thr, qp, coeff_ptr, iscan_ptr, qcoeff_ptr, dqcoeff_ptr,
                        &eob);

  coeff_ptr += step;
  qcoeff_ptr += step;
  dqcoeff_ptr += step;
  iscan_ptr += step;
  n_coeffs -= step;

  update_qp_avx512(&thr, qp);

  while (n_coeffs > 0) {
    quantize_fp_32_avx512(&thr, qp, coeff_ptr, iscan_ptr, qcoeff_ptr,
                          dqcoeff_ptr, &eob);

    coeff_ptr += step;
    qcoeff_ptr += step;
    dqcoeff_ptr += step;
    iscan_ptr += step;
    n_coeffs -= step;
  }
  *eob_ptr = quant_gather_eob_avx512(eob);
}

static AOM_FORCE_INLINE void quantize_fp_32x32_avx512(
    const __m512i *thr, const __m512i *qp, const tran_low_t *coeff_ptr,
    const int16_t *iscan_ptr, tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
    __m512i *eob) {
  const __m512i coeff = load_coefficients_avx512(coeff_ptr);
  const __m512i abs_coeff = _mm512_abs_epi16(coeff);
  const __mmask32 mask = _mm512_cmpgt_epi16_mask(abs_coeff, *thr);

  if (mask) {
    const __m512i tmp_rnd = _mm512_adds_epi16(abs_coeff, qp[0]);
    const __m512i abs_q = _mm512_mulhi_epu16(tmp_rnd, qp[1]);
    const __m512i q = mm512_sign_epi16(abs_q, coeff);
    const __m512i abs_dq =
        _mm512_srli_epi16(_mm512_mullo_epi16(abs_q, qp[2]), 1);
    const __mmask32 nz_mask =
        _mm512_cmpgt_epi16_mask(abs_q, _mm512_setzero_si512());
    const __m512i dq = mm512_sign_epi16(abs_dq, coeff);

    store_coefficients_avx512(q, qcoeff_ptr);
    store_coefficients_avx512(dq, dqcoeff_ptr);

    *eob = get_max_lane_eob_avx512(iscan_ptr, *eob, nz_mask);
  } else {
    write_zero_32_avx512(qcoeff_ptr);
    write_zero_32_avx512(dqcoeff_ptr);
  }
}

void av1_quantize_fp_32x32_avx512(
    const tran_low_t *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr,
    const int16_t *round_ptr, const int16_t *quant_ptr,
    const int16_t *quant_shift_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr,
    const int16_t *scan_ptr, const int16_t *iscan_ptr) {
  (void)scan_ptr;
  (void)zbin_ptr;
  (void)quant_shift_ptr;

  const int log_scale = 1;
  const unsigned int step = 32;
  __m512i qp[3], thr;
  __m512i eob = _mm512_setzero_si512();

  init_qp_avx512(round_ptr, quant_ptr, dequant_ptr, log_scale, &thr, qp);

  quantize_fp_32x32_avx512(&thr, qp, coeff_ptr, iscan_ptr, qcoeff_ptr,
                           dqcoeff_ptr, &eob);

  coeff_ptr += step;
  qcoeff_ptr += step;
  dqcoeff_ptr += step;
  iscan_ptr += step;
  n_coeffs -= step;

  update_qp_avx512(&thr, qp);

  while (n_coeffs > 0) {
    quantize_fp_32x32_avx512(&thr, qp, coeff_ptr, iscan_ptr, qcoeff_ptr,
                             dqcoeff_ptr, &eob);

    coeff_ptr += step;
    qcoeff_ptr += step;
    dqcoeff_ptr += step;
    iscan_ptr += step;
    n_coeffs -= step;
  }
  *eob_ptr = quant_gather_eob_avx512(eob);
}

static inline void quantize_fp_64x64_avx512(
    const __m512i *thr, const __m512i *qp, const tran_low_t *coeff_ptr,
    const int16_t *iscan_ptr, tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
    __m512i *eob) {
  const __m512i coeff = load_coefficients_avx512(coeff_ptr);
  const __m512i abs_coeff = _mm512_abs_epi16(coeff);
  const __mmask32 mask = _mm512_cmpgt_epi16_mask(abs_coeff, *thr);

  if (mask) {
    const __m512i tmp_rnd = _mm512_maskz_adds_epi16(mask, abs_coeff, qp[0]);
    const __m512i qh = _mm512_slli_epi16(_mm512_mulhi_epi16(tmp_rnd, qp[1]), 2);
    const __m512i ql =
        _mm512_srli_epi16(_mm512_mullo_epi16(tmp_rnd, qp[1]), 14);
    const __m512i abs_q = _mm512_or_si512(qh, ql);
    const __m512i dqh = _mm512_slli_epi16(_mm512_mulhi_epi16(abs_q, qp[2]), 14);
    const __m512i dql = _mm512_srli_epi16(_mm512_mullo_epi16(abs_q, qp[2]), 2);
    const __m512i abs_dq = _mm512_or_si512(dqh, dql);
    const __m512i q = mm512_sign_epi16(abs_q, coeff);
    const __m512i dq = mm512_sign_epi16(abs_dq, coeff);
    const __mmask32 nz_mask =
        _mm512_cmpneq_epi16_mask(dq, _mm512_setzero_si512());

    store_coefficients_avx512(q, qcoeff_ptr);
    store_coefficients_avx512(dq, dqcoeff_ptr);

    *eob = get_max_lane_eob_avx512(iscan_ptr, *eob, nz_mask);
  } else {
    write_zero_32_avx512(qcoeff_ptr);
    write_zero_32_avx512(dqcoeff_ptr);
  }
}

void av1_quantize_fp_64x64_avx512(
    const tran_low_t *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr,
    const int16_t *round_ptr, const int16_t *quant_ptr,
    const int16_t *quant_shift_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr,
    const int16_t *scan_ptr, const int16_t *iscan_ptr) {
  (void)scan_ptr;
  (void)zbin_ptr;
  (void)quant_shift_ptr;

  const int log_scale = 2;
  const unsigned int step = 32;
  __m512i qp[3], thr;
  __m512i eob = _mm512_setzero_si512();

  init_qp_avx512(round_ptr, quant_ptr, dequant_ptr, log_scale, &thr, qp);

  quantize_fp_64x64_avx512(&thr, qp, coeff_ptr, iscan_ptr, qcoeff_ptr,
                           dqcoeff_ptr, &eob);

  coeff_ptr += step;
  qcoeff_ptr += step;
  dqcoeff_ptr += step;
  iscan_ptr += step;
  n_coeffs -= step;

  update_qp_avx512(&thr, qp);

  while (n_coeffs > 0) {
    quantize_fp_64x64_avx512(&thr, qp, coeff_ptr, iscan_ptr, qcoeff_ptr,
                             dqcoeff_ptr, &eob);

    coeff_ptr += step;
    qcoeff_ptr += step;
    dqcoeff_ptr += step;
    iscan_ptr += step;
    n_coeffs -= step;
  }
  *eob_ptr = quant_gather_eob_avx512(eob);
}

int64_t av1_block_error_lp_avx512(const int16_t *coeff, const int16_t *dqcoeff,
                                  intptr_t block_size) {
  assert(block_size % 16 == 0);
  if (block_size == 16) {
    return av1_block_error_lp_avx2(coeff, dqcoeff, block_size);
  }
  const __m512i zero = _mm512_setzero_si512();
  __m512i sse_512 = zero;
  intptr_t i = 0;

  for (; i + 64 <= block_size; i += 64) {
    const __m512i c0 = _mm512_loadu_si512((const __m512i *)(coeff + i));
    const __m512i d0 = _mm512_loadu_si512((const __m512i *)(dqcoeff + i));
    const __m512i c1 = _mm512_loadu_si512((const __m512i *)(coeff + i + 32));
    const __m512i d1 = _mm512_loadu_si512((const __m512i *)(dqcoeff + i + 32));

    const __m512i diff0 = _mm512_sub_epi16(d0, c0);
    const __m512i diff1 = _mm512_sub_epi16(d1, c1);
    const __m512i err0 = _mm512_madd_epi16(diff0, diff0);
    const __m512i err1 = _mm512_madd_epi16(diff1, diff1);

    const __m512i exp0_lo = _mm512_unpacklo_epi32(err0, zero);
    const __m512i exp0_hi = _mm512_unpackhi_epi32(err0, zero);
    const __m512i exp1_lo = _mm512_unpacklo_epi32(err1, zero);
    const __m512i exp1_hi = _mm512_unpackhi_epi32(err1, zero);

    const __m512i sum0 = _mm512_add_epi64(exp0_lo, exp0_hi);
    const __m512i sum1 = _mm512_add_epi64(exp1_lo, exp1_hi);
    sse_512 = _mm512_add_epi64(sse_512, _mm512_add_epi64(sum0, sum1));
  }

  for (; i < block_size; i += 32) {
    const __m512i c0 = _mm512_loadu_si512((const __m512i *)(coeff + i));
    const __m512i d0 = _mm512_loadu_si512((const __m512i *)(dqcoeff + i));
    const __m512i diff0 = _mm512_sub_epi16(d0, c0);
    const __m512i err0 = _mm512_madd_epi16(diff0, diff0);
    const __m512i exp0_lo = _mm512_unpacklo_epi32(err0, zero);
    const __m512i exp0_hi = _mm512_unpackhi_epi32(err0, zero);
    sse_512 = _mm512_add_epi64(sse_512, _mm512_add_epi64(exp0_lo, exp0_hi));
  }

  return _mm512_reduce_add_epi64(sse_512);
}

int64_t av1_block_error_avx512(const tran_low_t *coeff,
                               const tran_low_t *dqcoeff, intptr_t block_size,
                               int64_t *ssz) {
  if (block_size < 32) {
    return av1_block_error_avx2(coeff, dqcoeff, block_size, ssz);
  }
  const __m512i zero = _mm512_setzero_si512();
  __m512i sse_reg = zero;
  __m512i ssz_reg = zero;

  for (intptr_t i = 0; i < block_size; i += 32) {
    const __m512i c0 = _mm512_loadu_si512((const __m512i *)(coeff + i));
    const __m512i c1 = _mm512_loadu_si512((const __m512i *)(coeff + i + 16));
    const __m512i coeff_reg = _mm512_packs_epi32(c0, c1);

    const __m512i d0 = _mm512_loadu_si512((const __m512i *)(dqcoeff + i));
    const __m512i d1 = _mm512_loadu_si512((const __m512i *)(dqcoeff + i + 16));
    const __m512i dqcoeff_reg = _mm512_packs_epi32(d0, d1);

    const __m512i diff = _mm512_sub_epi16(dqcoeff_reg, coeff_reg);
    const __m512i err = _mm512_madd_epi16(diff, diff);
    const __m512i sq_coeff = _mm512_madd_epi16(coeff_reg, coeff_reg);

    const __m512i exp_err_lo = _mm512_unpacklo_epi32(err, zero);
    const __m512i exp_err_hi = _mm512_unpackhi_epi32(err, zero);
    const __m512i exp_coeff_lo = _mm512_unpacklo_epi32(sq_coeff, zero);
    const __m512i exp_coeff_hi = _mm512_unpackhi_epi32(sq_coeff, zero);

    sse_reg =
        _mm512_add_epi64(sse_reg, _mm512_add_epi64(exp_err_lo, exp_err_hi));
    ssz_reg =
        _mm512_add_epi64(ssz_reg, _mm512_add_epi64(exp_coeff_lo, exp_coeff_hi));
  }

  *ssz = _mm512_reduce_add_epi64(ssz_reg);
  return _mm512_reduce_add_epi64(sse_reg);
}
