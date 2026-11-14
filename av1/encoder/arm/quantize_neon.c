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

#include <arm_neon.h>

#include <assert.h>
#include <math.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"

#include "aom_dsp/arm/mem_neon.h"
#include "aom_dsp/arm/sum_neon.h"
#include "aom_mem/aom_mem.h"

#include "av1/common/quant_common.h"
#include "av1/common/seg_common.h"

#include "av1/encoder/av1_quantize.h"
#include "av1/encoder/encoder.h"
#include "av1/encoder/rd.h"

static AOM_FORCE_INLINE uint64_t mask_nonzero_u16(uint16x8_t v_mask) {
#if AOM_ARCH_AARCH64
  return vmaxvq_u32(vreinterpretq_u32_u16(v_mask));
#else
  return vget_lane_u64(vreinterpret_u64_u8(vmovn_u16(v_mask)), 0);
#endif
}

static AOM_FORCE_INLINE void store_zero_8_tran_low(tran_low_t *ptr) {
  const int32x4_t zero = vdupq_n_s32(0);
  vst1q_s32(ptr, zero);
  vst1q_s32(ptr + 4, zero);
}

static inline uint16_t get_max_eob(int16x8_t v_eobmax) {
#if AOM_ARCH_AARCH64
  int16_t max_val = vmaxvq_s16(v_eobmax);
  return (uint16_t)max_val + 1;
#else
  const int16x4_t v_eobmax_3210 =
      vmax_s16(vget_low_s16(v_eobmax), vget_high_s16(v_eobmax));
  const int64x1_t v_eobmax_xx32 =
      vshr_n_s64(vreinterpret_s64_s16(v_eobmax_3210), 32);
  const int16x4_t v_eobmax_tmp =
      vmax_s16(v_eobmax_3210, vreinterpret_s16_s64(v_eobmax_xx32));
  const int64x1_t v_eobmax_xxx3 =
      vshr_n_s64(vreinterpret_s64_s16(v_eobmax_tmp), 16);
  const int16x4_t v_eobmax_final =
      vmax_s16(v_eobmax_tmp, vreinterpret_s16_s64(v_eobmax_xxx3));
  return (uint16_t)vget_lane_s16(v_eobmax_final, 0) + 1;
#endif
}

static inline int16x8_t get_max_lane_eob(const int16_t *iscan,
                                         int16x8_t v_eobmax,
                                         uint16x8_t v_mask) {
  const int16x8_t v_iscan = vld1q_s16(iscan);
  const int16x8_t v_nz_iscan = vbslq_s16(v_mask, v_iscan, vdupq_n_s16(-1));
  return vmaxq_s16(v_eobmax, v_nz_iscan);
}

static AOM_FORCE_INLINE int16x8_t
quantize_fp_8(const tran_low_t *coeff_ptr, tran_low_t *qcoeff_ptr,
              tran_low_t *dqcoeff_ptr, const int16_t *iscan, int16x8_t v_quant,
              int16x8_t v_dequant, int16x8_t v_round, uint16x8_t v_thresh,
              int16x8_t v_eobmax) {
  const int16x8_t v_coeff = load_tran_low_to_s16q(&coeff_ptr[0]);
  const int16x8_t v_abs = vabsq_s16(v_coeff);
  const uint16x8_t v_mask = vcgeq_u16(vreinterpretq_u16_s16(v_abs), v_thresh);
  if (mask_nonzero_u16(v_mask)) {
    const int16x8_t v_coeff_sign = vshrq_n_s16(v_coeff, 15);
    const int16x8_t v_tmp = vqaddq_s16(v_abs, v_round);
    const int16x8_t v_tmp2 = vshrq_n_s16(vqdmulhq_s16(v_tmp, v_quant), 1);
    const uint16x8_t v_nz_mask = vcgtq_s16(v_tmp2, vdupq_n_s16(0));
    const int16x8_t v_qcoeff_a = veorq_s16(v_tmp2, v_coeff_sign);
    const int16x8_t v_qcoeff = vsubq_s16(v_qcoeff_a, v_coeff_sign);
    const int16x8_t v_dqcoeff = vmulq_s16(v_qcoeff, v_dequant);
    store_s16q_to_tran_low(&qcoeff_ptr[0], v_qcoeff);
    store_s16q_to_tran_low(&dqcoeff_ptr[0], v_dqcoeff);
    return get_max_lane_eob(iscan, v_eobmax, v_nz_mask);
  }
  store_zero_8_tran_low(&qcoeff_ptr[0]);
  store_zero_8_tran_low(&dqcoeff_ptr[0]);
  return v_eobmax;
}

void av1_quantize_fp_neon(const tran_low_t *coeff_ptr, intptr_t count,
                          const int16_t *zbin_ptr, const int16_t *round_ptr,
                          const int16_t *quant_ptr,
                          const int16_t *quant_shift_ptr,
                          tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                          const int16_t *dequant_ptr, uint16_t *eob_ptr,
                          const int16_t *scan, const int16_t *iscan) {
  // TODO(jingning) Decide the need of these arguments after the
  // quantization process is completed.
  (void)zbin_ptr;
  (void)quant_shift_ptr;
  (void)scan;
  assert(count >= 16 && (count & 7) == 0);

  int16x8_t v_quant = vld1q_s16(quant_ptr);
  int16x8_t v_dequant = vld1q_s16(dequant_ptr);
  int16x8_t v_round = vld1q_s16(round_ptr);
  uint16x8_t v_thresh = vreinterpretq_u16_s16(vshrq_n_s16(v_dequant, 1));
  int16x8_t v_eobmax_76543210 = vdupq_n_s16(-1);

  // process dc and the first seven ac coeffs
  v_eobmax_76543210 =
      quantize_fp_8(coeff_ptr, qcoeff_ptr, dqcoeff_ptr, iscan, v_quant,
                    v_dequant, v_round, v_thresh, v_eobmax_76543210);

  // overwrite the dc constants with ac constants
  v_quant = vdupq_lane_s16(vget_low_s16(v_quant), 1);
  v_dequant = vdupq_lane_s16(vget_low_s16(v_dequant), 1);
  v_round = vdupq_lane_s16(vget_low_s16(v_round), 1);
  v_thresh = vdupq_lane_u16(vget_low_u16(v_thresh), 1);

  count -= 8;
  // now process the rest of the ac coeffs
  do {
    coeff_ptr += 8;
    qcoeff_ptr += 8;
    dqcoeff_ptr += 8;
    iscan += 8;
    v_eobmax_76543210 =
        quantize_fp_8(coeff_ptr, qcoeff_ptr, dqcoeff_ptr, iscan, v_quant,
                      v_dequant, v_round, v_thresh, v_eobmax_76543210);
    count -= 8;
  } while (count != 0);
  *eob_ptr = get_max_eob(v_eobmax_76543210);
}

static inline uint16x8_t quantize_lp_8(const int16_t *coeff_ptr,
                                       int16_t *qcoeff_ptr,
                                       int16_t *dqcoeff_ptr, int16x8_t v_quant,
                                       int16x8_t v_dequant, int16x8_t v_round,
                                       int16x8_t v_zero) {
  const int16x8_t v_coeff = vld1q_s16(&coeff_ptr[0]);
  const int16x8_t v_coeff_sign = vshrq_n_s16(v_coeff, 15);
  const int16x8_t v_abs = vabsq_s16(v_coeff);
  const int16x8_t v_tmp = vqaddq_s16(v_abs, v_round);
  const int16x8_t v_tmp2 = vshrq_n_s16(vqdmulhq_s16(v_tmp, v_quant), 1);
  const uint16x8_t v_nz_mask = vcgtq_s16(v_tmp2, v_zero);
  const int16x8_t v_qcoeff_a = veorq_s16(v_tmp2, v_coeff_sign);
  const int16x8_t v_qcoeff = vsubq_s16(v_qcoeff_a, v_coeff_sign);
  const int16x8_t v_dqcoeff = vmulq_s16(v_qcoeff, v_dequant);
  vst1q_s16(qcoeff_ptr, v_qcoeff);
  vst1q_s16(dqcoeff_ptr, v_dqcoeff);
  return v_nz_mask;
}

void av1_quantize_lp_neon(const int16_t *coeff_ptr, intptr_t n_coeffs,
                          const int16_t *round_ptr, const int16_t *quant_ptr,
                          int16_t *qcoeff_ptr, int16_t *dqcoeff_ptr,
                          const int16_t *dequant_ptr, uint16_t *eob_ptr,
                          const int16_t *scan, const int16_t *iscan) {
  (void)scan;
  // Quantization pass: All coefficients with index >= zero_flag are
  // skippable. Note: zero_flag can be zero.
  const int16x8_t v_zero = vdupq_n_s16(0);
  int16x8_t v_quant = vld1q_s16(quant_ptr);
  int16x8_t v_dequant = vld1q_s16(dequant_ptr);
  int16x8_t v_round = vld1q_s16(round_ptr);
  int16x8_t v_eobmax_76543210 = vdupq_n_s16(-1);
  uint16x8_t v_nz_mask;
  intptr_t count = n_coeffs;

  // process dc and the first seven ac coeffs
  v_nz_mask = quantize_lp_8(coeff_ptr, qcoeff_ptr, dqcoeff_ptr, v_quant,
                            v_dequant, v_round, v_zero);
  v_eobmax_76543210 = get_max_lane_eob(iscan, v_eobmax_76543210, v_nz_mask);
  // overwrite the dc constants with ac constants
  v_quant = vdupq_lane_s16(vget_low_s16(v_quant), 1);
  v_dequant = vdupq_lane_s16(vget_low_s16(v_dequant), 1);
  v_round = vdupq_lane_s16(vget_low_s16(v_round), 1);

  count -= 8;
  // now process the rest of the ac coeffs
  do {
    coeff_ptr += 8;
    qcoeff_ptr += 8;
    dqcoeff_ptr += 8;
    iscan += 8;
    v_nz_mask = quantize_lp_8(coeff_ptr, qcoeff_ptr, dqcoeff_ptr, v_quant,
                              v_dequant, v_round, v_zero);
    v_eobmax_76543210 = get_max_lane_eob(iscan, v_eobmax_76543210, v_nz_mask);
    count -= 8;
  } while (count != 0);
  *eob_ptr = get_max_eob(v_eobmax_76543210);
}

static AOM_FORCE_INLINE int16x8_t quantize_fp_logscale_8(
    const tran_low_t *coeff_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, const int16_t *iscan, int16x8_t v_quant,
    int16x8_t v_dequant, int16x8_t v_round, uint16x8_t v_thresh,
    int16x8_t v_eobmax, int log_scale) {
  const int16x8_t v_coeff = load_tran_low_to_s16q(coeff_ptr);
  const int16x8_t v_abs_coeff = vabsq_s16(v_coeff);
  const uint16x8_t v_mask =
      vcgeq_u16(vreinterpretq_u16_s16(v_abs_coeff), v_thresh);
  if (mask_nonzero_u16(v_mask)) {
    const int16x8_t v_coeff_sign = vshrq_n_s16(v_coeff, 15);
    // const int64_t tmp = vmask ? (int64_t)abs_coeff + log_scaled_round : 0
    const int16x8_t v_tmp = vandq_s16(vqaddq_s16(v_abs_coeff, v_round),
                                      vreinterpretq_s16_u16(v_mask));
    const int16x8_t v_tmp2 =
        vqdmulhq_s16(vshlq_s16(v_tmp, vdupq_n_s16(log_scale - 1)), v_quant);
    const uint16x8_t v_nz_mask = vcgtq_s16(v_tmp2, vdupq_n_s16(0));
    const int16x8_t v_qcoeff =
        vsubq_s16(veorq_s16(v_tmp2, v_coeff_sign), v_coeff_sign);
    // Multiplying by dequant here will use all 16 bits. Cast to unsigned
    // before shifting right.
    const uint16x8_t v_abs_dqcoeff =
        vshlq_u16(vreinterpretq_u16_s16(vmulq_s16(v_tmp2, v_dequant)),
                  vdupq_n_s16(-log_scale));
    const int16x8_t v_dqcoeff =
        vsubq_s16(veorq_s16(vreinterpretq_s16_u16(v_abs_dqcoeff), v_coeff_sign),
                  v_coeff_sign);
    store_s16q_to_tran_low(qcoeff_ptr, v_qcoeff);
    store_s16q_to_tran_low(dqcoeff_ptr, v_dqcoeff);
    return get_max_lane_eob(iscan, v_eobmax, v_nz_mask);
  }
  store_zero_8_tran_low(qcoeff_ptr);
  store_zero_8_tran_low(dqcoeff_ptr);
  return v_eobmax;
}

static AOM_FORCE_INLINE int16x8_t quantize_fp_logscale2_8(
    const tran_low_t *coeff_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, const int16_t *iscan, int16x8_t v_quant,
    int16x8_t v_dequant, int16x8_t v_round, uint16x8_t v_thresh,
    int16x8_t v_eobmax) {
  const int16x8_t v_coeff = load_tran_low_to_s16q(coeff_ptr);
  const int16x8_t v_abs_coeff = vabsq_s16(v_coeff);
  const uint16x8_t v_mask =
      vcgeq_u16(vreinterpretq_u16_s16(v_abs_coeff), v_thresh);
  if (mask_nonzero_u16(v_mask)) {
    const int16x8_t v_coeff_sign = vshrq_n_s16(v_coeff, 15);
    // abs_coeff = vmask ? (int64_t)abs_coeff + log_scaled_round : 0
    const int16x8_t v_tmp = vandq_s16(vqaddq_s16(v_abs_coeff, v_round),
                                      vreinterpretq_s16_u16(v_mask));
    // tmp32 = (int)((abs_coeff * quant_ptr[rc != 0]) >> (16 - log_scale));
    const int16x8_t v_tmp2 =
        vorrq_s16(vshlq_n_s16(vqdmulhq_s16(v_tmp, v_quant), 1),
                  vreinterpretq_s16_u16(vshrq_n_u16(
                      vreinterpretq_u16_s16(vmulq_s16(v_tmp, v_quant)), 14)));
    const uint16x8_t v_nz_mask = vcgtq_s16(v_tmp2, vdupq_n_s16(0));
    const int16x8_t v_qcoeff =
        vsubq_s16(veorq_s16(v_tmp2, v_coeff_sign), v_coeff_sign);
    // abs_dqcoeff = (tmp32 * dequant_ptr[rc != 0]) >> log_scale;
    const int16x8_t v_abs_dqcoeff =
        vorrq_s16(vshlq_n_s16(vqdmulhq_s16(v_tmp2, v_dequant), 13),
                  vreinterpretq_s16_u16(vshrq_n_u16(
                      vreinterpretq_u16_s16(vmulq_s16(v_tmp2, v_dequant)), 2)));
    const int16x8_t v_dqcoeff =
        vsubq_s16(veorq_s16(v_abs_dqcoeff, v_coeff_sign), v_coeff_sign);
    store_s16q_to_tran_low(qcoeff_ptr, v_qcoeff);
    store_s16q_to_tran_low(dqcoeff_ptr, v_dqcoeff);
    return get_max_lane_eob(iscan, v_eobmax, v_nz_mask);
  }
  store_zero_8_tran_low(qcoeff_ptr);
  store_zero_8_tran_low(dqcoeff_ptr);
  return v_eobmax;
}

static AOM_FORCE_INLINE void quantize_fp_no_qmatrix_neon(
    const tran_low_t *coeff_ptr, intptr_t n_coeffs, const int16_t *round_ptr,
    const int16_t *quant_ptr, tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
    const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *iscan,
    int log_scale) {
  assert(n_coeffs > 16 && (n_coeffs & 15) == 0);
  int16x8_t v_quant = vld1q_s16(quant_ptr);
  int16x8_t v_dequant = vld1q_s16(dequant_ptr);
  const int16x8_t v_round_no_scale = vld1q_s16(round_ptr);
  int16x8_t v_round =
      vqrdmulhq_n_s16(v_round_no_scale, (int16_t)(1 << (15 - log_scale)));
  int16x8_t v_eobmax_76543210 = vdupq_n_s16(-1);
  intptr_t non_zero_count = n_coeffs;

  // Pre-scan pass: compute exact ceil(dequant / 2^(1 + log_scale)) threshold.
  const uint16x8_t v_bias = vdupq_n_u16((uint16_t)((1 << (1 + log_scale)) - 1));
  uint16x8_t v_thresh =
      vshlq_u16(vaddq_u16(vreinterpretq_u16_s16(v_dequant), v_bias),
                vdupq_n_s16(-(1 + log_scale)));
  const uint16x8_t v_zbin_u16 = vdupq_lane_u16(vget_low_u16(v_thresh), 1);
  do {
    const int16x8_t v_coeff_a =
        load_tran_low_to_s16q(coeff_ptr + non_zero_count - 8);
    const int16x8_t v_coeff_b =
        load_tran_low_to_s16q(coeff_ptr + non_zero_count - 16);
    const uint16x8_t v_abs_a = vreinterpretq_u16_s16(vabsq_s16(v_coeff_a));
    const uint16x8_t v_abs_b = vreinterpretq_u16_s16(vabsq_s16(v_coeff_b));
    const uint16x8_t v_mask_a = vcgeq_u16(v_abs_a, v_zbin_u16);
    const uint16x8_t v_mask_b = vcgeq_u16(v_abs_b, v_zbin_u16);
    if (mask_nonzero_u16(vorrq_u16(v_mask_a, v_mask_b))) {
      break;
    }
    non_zero_count -= 16;
    store_zero_8_tran_low(qcoeff_ptr + non_zero_count);
    store_zero_8_tran_low(qcoeff_ptr + non_zero_count + 8);
    store_zero_8_tran_low(dqcoeff_ptr + non_zero_count);
    store_zero_8_tran_low(dqcoeff_ptr + non_zero_count + 8);
  } while (non_zero_count != 16);

  // process dc and the first seven ac coeffs
  if (log_scale == 2) {
    v_eobmax_76543210 = quantize_fp_logscale2_8(
        coeff_ptr, qcoeff_ptr, dqcoeff_ptr, iscan, v_quant, v_dequant, v_round,
        v_thresh, v_eobmax_76543210);
  } else {
    v_eobmax_76543210 = quantize_fp_logscale_8(
        coeff_ptr, qcoeff_ptr, dqcoeff_ptr, iscan, v_quant, v_dequant, v_round,
        v_thresh, v_eobmax_76543210, log_scale);
  }
  // overwrite the dc constants with ac constants
  v_quant = vdupq_lane_s16(vget_low_s16(v_quant), 1);
  v_dequant = vdupq_lane_s16(vget_low_s16(v_dequant), 1);
  v_round = vdupq_lane_s16(vget_low_s16(v_round), 1);
  v_thresh = v_zbin_u16;

  intptr_t count = non_zero_count - 8;
  do {
    coeff_ptr += 8;
    qcoeff_ptr += 8;
    dqcoeff_ptr += 8;
    iscan += 8;
    if (log_scale == 2) {
      v_eobmax_76543210 = quantize_fp_logscale2_8(
          coeff_ptr, qcoeff_ptr, dqcoeff_ptr, iscan, v_quant, v_dequant,
          v_round, v_thresh, v_eobmax_76543210);
    } else {
      v_eobmax_76543210 = quantize_fp_logscale_8(
          coeff_ptr, qcoeff_ptr, dqcoeff_ptr, iscan, v_quant, v_dequant,
          v_round, v_thresh, v_eobmax_76543210, log_scale);
    }
    count -= 8;
  } while (count != 0);
  *eob_ptr = get_max_eob(v_eobmax_76543210);
}

void av1_quantize_fp_32x32_neon(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                                const int16_t *zbin_ptr,
                                const int16_t *round_ptr,
                                const int16_t *quant_ptr,
                                const int16_t *quant_shift_ptr,
                                tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                                const int16_t *dequant_ptr, uint16_t *eob_ptr,
                                const int16_t *scan, const int16_t *iscan) {
  (void)zbin_ptr;
  (void)quant_shift_ptr;
  (void)scan;
  quantize_fp_no_qmatrix_neon(coeff_ptr, n_coeffs, round_ptr, quant_ptr,
                              qcoeff_ptr, dqcoeff_ptr, dequant_ptr, eob_ptr,
                              iscan, 1);
}

void av1_quantize_fp_64x64_neon(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                                const int16_t *zbin_ptr,
                                const int16_t *round_ptr,
                                const int16_t *quant_ptr,
                                const int16_t *quant_shift_ptr,
                                tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                                const int16_t *dequant_ptr, uint16_t *eob_ptr,
                                const int16_t *scan, const int16_t *iscan) {
  (void)zbin_ptr;
  (void)quant_shift_ptr;
  (void)scan;
  quantize_fp_no_qmatrix_neon(coeff_ptr, n_coeffs, round_ptr, quant_ptr,
                              qcoeff_ptr, dqcoeff_ptr, dequant_ptr, eob_ptr,
                              iscan, 2);
}

static inline uint16x8_t quantize_b_logscale0_8(
    int16x8_t coeff, int16x8_t abs, uint16x8_t cond, int16x8_t round,
    int16x8_t dequant, int16x8_t quant, int16x8_t quant_shift,
    tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr) {
  const int16x8_t zero = vdupq_n_s16(0);

  int16x8_t coeff_sign = vreinterpretq_s16_u16(vcltq_s16(coeff, zero));

  int16x8_t tmp = vqaddq_s16(abs, round);
  tmp = vsraq_n_s16(tmp, vqdmulhq_s16(tmp, quant), 1);
  tmp = vqdmulhq_s16(tmp, quant_shift);

  int16x8_t qcoeff = vsubq_s16(veorq_s16(tmp, coeff_sign), coeff_sign);
  qcoeff = vbslq_s16(cond, qcoeff, zero);
  store_s16q_to_tran_low(qcoeff_ptr, qcoeff);

  int16x8_t dqcoeff = vmulq_s16(tmp, dequant);
  dqcoeff = vsubq_s16(veorq_s16(dqcoeff, coeff_sign), coeff_sign);
  dqcoeff = vbslq_s16(cond, dqcoeff, zero);
  store_s16q_to_tran_low(dqcoeff_ptr, dqcoeff);

  uint16x8_t tmp_mask = vcgtq_s16(tmp, zero);
  uint16x8_t nz_mask = vandq_u16(tmp_mask, cond);

  return nz_mask;
}

void aom_quantize_b_neon(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                         const int16_t *zbin_ptr, const int16_t *round_ptr,
                         const int16_t *quant_ptr,
                         const int16_t *quant_shift_ptr, tran_low_t *qcoeff_ptr,
                         tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr,
                         uint16_t *eob_ptr, const int16_t *scan,
                         const int16_t *iscan) {
  (void)scan;

  int16x8_t v_eobmax_76543210 = vdupq_n_s16(-1);

  const int16x8_t v_zbins = vdupq_n_s16(zbin_ptr[1]);
  int16x8_t v_round = vdupq_n_s16(round_ptr[1]);
  int16x8_t v_dequant = vdupq_n_s16(dequant_ptr[1]);
  int16x8_t v_quant = vdupq_n_s16(quant_ptr[1]);
  // Shift by 1 in order to save one shift in the kernel function.
  int16x8_t v_quant_shift = vdupq_n_s16(quant_shift_ptr[1] >> 1);

  int16x8_t v_zbins0 = vsetq_lane_s16(zbin_ptr[0], v_zbins, 0);
  int16x8_t v_coeff = load_tran_low_to_s16q(coeff_ptr);
  int16x8_t v_abs = vabsq_s16(v_coeff);
  uint16x8_t v_cond = vcgeq_s16(v_abs, v_zbins0);

  if (mask_nonzero_u16(v_cond)) {
    v_round = vsetq_lane_s16(round_ptr[0], v_round, 0);
    v_quant = vsetq_lane_s16(quant_ptr[0], v_quant, 0);
    v_dequant = vsetq_lane_s16(dequant_ptr[0], v_dequant, 0);
    // Shift by 1 in order to save one shift in the kernel function.
    v_quant_shift = vsetq_lane_s16(quant_shift_ptr[0] >> 1, v_quant_shift, 0);

    const uint16x8_t v_nz_mask =
        quantize_b_logscale0_8(v_coeff, v_abs, v_cond, v_round, v_dequant,
                               v_quant, v_quant_shift, qcoeff_ptr, dqcoeff_ptr);

    v_round = vdupq_lane_s16(vget_low_s16(v_round), 1);
    v_quant = vdupq_lane_s16(vget_low_s16(v_quant), 1);
    v_dequant = vdupq_lane_s16(vget_low_s16(v_dequant), 1);
    v_quant_shift = vdupq_lane_s16(vget_low_s16(v_quant_shift), 1);

    int16x8_t v_iscan = vld1q_s16(iscan);
    int16x8_t v_eobmax = vmaxq_s16(v_iscan, v_eobmax_76543210);
    v_eobmax_76543210 = vbslq_s16(v_nz_mask, v_eobmax, v_eobmax_76543210);
  } else {
    store_zero_8_tran_low(qcoeff_ptr);
    store_zero_8_tran_low(dqcoeff_ptr);
  }

  intptr_t i = 8;
  do {
    v_coeff = load_tran_low_to_s16q(coeff_ptr + i);
    v_abs = vabsq_s16(v_coeff);
    v_cond = vcgeq_s16(v_abs, v_zbins);

    if (mask_nonzero_u16(v_cond)) {
      const uint16x8_t v_nz_mask = quantize_b_logscale0_8(
          v_coeff, v_abs, v_cond, v_round, v_dequant, v_quant, v_quant_shift,
          qcoeff_ptr + i, dqcoeff_ptr + i);

      int16x8_t v_iscan = vld1q_s16(iscan + i);
      int16x8_t v_eobmax = vmaxq_s16(v_iscan, v_eobmax_76543210);
      v_eobmax_76543210 = vbslq_s16(v_nz_mask, v_eobmax, v_eobmax_76543210);
    } else {
      store_zero_8_tran_low(qcoeff_ptr + i);
      store_zero_8_tran_low(dqcoeff_ptr + i);
    }
    i += 8;
  } while (i != n_coeffs);
  *eob_ptr = get_max_eob(v_eobmax_76543210);
}

#define QM_MULL_SHIFT(x0, x1)                                              \
  vreinterpretq_s16_u16(vorrq_u16(                                         \
      vreinterpretq_u16_s16(vshlq_n_s16(                                   \
          vqdmulhq_s16(x0, vreinterpretq_s16_u16(x1)), 15 - AOM_QM_BITS)), \
      vshrq_n_u16(vmulq_u16(vreinterpretq_u16_s16(x0), x1), AOM_QM_BITS)))

static void aom_quantize_b_helper_16x16_neon(
    const tran_low_t *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr,
    const int16_t *round_ptr, const int16_t *quant_ptr,
    const int16_t *quant_shift_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr,
    const int16_t *scan, const int16_t *iscan, const qm_val_t *qm_ptr,
    const qm_val_t *iqm_ptr) {
  (void)scan;

  uint16x8_t vwt, viwt;
  const int zbins[2] = { zbin_ptr[0], zbin_ptr[1] };

  memset(qcoeff_ptr, 0, n_coeffs * sizeof(*qcoeff_ptr));
  memset(dqcoeff_ptr, 0, n_coeffs * sizeof(*dqcoeff_ptr));

  const int16x8_t zero = vdupq_n_s16(0);
  int16x8_t v_eobmax_76543210 = vreinterpretq_s16_u16(vceqq_s16(zero, zero));

  int16x8_t vzbins = vdupq_n_s16(zbins[1]), vround = vdupq_n_s16(round_ptr[1]);
  int16x8_t vdequant = vdupq_n_s16(dequant_ptr[1]);
  int16x8_t vquant = vdupq_n_s16(quant_ptr[1]);
  int16x8_t vquant_shift = vdupq_n_s16(quant_shift_ptr[1]);

  int16x8_t v_coeff = load_tran_low_to_s16q(&coeff_ptr[0]);
  int16x8_t v_coeff_sign = vshrq_n_s16(v_coeff, 15);
  int16x8_t v_abs = vabsq_s16(v_coeff);
  vzbins = vsetq_lane_s16(zbins[0], vzbins, 0);
  uint16x8_t vcond;
  if (qm_ptr == NULL) {
    vcond = vcgeq_s16(v_abs, vzbins);
  } else {
    vwt = vmovl_u8(vld1_u8(&qm_ptr[0]));
    vcond = vcgeq_s16(QM_MULL_SHIFT(v_abs, vwt), vzbins);
  }
  uint64_t nz_check = vget_lane_u64(vreinterpret_u64_u8(vmovn_u16(vcond)), 0);
  if (nz_check) {
    vround = vsetq_lane_s16(round_ptr[0], vround, 0);
    vquant = vsetq_lane_s16(quant_ptr[0], vquant, 0);
    vdequant = vsetq_lane_s16(dequant_ptr[0], vdequant, 0);
    vquant_shift = vsetq_lane_s16(quant_shift_ptr[0], vquant_shift, 0);

    int16x8_t vtmp = vqaddq_s16(v_abs, vround);

    int16x8_t vtmp2;
    if (qm_ptr == NULL) {
      vtmp2 = vsraq_n_s16(vtmp, vqdmulhq_s16(vtmp, vquant), 1);
    } else {
      vtmp2 = QM_MULL_SHIFT(vtmp, vwt);
      vtmp2 = vaddq_s16(vtmp2, vtmp);
    }

    vtmp2 = vshrq_n_s16(vqdmulhq_s16(vtmp2, vquant_shift), 1);
    int16x8_t vdest = vsubq_s16(veorq_s16(vtmp2, v_coeff_sign), v_coeff_sign);
    int16x8_t coeff_nz_mask =
        vbslq_s16(vcond, vdest, load_tran_low_to_s16q(&qcoeff_ptr[0]));
    store_s16q_to_tran_low(&qcoeff_ptr[0], coeff_nz_mask);

    if (iqm_ptr != NULL) {
      viwt = vmovl_u8(vld1_u8(&iqm_ptr[0]));
      vdequant = QM_MULL_SHIFT(vdequant, viwt);
    }
    int16x8_t v_deq_abs = vmulq_s16(vtmp2, vdequant);
    vdest = vsubq_s16(veorq_s16(v_deq_abs, v_coeff_sign), v_coeff_sign);
    coeff_nz_mask =
        vbslq_s16(vcond, vdest, load_tran_low_to_s16q(&dqcoeff_ptr[0]));
    store_s16q_to_tran_low(&dqcoeff_ptr[0], coeff_nz_mask);

    vround = vsetq_lane_s16(round_ptr[1], vround, 0);
    vquant = vsetq_lane_s16(quant_ptr[1], vquant, 0);
    vdequant = vsetq_lane_s16(dequant_ptr[1], vdequant, 0);
    vquant_shift = vsetq_lane_s16(quant_shift_ptr[1], vquant_shift, 0);

    uint16x8_t vtmp_mask = vcgtq_s16(vtmp2, zero);
    const uint16x8_t v_nz_mask = vandq_u16(vtmp_mask, vcond);
    int16x8_t v_iscan = vld1q_s16(&iscan[0]);
    vcond = vandq_u16(v_nz_mask, vcgtq_s16(v_iscan, v_eobmax_76543210));
    v_eobmax_76543210 = vbslq_s16(vcond, v_iscan, v_eobmax_76543210);
  }
  vzbins = vsetq_lane_s16(zbins[1], vzbins, 0);

  for (int i = 8; i < n_coeffs; i += 8) {
    v_coeff = load_tran_low_to_s16q(&coeff_ptr[i]);
    v_coeff_sign = vshrq_n_s16(v_coeff, 15);
    v_abs = vabsq_s16(v_coeff);

    if (qm_ptr == NULL) {
      vcond = vcgeq_s16(v_abs, vzbins);
    } else {
      vwt = vmovl_u8(vld1_u8(&qm_ptr[i]));
      vcond = vcgeq_s16(QM_MULL_SHIFT(v_abs, vwt), vzbins);
    }
    nz_check = vget_lane_u64(vreinterpret_u64_u8(vmovn_u16(vcond)), 0);
    if (nz_check) {
      int16x8_t vtmp = vqaddq_s16(v_abs, vround);

      int16x8_t vtmp2;
      if (qm_ptr == NULL) {
        vtmp2 = vsraq_n_s16(vtmp, vqdmulhq_s16(vtmp, vquant), 1);
      } else {
        vtmp2 = QM_MULL_SHIFT(vtmp, vwt);
        vtmp2 = vaddq_s16(vtmp2, vtmp);
      }

      vtmp2 = vshrq_n_s16(vqdmulhq_s16(vtmp2, vquant_shift), 1);
      int16x8_t vdest = vsubq_s16(veorq_s16(vtmp2, v_coeff_sign), v_coeff_sign);
      int16x8_t coeff_nz_mask =
          vbslq_s16(vcond, vdest, load_tran_low_to_s16q(&qcoeff_ptr[i]));
      store_s16q_to_tran_low(&qcoeff_ptr[i], coeff_nz_mask);

      if (iqm_ptr != NULL) {
        viwt = vmovl_u8(vld1_u8(&iqm_ptr[i]));
        vdequant = QM_MULL_SHIFT(vdequant, viwt);
      }
      int16x8_t v_deq_abs = vmulq_s16(vtmp2, vdequant);
      vdest = vsubq_s16(veorq_s16(v_deq_abs, v_coeff_sign), v_coeff_sign);
      coeff_nz_mask =
          vbslq_s16(vcond, vdest, load_tran_low_to_s16q(&dqcoeff_ptr[i]));
      store_s16q_to_tran_low(&dqcoeff_ptr[i], coeff_nz_mask);

      uint16x8_t vtmp_mask = vcgtq_s16(vtmp2, zero);
      const uint16x8_t v_nz_mask = vandq_u16(vtmp_mask, vcond);
      int16x8_t v_iscan = vld1q_s16(&iscan[i]);
      vcond = vandq_u16(v_nz_mask, vcgtq_s16(v_iscan, v_eobmax_76543210));
      v_eobmax_76543210 = vbslq_s16(vcond, v_iscan, v_eobmax_76543210);
    }
  }
  *eob_ptr = get_max_eob(v_eobmax_76543210);
}

static void aom_quantize_b_helper_32x32_neon(
    const tran_low_t *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr,
    const int16_t *round_ptr, const int16_t *quant_ptr,
    const int16_t *quant_shift_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr,
    const int16_t *scan, const int16_t *iscan, const qm_val_t *qm_ptr,
    const qm_val_t *iqm_ptr) {
  (void)scan;

  uint16x8_t vwt, viwt;
  const int log_scale = 1;
  const int zbins[2] = { ROUND_POWER_OF_TWO(zbin_ptr[0], log_scale),
                         ROUND_POWER_OF_TWO(zbin_ptr[1], log_scale) };

  memset(qcoeff_ptr, 0, n_coeffs * sizeof(*qcoeff_ptr));
  memset(dqcoeff_ptr, 0, n_coeffs * sizeof(*dqcoeff_ptr));

  const int16x8_t zero = vdupq_n_s16(0);
  int16x8_t v_eobmax_76543210 = vreinterpretq_s16_u16(vceqq_s16(zero, zero));
  const int16x8_t v_log_scale = v_eobmax_76543210;

  int16x8_t vzbins = vdupq_n_s16(zbins[1]),
            vround = vdupq_n_s16(ROUND_POWER_OF_TWO(round_ptr[1], log_scale));
  int16x8_t vdequant = vdupq_n_s16(dequant_ptr[1]);
  int16x8_t vquant = vdupq_n_s16(quant_ptr[1]);
  int16x8_t vquant_shift = vdupq_n_s16(quant_shift_ptr[1]);

  int16x8_t v_coeff = load_tran_low_to_s16q(&coeff_ptr[0]);
  int16x8_t v_coeff_sign = vshrq_n_s16(v_coeff, 15);
  int16x8_t v_abs = vabsq_s16(v_coeff);
  vzbins = vsetq_lane_s16(zbins[0], vzbins, 0);
  uint16x8_t vcond;
  if (qm_ptr == NULL) {
    vcond = vcgeq_s16(v_abs, vzbins);
  } else {
    vwt = vmovl_u8(vld1_u8(&qm_ptr[0]));
    vcond = vcgeq_s16(QM_MULL_SHIFT(v_abs, vwt), vzbins);
  }
  uint64_t nz_check = vget_lane_u64(vreinterpret_u64_u8(vmovn_u16(vcond)), 0);
  if (nz_check) {
    vround =
        vsetq_lane_s16(ROUND_POWER_OF_TWO(round_ptr[0], log_scale), vround, 0);
    vquant = vsetq_lane_s16(quant_ptr[0], vquant, 0);
    vdequant = vsetq_lane_s16(dequant_ptr[0], vdequant, 0);
    vquant_shift = vsetq_lane_s16(quant_shift_ptr[0], vquant_shift, 0);

    int16x8_t vtmp = vqaddq_s16(v_abs, vround);

    int16x8_t vtmp2;
    if (qm_ptr == NULL) {
      vtmp2 = vsraq_n_s16(vtmp, vqdmulhq_s16(vtmp, vquant), 1);
    } else {
      vtmp2 = QM_MULL_SHIFT(vtmp, vwt);
      vtmp2 = vaddq_s16(vtmp2, vtmp);
    }

    vtmp2 = vqdmulhq_s16(vtmp2, vquant_shift);
    int16x8_t vdest = vsubq_s16(veorq_s16(vtmp2, v_coeff_sign), v_coeff_sign);
    int16x8_t coeff_nz_mask =
        vbslq_s16(vcond, vdest, load_tran_low_to_s16q(&qcoeff_ptr[0]));
    store_s16q_to_tran_low(&qcoeff_ptr[0], coeff_nz_mask);

    if (iqm_ptr != NULL) {
      viwt = vmovl_u8(vld1_u8(&iqm_ptr[0]));
      vdequant = QM_MULL_SHIFT(vdequant, viwt);
    }
    int16x8_t v_deq_abs = vreinterpretq_s16_u16(vshlq_u16(
        vreinterpretq_u16_s16(vmulq_s16(vtmp2, vdequant)), v_log_scale));
    vdest = vsubq_s16(veorq_s16(v_deq_abs, v_coeff_sign), v_coeff_sign);
    coeff_nz_mask =
        vbslq_s16(vcond, vdest, load_tran_low_to_s16q(&dqcoeff_ptr[0]));
    store_s16q_to_tran_low(&dqcoeff_ptr[0], coeff_nz_mask);

    vzbins = vsetq_lane_s16(zbins[1], vzbins, 0);
    vround =
        vsetq_lane_s16(ROUND_POWER_OF_TWO(round_ptr[1], log_scale), vround, 0);
    vquant = vsetq_lane_s16(quant_ptr[1], vquant, 0);
    vdequant = vsetq_lane_s16(dequant_ptr[1], vdequant, 0);
    vquant_shift = vsetq_lane_s16(quant_shift_ptr[1], vquant_shift, 0);

    uint16x8_t vtmp_mask = vcgtq_s16(vtmp2, zero);
    const uint16x8_t v_nz_mask = vandq_u16(vtmp_mask, vcond);
    int16x8_t v_iscan = vld1q_s16(&iscan[0]);
    vcond = vandq_u16(v_nz_mask, vcgtq_s16(v_iscan, v_eobmax_76543210));
    v_eobmax_76543210 = vbslq_s16(vcond, v_iscan, v_eobmax_76543210);
  }
  vzbins = vsetq_lane_s16(zbins[1], vzbins, 0);

  for (int i = 8; i < n_coeffs; i += 8) {
    v_coeff = load_tran_low_to_s16q(&coeff_ptr[i]);
    v_coeff_sign = vshrq_n_s16(v_coeff, 15);
    v_abs = vabsq_s16(v_coeff);

    if (qm_ptr == NULL) {
      vcond = vcgeq_s16(v_abs, vzbins);
    } else {
      vwt = vmovl_u8(vld1_u8(&qm_ptr[i]));
      vcond = vcgeq_s16(QM_MULL_SHIFT(v_abs, vwt), vzbins);
    }
    nz_check = vget_lane_u64(vreinterpret_u64_u8(vmovn_u16(vcond)), 0);
    if (nz_check) {
      int16x8_t vtmp = vqaddq_s16(v_abs, vround);

      int16x8_t vtmp2;
      if (qm_ptr == NULL) {
        vtmp2 = vsraq_n_s16(vtmp, vqdmulhq_s16(vtmp, vquant), 1);
      } else {
        vtmp2 = QM_MULL_SHIFT(vtmp, vwt);
        vtmp2 = vaddq_s16(vtmp2, vtmp);
      }
      vtmp2 = vqdmulhq_s16(vtmp2, vquant_shift);

      int16x8_t vdest = vsubq_s16(veorq_s16(vtmp2, v_coeff_sign), v_coeff_sign);
      int16x8_t coeff_nz_mask =
          vbslq_s16(vcond, vdest, load_tran_low_to_s16q(&qcoeff_ptr[i]));
      store_s16q_to_tran_low(&qcoeff_ptr[i], coeff_nz_mask);

      if (iqm_ptr != NULL) {
        viwt = vmovl_u8(vld1_u8(&iqm_ptr[i]));
        vdequant = QM_MULL_SHIFT(vdequant, viwt);
      }
      int16x8_t v_deq_abs = vreinterpretq_s16_u16(vshlq_u16(
          vreinterpretq_u16_s16(vmulq_s16(vtmp2, vdequant)), v_log_scale));
      vdest = vsubq_s16(veorq_s16(v_deq_abs, v_coeff_sign), v_coeff_sign);
      coeff_nz_mask =
          vbslq_s16(vcond, vdest, load_tran_low_to_s16q(&dqcoeff_ptr[i]));
      store_s16q_to_tran_low(&dqcoeff_ptr[i], coeff_nz_mask);

      uint16x8_t vtmp_mask = vcgtq_s16(vtmp2, zero);
      const uint16x8_t v_nz_mask = vandq_u16(vtmp_mask, vcond);
      int16x8_t v_iscan = vld1q_s16(&iscan[i]);
      vcond = vandq_u16(v_nz_mask, vcgtq_s16(v_iscan, v_eobmax_76543210));
      v_eobmax_76543210 = vbslq_s16(vcond, v_iscan, v_eobmax_76543210);
    }
  }
  *eob_ptr = get_max_eob(v_eobmax_76543210);
}

static void aom_quantize_b_helper_64x64_neon(
    const tran_low_t *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr,
    const int16_t *round_ptr, const int16_t *quant_ptr,
    const int16_t *quant_shift_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr,
    const int16_t *scan, const int16_t *iscan, const qm_val_t *qm_ptr,
    const qm_val_t *iqm_ptr) {
  (void)scan;

  uint16x8_t vwt, viwt;
  const int log_scale = 2;
  const int16x8_t v_log_scale =
      vreinterpretq_s16_s64(vdupq_n_s64(0xFFFEFFFEFFFEFFFE));

  const int zbins[2] = { ROUND_POWER_OF_TWO(zbin_ptr[0], log_scale),
                         ROUND_POWER_OF_TWO(zbin_ptr[1], log_scale) };

  memset(qcoeff_ptr, 0, n_coeffs * sizeof(*qcoeff_ptr));
  memset(dqcoeff_ptr, 0, n_coeffs * sizeof(*dqcoeff_ptr));

  const int16x8_t zero = vdupq_n_s16(0);
  int16x8_t v_eobmax_76543210 = vreinterpretq_s16_u16(vceqq_s16(zero, zero));
  int16x8_t v_ones = vnegq_s16(v_eobmax_76543210);

  int16x8_t vzbins = vdupq_n_s16(zbins[1]),
            vround = vdupq_n_s16(ROUND_POWER_OF_TWO(round_ptr[1], log_scale));
  int16x8_t vdequant = vdupq_n_s16(dequant_ptr[1]);
  int16x8_t vquant = vdupq_n_s16(quant_ptr[1]);
  int16x8_t vquant_shift = vdupq_n_s16(quant_shift_ptr[1]);

  int16x8_t v_coeff = load_tran_low_to_s16q(&coeff_ptr[0]);
  int16x8_t v_coeff_sign = vshrq_n_s16(v_coeff, 15);
  int16x8_t v_abs = vabsq_s16(v_coeff);
  vzbins = vsetq_lane_s16(zbins[0], vzbins, 0);
  uint16x8_t vcond;
  if (qm_ptr == NULL) {
    vcond = vcgeq_s16(v_abs, vzbins);
  } else {
    vwt = vmovl_u8(vld1_u8(&qm_ptr[0]));
    vcond = vcgeq_s16(QM_MULL_SHIFT(v_abs, vwt), vzbins);
  }
  uint64_t nz_check = vget_lane_u64(vreinterpret_u64_u8(vmovn_u16(vcond)), 0);
  if (nz_check) {
    vround =
        vsetq_lane_s16(ROUND_POWER_OF_TWO(round_ptr[0], log_scale), vround, 0);
    vquant = vsetq_lane_s16(quant_ptr[0], vquant, 0);
    vdequant = vsetq_lane_s16(dequant_ptr[0], vdequant, 0);
    vquant_shift = vsetq_lane_s16(quant_shift_ptr[0], vquant_shift, 0);
    int16x8_t vtmp = vqaddq_s16(v_abs, vround);

    int16x8_t vtmp2;
    if (qm_ptr == NULL) {
      vtmp2 = vsraq_n_s16(vtmp, vqdmulhq_s16(vtmp, vquant), 1);
    } else {
      vtmp2 = QM_MULL_SHIFT(vtmp, vwt);
      vtmp2 = vaddq_s16(vtmp2, vtmp);
    }

    int16x8_t ones =
        vandq_s16(vshrq_n_s16(vmulq_s16(vtmp2, vquant_shift), 14), v_ones);
    vtmp2 =
        vaddq_s16(vshlq_s16(vqdmulhq_s16(vtmp2, vquant_shift), v_ones), ones);
    int16x8_t vdest = vsubq_s16(veorq_s16(vtmp2, v_coeff_sign), v_coeff_sign);
    int16x8_t coeff_nz_mask =
        vbslq_s16(vcond, vdest, load_tran_low_to_s16q(&qcoeff_ptr[0]));
    store_s16q_to_tran_low(&qcoeff_ptr[0], coeff_nz_mask);

    if (iqm_ptr != NULL) {
      viwt = vmovl_u8(vld1_u8(&iqm_ptr[0]));
      vdequant = QM_MULL_SHIFT(vdequant, viwt);
    }
    int16x8_t v_deq_abs = vreinterpretq_s16_u16(vshlq_u16(
        vreinterpretq_u16_s16(vmulq_s16(vtmp2, vdequant)), v_log_scale));
    v_deq_abs =
        vorrq_s16(vshlq_n_s16(vqdmulhq_s16(vtmp2, vdequant), 13), v_deq_abs);
    vdest = vsubq_s16(veorq_s16(v_deq_abs, v_coeff_sign), v_coeff_sign);
    coeff_nz_mask =
        vbslq_s16(vcond, vdest, load_tran_low_to_s16q(&dqcoeff_ptr[0]));
    store_s16q_to_tran_low(&dqcoeff_ptr[0], coeff_nz_mask);

    vround =
        vsetq_lane_s16(ROUND_POWER_OF_TWO(round_ptr[1], log_scale), vround, 0);
    vquant = vsetq_lane_s16(quant_ptr[1], vquant, 0);
    vdequant = vsetq_lane_s16(dequant_ptr[1], vdequant, 0);
    vquant_shift = vsetq_lane_s16(quant_shift_ptr[1], vquant_shift, 0);

    uint16x8_t vtmp_mask = vcgtq_s16(vtmp2, zero);
    const uint16x8_t v_nz_mask = vandq_u16(vtmp_mask, vcond);
    int16x8_t v_iscan = vld1q_s16(&iscan[0]);
    vcond = vandq_u16(v_nz_mask, vcgtq_s16(v_iscan, v_eobmax_76543210));
    v_eobmax_76543210 = vbslq_s16(vcond, v_iscan, v_eobmax_76543210);
  }
  vzbins = vsetq_lane_s16(zbins[1], vzbins, 0);

  for (int i = 8; i < n_coeffs; i += 8) {
    v_coeff = load_tran_low_to_s16q(&coeff_ptr[i]);
    v_coeff_sign = vshrq_n_s16(v_coeff, 15);
    v_abs = vabsq_s16(v_coeff);

    if (qm_ptr == NULL) {
      vcond = vcgeq_s16(v_abs, vzbins);
    } else {
      vwt = vmovl_u8(vld1_u8(&qm_ptr[i]));
      vcond = vcgeq_s16(QM_MULL_SHIFT(v_abs, vwt), vzbins);
    }
    nz_check = vget_lane_u64(vreinterpret_u64_u8(vmovn_u16(vcond)), 0);
    if (nz_check) {
      int16x8_t vtmp = vqaddq_s16(v_abs, vround);

      int16x8_t vtmp2;
      if (qm_ptr == NULL) {
        vtmp2 = vsraq_n_s16(vtmp, vqdmulhq_s16(vtmp, vquant), 1);
      } else {
        vtmp2 = QM_MULL_SHIFT(vtmp, vwt);
        vtmp2 = vaddq_s16(vtmp2, vtmp);
      }

      int16x8_t ones =
          vandq_s16(vshrq_n_s16(vmulq_s16(vtmp2, vquant_shift), 14), v_ones);
      vtmp2 =
          vaddq_s16(vshlq_s16(vqdmulhq_s16(vtmp2, vquant_shift), v_ones), ones);
      int16x8_t vdest = vsubq_s16(veorq_s16(vtmp2, v_coeff_sign), v_coeff_sign);
      int16x8_t coeff_nz_mask =
          vbslq_s16(vcond, vdest, load_tran_low_to_s16q(&qcoeff_ptr[i]));
      store_s16q_to_tran_low(&qcoeff_ptr[i], coeff_nz_mask);

      if (iqm_ptr != NULL) {
        viwt = vmovl_u8(vld1_u8(&iqm_ptr[i]));
        vdequant = QM_MULL_SHIFT(vdequant, viwt);
      }
      int16x8_t v_deq_abs = vreinterpretq_s16_u16(vshlq_u16(
          vreinterpretq_u16_s16(vmulq_s16(vtmp2, vdequant)), v_log_scale));
      v_deq_abs =
          vorrq_s16(vshlq_n_s16(vqdmulhq_s16(vtmp2, vdequant), 13), v_deq_abs);
      vdest = vsubq_s16(veorq_s16(v_deq_abs, v_coeff_sign), v_coeff_sign);
      coeff_nz_mask =
          vbslq_s16(vcond, vdest, load_tran_low_to_s16q(&dqcoeff_ptr[i]));
      store_s16q_to_tran_low(&dqcoeff_ptr[i], coeff_nz_mask);

      uint16x8_t vtmp_mask = vcgtq_s16(vtmp2, zero);
      const uint16x8_t v_nz_mask = vandq_u16(vtmp_mask, vcond);
      int16x8_t v_iscan = vld1q_s16(&iscan[i]);
      vcond = vandq_u16(v_nz_mask, vcgtq_s16(v_iscan, v_eobmax_76543210));
      v_eobmax_76543210 = vbslq_s16(vcond, v_iscan, v_eobmax_76543210);
    }
  }
  *eob_ptr = get_max_eob(v_eobmax_76543210);
}

void aom_quantize_b_helper_neon(
    const tran_low_t *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr,
    const int16_t *round_ptr, const int16_t *quant_ptr,
    const int16_t *quant_shift_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr,
    const int16_t *scan, const int16_t *iscan, const qm_val_t *qm_ptr,
    const qm_val_t *iqm_ptr, const int log_scale) {
  switch (log_scale) {  // log_scale for AV1 encoder can be only 0, 1, 2
    case 0:
      aom_quantize_b_helper_16x16_neon(coeff_ptr, n_coeffs, zbin_ptr, round_ptr,
                                       quant_ptr, quant_shift_ptr, qcoeff_ptr,
                                       dqcoeff_ptr, dequant_ptr, eob_ptr, scan,
                                       iscan, qm_ptr, iqm_ptr);
      break;
    case 1:
      aom_quantize_b_helper_32x32_neon(coeff_ptr, n_coeffs, zbin_ptr, round_ptr,
                                       quant_ptr, quant_shift_ptr, qcoeff_ptr,
                                       dqcoeff_ptr, dequant_ptr, eob_ptr, scan,
                                       iscan, qm_ptr, iqm_ptr);
      break;
    case 2:
      aom_quantize_b_helper_64x64_neon(coeff_ptr, n_coeffs, zbin_ptr, round_ptr,
                                       quant_ptr, quant_shift_ptr, qcoeff_ptr,
                                       dqcoeff_ptr, dequant_ptr, eob_ptr, scan,
                                       iscan, qm_ptr, iqm_ptr);
      break;
  }
}

static inline uint16x8_t quantize_b_logscale1_8(
    int16x8_t coeff, int16x8_t abs, uint16x8_t cond, int16x8_t round,
    int16x8_t dequant, int16x8_t quant, int16x8_t quant_shift,
    tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr) {
  const int16x8_t zero = vdupq_n_s16(0);

  int16x8_t coeff_sign = vreinterpretq_s16_u16(vcltq_s16(coeff, zero));

  int16x8_t tmp = vqaddq_s16(abs, round);
  tmp = vsraq_n_s16(tmp, vqdmulhq_s16(tmp, quant), 1);
  tmp = vqdmulhq_s16(tmp, quant_shift);

  int16x8_t qcoeff = vsubq_s16(veorq_s16(tmp, coeff_sign), coeff_sign);
  qcoeff = vbslq_s16(cond, qcoeff, zero);
  store_s16q_to_tran_low(qcoeff_ptr, qcoeff);

  // Shift by log_scale = 1.
  int16x8_t dqcoeff = vreinterpretq_s16_u16(vhaddq_u16(
      vreinterpretq_u16_s16(vmulq_s16(tmp, dequant)), vdupq_n_u16(0)));
  dqcoeff = vsubq_s16(veorq_s16(dqcoeff, coeff_sign), coeff_sign);
  dqcoeff = vbslq_s16(cond, dqcoeff, zero);
  store_s16q_to_tran_low(dqcoeff_ptr, dqcoeff);

  uint16x8_t tmp_mask = vcgtq_s16(tmp, zero);
  const uint16x8_t nz_mask = vandq_u16(tmp_mask, cond);

  return nz_mask;
}

void aom_quantize_b_32x32_neon(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                               const int16_t *zbin_ptr,
                               const int16_t *round_ptr,
                               const int16_t *quant_ptr,
                               const int16_t *quant_shift_ptr,
                               tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                               const int16_t *dequant_ptr, uint16_t *eob_ptr,
                               const int16_t *scan, const int16_t *iscan) {
  (void)scan;

  const int log_scale = 1;
  const int zbins[2] = { ROUND_POWER_OF_TWO(zbin_ptr[0], log_scale),
                         ROUND_POWER_OF_TWO(zbin_ptr[1], log_scale) };
  const int rounds[2] = { ROUND_POWER_OF_TWO(round_ptr[0], log_scale),
                          ROUND_POWER_OF_TWO(round_ptr[1], log_scale) };

  int16x8_t v_eobmax_76543210 = vdupq_n_s16(-1);

  const int16x8_t v_zbins = vdupq_n_s16(zbins[1]);
  int16x8_t v_round = vdupq_n_s16(rounds[1]);
  int16x8_t v_dequant = vdupq_n_s16(dequant_ptr[1]);
  int16x8_t v_quant = vdupq_n_s16(quant_ptr[1]);
  int16x8_t v_quant_shift = vdupq_n_s16(quant_shift_ptr[1]);

  assert(n_coeffs > 16 && (n_coeffs & 15) == 0);
  intptr_t non_zero_count = n_coeffs;
  do {
    const int16x8_t v_coeff_a =
        load_tran_low_to_s16q(coeff_ptr + non_zero_count - 8);
    const int16x8_t v_coeff_b =
        load_tran_low_to_s16q(coeff_ptr + non_zero_count - 16);
    const uint16x8_t v_mask_a = vcgeq_s16(vabsq_s16(v_coeff_a), v_zbins);
    const uint16x8_t v_mask_b = vcgeq_s16(vabsq_s16(v_coeff_b), v_zbins);
    if (mask_nonzero_u16(vorrq_u16(v_mask_a, v_mask_b))) {
      break;
    }
    non_zero_count -= 16;
    store_zero_8_tran_low(qcoeff_ptr + non_zero_count);
    store_zero_8_tran_low(qcoeff_ptr + non_zero_count + 8);
    store_zero_8_tran_low(dqcoeff_ptr + non_zero_count);
    store_zero_8_tran_low(dqcoeff_ptr + non_zero_count + 8);
  } while (non_zero_count != 16);

  int16x8_t v_zbins0 = vsetq_lane_s16(zbins[0], v_zbins, 0);
  int16x8_t v_coeff = load_tran_low_to_s16q(coeff_ptr);
  int16x8_t v_abs = vabsq_s16(v_coeff);
  uint16x8_t v_cond = vcgeq_s16(v_abs, v_zbins0);

  if (mask_nonzero_u16(v_cond)) {
    v_round = vsetq_lane_s16(rounds[0], v_round, 0);
    v_quant = vsetq_lane_s16(quant_ptr[0], v_quant, 0);
    v_dequant = vsetq_lane_s16(dequant_ptr[0], v_dequant, 0);
    v_quant_shift = vsetq_lane_s16(quant_shift_ptr[0], v_quant_shift, 0);

    const uint16x8_t v_nz_mask =
        quantize_b_logscale1_8(v_coeff, v_abs, v_cond, v_round, v_dequant,
                               v_quant, v_quant_shift, qcoeff_ptr, dqcoeff_ptr);

    v_round = vdupq_lane_s16(vget_low_s16(v_round), 1);
    v_quant = vdupq_lane_s16(vget_low_s16(v_quant), 1);
    v_dequant = vdupq_lane_s16(vget_low_s16(v_dequant), 1);
    v_quant_shift = vdupq_lane_s16(vget_low_s16(v_quant_shift), 1);

    int16x8_t v_iscan = vld1q_s16(iscan);
    int16x8_t v_eobmax = vmaxq_s16(v_iscan, v_eobmax_76543210);
    v_eobmax_76543210 = vbslq_s16(v_nz_mask, v_eobmax, v_eobmax_76543210);
  } else {
    store_zero_8_tran_low(qcoeff_ptr);
    store_zero_8_tran_low(dqcoeff_ptr);
  }

  intptr_t i = 8;
  do {
    v_coeff = load_tran_low_to_s16q(coeff_ptr + i);
    v_abs = vabsq_s16(v_coeff);
    v_cond = vcgeq_s16(v_abs, v_zbins);

    if (mask_nonzero_u16(v_cond)) {
      const uint16x8_t v_nz_mask = quantize_b_logscale1_8(
          v_coeff, v_abs, v_cond, v_round, v_dequant, v_quant, v_quant_shift,
          qcoeff_ptr + i, dqcoeff_ptr + i);

      int16x8_t v_iscan = vld1q_s16(iscan + i);
      int16x8_t v_eobmax = vmaxq_s16(v_iscan, v_eobmax_76543210);
      v_eobmax_76543210 = vbslq_s16(v_nz_mask, v_eobmax, v_eobmax_76543210);
    } else {
      store_zero_8_tran_low(qcoeff_ptr + i);
      store_zero_8_tran_low(dqcoeff_ptr + i);
    }
    i += 8;
  } while (i != non_zero_count);
  *eob_ptr = get_max_eob(v_eobmax_76543210);
}

static inline uint16x8_t quantize_b_logscale2_8(
    int16x8_t coeff, int16x8_t abs, uint16x8_t cond, int16x8_t round,
    int16x8_t dequant, int16x8_t quant, int16x8_t quant_shift,
    tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr) {
  const int16x8_t zero = vdupq_n_s16(0);
  const int16x8_t one = vdupq_n_s16(1);

  int16x8_t coeff_sign = vreinterpretq_s16_u16(vcltq_s16(coeff, zero));

  int16x8_t tmp = vqaddq_s16(abs, round);
  tmp = vsraq_n_s16(tmp, vqdmulhq_s16(tmp, quant), 1);
  int16x8_t ones = vandq_s16(vshrq_n_s16(vmulq_s16(tmp, quant_shift), 14), one);
  tmp = vqdmulhq_s16(tmp, quant_shift);
  tmp = vaddq_s16(vshlq_s16(tmp, one), ones);

  int16x8_t qcoeff = vsubq_s16(veorq_s16(tmp, coeff_sign), coeff_sign);
  qcoeff = vbslq_s16(cond, qcoeff, zero);
  store_s16q_to_tran_low(qcoeff_ptr, qcoeff);

  // Shift right by log_scale = 2.
  int16x8_t dqcoeff = vreinterpretq_s16_u16(
      vshrq_n_u16(vreinterpretq_u16_s16(vmulq_s16(tmp, dequant)), 2));
  dqcoeff = vorrq_s16(vshlq_n_s16(vqdmulhq_s16(tmp, dequant), 13), dqcoeff);
  dqcoeff = vsubq_s16(veorq_s16(dqcoeff, coeff_sign), coeff_sign);
  dqcoeff = vbslq_s16(cond, dqcoeff, zero);
  store_s16q_to_tran_low(dqcoeff_ptr, dqcoeff);

  uint16x8_t tmp_mask = vcgtq_s16(tmp, zero);
  const uint16x8_t nz_mask = vandq_u16(tmp_mask, cond);

  return nz_mask;
}

void aom_quantize_b_64x64_neon(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                               const int16_t *zbin_ptr,
                               const int16_t *round_ptr,
                               const int16_t *quant_ptr,
                               const int16_t *quant_shift_ptr,
                               tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                               const int16_t *dequant_ptr, uint16_t *eob_ptr,
                               const int16_t *scan, const int16_t *iscan) {
  (void)scan;

  const int log_scale = 2;
  const int zbins[2] = { ROUND_POWER_OF_TWO(zbin_ptr[0], log_scale),
                         ROUND_POWER_OF_TWO(zbin_ptr[1], log_scale) };
  const int rounds[2] = { ROUND_POWER_OF_TWO(round_ptr[0], log_scale),
                          ROUND_POWER_OF_TWO(round_ptr[1], log_scale) };

  int16x8_t v_eobmax_76543210 = vdupq_n_s16(-1);

  const int16x8_t v_zbins = vdupq_n_s16(zbins[1]);
  int16x8_t v_round = vdupq_n_s16(rounds[1]);
  int16x8_t v_dequant = vdupq_n_s16(dequant_ptr[1]);
  int16x8_t v_quant = vdupq_n_s16(quant_ptr[1]);
  int16x8_t v_quant_shift = vdupq_n_s16(quant_shift_ptr[1]);

  assert(n_coeffs > 16 && (n_coeffs & 15) == 0);
  intptr_t non_zero_count = n_coeffs;
  do {
    const int16x8_t v_coeff_a =
        load_tran_low_to_s16q(coeff_ptr + non_zero_count - 8);
    const int16x8_t v_coeff_b =
        load_tran_low_to_s16q(coeff_ptr + non_zero_count - 16);
    const uint16x8_t v_mask_a = vcgeq_s16(vabsq_s16(v_coeff_a), v_zbins);
    const uint16x8_t v_mask_b = vcgeq_s16(vabsq_s16(v_coeff_b), v_zbins);
    if (mask_nonzero_u16(vorrq_u16(v_mask_a, v_mask_b))) {
      break;
    }
    non_zero_count -= 16;
    store_zero_8_tran_low(qcoeff_ptr + non_zero_count);
    store_zero_8_tran_low(qcoeff_ptr + non_zero_count + 8);
    store_zero_8_tran_low(dqcoeff_ptr + non_zero_count);
    store_zero_8_tran_low(dqcoeff_ptr + non_zero_count + 8);
  } while (non_zero_count != 16);

  int16x8_t v_zbins0 = vsetq_lane_s16(zbins[0], v_zbins, 0);
  int16x8_t v_coeff = load_tran_low_to_s16q(coeff_ptr);
  int16x8_t v_abs = vabsq_s16(v_coeff);
  uint16x8_t v_cond = vcgeq_s16(v_abs, v_zbins0);

  if (mask_nonzero_u16(v_cond)) {
    v_round = vsetq_lane_s16(rounds[0], v_round, 0);
    v_quant = vsetq_lane_s16(quant_ptr[0], v_quant, 0);
    v_dequant = vsetq_lane_s16(dequant_ptr[0], v_dequant, 0);
    v_quant_shift = vsetq_lane_s16(quant_shift_ptr[0], v_quant_shift, 0);

    const uint16x8_t v_nz_mask =
        quantize_b_logscale2_8(v_coeff, v_abs, v_cond, v_round, v_dequant,
                               v_quant, v_quant_shift, qcoeff_ptr, dqcoeff_ptr);

    v_round = vdupq_lane_s16(vget_low_s16(v_round), 1);
    v_quant = vdupq_lane_s16(vget_low_s16(v_quant), 1);
    v_dequant = vdupq_lane_s16(vget_low_s16(v_dequant), 1);
    v_quant_shift = vdupq_lane_s16(vget_low_s16(v_quant_shift), 1);

    int16x8_t v_iscan = vld1q_s16(iscan);
    int16x8_t v_eobmax = vmaxq_s16(v_iscan, v_eobmax_76543210);
    v_eobmax_76543210 = vbslq_s16(v_nz_mask, v_eobmax, v_eobmax_76543210);
  } else {
    store_zero_8_tran_low(qcoeff_ptr);
    store_zero_8_tran_low(dqcoeff_ptr);
  }

  intptr_t i = 8;
  do {
    v_coeff = load_tran_low_to_s16q(coeff_ptr + i);
    v_abs = vabsq_s16(v_coeff);
    v_cond = vcgeq_s16(v_abs, v_zbins);

    if (mask_nonzero_u16(v_cond)) {
      const uint16x8_t v_nz_mask = quantize_b_logscale2_8(
          v_coeff, v_abs, v_cond, v_round, v_dequant, v_quant, v_quant_shift,
          qcoeff_ptr + i, dqcoeff_ptr + i);

      int16x8_t v_iscan = vld1q_s16(iscan + i);
      int16x8_t v_eobmax = vmaxq_s16(v_iscan, v_eobmax_76543210);
      v_eobmax_76543210 = vbslq_s16(v_nz_mask, v_eobmax, v_eobmax_76543210);
    } else {
      store_zero_8_tran_low(qcoeff_ptr + i);
      store_zero_8_tran_low(dqcoeff_ptr + i);
    }
    i += 8;
  } while (i != non_zero_count);
  *eob_ptr = get_max_eob(v_eobmax_76543210);
}

static AOM_FORCE_INLINE int mask_any_nonzero_u32(uint32x4_t m0, uint32x4_t m1) {
  const uint32x4_t any = vorrq_u32(m0, m1);
#if AOM_ARCH_AARCH64
  return vmaxvq_u32(any) != 0;
#else
  const uint64x2_t u64 = vreinterpretq_u64_u32(any);
  return (vgetq_lane_u64(u64, 0) | vgetq_lane_u64(u64, 1)) != 0;
#endif
}

static AOM_FORCE_INLINE int16x8_t update_qm_eob(const int16_t *iscan_ptr,
                                                int16x8_t v_eobmax,
                                                uint32x4_t nz_mask0,
                                                uint32x4_t nz_mask1) {
#if AOM_ARCH_AARCH64
  const uint16x8_t v_nz_mask = vuzp1q_u16(vreinterpretq_u16_u32(nz_mask0),
                                          vreinterpretq_u16_u32(nz_mask1));
#else
  const uint16x8_t v_nz_mask =
      vcombine_u16(vmovn_u32(nz_mask0), vmovn_u32(nz_mask1));
#endif
  return get_max_lane_eob(iscan_ptr, v_eobmax, v_nz_mask);
}

static AOM_FORCE_INLINE int16x8_t quantize_fp_qm_8_neon(
    const tran_low_t *coeff_ptr, const int16_t *iscan_ptr,
    const qm_val_t *qm_ptr, const qm_val_t *iqm_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, int32x4_t v_thr0, int32x4_t v_thr1,
    int32x4_t v_round0, int32x4_t v_round1, int32x4_t v_quant0,
    int32x4_t v_quant1, int32x4_t v_dequant0, int32x4_t v_dequant1,
    int32x4_t v_log_scale, int32x4_t v_neg_log_scale, int16x8_t v_eobmax) {
  const int32x4_t v_coeff0 = vld1q_s32(coeff_ptr);
  const int32x4_t v_coeff1 = vld1q_s32(coeff_ptr + 4);
  const int32x4_t v_abs0 = vabsq_s32(v_coeff0);
  const int32x4_t v_abs1 = vabsq_s32(v_coeff1);
  const uint16x8_t v_wt_u16 = vmovl_u8(vld1_u8(qm_ptr));
  const int32x4_t v_wt0 =
      vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_wt_u16)));
  const int32x4_t v_wt1 =
      vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_wt_u16)));
  const uint32x4_t v_mask0 = vcgeq_s32(vmulq_s32(v_abs0, v_wt0), v_thr0);
  const uint32x4_t v_mask1 = vcgeq_s32(vmulq_s32(v_abs1, v_wt1), v_thr1);

  if (mask_any_nonzero_u32(v_mask0, v_mask1)) {
    const int32x4_t v_zero = vdupq_n_s32(0);
    const int32x4_t v_max_s16 = vdupq_n_s32(INT16_MAX);
    const int32x4_t v_sign0 = vshrq_n_s32(v_coeff0, 31);
    const int32x4_t v_sign1 = vshrq_n_s32(v_coeff1, 31);
    const int32x4_t v_tmp0 =
        vandq_s32(vminq_s32(vaddq_s32(v_abs0, v_round0), v_max_s16),
                  vreinterpretq_s32_u32(v_mask0));
    const int32x4_t v_tmp1 =
        vandq_s32(vminq_s32(vaddq_s32(v_abs1, v_round1), v_max_s16),
                  vreinterpretq_s32_u32(v_mask1));
    const int32x4_t v_tmp_wt0 = vmulq_s32(v_tmp0, v_wt0);
    const int32x4_t v_tmp_wt1 = vmulq_s32(v_tmp1, v_wt1);
    const int32x4_t v_abs_q0 = vshrq_n_s32(
        vqdmulhq_s32(vshlq_s32(v_tmp_wt0, v_log_scale), v_quant0), AOM_QM_BITS);
    const int32x4_t v_abs_q1 = vshrq_n_s32(
        vqdmulhq_s32(vshlq_s32(v_tmp_wt1, v_log_scale), v_quant1), AOM_QM_BITS);

    const uint16x8_t v_iwt_u16 = vmovl_u8(vld1_u8(iqm_ptr));
    const int32x4_t v_iwt0 =
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_iwt_u16)));
    const int32x4_t v_iwt1 =
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_iwt_u16)));
    const int32x4_t v_dq_rnd = vdupq_n_s32(1 << (AOM_QM_BITS - 1));
    const int32x4_t v_dq0 =
        vshrq_n_s32(vmlaq_s32(v_dq_rnd, v_dequant0, v_iwt0), AOM_QM_BITS);
    const int32x4_t v_dq1 =
        vshrq_n_s32(vmlaq_s32(v_dq_rnd, v_dequant1, v_iwt1), AOM_QM_BITS);
    const int32x4_t v_abs_dq0 =
        vshlq_s32(vmulq_s32(v_abs_q0, v_dq0), v_neg_log_scale);
    const int32x4_t v_abs_dq1 =
        vshlq_s32(vmulq_s32(v_abs_q1, v_dq1), v_neg_log_scale);

    const int32x4_t v_q0 = vsubq_s32(veorq_s32(v_abs_q0, v_sign0), v_sign0);
    const int32x4_t v_q1 = vsubq_s32(veorq_s32(v_abs_q1, v_sign1), v_sign1);
    const int32x4_t v_dq_out0 =
        vsubq_s32(veorq_s32(v_abs_dq0, v_sign0), v_sign0);
    const int32x4_t v_dq_out1 =
        vsubq_s32(veorq_s32(v_abs_dq1, v_sign1), v_sign1);
    vst1q_s32(qcoeff_ptr, v_q0);
    vst1q_s32(qcoeff_ptr + 4, v_q1);
    vst1q_s32(dqcoeff_ptr, v_dq_out0);
    vst1q_s32(dqcoeff_ptr + 4, v_dq_out1);

    const uint32x4_t v_nz0 = vcgtq_s32(v_abs_q0, v_zero);
    const uint32x4_t v_nz1 = vcgtq_s32(v_abs_q1, v_zero);
    return update_qm_eob(iscan_ptr, v_eobmax, v_nz0, v_nz1);
  }
  const int32x4_t v_zero = vdupq_n_s32(0);
  vst1q_s32(qcoeff_ptr, v_zero);
  vst1q_s32(qcoeff_ptr + 4, v_zero);
  vst1q_s32(dqcoeff_ptr, v_zero);
  vst1q_s32(dqcoeff_ptr + 4, v_zero);
  return v_eobmax;
}

void av1_quantize_fp_qm_neon(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                             const int16_t *round_ptr, const int16_t *quant_ptr,
                             tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                             const int16_t *dequant_ptr, uint16_t *eob_ptr,
                             const int16_t *scan, const int16_t *iscan,
                             const qm_val_t *qm_ptr, const qm_val_t *iqm_ptr,
                             int log_scale) {
  (void)scan;
  assert(qm_ptr != NULL && iqm_ptr != NULL);
  assert(n_coeffs >= 16 && (n_coeffs & 7) == 0);
  const int thr_shift = AOM_QM_BITS - (1 + log_scale);
  const int32x4_t v_thr_ac = vdupq_n_s32((int32_t)dequant_ptr[1] << thr_shift);
  const int32x4_t v_thr0 =
      vsetq_lane_s32((int32_t)dequant_ptr[0] << thr_shift, v_thr_ac, 0);
  const int32x4_t v_round_ac =
      vdupq_n_s32(ROUND_POWER_OF_TWO(round_ptr[1], log_scale));
  const int32x4_t v_round0 = vsetq_lane_s32(
      ROUND_POWER_OF_TWO(round_ptr[0], log_scale), v_round_ac, 0);
  const int32x4_t v_quant_ac =
      vdupq_n_s32((int32_t)((uint32_t)quant_ptr[1] << 15));
  const int32x4_t v_quant0 =
      vsetq_lane_s32((int32_t)((uint32_t)quant_ptr[0] << 15), v_quant_ac, 0);
  const int32x4_t v_dequant_ac = vdupq_n_s32(dequant_ptr[1]);
  const int32x4_t v_dequant0 = vsetq_lane_s32(dequant_ptr[0], v_dequant_ac, 0);
  const int32x4_t v_log_scale = vdupq_n_s32(log_scale);
  const int32x4_t v_neg_log_scale = vdupq_n_s32(-log_scale);
  int16x8_t v_eobmax = vdupq_n_s16(-1);

  v_eobmax = quantize_fp_qm_8_neon(
      coeff_ptr, iscan, qm_ptr, iqm_ptr, qcoeff_ptr, dqcoeff_ptr, v_thr0,
      v_thr_ac, v_round0, v_round_ac, v_quant0, v_quant_ac, v_dequant0,
      v_dequant_ac, v_log_scale, v_neg_log_scale, v_eobmax);

  intptr_t i = 8;
  do {
    v_eobmax = quantize_fp_qm_8_neon(
        coeff_ptr + i, iscan + i, qm_ptr + i, iqm_ptr + i, qcoeff_ptr + i,
        dqcoeff_ptr + i, v_thr_ac, v_thr_ac, v_round_ac, v_round_ac, v_quant_ac,
        v_quant_ac, v_dequant_ac, v_dequant_ac, v_log_scale, v_neg_log_scale,
        v_eobmax);
    i += 8;
  } while (i != n_coeffs);
  *eob_ptr = get_max_eob(v_eobmax);
}

static AOM_FORCE_INLINE int16x8_t quantize_b_qm_8_neon(
    const tran_low_t *coeff_ptr, const int16_t *iscan_ptr,
    const qm_val_t *qm_ptr, const qm_val_t *iqm_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, int32x4_t v_zbin0, int32x4_t v_zbin1,
    int32x4_t v_round0, int32x4_t v_round1, int32x4_t v_quant0,
    int32x4_t v_quant1, int32x4_t v_qshift0, int32x4_t v_qshift1,
    int32x4_t v_dequant0, int32x4_t v_dequant1, int32x4_t v_log_scale,
    int32x4_t v_neg_log_scale, int16x8_t v_eobmax) {
  const int32x4_t v_coeff0 = vld1q_s32(coeff_ptr);
  const int32x4_t v_coeff1 = vld1q_s32(coeff_ptr + 4);
  const int32x4_t v_abs0 = vabsq_s32(v_coeff0);
  const int32x4_t v_abs1 = vabsq_s32(v_coeff1);
  const uint16x8_t v_wt_u16 = vmovl_u8(vld1_u8(qm_ptr));
  const int32x4_t v_wt0 =
      vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_wt_u16)));
  const int32x4_t v_wt1 =
      vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_wt_u16)));
  const uint32x4_t v_mask0 = vcgeq_s32(vmulq_s32(v_abs0, v_wt0), v_zbin0);
  const uint32x4_t v_mask1 = vcgeq_s32(vmulq_s32(v_abs1, v_wt1), v_zbin1);

  if (mask_any_nonzero_u32(v_mask0, v_mask1)) {
    const int32x4_t v_zero = vdupq_n_s32(0);
    const int32x4_t v_max_s16 = vdupq_n_s32(INT16_MAX);
    const int32x4_t v_sign0 = vshrq_n_s32(v_coeff0, 31);
    const int32x4_t v_sign1 = vshrq_n_s32(v_coeff1, 31);
    const int32x4_t v_tmp0 =
        vandq_s32(vminq_s32(vaddq_s32(v_abs0, v_round0), v_max_s16),
                  vreinterpretq_s32_u32(v_mask0));
    const int32x4_t v_tmp1 =
        vandq_s32(vminq_s32(vaddq_s32(v_abs1, v_round1), v_max_s16),
                  vreinterpretq_s32_u32(v_mask1));
    const int32x4_t v_tmp_wt0 = vmulq_s32(v_tmp0, v_wt0);
    const int32x4_t v_tmp_wt1 = vmulq_s32(v_tmp1, v_wt1);
    const int32x4_t v_tmp2_0 =
        vaddq_s32(vqdmulhq_s32(v_tmp_wt0, v_quant0), v_tmp_wt0);
    const int32x4_t v_tmp2_1 =
        vaddq_s32(vqdmulhq_s32(v_tmp_wt1, v_quant1), v_tmp_wt1);
    const int32x4_t v_abs_q0 = vshrq_n_s32(
        vqdmulhq_s32(vshlq_s32(v_tmp2_0, v_log_scale), v_qshift0), AOM_QM_BITS);
    const int32x4_t v_abs_q1 = vshrq_n_s32(
        vqdmulhq_s32(vshlq_s32(v_tmp2_1, v_log_scale), v_qshift1), AOM_QM_BITS);

    const uint16x8_t v_iwt_u16 = vmovl_u8(vld1_u8(iqm_ptr));
    const int32x4_t v_iwt0 =
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_iwt_u16)));
    const int32x4_t v_iwt1 =
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_iwt_u16)));
    const int32x4_t v_dq_rnd = vdupq_n_s32(1 << (AOM_QM_BITS - 1));
    const int32x4_t v_dq0 =
        vshrq_n_s32(vmlaq_s32(v_dq_rnd, v_dequant0, v_iwt0), AOM_QM_BITS);
    const int32x4_t v_dq1 =
        vshrq_n_s32(vmlaq_s32(v_dq_rnd, v_dequant1, v_iwt1), AOM_QM_BITS);
    const int32x4_t v_abs_dq0 =
        vshlq_s32(vmulq_s32(v_abs_q0, v_dq0), v_neg_log_scale);
    const int32x4_t v_abs_dq1 =
        vshlq_s32(vmulq_s32(v_abs_q1, v_dq1), v_neg_log_scale);

    const int32x4_t v_q0 = vsubq_s32(veorq_s32(v_abs_q0, v_sign0), v_sign0);
    const int32x4_t v_q1 = vsubq_s32(veorq_s32(v_abs_q1, v_sign1), v_sign1);
    const int32x4_t v_dq_out0 =
        vsubq_s32(veorq_s32(v_abs_dq0, v_sign0), v_sign0);
    const int32x4_t v_dq_out1 =
        vsubq_s32(veorq_s32(v_abs_dq1, v_sign1), v_sign1);
    vst1q_s32(qcoeff_ptr, v_q0);
    vst1q_s32(qcoeff_ptr + 4, v_q1);
    vst1q_s32(dqcoeff_ptr, v_dq_out0);
    vst1q_s32(dqcoeff_ptr + 4, v_dq_out1);

    const uint32x4_t v_nz0 = vcgtq_s32(v_abs_q0, v_zero);
    const uint32x4_t v_nz1 = vcgtq_s32(v_abs_q1, v_zero);
    return update_qm_eob(iscan_ptr, v_eobmax, v_nz0, v_nz1);
  }
  const int32x4_t v_zero = vdupq_n_s32(0);
  vst1q_s32(qcoeff_ptr, v_zero);
  vst1q_s32(qcoeff_ptr + 4, v_zero);
  vst1q_s32(dqcoeff_ptr, v_zero);
  vst1q_s32(dqcoeff_ptr + 4, v_zero);
  return v_eobmax;
}

void av1_quantize_b_qm_neon(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                            const int16_t *zbin_ptr, const int16_t *round_ptr,
                            const int16_t *quant_ptr,
                            const int16_t *quant_shift_ptr,
                            tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                            const int16_t *dequant_ptr, uint16_t *eob_ptr,
                            const int16_t *scan, const int16_t *iscan,
                            const qm_val_t *qm_ptr, const qm_val_t *iqm_ptr,
                            int log_scale) {
  (void)scan;
  assert(qm_ptr != NULL && iqm_ptr != NULL);
  assert(n_coeffs >= 16 && (n_coeffs & 7) == 0);
  const int32x4_t v_zbin_ac =
      vdupq_n_s32(ROUND_POWER_OF_TWO(zbin_ptr[1], log_scale) << AOM_QM_BITS);
  const int32x4_t v_zbin0 = vsetq_lane_s32(
      ROUND_POWER_OF_TWO(zbin_ptr[0], log_scale) << AOM_QM_BITS, v_zbin_ac, 0);
  const int32x4_t v_round_ac =
      vdupq_n_s32(ROUND_POWER_OF_TWO(round_ptr[1], log_scale));
  const int32x4_t v_round0 = vsetq_lane_s32(
      ROUND_POWER_OF_TWO(round_ptr[0], log_scale), v_round_ac, 0);
  const int32x4_t v_quant_ac =
      vdupq_n_s32((int32_t)((uint32_t)quant_ptr[1] << 15));
  const int32x4_t v_quant0 =
      vsetq_lane_s32((int32_t)((uint32_t)quant_ptr[0] << 15), v_quant_ac, 0);
  const int32x4_t v_qshift_ac =
      vdupq_n_s32((int32_t)((uint32_t)quant_shift_ptr[1] << 15));
  const int32x4_t v_qshift0 = vsetq_lane_s32(
      (int32_t)((uint32_t)quant_shift_ptr[0] << 15), v_qshift_ac, 0);
  const int32x4_t v_dequant_ac = vdupq_n_s32(dequant_ptr[1]);
  const int32x4_t v_dequant0 = vsetq_lane_s32(dequant_ptr[0], v_dequant_ac, 0);
  const int32x4_t v_log_scale = vdupq_n_s32(log_scale);
  const int32x4_t v_neg_log_scale = vdupq_n_s32(-log_scale);
  int16x8_t v_eobmax = vdupq_n_s16(-1);

  v_eobmax = quantize_b_qm_8_neon(coeff_ptr, iscan, qm_ptr, iqm_ptr, qcoeff_ptr,
                                  dqcoeff_ptr, v_zbin0, v_zbin_ac, v_round0,
                                  v_round_ac, v_quant0, v_quant_ac, v_qshift0,
                                  v_qshift_ac, v_dequant0, v_dequant_ac,
                                  v_log_scale, v_neg_log_scale, v_eobmax);

  intptr_t i = 8;
  do {
    v_eobmax = quantize_b_qm_8_neon(
        coeff_ptr + i, iscan + i, qm_ptr + i, iqm_ptr + i, qcoeff_ptr + i,
        dqcoeff_ptr + i, v_zbin_ac, v_zbin_ac, v_round_ac, v_round_ac,
        v_quant_ac, v_quant_ac, v_qshift_ac, v_qshift_ac, v_dequant_ac,
        v_dequant_ac, v_log_scale, v_neg_log_scale, v_eobmax);
    i += 8;
  } while (i != n_coeffs);
  *eob_ptr = get_max_eob(v_eobmax);
}
