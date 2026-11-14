/*
 * Copyright (c) 2022, Alliance for Open Media. All rights reserved.
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

#include "config/aom_config.h"
#include "config/av1_rtcd.h"

#include "aom_dsp/arm/mem_neon.h"

#include "av1/common/quant_common.h"
#include "av1/encoder/av1_quantize.h"

static AOM_FORCE_INLINE uint16x4_t quantize_4_from_abs_ls0(
    int32x4_t v_coeff, int32x4_t v_abs_coeff, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, int32x4_t v_quant_s32, int32x4_t v_dequant_s32,
    int32x4_t v_round_s32) {
  const int32x4_t v_zero = vdupq_n_s32(0);
  const int32x4_t v_coeff_sign =
      vreinterpretq_s32_u32(vcltq_s32(v_coeff, v_zero));
  const int32x4_t v_abs_coeff_scaled = vshlq_n_s32(v_abs_coeff, 1);
  const uint32x4_t v_mask = vcgeq_s32(v_abs_coeff_scaled, v_dequant_s32);
  const int32x4_t v_tmp = vandq_s32(vaddq_s32(v_abs_coeff, v_round_s32),
                                    vreinterpretq_s32_u32(v_mask));
  const int32x4_t v_abs_qcoeff = vqdmulhq_s32(v_tmp, v_quant_s32);
  const int32x4_t v_qcoeff =
      vsubq_s32(veorq_s32(v_abs_qcoeff, v_coeff_sign), v_coeff_sign);
  const int32x4_t v_abs_dqcoeff = vmulq_s32(v_abs_qcoeff, v_dequant_s32);
  const int32x4_t v_dqcoeff =
      vsubq_s32(veorq_s32(v_abs_dqcoeff, v_coeff_sign), v_coeff_sign);
  vst1q_s32(qcoeff_ptr, v_qcoeff);
  vst1q_s32(dqcoeff_ptr, v_dqcoeff);
  const uint32x4_t nz_qcoeff_mask = vcgtq_s32(v_abs_qcoeff, v_zero);
  return vmovn_u32(nz_qcoeff_mask);
}

static AOM_FORCE_INLINE uint16x4_t quantize_4_from_abs_ls1(
    int32x4_t v_coeff, int32x4_t v_abs_coeff, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, int32x4_t v_quant_s32, int32x4_t v_dequant_s32,
    int32x4_t v_round_s32) {
  const int32x4_t v_zero = vdupq_n_s32(0);
  const int32x4_t v_coeff_sign =
      vreinterpretq_s32_u32(vcltq_s32(v_coeff, v_zero));
  const int32x4_t v_abs_coeff_scaled = vshlq_n_s32(v_abs_coeff, 2);
  const uint32x4_t v_mask = vcgeq_s32(v_abs_coeff_scaled, v_dequant_s32);
  const int32x4_t v_tmp = vandq_s32(vaddq_s32(v_abs_coeff, v_round_s32),
                                    vreinterpretq_s32_u32(v_mask));
  const int32x4_t v_abs_qcoeff =
      vqdmulhq_s32(vshlq_n_s32(v_tmp, 1), v_quant_s32);
  const int32x4_t v_qcoeff =
      vsubq_s32(veorq_s32(v_abs_qcoeff, v_coeff_sign), v_coeff_sign);
  const int32x4_t v_abs_dqcoeff =
      vshrq_n_s32(vmulq_s32(v_abs_qcoeff, v_dequant_s32), 1);
  const int32x4_t v_dqcoeff =
      vsubq_s32(veorq_s32(v_abs_dqcoeff, v_coeff_sign), v_coeff_sign);
  vst1q_s32(qcoeff_ptr, v_qcoeff);
  vst1q_s32(dqcoeff_ptr, v_dqcoeff);
  const uint32x4_t nz_qcoeff_mask = vcgtq_s32(v_abs_qcoeff, v_zero);
  return vmovn_u32(nz_qcoeff_mask);
}

static AOM_FORCE_INLINE uint16x4_t quantize_4_from_abs_ls2(
    int32x4_t v_coeff, int32x4_t v_abs_coeff, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, int32x4_t v_quant_s32, int32x4_t v_dequant_s32,
    int32x4_t v_round_s32) {
  const int32x4_t v_zero = vdupq_n_s32(0);
  const int32x4_t v_coeff_sign =
      vreinterpretq_s32_u32(vcltq_s32(v_coeff, v_zero));
  const int32x4_t v_abs_coeff_scaled = vshlq_n_s32(v_abs_coeff, 3);
  const uint32x4_t v_mask = vcgeq_s32(v_abs_coeff_scaled, v_dequant_s32);
  const int32x4_t v_tmp = vandq_s32(vaddq_s32(v_abs_coeff, v_round_s32),
                                    vreinterpretq_s32_u32(v_mask));
  const int32x4_t v_abs_qcoeff =
      vqdmulhq_s32(vshlq_n_s32(v_tmp, 2), v_quant_s32);
  const int32x4_t v_qcoeff =
      vsubq_s32(veorq_s32(v_abs_qcoeff, v_coeff_sign), v_coeff_sign);
  const int32x4_t v_abs_dqcoeff =
      vshrq_n_s32(vmulq_s32(v_abs_qcoeff, v_dequant_s32), 2);
  const int32x4_t v_dqcoeff =
      vsubq_s32(veorq_s32(v_abs_dqcoeff, v_coeff_sign), v_coeff_sign);
  vst1q_s32(qcoeff_ptr, v_qcoeff);
  vst1q_s32(dqcoeff_ptr, v_dqcoeff);
  const uint32x4_t nz_qcoeff_mask = vcgtq_s32(v_abs_qcoeff, v_zero);
  return vmovn_u32(nz_qcoeff_mask);
}

static AOM_FORCE_INLINE int hbd_zero_skip_u32(int32x4_t v_abs0,
                                              int32x4_t v_abs1,
                                              uint32_t ac_thresh,
                                              tran_low_t *qcoeff_ptr,
                                              tran_low_t *dqcoeff_ptr) {
#if AOM_ARCH_AARCH64
  const int32x4_t v_zero = vdupq_n_s32(0);
  const uint32x4_t v_abs_max =
      vmaxq_u32(vreinterpretq_u32_s32(v_abs0), vreinterpretq_u32_s32(v_abs1));
  if (vmaxvq_u32(v_abs_max) < ac_thresh) {
    vst1q_s32(qcoeff_ptr, v_zero);
    vst1q_s32(qcoeff_ptr + 4, v_zero);
    vst1q_s32(dqcoeff_ptr, v_zero);
    vst1q_s32(dqcoeff_ptr + 4, v_zero);
    return 1;
  }
#else
  (void)v_abs0;
  (void)v_abs1;
  (void)ac_thresh;
  (void)qcoeff_ptr;
  (void)dqcoeff_ptr;
#endif
  return 0;
}

static inline uint16x4_t quantize_4_from_abs(
    int32x4_t v_coeff, int32x4_t v_abs_coeff, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, int32x4_t v_quant_s32, int32x4_t v_dequant_s32,
    int32x4_t v_round_s32, int32x4_t v_log_scale, int32x4_t v_neg_log_scale,
    int32x4_t v_one_plus_log_scale) {
  const int32x4_t v_zero = vdupq_n_s32(0);
  const int32x4_t v_coeff_sign =
      vreinterpretq_s32_u32(vcltq_s32(v_coeff, v_zero));
  // ((abs_coeff << (1 + log_scale)) >= dequant_ptr[rc01])
  const int32x4_t v_abs_coeff_scaled =
      vshlq_s32(v_abs_coeff, v_one_plus_log_scale);
  const uint32x4_t v_mask = vcgeq_s32(v_abs_coeff_scaled, v_dequant_s32);
  // const int64_t tmp = vmask ? (int64_t)abs_coeff + log_scaled_round : 0
  const int32x4_t v_tmp = vandq_s32(vaddq_s32(v_abs_coeff, v_round_s32),
                                    vreinterpretq_s32_u32(v_mask));
  // const int abs_qcoeff = (int)((tmp * quant) >> (16 - log_scale));
  const int32x4_t v_abs_qcoeff =
      vqdmulhq_s32(vshlq_s32(v_tmp, v_log_scale), v_quant_s32);
  // qcoeff_ptr[rc] = (tran_low_t)((abs_qcoeff ^ coeff_sign) - coeff_sign);
  const int32x4_t v_qcoeff =
      vsubq_s32(veorq_s32(v_abs_qcoeff, v_coeff_sign), v_coeff_sign);
  // vshlq_s32 will shift right if shift value is negative.
  const int32x4_t v_abs_dqcoeff =
      vshlq_s32(vmulq_s32(v_abs_qcoeff, v_dequant_s32), v_neg_log_scale);
  // dqcoeff_ptr[rc] = (tran_low_t)((abs_dqcoeff ^ coeff_sign) - coeff_sign);
  const int32x4_t v_dqcoeff =
      vsubq_s32(veorq_s32(v_abs_dqcoeff, v_coeff_sign), v_coeff_sign);

  vst1q_s32(qcoeff_ptr, v_qcoeff);
  vst1q_s32(dqcoeff_ptr, v_dqcoeff);

  // Used to find eob.
  const uint32x4_t nz_qcoeff_mask = vcgtq_s32(v_abs_qcoeff, v_zero);
  return vmovn_u32(nz_qcoeff_mask);
}

static inline int16x8_t get_max_lane_eob(const int16_t *iscan,
                                         int16x8_t v_eobmax,
                                         uint16x8_t v_mask) {
  const int16x8_t v_iscan = vld1q_s16(iscan);
  const int16x8_t v_nz_iscan = vbslq_s16(v_mask, v_iscan, vdupq_n_s16(-1));
  return vmaxq_s16(v_eobmax, v_nz_iscan);
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

#define HBD_QUANTIZE_BODY(LS, V_ROUND_INIT)                                    \
  do {                                                                         \
    const int16x4_t v_round = (V_ROUND_INIT);                                  \
    int32x4_t v_round_s32 = vaddl_s16(v_round, v_zero);                        \
    int32x4_t v_quant_s32 = vshlq_n_s32(vaddl_s16(v_quant, v_zero), 15);       \
    int32x4_t v_dequant_s32 = vaddl_s16(v_dequant, v_zero);                    \
    const int32x4_t v_dc_coeff = vld1q_s32(coeff_ptr);                         \
    v_mask_lo = quantize_4_from_abs_ls##LS(                                    \
        v_dc_coeff, vabsq_s32(v_dc_coeff), qcoeff_ptr, dqcoeff_ptr,            \
        v_quant_s32, v_dequant_s32, v_round_s32);                              \
    v_round_s32 = vdupq_lane_s32(vget_low_s32(v_round_s32), 1);                \
    v_quant_s32 = vdupq_lane_s32(vget_low_s32(v_quant_s32), 1);                \
    v_dequant_s32 = vdupq_lane_s32(vget_low_s32(v_dequant_s32), 1);            \
    const int32x4_t v_ac_coeff = vld1q_s32(coeff_ptr + 4);                     \
    v_mask_hi = quantize_4_from_abs_ls##LS(                                    \
        v_ac_coeff, vabsq_s32(v_ac_coeff), qcoeff_ptr + 4, dqcoeff_ptr + 4,    \
        v_quant_s32, v_dequant_s32, v_round_s32);                              \
    v_eobmax =                                                                 \
        get_max_lane_eob(iscan, v_eobmax, vcombine_u16(v_mask_lo, v_mask_hi)); \
    count -= 8;                                                                \
    do {                                                                       \
      coeff_ptr += 8;                                                          \
      qcoeff_ptr += 8;                                                         \
      dqcoeff_ptr += 8;                                                        \
      iscan += 8;                                                              \
      const int32x4_t v_c0 = vld1q_s32(coeff_ptr);                             \
      const int32x4_t v_c1 = vld1q_s32(coeff_ptr + 4);                         \
      const int32x4_t v_abs0 = vabsq_s32(v_c0);                                \
      const int32x4_t v_abs1 = vabsq_s32(v_c1);                                \
      if (!hbd_zero_skip_u32(v_abs0, v_abs1, ac_thresh, qcoeff_ptr,            \
                             dqcoeff_ptr)) {                                   \
        v_mask_lo = quantize_4_from_abs_ls##LS(v_c0, v_abs0, qcoeff_ptr,       \
                                               dqcoeff_ptr, v_quant_s32,       \
                                               v_dequant_s32, v_round_s32);    \
        v_mask_hi = quantize_4_from_abs_ls##LS(v_c1, v_abs1, qcoeff_ptr + 4,   \
                                               dqcoeff_ptr + 4, v_quant_s32,   \
                                               v_dequant_s32, v_round_s32);    \
        v_eobmax = get_max_lane_eob(iscan, v_eobmax,                           \
                                    vcombine_u16(v_mask_lo, v_mask_hi));       \
      }                                                                        \
      count -= 8;                                                              \
    } while (count != 0);                                                      \
  } while (0)

void av1_highbd_quantize_fp_neon(
    const tran_low_t *coeff_ptr, intptr_t count, const int16_t *zbin_ptr,
    const int16_t *round_ptr, const int16_t *quant_ptr,
    const int16_t *quant_shift_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr,
    const int16_t *scan, const int16_t *iscan, int log_scale) {
  (void)scan;
  (void)zbin_ptr;
  (void)quant_shift_ptr;
  assert(count >= 16 && (count & 7) == 0);

  const int16x4_t v_quant = vld1_s16(quant_ptr);
  const int16x4_t v_dequant = vld1_s16(dequant_ptr);
  const int16x4_t v_zero = vdup_n_s16(0);
  uint16x4_t v_mask_lo, v_mask_hi;
  int16x8_t v_eobmax = vdupq_n_s16(-1);
  const int ac_dequant = dequant_ptr[1];
  const uint32_t ac_thresh =
      ac_dequant > 0 ? (((uint32_t)ac_dequant + (1u << (1 + log_scale)) - 1u) >>
                        (1 + log_scale))
                     : 0u;

  if (log_scale == 0) {
    HBD_QUANTIZE_BODY(0, vld1_s16(round_ptr));
  } else if (log_scale == 1) {
    HBD_QUANTIZE_BODY(1,
                      vqrdmulh_n_s16(vld1_s16(round_ptr), (int16_t)(1 << 14)));
  } else if (log_scale == 2) {
    HBD_QUANTIZE_BODY(2,
                      vqrdmulh_n_s16(vld1_s16(round_ptr), (int16_t)(1 << 13)));
  } else {
    const uint16x4_t v_round_select = vcgt_s16(vdup_n_s16(log_scale), v_zero);
    const int16x4_t v_round_no_scale = vld1_s16(round_ptr);
    const int16x4_t v_round_log_scale =
        vqrdmulh_n_s16(v_round_no_scale, (int16_t)(1 << (15 - log_scale)));
    const int16x4_t v_round =
        vbsl_s16(v_round_select, v_round_log_scale, v_round_no_scale);
    int32x4_t v_round_s32 = vaddl_s16(v_round, v_zero);
    int32x4_t v_quant_s32 = vshlq_n_s32(vaddl_s16(v_quant, v_zero), 15);
    int32x4_t v_dequant_s32 = vaddl_s16(v_dequant, v_zero);
    const int32x4_t v_log_scale = vdupq_n_s32(log_scale);
    const int32x4_t v_neg_log_scale = vnegq_s32(v_log_scale);
    const int32x4_t v_one_plus_log_scale = vdupq_n_s32(1 + log_scale);

    // DC and first 3 AC
    const int32x4_t v_dc_coeff = vld1q_s32(coeff_ptr);
    v_mask_lo = quantize_4_from_abs(v_dc_coeff, vabsq_s32(v_dc_coeff),
                                    qcoeff_ptr, dqcoeff_ptr, v_quant_s32,
                                    v_dequant_s32, v_round_s32, v_log_scale,
                                    v_neg_log_scale, v_one_plus_log_scale);

    // overwrite the DC constants with AC constants
    v_round_s32 = vdupq_lane_s32(vget_low_s32(v_round_s32), 1);
    v_quant_s32 = vdupq_lane_s32(vget_low_s32(v_quant_s32), 1);
    v_dequant_s32 = vdupq_lane_s32(vget_low_s32(v_dequant_s32), 1);

    // 4 more AC
    const int32x4_t v_ac_coeff = vld1q_s32(coeff_ptr + 4);
    v_mask_hi = quantize_4_from_abs(
        v_ac_coeff, vabsq_s32(v_ac_coeff), qcoeff_ptr + 4, dqcoeff_ptr + 4,
        v_quant_s32, v_dequant_s32, v_round_s32, v_log_scale, v_neg_log_scale,
        v_one_plus_log_scale);

    // Find the max lane eob for the first 8 coeffs.
    v_eobmax =
        get_max_lane_eob(iscan, v_eobmax, vcombine_u16(v_mask_lo, v_mask_hi));

    count -= 8;
    do {
      coeff_ptr += 8;
      qcoeff_ptr += 8;
      dqcoeff_ptr += 8;
      iscan += 8;
      const int32x4_t v_c0 = vld1q_s32(coeff_ptr);
      const int32x4_t v_c1 = vld1q_s32(coeff_ptr + 4);
      const int32x4_t v_abs0 = vabsq_s32(v_c0);
      const int32x4_t v_abs1 = vabsq_s32(v_c1);
      if (!hbd_zero_skip_u32(v_abs0, v_abs1, ac_thresh, qcoeff_ptr,
                             dqcoeff_ptr)) {
        v_mask_lo = quantize_4_from_abs(
            v_c0, v_abs0, qcoeff_ptr, dqcoeff_ptr, v_quant_s32, v_dequant_s32,
            v_round_s32, v_log_scale, v_neg_log_scale, v_one_plus_log_scale);
        v_mask_hi = quantize_4_from_abs(v_c1, v_abs1, qcoeff_ptr + 4,
                                        dqcoeff_ptr + 4, v_quant_s32,
                                        v_dequant_s32, v_round_s32, v_log_scale,
                                        v_neg_log_scale, v_one_plus_log_scale);
        v_eobmax = get_max_lane_eob(iscan, v_eobmax,
                                    vcombine_u16(v_mask_lo, v_mask_hi));
      }
      count -= 8;
    } while (count != 0);
  }

  *eob_ptr = get_max_eob(v_eobmax);
}
#undef HBD_QUANTIZE_BODY

#if CONFIG_AV1_HIGHBITDEPTH
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

static AOM_FORCE_INLINE int16x8_t highbd_quantize_fp_qm_8_neon(
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
    const int32x4_t v_sign0 = vshrq_n_s32(v_coeff0, 31);
    const int32x4_t v_sign1 = vshrq_n_s32(v_coeff1, 31);
    const int32x4_t v_tmp0 =
        vandq_s32(vaddq_s32(v_abs0, v_round0), vreinterpretq_s32_u32(v_mask0));
    const int32x4_t v_tmp1 =
        vandq_s32(vaddq_s32(v_abs1, v_round1), vreinterpretq_s32_u32(v_mask1));
    // Forward QM wt < 256 (AOM_QM_BITS = 5) and log_scale <= 2 fit in 32 bits.
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

void av1_highbd_quantize_fp_qm_neon(
    const tran_low_t *coeff_ptr, intptr_t n_coeffs, const int16_t *round_ptr,
    const int16_t *quant_ptr, tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
    const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan,
    const int16_t *iscan, const qm_val_t *qm_ptr, const qm_val_t *iqm_ptr,
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

  v_eobmax = highbd_quantize_fp_qm_8_neon(
      coeff_ptr, iscan, qm_ptr, iqm_ptr, qcoeff_ptr, dqcoeff_ptr, v_thr0,
      v_thr_ac, v_round0, v_round_ac, v_quant0, v_quant_ac, v_dequant0,
      v_dequant_ac, v_log_scale, v_neg_log_scale, v_eobmax);

  intptr_t i = 8;
  do {
    v_eobmax = highbd_quantize_fp_qm_8_neon(
        coeff_ptr + i, iscan + i, qm_ptr + i, iqm_ptr + i, qcoeff_ptr + i,
        dqcoeff_ptr + i, v_thr_ac, v_thr_ac, v_round_ac, v_round_ac, v_quant_ac,
        v_quant_ac, v_dequant_ac, v_dequant_ac, v_log_scale, v_neg_log_scale,
        v_eobmax);
    i += 8;
  } while (i != n_coeffs);
  *eob_ptr = get_max_eob(v_eobmax);
}

static AOM_FORCE_INLINE int16x8_t highbd_quantize_b_qm_8_neon(
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
    const int32x4_t v_sign0 = vshrq_n_s32(v_coeff0, 31);
    const int32x4_t v_sign1 = vshrq_n_s32(v_coeff1, 31);
    const int32x4_t v_tmp0 =
        vandq_s32(vaddq_s32(v_abs0, v_round0), vreinterpretq_s32_u32(v_mask0));
    const int32x4_t v_tmp1 =
        vandq_s32(vaddq_s32(v_abs1, v_round1), vreinterpretq_s32_u32(v_mask1));
    // Forward QM wt < 256 (AOM_QM_BITS = 5) and log_scale <= 2 fit in 32 bits.
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

void av1_highbd_quantize_b_qm_neon(
    const tran_low_t *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr,
    const int16_t *round_ptr, const int16_t *quant_ptr,
    const int16_t *quant_shift_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr,
    const int16_t *scan, const int16_t *iscan, const qm_val_t *qm_ptr,
    const qm_val_t *iqm_ptr, int log_scale) {
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

  v_eobmax = highbd_quantize_b_qm_8_neon(
      coeff_ptr, iscan, qm_ptr, iqm_ptr, qcoeff_ptr, dqcoeff_ptr, v_zbin0,
      v_zbin_ac, v_round0, v_round_ac, v_quant0, v_quant_ac, v_qshift0,
      v_qshift_ac, v_dequant0, v_dequant_ac, v_log_scale, v_neg_log_scale,
      v_eobmax);

  intptr_t i = 8;
  do {
    v_eobmax = highbd_quantize_b_qm_8_neon(
        coeff_ptr + i, iscan + i, qm_ptr + i, iqm_ptr + i, qcoeff_ptr + i,
        dqcoeff_ptr + i, v_zbin_ac, v_zbin_ac, v_round_ac, v_round_ac,
        v_quant_ac, v_quant_ac, v_qshift_ac, v_qshift_ac, v_dequant_ac,
        v_dequant_ac, v_log_scale, v_neg_log_scale, v_eobmax);
    i += 8;
  } while (i != n_coeffs);
  *eob_ptr = get_max_eob(v_eobmax);
}
#endif  // CONFIG_AV1_HIGHBITDEPTH
