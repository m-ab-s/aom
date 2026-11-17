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

#include "av1/common/arm/av1_inv_txfm_neon.h"

#include <arm_neon.h>
#include <assert.h>

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"
#include "config/av1_rtcd.h"

#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/arm/mem_neon.h"
#include "aom_dsp/arm/transpose_neon.h"
#include "av1/common/av1_inv_txfm1d_cfg.h"
#include "av1/common/av1_inv_txfm2d.h"
#include "av1/common/av1_txfm.h"
#include "av1/common/enums.h"
#include "av1/common/idct.h"

// 1D itx types
typedef enum ATTRIBUTE_PACKED {
  IDCT_1D,
  IADST_1D,
  IFLIPADST_1D = IADST_1D,
  IIDENTITY_1D,
  ITX_TYPES_1D,
} ITX_TYPE_1D;

static const ITX_TYPE_1D vitx_1d_tab[TX_TYPES] = {
  IDCT_1D,      IADST_1D,     IDCT_1D,      IADST_1D,
  IFLIPADST_1D, IDCT_1D,      IFLIPADST_1D, IADST_1D,
  IFLIPADST_1D, IIDENTITY_1D, IDCT_1D,      IIDENTITY_1D,
  IADST_1D,     IIDENTITY_1D, IFLIPADST_1D, IIDENTITY_1D,
};

static const ITX_TYPE_1D hitx_1d_tab[TX_TYPES] = {
  IDCT_1D,      IDCT_1D,      IADST_1D,     IADST_1D,
  IDCT_1D,      IFLIPADST_1D, IFLIPADST_1D, IFLIPADST_1D,
  IADST_1D,     IIDENTITY_1D, IIDENTITY_1D, IDCT_1D,
  IIDENTITY_1D, IADST_1D,     IIDENTITY_1D, IFLIPADST_1D,
};

static inline void lowbd_add_flip_buffer_8xn_neon(int16x8_t *in,
                                                  uint8_t *output, int stride,
                                                  int flipud,
                                                  const int height) {
  int j = flipud ? (height - 1) : 0;
  const int step = flipud ? -1 : 1;
  int16x8_t temp_output;
  for (int i = 0; i < height; ++i, j += step) {
    temp_output = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(output)));
    temp_output = vaddq_s16(temp_output, in[j]);
    vst1_u8(output, vqmovun_s16(temp_output));
    output += stride;
  }
}

static inline uint8x16_t lowbd_get_recon_16x16_neon(const uint8x16_t pred,
                                                    int16x8_t res0,
                                                    int16x8_t res1) {
  int16x8_t temp_output[2];
  uint8x16_t temp_output_8q;
  temp_output[0] = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(pred)));
  temp_output[0] = vaddq_s16(temp_output[0], res0);
  temp_output[1] = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(pred)));
  temp_output[1] = vaddq_s16(temp_output[1], res1);
  temp_output_8q =
      vcombine_u8(vqmovun_s16(temp_output[0]), vqmovun_s16(temp_output[1]));
  return temp_output_8q;
}

static inline void lowbd_add_flip_buffer_16xn_neon(int16x8_t *in,
                                                   uint8_t *output, int stride,
                                                   int flipud, int height) {
  uint8x16_t temp_output_8q;
  int j = flipud ? (height - 1) : 0;
  const int step = flipud ? -1 : 1;
  for (int i = 0; i < height; ++i, j += step) {
    temp_output_8q = vld1q_u8(output + i * stride);
    temp_output_8q =
        lowbd_get_recon_16x16_neon(temp_output_8q, in[j], in[j + height]);
    vst1q_u8((output + i * stride), temp_output_8q);
  }
}

static inline void lowbd_inv_txfm2d_memset_neon(int16x8_t *a, int size,
                                                int value) {
  for (int i = 0; i < size; i++) {
    a[i] = vdupq_n_s16((int16_t)value);
  }
}

static inline void btf_16_lane_0_1_neon(const int16x8_t in0,
                                        const int16x8_t in1, const int16x4_t c,
                                        int16x8_t *t0, int16x8_t *t1) {
  int32x4_t s0[2], s1[2];
  int16x4_t v0[2], v1[2];

  s0[0] = vmull_lane_s16(vget_low_s16(in0), c, 0);
  s0[1] = vmull_lane_s16(vget_high_s16(in0), c, 0);
  s1[0] = vmull_lane_s16(vget_low_s16(in0), c, 1);
  s1[1] = vmull_lane_s16(vget_high_s16(in0), c, 1);

  s0[0] = vmlal_lane_s16(s0[0], vget_low_s16(in1), c, 1);
  s0[1] = vmlal_lane_s16(s0[1], vget_high_s16(in1), c, 1);
  s1[0] = vmlsl_lane_s16(s1[0], vget_low_s16(in1), c, 0);
  s1[1] = vmlsl_lane_s16(s1[1], vget_high_s16(in1), c, 0);

  v0[0] = vrshrn_n_s32(s0[0], INV_COS_BIT);
  v0[1] = vrshrn_n_s32(s0[1], INV_COS_BIT);
  v1[0] = vrshrn_n_s32(s1[0], INV_COS_BIT);
  v1[1] = vrshrn_n_s32(s1[1], INV_COS_BIT);

  *t0 = vcombine_s16(v0[0], v0[1]);
  *t1 = vcombine_s16(v1[0], v1[1]);
}

static inline void btf_16_lane_1_0_neon(const int16x8_t in0,
                                        const int16x8_t in1, const int16x4_t c,
                                        int16x8_t *t0, int16x8_t *t1) {
  int32x4_t s0[2], s1[2];
  int16x4_t v0[2], v1[2];

  s0[0] = vmull_lane_s16(vget_low_s16(in0), c, 1);
  s0[1] = vmull_lane_s16(vget_high_s16(in0), c, 1);
  s1[0] = vmull_lane_s16(vget_low_s16(in0), c, 0);
  s1[1] = vmull_lane_s16(vget_high_s16(in0), c, 0);

  s0[0] = vmlal_lane_s16(s0[0], vget_low_s16(in1), c, 0);
  s0[1] = vmlal_lane_s16(s0[1], vget_high_s16(in1), c, 0);
  s1[0] = vmlsl_lane_s16(s1[0], vget_low_s16(in1), c, 1);
  s1[1] = vmlsl_lane_s16(s1[1], vget_high_s16(in1), c, 1);

  v0[0] = vrshrn_n_s32(s0[0], INV_COS_BIT);
  v0[1] = vrshrn_n_s32(s0[1], INV_COS_BIT);
  v1[0] = vrshrn_n_s32(s1[0], INV_COS_BIT);
  v1[1] = vrshrn_n_s32(s1[1], INV_COS_BIT);

  *t0 = vcombine_s16(v0[0], v0[1]);
  *t1 = vcombine_s16(v1[0], v1[1]);
}

static inline void btf_16_lane_2_3_neon(const int16x8_t in0,
                                        const int16x8_t in1, const int16x4_t c,
                                        int16x8_t *t0, int16x8_t *t1) {
  int32x4_t s0[2], s1[2];
  int16x4_t v0[2], v1[2];

  s0[0] = vmull_lane_s16(vget_low_s16(in0), c, 2);
  s0[1] = vmull_lane_s16(vget_high_s16(in0), c, 2);
  s1[0] = vmull_lane_s16(vget_low_s16(in0), c, 3);
  s1[1] = vmull_lane_s16(vget_high_s16(in0), c, 3);

  s0[0] = vmlal_lane_s16(s0[0], vget_low_s16(in1), c, 3);
  s0[1] = vmlal_lane_s16(s0[1], vget_high_s16(in1), c, 3);
  s1[0] = vmlsl_lane_s16(s1[0], vget_low_s16(in1), c, 2);
  s1[1] = vmlsl_lane_s16(s1[1], vget_high_s16(in1), c, 2);

  v0[0] = vrshrn_n_s32(s0[0], INV_COS_BIT);
  v0[1] = vrshrn_n_s32(s0[1], INV_COS_BIT);
  v1[0] = vrshrn_n_s32(s1[0], INV_COS_BIT);
  v1[1] = vrshrn_n_s32(s1[1], INV_COS_BIT);

  *t0 = vcombine_s16(v0[0], v0[1]);
  *t1 = vcombine_s16(v1[0], v1[1]);
}

static inline void btf_16_neon(const int16x8_t in0, int16_t coef1,
                               int16_t coef2, int16x8_t *t0, int16x8_t *t1) {
  int32x4_t s0_l, s0_h, s1_l, s1_h;
  int16x4_t v0[2], v1[2];

  s0_l = vmull_n_s16(vget_low_s16(in0), coef1);
  s0_h = vmull_n_s16(vget_high_s16(in0), coef1);
  s1_l = vmull_n_s16(vget_low_s16(in0), coef2);
  s1_h = vmull_n_s16(vget_high_s16(in0), coef2);

  v0[0] = vrshrn_n_s32(s0_l, INV_COS_BIT);
  v0[1] = vrshrn_n_s32(s0_h, INV_COS_BIT);
  v1[0] = vrshrn_n_s32(s1_l, INV_COS_BIT);
  v1[1] = vrshrn_n_s32(s1_h, INV_COS_BIT);

  *t0 = vcombine_s16(v0[0], v0[1]);
  *t1 = vcombine_s16(v1[0], v1[1]);
}

static inline void btf_16_lane_3_2_neon(const int16x8_t in0,
                                        const int16x8_t in1, const int16x4_t c,
                                        int16x8_t *t0, int16x8_t *t1) {
  int32x4_t s0[2], s1[2];
  int16x4_t v0[2], v1[2];

  s0[0] = vmull_lane_s16(vget_low_s16(in0), c, 3);
  s0[1] = vmull_lane_s16(vget_high_s16(in0), c, 3);
  s1[0] = vmull_lane_s16(vget_low_s16(in0), c, 2);
  s1[1] = vmull_lane_s16(vget_high_s16(in0), c, 2);

  s0[0] = vmlal_lane_s16(s0[0], vget_low_s16(in1), c, 2);
  s0[1] = vmlal_lane_s16(s0[1], vget_high_s16(in1), c, 2);
  s1[0] = vmlsl_lane_s16(s1[0], vget_low_s16(in1), c, 3);
  s1[1] = vmlsl_lane_s16(s1[1], vget_high_s16(in1), c, 3);

  v0[0] = vrshrn_n_s32(s0[0], INV_COS_BIT);
  v0[1] = vrshrn_n_s32(s0[1], INV_COS_BIT);
  v1[0] = vrshrn_n_s32(s1[0], INV_COS_BIT);
  v1[1] = vrshrn_n_s32(s1[1], INV_COS_BIT);

  *t0 = vcombine_s16(v0[0], v0[1]);
  *t1 = vcombine_s16(v1[0], v1[1]);
}

static inline void btf_16_half_neon(int16x8_t *const x, const int16x4_t c) {
  int32x4_t t0[2], t1[2];
  int16x4_t v0[2], v1[2];

  // Don't add/sub before multiply, which will overflow in iadst8.
  const int32x4_t x0_lo = vmull_lane_s16(vget_low_s16(x[0]), c, 0);
  const int32x4_t x0_hi = vmull_lane_s16(vget_high_s16(x[0]), c, 0);
  const int32x4_t x1_lo = vmull_lane_s16(vget_low_s16(x[1]), c, 0);
  const int32x4_t x1_hi = vmull_lane_s16(vget_high_s16(x[1]), c, 0);

  t0[0] = vaddq_s32(x0_lo, x1_lo);
  t0[1] = vaddq_s32(x0_hi, x1_hi);
  t1[0] = vsubq_s32(x0_lo, x1_lo);
  t1[1] = vsubq_s32(x0_hi, x1_hi);

  v0[0] = vrshrn_n_s32(t0[0], INV_COS_BIT);
  v0[1] = vrshrn_n_s32(t0[1], INV_COS_BIT);
  v1[0] = vrshrn_n_s32(t1[0], INV_COS_BIT);
  v1[1] = vrshrn_n_s32(t1[1], INV_COS_BIT);

  x[0] = vcombine_s16(v0[0], v0[1]);
  x[1] = vcombine_s16(v1[0], v1[1]);
}

static inline int16x4_t set_s16x4_neon(const int16_t c0, const int16_t c1,
                                       const int16_t c2, const int16_t c3) {
  int16x4_t val = vdup_n_s16(c0);
  val = vset_lane_s16(c1, val, 1);
  val = vset_lane_s16(c2, val, 2);
  val = vset_lane_s16(c3, val, 3);
  return val;
}

static AOM_FORCE_INLINE void iadst8_neon_impl(int16x8_t *const in,
                                              int16x8_t *out, int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);

  const int16x4_t c0 = set_s16x4_neon((int16_t)cospi[4], (int16_t)cospi[60],
                                      (int16_t)cospi[20], (int16_t)cospi[44]);
  const int16x4_t c1 = set_s16x4_neon((int16_t)cospi[36], (int16_t)cospi[28],
                                      (int16_t)cospi[52], (int16_t)cospi[12]);
  const int16x4_t c2 = set_s16x4_neon((int16_t)cospi[32], (int16_t)cospi[32],
                                      (int16_t)cospi[16], (int16_t)cospi[48]);

  int16x8_t x[8];
  int16x8_t s0, s1, s2, s3, s4, s5, s6, s7;

  // Stage 1
  x[0] = in[7];
  x[1] = in[0];
  x[2] = in[5];
  x[3] = in[2];
  x[4] = in[3];
  x[5] = in[4];
  x[6] = in[1];
  x[7] = in[6];

  // Stage 2
  btf_16_lane_0_1_neon(x[0], x[1], c0, &s0, &s1);
  btf_16_lane_2_3_neon(x[2], x[3], c0, &s2, &s3);
  btf_16_lane_0_1_neon(x[4], x[5], c1, &s4, &s5);
  btf_16_lane_2_3_neon(x[6], x[7], c1, &s6, &s7);

  // Stage 3
  x[0] = vqaddq_s16(s0, s4);
  x[1] = vqaddq_s16(s1, s5);
  x[2] = vqaddq_s16(s2, s6);
  x[3] = vqaddq_s16(s3, s7);
  x[4] = vqsubq_s16(s0, s4);
  x[5] = vqsubq_s16(s1, s5);
  x[6] = vqsubq_s16(s2, s6);
  x[7] = vqsubq_s16(s3, s7);

  // Stage 4
  s0 = x[0];
  s1 = x[1];
  s2 = x[2];
  s3 = x[3];
  btf_16_lane_2_3_neon(x[4], x[5], c2, &s4, &s5);
  btf_16_lane_3_2_neon(x[7], x[6], c2, &s7, &s6);

  // Stage 5
  x[0] = vqaddq_s16(s0, s2);
  x[1] = vqaddq_s16(s1, s3);
  x[2] = vqsubq_s16(s0, s2);
  x[3] = vqsubq_s16(s1, s3);
  x[4] = vqaddq_s16(s4, s6);
  x[5] = vqaddq_s16(s5, s7);
  x[6] = vqsubq_s16(s4, s6);
  x[7] = vqsubq_s16(s5, s7);

  // stage 6
  btf_16_half_neon(x + 2, c2);
  btf_16_half_neon(x + 6, c2);

  // Stage 7
  out[0] = x[0];
  out[1] = vqnegq_s16(x[4]);
  out[2] = x[6];
  out[3] = vqnegq_s16(x[2]);
  out[4] = x[3];
  out[5] = vqnegq_s16(x[7]);
  out[6] = x[5];
  out[7] = vqnegq_s16(x[1]);
}

static inline void iadst8_neon(int16x8_t *const in, int16x8_t *out,
                               int8_t cos_bit) {
  iadst8_neon_impl(in, out, cos_bit);
}

static AOM_FORCE_INLINE void iadst8_low1_neon_impl(int16x8_t *const in,
                                                   int16x8_t *out,
                                                   int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);
  const int16x4_t c2 = set_s16x4_neon((int16_t)cospi[32], (int16_t)cospi[32],
                                      (int16_t)cospi[16], (int16_t)cospi[48]);

  int16x8_t x[8];
  int16x8_t s0, s1, s4, s5;

  // Stage 1
  x[1] = in[0];

  // Stage 2

  btf_16_neon(x[1], cospi[60], -cospi[4], &s0, &s1);

  // Stage 3
  x[0] = s0;
  x[1] = s1;
  x[4] = s0;
  x[5] = s1;

  // Stage 4
  s0 = x[0];
  s1 = x[1];
  btf_16_lane_2_3_neon(x[4], x[5], c2, &s4, &s5);

  // Stage 5
  x[0] = s0;
  x[1] = s1;
  x[2] = s0;
  x[3] = s1;
  x[4] = s4;
  x[5] = s5;
  x[6] = s4;
  x[7] = s5;

  // stage 6
  btf_16_half_neon(x + 2, c2);
  btf_16_half_neon(x + 6, c2);

  // Stage 7
  out[0] = x[0];
  out[1] = vqnegq_s16(x[4]);
  out[2] = x[6];
  out[3] = vqnegq_s16(x[2]);
  out[4] = x[3];
  out[5] = vqnegq_s16(x[7]);
  out[6] = x[5];
  out[7] = vqnegq_s16(x[1]);
}

static inline void iadst8_low1_neon(int16x8_t *const in, int16x8_t *out,
                                    int8_t cos_bit) {
  iadst8_low1_neon_impl(in, out, cos_bit);
}

static AOM_FORCE_INLINE void idct8_neon_impl(int16x8_t *in, int16x8_t *out,
                                             int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);
  int16x8_t step1[8], step2[8];
  const int16x4_t c0 = set_s16x4_neon((int16_t)cospi[8], (int16_t)cospi[56],
                                      (int16_t)cospi[40], (int16_t)cospi[24]);
  const int16x4_t c1 = set_s16x4_neon((int16_t)cospi[32], (int16_t)cospi[32],
                                      (int16_t)cospi[16], (int16_t)cospi[48]);

  // stage 2
  btf_16_lane_0_1_neon(in[1], in[7], c0, &step1[7], &step1[4]);
  btf_16_lane_2_3_neon(in[5], in[3], c0, &step1[6], &step1[5]);

  // stage 3
  btf_16_lane_0_1_neon(in[0], in[4], c1, &step2[0], &step2[1]);
  btf_16_lane_2_3_neon(in[2], in[6], c1, &step2[3], &step2[2]);
  step2[4] = vqaddq_s16(step1[4], step1[5]);
  step2[5] = vqsubq_s16(step1[4], step1[5]);
  step2[6] = vqsubq_s16(step1[7], step1[6]);
  step2[7] = vqaddq_s16(step1[7], step1[6]);

  // stage 4
  step1[0] = vqaddq_s16(step2[0], step2[3]);
  step1[1] = vqaddq_s16(step2[1], step2[2]);
  step1[2] = vqsubq_s16(step2[1], step2[2]);
  step1[3] = vqsubq_s16(step2[0], step2[3]);
  btf_16_lane_0_1_neon(step2[6], step2[5], c1, &step1[6], &step1[5]);

  // stage 5
  out[0] = vqaddq_s16(step1[0], step2[7]);
  out[1] = vqaddq_s16(step1[1], step1[6]);
  out[2] = vqaddq_s16(step1[2], step1[5]);
  out[3] = vqaddq_s16(step1[3], step2[4]);
  out[4] = vqsubq_s16(step1[3], step2[4]);
  out[5] = vqsubq_s16(step1[2], step1[5]);
  out[6] = vqsubq_s16(step1[1], step1[6]);
  out[7] = vqsubq_s16(step1[0], step2[7]);
}

static inline void idct8_neon(int16x8_t *in, int16x8_t *out, int8_t cos_bit) {
  idct8_neon_impl(in, out, cos_bit);
}

static AOM_FORCE_INLINE void idct8_low1_neon_impl(int16x8_t *in, int16x8_t *out,
                                                  int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);
  int16x8_t step1;
  int32x4_t t32[2];

  // stage 1
  // stage 2
  // stage 3
  t32[0] = vmull_n_s16(vget_low_s16(in[0]), (int16_t)cospi[32]);
  t32[1] = vmull_n_s16(vget_high_s16(in[0]), (int16_t)cospi[32]);

  step1 = vcombine_s16(vrshrn_n_s32(t32[0], INV_COS_BIT),
                       vrshrn_n_s32(t32[1], INV_COS_BIT));

  // stage 4
  // stage 5
  out[0] = step1;
  out[1] = step1;
  out[2] = step1;
  out[3] = step1;
  out[4] = step1;
  out[5] = step1;
  out[6] = step1;
  out[7] = step1;
}

static inline void idct8_low1_neon(int16x8_t *in, int16x8_t *out,
                                   int8_t cos_bit) {
  idct8_low1_neon_impl(in, out, cos_bit);
}

static void round_shift_array_16_neon(int16x8_t *arr, int size, int bit) {
  assert(!(size % 4));
  if (!bit) return;
  const int16x8_t dup_bits_n_16x8 = vdupq_n_s16((int16_t)(-bit));
  for (int i = 0; i < size; i++) {
    arr[i] = vrshlq_s16(arr[i], dup_bits_n_16x8);
  }
}

static inline void flip_buf_ud_neon(int16x8_t *input, int size) {
  int16x8_t temp[8];
  for (int i = 0; i < size; ++i) {
    temp[i] = input[size - 1 - i];
  }
  for (int i = 0; i < size; ++i) {
    input[i] = temp[i];
  }
}

static inline void load_buffer_32bit_to_16bit_neon(const int32_t *input,
                                                   int stride,
                                                   int16x8_t *const a,
                                                   int out_size) {
  for (int i = 0; i < out_size; ++i) {
    a[i] = vcombine_s16(vmovn_s32(vld1q_s32(input)),
                        vmovn_s32(vld1q_s32(input + 4)));
    input += stride;
  }
}

static const int16_t sqrt_2_list[TX_SIZES] = { 5793, 2 * 4096, 2 * 5793,
                                               4 * 4096, 4 * 5793 };

static inline void identity_txfm_round_neon(int16x8_t *input, int16x8_t *output,
                                            int txw_idx, int8_t size, int bit) {
  const int32x4_t dup_bits_n_32x4 = vdupq_n_s32((int32_t)(-bit));
  int16x4_t scale = vdup_n_s16(sqrt_2_list[txw_idx]);
  int16x4_t low_i16, high_i16;
  int32x4_t low_i32, high_i32;
  for (int i = 0; i < size; i++) {
    int32x4_t temp_out_low = vmull_s16(vget_low_s16(input[i]), scale);
    int32x4_t temp_out_high = vmull_s16(vget_high_s16(input[i]), scale);
    low_i32 = vrshlq_s32(vrshrq_n_s32(temp_out_low, 12), dup_bits_n_32x4);
    high_i32 = vrshlq_s32(vrshrq_n_s32(temp_out_high, 12), dup_bits_n_32x4);
    low_i16 = vqmovn_s32(low_i32);
    high_i16 = vqmovn_s32(high_i32);
    output[i] = vcombine_s16(low_i16, high_i16);
  }
}

static inline void round_shift_for_rect(int16x8_t *input, int16x8_t *output,
                                        int size) {
  int32x4_t out_low, out_high;
  int16x4_t low, high;

  for (int z = 0; z < size; ++z) {
    out_low = vmull_n_s16(vget_low_s16(input[z]), (int16_t)NewInvSqrt2);
    out_high = vmull_n_s16(vget_high_s16(input[z]), (int16_t)NewInvSqrt2);

    low = vqrshrn_n_s32(out_low, (int32_t)NewSqrt2Bits);
    high = vqrshrn_n_s32(out_high, (int32_t)NewSqrt2Bits);

    output[z] = vcombine_s16(low, high);
  }
}

static inline void idct16_low1_neon(int16x8_t *in, int16x8_t *out,
                                    int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);
  int16x8_t step1;
  int32x4_t t32[2];

  // stage 4

  t32[0] = vmull_n_s16(vget_low_s16(in[0]), cospi[32]);
  t32[1] = vmull_n_s16(vget_high_s16(in[0]), cospi[32]);
  step1 = vcombine_s16(vrshrn_n_s32(t32[0], INV_COS_BIT),
                       vrshrn_n_s32(t32[1], INV_COS_BIT));

  // stage 6
  // stage 7
  out[0] = step1;
  out[1] = step1;
  out[2] = step1;
  out[3] = step1;
  out[4] = step1;
  out[5] = step1;
  out[6] = step1;
  out[7] = step1;
  out[8] = step1;
  out[9] = step1;
  out[10] = step1;
  out[11] = step1;
  out[12] = step1;
  out[13] = step1;
  out[14] = step1;
  out[15] = step1;
}

static inline void idct16_neon(int16x8_t *in, int16x8_t *out, int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);
  int16x8_t step1[16], step2[16];

  const int16x4_t c0 = set_s16x4_neon((int16_t)cospi[4], (int16_t)cospi[60],
                                      (int16_t)cospi[36], (int16_t)cospi[28]);
  const int16x4_t c1 = set_s16x4_neon((int16_t)cospi[20], (int16_t)cospi[44],
                                      (int16_t)cospi[52], (int16_t)cospi[12]);
  const int16x4_t c2 = set_s16x4_neon((int16_t)cospi[8], (int16_t)cospi[56],
                                      (int16_t)cospi[40], (int16_t)cospi[24]);
  const int16x4_t c3 = set_s16x4_neon((int16_t)cospi[32], (int16_t)cospi[32],
                                      (int16_t)cospi[16], (int16_t)cospi[48]);
  const int16x4_t c4 =
      set_s16x4_neon((int16_t)(-cospi[32]), (int16_t)(-cospi[32]),
                     (int16_t)(-cospi[16]), (int16_t)(-cospi[48]));
  // stage 2

  btf_16_lane_0_1_neon(in[1], in[15], c0, &step2[15], &step2[8]);
  btf_16_lane_2_3_neon(in[9], in[7], c0, &step2[14], &step2[9]);
  btf_16_lane_0_1_neon(in[5], in[11], c1, &step2[13], &step2[10]);
  btf_16_lane_2_3_neon(in[13], in[3], c1, &step2[12], &step2[11]);

  step2[0] = in[0];
  step2[1] = in[8];
  step2[2] = in[4];
  step2[3] = in[12];
  step2[4] = in[2];
  step2[5] = in[10];
  step2[6] = in[6];
  step2[7] = in[14];

  // stage 3

  btf_16_lane_0_1_neon(step2[4], step2[7], c2, &step1[7], &step1[4]);
  btf_16_lane_2_3_neon(step2[5], step2[6], c2, &step1[6], &step1[5]);

  step1[0] = step2[0];
  step1[1] = step2[1];
  step1[2] = step2[2];
  step1[3] = step2[3];
  step1[8] = vqaddq_s16(step2[8], step2[9]);
  step1[9] = vqsubq_s16(step2[8], step2[9]);
  step1[10] = vqsubq_s16(step2[11], step2[10]);
  step1[11] = vqaddq_s16(step2[11], step2[10]);
  step1[12] = vqaddq_s16(step2[12], step2[13]);
  step1[13] = vqsubq_s16(step2[12], step2[13]);
  step1[14] = vqsubq_s16(step2[15], step2[14]);
  step1[15] = vqaddq_s16(step2[15], step2[14]);

  // stage 4

  btf_16_lane_0_1_neon(step1[0], step1[1], c3, &step2[0], &step2[1]);
  btf_16_lane_2_3_neon(step1[2], step1[3], c3, &step2[3], &step2[2]);
  btf_16_lane_2_3_neon(step1[14], step1[9], c3, &step2[14], &step2[9]);
  btf_16_lane_3_2_neon(step1[10], step1[13], c4, &step2[10], &step2[13]);

  step2[4] = vqaddq_s16(step1[4], step1[5]);
  step2[5] = vqsubq_s16(step1[4], step1[5]);
  step2[6] = vqsubq_s16(step1[7], step1[6]);
  step2[7] = vqaddq_s16(step1[7], step1[6]);
  step2[8] = step1[8];
  step2[11] = step1[11];
  step2[12] = step1[12];
  step2[15] = step1[15];

  // stage 5

  btf_16_lane_0_1_neon(step2[6], step2[5], c3, &step1[6], &step1[5]);

  step1[0] = vqaddq_s16(step2[0], step2[3]);
  step1[1] = vqaddq_s16(step2[1], step2[2]);
  step1[2] = vqsubq_s16(step2[1], step2[2]);
  step1[3] = vqsubq_s16(step2[0], step2[3]);
  step1[4] = step2[4];
  step1[7] = step2[7];
  step1[8] = vqaddq_s16(step2[8], step2[11]);
  step1[9] = vqaddq_s16(step2[9], step2[10]);
  step1[10] = vqsubq_s16(step2[9], step2[10]);
  step1[11] = vqsubq_s16(step2[8], step2[11]);
  step1[12] = vqsubq_s16(step2[15], step2[12]);
  step1[13] = vqsubq_s16(step2[14], step2[13]);
  step1[14] = vqaddq_s16(step2[14], step2[13]);
  step1[15] = vqaddq_s16(step2[15], step2[12]);

  // stage 6

  btf_16_lane_0_1_neon(step1[13], step1[10], c3, &step2[13], &step2[10]);
  btf_16_lane_0_1_neon(step1[12], step1[11], c3, &step2[12], &step2[11]);

  step2[0] = vqaddq_s16(step1[0], step1[7]);
  step2[1] = vqaddq_s16(step1[1], step1[6]);
  step2[2] = vqaddq_s16(step1[2], step1[5]);
  step2[3] = vqaddq_s16(step1[3], step1[4]);
  step2[4] = vqsubq_s16(step1[3], step1[4]);
  step2[5] = vqsubq_s16(step1[2], step1[5]);
  step2[6] = vqsubq_s16(step1[1], step1[6]);
  step2[7] = vqsubq_s16(step1[0], step1[7]);
  step2[8] = step1[8];
  step2[9] = step1[9];
  step2[14] = step1[14];
  step2[15] = step1[15];

  // stage 7
  out[0] = vqaddq_s16(step2[0], step2[15]);
  out[1] = vqaddq_s16(step2[1], step2[14]);
  out[2] = vqaddq_s16(step2[2], step2[13]);
  out[3] = vqaddq_s16(step2[3], step2[12]);
  out[4] = vqaddq_s16(step2[4], step2[11]);
  out[5] = vqaddq_s16(step2[5], step2[10]);
  out[6] = vqaddq_s16(step2[6], step2[9]);
  out[7] = vqaddq_s16(step2[7], step2[8]);
  out[8] = vqsubq_s16(step2[7], step2[8]);
  out[9] = vqsubq_s16(step2[6], step2[9]);
  out[10] = vqsubq_s16(step2[5], step2[10]);
  out[11] = vqsubq_s16(step2[4], step2[11]);
  out[12] = vqsubq_s16(step2[3], step2[12]);
  out[13] = vqsubq_s16(step2[2], step2[13]);
  out[14] = vqsubq_s16(step2[1], step2[14]);
  out[15] = vqsubq_s16(step2[0], step2[15]);
}

static inline void idct16_low8_neon(int16x8_t *in, int16x8_t *out,
                                    int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);
  int16x8_t step1[16], step2[16];
  const int16x4_t c0 = set_s16x4_neon((int16_t)cospi[32], (int16_t)cospi[32],
                                      (int16_t)cospi[16], (int16_t)cospi[48]);
  const int16x4_t c1 =
      set_s16x4_neon((int16_t)(-cospi[32]), (int16_t)(-cospi[32]),
                     (int16_t)(-cospi[16]), (int16_t)(-cospi[48]));

  // stage 1
  // stage 2

  step2[0] = in[0];
  step2[2] = in[4];
  step2[4] = in[2];
  step2[6] = in[6];

  btf_16_neon(in[1], cospi[60], cospi[4], &step2[8], &step2[15]);
  btf_16_neon(in[7], -cospi[36], cospi[28], &step2[9], &step2[14]);
  btf_16_neon(in[5], cospi[44], cospi[20], &step2[10], &step2[13]);
  btf_16_neon(in[3], -cospi[52], cospi[12], &step2[11], &step2[12]);

  // stage 3

  btf_16_neon(step2[4], cospi[56], cospi[8], &step1[4], &step1[7]);
  btf_16_neon(step2[6], -cospi[40], cospi[24], &step1[5], &step1[6]);

  step1[0] = step2[0];
  step1[2] = step2[2];
  step1[8] = vqaddq_s16(step2[8], step2[9]);
  step1[9] = vqsubq_s16(step2[8], step2[9]);
  step1[10] = vqsubq_s16(step2[11], step2[10]);
  step1[11] = vqaddq_s16(step2[11], step2[10]);
  step1[12] = vqaddq_s16(step2[12], step2[13]);
  step1[13] = vqsubq_s16(step2[12], step2[13]);
  step1[14] = vqsubq_s16(step2[15], step2[14]);
  step1[15] = vqaddq_s16(step2[15], step2[14]);

  // stage 4

  btf_16_neon(step1[0], cospi[32], cospi[32], &step2[0], &step2[1]);
  btf_16_neon(step1[2], cospi[48], cospi[16], &step2[2], &step2[3]);
  btf_16_lane_2_3_neon(step1[14], step1[9], c0, &step2[14], &step2[9]);
  btf_16_lane_3_2_neon(step1[10], step1[13], c1, &step2[10], &step2[13]);

  step2[4] = vqaddq_s16(step1[4], step1[5]);
  step2[5] = vqsubq_s16(step1[4], step1[5]);
  step2[6] = vqsubq_s16(step1[7], step1[6]);
  step2[7] = vqaddq_s16(step1[7], step1[6]);
  step2[8] = step1[8];
  step2[11] = step1[11];
  step2[12] = step1[12];
  step2[15] = step1[15];

  // stage 5

  btf_16_lane_0_1_neon(step2[6], step2[5], c0, &step1[6], &step1[5]);
  step1[0] = vqaddq_s16(step2[0], step2[3]);
  step1[1] = vqaddq_s16(step2[1], step2[2]);
  step1[2] = vqsubq_s16(step2[1], step2[2]);
  step1[3] = vqsubq_s16(step2[0], step2[3]);
  step1[4] = step2[4];
  step1[7] = step2[7];
  step1[8] = vqaddq_s16(step2[8], step2[11]);
  step1[9] = vqaddq_s16(step2[9], step2[10]);
  step1[10] = vqsubq_s16(step2[9], step2[10]);
  step1[11] = vqsubq_s16(step2[8], step2[11]);
  step1[12] = vqsubq_s16(step2[15], step2[12]);
  step1[13] = vqsubq_s16(step2[14], step2[13]);
  step1[14] = vqaddq_s16(step2[14], step2[13]);
  step1[15] = vqaddq_s16(step2[15], step2[12]);

  // stage 6
  btf_16_lane_0_1_neon(step1[13], step1[10], c0, &step2[13], &step2[10]);
  btf_16_lane_0_1_neon(step1[12], step1[11], c0, &step2[12], &step2[11]);

  step2[0] = vqaddq_s16(step1[0], step1[7]);
  step2[1] = vqaddq_s16(step1[1], step1[6]);
  step2[2] = vqaddq_s16(step1[2], step1[5]);
  step2[3] = vqaddq_s16(step1[3], step1[4]);
  step2[4] = vqsubq_s16(step1[3], step1[4]);
  step2[5] = vqsubq_s16(step1[2], step1[5]);
  step2[6] = vqsubq_s16(step1[1], step1[6]);
  step2[7] = vqsubq_s16(step1[0], step1[7]);
  step2[8] = step1[8];
  step2[9] = step1[9];
  step2[14] = step1[14];
  step2[15] = step1[15];

  // stage 7

  out[0] = vqaddq_s16(step2[0], step2[15]);
  out[1] = vqaddq_s16(step2[1], step2[14]);
  out[2] = vqaddq_s16(step2[2], step2[13]);
  out[3] = vqaddq_s16(step2[3], step2[12]);
  out[4] = vqaddq_s16(step2[4], step2[11]);
  out[5] = vqaddq_s16(step2[5], step2[10]);
  out[6] = vqaddq_s16(step2[6], step2[9]);
  out[7] = vqaddq_s16(step2[7], step2[8]);
  out[8] = vqsubq_s16(step2[7], step2[8]);
  out[9] = vqsubq_s16(step2[6], step2[9]);
  out[10] = vqsubq_s16(step2[5], step2[10]);
  out[11] = vqsubq_s16(step2[4], step2[11]);
  out[12] = vqsubq_s16(step2[3], step2[12]);
  out[13] = vqsubq_s16(step2[2], step2[13]);
  out[14] = vqsubq_s16(step2[1], step2[14]);
  out[15] = vqsubq_s16(step2[0], step2[15]);
}

static inline void iadst16_neon(int16x8_t *const in, int16x8_t *out,
                                int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);

  const int16x4_t c0 = set_s16x4_neon((int16_t)cospi[2], (int16_t)cospi[62],
                                      (int16_t)cospi[10], (int16_t)cospi[54]);
  const int16x4_t c1 = set_s16x4_neon((int16_t)cospi[18], (int16_t)cospi[46],
                                      (int16_t)cospi[26], (int16_t)cospi[38]);
  const int16x4_t c2 = set_s16x4_neon((int16_t)cospi[34], (int16_t)cospi[30],
                                      (int16_t)cospi[42], (int16_t)cospi[22]);
  const int16x4_t c3 = set_s16x4_neon((int16_t)cospi[50], (int16_t)cospi[14],
                                      (int16_t)cospi[58], (int16_t)cospi[6]);
  const int16x4_t c4 = set_s16x4_neon((int16_t)cospi[8], (int16_t)cospi[56],
                                      (int16_t)cospi[40], (int16_t)cospi[24]);
  const int16x4_t c5 = set_s16x4_neon((int16_t)cospi[32], (int16_t)cospi[32],
                                      (int16_t)cospi[16], (int16_t)cospi[48]);

  int16x8_t x[16];
  int16x8_t t[14];
  int16x8_t s0, s1, s2, s3, s4, s5, s6, s7;
  int16x8_t s8, s9, s10, s11, s12, s13, s14, s15;

  // Stage 1
  x[0] = in[15];
  x[1] = in[0];
  x[2] = in[13];
  x[3] = in[2];
  x[4] = in[11];
  x[5] = in[4];
  x[6] = in[9];
  x[7] = in[6];
  x[8] = in[7];
  x[9] = in[8];
  x[10] = in[5];
  x[11] = in[10];
  x[12] = in[3];
  x[13] = in[12];
  x[14] = in[1];
  x[15] = in[14];

  // Stage 2
  btf_16_lane_0_1_neon(x[0], x[1], c0, &s0, &s1);
  btf_16_lane_2_3_neon(x[2], x[3], c0, &s2, &s3);
  btf_16_lane_0_1_neon(x[4], x[5], c1, &s4, &s5);
  btf_16_lane_2_3_neon(x[6], x[7], c1, &s6, &s7);
  btf_16_lane_0_1_neon(x[8], x[9], c2, &s8, &s9);
  btf_16_lane_2_3_neon(x[10], x[11], c2, &s10, &s11);
  btf_16_lane_0_1_neon(x[12], x[13], c3, &s12, &s13);
  btf_16_lane_2_3_neon(x[14], x[15], c3, &s14, &s15);

  // Stage 3
  x[0] = vqaddq_s16(s0, s8);
  x[1] = vqaddq_s16(s1, s9);
  x[2] = vqaddq_s16(s2, s10);
  x[3] = vqaddq_s16(s3, s11);
  x[4] = vqaddq_s16(s4, s12);
  x[5] = vqaddq_s16(s5, s13);
  x[6] = vqaddq_s16(s6, s14);
  x[7] = vqaddq_s16(s7, s15);
  x[8] = vqsubq_s16(s0, s8);
  x[9] = vqsubq_s16(s1, s9);
  x[10] = vqsubq_s16(s2, s10);
  x[11] = vqsubq_s16(s3, s11);
  x[12] = vqsubq_s16(s4, s12);
  x[13] = vqsubq_s16(s5, s13);
  x[14] = vqsubq_s16(s6, s14);
  x[15] = vqsubq_s16(s7, s15);

  // Stage 4
  t[0] = x[0];
  t[1] = x[1];
  t[2] = x[2];
  t[3] = x[3];
  t[4] = x[4];
  t[5] = x[5];
  t[6] = x[6];
  t[7] = x[7];
  btf_16_lane_0_1_neon(x[8], x[9], c4, &s8, &s9);
  btf_16_lane_2_3_neon(x[10], x[11], c4, &s10, &s11);
  btf_16_lane_1_0_neon(x[13], x[12], c4, &s13, &s12);
  btf_16_lane_3_2_neon(x[15], x[14], c4, &s15, &s14);

  // Stage 5
  x[0] = vqaddq_s16(t[0], t[4]);
  x[1] = vqaddq_s16(t[1], t[5]);
  x[2] = vqaddq_s16(t[2], t[6]);
  x[3] = vqaddq_s16(t[3], t[7]);
  x[4] = vqsubq_s16(t[0], t[4]);
  x[5] = vqsubq_s16(t[1], t[5]);
  x[6] = vqsubq_s16(t[2], t[6]);
  x[7] = vqsubq_s16(t[3], t[7]);
  x[8] = vqaddq_s16(s8, s12);
  x[9] = vqaddq_s16(s9, s13);
  x[10] = vqaddq_s16(s10, s14);
  x[11] = vqaddq_s16(s11, s15);
  x[12] = vqsubq_s16(s8, s12);
  x[13] = vqsubq_s16(s9, s13);
  x[14] = vqsubq_s16(s10, s14);
  x[15] = vqsubq_s16(s11, s15);

  // stage 6
  t[0] = x[0];
  t[1] = x[1];
  t[2] = x[2];
  t[3] = x[3];
  btf_16_lane_2_3_neon(x[4], x[5], c5, &s4, &s5);
  btf_16_lane_3_2_neon(x[7], x[6], c5, &s7, &s6);
  t[8] = x[8];
  t[9] = x[9];
  t[10] = x[10];
  t[11] = x[11];
  btf_16_lane_2_3_neon(x[12], x[13], c5, &s12, &s13);
  btf_16_lane_3_2_neon(x[15], x[14], c5, &s15, &s14);

  // Stage 7
  x[0] = vqaddq_s16(t[0], t[2]);
  x[1] = vqaddq_s16(t[1], t[3]);
  x[2] = vqsubq_s16(t[0], t[2]);
  x[3] = vqsubq_s16(t[1], t[3]);
  x[4] = vqaddq_s16(s4, s6);
  x[5] = vqaddq_s16(s5, s7);
  x[6] = vqsubq_s16(s4, s6);
  x[7] = vqsubq_s16(s5, s7);
  x[8] = vqaddq_s16(t[8], t[10]);
  x[9] = vqaddq_s16(t[9], t[11]);
  x[10] = vqsubq_s16(t[8], t[10]);
  x[11] = vqsubq_s16(t[9], t[11]);
  x[12] = vqaddq_s16(s12, s14);
  x[13] = vqaddq_s16(s13, s15);
  x[14] = vqsubq_s16(s12, s14);
  x[15] = vqsubq_s16(s13, s15);

  // Stage 8
  btf_16_half_neon(x + 2, c5);
  btf_16_half_neon(x + 6, c5);
  btf_16_half_neon(x + 10, c5);
  btf_16_half_neon(x + 14, c5);

  // Stage 9
  out[0] = x[0];
  out[1] = vqnegq_s16(x[8]);
  out[2] = x[12];
  out[3] = vqnegq_s16(x[4]);
  out[4] = x[6];
  out[5] = vqnegq_s16(x[14]);
  out[6] = x[10];
  out[7] = vqnegq_s16(x[2]);
  out[8] = x[3];
  out[9] = vqnegq_s16(x[11]);
  out[10] = x[15];
  out[11] = vqnegq_s16(x[7]);
  out[12] = x[5];
  out[13] = vqnegq_s16(x[13]);
  out[14] = x[9];
  out[15] = vqnegq_s16(x[1]);
}

static inline void iadst16_low1_neon(int16x8_t *const in, int16x8_t *out,
                                     int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);
  const int16x4_t c0 = set_s16x4_neon((int16_t)cospi[8], (int16_t)cospi[56],
                                      (int16_t)cospi[40], (int16_t)cospi[24]);
  const int16x4_t c1 = set_s16x4_neon((int16_t)cospi[32], (int16_t)cospi[32],
                                      (int16_t)cospi[16], (int16_t)cospi[48]);

  int16x8_t x[16];
  int16x8_t t[10];
  int16x8_t s0, s1, s4, s5;
  int16x8_t s8, s9, s12, s13;

  // Stage 1
  x[1] = in[0];

  // Stage 2
  btf_16_neon(x[1], cospi[62], -cospi[2], &s0, &s1);

  // Stage 3
  x[0] = s0;
  x[1] = s1;
  x[8] = s0;
  x[9] = s1;

  // Stage 4
  t[0] = x[0];
  t[1] = x[1];
  btf_16_lane_0_1_neon(x[8], x[9], c0, &s8, &s9);

  // Stage 5
  x[0] = t[0];
  x[1] = t[1];
  x[4] = t[0];
  x[5] = t[1];
  x[8] = s8;
  x[9] = s9;
  x[12] = s8;
  x[13] = s9;

  // stage 6
  t[0] = x[0];
  t[1] = x[1];
  btf_16_lane_2_3_neon(x[4], x[5], c1, &s4, &s5);
  t[8] = x[8];
  t[9] = x[9];
  btf_16_lane_2_3_neon(x[12], x[13], c1, &s12, &s13);

  // Stage 7
  x[0] = t[0];
  x[1] = t[1];
  x[2] = t[0];
  x[3] = t[1];
  x[4] = s4;
  x[5] = s5;
  x[6] = s4;
  x[7] = s5;
  x[8] = t[8];
  x[9] = t[9];
  x[10] = t[8];
  x[11] = t[9];
  x[12] = s12;
  x[13] = s13;
  x[14] = s12;
  x[15] = s13;

  // Stage 8
  btf_16_half_neon(x + 2, c1);
  btf_16_half_neon(x + 6, c1);
  btf_16_half_neon(x + 10, c1);
  btf_16_half_neon(x + 14, c1);

  // Stage 9
  out[0] = x[0];
  out[1] = vqnegq_s16(x[8]);
  out[2] = x[12];
  out[3] = vqnegq_s16(x[4]);
  out[4] = x[6];
  out[5] = vqnegq_s16(x[14]);
  out[6] = x[10];
  out[7] = vqnegq_s16(x[2]);
  out[8] = x[3];
  out[9] = vqnegq_s16(x[11]);
  out[10] = x[15];
  out[11] = vqnegq_s16(x[7]);
  out[12] = x[5];
  out[13] = vqnegq_s16(x[13]);
  out[14] = x[9];
  out[15] = vqnegq_s16(x[1]);
}

static inline void iadst16_low8_neon(int16x8_t *const in, int16x8_t *out,
                                     int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);

  const int16x4_t c0 = set_s16x4_neon((int16_t)cospi[8], (int16_t)cospi[56],
                                      (int16_t)cospi[40], (int16_t)cospi[24]);
  const int16x4_t c1 = set_s16x4_neon((int16_t)cospi[32], (int16_t)cospi[32],
                                      (int16_t)cospi[16], (int16_t)cospi[48]);

  int16x8_t x[16];
  int16x8_t t[14];
  int16x8_t s0, s1, s2, s3, s4, s5, s6, s7;
  int16x8_t s8, s9, s10, s11, s12, s13, s14, s15;

  // Stage 1
  x[1] = in[0];
  x[3] = in[2];
  x[5] = in[4];
  x[7] = in[6];
  x[8] = in[7];
  x[10] = in[5];
  x[12] = in[3];
  x[14] = in[1];

  // Stage 2
  btf_16_neon(x[1], cospi[62], -cospi[2], &s0, &s1);
  btf_16_neon(x[3], cospi[54], -cospi[10], &s2, &s3);
  btf_16_neon(x[5], cospi[46], -cospi[18], &s4, &s5);
  btf_16_neon(x[7], cospi[38], -cospi[26], &s6, &s7);

  btf_16_neon(x[8], cospi[34], cospi[30], &s8, &s9);
  btf_16_neon(x[10], cospi[42], cospi[22], &s10, &s11);
  btf_16_neon(x[12], cospi[50], cospi[14], &s12, &s13);
  btf_16_neon(x[14], cospi[58], cospi[6], &s14, &s15);

  // Stage 3
  x[0] = vqaddq_s16(s0, s8);
  x[1] = vqaddq_s16(s1, s9);
  x[2] = vqaddq_s16(s2, s10);
  x[3] = vqaddq_s16(s3, s11);
  x[4] = vqaddq_s16(s4, s12);
  x[5] = vqaddq_s16(s5, s13);
  x[6] = vqaddq_s16(s6, s14);
  x[7] = vqaddq_s16(s7, s15);
  x[8] = vqsubq_s16(s0, s8);
  x[9] = vqsubq_s16(s1, s9);
  x[10] = vqsubq_s16(s2, s10);
  x[11] = vqsubq_s16(s3, s11);
  x[12] = vqsubq_s16(s4, s12);
  x[13] = vqsubq_s16(s5, s13);
  x[14] = vqsubq_s16(s6, s14);
  x[15] = vqsubq_s16(s7, s15);

  // Stage 4
  t[0] = x[0];
  t[1] = x[1];
  t[2] = x[2];
  t[3] = x[3];
  t[4] = x[4];
  t[5] = x[5];
  t[6] = x[6];
  t[7] = x[7];
  btf_16_lane_0_1_neon(x[8], x[9], c0, &s8, &s9);
  btf_16_lane_2_3_neon(x[10], x[11], c0, &s10, &s11);
  btf_16_lane_1_0_neon(x[13], x[12], c0, &s13, &s12);
  btf_16_lane_3_2_neon(x[15], x[14], c0, &s15, &s14);

  // Stage 5
  x[0] = vqaddq_s16(t[0], t[4]);
  x[1] = vqaddq_s16(t[1], t[5]);
  x[2] = vqaddq_s16(t[2], t[6]);
  x[3] = vqaddq_s16(t[3], t[7]);
  x[4] = vqsubq_s16(t[0], t[4]);
  x[5] = vqsubq_s16(t[1], t[5]);
  x[6] = vqsubq_s16(t[2], t[6]);
  x[7] = vqsubq_s16(t[3], t[7]);
  x[8] = vqaddq_s16(s8, s12);
  x[9] = vqaddq_s16(s9, s13);
  x[10] = vqaddq_s16(s10, s14);
  x[11] = vqaddq_s16(s11, s15);
  x[12] = vqsubq_s16(s8, s12);
  x[13] = vqsubq_s16(s9, s13);
  x[14] = vqsubq_s16(s10, s14);
  x[15] = vqsubq_s16(s11, s15);

  // stage 6
  t[0] = x[0];
  t[1] = x[1];
  t[2] = x[2];
  t[3] = x[3];
  btf_16_lane_2_3_neon(x[4], x[5], c1, &s4, &s5);
  btf_16_lane_3_2_neon(x[7], x[6], c1, &s7, &s6);
  t[8] = x[8];
  t[9] = x[9];
  t[10] = x[10];
  t[11] = x[11];
  btf_16_lane_2_3_neon(x[12], x[13], c1, &s12, &s13);
  btf_16_lane_3_2_neon(x[15], x[14], c1, &s15, &s14);

  // Stage 7
  x[0] = vqaddq_s16(t[0], t[2]);
  x[1] = vqaddq_s16(t[1], t[3]);
  x[2] = vqsubq_s16(t[0], t[2]);
  x[3] = vqsubq_s16(t[1], t[3]);
  x[4] = vqaddq_s16(s4, s6);
  x[5] = vqaddq_s16(s5, s7);
  x[6] = vqsubq_s16(s4, s6);
  x[7] = vqsubq_s16(s5, s7);
  x[8] = vqaddq_s16(t[8], t[10]);
  x[9] = vqaddq_s16(t[9], t[11]);
  x[10] = vqsubq_s16(t[8], t[10]);
  x[11] = vqsubq_s16(t[9], t[11]);
  x[12] = vqaddq_s16(s12, s14);
  x[13] = vqaddq_s16(s13, s15);
  x[14] = vqsubq_s16(s12, s14);
  x[15] = vqsubq_s16(s13, s15);

  // Stage 8
  btf_16_half_neon(x + 2, c1);
  btf_16_half_neon(x + 6, c1);
  btf_16_half_neon(x + 10, c1);
  btf_16_half_neon(x + 14, c1);

  // Stage 9
  out[0] = x[0];
  out[1] = vqnegq_s16(x[8]);
  out[2] = x[12];
  out[3] = vqnegq_s16(x[4]);
  out[4] = x[6];
  out[5] = vqnegq_s16(x[14]);
  out[6] = x[10];
  out[7] = vqnegq_s16(x[2]);
  out[8] = x[3];
  out[9] = vqnegq_s16(x[11]);
  out[10] = x[15];
  out[11] = vqnegq_s16(x[7]);
  out[12] = x[5];
  out[13] = vqnegq_s16(x[13]);
  out[14] = x[9];
  out[15] = vqnegq_s16(x[1]);
}

static inline void idct32_neon(int16x8_t *in, int16x8_t *out, int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);
  int16x8_t step1[32], step2[32];

  const int16x4_t c0 = set_s16x4_neon((int16_t)cospi[2], (int16_t)cospi[62],
                                      (int16_t)cospi[34], (int16_t)cospi[30]);
  const int16x4_t c1 = set_s16x4_neon((int16_t)cospi[18], (int16_t)cospi[46],
                                      (int16_t)cospi[50], (int16_t)cospi[14]);
  const int16x4_t c2 = set_s16x4_neon((int16_t)cospi[10], (int16_t)cospi[54],
                                      (int16_t)cospi[42], (int16_t)cospi[22]);
  const int16x4_t c3 = set_s16x4_neon((int16_t)cospi[26], (int16_t)cospi[38],
                                      (int16_t)cospi[58], (int16_t)cospi[6]);
  const int16x4_t c4 = set_s16x4_neon((int16_t)cospi[4], (int16_t)cospi[60],
                                      (int16_t)cospi[36], (int16_t)cospi[28]);
  const int16x4_t c5 = set_s16x4_neon((int16_t)cospi[20], (int16_t)cospi[44],
                                      (int16_t)cospi[52], (int16_t)cospi[12]);
  const int16x4_t c6 = set_s16x4_neon((int16_t)cospi[8], (int16_t)cospi[56],
                                      (int16_t)cospi[40], (int16_t)cospi[24]);
  const int16x4_t c7 = set_s16x4_neon((int16_t)cospi[32], (int16_t)cospi[32],
                                      (int16_t)cospi[16], (int16_t)cospi[48]);
  const int16x4_t c8 =
      set_s16x4_neon((int16_t)(-cospi[8]), (int16_t)(-cospi[56]),
                     (int16_t)(-cospi[40]), (int16_t)(-cospi[24]));
  const int16x4_t c9 =
      set_s16x4_neon((int16_t)(-cospi[32]), (int16_t)(-cospi[32]),
                     (int16_t)(-cospi[16]), (int16_t)(-cospi[48]));

  // stage 2

  btf_16_lane_0_1_neon(in[1], in[31], c0, &step2[31], &step2[16]);
  btf_16_lane_2_3_neon(in[17], in[15], c0, &step2[30], &step2[17]);
  btf_16_lane_0_1_neon(in[9], in[23], c1, &step2[29], &step2[18]);
  btf_16_lane_2_3_neon(in[25], in[7], c1, &step2[28], &step2[19]);
  btf_16_lane_0_1_neon(in[5], in[27], c2, &step2[27], &step2[20]);
  btf_16_lane_2_3_neon(in[21], in[11], c2, &step2[26], &step2[21]);
  btf_16_lane_0_1_neon(in[13], in[19], c3, &step2[25], &step2[22]);
  btf_16_lane_2_3_neon(in[29], in[3], c3, &step2[24], &step2[23]);

  step2[0] = in[0];
  step2[1] = in[16];
  step2[2] = in[8];
  step2[3] = in[24];
  step2[4] = in[4];
  step2[5] = in[20];
  step2[6] = in[12];
  step2[7] = in[28];
  step2[8] = in[2];
  step2[9] = in[18];
  step2[10] = in[10];
  step2[11] = in[26];
  step2[12] = in[6];
  step2[13] = in[22];
  step2[14] = in[14];
  step2[15] = in[30];

  // stage 3

  btf_16_lane_0_1_neon(step2[8], step2[15], c4, &step1[15], &step1[8]);
  btf_16_lane_2_3_neon(step2[9], step2[14], c4, &step1[14], &step1[9]);
  btf_16_lane_0_1_neon(step2[10], step2[13], c5, &step1[13], &step1[10]);
  btf_16_lane_2_3_neon(step2[11], step2[12], c5, &step1[12], &step1[11]);

  step1[0] = step2[0];
  step1[1] = step2[1];
  step1[2] = step2[2];
  step1[3] = step2[3];
  step1[4] = step2[4];
  step1[5] = step2[5];
  step1[6] = step2[6];
  step1[7] = step2[7];

  step1[16] = vqaddq_s16(step2[16], step2[17]);
  step1[17] = vqsubq_s16(step2[16], step2[17]);
  step1[18] = vqsubq_s16(step2[19], step2[18]);
  step1[19] = vqaddq_s16(step2[19], step2[18]);
  step1[20] = vqaddq_s16(step2[20], step2[21]);
  step1[21] = vqsubq_s16(step2[20], step2[21]);
  step1[22] = vqsubq_s16(step2[23], step2[22]);
  step1[23] = vqaddq_s16(step2[23], step2[22]);
  step1[24] = vqaddq_s16(step2[24], step2[25]);
  step1[25] = vqsubq_s16(step2[24], step2[25]);
  step1[26] = vqsubq_s16(step2[27], step2[26]);
  step1[27] = vqaddq_s16(step2[27], step2[26]);
  step1[28] = vqaddq_s16(step2[28], step2[29]);
  step1[29] = vqsubq_s16(step2[28], step2[29]);
  step1[30] = vqsubq_s16(step2[31], step2[30]);
  step1[31] = vqaddq_s16(step2[31], step2[30]);

  // stage 4

  btf_16_lane_0_1_neon(step1[4], step1[7], c6, &step2[7], &step2[4]);
  btf_16_lane_2_3_neon(step1[5], step1[6], c6, &step2[6], &step2[5]);
  btf_16_lane_0_1_neon(step1[30], step1[17], c6, &step2[30], &step2[17]);
  btf_16_lane_1_0_neon(step1[18], step1[29], c8, &step2[18], &step2[29]);
  btf_16_lane_2_3_neon(step1[26], step1[21], c6, &step2[26], &step2[21]);
  btf_16_lane_3_2_neon(step1[22], step1[25], c8, &step2[22], &step2[25]);

  step2[0] = step1[0];
  step2[1] = step1[1];
  step2[2] = step1[2];
  step2[3] = step1[3];
  step2[8] = vqaddq_s16(step1[8], step1[9]);
  step2[9] = vqsubq_s16(step1[8], step1[9]);
  step2[10] = vqsubq_s16(step1[11], step1[10]);
  step2[11] = vqaddq_s16(step1[11], step1[10]);
  step2[12] = vqaddq_s16(step1[12], step1[13]);
  step2[13] = vqsubq_s16(step1[12], step1[13]);
  step2[14] = vqsubq_s16(step1[15], step1[14]);
  step2[15] = vqaddq_s16(step1[15], step1[14]);
  step2[16] = step1[16];
  step2[19] = step1[19];
  step2[20] = step1[20];
  step2[23] = step1[23];
  step2[24] = step1[24];
  step2[27] = step1[27];
  step2[28] = step1[28];
  step2[31] = step1[31];

  // stage 5

  btf_16_lane_0_1_neon(step2[0], step2[1], c7, &step1[0], &step1[1]);
  btf_16_lane_2_3_neon(step2[2], step2[3], c7, &step1[3], &step1[2]);
  btf_16_lane_2_3_neon(step2[14], step2[9], c7, &step1[14], &step1[9]);
  btf_16_lane_3_2_neon(step2[10], step2[13], c9, &step1[10], &step1[13]);

  step1[4] = vqaddq_s16(step2[4], step2[5]);
  step1[5] = vqsubq_s16(step2[4], step2[5]);
  step1[6] = vqsubq_s16(step2[7], step2[6]);
  step1[7] = vqaddq_s16(step2[7], step2[6]);
  step1[8] = step2[8];
  step1[11] = step2[11];
  step1[12] = step2[12];
  step1[15] = step2[15];
  step1[16] = vqaddq_s16(step2[16], step2[19]);
  step1[17] = vqaddq_s16(step2[17], step2[18]);
  step1[18] = vqsubq_s16(step2[17], step2[18]);
  step1[19] = vqsubq_s16(step2[16], step2[19]);
  step1[20] = vqsubq_s16(step2[23], step2[20]);
  step1[21] = vqsubq_s16(step2[22], step2[21]);
  step1[22] = vqaddq_s16(step2[22], step2[21]);
  step1[23] = vqaddq_s16(step2[23], step2[20]);
  step1[24] = vqaddq_s16(step2[24], step2[27]);
  step1[25] = vqaddq_s16(step2[25], step2[26]);
  step1[26] = vqsubq_s16(step2[25], step2[26]);
  step1[27] = vqsubq_s16(step2[24], step2[27]);
  step1[28] = vqsubq_s16(step2[31], step2[28]);
  step1[29] = vqsubq_s16(step2[30], step2[29]);
  step1[30] = vqaddq_s16(step2[30], step2[29]);
  step1[31] = vqaddq_s16(step2[31], step2[28]);

  // stage 6

  btf_16_lane_0_1_neon(step1[6], step1[5], c7, &step2[6], &step2[5]);
  btf_16_lane_2_3_neon(step1[29], step1[18], c7, &step2[29], &step2[18]);
  btf_16_lane_2_3_neon(step1[28], step1[19], c7, &step2[28], &step2[19]);
  btf_16_lane_3_2_neon(step1[20], step1[27], c9, &step2[20], &step2[27]);
  btf_16_lane_3_2_neon(step1[21], step1[26], c9, &step2[21], &step2[26]);

  step2[0] = vqaddq_s16(step1[0], step1[3]);
  step2[1] = vqaddq_s16(step1[1], step1[2]);
  step2[2] = vqsubq_s16(step1[1], step1[2]);
  step2[3] = vqsubq_s16(step1[0], step1[3]);
  step2[4] = step1[4];
  step2[7] = step1[7];
  step2[8] = vqaddq_s16(step1[8], step1[11]);
  step2[9] = vqaddq_s16(step1[9], step1[10]);
  step2[10] = vqsubq_s16(step1[9], step1[10]);
  step2[11] = vqsubq_s16(step1[8], step1[11]);
  step2[12] = vqsubq_s16(step1[15], step1[12]);
  step2[13] = vqsubq_s16(step1[14], step1[13]);
  step2[14] = vqaddq_s16(step1[14], step1[13]);
  step2[15] = vqaddq_s16(step1[15], step1[12]);
  step2[16] = step1[16];
  step2[17] = step1[17];
  step2[22] = step1[22];
  step2[23] = step1[23];
  step2[24] = step1[24];
  step2[25] = step1[25];
  step2[30] = step1[30];
  step2[31] = step1[31];

  // stage 7

  btf_16_lane_0_1_neon(step2[13], step2[10], c7, &step1[13], &step1[10]);
  btf_16_lane_0_1_neon(step2[12], step2[11], c7, &step1[12], &step1[11]);

  step1[0] = vqaddq_s16(step2[0], step2[7]);
  step1[1] = vqaddq_s16(step2[1], step2[6]);
  step1[2] = vqaddq_s16(step2[2], step2[5]);
  step1[3] = vqaddq_s16(step2[3], step2[4]);
  step1[4] = vqsubq_s16(step2[3], step2[4]);
  step1[5] = vqsubq_s16(step2[2], step2[5]);
  step1[6] = vqsubq_s16(step2[1], step2[6]);
  step1[7] = vqsubq_s16(step2[0], step2[7]);
  step1[8] = step2[8];
  step1[9] = step2[9];
  step1[14] = step2[14];
  step1[15] = step2[15];
  step1[16] = vqaddq_s16(step2[16], step2[23]);
  step1[17] = vqaddq_s16(step2[17], step2[22]);
  step1[18] = vqaddq_s16(step2[18], step2[21]);
  step1[19] = vqaddq_s16(step2[19], step2[20]);
  step1[20] = vqsubq_s16(step2[19], step2[20]);
  step1[21] = vqsubq_s16(step2[18], step2[21]);
  step1[22] = vqsubq_s16(step2[17], step2[22]);
  step1[23] = vqsubq_s16(step2[16], step2[23]);
  step1[24] = vqsubq_s16(step2[31], step2[24]);
  step1[25] = vqsubq_s16(step2[30], step2[25]);
  step1[26] = vqsubq_s16(step2[29], step2[26]);
  step1[27] = vqsubq_s16(step2[28], step2[27]);
  step1[28] = vqaddq_s16(step2[27], step2[28]);
  step1[29] = vqaddq_s16(step2[26], step2[29]);
  step1[30] = vqaddq_s16(step2[25], step2[30]);
  step1[31] = vqaddq_s16(step2[24], step2[31]);

  // stage 8

  btf_16_lane_0_1_neon(step1[27], step1[20], c7, &step2[27], &step2[20]);
  btf_16_lane_0_1_neon(step1[26], step1[21], c7, &step2[26], &step2[21]);
  btf_16_lane_0_1_neon(step1[25], step1[22], c7, &step2[25], &step2[22]);
  btf_16_lane_0_1_neon(step1[24], step1[23], c7, &step2[24], &step2[23]);

  step2[0] = vqaddq_s16(step1[0], step1[15]);
  step2[1] = vqaddq_s16(step1[1], step1[14]);
  step2[2] = vqaddq_s16(step1[2], step1[13]);
  step2[3] = vqaddq_s16(step1[3], step1[12]);
  step2[4] = vqaddq_s16(step1[4], step1[11]);
  step2[5] = vqaddq_s16(step1[5], step1[10]);
  step2[6] = vqaddq_s16(step1[6], step1[9]);
  step2[7] = vqaddq_s16(step1[7], step1[8]);
  step2[8] = vqsubq_s16(step1[7], step1[8]);
  step2[9] = vqsubq_s16(step1[6], step1[9]);
  step2[10] = vqsubq_s16(step1[5], step1[10]);
  step2[11] = vqsubq_s16(step1[4], step1[11]);
  step2[12] = vqsubq_s16(step1[3], step1[12]);
  step2[13] = vqsubq_s16(step1[2], step1[13]);
  step2[14] = vqsubq_s16(step1[1], step1[14]);
  step2[15] = vqsubq_s16(step1[0], step1[15]);
  step2[16] = step1[16];
  step2[17] = step1[17];
  step2[18] = step1[18];
  step2[19] = step1[19];
  step2[28] = step1[28];
  step2[29] = step1[29];
  step2[30] = step1[30];
  step2[31] = step1[31];

  // stage 9

  out[0] = vqaddq_s16(step2[0], step2[31]);
  out[1] = vqaddq_s16(step2[1], step2[30]);
  out[2] = vqaddq_s16(step2[2], step2[29]);
  out[3] = vqaddq_s16(step2[3], step2[28]);
  out[4] = vqaddq_s16(step2[4], step2[27]);
  out[5] = vqaddq_s16(step2[5], step2[26]);
  out[6] = vqaddq_s16(step2[6], step2[25]);
  out[7] = vqaddq_s16(step2[7], step2[24]);
  out[8] = vqaddq_s16(step2[8], step2[23]);
  out[9] = vqaddq_s16(step2[9], step2[22]);
  out[10] = vqaddq_s16(step2[10], step2[21]);
  out[11] = vqaddq_s16(step2[11], step2[20]);
  out[12] = vqaddq_s16(step2[12], step2[19]);
  out[13] = vqaddq_s16(step2[13], step2[18]);
  out[14] = vqaddq_s16(step2[14], step2[17]);
  out[15] = vqaddq_s16(step2[15], step2[16]);
  out[16] = vqsubq_s16(step2[15], step2[16]);
  out[17] = vqsubq_s16(step2[14], step2[17]);
  out[18] = vqsubq_s16(step2[13], step2[18]);
  out[19] = vqsubq_s16(step2[12], step2[19]);
  out[20] = vqsubq_s16(step2[11], step2[20]);
  out[21] = vqsubq_s16(step2[10], step2[21]);
  out[22] = vqsubq_s16(step2[9], step2[22]);
  out[23] = vqsubq_s16(step2[8], step2[23]);
  out[24] = vqsubq_s16(step2[7], step2[24]);
  out[25] = vqsubq_s16(step2[6], step2[25]);
  out[26] = vqsubq_s16(step2[5], step2[26]);
  out[27] = vqsubq_s16(step2[4], step2[27]);
  out[28] = vqsubq_s16(step2[3], step2[28]);
  out[29] = vqsubq_s16(step2[2], step2[29]);
  out[30] = vqsubq_s16(step2[1], step2[30]);
  out[31] = vqsubq_s16(step2[0], step2[31]);
}

static inline void idct32_low1_neon(int16x8_t *in, int16x8_t *out,
                                    int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);
  int16x8_t step1;
  int32x4_t t32[2];

  // stage 1
  // stage 2
  // stage 3
  // stage 4
  // stage 5

  t32[0] = vmull_n_s16(vget_low_s16(in[0]), cospi[32]);
  t32[1] = vmull_n_s16(vget_high_s16(in[0]), cospi[32]);
  step1 = vcombine_s16(vrshrn_n_s32(t32[0], INV_COS_BIT),
                       vrshrn_n_s32(t32[1], INV_COS_BIT));

  // stage 6
  // stage 7
  // stage 8
  // stage 9

  out[0] = step1;
  out[1] = step1;
  out[2] = step1;
  out[3] = step1;
  out[4] = step1;
  out[5] = step1;
  out[6] = step1;
  out[7] = step1;
  out[8] = step1;
  out[9] = step1;
  out[10] = step1;
  out[11] = step1;
  out[12] = step1;
  out[13] = step1;
  out[14] = step1;
  out[15] = step1;
  out[16] = step1;
  out[17] = step1;
  out[18] = step1;
  out[19] = step1;
  out[20] = step1;
  out[21] = step1;
  out[22] = step1;
  out[23] = step1;
  out[24] = step1;
  out[25] = step1;
  out[26] = step1;
  out[27] = step1;
  out[28] = step1;
  out[29] = step1;
  out[30] = step1;
  out[31] = step1;
}

static inline void idct32_low8_neon(int16x8_t *in, int16x8_t *out,
                                    int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);
  int16x8_t step1[32], step2[32];
  int32x4_t t32[16];
  const int16x4_t c0 = set_s16x4_neon((int16_t)cospi[8], (int16_t)cospi[56],
                                      (int16_t)cospi[40], (int16_t)cospi[24]);
  const int16x4_t c1 = set_s16x4_neon((int16_t)cospi[32], (int16_t)cospi[32],
                                      (int16_t)cospi[16], cospi[48]);
  const int16x4_t c2 =
      set_s16x4_neon((int16_t)(-cospi[8]), (int16_t)(-cospi[56]),
                     (int16_t)(-cospi[40]), (int16_t)(-cospi[24]));
  const int16x4_t c3 =
      set_s16x4_neon((int16_t)(-cospi[32]), (int16_t)(-cospi[32]),
                     (int16_t)(-cospi[16]), (int16_t)(-cospi[48]));
  // stage 1
  // stage 2

  step2[0] = in[0];
  step2[4] = in[4];
  step2[8] = in[2];
  step2[12] = in[6];

  btf_16_neon(in[1], cospi[62], cospi[2], &step2[16], &step2[31]);
  btf_16_neon(in[7], -cospi[50], cospi[14], &step2[19], &step2[28]);
  btf_16_neon(in[5], cospi[54], cospi[10], &step2[20], &step2[27]);
  btf_16_neon(in[3], -cospi[58], cospi[6], &step2[23], &step2[24]);

  // stage 3
  step1[0] = step2[0];
  step1[4] = step2[4];

  btf_16_neon(step2[8], cospi[60], cospi[4], &step1[8], &step1[15]);
  btf_16_neon(step2[12], -cospi[52], cospi[12], &step1[11], &step1[12]);

  step1[16] = step2[16];
  step1[17] = step2[16];
  step1[18] = step2[19];
  step1[19] = step2[19];
  step1[20] = step2[20];
  step1[21] = step2[20];
  step1[22] = step2[23];
  step1[23] = step2[23];
  step1[24] = step2[24];
  step1[25] = step2[24];
  step1[26] = step2[27];
  step1[27] = step2[27];
  step1[28] = step2[28];
  step1[29] = step2[28];
  step1[30] = step2[31];
  step1[31] = step2[31];

  // stage 4

  btf_16_neon(step1[4], cospi[56], cospi[8], &step2[4], &step2[7]);
  btf_16_lane_0_1_neon(step1[30], step1[17], c0, &step2[30], &step2[17]);
  btf_16_lane_1_0_neon(step1[18], step1[29], c2, &step2[18], &step2[29]);
  btf_16_lane_2_3_neon(step1[26], step1[21], c0, &step2[26], &step2[21]);
  btf_16_lane_3_2_neon(step1[22], step1[25], c2, &step2[22], &step2[25]);

  step2[0] = step1[0];
  step2[8] = step1[8];
  step2[9] = step1[8];
  step2[10] = step1[11];
  step2[11] = step1[11];
  step2[12] = step1[12];
  step2[13] = step1[12];
  step2[14] = step1[15];
  step2[15] = step1[15];
  step2[16] = step1[16];
  step2[19] = step1[19];
  step2[20] = step1[20];
  step2[23] = step1[23];
  step2[24] = step1[24];
  step2[27] = step1[27];
  step2[28] = step1[28];
  step2[31] = step1[31];

  // stage 5

  t32[0] = vmull_n_s16(vget_low_s16(step2[0]), cospi[32]);
  t32[1] = vmull_n_s16(vget_high_s16(step2[0]), cospi[32]);
  step1[0] = vcombine_s16(vrshrn_n_s32(t32[0], INV_COS_BIT),
                          vrshrn_n_s32(t32[1], INV_COS_BIT));

  btf_16_lane_2_3_neon(step2[14], step2[9], c1, &step1[14], &step1[9]);
  btf_16_lane_3_2_neon(step2[10], step2[13], c3, &step1[10], &step1[13]);

  step1[4] = step2[4];
  step1[5] = step2[4];
  step1[6] = step2[7];
  step1[7] = step2[7];
  step1[8] = step2[8];
  step1[11] = step2[11];
  step1[12] = step2[12];
  step1[15] = step2[15];
  step1[16] = vqaddq_s16(step2[16], step2[19]);
  step1[17] = vqaddq_s16(step2[17], step2[18]);
  step1[18] = vqsubq_s16(step2[17], step2[18]);
  step1[19] = vqsubq_s16(step2[16], step2[19]);
  step1[20] = vqsubq_s16(step2[23], step2[20]);
  step1[21] = vqsubq_s16(step2[22], step2[21]);
  step1[22] = vqaddq_s16(step2[22], step2[21]);
  step1[23] = vqaddq_s16(step2[23], step2[20]);
  step1[24] = vqaddq_s16(step2[24], step2[27]);
  step1[25] = vqaddq_s16(step2[25], step2[26]);
  step1[26] = vqsubq_s16(step2[25], step2[26]);
  step1[27] = vqsubq_s16(step2[24], step2[27]);
  step1[28] = vqsubq_s16(step2[31], step2[28]);
  step1[29] = vqsubq_s16(step2[30], step2[29]);
  step1[30] = vqaddq_s16(step2[30], step2[29]);
  step1[31] = vqaddq_s16(step2[31], step2[28]);

  // stage 6

  btf_16_lane_0_1_neon(step1[6], step1[5], c1, &step2[6], &step2[5]);
  btf_16_lane_2_3_neon(step1[29], step1[18], c1, &step2[29], &step2[18]);
  btf_16_lane_2_3_neon(step1[28], step1[19], c1, &step2[28], &step2[19]);
  btf_16_lane_3_2_neon(step1[20], step1[27], c3, &step2[20], &step2[27]);
  btf_16_lane_3_2_neon(step1[21], step1[26], c3, &step2[21], &step2[26]);

  step2[0] = step1[0];
  step2[1] = step1[0];
  step2[2] = step1[0];
  step2[3] = step1[0];
  step2[4] = step1[4];
  step2[7] = step1[7];
  step2[8] = vqaddq_s16(step1[8], step1[11]);
  step2[9] = vqaddq_s16(step1[9], step1[10]);
  step2[10] = vqsubq_s16(step1[9], step1[10]);
  step2[11] = vqsubq_s16(step1[8], step1[11]);
  step2[12] = vqsubq_s16(step1[15], step1[12]);
  step2[13] = vqsubq_s16(step1[14], step1[13]);
  step2[14] = vqaddq_s16(step1[14], step1[13]);
  step2[15] = vqaddq_s16(step1[15], step1[12]);
  step2[16] = step1[16];
  step2[17] = step1[17];
  step2[22] = step1[22];
  step2[23] = step1[23];
  step2[24] = step1[24];
  step2[25] = step1[25];
  step2[30] = step1[30];
  step2[31] = step1[31];

  // stage 7

  btf_16_lane_0_1_neon(step2[13], step2[10], c1, &step1[13], &step1[10]);
  btf_16_lane_0_1_neon(step2[12], step2[11], c1, &step1[12], &step1[11]);

  step1[0] = vqaddq_s16(step2[0], step2[7]);
  step1[1] = vqaddq_s16(step2[1], step2[6]);
  step1[2] = vqaddq_s16(step2[2], step2[5]);
  step1[3] = vqaddq_s16(step2[3], step2[4]);
  step1[4] = vqsubq_s16(step2[3], step2[4]);
  step1[5] = vqsubq_s16(step2[2], step2[5]);
  step1[6] = vqsubq_s16(step2[1], step2[6]);
  step1[7] = vqsubq_s16(step2[0], step2[7]);
  step1[8] = step2[8];
  step1[9] = step2[9];
  step1[14] = step2[14];
  step1[15] = step2[15];
  step1[16] = vqaddq_s16(step2[16], step2[23]);
  step1[17] = vqaddq_s16(step2[17], step2[22]);
  step1[18] = vqaddq_s16(step2[18], step2[21]);
  step1[19] = vqaddq_s16(step2[19], step2[20]);
  step1[20] = vqsubq_s16(step2[19], step2[20]);
  step1[21] = vqsubq_s16(step2[18], step2[21]);
  step1[22] = vqsubq_s16(step2[17], step2[22]);
  step1[23] = vqsubq_s16(step2[16], step2[23]);
  step1[24] = vqsubq_s16(step2[31], step2[24]);
  step1[25] = vqsubq_s16(step2[30], step2[25]);
  step1[26] = vqsubq_s16(step2[29], step2[26]);
  step1[27] = vqsubq_s16(step2[28], step2[27]);
  step1[28] = vqaddq_s16(step2[27], step2[28]);
  step1[29] = vqaddq_s16(step2[26], step2[29]);
  step1[30] = vqaddq_s16(step2[25], step2[30]);
  step1[31] = vqaddq_s16(step2[24], step2[31]);

  // stage 8

  btf_16_lane_0_1_neon(step1[27], step1[20], c1, &step2[27], &step2[20]);
  btf_16_lane_0_1_neon(step1[26], step1[21], c1, &step2[26], &step2[21]);
  btf_16_lane_0_1_neon(step1[25], step1[22], c1, &step2[25], &step2[22]);
  btf_16_lane_0_1_neon(step1[24], step1[23], c1, &step2[24], &step2[23]);

  step2[0] = vqaddq_s16(step1[0], step1[15]);
  step2[1] = vqaddq_s16(step1[1], step1[14]);
  step2[2] = vqaddq_s16(step1[2], step1[13]);
  step2[3] = vqaddq_s16(step1[3], step1[12]);
  step2[4] = vqaddq_s16(step1[4], step1[11]);
  step2[5] = vqaddq_s16(step1[5], step1[10]);
  step2[6] = vqaddq_s16(step1[6], step1[9]);
  step2[7] = vqaddq_s16(step1[7], step1[8]);
  step2[8] = vqsubq_s16(step1[7], step1[8]);
  step2[9] = vqsubq_s16(step1[6], step1[9]);
  step2[10] = vqsubq_s16(step1[5], step1[10]);
  step2[11] = vqsubq_s16(step1[4], step1[11]);
  step2[12] = vqsubq_s16(step1[3], step1[12]);
  step2[13] = vqsubq_s16(step1[2], step1[13]);
  step2[14] = vqsubq_s16(step1[1], step1[14]);
  step2[15] = vqsubq_s16(step1[0], step1[15]);
  step2[16] = step1[16];
  step2[17] = step1[17];
  step2[18] = step1[18];
  step2[19] = step1[19];
  step2[28] = step1[28];
  step2[29] = step1[29];
  step2[30] = step1[30];
  step2[31] = step1[31];

  // stage 9

  out[0] = vqaddq_s16(step2[0], step2[31]);
  out[1] = vqaddq_s16(step2[1], step2[30]);
  out[2] = vqaddq_s16(step2[2], step2[29]);
  out[3] = vqaddq_s16(step2[3], step2[28]);
  out[4] = vqaddq_s16(step2[4], step2[27]);
  out[5] = vqaddq_s16(step2[5], step2[26]);
  out[6] = vqaddq_s16(step2[6], step2[25]);
  out[7] = vqaddq_s16(step2[7], step2[24]);
  out[8] = vqaddq_s16(step2[8], step2[23]);
  out[9] = vqaddq_s16(step2[9], step2[22]);
  out[10] = vqaddq_s16(step2[10], step2[21]);
  out[11] = vqaddq_s16(step2[11], step2[20]);
  out[12] = vqaddq_s16(step2[12], step2[19]);
  out[13] = vqaddq_s16(step2[13], step2[18]);
  out[14] = vqaddq_s16(step2[14], step2[17]);
  out[15] = vqaddq_s16(step2[15], step2[16]);
  out[16] = vqsubq_s16(step2[15], step2[16]);
  out[17] = vqsubq_s16(step2[14], step2[17]);
  out[18] = vqsubq_s16(step2[13], step2[18]);
  out[19] = vqsubq_s16(step2[12], step2[19]);
  out[20] = vqsubq_s16(step2[11], step2[20]);
  out[21] = vqsubq_s16(step2[10], step2[21]);
  out[22] = vqsubq_s16(step2[9], step2[22]);
  out[23] = vqsubq_s16(step2[8], step2[23]);
  out[24] = vqsubq_s16(step2[7], step2[24]);
  out[25] = vqsubq_s16(step2[6], step2[25]);
  out[26] = vqsubq_s16(step2[5], step2[26]);
  out[27] = vqsubq_s16(step2[4], step2[27]);
  out[28] = vqsubq_s16(step2[3], step2[28]);
  out[29] = vqsubq_s16(step2[2], step2[29]);
  out[30] = vqsubq_s16(step2[1], step2[30]);
  out[31] = vqsubq_s16(step2[0], step2[31]);
}

static inline void idct32_low16_neon(int16x8_t *in, int16x8_t *out,
                                     int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);
  int16x8_t step1[32], step2[32];
  int32x4_t t32[16];
  const int16x4_t c0 = set_s16x4_neon((int16_t)cospi[8], (int16_t)cospi[56],
                                      (int16_t)cospi[40], (int16_t)cospi[24]);
  const int16x4_t c1 = set_s16x4_neon((int16_t)cospi[32], (int16_t)cospi[32],
                                      (int16_t)cospi[16], (int16_t)cospi[48]);
  const int16x4_t c2 =
      set_s16x4_neon((int16_t)(-cospi[8]), (int16_t)(-cospi[56]),
                     (int16_t)(-cospi[40]), (int16_t)(-cospi[24]));
  const int16x4_t c3 =
      set_s16x4_neon((int16_t)(-cospi[32]), (int16_t)(-cospi[32]),
                     (int16_t)(-cospi[16]), (int16_t)(-cospi[48]));

  // stage 1
  // stage 2

  btf_16_neon(in[1], cospi[62], cospi[2], &step2[16], &step2[31]);
  btf_16_neon(in[15], -cospi[34], cospi[30], &step2[17], &step2[30]);
  btf_16_neon(in[9], cospi[46], cospi[18], &step2[18], &step2[29]);
  btf_16_neon(in[7], -cospi[50], cospi[14], &step2[19], &step2[28]);
  btf_16_neon(in[5], cospi[54], cospi[10], &step2[20], &step2[27]);
  btf_16_neon(in[11], -cospi[42], cospi[22], &step2[21], &step2[26]);
  btf_16_neon(in[13], cospi[38], cospi[26], &step2[22], &step2[25]);
  btf_16_neon(in[3], -cospi[58], cospi[6], &step2[23], &step2[24]);

  step2[0] = in[0];
  step2[2] = in[8];
  step2[4] = in[4];
  step2[6] = in[12];
  step2[8] = in[2];
  step2[10] = in[10];
  step2[12] = in[6];
  step2[14] = in[14];

  // stage 3

  btf_16_neon(step2[8], cospi[60], cospi[4], &step1[8], &step1[15]);
  btf_16_neon(step2[14], -cospi[36], cospi[28], &step1[9], &step1[14]);
  btf_16_neon(step2[10], cospi[44], cospi[20], &step1[10], &step1[13]);
  btf_16_neon(step2[12], -cospi[52], cospi[12], &step1[11], &step1[12]);

  step1[0] = step2[0];
  step1[2] = step2[2];
  step1[4] = step2[4];
  step1[6] = step2[6];
  step1[16] = vqaddq_s16(step2[16], step2[17]);
  step1[17] = vqsubq_s16(step2[16], step2[17]);
  step1[18] = vqsubq_s16(step2[19], step2[18]);
  step1[19] = vqaddq_s16(step2[19], step2[18]);
  step1[20] = vqaddq_s16(step2[20], step2[21]);
  step1[21] = vqsubq_s16(step2[20], step2[21]);
  step1[22] = vqsubq_s16(step2[23], step2[22]);
  step1[23] = vqaddq_s16(step2[23], step2[22]);
  step1[24] = vqaddq_s16(step2[24], step2[25]);
  step1[25] = vqsubq_s16(step2[24], step2[25]);
  step1[26] = vqsubq_s16(step2[27], step2[26]);
  step1[27] = vqaddq_s16(step2[27], step2[26]);
  step1[28] = vqaddq_s16(step2[28], step2[29]);
  step1[29] = vqsubq_s16(step2[28], step2[29]);
  step1[30] = vqsubq_s16(step2[31], step2[30]);
  step1[31] = vqaddq_s16(step2[31], step2[30]);

  // stage 4

  btf_16_neon(step1[4], cospi[56], cospi[8], &step2[4], &step2[7]);
  btf_16_neon(step1[6], -cospi[40], cospi[24], &step2[5], &step2[6]);
  btf_16_lane_0_1_neon(step1[30], step1[17], c0, &step2[30], &step2[17]);
  btf_16_lane_1_0_neon(step1[18], step1[29], c2, &step2[18], &step2[29]);
  btf_16_lane_2_3_neon(step1[26], step1[21], c0, &step2[26], &step2[21]);
  btf_16_lane_3_2_neon(step1[22], step1[25], c2, &step2[22], &step2[25]);

  step2[0] = step1[0];
  step2[2] = step1[2];
  step2[8] = vqaddq_s16(step1[8], step1[9]);
  step2[9] = vqsubq_s16(step1[8], step1[9]);
  step2[10] = vqsubq_s16(step1[11], step1[10]);
  step2[11] = vqaddq_s16(step1[11], step1[10]);
  step2[12] = vqaddq_s16(step1[12], step1[13]);
  step2[13] = vqsubq_s16(step1[12], step1[13]);
  step2[14] = vqsubq_s16(step1[15], step1[14]);
  step2[15] = vqaddq_s16(step1[15], step1[14]);
  step2[16] = step1[16];
  step2[19] = step1[19];
  step2[20] = step1[20];
  step2[23] = step1[23];
  step2[24] = step1[24];
  step2[27] = step1[27];
  step2[28] = step1[28];
  step2[31] = step1[31];

  // stage 5

  t32[0] = vmull_n_s16(vget_low_s16(step2[0]), cospi[32]);
  t32[1] = vmull_n_s16(vget_high_s16(step2[0]), cospi[32]);

  step1[0] = vcombine_s16(vrshrn_n_s32(t32[0], INV_COS_BIT),
                          vrshrn_n_s32(t32[1], INV_COS_BIT));

  btf_16_neon(step2[2], cospi[48], cospi[16], &step1[2], &step1[3]);
  btf_16_lane_2_3_neon(step2[14], step2[9], c1, &step1[14], &step1[9]);
  btf_16_lane_3_2_neon(step2[10], step2[13], c3, &step1[10], &step1[13]);

  step1[4] = vqaddq_s16(step2[4], step2[5]);
  step1[5] = vqsubq_s16(step2[4], step2[5]);
  step1[6] = vqsubq_s16(step2[7], step2[6]);
  step1[7] = vqaddq_s16(step2[7], step2[6]);
  step1[8] = step2[8];
  step1[11] = step2[11];
  step1[12] = step2[12];
  step1[15] = step2[15];
  step1[16] = vqaddq_s16(step2[16], step2[19]);
  step1[17] = vqaddq_s16(step2[17], step2[18]);
  step1[18] = vqsubq_s16(step2[17], step2[18]);
  step1[19] = vqsubq_s16(step2[16], step2[19]);
  step1[20] = vqsubq_s16(step2[23], step2[20]);
  step1[21] = vqsubq_s16(step2[22], step2[21]);
  step1[22] = vqaddq_s16(step2[22], step2[21]);
  step1[23] = vqaddq_s16(step2[23], step2[20]);
  step1[24] = vqaddq_s16(step2[24], step2[27]);
  step1[25] = vqaddq_s16(step2[25], step2[26]);
  step1[26] = vqsubq_s16(step2[25], step2[26]);
  step1[27] = vqsubq_s16(step2[24], step2[27]);
  step1[28] = vqsubq_s16(step2[31], step2[28]);
  step1[29] = vqsubq_s16(step2[30], step2[29]);
  step1[30] = vqaddq_s16(step2[30], step2[29]);
  step1[31] = vqaddq_s16(step2[31], step2[28]);

  // stage 6

  btf_16_lane_0_1_neon(step1[6], step1[5], c1, &step2[6], &step2[5]);
  btf_16_lane_2_3_neon(step1[29], step1[18], c1, &step2[29], &step2[18]);
  btf_16_lane_2_3_neon(step1[28], step1[19], c1, &step2[28], &step2[19]);
  btf_16_lane_3_2_neon(step1[20], step1[27], c3, &step2[20], &step2[27]);
  btf_16_lane_3_2_neon(step1[21], step1[26], c3, &step2[21], &step2[26]);

  step2[0] = vqaddq_s16(step1[0], step1[3]);
  step2[1] = vqaddq_s16(step1[0], step1[2]);
  step2[2] = vqsubq_s16(step1[0], step1[2]);
  step2[3] = vqsubq_s16(step1[0], step1[3]);
  step2[4] = step1[4];
  step2[7] = step1[7];
  step2[8] = vqaddq_s16(step1[8], step1[11]);
  step2[9] = vqaddq_s16(step1[9], step1[10]);
  step2[10] = vqsubq_s16(step1[9], step1[10]);
  step2[11] = vqsubq_s16(step1[8], step1[11]);
  step2[12] = vqsubq_s16(step1[15], step1[12]);
  step2[13] = vqsubq_s16(step1[14], step1[13]);
  step2[14] = vqaddq_s16(step1[14], step1[13]);
  step2[15] = vqaddq_s16(step1[15], step1[12]);
  step2[16] = step1[16];
  step2[17] = step1[17];
  step2[22] = step1[22];
  step2[23] = step1[23];
  step2[24] = step1[24];
  step2[25] = step1[25];
  step2[30] = step1[30];
  step2[31] = step1[31];

  // stage 7

  btf_16_lane_0_1_neon(step2[13], step2[10], c1, &step1[13], &step1[10]);
  btf_16_lane_0_1_neon(step2[12], step2[11], c1, &step1[12], &step1[11]);

  step1[0] = vqaddq_s16(step2[0], step2[7]);
  step1[1] = vqaddq_s16(step2[1], step2[6]);
  step1[2] = vqaddq_s16(step2[2], step2[5]);
  step1[3] = vqaddq_s16(step2[3], step2[4]);
  step1[4] = vqsubq_s16(step2[3], step2[4]);
  step1[5] = vqsubq_s16(step2[2], step2[5]);
  step1[6] = vqsubq_s16(step2[1], step2[6]);
  step1[7] = vqsubq_s16(step2[0], step2[7]);
  step1[8] = step2[8];
  step1[9] = step2[9];
  step1[14] = step2[14];
  step1[15] = step2[15];
  step1[16] = vqaddq_s16(step2[16], step2[23]);
  step1[17] = vqaddq_s16(step2[17], step2[22]);
  step1[18] = vqaddq_s16(step2[18], step2[21]);
  step1[19] = vqaddq_s16(step2[19], step2[20]);
  step1[20] = vqsubq_s16(step2[19], step2[20]);
  step1[21] = vqsubq_s16(step2[18], step2[21]);
  step1[22] = vqsubq_s16(step2[17], step2[22]);
  step1[23] = vqsubq_s16(step2[16], step2[23]);
  step1[24] = vqsubq_s16(step2[31], step2[24]);
  step1[25] = vqsubq_s16(step2[30], step2[25]);
  step1[26] = vqsubq_s16(step2[29], step2[26]);
  step1[27] = vqsubq_s16(step2[28], step2[27]);
  step1[28] = vqaddq_s16(step2[27], step2[28]);
  step1[29] = vqaddq_s16(step2[26], step2[29]);
  step1[30] = vqaddq_s16(step2[25], step2[30]);
  step1[31] = vqaddq_s16(step2[24], step2[31]);

  // stage 8

  btf_16_lane_0_1_neon(step1[27], step1[20], c1, &step2[27], &step2[20]);
  btf_16_lane_0_1_neon(step1[26], step1[21], c1, &step2[26], &step2[21]);
  btf_16_lane_0_1_neon(step1[25], step1[22], c1, &step2[25], &step2[22]);
  btf_16_lane_0_1_neon(step1[24], step1[23], c1, &step2[24], &step2[23]);

  step2[0] = vqaddq_s16(step1[0], step1[15]);
  step2[1] = vqaddq_s16(step1[1], step1[14]);
  step2[2] = vqaddq_s16(step1[2], step1[13]);
  step2[3] = vqaddq_s16(step1[3], step1[12]);
  step2[4] = vqaddq_s16(step1[4], step1[11]);
  step2[5] = vqaddq_s16(step1[5], step1[10]);
  step2[6] = vqaddq_s16(step1[6], step1[9]);
  step2[7] = vqaddq_s16(step1[7], step1[8]);
  step2[8] = vqsubq_s16(step1[7], step1[8]);
  step2[9] = vqsubq_s16(step1[6], step1[9]);
  step2[10] = vqsubq_s16(step1[5], step1[10]);
  step2[11] = vqsubq_s16(step1[4], step1[11]);
  step2[12] = vqsubq_s16(step1[3], step1[12]);
  step2[13] = vqsubq_s16(step1[2], step1[13]);
  step2[14] = vqsubq_s16(step1[1], step1[14]);
  step2[15] = vqsubq_s16(step1[0], step1[15]);
  step2[16] = step1[16];
  step2[17] = step1[17];
  step2[18] = step1[18];
  step2[19] = step1[19];
  step2[28] = step1[28];
  step2[29] = step1[29];
  step2[30] = step1[30];
  step2[31] = step1[31];

  // stage 9

  out[0] = vqaddq_s16(step2[0], step2[31]);
  out[1] = vqaddq_s16(step2[1], step2[30]);
  out[2] = vqaddq_s16(step2[2], step2[29]);
  out[3] = vqaddq_s16(step2[3], step2[28]);
  out[4] = vqaddq_s16(step2[4], step2[27]);
  out[5] = vqaddq_s16(step2[5], step2[26]);
  out[6] = vqaddq_s16(step2[6], step2[25]);
  out[7] = vqaddq_s16(step2[7], step2[24]);
  out[8] = vqaddq_s16(step2[8], step2[23]);
  out[9] = vqaddq_s16(step2[9], step2[22]);
  out[10] = vqaddq_s16(step2[10], step2[21]);
  out[11] = vqaddq_s16(step2[11], step2[20]);
  out[12] = vqaddq_s16(step2[12], step2[19]);
  out[13] = vqaddq_s16(step2[13], step2[18]);
  out[14] = vqaddq_s16(step2[14], step2[17]);
  out[15] = vqaddq_s16(step2[15], step2[16]);
  out[16] = vqsubq_s16(step2[15], step2[16]);
  out[17] = vqsubq_s16(step2[14], step2[17]);
  out[18] = vqsubq_s16(step2[13], step2[18]);
  out[19] = vqsubq_s16(step2[12], step2[19]);
  out[20] = vqsubq_s16(step2[11], step2[20]);
  out[21] = vqsubq_s16(step2[10], step2[21]);
  out[22] = vqsubq_s16(step2[9], step2[22]);
  out[23] = vqsubq_s16(step2[8], step2[23]);
  out[24] = vqsubq_s16(step2[7], step2[24]);
  out[25] = vqsubq_s16(step2[6], step2[25]);
  out[26] = vqsubq_s16(step2[5], step2[26]);
  out[27] = vqsubq_s16(step2[4], step2[27]);
  out[28] = vqsubq_s16(step2[3], step2[28]);
  out[29] = vqsubq_s16(step2[2], step2[29]);
  out[30] = vqsubq_s16(step2[1], step2[30]);
  out[31] = vqsubq_s16(step2[0], step2[31]);
}
static inline void idct64_stage9_neon(int16x8_t *step2, int16x8_t *step1,
                                      int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);
  const int16x4_t c3 = set_s16x4_neon((int16_t)cospi[32], (int16_t)cospi[32],
                                      (int16_t)cospi[16], (int16_t)cospi[48]);

  btf_16_lane_0_1_neon(step2[27], step2[20], c3, &step1[27], &step1[20]);
  btf_16_lane_0_1_neon(step2[26], step2[21], c3, &step1[26], &step1[21]);
  btf_16_lane_0_1_neon(step2[25], step2[22], c3, &step1[25], &step1[22]);
  btf_16_lane_0_1_neon(step2[24], step2[23], c3, &step1[24], &step1[23]);

  step1[0] = vqaddq_s16(step2[0], step2[15]);
  step1[1] = vqaddq_s16(step2[1], step2[14]);
  step1[2] = vqaddq_s16(step2[2], step2[13]);
  step1[3] = vqaddq_s16(step2[3], step2[12]);
  step1[4] = vqaddq_s16(step2[4], step2[11]);
  step1[5] = vqaddq_s16(step2[5], step2[10]);
  step1[6] = vqaddq_s16(step2[6], step2[9]);
  step1[7] = vqaddq_s16(step2[7], step2[8]);
  step1[8] = vqsubq_s16(step2[7], step2[8]);
  step1[9] = vqsubq_s16(step2[6], step2[9]);
  step1[10] = vqsubq_s16(step2[5], step2[10]);
  step1[11] = vqsubq_s16(step2[4], step2[11]);
  step1[12] = vqsubq_s16(step2[3], step2[12]);
  step1[13] = vqsubq_s16(step2[2], step2[13]);
  step1[14] = vqsubq_s16(step2[1], step2[14]);
  step1[15] = vqsubq_s16(step2[0], step2[15]);
  step1[16] = step2[16];
  step1[17] = step2[17];
  step1[18] = step2[18];
  step1[19] = step2[19];
  step1[28] = step2[28];
  step1[29] = step2[29];
  step1[30] = step2[30];
  step1[31] = step2[31];
  step1[32] = vqaddq_s16(step2[32], step2[47]);
  step1[33] = vqaddq_s16(step2[33], step2[46]);
  step1[34] = vqaddq_s16(step2[34], step2[45]);
  step1[35] = vqaddq_s16(step2[35], step2[44]);
  step1[36] = vqaddq_s16(step2[36], step2[43]);
  step1[37] = vqaddq_s16(step2[37], step2[42]);
  step1[38] = vqaddq_s16(step2[38], step2[41]);
  step1[39] = vqaddq_s16(step2[39], step2[40]);
  step1[40] = vqsubq_s16(step2[39], step2[40]);
  step1[41] = vqsubq_s16(step2[38], step2[41]);
  step1[42] = vqsubq_s16(step2[37], step2[42]);
  step1[43] = vqsubq_s16(step2[36], step2[43]);
  step1[44] = vqsubq_s16(step2[35], step2[44]);
  step1[45] = vqsubq_s16(step2[34], step2[45]);
  step1[46] = vqsubq_s16(step2[33], step2[46]);
  step1[47] = vqsubq_s16(step2[32], step2[47]);
  step1[48] = vqsubq_s16(step2[63], step2[48]);
  step1[49] = vqsubq_s16(step2[62], step2[49]);
  step1[50] = vqsubq_s16(step2[61], step2[50]);
  step1[51] = vqsubq_s16(step2[60], step2[51]);
  step1[52] = vqsubq_s16(step2[59], step2[52]);
  step1[53] = vqsubq_s16(step2[58], step2[53]);
  step1[54] = vqsubq_s16(step2[57], step2[54]);
  step1[55] = vqsubq_s16(step2[56], step2[55]);
  step1[56] = vqaddq_s16(step2[56], step2[55]);
  step1[57] = vqaddq_s16(step2[57], step2[54]);
  step1[58] = vqaddq_s16(step2[58], step2[53]);
  step1[59] = vqaddq_s16(step2[59], step2[52]);
  step1[60] = vqaddq_s16(step2[60], step2[51]);
  step1[61] = vqaddq_s16(step2[61], step2[50]);
  step1[62] = vqaddq_s16(step2[62], step2[49]);
  step1[63] = vqaddq_s16(step2[63], step2[48]);
}

static inline void idct64_stage10_neon(int16x8_t *step1, int16x8_t *step2,
                                       int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);
  const int16x4_t c3 = set_s16x4_neon((int16_t)cospi[32], (int16_t)cospi[32],
                                      (int16_t)cospi[16], (int16_t)cospi[48]);

  btf_16_lane_0_1_neon(step1[55], step1[40], c3, &step2[55], &step2[40]);
  btf_16_lane_0_1_neon(step1[54], step1[41], c3, &step2[54], &step2[41]);
  btf_16_lane_0_1_neon(step1[53], step1[42], c3, &step2[53], &step2[42]);
  btf_16_lane_0_1_neon(step1[52], step1[43], c3, &step2[52], &step2[43]);
  btf_16_lane_0_1_neon(step1[51], step1[44], c3, &step2[51], &step2[44]);
  btf_16_lane_0_1_neon(step1[50], step1[45], c3, &step2[50], &step2[45]);
  btf_16_lane_0_1_neon(step1[49], step1[46], c3, &step2[49], &step2[46]);
  btf_16_lane_0_1_neon(step1[48], step1[47], c3, &step2[48], &step2[47]);

  step2[0] = vqaddq_s16(step1[0], step1[31]);
  step2[1] = vqaddq_s16(step1[1], step1[30]);
  step2[2] = vqaddq_s16(step1[2], step1[29]);
  step2[3] = vqaddq_s16(step1[3], step1[28]);
  step2[4] = vqaddq_s16(step1[4], step1[27]);
  step2[5] = vqaddq_s16(step1[5], step1[26]);
  step2[6] = vqaddq_s16(step1[6], step1[25]);
  step2[7] = vqaddq_s16(step1[7], step1[24]);
  step2[8] = vqaddq_s16(step1[8], step1[23]);
  step2[9] = vqaddq_s16(step1[9], step1[22]);
  step2[10] = vqaddq_s16(step1[10], step1[21]);
  step2[11] = vqaddq_s16(step1[11], step1[20]);
  step2[12] = vqaddq_s16(step1[12], step1[19]);
  step2[13] = vqaddq_s16(step1[13], step1[18]);
  step2[14] = vqaddq_s16(step1[14], step1[17]);
  step2[15] = vqaddq_s16(step1[15], step1[16]);
  step2[16] = vqsubq_s16(step1[15], step1[16]);
  step2[17] = vqsubq_s16(step1[14], step1[17]);
  step2[18] = vqsubq_s16(step1[13], step1[18]);
  step2[19] = vqsubq_s16(step1[12], step1[19]);
  step2[20] = vqsubq_s16(step1[11], step1[20]);
  step2[21] = vqsubq_s16(step1[10], step1[21]);
  step2[22] = vqsubq_s16(step1[9], step1[22]);
  step2[23] = vqsubq_s16(step1[8], step1[23]);
  step2[24] = vqsubq_s16(step1[7], step1[24]);
  step2[25] = vqsubq_s16(step1[6], step1[25]);
  step2[26] = vqsubq_s16(step1[5], step1[26]);
  step2[27] = vqsubq_s16(step1[4], step1[27]);
  step2[28] = vqsubq_s16(step1[3], step1[28]);
  step2[29] = vqsubq_s16(step1[2], step1[29]);
  step2[30] = vqsubq_s16(step1[1], step1[30]);
  step2[31] = vqsubq_s16(step1[0], step1[31]);
  step2[32] = step1[32];
  step2[33] = step1[33];
  step2[34] = step1[34];
  step2[35] = step1[35];
  step2[36] = step1[36];
  step2[37] = step1[37];
  step2[38] = step1[38];
  step2[39] = step1[39];
  step2[56] = step1[56];
  step2[57] = step1[57];
  step2[58] = step1[58];
  step2[59] = step1[59];
  step2[60] = step1[60];
  step2[61] = step1[61];
  step2[62] = step1[62];
  step2[63] = step1[63];
}

static inline void idct64_low32_neon(int16x8_t *in, int16x8_t *out,
                                     int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);
  int16x8_t step2[64], step1[64];
  const int16x4_t c0 = set_s16x4_neon((int16_t)cospi[4], (int16_t)cospi[60],
                                      (int16_t)cospi[36], (int16_t)cospi[28]);
  const int16x4_t c1 = set_s16x4_neon((int16_t)cospi[20], (int16_t)cospi[44],
                                      (int16_t)cospi[52], (int16_t)cospi[12]);
  const int16x4_t c2 = set_s16x4_neon((int16_t)cospi[8], (int16_t)cospi[56],
                                      (int16_t)cospi[40], (int16_t)cospi[24]);
  const int16x4_t c3 = set_s16x4_neon((int16_t)cospi[32], (int16_t)cospi[32],
                                      (int16_t)cospi[16], (int16_t)cospi[48]);
  const int16x4_t c4 =
      set_s16x4_neon((int16_t)(-cospi[4]), (int16_t)(-cospi[60]),
                     (int16_t)(-cospi[36]), (int16_t)(-cospi[28]));
  const int16x4_t c5 =
      set_s16x4_neon((int16_t)(-cospi[20]), (int16_t)(-cospi[44]),
                     (int16_t)(-cospi[52]), (int16_t)(-cospi[12]));
  const int16x4_t c6 =
      set_s16x4_neon((int16_t)(-cospi[8]), (int16_t)(-cospi[56]),
                     (int16_t)(-cospi[40]), (int16_t)(-cospi[24]));
  const int16x4_t c7 =
      set_s16x4_neon((int16_t)(-cospi[32]), (int16_t)(-cospi[32]),
                     (int16_t)(-cospi[16]), (int16_t)(-cospi[48]));

  // stage 1
  // stage 2

  step2[0] = in[0];
  step2[2] = in[16];
  step2[4] = in[8];
  step2[6] = in[24];
  step2[8] = in[4];
  step2[10] = in[20];
  step2[12] = in[12];
  step2[14] = in[28];
  step2[16] = in[2];
  step2[18] = in[18];
  step2[20] = in[10];
  step2[22] = in[26];
  step2[24] = in[6];
  step2[26] = in[22];
  step2[28] = in[14];
  step2[30] = in[30];

  btf_16_neon(in[1], cospi[63], cospi[1], &step2[32], &step2[63]);
  btf_16_neon(in[31], -cospi[33], cospi[31], &step2[33], &step2[62]);
  btf_16_neon(in[17], cospi[47], cospi[17], &step2[34], &step2[61]);
  btf_16_neon(in[15], -cospi[49], cospi[15], &step2[35], &step2[60]);
  btf_16_neon(in[9], cospi[55], cospi[9], &step2[36], &step2[59]);
  btf_16_neon(in[23], -cospi[41], cospi[23], &step2[37], &step2[58]);
  btf_16_neon(in[25], cospi[39], cospi[25], &step2[38], &step2[57]);
  btf_16_neon(in[7], -cospi[57], cospi[7], &step2[39], &step2[56]);
  btf_16_neon(in[5], cospi[59], cospi[5], &step2[40], &step2[55]);
  btf_16_neon(in[27], -cospi[37], cospi[27], &step2[41], &step2[54]);
  btf_16_neon(in[21], cospi[43], cospi[21], &step2[42], &step2[53]);
  btf_16_neon(in[11], -cospi[53], cospi[11], &step2[43], &step2[52]);
  btf_16_neon(in[13], cospi[51], cospi[13], &step2[44], &step2[51]);
  btf_16_neon(in[19], -cospi[45], cospi[19], &step2[45], &step2[50]);
  btf_16_neon(in[29], cospi[35], cospi[29], &step2[46], &step2[49]);
  btf_16_neon(in[3], -cospi[61], cospi[3], &step2[47], &step2[48]);

  // stage 3

  step1[0] = step2[0];
  step1[2] = step2[2];
  step1[4] = step2[4];
  step1[6] = step2[6];
  step1[8] = step2[8];
  step1[10] = step2[10];
  step1[12] = step2[12];
  step1[14] = step2[14];

  btf_16_neon(step2[16], cospi[62], cospi[2], &step1[16], &step1[31]);
  btf_16_neon(step2[30], -cospi[34], cospi[30], &step1[17], &step1[30]);
  btf_16_neon(step2[18], cospi[46], cospi[18], &step1[18], &step1[29]);
  btf_16_neon(step2[28], -cospi[50], cospi[14], &step1[19], &step1[28]);
  btf_16_neon(step2[20], cospi[54], cospi[10], &step1[20], &step1[27]);
  btf_16_neon(step2[26], -cospi[42], cospi[22], &step1[21], &step1[26]);
  btf_16_neon(step2[22], cospi[38], cospi[26], &step1[22], &step1[25]);
  btf_16_neon(step2[24], -cospi[58], cospi[6], &step1[23], &step1[24]);

  step1[32] = vqaddq_s16(step2[32], step2[33]);
  step1[33] = vqsubq_s16(step2[32], step2[33]);
  step1[34] = vqsubq_s16(step2[35], step2[34]);
  step1[35] = vqaddq_s16(step2[35], step2[34]);
  step1[36] = vqaddq_s16(step2[36], step2[37]);
  step1[37] = vqsubq_s16(step2[36], step2[37]);
  step1[38] = vqsubq_s16(step2[39], step2[38]);
  step1[39] = vqaddq_s16(step2[39], step2[38]);
  step1[40] = vqaddq_s16(step2[40], step2[41]);
  step1[41] = vqsubq_s16(step2[40], step2[41]);
  step1[42] = vqsubq_s16(step2[43], step2[42]);
  step1[43] = vqaddq_s16(step2[43], step2[42]);
  step1[44] = vqaddq_s16(step2[44], step2[45]);
  step1[45] = vqsubq_s16(step2[44], step2[45]);
  step1[46] = vqsubq_s16(step2[47], step2[46]);
  step1[47] = vqaddq_s16(step2[47], step2[46]);
  step1[48] = vqaddq_s16(step2[48], step2[49]);
  step1[49] = vqsubq_s16(step2[48], step2[49]);
  step1[50] = vqsubq_s16(step2[51], step2[50]);
  step1[51] = vqaddq_s16(step2[51], step2[50]);
  step1[52] = vqaddq_s16(step2[52], step2[53]);
  step1[53] = vqsubq_s16(step2[52], step2[53]);
  step1[54] = vqsubq_s16(step2[55], step2[54]);
  step1[55] = vqaddq_s16(step2[55], step2[54]);
  step1[56] = vqaddq_s16(step2[56], step2[57]);
  step1[57] = vqsubq_s16(step2[56], step2[57]);
  step1[58] = vqsubq_s16(step2[59], step2[58]);
  step1[59] = vqaddq_s16(step2[59], step2[58]);
  step1[60] = vqaddq_s16(step2[60], step2[61]);
  step1[61] = vqsubq_s16(step2[60], step2[61]);
  step1[62] = vqsubq_s16(step2[63], step2[62]);
  step1[63] = vqaddq_s16(step2[63], step2[62]);

  // stage 4

  step2[0] = step1[0];
  step2[2] = step1[2];
  step2[4] = step1[4];
  step2[6] = step1[6];

  btf_16_neon(step1[8], cospi[60], cospi[4], &step2[8], &step2[15]);
  btf_16_neon(step1[14], -cospi[36], cospi[28], &step2[9], &step2[14]);
  btf_16_neon(step1[10], cospi[44], cospi[20], &step2[10], &step2[13]);
  btf_16_neon(step1[12], -cospi[52], cospi[12], &step2[11], &step2[12]);
  btf_16_lane_0_1_neon(step1[62], step1[33], c0, &step2[62], &step2[33]);
  btf_16_lane_1_0_neon(step1[34], step1[61], c4, &step2[34], &step2[61]);
  btf_16_lane_2_3_neon(step1[58], step1[37], c0, &step2[58], &step2[37]);
  btf_16_lane_3_2_neon(step1[38], step1[57], c4, &step2[38], &step2[57]);
  btf_16_lane_0_1_neon(step1[54], step1[41], c1, &step2[54], &step2[41]);
  btf_16_lane_1_0_neon(step1[42], step1[53], c5, &step2[42], &step2[53]);
  btf_16_lane_2_3_neon(step1[50], step1[45], c1, &step2[50], &step2[45]);
  btf_16_lane_3_2_neon(step1[46], step1[49], c5, &step2[46], &step2[49]);

  step2[16] = vqaddq_s16(step1[16], step1[17]);
  step2[17] = vqsubq_s16(step1[16], step1[17]);
  step2[18] = vqsubq_s16(step1[19], step1[18]);
  step2[19] = vqaddq_s16(step1[19], step1[18]);
  step2[20] = vqaddq_s16(step1[20], step1[21]);
  step2[21] = vqsubq_s16(step1[20], step1[21]);
  step2[22] = vqsubq_s16(step1[23], step1[22]);
  step2[23] = vqaddq_s16(step1[23], step1[22]);
  step2[24] = vqaddq_s16(step1[24], step1[25]);
  step2[25] = vqsubq_s16(step1[24], step1[25]);
  step2[26] = vqsubq_s16(step1[27], step1[26]);
  step2[27] = vqaddq_s16(step1[27], step1[26]);
  step2[28] = vqaddq_s16(step1[28], step1[29]);
  step2[29] = vqsubq_s16(step1[28], step1[29]);
  step2[30] = vqsubq_s16(step1[31], step1[30]);
  step2[31] = vqaddq_s16(step1[31], step1[30]);
  step2[32] = step1[32];
  step2[35] = step1[35];
  step2[36] = step1[36];
  step2[39] = step1[39];
  step2[40] = step1[40];
  step2[43] = step1[43];
  step2[44] = step1[44];
  step2[47] = step1[47];
  step2[48] = step1[48];
  step2[51] = step1[51];
  step2[52] = step1[52];
  step2[55] = step1[55];
  step2[56] = step1[56];
  step2[59] = step1[59];
  step2[60] = step1[60];
  step2[63] = step1[63];

  // stage 5

  step1[0] = step2[0];
  step1[2] = step2[2];

  btf_16_neon(step2[4], cospi[56], cospi[8], &step1[4], &step1[7]);
  btf_16_neon(step2[6], -cospi[40], cospi[24], &step1[5], &step1[6]);
  btf_16_lane_0_1_neon(step2[30], step2[17], c2, &step1[30], &step1[17]);
  btf_16_lane_1_0_neon(step2[18], step2[29], c6, &step1[18], &step1[29]);
  btf_16_lane_2_3_neon(step2[26], step2[21], c2, &step1[26], &step1[21]);
  btf_16_lane_3_2_neon(step2[22], step2[25], c6, &step1[22], &step1[25]);

  step1[8] = vqaddq_s16(step2[8], step2[9]);
  step1[9] = vqsubq_s16(step2[8], step2[9]);
  step1[10] = vqsubq_s16(step2[11], step2[10]);
  step1[11] = vqaddq_s16(step2[11], step2[10]);
  step1[12] = vqaddq_s16(step2[12], step2[13]);
  step1[13] = vqsubq_s16(step2[12], step2[13]);
  step1[14] = vqsubq_s16(step2[15], step2[14]);
  step1[15] = vqaddq_s16(step2[15], step2[14]);
  step1[16] = step2[16];
  step1[19] = step2[19];
  step1[20] = step2[20];
  step1[23] = step2[23];
  step1[24] = step2[24];
  step1[27] = step2[27];
  step1[28] = step2[28];
  step1[31] = step2[31];
  step1[32] = vqaddq_s16(step2[32], step2[35]);
  step1[33] = vqaddq_s16(step2[33], step2[34]);
  step1[34] = vqsubq_s16(step2[33], step2[34]);
  step1[35] = vqsubq_s16(step2[32], step2[35]);
  step1[36] = vqsubq_s16(step2[39], step2[36]);
  step1[37] = vqsubq_s16(step2[38], step2[37]);
  step1[38] = vqaddq_s16(step2[38], step2[37]);
  step1[39] = vqaddq_s16(step2[39], step2[36]);
  step1[40] = vqaddq_s16(step2[40], step2[43]);
  step1[41] = vqaddq_s16(step2[41], step2[42]);
  step1[42] = vqsubq_s16(step2[41], step2[42]);
  step1[43] = vqsubq_s16(step2[40], step2[43]);
  step1[44] = vqsubq_s16(step2[47], step2[44]);
  step1[45] = vqsubq_s16(step2[46], step2[45]);
  step1[46] = vqaddq_s16(step2[46], step2[45]);
  step1[47] = vqaddq_s16(step2[47], step2[44]);
  step1[48] = vqaddq_s16(step2[48], step2[51]);
  step1[49] = vqaddq_s16(step2[49], step2[50]);
  step1[50] = vqsubq_s16(step2[49], step2[50]);
  step1[51] = vqsubq_s16(step2[48], step2[51]);
  step1[52] = vqsubq_s16(step2[55], step2[52]);
  step1[53] = vqsubq_s16(step2[54], step2[53]);
  step1[54] = vqaddq_s16(step2[54], step2[53]);
  step1[55] = vqaddq_s16(step2[55], step2[52]);
  step1[56] = vqaddq_s16(step2[56], step2[59]);
  step1[57] = vqaddq_s16(step2[57], step2[58]);
  step1[58] = vqsubq_s16(step2[57], step2[58]);
  step1[59] = vqsubq_s16(step2[56], step2[59]);
  step1[60] = vqsubq_s16(step2[63], step2[60]);
  step1[61] = vqsubq_s16(step2[62], step2[61]);
  step1[62] = vqaddq_s16(step2[62], step2[61]);
  step1[63] = vqaddq_s16(step2[63], step2[60]);

  // stage 6

  btf_16_neon(step1[0], cospi[32], cospi[32], &step2[0], &step2[1]);
  btf_16_neon(step1[2], cospi[48], cospi[16], &step2[2], &step2[3]);
  btf_16_lane_2_3_neon(step1[14], step1[9], c3, &step2[14], &step2[9]);
  btf_16_lane_3_2_neon(step1[10], step1[13], c7, &step2[10], &step2[13]);
  btf_16_lane_0_1_neon(step1[61], step1[34], c2, &step2[61], &step2[34]);
  btf_16_lane_0_1_neon(step1[60], step1[35], c2, &step2[60], &step2[35]);
  btf_16_lane_1_0_neon(step1[36], step1[59], c6, &step2[36], &step2[59]);
  btf_16_lane_1_0_neon(step1[37], step1[58], c6, &step2[37], &step2[58]);
  btf_16_lane_2_3_neon(step1[53], step1[42], c2, &step2[53], &step2[42]);
  btf_16_lane_2_3_neon(step1[52], step1[43], c2, &step2[52], &step2[43]);
  btf_16_lane_3_2_neon(step1[44], step1[51], c6, &step2[44], &step2[51]);
  btf_16_lane_3_2_neon(step1[45], step1[50], c6, &step2[45], &step2[50]);

  step2[4] = vqaddq_s16(step1[4], step1[5]);
  step2[5] = vqsubq_s16(step1[4], step1[5]);
  step2[6] = vqsubq_s16(step1[7], step1[6]);
  step2[7] = vqaddq_s16(step1[7], step1[6]);
  step2[8] = step1[8];
  step2[11] = step1[11];
  step2[12] = step1[12];
  step2[15] = step1[15];
  step2[16] = vqaddq_s16(step1[16], step1[19]);
  step2[17] = vqaddq_s16(step1[17], step1[18]);
  step2[18] = vqsubq_s16(step1[17], step1[18]);
  step2[19] = vqsubq_s16(step1[16], step1[19]);
  step2[20] = vqsubq_s16(step1[23], step1[20]);
  step2[21] = vqsubq_s16(step1[22], step1[21]);
  step2[22] = vqaddq_s16(step1[22], step1[21]);
  step2[23] = vqaddq_s16(step1[23], step1[20]);
  step2[24] = vqaddq_s16(step1[24], step1[27]);
  step2[25] = vqaddq_s16(step1[25], step1[26]);
  step2[26] = vqsubq_s16(step1[25], step1[26]);
  step2[27] = vqsubq_s16(step1[24], step1[27]);
  step2[28] = vqsubq_s16(step1[31], step1[28]);
  step2[29] = vqsubq_s16(step1[30], step1[29]);
  step2[30] = vqaddq_s16(step1[30], step1[29]);
  step2[31] = vqaddq_s16(step1[31], step1[28]);
  step2[32] = step1[32];
  step2[33] = step1[33];
  step2[38] = step1[38];
  step2[39] = step1[39];
  step2[40] = step1[40];
  step2[41] = step1[41];
  step2[46] = step1[46];
  step2[47] = step1[47];
  step2[48] = step1[48];
  step2[49] = step1[49];
  step2[54] = step1[54];
  step2[55] = step1[55];
  step2[56] = step1[56];
  step2[57] = step1[57];
  step2[62] = step1[62];
  step2[63] = step1[63];

  // stage 7

  btf_16_lane_0_1_neon(step2[6], step2[5], c3, &step1[6], &step1[5]);
  btf_16_lane_2_3_neon(step2[29], step2[18], c3, &step1[29], &step1[18]);
  btf_16_lane_2_3_neon(step2[28], step2[19], c3, &step1[28], &step1[19]);
  btf_16_lane_3_2_neon(step2[20], step2[27], c7, &step1[20], &step1[27]);
  btf_16_lane_3_2_neon(step2[21], step2[26], c7, &step1[21], &step1[26]);

  step1[0] = vqaddq_s16(step2[0], step2[3]);
  step1[1] = vqaddq_s16(step2[1], step2[2]);
  step1[2] = vqsubq_s16(step2[1], step2[2]);
  step1[3] = vqsubq_s16(step2[0], step2[3]);
  step1[4] = step2[4];
  step1[7] = step2[7];
  step1[8] = vqaddq_s16(step2[8], step2[11]);
  step1[9] = vqaddq_s16(step2[9], step2[10]);
  step1[10] = vqsubq_s16(step2[9], step2[10]);
  step1[11] = vqsubq_s16(step2[8], step2[11]);
  step1[12] = vqsubq_s16(step2[15], step2[12]);
  step1[13] = vqsubq_s16(step2[14], step2[13]);
  step1[14] = vqaddq_s16(step2[14], step2[13]);
  step1[15] = vqaddq_s16(step2[15], step2[12]);
  step1[16] = step2[16];
  step1[17] = step2[17];
  step1[22] = step2[22];
  step1[23] = step2[23];
  step1[24] = step2[24];
  step1[25] = step2[25];
  step1[30] = step2[30];
  step1[31] = step2[31];
  step1[32] = vqaddq_s16(step2[32], step2[39]);
  step1[33] = vqaddq_s16(step2[33], step2[38]);
  step1[34] = vqaddq_s16(step2[34], step2[37]);
  step1[35] = vqaddq_s16(step2[35], step2[36]);
  step1[36] = vqsubq_s16(step2[35], step2[36]);
  step1[37] = vqsubq_s16(step2[34], step2[37]);
  step1[38] = vqsubq_s16(step2[33], step2[38]);
  step1[39] = vqsubq_s16(step2[32], step2[39]);
  step1[40] = vqsubq_s16(step2[47], step2[40]);
  step1[41] = vqsubq_s16(step2[46], step2[41]);
  step1[42] = vqsubq_s16(step2[45], step2[42]);
  step1[43] = vqsubq_s16(step2[44], step2[43]);
  step1[44] = vqaddq_s16(step2[43], step2[44]);
  step1[45] = vqaddq_s16(step2[42], step2[45]);
  step1[46] = vqaddq_s16(step2[41], step2[46]);
  step1[47] = vqaddq_s16(step2[40], step2[47]);
  step1[48] = vqaddq_s16(step2[48], step2[55]);
  step1[49] = vqaddq_s16(step2[49], step2[54]);
  step1[50] = vqaddq_s16(step2[50], step2[53]);
  step1[51] = vqaddq_s16(step2[51], step2[52]);
  step1[52] = vqsubq_s16(step2[51], step2[52]);
  step1[53] = vqsubq_s16(step2[50], step2[53]);
  step1[54] = vqsubq_s16(step2[49], step2[54]);
  step1[55] = vqsubq_s16(step2[48], step2[55]);
  step1[56] = vqsubq_s16(step2[63], step2[56]);
  step1[57] = vqsubq_s16(step2[62], step2[57]);
  step1[58] = vqsubq_s16(step2[61], step2[58]);
  step1[59] = vqsubq_s16(step2[60], step2[59]);
  step1[60] = vqaddq_s16(step2[59], step2[60]);
  step1[61] = vqaddq_s16(step2[58], step2[61]);
  step1[62] = vqaddq_s16(step2[57], step2[62]);
  step1[63] = vqaddq_s16(step2[56], step2[63]);

  // stage 8

  btf_16_lane_0_1_neon(step1[13], step1[10], c3, &step2[13], &step2[10]);
  btf_16_lane_0_1_neon(step1[12], step1[11], c3, &step2[12], &step2[11]);
  btf_16_lane_2_3_neon(step1[59], step1[36], c3, &step2[59], &step2[36]);
  btf_16_lane_2_3_neon(step1[58], step1[37], c3, &step2[58], &step2[37]);
  btf_16_lane_2_3_neon(step1[57], step1[38], c3, &step2[57], &step2[38]);
  btf_16_lane_2_3_neon(step1[56], step1[39], c3, &step2[56], &step2[39]);
  btf_16_lane_3_2_neon(step1[40], step1[55], c7, &step2[40], &step2[55]);
  btf_16_lane_3_2_neon(step1[41], step1[54], c7, &step2[41], &step2[54]);
  btf_16_lane_3_2_neon(step1[42], step1[53], c7, &step2[42], &step2[53]);
  btf_16_lane_3_2_neon(step1[43], step1[52], c7, &step2[43], &step2[52]);

  step2[0] = vqaddq_s16(step1[0], step1[7]);
  step2[1] = vqaddq_s16(step1[1], step1[6]);
  step2[2] = vqaddq_s16(step1[2], step1[5]);
  step2[3] = vqaddq_s16(step1[3], step1[4]);
  step2[4] = vqsubq_s16(step1[3], step1[4]);
  step2[5] = vqsubq_s16(step1[2], step1[5]);
  step2[6] = vqsubq_s16(step1[1], step1[6]);
  step2[7] = vqsubq_s16(step1[0], step1[7]);
  step2[8] = step1[8];
  step2[9] = step1[9];
  step2[14] = step1[14];
  step2[15] = step1[15];
  step2[16] = vqaddq_s16(step1[16], step1[23]);
  step2[17] = vqaddq_s16(step1[17], step1[22]);
  step2[18] = vqaddq_s16(step1[18], step1[21]);
  step2[19] = vqaddq_s16(step1[19], step1[20]);
  step2[20] = vqsubq_s16(step1[19], step1[20]);
  step2[21] = vqsubq_s16(step1[18], step1[21]);
  step2[22] = vqsubq_s16(step1[17], step1[22]);
  step2[23] = vqsubq_s16(step1[16], step1[23]);
  step2[24] = vqsubq_s16(step1[31], step1[24]);
  step2[25] = vqsubq_s16(step1[30], step1[25]);
  step2[26] = vqsubq_s16(step1[29], step1[26]);
  step2[27] = vqsubq_s16(step1[28], step1[27]);
  step2[28] = vqaddq_s16(step1[28], step1[27]);
  step2[29] = vqaddq_s16(step1[29], step1[26]);
  step2[30] = vqaddq_s16(step1[30], step1[25]);
  step2[31] = vqaddq_s16(step1[31], step1[24]);
  step2[32] = step1[32];
  step2[33] = step1[33];
  step2[34] = step1[34];
  step2[35] = step1[35];
  step2[44] = step1[44];
  step2[45] = step1[45];
  step2[46] = step1[46];
  step2[47] = step1[47];
  step2[48] = step1[48];
  step2[49] = step1[49];
  step2[50] = step1[50];
  step2[51] = step1[51];
  step2[60] = step1[60];
  step2[61] = step1[61];
  step2[62] = step1[62];
  step2[63] = step1[63];

  // stage 9
  idct64_stage9_neon(step2, step1, cos_bit);

  // stage 10
  idct64_stage10_neon(step1, step2, cos_bit);

  // stage 11

  out[0] = vqaddq_s16(step2[0], step2[63]);
  out[1] = vqaddq_s16(step2[1], step2[62]);
  out[2] = vqaddq_s16(step2[2], step2[61]);
  out[3] = vqaddq_s16(step2[3], step2[60]);
  out[4] = vqaddq_s16(step2[4], step2[59]);
  out[5] = vqaddq_s16(step2[5], step2[58]);
  out[6] = vqaddq_s16(step2[6], step2[57]);
  out[7] = vqaddq_s16(step2[7], step2[56]);
  out[8] = vqaddq_s16(step2[8], step2[55]);
  out[9] = vqaddq_s16(step2[9], step2[54]);
  out[10] = vqaddq_s16(step2[10], step2[53]);
  out[11] = vqaddq_s16(step2[11], step2[52]);
  out[12] = vqaddq_s16(step2[12], step2[51]);
  out[13] = vqaddq_s16(step2[13], step2[50]);
  out[14] = vqaddq_s16(step2[14], step2[49]);
  out[15] = vqaddq_s16(step2[15], step2[48]);
  out[16] = vqaddq_s16(step2[16], step2[47]);
  out[17] = vqaddq_s16(step2[17], step2[46]);
  out[18] = vqaddq_s16(step2[18], step2[45]);
  out[19] = vqaddq_s16(step2[19], step2[44]);
  out[20] = vqaddq_s16(step2[20], step2[43]);
  out[21] = vqaddq_s16(step2[21], step2[42]);
  out[22] = vqaddq_s16(step2[22], step2[41]);
  out[23] = vqaddq_s16(step2[23], step2[40]);
  out[24] = vqaddq_s16(step2[24], step2[39]);
  out[25] = vqaddq_s16(step2[25], step2[38]);
  out[26] = vqaddq_s16(step2[26], step2[37]);
  out[27] = vqaddq_s16(step2[27], step2[36]);
  out[28] = vqaddq_s16(step2[28], step2[35]);
  out[29] = vqaddq_s16(step2[29], step2[34]);
  out[30] = vqaddq_s16(step2[30], step2[33]);
  out[31] = vqaddq_s16(step2[31], step2[32]);
  out[32] = vqsubq_s16(step2[31], step2[32]);
  out[33] = vqsubq_s16(step2[30], step2[33]);
  out[34] = vqsubq_s16(step2[29], step2[34]);
  out[35] = vqsubq_s16(step2[28], step2[35]);
  out[36] = vqsubq_s16(step2[27], step2[36]);
  out[37] = vqsubq_s16(step2[26], step2[37]);
  out[38] = vqsubq_s16(step2[25], step2[38]);
  out[39] = vqsubq_s16(step2[24], step2[39]);
  out[40] = vqsubq_s16(step2[23], step2[40]);
  out[41] = vqsubq_s16(step2[22], step2[41]);
  out[42] = vqsubq_s16(step2[21], step2[42]);
  out[43] = vqsubq_s16(step2[20], step2[43]);
  out[44] = vqsubq_s16(step2[19], step2[44]);
  out[45] = vqsubq_s16(step2[18], step2[45]);
  out[46] = vqsubq_s16(step2[17], step2[46]);
  out[47] = vqsubq_s16(step2[16], step2[47]);
  out[48] = vqsubq_s16(step2[15], step2[48]);
  out[49] = vqsubq_s16(step2[14], step2[49]);
  out[50] = vqsubq_s16(step2[13], step2[50]);
  out[51] = vqsubq_s16(step2[12], step2[51]);
  out[52] = vqsubq_s16(step2[11], step2[52]);
  out[53] = vqsubq_s16(step2[10], step2[53]);
  out[54] = vqsubq_s16(step2[9], step2[54]);
  out[55] = vqsubq_s16(step2[8], step2[55]);
  out[56] = vqsubq_s16(step2[7], step2[56]);
  out[57] = vqsubq_s16(step2[6], step2[57]);
  out[58] = vqsubq_s16(step2[5], step2[58]);
  out[59] = vqsubq_s16(step2[4], step2[59]);
  out[60] = vqsubq_s16(step2[3], step2[60]);
  out[61] = vqsubq_s16(step2[2], step2[61]);
  out[62] = vqsubq_s16(step2[1], step2[62]);
  out[63] = vqsubq_s16(step2[0], step2[63]);
}

static inline void idct64_low1_neon(int16x8_t *input, int16x8_t *out,
                                    int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);
  int16x8_t step1;
  int32x4_t t32[2];

  // stage 1
  // stage 2
  // stage 3
  // stage 4
  // stage 5
  // stage 6

  t32[0] = vmull_n_s16(vget_low_s16(input[0]), cospi[32]);
  t32[1] = vmull_n_s16(vget_high_s16(input[0]), cospi[32]);

  step1 = vcombine_s16(vrshrn_n_s32(t32[0], INV_COS_BIT),
                       vrshrn_n_s32(t32[1], INV_COS_BIT));
  // stage 7
  // stage 8
  // stage 9
  // stage 10
  // stage 11
  out[0] = step1;
  out[1] = step1;
  out[2] = step1;
  out[3] = step1;
  out[4] = step1;
  out[5] = step1;
  out[6] = step1;
  out[7] = step1;
  out[8] = step1;
  out[9] = step1;
  out[10] = step1;
  out[11] = step1;
  out[12] = step1;
  out[13] = step1;
  out[14] = step1;
  out[15] = step1;
  out[16] = step1;
  out[17] = step1;
  out[18] = step1;
  out[19] = step1;
  out[20] = step1;
  out[21] = step1;
  out[22] = step1;
  out[23] = step1;
  out[24] = step1;
  out[25] = step1;
  out[26] = step1;
  out[27] = step1;
  out[28] = step1;
  out[29] = step1;
  out[30] = step1;
  out[31] = step1;
  out[32] = step1;
  out[33] = step1;
  out[34] = step1;
  out[35] = step1;
  out[36] = step1;
  out[37] = step1;
  out[38] = step1;
  out[39] = step1;
  out[40] = step1;
  out[41] = step1;
  out[42] = step1;
  out[43] = step1;
  out[44] = step1;
  out[45] = step1;
  out[46] = step1;
  out[47] = step1;
  out[48] = step1;
  out[49] = step1;
  out[50] = step1;
  out[51] = step1;
  out[52] = step1;
  out[53] = step1;
  out[54] = step1;
  out[55] = step1;
  out[56] = step1;
  out[57] = step1;
  out[58] = step1;
  out[59] = step1;
  out[60] = step1;
  out[61] = step1;
  out[62] = step1;
  out[63] = step1;
}

static inline void idct64_low8_neon(int16x8_t *in, int16x8_t *out,
                                    int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);
  int16x8_t step2[64], step1[64];

  const int16x4_t c0 = set_s16x4_neon((int16_t)cospi[4], (int16_t)cospi[60],
                                      (int16_t)cospi[36], (int16_t)cospi[28]);
  const int16x4_t c1 = set_s16x4_neon((int16_t)cospi[20], (int16_t)cospi[44],
                                      (int16_t)cospi[52], (int16_t)cospi[12]);
  const int16x4_t c2 = set_s16x4_neon((int16_t)cospi[8], (int16_t)cospi[56],
                                      (int16_t)cospi[40], (int16_t)cospi[24]);
  const int16x4_t c3 = set_s16x4_neon((int16_t)cospi[32], (int16_t)cospi[32],
                                      (int16_t)cospi[16], (int16_t)cospi[48]);
  const int16x4_t c4 =
      set_s16x4_neon((int16_t)(-cospi[36]), (int16_t)(-cospi[28]),
                     (int16_t)(-cospi[52]), (int16_t)(-cospi[12]));
  const int16x4_t c5 =
      set_s16x4_neon((int16_t)(-cospi[8]), (int16_t)(-cospi[56]),
                     (int16_t)(-cospi[40]), (int16_t)(-cospi[24]));
  const int16x4_t c6 =
      set_s16x4_neon((int16_t)(-cospi[32]), (int16_t)(-cospi[32]),
                     (int16_t)(-cospi[16]), (int16_t)(-cospi[48]));

  // stage 1
  // stage 2

  step2[0] = in[0];
  step2[8] = in[4];
  step2[16] = in[2];
  step2[24] = in[6];

  btf_16_neon(in[1], cospi[63], cospi[1], &step2[32], &step2[63]);
  btf_16_neon(in[7], -cospi[57], cospi[7], &step2[39], &step2[56]);
  btf_16_neon(in[5], cospi[59], cospi[5], &step2[40], &step2[55]);
  btf_16_neon(in[3], -cospi[61], cospi[3], &step2[47], &step2[48]);

  // stage 3

  step1[0] = step2[0];
  step1[8] = step2[8];

  btf_16_neon(step2[16], cospi[62], cospi[2], &step1[16], &step1[31]);
  btf_16_neon(step2[24], -cospi[58], cospi[6], &step1[23], &step1[24]);

  step1[32] = step2[32];
  step1[33] = step2[32];
  step1[38] = step2[39];
  step1[39] = step2[39];
  step1[40] = step2[40];
  step1[41] = step2[40];
  step1[46] = step2[47];
  step1[47] = step2[47];
  step1[48] = step2[48];
  step1[49] = step2[48];
  step1[54] = step2[55];
  step1[55] = step2[55];
  step1[56] = step2[56];
  step1[57] = step2[56];
  step1[62] = step2[63];
  step1[63] = step2[63];

  // stage 4

  step2[0] = step1[0];

  btf_16_neon(step1[8], cospi[60], cospi[4], &step2[8], &step2[15]);
  btf_16_lane_0_1_neon(step1[62], step1[33], c0, &step2[62], &step2[33]);
  btf_16_lane_1_0_neon(step1[38], step1[57], c4, &step2[38], &step2[57]);
  btf_16_lane_0_1_neon(step1[54], step1[41], c1, &step2[54], &step2[41]);
  btf_16_lane_3_2_neon(step1[46], step1[49], c4, &step2[46], &step2[49]);

  step2[16] = step1[16];
  step2[17] = step1[16];
  step2[22] = step1[23];
  step2[23] = step1[23];
  step2[24] = step1[24];
  step2[25] = step1[24];
  step2[30] = step1[31];
  step2[31] = step1[31];
  step2[32] = step1[32];
  step2[39] = step1[39];
  step2[40] = step1[40];
  step2[47] = step1[47];
  step2[48] = step1[48];
  step2[55] = step1[55];
  step2[56] = step1[56];
  step2[63] = step1[63];

  // stage 5

  step1[0] = step2[0];

  btf_16_lane_0_1_neon(step2[30], step2[17], c2, &step1[30], &step1[17]);
  btf_16_lane_3_2_neon(step2[22], step2[25], c5, &step1[22], &step1[25]);

  step1[8] = step2[8];
  step1[9] = step2[8];
  step1[14] = step2[15];
  step1[15] = step2[15];

  step1[16] = step2[16];
  step1[23] = step2[23];
  step1[24] = step2[24];
  step1[31] = step2[31];
  step1[32] = step2[32];
  step1[33] = step2[33];
  step1[34] = step2[33];
  step1[35] = step2[32];
  step1[36] = step2[39];
  step1[37] = step2[38];
  step1[38] = step2[38];
  step1[39] = step2[39];
  step1[40] = step2[40];
  step1[41] = step2[41];
  step1[42] = step2[41];
  step1[43] = step2[40];
  step1[44] = step2[47];
  step1[45] = step2[46];
  step1[46] = step2[46];
  step1[47] = step2[47];
  step1[48] = step2[48];
  step1[49] = step2[49];
  step1[50] = step2[49];
  step1[51] = step2[48];
  step1[52] = step2[55];
  step1[53] = step2[54];
  step1[54] = step2[54];
  step1[55] = step2[55];
  step1[56] = step2[56];
  step1[57] = step2[57];
  step1[58] = step2[57];
  step1[59] = step2[56];
  step1[60] = step2[63];
  step1[61] = step2[62];
  step1[62] = step2[62];
  step1[63] = step2[63];

  // stage 6

  btf_16_neon(step1[0], cospi[32], cospi[32], &step2[0], &step2[1]);
  btf_16_lane_2_3_neon(step1[14], step1[9], c3, &step2[14], &step2[9]);
  btf_16_lane_0_1_neon(step1[61], step1[34], c2, &step2[61], &step2[34]);
  btf_16_lane_0_1_neon(step1[60], step1[35], c2, &step2[60], &step2[35]);
  btf_16_lane_1_0_neon(step1[36], step1[59], c5, &step2[36], &step2[59]);
  btf_16_lane_1_0_neon(step1[37], step1[58], c5, &step2[37], &step2[58]);
  btf_16_lane_2_3_neon(step1[53], step1[42], c2, &step2[53], &step2[42]);
  btf_16_lane_2_3_neon(step1[52], step1[43], c2, &step2[52], &step2[43]);
  btf_16_lane_3_2_neon(step1[44], step1[51], c5, &step2[44], &step2[51]);
  btf_16_lane_3_2_neon(step1[45], step1[50], c5, &step2[45], &step2[50]);

  step2[8] = step1[8];
  step2[15] = step1[15];
  step2[16] = step1[16];
  step2[17] = step1[17];
  step2[18] = step1[17];
  step2[19] = step1[16];
  step2[20] = step1[23];
  step2[21] = step1[22];
  step2[22] = step1[22];
  step2[23] = step1[23];
  step2[24] = step1[24];
  step2[25] = step1[25];
  step2[26] = step1[25];
  step2[27] = step1[24];
  step2[28] = step1[31];
  step2[29] = step1[30];
  step2[30] = step1[30];
  step2[31] = step1[31];
  step2[32] = step1[32];
  step2[33] = step1[33];
  step2[38] = step1[38];
  step2[39] = step1[39];
  step2[40] = step1[40];
  step2[41] = step1[41];
  step2[46] = step1[46];
  step2[47] = step1[47];
  step2[48] = step1[48];
  step2[49] = step1[49];
  step2[54] = step1[54];
  step2[55] = step1[55];
  step2[56] = step1[56];
  step2[57] = step1[57];
  step2[62] = step1[62];
  step2[63] = step1[63];

  // stage 7

  btf_16_lane_2_3_neon(step2[29], step2[18], c3, &step1[29], &step1[18]);
  btf_16_lane_2_3_neon(step2[28], step2[19], c3, &step1[28], &step1[19]);
  btf_16_lane_3_2_neon(step2[20], step2[27], c6, &step1[20], &step1[27]);
  btf_16_lane_3_2_neon(step2[21], step2[26], c6, &step1[21], &step1[26]);

  step1[0] = step2[0];
  step1[1] = step2[1];
  step1[2] = step2[1];
  step1[3] = step2[0];
  step1[8] = step2[8];
  step1[9] = step2[9];
  step1[10] = step2[9];
  step1[11] = step2[8];
  step1[12] = step2[15];
  step1[13] = step2[14];
  step1[14] = step2[14];
  step1[15] = step2[15];
  step1[16] = step2[16];
  step1[17] = step2[17];
  step1[22] = step2[22];
  step1[23] = step2[23];
  step1[24] = step2[24];
  step1[25] = step2[25];
  step1[30] = step2[30];
  step1[31] = step2[31];
  step1[32] = vqaddq_s16(step2[32], step2[39]);
  step1[33] = vqaddq_s16(step2[33], step2[38]);
  step1[34] = vqaddq_s16(step2[34], step2[37]);
  step1[35] = vqaddq_s16(step2[35], step2[36]);
  step1[36] = vqsubq_s16(step2[35], step2[36]);
  step1[37] = vqsubq_s16(step2[34], step2[37]);
  step1[38] = vqsubq_s16(step2[33], step2[38]);
  step1[39] = vqsubq_s16(step2[32], step2[39]);
  step1[40] = vqsubq_s16(step2[47], step2[40]);
  step1[41] = vqsubq_s16(step2[46], step2[41]);
  step1[42] = vqsubq_s16(step2[45], step2[42]);
  step1[43] = vqsubq_s16(step2[44], step2[43]);
  step1[44] = vqaddq_s16(step2[43], step2[44]);
  step1[45] = vqaddq_s16(step2[42], step2[45]);
  step1[46] = vqaddq_s16(step2[41], step2[46]);
  step1[47] = vqaddq_s16(step2[40], step2[47]);
  step1[48] = vqaddq_s16(step2[48], step2[55]);
  step1[49] = vqaddq_s16(step2[49], step2[54]);
  step1[50] = vqaddq_s16(step2[50], step2[53]);
  step1[51] = vqaddq_s16(step2[51], step2[52]);
  step1[52] = vqsubq_s16(step2[51], step2[52]);
  step1[53] = vqsubq_s16(step2[50], step2[53]);
  step1[54] = vqsubq_s16(step2[49], step2[54]);
  step1[55] = vqsubq_s16(step2[48], step2[55]);
  step1[56] = vqsubq_s16(step2[63], step2[56]);
  step1[57] = vqsubq_s16(step2[62], step2[57]);
  step1[58] = vqsubq_s16(step2[61], step2[58]);
  step1[59] = vqsubq_s16(step2[60], step2[59]);
  step1[60] = vqaddq_s16(step2[59], step2[60]);
  step1[61] = vqaddq_s16(step2[58], step2[61]);
  step1[62] = vqaddq_s16(step2[57], step2[62]);
  step1[63] = vqaddq_s16(step2[56], step2[63]);

  // stage 8

  btf_16_lane_0_1_neon(step1[13], step1[10], c3, &step2[13], &step2[10]);
  btf_16_lane_0_1_neon(step1[12], step1[11], c3, &step2[12], &step2[11]);
  btf_16_lane_2_3_neon(step1[59], step1[36], c3, &step2[59], &step2[36]);
  btf_16_lane_2_3_neon(step1[58], step1[37], c3, &step2[58], &step2[37]);
  btf_16_lane_2_3_neon(step1[57], step1[38], c3, &step2[57], &step2[38]);
  btf_16_lane_2_3_neon(step1[56], step1[39], c3, &step2[56], &step2[39]);
  btf_16_lane_3_2_neon(step1[40], step1[55], c6, &step2[40], &step2[55]);
  btf_16_lane_3_2_neon(step1[41], step1[54], c6, &step2[41], &step2[54]);
  btf_16_lane_3_2_neon(step1[42], step1[53], c6, &step2[42], &step2[53]);
  btf_16_lane_3_2_neon(step1[43], step1[52], c6, &step2[43], &step2[52]);

  step2[0] = step1[0];
  step2[1] = step1[1];
  step2[2] = step1[2];
  step2[3] = step1[3];
  step2[4] = step1[3];
  step2[5] = step1[2];
  step2[6] = step1[1];
  step2[7] = step1[0];
  step2[8] = step1[8];
  step2[9] = step1[9];
  step2[14] = step1[14];
  step2[15] = step1[15];
  step2[16] = vqaddq_s16(step1[16], step1[23]);
  step2[17] = vqaddq_s16(step1[17], step1[22]);
  step2[18] = vqaddq_s16(step1[18], step1[21]);
  step2[19] = vqaddq_s16(step1[19], step1[20]);
  step2[20] = vqsubq_s16(step1[19], step1[20]);
  step2[21] = vqsubq_s16(step1[18], step1[21]);
  step2[22] = vqsubq_s16(step1[17], step1[22]);
  step2[23] = vqsubq_s16(step1[16], step1[23]);
  step2[24] = vqsubq_s16(step1[31], step1[24]);
  step2[25] = vqsubq_s16(step1[30], step1[25]);
  step2[26] = vqsubq_s16(step1[29], step1[26]);
  step2[27] = vqsubq_s16(step1[28], step1[27]);
  step2[28] = vqaddq_s16(step1[28], step1[27]);
  step2[29] = vqaddq_s16(step1[29], step1[26]);
  step2[30] = vqaddq_s16(step1[30], step1[25]);
  step2[31] = vqaddq_s16(step1[31], step1[24]);
  step2[32] = step1[32];
  step2[33] = step1[33];
  step2[34] = step1[34];
  step2[35] = step1[35];
  step2[44] = step1[44];
  step2[45] = step1[45];
  step2[46] = step1[46];
  step2[47] = step1[47];
  step2[48] = step1[48];
  step2[49] = step1[49];
  step2[50] = step1[50];
  step2[51] = step1[51];
  step2[60] = step1[60];
  step2[61] = step1[61];
  step2[62] = step1[62];
  step2[63] = step1[63];

  // stage 9
  idct64_stage9_neon(step2, step1, cos_bit);

  // stage 10
  idct64_stage10_neon(step1, step2, cos_bit);

  // stage 11

  out[0] = vqaddq_s16(step2[0], step2[63]);
  out[1] = vqaddq_s16(step2[1], step2[62]);
  out[2] = vqaddq_s16(step2[2], step2[61]);
  out[3] = vqaddq_s16(step2[3], step2[60]);
  out[4] = vqaddq_s16(step2[4], step2[59]);
  out[5] = vqaddq_s16(step2[5], step2[58]);
  out[6] = vqaddq_s16(step2[6], step2[57]);
  out[7] = vqaddq_s16(step2[7], step2[56]);
  out[8] = vqaddq_s16(step2[8], step2[55]);
  out[9] = vqaddq_s16(step2[9], step2[54]);
  out[10] = vqaddq_s16(step2[10], step2[53]);
  out[11] = vqaddq_s16(step2[11], step2[52]);
  out[12] = vqaddq_s16(step2[12], step2[51]);
  out[13] = vqaddq_s16(step2[13], step2[50]);
  out[14] = vqaddq_s16(step2[14], step2[49]);
  out[15] = vqaddq_s16(step2[15], step2[48]);
  out[16] = vqaddq_s16(step2[16], step2[47]);
  out[17] = vqaddq_s16(step2[17], step2[46]);
  out[18] = vqaddq_s16(step2[18], step2[45]);
  out[19] = vqaddq_s16(step2[19], step2[44]);
  out[20] = vqaddq_s16(step2[20], step2[43]);
  out[21] = vqaddq_s16(step2[21], step2[42]);
  out[22] = vqaddq_s16(step2[22], step2[41]);
  out[23] = vqaddq_s16(step2[23], step2[40]);
  out[24] = vqaddq_s16(step2[24], step2[39]);
  out[25] = vqaddq_s16(step2[25], step2[38]);
  out[26] = vqaddq_s16(step2[26], step2[37]);
  out[27] = vqaddq_s16(step2[27], step2[36]);
  out[28] = vqaddq_s16(step2[28], step2[35]);
  out[29] = vqaddq_s16(step2[29], step2[34]);
  out[30] = vqaddq_s16(step2[30], step2[33]);
  out[31] = vqaddq_s16(step2[31], step2[32]);
  out[32] = vqsubq_s16(step2[31], step2[32]);
  out[33] = vqsubq_s16(step2[30], step2[33]);
  out[34] = vqsubq_s16(step2[29], step2[34]);
  out[35] = vqsubq_s16(step2[28], step2[35]);
  out[36] = vqsubq_s16(step2[27], step2[36]);
  out[37] = vqsubq_s16(step2[26], step2[37]);
  out[38] = vqsubq_s16(step2[25], step2[38]);
  out[39] = vqsubq_s16(step2[24], step2[39]);
  out[40] = vqsubq_s16(step2[23], step2[40]);
  out[41] = vqsubq_s16(step2[22], step2[41]);
  out[42] = vqsubq_s16(step2[21], step2[42]);
  out[43] = vqsubq_s16(step2[20], step2[43]);
  out[44] = vqsubq_s16(step2[19], step2[44]);
  out[45] = vqsubq_s16(step2[18], step2[45]);
  out[46] = vqsubq_s16(step2[17], step2[46]);
  out[47] = vqsubq_s16(step2[16], step2[47]);
  out[48] = vqsubq_s16(step2[15], step2[48]);
  out[49] = vqsubq_s16(step2[14], step2[49]);
  out[50] = vqsubq_s16(step2[13], step2[50]);
  out[51] = vqsubq_s16(step2[12], step2[51]);
  out[52] = vqsubq_s16(step2[11], step2[52]);
  out[53] = vqsubq_s16(step2[10], step2[53]);
  out[54] = vqsubq_s16(step2[9], step2[54]);
  out[55] = vqsubq_s16(step2[8], step2[55]);
  out[56] = vqsubq_s16(step2[7], step2[56]);
  out[57] = vqsubq_s16(step2[6], step2[57]);
  out[58] = vqsubq_s16(step2[5], step2[58]);
  out[59] = vqsubq_s16(step2[4], step2[59]);
  out[60] = vqsubq_s16(step2[3], step2[60]);
  out[61] = vqsubq_s16(step2[2], step2[61]);
  out[62] = vqsubq_s16(step2[1], step2[62]);
  out[63] = vqsubq_s16(step2[0], step2[63]);
}

static inline void idct64_low16_neon(int16x8_t *in, int16x8_t *out,
                                     int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);
  int16x8_t step2[64], step1[64];

  const int16x4_t c0 = set_s16x4_neon((int16_t)cospi[4], (int16_t)cospi[60],
                                      (int16_t)cospi[36], (int16_t)cospi[28]);
  const int16x4_t c1 = set_s16x4_neon((int16_t)cospi[20], (int16_t)cospi[44],
                                      (int16_t)cospi[52], (int16_t)cospi[12]);
  const int16x4_t c2 = set_s16x4_neon((int16_t)cospi[8], (int16_t)cospi[56],
                                      (int16_t)cospi[40], (int16_t)cospi[24]);
  const int16x4_t c3 = set_s16x4_neon((int16_t)cospi[32], (int16_t)cospi[32],
                                      (int16_t)cospi[16], (int16_t)cospi[48]);
  const int16x4_t c4 =
      set_s16x4_neon((int16_t)(-cospi[4]), (int16_t)(-cospi[60]),
                     (int16_t)(-cospi[36]), (int16_t)(-cospi[28]));
  const int16x4_t c5 =
      set_s16x4_neon((int16_t)(-cospi[20]), (int16_t)(-cospi[44]),
                     (int16_t)(-cospi[52]), (int16_t)(-cospi[12]));
  const int16x4_t c6 =
      set_s16x4_neon((int16_t)(-cospi[8]), (int16_t)(-cospi[56]),
                     (int16_t)(-cospi[40]), (int16_t)(-cospi[24]));
  const int16x4_t c7 =
      set_s16x4_neon((int16_t)(-cospi[32]), (int16_t)(-cospi[32]),
                     (int16_t)(-cospi[16]), (int16_t)(-cospi[48]));

  // stage 1
  // stage 2

  step2[0] = in[0];
  step2[4] = in[8];
  step2[8] = in[4];
  step2[12] = in[12];
  step2[16] = in[2];
  step2[20] = in[10];
  step2[24] = in[6];
  step2[28] = in[14];

  btf_16_neon(in[1], cospi[63], cospi[1], &step2[32], &step2[63]);
  btf_16_neon(in[15], -cospi[49], cospi[15], &step2[35], &step2[60]);
  btf_16_neon(in[9], cospi[55], cospi[9], &step2[36], &step2[59]);
  btf_16_neon(in[7], -cospi[57], cospi[7], &step2[39], &step2[56]);
  btf_16_neon(in[5], cospi[59], cospi[5], &step2[40], &step2[55]);
  btf_16_neon(in[11], -cospi[53], cospi[11], &step2[43], &step2[52]);
  btf_16_neon(in[13], cospi[51], cospi[13], &step2[44], &step2[51]);
  btf_16_neon(in[3], -cospi[61], cospi[3], &step2[47], &step2[48]);

  // stage 3

  step1[0] = step2[0];
  step1[4] = step2[4];
  step1[8] = step2[8];
  step1[12] = step2[12];

  btf_16_neon(step2[16], cospi[62], cospi[2], &step1[16], &step1[31]);
  btf_16_neon(step2[20], cospi[54], cospi[10], &step1[20], &step1[27]);
  btf_16_neon(step2[24], -cospi[58], cospi[6], &step1[23], &step1[24]);
  btf_16_neon(step2[28], -cospi[50], cospi[14], &step1[19], &step1[28]);

  step1[32] = step2[32];
  step1[33] = step2[32];
  step1[34] = step2[35];
  step1[35] = step2[35];
  step1[36] = step2[36];
  step1[37] = step2[36];
  step1[38] = step2[39];
  step1[39] = step2[39];
  step1[40] = step2[40];
  step1[41] = step2[40];
  step1[42] = step2[43];
  step1[43] = step2[43];
  step1[44] = step2[44];
  step1[45] = step2[44];
  step1[46] = step2[47];
  step1[47] = step2[47];
  step1[48] = step2[48];
  step1[49] = step2[48];
  step1[50] = step2[51];
  step1[51] = step2[51];
  step1[52] = step2[52];
  step1[53] = step2[52];
  step1[54] = step2[55];
  step1[55] = step2[55];
  step1[56] = step2[56];
  step1[57] = step2[56];
  step1[58] = step2[59];
  step1[59] = step2[59];
  step1[60] = step2[60];
  step1[61] = step2[60];
  step1[62] = step2[63];
  step1[63] = step2[63];

  // stage 4

  step2[0] = step1[0];
  step2[4] = step1[4];

  btf_16_neon(step1[8], cospi[60], cospi[4], &step2[8], &step2[15]);
  btf_16_neon(step1[12], -cospi[52], cospi[12], &step2[11], &step2[12]);
  btf_16_lane_0_1_neon(step1[62], step1[33], c0, &step2[62], &step2[33]);
  btf_16_lane_1_0_neon(step1[34], step1[61], c4, &step2[34], &step2[61]);
  btf_16_lane_2_3_neon(step1[58], step1[37], c0, &step2[58], &step2[37]);
  btf_16_lane_3_2_neon(step1[38], step1[57], c4, &step2[38], &step2[57]);
  btf_16_lane_0_1_neon(step1[54], step1[41], c1, &step2[54], &step2[41]);
  btf_16_lane_1_0_neon(step1[42], step1[53], c5, &step2[42], &step2[53]);
  btf_16_lane_2_3_neon(step1[50], step1[45], c1, &step2[50], &step2[45]);
  btf_16_lane_3_2_neon(step1[46], step1[49], c5, &step2[46], &step2[49]);

  step2[16] = step1[16];
  step2[17] = step1[16];
  step2[18] = step1[19];
  step2[19] = step1[19];
  step2[20] = step1[20];
  step2[21] = step1[20];
  step2[22] = step1[23];
  step2[23] = step1[23];
  step2[24] = step1[24];
  step2[25] = step1[24];
  step2[26] = step1[27];
  step2[27] = step1[27];
  step2[28] = step1[28];
  step2[29] = step1[28];
  step2[30] = step1[31];
  step2[31] = step1[31];
  step2[32] = step1[32];
  step2[35] = step1[35];
  step2[36] = step1[36];
  step2[39] = step1[39];
  step2[40] = step1[40];
  step2[43] = step1[43];
  step2[44] = step1[44];
  step2[47] = step1[47];
  step2[48] = step1[48];
  step2[51] = step1[51];
  step2[52] = step1[52];
  step2[55] = step1[55];
  step2[56] = step1[56];
  step2[59] = step1[59];
  step2[60] = step1[60];
  step2[63] = step1[63];

  // stage 5

  step1[0] = step2[0];

  btf_16_neon(step2[4], cospi[56], cospi[8], &step1[4], &step1[7]);
  btf_16_lane_0_1_neon(step2[30], step2[17], c2, &step1[30], &step1[17]);
  btf_16_lane_1_0_neon(step2[18], step2[29], c6, &step1[18], &step1[29]);
  btf_16_lane_2_3_neon(step2[26], step2[21], c2, &step1[26], &step1[21]);
  btf_16_lane_3_2_neon(step2[22], step2[25], c6, &step1[22], &step1[25]);

  step1[8] = step2[8];
  step1[9] = step2[8];
  step1[10] = step2[11];
  step1[11] = step2[11];
  step1[12] = step2[12];
  step1[13] = step2[12];
  step1[14] = step2[15];
  step1[15] = step2[15];
  step1[16] = step2[16];
  step1[19] = step2[19];
  step1[20] = step2[20];
  step1[23] = step2[23];
  step1[24] = step2[24];
  step1[27] = step2[27];
  step1[28] = step2[28];
  step1[31] = step2[31];
  step1[32] = vqaddq_s16(step2[32], step2[35]);
  step1[33] = vqaddq_s16(step2[33], step2[34]);
  step1[34] = vqsubq_s16(step2[33], step2[34]);
  step1[35] = vqsubq_s16(step2[32], step2[35]);
  step1[36] = vqsubq_s16(step2[39], step2[36]);
  step1[37] = vqsubq_s16(step2[38], step2[37]);
  step1[38] = vqaddq_s16(step2[38], step2[37]);
  step1[39] = vqaddq_s16(step2[39], step2[36]);
  step1[40] = vqaddq_s16(step2[40], step2[43]);
  step1[41] = vqaddq_s16(step2[41], step2[42]);
  step1[42] = vqsubq_s16(step2[41], step2[42]);
  step1[43] = vqsubq_s16(step2[40], step2[43]);
  step1[44] = vqsubq_s16(step2[47], step2[44]);
  step1[45] = vqsubq_s16(step2[46], step2[45]);
  step1[46] = vqaddq_s16(step2[46], step2[45]);
  step1[47] = vqaddq_s16(step2[47], step2[44]);
  step1[48] = vqaddq_s16(step2[48], step2[51]);
  step1[49] = vqaddq_s16(step2[49], step2[50]);
  step1[50] = vqsubq_s16(step2[49], step2[50]);
  step1[51] = vqsubq_s16(step2[48], step2[51]);
  step1[52] = vqsubq_s16(step2[55], step2[52]);
  step1[53] = vqsubq_s16(step2[54], step2[53]);
  step1[54] = vqaddq_s16(step2[54], step2[53]);
  step1[55] = vqaddq_s16(step2[55], step2[52]);
  step1[56] = vqaddq_s16(step2[56], step2[59]);
  step1[57] = vqaddq_s16(step2[57], step2[58]);
  step1[58] = vqsubq_s16(step2[57], step2[58]);
  step1[59] = vqsubq_s16(step2[56], step2[59]);
  step1[60] = vqsubq_s16(step2[63], step2[60]);
  step1[61] = vqsubq_s16(step2[62], step2[61]);
  step1[62] = vqaddq_s16(step2[62], step2[61]);
  step1[63] = vqaddq_s16(step2[63], step2[60]);

  // stage 6

  btf_16_neon(step1[0], cospi[32], cospi[32], &step2[0], &step2[1]);
  btf_16_lane_2_3_neon(step1[14], step1[9], c3, &step2[14], &step2[9]);
  btf_16_lane_3_2_neon(step1[10], step1[13], c7, &step2[10], &step2[13]);
  btf_16_lane_0_1_neon(step1[61], step1[34], c2, &step2[61], &step2[34]);
  btf_16_lane_0_1_neon(step1[60], step1[35], c2, &step2[60], &step2[35]);
  btf_16_lane_1_0_neon(step1[36], step1[59], c6, &step2[36], &step2[59]);
  btf_16_lane_1_0_neon(step1[37], step1[58], c6, &step2[37], &step2[58]);
  btf_16_lane_2_3_neon(step1[53], step1[42], c2, &step2[53], &step2[42]);
  btf_16_lane_2_3_neon(step1[52], step1[43], c2, &step2[52], &step2[43]);
  btf_16_lane_3_2_neon(step1[44], step1[51], c6, &step2[44], &step2[51]);
  btf_16_lane_3_2_neon(step1[45], step1[50], c6, &step2[45], &step2[50]);

  step2[4] = step1[4];
  step2[5] = step1[4];
  step2[6] = step1[7];
  step2[7] = step1[7];
  step2[8] = step1[8];
  step2[11] = step1[11];
  step2[12] = step1[12];
  step2[15] = step1[15];
  step2[16] = vqaddq_s16(step1[16], step1[19]);
  step2[17] = vqaddq_s16(step1[17], step1[18]);
  step2[18] = vqsubq_s16(step1[17], step1[18]);
  step2[19] = vqsubq_s16(step1[16], step1[19]);
  step2[20] = vqsubq_s16(step1[23], step1[20]);
  step2[21] = vqsubq_s16(step1[22], step1[21]);
  step2[22] = vqaddq_s16(step1[22], step1[21]);
  step2[23] = vqaddq_s16(step1[23], step1[20]);
  step2[24] = vqaddq_s16(step1[24], step1[27]);
  step2[25] = vqaddq_s16(step1[25], step1[26]);
  step2[26] = vqsubq_s16(step1[25], step1[26]);
  step2[27] = vqsubq_s16(step1[24], step1[27]);
  step2[28] = vqsubq_s16(step1[31], step1[28]);
  step2[29] = vqsubq_s16(step1[30], step1[29]);
  step2[30] = vqaddq_s16(step1[30], step1[29]);
  step2[31] = vqaddq_s16(step1[31], step1[28]);
  step2[32] = step1[32];
  step2[33] = step1[33];
  step2[38] = step1[38];
  step2[39] = step1[39];
  step2[40] = step1[40];
  step2[41] = step1[41];
  step2[46] = step1[46];
  step2[47] = step1[47];
  step2[48] = step1[48];
  step2[49] = step1[49];
  step2[54] = step1[54];
  step2[55] = step1[55];
  step2[56] = step1[56];
  step2[57] = step1[57];
  step2[62] = step1[62];
  step2[63] = step1[63];

  // stage 7

  btf_16_lane_0_1_neon(step2[6], step2[5], c3, &step1[6], &step1[5]);
  btf_16_lane_2_3_neon(step2[29], step2[18], c3, &step1[29], &step1[18]);
  btf_16_lane_2_3_neon(step2[28], step2[19], c3, &step1[28], &step1[19]);
  btf_16_lane_3_2_neon(step2[20], step2[27], c7, &step1[20], &step1[27]);
  btf_16_lane_3_2_neon(step2[21], step2[26], c7, &step1[21], &step1[26]);

  step1[0] = step2[0];
  step1[1] = step2[1];
  step1[2] = step2[1];
  step1[3] = step2[0];
  step1[4] = step2[4];
  step1[7] = step2[7];
  step1[8] = vqaddq_s16(step2[8], step2[11]);
  step1[9] = vqaddq_s16(step2[9], step2[10]);
  step1[10] = vqsubq_s16(step2[9], step2[10]);
  step1[11] = vqsubq_s16(step2[8], step2[11]);
  step1[12] = vqsubq_s16(step2[15], step2[12]);
  step1[13] = vqsubq_s16(step2[14], step2[13]);
  step1[14] = vqaddq_s16(step2[14], step2[13]);
  step1[15] = vqaddq_s16(step2[15], step2[12]);
  step1[16] = step2[16];
  step1[17] = step2[17];
  step1[22] = step2[22];
  step1[23] = step2[23];
  step1[24] = step2[24];
  step1[25] = step2[25];
  step1[30] = step2[30];
  step1[31] = step2[31];
  step1[32] = vqaddq_s16(step2[32], step2[39]);
  step1[33] = vqaddq_s16(step2[33], step2[38]);
  step1[34] = vqaddq_s16(step2[34], step2[37]);
  step1[35] = vqaddq_s16(step2[35], step2[36]);
  step1[36] = vqsubq_s16(step2[35], step2[36]);
  step1[37] = vqsubq_s16(step2[34], step2[37]);
  step1[38] = vqsubq_s16(step2[33], step2[38]);
  step1[39] = vqsubq_s16(step2[32], step2[39]);
  step1[40] = vqsubq_s16(step2[47], step2[40]);
  step1[41] = vqsubq_s16(step2[46], step2[41]);
  step1[42] = vqsubq_s16(step2[45], step2[42]);
  step1[43] = vqsubq_s16(step2[44], step2[43]);
  step1[44] = vqaddq_s16(step2[43], step2[44]);
  step1[45] = vqaddq_s16(step2[42], step2[45]);
  step1[46] = vqaddq_s16(step2[41], step2[46]);
  step1[47] = vqaddq_s16(step2[40], step2[47]);
  step1[48] = vqaddq_s16(step2[48], step2[55]);
  step1[49] = vqaddq_s16(step2[49], step2[54]);
  step1[50] = vqaddq_s16(step2[50], step2[53]);
  step1[51] = vqaddq_s16(step2[51], step2[52]);
  step1[52] = vqsubq_s16(step2[51], step2[52]);
  step1[53] = vqsubq_s16(step2[50], step2[53]);
  step1[54] = vqsubq_s16(step2[49], step2[54]);
  step1[55] = vqsubq_s16(step2[48], step2[55]);
  step1[56] = vqsubq_s16(step2[63], step2[56]);
  step1[57] = vqsubq_s16(step2[62], step2[57]);
  step1[58] = vqsubq_s16(step2[61], step2[58]);
  step1[59] = vqsubq_s16(step2[60], step2[59]);
  step1[60] = vqaddq_s16(step2[59], step2[60]);
  step1[61] = vqaddq_s16(step2[58], step2[61]);
  step1[62] = vqaddq_s16(step2[57], step2[62]);
  step1[63] = vqaddq_s16(step2[56], step2[63]);

  // stage 8

  btf_16_lane_0_1_neon(step1[13], step1[10], c3, &step2[13], &step2[10]);
  btf_16_lane_0_1_neon(step1[12], step1[11], c3, &step2[12], &step2[11]);
  btf_16_lane_2_3_neon(step1[59], step1[36], c3, &step2[59], &step2[36]);
  btf_16_lane_2_3_neon(step1[58], step1[37], c3, &step2[58], &step2[37]);
  btf_16_lane_2_3_neon(step1[57], step1[38], c3, &step2[57], &step2[38]);
  btf_16_lane_2_3_neon(step1[56], step1[39], c3, &step2[56], &step2[39]);
  btf_16_lane_3_2_neon(step1[40], step1[55], c7, &step2[40], &step2[55]);
  btf_16_lane_3_2_neon(step1[41], step1[54], c7, &step2[41], &step2[54]);
  btf_16_lane_3_2_neon(step1[42], step1[53], c7, &step2[42], &step2[53]);
  btf_16_lane_3_2_neon(step1[43], step1[52], c7, &step2[43], &step2[52]);

  step2[0] = vqaddq_s16(step1[0], step1[7]);
  step2[1] = vqaddq_s16(step1[1], step1[6]);
  step2[2] = vqaddq_s16(step1[2], step1[5]);
  step2[3] = vqaddq_s16(step1[3], step1[4]);
  step2[4] = vqsubq_s16(step1[3], step1[4]);
  step2[5] = vqsubq_s16(step1[2], step1[5]);
  step2[6] = vqsubq_s16(step1[1], step1[6]);
  step2[7] = vqsubq_s16(step1[0], step1[7]);
  step2[8] = step1[8];
  step2[9] = step1[9];
  step2[14] = step1[14];
  step2[15] = step1[15];
  step2[16] = vqaddq_s16(step1[16], step1[23]);
  step2[17] = vqaddq_s16(step1[17], step1[22]);
  step2[18] = vqaddq_s16(step1[18], step1[21]);
  step2[19] = vqaddq_s16(step1[19], step1[20]);
  step2[20] = vqsubq_s16(step1[19], step1[20]);
  step2[21] = vqsubq_s16(step1[18], step1[21]);
  step2[22] = vqsubq_s16(step1[17], step1[22]);
  step2[23] = vqsubq_s16(step1[16], step1[23]);
  step2[24] = vqsubq_s16(step1[31], step1[24]);
  step2[25] = vqsubq_s16(step1[30], step1[25]);
  step2[26] = vqsubq_s16(step1[29], step1[26]);
  step2[27] = vqsubq_s16(step1[28], step1[27]);
  step2[28] = vqaddq_s16(step1[28], step1[27]);
  step2[29] = vqaddq_s16(step1[29], step1[26]);
  step2[30] = vqaddq_s16(step1[30], step1[25]);
  step2[31] = vqaddq_s16(step1[31], step1[24]);
  step2[32] = step1[32];
  step2[33] = step1[33];
  step2[34] = step1[34];
  step2[35] = step1[35];
  step2[44] = step1[44];
  step2[45] = step1[45];
  step2[46] = step1[46];
  step2[47] = step1[47];
  step2[48] = step1[48];
  step2[49] = step1[49];
  step2[50] = step1[50];
  step2[51] = step1[51];
  step2[60] = step1[60];
  step2[61] = step1[61];
  step2[62] = step1[62];
  step2[63] = step1[63];

  // stage 9
  idct64_stage9_neon(step2, step1, cos_bit);

  // stage 10
  idct64_stage10_neon(step1, step2, cos_bit);

  // stage 11

  out[0] = vqaddq_s16(step2[0], step2[63]);
  out[1] = vqaddq_s16(step2[1], step2[62]);
  out[2] = vqaddq_s16(step2[2], step2[61]);
  out[3] = vqaddq_s16(step2[3], step2[60]);
  out[4] = vqaddq_s16(step2[4], step2[59]);
  out[5] = vqaddq_s16(step2[5], step2[58]);
  out[6] = vqaddq_s16(step2[6], step2[57]);
  out[7] = vqaddq_s16(step2[7], step2[56]);
  out[8] = vqaddq_s16(step2[8], step2[55]);
  out[9] = vqaddq_s16(step2[9], step2[54]);
  out[10] = vqaddq_s16(step2[10], step2[53]);
  out[11] = vqaddq_s16(step2[11], step2[52]);
  out[12] = vqaddq_s16(step2[12], step2[51]);
  out[13] = vqaddq_s16(step2[13], step2[50]);
  out[14] = vqaddq_s16(step2[14], step2[49]);
  out[15] = vqaddq_s16(step2[15], step2[48]);
  out[16] = vqaddq_s16(step2[16], step2[47]);
  out[17] = vqaddq_s16(step2[17], step2[46]);
  out[18] = vqaddq_s16(step2[18], step2[45]);
  out[19] = vqaddq_s16(step2[19], step2[44]);
  out[20] = vqaddq_s16(step2[20], step2[43]);
  out[21] = vqaddq_s16(step2[21], step2[42]);
  out[22] = vqaddq_s16(step2[22], step2[41]);
  out[23] = vqaddq_s16(step2[23], step2[40]);
  out[24] = vqaddq_s16(step2[24], step2[39]);
  out[25] = vqaddq_s16(step2[25], step2[38]);
  out[26] = vqaddq_s16(step2[26], step2[37]);
  out[27] = vqaddq_s16(step2[27], step2[36]);
  out[28] = vqaddq_s16(step2[28], step2[35]);
  out[29] = vqaddq_s16(step2[29], step2[34]);
  out[30] = vqaddq_s16(step2[30], step2[33]);
  out[31] = vqaddq_s16(step2[31], step2[32]);
  out[32] = vqsubq_s16(step2[31], step2[32]);
  out[33] = vqsubq_s16(step2[30], step2[33]);
  out[34] = vqsubq_s16(step2[29], step2[34]);
  out[35] = vqsubq_s16(step2[28], step2[35]);
  out[36] = vqsubq_s16(step2[27], step2[36]);
  out[37] = vqsubq_s16(step2[26], step2[37]);
  out[38] = vqsubq_s16(step2[25], step2[38]);
  out[39] = vqsubq_s16(step2[24], step2[39]);
  out[40] = vqsubq_s16(step2[23], step2[40]);
  out[41] = vqsubq_s16(step2[22], step2[41]);
  out[42] = vqsubq_s16(step2[21], step2[42]);
  out[43] = vqsubq_s16(step2[20], step2[43]);
  out[44] = vqsubq_s16(step2[19], step2[44]);
  out[45] = vqsubq_s16(step2[18], step2[45]);
  out[46] = vqsubq_s16(step2[17], step2[46]);
  out[47] = vqsubq_s16(step2[16], step2[47]);
  out[48] = vqsubq_s16(step2[15], step2[48]);
  out[49] = vqsubq_s16(step2[14], step2[49]);
  out[50] = vqsubq_s16(step2[13], step2[50]);
  out[51] = vqsubq_s16(step2[12], step2[51]);
  out[52] = vqsubq_s16(step2[11], step2[52]);
  out[53] = vqsubq_s16(step2[10], step2[53]);
  out[54] = vqsubq_s16(step2[9], step2[54]);
  out[55] = vqsubq_s16(step2[8], step2[55]);
  out[56] = vqsubq_s16(step2[7], step2[56]);
  out[57] = vqsubq_s16(step2[6], step2[57]);
  out[58] = vqsubq_s16(step2[5], step2[58]);
  out[59] = vqsubq_s16(step2[4], step2[59]);
  out[60] = vqsubq_s16(step2[3], step2[60]);
  out[61] = vqsubq_s16(step2[2], step2[61]);
  out[62] = vqsubq_s16(step2[1], step2[62]);
  out[63] = vqsubq_s16(step2[0], step2[63]);
}

// Functions for blocks with eob at DC and within
// topleft 8x8, 16x16, 32x32 corner
static const transform_neon
    lowbd_txfm_all_1d_zeros_w_arr[TX_SIZES][ITX_TYPES_1D][4] = {
      {
          { NULL, NULL, NULL, NULL },
          { NULL, NULL, NULL, NULL },
          { NULL, NULL, NULL, NULL },
      },
      { { idct8_low1_neon, idct8_neon, NULL, NULL },
        { iadst8_low1_neon, iadst8_neon, NULL, NULL },
        { NULL, NULL, NULL, NULL } },
      {
          { idct16_low1_neon, idct16_low8_neon, idct16_neon, NULL },
          { iadst16_low1_neon, iadst16_low8_neon, iadst16_neon, NULL },
          { NULL, NULL, NULL, NULL },
      },
      { { idct32_low1_neon, idct32_low8_neon, idct32_low16_neon, idct32_neon },
        { NULL, NULL, NULL, NULL },
        { NULL, NULL, NULL, NULL } },
      { { idct64_low1_neon, idct64_low8_neon, idct64_low16_neon,
          idct64_low32_neon },
        { NULL, NULL, NULL, NULL },
        { NULL, NULL, NULL, NULL } }
    };

static inline void lowbd_inv_txfm2d_add_idtx_neon(const int32_t *input,
                                                  uint8_t *output, int stride,
                                                  TX_TYPE tx_type,
                                                  TX_SIZE tx_size, int eob) {
  (void)tx_type;
  int16x8_t a[32 * 4];
  int16x8_t b[32 * 4];
  int eobx, eoby;
  get_eobx_eoby_scan_default(&eobx, &eoby, tx_size, eob);
  const int8_t *shift = av1_inv_txfm_shift_ls[tx_size];
  const int txw_idx = get_txw_idx(tx_size);
  const int txh_idx = get_txh_idx(tx_size);
  const int txfm_size_col = tx_size_wide[tx_size];
  const int txfm_size_row = tx_size_high[tx_size];
  lowbd_inv_txfm2d_memset_neon(&a[0], (txfm_size_col * (txfm_size_row) >> 3),
                               0);
  lowbd_inv_txfm2d_memset_neon(&b[0], (txfm_size_col * (txfm_size_row) >> 3),
                               0);
  const int buf_size_w_div8 = txfm_size_col >> 3;
  const int rect_type = get_rect_tx_log_ratio(txfm_size_col, txfm_size_row);
  const int buf_size_nonzero_h_div8 = (eoby + 8) >> 3;
  const int buf_size_nonzero_w = (eobx + 8) >> 3 << 3;
  const int input_stride = txfm_size_row;
  int temp_b = 0;

  for (int i = 0; i < buf_size_nonzero_h_div8; i++) {
    int16x8_t *cur_a = &a[i * txfm_size_col];
    load_buffer_32bit_to_16bit_neon(input, input_stride, cur_a,
                                    buf_size_nonzero_w);
    input += 8;
    if (abs(rect_type) == 1) {
      round_shift_for_rect(cur_a, cur_a, buf_size_nonzero_w);
    }
    identity_txfm_round_neon(cur_a, cur_a, txw_idx, buf_size_nonzero_w,
                             -shift[0]);
    for (int j = 0; j < buf_size_w_div8; ++j) {
      transpose_arrays_s16_8x8(&cur_a[j * 8], &b[temp_b + txfm_size_row * j]);
    }
    temp_b += 8;
  }
  for (int j = 0; j < buf_size_w_div8; ++j) {
    identity_txfm_round_neon(&b[j * txfm_size_row], &b[j * txfm_size_row],
                             txh_idx, txfm_size_row, -shift[1]);
  }
  if (txfm_size_col >= 16) {
    for (int i = 0; i < (txfm_size_col >> 4); i++) {
      lowbd_add_flip_buffer_16xn_neon(
          &b[i * txfm_size_row * 2], output + 16 * i, stride, 0, txfm_size_row);
    }
  } else if (txfm_size_col == 8) {
    lowbd_add_flip_buffer_8xn_neon(b, output, stride, 0, txfm_size_row);
  }
}

static inline void lowbd_inv_txfm2d_add_v_identity_neon(
    const int32_t *input, uint8_t *output, int stride, TX_TYPE tx_type,
    TX_SIZE tx_size, int eob) {
  int16x8_t a[16 * 2];
  int16x8_t b[16 * 2];
  int eobx, eoby, ud_flip, lr_flip;
  get_eobx_eoby_scan_v_identity(&eobx, &eoby, tx_size, eob);
  const int8_t *shift = av1_inv_txfm_shift_ls[tx_size];
  const int txw_idx = get_txw_idx(tx_size);
  const int txh_idx = get_txh_idx(tx_size);
  const int txfm_size_col = tx_size_wide[tx_size];
  const int txfm_size_row = tx_size_high[tx_size];
  lowbd_inv_txfm2d_memset_neon(&b[0], (txfm_size_col * (txfm_size_row) >> 3),
                               0);
  const int rect_type = get_rect_tx_log_ratio(txfm_size_col, txfm_size_row);
  const int buf_size_w_div8 = txfm_size_col >> 3;
  const int buf_size_nonzero_h_div8 = (eoby + 8) >> 3;
  const int buf_size_nonzero_w = (eobx + 8) >> 3 << 3;
  const int input_stride = txfm_size_row;
  const int fun_idx_x = lowbd_txfm_all_1d_zeros_idx[eobx];
  int temp_b = 0;
  const transform_neon row_txfm =
      lowbd_txfm_all_1d_zeros_w_arr[txw_idx][hitx_1d_tab[tx_type]][fun_idx_x];

  assert(row_txfm != NULL);

  get_flip_cfg(tx_type, &ud_flip, &lr_flip);

  for (int i = 0; i < buf_size_nonzero_h_div8; i++) {
    int16x8_t *cur_a = &a[i * txfm_size_col];
    load_buffer_32bit_to_16bit_neon(input, input_stride, cur_a,
                                    buf_size_nonzero_w);
    input += 8;
    if (abs(rect_type) == 1) {
      round_shift_for_rect(cur_a, cur_a, buf_size_nonzero_w);
    }
    row_txfm(cur_a, cur_a, INV_COS_BIT);
    round_shift_array_16_neon(cur_a, txfm_size_col, -shift[0]);
    if (lr_flip == 1) {
      for (int j = 0; j < buf_size_w_div8; ++j) {
        flip_buf_ud_neon(&cur_a[j * 8], 8);
        transpose_arrays_s16_8x8(
            &cur_a[j * 8],
            &b[temp_b + txfm_size_row * (buf_size_w_div8 - 1 - j)]);
      }
      temp_b += 8;
    } else {
      for (int j = 0; j < buf_size_w_div8; ++j) {
        transpose_arrays_s16_8x8(&cur_a[j * 8], &b[temp_b + txfm_size_row * j]);
      }
      temp_b += 8;
    }
  }
  for (int j = 0; j < buf_size_w_div8; ++j) {
    identity_txfm_round_neon(&b[j * txfm_size_row], &b[j * txfm_size_row],
                             txh_idx, txfm_size_row, -shift[1]);
  }
  if (txfm_size_col >= 16) {
    for (int i = 0; i < (txfm_size_col >> 4); i++) {
      lowbd_add_flip_buffer_16xn_neon(
          &b[i * txfm_size_row * 2], output + 16 * i, stride, 0, txfm_size_row);
    }
  } else if (txfm_size_col == 8) {
    lowbd_add_flip_buffer_8xn_neon(b, output, stride, 0, txfm_size_row);
  }
}

static inline void lowbd_inv_txfm2d_add_h_identity_neon(
    const int32_t *input, uint8_t *output, int stride, TX_TYPE tx_type,
    TX_SIZE tx_size, int eob) {
  int16x8_t a[16 * 2];
  int16x8_t b[16 * 2];
  int eobx, eoby, ud_flip, lr_flip;
  get_eobx_eoby_scan_h_identity(&eobx, &eoby, tx_size, eob);
  const int8_t *shift = av1_inv_txfm_shift_ls[tx_size];
  const int txw_idx = get_txw_idx(tx_size);
  const int txh_idx = get_txh_idx(tx_size);
  const int txfm_size_col = tx_size_wide[tx_size];
  const int txfm_size_row = tx_size_high[tx_size];
  lowbd_inv_txfm2d_memset_neon(&a[0], (txfm_size_col * (txfm_size_row) >> 3),
                               0);
  const int buf_size_w_div8 = txfm_size_col >> 3;
  const int rect_type = get_rect_tx_log_ratio(txfm_size_col, txfm_size_row);
  const int buf_size_nonzero_h_div8 = (eoby + 8) >> 3;
  const int buf_size_nonzero_w = (eobx + 8) >> 3 << 3;
  const int input_stride = txfm_size_row;
  const int fun_idx_y = lowbd_txfm_all_1d_zeros_idx[eoby];
  int temp_b = 0;
  const transform_neon col_txfm =
      lowbd_txfm_all_1d_zeros_w_arr[txh_idx][vitx_1d_tab[tx_type]][fun_idx_y];

  assert(col_txfm != NULL);

  get_flip_cfg(tx_type, &ud_flip, &lr_flip);

  for (int i = 0; i < buf_size_nonzero_h_div8; i++) {
    int16x8_t *cur_a = &a[i * txfm_size_col];
    load_buffer_32bit_to_16bit_neon(input, input_stride, cur_a,
                                    buf_size_nonzero_w);
    input += 8;
    if (abs(rect_type) == 1) {
      round_shift_for_rect(cur_a, cur_a, buf_size_nonzero_w);
    }
    identity_txfm_round_neon(cur_a, cur_a, txw_idx, buf_size_nonzero_w,
                             -shift[0]);
    for (int j = 0; j < buf_size_w_div8; ++j) {
      transpose_arrays_s16_8x8(&cur_a[j * 8], &b[temp_b + txfm_size_row * j]);
    }
    temp_b += 8;
  }
  for (int j = 0; j < buf_size_w_div8; ++j) {
    col_txfm(&b[j * txfm_size_row], &b[j * txfm_size_row], INV_COS_BIT);
    round_shift_array_16_neon(&b[j * txfm_size_row], txfm_size_row, -shift[1]);
  }
  if (txfm_size_col >= 16) {
    for (int i = 0; i < (txfm_size_col >> 4); i++) {
      lowbd_add_flip_buffer_16xn_neon(&b[i * txfm_size_row * 2],
                                      output + 16 * i, stride, ud_flip,
                                      txfm_size_row);
    }
  } else if (txfm_size_col == 8) {
    lowbd_add_flip_buffer_8xn_neon(b, output, stride, ud_flip, txfm_size_row);
  }
}

static inline void lowbd_add_flip_buffer_4xn_neon(const int16x8_t *in,
                                                  uint8_t *output, int stride,
                                                  int flipud, int height) {
  assert(height >= 2 && (height & 1) == 0);
  int j = flipud ? (height - 1) : 0;
  const int step = flipud ? -1 : 1;
  do {
    const uint8x8_t pred = load_u8_4x2(output, stride);
    const int16x8_t res =
        vcombine_s16(vget_low_s16(in[j]), vget_low_s16(in[j + step]));
    const int16x8_t sum =
        vqaddq_s16(vreinterpretq_s16_u16(vmovl_u8(pred)), res);
    store_u8x4_strided_x2(output, stride, vqmovun_s16(sum));
    output += 2 * stride;
    j += 2 * step;
    height -= 2;
  } while (height != 0);
}

static inline void lowbd_inv_txfm_dc_only_4xn_neon(uint8_t *output, int stride,
                                                   int32_t dc, int height) {
  assert(height >= 2 && (height & 1) == 0);
  const int16_t clamped_dc = (int16_t)clamp(dc, -255, 255);
  const int16x8_t dc_v = vdupq_n_s16(clamped_dc);
  do {
    const uint8x8_t pred = load_u8_4x2(output, stride);
    const int16x8_t sum =
        vaddq_s16(vreinterpretq_s16_u16(vmovl_u8(pred)), dc_v);
    store_u8x4_strided_x2(output, stride, vqmovun_s16(sum));
    output += 2 * stride;
    height -= 2;
  } while (height != 0);
}

static AOM_FORCE_INLINE int16x8_t narrow_inv_txfm4_out_neon(int32x4_t lo,
                                                            int32x4_t hi,
                                                            int out_rshift) {
  if (out_rshift == 4) {
    return vcombine_s16(vqrshrn_n_s32(lo, 4), vqrshrn_n_s32(hi, 4));
  }
  if (out_rshift == 1) {
    return vcombine_s16(vqrshrn_n_s32(lo, 1), vqrshrn_n_s32(hi, 1));
  }
  assert(out_rshift == 0);
  return vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi));
}

static AOM_FORCE_INLINE void idct4_s16x8_neon(const int16x8_t *in,
                                              int16x8_t *out, int out_rshift) {
  const int32_t *cospi = cospi_arr(INV_COS_BIT);
  const int32x4_t x0_lo = vmovl_s16(vget_low_s16(in[0]));
  const int32x4_t x0_hi = vmovl_s16(vget_high_s16(in[0]));
  const int32x4_t x2_lo = vmovl_s16(vget_low_s16(in[2]));
  const int32x4_t x2_hi = vmovl_s16(vget_high_s16(in[2]));
  const int16x4_t x1_lo = vget_low_s16(in[1]);
  const int16x4_t x1_hi = vget_high_s16(in[1]);
  const int16x4_t x3_lo = vget_low_s16(in[3]);
  const int16x4_t x3_hi = vget_high_s16(in[3]);

  const int32x4_t s0_lo = vrshrq_n_s32(
      vmulq_n_s32(vaddq_s32(x0_lo, x2_lo), cospi[32]), INV_COS_BIT);
  const int32x4_t s0_hi = vrshrq_n_s32(
      vmulq_n_s32(vaddq_s32(x0_hi, x2_hi), cospi[32]), INV_COS_BIT);
  const int32x4_t s1_lo = vrshrq_n_s32(
      vmulq_n_s32(vsubq_s32(x0_lo, x2_lo), cospi[32]), INV_COS_BIT);
  const int32x4_t s1_hi = vrshrq_n_s32(
      vmulq_n_s32(vsubq_s32(x0_hi, x2_hi), cospi[32]), INV_COS_BIT);

  const int32x4_t s2_lo =
      vrshrq_n_s32(vmlsl_n_s16(vmull_n_s16(x1_lo, (int16_t)cospi[48]), x3_lo,
                               (int16_t)cospi[16]),
                   INV_COS_BIT);
  const int32x4_t s2_hi =
      vrshrq_n_s32(vmlsl_n_s16(vmull_n_s16(x1_hi, (int16_t)cospi[48]), x3_hi,
                               (int16_t)cospi[16]),
                   INV_COS_BIT);
  const int32x4_t s3_lo =
      vrshrq_n_s32(vmlal_n_s16(vmull_n_s16(x1_lo, (int16_t)cospi[16]), x3_lo,
                               (int16_t)cospi[48]),
                   INV_COS_BIT);
  const int32x4_t s3_hi =
      vrshrq_n_s32(vmlal_n_s16(vmull_n_s16(x1_hi, (int16_t)cospi[16]), x3_hi,
                               (int16_t)cospi[48]),
                   INV_COS_BIT);

  out[0] = narrow_inv_txfm4_out_neon(vaddq_s32(s0_lo, s3_lo),
                                     vaddq_s32(s0_hi, s3_hi), out_rshift);
  out[1] = narrow_inv_txfm4_out_neon(vaddq_s32(s1_lo, s2_lo),
                                     vaddq_s32(s1_hi, s2_hi), out_rshift);
  out[2] = narrow_inv_txfm4_out_neon(vsubq_s32(s1_lo, s2_lo),
                                     vsubq_s32(s1_hi, s2_hi), out_rshift);
  out[3] = narrow_inv_txfm4_out_neon(vsubq_s32(s0_lo, s3_lo),
                                     vsubq_s32(s0_hi, s3_hi), out_rshift);
}

static AOM_FORCE_INLINE void iadst4_s16x8_neon(const int16x8_t *in,
                                               int16x8_t *out, int out_rshift) {
  const int32_t *sinpi = sinpi_arr(INV_COS_BIT);
  const int16_t sin1 = (int16_t)sinpi[1];
  const int16_t sin2 = (int16_t)sinpi[2];
  const int16_t sin3 = (int16_t)sinpi[3];
  const int16_t sin4 = (int16_t)sinpi[4];

  const int16x4_t x0_lo = vget_low_s16(in[0]);
  const int16x4_t x0_hi = vget_high_s16(in[0]);
  const int16x4_t x1_lo = vget_low_s16(in[1]);
  const int16x4_t x1_hi = vget_high_s16(in[1]);
  const int16x4_t x2_lo = vget_low_s16(in[2]);
  const int16x4_t x2_hi = vget_high_s16(in[2]);
  const int16x4_t x3_lo = vget_low_s16(in[3]);
  const int16x4_t x3_hi = vget_high_s16(in[3]);

  int32x4_t s0_lo = vmull_n_s16(x0_lo, sin1);
  int32x4_t s0_hi = vmull_n_s16(x0_hi, sin1);
  s0_lo = vmlal_n_s16(s0_lo, x2_lo, sin4);
  s0_hi = vmlal_n_s16(s0_hi, x2_hi, sin4);
  s0_lo = vmlal_n_s16(s0_lo, x3_lo, sin2);
  s0_hi = vmlal_n_s16(s0_hi, x3_hi, sin2);

  int32x4_t s1_lo = vmull_n_s16(x0_lo, sin2);
  int32x4_t s1_hi = vmull_n_s16(x0_hi, sin2);
  s1_lo = vmlsl_n_s16(s1_lo, x2_lo, sin1);
  s1_hi = vmlsl_n_s16(s1_hi, x2_hi, sin1);
  s1_lo = vmlsl_n_s16(s1_lo, x3_lo, sin4);
  s1_hi = vmlsl_n_s16(s1_hi, x3_hi, sin4);

  const int32x4_t s3_lo = vmull_n_s16(x1_lo, sin3);
  const int32x4_t s3_hi = vmull_n_s16(x1_hi, sin3);
  const int32x4_t s7_lo = vaddw_s16(vsubl_s16(x0_lo, x2_lo), x3_lo);
  const int32x4_t s7_hi = vaddw_s16(vsubl_s16(x0_hi, x2_hi), x3_hi);
  const int32x4_t s2_lo = vmulq_n_s32(s7_lo, sinpi[3]);
  const int32x4_t s2_hi = vmulq_n_s32(s7_hi, sinpi[3]);

  const int32x4_t y0_lo = vrshrq_n_s32(vaddq_s32(s0_lo, s3_lo), INV_COS_BIT);
  const int32x4_t y0_hi = vrshrq_n_s32(vaddq_s32(s0_hi, s3_hi), INV_COS_BIT);
  const int32x4_t y1_lo = vrshrq_n_s32(vaddq_s32(s1_lo, s3_lo), INV_COS_BIT);
  const int32x4_t y1_hi = vrshrq_n_s32(vaddq_s32(s1_hi, s3_hi), INV_COS_BIT);
  const int32x4_t y2_lo = vrshrq_n_s32(s2_lo, INV_COS_BIT);
  const int32x4_t y2_hi = vrshrq_n_s32(s2_hi, INV_COS_BIT);
  const int32x4_t y3_lo =
      vrshrq_n_s32(vsubq_s32(vaddq_s32(s0_lo, s1_lo), s3_lo), INV_COS_BIT);
  const int32x4_t y3_hi =
      vrshrq_n_s32(vsubq_s32(vaddq_s32(s0_hi, s1_hi), s3_hi), INV_COS_BIT);

  out[0] = narrow_inv_txfm4_out_neon(y0_lo, y0_hi, out_rshift);
  out[1] = narrow_inv_txfm4_out_neon(y1_lo, y1_hi, out_rshift);
  out[2] = narrow_inv_txfm4_out_neon(y2_lo, y2_hi, out_rshift);
  out[3] = narrow_inv_txfm4_out_neon(y3_lo, y3_hi, out_rshift);
}

static AOM_FORCE_INLINE void apply_txfm4_s16x8_neon(int16x8_t *a,
                                                    ITX_TYPE_1D type,
                                                    int out_rshift) {
  switch (type) {
    case IDCT_1D: idct4_s16x8_neon(a, a, out_rshift); break;
    case IADST_1D: iadst4_s16x8_neon(a, a, out_rshift); break;
    default: identity_txfm_round_neon(a, a, 0, 4, out_rshift); break;
  }
}

static AOM_FORCE_INLINE int16x4_t narrow_inv_txfm4_s16x4_neon(int32x4_t v,
                                                              int out_rshift) {
  if (out_rshift == 4) {
    return vqrshrn_n_s32(v, 4);
  }
  assert(out_rshift == 0);
  return vqmovn_s32(v);
}

static AOM_FORCE_INLINE void idct4_s16x4_neon(const int16x4_t *in,
                                              int16x4_t *out, int out_rshift) {
  const int32_t *cospi = cospi_arr(INV_COS_BIT);
  const int16_t cospi32 = (int16_t)cospi[32];
  const int32x4_t p = vmull_n_s16(in[0], cospi32);

  const int32x4_t s0 =
      vrshrq_n_s32(vmlal_n_s16(p, in[2], cospi32), INV_COS_BIT);
  const int32x4_t s1 =
      vrshrq_n_s32(vmlsl_n_s16(p, in[2], cospi32), INV_COS_BIT);
  const int32x4_t s2 =
      vrshrq_n_s32(vmlsl_n_s16(vmull_n_s16(in[1], (int16_t)cospi[48]), in[3],
                               (int16_t)cospi[16]),
                   INV_COS_BIT);
  const int32x4_t s3 =
      vrshrq_n_s32(vmlal_n_s16(vmull_n_s16(in[1], (int16_t)cospi[16]), in[3],
                               (int16_t)cospi[48]),
                   INV_COS_BIT);

  out[0] = narrow_inv_txfm4_s16x4_neon(vaddq_s32(s0, s3), out_rshift);
  out[1] = narrow_inv_txfm4_s16x4_neon(vaddq_s32(s1, s2), out_rshift);
  out[2] = narrow_inv_txfm4_s16x4_neon(vsubq_s32(s1, s2), out_rshift);
  out[3] = narrow_inv_txfm4_s16x4_neon(vsubq_s32(s0, s3), out_rshift);
}

static AOM_FORCE_INLINE void iadst4_s16x4_neon(const int16x4_t *in,
                                               int16x4_t *out, int out_rshift) {
  const int32_t *sinpi = sinpi_arr(INV_COS_BIT);
  const int16_t sin1 = (int16_t)sinpi[1];
  const int16_t sin2 = (int16_t)sinpi[2];
  const int16_t sin3 = (int16_t)sinpi[3];
  const int16_t sin4 = (int16_t)sinpi[4];

  int32x4_t s0 = vmull_n_s16(in[0], sin1);
  s0 = vmlal_n_s16(s0, in[2], sin4);
  s0 = vmlal_n_s16(s0, in[3], sin2);

  int32x4_t s1 = vmull_n_s16(in[0], sin2);
  s1 = vmlsl_n_s16(s1, in[2], sin1);
  s1 = vmlsl_n_s16(s1, in[3], sin4);

  const int32x4_t s3 = vmull_n_s16(in[1], sin3);
  const int32x4_t s7 = vaddw_s16(vsubl_s16(in[0], in[2]), in[3]);
  const int32x4_t s2 = vmulq_n_s32(s7, sinpi[3]);

  const int32x4_t y0 = vrshrq_n_s32(vaddq_s32(s0, s3), INV_COS_BIT);
  const int32x4_t y1 = vrshrq_n_s32(vaddq_s32(s1, s3), INV_COS_BIT);
  const int32x4_t y2 = vrshrq_n_s32(s2, INV_COS_BIT);
  const int32x4_t y3 =
      vrshrq_n_s32(vsubq_s32(vaddq_s32(s0, s1), s3), INV_COS_BIT);

  out[0] = narrow_inv_txfm4_s16x4_neon(y0, out_rshift);
  out[1] = narrow_inv_txfm4_s16x4_neon(y1, out_rshift);
  out[2] = narrow_inv_txfm4_s16x4_neon(y2, out_rshift);
  out[3] = narrow_inv_txfm4_s16x4_neon(y3, out_rshift);
}

static AOM_FORCE_INLINE void identity_txfm4_s16x4_neon(const int16x4_t *in,
                                                       int16x4_t *out,
                                                       int out_rshift) {
  const int16_t scale = (int16_t)NewSqrt2;
  int i = 0;
  if (out_rshift == 4) {
    do {
      out[i] = vqrshrn_n_s32(vrshrq_n_s32(vmull_n_s16(in[i], scale), 12), 4);
    } while (++i != 4);
  } else {
    assert(out_rshift == 0);
    do {
      out[i] = vqrshrn_n_s32(vmull_n_s16(in[i], scale), 12);
    } while (++i != 4);
  }
}

static AOM_FORCE_INLINE void apply_txfm4_s16x4_neon(int16x4_t *a,
                                                    ITX_TYPE_1D type,
                                                    int out_rshift) {
  switch (type) {
    case IDCT_1D: idct4_s16x4_neon(a, a, out_rshift); break;
    case IADST_1D: iadst4_s16x4_neon(a, a, out_rshift); break;
    default:
      assert(type == IIDENTITY_1D);
      identity_txfm4_s16x4_neon(a, a, out_rshift);
      break;
  }
}

static AOM_FORCE_INLINE void flip_buf_4_neon(int16x4_t *a) {
  const int16x4_t t0 = a[0];
  const int16x4_t t1 = a[1];
  a[0] = a[3];
  a[1] = a[2];
  a[2] = t1;
  a[3] = t0;
}

static AOM_FORCE_INLINE void lowbd_add_buffer_4x4_neon(const int16x4_t *in,
                                                       uint8_t *output,
                                                       int stride) {
  const uint8x8_t pred01 = load_u8_4x2(output, stride);
  const int16x8_t res01 = vcombine_s16(in[0], in[1]);
  const int16x8_t sum01 =
      vaddq_s16(vreinterpretq_s16_u16(vmovl_u8(pred01)), res01);
  store_u8x4_strided_x2(output, stride, vqmovun_s16(sum01));

  output += 2 * stride;
  const uint8x8_t pred23 = load_u8_4x2(output, stride);
  const int16x8_t res23 = vcombine_s16(in[2], in[3]);
  const int16x8_t sum23 =
      vaddq_s16(vreinterpretq_s16_u16(vmovl_u8(pred23)), res23);
  store_u8x4_strided_x2(output, stride, vqmovun_s16(sum23));
}

static inline void lowbd_inv_txfm2d_add_4x4_neon(const int32_t *input,
                                                 uint8_t *output, int stride,
                                                 TX_TYPE tx_type, int eob) {
  if (eob == 1 && tx_type == DCT_DCT) {
    const int32_t *cospi = cospi_arr(INV_COS_BIT);
    int32_t dc = round_shift((int64_t)input[0] * cospi[32], INV_COS_BIT);
    dc = round_shift((int64_t)dc * cospi[32], INV_COS_BIT);
    dc = round_shift((int64_t)dc, 4);
    lowbd_inv_txfm_dc_only_4xn_neon(output, stride, dc, 4);
    return;
  }

  int16x4_t a[4];
  int i = 0;
  do {
    a[i] = vqmovn_s32(vld1q_s32(input + i * 4));
  } while (++i != 4);

  // Avoid get_flip_cfg and 1D transform switch dispatch for common 4x4 DCT.
  if (tx_type == DCT_DCT) {
    idct4_s16x4_neon(a, a, 0);
    transpose_elems_inplace_s16_4x4(&a[0], &a[1], &a[2], &a[3]);
    idct4_s16x4_neon(a, a, 4);
    lowbd_add_buffer_4x4_neon(a, output, stride);
    return;
  }

  int ud_flip, lr_flip;
  get_flip_cfg(tx_type, &ud_flip, &lr_flip);

  apply_txfm4_s16x4_neon(a, hitx_1d_tab[tx_type], 0);
  if (lr_flip) {
    flip_buf_4_neon(a);
  }
  transpose_elems_inplace_s16_4x4(&a[0], &a[1], &a[2], &a[3]);
  apply_txfm4_s16x4_neon(a, vitx_1d_tab[tx_type], 4);
  if (ud_flip) {
    flip_buf_4_neon(a);
  }
  lowbd_add_buffer_4x4_neon(a, output, stride);
}

static AOM_FORCE_INLINE void identity_txfm8_s16x8_neon(int16x8_t *b,
                                                       int out_rshift) {
  if (out_rshift == 4) {
    int i = 0;
    do {
      b[i] = vrshrq_n_s16(b[i], 3);
    } while (++i != 8);
  } else if (out_rshift == 0) {
    int i = 0;
    do {
      b[i] = vqaddq_s16(b[i], b[i]);
    } while (++i != 8);
  } else {
    assert(out_rshift == 1);
  }
}

static AOM_FORCE_INLINE void apply_txfm8_s16x8_neon(int16x8_t *b,
                                                    ITX_TYPE_1D type,
                                                    int out_rshift) {
  switch (type) {
    case IDCT_1D:
      idct8_neon_impl(b, b, INV_COS_BIT);
      round_shift_array_16_neon(b, 8, out_rshift);
      break;
    case IADST_1D:
      iadst8_neon_impl(b, b, INV_COS_BIT);
      round_shift_array_16_neon(b, 8, out_rshift);
      break;
    default:
      assert(type == IIDENTITY_1D);
      identity_txfm8_s16x8_neon(b, out_rshift);
      break;
  }
}

static AOM_FORCE_INLINE void apply_txfm8_low1_s16x8_neon(int16x8_t *b,
                                                         ITX_TYPE_1D type,
                                                         int out_rshift) {
  switch (type) {
    case IDCT_1D:
      idct8_low1_neon_impl(b, b, INV_COS_BIT);
      round_shift_array_16_neon(b, 8, out_rshift);
      break;
    case IADST_1D:
      iadst8_low1_neon_impl(b, b, INV_COS_BIT);
      round_shift_array_16_neon(b, 8, out_rshift);
      break;
    default:
      assert(type == IIDENTITY_1D);
      if (out_rshift == 4) {
        b[0] = vrshrq_n_s16(b[0], 3);
      } else {
        assert(out_rshift == 1);
      }
      break;
  }
}

static AOM_FORCE_INLINE void apply_txfm16_s16x8_neon(int16x8_t *b,
                                                     ITX_TYPE_1D type,
                                                     int out_rshift) {
  switch (type) {
    case IDCT_1D:
      idct16_neon(b, b, INV_COS_BIT);
      round_shift_array_16_neon(b, 16, out_rshift);
      break;
    case IADST_1D:
      iadst16_neon(b, b, INV_COS_BIT);
      round_shift_array_16_neon(b, 16, out_rshift);
      break;
    default: identity_txfm_round_neon(b, b, 2, 16, out_rshift); break;
  }
}

static AOM_FORCE_INLINE int32_t lowbd_dc_only_rect1_val(int32_t dc_in) {
  const int32_t *cospi = cospi_arr(INV_COS_BIT);
  int32_t dc = round_shift((int64_t)dc_in * NewInvSqrt2, NewSqrt2Bits);
  dc = round_shift((int64_t)dc * cospi[32], INV_COS_BIT);
  dc = round_shift((int64_t)dc * cospi[32], INV_COS_BIT);
  return round_shift((int64_t)dc, 4);
}

static AOM_FORCE_INLINE int32_t lowbd_dc_only_rect2_val(int32_t dc_in) {
  const int32_t *cospi = cospi_arr(INV_COS_BIT);
  int32_t dc = round_shift((int64_t)dc_in * cospi[32], INV_COS_BIT);
  dc = round_shift((int64_t)dc, 1);
  dc = round_shift((int64_t)dc * cospi[32], INV_COS_BIT);
  return round_shift((int64_t)dc, 4);
}

static AOM_FORCE_INLINE void lowbd_inv_txfm_dc_only_8x4_neon(uint8_t *output,
                                                             int stride,
                                                             int32_t dc) {
  const int16x8_t dc_v = vdupq_n_s16((int16_t)clamp(dc, -255, 255));
  int h = 4;
  do {
    const int16x8_t pred = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(output)));
    vst1_u8(output, vqmovun_s16(vaddq_s16(pred, dc_v)));
    output += stride;
  } while (--h != 0);
}

static AOM_FORCE_INLINE void lowbd_inv_txfm_dc_only_16x4_neon(uint8_t *output,
                                                              int stride,
                                                              int32_t dc) {
  const int16x8_t dc_v = vdupq_n_s16((int16_t)clamp(dc, -255, 255));
  int h = 4;
  do {
    const uint8x16_t pred = vld1q_u8(output);
    vst1q_u8(output, lowbd_get_recon_16x16_neon(pred, dc_v, dc_v));
    output += stride;
  } while (--h != 0);
}

// Keep out-of-line so large stack buffers do not force spills in the
// dispatcher / TX_4X4 fast path.
static AOM_NOINLINE void lowbd_inv_txfm2d_add_4x8_neon(const int32_t *input,
                                                       uint8_t *output,
                                                       int stride,
                                                       TX_TYPE tx_type) {
  int16x8_t a[4];
  int16x4_t t[8];
  int16x8_t b[8];
  int ud_flip, lr_flip;
  get_flip_cfg(tx_type, &ud_flip, &lr_flip);

  load_buffer_32bit_to_16bit_neon(input, 8, a, 4);
  round_shift_for_rect(a, a, 4);
  apply_txfm4_s16x8_neon(a, hitx_1d_tab[tx_type], 0);
  if (lr_flip) {
    flip_buf_ud_neon(a, 4);
  }
  transpose_arrays_s16_8x4(a, t);
  for (int i = 0; i < 8; ++i) {
    b[i] = vcombine_s16(t[i], vdup_n_s16(0));
  }
  apply_txfm8_s16x8_neon(b, vitx_1d_tab[tx_type], 4);
  lowbd_add_flip_buffer_4xn_neon(b, output, stride, ud_flip, 8);
}

// Keep out-of-line so large stack buffers do not force spills in the
// dispatcher / TX_4X4 fast path.
static AOM_NOINLINE void lowbd_inv_txfm2d_add_8x4_neon(const int32_t *input,
                                                       uint8_t *output,
                                                       int stride,
                                                       TX_TYPE tx_type) {
  int16x8_t a[8];
  int16x4_t row_in[8];
  int16x8_t b[4];
  int ud_flip, lr_flip;
  get_flip_cfg(tx_type, &ud_flip, &lr_flip);

  for (int c = 0; c < 8; ++c) {
    a[c] = vcombine_s16(vmovn_s32(vld1q_s32(input + c * 4)), vdup_n_s16(0));
  }
  round_shift_for_rect(a, a, 8);
  apply_txfm8_s16x8_neon(a, hitx_1d_tab[tx_type], 0);
  if (lr_flip) {
    for (int c = 0; c < 8; ++c) {
      row_in[c] = vget_low_s16(a[7 - c]);
    }
  } else {
    for (int c = 0; c < 8; ++c) {
      row_in[c] = vget_low_s16(a[c]);
    }
  }
  transpose_arrays_s16_4x8(row_in, b);
  apply_txfm4_s16x8_neon(b, vitx_1d_tab[tx_type], 4);
  lowbd_add_flip_buffer_8xn_neon(b, output, stride, ud_flip, 4);
}

// Keep out-of-line so large stack buffers do not force spills in the
// dispatcher / TX_4X4 fast path.
static AOM_NOINLINE void lowbd_inv_txfm2d_add_4x16_neon(const int32_t *input,
                                                        uint8_t *output,
                                                        int stride,
                                                        TX_TYPE tx_type) {
  int16x8_t a0[4], a1[4];
  int16x4_t t[16];
  int16x8_t b[16];
  int ud_flip, lr_flip;
  get_flip_cfg(tx_type, &ud_flip, &lr_flip);

  load_buffer_32bit_to_16bit_neon(input, 16, a0, 4);
  load_buffer_32bit_to_16bit_neon(input + 8, 16, a1, 4);
  apply_txfm4_s16x8_neon(a0, hitx_1d_tab[tx_type], 1);
  apply_txfm4_s16x8_neon(a1, hitx_1d_tab[tx_type], 1);
  if (lr_flip) {
    flip_buf_ud_neon(a0, 4);
    flip_buf_ud_neon(a1, 4);
  }
  transpose_arrays_s16_8x4(a0, t);
  transpose_arrays_s16_8x4(a1, t + 8);
  for (int i = 0; i < 16; ++i) {
    b[i] = vcombine_s16(t[i], vdup_n_s16(0));
  }
  apply_txfm16_s16x8_neon(b, vitx_1d_tab[tx_type], 4);
  lowbd_add_flip_buffer_4xn_neon(b, output, stride, ud_flip, 16);
}

// Keep out-of-line so large stack buffers do not force spills in the
// dispatcher / TX_4X4 fast path.
static AOM_NOINLINE void lowbd_inv_txfm2d_add_16x4_neon(const int32_t *input,
                                                        uint8_t *output,
                                                        int stride,
                                                        TX_TYPE tx_type) {
  int16x8_t a[16];
  int16x4_t row_lo[8], row_hi[8];
  int16x8_t b[8];
  int ud_flip, lr_flip;
  get_flip_cfg(tx_type, &ud_flip, &lr_flip);

  for (int c = 0; c < 16; ++c) {
    a[c] = vcombine_s16(vmovn_s32(vld1q_s32(input + c * 4)), vdup_n_s16(0));
  }
  apply_txfm16_s16x8_neon(a, hitx_1d_tab[tx_type], 1);
  if (lr_flip) {
    for (int c = 0; c < 8; ++c) {
      row_lo[c] = vget_low_s16(a[15 - c]);
      row_hi[c] = vget_low_s16(a[7 - c]);
    }
  } else {
    for (int c = 0; c < 8; ++c) {
      row_lo[c] = vget_low_s16(a[c]);
      row_hi[c] = vget_low_s16(a[c + 8]);
    }
  }
  transpose_arrays_s16_4x8(row_lo, b);
  transpose_arrays_s16_4x8(row_hi, b + 4);
  apply_txfm4_s16x8_neon(b, vitx_1d_tab[tx_type], 4);
  apply_txfm4_s16x8_neon(b + 4, vitx_1d_tab[tx_type], 4);
  lowbd_add_flip_buffer_16xn_neon(b, output, stride, ud_flip, 4);
}

// Keep out-of-line to isolate TX_8X8 register/stack footprint from TX_4X4.
static AOM_NOINLINE void lowbd_inv_txfm2d_add_8x8_dct_neon(const int32_t *input,
                                                           uint8_t *output,
                                                           int stride) {
  int16x8_t a[8], b[8];
  load_buffer_32bit_to_16bit_neon(input, 8, a, 8);
  idct8_neon_impl(a, a, INV_COS_BIT);
  round_shift_array_16_neon(a, 8, 1);
  transpose_arrays_s16_8x8(a, b);
  idct8_neon_impl(b, b, INV_COS_BIT);
  round_shift_array_16_neon(b, 8, 4);
  lowbd_add_flip_buffer_8xn_neon(b, output, stride, 0, 8);
}

// Keep out-of-line to isolate non-DCT TX_8X8 paths from the common path.
static AOM_NOINLINE void lowbd_inv_txfm2d_add_8x8_non_dct_neon(
    const int32_t *input, uint8_t *output, int stride, TX_TYPE tx_type,
    int eob) {
  int16x8_t a[8], b[8];
  int ud_flip, lr_flip;
  get_flip_cfg(tx_type, &ud_flip, &lr_flip);

  if (eob == 1) {
    a[0] = vsetq_lane_s16((int16_t)input[0], vdupq_n_s16(0), 0);
    if (hitx_1d_tab[tx_type] == IIDENTITY_1D) {
      int i = 1;
      do {
        a[i] = vdupq_n_s16(0);
      } while (++i != 8);
    }
    apply_txfm8_low1_s16x8_neon(a, hitx_1d_tab[tx_type], 1);
    if (lr_flip) {
      flip_buf_ud_neon(a, 8);
    }
    transpose_arrays_s16_8x8(a, b);
    if (vitx_1d_tab[tx_type] == IIDENTITY_1D) {
      assert(!ud_flip);
      b[0] = vrshrq_n_s16(b[0], 3);
      lowbd_add_flip_buffer_8xn_neon(b, output, stride, 0, 1);
      return;
    }
    apply_txfm8_low1_s16x8_neon(b, vitx_1d_tab[tx_type], 4);
    if (ud_flip) {
      flip_buf_ud_neon(b, 8);
    }
    lowbd_add_flip_buffer_8xn_neon(b, output, stride, 0, 8);
    return;
  }

  load_buffer_32bit_to_16bit_neon(input, 8, a, 8);
  apply_txfm8_s16x8_neon(a, hitx_1d_tab[tx_type], 1);
  if (lr_flip) {
    flip_buf_ud_neon(a, 8);
  }
  transpose_arrays_s16_8x8(a, b);
  apply_txfm8_s16x8_neon(b, vitx_1d_tab[tx_type], 4);
  if (ud_flip) {
    flip_buf_ud_neon(b, 8);
  }
  lowbd_add_flip_buffer_8xn_neon(b, output, stride, 0, 8);
}

static inline void lowbd_inv_txfm2d_add_no_identity_neon(
    const int32_t *input, uint8_t *output, int stride, TX_TYPE tx_type,
    TX_SIZE tx_size, int eob) {
  int16x8_t a[64 * 8];
  int16x8_t b[64 * 8];
  int eobx, eoby, ud_flip, lr_flip;
  get_eobx_eoby_scan_default(&eobx, &eoby, tx_size, eob);
  const int8_t *shift = av1_inv_txfm_shift_ls[tx_size];
  const int txw_idx = get_txw_idx(tx_size);
  const int txh_idx = get_txh_idx(tx_size);
  const int txfm_size_col = tx_size_wide[tx_size];
  const int txfm_size_row = tx_size_high[tx_size];
  const int rect_type = get_rect_tx_log_ratio(txfm_size_col, txfm_size_row);
  const int buf_size_w_div8 = txfm_size_col >> 3;
  const int buf_size_nonzero_h_div8 = (eoby + 8) >> 3;
  const int buf_size_nonzero_w = (eobx + 8) >> 3 << 3;
  const int input_stride = AOMMIN(32, txfm_size_row);
  const int fun_idx_x = lowbd_txfm_all_1d_zeros_idx[eobx];
  const int fun_idx_y = lowbd_txfm_all_1d_zeros_idx[eoby];
  int temp_b = 0;

  if (tx_type == DCT_DCT) {
    const transform_neon row_txfm =
        lowbd_txfm_all_1d_zeros_w_arr[txw_idx][IDCT_1D][fun_idx_x];
    const transform_neon col_txfm =
        lowbd_txfm_all_1d_zeros_w_arr[txh_idx][IDCT_1D][fun_idx_y];

    for (int i = 0; i < buf_size_nonzero_h_div8; i++) {
      int16x8_t *cur_a = &a[i * txfm_size_col];
      load_buffer_32bit_to_16bit_neon(input, input_stride, cur_a,
                                      buf_size_nonzero_w);
      input += 8;
      if (abs(rect_type) == 1) {
        round_shift_for_rect(cur_a, cur_a, buf_size_nonzero_w);
      }
      row_txfm(cur_a, cur_a, INV_COS_BIT);
      round_shift_array_16_neon(cur_a, txfm_size_col, -shift[0]);
      for (int j = 0; j < buf_size_w_div8; ++j) {
        transpose_arrays_s16_8x8(&cur_a[j * 8], &b[temp_b + txfm_size_row * j]);
      }
      temp_b += 8;
    }
    for (int j = 0; j < buf_size_w_div8; ++j) {
      col_txfm(&b[j * txfm_size_row], &b[j * txfm_size_row], INV_COS_BIT);
      round_shift_array_16_neon(&b[j * txfm_size_row], txfm_size_row,
                                -shift[1]);
    }

    if (txfm_size_col >= 16) {
      for (int i = 0; i < (txfm_size_col >> 4); i++) {
        lowbd_add_flip_buffer_16xn_neon(&b[i * txfm_size_row * 2],
                                        output + 16 * i, stride, 0,
                                        txfm_size_row);
      }
    } else if (txfm_size_col == 8) {
      lowbd_add_flip_buffer_8xn_neon(b, output, stride, 0, txfm_size_row);
    }
    return;
  }

  const transform_neon row_txfm =
      lowbd_txfm_all_1d_zeros_w_arr[txw_idx][hitx_1d_tab[tx_type]][fun_idx_x];
  const transform_neon col_txfm =
      lowbd_txfm_all_1d_zeros_w_arr[txh_idx][vitx_1d_tab[tx_type]][fun_idx_y];

  assert(col_txfm != NULL);
  assert(row_txfm != NULL);

  get_flip_cfg(tx_type, &ud_flip, &lr_flip);

  for (int i = 0; i < buf_size_nonzero_h_div8; i++) {
    int16x8_t *cur_a = &a[i * txfm_size_col];
    load_buffer_32bit_to_16bit_neon(input, input_stride, cur_a,
                                    buf_size_nonzero_w);
    input += 8;
    if (abs(rect_type) == 1) {
      round_shift_for_rect(cur_a, cur_a, buf_size_nonzero_w);
    }
    row_txfm(cur_a, cur_a, INV_COS_BIT);
    round_shift_array_16_neon(cur_a, txfm_size_col, -shift[0]);
    if (lr_flip == 1) {
      for (int j = 0; j < buf_size_w_div8; ++j) {
        flip_buf_ud_neon(&cur_a[j * 8], 8);
        transpose_arrays_s16_8x8(
            &cur_a[j * 8],
            &b[temp_b + txfm_size_row * (buf_size_w_div8 - 1 - j)]);
      }
      temp_b += 8;
    } else {
      for (int j = 0; j < buf_size_w_div8; ++j) {
        transpose_arrays_s16_8x8(&cur_a[j * 8], &b[temp_b + txfm_size_row * j]);
      }
      temp_b += 8;
    }
  }
  for (int j = 0; j < buf_size_w_div8; ++j) {
    col_txfm(&b[j * txfm_size_row], &b[j * txfm_size_row], INV_COS_BIT);
    round_shift_array_16_neon(&b[j * txfm_size_row], txfm_size_row, -shift[1]);
  }

  if (txfm_size_col >= 16) {
    for (int i = 0; i < (txfm_size_col >> 4); i++) {
      lowbd_add_flip_buffer_16xn_neon(&b[i * txfm_size_row * 2],
                                      output + 16 * i, stride, ud_flip,
                                      txfm_size_row);
    }
  } else if (txfm_size_col == 8) {
    lowbd_add_flip_buffer_8xn_neon(b, output, stride, ud_flip, txfm_size_row);
  }
}

static inline int16_t lowbd_inv_txfm_dc_coeff_neon(int32_t dc,
                                                   TX_SIZE tx_size) {
  const int8_t *shift = av1_inv_txfm_shift_ls[tx_size];
  const int txfm_size_col = tx_size_wide[tx_size];
  const int txfm_size_row = tx_size_high[tx_size];
  const int rect_type = get_rect_tx_log_ratio(txfm_size_col, txfm_size_row);
  const int16_t cospi32 = (int16_t)cospi_arr(INV_COS_BIT)[32];
  int16x4_t v_dc = vdup_n_s16((int16_t)dc);
  if (abs(rect_type) == 1) {
    v_dc = vqrshrn_n_s32(vmull_n_s16(v_dc, (int16_t)NewInvSqrt2),
                         (int32_t)NewSqrt2Bits);
  }
  v_dc = vrshrn_n_s32(vmull_n_s16(v_dc, cospi32), INV_COS_BIT);
  int16x8_t v_dc8 = vcombine_s16(v_dc, v_dc);
  if (shift[0]) {
    v_dc8 = vrshlq_s16(v_dc8, vdupq_n_s16(shift[0]));
  }
  v_dc = vrshrn_n_s32(vmull_n_s16(vget_low_s16(v_dc8), cospi32), INV_COS_BIT);
  v_dc8 = vcombine_s16(v_dc, v_dc);
  if (shift[1]) {
    v_dc8 = vrshlq_s16(v_dc8, vdupq_n_s16(shift[1]));
  }
  return vgetq_lane_s16(v_dc8, 0);
}

static void lowbd_inv_txfm2d_add_dc_only_neon(int32_t dc, uint8_t *output,
                                              int stride, TX_SIZE tx_size) {
  const int16x8_t v_dc = vdupq_n_s16(lowbd_inv_txfm_dc_coeff_neon(dc, tx_size));
  const int width = tx_size_wide[tx_size];
  int height = tx_size_high[tx_size];
  assert(height >= 8 && (height & 7) == 0);
  if (width == 8) {
    do {
      const int16x8_t p = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(output)));
      vst1_u8(output, vqmovun_s16(vaddq_s16(p, v_dc)));
      output += stride;
    } while (--height != 0);
  } else {
    assert(width >= 16 && (width & 15) == 0);
    do {
      int c = 0;
      do {
        const uint8x16_t p = vld1q_u8(output + c);
        vst1q_u8(output + c, lowbd_get_recon_16x16_neon(p, v_dc, v_dc));
        c += 16;
      } while (c != width);
      output += stride;
    } while (--height != 0);
  }
}

// Keep universe dispatcher out of the common fast path (small blocks & DC-only)
// to prevent code bloat and preserve small-block invocation latency.
static AOM_NOINLINE void lowbd_inv_txfm2d_add_universe_neon(
    const int32_t *input, uint8_t *output, int stride, TX_TYPE tx_type,
    TX_SIZE tx_size, int eob) {
  switch (tx_type) {
    case IDTX:
      lowbd_inv_txfm2d_add_idtx_neon(input, output, stride, tx_type, tx_size,
                                     eob);
      break;

    case H_DCT:
    case H_ADST:
    case H_FLIPADST:
      lowbd_inv_txfm2d_add_v_identity_neon(input, output, stride, tx_type,
                                           tx_size, eob);
      break;

    case V_DCT:
    case V_ADST:
    case V_FLIPADST:
      lowbd_inv_txfm2d_add_h_identity_neon(input, output, stride, tx_type,
                                           tx_size, eob);
      break;

    default:
      lowbd_inv_txfm2d_add_no_identity_neon(input, output, stride, tx_type,
                                            tx_size, eob);
      break;
  }
}

void av1_lowbd_inv_txfm2d_add_neon(const int32_t *input, uint8_t *output,
                                   int stride, TX_TYPE tx_type, TX_SIZE tx_size,
                                   int eob) {
  switch (tx_size) {
    case TX_4X4:
      lowbd_inv_txfm2d_add_4x4_neon(input, output, stride, tx_type, eob);
      break;

    case TX_4X8:
      if (eob == 1 && tx_type == DCT_DCT) {
        lowbd_inv_txfm_dc_only_4xn_neon(output, stride,
                                        lowbd_dc_only_rect1_val(input[0]), 8);
      } else {
        lowbd_inv_txfm2d_add_4x8_neon(input, output, stride, tx_type);
      }
      break;

    case TX_8X4:
      if (eob == 1 && tx_type == DCT_DCT) {
        lowbd_inv_txfm_dc_only_8x4_neon(output, stride,
                                        lowbd_dc_only_rect1_val(input[0]));
      } else {
        lowbd_inv_txfm2d_add_8x4_neon(input, output, stride, tx_type);
      }
      break;

    case TX_4X16:
      if (eob == 1 && tx_type == DCT_DCT) {
        lowbd_inv_txfm_dc_only_4xn_neon(output, stride,
                                        lowbd_dc_only_rect2_val(input[0]), 16);
      } else {
        lowbd_inv_txfm2d_add_4x16_neon(input, output, stride, tx_type);
      }
      break;

    case TX_16X4:
      if (eob == 1 && tx_type == DCT_DCT) {
        lowbd_inv_txfm_dc_only_16x4_neon(output, stride,
                                         lowbd_dc_only_rect2_val(input[0]));
      } else {
        lowbd_inv_txfm2d_add_16x4_neon(input, output, stride, tx_type);
      }
      break;

    case TX_8X8:
      if (tx_type != DCT_DCT) {
        lowbd_inv_txfm2d_add_8x8_non_dct_neon(input, output, stride, tx_type,
                                              eob);
      } else if (eob == 1) {
        lowbd_inv_txfm2d_add_dc_only_neon(input[0], output, stride, tx_size);
      } else {
        lowbd_inv_txfm2d_add_8x8_dct_neon(input, output, stride);
      }
      break;

    default:
      if (eob == 1 && tx_type == DCT_DCT) {
        lowbd_inv_txfm2d_add_dc_only_neon(input[0], output, stride, tx_size);
      } else {
        lowbd_inv_txfm2d_add_universe_neon(input, output, stride, tx_type,
                                           tx_size, eob);
      }
      break;
  }
}
void av1_inv_txfm_add_neon(const tran_low_t *dqcoeff, uint8_t *dst, int stride,
                           const TxfmParam *txfm_param) {
  if (!txfm_param->lossless) {
    // TODO(bug 528050364): Remove the C fallback after issues with
    // arm-linux-gnueabi-gcc-14+ are addressed.
#if defined(__GNUC__) && __GNUC__ >= 14 && defined(__ARM_ARCH) && __ARM_ARCH < 7
    av1_inv_txfm_add_c(dqcoeff, dst, stride, txfm_param);
#else
    const TX_TYPE tx_type = txfm_param->tx_type;
    av1_lowbd_inv_txfm2d_add_neon(dqcoeff, dst, stride, tx_type,
                                  txfm_param->tx_size, txfm_param->eob);
#endif
  } else {
    av1_inv_txfm_add_c(dqcoeff, dst, stride, txfm_param);
  }
}
