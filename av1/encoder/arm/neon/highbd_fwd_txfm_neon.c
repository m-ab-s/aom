/*
 * Copyright (c) 2020, Alliance for Open Media. All rights reserved
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

#include "av1/common/av1_txfm.h"
#include "av1/encoder/av1_fwd_txfm1d_cfg.h"
#include "aom_dsp/arm/transpose_neon.h"
#include "aom_dsp/txfm_common.h"
#include "aom_ports/mem.h"
#include "config/av1_rtcd.h"
#include "config/aom_config.h"
#include "txfm_neon.h"

static INLINE void transpose_arrays_s32_64x64(const int32x4_t *in,
                                              int32x4_t *out) {
  // This is not quite the same as the other transposes defined in
  // transpose_neon.h: We only write the low 64x32 sub-matrix since the rest is
  // unused by the following row transform.
  for (int j = 0; j < 8; ++j) {
    for (int i = 0; i < 16; ++i) {
      transpose_arrays_s32_4x4(in + 64 * i + 4 * j, out + 64 * j + 4 * i);
    }
  }
}

static INLINE int32x4_t half_btf_neon(const int32_t *w0, const int32x4_t *n0,
                                      const int32_t *w1, const int32x4_t *n1,
                                      const int32x4_t v_bit) {
  int32x4_t x;
  x = vmulq_n_s32(*n0, *w0);
  x = vmlaq_n_s32(x, *n1, *w1);
  x = vrshlq_s32(x, v_bit);
  return x;
}

static INLINE int32x4_t half_btf_neon_m(const int32_t *w0, const int32x4_t *n0,
                                        const int32_t *w1, const int32x4_t *n1,
                                        const int32x4_t v_bit) {
  int32x4_t x;
  x = vmulq_n_s32(*n0, *w0);
  x = vmlsq_n_s32(x, *n1, *w1);
  x = vrshlq_s32(x, v_bit);
  return x;
}

static INLINE void av1_round_shift_rect_array_32_neon(const int32x4_t *input,
                                                      int32x4_t *output,
                                                      const int size,
                                                      const int bit,
                                                      const int val) {
  const int32x4_t sqrt2 = vdupq_n_s32(val);
  const int32x4_t v_bit = vdupq_n_s32(-bit);
  for (int i = 0; i < size; i++) {
    const int32x4_t r0 = vrshlq_s32(input[i], v_bit);
    const int32x4_t r1 = vmulq_s32(sqrt2, r0);
    output[i] = vrshrq_n_s32(r1, NewSqrt2Bits);
  }
}

#define btf_32_neon_type0(w0, w1, in0, in1, out0, out1, v_cos_bit) \
  do {                                                             \
    out0 = vmulq_n_s32(in0, w0);                                   \
    out0 = vmlaq_n_s32(out0, in1, w1);                             \
    out0 = vrshlq_s32(out0, v_cos_bit);                            \
    out1 = vmulq_n_s32(in0, w1);                                   \
    out1 = vmlsq_n_s32(out1, in1, w0);                             \
    out1 = vrshlq_s32(out1, v_cos_bit);                            \
  } while (0)

#define btf_32_neon_type1(w0, w1, in0, in1, out0, out1, bit) \
  do {                                                       \
    btf_32_neon_type0(w1, w0, in1, in0, out0, out1, bit);    \
  } while (0)

#define LOAD_BUFFER_4XH(w, h)                                                \
  static INLINE void load_buffer_##w##x##h(const int16_t *input,             \
                                           int32x4_t *in, int stride,        \
                                           int fliplr, const int shift) {    \
    int16x4_t tmp[(w) * (h) / 4];                                            \
    for (int i = 0; i < (h); ++i) {                                          \
      for (int j = 0; j < (w) / 4; ++j) {                                    \
        tmp[i * ((w) / 4) + j] = vld1_s16(input + i * stride + j * 4);       \
      }                                                                      \
    }                                                                        \
    int16x4_t tmp2[(w) * (h) / 4];                                           \
    const int16x4_t *t = tmp;                                                \
    if (fliplr) {                                                            \
      for (int i = 0; i < (h); ++i) {                                        \
        for (int j = 0; j < (w) / 4; ++j) {                                  \
          tmp2[i * ((w) / 4) + ((w) / 4) - j - 1] =                          \
              vrev64_s16(tmp[i * ((w) / 4) + j]);                            \
        }                                                                    \
      }                                                                      \
      t = tmp2;                                                              \
    }                                                                        \
    const int32x4_t v_shift = vdupq_n_s32(shift);                            \
    for (int i = 0; i < (h); ++i) {                                          \
      for (int j = 0; j < (w) / 4; ++j) {                                    \
        in[i + (h)*j] = vshlq_s32(vmovl_s16(t[i * ((w) / 4) + j]), v_shift); \
      }                                                                      \
    }                                                                        \
  }

#define LOAD_BUFFER_WXH(w, h)                                             \
  static INLINE void load_buffer_##w##x##h(const int16_t *input,          \
                                           int32x4_t *in, int stride,     \
                                           int fliplr, const int shift) { \
    int16x8_t tmp[(w) * (h) / 8];                                         \
    for (int i = 0; i < (h); ++i) {                                       \
      for (int j = 0; j < (w) / 8; ++j) {                                 \
        tmp[i * ((w) / 8) + j] = vld1q_s16(input + i * stride + j * 8);   \
      }                                                                   \
    }                                                                     \
    int16x8_t tmp2[(w) * (h) / 8];                                        \
    const int16x8_t *t = tmp;                                             \
    if (fliplr) {                                                         \
      for (int i = 0; i < (h); ++i) {                                     \
        for (int j = 0; j < (w) / 8; ++j) {                               \
          int16x8_t a = vrev64q_s16(tmp[i * ((w) / 8) + j]);              \
          tmp2[i * ((w) / 8) + ((w) / 8) - j - 1] = vextq_s16(a, a, 4);   \
        }                                                                 \
      }                                                                   \
      t = tmp2;                                                           \
    }                                                                     \
    const int32x4_t v_shift = vdupq_n_s32(shift);                         \
    for (int i = 0; i < (h); ++i) {                                       \
      for (int j = 0; j < (w) / 8; ++j) {                                 \
        int16x8_t a = t[i * ((w) / 8) + j];                               \
        in[i + (h) * (2 * j + 0)] =                                       \
            vshlq_s32(vmovl_s16(vget_low_s16(a)), v_shift);               \
        in[i + (h) * (2 * j + 1)] =                                       \
            vshlq_s32(vmovl_s16(vget_high_s16(a)), v_shift);              \
      }                                                                   \
    }                                                                     \
  }

LOAD_BUFFER_4XH(4, 4)
LOAD_BUFFER_4XH(4, 8)
LOAD_BUFFER_4XH(4, 16)
LOAD_BUFFER_WXH(8, 4)
LOAD_BUFFER_WXH(8, 8)
LOAD_BUFFER_WXH(8, 16)
LOAD_BUFFER_WXH(8, 32)
LOAD_BUFFER_WXH(16, 4)
LOAD_BUFFER_WXH(16, 8)
LOAD_BUFFER_WXH(16, 16)
LOAD_BUFFER_WXH(16, 32)
LOAD_BUFFER_WXH(16, 64)
LOAD_BUFFER_WXH(32, 8)
LOAD_BUFFER_WXH(32, 16)
LOAD_BUFFER_WXH(32, 32)
LOAD_BUFFER_WXH(32, 64)
LOAD_BUFFER_WXH(64, 16)
LOAD_BUFFER_WXH(64, 32)
LOAD_BUFFER_WXH(64, 64)

#define STORE_BUFFER_WXH(w, h)                                   \
  static INLINE void store_buffer_##w##x##h(const int32x4_t *in, \
                                            int32_t *out) {      \
    for (int i = 0; i < (w); ++i) {                              \
      for (int j = 0; j < (h) / 4; ++j) {                        \
        vst1q_s32(out, in[i + j * (w)]);                         \
        out += 4;                                                \
      }                                                          \
    }                                                            \
  }

STORE_BUFFER_WXH(4, 4)
STORE_BUFFER_WXH(4, 8)
STORE_BUFFER_WXH(4, 16)
STORE_BUFFER_WXH(8, 4)
STORE_BUFFER_WXH(8, 8)
STORE_BUFFER_WXH(8, 16)
STORE_BUFFER_WXH(8, 32)
STORE_BUFFER_WXH(16, 4)
STORE_BUFFER_WXH(16, 8)
STORE_BUFFER_WXH(16, 16)
STORE_BUFFER_WXH(16, 32)
STORE_BUFFER_WXH(32, 8)
STORE_BUFFER_WXH(32, 16)
STORE_BUFFER_WXH(32, 32)
STORE_BUFFER_WXH(64, 16)
STORE_BUFFER_WXH(64, 32)

static void fdct4x4_neon(int32x4_t *in, int32x4_t *out, int bit) {
  const int32_t *cospi = cospi_arr(bit);
  const int32x4_t cospi32 = vdupq_n_s32(cospi[32]);
  const int32x4_t cospi48 = vdupq_n_s32(cospi[48]);
  const int32x4_t cospi16 = vdupq_n_s32(cospi[16]);
  int32x4_t s0, s1, s2, s3;
  int32x4_t u0, u1, u2, u3;
  int32x4_t v0, v2;

  s0 = vaddq_s32(in[0], in[3]);
  s3 = vsubq_s32(in[0], in[3]);
  s1 = vaddq_s32(in[1], in[2]);
  s2 = vsubq_s32(in[1], in[2]);

  u0 = vmulq_s32(s0, cospi32);
  u1 = vmulq_s32(s1, cospi32);
  u2 = vaddq_s32(u0, u1);
  v0 = vsubq_s32(u0, u1);
  const int32x4_t v_bit = vdupq_n_s32(-bit);
  u0 = vrshlq_s32(u2, v_bit);
  u2 = vrshlq_s32(v0, v_bit);

  v0 = vmulq_s32(s2, cospi48);
  v2 = vmlaq_s32(v0, s3, cospi16);

  u1 = vrshlq_s32(v2, v_bit);

  v0 = vmulq_s32(s3, cospi48);
  v2 = vmlsq_s32(v0, s2, cospi16);

  u3 = vrshlq_s32(v2, v_bit);

  out[0] = u0;
  out[1] = u1;
  out[2] = u2;
  out[3] = u3;
}

static void fadst4x4_neon(int32x4_t *in, int32x4_t *out, int bit) {
  const int32_t *sinpi = sinpi_arr(bit);
  const int32x4_t sinpi4x = vld1q_s32(&sinpi[1]);

  const int32x4_t sinpi1 = vdupq_lane_s32(vget_low_s32(sinpi4x), 0);
  const int32x4_t sinpi2 = vdupq_lane_s32(vget_low_s32(sinpi4x), 1);
  const int32x4_t sinpi3 = vdupq_lane_s32(vget_high_s32(sinpi4x), 0);
  const int32x4_t sinpi4 = vdupq_lane_s32(vget_high_s32(sinpi4x), 1);
  int32x4_t t;
  int32x4_t s0, s1, s2, s3, s7;
  int32x4_t x0, x1, x2, x3;

  s0 = vmulq_s32(in[0], sinpi1);
  s1 = vmulq_s32(in[0], sinpi4);
  t = vaddq_s32(in[0], in[1]);
  x3 = vmulq_s32(in[2], sinpi3);
  s7 = vsubq_s32(t, in[3]);

  t = vmlaq_s32(s0, in[1], sinpi2);
  x0 = vmlaq_s32(t, in[3], sinpi4);
  x1 = vmulq_s32(s7, sinpi3);
  t = vmlsq_s32(s1, in[1], sinpi1);
  x2 = vmlaq_s32(t, in[3], sinpi2);

  s0 = vaddq_s32(x0, x3);
  s1 = x1;
  s2 = vsubq_s32(x2, x3);
  t = vsubq_s32(x2, x0);
  s3 = vaddq_s32(t, x3);

  const int32x4_t v_bit = vdupq_n_s32(-bit);
  out[0] = vrshlq_s32(s0, v_bit);
  out[1] = vrshlq_s32(s1, v_bit);
  out[2] = vrshlq_s32(s2, v_bit);
  out[3] = vrshlq_s32(s3, v_bit);
}

static void idtx4x4_neon(int32x4_t *in, int32x4_t *out, int bit) {
  (void)bit;
  int32x4_t fact = vdupq_n_s32(NewSqrt2);
  int32x4_t a_low;

  for (int i = 0; i < 4; i++) {
    a_low = vmulq_s32(in[i], fact);
    out[i] = vrshrq_n_s32(a_low, NewSqrt2Bits);
  }
}

void av1_fwd_txfm2d_4x4_neon(const int16_t *input, int32_t *coeff,
                             int input_stride, TX_TYPE tx_type, int bd) {
  (void)bd;

  int ud_flip, lr_flip;
  get_flip_cfg(tx_type, &ud_flip, &lr_flip);
  ud_adjust_input_and_stride(ud_flip, &input, &input_stride, 4);

  // Workspace for column/row-wise transforms.
  int32x4_t buf[4];

  switch (tx_type) {
    case DCT_DCT:
      load_buffer_4x4(input, buf, input_stride, 0, 2);
      fdct4x4_neon(buf, buf, av1_fwd_cos_bit_col[0][0]);
      transpose_arrays_s32_4x4(buf, buf);
      fdct4x4_neon(buf, buf, av1_fwd_cos_bit_row[0][0]);
      store_buffer_4x4(buf, coeff);
      break;
    case ADST_DCT:
      load_buffer_4x4(input, buf, input_stride, 0, 2);
      fadst4x4_neon(buf, buf, av1_fwd_cos_bit_col[0][0]);
      transpose_arrays_s32_4x4(buf, buf);
      fdct4x4_neon(buf, buf, av1_fwd_cos_bit_row[0][0]);
      store_buffer_4x4(buf, coeff);
      break;
    case DCT_ADST:
      load_buffer_4x4(input, buf, input_stride, 0, 2);
      fdct4x4_neon(buf, buf, av1_fwd_cos_bit_col[0][0]);
      transpose_arrays_s32_4x4(buf, buf);
      fadst4x4_neon(buf, buf, av1_fwd_cos_bit_row[0][0]);
      store_buffer_4x4(buf, coeff);
      break;
    case ADST_ADST:
      load_buffer_4x4(input, buf, input_stride, 0, 2);
      fadst4x4_neon(buf, buf, av1_fwd_cos_bit_col[0][0]);
      transpose_arrays_s32_4x4(buf, buf);
      fadst4x4_neon(buf, buf, av1_fwd_cos_bit_row[0][0]);
      store_buffer_4x4(buf, coeff);
      break;
    case FLIPADST_DCT:
      load_buffer_4x4(input, buf, input_stride, 0, 2);
      fadst4x4_neon(buf, buf, av1_fwd_cos_bit_col[0][0]);
      transpose_arrays_s32_4x4(buf, buf);
      fdct4x4_neon(buf, buf, av1_fwd_cos_bit_row[0][0]);
      store_buffer_4x4(buf, coeff);
      break;
    case DCT_FLIPADST:
      load_buffer_4x4(input, buf, input_stride, 1, 2);
      fdct4x4_neon(buf, buf, av1_fwd_cos_bit_col[0][0]);
      transpose_arrays_s32_4x4(buf, buf);
      fadst4x4_neon(buf, buf, av1_fwd_cos_bit_row[0][0]);
      store_buffer_4x4(buf, coeff);
      break;
    case FLIPADST_FLIPADST:
      load_buffer_4x4(input, buf, input_stride, 1, 2);
      fadst4x4_neon(buf, buf, av1_fwd_cos_bit_col[0][0]);
      transpose_arrays_s32_4x4(buf, buf);
      fadst4x4_neon(buf, buf, av1_fwd_cos_bit_row[0][0]);
      store_buffer_4x4(buf, coeff);
      break;
    case ADST_FLIPADST:
      load_buffer_4x4(input, buf, input_stride, 1, 2);
      fadst4x4_neon(buf, buf, av1_fwd_cos_bit_col[0][0]);
      transpose_arrays_s32_4x4(buf, buf);
      fadst4x4_neon(buf, buf, av1_fwd_cos_bit_row[0][0]);
      store_buffer_4x4(buf, coeff);
      break;
    case FLIPADST_ADST:
      load_buffer_4x4(input, buf, input_stride, 0, 2);
      fadst4x4_neon(buf, buf, av1_fwd_cos_bit_col[0][0]);
      transpose_arrays_s32_4x4(buf, buf);
      fadst4x4_neon(buf, buf, av1_fwd_cos_bit_row[0][0]);
      store_buffer_4x4(buf, coeff);
      break;
    case IDTX:
      load_buffer_4x4(input, buf, input_stride, 0, 2);
      idtx4x4_neon(buf, buf, av1_fwd_cos_bit_col[0][0]);
      transpose_arrays_s32_4x4(buf, buf);
      idtx4x4_neon(buf, buf, av1_fwd_cos_bit_row[0][0]);
      store_buffer_4x4(buf, coeff);
      break;
    case V_DCT:
      load_buffer_4x4(input, buf, input_stride, 0, 2);
      fdct4x4_neon(buf, buf, av1_fwd_cos_bit_col[0][0]);
      transpose_arrays_s32_4x4(buf, buf);
      idtx4x4_neon(buf, buf, av1_fwd_cos_bit_row[0][0]);
      store_buffer_4x4(buf, coeff);
      break;
    case H_DCT:
      load_buffer_4x4(input, buf, input_stride, 0, 2);
      idtx4x4_neon(buf, buf, av1_fwd_cos_bit_row[0][0]);
      transpose_arrays_s32_4x4(buf, buf);
      fdct4x4_neon(buf, buf, av1_fwd_cos_bit_col[0][0]);
      store_buffer_4x4(buf, coeff);
      break;
    case V_ADST:
      load_buffer_4x4(input, buf, input_stride, 0, 2);
      fadst4x4_neon(buf, buf, av1_fwd_cos_bit_col[0][0]);
      transpose_arrays_s32_4x4(buf, buf);
      idtx4x4_neon(buf, buf, av1_fwd_cos_bit_row[0][0]);
      store_buffer_4x4(buf, coeff);
      break;
    case H_ADST:
      load_buffer_4x4(input, buf, input_stride, 0, 2);
      idtx4x4_neon(buf, buf, av1_fwd_cos_bit_row[0][0]);
      transpose_arrays_s32_4x4(buf, buf);
      fadst4x4_neon(buf, buf, av1_fwd_cos_bit_col[0][0]);
      store_buffer_4x4(buf, coeff);
      break;
    case V_FLIPADST:
      load_buffer_4x4(input, buf, input_stride, 0, 2);
      fadst4x4_neon(buf, buf, av1_fwd_cos_bit_row[0][0]);
      transpose_arrays_s32_4x4(buf, buf);
      idtx4x4_neon(buf, buf, av1_fwd_cos_bit_row[0][0]);
      store_buffer_4x4(buf, coeff);
      break;
    case H_FLIPADST:
      load_buffer_4x4(input, buf, input_stride, 1, 2);
      idtx4x4_neon(buf, buf, av1_fwd_cos_bit_row[0][0]);
      transpose_arrays_s32_4x4(buf, buf);
      fadst4x4_neon(buf, buf, av1_fwd_cos_bit_row[0][0]);
      store_buffer_4x4(buf, coeff);
      break;
    default: assert(0);
  }
}

static INLINE void col_txfm_8x8_rounding(int32x4_t *in, const int shift) {
  const int32x4_t v_shift = vdupq_n_s32(shift);
  in[0] = vrshlq_s32(in[0], v_shift);
  in[1] = vrshlq_s32(in[1], v_shift);
  in[2] = vrshlq_s32(in[2], v_shift);
  in[3] = vrshlq_s32(in[3], v_shift);
  in[4] = vrshlq_s32(in[4], v_shift);
  in[5] = vrshlq_s32(in[5], v_shift);
  in[6] = vrshlq_s32(in[6], v_shift);
  in[7] = vrshlq_s32(in[7], v_shift);
  in[8] = vrshlq_s32(in[8], v_shift);
  in[9] = vrshlq_s32(in[9], v_shift);
  in[10] = vrshlq_s32(in[10], v_shift);
  in[11] = vrshlq_s32(in[11], v_shift);
  in[12] = vrshlq_s32(in[12], v_shift);
  in[13] = vrshlq_s32(in[13], v_shift);
  in[14] = vrshlq_s32(in[14], v_shift);
  in[15] = vrshlq_s32(in[15], v_shift);
}

static INLINE void col_txfm_4x8_rounding(int32x4_t *in, const int shift) {
  const int32x4_t v_shift = vdupq_n_s32(shift);
  in[0] = vrshlq_s32(in[0], v_shift);
  in[1] = vrshlq_s32(in[1], v_shift);
  in[2] = vrshlq_s32(in[2], v_shift);
  in[3] = vrshlq_s32(in[3], v_shift);
  in[4] = vrshlq_s32(in[4], v_shift);
  in[5] = vrshlq_s32(in[5], v_shift);
  in[6] = vrshlq_s32(in[6], v_shift);
  in[7] = vrshlq_s32(in[7], v_shift);
}

static void fdct4x8_neon(int32x4_t *in, int32x4_t *out, int bit, int howmany,
                         int stride) {
  (void)howmany;
  (void)stride;
  const int32_t *cospi = cospi_arr(bit);
  const int32x4_t v_bit = vdupq_n_s32(-bit);
  int32x4_t u[8], v[8];

  // stage 0-1
  u[0] = vaddq_s32(in[0], in[7]);
  v[7] = vsubq_s32(in[0], in[7]);
  u[1] = vaddq_s32(in[1], in[6]);
  u[6] = vsubq_s32(in[1], in[6]);
  u[2] = vaddq_s32(in[2], in[5]);
  u[5] = vsubq_s32(in[2], in[5]);
  u[3] = vaddq_s32(in[3], in[4]);
  v[4] = vsubq_s32(in[3], in[4]);

  // stage 2
  v[0] = vaddq_s32(u[0], u[3]);
  v[3] = vsubq_s32(u[0], u[3]);
  v[1] = vaddq_s32(u[1], u[2]);
  v[2] = vsubq_s32(u[1], u[2]);

  v[5] = vmulq_n_s32(u[6], cospi[32]);
  v[5] = vmlsq_n_s32(v[5], u[5], cospi[32]);
  v[5] = vrshlq_s32(v[5], v_bit);

  u[0] = vmulq_n_s32(u[5], cospi[32]);
  v[6] = vmlaq_n_s32(u[0], u[6], cospi[32]);
  v[6] = vrshlq_s32(v[6], v_bit);

  // stage 3
  // type 0
  v[0] = vmulq_n_s32(v[0], cospi[32]);
  v[1] = vmulq_n_s32(v[1], cospi[32]);
  u[0] = vaddq_s32(v[0], v[1]);
  u[0] = vrshlq_s32(u[0], v_bit);

  u[1] = vsubq_s32(v[0], v[1]);
  u[1] = vrshlq_s32(u[1], v_bit);

  // type 1
  v[0] = vmulq_n_s32(v[2], cospi[48]);
  u[2] = vmlaq_n_s32(v[0], v[3], cospi[16]);
  u[2] = vrshlq_s32(u[2], v_bit);

  v[1] = vmulq_n_s32(v[3], cospi[48]);
  u[3] = vmlsq_n_s32(v[1], v[2], cospi[16]);
  u[3] = vrshlq_s32(u[3], v_bit);

  u[4] = vaddq_s32(v[4], v[5]);
  u[5] = vsubq_s32(v[4], v[5]);
  u[6] = vsubq_s32(v[7], v[6]);
  u[7] = vaddq_s32(v[7], v[6]);

  // stage 4-5
  v[0] = vmulq_n_s32(u[4], cospi[56]);
  v[0] = vmlaq_n_s32(v[0], u[7], cospi[8]);
  out[1] = vrshlq_s32(v[0], v_bit);

  v[1] = vmulq_n_s32(u[7], cospi[56]);
  v[0] = vmlsq_n_s32(v[1], u[4], cospi[8]);
  out[7] = vrshlq_s32(v[0], v_bit);

  v[0] = vmulq_n_s32(u[5], cospi[24]);
  v[0] = vmlaq_n_s32(v[0], u[6], cospi[40]);
  out[5] = vrshlq_s32(v[0], v_bit);

  v[1] = vmulq_n_s32(u[6], cospi[24]);
  v[0] = vmlsq_n_s32(v[1], u[5], cospi[40]);
  out[3] = vrshlq_s32(v[0], v_bit);

  out[0] = u[0];
  out[4] = u[1];
  out[2] = u[2];
  out[6] = u[3];
}

static void fdct8x8_neon(int32x4_t *in, int32x4_t *out, int bit, int howmany,
                         int stride) {
  for (int i = 0; i < howmany; ++i) {
    fdct4x8_neon(in + i * stride, out + i * stride, bit, 1, 1);
  }
}

static void fadst8x8_neon(int32x4_t *in, int32x4_t *out, int bit, int howmany,
                          int stride) {
  const int32_t *cospi = cospi_arr(bit);

  const int32x4_t v_bit = vdupq_n_s32(-bit);
  int32x4_t u0, u1, u2, u3, u4, u5, u6, u7;
  int32x4_t v0, v1, v2, v3, v4, v5, v6, v7;
  int32x4_t x, y;

  for (int col = 0; col < howmany; ++col) {
    // stage 0-1
    u0 = in[0 + col * stride];
    u1 = vnegq_s32(in[7 + col * stride]);
    u2 = vnegq_s32(in[3 + col * stride]);
    u3 = in[4 + col * stride];
    u4 = vnegq_s32(in[1 + col * stride]);
    u5 = in[6 + col * stride];
    u6 = in[2 + col * stride];
    u7 = vnegq_s32(in[5 + col * stride]);

    // stage 2
    v0 = u0;
    v1 = u1;

    x = vmulq_n_s32(u2, cospi[32]);
    y = vmulq_n_s32(u3, cospi[32]);
    v2 = vaddq_s32(x, y);
    v2 = vrshlq_s32(v2, v_bit);

    v3 = vsubq_s32(x, y);
    v3 = vrshlq_s32(v3, v_bit);

    v4 = u4;
    v5 = u5;

    x = vmulq_n_s32(u6, cospi[32]);
    y = vmulq_n_s32(u7, cospi[32]);
    v6 = vaddq_s32(x, y);
    v6 = vrshlq_s32(v6, v_bit);

    v7 = vsubq_s32(x, y);
    v7 = vrshlq_s32(v7, v_bit);

    // stage 3
    u0 = vaddq_s32(v0, v2);
    u1 = vaddq_s32(v1, v3);
    u2 = vsubq_s32(v0, v2);
    u3 = vsubq_s32(v1, v3);
    u4 = vaddq_s32(v4, v6);
    u5 = vaddq_s32(v5, v7);
    u6 = vsubq_s32(v4, v6);
    u7 = vsubq_s32(v5, v7);

    // stage 4
    v0 = u0;
    v1 = u1;
    v2 = u2;
    v3 = u3;

    v4 = vmulq_n_s32(u4, cospi[16]);
    v4 = vmlaq_n_s32(v4, u5, cospi[48]);
    v4 = vrshlq_s32(v4, v_bit);

    v5 = vmulq_n_s32(u4, cospi[48]);
    v5 = vmlsq_n_s32(v5, u5, cospi[16]);
    v5 = vrshlq_s32(v5, v_bit);

    v6 = vmulq_n_s32(u7, cospi[16]);
    v6 = vmlsq_n_s32(v6, u6, cospi[48]);
    v6 = vrshlq_s32(v6, v_bit);

    v7 = vmulq_n_s32(u6, cospi[16]);
    v7 = vmlaq_n_s32(v7, u7, cospi[48]);
    v7 = vrshlq_s32(v7, v_bit);

    // stage 5
    u0 = vaddq_s32(v0, v4);
    u1 = vaddq_s32(v1, v5);
    u2 = vaddq_s32(v2, v6);
    u3 = vaddq_s32(v3, v7);
    u4 = vsubq_s32(v0, v4);
    u5 = vsubq_s32(v1, v5);
    u6 = vsubq_s32(v2, v6);
    u7 = vsubq_s32(v3, v7);

    // stage 6
    v0 = vmulq_n_s32(u0, cospi[4]);
    v0 = vmlaq_n_s32(v0, u1, cospi[60]);
    v0 = vrshlq_s32(v0, v_bit);

    v1 = vmulq_n_s32(u0, cospi[60]);
    v1 = vmlsq_n_s32(v1, u1, cospi[4]);
    v1 = vrshlq_s32(v1, v_bit);

    v2 = vmulq_n_s32(u2, cospi[20]);
    v2 = vmlaq_n_s32(v2, u3, cospi[44]);
    v2 = vrshlq_s32(v2, v_bit);

    v3 = vmulq_n_s32(u2, cospi[44]);
    v3 = vmlsq_n_s32(v3, u3, cospi[20]);
    v3 = vrshlq_s32(v3, v_bit);

    v4 = vmulq_n_s32(u4, cospi[36]);
    v4 = vmlaq_n_s32(v4, u5, cospi[28]);
    v4 = vrshlq_s32(v4, v_bit);

    v5 = vmulq_n_s32(u4, cospi[28]);
    v5 = vmlsq_n_s32(v5, u5, cospi[36]);
    v5 = vrshlq_s32(v5, v_bit);

    x = vmulq_n_s32(u6, cospi[52]);
    v6 = vmlaq_n_s32(x, u7, cospi[12]);
    v6 = vrshlq_s32(v6, v_bit);

    v7 = vmulq_n_s32(u6, cospi[12]);
    v7 = vmlsq_n_s32(v7, u7, cospi[52]);
    v7 = vrshlq_s32(v7, v_bit);

    // stage 7
    out[0 + col * stride] = v1;
    out[1 + col * stride] = v6;
    out[2 + col * stride] = v3;
    out[3 + col * stride] = v4;
    out[4 + col * stride] = v5;
    out[5 + col * stride] = v2;
    out[6 + col * stride] = v7;
    out[7 + col * stride] = v0;
  }
}

static void idtx8x8_neon(int32x4_t *in, int32x4_t *out, int bit, int howmany,
                         int stride) {
  (void)bit;
  for (int i = 0; i < howmany; i += 1) {
    out[0 + stride * i] = vshlq_n_s32(in[0 + stride * i], 1);
    out[1 + stride * i] = vshlq_n_s32(in[1 + stride * i], 1);
    out[2 + stride * i] = vshlq_n_s32(in[2 + stride * i], 1);
    out[3 + stride * i] = vshlq_n_s32(in[3 + stride * i], 1);
    out[4 + stride * i] = vshlq_n_s32(in[4 + stride * i], 1);
    out[5 + stride * i] = vshlq_n_s32(in[5 + stride * i], 1);
    out[6 + stride * i] = vshlq_n_s32(in[6 + stride * i], 1);
    out[7 + stride * i] = vshlq_n_s32(in[7 + stride * i], 1);
  }
}

#if !CONFIG_REALTIME_ONLY
static void idtx32x8_neon(int32x4_t *in, int32x4_t *out, int bit, int howmany,
                          int stride) {
  (void)bit;
  for (int j = 0; j < howmany; j++) {
    out[j * stride + 0] = vshlq_n_s32(in[j * stride + 0], 1);
    out[j * stride + 1] = vshlq_n_s32(in[j * stride + 1], 1);
    out[j * stride + 2] = vshlq_n_s32(in[j * stride + 2], 1);
    out[j * stride + 3] = vshlq_n_s32(in[j * stride + 3], 1);
    out[j * stride + 4] = vshlq_n_s32(in[j * stride + 4], 1);
    out[j * stride + 5] = vshlq_n_s32(in[j * stride + 5], 1);
    out[j * stride + 6] = vshlq_n_s32(in[j * stride + 6], 1);
    out[j * stride + 7] = vshlq_n_s32(in[j * stride + 7], 1);
  }
}
#endif

void av1_fwd_txfm2d_8x8_neon(const int16_t *input, int32_t *coeff, int stride,
                             TX_TYPE tx_type, int bd) {
  (void)bd;

  int ud_flip, lr_flip;
  get_flip_cfg(tx_type, &ud_flip, &lr_flip);
  ud_adjust_input_and_stride(ud_flip, &input, &stride, 8);

  // Workspaces for column/row-wise transforms.
  int32x4_t buf0[16], buf1[16];

  switch (tx_type) {
    case DCT_DCT:
      load_buffer_8x8(input, buf0, stride, 0, 2);
      fdct8x8_neon(buf0, buf0, av1_fwd_cos_bit_col[1][1], 2, 8);
      col_txfm_8x8_rounding(buf0, -1);
      transpose_arrays_s32_8x8(buf0, buf1);
      fdct8x8_neon(buf1, buf1, av1_fwd_cos_bit_row[1][1], 2, 8);
      store_buffer_8x8(buf1, coeff);
      break;
    case ADST_DCT:
      load_buffer_8x8(input, buf0, stride, 0, 2);
      fadst8x8_neon(buf0, buf0, av1_fwd_cos_bit_col[1][1], 2, 8);
      col_txfm_8x8_rounding(buf0, -1);
      transpose_arrays_s32_8x8(buf0, buf1);
      fdct8x8_neon(buf1, buf1, av1_fwd_cos_bit_row[1][1], 2, 8);
      store_buffer_8x8(buf1, coeff);
      break;
    case DCT_ADST:
      load_buffer_8x8(input, buf0, stride, 0, 2);
      fdct8x8_neon(buf0, buf0, av1_fwd_cos_bit_col[1][1], 2, 8);
      col_txfm_8x8_rounding(buf0, -1);
      transpose_arrays_s32_8x8(buf0, buf1);
      fadst8x8_neon(buf1, buf1, av1_fwd_cos_bit_row[1][1], 2, 8);
      store_buffer_8x8(buf1, coeff);
      break;
    case ADST_ADST:
      load_buffer_8x8(input, buf0, stride, 0, 2);
      fadst8x8_neon(buf0, buf0, av1_fwd_cos_bit_col[1][1], 2, 8);
      col_txfm_8x8_rounding(buf0, -1);
      transpose_arrays_s32_8x8(buf0, buf1);
      fadst8x8_neon(buf1, buf1, av1_fwd_cos_bit_row[1][1], 2, 8);
      store_buffer_8x8(buf1, coeff);
      break;
    case FLIPADST_DCT:
      load_buffer_8x8(input, buf0, stride, 0, 2);
      fadst8x8_neon(buf0, buf0, av1_fwd_cos_bit_col[1][1], 2, 8);
      col_txfm_8x8_rounding(buf0, -1);
      transpose_arrays_s32_8x8(buf0, buf1);
      fdct8x8_neon(buf1, buf1, av1_fwd_cos_bit_row[1][1], 2, 8);
      store_buffer_8x8(buf1, coeff);
      break;
    case DCT_FLIPADST:
      load_buffer_8x8(input, buf0, stride, 1, 2);
      fdct8x8_neon(buf0, buf0, av1_fwd_cos_bit_col[1][1], 2, 8);
      col_txfm_8x8_rounding(buf0, -1);
      transpose_arrays_s32_8x8(buf0, buf1);
      fadst8x8_neon(buf1, buf1, av1_fwd_cos_bit_row[1][1], 2, 8);
      store_buffer_8x8(buf1, coeff);
      break;
    case FLIPADST_FLIPADST:
      load_buffer_8x8(input, buf0, stride, 1, 2);
      fadst8x8_neon(buf0, buf0, av1_fwd_cos_bit_col[1][1], 2, 8);
      col_txfm_8x8_rounding(buf0, -1);
      transpose_arrays_s32_8x8(buf0, buf1);
      fadst8x8_neon(buf1, buf1, av1_fwd_cos_bit_row[1][1], 2, 8);
      store_buffer_8x8(buf1, coeff);
      break;
    case ADST_FLIPADST:
      load_buffer_8x8(input, buf0, stride, 1, 2);
      fadst8x8_neon(buf0, buf0, av1_fwd_cos_bit_col[1][1], 2, 8);
      col_txfm_8x8_rounding(buf0, -1);
      transpose_arrays_s32_8x8(buf0, buf1);
      fadst8x8_neon(buf1, buf1, av1_fwd_cos_bit_row[1][1], 2, 8);
      store_buffer_8x8(buf1, coeff);
      break;
    case FLIPADST_ADST:
      load_buffer_8x8(input, buf0, stride, 0, 2);
      fadst8x8_neon(buf0, buf0, av1_fwd_cos_bit_col[1][1], 2, 8);
      col_txfm_8x8_rounding(buf0, -1);
      transpose_arrays_s32_8x8(buf0, buf1);
      fadst8x8_neon(buf1, buf1, av1_fwd_cos_bit_row[1][1], 2, 8);
      store_buffer_8x8(buf1, coeff);
      break;
    case IDTX:
      load_buffer_8x8(input, buf0, stride, 0, 2);
      idtx8x8_neon(buf0, buf0, av1_fwd_cos_bit_col[1][1], 2, 8);
      col_txfm_8x8_rounding(buf0, -1);
      transpose_arrays_s32_8x8(buf0, buf1);
      idtx8x8_neon(buf1, buf1, av1_fwd_cos_bit_col[1][1], 2, 8);
      store_buffer_8x8(buf1, coeff);
      break;
    case V_DCT:
      load_buffer_8x8(input, buf0, stride, 0, 2);
      fdct8x8_neon(buf0, buf0, av1_fwd_cos_bit_col[1][1], 2, 8);
      col_txfm_8x8_rounding(buf0, -1);
      transpose_arrays_s32_8x8(buf0, buf1);
      idtx8x8_neon(buf1, buf1, av1_fwd_cos_bit_col[1][1], 2, 8);
      store_buffer_8x8(buf1, coeff);
      break;
    case H_DCT:
      load_buffer_8x8(input, buf0, stride, 0, 2);
      idtx8x8_neon(buf0, buf0, av1_fwd_cos_bit_col[1][1], 2, 8);
      col_txfm_8x8_rounding(buf0, -1);
      transpose_arrays_s32_8x8(buf0, buf1);
      fdct8x8_neon(buf1, buf1, av1_fwd_cos_bit_col[1][1], 2, 8);
      store_buffer_8x8(buf1, coeff);
      break;
    case V_ADST:
      load_buffer_8x8(input, buf0, stride, 0, 2);
      fadst8x8_neon(buf0, buf0, av1_fwd_cos_bit_col[1][1], 2, 8);
      col_txfm_8x8_rounding(buf0, -1);
      transpose_arrays_s32_8x8(buf0, buf1);
      idtx8x8_neon(buf1, buf1, av1_fwd_cos_bit_col[1][1], 2, 8);
      store_buffer_8x8(buf1, coeff);
      break;
    case H_ADST:
      load_buffer_8x8(input, buf0, stride, 0, 2);
      idtx8x8_neon(buf0, buf0, av1_fwd_cos_bit_col[1][1], 2, 8);
      col_txfm_8x8_rounding(buf0, -1);
      transpose_arrays_s32_8x8(buf0, buf1);
      fadst8x8_neon(buf1, buf1, av1_fwd_cos_bit_col[1][1], 2, 8);
      store_buffer_8x8(buf1, coeff);
      break;
    case V_FLIPADST:
      load_buffer_8x8(input, buf0, stride, 0, 2);
      fadst8x8_neon(buf0, buf0, av1_fwd_cos_bit_col[1][1], 2, 8);
      col_txfm_8x8_rounding(buf0, -1);
      transpose_arrays_s32_8x8(buf0, buf1);
      idtx8x8_neon(buf1, buf1, av1_fwd_cos_bit_col[1][1], 2, 8);
      store_buffer_8x8(buf1, coeff);
      break;
    case H_FLIPADST:
      load_buffer_8x8(input, buf0, stride, 1, 2);
      idtx8x8_neon(buf0, buf0, av1_fwd_cos_bit_col[1][1], 2, 8);
      col_txfm_8x8_rounding(buf0, -1);
      transpose_arrays_s32_8x8(buf0, buf1);
      fadst8x8_neon(buf1, buf1, av1_fwd_cos_bit_col[1][1], 2, 8);
      store_buffer_8x8(buf1, coeff);
      break;
    default: assert(0);
  }
}

static void fdct16x16_neon(int32x4_t *in, int32x4_t *out, int bit,
                           const int howmany, const int stride) {
  const int32_t *cospi = cospi_arr(bit);
  const int32x4_t v_bit = vdupq_n_s32(-bit);
  int32x4_t u[16], v[16];

  // Calculate the column 0, 1, 2, 3
  for (int col = 0; col < howmany; ++col) {
    // stage 0
    // stage 1
    u[0] = vaddq_s32(in[0 + col * stride], in[15 + col * stride]);
    u[15] = vsubq_s32(in[0 + col * stride], in[15 + col * stride]);
    u[1] = vaddq_s32(in[1 + col * stride], in[14 + col * stride]);
    u[14] = vsubq_s32(in[1 + col * stride], in[14 + col * stride]);
    u[2] = vaddq_s32(in[2 + col * stride], in[13 + col * stride]);
    u[13] = vsubq_s32(in[2 + col * stride], in[13 + col * stride]);
    u[3] = vaddq_s32(in[3 + col * stride], in[12 + col * stride]);
    u[12] = vsubq_s32(in[3 + col * stride], in[12 + col * stride]);
    u[4] = vaddq_s32(in[4 + col * stride], in[11 + col * stride]);
    u[11] = vsubq_s32(in[4 + col * stride], in[11 + col * stride]);
    u[5] = vaddq_s32(in[5 + col * stride], in[10 + col * stride]);
    u[10] = vsubq_s32(in[5 + col * stride], in[10 + col * stride]);
    u[6] = vaddq_s32(in[6 + col * stride], in[9 + col * stride]);
    u[9] = vsubq_s32(in[6 + col * stride], in[9 + col * stride]);
    u[7] = vaddq_s32(in[7 + col * stride], in[8 + col * stride]);
    u[8] = vsubq_s32(in[7 + col * stride], in[8 + col * stride]);

    // stage 2
    v[0] = vaddq_s32(u[0], u[7]);
    v[7] = vsubq_s32(u[0], u[7]);
    v[1] = vaddq_s32(u[1], u[6]);
    v[6] = vsubq_s32(u[1], u[6]);
    v[2] = vaddq_s32(u[2], u[5]);
    v[5] = vsubq_s32(u[2], u[5]);
    v[3] = vaddq_s32(u[3], u[4]);
    v[4] = vsubq_s32(u[3], u[4]);
    v[8] = u[8];
    v[9] = u[9];

    v[10] = vmulq_n_s32(u[13], cospi[32]);
    v[10] = vmlsq_n_s32(v[10], u[10], cospi[32]);
    v[10] = vrshlq_s32(v[10], v_bit);

    v[13] = vmulq_n_s32(u[10], cospi[32]);
    v[13] = vmlaq_n_s32(v[13], u[13], cospi[32]);
    v[13] = vrshlq_s32(v[13], v_bit);

    v[11] = vmulq_n_s32(u[12], cospi[32]);
    v[11] = vmlsq_n_s32(v[11], u[11], cospi[32]);
    v[11] = vrshlq_s32(v[11], v_bit);

    v[12] = vmulq_n_s32(u[11], cospi[32]);
    v[12] = vmlaq_n_s32(v[12], u[12], cospi[32]);
    v[12] = vrshlq_s32(v[12], v_bit);
    v[14] = u[14];
    v[15] = u[15];

    // stage 3
    u[0] = vaddq_s32(v[0], v[3]);
    u[3] = vsubq_s32(v[0], v[3]);
    u[1] = vaddq_s32(v[1], v[2]);
    u[2] = vsubq_s32(v[1], v[2]);
    u[4] = v[4];

    u[5] = vmulq_n_s32(v[6], cospi[32]);
    u[5] = vmlsq_n_s32(u[5], v[5], cospi[32]);
    u[5] = vrshlq_s32(u[5], v_bit);

    u[6] = vmulq_n_s32(v[5], cospi[32]);
    u[6] = vmlaq_n_s32(u[6], v[6], cospi[32]);
    u[6] = vrshlq_s32(u[6], v_bit);

    u[7] = v[7];
    u[8] = vaddq_s32(v[8], v[11]);
    u[11] = vsubq_s32(v[8], v[11]);
    u[9] = vaddq_s32(v[9], v[10]);
    u[10] = vsubq_s32(v[9], v[10]);
    u[12] = vsubq_s32(v[15], v[12]);
    u[15] = vaddq_s32(v[15], v[12]);
    u[13] = vsubq_s32(v[14], v[13]);
    u[14] = vaddq_s32(v[14], v[13]);

    // stage 4
    u[0] = vmulq_n_s32(u[0], cospi[32]);
    u[1] = vmulq_n_s32(u[1], cospi[32]);
    v[0] = vaddq_s32(u[0], u[1]);
    v[0] = vrshlq_s32(v[0], v_bit);

    v[1] = vsubq_s32(u[0], u[1]);
    v[1] = vrshlq_s32(v[1], v_bit);

    v[2] = vmulq_n_s32(u[2], cospi[48]);
    v[2] = vmlaq_n_s32(v[2], u[3], cospi[16]);
    v[2] = vrshlq_s32(v[2], v_bit);

    v[3] = vmulq_n_s32(u[3], cospi[48]);
    v[3] = vmlsq_n_s32(v[3], u[2], cospi[16]);
    v[3] = vrshlq_s32(v[3], v_bit);

    v[4] = vaddq_s32(u[4], u[5]);
    v[5] = vsubq_s32(u[4], u[5]);
    v[6] = vsubq_s32(u[7], u[6]);
    v[7] = vaddq_s32(u[7], u[6]);
    v[8] = u[8];

    v[9] = vmulq_n_s32(u[14], cospi[48]);
    v[9] = vmlsq_n_s32(v[9], u[9], cospi[16]);
    v[9] = vrshlq_s32(v[9], v_bit);

    v[14] = vmulq_n_s32(u[9], cospi[48]);
    v[14] = vmlaq_n_s32(v[14], u[14], cospi[16]);
    v[14] = vrshlq_s32(v[14], v_bit);

    v[10] = vmulq_n_s32(u[13], -cospi[16]);
    v[10] = vmlsq_n_s32(v[10], u[10], cospi[48]);
    v[10] = vrshlq_s32(v[10], v_bit);

    v[13] = vmulq_n_s32(u[10], -cospi[16]);
    v[13] = vmlaq_n_s32(v[13], u[13], cospi[48]);
    v[13] = vrshlq_s32(v[13], v_bit);

    v[11] = u[11];
    v[12] = u[12];
    v[15] = u[15];

    // stage 5
    u[0] = v[0];
    u[1] = v[1];
    u[2] = v[2];
    u[3] = v[3];

    u[4] = vmulq_n_s32(v[4], cospi[56]);
    u[4] = vmlaq_n_s32(u[4], v[7], cospi[8]);
    u[4] = vrshlq_s32(u[4], v_bit);

    u[7] = vmulq_n_s32(v[7], cospi[56]);
    u[7] = vmlsq_n_s32(u[7], v[4], cospi[8]);
    u[7] = vrshlq_s32(u[7], v_bit);

    u[5] = vmulq_n_s32(v[5], cospi[24]);
    u[5] = vmlaq_n_s32(u[5], v[6], cospi[40]);
    u[5] = vrshlq_s32(u[5], v_bit);

    u[6] = vmulq_n_s32(v[6], cospi[24]);
    u[6] = vmlsq_n_s32(u[6], v[5], cospi[40]);
    u[6] = vrshlq_s32(u[6], v_bit);

    u[8] = vaddq_s32(v[8], v[9]);
    u[9] = vsubq_s32(v[8], v[9]);
    u[10] = vsubq_s32(v[11], v[10]);
    u[11] = vaddq_s32(v[11], v[10]);
    u[12] = vaddq_s32(v[12], v[13]);
    u[13] = vsubq_s32(v[12], v[13]);
    u[14] = vsubq_s32(v[15], v[14]);
    u[15] = vaddq_s32(v[15], v[14]);

    // stage 6
    v[0] = u[0];
    v[1] = u[1];
    v[2] = u[2];
    v[3] = u[3];
    v[4] = u[4];
    v[5] = u[5];
    v[6] = u[6];
    v[7] = u[7];

    v[8] = vmulq_n_s32(u[8], cospi[60]);
    v[8] = vmlaq_n_s32(v[8], u[15], cospi[4]);
    v[8] = vrshlq_s32(v[8], v_bit);

    v[15] = vmulq_n_s32(u[15], cospi[60]);
    v[15] = vmlsq_n_s32(v[15], u[8], cospi[4]);
    v[15] = vrshlq_s32(v[15], v_bit);

    v[9] = vmulq_n_s32(u[9], cospi[28]);
    v[9] = vmlaq_n_s32(v[9], u[14], cospi[36]);
    v[9] = vrshlq_s32(v[9], v_bit);

    v[14] = vmulq_n_s32(u[14], cospi[28]);
    v[14] = vmlsq_n_s32(v[14], u[9], cospi[36]);
    v[14] = vrshlq_s32(v[14], v_bit);

    v[10] = vmulq_n_s32(u[10], cospi[44]);
    v[10] = vmlaq_n_s32(v[10], u[13], cospi[20]);
    v[10] = vrshlq_s32(v[10], v_bit);

    v[13] = vmulq_n_s32(u[13], cospi[44]);
    v[13] = vmlsq_n_s32(v[13], u[10], cospi[20]);
    v[13] = vrshlq_s32(v[13], v_bit);

    v[11] = vmulq_n_s32(u[11], cospi[12]);
    v[11] = vmlaq_n_s32(v[11], u[12], cospi[52]);
    v[11] = vrshlq_s32(v[11], v_bit);

    v[12] = vmulq_n_s32(u[12], cospi[12]);
    v[12] = vmlsq_n_s32(v[12], u[11], cospi[52]);
    v[12] = vrshlq_s32(v[12], v_bit);

    out[0 + col * stride] = v[0];
    out[1 + col * stride] = v[8];
    out[2 + col * stride] = v[4];
    out[3 + col * stride] = v[12];
    out[4 + col * stride] = v[2];
    out[5 + col * stride] = v[10];
    out[6 + col * stride] = v[6];
    out[7 + col * stride] = v[14];
    out[8 + col * stride] = v[1];
    out[9 + col * stride] = v[9];
    out[10 + col * stride] = v[5];
    out[11 + col * stride] = v[13];
    out[12 + col * stride] = v[3];
    out[13 + col * stride] = v[11];
    out[14 + col * stride] = v[7];
    out[15 + col * stride] = v[15];
  }
}

static void fadst16x16_neon(int32x4_t *in, int32x4_t *out, int bit, int howmany,
                            int stride) {
  const int32_t *cospi = cospi_arr(bit);
  const int32x4_t v_bit = vdupq_n_s32(-bit);
  int32x4_t u[16], v[16], x, y;

  for (int col = 0; col < howmany; ++col) {
    // stage 0-1
    u[0] = in[0 + col * stride];
    u[1] = vnegq_s32(in[15 + col * stride]);
    u[2] = vnegq_s32(in[7 + col * stride]);
    u[3] = in[8 + col * stride];
    u[4] = vnegq_s32(in[3 + col * stride]);
    u[5] = in[12 + col * stride];
    u[6] = in[4 + col * stride];
    u[7] = vnegq_s32(in[11 + col * stride]);
    u[8] = vnegq_s32(in[1 + col * stride]);
    u[9] = in[14 + col * stride];
    u[10] = in[6 + col * stride];
    u[11] = vnegq_s32(in[9 + col * stride]);
    u[12] = in[2 + col * stride];
    u[13] = vnegq_s32(in[13 + col * stride]);
    u[14] = vnegq_s32(in[5 + col * stride]);
    u[15] = in[10 + col * stride];

    // stage 2
    v[0] = u[0];
    v[1] = u[1];

    x = vmulq_n_s32(u[2], cospi[32]);
    y = vmulq_n_s32(u[3], cospi[32]);
    v[2] = vaddq_s32(x, y);
    v[2] = vrshlq_s32(v[2], v_bit);

    v[3] = vsubq_s32(x, y);
    v[3] = vrshlq_s32(v[3], v_bit);

    v[4] = u[4];
    v[5] = u[5];

    x = vmulq_n_s32(u[6], cospi[32]);
    y = vmulq_n_s32(u[7], cospi[32]);
    v[6] = vaddq_s32(x, y);
    v[6] = vrshlq_s32(v[6], v_bit);

    v[7] = vsubq_s32(x, y);
    v[7] = vrshlq_s32(v[7], v_bit);

    v[8] = u[8];
    v[9] = u[9];

    x = vmulq_n_s32(u[10], cospi[32]);
    y = vmulq_n_s32(u[11], cospi[32]);
    v[10] = vaddq_s32(x, y);
    v[10] = vrshlq_s32(v[10], v_bit);

    v[11] = vsubq_s32(x, y);
    v[11] = vrshlq_s32(v[11], v_bit);

    v[12] = u[12];
    v[13] = u[13];

    x = vmulq_n_s32(u[14], cospi[32]);
    y = vmulq_n_s32(u[15], cospi[32]);
    v[14] = vaddq_s32(x, y);
    v[14] = vrshlq_s32(v[14], v_bit);

    v[15] = vsubq_s32(x, y);
    v[15] = vrshlq_s32(v[15], v_bit);

    // stage 3
    u[0] = vaddq_s32(v[0], v[2]);
    u[1] = vaddq_s32(v[1], v[3]);
    u[2] = vsubq_s32(v[0], v[2]);
    u[3] = vsubq_s32(v[1], v[3]);
    u[4] = vaddq_s32(v[4], v[6]);
    u[5] = vaddq_s32(v[5], v[7]);
    u[6] = vsubq_s32(v[4], v[6]);
    u[7] = vsubq_s32(v[5], v[7]);
    u[8] = vaddq_s32(v[8], v[10]);
    u[9] = vaddq_s32(v[9], v[11]);
    u[10] = vsubq_s32(v[8], v[10]);
    u[11] = vsubq_s32(v[9], v[11]);
    u[12] = vaddq_s32(v[12], v[14]);
    u[13] = vaddq_s32(v[13], v[15]);
    u[14] = vsubq_s32(v[12], v[14]);
    u[15] = vsubq_s32(v[13], v[15]);

    // stage 4
    v[0] = u[0];
    v[1] = u[1];
    v[2] = u[2];
    v[3] = u[3];
    v[4] = half_btf_neon(&cospi[16], &u[4], &cospi[48], &u[5], v_bit);
    v[7] = half_btf_neon(&cospi[16], &u[6], &cospi[48], &u[7], v_bit);
    v[5] = half_btf_neon_m(&cospi[48], &u[4], &cospi[16], &u[5], v_bit);
    v[6] = half_btf_neon_m(&cospi[16], &u[7], &cospi[48], &u[6], v_bit);

    v[8] = u[8];
    v[9] = u[9];
    v[10] = u[10];
    v[11] = u[11];

    v[12] = half_btf_neon(&cospi[16], &u[12], &cospi[48], &u[13], v_bit);
    v[15] = half_btf_neon(&cospi[16], &u[14], &cospi[48], &u[15], v_bit);
    v[13] = half_btf_neon_m(&cospi[48], &u[12], &cospi[16], &u[13], v_bit);
    v[14] = half_btf_neon_m(&cospi[16], &u[15], &cospi[48], &u[14], v_bit);

    // stage 5
    u[0] = vaddq_s32(v[0], v[4]);
    u[1] = vaddq_s32(v[1], v[5]);
    u[2] = vaddq_s32(v[2], v[6]);
    u[3] = vaddq_s32(v[3], v[7]);
    u[4] = vsubq_s32(v[0], v[4]);
    u[5] = vsubq_s32(v[1], v[5]);
    u[6] = vsubq_s32(v[2], v[6]);
    u[7] = vsubq_s32(v[3], v[7]);
    u[8] = vaddq_s32(v[8], v[12]);
    u[9] = vaddq_s32(v[9], v[13]);
    u[10] = vaddq_s32(v[10], v[14]);
    u[11] = vaddq_s32(v[11], v[15]);
    u[12] = vsubq_s32(v[8], v[12]);
    u[13] = vsubq_s32(v[9], v[13]);
    u[14] = vsubq_s32(v[10], v[14]);
    u[15] = vsubq_s32(v[11], v[15]);

    // stage 6
    v[0] = u[0];
    v[1] = u[1];
    v[2] = u[2];
    v[3] = u[3];
    v[4] = u[4];
    v[5] = u[5];
    v[6] = u[6];
    v[7] = u[7];

    v[8] = half_btf_neon(&cospi[8], &u[8], &cospi[56], &u[9], v_bit);
    v[13] = half_btf_neon(&cospi[8], &u[12], &cospi[56], &u[13], v_bit);
    v[9] = half_btf_neon_m(&cospi[56], &u[8], &cospi[8], &u[9], v_bit);
    v[12] = half_btf_neon_m(&cospi[8], &u[13], &cospi[56], &u[12], v_bit);

    v[10] = half_btf_neon(&cospi[40], &u[10], &cospi[24], &u[11], v_bit);
    v[15] = half_btf_neon(&cospi[40], &u[14], &cospi[24], &u[15], v_bit);
    v[11] = half_btf_neon_m(&cospi[24], &u[10], &cospi[40], &u[11], v_bit);
    v[14] = half_btf_neon_m(&cospi[40], &u[15], &cospi[24], &u[14], v_bit);

    // stage 7
    u[0] = vaddq_s32(v[0], v[8]);
    u[1] = vaddq_s32(v[1], v[9]);
    u[2] = vaddq_s32(v[2], v[10]);
    u[3] = vaddq_s32(v[3], v[11]);
    u[4] = vaddq_s32(v[4], v[12]);
    u[5] = vaddq_s32(v[5], v[13]);
    u[6] = vaddq_s32(v[6], v[14]);
    u[7] = vaddq_s32(v[7], v[15]);
    u[8] = vsubq_s32(v[0], v[8]);
    u[9] = vsubq_s32(v[1], v[9]);
    u[10] = vsubq_s32(v[2], v[10]);
    u[11] = vsubq_s32(v[3], v[11]);
    u[12] = vsubq_s32(v[4], v[12]);
    u[13] = vsubq_s32(v[5], v[13]);
    u[14] = vsubq_s32(v[6], v[14]);
    u[15] = vsubq_s32(v[7], v[15]);

    // stage 8
    v[0] = half_btf_neon(&cospi[2], &u[0], &cospi[62], &u[1], v_bit);
    v[1] = half_btf_neon_m(&cospi[62], &u[0], &cospi[2], &u[1], v_bit);
    v[2] = half_btf_neon(&cospi[10], &u[2], &cospi[54], &u[3], v_bit);
    v[3] = half_btf_neon_m(&cospi[54], &u[2], &cospi[10], &u[3], v_bit);
    v[4] = half_btf_neon(&cospi[18], &u[4], &cospi[46], &u[5], v_bit);
    v[5] = half_btf_neon_m(&cospi[46], &u[4], &cospi[18], &u[5], v_bit);
    v[6] = half_btf_neon(&cospi[26], &u[6], &cospi[38], &u[7], v_bit);
    v[7] = half_btf_neon_m(&cospi[38], &u[6], &cospi[26], &u[7], v_bit);
    v[8] = half_btf_neon(&cospi[34], &u[8], &cospi[30], &u[9], v_bit);
    v[9] = half_btf_neon_m(&cospi[30], &u[8], &cospi[34], &u[9], v_bit);
    v[10] = half_btf_neon(&cospi[42], &u[10], &cospi[22], &u[11], v_bit);
    v[11] = half_btf_neon_m(&cospi[22], &u[10], &cospi[42], &u[11], v_bit);
    v[12] = half_btf_neon(&cospi[50], &u[12], &cospi[14], &u[13], v_bit);
    v[13] = half_btf_neon_m(&cospi[14], &u[12], &cospi[50], &u[13], v_bit);
    v[14] = half_btf_neon(&cospi[58], &u[14], &cospi[6], &u[15], v_bit);
    v[15] = half_btf_neon_m(&cospi[6], &u[14], &cospi[58], &u[15], v_bit);

    // stage 9
    out[0 + col * stride] = v[1];
    out[1 + col * stride] = v[14];
    out[2 + col * stride] = v[3];
    out[3 + col * stride] = v[12];
    out[4 + col * stride] = v[5];
    out[5 + col * stride] = v[10];
    out[6 + col * stride] = v[7];
    out[7 + col * stride] = v[8];
    out[8 + col * stride] = v[9];
    out[9 + col * stride] = v[6];
    out[10 + col * stride] = v[11];
    out[11 + col * stride] = v[4];
    out[12 + col * stride] = v[13];
    out[13 + col * stride] = v[2];
    out[14 + col * stride] = v[15];
    out[15 + col * stride] = v[0];
  }
}

static void col_txfm_16x16_rounding(int32x4_t *in, const int shift) {
  // Note:
  //  We split 16x16 rounding into 4 sections of 8x8 rounding,
  //  instead of 4 columns
  col_txfm_8x8_rounding(&in[0], shift);
  col_txfm_8x8_rounding(&in[16], shift);
  col_txfm_8x8_rounding(&in[32], shift);
  col_txfm_8x8_rounding(&in[48], shift);
}

static void col_txfm_8x16_rounding(int32x4_t *in, const int shift) {
  col_txfm_8x8_rounding(&in[0], shift);
  col_txfm_8x8_rounding(&in[16], shift);
}

static void idtx16x16_neon(int32x4_t *in, int32x4_t *out, int bit, int howmany,
                           int stride) {
  (void)bit;
  (void)stride;
  int32x4_t fact = vdupq_n_s32(2 * NewSqrt2);
  int32x4_t offset = vdupq_n_s32(1 << (NewSqrt2Bits - 1));
  int32x4_t a_low;

  int num_iters = 16 * howmany;
  for (int i = 0; i < num_iters; i++) {
    a_low = vmulq_s32(in[i], fact);
    a_low = vaddq_s32(a_low, offset);
    out[i] = vshrq_n_s32(a_low, NewSqrt2Bits);
  }
}

void av1_fwd_txfm2d_16x16_neon(const int16_t *input, int32_t *coeff, int stride,
                               TX_TYPE tx_type, int bd) {
  (void)bd;
  int ud_flip, lr_flip;
  get_flip_cfg(tx_type, &ud_flip, &lr_flip);
  ud_adjust_input_and_stride(ud_flip, &input, &stride, 16);

  // Workspaces for column/row-wise transforms.
  int32x4_t buf0[64], buf1[64];

  switch (tx_type) {
    case DCT_DCT:
      load_buffer_16x16(input, buf0, stride, 0, 2);
      fdct16x16_neon(buf0, buf0, av1_fwd_cos_bit_col[2][2], 4, 16);
      col_txfm_16x16_rounding(buf0, -2);
      transpose_arrays_s32_16x16(buf0, buf1);
      fdct16x16_neon(buf1, buf1, av1_fwd_cos_bit_row[2][2], 4, 16);
      store_buffer_16x16(buf1, coeff);
      break;
    case ADST_DCT:
      load_buffer_16x16(input, buf0, stride, 0, 2);
      fadst16x16_neon(buf0, buf0, av1_fwd_cos_bit_col[2][2], 4, 16);
      col_txfm_16x16_rounding(buf0, -2);
      transpose_arrays_s32_16x16(buf0, buf1);
      fdct16x16_neon(buf1, buf1, av1_fwd_cos_bit_row[2][2], 4, 16);
      store_buffer_16x16(buf1, coeff);
      break;
    case DCT_ADST:
      load_buffer_16x16(input, buf0, stride, 0, 2);
      fdct16x16_neon(buf0, buf0, av1_fwd_cos_bit_col[2][2], 4, 16);
      col_txfm_16x16_rounding(buf0, -2);
      transpose_arrays_s32_16x16(buf0, buf1);
      fadst16x16_neon(buf1, buf1, av1_fwd_cos_bit_row[2][2], 4, 16);
      store_buffer_16x16(buf1, coeff);
      break;
    case ADST_ADST:
      load_buffer_16x16(input, buf0, stride, 0, 2);
      fadst16x16_neon(buf0, buf0, av1_fwd_cos_bit_col[2][2], 4, 16);
      col_txfm_16x16_rounding(buf0, -2);
      transpose_arrays_s32_16x16(buf0, buf1);
      fadst16x16_neon(buf1, buf1, av1_fwd_cos_bit_row[2][2], 4, 16);
      store_buffer_16x16(buf1, coeff);
      break;
    case FLIPADST_DCT:
      load_buffer_16x16(input, buf0, stride, 0, 2);
      fadst16x16_neon(buf0, buf0, av1_fwd_cos_bit_col[2][2], 4, 16);
      col_txfm_16x16_rounding(buf0, -2);
      transpose_arrays_s32_16x16(buf0, buf1);
      fdct16x16_neon(buf1, buf1, av1_fwd_cos_bit_row[2][2], 4, 16);
      store_buffer_16x16(buf1, coeff);
      break;
    case DCT_FLIPADST:
      load_buffer_16x16(input, buf0, stride, 1, 2);
      fdct16x16_neon(buf0, buf0, av1_fwd_cos_bit_col[2][2], 4, 16);
      col_txfm_16x16_rounding(buf0, -2);
      transpose_arrays_s32_16x16(buf0, buf1);
      fadst16x16_neon(buf1, buf1, av1_fwd_cos_bit_row[2][2], 4, 16);
      store_buffer_16x16(buf1, coeff);
      break;
    case FLIPADST_FLIPADST:
      load_buffer_16x16(input, buf0, stride, 1, 2);
      fadst16x16_neon(buf0, buf0, av1_fwd_cos_bit_col[2][2], 4, 16);
      col_txfm_16x16_rounding(buf0, -2);
      transpose_arrays_s32_16x16(buf0, buf1);
      fadst16x16_neon(buf1, buf1, av1_fwd_cos_bit_row[2][2], 4, 16);
      store_buffer_16x16(buf1, coeff);
      break;
    case ADST_FLIPADST:
      load_buffer_16x16(input, buf0, stride, 1, 2);
      fadst16x16_neon(buf0, buf0, av1_fwd_cos_bit_col[2][2], 4, 16);
      col_txfm_16x16_rounding(buf0, -2);
      transpose_arrays_s32_16x16(buf0, buf1);
      fadst16x16_neon(buf1, buf1, av1_fwd_cos_bit_row[2][2], 4, 16);
      store_buffer_16x16(buf1, coeff);
      break;
    case FLIPADST_ADST:
      load_buffer_16x16(input, buf0, stride, 0, 2);
      fadst16x16_neon(buf0, buf0, av1_fwd_cos_bit_col[2][2], 4, 16);
      col_txfm_16x16_rounding(buf0, -2);
      transpose_arrays_s32_16x16(buf0, buf1);
      fadst16x16_neon(buf1, buf1, av1_fwd_cos_bit_row[2][2], 4, 16);
      store_buffer_16x16(buf1, coeff);
      break;
    case IDTX:
      load_buffer_16x16(input, buf0, stride, 0, 2);
      idtx16x16_neon(buf0, buf0, av1_fwd_cos_bit_col[2][2], 4, 16);
      col_txfm_16x16_rounding(buf0, -2);
      transpose_arrays_s32_16x16(buf0, buf1);
      idtx16x16_neon(buf1, buf1, av1_fwd_cos_bit_row[2][2], 4, 16);
      store_buffer_16x16(buf1, coeff);
      break;
    case V_DCT:
      load_buffer_16x16(input, buf0, stride, 0, 2);
      fdct16x16_neon(buf0, buf0, av1_fwd_cos_bit_col[2][2], 4, 16);
      col_txfm_16x16_rounding(buf0, -2);
      transpose_arrays_s32_16x16(buf0, buf1);
      idtx16x16_neon(buf1, buf1, av1_fwd_cos_bit_row[2][2], 4, 16);
      store_buffer_16x16(buf1, coeff);
      break;
    case H_DCT:
      load_buffer_16x16(input, buf0, stride, 0, 2);
      idtx16x16_neon(buf0, buf0, av1_fwd_cos_bit_col[2][2], 4, 16);
      col_txfm_16x16_rounding(buf0, -2);
      transpose_arrays_s32_16x16(buf0, buf1);
      fdct16x16_neon(buf1, buf1, av1_fwd_cos_bit_row[2][2], 4, 16);
      store_buffer_16x16(buf1, coeff);
      break;
    case V_ADST:
      load_buffer_16x16(input, buf0, stride, 0, 2);
      fadst16x16_neon(buf0, buf0, av1_fwd_cos_bit_col[2][2], 4, 16);
      col_txfm_16x16_rounding(buf0, -2);
      transpose_arrays_s32_16x16(buf0, buf1);
      idtx16x16_neon(buf1, buf1, av1_fwd_cos_bit_row[2][2], 4, 16);
      store_buffer_16x16(buf1, coeff);
      break;
    case H_ADST:
      load_buffer_16x16(input, buf0, stride, 0, 2);
      idtx16x16_neon(buf0, buf0, av1_fwd_cos_bit_col[2][2], 4, 16);
      col_txfm_16x16_rounding(buf0, -2);
      transpose_arrays_s32_16x16(buf0, buf1);
      fadst16x16_neon(buf1, buf1, av1_fwd_cos_bit_row[2][2], 4, 16);
      store_buffer_16x16(buf1, coeff);
      break;
    case V_FLIPADST:
      load_buffer_16x16(input, buf0, stride, 0, 2);
      fadst16x16_neon(buf0, buf0, av1_fwd_cos_bit_col[2][2], 4, 16);
      col_txfm_16x16_rounding(buf0, -2);
      transpose_arrays_s32_16x16(buf0, buf1);
      idtx16x16_neon(buf1, buf1, av1_fwd_cos_bit_row[2][2], 4, 16);
      store_buffer_16x16(buf1, coeff);
      break;
    case H_FLIPADST:
      load_buffer_16x16(input, buf0, stride, 1, 2);
      idtx16x16_neon(buf0, buf0, av1_fwd_cos_bit_col[2][2], 4, 16);
      col_txfm_16x16_rounding(buf0, -2);
      transpose_arrays_s32_16x16(buf0, buf1);
      fadst16x16_neon(buf1, buf1, av1_fwd_cos_bit_row[2][2], 4, 16);
      store_buffer_16x16(buf1, coeff);
      break;
    default: assert(0);
  }
}

typedef void (*fwd_transform_1d_neon)(int32x4_t *in, int32x4_t *out, int bit);
typedef void (*fwd_transform_1d_many_neon)(int32x4_t *in, int32x4_t *out,
                                           int bit, int howmany, int stride);

static const fwd_transform_1d_many_neon col_highbd_txfm8x8_arr[TX_TYPES] = {
  fdct8x8_neon,   // DCT_DCT
  fadst8x8_neon,  // ADST_DCT
  fdct8x8_neon,   // DCT_ADST
  fadst8x8_neon,  // ADST_ADST
  fadst8x8_neon,  // FLIPADST_DCT
  fdct8x8_neon,   // DCT_FLIPADST
  fadst8x8_neon,  // FLIPADST_FLIPADST
  fadst8x8_neon,  // ADST_FLIPADST
  fadst8x8_neon,  // FLIPADST_ADST
  idtx8x8_neon,   // IDTX
  fdct8x8_neon,   // V_DCT
  idtx8x8_neon,   // H_DCT
  fadst8x8_neon,  // V_ADST
  idtx8x8_neon,   // H_ADST
  fadst8x8_neon,  // V_FLIPADST
  idtx8x8_neon    // H_FLIPADST
};

#if !CONFIG_REALTIME_ONLY
static const fwd_transform_1d_many_neon row_highbd_txfm32x8_arr[TX_TYPES] = {
  fdct8x8_neon,   // DCT_DCT
  NULL,           // ADST_DCT
  NULL,           // DCT_ADST
  NULL,           // ADST_ADST
  NULL,           // FLIPADST_DCT
  NULL,           // DCT_FLIPADST
  NULL,           // FLIPADST_FLIPADST
  NULL,           // ADST_FLIPADST
  NULL,           // FLIPADST-ADST
  idtx32x8_neon,  // IDTX
  NULL,           // V_DCT
  NULL,           // H_DCT
  NULL,           // V_ADST
  NULL,           // H_ADST
  NULL,           // V_FLIPADST
  NULL,           // H_FLIPADST
};
#endif

static const fwd_transform_1d_many_neon col_highbd_txfm4x8_arr[TX_TYPES] = {
  fdct4x8_neon,   // DCT_DCT
  fadst8x8_neon,  // ADST_DCT
  fdct4x8_neon,   // DCT_ADST
  fadst8x8_neon,  // ADST_ADST
  fadst8x8_neon,  // FLIPADST_DCT
  fdct4x8_neon,   // DCT_FLIPADST
  fadst8x8_neon,  // FLIPADST_FLIPADST
  fadst8x8_neon,  // ADST_FLIPADST
  fadst8x8_neon,  // FLIPADST_ADST
  idtx8x8_neon,   // IDTX
  fdct4x8_neon,   // V_DCT
  idtx8x8_neon,   // H_DCT
  fadst8x8_neon,  // V_ADST
  idtx8x8_neon,   // H_ADST
  fadst8x8_neon,  // V_FLIPADST
  idtx8x8_neon    // H_FLIPADST
};

static const fwd_transform_1d_many_neon row_highbd_txfm8x16_arr[TX_TYPES] = {
  fdct16x16_neon,   // DCT_DCT
  fdct16x16_neon,   // ADST_DCT
  fadst16x16_neon,  // DCT_ADST
  fadst16x16_neon,  // ADST_ADST
  fdct16x16_neon,   // FLIPADST_DCT
  fadst16x16_neon,  // DCT_FLIPADST
  fadst16x16_neon,  // FLIPADST_FLIPADST
  fadst16x16_neon,  // ADST_FLIPADST
  fadst16x16_neon,  // FLIPADST_ADST
  idtx16x16_neon,   // IDTX
  idtx16x16_neon,   // V_DCT
  fdct16x16_neon,   // H_DCT
  idtx16x16_neon,   // V_ADST
  fadst16x16_neon,  // H_ADST
  idtx16x16_neon,   // V_FLIPADST
  fadst16x16_neon   // H_FLIPADST
};

static const fwd_transform_1d_many_neon col_highbd_txfm8x16_arr[TX_TYPES] = {
  fdct16x16_neon,   // DCT_DCT
  fadst16x16_neon,  // ADST_DCT
  fdct16x16_neon,   // DCT_ADST
  fadst16x16_neon,  // ADST_ADST
  fadst16x16_neon,  // FLIPADST_DCT
  fdct16x16_neon,   // DCT_FLIPADST
  fadst16x16_neon,  // FLIPADST_FLIPADST
  fadst16x16_neon,  // ADST_FLIPADST
  fadst16x16_neon,  // FLIPADST_ADST
  idtx16x16_neon,   // IDTX
  fdct16x16_neon,   // V_DCT
  idtx16x16_neon,   // H_DCT
  fadst16x16_neon,  // V_ADST
  idtx16x16_neon,   // H_ADST
  fadst16x16_neon,  // V_FLIPADST
  idtx16x16_neon    // H_FLIPADST
};

static const fwd_transform_1d_many_neon row_highbd_txfm8x8_arr[TX_TYPES] = {
  fdct8x8_neon,   // DCT_DCT
  fdct8x8_neon,   // ADST_DCT
  fadst8x8_neon,  // DCT_ADST
  fadst8x8_neon,  // ADST_ADST
  fdct8x8_neon,   // FLIPADST_DCT
  fadst8x8_neon,  // DCT_FLIPADST
  fadst8x8_neon,  // FLIPADST_FLIPADST
  fadst8x8_neon,  // ADST_FLIPADST
  fadst8x8_neon,  // FLIPADST_ADST
  idtx8x8_neon,   // IDTX
  idtx8x8_neon,   // V_DCT
  fdct8x8_neon,   // H_DCT
  idtx8x8_neon,   // V_ADST
  fadst8x8_neon,  // H_ADST
  idtx8x8_neon,   // V_FLIPADST
  fadst8x8_neon   // H_FLIPADST
};

static const fwd_transform_1d_many_neon row_highbd_txfm4x8_arr[TX_TYPES] = {
  fdct4x8_neon,   // DCT_DCT
  fdct4x8_neon,   // ADST_DCT
  fadst8x8_neon,  // DCT_ADST
  fadst8x8_neon,  // ADST_ADST
  fdct4x8_neon,   // FLIPADST_DCT
  fadst8x8_neon,  // DCT_FLIPADST
  fadst8x8_neon,  // FLIPADST_FLIPADST
  fadst8x8_neon,  // ADST_FLIPADST
  fadst8x8_neon,  // FLIPADST_ADST
  idtx8x8_neon,   // IDTX
  idtx8x8_neon,   // V_DCT
  fdct4x8_neon,   // H_DCT
  idtx8x8_neon,   // V_ADST
  fadst8x8_neon,  // H_ADST
  idtx8x8_neon,   // V_FLIPADST
  fadst8x8_neon   // H_FLIPADST
};

static const fwd_transform_1d_neon row_highbd_txfm4x4_arr[TX_TYPES] = {
  fdct4x4_neon,   // DCT_DCT
  fdct4x4_neon,   // ADST_DCT
  fadst4x4_neon,  // DCT_ADST
  fadst4x4_neon,  // ADST_ADST
  fdct4x4_neon,   // FLIPADST_DCT
  fadst4x4_neon,  // DCT_FLIPADST
  fadst4x4_neon,  // FLIPADST_FLIPADST
  fadst4x4_neon,  // ADST_FLIPADST
  fadst4x4_neon,  // FLIPADST_ADST
  idtx4x4_neon,   // IDTX
  idtx4x4_neon,   // V_DCT
  fdct4x4_neon,   // H_DCT
  idtx4x4_neon,   // V_ADST
  fadst4x4_neon,  // H_ADST
  idtx4x4_neon,   // V_FLIPADST
  fadst4x4_neon   // H_FLIPADST
};

static const fwd_transform_1d_neon col_highbd_txfm4x4_arr[TX_TYPES] = {
  fdct4x4_neon,   // DCT_DCT
  fadst4x4_neon,  // ADST_DCT
  fdct4x4_neon,   // DCT_ADST
  fadst4x4_neon,  // ADST_ADST
  fadst4x4_neon,  // FLIPADST_DCT
  fdct4x4_neon,   // DCT_FLIPADST
  fadst4x4_neon,  // FLIPADST_FLIPADST
  fadst4x4_neon,  // ADST_FLIPADST
  fadst4x4_neon,  // FLIPADST_ADST
  idtx4x4_neon,   // IDTX
  fdct4x4_neon,   // V_DCT
  idtx4x4_neon,   // H_DCT
  fadst4x4_neon,  // V_ADST
  idtx4x4_neon,   // H_ADST
  fadst4x4_neon,  // V_FLIPADST
  idtx4x4_neon    // H_FLIPADST
};

void av1_fdct32_new_neon(int32x4_t *input, int32x4_t *output, int cos_bit) {
  const int32_t *cospi;
  const int32x4_t v_cos_bit = vdupq_n_s32(-cos_bit);

  // Workspaces for intermediate transform steps.
  int32x4_t buf0[32];
  int32x4_t buf1[32];

  // stage 0
  // stage 1
  buf1[0] = vaddq_s32(input[0], input[31]);
  buf1[31] = vsubq_s32(input[0], input[31]);
  buf1[1] = vaddq_s32(input[1], input[30]);
  buf1[30] = vsubq_s32(input[1], input[30]);
  buf1[2] = vaddq_s32(input[2], input[29]);
  buf1[29] = vsubq_s32(input[2], input[29]);
  buf1[3] = vaddq_s32(input[3], input[28]);
  buf1[28] = vsubq_s32(input[3], input[28]);
  buf1[4] = vaddq_s32(input[4], input[27]);
  buf1[27] = vsubq_s32(input[4], input[27]);
  buf1[5] = vaddq_s32(input[5], input[26]);
  buf1[26] = vsubq_s32(input[5], input[26]);
  buf1[6] = vaddq_s32(input[6], input[25]);
  buf1[25] = vsubq_s32(input[6], input[25]);
  buf1[7] = vaddq_s32(input[7], input[24]);
  buf1[24] = vsubq_s32(input[7], input[24]);
  buf1[8] = vaddq_s32(input[8], input[23]);
  buf1[23] = vsubq_s32(input[8], input[23]);
  buf1[9] = vaddq_s32(input[9], input[22]);
  buf1[22] = vsubq_s32(input[9], input[22]);
  buf1[10] = vaddq_s32(input[10], input[21]);
  buf1[21] = vsubq_s32(input[10], input[21]);
  buf1[11] = vaddq_s32(input[11], input[20]);
  buf1[20] = vsubq_s32(input[11], input[20]);
  buf1[12] = vaddq_s32(input[12], input[19]);
  buf1[19] = vsubq_s32(input[12], input[19]);
  buf1[13] = vaddq_s32(input[13], input[18]);
  buf1[18] = vsubq_s32(input[13], input[18]);
  buf1[14] = vaddq_s32(input[14], input[17]);
  buf1[17] = vsubq_s32(input[14], input[17]);
  buf1[15] = vaddq_s32(input[15], input[16]);
  buf1[16] = vsubq_s32(input[15], input[16]);

  // stage 2
  cospi = cospi_arr(cos_bit);
  buf0[0] = vaddq_s32(buf1[0], buf1[15]);
  buf0[15] = vsubq_s32(buf1[0], buf1[15]);
  buf0[1] = vaddq_s32(buf1[1], buf1[14]);
  buf0[14] = vsubq_s32(buf1[1], buf1[14]);
  buf0[2] = vaddq_s32(buf1[2], buf1[13]);
  buf0[13] = vsubq_s32(buf1[2], buf1[13]);
  buf0[3] = vaddq_s32(buf1[3], buf1[12]);
  buf0[12] = vsubq_s32(buf1[3], buf1[12]);
  buf0[4] = vaddq_s32(buf1[4], buf1[11]);
  buf0[11] = vsubq_s32(buf1[4], buf1[11]);
  buf0[5] = vaddq_s32(buf1[5], buf1[10]);
  buf0[10] = vsubq_s32(buf1[5], buf1[10]);
  buf0[6] = vaddq_s32(buf1[6], buf1[9]);
  buf0[9] = vsubq_s32(buf1[6], buf1[9]);
  buf0[7] = vaddq_s32(buf1[7], buf1[8]);
  buf0[8] = vsubq_s32(buf1[7], buf1[8]);
  buf0[16] = buf1[16];
  buf0[17] = buf1[17];
  buf0[18] = buf1[18];
  buf0[19] = buf1[19];
  btf_32_neon_type0(-cospi[32], cospi[32], buf1[20], buf1[27], buf0[20],
                    buf0[27], v_cos_bit);
  btf_32_neon_type0(-cospi[32], cospi[32], buf1[21], buf1[26], buf0[21],
                    buf0[26], v_cos_bit);
  btf_32_neon_type0(-cospi[32], cospi[32], buf1[22], buf1[25], buf0[22],
                    buf0[25], v_cos_bit);
  btf_32_neon_type0(-cospi[32], cospi[32], buf1[23], buf1[24], buf0[23],
                    buf0[24], v_cos_bit);
  buf0[28] = buf1[28];
  buf0[29] = buf1[29];
  buf0[30] = buf1[30];
  buf0[31] = buf1[31];

  // stage 3
  cospi = cospi_arr(cos_bit);
  buf1[0] = vaddq_s32(buf0[0], buf0[7]);
  buf1[7] = vsubq_s32(buf0[0], buf0[7]);
  buf1[1] = vaddq_s32(buf0[1], buf0[6]);
  buf1[6] = vsubq_s32(buf0[1], buf0[6]);
  buf1[2] = vaddq_s32(buf0[2], buf0[5]);
  buf1[5] = vsubq_s32(buf0[2], buf0[5]);
  buf1[3] = vaddq_s32(buf0[3], buf0[4]);
  buf1[4] = vsubq_s32(buf0[3], buf0[4]);
  buf1[8] = buf0[8];
  buf1[9] = buf0[9];
  btf_32_neon_type0(-cospi[32], cospi[32], buf0[10], buf0[13], buf1[10],
                    buf1[13], v_cos_bit);
  btf_32_neon_type0(-cospi[32], cospi[32], buf0[11], buf0[12], buf1[11],
                    buf1[12], v_cos_bit);
  buf1[14] = buf0[14];
  buf1[15] = buf0[15];
  buf1[16] = vaddq_s32(buf0[16], buf0[23]);
  buf1[23] = vsubq_s32(buf0[16], buf0[23]);
  buf1[17] = vaddq_s32(buf0[17], buf0[22]);
  buf1[22] = vsubq_s32(buf0[17], buf0[22]);
  buf1[18] = vaddq_s32(buf0[18], buf0[21]);
  buf1[21] = vsubq_s32(buf0[18], buf0[21]);
  buf1[19] = vaddq_s32(buf0[19], buf0[20]);
  buf1[20] = vsubq_s32(buf0[19], buf0[20]);
  buf1[24] = vsubq_s32(buf0[31], buf0[24]);
  buf1[31] = vaddq_s32(buf0[31], buf0[24]);
  buf1[25] = vsubq_s32(buf0[30], buf0[25]);
  buf1[30] = vaddq_s32(buf0[30], buf0[25]);
  buf1[26] = vsubq_s32(buf0[29], buf0[26]);
  buf1[29] = vaddq_s32(buf0[29], buf0[26]);
  buf1[27] = vsubq_s32(buf0[28], buf0[27]);
  buf1[28] = vaddq_s32(buf0[28], buf0[27]);

  // stage 4
  cospi = cospi_arr(cos_bit);
  buf0[0] = vaddq_s32(buf1[0], buf1[3]);
  buf0[3] = vsubq_s32(buf1[0], buf1[3]);
  buf0[1] = vaddq_s32(buf1[1], buf1[2]);
  buf0[2] = vsubq_s32(buf1[1], buf1[2]);
  buf0[4] = buf1[4];
  btf_32_neon_type0(-cospi[32], cospi[32], buf1[5], buf1[6], buf0[5], buf0[6],
                    v_cos_bit);
  buf0[7] = buf1[7];
  buf0[8] = vaddq_s32(buf1[8], buf1[11]);
  buf0[11] = vsubq_s32(buf1[8], buf1[11]);
  buf0[9] = vaddq_s32(buf1[9], buf1[10]);
  buf0[10] = vsubq_s32(buf1[9], buf1[10]);
  buf0[12] = vsubq_s32(buf1[15], buf1[12]);
  buf0[15] = vaddq_s32(buf1[15], buf1[12]);
  buf0[13] = vsubq_s32(buf1[14], buf1[13]);
  buf0[14] = vaddq_s32(buf1[14], buf1[13]);
  buf0[16] = buf1[16];
  buf0[17] = buf1[17];

  btf_32_neon_type0(-cospi[16], cospi[48], buf1[18], buf1[29], buf0[18],
                    buf0[29], v_cos_bit);
  btf_32_neon_type0(-cospi[16], cospi[48], buf1[19], buf1[28], buf0[19],
                    buf0[28], v_cos_bit);

  btf_32_neon_type0(-cospi[48], -cospi[16], buf1[20], buf1[27], buf0[20],
                    buf0[27], v_cos_bit);
  btf_32_neon_type0(-cospi[48], -cospi[16], buf1[21], buf1[26], buf0[21],
                    buf0[26], v_cos_bit);

  buf0[22] = buf1[22];
  buf0[23] = buf1[23];
  buf0[24] = buf1[24];
  buf0[25] = buf1[25];
  buf0[30] = buf1[30];
  buf0[31] = buf1[31];

  // stage 5
  btf_32_neon_type0(cospi[32], cospi[32], buf0[0], buf0[1], buf1[0], buf1[1],
                    v_cos_bit);

  btf_32_neon_type1(cospi[48], cospi[16], buf0[2], buf0[3], buf1[2], buf1[3],
                    v_cos_bit);
  buf1[4] = vaddq_s32(buf0[4], buf0[5]);
  buf1[5] = vsubq_s32(buf0[4], buf0[5]);
  buf1[6] = vsubq_s32(buf0[7], buf0[6]);
  buf1[7] = vaddq_s32(buf0[7], buf0[6]);
  buf1[8] = buf0[8];
  btf_32_neon_type0(-cospi[16], cospi[48], buf0[9], buf0[14], buf1[9], buf1[14],
                    v_cos_bit);
  btf_32_neon_type0(-cospi[48], -cospi[16], buf0[10], buf0[13], buf1[10],
                    buf1[13], v_cos_bit);
  buf1[11] = buf0[11];
  buf1[12] = buf0[12];
  buf1[15] = buf0[15];
  buf1[16] = vaddq_s32(buf0[16], buf0[19]);
  buf1[19] = vsubq_s32(buf0[16], buf0[19]);
  buf1[17] = vaddq_s32(buf0[17], buf0[18]);
  buf1[18] = vsubq_s32(buf0[17], buf0[18]);
  buf1[20] = vsubq_s32(buf0[23], buf0[20]);
  buf1[23] = vaddq_s32(buf0[23], buf0[20]);
  buf1[21] = vsubq_s32(buf0[22], buf0[21]);
  buf1[22] = vaddq_s32(buf0[22], buf0[21]);
  buf1[24] = vaddq_s32(buf0[24], buf0[27]);
  buf1[27] = vsubq_s32(buf0[24], buf0[27]);
  buf1[25] = vaddq_s32(buf0[25], buf0[26]);
  buf1[26] = vsubq_s32(buf0[25], buf0[26]);
  buf1[28] = vsubq_s32(buf0[31], buf0[28]);
  buf1[31] = vaddq_s32(buf0[31], buf0[28]);
  buf1[29] = vsubq_s32(buf0[30], buf0[29]);
  buf1[30] = vaddq_s32(buf0[30], buf0[29]);

  // stage 6
  cospi = cospi_arr(cos_bit);
  buf0[0] = buf1[0];
  buf0[1] = buf1[1];
  buf0[2] = buf1[2];
  buf0[3] = buf1[3];

  btf_32_neon_type1(cospi[56], cospi[8], buf1[4], buf1[7], buf0[4], buf0[7],
                    v_cos_bit);
  btf_32_neon_type0(-cospi[8], cospi[56], buf1[17], buf1[30], buf0[17],
                    buf0[30], v_cos_bit);
  btf_32_neon_type0(-cospi[56], -cospi[8], buf1[18], buf1[29], buf0[18],
                    buf0[29], v_cos_bit);

  buf0[8] = vaddq_s32(buf1[8], buf1[9]);
  buf0[9] = vsubq_s32(buf1[8], buf1[9]);
  buf0[10] = vsubq_s32(buf1[11], buf1[10]);
  buf0[11] = vaddq_s32(buf1[11], buf1[10]);
  buf0[12] = vaddq_s32(buf1[12], buf1[13]);
  buf0[13] = vsubq_s32(buf1[12], buf1[13]);
  buf0[14] = vsubq_s32(buf1[15], buf1[14]);
  buf0[15] = vaddq_s32(buf1[15], buf1[14]);
  buf0[16] = buf1[16];
  buf0[19] = buf1[19];
  buf0[20] = buf1[20];

  btf_32_neon_type1(cospi[24], cospi[40], buf1[5], buf1[6], buf0[5], buf0[6],
                    v_cos_bit);
  btf_32_neon_type0(-cospi[40], cospi[24], buf1[21], buf1[26], buf0[21],
                    buf0[26], v_cos_bit);
  btf_32_neon_type0(-cospi[24], -cospi[40], buf1[22], buf1[25], buf0[22],
                    buf0[25], v_cos_bit);

  buf0[23] = buf1[23];
  buf0[24] = buf1[24];
  buf0[27] = buf1[27];
  buf0[28] = buf1[28];
  buf0[31] = buf1[31];

  // stage 7
  cospi = cospi_arr(cos_bit);
  buf1[0] = buf0[0];
  buf1[1] = buf0[1];
  buf1[2] = buf0[2];
  buf1[3] = buf0[3];
  buf1[4] = buf0[4];
  buf1[5] = buf0[5];
  buf1[6] = buf0[6];
  buf1[7] = buf0[7];
  btf_32_neon_type1(cospi[60], cospi[4], buf0[8], buf0[15], buf1[8], buf1[15],
                    v_cos_bit);
  btf_32_neon_type1(cospi[28], cospi[36], buf0[9], buf0[14], buf1[9], buf1[14],
                    v_cos_bit);
  btf_32_neon_type1(cospi[44], cospi[20], buf0[10], buf0[13], buf1[10],
                    buf1[13], v_cos_bit);
  btf_32_neon_type1(cospi[12], cospi[52], buf0[11], buf0[12], buf1[11],
                    buf1[12], v_cos_bit);
  buf1[16] = vaddq_s32(buf0[16], buf0[17]);
  buf1[17] = vsubq_s32(buf0[16], buf0[17]);
  buf1[18] = vsubq_s32(buf0[19], buf0[18]);
  buf1[19] = vaddq_s32(buf0[19], buf0[18]);
  buf1[20] = vaddq_s32(buf0[20], buf0[21]);
  buf1[21] = vsubq_s32(buf0[20], buf0[21]);
  buf1[22] = vsubq_s32(buf0[23], buf0[22]);
  buf1[23] = vaddq_s32(buf0[23], buf0[22]);
  buf1[24] = vaddq_s32(buf0[24], buf0[25]);
  buf1[25] = vsubq_s32(buf0[24], buf0[25]);
  buf1[26] = vsubq_s32(buf0[27], buf0[26]);
  buf1[27] = vaddq_s32(buf0[27], buf0[26]);
  buf1[28] = vaddq_s32(buf0[28], buf0[29]);
  buf1[29] = vsubq_s32(buf0[28], buf0[29]);
  buf1[30] = vsubq_s32(buf0[31], buf0[30]);
  buf1[31] = vaddq_s32(buf0[31], buf0[30]);

  // stage 8
  cospi = cospi_arr(cos_bit);
  buf0[0] = buf1[0];
  buf0[1] = buf1[1];
  buf0[2] = buf1[2];
  buf0[3] = buf1[3];
  buf0[4] = buf1[4];
  buf0[5] = buf1[5];
  buf0[6] = buf1[6];
  buf0[7] = buf1[7];
  buf0[8] = buf1[8];
  buf0[9] = buf1[9];
  buf0[10] = buf1[10];
  buf0[11] = buf1[11];
  buf0[12] = buf1[12];
  buf0[13] = buf1[13];
  buf0[14] = buf1[14];
  buf0[15] = buf1[15];
  btf_32_neon_type1(cospi[62], cospi[2], buf1[16], buf1[31], buf0[16], buf0[31],
                    v_cos_bit);
  btf_32_neon_type1(cospi[30], cospi[34], buf1[17], buf1[30], buf0[17],
                    buf0[30], v_cos_bit);
  btf_32_neon_type1(cospi[46], cospi[18], buf1[18], buf1[29], buf0[18],
                    buf0[29], v_cos_bit);
  btf_32_neon_type1(cospi[14], cospi[50], buf1[19], buf1[28], buf0[19],
                    buf0[28], v_cos_bit);
  btf_32_neon_type1(cospi[54], cospi[10], buf1[20], buf1[27], buf0[20],
                    buf0[27], v_cos_bit);
  btf_32_neon_type1(cospi[22], cospi[42], buf1[21], buf1[26], buf0[21],
                    buf0[26], v_cos_bit);
  btf_32_neon_type1(cospi[38], cospi[26], buf1[22], buf1[25], buf0[22],
                    buf0[25], v_cos_bit);
  btf_32_neon_type1(cospi[6], cospi[58], buf1[23], buf1[24], buf0[23], buf0[24],
                    v_cos_bit);

  // stage 9
  output[0] = buf0[0];
  output[1] = buf0[16];
  output[2] = buf0[8];
  output[3] = buf0[24];
  output[4] = buf0[4];
  output[5] = buf0[20];
  output[6] = buf0[12];
  output[7] = buf0[28];
  output[8] = buf0[2];
  output[9] = buf0[18];
  output[10] = buf0[10];
  output[11] = buf0[26];
  output[12] = buf0[6];
  output[13] = buf0[22];
  output[14] = buf0[14];
  output[15] = buf0[30];
  output[16] = buf0[1];
  output[17] = buf0[17];
  output[18] = buf0[9];
  output[19] = buf0[25];
  output[20] = buf0[5];
  output[21] = buf0[21];
  output[22] = buf0[13];
  output[23] = buf0[29];
  output[24] = buf0[3];
  output[25] = buf0[19];
  output[26] = buf0[11];
  output[27] = buf0[27];
  output[28] = buf0[7];
  output[29] = buf0[23];
  output[30] = buf0[15];
  output[31] = buf0[31];
}

static void av1_fdct64_new_stage12345_neon(int32x4_t *input, int32x4_t *x5,
                                           const int32_t *cospi,
                                           const int32x4_t *v_cos_bit) {
  int32x4_t x1[64];
  x1[0] = vaddq_s32(input[0], input[63]);
  x1[63] = vsubq_s32(input[0], input[63]);
  x1[1] = vaddq_s32(input[1], input[62]);
  x1[62] = vsubq_s32(input[1], input[62]);
  x1[2] = vaddq_s32(input[2], input[61]);
  x1[61] = vsubq_s32(input[2], input[61]);
  x1[3] = vaddq_s32(input[3], input[60]);
  x1[60] = vsubq_s32(input[3], input[60]);
  x1[4] = vaddq_s32(input[4], input[59]);
  x1[59] = vsubq_s32(input[4], input[59]);
  x1[5] = vaddq_s32(input[5], input[58]);
  x1[58] = vsubq_s32(input[5], input[58]);
  x1[6] = vaddq_s32(input[6], input[57]);
  x1[57] = vsubq_s32(input[6], input[57]);
  x1[7] = vaddq_s32(input[7], input[56]);
  x1[56] = vsubq_s32(input[7], input[56]);
  x1[8] = vaddq_s32(input[8], input[55]);
  x1[55] = vsubq_s32(input[8], input[55]);
  x1[9] = vaddq_s32(input[9], input[54]);
  x1[54] = vsubq_s32(input[9], input[54]);
  x1[10] = vaddq_s32(input[10], input[53]);
  x1[53] = vsubq_s32(input[10], input[53]);
  x1[11] = vaddq_s32(input[11], input[52]);
  x1[52] = vsubq_s32(input[11], input[52]);
  x1[12] = vaddq_s32(input[12], input[51]);
  x1[51] = vsubq_s32(input[12], input[51]);
  x1[13] = vaddq_s32(input[13], input[50]);
  x1[50] = vsubq_s32(input[13], input[50]);
  x1[14] = vaddq_s32(input[14], input[49]);
  x1[49] = vsubq_s32(input[14], input[49]);
  x1[15] = vaddq_s32(input[15], input[48]);
  x1[48] = vsubq_s32(input[15], input[48]);
  x1[16] = vaddq_s32(input[16], input[47]);
  x1[47] = vsubq_s32(input[16], input[47]);
  x1[17] = vaddq_s32(input[17], input[46]);
  x1[46] = vsubq_s32(input[17], input[46]);
  x1[18] = vaddq_s32(input[18], input[45]);
  x1[45] = vsubq_s32(input[18], input[45]);
  x1[19] = vaddq_s32(input[19], input[44]);
  x1[44] = vsubq_s32(input[19], input[44]);
  x1[20] = vaddq_s32(input[20], input[43]);
  x1[43] = vsubq_s32(input[20], input[43]);
  x1[21] = vaddq_s32(input[21], input[42]);
  x1[42] = vsubq_s32(input[21], input[42]);
  x1[22] = vaddq_s32(input[22], input[41]);
  x1[41] = vsubq_s32(input[22], input[41]);
  x1[23] = vaddq_s32(input[23], input[40]);
  x1[40] = vsubq_s32(input[23], input[40]);
  x1[24] = vaddq_s32(input[24], input[39]);
  x1[39] = vsubq_s32(input[24], input[39]);
  x1[25] = vaddq_s32(input[25], input[38]);
  x1[38] = vsubq_s32(input[25], input[38]);
  x1[26] = vaddq_s32(input[26], input[37]);
  x1[37] = vsubq_s32(input[26], input[37]);
  x1[27] = vaddq_s32(input[27], input[36]);
  x1[36] = vsubq_s32(input[27], input[36]);
  x1[28] = vaddq_s32(input[28], input[35]);
  x1[35] = vsubq_s32(input[28], input[35]);
  x1[29] = vaddq_s32(input[29], input[34]);
  x1[34] = vsubq_s32(input[29], input[34]);
  x1[30] = vaddq_s32(input[30], input[33]);
  x1[33] = vsubq_s32(input[30], input[33]);
  x1[31] = vaddq_s32(input[31], input[32]);
  x1[32] = vsubq_s32(input[31], input[32]);

  // stage 2
  int32x4_t x2[64];
  x2[0] = vaddq_s32(x1[0], x1[31]);
  x2[31] = vsubq_s32(x1[0], x1[31]);
  x2[1] = vaddq_s32(x1[1], x1[30]);
  x2[30] = vsubq_s32(x1[1], x1[30]);
  x2[2] = vaddq_s32(x1[2], x1[29]);
  x2[29] = vsubq_s32(x1[2], x1[29]);
  x2[3] = vaddq_s32(x1[3], x1[28]);
  x2[28] = vsubq_s32(x1[3], x1[28]);
  x2[4] = vaddq_s32(x1[4], x1[27]);
  x2[27] = vsubq_s32(x1[4], x1[27]);
  x2[5] = vaddq_s32(x1[5], x1[26]);
  x2[26] = vsubq_s32(x1[5], x1[26]);
  x2[6] = vaddq_s32(x1[6], x1[25]);
  x2[25] = vsubq_s32(x1[6], x1[25]);
  x2[7] = vaddq_s32(x1[7], x1[24]);
  x2[24] = vsubq_s32(x1[7], x1[24]);
  x2[8] = vaddq_s32(x1[8], x1[23]);
  x2[23] = vsubq_s32(x1[8], x1[23]);
  x2[9] = vaddq_s32(x1[9], x1[22]);
  x2[22] = vsubq_s32(x1[9], x1[22]);
  x2[10] = vaddq_s32(x1[10], x1[21]);
  x2[21] = vsubq_s32(x1[10], x1[21]);
  x2[11] = vaddq_s32(x1[11], x1[20]);
  x2[20] = vsubq_s32(x1[11], x1[20]);
  x2[12] = vaddq_s32(x1[12], x1[19]);
  x2[19] = vsubq_s32(x1[12], x1[19]);
  x2[13] = vaddq_s32(x1[13], x1[18]);
  x2[18] = vsubq_s32(x1[13], x1[18]);
  x2[14] = vaddq_s32(x1[14], x1[17]);
  x2[17] = vsubq_s32(x1[14], x1[17]);
  x2[15] = vaddq_s32(x1[15], x1[16]);
  x2[16] = vsubq_s32(x1[15], x1[16]);
  x2[32] = x1[32];
  x2[33] = x1[33];
  x2[34] = x1[34];
  x2[35] = x1[35];
  x2[36] = x1[36];
  x2[37] = x1[37];
  x2[38] = x1[38];
  x2[39] = x1[39];

  btf_32_neon_type0(-cospi[32], cospi[32], x1[40], x1[55], x2[40], x2[55],
                    *v_cos_bit);
  btf_32_neon_type0(-cospi[32], cospi[32], x1[41], x1[54], x2[41], x2[54],
                    *v_cos_bit);
  btf_32_neon_type0(-cospi[32], cospi[32], x1[42], x1[53], x2[42], x2[53],
                    *v_cos_bit);
  btf_32_neon_type0(-cospi[32], cospi[32], x1[43], x1[52], x2[43], x2[52],
                    *v_cos_bit);
  btf_32_neon_type0(-cospi[32], cospi[32], x1[44], x1[51], x2[44], x2[51],
                    *v_cos_bit);
  btf_32_neon_type0(-cospi[32], cospi[32], x1[45], x1[50], x2[45], x2[50],
                    *v_cos_bit);
  btf_32_neon_type0(-cospi[32], cospi[32], x1[46], x1[49], x2[46], x2[49],
                    *v_cos_bit);
  btf_32_neon_type0(-cospi[32], cospi[32], x1[47], x1[48], x2[47], x2[48],
                    *v_cos_bit);
  x2[56] = x1[56];
  x2[57] = x1[57];
  x2[58] = x1[58];
  x2[59] = x1[59];
  x2[60] = x1[60];
  x2[61] = x1[61];
  x2[62] = x1[62];
  x2[63] = x1[63];

  // stage 3
  int32x4_t x3[64];
  x3[0] = vaddq_s32(x2[0], x2[15]);
  x3[15] = vsubq_s32(x2[0], x2[15]);
  x3[1] = vaddq_s32(x2[1], x2[14]);
  x3[14] = vsubq_s32(x2[1], x2[14]);
  x3[2] = vaddq_s32(x2[2], x2[13]);
  x3[13] = vsubq_s32(x2[2], x2[13]);
  x3[3] = vaddq_s32(x2[3], x2[12]);
  x3[12] = vsubq_s32(x2[3], x2[12]);
  x3[4] = vaddq_s32(x2[4], x2[11]);
  x3[11] = vsubq_s32(x2[4], x2[11]);
  x3[5] = vaddq_s32(x2[5], x2[10]);
  x3[10] = vsubq_s32(x2[5], x2[10]);
  x3[6] = vaddq_s32(x2[6], x2[9]);
  x3[9] = vsubq_s32(x2[6], x2[9]);
  x3[7] = vaddq_s32(x2[7], x2[8]);
  x3[8] = vsubq_s32(x2[7], x2[8]);
  x3[16] = x2[16];
  x3[17] = x2[17];
  x3[18] = x2[18];
  x3[19] = x2[19];
  btf_32_neon_type0(-cospi[32], cospi[32], x2[20], x2[27], x3[20], x3[27],
                    *v_cos_bit);
  btf_32_neon_type0(-cospi[32], cospi[32], x2[21], x2[26], x3[21], x3[26],
                    *v_cos_bit);
  btf_32_neon_type0(-cospi[32], cospi[32], x2[22], x2[25], x3[22], x3[25],
                    *v_cos_bit);
  btf_32_neon_type0(-cospi[32], cospi[32], x2[23], x2[24], x3[23], x3[24],
                    *v_cos_bit);
  x3[28] = x2[28];
  x3[29] = x2[29];
  x3[30] = x2[30];
  x3[31] = x2[31];
  x3[32] = vaddq_s32(x2[32], x2[47]);
  x3[47] = vsubq_s32(x2[32], x2[47]);
  x3[33] = vaddq_s32(x2[33], x2[46]);
  x3[46] = vsubq_s32(x2[33], x2[46]);
  x3[34] = vaddq_s32(x2[34], x2[45]);
  x3[45] = vsubq_s32(x2[34], x2[45]);
  x3[35] = vaddq_s32(x2[35], x2[44]);
  x3[44] = vsubq_s32(x2[35], x2[44]);
  x3[36] = vaddq_s32(x2[36], x2[43]);
  x3[43] = vsubq_s32(x2[36], x2[43]);
  x3[37] = vaddq_s32(x2[37], x2[42]);
  x3[42] = vsubq_s32(x2[37], x2[42]);
  x3[38] = vaddq_s32(x2[38], x2[41]);
  x3[41] = vsubq_s32(x2[38], x2[41]);
  x3[39] = vaddq_s32(x2[39], x2[40]);
  x3[40] = vsubq_s32(x2[39], x2[40]);
  x3[48] = vsubq_s32(x2[63], x2[48]);
  x3[63] = vaddq_s32(x2[63], x2[48]);
  x3[49] = vsubq_s32(x2[62], x2[49]);
  x3[62] = vaddq_s32(x2[62], x2[49]);
  x3[50] = vsubq_s32(x2[61], x2[50]);
  x3[61] = vaddq_s32(x2[61], x2[50]);
  x3[51] = vsubq_s32(x2[60], x2[51]);
  x3[60] = vaddq_s32(x2[60], x2[51]);
  x3[52] = vsubq_s32(x2[59], x2[52]);
  x3[59] = vaddq_s32(x2[59], x2[52]);
  x3[53] = vsubq_s32(x2[58], x2[53]);
  x3[58] = vaddq_s32(x2[58], x2[53]);
  x3[54] = vsubq_s32(x2[57], x2[54]);
  x3[57] = vaddq_s32(x2[57], x2[54]);
  x3[55] = vsubq_s32(x2[56], x2[55]);
  x3[56] = vaddq_s32(x2[56], x2[55]);

  // stage 4
  int32x4_t x4[64];
  x4[0] = vaddq_s32(x3[0], x3[7]);
  x4[7] = vsubq_s32(x3[0], x3[7]);
  x4[1] = vaddq_s32(x3[1], x3[6]);
  x4[6] = vsubq_s32(x3[1], x3[6]);
  x4[2] = vaddq_s32(x3[2], x3[5]);
  x4[5] = vsubq_s32(x3[2], x3[5]);
  x4[3] = vaddq_s32(x3[3], x3[4]);
  x4[4] = vsubq_s32(x3[3], x3[4]);
  x4[8] = x3[8];
  x4[9] = x3[9];
  btf_32_neon_type0(-cospi[32], cospi[32], x3[10], x3[13], x4[10], x4[13],
                    *v_cos_bit);
  btf_32_neon_type0(-cospi[32], cospi[32], x3[11], x3[12], x4[11], x4[12],
                    *v_cos_bit);
  x4[14] = x3[14];
  x4[15] = x3[15];
  x4[16] = vaddq_s32(x3[16], x3[23]);
  x4[23] = vsubq_s32(x3[16], x3[23]);
  x4[17] = vaddq_s32(x3[17], x3[22]);
  x4[22] = vsubq_s32(x3[17], x3[22]);
  x4[18] = vaddq_s32(x3[18], x3[21]);
  x4[21] = vsubq_s32(x3[18], x3[21]);
  x4[19] = vaddq_s32(x3[19], x3[20]);
  x4[20] = vsubq_s32(x3[19], x3[20]);
  x4[24] = vsubq_s32(x3[31], x3[24]);
  x4[31] = vaddq_s32(x3[31], x3[24]);
  x4[25] = vsubq_s32(x3[30], x3[25]);
  x4[30] = vaddq_s32(x3[30], x3[25]);
  x4[26] = vsubq_s32(x3[29], x3[26]);
  x4[29] = vaddq_s32(x3[29], x3[26]);
  x4[27] = vsubq_s32(x3[28], x3[27]);
  x4[28] = vaddq_s32(x3[28], x3[27]);
  x4[32] = x3[32];
  x4[33] = x3[33];
  x4[34] = x3[34];
  x4[35] = x3[35];

  btf_32_neon_type0(-cospi[16], cospi[48], x3[36], x3[59], x4[36], x4[59],
                    *v_cos_bit);
  btf_32_neon_type0(-cospi[16], cospi[48], x3[37], x3[58], x4[37], x4[58],
                    *v_cos_bit);
  btf_32_neon_type0(-cospi[16], cospi[48], x3[38], x3[57], x4[38], x4[57],
                    *v_cos_bit);
  btf_32_neon_type0(-cospi[16], cospi[48], x3[39], x3[56], x4[39], x4[56],
                    *v_cos_bit);
  btf_32_neon_type0(-cospi[48], -cospi[16], x3[40], x3[55], x4[40], x4[55],
                    *v_cos_bit);
  btf_32_neon_type0(-cospi[48], -cospi[16], x3[41], x3[54], x4[41], x4[54],
                    *v_cos_bit);
  btf_32_neon_type0(-cospi[48], -cospi[16], x3[42], x3[53], x4[42], x4[53],
                    *v_cos_bit);
  btf_32_neon_type0(-cospi[48], -cospi[16], x3[43], x3[52], x4[43], x4[52],
                    *v_cos_bit);
  x4[44] = x3[44];
  x4[45] = x3[45];
  x4[46] = x3[46];
  x4[47] = x3[47];
  x4[48] = x3[48];
  x4[49] = x3[49];
  x4[50] = x3[50];
  x4[51] = x3[51];
  x4[60] = x3[60];
  x4[61] = x3[61];
  x4[62] = x3[62];
  x4[63] = x3[63];

  // stage 5
  x5[0] = vaddq_s32(x4[0], x4[3]);
  x5[3] = vsubq_s32(x4[0], x4[3]);
  x5[1] = vaddq_s32(x4[1], x4[2]);
  x5[2] = vsubq_s32(x4[1], x4[2]);
  x5[4] = x4[4];

  btf_32_neon_type0(-cospi[32], cospi[32], x4[5], x4[6], x5[5], x5[6],
                    *v_cos_bit);
  x5[7] = x4[7];
  x5[8] = vaddq_s32(x4[8], x4[11]);
  x5[11] = vsubq_s32(x4[8], x4[11]);
  x5[9] = vaddq_s32(x4[9], x4[10]);
  x5[10] = vsubq_s32(x4[9], x4[10]);
  x5[12] = vsubq_s32(x4[15], x4[12]);
  x5[15] = vaddq_s32(x4[15], x4[12]);
  x5[13] = vsubq_s32(x4[14], x4[13]);
  x5[14] = vaddq_s32(x4[14], x4[13]);
  x5[16] = x4[16];
  x5[17] = x4[17];

  btf_32_neon_type0(-cospi[16], cospi[48], x4[18], x4[29], x5[18], x5[29],
                    *v_cos_bit);
  btf_32_neon_type0(-cospi[16], cospi[48], x4[19], x4[28], x5[19], x5[28],
                    *v_cos_bit);
  btf_32_neon_type0(-cospi[48], -cospi[16], x4[20], x4[27], x5[20], x5[27],
                    *v_cos_bit);
  btf_32_neon_type0(-cospi[48], -cospi[16], x4[21], x4[26], x5[21], x5[26],
                    *v_cos_bit);
  x5[22] = x4[22];
  x5[23] = x4[23];
  x5[24] = x4[24];
  x5[25] = x4[25];
  x5[30] = x4[30];
  x5[31] = x4[31];
  x5[32] = vaddq_s32(x4[32], x4[39]);
  x5[39] = vsubq_s32(x4[32], x4[39]);
  x5[33] = vaddq_s32(x4[33], x4[38]);
  x5[38] = vsubq_s32(x4[33], x4[38]);
  x5[34] = vaddq_s32(x4[34], x4[37]);
  x5[37] = vsubq_s32(x4[34], x4[37]);
  x5[35] = vaddq_s32(x4[35], x4[36]);
  x5[36] = vsubq_s32(x4[35], x4[36]);
  x5[40] = vsubq_s32(x4[47], x4[40]);
  x5[47] = vaddq_s32(x4[47], x4[40]);
  x5[41] = vsubq_s32(x4[46], x4[41]);
  x5[46] = vaddq_s32(x4[46], x4[41]);
  x5[42] = vsubq_s32(x4[45], x4[42]);
  x5[45] = vaddq_s32(x4[45], x4[42]);
  x5[43] = vsubq_s32(x4[44], x4[43]);
  x5[44] = vaddq_s32(x4[44], x4[43]);
  x5[48] = vaddq_s32(x4[48], x4[55]);
  x5[55] = vsubq_s32(x4[48], x4[55]);
  x5[49] = vaddq_s32(x4[49], x4[54]);
  x5[54] = vsubq_s32(x4[49], x4[54]);
  x5[50] = vaddq_s32(x4[50], x4[53]);
  x5[53] = vsubq_s32(x4[50], x4[53]);
  x5[51] = vaddq_s32(x4[51], x4[52]);
  x5[52] = vsubq_s32(x4[51], x4[52]);
  x5[56] = vsubq_s32(x4[63], x4[56]);
  x5[63] = vaddq_s32(x4[63], x4[56]);
  x5[57] = vsubq_s32(x4[62], x4[57]);
  x5[62] = vaddq_s32(x4[62], x4[57]);
  x5[58] = vsubq_s32(x4[61], x4[58]);
  x5[61] = vaddq_s32(x4[61], x4[58]);
  x5[59] = vsubq_s32(x4[60], x4[59]);
  x5[60] = vaddq_s32(x4[60], x4[59]);
}

static void av1_fdct64_new_neon(int32x4_t *input, int32x4_t *output,
                                int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);
  const int32x4_t v_cos_bit = vdupq_n_s32(-cos_bit);

  // stage 1-2-3-4-5
  int32x4_t x5[64];
  av1_fdct64_new_stage12345_neon(input, x5, cospi, &v_cos_bit);

  // stage 6
  int32x4_t x6[64];
  btf_32_neon_type0(cospi[32], cospi[32], x5[0], x5[1], x6[0], x6[1],
                    v_cos_bit);
  btf_32_neon_type1(cospi[48], cospi[16], x5[2], x5[3], x6[2], x6[3],
                    v_cos_bit);
  x6[4] = vaddq_s32(x5[4], x5[5]);
  x6[5] = vsubq_s32(x5[4], x5[5]);
  x6[6] = vsubq_s32(x5[7], x5[6]);
  x6[7] = vaddq_s32(x5[7], x5[6]);
  x6[8] = x5[8];
  btf_32_neon_type0(-cospi[16], cospi[48], x5[9], x5[14], x6[9], x6[14],
                    v_cos_bit);
  btf_32_neon_type0(-cospi[48], -cospi[16], x5[10], x5[13], x6[10], x6[13],
                    v_cos_bit);
  x6[11] = x5[11];
  x6[12] = x5[12];
  x6[15] = x5[15];
  x6[16] = vaddq_s32(x5[16], x5[19]);
  x6[19] = vsubq_s32(x5[16], x5[19]);
  x6[17] = vaddq_s32(x5[17], x5[18]);
  x6[18] = vsubq_s32(x5[17], x5[18]);
  x6[20] = vsubq_s32(x5[23], x5[20]);
  x6[23] = vaddq_s32(x5[23], x5[20]);
  x6[21] = vsubq_s32(x5[22], x5[21]);
  x6[22] = vaddq_s32(x5[22], x5[21]);
  x6[24] = vaddq_s32(x5[24], x5[27]);
  x6[27] = vsubq_s32(x5[24], x5[27]);
  x6[25] = vaddq_s32(x5[25], x5[26]);
  x6[26] = vsubq_s32(x5[25], x5[26]);
  x6[28] = vsubq_s32(x5[31], x5[28]);
  x6[31] = vaddq_s32(x5[31], x5[28]);
  x6[29] = vsubq_s32(x5[30], x5[29]);
  x6[30] = vaddq_s32(x5[30], x5[29]);
  x6[32] = x5[32];
  x6[33] = x5[33];

  btf_32_neon_type0(-cospi[40], cospi[24], x5[42], x5[53], x6[42], x6[53],
                    v_cos_bit);
  btf_32_neon_type0(-cospi[40], cospi[24], x5[43], x5[52], x6[43], x6[52],
                    v_cos_bit);
  btf_32_neon_type0(-cospi[24], -cospi[40], x5[44], x5[51], x6[44], x6[51],
                    v_cos_bit);
  btf_32_neon_type0(-cospi[24], -cospi[40], x5[45], x5[50], x6[45], x6[50],
                    v_cos_bit);

  x6[46] = x5[46];
  x6[47] = x5[47];
  x6[48] = x5[48];
  x6[49] = x5[49];
  x6[54] = x5[54];
  x6[55] = x5[55];
  x6[56] = x5[56];
  x6[57] = x5[57];
  x6[62] = x5[62];
  x6[63] = x5[63];

  // stage 7
  int32x4_t x7[64];
  x7[0] = x6[0];
  x7[1] = x6[1];
  x7[2] = x6[2];
  x7[3] = x6[3];
  btf_32_neon_type1(cospi[24], cospi[40], x6[5], x6[6], x7[5], x7[6],
                    v_cos_bit);

  x7[8] = vaddq_s32(x6[8], x6[9]);
  x7[9] = vsubq_s32(x6[8], x6[9]);
  x7[10] = vsubq_s32(x6[11], x6[10]);
  x7[11] = vaddq_s32(x6[11], x6[10]);
  x7[12] = vaddq_s32(x6[12], x6[13]);
  x7[13] = vsubq_s32(x6[12], x6[13]);
  x7[14] = vsubq_s32(x6[15], x6[14]);
  x7[15] = vaddq_s32(x6[15], x6[14]);
  x7[16] = x6[16];

  btf_32_neon_type0(-cospi[40], cospi[24], x6[21], x6[26], x7[21], x7[26],
                    v_cos_bit);
  btf_32_neon_type0(-cospi[24], -cospi[40], x6[22], x6[25], x7[22], x7[25],
                    v_cos_bit);
  x7[23] = x6[23];
  x7[24] = x6[24];
  x7[27] = x6[27];
  x7[28] = x6[28];
  x7[31] = x6[31];

  btf_32_neon_type0(-cospi[8], cospi[56], x5[34], x5[61], x6[34], x6[61],
                    v_cos_bit);
  btf_32_neon_type0(-cospi[8], cospi[56], x5[35], x5[60], x6[35], x6[60],
                    v_cos_bit);
  btf_32_neon_type0(-cospi[56], -cospi[8], x5[36], x5[59], x6[36], x6[59],
                    v_cos_bit);
  btf_32_neon_type0(-cospi[56], -cospi[8], x5[37], x5[58], x6[37], x6[58],
                    v_cos_bit);
  x6[38] = x5[38];
  x6[39] = x5[39];
  x6[40] = x5[40];
  x6[41] = x5[41];

  btf_32_neon_type1(cospi[56], cospi[8], x6[4], x6[7], x7[4], x7[7], v_cos_bit);
  btf_32_neon_type0(-cospi[8], cospi[56], x6[17], x6[30], x7[17], x7[30],
                    v_cos_bit);
  btf_32_neon_type0(-cospi[56], -cospi[8], x6[18], x6[29], x7[18], x7[29],
                    v_cos_bit);
  x7[19] = x6[19];
  x7[20] = x6[20];

  x7[32] = vaddq_s32(x6[32], x6[35]);
  x7[35] = vsubq_s32(x6[32], x6[35]);
  x7[33] = vaddq_s32(x6[33], x6[34]);
  x7[34] = vsubq_s32(x6[33], x6[34]);
  x7[36] = vsubq_s32(x6[39], x6[36]);
  x7[39] = vaddq_s32(x6[39], x6[36]);
  x7[37] = vsubq_s32(x6[38], x6[37]);
  x7[38] = vaddq_s32(x6[38], x6[37]);
  x7[40] = vaddq_s32(x6[40], x6[43]);
  x7[43] = vsubq_s32(x6[40], x6[43]);
  x7[41] = vaddq_s32(x6[41], x6[42]);
  x7[42] = vsubq_s32(x6[41], x6[42]);
  x7[44] = vsubq_s32(x6[47], x6[44]);
  x7[47] = vaddq_s32(x6[47], x6[44]);
  x7[45] = vsubq_s32(x6[46], x6[45]);
  x7[46] = vaddq_s32(x6[46], x6[45]);
  x7[48] = vaddq_s32(x6[48], x6[51]);
  x7[51] = vsubq_s32(x6[48], x6[51]);
  x7[49] = vaddq_s32(x6[49], x6[50]);
  x7[50] = vsubq_s32(x6[49], x6[50]);
  x7[52] = vsubq_s32(x6[55], x6[52]);
  x7[55] = vaddq_s32(x6[55], x6[52]);
  x7[53] = vsubq_s32(x6[54], x6[53]);
  x7[54] = vaddq_s32(x6[54], x6[53]);
  x7[56] = vaddq_s32(x6[56], x6[59]);
  x7[59] = vsubq_s32(x6[56], x6[59]);
  x7[57] = vaddq_s32(x6[57], x6[58]);
  x7[58] = vsubq_s32(x6[57], x6[58]);
  x7[60] = vsubq_s32(x6[63], x6[60]);
  x7[63] = vaddq_s32(x6[63], x6[60]);
  x7[61] = vsubq_s32(x6[62], x6[61]);
  x7[62] = vaddq_s32(x6[62], x6[61]);

  // stage 8
  int32x4_t x8[64];
  x8[0] = x7[0];
  x8[1] = x7[1];
  x8[2] = x7[2];
  x8[3] = x7[3];
  x8[4] = x7[4];
  x8[5] = x7[5];
  x8[6] = x7[6];
  x8[7] = x7[7];

  btf_32_neon_type1(cospi[60], cospi[4], x7[8], x7[15], x8[8], x8[15],
                    v_cos_bit);
  btf_32_neon_type1(cospi[28], cospi[36], x7[9], x7[14], x8[9], x8[14],
                    v_cos_bit);
  btf_32_neon_type1(cospi[44], cospi[20], x7[10], x7[13], x8[10], x8[13],
                    v_cos_bit);
  btf_32_neon_type1(cospi[12], cospi[52], x7[11], x7[12], x8[11], x8[12],
                    v_cos_bit);
  x8[16] = vaddq_s32(x7[16], x7[17]);
  x8[17] = vsubq_s32(x7[16], x7[17]);
  x8[18] = vsubq_s32(x7[19], x7[18]);
  x8[19] = vaddq_s32(x7[19], x7[18]);
  x8[20] = vaddq_s32(x7[20], x7[21]);
  x8[21] = vsubq_s32(x7[20], x7[21]);
  x8[22] = vsubq_s32(x7[23], x7[22]);
  x8[23] = vaddq_s32(x7[23], x7[22]);
  x8[24] = vaddq_s32(x7[24], x7[25]);
  x8[25] = vsubq_s32(x7[24], x7[25]);
  x8[26] = vsubq_s32(x7[27], x7[26]);
  x8[27] = vaddq_s32(x7[27], x7[26]);
  x8[28] = vaddq_s32(x7[28], x7[29]);
  x8[29] = vsubq_s32(x7[28], x7[29]);
  x8[30] = vsubq_s32(x7[31], x7[30]);
  x8[31] = vaddq_s32(x7[31], x7[30]);
  x8[32] = x7[32];

  btf_32_neon_type0(-cospi[4], cospi[60], x7[33], x7[62], x8[33], x8[62],
                    v_cos_bit);
  btf_32_neon_type0(-cospi[60], -cospi[4], x7[34], x7[61], x8[34], x8[61],
                    v_cos_bit);
  x8[35] = x7[35];
  x8[36] = x7[36];
  btf_32_neon_type0(-cospi[36], cospi[28], x7[37], x7[58], x8[37], x8[58],
                    v_cos_bit);
  btf_32_neon_type0(-cospi[28], -cospi[36], x7[38], x7[57], x8[38], x8[57],
                    v_cos_bit);
  x8[39] = x7[39];
  x8[40] = x7[40];
  btf_32_neon_type0(-cospi[20], cospi[44], x7[41], x7[54], x8[41], x8[54],
                    v_cos_bit);
  btf_32_neon_type0(-cospi[44], -cospi[20], x7[42], x7[53], x8[42], x8[53],
                    v_cos_bit);
  x8[43] = x7[43];
  x8[44] = x7[44];
  btf_32_neon_type0(-cospi[52], cospi[12], x7[45], x7[50], x8[45], x8[50],
                    v_cos_bit);
  btf_32_neon_type0(-cospi[12], -cospi[52], x7[46], x7[49], x8[46], x8[49],
                    v_cos_bit);
  x8[47] = x7[47];
  x8[48] = x7[48];
  x8[51] = x7[51];
  x8[52] = x7[52];
  x8[55] = x7[55];
  x8[56] = x7[56];
  x8[59] = x7[59];
  x8[60] = x7[60];
  x8[63] = x7[63];

  // stage 9
  int32x4_t x9[64];
  x9[0] = x8[0];
  x9[1] = x8[1];
  x9[2] = x8[2];
  x9[3] = x8[3];
  x9[4] = x8[4];
  x9[5] = x8[5];
  x9[6] = x8[6];
  x9[7] = x8[7];
  x9[8] = x8[8];
  x9[9] = x8[9];
  x9[10] = x8[10];
  x9[11] = x8[11];
  x9[12] = x8[12];
  x9[13] = x8[13];
  x9[14] = x8[14];
  x9[15] = x8[15];

  btf_32_neon_type1(cospi[62], cospi[2], x8[16], x8[31], x9[16], x9[31],
                    v_cos_bit);
  btf_32_neon_type1(cospi[30], cospi[34], x8[17], x8[30], x9[17], x9[30],
                    v_cos_bit);
  btf_32_neon_type1(cospi[46], cospi[18], x8[18], x8[29], x9[18], x9[29],
                    v_cos_bit);
  btf_32_neon_type1(cospi[14], cospi[50], x8[19], x8[28], x9[19], x9[28],
                    v_cos_bit);
  btf_32_neon_type1(cospi[54], cospi[10], x8[20], x8[27], x9[20], x9[27],
                    v_cos_bit);
  btf_32_neon_type1(cospi[22], cospi[42], x8[21], x8[26], x9[21], x9[26],
                    v_cos_bit);
  btf_32_neon_type1(cospi[38], cospi[26], x8[22], x8[25], x9[22], x9[25],
                    v_cos_bit);
  btf_32_neon_type1(cospi[6], cospi[58], x8[23], x8[24], x9[23], x9[24],
                    v_cos_bit);

  x9[32] = vaddq_s32(x8[32], x8[33]);
  x9[33] = vsubq_s32(x8[32], x8[33]);
  x9[34] = vsubq_s32(x8[35], x8[34]);
  x9[35] = vaddq_s32(x8[35], x8[34]);
  x9[36] = vaddq_s32(x8[36], x8[37]);
  x9[37] = vsubq_s32(x8[36], x8[37]);
  x9[38] = vsubq_s32(x8[39], x8[38]);
  x9[39] = vaddq_s32(x8[39], x8[38]);
  x9[40] = vaddq_s32(x8[40], x8[41]);
  x9[41] = vsubq_s32(x8[40], x8[41]);
  x9[42] = vsubq_s32(x8[43], x8[42]);
  x9[43] = vaddq_s32(x8[43], x8[42]);
  x9[44] = vaddq_s32(x8[44], x8[45]);
  x9[45] = vsubq_s32(x8[44], x8[45]);
  x9[46] = vsubq_s32(x8[47], x8[46]);
  x9[47] = vaddq_s32(x8[47], x8[46]);
  x9[48] = vaddq_s32(x8[48], x8[49]);
  x9[49] = vsubq_s32(x8[48], x8[49]);
  x9[50] = vsubq_s32(x8[51], x8[50]);
  x9[51] = vaddq_s32(x8[51], x8[50]);
  x9[52] = vaddq_s32(x8[52], x8[53]);
  x9[53] = vsubq_s32(x8[52], x8[53]);
  x9[54] = vsubq_s32(x8[55], x8[54]);
  x9[55] = vaddq_s32(x8[55], x8[54]);
  x9[56] = vaddq_s32(x8[56], x8[57]);
  x9[57] = vsubq_s32(x8[56], x8[57]);
  x9[58] = vsubq_s32(x8[59], x8[58]);
  x9[59] = vaddq_s32(x8[59], x8[58]);
  x9[60] = vaddq_s32(x8[60], x8[61]);
  x9[61] = vsubq_s32(x8[60], x8[61]);
  x9[62] = vsubq_s32(x8[63], x8[62]);
  x9[63] = vaddq_s32(x8[63], x8[62]);

  // stage 10
  int32x4_t x10[64];
  x10[0] = x9[0];
  x10[1] = x9[1];
  x10[2] = x9[2];
  x10[3] = x9[3];
  x10[4] = x9[4];
  x10[5] = x9[5];
  x10[6] = x9[6];
  x10[7] = x9[7];
  x10[8] = x9[8];
  x10[9] = x9[9];
  x10[10] = x9[10];
  x10[11] = x9[11];
  x10[12] = x9[12];
  x10[13] = x9[13];
  x10[14] = x9[14];
  x10[15] = x9[15];
  x10[16] = x9[16];
  x10[17] = x9[17];
  x10[18] = x9[18];
  x10[19] = x9[19];
  x10[20] = x9[20];
  x10[21] = x9[21];
  x10[22] = x9[22];
  x10[23] = x9[23];
  x10[24] = x9[24];
  x10[25] = x9[25];
  x10[26] = x9[26];
  x10[27] = x9[27];
  x10[28] = x9[28];
  x10[29] = x9[29];
  x10[30] = x9[30];
  x10[31] = x9[31];
  btf_32_neon_type1(cospi[63], cospi[1], x9[32], x9[63], x10[32], x10[63],
                    v_cos_bit);
  btf_32_neon_type1(cospi[31], cospi[33], x9[33], x9[62], x10[33], x10[62],
                    v_cos_bit);
  btf_32_neon_type1(cospi[47], cospi[17], x9[34], x9[61], x10[34], x10[61],
                    v_cos_bit);
  btf_32_neon_type1(cospi[15], cospi[49], x9[35], x9[60], x10[35], x10[60],
                    v_cos_bit);
  btf_32_neon_type1(cospi[55], cospi[9], x9[36], x9[59], x10[36], x10[59],
                    v_cos_bit);
  btf_32_neon_type1(cospi[23], cospi[41], x9[37], x9[58], x10[37], x10[58],
                    v_cos_bit);
  btf_32_neon_type1(cospi[39], cospi[25], x9[38], x9[57], x10[38], x10[57],
                    v_cos_bit);
  btf_32_neon_type1(cospi[7], cospi[57], x9[39], x9[56], x10[39], x10[56],
                    v_cos_bit);
  btf_32_neon_type1(cospi[59], cospi[5], x9[40], x9[55], x10[40], x10[55],
                    v_cos_bit);
  btf_32_neon_type1(cospi[27], cospi[37], x9[41], x9[54], x10[41], x10[54],
                    v_cos_bit);
  btf_32_neon_type1(cospi[43], cospi[21], x9[42], x9[53], x10[42], x10[53],
                    v_cos_bit);
  btf_32_neon_type1(cospi[11], cospi[53], x9[43], x9[52], x10[43], x10[52],
                    v_cos_bit);
  btf_32_neon_type1(cospi[51], cospi[13], x9[44], x9[51], x10[44], x10[51],
                    v_cos_bit);
  btf_32_neon_type1(cospi[19], cospi[45], x9[45], x9[50], x10[45], x10[50],
                    v_cos_bit);
  btf_32_neon_type1(cospi[35], cospi[29], x9[46], x9[49], x10[46], x10[49],
                    v_cos_bit);
  btf_32_neon_type1(cospi[3], cospi[61], x9[47], x9[48], x10[47], x10[48],
                    v_cos_bit);

  // stage 11
  output[0] = x10[0];
  output[1] = x10[32];
  output[2] = x10[16];
  output[3] = x10[48];
  output[4] = x10[8];
  output[5] = x10[40];
  output[6] = x10[24];
  output[7] = x10[56];
  output[8] = x10[4];
  output[9] = x10[36];
  output[10] = x10[20];
  output[11] = x10[52];
  output[12] = x10[12];
  output[13] = x10[44];
  output[14] = x10[28];
  output[15] = x10[60];
  output[16] = x10[2];
  output[17] = x10[34];
  output[18] = x10[18];
  output[19] = x10[50];
  output[20] = x10[10];
  output[21] = x10[42];
  output[22] = x10[26];
  output[23] = x10[58];
  output[24] = x10[6];
  output[25] = x10[38];
  output[26] = x10[22];
  output[27] = x10[54];
  output[28] = x10[14];
  output[29] = x10[46];
  output[30] = x10[30];
  output[31] = x10[62];
  output[32] = x10[1];
  output[33] = x10[33];
  output[34] = x10[17];
  output[35] = x10[49];
  output[36] = x10[9];
  output[37] = x10[41];
  output[38] = x10[25];
  output[39] = x10[57];
  output[40] = x10[5];
  output[41] = x10[37];
  output[42] = x10[21];
  output[43] = x10[53];
  output[44] = x10[13];
  output[45] = x10[45];
  output[46] = x10[29];
  output[47] = x10[61];
  output[48] = x10[3];
  output[49] = x10[35];
  output[50] = x10[19];
  output[51] = x10[51];
  output[52] = x10[11];
  output[53] = x10[43];
  output[54] = x10[27];
  output[55] = x10[59];
  output[56] = x10[7];
  output[57] = x10[39];
  output[58] = x10[23];
  output[59] = x10[55];
  output[60] = x10[15];
  output[61] = x10[47];
  output[62] = x10[31];
  output[63] = x10[63];
}

void av1_idtx32_new_neon(int32x4_t *input, int32x4_t *output, int cos_bit) {
  (void)cos_bit;
  for (int i = 0; i < 32; i++) {
    output[i] = vshlq_n_s32(input[i], 2);
  }
}

static const fwd_transform_1d_neon col_highbd_txfm8x32_arr[TX_TYPES] = {
  av1_fdct32_new_neon,  // DCT_DCT
  NULL,                 // ADST_DCT
  NULL,                 // DCT_ADST
  NULL,                 // ADST_ADST
  NULL,                 // FLIPADST_DCT
  NULL,                 // DCT_FLIPADST
  NULL,                 // FLIPADST_FLIPADST
  NULL,                 // ADST_FLIPADST
  NULL,                 // FLIPADST_ADST
  av1_idtx32_new_neon,  // IDTX
  NULL,                 // V_DCT
  NULL,                 // H_DCT
  NULL,                 // V_ADST
  NULL,                 // H_ADST
  NULL,                 // V_FLIPADST
  NULL                  // H_FLIPADST
};

static const fwd_transform_1d_many_neon row_highbd_txfm8x32_arr[TX_TYPES] = {
  fdct16x16_neon,  // DCT_DCT
  NULL,            // ADST_DCT
  NULL,            // DCT_ADST
  NULL,            // ADST_ADST
  NULL,            // FLIPADST_DCT
  NULL,            // DCT_FLIPADST
  NULL,            // FLIPADST_FLIPADST
  NULL,            // ADST_FLIPADST
  NULL,            // FLIPADST_ADST
  idtx16x16_neon,  // IDTX
  NULL,            // V_DCT
  NULL,            // H_DCT
  NULL,            // V_ADST
  NULL,            // H_ADST
  NULL,            // V_FLIPADST
  NULL             // H_FLIPADST
};

void av1_fwd_txfm2d_16x8_neon(const int16_t *input, int32_t *coeff, int stride,
                              TX_TYPE tx_type, int bd) {
  (void)bd;
  const fwd_transform_1d_many_neon col_txfm = col_highbd_txfm8x8_arr[tx_type];
  const fwd_transform_1d_many_neon row_txfm = row_highbd_txfm8x16_arr[tx_type];
  int bit = av1_fwd_cos_bit_col[2][1];

  int ud_flip, lr_flip;
  get_flip_cfg(tx_type, &ud_flip, &lr_flip);
  ud_adjust_input_and_stride(ud_flip, &input, &stride, 8);

  int32x4_t buf0[32];
  load_buffer_16x8(input, buf0, stride, lr_flip, 2);
  for (int i = 0; i < 2; i++) {
    col_txfm(buf0 + i * 16, buf0 + i * 16, bit, 2, 8);
    col_txfm_8x8_rounding(buf0 + i * 16, -2);
  }

  int32x4_t buf1[32];
  transpose_arrays_s32_16x8(buf0, buf1);

  row_txfm(buf1, buf1, bit, 2, 16);
  av1_round_shift_rect_array_32_neon(buf1, buf1, 32, 0, NewSqrt2);
  store_buffer_16x8(buf1, coeff);
}

void av1_fwd_txfm2d_8x16_neon(const int16_t *input, int32_t *coeff, int stride,
                              TX_TYPE tx_type, int bd) {
  (void)bd;
  const fwd_transform_1d_many_neon col_txfm = col_highbd_txfm8x16_arr[tx_type];
  const fwd_transform_1d_many_neon row_txfm = row_highbd_txfm8x8_arr[tx_type];
  int bit = av1_fwd_cos_bit_col[1][2];

  int ud_flip, lr_flip;
  get_flip_cfg(tx_type, &ud_flip, &lr_flip);
  ud_adjust_input_and_stride(ud_flip, &input, &stride, 16);

  int32x4_t buf0[32];
  load_buffer_8x16(input, buf0, stride, lr_flip, 2);
  col_txfm(buf0, buf0, bit, 2, 16);
  col_txfm_8x16_rounding(buf0, -2);

  int32x4_t buf1[32];
  transpose_arrays_s32_8x16(buf0, buf1);

  row_txfm(buf1, buf1, bit, 4, 8);
  av1_round_shift_rect_array_32_neon(buf1, buf1, 32, 0, NewSqrt2);
  store_buffer_8x16(buf1, coeff);
}

#if !CONFIG_REALTIME_ONLY
void av1_fwd_txfm2d_4x16_neon(const int16_t *input, int32_t *coeff, int stride,
                              TX_TYPE tx_type, int bd) {
  (void)bd;
  int bitcol = av1_fwd_cos_bit_col[0][2];
  int bitrow = av1_fwd_cos_bit_row[0][2];
  const fwd_transform_1d_many_neon col_txfm = col_highbd_txfm8x16_arr[tx_type];
  const fwd_transform_1d_neon row_txfm = row_highbd_txfm4x4_arr[tx_type];

  int ud_flip, lr_flip;
  get_flip_cfg(tx_type, &ud_flip, &lr_flip);
  ud_adjust_input_and_stride(ud_flip, &input, &stride, 16);

  // Column-wise transform.
  int32x4_t buf0[16];
  load_buffer_4x16(input, buf0, stride, lr_flip, 2);
  col_txfm(buf0, buf0, bitcol, 1, 0);
  col_txfm_8x8_rounding(buf0, -1);

  int32x4_t buf1[16];
  transpose_arrays_s32_4x16(buf0, buf1);

  // Row-wise transform.
  for (int i = 0; i < 4; i++) {
    row_txfm(buf1 + i * 4, buf1 + i * 4, bitrow);
  }
  store_buffer_4x16(buf1, coeff);
}
#endif

void av1_fwd_txfm2d_16x4_neon(const int16_t *input, int32_t *coeff, int stride,
                              TX_TYPE tx_type, int bd) {
  (void)bd;
  int bitcol = av1_fwd_cos_bit_col[2][0];
  int bitrow = av1_fwd_cos_bit_row[2][0];
  const fwd_transform_1d_neon col_txfm = col_highbd_txfm4x4_arr[tx_type];
  const fwd_transform_1d_many_neon row_txfm = row_highbd_txfm8x16_arr[tx_type];

  int ud_flip, lr_flip;
  get_flip_cfg(tx_type, &ud_flip, &lr_flip);
  ud_adjust_input_and_stride(ud_flip, &input, &stride, 4);

  // Column-wise transform.
  int32x4_t buf0[16];
  load_buffer_16x4(input, buf0, stride, lr_flip, 2);
  for (int i = 0; i < 4; i++) {
    int32x4_t *cur_in = buf0 + i * 4;
    col_txfm(cur_in, cur_in, bitcol);
    transpose_arrays_s32_4x4(cur_in, cur_in);
  }
  col_txfm_8x8_rounding(buf0, -1);

  // Row-wise transform.
  row_txfm(buf0, buf0, bitrow, 1, 0);
  store_buffer_16x4(buf0, coeff);
}

void av1_fwd_txfm2d_16x32_neon(const int16_t *input, int32_t *coeff, int stride,
                               TX_TYPE tx_type, int bd) {
  (void)bd;
  const fwd_transform_1d_neon col_txfm = col_highbd_txfm8x32_arr[tx_type];
  const fwd_transform_1d_many_neon row_txfm = row_highbd_txfm8x32_arr[tx_type];
  int bitcol = av1_fwd_cos_bit_col[2][3];
  int bitrow = av1_fwd_cos_bit_row[2][3];

  // Column-wise transform.
  int32x4_t buf0[128];
  load_buffer_16x32(input, buf0, stride, 0, 2);
  for (int i = 0; i < 4; i++) {
    col_txfm(buf0 + i * 32, buf0 + i * 32, bitcol);
  }
  col_txfm_16x16_rounding(&buf0[0], -4);
  col_txfm_16x16_rounding(&buf0[64], -4);

  int32x4_t buf1[128];
  transpose_arrays_s32_16x32(buf0, buf1);

  // Row-wise transform.
  row_txfm(buf1, buf1, bitrow, 8, 16);
  av1_round_shift_rect_array_32_neon(buf1, buf1, 128, 0, NewSqrt2);
  store_buffer_16x32(buf1, coeff);
}

void av1_fwd_txfm2d_32x64_neon(const int16_t *input, int32_t *coeff, int stride,
                               TX_TYPE tx_type, int bd) {
  (void)bd;
  (void)tx_type;
  int bitcol = av1_fwd_cos_bit_col[3][4];
  int bitrow = av1_fwd_cos_bit_row[3][4];

  // Column-wise transform.
  int32x4_t buf0[512];
  load_buffer_32x64(input, buf0, stride, 0, 0);
  for (int i = 0; i < 8; i++) {
    av1_fdct64_new_neon(buf0 + i * 64, buf0 + i * 64, bitcol);
  }
  for (int i = 0; i < 8; i++) {
    col_txfm_16x16_rounding(buf0 + i * 64, -2);
  }

  int32x4_t buf1[512];
  transpose_arrays_s32_32x64(buf0, buf1);

  // Row-wise transform.
  for (int i = 0; i < 16; i++) {
    av1_fdct32_new_neon(buf1 + i * 32, buf1 + i * 32, bitrow);
  }
  av1_round_shift_rect_array_32_neon(buf1, buf1, 512, 2, NewSqrt2);
  store_buffer_32x32(buf1, coeff);
}

void av1_fwd_txfm2d_64x32_neon(const int16_t *input, int32_t *coeff, int stride,
                               TX_TYPE tx_type, int bd) {
  (void)bd;
  (void)tx_type;
  int bitcol = av1_fwd_cos_bit_col[4][3];
  int bitrow = av1_fwd_cos_bit_row[4][3];

  // Column-wise transform.
  int32x4_t buf0[512];
  load_buffer_64x32(input, buf0, stride, 0, 2);
  for (int i = 0; i < 16; i++) {
    av1_fdct32_new_neon(buf0 + i * 32, buf0 + i * 32, bitcol);
  }
  for (int i = 0; i < 8; i++) {
    col_txfm_16x16_rounding(buf0 + i * 64, -4);
  }

  int32x4_t buf1[512];
  transpose_arrays_s32_64x32(buf0, buf1);

  // Row-wise transform.
  for (int i = 0; i < 8; i++) {
    av1_fdct64_new_neon(buf1 + i * 64, buf1 + i * 64, bitrow);
  }
  av1_round_shift_rect_array_32_neon(buf1, buf1, 512, 2, NewSqrt2);
  store_buffer_64x32(buf1, coeff);
}

void av1_fwd_txfm2d_32x16_neon(const int16_t *input, int32_t *coeff, int stride,
                               TX_TYPE tx_type, int bd) {
  (void)bd;
  const fwd_transform_1d_many_neon col_txfm = row_highbd_txfm8x32_arr[tx_type];
  const fwd_transform_1d_neon row_txfm = col_highbd_txfm8x32_arr[tx_type];
  int bitcol = av1_fwd_cos_bit_col[3][2];
  int bitrow = av1_fwd_cos_bit_row[3][2];

  // Column-wise transform.
  int32x4_t buf0[128];
  load_buffer_32x16(input, buf0, stride, 0, 2);
  col_txfm(buf0, buf0, bitcol, 8, 16);
  col_txfm_16x16_rounding(&buf0[0], -4);
  col_txfm_16x16_rounding(&buf0[64], -4);

  int32x4_t buf1[128];
  transpose_arrays_s32_32x16(buf0, buf1);

  // Row-wise transform.
  for (int i = 0; i < 4; i++) {
    row_txfm(buf1 + i * 32, buf1 + i * 32, bitrow);
  }
  av1_round_shift_rect_array_32_neon(buf1, buf1, 128, 0, NewSqrt2);
  store_buffer_32x16(buf1, coeff);
}

#if !CONFIG_REALTIME_ONLY
void av1_fwd_txfm2d_8x32_neon(const int16_t *input, int32_t *coeff, int stride,
                              TX_TYPE tx_type, int bd) {
  (void)bd;
  const fwd_transform_1d_neon col_txfm = col_highbd_txfm8x32_arr[tx_type];
  const fwd_transform_1d_many_neon row_txfm = row_highbd_txfm32x8_arr[tx_type];
  int bitcol = av1_fwd_cos_bit_col[1][3];
  int bitrow = av1_fwd_cos_bit_row[1][3];

  // Column-wise transform.
  int32x4_t buf0[64];
  load_buffer_8x32(input, buf0, stride, 0, 2);
  for (int i = 0; i < 2; i++) {
    col_txfm(buf0 + i * 32, buf0 + i * 32, bitcol);
  }
  col_txfm_16x16_rounding(buf0, -2);

  int32x4_t buf1[64];
  transpose_arrays_s32_8x32(buf0, buf1);

  // Row-wise transform.
  row_txfm(buf1, buf1, bitrow, 8, 8);
  store_buffer_8x32(buf1, coeff);
}

void av1_fwd_txfm2d_32x8_neon(const int16_t *input, int32_t *coeff, int stride,
                              TX_TYPE tx_type, int bd) {
  (void)bd;
  const fwd_transform_1d_many_neon col_txfm = row_highbd_txfm32x8_arr[tx_type];
  const fwd_transform_1d_neon row_txfm = col_highbd_txfm8x32_arr[tx_type];
  int bitcol = av1_fwd_cos_bit_col[3][1];
  int bitrow = av1_fwd_cos_bit_row[3][1];

  // Column-wise transform.
  int32x4_t buf0[64];
  load_buffer_32x8(input, buf0, stride, 0, 2);
  col_txfm(buf0, buf0, bitcol, 8, 8);
  col_txfm_16x16_rounding(&buf0[0], -2);

  int32x4_t buf1[64];
  transpose_arrays_s32_32x8(buf0, buf1);

  // Row-wise transform.
  for (int i = 0; i < 2; i++) {
    row_txfm(buf1 + i * 32, buf1 + i * 32, bitrow);
  }
  store_buffer_32x8(buf1, coeff);
}
#endif

void av1_fwd_txfm2d_4x8_neon(const int16_t *input, int32_t *coeff, int stride,
                             TX_TYPE tx_type, int bd) {
  (void)bd;
  int bitcol = av1_fwd_cos_bit_col[0][1];
  int bitrow = av1_fwd_cos_bit_row[0][1];
  const fwd_transform_1d_many_neon col_txfm = col_highbd_txfm4x8_arr[tx_type];
  const fwd_transform_1d_neon row_txfm = row_highbd_txfm4x4_arr[tx_type];

  int ud_flip, lr_flip;
  get_flip_cfg(tx_type, &ud_flip, &lr_flip);
  ud_adjust_input_and_stride(ud_flip, &input, &stride, 8);

  int32x4_t buf0[8];
  load_buffer_4x8(input, buf0, stride, lr_flip, 2);
  col_txfm(buf0, buf0, bitcol, 1, 0);
  col_txfm_4x8_rounding(buf0, -1);

  int32x4_t buf1[8];
  transpose_arrays_s32_4x8(buf0, buf1);

  for (int i = 0; i < 2; i++) {
    row_txfm(buf1 + 4 * i, buf1 + 4 * i, bitrow);
  }
  av1_round_shift_rect_array_32_neon(buf1, buf1, 8, 0, NewSqrt2);
  store_buffer_4x8(buf1, coeff);
}

void av1_fwd_txfm2d_8x4_neon(const int16_t *input, int32_t *coeff, int stride,
                             TX_TYPE tx_type, int bd) {
  (void)bd;
  int bitcol = av1_fwd_cos_bit_col[1][0];
  int bitrow = av1_fwd_cos_bit_row[1][0];
  const fwd_transform_1d_neon col_txfm = col_highbd_txfm4x4_arr[tx_type];
  const fwd_transform_1d_many_neon row_txfm = row_highbd_txfm4x8_arr[tx_type];

  int ud_flip, lr_flip;
  get_flip_cfg(tx_type, &ud_flip, &lr_flip);
  ud_adjust_input_and_stride(ud_flip, &input, &stride, 4);

  // Column-wise transform.
  int32x4_t buf0[8];
  load_buffer_8x4(input, buf0, stride, lr_flip, 2);
  for (int i = 0; i < 2; i++) {
    col_txfm(buf0 + i * 4, buf0 + i * 4, bitcol);
  }
  col_txfm_4x8_rounding(buf0, -1);

  int32x4_t buf1[8];
  transpose_arrays_s32_8x4(buf0, buf1);

  // Row-wise transform.
  row_txfm(buf1, buf1, bitrow, 1, 0);
  av1_round_shift_rect_array_32_neon(buf1, buf1, 8, 0, NewSqrt2);
  store_buffer_8x4(buf1, coeff);
}

#if !CONFIG_REALTIME_ONLY
void av1_fwd_txfm2d_16x64_neon(const int16_t *input, int32_t *coeff, int stride,
                               TX_TYPE tx_type, int bd) {
  (void)bd;
  const int bitcol = av1_fwd_cos_bit_col[2][4];
  const int bitrow = av1_fwd_cos_bit_row[2][4];

  int ud_flip, lr_flip;
  get_flip_cfg(tx_type, &ud_flip, &lr_flip);
  ud_adjust_input_and_stride(ud_flip, &input, &stride, 64);

  // Column-wise transform.
  int32x4_t buf0[256];
  load_buffer_16x64(input, buf0, stride, lr_flip, 0);
  for (int i = 0; i < 4; i++) {
    av1_fdct64_new_neon(buf0 + i * 64, buf0 + i * 64, bitcol);
  }
  col_txfm_16x16_rounding(buf0, -2);
  col_txfm_16x16_rounding(buf0 + 64, -2);
  col_txfm_16x16_rounding(buf0 + 128, -2);
  col_txfm_16x16_rounding(buf0 + 192, -2);

  int32x4_t buf1[256];
  transpose_arrays_s32_16x64(buf0, buf1);

  // Row-wise transform.
  fdct16x16_neon(buf1, buf1, bitrow, 8, 16);
  store_buffer_16x32(buf1, coeff);
}

void av1_fwd_txfm2d_64x16_neon(const int16_t *input, int32_t *coeff, int stride,
                               TX_TYPE tx_type, int bd) {
  (void)bd;
  const int bitcol = av1_fwd_cos_bit_col[4][2];
  const int bitrow = av1_fwd_cos_bit_row[4][2];

  int ud_flip, lr_flip;
  get_flip_cfg(tx_type, &ud_flip, &lr_flip);
  ud_adjust_input_and_stride(ud_flip, &input, &stride, 16);

  // Column-wise transform.
  int32x4_t buf0[256];
  load_buffer_64x16(input, buf0, stride, lr_flip, 2);
  fdct16x16_neon(buf0, buf0, bitcol, 16, 16);
  col_txfm_16x16_rounding(buf0, -4);
  col_txfm_16x16_rounding(buf0 + 64, -4);
  col_txfm_16x16_rounding(buf0 + 128, -4);
  col_txfm_16x16_rounding(buf0 + 192, -4);

  int32x4_t buf1[256];
  transpose_arrays_s32_64x16(buf0, buf1);

  // Row-wise transform.
  for (int i = 0; i < 4; i++) {
    av1_fdct64_new_neon(buf1 + i * 64, buf1 + i * 64, bitrow);
  }
  store_buffer_64x16(buf1, coeff);
  memset(coeff + 16 * 32, 0, 16 * 32 * sizeof(*coeff));
}
#endif

static INLINE fwd_transform_1d_neon fwd_txfm_type_to_func(TXFM_TYPE txfm_type) {
  switch (txfm_type) {
    case TXFM_TYPE_DCT32: return av1_fdct32_new_neon;
    case TXFM_TYPE_IDENTITY32: return av1_idtx32_new_neon;
    default: assert(0);
  }
  return NULL;
}

static INLINE void av1_round_shift_array_32_neon(int32x4_t *input,
                                                 int32x4_t *output,
                                                 const int size,
                                                 const int bit) {
  const int32x4_t v_bit = vdupq_n_s32(-bit);
  for (int i = 0; i < size; i++) output[i] = vrshlq_s32(input[i], v_bit);
}

void av1_fwd_txfm2d_32x32_neon(const int16_t *input, int32_t *output,
                               int stride, TX_TYPE tx_type, int bd) {
  (void)bd;
  TXFM_2D_FLIP_CFG cfg;
  av1_get_fwd_txfm_cfg(tx_type, TX_32X32, &cfg);
  assert(cfg.tx_size < TX_SIZES);
  const fwd_transform_1d_neon col_txfm =
      fwd_txfm_type_to_func(cfg.txfm_type_col);
  const fwd_transform_1d_neon row_txfm =
      fwd_txfm_type_to_func(cfg.txfm_type_row);

  // Column-wise transform.
  int32x4_t buf0[256];
  load_buffer_32x32(input, buf0, stride, 0, 0);
  av1_round_shift_array_32_neon(buf0, buf0, 256, -2);
  for (int col = 0; col < 8; col++) {
    col_txfm(buf0 + col * 32, buf0 + col * 32, 12);
  }
  av1_round_shift_array_32_neon(buf0, buf0, 256, 4);

  int32x4_t buf1[256];
  transpose_arrays_s32_32x32(buf0, buf1);

  // Row-wise transform.
  for (int col = 0; col < 8; col++) {
    row_txfm(buf1 + col * 32, buf1 + col * 32, 12);
  }
  store_buffer_32x32(buf1, output);
}

void av1_fwd_txfm2d_64x64_neon(const int16_t *input, int32_t *output,
                               int stride, TX_TYPE tx_type, int bd) {
  (void)bd;
  (void)tx_type;

  // Column-wise transform.
  int32x4_t buf0[1024];
  load_buffer_64x64(input, buf0, stride, 0, 0);
  for (int col = 0; col < 16; col++) {
    av1_fdct64_new_neon(buf0 + col * 64, buf0 + col * 64, 13);
  }
  av1_round_shift_array_32_neon(buf0, buf0, 1024, 2);

  int32x4_t buf1[1024];
  transpose_arrays_s32_64x64(buf0, buf1);

  // Row-wise transform.
  for (int col = 0; col < 8; col++) {
    av1_fdct64_new_neon(buf1 + col * 64, buf1 + col * 64, 10);
  }
  av1_round_shift_array_32_neon(buf1, buf1, 512, 2);
  store_buffer_64x32(buf1, output);
}
