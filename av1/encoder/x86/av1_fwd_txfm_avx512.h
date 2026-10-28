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

#ifndef AOM_AV1_ENCODER_X86_AV1_FWD_TXFM_AVX512_H_
#define AOM_AV1_ENCODER_X86_AV1_FWD_TXFM_AVX512_H_

#if !defined(AVX512_EMU_H_) && defined(__GNUC__) && !defined(__clang__)
#pragma GCC target("avx2,avx512f,avx512cd,avx512bw,avx512dq,avx512vl")
#endif

#include <immintrin.h>

#include "aom/aom_integer.h"
#include "aom_dsp/x86/txfm_common_avx2.h"
#include "av1/common/av1_txfm.h"
#include "av1/encoder/av1_fwd_txfm1d_cfg.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline __m512i pair_set_w32_epi16(int16_t a, int16_t b) {
  return _mm512_set1_epi32(
      (int32_t)(((uint16_t)(a)) | (((uint32_t)(uint16_t)(b)) << 16)));
}

static inline void btf_16_w32_avx512(const __m512i w0, const __m512i w1,
                                     __m512i *in0, __m512i *in1,
                                     const __m512i _r, const int32_t cos_bit) {
  const __m512i t0 = _mm512_unpacklo_epi16(*in0, *in1);
  const __m512i t1 = _mm512_unpackhi_epi16(*in0, *in1);
  const __m512i u0 = _mm512_madd_epi16(t0, w0);
  const __m512i u1 = _mm512_madd_epi16(t1, w0);
  const __m512i v0 = _mm512_madd_epi16(t0, w1);
  const __m512i v1 = _mm512_madd_epi16(t1, w1);

  const __m512i a0 = _mm512_add_epi32(u0, _r);
  const __m512i a1 = _mm512_add_epi32(u1, _r);
  const __m512i b0 = _mm512_add_epi32(v0, _r);
  const __m512i b1 = _mm512_add_epi32(v1, _r);

  const __m512i c0 = _mm512_srai_epi32(a0, (unsigned int)cos_bit);
  const __m512i c1 = _mm512_srai_epi32(a1, (unsigned int)cos_bit);
  const __m512i d0 = _mm512_srai_epi32(b0, (unsigned int)cos_bit);
  const __m512i d1 = _mm512_srai_epi32(b1, (unsigned int)cos_bit);

  *in0 = _mm512_packs_epi32(c0, c1);
  *in1 = _mm512_packs_epi32(d0, d1);
}

static inline void btf_16_adds_subs_avx512(__m512i *in0, __m512i *in1) {
  const __m512i _in0 = *in0;
  const __m512i _in1 = *in1;
  *in0 = _mm512_adds_epi16(_in0, _in1);
  *in1 = _mm512_subs_epi16(_in0, _in1);
}

static inline void btf_32_add_sub_avx512(__m512i *in0, __m512i *in1) {
  const __m512i _in0 = *in0;
  const __m512i _in1 = *in1;
  *in0 = _mm512_add_epi32(_in0, _in1);
  *in1 = _mm512_sub_epi32(_in0, _in1);
}

static inline void btf_16_adds_subs_out_avx512(__m512i *out0, __m512i *out1,
                                               __m512i in0, __m512i in1) {
  *out0 = _mm512_adds_epi16(in0, in1);
  *out1 = _mm512_subs_epi16(in0, in1);
}

static inline void btf_32_add_sub_out_avx512(__m512i *out0, __m512i *out1,
                                             __m512i in0, __m512i in1) {
  *out0 = _mm512_add_epi32(in0, in1);
  *out1 = _mm512_sub_epi32(in0, in1);
}

// out0 = in0*w0 + in1*w1
// out1 = -in1*w0 + in0*w1
static inline void btf_32_avx512_type0(const int32_t w0, const int32_t w1,
                                       __m512i *in0, __m512i *in1,
                                       const __m512i _r,
                                       const int32_t cos_bit) {
  const __m512i _in0 = *in0;
  const __m512i _in1 = *in1;
  const __m512i ww0 = _mm512_set1_epi32(w0);
  const __m512i ww1 = _mm512_set1_epi32(w1);
  const __m512i in0_w0 = _mm512_mullo_epi32(_in0, ww0);
  const __m512i in1_w1 = _mm512_mullo_epi32(_in1, ww1);
  __m512i temp0 = _mm512_add_epi32(in0_w0, in1_w1);
  temp0 = _mm512_add_epi32(temp0, _r);
  *in0 = _mm512_srai_epi32(temp0, (unsigned int)cos_bit);
  const __m512i in0_w1 = _mm512_mullo_epi32(_in0, ww1);
  const __m512i in1_w0 = _mm512_mullo_epi32(_in1, ww0);
  __m512i temp1 = _mm512_sub_epi32(in0_w1, in1_w0);
  temp1 = _mm512_add_epi32(temp1, _r);
  *in1 = _mm512_srai_epi32(temp1, (unsigned int)cos_bit);
}

// out0 = in0*w0 + in1*w1
// out1 = in1*w0 - in0*w1
static inline void btf_32_avx512_type1(const int32_t w0, const int32_t w1,
                                       __m512i *in0, __m512i *in1,
                                       const __m512i _r,
                                       const int32_t cos_bit) {
  const __m512i _in0 = *in0;
  const __m512i _in1 = *in1;
  const __m512i ww0 = _mm512_set1_epi32(w0);
  const __m512i ww1 = _mm512_set1_epi32(w1);
  const __m512i in0_w0 = _mm512_mullo_epi32(_in0, ww0);
  const __m512i in1_w1 = _mm512_mullo_epi32(_in1, ww1);
  __m512i temp0 = _mm512_add_epi32(in0_w0, in1_w1);
  temp0 = _mm512_add_epi32(temp0, _r);
  *in0 = _mm512_srai_epi32(temp0, (unsigned int)cos_bit);
  const __m512i in0_w1 = _mm512_mullo_epi32(_in0, ww1);
  const __m512i in1_w0 = _mm512_mullo_epi32(_in1, ww0);
  __m512i temp1 = _mm512_sub_epi32(in1_w0, in0_w1);
  temp1 = _mm512_add_epi32(temp1, _r);
  *in1 = _mm512_srai_epi32(temp1, (unsigned int)cos_bit);
}

static inline void btf_32_avx512_type0_new(const __m512i ww0, const __m512i ww1,
                                           __m512i *in0, __m512i *in1,
                                           const __m512i _r,
                                           const int32_t cos_bit) {
  const __m512i _in0 = *in0;
  const __m512i _in1 = *in1;
  const __m512i in0_w0 = _mm512_mullo_epi32(_in0, ww0);
  const __m512i in1_w1 = _mm512_mullo_epi32(_in1, ww1);
  __m512i temp0 = _mm512_add_epi32(in0_w0, in1_w1);
  temp0 = _mm512_add_epi32(temp0, _r);
  *in0 = _mm512_srai_epi32(temp0, (unsigned int)cos_bit);
  const __m512i in0_w1 = _mm512_mullo_epi32(_in0, ww1);
  const __m512i in1_w0 = _mm512_mullo_epi32(_in1, ww0);
  __m512i temp1 = _mm512_sub_epi32(in0_w1, in1_w0);
  temp1 = _mm512_add_epi32(temp1, _r);
  *in1 = _mm512_srai_epi32(temp1, (unsigned int)cos_bit);
}

static inline void btf_32_avx512_type1_new(const __m512i ww0, const __m512i ww1,
                                           __m512i *in0, __m512i *in1,
                                           const __m512i _r,
                                           const int32_t cos_bit) {
  const __m512i _in0 = *in0;
  const __m512i _in1 = *in1;
  const __m512i in0_w0 = _mm512_mullo_epi32(_in0, ww0);
  const __m512i in1_w1 = _mm512_mullo_epi32(_in1, ww1);
  __m512i temp0 = _mm512_add_epi32(in0_w0, in1_w1);
  temp0 = _mm512_add_epi32(temp0, _r);
  *in0 = _mm512_srai_epi32(temp0, (unsigned int)cos_bit);
  const __m512i in0_w1 = _mm512_mullo_epi32(_in0, ww1);
  const __m512i in1_w0 = _mm512_mullo_epi32(_in1, ww0);
  __m512i temp1 = _mm512_sub_epi32(in1_w0, in0_w1);
  temp1 = _mm512_add_epi32(temp1, _r);
  *in1 = _mm512_srai_epi32(temp1, (unsigned int)cos_bit);
}

static inline __m512i scale_round_avx512(const __m512i a, const int scale) {
  const __m512i scale_rounding =
      pair_set_w32_epi16(scale, 1 << (NewSqrt2Bits - 1));
  const __m512i b = _mm512_madd_epi16(a, scale_rounding);
  return _mm512_srai_epi32(b, NewSqrt2Bits);
}

static inline void load_buffer_16bit_to_16bit_w32_avx512(const int16_t *in,
                                                         int stride,
                                                         __m512i *out,
                                                         int out_size) {
  for (int i = 0; i < out_size; ++i) {
    out[i] = _mm512_loadu_si512((const __m512i *)(in + i * stride));
  }
}

static inline void load_buffer_16bit_to_16bit_w32_flip_avx512(const int16_t *in,
                                                              int stride,
                                                              __m512i *out,
                                                              int out_size) {
  for (int i = 0; i < out_size; ++i) {
    out[out_size - i - 1] =
        _mm512_loadu_si512((const __m512i *)(in + i * stride));
  }
}

// Pack two 16-wide strips separated by `strip_stride` int16_t elements into
// a single 32-wide __m512i array.
static inline void load_two_buffers_16bit_to_16bit_w32_avx512(
    const int16_t *in0, const int16_t *in1, int stride, __m512i *out,
    int out_size) {
  for (int i = 0; i < out_size; ++i) {
    const __m256i v0 = _mm256_loadu_si256((const __m256i *)(in0 + i * stride));
    const __m256i v1 = _mm256_loadu_si256((const __m256i *)(in1 + i * stride));
    out[i] = _mm512_inserti64x4(_mm512_castsi256_si512(v0), v1, 1);
  }
}

static inline void load_two_buffers_16bit_to_16bit_w32_flip_avx512(
    const int16_t *in0, const int16_t *in1, int stride, __m512i *out,
    int out_size) {
  for (int i = 0; i < out_size; ++i) {
    const __m256i v0 = _mm256_loadu_si256((const __m256i *)(in0 + i * stride));
    const __m256i v1 = _mm256_loadu_si256((const __m256i *)(in1 + i * stride));
    out[out_size - i - 1] =
        _mm512_inserti64x4(_mm512_castsi256_si512(v0), v1, 1);
  }
}

static inline void round_shift_16bit_w32_avx512(__m512i *in, int size,
                                                int bit) {
  if (bit < 0) {
    bit = -bit;
    const __m512i round = _mm512_set1_epi16(1 << (bit - 1));
    for (int i = 0; i < size; ++i) {
      in[i] = _mm512_adds_epi16(in[i], round);
      in[i] = _mm512_srai_epi16(in[i], (unsigned int)bit);
    }
  } else if (bit > 0) {
    for (int i = 0; i < size; ++i) {
      in[i] = _mm512_slli_epi16(in[i], (unsigned int)bit);
    }
  }
}

static inline __m512i round_shift_32_avx512(__m512i vec, int bit) {
  const __m512i round = _mm512_set1_epi32(1 << (bit - 1));
  const __m512i tmp = _mm512_add_epi32(vec, round);
  return _mm512_srai_epi32(tmp, (unsigned int)bit);
}

static inline void round_shift_array_32_avx512(__m512i *input, __m512i *output,
                                               const int size, const int bit) {
  if (bit > 0) {
    for (int i = 0; i < size; i++) {
      output[i] = round_shift_32_avx512(input[i], bit);
    }
  } else {
    for (int i = 0; i < size; i++) {
      output[i] = _mm512_slli_epi32(input[i], (unsigned int)(-bit));
    }
  }
}

static inline void round_shift_rect_array_32_avx512(__m512i *input,
                                                    __m512i *output,
                                                    const int size,
                                                    const int bit,
                                                    const int val) {
  const __m512i sqrt2 = _mm512_set1_epi32(val);
  if (bit > 0) {
    for (int i = 0; i < size; i++) {
      const __m512i r0 = round_shift_32_avx512(input[i], bit);
      const __m512i r1 = _mm512_mullo_epi32(sqrt2, r0);
      output[i] = round_shift_32_avx512(r1, NewSqrt2Bits);
    }
  } else {
    for (int i = 0; i < size; i++) {
      const __m512i r0 = _mm512_slli_epi32(input[i], (unsigned int)(-bit));
      const __m512i r1 = _mm512_mullo_epi32(sqrt2, r0);
      output[i] = round_shift_32_avx512(r1, NewSqrt2Bits);
    }
  }
}

static inline void store_output_32bit_w16_avx512(int32_t *const out,
                                                 const __m512i *const in,
                                                 const int stride,
                                                 const int out_size) {
  for (int i = 0; i < out_size; ++i) {
    _mm512_storeu_si512((__m512i *)(out + stride * i),
                        _mm512_loadu_si512((const __m512i *)&in[i]));
  }
}

static inline void store_buffer_16bit_to_32bit_w16_avx512(
    const __m256i *const in, int32_t *out, const int stride,
    const int out_size) {
  for (int i = 0; i < out_size; ++i) {
    _mm512_storeu_si512((__m512i *)out, _mm512_cvtepi16_epi32(in[i]));
    out += stride;
  }
}

// Store 32-wide __m512i (two 16-wide strips in low 256-bit and high 256-bit)
// into `out0` and `out1`.
static inline void store_two_buffers_16bit_to_32bit_w16_avx512(
    const __m512i *const in, int32_t *out0, int32_t *out1, const int stride,
    const int out_size) {
  for (int i = 0; i < out_size; ++i) {
    const __m256i lo = _mm512_castsi512_si256(in[i]);
    const __m256i hi = _mm512_extracti64x4_epi64(in[i], 1);
    _mm512_storeu_si512((__m512i *)(out0 + i * stride),
                        _mm512_cvtepi16_epi32(lo));
    _mm512_storeu_si512((__m512i *)(out1 + i * stride),
                        _mm512_cvtepi16_epi32(hi));
  }
}

static inline void store_rect_16bit_to_32bit_w16_avx512(const __m256i a,
                                                        int32_t *const b) {
  const __m512i a_32 = _mm512_cvtepi16_epi32(a);
  const __m512i sqrt2 = _mm512_set1_epi32(NewSqrt2);
  const __m512i prod = _mm512_mullo_epi32(a_32, sqrt2);
  const __m512i res = round_shift_32_avx512(prod, NewSqrt2Bits);
  _mm512_storeu_si512((__m512i *)b, res);
}

static inline void store_rect_buffer_16bit_to_32bit_w16_avx512(
    const __m256i *const in, int32_t *const out, const int stride,
    const int out_size) {
  for (int i = 0; i < out_size; ++i) {
    store_rect_16bit_to_32bit_w16_avx512(in[i], out + i * stride);
  }
}

static inline void store_two_rect_buffers_16bit_to_32bit_w16_avx512(
    const __m512i *const in, int32_t *const out0, int32_t *const out1,
    const int stride, const int out_size) {
  const __m512i sqrt2 = _mm512_set1_epi32(NewSqrt2);
  for (int i = 0; i < out_size; ++i) {
    const __m256i lo = _mm512_castsi512_si256(in[i]);
    const __m256i hi = _mm512_extracti64x4_epi64(in[i], 1);
    const __m512i lo_32 = _mm512_cvtepi16_epi32(lo);
    const __m512i hi_32 = _mm512_cvtepi16_epi32(hi);
    const __m512i res0 =
        round_shift_32_avx512(_mm512_mullo_epi32(lo_32, sqrt2), NewSqrt2Bits);
    const __m512i res1 =
        round_shift_32_avx512(_mm512_mullo_epi32(hi_32, sqrt2), NewSqrt2Bits);
    _mm512_storeu_si512((__m512i *)(out0 + i * stride), res0);
    _mm512_storeu_si512((__m512i *)(out1 + i * stride), res1);
  }
}

// Transpose two 16x16 blocks held in the low 256-bit and high 256-bit halves
// of `in[0..15]` into `out0[0..15]` and `out1[0..15]` (each 256-bit).
static inline void transpose_two_16bit_16x16_avx512(const __m512i *const in,
                                                    __m256i *const out0,
                                                    __m256i *const out1) {
  __m256i in_lo[16], in_hi[16];
  for (int i = 0; i < 16; ++i) {
    in_lo[i] = _mm512_castsi512_si256(in[i]);
    in_hi[i] = _mm512_extracti64x4_epi64(in[i], 1);
  }
  transpose_16bit_16x16_avx2(in_lo, out0);
  transpose_16bit_16x16_avx2(in_hi, out1);
}

#ifdef __cplusplus
}
#endif

#endif  // AOM_AV1_ENCODER_X86_AV1_FWD_TXFM_AVX512_H_
