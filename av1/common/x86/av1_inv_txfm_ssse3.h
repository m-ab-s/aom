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
#ifndef AOM_AV1_COMMON_X86_AV1_INV_TXFM_SSSE3_H_
#define AOM_AV1_COMMON_X86_AV1_INV_TXFM_SSSE3_H_

#include <emmintrin.h>  // SSE2
#include <tmmintrin.h>  // SSSE3

#include "config/aom_config.h"
#include "config/av1_rtcd.h"

#include "aom/aom_integer.h"
#include "aom_dsp/x86/transpose_sse2.h"
#include "av1/common/av1_inv_txfm1d_cfg.h"
#include "av1/common/x86/av1_txfm_sse2.h"

#ifdef __cplusplus
extern "C" {
#endif

#define btf_16_ssse3(w0, w1, in, out0, out1)    \
  do {                                          \
    const __m128i _w0 = _mm_set1_epi16(w0 * 8); \
    const __m128i _w1 = _mm_set1_epi16(w1 * 8); \
    const __m128i _in = in;                     \
    out0 = _mm_mulhrs_epi16(_in, _w0);          \
    out1 = _mm_mulhrs_epi16(_in, _w1);          \
  } while (0)

#define btf_16_adds_subs_sse2(in0, in1) \
  do {                                  \
    const __m128i _in0 = in0;           \
    const __m128i _in1 = in1;           \
    in0 = _mm_adds_epi16(_in0, _in1);   \
    in1 = _mm_subs_epi16(_in0, _in1);   \
  } while (0)

#define btf_16_subs_adds_sse2(in0, in1) \
  do {                                  \
    const __m128i _in0 = in0;           \
    const __m128i _in1 = in1;           \
    in1 = _mm_subs_epi16(_in0, _in1);   \
    in0 = _mm_adds_epi16(_in0, _in1);   \
  } while (0)

#define btf_16_adds_subs_out_sse2(out0, out1, in0, in1) \
  do {                                                  \
    const __m128i _in0 = in0;                           \
    const __m128i _in1 = in1;                           \
    out0 = _mm_adds_epi16(_in0, _in1);                  \
    out1 = _mm_subs_epi16(_in0, _in1);                  \
  } while (0)

static inline void round_shift_16bit_ssse3(__m128i *in, int size, int bit) {
  if (bit < 0) {
    const __m128i scale = _mm_set1_epi16(1 << (15 + bit));
    for (int i = 0; i < size; ++i) {
      in[i] = _mm_mulhrs_epi16(in[i], scale);
    }
  } else if (bit > 0) {
    for (int i = 0; i < size; ++i) {
      in[i] = _mm_slli_epi16(in[i], bit);
    }
  }
}

// 1D itx types
enum {
  IDCT_1D,
  IADST_1D,
  IFLIPADST_1D = IADST_1D,
  IIDENTITY_1D,
  ITX_TYPES_1D,
} UENUM1BYTE(ITX_TYPE_1D);

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

DECLARE_ALIGNED(16, static const int16_t, av1_eob_to_eobxy_8x8_default[8]) = {
  0x0707, 0x0707, 0x0707, 0x0707, 0x0707, 0x0707, 0x0707, 0x0707,
};

DECLARE_ALIGNED(16, static const int16_t,
                av1_eob_to_eobxy_16x16_default[16]) = {
  0x0707, 0x0707, 0x0f0f, 0x0f0f, 0x0f0f, 0x0f0f, 0x0f0f, 0x0f0f,
  0x0f0f, 0x0f0f, 0x0f0f, 0x0f0f, 0x0f0f, 0x0f0f, 0x0f0f, 0x0f0f,
};

DECLARE_ALIGNED(16, static const int16_t,
                av1_eob_to_eobxy_32x32_default[32]) = {
  0x0707, 0x0f0f, 0x0f0f, 0x0f0f, 0x1f1f, 0x1f1f, 0x1f1f, 0x1f1f,
  0x1f1f, 0x1f1f, 0x1f1f, 0x1f1f, 0x1f1f, 0x1f1f, 0x1f1f, 0x1f1f,
  0x1f1f, 0x1f1f, 0x1f1f, 0x1f1f, 0x1f1f, 0x1f1f, 0x1f1f, 0x1f1f,
  0x1f1f, 0x1f1f, 0x1f1f, 0x1f1f, 0x1f1f, 0x1f1f, 0x1f1f, 0x1f1f,
};

DECLARE_ALIGNED(16, static const int16_t, av1_eob_to_eobxy_8x16_default[16]) = {
  0x0707, 0x0707, 0x0707, 0x0707, 0x0707, 0x0f07, 0x0f07, 0x0f07,
  0x0f07, 0x0f07, 0x0f07, 0x0f07, 0x0f07, 0x0f07, 0x0f07, 0x0f07,
};

DECLARE_ALIGNED(16, static const int16_t, av1_eob_to_eobxy_16x8_default[8]) = {
  0x0707, 0x0707, 0x070f, 0x070f, 0x070f, 0x070f, 0x070f, 0x070f,
};

DECLARE_ALIGNED(16, static const int16_t,
                av1_eob_to_eobxy_16x32_default[32]) = {
  0x0707, 0x0707, 0x0f0f, 0x0f0f, 0x0f0f, 0x0f0f, 0x0f0f, 0x0f0f,
  0x0f0f, 0x1f0f, 0x1f0f, 0x1f0f, 0x1f0f, 0x1f0f, 0x1f0f, 0x1f0f,
  0x1f0f, 0x1f0f, 0x1f0f, 0x1f0f, 0x1f0f, 0x1f0f, 0x1f0f, 0x1f0f,
  0x1f0f, 0x1f0f, 0x1f0f, 0x1f0f, 0x1f0f, 0x1f0f, 0x1f0f, 0x1f0f,
};

DECLARE_ALIGNED(16, static const int16_t,
                av1_eob_to_eobxy_32x16_default[16]) = {
  0x0707, 0x0f0f, 0x0f0f, 0x0f0f, 0x0f1f, 0x0f1f, 0x0f1f, 0x0f1f,
  0x0f1f, 0x0f1f, 0x0f1f, 0x0f1f, 0x0f1f, 0x0f1f, 0x0f1f, 0x0f1f,
};

DECLARE_ALIGNED(16, static const int16_t, av1_eob_to_eobxy_8x32_default[32]) = {
  0x0707, 0x0707, 0x0707, 0x0707, 0x0707, 0x0f07, 0x0f07, 0x0f07,
  0x0f07, 0x0f07, 0x0f07, 0x0f07, 0x0f07, 0x1f07, 0x1f07, 0x1f07,
  0x1f07, 0x1f07, 0x1f07, 0x1f07, 0x1f07, 0x1f07, 0x1f07, 0x1f07,
  0x1f07, 0x1f07, 0x1f07, 0x1f07, 0x1f07, 0x1f07, 0x1f07, 0x1f07,
};

DECLARE_ALIGNED(16, static const int16_t, av1_eob_to_eobxy_32x8_default[8]) = {
  0x0707, 0x070f, 0x070f, 0x071f, 0x071f, 0x071f, 0x071f, 0x071f,
};

DECLARE_ALIGNED(16, static const int16_t *,
                av1_eob_to_eobxy_default[TX_SIZES_ALL]) = {
  NULL,
  av1_eob_to_eobxy_8x8_default,
  av1_eob_to_eobxy_16x16_default,
  av1_eob_to_eobxy_32x32_default,
  av1_eob_to_eobxy_32x32_default,
  NULL,
  NULL,
  av1_eob_to_eobxy_8x16_default,
  av1_eob_to_eobxy_16x8_default,
  av1_eob_to_eobxy_16x32_default,
  av1_eob_to_eobxy_32x16_default,
  av1_eob_to_eobxy_32x32_default,
  av1_eob_to_eobxy_32x32_default,
  NULL,
  NULL,
  av1_eob_to_eobxy_8x32_default,
  av1_eob_to_eobxy_32x8_default,
  av1_eob_to_eobxy_16x32_default,
  av1_eob_to_eobxy_32x16_default,
};

static const int lowbd_txfm_all_1d_zeros_idx[32] = {
  0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
};

// Transform block width in log2 for eob (size of 64 map to 32)
static const int tx_size_wide_log2_eob[TX_SIZES_ALL] = {
  2, 3, 4, 5, 5, 2, 3, 3, 4, 4, 5, 5, 5, 2, 4, 3, 5, 4, 5,
};

static inline void get_eobx_eoby_scan_default(int *eobx, int *eoby,
                                              TX_SIZE tx_size, int eob) {
  if (eob == 1) {
    *eobx = 0;
    *eoby = 0;
    return;
  }

  const int tx_w_log2 = tx_size_wide_log2_eob[tx_size];
  const int eob_row = (eob - 1) >> tx_w_log2;
  const int eobxy = av1_eob_to_eobxy_default[tx_size][eob_row];
  *eobx = eobxy & 0xFF;
  *eoby = eobxy >> 8;
}

static const int eob_fill[32] = {
  0,  7,  7,  7,  7,  7,  7,  7,  15, 15, 15, 15, 15, 15, 15, 15,
  31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31,
};

static inline void get_eobx_eoby_scan_h_identity(int *eobx, int *eoby,
                                                 TX_SIZE tx_size, int eob) {
  eob -= 1;
  const int txfm_size_col = tx_size_wide[tx_size];
  const int eobx_max = AOMMIN(32, txfm_size_col) - 1;
  *eobx = (eob >= eobx_max) ? eobx_max : eob_fill[eob];
  const int temp_eoby = eob / (eobx_max + 1);
  assert(temp_eoby < 32);
  *eoby = eob_fill[temp_eoby];
}

static inline void get_eobx_eoby_scan_v_identity(int *eobx, int *eoby,
                                                 TX_SIZE tx_size, int eob) {
  eob -= 1;
  const int txfm_size_row = tx_size_high[tx_size];
  const int eoby_max = AOMMIN(32, txfm_size_row) - 1;
  *eobx = eob_fill[eob / (eoby_max + 1)];
  *eoby = (eob >= eoby_max) ? eoby_max : eob_fill[eob];
}

typedef void (*transform_1d_ssse3)(const __m128i *input, __m128i *output);

void av1_lowbd_inv_txfm2d_add_idtx_ssse3(const int32_t *input, uint8_t *output,
                                         int stride, TX_SIZE tx_size);

void av1_lowbd_inv_txfm2d_add_h_identity_ssse3(const int32_t *input,
                                               uint8_t *output, int stride,
                                               TX_TYPE tx_type, TX_SIZE tx_size,
                                               int eob);
void av1_lowbd_inv_txfm2d_add_v_identity_ssse3(const int32_t *input,
                                               uint8_t *output, int stride,
                                               TX_TYPE tx_type, TX_SIZE tx_size,
                                               int eob);

static inline void idct8_low1_ssse3_inline(const __m128i *input,
                                           __m128i *output) {
  const int32_t *cospi = cospi_arr(INV_COS_BIT);

  __m128i x[2];
  x[0] = input[0];

  btf_16_ssse3(cospi[32], cospi[32], x[0], x[0], x[1]);

  output[0] = x[0];
  output[7] = x[0];
  output[1] = x[1];
  output[6] = x[1];
  output[2] = x[1];
  output[5] = x[1];
  output[3] = x[0];
  output[4] = x[0];
}

static inline void idct8_sse2_inline(const __m128i *input, __m128i *output) {
  const int8_t cos_bit = INV_COS_BIT;
  const int32_t *cospi = cospi_arr(INV_COS_BIT);
  const __m128i __rounding = _mm_set1_epi32(1 << (INV_COS_BIT - 1));

  const __m128i cospi_p56_m08 = pair_set_epi16(cospi[56], -cospi[8]);
  const __m128i cospi_p08_p56 = pair_set_epi16(cospi[8], cospi[56]);
  const __m128i cospi_p24_m40 = pair_set_epi16(cospi[24], -cospi[40]);
  const __m128i cospi_p40_p24 = pair_set_epi16(cospi[40], cospi[24]);
  const __m128i cospi_p32_p32 = pair_set_epi16(cospi[32], cospi[32]);
  const __m128i cospi_p32_m32 = pair_set_epi16(cospi[32], -cospi[32]);
  const __m128i cospi_p48_m16 = pair_set_epi16(cospi[48], -cospi[16]);
  const __m128i cospi_p16_p48 = pair_set_epi16(cospi[16], cospi[48]);
  const __m128i cospi_m32_p32 = pair_set_epi16(-cospi[32], cospi[32]);

  // stage 1
  __m128i x[8];
  x[0] = input[0];
  x[1] = input[4];
  x[2] = input[2];
  x[3] = input[6];
  x[4] = input[1];
  x[5] = input[5];
  x[6] = input[3];
  x[7] = input[7];

  // stage 2
  btf_16_sse2(cospi_p56_m08, cospi_p08_p56, x[4], x[7], x[4], x[7]);
  btf_16_sse2(cospi_p24_m40, cospi_p40_p24, x[5], x[6], x[5], x[6]);

  // stage 3
  btf_16_sse2(cospi_p32_p32, cospi_p32_m32, x[0], x[1], x[0], x[1]);
  btf_16_sse2(cospi_p48_m16, cospi_p16_p48, x[2], x[3], x[2], x[3]);
  btf_16_adds_subs_sse2(x[4], x[5]);
  btf_16_subs_adds_sse2(x[7], x[6]);

  // stage 4
  btf_16_adds_subs_sse2(x[0], x[3]);
  btf_16_adds_subs_sse2(x[1], x[2]);
  btf_16_sse2(cospi_m32_p32, cospi_p32_p32, x[5], x[6], x[5], x[6]);

  // stage 5
  btf_16_adds_subs_out_sse2(output[0], output[7], x[0], x[7]);
  btf_16_adds_subs_out_sse2(output[1], output[6], x[1], x[6]);
  btf_16_adds_subs_out_sse2(output[2], output[5], x[2], x[5]);
  btf_16_adds_subs_out_sse2(output[3], output[4], x[3], x[4]);
}

static inline void iadst8_low1_ssse3_inline(const __m128i *input,
                                            __m128i *output) {
  const int8_t cos_bit = INV_COS_BIT;
  const int32_t *cospi = cospi_arr(INV_COS_BIT);
  const __m128i __zero = _mm_setzero_si128();
  const __m128i __rounding = _mm_set1_epi32(1 << (INV_COS_BIT - 1));

  const __m128i cospi_p16_p48 = pair_set_epi16(cospi[16], cospi[48]);
  const __m128i cospi_p48_m16 = pair_set_epi16(cospi[48], -cospi[16]);
  const __m128i cospi_p32_p32 = pair_set_epi16(cospi[32], cospi[32]);
  const __m128i cospi_p32_m32 = pair_set_epi16(cospi[32], -cospi[32]);

  __m128i x[8];
  x[1] = input[0];

  btf_16_ssse3(cospi[60], -cospi[4], x[1], x[0], x[1]);

  x[4] = x[0];
  x[5] = x[1];

  btf_16_sse2(cospi_p16_p48, cospi_p48_m16, x[4], x[5], x[4], x[5]);

  x[2] = x[0];
  x[3] = x[1];
  x[6] = x[4];
  x[7] = x[5];

  btf_16_sse2(cospi_p32_p32, cospi_p32_m32, x[2], x[3], x[2], x[3]);
  btf_16_sse2(cospi_p32_p32, cospi_p32_m32, x[6], x[7], x[6], x[7]);

  output[0] = x[0];
  output[1] = _mm_subs_epi16(__zero, x[4]);
  output[2] = x[6];
  output[3] = _mm_subs_epi16(__zero, x[2]);
  output[4] = x[3];
  output[5] = _mm_subs_epi16(__zero, x[7]);
  output[6] = x[5];
  output[7] = _mm_subs_epi16(__zero, x[1]);
}

static inline void iadst8_sse2_inline(const __m128i *input, __m128i *output) {
  const int8_t cos_bit = INV_COS_BIT;
  const int32_t *cospi = cospi_arr(INV_COS_BIT);
  const __m128i __zero = _mm_setzero_si128();
  const __m128i __rounding = _mm_set1_epi32(1 << (INV_COS_BIT - 1));

  const __m128i cospi_p04_p60 = pair_set_epi16(cospi[4], cospi[60]);
  const __m128i cospi_p60_m04 = pair_set_epi16(cospi[60], -cospi[4]);
  const __m128i cospi_p20_p44 = pair_set_epi16(cospi[20], cospi[44]);
  const __m128i cospi_p44_m20 = pair_set_epi16(cospi[44], -cospi[20]);
  const __m128i cospi_p36_p28 = pair_set_epi16(cospi[36], cospi[28]);
  const __m128i cospi_p28_m36 = pair_set_epi16(cospi[28], -cospi[36]);
  const __m128i cospi_p52_p12 = pair_set_epi16(cospi[52], cospi[12]);
  const __m128i cospi_p12_m52 = pair_set_epi16(cospi[12], -cospi[52]);
  const __m128i cospi_p16_p48 = pair_set_epi16(cospi[16], cospi[48]);
  const __m128i cospi_p48_m16 = pair_set_epi16(cospi[48], -cospi[16]);
  const __m128i cospi_m48_p16 = pair_set_epi16(-cospi[48], cospi[16]);
  const __m128i cospi_p32_p32 = pair_set_epi16(cospi[32], cospi[32]);
  const __m128i cospi_p32_m32 = pair_set_epi16(cospi[32], -cospi[32]);

  // stage 1
  __m128i x[8];
  x[0] = input[7];
  x[1] = input[0];
  x[2] = input[5];
  x[3] = input[2];
  x[4] = input[3];
  x[5] = input[4];
  x[6] = input[1];
  x[7] = input[6];

  // stage 2
  btf_16_sse2(cospi_p04_p60, cospi_p60_m04, x[0], x[1], x[0], x[1]);
  btf_16_sse2(cospi_p20_p44, cospi_p44_m20, x[2], x[3], x[2], x[3]);
  btf_16_sse2(cospi_p36_p28, cospi_p28_m36, x[4], x[5], x[4], x[5]);
  btf_16_sse2(cospi_p52_p12, cospi_p12_m52, x[6], x[7], x[6], x[7]);

  // stage 3
  btf_16_adds_subs_sse2(x[0], x[4]);
  btf_16_adds_subs_sse2(x[1], x[5]);
  btf_16_adds_subs_sse2(x[2], x[6]);
  btf_16_adds_subs_sse2(x[3], x[7]);

  // stage 4
  btf_16_sse2(cospi_p16_p48, cospi_p48_m16, x[4], x[5], x[4], x[5]);
  btf_16_sse2(cospi_m48_p16, cospi_p16_p48, x[6], x[7], x[6], x[7]);

  // stage 5
  btf_16_adds_subs_sse2(x[0], x[2]);
  btf_16_adds_subs_sse2(x[1], x[3]);
  btf_16_adds_subs_sse2(x[4], x[6]);
  btf_16_adds_subs_sse2(x[5], x[7]);

  // stage 6
  btf_16_sse2(cospi_p32_p32, cospi_p32_m32, x[2], x[3], x[2], x[3]);
  btf_16_sse2(cospi_p32_p32, cospi_p32_m32, x[6], x[7], x[6], x[7]);

  // stage 7
  output[0] = x[0];
  output[1] = _mm_subs_epi16(__zero, x[4]);
  output[2] = x[6];
  output[3] = _mm_subs_epi16(__zero, x[2]);
  output[4] = x[3];
  output[5] = _mm_subs_epi16(__zero, x[7]);
  output[6] = x[5];
  output[7] = _mm_subs_epi16(__zero, x[1]);
}

void av1_iadst8_low1_ssse3(const __m128i *input, __m128i *output);

void av1_idct8_low1_ssse3(const __m128i *input, __m128i *output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_X86_AV1_INV_TXFM_SSSE3_H_
