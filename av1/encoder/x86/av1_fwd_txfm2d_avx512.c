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

#include "config/aom_config.h"
#include "config/av1_rtcd.h"

#include "aom_dsp/x86/txfm_common_avx2.h"
#include "av1/common/av1_txfm.h"
#include "av1/common/enums.h"
#include "av1/common/x86/av1_txfm_sse2.h"
#include "av1/encoder/av1_fwd_txfm1d_cfg.h"
#include "av1/encoder/x86/av1_fwd_txfm_avx2.h"
#include "av1/encoder/x86/av1_fwd_txfm_avx512.h"
#include "av1/encoder/x86/av1_fwd_txfm_sse2.h"
#include "av1/encoder/x86/av1_txfm1d_sse4.h"

#if !CONFIG_HIGHWAY

static inline void fdct32x16_new_avx512(const __m512i *input, __m512i *output,
                                        int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);
  const __m512i _r = _mm512_set1_epi32(1 << (cos_bit - 1));

  const __m512i cospi_m32_p32 = pair_set_w32_epi16(-cospi[32], cospi[32]);
  const __m512i cospi_p32_p32 = pair_set_w32_epi16(cospi[32], cospi[32]);
  const __m512i cospi_p32_m32 = pair_set_w32_epi16(cospi[32], -cospi[32]);
  const __m512i cospi_p48_p16 = pair_set_w32_epi16(cospi[48], cospi[16]);
  const __m512i cospi_m16_p48 = pair_set_w32_epi16(-cospi[16], cospi[48]);
  const __m512i cospi_m48_m16 = pair_set_w32_epi16(-cospi[48], -cospi[16]);
  const __m512i cospi_p56_p08 = pair_set_w32_epi16(cospi[56], cospi[8]);
  const __m512i cospi_m08_p56 = pair_set_w32_epi16(-cospi[8], cospi[56]);
  const __m512i cospi_p24_p40 = pair_set_w32_epi16(cospi[24], cospi[40]);
  const __m512i cospi_m40_p24 = pair_set_w32_epi16(-cospi[40], cospi[24]);
  const __m512i cospi_p60_p04 = pair_set_w32_epi16(cospi[60], cospi[4]);
  const __m512i cospi_m04_p60 = pair_set_w32_epi16(-cospi[4], cospi[60]);
  const __m512i cospi_p28_p36 = pair_set_w32_epi16(cospi[28], cospi[36]);
  const __m512i cospi_m36_p28 = pair_set_w32_epi16(-cospi[36], cospi[28]);
  const __m512i cospi_p44_p20 = pair_set_w32_epi16(cospi[44], cospi[20]);
  const __m512i cospi_m20_p44 = pair_set_w32_epi16(-cospi[20], cospi[44]);
  const __m512i cospi_p12_p52 = pair_set_w32_epi16(cospi[12], cospi[52]);
  const __m512i cospi_m52_p12 = pair_set_w32_epi16(-cospi[52], cospi[12]);

  __m512i x1[16];
  for (int i = 0; i < 8; ++i) {
    btf_16_adds_subs_out_avx512(&x1[i], &x1[15 - i], input[i], input[15 - i]);
  }

  for (int i = 0; i < 4; ++i) btf_16_adds_subs_avx512(&x1[i], &x1[7 - i]);
  btf_16_w32_avx512(cospi_m32_p32, cospi_p32_p32, &x1[10], &x1[13], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_m32_p32, cospi_p32_p32, &x1[11], &x1[12], _r,
                    cos_bit);

  btf_16_adds_subs_avx512(&x1[0], &x1[3]);
  btf_16_adds_subs_avx512(&x1[1], &x1[2]);
  btf_16_w32_avx512(cospi_m32_p32, cospi_p32_p32, &x1[5], &x1[6], _r, cos_bit);
  btf_16_adds_subs_avx512(&x1[8], &x1[11]);
  btf_16_adds_subs_avx512(&x1[9], &x1[10]);
  btf_16_adds_subs_avx512(&x1[15], &x1[12]);
  btf_16_adds_subs_avx512(&x1[14], &x1[13]);

  btf_16_w32_avx512(cospi_p32_p32, cospi_p32_m32, &x1[0], &x1[1], _r, cos_bit);
  btf_16_w32_avx512(cospi_p48_p16, cospi_m16_p48, &x1[2], &x1[3], _r, cos_bit);
  btf_16_adds_subs_avx512(&x1[4], &x1[5]);
  btf_16_adds_subs_avx512(&x1[7], &x1[6]);
  btf_16_w32_avx512(cospi_m16_p48, cospi_p48_p16, &x1[9], &x1[14], _r, cos_bit);
  btf_16_w32_avx512(cospi_m48_m16, cospi_m16_p48, &x1[10], &x1[13], _r,
                    cos_bit);

  btf_16_w32_avx512(cospi_p56_p08, cospi_m08_p56, &x1[4], &x1[7], _r, cos_bit);
  btf_16_w32_avx512(cospi_p24_p40, cospi_m40_p24, &x1[5], &x1[6], _r, cos_bit);
  btf_16_adds_subs_avx512(&x1[8], &x1[9]);
  btf_16_adds_subs_avx512(&x1[11], &x1[10]);
  btf_16_adds_subs_avx512(&x1[12], &x1[13]);
  btf_16_adds_subs_avx512(&x1[15], &x1[14]);

  btf_16_w32_avx512(cospi_p60_p04, cospi_m04_p60, &x1[8], &x1[15], _r, cos_bit);
  btf_16_w32_avx512(cospi_p28_p36, cospi_m36_p28, &x1[9], &x1[14], _r, cos_bit);
  btf_16_w32_avx512(cospi_p44_p20, cospi_m20_p44, &x1[10], &x1[13], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p12_p52, cospi_m52_p12, &x1[11], &x1[12], _r,
                    cos_bit);

  output[0] = x1[0];
  output[1] = x1[8];
  output[2] = x1[4];
  output[3] = x1[12];
  output[4] = x1[2];
  output[5] = x1[10];
  output[6] = x1[6];
  output[7] = x1[14];
  output[8] = x1[1];
  output[9] = x1[9];
  output[10] = x1[5];
  output[11] = x1[13];
  output[12] = x1[3];
  output[13] = x1[11];
  output[14] = x1[7];
  output[15] = x1[15];
}

static inline void fadst32x16_new_avx512(const __m512i *input, __m512i *output,
                                         int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);
  const __m512i __zero = _mm512_setzero_si512();
  const __m512i _r = _mm512_set1_epi32(1 << (cos_bit - 1));

  const __m512i cospi_p32_p32 = pair_set_w32_epi16(cospi[32], cospi[32]);
  const __m512i cospi_p32_m32 = pair_set_w32_epi16(cospi[32], -cospi[32]);
  const __m512i cospi_p16_p48 = pair_set_w32_epi16(cospi[16], cospi[48]);
  const __m512i cospi_p48_m16 = pair_set_w32_epi16(cospi[48], -cospi[16]);
  const __m512i cospi_m48_p16 = pair_set_w32_epi16(-cospi[48], cospi[16]);
  const __m512i cospi_p08_p56 = pair_set_w32_epi16(cospi[8], cospi[56]);
  const __m512i cospi_p56_m08 = pair_set_w32_epi16(cospi[56], -cospi[8]);
  const __m512i cospi_p40_p24 = pair_set_w32_epi16(cospi[40], cospi[24]);
  const __m512i cospi_p24_m40 = pair_set_w32_epi16(cospi[24], -cospi[40]);
  const __m512i cospi_m56_p08 = pair_set_w32_epi16(-cospi[56], cospi[8]);
  const __m512i cospi_m24_p40 = pair_set_w32_epi16(-cospi[24], cospi[40]);
  const __m512i cospi_p02_p62 = pair_set_w32_epi16(cospi[2], cospi[62]);
  const __m512i cospi_p62_m02 = pair_set_w32_epi16(cospi[62], -cospi[2]);
  const __m512i cospi_p10_p54 = pair_set_w32_epi16(cospi[10], cospi[54]);
  const __m512i cospi_p54_m10 = pair_set_w32_epi16(cospi[54], -cospi[10]);
  const __m512i cospi_p18_p46 = pair_set_w32_epi16(cospi[18], cospi[46]);
  const __m512i cospi_p46_m18 = pair_set_w32_epi16(cospi[46], -cospi[18]);
  const __m512i cospi_p26_p38 = pair_set_w32_epi16(cospi[26], cospi[38]);
  const __m512i cospi_p38_m26 = pair_set_w32_epi16(cospi[38], -cospi[26]);
  const __m512i cospi_p34_p30 = pair_set_w32_epi16(cospi[34], cospi[30]);
  const __m512i cospi_p30_m34 = pair_set_w32_epi16(cospi[30], -cospi[34]);
  const __m512i cospi_p42_p22 = pair_set_w32_epi16(cospi[42], cospi[22]);
  const __m512i cospi_p22_m42 = pair_set_w32_epi16(cospi[22], -cospi[42]);
  const __m512i cospi_p50_p14 = pair_set_w32_epi16(cospi[50], cospi[14]);
  const __m512i cospi_p14_m50 = pair_set_w32_epi16(cospi[14], -cospi[50]);
  const __m512i cospi_p58_p06 = pair_set_w32_epi16(cospi[58], cospi[6]);
  const __m512i cospi_p06_m58 = pair_set_w32_epi16(cospi[6], -cospi[58]);

  __m512i x1[16];
  x1[0] = input[0];
  x1[1] = _mm512_subs_epi16(__zero, input[15]);
  x1[2] = _mm512_subs_epi16(__zero, input[7]);
  x1[3] = input[8];
  x1[4] = _mm512_subs_epi16(__zero, input[3]);
  x1[5] = input[12];
  x1[6] = input[4];
  x1[7] = _mm512_subs_epi16(__zero, input[11]);
  x1[8] = _mm512_subs_epi16(__zero, input[1]);
  x1[9] = input[14];
  x1[10] = input[6];
  x1[11] = _mm512_subs_epi16(__zero, input[9]);
  x1[12] = input[2];
  x1[13] = _mm512_subs_epi16(__zero, input[13]);
  x1[14] = _mm512_subs_epi16(__zero, input[5]);
  x1[15] = input[10];

  btf_16_w32_avx512(cospi_p32_p32, cospi_p32_m32, &x1[2], &x1[3], _r, cos_bit);
  btf_16_w32_avx512(cospi_p32_p32, cospi_p32_m32, &x1[6], &x1[7], _r, cos_bit);
  btf_16_w32_avx512(cospi_p32_p32, cospi_p32_m32, &x1[10], &x1[11], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p32_p32, cospi_p32_m32, &x1[14], &x1[15], _r,
                    cos_bit);

  for (int i = 0; i < 16; i += 4) {
    btf_16_adds_subs_avx512(&x1[i], &x1[i + 2]);
    btf_16_adds_subs_avx512(&x1[i + 1], &x1[i + 3]);
  }

  btf_16_w32_avx512(cospi_p16_p48, cospi_p48_m16, &x1[4], &x1[5], _r, cos_bit);
  btf_16_w32_avx512(cospi_m48_p16, cospi_p16_p48, &x1[6], &x1[7], _r, cos_bit);
  btf_16_w32_avx512(cospi_p16_p48, cospi_p48_m16, &x1[12], &x1[13], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_m48_p16, cospi_p16_p48, &x1[14], &x1[15], _r,
                    cos_bit);

  for (int i = 0; i < 4; ++i) {
    btf_16_adds_subs_avx512(&x1[i], &x1[i + 4]);
    btf_16_adds_subs_avx512(&x1[i + 8], &x1[i + 12]);
  }

  btf_16_w32_avx512(cospi_p08_p56, cospi_p56_m08, &x1[8], &x1[9], _r, cos_bit);
  btf_16_w32_avx512(cospi_p40_p24, cospi_p24_m40, &x1[10], &x1[11], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_m56_p08, cospi_p08_p56, &x1[12], &x1[13], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_m24_p40, cospi_p40_p24, &x1[14], &x1[15], _r,
                    cos_bit);

  for (int i = 0; i < 8; ++i) btf_16_adds_subs_avx512(&x1[i], &x1[i + 8]);

  btf_16_w32_avx512(cospi_p02_p62, cospi_p62_m02, &x1[0], &x1[1], _r, cos_bit);
  btf_16_w32_avx512(cospi_p10_p54, cospi_p54_m10, &x1[2], &x1[3], _r, cos_bit);
  btf_16_w32_avx512(cospi_p18_p46, cospi_p46_m18, &x1[4], &x1[5], _r, cos_bit);
  btf_16_w32_avx512(cospi_p26_p38, cospi_p38_m26, &x1[6], &x1[7], _r, cos_bit);
  btf_16_w32_avx512(cospi_p34_p30, cospi_p30_m34, &x1[8], &x1[9], _r, cos_bit);
  btf_16_w32_avx512(cospi_p42_p22, cospi_p22_m42, &x1[10], &x1[11], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p50_p14, cospi_p14_m50, &x1[12], &x1[13], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p58_p06, cospi_p06_m58, &x1[14], &x1[15], _r,
                    cos_bit);

  output[0] = x1[1];
  output[1] = x1[14];
  output[2] = x1[3];
  output[3] = x1[12];
  output[4] = x1[5];
  output[5] = x1[10];
  output[6] = x1[7];
  output[7] = x1[8];
  output[8] = x1[9];
  output[9] = x1[6];
  output[10] = x1[11];
  output[11] = x1[4];
  output[12] = x1[13];
  output[13] = x1[2];
  output[14] = x1[15];
  output[15] = x1[0];
}

static inline void fidentity32x16_new_avx512(const __m512i *input,
                                             __m512i *output, int8_t cos_bit) {
  (void)cos_bit;
  const __m512i one = _mm512_set1_epi16(1);
  for (int i = 0; i < 16; ++i) {
    const __m512i a_lo = _mm512_unpacklo_epi16(input[i], one);
    const __m512i a_hi = _mm512_unpackhi_epi16(input[i], one);
    const __m512i b_lo = scale_round_avx512(a_lo, 2 * NewSqrt2);
    const __m512i b_hi = scale_round_avx512(a_hi, 2 * NewSqrt2);
    output[i] = _mm512_packs_epi32(b_lo, b_hi);
  }
}

static inline void fdct32x32_avx512(const __m512i *input, __m512i *output,
                                    int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);
  const __m512i _r = _mm512_set1_epi32(1 << (cos_bit - 1));

  const __m512i cospi_m32_p32 = pair_set_w32_epi16(-cospi[32], cospi[32]);
  const __m512i cospi_p32_p32 = pair_set_w32_epi16(cospi[32], cospi[32]);
  const __m512i cospi_m16_p48 = pair_set_w32_epi16(-cospi[16], cospi[48]);
  const __m512i cospi_p48_p16 = pair_set_w32_epi16(cospi[48], cospi[16]);
  const __m512i cospi_m48_m16 = pair_set_w32_epi16(-cospi[48], -cospi[16]);
  const __m512i cospi_p32_m32 = pair_set_w32_epi16(cospi[32], -cospi[32]);
  const __m512i cospi_p56_p08 = pair_set_w32_epi16(cospi[56], cospi[8]);
  const __m512i cospi_m08_p56 = pair_set_w32_epi16(-cospi[8], cospi[56]);
  const __m512i cospi_p24_p40 = pair_set_w32_epi16(cospi[24], cospi[40]);
  const __m512i cospi_m40_p24 = pair_set_w32_epi16(-cospi[40], cospi[24]);
  const __m512i cospi_m56_m08 = pair_set_w32_epi16(-cospi[56], -cospi[8]);
  const __m512i cospi_m24_m40 = pair_set_w32_epi16(-cospi[24], -cospi[40]);
  const __m512i cospi_p60_p04 = pair_set_w32_epi16(cospi[60], cospi[4]);
  const __m512i cospi_m04_p60 = pair_set_w32_epi16(-cospi[4], cospi[60]);
  const __m512i cospi_p28_p36 = pair_set_w32_epi16(cospi[28], cospi[36]);
  const __m512i cospi_m36_p28 = pair_set_w32_epi16(-cospi[36], cospi[28]);
  const __m512i cospi_p44_p20 = pair_set_w32_epi16(cospi[44], cospi[20]);
  const __m512i cospi_m20_p44 = pair_set_w32_epi16(-cospi[20], cospi[44]);
  const __m512i cospi_p12_p52 = pair_set_w32_epi16(cospi[12], cospi[52]);
  const __m512i cospi_m52_p12 = pair_set_w32_epi16(-cospi[52], cospi[12]);
  const __m512i cospi_p62_p02 = pair_set_w32_epi16(cospi[62], cospi[2]);
  const __m512i cospi_m02_p62 = pair_set_w32_epi16(-cospi[2], cospi[62]);
  const __m512i cospi_p30_p34 = pair_set_w32_epi16(cospi[30], cospi[34]);
  const __m512i cospi_m34_p30 = pair_set_w32_epi16(-cospi[34], cospi[30]);
  const __m512i cospi_p46_p18 = pair_set_w32_epi16(cospi[46], cospi[18]);
  const __m512i cospi_m18_p46 = pair_set_w32_epi16(-cospi[18], cospi[46]);
  const __m512i cospi_p14_p50 = pair_set_w32_epi16(cospi[14], cospi[50]);
  const __m512i cospi_m50_p14 = pair_set_w32_epi16(-cospi[50], cospi[14]);
  const __m512i cospi_p54_p10 = pair_set_w32_epi16(cospi[54], cospi[10]);
  const __m512i cospi_m10_p54 = pair_set_w32_epi16(-cospi[10], cospi[54]);
  const __m512i cospi_p22_p42 = pair_set_w32_epi16(cospi[22], cospi[42]);
  const __m512i cospi_m42_p22 = pair_set_w32_epi16(-cospi[42], cospi[22]);
  const __m512i cospi_p38_p26 = pair_set_w32_epi16(cospi[38], cospi[26]);
  const __m512i cospi_m26_p38 = pair_set_w32_epi16(-cospi[26], cospi[38]);
  const __m512i cospi_p06_p58 = pair_set_w32_epi16(cospi[6], cospi[58]);
  const __m512i cospi_m58_p06 = pair_set_w32_epi16(-cospi[58], cospi[6]);

  __m512i x1[32];
  for (int i = 0; i < 16; ++i) {
    btf_16_adds_subs_out_avx512(&x1[i], &x1[31 - i], input[i], input[31 - i]);
  }

  for (int i = 0; i < 8; ++i) btf_16_adds_subs_avx512(&x1[i], &x1[15 - i]);
  for (int i = 0; i < 4; ++i) {
    btf_16_w32_avx512(cospi_m32_p32, cospi_p32_p32, &x1[20 + i], &x1[27 - i],
                      _r, cos_bit);
  }

  for (int i = 0; i < 4; ++i) btf_16_adds_subs_avx512(&x1[i], &x1[7 - i]);
  btf_16_w32_avx512(cospi_m32_p32, cospi_p32_p32, &x1[10], &x1[13], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_m32_p32, cospi_p32_p32, &x1[11], &x1[12], _r,
                    cos_bit);
  for (int i = 0; i < 4; ++i) {
    btf_16_adds_subs_avx512(&x1[16 + i], &x1[23 - i]);
    btf_16_adds_subs_avx512(&x1[31 - i], &x1[24 + i]);
  }

  btf_16_adds_subs_avx512(&x1[0], &x1[3]);
  btf_16_adds_subs_avx512(&x1[1], &x1[2]);
  btf_16_w32_avx512(cospi_m32_p32, cospi_p32_p32, &x1[5], &x1[6], _r, cos_bit);
  btf_16_adds_subs_avx512(&x1[8], &x1[11]);
  btf_16_adds_subs_avx512(&x1[9], &x1[10]);
  btf_16_adds_subs_avx512(&x1[15], &x1[12]);
  btf_16_adds_subs_avx512(&x1[14], &x1[13]);
  btf_16_w32_avx512(cospi_m16_p48, cospi_p48_p16, &x1[18], &x1[29], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_m16_p48, cospi_p48_p16, &x1[19], &x1[28], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_m48_m16, cospi_m16_p48, &x1[20], &x1[27], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_m48_m16, cospi_m16_p48, &x1[21], &x1[26], _r,
                    cos_bit);

  btf_16_w32_avx512(cospi_p32_p32, cospi_p32_m32, &x1[0], &x1[1], _r, cos_bit);
  btf_16_w32_avx512(cospi_p48_p16, cospi_m16_p48, &x1[2], &x1[3], _r, cos_bit);
  btf_16_adds_subs_avx512(&x1[4], &x1[5]);
  btf_16_adds_subs_avx512(&x1[7], &x1[6]);
  btf_16_w32_avx512(cospi_m16_p48, cospi_p48_p16, &x1[9], &x1[14], _r, cos_bit);
  btf_16_w32_avx512(cospi_m48_m16, cospi_m16_p48, &x1[10], &x1[13], _r,
                    cos_bit);
  btf_16_adds_subs_avx512(&x1[16], &x1[19]);
  btf_16_adds_subs_avx512(&x1[17], &x1[18]);
  btf_16_adds_subs_avx512(&x1[23], &x1[20]);
  btf_16_adds_subs_avx512(&x1[22], &x1[21]);
  btf_16_adds_subs_avx512(&x1[24], &x1[27]);
  btf_16_adds_subs_avx512(&x1[25], &x1[26]);
  btf_16_adds_subs_avx512(&x1[31], &x1[28]);
  btf_16_adds_subs_avx512(&x1[30], &x1[29]);

  btf_16_w32_avx512(cospi_p56_p08, cospi_m08_p56, &x1[4], &x1[7], _r, cos_bit);
  btf_16_w32_avx512(cospi_p24_p40, cospi_m40_p24, &x1[5], &x1[6], _r, cos_bit);
  btf_16_adds_subs_avx512(&x1[8], &x1[9]);
  btf_16_adds_subs_avx512(&x1[11], &x1[10]);
  btf_16_adds_subs_avx512(&x1[12], &x1[13]);
  btf_16_adds_subs_avx512(&x1[15], &x1[14]);
  btf_16_w32_avx512(cospi_m08_p56, cospi_p56_p08, &x1[17], &x1[30], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_m56_m08, cospi_m08_p56, &x1[18], &x1[29], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_m40_p24, cospi_p24_p40, &x1[21], &x1[26], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_m24_m40, cospi_m40_p24, &x1[22], &x1[25], _r,
                    cos_bit);

  btf_16_w32_avx512(cospi_p60_p04, cospi_m04_p60, &x1[8], &x1[15], _r, cos_bit);
  btf_16_w32_avx512(cospi_p28_p36, cospi_m36_p28, &x1[9], &x1[14], _r, cos_bit);
  btf_16_w32_avx512(cospi_p44_p20, cospi_m20_p44, &x1[10], &x1[13], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p12_p52, cospi_m52_p12, &x1[11], &x1[12], _r,
                    cos_bit);
  for (int i = 16; i < 32; i += 4) {
    btf_16_adds_subs_avx512(&x1[i], &x1[i + 1]);
    btf_16_adds_subs_avx512(&x1[i + 3], &x1[i + 2]);
  }

  btf_16_w32_avx512(cospi_p62_p02, cospi_m02_p62, &x1[16], &x1[31], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p30_p34, cospi_m34_p30, &x1[17], &x1[30], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p46_p18, cospi_m18_p46, &x1[18], &x1[29], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p14_p50, cospi_m50_p14, &x1[19], &x1[28], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p54_p10, cospi_m10_p54, &x1[20], &x1[27], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p22_p42, cospi_m42_p22, &x1[21], &x1[26], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p38_p26, cospi_m26_p38, &x1[22], &x1[25], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p06_p58, cospi_m58_p06, &x1[23], &x1[24], _r,
                    cos_bit);

  output[0] = x1[0];
  output[1] = x1[16];
  output[2] = x1[8];
  output[3] = x1[24];
  output[4] = x1[4];
  output[5] = x1[20];
  output[6] = x1[12];
  output[7] = x1[28];
  output[8] = x1[2];
  output[9] = x1[18];
  output[10] = x1[10];
  output[11] = x1[26];
  output[12] = x1[6];
  output[13] = x1[22];
  output[14] = x1[14];
  output[15] = x1[30];
  output[16] = x1[1];
  output[17] = x1[17];
  output[18] = x1[9];
  output[19] = x1[25];
  output[20] = x1[5];
  output[21] = x1[21];
  output[22] = x1[13];
  output[23] = x1[29];
  output[24] = x1[3];
  output[25] = x1[19];
  output[26] = x1[11];
  output[27] = x1[27];
  output[28] = x1[7];
  output[29] = x1[23];
  output[30] = x1[15];
  output[31] = x1[31];
}

static inline void fidentity32x32_avx512(const __m512i *input, __m512i *output,
                                         int8_t cos_bit) {
  (void)cos_bit;
  for (int i = 0; i < 32; ++i) {
    output[i] = _mm512_slli_epi16(input[i], 2);
  }
}

static inline void fdct32x64_new_avx512(const __m512i *input, __m512i *output,
                                        int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);
  const __m512i _r = _mm512_set1_epi32(1 << (cos_bit - 1));

  const __m512i cospi_m32_p32 = pair_set_w32_epi16(-cospi[32], cospi[32]);
  const __m512i cospi_p32_p32 = pair_set_w32_epi16(cospi[32], cospi[32]);
  const __m512i cospi_m16_p48 = pair_set_w32_epi16(-cospi[16], cospi[48]);
  const __m512i cospi_p48_p16 = pair_set_w32_epi16(cospi[48], cospi[16]);
  const __m512i cospi_m48_m16 = pair_set_w32_epi16(-cospi[48], -cospi[16]);
  const __m512i cospi_p32_m32 = pair_set_w32_epi16(cospi[32], -cospi[32]);
  const __m512i cospi_m08_p56 = pair_set_w32_epi16(-cospi[8], cospi[56]);
  const __m512i cospi_p56_p08 = pair_set_w32_epi16(cospi[56], cospi[8]);
  const __m512i cospi_m56_m08 = pair_set_w32_epi16(-cospi[56], -cospi[8]);
  const __m512i cospi_m40_p24 = pair_set_w32_epi16(-cospi[40], cospi[24]);
  const __m512i cospi_p24_p40 = pair_set_w32_epi16(cospi[24], cospi[40]);
  const __m512i cospi_m24_m40 = pair_set_w32_epi16(-cospi[24], -cospi[40]);
  const __m512i cospi_p60_p04 = pair_set_w32_epi16(cospi[60], cospi[4]);
  const __m512i cospi_m04_p60 = pair_set_w32_epi16(-cospi[4], cospi[60]);
  const __m512i cospi_p28_p36 = pair_set_w32_epi16(cospi[28], cospi[36]);
  const __m512i cospi_m36_p28 = pair_set_w32_epi16(-cospi[36], cospi[28]);
  const __m512i cospi_p44_p20 = pair_set_w32_epi16(cospi[44], cospi[20]);
  const __m512i cospi_m20_p44 = pair_set_w32_epi16(-cospi[20], cospi[44]);
  const __m512i cospi_p12_p52 = pair_set_w32_epi16(cospi[12], cospi[52]);
  const __m512i cospi_m52_p12 = pair_set_w32_epi16(-cospi[52], cospi[12]);
  const __m512i cospi_m60_m04 = pair_set_w32_epi16(-cospi[60], -cospi[4]);
  const __m512i cospi_m28_m36 = pair_set_w32_epi16(-cospi[28], -cospi[36]);
  const __m512i cospi_m44_m20 = pair_set_w32_epi16(-cospi[44], -cospi[20]);
  const __m512i cospi_m12_m52 = pair_set_w32_epi16(-cospi[12], -cospi[52]);
  const __m512i cospi_p62_p02 = pair_set_w32_epi16(cospi[62], cospi[2]);
  const __m512i cospi_m02_p62 = pair_set_w32_epi16(-cospi[2], cospi[62]);
  const __m512i cospi_p30_p34 = pair_set_w32_epi16(cospi[30], cospi[34]);
  const __m512i cospi_m34_p30 = pair_set_w32_epi16(-cospi[34], cospi[30]);
  const __m512i cospi_p46_p18 = pair_set_w32_epi16(cospi[46], cospi[18]);
  const __m512i cospi_m18_p46 = pair_set_w32_epi16(-cospi[18], cospi[46]);
  const __m512i cospi_p14_p50 = pair_set_w32_epi16(cospi[14], cospi[50]);
  const __m512i cospi_m50_p14 = pair_set_w32_epi16(-cospi[50], cospi[14]);
  const __m512i cospi_p54_p10 = pair_set_w32_epi16(cospi[54], cospi[10]);
  const __m512i cospi_m10_p54 = pair_set_w32_epi16(-cospi[10], cospi[54]);
  const __m512i cospi_p22_p42 = pair_set_w32_epi16(cospi[22], cospi[42]);
  const __m512i cospi_m42_p22 = pair_set_w32_epi16(-cospi[42], cospi[22]);
  const __m512i cospi_p38_p26 = pair_set_w32_epi16(cospi[38], cospi[26]);
  const __m512i cospi_m26_p38 = pair_set_w32_epi16(-cospi[26], cospi[38]);
  const __m512i cospi_p06_p58 = pair_set_w32_epi16(cospi[6], cospi[58]);
  const __m512i cospi_m58_p06 = pair_set_w32_epi16(-cospi[58], cospi[6]);
  const __m512i cospi_p63_p01 = pair_set_w32_epi16(cospi[63], cospi[1]);
  const __m512i cospi_m01_p63 = pair_set_w32_epi16(-cospi[1], cospi[63]);
  const __m512i cospi_p31_p33 = pair_set_w32_epi16(cospi[31], cospi[33]);
  const __m512i cospi_m33_p31 = pair_set_w32_epi16(-cospi[33], cospi[31]);
  const __m512i cospi_p47_p17 = pair_set_w32_epi16(cospi[47], cospi[17]);
  const __m512i cospi_m17_p47 = pair_set_w32_epi16(-cospi[17], cospi[47]);
  const __m512i cospi_p15_p49 = pair_set_w32_epi16(cospi[15], cospi[49]);
  const __m512i cospi_m49_p15 = pair_set_w32_epi16(-cospi[49], cospi[15]);
  const __m512i cospi_p55_p09 = pair_set_w32_epi16(cospi[55], cospi[9]);
  const __m512i cospi_m09_p55 = pair_set_w32_epi16(-cospi[9], cospi[55]);
  const __m512i cospi_p23_p41 = pair_set_w32_epi16(cospi[23], cospi[41]);
  const __m512i cospi_m41_p23 = pair_set_w32_epi16(-cospi[41], cospi[23]);
  const __m512i cospi_p39_p25 = pair_set_w32_epi16(cospi[39], cospi[25]);
  const __m512i cospi_m25_p39 = pair_set_w32_epi16(-cospi[25], cospi[39]);
  const __m512i cospi_p07_p57 = pair_set_w32_epi16(cospi[7], cospi[57]);
  const __m512i cospi_m57_p07 = pair_set_w32_epi16(-cospi[57], cospi[7]);
  const __m512i cospi_p59_p05 = pair_set_w32_epi16(cospi[59], cospi[5]);
  const __m512i cospi_m05_p59 = pair_set_w32_epi16(-cospi[5], cospi[59]);
  const __m512i cospi_p27_p37 = pair_set_w32_epi16(cospi[27], cospi[37]);
  const __m512i cospi_m37_p27 = pair_set_w32_epi16(-cospi[37], cospi[27]);
  const __m512i cospi_p43_p21 = pair_set_w32_epi16(cospi[43], cospi[21]);
  const __m512i cospi_m21_p43 = pair_set_w32_epi16(-cospi[21], cospi[43]);
  const __m512i cospi_p11_p53 = pair_set_w32_epi16(cospi[11], cospi[53]);
  const __m512i cospi_m53_p11 = pair_set_w32_epi16(-cospi[53], cospi[11]);
  const __m512i cospi_p51_p13 = pair_set_w32_epi16(cospi[51], cospi[13]);
  const __m512i cospi_m13_p51 = pair_set_w32_epi16(-cospi[13], cospi[51]);
  const __m512i cospi_p19_p45 = pair_set_w32_epi16(cospi[19], cospi[45]);
  const __m512i cospi_m45_p19 = pair_set_w32_epi16(-cospi[45], cospi[19]);
  const __m512i cospi_p35_p29 = pair_set_w32_epi16(cospi[35], cospi[29]);
  const __m512i cospi_m29_p35 = pair_set_w32_epi16(-cospi[29], cospi[35]);
  const __m512i cospi_p03_p61 = pair_set_w32_epi16(cospi[3], cospi[61]);
  const __m512i cospi_m61_p03 = pair_set_w32_epi16(-cospi[61], cospi[3]);

  __m512i x1[64];
  for (int i = 0; i < 32; ++i) {
    btf_16_adds_subs_out_avx512(&x1[i], &x1[63 - i], input[i], input[63 - i]);
  }
  for (int i = 0; i < 16; ++i) btf_16_adds_subs_avx512(&x1[i], &x1[31 - i]);
  for (int i = 0; i < 8; ++i) {
    btf_16_w32_avx512(cospi_m32_p32, cospi_p32_p32, &x1[40 + i], &x1[55 - i],
                      _r, cos_bit);
  }
  for (int i = 0; i < 8; ++i) btf_16_adds_subs_avx512(&x1[i], &x1[15 - i]);
  for (int i = 0; i < 4; ++i) {
    btf_16_w32_avx512(cospi_m32_p32, cospi_p32_p32, &x1[20 + i], &x1[27 - i],
                      _r, cos_bit);
  }
  for (int i = 0; i < 8; ++i) {
    btf_16_adds_subs_avx512(&x1[32 + i], &x1[47 - i]);
    btf_16_adds_subs_avx512(&x1[63 - i], &x1[48 + i]);
  }
  for (int i = 0; i < 4; ++i) btf_16_adds_subs_avx512(&x1[i], &x1[7 - i]);
  btf_16_w32_avx512(cospi_m32_p32, cospi_p32_p32, &x1[10], &x1[13], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_m32_p32, cospi_p32_p32, &x1[11], &x1[12], _r,
                    cos_bit);
  for (int i = 0; i < 4; ++i) {
    btf_16_adds_subs_avx512(&x1[16 + i], &x1[23 - i]);
    btf_16_adds_subs_avx512(&x1[31 - i], &x1[24 + i]);
    btf_16_w32_avx512(cospi_m16_p48, cospi_p48_p16, &x1[36 + i], &x1[59 - i],
                      _r, cos_bit);
    btf_16_w32_avx512(cospi_m48_m16, cospi_m16_p48, &x1[40 + i], &x1[55 - i],
                      _r, cos_bit);
  }
  btf_16_adds_subs_avx512(&x1[0], &x1[3]);
  btf_16_adds_subs_avx512(&x1[1], &x1[2]);
  btf_16_w32_avx512(cospi_m32_p32, cospi_p32_p32, &x1[5], &x1[6], _r, cos_bit);
  btf_16_adds_subs_avx512(&x1[8], &x1[11]);
  btf_16_adds_subs_avx512(&x1[9], &x1[10]);
  btf_16_adds_subs_avx512(&x1[15], &x1[12]);
  btf_16_adds_subs_avx512(&x1[14], &x1[13]);
  btf_16_w32_avx512(cospi_m16_p48, cospi_p48_p16, &x1[18], &x1[29], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_m16_p48, cospi_p48_p16, &x1[19], &x1[28], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_m48_m16, cospi_m16_p48, &x1[20], &x1[27], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_m48_m16, cospi_m16_p48, &x1[21], &x1[26], _r,
                    cos_bit);
  for (int i = 0; i < 4; ++i) {
    btf_16_adds_subs_avx512(&x1[32 + i], &x1[39 - i]);
    btf_16_adds_subs_avx512(&x1[47 - i], &x1[40 + i]);
    btf_16_adds_subs_avx512(&x1[48 + i], &x1[55 - i]);
    btf_16_adds_subs_avx512(&x1[63 - i], &x1[56 + i]);
  }
  btf_16_w32_avx512(cospi_p32_p32, cospi_p32_m32, &x1[0], &x1[1], _r, cos_bit);
  btf_16_w32_avx512(cospi_p48_p16, cospi_m16_p48, &x1[2], &x1[3], _r, cos_bit);
  btf_16_adds_subs_avx512(&x1[4], &x1[5]);
  btf_16_adds_subs_avx512(&x1[7], &x1[6]);
  btf_16_w32_avx512(cospi_m16_p48, cospi_p48_p16, &x1[9], &x1[14], _r, cos_bit);
  btf_16_w32_avx512(cospi_m48_m16, cospi_m16_p48, &x1[10], &x1[13], _r,
                    cos_bit);
  for (int i = 0; i < 2; ++i) {
    btf_16_adds_subs_avx512(&x1[16 + i], &x1[19 - i]);
    btf_16_adds_subs_avx512(&x1[23 - i], &x1[20 + i]);
    btf_16_adds_subs_avx512(&x1[24 + i], &x1[27 - i]);
    btf_16_adds_subs_avx512(&x1[31 - i], &x1[28 + i]);
    btf_16_w32_avx512(cospi_m08_p56, cospi_p56_p08, &x1[34 + i], &x1[61 - i],
                      _r, cos_bit);
    btf_16_w32_avx512(cospi_m56_m08, cospi_m08_p56, &x1[36 + i], &x1[59 - i],
                      _r, cos_bit);
    btf_16_w32_avx512(cospi_m40_p24, cospi_p24_p40, &x1[42 + i], &x1[53 - i],
                      _r, cos_bit);
    btf_16_w32_avx512(cospi_m24_m40, cospi_m40_p24, &x1[44 + i], &x1[51 - i],
                      _r, cos_bit);
  }
  btf_16_w32_avx512(cospi_p56_p08, cospi_m08_p56, &x1[4], &x1[7], _r, cos_bit);
  btf_16_w32_avx512(cospi_p24_p40, cospi_m40_p24, &x1[5], &x1[6], _r, cos_bit);
  btf_16_adds_subs_avx512(&x1[8], &x1[9]);
  btf_16_adds_subs_avx512(&x1[11], &x1[10]);
  btf_16_adds_subs_avx512(&x1[12], &x1[13]);
  btf_16_adds_subs_avx512(&x1[15], &x1[14]);
  btf_16_w32_avx512(cospi_m08_p56, cospi_p56_p08, &x1[17], &x1[30], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_m56_m08, cospi_m08_p56, &x1[18], &x1[29], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_m40_p24, cospi_p24_p40, &x1[21], &x1[26], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_m24_m40, cospi_m40_p24, &x1[22], &x1[25], _r,
                    cos_bit);
  for (int k = 32; k < 64; k += 8) {
    btf_16_adds_subs_avx512(&x1[k + 0], &x1[k + 3]);
    btf_16_adds_subs_avx512(&x1[k + 1], &x1[k + 2]);
    btf_16_adds_subs_avx512(&x1[k + 7], &x1[k + 4]);
    btf_16_adds_subs_avx512(&x1[k + 6], &x1[k + 5]);
  }
  btf_16_w32_avx512(cospi_p60_p04, cospi_m04_p60, &x1[8], &x1[15], _r, cos_bit);
  btf_16_w32_avx512(cospi_p28_p36, cospi_m36_p28, &x1[9], &x1[14], _r, cos_bit);
  btf_16_w32_avx512(cospi_p44_p20, cospi_m20_p44, &x1[10], &x1[13], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p12_p52, cospi_m52_p12, &x1[11], &x1[12], _r,
                    cos_bit);
  for (int k = 16; k < 32; k += 4) {
    btf_16_adds_subs_avx512(&x1[k + 0], &x1[k + 1]);
    btf_16_adds_subs_avx512(&x1[k + 3], &x1[k + 2]);
  }
  btf_16_w32_avx512(cospi_m04_p60, cospi_p60_p04, &x1[33], &x1[62], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_m60_m04, cospi_m04_p60, &x1[34], &x1[61], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_m36_p28, cospi_p28_p36, &x1[37], &x1[58], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_m28_m36, cospi_m36_p28, &x1[38], &x1[57], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_m20_p44, cospi_p44_p20, &x1[41], &x1[54], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_m44_m20, cospi_m20_p44, &x1[42], &x1[53], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_m52_p12, cospi_p12_p52, &x1[45], &x1[50], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_m12_m52, cospi_m52_p12, &x1[46], &x1[49], _r,
                    cos_bit);

  btf_16_w32_avx512(cospi_p62_p02, cospi_m02_p62, &x1[16], &x1[31], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p30_p34, cospi_m34_p30, &x1[17], &x1[30], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p46_p18, cospi_m18_p46, &x1[18], &x1[29], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p14_p50, cospi_m50_p14, &x1[19], &x1[28], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p54_p10, cospi_m10_p54, &x1[20], &x1[27], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p22_p42, cospi_m42_p22, &x1[21], &x1[26], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p38_p26, cospi_m26_p38, &x1[22], &x1[25], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p06_p58, cospi_m58_p06, &x1[23], &x1[24], _r,
                    cos_bit);
  for (int k = 32; k < 64; k += 4) {
    btf_16_adds_subs_avx512(&x1[k + 0], &x1[k + 1]);
    btf_16_adds_subs_avx512(&x1[k + 3], &x1[k + 2]);
  }
  btf_16_w32_avx512(cospi_p63_p01, cospi_m01_p63, &x1[32], &x1[63], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p31_p33, cospi_m33_p31, &x1[33], &x1[62], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p47_p17, cospi_m17_p47, &x1[34], &x1[61], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p15_p49, cospi_m49_p15, &x1[35], &x1[60], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p55_p09, cospi_m09_p55, &x1[36], &x1[59], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p23_p41, cospi_m41_p23, &x1[37], &x1[58], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p39_p25, cospi_m25_p39, &x1[38], &x1[57], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p07_p57, cospi_m57_p07, &x1[39], &x1[56], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p59_p05, cospi_m05_p59, &x1[40], &x1[55], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p27_p37, cospi_m37_p27, &x1[41], &x1[54], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p43_p21, cospi_m21_p43, &x1[42], &x1[53], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p11_p53, cospi_m53_p11, &x1[43], &x1[52], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p51_p13, cospi_m13_p51, &x1[44], &x1[51], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p19_p45, cospi_m45_p19, &x1[45], &x1[50], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p35_p29, cospi_m29_p35, &x1[46], &x1[49], _r,
                    cos_bit);
  btf_16_w32_avx512(cospi_p03_p61, cospi_m61_p03, &x1[47], &x1[48], _r,
                    cos_bit);

  static const uint8_t kBitRev64[64] = {
    0, 32, 16, 48, 8,  40, 24, 56, 4, 36, 20, 52, 12, 44, 28, 60,
    2, 34, 18, 50, 10, 42, 26, 58, 6, 38, 22, 54, 14, 46, 30, 62,
    1, 33, 17, 49, 9,  41, 25, 57, 5, 37, 21, 53, 13, 45, 29, 61,
    3, 35, 19, 51, 11, 43, 27, 59, 7, 39, 23, 55, 15, 47, 31, 63
  };
  for (int i = 0; i < 64; ++i) {
    output[i] = x1[kBitRev64[i]];
  }
}

static inline void fdct32_avx512(const __m512i *input, __m512i *output,
                                 int8_t cos_bit) {
  __m512i x1[32];
  const int32_t *cospi = cospi_arr(cos_bit);
  const __m512i _r = _mm512_set1_epi32(1 << (cos_bit - 1));

  for (int i = 0; i < 16; ++i) {
    btf_32_add_sub_out_avx512(&x1[i], &x1[31 - i], input[i], input[31 - i]);
  }
  for (int i = 0; i < 8; ++i) btf_32_add_sub_avx512(&x1[i], &x1[15 - i]);
  for (int i = 0; i < 4; ++i) {
    btf_32_avx512_type0(-cospi[32], cospi[32], &x1[20 + i], &x1[27 - i], _r,
                        cos_bit);
  }
  for (int i = 0; i < 4; ++i) btf_32_add_sub_avx512(&x1[i], &x1[7 - i]);
  btf_32_avx512_type0(-cospi[32], cospi[32], &x1[10], &x1[13], _r, cos_bit);
  btf_32_avx512_type0(-cospi[32], cospi[32], &x1[11], &x1[12], _r, cos_bit);
  for (int i = 0; i < 4; ++i) {
    btf_32_add_sub_avx512(&x1[16 + i], &x1[23 - i]);
    btf_32_add_sub_avx512(&x1[31 - i], &x1[24 + i]);
  }
  btf_32_add_sub_avx512(&x1[0], &x1[3]);
  btf_32_add_sub_avx512(&x1[1], &x1[2]);
  btf_32_avx512_type0(-cospi[32], cospi[32], &x1[5], &x1[6], _r, cos_bit);
  btf_32_add_sub_avx512(&x1[8], &x1[11]);
  btf_32_add_sub_avx512(&x1[9], &x1[10]);
  btf_32_add_sub_avx512(&x1[15], &x1[12]);
  btf_32_add_sub_avx512(&x1[14], &x1[13]);
  btf_32_avx512_type0(-cospi[16], cospi[48], &x1[18], &x1[29], _r, cos_bit);
  btf_32_avx512_type0(-cospi[16], cospi[48], &x1[19], &x1[28], _r, cos_bit);
  btf_32_avx512_type0(-cospi[48], -cospi[16], &x1[20], &x1[27], _r, cos_bit);
  btf_32_avx512_type0(-cospi[48], -cospi[16], &x1[21], &x1[26], _r, cos_bit);

  btf_32_avx512_type0(cospi[32], cospi[32], &x1[0], &x1[1], _r, cos_bit);
  btf_32_avx512_type1(cospi[48], cospi[16], &x1[2], &x1[3], _r, cos_bit);
  btf_32_add_sub_avx512(&x1[4], &x1[5]);
  btf_32_add_sub_avx512(&x1[7], &x1[6]);
  btf_32_avx512_type0(-cospi[16], cospi[48], &x1[9], &x1[14], _r, cos_bit);
  btf_32_avx512_type0(-cospi[48], -cospi[16], &x1[10], &x1[13], _r, cos_bit);
  for (int i = 0; i < 2; ++i) {
    btf_32_add_sub_avx512(&x1[16 + i], &x1[19 - i]);
    btf_32_add_sub_avx512(&x1[23 - i], &x1[20 + i]);
    btf_32_add_sub_avx512(&x1[24 + i], &x1[27 - i]);
    btf_32_add_sub_avx512(&x1[31 - i], &x1[28 + i]);
  }
  btf_32_avx512_type1(cospi[56], cospi[8], &x1[4], &x1[7], _r, cos_bit);
  btf_32_avx512_type1(cospi[24], cospi[40], &x1[5], &x1[6], _r, cos_bit);
  btf_32_add_sub_avx512(&x1[8], &x1[9]);
  btf_32_add_sub_avx512(&x1[11], &x1[10]);
  btf_32_add_sub_avx512(&x1[12], &x1[13]);
  btf_32_add_sub_avx512(&x1[15], &x1[14]);
  btf_32_avx512_type0(-cospi[8], cospi[56], &x1[17], &x1[30], _r, cos_bit);
  btf_32_avx512_type0(-cospi[56], -cospi[8], &x1[18], &x1[29], _r, cos_bit);
  btf_32_avx512_type0(-cospi[40], cospi[24], &x1[21], &x1[26], _r, cos_bit);
  btf_32_avx512_type0(-cospi[24], -cospi[40], &x1[22], &x1[25], _r, cos_bit);

  btf_32_avx512_type1(cospi[60], cospi[4], &x1[8], &x1[15], _r, cos_bit);
  btf_32_avx512_type1(cospi[28], cospi[36], &x1[9], &x1[14], _r, cos_bit);
  btf_32_avx512_type1(cospi[44], cospi[20], &x1[10], &x1[13], _r, cos_bit);
  btf_32_avx512_type1(cospi[12], cospi[52], &x1[11], &x1[12], _r, cos_bit);
  for (int k = 16; k < 32; k += 4) {
    btf_32_add_sub_avx512(&x1[k + 0], &x1[k + 1]);
    btf_32_add_sub_avx512(&x1[k + 3], &x1[k + 2]);
  }
  btf_32_avx512_type1(cospi[62], cospi[2], &x1[16], &x1[31], _r, cos_bit);
  btf_32_avx512_type1(cospi[30], cospi[34], &x1[17], &x1[30], _r, cos_bit);
  btf_32_avx512_type1(cospi[46], cospi[18], &x1[18], &x1[29], _r, cos_bit);
  btf_32_avx512_type1(cospi[14], cospi[50], &x1[19], &x1[28], _r, cos_bit);
  btf_32_avx512_type1(cospi[54], cospi[10], &x1[20], &x1[27], _r, cos_bit);
  btf_32_avx512_type1(cospi[22], cospi[42], &x1[21], &x1[26], _r, cos_bit);
  btf_32_avx512_type1(cospi[38], cospi[26], &x1[22], &x1[25], _r, cos_bit);
  btf_32_avx512_type1(cospi[6], cospi[58], &x1[23], &x1[24], _r, cos_bit);

  static const uint8_t kBitRev32[32] = { 0, 16, 8,  24, 4, 20, 12, 28,
                                         2, 18, 10, 26, 6, 22, 14, 30,
                                         1, 17, 9,  25, 5, 21, 13, 29,
                                         3, 19, 11, 27, 7, 23, 15, 31 };
  for (int i = 0; i < 32; ++i) {
    output[i] = x1[kBitRev32[i]];
  }
}

static inline void fdct64_new_avx512(const __m512i *input, __m512i *output,
                                     int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);
  const __m512i _r = _mm512_set1_epi32(1 << (cos_bit - 1));

  __m512i x1[64];
  for (int i = 0; i < 32; ++i) {
    btf_32_add_sub_out_avx512(&x1[i], &x1[63 - i], input[i], input[63 - i]);
  }
  for (int i = 0; i < 16; ++i) btf_32_add_sub_avx512(&x1[i], &x1[31 - i]);
  for (int i = 0; i < 8; ++i) {
    btf_32_avx512_type0(-cospi[32], cospi[32], &x1[40 + i], &x1[55 - i], _r,
                        cos_bit);
  }
  for (int i = 0; i < 8; ++i) btf_32_add_sub_avx512(&x1[i], &x1[15 - i]);
  for (int i = 0; i < 4; ++i) {
    btf_32_avx512_type0(-cospi[32], cospi[32], &x1[20 + i], &x1[27 - i], _r,
                        cos_bit);
  }
  for (int i = 0; i < 8; ++i) {
    btf_32_add_sub_avx512(&x1[32 + i], &x1[47 - i]);
    btf_32_add_sub_avx512(&x1[63 - i], &x1[48 + i]);
  }
  for (int i = 0; i < 4; ++i) btf_32_add_sub_avx512(&x1[i], &x1[7 - i]);
  btf_32_avx512_type0(-cospi[32], cospi[32], &x1[10], &x1[13], _r, cos_bit);
  btf_32_avx512_type0(-cospi[32], cospi[32], &x1[11], &x1[12], _r, cos_bit);
  for (int i = 0; i < 4; ++i) {
    btf_32_add_sub_avx512(&x1[16 + i], &x1[23 - i]);
    btf_32_add_sub_avx512(&x1[31 - i], &x1[24 + i]);
    btf_32_avx512_type0(-cospi[16], cospi[48], &x1[36 + i], &x1[59 - i], _r,
                        cos_bit);
    btf_32_avx512_type0(-cospi[48], -cospi[16], &x1[40 + i], &x1[55 - i], _r,
                        cos_bit);
  }
  btf_32_add_sub_avx512(&x1[0], &x1[3]);
  btf_32_add_sub_avx512(&x1[1], &x1[2]);
  btf_32_avx512_type0(-cospi[32], cospi[32], &x1[5], &x1[6], _r, cos_bit);
  btf_32_add_sub_avx512(&x1[8], &x1[11]);
  btf_32_add_sub_avx512(&x1[9], &x1[10]);
  btf_32_add_sub_avx512(&x1[15], &x1[12]);
  btf_32_add_sub_avx512(&x1[14], &x1[13]);
  btf_32_avx512_type0(-cospi[16], cospi[48], &x1[18], &x1[29], _r, cos_bit);
  btf_32_avx512_type0(-cospi[16], cospi[48], &x1[19], &x1[28], _r, cos_bit);
  btf_32_avx512_type0(-cospi[48], -cospi[16], &x1[20], &x1[27], _r, cos_bit);
  btf_32_avx512_type0(-cospi[48], -cospi[16], &x1[21], &x1[26], _r, cos_bit);
  for (int i = 0; i < 4; ++i) {
    btf_32_add_sub_avx512(&x1[32 + i], &x1[39 - i]);
    btf_32_add_sub_avx512(&x1[47 - i], &x1[40 + i]);
    btf_32_add_sub_avx512(&x1[48 + i], &x1[55 - i]);
    btf_32_add_sub_avx512(&x1[63 - i], &x1[56 + i]);
  }
  btf_32_avx512_type0(cospi[32], cospi[32], &x1[0], &x1[1], _r, cos_bit);
  btf_32_avx512_type1(cospi[48], cospi[16], &x1[2], &x1[3], _r, cos_bit);
  btf_32_add_sub_avx512(&x1[4], &x1[5]);
  btf_32_add_sub_avx512(&x1[7], &x1[6]);
  btf_32_avx512_type0(-cospi[16], cospi[48], &x1[9], &x1[14], _r, cos_bit);
  btf_32_avx512_type0(-cospi[48], -cospi[16], &x1[10], &x1[13], _r, cos_bit);
  for (int i = 0; i < 2; ++i) {
    btf_32_add_sub_avx512(&x1[16 + i], &x1[19 - i]);
    btf_32_add_sub_avx512(&x1[23 - i], &x1[20 + i]);
    btf_32_add_sub_avx512(&x1[24 + i], &x1[27 - i]);
    btf_32_add_sub_avx512(&x1[31 - i], &x1[28 + i]);
    btf_32_avx512_type0(-cospi[8], cospi[56], &x1[34 + i], &x1[61 - i], _r,
                        cos_bit);
    btf_32_avx512_type0(-cospi[56], -cospi[8], &x1[36 + i], &x1[59 - i], _r,
                        cos_bit);
    btf_32_avx512_type0(-cospi[40], cospi[24], &x1[42 + i], &x1[53 - i], _r,
                        cos_bit);
    btf_32_avx512_type0(-cospi[24], -cospi[40], &x1[44 + i], &x1[51 - i], _r,
                        cos_bit);
  }
  btf_32_avx512_type1(cospi[56], cospi[8], &x1[4], &x1[7], _r, cos_bit);
  btf_32_avx512_type1(cospi[24], cospi[40], &x1[5], &x1[6], _r, cos_bit);
  btf_32_add_sub_avx512(&x1[8], &x1[9]);
  btf_32_add_sub_avx512(&x1[11], &x1[10]);
  btf_32_add_sub_avx512(&x1[12], &x1[13]);
  btf_32_add_sub_avx512(&x1[15], &x1[14]);
  btf_32_avx512_type0(-cospi[8], cospi[56], &x1[17], &x1[30], _r, cos_bit);
  btf_32_avx512_type0(-cospi[56], -cospi[8], &x1[18], &x1[29], _r, cos_bit);
  btf_32_avx512_type0(-cospi[40], cospi[24], &x1[21], &x1[26], _r, cos_bit);
  btf_32_avx512_type0(-cospi[24], -cospi[40], &x1[22], &x1[25], _r, cos_bit);
  for (int k = 32; k < 64; k += 8) {
    btf_32_add_sub_avx512(&x1[k + 0], &x1[k + 3]);
    btf_32_add_sub_avx512(&x1[k + 1], &x1[k + 2]);
    btf_32_add_sub_avx512(&x1[k + 7], &x1[k + 4]);
    btf_32_add_sub_avx512(&x1[k + 6], &x1[k + 5]);
  }
  btf_32_avx512_type1(cospi[60], cospi[4], &x1[8], &x1[15], _r, cos_bit);
  btf_32_avx512_type1(cospi[28], cospi[36], &x1[9], &x1[14], _r, cos_bit);
  btf_32_avx512_type1(cospi[44], cospi[20], &x1[10], &x1[13], _r, cos_bit);
  btf_32_avx512_type1(cospi[12], cospi[52], &x1[11], &x1[12], _r, cos_bit);
  for (int k = 16; k < 32; k += 4) {
    btf_32_add_sub_avx512(&x1[k + 0], &x1[k + 1]);
    btf_32_add_sub_avx512(&x1[k + 3], &x1[k + 2]);
  }
  btf_32_avx512_type0(-cospi[4], cospi[60], &x1[33], &x1[62], _r, cos_bit);
  btf_32_avx512_type0(-cospi[60], -cospi[4], &x1[34], &x1[61], _r, cos_bit);
  btf_32_avx512_type0(-cospi[36], cospi[28], &x1[37], &x1[58], _r, cos_bit);
  btf_32_avx512_type0(-cospi[28], -cospi[36], &x1[38], &x1[57], _r, cos_bit);
  btf_32_avx512_type0(-cospi[20], cospi[44], &x1[41], &x1[54], _r, cos_bit);
  btf_32_avx512_type0(-cospi[44], -cospi[20], &x1[42], &x1[53], _r, cos_bit);
  btf_32_avx512_type0(-cospi[52], cospi[12], &x1[45], &x1[50], _r, cos_bit);
  btf_32_avx512_type0(-cospi[12], -cospi[52], &x1[46], &x1[49], _r, cos_bit);

  btf_32_avx512_type1(cospi[62], cospi[2], &x1[16], &x1[31], _r, cos_bit);
  btf_32_avx512_type1(cospi[30], cospi[34], &x1[17], &x1[30], _r, cos_bit);
  btf_32_avx512_type1(cospi[46], cospi[18], &x1[18], &x1[29], _r, cos_bit);
  btf_32_avx512_type1(cospi[14], cospi[50], &x1[19], &x1[28], _r, cos_bit);
  btf_32_avx512_type1(cospi[54], cospi[10], &x1[20], &x1[27], _r, cos_bit);
  btf_32_avx512_type1(cospi[22], cospi[42], &x1[21], &x1[26], _r, cos_bit);
  btf_32_avx512_type1(cospi[38], cospi[26], &x1[22], &x1[25], _r, cos_bit);
  btf_32_avx512_type1(cospi[6], cospi[58], &x1[23], &x1[24], _r, cos_bit);
  for (int k = 32; k < 64; k += 4) {
    btf_32_add_sub_avx512(&x1[k + 0], &x1[k + 1]);
    btf_32_add_sub_avx512(&x1[k + 3], &x1[k + 2]);
  }
  btf_32_avx512_type1(cospi[63], cospi[1], &x1[32], &x1[63], _r, cos_bit);
  btf_32_avx512_type1(cospi[31], cospi[33], &x1[33], &x1[62], _r, cos_bit);
  btf_32_avx512_type1(cospi[47], cospi[17], &x1[34], &x1[61], _r, cos_bit);
  btf_32_avx512_type1(cospi[15], cospi[49], &x1[35], &x1[60], _r, cos_bit);
  btf_32_avx512_type1(cospi[55], cospi[9], &x1[36], &x1[59], _r, cos_bit);
  btf_32_avx512_type1(cospi[23], cospi[41], &x1[37], &x1[58], _r, cos_bit);
  btf_32_avx512_type1(cospi[39], cospi[25], &x1[38], &x1[57], _r, cos_bit);
  btf_32_avx512_type1(cospi[7], cospi[57], &x1[39], &x1[56], _r, cos_bit);
  btf_32_avx512_type1(cospi[59], cospi[5], &x1[40], &x1[55], _r, cos_bit);
  btf_32_avx512_type1(cospi[27], cospi[37], &x1[41], &x1[54], _r, cos_bit);
  btf_32_avx512_type1(cospi[43], cospi[21], &x1[42], &x1[53], _r, cos_bit);
  btf_32_avx512_type1(cospi[11], cospi[53], &x1[43], &x1[52], _r, cos_bit);
  btf_32_avx512_type1(cospi[51], cospi[13], &x1[44], &x1[51], _r, cos_bit);
  btf_32_avx512_type1(cospi[19], cospi[45], &x1[45], &x1[50], _r, cos_bit);
  btf_32_avx512_type1(cospi[35], cospi[29], &x1[46], &x1[49], _r, cos_bit);
  btf_32_avx512_type1(cospi[3], cospi[61], &x1[47], &x1[48], _r, cos_bit);

  static const uint8_t kBitRev64[64] = {
    0, 32, 16, 48, 8,  40, 24, 56, 4, 36, 20, 52, 12, 44, 28, 60,
    2, 34, 18, 50, 10, 42, 26, 58, 6, 38, 22, 54, 14, 46, 30, 62,
    1, 33, 17, 49, 9,  41, 25, 57, 5, 37, 21, 53, 13, 45, 29, 61,
    3, 35, 19, 51, 11, 43, 27, 59, 7, 39, 23, 55, 15, 47, 31, 63
  };
  for (int i = 0; i < 64; ++i) {
    output[i] = x1[kBitRev64[i]];
  }
}

typedef void (*transform_1d_avx512)(const __m512i *input, __m512i *output,
                                    int8_t cos_bit);

static const transform_1d_avx512 col_txfm32x32_arr[TX_TYPES] = {
  fdct32x32_avx512,       // DCT_DCT
  NULL,                   // ADST_DCT
  NULL,                   // DCT_ADST
  NULL,                   // ADST_ADST
  NULL,                   // FLIPADST_DCT
  NULL,                   // DCT_FLIPADST
  NULL,                   // FLIPADST_FLIPADST
  NULL,                   // ADST_FLIPADST
  NULL,                   // FLIPADST_ADST
  fidentity32x32_avx512,  // IDTX
  fdct32x32_avx512,       // V_DCT
  fidentity32x32_avx512,  // H_DCT
  NULL,                   // V_ADST
  NULL,                   // H_ADST
  NULL,                   // V_FLIPADST
  NULL                    // H_FLIPADST
};

static const transform_1d_avx512 row_txfm32x32_arr[TX_TYPES] = {
  fdct32x32_avx512,       // DCT_DCT
  NULL,                   // ADST_DCT
  NULL,                   // DCT_ADST
  NULL,                   // ADST_ADST
  NULL,                   // FLIPADST_DCT
  NULL,                   // DCT_FLIPADST
  NULL,                   // FLIPADST_FLIPADST
  NULL,                   // ADST_FLIPADST
  NULL,                   // FLIPADST_ADST
  fidentity32x32_avx512,  // IDTX
  fidentity32x32_avx512,  // V_DCT
  fdct32x32_avx512,       // H_DCT
  NULL,                   // V_ADST
  NULL,                   // H_ADST
  NULL,                   // V_FLIPADST
  NULL                    // H_FLIPADST
};

static const transform_1d_avx512 col_txfm32x16_arr[TX_TYPES] = {
  fdct32x16_new_avx512,       // DCT_DCT
  fadst32x16_new_avx512,      // ADST_DCT
  fdct32x16_new_avx512,       // DCT_ADST
  fadst32x16_new_avx512,      // ADST_ADST
  fadst32x16_new_avx512,      // FLIPADST_DCT
  fdct32x16_new_avx512,       // DCT_FLIPADST
  fadst32x16_new_avx512,      // FLIPADST_FLIPADST
  fadst32x16_new_avx512,      // ADST_FLIPADST
  fadst32x16_new_avx512,      // FLIPADST_ADST
  fidentity32x16_new_avx512,  // IDTX
  fdct32x16_new_avx512,       // V_DCT
  fidentity32x16_new_avx512,  // H_DCT
  fadst32x16_new_avx512,      // V_ADST
  fidentity32x16_new_avx512,  // H_ADST
  fadst32x16_new_avx512,      // V_FLIPADST
  fidentity32x16_new_avx512   // H_FLIPADST
};

static const transform_1d_avx512 row_txfm32x16_arr[TX_TYPES] = {
  fdct32x16_new_avx512,       // DCT_DCT
  fdct32x16_new_avx512,       // ADST_DCT
  fadst32x16_new_avx512,      // DCT_ADST
  fadst32x16_new_avx512,      // ADST_ADST
  fdct32x16_new_avx512,       // FLIPADST_DCT
  fadst32x16_new_avx512,      // DCT_FLIPADST
  fadst32x16_new_avx512,      // FLIPADST_FLIPADST
  fadst32x16_new_avx512,      // ADST_FLIPADST
  fadst32x16_new_avx512,      // FLIPADST_ADST
  fidentity32x16_new_avx512,  // IDTX
  fidentity32x16_new_avx512,  // V_DCT
  fdct32x16_new_avx512,       // H_DCT
  fidentity32x16_new_avx512,  // V_ADST
  fadst32x16_new_avx512,      // H_ADST
  fidentity32x16_new_avx512,  // V_FLIPADST
  fadst32x16_new_avx512       // H_FLIPADST
};

static void lowbd_fwd_txfm2d_16x16_avx512(const int16_t *input, int32_t *output,
                                          int stride, TX_TYPE tx_type, int bd) {
  (void)bd;
  const TX_SIZE tx_size = TX_16X16;
  __m512i buf512[16];
  __m256i buf0[16], buf1[16];
  const int8_t *shift = av1_fwd_txfm_shift_ls[tx_size];
  const int txw_idx = get_txw_idx(tx_size);
  const int txh_idx = get_txh_idx(tx_size);
  const int cos_bit_col = av1_fwd_cos_bit_col[txw_idx][txh_idx];
  const int cos_bit_row = av1_fwd_cos_bit_row[txw_idx][txh_idx];
  const transform_1d_avx512 col_txfm = col_txfm32x16_arr[tx_type];
  const transform_1d_avx512 row_txfm = row_txfm32x16_arr[tx_type];
  int ud_flip, lr_flip;
  get_flip_cfg(tx_type, &ud_flip, &lr_flip);

  for (int i = 0; i < 16; ++i) {
    const int r = ud_flip ? (15 - i) : i;
    buf512[i] = _mm512_castsi256_si512(
        _mm256_loadu_si256((const __m256i *)(input + r * stride)));
  }
  round_shift_16bit_w32_avx512(buf512, 16, shift[0]);
  col_txfm(buf512, buf512, cos_bit_col);
  round_shift_16bit_w32_avx512(buf512, 16, shift[1]);
  for (int i = 0; i < 16; ++i) buf0[i] = _mm512_castsi512_si256(buf512[i]);
  transpose_16bit_16x16_avx2(buf0, buf1);

  for (int i = 0; i < 16; ++i) {
    const int r = lr_flip ? (15 - i) : i;
    buf512[i] = _mm512_castsi256_si512(buf1[r]);
  }
  row_txfm(buf512, buf512, cos_bit_row);
  round_shift_16bit_w32_avx512(buf512, 16, shift[2]);
  for (int i = 0; i < 16; ++i) buf0[i] = _mm512_castsi512_si256(buf512[i]);
  store_buffer_16bit_to_32bit_w16_avx512(buf0, output, 16, 16);
}

static void lowbd_fwd_txfm2d_32x32_avx512(const int16_t *input, int32_t *output,
                                          int stride, TX_TYPE tx_type, int bd) {
  (void)bd;
  const TX_SIZE tx_size = TX_32X32;
  __m512i buf0[32], buf_row[32];
  __m256i buf1[64];
  const int8_t *shift = av1_fwd_txfm_shift_ls[tx_size];
  const int txw_idx = get_txw_idx(tx_size);
  const int txh_idx = get_txh_idx(tx_size);
  const int cos_bit_col = av1_fwd_cos_bit_col[txw_idx][txh_idx];
  const int cos_bit_row = av1_fwd_cos_bit_row[txw_idx][txh_idx];
  const transform_1d_avx512 col_txfm = col_txfm32x32_arr[tx_type];
  const transform_1d_avx512 row_txfm = row_txfm32x32_arr[tx_type];
  int ud_flip, lr_flip;
  get_flip_cfg(tx_type, &ud_flip, &lr_flip);

  if (ud_flip) {
    load_buffer_16bit_to_16bit_w32_flip_avx512(input, stride, buf0, 32);
  } else {
    load_buffer_16bit_to_16bit_w32_avx512(input, stride, buf0, 32);
  }
  round_shift_16bit_w32_avx512(buf0, 32, shift[0]);
  col_txfm(buf0, buf0, cos_bit_col);
  round_shift_16bit_w32_avx512(buf0, 32, shift[1]);
  transpose_two_16bit_16x16_avx512(buf0 + 0, buf1 + 0, buf1 + 16);
  transpose_two_16bit_16x16_avx512(buf0 + 16, buf1 + 32, buf1 + 48);

  for (int i = 0; i < 32; ++i) {
    const int idx = lr_flip ? (31 - i) : i;
    buf_row[i] = _mm512_inserti64x4(_mm512_castsi256_si512(buf1[idx]),
                                    buf1[32 + idx], 1);
  }
  row_txfm(buf_row, buf_row, cos_bit_row);
  round_shift_16bit_w32_avx512(buf_row, 32, shift[2]);
  store_two_buffers_16bit_to_32bit_w16_avx512(buf_row, output, output + 16, 32,
                                              32);
}

static void lowbd_fwd_txfm2d_64x64_avx512(const int16_t *input, int32_t *output,
                                          int stride, TX_TYPE tx_type, int bd) {
  (void)bd;
  (void)tx_type;
  assert(tx_type == DCT_DCT);
  const TX_SIZE tx_size = TX_64X64;
  __m512i buf0[64];
  __m256i buf1[128];
  const int8_t *shift = av1_fwd_txfm_shift_ls[tx_size];
  const int txw_idx = get_txw_idx(tx_size);
  const int txh_idx = get_txh_idx(tx_size);
  const int cos_bit_col = av1_fwd_cos_bit_col[txw_idx][txh_idx];
  const int cos_bit_row = av1_fwd_cos_bit_row[txw_idx][txh_idx];

  for (int i = 0; i < 2; ++i) {
    load_buffer_16bit_to_16bit_w32_avx512(input + 32 * i, stride, buf0, 64);
    round_shift_16bit_w32_avx512(buf0, 64, shift[0]);
    fdct32x64_new_avx512(buf0, buf0, cos_bit_col);
    round_shift_16bit_w32_avx512(buf0, 64, shift[1]);
    for (int j = 0; j < 2; ++j) {
      transpose_two_16bit_16x16_avx512(buf0 + j * 16, buf1 + j * 64 + 32 * i,
                                       buf1 + j * 64 + 32 * i + 16);
    }
  }

  for (int i = 0; i < 2; ++i) {
    __m512i buf_row[64];
    const __m256i *src = buf1 + 64 * i;
    for (int j = 0; j < 64; ++j) {
      buf_row[j] = _mm512_cvtepi16_epi32(src[j]);
    }
    fdct64_new_avx512(buf_row, buf_row, cos_bit_row);
    round_shift_array_32_avx512(buf_row, buf_row, 32, -shift[2]);
    store_output_32bit_w16_avx512(output + i * 16, buf_row, 32, 32);
  }
}

static void lowbd_fwd_txfm2d_16x32_avx512(const int16_t *input, int32_t *output,
                                          int stride, TX_TYPE tx_type, int bd) {
  (void)bd;
  const TX_SIZE tx_size = TX_16X32;
  __m512i buf512[32], buf_row[16];
  __m256i buf0[32], buf1[32];
  const int8_t *shift = av1_fwd_txfm_shift_ls[tx_size];
  const int txw_idx = get_txw_idx(tx_size);
  const int txh_idx = get_txh_idx(tx_size);
  const int cos_bit_col = av1_fwd_cos_bit_col[txw_idx][txh_idx];
  const int cos_bit_row = av1_fwd_cos_bit_row[txw_idx][txh_idx];
  const transform_1d_avx512 col_txfm = col_txfm32x32_arr[tx_type];
  const transform_1d_avx512 row_txfm = row_txfm32x16_arr[tx_type];
  int ud_flip, lr_flip;
  get_flip_cfg(tx_type, &ud_flip, &lr_flip);

  for (int i = 0; i < 32; ++i) {
    const int r = ud_flip ? (31 - i) : i;
    buf512[i] = _mm512_castsi256_si512(
        _mm256_loadu_si256((const __m256i *)(input + r * stride)));
  }
  round_shift_16bit_w32_avx512(buf512, 32, shift[0]);
  col_txfm(buf512, buf512, cos_bit_col);
  round_shift_16bit_w32_avx512(buf512, 32, shift[1]);
  for (int i = 0; i < 32; ++i) buf0[i] = _mm512_castsi512_si256(buf512[i]);
  transpose_16bit_16x16_avx2(buf0, buf1);
  transpose_16bit_16x16_avx2(buf0 + 16, buf1 + 16);

  for (int i = 0; i < 16; ++i) {
    const int idx = lr_flip ? (15 - i) : i;
    buf_row[i] = _mm512_inserti64x4(_mm512_castsi256_si512(buf1[idx]),
                                    buf1[16 + idx], 1);
  }
  row_txfm(buf_row, buf_row, cos_bit_row);
  round_shift_16bit_w32_avx512(buf_row, 16, shift[2]);
  store_two_rect_buffers_16bit_to_32bit_w16_avx512(buf_row, output, output + 16,
                                                   32, 16);
}

static void lowbd_fwd_txfm2d_32x16_avx512(const int16_t *input, int32_t *output,
                                          int stride, TX_TYPE tx_type, int bd) {
  (void)bd;
  __m512i buf0[32];
  __m256i buf1[32];
  const int8_t *shift = av1_fwd_txfm_shift_ls[TX_32X16];
  const int txw_idx = get_txw_idx(TX_32X16);
  const int txh_idx = get_txh_idx(TX_32X16);
  const int cos_bit_col = av1_fwd_cos_bit_col[txw_idx][txh_idx];
  const int cos_bit_row = av1_fwd_cos_bit_row[txw_idx][txh_idx];
  const transform_1d_avx512 col_txfm = col_txfm32x16_arr[tx_type];
  const transform_1d_avx512 row_txfm = row_txfm32x32_arr[tx_type];
  int ud_flip, lr_flip;
  get_flip_cfg(tx_type, &ud_flip, &lr_flip);

  if (ud_flip) {
    load_buffer_16bit_to_16bit_w32_flip_avx512(input, stride, buf0, 16);
  } else {
    load_buffer_16bit_to_16bit_w32_avx512(input, stride, buf0, 16);
  }
  round_shift_16bit_w32_avx512(buf0, 16, shift[0]);
  col_txfm(buf0, buf0, cos_bit_col);
  round_shift_16bit_w32_avx512(buf0, 16, shift[1]);
  transpose_two_16bit_16x16_avx512(buf0, buf1, buf1 + 16);

  for (int i = 0; i < 32; ++i) {
    const int idx = lr_flip ? (31 - i) : i;
    buf0[i] = _mm512_castsi256_si512(buf1[idx]);
  }
  row_txfm(buf0, buf0, cos_bit_row);
  round_shift_16bit_w32_avx512(buf0, 32, shift[2]);
  for (int i = 0; i < 32; ++i) buf1[i] = _mm512_castsi512_si256(buf0[i]);
  store_rect_buffer_16bit_to_32bit_w16_avx512(buf1, output, 16, 32);
}

static void lowbd_fwd_txfm2d_64x32_avx512(const int16_t *input, int32_t *output,
                                          int stride, TX_TYPE tx_type, int bd) {
  (void)bd;
  assert(tx_type == DCT_DCT);
  const TX_SIZE tx_size = TX_64X32;
  __m512i buf0[32];
  __m256i buf1[128];
  const int8_t *shift = av1_fwd_txfm_shift_ls[tx_size];
  const int txw_idx = get_txw_idx(tx_size);
  const int txh_idx = get_txh_idx(tx_size);
  const int cos_bit_col = av1_fwd_cos_bit_col[txw_idx][txh_idx];
  const int cos_bit_row = av1_fwd_cos_bit_row[txw_idx][txh_idx];
  const transform_1d_avx512 col_txfm = col_txfm32x32_arr[tx_type];

  for (int i = 0; i < 2; ++i) {
    load_buffer_16bit_to_16bit_w32_avx512(input + 32 * i, stride, buf0, 32);
    round_shift_16bit_w32_avx512(buf0, 32, shift[0]);
    col_txfm(buf0, buf0, cos_bit_col);
    round_shift_16bit_w32_avx512(buf0, 32, shift[1]);
    for (int j = 0; j < 2; ++j) {
      transpose_two_16bit_16x16_avx512(buf0 + j * 16, buf1 + j * 64 + 32 * i,
                                       buf1 + j * 64 + 32 * i + 16);
    }
  }

  for (int i = 0; i < 2; ++i) {
    __m512i buf_row[64];
    const __m256i *src = buf1 + 64 * i;
    for (int j = 0; j < 64; ++j) {
      buf_row[j] = _mm512_cvtepi16_epi32(src[j]);
    }
    fdct64_new_avx512(buf_row, buf_row, cos_bit_row);
    round_shift_rect_array_32_avx512(buf_row, buf_row, 32, -shift[2], NewSqrt2);
    store_output_32bit_w16_avx512(output + i * 16, buf_row, 32, 32);
  }
}

static void lowbd_fwd_txfm2d_32x64_avx512(const int16_t *input, int32_t *output,
                                          int stride, TX_TYPE tx_type, int bd) {
  (void)bd;
  (void)tx_type;
  assert(tx_type == DCT_DCT);
  const TX_SIZE tx_size = TX_32X64;
  __m512i buf0[64];
  __m256i buf1[64];
  const int8_t *shift = av1_fwd_txfm_shift_ls[tx_size];
  const int txw_idx = get_txw_idx(tx_size);
  const int txh_idx = get_txh_idx(tx_size);
  const int cos_bit_col = av1_fwd_cos_bit_col[txw_idx][txh_idx];
  const int cos_bit_row = av1_fwd_cos_bit_row[txw_idx][txh_idx];

  load_buffer_16bit_to_16bit_w32_avx512(input, stride, buf0, 64);
  round_shift_16bit_w32_avx512(buf0, 64, shift[0]);
  fdct32x64_new_avx512(buf0, buf0, cos_bit_col);
  round_shift_16bit_w32_avx512(buf0, 64, shift[1]);
  for (int j = 0; j < 2; ++j) {
    transpose_two_16bit_16x16_avx512(buf0 + j * 16, buf1 + j * 32,
                                     buf1 + j * 32 + 16);
  }

  for (int i = 0; i < 2; ++i) {
    __m512i buf_row[32];
    const __m256i *src = buf1 + 32 * i;
    for (int j = 0; j < 32; ++j) {
      buf_row[j] = _mm512_cvtepi16_epi32(src[j]);
    }
    fdct32_avx512(buf_row, buf_row, cos_bit_row);
    round_shift_rect_array_32_avx512(buf_row, buf_row, 32, -shift[2], NewSqrt2);
    store_output_32bit_w16_avx512(output + i * 16, buf_row, 32, 32);
  }
}

static void lowbd_fwd_txfm2d_16x64_avx512(const int16_t *input, int32_t *output,
                                          int stride, TX_TYPE tx_type, int bd) {
  (void)bd;
  (void)tx_type;
  assert(tx_type == DCT_DCT);
  const TX_SIZE tx_size = TX_16X64;
  __m512i buf512[64], buf_row[16];
  __m256i buf0[32], buf1[32];
  const int8_t *shift = av1_fwd_txfm_shift_ls[tx_size];
  const int txw_idx = get_txw_idx(tx_size);
  const int txh_idx = get_txh_idx(tx_size);
  const int cos_bit_col = av1_fwd_cos_bit_col[txw_idx][txh_idx];
  const int cos_bit_row = av1_fwd_cos_bit_row[txw_idx][txh_idx];

  for (int i = 0; i < 64; ++i) {
    buf512[i] = _mm512_castsi256_si512(
        _mm256_loadu_si256((const __m256i *)(input + i * stride)));
  }
  round_shift_16bit_w32_avx512(buf512, 64, shift[0]);
  fdct32x64_new_avx512(buf512, buf512, cos_bit_col);
  round_shift_16bit_w32_avx512(buf512, 64, shift[1]);
  for (int i = 0; i < 32; ++i) buf0[i] = _mm512_castsi512_si256(buf512[i]);
  transpose_16bit_16x16_avx2(buf0, buf1);
  transpose_16bit_16x16_avx2(buf0 + 16, buf1 + 16);

  for (int i = 0; i < 16; ++i) {
    buf_row[i] =
        _mm512_inserti64x4(_mm512_castsi256_si512(buf1[i]), buf1[16 + i], 1);
  }
  fdct32x16_new_avx512(buf_row, buf_row, cos_bit_row);
  round_shift_16bit_w32_avx512(buf_row, 16, shift[2]);
  store_two_buffers_16bit_to_32bit_w16_avx512(buf_row, output, output + 16, 32,
                                              16);
}

static void lowbd_fwd_txfm2d_64x16_avx512(const int16_t *input, int32_t *output,
                                          int stride, TX_TYPE tx_type, int bd) {
  (void)bd;
  (void)tx_type;
  assert(tx_type == DCT_DCT);
  const TX_SIZE tx_size = TX_64X16;
  __m512i buf0[64];
  __m256i buf1[64];
  const int8_t *shift = av1_fwd_txfm_shift_ls[tx_size];
  const int txw_idx = get_txw_idx(tx_size);
  const int txh_idx = get_txh_idx(tx_size);
  const int cos_bit_col = av1_fwd_cos_bit_col[txw_idx][txh_idx];
  const int cos_bit_row = av1_fwd_cos_bit_row[txw_idx][txh_idx];

  for (int i = 0; i < 2; ++i) {
    load_buffer_16bit_to_16bit_w32_avx512(input + 32 * i, stride, buf0, 16);
    round_shift_16bit_w32_avx512(buf0, 16, shift[0]);
    fdct32x16_new_avx512(buf0, buf0, cos_bit_col);
    round_shift_16bit_w32_avx512(buf0, 16, shift[1]);
    transpose_two_16bit_16x16_avx512(buf0, buf1 + 32 * i, buf1 + 32 * i + 16);
  }

  for (int i = 0; i < 64; ++i) {
    buf0[i] = _mm512_castsi256_si512(buf1[i]);
  }
  fdct32x64_new_avx512(buf0, buf0, cos_bit_row);
  round_shift_16bit_w32_avx512(buf0, 64, shift[2]);
  for (int i = 0; i < 32; ++i) buf1[i] = _mm512_castsi512_si256(buf0[i]);
  store_buffer_16bit_to_32bit_w16_avx512(buf1, output, 16, 32);
  memset(output + 16 * 32, 0, 16 * 32 * sizeof(*output));
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((force_align_arg_pointer))
#endif
void av1_lowbd_fwd_txfm_avx512(const int16_t *src_diff, tran_low_t *coeff,
                               int diff_stride, TxfmParam *txfm_param) {
  volatile __m512i stack_align_guard = _mm512_setzero_si512();
  (void)stack_align_guard;
  if (txfm_param->lossless && txfm_param->tx_size == TX_4X4) {
    av1_lowbd_fwd_txfm_c(src_diff, coeff, diff_stride, txfm_param);
    return;
  }
  switch (txfm_param->tx_size) {
    case TX_16X16:
      lowbd_fwd_txfm2d_16x16_avx512(src_diff, coeff, diff_stride,
                                    txfm_param->tx_type, txfm_param->bd);
      break;
    case TX_32X32:
      lowbd_fwd_txfm2d_32x32_avx512(src_diff, coeff, diff_stride,
                                    txfm_param->tx_type, txfm_param->bd);
      break;
    case TX_64X64:
      lowbd_fwd_txfm2d_64x64_avx512(src_diff, coeff, diff_stride,
                                    txfm_param->tx_type, txfm_param->bd);
      break;
    case TX_16X32:
      lowbd_fwd_txfm2d_16x32_avx512(src_diff, coeff, diff_stride,
                                    txfm_param->tx_type, txfm_param->bd);
      break;
    case TX_32X16:
      lowbd_fwd_txfm2d_32x16_avx512(src_diff, coeff, diff_stride,
                                    txfm_param->tx_type, txfm_param->bd);
      break;
    case TX_32X64:
      lowbd_fwd_txfm2d_32x64_avx512(src_diff, coeff, diff_stride,
                                    txfm_param->tx_type, txfm_param->bd);
      break;
    case TX_64X32:
      lowbd_fwd_txfm2d_64x32_avx512(src_diff, coeff, diff_stride,
                                    txfm_param->tx_type, txfm_param->bd);
      break;
    case TX_16X64:
      lowbd_fwd_txfm2d_16x64_avx512(src_diff, coeff, diff_stride,
                                    txfm_param->tx_type, txfm_param->bd);
      break;
    case TX_64X16:
      lowbd_fwd_txfm2d_64x16_avx512(src_diff, coeff, diff_stride,
                                    txfm_param->tx_type, txfm_param->bd);
      break;
    default:
      av1_lowbd_fwd_txfm_avx2(src_diff, coeff, diff_stride, txfm_param);
      break;
  }
}

#endif  // !CONFIG_HIGHWAY
