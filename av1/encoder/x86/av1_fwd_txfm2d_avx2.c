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

#include "config/av1_rtcd.h"

#include "av1/common/enums.h"
#include "av1/common/av1_txfm.h"
#include "av1/encoder/x86/av1_fwd_txfm_avx2.h"
#include "av1/common/x86/av1_txfm_sse2.h"
#include "av1/encoder/av1_fwd_txfm1d_cfg.h"
#include "av1/encoder/x86/av1_txfm1d_sse4.h"
#include "av1/encoder/x86/av1_fwd_txfm_sse2.h"
#include "aom_dsp/x86/txfm_common_avx2.h"

static inline void fdct16x16_new_avx2(const __m256i *input, __m256i *output,
                                      int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);
  const __m256i _r = _mm256_set1_epi32(1 << (cos_bit - 1));

  __m256i cospi_m32_p32 = pair_set_w16_epi16(-cospi[32], cospi[32]);
  __m256i cospi_p32_p32 = pair_set_w16_epi16(cospi[32], cospi[32]);
  __m256i cospi_p32_m32 = pair_set_w16_epi16(cospi[32], -cospi[32]);
  __m256i cospi_p48_p16 = pair_set_w16_epi16(cospi[48], cospi[16]);
  __m256i cospi_m16_p48 = pair_set_w16_epi16(-cospi[16], cospi[48]);
  __m256i cospi_m48_m16 = pair_set_w16_epi16(-cospi[48], -cospi[16]);
  __m256i cospi_p56_p08 = pair_set_w16_epi16(cospi[56], cospi[8]);
  __m256i cospi_m08_p56 = pair_set_w16_epi16(-cospi[8], cospi[56]);
  __m256i cospi_p24_p40 = pair_set_w16_epi16(cospi[24], cospi[40]);
  __m256i cospi_m40_p24 = pair_set_w16_epi16(-cospi[40], cospi[24]);
  __m256i cospi_p60_p04 = pair_set_w16_epi16(cospi[60], cospi[4]);
  __m256i cospi_m04_p60 = pair_set_w16_epi16(-cospi[4], cospi[60]);
  __m256i cospi_p28_p36 = pair_set_w16_epi16(cospi[28], cospi[36]);
  __m256i cospi_m36_p28 = pair_set_w16_epi16(-cospi[36], cospi[28]);
  __m256i cospi_p44_p20 = pair_set_w16_epi16(cospi[44], cospi[20]);
  __m256i cospi_m20_p44 = pair_set_w16_epi16(-cospi[20], cospi[44]);
  __m256i cospi_p12_p52 = pair_set_w16_epi16(cospi[12], cospi[52]);
  __m256i cospi_m52_p12 = pair_set_w16_epi16(-cospi[52], cospi[12]);

  // stage 1
  __m256i x1[16];
  btf_16_adds_subs_out_avx2(&x1[0], &x1[15], input[0], input[15]);
  btf_16_adds_subs_out_avx2(&x1[1], &x1[14], input[1], input[14]);
  btf_16_adds_subs_out_avx2(&x1[2], &x1[13], input[2], input[13]);
  btf_16_adds_subs_out_avx2(&x1[3], &x1[12], input[3], input[12]);
  btf_16_adds_subs_out_avx2(&x1[4], &x1[11], input[4], input[11]);
  btf_16_adds_subs_out_avx2(&x1[5], &x1[10], input[5], input[10]);
  btf_16_adds_subs_out_avx2(&x1[6], &x1[9], input[6], input[9]);
  btf_16_adds_subs_out_avx2(&x1[7], &x1[8], input[7], input[8]);

  // stage 2
  btf_16_adds_subs_avx2(&x1[0], &x1[7]);
  btf_16_adds_subs_avx2(&x1[1], &x1[6]);
  btf_16_adds_subs_avx2(&x1[2], &x1[5]);
  btf_16_adds_subs_avx2(&x1[3], &x1[4]);
  btf_16_w16_avx2(cospi_m32_p32, cospi_p32_p32, &x1[10], &x1[13], _r, cos_bit);
  btf_16_w16_avx2(cospi_m32_p32, cospi_p32_p32, &x1[11], &x1[12], _r, cos_bit);

  // stage 3
  btf_16_adds_subs_avx2(&x1[0], &x1[3]);
  btf_16_adds_subs_avx2(&x1[1], &x1[2]);
  btf_16_w16_avx2(cospi_m32_p32, cospi_p32_p32, &x1[5], &x1[6], _r, cos_bit);
  btf_16_adds_subs_avx2(&x1[8], &x1[11]);
  btf_16_adds_subs_avx2(&x1[9], &x1[10]);
  btf_16_adds_subs_avx2(&x1[15], &x1[12]);
  btf_16_adds_subs_avx2(&x1[14], &x1[13]);

  // stage 4
  btf_16_w16_avx2(cospi_p32_p32, cospi_p32_m32, &x1[0], &x1[1], _r, cos_bit);
  btf_16_w16_avx2(cospi_p48_p16, cospi_m16_p48, &x1[2], &x1[3], _r, cos_bit);
  btf_16_adds_subs_avx2(&x1[4], &x1[5]);
  btf_16_adds_subs_avx2(&x1[7], &x1[6]);
  btf_16_w16_avx2(cospi_m16_p48, cospi_p48_p16, &x1[9], &x1[14], _r, cos_bit);
  btf_16_w16_avx2(cospi_m48_m16, cospi_m16_p48, &x1[10], &x1[13], _r, cos_bit);

  // stage 5
  btf_16_w16_avx2(cospi_p56_p08, cospi_m08_p56, &x1[4], &x1[7], _r, cos_bit);
  btf_16_w16_avx2(cospi_p24_p40, cospi_m40_p24, &x1[5], &x1[6], _r, cos_bit);
  btf_16_adds_subs_avx2(&x1[8], &x1[9]);
  btf_16_adds_subs_avx2(&x1[11], &x1[10]);
  btf_16_adds_subs_avx2(&x1[12], &x1[13]);
  btf_16_adds_subs_avx2(&x1[15], &x1[14]);

  // stage 6
  btf_16_w16_avx2(cospi_p60_p04, cospi_m04_p60, &x1[8], &x1[15], _r, cos_bit);
  btf_16_w16_avx2(cospi_p28_p36, cospi_m36_p28, &x1[9], &x1[14], _r, cos_bit);
  btf_16_w16_avx2(cospi_p44_p20, cospi_m20_p44, &x1[10], &x1[13], _r, cos_bit);
  btf_16_w16_avx2(cospi_p12_p52, cospi_m52_p12, &x1[11], &x1[12], _r, cos_bit);

  // stage 7
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

static inline void fdct16x32_avx2(const __m256i *input, __m256i *output,
                                  int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);
  const __m256i _r = _mm256_set1_epi32(1 << (cos_bit - 1));

  __m256i cospi_m32_p32 = pair_set_w16_epi16(-cospi[32], cospi[32]);
  __m256i cospi_p32_p32 = pair_set_w16_epi16(cospi[32], cospi[32]);
  __m256i cospi_m16_p48 = pair_set_w16_epi16(-cospi[16], cospi[48]);
  __m256i cospi_p48_p16 = pair_set_w16_epi16(cospi[48], cospi[16]);
  __m256i cospi_m48_m16 = pair_set_w16_epi16(-cospi[48], -cospi[16]);
  __m256i cospi_p32_m32 = pair_set_w16_epi16(cospi[32], -cospi[32]);
  __m256i cospi_p56_p08 = pair_set_w16_epi16(cospi[56], cospi[8]);
  __m256i cospi_m08_p56 = pair_set_w16_epi16(-cospi[8], cospi[56]);
  __m256i cospi_p24_p40 = pair_set_w16_epi16(cospi[24], cospi[40]);
  __m256i cospi_m40_p24 = pair_set_w16_epi16(-cospi[40], cospi[24]);
  __m256i cospi_m56_m08 = pair_set_w16_epi16(-cospi[56], -cospi[8]);
  __m256i cospi_m24_m40 = pair_set_w16_epi16(-cospi[24], -cospi[40]);
  __m256i cospi_p60_p04 = pair_set_w16_epi16(cospi[60], cospi[4]);
  __m256i cospi_m04_p60 = pair_set_w16_epi16(-cospi[4], cospi[60]);
  __m256i cospi_p28_p36 = pair_set_w16_epi16(cospi[28], cospi[36]);
  __m256i cospi_m36_p28 = pair_set_w16_epi16(-cospi[36], cospi[28]);
  __m256i cospi_p44_p20 = pair_set_w16_epi16(cospi[44], cospi[20]);
  __m256i cospi_m20_p44 = pair_set_w16_epi16(-cospi[20], cospi[44]);
  __m256i cospi_p12_p52 = pair_set_w16_epi16(cospi[12], cospi[52]);
  __m256i cospi_m52_p12 = pair_set_w16_epi16(-cospi[52], cospi[12]);
  __m256i cospi_p62_p02 = pair_set_w16_epi16(cospi[62], cospi[2]);
  __m256i cospi_m02_p62 = pair_set_w16_epi16(-cospi[2], cospi[62]);
  __m256i cospi_p30_p34 = pair_set_w16_epi16(cospi[30], cospi[34]);
  __m256i cospi_m34_p30 = pair_set_w16_epi16(-cospi[34], cospi[30]);
  __m256i cospi_p46_p18 = pair_set_w16_epi16(cospi[46], cospi[18]);
  __m256i cospi_m18_p46 = pair_set_w16_epi16(-cospi[18], cospi[46]);
  __m256i cospi_p14_p50 = pair_set_w16_epi16(cospi[14], cospi[50]);
  __m256i cospi_m50_p14 = pair_set_w16_epi16(-cospi[50], cospi[14]);
  __m256i cospi_p54_p10 = pair_set_w16_epi16(cospi[54], cospi[10]);
  __m256i cospi_m10_p54 = pair_set_w16_epi16(-cospi[10], cospi[54]);
  __m256i cospi_p22_p42 = pair_set_w16_epi16(cospi[22], cospi[42]);
  __m256i cospi_m42_p22 = pair_set_w16_epi16(-cospi[42], cospi[22]);
  __m256i cospi_p38_p26 = pair_set_w16_epi16(cospi[38], cospi[26]);
  __m256i cospi_m26_p38 = pair_set_w16_epi16(-cospi[26], cospi[38]);
  __m256i cospi_p06_p58 = pair_set_w16_epi16(cospi[6], cospi[58]);
  __m256i cospi_m58_p06 = pair_set_w16_epi16(-cospi[58], cospi[6]);

  // stage 1
  __m256i x1[32];
  btf_16_adds_subs_out_avx2(&x1[0], &x1[31], input[0], input[31]);
  btf_16_adds_subs_out_avx2(&x1[1], &x1[30], input[1], input[30]);
  btf_16_adds_subs_out_avx2(&x1[2], &x1[29], input[2], input[29]);
  btf_16_adds_subs_out_avx2(&x1[3], &x1[28], input[3], input[28]);
  btf_16_adds_subs_out_avx2(&x1[4], &x1[27], input[4], input[27]);
  btf_16_adds_subs_out_avx2(&x1[5], &x1[26], input[5], input[26]);
  btf_16_adds_subs_out_avx2(&x1[6], &x1[25], input[6], input[25]);
  btf_16_adds_subs_out_avx2(&x1[7], &x1[24], input[7], input[24]);
  btf_16_adds_subs_out_avx2(&x1[8], &x1[23], input[8], input[23]);
  btf_16_adds_subs_out_avx2(&x1[9], &x1[22], input[9], input[22]);
  btf_16_adds_subs_out_avx2(&x1[10], &x1[21], input[10], input[21]);
  btf_16_adds_subs_out_avx2(&x1[11], &x1[20], input[11], input[20]);
  btf_16_adds_subs_out_avx2(&x1[12], &x1[19], input[12], input[19]);
  btf_16_adds_subs_out_avx2(&x1[13], &x1[18], input[13], input[18]);
  btf_16_adds_subs_out_avx2(&x1[14], &x1[17], input[14], input[17]);
  btf_16_adds_subs_out_avx2(&x1[15], &x1[16], input[15], input[16]);

  // stage 2
  btf_16_adds_subs_avx2(&x1[0], &x1[15]);
  btf_16_adds_subs_avx2(&x1[1], &x1[14]);
  btf_16_adds_subs_avx2(&x1[2], &x1[13]);
  btf_16_adds_subs_avx2(&x1[3], &x1[12]);
  btf_16_adds_subs_avx2(&x1[4], &x1[11]);
  btf_16_adds_subs_avx2(&x1[5], &x1[10]);
  btf_16_adds_subs_avx2(&x1[6], &x1[9]);
  btf_16_adds_subs_avx2(&x1[7], &x1[8]);
  btf_16_w16_avx2(cospi_m32_p32, cospi_p32_p32, &x1[20], &x1[27], _r, cos_bit);
  btf_16_w16_avx2(cospi_m32_p32, cospi_p32_p32, &x1[21], &x1[26], _r, cos_bit);
  btf_16_w16_avx2(cospi_m32_p32, cospi_p32_p32, &x1[22], &x1[25], _r, cos_bit);
  btf_16_w16_avx2(cospi_m32_p32, cospi_p32_p32, &x1[23], &x1[24], _r, cos_bit);

  // stage 3
  btf_16_adds_subs_avx2(&x1[0], &x1[7]);
  btf_16_adds_subs_avx2(&x1[1], &x1[6]);
  btf_16_adds_subs_avx2(&x1[2], &x1[5]);
  btf_16_adds_subs_avx2(&x1[3], &x1[4]);
  btf_16_w16_avx2(cospi_m32_p32, cospi_p32_p32, &x1[10], &x1[13], _r, cos_bit);
  btf_16_w16_avx2(cospi_m32_p32, cospi_p32_p32, &x1[11], &x1[12], _r, cos_bit);
  btf_16_adds_subs_avx2(&x1[16], &x1[23]);
  btf_16_adds_subs_avx2(&x1[17], &x1[22]);
  btf_16_adds_subs_avx2(&x1[18], &x1[21]);
  btf_16_adds_subs_avx2(&x1[19], &x1[20]);
  btf_16_adds_subs_avx2(&x1[31], &x1[24]);
  btf_16_adds_subs_avx2(&x1[30], &x1[25]);
  btf_16_adds_subs_avx2(&x1[29], &x1[26]);
  btf_16_adds_subs_avx2(&x1[28], &x1[27]);

  // stage 4
  btf_16_adds_subs_avx2(&x1[0], &x1[3]);
  btf_16_adds_subs_avx2(&x1[1], &x1[2]);
  btf_16_w16_avx2(cospi_m32_p32, cospi_p32_p32, &x1[5], &x1[6], _r, cos_bit);
  btf_16_adds_subs_avx2(&x1[8], &x1[11]);
  btf_16_adds_subs_avx2(&x1[9], &x1[10]);
  btf_16_adds_subs_avx2(&x1[15], &x1[12]);
  btf_16_adds_subs_avx2(&x1[14], &x1[13]);
  btf_16_w16_avx2(cospi_m16_p48, cospi_p48_p16, &x1[18], &x1[29], _r, cos_bit);
  btf_16_w16_avx2(cospi_m16_p48, cospi_p48_p16, &x1[19], &x1[28], _r, cos_bit);
  btf_16_w16_avx2(cospi_m48_m16, cospi_m16_p48, &x1[20], &x1[27], _r, cos_bit);
  btf_16_w16_avx2(cospi_m48_m16, cospi_m16_p48, &x1[21], &x1[26], _r, cos_bit);

  // stage 5
  btf_16_w16_avx2(cospi_p32_p32, cospi_p32_m32, &x1[0], &x1[1], _r, cos_bit);
  btf_16_w16_avx2(cospi_p48_p16, cospi_m16_p48, &x1[2], &x1[3], _r, cos_bit);
  btf_16_adds_subs_avx2(&x1[4], &x1[5]);
  btf_16_adds_subs_avx2(&x1[7], &x1[6]);
  btf_16_w16_avx2(cospi_m16_p48, cospi_p48_p16, &x1[9], &x1[14], _r, cos_bit);
  btf_16_w16_avx2(cospi_m48_m16, cospi_m16_p48, &x1[10], &x1[13], _r, cos_bit);
  btf_16_adds_subs_avx2(&x1[16], &x1[19]);
  btf_16_adds_subs_avx2(&x1[17], &x1[18]);
  btf_16_adds_subs_avx2(&x1[23], &x1[20]);
  btf_16_adds_subs_avx2(&x1[22], &x1[21]);
  btf_16_adds_subs_avx2(&x1[24], &x1[27]);
  btf_16_adds_subs_avx2(&x1[25], &x1[26]);
  btf_16_adds_subs_avx2(&x1[31], &x1[28]);
  btf_16_adds_subs_avx2(&x1[30], &x1[29]);

  // stage 6
  btf_16_w16_avx2(cospi_p56_p08, cospi_m08_p56, &x1[4], &x1[7], _r, cos_bit);
  btf_16_w16_avx2(cospi_p24_p40, cospi_m40_p24, &x1[5], &x1[6], _r, cos_bit);
  btf_16_adds_subs_avx2(&x1[8], &x1[9]);
  btf_16_adds_subs_avx2(&x1[11], &x1[10]);
  btf_16_adds_subs_avx2(&x1[12], &x1[13]);
  btf_16_adds_subs_avx2(&x1[15], &x1[14]);
  btf_16_w16_avx2(cospi_m08_p56, cospi_p56_p08, &x1[17], &x1[30], _r, cos_bit);
  btf_16_w16_avx2(cospi_m56_m08, cospi_m08_p56, &x1[18], &x1[29], _r, cos_bit);
  btf_16_w16_avx2(cospi_m40_p24, cospi_p24_p40, &x1[21], &x1[26], _r, cos_bit);
  btf_16_w16_avx2(cospi_m24_m40, cospi_m40_p24, &x1[22], &x1[25], _r, cos_bit);

  // stage 7
  btf_16_w16_avx2(cospi_p60_p04, cospi_m04_p60, &x1[8], &x1[15], _r, cos_bit);
  btf_16_w16_avx2(cospi_p28_p36, cospi_m36_p28, &x1[9], &x1[14], _r, cos_bit);
  btf_16_w16_avx2(cospi_p44_p20, cospi_m20_p44, &x1[10], &x1[13], _r, cos_bit);
  btf_16_w16_avx2(cospi_p12_p52, cospi_m52_p12, &x1[11], &x1[12], _r, cos_bit);
  btf_16_adds_subs_avx2(&x1[16], &x1[17]);
  btf_16_adds_subs_avx2(&x1[19], &x1[18]);
  btf_16_adds_subs_avx2(&x1[20], &x1[21]);
  btf_16_adds_subs_avx2(&x1[23], &x1[22]);
  btf_16_adds_subs_avx2(&x1[24], &x1[25]);
  btf_16_adds_subs_avx2(&x1[27], &x1[26]);
  btf_16_adds_subs_avx2(&x1[28], &x1[29]);
  btf_16_adds_subs_avx2(&x1[31], &x1[30]);

  // stage 8
  btf_16_w16_avx2(cospi_p62_p02, cospi_m02_p62, &x1[16], &x1[31], _r, cos_bit);
  btf_16_w16_avx2(cospi_p30_p34, cospi_m34_p30, &x1[17], &x1[30], _r, cos_bit);
  btf_16_w16_avx2(cospi_p46_p18, cospi_m18_p46, &x1[18], &x1[29], _r, cos_bit);
  btf_16_w16_avx2(cospi_p14_p50, cospi_m50_p14, &x1[19], &x1[28], _r, cos_bit);
  btf_16_w16_avx2(cospi_p54_p10, cospi_m10_p54, &x1[20], &x1[27], _r, cos_bit);
  btf_16_w16_avx2(cospi_p22_p42, cospi_m42_p22, &x1[21], &x1[26], _r, cos_bit);
  btf_16_w16_avx2(cospi_p38_p26, cospi_m26_p38, &x1[22], &x1[25], _r, cos_bit);
  btf_16_w16_avx2(cospi_p06_p58, cospi_m58_p06, &x1[23], &x1[24], _r, cos_bit);

  // stage 9
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

static inline void fadst16x16_new_avx2(const __m256i *input, __m256i *output,
                                       int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);
  const __m256i __zero = _mm256_setzero_si256();
  const __m256i _r = _mm256_set1_epi32(1 << (cos_bit - 1));

  __m256i cospi_p32_p32 = pair_set_w16_epi16(cospi[32], cospi[32]);
  __m256i cospi_p32_m32 = pair_set_w16_epi16(cospi[32], -cospi[32]);
  __m256i cospi_p16_p48 = pair_set_w16_epi16(cospi[16], cospi[48]);
  __m256i cospi_p48_m16 = pair_set_w16_epi16(cospi[48], -cospi[16]);
  __m256i cospi_m48_p16 = pair_set_w16_epi16(-cospi[48], cospi[16]);
  __m256i cospi_p08_p56 = pair_set_w16_epi16(cospi[8], cospi[56]);
  __m256i cospi_p56_m08 = pair_set_w16_epi16(cospi[56], -cospi[8]);
  __m256i cospi_p40_p24 = pair_set_w16_epi16(cospi[40], cospi[24]);
  __m256i cospi_p24_m40 = pair_set_w16_epi16(cospi[24], -cospi[40]);
  __m256i cospi_m56_p08 = pair_set_w16_epi16(-cospi[56], cospi[8]);
  __m256i cospi_m24_p40 = pair_set_w16_epi16(-cospi[24], cospi[40]);
  __m256i cospi_p02_p62 = pair_set_w16_epi16(cospi[2], cospi[62]);
  __m256i cospi_p62_m02 = pair_set_w16_epi16(cospi[62], -cospi[2]);
  __m256i cospi_p10_p54 = pair_set_w16_epi16(cospi[10], cospi[54]);
  __m256i cospi_p54_m10 = pair_set_w16_epi16(cospi[54], -cospi[10]);
  __m256i cospi_p18_p46 = pair_set_w16_epi16(cospi[18], cospi[46]);
  __m256i cospi_p46_m18 = pair_set_w16_epi16(cospi[46], -cospi[18]);
  __m256i cospi_p26_p38 = pair_set_w16_epi16(cospi[26], cospi[38]);
  __m256i cospi_p38_m26 = pair_set_w16_epi16(cospi[38], -cospi[26]);
  __m256i cospi_p34_p30 = pair_set_w16_epi16(cospi[34], cospi[30]);
  __m256i cospi_p30_m34 = pair_set_w16_epi16(cospi[30], -cospi[34]);
  __m256i cospi_p42_p22 = pair_set_w16_epi16(cospi[42], cospi[22]);
  __m256i cospi_p22_m42 = pair_set_w16_epi16(cospi[22], -cospi[42]);
  __m256i cospi_p50_p14 = pair_set_w16_epi16(cospi[50], cospi[14]);
  __m256i cospi_p14_m50 = pair_set_w16_epi16(cospi[14], -cospi[50]);
  __m256i cospi_p58_p06 = pair_set_w16_epi16(cospi[58], cospi[6]);
  __m256i cospi_p06_m58 = pair_set_w16_epi16(cospi[6], -cospi[58]);

  // stage 1
  __m256i x1[16];
  x1[0] = input[0];
  x1[1] = _mm256_subs_epi16(__zero, input[15]);
  x1[2] = _mm256_subs_epi16(__zero, input[7]);
  x1[3] = input[8];
  x1[4] = _mm256_subs_epi16(__zero, input[3]);
  x1[5] = input[12];
  x1[6] = input[4];
  x1[7] = _mm256_subs_epi16(__zero, input[11]);
  x1[8] = _mm256_subs_epi16(__zero, input[1]);
  x1[9] = input[14];
  x1[10] = input[6];
  x1[11] = _mm256_subs_epi16(__zero, input[9]);
  x1[12] = input[2];
  x1[13] = _mm256_subs_epi16(__zero, input[13]);
  x1[14] = _mm256_subs_epi16(__zero, input[5]);
  x1[15] = input[10];

  // stage 2
  btf_16_w16_avx2(cospi_p32_p32, cospi_p32_m32, &x1[2], &x1[3], _r, cos_bit);
  btf_16_w16_avx2(cospi_p32_p32, cospi_p32_m32, &x1[6], &x1[7], _r, cos_bit);
  btf_16_w16_avx2(cospi_p32_p32, cospi_p32_m32, &x1[10], &x1[11], _r, cos_bit);
  btf_16_w16_avx2(cospi_p32_p32, cospi_p32_m32, &x1[14], &x1[15], _r, cos_bit);

  // stage 3
  btf_16_adds_subs_avx2(&x1[0], &x1[2]);
  btf_16_adds_subs_avx2(&x1[1], &x1[3]);
  btf_16_adds_subs_avx2(&x1[4], &x1[6]);
  btf_16_adds_subs_avx2(&x1[5], &x1[7]);
  btf_16_adds_subs_avx2(&x1[8], &x1[10]);
  btf_16_adds_subs_avx2(&x1[9], &x1[11]);
  btf_16_adds_subs_avx2(&x1[12], &x1[14]);
  btf_16_adds_subs_avx2(&x1[13], &x1[15]);

  // stage 4
  btf_16_w16_avx2(cospi_p16_p48, cospi_p48_m16, &x1[4], &x1[5], _r, cos_bit);
  btf_16_w16_avx2(cospi_m48_p16, cospi_p16_p48, &x1[6], &x1[7], _r, cos_bit);
  btf_16_w16_avx2(cospi_p16_p48, cospi_p48_m16, &x1[12], &x1[13], _r, cos_bit);
  btf_16_w16_avx2(cospi_m48_p16, cospi_p16_p48, &x1[14], &x1[15], _r, cos_bit);

  // stage 5
  btf_16_adds_subs_avx2(&x1[0], &x1[4]);
  btf_16_adds_subs_avx2(&x1[1], &x1[5]);
  btf_16_adds_subs_avx2(&x1[2], &x1[6]);
  btf_16_adds_subs_avx2(&x1[3], &x1[7]);
  btf_16_adds_subs_avx2(&x1[8], &x1[12]);
  btf_16_adds_subs_avx2(&x1[9], &x1[13]);
  btf_16_adds_subs_avx2(&x1[10], &x1[14]);
  btf_16_adds_subs_avx2(&x1[11], &x1[15]);

  // stage 6
  btf_16_w16_avx2(cospi_p08_p56, cospi_p56_m08, &x1[8], &x1[9], _r, cos_bit);
  btf_16_w16_avx2(cospi_p40_p24, cospi_p24_m40, &x1[10], &x1[11], _r, cos_bit);
  btf_16_w16_avx2(cospi_m56_p08, cospi_p08_p56, &x1[12], &x1[13], _r, cos_bit);
  btf_16_w16_avx2(cospi_m24_p40, cospi_p40_p24, &x1[14], &x1[15], _r, cos_bit);

  // stage 7
  btf_16_adds_subs_avx2(&x1[0], &x1[8]);
  btf_16_adds_subs_avx2(&x1[1], &x1[9]);
  btf_16_adds_subs_avx2(&x1[2], &x1[10]);
  btf_16_adds_subs_avx2(&x1[3], &x1[11]);
  btf_16_adds_subs_avx2(&x1[4], &x1[12]);
  btf_16_adds_subs_avx2(&x1[5], &x1[13]);
  btf_16_adds_subs_avx2(&x1[6], &x1[14]);
  btf_16_adds_subs_avx2(&x1[7], &x1[15]);

  // stage 8
  btf_16_w16_avx2(cospi_p02_p62, cospi_p62_m02, &x1[0], &x1[1], _r, cos_bit);
  btf_16_w16_avx2(cospi_p10_p54, cospi_p54_m10, &x1[2], &x1[3], _r, cos_bit);
  btf_16_w16_avx2(cospi_p18_p46, cospi_p46_m18, &x1[4], &x1[5], _r, cos_bit);
  btf_16_w16_avx2(cospi_p26_p38, cospi_p38_m26, &x1[6], &x1[7], _r, cos_bit);
  btf_16_w16_avx2(cospi_p34_p30, cospi_p30_m34, &x1[8], &x1[9], _r, cos_bit);
  btf_16_w16_avx2(cospi_p42_p22, cospi_p22_m42, &x1[10], &x1[11], _r, cos_bit);
  btf_16_w16_avx2(cospi_p50_p14, cospi_p14_m50, &x1[12], &x1[13], _r, cos_bit);
  btf_16_w16_avx2(cospi_p58_p06, cospi_p06_m58, &x1[14], &x1[15], _r, cos_bit);

  // stage 9
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

static inline void fidentity16x16_new_avx2(const __m256i *input,
                                           __m256i *output, int8_t cos_bit) {
  (void)cos_bit;
  const __m256i one = _mm256_set1_epi16(1);

  for (int i = 0; i < 16; ++i) {
    const __m256i a_lo = _mm256_unpacklo_epi16(input[i], one);
    const __m256i a_hi = _mm256_unpackhi_epi16(input[i], one);
    const __m256i b_lo = scale_round_avx2(a_lo, 2 * NewSqrt2);
    const __m256i b_hi = scale_round_avx2(a_hi, 2 * NewSqrt2);
    output[i] = _mm256_packs_epi32(b_lo, b_hi);
  }
}

static inline void fidentity16x32_avx2(const __m256i *input, __m256i *output,
                                       int8_t cos_bit) {
  (void)cos_bit;
  for (int i = 0; i < 32; ++i) {
    output[i] = _mm256_slli_epi16(input[i], 2);
  }
}

static inline void store_rect_16bit_to_32bit_avx2(const __m256i a,
                                                  int32_t *const b) {
  const __m256i one = _mm256_set1_epi16(1);
  const __m256i a_reoder = _mm256_permute4x64_epi64(a, 0xd8);
  const __m256i a_lo = _mm256_unpacklo_epi16(a_reoder, one);
  const __m256i a_hi = _mm256_unpackhi_epi16(a_reoder, one);
  const __m256i b_lo = scale_round_avx2(a_lo, NewSqrt2);
  const __m256i b_hi = scale_round_avx2(a_hi, NewSqrt2);
  _mm256_store_si256((__m256i *)b, b_lo);
  _mm256_store_si256((__m256i *)(b + 8), b_hi);
}

static inline void store_rect_buffer_16bit_to_32bit_w16_avx2(
    const __m256i *const in, int32_t *const out, const int stride,
    const int out_size) {
  for (int i = 0; i < out_size; ++i) {
    store_rect_16bit_to_32bit_avx2(in[i], out + i * stride);
  }
}

typedef void (*transform_1d_avx2)(const __m256i *input, __m256i *output,
                                  int8_t cos_bit);

static const transform_1d_avx2 col_txfm16x32_arr[TX_TYPES] = {
  fdct16x32_avx2,       // DCT_DCT
  NULL,                 // ADST_DCT
  NULL,                 // DCT_ADST
  NULL,                 // ADST_ADST
  NULL,                 // FLIPADST_DCT
  NULL,                 // DCT_FLIPADST
  NULL,                 // FLIPADST_FLIPADST
  NULL,                 // ADST_FLIPADST
  NULL,                 // FLIPADST_ADST
  fidentity16x32_avx2,  // IDTX
  fdct16x32_avx2,       // V_DCT
  fidentity16x32_avx2,  // H_DCT
  NULL,                 // V_ADST
  NULL,                 // H_ADST
  NULL,                 // V_FLIPADST
  NULL                  // H_FLIPADST
};

static const transform_1d_avx2 row_txfm16x32_arr[TX_TYPES] = {
  fdct16x32_avx2,       // DCT_DCT
  NULL,                 // ADST_DCT
  NULL,                 // DCT_ADST
  NULL,                 // ADST_ADST
  NULL,                 // FLIPADST_DCT
  NULL,                 // DCT_FLIPADST
  NULL,                 // FLIPADST_FLIPADST
  NULL,                 // ADST_FLIPADST
  NULL,                 // FLIPADST_ADST
  fidentity16x32_avx2,  // IDTX
  fidentity16x32_avx2,  // V_DCT
  fdct16x32_avx2,       // H_DCT
  NULL,                 // V_ADST
  NULL,                 // H_ADST
  NULL,                 // V_FLIPADST
  NULL                  // H_FLIPADST
};

static const transform_1d_avx2 col_txfm16x16_arr[TX_TYPES] = {
  fdct16x16_new_avx2,       // DCT_DCT
  fadst16x16_new_avx2,      // ADST_DCT
  fdct16x16_new_avx2,       // DCT_ADST
  fadst16x16_new_avx2,      // ADST_ADST
  fadst16x16_new_avx2,      // FLIPADST_DCT
  fdct16x16_new_avx2,       // DCT_FLIPADST
  fadst16x16_new_avx2,      // FLIPADST_FLIPADST
  fadst16x16_new_avx2,      // ADST_FLIPADST
  fadst16x16_new_avx2,      // FLIPADST_ADST
  fidentity16x16_new_avx2,  // IDTX
  fdct16x16_new_avx2,       // V_DCT
  fidentity16x16_new_avx2,  // H_DCT
  fadst16x16_new_avx2,      // V_ADST
  fidentity16x16_new_avx2,  // H_ADST
  fadst16x16_new_avx2,      // V_FLIPADST
  fidentity16x16_new_avx2   // H_FLIPADST
};

static const transform_1d_avx2 row_txfm16x16_arr[TX_TYPES] = {
  fdct16x16_new_avx2,       // DCT_DCT
  fdct16x16_new_avx2,       // ADST_DCT
  fadst16x16_new_avx2,      // DCT_ADST
  fadst16x16_new_avx2,      // ADST_ADST
  fdct16x16_new_avx2,       // FLIPADST_DCT
  fadst16x16_new_avx2,      // DCT_FLIPADST
  fadst16x16_new_avx2,      // FLIPADST_FLIPADST
  fadst16x16_new_avx2,      // ADST_FLIPADST
  fadst16x16_new_avx2,      // FLIPADST_ADST
  fidentity16x16_new_avx2,  // IDTX
  fidentity16x16_new_avx2,  // V_DCT
  fdct16x16_new_avx2,       // H_DCT
  fidentity16x16_new_avx2,  // V_ADST
  fadst16x16_new_avx2,      // H_ADST
  fidentity16x16_new_avx2,  // V_FLIPADST
  fadst16x16_new_avx2       // H_FLIPADST
};

static void lowbd_fwd_txfm2d_16x32_avx2(const int16_t *input, int32_t *output,
                                        int stride, TX_TYPE tx_type, int bd) {
  (void)bd;
  const TX_SIZE tx_size = TX_16X32;
  __m256i buf0[32], buf1[32];
  const int8_t *shift = av1_fwd_txfm_shift_ls[tx_size];
  const int txw_idx = get_txw_idx(tx_size);
  const int txh_idx = get_txh_idx(tx_size);
  const int cos_bit_col = av1_fwd_cos_bit_col[txw_idx][txh_idx];
  const int cos_bit_row = av1_fwd_cos_bit_row[txw_idx][txh_idx];
  const int width = tx_size_wide[tx_size];
  const int height = tx_size_high[tx_size];
  const transform_1d_avx2 col_txfm = col_txfm16x32_arr[tx_type];
  const transform_1d_avx2 row_txfm = row_txfm16x16_arr[tx_type];

  int ud_flip, lr_flip;
  get_flip_cfg(tx_type, &ud_flip, &lr_flip);

  if (ud_flip) {
    load_buffer_16bit_to_16bit_flip_avx2(input, stride, buf0, height);
  } else {
    load_buffer_16bit_to_16bit_avx2(input, stride, buf0, height);
  }
  round_shift_16bit_w16_avx2(buf0, height, shift[0]);
  col_txfm(buf0, buf0, cos_bit_col);
  round_shift_16bit_w16_avx2(buf0, height, shift[1]);
  transpose_16bit_16x16_avx2(buf0, buf1);
  transpose_16bit_16x16_avx2(buf0 + 16, buf1 + 16);

  for (int i = 0; i < 2; i++) {
    __m256i *buf;
    if (lr_flip) {
      buf = buf0;
      flip_buf_avx2(buf1 + width * i, buf, width);
    } else {
      buf = buf1 + width * i;
    }
    row_txfm(buf, buf, cos_bit_row);
    round_shift_16bit_w16_avx2(buf, width, shift[2]);
    store_rect_buffer_16bit_to_32bit_w16_avx2(buf, output + i * 16, height,
                                              width);
  }
}

static void lowbd_fwd_txfm2d_32x16_avx2(const int16_t *input, int32_t *output,
                                        int stride, TX_TYPE tx_type, int bd) {
  (void)bd;
  __m256i buf0[32], buf1[64];
  const int8_t *shift = av1_fwd_txfm_shift_ls[TX_32X16];
  const int txw_idx = get_txw_idx(TX_32X16);
  const int txh_idx = get_txh_idx(TX_32X16);
  const int cos_bit_col = av1_fwd_cos_bit_col[txw_idx][txh_idx];
  const int cos_bit_row = av1_fwd_cos_bit_row[txw_idx][txh_idx];
  const int width = 32;
  const int height = 16;
  const transform_1d_avx2 col_txfm = col_txfm16x16_arr[tx_type];
  const transform_1d_avx2 row_txfm = row_txfm16x32_arr[tx_type];

  int ud_flip, lr_flip;
  get_flip_cfg(tx_type, &ud_flip, &lr_flip);

  for (int i = 0; i < 2; i++) {
    if (ud_flip) {
      load_buffer_16bit_to_16bit_flip_avx2(input + 16 * i, stride, buf0,
                                           height);
    } else {
      load_buffer_16bit_to_16bit_avx2(input + 16 * i, stride, buf0, height);
    }
    round_shift_16bit_w16_avx2(buf0, height, shift[0]);
    col_txfm(buf0, buf0, cos_bit_col);
    round_shift_16bit_w16_avx2(buf0, height, shift[1]);
    transpose_16bit_16x16_avx2(buf0, buf1 + 0 * width + 16 * i);
  }

  __m256i *buf;
  if (lr_flip) {
    buf = buf0;
    flip_buf_avx2(buf1, buf, width);
  } else {
    buf = buf1;
  }
  row_txfm(buf, buf, cos_bit_row);
  round_shift_16bit_w16_avx2(buf, width, shift[2]);
  store_rect_buffer_16bit_to_32bit_w16_avx2(buf, output, height, width);
}

static inline void btf_16_avx2(__m256i *w0, __m256i *w1, __m256i *in0,
                               __m256i *in1, __m128i *out0, __m128i *out1,
                               __m128i *out2, __m128i *out3,
                               const __m256i *__rounding, int8_t *cos_bit) {
  __m256i t0 = _mm256_unpacklo_epi16(*in0, *in1);
  __m256i t1 = _mm256_unpackhi_epi16(*in0, *in1);
  __m256i u0 = _mm256_madd_epi16(t0, *w0);
  __m256i u1 = _mm256_madd_epi16(t1, *w0);
  __m256i v0 = _mm256_madd_epi16(t0, *w1);
  __m256i v1 = _mm256_madd_epi16(t1, *w1);

  __m256i a0 = _mm256_add_epi32(u0, *__rounding);
  __m256i a1 = _mm256_add_epi32(u1, *__rounding);
  __m256i b0 = _mm256_add_epi32(v0, *__rounding);
  __m256i b1 = _mm256_add_epi32(v1, *__rounding);

  __m256i c0 = _mm256_srai_epi32(a0, *cos_bit);
  __m256i c1 = _mm256_srai_epi32(a1, *cos_bit);
  __m256i d0 = _mm256_srai_epi32(b0, *cos_bit);
  __m256i d1 = _mm256_srai_epi32(b1, *cos_bit);

  __m256i temp0 = _mm256_packs_epi32(c0, c1);
  __m256i temp1 = _mm256_packs_epi32(d0, d1);

  *out0 = _mm256_castsi256_si128(temp0);
  *out1 = _mm256_castsi256_si128(temp1);
  *out2 = _mm256_extracti128_si256(temp0, 0x01);
  *out3 = _mm256_extracti128_si256(temp1, 0x01);
}

static inline void fdct8x8_new_avx2(const __m256i *input, __m256i *output,
                                    int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);
  const __m256i __rounding = _mm256_set1_epi32(1 << (cos_bit - 1));

  __m256i cospi_m32_p32 = pair_set_w16_epi16(-cospi[32], cospi[32]);
  __m256i cospi_p32_p32 = pair_set_w16_epi16(cospi[32], cospi[32]);
  __m256i cospi_p32_m32 = pair_set_w16_epi16(cospi[32], -cospi[32]);
  __m256i cospi_p48_p16 = pair_set_w16_epi16(cospi[48], cospi[16]);
  __m256i cospi_m16_p48 = pair_set_w16_epi16(-cospi[16], cospi[48]);
  __m256i cospi_p56_p08 = pair_set_w16_epi16(cospi[56], cospi[8]);
  __m256i cospi_m08_p56 = pair_set_w16_epi16(-cospi[8], cospi[56]);
  __m256i cospi_p24_p40 = pair_set_w16_epi16(cospi[24], cospi[40]);
  __m256i cospi_m40_p24 = pair_set_w16_epi16(-cospi[40], cospi[24]);

  // stage 1
  __m256i x1[8];
  x1[0] = _mm256_adds_epi16(input[0], input[7]);
  x1[7] = _mm256_subs_epi16(input[0], input[7]);
  x1[1] = _mm256_adds_epi16(input[1], input[6]);
  x1[6] = _mm256_subs_epi16(input[1], input[6]);
  x1[2] = _mm256_adds_epi16(input[2], input[5]);
  x1[5] = _mm256_subs_epi16(input[2], input[5]);
  x1[3] = _mm256_adds_epi16(input[3], input[4]);
  x1[4] = _mm256_subs_epi16(input[3], input[4]);

  // stage 2
  __m256i x2[8];
  x2[0] = _mm256_adds_epi16(x1[0], x1[3]);
  x2[3] = _mm256_subs_epi16(x1[0], x1[3]);
  x2[1] = _mm256_adds_epi16(x1[1], x1[2]);
  x2[2] = _mm256_subs_epi16(x1[1], x1[2]);
  x2[4] = x1[4];
  btf_16_w16_avx2(cospi_m32_p32, cospi_p32_p32, &x1[5], &x1[6], __rounding,
                  cos_bit);
  x2[5] = x1[5];
  x2[6] = x1[6];
  x2[7] = x1[7];

  // stage 3
  __m256i x3[8];
  btf_16_w16_avx2(cospi_p32_p32, cospi_p32_m32, &x2[0], &x2[1], __rounding,
                  cos_bit);
  x3[0] = x2[0];
  x3[1] = x2[1];
  btf_16_w16_avx2(cospi_p48_p16, cospi_m16_p48, &x2[2], &x2[3], __rounding,
                  cos_bit);
  x3[2] = x2[2];
  x3[3] = x2[3];
  x3[4] = _mm256_adds_epi16(x2[4], x2[5]);
  x3[5] = _mm256_subs_epi16(x2[4], x2[5]);
  x3[6] = _mm256_subs_epi16(x2[7], x2[6]);
  x3[7] = _mm256_adds_epi16(x2[7], x2[6]);

  // stage 4
  __m256i x4[8];
  x4[0] = x3[0];
  x4[1] = x3[1];
  x4[2] = x3[2];
  x4[3] = x3[3];
  btf_16_w16_avx2(cospi_p56_p08, cospi_m08_p56, &x3[4], &x3[7], __rounding,
                  cos_bit);
  x4[4] = x3[4];
  x4[7] = x3[7];
  btf_16_w16_avx2(cospi_p24_p40, cospi_m40_p24, &x3[5], &x3[6], __rounding,
                  cos_bit);
  x4[5] = x3[5];
  x4[6] = x3[6];
  // stage 5
  output[0] = x4[0];
  output[1] = x4[4];
  output[2] = x4[2];
  output[3] = x4[6];
  output[4] = x4[1];
  output[5] = x4[5];
  output[6] = x4[3];
  output[7] = x4[7];
}

static inline void fadst8x8_new_avx2(const __m256i *input, __m256i *output,
                                     int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);
  const __m256i __zero = _mm256_setzero_si256();
  const __m256i __rounding = _mm256_set1_epi32(1 << (cos_bit - 1));

  __m256i cospi_p32_p32 = pair_set_w16_epi16(cospi[32], cospi[32]);
  __m256i cospi_p32_m32 = pair_set_w16_epi16(cospi[32], -cospi[32]);
  __m256i cospi_p16_p48 = pair_set_w16_epi16(cospi[16], cospi[48]);
  __m256i cospi_p48_m16 = pair_set_w16_epi16(cospi[48], -cospi[16]);
  __m256i cospi_m48_p16 = pair_set_w16_epi16(-cospi[48], cospi[16]);
  __m256i cospi_p04_p60 = pair_set_w16_epi16(cospi[4], cospi[60]);
  __m256i cospi_p60_m04 = pair_set_w16_epi16(cospi[60], -cospi[4]);
  __m256i cospi_p20_p44 = pair_set_w16_epi16(cospi[20], cospi[44]);
  __m256i cospi_p44_m20 = pair_set_w16_epi16(cospi[44], -cospi[20]);
  __m256i cospi_p36_p28 = pair_set_w16_epi16(cospi[36], cospi[28]);
  __m256i cospi_p28_m36 = pair_set_w16_epi16(cospi[28], -cospi[36]);
  __m256i cospi_p52_p12 = pair_set_w16_epi16(cospi[52], cospi[12]);
  __m256i cospi_p12_m52 = pair_set_w16_epi16(cospi[12], -cospi[52]);

  // stage 1
  __m256i x1[8];
  x1[0] = input[0];
  x1[1] = _mm256_subs_epi16(__zero, input[7]);
  x1[2] = _mm256_subs_epi16(__zero, input[3]);
  x1[3] = input[4];
  x1[4] = _mm256_subs_epi16(__zero, input[1]);
  x1[5] = input[6];
  x1[6] = input[2];
  x1[7] = _mm256_subs_epi16(__zero, input[5]);

  // stage 2
  __m256i x2[8];
  x2[0] = x1[0];
  x2[1] = x1[1];
  btf_16_w16_avx2(cospi_p32_p32, cospi_p32_m32, &x1[2], &x1[3], __rounding,
                  cos_bit);
  x2[2] = x1[2];
  x2[3] = x1[3];
  x2[4] = x1[4];
  x2[5] = x1[5];
  btf_16_w16_avx2(cospi_p32_p32, cospi_p32_m32, &x1[6], &x1[7], __rounding,
                  cos_bit);
  x2[6] = x1[6];
  x2[7] = x1[7];

  // stage 3
  __m256i x3[8];
  x3[0] = _mm256_adds_epi16(x2[0], x2[2]);
  x3[2] = _mm256_subs_epi16(x2[0], x2[2]);
  x3[1] = _mm256_adds_epi16(x2[1], x2[3]);
  x3[3] = _mm256_subs_epi16(x2[1], x2[3]);
  x3[4] = _mm256_adds_epi16(x2[4], x2[6]);
  x3[6] = _mm256_subs_epi16(x2[4], x2[6]);
  x3[5] = _mm256_adds_epi16(x2[5], x2[7]);
  x3[7] = _mm256_subs_epi16(x2[5], x2[7]);

  // stage 4
  __m256i x4[8];
  x4[0] = x3[0];
  x4[1] = x3[1];
  x4[2] = x3[2];
  x4[3] = x3[3];
  btf_16_w16_avx2(cospi_p16_p48, cospi_p48_m16, &x3[4], &x3[5], __rounding,
                  cos_bit);
  x4[4] = x3[4];
  x4[5] = x3[5];
  btf_16_w16_avx2(cospi_m48_p16, cospi_p16_p48, &x3[6], &x3[7], __rounding,
                  cos_bit);
  x4[6] = x3[6];
  x4[7] = x3[7];

  // stage 5
  __m256i x5[8];
  x5[0] = _mm256_adds_epi16(x4[0], x4[4]);
  x5[4] = _mm256_subs_epi16(x4[0], x4[4]);
  x5[1] = _mm256_adds_epi16(x4[1], x4[5]);
  x5[5] = _mm256_subs_epi16(x4[1], x4[5]);
  x5[2] = _mm256_adds_epi16(x4[2], x4[6]);
  x5[6] = _mm256_subs_epi16(x4[2], x4[6]);
  x5[3] = _mm256_adds_epi16(x4[3], x4[7]);
  x5[7] = _mm256_subs_epi16(x4[3], x4[7]);

  // stage 6
  __m256i x6[8];
  btf_16_w16_avx2(cospi_p04_p60, cospi_p60_m04, &x5[0], &x5[1], __rounding,
                  cos_bit);
  x6[0] = x5[0];
  x6[1] = x5[1];
  btf_16_w16_avx2(cospi_p20_p44, cospi_p44_m20, &x5[2], &x5[3], __rounding,
                  cos_bit);
  x6[2] = x5[2];
  x6[3] = x5[3];
  btf_16_w16_avx2(cospi_p36_p28, cospi_p28_m36, &x5[4], &x5[5], __rounding,
                  cos_bit);
  x6[4] = x5[4];
  x6[5] = x5[5];
  btf_16_w16_avx2(cospi_p52_p12, cospi_p12_m52, &x5[6], &x5[7], __rounding,
                  cos_bit);
  x6[6] = x5[6];
  x6[7] = x5[7];

  // stage 7
  output[0] = x6[1];
  output[1] = x6[6];
  output[2] = x6[3];
  output[3] = x6[4];
  output[4] = x6[5];
  output[5] = x6[2];
  output[6] = x6[7];
  output[7] = x6[0];
}

static inline void fidentity8x8_new_avx2(const __m256i *input, __m256i *output,
                                         int8_t cos_bit) {
  (void)cos_bit;

  output[0] = _mm256_adds_epi16(input[0], input[0]);
  output[1] = _mm256_adds_epi16(input[1], input[1]);
  output[2] = _mm256_adds_epi16(input[2], input[2]);
  output[3] = _mm256_adds_epi16(input[3], input[3]);
  output[4] = _mm256_adds_epi16(input[4], input[4]);
  output[5] = _mm256_adds_epi16(input[5], input[5]);
  output[6] = _mm256_adds_epi16(input[6], input[6]);
  output[7] = _mm256_adds_epi16(input[7], input[7]);
}

static inline void fdct8x16_new_avx2(const __m128i *input, __m128i *output,
                                     int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);
  const __m256i __rounding_256 = _mm256_set1_epi32(1 << (cos_bit - 1));
  const __m128i __rounding = _mm_set1_epi32(1 << (cos_bit - 1));
  __m128i temp0, temp1, temp2, temp3;
  __m256i in0, in1;
  __m128i cospi_m32_p32 = pair_set_epi16(-cospi[32], cospi[32]);
  __m128i cospi_p32_p32 = pair_set_epi16(cospi[32], cospi[32]);
  __m128i cospi_p32_m32 = pair_set_epi16(cospi[32], -cospi[32]);
  __m128i cospi_p48_p16 = pair_set_epi16(cospi[48], cospi[16]);
  __m128i cospi_m16_p48 = pair_set_epi16(-cospi[16], cospi[48]);
  __m128i cospi_m48_m16 = pair_set_epi16(-cospi[48], -cospi[16]);
  __m128i cospi_p56_p08 = pair_set_epi16(cospi[56], cospi[8]);
  __m128i cospi_m08_p56 = pair_set_epi16(-cospi[8], cospi[56]);
  __m128i cospi_p24_p40 = pair_set_epi16(cospi[24], cospi[40]);
  __m128i cospi_m40_p24 = pair_set_epi16(-cospi[40], cospi[24]);
  __m128i cospi_p60_p04 = pair_set_epi16(cospi[60], cospi[4]);
  __m128i cospi_m04_p60 = pair_set_epi16(-cospi[4], cospi[60]);
  __m128i cospi_p28_p36 = pair_set_epi16(cospi[28], cospi[36]);
  __m128i cospi_m36_p28 = pair_set_epi16(-cospi[36], cospi[28]);
  __m128i cospi_p44_p20 = pair_set_epi16(cospi[44], cospi[20]);
  __m128i cospi_m20_p44 = pair_set_epi16(-cospi[20], cospi[44]);
  __m128i cospi_p12_p52 = pair_set_epi16(cospi[12], cospi[52]);
  __m128i cospi_m52_p12 = pair_set_epi16(-cospi[52], cospi[12]);

  __m256i cospi_arr[12];

  cospi_arr[0] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_m32_p32),
                                         cospi_m32_p32, 0x1);
  cospi_arr[1] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_p32_p32),
                                         cospi_p32_p32, 0x1);
  cospi_arr[2] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_p32_p32),
                                         cospi_p48_p16, 0x1);
  cospi_arr[3] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_p32_m32),
                                         cospi_m16_p48, 0x1);
  cospi_arr[4] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_m16_p48),
                                         cospi_m48_m16, 0x1);
  cospi_arr[5] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_p48_p16),
                                         cospi_m16_p48, 0x1);
  cospi_arr[6] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_p56_p08),
                                         cospi_p24_p40, 0x1);
  cospi_arr[7] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_m08_p56),
                                         cospi_m40_p24, 0x1);
  cospi_arr[8] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_p60_p04),
                                         cospi_p28_p36, 0x1);
  cospi_arr[9] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_m04_p60),
                                         cospi_m36_p28, 0x1);
  cospi_arr[10] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_p44_p20),
                                          cospi_p12_p52, 0x1);
  cospi_arr[11] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_m20_p44),
                                          cospi_m52_p12, 0x1);

  __m256i x[8];
  x[0] =
      _mm256_insertf128_si256(_mm256_castsi128_si256(input[0]), input[1], 0x1);
  x[1] = _mm256_insertf128_si256(_mm256_castsi128_si256(input[15]), input[14],
                                 0x1);
  x[2] =
      _mm256_insertf128_si256(_mm256_castsi128_si256(input[2]), input[3], 0x1);
  x[3] = _mm256_insertf128_si256(_mm256_castsi128_si256(input[13]), input[12],
                                 0x1);
  x[4] =
      _mm256_insertf128_si256(_mm256_castsi128_si256(input[5]), input[4], 0x1);
  x[5] = _mm256_insertf128_si256(_mm256_castsi128_si256(input[10]), input[11],
                                 0x1);
  x[6] =
      _mm256_insertf128_si256(_mm256_castsi128_si256(input[7]), input[6], 0x1);
  x[7] =
      _mm256_insertf128_si256(_mm256_castsi128_si256(input[8]), input[9], 0x1);

  // stage 1
  __m256i x1[8];
  x1[0] = _mm256_adds_epi16(x[0], x[1]);
  x1[7] = _mm256_subs_epi16(x[0], x[1]);
  x1[1] = _mm256_adds_epi16(x[2], x[3]);
  x1[6] = _mm256_subs_epi16(x[2], x[3]);
  x1[2] = _mm256_adds_epi16(x[4], x[5]);
  x1[5] = _mm256_subs_epi16(x[4], x[5]);
  x1[3] = _mm256_adds_epi16(x[6], x[7]);
  x1[4] = _mm256_subs_epi16(x[6], x[7]);

  // stage 2
  __m256i x2[8];
  x2[0] = _mm256_adds_epi16(x1[0], x1[3]);
  x2[7] = _mm256_subs_epi16(x1[0], x1[3]);
  x2[1] = _mm256_adds_epi16(x1[1], x1[2]);
  x2[6] = _mm256_subs_epi16(x1[1], x1[2]);
  x2[2] = x1[4];
  x2[3] = x1[7];
  btf_16_avx2(&cospi_arr[0], &cospi_arr[1], &x1[5], &x1[6], &temp0, &temp1,
              &temp2, &temp3, &__rounding_256, &cos_bit);
  x2[4] = _mm256_insertf128_si256(_mm256_castsi128_si256(temp2), temp0, 0x1);
  x2[5] = _mm256_insertf128_si256(_mm256_castsi128_si256(temp3), temp1, 0x1);

  // stage 3
  __m256i x3[8];
  x2[1] = _mm256_permute4x64_epi64(x2[1], 0x4e);
  x3[0] = _mm256_adds_epi16(x2[0], x2[1]);
  x3[1] = _mm256_subs_epi16(x2[0], x2[1]);
  x3[2] = _mm256_blend_epi32(x2[7], x2[6], 0xf0);
  btf_16_sse2(cospi_m32_p32, cospi_p32_p32, _mm256_castsi256_si128(x2[6]),
              _mm256_extractf128_si256(x2[7], 0x01), temp0, temp1);
  x3[7] = _mm256_insertf128_si256(_mm256_castsi128_si256(temp1), temp0, 0x1);
  x3[3] = _mm256_adds_epi16(x2[2], x2[4]);
  x3[4] = _mm256_subs_epi16(x2[2], x2[4]);
  x3[5] = _mm256_adds_epi16(x2[3], x2[5]);
  x3[6] = _mm256_subs_epi16(x2[3], x2[5]);

  // stage 4
  __m256i x4[8];
  x4[0] = _mm256_blend_epi32(x3[0], x3[1], 0xf0);
  x4[1] = _mm256_permute2f128_si256(x3[0], x3[1], 0x21);
  btf_16_avx2(&cospi_arr[2], &cospi_arr[3], &x4[0], &x4[1], &output[0],
              &output[8], &output[4], &output[12], &__rounding_256, &cos_bit);
  x4[2] = _mm256_adds_epi16(x3[2], x3[7]);
  x4[3] = _mm256_subs_epi16(x3[2], x3[7]);
  x4[4] = _mm256_permute2f128_si256(x3[3], x3[4], 0x20);
  x4[5] = _mm256_permute2f128_si256(x3[6], x3[5], 0x20);
  in0 = _mm256_permute2f128_si256(x3[3], x3[4], 0x31);
  in1 = _mm256_permute2f128_si256(x3[5], x3[6], 0x31);
  btf_16_avx2(&cospi_arr[4], &cospi_arr[5], &in0, &in1, &temp0, &temp1, &temp2,
              &temp3, &__rounding_256, &cos_bit);

  x4[6] = _mm256_insertf128_si256(_mm256_castsi128_si256(temp0), temp2, 0x1);
  x4[7] = _mm256_insertf128_si256(_mm256_castsi128_si256(temp3), temp1, 0x1);

  // stage 5
  __m256i x5[4];
  in0 = _mm256_permute2f128_si256(x4[2], x4[3], 0x31);
  in1 = _mm256_permute2f128_si256(x4[2], x4[3], 0x20);
  btf_16_avx2(&cospi_arr[6], &cospi_arr[7], &in0, &in1, &output[2], &output[14],
              &output[10], &output[6], &__rounding_256, &cos_bit);
  x5[0] = _mm256_adds_epi16(x4[4], x4[6]);
  x5[1] = _mm256_subs_epi16(x4[4], x4[6]);
  x5[2] = _mm256_adds_epi16(x4[5], x4[7]);
  x5[3] = _mm256_subs_epi16(x4[5], x4[7]);

  // stage 6
  in0 = _mm256_permute2f128_si256(x5[0], x5[1], 0x20);
  in1 = _mm256_permute2f128_si256(x5[2], x5[3], 0x31);
  btf_16_avx2(&cospi_arr[8], &cospi_arr[9], &in0, &in1, &output[1], &output[15],
              &output[9], &output[7], &__rounding_256, &cos_bit);
  in0 = _mm256_permute2f128_si256(x5[1], x5[0], 0x31);
  in1 = _mm256_permute2f128_si256(x5[3], x5[2], 0x20);
  btf_16_avx2(&cospi_arr[10], &cospi_arr[11], &in0, &in1, &output[5],
              &output[11], &output[13], &output[3], &__rounding_256, &cos_bit);
}

static inline void fadst8x16_new_avx2(const __m128i *input, __m128i *output,
                                      int8_t cos_bit) {
  const int32_t *cospi = cospi_arr(cos_bit);
  const __m256i __zero = _mm256_setzero_si256();
  const __m256i __rounding_256 = _mm256_set1_epi32(1 << (cos_bit - 1));
  __m256i in0, in1;
  __m128i temp0, temp1, temp2, temp3;

  __m128i cospi_p32_p32 = pair_set_epi16(cospi[32], cospi[32]);
  __m128i cospi_p32_m32 = pair_set_epi16(cospi[32], -cospi[32]);
  __m128i cospi_p16_p48 = pair_set_epi16(cospi[16], cospi[48]);
  __m128i cospi_p48_m16 = pair_set_epi16(cospi[48], -cospi[16]);
  __m128i cospi_m48_p16 = pair_set_epi16(-cospi[48], cospi[16]);
  __m128i cospi_p08_p56 = pair_set_epi16(cospi[8], cospi[56]);
  __m128i cospi_p56_m08 = pair_set_epi16(cospi[56], -cospi[8]);
  __m128i cospi_p40_p24 = pair_set_epi16(cospi[40], cospi[24]);
  __m128i cospi_p24_m40 = pair_set_epi16(cospi[24], -cospi[40]);
  __m128i cospi_m56_p08 = pair_set_epi16(-cospi[56], cospi[8]);
  __m128i cospi_m24_p40 = pair_set_epi16(-cospi[24], cospi[40]);
  __m128i cospi_p02_p62 = pair_set_epi16(cospi[2], cospi[62]);
  __m128i cospi_p62_m02 = pair_set_epi16(cospi[62], -cospi[2]);
  __m128i cospi_p10_p54 = pair_set_epi16(cospi[10], cospi[54]);
  __m128i cospi_p54_m10 = pair_set_epi16(cospi[54], -cospi[10]);
  __m128i cospi_p18_p46 = pair_set_epi16(cospi[18], cospi[46]);
  __m128i cospi_p46_m18 = pair_set_epi16(cospi[46], -cospi[18]);
  __m128i cospi_p26_p38 = pair_set_epi16(cospi[26], cospi[38]);
  __m128i cospi_p38_m26 = pair_set_epi16(cospi[38], -cospi[26]);
  __m128i cospi_p34_p30 = pair_set_epi16(cospi[34], cospi[30]);
  __m128i cospi_p30_m34 = pair_set_epi16(cospi[30], -cospi[34]);
  __m128i cospi_p42_p22 = pair_set_epi16(cospi[42], cospi[22]);
  __m128i cospi_p22_m42 = pair_set_epi16(cospi[22], -cospi[42]);
  __m128i cospi_p50_p14 = pair_set_epi16(cospi[50], cospi[14]);
  __m128i cospi_p14_m50 = pair_set_epi16(cospi[14], -cospi[50]);
  __m128i cospi_p58_p06 = pair_set_epi16(cospi[58], cospi[6]);
  __m128i cospi_p06_m58 = pair_set_epi16(cospi[6], -cospi[58]);

  __m256i cospi_arr[20];

  cospi_arr[0] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_p32_p32),
                                         cospi_p32_p32, 0x1);
  cospi_arr[1] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_p32_m32),
                                         cospi_p32_m32, 0x1);
  cospi_arr[2] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_p32_p32),
                                         cospi_p32_p32, 0x1);
  cospi_arr[3] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_p32_m32),
                                         cospi_p32_m32, 0x1);
  cospi_arr[4] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_p16_p48),
                                         cospi_m48_p16, 0x1);
  cospi_arr[5] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_p48_m16),
                                         cospi_p16_p48, 0x1);
  cospi_arr[6] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_p16_p48),
                                         cospi_m48_p16, 0x1);
  cospi_arr[7] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_p48_m16),
                                         cospi_p16_p48, 0x1);
  cospi_arr[8] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_p08_p56),
                                         cospi_p40_p24, 0x1);
  cospi_arr[9] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_p56_m08),
                                         cospi_p24_m40, 0x1);
  cospi_arr[10] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_m56_p08),
                                          cospi_m24_p40, 0x1);
  cospi_arr[11] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_p08_p56),
                                          cospi_p40_p24, 0x1);
  cospi_arr[12] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_p02_p62),
                                          cospi_p10_p54, 0x1);
  cospi_arr[13] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_p62_m02),
                                          cospi_p54_m10, 0x1);
  cospi_arr[14] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_p18_p46),
                                          cospi_p26_p38, 0x1);
  cospi_arr[15] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_p46_m18),
                                          cospi_p38_m26, 0x1);
  cospi_arr[16] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_p34_p30),
                                          cospi_p42_p22, 0x1);
  cospi_arr[17] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_p30_m34),
                                          cospi_p22_m42, 0x1);
  cospi_arr[18] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_p50_p14),
                                          cospi_p58_p06, 0x1);
  cospi_arr[19] = _mm256_insertf128_si256(_mm256_castsi128_si256(cospi_p14_m50),
                                          cospi_p06_m58, 0x1);

  __m256i x[8];
  x[0] =
      _mm256_insertf128_si256(_mm256_castsi128_si256(input[0]), input[4], 0x1);
  x[1] =
      _mm256_insertf128_si256(_mm256_castsi128_si256(input[2]), input[6], 0x1);
  x[2] =
      _mm256_insertf128_si256(_mm256_castsi128_si256(input[8]), input[12], 0x1);
  x[3] = _mm256_insertf128_si256(_mm256_castsi128_si256(input[10]), input[14],
                                 0x1);
  x[4] =
      _mm256_insertf128_si256(_mm256_castsi128_si256(input[1]), input[9], 0x1);
  x[5] =
      _mm256_insertf128_si256(_mm256_castsi128_si256(input[3]), input[11], 0x1);
  x[6] =
      _mm256_insertf128_si256(_mm256_castsi128_si256(input[5]), input[13], 0x1);
  x[7] =
      _mm256_insertf128_si256(_mm256_castsi128_si256(input[7]), input[15], 0x1);

  // stage 1
  __m256i x1[8];
  x1[0] = x[0];
  x1[1] = _mm256_subs_epi16(__zero, x[7]);
  x1[2] = x[2];
  x1[3] = _mm256_subs_epi16(__zero, x[5]);
  x1[4] = _mm256_subs_epi16(__zero, x[4]);
  x1[5] = x[3];
  x1[6] = _mm256_subs_epi16(__zero, x[6]);
  x1[7] = x[1];

  // stage 2
  __m256i x2[8];
  x2[0] = _mm256_blend_epi32(x1[0], x1[1], 0xf0);
  x2[3] = _mm256_blend_epi32(x1[3], x1[2], 0xf0);
  x2[4] = _mm256_blend_epi32(x1[4], x1[5], 0xf0);
  x2[7] = _mm256_blend_epi32(x1[7], x1[6], 0xf0);
  in0 = _mm256_blend_epi32(x1[1], x1[0], 0xf0);
  in1 = _mm256_blend_epi32(x1[2], x1[3], 0xf0);
  btf_16_avx2(&cospi_arr[0], &cospi_arr[1], &in0, &in1, &temp0, &temp1, &temp2,
              &temp3, &__rounding_256, &cos_bit);
  x2[1] = _mm256_insertf128_si256(_mm256_castsi128_si256(temp0), temp1, 0x1);
  x2[2] = _mm256_insertf128_si256(_mm256_castsi128_si256(temp2), temp3, 0x1);
  in0 = _mm256_permute2f128_si256(x1[7], x1[6], 0x21);
  in1 = _mm256_permute2f128_si256(x1[4], x1[5], 0x21);
  btf_16_avx2(&cospi_arr[2], &cospi_arr[3], &in0, &in1, &temp0, &temp1, &temp2,
              &temp3, &__rounding_256, &cos_bit);
  x2[5] = _mm256_insertf128_si256(_mm256_castsi128_si256(temp0), temp1, 0x1);
  x2[6] = _mm256_insertf128_si256(_mm256_castsi128_si256(temp2), temp3, 0x1);

  // stage 3
  __m256i x3[8];
  x3[0] = _mm256_adds_epi16(x2[0], x2[1]);
  x3[1] = _mm256_subs_epi16(x2[0], x2[1]);
  x3[2] = _mm256_adds_epi16(x2[3], x2[2]);
  x3[3] = _mm256_subs_epi16(x2[3], x2[2]);
  x3[4] = _mm256_adds_epi16(x2[4], x2[5]);
  x3[5] = _mm256_subs_epi16(x2[4], x2[5]);
  x3[6] = _mm256_adds_epi16(x2[7], x2[6]);
  x3[7] = _mm256_subs_epi16(x2[7], x2[6]);

  // stage 4
  __m256i x4[8];
  x4[0] = x3[0];
  x4[1] = x3[1];
  x4[4] = x3[4];
  x4[5] = x3[5];
  in0 = _mm256_permute2f128_si256(x3[2], x3[3], 0x20);
  in1 = _mm256_permute2f128_si256(x3[2], x3[3], 0x31);
  btf_16_avx2(&cospi_arr[4], &cospi_arr[5], &in0, &in1, &temp0, &temp1, &temp2,
              &temp3, &__rounding_256, &cos_bit);
  x4[2] = _mm256_insertf128_si256(_mm256_castsi128_si256(temp0), temp1, 0x1);
  x4[3] = _mm256_insertf128_si256(_mm256_castsi128_si256(temp2), temp3, 0x1);
  in0 = _mm256_permute2f128_si256(x3[6], x3[7], 0x20);
  in1 = _mm256_permute2f128_si256(x3[6], x3[7], 0x31);
  btf_16_avx2(&cospi_arr[6], &cospi_arr[7], &in0, &in1, &temp0, &temp1, &temp2,
              &temp3, &__rounding_256, &cos_bit);
  x4[6] = _mm256_insertf128_si256(_mm256_castsi128_si256(temp0), temp1, 0x1);
  x4[7] = _mm256_insertf128_si256(_mm256_castsi128_si256(temp2), temp3, 0x1);

  // stage 5
  __m256i x5[8];
  x5[0] = _mm256_adds_epi16(x4[0], x4[2]);
  x5[1] = _mm256_subs_epi16(x4[0], x4[2]);
  x5[2] = _mm256_adds_epi16(x4[1], x4[3]);
  x5[3] = _mm256_subs_epi16(x4[1], x4[3]);
  x5[4] = _mm256_adds_epi16(x4[4], x4[6]);
  x5[5] = _mm256_subs_epi16(x4[4], x4[6]);
  x5[6] = _mm256_adds_epi16(x4[5], x4[7]);
  x5[7] = _mm256_subs_epi16(x4[5], x4[7]);

  // stage 6
  __m256i x6[8];
  x6[0] = x5[0];
  x6[1] = x5[2];
  x6[2] = x5[1];
  x6[3] = x5[3];
  in0 = _mm256_permute2f128_si256(x5[4], x5[6], 0x20);
  in1 = _mm256_permute2f128_si256(x5[4], x5[6], 0x31);
  btf_16_avx2(&cospi_arr[8], &cospi_arr[9], &in0, &in1, &temp0, &temp1, &temp2,
              &temp3, &__rounding_256, &cos_bit);
  x6[4] = _mm256_insertf128_si256(_mm256_castsi128_si256(temp0), temp1, 0x1);
  x6[5] = _mm256_insertf128_si256(_mm256_castsi128_si256(temp2), temp3, 0x1);
  in0 = _mm256_permute2f128_si256(x5[5], x5[7], 0x20);
  in1 = _mm256_permute2f128_si256(x5[5], x5[7], 0x31);
  btf_16_avx2(&cospi_arr[10], &cospi_arr[11], &in0, &in1, &temp0, &temp1,
              &temp2, &temp3, &__rounding_256, &cos_bit);
  x6[6] = _mm256_insertf128_si256(_mm256_castsi128_si256(temp0), temp1, 0x1);
  x6[7] = _mm256_insertf128_si256(_mm256_castsi128_si256(temp2), temp3, 0x1);

  // stage 7
  __m256i x7[8];
  x7[0] = _mm256_adds_epi16(x6[0], x6[4]);
  x7[1] = _mm256_subs_epi16(x6[0], x6[4]);
  x7[2] = _mm256_adds_epi16(x6[1], x6[5]);
  x7[3] = _mm256_subs_epi16(x6[1], x6[5]);
  x7[4] = _mm256_adds_epi16(x6[2], x6[6]);
  x7[5] = _mm256_subs_epi16(x6[2], x6[6]);
  x7[6] = _mm256_adds_epi16(x6[3], x6[7]);
  x7[7] = _mm256_subs_epi16(x6[3], x6[7]);

  // stage 8
  in0 = _mm256_permute2f128_si256(x7[0], x7[2], 0x20);
  in1 = _mm256_permute2f128_si256(x7[0], x7[2], 0x31);
  btf_16_avx2(&cospi_arr[12], &cospi_arr[13], &in0, &in1, &output[15],
              &output[0], &output[13], &output[2], &__rounding_256, &cos_bit);
  in0 = _mm256_permute2f128_si256(x7[4], x7[6], 0x20);
  in1 = _mm256_permute2f128_si256(x7[4], x7[6], 0x31);
  btf_16_avx2(&cospi_arr[14], &cospi_arr[15], &in0, &in1, &output[11],
              &output[4], &output[9], &output[6], &__rounding_256, &cos_bit);
  in0 = _mm256_permute2f128_si256(x7[1], x7[3], 0x20);
  in1 = _mm256_permute2f128_si256(x7[1], x7[3], 0x31);
  btf_16_avx2(&cospi_arr[16], &cospi_arr[17], &in0, &in1, &output[7],
              &output[8], &output[5], &output[10], &__rounding_256, &cos_bit);
  in0 = _mm256_permute2f128_si256(x7[5], x7[7], 0x20);
  in1 = _mm256_permute2f128_si256(x7[5], x7[7], 0x31);
  btf_16_avx2(&cospi_arr[18], &cospi_arr[19], &in0, &in1, &output[3],
              &output[12], &output[1], &output[14], &__rounding_256, &cos_bit);
}

static inline void fidentity8x16_new_avx2(const __m128i *input, __m128i *output,
                                          int8_t cos_bit) {
  (void)cos_bit;
  const __m256i one = _mm256_set1_epi16(1);
  __m256i temp;
  for (int i = 0; i < 16; i += 2) {
    temp = _mm256_insertf128_si256(_mm256_castsi128_si256(input[i]),
                                   input[i + 1], 0x1);
    const __m256i a_lo = _mm256_unpacklo_epi16(temp, one);
    const __m256i a_hi = _mm256_unpackhi_epi16(temp, one);
    const __m256i b_lo = scale_round_avx2(a_lo, 2 * NewSqrt2);
    const __m256i b_hi = scale_round_avx2(a_hi, 2 * NewSqrt2);
    temp = _mm256_packs_epi32(b_lo, b_hi);
    output[i] = _mm256_castsi256_si128(temp);
    output[i + 1] = _mm256_extractf128_si256(temp, 0x1);
  }
}

static const transform_1d_avx2 col_txfm16x8_arr[TX_TYPES] = {
  fdct8x8_new_avx2,       // DCT_DCT
  fadst8x8_new_avx2,      // ADST_DCT
  fdct8x8_new_avx2,       // DCT_ADST
  fadst8x8_new_avx2,      // ADST_ADST
  fadst8x8_new_avx2,      // FLIPADST_DCT
  fdct8x8_new_avx2,       // DCT_FLIPADST
  fadst8x8_new_avx2,      // FLIPADST_FLIPADST
  fadst8x8_new_avx2,      // ADST_FLIPADST
  fadst8x8_new_avx2,      // FLIPADST_ADST
  fidentity8x8_new_avx2,  // IDTX
  fdct8x8_new_avx2,       // V_DCT
  fidentity8x8_new_avx2,  // H_DCT
  fadst8x8_new_avx2,      // V_ADST
  fidentity8x8_new_avx2,  // H_ADST
  fadst8x8_new_avx2,      // V_FLIPADST
  fidentity8x8_new_avx2,  // H_FLIPADST
};

static const transform_1d_sse2 row_txfm16x8_arr[TX_TYPES] = {
  fdct8x16_new_avx2,       // DCT_DCT
  fdct8x16_new_avx2,       // ADST_DCT
  fadst8x16_new_avx2,      // DCT_ADST
  fadst8x16_new_avx2,      // ADST_ADST
  fdct8x16_new_avx2,       // FLIPADST_DCT
  fadst8x16_new_avx2,      // DCT_FLIPADST
  fadst8x16_new_avx2,      // FLIPADST_FLIPADST
  fadst8x16_new_avx2,      // ADST_FLIPADST
  fadst8x16_new_avx2,      // FLIPADST_ADST
  fidentity8x16_new_avx2,  // IDTX
  fidentity8x16_new_avx2,  // V_DCT
  fdct8x16_new_avx2,       // H_DCT
  fidentity8x16_new_avx2,  // V_ADST
  fadst8x16_new_avx2,      // H_ADST
  fidentity8x16_new_avx2,  // V_FLIPADST
  fadst8x16_new_avx2       // H_FLIPADST
};

static void lowbd_fwd_txfm2d_16x8_avx2(const int16_t *input, int32_t *output,
                                       int stride, TX_TYPE tx_type, int bd) {
  (void)bd;
  __m128i buf0[16], buf1[16];
  __m256i buf2[8];
  const int8_t *shift = av1_fwd_txfm_shift_ls[TX_16X8];
  const int txw_idx = get_txw_idx(TX_16X8);
  const int txh_idx = get_txh_idx(TX_16X8);
  const int cos_bit_col = av1_fwd_cos_bit_col[txw_idx][txh_idx];
  const int cos_bit_row = av1_fwd_cos_bit_row[txw_idx][txh_idx];
  const int width = 16;
  const int height = 8;
  const transform_1d_avx2 col_txfm = col_txfm16x8_arr[tx_type];
  const transform_1d_sse2 row_txfm = row_txfm16x8_arr[tx_type];
  __m128i *buf;
  int ud_flip, lr_flip;

  get_flip_cfg(tx_type, &ud_flip, &lr_flip);

  if (ud_flip) {
    load_buffer_16bit_to_16bit_flip(input + 8 * 0, stride, buf0, height);
    load_buffer_16bit_to_16bit_flip(input + 8 * 1, stride, &buf0[8], height);
  } else {
    load_buffer_16bit_to_16bit(input + 8 * 0, stride, buf0, height);
    load_buffer_16bit_to_16bit(input + 8 * 1, stride, &buf0[8], height);
  }
  pack_reg(buf0, &buf0[8], buf2);
  round_shift_16bit_w16_avx2(buf2, height, shift[0]);
  col_txfm(buf2, buf2, cos_bit_col);
  round_shift_16bit_w16_avx2(buf2, height, shift[1]);
  transpose_16bit_16x8_avx2(buf2, buf2);
  extract_reg(buf2, buf1);

  if (lr_flip) {
    buf = buf0;
    flip_buf_sse2(buf1, buf, width);
  } else {
    buf = buf1;
  }
  row_txfm(buf, buf, cos_bit_row);
  round_shift_16bit(buf, width, shift[2]);
  store_rect_buffer_16bit_to_32bit_w8(buf, output, height, width);
}

#define DECLARE_LOWBD_TXFM2D(w, h)                                        \
  extern void av1_lowbd_fwd_txfm2d_##w##x##h##_avx2(                      \
      const int16_t *input, int32_t *output, int stride, TX_TYPE tx_type, \
      int bd);
DECLARE_LOWBD_TXFM2D(4, 4)
DECLARE_LOWBD_TXFM2D(8, 8)
DECLARE_LOWBD_TXFM2D(16, 16)
DECLARE_LOWBD_TXFM2D(32, 32)
DECLARE_LOWBD_TXFM2D(64, 64)
DECLARE_LOWBD_TXFM2D(4, 8)
DECLARE_LOWBD_TXFM2D(8, 16)
DECLARE_LOWBD_TXFM2D(32, 64)
DECLARE_LOWBD_TXFM2D(64, 32)
DECLARE_LOWBD_TXFM2D(4, 16)
DECLARE_LOWBD_TXFM2D(16, 4)
DECLARE_LOWBD_TXFM2D(8, 32)
DECLARE_LOWBD_TXFM2D(32, 8)
DECLARE_LOWBD_TXFM2D(16, 64)
DECLARE_LOWBD_TXFM2D(64, 16)

static FwdTxfm2dFunc fwd_txfm2d_func_ls[TX_SIZES_ALL] = {
  av1_lowbd_fwd_txfm2d_4x4_avx2,    // 4x4 transform
  av1_lowbd_fwd_txfm2d_8x8_avx2,    // 8x8 transform
  av1_lowbd_fwd_txfm2d_16x16_avx2,  // 16x16 transform
  av1_lowbd_fwd_txfm2d_32x32_avx2,  // 32x32 transform
  av1_lowbd_fwd_txfm2d_64x64_avx2,  // 64x64 transform
  av1_lowbd_fwd_txfm2d_4x8_avx2,    // 4x8 transform
  av1_lowbd_fwd_txfm2d_8x4_sse2,    // 8x4 transform
  av1_lowbd_fwd_txfm2d_8x16_avx2,   // 8x16 transform
  lowbd_fwd_txfm2d_16x8_avx2,       // 16x8 transform
  lowbd_fwd_txfm2d_16x32_avx2,      // 16x32 transform
  lowbd_fwd_txfm2d_32x16_avx2,      // 32x16 transform
  av1_lowbd_fwd_txfm2d_32x64_avx2,  // 32x64 transform
  av1_lowbd_fwd_txfm2d_64x32_avx2,  // 64x32 transform
  av1_lowbd_fwd_txfm2d_4x16_avx2,   // 4x16 transform
  av1_lowbd_fwd_txfm2d_16x4_avx2,   // 16x4 transform
  av1_lowbd_fwd_txfm2d_8x32_avx2,   // 8x32 transform
  av1_lowbd_fwd_txfm2d_32x8_avx2,   // 32x8 transform
  av1_lowbd_fwd_txfm2d_16x64_avx2,  // 16x64 transform
  av1_lowbd_fwd_txfm2d_64x16_avx2,  // 64x16 transform
};

void av1_lowbd_fwd_txfm_avx2(const int16_t *src_diff, tran_low_t *coeff,
                             int diff_stride, TxfmParam *txfm_param) {
  FwdTxfm2dFunc fwd_txfm2d_func = fwd_txfm2d_func_ls[txfm_param->tx_size];
  if (txfm_param->lossless && txfm_param->tx_size == TX_4X4) {
    av1_lowbd_fwd_txfm_c(src_diff, coeff, diff_stride, txfm_param);
  } else {
    fwd_txfm2d_func(src_diff, coeff, diff_stride, txfm_param->tx_type,
                    txfm_param->bd);
  }
}
