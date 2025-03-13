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
#include <assert.h>
#include <smmintrin.h> /* SSE4.1 */

#include "aom_dsp/txfm_common.h"
#include "aom_dsp/x86/transpose_sse2.h"
#include "aom_dsp/x86/txfm_common_sse2.h"
#include "aom_ports/mem.h"
#include "av1/common/av1_txfm.h"
#include "av1/common/x86/highbd_txfm_utility_sse4.h"
#include "av1/encoder/av1_fwd_txfm1d_cfg.h"
#include "av1/encoder/x86/av1_txfm1d_sse4.h"
#include "config/aom_config.h"
#include "config/av1_rtcd.h"

void av1_fwht4x4_sse4_1(const int16_t *input, tran_low_t *output, int stride) {
  __m128i in[4];
  in[0] = _mm_loadl_epi64((const __m128i *)(input + 0 * stride));
  in[1] = _mm_loadl_epi64((const __m128i *)(input + 1 * stride));
  in[2] = _mm_loadl_epi64((const __m128i *)(input + 2 * stride));
  in[3] = _mm_loadl_epi64((const __m128i *)(input + 3 * stride));

  // Convert to int32_t.
  __m128i op[4];
  op[0] = _mm_cvtepi16_epi32(in[0]);
  op[1] = _mm_cvtepi16_epi32(in[1]);
  op[2] = _mm_cvtepi16_epi32(in[2]);
  op[3] = _mm_cvtepi16_epi32(in[3]);

  for (int i = 0; i < 2; ++i) {
    __m128i a1 = op[0];
    __m128i b1 = op[1];
    __m128i c1 = op[2];
    __m128i d1 = op[3];
    __m128i e1;

    a1 = _mm_add_epi32(a1, b1);  // a1 += b1
    d1 = _mm_sub_epi32(d1, c1);  // d1 = d1 - c1
    e1 = _mm_sub_epi32(a1, d1);  // e1 = (a1 - d1) >> 1
    e1 = _mm_srai_epi32(e1, 1);
    b1 = _mm_sub_epi32(e1, b1);  // b1 = e1 - b1
    c1 = _mm_sub_epi32(e1, c1);  // c1 = e1 - c1
    a1 = _mm_sub_epi32(a1, c1);  // a1 -= c1
    d1 = _mm_add_epi32(d1, b1);  // d1 += b1

    op[0] = a1;
    op[1] = c1;
    op[2] = d1;
    op[3] = b1;

    if (i == 0) {
      transpose_32bit_4x4(op, op);
    }
  }

  op[0] = _mm_slli_epi32(op[0], UNIT_QUANT_SHIFT);
  op[1] = _mm_slli_epi32(op[1], UNIT_QUANT_SHIFT);
  op[2] = _mm_slli_epi32(op[2], UNIT_QUANT_SHIFT);
  op[3] = _mm_slli_epi32(op[3], UNIT_QUANT_SHIFT);

  _mm_storeu_si128((__m128i *)(output + 0), op[0]);
  _mm_storeu_si128((__m128i *)(output + 4), op[1]);
  _mm_storeu_si128((__m128i *)(output + 8), op[2]);
  _mm_storeu_si128((__m128i *)(output + 12), op[3]);
}
