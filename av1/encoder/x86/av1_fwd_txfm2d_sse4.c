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

#include "config/av1_rtcd.h"

#include "av1/common/enums.h"
#include "av1/common/av1_txfm.h"
#include "av1/common/x86/av1_txfm_sse2.h"
#include "av1/common/x86/highbd_txfm_utility_sse4.h"
#include "av1/encoder/av1_fwd_txfm1d_cfg.h"
#include "av1/encoder/x86/av1_txfm1d_sse4.h"
#include "av1/encoder/x86/av1_fwd_txfm_sse2.h"

static inline void store_output_32bit_w8(int32_t *const out,
                                         const __m128i *const in1,
                                         const __m128i *const in2,
                                         const int stride, const int out_size) {
  for (int i = 0; i < out_size; ++i) {
    _mm_store_si128((__m128i *)(out + stride * i), in1[i]);
    _mm_store_si128((__m128i *)(out + stride * i + 4), in2[i]);
  }
}

static void lowbd_fwd_txfm2d_64x64_sse4_1(const int16_t *input, int32_t *output,
                                          int stride, TX_TYPE tx_type, int bd) {
  (void)bd;
  (void)tx_type;
  assert(tx_type == DCT_DCT);
  const TX_SIZE tx_size = TX_64X64;
  __m128i buf0[64], buf1[512];
  const int8_t *shift = av1_fwd_txfm_shift_ls[tx_size];
  const int txw_idx = get_txw_idx(tx_size);
  const int txh_idx = get_txh_idx(tx_size);
  const int cos_bit_col = av1_fwd_cos_bit_col[txw_idx][txh_idx];
  const int cos_bit_row = av1_fwd_cos_bit_row[txw_idx][txh_idx];
  const int width = tx_size_wide[tx_size];
  const int height = tx_size_high[tx_size];
  const transform_1d_sse2 col_txfm = av1_fdct8x64_new_sse2;
  const int width_div8 = (width >> 3);
  const int height_div8 = (height >> 3);

  for (int i = 0; i < width_div8; i++) {
    load_buffer_16bit_to_16bit(input + 8 * i, stride, buf0, height);
    round_shift_16bit(buf0, height, shift[0]);
    col_txfm(buf0, buf0, cos_bit_col);
    round_shift_16bit(buf0, height, shift[1]);
    for (int j = 0; j < AOMMIN(4, height_div8); ++j) {
      transpose_16bit_8x8(buf0 + j * 8, buf1 + j * width + 8 * i);
    }
  }
  for (int i = 0; i < AOMMIN(4, height_div8); i++) {
    __m128i bufA[64];
    __m128i bufB[64];
    __m128i *buf = buf1 + width * i;
    for (int j = 0; j < width; ++j) {
      bufA[j] = _mm_cvtepi16_epi32(buf[j]);
      bufB[j] = _mm_cvtepi16_epi32(_mm_unpackhi_epi64(buf[j], buf[j]));
    }
    av1_fdct64_sse4_1(bufA, bufA, cos_bit_row, 1, 1);
    av1_fdct64_sse4_1(bufB, bufB, cos_bit_row, 1, 1);
    av1_round_shift_array_32_sse4_1(bufA, bufA, 32, -shift[2]);
    av1_round_shift_array_32_sse4_1(bufB, bufB, 32, -shift[2]);

    store_output_32bit_w8(output + i * 8, bufA, bufB, 32, 32);
  }
}

static void lowbd_fwd_txfm2d_64x32_sse4_1(const int16_t *input, int32_t *output,
                                          int stride, TX_TYPE tx_type, int bd) {
  (void)bd;
  const TX_SIZE tx_size = TX_64X32;
  __m128i buf0[64], buf1[256];
  const int8_t *shift = av1_fwd_txfm_shift_ls[tx_size];
  const int txw_idx = get_txw_idx(tx_size);
  const int txh_idx = get_txh_idx(tx_size);
  const int cos_bit_col = av1_fwd_cos_bit_col[txw_idx][txh_idx];
  const int cos_bit_row = av1_fwd_cos_bit_row[txw_idx][txh_idx];
  const int width = tx_size_wide[tx_size];
  const int height = tx_size_high[tx_size];
  const transform_1d_sse2 col_txfm = col_txfm8x32_arr[tx_type];
  const int width_div8 = (width >> 3);
  const int height_div8 = (height >> 3);

  for (int i = 0; i < width_div8; i++) {
    load_buffer_16bit_to_16bit(input + 8 * i, stride, buf0, height);
    round_shift_16bit(buf0, height, shift[0]);
    col_txfm(buf0, buf0, cos_bit_col);
    round_shift_16bit(buf0, height, shift[1]);
    for (int j = 0; j < AOMMIN(4, height_div8); ++j) {
      transpose_16bit_8x8(buf0 + j * 8, buf1 + j * width + 8 * i);
    }
  }
  assert(tx_type == DCT_DCT);
  for (int i = 0; i < AOMMIN(4, height_div8); i++) {
    __m128i bufA[64];
    __m128i bufB[64];
    __m128i *buf = buf1 + width * i;
    for (int j = 0; j < width; ++j) {
      bufA[j] = _mm_cvtepi16_epi32(buf[j]);
      bufB[j] = _mm_cvtepi16_epi32(_mm_unpackhi_epi64(buf[j], buf[j]));
    }
    av1_fdct64_sse4_1(bufA, bufA, cos_bit_row, 1, 1);
    av1_fdct64_sse4_1(bufB, bufB, cos_bit_row, 1, 1);
    av1_round_shift_rect_array_32_sse4_1(bufA, bufA, 32, -shift[2], NewSqrt2);
    av1_round_shift_rect_array_32_sse4_1(bufB, bufB, 32, -shift[2], NewSqrt2);

    store_output_32bit_w8(output + i * 8, bufA, bufB, 32, 32);
  }
}

static void lowbd_fwd_txfm2d_32x64_sse4_1(const int16_t *input, int32_t *output,
                                          int stride, TX_TYPE tx_type, int bd) {
  (void)bd;
  (void)tx_type;
  assert(tx_type == DCT_DCT);
  const TX_SIZE tx_size = TX_32X64;
  __m128i buf0[64], buf1[256];
  const int8_t *shift = av1_fwd_txfm_shift_ls[tx_size];
  const int txw_idx = get_txw_idx(tx_size);
  const int txh_idx = get_txh_idx(tx_size);
  const int cos_bit_col = av1_fwd_cos_bit_col[txw_idx][txh_idx];
  const int cos_bit_row = av1_fwd_cos_bit_row[txw_idx][txh_idx];
  const int width = tx_size_wide[tx_size];
  const int height = tx_size_high[tx_size];
  const transform_1d_sse2 col_txfm = av1_fdct8x64_new_sse2;
  const int width_div8 = (width >> 3);
  const int height_div8 = (height >> 3);

  for (int i = 0; i < width_div8; i++) {
    load_buffer_16bit_to_16bit(input + 8 * i, stride, buf0, height);
    round_shift_16bit(buf0, height, shift[0]);
    col_txfm(buf0, buf0, cos_bit_col);
    round_shift_16bit(buf0, height, shift[1]);
    for (int j = 0; j < AOMMIN(4, height_div8); ++j) {
      transpose_16bit_8x8(buf0 + j * 8, buf1 + j * width + 8 * i);
    }
  }

  for (int i = 0; i < AOMMIN(4, height_div8); i++) {
    __m128i bufA[32];
    __m128i bufB[32];
    __m128i *buf = buf1 + width * i;
    for (int j = 0; j < width; ++j) {
      bufA[j] = _mm_cvtepi16_epi32(buf[j]);
      bufB[j] = _mm_cvtepi16_epi32(_mm_unpackhi_epi64(buf[j], buf[j]));
    }
    av1_fdct32_sse4_1(bufA, bufA, cos_bit_row, 1);
    av1_fdct32_sse4_1(bufB, bufB, cos_bit_row, 1);
    av1_round_shift_rect_array_32_sse4_1(bufA, bufA, 32, -shift[2], NewSqrt2);
    av1_round_shift_rect_array_32_sse4_1(bufB, bufB, 32, -shift[2], NewSqrt2);

    store_output_32bit_w8(output + i * 8, bufA, bufB, 32, 32);
  }
}

#define DECLARE_LOWBD_TXFM2D(w, h)                                        \
  extern void av1_lowbd_fwd_txfm2d_##w##x##h##_sse4_1(                    \
      const int16_t *input, int32_t *output, int stride, TX_TYPE tx_type, \
      int bd);
DECLARE_LOWBD_TXFM2D(4, 4)
DECLARE_LOWBD_TXFM2D(8, 8)
DECLARE_LOWBD_TXFM2D(16, 16)
DECLARE_LOWBD_TXFM2D(32, 32)
DECLARE_LOWBD_TXFM2D(4, 8)
DECLARE_LOWBD_TXFM2D(8, 16)
DECLARE_LOWBD_TXFM2D(32, 16)
DECLARE_LOWBD_TXFM2D(4, 16)
DECLARE_LOWBD_TXFM2D(16, 4)
DECLARE_LOWBD_TXFM2D(8, 32)
DECLARE_LOWBD_TXFM2D(32, 8)
DECLARE_LOWBD_TXFM2D(64, 16)

static FwdTxfm2dFunc fwd_txfm2d_func_ls[TX_SIZES_ALL] = {
  av1_lowbd_fwd_txfm2d_4x4_sse4_1,    // 4x4 transform
  av1_lowbd_fwd_txfm2d_8x8_sse4_1,    // 8x8 transform
  av1_lowbd_fwd_txfm2d_16x16_sse4_1,  // 16x16 transform
  av1_lowbd_fwd_txfm2d_32x32_sse4_1,  // 32x32 transform
  lowbd_fwd_txfm2d_64x64_sse4_1,      // 64x64 transform
  av1_lowbd_fwd_txfm2d_4x8_sse4_1,    // 4x8 transform
  av1_lowbd_fwd_txfm2d_8x4_sse2,      // 8x4 transform
  av1_lowbd_fwd_txfm2d_8x16_sse4_1,   // 8x16 transform
  av1_lowbd_fwd_txfm2d_16x8_sse2,     // 16x8 transform
  av1_lowbd_fwd_txfm2d_16x32_sse2,    // 16x32 transform
  av1_lowbd_fwd_txfm2d_32x16_sse4_1,  // 32x16 transform
  lowbd_fwd_txfm2d_32x64_sse4_1,      // 32x64 transform
  lowbd_fwd_txfm2d_64x32_sse4_1,      // 64x32 transform
  av1_lowbd_fwd_txfm2d_4x16_sse4_1,   // 4x16 transform
  av1_lowbd_fwd_txfm2d_16x4_sse4_1,   // 16x4 transform
  av1_lowbd_fwd_txfm2d_8x32_sse4_1,   // 8x32 transform
  av1_lowbd_fwd_txfm2d_32x8_sse4_1,   // 32x8 transform
  av1_lowbd_fwd_txfm2d_16x64_sse2,    // 16x64 transform
  av1_lowbd_fwd_txfm2d_64x16_sse4_1,  // 64x16 transform
};

void av1_lowbd_fwd_txfm_sse4_1(const int16_t *src_diff, tran_low_t *coeff,
                               int diff_stride, TxfmParam *txfm_param) {
  FwdTxfm2dFunc fwd_txfm2d_func = fwd_txfm2d_func_ls[txfm_param->tx_size];
  if (txfm_param->lossless && txfm_param->tx_size == TX_4X4) {
    av1_lowbd_fwd_txfm_c(src_diff, coeff, diff_stride, txfm_param);
  } else {
    fwd_txfm2d_func(src_diff, coeff, diff_stride, txfm_param->tx_type,
                    txfm_param->bd);
  }
}
