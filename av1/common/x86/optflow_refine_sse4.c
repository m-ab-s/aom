/*
 * Copyright (c) 2021, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <assert.h>
#include <emmintrin.h>  // SSE2
#include <smmintrin.h>  /* SSE4.1 */

#include "aom/aom_integer.h"
#include "av1/common/blockd.h"
#include "av1/common/reconinter.h"
#include "aom_dsp/x86/synonyms.h"

#if CONFIG_OPTFLOW_REFINEMENT
static INLINE __m128i round_power_of_two_signed_epi16(__m128i v_val_d,
                                                      const __m128i v_bias_d,
                                                      const __m128i ones,
                                                      const int bits) {
  const __m128i v_sign_d = _mm_sign_epi16(ones, v_val_d);
  __m128i reg = _mm_mullo_epi16(v_val_d, v_sign_d);
  reg = _mm_srli_epi16(_mm_adds_epi16(reg, v_bias_d), bits);
  return _mm_mullo_epi16(reg, v_sign_d);
}

static INLINE __m128i round_power_of_two_signed_epi32(__m128i temp1,
                                                      __m128i temp2,
                                                      const __m128i v_bias_d,
                                                      const __m128i ones,
                                                      const int bits) {
  __m128i v_sign_d = _mm_sign_epi32(ones, temp1);
  __m128i reg = _mm_mullo_epi32(temp1, v_sign_d);
  reg = _mm_srli_epi32(_mm_add_epi32(reg, v_bias_d), bits);
  temp1 = _mm_mullo_epi32(reg, v_sign_d);

  v_sign_d = _mm_sign_epi32(ones, temp2);
  reg = _mm_mullo_epi32(temp2, v_sign_d);
  reg = _mm_srli_epi32(_mm_add_epi32(reg, v_bias_d), bits);
  temp2 = _mm_mullo_epi32(reg, v_sign_d);
  return (_mm_packs_epi32(temp1, temp2));
}

void av1_bicubic_grad_interpolation_sse4_1(const int16_t *pred_src,
                                           int16_t *x_grad, int16_t *y_grad,
                                           const int bw, const int bh) {
#if OPFL_BICUBIC_GRAD
  assert(bw % 8 == 0);
  assert(bh % 8 == 0);

  __m128i coeff_bi[4][2];
  coeff_bi[0][0] =
      _mm_set1_epi16((int16_t)coeffs_bicubic[SUBPEL_GRAD_DELTA_BITS][0][0]);
  coeff_bi[0][1] =
      _mm_set1_epi16((int16_t)coeffs_bicubic[SUBPEL_GRAD_DELTA_BITS][1][0]);
  coeff_bi[1][0] =
      _mm_set1_epi16((int16_t)coeffs_bicubic[SUBPEL_GRAD_DELTA_BITS][0][1]);
  coeff_bi[1][1] =
      _mm_set1_epi16((int16_t)coeffs_bicubic[SUBPEL_GRAD_DELTA_BITS][1][1]);
  coeff_bi[2][0] = _mm_insert_epi16(coeff_bi[0][0], 42, 0);
  coeff_bi[2][1] = _mm_insert_epi16(coeff_bi[0][1], -6, 0);
  coeff_bi[3][0] = _mm_insert_epi16(coeff_bi[0][0], 42, 7);
  coeff_bi[3][1] = _mm_insert_epi16(coeff_bi[0][1], -6, 7);
  const __m128i v_bias_d = _mm_set1_epi16((1 << bicubic_bits) >> 1);
  const __m128i ones = _mm_set1_epi16(1);

#if OPFL_DOWNSAMP_QUINCUNX
  __m128i mask_val[2] = { _mm_set_epi16(0, 1, 0, 1, 0, 1, 0, 1),
                          _mm_set_epi16(1, 0, 1, 0, 1, 0, 1, 0) };
#endif
  if (bw < 16) {
    __m128i coeff[2];
    coeff[0] = _mm_insert_epi16(coeff_bi[2][0], 42, 7);
    coeff[1] = _mm_insert_epi16(coeff_bi[2][1], -6, 7);

    for (int col = 0; col < bh; col++) {
      const int is_y_boundary = (col + 1 > bh - 1 || col - 1 < 0);
      const int id_prev1 = AOMMAX(col - 1, 0);
      const int id_prev2 = AOMMAX(col - 2, 0);
      const int id_next1 = AOMMIN(col + 1, bh - 1);
      const int id_next2 = AOMMIN(col + 2, bh - 1);
#if OPFL_DOWNSAMP_QUINCUNX
      __m128i mask = mask_val[col & 0x1];
#endif
      for (int row = 0; row < bw; row += 8) {
        __m128i vpred_next1, vpred_prev1, vpred_next2, vpred_prev2;
        __m128i temp, sub1, sub2;

        // Subtract interpolated pixel at (i, j+delta) by the one at (i,
        // j-delta)
        const int16_t *src = &pred_src[col * bw + row];
        vpred_prev1 =
            _mm_set_epi16(*(src + 6), *(src + 5), *(src + 4), *(src + 3),
                          *(src + 2), *(src + 1), *src, *src);
        vpred_prev2 = _mm_set_epi16(*(src + 5), *(src + 4), *(src + 3),
                                    *(src + 2), *(src + 1), *src, *src, *src);
        vpred_next1 =
            _mm_set_epi16(*(src + 7), *(src + 7), *(src + 6), *(src + 5),
                          *(src + 4), *(src + 3), *(src + 2), *(src + 1));
        vpred_next2 =
            _mm_set_epi16(*(src + 7), *(src + 7), *(src + 7), *(src + 6),
                          *(src + 5), *(src + 4), *(src + 3), *(src + 2));

        sub1 = _mm_sub_epi16(vpred_next1, vpred_prev1);
        sub2 = _mm_sub_epi16(vpred_next2, vpred_prev2);

        temp = _mm_add_epi16(_mm_mullo_epi16(sub1, coeff[0]),
                             _mm_mullo_epi16(sub2, coeff[1]));

#if OPFL_DOWNSAMP_QUINCUNX
        temp = _mm_mullo_epi16(temp, mask);
#endif
        temp =
            round_power_of_two_signed_epi16(temp, v_bias_d, ones, bicubic_bits);

        const int idx = col * bw + row;
        xx_storeu_128(x_grad + idx, temp);

        // Subtract interpolated pixel at (i+delta, j) by the one at (i-delta,
        // j)
        src = pred_src + row;
        vpred_prev1 = xx_loadu_128(src + id_prev1 * bw);
        vpred_prev2 = xx_loadu_128(src + id_prev2 * bw);
        vpred_next1 = xx_loadu_128(src + id_next1 * bw);
        vpred_next2 = xx_loadu_128(src + id_next2 * bw);

        sub1 = _mm_sub_epi16(vpred_next1, vpred_prev1);
        sub2 = _mm_sub_epi16(vpred_next2, vpred_prev2);

        temp = _mm_add_epi16(_mm_mullo_epi16(sub1, coeff_bi[is_y_boundary][0]),
                             _mm_mullo_epi16(sub2, coeff_bi[is_y_boundary][1]));

#if OPFL_DOWNSAMP_QUINCUNX
        temp = _mm_mullo_epi16(temp, mask);
#endif
        temp =
            round_power_of_two_signed_epi16(temp, v_bias_d, ones, bicubic_bits);
        xx_storeu_128(y_grad + idx, temp);
      }
    }
  } else {
    for (int col = 0; col < bh; col++) {
      const int is_y_boundary = (col + 1 > bh - 1 || col - 1 < 0);
      const int id_prev = AOMMAX(col - 1, 0);
      const int id_prev2 = AOMMAX(col - 2, 0);
      const int id_next = AOMMIN(col + 1, bh - 1);
      const int id_next2 = AOMMIN(col + 2, bh - 1);
#if OPFL_DOWNSAMP_QUINCUNX
      __m128i mask = mask_val[col & 0x1];
#endif
      for (int row = 0; row < bw; row += 16) {
        __m128i vpred_next1_1, vpred_prev1_1, vpred_next2_1, vpred_prev2_1;
        __m128i vpred_next1_2, vpred_prev1_2, vpred_next2_2, vpred_prev2_2;
        __m128i temp1, temp2;
        __m128i sub1, sub2, sub3, sub4;

        // Subtract interpolated pixel at (i, j+delta) by the one at (i,
        // j-delta)
        const int16_t *src = &pred_src[col * bw + row];
        if (row - 1 < 0) {
          vpred_prev1_1 =
              _mm_set_epi16(*(src + 6), *(src + 5), *(src + 4), *(src + 3),
                            *(src + 2), *(src + 1), *src, *src);
          vpred_prev2_1 =
              _mm_set_epi16(*(src + 5), *(src + 4), *(src + 3), *(src + 2),
                            *(src + 1), *src, *src, *src);
        } else {
          vpred_prev1_1 = xx_loadu_128((__m128i *)(src - 1));
          vpred_prev2_1 = xx_loadu_128((__m128i *)(src - 2));
        }
        if (row + 16 > bw - 1) {
          vpred_next1_2 =
              _mm_set_epi16(*(src + 15), *(src + 15), *(src + 14), *(src + 13),
                            *(src + 12), *(src + 11), *(src + 10), *(src + 9));
          vpred_next2_2 =
              _mm_set_epi16(*(src + 15), *(src + 15), *(src + 15), *(src + 14),
                            *(src + 13), *(src + 12), *(src + 11), *(src + 10));
        } else {
          vpred_next1_2 = xx_loadu_128(src + 9);
          vpred_next2_2 = xx_loadu_128(src + 10);
        }
        vpred_prev1_2 = xx_loadu_128(src + 7);
        vpred_prev2_2 = xx_loadu_128(src + 6);
        vpred_next1_1 = xx_loadu_128(src + 1);
        vpred_next2_1 = xx_loadu_128(src + 2);

        sub1 = _mm_sub_epi16(vpred_next1_1, vpred_prev1_1);
        sub2 = _mm_sub_epi16(vpred_next2_1, vpred_prev2_1);

        sub3 = _mm_sub_epi16(vpred_next1_2, vpred_prev1_2);
        sub4 = _mm_sub_epi16(vpred_next2_2, vpred_prev2_2);

        const int is_left_boundary = row - 1 < 0 ? 2 : 0;
        const int is_right_boundary = row + 16 > bw - 1 ? 3 : 0;
        temp1 =
            _mm_add_epi16(_mm_mullo_epi16(sub1, coeff_bi[is_left_boundary][0]),
                          _mm_mullo_epi16(sub2, coeff_bi[is_left_boundary][1]));
        temp2 = _mm_add_epi16(
            _mm_mullo_epi16(sub3, coeff_bi[is_right_boundary][0]),
            _mm_mullo_epi16(sub4, coeff_bi[is_right_boundary][1]));

#if OPFL_DOWNSAMP_QUINCUNX
        temp1 = _mm_mullo_epi16(temp1, mask);
        temp2 = _mm_mullo_epi16(temp2, mask);
#endif
        temp1 = round_power_of_two_signed_epi16(temp1, v_bias_d, ones,
                                                bicubic_bits);
        temp2 = round_power_of_two_signed_epi16(temp2, v_bias_d, ones,
                                                bicubic_bits);

        const int idx = col * bw + row;
        xx_storeu_128(x_grad + idx, temp1);
        xx_storeu_128(x_grad + idx + 8, temp2);

        // Subtract interpolated pixel at (i+delta, j) by the one at (i-delta,
        // j)
        src = pred_src + row;
        vpred_prev1_1 = xx_loadu_128(src + bw * id_prev);
        vpred_prev2_1 = xx_loadu_128(src + bw * id_prev2);
        vpred_next1_1 = xx_loadu_128(src + id_next * bw);
        vpred_next2_1 = xx_loadu_128(src + id_next2 * bw);

        vpred_prev1_2 = xx_loadu_128(src + bw * id_prev + 8);
        vpred_prev2_2 = xx_loadu_128(src + bw * id_prev2 + 8);
        vpred_next1_2 = xx_loadu_128(src + id_next * bw + 8);
        vpred_next2_2 = xx_loadu_128(src + id_next2 * bw + 8);

        sub1 = _mm_sub_epi16(vpred_next1_1, vpred_prev1_1);
        sub2 = _mm_sub_epi16(vpred_next2_1, vpred_prev2_1);

        sub3 = _mm_sub_epi16(vpred_next1_2, vpred_prev1_2);
        sub4 = _mm_sub_epi16(vpred_next2_2, vpred_prev2_2);

        temp1 =
            _mm_add_epi16(_mm_mullo_epi16(sub1, coeff_bi[is_y_boundary][0]),
                          _mm_mullo_epi16(sub2, coeff_bi[is_y_boundary][1]));
        temp2 =
            _mm_add_epi16(_mm_mullo_epi16(sub3, coeff_bi[is_y_boundary][0]),
                          _mm_mullo_epi16(sub4, coeff_bi[is_y_boundary][1]));
#if OPFL_DOWNSAMP_QUINCUNX
        temp1 = _mm_mullo_epi16(temp1, mask);
        temp2 = _mm_mullo_epi16(temp2, mask);
#endif
        temp1 = round_power_of_two_signed_epi16(temp1, v_bias_d, ones,
                                                bicubic_bits);
        temp2 = round_power_of_two_signed_epi16(temp2, v_bias_d, ones,
                                                bicubic_bits);

        xx_storeu_128(y_grad + idx, temp1);
        xx_storeu_128(y_grad + idx + 8, temp2);
      }
    }
  }
#else
  (void)pred_src;
  (void)x_grad;
  (void)y_grad;
  (void)bw;
  (void)bh;
#endif  // OPFL_BICUBIC_GRAD
}

void av1_bicubic_grad_interpolation_highbd_sse4_1(const int16_t *pred_src,
                                                  int16_t *x_grad,
                                                  int16_t *y_grad, const int bw,
                                                  const int bh) {
#if OPFL_BICUBIC_GRAD
  assert(bw % 8 == 0);
  assert(bh % 8 == 0);

  __m128i coeff_bi[4][2];
  coeff_bi[0][0] = _mm_set1_epi32(coeffs_bicubic[SUBPEL_GRAD_DELTA_BITS][0][0]);
  coeff_bi[0][1] = _mm_set1_epi32(coeffs_bicubic[SUBPEL_GRAD_DELTA_BITS][1][0]);
  coeff_bi[1][0] = _mm_set1_epi32(coeffs_bicubic[SUBPEL_GRAD_DELTA_BITS][0][1]);
  coeff_bi[1][1] = _mm_set1_epi32(coeffs_bicubic[SUBPEL_GRAD_DELTA_BITS][1][1]);
  coeff_bi[2][0] = _mm_insert_epi32(coeff_bi[0][0], 42, 0);
  coeff_bi[2][1] = _mm_insert_epi32(coeff_bi[0][1], -6, 0);
  coeff_bi[3][0] = _mm_insert_epi32(coeff_bi[0][0], 42, 3);
  coeff_bi[3][1] = _mm_insert_epi32(coeff_bi[0][1], -6, 3);
  const __m128i v_bias_d = _mm_set1_epi32((1 << bicubic_bits) >> 1);
  __m128i zero = _mm_setzero_si128();
  const __m128i ones = _mm_set1_epi32(1);

#if OPFL_DOWNSAMP_QUINCUNX
  __m128i mask_val[2] = { _mm_set_epi32(0, 1, 0, 1),
                          _mm_set_epi32(1, 0, 1, 0) };
#endif

  if (bw < 16) {
    for (int col = 0; col < bh; col++) {
      const int is_y_boundary = (col + 1 > bh - 1 || col - 1 < 0);
      const int id_prev1 = AOMMAX(col - 1, 0);
      const int id_prev2 = AOMMAX(col - 2, 0);
      const int id_next1 = AOMMIN(col + 1, bh - 1);
      const int id_next2 = AOMMIN(col + 2, bh - 1);

#if OPFL_DOWNSAMP_QUINCUNX
      __m128i mask = mask_val[col & 0x1];
#endif
      for (int row = 0; row < bw; row += 8) {
        __m128i vpred_next1, vpred_prev1, vpred_next2, vpred_prev2;
        __m128i temp1, temp2, sub1, sub2, sub3, sub4;
        const int16_t *src = &pred_src[col * bw + row];
        vpred_prev1 =
            _mm_set_epi16(*(src + 6), *(src + 5), *(src + 4), *(src + 3),
                          *(src + 2), *(src + 1), *src, *src);
        vpred_prev2 = _mm_set_epi16(*(src + 5), *(src + 4), *(src + 3),
                                    *(src + 2), *(src + 1), *src, *src, *src);
        vpred_next1 =
            _mm_set_epi16(*(src + 7), *(src + 7), *(src + 6), *(src + 5),
                          *(src + 4), *(src + 3), *(src + 2), *(src + 1));
        vpred_next2 =
            _mm_set_epi16(*(src + 7), *(src + 7), *(src + 7), *(src + 6),
                          *(src + 5), *(src + 4), *(src + 3), *(src + 2));

        sub1 = _mm_sub_epi32(_mm_unpacklo_epi16(vpred_next1, zero),
                             _mm_unpacklo_epi16(vpred_prev1, zero));
        sub2 = _mm_sub_epi32(_mm_unpackhi_epi16(vpred_next1, zero),
                             _mm_unpackhi_epi16(vpred_prev1, zero));
        sub3 = _mm_sub_epi32(_mm_unpacklo_epi16(vpred_next2, zero),
                             _mm_unpacklo_epi16(vpred_prev2, zero));
        sub4 = _mm_sub_epi32(_mm_unpackhi_epi16(vpred_next2, zero),
                             _mm_unpackhi_epi16(vpred_prev2, zero));

        temp1 = _mm_add_epi32(_mm_mullo_epi32(sub1, coeff_bi[2][0]),
                              _mm_mullo_epi32(sub3, coeff_bi[2][1]));
        temp2 = _mm_add_epi32(_mm_mullo_epi32(sub2, coeff_bi[3][0]),
                              _mm_mullo_epi32(sub4, coeff_bi[3][1]));

#if OPFL_DOWNSAMP_QUINCUNX
        temp1 = _mm_mullo_epi32(temp1, mask);
        temp2 = _mm_mullo_epi32(temp2, mask);
#endif
        temp1 = round_power_of_two_signed_epi32(temp1, temp2, v_bias_d, ones,
                                                bicubic_bits);

        const int idx = col * bw + row;
        xx_storeu_128(x_grad + idx, temp1);

        src = pred_src + row;
        vpred_prev1 = xx_loadu_128(src + id_prev1 * bw);
        vpred_prev2 = xx_loadu_128(src + id_prev2 * bw);
        vpred_next1 = xx_loadu_128(src + id_next1 * bw);
        vpred_next2 = xx_loadu_128(src + id_next2 * bw);

        sub1 = _mm_sub_epi32(_mm_unpacklo_epi16(vpred_next1, zero),
                             _mm_unpacklo_epi16(vpred_prev1, zero));
        sub2 = _mm_sub_epi32(_mm_unpackhi_epi16(vpred_next1, zero),
                             _mm_unpackhi_epi16(vpred_prev1, zero));
        sub3 = _mm_sub_epi32(_mm_unpacklo_epi16(vpred_next2, zero),
                             _mm_unpacklo_epi16(vpred_prev2, zero));
        sub4 = _mm_sub_epi32(_mm_unpackhi_epi16(vpred_next2, zero),
                             _mm_unpackhi_epi16(vpred_prev2, zero));

        temp1 =
            _mm_add_epi32(_mm_mullo_epi32(sub1, coeff_bi[is_y_boundary][0]),
                          _mm_mullo_epi32(sub3, coeff_bi[is_y_boundary][1]));
        temp2 =
            _mm_add_epi32(_mm_mullo_epi32(sub2, coeff_bi[is_y_boundary][0]),
                          _mm_mullo_epi32(sub4, coeff_bi[is_y_boundary][1]));

#if OPFL_DOWNSAMP_QUINCUNX
        temp1 = _mm_mullo_epi32(temp1, mask);
        temp2 = _mm_mullo_epi32(temp2, mask);
#endif
        temp1 = round_power_of_two_signed_epi32(temp1, temp2, v_bias_d, ones,
                                                bicubic_bits);
        xx_storeu_128(y_grad + idx, temp1);
      }
    }
  } else {
    for (int col = 0; col < bh; col++) {
      const int is_y_boundary = (col + 1 > bh - 1 || col - 1 < 0);
      const int id_prev = AOMMAX(col - 1, 0);
      const int id_prev2 = AOMMAX(col - 2, 0);
      const int id_next = AOMMIN(col + 1, bh - 1);
      const int id_next2 = AOMMIN(col + 2, bh - 1);
#if OPFL_DOWNSAMP_QUINCUNX
      __m128i mask = mask_val[col & 0x1];
#endif
      for (int row = 0; row < bw; row += 16) {
        __m128i vpred_next1_1, vpred_prev1_1, vpred_next2_1, vpred_prev2_1;
        __m128i vpred_next1_2, vpred_prev1_2, vpred_next2_2, vpred_prev2_2;
        __m128i temp1, temp2;
        __m128i sub1, sub2, sub3, sub4;

        const int16_t *src = &pred_src[col * bw + row];

        if (row - 1 < 0) {
          vpred_prev1_1 =
              _mm_set_epi16(*(src + 6), *(src + 5), *(src + 4), *(src + 3),
                            *(src + 2), *(src + 1), *src, *src);
          vpred_prev2_1 =
              _mm_set_epi16(*(src + 5), *(src + 4), *(src + 3), *(src + 2),
                            *(src + 1), *src, *src, *src);
        } else {
          vpred_prev1_1 = xx_loadu_128((__m128i *)(src - 1));
          vpred_prev2_1 = xx_loadu_128((__m128i *)(src - 2));
        }
        if (row + 16 > bw - 1) {
          vpred_next1_2 =
              _mm_set_epi16(*(src + 15), *(src + 15), *(src + 14), *(src + 13),
                            *(src + 12), *(src + 11), *(src + 10), *(src + 9));
          vpred_next2_2 =
              _mm_set_epi16(*(src + 15), *(src + 15), *(src + 15), *(src + 14),
                            *(src + 13), *(src + 12), *(src + 11), *(src + 10));
        } else {
          vpred_next1_2 = xx_loadu_128(src + 9);
          vpred_next2_2 = xx_loadu_128(src + 10);
        }
        vpred_prev1_2 = xx_loadu_128(src + 7);
        vpred_prev2_2 = xx_loadu_128(src + 6);
        vpred_next1_1 = xx_loadu_128(src + 1);
        vpred_next2_1 = xx_loadu_128(src + 2);

        sub1 = _mm_sub_epi32(_mm_unpacklo_epi16(vpred_next1_1, zero),
                             _mm_unpacklo_epi16(vpred_prev1_1, zero));
        sub2 = _mm_sub_epi32(_mm_unpackhi_epi16(vpred_next1_1, zero),
                             _mm_unpackhi_epi16(vpred_prev1_1, zero));
        sub3 = _mm_sub_epi32(_mm_unpacklo_epi16(vpred_next2_1, zero),
                             _mm_unpacklo_epi16(vpred_prev2_1, zero));
        sub4 = _mm_sub_epi32(_mm_unpackhi_epi16(vpred_next2_1, zero),
                             _mm_unpackhi_epi16(vpred_prev2_1, zero));

        const int is_left_boundary = row - 1 < 0 ? 2 : 0;
        const int is_right_boundary = row + 16 > bw - 1 ? 3 : 0;
        temp1 =
            _mm_add_epi32(_mm_mullo_epi32(sub1, coeff_bi[is_left_boundary][0]),
                          _mm_mullo_epi32(sub3, coeff_bi[is_left_boundary][1]));
        temp2 = _mm_add_epi32(_mm_mullo_epi32(sub2, coeff_bi[0][0]),
                              _mm_mullo_epi32(sub4, coeff_bi[0][1]));

#if OPFL_DOWNSAMP_QUINCUNX
        temp1 = _mm_mullo_epi32(temp1, mask);
        temp2 = _mm_mullo_epi32(temp2, mask);
#endif
        temp1 = round_power_of_two_signed_epi32(temp1, temp2, v_bias_d, ones,
                                                bicubic_bits);

        const int idx = col * bw + row;
        xx_storeu_128(x_grad + idx, temp1);

        sub1 = _mm_sub_epi32(_mm_unpacklo_epi16(vpred_next1_2, zero),
                             _mm_unpacklo_epi16(vpred_prev1_2, zero));
        sub2 = _mm_sub_epi32(_mm_unpackhi_epi16(vpred_next1_2, zero),
                             _mm_unpackhi_epi16(vpred_prev1_2, zero));
        sub3 = _mm_sub_epi32(_mm_unpacklo_epi16(vpred_next2_2, zero),
                             _mm_unpacklo_epi16(vpred_prev2_2, zero));
        sub4 = _mm_sub_epi32(_mm_unpackhi_epi16(vpred_next2_2, zero),
                             _mm_unpackhi_epi16(vpred_prev2_2, zero));

        temp1 = _mm_add_epi32(_mm_mullo_epi32(sub1, coeff_bi[0][0]),
                              _mm_mullo_epi32(sub3, coeff_bi[0][1]));
        temp2 = _mm_add_epi32(
            _mm_mullo_epi32(sub2, coeff_bi[is_right_boundary][0]),
            _mm_mullo_epi32(sub4, coeff_bi[is_right_boundary][1]));

#if OPFL_DOWNSAMP_QUINCUNX
        temp1 = _mm_mullo_epi32(temp1, mask);
        temp2 = _mm_mullo_epi32(temp2, mask);
#endif
        temp1 = round_power_of_two_signed_epi32(temp1, temp2, v_bias_d, ones,
                                                bicubic_bits);
        xx_storeu_128(x_grad + idx + 8, temp1);

        src = pred_src + row;
        vpred_prev1_1 = xx_loadu_128(src + bw * id_prev);
        vpred_prev2_1 = xx_loadu_128(src + bw * id_prev2);
        vpred_next1_1 = xx_loadu_128(src + id_next * bw);
        vpred_next2_1 = xx_loadu_128(src + id_next2 * bw);

        vpred_prev1_2 = xx_loadu_128(src + bw * id_prev + 8);
        vpred_prev2_2 = xx_loadu_128(src + bw * id_prev2 + 8);
        vpred_next1_2 = xx_loadu_128(src + id_next * bw + 8);
        vpred_next2_2 = xx_loadu_128(src + id_next2 * bw + 8);

        sub1 = _mm_sub_epi32(_mm_unpacklo_epi16(vpred_next1_1, zero),
                             _mm_unpacklo_epi16(vpred_prev1_1, zero));
        sub2 = _mm_sub_epi32(_mm_unpackhi_epi16(vpred_next1_1, zero),
                             _mm_unpackhi_epi16(vpred_prev1_1, zero));
        sub3 = _mm_sub_epi32(_mm_unpacklo_epi16(vpred_next2_1, zero),
                             _mm_unpacklo_epi16(vpred_prev2_1, zero));
        sub4 = _mm_sub_epi32(_mm_unpackhi_epi16(vpred_next2_1, zero),
                             _mm_unpackhi_epi16(vpred_prev2_1, zero));

        temp1 =
            _mm_add_epi32(_mm_mullo_epi32(sub1, coeff_bi[is_y_boundary][0]),
                          _mm_mullo_epi32(sub3, coeff_bi[is_y_boundary][1]));
        temp2 =
            _mm_add_epi32(_mm_mullo_epi32(sub2, coeff_bi[is_y_boundary][0]),
                          _mm_mullo_epi32(sub4, coeff_bi[is_y_boundary][1]));
#if OPFL_DOWNSAMP_QUINCUNX
        temp1 = _mm_mullo_epi32(temp1, mask);
        temp2 = _mm_mullo_epi32(temp2, mask);
#endif

        temp1 = round_power_of_two_signed_epi32(temp1, temp2, v_bias_d, ones,
                                                bicubic_bits);
        xx_storeu_128(y_grad + idx, temp1);

        sub1 = _mm_sub_epi32(_mm_unpacklo_epi16(vpred_next1_2, zero),
                             _mm_unpacklo_epi16(vpred_prev1_2, zero));
        sub2 = _mm_sub_epi32(_mm_unpackhi_epi16(vpred_next1_2, zero),
                             _mm_unpackhi_epi16(vpred_prev1_2, zero));
        sub3 = _mm_sub_epi32(_mm_unpacklo_epi16(vpred_next2_2, zero),
                             _mm_unpacklo_epi16(vpred_prev2_2, zero));
        sub4 = _mm_sub_epi32(_mm_unpackhi_epi16(vpred_next2_2, zero),
                             _mm_unpackhi_epi16(vpred_prev2_2, zero));
        temp1 =
            _mm_add_epi32(_mm_mullo_epi32(sub1, coeff_bi[is_y_boundary][0]),
                          _mm_mullo_epi32(sub3, coeff_bi[is_y_boundary][1]));
        temp2 =
            _mm_add_epi32(_mm_mullo_epi32(sub2, coeff_bi[is_y_boundary][0]),
                          _mm_mullo_epi32(sub4, coeff_bi[is_y_boundary][1]));
#if OPFL_DOWNSAMP_QUINCUNX
        temp1 = _mm_mullo_epi32(temp1, mask);
        temp2 = _mm_mullo_epi32(temp2, mask);
#endif
        temp1 = round_power_of_two_signed_epi32(temp1, temp2, v_bias_d, ones,
                                                bicubic_bits);
        xx_storeu_128(y_grad + idx + 8, temp1);
      }
    }
  }
#else
  (void)pred_src;
  (void)x_grad;
  (void)y_grad;
  (void)bw;
  (void)bh;
#endif  // OPFL_BICUBIC_GRAD
}

inline __m128i LoadLo8(const void *a) {
  return _mm_loadl_epi64((const __m128i *)a);
}

inline __m128i LoadUnaligned16(const void *a) {
  return _mm_loadu_si128((const __m128i *)a);
}

static AOM_FORCE_INLINE void set_distance(__m128i *dist_d0, __m128i *dist_d0d1,
                                          int d0, int d1) {
  __m128i zero = _mm_setzero_si128();
  dist_d0[0] = _mm_set1_epi16(d0);
  dist_d0d1[0] = _mm_set1_epi16(d1);
  dist_d0d1[0] = _mm_sub_epi16(zero, dist_d0d1[0]);
  dist_d0d1[0] = _mm_unpacklo_epi16(_mm_set1_epi16(d0), dist_d0d1[0]);
  dist_d0[0] = _mm_sub_epi16(zero, dist_d0[0]);
  dist_d0[0] = _mm_unpacklo_epi16(_mm_set1_epi16(d0), dist_d0[0]);
}

static AOM_FORCE_INLINE void leastsquare_8x8(__m128i *grad0, __m128i *grad1,
                                             __m128i *grad0_13,
                                             __m128i *grad1_13, __m128i *g2,
                                             const __m128i dist_d0d1) {
  __m128i samplesL, samplesH;

  samplesL = _mm_unpacklo_epi16(grad0[0], grad1[0]);
  samplesH = _mm_unpackhi_epi16(grad0[0], grad1[0]);
  grad0[0] = _mm_madd_epi16(samplesL, dist_d0d1);
  grad1[0] = _mm_madd_epi16(samplesH, dist_d0d1);
  g2[0] = _mm_add_epi64(g2[0], _mm_mul_epi32(grad0[0], grad0[0]));
  g2[0] = _mm_add_epi64(g2[0], _mm_mul_epi32(grad1[0], grad1[0]));
  grad0_13[0] = _mm_srli_si128(grad0[0], 4);
  g2[0] = _mm_add_epi64(g2[0], _mm_mul_epi32(grad0_13[0], grad0_13[0]));
  grad1_13[0] = _mm_srli_si128(grad1[0], 4);
  g2[0] = _mm_add_epi64(g2[0], _mm_mul_epi32(grad1_13[0], grad1_13[0]));
}

static AOM_FORCE_INLINE void leastsquare_8x4(__m128i *grad0, __m128i *grad1,
                                             __m128i *grad0_13,
                                             __m128i *grad1_13, __m128i *g2_0,
                                             __m128i *g2_1,
                                             const __m128i dist_d0d1) {
  __m128i samplesL, samplesH;

  samplesL = _mm_unpacklo_epi16(grad0[0], grad1[0]);
  samplesH = _mm_unpackhi_epi16(grad0[0], grad1[0]);
  grad0[0] = _mm_madd_epi16(samplesL, dist_d0d1);
  g2_0[0] = _mm_add_epi64(g2_0[0], _mm_mul_epi32(grad0[0], grad0[0]));
  grad0_13[0] = _mm_srli_si128(grad0[0], 4);
  g2_0[0] = _mm_add_epi64(g2_0[0], _mm_mul_epi32(grad0_13[0], grad0_13[0]));

  grad1[0] = _mm_madd_epi16(samplesH, dist_d0d1);
  g2_1[0] = _mm_add_epi64(g2_1[0], _mm_mul_epi32(grad1[0], grad1[0]));
  grad1_13[0] = _mm_srli_si128(grad1[0], 4);
  g2_1[0] = _mm_add_epi64(g2_1[0], _mm_mul_epi32(grad1_13[0], grad1_13[0]));
}

static AOM_FORCE_INLINE void accumulate_8x4(
    const __m128i gradX0_02, const __m128i gradY0_02, const __m128i gradX1_02,
    const __m128i gradY1_02, const __m128i gradX0_13, const __m128i gradY0_13,
    const __m128i gradX1_13, const __m128i gradY1_13, __m128i *gg_0,
    __m128i *gg_1) {
  gg_0[0] = _mm_add_epi64(gg_0[0], _mm_mul_epi32(gradX0_02, gradY0_02));
  gg_0[0] = _mm_add_epi64(gg_0[0], _mm_mul_epi32(gradX0_13, gradY0_13));

  gg_1[0] = _mm_add_epi64(gg_1[0], _mm_mul_epi32(gradX1_02, gradY1_02));
  gg_1[0] = _mm_add_epi64(gg_1[0], _mm_mul_epi32(gradX1_13, gradY1_13));
}

static AOM_FORCE_INLINE void accumulate_8x8(
    const __m128i gradX0_02, const __m128i gradY0_02, const __m128i gradX1_02,
    const __m128i gradY1_02, const __m128i gradX0_13, const __m128i gradY0_13,
    const __m128i gradX1_13, const __m128i gradY1_13, __m128i *gg) {
  gg[0] = _mm_add_epi64(gg[0], _mm_mul_epi32(gradX0_02, gradY0_02));
  gg[0] = _mm_add_epi64(gg[0], _mm_mul_epi32(gradX1_02, gradY1_02));
  gg[0] = _mm_add_epi64(gg[0], _mm_mul_epi32(gradX0_13, gradY0_13));
  gg[0] = _mm_add_epi64(gg[0], _mm_mul_epi32(gradX1_13, gradY1_13));
}

static void opfl_mv_refinement_lowbd_8x4_sse4_1(
    const __m128i dist_d0, const __m128i dist_d0d1, const uint8_t *p0,
    int pstride0, const uint8_t *p1, int pstride1, const int16_t *gx0,
    const int16_t *gy0, const int16_t *gx1, const int16_t *gy1, int gstride,
    int d0, int d1, int grad_prec_bits, int mv_prec_bits, int *vx0, int *vy0,
    int *vx1, int *vy1) {
  int bHeight = 4;

  __m128i u2_0 = _mm_setzero_si128();
  __m128i u2_1 = _mm_setzero_si128();
  __m128i v2_0 = _mm_setzero_si128();
  __m128i v2_1 = _mm_setzero_si128();
  __m128i uv_0 = _mm_setzero_si128();
  __m128i uv_1 = _mm_setzero_si128();
  __m128i uw_0 = _mm_setzero_si128();
  __m128i uw_1 = _mm_setzero_si128();
  __m128i vw_0 = _mm_setzero_si128();
  __m128i vw_1 = _mm_setzero_si128();

  do {
    __m128i gradX0, gradX1, gradY0, gradY1, pred0, pred1;
    __m128i gradX0_13, gradX1_13, gradY0_13, gradY1_13;
    __m128i samplesL, samplesH;

    gradX0 = LoadUnaligned16(gx0);
    gradX1 = LoadUnaligned16(gx1);
    leastsquare_8x4(&gradX0, &gradX1, &gradX0_13, &gradX1_13, &u2_0, &u2_1,
                    dist_d0d1);

    gradY0 = LoadUnaligned16(gy0);
    gradY1 = LoadUnaligned16(gy1);
    leastsquare_8x4(&gradY0, &gradY1, &gradY0_13, &gradY1_13, &v2_0, &v2_1,
                    dist_d0d1);

    accumulate_8x4(gradX0, gradY0, gradX1, gradY1, gradX0_13, gradY0_13,
                   gradX1_13, gradY1_13, &uv_0, &uv_1);

    pred0 = _mm_cvtepu8_epi16(LoadLo8(p0));
    pred1 = _mm_cvtepu8_epi16(LoadLo8(p1));
    samplesL = _mm_unpacklo_epi16(pred0, pred1);
    samplesL = _mm_madd_epi16(samplesL, dist_d0);

    samplesH = _mm_unpackhi_epi16(pred0, pred1);
    samplesH = _mm_madd_epi16(samplesH, dist_d0);

    pred0 = _mm_srli_si128(samplesL, 4);
    pred1 = _mm_srli_si128(samplesH, 4);
    accumulate_8x4(gradX0, samplesL, gradX1, samplesH, gradX0_13, pred0,
                   gradX1_13, pred1, &uw_0, &uw_1);
    accumulate_8x4(gradY0, samplesL, gradY1, samplesH, gradY0_13, pred0,
                   gradY1_13, pred1, &vw_0, &vw_1);

    gx0 += gstride;
    gx1 += gstride;
    gy0 += gstride;
    gy1 += gstride;
    p0 += pstride0;
    p1 += pstride1;
    bHeight -= 1;
  } while (bHeight != 0);

  u2_0 = _mm_add_epi64(u2_0, _mm_srli_si128(u2_0, 8));
  u2_1 = _mm_add_epi64(u2_1, _mm_srli_si128(u2_1, 8));

  v2_0 = _mm_add_epi64(v2_0, _mm_srli_si128(v2_0, 8));
  v2_1 = _mm_add_epi64(v2_1, _mm_srli_si128(v2_1, 8));

  uv_0 = _mm_add_epi64(uv_0, _mm_srli_si128(uv_0, 8));
  uv_1 = _mm_add_epi64(uv_1, _mm_srli_si128(uv_1, 8));

  uw_0 = _mm_add_epi64(uw_0, _mm_srli_si128(uw_0, 8));
  uw_1 = _mm_add_epi64(uw_1, _mm_srli_si128(uw_1, 8));

  vw_0 = _mm_add_epi64(vw_0, _mm_srli_si128(vw_0, 8));
  vw_1 = _mm_add_epi64(vw_1, _mm_srli_si128(vw_1, 8));

  int64_t su2, suv, sv2, suw, svw;
  _mm_storel_epi64((__m128i *)&su2, u2_0);
  _mm_storel_epi64((__m128i *)&suv, uv_0);
  _mm_storel_epi64((__m128i *)&sv2, v2_0);
  _mm_storel_epi64((__m128i *)&suw, uw_0);
  _mm_storel_epi64((__m128i *)&svw, vw_0);
#if OPFL_REGULARIZED_LS
  // As processing block size is 4x4, here '(bw * bh >> 4)' can be replaced
  // by 1.
  int rls_alpha = 1 << OPFL_RLS_PARAM_BITS;
  su2 += rls_alpha;
  sv2 += rls_alpha;
#endif
  // Clamp su2, sv2, suv, suw, and svw to avoid overflow in D, Px, and Py
  su2 = clamp(su2, -OPFL_COV_CLAMP_VAL, OPFL_COV_CLAMP_VAL);
  sv2 = clamp(sv2, -OPFL_COV_CLAMP_VAL, OPFL_COV_CLAMP_VAL);
  suv = clamp(suv, -OPFL_COV_CLAMP_VAL, OPFL_COV_CLAMP_VAL);
  suw = clamp(suw, -OPFL_COV_CLAMP_VAL, OPFL_COV_CLAMP_VAL);
  svw = clamp(svw, -OPFL_COV_CLAMP_VAL, OPFL_COV_CLAMP_VAL);

  int64_t su2_1, suv_1, sv2_1, suw_1, svw_1;
  _mm_storel_epi64((__m128i *)&su2_1, u2_1);
  _mm_storel_epi64((__m128i *)&suv_1, uv_1);
  _mm_storel_epi64((__m128i *)&sv2_1, v2_1);
  _mm_storel_epi64((__m128i *)&suw_1, uw_1);
  _mm_storel_epi64((__m128i *)&svw_1, vw_1);
#if OPFL_REGULARIZED_LS
  su2 += rls_alpha;
  sv2 += rls_alpha;
#endif
  // Clamp su2, sv2, suv, suw, and svw to avoid overflow in D, Px, and Py
  su2_1 = clamp(su2_1, -OPFL_COV_CLAMP_VAL, OPFL_COV_CLAMP_VAL);
  sv2_1 = clamp(sv2_1, -OPFL_COV_CLAMP_VAL, OPFL_COV_CLAMP_VAL);
  suv_1 = clamp(suv_1, -OPFL_COV_CLAMP_VAL, OPFL_COV_CLAMP_VAL);
  suw_1 = clamp(suw_1, -OPFL_COV_CLAMP_VAL, OPFL_COV_CLAMP_VAL);
  svw_1 = clamp(svw_1, -OPFL_COV_CLAMP_VAL, OPFL_COV_CLAMP_VAL);

  int bits = mv_prec_bits + grad_prec_bits;
  int64_t D = su2 * sv2 - suv * suv;
  int64_t Px, Py;

  if (D != 0) {
    Px = (suv * svw - sv2 * suw) * (1 << bits);
    Py = (suv * suw - su2 * svw) * (1 << bits);
    *vx0 = (int)divide_and_round_signed(Px, D);
    *vy0 = (int)divide_and_round_signed(Py, D);
    int tx1 = (*vx0) * d1;
    int ty1 = (*vy0) * d1;
    *vx1 = (int)divide_and_round_signed(tx1, d0);
    *vy1 = (int)divide_and_round_signed(ty1, d0);
  }

  D = su2_1 * sv2_1 - suv_1 * suv_1;

  if (D != 0) {
    Px = (suv_1 * svw_1 - sv2_1 * suw_1) * (1 << bits);
    Py = (suv_1 * suw_1 - su2_1 * svw_1) * (1 << bits);
    *(vx0 + 1) = (int)divide_and_round_signed(Px, D);
    *(vy0 + 1) = (int)divide_and_round_signed(Py, D);
    int tx1 = (*(vx0 + 1)) * d1;
    int ty1 = (*(vy0 + 1)) * d1;
    *(vx1 + 1) = (int)divide_and_round_signed(tx1, d0);
    *(vy1 + 1) = (int)divide_and_round_signed(ty1, d0);
  }
}

static void opfl_mv_refinement_lowbd_8x8_sse4_1(
    const __m128i dist_d0, const __m128i dist_d0d1, const uint8_t *p0,
    int pstride0, const uint8_t *p1, int pstride1, const int16_t *gx0,
    const int16_t *gy0, const int16_t *gx1, const int16_t *gy1, int gstride,
    int d0, int d1, int grad_prec_bits, int mv_prec_bits, int *vx0, int *vy0,
    int *vx1, int *vy1) {
  int bHeight = 8;

  __m128i u2 = _mm_setzero_si128();
  __m128i uv = _mm_setzero_si128();
  __m128i v2 = _mm_setzero_si128();
  __m128i uw = _mm_setzero_si128();
  __m128i vw = _mm_setzero_si128();

  do {
    __m128i gradX0, gradX1, gradY0, gradY1, pred0, pred1;
    __m128i gradX0_13, gradX1_13, gradY0_13, gradY1_13;
    __m128i samplesL, samplesH;

    gradX0 = LoadUnaligned16(gx0);
    gradX1 = LoadUnaligned16(gx1);
    leastsquare_8x8(&gradX0, &gradX1, &gradX0_13, &gradX1_13, &u2, dist_d0d1);

    gradY0 = LoadUnaligned16(gy0);
    gradY1 = LoadUnaligned16(gy1);
    leastsquare_8x8(&gradY0, &gradY1, &gradY0_13, &gradY1_13, &v2, dist_d0d1);

    accumulate_8x8(gradX0, gradY0, gradX1, gradY1, gradX0_13, gradY0_13,
                   gradX1_13, gradY1_13, &uv);

    pred0 = _mm_cvtepu8_epi16(LoadLo8(p0));
    pred1 = _mm_cvtepu8_epi16(LoadLo8(p1));
    samplesL = _mm_unpacklo_epi16(pred0, pred1);
    samplesH = _mm_unpackhi_epi16(pred0, pred1);
    samplesL = _mm_madd_epi16(samplesL, dist_d0);
    samplesH = _mm_madd_epi16(samplesH, dist_d0);

    pred0 = _mm_srli_si128(samplesL, 4);
    pred1 = _mm_srli_si128(samplesH, 4);
    accumulate_8x8(gradX0, samplesL, gradX1, samplesH, gradX0_13, pred0,
                   gradX1_13, pred1, &uw);
    accumulate_8x8(gradY0, samplesL, gradY1, samplesH, gradY0_13, pred0,
                   gradY1_13, pred1, &vw);

    gx0 += gstride;
    gx1 += gstride;
    gy0 += gstride;
    gy1 += gstride;
    p0 += pstride0;
    p1 += pstride1;
    bHeight -= 1;
  } while (bHeight != 0);

  u2 = _mm_add_epi64(u2, _mm_srli_si128(u2, 8));
  v2 = _mm_add_epi64(v2, _mm_srli_si128(v2, 8));
  uv = _mm_add_epi64(uv, _mm_srli_si128(uv, 8));
  uw = _mm_add_epi64(uw, _mm_srli_si128(uw, 8));
  vw = _mm_add_epi64(vw, _mm_srli_si128(vw, 8));

  int64_t su2, suv, sv2, suw, svw;
  _mm_storel_epi64((__m128i *)&su2, u2);
  _mm_storel_epi64((__m128i *)&suv, uv);
  _mm_storel_epi64((__m128i *)&sv2, v2);
  _mm_storel_epi64((__m128i *)&suw, uw);
  _mm_storel_epi64((__m128i *)&svw, vw);
#if OPFL_REGULARIZED_LS
  // As processing block size is 8x8, here '(bw * bh >> 4)' can be replaced
  // by 4.
  int rls_alpha = 4 << OPFL_RLS_PARAM_BITS;
  su2 += rls_alpha;
  sv2 += rls_alpha;
#endif
  // Clamp su2, sv2, suv, suw, and svw to avoid overflow in D, Px, and Py
  su2 = clamp(su2, -OPFL_COV_CLAMP_VAL, OPFL_COV_CLAMP_VAL);
  sv2 = clamp(sv2, -OPFL_COV_CLAMP_VAL, OPFL_COV_CLAMP_VAL);
  suv = clamp(suv, -OPFL_COV_CLAMP_VAL, OPFL_COV_CLAMP_VAL);
  suw = clamp(suw, -OPFL_COV_CLAMP_VAL, OPFL_COV_CLAMP_VAL);
  svw = clamp(svw, -OPFL_COV_CLAMP_VAL, OPFL_COV_CLAMP_VAL);

  int bits = mv_prec_bits + grad_prec_bits;
  const int64_t D = su2 * sv2 - suv * suv;
  if (D == 0) return;
  const int64_t Px = (suv * svw - sv2 * suw) * (1 << bits);
  const int64_t Py = (suv * suw - su2 * svw) * (1 << bits);
  *vx0 = (int)divide_and_round_signed(Px, D);
  *vy0 = (int)divide_and_round_signed(Py, D);
  const int tx1 = (*vx0) * d1;
  const int ty1 = (*vy0) * d1;
  *vx1 = (int)divide_and_round_signed(tx1, d0);
  *vy1 = (int)divide_and_round_signed(ty1, d0);
}

static void opfl_mv_refinement_lowbd_sse4_1(
    const __m128i dist_d0, const __m128i dist_d0d1, const uint8_t *p0,
    int pstride0, const uint8_t *p1, int pstride1, const int16_t *gx0,
    const int16_t *gy0, const int16_t *gx1, const int16_t *gy1, int gstride,
    int bw, int bh, int d0, int d1, int grad_prec_bits, int mv_prec_bits,
    int *vx0, int *vy0, int *vx1, int *vy1) {
  (void)bh;
  if (bw == 4)
    opfl_mv_refinement_lowbd_8x4_sse4_1(
        dist_d0, dist_d0d1, p0, pstride0, p1, pstride1, gx0, gy0, gx1, gy1,
        gstride, d0, d1, grad_prec_bits, mv_prec_bits, vx0, vy0, vx1, vy1);
  else
    opfl_mv_refinement_lowbd_8x8_sse4_1(
        dist_d0, dist_d0d1, p0, pstride0, p1, pstride1, gx0, gy0, gx1, gy1,
        gstride, d0, d1, grad_prec_bits, mv_prec_bits, vx0, vy0, vx1, vy1);
}

// Function to compute optical flow offsets in nxn blocks
int av1_opfl_mv_refinement_nxn_lowbd_sse4_1(
    const uint8_t *p0, int pstride0, const uint8_t *p1, int pstride1,
    const int16_t *gx0, const int16_t *gy0, const int16_t *gx1,
    const int16_t *gy1, int gstride, int bw, int bh, int n, int d0, int d1,
    int grad_prec_bits, int mv_prec_bits, int *vx0, int *vy0, int *vx1,
    int *vy1) {
  assert(bw % n == 0 && bh % n == 0);
  int n_blocks = 0;

  __m128i dist_d0, dist_d0d1;
  set_distance(&dist_d0, &dist_d0d1, d0, d1);

  for (int i = 0; i < bh; i += n) {
    for (int j = 0; j < bw; j += 8) {
      opfl_mv_refinement_lowbd_sse4_1(
          dist_d0, dist_d0d1, p0 + (i * pstride0 + j), pstride0,
          p1 + (i * pstride1 + j), pstride1, gx0 + (i * gstride + j),
          gy0 + (i * gstride + j), gx1 + (i * gstride + j),
          gy1 + (i * gstride + j), gstride, n, n, d0, d1, grad_prec_bits,
          mv_prec_bits, vx0 + n_blocks, vy0 + n_blocks, vx1 + n_blocks,
          vy1 + n_blocks);
      n_blocks += (n == 4) ? 2 : 1;
    }
  }
  return n_blocks;
}

static void opfl_mv_refinement_highbd_8x4_sse4_1(
    const __m128i dist_d0, const __m128i dist_d0d1, const uint16_t *p0,
    int pstride0, const uint16_t *p1, int pstride1, const int16_t *gx0,
    const int16_t *gy0, const int16_t *gx1, const int16_t *gy1, int gstride,
    int d0, int d1, int grad_prec_bits, int mv_prec_bits, int *vx0, int *vy0,
    int *vx1, int *vy1) {
  int bHeight = 4;

  __m128i u2_0 = _mm_setzero_si128();
  __m128i u2_1 = _mm_setzero_si128();
  __m128i v2_0 = _mm_setzero_si128();
  __m128i v2_1 = _mm_setzero_si128();
  __m128i uv_0 = _mm_setzero_si128();
  __m128i uv_1 = _mm_setzero_si128();
  __m128i uw_0 = _mm_setzero_si128();
  __m128i uw_1 = _mm_setzero_si128();
  __m128i vw_0 = _mm_setzero_si128();
  __m128i vw_1 = _mm_setzero_si128();

  do {
    __m128i gradX0, gradX1, gradY0, gradY1, pred0, pred1;
    __m128i gradX0_13, gradX1_13, gradY0_13, gradY1_13;
    __m128i samplesL, samplesH;

    gradX0 = LoadUnaligned16(gx0);
    gradX1 = LoadUnaligned16(gx1);
    leastsquare_8x4(&gradX0, &gradX1, &gradX0_13, &gradX1_13, &u2_0, &u2_1,
                    dist_d0d1);

    gradY0 = LoadUnaligned16(gy0);
    gradY1 = LoadUnaligned16(gy1);
    leastsquare_8x4(&gradY0, &gradY1, &gradY0_13, &gradY1_13, &v2_0, &v2_1,
                    dist_d0d1);

    accumulate_8x4(gradX0, gradY0, gradX1, gradY1, gradX0_13, gradY0_13,
                   gradX1_13, gradY1_13, &uv_0, &uv_1);

    pred0 = LoadUnaligned16(p0);
    pred1 = LoadUnaligned16(p1);
    samplesL = _mm_unpacklo_epi16(pred0, pred1);
    samplesL = _mm_madd_epi16(samplesL, dist_d0);

    samplesH = _mm_unpackhi_epi16(pred0, pred1);
    samplesH = _mm_madd_epi16(samplesH, dist_d0);

    pred0 = _mm_srli_si128(samplesL, 4);
    pred1 = _mm_srli_si128(samplesH, 4);
    accumulate_8x4(gradX0, samplesL, gradX1, samplesH, gradX0_13, pred0,
                   gradX1_13, pred1, &uw_0, &uw_1);
    accumulate_8x4(gradY0, samplesL, gradY1, samplesH, gradY0_13, pred0,
                   gradY1_13, pred1, &vw_0, &vw_1);

    gx0 += gstride;
    gx1 += gstride;
    gy0 += gstride;
    gy1 += gstride;
    p0 += pstride0;
    p1 += pstride1;
    bHeight -= 1;
  } while (bHeight != 0);

  u2_0 = _mm_add_epi64(u2_0, _mm_srli_si128(u2_0, 8));
  u2_1 = _mm_add_epi64(u2_1, _mm_srli_si128(u2_1, 8));

  v2_0 = _mm_add_epi64(v2_0, _mm_srli_si128(v2_0, 8));
  v2_1 = _mm_add_epi64(v2_1, _mm_srli_si128(v2_1, 8));

  uv_0 = _mm_add_epi64(uv_0, _mm_srli_si128(uv_0, 8));
  uv_1 = _mm_add_epi64(uv_1, _mm_srli_si128(uv_1, 8));

  uw_0 = _mm_add_epi64(uw_0, _mm_srli_si128(uw_0, 8));
  uw_1 = _mm_add_epi64(uw_1, _mm_srli_si128(uw_1, 8));

  vw_0 = _mm_add_epi64(vw_0, _mm_srli_si128(vw_0, 8));
  vw_1 = _mm_add_epi64(vw_1, _mm_srli_si128(vw_1, 8));

  int64_t su2, suv, sv2, suw, svw;
  _mm_storel_epi64((__m128i *)&su2, u2_0);
  _mm_storel_epi64((__m128i *)&suv, uv_0);
  _mm_storel_epi64((__m128i *)&sv2, v2_0);
  _mm_storel_epi64((__m128i *)&suw, uw_0);
  _mm_storel_epi64((__m128i *)&svw, vw_0);
#if OPFL_REGULARIZED_LS
  // As processing block size is 4x4, here '(bw * bh >> 4)' can be replaced
  // by 1.
  int rls_alpha = 1 << OPFL_RLS_PARAM_BITS;
  su2 += rls_alpha;
  sv2 += rls_alpha;
#endif
  // Clamp su2, sv2, suv, suw, and svw to avoid overflow in D, Px, and Py
  su2 = clamp(su2, -OPFL_COV_CLAMP_VAL, OPFL_COV_CLAMP_VAL);
  sv2 = clamp(sv2, -OPFL_COV_CLAMP_VAL, OPFL_COV_CLAMP_VAL);
  suv = clamp(suv, -OPFL_COV_CLAMP_VAL, OPFL_COV_CLAMP_VAL);
  suw = clamp(suw, -OPFL_COV_CLAMP_VAL, OPFL_COV_CLAMP_VAL);
  svw = clamp(svw, -OPFL_COV_CLAMP_VAL, OPFL_COV_CLAMP_VAL);

  int64_t su2_1, suv_1, sv2_1, suw_1, svw_1;
  _mm_storel_epi64((__m128i *)&su2_1, u2_1);
  _mm_storel_epi64((__m128i *)&suv_1, uv_1);
  _mm_storel_epi64((__m128i *)&sv2_1, v2_1);
  _mm_storel_epi64((__m128i *)&suw_1, uw_1);
  _mm_storel_epi64((__m128i *)&svw_1, vw_1);
#if OPFL_REGULARIZED_LS
  su2 += rls_alpha;
  sv2 += rls_alpha;
#endif
  // Clamp su2, sv2, suv, suw, and svw to avoid overflow in D, Px, and Py
  su2_1 = clamp(su2_1, -OPFL_COV_CLAMP_VAL, OPFL_COV_CLAMP_VAL);
  sv2_1 = clamp(sv2_1, -OPFL_COV_CLAMP_VAL, OPFL_COV_CLAMP_VAL);
  suv_1 = clamp(suv_1, -OPFL_COV_CLAMP_VAL, OPFL_COV_CLAMP_VAL);
  suw_1 = clamp(suw_1, -OPFL_COV_CLAMP_VAL, OPFL_COV_CLAMP_VAL);
  svw_1 = clamp(svw_1, -OPFL_COV_CLAMP_VAL, OPFL_COV_CLAMP_VAL);

  int bits = mv_prec_bits + grad_prec_bits;
  int64_t D = su2 * sv2 - suv * suv;
  int64_t Px, Py;

  if (D != 0) {
    Px = (suv * svw - sv2 * suw) * (1 << bits);
    Py = (suv * suw - su2 * svw) * (1 << bits);
    *vx0 = (int)divide_and_round_signed(Px, D);
    *vy0 = (int)divide_and_round_signed(Py, D);
    int tx1 = (*vx0) * d1;
    int ty1 = (*vy0) * d1;
    *vx1 = (int)divide_and_round_signed(tx1, d0);
    *vy1 = (int)divide_and_round_signed(ty1, d0);
  }

  D = su2_1 * sv2_1 - suv_1 * suv_1;

  if (D != 0) {
    Px = (suv_1 * svw_1 - sv2_1 * suw_1) * (1 << bits);
    Py = (suv_1 * suw_1 - su2_1 * svw_1) * (1 << bits);
    *(vx0 + 1) = (int)divide_and_round_signed(Px, D);
    *(vy0 + 1) = (int)divide_and_round_signed(Py, D);
    int tx1 = (*(vx0 + 1)) * d1;
    int ty1 = (*(vy0 + 1)) * d1;
    *(vx1 + 1) = (int)divide_and_round_signed(tx1, d0);
    *(vy1 + 1) = (int)divide_and_round_signed(ty1, d0);
  }
}

static void opfl_mv_refinement_highbd_8x8_sse4_1(
    const __m128i dist_d0, const __m128i dist_d0d1, const uint16_t *p0,
    int pstride0, const uint16_t *p1, int pstride1, const int16_t *gx0,
    const int16_t *gy0, const int16_t *gx1, const int16_t *gy1, int gstride,
    int d0, int d1, int grad_prec_bits, int mv_prec_bits, int *vx0, int *vy0,
    int *vx1, int *vy1) {
  int bHeight = 8;

  __m128i u2 = _mm_setzero_si128();
  __m128i uv = _mm_setzero_si128();
  __m128i v2 = _mm_setzero_si128();
  __m128i uw = _mm_setzero_si128();
  __m128i vw = _mm_setzero_si128();

  do {
    __m128i gradX0, gradX1, gradY0, gradY1, pred0, pred1;
    __m128i gradX0_13, gradX1_13, gradY0_13, gradY1_13;
    __m128i samplesL, samplesH;

    gradX0 = LoadUnaligned16(gx0);
    gradX1 = LoadUnaligned16(gx1);
    leastsquare_8x8(&gradX0, &gradX1, &gradX0_13, &gradX1_13, &u2, dist_d0d1);

    gradY0 = LoadUnaligned16(gy0);
    gradY1 = LoadUnaligned16(gy1);
    leastsquare_8x8(&gradY0, &gradY1, &gradY0_13, &gradY1_13, &v2, dist_d0d1);

    accumulate_8x8(gradX0, gradY0, gradX1, gradY1, gradX0_13, gradY0_13,
                   gradX1_13, gradY1_13, &uv);

    pred0 = LoadUnaligned16(p0);
    pred1 = LoadUnaligned16(p1);
    samplesL = _mm_unpacklo_epi16(pred0, pred1);
    samplesH = _mm_unpackhi_epi16(pred0, pred1);
    samplesL = _mm_madd_epi16(samplesL, dist_d0);
    samplesH = _mm_madd_epi16(samplesH, dist_d0);

    pred0 = _mm_srli_si128(samplesL, 4);
    pred1 = _mm_srli_si128(samplesH, 4);
    accumulate_8x8(gradX0, samplesL, gradX1, samplesH, gradX0_13, pred0,
                   gradX1_13, pred1, &uw);
    accumulate_8x8(gradY0, samplesL, gradY1, samplesH, gradY0_13, pred0,
                   gradY1_13, pred1, &vw);

    gx0 += gstride;
    gx1 += gstride;
    gy0 += gstride;
    gy1 += gstride;
    p0 += pstride0;
    p1 += pstride1;
    bHeight -= 1;
  } while (bHeight != 0);

  u2 = _mm_add_epi64(u2, _mm_srli_si128(u2, 8));
  v2 = _mm_add_epi64(v2, _mm_srli_si128(v2, 8));
  uv = _mm_add_epi64(uv, _mm_srli_si128(uv, 8));
  uw = _mm_add_epi64(uw, _mm_srli_si128(uw, 8));
  vw = _mm_add_epi64(vw, _mm_srli_si128(vw, 8));

  int64_t su2, suv, sv2, suw, svw;
  _mm_storel_epi64((__m128i *)&su2, u2);
  _mm_storel_epi64((__m128i *)&suv, uv);
  _mm_storel_epi64((__m128i *)&sv2, v2);
  _mm_storel_epi64((__m128i *)&suw, uw);
  _mm_storel_epi64((__m128i *)&svw, vw);
#if OPFL_REGULARIZED_LS
  // As processing block size is 8x8, here '(bw * bh >> 4)' can be replaced
  // by 4.
  int rls_alpha = 4 << OPFL_RLS_PARAM_BITS;
  su2 += rls_alpha;
  sv2 += rls_alpha;
#endif
  // Clamp su2, sv2, suv, suw, and svw to avoid overflow in D, Px, and Py
  su2 = clamp(su2, -OPFL_COV_CLAMP_VAL, OPFL_COV_CLAMP_VAL);
  sv2 = clamp(sv2, -OPFL_COV_CLAMP_VAL, OPFL_COV_CLAMP_VAL);
  suv = clamp(suv, -OPFL_COV_CLAMP_VAL, OPFL_COV_CLAMP_VAL);
  suw = clamp(suw, -OPFL_COV_CLAMP_VAL, OPFL_COV_CLAMP_VAL);
  svw = clamp(svw, -OPFL_COV_CLAMP_VAL, OPFL_COV_CLAMP_VAL);

  int bits = mv_prec_bits + grad_prec_bits;
  const int64_t D = su2 * sv2 - suv * suv;
  if (D == 0) return;

  const int64_t Px = (suv * svw - sv2 * suw) * (1 << bits);
  const int64_t Py = (suv * suw - su2 * svw) * (1 << bits);
  *vx0 = (int)divide_and_round_signed(Px, D);
  *vy0 = (int)divide_and_round_signed(Py, D);
  const int tx1 = (*vx0) * d1;
  const int ty1 = (*vy0) * d1;
  *vx1 = (int)divide_and_round_signed(tx1, d0);
  *vy1 = (int)divide_and_round_signed(ty1, d0);
}

static void opfl_mv_refinement_highbd_sse4_1(
    const __m128i dist_d0, const __m128i dist_d0d1, const uint16_t *p0,
    int pstride0, const uint16_t *p1, int pstride1, const int16_t *gx0,
    const int16_t *gy0, const int16_t *gx1, const int16_t *gy1, int gstride,
    int bw, int bh, int d0, int d1, int grad_prec_bits, int mv_prec_bits,
    int *vx0, int *vy0, int *vx1, int *vy1) {
  (void)bh;
  if (bw == 4)
    opfl_mv_refinement_highbd_8x4_sse4_1(
        dist_d0, dist_d0d1, p0, pstride0, p1, pstride1, gx0, gy0, gx1, gy1,
        gstride, d0, d1, grad_prec_bits, mv_prec_bits, vx0, vy0, vx1, vy1);
  else
    opfl_mv_refinement_highbd_8x8_sse4_1(
        dist_d0, dist_d0d1, p0, pstride0, p1, pstride1, gx0, gy0, gx1, gy1,
        gstride, d0, d1, grad_prec_bits, mv_prec_bits, vx0, vy0, vx1, vy1);
}

// Function to compute optical flow offsets in nxn blocks
int av1_opfl_mv_refinement_nxn_highbd_sse4_1(
    const uint16_t *p0, int pstride0, const uint16_t *p1, int pstride1,
    const int16_t *gx0, const int16_t *gy0, const int16_t *gx1,
    const int16_t *gy1, int gstride, int bw, int bh, int n, int d0, int d1,
    int grad_prec_bits, int mv_prec_bits, int *vx0, int *vy0, int *vx1,
    int *vy1) {
  assert(bw % n == 0 && bh % n == 0);
  int n_blocks = 0;

  __m128i dist_d0, dist_d0d1;
  set_distance(&dist_d0, &dist_d0d1, d0, d1);

  for (int i = 0; i < bh; i += n) {
    for (int j = 0; j < bw; j += 8) {
      opfl_mv_refinement_highbd_sse4_1(
          dist_d0, dist_d0d1, p0 + (i * pstride0 + j), pstride0,
          p1 + (i * pstride1 + j), pstride1, gx0 + (i * gstride + j),
          gy0 + (i * gstride + j), gx1 + (i * gstride + j),
          gy1 + (i * gstride + j), gstride, n, n, d0, d1, grad_prec_bits,
          mv_prec_bits, vx0 + n_blocks, vy0 + n_blocks, vx1 + n_blocks,
          vy1 + n_blocks);
      n_blocks += (n == 4) ? 2 : 1;
    }
  }
  return n_blocks;
}

#endif  // CONFIG_OPTFLOW_REFINEMENT
