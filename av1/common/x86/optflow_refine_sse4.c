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

#endif  // CONFIG_OPTFLOW_REFINEMENT
