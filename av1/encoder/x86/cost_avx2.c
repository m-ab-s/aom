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

#include <immintrin.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"

#include "av1/encoder/cost.h"

// Vector form of av1_cost_symbol_clamped for 8 lanes. Bit-exact with the
// scalar path:
//   p15c  = clamp(p15, EC_MIN_PROB, CDF_PROB_TOP - 1)
//   shift = (CDF_PROB_BITS - 1) - floor(log2(p15c))   // via float exponent
//   idx   = min(((p15c << shift) + 64) >> 7, 255) - 128
//   cost  = av1_prob_cost[idx] + shift * 512
// The scalar path can reach idx 128, which duplicates idx 127, so clamping to
// 127 gives the same cost and keeps the 32-bit gather below within the table.
static inline __m256i cost_symbol_x8(__m256i p15) {
  p15 = _mm256_max_epi32(p15, _mm256_set1_epi32(EC_MIN_PROB));
  p15 = _mm256_min_epi32(p15, _mm256_set1_epi32(CDF_PROB_TOP - 1));
  // floor(log2(p15)) = (biased exponent) - 127; exact since p15 < 2^24.
  const __m256i bits = _mm256_castps_si256(_mm256_cvtepi32_ps(p15));
  const __m256i exp =
      _mm256_and_si256(_mm256_srli_epi32(bits, 23), _mm256_set1_epi32(0xFF));
  // shift = (CDF_PROB_BITS - 1) - (exp - 127) = 141 - exp.
  const __m256i shift = _mm256_sub_epi32(_mm256_set1_epi32(141), exp);
  const __m256i norm = _mm256_sllv_epi32(p15, shift);
  __m256i prob =
      _mm256_srli_epi32(_mm256_add_epi32(norm, _mm256_set1_epi32(64)), 7);
  prob = _mm256_min_epi32(prob, _mm256_set1_epi32(255));
  const __m256i idx = _mm256_sub_epi32(prob, _mm256_set1_epi32(128));
  // 16-bit table read via 32-bit gather; mask off the adjacent entry (idx + 1
  // is at most 128, which is inside the 129-entry table).
  __m256i tbl = _mm256_i32gather_epi32((const int *)av1_prob_cost, idx, 2);
  tbl = _mm256_and_si256(tbl, _mm256_set1_epi32(0xFFFF));
  const __m256i lit = _mm256_slli_epi32(shift, AV1_PROB_COST_SHIFT);
  return _mm256_add_epi32(tbl, lit);
}

void av1_cost_tokens_from_cdf_4_batch_avx2(int *costs, int costs_stride,
                                           const uint16_t *cdf, int cdf_stride,
                                           int n) {
  const int s = cdf_stride;
  // Per-lane element offset of the 8 CDFs processed together.
  const __m256i lane =
      _mm256_setr_epi32(0, s, 2 * s, 3 * s, 4 * s, 5 * s, 6 * s, 7 * s);
  const __m256i mask16 = _mm256_set1_epi32(0xFFFF);
  const __m256i top = _mm256_set1_epi32(CDF_PROB_TOP);

  int c = 0;
  for (; c + 8 <= n; c += 8) {
    const __m256i base = _mm256_add_epi32(_mm256_set1_epi32(c * s), lane);
    // Gather {c0,c1} and {c2,c3} pairs (16-bit values packed in 32-bit lanes).
    const __m256i g01 = _mm256_i32gather_epi32((const int *)cdf, base, 2);
    const __m256i g23 = _mm256_i32gather_epi32(
        (const int *)cdf, _mm256_add_epi32(base, _mm256_set1_epi32(2)), 2);
    // ICDF(x) = CDF_PROB_TOP - x; p15[i] = ICDF[i] - ICDF[i-1], ICDF[-1] = 0.
    const __m256i i0 = _mm256_sub_epi32(top, _mm256_and_si256(g01, mask16));
    const __m256i i1 = _mm256_sub_epi32(top, _mm256_srli_epi32(g01, 16));
    const __m256i i2 = _mm256_sub_epi32(top, _mm256_and_si256(g23, mask16));
    const __m256i i3 = _mm256_sub_epi32(top, _mm256_srli_epi32(g23, 16));
    const __m256i r0 = cost_symbol_x8(i0);
    const __m256i r1 = cost_symbol_x8(_mm256_sub_epi32(i1, i0));
    const __m256i r2 = cost_symbol_x8(_mm256_sub_epi32(i2, i1));
    const __m256i r3 = cost_symbol_x8(_mm256_sub_epi32(i3, i2));

    // Transpose the 4 SoA cost vectors (8 ctx each) into 8 AoS rows of 4.
    const __m256i t0 = _mm256_unpacklo_epi32(r0, r1);
    const __m256i t1 = _mm256_unpackhi_epi32(r0, r1);
    const __m256i t2 = _mm256_unpacklo_epi32(r2, r3);
    const __m256i t3 = _mm256_unpackhi_epi32(r2, r3);
    const __m256i q0 = _mm256_unpacklo_epi64(t0, t2);  // ctx 0 | ctx 4
    const __m256i q1 = _mm256_unpackhi_epi64(t0, t2);  // ctx 1 | ctx 5
    const __m256i q2 = _mm256_unpacklo_epi64(t1, t3);  // ctx 2 | ctx 6
    const __m256i q3 = _mm256_unpackhi_epi64(t1, t3);  // ctx 3 | ctx 7

    int *o = costs + c * costs_stride;
    _mm_storeu_si128((__m128i *)(o + 0 * costs_stride),
                     _mm256_castsi256_si128(q0));
    _mm_storeu_si128((__m128i *)(o + 1 * costs_stride),
                     _mm256_castsi256_si128(q1));
    _mm_storeu_si128((__m128i *)(o + 2 * costs_stride),
                     _mm256_castsi256_si128(q2));
    _mm_storeu_si128((__m128i *)(o + 3 * costs_stride),
                     _mm256_castsi256_si128(q3));
    _mm_storeu_si128((__m128i *)(o + 4 * costs_stride),
                     _mm256_extracti128_si256(q0, 1));
    _mm_storeu_si128((__m128i *)(o + 5 * costs_stride),
                     _mm256_extracti128_si256(q1, 1));
    _mm_storeu_si128((__m128i *)(o + 6 * costs_stride),
                     _mm256_extracti128_si256(q2, 1));
    _mm_storeu_si128((__m128i *)(o + 7 * costs_stride),
                     _mm256_extracti128_si256(q3, 1));
  }
  for (; c < n; ++c)
    av1_cost_tokens_from_cdf_4(costs + c * costs_stride, cdf + c * cdf_stride);
}

void av1_cost_tokens_from_cdf_2_batch_avx2(int *costs, int costs_stride,
                                           const uint16_t *cdf, int cdf_stride,
                                           int n) {
  const int s = cdf_stride;
  const __m256i lane =
      _mm256_setr_epi32(0, s, 2 * s, 3 * s, 4 * s, 5 * s, 6 * s, 7 * s);
  const __m256i top = _mm256_set1_epi32(CDF_PROB_TOP);

  int c = 0;
  for (; c + 8 <= n; c += 8) {
    const __m256i base = _mm256_add_epi32(_mm256_set1_epi32(c * s), lane);
    // Only cdf[0] is needed for a 2-symbol CDF; the 32-bit read's high half is
    // the terminator and is masked away.
    const __m256i g = _mm256_i32gather_epi32((const int *)cdf, base, 2);
    const __m256i c0 = _mm256_and_si256(g, _mm256_set1_epi32(0xFFFF));
    // p15[0] = ICDF[0] = CDF_PROB_TOP - cdf[0]; p15[1] = cdf[0].
    const __m256i r0 = cost_symbol_x8(_mm256_sub_epi32(top, c0));
    const __m256i r1 = cost_symbol_x8(c0);

    // Transpose the 2 SoA cost vectors into 8 AoS pairs and store 2 ints/ctx.
    const __m256i t0 = _mm256_unpacklo_epi32(r0, r1);  // ctx 0,1 | ctx 4,5
    const __m256i t1 = _mm256_unpackhi_epi32(r0, r1);  // ctx 2,3 | ctx 6,7
    const __m128i lo0 = _mm256_castsi256_si128(t0);
    const __m128i hi0 = _mm256_extracti128_si256(t0, 1);
    const __m128i lo1 = _mm256_castsi256_si128(t1);
    const __m128i hi1 = _mm256_extracti128_si256(t1, 1);
    int *o = costs + c * costs_stride;
    _mm_storel_epi64((__m128i *)(o + 0 * costs_stride), lo0);
    _mm_storel_epi64((__m128i *)(o + 1 * costs_stride),
                     _mm_unpackhi_epi64(lo0, lo0));
    _mm_storel_epi64((__m128i *)(o + 2 * costs_stride), lo1);
    _mm_storel_epi64((__m128i *)(o + 3 * costs_stride),
                     _mm_unpackhi_epi64(lo1, lo1));
    _mm_storel_epi64((__m128i *)(o + 4 * costs_stride), hi0);
    _mm_storel_epi64((__m128i *)(o + 5 * costs_stride),
                     _mm_unpackhi_epi64(hi0, hi0));
    _mm_storel_epi64((__m128i *)(o + 6 * costs_stride), hi1);
    _mm_storel_epi64((__m128i *)(o + 7 * costs_stride),
                     _mm_unpackhi_epi64(hi1, hi1));
  }
  for (; c < n; ++c)
    av1_cost_tokens_from_cdf_2(costs + c * costs_stride, cdf + c * cdf_stride);
}
