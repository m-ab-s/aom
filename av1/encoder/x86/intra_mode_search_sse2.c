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

#include <emmintrin.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"

#include "aom_ports/bitops.h"
#include "av1/encoder/intra_mode_search_utils.h"

static inline int get_hist_bin_idx_sse2(int dx, int dy) {
  if (dy == 0) return 16;
  const int32_t ratio = (dy * (1 << HOG_FIX_PREC_BITS)) / dx;
  const int32_t *const thresholds = av1_get_hog_thresholds();

  // Select the lower (0..15) or upper (16..31) half of thresholds, then
  // compare all 16 bins in parallel using SSE2.
  const int lo_idx = (ratio > thresholds[15]) << 4;
  const __m128i ratio_vec = _mm_set1_epi32(ratio);
  const __m128i c0 = _mm_cmpgt_epi32(
      ratio_vec, _mm_load_si128((const __m128i *)&thresholds[lo_idx]));
  const __m128i c1 = _mm_cmpgt_epi32(
      ratio_vec, _mm_load_si128((const __m128i *)&thresholds[lo_idx + 4]));
  const __m128i c2 = _mm_cmpgt_epi32(
      ratio_vec, _mm_load_si128((const __m128i *)&thresholds[lo_idx + 8]));
  const __m128i c3 = _mm_cmpgt_epi32(
      ratio_vec, _mm_load_si128((const __m128i *)&thresholds[lo_idx + 12]));

  // Pack 32-bit comparison masks down to 8-bit lanes and find the first bin
  // where ratio <= thresholds[idx].
  const __m128i p01 = _mm_packs_epi32(c0, c1);
  const __m128i p23 = _mm_packs_epi32(c2, c3);
  const __m128i p0123 = _mm_packs_epi16(p01, p23);
  const uint32_t inv_mask = ~(uint32_t)_mm_movemask_epi8(p0123);
  return lo_idx + get_msb(inv_mask ^ (inv_mask - 1));
}

void av1_lowbd_generate_hog_sse2(const uint8_t *src, int stride, int rows,
                                 int cols, float *hist) {
  lowbd_generate_hog(src, stride, rows, cols, hist, get_hist_bin_idx_sse2);
}

void av1_lowbd_compute_gradient_info_sb_sse2(MACROBLOCK *const x,
                                             BLOCK_SIZE sb_size,
                                             PLANE_TYPE plane) {
  lowbd_compute_gradient_info_sb(x, sb_size, plane, get_hist_bin_idx_sse2);
}

#if CONFIG_AV1_HIGHBITDEPTH
void av1_highbd_generate_hog_sse2(const uint8_t *src8, int stride, int rows,
                                  int cols, float *hist) {
  highbd_generate_hog(src8, stride, rows, cols, hist, get_hist_bin_idx_sse2);
}

void av1_highbd_compute_gradient_info_sb_sse2(MACROBLOCK *const x,
                                              BLOCK_SIZE sb_size,
                                              PLANE_TYPE plane) {
  highbd_compute_gradient_info_sb(x, sb_size, plane, get_hist_bin_idx_sse2);
}
#endif  // CONFIG_AV1_HIGHBITDEPTH
