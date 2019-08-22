#include <emmintrin.h>
#include <stdbool.h>
#include <assert.h>
#include <smmintrin.h>
#include <strings.h>

#include "config/av1_rtcd.h"

#include "aom_ports/system_state.h"

#include "av1/common/entropymode.h"

#if CONFIG_INTRA_ENTROPY
#define USE_MADD 1
#define HUGE_ARR 0

#if USE_MADD
static const int16_t cos_sin_angle[8][8] = {
  // 45 degrees
  { 45, 45, 45, 45, 45, 45, 45, 45 },
  // 22.5 degrees
  { 59, 24, 59, 24, 59, 24, 59, 24 },
  // 0 degrees
  { 64, 0, 64, 0, 64, 0, 64, 0 },
  // -22.5 degrees,
  { 59, -24, 59, -24, 59, -24, 59, -24 },
  // -45 degrees
  { 45, -45, 45, -45, 45, -45, 45, -45 },
  // -67.5 degrees
  { 24, -59, 24, -59, 24, -59, 24, -59 },
  // -90 degrees
  { 0, 64, 0, 64, 0, 64, 0, 64 },
  // 67.5 degrees
  { 24, 59, 24, 59, 24, 59, 24, 59 },
};
#else
static const int16_t cos_angle[8][8] = {
  // 45 degrees
  { 45, 45, 45, 45, 45, 45, 45, 45 },
  // 22.5 degrees
  { 59, 59, 59, 59, 59, 59, 59, 59 },
  // 0 degrees
  { 64, 64, 64, 64, 64, 64, 64, 64 },
  // -22.5 degrees,
  { 59, 59, 59, 59, 59, 59, 59, 59 },
  // -45 degrees
  { 45, 45, 45, 45, 45, 45, 45, 45 },
  // -67.5 degrees
  { 24, 24, 24, 24, 24, 24, 24, 24 },
  // -90 degrees
  { 0, 0, 0, 0, 0, 0, 0, 0 },
  // 67.5 degrees
  { 24, 24, 24, 24, 24, 24, 24, 24 },
};

static const int16_t sin_angle[8][8] = {
  // 45 degrees
  { 45, 45, 45, 45, 45, 45, 45, 45 },
  // 22.5 degrees
  { 24, 24, 24, 24, 24, 24, 24, 24 },
  // 0 degrees
  { 0, 0, 0, 0, 0, 0, 0, 0 },
  // -22.5 degrees,
  { -24, -24, -24, -24, -24, -24, -24, -24 },
  // -45 degrees
  { -45, -45, -45, -45, -45, -45, -45, -45 },
  // -67.5 degrees
  { -59, -59, -59, -59, -59, -59, -59, -59 },
  // -90 degrees
  { 64, 64, 64, 64, 64, 64, 64, 64 },
  // 67.5 degrees
  { 59, 59, 59, 59, 59, 59, 59, 59 },
};
#endif  // USE_MADD

#if USE_MADD
static INLINE __m128i get_angle_idx_vec(__m128i dxdy_lo, __m128i dxdy_hi) {
  __m128i max_val = _mm_setzero_si128();
  __m128i max_idx = _mm_setzero_si128();
  for (int angle_idx = 0; angle_idx < 8; angle_idx++) {
    __m128i cos_sin =
        _mm_loadu_si128((const __m128i *)cos_sin_angle[angle_idx]);
    __m128i prod_lo = _mm_madd_epi16(dxdy_lo, cos_sin);
    __m128i prod_hi = _mm_madd_epi16(dxdy_hi, cos_sin);
    __m128i prod = _mm_packs_epi32(prod_lo, prod_hi);
    prod = _mm_abs_epi16(prod);

    const __m128i update_mask = _mm_cmpgt_epi16(prod, max_val);
    max_val = _mm_blendv_epi8(max_val, prod, update_mask);
    max_idx = _mm_blendv_epi8(max_idx, _mm_set1_epi16(angle_idx), update_mask);
  }
  return max_idx;
}
#else
static INLINE __m128i get_angle_idx_vec(__m128i dx, __m128i dy) {
  __m128i max_val = _mm_setzero_si128();
  __m128i max_idx = _mm_setzero_si128();
  for (int angle_idx = 0; angle_idx < 8; angle_idx++) {
    const __m128i cos_reg =
        _mm_loadu_si128((const __m128i *)cos_angle[angle_idx]);
    const __m128i sin_reg =
        _mm_loadu_si128((const __m128i *)sin_angle[angle_idx]);
    const __m128i prod_x = _mm_mullo_epi16(dx, cos_reg);
    const __m128i prod_y = _mm_mullo_epi16(dy, sin_reg);
    __m128i prod = _mm_adds_epi16(prod_x, prod_y);
    prod = _mm_abs_epi16(prod);

    const __m128i update_mask = _mm_cmpgt_epi16(prod, max_val);
    max_val = _mm_blendv_epi8(max_val, prod, update_mask);
    max_idx = _mm_blendv_epi8(max_idx, _mm_set1_epi16(angle_idx), update_mask);
  }
  return max_idx;
}
#endif

void av1_get_gradient_hist_lbd_sse4_1(const uint8_t *dst, int stride, int rows,
                                      int cols, uint64_t *hist) {
  const int cols_rem = (cols - 1) % 8;
  const int cols_whole = ((cols - 1) - cols_rem);
  __m128i zero = _mm_setzero_si128();
#if HUGE_ARR
#define ARR_SIZE (128 * 128)
#else
#define ARR_SIZE (8)
#endif

#if USE_MADD
  uint32_t mag_array[ARR_SIZE];
#else
  uint16_t mag_array[ARR_SIZE];
#endif
  uint16_t index_array[ARR_SIZE];

#if USE_MADD
  uint32_t *mag_ptr = mag_array;
#else
  uint16_t *mag_ptr = mag_array;
#endif
  uint16_t *index_ptr = index_array;

  dst += stride;
  for (int r = 1; r < rows; ++r) {
    int c;
    __m128i dst_reg, dst_next_reg, dst_shift_reg;
    for (c = 1; c < cols_whole + 1; c += 8) {
      dst_reg = _mm_loadu_si128((const __m128i *)(dst + c));
      dst_shift_reg = _mm_loadu_si128((const __m128i *)(dst + c - 1));
      dst_next_reg = _mm_loadu_si128((const __m128i *)(dst + c - stride));
      dst_reg = _mm_unpacklo_epi8(dst_reg, zero);
      dst_shift_reg = _mm_unpacklo_epi8(dst_shift_reg, zero);
      dst_next_reg = _mm_unpacklo_epi8(dst_next_reg, zero);

      // 8 of them
      const __m128i dx = _mm_sub_epi16(dst_reg, dst_shift_reg);
      const __m128i dy = _mm_sub_epi16(dst_reg, dst_next_reg);

#if USE_MADD
      // Index with madd
      const __m128i dxdy_lo = _mm_unpacklo_epi16(dx, dy);
      const __m128i dxdy_hi = _mm_unpackhi_epi16(dx, dy);
      const __m128i mag_lo = _mm_madd_epi16(dxdy_lo, dxdy_lo);
      const __m128i mag_hi = _mm_madd_epi16(dxdy_hi, dxdy_hi);
      const __m128i index = get_angle_idx_vec(dxdy_lo, dxdy_hi);
#else
      // Index with mullo
      const __m128i dx_2 = _mm_mullo_epi16(dx, dx);
      const __m128i dy_2 = _mm_mullo_epi16(dy, dy);
      const __m128i mag = _mm_adds_epu16(dx_2, dy_2);
      const __m128i index = get_angle_idx_vec(dx, dy);
#endif

#if USE_MADD
      _mm_storeu_si128((__m128i *)mag_ptr, mag_lo);
      _mm_storeu_si128((__m128i *)(mag_ptr + 4), mag_hi);
#else
      _mm_storeu_si128((__m128i *)mag_ptr, mag);
#endif
      _mm_storeu_si128((__m128i *)index_ptr, index);

      // Compute
#if HUGE_ARR
      mag_ptr += 8;
      index_ptr += 8;
#else
      for (int idx = 0; idx < 8; idx++) {
        const uint8_t index_0 = index_array[idx];
        hist[index_0] += mag_array[idx];
      }
#endif
    }

    if (cols_rem > 0) {
      dst_reg = _mm_loadu_si128((const __m128i *)(dst + c));
      dst_shift_reg = _mm_loadu_si128((const __m128i *)(dst + c - 1));
      dst_next_reg = _mm_loadu_si128((const __m128i *)(dst + c - stride));
      dst_reg = _mm_unpacklo_epi8(dst_reg, zero);
      dst_shift_reg = _mm_unpacklo_epi8(dst_shift_reg, zero);
      dst_next_reg = _mm_unpacklo_epi8(dst_next_reg, zero);

      // 8 of them
      const __m128i dx = _mm_sub_epi16(dst_reg, dst_shift_reg);
      const __m128i dy = _mm_sub_epi16(dst_reg, dst_next_reg);

#if USE_MADD
      // Index with madd
      const __m128i dxdy_lo = _mm_unpacklo_epi16(dx, dy);
      const __m128i dxdy_hi = _mm_unpackhi_epi16(dx, dy);
      const __m128i mag_lo = _mm_madd_epi16(dxdy_lo, dxdy_lo);
      const __m128i mag_hi = _mm_madd_epi16(dxdy_hi, dxdy_hi);
      const __m128i index = get_angle_idx_vec(dxdy_lo, dxdy_hi);
#else
      // Index with mullo
      const __m128i dx_2 = _mm_mullo_epi16(dx, dx);
      const __m128i dy_2 = _mm_mullo_epi16(dy, dy);
      const __m128i mag = _mm_adds_epu16(dx_2, dy_2);
      const __m128i index = get_angle_idx_vec(dx, dy);
#endif

#if USE_MADD
      _mm_storeu_si128((__m128i *)mag_ptr, mag_lo);
      _mm_storeu_si128((__m128i *)(mag_ptr + 4), mag_hi);
#else
      _mm_storeu_si128((__m128i *)mag_ptr, mag);
#endif
      _mm_storeu_si128((__m128i *)index_ptr, index);

      // Compute
#if HUGE_ARR
      mag_ptr += cols_rem;
      index_ptr += cols_rem;
#else
      for (int idx = 0; idx < cols_rem; idx++) {
        const uint8_t index_0 = index_array[idx];
        hist[index_0] += mag_array[idx];
      }
#endif
    }

    dst += stride;
  }

#if HUGE_ARR
  for (int idx = 0; idx < (cols - 1) * (rows - 1); idx++) {
    hist[index_array[idx]] += mag_array[idx];
  }
#endif
}

#endif  // CONFIG_INTRA_ENTROPY
