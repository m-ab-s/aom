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

#include <immintrin.h>
#include <stdint.h>

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"
#include "aom_dsp/x86/synonyms.h"
#include "aom_dsp/x86/synonyms_avx2.h"
#include "aom_ports/mem.h"

static AOM_FORCE_INLINE unsigned int hsum_sad_zmm(__m512i sum_sad) {
  const __m256i sum256 = _mm256_add_epi32(
      _mm512_castsi512_si256(sum_sad), _mm512_extracti64x4_epi64(sum_sad, 1));
  const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum256),
                                       _mm256_extracti128_si256(sum256, 1));
  const __m128i sum64 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
  return (unsigned int)_mm_cvtsi128_si32(sum64);
}

static AOM_FORCE_INLINE void aggregate_and_store_sum_zmm(uint32_t *res,
                                                         __m512i sum_ref0,
                                                         __m512i sum_ref1,
                                                         __m512i sum_ref2,
                                                         __m512i sum_ref3) {
  __m256i s0 = _mm256_add_epi32(_mm512_castsi512_si256(sum_ref0),
                                _mm512_extracti64x4_epi64(sum_ref0, 1));
  __m256i s1 = _mm256_add_epi32(_mm512_castsi512_si256(sum_ref1),
                                _mm512_extracti64x4_epi64(sum_ref1, 1));
  __m256i s2 = _mm256_add_epi32(_mm512_castsi512_si256(sum_ref2),
                                _mm512_extracti64x4_epi64(sum_ref2, 1));
  __m256i s3 = _mm256_add_epi32(_mm512_castsi512_si256(sum_ref3),
                                _mm512_extracti64x4_epi64(sum_ref3, 1));

  const __m256i sum_even = _mm256_unpacklo_epi32(s0, s1);
  const __m256i sum_odd = _mm256_unpackhi_epi32(s0, s1);
  const __m256i sum_01 = _mm256_add_epi32(sum_even, sum_odd);

  const __m256i sum_even23 = _mm256_unpacklo_epi32(s2, s3);
  const __m256i sum_odd23 = _mm256_unpackhi_epi32(s2, s3);
  const __m256i sum_23 = _mm256_add_epi32(sum_even23, sum_odd23);

  const __m256i sum = _mm256_unpacklo_epi64(sum_01, sum_23);
  const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum),
                                       _mm256_extracti128_si256(sum, 1));
  _mm_storeu_si128((__m128i *)res, sum128);
}

static AOM_FORCE_INLINE void aggregate_and_store_sum_3d_zmm(uint32_t *res,
                                                            __m512i sum_ref0,
                                                            __m512i sum_ref1,
                                                            __m512i sum_ref2) {
  __m256i s0 = _mm256_add_epi32(_mm512_castsi512_si256(sum_ref0),
                                _mm512_extracti64x4_epi64(sum_ref0, 1));
  __m256i s1 = _mm256_add_epi32(_mm512_castsi512_si256(sum_ref1),
                                _mm512_extracti64x4_epi64(sum_ref1, 1));
  __m256i s2 = _mm256_add_epi32(_mm512_castsi512_si256(sum_ref2),
                                _mm512_extracti64x4_epi64(sum_ref2, 1));

  const __m256i sum_even = _mm256_unpacklo_epi32(s0, s1);
  const __m256i sum_odd = _mm256_unpackhi_epi32(s0, s1);
  const __m256i sum_01 = _mm256_add_epi32(sum_even, sum_odd);

  const __m256i sum_even2 = _mm256_unpacklo_epi32(s2, _mm256_setzero_si256());
  const __m256i sum_odd2 = _mm256_unpackhi_epi32(s2, _mm256_setzero_si256());
  const __m256i sum_23 = _mm256_add_epi32(sum_even2, sum_odd2);

  const __m256i sum = _mm256_unpacklo_epi64(sum_01, sum_23);
  const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum),
                                       _mm256_extracti128_si256(sum, 1));
  _mm_storeu_si128((__m128i *)res, sum128);
}

#if !CONFIG_HIGHWAY || !CONFIG_REALTIME_ONLY
static AOM_FORCE_INLINE unsigned int sad64xh_avx512(const uint8_t *src_ptr,
                                                    int src_stride,
                                                    const uint8_t *ref_ptr,
                                                    int ref_stride, int h) {
  __m512i sum_sad = _mm512_setzero_si512();
  for (int i = 0; i < h; i += 2) {
    const __m512i s0 = _mm512_loadu_si512((const __m512i *)src_ptr);
    const __m512i r0 = _mm512_loadu_si512((const __m512i *)ref_ptr);
    const __m512i s1 =
        _mm512_loadu_si512((const __m512i *)(src_ptr + src_stride));
    const __m512i r1 =
        _mm512_loadu_si512((const __m512i *)(ref_ptr + ref_stride));
    sum_sad = _mm512_add_epi32(sum_sad, _mm512_sad_epu8(r0, s0));
    sum_sad = _mm512_add_epi32(sum_sad, _mm512_sad_epu8(r1, s1));
    src_ptr += src_stride * 2;
    ref_ptr += ref_stride * 2;
  }
  return hsum_sad_zmm(sum_sad);
}

static AOM_FORCE_INLINE unsigned int sad64xh_avg_avx512(
    const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,
    int ref_stride, int h, const uint8_t *second_pred) {
  __m512i sum_sad = _mm512_setzero_si512();
  for (int i = 0; i < h; i += 2) {
    const __m512i r0 =
        _mm512_avg_epu8(_mm512_loadu_si512((const __m512i *)ref_ptr),
                        _mm512_loadu_si512((const __m512i *)second_pred));
    const __m512i r1 = _mm512_avg_epu8(
        _mm512_loadu_si512((const __m512i *)(ref_ptr + ref_stride)),
        _mm512_loadu_si512((const __m512i *)(second_pred + 64)));
    sum_sad = _mm512_add_epi32(
        sum_sad,
        _mm512_sad_epu8(r0, _mm512_loadu_si512((const __m512i *)src_ptr)));
    sum_sad = _mm512_add_epi32(
        sum_sad,
        _mm512_sad_epu8(
            r1, _mm512_loadu_si512((const __m512i *)(src_ptr + src_stride))));
    src_ptr += src_stride * 2;
    ref_ptr += ref_stride * 2;
    second_pred += 128;
  }
  return hsum_sad_zmm(sum_sad);
}

static AOM_FORCE_INLINE void sad64xhx4d_avx512(int h, const uint8_t *src,
                                               int src_stride,
                                               const uint8_t *const ref[4],
                                               int ref_stride, uint32_t *res) {
  const uint8_t *ref0 = ref[0];
  const uint8_t *ref1 = ref[1];
  const uint8_t *ref2 = ref[2];
  const uint8_t *ref3 = ref[3];
  __m512i sum_ref0 = _mm512_setzero_si512();
  __m512i sum_ref1 = _mm512_setzero_si512();
  __m512i sum_ref2 = _mm512_setzero_si512();
  __m512i sum_ref3 = _mm512_setzero_si512();

  for (int i = 0; i < h; i++) {
    const __m512i s = _mm512_loadu_si512((const __m512i *)src);
    sum_ref0 = _mm512_add_epi32(
        sum_ref0,
        _mm512_sad_epu8(_mm512_loadu_si512((const __m512i *)ref0), s));
    sum_ref1 = _mm512_add_epi32(
        sum_ref1,
        _mm512_sad_epu8(_mm512_loadu_si512((const __m512i *)ref1), s));
    sum_ref2 = _mm512_add_epi32(
        sum_ref2,
        _mm512_sad_epu8(_mm512_loadu_si512((const __m512i *)ref2), s));
    sum_ref3 = _mm512_add_epi32(
        sum_ref3,
        _mm512_sad_epu8(_mm512_loadu_si512((const __m512i *)ref3), s));
    src += src_stride;
    ref0 += ref_stride;
    ref1 += ref_stride;
    ref2 += ref_stride;
    ref3 += ref_stride;
  }
  aggregate_and_store_sum_zmm(res, sum_ref0, sum_ref1, sum_ref2, sum_ref3);
}

static AOM_FORCE_INLINE void sad64xhx3d_avx512(int h, const uint8_t *src,
                                               int src_stride,
                                               const uint8_t *const ref[4],
                                               int ref_stride, uint32_t *res) {
  const uint8_t *ref0 = ref[0];
  const uint8_t *ref1 = ref[1];
  const uint8_t *ref2 = ref[2];
  __m512i sum_ref0 = _mm512_setzero_si512();
  __m512i sum_ref1 = _mm512_setzero_si512();
  __m512i sum_ref2 = _mm512_setzero_si512();

  for (int i = 0; i < h; i++) {
    const __m512i s = _mm512_loadu_si512((const __m512i *)src);
    sum_ref0 = _mm512_add_epi32(
        sum_ref0,
        _mm512_sad_epu8(_mm512_loadu_si512((const __m512i *)ref0), s));
    sum_ref1 = _mm512_add_epi32(
        sum_ref1,
        _mm512_sad_epu8(_mm512_loadu_si512((const __m512i *)ref1), s));
    sum_ref2 = _mm512_add_epi32(
        sum_ref2,
        _mm512_sad_epu8(_mm512_loadu_si512((const __m512i *)ref2), s));
    src += src_stride;
    ref0 += ref_stride;
    ref1 += ref_stride;
    ref2 += ref_stride;
  }
  aggregate_and_store_sum_3d_zmm(res, sum_ref0, sum_ref1, sum_ref2);
}
#endif  // !CONFIG_HIGHWAY || !CONFIG_REALTIME_ONLY

#if !CONFIG_HIGHWAY
static AOM_FORCE_INLINE unsigned int sad128xh_avx512(const uint8_t *src_ptr,
                                                     int src_stride,
                                                     const uint8_t *ref_ptr,
                                                     int ref_stride, int h) {
  __m512i sum_sad0 = _mm512_setzero_si512();
  __m512i sum_sad1 = _mm512_setzero_si512();
  for (int i = 0; i < h; i++) {
    const __m512i s0 = _mm512_loadu_si512((const __m512i *)src_ptr);
    const __m512i r0 = _mm512_loadu_si512((const __m512i *)ref_ptr);
    const __m512i s1 = _mm512_loadu_si512((const __m512i *)(src_ptr + 64));
    const __m512i r1 = _mm512_loadu_si512((const __m512i *)(ref_ptr + 64));
    sum_sad0 = _mm512_add_epi32(sum_sad0, _mm512_sad_epu8(r0, s0));
    sum_sad1 = _mm512_add_epi32(sum_sad1, _mm512_sad_epu8(r1, s1));
    src_ptr += src_stride;
    ref_ptr += ref_stride;
  }
  return hsum_sad_zmm(_mm512_add_epi32(sum_sad0, sum_sad1));
}

static AOM_FORCE_INLINE unsigned int sad128xh_avg_avx512(
    const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,
    int ref_stride, int h, const uint8_t *second_pred) {
  __m512i sum_sad0 = _mm512_setzero_si512();
  __m512i sum_sad1 = _mm512_setzero_si512();
  for (int i = 0; i < h; i++) {
    const __m512i r0 =
        _mm512_avg_epu8(_mm512_loadu_si512((const __m512i *)ref_ptr),
                        _mm512_loadu_si512((const __m512i *)second_pred));
    const __m512i r1 = _mm512_avg_epu8(
        _mm512_loadu_si512((const __m512i *)(ref_ptr + 64)),
        _mm512_loadu_si512((const __m512i *)(second_pred + 64)));
    sum_sad0 = _mm512_add_epi32(
        sum_sad0,
        _mm512_sad_epu8(r0, _mm512_loadu_si512((const __m512i *)src_ptr)));
    sum_sad1 = _mm512_add_epi32(
        sum_sad1, _mm512_sad_epu8(
                      r1, _mm512_loadu_si512((const __m512i *)(src_ptr + 64))));
    src_ptr += src_stride;
    ref_ptr += ref_stride;
    second_pred += 128;
  }
  return hsum_sad_zmm(_mm512_add_epi32(sum_sad0, sum_sad1));
}

static AOM_FORCE_INLINE void sad128xhx4d_avx512(int h, const uint8_t *src,
                                                int src_stride,
                                                const uint8_t *const ref[4],
                                                int ref_stride, uint32_t *res) {
  const uint8_t *ref0 = ref[0];
  const uint8_t *ref1 = ref[1];
  const uint8_t *ref2 = ref[2];
  const uint8_t *ref3 = ref[3];
  __m512i sum_ref0 = _mm512_setzero_si512();
  __m512i sum_ref1 = _mm512_setzero_si512();
  __m512i sum_ref2 = _mm512_setzero_si512();
  __m512i sum_ref3 = _mm512_setzero_si512();

  for (int i = 0; i < h; i++) {
    const __m512i s0 = _mm512_loadu_si512((const __m512i *)src);
    const __m512i s1 = _mm512_loadu_si512((const __m512i *)(src + 64));
    sum_ref0 = _mm512_add_epi32(
        sum_ref0,
        _mm512_add_epi32(
            _mm512_sad_epu8(_mm512_loadu_si512((const __m512i *)ref0), s0),
            _mm512_sad_epu8(_mm512_loadu_si512((const __m512i *)(ref0 + 64)),
                            s1)));
    sum_ref1 = _mm512_add_epi32(
        sum_ref1,
        _mm512_add_epi32(
            _mm512_sad_epu8(_mm512_loadu_si512((const __m512i *)ref1), s0),
            _mm512_sad_epu8(_mm512_loadu_si512((const __m512i *)(ref1 + 64)),
                            s1)));
    sum_ref2 = _mm512_add_epi32(
        sum_ref2,
        _mm512_add_epi32(
            _mm512_sad_epu8(_mm512_loadu_si512((const __m512i *)ref2), s0),
            _mm512_sad_epu8(_mm512_loadu_si512((const __m512i *)(ref2 + 64)),
                            s1)));
    sum_ref3 = _mm512_add_epi32(
        sum_ref3,
        _mm512_add_epi32(
            _mm512_sad_epu8(_mm512_loadu_si512((const __m512i *)ref3), s0),
            _mm512_sad_epu8(_mm512_loadu_si512((const __m512i *)(ref3 + 64)),
                            s1)));
    src += src_stride;
    ref0 += ref_stride;
    ref1 += ref_stride;
    ref2 += ref_stride;
    ref3 += ref_stride;
  }
  aggregate_and_store_sum_zmm(res, sum_ref0, sum_ref1, sum_ref2, sum_ref3);
}

static AOM_FORCE_INLINE void sad128xhx3d_avx512(int h, const uint8_t *src,
                                                int src_stride,
                                                const uint8_t *const ref[4],
                                                int ref_stride, uint32_t *res) {
  const uint8_t *ref0 = ref[0];
  const uint8_t *ref1 = ref[1];
  const uint8_t *ref2 = ref[2];
  __m512i sum_ref0 = _mm512_setzero_si512();
  __m512i sum_ref1 = _mm512_setzero_si512();
  __m512i sum_ref2 = _mm512_setzero_si512();

  for (int i = 0; i < h; i++) {
    const __m512i s0 = _mm512_loadu_si512((const __m512i *)src);
    const __m512i s1 = _mm512_loadu_si512((const __m512i *)(src + 64));
    sum_ref0 = _mm512_add_epi32(
        sum_ref0,
        _mm512_add_epi32(
            _mm512_sad_epu8(_mm512_loadu_si512((const __m512i *)ref0), s0),
            _mm512_sad_epu8(_mm512_loadu_si512((const __m512i *)(ref0 + 64)),
                            s1)));
    sum_ref1 = _mm512_add_epi32(
        sum_ref1,
        _mm512_add_epi32(
            _mm512_sad_epu8(_mm512_loadu_si512((const __m512i *)ref1), s0),
            _mm512_sad_epu8(_mm512_loadu_si512((const __m512i *)(ref1 + 64)),
                            s1)));
    sum_ref2 = _mm512_add_epi32(
        sum_ref2,
        _mm512_add_epi32(
            _mm512_sad_epu8(_mm512_loadu_si512((const __m512i *)ref2), s0),
            _mm512_sad_epu8(_mm512_loadu_si512((const __m512i *)(ref2 + 64)),
                            s1)));
    src += src_stride;
    ref0 += ref_stride;
    ref1 += ref_stride;
    ref2 += ref_stride;
  }
  aggregate_and_store_sum_3d_zmm(res, sum_ref0, sum_ref1, sum_ref2);
}

#define SAD64_AVX512(h)                                                        \
  unsigned int aom_sad64x##h##_avx512(const uint8_t *src_ptr, int src_stride,  \
                                      const uint8_t *ref_ptr,                  \
                                      int ref_stride) {                        \
    return sad64xh_avx512(src_ptr, src_stride, ref_ptr, ref_stride, h);        \
  }                                                                            \
  unsigned int aom_sad_skip_64x##h##_avx512(                                   \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,          \
      int ref_stride) {                                                        \
    return 2 * sad64xh_avx512(src_ptr, src_stride * 2, ref_ptr,                \
                              ref_stride * 2, (h) / 2);                        \
  }                                                                            \
  unsigned int aom_sad64x##h##_avg_avx512(                                     \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,          \
      int ref_stride, const uint8_t *second_pred) {                            \
    return sad64xh_avg_avx512(src_ptr, src_stride, ref_ptr, ref_stride, h,     \
                              second_pred);                                    \
  }                                                                            \
  void aom_sad64x##h##x4d_avx512(const uint8_t *src_ptr, int src_stride,       \
                                 const uint8_t *const ref_ptr[4],              \
                                 int ref_stride, uint32_t sad_array[4]) {      \
    sad64xhx4d_avx512(h, src_ptr, src_stride, ref_ptr, ref_stride, sad_array); \
  }                                                                            \
  void aom_sad_skip_64x##h##x4d_avx512(                                        \
      const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_ptr[4], \
      int ref_stride, uint32_t sad_array[4]) {                                 \
    sad64xhx4d_avx512((h) >> 1, src_ptr, src_stride << 1, ref_ptr,             \
                      ref_stride << 1, sad_array);                             \
    sad_array[0] <<= 1;                                                        \
    sad_array[1] <<= 1;                                                        \
    sad_array[2] <<= 1;                                                        \
    sad_array[3] <<= 1;                                                        \
  }                                                                            \
  void aom_sad64x##h##x3d_avx512(const uint8_t *src_ptr, int src_stride,       \
                                 const uint8_t *const ref_ptr[4],              \
                                 int ref_stride, uint32_t sad_array[4]) {      \
    sad64xhx3d_avx512(h, src_ptr, src_stride, ref_ptr, ref_stride, sad_array); \
  }

#define SAD128_AVX512(h)                                                       \
  unsigned int aom_sad128x##h##_avx512(const uint8_t *src_ptr, int src_stride, \
                                       const uint8_t *ref_ptr,                 \
                                       int ref_stride) {                       \
    return sad128xh_avx512(src_ptr, src_stride, ref_ptr, ref_stride, h);       \
  }                                                                            \
  unsigned int aom_sad_skip_128x##h##_avx512(                                  \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,          \
      int ref_stride) {                                                        \
    return 2 * sad128xh_avx512(src_ptr, src_stride * 2, ref_ptr,               \
                               ref_stride * 2, (h) / 2);                       \
  }                                                                            \
  unsigned int aom_sad128x##h##_avg_avx512(                                    \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,          \
      int ref_stride, const uint8_t *second_pred) {                            \
    return sad128xh_avg_avx512(src_ptr, src_stride, ref_ptr, ref_stride, h,    \
                               second_pred);                                   \
  }                                                                            \
  void aom_sad128x##h##x4d_avx512(const uint8_t *src_ptr, int src_stride,      \
                                  const uint8_t *const ref_ptr[4],             \
                                  int ref_stride, uint32_t sad_array[4]) {     \
    sad128xhx4d_avx512(h, src_ptr, src_stride, ref_ptr, ref_stride,            \
                       sad_array);                                             \
  }                                                                            \
  void aom_sad_skip_128x##h##x4d_avx512(                                       \
      const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_ptr[4], \
      int ref_stride, uint32_t sad_array[4]) {                                 \
    sad128xhx4d_avx512((h) >> 1, src_ptr, src_stride << 1, ref_ptr,            \
                       ref_stride << 1, sad_array);                            \
    sad_array[0] <<= 1;                                                        \
    sad_array[1] <<= 1;                                                        \
    sad_array[2] <<= 1;                                                        \
    sad_array[3] <<= 1;                                                        \
  }                                                                            \
  void aom_sad128x##h##x3d_avx512(const uint8_t *src_ptr, int src_stride,      \
                                  const uint8_t *const ref_ptr[4],             \
                                  int ref_stride, uint32_t sad_array[4]) {     \
    sad128xhx3d_avx512(h, src_ptr, src_stride, ref_ptr, ref_stride,            \
                       sad_array);                                             \
  }

SAD128_AVX512(128)
SAD128_AVX512(64)
SAD64_AVX512(128)
SAD64_AVX512(64)
SAD64_AVX512(32)
#endif  // !CONFIG_HIGHWAY

#if !CONFIG_REALTIME_ONLY
// 64x16 is not provided by Highway, so always compile it under AVX-512.
unsigned int aom_sad64x16_avx512(const uint8_t *src_ptr, int src_stride,
                                 const uint8_t *ref_ptr, int ref_stride) {
  return sad64xh_avx512(src_ptr, src_stride, ref_ptr, ref_stride, 16);
}

unsigned int aom_sad_skip_64x16_avx512(const uint8_t *src_ptr, int src_stride,
                                       const uint8_t *ref_ptr, int ref_stride) {
  return 2 *
         sad64xh_avx512(src_ptr, src_stride * 2, ref_ptr, ref_stride * 2, 8);
}

unsigned int aom_sad64x16_avg_avx512(const uint8_t *src_ptr, int src_stride,
                                     const uint8_t *ref_ptr, int ref_stride,
                                     const uint8_t *second_pred) {
  return sad64xh_avg_avx512(src_ptr, src_stride, ref_ptr, ref_stride, 16,
                            second_pred);
}

void aom_sad64x16x4d_avx512(const uint8_t *src_ptr, int src_stride,
                            const uint8_t *const ref_ptr[4], int ref_stride,
                            uint32_t sad_array[4]) {
  sad64xhx4d_avx512(16, src_ptr, src_stride, ref_ptr, ref_stride, sad_array);
}

void aom_sad_skip_64x16x4d_avx512(const uint8_t *src_ptr, int src_stride,
                                  const uint8_t *const ref_ptr[4],
                                  int ref_stride, uint32_t sad_array[4]) {
  sad64xhx4d_avx512(8, src_ptr, src_stride << 1, ref_ptr, ref_stride << 1,
                    sad_array);
  sad_array[0] <<= 1;
  sad_array[1] <<= 1;
  sad_array[2] <<= 1;
  sad_array[3] <<= 1;
}

void aom_sad64x16x3d_avx512(const uint8_t *src_ptr, int src_stride,
                            const uint8_t *const ref_ptr[4], int ref_stride,
                            uint32_t sad_array[4]) {
  sad64xhx3d_avx512(16, src_ptr, src_stride, ref_ptr, ref_stride, sad_array);
}
#endif  // !CONFIG_REALTIME_ONLY

// 32xh blocks: pack 2 rows (64 bytes) per 512-bit ZMM register.
static AOM_FORCE_INLINE __m512i loadu2_256_zmm(const uint8_t *hi,
                                               const uint8_t *lo) {
  return _mm512_inserti64x4(
      _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)lo)),
      _mm256_loadu_si256((const __m256i *)hi), 1);
}

static AOM_FORCE_INLINE unsigned int sad32xh_avx512(const uint8_t *src_ptr,
                                                    int src_stride,
                                                    const uint8_t *ref_ptr,
                                                    int ref_stride, int h) {
  __m512i sum_sad = _mm512_setzero_si512();
  for (int i = 0; i < h; i += 2) {
    const __m512i s = loadu2_256_zmm(src_ptr + src_stride, src_ptr);
    const __m512i r = loadu2_256_zmm(ref_ptr + ref_stride, ref_ptr);
    sum_sad = _mm512_add_epi32(sum_sad, _mm512_sad_epu8(r, s));
    src_ptr += src_stride * 2;
    ref_ptr += ref_stride * 2;
  }
  return hsum_sad_zmm(sum_sad);
}

static AOM_FORCE_INLINE unsigned int sad32xh_avg_avx512(
    const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,
    int ref_stride, int h, const uint8_t *second_pred) {
  __m512i sum_sad = _mm512_setzero_si512();
  for (int i = 0; i < h; i += 2) {
    const __m512i s = loadu2_256_zmm(src_ptr + src_stride, src_ptr);
    const __m512i r =
        _mm512_avg_epu8(loadu2_256_zmm(ref_ptr + ref_stride, ref_ptr),
                        _mm512_loadu_si512((const __m512i *)second_pred));
    sum_sad = _mm512_add_epi32(sum_sad, _mm512_sad_epu8(r, s));
    src_ptr += src_stride * 2;
    ref_ptr += ref_stride * 2;
    second_pred += 64;
  }
  return hsum_sad_zmm(sum_sad);
}

static AOM_FORCE_INLINE void sad32xhx4d_avx512(int h, const uint8_t *src,
                                               int src_stride,
                                               const uint8_t *const ref[4],
                                               int ref_stride, uint32_t *res) {
  const uint8_t *ref0 = ref[0];
  const uint8_t *ref1 = ref[1];
  const uint8_t *ref2 = ref[2];
  const uint8_t *ref3 = ref[3];
  __m512i sum_ref0 = _mm512_setzero_si512();
  __m512i sum_ref1 = _mm512_setzero_si512();
  __m512i sum_ref2 = _mm512_setzero_si512();
  __m512i sum_ref3 = _mm512_setzero_si512();

  for (int i = 0; i < h; i += 2) {
    const __m512i s = loadu2_256_zmm(src + src_stride, src);
    const __m512i r0 = loadu2_256_zmm(ref0 + ref_stride, ref0);
    const __m512i r1 = loadu2_256_zmm(ref1 + ref_stride, ref1);
    const __m512i r2 = loadu2_256_zmm(ref2 + ref_stride, ref2);
    const __m512i r3 = loadu2_256_zmm(ref3 + ref_stride, ref3);
    sum_ref0 = _mm512_add_epi32(sum_ref0, _mm512_sad_epu8(r0, s));
    sum_ref1 = _mm512_add_epi32(sum_ref1, _mm512_sad_epu8(r1, s));
    sum_ref2 = _mm512_add_epi32(sum_ref2, _mm512_sad_epu8(r2, s));
    sum_ref3 = _mm512_add_epi32(sum_ref3, _mm512_sad_epu8(r3, s));
    src += src_stride * 2;
    ref0 += ref_stride * 2;
    ref1 += ref_stride * 2;
    ref2 += ref_stride * 2;
    ref3 += ref_stride * 2;
  }
  aggregate_and_store_sum_zmm(res, sum_ref0, sum_ref1, sum_ref2, sum_ref3);
}

static AOM_FORCE_INLINE void sad32xhx3d_avx512(int h, const uint8_t *src,
                                               int src_stride,
                                               const uint8_t *const ref[4],
                                               int ref_stride, uint32_t *res) {
  const uint8_t *ref0 = ref[0];
  const uint8_t *ref1 = ref[1];
  const uint8_t *ref2 = ref[2];
  __m512i sum_ref0 = _mm512_setzero_si512();
  __m512i sum_ref1 = _mm512_setzero_si512();
  __m512i sum_ref2 = _mm512_setzero_si512();

  for (int i = 0; i < h; i += 2) {
    const __m512i s = loadu2_256_zmm(src + src_stride, src);
    const __m512i r0 = loadu2_256_zmm(ref0 + ref_stride, ref0);
    const __m512i r1 = loadu2_256_zmm(ref1 + ref_stride, ref1);
    const __m512i r2 = loadu2_256_zmm(ref2 + ref_stride, ref2);
    sum_ref0 = _mm512_add_epi32(sum_ref0, _mm512_sad_epu8(r0, s));
    sum_ref1 = _mm512_add_epi32(sum_ref1, _mm512_sad_epu8(r1, s));
    sum_ref2 = _mm512_add_epi32(sum_ref2, _mm512_sad_epu8(r2, s));
    src += src_stride * 2;
    ref0 += ref_stride * 2;
    ref1 += ref_stride * 2;
    ref2 += ref_stride * 2;
  }
  aggregate_and_store_sum_3d_zmm(res, sum_ref0, sum_ref1, sum_ref2);
}

#define SAD32_AVX512(h)                                                        \
  unsigned int aom_sad32x##h##_avx512(const uint8_t *src_ptr, int src_stride,  \
                                      const uint8_t *ref_ptr,                  \
                                      int ref_stride) {                        \
    return sad32xh_avx512(src_ptr, src_stride, ref_ptr, ref_stride, h);        \
  }                                                                            \
  unsigned int aom_sad_skip_32x##h##_avx512(                                   \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,          \
      int ref_stride) {                                                        \
    return 2 * sad32xh_avx512(src_ptr, src_stride * 2, ref_ptr,                \
                              ref_stride * 2, (h) / 2);                        \
  }                                                                            \
  unsigned int aom_sad32x##h##_avg_avx512(                                     \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,          \
      int ref_stride, const uint8_t *second_pred) {                            \
    return sad32xh_avg_avx512(src_ptr, src_stride, ref_ptr, ref_stride, h,     \
                              second_pred);                                    \
  }                                                                            \
  void aom_sad32x##h##x4d_avx512(const uint8_t *src_ptr, int src_stride,       \
                                 const uint8_t *const ref_ptr[4],              \
                                 int ref_stride, uint32_t sad_array[4]) {      \
    sad32xhx4d_avx512(h, src_ptr, src_stride, ref_ptr, ref_stride, sad_array); \
  }                                                                            \
  void aom_sad_skip_32x##h##x4d_avx512(                                        \
      const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_ptr[4], \
      int ref_stride, uint32_t sad_array[4]) {                                 \
    sad32xhx4d_avx512((h) >> 1, src_ptr, src_stride << 1, ref_ptr,             \
                      ref_stride << 1, sad_array);                             \
    sad_array[0] <<= 1;                                                        \
    sad_array[1] <<= 1;                                                        \
    sad_array[2] <<= 1;                                                        \
    sad_array[3] <<= 1;                                                        \
  }                                                                            \
  void aom_sad32x##h##x3d_avx512(const uint8_t *src_ptr, int src_stride,       \
                                 const uint8_t *const ref_ptr[4],              \
                                 int ref_stride, uint32_t sad_array[4]) {      \
    sad32xhx3d_avx512(h, src_ptr, src_stride, ref_ptr, ref_stride, sad_array); \
  }

SAD32_AVX512(64)
SAD32_AVX512(32)
SAD32_AVX512(16)

#if !CONFIG_REALTIME_ONLY
unsigned int aom_sad32x8_avx512(const uint8_t *src_ptr, int src_stride,
                                const uint8_t *ref_ptr, int ref_stride) {
  return sad32xh_avx512(src_ptr, src_stride, ref_ptr, ref_stride, 8);
}

unsigned int aom_sad32x8_avg_avx512(const uint8_t *src_ptr, int src_stride,
                                    const uint8_t *ref_ptr, int ref_stride,
                                    const uint8_t *second_pred) {
  return sad32xh_avg_avx512(src_ptr, src_stride, ref_ptr, ref_stride, 8,
                            second_pred);
}

void aom_sad32x8x4d_avx512(const uint8_t *src_ptr, int src_stride,
                           const uint8_t *const ref_ptr[4], int ref_stride,
                           uint32_t sad_array[4]) {
  sad32xhx4d_avx512(8, src_ptr, src_stride, ref_ptr, ref_stride, sad_array);
}

void aom_sad32x8x3d_avx512(const uint8_t *src_ptr, int src_stride,
                           const uint8_t *const ref_ptr[4], int ref_stride,
                           uint32_t sad_array[4]) {
  sad32xhx3d_avx512(8, src_ptr, src_stride, ref_ptr, ref_stride, sad_array);
}
#endif  // !CONFIG_REALTIME_ONLY

// 16xh blocks: pack 4 rows (64 bytes) per 512-bit ZMM register.
static AOM_FORCE_INLINE __m512i loadu4_128_zmm(const uint8_t *p, int stride) {
  const __m256i lo = yy_loadu2_128(p + stride, p);
  const __m256i hi = yy_loadu2_128(p + stride * 3, p + stride * 2);
  return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

static AOM_FORCE_INLINE unsigned int sad16xh_avx512(const uint8_t *src_ptr,
                                                    int src_stride,
                                                    const uint8_t *ref_ptr,
                                                    int ref_stride, int h) {
  __m512i sum_sad = _mm512_setzero_si512();
  for (int i = 0; i < h; i += 4) {
    const __m512i s = loadu4_128_zmm(src_ptr, src_stride);
    const __m512i r = loadu4_128_zmm(ref_ptr, ref_stride);
    sum_sad = _mm512_add_epi32(sum_sad, _mm512_sad_epu8(r, s));
    src_ptr += src_stride * 4;
    ref_ptr += ref_stride * 4;
  }
  return hsum_sad_zmm(sum_sad);
}

static AOM_FORCE_INLINE unsigned int sad16xh_avg_avx512(
    const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,
    int ref_stride, int h, const uint8_t *second_pred) {
  __m512i sum_sad = _mm512_setzero_si512();
  for (int i = 0; i < h; i += 4) {
    const __m512i s = loadu4_128_zmm(src_ptr, src_stride);
    const __m512i r =
        _mm512_avg_epu8(loadu4_128_zmm(ref_ptr, ref_stride),
                        _mm512_loadu_si512((const __m512i *)second_pred));
    sum_sad = _mm512_add_epi32(sum_sad, _mm512_sad_epu8(r, s));
    src_ptr += src_stride * 4;
    ref_ptr += ref_stride * 4;
    second_pred += 64;
  }
  return hsum_sad_zmm(sum_sad);
}

static AOM_FORCE_INLINE void sad16xhx4d_avx512(int h, const uint8_t *src,
                                               int src_stride,
                                               const uint8_t *const ref[4],
                                               int ref_stride, uint32_t *res) {
  const uint8_t *ref0 = ref[0];
  const uint8_t *ref1 = ref[1];
  const uint8_t *ref2 = ref[2];
  const uint8_t *ref3 = ref[3];
  __m512i sum_ref0 = _mm512_setzero_si512();
  __m512i sum_ref1 = _mm512_setzero_si512();
  __m512i sum_ref2 = _mm512_setzero_si512();
  __m512i sum_ref3 = _mm512_setzero_si512();

  for (int i = 0; i < h; i += 4) {
    const __m512i s = loadu4_128_zmm(src, src_stride);
    const __m512i r0 = loadu4_128_zmm(ref0, ref_stride);
    const __m512i r1 = loadu4_128_zmm(ref1, ref_stride);
    const __m512i r2 = loadu4_128_zmm(ref2, ref_stride);
    const __m512i r3 = loadu4_128_zmm(ref3, ref_stride);
    sum_ref0 = _mm512_add_epi32(sum_ref0, _mm512_sad_epu8(r0, s));
    sum_ref1 = _mm512_add_epi32(sum_ref1, _mm512_sad_epu8(r1, s));
    sum_ref2 = _mm512_add_epi32(sum_ref2, _mm512_sad_epu8(r2, s));
    sum_ref3 = _mm512_add_epi32(sum_ref3, _mm512_sad_epu8(r3, s));
    src += src_stride * 4;
    ref0 += ref_stride * 4;
    ref1 += ref_stride * 4;
    ref2 += ref_stride * 4;
    ref3 += ref_stride * 4;
  }
  aggregate_and_store_sum_zmm(res, sum_ref0, sum_ref1, sum_ref2, sum_ref3);
}

static AOM_FORCE_INLINE void sad16xhx3d_avx512(int h, const uint8_t *src,
                                               int src_stride,
                                               const uint8_t *const ref[4],
                                               int ref_stride, uint32_t *res) {
  const uint8_t *ref0 = ref[0];
  const uint8_t *ref1 = ref[1];
  const uint8_t *ref2 = ref[2];
  __m512i sum_ref0 = _mm512_setzero_si512();
  __m512i sum_ref1 = _mm512_setzero_si512();
  __m512i sum_ref2 = _mm512_setzero_si512();

  for (int i = 0; i < h; i += 4) {
    const __m512i s = loadu4_128_zmm(src, src_stride);
    const __m512i r0 = loadu4_128_zmm(ref0, ref_stride);
    const __m512i r1 = loadu4_128_zmm(ref1, ref_stride);
    const __m512i r2 = loadu4_128_zmm(ref2, ref_stride);
    sum_ref0 = _mm512_add_epi32(sum_ref0, _mm512_sad_epu8(r0, s));
    sum_ref1 = _mm512_add_epi32(sum_ref1, _mm512_sad_epu8(r1, s));
    sum_ref2 = _mm512_add_epi32(sum_ref2, _mm512_sad_epu8(r2, s));
    src += src_stride * 4;
    ref0 += ref_stride * 4;
    ref1 += ref_stride * 4;
    ref2 += ref_stride * 4;
  }
  aggregate_and_store_sum_3d_zmm(res, sum_ref0, sum_ref1, sum_ref2);
}

#define SAD16_AVX512(h)                                                        \
  unsigned int aom_sad16x##h##_avx512(const uint8_t *src_ptr, int src_stride,  \
                                      const uint8_t *ref_ptr,                  \
                                      int ref_stride) {                        \
    return sad16xh_avx512(src_ptr, src_stride, ref_ptr, ref_stride, h);        \
  }                                                                            \
  unsigned int aom_sad_skip_16x##h##_avx512(                                   \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,          \
      int ref_stride) {                                                        \
    return 2 * sad16xh_avx512(src_ptr, src_stride * 2, ref_ptr,                \
                              ref_stride * 2, (h) / 2);                        \
  }                                                                            \
  unsigned int aom_sad16x##h##_avg_avx512(                                     \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,          \
      int ref_stride, const uint8_t *second_pred) {                            \
    return sad16xh_avg_avx512(src_ptr, src_stride, ref_ptr, ref_stride, h,     \
                              second_pred);                                    \
  }                                                                            \
  void aom_sad16x##h##x4d_avx512(const uint8_t *src_ptr, int src_stride,       \
                                 const uint8_t *const ref_ptr[4],              \
                                 int ref_stride, uint32_t sad_array[4]) {      \
    sad16xhx4d_avx512(h, src_ptr, src_stride, ref_ptr, ref_stride, sad_array); \
  }                                                                            \
  void aom_sad_skip_16x##h##x4d_avx512(                                        \
      const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_ptr[4], \
      int ref_stride, uint32_t sad_array[4]) {                                 \
    sad16xhx4d_avx512((h) >> 1, src_ptr, src_stride << 1, ref_ptr,             \
                      ref_stride << 1, sad_array);                             \
    sad_array[0] <<= 1;                                                        \
    sad_array[1] <<= 1;                                                        \
    sad_array[2] <<= 1;                                                        \
    sad_array[3] <<= 1;                                                        \
  }                                                                            \
  void aom_sad16x##h##x3d_avx512(const uint8_t *src_ptr, int src_stride,       \
                                 const uint8_t *const ref_ptr[4],              \
                                 int ref_stride, uint32_t sad_array[4]) {      \
    sad16xhx3d_avx512(h, src_ptr, src_stride, ref_ptr, ref_stride, sad_array); \
  }

SAD16_AVX512(32)
SAD16_AVX512(16)

unsigned int aom_sad16x8_avx512(const uint8_t *src_ptr, int src_stride,
                                const uint8_t *ref_ptr, int ref_stride) {
  return sad16xh_avx512(src_ptr, src_stride, ref_ptr, ref_stride, 8);
}

unsigned int aom_sad16x8_avg_avx512(const uint8_t *src_ptr, int src_stride,
                                    const uint8_t *ref_ptr, int ref_stride,
                                    const uint8_t *second_pred) {
  return sad16xh_avg_avx512(src_ptr, src_stride, ref_ptr, ref_stride, 8,
                            second_pred);
}

void aom_sad16x8x4d_avx512(const uint8_t *src_ptr, int src_stride,
                           const uint8_t *const ref_ptr[4], int ref_stride,
                           uint32_t sad_array[4]) {
  sad16xhx4d_avx512(8, src_ptr, src_stride, ref_ptr, ref_stride, sad_array);
}

void aom_sad16x8x3d_avx512(const uint8_t *src_ptr, int src_stride,
                           const uint8_t *const ref_ptr[4], int ref_stride,
                           uint32_t sad_array[4]) {
  sad16xhx3d_avx512(8, src_ptr, src_stride, ref_ptr, ref_stride, sad_array);
}

#if !CONFIG_REALTIME_ONLY
SAD16_AVX512(64)

unsigned int aom_sad16x4_avx512(const uint8_t *src_ptr, int src_stride,
                                const uint8_t *ref_ptr, int ref_stride) {
  return sad16xh_avx512(src_ptr, src_stride, ref_ptr, ref_stride, 4);
}

void aom_sad16x4x4d_avx512(const uint8_t *src_ptr, int src_stride,
                           const uint8_t *const ref_ptr[4], int ref_stride,
                           uint32_t sad_array[4]) {
  sad16xhx4d_avx512(4, src_ptr, src_stride, ref_ptr, ref_stride, sad_array);
}

void aom_sad16x4x3d_avx512(const uint8_t *src_ptr, int src_stride,
                           const uint8_t *const ref_ptr[4], int ref_stride,
                           uint32_t sad_array[4]) {
  sad16xhx3d_avx512(4, src_ptr, src_stride, ref_ptr, ref_stride, sad_array);
}
#endif  // !CONFIG_REALTIME_ONLY
