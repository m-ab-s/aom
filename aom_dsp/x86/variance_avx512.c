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

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"

#include "aom/aom_integer.h"
#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/aom_filter.h"
#include "aom_dsp/blend.h"
#include "aom_dsp/x86/masked_sad_intrin_ssse3.h"
#include "aom_dsp/x86/obmc_intrinsic_sse4.h"
#include "aom_dsp/x86/obmc_intrinsic_ssse3.h"
#include "aom_dsp/x86/synonyms.h"
#include "aom_dsp/x86/synonyms_avx2.h"
#include "aom_dsp/x86/variance_impl_ssse3.h"
#include "aom_ports/mem.h"

static AOM_FORCE_INLINE int32_t hsum_epi32_zmm(__m512i v) {
  const __m256i v256 = _mm256_add_epi32(_mm512_castsi512_si256(v),
                                        _mm512_extracti64x4_epi64(v, 1));
  const __m128i v128 = _mm_add_epi32(_mm256_castsi256_si128(v256),
                                     _mm256_extracti128_si256(v256, 1));
  const __m128i s1 = _mm_hadd_epi32(v128, v128);
  const __m128i s2 = _mm_hadd_epi32(s1, s1);
  return _mm_cvtsi128_si32(s2);
}

static AOM_FORCE_INLINE int64_t hsum_epi64_zmm(__m512i v) {
  const __m256i v256 = _mm256_add_epi64(_mm512_castsi512_si256(v),
                                        _mm512_extracti64x4_epi64(v, 1));
  const __m128i v128 = _mm_add_epi64(_mm256_castsi256_si128(v256),
                                     _mm256_extracti128_si256(v256, 1));
  const __m128i s = _mm_add_epi64(v128, _mm_srli_si128(v128, 8));
  return _mm_cvtsi128_si64(s);
}

static AOM_FORCE_INLINE void variance_kernel_zmm(const __m512i src,
                                                 const __m512i ref,
                                                 __m512i *const sse,
                                                 __m512i *const sum) {
  const __m512i adj_sub = _mm512_set1_epi16((int16_t)0xff01);
  const __m512i s_lo = _mm512_unpacklo_epi8(src, ref);
  const __m512i s_hi = _mm512_unpackhi_epi8(src, ref);
  const __m512i diff_lo = _mm512_maddubs_epi16(s_lo, adj_sub);
  const __m512i diff_hi = _mm512_maddubs_epi16(s_hi, adj_sub);

  *sum = _mm512_add_epi16(*sum, _mm512_add_epi16(diff_lo, diff_hi));
  *sse = _mm512_add_epi32(
      *sse, _mm512_add_epi32(_mm512_madd_epi16(diff_lo, diff_lo),
                             _mm512_madd_epi16(diff_hi, diff_hi)));
}

static AOM_FORCE_INLINE int variance_final_from_16bit_sum_zmm(
    __m512i vsse, __m512i vsum, unsigned int *const sse) {
  const __m512i vsum32 = _mm512_madd_epi16(vsum, _mm512_set1_epi16(1));
  const __m256i sse256 = _mm256_add_epi32(_mm512_castsi512_si256(vsse),
                                          _mm512_extracti64x4_epi64(vsse, 1));
  const __m256i sum256 = _mm256_add_epi32(_mm512_castsi512_si256(vsum32),
                                          _mm512_extracti64x4_epi64(vsum32, 1));
  const __m128i sse128 = _mm_add_epi32(_mm256_castsi256_si128(sse256),
                                       _mm256_extracti128_si256(sse256, 1));
  const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum256),
                                       _mm256_extracti128_si256(sum256, 1));
  const __m128i sse_sum = _mm_hadd_epi32(sse128, sum128);
  const __m128i res = _mm_hadd_epi32(sse_sum, sse_sum);
  *sse = (unsigned int)_mm_cvtsi128_si32(res);
  return _mm_extract_epi32(res, 1);
}

static AOM_FORCE_INLINE int variance_final_from_32bit_sum_zmm(
    __m512i vsse, __m512i vsum32, unsigned int *const sse) {
  const __m256i sse256 = _mm256_add_epi32(_mm512_castsi512_si256(vsse),
                                          _mm512_extracti64x4_epi64(vsse, 1));
  const __m256i sum256 = _mm256_add_epi32(_mm512_castsi512_si256(vsum32),
                                          _mm512_extracti64x4_epi64(vsum32, 1));
  const __m128i sse128 = _mm_add_epi32(_mm256_castsi256_si128(sse256),
                                       _mm256_extracti128_si256(sse256, 1));
  const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum256),
                                       _mm256_extracti128_si256(sum256, 1));
  const __m128i sse_sum = _mm_hadd_epi32(sse128, sum128);
  const __m128i res = _mm_hadd_epi32(sse_sum, sse_sum);
  *sse = (unsigned int)_mm_cvtsi128_si32(res);
  return _mm_extract_epi32(res, 1);
}

static AOM_FORCE_INLINE __m512i loadu2_256_zmm(const uint8_t *hi,
                                               const uint8_t *lo) {
  return _mm512_inserti64x4(
      _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)lo)),
      _mm256_loadu_si256((const __m256i *)hi), 1);
}

static AOM_FORCE_INLINE __m512i loadu4_128_zmm(const uint8_t *p, int stride) {
  const __m256i lo = yy_loadu2_128(p + stride, p);
  const __m256i hi = yy_loadu2_128(p + stride * 3, p + stride * 2);
  return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

static AOM_FORCE_INLINE void variance16_avx512(const uint8_t *src,
                                               int src_stride,
                                               const uint8_t *ref,
                                               int ref_stride, int h,
                                               __m512i *vsse, __m512i *vsum) {
  *vsum = _mm512_setzero_si512();
  for (int i = 0; i < h; i += 4) {
    const __m512i s = loadu4_128_zmm(src, src_stride);
    const __m512i r = loadu4_128_zmm(ref, ref_stride);
    variance_kernel_zmm(s, r, vsse, vsum);
    src += 4 * src_stride;
    ref += 4 * ref_stride;
  }
}

static AOM_FORCE_INLINE void variance32_avx512(const uint8_t *src,
                                               int src_stride,
                                               const uint8_t *ref,
                                               int ref_stride, int h,
                                               __m512i *vsse, __m512i *vsum) {
  *vsum = _mm512_setzero_si512();
  for (int i = 0; i < h; i += 2) {
    const __m512i s = loadu2_256_zmm(src + src_stride, src);
    const __m512i r = loadu2_256_zmm(ref + ref_stride, ref);
    variance_kernel_zmm(s, r, vsse, vsum);
    src += 2 * src_stride;
    ref += 2 * ref_stride;
  }
}

static AOM_FORCE_INLINE void variance64_avx512(const uint8_t *src,
                                               int src_stride,
                                               const uint8_t *ref,
                                               int ref_stride, int h,
                                               __m512i *vsse, __m512i *vsum) {
  *vsum = _mm512_setzero_si512();
  for (int i = 0; i < h; i++) {
    const __m512i s = _mm512_loadu_si512((const __m512i *)src);
    const __m512i r = _mm512_loadu_si512((const __m512i *)ref);
    variance_kernel_zmm(s, r, vsse, vsum);
    src += src_stride;
    ref += ref_stride;
  }
}

static AOM_FORCE_INLINE void variance128_avx512(const uint8_t *src,
                                                int src_stride,
                                                const uint8_t *ref,
                                                int ref_stride, int h,
                                                __m512i *vsse, __m512i *vsum) {
  *vsum = _mm512_setzero_si512();
  for (int i = 0; i < h; i++) {
    variance_kernel_zmm(_mm512_loadu_si512((const __m512i *)src),
                        _mm512_loadu_si512((const __m512i *)ref), vsse, vsum);
    variance_kernel_zmm(_mm512_loadu_si512((const __m512i *)(src + 64)),
                        _mm512_loadu_si512((const __m512i *)(ref + 64)), vsse,
                        vsum);
    src += src_stride;
    ref += ref_stride;
  }
}

// In a 512-bit ZMM register, vsum has 32 int16_t lanes.
// Each 64-byte ZMM kernel adds 2 pixels per lane, so up to 64 rows * 64 cols
// (= 4096 pixels, 128 pixels per lane, max 128*255 = 32640 <= 32767) fits in
// 16-bit vsum without any inner chunk loop!
#define AOM_VAR_NO_LOOP_AVX512(bw, bh, bits)                                   \
  unsigned int aom_variance##bw##x##bh##_avx512(                               \
      const uint8_t *src, int src_stride, const uint8_t *ref, int ref_stride,  \
      unsigned int *sse) {                                                     \
    __m512i vsse = _mm512_setzero_si512();                                     \
    __m512i vsum;                                                              \
    variance##bw##_avx512(src, src_stride, ref, ref_stride, bh, &vsse, &vsum); \
    const int sum = variance_final_from_16bit_sum_zmm(vsse, vsum, sse);        \
    return *sse - (uint32_t)(((int64_t)sum * sum) >> (bits));                  \
  }

AOM_VAR_NO_LOOP_AVX512(16, 8, 7)
AOM_VAR_NO_LOOP_AVX512(16, 16, 8)
AOM_VAR_NO_LOOP_AVX512(16, 32, 9)

AOM_VAR_NO_LOOP_AVX512(32, 16, 9)
AOM_VAR_NO_LOOP_AVX512(32, 32, 10)
AOM_VAR_NO_LOOP_AVX512(32, 64, 11)

AOM_VAR_NO_LOOP_AVX512(64, 32, 11)
AOM_VAR_NO_LOOP_AVX512(64, 64, 12)

#if !CONFIG_REALTIME_ONLY
AOM_VAR_NO_LOOP_AVX512(16, 4, 6)
AOM_VAR_NO_LOOP_AVX512(16, 64, 10)
AOM_VAR_NO_LOOP_AVX512(32, 8, 8)
AOM_VAR_NO_LOOP_AVX512(64, 16, 10)
#endif  // !CONFIG_REALTIME_ONLY

#define AOM_VAR_LOOP_AVX512(bw, bh, bits, uh)                                 \
  unsigned int aom_variance##bw##x##bh##_avx512(                              \
      const uint8_t *src, int src_stride, const uint8_t *ref, int ref_stride, \
      unsigned int *sse) {                                                    \
    __m512i vsse = _mm512_setzero_si512();                                    \
    __m512i vsum32 = _mm512_setzero_si512();                                  \
    const __m512i one = _mm512_set1_epi16(1);                                 \
    for (int i = 0; i < (bh) / (uh); i++) {                                   \
      __m512i vsum16;                                                         \
      variance##bw##_avx512(src, src_stride, ref, ref_stride, uh, &vsse,      \
                            &vsum16);                                         \
      vsum32 = _mm512_add_epi32(vsum32, _mm512_madd_epi16(vsum16, one));      \
      src += (uh) * src_stride;                                               \
      ref += (uh) * ref_stride;                                               \
    }                                                                         \
    const int sum = variance_final_from_32bit_sum_zmm(vsse, vsum32, sse);     \
    return *sse - (unsigned int)(((int64_t)sum * sum) >> (bits));             \
  }

AOM_VAR_LOOP_AVX512(64, 128, 13, 64)
AOM_VAR_LOOP_AVX512(128, 64, 13, 32)
AOM_VAR_LOOP_AVX512(128, 128, 14, 32)

unsigned int aom_mse16x16_avx512(const uint8_t *src, int src_stride,
                                 const uint8_t *ref, int ref_stride,
                                 unsigned int *sse) {
  aom_variance16x16_avx512(src, src_stride, ref, ref_stride, sse);
  return *sse;
}

// -----------------------------------------------------------------------------
// 512-bit ZMM Subpixel Variance & Subpixel Avg Variance
// -----------------------------------------------------------------------------

static AOM_FORCE_INLINE __m512i bilinear_filter_zmm(int offset) {
  const int f0 = 16 - 2 * offset;
  const int f1 = 2 * offset;
  return _mm512_set1_epi16((int16_t)(f0 | (f1 << 8)));
}

static AOM_FORCE_INLINE int sub_pixel_variance64xh_imp_avx512(
    const uint8_t *src, int src_stride, int x_offset, int y_offset,
    const uint8_t *dst, int dst_stride, int height, unsigned int *sse) {
  __m512i sum_reg = _mm512_setzero_si512();
  __m512i sse_reg = _mm512_setzero_si512();
  const __m512i zero_reg = _mm512_setzero_si512();
  const __m512i pw8 = _mm512_set1_epi16(8);

#define ACCUM_DIFF_ZMM(exp_src_lo, exp_src_hi, dst_reg)                 \
  do {                                                                  \
    const __m512i exp_dst_lo = _mm512_unpacklo_epi8(dst_reg, zero_reg); \
    const __m512i exp_dst_hi = _mm512_unpackhi_epi8(dst_reg, zero_reg); \
    const __m512i d_lo = _mm512_sub_epi16(exp_src_lo, exp_dst_lo);      \
    const __m512i d_hi = _mm512_sub_epi16(exp_src_hi, exp_dst_hi);      \
    sum_reg = _mm512_add_epi16(sum_reg, _mm512_add_epi16(d_lo, d_hi));  \
    sse_reg = _mm512_add_epi32(                                         \
        sse_reg, _mm512_add_epi32(_mm512_madd_epi16(d_lo, d_lo),        \
                                  _mm512_madd_epi16(d_hi, d_hi)));      \
  } while (0)

#define FILTER_PAIR_ZMM(s0, s1, filt, out_lo, out_hi)                  \
  do {                                                                 \
    out_lo = _mm512_maddubs_epi16(_mm512_unpacklo_epi8(s0, s1), filt); \
    out_hi = _mm512_maddubs_epi16(_mm512_unpackhi_epi8(s0, s1), filt); \
    out_lo = _mm512_srai_epi16(_mm512_add_epi16(out_lo, pw8), 4);      \
    out_hi = _mm512_srai_epi16(_mm512_add_epi16(out_hi, pw8), 4);      \
  } while (0)

  if (x_offset == 0) {
    if (y_offset == 0) {
      for (int i = 0; i < height; i++) {
        const __m512i s = _mm512_loadu_si512((const __m512i *)src);
        const __m512i d = _mm512_loadu_si512((const __m512i *)dst);
        variance_kernel_zmm(s, d, &sse_reg, &sum_reg);
        src += src_stride;
        dst += dst_stride;
      }
    } else if (y_offset == 4) {
      for (int i = 0; i < height; i++) {
        const __m512i s = _mm512_avg_epu8(
            _mm512_loadu_si512((const __m512i *)src),
            _mm512_loadu_si512((const __m512i *)(src + src_stride)));
        const __m512i d = _mm512_loadu_si512((const __m512i *)dst);
        variance_kernel_zmm(s, d, &sse_reg, &sum_reg);
        src += src_stride;
        dst += dst_stride;
      }
    } else {
      const __m512i filter = bilinear_filter_zmm(y_offset);
      for (int i = 0; i < height; i++) {
        const __m512i s0 = _mm512_loadu_si512((const __m512i *)src);
        const __m512i s1 =
            _mm512_loadu_si512((const __m512i *)(src + src_stride));
        const __m512i d = _mm512_loadu_si512((const __m512i *)dst);
        __m512i exp_lo, exp_hi;
        FILTER_PAIR_ZMM(s0, s1, filter, exp_lo, exp_hi);
        ACCUM_DIFF_ZMM(exp_lo, exp_hi, d);
        src += src_stride;
        dst += dst_stride;
      }
    }
  } else if (x_offset == 4) {
    if (y_offset == 0) {
      for (int i = 0; i < height; i++) {
        const __m512i s =
            _mm512_avg_epu8(_mm512_loadu_si512((const __m512i *)src),
                            _mm512_loadu_si512((const __m512i *)(src + 1)));
        const __m512i d = _mm512_loadu_si512((const __m512i *)dst);
        variance_kernel_zmm(s, d, &sse_reg, &sum_reg);
        src += src_stride;
        dst += dst_stride;
      }
    } else if (y_offset == 4) {
      __m512i s_prev =
          _mm512_avg_epu8(_mm512_loadu_si512((const __m512i *)src),
                          _mm512_loadu_si512((const __m512i *)(src + 1)));
      for (int i = 0; i < height; i++) {
        src += src_stride;
        const __m512i s_curr =
            _mm512_avg_epu8(_mm512_loadu_si512((const __m512i *)src),
                            _mm512_loadu_si512((const __m512i *)(src + 1)));
        const __m512i s_avg = _mm512_avg_epu8(s_prev, s_curr);
        const __m512i d = _mm512_loadu_si512((const __m512i *)dst);
        variance_kernel_zmm(s_avg, d, &sse_reg, &sum_reg);
        s_prev = s_curr;
        dst += dst_stride;
      }
    } else {
      const __m512i filter = bilinear_filter_zmm(y_offset);
      __m512i s_prev =
          _mm512_avg_epu8(_mm512_loadu_si512((const __m512i *)src),
                          _mm512_loadu_si512((const __m512i *)(src + 1)));
      for (int i = 0; i < height; i++) {
        src += src_stride;
        const __m512i s_curr =
            _mm512_avg_epu8(_mm512_loadu_si512((const __m512i *)src),
                            _mm512_loadu_si512((const __m512i *)(src + 1)));
        const __m512i d = _mm512_loadu_si512((const __m512i *)dst);
        __m512i exp_lo, exp_hi;
        FILTER_PAIR_ZMM(s_prev, s_curr, filter, exp_lo, exp_hi);
        ACCUM_DIFF_ZMM(exp_lo, exp_hi, d);
        s_prev = s_curr;
        dst += dst_stride;
      }
    }
  } else {
    const __m512i xfilter = bilinear_filter_zmm(x_offset);
    if (y_offset == 0) {
      for (int i = 0; i < height; i++) {
        const __m512i s0 = _mm512_loadu_si512((const __m512i *)src);
        const __m512i s1 = _mm512_loadu_si512((const __m512i *)(src + 1));
        const __m512i d = _mm512_loadu_si512((const __m512i *)dst);
        __m512i exp_lo, exp_hi;
        FILTER_PAIR_ZMM(s0, s1, xfilter, exp_lo, exp_hi);
        ACCUM_DIFF_ZMM(exp_lo, exp_hi, d);
        src += src_stride;
        dst += dst_stride;
      }
    } else if (y_offset == 4) {
      __m512i exp_lo, exp_hi;
      FILTER_PAIR_ZMM(_mm512_loadu_si512((const __m512i *)src),
                      _mm512_loadu_si512((const __m512i *)(src + 1)), xfilter,
                      exp_lo, exp_hi);
      __m512i src_pack = _mm512_packus_epi16(exp_lo, exp_hi);
      for (int i = 0; i < height; i++) {
        src += src_stride;
        FILTER_PAIR_ZMM(_mm512_loadu_si512((const __m512i *)src),
                        _mm512_loadu_si512((const __m512i *)(src + 1)), xfilter,
                        exp_lo, exp_hi);
        const __m512i src_curr = _mm512_packus_epi16(exp_lo, exp_hi);
        const __m512i src_avg = _mm512_avg_epu8(src_pack, src_curr);
        const __m512i d = _mm512_loadu_si512((const __m512i *)dst);
        variance_kernel_zmm(src_avg, d, &sse_reg, &sum_reg);
        src_pack = src_curr;
        dst += dst_stride;
      }
    } else {
      const __m512i yfilter = bilinear_filter_zmm(y_offset);
      __m512i exp_lo, exp_hi;
      FILTER_PAIR_ZMM(_mm512_loadu_si512((const __m512i *)src),
                      _mm512_loadu_si512((const __m512i *)(src + 1)), xfilter,
                      exp_lo, exp_hi);
      __m512i src_pack = _mm512_packus_epi16(exp_lo, exp_hi);
      for (int i = 0; i < height; i++) {
        src += src_stride;
        FILTER_PAIR_ZMM(_mm512_loadu_si512((const __m512i *)src),
                        _mm512_loadu_si512((const __m512i *)(src + 1)), xfilter,
                        exp_lo, exp_hi);
        const __m512i src_curr = _mm512_packus_epi16(exp_lo, exp_hi);
        __m512i v_lo, v_hi;
        FILTER_PAIR_ZMM(src_pack, src_curr, yfilter, v_lo, v_hi);
        const __m512i d = _mm512_loadu_si512((const __m512i *)dst);
        ACCUM_DIFF_ZMM(v_lo, v_hi, d);
        src_pack = src_curr;
        dst += dst_stride;
      }
    }
  }
  return variance_final_from_16bit_sum_zmm(sse_reg, sum_reg, sse);
}

static AOM_FORCE_INLINE int sub_pixel_avg_variance64xh_imp_avx512(
    const uint8_t *src, int src_stride, int x_offset, int y_offset,
    const uint8_t *dst, int dst_stride, const uint8_t *sec, int sec_stride,
    int height, unsigned int *sse) {
  __m512i sum_reg = _mm512_setzero_si512();
  __m512i sse_reg = _mm512_setzero_si512();
  const __m512i pw8 = _mm512_set1_epi16(8);

  if (x_offset == 0) {
    if (y_offset == 0) {
      for (int i = 0; i < height; i++) {
        const __m512i s =
            _mm512_avg_epu8(_mm512_loadu_si512((const __m512i *)src),
                            _mm512_loadu_si512((const __m512i *)sec));
        const __m512i d = _mm512_loadu_si512((const __m512i *)dst);
        variance_kernel_zmm(s, d, &sse_reg, &sum_reg);
        src += src_stride;
        dst += dst_stride;
        sec += sec_stride;
      }
    } else if (y_offset == 4) {
      for (int i = 0; i < height; i++) {
        __m512i s = _mm512_avg_epu8(
            _mm512_loadu_si512((const __m512i *)src),
            _mm512_loadu_si512((const __m512i *)(src + src_stride)));
        s = _mm512_avg_epu8(s, _mm512_loadu_si512((const __m512i *)sec));
        const __m512i d = _mm512_loadu_si512((const __m512i *)dst);
        variance_kernel_zmm(s, d, &sse_reg, &sum_reg);
        src += src_stride;
        dst += dst_stride;
        sec += sec_stride;
      }
    } else {
      const __m512i filter = bilinear_filter_zmm(y_offset);
      for (int i = 0; i < height; i++) {
        __m512i exp_lo, exp_hi;
        FILTER_PAIR_ZMM(_mm512_loadu_si512((const __m512i *)src),
                        _mm512_loadu_si512((const __m512i *)(src + src_stride)),
                        filter, exp_lo, exp_hi);
        const __m512i s =
            _mm512_avg_epu8(_mm512_packus_epi16(exp_lo, exp_hi),
                            _mm512_loadu_si512((const __m512i *)sec));
        const __m512i d = _mm512_loadu_si512((const __m512i *)dst);
        variance_kernel_zmm(s, d, &sse_reg, &sum_reg);
        src += src_stride;
        dst += dst_stride;
        sec += sec_stride;
      }
    }
  } else if (x_offset == 4) {
    if (y_offset == 0) {
      for (int i = 0; i < height; i++) {
        __m512i s =
            _mm512_avg_epu8(_mm512_loadu_si512((const __m512i *)src),
                            _mm512_loadu_si512((const __m512i *)(src + 1)));
        s = _mm512_avg_epu8(s, _mm512_loadu_si512((const __m512i *)sec));
        const __m512i d = _mm512_loadu_si512((const __m512i *)dst);
        variance_kernel_zmm(s, d, &sse_reg, &sum_reg);
        src += src_stride;
        dst += dst_stride;
        sec += sec_stride;
      }
    } else if (y_offset == 4) {
      __m512i s_prev =
          _mm512_avg_epu8(_mm512_loadu_si512((const __m512i *)src),
                          _mm512_loadu_si512((const __m512i *)(src + 1)));
      for (int i = 0; i < height; i++) {
        src += src_stride;
        const __m512i s_curr =
            _mm512_avg_epu8(_mm512_loadu_si512((const __m512i *)src),
                            _mm512_loadu_si512((const __m512i *)(src + 1)));
        const __m512i s_avg =
            _mm512_avg_epu8(_mm512_avg_epu8(s_prev, s_curr),
                            _mm512_loadu_si512((const __m512i *)sec));
        const __m512i d = _mm512_loadu_si512((const __m512i *)dst);
        variance_kernel_zmm(s_avg, d, &sse_reg, &sum_reg);
        s_prev = s_curr;
        dst += dst_stride;
        sec += sec_stride;
      }
    } else {
      const __m512i filter = bilinear_filter_zmm(y_offset);
      __m512i s_prev =
          _mm512_avg_epu8(_mm512_loadu_si512((const __m512i *)src),
                          _mm512_loadu_si512((const __m512i *)(src + 1)));
      for (int i = 0; i < height; i++) {
        src += src_stride;
        const __m512i s_curr =
            _mm512_avg_epu8(_mm512_loadu_si512((const __m512i *)src),
                            _mm512_loadu_si512((const __m512i *)(src + 1)));
        __m512i exp_lo, exp_hi;
        FILTER_PAIR_ZMM(s_prev, s_curr, filter, exp_lo, exp_hi);
        const __m512i s_avg =
            _mm512_avg_epu8(_mm512_packus_epi16(exp_lo, exp_hi),
                            _mm512_loadu_si512((const __m512i *)sec));
        const __m512i d = _mm512_loadu_si512((const __m512i *)dst);
        variance_kernel_zmm(s_avg, d, &sse_reg, &sum_reg);
        s_prev = s_curr;
        dst += dst_stride;
        sec += sec_stride;
      }
    }
  } else {
    const __m512i xfilter = bilinear_filter_zmm(x_offset);
    if (y_offset == 0) {
      for (int i = 0; i < height; i++) {
        __m512i exp_lo, exp_hi;
        FILTER_PAIR_ZMM(_mm512_loadu_si512((const __m512i *)src),
                        _mm512_loadu_si512((const __m512i *)(src + 1)), xfilter,
                        exp_lo, exp_hi);
        const __m512i s =
            _mm512_avg_epu8(_mm512_packus_epi16(exp_lo, exp_hi),
                            _mm512_loadu_si512((const __m512i *)sec));
        const __m512i d = _mm512_loadu_si512((const __m512i *)dst);
        variance_kernel_zmm(s, d, &sse_reg, &sum_reg);
        src += src_stride;
        dst += dst_stride;
        sec += sec_stride;
      }
    } else if (y_offset == 4) {
      __m512i exp_lo, exp_hi;
      FILTER_PAIR_ZMM(_mm512_loadu_si512((const __m512i *)src),
                      _mm512_loadu_si512((const __m512i *)(src + 1)), xfilter,
                      exp_lo, exp_hi);
      __m512i src_pack = _mm512_packus_epi16(exp_lo, exp_hi);
      for (int i = 0; i < height; i++) {
        src += src_stride;
        FILTER_PAIR_ZMM(_mm512_loadu_si512((const __m512i *)src),
                        _mm512_loadu_si512((const __m512i *)(src + 1)), xfilter,
                        exp_lo, exp_hi);
        const __m512i src_curr = _mm512_packus_epi16(exp_lo, exp_hi);
        const __m512i src_avg =
            _mm512_avg_epu8(_mm512_avg_epu8(src_pack, src_curr),
                            _mm512_loadu_si512((const __m512i *)sec));
        const __m512i d = _mm512_loadu_si512((const __m512i *)dst);
        variance_kernel_zmm(src_avg, d, &sse_reg, &sum_reg);
        src_pack = src_curr;
        dst += dst_stride;
        sec += sec_stride;
      }
    } else {
      const __m512i yfilter = bilinear_filter_zmm(y_offset);
      __m512i exp_lo, exp_hi;
      FILTER_PAIR_ZMM(_mm512_loadu_si512((const __m512i *)src),
                      _mm512_loadu_si512((const __m512i *)(src + 1)), xfilter,
                      exp_lo, exp_hi);
      __m512i src_pack = _mm512_packus_epi16(exp_lo, exp_hi);
      for (int i = 0; i < height; i++) {
        src += src_stride;
        FILTER_PAIR_ZMM(_mm512_loadu_si512((const __m512i *)src),
                        _mm512_loadu_si512((const __m512i *)(src + 1)), xfilter,
                        exp_lo, exp_hi);
        const __m512i src_curr = _mm512_packus_epi16(exp_lo, exp_hi);
        __m512i v_lo, v_hi;
        FILTER_PAIR_ZMM(src_pack, src_curr, yfilter, v_lo, v_hi);
        const __m512i src_avg =
            _mm512_avg_epu8(_mm512_packus_epi16(v_lo, v_hi),
                            _mm512_loadu_si512((const __m512i *)sec));
        const __m512i d = _mm512_loadu_si512((const __m512i *)dst);
        variance_kernel_zmm(src_avg, d, &sse_reg, &sum_reg);
        src_pack = src_curr;
        dst += dst_stride;
        sec += sec_stride;
      }
    }
  }
  return variance_final_from_16bit_sum_zmm(sse_reg, sum_reg, sse);
}

// 32xh subpixel variance packing 2 rows of 32 bytes per 512-bit ZMM register.
static AOM_FORCE_INLINE int sub_pixel_variance32xh_imp_avx512(
    const uint8_t *src, int src_stride, int x_offset, int y_offset,
    const uint8_t *dst, int dst_stride, int height, unsigned int *sse) {
  __m512i sum_reg = _mm512_setzero_si512();
  __m512i sse_reg = _mm512_setzero_si512();
  const __m512i zero_reg = _mm512_setzero_si512();
  const __m512i pw8 = _mm512_set1_epi16(8);

  if (x_offset == 0) {
    if (y_offset == 0) {
      for (int i = 0; i < height; i += 2) {
        const __m512i s = loadu2_256_zmm(src + src_stride, src);
        const __m512i d = loadu2_256_zmm(dst + dst_stride, dst);
        variance_kernel_zmm(s, d, &sse_reg, &sum_reg);
        src += src_stride * 2;
        dst += dst_stride * 2;
      }
    } else if (y_offset == 4) {
      for (int i = 0; i < height; i += 2) {
        const __m512i s0 = loadu2_256_zmm(src + src_stride, src);
        const __m512i s1 =
            loadu2_256_zmm(src + src_stride * 2, src + src_stride);
        const __m512i d = loadu2_256_zmm(dst + dst_stride, dst);
        variance_kernel_zmm(_mm512_avg_epu8(s0, s1), d, &sse_reg, &sum_reg);
        src += src_stride * 2;
        dst += dst_stride * 2;
      }
    } else {
      const __m512i filter = bilinear_filter_zmm(y_offset);
      for (int i = 0; i < height; i += 2) {
        const __m512i s0 = loadu2_256_zmm(src + src_stride, src);
        const __m512i s1 =
            loadu2_256_zmm(src + src_stride * 2, src + src_stride);
        const __m512i d = loadu2_256_zmm(dst + dst_stride, dst);
        __m512i exp_lo, exp_hi;
        FILTER_PAIR_ZMM(s0, s1, filter, exp_lo, exp_hi);
        ACCUM_DIFF_ZMM(exp_lo, exp_hi, d);
        src += src_stride * 2;
        dst += dst_stride * 2;
      }
    }
  } else if (x_offset == 4) {
    if (y_offset == 0) {
      for (int i = 0; i < height; i += 2) {
        const __m512i s0 = loadu2_256_zmm(src + src_stride, src);
        const __m512i s1 = loadu2_256_zmm(src + src_stride + 1, src + 1);
        const __m512i d = loadu2_256_zmm(dst + dst_stride, dst);
        variance_kernel_zmm(_mm512_avg_epu8(s0, s1), d, &sse_reg, &sum_reg);
        src += src_stride * 2;
        dst += dst_stride * 2;
      }
    } else if (y_offset == 4) {
      for (int i = 0; i < height; i += 2) {
        const __m512i s_row01 =
            _mm512_avg_epu8(loadu2_256_zmm(src + src_stride, src),
                            loadu2_256_zmm(src + src_stride + 1, src + 1));
        const __m512i s_row12 = _mm512_avg_epu8(
            loadu2_256_zmm(src + src_stride * 2, src + src_stride),
            loadu2_256_zmm(src + src_stride * 2 + 1, src + src_stride + 1));
        const __m512i d = loadu2_256_zmm(dst + dst_stride, dst);
        variance_kernel_zmm(_mm512_avg_epu8(s_row01, s_row12), d, &sse_reg,
                            &sum_reg);
        src += src_stride * 2;
        dst += dst_stride * 2;
      }
    } else {
      const __m512i filter = bilinear_filter_zmm(y_offset);
      for (int i = 0; i < height; i += 2) {
        const __m512i s_row01 =
            _mm512_avg_epu8(loadu2_256_zmm(src + src_stride, src),
                            loadu2_256_zmm(src + src_stride + 1, src + 1));
        const __m512i s_row12 = _mm512_avg_epu8(
            loadu2_256_zmm(src + src_stride * 2, src + src_stride),
            loadu2_256_zmm(src + src_stride * 2 + 1, src + src_stride + 1));
        const __m512i d = loadu2_256_zmm(dst + dst_stride, dst);
        __m512i exp_lo, exp_hi;
        FILTER_PAIR_ZMM(s_row01, s_row12, filter, exp_lo, exp_hi);
        ACCUM_DIFF_ZMM(exp_lo, exp_hi, d);
        src += src_stride * 2;
        dst += dst_stride * 2;
      }
    }
  } else {
    const __m512i xfilter = bilinear_filter_zmm(x_offset);
    if (y_offset == 0) {
      for (int i = 0; i < height; i += 2) {
        const __m512i s0 = loadu2_256_zmm(src + src_stride, src);
        const __m512i s1 = loadu2_256_zmm(src + src_stride + 1, src + 1);
        const __m512i d = loadu2_256_zmm(dst + dst_stride, dst);
        __m512i exp_lo, exp_hi;
        FILTER_PAIR_ZMM(s0, s1, xfilter, exp_lo, exp_hi);
        ACCUM_DIFF_ZMM(exp_lo, exp_hi, d);
        src += src_stride * 2;
        dst += dst_stride * 2;
      }
    } else if (y_offset == 4) {
      __m512i exp_lo, exp_hi;
      FILTER_PAIR_ZMM(loadu2_256_zmm(src + src_stride, src),
                      loadu2_256_zmm(src + src_stride + 1, src + 1), xfilter,
                      exp_lo, exp_hi);
      __m512i p01 = _mm512_packus_epi16(exp_lo, exp_hi);
      src += src_stride * 2;
      for (int i = 0; i < height - 2; i += 2) {
        FILTER_PAIR_ZMM(loadu2_256_zmm(src + src_stride, src),
                        loadu2_256_zmm(src + src_stride + 1, src + 1), xfilter,
                        exp_lo, exp_hi);
        const __m512i p23 = _mm512_packus_epi16(exp_lo, exp_hi);
        const __m512i p12 = _mm512_inserti64x4(
            _mm512_castsi256_si512(_mm512_extracti64x4_epi64(p01, 1)),
            _mm512_castsi512_si256(p23), 1);
        const __m512i d = loadu2_256_zmm(dst + dst_stride, dst);
        variance_kernel_zmm(_mm512_avg_epu8(p01, p12), d, &sse_reg, &sum_reg);
        p01 = p23;
        src += src_stride * 2;
        dst += dst_stride * 2;
      }
      {
        const __m256i s0 = _mm256_loadu_si256((const __m256i *)src);
        const __m256i s1 = _mm256_loadu_si256((const __m256i *)(src + 1));
        const __m256i xf256 = _mm512_castsi512_si256(xfilter);
        const __m256i pw8_256 = _mm256_set1_epi16(8);
        const __m256i l = _mm256_srai_epi16(
            _mm256_add_epi16(
                _mm256_maddubs_epi16(_mm256_unpacklo_epi8(s0, s1), xf256),
                pw8_256),
            4);
        const __m256i h = _mm256_srai_epi16(
            _mm256_add_epi16(
                _mm256_maddubs_epi16(_mm256_unpackhi_epi8(s0, s1), xf256),
                pw8_256),
            4);
        const __m256i p_last = _mm256_packus_epi16(l, h);
        const __m512i p12 = _mm512_inserti64x4(
            _mm512_castsi256_si512(_mm512_extracti64x4_epi64(p01, 1)), p_last,
            1);
        const __m512i d = loadu2_256_zmm(dst + dst_stride, dst);
        variance_kernel_zmm(_mm512_avg_epu8(p01, p12), d, &sse_reg, &sum_reg);
      }
    } else {
      const __m512i yfilter = bilinear_filter_zmm(y_offset);
      __m512i exp_lo, exp_hi;
      FILTER_PAIR_ZMM(loadu2_256_zmm(src + src_stride, src),
                      loadu2_256_zmm(src + src_stride + 1, src + 1), xfilter,
                      exp_lo, exp_hi);
      __m512i p01 = _mm512_packus_epi16(exp_lo, exp_hi);
      src += src_stride * 2;
      for (int i = 0; i < height - 2; i += 2) {
        FILTER_PAIR_ZMM(loadu2_256_zmm(src + src_stride, src),
                        loadu2_256_zmm(src + src_stride + 1, src + 1), xfilter,
                        exp_lo, exp_hi);
        const __m512i p23 = _mm512_packus_epi16(exp_lo, exp_hi);
        const __m512i p12 = _mm512_inserti64x4(
            _mm512_castsi256_si512(_mm512_extracti64x4_epi64(p01, 1)),
            _mm512_castsi512_si256(p23), 1);
        __m512i v_lo, v_hi;
        FILTER_PAIR_ZMM(p01, p12, yfilter, v_lo, v_hi);
        const __m512i d = loadu2_256_zmm(dst + dst_stride, dst);
        ACCUM_DIFF_ZMM(v_lo, v_hi, d);
        p01 = p23;
        src += src_stride * 2;
        dst += dst_stride * 2;
      }
      {
        const __m256i s0 = _mm256_loadu_si256((const __m256i *)src);
        const __m256i s1 = _mm256_loadu_si256((const __m256i *)(src + 1));
        const __m256i xf256 = _mm512_castsi512_si256(xfilter);
        const __m256i pw8_256 = _mm256_set1_epi16(8);
        const __m256i l = _mm256_srai_epi16(
            _mm256_add_epi16(
                _mm256_maddubs_epi16(_mm256_unpacklo_epi8(s0, s1), xf256),
                pw8_256),
            4);
        const __m256i h = _mm256_srai_epi16(
            _mm256_add_epi16(
                _mm256_maddubs_epi16(_mm256_unpackhi_epi8(s0, s1), xf256),
                pw8_256),
            4);
        const __m256i p_last = _mm256_packus_epi16(l, h);
        const __m512i p12 = _mm512_inserti64x4(
            _mm512_castsi256_si512(_mm512_extracti64x4_epi64(p01, 1)), p_last,
            1);
        __m512i v_lo, v_hi;
        FILTER_PAIR_ZMM(p01, p12, yfilter, v_lo, v_hi);
        const __m512i d = loadu2_256_zmm(dst + dst_stride, dst);
        ACCUM_DIFF_ZMM(v_lo, v_hi, d);
      }
    }
  }
  return variance_final_from_16bit_sum_zmm(sse_reg, sum_reg, sse);
}

static AOM_FORCE_INLINE int sub_pixel_avg_variance32xh_imp_avx512(
    const uint8_t *src, int src_stride, int x_offset, int y_offset,
    const uint8_t *dst, int dst_stride, const uint8_t *sec, int sec_stride,
    int height, unsigned int *sse) {
  __m512i sum_reg = _mm512_setzero_si512();
  __m512i sse_reg = _mm512_setzero_si512();
  const __m512i pw8 = _mm512_set1_epi16(8);

  if (x_offset == 0) {
    if (y_offset == 0) {
      for (int i = 0; i < height; i += 2) {
        const __m512i s =
            _mm512_avg_epu8(loadu2_256_zmm(src + src_stride, src),
                            loadu2_256_zmm(sec + sec_stride, sec));
        const __m512i d = loadu2_256_zmm(dst + dst_stride, dst);
        variance_kernel_zmm(s, d, &sse_reg, &sum_reg);
        src += src_stride * 2;
        dst += dst_stride * 2;
        sec += sec_stride * 2;
      }
    } else if (y_offset == 4) {
      for (int i = 0; i < height; i += 2) {
        __m512i s = _mm512_avg_epu8(
            loadu2_256_zmm(src + src_stride, src),
            loadu2_256_zmm(src + src_stride * 2, src + src_stride));
        s = _mm512_avg_epu8(s, loadu2_256_zmm(sec + sec_stride, sec));
        const __m512i d = loadu2_256_zmm(dst + dst_stride, dst);
        variance_kernel_zmm(s, d, &sse_reg, &sum_reg);
        src += src_stride * 2;
        dst += dst_stride * 2;
        sec += sec_stride * 2;
      }
    } else {
      const __m512i filter = bilinear_filter_zmm(y_offset);
      for (int i = 0; i < height; i += 2) {
        __m512i exp_lo, exp_hi;
        FILTER_PAIR_ZMM(loadu2_256_zmm(src + src_stride, src),
                        loadu2_256_zmm(src + src_stride * 2, src + src_stride),
                        filter, exp_lo, exp_hi);
        const __m512i s =
            _mm512_avg_epu8(_mm512_packus_epi16(exp_lo, exp_hi),
                            loadu2_256_zmm(sec + sec_stride, sec));
        const __m512i d = loadu2_256_zmm(dst + dst_stride, dst);
        variance_kernel_zmm(s, d, &sse_reg, &sum_reg);
        src += src_stride * 2;
        dst += dst_stride * 2;
        sec += sec_stride * 2;
      }
    }
  } else if (x_offset == 4) {
    if (y_offset == 0) {
      for (int i = 0; i < height; i += 2) {
        __m512i s =
            _mm512_avg_epu8(loadu2_256_zmm(src + src_stride, src),
                            loadu2_256_zmm(src + src_stride + 1, src + 1));
        s = _mm512_avg_epu8(s, loadu2_256_zmm(sec + sec_stride, sec));
        const __m512i d = loadu2_256_zmm(dst + dst_stride, dst);
        variance_kernel_zmm(s, d, &sse_reg, &sum_reg);
        src += src_stride * 2;
        dst += dst_stride * 2;
        sec += sec_stride * 2;
      }
    } else if (y_offset == 4) {
      for (int i = 0; i < height; i += 2) {
        const __m512i s_row01 =
            _mm512_avg_epu8(loadu2_256_zmm(src + src_stride, src),
                            loadu2_256_zmm(src + src_stride + 1, src + 1));
        const __m512i s_row12 = _mm512_avg_epu8(
            loadu2_256_zmm(src + src_stride * 2, src + src_stride),
            loadu2_256_zmm(src + src_stride * 2 + 1, src + src_stride + 1));
        const __m512i s =
            _mm512_avg_epu8(_mm512_avg_epu8(s_row01, s_row12),
                            loadu2_256_zmm(sec + sec_stride, sec));
        const __m512i d = loadu2_256_zmm(dst + dst_stride, dst);
        variance_kernel_zmm(s, d, &sse_reg, &sum_reg);
        src += src_stride * 2;
        dst += dst_stride * 2;
        sec += sec_stride * 2;
      }
    } else {
      const __m512i filter = bilinear_filter_zmm(y_offset);
      for (int i = 0; i < height; i += 2) {
        const __m512i s_row01 =
            _mm512_avg_epu8(loadu2_256_zmm(src + src_stride, src),
                            loadu2_256_zmm(src + src_stride + 1, src + 1));
        const __m512i s_row12 = _mm512_avg_epu8(
            loadu2_256_zmm(src + src_stride * 2, src + src_stride),
            loadu2_256_zmm(src + src_stride * 2 + 1, src + src_stride + 1));
        __m512i exp_lo, exp_hi;
        FILTER_PAIR_ZMM(s_row01, s_row12, filter, exp_lo, exp_hi);
        const __m512i s =
            _mm512_avg_epu8(_mm512_packus_epi16(exp_lo, exp_hi),
                            loadu2_256_zmm(sec + sec_stride, sec));
        const __m512i d = loadu2_256_zmm(dst + dst_stride, dst);
        variance_kernel_zmm(s, d, &sse_reg, &sum_reg);
        src += src_stride * 2;
        dst += dst_stride * 2;
        sec += sec_stride * 2;
      }
    }
  } else {
    const __m512i xfilter = bilinear_filter_zmm(x_offset);
    if (y_offset == 0) {
      for (int i = 0; i < height; i += 2) {
        __m512i exp_lo, exp_hi;
        FILTER_PAIR_ZMM(loadu2_256_zmm(src + src_stride, src),
                        loadu2_256_zmm(src + src_stride + 1, src + 1), xfilter,
                        exp_lo, exp_hi);
        const __m512i s =
            _mm512_avg_epu8(_mm512_packus_epi16(exp_lo, exp_hi),
                            loadu2_256_zmm(sec + sec_stride, sec));
        const __m512i d = loadu2_256_zmm(dst + dst_stride, dst);
        variance_kernel_zmm(s, d, &sse_reg, &sum_reg);
        src += src_stride * 2;
        dst += dst_stride * 2;
        sec += sec_stride * 2;
      }
    } else if (y_offset == 4) {
      __m512i exp_lo, exp_hi;
      FILTER_PAIR_ZMM(loadu2_256_zmm(src + src_stride, src),
                      loadu2_256_zmm(src + src_stride + 1, src + 1), xfilter,
                      exp_lo, exp_hi);
      __m512i p01 = _mm512_packus_epi16(exp_lo, exp_hi);
      src += src_stride * 2;
      for (int i = 0; i < height - 2; i += 2) {
        FILTER_PAIR_ZMM(loadu2_256_zmm(src + src_stride, src),
                        loadu2_256_zmm(src + src_stride + 1, src + 1), xfilter,
                        exp_lo, exp_hi);
        const __m512i p23 = _mm512_packus_epi16(exp_lo, exp_hi);
        const __m512i p12 = _mm512_inserti64x4(
            _mm512_castsi256_si512(_mm512_extracti64x4_epi64(p01, 1)),
            _mm512_castsi512_si256(p23), 1);
        const __m512i s = _mm512_avg_epu8(
            _mm512_avg_epu8(p01, p12), loadu2_256_zmm(sec + sec_stride, sec));
        const __m512i d = loadu2_256_zmm(dst + dst_stride, dst);
        variance_kernel_zmm(s, d, &sse_reg, &sum_reg);
        p01 = p23;
        src += src_stride * 2;
        dst += dst_stride * 2;
        sec += sec_stride * 2;
      }
      {
        const __m256i s0 = _mm256_loadu_si256((const __m256i *)src);
        const __m256i s1 = _mm256_loadu_si256((const __m256i *)(src + 1));
        const __m256i xf256 = _mm512_castsi512_si256(xfilter);
        const __m256i pw8_256 = _mm256_set1_epi16(8);
        const __m256i l = _mm256_srai_epi16(
            _mm256_add_epi16(
                _mm256_maddubs_epi16(_mm256_unpacklo_epi8(s0, s1), xf256),
                pw8_256),
            4);
        const __m256i h = _mm256_srai_epi16(
            _mm256_add_epi16(
                _mm256_maddubs_epi16(_mm256_unpackhi_epi8(s0, s1), xf256),
                pw8_256),
            4);
        const __m256i p_last = _mm256_packus_epi16(l, h);
        const __m512i p12 = _mm512_inserti64x4(
            _mm512_castsi256_si512(_mm512_extracti64x4_epi64(p01, 1)), p_last,
            1);
        const __m512i s = _mm512_avg_epu8(
            _mm512_avg_epu8(p01, p12), loadu2_256_zmm(sec + sec_stride, sec));
        const __m512i d = loadu2_256_zmm(dst + dst_stride, dst);
        variance_kernel_zmm(s, d, &sse_reg, &sum_reg);
      }
    } else {
      const __m512i yfilter = bilinear_filter_zmm(y_offset);
      __m512i exp_lo, exp_hi;
      FILTER_PAIR_ZMM(loadu2_256_zmm(src + src_stride, src),
                      loadu2_256_zmm(src + src_stride + 1, src + 1), xfilter,
                      exp_lo, exp_hi);
      __m512i p01 = _mm512_packus_epi16(exp_lo, exp_hi);
      src += src_stride * 2;
      for (int i = 0; i < height - 2; i += 2) {
        FILTER_PAIR_ZMM(loadu2_256_zmm(src + src_stride, src),
                        loadu2_256_zmm(src + src_stride + 1, src + 1), xfilter,
                        exp_lo, exp_hi);
        const __m512i p23 = _mm512_packus_epi16(exp_lo, exp_hi);
        const __m512i p12 = _mm512_inserti64x4(
            _mm512_castsi256_si512(_mm512_extracti64x4_epi64(p01, 1)),
            _mm512_castsi512_si256(p23), 1);
        __m512i v_lo, v_hi;
        FILTER_PAIR_ZMM(p01, p12, yfilter, v_lo, v_hi);
        const __m512i s =
            _mm512_avg_epu8(_mm512_packus_epi16(v_lo, v_hi),
                            loadu2_256_zmm(sec + sec_stride, sec));
        const __m512i d = loadu2_256_zmm(dst + dst_stride, dst);
        variance_kernel_zmm(s, d, &sse_reg, &sum_reg);
        p01 = p23;
        src += src_stride * 2;
        dst += dst_stride * 2;
        sec += sec_stride * 2;
      }
      {
        const __m256i s0 = _mm256_loadu_si256((const __m256i *)src);
        const __m256i s1 = _mm256_loadu_si256((const __m256i *)(src + 1));
        const __m256i xf256 = _mm512_castsi512_si256(xfilter);
        const __m256i pw8_256 = _mm256_set1_epi16(8);
        const __m256i l = _mm256_srai_epi16(
            _mm256_add_epi16(
                _mm256_maddubs_epi16(_mm256_unpacklo_epi8(s0, s1), xf256),
                pw8_256),
            4);
        const __m256i h = _mm256_srai_epi16(
            _mm256_add_epi16(
                _mm256_maddubs_epi16(_mm256_unpackhi_epi8(s0, s1), xf256),
                pw8_256),
            4);
        const __m256i p_last = _mm256_packus_epi16(l, h);
        const __m512i p12 = _mm512_inserti64x4(
            _mm512_castsi256_si512(_mm512_extracti64x4_epi64(p01, 1)), p_last,
            1);
        __m512i v_lo, v_hi;
        FILTER_PAIR_ZMM(p01, p12, yfilter, v_lo, v_hi);
        const __m512i s =
            _mm512_avg_epu8(_mm512_packus_epi16(v_lo, v_hi),
                            loadu2_256_zmm(sec + sec_stride, sec));
        const __m512i d = loadu2_256_zmm(dst + dst_stride, dst);
        variance_kernel_zmm(s, d, &sse_reg, &sum_reg);
      }
    }
  }
  return variance_final_from_16bit_sum_zmm(sse_reg, sum_reg, sse);
}

#undef ACCUM_DIFF_ZMM
#undef FILTER_PAIR_ZMM

#define MAKE_SUB_PIXEL_VAR_32XH_AVX512(h, log2h)                              \
  unsigned int aom_sub_pixel_variance32x##h##_avx512(                         \
      const uint8_t *src, int src_stride, int x_offset, int y_offset,         \
      const uint8_t *dst, int dst_stride, unsigned int *sse) {                \
    const int sum = sub_pixel_variance32xh_imp_avx512(                        \
        src, src_stride, x_offset, y_offset, dst, dst_stride, h, sse);        \
    return *sse - (unsigned int)(((int64_t)sum * sum) >> (5 + (log2h)));      \
  }                                                                           \
  unsigned int aom_sub_pixel_avg_variance32x##h##_avx512(                     \
      const uint8_t *src, int src_stride, int x_offset, int y_offset,         \
      const uint8_t *dst, int dst_stride, unsigned int *sse,                  \
      const uint8_t *sec_ptr) {                                               \
    const int sum = sub_pixel_avg_variance32xh_imp_avx512(                    \
        src, src_stride, x_offset, y_offset, dst, dst_stride, sec_ptr, 32, h, \
        sse);                                                                 \
    return *sse - (unsigned int)(((int64_t)sum * sum) >> (5 + (log2h)));      \
  }

MAKE_SUB_PIXEL_VAR_32XH_AVX512(64, 6)
MAKE_SUB_PIXEL_VAR_32XH_AVX512(32, 5)
MAKE_SUB_PIXEL_VAR_32XH_AVX512(16, 4)
#if !CONFIG_REALTIME_ONLY
MAKE_SUB_PIXEL_VAR_32XH_AVX512(8, 3)
#endif

#define MAKE_SUB_PIXEL_VAR_64_128_AVX512(w, h, hf, wlog2, hlog2)              \
  unsigned int aom_sub_pixel_variance##w##x##h##_avx512(                      \
      const uint8_t *src, int src_stride, int x_offset, int y_offset,         \
      const uint8_t *dst, int dst_stride, unsigned int *sse_ptr) {            \
    unsigned int sse = 0;                                                     \
    int se = 0;                                                               \
    for (int i = 0; i < (w) / 64; ++i) {                                      \
      const uint8_t *src_ptr = src;                                           \
      const uint8_t *dst_ptr = dst;                                           \
      for (int j = 0; j < (h) / (hf); ++j) {                                  \
        unsigned int sse2;                                                    \
        const int se2 = sub_pixel_variance64xh_imp_avx512(                    \
            src_ptr, src_stride, x_offset, y_offset, dst_ptr, dst_stride, hf, \
            &sse2);                                                           \
        dst_ptr += (hf) * dst_stride;                                         \
        src_ptr += (hf) * src_stride;                                         \
        se += se2;                                                            \
        sse += sse2;                                                          \
      }                                                                       \
      src += 64;                                                              \
      dst += 64;                                                              \
    }                                                                         \
    *sse_ptr = sse;                                                           \
    return sse - (unsigned int)(((int64_t)se * se) >> ((wlog2) + (hlog2)));   \
  }                                                                           \
  unsigned int aom_sub_pixel_avg_variance##w##x##h##_avx512(                  \
      const uint8_t *src, int src_stride, int x_offset, int y_offset,         \
      const uint8_t *dst, int dst_stride, unsigned int *sse_ptr,              \
      const uint8_t *sec) {                                                   \
    unsigned int sse = 0;                                                     \
    int se = 0;                                                               \
    for (int i = 0; i < (w) / 64; ++i) {                                      \
      const uint8_t *src_ptr = src;                                           \
      const uint8_t *dst_ptr = dst;                                           \
      const uint8_t *sec_ptr = sec;                                           \
      for (int j = 0; j < (h) / (hf); ++j) {                                  \
        unsigned int sse2;                                                    \
        const int se2 = sub_pixel_avg_variance64xh_imp_avx512(                \
            src_ptr, src_stride, x_offset, y_offset, dst_ptr, dst_stride,     \
            sec_ptr, w, hf, &sse2);                                           \
        dst_ptr += (hf) * dst_stride;                                         \
        src_ptr += (hf) * src_stride;                                         \
        sec_ptr += (hf) * (w);                                                \
        se += se2;                                                            \
        sse += sse2;                                                          \
      }                                                                       \
      src += 64;                                                              \
      dst += 64;                                                              \
      sec += 64;                                                              \
    }                                                                         \
    *sse_ptr = sse;                                                           \
    return sse - (unsigned int)(((int64_t)se * se) >> ((wlog2) + (hlog2)));   \
  }

MAKE_SUB_PIXEL_VAR_64_128_AVX512(128, 128, 64, 7, 7)
MAKE_SUB_PIXEL_VAR_64_128_AVX512(128, 64, 64, 7, 6)
MAKE_SUB_PIXEL_VAR_64_128_AVX512(64, 128, 64, 6, 7)
MAKE_SUB_PIXEL_VAR_64_128_AVX512(64, 64, 64, 6, 6)
MAKE_SUB_PIXEL_VAR_64_128_AVX512(64, 32, 32, 6, 5)
#if !CONFIG_REALTIME_ONLY
MAKE_SUB_PIXEL_VAR_64_128_AVX512(64, 16, 16, 6, 4)
#endif

// -----------------------------------------------------------------------------
// Compound Average & Masked Prediction (512-bit ZMM)
// -----------------------------------------------------------------------------

static AOM_FORCE_INLINE void comp_mask_pred_line_zmm(const __m512i s0,
                                                     const __m512i s1,
                                                     const __m512i a,
                                                     uint8_t *comp_pred) {
  const __m512i alpha_max = _mm512_set1_epi8(AOM_BLEND_A64_MAX_ALPHA);
  const int16_t round_bits = 15 - AOM_BLEND_A64_ROUND_BITS;
  const __m512i round_offset = _mm512_set1_epi16(1 << round_bits);
  const __m512i ma = _mm512_sub_epi8(alpha_max, a);

  const __m512i ssAL = _mm512_unpacklo_epi8(s0, s1);
  const __m512i aaAL = _mm512_unpacklo_epi8(a, ma);
  const __m512i ssAH = _mm512_unpackhi_epi8(s0, s1);
  const __m512i aaAH = _mm512_unpackhi_epi8(a, ma);

  const __m512i blendAL = _mm512_maddubs_epi16(ssAL, aaAL);
  const __m512i blendAH = _mm512_maddubs_epi16(ssAH, aaAH);
  const __m512i roundAL = _mm512_mulhrs_epi16(blendAL, round_offset);
  const __m512i roundAH = _mm512_mulhrs_epi16(blendAH, round_offset);

  const __m512i roundA = _mm512_packus_epi16(roundAL, roundAH);
  _mm512_storeu_si512((__m512i *)comp_pred, roundA);
}

void aom_comp_avg_pred_avx512(uint8_t *comp_pred, const uint8_t *pred,
                              int width, int height, const uint8_t *ref,
                              int ref_stride) {
  int row = 0;
  if (width == 8) {
    do {
      const __m256i pred_0123 = _mm256_loadu_si256((const __m256i *)pred);
      const __m128i ref_0 = _mm_loadl_epi64((const __m128i *)ref);
      const __m128i ref_1 =
          _mm_loadl_epi64((const __m128i *)(ref + ref_stride));
      const __m128i ref_2 =
          _mm_loadl_epi64((const __m128i *)(ref + 2 * ref_stride));
      const __m128i ref_3 =
          _mm_loadl_epi64((const __m128i *)(ref + 3 * ref_stride));
      const __m128i ref_01 = _mm_unpacklo_epi64(ref_0, ref_1);
      const __m128i ref_23 = _mm_unpacklo_epi64(ref_2, ref_3);
      const __m256i ref_0123 =
          _mm256_inserti128_si256(_mm256_castsi128_si256(ref_01), ref_23, 1);
      _mm256_storeu_si256((__m256i *)comp_pred,
                          _mm256_avg_epu8(pred_0123, ref_0123));
      row += 4;
      pred += 32;
      comp_pred += 32;
      ref += 4 * ref_stride;
    } while (row < height);
  } else if (width == 16) {
    do {
      const __m512i pred_0123 = _mm512_loadu_si512((const __m512i *)pred);
      const __m512i ref_0123 = loadu4_128_zmm(ref, ref_stride);
      _mm512_storeu_si512((__m512i *)comp_pred,
                          _mm512_avg_epu8(pred_0123, ref_0123));
      row += 4;
      pred += 64;
      comp_pred += 64;
      ref += 4 * ref_stride;
    } while (row < height);
  } else if (width == 32) {
    do {
      const __m512i pred_01 = _mm512_loadu_si512((const __m512i *)pred);
      const __m512i ref_01 = loadu2_256_zmm(ref + ref_stride, ref);
      _mm512_storeu_si512((__m512i *)comp_pred,
                          _mm512_avg_epu8(pred_01, ref_01));
      row += 2;
      pred += 64;
      comp_pred += 64;
      ref += 2 * ref_stride;
    } while (row < height);
  } else if (width % 64 == 0) {
    do {
      for (int x = 0; x < width; x += 64) {
        const __m512i p = _mm512_loadu_si512((const __m512i *)(pred + x));
        const __m512i r = _mm512_loadu_si512((const __m512i *)(ref + x));
        _mm512_storeu_si512((__m512i *)(comp_pred + x), _mm512_avg_epu8(p, r));
      }
      row++;
      pred += width;
      comp_pred += width;
      ref += ref_stride;
    } while (row < height);
  } else {
    aom_comp_avg_pred_c(comp_pred, pred, width, height, ref, ref_stride);
  }
}

void aom_comp_mask_pred_avx512(uint8_t *comp_pred, const uint8_t *pred,
                               int width, int height, const uint8_t *ref,
                               int ref_stride, const uint8_t *mask,
                               int mask_stride, int invert_mask) {
  int i = 0;
  const uint8_t *src0 = invert_mask ? pred : ref;
  const uint8_t *src1 = invert_mask ? ref : pred;
  const int stride0 = invert_mask ? width : ref_stride;
  const int stride1 = invert_mask ? ref_stride : width;
  if (width == 8) {
    aom_comp_mask_pred_avx2(comp_pred, pred, width, height, ref, ref_stride,
                            mask, mask_stride, invert_mask);
  } else if (width == 16) {
    do {
      const __m512i s0 = loadu4_128_zmm(src0, stride0);
      const __m512i s1 = loadu4_128_zmm(src1, stride1);
      const __m512i a = loadu4_128_zmm(mask, mask_stride);
      comp_mask_pred_line_zmm(s0, s1, a, comp_pred);
      src0 += stride0 * 4;
      src1 += stride1 * 4;
      mask += mask_stride * 4;
      comp_pred += 64;
      i += 4;
    } while (i < height);
  } else if (width == 32) {
    do {
      const __m512i s0 = loadu2_256_zmm(src0 + stride0, src0);
      const __m512i s1 = loadu2_256_zmm(src1 + stride1, src1);
      const __m512i a = loadu2_256_zmm(mask + mask_stride, mask);
      comp_mask_pred_line_zmm(s0, s1, a, comp_pred);
      src0 += stride0 * 2;
      src1 += stride1 * 2;
      mask += mask_stride * 2;
      comp_pred += 64;
      i += 2;
    } while (i < height);
  } else {
    do {
      for (int x = 0; x < width; x += 64) {
        const __m512i s0 = _mm512_loadu_si512((const __m512i *)(src0 + x));
        const __m512i s1 = _mm512_loadu_si512((const __m512i *)(src1 + x));
        const __m512i a = _mm512_loadu_si512((const __m512i *)(mask + x));
        comp_mask_pred_line_zmm(s0, s1, a, comp_pred);
        comp_pred += 64;
      }
      src0 += stride0;
      src1 += stride1;
      mask += mask_stride;
      i++;
    } while (i < height);
  }
}

#if CONFIG_AV1_HIGHBITDEPTH
static AOM_FORCE_INLINE __m512i highbd_comp_mask_pred_line_zmm(
    const __m512i s0, const __m512i s1, const __m512i a) {
  const __m512i alpha_max = _mm512_set1_epi16(1 << AOM_BLEND_A64_ROUND_BITS);
  const __m512i round_const =
      _mm512_set1_epi32((1 << AOM_BLEND_A64_ROUND_BITS) >> 1);
  const __m512i a_inv = _mm512_sub_epi16(alpha_max, a);

  const __m512i s_lo = _mm512_unpacklo_epi16(s0, s1);
  const __m512i a_lo = _mm512_unpacklo_epi16(a, a_inv);
  const __m512i pred_l = _mm512_srai_epi32(
      _mm512_add_epi32(_mm512_madd_epi16(s_lo, a_lo), round_const),
      AOM_BLEND_A64_ROUND_BITS);

  const __m512i s_hi = _mm512_unpackhi_epi16(s0, s1);
  const __m512i a_hi = _mm512_unpackhi_epi16(a, a_inv);
  const __m512i pred_h = _mm512_srai_epi32(
      _mm512_add_epi32(_mm512_madd_epi16(s_hi, a_hi), round_const),
      AOM_BLEND_A64_ROUND_BITS);

  return _mm512_packs_epi32(pred_l, pred_h);
}

void aom_highbd_comp_mask_pred_avx512(uint8_t *comp_pred8, const uint8_t *pred8,
                                      int width, int height,
                                      const uint8_t *ref8, int ref_stride,
                                      const uint8_t *mask, int mask_stride,
                                      int invert_mask) {
  if (width < 16) {
    aom_highbd_comp_mask_pred_avx2(comp_pred8, pred8, width, height, ref8,
                                   ref_stride, mask, mask_stride, invert_mask);
    return;
  }
  uint16_t *pred = CONVERT_TO_SHORTPTR(pred8);
  uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  uint16_t *comp_pred = CONVERT_TO_SHORTPTR(comp_pred8);
  const uint16_t *src0 = invert_mask ? pred : ref;
  const uint16_t *src1 = invert_mask ? ref : pred;
  const int stride0 = invert_mask ? width : ref_stride;
  const int stride1 = invert_mask ? ref_stride : width;

  if (width == 16) {
    for (int i = 0; i < height; i += 2) {
      const __m512i s0 = loadu2_256_zmm((const uint8_t *)(src0 + stride0),
                                        (const uint8_t *)src0);
      const __m512i s1 = loadu2_256_zmm((const uint8_t *)(src1 + stride1),
                                        (const uint8_t *)src1);
      const __m256i m8 = yy_loadu2_128(mask + mask_stride, mask);
      const __m512i m16 = _mm512_cvtepu8_epi16(m8);
      const __m512i comp = highbd_comp_mask_pred_line_zmm(s0, s1, m16);
      _mm512_storeu_si512((__m512i *)comp_pred, comp);
      src0 += stride0 * 2;
      src1 += stride1 * 2;
      mask += mask_stride * 2;
      comp_pred += 32;
    }
  } else {
    for (int i = 0; i < height; i++) {
      for (int x = 0; x < width; x += 32) {
        const __m512i s0 = _mm512_loadu_si512((const __m512i *)(src0 + x));
        const __m512i s1 = _mm512_loadu_si512((const __m512i *)(src1 + x));
        const __m256i m8 = _mm256_loadu_si256((const __m256i *)(mask + x));
        const __m512i m16 = _mm512_cvtepu8_epi16(m8);
        const __m512i comp = highbd_comp_mask_pred_line_zmm(s0, s1, m16);
        _mm512_storeu_si512((__m512i *)comp_pred, comp);
        comp_pred += 32;
      }
      src0 += stride0;
      src1 += stride1;
      mask += mask_stride;
    }
  }
}
#endif  // CONFIG_AV1_HIGHBITDEPTH

// -----------------------------------------------------------------------------
// Block SSE & Sum (aom_get_blk_sse_sum_avx512)
// -----------------------------------------------------------------------------

void aom_get_blk_sse_sum_avx512(const int16_t *data, int stride, int bw, int bh,
                                int *x_sum, int64_t *x2_sum) {
  if (bw < 32) {
    aom_get_blk_sse_sum_avx2(data, stride, bw, bh, x_sum, x2_sum);
    return;
  }
  const __m512i one = _mm512_set1_epi16(1);
  __m512i sum_32 = _mm512_setzero_si512();
  __m512i sse_64 = _mm512_setzero_si512();

  for (int i = 0; i < bh; ++i) {
    for (int j = 0; j < bw; j += 32) {
      const __m512i d = _mm512_loadu_si512((const __m512i *)(data + j));
      sum_32 = _mm512_add_epi32(sum_32, _mm512_madd_epi16(d, one));
      const __m512i sq = _mm512_madd_epi16(d, d);
      const __m512i sq_lo = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(sq));
      const __m512i sq_hi =
          _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(sq, 1));
      sse_64 = _mm512_add_epi64(sse_64, _mm512_add_epi64(sq_lo, sq_hi));
    }
    data += stride;
  }
  *x_sum = hsum_epi32_zmm(sum_32);
  *x2_sum = hsum_epi64_zmm(sse_64);
}

// -----------------------------------------------------------------------------
// Single-pass ZMM Register-Pipelined Variance Stat
// -----------------------------------------------------------------------------

static AOM_FORCE_INLINE void load_row32_horiz_filter_zmm(const uint8_t *src_row,
                                                         int x, int bw,
                                                         __m512i *u16_curr_out,
                                                         __m512i *u16_H_out) {
  const __m256i v_curr = _mm256_loadu_si256((const __m256i *)(src_row + x));
  __m256i v_left, v_right;
  if (x == 0) {
    const __m256i shifted = _mm256_alignr_epi8(
        v_curr, _mm256_permute2x128_si256(v_curr, v_curr, 0x08), 15);
    v_left = _mm256_insert_epi8(shifted, (int8_t)src_row[0], 0);
  } else {
    v_left = _mm256_loadu_si256((const __m256i *)(src_row + x - 1));
  }
  if (x + 32 < bw) {
    v_right = _mm256_loadu_si256((const __m256i *)(src_row + x + 1));
  } else {
    const __m256i shifted = _mm256_alignr_epi8(
        _mm256_permute2x128_si256(v_curr, v_curr, 0x81), v_curr, 1);
    v_right = _mm256_insert_epi8(shifted, (int8_t)src_row[bw - 1], 31);
  }
  const __m512i u16_left = _mm512_cvtepu8_epi16(v_left);
  const __m512i u16_curr = _mm512_cvtepu8_epi16(v_curr);
  const __m512i u16_right = _mm512_cvtepu8_epi16(v_right);
  *u16_curr_out = u16_curr;
  *u16_H_out = _mm512_add_epi16(_mm512_add_epi16(u16_left, u16_right),
                                _mm512_slli_epi16(u16_curr, 1));
}

int64_t aom_calc_variance_stat_avx512(const uint8_t *src, int stride, int bw,
                                      int bh) {
  if (bw < 32) {
    return aom_calc_variance_stat_avx2(src, stride, bw, bh);
  }
  __m512i acc_var_512 = _mm512_setzero_si512();

  for (int x = 0; x < bw; x += 32) {
    __m512i u16_curr, H_curr;
    load_row32_horiz_filter_zmm(src, x, bw, &u16_curr, &H_curr);
    __m512i H_top = H_curr;

    for (int y = 0; y < bh - 1; ++y) {
      __m512i u16_next, H_bot;
      load_row32_horiz_filter_zmm(src + (y + 1) * stride, x, bw, &u16_next,
                                  &H_bot);
      const __m512i u16_V = _mm512_add_epi16(_mm512_add_epi16(H_top, H_bot),
                                             _mm512_slli_epi16(H_curr, 1));
      const __m512i u16_sum = _mm512_srli_epi16(u16_V, 4);
      const __m512i diff = _mm512_sub_epi16(u16_curr, u16_sum);
      acc_var_512 =
          _mm512_add_epi32(acc_var_512, _mm512_madd_epi16(diff, diff));
      H_top = H_curr;
      H_curr = H_bot;
      u16_curr = u16_next;
    }
    // Last row (y = bh - 1): H_bot == H_curr
    {
      const __m512i u16_V = _mm512_add_epi16(_mm512_add_epi16(H_top, H_curr),
                                             _mm512_slli_epi16(H_curr, 1));
      const __m512i u16_sum = _mm512_srli_epi16(u16_V, 4);
      const __m512i diff = _mm512_sub_epi16(u16_curr, u16_sum);
      acc_var_512 =
          _mm512_add_epi32(acc_var_512, _mm512_madd_epi16(diff, diff));
    }
  }
  return ((int64_t)hsum_epi32_zmm(acc_var_512)) << 4;
}

#if CONFIG_AV1_HIGHBITDEPTH
static AOM_FORCE_INLINE void highbd_load_row32_horiz_filter_zmm(
    const uint16_t *src_row, int x, int bw, __m512i *v_curr_out,
    __m512i *u16_H_out) {
  const __m512i v_curr = _mm512_loadu_si512((const __m512i *)(src_row + x));
  __m512i v_left, v_right;
  if (x == 0) {
    uint16_t tmp[32];
    tmp[0] = src_row[0];
    memcpy(&tmp[1], src_row, 31 * sizeof(uint16_t));
    v_left = _mm512_loadu_si512((const __m512i *)tmp);
  } else {
    v_left = _mm512_loadu_si512((const __m512i *)(src_row + x - 1));
  }
  if (x + 32 < bw) {
    v_right = _mm512_loadu_si512((const __m512i *)(src_row + x + 1));
  } else {
    uint16_t tmp[32];
    memcpy(tmp, src_row + x + 1, 31 * sizeof(uint16_t));
    tmp[31] = src_row[bw - 1];
    v_right = _mm512_loadu_si512((const __m512i *)tmp);
  }
  *v_curr_out = v_curr;
  *u16_H_out = _mm512_add_epi16(_mm512_add_epi16(v_left, v_right),
                                _mm512_slli_epi16(v_curr, 1));
}

int64_t aom_highbd_calc_variance_stat_avx512(const uint16_t *src, int stride,
                                             int bw, int bh) {
  if (bw < 32) {
    return aom_highbd_calc_variance_stat_avx2(src, stride, bw, bh);
  }
  __m512i acc_var_64 = _mm512_setzero_si512();

  for (int x = 0; x < bw; x += 32) {
    __m512i v_curr, H_curr;
    highbd_load_row32_horiz_filter_zmm(src, x, bw, &v_curr, &H_curr);
    __m512i H_top = H_curr;

    for (int y = 0; y < bh - 1; ++y) {
      __m512i v_next, H_bot;
      highbd_load_row32_horiz_filter_zmm(src + (y + 1) * stride, x, bw, &v_next,
                                         &H_bot);
      const __m512i u16_V = _mm512_add_epi16(_mm512_add_epi16(H_top, H_bot),
                                             _mm512_slli_epi16(H_curr, 1));
      const __m512i u16_sum = _mm512_srli_epi16(u16_V, 4);
      const __m512i diff = _mm512_sub_epi16(v_curr, u16_sum);
      const __m512i diff_sq = _mm512_madd_epi16(diff, diff);
      acc_var_64 = _mm512_add_epi64(
          acc_var_64,
          _mm512_add_epi64(
              _mm512_cvtepi32_epi64(_mm512_castsi512_si256(diff_sq)),
              _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(diff_sq, 1))));
      H_top = H_curr;
      H_curr = H_bot;
      v_curr = v_next;
    }
    {
      const __m512i u16_V = _mm512_add_epi16(_mm512_add_epi16(H_top, H_curr),
                                             _mm512_slli_epi16(H_curr, 1));
      const __m512i u16_sum = _mm512_srli_epi16(u16_V, 4);
      const __m512i diff = _mm512_sub_epi16(v_curr, u16_sum);
      const __m512i diff_sq = _mm512_madd_epi16(diff, diff);
      acc_var_64 = _mm512_add_epi64(
          acc_var_64,
          _mm512_add_epi64(
              _mm512_cvtepi32_epi64(_mm512_castsi512_si256(diff_sq)),
              _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(diff_sq, 1))));
    }
  }
  return hsum_epi64_zmm(acc_var_64) << 4;
}
#endif  // CONFIG_AV1_HIGHBITDEPTH

// -----------------------------------------------------------------------------
// OBMC SAD & OBMC Variance (512-bit ZMM)
// -----------------------------------------------------------------------------

#if !CONFIG_REALTIME_ONLY
static AOM_FORCE_INLINE unsigned int obmc_sad_w16n_avx512(
    const uint8_t *pre, const int pre_stride, const int32_t *wsrc,
    const int32_t *mask, const int width, const int height) {
  const int pre_step = pre_stride - width;
  int n = 0;
  __m512i v_sad_d = _mm512_setzero_si512();
  const __m512i v_bias_d = _mm512_set1_epi32((1 << 12) >> 1);

  do {
    const __m128i v_p0_b = _mm_loadu_si128((const __m128i *)(pre + n));
    const __m512i v_m0_d = _mm512_loadu_si512((const __m512i *)(mask + n));
    const __m512i v_w0_d = _mm512_loadu_si512((const __m512i *)(wsrc + n));
    const __m512i v_p0_d = _mm512_cvtepu8_epi32(v_p0_b);
    const __m512i v_pm0_d = _mm512_madd_epi16(v_p0_d, v_m0_d);
    const __m512i v_diff0_d = _mm512_sub_epi32(v_w0_d, v_pm0_d);
    const __m512i v_absdiff0_d = _mm512_abs_epi32(v_diff0_d);
    const __m512i v_rad0_d =
        _mm512_srli_epi32(_mm512_add_epi32(v_absdiff0_d, v_bias_d), 12);
    v_sad_d = _mm512_add_epi32(v_sad_d, v_rad0_d);
    n += 16;
    if ((n & (width - 1)) == 0) pre += pre_step;
  } while (n < width * height);

  return (unsigned int)hsum_epi32_zmm(v_sad_d);
}

#define OBMCSADWXH_AVX512(w, h)                                        \
  unsigned int aom_obmc_sad##w##x##h##_avx512(                         \
      const uint8_t *pre, int pre_stride, const int32_t *wsrc,         \
      const int32_t *msk) {                                            \
    if ((w) >= 16) {                                                   \
      return obmc_sad_w16n_avx512(pre, pre_stride, wsrc, msk, w, h);   \
    } else {                                                           \
      return aom_obmc_sad##w##x##h##_avx2(pre, pre_stride, wsrc, msk); \
    }                                                                  \
  }

OBMCSADWXH_AVX512(128, 128)
OBMCSADWXH_AVX512(128, 64)
OBMCSADWXH_AVX512(64, 128)
OBMCSADWXH_AVX512(64, 64)
OBMCSADWXH_AVX512(64, 32)
OBMCSADWXH_AVX512(32, 64)
OBMCSADWXH_AVX512(32, 32)
OBMCSADWXH_AVX512(32, 16)
OBMCSADWXH_AVX512(16, 32)
OBMCSADWXH_AVX512(16, 16)
OBMCSADWXH_AVX512(16, 8)
OBMCSADWXH_AVX512(8, 16)
OBMCSADWXH_AVX512(8, 8)
OBMCSADWXH_AVX512(8, 4)
OBMCSADWXH_AVX512(4, 8)
OBMCSADWXH_AVX512(4, 4)
OBMCSADWXH_AVX512(4, 16)
OBMCSADWXH_AVX512(16, 4)
OBMCSADWXH_AVX512(8, 32)
OBMCSADWXH_AVX512(32, 8)
OBMCSADWXH_AVX512(16, 64)
OBMCSADWXH_AVX512(64, 16)

static AOM_FORCE_INLINE void obmc_variance_w16n_avx512(
    const uint8_t *pre, const int pre_stride, const int32_t *wsrc,
    const int32_t *mask, unsigned int *const sse, int *const sum, const int w,
    const int h) {
  int n = 0;
  const __m512i v_bias_d = _mm512_set1_epi32((1 << 12) >> 1);
  __m512i v_sum_d = _mm512_setzero_si512();
  __m512i v_sse_d = _mm512_setzero_si512();

  for (int row = 0; row < h; row++) {
    for (int col = 0; col < w; col += 16) {
      const __m128i v_p_b = _mm_loadu_si128((const __m128i *)(pre + col));
      const __m512i v_m_d = _mm512_loadu_si512((const __m512i *)(mask + n));
      const __m512i v_w_d = _mm512_loadu_si512((const __m512i *)(wsrc + n));
      const __m512i v_p_d = _mm512_cvtepu8_epi32(v_p_b);
      const __m512i v_pm_d = _mm512_madd_epi16(v_p_d, v_m_d);
      const __m512i v_diff_d = _mm512_sub_epi32(v_w_d, v_pm_d);
      const __m512i v_sign_d = _mm512_srai_epi32(v_diff_d, 31);
      const __m512i v_tmp_d =
          _mm512_add_epi32(_mm512_add_epi32(v_diff_d, v_bias_d), v_sign_d);
      const __m512i v_rdiff_d = _mm512_srai_epi32(v_tmp_d, 12);

      v_sum_d = _mm512_add_epi32(v_sum_d, v_rdiff_d);
      v_sse_d =
          _mm512_add_epi32(v_sse_d, _mm512_mullo_epi32(v_rdiff_d, v_rdiff_d));
      n += 16;
    }
    pre += pre_stride;
  }
  *sum = variance_final_from_32bit_sum_zmm(v_sse_d, v_sum_d, sse);
}

#define OBMCVARWXH_AVX512(W, H)                                                \
  unsigned int aom_obmc_variance##W##x##H##_avx512(                            \
      const uint8_t *pre, int pre_stride, const int32_t *wsrc,                 \
      const int32_t *mask, unsigned int *sse) {                                \
    if ((W) >= 16) {                                                           \
      int sum;                                                                 \
      obmc_variance_w16n_avx512(pre, pre_stride, wsrc, mask, sse, &sum, W, H); \
      return *sse - (unsigned int)(((int64_t)sum * sum) / ((W) * (H)));        \
    } else {                                                                   \
      return aom_obmc_variance##W##x##H##_avx2(pre, pre_stride, wsrc, mask,    \
                                               sse);                           \
    }                                                                          \
  }

OBMCVARWXH_AVX512(128, 128)
OBMCVARWXH_AVX512(128, 64)
OBMCVARWXH_AVX512(64, 128)
OBMCVARWXH_AVX512(64, 64)
OBMCVARWXH_AVX512(64, 32)
OBMCVARWXH_AVX512(32, 64)
OBMCVARWXH_AVX512(32, 32)
OBMCVARWXH_AVX512(32, 16)
OBMCVARWXH_AVX512(16, 32)
OBMCVARWXH_AVX512(16, 16)
OBMCVARWXH_AVX512(16, 8)
OBMCVARWXH_AVX512(8, 16)
OBMCVARWXH_AVX512(8, 8)
OBMCVARWXH_AVX512(8, 4)
OBMCVARWXH_AVX512(4, 8)
OBMCVARWXH_AVX512(4, 4)
OBMCVARWXH_AVX512(4, 16)
OBMCVARWXH_AVX512(16, 4)
OBMCVARWXH_AVX512(8, 32)
OBMCVARWXH_AVX512(32, 8)
OBMCVARWXH_AVX512(16, 64)
OBMCVARWXH_AVX512(64, 16)
#endif  // !CONFIG_REALTIME_ONLY

// -----------------------------------------------------------------------------
// Masked SAD (512-bit ZMM)
// -----------------------------------------------------------------------------

static AOM_FORCE_INLINE __m512i masked_sad_step_zmm(
    __m512i res, const __m512i src, const __m512i a, const __m512i b,
    const __m512i m, const __m512i mask_max, const __m512i round_scale) {
  const __m512i m_inv = _mm512_sub_epi8(mask_max, m);
  const __m512i data_l = _mm512_unpacklo_epi8(a, b);
  const __m512i mask_l = _mm512_unpacklo_epi8(m, m_inv);
  const __m512i pred_l =
      _mm512_mulhrs_epi16(_mm512_maddubs_epi16(data_l, mask_l), round_scale);

  const __m512i data_r = _mm512_unpackhi_epi8(a, b);
  const __m512i mask_r = _mm512_unpackhi_epi8(m, m_inv);
  const __m512i pred_r =
      _mm512_mulhrs_epi16(_mm512_maddubs_epi16(data_r, mask_r), round_scale);

  const __m512i pred = _mm512_packus_epi16(pred_l, pred_r);
  return _mm512_add_epi32(res, _mm512_sad_epu8(pred, src));
}

static AOM_FORCE_INLINE unsigned int masked_sad64xh_avx512(
    const uint8_t *src_ptr, int src_stride, const uint8_t *a_ptr, int a_stride,
    const uint8_t *b_ptr, int b_stride, const uint8_t *m_ptr, int m_stride,
    int width, int height) {
  __m512i res = _mm512_setzero_si512();
  const __m512i mask_max = _mm512_set1_epi8(1 << AOM_BLEND_A64_ROUND_BITS);
  const __m512i round_scale =
      _mm512_set1_epi16(1 << (15 - AOM_BLEND_A64_ROUND_BITS));
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x += 64) {
      res = masked_sad_step_zmm(
          res, _mm512_loadu_si512((const __m512i *)&src_ptr[x]),
          _mm512_loadu_si512((const __m512i *)&a_ptr[x]),
          _mm512_loadu_si512((const __m512i *)&b_ptr[x]),
          _mm512_loadu_si512((const __m512i *)&m_ptr[x]), mask_max,
          round_scale);
    }
    src_ptr += src_stride;
    a_ptr += a_stride;
    b_ptr += b_stride;
    m_ptr += m_stride;
  }
  const __m256i r256 = _mm256_add_epi32(_mm512_castsi512_si256(res),
                                        _mm512_extracti64x4_epi64(res, 1));
  const __m128i r128 = _mm_add_epi32(_mm256_castsi256_si128(r256),
                                     _mm256_extracti128_si256(r256, 1));
  return (unsigned int)_mm_cvtsi128_si32(
      _mm_add_epi32(r128, _mm_srli_si128(r128, 8)));
}

static AOM_FORCE_INLINE unsigned int masked_sad32xh_avx512(
    const uint8_t *src_ptr, int src_stride, const uint8_t *a_ptr, int a_stride,
    const uint8_t *b_ptr, int b_stride, const uint8_t *m_ptr, int m_stride,
    int height) {
  __m512i res = _mm512_setzero_si512();
  const __m512i mask_max = _mm512_set1_epi8(1 << AOM_BLEND_A64_ROUND_BITS);
  const __m512i round_scale =
      _mm512_set1_epi16(1 << (15 - AOM_BLEND_A64_ROUND_BITS));
  for (int y = 0; y < height; y += 2) {
    res = masked_sad_step_zmm(
        res, loadu2_256_zmm(src_ptr + src_stride, src_ptr),
        loadu2_256_zmm(a_ptr + a_stride, a_ptr),
        loadu2_256_zmm(b_ptr + b_stride, b_ptr),
        loadu2_256_zmm(m_ptr + m_stride, m_ptr), mask_max, round_scale);
    src_ptr += src_stride * 2;
    a_ptr += a_stride * 2;
    b_ptr += b_stride * 2;
    m_ptr += m_stride * 2;
  }
  const __m256i r256 = _mm256_add_epi32(_mm512_castsi512_si256(res),
                                        _mm512_extracti64x4_epi64(res, 1));
  const __m128i r128 = _mm_add_epi32(_mm256_castsi256_si128(r256),
                                     _mm256_extracti128_si256(r256, 1));
  return (unsigned int)_mm_cvtsi128_si32(
      _mm_add_epi32(r128, _mm_srli_si128(r128, 8)));
}

static AOM_FORCE_INLINE unsigned int masked_sad16xh_avx512(
    const uint8_t *src_ptr, int src_stride, const uint8_t *a_ptr, int a_stride,
    const uint8_t *b_ptr, int b_stride, const uint8_t *m_ptr, int m_stride,
    int height) {
  __m512i res = _mm512_setzero_si512();
  const __m512i mask_max = _mm512_set1_epi8(1 << AOM_BLEND_A64_ROUND_BITS);
  const __m512i round_scale =
      _mm512_set1_epi16(1 << (15 - AOM_BLEND_A64_ROUND_BITS));
  for (int y = 0; y < height; y += 4) {
    res = masked_sad_step_zmm(
        res, loadu4_128_zmm(src_ptr, src_stride),
        loadu4_128_zmm(a_ptr, a_stride), loadu4_128_zmm(b_ptr, b_stride),
        loadu4_128_zmm(m_ptr, m_stride), mask_max, round_scale);
    src_ptr += src_stride * 4;
    a_ptr += a_stride * 4;
    b_ptr += b_stride * 4;
    m_ptr += m_stride * 4;
  }
  const __m256i r256 = _mm256_add_epi32(_mm512_castsi512_si256(res),
                                        _mm512_extracti64x4_epi64(res, 1));
  const __m128i r128 = _mm_add_epi32(_mm256_castsi256_si128(r256),
                                     _mm256_extracti128_si256(r256, 1));
  return (unsigned int)_mm_cvtsi128_si32(
      _mm_add_epi32(r128, _mm_srli_si128(r128, 8)));
}

static AOM_FORCE_INLINE unsigned int aom_masked_sad_avx512(
    const uint8_t *src, int src_stride, const uint8_t *ref, int ref_stride,
    const uint8_t *second_pred, const uint8_t *msk, int msk_stride,
    int invert_mask, int m, int n) {
  const uint8_t *a_ptr = invert_mask ? second_pred : ref;
  const uint8_t *b_ptr = invert_mask ? ref : second_pred;
  const int a_stride = invert_mask ? m : ref_stride;
  const int b_stride = invert_mask ? ref_stride : m;
  switch (m) {
    case 4:
      return aom_masked_sad4xh_ssse3(src, src_stride, a_ptr, a_stride, b_ptr,
                                     b_stride, msk, msk_stride, n);
    case 8:
      return aom_masked_sad8xh_ssse3(src, src_stride, a_ptr, a_stride, b_ptr,
                                     b_stride, msk, msk_stride, n);
    case 16:
      return masked_sad16xh_avx512(src, src_stride, a_ptr, a_stride, b_ptr,
                                   b_stride, msk, msk_stride, n);
    case 32:
      return masked_sad32xh_avx512(src, src_stride, a_ptr, a_stride, b_ptr,
                                   b_stride, msk, msk_stride, n);
    default:
      return masked_sad64xh_avx512(src, src_stride, a_ptr, a_stride, b_ptr,
                                   b_stride, msk, msk_stride, m, n);
  }
}

#define MASKSADMXN_AVX512(m, n)                                                \
  unsigned int aom_masked_sad##m##x##n##_avx512(                               \
      const uint8_t *src, int src_stride, const uint8_t *ref, int ref_stride,  \
      const uint8_t *second_pred, const uint8_t *msk, int msk_stride,          \
      int invert_mask) {                                                       \
    return aom_masked_sad_avx512(src, src_stride, ref, ref_stride,             \
                                 second_pred, msk, msk_stride, invert_mask, m, \
                                 n);                                           \
  }

MASKSADMXN_AVX512(4, 4)
MASKSADMXN_AVX512(4, 8)
MASKSADMXN_AVX512(8, 4)
MASKSADMXN_AVX512(8, 8)
MASKSADMXN_AVX512(8, 16)
MASKSADMXN_AVX512(16, 8)
MASKSADMXN_AVX512(16, 16)
MASKSADMXN_AVX512(16, 32)
MASKSADMXN_AVX512(32, 16)
MASKSADMXN_AVX512(32, 32)
MASKSADMXN_AVX512(32, 64)
MASKSADMXN_AVX512(64, 32)
MASKSADMXN_AVX512(64, 64)
MASKSADMXN_AVX512(64, 128)
MASKSADMXN_AVX512(128, 64)
MASKSADMXN_AVX512(128, 128)

#if !CONFIG_REALTIME_ONLY
MASKSADMXN_AVX512(4, 16)
MASKSADMXN_AVX512(16, 4)
MASKSADMXN_AVX512(8, 32)
MASKSADMXN_AVX512(32, 8)
MASKSADMXN_AVX512(16, 64)
MASKSADMXN_AVX512(64, 16)
#endif
