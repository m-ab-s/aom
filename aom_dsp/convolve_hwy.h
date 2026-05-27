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

#ifndef AOM_AOM_DSP_CONVOLVE_HWY_H_
#define AOM_AOM_DSP_CONVOLVE_HWY_H_

#include <cassert>

#include "aom_dsp/arm/aom_filter.h"
#include "third_party/highway/hwy/highway.h"

HWY_BEFORE_NAMESPACE();

namespace {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

template <typename D>
HWY_ATTR HWY_INLINE hn::VFromD<D> LoadUnaligned4x4(D tag16, const uint8_t *buf,
                                                   ptrdiff_t stride) {
  hn::CappedTag<uint32_t, 4> tag32;
  uint32_t r0, r1, r2, r3;
  memcpy(&r0, buf, 4);
  memcpy(&r1, buf + stride, 4);
  memcpy(&r2, buf + 2 * stride, 4);
  memcpy(&r3, buf + 3 * stride, 4);
  auto v32 = hn::Zero(tag32);
  v32 = hn::InsertLane(v32, 0, r0);
  v32 = hn::InsertLane(v32, 1, r1);
  v32 = hn::InsertLane(v32, 2, r2);
  v32 = hn::InsertLane(v32, 3, r3);
  hn::Rebind<uint8_t, D> tag8;
  return hn::PromoteTo(tag16, hn::BitCast(tag8, v32));
}

template <typename D>
HWY_ATTR HWY_INLINE void StoreUnaligned4x4(D tag16, uint8_t *buf,
                                           ptrdiff_t stride,
                                           hn::VFromD<D> &vec) {
  (void)tag16;
  hn::Rebind<uint8_t, D> tag8;
  constexpr hn::Half<D> half_tag16;
  auto vec_demoted = hn::ReorderDemote2To(tag8, hn::LowerHalf(half_tag16, vec),
                                          hn::UpperHalf(half_tag16, vec));
  constexpr hn::Half<decltype(tag8)> half_tag;
  constexpr hn::Half<decltype(half_tag)> quarter_tag;
  auto vec1_2 = hn::LowerHalf(half_tag, vec_demoted);
  auto vec2_2 = hn::UpperHalf(half_tag, vec_demoted);
  auto vec1_4 = hn::LowerHalf(quarter_tag, vec1_2);
  auto vec2_4 = hn::UpperHalf(quarter_tag, vec1_2);
  auto vec3_4 = hn::LowerHalf(quarter_tag, vec2_2);
  auto vec4_4 = hn::UpperHalf(quarter_tag, vec2_2);
  hn::StoreU(vec1_4, quarter_tag, buf);
  hn::StoreU(vec2_4, quarter_tag, buf + stride);
  hn::StoreU(vec3_4, quarter_tag, buf + 2 * stride);
  hn::StoreU(vec4_4, quarter_tag, buf + 3 * stride);
}

template <typename D>
HWY_ATTR HWY_INLINE hn::VFromD<D> LoadUnaligned2x8(D tag16, const uint8_t *buf,
                                                   ptrdiff_t stride) {
  hn::Rebind<uint8_t, D> tag8;
  constexpr hn::Half<decltype(tag8)> half_tag8;
  auto first_half = hn::LoadU(half_tag8, buf);
  auto second_half = hn::LoadU(half_tag8, buf + stride);
  return hn::PromoteTo(tag16, hn::Combine(tag8, first_half, second_half));
}

template <typename D>
HWY_ATTR HWY_INLINE void StoreUnaligned2x8(D tag, uint8_t *buf,
                                           ptrdiff_t stride,
                                           hn::VFromD<D> &vec) {
  (void)tag;
  hn::Rebind<uint8_t, D> tag8;
  constexpr hn::Half<D> half_tag16;
  auto vec_demoted = hn::ReorderDemote2To(tag8, hn::LowerHalf(half_tag16, vec),
                                          hn::UpperHalf(half_tag16, vec));
  constexpr hn::Half<decltype(tag8)> half_tag8;
  auto vec1_2 = hn::UpperHalf(half_tag8, vec_demoted);
  auto vec2_2 = hn::LowerHalf(half_tag8, vec_demoted);
  hn::StoreU(vec1_2, half_tag8, buf);
  hn::StoreU(vec2_2, half_tag8, buf + stride);
}

template <typename D>
HWY_ATTR HWY_INLINE hn::VFromD<D> LoadUnaligned4x8(D scalable_tag,
                                                   const uint8_t *buf,
                                                   ptrdiff_t stride) {
  hn::Rebind<uint8_t, D> tag8;
  constexpr hn::Half<decltype(tag8)> half_tag8;
  constexpr hn::Half<decltype(half_tag8)> quarter_tag8;
  auto first_quarter = hn::LoadU(quarter_tag8, buf);
  auto second_quarter = hn::LoadU(quarter_tag8, buf + stride);
  auto third_quarter = hn::LoadU(quarter_tag8, buf + 2 * stride);
  auto fourth_quarter = hn::LoadU(quarter_tag8, buf + 3 * stride);
  return hn::PromoteTo(
      scalable_tag,
      hn::Combine(tag8, hn::Combine(half_tag8, first_quarter, second_quarter),
                  hn::Combine(half_tag8, third_quarter, fourth_quarter)));
}

template <typename D>
HWY_ATTR HWY_INLINE void StoreUnaligned4x8(D tag, uint8_t *buf,
                                           ptrdiff_t stride,
                                           hn::VFromD<D> &vec) {
  (void)tag;
  hn::Rebind<uint8_t, D> tag8;
  constexpr hn::Half<D> half_tag16;
  auto vec_demoted = hn::ReorderDemote2To(tag8, hn::LowerHalf(half_tag16, vec),
                                          hn::UpperHalf(half_tag16, vec));
  constexpr hn::Half<decltype(tag8)> half_tag8;
  constexpr hn::Half<decltype(half_tag8)> quarter_tag8;
  auto vec1_2 = hn::UpperHalf(half_tag8, vec_demoted);
  auto vec2_2 = hn::LowerHalf(half_tag8, vec_demoted);
  auto vec1_4 = hn::UpperHalf(quarter_tag8, vec1_2);
  auto vec2_4 = hn::LowerHalf(quarter_tag8, vec1_2);
  auto vec3_4 = hn::UpperHalf(quarter_tag8, vec2_2);
  auto vec4_4 = hn::LowerHalf(quarter_tag8, vec2_2);
  hn::StoreU(vec1_4, quarter_tag8, buf);
  hn::StoreU(vec2_4, quarter_tag8, buf + stride);
  hn::StoreU(vec3_4, quarter_tag8, buf + 2 * stride);
  hn::StoreU(vec4_4, quarter_tag8, buf + 3 * stride);
}

HWY_ATTR inline void ConvolveHoriz2Tap(const uint8_t *src, ptrdiff_t src_stride,
                                       uint8_t *dst, ptrdiff_t dst_stride,
                                       const int16_t *filter_x, int w, int h) {
  const bool can_use_optimized_path =
      (w <= 32) && (filter_x[3] % 2 == 0) && (filter_x[4] % 2 == 0);

  if (can_use_optimized_path) {
    hn::CappedTag<uint8_t, 16> tag8_16;
    hn::CappedTag<int8_t, 16> tag_i8;
    hn::CappedTag<int16_t, 8> tag16_8;
    hn::CappedTag<uint8_t, 8> tag8_8;
    hn::CappedTag<uint8_t, 4> tag8_4;
    const auto bias_val = hn::Set(tag16_8, 1 << (FILTER_BITS - 2));

    const auto shuffle_mask = hn::Dup128VecFromValues(
        tag8_16, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8);

    const int8_t c3 = static_cast<int8_t>(filter_x[3] / 2);
    const int8_t c4 = static_cast<int8_t>(filter_x[4] / 2);

    const auto coeff34 = hn::Dup128VecFromValues(
        tag_i8, c3, c4, c3, c4, c3, c4, c3, c4, c3, c4, c3, c4, c3, c4, c3, c4);

    if (w == 4) {
      while (h >= 2) {
        auto r0_d0 = hn::LoadU(tag8_16, src + 0);
        auto r1_d0 = hn::LoadU(tag8_16, src + src_stride + 0);

        auto r0_sum = hn::SatWidenMulPairwiseAdd(
            tag16_8, hn::TableLookupBytes(r0_d0, shuffle_mask), coeff34);
        auto r1_sum = hn::SatWidenMulPairwiseAdd(
            tag16_8, hn::TableLookupBytes(r1_d0, shuffle_mask), coeff34);

        hn::StoreU(
            hn::LowerHalf(tag8_4,
                          hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                                   hn::Add(r0_sum, bias_val)))),
            tag8_4, dst);
        hn::StoreU(
            hn::LowerHalf(tag8_4,
                          hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                                   hn::Add(r1_sum, bias_val)))),
            tag8_4, dst + dst_stride);

        src += 2 * src_stride;
        dst += 2 * dst_stride;
        h -= 2;
      }
    } else if (w == 8) {
      while (h >= 2) {
        auto r0_d0 = hn::LoadU(tag8_16, src + 0);
        auto r1_d0 = hn::LoadU(tag8_16, src + src_stride + 0);

        auto r0_sum = hn::SatWidenMulPairwiseAdd(
            tag16_8, hn::TableLookupBytes(r0_d0, shuffle_mask), coeff34);
        auto r1_sum = hn::SatWidenMulPairwiseAdd(
            tag16_8, hn::TableLookupBytes(r1_d0, shuffle_mask), coeff34);

        hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                            hn::Add(r0_sum, bias_val))),
                   tag8_8, dst);
        hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                            hn::Add(r1_sum, bias_val))),
                   tag8_8, dst + dst_stride);

        src += 2 * src_stride;
        dst += 2 * dst_stride;
        h -= 2;
      }
    } else if (w == 16) {
      while (h >= 2) {
        auto r0_j0_d0 = hn::LoadU(tag8_16, src + 0);
        auto r0_j8_d0 = hn::LoadU(tag8_16, src + 8);

        auto r1_j0_d0 = hn::LoadU(tag8_16, src + src_stride + 0);
        auto r1_j8_d0 = hn::LoadU(tag8_16, src + src_stride + 8);

        auto r0_j0_sum = hn::SatWidenMulPairwiseAdd(
            tag16_8, hn::TableLookupBytes(r0_j0_d0, shuffle_mask), coeff34);
        auto r0_j8_sum = hn::SatWidenMulPairwiseAdd(
            tag16_8, hn::TableLookupBytes(r0_j8_d0, shuffle_mask), coeff34);

        auto r1_j0_sum = hn::SatWidenMulPairwiseAdd(
            tag16_8, hn::TableLookupBytes(r1_j0_d0, shuffle_mask), coeff34);
        auto r1_j8_sum = hn::SatWidenMulPairwiseAdd(
            tag16_8, hn::TableLookupBytes(r1_j8_d0, shuffle_mask), coeff34);

        hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                            hn::Add(r0_j0_sum, bias_val))),
                   tag8_8, dst + 0);
        hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                            hn::Add(r0_j8_sum, bias_val))),
                   tag8_8, dst + 8);
        hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                            hn::Add(r1_j0_sum, bias_val))),
                   tag8_8, dst + dst_stride + 0);
        hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                            hn::Add(r1_j8_sum, bias_val))),
                   tag8_8, dst + dst_stride + 8);

        src += 2 * src_stride;
        dst += 2 * dst_stride;
        h -= 2;
      }
    } else if (w == 32) {
      while (h >= 2) {
        {
          auto r0_j0_d0 = hn::LoadU(tag8_16, src + 0);
          auto r0_j8_d0 = hn::LoadU(tag8_16, src + 8);

          auto r1_j0_d0 = hn::LoadU(tag8_16, src + src_stride + 0);
          auto r1_j8_d0 = hn::LoadU(tag8_16, src + src_stride + 8);

          auto r0_j0_sum = hn::SatWidenMulPairwiseAdd(
              tag16_8, hn::TableLookupBytes(r0_j0_d0, shuffle_mask), coeff34);
          auto r0_j8_sum = hn::SatWidenMulPairwiseAdd(
              tag16_8, hn::TableLookupBytes(r0_j8_d0, shuffle_mask), coeff34);

          auto r1_j0_sum = hn::SatWidenMulPairwiseAdd(
              tag16_8, hn::TableLookupBytes(r1_j0_d0, shuffle_mask), coeff34);
          auto r1_j8_sum = hn::SatWidenMulPairwiseAdd(
              tag16_8, hn::TableLookupBytes(r1_j8_d0, shuffle_mask), coeff34);

          hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                              hn::Add(r0_j0_sum, bias_val))),
                     tag8_8, dst + 0);
          hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                              hn::Add(r0_j8_sum, bias_val))),
                     tag8_8, dst + 8);
          hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                              hn::Add(r1_j0_sum, bias_val))),
                     tag8_8, dst + dst_stride + 0);
          hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                              hn::Add(r1_j8_sum, bias_val))),
                     tag8_8, dst + dst_stride + 8);
        }
        {
          auto r0_j16_d0 = hn::LoadU(tag8_16, src + 16);
          auto r0_j24_d0 = hn::LoadU(tag8_16, src + 24);

          auto r1_j16_d0 = hn::LoadU(tag8_16, src + src_stride + 16);
          auto r1_j24_d0 = hn::LoadU(tag8_16, src + src_stride + 24);

          auto r0_j16_sum = hn::SatWidenMulPairwiseAdd(
              tag16_8, hn::TableLookupBytes(r0_j16_d0, shuffle_mask), coeff34);
          auto r0_j24_sum = hn::SatWidenMulPairwiseAdd(
              tag16_8, hn::TableLookupBytes(r0_j24_d0, shuffle_mask), coeff34);

          auto r1_j16_sum = hn::SatWidenMulPairwiseAdd(
              tag16_8, hn::TableLookupBytes(r1_j16_d0, shuffle_mask), coeff34);
          auto r1_j24_sum = hn::SatWidenMulPairwiseAdd(
              tag16_8, hn::TableLookupBytes(r1_j24_d0, shuffle_mask), coeff34);

          hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                              hn::Add(r0_j16_sum, bias_val))),
                     tag8_8, dst + 16);
          hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                              hn::Add(r0_j24_sum, bias_val))),
                     tag8_8, dst + 24);
          hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                              hn::Add(r1_j16_sum, bias_val))),
                     tag8_8, dst + dst_stride + 16);
          hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                              hn::Add(r1_j24_sum, bias_val))),
                     tag8_8, dst + dst_stride + 24);
        }
        src += 2 * src_stride;
        dst += 2 * dst_stride;
        h -= 2;
      }
    }
  } else {
    hn::ScalableTag<int16_t> mul_tag;
    hn::Rebind<uint8_t, decltype(mul_tag)> pixel_tag;
    auto filter_0 = hn::Set(mul_tag, filter_x[3]);
    auto filter_1 = hn::Set(mul_tag, filter_x[4]);
    auto vw = hn::Lanes(mul_tag);
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; j += vw) {
        auto src0 = hn::PromoteTo(mul_tag, hn::LoadU(pixel_tag, &src[j]));
        auto src1 = hn::PromoteTo(mul_tag, hn::LoadU(pixel_tag, &src[j + 1]));
        auto mulv = hn::RoundingShiftRight<FILTER_BITS>(src0 * filter_0 +
                                                        src1 * filter_1);
        auto mulv_demoted = hn::DemoteTo(pixel_tag, mulv);
        if (j + static_cast<int>(vw) > w) {
          hn::StoreN(mulv_demoted, pixel_tag, &dst[j], w - j);
        } else {
          hn::StoreU(mulv_demoted, pixel_tag, &dst[j]);
        }
      }
      src += src_stride;
      dst += dst_stride;
    }
  }
}

template <typename D, typename DFilter>
HWY_ATTR HWY_INLINE hn::VFromD<D> Convolve4_8(
    D tag16, DFilter tag_filter, hn::VFromD<D> &s0, hn::VFromD<D> &s1,
    hn::VFromD<D> &s2, hn::VFromD<D> &s3, hn::VFromD<DFilter> &filter) {
  (void)tag_filter;
  auto mul0 = hn::Mul(s0, hn::Set(tag16, hn::ExtractLane(filter, 0)));
  auto mul1 = hn::Mul(s1, hn::Set(tag16, hn::ExtractLane(filter, 1)));
  auto mul2 = hn::Mul(s2, hn::Set(tag16, hn::ExtractLane(filter, 2)));
  auto mul3 = hn::Mul(s3, hn::Set(tag16, hn::ExtractLane(filter, 3)));

  auto res = mul0 + mul1 + mul2 + mul3;
  // Shift (FILTER_BITS - 1) because filter values were halved.
  return hn::RoundingShiftRight<FILTER_BITS - 1>(res);
}

HWY_ATTR inline void ConvolveHoriz4Tap(const uint8_t *src, ptrdiff_t src_stride,
                                       uint8_t *dst, ptrdiff_t dst_stride,
                                       const int16_t *filter_x, int w, int h) {
  const bool can_use_optimized_path =
      (w <= 32) && (filter_x[2] % 2 == 0) && (filter_x[3] % 2 == 0) &&
      (filter_x[4] % 2 == 0) && (filter_x[5] % 2 == 0);

  if (can_use_optimized_path) {
    hn::CappedTag<uint8_t, 16> tag8_16;
    hn::CappedTag<int8_t, 16> tag_i8;
    hn::CappedTag<int16_t, 8> tag16_8;
    hn::CappedTag<uint8_t, 8> tag8_8;
    hn::CappedTag<uint8_t, 4> tag8_4;
    const auto bias_val = hn::Set(tag16_8, 1 << (FILTER_BITS - 2));

    const auto shuffle_mask = hn::Dup128VecFromValues(
        tag8_16, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8);

    const int8_t c2 = static_cast<int8_t>(filter_x[2] / 2);
    const int8_t c3 = static_cast<int8_t>(filter_x[3] / 2);
    const int8_t c4 = static_cast<int8_t>(filter_x[4] / 2);
    const int8_t c5 = static_cast<int8_t>(filter_x[5] / 2);

    const auto coeff23 = hn::Dup128VecFromValues(
        tag_i8, c2, c3, c2, c3, c2, c3, c2, c3, c2, c3, c2, c3, c2, c3, c2, c3);

    const auto coeff45 = hn::Dup128VecFromValues(
        tag_i8, c4, c5, c4, c5, c4, c5, c4, c5, c4, c5, c4, c5, c4, c5, c4, c5);

    if (w == 4) {
      while (h >= 2) {
        auto r0_d0 = hn::LoadU(tag8_16, src + 0);
        auto r0_d2 = hn::LoadU(tag8_16, src + 2);

        auto r1_d0 = hn::LoadU(tag8_16, src + src_stride + 0);
        auto r1_d2 = hn::LoadU(tag8_16, src + src_stride + 2);

        auto r0_sum = hn::Add(
            hn::SatWidenMulPairwiseAdd(
                tag16_8, hn::TableLookupBytes(r0_d0, shuffle_mask), coeff23),
            hn::SatWidenMulPairwiseAdd(
                tag16_8, hn::TableLookupBytes(r0_d2, shuffle_mask), coeff45));

        auto r1_sum = hn::Add(
            hn::SatWidenMulPairwiseAdd(
                tag16_8, hn::TableLookupBytes(r1_d0, shuffle_mask), coeff23),
            hn::SatWidenMulPairwiseAdd(
                tag16_8, hn::TableLookupBytes(r1_d2, shuffle_mask), coeff45));

        hn::StoreU(
            hn::LowerHalf(tag8_4,
                          hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                                   hn::Add(r0_sum, bias_val)))),
            tag8_4, dst);
        hn::StoreU(
            hn::LowerHalf(tag8_4,
                          hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                                   hn::Add(r1_sum, bias_val)))),
            tag8_4, dst + dst_stride);

        src += 2 * src_stride;
        dst += 2 * dst_stride;
        h -= 2;
      }
    } else if (w == 8) {
      while (h >= 2) {
        auto r0_d0 = hn::LoadU(tag8_16, src + 0);
        auto r0_d2 = hn::LoadU(tag8_16, src + 2);

        auto r1_d0 = hn::LoadU(tag8_16, src + src_stride + 0);
        auto r1_d2 = hn::LoadU(tag8_16, src + src_stride + 2);

        auto r0_sum = hn::Add(
            hn::SatWidenMulPairwiseAdd(
                tag16_8, hn::TableLookupBytes(r0_d0, shuffle_mask), coeff23),
            hn::SatWidenMulPairwiseAdd(
                tag16_8, hn::TableLookupBytes(r0_d2, shuffle_mask), coeff45));

        auto r1_sum = hn::Add(
            hn::SatWidenMulPairwiseAdd(
                tag16_8, hn::TableLookupBytes(r1_d0, shuffle_mask), coeff23),
            hn::SatWidenMulPairwiseAdd(
                tag16_8, hn::TableLookupBytes(r1_d2, shuffle_mask), coeff45));

        hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                            hn::Add(r0_sum, bias_val))),
                   tag8_8, dst);
        hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                            hn::Add(r1_sum, bias_val))),
                   tag8_8, dst + dst_stride);

        src += 2 * src_stride;
        dst += 2 * dst_stride;
        h -= 2;
      }
    } else if (w == 16) {
      while (h >= 2) {
        auto r0_j0_d0 = hn::LoadU(tag8_16, src + 0);
        auto r0_j0_d2 = hn::LoadU(tag8_16, src + 2);

        auto r0_j8_d0 = hn::LoadU(tag8_16, src + 8);
        auto r0_j8_d2 = hn::LoadU(tag8_16, src + 10);

        auto r1_j0_d0 = hn::LoadU(tag8_16, src + src_stride + 0);
        auto r1_j0_d2 = hn::LoadU(tag8_16, src + src_stride + 2);

        auto r1_j8_d0 = hn::LoadU(tag8_16, src + src_stride + 8);
        auto r1_j8_d2 = hn::LoadU(tag8_16, src + src_stride + 10);

        auto r0_j0_sum = hn::Add(
            hn::SatWidenMulPairwiseAdd(
                tag16_8, hn::TableLookupBytes(r0_j0_d0, shuffle_mask), coeff23),
            hn::SatWidenMulPairwiseAdd(
                tag16_8, hn::TableLookupBytes(r0_j0_d2, shuffle_mask),
                coeff45));

        auto r0_j8_sum = hn::Add(
            hn::SatWidenMulPairwiseAdd(
                tag16_8, hn::TableLookupBytes(r0_j8_d0, shuffle_mask), coeff23),
            hn::SatWidenMulPairwiseAdd(
                tag16_8, hn::TableLookupBytes(r0_j8_d2, shuffle_mask),
                coeff45));

        auto r1_j0_sum = hn::Add(
            hn::SatWidenMulPairwiseAdd(
                tag16_8, hn::TableLookupBytes(r1_j0_d0, shuffle_mask), coeff23),
            hn::SatWidenMulPairwiseAdd(
                tag16_8, hn::TableLookupBytes(r1_j0_d2, shuffle_mask),
                coeff45));

        auto r1_j8_sum = hn::Add(
            hn::SatWidenMulPairwiseAdd(
                tag16_8, hn::TableLookupBytes(r1_j8_d0, shuffle_mask), coeff23),
            hn::SatWidenMulPairwiseAdd(
                tag16_8, hn::TableLookupBytes(r1_j8_d2, shuffle_mask),
                coeff45));

        hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                            hn::Add(r0_j0_sum, bias_val))),
                   tag8_8, dst + 0);
        hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                            hn::Add(r0_j8_sum, bias_val))),
                   tag8_8, dst + 8);
        hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                            hn::Add(r1_j0_sum, bias_val))),
                   tag8_8, dst + dst_stride + 0);
        hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                            hn::Add(r1_j8_sum, bias_val))),
                   tag8_8, dst + dst_stride + 8);

        src += 2 * src_stride;
        dst += 2 * dst_stride;
        h -= 2;
      }
    } else if (w == 32) {
      while (h >= 2) {
        {
          auto r0_j0_d0 = hn::LoadU(tag8_16, src + 0);
          auto r0_j0_d2 = hn::LoadU(tag8_16, src + 2);

          auto r0_j8_d0 = hn::LoadU(tag8_16, src + 8);
          auto r0_j8_d2 = hn::LoadU(tag8_16, src + 10);

          auto r1_j0_d0 = hn::LoadU(tag8_16, src + src_stride + 0);
          auto r1_j0_d2 = hn::LoadU(tag8_16, src + src_stride + 2);

          auto r1_j8_d0 = hn::LoadU(tag8_16, src + src_stride + 8);
          auto r1_j8_d2 = hn::LoadU(tag8_16, src + src_stride + 10);

          auto r0_j0_sum =
              hn::Add(hn::SatWidenMulPairwiseAdd(
                          tag16_8, hn::TableLookupBytes(r0_j0_d0, shuffle_mask),
                          coeff23),
                      hn::SatWidenMulPairwiseAdd(
                          tag16_8, hn::TableLookupBytes(r0_j0_d2, shuffle_mask),
                          coeff45));

          auto r0_j8_sum =
              hn::Add(hn::SatWidenMulPairwiseAdd(
                          tag16_8, hn::TableLookupBytes(r0_j8_d0, shuffle_mask),
                          coeff23),
                      hn::SatWidenMulPairwiseAdd(
                          tag16_8, hn::TableLookupBytes(r0_j8_d2, shuffle_mask),
                          coeff45));

          auto r1_j0_sum =
              hn::Add(hn::SatWidenMulPairwiseAdd(
                          tag16_8, hn::TableLookupBytes(r1_j0_d0, shuffle_mask),
                          coeff23),
                      hn::SatWidenMulPairwiseAdd(
                          tag16_8, hn::TableLookupBytes(r1_j0_d2, shuffle_mask),
                          coeff45));

          auto r1_j8_sum =
              hn::Add(hn::SatWidenMulPairwiseAdd(
                          tag16_8, hn::TableLookupBytes(r1_j8_d0, shuffle_mask),
                          coeff23),
                      hn::SatWidenMulPairwiseAdd(
                          tag16_8, hn::TableLookupBytes(r1_j8_d2, shuffle_mask),
                          coeff45));

          hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                              hn::Add(r0_j0_sum, bias_val))),
                     tag8_8, dst + 0);
          hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                              hn::Add(r0_j8_sum, bias_val))),
                     tag8_8, dst + 8);
          hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                              hn::Add(r1_j0_sum, bias_val))),
                     tag8_8, dst + dst_stride + 0);
          hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                              hn::Add(r1_j8_sum, bias_val))),
                     tag8_8, dst + dst_stride + 8);
        }
        {
          auto r0_j16_d0 = hn::LoadU(tag8_16, src + 16);
          auto r0_j16_d2 = hn::LoadU(tag8_16, src + 18);

          auto r0_j24_d0 = hn::LoadU(tag8_16, src + 24);
          auto r0_j24_d2 = hn::LoadU(tag8_16, src + 26);

          auto r1_j16_d0 = hn::LoadU(tag8_16, src + src_stride + 16);
          auto r1_j16_d2 = hn::LoadU(tag8_16, src + src_stride + 18);

          auto r1_j24_d0 = hn::LoadU(tag8_16, src + src_stride + 24);
          auto r1_j24_d2 = hn::LoadU(tag8_16, src + src_stride + 26);

          auto r0_j16_sum = hn::Add(
              hn::SatWidenMulPairwiseAdd(
                  tag16_8, hn::TableLookupBytes(r0_j16_d0, shuffle_mask),
                  coeff23),
              hn::SatWidenMulPairwiseAdd(
                  tag16_8, hn::TableLookupBytes(r0_j16_d2, shuffle_mask),
                  coeff45));

          auto r0_j24_sum = hn::Add(
              hn::SatWidenMulPairwiseAdd(
                  tag16_8, hn::TableLookupBytes(r0_j24_d0, shuffle_mask),
                  coeff23),
              hn::SatWidenMulPairwiseAdd(
                  tag16_8, hn::TableLookupBytes(r0_j24_d2, shuffle_mask),
                  coeff45));

          auto r1_j16_sum = hn::Add(
              hn::SatWidenMulPairwiseAdd(
                  tag16_8, hn::TableLookupBytes(r1_j16_d0, shuffle_mask),
                  coeff23),
              hn::SatWidenMulPairwiseAdd(
                  tag16_8, hn::TableLookupBytes(r1_j16_d2, shuffle_mask),
                  coeff45));

          auto r1_j24_sum = hn::Add(
              hn::SatWidenMulPairwiseAdd(
                  tag16_8, hn::TableLookupBytes(r1_j24_d0, shuffle_mask),
                  coeff23),
              hn::SatWidenMulPairwiseAdd(
                  tag16_8, hn::TableLookupBytes(r1_j24_d2, shuffle_mask),
                  coeff45));

          hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                              hn::Add(r0_j16_sum, bias_val))),
                     tag8_8, dst + 16);
          hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                              hn::Add(r0_j24_sum, bias_val))),
                     tag8_8, dst + 24);
          hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                              hn::Add(r1_j16_sum, bias_val))),
                     tag8_8, dst + dst_stride + 16);
          hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                              hn::Add(r1_j24_sum, bias_val))),
                     tag8_8, dst + dst_stride + 24);
        }
        src += 2 * src_stride;
        dst += 2 * dst_stride;
        h -= 2;
      }
    }
  } else {
    hn::ScalableTag<int16_t> mul_tag;
    hn::Rebind<uint8_t, decltype(mul_tag)> pixel_tag;
    hn::CappedTag<int16_t, 4> filter_tag;
    auto f_vec = hn::LoadU(filter_tag, filter_x + 2);
    f_vec = hn::ShiftRight<1>(f_vec);
    auto vw = hn::Lanes(mul_tag);
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; j += vw) {
        auto src0 = hn::PromoteTo(mul_tag, hn::LoadU(pixel_tag, &src[j]));
        auto src1 = hn::PromoteTo(mul_tag, hn::LoadU(pixel_tag, &src[j + 1]));
        auto src2 = hn::PromoteTo(mul_tag, hn::LoadU(pixel_tag, &src[j + 2]));
        auto src3 = hn::PromoteTo(mul_tag, hn::LoadU(pixel_tag, &src[j + 3]));
        auto result =
            Convolve4_8(mul_tag, filter_tag, src0, src1, src2, src3, f_vec);
        auto result_demoted = hn::DemoteTo(pixel_tag, result);
        if (j + static_cast<int>(vw) > w) {
          hn::StoreN(result_demoted, pixel_tag, &dst[j], w - j);
        } else {
          hn::StoreU(result_demoted, pixel_tag, &dst[j]);
        }
      }
      src += src_stride;
      dst += dst_stride;
    }
  }
}

template <typename D, typename DFilter>
HWY_ATTR HWY_INLINE hn::VFromD<D> Convolve8_8(
    D tag16, DFilter tag_filter, hn::VFromD<D> &s0, hn::VFromD<D> &s1,
    hn::VFromD<D> &s2, hn::VFromD<D> &s3, hn::VFromD<D> &s4, hn::VFromD<D> &s5,
    hn::VFromD<D> &s6, hn::VFromD<D> &s7, hn::VFromD<DFilter> &filter) {
  (void)tag_filter;
  auto filter_0 = hn::ExtractLane(filter, 0);
  auto filter_1 = hn::ExtractLane(filter, 1);
  auto filter_2 = hn::ExtractLane(filter, 2);
  auto filter_3 = hn::ExtractLane(filter, 3);
  auto filter_4 = hn::ExtractLane(filter, 4);
  auto filter_5 = hn::ExtractLane(filter, 5);
  auto filter_6 = hn::ExtractLane(filter, 6);
  auto filter_7 = hn::ExtractLane(filter, 7);
  auto mul0 = hn::Mul(s0, hn::Set(tag16, filter_0));
  auto mul1 = hn::Mul(s1, hn::Set(tag16, filter_1));
  auto mul2 = hn::Mul(s2, hn::Set(tag16, filter_2));
  auto mul3 = hn::Mul(s3, hn::Set(tag16, filter_3));
  auto mul4 = hn::Mul(s4, hn::Set(tag16, filter_4));
  auto mul5 = hn::Mul(s5, hn::Set(tag16, filter_5));
  auto mul6 = hn::Mul(s6, hn::Set(tag16, filter_6));
  auto mul7 = hn::Mul(s7, hn::Set(tag16, filter_7));

  auto res = mul0 + mul1 + mul2 + mul3 + mul4 + mul5 + mul6 + mul7;
  // Shift (FILTER_BITS - 1) because filter values were halved.
  return hn::RoundingShiftRight<FILTER_BITS - 1>(res);
}

HWY_ATTR inline void ConvolveHoriz8Tap(const uint8_t *src, ptrdiff_t src_stride,
                                       uint8_t *dst, ptrdiff_t dst_stride,
                                       const int16_t *filter_x, int w, int h) {
  const bool can_use_optimized_path =
      (w <= 32) && (filter_x[0] % 2 == 0) && (filter_x[1] % 2 == 0) &&
      (filter_x[2] % 2 == 0) && (filter_x[3] % 2 == 0) &&
      (filter_x[4] % 2 == 0) && (filter_x[5] % 2 == 0) &&
      (filter_x[6] % 2 == 0) && (filter_x[7] % 2 == 0);

  if (can_use_optimized_path) {
    hn::CappedTag<uint8_t, 16> tag8_16;
    hn::CappedTag<int8_t, 16> tag_i8;
    hn::CappedTag<int16_t, 8> tag16_8;
    hn::CappedTag<uint8_t, 8> tag8_8;
    hn::CappedTag<uint8_t, 4> tag8_4;
    const auto bias_val = hn::Set(tag16_8, 1 << (FILTER_BITS - 2));

    const auto shuffle_mask = hn::Dup128VecFromValues(
        tag8_16, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8);

    const int8_t c0 = static_cast<int8_t>(filter_x[0] / 2);
    const int8_t c1 = static_cast<int8_t>(filter_x[1] / 2);
    const int8_t c2 = static_cast<int8_t>(filter_x[2] / 2);
    const int8_t c3 = static_cast<int8_t>(filter_x[3] / 2);
    const int8_t c4 = static_cast<int8_t>(filter_x[4] / 2);
    const int8_t c5 = static_cast<int8_t>(filter_x[5] / 2);
    const int8_t c6 = static_cast<int8_t>(filter_x[6] / 2);
    const int8_t c7 = static_cast<int8_t>(filter_x[7] / 2);

    const auto coeff01 = hn::Dup128VecFromValues(
        tag_i8, c0, c1, c0, c1, c0, c1, c0, c1, c0, c1, c0, c1, c0, c1, c0, c1);

    const auto coeff23 = hn::Dup128VecFromValues(
        tag_i8, c2, c3, c2, c3, c2, c3, c2, c3, c2, c3, c2, c3, c2, c3, c2, c3);

    const auto coeff45 = hn::Dup128VecFromValues(
        tag_i8, c4, c5, c4, c5, c4, c5, c4, c5, c4, c5, c4, c5, c4, c5, c4, c5);

    const auto coeff67 = hn::Dup128VecFromValues(
        tag_i8, c6, c7, c6, c7, c6, c7, c6, c7, c6, c7, c6, c7, c6, c7, c6, c7);

    if (w == 4) {
      while (h >= 2) {
        auto r0_d0 = hn::LoadU(tag8_16, src + 0);
        auto r0_d2 = hn::LoadU(tag8_16, src + 2);
        auto r0_d4 = hn::LoadU(tag8_16, src + 4);
        auto r0_d6 = hn::LoadU(tag8_16, src + 6);

        auto r1_d0 = hn::LoadU(tag8_16, src + src_stride + 0);
        auto r1_d2 = hn::LoadU(tag8_16, src + src_stride + 2);
        auto r1_d4 = hn::LoadU(tag8_16, src + src_stride + 4);
        auto r1_d6 = hn::LoadU(tag8_16, src + src_stride + 6);

        auto r0_sum = hn::Add(
            hn::Add(hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r0_d0, shuffle_mask),
                        coeff01),
                    hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r0_d2, shuffle_mask),
                        coeff23)),
            hn::Add(hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r0_d4, shuffle_mask),
                        coeff45),
                    hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r0_d6, shuffle_mask),
                        coeff67)));

        auto r1_sum = hn::Add(
            hn::Add(hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r1_d0, shuffle_mask),
                        coeff01),
                    hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r1_d2, shuffle_mask),
                        coeff23)),
            hn::Add(hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r1_d4, shuffle_mask),
                        coeff45),
                    hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r1_d6, shuffle_mask),
                        coeff67)));

        hn::StoreU(
            hn::LowerHalf(tag8_4,
                          hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                                   hn::Add(r0_sum, bias_val)))),
            tag8_4, dst);
        hn::StoreU(
            hn::LowerHalf(tag8_4,
                          hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                                   hn::Add(r1_sum, bias_val)))),
            tag8_4, dst + dst_stride);

        src += 2 * src_stride;
        dst += 2 * dst_stride;
        h -= 2;
      }
    } else if (w == 8) {
      while (h >= 2) {
        auto r0_d0 = hn::LoadU(tag8_16, src + 0);
        auto r0_d2 = hn::LoadU(tag8_16, src + 2);
        auto r0_d4 = hn::LoadU(tag8_16, src + 4);
        auto r0_d6 = hn::LoadU(tag8_16, src + 6);

        auto r1_d0 = hn::LoadU(tag8_16, src + src_stride + 0);
        auto r1_d2 = hn::LoadU(tag8_16, src + src_stride + 2);
        auto r1_d4 = hn::LoadU(tag8_16, src + src_stride + 4);
        auto r1_d6 = hn::LoadU(tag8_16, src + src_stride + 6);

        auto r0_sum = hn::Add(
            hn::Add(hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r0_d0, shuffle_mask),
                        coeff01),
                    hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r0_d2, shuffle_mask),
                        coeff23)),
            hn::Add(hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r0_d4, shuffle_mask),
                        coeff45),
                    hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r0_d6, shuffle_mask),
                        coeff67)));

        auto r1_sum = hn::Add(
            hn::Add(hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r1_d0, shuffle_mask),
                        coeff01),
                    hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r1_d2, shuffle_mask),
                        coeff23)),
            hn::Add(hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r1_d4, shuffle_mask),
                        coeff45),
                    hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r1_d6, shuffle_mask),
                        coeff67)));

        hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                            hn::Add(r0_sum, bias_val))),
                   tag8_8, dst);
        hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                            hn::Add(r1_sum, bias_val))),
                   tag8_8, dst + dst_stride);

        src += 2 * src_stride;
        dst += 2 * dst_stride;
        h -= 2;
      }
    } else if (w == 16) {
      while (h >= 2) {
        auto r0_j0_d0 = hn::LoadU(tag8_16, src + 0);
        auto r0_j0_d2 = hn::LoadU(tag8_16, src + 2);
        auto r0_j0_d4 = hn::LoadU(tag8_16, src + 4);
        auto r0_j0_d6 = hn::LoadU(tag8_16, src + 6);

        auto r0_j8_d0 = hn::LoadU(tag8_16, src + 8);
        auto r0_j8_d2 = hn::LoadU(tag8_16, src + 10);
        auto r0_j8_d4 = hn::LoadU(tag8_16, src + 12);
        auto r0_j8_d6 = hn::LoadU(tag8_16, src + 14);

        auto r1_j0_d0 = hn::LoadU(tag8_16, src + src_stride + 0);
        auto r1_j0_d2 = hn::LoadU(tag8_16, src + src_stride + 2);
        auto r1_j0_d4 = hn::LoadU(tag8_16, src + src_stride + 4);
        auto r1_j0_d6 = hn::LoadU(tag8_16, src + src_stride + 6);

        auto r1_j8_d0 = hn::LoadU(tag8_16, src + src_stride + 8);
        auto r1_j8_d2 = hn::LoadU(tag8_16, src + src_stride + 10);
        auto r1_j8_d4 = hn::LoadU(tag8_16, src + src_stride + 12);
        auto r1_j8_d6 = hn::LoadU(tag8_16, src + src_stride + 14);

        auto r0_j0_sum = hn::Add(
            hn::Add(hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r0_j0_d0, shuffle_mask),
                        coeff01),
                    hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r0_j0_d2, shuffle_mask),
                        coeff23)),
            hn::Add(hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r0_j0_d4, shuffle_mask),
                        coeff45),
                    hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r0_j0_d6, shuffle_mask),
                        coeff67)));

        auto r0_j8_sum = hn::Add(
            hn::Add(hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r0_j8_d0, shuffle_mask),
                        coeff01),
                    hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r0_j8_d2, shuffle_mask),
                        coeff23)),
            hn::Add(hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r0_j8_d4, shuffle_mask),
                        coeff45),
                    hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r0_j8_d6, shuffle_mask),
                        coeff67)));

        auto r1_j0_sum = hn::Add(
            hn::Add(hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r1_j0_d0, shuffle_mask),
                        coeff01),
                    hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r1_j0_d2, shuffle_mask),
                        coeff23)),
            hn::Add(hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r1_j0_d4, shuffle_mask),
                        coeff45),
                    hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r1_j0_d6, shuffle_mask),
                        coeff67)));

        auto r1_j8_sum = hn::Add(
            hn::Add(hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r1_j8_d0, shuffle_mask),
                        coeff01),
                    hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r1_j8_d2, shuffle_mask),
                        coeff23)),
            hn::Add(hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r1_j8_d4, shuffle_mask),
                        coeff45),
                    hn::SatWidenMulPairwiseAdd(
                        tag16_8, hn::TableLookupBytes(r1_j8_d6, shuffle_mask),
                        coeff67)));

        hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                            hn::Add(r0_j0_sum, bias_val))),
                   tag8_8, dst + 0);
        hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                            hn::Add(r0_j8_sum, bias_val))),
                   tag8_8, dst + 8);
        hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                            hn::Add(r1_j0_sum, bias_val))),
                   tag8_8, dst + dst_stride + 0);
        hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                            hn::Add(r1_j8_sum, bias_val))),
                   tag8_8, dst + dst_stride + 8);

        src += 2 * src_stride;
        dst += 2 * dst_stride;
        h -= 2;
      }
    } else if (w == 32) {
      while (h >= 2) {
        {
          auto r0_j0_d0 = hn::LoadU(tag8_16, src + 0);
          auto r0_j0_d2 = hn::LoadU(tag8_16, src + 2);
          auto r0_j0_d4 = hn::LoadU(tag8_16, src + 4);
          auto r0_j0_d6 = hn::LoadU(tag8_16, src + 6);

          auto r0_j8_d0 = hn::LoadU(tag8_16, src + 8);
          auto r0_j8_d2 = hn::LoadU(tag8_16, src + 10);
          auto r0_j8_d4 = hn::LoadU(tag8_16, src + 12);
          auto r0_j8_d6 = hn::LoadU(tag8_16, src + 14);

          auto r1_j0_d0 = hn::LoadU(tag8_16, src + src_stride + 0);
          auto r1_j0_d2 = hn::LoadU(tag8_16, src + src_stride + 2);
          auto r1_j0_d4 = hn::LoadU(tag8_16, src + src_stride + 4);
          auto r1_j0_d6 = hn::LoadU(tag8_16, src + src_stride + 6);

          auto r1_j8_d0 = hn::LoadU(tag8_16, src + src_stride + 8);
          auto r1_j8_d2 = hn::LoadU(tag8_16, src + src_stride + 10);
          auto r1_j8_d4 = hn::LoadU(tag8_16, src + src_stride + 12);
          auto r1_j8_d6 = hn::LoadU(tag8_16, src + src_stride + 14);

          auto r0_j0_sum = hn::Add(
              hn::Add(hn::SatWidenMulPairwiseAdd(
                          tag16_8, hn::TableLookupBytes(r0_j0_d0, shuffle_mask),
                          coeff01),
                      hn::SatWidenMulPairwiseAdd(
                          tag16_8, hn::TableLookupBytes(r0_j0_d2, shuffle_mask),
                          coeff23)),
              hn::Add(hn::SatWidenMulPairwiseAdd(
                          tag16_8, hn::TableLookupBytes(r0_j0_d4, shuffle_mask),
                          coeff45),
                      hn::SatWidenMulPairwiseAdd(
                          tag16_8, hn::TableLookupBytes(r0_j0_d6, shuffle_mask),
                          coeff67)));

          auto r0_j8_sum = hn::Add(
              hn::Add(hn::SatWidenMulPairwiseAdd(
                          tag16_8, hn::TableLookupBytes(r0_j8_d0, shuffle_mask),
                          coeff01),
                      hn::SatWidenMulPairwiseAdd(
                          tag16_8, hn::TableLookupBytes(r0_j8_d2, shuffle_mask),
                          coeff23)),
              hn::Add(hn::SatWidenMulPairwiseAdd(
                          tag16_8, hn::TableLookupBytes(r0_j8_d4, shuffle_mask),
                          coeff45),
                      hn::SatWidenMulPairwiseAdd(
                          tag16_8, hn::TableLookupBytes(r0_j8_d6, shuffle_mask),
                          coeff67)));

          auto r1_j0_sum = hn::Add(
              hn::Add(hn::SatWidenMulPairwiseAdd(
                          tag16_8, hn::TableLookupBytes(r1_j0_d0, shuffle_mask),
                          coeff01),
                      hn::SatWidenMulPairwiseAdd(
                          tag16_8, hn::TableLookupBytes(r1_j0_d2, shuffle_mask),
                          coeff23)),
              hn::Add(hn::SatWidenMulPairwiseAdd(
                          tag16_8, hn::TableLookupBytes(r1_j0_d4, shuffle_mask),
                          coeff45),
                      hn::SatWidenMulPairwiseAdd(
                          tag16_8, hn::TableLookupBytes(r1_j0_d6, shuffle_mask),
                          coeff67)));

          auto r1_j8_sum = hn::Add(
              hn::Add(hn::SatWidenMulPairwiseAdd(
                          tag16_8, hn::TableLookupBytes(r1_j8_d0, shuffle_mask),
                          coeff01),
                      hn::SatWidenMulPairwiseAdd(
                          tag16_8, hn::TableLookupBytes(r1_j8_d2, shuffle_mask),
                          coeff23)),
              hn::Add(hn::SatWidenMulPairwiseAdd(
                          tag16_8, hn::TableLookupBytes(r1_j8_d4, shuffle_mask),
                          coeff45),
                      hn::SatWidenMulPairwiseAdd(
                          tag16_8, hn::TableLookupBytes(r1_j8_d6, shuffle_mask),
                          coeff67)));

          hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                              hn::Add(r0_j0_sum, bias_val))),
                     tag8_8, dst + 0);
          hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                              hn::Add(r0_j8_sum, bias_val))),
                     tag8_8, dst + 8);
          hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                              hn::Add(r1_j0_sum, bias_val))),
                     tag8_8, dst + dst_stride + 0);
          hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                              hn::Add(r1_j8_sum, bias_val))),
                     tag8_8, dst + dst_stride + 8);
        }
        {
          auto r0_j16_d0 = hn::LoadU(tag8_16, src + 16);
          auto r0_j16_d2 = hn::LoadU(tag8_16, src + 18);
          auto r0_j16_d4 = hn::LoadU(tag8_16, src + 20);
          auto r0_j16_d6 = hn::LoadU(tag8_16, src + 22);

          auto r0_j24_d0 = hn::LoadU(tag8_16, src + 24);
          auto r0_j24_d2 = hn::LoadU(tag8_16, src + 26);
          auto r0_j24_d4 = hn::LoadU(tag8_16, src + 28);
          auto r0_j24_d6 = hn::LoadU(tag8_16, src + 30);

          auto r1_j16_d0 = hn::LoadU(tag8_16, src + src_stride + 16);
          auto r1_j16_d2 = hn::LoadU(tag8_16, src + src_stride + 18);
          auto r1_j16_d4 = hn::LoadU(tag8_16, src + src_stride + 20);
          auto r1_j16_d6 = hn::LoadU(tag8_16, src + src_stride + 22);

          auto r1_j24_d0 = hn::LoadU(tag8_16, src + src_stride + 24);
          auto r1_j24_d2 = hn::LoadU(tag8_16, src + src_stride + 26);
          auto r1_j24_d4 = hn::LoadU(tag8_16, src + src_stride + 28);
          auto r1_j24_d6 = hn::LoadU(tag8_16, src + src_stride + 30);

          auto r0_j16_sum = hn::Add(
              hn::Add(
                  hn::SatWidenMulPairwiseAdd(
                      tag16_8, hn::TableLookupBytes(r0_j16_d0, shuffle_mask),
                      coeff01),
                  hn::SatWidenMulPairwiseAdd(
                      tag16_8, hn::TableLookupBytes(r0_j16_d2, shuffle_mask),
                      coeff23)),
              hn::Add(
                  hn::SatWidenMulPairwiseAdd(
                      tag16_8, hn::TableLookupBytes(r0_j16_d4, shuffle_mask),
                      coeff45),
                  hn::SatWidenMulPairwiseAdd(
                      tag16_8, hn::TableLookupBytes(r0_j16_d6, shuffle_mask),
                      coeff67)));

          auto r0_j24_sum = hn::Add(
              hn::Add(
                  hn::SatWidenMulPairwiseAdd(
                      tag16_8, hn::TableLookupBytes(r0_j24_d0, shuffle_mask),
                      coeff01),
                  hn::SatWidenMulPairwiseAdd(
                      tag16_8, hn::TableLookupBytes(r0_j24_d2, shuffle_mask),
                      coeff23)),
              hn::Add(
                  hn::SatWidenMulPairwiseAdd(
                      tag16_8, hn::TableLookupBytes(r0_j24_d4, shuffle_mask),
                      coeff45),
                  hn::SatWidenMulPairwiseAdd(
                      tag16_8, hn::TableLookupBytes(r0_j24_d6, shuffle_mask),
                      coeff67)));

          auto r1_j16_sum = hn::Add(
              hn::Add(
                  hn::SatWidenMulPairwiseAdd(
                      tag16_8, hn::TableLookupBytes(r1_j16_d0, shuffle_mask),
                      coeff01),
                  hn::SatWidenMulPairwiseAdd(
                      tag16_8, hn::TableLookupBytes(r1_j16_d2, shuffle_mask),
                      coeff23)),
              hn::Add(
                  hn::SatWidenMulPairwiseAdd(
                      tag16_8, hn::TableLookupBytes(r1_j16_d4, shuffle_mask),
                      coeff45),
                  hn::SatWidenMulPairwiseAdd(
                      tag16_8, hn::TableLookupBytes(r1_j16_d6, shuffle_mask),
                      coeff67)));

          auto r1_j24_sum = hn::Add(
              hn::Add(
                  hn::SatWidenMulPairwiseAdd(
                      tag16_8, hn::TableLookupBytes(r1_j24_d0, shuffle_mask),
                      coeff01),
                  hn::SatWidenMulPairwiseAdd(
                      tag16_8, hn::TableLookupBytes(r1_j24_d2, shuffle_mask),
                      coeff23)),
              hn::Add(
                  hn::SatWidenMulPairwiseAdd(
                      tag16_8, hn::TableLookupBytes(r1_j24_d4, shuffle_mask),
                      coeff45),
                  hn::SatWidenMulPairwiseAdd(
                      tag16_8, hn::TableLookupBytes(r1_j24_d6, shuffle_mask),
                      coeff67)));

          hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                              hn::Add(r0_j16_sum, bias_val))),
                     tag8_8, dst + 16);
          hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                              hn::Add(r0_j24_sum, bias_val))),
                     tag8_8, dst + 24);
          hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                              hn::Add(r1_j16_sum, bias_val))),
                     tag8_8, dst + dst_stride + 16);
          hn::StoreU(hn::DemoteTo(tag8_8, hn::ShiftRight<FILTER_BITS - 1>(
                                              hn::Add(r1_j24_sum, bias_val))),
                     tag8_8, dst + dst_stride + 24);
        }
        src += 2 * src_stride;
        dst += 2 * dst_stride;
        h -= 2;
      }
    }
  } else {
    hn::CappedTag<int16_t, 8> filter_tag;
    auto f_vec = hn::LoadU(filter_tag, filter_x);
    f_vec = hn::ShiftRight<1>(f_vec);
    hn::ScalableTag<int16_t> mul_tag;
    hn::Rebind<uint8_t, decltype(mul_tag)> pixel_tag;
    auto vw = hn::Lanes(mul_tag);
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; j += vw) {
        auto s0 = hn::LoadU(pixel_tag, &src[j]);
        auto s1 = hn::LoadU(pixel_tag, &src[j + 1]);
        auto s2 = hn::LoadU(pixel_tag, &src[j + 2]);
        auto s3 = hn::LoadU(pixel_tag, &src[j + 3]);
        auto s4 = hn::LoadU(pixel_tag, &src[j + 4]);
        auto s5 = hn::LoadU(pixel_tag, &src[j + 5]);
        auto s6 = hn::LoadU(pixel_tag, &src[j + 6]);
        auto s7 = hn::LoadU(pixel_tag, &src[j + 7]);
        auto src0 = hn::PromoteTo(mul_tag, s0);
        auto src1 = hn::PromoteTo(mul_tag, s1);
        auto src2 = hn::PromoteTo(mul_tag, s2);
        auto src3 = hn::PromoteTo(mul_tag, s3);
        auto src4 = hn::PromoteTo(mul_tag, s4);
        auto src5 = hn::PromoteTo(mul_tag, s5);
        auto src6 = hn::PromoteTo(mul_tag, s6);
        auto src7 = hn::PromoteTo(mul_tag, s7);
        auto result = Convolve8_8(mul_tag, filter_tag, src0, src1, src2, src3,
                                  src4, src5, src6, src7, f_vec);
        auto result_demoted = hn::DemoteTo(pixel_tag, result);
        if (j + static_cast<int>(vw) > w) {
          hn::StoreN(result_demoted, pixel_tag, &dst[j], w - j);
        } else {
          hn::StoreU(result_demoted, pixel_tag, &dst[j]);
        }
      }
      src += src_stride;
      dst += dst_stride;
    }
  }
}

HWY_ATTR inline void ConvolveVert2Tap(const uint8_t *src, ptrdiff_t src_stride,
                                      uint8_t *dst, ptrdiff_t dst_stride,
                                      const int16_t *filter_y, int w, int h) {
  hn::CappedTag<int16_t, 16> tag16;
  hn::Rebind<uint8_t, decltype(tag16)> pixel_tag;

  auto f0 = hn::Set(tag16, filter_y[3]);
  auto f1 = hn::Set(tag16, filter_y[4]);
  auto round_offset = hn::Set(tag16, 1 << (FILTER_BITS - 1));

  if (w == 4) {
    for (int y = 0; y < h; y += 4) {
      auto s0 = LoadUnaligned4x4(tag16, src + y * src_stride, src_stride);
      auto s1 = LoadUnaligned4x4(tag16, src + (y + 1) * src_stride, src_stride);
      auto res = hn::ShiftRight<FILTER_BITS>(s0 * f0 + s1 * f1 + round_offset);
      StoreUnaligned4x4(tag16, dst + y * dst_stride, dst_stride, res);
    }
  } else if (w == 8) {
    for (int y = 0; y < h; y += 4) {
      auto s0 = LoadUnaligned2x8(tag16, src + y * src_stride, src_stride);
      auto s1 = LoadUnaligned2x8(tag16, src + (y + 1) * src_stride, src_stride);
      auto s2 = LoadUnaligned2x8(tag16, src + (y + 2) * src_stride, src_stride);
      auto s3 = LoadUnaligned2x8(tag16, src + (y + 3) * src_stride, src_stride);
      auto res0 = hn::ShiftRight<FILTER_BITS>(s0 * f0 + s1 * f1 + round_offset);
      auto res1 = hn::ShiftRight<FILTER_BITS>(s2 * f0 + s3 * f1 + round_offset);
      StoreUnaligned2x8(tag16, dst + y * dst_stride, dst_stride, res0);
      StoreUnaligned2x8(tag16, dst + (y + 2) * dst_stride, dst_stride, res1);
    }
  } else if (w == 16) {
    constexpr hn::Half<decltype(tag16)> half_tag16;
    auto s0 = hn::PromoteTo(tag16, hn::LoadU(pixel_tag, src));
    for (int y = 0; y < h; y += 4) {
      auto s1 = hn::PromoteTo(tag16,
                              hn::LoadU(pixel_tag, src + (y + 1) * src_stride));
      auto s2 = hn::PromoteTo(tag16,
                              hn::LoadU(pixel_tag, src + (y + 2) * src_stride));
      auto s3 = hn::PromoteTo(tag16,
                              hn::LoadU(pixel_tag, src + (y + 3) * src_stride));
      auto s4 = hn::PromoteTo(tag16,
                              hn::LoadU(pixel_tag, src + (y + 4) * src_stride));

      auto res0 = hn::ShiftRight<FILTER_BITS>(s0 * f0 + s1 * f1 + round_offset);
      auto res1 = hn::ShiftRight<FILTER_BITS>(s1 * f0 + s2 * f1 + round_offset);
      auto res2 = hn::ShiftRight<FILTER_BITS>(s2 * f0 + s3 * f1 + round_offset);
      auto res3 = hn::ShiftRight<FILTER_BITS>(s3 * f0 + s4 * f1 + round_offset);

      hn::StoreU(
          hn::ReorderDemote2To(pixel_tag, hn::LowerHalf(half_tag16, res0),
                               hn::UpperHalf(half_tag16, res0)),
          pixel_tag, dst + y * dst_stride);
      hn::StoreU(
          hn::ReorderDemote2To(pixel_tag, hn::LowerHalf(half_tag16, res1),
                               hn::UpperHalf(half_tag16, res1)),
          pixel_tag, dst + (y + 1) * dst_stride);
      hn::StoreU(
          hn::ReorderDemote2To(pixel_tag, hn::LowerHalf(half_tag16, res2),
                               hn::UpperHalf(half_tag16, res2)),
          pixel_tag, dst + (y + 2) * dst_stride);
      hn::StoreU(
          hn::ReorderDemote2To(pixel_tag, hn::LowerHalf(half_tag16, res3),
                               hn::UpperHalf(half_tag16, res3)),
          pixel_tag, dst + (y + 3) * dst_stride);

      s0 = s4;
    }
  } else {
    hn::ScalableTag<int16_t> mul_tag;
    hn::Rebind<uint8_t, decltype(mul_tag)> p_tag;
    auto f0_s = hn::Set(mul_tag, filter_y[3]);
    auto f1_s = hn::Set(mul_tag, filter_y[4]);
    auto round_offset_s = hn::Set(mul_tag, 1 << (FILTER_BITS - 1));
    auto vw = hn::Lanes(mul_tag);
    for (int x = 0; x < w; x += vw) {
      auto s0 = hn::PromoteTo(mul_tag, hn::LoadU(p_tag, src + x));
      for (int y = 0; y < h; ++y) {
        auto s1 = hn::PromoteTo(
            mul_tag, hn::LoadU(p_tag, src + x + (y + 1) * src_stride));
        auto res =
            hn::ShiftRight<FILTER_BITS>(s0 * f0_s + s1 * f1_s + round_offset_s);
        auto res_demoted = hn::DemoteTo(p_tag, res);
        if (x + static_cast<int>(vw) > w) {
          hn::StoreN(res_demoted, p_tag, dst + x + y * dst_stride, w - x);
        } else {
          hn::StoreU(res_demoted, p_tag, dst + x + y * dst_stride);
        }
        s0 = s1;
      }
    }
  }
}

HWY_ATTR inline void ConvolveVert4Tap(const uint8_t *src, ptrdiff_t src_stride,
                                      uint8_t *dst, ptrdiff_t dst_stride,
                                      const int16_t *filter_y, int w, int h) {
  hn::CappedTag<int16_t, 16> tag16;
  hn::Rebind<uint8_t, decltype(tag16)> pixel_tag;

  auto f0 = hn::Set(tag16, filter_y[2] >> 1);
  auto f1 = hn::Set(tag16, filter_y[3] >> 1);
  auto f2 = hn::Set(tag16, filter_y[4] >> 1);
  auto f3 = hn::Set(tag16, filter_y[5] >> 1);
  auto round_offset = hn::Set(tag16, 1 << (FILTER_BITS - 2));

  if (w == 4) {
    for (int y = 0; y < h; y += 4) {
      auto s0 = LoadUnaligned4x4(tag16, src + y * src_stride, src_stride);
      auto s1 = LoadUnaligned4x4(tag16, src + (y + 1) * src_stride, src_stride);
      auto s2 = LoadUnaligned4x4(tag16, src + (y + 2) * src_stride, src_stride);
      auto s3 = LoadUnaligned4x4(tag16, src + (y + 3) * src_stride, src_stride);
      auto res = hn::ShiftRight<FILTER_BITS - 1>(s0 * f0 + s1 * f1 + s2 * f2 +
                                                 s3 * f3 + round_offset);
      StoreUnaligned4x4(tag16, dst + y * dst_stride, dst_stride, res);
    }
  } else if (w == 8) {
    auto s0 = LoadUnaligned2x8(tag16, src + 0 * src_stride, src_stride);
    auto s1 = LoadUnaligned2x8(tag16, src + 1 * src_stride, src_stride);
    for (int y = 0; y < h; y += 4) {
      auto s2 = LoadUnaligned2x8(tag16, src + (y + 2) * src_stride, src_stride);
      auto s3 = LoadUnaligned2x8(tag16, src + (y + 3) * src_stride, src_stride);
      auto s4 = LoadUnaligned2x8(tag16, src + (y + 4) * src_stride, src_stride);
      auto s5 = LoadUnaligned2x8(tag16, src + (y + 5) * src_stride, src_stride);
      auto res0 = hn::ShiftRight<FILTER_BITS - 1>(s0 * f0 + s1 * f1 + s2 * f2 +
                                                  s3 * f3 + round_offset);
      auto res1 = hn::ShiftRight<FILTER_BITS - 1>(s2 * f0 + s3 * f1 + s4 * f2 +
                                                  s5 * f3 + round_offset);
      StoreUnaligned2x8(tag16, dst + y * dst_stride, dst_stride, res0);
      StoreUnaligned2x8(tag16, dst + (y + 2) * dst_stride, dst_stride, res1);
      s0 = s4;
      s1 = s5;
    }
  } else if (w == 16) {
    constexpr hn::Half<decltype(tag16)> half_tag16;
    auto s0 = hn::PromoteTo(tag16, hn::LoadU(pixel_tag, src + 0 * src_stride));
    auto s1 = hn::PromoteTo(tag16, hn::LoadU(pixel_tag, src + 1 * src_stride));
    auto s2 = hn::PromoteTo(tag16, hn::LoadU(pixel_tag, src + 2 * src_stride));
    for (int y = 0; y < h; y += 4) {
      auto s3 = hn::PromoteTo(tag16,
                              hn::LoadU(pixel_tag, src + (y + 3) * src_stride));
      auto s4 = hn::PromoteTo(tag16,
                              hn::LoadU(pixel_tag, src + (y + 4) * src_stride));
      auto s5 = hn::PromoteTo(tag16,
                              hn::LoadU(pixel_tag, src + (y + 5) * src_stride));
      auto s6 = hn::PromoteTo(tag16,
                              hn::LoadU(pixel_tag, src + (y + 6) * src_stride));

      auto res0 = hn::ShiftRight<FILTER_BITS - 1>(s0 * f0 + s1 * f1 + s2 * f2 +
                                                  s3 * f3 + round_offset);
      auto res1 = hn::ShiftRight<FILTER_BITS - 1>(s1 * f0 + s2 * f1 + s3 * f2 +
                                                  s4 * f3 + round_offset);
      auto res2 = hn::ShiftRight<FILTER_BITS - 1>(s2 * f0 + s3 * f1 + s4 * f2 +
                                                  s5 * f3 + round_offset);
      auto res3 = hn::ShiftRight<FILTER_BITS - 1>(s3 * f0 + s4 * f1 + s5 * f2 +
                                                  s6 * f3 + round_offset);

      hn::StoreU(
          hn::ReorderDemote2To(pixel_tag, hn::LowerHalf(half_tag16, res0),
                               hn::UpperHalf(half_tag16, res0)),
          pixel_tag, dst + y * dst_stride);
      hn::StoreU(
          hn::ReorderDemote2To(pixel_tag, hn::LowerHalf(half_tag16, res1),
                               hn::UpperHalf(half_tag16, res1)),
          pixel_tag, dst + (y + 1) * dst_stride);
      hn::StoreU(
          hn::ReorderDemote2To(pixel_tag, hn::LowerHalf(half_tag16, res2),
                               hn::UpperHalf(half_tag16, res2)),
          pixel_tag, dst + (y + 2) * dst_stride);
      hn::StoreU(
          hn::ReorderDemote2To(pixel_tag, hn::LowerHalf(half_tag16, res3),
                               hn::UpperHalf(half_tag16, res3)),
          pixel_tag, dst + (y + 3) * dst_stride);

      s0 = s4;
      s1 = s5;
      s2 = s6;
    }
  } else {
    hn::ScalableTag<int16_t> mul_tag;
    hn::Rebind<uint8_t, decltype(mul_tag)> p_tag;
    auto f0_s = hn::Set(mul_tag, filter_y[2] >> 1);
    auto f1_s = hn::Set(mul_tag, filter_y[3] >> 1);
    auto f2_s = hn::Set(mul_tag, filter_y[4] >> 1);
    auto f3_s = hn::Set(mul_tag, filter_y[5] >> 1);
    auto round_offset_s = hn::Set(mul_tag, 1 << (FILTER_BITS - 2));
    auto vw = hn::Lanes(mul_tag);
    for (int x = 0; x < w; x += vw) {
      auto s0 = hn::PromoteTo(mul_tag, hn::LoadU(p_tag, src + x));
      auto s1 =
          hn::PromoteTo(mul_tag, hn::LoadU(p_tag, src + x + 1 * src_stride));
      auto s2 =
          hn::PromoteTo(mul_tag, hn::LoadU(p_tag, src + x + 2 * src_stride));
      for (int y = 0; y < h; ++y) {
        auto s3 = hn::PromoteTo(
            mul_tag, hn::LoadU(p_tag, src + x + (y + 3) * src_stride));
        auto res = hn::ShiftRight<FILTER_BITS - 1>(
            s0 * f0_s + s1 * f1_s + s2 * f2_s + s3 * f3_s + round_offset_s);
        auto res_demoted = hn::DemoteTo(p_tag, res);
        if (x + static_cast<int>(vw) > w) {
          hn::StoreN(res_demoted, p_tag, dst + x + y * dst_stride, w - x);
        } else {
          hn::StoreU(res_demoted, p_tag, dst + x + y * dst_stride);
        }
        s0 = s1;
        s1 = s2;
        s2 = s3;
      }
    }
  }
}

HWY_ATTR inline void ConvolveVert8Tap(const uint8_t *src, ptrdiff_t src_stride,
                                      uint8_t *dst, ptrdiff_t dst_stride,
                                      const int16_t *filter_y, int w, int h) {
  hn::CappedTag<int16_t, 16> tag16;
  hn::Rebind<uint8_t, decltype(tag16)> pixel_tag;

  auto f0 = hn::Set(tag16, filter_y[0] >> 1);
  auto f1 = hn::Set(tag16, filter_y[1] >> 1);
  auto f2 = hn::Set(tag16, filter_y[2] >> 1);
  auto f3 = hn::Set(tag16, filter_y[3] >> 1);
  auto f4 = hn::Set(tag16, filter_y[4] >> 1);
  auto f5 = hn::Set(tag16, filter_y[5] >> 1);
  auto f6 = hn::Set(tag16, filter_y[6] >> 1);
  auto f7 = hn::Set(tag16, filter_y[7] >> 1);
  auto round_offset = hn::Set(tag16, 1 << (FILTER_BITS - 2));

  if (w == 4) {
    for (int y = 0; y < h; y += 4) {
      auto s0 = LoadUnaligned4x4(tag16, src + (y + 0) * src_stride, src_stride);
      auto s1 = LoadUnaligned4x4(tag16, src + (y + 1) * src_stride, src_stride);
      auto s2 = LoadUnaligned4x4(tag16, src + (y + 2) * src_stride, src_stride);
      auto s3 = LoadUnaligned4x4(tag16, src + (y + 3) * src_stride, src_stride);
      auto s4 = LoadUnaligned4x4(tag16, src + (y + 4) * src_stride, src_stride);
      auto s5 = LoadUnaligned4x4(tag16, src + (y + 5) * src_stride, src_stride);
      auto s6 = LoadUnaligned4x4(tag16, src + (y + 6) * src_stride, src_stride);
      auto s7 = LoadUnaligned4x4(tag16, src + (y + 7) * src_stride, src_stride);
      auto res = hn::ShiftRight<FILTER_BITS - 1>(
          s0 * f0 + s1 * f1 + s2 * f2 + s3 * f3 + s4 * f4 + s5 * f5 + s6 * f6 +
          s7 * f7 + round_offset);
      StoreUnaligned4x4(tag16, dst + y * dst_stride, dst_stride, res);
    }
  } else if (w == 8) {
    auto s0 = LoadUnaligned2x8(tag16, src + 0 * src_stride, src_stride);
    auto s1 = LoadUnaligned2x8(tag16, src + 1 * src_stride, src_stride);
    auto s2 = LoadUnaligned2x8(tag16, src + 2 * src_stride, src_stride);
    auto s3 = LoadUnaligned2x8(tag16, src + 3 * src_stride, src_stride);
    auto s4 = LoadUnaligned2x8(tag16, src + 4 * src_stride, src_stride);
    auto s5 = LoadUnaligned2x8(tag16, src + 5 * src_stride, src_stride);
    for (int y = 0; y < h; y += 2) {
      auto s6 = LoadUnaligned2x8(tag16, src + (y + 6) * src_stride, src_stride);
      auto s7 = LoadUnaligned2x8(tag16, src + (y + 7) * src_stride, src_stride);
      auto res = hn::ShiftRight<FILTER_BITS - 1>(
          s0 * f0 + s1 * f1 + s2 * f2 + s3 * f3 + s4 * f4 + s5 * f5 + s6 * f6 +
          s7 * f7 + round_offset);
      StoreUnaligned2x8(tag16, dst + y * dst_stride, dst_stride, res);
      s0 = s2;
      s1 = s3;
      s2 = s4;
      s3 = s5;
      s4 = s6;
      s5 = s7;
    }
  } else if (w == 16) {
    constexpr hn::Half<decltype(tag16)> half_tag16;
    auto s0 = hn::PromoteTo(tag16, hn::LoadU(pixel_tag, src + 0 * src_stride));
    auto s1 = hn::PromoteTo(tag16, hn::LoadU(pixel_tag, src + 1 * src_stride));
    auto s2 = hn::PromoteTo(tag16, hn::LoadU(pixel_tag, src + 2 * src_stride));
    auto s3 = hn::PromoteTo(tag16, hn::LoadU(pixel_tag, src + 3 * src_stride));
    auto s4 = hn::PromoteTo(tag16, hn::LoadU(pixel_tag, src + 4 * src_stride));
    auto s5 = hn::PromoteTo(tag16, hn::LoadU(pixel_tag, src + 5 * src_stride));
    auto s6 = hn::PromoteTo(tag16, hn::LoadU(pixel_tag, src + 6 * src_stride));
    for (int y = 0; y < h; ++y) {
      auto s7 = hn::PromoteTo(tag16,
                              hn::LoadU(pixel_tag, src + (y + 7) * src_stride));
      auto res = hn::ShiftRight<FILTER_BITS - 1>(
          s0 * f0 + s1 * f1 + s2 * f2 + s3 * f3 + s4 * f4 + s5 * f5 + s6 * f6 +
          s7 * f7 + round_offset);
      hn::StoreU(hn::ReorderDemote2To(pixel_tag, hn::LowerHalf(half_tag16, res),
                                      hn::UpperHalf(half_tag16, res)),
                 pixel_tag, dst + y * dst_stride);
      s0 = s1;
      s1 = s2;
      s2 = s3;
      s3 = s4;
      s4 = s5;
      s5 = s6;
      s6 = s7;
    }
  } else {
    hn::ScalableTag<int16_t> mul_tag;
    hn::Rebind<uint8_t, decltype(mul_tag)> p_tag;
    auto f0_s = hn::Set(mul_tag, filter_y[0] >> 1);
    auto f1_s = hn::Set(mul_tag, filter_y[1] >> 1);
    auto f2_s = hn::Set(mul_tag, filter_y[2] >> 1);
    auto f3_s = hn::Set(mul_tag, filter_y[3] >> 1);
    auto f4_s = hn::Set(mul_tag, filter_y[4] >> 1);
    auto f5_s = hn::Set(mul_tag, filter_y[5] >> 1);
    auto f6_s = hn::Set(mul_tag, filter_y[6] >> 1);
    auto f7_s = hn::Set(mul_tag, filter_y[7] >> 1);
    auto round_offset_s = hn::Set(mul_tag, 1 << (FILTER_BITS - 2));
    auto vw = hn::Lanes(mul_tag);
    for (int x = 0; x < w; x += vw) {
      auto s0 =
          hn::PromoteTo(mul_tag, hn::LoadU(p_tag, src + x + 0 * src_stride));
      auto s1 =
          hn::PromoteTo(mul_tag, hn::LoadU(p_tag, src + x + 1 * src_stride));
      auto s2 =
          hn::PromoteTo(mul_tag, hn::LoadU(p_tag, src + x + 2 * src_stride));
      auto s3 =
          hn::PromoteTo(mul_tag, hn::LoadU(p_tag, src + x + 3 * src_stride));
      auto s4 =
          hn::PromoteTo(mul_tag, hn::LoadU(p_tag, src + x + 4 * src_stride));
      auto s5 =
          hn::PromoteTo(mul_tag, hn::LoadU(p_tag, src + x + 5 * src_stride));
      auto s6 =
          hn::PromoteTo(mul_tag, hn::LoadU(p_tag, src + x + 6 * src_stride));
      for (int y = 0; y < h; ++y) {
        auto s7 = hn::PromoteTo(
            mul_tag, hn::LoadU(p_tag, src + x + (y + 7) * src_stride));
        auto sum = s0 * f0_s + s1 * f1_s + s2 * f2_s + s3 * f3_s + s4 * f4_s +
                   s5 * f5_s + s6 * f6_s + s7 * f7_s;
        auto res = hn::ShiftRight<FILTER_BITS - 1>(sum + round_offset_s);
        auto res_demoted = hn::DemoteTo(p_tag, res);
        if (x + static_cast<int>(vw) > w) {
          hn::StoreN(res_demoted, p_tag, dst + x + y * dst_stride, w - x);
        } else {
          hn::StoreU(res_demoted, p_tag, dst + x + y * dst_stride);
        }
        s0 = s1;
        s1 = s2;
        s2 = s3;
        s3 = s4;
        s4 = s5;
        s5 = s6;
        s6 = s7;
      }
    }
  }
}

HWY_MAYBE_UNUSED void Convolve8Vert(const uint8_t *src, ptrdiff_t src_stride,
                                    uint8_t *dst, ptrdiff_t dst_stride,
                                    const int16_t *filter_x, int x_step_q4,
                                    const int16_t *filter_y, int y_step_q4,
                                    int w, int h) {
  assert((intptr_t)dst % 4 == 0);
  assert(dst_stride % 4 == 0);

  (void)x_step_q4;
  (void)filter_x;
  (void)y_step_q4;

  src -= src_stride * ((SUBPEL_TAPS / 2) - 1);
  int filter_taps = get_filter_taps_convolve8(filter_y);
  if (filter_taps == 2) {
    ConvolveVert2Tap(src + src_stride * 3, src_stride, dst, dst_stride,
                     filter_y, w, h);
  } else if (filter_taps == 4) {
    ConvolveVert4Tap(src + src_stride * 2, src_stride, dst, dst_stride,
                     filter_y, w, h);
  } else {
    // filter_taps = 8
    ConvolveVert8Tap(src, src_stride, dst, dst_stride, filter_y, w, h);
  }
}

HWY_MAYBE_UNUSED void Convolve8Horiz(const uint8_t *src, ptrdiff_t src_stride,
                                     uint8_t *dst, ptrdiff_t dst_stride,
                                     const int16_t *filter_x, int x_step_q4,
                                     const int16_t *filter_y, int y_step_q4,
                                     int w, int h) {
  assert((intptr_t)dst % 4 == 0);
  assert(dst_stride % 4 == 0);

  (void)x_step_q4;
  (void)filter_y;
  (void)y_step_q4;

  src -= ((SUBPEL_TAPS / 2) - 1);
  int filter_taps = get_filter_taps_convolve8(filter_x);
  if (filter_taps == 2) {
    ConvolveHoriz2Tap(src + 3, src_stride, dst, dst_stride, filter_x, w, h);
  } else if (filter_taps == 4) {
    ConvolveHoriz4Tap(src + 2, src_stride, dst, dst_stride, filter_x, w, h);
  } else {
    // filter_taps = 8
    ConvolveHoriz8Tap(src, src_stride, dst, dst_stride, filter_x, w, h);
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace

#define CONVOLVE8HORIZ(suffix)                                                \
  extern "C" void aom_convolve8_horiz_##suffix(                               \
      const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,                 \
      ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4,           \
      const int16_t *filter_y, int y_step_q4, int w, int h);                  \
  HWY_ATTR void aom_convolve8_horiz_##suffix(                                 \
      const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,                 \
      ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4,           \
      const int16_t *filter_y, int y_step_q4, int w, int h) {                 \
    HWY_NAMESPACE::Convolve8Horiz(src, src_stride, dst, dst_stride, filter_x, \
                                  x_step_q4, filter_y, y_step_q4, w, h);      \
  }

HWY_AFTER_NAMESPACE();

#endif  // AOM_AOM_DSP_CONVOLVE_HWY_H_