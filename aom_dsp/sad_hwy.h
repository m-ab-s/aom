/*
 * Copyright (c) 2025, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */
#ifndef AOM_AOM_DSP_SAD_HWY_H_
#define AOM_AOM_DSP_SAD_HWY_H_

#include <assert.h>

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"

#include "aom_dsp/blend.h"
#include "aom_dsp/reduce_sum_hwy.h"
#include "third_party/highway/hwy/highway.h"

HWY_BEFORE_NAMESPACE();

namespace {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

template <class D, class VHalf>
HWY_ATTR HWY_INLINE hn::VFromD<D> Combine2Rows(D d, VHalf r0, VHalf r1) {
  if constexpr (hn::MaxLanes(d) * sizeof(hn::TFromD<D>) >= 32) {
    return hn::ConcatLowerLower(d, hn::ResizeBitCast(d, r1),
                                hn::ResizeBitCast(d, r0));
  } else {
    return hn::Combine(d, r1, r0);
  }
}

template <class D, class VQuarter>
HWY_ATTR HWY_INLINE hn::VFromD<D> Combine4Rows(D d, VQuarter r0, VQuarter r1,
                                               VQuarter r2, VQuarter r3) {
  if constexpr (hn::MaxLanes(d) * sizeof(hn::TFromD<D>) == 64) {
    auto v = hn::ResizeBitCast(d, r0);
    v = hn::InsertBlock<1>(v, r1);
    v = hn::InsertBlock<2>(v, r2);
    return hn::InsertBlock<3>(v, r3);
  } else {
    constexpr hn::Half<D> d_half;
    auto r01 = Combine2Rows(d_half, r0, r1);
    auto r23 = Combine2Rows(d_half, r2, r3);
    return Combine2Rows(d, r01, r23);
  }
}

template <int RowsPerVec, class D>
HWY_ATTR HWY_INLINE hn::VFromD<D> LoadRowsU(D full_tag, const uint8_t *ptr,
                                            ptrdiff_t stride) {
  static_assert(RowsPerVec == 1 || RowsPerVec == 2 || RowsPerVec == 4,
                "RowsPerVec must be 1, 2, or 4.");
  if constexpr (RowsPerVec == 4) {
    constexpr hn::Half<hn::Half<D>> row_tag;
    const uint8_t *ptr2 = ptr + stride * 2;
    return Combine4Rows(
        full_tag, hn::LoadU(row_tag, ptr), hn::LoadU(row_tag, ptr + stride),
        hn::LoadU(row_tag, ptr2), hn::LoadU(row_tag, ptr2 + stride));
  } else if constexpr (RowsPerVec == 2) {
    constexpr hn::Half<D> row_tag;
    return Combine2Rows(full_tag, hn::LoadU(row_tag, ptr),
                        hn::LoadU(row_tag, ptr + stride));
  } else {
    return hn::LoadU(full_tag, ptr);
  }
}

template <int BlockWidth, bool IsSadAvg = false>
HWY_ATTR HWY_INLINE unsigned int SumOfAbsoluteDiff(
    const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,
    int ref_stride, int h, const uint8_t *second_pred = nullptr) {
  constexpr hn::ScalableTag<uint8_t> pixel_tag;
  constexpr int vw = hn::MaxLanes(pixel_tag);
  constexpr int kRowsPerVec = (BlockWidth < vw) ? (vw / BlockWidth) : 1;
  constexpr int kRowSpan = (BlockWidth < vw) ? vw : BlockWidth;
  assert(h % kRowsPerVec == 0);
  constexpr hn::Repartition<uint64_t, decltype(pixel_tag)> intermediate_sum_tag;
  auto sum_sad = hn::Zero(intermediate_sum_tag);
  for (int i = 0; i < h; i += kRowsPerVec) {
    for (int j = 0; j < kRowSpan; j += vw) {
      auto src_vec = LoadRowsU<kRowsPerVec>(pixel_tag, &src_ptr[j], src_stride);
      auto ref_vec = LoadRowsU<kRowsPerVec>(pixel_tag, &ref_ptr[j], ref_stride);
      if constexpr (IsSadAvg) {
        auto sec_pred_vec = hn::LoadU(pixel_tag, &second_pred[j]);
        ref_vec = hn::AverageRound(ref_vec, sec_pred_vec);
      }
      auto sad = hn::SumsOf8AbsDiff(src_vec, ref_vec);
      sum_sad = hn::Add(sum_sad, sad);
    }
    src_ptr += src_stride * kRowsPerVec;
    ref_ptr += ref_stride * kRowsPerVec;
    if constexpr (IsSadAvg) {
      second_pred += BlockWidth * kRowsPerVec;
    }
  }
  return static_cast<unsigned int>(
      hn::ReduceSum(intermediate_sum_tag, sum_sad));
}

template <int BlockWidth, int NumRef>
HWY_ATTR HWY_INLINE void SumOfAbsoluteDiffND(const uint8_t *src_ptr,
                                             int src_stride,
                                             const uint8_t *const ref_ptr[4],
                                             int ref_stride, int h,
                                             uint32_t res[4]) {
  static_assert(NumRef == 3 || NumRef == 4, "NumRef must be 3 or 4.");
  constexpr hn::ScalableTag<uint8_t> pixel_tag;
  constexpr int vw = hn::MaxLanes(pixel_tag);
  constexpr int kRowsPerVec = (BlockWidth < vw) ? (vw / BlockWidth) : 1;
  constexpr int kRowSpan = (BlockWidth < vw) ? vw : BlockWidth;
  assert(h % kRowsPerVec == 0);
  constexpr hn::Repartition<uint64_t, decltype(pixel_tag)> intermediate_sum_tag;
  auto sum_sad_0 = hn::Zero(intermediate_sum_tag);
  auto sum_sad_1 = hn::Zero(intermediate_sum_tag);
  auto sum_sad_2 = hn::Zero(intermediate_sum_tag);
  auto sum_sad_3 = hn::Zero(intermediate_sum_tag);
  const uint8_t *ref_0, *ref_1, *ref_2;
  const uint8_t *ref_3 = nullptr;
  ref_0 = ref_ptr[0];
  ref_1 = ref_ptr[1];
  ref_2 = ref_ptr[2];
  if constexpr (NumRef == 4) {
    ref_3 = ref_ptr[3];
  }
  for (int i = 0; i < h; i += kRowsPerVec) {
    for (int j = 0; j < kRowSpan; j += vw) {
      auto src_vec = LoadRowsU<kRowsPerVec>(pixel_tag, &src_ptr[j], src_stride);
      auto ref_vec_0 = LoadRowsU<kRowsPerVec>(pixel_tag, &ref_0[j], ref_stride);
      auto ref_vec_1 = LoadRowsU<kRowsPerVec>(pixel_tag, &ref_1[j], ref_stride);
      auto ref_vec_2 = LoadRowsU<kRowsPerVec>(pixel_tag, &ref_2[j], ref_stride);
      auto sad_0 = hn::SumsOf8AbsDiff(src_vec, ref_vec_0);
      auto sad_1 = hn::SumsOf8AbsDiff(src_vec, ref_vec_1);
      auto sad_2 = hn::SumsOf8AbsDiff(src_vec, ref_vec_2);
      sum_sad_0 = hn::Add(sum_sad_0, sad_0);
      sum_sad_1 = hn::Add(sum_sad_1, sad_1);
      sum_sad_2 = hn::Add(sum_sad_2, sad_2);
      if constexpr (NumRef == 4) {
        auto ref_vec_3 =
            LoadRowsU<kRowsPerVec>(pixel_tag, &ref_3[j], ref_stride);
        auto sad_3 = hn::SumsOf8AbsDiff(src_vec, ref_vec_3);
        sum_sad_3 = hn::Add(sum_sad_3, sad_3);
      }
    }
    src_ptr += src_stride * kRowsPerVec;
    ref_0 += ref_stride * kRowsPerVec;
    ref_1 += ref_stride * kRowsPerVec;
    ref_2 += ref_stride * kRowsPerVec;
    if constexpr (NumRef == 4) {
      ref_3 += ref_stride * kRowsPerVec;
    }
  }
  constexpr hn::Repartition<uint32_t, decltype(pixel_tag)> uint32_tag;
  auto r02 = hn::InterleaveEven(uint32_tag, hn::BitCast(uint32_tag, sum_sad_0),
                                hn::BitCast(uint32_tag, sum_sad_2));
  auto r13 = hn::InterleaveEven(uint32_tag, hn::BitCast(uint32_tag, sum_sad_1),
                                hn::BitCast(uint32_tag, sum_sad_3));
  auto r0123 = hn::Add(hn::InterleaveLower(uint32_tag, r02, r13),
                       hn::InterleaveUpper(uint32_tag, r02, r13));

  auto block_sum = BlockReduceSum(uint32_tag, r0123);
  constexpr hn::FixedTag<uint32_t, 4> block_sum_tag;
  hn::StoreU(block_sum, block_sum_tag, res);
}

template <int BlockWidth>
HWY_ATTR HWY_INLINE unsigned int MaskedSumOfAbsoluteDiff(
    const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,
    int ref_stride, const uint8_t *second_pred, const uint8_t *msk_ptr,
    int msk_stride, int invert_mask, int h) {
  constexpr hn::ScalableTag<uint8_t> pixel_tag;
  constexpr hn::Repartition<int8_t, decltype(pixel_tag)> int8_tag;
  constexpr hn::Repartition<int16_t, decltype(pixel_tag)> int16_tag;
  constexpr hn::Repartition<uint64_t, decltype(pixel_tag)> intermediate_sum_tag;
  constexpr int vw = hn::MaxLanes(pixel_tag);
  constexpr int kRowsPerVec = (BlockWidth < vw) ? (vw / BlockWidth) : 1;
  constexpr int kRowSpan = (BlockWidth < vw) ? vw : BlockWidth;
  assert(h % kRowsPerVec == 0);

  const auto mask_max = hn::Set(pixel_tag, 1 << AOM_BLEND_A64_ROUND_BITS);
  const auto round_scale =
      hn::Set(int16_tag, 1 << (15 - AOM_BLEND_A64_ROUND_BITS));
  auto sum_sad = hn::Zero(intermediate_sum_tag);

  for (int y = 0; y < h; y += kRowsPerVec) {
    for (int x = 0; x < kRowSpan; x += vw) {
      auto src = LoadRowsU<kRowsPerVec>(pixel_tag, &src_ptr[x], src_stride);
      auto ref = LoadRowsU<kRowsPerVec>(pixel_tag, &ref_ptr[x], ref_stride);
      auto sec = hn::LoadU(pixel_tag, &second_pred[x]);
      auto m = LoadRowsU<kRowsPerVec>(pixel_tag, &msk_ptr[x], msk_stride);
      auto m_inv = hn::Sub(mask_max, m);
      auto a = invert_mask ? sec : ref;
      auto b = invert_mask ? ref : sec;

      auto data_l = hn::InterleaveLower(pixel_tag, a, b);
      auto mask_l = hn::InterleaveLower(pixel_tag, m, m_inv);
      auto pred_l = hn::MulFixedPoint15(
          hn::SatWidenMulPairwiseAdd(int16_tag, data_l,
                                     hn::BitCast(int8_tag, mask_l)),
          round_scale);

      auto data_r = hn::InterleaveUpper(pixel_tag, a, b);
      auto mask_r = hn::InterleaveUpper(pixel_tag, m, m_inv);
      auto pred_r = hn::MulFixedPoint15(
          hn::SatWidenMulPairwiseAdd(int16_tag, data_r,
                                     hn::BitCast(int8_tag, mask_r)),
          round_scale);

      auto pred = hn::ReorderDemote2To(pixel_tag, pred_l, pred_r);
      sum_sad = hn::Add(sum_sad, hn::SumsOf8AbsDiff(pred, src));
    }
    src_ptr += src_stride * kRowsPerVec;
    ref_ptr += ref_stride * kRowsPerVec;
    second_pred += BlockWidth * kRowsPerVec;
    msk_ptr += msk_stride * kRowsPerVec;
  }
  return static_cast<unsigned int>(
      hn::ReduceSum(intermediate_sum_tag, sum_sad));
}

#if !CONFIG_REALTIME_ONLY
template <int BlockWidth>
HWY_ATTR HWY_INLINE unsigned int ObmcSumOfAbsoluteDiff(const uint8_t *pre,
                                                       int pre_stride,
                                                       const int32_t *wsrc,
                                                       const int32_t *mask,
                                                       int h) {
  constexpr hn::ScalableTag<int32_t> d32;
  constexpr hn::Repartition<int16_t, decltype(d32)> d16;
  constexpr hn::Rebind<uint8_t, decltype(d32)> du8;
  constexpr int kLanes = hn::MaxLanes(d32);
  constexpr int kRowsPerVec = (BlockWidth < kLanes) ? (kLanes / BlockWidth) : 1;
  constexpr int kColsPerRow = (BlockWidth < kLanes) ? kLanes : BlockWidth;
  assert(h % kRowsPerVec == 0);

  const auto v_bias_d = hn::Set(d32, (1 << 12) >> 1);
  auto v_sad_d = hn::Zero(d32);
  int n = 0;

  for (int r = 0; r < h; r += kRowsPerVec) {
    for (int c = 0; c < kColsPerRow; c += kLanes) {
      auto v_p_b = LoadRowsU<kRowsPerVec>(du8, pre + c, pre_stride);
      auto v_p_d = hn::PromoteTo(d32, v_p_b);
      auto v_m_d = hn::LoadU(d32, mask + n);
      auto v_w_d = hn::LoadU(d32, wsrc + n);
      // Values in both pre and mask fit in signed 15 bits and are packed at
      // 32-bit boundaries, so WidenMulPairwiseAdd (vpmaddwd) produces the
      // exact product without overflow (matching obmc_sad_avx2.c).
      auto v_pm_d = hn::WidenMulPairwiseAdd(d32, hn::BitCast(d16, v_p_d),
                                            hn::BitCast(d16, v_m_d));
      auto v_diff_d = hn::Sub(v_w_d, v_pm_d);
      auto v_absdiff_d = hn::Abs(v_diff_d);
      auto v_rad_d = hn::ShiftRight<12>(hn::Add(v_absdiff_d, v_bias_d));
      v_sad_d = hn::Add(v_sad_d, v_rad_d);
      n += kLanes;
    }
    pre += pre_stride * kRowsPerVec;
  }
  return static_cast<unsigned int>(hn::ReduceSum(d32, v_sad_d));
}
#endif  // !CONFIG_REALTIME_ONLY

}  // namespace HWY_NAMESPACE
}  // namespace

#define FSAD(w, h, suffix)                                                   \
  extern "C" unsigned int aom_sad##w##x##h##_##suffix(                       \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,        \
      int ref_stride);                                                       \
  HWY_ATTR unsigned int aom_sad##w##x##h##_##suffix(                         \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,        \
      int ref_stride) {                                                      \
    return HWY_NAMESPACE::SumOfAbsoluteDiff<w>(src_ptr, src_stride, ref_ptr, \
                                               ref_stride, h);               \
  }

#define FSAD_4D(w, h, suffix)                                                  \
  extern "C" void aom_sad##w##x##h##x4d_##suffix(                              \
      const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_ptr[4], \
      int ref_stride, uint32_t res[4]);                                        \
  HWY_ATTR void aom_sad##w##x##h##x4d_##suffix(                                \
      const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_ptr[4], \
      int ref_stride, uint32_t res[4]) {                                       \
    HWY_NAMESPACE::SumOfAbsoluteDiffND<w, 4>(src_ptr, src_stride, ref_ptr,     \
                                             ref_stride, h, res);              \
  }

#define FSAD_3D(w, h, suffix)                                                  \
  extern "C" void aom_sad##w##x##h##x3d_##suffix(                              \
      const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_ptr[4], \
      int ref_stride, uint32_t res[4]);                                        \
  HWY_ATTR void aom_sad##w##x##h##x3d_##suffix(                                \
      const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_ptr[4], \
      int ref_stride, uint32_t res[4]) {                                       \
    HWY_NAMESPACE::SumOfAbsoluteDiffND<w, 3>(src_ptr, src_stride, ref_ptr,     \
                                             ref_stride, h, res);              \
  }

#define FSAD_SKIP(w, h, suffix)                                              \
  extern "C" unsigned int aom_sad_skip_##w##x##h##_##suffix(                 \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,        \
      int ref_stride);                                                       \
  HWY_ATTR unsigned int aom_sad_skip_##w##x##h##_##suffix(                   \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,        \
      int ref_stride) {                                                      \
    return 2 * HWY_NAMESPACE::SumOfAbsoluteDiff<w>(                          \
                   src_ptr, src_stride * 2, ref_ptr, ref_stride * 2, h / 2); \
  }

#define FSAD_4D_SKIP(w, h, suffix)                                             \
  extern "C" void aom_sad_skip_##w##x##h##x4d_##suffix(                        \
      const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_ptr[4], \
      int ref_stride, uint32_t res[4]);                                        \
  HWY_ATTR void aom_sad_skip_##w##x##h##x4d_##suffix(                          \
      const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_ptr[4], \
      int ref_stride, uint32_t res[4]) {                                       \
    HWY_NAMESPACE::SumOfAbsoluteDiffND<w, 4>(src_ptr, 2 * src_stride, ref_ptr, \
                                             2 * ref_stride, ((h) >> 1), res); \
    res[0] <<= 1;                                                              \
    res[1] <<= 1;                                                              \
    res[2] <<= 1;                                                              \
    res[3] <<= 1;                                                              \
  }

#define FSAD_AVG(w, h, suffix)                                        \
  extern "C" unsigned int aom_sad##w##x##h##_avg_##suffix(            \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, \
      int ref_stride, const uint8_t *second_pred);                    \
  HWY_ATTR unsigned int aom_sad##w##x##h##_avg_##suffix(              \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, \
      int ref_stride, const uint8_t *second_pred) {                   \
    return HWY_NAMESPACE::SumOfAbsoluteDiff<w, true>(                 \
        src_ptr, src_stride, ref_ptr, ref_stride, h, second_pred);    \
  }

#define FMASKED_SAD(w, h, suffix)                                             \
  extern "C" unsigned int aom_masked_sad##w##x##h##_##suffix(                 \
      const uint8_t *src, int src_stride, const uint8_t *ref, int ref_stride, \
      const uint8_t *second_pred, const uint8_t *msk, int msk_stride,         \
      int invert_mask);                                                       \
  HWY_ATTR unsigned int aom_masked_sad##w##x##h##_##suffix(                   \
      const uint8_t *src, int src_stride, const uint8_t *ref, int ref_stride, \
      const uint8_t *second_pred, const uint8_t *msk, int msk_stride,         \
      int invert_mask) {                                                      \
    return HWY_NAMESPACE::MaskedSumOfAbsoluteDiff<w>(                         \
        src, src_stride, ref, ref_stride, second_pred, msk, msk_stride,       \
        invert_mask, h);                                                      \
  }

#if !CONFIG_REALTIME_ONLY
#define FOBMC_SAD(w, h, suffix)                                           \
  extern "C" unsigned int aom_obmc_sad##w##x##h##_##suffix(               \
      const uint8_t *pre, int pre_stride, const int32_t *wsrc,            \
      const int32_t *mask);                                               \
  HWY_ATTR unsigned int aom_obmc_sad##w##x##h##_##suffix(                 \
      const uint8_t *pre, int pre_stride, const int32_t *wsrc,            \
      const int32_t *mask) {                                              \
    return HWY_NAMESPACE::ObmcSumOfAbsoluteDiff<w>(pre, pre_stride, wsrc, \
                                                   mask, h);              \
  }
#endif  // !CONFIG_REALTIME_ONLY

#define FOR_EACH_SAD_BLOCK_SIZE(X, suffix) \
  X(128, 128, suffix)                      \
  X(128, 64, suffix)                       \
  X(64, 128, suffix)                       \
  X(64, 64, suffix)                        \
  X(64, 32, suffix)

#if !CONFIG_REALTIME_ONLY
#define FOR_EACH_SAD_SMALL_SKIP_RECT_BLOCK_SIZE(X, suffix) \
  X(64, 16, suffix)                                        \
  X(16, 64, suffix)

#define FOR_EACH_SAD_H8_RECT_BLOCK_SIZE(X, suffix) X(32, 8, suffix)

#define FOR_EACH_SAD_H4_BLOCK_SIZE(X, suffix) X(16, 4, suffix)

#define FOR_EACH_MASKED_OBMC_RECT_BLOCK_SIZE(X, suffix) \
  X(64, 16, suffix)                                     \
  X(16, 64, suffix)                                     \
  X(32, 8, suffix)                                      \
  X(16, 4, suffix)
#else
#define FOR_EACH_SAD_SMALL_SKIP_RECT_BLOCK_SIZE(X, suffix)
#define FOR_EACH_SAD_H8_RECT_BLOCK_SIZE(X, suffix)
#define FOR_EACH_SAD_H4_BLOCK_SIZE(X, suffix)
#define FOR_EACH_MASKED_OBMC_RECT_BLOCK_SIZE(X, suffix)
#endif  // !CONFIG_REALTIME_ONLY

#define FOR_EACH_SAD_SMALL_SKIP_BLOCK_SIZE(X, suffix) \
  X(32, 32, suffix)                                   \
  X(32, 16, suffix)                                   \
  X(16, 32, suffix)                                   \
  X(16, 16, suffix)                                   \
  FOR_EACH_SAD_SMALL_SKIP_RECT_BLOCK_SIZE(X, suffix)

#define FOR_EACH_SAD_H8_BLOCK_SIZE(X, suffix) \
  X(16, 8, suffix)                            \
  FOR_EACH_SAD_H8_RECT_BLOCK_SIZE(X, suffix)

#define FOR_EACH_MASKED_OBMC_SAD_BLOCK_SIZE(X, suffix) \
  X(128, 128, suffix)                                  \
  X(128, 64, suffix)                                   \
  X(64, 128, suffix)                                   \
  X(64, 64, suffix)                                    \
  X(64, 32, suffix)                                    \
  X(32, 64, suffix)                                    \
  X(32, 32, suffix)                                    \
  X(32, 16, suffix)                                    \
  X(16, 32, suffix)                                    \
  X(16, 16, suffix)                                    \
  X(16, 8, suffix)                                     \
  FOR_EACH_MASKED_OBMC_RECT_BLOCK_SIZE(X, suffix)

HWY_AFTER_NAMESPACE();

#endif  // AOM_AOM_DSP_SAD_HWY_H_
