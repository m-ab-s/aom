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

#ifndef AOM_AV1_ENCODER_WEDGE_UTILS_HWY_H_
#define AOM_AV1_ENCODER_WEDGE_UTILS_HWY_H_

#include <assert.h>
#include <stdint.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"

#include "aom/aom_integer.h"
#include "aom_dsp/aom_dsp_common.h"
#include "av1/common/reconinter.h"
#include "third_party/highway/hwy/highway.h"

HWY_BEFORE_NAMESPACE();

namespace {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

HWY_ATTR HWY_INLINE uint64_t WedgeSseFromResiduals(const int16_t *r1,
                                                   const int16_t *d,
                                                   const uint8_t *m, int N) {
  assert(N > 0 && N % 64 == 0);
  constexpr hn::ScalableTag<int16_t> int16xN_tag;
  constexpr hn::ScalableTag<int32_t> int32xN_tag;
  constexpr hn::ScalableTag<uint32_t> uint32xN_tag;
  constexpr hn::ScalableTag<uint64_t> uint64xN_tag;
  constexpr int kStep = static_cast<int>(hn::MaxLanes(int16xN_tag));
  static_assert(64 % kStep == 0, "N must be a multiple of kStep");
  constexpr hn::Rebind<uint8_t, decltype(int16xN_tag)> uint8_half_tag;
  const auto v_max_mask = hn::Set(int16xN_tag, 1 << WEDGE_WEIGHT_BITS);
  auto v_acc = hn::Zero(uint64xN_tag);

  for (int i = 0; i < N; i += kStep) {
    const auto r1_v = hn::LoadU(int16xN_tag, r1 + i);
    const auto d_v = hn::LoadU(int16xN_tag, d + i);
    const auto m_v =
        hn::PromoteTo(int16xN_tag, hn::LoadU(uint8_half_tag, m + i));

    const auto rd_lo = hn::InterleaveLower(d_v, r1_v);
    const auto rd_hi = hn::InterleaveUpper(int16xN_tag, d_v, r1_v);
    const auto m_lo = hn::InterleaveLower(m_v, v_max_mask);
    const auto m_hi = hn::InterleaveUpper(int16xN_tag, m_v, v_max_mask);

    const auto t = hn::ReorderDemote2To(
        int16xN_tag, hn::WidenMulPairwiseAdd(int32xN_tag, rd_lo, m_lo),
        hn::WidenMulPairwiseAdd(int32xN_tag, rd_hi, m_hi));
    const auto sq =
        hn::BitCast(uint32xN_tag, hn::WidenMulPairwiseAdd(int32xN_tag, t, t));
    v_acc = hn::Add(v_acc, hn::Add(hn::PromoteEvenTo(uint64xN_tag, sq),
                                   hn::PromoteOddTo(uint64xN_tag, sq)));
  }

  const uint64_t csse = hn::ReduceSum(uint64xN_tag, v_acc);
  return ROUND_POWER_OF_TWO(csse, 2 * WEDGE_WEIGHT_BITS);
}

HWY_ATTR HWY_INLINE int8_t WedgeSignFromResiduals(const int16_t *ds,
                                                  const uint8_t *m, int N,
                                                  int64_t limit) {
  assert(N > 0 && N % 64 == 0);
  // 32-bit per-lane accumulators; see av1_wedge_sign_from_residuals_avx2.
  assert(N < 8192);
  constexpr hn::ScalableTag<int16_t> int16xN_tag;
  constexpr hn::ScalableTag<int32_t> int32xN_tag;
  constexpr hn::ScalableTag<int64_t> int64xN_tag;
  static_assert(hn::MaxLanes(int32xN_tag) >= 4, "Requires >= 128-bit SIMD");
  constexpr int kStep = static_cast<int>(hn::MaxLanes(int16xN_tag));
  static_assert(64 % (2 * kStep) == 0, "N must be a multiple of 2 * kStep");
  constexpr hn::Rebind<uint8_t, decltype(int16xN_tag)> uint8_half_tag;

  auto sum0 = hn::Zero(int32xN_tag);
  auto sum1 = hn::Zero(int32xN_tag);
  auto sum2 = hn::Zero(int32xN_tag);
  auto sum3 = hn::Zero(int32xN_tag);

  for (int i = 0; i < N; i += 2 * kStep) {
    const auto ds0 = hn::LoadU(int16xN_tag, ds + i);
    const auto ds1 = hn::LoadU(int16xN_tag, ds + i + kStep);
    const auto m0 =
        hn::PromoteTo(int16xN_tag, hn::LoadU(uint8_half_tag, m + i));
    const auto m1 =
        hn::PromoteTo(int16xN_tag, hn::LoadU(uint8_half_tag, m + i + kStep));
    sum0 = hn::ReorderWidenMulAccumulate(int32xN_tag, ds0, m0, sum0, sum1);
    sum2 = hn::ReorderWidenMulAccumulate(int32xN_tag, ds1, m1, sum2, sum3);
  }

  hn::Vec<decltype(int64xN_tag)> acc64;
  if constexpr (hn::MaxLanes(int32xN_tag) >= 8) {
    const auto acc = hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3));
    // Widen to 64-bit before the cross-lane reduction; the total across lanes
    // can exceed INT32_MAX even though each lane cannot.
    acc64 = hn::Add(hn::PromoteLowerTo(int64xN_tag, acc),
                    hn::PromoteUpperTo(int64xN_tag, acc));
  } else {
    // On 128-bit targets (4 int32_t lanes), summing all four accumulators in
    // 32-bit can overflow INT32_MAX for N >= 4160; widen each pair first.
    const auto sum01 = hn::Add(sum0, sum1);
    const auto sum23 = hn::Add(sum2, sum3);
    const auto acc64_01 = hn::Add(hn::PromoteLowerTo(int64xN_tag, sum01),
                                  hn::PromoteUpperTo(int64xN_tag, sum01));
    const auto acc64_23 = hn::Add(hn::PromoteLowerTo(int64xN_tag, sum23),
                                  hn::PromoteUpperTo(int64xN_tag, sum23));
    acc64 = hn::Add(acc64_01, acc64_23);
  }
  const int64_t sign_sum = hn::ReduceSum(int64xN_tag, acc64);
  return sign_sum > limit;
}

HWY_ATTR HWY_INLINE void WedgeComputeDeltaSquares(int16_t *d, const int16_t *a,
                                                  const int16_t *b, int N) {
  assert(N > 0 && N % 64 == 0);
  constexpr hn::ScalableTag<int16_t> int16xN_tag;
  constexpr hn::ScalableTag<int32_t> int32xN_tag;
  constexpr int kStep = static_cast<int>(hn::MaxLanes(int16xN_tag));
  static_assert(64 % kStep == 0, "N must be a multiple of kStep");
  const auto zero = hn::Zero(int16xN_tag);

  for (int i = 0; i < N; i += kStep) {
    const auto a_v = hn::LoadU(int16xN_tag, a + i);
    const auto b_v = hn::LoadU(int16xN_tag, b + i);
    // Assumes residuals satisfy b[i] != INT16_MIN (SaturatedSub(0, -32768)
    // saturates to 32767, matching SSE2/AVX2).
    const auto nb_v = hn::SaturatedSub(zero, b_v);

    const auto r0 =
        hn::WidenMulPairwiseAdd(int32xN_tag, hn::InterleaveLower(a_v, b_v),
                                hn::InterleaveLower(a_v, nb_v));
    const auto r1 = hn::WidenMulPairwiseAdd(
        int32xN_tag, hn::InterleaveUpper(int16xN_tag, a_v, b_v),
        hn::InterleaveUpper(int16xN_tag, a_v, nb_v));

    const auto res = hn::ReorderDemote2To(int16xN_tag, r0, r1);
    hn::StoreU(res, int16xN_tag, d + i);
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace

HWY_AFTER_NAMESPACE();

#define MAKE_AV1_WEDGE_SSE_FROM_RESIDUALS(suffix)                     \
  extern "C" HWY_ATTR uint64_t av1_wedge_sse_from_residuals_##suffix( \
      const int16_t *r1, const int16_t *d, const uint8_t *m, int N) { \
    return HWY_NAMESPACE::WedgeSseFromResiduals(r1, d, m, N);         \
  }

#define MAKE_AV1_WEDGE_SIGN_FROM_RESIDUALS(suffix)                   \
  extern "C" HWY_ATTR int8_t av1_wedge_sign_from_residuals_##suffix( \
      const int16_t *ds, const uint8_t *m, int N, int64_t limit) {   \
    return HWY_NAMESPACE::WedgeSignFromResiduals(ds, m, N, limit);   \
  }

#define MAKE_AV1_WEDGE_COMPUTE_DELTA_SQUARES(suffix)                 \
  extern "C" HWY_ATTR void av1_wedge_compute_delta_squares_##suffix( \
      int16_t *d, const int16_t *a, const int16_t *b, int N) {       \
    HWY_NAMESPACE::WedgeComputeDeltaSquares(d, a, b, N);             \
  }

#endif  // AOM_AV1_ENCODER_WEDGE_UTILS_HWY_H_
