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

#ifndef AOM_AOM_DSP_AVG_HWY_H_
#define AOM_AOM_DSP_AVG_HWY_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <type_traits>

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"
#include "aom/aom_integer.h"
#include "third_party/highway/hwy/highway.h"

HWY_BEFORE_NAMESPACE();

namespace {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Sign-extends 32 int16_t values from a 512-bit register into 32 tran_low_t
// (int32_t) slots linearly.
template <class D16>
HWY_ATTR HWY_INLINE void StoreTranLow512(D16 d16, hn::VFromD<D16> a,
                                         tran_low_t *b) {
  (void)d16;
  constexpr hn::Repartition<tran_low_t, D16> d32;
  constexpr hn::Half<D16> d16_half;
  const auto lo = hn::PromoteLowerTo(d32, a);
  const auto hi = hn::PromoteTo(d32, hn::UpperHalf(d16_half, a));
  hn::StoreU(lo, d32, b);
  hn::StoreU(hi, d32, b + 16);
}

// 8-point 1D Hadamard butterfly across in[0..7] (int16_t or int32_t lanes).
template <class D>
HWY_ATTR HWY_INLINE void Hadamard8x8Butterfly(D d, hn::VFromD<D> *in) {
  (void)d;
  auto a0 = in[0];
  auto a1 = in[1];
  auto a2 = in[2];
  auto a3 = in[3];
  auto a4 = in[4];
  auto a5 = in[5];
  auto a6 = in[6];
  auto a7 = in[7];

  const auto b0 = hn::Add(a0, a1);
  const auto b1 = hn::Sub(a0, a1);
  const auto b2 = hn::Add(a2, a3);
  const auto b3 = hn::Sub(a2, a3);
  const auto b4 = hn::Add(a4, a5);
  const auto b5 = hn::Sub(a4, a5);
  const auto b6 = hn::Add(a6, a7);
  const auto b7 = hn::Sub(a6, a7);

  a0 = hn::Add(b0, b2);
  a1 = hn::Add(b1, b3);
  a2 = hn::Sub(b0, b2);
  a3 = hn::Sub(b1, b3);
  a4 = hn::Add(b4, b6);
  a5 = hn::Add(b5, b7);
  a6 = hn::Sub(b4, b6);
  a7 = hn::Sub(b5, b7);

  in[0] = hn::Add(a0, a4);
  in[7] = hn::Add(a1, a5);
  in[3] = hn::Add(a2, a6);
  in[4] = hn::Add(a3, a7);
  in[2] = hn::Sub(a0, a4);
  in[6] = hn::Sub(a1, a5);
  in[1] = hn::Sub(a2, a6);
  in[5] = hn::Sub(a3, a7);
}

// Transposes 8x8 blocks in-place across in[0..7] (for 16-bit elements per
// 128-bit lane or 32-bit elements per 256-bit half).
template <class D>
HWY_ATTR HWY_INLINE void Transpose8x8(D d, hn::VFromD<D> *in) {
  using T = hn::TFromD<D>;
  constexpr hn::RepartitionToWide<D> d_wide;
  constexpr hn::Repartition<int64_t, D> d64;

  const auto a0 = hn::InterleaveLower(d, in[0], in[1]);
  const auto a1 = hn::InterleaveLower(d, in[2], in[3]);
  const auto a2 = hn::InterleaveUpper(d, in[0], in[1]);
  const auto a3 = hn::InterleaveUpper(d, in[2], in[3]);
  const auto a4 = hn::InterleaveLower(d, in[4], in[5]);
  const auto a5 = hn::InterleaveLower(d, in[6], in[7]);
  const auto a6 = hn::InterleaveUpper(d, in[4], in[5]);
  const auto a7 = hn::InterleaveUpper(d, in[6], in[7]);

  const auto c0 =
      hn::BitCast(d64, hn::InterleaveLower(d_wide, hn::BitCast(d_wide, a0),
                                           hn::BitCast(d_wide, a1)));
  const auto c1 =
      hn::BitCast(d64, hn::InterleaveLower(d_wide, hn::BitCast(d_wide, a4),
                                           hn::BitCast(d_wide, a5)));
  const auto c2 =
      hn::BitCast(d64, hn::InterleaveUpper(d_wide, hn::BitCast(d_wide, a0),
                                           hn::BitCast(d_wide, a1)));
  const auto c3 =
      hn::BitCast(d64, hn::InterleaveUpper(d_wide, hn::BitCast(d_wide, a4),
                                           hn::BitCast(d_wide, a5)));
  const auto c4 =
      hn::BitCast(d64, hn::InterleaveLower(d_wide, hn::BitCast(d_wide, a2),
                                           hn::BitCast(d_wide, a3)));
  const auto c5 =
      hn::BitCast(d64, hn::InterleaveLower(d_wide, hn::BitCast(d_wide, a6),
                                           hn::BitCast(d_wide, a7)));
  const auto c6 =
      hn::BitCast(d64, hn::InterleaveUpper(d_wide, hn::BitCast(d_wide, a2),
                                           hn::BitCast(d_wide, a3)));
  const auto c7 =
      hn::BitCast(d64, hn::InterleaveUpper(d_wide, hn::BitCast(d_wide, a6),
                                           hn::BitCast(d_wide, a7)));

  HWY_IF_CONSTEXPR(sizeof(T) == 2) {
    in[0] = hn::BitCast(d, hn::InterleaveLower(d64, c0, c1));
    in[1] = hn::BitCast(d, hn::InterleaveUpper(d64, c0, c1));
    in[2] = hn::BitCast(d, hn::InterleaveLower(d64, c2, c3));
    in[3] = hn::BitCast(d, hn::InterleaveUpper(d64, c2, c3));
    in[4] = hn::BitCast(d, hn::InterleaveLower(d64, c4, c5));
    in[5] = hn::BitCast(d, hn::InterleaveUpper(d64, c4, c5));
    in[6] = hn::BitCast(d, hn::InterleaveLower(d64, c6, c7));
    in[7] = hn::BitCast(d, hn::InterleaveUpper(d64, c6, c7));
  }
  HWY_IF_CONSTEXPR(sizeof(T) == 4) {
    HWY_ALIGN static constexpr int64_t kIdxLo[8] = { 0, 1, 8, 9, 4, 5, 12, 13 };
    HWY_ALIGN static constexpr int64_t kIdxHi[8] = {
      2, 3, 10, 11, 6, 7, 14, 15
    };
    const auto idx_lo = hn::SetTableIndices(d64, kIdxLo);
    const auto idx_hi = hn::SetTableIndices(d64, kIdxHi);
    in[0] = hn::BitCast(d, hn::TwoTablesLookupLanes(d64, c0, c1, idx_lo));
    in[1] = hn::BitCast(d, hn::TwoTablesLookupLanes(d64, c0, c1, idx_hi));
    in[2] = hn::BitCast(d, hn::TwoTablesLookupLanes(d64, c2, c3, idx_lo));
    in[3] = hn::BitCast(d, hn::TwoTablesLookupLanes(d64, c2, c3, idx_hi));
    in[4] = hn::BitCast(d, hn::TwoTablesLookupLanes(d64, c4, c5, idx_lo));
    in[5] = hn::BitCast(d, hn::TwoTablesLookupLanes(d64, c4, c5, idx_hi));
    in[6] = hn::BitCast(d, hn::TwoTablesLookupLanes(d64, c6, c7, idx_lo));
    in[7] = hn::BitCast(d, hn::TwoTablesLookupLanes(d64, c6, c7, idx_hi));
  }
}

template <int Shift, class D>
HWY_ATTR HWY_INLINE void Hadamard2x2Butterfly(
    D d, hn::VFromD<D> c0, hn::VFromD<D> c1, hn::VFromD<D> c2, hn::VFromD<D> c3,
    hn::VFromD<D> *out0, hn::VFromD<D> *out1, hn::VFromD<D> *out2,
    hn::VFromD<D> *out3) {
  (void)d;
  const auto b0 = hn::ShiftRight<Shift>(hn::Add(c0, c1));
  const auto b1 = hn::ShiftRight<Shift>(hn::Sub(c0, c1));
  const auto b2 = hn::ShiftRight<Shift>(hn::Add(c2, c3));
  const auto b3 = hn::ShiftRight<Shift>(hn::Sub(c2, c3));

  *out0 = hn::Add(b0, b2);
  *out1 = hn::Add(b1, b3);
  *out2 = hn::Sub(b0, b2);
  *out3 = hn::Sub(b1, b3);
}

// Transposes 4x4 128-bit lanes across 4 ZMM registers so coeff0..coeff3 hold 32
// int16_t coefficients from quadrants 0..3 respectively (optionally swapping
// [4..7] with [8..11] in each 16-element group when DoShift is true to match
// aom_hadamard_16x16_c / aom_hadamard_16x16_avx2), then runs the 2x2
// inter-quadrant Hadamard butterfly in-register.
template <bool DoShift, class D16>
HWY_ATTR HWY_INLINE void Hadamard16x16QuadrantButterfly512(
    D16 d16, hn::VFromD<D16> s0, hn::VFromD<D16> s1, hn::VFromD<D16> s2,
    hn::VFromD<D16> s3, hn::VFromD<D16> *out0, hn::VFromD<D16> *out1,
    hn::VFromD<D16> *out2, hn::VFromD<D16> *out3) {
  constexpr hn::Repartition<int64_t, D16> d64;
  HWY_ALIGN static constexpr int64_t kEvenBlocks[8] = {
    0,     DoShift ? 4 : 1,         DoShift ? 1 : 4,         5,
    8 + 0, DoShift ? 8 + 4 : 8 + 1, DoShift ? 8 + 1 : 8 + 4, 8 + 5
  };
  HWY_ALIGN static constexpr int64_t kOddBlocks[8] = {
    2,     DoShift ? 6 : 3,         DoShift ? 3 : 6,         7,
    8 + 2, DoShift ? 8 + 6 : 8 + 3, DoShift ? 8 + 3 : 8 + 6, 8 + 7
  };
  const auto idx_even = hn::SetTableIndices(d64, kEvenBlocks);
  const auto idx_odd = hn::SetTableIndices(d64, kOddBlocks);

  const auto t0 = hn::BitCast(d64, hn::ConcatLowerLower(d16, s1, s0));
  const auto t1 = hn::BitCast(d64, hn::ConcatLowerLower(d16, s3, s2));
  const auto t2 = hn::BitCast(d64, hn::ConcatUpperUpper(d16, s1, s0));
  const auto t3 = hn::BitCast(d64, hn::ConcatUpperUpper(d16, s3, s2));

  const auto coeff0 =
      hn::BitCast(d16, hn::TwoTablesLookupLanes(d64, t0, t1, idx_even));
  const auto coeff1 =
      hn::BitCast(d16, hn::TwoTablesLookupLanes(d64, t0, t1, idx_odd));
  const auto coeff2 =
      hn::BitCast(d16, hn::TwoTablesLookupLanes(d64, t2, t3, idx_even));
  const auto coeff3 =
      hn::BitCast(d16, hn::TwoTablesLookupLanes(d64, t2, t3, idx_odd));

  Hadamard2x2Butterfly<1>(d16, coeff0, coeff1, coeff2, coeff3, out0, out1, out2,
                          out3);
}

template <bool DoShift, typename CoeffType>
HWY_ATTR HWY_INLINE void Hadamard16x16Impl(const int16_t *src_diff,
                                           ptrdiff_t src_stride,
                                           CoeffType *coeff) {
  constexpr hn::ScalableTag<int16_t> d16;
  constexpr hn::Half<decltype(d16)> d16_half;
  hn::VFromD<decltype(d16)> src[8];
  const int16_t *src_top = src_diff;
  const int16_t *src_bot = src_diff + 8 * src_stride;

  for (int i = 0; i < 8; ++i) {
    const auto top = hn::LoadU(d16_half, src_top);
    const auto bot = hn::LoadU(d16_half, src_bot);
    src[i] = hn::Combine(d16, bot, top);
    src_top += src_stride;
    src_bot += src_stride;
  }

  Hadamard8x8Butterfly(d16, src);
  Transpose8x8(d16, src);
  Hadamard8x8Butterfly(d16, src);

  for (int half = 0; half < 2; ++half) {
    hn::VFromD<decltype(d16)> out0, out1, out2, out3;
    const auto *s = src + 4 * half;
    CoeffType *c = coeff + 32 * half;
    Hadamard16x16QuadrantButterfly512<DoShift>(d16, s[0], s[1], s[2], s[3],
                                               &out0, &out1, &out2, &out3);
    HWY_IF_CONSTEXPR(std::is_same<CoeffType, tran_low_t>::value) {
      StoreTranLow512(d16, out0, c);
      StoreTranLow512(d16, out1, c + 64);
      StoreTranLow512(d16, out2, c + 128);
      StoreTranLow512(d16, out3, c + 192);
    }
    HWY_IF_CONSTEXPR(std::is_same<CoeffType, int16_t>::value) {
      hn::StoreU(out0, d16, c);
      hn::StoreU(out1, d16, c + 64);
      hn::StoreU(out2, d16, c + 128);
      hn::StoreU(out3, d16, c + 192);
    }
  }
}

HWY_ATTR HWY_INLINE void Hadamard16x16(const int16_t *src_diff,
                                       ptrdiff_t src_stride,
                                       tran_low_t *coeff) {
  Hadamard16x16Impl<true>(src_diff, src_stride, coeff);
}

HWY_ATTR HWY_INLINE void HadamardLp16x16(const int16_t *src_diff,
                                         ptrdiff_t src_stride, int16_t *coeff) {
  Hadamard16x16Impl<false>(src_diff, src_stride, coeff);
}

HWY_ATTR HWY_INLINE void Hadamard32x32(const int16_t *src_diff,
                                       ptrdiff_t src_stride,
                                       tran_low_t *coeff) {
  constexpr hn::ScalableTag<int32_t> d32;
  constexpr hn::Rebind<int16_t, decltype(d32)> d16_half;
  HWY_ALIGN int16_t temp_coeff[32 * 32];
  int16_t *t_coeff = temp_coeff;
  for (int idx = 0; idx < 4; ++idx) {
    const int16_t *src_ptr =
        src_diff + (idx >> 1) * 16 * src_stride + (idx & 0x01) * 16;
    Hadamard16x16Impl<true>(src_ptr, src_stride, t_coeff + idx * 256);
  }

  for (int idx = 0; idx < 256; idx += 16) {
    const auto coeff0 = hn::PromoteTo(d32, hn::Load(d16_half, t_coeff));
    const auto coeff1 = hn::PromoteTo(d32, hn::Load(d16_half, t_coeff + 256));
    const auto coeff2 = hn::PromoteTo(d32, hn::Load(d16_half, t_coeff + 512));
    const auto coeff3 = hn::PromoteTo(d32, hn::Load(d16_half, t_coeff + 768));

    hn::VFromD<decltype(d32)> out0, out1, out2, out3;
    Hadamard2x2Butterfly<2>(d32, coeff0, coeff1, coeff2, coeff3, &out0, &out1,
                            &out2, &out3);

    hn::StoreU(out0, d32, coeff);
    hn::StoreU(out1, d32, coeff + 256);
    hn::StoreU(out2, d32, coeff + 512);
    hn::StoreU(out3, d32, coeff + 768);

    coeff += 16;
    t_coeff += 16;
  }
}

#if CONFIG_AV1_HIGHBITDEPTH
// Processes two 8x8 32-bit Hadamard blocks simultaneously (block 0 in the low
// 256-bit half, block 1 in the high 256-bit half of src32[0..7]).
HWY_ATTR HWY_INLINE void HighbdHadamard8x8Dual512(const int16_t *src_diff,
                                                  ptrdiff_t src_stride,
                                                  tran_low_t *coeff) {
  constexpr hn::ScalableTag<int32_t> d32;
  constexpr hn::Rebind<int16_t, decltype(d32)> d16_half;
  hn::VFromD<decltype(d32)> src32[8];
  for (int i = 0; i < 8; ++i) {
    src32[i] = hn::PromoteTo(d32, hn::LoadU(d16_half, src_diff));
    src_diff += src_stride;
  }

  Hadamard8x8Butterfly(d32, src32);
  Transpose8x8(d32, src32);
  Hadamard8x8Butterfly(d32, src32);

  hn::StoreU(hn::ConcatLowerLower(d32, src32[1], src32[0]), d32, coeff + 0);
  hn::StoreU(hn::ConcatLowerLower(d32, src32[3], src32[2]), d32, coeff + 16);
  hn::StoreU(hn::ConcatLowerLower(d32, src32[5], src32[4]), d32, coeff + 32);
  hn::StoreU(hn::ConcatLowerLower(d32, src32[7], src32[6]), d32, coeff + 48);
  hn::StoreU(hn::ConcatUpperUpper(d32, src32[1], src32[0]), d32, coeff + 64);
  hn::StoreU(hn::ConcatUpperUpper(d32, src32[3], src32[2]), d32, coeff + 80);
  hn::StoreU(hn::ConcatUpperUpper(d32, src32[5], src32[4]), d32, coeff + 96);
  hn::StoreU(hn::ConcatUpperUpper(d32, src32[7], src32[6]), d32, coeff + 112);
}

HWY_ATTR HWY_INLINE void HighbdHadamard16x16(const int16_t *src_diff,
                                             ptrdiff_t src_stride,
                                             tran_low_t *coeff) {
  constexpr hn::ScalableTag<int32_t> d32;
  tran_low_t *t_coeff = coeff;
  HighbdHadamard8x8Dual512(src_diff, src_stride, t_coeff);
  HighbdHadamard8x8Dual512(src_diff + 8 * src_stride, src_stride,
                           t_coeff + 128);

  for (int idx = 0; idx < 64; idx += 16) {
    const auto coeff0 = hn::LoadU(d32, t_coeff);
    const auto coeff1 = hn::LoadU(d32, t_coeff + 64);
    const auto coeff2 = hn::LoadU(d32, t_coeff + 128);
    const auto coeff3 = hn::LoadU(d32, t_coeff + 192);

    hn::VFromD<decltype(d32)> out0, out1, out2, out3;
    Hadamard2x2Butterfly<1>(d32, coeff0, coeff1, coeff2, coeff3, &out0, &out1,
                            &out2, &out3);

    hn::StoreU(out0, d32, coeff);
    hn::StoreU(out1, d32, coeff + 64);
    hn::StoreU(out2, d32, coeff + 128);
    hn::StoreU(out3, d32, coeff + 192);

    coeff += 16;
    t_coeff += 16;
  }
}

HWY_ATTR HWY_INLINE void HighbdHadamard32x32(const int16_t *src_diff,
                                             ptrdiff_t src_stride,
                                             tran_low_t *coeff) {
  constexpr hn::ScalableTag<int32_t> d32;
  tran_low_t *t_coeff = coeff;
  for (int idx = 0; idx < 4; ++idx) {
    const int16_t *src_ptr =
        src_diff + (idx >> 1) * 16 * src_stride + (idx & 0x01) * 16;
    HighbdHadamard16x16(src_ptr, src_stride, t_coeff + idx * 256);
  }

  for (int idx = 0; idx < 256; idx += 16) {
    const auto coeff0 = hn::LoadU(d32, t_coeff);
    const auto coeff1 = hn::LoadU(d32, t_coeff + 256);
    const auto coeff2 = hn::LoadU(d32, t_coeff + 512);
    const auto coeff3 = hn::LoadU(d32, t_coeff + 768);

    hn::VFromD<decltype(d32)> out0, out1, out2, out3;
    Hadamard2x2Butterfly<2>(d32, coeff0, coeff1, coeff2, coeff3, &out0, &out1,
                            &out2, &out3);

    hn::StoreU(out0, d32, coeff);
    hn::StoreU(out1, d32, coeff + 256);
    hn::StoreU(out2, d32, coeff + 512);
    hn::StoreU(out3, d32, coeff + 768);

    coeff += 16;
    t_coeff += 16;
  }
}
#endif  // CONFIG_AV1_HIGHBITDEPTH

HWY_ATTR HWY_INLINE int Satd(const tran_low_t *coeff, int length) {
  constexpr hn::ScalableTag<int32_t> d32;
  constexpr int kLanes = static_cast<int>(hn::MaxLanes(d32));
  auto accum0 = hn::Zero(d32);
  auto accum1 = hn::Zero(d32);
  int i = 0;
  for (; i + 2 * kLanes <= length; i += 2 * kLanes, coeff += 2 * kLanes) {
    const auto s0 = hn::LoadU(d32, coeff);
    const auto s1 = hn::LoadU(d32, coeff + kLanes);
    accum0 = hn::Add(accum0, hn::Abs(s0));
    accum1 = hn::Add(accum1, hn::Abs(s1));
  }
  if (i < length) {
    const auto s0 = hn::LoadU(d32, coeff);
    accum0 = hn::Add(accum0, hn::Abs(s0));
  }
  accum0 = hn::Add(accum0, accum1);
  return static_cast<int>(hn::ReduceSum(d32, accum0));
}

HWY_ATTR HWY_INLINE int SatdLp(const int16_t *coeff, int length) {
  constexpr hn::ScalableTag<int16_t> d16;
  constexpr int kLanes = static_cast<int>(hn::MaxLanes(d16));
  if (length < kLanes) {
    return aom_satd_lp_avx2(coeff, length);
  }
  constexpr hn::Repartition<int32_t, decltype(d16)> d32;
  const auto one = hn::Set(d16, static_cast<int16_t>(1));
  auto accum0 = hn::Zero(d32);
  auto accum1 = hn::Zero(d32);
  auto accum2 = hn::Zero(d32);
  auto accum3 = hn::Zero(d32);
  int i = 0;
  for (; i + 2 * kLanes <= length; i += 2 * kLanes, coeff += 2 * kLanes) {
    const auto s0 = hn::Abs(hn::LoadU(d16, coeff));
    const auto s1 = hn::Abs(hn::LoadU(d16, coeff + kLanes));
    accum0 = hn::ReorderWidenMulAccumulate(d32, s0, one, accum0, accum1);
    accum2 = hn::ReorderWidenMulAccumulate(d32, s1, one, accum2, accum3);
  }
  if (i < length) {
    const auto s0 = hn::Abs(hn::LoadU(d16, coeff));
    accum0 = hn::ReorderWidenMulAccumulate(d32, s0, one, accum0, accum1);
  }
  const auto sum01 = hn::Add(accum0, accum1);
  const auto sum23 = hn::Add(accum2, accum3);
  return static_cast<int>(hn::ReduceSum(d32, hn::Add(sum01, sum23)));
}

HWY_ATTR HWY_INLINE void IntProRow(int16_t *hbuf, const uint8_t *ref,
                                   const int ref_stride, const int width,
                                   const int height, int norm_factor) {
  assert(width % 16 == 0 && height % 2 == 0);
  constexpr hn::ScalableTag<int16_t> d16;
  constexpr hn::Rebind<uint8_t, decltype(d16)> d8_half;
  int wd = 0;
  for (; wd + 64 <= width; wd += 64) {
    const uint8_t *ref_tmp = ref + wd;
    auto s0 = hn::Zero(d16);
    auto s1 = hn::Zero(d16);
    for (int idx = 0; idx < height; idx += 2) {
      const auto r0_lo = hn::LoadU(d8_half, ref_tmp);
      const auto r0_hi = hn::LoadU(d8_half, ref_tmp + 32);
      ref_tmp += ref_stride;
      const auto r1_lo = hn::LoadU(d8_half, ref_tmp);
      const auto r1_hi = hn::LoadU(d8_half, ref_tmp + 32);
      ref_tmp += ref_stride;

      s0 = hn::Add(s0, hn::PromoteTo(d16, r0_lo));
      s1 = hn::Add(s1, hn::PromoteTo(d16, r0_hi));
      s0 = hn::Add(s0, hn::PromoteTo(d16, r1_lo));
      s1 = hn::Add(s1, hn::PromoteTo(d16, r1_hi));
    }
    s0 = hn::ShiftRightSame(s0, norm_factor);
    s1 = hn::ShiftRightSame(s1, norm_factor);
    hn::StoreU(s0, d16, hbuf + wd);
    hn::StoreU(s1, d16, hbuf + wd + 32);
  }
  if (wd + 32 <= width) {
    const uint8_t *ref_tmp = ref + wd;
    auto s0 = hn::Zero(d16);
    for (int idx = 0; idx < height; idx += 2) {
      const auto r0 = hn::LoadU(d8_half, ref_tmp);
      ref_tmp += ref_stride;
      const auto r1 = hn::LoadU(d8_half, ref_tmp);
      ref_tmp += ref_stride;
      s0 = hn::Add(s0, hn::PromoteTo(d16, r0));
      s0 = hn::Add(s0, hn::PromoteTo(d16, r1));
    }
    s0 = hn::ShiftRightSame(s0, norm_factor);
    hn::StoreU(s0, d16, hbuf + wd);
    wd += 32;
  }
  if (wd < width) {
    aom_int_pro_row_sse2(hbuf + wd, ref + wd, ref_stride, width - wd, height,
                         norm_factor);
  }
}

template <class D16>
HWY_ATTR HWY_INLINE void CalcTotSadAndStore512(D16 d16, hn::VFromD<D16> r0,
                                               hn::VFromD<D16> r1,
                                               hn::VFromD<D16> r2,
                                               hn::VFromD<D16> r3,
                                               int norm_factor, int16_t *vbuf) {
  constexpr hn::Half<D16> d16_256;
  constexpr hn::Half<decltype(d16_256)> d16_128;
  constexpr hn::Half<decltype(d16_128)> d16_64;
  constexpr hn::RebindToUnsigned<decltype(d16_128)> du16_128;
  const auto r01 = hn::Add(hn::ShiftLeftBytes<2>(d16, r1), r0);
  const auto r012 = hn::Add(hn::ShiftLeftBytes<4>(d16, r2), r01);
  const auto result0 = hn::Add(hn::ShiftLeftBytes<6>(d16, r3), r012);
  const auto res256 =
      hn::Add(hn::LowerHalf(d16_256, result0), hn::UpperHalf(d16_256, result0));
  const auto res128 =
      hn::Add(hn::LowerHalf(d16_128, res256), hn::UpperHalf(d16_128, res256));
  const auto results1 =
      hn::Add(res128, hn::ShiftRightBytes<8>(d16_128, res128));
  const auto shifted = hn::BitCast(
      d16_128,
      hn::ShiftRightSame(hn::BitCast(du16_128, results1), norm_factor));
  hn::StoreU(hn::LowerHalf(d16_64, shifted), d16_64, vbuf);
}

HWY_ATTR HWY_INLINE void IntProCol(int16_t *vbuf, const uint8_t *ref,
                                   const int ref_stride, const int width,
                                   const int height, int norm_factor) {
  assert(width % 16 == 0 && height % 4 == 0);
  constexpr hn::ScalableTag<uint8_t> d8;
  constexpr hn::Repartition<int16_t, decltype(d8)> d16;
  if (width == 128) {
    const auto zero = hn::Zero(d8);
    for (int ht = 0; ht < height; ht += 4) {
      const auto s0_0 = hn::LoadU(d8, ref);
      const auto s0_1 = hn::LoadU(d8, ref + 64);
      const auto s1_0 = hn::LoadU(d8, ref + ref_stride);
      const auto s1_1 = hn::LoadU(d8, ref + ref_stride + 64);
      const auto s2_0 = hn::LoadU(d8, ref + 2 * ref_stride);
      const auto s2_1 = hn::LoadU(d8, ref + 2 * ref_stride + 64);
      const auto s3_0 = hn::LoadU(d8, ref + 3 * ref_stride);
      const auto s3_1 = hn::LoadU(d8, ref + 3 * ref_stride + 64);

      const auto r0 = hn::Add(hn::BitCast(d16, hn::SumsOf8AbsDiff(s0_0, zero)),
                              hn::BitCast(d16, hn::SumsOf8AbsDiff(s0_1, zero)));
      const auto r1 = hn::Add(hn::BitCast(d16, hn::SumsOf8AbsDiff(s1_0, zero)),
                              hn::BitCast(d16, hn::SumsOf8AbsDiff(s1_1, zero)));
      const auto r2 = hn::Add(hn::BitCast(d16, hn::SumsOf8AbsDiff(s2_0, zero)),
                              hn::BitCast(d16, hn::SumsOf8AbsDiff(s2_1, zero)));
      const auto r3 = hn::Add(hn::BitCast(d16, hn::SumsOf8AbsDiff(s3_0, zero)),
                              hn::BitCast(d16, hn::SumsOf8AbsDiff(s3_1, zero)));

      CalcTotSadAndStore512(d16, r0, r1, r2, r3, norm_factor, vbuf);
      vbuf += 4;
      ref += ref_stride << 2;
    }
  } else if (width == 64) {
    const auto zero = hn::Zero(d8);
    for (int ht = 0; ht < height; ht += 4) {
      const auto s0 = hn::LoadU(d8, ref);
      const auto s1 = hn::LoadU(d8, ref + ref_stride);
      const auto s2 = hn::LoadU(d8, ref + 2 * ref_stride);
      const auto s3 = hn::LoadU(d8, ref + 3 * ref_stride);

      const auto r0 = hn::BitCast(d16, hn::SumsOf8AbsDiff(s0, zero));
      const auto r1 = hn::BitCast(d16, hn::SumsOf8AbsDiff(s1, zero));
      const auto r2 = hn::BitCast(d16, hn::SumsOf8AbsDiff(s2, zero));
      const auto r3 = hn::BitCast(d16, hn::SumsOf8AbsDiff(s3, zero));

      CalcTotSadAndStore512(d16, r0, r1, r2, r3, norm_factor, vbuf);
      vbuf += 4;
      ref += ref_stride << 2;
    }
  } else {
    aom_int_pro_col_avx2(vbuf, ref, ref_stride, width, height, norm_factor);
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace

HWY_AFTER_NAMESPACE();

#define MAKE_HADAMARD(w, h, suffix)                                       \
  extern "C" HWY_ATTR void aom_hadamard_##w##x##h##_##suffix(             \
      const int16_t *src_diff, ptrdiff_t src_stride, tran_low_t *coeff) { \
    HWY_NAMESPACE::Hadamard##w##x##h(src_diff, src_stride, coeff);        \
  }

#define MAKE_HADAMARD_LP(w, h, suffix)                                 \
  extern "C" HWY_ATTR void aom_hadamard_lp_##w##x##h##_##suffix(       \
      const int16_t *src_diff, ptrdiff_t src_stride, int16_t *coeff) { \
    HWY_NAMESPACE::HadamardLp##w##x##h(src_diff, src_stride, coeff);   \
  }

#define MAKE_SATD_INT_PRO_FUNCS(suffix)                              \
  extern "C" HWY_ATTR int aom_satd_##suffix(const tran_low_t *coeff, \
                                            int length) {            \
    return HWY_NAMESPACE::Satd(coeff, length);                       \
  }                                                                  \
  extern "C" HWY_ATTR int aom_satd_lp_##suffix(const int16_t *coeff, \
                                               int length) {         \
    return HWY_NAMESPACE::SatdLp(coeff, length);                     \
  }                                                                  \
  extern "C" HWY_ATTR void aom_int_pro_row_##suffix(                 \
      int16_t *hbuf, const uint8_t *ref, const int ref_stride,       \
      const int width, const int height, int norm_factor) {          \
    HWY_NAMESPACE::IntProRow(hbuf, ref, ref_stride, width, height,   \
                             norm_factor);                           \
  }                                                                  \
  extern "C" HWY_ATTR void aom_int_pro_col_##suffix(                 \
      int16_t *vbuf, const uint8_t *ref, const int ref_stride,       \
      const int width, const int height, int norm_factor) {          \
    HWY_NAMESPACE::IntProCol(vbuf, ref, ref_stride, width, height,   \
                             norm_factor);                           \
  }

#if CONFIG_AV1_HIGHBITDEPTH
#define MAKE_HIGHBD_HADAMARD(w, h, suffix)                                \
  extern "C" HWY_ATTR void aom_highbd_hadamard_##w##x##h##_##suffix(      \
      const int16_t *src_diff, ptrdiff_t src_stride, tran_low_t *coeff) { \
    HWY_NAMESPACE::HighbdHadamard##w##x##h(src_diff, src_stride, coeff);  \
  }
#else
#define MAKE_HIGHBD_HADAMARD(w, h, suffix)
#endif  // CONFIG_AV1_HIGHBITDEPTH

#endif  // AOM_AOM_DSP_AVG_HWY_H_
