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

#ifndef AOM_AV1_ENCODER_ENCODETXB_HWY_H_
#define AOM_AV1_ENCODER_ENCODETXB_HWY_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"

#include "aom/aom_integer.h"
#include "av1/common/av1_common_int.h"
#include "av1/common/txb_common.h"
#include "third_party/highway/hwy/highway.h"

HWY_BEFORE_NAMESPACE();

namespace {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

constexpr hn::FixedTag<uint8_t, 8> uint8x8_tag;
constexpr hn::FixedTag<uint8_t, 16> uint8x16_tag;
constexpr hn::FixedTag<int8_t, 32> int8x32_tag;
constexpr hn::FixedTag<uint8_t, 32> uint8x32_tag;

constexpr hn::ScalableTag<int8_t> int8xN_tag;
constexpr hn::ScalableTag<uint8_t> uint8xN_tag;

using UVec8 = hn::Vec<decltype(uint8xN_tag)>;

constexpr bool kIs512Bit = hn::MaxLanes(uint8xN_tag) == 64;

// Packs 32 tran_low_t values into 32 uint8_t levels in sequential order.
HWY_ATTR HWY_INLINE hn::Vec<decltype(uint8x32_tag)> Pack32Levels(
    const tran_low_t *coeff) {
  constexpr hn::FixedTag<int32_t, 8> d32;
  constexpr hn::FixedTag<int16_t, 16> d16;
  alignas(
      32) static constexpr int32_t kPermIdx8[8] = { 0, 4, 1, 5, 2, 6, 3, 7 };

  const auto c0 = hn::Abs(hn::LoadU(d32, coeff));
  const auto c1 = hn::Abs(hn::LoadU(d32, coeff + 8));
  const auto c2 = hn::Abs(hn::LoadU(d32, coeff + 16));
  const auto c3 = hn::Abs(hn::LoadU(d32, coeff + 24));

  const auto p01 = hn::ReorderDemote2To(d16, c0, c1);
  const auto p23 = hn::ReorderDemote2To(d16, c2, c3);
  const auto p0123 = hn::ReorderDemote2To(int8x32_tag, p01, p23);
  const auto idx = hn::SetTableIndices(d32, kPermIdx8);
  return hn::BitCast(uint8x32_tag,
                     hn::TableLookupLanes(hn::BitCast(d32, p0123), idx));
}

// Packs 64 tran_low_t values into two 32-byte uint8_t vectors in sequential
// order.
template <typename D8 = decltype(uint8xN_tag)>
HWY_ATTR HWY_INLINE void Pack64LevelsTo2x32(
    const tran_low_t *coeff, hn::Vec<decltype(uint8x32_tag)> *lv_lo,
    hn::Vec<decltype(uint8x32_tag)> *lv_hi) {
  constexpr D8 d8;
  if constexpr (hn::MaxLanes(d8) == 64) {
    constexpr hn::RebindToSigned<D8> di8;
    constexpr hn::Repartition<int16_t, D8> di16;
    constexpr hn::Repartition<int32_t, D8> di32;
    constexpr hn::Half<D8> d8_half;
    alignas(64) static constexpr int32_t kPermIdx16[16] = {
      0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15
    };
    const auto c0 = hn::Abs(hn::LoadU(di32, coeff));
    const auto c1 = hn::Abs(hn::LoadU(di32, coeff + 16));
    const auto c2 = hn::Abs(hn::LoadU(di32, coeff + 32));
    const auto c3 = hn::Abs(hn::LoadU(di32, coeff + 48));

    const auto p01 = hn::ReorderDemote2To(di16, c0, c1);
    const auto p23 = hn::ReorderDemote2To(di16, c2, c3);
    const auto p0123 = hn::ReorderDemote2To(di8, p01, p23);
    const auto idx = hn::SetTableIndices(di32, kPermIdx16);
    const auto lv =
        hn::BitCast(d8, hn::TableLookupLanes(hn::BitCast(di32, p0123), idx));
    *lv_lo = hn::LowerHalf(d8_half, lv);
    *lv_hi = hn::UpperHalf(d8_half, lv);
  } else {
    *lv_lo = Pack32Levels(coeff);
    *lv_hi = Pack32Levels(coeff + 32);
  }
}

template <typename D>
HWY_ATTR HWY_INLINE void StoreRowWithPad(D d, hn::Vec<D> row, uint8_t *dst) {
  hn::StoreU(row, d, dst);
  memset(dst + hn::MaxLanes(d), 0, TX_PAD_HOR);
}

HWY_ATTR HWY_INLINE void TxbInitLevels(const tran_low_t *const coeff,
                                       const int width, const int height,
                                       uint8_t *const levels) {
  static_assert(TX_PAD_HOR == 4 && TX_PAD_BOTTOM == 4, "Unexpected TX_PAD");
  if (width * height < 64 || height == 4) {
    av1_txb_init_levels_avx2(coeff, width, height, levels);
    return;
  }

  uint8_t *ls = levels;
  const tran_low_t *cf = coeff;

  if (height == 32) {
    assert(width > 0 && width % 2 == 0);
    constexpr ptrdiff_t kStride = 32 + TX_PAD_HOR;
    uint8_t *const ls_end = levels + width * kStride;
    do {
      hn::Vec<decltype(uint8x32_tag)> lv_lo, lv_hi;
      Pack64LevelsTo2x32(cf, &lv_lo, &lv_hi);
      StoreRowWithPad(uint8x32_tag, lv_lo, ls);
      StoreRowWithPad(uint8x32_tag, lv_hi, ls + kStride);
      ls += 2 * kStride;
      cf += 64;
    } while (ls < ls_end);
    // Zero TX_PAD_BOTTOM * kStride (4 * 36 = 144) trailing bytes.
    if constexpr (kIs512Bit) {
      const auto zero64 = hn::Zero(uint8xN_tag);
      hn::StoreU(zero64, uint8xN_tag, ls);
      hn::StoreU(zero64, uint8xN_tag, ls + 64);
    } else {
      const auto zero32 = hn::Zero(uint8x32_tag);
      hn::StoreU(zero32, uint8x32_tag, ls);
      hn::StoreU(zero32, uint8x32_tag, ls + 32);
      hn::StoreU(zero32, uint8x32_tag, ls + 64);
      hn::StoreU(zero32, uint8x32_tag, ls + 96);
    }
    const auto zero16 = hn::Zero(uint8x16_tag);
    hn::StoreU(zero16, uint8x16_tag, ls + 128);
  } else if (height == 16) {
    assert(width > 0 && width % 4 == 0);
    constexpr ptrdiff_t kStride = 16 + TX_PAD_HOR;
    uint8_t *const ls_end = levels + width * kStride;
    do {
      hn::Vec<decltype(uint8x32_tag)> lv_lo, lv_hi;
      Pack64LevelsTo2x32(cf, &lv_lo, &lv_hi);
      StoreRowWithPad(uint8x16_tag, hn::LowerHalf(uint8x16_tag, lv_lo),
                      ls + 0 * kStride);
      StoreRowWithPad(uint8x16_tag, hn::UpperHalf(uint8x16_tag, lv_lo),
                      ls + 1 * kStride);
      StoreRowWithPad(uint8x16_tag, hn::LowerHalf(uint8x16_tag, lv_hi),
                      ls + 2 * kStride);
      StoreRowWithPad(uint8x16_tag, hn::UpperHalf(uint8x16_tag, lv_hi),
                      ls + 3 * kStride);
      ls += 4 * kStride;
      cf += 64;
    } while (ls < ls_end);
    // Zero TX_PAD_BOTTOM * kStride (4 * 20 = 80) trailing bytes.
    if constexpr (kIs512Bit) {
      hn::StoreU(hn::Zero(uint8xN_tag), uint8xN_tag, ls);
    } else {
      const auto zero32 = hn::Zero(uint8x32_tag);
      hn::StoreU(zero32, uint8x32_tag, ls);
      hn::StoreU(zero32, uint8x32_tag, ls + 32);
    }
    const auto zero16 = hn::Zero(uint8x16_tag);
    hn::StoreU(zero16, uint8x16_tag, ls + 64);
  } else if (height == 8) {
    assert(width > 0 && width % 8 == 0);
    constexpr ptrdiff_t kStride = 8 + TX_PAD_HOR;
    uint8_t *const ls_end = levels + width * kStride;
    do {
      hn::Vec<decltype(uint8x32_tag)> lv_lo, lv_hi;
      Pack64LevelsTo2x32(cf, &lv_lo, &lv_hi);
      const auto r01 = hn::LowerHalf(uint8x16_tag, lv_lo);
      const auto r23 = hn::UpperHalf(uint8x16_tag, lv_lo);
      const auto r45 = hn::LowerHalf(uint8x16_tag, lv_hi);
      const auto r67 = hn::UpperHalf(uint8x16_tag, lv_hi);
      StoreRowWithPad(uint8x8_tag, hn::LowerHalf(uint8x8_tag, r01),
                      ls + 0 * kStride);
      StoreRowWithPad(uint8x8_tag, hn::UpperHalf(uint8x8_tag, r01),
                      ls + 1 * kStride);
      StoreRowWithPad(uint8x8_tag, hn::LowerHalf(uint8x8_tag, r23),
                      ls + 2 * kStride);
      StoreRowWithPad(uint8x8_tag, hn::UpperHalf(uint8x8_tag, r23),
                      ls + 3 * kStride);
      StoreRowWithPad(uint8x8_tag, hn::LowerHalf(uint8x8_tag, r45),
                      ls + 4 * kStride);
      StoreRowWithPad(uint8x8_tag, hn::UpperHalf(uint8x8_tag, r45),
                      ls + 5 * kStride);
      StoreRowWithPad(uint8x8_tag, hn::LowerHalf(uint8x8_tag, r67),
                      ls + 6 * kStride);
      StoreRowWithPad(uint8x8_tag, hn::UpperHalf(uint8x8_tag, r67),
                      ls + 7 * kStride);
      ls += 8 * kStride;
      cf += 64;
    } while (ls < ls_end);
    // Zero TX_PAD_BOTTOM * kStride (4 * 12 = 48) trailing bytes.
    const auto zero32 = hn::Zero(uint8x32_tag);
    const auto zero16 = hn::Zero(uint8x16_tag);
    hn::StoreU(zero32, uint8x32_tag, ls);
    hn::StoreU(zero16, uint8x16_tag, ls + 32);
  } else {
    av1_txb_init_levels_avx2(coeff, width, height, levels);
  }
}

// -----------------------------------------------------------------------------
// av1_get_nz_map_contexts
// -----------------------------------------------------------------------------

// A 64-byte block of uint8_t represented as either 1 Scalable vector (on
// 512-bit targets) or 2 256-bit vectors.
template <bool kUse512>
struct Vec64Impl {
  hn::Vec<decltype(uint8x32_tag)> lo;
  hn::Vec<decltype(uint8x32_tag)> hi;
};

template <>
struct Vec64Impl<true> {
  UVec8 v;
};

using Vec64 = Vec64Impl<kIs512Bit>;

template <bool kUse512 = kIs512Bit>
HWY_ATTR HWY_INLINE Vec64Impl<kUse512> Set1_64(uint8_t val) {
  Vec64Impl<kUse512> res;
  if constexpr (kUse512) {
    res.v = hn::Set(uint8xN_tag, val);
  } else {
    res.lo = hn::Set(uint8x32_tag, val);
    res.hi = res.lo;
  }
  return res;
}

template <bool kUse512 = kIs512Bit>
HWY_ATTR HWY_INLINE Vec64Impl<kUse512> Broadcast128To64(
    hn::Vec<decltype(uint8x16_tag)> a) {
  Vec64Impl<kUse512> res;
  if constexpr (kUse512) {
    res.v = hn::BroadcastBlock<0>(hn::ResizeBitCast(uint8xN_tag, a));
  } else {
    res.lo = hn::Combine(uint8x32_tag, a, a);
    res.hi = res.lo;
  }
  return res;
}

template <bool kUse512 = kIs512Bit>
HWY_ATTR HWY_INLINE Vec64Impl<kUse512> Pack4x128To64(
    hn::Vec<decltype(uint8x16_tag)> a, hn::Vec<decltype(uint8x16_tag)> b,
    hn::Vec<decltype(uint8x16_tag)> c, hn::Vec<decltype(uint8x16_tag)> d) {
  Vec64Impl<kUse512> res;
  if constexpr (kUse512) {
    res.v = hn::InsertBlock<3>(
        hn::InsertBlock<2>(
            hn::InsertBlock<1>(hn::ResizeBitCast(uint8xN_tag, a), b), c),
        d);
  } else {
    res.lo = hn::Combine(uint8x32_tag, b, a);
    res.hi = hn::Combine(uint8x32_tag, d, c);
  }
  return res;
}

template <bool kUse512 = kIs512Bit>
HWY_ATTR HWY_INLINE Vec64Impl<kUse512> Add64(Vec64Impl<kUse512> a,
                                             Vec64Impl<kUse512> b) {
  Vec64Impl<kUse512> res;
  if constexpr (kUse512) {
    res.v = hn::Add(a.v, b.v);
  } else {
    res.lo = hn::Add(a.lo, b.lo);
    res.hi = hn::Add(a.hi, b.hi);
  }
  return res;
}

template <bool kUse512 = kIs512Bit>
HWY_ATTR HWY_INLINE void StoreU64(Vec64Impl<kUse512> a, int8_t *dst) {
  if constexpr (kUse512) {
    hn::StoreU(hn::BitCast(int8xN_tag, a.v), int8xN_tag, dst);
  } else {
    hn::StoreU(hn::BitCast(int8x32_tag, a.lo), int8x32_tag, dst);
    hn::StoreU(hn::BitCast(int8x32_tag, a.hi), int8x32_tag, dst + 32);
  }
}

HWY_ATTR HWY_INLINE hn::Vec<decltype(uint8x16_tag)> Load8bit8x2To128(
    const uint8_t *const src, const ptrdiff_t stride) {
  const auto r0 = hn::LoadU(uint8x8_tag, src + 0 * stride);
  const auto r1 = hn::LoadU(uint8x8_tag, src + 1 * stride);
  return hn::Combine(uint8x16_tag, r1, r0);
}

template <int kHeight, bool kUse512 = kIs512Bit>
HWY_ATTR HWY_INLINE Vec64Impl<kUse512> LoadLevels64(const uint8_t *const src) {
  static_assert(kHeight == 8 || kHeight == 16 || kHeight == 32,
                "Unsupported height");
  constexpr ptrdiff_t kStride = kHeight + TX_PAD_HOR;
  if constexpr (kHeight == 8) {
    const auto r01 = Load8bit8x2To128(src + 0 * kStride, kStride);
    const auto r23 = Load8bit8x2To128(src + 2 * kStride, kStride);
    const auto r45 = Load8bit8x2To128(src + 4 * kStride, kStride);
    const auto r67 = Load8bit8x2To128(src + 6 * kStride, kStride);
    return Pack4x128To64<kUse512>(r01, r23, r45, r67);
  } else if constexpr (kHeight == 16) {
    const auto r0 = hn::LoadU(uint8x16_tag, src + 0 * kStride);
    const auto r1 = hn::LoadU(uint8x16_tag, src + 1 * kStride);
    const auto r2 = hn::LoadU(uint8x16_tag, src + 2 * kStride);
    const auto r3 = hn::LoadU(uint8x16_tag, src + 3 * kStride);
    return Pack4x128To64<kUse512>(r0, r1, r2, r3);
  } else {
    const auto r0 = hn::LoadU(uint8x32_tag, src + 0 * kStride);
    const auto r1 = hn::LoadU(uint8x32_tag, src + 1 * kStride);
    Vec64Impl<kUse512> res;
    if constexpr (kUse512) {
      res.v =
          hn::ConcatLowerLower(uint8xN_tag, hn::ResizeBitCast(uint8xN_tag, r1),
                               hn::ResizeBitCast(uint8xN_tag, r0));
    } else {
      res.lo = r0;
      res.hi = r1;
    }
    return res;
  }
}

template <int kHeight, TX_CLASS kTxClass>
HWY_ATTR HWY_INLINE void LoadLevels5x64(const uint8_t *const src,
                                        Vec64 *const level) {
  static_assert(kTxClass == TX_CLASS_2D || kTxClass == TX_CLASS_HORIZ ||
                    kTxClass == TX_CLASS_VERT,
                "Unsupported tx_class");
  constexpr ptrdiff_t kStride = kHeight + TX_PAD_HOR;
  level[0] = LoadLevels64<kHeight>(src + 1);
  level[1] = LoadLevels64<kHeight>(src + kStride);
  if constexpr (kTxClass == TX_CLASS_2D) {
    level[2] = LoadLevels64<kHeight>(src + 2);
    level[3] = LoadLevels64<kHeight>(src + kStride + 1);
    level[4] = LoadLevels64<kHeight>(src + 2 * kStride);
  } else if constexpr (kTxClass == TX_CLASS_HORIZ) {
    level[2] = LoadLevels64<kHeight>(src + 2 * kStride);
    level[3] = LoadLevels64<kHeight>(src + 3 * kStride);
    level[4] = LoadLevels64<kHeight>(src + 4 * kStride);
  } else {
    level[2] = LoadLevels64<kHeight>(src + 2);
    level[3] = LoadLevels64<kHeight>(src + 3);
    level[4] = LoadLevels64<kHeight>(src + 4);
  }
}

template <typename D>
HWY_ATTR HWY_INLINE hn::Vec<D> GetCoeffContextsKernelVec(D d, hn::Vec<D> l0,
                                                         hn::Vec<D> l1,
                                                         hn::Vec<D> l2,
                                                         hn::Vec<D> l3,
                                                         hn::Vec<D> l4) {
  const auto const_3 = hn::Set(d, 3);
  const auto const_4 = hn::Set(d, 4);
  auto count = hn::Min(l0, const_3);
  count = hn::Add(count, hn::Min(l1, const_3));
  count = hn::Add(count, hn::Min(l2, const_3));
  count = hn::Add(count, hn::Min(l3, const_3));
  count = hn::Add(count, hn::Min(l4, const_3));
  count = hn::AverageRound(count, hn::Zero(d));
  return hn::Min(count, const_4);
}

template <bool kUse512 = kIs512Bit>
HWY_ATTR HWY_INLINE Vec64Impl<kUse512> GetCoeffContextsKernel(
    const Vec64Impl<kUse512> *const level) {
  Vec64Impl<kUse512> res;
  if constexpr (kUse512) {
    res.v = GetCoeffContextsKernelVec(uint8xN_tag, level[0].v, level[1].v,
                                      level[2].v, level[3].v, level[4].v);
  } else {
    res.lo = GetCoeffContextsKernelVec(uint8x32_tag, level[0].lo, level[1].lo,
                                       level[2].lo, level[3].lo, level[4].lo);
    res.hi = GetCoeffContextsKernelVec(uint8x32_tag, level[0].hi, level[1].hi,
                                       level[2].hi, level[3].hi, level[4].hi);
  }
  return res;
}

// Shared 2D position-to-offset tables for height == 16 and height == 32.
alignas(16) constexpr uint8_t kPosToOffset2DEq[4][16] = {
  { 0, 1, 6, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21 },
  { 1, 6, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21 },
  { 6, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21 },
  { 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21 },
};

alignas(16) constexpr uint8_t kPosToOffset2DLt[3][16] = {
  { 0, 11, 6, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21 },
  { 11, 11, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21 },
  { 11, 11, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21 },
};

// Shared vertical position-to-offset table for height == 16 and height == 32.
alignas(16) constexpr uint8_t kPosToOffsetVer16[16] = {
  SIG_COEF_CONTEXTS_2D + 0,  SIG_COEF_CONTEXTS_2D + 5,
  SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
  SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
  SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
  SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
  SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
  SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
  SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
};

HWY_ATTR HWY_INLINE void Get8CoeffContexts2D(const uint8_t *levels,
                                             const int width,
                                             int8_t *coeff_contexts) {
  constexpr int kColsPerStep = 8;
  assert(width > 0 && width % kColsPerStep == 0);
  constexpr ptrdiff_t kStride = 8 + TX_PAD_HOR;
  int8_t *cc = coeff_contexts;
  int col = width;
  Vec64 level[5];

  alignas(16) static constexpr uint8_t kP0Eq8[16] = { 0,  1,  6,  6, 21, 21,
                                                      21, 21, 1,  6, 6,  21,
                                                      21, 21, 21, 21 };
  alignas(16) static constexpr uint8_t kP0Ne8[16] = { 0,  16, 16, 16, 16, 16,
                                                      16, 16, 16, 16, 16, 16,
                                                      16, 16, 16, 16 };
  alignas(16) static constexpr uint8_t kP1[16] = { 6,  6,  21, 21, 21, 21,
                                                   21, 21, 6,  21, 21, 21,
                                                   21, 21, 21, 21 };

  const auto p0 = hn::Load(uint8x16_tag, (width == 8) ? kP0Eq8 : kP0Ne8);
  const auto p1 = hn::Load(uint8x16_tag, kP1);
  const auto p2 = hn::Set(uint8x16_tag, 21);

  Vec64 pos_to_offset = Pack4x128To64(p0, p1, p2, p2);
  const Vec64 pos_to_offset_large = Set1_64(21);

  do {
    LoadLevels5x64<8, TX_CLASS_2D>(levels, level);
    Vec64 count = GetCoeffContextsKernel(level);
    count = Add64(count, pos_to_offset);
    StoreU64(count, cc);
    pos_to_offset = pos_to_offset_large;
    levels += kColsPerStep * kStride;
    cc += 64;
    col -= kColsPerStep;
  } while (col > 0);

  coeff_contexts[0] = 0;
}

HWY_ATTR HWY_INLINE void Get16CoeffContexts2D(const uint8_t *levels,
                                              const int real_width,
                                              const int real_height,
                                              const int width,
                                              int8_t *coeff_contexts) {
  constexpr int kColsPerStep = 4;
  assert(width > 0 && width % kColsPerStep == 0);
  constexpr ptrdiff_t kStride = 16 + TX_PAD_HOR;
  int8_t *cc = coeff_contexts;
  int col = width;
  Vec64 level[5];
  Vec64 pos_to_offset;
  Vec64 pos_to_offset_large;

  if (real_width == real_height) {
    const auto p0 = hn::Load(uint8x16_tag, kPosToOffset2DEq[0]);
    const auto p1 = hn::Load(uint8x16_tag, kPosToOffset2DEq[1]);
    const auto p2 = hn::Load(uint8x16_tag, kPosToOffset2DEq[2]);
    const auto p3 = hn::Load(uint8x16_tag, kPosToOffset2DEq[3]);
    pos_to_offset = Pack4x128To64(p0, p1, p2, p3);
    pos_to_offset_large = Set1_64(21);
  } else if (real_width < real_height) {
    const auto p0 = hn::Load(uint8x16_tag, kPosToOffset2DLt[0]);
    const auto p1 = hn::Load(uint8x16_tag, kPosToOffset2DLt[1]);
    const auto p2 = hn::Load(uint8x16_tag, kPosToOffset2DLt[2]);
    pos_to_offset = Pack4x128To64(p0, p1, p2, p2);
    pos_to_offset_large = Broadcast128To64(p2);
  } else {
    const auto p0 = hn::Set(uint8x16_tag, 16);
    const auto p2 = hn::Load(uint8x16_tag, kPosToOffset2DEq[2]);
    const auto p3 = hn::Load(uint8x16_tag, kPosToOffset2DEq[3]);
    pos_to_offset = Pack4x128To64(p0, p0, p2, p3);
    pos_to_offset_large = Set1_64(21);
  }

  do {
    LoadLevels5x64<16, TX_CLASS_2D>(levels, level);
    Vec64 count = GetCoeffContextsKernel(level);
    count = Add64(count, pos_to_offset);
    StoreU64(count, cc);
    pos_to_offset = pos_to_offset_large;
    levels += kColsPerStep * kStride;
    cc += 64;
    col -= kColsPerStep;
  } while (col > 0);

  coeff_contexts[0] = 0;
}

HWY_ATTR HWY_INLINE void Get32CoeffContexts2D(const uint8_t *levels,
                                              const int real_width,
                                              const int real_height,
                                              const int width,
                                              int8_t *coeff_contexts) {
  constexpr int kColsPerStep = 2;
  assert(width > 0 && width % kColsPerStep == 0);
  constexpr ptrdiff_t kStride = 32 + TX_PAD_HOR;
  int8_t *cc = coeff_contexts;
  int col = width;
  Vec64 level[5];
  Vec64 pos_to_offset[3];
  const auto p_21 = hn::Set(uint8x16_tag, 21);

  if (real_width == real_height) {
    const auto p0 = hn::Load(uint8x16_tag, kPosToOffset2DEq[0]);
    const auto p1 = hn::Load(uint8x16_tag, kPosToOffset2DEq[1]);
    const auto p2 = hn::Load(uint8x16_tag, kPosToOffset2DEq[2]);
    const auto p3 = hn::Load(uint8x16_tag, kPosToOffset2DEq[3]);
    pos_to_offset[0] = Pack4x128To64(p0, p_21, p1, p_21);
    pos_to_offset[1] = Pack4x128To64(p2, p_21, p3, p_21);
    pos_to_offset[2] = Set1_64(21);
  } else if (real_width < real_height) {
    const auto p0 = hn::Load(uint8x16_tag, kPosToOffset2DLt[0]);
    const auto p1 = hn::Load(uint8x16_tag, kPosToOffset2DLt[1]);
    const auto p2 = hn::Load(uint8x16_tag, kPosToOffset2DLt[2]);
    pos_to_offset[0] = Pack4x128To64(p0, p_21, p1, p_21);
    pos_to_offset[1] = pos_to_offset[2] = Pack4x128To64(p2, p_21, p2, p_21);
  } else {
    const auto p2 = hn::Load(uint8x16_tag, kPosToOffset2DEq[2]);
    const auto p3 = hn::Load(uint8x16_tag, kPosToOffset2DEq[3]);
    pos_to_offset[0] = Set1_64(16);
    pos_to_offset[1] = Pack4x128To64(p2, p_21, p3, p_21);
    pos_to_offset[2] = Set1_64(21);
  }

  do {
    LoadLevels5x64<32, TX_CLASS_2D>(levels, level);
    Vec64 count = GetCoeffContextsKernel(level);
    count = Add64(count, pos_to_offset[0]);
    StoreU64(count, cc);
    pos_to_offset[0] = pos_to_offset[1];
    pos_to_offset[1] = pos_to_offset[2];
    levels += kColsPerStep * kStride;
    cc += 64;
    col -= kColsPerStep;
  } while (col > 0);

  coeff_contexts[0] = 0;
}

template <int kHeight>
HWY_ATTR HWY_INLINE void GetCoeffContextsVer(const uint8_t *levels,
                                             const int width,
                                             int8_t *coeff_contexts) {
  static_assert(kHeight == 8 || kHeight == 16 || kHeight == 32,
                "Unsupported height");
  constexpr int kColsPerStep = 64 / kHeight;
  assert(width > 0 && width % kColsPerStep == 0);
  constexpr ptrdiff_t kStride = kHeight + TX_PAD_HOR;
  Vec64 pos_to_offset;

  if constexpr (kHeight == 8) {
    alignas(16) static constexpr uint8_t kP0[16] = {
      SIG_COEF_CONTEXTS_2D + 0,  SIG_COEF_CONTEXTS_2D + 5,
      SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
      SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
      SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
      SIG_COEF_CONTEXTS_2D + 0,  SIG_COEF_CONTEXTS_2D + 5,
      SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
      SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
      SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
    };
    pos_to_offset = Broadcast128To64(hn::Load(uint8x16_tag, kP0));
  } else if constexpr (kHeight == 16) {
    pos_to_offset = Broadcast128To64(hn::Load(uint8x16_tag, kPosToOffsetVer16));
  } else {
    const auto p0 = hn::Load(uint8x16_tag, kPosToOffsetVer16);
    const auto p1 = hn::Set(uint8x16_tag, SIG_COEF_CONTEXTS_2D + 10);
    pos_to_offset = Pack4x128To64(p0, p1, p0, p1);
  }

  int col = width;
  Vec64 level[5];
  do {
    LoadLevels5x64<kHeight, TX_CLASS_VERT>(levels, level);
    Vec64 count = GetCoeffContextsKernel(level);
    count = Add64(count, pos_to_offset);
    StoreU64(count, coeff_contexts);
    levels += kColsPerStep * kStride;
    coeff_contexts += 64;
    col -= kColsPerStep;
  } while (col > 0);
}

template <int kHeight>
HWY_ATTR HWY_INLINE void GetCoeffContextsHor(const uint8_t *levels,
                                             const int width,
                                             int8_t *coeff_contexts) {
  static_assert(kHeight == 8 || kHeight == 16 || kHeight == 32,
                "Unsupported height");
  constexpr int kColsPerStep = 64 / kHeight;
  assert(width > 0 && width % kColsPerStep == 0);
  constexpr ptrdiff_t kStride = kHeight + TX_PAD_HOR;
  Vec64 pos_to_offset;

  if constexpr (kHeight == 8) {
    alignas(16) static constexpr uint8_t kP0[16] = {
      SIG_COEF_CONTEXTS_2D + 0, SIG_COEF_CONTEXTS_2D + 0,
      SIG_COEF_CONTEXTS_2D + 0, SIG_COEF_CONTEXTS_2D + 0,
      SIG_COEF_CONTEXTS_2D + 0, SIG_COEF_CONTEXTS_2D + 0,
      SIG_COEF_CONTEXTS_2D + 0, SIG_COEF_CONTEXTS_2D + 0,
      SIG_COEF_CONTEXTS_2D + 5, SIG_COEF_CONTEXTS_2D + 5,
      SIG_COEF_CONTEXTS_2D + 5, SIG_COEF_CONTEXTS_2D + 5,
      SIG_COEF_CONTEXTS_2D + 5, SIG_COEF_CONTEXTS_2D + 5,
      SIG_COEF_CONTEXTS_2D + 5, SIG_COEF_CONTEXTS_2D + 5,
    };
    const auto p0 = hn::Load(uint8x16_tag, kP0);
    const auto p_large = hn::Set(uint8x16_tag, SIG_COEF_CONTEXTS_2D + 10);
    pos_to_offset = Pack4x128To64(p0, p_large, p_large, p_large);
  } else if constexpr (kHeight == 16) {
    const auto p0 = hn::Set(uint8x16_tag, SIG_COEF_CONTEXTS_2D + 0);
    const auto p1 = hn::Set(uint8x16_tag, SIG_COEF_CONTEXTS_2D + 5);
    const auto p2 = hn::Set(uint8x16_tag, SIG_COEF_CONTEXTS_2D + 10);
    pos_to_offset = Pack4x128To64(p0, p1, p2, p2);
  } else {
    const auto p0 = hn::Set(uint8x16_tag, SIG_COEF_CONTEXTS_2D + 0);
    const auto p1 = hn::Set(uint8x16_tag, SIG_COEF_CONTEXTS_2D + 5);
    pos_to_offset = Pack4x128To64(p0, p0, p1, p1);
  }

  const Vec64 pos_to_offset_large = Set1_64(SIG_COEF_CONTEXTS_2D + 10);
  int col = width;
  Vec64 level[5];
  do {
    LoadLevels5x64<kHeight, TX_CLASS_HORIZ>(levels, level);
    Vec64 count = GetCoeffContextsKernel(level);
    count = Add64(count, pos_to_offset);
    StoreU64(count, coeff_contexts);
    pos_to_offset = pos_to_offset_large;
    levels += kColsPerStep * kStride;
    coeff_contexts += 64;
    col -= kColsPerStep;
  } while (col > 0);
}

HWY_ATTR HWY_INLINE void GetNzMapContexts(const uint8_t *const levels,
                                          const int16_t *const scan,
                                          const int eob, const TX_SIZE tx_size,
                                          const TX_CLASS tx_class,
                                          int8_t *const coeff_contexts) {
  assert(eob > 0);
  const int last_idx = eob - 1;
  if (!last_idx) {
    coeff_contexts[0] = 0;
    return;
  }

  /* coeff_contexts must be 16 byte aligned. */
  assert(!(reinterpret_cast<intptr_t>(coeff_contexts) & 0xf));

  const int width = get_txb_wide(tx_size);
  const int height = get_txb_high(tx_size);
  if (height < 8 || (height == 8 && width < 8)) {
    av1_get_nz_map_contexts_sse2(levels, scan, eob, tx_size, tx_class,
                                 coeff_contexts);
    return;
  }

  if (tx_class == TX_CLASS_2D) {
    const int real_width = tx_size_wide[tx_size];
    const int real_height = tx_size_high[tx_size];
    if (height == 8) {
      Get8CoeffContexts2D(levels, width, coeff_contexts);
    } else if (height == 16) {
      Get16CoeffContexts2D(levels, real_width, real_height, width,
                           coeff_contexts);
    } else {
      assert(height == 32);
      Get32CoeffContexts2D(levels, real_width, real_height, width,
                           coeff_contexts);
    }
  } else if (tx_class == TX_CLASS_HORIZ) {
    if (height == 8) {
      GetCoeffContextsHor<8>(levels, width, coeff_contexts);
    } else if (height == 16) {
      GetCoeffContextsHor<16>(levels, width, coeff_contexts);
    } else {
      assert(height == 32);
      GetCoeffContextsHor<32>(levels, width, coeff_contexts);
    }
  } else {
    assert(tx_class == TX_CLASS_VERT);
    if (height == 8) {
      GetCoeffContextsVer<8>(levels, width, coeff_contexts);
    } else if (height == 16) {
      GetCoeffContextsVer<16>(levels, width, coeff_contexts);
    } else {
      assert(height == 32);
      GetCoeffContextsVer<32>(levels, width, coeff_contexts);
    }
  }

  const int bhl = get_txb_bhl(tx_size);
  const int pos = scan[last_idx];
  if (last_idx <= (width << bhl) / 8)
    coeff_contexts[pos] = 1;
  else if (last_idx <= (width << bhl) / 4)
    coeff_contexts[pos] = 2;
  else
    coeff_contexts[pos] = 3;
}

}  // namespace HWY_NAMESPACE
}  // namespace

HWY_AFTER_NAMESPACE();

#define MAKE_AV1_TXB_INIT_LEVELS(suffix)                                \
  extern "C" HWY_ATTR void av1_txb_init_levels_##suffix(                \
      const tran_low_t *const coeff, const int width, const int height, \
      uint8_t *const levels) {                                          \
    HWY_NAMESPACE::TxbInitLevels(coeff, width, height, levels);         \
  }

#define MAKE_AV1_GET_NZ_MAP_CONTEXTS(suffix)                                 \
  extern "C" HWY_ATTR void av1_get_nz_map_contexts_##suffix(                 \
      const uint8_t *const levels, const int16_t *const scan, const int eob, \
      const TX_SIZE tx_size, const TX_CLASS tx_class,                        \
      int8_t *const coeff_contexts) {                                        \
    HWY_NAMESPACE::GetNzMapContexts(levels, scan, eob, tx_size, tx_class,    \
                                    coeff_contexts);                         \
  }

#endif  // AOM_AV1_ENCODER_ENCODETXB_HWY_H_
