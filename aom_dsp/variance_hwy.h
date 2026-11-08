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

#ifndef AOM_AOM_DSP_VARIANCE_HWY_H_
#define AOM_AOM_DSP_VARIANCE_HWY_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"

#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/blend.h"
#include "aom_dsp/reduce_sum_hwy.h"
#include "third_party/highway/hwy/highway.h"

HWY_BEFORE_NAMESPACE();

namespace {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

template <class D, class VHalf>
HWY_ATTR HWY_INLINE hn::VFromD<D> Combine2Rows(D d, VHalf r0, VHalf r1) {
  HWY_IF_CONSTEXPR(hn::MaxLanes(d) * sizeof(hn::TFromD<D>) >= 32) {
    return hn::ConcatLowerLower(d, hn::ResizeBitCast(d, r1),
                                hn::ResizeBitCast(d, r0));
  }
  return hn::Combine(d, r1, r0);
}

template <class D, class VQuarter>
HWY_ATTR HWY_INLINE hn::VFromD<D> Combine4Rows(D d, VQuarter r0, VQuarter r1,
                                               VQuarter r2, VQuarter r3) {
  HWY_IF_CONSTEXPR(hn::MaxLanes(d) * sizeof(hn::TFromD<D>) == 64) {
    auto v = hn::ResizeBitCast(d, r0);
    v = hn::InsertBlock<1>(v, r1);
    v = hn::InsertBlock<2>(v, r2);
    return hn::InsertBlock<3>(v, r3);
  }
  constexpr hn::Half<D> d_half;
  auto r01 = Combine2Rows(d_half, r0, r1);
  auto r23 = Combine2Rows(d_half, r2, r3);
  return Combine2Rows(d, r01, r23);
}

template <int RowsPerVec, class D>
HWY_ATTR HWY_INLINE hn::VFromD<D> LoadRowsU(D full_tag,
                                            const hn::TFromD<D> *ptr,
                                            ptrdiff_t stride) {
  HWY_IF_CONSTEXPR(RowsPerVec == 4) {
    constexpr hn::Half<hn::Half<D>> row_tag;
    const hn::TFromD<D> *ptr2 = ptr + stride * 2;
    return Combine4Rows(
        full_tag, hn::LoadU(row_tag, ptr), hn::LoadU(row_tag, ptr + stride),
        hn::LoadU(row_tag, ptr2), hn::LoadU(row_tag, ptr2 + stride));
  }
  HWY_IF_CONSTEXPR(RowsPerVec == 2) {
    constexpr hn::Half<D> row_tag;
    return Combine2Rows(full_tag, hn::LoadU(row_tag, ptr),
                        hn::LoadU(row_tag, ptr + stride));
  }
  return hn::LoadU(full_tag, ptr);
}

template <class DU8, class DI32, class DI16>
HWY_ATTR HWY_INLINE void VarianceKernel(DU8 du8, DI32 di32, DI16 di16,
                                        hn::VFromD<DU8> src,
                                        hn::VFromD<DU8> ref,
                                        hn::VFromD<DI32> *sse,
                                        hn::VFromD<DI16> *sum) {
  constexpr hn::Repartition<int8_t, DU8> di8;
  const auto adj_sub =
      hn::BitCast(di8, hn::Set(di16, static_cast<int16_t>(0xff01)));
  const auto s_lo = hn::InterleaveLower(du8, src, ref);
  const auto s_hi = hn::InterleaveUpper(du8, src, ref);
  const auto diff_lo = hn::SatWidenMulPairwiseAdd(di16, s_lo, adj_sub);
  const auto diff_hi = hn::SatWidenMulPairwiseAdd(di16, s_hi, adj_sub);

  *sum = hn::Add(*sum, hn::Add(diff_lo, diff_hi));
  const auto madd_lo = hn::WidenMulPairwiseAdd(di32, diff_lo, diff_lo);
  const auto madd_hi = hn::WidenMulPairwiseAdd(di32, diff_hi, diff_hi);
  *sse = hn::Add(*sse, hn::Add(madd_lo, madd_hi));
}

template <class DI32>
HWY_ATTR HWY_INLINE int VarianceFinalFrom32BitSum(DI32 di32,
                                                  hn::VFromD<DI32> vsse,
                                                  hn::VFromD<DI32> vsum32,
                                                  unsigned int *const sse) {
  constexpr hn::BlockDFromD<DI32> d128;
  const auto sse128 = BlockReduceSum(di32, vsse);
  const auto sum128 = BlockReduceSum(di32, vsum32);
  const auto pair0 = hn::InterleaveLower(d128, sse128, sum128);
  const auto pair1 = hn::InterleaveUpper(d128, sse128, sum128);
  const auto sum_pair = hn::Add(pair0, pair1);
  constexpr hn::FixedTag<int32_t, 2> d64;
  const auto res =
      hn::Add(hn::LowerHalf(d64, sum_pair), hn::UpperHalf(d64, sum_pair));
  *sse = static_cast<unsigned int>(hn::GetLane(res));
  return hn::ExtractLane(res, 1);
}

template <class DI32, class DI16>
HWY_ATTR HWY_INLINE int VarianceFinalFrom16BitSum(DI32 di32, DI16 di16,
                                                  hn::VFromD<DI32> vsse,
                                                  hn::VFromD<DI16> vsum,
                                                  unsigned int *const sse) {
  const auto one = hn::Set(di16, int16_t{ 1 });
  const auto vsum32 = hn::WidenMulPairwiseAdd(di32, vsum, one);
  return VarianceFinalFrom32BitSum(di32, vsse, vsum32, sse);
}

template <int BlockWidth>
HWY_ATTR HWY_INLINE void VarianceBlock(
    const uint8_t *src, int src_stride, const uint8_t *ref, int ref_stride,
    int h, hn::Vec<hn::ScalableTag<int32_t>> *vsse,
    hn::Vec<hn::ScalableTag<int16_t>> *vsum) {
  constexpr hn::ScalableTag<uint8_t> du8;
  constexpr hn::ScalableTag<int16_t> di16;
  constexpr hn::ScalableTag<int32_t> di32;
  constexpr int vw = hn::MaxLanes(du8);
  constexpr int kRowsPerVec = (BlockWidth < vw) ? (vw / BlockWidth) : 1;
  constexpr int kRowSpan = (BlockWidth < vw) ? vw : BlockWidth;
  assert(h % kRowsPerVec == 0);

  *vsum = hn::Zero(di16);
  for (int i = 0; i < h; i += kRowsPerVec) {
    for (int j = 0; j < kRowSpan; j += vw) {
      const auto s = LoadRowsU<kRowsPerVec>(du8, src + j, src_stride);
      const auto r = LoadRowsU<kRowsPerVec>(du8, ref + j, ref_stride);
      VarianceKernel(du8, di32, di16, s, r, vsse, vsum);
    }
    src += kRowsPerVec * src_stride;
    ref += kRowsPerVec * ref_stride;
  }
}

template <int BlockWidth, int BlockHeight>
HWY_ATTR HWY_INLINE void GetBlockVar(const uint8_t *src, int src_stride,
                                     const uint8_t *ref, int ref_stride,
                                     unsigned int *sse, int *sum) {
  constexpr hn::ScalableTag<uint8_t> du8;
  constexpr hn::ScalableTag<int16_t> di16;
  constexpr hn::ScalableTag<int32_t> di32;
  constexpr int vw = hn::MaxLanes(du8);
  constexpr int kRowsPerVec = (BlockWidth < vw) ? (vw / BlockWidth) : 1;
  // Each int16_t lane in vsum accumulates 2 pixel diffs per iteration of vw
  // bytes. With 128 diffs per lane (128 * 255 = 32640 <= 32767), we can safely
  // accumulate up to 64 * vw pixels before widening vsum to 32-bit.
  constexpr int kMaxPixelsPerChunk = 64 * vw;
  constexpr int kChunkHeight = (BlockWidth * BlockHeight <= kMaxPixelsPerChunk)
                                   ? BlockHeight
                                   : (kMaxPixelsPerChunk / BlockWidth);
  static_assert(BlockHeight % kRowsPerVec == 0,
                "BlockHeight must be a multiple of kRowsPerVec.");
  static_assert(BlockHeight % kChunkHeight == 0,
                "BlockHeight must be a multiple of kChunkHeight.");
  static_assert(kChunkHeight % kRowsPerVec == 0,
                "kChunkHeight must be a multiple of kRowsPerVec.");

  auto vsse = hn::Zero(di32);
  HWY_IF_CONSTEXPR(kChunkHeight == BlockHeight) {
    hn::Vec<decltype(di16)> vsum;
    VarianceBlock<BlockWidth>(src, src_stride, ref, ref_stride, BlockHeight,
                              &vsse, &vsum);
    *sum = VarianceFinalFrom16BitSum(di32, di16, vsse, vsum, sse);
  }
  HWY_IF_CONSTEXPR(kChunkHeight != BlockHeight) {
    auto vsum32 = hn::Zero(di32);
    const auto one = hn::Set(di16, int16_t{ 1 });
    for (int i = 0; i < BlockHeight / kChunkHeight; ++i) {
      hn::Vec<decltype(di16)> vsum16;
      VarianceBlock<BlockWidth>(src, src_stride, ref, ref_stride, kChunkHeight,
                                &vsse, &vsum16);
      vsum32 = hn::Add(vsum32, hn::WidenMulPairwiseAdd(di32, vsum16, one));
      src += kChunkHeight * src_stride;
      ref += kChunkHeight * ref_stride;
    }
    *sum = VarianceFinalFrom32BitSum(di32, vsse, vsum32, sse);
  }
}

template <int BlockWidth, int BlockHeight, int Bits>
HWY_ATTR HWY_INLINE unsigned int VarianceWxH(const uint8_t *src, int src_stride,
                                             const uint8_t *ref, int ref_stride,
                                             unsigned int *sse) {
  int sum = 0;
  GetBlockVar<BlockWidth, BlockHeight>(src, src_stride, ref, ref_stride, sse,
                                       &sum);
  return *sse -
         static_cast<unsigned int>((static_cast<int64_t>(sum) * sum) >> Bits);
}

HWY_ATTR HWY_INLINE void GetBlkSseSum(const int16_t *data, int stride, int bw,
                                      int bh, int *x_sum, int64_t *x2_sum) {
  constexpr hn::ScalableTag<int16_t> di16;
  constexpr hn::ScalableTag<int32_t> di32;
  constexpr hn::ScalableTag<uint32_t> du32;
  constexpr hn::ScalableTag<uint64_t> du64;
  constexpr int kLanes = hn::MaxLanes(di16);

  if (bw <= 0 || bh <= 0) {
    *x_sum = 0;
    *x2_sum = 0;
    return;
  }
  if (bw < kLanes || bw % kLanes != 0) {
    aom_get_blk_sse_sum_avx2(data, stride, bw, bh, x_sum, x2_sum);
    return;
  }

  const auto one = hn::Set(di16, int16_t{ 1 });
  auto sum_32 = hn::Zero(di32);
  auto sse_64 = hn::Zero(du64);

  for (int i = 0; i < bh; ++i) {
    for (int j = 0; j < bw; j += kLanes) {
      const auto d = hn::LoadU(di16, data + j);
      sum_32 = hn::Add(sum_32, hn::WidenMulPairwiseAdd(di32, d, one));
      const auto sq = hn::BitCast(du32, hn::WidenMulPairwiseAdd(di32, d, d));
      const auto sq_lo = hn::PromoteLowerTo(du64, sq);
      const auto sq_hi = hn::PromoteUpperTo(du64, sq);
      sse_64 = hn::Add(sse_64, hn::Add(sq_lo, sq_hi));
    }
    data += stride;
  }
  *x_sum = static_cast<int>(hn::ReduceSum(di32, sum_32));
  *x2_sum = static_cast<int64_t>(hn::ReduceSum(du64, sse_64));
}

HWY_ATTR HWY_INLINE void LoadRowHorizFilterU8(
    const uint8_t *src_row, int x, int bw,
    hn::Vec<hn::ScalableTag<int16_t>> *u16_curr_out,
    hn::Vec<hn::ScalableTag<int16_t>> *u16_H_out) {
  constexpr hn::ScalableTag<int16_t> di16;
  constexpr hn::Rebind<uint8_t, decltype(di16)> du8_half;
  constexpr int kCols = hn::MaxLanes(du8_half);
  static_assert(kCols == 16 || kCols == 32, "kCols must be 16 or 32.");
  alignas(64) static constexpr int16_t kLeftIdx[32] = {
    0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
  };
  alignas(64) static constexpr int16_t kRightIdx32[32] = {
    1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 31
  };
  alignas(32) static constexpr int16_t kRightIdx16[16] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15
  };

  const auto u16_curr = hn::PromoteTo(di16, hn::LoadU(du8_half, src_row + x));
  hn::Vec<decltype(di16)> u16_left;
  hn::Vec<decltype(di16)> u16_right;

  if (x == 0) {
    u16_left =
        hn::TableLookupLanes(u16_curr, hn::SetTableIndices(di16, kLeftIdx));
  } else {
    u16_left = hn::PromoteTo(di16, hn::LoadU(du8_half, src_row + x - 1));
  }

  if (x + kCols < bw) {
    u16_right = hn::PromoteTo(di16, hn::LoadU(du8_half, src_row + x + 1));
  } else {
    const int16_t *right_idx = (kCols == 32) ? kRightIdx32 : kRightIdx16;
    u16_right =
        hn::TableLookupLanes(u16_curr, hn::SetTableIndices(di16, right_idx));
  }

  *u16_curr_out = u16_curr;
  *u16_H_out =
      hn::Add(hn::Add(u16_left, u16_right), hn::ShiftLeft<1>(u16_curr));
}

HWY_ATTR HWY_INLINE int64_t CalcVarianceStat(const uint8_t *src, int stride,
                                             int bw, int bh) {
  constexpr hn::ScalableTag<int16_t> di16;
  constexpr hn::ScalableTag<int32_t> di32;
  constexpr int kCols = hn::MaxLanes(di16);

  if (bw <= 0 || bh <= 0) return 0;
  if (bw == 4 || bw == 8 || bw == 16) {
    return aom_calc_variance_stat_avx2(src, stride, bw, bh);
  }
  if (bw % kCols != 0) {
    return aom_calc_variance_stat_c(src, stride, bw, bh);
  }

  auto acc_var = hn::Zero(di32);

  for (int x = 0; x < bw; x += kCols) {
    hn::Vec<decltype(di16)> u16_curr, H_curr;
    LoadRowHorizFilterU8(src, x, bw, &u16_curr, &H_curr);
    auto H_top = H_curr;

    for (int y = 0; y < bh - 1; ++y) {
      hn::Vec<decltype(di16)> u16_next, H_bot;
      LoadRowHorizFilterU8(src + (y + 1) * stride, x, bw, &u16_next, &H_bot);
      const auto u16_V =
          hn::Add(hn::Add(H_top, H_bot), hn::ShiftLeft<1>(H_curr));
      const auto u16_sum = hn::ShiftRight<4>(u16_V);
      const auto diff = hn::Sub(u16_curr, u16_sum);
      acc_var = hn::Add(acc_var, hn::WidenMulPairwiseAdd(di32, diff, diff));
      H_top = H_curr;
      H_curr = H_bot;
      u16_curr = u16_next;
    }
    {
      const auto u16_V =
          hn::Add(hn::Add(H_top, H_curr), hn::ShiftLeft<1>(H_curr));
      const auto u16_sum = hn::ShiftRight<4>(u16_V);
      const auto diff = hn::Sub(u16_curr, u16_sum);
      acc_var = hn::Add(acc_var, hn::WidenMulPairwiseAdd(di32, diff, diff));
    }
  }
  return static_cast<int64_t>(hn::ReduceSum(di32, acc_var)) << 4;
}

#if CONFIG_AV1_HIGHBITDEPTH
HWY_ATTR HWY_INLINE void HighbdLoadRowHorizFilterU16(
    const uint16_t *src_row, int x, int bw,
    hn::Vec<hn::ScalableTag<int16_t>> *v_curr_out,
    hn::Vec<hn::ScalableTag<uint16_t>> *u16_H_out) {
  constexpr hn::ScalableTag<uint16_t> du16;
  constexpr hn::ScalableTag<int16_t> di16;
  constexpr int kCols = hn::MaxLanes(du16);
  static_assert(kCols == 16 || kCols == 32, "kCols must be 16 or 32.");
  alignas(64) static constexpr uint16_t kLeftIdx[32] = {
    0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
  };
  alignas(64) static constexpr uint16_t kRightIdx32[32] = {
    1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 31
  };
  alignas(32) static constexpr uint16_t kRightIdx16[16] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15
  };

  const auto v_curr = hn::LoadU(du16, src_row + x);
  // Input pixels must be <= 4095 (12-bit) so that the 3x3 Gaussian filter sum
  // u16_V = H_top + 2 * H_curr + H_bot <= 16 * 4095 = 65520 fits in uint16_t.
  assert(hn::AllTrue(du16, hn::Le(v_curr, hn::Set(du16, uint16_t{ 4095 }))));
  hn::Vec<decltype(du16)> v_left;
  hn::Vec<decltype(du16)> v_right;

  if (x == 0) {
    v_left = hn::TableLookupLanes(v_curr, hn::SetTableIndices(du16, kLeftIdx));
  } else {
    v_left = hn::LoadU(du16, src_row + x - 1);
  }

  if (x + kCols < bw) {
    v_right = hn::LoadU(du16, src_row + x + 1);
  } else {
    const uint16_t *right_idx = (kCols == 32) ? kRightIdx32 : kRightIdx16;
    v_right =
        hn::TableLookupLanes(v_curr, hn::SetTableIndices(du16, right_idx));
  }

  *v_curr_out = hn::BitCast(di16, v_curr);
  *u16_H_out = hn::Add(hn::Add(v_left, v_right), hn::ShiftLeft<1>(v_curr));
}

// Assumes 8/10/12-bit input pixels (src[i] <= 4095) so that the 16-weight 3x3
// Gaussian filter sum (<= 65520) does not overflow uint16_t.
HWY_ATTR HWY_INLINE int64_t HighbdCalcVarianceStat(const uint16_t *src,
                                                   int stride, int bw, int bh) {
  constexpr hn::ScalableTag<uint16_t> du16;
  constexpr hn::ScalableTag<int16_t> di16;
  constexpr hn::ScalableTag<int32_t> di32;
  constexpr hn::ScalableTag<int64_t> di64;
  constexpr int kCols = hn::MaxLanes(du16);

  if (bw <= 0 || bh <= 0) return 0;
  if (bw == 4 || bw == 8 || bw == 16) {
    return aom_highbd_calc_variance_stat_avx2(src, stride, bw, bh);
  }
  if (bw % kCols != 0) {
    return aom_highbd_calc_variance_stat_c(src, stride, bw, bh);
  }

  auto acc_var_64 = hn::Zero(di64);

  for (int x = 0; x < bw; x += kCols) {
    hn::Vec<decltype(di16)> v_curr;
    hn::Vec<decltype(du16)> H_curr;
    HighbdLoadRowHorizFilterU16(src, x, bw, &v_curr, &H_curr);
    auto H_top = H_curr;

    for (int y = 0; y < bh - 1; ++y) {
      hn::Vec<decltype(di16)> v_next;
      hn::Vec<decltype(du16)> H_bot;
      HighbdLoadRowHorizFilterU16(src + (y + 1) * stride, x, bw, &v_next,
                                  &H_bot);
      const auto u16_V =
          hn::Add(hn::Add(H_top, H_bot), hn::ShiftLeft<1>(H_curr));
      const auto u16_sum = hn::BitCast(di16, hn::ShiftRight<4>(u16_V));
      const auto diff = hn::Sub(v_curr, u16_sum);
      const auto diff_sq = hn::WidenMulPairwiseAdd(di32, diff, diff);
      acc_var_64 =
          hn::Add(acc_var_64, hn::Add(hn::PromoteLowerTo(di64, diff_sq),
                                      hn::PromoteUpperTo(di64, diff_sq)));
      H_top = H_curr;
      H_curr = H_bot;
      v_curr = v_next;
    }
    {
      const auto u16_V =
          hn::Add(hn::Add(H_top, H_curr), hn::ShiftLeft<1>(H_curr));
      const auto u16_sum = hn::BitCast(di16, hn::ShiftRight<4>(u16_V));
      const auto diff = hn::Sub(v_curr, u16_sum);
      const auto diff_sq = hn::WidenMulPairwiseAdd(di32, diff, diff);
      acc_var_64 =
          hn::Add(acc_var_64, hn::Add(hn::PromoteLowerTo(di64, diff_sq),
                                      hn::PromoteUpperTo(di64, diff_sq)));
    }
  }
  return static_cast<int64_t>(hn::ReduceSum(di64, acc_var_64)) << 4;
}
#endif  // CONFIG_AV1_HIGHBITDEPTH

// -----------------------------------------------------------------------------
// Subpixel Variance & Subpixel Avg Variance
// -----------------------------------------------------------------------------

template <class DI8>
HWY_ATTR HWY_INLINE hn::VFromD<DI8> BilinearFilter(DI8 di8, int offset) {
  assert(offset >= 0 && offset < 8);
  constexpr hn::Repartition<int16_t, DI8> di16;
  const int f0 = 16 - 2 * offset;
  const int f1 = 2 * offset;
  return hn::BitCast(di8, hn::Set(di16, static_cast<int16_t>(f0 | (f1 << 8))));
}

template <class DU8, class DI16>
HWY_ATTR HWY_INLINE void FilterPair(
    DU8 du8, DI16 di16, hn::VFromD<DU8> s0, hn::VFromD<DU8> s1,
    hn::VFromD<hn::Repartition<int8_t, DU8>> filt, hn::VFromD<DI16> *out_lo,
    hn::VFromD<DI16> *out_hi) {
  const auto pw8 = hn::Set(di16, int16_t{ 8 });
  const auto lo =
      hn::SatWidenMulPairwiseAdd(di16, hn::InterleaveLower(du8, s0, s1), filt);
  const auto hi =
      hn::SatWidenMulPairwiseAdd(di16, hn::InterleaveUpper(du8, s0, s1), filt);
  *out_lo = hn::ShiftRight<4>(hn::Add(lo, pw8));
  *out_hi = hn::ShiftRight<4>(hn::Add(hi, pw8));
}

template <class DU8>
HWY_ATTR HWY_INLINE hn::VFromD<DU8> FilterPairPack(
    DU8 du8, hn::VFromD<DU8> s0, hn::VFromD<DU8> s1,
    hn::VFromD<hn::Repartition<int8_t, DU8>> filt) {
  constexpr hn::Repartition<int16_t, DU8> di16;
  hn::VFromD<decltype(di16)> lo, hi;
  FilterPair(du8, di16, s0, s1, filt, &lo, &hi);
  return hn::ReorderDemote2To(du8, lo, hi);
}

template <class DU8, class DI16, class DI32>
HWY_ATTR HWY_INLINE void AccumDiff(DU8 du8, DI16 di16, DI32 di32,
                                   hn::VFromD<DI16> exp_src_lo,
                                   hn::VFromD<DI16> exp_src_hi,
                                   hn::VFromD<DU8> dst_reg,
                                   hn::VFromD<DI32> *sse_reg,
                                   hn::VFromD<DI16> *sum_reg) {
  const auto zero = hn::Zero(du8);
  const auto exp_dst_lo =
      hn::BitCast(di16, hn::InterleaveLower(du8, dst_reg, zero));
  const auto exp_dst_hi =
      hn::BitCast(di16, hn::InterleaveUpper(du8, dst_reg, zero));
  const auto d_lo = hn::Sub(exp_src_lo, exp_dst_lo);
  const auto d_hi = hn::Sub(exp_src_hi, exp_dst_hi);
  *sum_reg = hn::Add(*sum_reg, hn::Add(d_lo, d_hi));
  const auto madd_lo = hn::WidenMulPairwiseAdd(di32, d_lo, d_lo);
  const auto madd_hi = hn::WidenMulPairwiseAdd(di32, d_hi, d_hi);
  *sse_reg = hn::Add(*sse_reg, hn::Add(madd_lo, madd_hi));
}

template <int Width, class D>
HWY_ATTR HWY_INLINE hn::VFromD<D> SubPelLoad(D d, const uint8_t *ptr,
                                             ptrdiff_t half_stride) {
  HWY_IF_CONSTEXPR(Width == 32 && hn::MaxLanes(d) == 64) {
    constexpr hn::Half<D> d_half;
    return Combine2Rows(d, hn::LoadU(d_half, ptr),
                        hn::LoadU(d_half, ptr + half_stride));
  }
  return hn::LoadU(d, ptr);
}

template <int Width, bool HasSec, class LoadRowFn>
HWY_ATTR HWY_INLINE void SubPelVertLoop(
    const uint8_t *s_ptr, int src_stride, int y_offset, const uint8_t *d_ptr,
    int dst_stride, ptrdiff_t half_dst_stride, const uint8_t *sec_ptr,
    int sec_stride, ptrdiff_t half_sec_stride, int loop_h, LoadRowFn load_row,
    hn::Vec<hn::ScalableTag<int32_t>> *sse_reg,
    hn::Vec<hn::ScalableTag<int16_t>> *sum_reg) {
  constexpr hn::ScalableTag<uint8_t> du8;
  constexpr hn::Repartition<int8_t, decltype(du8)> di8;
  constexpr hn::Repartition<int16_t, decltype(du8)> di16;
  constexpr hn::Repartition<int32_t, decltype(du8)> di32;

  if (y_offset == 0) {
    for (int i = 0; i < loop_h; ++i) {
      auto s = load_row(s_ptr);
      HWY_IF_CONSTEXPR(HasSec) {
        s = hn::AverageRound(s,
                             SubPelLoad<Width>(du8, sec_ptr, half_sec_stride));
        sec_ptr += sec_stride;
      }
      const auto d = SubPelLoad<Width>(du8, d_ptr, half_dst_stride);
      VarianceKernel(du8, di32, di16, s, d, sse_reg, sum_reg);
      s_ptr += src_stride;
      d_ptr += dst_stride;
    }
  } else if (y_offset == 4) {
    auto s_prev = load_row(s_ptr);
    for (int i = 0; i < loop_h; ++i) {
      s_ptr += src_stride;
      const auto s_curr = load_row(s_ptr);
      auto s_avg = hn::AverageRound(s_prev, s_curr);
      HWY_IF_CONSTEXPR(HasSec) {
        s_avg = hn::AverageRound(
            s_avg, SubPelLoad<Width>(du8, sec_ptr, half_sec_stride));
        sec_ptr += sec_stride;
      }
      const auto d = SubPelLoad<Width>(du8, d_ptr, half_dst_stride);
      VarianceKernel(du8, di32, di16, s_avg, d, sse_reg, sum_reg);
      s_prev = s_curr;
      d_ptr += dst_stride;
    }
  } else {
    const auto yfilter = BilinearFilter(di8, y_offset);
    auto s_prev = load_row(s_ptr);
    for (int i = 0; i < loop_h; ++i) {
      s_ptr += src_stride;
      const auto s_curr = load_row(s_ptr);
      const auto d = SubPelLoad<Width>(du8, d_ptr, half_dst_stride);
      HWY_IF_CONSTEXPR(HasSec) {
        const auto s_avg =
            hn::AverageRound(FilterPairPack(du8, s_prev, s_curr, yfilter),
                             SubPelLoad<Width>(du8, sec_ptr, half_sec_stride));
        VarianceKernel(du8, di32, di16, s_avg, d, sse_reg, sum_reg);
        sec_ptr += sec_stride;
      }
      HWY_IF_CONSTEXPR(!HasSec) {
        hn::VFromD<decltype(di16)> exp_lo, exp_hi;
        FilterPair(du8, di16, s_prev, s_curr, yfilter, &exp_lo, &exp_hi);
        AccumDiff(du8, di16, di32, exp_lo, exp_hi, d, sse_reg, sum_reg);
      }
      s_prev = s_curr;
      d_ptr += dst_stride;
    }
  }
}

template <int Width, bool HasSec>
HWY_ATTR HWY_INLINE int SubPixelVarianceWxHImp(
    const uint8_t *src, int src_stride, int x_offset, int y_offset,
    const uint8_t *dst, int dst_stride, const uint8_t *sec, int sec_stride,
    int height, unsigned int *sse) {
  constexpr hn::ScalableTag<uint8_t> du8;
  constexpr hn::Repartition<int8_t, decltype(du8)> di8;
  constexpr hn::Repartition<int16_t, decltype(du8)> di16;
  constexpr hn::Repartition<int32_t, decltype(du8)> di32;
  constexpr int vw = hn::MaxLanes(du8);
  constexpr int kRowsPerIter = (Width == 32 && vw == 64) ? 2 : 1;
  constexpr int kColStep = (Width == 32 && vw == 64) ? 32 : vw;
  static_assert(Width == 32 || Width % kColStep == 0,
                "Width must be 32 or a multiple of kColStep.");

  const int loop_h = height / kRowsPerIter;
  // Each int16_t lane in sum_reg accumulates 2 pixel diffs per iteration, so
  // loop_h must be <= 64 (128 * 255 = 32640 <= 32767) to avoid overflow.
  assert(height > 0 && height % kRowsPerIter == 0 && loop_h <= 64);
  const ptrdiff_t half_src_stride = static_cast<ptrdiff_t>(loop_h) * src_stride;
  const ptrdiff_t half_dst_stride = static_cast<ptrdiff_t>(loop_h) * dst_stride;
  const ptrdiff_t half_sec_stride = static_cast<ptrdiff_t>(loop_h) * sec_stride;

  const auto one = hn::Set(di16, int16_t{ 1 });
  auto sum32_reg = hn::Zero(di32);
  auto sse_reg = hn::Zero(di32);

  for (int col = 0; col < Width; col += kColStep) {
    auto sum_reg = hn::Zero(di16);
    const uint8_t *s_ptr = src + col;
    const uint8_t *d_ptr = dst + col;
    const uint8_t *sec_ptr = HasSec ? (sec + col) : nullptr;

    if (x_offset == 0) {
      SubPelVertLoop<Width, HasSec>(
          s_ptr, src_stride, y_offset, d_ptr, dst_stride, half_dst_stride,
          sec_ptr, sec_stride, half_sec_stride, loop_h,
          [=](const uint8_t *p)
              HWY_ATTR { return SubPelLoad<Width>(du8, p, half_src_stride); },
          &sse_reg, &sum_reg);
    } else if (x_offset == 4) {
      SubPelVertLoop<Width, HasSec>(
          s_ptr, src_stride, y_offset, d_ptr, dst_stride, half_dst_stride,
          sec_ptr, sec_stride, half_sec_stride, loop_h,
          [=](const uint8_t *p) HWY_ATTR {
            return hn::AverageRound(
                SubPelLoad<Width>(du8, p, half_src_stride),
                SubPelLoad<Width>(du8, p + 1, half_src_stride));
          },
          &sse_reg, &sum_reg);
    } else {
      const auto xfilter = BilinearFilter(di8, x_offset);
      if (y_offset == 0) {
        for (int i = 0; i < loop_h; ++i) {
          const auto s0 = SubPelLoad<Width>(du8, s_ptr, half_src_stride);
          const auto s1 = SubPelLoad<Width>(du8, s_ptr + 1, half_src_stride);
          const auto d = SubPelLoad<Width>(du8, d_ptr, half_dst_stride);
          HWY_IF_CONSTEXPR(HasSec) {
            const auto s = hn::AverageRound(
                FilterPairPack(du8, s0, s1, xfilter),
                SubPelLoad<Width>(du8, sec_ptr, half_sec_stride));
            VarianceKernel(du8, di32, di16, s, d, &sse_reg, &sum_reg);
            sec_ptr += sec_stride;
          }
          HWY_IF_CONSTEXPR(!HasSec) {
            hn::VFromD<decltype(di16)> exp_lo, exp_hi;
            FilterPair(du8, di16, s0, s1, xfilter, &exp_lo, &exp_hi);
            AccumDiff(du8, di16, di32, exp_lo, exp_hi, d, &sse_reg, &sum_reg);
          }
          s_ptr += src_stride;
          d_ptr += dst_stride;
        }
      } else {
        SubPelVertLoop<Width, HasSec>(
            s_ptr, src_stride, y_offset, d_ptr, dst_stride, half_dst_stride,
            sec_ptr, sec_stride, half_sec_stride, loop_h,
            [=](const uint8_t *p) HWY_ATTR {
              return FilterPairPack(
                  du8, SubPelLoad<Width>(du8, p, half_src_stride),
                  SubPelLoad<Width>(du8, p + 1, half_src_stride), xfilter);
            },
            &sse_reg, &sum_reg);
      }
    }
    sum32_reg = hn::Add(sum32_reg, hn::WidenMulPairwiseAdd(di32, sum_reg, one));
  }
  return VarianceFinalFrom32BitSum(di32, sse_reg, sum32_reg, sse);
}

template <int BlockWidth, int BlockHeight, int ChunkHeight, int WLog2,
          int HLog2, bool HasSec>
HWY_ATTR HWY_INLINE unsigned int SubPixelVarianceWxH(
    const uint8_t *src, int src_stride, int x_offset, int y_offset,
    const uint8_t *dst, int dst_stride, unsigned int *sse_ptr,
    const uint8_t *sec) {
  constexpr int kChunkWidth = (BlockWidth == 32) ? 32 : 64;
  static_assert(BlockWidth == 32 || BlockWidth % 64 == 0,
                "BlockWidth must be 32 or a multiple of 64.");
  static_assert(BlockHeight % ChunkHeight == 0,
                "BlockHeight must be a multiple of ChunkHeight.");
  // ChunkHeight <= 64 keeps the int16_t accumulator in SubPixelVarianceWxHImp
  // within bounds (at most 64 iterations * 2 diffs = 128 diffs per lane).
  static_assert(ChunkHeight > 0 && ChunkHeight <= 64,
                "ChunkHeight must be in (0, 64] to avoid int16_t overflow.");
  static_assert((1 << (WLog2 + HLog2)) == BlockWidth * BlockHeight,
                "(1 << (WLog2 + HLog2)) must equal BlockWidth * BlockHeight.");
  HWY_IF_CONSTEXPR(BlockWidth == kChunkWidth && BlockHeight == ChunkHeight) {
    const int sum = SubPixelVarianceWxHImp<kChunkWidth, HasSec>(
        src, src_stride, x_offset, y_offset, dst, dst_stride,
        HasSec ? sec : nullptr, HasSec ? BlockWidth : 0, BlockHeight, sse_ptr);
    return *sse_ptr - static_cast<unsigned int>(
                          (static_cast<int64_t>(sum) * sum) >> (WLog2 + HLog2));
  }
  unsigned int sse = 0;
  int se = 0;
  for (int i = 0; i < BlockWidth / kChunkWidth; ++i) {
    const uint8_t *src_ptr = src;
    const uint8_t *dst_ptr = dst;
    const uint8_t *sec_ptr = sec;
    for (int j = 0; j < BlockHeight / ChunkHeight; ++j) {
      unsigned int sse2;
      const int se2 = SubPixelVarianceWxHImp<kChunkWidth, HasSec>(
          src_ptr, src_stride, x_offset, y_offset, dst_ptr, dst_stride,
          HasSec ? sec_ptr : nullptr, HasSec ? BlockWidth : 0, ChunkHeight,
          &sse2);
      dst_ptr += ChunkHeight * dst_stride;
      src_ptr += ChunkHeight * src_stride;
      if (HasSec) sec_ptr += ChunkHeight * BlockWidth;
      se += se2;
      sse += sse2;
    }
    src += kChunkWidth;
    dst += kChunkWidth;
    if (HasSec) sec += kChunkWidth;
  }
  *sse_ptr = sse;
  return sse - static_cast<unsigned int>((static_cast<int64_t>(se) * se) >>
                                         (WLog2 + HLog2));
}

// -----------------------------------------------------------------------------
// OBMC Variance & CompoundAvg / CompoundMask Pred
// -----------------------------------------------------------------------------

#if !CONFIG_REALTIME_ONLY
HWY_ATTR HWY_INLINE void ObmcVarianceW16N(
    const uint8_t *pre, const int pre_stride, const int32_t *wsrc,
    const int32_t *mask, unsigned int *const sse, int *const sum, const int w,
    const int h) {
  constexpr hn::ScalableTag<int32_t> di32;
  constexpr hn::Repartition<int16_t, decltype(di32)> di16;
  constexpr hn::Rebind<uint8_t, decltype(di32)> du8_quarter;
  constexpr int kStep = hn::MaxLanes(di32);

  assert(w >= kStep);
  assert(IS_POWER_OF_TWO(w));
  assert(IS_POWER_OF_TWO(h));

  int n = 0;
  const auto v_bias_d = hn::Set(di32, (1 << 12) >> 1);
  auto v_sum_d = hn::Zero(di32);
  auto v_sse_d = hn::Zero(di32);

  for (int row = 0; row < h; ++row) {
    for (int col = 0; col < w; col += kStep) {
      const auto v_p_b = hn::LoadU(du8_quarter, pre + col);
      const auto v_m_d = hn::LoadU(di32, mask + n);
      const auto v_w_d = hn::LoadU(di32, wsrc + n);
      const auto v_p_d = hn::PromoteTo(di32, v_p_b);
      // Both pre ([0, 255]) and mask ([0, 64 * 64]) fit in signed 15-bit
      // integers, so their upper int16_t halves in int32_t lanes are 0 and
      // WidenMulPairwiseAdd computes exact 32-bit products pre * mask.
      const auto v_pm_d = hn::WidenMulPairwiseAdd(
          di32, hn::BitCast(di16, v_p_d), hn::BitCast(di16, v_m_d));
      const auto v_diff_d = hn::Sub(v_w_d, v_pm_d);
      const auto v_sign_d = hn::ShiftRight<31>(v_diff_d);
      const auto v_tmp_d = hn::Add(hn::Add(v_diff_d, v_bias_d), v_sign_d);
      const auto v_rdiff_d = hn::ShiftRight<12>(v_tmp_d);

      v_sum_d = hn::Add(v_sum_d, v_rdiff_d);
      v_sse_d = hn::Add(v_sse_d, hn::Mul(v_rdiff_d, v_rdiff_d));
      n += kStep;
    }
    pre += pre_stride;
  }
  *sum = VarianceFinalFrom32BitSum(di32, v_sse_d, v_sum_d, sse);
}
#endif  // !CONFIG_REALTIME_ONLY

HWY_ATTR HWY_INLINE void CompAvgPred(uint8_t *comp_pred, const uint8_t *pred,
                                     int width, int height, const uint8_t *ref,
                                     int ref_stride) {
  constexpr hn::ScalableTag<uint8_t> du8;
  constexpr int vw = hn::MaxLanes(du8);
  int row = 0;
  if (width == 8) {
    assert(height % 4 == 0);
    constexpr hn::FixedTag<uint8_t, 32> du8_32;
    do {
      const auto pred_0123 = hn::LoadU(du8_32, pred);
      const auto ref_0123 = LoadRowsU<4>(du8_32, ref, ref_stride);
      hn::StoreU(hn::AverageRound(pred_0123, ref_0123), du8_32, comp_pred);
      row += 4;
      pred += 32;
      comp_pred += 32;
      ref += 4 * ref_stride;
    } while (row < height);
  } else if (width == 16) {
    constexpr int kRows = (16 < vw) ? (vw / 16) : 1;
    do {
      const auto pred_vec = hn::LoadU(du8, pred);
      const auto ref_vec = LoadRowsU<kRows>(du8, ref, ref_stride);
      hn::StoreU(hn::AverageRound(pred_vec, ref_vec), du8, comp_pred);
      row += kRows;
      pred += 16 * kRows;
      comp_pred += 16 * kRows;
      ref += kRows * ref_stride;
    } while (row < height);
  } else if (width == 32) {
    constexpr int kRows = (32 < vw) ? (vw / 32) : 1;
    do {
      const auto pred_vec = hn::LoadU(du8, pred);
      const auto ref_vec = LoadRowsU<kRows>(du8, ref, ref_stride);
      hn::StoreU(hn::AverageRound(pred_vec, ref_vec), du8, comp_pred);
      row += kRows;
      pred += 32 * kRows;
      comp_pred += 32 * kRows;
      ref += kRows * ref_stride;
    } while (row < height);
  } else if (width % vw == 0) {
    do {
      for (int x = 0; x < width; x += vw) {
        const auto p0 = hn::LoadU(du8, pred + x);
        const auto r0 = hn::LoadU(du8, ref + x);
        const auto p1 = hn::LoadU(du8, pred + width + x);
        const auto r1 = hn::LoadU(du8, ref + ref_stride + x);
        hn::StoreU(hn::AverageRound(p0, r0), du8, comp_pred + x);
        hn::StoreU(hn::AverageRound(p1, r1), du8, comp_pred + width + x);
      }
      row += 2;
      pred += 2 * width;
      comp_pred += 2 * width;
      ref += 2 * ref_stride;
    } while (row < height);
  } else {
    aom_comp_avg_pred_c(comp_pred, pred, width, height, ref, ref_stride);
  }
}

template <class DU8>
HWY_ATTR HWY_INLINE void CompMaskPredLine(DU8 du8, hn::VFromD<DU8> s0,
                                          hn::VFromD<DU8> s1, hn::VFromD<DU8> a,
                                          uint8_t *comp_pred) {
  constexpr hn::Repartition<int8_t, DU8> di8;
  constexpr hn::Repartition<int16_t, DU8> di16;
  const auto alpha_max =
      hn::Set(du8, static_cast<uint8_t>(AOM_BLEND_A64_MAX_ALPHA));
  constexpr int16_t round_bits = 15 - AOM_BLEND_A64_ROUND_BITS;
  const auto round_offset =
      hn::Set(di16, static_cast<int16_t>(1 << round_bits));
  const auto ma = hn::Sub(alpha_max, a);

  const auto ssAL = hn::InterleaveLower(du8, s0, s1);
  const auto aaAL = hn::BitCast(di8, hn::InterleaveLower(du8, a, ma));
  const auto ssAH = hn::InterleaveUpper(du8, s0, s1);
  const auto aaAH = hn::BitCast(di8, hn::InterleaveUpper(du8, a, ma));

  const auto blendAL = hn::SatWidenMulPairwiseAdd(di16, ssAL, aaAL);
  const auto blendAH = hn::SatWidenMulPairwiseAdd(di16, ssAH, aaAH);
  const auto roundAL = hn::MulFixedPoint15(blendAL, round_offset);
  const auto roundAH = hn::MulFixedPoint15(blendAH, round_offset);

  const auto roundA = hn::ReorderDemote2To(du8, roundAL, roundAH);
  hn::StoreU(roundA, du8, comp_pred);
}

HWY_ATTR HWY_INLINE void CompMaskPred(uint8_t *comp_pred, const uint8_t *pred,
                                      int width, int height, const uint8_t *ref,
                                      int ref_stride, const uint8_t *mask,
                                      int mask_stride, int invert_mask) {
  constexpr hn::ScalableTag<uint8_t> du8;
  constexpr int vw = hn::MaxLanes(du8);
  int i = 0;
  const uint8_t *src0 = invert_mask ? pred : ref;
  const uint8_t *src1 = invert_mask ? ref : pred;
  const int stride0 = invert_mask ? width : ref_stride;
  const int stride1 = invert_mask ? ref_stride : width;

  if (width == 8) {
    aom_comp_mask_pred_avx2(comp_pred, pred, width, height, ref, ref_stride,
                            mask, mask_stride, invert_mask);
  } else if (width == 16) {
    constexpr int kRows = (16 < vw) ? (vw / 16) : 1;
    do {
      const auto s0 = LoadRowsU<kRows>(du8, src0, stride0);
      const auto s1 = LoadRowsU<kRows>(du8, src1, stride1);
      const auto a = LoadRowsU<kRows>(du8, mask, mask_stride);
      CompMaskPredLine(du8, s0, s1, a, comp_pred);
      src0 += stride0 * kRows;
      src1 += stride1 * kRows;
      mask += mask_stride * kRows;
      comp_pred += 16 * kRows;
      i += kRows;
    } while (i < height);
  } else if (width == 32) {
    constexpr int kRows = (32 < vw) ? (vw / 32) : 1;
    do {
      const auto s0 = LoadRowsU<kRows>(du8, src0, stride0);
      const auto s1 = LoadRowsU<kRows>(du8, src1, stride1);
      const auto a = LoadRowsU<kRows>(du8, mask, mask_stride);
      CompMaskPredLine(du8, s0, s1, a, comp_pred);
      src0 += stride0 * kRows;
      src1 += stride1 * kRows;
      mask += mask_stride * kRows;
      comp_pred += 32 * kRows;
      i += kRows;
    } while (i < height);
  } else {
    do {
      for (int x = 0; x < width; x += vw) {
        const auto s0 = hn::LoadU(du8, src0 + x);
        const auto s1 = hn::LoadU(du8, src1 + x);
        const auto a = hn::LoadU(du8, mask + x);
        CompMaskPredLine(du8, s0, s1, a, comp_pred);
        comp_pred += vw;
      }
      src0 += stride0;
      src1 += stride1;
      mask += mask_stride;
      i++;
    } while (i < height);
  }
}

#if CONFIG_AV1_HIGHBITDEPTH
template <class DI16>
HWY_ATTR HWY_INLINE hn::VFromD<DI16> HighbdCompMaskPredLine(
    DI16 di16, hn::VFromD<DI16> s0, hn::VFromD<DI16> s1, hn::VFromD<DI16> a) {
  constexpr hn::Repartition<int32_t, DI16> di32;
  const auto alpha_max =
      hn::Set(di16, static_cast<int16_t>(1 << AOM_BLEND_A64_ROUND_BITS));
  const auto round_const = hn::Set(di32, (1 << AOM_BLEND_A64_ROUND_BITS) >> 1);
  const auto a_inv = hn::Sub(alpha_max, a);

  const auto s_lo = hn::InterleaveLower(di16, s0, s1);
  const auto a_lo = hn::InterleaveLower(di16, a, a_inv);
  const auto pred_l = hn::ShiftRight<AOM_BLEND_A64_ROUND_BITS>(
      hn::Add(hn::WidenMulPairwiseAdd(di32, s_lo, a_lo), round_const));

  const auto s_hi = hn::InterleaveUpper(di16, s0, s1);
  const auto a_hi = hn::InterleaveUpper(di16, a, a_inv);
  const auto pred_h = hn::ShiftRight<AOM_BLEND_A64_ROUND_BITS>(
      hn::Add(hn::WidenMulPairwiseAdd(di32, s_hi, a_hi), round_const));

  return hn::ReorderDemote2To(di16, pred_l, pred_h);
}

HWY_ATTR HWY_INLINE void HighbdCompMaskPred(uint8_t *comp_pred8,
                                            const uint8_t *pred8, int width,
                                            int height, const uint8_t *ref8,
                                            int ref_stride, const uint8_t *mask,
                                            int mask_stride, int invert_mask) {
  if (width < 16) {
    aom_highbd_comp_mask_pred_avx2(comp_pred8, pred8, width, height, ref8,
                                   ref_stride, mask, mask_stride, invert_mask);
    return;
  }
  constexpr hn::ScalableTag<int16_t> di16;
  constexpr hn::Rebind<uint8_t, decltype(di16)> du8_half;
  constexpr int kLanes16 = hn::MaxLanes(di16);

  const uint16_t *pred = CONVERT_TO_SHORTPTR(pred8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  int16_t *comp_pred =
      reinterpret_cast<int16_t *>(CONVERT_TO_SHORTPTR(comp_pred8));
  const int16_t *src0 =
      reinterpret_cast<const int16_t *>(invert_mask ? pred : ref);
  const int16_t *src1 =
      reinterpret_cast<const int16_t *>(invert_mask ? ref : pred);
  const int stride0 = invert_mask ? width : ref_stride;
  const int stride1 = invert_mask ? ref_stride : width;

  if (width == 16) {
    constexpr int kRows = (16 < kLanes16) ? (kLanes16 / 16) : 1;
    for (int i = 0; i < height; i += kRows) {
      const auto s0 = LoadRowsU<kRows>(di16, src0, stride0);
      const auto s1 = LoadRowsU<kRows>(di16, src1, stride1);
      const auto m8 = LoadRowsU<kRows>(du8_half, mask, mask_stride);
      const auto m16 = hn::PromoteTo(di16, m8);
      const auto comp = HighbdCompMaskPredLine(di16, s0, s1, m16);
      hn::StoreU(comp, di16, comp_pred);
      src0 += stride0 * kRows;
      src1 += stride1 * kRows;
      mask += mask_stride * kRows;
      comp_pred += 16 * kRows;
    }
  } else {
    for (int i = 0; i < height; ++i) {
      for (int x = 0; x < width; x += kLanes16) {
        const auto s0 = hn::LoadU(di16, src0 + x);
        const auto s1 = hn::LoadU(di16, src1 + x);
        const auto m8 = hn::LoadU(du8_half, mask + x);
        const auto m16 = hn::PromoteTo(di16, m8);
        const auto comp = HighbdCompMaskPredLine(di16, s0, s1, m16);
        hn::StoreU(comp, di16, comp_pred);
        comp_pred += kLanes16;
      }
      src0 += stride0;
      src1 += stride1;
      mask += mask_stride;
    }
  }
}
#endif  // CONFIG_AV1_HIGHBITDEPTH

}  // namespace HWY_NAMESPACE
}  // namespace

// -----------------------------------------------------------------------------
// Export Macros
// -----------------------------------------------------------------------------

#define FVAR(w, h, bits, suffix)                                              \
  extern "C" unsigned int aom_variance##w##x##h##_##suffix(                   \
      const uint8_t *src, int src_stride, const uint8_t *ref, int ref_stride, \
      unsigned int *sse);                                                     \
  HWY_ATTR unsigned int aom_variance##w##x##h##_##suffix(                     \
      const uint8_t *src, int src_stride, const uint8_t *ref, int ref_stride, \
      unsigned int *sse) {                                                    \
    return HWY_NAMESPACE::VarianceWxH<w, h, bits>(src, src_stride, ref,       \
                                                  ref_stride, sse);           \
  }

#define FMSE16X16(suffix)                                                     \
  extern "C" unsigned int aom_mse16x16_##suffix(                              \
      const uint8_t *src, int src_stride, const uint8_t *ref, int ref_stride, \
      unsigned int *sse);                                                     \
  HWY_ATTR unsigned int aom_mse16x16_##suffix(                                \
      const uint8_t *src, int src_stride, const uint8_t *ref, int ref_stride, \
      unsigned int *sse) {                                                    \
    int sum = 0;                                                              \
    HWY_NAMESPACE::GetBlockVar<16, 16>(src, src_stride, ref, ref_stride, sse, \
                                       &sum);                                 \
    return *sse;                                                              \
  }

#define FGET_BLK_SSE_SUM(suffix)                                              \
  extern "C" void aom_get_blk_sse_sum_##suffix(const int16_t *data,           \
                                               int stride, int bw, int bh,    \
                                               int *x_sum, int64_t *x2_sum);  \
  HWY_ATTR void aom_get_blk_sse_sum_##suffix(const int16_t *data, int stride, \
                                             int bw, int bh, int *x_sum,      \
                                             int64_t *x2_sum) {               \
    HWY_NAMESPACE::GetBlkSseSum(data, stride, bw, bh, x_sum, x2_sum);         \
  }

#if CONFIG_AV1_HIGHBITDEPTH
#define FHIGHBD_CALC_VAR_STAT(suffix)                                  \
  extern "C" int64_t aom_highbd_calc_variance_stat_##suffix(           \
      const uint16_t *src, int stride, int bw, int bh);                \
  HWY_ATTR int64_t aom_highbd_calc_variance_stat_##suffix(             \
      const uint16_t *src, int stride, int bw, int bh) {               \
    return HWY_NAMESPACE::HighbdCalcVarianceStat(src, stride, bw, bh); \
  }
#else
#define FHIGHBD_CALC_VAR_STAT(suffix)
#endif

#define FCALC_VAR_STAT(suffix)                                   \
  extern "C" int64_t aom_calc_variance_stat_##suffix(            \
      const uint8_t *src, int stride, int bw, int bh);           \
  HWY_ATTR int64_t aom_calc_variance_stat_##suffix(              \
      const uint8_t *src, int stride, int bw, int bh) {          \
    return HWY_NAMESPACE::CalcVarianceStat(src, stride, bw, bh); \
  }                                                              \
  FHIGHBD_CALC_VAR_STAT(suffix)

#if !CONFIG_REALTIME_ONLY
#define FOR_EACH_VAR_RECT_BLOCK_SIZE(X, suffix) \
  X(64, 16, 10, suffix)                         \
  X(32, 8, 8, suffix)                           \
  X(16, 64, 10, suffix)                         \
  X(16, 4, 6, suffix)
#else
#define FOR_EACH_VAR_RECT_BLOCK_SIZE(X, suffix)
#endif

#define FOR_EACH_VAR_BLOCK_SIZE(X, suffix) \
  X(128, 128, 14, suffix)                  \
  X(128, 64, 13, suffix)                   \
  X(64, 128, 13, suffix)                   \
  X(64, 64, 12, suffix)                    \
  X(64, 32, 11, suffix)                    \
  X(32, 64, 11, suffix)                    \
  X(32, 32, 10, suffix)                    \
  X(32, 16, 9, suffix)                     \
  X(16, 32, 9, suffix)                     \
  X(16, 16, 8, suffix)                     \
  X(16, 8, 7, suffix)                      \
  FOR_EACH_VAR_RECT_BLOCK_SIZE(X, suffix)

#define FSUBPIX_VAR_32XH(h, log2h, suffix)                                   \
  extern "C" unsigned int aom_sub_pixel_variance32x##h##_##suffix(           \
      const uint8_t *src, int src_stride, int x_offset, int y_offset,        \
      const uint8_t *dst, int dst_stride, unsigned int *sse);                \
  HWY_ATTR unsigned int aom_sub_pixel_variance32x##h##_##suffix(             \
      const uint8_t *src, int src_stride, int x_offset, int y_offset,        \
      const uint8_t *dst, int dst_stride, unsigned int *sse) {               \
    return HWY_NAMESPACE::SubPixelVarianceWxH<32, h, h, 5, log2h, false>(    \
        src, src_stride, x_offset, y_offset, dst, dst_stride, sse, nullptr); \
  }                                                                          \
  extern "C" unsigned int aom_sub_pixel_avg_variance32x##h##_##suffix(       \
      const uint8_t *src, int src_stride, int x_offset, int y_offset,        \
      const uint8_t *dst, int dst_stride, unsigned int *sse,                 \
      const uint8_t *sec_ptr);                                               \
  HWY_ATTR unsigned int aom_sub_pixel_avg_variance32x##h##_##suffix(         \
      const uint8_t *src, int src_stride, int x_offset, int y_offset,        \
      const uint8_t *dst, int dst_stride, unsigned int *sse,                 \
      const uint8_t *sec_ptr) {                                              \
    return HWY_NAMESPACE::SubPixelVarianceWxH<32, h, h, 5, log2h, true>(     \
        src, src_stride, x_offset, y_offset, dst, dst_stride, sse, sec_ptr); \
  }

#define FSUBPIX_VAR_64_128(w, h, hf, wlog2, hlog2, suffix)                    \
  extern "C" unsigned int aom_sub_pixel_variance##w##x##h##_##suffix(         \
      const uint8_t *src, int src_stride, int x_offset, int y_offset,         \
      const uint8_t *dst, int dst_stride, unsigned int *sse_ptr);             \
  HWY_ATTR unsigned int aom_sub_pixel_variance##w##x##h##_##suffix(           \
      const uint8_t *src, int src_stride, int x_offset, int y_offset,         \
      const uint8_t *dst, int dst_stride, unsigned int *sse_ptr) {            \
    return HWY_NAMESPACE::SubPixelVarianceWxH<w, h, hf, wlog2, hlog2, false>( \
        src, src_stride, x_offset, y_offset, dst, dst_stride, sse_ptr,        \
        nullptr);                                                             \
  }                                                                           \
  extern "C" unsigned int aom_sub_pixel_avg_variance##w##x##h##_##suffix(     \
      const uint8_t *src, int src_stride, int x_offset, int y_offset,         \
      const uint8_t *dst, int dst_stride, unsigned int *sse_ptr,              \
      const uint8_t *sec);                                                    \
  HWY_ATTR unsigned int aom_sub_pixel_avg_variance##w##x##h##_##suffix(       \
      const uint8_t *src, int src_stride, int x_offset, int y_offset,         \
      const uint8_t *dst, int dst_stride, unsigned int *sse_ptr,              \
      const uint8_t *sec) {                                                   \
    return HWY_NAMESPACE::SubPixelVarianceWxH<w, h, hf, wlog2, hlog2, true>(  \
        src, src_stride, x_offset, y_offset, dst, dst_stride, sse_ptr, sec);  \
  }

#if !CONFIG_REALTIME_ONLY
#define FOR_EACH_SUBPIX_VAR_32XH_RECT(X, suffix) X(8, 3, suffix)
#define FOR_EACH_SUBPIX_VAR_64_128_RECT(X, suffix) X(64, 16, 16, 6, 4, suffix)
#else
#define FOR_EACH_SUBPIX_VAR_32XH_RECT(X, suffix)
#define FOR_EACH_SUBPIX_VAR_64_128_RECT(X, suffix)
#endif

#define FOR_EACH_SUBPIX_VAR_32XH(X, suffix) \
  X(64, 6, suffix)                          \
  X(32, 5, suffix)                          \
  X(16, 4, suffix)                          \
  FOR_EACH_SUBPIX_VAR_32XH_RECT(X, suffix)

#define FOR_EACH_SUBPIX_VAR_64_128(X, suffix) \
  X(128, 128, 64, 7, 7, suffix)               \
  X(128, 64, 64, 7, 6, suffix)                \
  X(64, 128, 64, 6, 7, suffix)                \
  X(64, 64, 64, 6, 6, suffix)                 \
  X(64, 32, 32, 6, 5, suffix)                 \
  FOR_EACH_SUBPIX_VAR_64_128_RECT(X, suffix)

#if !CONFIG_REALTIME_ONLY
#define FOBMC_VAR(W, H, suffix)                                                \
  extern "C" unsigned int aom_obmc_variance##W##x##H##_##suffix(               \
      const uint8_t *pre, int pre_stride, const int32_t *wsrc,                 \
      const int32_t *mask, unsigned int *sse);                                 \
  HWY_ATTR unsigned int aom_obmc_variance##W##x##H##_##suffix(                 \
      const uint8_t *pre, int pre_stride, const int32_t *wsrc,                 \
      const int32_t *mask, unsigned int *sse) {                                \
    static_assert((W) >= 16, "FOBMC_VAR requires W >= 16.");                   \
    int sum = 0;                                                               \
    HWY_NAMESPACE::ObmcVarianceW16N(pre, pre_stride, wsrc, mask, sse, &sum, W, \
                                    H);                                        \
    return *sse - static_cast<unsigned int>(                                   \
                      (static_cast<int64_t>(sum) * sum) / ((W) * (H)));        \
  }

#define FOR_EACH_OBMC_VAR_BLOCK_SIZE(X, suffix) \
  X(128, 128, suffix)                           \
  X(128, 64, suffix)                            \
  X(64, 128, suffix)                            \
  X(64, 64, suffix)                             \
  X(64, 32, suffix)                             \
  X(64, 16, suffix)                             \
  X(32, 64, suffix)                             \
  X(32, 32, suffix)                             \
  X(32, 16, suffix)                             \
  X(32, 8, suffix)                              \
  X(16, 64, suffix)                             \
  X(16, 32, suffix)                             \
  X(16, 16, suffix)                             \
  X(16, 8, suffix)                              \
  X(16, 4, suffix)
#else
#define FOBMC_VAR(W, H, suffix)
#define FOR_EACH_OBMC_VAR_BLOCK_SIZE(X, suffix)
#endif  // !CONFIG_REALTIME_ONLY

#if CONFIG_AV1_HIGHBITDEPTH
#define FHIGHBD_COMP_MASK_PRED(suffix)                                        \
  extern "C" void aom_highbd_comp_mask_pred_##suffix(                         \
      uint8_t *comp_pred8, const uint8_t *pred8, int width, int height,       \
      const uint8_t *ref8, int ref_stride, const uint8_t *mask,               \
      int mask_stride, int invert_mask);                                      \
  HWY_ATTR void aom_highbd_comp_mask_pred_##suffix(                           \
      uint8_t *comp_pred8, const uint8_t *pred8, int width, int height,       \
      const uint8_t *ref8, int ref_stride, const uint8_t *mask,               \
      int mask_stride, int invert_mask) {                                     \
    HWY_NAMESPACE::HighbdCompMaskPred(comp_pred8, pred8, width, height, ref8, \
                                      ref_stride, mask, mask_stride,          \
                                      invert_mask);                           \
  }
#else
#define FHIGHBD_COMP_MASK_PRED(suffix)
#endif

#define FCOMP_AVG_MASK_PRED(suffix)                                          \
  extern "C" void aom_comp_avg_pred_##suffix(                                \
      uint8_t *comp_pred, const uint8_t *pred, int width, int height,        \
      const uint8_t *ref, int ref_stride);                                   \
  HWY_ATTR void aom_comp_avg_pred_##suffix(                                  \
      uint8_t *comp_pred, const uint8_t *pred, int width, int height,        \
      const uint8_t *ref, int ref_stride) {                                  \
    HWY_NAMESPACE::CompAvgPred(comp_pred, pred, width, height, ref,          \
                               ref_stride);                                  \
  }                                                                          \
  extern "C" void aom_comp_mask_pred_##suffix(                               \
      uint8_t *comp_pred, const uint8_t *pred, int width, int height,        \
      const uint8_t *ref, int ref_stride, const uint8_t *mask,               \
      int mask_stride, int invert_mask);                                     \
  HWY_ATTR void aom_comp_mask_pred_##suffix(                                 \
      uint8_t *comp_pred, const uint8_t *pred, int width, int height,        \
      const uint8_t *ref, int ref_stride, const uint8_t *mask,               \
      int mask_stride, int invert_mask) {                                    \
    HWY_NAMESPACE::CompMaskPred(comp_pred, pred, width, height, ref,         \
                                ref_stride, mask, mask_stride, invert_mask); \
  }                                                                          \
  FHIGHBD_COMP_MASK_PRED(suffix)

HWY_AFTER_NAMESPACE();

#endif  // AOM_AOM_DSP_VARIANCE_HWY_H_
