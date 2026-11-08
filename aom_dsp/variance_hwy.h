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

}  // namespace HWY_NAMESPACE
}  // namespace

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

HWY_AFTER_NAMESPACE();

#endif  // AOM_AOM_DSP_VARIANCE_HWY_H_
