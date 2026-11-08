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

#ifndef AOM_AV1_ENCODER_TEMPORAL_FILTER_HWY_H_
#define AOM_AV1_ENCODER_TEMPORAL_FILTER_HWY_H_

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <type_traits>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"

#include "aom_ports/mem.h"
#include "av1/encoder/encoder.h"
#include "av1/encoder/temporal_filter.h"
#include "third_party/highway/hwy/highway.h"

HWY_BEFORE_NAMESPACE();

namespace {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

constexpr int kSseStrideHwy = BW + 4;

template <bool kHighbd, typename D16, typename Pixel>
HWY_ATTR HWY_INLINE hn::Vec<D16> LoadNoiseRow(D16 di16, const Pixel *p) {
  if constexpr (kHighbd) {
    const hn::RebindToUnsigned<D16> du16;
    return hn::BitCast(di16, hn::LoadU(du16, p));
  } else {
    const hn::Rebind<uint8_t, D16> du8;
    return hn::PromoteTo(di16, hn::LoadU(du8, p));
  }
}

template <bool kHighbd,
          typename Pixel = std::conditional_t<kHighbd, uint16_t, uint8_t>>
HWY_ATTR HWY_INLINE double EstimateNoiseFromSinglePlaneHwy(
    const Pixel *src, int height, int width, int stride, int bit_depth,
    int edge_thresh) {
  if constexpr (kHighbd) {
    assert(bit_depth >= 8 && bit_depth <= 12);
  } else {
    (void)bit_depth;
  }
  if (width < 3 || height < 3 || edge_thresh <= 0) return -1.0;

  // Up to 12-bit pixels (0..4095) satisfy |gx|, |gy| <= 16380 and |v| <= 32760,
  // so both 8-bit and high-bit-depth planes fit in 16-bit lanes.
  const hn::ScalableTag<uint16_t> du16;
  const hn::RebindToSigned<decltype(du16)> di16;
  const hn::Repartition<int32_t, decltype(du16)> di32;
  const hn::Repartition<uint32_t, decltype(du16)> du32;
  const hn::Repartition<uint64_t, decltype(du16)> du64;
  const int step = static_cast<int>(hn::Lanes(du16));
  const int w_vec = (width - 2) & ~(step - 1);

  const int shift = kHighbd ? (bit_depth - 8) : 0;
  const auto v_edge_thresh =
      hn::Set(du16, static_cast<uint16_t>(AOMMIN(edge_thresh, 65535)));
  const auto one16 = hn::Set(di16, static_cast<int16_t>(1));

  auto total_sum64 = hn::Zero(du64);
  auto total_count64 = hn::Zero(du64);

  for (int w = 1; w < w_vec; w += step) {
    int h = 1;
    const int start_idx = h * stride + w;
    const int stride_0 = start_idx - stride;

    auto col_sum = hn::Zero(du32);
    auto col_count = hn::Zero(du32);

    const auto A = LoadNoiseRow<kHighbd>(di16, &src[stride_0 - 1]);
    const auto C = LoadNoiseRow<kHighbd>(di16, &src[stride_0 + 1]);
    const auto D = LoadNoiseRow<kHighbd>(di16, &src[start_idx - 1]);
    const auto F = LoadNoiseRow<kHighbd>(di16, &src[start_idx + 1]);
    auto B = LoadNoiseRow<kHighbd>(di16, &src[stride_0]);
    auto E = LoadNoiseRow<kHighbd>(di16, &src[start_idx]);

    auto sub_AC = hn::Sub(A, C);
    auto sum_AC = hn::Add(A, C);
    auto sub_DF = hn::Sub(D, F);
    auto sum_DF = hn::Add(D, F);

    for (; h < height - 1; ++h) {
      const int k = h * stride + w;
      const auto G = LoadNoiseRow<kHighbd>(di16, &src[k + stride - 1]);
      const auto H = LoadNoiseRow<kHighbd>(di16, &src[k + stride]);
      const auto I = LoadNoiseRow<kHighbd>(di16, &src[k + stride + 1]);

      const auto sub_GI = hn::Sub(G, I);
      const auto sum_GI = hn::Add(G, I);
      const auto sub_BH = hn::Sub(B, H);

      const auto gx = hn::BitCast(
          du16,
          hn::Abs(hn::Add(hn::Add(sub_AC, sub_GI), hn::ShiftLeft<1>(sub_DF))));
      const auto gy = hn::BitCast(
          du16,
          hn::Abs(hn::Add(hn::Sub(sum_AC, sum_GI), hn::ShiftLeft<1>(sub_BH))));
      auto ga = hn::Add(gx, gy);
      if constexpr (kHighbd) {
        const uint16_t round_val =
            shift > 0 ? static_cast<uint16_t>((1 << shift) >> 1) : 0;
        ga = hn::ShiftRightSame(hn::Add(ga, hn::Set(du16, round_val)), shift);
      }

      const auto mask_u16 = hn::Lt(ga, v_edge_thresh);
      if (!hn::AllFalse(du16, mask_u16)) {
        const auto mask_i16 = hn::RebindMask(di16, mask_u16);
        const auto sum_BH = hn::Add(B, H);
        const auto sum_DFBH = hn::ShiftLeft<1>(hn::Add(sum_DF, sum_BH));
        const auto sum_ACGI = hn::Add(sum_AC, sum_GI);
        const auto v =
            hn::Add(hn::Sub(hn::ShiftLeft<2>(E), sum_DFBH), sum_ACGI);
        auto abs_v = hn::BitCast(du16, hn::Abs(v));
        if constexpr (kHighbd) {
          const uint16_t round_val =
              shift > 0 ? static_cast<uint16_t>((1 << shift) >> 1) : 0;
          abs_v = hn::ShiftRightSame(hn::Add(abs_v, hn::Set(du16, round_val)),
                                     shift);
        }
        const auto var = hn::IfThenElseZero(mask_i16, hn::BitCast(di16, abs_v));
        const auto cnt = hn::IfThenElseZero(mask_i16, one16);
        col_sum = hn::Add(col_sum, hn::BitCast(du32, hn::WidenMulPairwiseAdd(
                                                         di32, var, one16)));
        col_count = hn::Add(
            col_count,
            hn::BitCast(du32, hn::WidenMulPairwiseAdd(di32, cnt, one16)));
      }

      sub_AC = sub_DF;
      sub_DF = sub_GI;
      sum_AC = sum_DF;
      sum_DF = sum_GI;
      B = E;
      E = H;
    }

    total_sum64 =
        hn::Add(total_sum64, hn::Add(hn::PromoteEvenTo(du64, col_sum),
                                     hn::PromoteOddTo(du64, col_sum)));
    total_count64 =
        hn::Add(total_count64, hn::Add(hn::PromoteEvenTo(du64, col_count),
                                       hn::PromoteOddTo(du64, col_count)));
  }

  int64_t accum = static_cast<int64_t>(hn::ReduceSum(du64, total_sum64));
  int64_t count = static_cast<int64_t>(hn::ReduceSum(du64, total_count64));

  for (int h = 1; h < height - 1; ++h) {
    for (int w = w_vec + 1; w < width - 1; ++w) {
      const int k = h * stride + w;
      const int gx = (src[k - stride - 1] - src[k - stride + 1]) +
                     (src[k + stride - 1] - src[k + stride + 1]) +
                     2 * (src[k - 1] - src[k + 1]);
      const int gy = (src[k - stride - 1] - src[k + stride - 1]) +
                     (src[k - stride + 1] - src[k + stride + 1]) +
                     2 * (src[k - stride] - src[k + stride]);
      int ga = abs(gx) + abs(gy);
      if constexpr (kHighbd) {
        ga = ROUND_POWER_OF_TWO(ga, shift);
      }
      if (ga < edge_thresh) {
        const int v =
            4 * src[k] -
            2 * (src[k - 1] + src[k + 1] + src[k - stride] + src[k + stride]) +
            (src[k - stride - 1] + src[k - stride + 1] + src[k + stride - 1] +
             src[k + stride + 1]);
        int abs_v = abs(v);
        if constexpr (kHighbd) {
          abs_v = ROUND_POWER_OF_TWO(abs_v, shift);
        }
        accum += abs_v;
        ++count;
      }
    }
  }

  return (count < 16) ? -1.0
                      : static_cast<double>(accum) /
                            (6.0 * static_cast<double>(count)) * SQRT_PI_BY_2;
}

template <typename Pixel, typename SseType>
HWY_INLINE HWY_ATTR void GetSquaredErrorHwy(
    const Pixel *frame1, const unsigned int stride, const Pixel *frame2,
    const unsigned int stride2, const int block_width, const int block_height,
    SseType *frame_sse, const unsigned int sse_stride) {
  const Pixel *src1 = frame1;
  const Pixel *src2 = frame2;
  SseType *dst = frame_sse + 2;
  using SignedDiff = std::make_signed_t<SseType>;
  const hn::ScalableTag<SignedDiff> d_diff;
  const hn::Rebind<Pixel, decltype(d_diff)> d_pix;
  const hn::Rebind<SseType, decltype(d_diff)> d_sse;
  const int step = static_cast<int>(hn::Lanes(d_diff));
  assert(block_width % step == 0);

  for (int i = 0; i < block_height; i++) {
    for (int j = 0; j < block_width; j += step) {
      const auto s1 = hn::PromoteTo(d_diff, hn::LoadU(d_pix, src1 + j));
      const auto s2 = hn::PromoteTo(d_diff, hn::LoadU(d_pix, src2 + j));
      const auto diff = hn::Sub(s1, s2);
      const auto res = hn::BitCast(d_sse, hn::Mul(diff, diff));
      hn::StoreU(res, d_sse, dst + j);
    }
    dst[-2] = dst[-1] = dst[0];
    dst[block_width] = dst[block_width + 1] = dst[block_width - 1];
    src1 += stride;
    src2 += stride2;
    dst += sse_stride;
  }
}

template <typename DU32, typename SseType>
HWY_INLINE HWY_ATTR hn::Vec<DU32> LoadSseAsU32(DU32 du32, const SseType *p) {
  if constexpr (sizeof(SseType) == 2) {
    const hn::Rebind<uint16_t, DU32> du16;
    return hn::PromoteTo(du32, hn::LoadU(du16, p));
  } else {
    return hn::LoadU(du32, p);
  }
}

template <typename SseType>
HWY_INLINE HWY_ATTR hn::Vec<hn::ScalableTag<uint32_t>> LoadAndSum5RowHwy(
    const SseType *src) {
  const hn::ScalableTag<uint32_t> du32;
  const auto c0 = LoadSseAsU32(du32, src - 2);
  const auto c1 = LoadSseAsU32(du32, src - 1);
  const auto c2 = LoadSseAsU32(du32, src);
  const auto c3 = LoadSseAsU32(du32, src + 1);
  const auto c4 = LoadSseAsU32(du32, src + 2);
  return hn::Add(hn::Add(c0, c1), hn::Add(hn::Add(c2, c3), c4));
}

template <class DF32>
HWY_INLINE HWY_ATTR hn::Vec<DF32> ApproxExpHwy(DF32 df32, hn::Vec<DF32> y) {
  const hn::RebindToSigned<DF32> di32;
  constexpr float kA = (1 << 23) / 0.69314718056f;
  constexpr int32_t kB = 127;
  constexpr int32_t kC = 60801;
  const auto multiplier = hn::Set(df32, kA);
  const auto offset = hn::Set(di32, kB * (1 << 23) - kC);

  y = hn::Mul(y, multiplier);
  y = hn::BitCast(df32, hn::Add(hn::ConvertInRangeTo(di32, y), offset));
  return y;
}

// Double-precision implementation of exp(-x) for x >= 0 (clamped to [0, 7])
// using Cody-Waite range reduction and a degree-12 Taylor polynomial.
template <class DF64>
HWY_INLINE HWY_ATTR hn::Vec<DF64> ExactExpNegPdHwy(DF64 df64, hn::Vec<DF64> x) {
  const hn::RebindToSigned<DF64> di64;
  x = hn::Min(x, hn::Set(df64, 7.0));
  const auto inv_ln2 = hn::Set(df64, 1.44269504088896340736);
  const auto ln2_hi = hn::Set(df64, 0.6931471805596629565116);
  const auto ln2_lo = hn::Set(df64, 0.2823529056303157712259e-12);
  const auto k = hn::Round(hn::Mul(x, inv_ln2));
  const auto r = hn::MulAdd(k, ln2_lo, hn::Sub(hn::Mul(k, ln2_hi), x));

  auto p = hn::Set(df64, 1.0 / 479001600.0);
  p = hn::MulAdd(p, r, hn::Set(df64, 1.0 / 39916800.0));
  p = hn::MulAdd(p, r, hn::Set(df64, 1.0 / 3628800.0));
  p = hn::MulAdd(p, r, hn::Set(df64, 1.0 / 362880.0));
  p = hn::MulAdd(p, r, hn::Set(df64, 1.0 / 40320.0));
  p = hn::MulAdd(p, r, hn::Set(df64, 1.0 / 5040.0));
  p = hn::MulAdd(p, r, hn::Set(df64, 1.0 / 720.0));
  p = hn::MulAdd(p, r, hn::Set(df64, 1.0 / 120.0));
  p = hn::MulAdd(p, r, hn::Set(df64, 1.0 / 24.0));
  p = hn::MulAdd(p, r, hn::Set(df64, 1.0 / 6.0));
  p = hn::MulAdd(p, r, hn::Set(df64, 0.5));
  p = hn::MulAdd(p, r, hn::Set(df64, 1.0));
  p = hn::MulAdd(p, r, hn::Set(df64, 1.0));

  const auto ki64 = hn::ConvertTo(di64, k);
  const auto pow2 =
      hn::BitCast(df64, hn::ShiftLeft<52>(hn::Sub(hn::Set(di64, 1023), ki64)));
  return hn::Mul(p, pow2);
}

template <bool kHighbd, int kWgtCalcLvl, bool kHasLumaSse,
          typename Pixel = std::conditional_t<kHighbd, uint16_t, uint8_t>>
HWY_INLINE HWY_ATTR void AccumulateTemporalFilterWeights(
    const Pixel *frame2, const unsigned int stride2, const int block_width,
    const int block_height, const uint32_t acc_5x5_sse[BH][BW],
    const uint32_t *luma_sse_sum, int bd, const double inv_num_ref_pixels,
    const double weight_factor, const double subblock_mses_scaled[NUM_16X16],
    const double d_factor_decayed[NUM_16X16], unsigned int *accumulator,
    uint16_t *count) {
  const hn::ScalableTag<double> df64;
  const hn::Rebind<int32_t, decltype(df64)> di32_half;
  const hn::Rebind<uint32_t, decltype(df64)> du32_half;
  const hn::Rebind<uint16_t, decltype(df64)> du16_qtr;
  const hn::Rebind<Pixel, decltype(df64)> d_pix;
  const int wgt_step = static_cast<int>(hn::Lanes(df64));
  assert(wgt_step <= (block_width >> 2));
  assert(block_width % wgt_step == 0);

  const int bd_shift = kHighbd ? (bd - 8) * 2 : 0;
  (void)bd;
  const auto inv_num_ref_pixel_vec = hn::Set(df64, inv_num_ref_pixels);
  const auto weight_factor_vec = hn::Set(df64, weight_factor);

  for (int i = 0; i < block_height; i++) {
    const int y32_blk_raster_offset = (i >= (block_height >> 1)) << 1;
    const int y16_blk_raster_offset =
        ((i % (block_height >> 1)) >= (block_height >> 2)) << 1;
    const uint32_t *luma_sse_sum_row =
        kHasLumaSse ? (luma_sse_sum + i * BW) : nullptr;
    for (int j = 0; j < block_width; j += wgt_step) {
      auto diff_sse_u32 = hn::LoadU(du32_half, acc_5x5_sse[i] + j);
      if constexpr (kHasLumaSse) {
        diff_sse_u32 =
            hn::Add(diff_sse_u32, hn::LoadU(du32_half, luma_sse_sum_row + j));
      }
      if constexpr (kHighbd) {
        diff_sse_u32 = hn::ShiftRightSame(diff_sse_u32, bd_shift);
      }
      const auto diff_sse_pd =
          hn::PromoteTo(df64, hn::BitCast(di32_half, diff_sse_u32));
      const auto window_error = hn::Mul(diff_sse_pd, inv_num_ref_pixel_vec);

      const int x32_blk_raster_offset = (j >= (block_width >> 1));
      const int x16_blk_raster_offset =
          ((j % (block_width >> 1)) >= (block_width >> 2));
      const int subblock_idx =
          ((y32_blk_raster_offset + x32_blk_raster_offset) << 2) +
          (y16_blk_raster_offset + x16_blk_raster_offset);

      const auto combined_error =
          hn::MulAdd(window_error, weight_factor_vec,
                     hn::Set(df64, subblock_mses_scaled[subblock_idx]));
      const auto scaled_error = hn::Mul(
          combined_error, hn::Set(df64, d_factor_decayed[subblock_idx]));

      hn::Vec<decltype(di32_half)> weights_in_32bit;
      if constexpr (kWgtCalcLvl == 0) {
        const auto tf_weight_scale_pd =
            hn::Set(df64, static_cast<double>(TF_WEIGHT_SCALE));
        weights_in_32bit = hn::DemoteTo(
            di32_half,
            hn::Mul(ExactExpNegPdHwy(df64, scaled_error), tf_weight_scale_pd));
      } else {
        const hn::Rebind<float, decltype(df64)> df32;
        const auto tf_weight_scale_ps =
            hn::Set(df32, static_cast<float>(TF_WEIGHT_SCALE));
        const auto point_five = hn::Set(df32, 0.5f);
        const auto seven = hn::Set(df32, 7.0f);
        const auto scaled_error_ps = hn::DemoteTo(df32, scaled_error);
        const auto scaled_diff_ps = hn::Min(scaled_error_ps, seven);
        const auto exp_result = ApproxExpHwy(df32, hn::Neg(scaled_diff_ps));
        const auto round_result =
            hn::MulAdd(exp_result, tf_weight_scale_ps, point_five);
        weights_in_32bit = hn::ConvertInRangeTo(di32_half, round_result);
      }

      const auto weights_in_16bit = hn::DemoteTo(du16_qtr, weights_in_32bit);

      const int stride_idx = i * stride2 + j;
      const auto count_array = hn::LoadU(du16_qtr, count + stride_idx);
      hn::StoreU(hn::Add(count_array, weights_in_16bit), du16_qtr,
                 count + stride_idx);

      const auto accumulator_array =
          hn::LoadU(du32_half, accumulator + stride_idx);
      const auto pred_values = hn::LoadU(d_pix, frame2 + stride_idx);
      const auto pred_values_u32 = hn::PromoteTo(du32_half, pred_values);
      const auto mul_frame2_weight_u32 =
          hn::Mul(pred_values_u32, hn::BitCast(du32_half, weights_in_32bit));
      hn::StoreU(hn::Add(accumulator_array, mul_frame2_weight_u32), du32_half,
                 accumulator + stride_idx);
    }
  }
}

template <bool kHighbd,
          typename Pixel = std::conditional_t<kHighbd, uint16_t, uint8_t>,
          typename SseType = std::conditional_t<kHighbd, uint32_t, uint16_t>>
HWY_INLINE HWY_ATTR void ApplyTemporalFilterPlaneHwy(
    const Pixel *frame1, const unsigned int stride, const Pixel *frame2,
    const unsigned int stride2, const int block_width, const int block_height,
    const int *subblock_mses, unsigned int *accumulator, uint16_t *count,
    SseType *frame_sse, const uint32_t *luma_sse_sum, int bd,
    const double inv_num_ref_pixels, const double decay_factor,
    const double inv_factor, const double weight_factor, const double *d_factor,
    int tf_wgt_calc_lvl) {
  assert(((block_width == 64) || (block_width == 32)) &&
         ((block_height == 64) || (block_height == 32)));

  DECLARE_ALIGNED(64, uint32_t, acc_5x5_sse[BH][BW]);

  GetSquaredErrorHwy(frame1, stride, frame2, stride2, block_width, block_height,
                     frame_sse, kSseStrideHwy);

  const hn::ScalableTag<uint32_t> du32_vec;
  const int col_step = static_cast<int>(hn::Lanes(du32_vec));
  assert(block_width % col_step == 0);
  hn::Vec<hn::ScalableTag<uint32_t>> vsrc[5];

  for (int col = 0; col < block_width; col += col_step) {
    const SseType *src = frame_sse + 2 + col;

    for (int i = 2; i < 5; i++) {
      vsrc[i] = LoadAndSum5RowHwy(src);
      src += kSseStrideHwy;
    }

    vsrc[0] = vsrc[2];
    vsrc[1] = vsrc[2];

    for (int row = 0; row < block_height; row++) {
      const auto vsum01 = hn::Add(vsrc[0], vsrc[1]);
      const auto vsum23 = hn::Add(vsrc[2], vsrc[3]);
      const auto vsum = hn::Add(hn::Add(vsum01, vsum23), vsrc[4]);

      hn::StoreU(vsum, du32_vec, &acc_5x5_sse[row][col]);

      vsrc[0] = vsrc[1];
      vsrc[1] = vsrc[2];
      vsrc[2] = vsrc[3];
      vsrc[3] = vsrc[4];

      if (row <= block_height - 4) {
        vsrc[4] = LoadAndSum5RowHwy(src);
        src += kSseStrideHwy;
      } else {
        vsrc[4] = vsrc[3];
      }
    }
  }

  double subblock_mses_scaled[NUM_16X16];
  double d_factor_decayed[NUM_16X16];
  for (int idx = 0; idx < NUM_16X16; idx++) {
    subblock_mses_scaled[idx] = subblock_mses[idx] * inv_factor;
    d_factor_decayed[idx] = d_factor[idx] * decay_factor;
  }

  if (luma_sse_sum == nullptr) {
    if (tf_wgt_calc_lvl == 0) {
      AccumulateTemporalFilterWeights<kHighbd, 0, false>(
          frame2, stride2, block_width, block_height, acc_5x5_sse, nullptr, bd,
          inv_num_ref_pixels, weight_factor, subblock_mses_scaled,
          d_factor_decayed, accumulator, count);
    } else {
      AccumulateTemporalFilterWeights<kHighbd, 1, false>(
          frame2, stride2, block_width, block_height, acc_5x5_sse, nullptr, bd,
          inv_num_ref_pixels, weight_factor, subblock_mses_scaled,
          d_factor_decayed, accumulator, count);
    }
  } else {
    if (tf_wgt_calc_lvl == 0) {
      AccumulateTemporalFilterWeights<kHighbd, 0, true>(
          frame2, stride2, block_width, block_height, acc_5x5_sse, luma_sse_sum,
          bd, inv_num_ref_pixels, weight_factor, subblock_mses_scaled,
          d_factor_decayed, accumulator, count);
    } else {
      AccumulateTemporalFilterWeights<kHighbd, 1, true>(
          frame2, stride2, block_width, block_height, acc_5x5_sse, luma_sse_sum,
          bd, inv_num_ref_pixels, weight_factor, subblock_mses_scaled,
          d_factor_decayed, accumulator, count);
    }
  }
}

template <bool kHighbd>
HWY_ATTR HWY_INLINE void ApplyTemporalFilterHwy(
    const YV12_BUFFER_CONFIG *frame_to_filter, const MACROBLOCKD *mbd,
    const BLOCK_SIZE block_size, const int mb_row, const int mb_col,
    const int num_planes, const double *noise_levels, const MV *subblock_mvs,
    const int *subblock_mses, const int q_factor, const int filter_strength,
    int tf_wgt_calc_lvl, const uint8_t *pred, uint32_t *accum,
    uint16_t *count) {
  using Pixel = std::conditional_t<kHighbd, uint16_t, uint8_t>;
  using SseType = std::conditional_t<kHighbd, uint32_t, uint16_t>;

  const int is_high_bitdepth = frame_to_filter->flags & YV12_FLAG_HIGHBITDEPTH;
  assert(block_size == BLOCK_64X64 && "Only support 64x64 block!");
  assert(TF_WINDOW_LENGTH == 5 && "Only support window length 5!");
  if constexpr (kHighbd) {
    assert(is_high_bitdepth && "Only support high bit-depth!");
    assert(mbd->bd >= 8 && mbd->bd <= 12);
  } else {
    assert(!is_high_bitdepth && "Only support low bit-depth!");
  }
  assert(num_planes >= 1 && num_planes <= MAX_MB_PLANE);
  (void)is_high_bitdepth;

  const int mb_height = block_size_high[block_size];
  const int mb_width = block_size_wide[block_size];
  const int frame_height = frame_to_filter->y_crop_height;
  const int frame_width = frame_to_filter->y_crop_width;
  const int min_frame_size = AOMMIN(frame_height, frame_width);
  const double inv_factor = 1.0 / ((TF_WINDOW_BLOCK_BALANCE_WEIGHT + 1) *
                                   TF_SEARCH_ERROR_NORM_WEIGHT);
  const double weight_factor =
      static_cast<double>(TF_WINDOW_BLOCK_BALANCE_WEIGHT) * inv_factor;
  double q_decay = pow(static_cast<double>(q_factor) / TF_Q_DECAY_THRESHOLD, 2);
  q_decay = CLIP(q_decay, 1e-5, 1);
  if (q_factor >= TF_QINDEX_CUTOFF) {
    q_decay = 0.5 * pow(static_cast<double>(q_factor) / 64, 2);
  }
  double s_decay =
      pow(static_cast<double>(filter_strength) / TF_STRENGTH_THRESHOLD, 2);
  s_decay = CLIP(s_decay, 1e-5, 1);
  double d_factor[NUM_16X16] = { 0 };
  DECLARE_ALIGNED(64, SseType, frame_sse[kSseStrideHwy * BH]);
  DECLARE_ALIGNED(64, uint32_t, luma_sse_sum[BW * BH]);
  const Pixel *pred_ptr;
  if constexpr (kHighbd) {
    pred_ptr = CONVERT_TO_SHORTPTR(pred);
  } else {
    pred_ptr = pred;
  }

  double distance_threshold = min_frame_size * TF_SEARCH_DISTANCE_THRESHOLD;
  distance_threshold = AOMMAX(distance_threshold, 1);
  for (int subblock_idx = 0; subblock_idx < NUM_16X16; subblock_idx++) {
    const MV mv = subblock_mvs[subblock_idx];
    const double distance = sqrt(pow(mv.row, 2) + pow(mv.col, 2));
    d_factor[subblock_idx] = distance / distance_threshold;
    d_factor[subblock_idx] = AOMMAX(d_factor[subblock_idx], 1);
  }

  const int bd = kHighbd ? mbd->bd : 8;
  int plane_offset = 0;
  for (int plane = 0; plane < num_planes; ++plane) {
    const uint32_t plane_h = mb_height >> mbd->plane[plane].subsampling_y;
    const uint32_t plane_w = mb_width >> mbd->plane[plane].subsampling_x;
    const uint32_t frame_stride = frame_to_filter->strides[plane == 0 ? 0 : 1];
    const int frame_offset = mb_row * plane_h * frame_stride + mb_col * plane_w;

    const Pixel *ref;
    if constexpr (kHighbd) {
      ref = CONVERT_TO_SHORTPTR(frame_to_filter->buffers[plane]) + frame_offset;
    } else {
      ref = frame_to_filter->buffers[plane] + frame_offset;
    }
    const int ss_x_shift =
        mbd->plane[plane].subsampling_x - mbd->plane[AOM_PLANE_Y].subsampling_x;
    const int ss_y_shift =
        mbd->plane[plane].subsampling_y - mbd->plane[AOM_PLANE_Y].subsampling_y;
    const int num_ref_pixels = TF_WINDOW_LENGTH * TF_WINDOW_LENGTH +
                               ((plane) ? (1 << (ss_x_shift + ss_y_shift)) : 0);
    const double inv_num_ref_pixels = 1.0 / num_ref_pixels;
    const double n_decay = 0.5 + log(2 * noise_levels[plane] + 5.0);
    const double decay_factor = 1 / (n_decay * q_decay * s_decay);

    if (plane == AOM_PLANE_U) {
      if (ss_x_shift == 1 && ss_y_shift == 1) {
        const hn::ScalableTag<uint32_t> du32_vec;
        const unsigned int step =
            static_cast<unsigned int>(hn::Lanes(du32_vec));
        assert(plane_w % step == 0);
        if constexpr (kHighbd) {
          for (unsigned int i = 0; i < plane_h; i++) {
            const uint32_t *src_0 = &frame_sse[2 * i * kSseStrideHwy + 2];
            const uint32_t *src_1 = &frame_sse[(2 * i + 1) * kSseStrideHwy + 2];
            for (unsigned int j = 0; j < plane_w; j += step) {
              const auto r0_lo = hn::LoadU(du32_vec, src_0 + 2 * j);
              const auto r0_hi = hn::LoadU(du32_vec, src_0 + 2 * j + step);
              const auto r1_lo = hn::LoadU(du32_vec, src_1 + 2 * j);
              const auto r1_hi = hn::LoadU(du32_vec, src_1 + 2 * j + step);
              const auto s_lo = hn::Add(r0_lo, r1_lo);
              const auto s_hi = hn::Add(r0_hi, r1_hi);
              const auto evens = hn::ConcatEven(du32_vec, s_hi, s_lo);
              const auto odds = hn::ConcatOdd(du32_vec, s_hi, s_lo);
              hn::StoreU(hn::Add(evens, odds), du32_vec,
                         &luma_sse_sum[i * BW + j]);
            }
          }
        } else {
          const hn::ScalableTag<uint16_t> du16_vec;
          const auto mask_lo16 = hn::Set(du32_vec, 0x0000FFFFu);
          for (unsigned int i = 0; i < plane_h; i++) {
            const uint16_t *src_0 = &frame_sse[2 * i * kSseStrideHwy + 2];
            const uint16_t *src_1 = &frame_sse[(2 * i + 1) * kSseStrideHwy + 2];
            for (unsigned int j = 0; j < plane_w; j += step) {
              const auto r0 =
                  hn::BitCast(du32_vec, hn::LoadU(du16_vec, src_0 + j * 2));
              const auto r1 =
                  hn::BitCast(du32_vec, hn::LoadU(du16_vec, src_1 + j * 2));
              const auto sum0 =
                  hn::Add(hn::And(r0, mask_lo16), hn::ShiftRight<16>(r0));
              const auto sum1 =
                  hn::Add(hn::And(r1, mask_lo16), hn::ShiftRight<16>(r1));
              hn::StoreU(hn::Add(sum0, sum1), du32_vec,
                         &luma_sse_sum[i * BW + j]);
            }
          }
        }
      } else {
        for (unsigned int i = 0; i < plane_h; i++) {
          for (unsigned int j = 0; j < plane_w; j++) {
            uint32_t sum = 0;
            for (int ii = 0; ii < (1 << ss_y_shift); ++ii) {
              for (int jj = 0; jj < (1 << ss_x_shift); ++jj) {
                const int yy = (i << ss_y_shift) + ii;
                const int xx = (j << ss_x_shift) + jj;
                sum += frame_sse[yy * kSseStrideHwy + xx + 2];
              }
            }
            luma_sse_sum[i * BW + j] = sum;
          }
        }
      }
    }

    ApplyTemporalFilterPlaneHwy<kHighbd>(
        ref, frame_stride, pred_ptr + plane_offset, plane_w, plane_w, plane_h,
        subblock_mses, accum + plane_offset, count + plane_offset, frame_sse,
        plane == AOM_PLANE_Y ? nullptr : luma_sse_sum, bd, inv_num_ref_pixels,
        decay_factor, inv_factor, weight_factor, d_factor, tf_wgt_calc_lvl);
    plane_offset += plane_h * plane_w;
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace

HWY_AFTER_NAMESPACE();

#define MAKE_TEMPORAL_FILTER(suffix)                                          \
  extern "C" HWY_ATTR double av1_estimate_noise_from_single_plane_##suffix(   \
      const uint8_t *src, int height, int width, int stride,                  \
      int edge_thresh) {                                                      \
    return HWY_NAMESPACE::EstimateNoiseFromSinglePlaneHwy<false>(             \
        src, height, width, stride, 8, edge_thresh);                          \
  }                                                                           \
  extern "C" HWY_ATTR void av1_apply_temporal_filter_##suffix(                \
      const YV12_BUFFER_CONFIG *frame_to_filter, const MACROBLOCKD *mbd,      \
      const BLOCK_SIZE block_size, const int mb_row, const int mb_col,        \
      const int num_planes, const double *noise_levels,                       \
      const MV *subblock_mvs, const int *subblock_mses, const int q_factor,   \
      const int filter_strength, int tf_wgt_calc_lvl, const uint8_t *pred,    \
      uint32_t *accum, uint16_t *count) {                                     \
    HWY_NAMESPACE::ApplyTemporalFilterHwy<false>(                             \
        frame_to_filter, mbd, block_size, mb_row, mb_col, num_planes,         \
        noise_levels, subblock_mvs, subblock_mses, q_factor, filter_strength, \
        tf_wgt_calc_lvl, pred, accum, count);                                 \
  }

#if CONFIG_AV1_HIGHBITDEPTH
#define MAKE_HIGHBD_TEMPORAL_FILTER(suffix)                                   \
  extern "C" HWY_ATTR double                                                  \
  av1_highbd_estimate_noise_from_single_plane_##suffix(                       \
      const uint16_t *src, int height, int width, int stride, int bit_depth,  \
      int edge_thresh) {                                                      \
    return HWY_NAMESPACE::EstimateNoiseFromSinglePlaneHwy<true>(              \
        src, height, width, stride, bit_depth, edge_thresh);                  \
  }                                                                           \
  extern "C" HWY_ATTR void av1_highbd_apply_temporal_filter_##suffix(         \
      const YV12_BUFFER_CONFIG *frame_to_filter, const MACROBLOCKD *mbd,      \
      const BLOCK_SIZE block_size, const int mb_row, const int mb_col,        \
      const int num_planes, const double *noise_levels,                       \
      const MV *subblock_mvs, const int *subblock_mses, const int q_factor,   \
      const int filter_strength, int tf_wgt_calc_lvl, const uint8_t *pred,    \
      uint32_t *accum, uint16_t *count) {                                     \
    HWY_NAMESPACE::ApplyTemporalFilterHwy<true>(                              \
        frame_to_filter, mbd, block_size, mb_row, mb_col, num_planes,         \
        noise_levels, subblock_mvs, subblock_mses, q_factor, filter_strength, \
        tf_wgt_calc_lvl, pred, accum, count);                                 \
  }
#endif  // CONFIG_AV1_HIGHBITDEPTH

#endif  // AOM_AV1_ENCODER_TEMPORAL_FILTER_HWY_H_
