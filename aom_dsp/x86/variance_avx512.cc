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

#define HWY_BASELINE_TARGETS HWY_AVX3
#define HWY_BROKEN_32BIT 0

#include "aom_dsp/reduce_sum_hwy.h"
#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"
#include "third_party/highway/hwy/highway.h"

HWY_BEFORE_NAMESPACE();

namespace {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

#if HWY_CXX_LANG >= 201703L
#define CONSTEXPR_IF constexpr
#else
#define CONSTEXPR_IF
#endif

template <int Lines, typename DL, typename DZ>
HWY_ATTR HWY_INLINE hn::VFromD<DZ> LoadLines(DZ extend_tag, DL load_tag,
                                             const hn::TFromD<DL> *src,
                                             int stride) {
  constexpr size_t kNumBytes = sizeof(*src) * hn::MaxLanes(load_tag) / Lines;
  HWY_ALIGN_MAX hn::TFromD<DZ> data[hn::MaxLanes(extend_tag)] = {};
  for (int line = 0; line < Lines; ++line) {
    hwy::CopyBytes<kNumBytes>(src + line * stride, data + line * kNumBytes);
  }
  return hn::Load(extend_tag, data);
}

template <int Width, int Height>
HWY_ATTR HWY_INLINE uint32_t Variance(const uint8_t *a, int a_stride,
                                      const uint8_t *b, int b_stride,
                                      uint32_t *HWY_RESTRICT sse) {
  constexpr int kOpportunisticWidth = Width * 2;
  constexpr hn::CappedTag<uint8_t, Width> load_tag;
  constexpr hn::CappedTag<uint8_t, kOpportunisticWidth> lines_tag;
  // Extend vectors to the natural vector width if they're shorter.
  constexpr hn::CappedTag<uint8_t, AOMMAX(16, hn::MaxLanes(lines_tag))>
      extended_lines_tag;
  constexpr hn::Repartition<int16_t, decltype(extended_lines_tag)> sum_tag;
  constexpr hn::Repartition<int32_t, decltype(extended_lines_tag)> wide_tag;
  constexpr int kLinesPerIteration =
      hn::MaxLanes(lines_tag) / hn::MaxLanes(load_tag);
  // 2^7 signed 8-bit integer pairs can be accumulated into a 16-bit signed
  // integer without precision loss.
  constexpr int kPrecisionAddsPerLane = 128;
  constexpr int kElementsPerLoad = hn::MaxLanes(lines_tag) / 2;
  constexpr int kPrecisionLines =
      AOMMIN(Height, (kElementsPerLoad * kPrecisionAddsPerLane) / Width);
  constexpr bool kIntermediateSummation = Height > kPrecisionLines;
  constexpr int kOptimalBufferWidth = 256;
  constexpr int kStoresPerLoad = hn::MaxLanes(lines_tag) <= 8 ? 1 : 2;
  constexpr int kAlignedWidth = AOMMAX(kStoresPerLoad == 1 ? 4 : 8, Width);
  constexpr int kLineBufferWidth =
      AOMMIN(kAlignedWidth * Height, kOptimalBufferWidth);
  const auto adj_sub = hn::BitCast(
      hn::RebindToSigned<decltype(extended_lines_tag)>(),
      hn::Set(hn::Repartition<uint16_t, decltype(extended_lines_tag)>(),
              0xff01));
  auto sumw = hn::Zero(wide_tag);
  auto sumn0 = hn::Zero(sum_tag);
  auto ssev0 = hn::Zero(wide_tag);
  auto ssev1 = hn::Zero(wide_tag);
  for (int y = 0; y < Height;) {
    constexpr int kBufferedRows =
        kLineBufferWidth / (kAlignedWidth * kLinesPerIteration);
    static_assert(kBufferedRows > 0, "Somehow 0 rows are being buffered.");
    for (int yp = 0; yp < kPrecisionLines;) {
      HWY_ALIGN_MAX int16_t diff[kLineBufferWidth];
      int n = 0;
      for (int yb = 0; yb < kBufferedRows;
           ++yb, yp += kLinesPerIteration, y += kLinesPerIteration) {
        assert(yp < Height);
        assert(y < Height);
        for (int x = 0; x < Width; x += hn::MaxLanes(load_tag),
                 n += kStoresPerLoad * hn::MaxLanes(sum_tag)) {
          const auto av = LoadLines<kLinesPerIteration>(
              extended_lines_tag, lines_tag, a + x, a_stride);
          const auto bv = LoadLines<kLinesPerIteration>(
              extended_lines_tag, lines_tag, b + x, b_stride);
          if CONSTEXPR_IF (kStoresPerLoad == 1) {
            // Unpack into pairs of source and reference values.
            const auto abl = hn::InterleaveLower(extended_lines_tag, av, bv);
            // Subtract adjacent elements using src*1 + ref*-1.
            const auto diffl =
                hn::SatWidenMulPairwiseAdd(sum_tag, abl, adj_sub);
            hn::Store(diffl, sum_tag, &diff[n]);
          } else {
            // Unpack into pairs of source and reference values.
            const auto abl = hn::InterleaveLower(extended_lines_tag, av, bv);
            const auto abh = hn::InterleaveUpper(extended_lines_tag, av, bv);
            // Subtract adjacent elements using src*1 + ref*-1.
            const auto diffl =
                hn::SatWidenMulPairwiseAdd(sum_tag, abl, adj_sub);
            const auto diffh =
                hn::SatWidenMulPairwiseAdd(sum_tag, abh, adj_sub);
            hn::Store(diffl, sum_tag, &diff[n]);
            hn::Store(diffh, sum_tag, &diff[n + hn::MaxLanes(sum_tag)]);
          }
        }
        a += a_stride * kLinesPerIteration;
        b += b_stride * kLinesPerIteration;
      }
      assert(n == kLineBufferWidth);
      for (int x = 0; x < kLineBufferWidth; x += hn::MaxLanes(sum_tag)) {
        const auto d0 = hn::Load(sum_tag, &diff[x]);
        ssev0 = hn::ReorderWidenMulAccumulate(wide_tag, d0, d0, ssev0, ssev1);
        sumn0 = hn::Add(sumn0, d0);
      }
    }
    if CONSTEXPR_IF (kIntermediateSummation) {
      sumw = hn::Add(sumw, hn::PromoteLowerTo(wide_tag, sumn0));
      sumw = hn::Add(sumw, hn::PromoteUpperTo(wide_tag, sumn0));
      sumn0 = hn::Zero(sum_tag);
    }
  }
  ssev0 = hn::Add(ssev0, ssev1);
  constexpr hn::BlockDFromD<decltype(wide_tag)> wide_block_tag;
  if CONSTEXPR_IF (!kIntermediateSummation) {
    sumw = hn::PromoteLowerTo(wide_tag, sumn0);
    sumw = hn::Add(sumw, hn::PromoteUpperTo(wide_tag, sumn0));
  }
  auto sump = BlockReduceSum(wide_tag, sumw);
  auto ssep = BlockReduceSum(wide_tag, ssev0);
  auto sse_sum_lo = hn::InterleaveLower(wide_block_tag, ssep, sump);
  auto sse_sum_hi = hn::InterleaveUpper(wide_block_tag, ssep, sump);
  auto sse_sum = hn::Add(sse_sum_lo, sse_sum_hi);
  auto res = hn::Add(sse_sum, hn::ShiftRightLanes<2>(wide_block_tag, sse_sum));
  *sse = hn::ExtractLane(res, 0);
  int32_t sum = hn::ExtractLane(res, 1);
  return static_cast<uint32_t>(
      *sse - ((static_cast<int64_t>(sum) * sum) / (Width * Height)));
}

}  // namespace HWY_NAMESPACE
}  // namespace

#define MAKE_VARIANCE(w, h, suffix)                                      \
  extern "C" uint32_t aom_variance##w##x##h##_##suffix(                  \
      const uint8_t *a, int a_stride, const uint8_t *b, int b_stride,    \
      uint32_t *sse);                                                    \
  HWY_ATTR uint32_t aom_variance##w##x##h##_##suffix(                    \
      const uint8_t *a, int a_stride, const uint8_t *b, int b_stride,    \
      uint32_t *sse) {                                                   \
    return HWY_NAMESPACE::Variance<w, h>(a, a_stride, b, b_stride, sse); \
  }

#if HWY_TARGET != HWY_AVX3
#error "variance_avx512.cc is misconfigured; it must be built for AVX512."
#endif

MAKE_VARIANCE(128, 128, avx512)
MAKE_VARIANCE(128, 64, avx512)
MAKE_VARIANCE(64, 128, avx512)
MAKE_VARIANCE(64, 64, avx512)
MAKE_VARIANCE(64, 32, avx512)
MAKE_VARIANCE(32, 64, avx512)
MAKE_VARIANCE(32, 32, avx512)
MAKE_VARIANCE(32, 16, avx512)
MAKE_VARIANCE(16, 32, avx512)
MAKE_VARIANCE(16, 16, avx512)
MAKE_VARIANCE(16, 8, avx512)
MAKE_VARIANCE(8, 16, avx512)
MAKE_VARIANCE(8, 8, avx512)
MAKE_VARIANCE(8, 4, avx512)
MAKE_VARIANCE(4, 8, avx512)
MAKE_VARIANCE(4, 4, avx512)
MAKE_VARIANCE(4, 16, avx512)
MAKE_VARIANCE(16, 4, avx512)
MAKE_VARIANCE(8, 32, avx512)
MAKE_VARIANCE(32, 8, avx512)
MAKE_VARIANCE(16, 64, avx512)
MAKE_VARIANCE(64, 16, avx512)

HWY_AFTER_NAMESPACE();
