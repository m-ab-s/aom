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

#define HWY_COMPILE_ALL_ATTAINABLE 1

#define HWY_TARGET_INCLUDE "aom_dsp/sad_hwy.cc"

#include "third_party/highway/hwy/foreach_target.h"
#include "third_party/highway/hwy/highway.h"

#include "config/aom_config.h"

HWY_BEFORE_NAMESPACE();

namespace {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

template <int BlockWidth>
HWY_MAYBE_UNUSED unsigned int SumOfAbsoluteDiff(const uint8_t *src_ptr,
                                                int src_stride,
                                                const uint8_t *ref_ptr,
                                                int ref_stride, int h) {
  constexpr hn::CappedTag<uint8_t, BlockWidth> pixel_tag;
  constexpr hn::Repartition<uint64_t, decltype(pixel_tag)> intermediate_sum_tag;
  const int vw = hn::Lanes(pixel_tag);
  auto sum_sad = hn::Zero(intermediate_sum_tag);
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < BlockWidth; j += vw) {
      auto src_vec = hn::LoadU(pixel_tag, &src_ptr[j]);
      auto ref_vec = hn::LoadU(pixel_tag, &ref_ptr[j]);
      auto sad = hn::SumsOf8AbsDiff(src_vec, ref_vec);
      sum_sad = hn::Add(sum_sad, sad);
    }
    src_ptr += src_stride;
    ref_ptr += ref_stride;
  }
  return static_cast<unsigned int>(
      hn::ReduceSum(intermediate_sum_tag, sum_sad));
}
}  // namespace HWY_NAMESPACE
}  // namespace

HWY_AFTER_NAMESPACE();

#define FSAD(w, h, suffix)                                                   \
  extern "C" unsigned int SumOfAbsoluteDiff##w##x##h##_##suffix(             \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,        \
      int ref_stride) {                                                      \
    return HWY_NAMESPACE::SumOfAbsoluteDiff<w>(src_ptr, src_stride, ref_ptr, \
                                               ref_stride, h);               \
  }

#define FOR_EACH_BLOCK_SIZE(X, suffix) \
  X(64, 32, suffix)                    \
  X(64, 64, suffix)

#if HWY_TARGET == HWY_AVX2
FOR_EACH_BLOCK_SIZE(FSAD, avx2)
#endif  // HWY_TARGET == HWY_AVX2

#if HWY_TARGET == HWY_AVX3
FOR_EACH_BLOCK_SIZE(FSAD, avx512)
#endif  // HWY_TARGET == HWY_AVX3
