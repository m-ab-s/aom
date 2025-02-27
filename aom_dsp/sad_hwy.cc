#define HWY_COMPILE_ALL_ATTAINABLE 1

#define HWY_TARGET_INCLUDE "aom_dsp/sad_hwy.cc"

#include "third_party/highway/hwy/foreach_target.h"
#include "third_party/highway/hwy/highway.h"

#include "config/aom_config.h"

HWY_BEFORE_NAMESPACE();

namespace {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

HWY_MAYBE_UNUSED unsigned int SumOfAbsoluteDiff64xN(const uint8_t *src_ptr,
                                                    int src_stride,
                                                    const uint8_t *ref_ptr,
                                                    int ref_stride, int h) {
  constexpr int kBlockWidth = 64;
  constexpr hn::CappedTag<uint8_t, kBlockWidth> pixel_tag;
  constexpr hn::Repartition<uint64_t, decltype(pixel_tag)> intermediate_sum_tag;
  const int vw = hn::Lanes(pixel_tag);
  auto sum_sad = hn::Zero(intermediate_sum_tag);
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < kBlockWidth; j += vw) {
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

#define FSAD64_H(h, suffix)                                                   \
  extern "C" unsigned int SumOfAbsoluteDiff64x##h##_##suffix(                 \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,         \
      int ref_stride) {                                                       \
    return HWY_NAMESPACE::SumOfAbsoluteDiff64xN(src_ptr, src_stride, ref_ptr, \
                                                ref_stride, h);               \
  }

#if HWY_TARGET == HWY_AVX2
FSAD64_H(32, avx2)
FSAD64_H(64, avx2)
#endif  // HWY_TARGET == HWY_AVX2
