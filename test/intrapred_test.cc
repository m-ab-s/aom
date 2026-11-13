/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"
#include "config/av1_rtcd.h"

#include "test/acm_random.h"
#include "test/register_state_check.h"
#include "test/util.h"
#include "av1/common/blockd.h"
#include "av1/common/common.h"
#include "av1/common/pred_common.h"
#include "aom_mem/aom_mem.h"

namespace {

using libaom_test::ACMRandom;

const int count_test_block = 100000;

using HighbdIntraPred = void (*)(uint16_t *dst, ptrdiff_t stride,
                                 const uint16_t *above, const uint16_t *left,
                                 int bps);
using IntraPred = void (*)(uint8_t *dst, ptrdiff_t stride, const uint8_t *above,
                           const uint8_t *left);

}  // namespace

// NOTE: Under gcc version 7.3.0 (Debian 7.3.0-5), if this template is in the
// anonymous namespace, then we get a strange compiler warning in
// the begin() and end() methods of the ParamGenerator template class in
// gtest/internal/gtest-param-util.h:
//   warning: ‘<anonymous>’ is used uninitialized in this function
// As a workaround, put this template outside the anonymous namespace.
// See bug aomedia:2003.
template <typename FuncType>
struct IntraPredFunc {
  IntraPredFunc(FuncType pred = nullptr, FuncType ref = nullptr,
                int block_width_value = 0, int block_height_value = 0,
                int bit_depth_value = 0)
      : pred_fn(pred), ref_fn(ref), block_width(block_width_value),
        block_height(block_height_value), bit_depth(bit_depth_value) {}

  FuncType pred_fn;
  FuncType ref_fn;
  int block_width;
  int block_height;
  int bit_depth;
};

namespace {

template <typename FuncType, typename Pixel>
class AV1IntraPredTest
    : public ::testing::TestWithParam<IntraPredFunc<FuncType> > {
 public:
  void RunTest(Pixel *left_col, Pixel *above_data, Pixel *dst, Pixel *ref_dst) {
    ACMRandom rnd(ACMRandom::DeterministicSeed());
    const int block_width = params_.block_width;
    const int block_height = params_.block_height;
    above_row_ = above_data + 16;
    left_col_ = left_col;
    dst_ = dst;
    ref_dst_ = ref_dst;
    int error_count = 0;
    for (int i = 0; i < count_test_block; ++i) {
      // Fill edges with random data, try first with saturated values.
      for (int x = -1; x <= block_width * 2; x++) {
        if (i == 0) {
          above_row_[x] = mask_;
        } else {
          above_row_[x] = rnd.Rand16() & mask_;
        }
      }
      for (int y = 0; y < block_height; y++) {
        if (i == 0) {
          left_col_[y] = mask_;
        } else {
          left_col_[y] = rnd.Rand16() & mask_;
        }
      }
      Predict();
      CheckPrediction(i, &error_count);
    }
    ASSERT_EQ(0, error_count);
  }
  void RunSpeedTest(Pixel *left_col, Pixel *above_data, Pixel *dst,
                    Pixel *ref_dst) {
    ACMRandom rnd(ACMRandom::DeterministicSeed());
    const int block_width = params_.block_width;
    const int block_height = params_.block_height;
    above_row_ = above_data + 16;
    left_col_ = left_col;
    dst_ = dst;
    ref_dst_ = ref_dst;
    int error_count = 0;
    const int numIter = 100;

    int c_sum_time = 0;
    int simd_sum_time = 0;
    for (int i = 0; i < count_test_block; ++i) {
      // Fill edges with random data, try first with saturated values.
      for (int x = -1; x <= block_width * 2; x++) {
        if (i == 0) {
          above_row_[x] = mask_;
        } else {
          above_row_[x] = rnd.Rand16() & mask_;
        }
      }
      for (int y = 0; y < block_height; y++) {
        if (i == 0) {
          left_col_[y] = mask_;
        } else {
          left_col_[y] = rnd.Rand16() & mask_;
        }
      }

      aom_usec_timer c_timer_;
      aom_usec_timer_start(&c_timer_);

      PredictRefSpeedTest(numIter);

      aom_usec_timer_mark(&c_timer_);

      aom_usec_timer simd_timer_;
      aom_usec_timer_start(&simd_timer_);

      PredictFncSpeedTest(numIter);

      aom_usec_timer_mark(&simd_timer_);

      c_sum_time += static_cast<int>(aom_usec_timer_elapsed(&c_timer_));
      simd_sum_time += static_cast<int>(aom_usec_timer_elapsed(&simd_timer_));

      CheckPrediction(i, &error_count);
    }

    printf(
        "blockWxH = %d x %d c_time = %d \t simd_time = %d \t Gain = %4.2f \n",
        block_width, block_height, c_sum_time, simd_sum_time,
        (static_cast<float>(c_sum_time) / static_cast<float>(simd_sum_time)));
    ASSERT_EQ(0, error_count);
  }

 protected:
  void SetUp() override {
    params_ = this->GetParam();
    stride_ = params_.block_width * 3;
    mask_ = (1 << params_.bit_depth) - 1;
  }

  virtual void Predict() = 0;

  virtual void PredictRefSpeedTest(int num) = 0;
  virtual void PredictFncSpeedTest(int num) = 0;

  void CheckPrediction(int test_case_number, int *error_count) const {
    // For each pixel ensure that the calculated value is the same as reference.
    const int block_width = params_.block_width;
    const int block_height = params_.block_height;
    for (int y = 0; y < block_height; y++) {
      for (int x = 0; x < block_width; x++) {
        *error_count += ref_dst_[x + y * stride_] != dst_[x + y * stride_];
        if (*error_count == 1) {
          ASSERT_EQ(ref_dst_[x + y * stride_], dst_[x + y * stride_])
              << " Failed on Test Case Number " << test_case_number
              << " location: x = " << x << " y = " << y;
        }
      }
    }
  }

  Pixel *above_row_;
  Pixel *left_col_;
  Pixel *dst_;
  Pixel *ref_dst_;
  ptrdiff_t stride_;
  int mask_;

  IntraPredFunc<FuncType> params_;
};

#if CONFIG_AV1_HIGHBITDEPTH
class HighbdIntraPredTest : public AV1IntraPredTest<HighbdIntraPred, uint16_t> {
 protected:
  void Predict() override {
    const int bit_depth = params_.bit_depth;
    params_.ref_fn(ref_dst_, stride_, above_row_, left_col_, bit_depth);
    API_REGISTER_STATE_CHECK(
        params_.pred_fn(dst_, stride_, above_row_, left_col_, bit_depth));
  }
  void PredictRefSpeedTest(int num) override {
    const int bit_depth = params_.bit_depth;
    for (int i = 0; i < num; i++) {
      params_.ref_fn(ref_dst_, stride_, above_row_, left_col_, bit_depth);
    }
  }
  void PredictFncSpeedTest(int num) override {
    const int bit_depth = params_.bit_depth;
    for (int i = 0; i < num; i++) {
      params_.pred_fn(dst_, stride_, above_row_, left_col_, bit_depth);
    }
  }
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(HighbdIntraPredTest);

#endif

class LowbdIntraPredTest : public AV1IntraPredTest<IntraPred, uint8_t> {
 protected:
  void Predict() override {
    params_.ref_fn(ref_dst_, stride_, above_row_, left_col_);
    API_REGISTER_STATE_CHECK(
        params_.pred_fn(dst_, stride_, above_row_, left_col_));
  }
  void PredictRefSpeedTest(int num) override {
    for (int i = 0; i < num; i++) {
      params_.ref_fn(ref_dst_, stride_, above_row_, left_col_);
    }
  }
  void PredictFncSpeedTest(int num) override {
    for (int i = 0; i < num; i++) {
      params_.pred_fn(dst_, stride_, above_row_, left_col_);
    }
  }
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(LowbdIntraPredTest);

#if CONFIG_AV1_HIGHBITDEPTH
TEST_P(HighbdIntraPredTest, Bitexact) {
  // max block size is 64
  DECLARE_ALIGNED(16, uint16_t, left_col[2 * 64]);
  DECLARE_ALIGNED(16, uint16_t, above_data[2 * 64 + 64]);
  DECLARE_ALIGNED(16, uint16_t, dst[3 * 64 * 64]);
  DECLARE_ALIGNED(16, uint16_t, ref_dst[3 * 64 * 64]);
  av1_zero(left_col);
  av1_zero(above_data);
  RunTest(left_col, above_data, dst, ref_dst);
}

TEST_P(HighbdIntraPredTest, DISABLED_Speed) {
  // max block size is 64
  DECLARE_ALIGNED(16, uint16_t, left_col[2 * 64]);
  DECLARE_ALIGNED(16, uint16_t, above_data[2 * 64 + 64]);
  DECLARE_ALIGNED(16, uint16_t, dst[3 * 64 * 64]);
  DECLARE_ALIGNED(16, uint16_t, ref_dst[3 * 64 * 64]);
  av1_zero(left_col);
  av1_zero(above_data);
  RunSpeedTest(left_col, above_data, dst, ref_dst);
}
#endif

TEST_P(LowbdIntraPredTest, Bitexact) {
  // max block size is 64
  DECLARE_ALIGNED(16, uint8_t, left_col[2 * 64]);
  DECLARE_ALIGNED(16, uint8_t, above_data[2 * 64 + 64]);
  DECLARE_ALIGNED(16, uint8_t, dst[3 * 64 * 64]);
  DECLARE_ALIGNED(16, uint8_t, ref_dst[3 * 64 * 64]);
  av1_zero(left_col);
  av1_zero(above_data);
  RunTest(left_col, above_data, dst, ref_dst);
}
TEST_P(LowbdIntraPredTest, DISABLED_Speed) {
  // max block size is 64
  DECLARE_ALIGNED(16, uint8_t, left_col[2 * 64]);
  DECLARE_ALIGNED(16, uint8_t, above_data[2 * 64 + 64]);
  DECLARE_ALIGNED(16, uint8_t, dst[3 * 64 * 64]);
  DECLARE_ALIGNED(16, uint8_t, ref_dst[3 * 64 * 64]);
  av1_zero(left_col);
  av1_zero(above_data);
  RunSpeedTest(left_col, above_data, dst, ref_dst);
}

#if CONFIG_AV1_HIGHBITDEPTH
// -----------------------------------------------------------------------------
// High Bit Depth Tests
#define highbd_entry(type, width, height, opt, bd)                          \
  IntraPredFunc<HighbdIntraPred>(                                           \
      &aom_highbd_##type##_predictor_##width##x##height##_##opt,            \
      &aom_highbd_##type##_predictor_##width##x##height##_c, width, height, \
      bd)

#if !CONFIG_REALTIME_ONLY || CONFIG_AV1_DECODER
#define highbd_intrapred(type, opt, bd)                                        \
  highbd_entry(type, 4, 4, opt, bd), highbd_entry(type, 4, 8, opt, bd),        \
      highbd_entry(type, 4, 16, opt, bd), highbd_entry(type, 8, 4, opt, bd),   \
      highbd_entry(type, 8, 8, opt, bd), highbd_entry(type, 8, 16, opt, bd),   \
      highbd_entry(type, 8, 32, opt, bd), highbd_entry(type, 16, 4, opt, bd),  \
      highbd_entry(type, 16, 8, opt, bd), highbd_entry(type, 16, 16, opt, bd), \
      highbd_entry(type, 16, 32, opt, bd),                                     \
      highbd_entry(type, 16, 64, opt, bd), highbd_entry(type, 32, 8, opt, bd), \
      highbd_entry(type, 32, 16, opt, bd),                                     \
      highbd_entry(type, 32, 32, opt, bd),                                     \
      highbd_entry(type, 32, 64, opt, bd),                                     \
      highbd_entry(type, 64, 16, opt, bd),                                     \
      highbd_entry(type, 64, 32, opt, bd), highbd_entry(type, 64, 64, opt, bd)
#else
#define highbd_intrapred(type, opt, bd)                                       \
  highbd_entry(type, 4, 4, opt, bd), highbd_entry(type, 4, 8, opt, bd),       \
      highbd_entry(type, 8, 4, opt, bd), highbd_entry(type, 8, 8, opt, bd),   \
      highbd_entry(type, 8, 16, opt, bd), highbd_entry(type, 16, 8, opt, bd), \
      highbd_entry(type, 16, 16, opt, bd),                                    \
      highbd_entry(type, 16, 32, opt, bd),                                    \
      highbd_entry(type, 32, 16, opt, bd),                                    \
      highbd_entry(type, 32, 32, opt, bd),                                    \
      highbd_entry(type, 32, 64, opt, bd),                                    \
      highbd_entry(type, 64, 32, opt, bd), highbd_entry(type, 64, 64, opt, bd)
#endif  // !CONFIG_REALTIME_ONLY || CONFIG_AV1_DECODER
#endif  // CONFIG_AV1_HIGHBITDEPTH

// ---------------------------------------------------------------------------
// Low Bit Depth Tests

#define lowbd_entry(type, width, height, opt)                                  \
  IntraPredFunc<IntraPred>(&aom_##type##_predictor_##width##x##height##_##opt, \
                           &aom_##type##_predictor_##width##x##height##_c,     \
                           width, height, 8)

#if !CONFIG_REALTIME_ONLY || CONFIG_AV1_DECODER
#define lowbd_intrapred(type, opt)                                    \
  lowbd_entry(type, 4, 4, opt), lowbd_entry(type, 4, 8, opt),         \
      lowbd_entry(type, 4, 16, opt), lowbd_entry(type, 8, 4, opt),    \
      lowbd_entry(type, 8, 8, opt), lowbd_entry(type, 8, 16, opt),    \
      lowbd_entry(type, 8, 32, opt), lowbd_entry(type, 16, 4, opt),   \
      lowbd_entry(type, 16, 8, opt), lowbd_entry(type, 16, 16, opt),  \
      lowbd_entry(type, 16, 32, opt), lowbd_entry(type, 16, 64, opt), \
      lowbd_entry(type, 32, 8, opt), lowbd_entry(type, 32, 16, opt),  \
      lowbd_entry(type, 32, 32, opt), lowbd_entry(type, 32, 64, opt), \
      lowbd_entry(type, 64, 16, opt), lowbd_entry(type, 64, 32, opt), \
      lowbd_entry(type, 64, 64, opt)
#else
#define lowbd_intrapred(type, opt)                                    \
  lowbd_entry(type, 4, 4, opt), lowbd_entry(type, 4, 8, opt),         \
      lowbd_entry(type, 8, 4, opt), lowbd_entry(type, 8, 8, opt),     \
      lowbd_entry(type, 8, 16, opt), lowbd_entry(type, 16, 8, opt),   \
      lowbd_entry(type, 16, 16, opt), lowbd_entry(type, 16, 32, opt), \
      lowbd_entry(type, 32, 16, opt), lowbd_entry(type, 32, 32, opt), \
      lowbd_entry(type, 32, 64, opt), lowbd_entry(type, 64, 32, opt), \
      lowbd_entry(type, 64, 64, opt)
#endif  // !CONFIG_REALTIME_ONLY || CONFIG_AV1_DECODER

#if HAVE_SSE2
#if !HAVE_X86_ASM
// The SSE2 4x4, 8x8, 16x16 and 32x32 predictors below are written in x86
// assembly, so only the other sizes are available without it.
#if !CONFIG_REALTIME_ONLY || CONFIG_AV1_DECODER
#define lowbd_intrapred_sse2_no_asm(type)                               \
  lowbd_entry(type, 4, 8, sse2), lowbd_entry(type, 4, 16, sse2),        \
      lowbd_entry(type, 8, 4, sse2), lowbd_entry(type, 8, 16, sse2),    \
      lowbd_entry(type, 8, 32, sse2), lowbd_entry(type, 16, 4, sse2),   \
      lowbd_entry(type, 16, 8, sse2), lowbd_entry(type, 16, 32, sse2),  \
      lowbd_entry(type, 16, 64, sse2), lowbd_entry(type, 32, 8, sse2),  \
      lowbd_entry(type, 32, 16, sse2), lowbd_entry(type, 32, 64, sse2), \
      lowbd_entry(type, 64, 16, sse2), lowbd_entry(type, 64, 32, sse2), \
      lowbd_entry(type, 64, 64, sse2)
#else
#define lowbd_intrapred_sse2_no_asm(type)                               \
  lowbd_entry(type, 4, 8, sse2), lowbd_entry(type, 8, 4, sse2),         \
      lowbd_entry(type, 8, 16, sse2), lowbd_entry(type, 16, 8, sse2),   \
      lowbd_entry(type, 16, 32, sse2), lowbd_entry(type, 32, 16, sse2), \
      lowbd_entry(type, 32, 64, sse2), lowbd_entry(type, 64, 32, sse2), \
      lowbd_entry(type, 64, 64, sse2)
#endif  // !CONFIG_REALTIME_ONLY || CONFIG_AV1_DECODER
#endif  // !HAVE_X86_ASM

const IntraPredFunc<IntraPred> LowbdIntraPredTestVector[] = {
#if HAVE_X86_ASM
  lowbd_intrapred(dc, sse2),      lowbd_intrapred(dc_top, sse2),
  lowbd_intrapred(dc_left, sse2), lowbd_intrapred(dc_128, sse2),
  lowbd_intrapred(v, sse2),       lowbd_intrapred(h, sse2),
#else
  lowbd_intrapred_sse2_no_asm(dc),      lowbd_intrapred_sse2_no_asm(dc_top),
  lowbd_intrapred_sse2_no_asm(dc_left), lowbd_intrapred_sse2_no_asm(dc_128),
  lowbd_intrapred_sse2_no_asm(v),       lowbd_intrapred_sse2_no_asm(h),
#endif  // HAVE_X86_ASM
};

INSTANTIATE_TEST_SUITE_P(SSE2, LowbdIntraPredTest,
                         ::testing::ValuesIn(LowbdIntraPredTestVector));
#endif  // HAVE_SSE2

#if HAVE_NEON
const IntraPredFunc<IntraPred> LowbdIntraPredTestVectorNeon[] = {
  lowbd_intrapred(dc, neon),       lowbd_intrapred(dc_top, neon),
  lowbd_intrapred(dc_left, neon),  lowbd_intrapred(dc_128, neon),
  lowbd_intrapred(v, neon),        lowbd_intrapred(h, neon),
  lowbd_intrapred(smooth, neon),   lowbd_intrapred(smooth_v, neon),
  lowbd_intrapred(smooth_h, neon), lowbd_intrapred(paeth, neon),
};

INSTANTIATE_TEST_SUITE_P(NEON, LowbdIntraPredTest,
                         ::testing::ValuesIn(LowbdIntraPredTestVectorNeon));
#endif  // HAVE_NEON

#if HAVE_SSSE3
const IntraPredFunc<IntraPred> LowbdIntraPredTestVectorSsse3[] = {
  lowbd_intrapred(paeth, ssse3),
  lowbd_intrapred(smooth, ssse3),
  lowbd_intrapred(smooth_v, ssse3),
  lowbd_intrapred(smooth_h, ssse3),
};

INSTANTIATE_TEST_SUITE_P(SSSE3, LowbdIntraPredTest,
                         ::testing::ValuesIn(LowbdIntraPredTestVectorSsse3));
#endif  // HAVE_SSSE3

#if HAVE_AVX2
const IntraPredFunc<IntraPred> LowbdIntraPredTestVectorAvx2[] = {
#if !CONFIG_REALTIME_ONLY || CONFIG_AV1_DECODER
  lowbd_entry(dc, 32, 16, avx2),      lowbd_entry(dc, 32, 32, avx2),
  lowbd_entry(dc, 32, 64, avx2),      lowbd_entry(dc, 64, 16, avx2),
  lowbd_entry(dc, 64, 32, avx2),      lowbd_entry(dc, 64, 64, avx2),

  lowbd_entry(dc_top, 32, 16, avx2),  lowbd_entry(dc_top, 32, 32, avx2),
  lowbd_entry(dc_top, 32, 64, avx2),  lowbd_entry(dc_top, 64, 16, avx2),
  lowbd_entry(dc_top, 64, 32, avx2),  lowbd_entry(dc_top, 64, 64, avx2),

  lowbd_entry(dc_left, 32, 16, avx2), lowbd_entry(dc_left, 32, 32, avx2),
  lowbd_entry(dc_left, 32, 64, avx2), lowbd_entry(dc_left, 64, 16, avx2),
  lowbd_entry(dc_left, 64, 32, avx2), lowbd_entry(dc_left, 64, 64, avx2),

  lowbd_entry(dc_128, 32, 16, avx2),  lowbd_entry(dc_128, 32, 32, avx2),
  lowbd_entry(dc_128, 32, 64, avx2),  lowbd_entry(dc_128, 64, 16, avx2),
  lowbd_entry(dc_128, 64, 32, avx2),  lowbd_entry(dc_128, 64, 64, avx2),

  lowbd_entry(v, 32, 16, avx2),       lowbd_entry(v, 32, 32, avx2),
  lowbd_entry(v, 32, 64, avx2),       lowbd_entry(v, 64, 16, avx2),
  lowbd_entry(v, 64, 32, avx2),       lowbd_entry(v, 64, 64, avx2),

  lowbd_entry(h, 32, 32, avx2),

  lowbd_entry(paeth, 16, 8, avx2),    lowbd_entry(paeth, 16, 16, avx2),
  lowbd_entry(paeth, 16, 32, avx2),   lowbd_entry(paeth, 32, 16, avx2),
  lowbd_entry(paeth, 32, 32, avx2),   lowbd_entry(paeth, 32, 64, avx2),
  lowbd_entry(paeth, 64, 32, avx2),   lowbd_entry(paeth, 64, 64, avx2),
#else
  lowbd_entry(dc, 32, 16, avx2),      lowbd_entry(dc, 32, 32, avx2),
  lowbd_entry(dc, 32, 64, avx2),      lowbd_entry(dc, 64, 32, avx2),
  lowbd_entry(dc, 64, 64, avx2),

  lowbd_entry(dc_top, 32, 16, avx2),  lowbd_entry(dc_top, 32, 32, avx2),
  lowbd_entry(dc_top, 32, 64, avx2),  lowbd_entry(dc_top, 64, 32, avx2),
  lowbd_entry(dc_top, 64, 64, avx2),

  lowbd_entry(dc_left, 32, 16, avx2), lowbd_entry(dc_left, 32, 32, avx2),
  lowbd_entry(dc_left, 32, 64, avx2), lowbd_entry(dc_left, 64, 32, avx2),
  lowbd_entry(dc_left, 64, 64, avx2),

  lowbd_entry(dc_128, 32, 16, avx2),  lowbd_entry(dc_128, 32, 32, avx2),
  lowbd_entry(dc_128, 32, 64, avx2),  lowbd_entry(dc_128, 64, 32, avx2),
  lowbd_entry(dc_128, 64, 64, avx2),

  lowbd_entry(v, 32, 16, avx2),       lowbd_entry(v, 32, 32, avx2),
  lowbd_entry(v, 32, 64, avx2),       lowbd_entry(v, 64, 32, avx2),
  lowbd_entry(v, 64, 64, avx2),

  lowbd_entry(h, 32, 32, avx2),

  lowbd_entry(paeth, 16, 8, avx2),    lowbd_entry(paeth, 16, 16, avx2),
  lowbd_entry(paeth, 16, 32, avx2),   lowbd_entry(paeth, 32, 16, avx2),
  lowbd_entry(paeth, 32, 32, avx2),   lowbd_entry(paeth, 32, 64, avx2),
  lowbd_entry(paeth, 64, 32, avx2),   lowbd_entry(paeth, 64, 64, avx2),
#endif  // !CONFIG_REALTIME_ONLY || CONFIG_AV1_DECODER
};

INSTANTIATE_TEST_SUITE_P(AVX2, LowbdIntraPredTest,
                         ::testing::ValuesIn(LowbdIntraPredTestVectorAvx2));
#endif  // HAVE_AVX2

#if CONFIG_AV1_HIGHBITDEPTH
#if HAVE_NEON
const IntraPredFunc<HighbdIntraPred> HighbdIntraPredTestVectorNeon[] = {
  highbd_intrapred(dc, neon, 12),       highbd_intrapred(dc_top, neon, 12),
  highbd_intrapred(dc_left, neon, 12),  highbd_intrapred(dc_128, neon, 12),
  highbd_intrapred(v, neon, 12),        highbd_intrapred(h, neon, 12),
  highbd_intrapred(paeth, neon, 12),    highbd_intrapred(smooth, neon, 12),
  highbd_intrapred(smooth_v, neon, 12), highbd_intrapred(smooth_h, neon, 12),
};

INSTANTIATE_TEST_SUITE_P(NEON, HighbdIntraPredTest,
                         ::testing::ValuesIn(HighbdIntraPredTestVectorNeon));
#endif  // HAVE_NEON

#if HAVE_SSE2
const IntraPredFunc<HighbdIntraPred> HighbdIntraPredTestVectorSse2[] = {
#if HAVE_X86_ASM
  highbd_entry(dc, 4, 4, sse2, 12),
#endif  // HAVE_X86_ASM
  highbd_entry(dc, 4, 8, sse2, 12),
  highbd_entry(dc, 8, 4, sse2, 12),
#if HAVE_X86_ASM
  highbd_entry(dc, 8, 8, sse2, 12),
#endif  // HAVE_X86_ASM
  highbd_entry(dc, 8, 16, sse2, 12),
  highbd_entry(dc, 16, 8, sse2, 12),
#if HAVE_X86_ASM
  highbd_entry(dc, 16, 16, sse2, 12),
#endif  // HAVE_X86_ASM
  highbd_entry(dc, 16, 32, sse2, 12),
  highbd_entry(dc, 32, 16, sse2, 12),
#if HAVE_X86_ASM
  highbd_entry(dc, 32, 32, sse2, 12),
#endif  // HAVE_X86_ASM

  highbd_entry(dc_top, 4, 4, sse2, 12),
  highbd_entry(dc_top, 4, 8, sse2, 12),
  highbd_entry(dc_top, 8, 4, sse2, 12),
  highbd_entry(dc_top, 8, 8, sse2, 12),
  highbd_entry(dc_top, 8, 16, sse2, 12),
  highbd_entry(dc_top, 16, 8, sse2, 12),
  highbd_entry(dc_top, 16, 16, sse2, 12),
  highbd_entry(dc_top, 16, 32, sse2, 12),
  highbd_entry(dc_top, 32, 16, sse2, 12),
  highbd_entry(dc_top, 32, 32, sse2, 12),

  highbd_entry(dc_left, 4, 4, sse2, 12),
  highbd_entry(dc_left, 4, 8, sse2, 12),
  highbd_entry(dc_left, 8, 4, sse2, 12),
  highbd_entry(dc_left, 8, 8, sse2, 12),
  highbd_entry(dc_left, 8, 16, sse2, 12),
  highbd_entry(dc_left, 16, 8, sse2, 12),
  highbd_entry(dc_left, 16, 16, sse2, 12),
  highbd_entry(dc_left, 16, 32, sse2, 12),
  highbd_entry(dc_left, 32, 16, sse2, 12),
  highbd_entry(dc_left, 32, 32, sse2, 12),

  highbd_entry(dc_128, 4, 4, sse2, 12),
  highbd_entry(dc_128, 4, 8, sse2, 12),
  highbd_entry(dc_128, 8, 4, sse2, 12),
  highbd_entry(dc_128, 8, 8, sse2, 12),
  highbd_entry(dc_128, 8, 16, sse2, 12),
  highbd_entry(dc_128, 16, 8, sse2, 12),
  highbd_entry(dc_128, 16, 16, sse2, 12),
  highbd_entry(dc_128, 16, 32, sse2, 12),
  highbd_entry(dc_128, 32, 16, sse2, 12),
  highbd_entry(dc_128, 32, 32, sse2, 12),

#if HAVE_X86_ASM
  highbd_entry(v, 4, 4, sse2, 12),
#endif  // HAVE_X86_ASM
  highbd_entry(v, 4, 8, sse2, 12),
  highbd_entry(v, 8, 4, sse2, 12),
#if HAVE_X86_ASM
  highbd_entry(v, 8, 8, sse2, 12),
#endif  // HAVE_X86_ASM
  highbd_entry(v, 8, 16, sse2, 12),
  highbd_entry(v, 16, 8, sse2, 12),
#if HAVE_X86_ASM
  highbd_entry(v, 16, 16, sse2, 12),
#endif  // HAVE_X86_ASM
  highbd_entry(v, 16, 32, sse2, 12),
  highbd_entry(v, 32, 16, sse2, 12),
#if HAVE_X86_ASM
  highbd_entry(v, 32, 32, sse2, 12),
#endif  // HAVE_X86_ASM

  highbd_entry(h, 4, 4, sse2, 12),
  highbd_entry(h, 4, 8, sse2, 12),
  highbd_entry(h, 8, 4, sse2, 12),
  highbd_entry(h, 8, 8, sse2, 12),
  highbd_entry(h, 8, 16, sse2, 12),
  highbd_entry(h, 16, 8, sse2, 12),
  highbd_entry(h, 16, 16, sse2, 12),
  highbd_entry(h, 16, 32, sse2, 12),
  highbd_entry(h, 32, 16, sse2, 12),
  highbd_entry(h, 32, 32, sse2, 12),
};

INSTANTIATE_TEST_SUITE_P(SSE2, HighbdIntraPredTest,
                         ::testing::ValuesIn(HighbdIntraPredTestVectorSse2));
#endif  // HAVE_SSE2
#endif  // CONFIG_AV1_HIGHBITDEPTH

using PredictDcBlockFunc = void (*)(const uint8_t *ref, int ref_stride,
                                    uint8_t *dst, int dst_stride,
                                    TX_SIZE tx_size, int txwpx, int txhpx,
                                    int n_top_px, int n_left_px);

uint8_t ReferenceLowbdDcValue(const uint8_t *ref, int ref_stride, int txwpx,
                              int txhpx, int n_top_px, int n_left_px) {
  int sum = 0;
  int count = 0;
  if (n_top_px > 0) {
    const uint8_t *above = ref - ref_stride;
    for (int c = 0; c < txwpx; ++c) {
      sum += above[c < n_top_px ? c : n_top_px - 1];
    }
    count += txwpx;
  }
  if (n_left_px > 0) {
    const uint8_t *left = ref - 1;
    for (int r = 0; r < txhpx; ++r) {
      sum += left[(r < n_left_px ? r : n_left_px - 1) * ref_stride];
    }
    count += txhpx;
  }
  if (count == 0) return 128;
  return static_cast<uint8_t>((sum + (count >> 1)) / count);
}

void CheckLowbdPredictDcBlockMatchesC(PredictDcBlockFunc fn) {
  ACMRandom rnd(ACMRandom::DeterministicSeed());
  constexpr int kBufSize = 96 * 96;
  std::vector<uint8_t> rnd_buf(kBufSize);
  for (int idx = 0; idx < kBufSize; ++idx) {
    rnd_buf[idx] = rnd.Rand8();
  }
  std::vector<uint8_t> ref_buf(kBufSize);
  std::vector<uint8_t> dst_c(kBufSize);
  std::vector<uint8_t> dst_test(kBufSize);

  for (int tx = 0; tx < TX_SIZES_ALL; ++tx) {
    const TX_SIZE tx_size = static_cast<TX_SIZE>(tx);
    const int txwpx = tx_size_wide[tx_size];
    const int txhpx = tx_size_high[tx_size];
    const int ref_stride = txwpx + 19;
    const int dst_stride = txwpx + 23;
    const int top_cases[] = { 0, 1, txwpx / 2, txwpx };
    const int left_cases[] = { 0, 1, txhpx / 2, txhpx };

    for (int align_off = 0; align_off <= 3; ++align_off) {
      const uint8_t *ref = ref_buf.data() + ref_stride + 16 + align_off;
      for (int n_top_px : top_cases) {
        for (int n_left_px : left_cases) {
          for (int pat = 0; pat < 3; ++pat) {
            if (pat == 0) {
              memset(ref_buf.data(), 0, kBufSize);
            } else if (pat == 1) {
              memset(ref_buf.data(), 255, kBufSize);
            } else {
              memcpy(ref_buf.data(), rnd_buf.data(), kBufSize);
            }
            memset(dst_c.data(), 0xA5, kBufSize);
            memset(dst_test.data(), 0xA5, kBufSize);
            uint8_t *out_c = dst_c.data() + dst_stride + 16 + align_off;
            uint8_t *out_test = dst_test.data() + dst_stride + 16 + align_off;
            const uint8_t *const cur_ref =
                (n_top_px == 0 && n_left_px == 0 && pat == 0) ? ref_buf.data()
                                                              : ref;
            const uint8_t expected_dc = ReferenceLowbdDcValue(
                cur_ref, ref_stride, txwpx, txhpx, n_top_px, n_left_px);

            API_REGISTER_STATE_CHECK(av1_predict_dc_block_lbd_c(
                cur_ref, ref_stride, out_c, dst_stride, tx_size, txwpx, txhpx,
                n_top_px, n_left_px));
            API_REGISTER_STATE_CHECK(fn(cur_ref, ref_stride, out_test,
                                        dst_stride, tx_size, txwpx, txhpx,
                                        n_top_px, n_left_px));

            for (int c = -1; c <= txwpx; ++c) {
              ASSERT_EQ(out_c[-dst_stride + c], 0xA5);
              ASSERT_EQ(out_test[-dst_stride + c], 0xA5);
              ASSERT_EQ(out_c[txhpx * dst_stride + c], 0xA5);
              ASSERT_EQ(out_test[txhpx * dst_stride + c], 0xA5);
            }
            for (int r = 0; r < txhpx; ++r) {
              ASSERT_EQ(out_c[r * dst_stride - 1], 0xA5);
              ASSERT_EQ(out_test[r * dst_stride - 1], 0xA5);
              for (int c = 0; c < txwpx; ++c) {
                ASSERT_EQ(out_c[r * dst_stride + c], expected_dc)
                    << "tx_size=" << tx << " align=" << align_off
                    << " n_top=" << n_top_px << " n_left=" << n_left_px
                    << " pat=" << pat << " r=" << r << " c=" << c;
                ASSERT_EQ(out_test[r * dst_stride + c], expected_dc)
                    << "tx_size=" << tx << " align=" << align_off
                    << " n_top=" << n_top_px << " n_left=" << n_left_px
                    << " pat=" << pat << " r=" << r << " c=" << c;
              }
              ASSERT_EQ(out_c[r * dst_stride + txwpx], 0xA5);
              ASSERT_EQ(out_test[r * dst_stride + txwpx], 0xA5);
            }
          }
        }
      }
    }
  }
}

TEST(LowbdPredictDcBlockTest, RtcdMatchesC) {
  CheckLowbdPredictDcBlockMatchesC(av1_predict_dc_block_lbd);
}

#if HAVE_SSE2
TEST(SSE2, LowbdPredictDcBlockMatchesC) {
  CheckLowbdPredictDcBlockMatchesC(av1_predict_dc_block_lbd_sse2);
}
#endif  // HAVE_SSE2
}  // namespace
