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

#include <algorithm>
#include <cstring>
#include <ctime>
#include <memory>
#include <tuple>

#include "gtest/gtest.h"

#include "config/aom_config.h"
#include "config/av1_rtcd.h"

#include "test/acm_random.h"
#include "test/register_state_check.h"
#include "test/util.h"

#include "aom_ports/aom_timer.h"
#include "av1/common/mv.h"
#include "av1/common/restoration.h"

namespace {

using libaom_test::ACMRandom;
using std::make_tuple;
using std::tuple;

using SgrFunc = int (*)(const uint8_t *dat8, int width, int height, int stride,
                        int eps, const int *xqd, uint8_t *dst8, int dst_stride,
                        int32_t *tmpbuf, int bit_depth, int highbd);

struct AomFree {
  void operator()(void *p) const { aom_free(p); }
};

inline uint8_t *ToSgrBytePtr(uint8_t *p) { return p; }
#if CONFIG_AV1_HIGHBITDEPTH
inline uint8_t *ToSgrBytePtr(uint16_t *p) { return CONVERT_TO_BYTEPTR(p); }
#endif  // CONFIG_AV1_HIGHBITDEPTH

template <typename Pixel>
struct SelfguidedTestBuffers {
  static constexpr int kBorder = 16;
  int stride = 0;
  int out_stride = 0;
  std::unique_ptr<Pixel, AomFree> input_buf;
  std::unique_ptr<Pixel, AomFree> output_buf;
  std::unique_ptr<Pixel, AomFree> output2_buf;
  std::unique_ptr<int32_t, AomFree> tmpbuf;
  Pixel *input = nullptr;
  Pixel *output = nullptr;
  Pixel *output2 = nullptr;

  bool Init(int max_h, int in_stride, int out_s, bool allocate_output2) {
    stride = in_stride;
    out_stride = out_s;
    const size_t in_count =
        static_cast<size_t>(stride) * static_cast<size_t>(max_h + 2 * kBorder);
    const size_t out_count = static_cast<size_t>(out_stride) *
                             static_cast<size_t>(max_h + 2 * kBorder);

    input_buf.reset(
        static_cast<Pixel *>(aom_memalign(32, in_count * sizeof(Pixel))));
    if (!input_buf) return false;
    output_buf.reset(
        static_cast<Pixel *>(aom_memalign(32, out_count * sizeof(Pixel))));
    if (!output_buf) return false;
    if (allocate_output2) {
      output2_buf.reset(
          static_cast<Pixel *>(aom_memalign(32, out_count * sizeof(Pixel))));
      if (!output2_buf) return false;
      memset(output2_buf.get(), 0, out_count * sizeof(Pixel));
      output2 = output2_buf.get() + out_stride * kBorder + kBorder;
    }
    tmpbuf.reset(
        static_cast<int32_t *>(aom_memalign(32, RESTORATION_TMPBUF_SIZE)));
    if (!tmpbuf) return false;

    memset(input_buf.get(), 0, in_count * sizeof(Pixel));
    memset(output_buf.get(), 0, out_count * sizeof(Pixel));
    input = input_buf.get() + stride * kBorder + kBorder;
    output = output_buf.get() + out_stride * kBorder + kBorder;
    return true;
  }
};

template <typename Pixel>
void VerifySelfguidedTiles(SgrFunc tst_fun, SelfguidedTestBuffers<Pixel> *bufs,
                           int width, int height, int eps, const int *xqd,
                           int bit_depth, int highbd) {
  const int pu_width = RESTORATION_PROC_UNIT_SIZE;
  const int pu_height = RESTORATION_PROC_UNIT_SIZE;
  for (int k = 0; k < height; k += pu_height) {
    for (int j = 0; j < width; j += pu_width) {
      const int w = AOMMIN(pu_width, width - j);
      const int h = AOMMIN(pu_height, height - k);
      Pixel *input_p = bufs->input + k * bufs->stride + j;
      Pixel *output_p = bufs->output + k * bufs->out_stride + j;
      Pixel *output2_p = bufs->output2 + k * bufs->out_stride + j;
      ASSERT_EQ(tst_fun(ToSgrBytePtr(input_p), w, h, bufs->stride, eps, xqd,
                        ToSgrBytePtr(output_p), bufs->out_stride,
                        bufs->tmpbuf.get(), bit_depth, highbd),
                0);
      ASSERT_EQ(av1_apply_selfguided_restoration_c(
                    ToSgrBytePtr(input_p), w, h, bufs->stride, eps, xqd,
                    ToSgrBytePtr(output2_p), bufs->out_stride,
                    bufs->tmpbuf.get(), bit_depth, highbd),
                0);
    }
  }

  for (int j = 0; j < height; ++j) {
    for (int k = 0; k < width; ++k) {
      ASSERT_EQ(bufs->output[j * bufs->out_stride + k],
                bufs->output2[j * bufs->out_stride + k]);
    }
  }
}

template <typename Pixel>
void RunSelfguidedSpeedTest(SgrFunc tst_fun, int bit_depth, int highbd,
                            const char *suite_name) {
  const int pu_width = RESTORATION_PROC_UNIT_SIZE;
  const int pu_height = RESTORATION_PROC_UNIT_SIZE;
  const int width = 256, height = 256, stride = 288, out_stride = 288;
  const int NUM_ITERS = 2000;
  const int mask = (1 << bit_depth) - 1;

  SelfguidedTestBuffers<Pixel> bufs;
  ASSERT_TRUE(bufs.Init(height, stride, out_stride, false));

  ACMRandom rnd(ACMRandom::DeterministicSeed());

  for (int i = -16; i < height + 16; ++i)
    for (int j = -16; j < width + 16; ++j)
      bufs.input[i * stride + j] = static_cast<Pixel>(rnd.Rand16() & mask);

  int xqd[2] = { SGRPROJ_PRJ_MIN0 +
                     rnd.PseudoUniform(SGRPROJ_PRJ_MAX0 + 1 - SGRPROJ_PRJ_MIN0),
                 SGRPROJ_PRJ_MIN1 + rnd.PseudoUniform(SGRPROJ_PRJ_MAX1 + 1 -
                                                      SGRPROJ_PRJ_MIN1) };
  // Fix a parameter set, since the speed depends slightly on r.
  // Change this to test different combinations of values of r.
  int eps = 15;

  av1_loop_restoration_precal();

  aom_usec_timer ref_timer;
  aom_usec_timer_start(&ref_timer);
  for (int i = 0; i < NUM_ITERS; ++i) {
    for (int k = 0; k < height; k += pu_height)
      for (int j = 0; j < width; j += pu_width) {
        int w = AOMMIN(pu_width, width - j);
        int h = AOMMIN(pu_height, height - k);
        Pixel *input_p = bufs.input + k * stride + j;
        Pixel *output_p = bufs.output + k * out_stride + j;
        ASSERT_EQ(av1_apply_selfguided_restoration_c(
                      ToSgrBytePtr(input_p), w, h, stride, eps, xqd,
                      ToSgrBytePtr(output_p), out_stride, bufs.tmpbuf.get(),
                      bit_depth, highbd),
                  0);
      }
  }
  aom_usec_timer_mark(&ref_timer);
  const int64_t ref_time = aom_usec_timer_elapsed(&ref_timer);

  aom_usec_timer tst_timer;
  aom_usec_timer_start(&tst_timer);
  for (int i = 0; i < NUM_ITERS; ++i) {
    for (int k = 0; k < height; k += pu_height)
      for (int j = 0; j < width; j += pu_width) {
        int w = AOMMIN(pu_width, width - j);
        int h = AOMMIN(pu_height, height - k);
        Pixel *input_p = bufs.input + k * stride + j;
        Pixel *output_p = bufs.output + k * out_stride + j;
        ASSERT_EQ(tst_fun(ToSgrBytePtr(input_p), w, h, stride, eps, xqd,
                          ToSgrBytePtr(output_p), out_stride, bufs.tmpbuf.get(),
                          bit_depth, highbd),
                  0);
      }
  }
  aom_usec_timer_mark(&tst_timer);
  const int64_t tst_time = aom_usec_timer_elapsed(&tst_timer);

  std::cout << "[          ] C time = " << ref_time / 1000
            << " ms, SIMD time = " << tst_time / 1000 << " ms\n";

  EXPECT_GT(ref_time, tst_time)
      << "Error: " << suite_name << ".SpeedTest, SIMD slower than C.\n"
      << "C time: " << ref_time << " us\n"
      << "SIMD time: " << tst_time << " us\n";
}

template <typename Pixel>
void RunSelfguidedCorrectnessTest(SgrFunc tst_fun, int bit_depth, int highbd) {
  // Set the maximum width/height to test here. We actually test a small
  // range of sizes *up to* this size, so that we can check, eg.,
  // the behaviour on tiles which are not a multiple of 4 wide.
  const int max_w = 260, max_h = 260, stride = 672, out_stride = 672;
  const int NUM_ITERS = 81;
  const int mask = (1 << bit_depth) - 1;

  SelfguidedTestBuffers<Pixel> bufs;
  ASSERT_TRUE(bufs.Init(max_h, stride, out_stride, true));

  ACMRandom rnd(ACMRandom::DeterministicSeed());

  av1_loop_restoration_precal();

  for (int i = 0; i < NUM_ITERS; ++i) {
    for (int j = -16; j < max_h + 16; ++j)
      for (int k = -16; k < max_w + 16; ++k)
        bufs.input[j * stride + k] = static_cast<Pixel>(rnd.Rand16() & mask);

    int xqd[2] = { SGRPROJ_PRJ_MIN0 + rnd.PseudoUniform(SGRPROJ_PRJ_MAX0 + 1 -
                                                        SGRPROJ_PRJ_MIN0),
                   SGRPROJ_PRJ_MIN1 + rnd.PseudoUniform(SGRPROJ_PRJ_MAX1 + 1 -
                                                        SGRPROJ_PRJ_MIN1) };
    int eps = rnd.PseudoUniform(1 << SGRPROJ_PARAMS_BITS);

    // Test various tile sizes around 256x256
    int test_w = max_w - (i / 9);
    int test_h = max_h - (i % 9);

    ASSERT_NO_FATAL_FAILURE(VerifySelfguidedTiles(
        tst_fun, &bufs, test_w, test_h, eps, xqd, bit_depth, highbd));
  }
}

template <typename Pixel>
void RunSelfguidedExtremeValuesTest(SgrFunc tst_fun, int bit_depth,
                                    int highbd) {
  const int max_w = 247, max_h = 35, stride = 672, out_stride = 672;
  const int max_val = (1 << bit_depth) - 1;

  SelfguidedTestBuffers<Pixel> bufs;
  ASSERT_TRUE(bufs.Init(max_h, stride, out_stride, true));

  av1_loop_restoration_precal();

  // Cover boundary values around the 32-bit signed shift overflow threshold
  // in 12-bit flat/nearly-flat regions (2055 for r=2, 2057 for r=1, 2313 from
  // b/585314885, and max_val).
  const int values[] = {
    0,
    1 << (bit_depth - 1),
    std::min((1 << (bit_depth - 1)) + 9, max_val),
    std::min(2313, max_val),
    max_val,
  };
  const struct {
    int width;
    int height;
  } kSizes[] = { { 247, 1 }, { 67, 35 } };
  const int xqd[2] = { 16, 16 };

  for (const int base_val : values) {
    for (int pattern = 0; pattern < 2; ++pattern) {
      for (int j = -16; j < max_h + 16; ++j) {
        for (int k = -16; k < max_w + 16; ++k) {
          const int delta = (pattern == 0) ? 0 : (((j ^ k) & 1) ? 1 : -1);
          bufs.input[j * stride + k] =
              static_cast<Pixel>(std::clamp(base_val + delta, 0, max_val));
        }
      }

      for (const auto &sz : kSizes) {
        for (int eps = 0; eps < (1 << SGRPROJ_PARAMS_BITS); ++eps) {
          SCOPED_TRACE(testing::Message()
                       << "bit_depth=" << bit_depth << " highbd=" << highbd
                       << " base_val=" << base_val << " pattern=" << pattern
                       << " size=" << sz.width << "x" << sz.height
                       << " eps=" << eps);
          ASSERT_NO_FATAL_FAILURE(
              VerifySelfguidedTiles(tst_fun, &bufs, sz.width, sz.height, eps,
                                    xqd, bit_depth, highbd));
        }
      }
    }
  }
}

// Test parameter list:
//  <tst_fun_>
using FilterTestParam = tuple<SgrFunc>;

class AV1SelfguidedFilterTest
    : public ::testing::TestWithParam<FilterTestParam> {
 public:
  ~AV1SelfguidedFilterTest() override = default;
  void SetUp() override {}

 protected:
  void RunSpeedTest() {
    RunSelfguidedSpeedTest<uint8_t>(GET_PARAM(0), 8, 0,
                                    "AV1SelfguidedFilterTest");
  }

  void RunCorrectnessTest() {
    RunSelfguidedCorrectnessTest<uint8_t>(GET_PARAM(0), 8, 0);
  }

  void RunExtremeValuesTest() {
    RunSelfguidedExtremeValuesTest<uint8_t>(GET_PARAM(0), 8, 0);
  }
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AV1SelfguidedFilterTest);

TEST_P(AV1SelfguidedFilterTest, DISABLED_SpeedTest) { RunSpeedTest(); }
TEST_P(AV1SelfguidedFilterTest, CorrectnessTest) { RunCorrectnessTest(); }
TEST_P(AV1SelfguidedFilterTest, ExtremeValuesTest) { RunExtremeValuesTest(); }

#if HAVE_SSE4_1
INSTANTIATE_TEST_SUITE_P(
    SSE4_1, AV1SelfguidedFilterTest,
    ::testing::Values(av1_apply_selfguided_restoration_sse4_1));
#endif

#if HAVE_AVX2
INSTANTIATE_TEST_SUITE_P(
    AVX2, AV1SelfguidedFilterTest,
    ::testing::Values(av1_apply_selfguided_restoration_avx2));
#endif

#if HAVE_AVX512 && CONFIG_HIGHWAY
INSTANTIATE_TEST_SUITE_P(
    AVX512, AV1SelfguidedFilterTest,
    ::testing::Values(av1_apply_selfguided_restoration_avx512));
#endif

#if HAVE_NEON
INSTANTIATE_TEST_SUITE_P(
    NEON, AV1SelfguidedFilterTest,
    ::testing::Values(av1_apply_selfguided_restoration_neon));
#endif

#if CONFIG_AV1_HIGHBITDEPTH
// Test parameter list:
//  <tst_fun_, bit_depth>
using HighbdFilterTestParam = tuple<SgrFunc, int>;

class AV1HighbdSelfguidedFilterTest
    : public ::testing::TestWithParam<HighbdFilterTestParam> {
 public:
  ~AV1HighbdSelfguidedFilterTest() override = default;
  void SetUp() override {}

 protected:
  void RunSpeedTest() {
    RunSelfguidedSpeedTest<uint16_t>(GET_PARAM(0), GET_PARAM(1), 1,
                                     "AV1HighbdSelfguidedFilterTest");
  }

  void RunCorrectnessTest() {
    RunSelfguidedCorrectnessTest<uint16_t>(GET_PARAM(0), GET_PARAM(1), 1);
  }

  void RunExtremeValuesTest() {
    RunSelfguidedExtremeValuesTest<uint16_t>(GET_PARAM(0), GET_PARAM(1), 1);
  }
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AV1HighbdSelfguidedFilterTest);

TEST_P(AV1HighbdSelfguidedFilterTest, DISABLED_SpeedTest) { RunSpeedTest(); }
TEST_P(AV1HighbdSelfguidedFilterTest, CorrectnessTest) { RunCorrectnessTest(); }
TEST_P(AV1HighbdSelfguidedFilterTest, ExtremeValuesTest) {
  RunExtremeValuesTest();
}

#if HAVE_SSE4_1
const int highbd_params_sse4_1[] = { 8, 10, 12 };
INSTANTIATE_TEST_SUITE_P(
    SSE4_1, AV1HighbdSelfguidedFilterTest,
    ::testing::Combine(
        ::testing::Values(av1_apply_selfguided_restoration_sse4_1),
        ::testing::ValuesIn(highbd_params_sse4_1)));
#endif

#if HAVE_AVX2
const int highbd_params_avx2[] = { 8, 10, 12 };
INSTANTIATE_TEST_SUITE_P(
    AVX2, AV1HighbdSelfguidedFilterTest,
    ::testing::Combine(::testing::Values(av1_apply_selfguided_restoration_avx2),
                       ::testing::ValuesIn(highbd_params_avx2)));
#endif

#if HAVE_AVX512 && CONFIG_HIGHWAY
const int highbd_params_avx512[] = { 8, 10, 12 };
INSTANTIATE_TEST_SUITE_P(
    AVX512, AV1HighbdSelfguidedFilterTest,
    ::testing::Combine(
        ::testing::Values(av1_apply_selfguided_restoration_avx512),
        ::testing::ValuesIn(highbd_params_avx512)));
#endif

#if HAVE_NEON
const int highbd_params_neon[] = { 8, 10, 12 };
INSTANTIATE_TEST_SUITE_P(
    NEON, AV1HighbdSelfguidedFilterTest,
    ::testing::Combine(::testing::Values(av1_apply_selfguided_restoration_neon),
                       ::testing::ValuesIn(highbd_params_neon)));
#endif
#endif  // CONFIG_AV1_HIGHBITDEPTH
}  // namespace
