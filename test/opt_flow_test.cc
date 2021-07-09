/*
 * Copyright (c) 2021, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <set>
#include <vector>
#include "config/av1_rtcd.h"
#include "config/aom_dsp_rtcd.h"
#include "test/acm_random.h"
#include "test/clear_system_state.h"
#include "third_party/googletest/src/googletest/include/gtest/gtest.h"

#include "aom_ports/aom_timer.h"
#include "av1/common/reconinter.h"

#if CONFIG_OPTFLOW_REFINEMENT
namespace {

class BlockSize {
 public:
  BlockSize(int w, int h) : width_(w), height_(h) {
    n_ = (w <= 16 && h <= 16) ? OF_MIN_BSIZE : OF_BSIZE;
  }

  int Width() const { return width_; }
  int Height() const { return height_; }
  int OptFlowBlkSize() const { return n_; }

  bool operator<(const BlockSize &other) const {
    if (Width() == other.Width()) {
      return Height() < other.Height();
    }
    return Width() < other.Width();
  }

  bool operator==(const BlockSize &other) const {
    return Width() == other.Width() && Height() == other.Height();
  }

 private:
  int width_;
  int height_;
  int n_;
};

// Block size / bit depth / test function used to parameterize the tests.
template <typename T>
class TestParam {
 public:
  TestParam(const BlockSize &block, int bd, T test_func)
      : block_(block), bd_(bd), test_func_(test_func) {}

  const BlockSize &Block() const { return block_; }
  int BitDepth() const { return bd_; }
  T TestFunction() const { return test_func_; }

  bool operator==(const TestParam &other) const {
    return Block() == other.Block() && BitDepth() == other.BitDepth() &&
           TestFunction() == other.TestFunction();
  }

 private:
  BlockSize block_;
  int bd_;
  T test_func_;
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const TestParam<T> &test_arg) {
  return os << "TestParam { width:" << test_arg.Block().Width()
            << " height:" << test_arg.Block().Height()
            << " bd:" << test_arg.BitDepth() << " }";
}

// AV1OptFlowTest is the base class that all optical flow tests should derive
// from.
template <typename T>
class AV1OptFlowTest : public ::testing::TestWithParam<TestParam<T>> {
 public:
  virtual ~AV1OptFlowTest() { TearDown(); }

  virtual void SetUp() override {
    rnd_.Reset(libaom_test::ACMRandom::DeterministicSeed());
  }

  virtual void TearDown() override { libaom_test::ClearSystemState(); }

  // Check that two 8-bit output buffers are identical.
  void AssertOutputEq(const int *ref, const int *test, int n) {
    ASSERT_TRUE(ref != test) << "Buffers must be at different memory locations";
    for (int idx = 0; idx < n; ++idx) {
      ASSERT_EQ(ref[idx], test[idx]) << "Mismatch at index " << idx;
    }
  }

  // Check that two 16-bit output buffers are identical.
  void AssertOutputBufferEq(const int16_t *ref, const int16_t *test, int width,
                            int height) {
    ASSERT_TRUE(ref != test) << "Buffers must be in different memory locations";
    for (int row = 0; row < height; ++row) {
      for (int col = 0; col < width; ++col) {
        ASSERT_EQ(ref[row * width + col], test[row * width + col])
            << width << "x" << height << " Pixel mismatch at (" << col << ", "
            << row << ")";
      }
    }
  }

  uint8_t RandomFrameIdx(int max_bit_range) {
    const int max_val = (1 << max_bit_range) - 1;
    uint8_t rand_val = rnd_.Rand8() & max_val;
    return rand_val;
  }

  int8_t RelativeDistExtreme(int max_bit_range) {
    return Rand8SingedExtremes(max_bit_range);
  }

  void RandomInput8(uint8_t *p, const TestParam<T> &param) {
    EXPECT_EQ(8, param.BitDepth());
    EXPECT_GE(MAX_SB_SIZE, param.Block().Width());
    EXPECT_GE(MAX_SB_SIZE, param.Block().Height());
    const int bw = param.Block().Width();
    const int bh = param.Block().Height();
    Randomize(p, bw * bh);
  }

  void RandomInput16(uint16_t *p, const TestParam<T> &param,
                     int max_bit_range) {
    EXPECT_GE(12, param.BitDepth());
    EXPECT_GE(MAX_SB_SIZE, param.Block().Width());
    EXPECT_GE(MAX_SB_SIZE, param.Block().Height());
    const int bw = param.Block().Width();
    const int bh = param.Block().Height();
    Randomize(p, bw * bh, max_bit_range);
  }

  void RandomInput16(int16_t *p, const TestParam<T> &param, int max_bit_range) {
    EXPECT_GE(MAX_SB_SIZE, param.Block().Width());
    EXPECT_GE(MAX_SB_SIZE, param.Block().Height());
    const int bw = param.Block().Width();
    const int bh = param.Block().Height();
    Randomize(p, bw * bh, max_bit_range);
  }

  void RandomInput8Extreme(uint8_t *p, const TestParam<T> &param) {
    EXPECT_EQ(8, param.BitDepth());
    EXPECT_GE(MAX_SB_SIZE, param.Block().Width());
    EXPECT_GE(MAX_SB_SIZE, param.Block().Height());
    const int bw = param.Block().Width();
    const int bh = param.Block().Height();
    RandomizeExtreme(p, bw * bh);
  }

  void RandomInput16Extreme(uint16_t *p, const TestParam<T> &param,
                            int max_bit_range) {
    EXPECT_GE(12, param.BitDepth());
    EXPECT_GE(MAX_SB_SIZE, param.Block().Width());
    EXPECT_GE(MAX_SB_SIZE, param.Block().Height());
    const int bw = param.Block().Width();
    const int bh = param.Block().Height();
    RandomizeExtreme(p, bw * bh, max_bit_range);
  }

  void RandomInput16Extreme(int16_t *p, const TestParam<T> &param,
                            int max_bit_range) {
    EXPECT_GE(12, param.BitDepth());
    EXPECT_GE(MAX_SB_SIZE, param.Block().Width());
    EXPECT_GE(MAX_SB_SIZE, param.Block().Height());
    const int bw = param.Block().Width();
    const int bh = param.Block().Height();
    RandomizeExtreme(p, bw * bh, max_bit_range);
  }

 private:
  void Randomize(uint8_t *p, int size) {
    for (int i = 0; i < size; ++i) {
      p[i] = rnd_.Rand8();
    }
  }

  void Randomize(uint16_t *p, int size, int max_bit_range) {
    assert(max_bit_range < 16);
    for (int i = 0; i < size; ++i) {
      p[i] = rnd_.Rand16() & ((1 << max_bit_range) - 1);
    }
  }

  void Randomize(int16_t *p, int size, int max_bit_range) {
    assert(max_bit_range < 16);
    for (int i = 0; i < size; ++i) {
      p[i] = rnd_.Rand15Signed() & ((1 << max_bit_range) - 1);
    }
  }

  int RandBool() {
    const uint32_t value = rnd_.Rand8();
    // There's a bit more entropy in the upper bits of this implementation.
    return (value >> 7) & 0x1;
  }

  uint8_t Rand8Extremes() { return static_cast<uint8_t>(RandBool() ? 255 : 0); }

  int8_t Rand8SingedExtremes(int max_bit_range) {
    const int max_val = (1 << max_bit_range) - 1;
    const int half_max_val = 1 << (max_bit_range - 1);
    uint8_t r_u8 = Rand8Extremes() & max_val;
    return static_cast<int8_t>(r_u8 - half_max_val);
  }

  uint16_t Rand16Extremes(int max_bit_range) {
    const int max_val = (1 << max_bit_range) - 1;
    return static_cast<uint16_t>(RandBool() ? max_val : 0);
  }

  int16_t Rand16SingedExtremes(int max_bit_range) {
    const int half_max_val = 1 << (max_bit_range - 1);
    uint16_t r_u16 = Rand16Extremes(max_bit_range);
    return static_cast<int16_t>(r_u16 - half_max_val);
  }

  void RandomizeExtreme(uint8_t *p, int size) {
    for (int i = 0; i < size; ++i) {
      p[i] = Rand8Extremes();
    }
  }

  void RandomizeExtreme(uint16_t *p, int size, int max_bit_range) {
    for (int i = 0; i < size; ++i) {
      p[i] = Rand16Extremes(max_bit_range);
    }
  }

  void RandomizeExtreme(int16_t *p, int size, int max_bit_range) {
    for (int i = 0; i < size; ++i) {
      p[i] = Rand16SingedExtremes(max_bit_range);
    }
  }

  libaom_test::ACMRandom rnd_;
};

// a function to generate test parameters for just luma block sizes.
template <typename T>
std::vector<TestParam<T>> GetOptFlowTestParams(
    std::initializer_list<int> bit_depths, T test_func) {
  std::set<BlockSize> sizes;
  for (int bsize = BLOCK_8X8; bsize < BLOCK_SIZES_ALL; ++bsize) {
    const int w = block_size_wide[bsize];
    const int h = block_size_high[bsize];
    if (w < 8 || h < 8) continue;
    sizes.insert(BlockSize(w, h));
  }
  std::vector<TestParam<T>> result;
  for (int bit_depth : bit_depths) {
    for (const auto &block : sizes) {
      result.push_back(TestParam<T>(block, bit_depth, test_func));
    }
  }
  return result;
}

template <typename T>
std::vector<TestParam<T>> GetOptFlowLowbdTestParams(T test_func) {
  return GetOptFlowTestParams({ 8 }, test_func);
}

template <typename T>
::testing::internal::ParamGenerator<TestParam<T>> BuildOptFlowParams(
    T test_func) {
  return ::testing::ValuesIn(GetOptFlowLowbdTestParams(test_func));
}

#if OPFL_BICUBIC_GRAD
typedef void (*bicubic_grad_interp_lowbd)(const int16_t *pred_src,
                                          int16_t *x_grad, int16_t *y_grad,
                                          const int blk_width,
                                          const int blk_height);

class AV1OptFlowBiCubicGradLowbdTest
    : public AV1OptFlowTest<bicubic_grad_interp_lowbd> {
 public:
  AV1OptFlowBiCubicGradLowbdTest() {
    const BlockSize &block = GetParam().Block();
    const int bw = block.Width();
    const int bh = block.Height();

    pred_src_ = (int16_t *)aom_memalign(16, bw * bh * sizeof(int16_t));
    x_grad_ref_ = (int16_t *)aom_memalign(16, bw * bh * sizeof(int16_t));
    y_grad_ref_ = (int16_t *)aom_memalign(16, bw * bh * sizeof(int16_t));
    x_grad_test_ = (int16_t *)aom_memalign(16, bw * bh * sizeof(int16_t));
    y_grad_test_ = (int16_t *)aom_memalign(16, bw * bh * sizeof(int16_t));

    memset(x_grad_ref_, 0, bw * bh * sizeof(int16_t));
    memset(y_grad_ref_, 0, bw * bh * sizeof(int16_t));
    memset(x_grad_test_, 0, bw * bh * sizeof(int16_t));
    memset(y_grad_test_, 0, bw * bh * sizeof(int16_t));
  }

  ~AV1OptFlowBiCubicGradLowbdTest() {
    aom_free(pred_src_);
    aom_free(x_grad_ref_);
    aom_free(y_grad_ref_);
    aom_free(x_grad_test_);
    aom_free(y_grad_test_);
  }

  void RunTest(const int is_speed) {
    const BlockSize &block = GetParam().Block();
    const int bw_log2 = block.Width() >> MI_SIZE_LOG2;
    const int bh_log2 = block.Height() >> MI_SIZE_LOG2;
    const int numIter = is_speed ? 1 : 16384 / (bw_log2 * bh_log2);

    for (int count = 0; count < numIter; count++) {
      RandomInput16(pred_src_, GetParam(), 8);
      TestBicubicGrad(pred_src_, x_grad_ref_, y_grad_ref_, x_grad_test_,
                      y_grad_test_, is_speed);
    }
    if (is_speed) return;

    for (int count = 0; count < numIter; count++) {
      RandomInput16Extreme(pred_src_, GetParam(), 8);
      TestBicubicGrad(pred_src_, x_grad_ref_, y_grad_ref_, x_grad_test_,
                      y_grad_test_, 0);
    }
  }

 private:
  void TestBicubicGrad(int16_t *pred_src, int16_t *x_grad_ref,
                       int16_t *y_grad_ref, int16_t *x_grad_test,
                       int16_t *y_grad_test, int is_speed) {
    const BlockSize &block = GetParam().Block();
    const int bw = block.Width();
    const int bh = block.Height();

    bicubic_grad_interp_lowbd ref_func = av1_bicubic_grad_interpolation_c;
    bicubic_grad_interp_lowbd test_func = GetParam().TestFunction();
    if (is_speed)
      BicubicGradSpeed(ref_func, test_func, pred_src, x_grad_ref, y_grad_ref,
                       x_grad_test, y_grad_test, bw, bh);
    else
      BicubicGrad(ref_func, test_func, pred_src, x_grad_ref, y_grad_ref,
                  x_grad_test, y_grad_test, bw, bh);
  }

  void BicubicGrad(bicubic_grad_interp_lowbd ref_func,
                   bicubic_grad_interp_lowbd test_func, const int16_t *pred_src,
                   int16_t *x_grad_ref, int16_t *y_grad_ref,
                   int16_t *x_grad_test, int16_t *y_grad_test, const int bw,
                   const int bh) {
    ref_func(pred_src, x_grad_ref, y_grad_ref, bw, bh);
    test_func(pred_src, x_grad_test, y_grad_test, bw, bh);

    AssertOutputBufferEq(x_grad_ref, x_grad_test, bw, bh);
    AssertOutputBufferEq(y_grad_ref, y_grad_test, bw, bh);
  }

  void BicubicGradSpeed(bicubic_grad_interp_lowbd ref_func,
                        bicubic_grad_interp_lowbd test_func, int16_t *pred_src,
                        int16_t *x_grad_ref, int16_t *y_grad_ref,
                        int16_t *x_grad_test, int16_t *y_grad_test,
                        const int bw, const int bh) {
    const int bw_log2 = bw >> MI_SIZE_LOG2;
    const int bh_log2 = bh >> MI_SIZE_LOG2;

    const int numIter = 2097152 / (bw_log2 * bh_log2);
    aom_usec_timer timer_ref;
    aom_usec_timer timer_test;

    aom_usec_timer_start(&timer_ref);
    for (int count = 0; count < numIter; count++)
      ref_func(pred_src, x_grad_ref, y_grad_ref, bw, bh);
    aom_usec_timer_mark(&timer_ref);

    aom_usec_timer_start(&timer_test);
    for (int count = 0; count < numIter; count++)
      test_func(pred_src, x_grad_test, y_grad_test, bw, bh);
    aom_usec_timer_mark(&timer_test);

    const int total_time_ref =
        static_cast<int>(aom_usec_timer_elapsed(&timer_ref));
    const int total_time_test =
        static_cast<int>(aom_usec_timer_elapsed(&timer_test));

    printf("ref_time = %d \t simd_time = %d \t Gain = %4.2f \n", total_time_ref,
           total_time_test,
           (static_cast<float>(total_time_ref) /
            static_cast<float>(total_time_test)));
  }

  int16_t *pred_src_;
  int16_t *x_grad_ref_;
  int16_t *y_grad_ref_;
  int16_t *x_grad_test_;
  int16_t *y_grad_test_;
};
TEST_P(AV1OptFlowBiCubicGradLowbdTest, CheckOutput) { RunTest(0); }
TEST_P(AV1OptFlowBiCubicGradLowbdTest, DISABLED_Speed) { RunTest(1); }

INSTANTIATE_TEST_SUITE_P(C, AV1OptFlowBiCubicGradLowbdTest,
                         BuildOptFlowParams(av1_bicubic_grad_interpolation_c));

#if HAVE_SSE4_1
INSTANTIATE_TEST_SUITE_P(
    SSE4_1, AV1OptFlowBiCubicGradLowbdTest,
    BuildOptFlowParams(av1_bicubic_grad_interpolation_sse4_1));
#endif

typedef void (*bicubic_grad_interp_highbd)(const int16_t *pred_src,
                                           int16_t *x_grad, int16_t *y_grad,
                                           const int blk_width,
                                           const int blk_height);

class AV1OptFlowBiCubicGradHighbdTest
    : public AV1OptFlowTest<bicubic_grad_interp_highbd> {
 public:
  AV1OptFlowBiCubicGradHighbdTest() {
    const BlockSize &block = GetParam().Block();
    const int bw = block.Width();
    const int bh = block.Height();

    pred_src_ = (int16_t *)aom_memalign(16, bw * bh * sizeof(int16_t));
    x_grad_ref_ = (int16_t *)aom_memalign(16, bw * bh * sizeof(int16_t));
    y_grad_ref_ = (int16_t *)aom_memalign(16, bw * bh * sizeof(int16_t));
    x_grad_test_ = (int16_t *)aom_memalign(16, bw * bh * sizeof(int16_t));
    y_grad_test_ = (int16_t *)aom_memalign(16, bw * bh * sizeof(int16_t));

    memset(x_grad_ref_, 0, bw * bh * sizeof(int16_t));
    memset(y_grad_ref_, 0, bw * bh * sizeof(int16_t));
    memset(x_grad_test_, 0, bw * bh * sizeof(int16_t));
    memset(y_grad_test_, 0, bw * bh * sizeof(int16_t));
  }

  ~AV1OptFlowBiCubicGradHighbdTest() {
    aom_free(pred_src_);
    aom_free(x_grad_ref_);
    aom_free(y_grad_ref_);
    aom_free(x_grad_test_);
    aom_free(y_grad_test_);
  }

  void Run(const int is_speed) {
    const BlockSize &block = GetParam().Block();
    const int bw_log2 = block.Width() >> MI_SIZE_LOG2;
    const int bh_log2 = block.Height() >> MI_SIZE_LOG2;
    const int numIter = is_speed ? 1 : 16384 / (bw_log2 * bh_log2);

    for (int count = 0; count < numIter; count++) {
      RandomInput16(pred_src_, GetParam(), 12);
      TestBicubicGradHighbd(pred_src_, x_grad_ref_, y_grad_ref_, x_grad_test_,
                            y_grad_test_, is_speed);
    }
    if (is_speed) return;

    for (int count = 0; count < numIter; count++) {
      RandomInput16Extreme((uint16_t *)pred_src_, GetParam(), 12);
      TestBicubicGradHighbd(pred_src_, x_grad_ref_, y_grad_ref_, x_grad_test_,
                            y_grad_test_, 0);
    }
  }

 private:
  void TestBicubicGradHighbd(int16_t *pred_src, int16_t *x_grad_ref,
                             int16_t *y_grad_ref, int16_t *x_grad_test,
                             int16_t *y_grad_test, int is_speed) {
    const BlockSize &block = GetParam().Block();
    const int bw = block.Width();
    const int bh = block.Height();

    bicubic_grad_interp_highbd ref_func =
        av1_bicubic_grad_interpolation_highbd_c;
    bicubic_grad_interp_highbd test_func = GetParam().TestFunction();
    if (is_speed)
      BicubicGradHighbdSpeed(ref_func, test_func, pred_src, x_grad_ref,
                             y_grad_ref, x_grad_test, y_grad_test, bw, bh);
    else
      BicubicGradHighbd(ref_func, test_func, pred_src, x_grad_ref, y_grad_ref,
                        x_grad_test, y_grad_test, bw, bh);
  }

  void BicubicGradHighbd(bicubic_grad_interp_highbd ref_func,
                         bicubic_grad_interp_highbd test_func,
                         const int16_t *pred_src, int16_t *x_grad_ref,
                         int16_t *y_grad_ref, int16_t *x_grad_test,
                         int16_t *y_grad_test, const int bw, const int bh) {
    ref_func(pred_src, x_grad_ref, y_grad_ref, bw, bh);
    test_func(pred_src, x_grad_test, y_grad_test, bw, bh);

    AssertOutputBufferEq(x_grad_ref, x_grad_test, bw, bh);
    AssertOutputBufferEq(y_grad_ref, y_grad_test, bw, bh);
  }

  void BicubicGradHighbdSpeed(bicubic_grad_interp_highbd ref_func,
                              bicubic_grad_interp_highbd test_func,
                              int16_t *pred_src, int16_t *x_grad_ref,
                              int16_t *y_grad_ref, int16_t *x_grad_test,
                              int16_t *y_grad_test, const int bw,
                              const int bh) {
    const int bw_log2 = bw >> MI_SIZE_LOG2;
    const int bh_log2 = bh >> MI_SIZE_LOG2;

    const int numIter = 2097152 / (bw_log2 * bh_log2);
    aom_usec_timer timer_ref;
    aom_usec_timer timer_test;

    aom_usec_timer_start(&timer_ref);
    for (int count = 0; count < numIter; count++)
      ref_func(pred_src, x_grad_ref, y_grad_ref, bw, bh);
    aom_usec_timer_mark(&timer_ref);

    aom_usec_timer_start(&timer_test);
    for (int count = 0; count < numIter; count++)
      test_func(pred_src, x_grad_test, y_grad_test, bw, bh);
    aom_usec_timer_mark(&timer_test);

    const int total_time_ref =
        static_cast<int>(aom_usec_timer_elapsed(&timer_ref));
    const int total_time_test =
        static_cast<int>(aom_usec_timer_elapsed(&timer_test));

    printf("ref_time = %d \t simd_time = %d \t Gain = %4.2f \n", total_time_ref,
           total_time_test,
           (static_cast<float>(total_time_ref) /
            static_cast<float>(total_time_test)));
  }

  int16_t *pred_src_;
  int16_t *x_grad_ref_;
  int16_t *y_grad_ref_;
  int16_t *x_grad_test_;
  int16_t *y_grad_test_;
};
TEST_P(AV1OptFlowBiCubicGradHighbdTest, CheckOutput) { Run(0); }
TEST_P(AV1OptFlowBiCubicGradHighbdTest, DISABLED_Speed) { Run(1); }

INSTANTIATE_TEST_SUITE_P(
    C, AV1OptFlowBiCubicGradHighbdTest,
    BuildOptFlowParams(av1_bicubic_grad_interpolation_highbd_c));

#if HAVE_SSE4_1
INSTANTIATE_TEST_SUITE_P(
    SSE4_1, AV1OptFlowBiCubicGradHighbdTest,
    BuildOptFlowParams(av1_bicubic_grad_interpolation_highbd_sse4_1));
#endif
#endif  // OPFL_BICUBIC_GRAD

}  // namespace

#endif  // CONFIG_OPTFLOW_REFINEMENT
