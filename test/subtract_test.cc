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
#include <tuple>

#include "gtest/gtest.h"

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"

#include "test/acm_random.h"
#include "test/register_state_check.h"
#include "test/util.h"
#include "av1/common/blockd.h"
#include "aom_mem/aom_mem.h"
#include "aom_ports/mem.h"

using SubtractFunc = void (*)(int rows, int cols, int16_t *diff_ptr,
                              ptrdiff_t diff_stride, const uint8_t *src_ptr,
                              ptrdiff_t src_stride, const uint8_t *pred_ptr,
                              ptrdiff_t pred_stride);

namespace {

using std::get;
using std::make_tuple;
using std::tuple;

using libaom_test::ACMRandom;

// <BLOCK_SIZE, optimized subtract func, reference subtract func>
using Params = tuple<BLOCK_SIZE, SubtractFunc, SubtractFunc>;

class AV1SubtractBlockTestBase : public ::testing::Test {
 public:
  AV1SubtractBlockTestBase(BLOCK_SIZE bs, int bit_depth, SubtractFunc func,
                           SubtractFunc ref_func) {
    block_width_ = block_size_wide[bs];
    block_height_ = block_size_high[bs];
    func_ = func;
    ref_func_ = ref_func;
    if (bit_depth == -1) {
      hbd_ = false;
      bit_depth_ = AOM_BITS_8;
    } else {
      hbd_ = true;
      bit_depth_ = static_cast<aom_bit_depth_t>(bit_depth);
    }
  }

  void SetUp() override {
    rnd_.Reset(ACMRandom::DeterministicSeed());

    if (hbd_) {
      src_ = CONVERT_TO_BYTEPTR(reinterpret_cast<uint16_t *>(
          aom_memalign(32, kMaxBufElems * sizeof(uint16_t))));
      ASSERT_NE(src_, nullptr);
      pred_ = CONVERT_TO_BYTEPTR(reinterpret_cast<uint16_t *>(
          aom_memalign(32, kMaxBufElems * sizeof(uint16_t))));
      ASSERT_NE(pred_, nullptr);
    } else {
      src_ = reinterpret_cast<uint8_t *>(
          aom_memalign(32, kMaxBufElems * sizeof(uint8_t)));
      ASSERT_NE(src_, nullptr);
      pred_ = reinterpret_cast<uint8_t *>(
          aom_memalign(32, kMaxBufElems * sizeof(uint8_t)));
      ASSERT_NE(pred_, nullptr);
    }
    diff_ = reinterpret_cast<int16_t *>(
        aom_memalign(32, kMaxBufElems * sizeof(int16_t)));
    ASSERT_NE(diff_, nullptr);
  }

  void TearDown() override {
    if (hbd_) {
      aom_free(CONVERT_TO_SHORTPTR(src_));
      aom_free(CONVERT_TO_SHORTPTR(pred_));
    } else {
      aom_free(src_);
      aom_free(pred_);
    }
    aom_free(diff_);
  }

 protected:
  static constexpr int kMaxHeight = 128;
  static constexpr int kMaxStride = 256;
  static constexpr int kMaxOffset = 64;
  static constexpr size_t kMaxBufElems =
      static_cast<size_t>(kMaxHeight * kMaxStride + kMaxOffset);

  void CheckResult();
  void RunForSpeed();

 private:
  void FillInputs(size_t count);

  ACMRandom rnd_;
  int block_height_;
  int block_width_;
  bool hbd_;
  aom_bit_depth_t bit_depth_;
  SubtractFunc func_;
  SubtractFunc ref_func_;
  uint8_t *src_;
  uint8_t *pred_;
  int16_t *diff_;
};

void AV1SubtractBlockTestBase::FillInputs(size_t count) {
  if (hbd_) {
    const int mask = (1 << bit_depth_) - 1;
    uint16_t *const src16 = CONVERT_TO_SHORTPTR(src_);
    uint16_t *const pred16 = CONVERT_TO_SHORTPTR(pred_);
    for (size_t i = 0; i < count; ++i) {
      src16[i] = rnd_.Rand16() & mask;
      pred16[i] = rnd_.Rand16() & mask;
    }
  } else {
    for (size_t i = 0; i < count; ++i) {
      src_[i] = rnd_.Rand8();
      pred_[i] = rnd_.Rand8();
    }
  }
}

void AV1SubtractBlockTestBase::CheckResult() {
  const int test_num = 100;
  // Encoder residual sub-blocks keep diff_ptr and diff_stride aligned to
  // min(cols, 16) int16_t elements (8B for cols==4, 16B for cols==8, 32B for
  // cols>=16).
  const int diff_align = (block_width_ < 16) ? block_width_ : 16;
  bool sse2_aligned_src = false;
#if HAVE_SSE2 && HAVE_X86_ASM
  if (!hbd_ && func_ == &aom_subtract_block_sse2 && block_width_ >= 16) {
    sse2_aligned_src = true;
  }
#endif

  for (int i = 0; i < test_num; ++i) {
    ptrdiff_t src_stride = block_width_;
    ptrdiff_t pred_stride = block_width_;
    ptrdiff_t diff_stride = block_width_;
    int src_off = 0;
    int pred_off = 0;
    int diff_off = 0;
    if (i > 0) {
      if (sse2_aligned_src) {
        src_stride = block_width_ + (rnd_(5) * 16);
        src_off = rnd_(4) * 16;
      } else {
        src_stride = block_width_ + rnd_(65);
        src_off = rnd_(33);
      }
      pred_stride = block_width_ + rnd_(65);
      pred_off = rnd_(33);
      // Ensure diff_stride >= diff_align so cols<16 sub-block strides can also
      // be wider (e.g. diff_stride == 16..192 for cols == 4 or 8).
      diff_stride = ((block_width_ + diff_align - 1) & ~(diff_align - 1)) +
                    rnd_(5) * diff_align;
      diff_off = rnd_(4) * diff_align;
    }

    const size_t src_elems =
        static_cast<size_t>(src_off + block_height_ * src_stride);
    const size_t pred_elems =
        static_cast<size_t>(pred_off + block_height_ * pred_stride);
    const size_t diff_elems =
        static_cast<size_t>(diff_off + block_height_ * diff_stride);
    ASSERT_LE(src_elems, kMaxBufElems);
    ASSERT_LE(pred_elems, kMaxBufElems);
    ASSERT_LE(diff_elems, kMaxBufElems);

    FillInputs(src_elems > pred_elems ? src_elems : pred_elems);
    static constexpr int16_t kDiffCanary = 0x7abc;
    for (size_t k = 0; k < diff_elems; ++k) diff_[k] = kDiffCanary;

    const uint8_t *const src_ptr =
        hbd_ ? CONVERT_TO_BYTEPTR(CONVERT_TO_SHORTPTR(src_) + src_off)
             : (src_ + src_off);
    const uint8_t *const pred_ptr =
        hbd_ ? CONVERT_TO_BYTEPTR(CONVERT_TO_SHORTPTR(pred_) + pred_off)
             : (pred_ + pred_off);
    int16_t *const diff_ptr = diff_ + diff_off;

    API_REGISTER_STATE_CHECK(func_(block_height_, block_width_, diff_ptr,
                                   diff_stride, src_ptr, src_stride, pred_ptr,
                                   pred_stride));

    for (int k = 0; k < diff_off; ++k) {
      ASSERT_EQ(diff_[k], kDiffCanary)
          << "clobbered before diff_ptr, test " << i;
    }
    if (hbd_) {
      const uint16_t *const s16 = CONVERT_TO_SHORTPTR(src_ptr);
      const uint16_t *const p16 = CONVERT_TO_SHORTPTR(pred_ptr);
      for (int r = 0; r < block_height_; ++r) {
        for (int c = 0; c < block_width_; ++c) {
          ASSERT_EQ(diff_ptr[r * diff_stride + c],
                    static_cast<int16_t>(s16[r * src_stride + c] -
                                         p16[r * pred_stride + c]))
              << "r = " << r << ", c = " << c << ", test: " << i;
        }
        for (ptrdiff_t c = block_width_; c < diff_stride; ++c) {
          ASSERT_EQ(diff_ptr[r * diff_stride + c], kDiffCanary)
              << "padding clobbered r = " << r << ", c = " << c;
        }
      }
    } else {
      for (int r = 0; r < block_height_; ++r) {
        for (int c = 0; c < block_width_; ++c) {
          ASSERT_EQ(diff_ptr[r * diff_stride + c],
                    src_ptr[r * src_stride + c] - pred_ptr[r * pred_stride + c])
              << "r = " << r << ", c = " << c << ", test: " << i;
        }
        for (ptrdiff_t c = block_width_; c < diff_stride; ++c) {
          ASSERT_EQ(diff_ptr[r * diff_stride + c], kDiffCanary)
              << "padding clobbered r = " << r << ", c = " << c;
        }
      }
    }
  }
}

void AV1SubtractBlockTestBase::RunForSpeed() {
  const int test_num = 200000;
  int i;

  if (ref_func_ == func_) GTEST_SKIP();

  const size_t block_elems = static_cast<size_t>(block_width_ * block_height_);
  FillInputs(block_elems);

  aom_usec_timer ref_timer;
  aom_usec_timer_start(&ref_timer);
  for (i = 0; i < test_num; ++i) {
    ref_func_(block_height_, block_width_, diff_, block_width_, src_,
              block_width_, pred_, block_width_);
  }
  aom_usec_timer_mark(&ref_timer);
  const int64_t ref_elapsed_time = aom_usec_timer_elapsed(&ref_timer);

  FillInputs(block_elems);

  aom_usec_timer timer;
  aom_usec_timer_start(&timer);
  for (i = 0; i < test_num; ++i) {
    func_(block_height_, block_width_, diff_, block_width_, src_, block_width_,
          pred_, block_width_);
  }
  aom_usec_timer_mark(&timer);
  const int64_t elapsed_time = aom_usec_timer_elapsed(&timer);

  printf(
      "[%dx%d]: "
      "ref_time=%6" PRId64 " \t simd_time=%6" PRId64
      " \t "
      "gain=%f \n",
      block_width_, block_height_, ref_elapsed_time, elapsed_time,
      static_cast<double>(ref_elapsed_time) /
          static_cast<double>(elapsed_time));
}

class AV1SubtractBlockTest : public ::testing::WithParamInterface<Params>,
                             public AV1SubtractBlockTestBase {
 public:
  AV1SubtractBlockTest()
      : AV1SubtractBlockTestBase(GET_PARAM(0), -1, GET_PARAM(1), GET_PARAM(2)) {
  }
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AV1SubtractBlockTest);

TEST_P(AV1SubtractBlockTest, CheckResult) { CheckResult(); }
TEST_P(AV1SubtractBlockTest, DISABLED_Speed) { RunForSpeed(); }

const BLOCK_SIZE kValidBlockSize[] = {
  BLOCK_4X4,     BLOCK_4X8,   BLOCK_8X4,   BLOCK_8X8,    BLOCK_8X16,
  BLOCK_16X8,    BLOCK_16X16, BLOCK_16X32, BLOCK_32X16,  BLOCK_32X32,
  BLOCK_32X64,   BLOCK_64X32, BLOCK_64X64, BLOCK_64X128, BLOCK_128X64,
  BLOCK_128X128, BLOCK_4X16,  BLOCK_16X4,  BLOCK_8X32,   BLOCK_32X8,
  BLOCK_16X64,   BLOCK_64X16
};

INSTANTIATE_TEST_SUITE_P(
    C, AV1SubtractBlockTest,
    ::testing::Combine(::testing::ValuesIn(kValidBlockSize),
                       ::testing::Values(&aom_subtract_block_c),
                       ::testing::Values(&aom_subtract_block_c)));

#if HAVE_SSE2 && HAVE_X86_ASM
INSTANTIATE_TEST_SUITE_P(
    SSE2, AV1SubtractBlockTest,
    ::testing::Combine(::testing::ValuesIn(kValidBlockSize),
                       ::testing::Values(&aom_subtract_block_sse2),
                       ::testing::Values(&aom_subtract_block_c)));
#endif
#if HAVE_AVX2
INSTANTIATE_TEST_SUITE_P(
    AVX2, AV1SubtractBlockTest,
    ::testing::Combine(::testing::ValuesIn(kValidBlockSize),
                       ::testing::Values(&aom_subtract_block_avx2),
                       ::testing::Values(&aom_subtract_block_c)));

#endif
#if HAVE_NEON
INSTANTIATE_TEST_SUITE_P(
    NEON, AV1SubtractBlockTest,
    ::testing::Combine(::testing::ValuesIn(kValidBlockSize),
                       ::testing::Values(&aom_subtract_block_neon),
                       ::testing::Values(&aom_subtract_block_c)));

#endif

#if CONFIG_AV1_HIGHBITDEPTH

// <BLOCK_SIZE, bit_depth, optimized subtract func, reference subtract func>
using ParamsHBD = tuple<BLOCK_SIZE, int, SubtractFunc, SubtractFunc>;

class AV1HBDSubtractBlockTest : public ::testing::WithParamInterface<ParamsHBD>,
                                public AV1SubtractBlockTestBase {
 public:
  AV1HBDSubtractBlockTest()
      : AV1SubtractBlockTestBase(GET_PARAM(0), GET_PARAM(1), GET_PARAM(2),
                                 GET_PARAM(3)) {}
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AV1HBDSubtractBlockTest);

INSTANTIATE_TEST_SUITE_P(
    C, AV1HBDSubtractBlockTest,
    ::testing::Combine(::testing::ValuesIn(kValidBlockSize),
                       ::testing::Values(12),
                       ::testing::Values(&aom_highbd_subtract_block_c),
                       ::testing::Values(&aom_highbd_subtract_block_c)));

#if HAVE_SSE2
INSTANTIATE_TEST_SUITE_P(
    SSE2, AV1HBDSubtractBlockTest,
    ::testing::Combine(::testing::ValuesIn(kValidBlockSize),
                       ::testing::Values(12),
                       ::testing::Values(&aom_highbd_subtract_block_sse2),
                       ::testing::Values(&aom_highbd_subtract_block_c)));
#endif  // HAVE_SSE2

#if HAVE_NEON
INSTANTIATE_TEST_SUITE_P(
    NEON, AV1HBDSubtractBlockTest,
    ::testing::Combine(::testing::ValuesIn(kValidBlockSize),
                       ::testing::Values(12),
                       ::testing::Values(&aom_highbd_subtract_block_neon),
                       ::testing::Values(&aom_highbd_subtract_block_c)));
#endif
#endif  // CONFIG_AV1_HIGHBITDEPTH
}  // namespace
