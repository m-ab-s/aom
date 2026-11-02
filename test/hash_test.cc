/*
 * Copyright (c) 2018, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <cstdlib>
#include <new>
#include <tuple>
#include <vector>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"

#include "aom_ports/aom_timer.h"
#include "aom_ports/mem.h"
#include "av1/encoder/hash.h"
#include "av1/encoder/hash_motion.h"
#include "gtest/gtest.h"
#include "test/acm_random.h"
#include "test/util.h"

namespace {

using get_crc32c_value_func = uint32_t (*)(void *calculator, const uint8_t *p,
                                           size_t length);

using HashParam = std::tuple<get_crc32c_value_func, int>;

class AV1Crc32cHashTest : public ::testing::TestWithParam<HashParam> {
 public:
  ~AV1Crc32cHashTest() override;
  void SetUp() override;

  void TearDown() override;

 protected:
  void RunCheckOutput(get_crc32c_value_func test_impl);
  void RunSpeedTest(get_crc32c_value_func test_impl);

  void RunZeroTest(get_crc32c_value_func test_impl);

  libaom_test::ACMRandom rnd_;
  CRC32C calc_;
  uint8_t *buffer_;
  int bsize_;
  size_t length_;
};

AV1Crc32cHashTest::~AV1Crc32cHashTest() = default;

void AV1Crc32cHashTest::SetUp() {
  rnd_.Reset(libaom_test::ACMRandom::DeterministicSeed());
  av1_crc32c_calculator_init(&calc_);

  bsize_ = GET_PARAM(1);
  length_ = bsize_ * bsize_ * sizeof(uint16_t);
  buffer_ = new uint8_t[length_];
  ASSERT_NE(buffer_, nullptr);
  for (size_t i = 0; i < length_; ++i) {
    buffer_[i] = rnd_.Rand8();
  }
}

void AV1Crc32cHashTest::TearDown() { delete[] buffer_; }

void AV1Crc32cHashTest::RunCheckOutput(get_crc32c_value_func test_impl) {
  get_crc32c_value_func ref_impl = av1_get_crc32c_value_c;
  // for the same buffer crc should be the same
  uint32_t crc0 = test_impl(&calc_, buffer_, length_);
  uint32_t crc1 = test_impl(&calc_, buffer_, length_);
  uint32_t crc2 = ref_impl(&calc_, buffer_, length_);
  ASSERT_EQ(crc0, crc1);
  ASSERT_EQ(crc0, crc2);  // should equal to software version
  // modify buffer
  buffer_[0] += 1;
  uint32_t crc3 = test_impl(&calc_, buffer_, length_);
  uint32_t crc4 = ref_impl(&calc_, buffer_, length_);
  ASSERT_NE(crc0, crc3);  // crc shoud not equal to previous one
  ASSERT_EQ(crc3, crc4);
}

void AV1Crc32cHashTest::RunSpeedTest(get_crc32c_value_func test_impl) {
  get_crc32c_value_func impls[] = { av1_get_crc32c_value_c, test_impl };
  const int repeat = 10000000 / (bsize_ + bsize_);

  aom_usec_timer timer;
  double time[2];
  for (int i = 0; i < 2; ++i) {
    aom_usec_timer_start(&timer);
    for (int j = 0; j < repeat; ++j) {
      impls[i](&calc_, buffer_, length_);
    }
    aom_usec_timer_mark(&timer);
    time[i] = static_cast<double>(aom_usec_timer_elapsed(&timer));
  }
  printf("hash %3dx%-3d:%7.2f/%7.2fus", bsize_, bsize_, time[0], time[1]);
  printf("(%3.2f)\n", time[0] / time[1]);
}

void AV1Crc32cHashTest::RunZeroTest(get_crc32c_value_func test_impl) {
  uint8_t buffer0[1024] = { 0 };
  // for buffer with different size the crc should not be the same
  const uint32_t crc0 = test_impl(&calc_, buffer0, 32);
  const uint32_t crc1 = test_impl(&calc_, buffer0, 128);
  const uint32_t crc2 = test_impl(&calc_, buffer0, 1024);
  ASSERT_NE(crc0, crc1);
  ASSERT_NE(crc0, crc2);
  ASSERT_NE(crc1, crc2);
}

TEST_P(AV1Crc32cHashTest, CheckOutput) { RunCheckOutput(GET_PARAM(0)); }

TEST_P(AV1Crc32cHashTest, CheckZero) { RunZeroTest(GET_PARAM(0)); }

TEST_P(AV1Crc32cHashTest, DISABLED_Speed) { RunSpeedTest(GET_PARAM(0)); }

const int kValidBlockSize[] = { 64, 32, 8, 4 };

INSTANTIATE_TEST_SUITE_P(
    C, AV1Crc32cHashTest,
    ::testing::Combine(::testing::Values(&av1_get_crc32c_value_c),
                       ::testing::ValuesIn(kValidBlockSize)));

#if HAVE_SSE4_2
INSTANTIATE_TEST_SUITE_P(
    SSE4_2, AV1Crc32cHashTest,
    ::testing::Combine(::testing::Values(&av1_get_crc32c_value_sse4_2),
                       ::testing::ValuesIn(kValidBlockSize)));
#endif

#if HAVE_ARM_CRC32
INSTANTIATE_TEST_SUITE_P(
    ARM_CRC32, AV1Crc32cHashTest,
    ::testing::Combine(::testing::Values(&av1_get_crc32c_value_arm_crc32),
                       ::testing::ValuesIn(kValidBlockSize)));
#endif

using combine_block_hashes_func = void (*)(void *crc_calculator,
                                           const uint32_t *top,
                                           const uint32_t *bottom,
                                           uint32_t *dst, int width,
                                           int sub_size);

class AV1CombineBlockHashesTest
    : public ::testing::TestWithParam<combine_block_hashes_func> {
 protected:
  void SetUp() override {
    rnd_.Reset(libaom_test::ACMRandom::DeterministicSeed());
    av1_crc32c_calculator_init(&calc_);
  }

  uint32_t Rand32() {
    return static_cast<uint32_t>(rnd_.Rand16()) << 16 | rnd_.Rand16();
  }

  // Compares the whole output row with the C version, so writes past width
  // show up too. The two source rows are separate buffers, like rows of a ring
  // buffer. flat fills them with few values, like a screen with large flat
  // areas.
  void RunCheckOutput(int width, int sub_size, bool flat) {
    const int size = width + sub_size + 3;
    std::vector<uint32_t> top(size);
    std::vector<uint32_t> bottom(size);
    for (int i = 0; i < size; ++i) {
      top[i] = flat ? rnd_.Rand8() % 2 : Rand32();
      bottom[i] = flat ? rnd_.Rand8() % 2 : Rand32();
    }
    std::vector<uint32_t> ref(size, 0xDEADBEEF);
    std::vector<uint32_t> out(size, 0xDEADBEEF);
    av1_combine_block_hashes_c(&calc_, top.data(), bottom.data(), ref.data(),
                               width, sub_size);
    GetParam()(&calc_, top.data(), bottom.data(), out.data(), width, sub_size);
    ASSERT_EQ(ref, out) << "width " << width << " sub_size " << sub_size
                        << " flat " << flat;
  }

  libaom_test::ACMRandom rnd_;
  CRC32C calc_;
};

TEST_P(AV1CombineBlockHashesTest, CheckOutput) {
  const int kSubSizes[] = { 2, 4, 8, 16, 32, 64 };
  const int kWidths[] = { 0, 1, 7, 64, 129 };
  for (int sub_size : kSubSizes) {
    for (int width : kWidths) {
      for (int i = 0; i < 5; ++i) {
        RunCheckOutput(width, sub_size, false);
        RunCheckOutput(width, sub_size, true);
      }
    }
  }
}

TEST_P(AV1CombineBlockHashesTest, DISABLED_Speed) {
  // One 4x4 pass over a 2560x1440 frame.
  const int stride = 2560;
  const int height = 1440;
  const int sub_size = 2;
  std::vector<uint32_t> src(stride * height);
  for (uint32_t &v : src) v = Rand32();
  std::vector<uint32_t> dst(stride * height);
  const combine_block_hashes_func impls[] = { av1_combine_block_hashes_c,
                                              GetParam() };
  const int width = stride - 2 * sub_size + 1;
  const int rows = height - 2 * sub_size + 1;
  const int repeat = 20;

  double time[2];
  for (int i = 0; i < 2; ++i) {
    aom_usec_timer timer;
    aom_usec_timer_start(&timer);
    for (int j = 0; j < repeat; ++j) {
      for (int y = 0; y < rows; ++y) {
        const uint32_t *top = src.data() + y * stride;
        impls[i](&calc_, top, top + sub_size * stride, dst.data() + y * stride,
                 width, sub_size);
      }
    }
    aom_usec_timer_mark(&timer);
    time[i] = static_cast<double>(aom_usec_timer_elapsed(&timer));
  }
  printf("combine 2560x1440: %8.2f/%8.2fus (%3.2f)\n", time[0], time[1],
         time[0] / time[1]);
}

INSTANTIATE_TEST_SUITE_P(C, AV1CombineBlockHashesTest,
                         ::testing::Values(&av1_combine_block_hashes_c));

#if HAVE_SSE4_2
INSTANTIATE_TEST_SUITE_P(SSE4_2, AV1CombineBlockHashesTest,
                         ::testing::Values(&av1_combine_block_hashes_sse4_2));
#endif

#if HAVE_ARM_CRC32
INSTANTIATE_TEST_SUITE_P(
    ARM_CRC32, AV1CombineBlockHashesTest,
    ::testing::Values(&av1_combine_block_hashes_arm_crc32));
#endif

// Checks the frame pass for 4x4 and 8x8 blocks against
// av1_get_block_hash_value(), which hashes one block from its pixels, the way
// the search does. The regions cover the right and bottom borders, offsets and
// regions too small for any block.
TEST(AV1BlockHashValueTest, MatchesSingleBlockHash) {
  libaom_test::ACMRandom rnd(libaom_test::ACMRandom::DeterministicSeed());
  const int kWidth = 75;
  const int kHeight = 41;
  const int kStride = kWidth + 5;
  const int kHashStride = kWidth + 9;
  const uint32_t kUnset = 0xDEADBEEF;
  const IntraBCHashRegion kRegions[] = {
    { 0, 0, kWidth, kHeight },
    { 3, 5, 10, 12 },
    { 0, 0, 4, 4 },
    { 1, 2, 8, 8 },
    { 5, 7, 12, 40 },
    { 64, 30, kWidth, kHeight },
    { 0, 0, 7, kHeight },
    { 2, 0, kWidth, 3 },
    { 9, 9, 9, 20 },
  };

  IntraBCHashInfo info = {};
  av1_hash_table_init(&info);
  std::vector<uint32_t> buf0(AOM_BUFFER_SIZE_FOR_BLOCK_HASH);
  std::vector<uint32_t> buf1(AOM_BUFFER_SIZE_FOR_BLOCK_HASH);
  info.hash_value_buffer[0] = buf0.data();
  info.hash_value_buffer[1] = buf1.data();

  std::vector<uint8_t> pixels8(kStride * kHeight);
  std::vector<uint16_t> pixels16(kStride * kHeight);
  for (int hbd = 0; hbd <= 1; ++hbd) {
    // Half of the pixels share one value, like the flat areas of a screen.
    for (int i = 0; i < kStride * kHeight; ++i) {
      const bool flat = rnd.Rand8() < 128;
      pixels8[i] = flat ? 7 : rnd.Rand8();
      pixels16[i] = flat ? 7 : rnd.Rand16() & 0x3ff;
    }
    YV12_BUFFER_CONFIG pic = {};
    pic.y_stride = kStride;
    if (hbd) {
      pic.y_buffer = CONVERT_TO_BYTEPTR(pixels16.data());
      pic.flags = YV12_FLAG_HIGHBITDEPTH;
    } else {
      pic.y_buffer = pixels8.data();
    }

    for (const IntraBCHashRegion &region : kRegions) {
      for (int with_4x4 = 0; with_4x4 <= 1; ++with_4x4) {
        std::vector<uint32_t> hash_4x4(kHashStride * kHeight, kUnset);
        std::vector<uint32_t> hash_8x8(kHashStride * kHeight, kUnset);
        ASSERT_TRUE(av1_generate_block_8x8_hash_value(
            &info, &pic, &region, with_4x4 ? hash_4x4.data() : nullptr,
            hash_8x8.data(), kHashStride));
        for (int size = 4; size <= 8; size *= 2) {
          const std::vector<uint32_t> &hashes = size == 4 ? hash_4x4 : hash_8x8;
          for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
              const bool inside = x >= region.x_start && y >= region.y_start &&
                                  x + size <= region.x_end &&
                                  y + size <= region.y_end;
              uint32_t expected = kUnset;
              if (inside && (size == 8 || with_4x4)) {
                uint32_t hash_value1;
                av1_get_block_hash_value(&info, pic.y_buffer + y * kStride + x,
                                         kStride, size, &hash_value1, &expected,
                                         hbd);
              }
              ASSERT_EQ(expected, hashes[y * kHashStride + x])
                  << "hbd " << hbd << " size " << size << " x " << x << " y "
                  << y << " region " << region.x_start << "," << region.y_start
                  << " " << region.x_end << "," << region.y_end;
            }
          }
        }
      }
    }
  }
}

}  // namespace
