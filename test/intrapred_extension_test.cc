/*
 * Copyright (c) 2020, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <algorithm>
#include <functional>
#include <memory>
#include <unordered_set>
#include "av1/common/reconintra.h"
#include "test/acm_random.h"
#include "test/clear_system_state.h"
#include "third_party/googletest/src/googletest/include/gtest/gtest.h"

namespace {

class TestParam {
 public:
  TestParam(int width, int height, int dst_stride_padding,
            int ref_stride_padding, int border, aom_bit_depth_t bd, bool is_hbd,
            int top_available, int right_unavailable, int left_available,
            int bottom_unavailable)
      : width_(width), height_(height), dst_stride_padding_(dst_stride_padding),
        ref_stride_padding_(ref_stride_padding), border_(border), bd_(bd),
        is_hbd_(is_hbd), top_available_(top_available),
        right_unavailable_(right_unavailable), left_available_(left_available),
        bottom_unavailable_(bottom_unavailable) {}

  int Width() const { return width_; }
  int Height() const { return height_; }
  int DstStridePadding() const { return dst_stride_padding_; }
  int RefStridePadding() const { return ref_stride_padding_; }
  int Border() const { return border_; }
  aom_bit_depth_t BitDepth() const { return bd_; }
  bool IsHighBitDepth() const { return is_hbd_; }
  int TopAvailable() const { return top_available_; }
  int RightUnavailable() const { return right_unavailable_; }
  int LeftAvailable() const { return left_available_; }
  int BottomUnavailable() const { return bottom_unavailable_; }

  bool operator==(const TestParam &other) const {
    return Width() == other.Width() && Height() == other.Height() &&
           DstStridePadding() == other.DstStridePadding() &&
           RefStridePadding() == other.RefStridePadding() &&
           Border() == other.Border() && BitDepth() == other.BitDepth() &&
           IsHighBitDepth() == other.IsHighBitDepth() &&
           TopAvailable() == other.TopAvailable() &&
           RightUnavailable() == other.RightUnavailable() &&
           LeftAvailable() == other.LeftAvailable() &&
           BottomUnavailable() == other.BottomUnavailable();
  }

 private:
  int width_;
  int height_;
  int dst_stride_padding_;
  int ref_stride_padding_;
  int border_;
  aom_bit_depth_t bd_;
  bool is_hbd_;
  int top_available_;
  int right_unavailable_;
  int left_available_;
  int bottom_unavailable_;
};

struct TestParamHash {
  size_t operator()(const TestParam &p) const {
    // Arbitrary primes of 17 and 19 for initial value and multiplier.
    size_t result = 17 + std::hash<int>{}(p.Width());
    result = result * 19 + std::hash<int>{}(p.Height());
    result = result * 19 + std::hash<int>{}(p.DstStridePadding());
    result = result * 19 + std::hash<int>{}(p.RefStridePadding());
    result = result * 19 + std::hash<int>{}(p.Border());
    result = result * 19 + std::hash<int>{}(static_cast<int>(p.BitDepth()));
    result = result * 19 + std::hash<bool>{}(p.IsHighBitDepth());
    result = result * 19 + std::hash<int>{}(p.TopAvailable());
    result = result * 19 + std::hash<int>{}(p.RightUnavailable());
    result = result * 19 + std::hash<int>{}(p.LeftAvailable());
    result = result * 19 + std::hash<int>{}(p.BottomUnavailable());
    return result;
  }
};

std::ostream &operator<<(std::ostream &os, const TestParam &test_arg) {
  return os << "TestParam { width:" << test_arg.Width()
            << " height:" << test_arg.Height()
            << " ref_padding:" << test_arg.RefStridePadding()
            << " dst_padding:" << test_arg.DstStridePadding()
            << " border:" << test_arg.Border() << " bd:" << test_arg.BitDepth()
            << " is_hbd:" << test_arg.IsHighBitDepth()
            << " top_available:" << test_arg.TopAvailable()
            << " right_unavailable:" << test_arg.RightUnavailable()
            << " left_available:" << test_arg.LeftAvailable()
            << " bottom_unavailable:" << test_arg.BottomUnavailable() << " }";
}

constexpr int NUM_SAMPLED = 200;

// Random sample testing, for faster test suite execution.
std::unordered_set<TestParam, TestParamHash> GetSampledParams() {
  libaom_test::ACMRandom rnd;
  rnd.Reset(libaom_test::ACMRandom::DeterministicSeed());
  std::unordered_set<TestParam, TestParamHash> params;
  while (params.size() < NUM_SAMPLED) {
    if (params.size() % 10000 == 0) {
      std::cout << params.size() << std::endl;
    }
    const int block = rnd.Rand8() % BLOCK_SIZES_ALL;
    const int w = block_size_wide[block];
    const int h = block_size_high[block];
    const int dst_padding = 16 * (rnd.Rand8() % 3);
    const int ref_padding = 16 * (rnd.Rand8() % 3);
    const int border = 4 * (1 + rnd.Rand8() % 4);
    const int top_available = rnd.Rand8() % (border + 4);
    const int right_unavailable = rnd.Rand8() % w;
    const int left_available = rnd.Rand8() % (border + 4);
    const int bottom_unavailable = rnd.Rand8() % h;
    const bool is_hbd = rnd.Rand8() % 2;
    if (!is_hbd) {
      params.insert(TestParam(w, h, dst_padding, ref_padding, border,
                              AOM_BITS_8, false, top_available,
                              right_unavailable, left_available,
                              bottom_unavailable));
      continue;
    }
    // 0 == 8-bit, 1 == 10-bit, 2 == 12-bit.
    const int depth = rnd.Rand8() % 3;
    const aom_bit_depth_t bd =
        (depth == 0) ? AOM_BITS_8 : (depth == 1 ? AOM_BITS_10 : AOM_BITS_12);
    params.insert(TestParam(w, h, dst_padding, ref_padding, border, bd, true,
                            top_available, right_unavailable, left_available,
                            bottom_unavailable));
  }
  return params;
}

class IntrapredExtensionTest : public ::testing::TestWithParam<TestParam> {
 public:
  virtual ~IntrapredExtensionTest() {}

  virtual void SetUp() override {
    rnd_.Reset(libaom_test::ACMRandom::DeterministicSeed());
    const TestParam &p = GetParam();
    const int bytes = p.IsHighBitDepth() ? sizeof(uint16_t) : sizeof(uint8_t);
    int height = p.Height() + p.Border();
    const int ref_size = height * RefStride() * bytes;
    ref_ = reinterpret_cast<uint8_t *>(aom_memalign(16, ref_size));
    const int dst_size = height * DstStride() * bytes;
    dst_ = reinterpret_cast<uint8_t *>(aom_memalign(16, dst_size));
    Randomize();
  }

  virtual void TearDown() override {
    aom_free(ref_);
    aom_free(dst_);
    libaom_test::ClearSystemState();
  }

  // Return a pointer to the start of the reference. The border region is
  // negative offset.
  const uint8_t *Ref() const {
    const int offset = RefStride() * GetParam().Border() + GetParam().Border();
    if (GetParam().IsHighBitDepth()) {
      return CONVERT_TO_BYTEPTR(ref_) + offset;
    }
    return ref_ + offset;
  }

  // Return a pointer to the start of the destination buffer. The border region
  // is negative offset.
  uint8_t *Dst() const {
    const int offset = DstStride() * GetParam().Border() + GetParam().Border();
    if (GetParam().IsHighBitDepth()) {
      return CONVERT_TO_BYTEPTR(dst_) + offset;
    }
    return dst_ + offset;
  }

  int RefStride() const {
    const TestParam &p = GetParam();
    return p.Width() + p.Border() + p.RefStridePadding();
  }

  int DstStride() const {
    const TestParam &p = GetParam();
    return p.Width() + p.Border() + p.DstStridePadding();
  }

  int Base() const {
    switch (GetParam().BitDepth()) {
      case AOM_BITS_8: return 128;
      case AOM_BITS_10: return 512;
      case AOM_BITS_12: return 2048;
      default: EXPECT_TRUE(false); return 0;
    }
  }

  void ExtendBorder() {
    const TestParam &param = GetParam();
    av1_extend_intra_border(Ref(), RefStride(), Dst(), DstStride(),
                            param.TopAvailable(), param.RightUnavailable(),
                            param.LeftAvailable(), param.BottomUnavailable(),
                            param.Width(), param.Height(), param.Border(),
                            param.BitDepth(), param.IsHighBitDepth());
  }

  void Randomize() {
    const TestParam &p = GetParam();
    int height = p.Height() + p.Border();
    const int ref_size = height * RefStride();
    const int dst_size = height * DstStride();
    if (!p.IsHighBitDepth()) {
      for (int i = 0; i < ref_size; ++i) {
        ref_[i] = rnd_.Rand8();
      }
      for (int i = 0; i < dst_size; ++i) {
        dst_[i] = rnd_.Rand8();
      }
      return;
    }
    uint16_t *ref16 = reinterpret_cast<uint16_t *>(ref_);
    for (int i = 0; i < ref_size; ++i) {
      ref16[i] = rnd_.Rand16() & ((1 << p.BitDepth()) - 1);
    }
    uint16_t *dst16 = reinterpret_cast<uint16_t *>(dst_);
    for (int i = 0; i < dst_size; ++i) {
      dst16[i] = rnd_.Rand16() & ((1 << p.BitDepth()) - 1);
    }
  }

  // The two buffers should be equal for the given width/height.
  void ExpectBufEq(const uint8_t *buf1, int buf1_stride, const uint8_t *buf2,
                   int buf2_stride, int width, int height) const {
    EXPECT_GE(width, 0);
    EXPECT_GE(height, 0);
    if (GetParam().IsHighBitDepth()) {
      const uint16_t *buf16_1 = CONVERT_TO_SHORTPTR(buf1);
      const uint16_t *buf16_2 = CONVERT_TO_SHORTPTR(buf2);
      for (int j = 0; j < height; ++j) {
        EXPECT_EQ(0,
                  memcmp(buf16_1 + j * buf1_stride, buf16_2 + j * buf2_stride,
                         width * sizeof(uint16_t)));
      }
    } else {
      for (int j = 0; j < height; ++j) {
        EXPECT_EQ(
            0, memcmp(buf1 + j * buf1_stride, buf2 + j * buf2_stride, width));
      }
    }
  }

  // All values in thein the rectangle of width/height should
  // equal the value.
  void ExpectBufEqVal(int val, const uint8_t *buf, int buf_stride, int width,
                      int height) const {
    EXPECT_GE(width, 0);
    EXPECT_GE(height, 0);
    for (int j = 0; j < height; ++j) {
      for (int i = 0; i < width; ++i) {
        EXPECT_EQ(val, GetValue(buf + buf_stride * j + i));
      }
    }
  }

  int GetValue(const uint8_t *ptr) const {
    if (GetParam().IsHighBitDepth()) {
      return *CONVERT_TO_SHORTPTR(ptr);
    } else {
      return *ptr;
    }
  }

 private:
  uint8_t *ref_;
  uint8_t *dst_;
  libaom_test::ACMRandom rnd_;
};

TEST_P(IntrapredExtensionTest, MainBlockUntouched) {
  const TestParam &p = GetParam();
  // Allocate enough for either 8-bit or 16-bit pipeline.
  std::unique_ptr<uint8_t[]> copy(
      new uint8_t[p.Width() * p.Height() * sizeof(uint16_t)]);
  for (int j = 0; j < p.Height(); ++j) {
    if (p.IsHighBitDepth()) {
      memmove(copy.get() + j * p.Width() * sizeof(uint16_t),
              CONVERT_TO_SHORTPTR(Dst()) + j * DstStride(),
              p.Width() * sizeof(uint16_t));
    } else {
      memmove(copy.get() + j * p.Width(), Dst() + j * DstStride(), p.Width());
    }
  }
  ExtendBorder();
  if (p.IsHighBitDepth()) {
    ExpectBufEq(Dst(), DstStride(), CONVERT_TO_BYTEPTR(copy.get()), p.Width(),
                p.Width(), p.Height());
  } else {
    ExpectBufEq(Dst(), DstStride(), copy.get(), p.Width(), p.Width(),
                p.Height());
  }
}

TEST_P(IntrapredExtensionTest, TopRows) {
  const TestParam &p = GetParam();
  ExtendBorder();
  // Special case that no row is available. Base value is used.
  if (p.TopAvailable() == 0) {
    ExpectBufEqVal(Base() - 1, Dst() - p.Border() * DstStride(), DstStride(),
                   p.Width(), p.Border());
    return;
  }

  // The rows that could be copied, should be copied.
  const int valid_height = std::min(p.Border(), p.TopAvailable());
  const int valid_width = p.Width() - p.RightUnavailable();
  ExpectBufEq(Dst() - valid_height * DstStride(), DstStride(),
              Ref() - valid_height * RefStride(), RefStride(), valid_width,
              valid_height);

  // The rows should be extended to the right, if data is missing.
  for (int j = 1; j <= valid_height; ++j) {
    int val = GetValue(Dst() - j * DstStride() + valid_width - 1);
    ExpectBufEqVal(val, Dst() - j * DstStride() + valid_width, DstStride(),
                   p.Width() - valid_width, 1);
  }

  // Missing rows should be copied from the last valid row.
  for (int j = valid_height + 1; j <= p.Border(); ++j) {
    ExpectBufEq(Dst() - valid_height * DstStride(), DstStride(),
                Dst() - j * DstStride(), DstStride(), p.Width(), 1);
  }
}

TEST_P(IntrapredExtensionTest, LeftCols) {
  const TestParam &p = GetParam();
  ExtendBorder();
  // Special case that no col is available. Base value is used.
  if (p.LeftAvailable() == 0) {
    ExpectBufEqVal(Base() + 1, Dst() - p.Border(), DstStride(), p.Border(),
                   p.Height());
    return;
  }

  // The columns that could be copied, should be copied.
  const int valid_height = p.Height() - p.BottomUnavailable();
  const int valid_width = std::min(p.Border(), p.LeftAvailable());
  ExpectBufEq(Dst() - valid_width, DstStride(), Ref() - valid_width,
              RefStride(), valid_width, valid_height);

  // The columns should be extended down, if data is missing.
  for (int j = valid_height; j < p.Height(); ++j) {
    for (int i = 1; i <= valid_width; ++i) {
      int last_good = GetValue(Dst() + (valid_height - 1) * DstStride() - i);
      EXPECT_EQ(last_good, GetValue(Dst() + j * DstStride() - i));
    }
  }

  // Missing columns should be copied from the last valid column.
  for (int j = 0; j < p.Height(); ++j) {
    int last_good = GetValue(Dst() + j * DstStride() - valid_width);
    ExpectBufEqVal(last_good, Dst() + j * DstStride() - p.Border(), DstStride(),
                   p.Border() - valid_width, 1);
  }
}

TEST_P(IntrapredExtensionTest, TopLeftCorner) {
  const TestParam &p = GetParam();
  ExtendBorder();
  // Special case -- no data available.
  if (p.LeftAvailable() == 0 || p.TopAvailable() == 0) {
    ExpectBufEqVal(Base() - 1, Dst() - p.Border() * DstStride() - p.Border(),
                   DstStride(), p.Border(), p.Border());
    return;
  }
  // Any data that can be copied, should be copied.
  int viable_corner =
      std::min(p.TopAvailable(), std::min(p.Border(), p.LeftAvailable()));
  ExpectBufEq(Dst() - viable_corner * DstStride() - viable_corner, DstStride(),
              Ref() - viable_corner * RefStride() - viable_corner, RefStride(),
              viable_corner, viable_corner);

  // Any data missing on the left side should be extended from existing data.
  for (int j = 1; j <= viable_corner; ++j) {
    int last_good = GetValue(Dst() - j * DstStride() - viable_corner);
    ExpectBufEqVal(last_good, Dst() - j * DstStride() - p.Border(), DstStride(),
                   p.Border() - viable_corner, 1);
  }

  // Any rows missing should be copied from the last good row.
  uint8_t *last_good_row = Dst() - viable_corner * DstStride() - p.Border();
  for (int j = viable_corner + 1; j <= p.Border(); ++j) {
    ExpectBufEq(last_good_row, DstStride(),
                Dst() - j * DstStride() - p.Border(), DstStride(), p.Border(),
                1);
  }
}

INSTANTIATE_TEST_CASE_P(IntrapredExtensionTests, IntrapredExtensionTest,
                        ::testing::ValuesIn(GetSampledParams()));

}  // namespace
