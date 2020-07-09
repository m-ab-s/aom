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

#include <stdbool.h>
#include <memory>
#include "av1/common/reconintra.h"
#include "test/acm_random.h"
#include "test/clear_system_state.h"
#include "third_party/googletest/src/googletest/include/gtest/gtest.h"

namespace {

class TestParam {
 public:
  TestParam(int width, int height, int dst_stride_padding,
            int ref_stride_padding, int border, aom_bit_depth_t bd)
      : width_(width), height_(height), dst_stride_padding_(dst_stride_padding),
        ref_stride_padding_(ref_stride_padding), border_(border), bd_(bd) {}

  int Width() const { return width_; }
  int Height() const { return height_; }
  int DstStridePadding() const { return dst_stride_padding_; }
  int RefStridePadding() const { return ref_stride_padding_; }
  int Border() const { return border_; }
  aom_bit_depth_t BitDepth() const { return bd_; }

  bool operator==(const TestParam &other) const {
    return Width() == other.Width() && Height() == other.Height() &&
           DstStridePadding() == other.DstStridePadding() &&
           RefStridePadding() == other.RefStridePadding() &&
           Border() == other.Border() && BitDepth() == other.BitDepth();
  }

 private:
  int width_;
  int height_;
  int dst_stride_padding_;
  int ref_stride_padding_;
  int border_;
  aom_bit_depth_t bd_;
};

// For exhaustive testing.
std::vector<TestParam> GetTestParams() {
  std::vector<TestParam> params;
  for (int b = BLOCK_4X4; b < BLOCK_SIZES_ALL; ++b) {
    const int w = block_size_wide[b];
    const int h = block_size_high[b];
    for (int dst_padding = 0; dst_padding < 48; dst_padding += 16) {
      for (int ref_padding = 0; ref_padding < 48; ref_padding += 16) {
        for (int border = 4; border < 16; border += 4) {
          params.push_back(
              TestParam(w, h, dst_padding, ref_padding, border, AOM_BITS_8));
          params.push_back(
              TestParam(w, h, dst_padding, ref_padding, border, AOM_BITS_10));
          params.push_back(
              TestParam(w, h, dst_padding, ref_padding, border, AOM_BITS_12));
        }
      }
    }
  }
  return params;
}

// Random sample testing, for faster test suite execution.
::testing::internal::ParamGenerator<TestParam> GetSampledParams() {
  libaom_test::ACMRandom rnd;
  rnd.Reset(libaom_test::ACMRandom::DeterministicSeed());
  std::vector<TestParam> params = GetTestParams();
  std::vector<TestParam> sampled;
  // Roughly 5% sampling. + 1 in case params.size() / 20 == 0.
  const int num_samples = 1 + params.size() / 20;
  for (int i = 0; i < num_samples; ++i) {
    int r = rnd.Rand31() % params.size();
    TestParam last = params.back();
    TestParam curr = params[r];
    sampled.push_back(curr);
    params[r] = last;
    params.pop_back();
  }
  return ::testing::ValuesIn(sampled);
}

std::ostream &operator<<(std::ostream &os, const TestParam &test_arg) {
  return os << "TestParam { width:" << test_arg.Width()
            << " height:" << test_arg.Height()
            << " ref_padding:" << test_arg.RefStridePadding()
            << " dst_padding:" << test_arg.DstStridePadding()
            << " border:" << test_arg.Border() << " bd:" << test_arg.BitDepth()
            << " }";
}

class IntrapredExtensionTest : public ::testing::TestWithParam<TestParam> {
 public:
  virtual ~IntrapredExtensionTest() {}

  virtual void SetUp() override {
    rnd_.Reset(libaom_test::ACMRandom::DeterministicSeed());
    const TestParam &p = GetParam();
    int height = p.Height() + p.Border();
    int width = p.Width() + p.Border();
    int bytes = p.BitDepth() == AOM_BITS_8 ? sizeof(uint8_t) : sizeof(uint16_t);
    const int ref_size = height * (width + p.RefStridePadding()) * bytes;
    ref_ = reinterpret_cast<uint8_t *>(aom_memalign(16, ref_size));
    const int dst_size = height * (width + p.DstStridePadding()) * bytes;
    dst_ = reinterpret_cast<uint8_t *>(aom_memalign(16, dst_size));
    Randomize();
  }

  virtual void TearDown() override {
    aom_free(ref_);
    aom_free(dst_);
    libaom_test::ClearSystemState();
  }

  const uint8_t *Ref() const {
    const int offset = RefStride() * GetParam().Border() + GetParam().Border();
    if (GetParam().BitDepth() != AOM_BITS_8) {
      return CONVERT_TO_BYTEPTR(ref_) + offset;
    }
    return ref_ + offset;
  }

  uint8_t *Dst() const {
    const int offset = DstStride() * GetParam().Border() + GetParam().Border();
    if (GetParam().BitDepth() != AOM_BITS_8) {
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

  // Check that the destination has this many rows copied directly from the
  // reference buffer. Note that this only applies to values *directly* above
  // the block.
  void CheckTopCopied(int num_rows) {
    const uint8_t *ref = Ref();
    const uint8_t *dst = Dst();
    ref -= RefStride() * num_rows;
    dst -= DstStride() * num_rows;
    for (int i = 0; i < num_rows; ++i) {
      const uint8_t *ref_row = ref + i * RefStride();
      const uint8_t *dst_row = dst + i * DstStride();
      int bytes = sizeof(uint8_t);
      if (GetParam().BitDepth() != AOM_BITS_8) {
        ref_row = reinterpret_cast<uint8_t *>(CONVERT_TO_SHORTPTR(ref_row));
        dst_row = reinterpret_cast<uint8_t *>(CONVERT_TO_SHORTPTR(dst_row));
        bytes = sizeof(uint16_t);
      }
      EXPECT_EQ(0, memcmp(ref_row, dst_row, GetParam().Width() * bytes));
    }
  }

  // Check that the destination replicates upward.
  void CheckTopExtended(int num_rows) {
    const int border = GetParam().Border();
    const uint8_t *ref = Ref();
    const uint8_t *dst = Dst();
    ref -= RefStride() * num_rows;
    dst -= DstStride() * border;
    for (int i = 0; i < border - num_rows; ++i) {
      const uint8_t *ref_row = ref;
      const uint8_t *dst_row = dst + i * DstStride();
      int bytes = sizeof(uint8_t);
      if (GetParam().BitDepth() != AOM_BITS_8) {
        ref_row = reinterpret_cast<uint8_t *>(CONVERT_TO_SHORTPTR(ref_row));
        dst_row = reinterpret_cast<uint8_t *>(CONVERT_TO_SHORTPTR(dst_row));
        bytes = sizeof(uint16_t);
      }
      EXPECT_EQ(0, memcmp(ref_row, dst_row, GetParam().Width() * bytes));
    }
  }

  // Check that the columns replicate leftward.
  void CheckLeftExtended(int num_cols) {
    const int border = GetParam().Border();
    const uint8_t *ref = Ref();
    const uint8_t *dst = Dst();
    ref -= num_cols;
    dst -= border;
    for (int j = 0; j < GetParam().Height(); ++j) {
      for (int i = 0; i < border - num_cols; ++i) {
        const uint8_t *ref_row = ref + j * RefStride();
        const uint8_t *dst_row = dst + j * DstStride() + i;
        if (GetParam().BitDepth() == AOM_BITS_8) {
          EXPECT_EQ(ref_row[0], dst_row[0]);
        } else {
          EXPECT_EQ(CONVERT_TO_SHORTPTR(ref_row)[0],
                    CONVERT_TO_SHORTPTR(dst_row)[0]);
        }
      }
    }
  }

  // Check that the destination has this may columns copied directly from the
  // reference buffer. Note that this only applies to values *directly* to
  // the left of the block.
  void CheckLeftCopied(int num_cols) {
    const uint8_t *ref = Ref();
    const uint8_t *dst = Dst();
    ref -= num_cols;
    dst -= num_cols;
    for (int i = 0; i < GetParam().Height(); ++i) {
      const uint8_t *ref_row = ref + i * RefStride();
      const uint8_t *dst_row = dst + i * DstStride();
      int bytes = sizeof(uint8_t);
      if (GetParam().BitDepth() != AOM_BITS_8) {
        ref_row = reinterpret_cast<uint8_t *>(CONVERT_TO_SHORTPTR(ref_row));
        dst_row = reinterpret_cast<uint8_t *>(CONVERT_TO_SHORTPTR(dst_row));
        bytes = sizeof(uint16_t);
      }
      EXPECT_EQ(0, memcmp(ref_row, dst_row, num_cols * bytes));
    }
  }

  // Check that the top-left corner has this many rows and columns copied
  // from the reference buffer.
  void CheckTopLeftCopied(int num_pix) {
    const uint8_t *ref = Ref();
    const uint8_t *dst = Dst();
    ref -= (num_pix + num_pix * RefStride());
    dst -= (num_pix + num_pix * DstStride());
    for (int i = 0; i < num_pix; ++i) {
      const uint8_t *ref_row = ref + i * RefStride();
      const uint8_t *dst_row = dst + i * DstStride();
      int bytes = sizeof(uint8_t);
      if (GetParam().BitDepth() != AOM_BITS_8) {
        ref_row = reinterpret_cast<uint8_t *>(CONVERT_TO_SHORTPTR(ref_row));
        dst_row = reinterpret_cast<uint8_t *>(CONVERT_TO_SHORTPTR(dst_row));
        bytes = sizeof(uint16_t);
      }
      EXPECT_EQ(0, memcmp(ref_row, dst_row, num_pix * bytes));
    }
  }

  // Check that the entire top-left corner is filled with base-values.
  void CheckTopLeftBaseEq(int base_val) const {
    const int border = GetParam().Border();
    uint8_t *dst = Dst();
    dst -= DstStride() * border;
    dst -= border;
    for (int j = 0; j < border; ++j) {
      for (int i = 0; i < border; ++i) {
        if (GetParam().BitDepth() != AOM_BITS_8) {
          EXPECT_EQ(base_val, CONVERT_TO_SHORTPTR(dst)[i]);
        } else {
          EXPECT_EQ(base_val, dst[i]);
        }
      }
      dst += DstStride();
    }
  }

  // Check that the entire top part of the border is filled with base-values.
  void CheckTopBaseEq(int base_val) const {
    const int border = GetParam().Border();
    uint8_t *dst = Dst();
    dst -= DstStride() * border;
    for (int j = 0; j < border; ++j) {
      for (int i = 0; i < GetParam().Width(); ++i) {
        if (GetParam().BitDepth() != AOM_BITS_8) {
          EXPECT_EQ(base_val, CONVERT_TO_SHORTPTR(dst)[i]);
        } else {
          EXPECT_EQ(base_val, dst[i]);
        }
      }
      dst += DstStride();
    }
  }

  // Check that all left columns are filled with base-values.
  void CheckLeftBaseEq(int base_val) const {
    const int border = GetParam().Border();
    uint8_t *dst = Dst();
    dst -= border;
    for (int j = 0; j < GetParam().Height(); ++j) {
      for (int i = 0; i < border; ++i) {
        if (GetParam().BitDepth() != AOM_BITS_8) {
          EXPECT_EQ(base_val, CONVERT_TO_SHORTPTR(dst)[i]);
        } else {
          EXPECT_EQ(base_val, dst[i]);
        }
      }
      dst += DstStride();
    }
  }

  // Checks that the top num_pix rows/cols are extended in the
  // top-left region.
  void CheckTopLeftExtended(int num_pix) {
    const int border = GetParam().Border();
    const uint8_t *ref = Ref();
    const uint8_t *dst = Dst();
    ref -= RefStride() * num_pix + num_pix;
    dst -= DstStride() * num_pix + border;
    // Check left-replicated.
    for (int j = 0; j < num_pix; ++j) {
      int last_val;
      if (GetParam().BitDepth() == AOM_BITS_8) {
        last_val = ref[0];
      } else {
        last_val = CONVERT_TO_SHORTPTR(ref)[0];
      }

      for (int i = 0; i < border - num_pix; ++i) {
        if (GetParam().BitDepth() == AOM_BITS_8) {
          EXPECT_EQ(last_val, dst[i]);
        } else {
          EXPECT_EQ(last_val, CONVERT_TO_SHORTPTR(dst)[i]);
        }
      }
      ref += RefStride();
      dst += DstStride();
    }
    // Check top-replicated.
    const uint8_t *dst_ref = Dst() - DstStride() * num_pix - border;
    dst = Dst() - DstStride() * border - border;
    for (int j = 0; j < border - num_pix; ++j) {
      const uint8_t *ref_row = dst_ref;
      const uint8_t *dst_row = dst + j * DstStride();
      int bytes = sizeof(uint8_t);
      if (GetParam().BitDepth() != AOM_BITS_8) {
        ref_row = reinterpret_cast<uint8_t *>(CONVERT_TO_SHORTPTR(ref_row));
        dst_row = reinterpret_cast<uint8_t *>(CONVERT_TO_SHORTPTR(dst_row));
        bytes = sizeof(uint16_t);
      }
      EXPECT_EQ(0, memcmp(ref_row, dst_row, border * bytes));
    }
  }

  int Base() const {
    switch (GetParam().BitDepth()) {
      case AOM_BITS_8: return 128;
      case AOM_BITS_10: return 512;
      case AOM_BITS_12: return 2048;
      default: EXPECT_TRUE(false); return 0;
    }
  }

  void Randomize() {
    const TestParam &p = GetParam();
    int height = p.Height() + p.Border();
    int width = p.Width() + p.Border();
    int bytes = p.BitDepth() == AOM_BITS_8 ? sizeof(uint8_t) : sizeof(uint16_t);

    const int ref_size = height * (width + p.RefStridePadding()) * bytes;
    for (int i = 0; i < ref_size; ++i) {
      ref_[i] = rnd_.Rand8();
    }

    const int dst_size = height * (width + p.DstStridePadding()) * bytes;
    for (int i = 0; i < dst_size; ++i) {
      dst_[i] = rnd_.Rand8();
    }
  }

 private:
  uint8_t *ref_;
  uint8_t *dst_;
  libaom_test::ACMRandom rnd_;
};

// Each of the left/top areas can either be: completely void, partially
// there, or fully there. Test all 3x3 = 9 combinations.
TEST_P(IntrapredExtensionTest, NoLeftNoTop) {
  const uint8_t *ref = Ref();
  uint8_t *dst = Dst();
  int border = GetParam().Border();
  av1_extend_intra_border(ref, RefStride(), dst, DstStride(), 0, 0,
                          GetParam().Width(), GetParam().Height(), border,
                          GetParam().BitDepth());
  CheckTopLeftBaseEq(Base() - 1);
  CheckTopBaseEq(Base() - 1);
  CheckLeftBaseEq(Base() + 1);
}

TEST_P(IntrapredExtensionTest, NoLeftPartialTop) {
  const int border = GetParam().Border();
  for (int rows = 1; rows < border; ++rows) {
    Randomize();
    const uint8_t *ref = Ref();
    uint8_t *dst = Dst();
    av1_extend_intra_border(ref, RefStride(), dst, DstStride(), rows, 0,
                            GetParam().Width(), GetParam().Height(), border,
                            GetParam().BitDepth());
    CheckTopCopied(rows);
    CheckTopExtended(rows);
    CheckLeftBaseEq(Base() + 1);
    CheckTopLeftBaseEq(Base() - 1);
  }
}

TEST_P(IntrapredExtensionTest, NoLeftSufficientTop) {
  const int border = GetParam().Border();
  for (int extra_rows = 0; extra_rows < 3; ++extra_rows) {
    Randomize();
    const uint8_t *ref = Ref();
    uint8_t *dst = Dst();
    av1_extend_intra_border(ref, RefStride(), dst, DstStride(),
                            border + extra_rows, 0, GetParam().Width(),
                            GetParam().Height(), border, GetParam().BitDepth());
    CheckTopCopied(border);
    CheckLeftBaseEq(Base() + 1);
    CheckTopLeftBaseEq(Base() - 1);
  }
}

TEST_P(IntrapredExtensionTest, PartialLeftNoTop) {
  const int border = GetParam().Border();
  for (int cols = 1; cols < border; ++cols) {
    Randomize();
    const uint8_t *ref = Ref();
    uint8_t *dst = Dst();
    av1_extend_intra_border(ref, RefStride(), dst, DstStride(), 0, cols,
                            GetParam().Width(), GetParam().Height(), border,
                            GetParam().BitDepth());
    CheckTopBaseEq(Base() - 1);
    CheckLeftCopied(cols);
    CheckLeftExtended(cols);
    CheckTopLeftBaseEq(Base() - 1);
  }
}

TEST_P(IntrapredExtensionTest, PartialLeftPartialTop) {
  const int border = GetParam().Border();
  for (int rows = 1; rows < border; ++rows) {
    for (int cols = 1; cols < border; ++cols) {
      Randomize();
      const uint8_t *ref = Ref();
      uint8_t *dst = Dst();

      av1_extend_intra_border(ref, RefStride(), dst, DstStride(), rows, cols,
                              GetParam().Width(), GetParam().Height(), border,
                              GetParam().BitDepth());

      CheckTopCopied(rows);
      CheckTopExtended(rows);
      CheckLeftCopied(cols);
      CheckLeftExtended(cols);
      CheckTopLeftCopied(AOMMIN(rows, cols));
      CheckTopLeftExtended(AOMMIN(rows, cols));
    }
  }
}

TEST_P(IntrapredExtensionTest, PartialLeftSufficientTop) {
  const int border = GetParam().Border();
  for (int extra_rows = 0; extra_rows < 3; ++extra_rows) {
    for (int cols = 1; cols < border; ++cols) {
      Randomize();
      const uint8_t *ref = Ref();
      uint8_t *dst = Dst();

      av1_extend_intra_border(ref, RefStride(), dst, DstStride(),
                              border + extra_rows, cols, GetParam().Width(),
                              GetParam().Height(), border,
                              GetParam().BitDepth());
      CheckTopCopied(border);
      CheckLeftCopied(cols);
      CheckLeftExtended(cols);
      CheckTopLeftCopied(cols);
      CheckTopLeftExtended(cols);
    }
  }
}

TEST_P(IntrapredExtensionTest, SufficientLeftNoTop) {
  const int border = GetParam().Border();
  for (int extra_cols = 0; extra_cols < 3; ++extra_cols) {
    Randomize();
    const uint8_t *ref = Ref();
    uint8_t *dst = Dst();
    av1_extend_intra_border(ref, RefStride(), dst, DstStride(), 0,
                            border + extra_cols, GetParam().Width(),
                            GetParam().Height(), border, GetParam().BitDepth());
    CheckTopBaseEq(Base() - 1);
    CheckLeftCopied(border);
    CheckTopLeftBaseEq(Base() - 1);
  }
}

TEST_P(IntrapredExtensionTest, SufficientLeftPartialTop) {
  const int border = GetParam().Border();
  for (int extra_cols = 0; extra_cols < 3; ++extra_cols) {
    for (int rows = 1; rows < border; ++rows) {
      Randomize();
      const uint8_t *ref = Ref();
      uint8_t *dst = Dst();

      av1_extend_intra_border(ref, RefStride(), dst, DstStride(), rows,
                              border + extra_cols, GetParam().Width(),
                              GetParam().Height(), border,
                              GetParam().BitDepth());
      CheckTopCopied(rows);
      CheckTopExtended(rows);
      CheckLeftCopied(border);
      CheckTopLeftCopied(rows);
      CheckTopLeftExtended(rows);
    }
  }
}

TEST_P(IntrapredExtensionTest, SufficientTopLeft) {
  const uint8_t *ref = Ref();
  uint8_t *dst = Dst();
  int border = GetParam().Border();

  // Should work even if the number of available cols and rows is
  // greater than the border size; these extra cols/rows should never
  // be referenced.
  for (int extra_rows = 0; extra_rows < 3; ++extra_rows) {
    for (int extra_cols = 0; extra_cols < 3; ++extra_cols) {
      Randomize();
      av1_extend_intra_border(ref, RefStride(), dst, DstStride(),
                              border + extra_rows, border + extra_cols,
                              GetParam().Width(), GetParam().Height(), border,
                              GetParam().BitDepth());
      CheckTopCopied(border);
      CheckLeftCopied(border);
      CheckTopLeftCopied(border);
    }
  }
}

INSTANTIATE_TEST_CASE_P(IntrapredExtensionTests, IntrapredExtensionTest,
                        GetSampledParams());

}  // namespace
