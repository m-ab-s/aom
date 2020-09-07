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

#include "third_party/googletest/src/googletest/include/gtest/gtest.h"

#include "av1/common/mfqe.h"

#define MFQE_TEST_STRIDE 80
#define MFQE_TEST_HEIGHT 32
#define MFQE_TEST_WIDTH 32

namespace {

class MFQETest : public ::testing::Test {
 protected:
  // Sets up current frame and reference frames as 0-filled data.
  void SetUp() override {
    tmp_.stride = MFQE_TEST_STRIDE;
    tmp_.height = MFQE_TEST_HEIGHT;
    tmp_.width = MFQE_TEST_WIDTH;

    int buf_bytes = (tmp_.height + 2 * MFQE_PADDING_SIZE) * tmp_.stride;
    tmp_.buffer_orig = reinterpret_cast<uint8_t *>(
        aom_memalign(32, sizeof(uint8_t) * buf_bytes));
    tmp_.buffer = tmp_.buffer_orig + MFQE_PADDING_SIZE * tmp_.stride;
    memset(tmp_.buffer_orig, 0, sizeof(uint8_t) * buf_bytes);

    ref_frames_ = new RefCntBuffer *[MFQE_NUM_REFS];

    for (int i = 0; i < MFQE_NUM_REFS; i++) {
      ref_frames_[i] = new RefCntBuffer;
      ref_frames_[i]->buf.y_width = MFQE_TEST_WIDTH;
      ref_frames_[i]->buf.y_height = MFQE_TEST_HEIGHT;
      ref_frames_[i]->buf.y_stride = MFQE_TEST_STRIDE;
      buf_bytes = (MFQE_TEST_HEIGHT + 2 * MFQE_PADDING_SIZE) * MFQE_TEST_STRIDE;
      uint8_t *buffer = reinterpret_cast<uint8_t *>(
          aom_memalign(32, sizeof(uint8_t) * buf_bytes));
      ref_frames_[i]->buf.y_buffer =
          buffer + MFQE_PADDING_SIZE * MFQE_TEST_STRIDE;
      memset(buffer, 0, sizeof(uint8_t) * buf_bytes);
    }

    // Post-condition: verify that current frame is 0-filled.
    for (int i = 0; i < tmp_.stride * tmp_.height; i++) {
      ASSERT_EQ(0, tmp_.buffer[i]);
    }
    // Verify that reference frames are 0-filled.
    for (int i = 0; i < MFQE_NUM_REFS; i++) {
      uint8_t *ref_y_buffer = ref_frames_[i]->buf.y_buffer;
      for (int j = 0; j < tmp_.stride * tmp_.height; j++) {
        ASSERT_EQ(0, ref_y_buffer[i]);
      }
    }
  }

  void TearDown() override {
    aom_free(tmp_.buffer_orig);
    for (int i = 0; i < MFQE_NUM_REFS; i++) {
      uint8_t *buffer =
          ref_frames_[i]->buf.y_buffer - MFQE_PADDING_SIZE * MFQE_TEST_STRIDE;
      aom_free(buffer);
      delete ref_frames_[i];
    }
    delete[] ref_frames_;
  }

  Y_BUFFER_CONFIG tmp_;
  RefCntBuffer **ref_frames_;
};

}  // namespace

// Test that if MFQE is applied to all zeroes, the result is all 0.
TEST_F(MFQETest, TestLowBdAllZero) {
  const int buf_size = tmp_.stride * tmp_.height;
  const int high_bd = 0;
  const int bitdepth = 8;
  av1_apply_loop_mfqe(&tmp_, ref_frames_, MFQE_BLOCK_SIZE, MFQE_SCALE_SIZE,
                      high_bd, bitdepth);
  for (int i = 0; i < buf_size; i++) {
    ASSERT_EQ(0, tmp_.buffer[i]);
  }
}

// Test that if MFQE is applied to reference frames of all zeroes and a
// current frame of occassional 1's, the result has either 0 or 1 for
// each value.
TEST_F(MFQETest, TestLowBdSomeOnes) {
  const int buf_size = tmp_.stride * tmp_.height;
  const int high_bd = 0;
  const int bitdepth = 8;
  for (int i = 0; i < buf_size; i++) {
    if (i % 11 == 0) {
      tmp_.buffer[i] = 1;
    }
  }
  av1_apply_loop_mfqe(&tmp_, ref_frames_, MFQE_BLOCK_SIZE, MFQE_SCALE_SIZE,
                      high_bd, bitdepth);
  for (int i = 0; i < buf_size; i++) {
    ASSERT_LE(tmp_.buffer[i], 1);
  }
}

// If the current frame has values between 0 and 6, then after MFQE,
// the resulting values must be between 0 and 6 (since the reference
// frames are all zero).
TEST_F(MFQETest, TestLowBdZeroToSix) {
  const int buf_size = tmp_.stride * tmp_.height;
  const int high_bd = 0;
  const int bitdepth = 8;
  for (int i = 0; i < buf_size; i++) {
    tmp_.buffer[i] = (i % 7);
  }
  av1_apply_loop_mfqe(&tmp_, ref_frames_, MFQE_BLOCK_SIZE, MFQE_SCALE_SIZE,
                      high_bd, bitdepth);
  for (int i = 0; i < buf_size; i++) {
    ASSERT_LE(tmp_.buffer[i], 7);
  }
}

// If the current frame has values between 10 and 18, then after MFQE,
// the results values are greater than or equal to 10.
TEST_F(MFQETest, TestLowBdTenToEighteen) {
  const int buf_size = tmp_.stride * tmp_.height;
  const int high_bd = 0;
  const int bitdepth = 8;
  for (int i = 0; i < buf_size; i++) {
    tmp_.buffer[i] = (i % 9) + 10;
  }
  av1_apply_loop_mfqe(&tmp_, ref_frames_, MFQE_BLOCK_SIZE, MFQE_SCALE_SIZE,
                      high_bd, bitdepth);
  for (int i = 0; i < buf_size; i++) {
    ASSERT_GE(tmp_.buffer[i], 10);
  }
}

// If the current frame is set to a constant value, and the reference frames
// are all zero, after MFQE, the result is the same as the current frame.
TEST_F(MFQETest, TestLowBdFifty) {
  const int buf_size = tmp_.stride * tmp_.height;
  const int high_bd = 0;
  const int bitdepth = 8;
  for (int i = 0; i < buf_size; i++) {
    tmp_.buffer[i] = 50;
  }
  av1_apply_loop_mfqe(&tmp_, ref_frames_, MFQE_BLOCK_SIZE, MFQE_SCALE_SIZE,
                      high_bd, bitdepth);
  for (int i = 0; i < buf_size; i++) {
    ASSERT_EQ(50, tmp_.buffer[i]);
  }
}

TEST_F(MFQETest, TestLowBdHundred) {
  const int buf_size = tmp_.stride * tmp_.height;
  const int high_bd = 0;
  const int bitdepth = 8;
  for (int i = 0; i < buf_size; i++) {
    tmp_.buffer[i] = 100;
  }
  av1_apply_loop_mfqe(&tmp_, ref_frames_, MFQE_BLOCK_SIZE, MFQE_SCALE_SIZE,
                      high_bd, bitdepth);
  for (int i = 0; i < buf_size; i++) {
    ASSERT_EQ(100, tmp_.buffer[i]);
  }
}
