/*
 * Copyright (c) 2019, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <stdbool.h>
#include "test/util.h"
#include "av1/common/reconinter.h"
#include "third_party/googletest/src/googletest/include/gtest/gtest.h"

#if CONFIG_ILLUM_MCOMP

namespace {

// To instantiate a regular test.
class IllumMcompTest : public ::testing::Test {};

TEST_F(IllumMcompTest, ComputeDc) {
  uint8_t pred[256];
  // Averaging the same color, is always that color.
  for (int i = 0; i < 256; ++i) {
    for (int j = 0; j < 256; ++j) {
      pred[j] = i;
    }
    // Try different shapes/sizes/strides. They should all
    // produce the same value.
    ASSERT_EQ(i, illum_mcomp_compute_dc_lowbd(pred, 8, 8, 8));
    ASSERT_EQ(i, illum_mcomp_compute_dc_lowbd(pred, 1, 1, 256));
    ASSERT_EQ(i, illum_mcomp_compute_dc_lowbd(pred, 2, 1, 128));
    ASSERT_EQ(i, illum_mcomp_compute_dc_lowbd(pred, 2, 2, 128));
  }

  // Try half black, half white. Average should be grey.
  for (int j = 0; j < 256; ++j) {
    pred[j] = (j < 128) ? 0 : 255;
  }
  ASSERT_EQ(128, illum_mcomp_compute_dc_lowbd(pred, 16, 16, 16));
  ASSERT_EQ(128, illum_mcomp_compute_dc_lowbd(pred, 1, 1, 256));
  ASSERT_EQ(128, illum_mcomp_compute_dc_lowbd(pred, 2, 1, 128));
  ASSERT_EQ(128, illum_mcomp_compute_dc_lowbd(pred, 2, 2, 128));
}

TEST_F(IllumMcompTest, ComputeDcHigh) {
  uint16_t pred[256];
  for (int i = 0; i < 1024; ++i) {
    for (int j = 0; j < 256; ++j) {
      pred[j] = i;
    }
    // Try different shapes/sizes/strides. They should all
    // produce the same value.
    ASSERT_EQ(i, illum_mcomp_compute_dc_highbd(pred, 16, 16, 16));
    ASSERT_EQ(i, illum_mcomp_compute_dc_highbd(pred, 1, 1, 256));
    ASSERT_EQ(i, illum_mcomp_compute_dc_highbd(pred, 2, 1, 128));
    ASSERT_EQ(i, illum_mcomp_compute_dc_highbd(pred, 2, 2, 128));
  }

  // Try half black, half white. Average should be grey.
  for (int j = 0; j < 256; ++j) {
    pred[j] = (j < 128) ? 0 : 1023;
  }
  ASSERT_EQ(512, illum_mcomp_compute_dc_highbd(pred, 16, 16, 16));
  ASSERT_EQ(512, illum_mcomp_compute_dc_highbd(pred, 1, 1, 256));
  ASSERT_EQ(512, illum_mcomp_compute_dc_highbd(pred, 2, 1, 128));
  ASSERT_EQ(512, illum_mcomp_compute_dc_highbd(pred, 2, 2, 128));
}

}  // namespace

#endif  // CONFIG_ILLUM_MCOMP
