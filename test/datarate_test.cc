/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include "config/aom_config.h"

#include "third_party/googletest/src/googletest/include/gtest/gtest.h"
#include "test/codec_factory.h"
#include "test/datarate_test.h"
#include "test/encode_test_driver.h"
#include "test/i420_video_source.h"
#include "test/util.h"
#include "test/y4m_video_source.h"
#include "aom/aom_codec.h"

namespace datarate_test {
namespace {

// Params: test mode, speed, aq mode and index for bitrate array.
class DatarateTestLarge
    : public ::libaom_test::CodecTestWith4Params<libaom_test::TestMode, int,
                                                 unsigned int, int>,
      public DatarateTest {
 public:
  DatarateTestLarge() : DatarateTest(GET_PARAM(0)) {
    set_cpu_used_ = GET_PARAM(2);
    aq_mode_ = GET_PARAM(3);
  }

 protected:
  virtual ~DatarateTestLarge() {}

  virtual void SetUp() {
    InitializeConfig();
    SetMode(GET_PARAM(1));
    ResetModel();
  }

  virtual void BasicRateTargetingVBRTest() {
    cfg_.rc_min_quantizer = 0;
    cfg_.rc_max_quantizer = 255;
    cfg_.g_error_resilient = 0;
    cfg_.rc_end_usage = AOM_VBR;
    cfg_.g_lag_in_frames = 0;

    ::libaom_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352,
                                         288, 30, 1, 0, 140);
    const int bitrate_array[2] = { 400, 800 };
    cfg_.rc_target_bitrate = bitrate_array[GET_PARAM(4)];
    ResetModel();
    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
    ASSERT_GE(effective_datarate_, cfg_.rc_target_bitrate * 0.7)
        << " The datarate for the file is lower than target by too much!";
    ASSERT_LE(effective_datarate_, cfg_.rc_target_bitrate * 1.3)
        << " The datarate for the file is greater than target by too much!";
  }

  virtual void BasicRateTargetingCBRTest() {
    cfg_.rc_buf_initial_sz = 500;
    cfg_.rc_buf_optimal_sz = 500;
    cfg_.rc_buf_sz = 1000;
    cfg_.rc_dropframe_thresh = 1;
    cfg_.rc_min_quantizer = 0;
#if CONFIG_EXTQUANT
    cfg_.rc_max_quantizer = 208;
#else
    cfg_.rc_max_quantizer = 255;
#endif  // CONFIG_EXTQUANT
    cfg_.rc_end_usage = AOM_CBR;
    cfg_.g_lag_in_frames = 0;

    ::libaom_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352,
                                         288, 30, 1, 0, 140);
    const int bitrate_array[2] = { 150, 550 };
    cfg_.rc_target_bitrate = bitrate_array[GET_PARAM(4)];
    ResetModel();
    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
    ASSERT_GE(effective_datarate_, cfg_.rc_target_bitrate * 0.85)
        << " The datarate for the file is lower than target by too much!";
    ASSERT_LE(effective_datarate_, cfg_.rc_target_bitrate * 1.15)
        << " The datarate for the file is greater than target by too much!";
  }

  virtual void ErrorResilienceOnSceneCuts() {
    if (GET_PARAM(4) > 0) return;
    cfg_.rc_buf_initial_sz = 500;
    cfg_.rc_buf_optimal_sz = 500;
    cfg_.rc_buf_sz = 1000;
    cfg_.rc_dropframe_thresh = 0;
    cfg_.g_error_resilient = 1;
    cfg_.rc_min_quantizer = 0;
#if CONFIG_EXTQUANT
    cfg_.rc_max_quantizer = 208;
#else
    cfg_.rc_max_quantizer = 255;
#endif  // CONFIG_EXTQUANT
    cfg_.rc_end_usage = AOM_CBR;
    cfg_.g_lag_in_frames = 0;

    ::libaom_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352,
                                         288, 30, 1, 0, 300);
    cfg_.rc_target_bitrate = 500;
    ResetModel();
    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
    ASSERT_GE(effective_datarate_, cfg_.rc_target_bitrate * 0.85)
        << " The datarate for the file is lower than target by too much!";
    ASSERT_LE(effective_datarate_, cfg_.rc_target_bitrate * 1.15)
        << " The datarate for the file is greater than target by too much!";
  }

  virtual void BasicRateTargetingCBRPeriodicKeyFrameTest() {
    cfg_.rc_buf_initial_sz = 500;
    cfg_.rc_buf_optimal_sz = 500;
    cfg_.rc_buf_sz = 1000;
    cfg_.rc_dropframe_thresh = 1;
    cfg_.rc_min_quantizer = 0;
    cfg_.rc_max_quantizer = 255;
    cfg_.rc_end_usage = AOM_CBR;
    cfg_.g_lag_in_frames = 0;
    // Periodic keyframe
    cfg_.kf_max_dist = 50;

    ::libaom_test::I420VideoSource video("pixel_capture_w320h240.yuv", 320, 240,
                                         30, 1, 0, 310);
    const int bitrate_array[2] = { 150, 550 };
    cfg_.rc_target_bitrate = bitrate_array[GET_PARAM(4)];
    ResetModel();
    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
    ASSERT_GE(effective_datarate_, cfg_.rc_target_bitrate * 0.85)
        << " The datarate for the file is lower than target by too much!";
    ASSERT_LE(effective_datarate_, cfg_.rc_target_bitrate * 1.15)
        << " The datarate for the file is greater than target by too much!";
  }

  virtual void CBRPeriodicKeyFrameOnSceneCuts() {
    if (GET_PARAM(4) > 0) return;
    cfg_.rc_buf_initial_sz = 500;
    cfg_.rc_buf_optimal_sz = 500;
    cfg_.rc_buf_sz = 1000;
    cfg_.rc_dropframe_thresh = 0;
    cfg_.rc_min_quantizer = 0;
    cfg_.rc_max_quantizer = 255;
    cfg_.rc_end_usage = AOM_CBR;
    cfg_.g_lag_in_frames = 0;
    // Periodic keyframe
    cfg_.kf_max_dist = 30;
    cfg_.kf_min_dist = 30;

    ::libaom_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352,
                                         288, 30, 1, 0, 300);
    cfg_.rc_target_bitrate = 500;
    ResetModel();
    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
    ASSERT_GE(effective_datarate_, cfg_.rc_target_bitrate * 0.85)
        << " The datarate for the file is lower than target by too much!";
    ASSERT_LE(effective_datarate_, cfg_.rc_target_bitrate * 1.3)
        << " The datarate for the file is greater than target by too much!";
  }
};
// Params: test mode, speed, aq mode.
class DatarateTestFrameDropLarge
    : public ::libaom_test::CodecTestWith3Params<libaom_test::TestMode, int,
                                                 unsigned int>,
      public DatarateTest {
 public:
  DatarateTestFrameDropLarge() : DatarateTest(GET_PARAM(0)) {
    set_cpu_used_ = GET_PARAM(2);
    aq_mode_ = GET_PARAM(3);
  }

 protected:
  virtual ~DatarateTestFrameDropLarge() {}

  virtual void SetUp() {
    InitializeConfig();
    SetMode(GET_PARAM(1));
    ResetModel();
  }

  virtual void ChangingDropFrameThreshTest() {
    cfg_.rc_buf_initial_sz = 500;
    cfg_.rc_buf_optimal_sz = 500;
    cfg_.rc_buf_sz = 1000;
    cfg_.rc_undershoot_pct = 20;
    cfg_.rc_undershoot_pct = 20;
    cfg_.rc_dropframe_thresh = 10;
    cfg_.rc_min_quantizer = 0;
    cfg_.rc_max_quantizer = 200;
    cfg_.rc_end_usage = AOM_CBR;
    cfg_.rc_target_bitrate = 200;
    cfg_.g_lag_in_frames = 0;
    cfg_.g_error_resilient = 1;
    // TODO(marpan): Investigate datarate target failures with a smaller
    // keyframe interval (128).
    cfg_.kf_max_dist = 9999;

    ::libaom_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352,
                                         288, 30, 1, 0, 100);

    const int kDropFrameThreshTestStep = 30;
    aom_codec_pts_t last_drop = 140;
    int last_num_drops = 0;
    for (int i = 40; i < 100; i += kDropFrameThreshTestStep) {
      cfg_.rc_dropframe_thresh = i;
      ResetModel();
      ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
      ASSERT_GE(effective_datarate_, cfg_.rc_target_bitrate * 0.85)
          << " The datarate for the file is lower than target by too much!";
      ASSERT_LE(effective_datarate_, cfg_.rc_target_bitrate * 1.17)
          << " The datarate for the file is greater than target by too much!";
      if (last_drop > 0) {
        ASSERT_LE(first_drop_, last_drop)
            << " The first dropped frame for drop_thresh " << i
            << " > first dropped frame for drop_thresh "
            << i - kDropFrameThreshTestStep;
      }
      ASSERT_GE(num_drops_, last_num_drops * 0.7)
          << " The number of dropped frames for drop_thresh " << i
          << " < number of dropped frames for drop_thresh "
          << i - kDropFrameThreshTestStep;
      last_drop = first_drop_;
      last_num_drops = num_drops_;
    }
  }
};

// Check basic rate targeting for VBR mode.
TEST_P(DatarateTestLarge, BasicRateTargetingVBR) {
  BasicRateTargetingVBRTest();
}

// Check basic rate targeting for CBR.
TEST_P(DatarateTestLarge, BasicRateTargetingCBR) {
  BasicRateTargetingCBRTest();
}

// Check basic rate targeting for periodic key frame.
TEST_P(DatarateTestLarge, PeriodicKeyFrameCBR) {
  BasicRateTargetingCBRPeriodicKeyFrameTest();
}

// Check basic rate targeting for periodic key frame, aligned with scene change.
TEST_P(DatarateTestLarge, PeriodicKeyFrameCBROnSceneCuts) {
  CBRPeriodicKeyFrameOnSceneCuts();
}

// Check basic rate targeting with error resilience on for scene cuts.
TEST_P(DatarateTestLarge, ErrorResilienceOnSceneCuts) {
  ErrorResilienceOnSceneCuts();
}

// Check that (1) the first dropped frame gets earlier and earlier
// as the drop frame threshold is increased, and (2) that the total number of
// frame drops does not decrease as we increase frame drop threshold.
// Use a lower qp-max to force some frame drops.
TEST_P(DatarateTestFrameDropLarge, ChangingDropFrameThresh) {
  ChangingDropFrameThreshTest();
}
}  // namespace
}  // namespace datarate_test
