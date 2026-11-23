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
#include <cstdint>

#include "config/aom_config.h"

#include "aom/aom_encoder.h"
#include "aom/aom_image.h"
#include "aom/aomcx.h"
#include "gtest/gtest.h"
#include "test/codec_factory.h"
#include "test/encode_test_driver.h"
#include "test/i420_video_source.h"
#include "test/util.h"
#include "test/video_source.h"

namespace {

const libaom_test::TestMode kTestModeParams[] =
#if CONFIG_REALTIME_ONLY
    { ::libaom_test::kRealTime };
#else
    { ::libaom_test::kRealTime, ::libaom_test::kOnePassGood };
#endif

class AqSegmentTest
    : public ::libaom_test::CodecTestWith3Params<libaom_test::TestMode, int,
                                                 int>,
      public ::libaom_test::EncoderTest {
 protected:
  AqSegmentTest() : EncoderTest(GET_PARAM(0)) {}
  ~AqSegmentTest() override = default;

  void SetUp() override {
    InitializeConfig(GET_PARAM(1));
    set_cpu_used_ = GET_PARAM(2);
    aq_mode_ = 0;
  }

  void PreEncodeFrameHook(::libaom_test::VideoSource *video,
                          ::libaom_test::Encoder *encoder) override {
    if (video->frame() == 0) {
      encoder->Control(AOME_SET_CPUUSED, set_cpu_used_);
      encoder->Control(AV1E_SET_AQ_MODE, aq_mode_);
      encoder->Control(AV1E_SET_DELTAQ_MODE, deltaq_mode_);
      encoder->Control(AOME_SET_MAX_INTRA_BITRATE_PCT, 100);
      if (mode_ == ::libaom_test::kRealTime) {
        encoder->Control(AV1E_SET_ALLOW_WARPED_MOTION, 0);
        encoder->Control(AV1E_SET_ENABLE_GLOBAL_MOTION, 0);
        encoder->Control(AV1E_SET_ENABLE_OBMC, 0);
      }
    }
  }

  void DoTest(int aq_mode) {
    aq_mode_ = aq_mode;
    deltaq_mode_ = 0;
    cfg_.kf_max_dist = 12;
    cfg_.rc_min_quantizer = 8;
    cfg_.rc_max_quantizer = 56;
    cfg_.rc_end_usage = AOM_CBR;
    cfg_.g_lag_in_frames = 6;
    cfg_.rc_buf_initial_sz = 500;
    cfg_.rc_buf_optimal_sz = 500;
    cfg_.rc_buf_sz = 1000;
    cfg_.rc_target_bitrate = 300;
    ::libaom_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352,
                                         288, 30, 1, 0, 15);
    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  }

  int set_cpu_used_;
  int aq_mode_;
  int deltaq_mode_;
};

// Validate that this AQ segmentation mode (1-variance_aq, 2-complexity_aq,
// 3-cyclic_refresh_aq) encodes and decodes without a mismatch.
TEST_P(AqSegmentTest, TestNoMisMatch) { DoTest(GET_PARAM(3)); }

#if !CONFIG_REALTIME_ONLY
// Validate that this delta q mode
// encodes and decodes without a mismatch.
TEST_P(AqSegmentTest, TestNoMisMatchExtDeltaQ) {
  cfg_.rc_end_usage = AOM_CQ;
  aq_mode_ = 0;
  deltaq_mode_ = 2;
  ::libaom_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352, 288,
                                       30, 1, 0, 15);

  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}
#endif

AV1_INSTANTIATE_TEST_SUITE(AqSegmentTest, ::testing::ValuesIn(kTestModeParams),
                           ::testing::Range(5, 9), ::testing::Range(0, 4));

#if !CONFIG_REALTIME_ONLY
class AqSegmentTestLarge : public AqSegmentTest {};

TEST_P(AqSegmentTestLarge, TestNoMisMatch) { DoTest(GET_PARAM(3)); }

AV1_INSTANTIATE_TEST_SUITE(AqSegmentTestLarge,
                           ::testing::Values(::libaom_test::kOnePassGood),
                           ::testing::Range(3, 5), ::testing::Range(0, 4));

// Synthetic source with low-frequency gradients and textured patches so
// complexity AQ exercises large superblocks and high-complexity segments when
// base_qindex reaches 0.
class GradientTextureVideoSource : public ::libaom_test::DummyVideoSource {
 protected:
  void FillFrame() override {
    ASSERT_NE(img_, nullptr);
    const unsigned int f = frame_;
    for (int plane = 0; plane < 3; ++plane) {
      const int w = aom_img_plane_width(img_, plane);
      const int h = aom_img_plane_height(img_, plane);
      const int stride = img_->stride[plane];
      uint8_t *const buf = img_->planes[plane];
      for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
          if (plane == 0 && c >= w / 2) {
            buf[r * stride + c] =
                static_cast<uint8_t>((r * 37 + c * 53 + f * 19) & 0xff);
          } else {
            buf[r * stride + c] =
                static_cast<uint8_t>((128 + ((r + c + f * 3) >> 2)) & 0xff);
          }
        }
      }
    }
  }
};

// Maps a 0-63 quantizer level to a qindex, matching av1_quantizer_to_qindex().
int QuantizerToQindex(int quantizer) {
  if (quantizer >= 62) return quantizer == 62 ? 249 : 255;
  return 4 * quantizer;
}

// Fixture for complexity AQ (aq-mode=2) regression tests. The parameter is a
// 0-63 quantizer level whose meaning depends on the test. Records the min/max
// base_qindex over the frames that produced a compressed frame packet.
class AqComplexityTest
    : public ::libaom_test::CodecTestWith2Params<libaom_test::TestMode, int>,
      public ::libaom_test::EncoderTest {
 protected:
  AqComplexityTest() : EncoderTest(GET_PARAM(0)), q_level_(GET_PARAM(2)) {}
  ~AqComplexityTest() override = default;

  void SetUp() override {
    InitializeConfig(GET_PARAM(1));
    cfg_.rc_end_usage = AOM_Q;
    cfg_.rc_min_quantizer = 0;
    cfg_.rc_max_quantizer = 63;
    // In two-pass encoding, rc_target_bitrate drives rc.sb64_target_rate
    // even in AOM_Q mode.
    cfg_.rc_target_bitrate = 5000;
  }

  void BeginPassHook(unsigned int /*pass*/) override {
    encoder_ = nullptr;
    min_qindex_ = 256;
    max_qindex_ = -1;
    encoded_frames_ = 0;
  }

  void EndPassHook() override { encoder_ = nullptr; }

  void PreEncodeFrameHook(::libaom_test::VideoSource *video,
                          ::libaom_test::Encoder *encoder) override {
    encoder_ = encoder;
    if (video->frame() == 0) {
      encoder->Control(AOME_SET_CPUUSED, 5);
      encoder->Control(AV1E_SET_AQ_MODE, 2);
      encoder->Control(AOME_SET_CQ_LEVEL, q_level_);
      encoder->Control(AV1E_SET_SUPERBLOCK_SIZE, AOM_SUPERBLOCK_SIZE_128X128);
    }
  }

  // AOME_GET_LAST_QUANTIZER reports the base_qindex of the last frame encoded
  // by the preceding aom_codec_encode() call.
  void FramePktHook(const aom_codec_cx_pkt_t * /*pkt*/) override {
    ASSERT_NE(encoder_, nullptr);
    int qindex = -1;
    encoder_->Control(AOME_GET_LAST_QUANTIZER, &qindex);
    min_qindex_ = std::min(min_qindex_, qindex);
    max_qindex_ = std::max(max_qindex_, qindex);
    ++encoded_frames_;
  }

  void RunTest() {
    GradientTextureVideoSource video;
    video.SetSize(256, 256);
    video.set_limit(kNumFrames);
    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
    EXPECT_EQ(encoded_frames_, kNumFrames);
  }

  static constexpr int kNumFrames = 5;
  const int q_level_;
  ::libaom_test::Encoder *encoder_ = nullptr;
  int min_qindex_ = 256;
  int max_qindex_ = -1;
  int encoded_frames_ = 0;
};

// In one-pass (LAP) AOM_Q mode this_frame_target is 0, so
// rc.sb64_target_rate is 0 and complexity AQ never selects segments; use
// two-pass.
class AqComplexityLowCqTest : public AqComplexityTest {};

// Regression test for aomedia:589732449: verify complexity AQ encodes and
// decodes cleanly when rate control selects base_qindex 0. The cq_level 1 key
// frame reaches qindex 0 because its TPL-based q is clamped only to
// rc_min_quantizer.
TEST_P(AqComplexityLowCqTest, EncodeDecodeLowCq) {
  RunTest();
  EXPECT_EQ(min_qindex_, 0);
  if (q_level_ == 0) {
    EXPECT_EQ(max_qindex_, 0);
  } else {
    EXPECT_GT(max_qindex_, 0);
  }
}

AV1_INSTANTIATE_TEST_SUITE(AqComplexityLowCqTest,
                           ::testing::Values(::libaom_test::kTwoPassGood),
                           ::testing::Values(0, 1));

class AqComplexityFixedQTest : public AqComplexityTest {};

// With rc_min_quantizer equal to rc_max_quantizer, rc->best_quality ==
// rc->worst_quality, so every segment qindex delta is 0. All AQ segments must
// remain active since av1_caq_select_segment() assigns segment IDs after RD.
// Use a moderate bitrate target so high-Q blocks still exceed target_rate and
// exercise non-zero segment IDs.
TEST_P(AqComplexityFixedQTest, EncodeDecodeFixedQ) {
  cfg_.rc_min_quantizer = q_level_;
  cfg_.rc_max_quantizer = q_level_;
  cfg_.rc_target_bitrate = 200;
  RunTest();
  const int expected_qindex = QuantizerToQindex(q_level_);
  EXPECT_EQ(min_qindex_, expected_qindex);
  EXPECT_EQ(max_qindex_, expected_qindex);
}

AV1_INSTANTIATE_TEST_SUITE(AqComplexityFixedQTest,
                           ::testing::Values(::libaom_test::kTwoPassGood),
                           ::testing::Values(16, 32, 48, 63));
#endif  // !CONFIG_REALTIME_ONLY
}  // namespace
