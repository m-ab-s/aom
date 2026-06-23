/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */
#include "config/aom_config.h"

#include "av1/encoder/av1_quantize.h"
#include "gtest/gtest.h"
#include "test/codec_factory.h"
#include "test/encode_test_driver.h"
#include "test/i420_video_source.h"
#include "test/util.h"

namespace {

const ::libaom_test::TestMode kTestMode[] =
#if CONFIG_REALTIME_ONLY
    { ::libaom_test::kRealTime };
#else
    { ::libaom_test::kRealTime, ::libaom_test::kOnePassGood };
#endif

const int kUseFixedQPOffsetsMode[] = { 1, 2 };
const int kCpuUsed = 6;

class UseFixedQPOffsetsTest
    : public ::libaom_test::CodecTestWith3Params<libaom_test::TestMode, int,
                                                 int>,
      public ::libaom_test::EncoderTest {
 protected:
  UseFixedQPOffsetsTest() : EncoderTest(GET_PARAM(0)) {}
  ~UseFixedQPOffsetsTest() override = default;

  void SetUp() override {
    InitializeConfig(GET_PARAM(1));
    cfg_.kf_max_dist = 9999;
    cfg_.rc_end_usage = AOM_Q;
    cfg_.use_fixed_qp_offsets = GET_PARAM(2);
    frame_qp_ = GET_PARAM(3);
    expected_leaf_qindex_ =
        av1_quantizer_to_qindex(frame_qp_ > 63 ? 63 : frame_qp_);
  }

  void PreEncodeFrameHook(::libaom_test::VideoSource *video,
                          ::libaom_test::Encoder *encoder) override {
    if (video->frame() == 0) {
      encoder->Control(AOME_SET_CPUUSED, kCpuUsed);
      encoder->Control(AOME_SET_CQ_LEVEL, frame_qp_);
    }
  }

  void PostEncodeFrameHook(::libaom_test::Encoder *encoder) override {
    int q = 0;
    encoder->Control(AOME_GET_LAST_QUANTIZER, &q);

    // If a call to encoder->EncodeFrame() results in a last QP of 0,
    // interpret as the frame being read into the lookahead buffer.
    if (q == 0) return;

    if (cfg_.use_fixed_qp_offsets == 2) {
      // Setting use_fixed_qp_offsets = 2 means every frame should use the same
      // qindex.
      ASSERT_EQ(q, expected_leaf_qindex_);
    } else {
      ASSERT_LE(q, expected_leaf_qindex_);
    }
  }

  void DoTest() {
    ::libaom_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352,
                                         288, 30, 1, 0, 33);
    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  }

  int frame_qp_;
  int expected_leaf_qindex_;
};

#if !CONFIG_REALTIME_ONLY
const int kFrameCount = 17;

class ExtendedCqLevelEndToEndTest
    : public ::libaom_test::CodecTestWithParam<libaom_test::TestMode>,
      public ::libaom_test::EncoderTest {
 protected:
  ExtendedCqLevelEndToEndTest()
      : EncoderTest(GET_PARAM(0)), encoding_mode_(GET_PARAM(1)) {}
  ~ExtendedCqLevelEndToEndTest() override = default;

  void SetUp() override { InitializeConfig(encoding_mode_); }

  void BeginPassHook(unsigned int) override {
    encoded_bytes_ = 0;
    decoded_frames_ = 0;
  }

  void PreEncodeFrameHook(::libaom_test::VideoSource *video,
                          ::libaom_test::Encoder *encoder) override {
    if (video->frame() == 0) {
      encoder->Control(AOME_SET_CPUUSED, kCpuUsed);
      encoder->Control(AOME_SET_CQ_LEVEL, cq_level_);
      encoder->Control(AOME_SET_ENABLEAUTOALTREF, 1);
      encoder->Control(AV1E_SET_FRAME_PARALLEL_DECODING, 0);
      encoder->Control(AV1E_SET_DELTAQ_MODE, 0);
      encoder->Control(AV1E_SET_ENABLE_TPL_MODEL, 0);
      encoder->Control(AV1E_SET_ENABLE_KEYFRAME_FILTERING, 1);
      encoder->Control(AV1E_SET_MIN_GF_INTERVAL, 16);
      encoder->Control(AV1E_SET_MAX_GF_INTERVAL, 16);
      encoder->Control(AV1E_SET_GF_MIN_PYRAMID_HEIGHT, 4);
      encoder->Control(AV1E_SET_GF_MAX_PYRAMID_HEIGHT, 4);
    }
  }

  void FramePktHook(const aom_codec_cx_pkt_t *pkt) override {
    encoded_bytes_ += pkt->data.frame.sz;
  }

  void DecompressedFrameHook(const aom_image_t &, aom_codec_pts_t) override {
    ++decoded_frames_;
  }

  void EncodeAtCqLevel(int cq_level, size_t *encoded_bytes) {
    cq_level_ = cq_level;
    InitializeConfig(encoding_mode_);
    cfg_.rc_end_usage = AOM_Q;
    cfg_.use_fixed_qp_offsets = 1;
    cfg_.g_lag_in_frames = 19;
    cfg_.kf_min_dist = 65;
    cfg_.kf_max_dist = 65;

    ::libaom_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352,
                                         288, 30, 1, 0, kFrameCount);
    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
    EXPECT_EQ(decoded_frames_, kFrameCount);
    EXPECT_GT(encoded_bytes_, 0u);
    *encoded_bytes = encoded_bytes_;
  }

 private:
  const ::libaom_test::TestMode encoding_mode_;
  int cq_level_;
  size_t encoded_bytes_;
  int decoded_frames_;
};
#endif  // !CONFIG_REALTIME_ONLY

TEST_P(UseFixedQPOffsetsTest, TestQPOffsets) { DoTest(); }

#if !CONFIG_REALTIME_ONLY
TEST_P(ExtendedCqLevelEndToEndTest,
       HighCqLevelsProduceDecodableMonotonicBitstreams) {
  size_t previous_encoded_bytes = 0;
  ASSERT_NO_FATAL_FAILURE(EncodeAtCqLevel(63, &previous_encoded_bytes));
  for (int cq_level = 64; cq_level <= 70; ++cq_level) {
    size_t encoded_bytes = 0;
    ASSERT_NO_FATAL_FAILURE(EncodeAtCqLevel(cq_level, &encoded_bytes));
    EXPECT_LT(encoded_bytes, previous_encoded_bytes)
        << "cq_level = " << cq_level;
    previous_encoded_bytes = encoded_bytes;
  }
}
#endif  // !CONFIG_REALTIME_ONLY

TEST(ExtendedCqLevelTest, AcceptsHighCqLevelOnlyInConstantQualityMode) {
  aom_codec_enc_cfg_t cfg;
  ASSERT_EQ(AOM_CODEC_OK, aom_codec_enc_config_default(aom_codec_av1_cx(), &cfg,
                                                       AOM_USAGE_REALTIME));

  cfg.rc_end_usage = AOM_Q;
  aom_codec_ctx_t enc_q;
  ASSERT_EQ(AOM_CODEC_OK,
            aom_codec_enc_init(&enc_q, aom_codec_av1_cx(), &cfg, 0));
  EXPECT_EQ(AOM_CODEC_OK, aom_codec_control(&enc_q, AOME_SET_CQ_LEVEL, 70));
  EXPECT_EQ(AOM_CODEC_INVALID_PARAM,
            aom_codec_control(&enc_q, AOME_SET_CQ_LEVEL, 71));
  EXPECT_EQ(AOM_CODEC_OK, aom_codec_destroy(&enc_q));

  cfg.rc_end_usage = AOM_CQ;
  aom_codec_ctx_t enc_cq;
  ASSERT_EQ(AOM_CODEC_OK,
            aom_codec_enc_init(&enc_cq, aom_codec_av1_cx(), &cfg, 0));
  EXPECT_EQ(AOM_CODEC_INVALID_PARAM,
            aom_codec_control(&enc_cq, AOME_SET_CQ_LEVEL, 70));
  EXPECT_EQ(AOM_CODEC_OK, aom_codec_destroy(&enc_cq));
}

AV1_INSTANTIATE_TEST_SUITE(UseFixedQPOffsetsTest,
                           ::testing::ValuesIn(kTestMode),
                           ::testing::ValuesIn(kUseFixedQPOffsetsMode),
                           ::testing::Values(35, 70));

#if !CONFIG_REALTIME_ONLY
AV1_INSTANTIATE_TEST_SUITE(ExtendedCqLevelEndToEndTest,
                           ::testing::Values(::libaom_test::kOnePassGood));
#endif  // !CONFIG_REALTIME_ONLY
}  // namespace
