/*
 * Copyright (c) 2025, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include "test/codec_factory.h"
#include "test/encode_test_driver.h"
#include "test/util.h"
#include "test/y4m_video_source.h"
#include "aom/aom_ext_ratectrl.h"
#include "gtest/gtest.h"

namespace {

const int kFrameNum = 5;

// A mock rate control model.
struct MockRateCtrlModel {};

// A mock private data for rate control callbacks.
struct MockRC {};
struct MockRC g_priv;

// A flag to indicate if create_model() is called.
bool is_create_model_called = false;

// A flag to indicate if delete_model() is called.
bool is_delete_model_called = false;

// A flag to indicate if send_firstpass_stats() is called.
bool is_send_firstpass_stats_called = false;

// A flag to indicate if send_tpl_gop_stats() is called.
bool is_send_extrc_tpl_gop_stats_called = false;

aom_rc_status_t mock_create_model(void *priv,
                                  const aom_rc_config_t *ratectrl_config,
                                  aom_rc_model_t *ratectrl_model) {
  (void)priv;
  (void)ratectrl_config;
  EXPECT_NE(ratectrl_model, nullptr);
  *ratectrl_model = (aom_rc_model_t)(new MockRateCtrlModel());
  is_create_model_called = true;
  return AOM_RC_OK;
}

aom_rc_status_t mock_delete_model(aom_rc_model_t ratectrl_model) {
  EXPECT_NE(ratectrl_model, nullptr);
  delete (MockRateCtrlModel *)ratectrl_model;
  is_delete_model_called = true;
  return AOM_RC_OK;
}

aom_rc_status_t mock_send_firstpass_stats(
    aom_rc_model_t ratectrl_model,
    const aom_rc_firstpass_stats_t *firstpass_stats) {
  EXPECT_NE(ratectrl_model, nullptr);
  EXPECT_NE(firstpass_stats, nullptr);
  EXPECT_EQ(firstpass_stats->num_frames, kFrameNum);
  EXPECT_NE(firstpass_stats->frame_stats, nullptr);
  is_send_firstpass_stats_called = true;
  return AOM_RC_OK;
}

aom_rc_status_t mock_send_extrc_tpl_gop_stats(
    aom_rc_model_t ratectrl_model, const AomTplGopStats *extrc_tpl_gop_stats) {
  EXPECT_NE(ratectrl_model, nullptr);
  EXPECT_NE(extrc_tpl_gop_stats, nullptr);
  EXPECT_GT(extrc_tpl_gop_stats->size, 0);
  EXPECT_NE(extrc_tpl_gop_stats->frame_stats_list, nullptr);
  is_send_extrc_tpl_gop_stats_called = true;
  return AOM_RC_OK;
}

class ExtRateCtrlTest : public ::libaom_test::EncoderTest,
                        public ::libaom_test::CodecTestWith2Params<int, int> {
 protected:
  ExtRateCtrlTest() : EncoderTest(GET_PARAM(0)), cpu_used_(GET_PARAM(2)) {
    aom_rc_funcs_t *rc_funcs = &rc_funcs_;
    rc_funcs->priv = &g_priv;
    rc_funcs->rc_type = AOM_RC_QP;
    rc_funcs->create_model = mock_create_model;
    rc_funcs->delete_model = mock_delete_model;
    rc_funcs->send_firstpass_stats = mock_send_firstpass_stats;
    rc_funcs->send_tpl_gop_stats = mock_send_extrc_tpl_gop_stats;
    rc_funcs->get_encodeframe_decision = nullptr;
    rc_funcs->update_encodeframe_result = nullptr;
  }
  ~ExtRateCtrlTest() override = default;

  void SetUp() override {
    InitializeConfig(static_cast<libaom_test::TestMode>(GET_PARAM(1)));
    cfg_.g_threads = 1;
    cfg_.g_limit = kFrameNum;
    is_create_model_called = false;
    is_delete_model_called = false;
    is_send_firstpass_stats_called = false;
    is_send_extrc_tpl_gop_stats_called = false;
  }

  void PreEncodeFrameHook(::libaom_test::VideoSource *video,
                          ::libaom_test::Encoder *encoder) override {
    if (video->frame() == 0) {
      encoder->Control(AOME_SET_CPUUSED, cpu_used_);
      encoder->Control(AV1E_SET_EXTERNAL_RATE_CONTROL, &rc_funcs_);
    }
    current_encoder_ = encoder;
  }

  ::libaom_test::Encoder *current_encoder_;
  aom_rc_funcs_t rc_funcs_;
  int cpu_used_;
};

TEST_P(ExtRateCtrlTest, TestExternalRateCtrl) {
  ::libaom_test::Y4mVideoSource video("screendata.y4m", 0, kFrameNum);
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  EXPECT_TRUE(is_create_model_called);
  EXPECT_TRUE(is_send_firstpass_stats_called);
  EXPECT_TRUE(is_send_extrc_tpl_gop_stats_called);
  EXPECT_TRUE(is_delete_model_called);
}

AV1_INSTANTIATE_TEST_SUITE(ExtRateCtrlTest,
                           ::testing::Values(::libaom_test::kTwoPassGood),
                           ::testing::Values(0));

// Corresponds to QP 43 for 8-bit video.
const int kFrameQindex = 172;

aom_rc_status_t mock_get_encodeframe_decision_const_q(
    aom_rc_model_t ratectrl_model, int frame_gop_index,
    aom_rc_encodeframe_decision_t *frame_decision) {
  (void)ratectrl_model;
  (void)frame_gop_index;
  // Set constant QP.
  frame_decision->q_index = kFrameQindex;
  frame_decision->sb_params_list = NULL;  // Initialize to NULL
  return AOM_RC_OK;
}

class ExtRateCtrlQpTest : public ExtRateCtrlTest {
 protected:
  ExtRateCtrlQpTest() {
    rc_funcs_.get_encodeframe_decision = mock_get_encodeframe_decision_const_q;
  }
  ~ExtRateCtrlQpTest() override = default;

  void SetUp() override {
    InitializeConfig(static_cast<libaom_test::TestMode>(GET_PARAM(1)));
    cfg_.g_threads = 1;
    cfg_.g_limit = kFrameNum;
    is_create_model_called = false;
    is_delete_model_called = false;
  }

  void FramePktHook(const aom_codec_cx_pkt_t *pkt) override {
    if (pkt->kind != AOM_CODEC_CX_FRAME_PKT) return;
    int q_index = -1;
    current_encoder_->Control(AOME_GET_LAST_QUANTIZER, &q_index);
    std::cout << q_index << std::endl;
    EXPECT_EQ(q_index, kFrameQindex);
  }
};

TEST_P(ExtRateCtrlQpTest, TestExternalRateCtrlConstQp) {
  ::libaom_test::Y4mVideoSource video("screendata.y4m", 0, kFrameNum);
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  EXPECT_TRUE(is_create_model_called);
  EXPECT_TRUE(is_delete_model_called);
}

AV1_INSTANTIATE_TEST_SUITE(ExtRateCtrlQpTest,
                           ::testing::Values(::libaom_test::kTwoPassGood),
                           ::testing::Values(0));
}  // namespace
