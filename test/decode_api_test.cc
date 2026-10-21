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

#include <climits>
#include <vector>

#include "gtest/gtest.h"

#include "config/aom_config.h"

#include "aom/aomdx.h"
#include "aom/aom_decoder.h"

namespace {

TEST(DecodeAPI, InvalidParams) {
  uint8_t buf[1] = { 0 };
  aom_codec_ctx_t dec;

  EXPECT_EQ(AOM_CODEC_INVALID_PARAM,
            aom_codec_dec_init(nullptr, nullptr, nullptr, 0));
  EXPECT_EQ(AOM_CODEC_INVALID_PARAM,
            aom_codec_dec_init(&dec, nullptr, nullptr, 0));
  EXPECT_EQ(AOM_CODEC_INVALID_PARAM,
            aom_codec_decode(nullptr, nullptr, 0, nullptr));
  EXPECT_EQ(AOM_CODEC_INVALID_PARAM,
            aom_codec_decode(nullptr, buf, 0, nullptr));
  EXPECT_EQ(AOM_CODEC_INVALID_PARAM,
            aom_codec_decode(nullptr, buf, sizeof(buf), nullptr));
  EXPECT_EQ(AOM_CODEC_INVALID_PARAM,
            aom_codec_decode(nullptr, nullptr, sizeof(buf), nullptr));
  EXPECT_EQ(AOM_CODEC_INVALID_PARAM, aom_codec_destroy(nullptr));
  EXPECT_NE(aom_codec_error(nullptr), nullptr);
  EXPECT_EQ(aom_codec_error_detail(nullptr), nullptr);

  aom_codec_iface_t *iface = aom_codec_av1_dx();
  EXPECT_EQ(AOM_CODEC_INVALID_PARAM,
            aom_codec_dec_init(nullptr, iface, nullptr, 0));

  EXPECT_EQ(AOM_CODEC_OK, aom_codec_dec_init(&dec, iface, nullptr, 0));
  EXPECT_EQ(AOM_CODEC_INVALID_PARAM,
            aom_codec_decode(&dec, nullptr, sizeof(buf), nullptr));
  EXPECT_EQ(AOM_CODEC_INVALID_PARAM, aom_codec_decode(&dec, buf, 0, nullptr));

  EXPECT_EQ(AOM_CODEC_OK, aom_codec_destroy(&dec));
}

TEST(DecodeAPI, InvalidControlId) {
  aom_codec_iface_t *iface = aom_codec_av1_dx();
  aom_codec_ctx_t dec;
  EXPECT_EQ(AOM_CODEC_OK, aom_codec_dec_init(&dec, iface, nullptr, 0));
  EXPECT_EQ(AOM_CODEC_ERROR, aom_codec_control(&dec, -1, 0));
  EXPECT_EQ(AOM_CODEC_INVALID_PARAM, aom_codec_control(&dec, 0, 0));
  EXPECT_EQ(AOM_CODEC_OK, aom_codec_destroy(&dec));
}

TEST(DecodeAPI, Buganizer503691210) {
  aom_codec_iface_t *iface = aom_codec_av1_dx();
  aom_codec_ctx_t dec;
  aom_codec_dec_cfg_t cfg = { 1, 424, 144, 1 };
  EXPECT_EQ(AOM_CODEC_OK, aom_codec_dec_init(&dec, iface, &cfg, 0));

  av1_ref_frame_t ref;
  ref.idx = 0;
  // These control calls should fail with AOM_CODEC_ERROR because the decoder
  // is not fully initialized yet (no frames decoded).
  EXPECT_EQ(AOM_CODEC_ERROR, aom_codec_control(&dec, AV1_GET_REFERENCE, &ref));
  EXPECT_EQ(AOM_CODEC_ERROR, aom_codec_control(&dec, AV1_SET_REFERENCE, &ref));
  EXPECT_EQ(AOM_CODEC_ERROR, aom_codec_control(&dec, AV1_COPY_REFERENCE, &ref));

  aom_image_t img;
  EXPECT_EQ(AOM_CODEC_ERROR,
            aom_codec_control(&dec, AV1_COPY_NEW_FRAME_IMAGE, &img));

  EXPECT_EQ(AOM_CODEC_OK, aom_codec_destroy(&dec));
}

TEST(DecodeAPI, SetExtRefPtr) {
  aom_codec_iface_t *iface = aom_codec_av1_dx();
  aom_codec_ctx_t dec;
  EXPECT_EQ(AOM_CODEC_OK, aom_codec_dec_init(&dec, iface, nullptr, 0));

  constexpr int kMaxExternalReferences = 128;
  std::vector<aom_image_t> imgs(2048);
  for (aom_image_t &img : imgs) {
    img = {};
    img.w = 1000;
    img.h = 1000;
    img.d_w = 1000;
    img.d_h = 1000;
    img.fmt = AOM_IMG_FMT_I420;
  }

  av1_ext_ref_frame_t ext_ref = { imgs.data(), kMaxExternalReferences + 1 };
  EXPECT_EQ(AOM_CODEC_INVALID_PARAM,
            aom_codec_control(&dec, AV1D_SET_EXT_REF_PTR, &ext_ref));

  ext_ref = { imgs.data(), static_cast<int>(imgs.size()) };
  EXPECT_EQ(AOM_CODEC_INVALID_PARAM,
            aom_codec_control(&dec, AV1D_SET_EXT_REF_PTR, &ext_ref));

  ext_ref = { imgs.data(), INT_MIN };
  EXPECT_EQ(AOM_CODEC_INVALID_PARAM,
            aom_codec_control(&dec, AV1D_SET_EXT_REF_PTR, &ext_ref));

  ext_ref = { imgs.data(), -1 };
  EXPECT_EQ(AOM_CODEC_INVALID_PARAM,
            aom_codec_control(&dec, AV1D_SET_EXT_REF_PTR, &ext_ref));

  ext_ref = { imgs.data(), 0 };
  EXPECT_EQ(AOM_CODEC_INVALID_PARAM,
            aom_codec_control(&dec, AV1D_SET_EXT_REF_PTR, &ext_ref));

  EXPECT_EQ(AOM_CODEC_INVALID_PARAM,
            aom_codec_control(&dec, AV1D_SET_EXT_REF_PTR, nullptr));

  ext_ref = { nullptr, 0 };
  EXPECT_EQ(AOM_CODEC_INVALID_PARAM,
            aom_codec_control(&dec, AV1D_SET_EXT_REF_PTR, &ext_ref));

  ext_ref = { nullptr, 1 };
  EXPECT_EQ(AOM_CODEC_INVALID_PARAM,
            aom_codec_control(&dec, AV1D_SET_EXT_REF_PTR, &ext_ref));

  ext_ref = { imgs.data(), 1 };
  EXPECT_EQ(AOM_CODEC_OK,
            aom_codec_control(&dec, AV1D_SET_EXT_REF_PTR, &ext_ref));
  EXPECT_EQ(ext_ref.img, imgs.data());

  ext_ref = { imgs.data(), kMaxExternalReferences };
  EXPECT_EQ(AOM_CODEC_OK,
            aom_codec_control(&dec, AV1D_SET_EXT_REF_PTR, &ext_ref));
  EXPECT_EQ(ext_ref.img, imgs.data());

  EXPECT_EQ(AOM_CODEC_OK, aom_codec_destroy(&dec));
}

}  // namespace
