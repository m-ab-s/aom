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

#include <assert.h>
#include <string.h>
#include <string>
#include <vector>
#include "aom/aom_integer.h"
#include "gtest/gtest.h"
#include "test/codec_factory.h"
#include "test/encode_test_driver.h"
#include "test/i420_video_source.h"
#include "test/md5_helper.h"
#include "test/util.h"

namespace {
// The number of frames to be encoded/decoded
const int kLimit = 8;
// Skip 1 frame to check the frame decoding independency.
const int kSkip = 5;
const int kTileSize = 1;
const int kTIleSizeInPixels = (kTileSize << 6);
// Fake width and height so that they can be multiples of the tile size.
const int kImgWidth = 704;
const int kImgHeight = 576;
const int kInvalidTileImgSize = 256;
const int kOutOfRangeTileCol = 2;
const int kPartialTileImgSize = 130;

// This test tests large scale tile coding case. Non-large-scale tile coding
// is tested by the tile_independence test.
class AV1ExtTileTest
    : public ::libaom_test::CodecTestWith2Params<libaom_test::TestMode, int>,
      public ::libaom_test::EncoderTest {
 protected:
  AV1ExtTileTest()
      : EncoderTest(GET_PARAM(0)), encoding_mode_(GET_PARAM(1)),
        set_cpu_used_(GET_PARAM(2)) {
    init_flags_ = AOM_CODEC_USE_PSNR;
    aom_codec_dec_cfg_t cfg = aom_codec_dec_cfg_t();
    cfg.w = kImgWidth;
    cfg.h = kImgHeight;
    cfg.allow_lowbitdepth = 1;

    decoder_ = codec_->CreateDecoder(cfg, 0);
    decoder_->Control(AV1_SET_TILE_MODE, 1);
    decoder_->Control(AV1D_EXT_TILE_DEBUG, 1);
    decoder_->Control(AV1_SET_DECODE_TILE_ROW, -1);
    decoder_->Control(AV1_SET_DECODE_TILE_COL, -1);

    // Allocate buffer to store tile image.
    aom_img_alloc(&tile_img_, AOM_IMG_FMT_I420, kImgWidth, kImgHeight, 32);

    md5_.clear();
    tile_md5_.clear();
  }

  ~AV1ExtTileTest() override {
    if (ext_ref_allocated_) aom_img_free(&ext_ref_img_);
    aom_img_free(&tile_img_);
    delete decoder_;
  }

  void SetUp() override {
    InitializeConfig(encoding_mode_);

    cfg_.g_lag_in_frames = 0;
    cfg_.rc_end_usage = AOM_VBR;
    cfg_.g_error_resilient = 1;

    cfg_.rc_max_quantizer = 56;
    cfg_.rc_min_quantizer = 0;
  }

  void PreEncodeFrameHook(::libaom_test::VideoSource *video,
                          ::libaom_test::Encoder *encoder) override {
    if (video->frame() == 0) {
      // Encode setting
      encoder->Control(
          AOME_SET_CPUUSED,
          (test_invalid_tile_ || test_partial_tile_list_) ? 8 : set_cpu_used_);
      encoder->Control(AOME_SET_ENABLEAUTOALTREF, 0);
      encoder->Control(AV1E_SET_FRAME_PARALLEL_DECODING, 1);

      // TODO(yunqingwang): test single_tile_decoding = 0.
      encoder->Control(AV1E_SET_SINGLE_TILE_DECODING, 1);
      // Always use 64x64 max partition.
      encoder->Control(AV1E_SET_SUPERBLOCK_SIZE, AOM_SUPERBLOCK_SIZE_64X64);
      // Set tile_columns and tile_rows to MAX values, which guarantees the tile
      // size of 64 x 64 pixels(i.e. 1 SB) for <= 4k resolution.
      encoder->Control(AV1E_SET_TILE_COLUMNS,
                       (test_invalid_tile_ || test_partial_tile_list_) ? 1 : 6);
      encoder->Control(AV1E_SET_TILE_ROWS,
                       (test_invalid_tile_ || test_partial_tile_list_) ? 1 : 6);
    } else if (video->frame() == 1) {
      frame_flags_ =
          AOM_EFLAG_NO_UPD_LAST | AOM_EFLAG_NO_UPD_GF | AOM_EFLAG_NO_UPD_ARF;
    }
  }

  void DecompressedFrameHook(const aom_image_t &img,
                             aom_codec_pts_t pts) override {
    // Skip 1 already decoded frame to be consistent with the decoder in this
    // test.
    if (pts == (aom_codec_pts_t)kSkip) return;

    // Calculate MD5 as the reference.
    ::libaom_test::MD5 md5_res;
    md5_res.Add(&img);
    md5_.push_back(md5_res.Get());
  }

  void FramePktHook(const aom_codec_cx_pkt_t *pkt) override {
    if (test_invalid_tile_) {
      decoder_->Control(AV1_SET_DECODE_TILE_ROW, 0);
      decoder_->Control(AV1_SET_DECODE_TILE_COL, kOutOfRangeTileCol);
      EXPECT_EQ(AOM_CODEC_INVALID_PARAM,
                decoder_->DecodeFrame(
                    reinterpret_cast<uint8_t *>(pkt->data.frame.buf),
                    pkt->data.frame.sz));
      return;
    }

    if (test_partial_tile_list_) {
      const uint8_t *const frame =
          reinterpret_cast<const uint8_t *>(pkt->data.frame.buf);
      const size_t frame_sz = pkt->data.frame.sz;
      if (pkt->data.frame.pts == 0) {
        decoder_->Control(AV1_SET_DECODE_TILE_ROW, -1);
        decoder_->Control(AV1_SET_DECODE_TILE_COL, -1);
        ASSERT_EQ(AOM_CODEC_OK, decoder_->DecodeFrame(frame, frame_sz));
        ASSERT_NE(nullptr,
                  aom_img_alloc_with_border(&ext_ref_img_, AOM_IMG_FMT_I420,
                                            kPartialTileImgSize,
                                            kPartialTileImgSize, 32, 8, 64));
        ext_ref_allocated_ = true;
        decoder_->Control(AV1_COPY_NEW_FRAME_IMAGE, &ext_ref_img_);
        return;
      }

      // Decode the full frame once (to ensure an additional frame buffer slot
      // in the buffer pool is allocated) and then decode the bottom-right
      // partial tile (row=1, col=1) to extract the camera frame header and
      // tile data.
      decoder_->Control(AV1_SET_DECODE_TILE_ROW, -1);
      decoder_->Control(AV1_SET_DECODE_TILE_COL, -1);
      ASSERT_EQ(AOM_CODEC_OK, decoder_->DecodeFrame(frame, frame_sz));

      decoder_->Control(AV1_SET_DECODE_TILE_ROW, 1);
      decoder_->Control(AV1_SET_DECODE_TILE_COL, 1);
      ASSERT_EQ(AOM_CODEC_OK, decoder_->DecodeFrame(frame, frame_sz));
      const aom_image_t *single_tile_img = decoder_->GetDxData().Next();
      ASSERT_NE(nullptr, single_tile_img);
      ASSERT_GE(single_tile_img->d_w, 2u);
      ASSERT_GE(single_tile_img->d_h, 2u);
      const uint8_t expected_y[2][2] = {
        { single_tile_img->planes[0][0], single_tile_img->planes[0][1] },
        { single_tile_img->planes[0][single_tile_img->stride[0]],
          single_tile_img->planes[0][single_tile_img->stride[0] + 1] },
      };
      const uint8_t expected_u = single_tile_img->planes[1][0];
      const uint8_t expected_v = single_tile_img->planes[2][0];

      aom_tile_data frame_header_info = { 0, nullptr, 0 };
      decoder_->Control(AV1D_GET_FRAME_HEADER_INFO, &frame_header_info);
      const size_t obu_size_offset =
          static_cast<const uint8_t *>(frame_header_info.coded_tile_data) -
          frame;
      const size_t length_field_size = frame_header_info.coded_tile_data_size;
      const uint32_t frame_header_size =
          static_cast<uint32_t>(frame_header_info.extra_size) - 1;
      const size_t bytes_to_copy =
          obu_size_offset + length_field_size + frame_header_size;
      std::vector<uint8_t> frame_hdr_buf(frame, frame + bytes_to_copy);
      size_t bytes_written = 0;
      ASSERT_EQ(0, aom_uleb_encode_fixed_size(
                       frame_header_size, length_field_size, length_field_size,
                       frame_hdr_buf.data() + obu_size_offset, &bytes_written));

      aom_tile_data tile_data = { 0, nullptr, 0 };
      decoder_->Control(AV1D_GET_TILE_DATA, &tile_data);
      ASSERT_GT(tile_data.coded_tile_data_size, 0u);
      const uint32_t tile_list_payload_size =
          4 + 5 + static_cast<uint32_t>(tile_data.coded_tile_data_size);
      std::vector<uint8_t> tile_list_buf(5 + tile_list_payload_size, 0);
      tile_list_buf[0] = (OBU_TILE_LIST << 3) | (1 << 1);
      ASSERT_EQ(0, aom_uleb_encode_fixed_size(tile_list_payload_size, 4, 4,
                                              tile_list_buf.data() + 1,
                                              &bytes_written));
      uint8_t *const tl = tile_list_buf.data() + 9;
      tl[0] = 0;  // anchor_frame_idx
      tl[1] = 1;  // dec_tile_row (partial bottom edge tile)
      tl[2] = 1;  // dec_tile_col (partial right edge tile)
      const uint16_t coded_size_minus_1 =
          static_cast<uint16_t>(tile_data.coded_tile_data_size - 1);
      tl[3] = (coded_size_minus_1 >> 8) & 0xff;
      tl[4] = coded_size_minus_1 & 0xff;
      memcpy(tl + 5, tile_data.coded_tile_data, tile_data.coded_tile_data_size);

      av1_ext_ref_frame_t ext_ref = { &ext_ref_img_, 1 };
      decoder_->Control(AV1D_EXT_TILE_DEBUG, 0);
      decoder_->Control(AV1_SET_TILE_MODE, 1);
      decoder_->Control(AV1D_SET_EXT_REF_PTR, &ext_ref);
      ASSERT_EQ(AOM_CODEC_OK, decoder_->DecodeFrame(frame_hdr_buf.data(),
                                                    frame_hdr_buf.size()));
      ASSERT_EQ(AOM_CODEC_OK, decoder_->DecodeFrame(tile_list_buf.data(),
                                                    tile_list_buf.size()));
      const aom_image_t *img = decoder_->GetDxData().Next();
      ASSERT_NE(nullptr, img);
      EXPECT_EQ(expected_y[0][0], img->planes[0][0]);
      EXPECT_EQ(expected_y[0][1], img->planes[0][1]);
      EXPECT_EQ(expected_y[1][0], img->planes[0][img->stride[0]]);
      EXPECT_EQ(expected_y[1][1], img->planes[0][img->stride[0] + 1]);
      EXPECT_EQ(0u, img->planes[0][2]);
      EXPECT_EQ(0u, img->planes[0][2 * img->stride[0]]);
      EXPECT_EQ(expected_u, img->planes[1][0]);
      EXPECT_EQ(0u, img->planes[1][1]);
      EXPECT_EQ(0u, img->planes[1][img->stride[1]]);
      EXPECT_EQ(expected_v, img->planes[2][0]);
      EXPECT_EQ(0u, img->planes[2][1]);
      EXPECT_EQ(0u, img->planes[2][img->stride[2]]);
      return;
    }

    // Skip decoding 1 frame.
    if (pkt->data.frame.pts == (aom_codec_pts_t)kSkip) return;

    bool IsLastFrame = (pkt->data.frame.pts == (aom_codec_pts_t)(kLimit - 1));

    // Decode the first (kLimit - 1) frames as whole frame, and decode the last
    // frame in single tiles.
    for (int r = 0; r < kImgHeight / kTIleSizeInPixels; ++r) {
      for (int c = 0; c < kImgWidth / kTIleSizeInPixels; ++c) {
        if (!IsLastFrame) {
          decoder_->Control(AV1_SET_DECODE_TILE_ROW, -1);
          decoder_->Control(AV1_SET_DECODE_TILE_COL, -1);
        } else {
          decoder_->Control(AV1_SET_DECODE_TILE_ROW, r);
          decoder_->Control(AV1_SET_DECODE_TILE_COL, c);
        }

        const aom_codec_err_t res = decoder_->DecodeFrame(
            reinterpret_cast<uint8_t *>(pkt->data.frame.buf),
            pkt->data.frame.sz);
        if (res != AOM_CODEC_OK) {
          abort_ = true;
          ASSERT_EQ(AOM_CODEC_OK, res);
        }
        const aom_image_t *img = decoder_->GetDxData().Next();

        if (!IsLastFrame) {
          if (img) {
            ::libaom_test::MD5 md5_res;
            md5_res.Add(img);
            tile_md5_.push_back(md5_res.Get());
          }
          break;
        }

        const int kMaxMBPlane = 3;
        for (int plane = 0; plane < kMaxMBPlane; ++plane) {
          const int shift = (plane == 0) ? 0 : 1;
          int tile_height = kTIleSizeInPixels >> shift;
          int tile_width = kTIleSizeInPixels >> shift;

          for (int tr = 0; tr < tile_height; ++tr) {
            memcpy(tile_img_.planes[plane] +
                       tile_img_.stride[plane] * (r * tile_height + tr) +
                       c * tile_width,
                   img->planes[plane] + img->stride[plane] * tr, tile_width);
          }
        }
      }

      if (!IsLastFrame) break;
    }

    if (IsLastFrame) {
      ::libaom_test::MD5 md5_res;
      md5_res.Add(&tile_img_);
      tile_md5_.push_back(md5_res.Get());
    }
  }

  void TestRoundTrip() {
    ::libaom_test::I420VideoSource video(
        "hantro_collage_w352h288.yuv", kImgWidth, kImgHeight, 30, 1, 0, kLimit);
    cfg_.rc_target_bitrate = 500;
    cfg_.g_error_resilient = AOM_ERROR_RESILIENT_DEFAULT;
    cfg_.large_scale_tile = 1;
    cfg_.g_lag_in_frames = 0;
    cfg_.g_threads = 1;

    // Tile encoding
    init_flags_ = AOM_CODEC_USE_PSNR;
    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));

    // Compare to check if two vectors are equal.
    ASSERT_EQ(md5_, tile_md5_);
  }

  void TestInvalidTile() {
    ::libaom_test::I420VideoSource video("hantro_collage_w352h288.yuv",
                                         kInvalidTileImgSize,
                                         kInvalidTileImgSize, 30, 1, 0, 1);
    cfg_.rc_target_bitrate = 500;
    cfg_.g_error_resilient = AOM_ERROR_RESILIENT_DEFAULT;
    cfg_.large_scale_tile = 1;
    cfg_.g_lag_in_frames = 0;
    cfg_.g_threads = 1;
    test_invalid_tile_ = true;

    init_flags_ = AOM_CODEC_USE_PSNR;
    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  }

  void TestPartialTileList() {
    ::libaom_test::DummyVideoSource video;
    video.SetSize(kPartialTileImgSize, kPartialTileImgSize);
    video.set_limit(2);
    cfg_.rc_target_bitrate = 500;
    cfg_.g_error_resilient = AOM_ERROR_RESILIENT_DEFAULT;
    cfg_.large_scale_tile = 1;
    cfg_.g_lag_in_frames = 0;
    cfg_.g_threads = 1;
    test_partial_tile_list_ = true;

    init_flags_ = AOM_CODEC_USE_PSNR;
    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  }

  ::libaom_test::TestMode encoding_mode_;
  int set_cpu_used_;
  ::libaom_test::Decoder *decoder_;
  aom_image_t tile_img_;
  aom_image_t ext_ref_img_;
  bool ext_ref_allocated_ = false;
  std::vector<std::string> md5_;
  std::vector<std::string> tile_md5_;
  bool test_invalid_tile_ = false;
  bool test_partial_tile_list_ = false;
};

TEST_P(AV1ExtTileTest, DecoderResultTest) { TestRoundTrip(); }

AV1_INSTANTIATE_TEST_SUITE(
    // Now only test 2-pass mode.
    AV1ExtTileTest, ::testing::Values(::libaom_test::kTwoPassGood),
    ::testing::Range(1, 4));

class AV1ExtTileTestLarge : public AV1ExtTileTest {};

TEST_P(AV1ExtTileTestLarge, DecoderResultTest) { TestRoundTrip(); }

TEST_P(AV1ExtTileTestLarge, RejectsInvalidDecodeTile) { TestInvalidTile(); }

TEST_P(AV1ExtTileTestLarge, PartialTileList) { TestPartialTileList(); }

AV1_INSTANTIATE_TEST_SUITE(
    // Now only test 2-pass mode.
    AV1ExtTileTestLarge, ::testing::Values(::libaom_test::kTwoPassGood),
    ::testing::Range(0, 1));
}  // namespace
