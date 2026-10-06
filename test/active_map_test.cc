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
#include <climits>
#include <cstring>
#include <vector>
#include "gtest/gtest.h"
#include "test/acm_random.h"
#include "test/codec_factory.h"
#include "test/encode_test_driver.h"
#include "test/i420_video_source.h"
#include "test/util.h"
#include "test/video_source.h"

namespace {

// Params: test mode, speed, aq_mode and screen_content mode.
class ActiveMapTest
    : public ::libaom_test::CodecTestWith4Params<libaom_test::TestMode, int,
                                                 int, int>,
      public ::libaom_test::EncoderTest {
 protected:
  static const int kWidth = 208;
  static const int kHeight = 144;

  ActiveMapTest() : EncoderTest(GET_PARAM(0)) {}
  ~ActiveMapTest() override = default;

  void SetUp() override {
    InitializeConfig(GET_PARAM(1));
    cpu_used_ = GET_PARAM(2);
    aq_mode_ = GET_PARAM(3);
    screen_mode_ = GET_PARAM(4);
  }

  void PreEncodeFrameHook(::libaom_test::VideoSource *video,
                          ::libaom_test::Encoder *encoder) override {
    if (video->frame() == 0) {
      encoder->Control(AOME_SET_CPUUSED, cpu_used_);
      encoder->Control(AV1E_SET_ALLOW_WARPED_MOTION, 0);
      encoder->Control(AV1E_SET_ENABLE_GLOBAL_MOTION, 0);
      encoder->Control(AV1E_SET_ENABLE_OBMC, 0);
      encoder->Control(AV1E_SET_AQ_MODE, aq_mode_);
      encoder->Control(AV1E_SET_TUNE_CONTENT, screen_mode_);
      if (screen_mode_) encoder->Control(AV1E_SET_ENABLE_PALETTE, 1);
    } else if (video->frame() == 3) {
      aom_active_map_t map = aom_active_map_t();
      /* clang-format off */
      uint8_t active_map[9 * 13] = {
        1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1,
        0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1,
        0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1,
        0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1,
        1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0,
      };
      /* clang-format on */
      map.cols = (kWidth + 15) / 16;
      map.rows = (kHeight + 15) / 16;
      ASSERT_EQ(map.cols, 13u);
      ASSERT_EQ(map.rows, 9u);
      map.active_map = active_map;
      encoder->Control(AOME_SET_ACTIVEMAP, &map);
    } else if (video->frame() == 15) {
      aom_active_map_t map = aom_active_map_t();
      map.cols = (kWidth + 15) / 16;
      map.rows = (kHeight + 15) / 16;
      map.active_map = nullptr;
      encoder->Control(AOME_SET_ACTIVEMAP, &map);
    }
  }

  void DoTest() {
    // Validate that this non multiple of 64 wide clip encodes
    cfg_.g_lag_in_frames = 0;
    cfg_.rc_target_bitrate = 400;
    cfg_.rc_resize_mode = 0;
    cfg_.g_pass = AOM_RC_ONE_PASS;
    cfg_.rc_end_usage = AOM_CBR;
    cfg_.kf_max_dist = 90000;
    ::libaom_test::I420VideoSource video("hantro_odd.yuv", kWidth, kHeight, 100,
                                         1, 0, 100);

    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  }

  int cpu_used_;
  int aq_mode_;
  int screen_mode_;
};

TEST_P(ActiveMapTest, Test) { DoTest(); }

AV1_INSTANTIATE_TEST_SUITE(ActiveMapTest,
                           ::testing::Values(::libaom_test::kRealTime),
                           ::testing::Range(5, 12), ::testing::Values(0, 3),
                           ::testing::Values(0, 1));

constexpr int kSrcWidth = 384;
constexpr int kSrcHeight = 256;
constexpr int kMapCols = kSrcWidth / 16;
constexpr int kMapRows = kSrcHeight / 16;
constexpr int kWinSize = 128;
constexpr int kDocHeight = 1024;
constexpr int kNumFrames = 60;
constexpr int kEdgeMargin = 8;

void WindowPos(int frame, int *x, int *y) {
  static const int kPos[4][2] = {
    { 64, 64 }, { 192, 0 }, { 0, 128 }, { 256, 128 }
  };
  const int i = (frame / 10) % 4;
  *x = kPos[i][0];
  *y = kPos[i][1];
}

int ScrollPos(int frame) {
  static const int kSteps[4] = { 1, 2, 4, 8 };
  int pos = 0;
  for (int f = 1; f <= frame; ++f) pos += kSteps[(f / 7) % 4];
  return pos % (kDocHeight - kWinSize);
}

void BuildActiveMap(int frame, uint8_t *map) {
  memset(map, 0, kMapRows * kMapCols);
  for (int f = frame - 1; f <= frame; ++f) {
    int win_x;
    int win_y;
    WindowPos(f, &win_x, &win_y);
    for (int r = win_y / 16; r < (win_y + kWinSize) / 16; ++r) {
      for (int c = win_x / 16; c < (win_x + kWinSize) / 16; ++c) {
        map[r * kMapCols + c] = 1;
      }
    }
  }
}

class ScrollingWindowSource : public ::libaom_test::DummyVideoSource {
 public:
  ScrollingWindowSource() : doc_(kWinSize * kDocHeight) {
    SetSize(kSrcWidth, kSrcHeight);
    set_limit(kNumFrames);
    ::libaom_test::ACMRandom rnd(0x5eed);
    for (int y = 0; y < kDocHeight; ++y) {
      for (int x = 0; x < kWinSize; ++x) {
        const bool glyph_row = (y % 16) < 12;
        doc_[y * kWinSize + x] = (glyph_row && rnd.Rand8() < 77) ? 20 : 235;
      }
    }
  }

 protected:
  void FillFrame() override {
    const int frame = static_cast<int>(frame_);
    uint8_t *const y_plane = img_->planes[AOM_PLANE_Y];
    const int y_stride = img_->stride[AOM_PLANE_Y];
    for (int y = 0; y < kSrcHeight; ++y) {
      for (int x = 0; x < kSrcWidth; ++x) {
        y_plane[y * y_stride + x] =
            static_cast<uint8_t>(40 + x * 150 / kSrcWidth + (y * 37) % 23);
      }
    }
    for (int plane = AOM_PLANE_U; plane <= AOM_PLANE_V; ++plane) {
      for (int y = 0; y < kSrcHeight / 2; ++y) {
        memset(img_->planes[plane] + y * img_->stride[plane],
               plane == AOM_PLANE_U ? 110 : 140, kSrcWidth / 2);
      }
    }

    int win_x;
    int win_y;
    WindowPos(frame, &win_x, &win_y);
    const int scroll = ScrollPos(frame);
    for (int y = 0; y < kWinSize; ++y) {
      memcpy(y_plane + (win_y + y) * y_stride + win_x,
             &doc_[(scroll + y) * kWinSize], kWinSize);
    }

    if (frame == 0) return;

    uint8_t map[kMapRows * kMapCols];
    BuildActiveMap(frame, map);
    ::libaom_test::ACMRandom rnd(frame);
    for (int y = 0; y < kSrcHeight; ++y) {
      for (int x = 0; x < kSrcWidth; ++x) {
        if (map[(y / 16) * kMapCols + x / 16]) continue;
        if (rnd.Rand8() < 16) y_plane[y * y_stride + x] ^= 0x10;
      }
    }
  }

 private:
  std::vector<uint8_t> doc_;
};

int GetPixel(const aom_image_t &img, int plane, int x, int y) {
  const uint8_t *row = img.planes[plane] + y * img.stride[plane];
  if (img.fmt & AOM_IMG_FMT_HIGHBITDEPTH) {
    return reinterpret_cast<const uint16_t *>(row)[x];
  }
  return row[x];
}

// Params: test mode, speed, aq_mode and number of threads.
class InactiveAreaTest
    : public ::libaom_test::CodecTestWith4Params<libaom_test::TestMode, int,
                                                 int, int>,
      public ::libaom_test::EncoderTest {
 protected:
  InactiveAreaTest() : EncoderTest(GET_PARAM(0)) {}
  ~InactiveAreaTest() override = default;

  void SetUp() override {
    InitializeConfig(GET_PARAM(1));
    cpu_used_ = GET_PARAM(2);
    aq_mode_ = GET_PARAM(3);
    cfg_.g_threads = GET_PARAM(4);
    cfg_.g_lag_in_frames = 0;
    cfg_.rc_target_bitrate = 300;
    cfg_.rc_end_usage = AOM_CBR;
    cfg_.g_pass = AOM_RC_ONE_PASS;
    cfg_.kf_max_dist = 90000;
  }

  void PreEncodeFrameHook(::libaom_test::VideoSource *video,
                          ::libaom_test::Encoder *encoder) override {
    const int frame = static_cast<int>(video->frame());
    if (frame == 0) {
      encoder->Control(AOME_SET_CPUUSED, cpu_used_);
      encoder->Control(AV1E_SET_AQ_MODE, aq_mode_);
      encoder->Control(AV1E_SET_TUNE_CONTENT, AOM_CONTENT_SCREEN);
      encoder->Control(AV1E_SET_ROW_MT, cfg_.g_threads > 1);
      return;
    }

    uint8_t map[kMapRows * kMapCols];
    BuildActiveMap(frame, map);
    if (use_roi_) {
      const int mi_cols = kSrcWidth / 4;
      const int mi_rows = kSrcHeight / 4;
      std::vector<uint8_t> roi_map(mi_rows * mi_cols);
      for (int r = 0; r < mi_rows; ++r) {
        for (int c = 0; c < mi_cols; ++c) {
          roi_map[r * mi_cols + c] = map[(r / 4) * kMapCols + c / 4] ? 0 : 3;
        }
      }
      aom_roi_map_t roi = {};
      roi.enabled = 1;
      roi.roi_map = roi_map.data();
      roi.rows = mi_rows;
      roi.cols = mi_cols;
      for (int i = 0; i < AOM_MAX_SEGMENTS; ++i) roi.ref_frame[i] = -1;
      roi.skip[3] = 1;
      encoder->Control(AOME_SET_ROI_MAP, &roi);
    } else {
      aom_active_map_t active_map = {};
      active_map.rows = kMapRows;
      active_map.cols = kMapCols;
      active_map.active_map = map;
      encoder->Control(AOME_SET_ACTIVEMAP, &active_map);
    }
  }

  void DecompressedFrameHook(const aom_image_t &img,
                             aom_codec_pts_t pts) override {
    const int frame = static_cast<int>(pts);
    if (frame > 0) {
      uint8_t map[kMapRows * kMapCols];
      BuildActiveMap(frame, map);
      std::vector<uint8_t> near_active(kSrcWidth * kSrcHeight, 0);
      for (int r = 0; r < kMapRows; ++r) {
        for (int c = 0; c < kMapCols; ++c) {
          if (!map[r * kMapCols + c]) continue;
          const int y0 = std::max(r * 16 - kEdgeMargin, 0);
          const int y1 = std::min((r + 1) * 16 + kEdgeMargin, kSrcHeight);
          const int x0 = std::max(c * 16 - kEdgeMargin, 0);
          const int x1 = std::min((c + 1) * 16 + kEdgeMargin, kSrcWidth);
          for (int y = y0; y < y1; ++y) {
            memset(&near_active[y * kSrcWidth + x0], 1, x1 - x0);
          }
        }
      }
      int changed = 0;
      for (int plane = 0; plane < 3; ++plane) {
        const int shift = plane ? 1 : 0;
        const int width = kSrcWidth >> shift;
        const int height = kSrcHeight >> shift;
        for (int y = 0; y < height; ++y) {
          for (int x = 0; x < width; ++x) {
            if (near_active[(y << shift) * kSrcWidth + (x << shift)]) continue;
            changed +=
                GetPixel(img, plane, x, y) != prev_[plane][y * width + x];
          }
        }
      }
      EXPECT_EQ(changed, 0) << "Inactive pixels changed in frame " << frame;
    }

    for (int plane = 0; plane < 3; ++plane) {
      const int shift = plane ? 1 : 0;
      const int width = kSrcWidth >> shift;
      const int height = kSrcHeight >> shift;
      prev_[plane].resize(width * height);
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          prev_[plane][y * width + x] =
              static_cast<uint8_t>(GetPixel(img, plane, x, y));
        }
      }
    }
  }

  void DoTest(bool use_roi) {
    use_roi_ = use_roi;
    ScrollingWindowSource video;
    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  }

  int cpu_used_;
  int aq_mode_;
  bool use_roi_ = false;
  std::vector<uint8_t> prev_[3];
};

TEST_P(InactiveAreaTest, ActiveMap) { DoTest(false); }

TEST_P(InactiveAreaTest, RoiSkip) { DoTest(true); }

AV1_INSTANTIATE_TEST_SUITE(InactiveAreaTest,
                           ::testing::Values(::libaom_test::kRealTime),
                           ::testing::Range(7, 12), ::testing::Values(0, 3),
                           ::testing::Values(1, 4));

}  // namespace
