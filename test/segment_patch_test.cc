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

#include "av1/encoder/segment_patch.h"
#include "third_party/googletest/src/googletest/include/gtest/gtest.h"
#include "test/codec_factory.h"
#include "test/i420_video_source.h"

namespace {

const int kNumFrames = 10;

}  // namespace

// Test av1_get_segments() API.
TEST(SegmentPatchTest, get_segments) {
  libaom_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352, 288, 1,
                                     30, 0, kNumFrames);
  Av1SegmentParams params;
  av1_get_default_segment_params(&params);
  std::unique_ptr<uint8_t> output;

  ASSERT_NO_FATAL_FAILURE(video.Begin());
  for (int i = 0; i < kNumFrames; ++i) {
    const aom_image_t *frame = video.img();
    ASSERT_TRUE(frame != nullptr)
        << "Could not read frame# " << i << " from source video";
    if (i == 0) {
      // Allocate output buffer.
      output.reset(new uint8_t[frame->w * frame->h]);
    }
    int num_components = -1;
    av1_get_segments(frame->planes[0], frame->w, frame->h, frame->stride[0],
                     &params, output.get(), &num_components);
    ASSERT_GT(num_components, 0);
    printf("Segmented frame# %d: num_components = %d\n", i, num_components);
    video.Next();
  }
}
