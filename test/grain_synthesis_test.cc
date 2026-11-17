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

#include <climits>
#include <cstring>
#include <ostream>
#include <string>

#include "gtest/gtest.h"

#include "aom/aom_image.h"
#include "aom_dsp/grain_params.h"
#include "av1/decoder/grain_synthesis.h"

namespace {

// Film grain parameters that are valid for every image format used below.
aom_film_grain_t ValidParams() {
  aom_film_grain_t params = {};
  params.apply_grain = 1;
  params.update_parameters = 1;
  params.num_y_points = 2;
  params.scaling_points_y[0][0] = 0;
  params.scaling_points_y[0][1] = 96;
  params.scaling_points_y[1][0] = 255;
  params.scaling_points_y[1][1] = 96;
  params.num_cb_points = 2;
  params.scaling_points_cb[0][0] = 0;
  params.scaling_points_cb[0][1] = 64;
  params.scaling_points_cb[1][0] = 255;
  params.scaling_points_cb[1][1] = 64;
  params.num_cr_points = 2;
  params.scaling_points_cr[0][0] = 0;
  params.scaling_points_cr[0][1] = 64;
  params.scaling_points_cr[1][0] = 255;
  params.scaling_points_cr[1][1] = 64;
  params.scaling_shift = 11;
  params.ar_coeff_lag = 3;
  for (int i = 0; i < 24; ++i) params.ar_coeffs_y[i] = (i & 1) ? 4 : -4;
  for (int i = 0; i < 25; ++i) {
    params.ar_coeffs_cb[i] = (i & 1) ? -3 : 3;
    params.ar_coeffs_cr[i] = (i & 1) ? 2 : -2;
  }
  params.ar_coeff_shift = 7;
  params.cb_mult = 128;
  params.cb_luma_mult = 192;
  params.cb_offset = 256;
  params.cr_mult = 128;
  params.cr_luma_mult = 192;
  params.cr_offset = 256;
  params.overlap_flag = 1;
  params.bit_depth = 8;
  params.random_seed = 7391;
  return params;
}

// Calls av1_add_film_grain() on a 64x64 image of the given format.
int AddGrain(const aom_film_grain_t &params, aom_img_fmt_t fmt,
             unsigned int bit_depth) {
  constexpr int kSize = 64;
  aom_image_t *src = aom_img_alloc(nullptr, fmt, kSize, kSize, 32);
  aom_image_t *dst = aom_img_alloc(nullptr, fmt, kSize, kSize, 32);
  EXPECT_NE(src, nullptr);
  EXPECT_NE(dst, nullptr);
  if (src == nullptr || dst == nullptr) {
    aom_img_free(src);
    aom_img_free(dst);
    return INT_MIN;
  }
  const int bytes = (fmt & AOM_IMG_FMT_HIGHBITDEPTH) ? 2 : 1;
  // Keep the samples within the range of the smaller of the image bit depth
  // and 12 bits.
  const int sample_mask =
      bytes == 2 ? (1 << (bit_depth < 12 ? bit_depth : 12)) - 1 : 0xff;
  for (int plane = 0; plane < 3; ++plane) {
    const int w = aom_img_plane_width(src, plane);
    const int h = aom_img_plane_height(src, plane);
    for (int y = 0; y < h; ++y) {
      uint8_t *row = src->planes[plane] + y * src->stride[plane];
      if (bytes == 2) {
        uint16_t *row16 = reinterpret_cast<uint16_t *>(row);
        for (int x = 0; x < w; ++x) row16[x] = (x * 37 + y * 11) & sample_mask;
      } else {
        for (int x = 0; x < w; ++x) row[x] = (x * 7 + y * 3) & sample_mask;
      }
    }
  }
  src->bit_depth = bit_depth;
  src->mc = AOM_CICP_MC_BT_709;
  const int ret = av1_add_film_grain(&params, src, dst);
  aom_img_free(src);
  aom_img_free(dst);
  return ret;
}

TEST(GrainSynthesisParamsTest, AcceptsValidParams) {
  const aom_film_grain_t params = ValidParams();
  EXPECT_EQ(AddGrain(params, AOM_IMG_FMT_I420, 8), 0);
  EXPECT_EQ(AddGrain(params, AOM_IMG_FMT_I422, 8), 0);
  EXPECT_EQ(AddGrain(params, AOM_IMG_FMT_I444, 8), 0);
  aom_film_grain_t params_hbd = params;
  params_hbd.bit_depth = 10;
  EXPECT_EQ(AddGrain(params_hbd, AOM_IMG_FMT_I42016, 10), 0);
  params_hbd.bit_depth = 12;
  EXPECT_EQ(AddGrain(params_hbd, AOM_IMG_FMT_I44416, 12), 0);
  // 8-bit content stored in a high bit depth buffer.
  EXPECT_EQ(AddGrain(params, AOM_IMG_FMT_I42016, 8), 0);
}

TEST(GrainSynthesisParamsTest, AcceptsBoundaryValues) {
  aom_film_grain_t params = ValidParams();
  params.num_y_points = 14;
  for (int i = 0; i < 14; ++i) {
    params.scaling_points_y[i][0] = (i == 13) ? 255 : i * 18;
    params.scaling_points_y[i][1] = (i & 1) ? 255 : 0;
  }
  params.num_cb_points = 10;
  params.num_cr_points = 10;
  for (int i = 0; i < 10; ++i) {
    params.scaling_points_cb[i][0] = params.scaling_points_cr[i][0] = i * 28;
    params.scaling_points_cb[i][1] = params.scaling_points_cr[i][1] = 255 - i;
  }
  for (int i = 0; i < 24; ++i) params.ar_coeffs_y[i] = (i & 1) ? 127 : -128;
  for (int i = 0; i < 25; ++i) {
    params.ar_coeffs_cb[i] = (i & 1) ? -128 : 127;
    params.ar_coeffs_cr[i] = (i & 1) ? 127 : -128;
  }
  params.scaling_shift = 8;
  params.ar_coeff_shift = 6;
  params.grain_scale_shift = 3;
  params.cb_mult = params.cr_luma_mult = 255;
  params.cb_luma_mult = params.cr_mult = 0;
  params.cb_offset = 511;
  params.cr_offset = 0;
  params.clip_to_restricted_range = 1;
  EXPECT_EQ(AddGrain(params, AOM_IMG_FMT_I420, 8), 0);
  params.bit_depth = 12;
  EXPECT_EQ(AddGrain(params, AOM_IMG_FMT_I42216, 12), 0);
}

struct InvalidParamsCase {
  const char *name;
  void (*mutate)(aom_film_grain_t *params);
  aom_img_fmt_t fmt;
  unsigned int image_bit_depth;
};

void PrintTo(const InvalidParamsCase &c, std::ostream *os) { *os << c.name; }

// Each case corresponds to one of the crash inputs attached to b/527353135
// (fuzz_film_grain.c), or to a parameter combination that cannot be signaled
// in an AV1 bitstream.
const InvalidParamsCase kInvalidParamsCases[] = {
  // id:000000: negative num_y_points, OOB read in init_scaling_function().
  { "NegativeNumYPoints", [](aom_film_grain_t *p) { p->num_y_points = -1; },
    AOM_IMG_FMT_I420, 8 },
  { "TooManyYPoints", [](aom_film_grain_t *p) { p->num_y_points = 15; },
    AOM_IMG_FMT_I420, 8 },
  { "NegativeNumCrPoints", [](aom_film_grain_t *p) { p->num_cr_points = -1; },
    AOM_IMG_FMT_I444, 8 },
  { "TooManyCbPoints", [](aom_film_grain_t *p) { p->num_cb_points = 11; },
    AOM_IMG_FMT_I444, 8 },
  // id:000001..000003: repeated x coordinate, division by zero.
  { "RepeatedYPointX",
    [](aom_film_grain_t *p) { p->scaling_points_y[1][0] = 0; },
    AOM_IMG_FMT_I420, 8 },
  { "RepeatedCbPointX",
    [](aom_film_grain_t *p) { p->scaling_points_cb[1][0] = 0; },
    AOM_IMG_FMT_I420, 8 },
  { "RepeatedCrPointX",
    [](aom_film_grain_t *p) { p->scaling_points_cr[1][0] = 0; },
    AOM_IMG_FMT_I420, 8 },
  { "DecreasingYPointX",
    [](aom_film_grain_t *p) {
      p->num_y_points = 3;
      p->scaling_points_y[2][0] = 100;
      p->scaling_points_y[2][1] = 10;
    },
    AOM_IMG_FMT_I420, 8 },
  // id:000010, id:000037: x coordinate out of range, OOB write.
  { "NegativeCbPointX",
    [](aom_film_grain_t *p) { p->scaling_points_cb[1][0] = -16842752; },
    AOM_IMG_FMT_I420, 8 },
  { "LargeYPointX",
    [](aom_film_grain_t *p) { p->scaling_points_y[1][0] = 65535; },
    AOM_IMG_FMT_I420, 8 },
  // id:000037: y coordinate out of range, signed overflow.
  { "YPointYOutOfRange",
    [](aom_film_grain_t *p) { p->scaling_points_y[0][1] = INT_MIN + 50; },
    AOM_IMG_FMT_I420, 8 },
  { "CrPointYOutOfRange",
    [](aom_film_grain_t *p) { p->scaling_points_cr[1][1] = 256; },
    AOM_IMG_FMT_I420, 8 },
  // AR coefficients out of range, signed overflow (id:000033 for chroma).
  { "ArCoeffYOutOfRange",
    [](aom_film_grain_t *p) { p->ar_coeffs_y[5] = 1 << 24; }, AOM_IMG_FMT_I420,
    8 },
  { "ArCoeffCrOutOfRange",
    [](aom_film_grain_t *p) { p->ar_coeffs_cr[12] = -50331648; },
    AOM_IMG_FMT_I420, 8 },
  { "ArCoeffCbOutOfRange",
    [](aom_film_grain_t *p) { p->ar_coeffs_cb[24] = 128; }, AOM_IMG_FMT_I420,
    8 },
  // id:000036, id:000040: negative ar_coeff_lag, uninitialized pred_pos and
  // signed overflow of the number of AR positions.
  { "NegativeArCoeffLag", [](aom_film_grain_t *p) { p->ar_coeff_lag = -4; },
    AOM_IMG_FMT_I420, 8 },
  { "HugeNegativeArCoeffLag",
    [](aom_film_grain_t *p) { p->ar_coeff_lag = -2147483420; },
    AOM_IMG_FMT_I420, 8 },
  { "ArCoeffLagTooLarge", [](aom_film_grain_t *p) { p->ar_coeff_lag = 4; },
    AOM_IMG_FMT_I420, 8 },
  // id:000020: negative grain_scale_shift, negative shift exponent.
  { "NegativeGrainScaleShift",
    [](aom_film_grain_t *p) { p->grain_scale_shift = -486539264; },
    AOM_IMG_FMT_I420, 8 },
  { "ScalingShiftTooSmall", [](aom_film_grain_t *p) { p->scaling_shift = 7; },
    AOM_IMG_FMT_I420, 8 },
  { "ArCoeffShiftTooLarge", [](aom_film_grain_t *p) { p->ar_coeff_shift = 10; },
    AOM_IMG_FMT_I420, 8 },
  { "CbMultOutOfRange", [](aom_film_grain_t *p) { p->cb_mult = 1 << 30; },
    AOM_IMG_FMT_I444, 8 },
  { "CrOffsetOutOfRange", [](aom_film_grain_t *p) { p->cr_offset = 512; },
    AOM_IMG_FMT_I444, 8 },
  // bit_depth > 8 with an 8-bit image: OOB read of the scaling LUT.
  { "HighBitDepthParamsOn8BitImage",
    [](aom_film_grain_t *p) {
      p->bit_depth = 10;
      p->cr_offset = 511;
    },
    AOM_IMG_FMT_I420, 10 },
  // aom_img_alloc() sets bit_depth to 16 for high bit depth formats
  // (examples/noise_model.c --bit-depth=10 --debug-file).
  { "BitDepth16", [](aom_film_grain_t *p) { p->bit_depth = 16; },
    AOM_IMG_FMT_I42016, 16 },
  { "BitDepthMismatch", [](aom_film_grain_t *p) { p->bit_depth = 12; },
    AOM_IMG_FMT_I42016, 10 },
  // In 4:2:0, grain must be applied to both chroma planes or neither.
  { "Only420CbPoints", [](aom_film_grain_t *p) { p->num_cr_points = 0; },
    AOM_IMG_FMT_I420, 8 },
  { "ChromaScalingFromLumaWithChromaPoints",
    [](aom_film_grain_t *p) { p->chroma_scaling_from_luma = 1; },
    AOM_IMG_FMT_I444, 8 },
};

class GrainSynthesisInvalidParamsTest
    : public ::testing::TestWithParam<InvalidParamsCase> {};

TEST_P(GrainSynthesisInvalidParamsTest, Rejects) {
  const InvalidParamsCase &c = GetParam();
  aom_film_grain_t params = ValidParams();
  c.mutate(&params);
  EXPECT_EQ(AddGrain(params, c.fmt, c.image_bit_depth), -1);
}

INSTANTIATE_TEST_SUITE_P(
    All, GrainSynthesisInvalidParamsTest,
    ::testing::ValuesIn(kInvalidParamsCases),
    [](const ::testing::TestParamInfo<InvalidParamsCase> &param_info) {
      return std::string(param_info.param.name);
    });

}  // namespace
