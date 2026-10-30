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

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"

#include "config/aom_config.h"
#include "config/av1_rtcd.h"

#include "aom_mem/aom_mem.h"
#include "av1/encoder/block.h"
#include "test/acm_random.h"
#include "test/register_state_check.h"
#include "test/util.h"

namespace {

using libaom_test::ACMRandom;

constexpr int kHogBins = 32;
constexpr int kFixPrecBits = 16;
constexpr float kHogTol = 1e-6f;

constexpr int32_t kHogThresholds[kHogBins] = {
  -1334015, -441798, -261605, -183158, -138560, -109331, -88359, -72303,
  -59392,   -48579,  -39272,  -30982,  -23445,  -16400,  -9715,  -3194,
  3227,     9748,    16433,   23478,   31015,   39305,   48611,  59425,
  72336,    88392,   109364,  138593,  183191,  261638,  441831, INT32_MAX
};

int RefGetHistBinIdx(int dx, int dy) {
  if (dy == 0) return 16;
  const int32_t ratio = (dy * (1 << kFixPrecBits)) / dx;
  for (int idx = 0; idx < kHogBins - 1; ++idx) {
    if (ratio <= kHogThresholds[idx]) return idx;
  }
  return kHogBins - 1;
}

template <typename Pixel>
void RefGenerateHog(const Pixel *src, int stride, int rows, int cols,
                    float *hist) {
  static constexpr int kSobelX[3][3] = { { -1, 0, 1 },
                                         { -2, 0, 2 },
                                         { -1, 0, 1 } };
  static constexpr int kSobelY[3][3] = { { -1, -2, -1 },
                                         { 0, 0, 0 },
                                         { 1, 2, 1 } };
  float total = 0.1f;
  for (int r = 1; r < rows - 1; ++r) {
    for (int c = 1; c < cols - 1; ++c) {
      int dx = 0;
      int dy = 0;
      for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
          const int val = src[(r + y) * stride + (c + x)];
          dx += kSobelX[y + 1][x + 1] * val;
          dy += kSobelY[y + 1][x + 1] * val;
        }
      }
      if (dx == 0 && dy == 0) continue;
      const int temp = std::abs(dx) + std::abs(dy);
      total += temp;
      if (dx == 0) {
        hist[0] += temp / 2;
        hist[kHogBins - 1] += temp / 2;
      } else {
        hist[RefGetHistBinIdx(dx, dy)] += temp;
      }
    }
  }
  for (int i = 0; i < kHogBins; ++i) hist[i] /= total;
}

enum class Pattern {
  kRandom = 0,
  kUniform,         // dx == 0 && dy == 0
  kHorizontalEdge,  // dx == 0, dy != 0
  kVerticalEdge,    // dx != 0, dy == 0
  kMaxChecker,      // alternating 0 and max_val
  kMaxDiagStep,     // maximum simultaneous |dx| and |dy|
  kGradientRamp,    // sweeps many dy/dx ratios across threshold boundaries
};

template <typename Pixel>
void FillPattern(Pixel *buf, int stride, int rows, int cols, int bd,
                 Pattern pat, ACMRandom *rnd) {
  const int max_val = (1 << bd) - 1;
  // Fill full stride including padding columns with sentinel noise to catch any
  // out-of-bounds column reads (such as cols < 3 priming).
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < stride; ++c) {
      buf[r * stride + c] = static_cast<Pixel>(rnd->Rand16() & max_val);
    }
  }
  const int const_val = rnd->Rand16() & max_val;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      int v = 0;
      switch (pat) {
        case Pattern::kRandom: v = rnd->Rand16() & max_val; break;
        case Pattern::kUniform: v = const_val; break;
        case Pattern::kHorizontalEdge: v = (r & 2) ? max_val : 0; break;
        case Pattern::kVerticalEdge: v = (c & 2) ? max_val : 0; break;
        case Pattern::kMaxChecker: v = ((r ^ c) & 1) ? max_val : 0; break;
        case Pattern::kMaxDiagStep: v = (r + c >= 3) ? max_val : 0; break;
        case Pattern::kGradientRamp:
          v = ((r * 37 + c * 13 + (r * c) * 3) * (1 << (bd - 8))) & max_val;
          break;
      }
      buf[r * stride + c] = static_cast<Pixel>(v);
    }
  }
}

void ExpectHogNear(const float *ref, const float *tst, int rows, int cols,
                   int stride, int bd) {
  for (int b = 0; b < kHogBins; ++b) {
    EXPECT_NEAR(ref[b], tst[b], kHogTol)
        << "bin=" << b << " rows=" << rows << " cols=" << cols
        << " stride=" << stride << " bd=" << bd;
  }
}

void ExpectGradInfoEq(const PixelLevelGradientInfo *ref,
                      const PixelLevelGradientInfo *tst, size_t count,
                      BLOCK_SIZE sb_size, PLANE_TYPE plane, int ss_x, int ss_y,
                      int stride, int bd) {
  for (size_t i = 0; i < count; ++i) {
    ASSERT_EQ(ref[i].abs_dx_abs_dy_sum, tst[i].abs_dx_abs_dy_sum)
        << "idx=" << i << " sb_size=" << sb_size << " plane=" << plane
        << " ss=(" << ss_x << "," << ss_y << ") stride=" << stride
        << " bd=" << bd;
    ASSERT_EQ(ref[i].hist_bin_idx, tst[i].hist_bin_idx)
        << "idx=" << i << " sb_size=" << sb_size << " plane=" << plane;
    ASSERT_EQ(ref[i].is_dx_zero, tst[i].is_dx_zero)
        << "idx=" << i << " sb_size=" << sb_size << " plane=" << plane;
  }
}

// Exercises the C Sobel sliding window and branchless binary search directly
// against an independent 3x3 Sobel + linear-scan oracle on all platforms,
// including cols == 1 with stride == 1 under ASan to verify no OOB column load.
TEST(IntraHogRefTest, LowbdGenerateHogMatchesTextbookOracle) {
  ACMRandom rnd(ACMRandom::DeterministicSeed());
  constexpr int kDims[] = { 1, 2, 3, 4, 5, 7, 8, 16, 33 };
  constexpr Pattern kPatterns[] = {
    Pattern::kRandom,       Pattern::kUniform,    Pattern::kHorizontalEdge,
    Pattern::kVerticalEdge, Pattern::kMaxChecker, Pattern::kMaxDiagStep,
    Pattern::kGradientRamp,
  };

  for (int rows : kDims) {
    for (int cols : kDims) {
      for (int pad : { 0, 1, 13 }) {
        const int stride = cols + pad;
        // Exact-sized heap buffer so ASan catches any OOB read when cols < 3.
        std::vector<uint8_t> buf(static_cast<size_t>(rows * stride));
        for (Pattern pat : kPatterns) {
          FillPattern(buf.data(), stride, rows, cols, 8, pat, &rnd);
          float hist_ref[kHogBins] = { 0.0f };
          float hist_c[kHogBins] = { 0.0f };
          if (rows < 3 || cols < 3) {
            // Also verify non-zero initial bins are normalized identically.
            hist_ref[0] = hist_c[0] = 1.0f;
          }
          RefGenerateHog(buf.data(), stride, rows, cols, hist_ref);
          av1_lowbd_generate_hog_c(buf.data(), stride, rows, cols, hist_c);
          ExpectHogNear(hist_ref, hist_c, rows, cols, stride, 8);
        }
      }
    }
  }

  // Probe random 3x3 patches to cover diverse signed dy/dx ratios.
  uint8_t patch[9];
  for (int iter = 0; iter < 1000; ++iter) {
    for (int i = 0; i < 9; ++i) patch[i] = rnd.Rand8();
    float hist_ref[kHogBins] = { 0.0f };
    float hist_c[kHogBins] = { 0.0f };
    RefGenerateHog(patch, 3, 3, 3, hist_ref);
    av1_lowbd_generate_hog_c(patch, 3, 3, 3, hist_c);
    ExpectHogNear(hist_ref, hist_c, 3, 3, 3, 8);
  }
}

#if CONFIG_AV1_HIGHBITDEPTH
TEST(IntraHogRefTest, HighbdGenerateHogMatchesTextbookOracle) {
  ACMRandom rnd(ACMRandom::DeterministicSeed());
  constexpr int kDims[] = { 1, 2, 3, 4, 5, 8, 17 };
  constexpr Pattern kPatterns[] = {
    Pattern::kRandom,       Pattern::kUniform,    Pattern::kHorizontalEdge,
    Pattern::kVerticalEdge, Pattern::kMaxChecker, Pattern::kMaxDiagStep,
    Pattern::kGradientRamp,
  };

  for (int bd : { 8, 10, 12 }) {
    for (int rows : kDims) {
      for (int cols : kDims) {
        for (int pad : { 0, 7 }) {
          const int stride = cols + pad;
          std::vector<uint16_t> buf(static_cast<size_t>(rows * stride));
          for (Pattern pat : kPatterns) {
            FillPattern(buf.data(), stride, rows, cols, bd, pat, &rnd);
            float hist_ref[kHogBins] = { 0.0f };
            float hist_c[kHogBins] = { 0.0f };
            RefGenerateHog(buf.data(), stride, rows, cols, hist_ref);
            av1_highbd_generate_hog_c(CONVERT_TO_BYTEPTR(buf.data()), stride,
                                      rows, cols, hist_c);
            ExpectHogNear(hist_ref, hist_c, rows, cols, stride, bd);
          }
        }
      }
    }
  }

  // Explicitly probe every threshold boundary kHogThresholds[0..30] on both C
  // and SSE2 using 12-bit 3x3 patches where dx = +/-128 (via p12=64 or p10=64,
  // which leaves dy=0) and dy sweeps across the exact boundary value (via
  // p21/p01 and p20=p22, keeping dx unchanged when even, or shifting 1 from
  // p12=63 + p02=1 + p8=1 for even/odd dy).
  for (int t = 0; t < kHogBins - 1; ++t) {
    for (int sign_dx : { 1, -1 }) {
      const int dx = sign_dx * 128;
      const int center_dy = static_cast<int>(
          (static_cast<int64_t>(kHogThresholds[t]) * dx) / (1 << kFixPrecBits));
      for (int delta = -2; delta <= 2; ++delta) {
        const int dy = center_dy + delta;
        if (std::abs(dy) > 8190 || (dy & 1) != 0) continue;
        uint16_t patch[9] = { 0 };
        if (dx > 0) {
          patch[5] = 64;  // dx = +128, dy += 0
        } else {
          patch[3] = 64;  // dx = -128, dy += 0
        }
        if (dy >= 0) {
          patch[7] = static_cast<uint16_t>(dy / 2);  // dy = +2 * patch[7]
        } else {
          patch[1] = static_cast<uint16_t>(-dy / 2);  // dy = -2 * patch[1]
        }
        float hist_ref[kHogBins] = { 0.0f };
        float hist_c[kHogBins] = { 0.0f };
        RefGenerateHog(patch, 3, 3, 3, hist_ref);
        av1_highbd_generate_hog_c(CONVERT_TO_BYTEPTR(patch), 3, 3, 3, hist_c);
        ExpectHogNear(hist_ref, hist_c, 3, 3, 3, 12);
#if HAVE_SSE2
        float hist_sse2[kHogBins] = { 0.0f };
        av1_highbd_generate_hog_sse2(CONVERT_TO_BYTEPTR(patch), 3, 3, 3,
                                     hist_sse2);
        ExpectHogNear(hist_ref, hist_sse2, 3, 3, 3, 12);
#endif
      }
    }
  }
}
#endif  // CONFIG_AV1_HIGHBITDEPTH

using GenerateHogFunc = void (*)(const uint8_t *src, int stride, int rows,
                                 int cols, float *hist);
using ComputeGradSbFunc = void (*)(MACROBLOCK *const x, BLOCK_SIZE sb_size,
                                   PLANE_TYPE plane);

class LowbdIntraHogSimdTest
    : public ::testing::TestWithParam<
          std::tuple<GenerateHogFunc, ComputeGradSbFunc>> {};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(LowbdIntraHogSimdTest);

TEST_P(LowbdIntraHogSimdTest, GenerateHogMatchesC) {
  const GenerateHogFunc test_fn = std::get<0>(GetParam());
  ACMRandom rnd(ACMRandom::DeterministicSeed());
  constexpr int kDims[] = { 1, 2, 3, 4, 5, 7, 8, 16, 32, 33, 64 };
  constexpr Pattern kPatterns[] = {
    Pattern::kRandom,       Pattern::kUniform,    Pattern::kHorizontalEdge,
    Pattern::kVerticalEdge, Pattern::kMaxChecker, Pattern::kMaxDiagStep,
    Pattern::kGradientRamp,
  };

  for (int rows : kDims) {
    for (int cols : kDims) {
      for (int pad : { 0, 1, 19 }) {
        const int stride = cols + pad;
        std::vector<uint8_t> buf(static_cast<size_t>(rows * stride));
        for (Pattern pat : kPatterns) {
          FillPattern(buf.data(), stride, rows, cols, 8, pat, &rnd);
          float hist_ref[kHogBins] = { 0.0f };
          float hist_tst[kHogBins] = { 0.0f };
          av1_lowbd_generate_hog_c(buf.data(), stride, rows, cols, hist_ref);
          API_REGISTER_STATE_CHECK(
              test_fn(buf.data(), stride, rows, cols, hist_tst));
          ExpectHogNear(hist_ref, hist_tst, rows, cols, stride, 8);
        }
      }
    }
  }
}

TEST_P(LowbdIntraHogSimdTest, ComputeGradientInfoSbMatchesC) {
  const ComputeGradSbFunc test_fn = std::get<1>(GetParam());
  ACMRandom rnd(ACMRandom::DeterministicSeed());
  constexpr BLOCK_SIZE kSbSizes[] = { BLOCK_4X4,   BLOCK_4X8,   BLOCK_8X4,
                                      BLOCK_16X16, BLOCK_64X64, BLOCK_128X128 };
  constexpr Pattern kPatterns[] = {
    Pattern::kRandom,       Pattern::kUniform,    Pattern::kHorizontalEdge,
    Pattern::kVerticalEdge, Pattern::kMaxChecker, Pattern::kMaxDiagStep,
    Pattern::kGradientRamp,
  };

  const size_t grad_count = PLANE_TYPES * MAX_SB_SQUARE;
  std::vector<PixelLevelGradientInfo> grad_ref(grad_count);
  std::vector<PixelLevelGradientInfo> grad_tst(grad_count);

  for (BLOCK_SIZE sb_size : kSbSizes) {
    for (PLANE_TYPE plane : { PLANE_TYPE_Y, PLANE_TYPE_UV }) {
      for (int ss_x : { 0, 1 }) {
        for (int ss_y : { 0, 1 }) {
          if (plane == PLANE_TYPE_Y && (ss_x != 0 || ss_y != 0)) continue;
          const int sb_h = block_size_high[sb_size] >> ss_y;
          const int sb_w = block_size_wide[sb_size] >> ss_x;
          for (int pad : { 0, 17 }) {
            const int stride = sb_w + pad;
            std::vector<uint8_t> buf(static_cast<size_t>(sb_h * stride));
            for (Pattern pat : kPatterns) {
              FillPattern(buf.data(), stride, sb_h, sb_w, 8, pat, &rnd);
              const PixelLevelGradientInfo kCanary = { 0x5a5a, 0x5a, true };
              std::fill(grad_ref.begin(), grad_ref.end(), kCanary);
              std::fill(grad_tst.begin(), grad_tst.end(), kCanary);

              auto x = std::make_unique<MACROBLOCK>();
              memset(x.get(), 0, sizeof(*x));
              x->plane[plane].src.buf = buf.data();
              x->plane[plane].src.stride = stride;
              x->e_mbd.plane[plane].subsampling_x = ss_x;
              x->e_mbd.plane[plane].subsampling_y = ss_y;

              x->pixel_gradient_info = grad_ref.data();
              av1_lowbd_compute_gradient_info_sb_c(x.get(), sb_size, plane);

              x->pixel_gradient_info = grad_tst.data();
              API_REGISTER_STATE_CHECK(test_fn(x.get(), sb_size, plane));

              ExpectGradInfoEq(grad_ref.data(), grad_tst.data(), grad_count,
                               sb_size, plane, ss_x, ss_y, stride, 8);
            }
          }
        }
      }
    }
  }
}

#if HAVE_SSE2
INSTANTIATE_TEST_SUITE_P(SSE2, LowbdIntraHogSimdTest,
                         ::testing::Values(std::make_tuple(
                             &av1_lowbd_generate_hog_sse2,
                             &av1_lowbd_compute_gradient_info_sb_sse2)));
#endif  // HAVE_SSE2

#if CONFIG_AV1_HIGHBITDEPTH
class HighbdIntraHogSimdTest
    : public ::testing::TestWithParam<
          std::tuple<GenerateHogFunc, ComputeGradSbFunc, int>> {};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(HighbdIntraHogSimdTest);

TEST_P(HighbdIntraHogSimdTest, GenerateHogMatchesC) {
  const GenerateHogFunc test_fn = std::get<0>(GetParam());
  const int bd = std::get<2>(GetParam());
  ACMRandom rnd(ACMRandom::DeterministicSeed());
  constexpr int kDims[] = { 1, 2, 3, 4, 5, 7, 8, 16, 33, 64 };
  constexpr Pattern kPatterns[] = {
    Pattern::kRandom,       Pattern::kUniform,    Pattern::kHorizontalEdge,
    Pattern::kVerticalEdge, Pattern::kMaxChecker, Pattern::kMaxDiagStep,
    Pattern::kGradientRamp,
  };

  for (int rows : kDims) {
    for (int cols : kDims) {
      for (int pad : { 0, 1, 11 }) {
        const int stride = cols + pad;
        std::vector<uint16_t> buf(static_cast<size_t>(rows * stride));
        for (Pattern pat : kPatterns) {
          FillPattern(buf.data(), stride, rows, cols, bd, pat, &rnd);
          float hist_ref[kHogBins] = { 0.0f };
          float hist_tst[kHogBins] = { 0.0f };
          const uint8_t *const src8 = CONVERT_TO_BYTEPTR(buf.data());
          av1_highbd_generate_hog_c(src8, stride, rows, cols, hist_ref);
          API_REGISTER_STATE_CHECK(test_fn(src8, stride, rows, cols, hist_tst));
          ExpectHogNear(hist_ref, hist_tst, rows, cols, stride, bd);
        }
      }
    }
  }
}

TEST_P(HighbdIntraHogSimdTest, ComputeGradientInfoSbMatchesC) {
  const ComputeGradSbFunc test_fn = std::get<1>(GetParam());
  const int bd = std::get<2>(GetParam());
  ACMRandom rnd(ACMRandom::DeterministicSeed());
  constexpr BLOCK_SIZE kSbSizes[] = { BLOCK_4X4,   BLOCK_4X8,   BLOCK_8X4,
                                      BLOCK_16X16, BLOCK_64X64, BLOCK_128X128 };
  constexpr Pattern kPatterns[] = {
    Pattern::kRandom,       Pattern::kUniform,    Pattern::kHorizontalEdge,
    Pattern::kVerticalEdge, Pattern::kMaxChecker, Pattern::kMaxDiagStep,
    Pattern::kGradientRamp,
  };

  const size_t grad_count = PLANE_TYPES * MAX_SB_SQUARE;
  std::vector<PixelLevelGradientInfo> grad_ref(grad_count);
  std::vector<PixelLevelGradientInfo> grad_tst(grad_count);

  for (BLOCK_SIZE sb_size : kSbSizes) {
    for (PLANE_TYPE plane : { PLANE_TYPE_Y, PLANE_TYPE_UV }) {
      for (int ss_x : { 0, 1 }) {
        for (int ss_y : { 0, 1 }) {
          if (plane == PLANE_TYPE_Y && (ss_x != 0 || ss_y != 0)) continue;
          const int sb_h = block_size_high[sb_size] >> ss_y;
          const int sb_w = block_size_wide[sb_size] >> ss_x;
          for (int pad : { 0, 13 }) {
            const int stride = sb_w + pad;
            std::vector<uint16_t> buf(static_cast<size_t>(sb_h * stride));
            for (Pattern pat : kPatterns) {
              FillPattern(buf.data(), stride, sb_h, sb_w, bd, pat, &rnd);
              const PixelLevelGradientInfo kCanary = { 0x5a5a, 0x5a, true };
              std::fill(grad_ref.begin(), grad_ref.end(), kCanary);
              std::fill(grad_tst.begin(), grad_tst.end(), kCanary);

              auto x = std::make_unique<MACROBLOCK>();
              memset(x.get(), 0, sizeof(*x));
              x->plane[plane].src.buf = CONVERT_TO_BYTEPTR(buf.data());
              x->plane[plane].src.stride = stride;
              x->e_mbd.plane[plane].subsampling_x = ss_x;
              x->e_mbd.plane[plane].subsampling_y = ss_y;
              x->e_mbd.bd = bd;
              x->e_mbd.cur_buf = nullptr;

              x->pixel_gradient_info = grad_ref.data();
              av1_highbd_compute_gradient_info_sb_c(x.get(), sb_size, plane);

              x->pixel_gradient_info = grad_tst.data();
              API_REGISTER_STATE_CHECK(test_fn(x.get(), sb_size, plane));

              ExpectGradInfoEq(grad_ref.data(), grad_tst.data(), grad_count,
                               sb_size, plane, ss_x, ss_y, stride, bd);
            }
          }
        }
      }
    }
  }
}

#if HAVE_SSE2
INSTANTIATE_TEST_SUITE_P(
    SSE2, HighbdIntraHogSimdTest,
    ::testing::Combine(
        ::testing::Values(&av1_highbd_generate_hog_sse2),
        ::testing::Values(&av1_highbd_compute_gradient_info_sb_sse2),
        ::testing::Values(8, 10, 12)));
#endif  // HAVE_SSE2
#endif  // CONFIG_AV1_HIGHBITDEPTH

}  // namespace
