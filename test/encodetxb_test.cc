/*
 * Copyright (c) 2017, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <tuple>

#include "gtest/gtest.h"

#include "config/aom_config.h"
#include "config/av1_rtcd.h"

#include "aom_ports/aom_timer.h"
#include "aom_ports/mem.h"
#include "av1/common/av1_common_int.h"
#include "av1/common/idct.h"
#include "av1/common/scan.h"
#include "av1/common/txb_common.h"
#include "test/acm_random.h"
#include "test/register_state_check.h"
#include "test/util.h"

namespace {
using libaom_test::ACMRandom;

using GetNzMapContextsFunc = void (*)(const uint8_t *const levels,
                                      const int16_t *const scan, const int eob,
                                      const TX_SIZE tx_size,
                                      const TX_CLASS tx_class,
                                      int8_t *const coeff_contexts);

class EncodeTxbTest : public ::testing::TestWithParam<GetNzMapContextsFunc> {
 public:
  EncodeTxbTest() : get_nz_map_contexts_func_(GetParam()) {}

  ~EncodeTxbTest() override = default;

  void SetUp() override {
    coeff_contexts_ref_ = reinterpret_cast<int8_t *>(
        aom_memalign(16, sizeof(*coeff_contexts_ref_) * MAX_TX_SQUARE));
    ASSERT_NE(coeff_contexts_ref_, nullptr);
    coeff_contexts_ = reinterpret_cast<int8_t *>(
        aom_memalign(16, sizeof(*coeff_contexts_) * MAX_TX_SQUARE));
    ASSERT_NE(coeff_contexts_, nullptr);
  }

  void TearDown() override {
    aom_free(coeff_contexts_ref_);
    aom_free(coeff_contexts_);
  }

  void GetNzMapContextsRun(bool sparse = false) {
    const int kNumTests = sparse ? 4 : 10;
    int result = 0;

    static constexpr TX_TYPE kSparseTxTypes[] = { DCT_DCT, V_DCT, H_DCT };
    const int num_inter = sparse ? 1 : 2;
    const int num_tx_types = sparse
                                 ? static_cast<int>(sizeof(kSparseTxTypes) /
                                                    sizeof(kSparseTxTypes[0]))
                                 : static_cast<int>(TX_TYPES);

    for (int is_inter = 0; is_inter < num_inter; ++is_inter) {
      for (int tx_type_idx = 0; tx_type_idx < num_tx_types; ++tx_type_idx) {
        const int tx_type = sparse
                                ? static_cast<int>(kSparseTxTypes[tx_type_idx])
                                : tx_type_idx;
        const TX_CLASS tx_class = tx_type_to_class[tx_type];
        for (int tx_size = TX_4X4; tx_size < TX_SIZES_ALL; ++tx_size) {
          const int bhl = get_txb_bhl((TX_SIZE)tx_size);
          const int width = get_txb_wide((TX_SIZE)tx_size);
          const int height = get_txb_high((TX_SIZE)tx_size);
          const int real_width = tx_size_wide[tx_size];
          const int real_height = tx_size_high[tx_size];
          const int16_t *const scan = av1_scan_orders[tx_size][tx_type].scan;

          levels_ = set_levels(levels_buf_, height);
          for (int i = 0; i < kNumTests && !result; ++i) {
            for (int eob = 1; eob <= width * height && !result; ++eob) {
              InitDataWithEob(scan, bhl, eob, sparse);

              av1_get_nz_map_contexts_c(levels_, scan, eob, (TX_SIZE)tx_size,
                                        tx_class, coeff_contexts_ref_);
              if (is_inter == 0 && i == 0 &&
                  (eob <= 4 || eob == width * height || (eob & 63) == 0)) {
                API_REGISTER_STATE_CHECK(get_nz_map_contexts_func_(
                    levels_, scan, eob, (TX_SIZE)tx_size, tx_class,
                    coeff_contexts_));
              } else {
                get_nz_map_contexts_func_(levels_, scan, eob, (TX_SIZE)tx_size,
                                          tx_class, coeff_contexts_);
              }

              result = Compare(scan, eob);

              EXPECT_EQ(result, 0)
                  << " tx_class " << (int)tx_class << " width " << real_width
                  << " height " << real_height << " eob " << eob;
            }
          }
        }
      }
    }
  }

  void SpeedTestGetNzMapContextsRun() {
    const int kNumTests = 2000000000;
    aom_usec_timer timer;
    aom_usec_timer timer_ref;

    printf("Note: Only test the largest possible eob case!\n");
    for (int tx_size = TX_4X4; tx_size < TX_SIZES_ALL; ++tx_size) {
      const int bhl = get_txb_bhl((TX_SIZE)tx_size);
      const int width = get_txb_wide((TX_SIZE)tx_size);
      const int height = get_txb_high((TX_SIZE)tx_size);
      const int real_width = tx_size_wide[tx_size];
      const int real_height = tx_size_high[tx_size];
      const TX_TYPE tx_type = DCT_DCT;
      const TX_CLASS tx_class = tx_type_to_class[tx_type];
      const int16_t *const scan = av1_scan_orders[tx_size][tx_type].scan;
      const int eob = width * height;
      const int numTests = kNumTests / (width * height);

      levels_ = set_levels(levels_buf_, height);
      InitDataWithEob(scan, bhl, eob);

      aom_usec_timer_start(&timer_ref);
      for (int i = 0; i < numTests; ++i) {
        av1_get_nz_map_contexts_c(levels_, scan, eob, (TX_SIZE)tx_size,
                                  tx_class, coeff_contexts_ref_);
      }
      aom_usec_timer_mark(&timer_ref);

      levels_ = set_levels(levels_buf_, height);
      InitDataWithEob(scan, bhl, eob);

      aom_usec_timer_start(&timer);
      for (int i = 0; i < numTests; ++i) {
        get_nz_map_contexts_func_(levels_, scan, eob, (TX_SIZE)tx_size,
                                  tx_class, coeff_contexts_);
      }
      aom_usec_timer_mark(&timer);

      const int elapsed_time_ref =
          static_cast<int>(aom_usec_timer_elapsed(&timer_ref));
      const int elapsed_time = static_cast<int>(aom_usec_timer_elapsed(&timer));

      printf("get_nz_map_contexts_%2dx%2d: %7.1f ms ref %7.1f ms gain %4.2f\n",
             real_width, real_height, elapsed_time / 1000.0,
             elapsed_time_ref / 1000.0,
             (elapsed_time_ref * 1.0) / (elapsed_time * 1.0));
    }
  }

 private:
  void InitDataWithEob(const int16_t *const scan, const int bhl, const int eob,
                       bool sparse = false) {
    memset(levels_buf_, 0, sizeof(levels_buf_));
    memset(coeff_contexts_, 0, sizeof(*coeff_contexts_) * MAX_TX_SQUARE);

    for (int c = 0; c < eob; ++c) {
      levels_[get_padded_idx(scan[c], bhl)] =
          sparse ? static_cast<uint8_t>(rnd_.Rand8() & 3)
                 : static_cast<uint8_t>(clamp(rnd_.Rand8(), 0, INT8_MAX));
      coeff_contexts_[scan[c]] = static_cast<int8_t>(rnd_.Rand16() >> 1);
    }

    memcpy(coeff_contexts_ref_, coeff_contexts_,
           sizeof(*coeff_contexts_) * MAX_TX_SQUARE);
  }

  bool Compare(const int16_t *const scan, const int eob) const {
    bool result = false;
    if (memcmp(coeff_contexts_, coeff_contexts_ref_,
               sizeof(*coeff_contexts_ref_) * MAX_TX_SQUARE)) {
      for (int i = 0; i < eob; i++) {
        const int pos = scan[i];
        if (coeff_contexts_ref_[pos] != coeff_contexts_[pos]) {
          printf("coeff_contexts_[%d] diff:%6d (ref),%6d (opt)\n", pos,
                 coeff_contexts_ref_[pos], coeff_contexts_[pos]);
          result = true;
          break;
        }
      }
    }
    return result;
  }

  GetNzMapContextsFunc get_nz_map_contexts_func_;
  ACMRandom rnd_;
  uint8_t levels_buf_[TX_PAD_2D];
  uint8_t *levels_;
  int8_t *coeff_contexts_ref_;
  int8_t *coeff_contexts_;
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(EncodeTxbTest);

TEST_P(EncodeTxbTest, GetNzMapContexts) { GetNzMapContextsRun(); }

TEST_P(EncodeTxbTest, GetNzMapContextsSparse) { GetNzMapContextsRun(true); }

TEST_P(EncodeTxbTest, DISABLED_SpeedTestGetNzMapContexts) {
  SpeedTestGetNzMapContextsRun();
}

#if HAVE_SSE2
INSTANTIATE_TEST_SUITE_P(SSE2, EncodeTxbTest,
                         ::testing::Values(av1_get_nz_map_contexts_sse2));
#endif

#if CONFIG_HIGHWAY && HAVE_AVX512
INSTANTIATE_TEST_SUITE_P(AVX512, EncodeTxbTest,
                         ::testing::Values(av1_get_nz_map_contexts_avx512));
#endif

#if HAVE_NEON
INSTANTIATE_TEST_SUITE_P(NEON, EncodeTxbTest,
                         ::testing::Values(av1_get_nz_map_contexts_neon));
#endif

using av1_txb_init_levels_func = void (*)(const tran_low_t *const coeff,
                                          const int width, const int height,
                                          uint8_t *const levels);

using TxbInitLevelParam = std::tuple<av1_txb_init_levels_func, int>;

class EncodeTxbInitLevelTest
    : public ::testing::TestWithParam<TxbInitLevelParam> {
 public:
  ~EncodeTxbInitLevelTest() override = default;
  void RunTest(av1_txb_init_levels_func test_func, int tx_size, int is_speed);
  void RunHighBdTest(av1_txb_init_levels_func test_func, int tx_size);
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(EncodeTxbInitLevelTest);

void EncodeTxbInitLevelTest::RunTest(av1_txb_init_levels_func test_func,
                                     int tx_size, int is_speed) {
  const int width = get_txb_wide((TX_SIZE)tx_size);
  const int height = get_txb_high((TX_SIZE)tx_size);
  tran_low_t coeff[MAX_TX_SQUARE];

  uint8_t levels_buf[2][TX_PAD_2D];
  alignas(4) uint8_t levels_unaligned_buf[TX_PAD_2D + 1];
  uint8_t *const levels0 = set_levels(levels_buf[0], height);
  uint8_t *const levels1 = set_levels(levels_buf[1], height);
  uint8_t *const levels_unaligned =
      set_levels(levels_unaligned_buf + 1, height);

  ACMRandom rnd(ACMRandom::DeterministicSeed());
  const int num_passes = is_speed ? 1 : 2;
  for (int pass = 0; pass < num_passes; ++pass) {
    for (int i = 0; i < width * height; i++) {
      // Pass 0 tests full 16-bit clamping; pass 1 tests [-128, 128] so
      // distinct non-saturated lane values verify lane permutations and row
      // ordering.
      coeff[i] = (pass == 0) ? rnd.Rand16Signed() : (rnd(257) - 128);
    }
    const uint8_t canary = rnd.Rand8();
    levels_unaligned_buf[0] = canary;
    for (int i = 0; i < TX_PAD_2D; i++) {
      const uint8_t init_byte = rnd.Rand8();
      levels_buf[0][i] = init_byte;
      levels_buf[1][i] = init_byte;
      levels_unaligned_buf[i + 1] = init_byte;
    }
    const int run_times = is_speed ? (width * height) * 10000 : 1;
    aom_usec_timer timer;
    aom_usec_timer_start(&timer);
    for (int i = 0; i < run_times; ++i) {
      av1_txb_init_levels_c(coeff, width, height, levels0);
    }
    const double t1 = get_time_mark(&timer);
    aom_usec_timer_start(&timer);
    for (int i = 0; i < run_times; ++i) {
      test_func(coeff, width, height, levels1);
    }
    const double t2 = get_time_mark(&timer);
    if (is_speed) {
      printf("init %3dx%-3d:%7.2f/%7.2fns", width, height, t1, t2);
      printf("(%3.2f)\n", t1 / t2);
    } else {
      API_REGISTER_STATE_CHECK(
          test_func(coeff, width, height, levels_unaligned));
      ASSERT_EQ(canary, levels_unaligned_buf[0]);
    }
    const int stride = width + TX_PAD_HOR;
    for (int r = 0; r < height + TX_PAD_VER; ++r) {
      for (int c = 0; c < stride; ++c) {
        ASSERT_EQ(levels_buf[0][c + r * stride], levels_buf[1][c + r * stride])
            << "[" << r << "," << c << "] pass=" << pass << " " << run_times
            << " " << width << "x" << height;
        if (!is_speed) {
          ASSERT_EQ(levels_buf[0][c + r * stride],
                    levels_unaligned_buf[1 + c + r * stride])
              << "unaligned [" << r << "," << c << "] pass=" << pass << " "
              << width << "x" << height;
        }
      }
    }
  }
}

void EncodeTxbInitLevelTest::RunHighBdTest(av1_txb_init_levels_func test_func,
                                           int tx_size) {
  const int width = get_txb_wide((TX_SIZE)tx_size);
  const int height = get_txb_high((TX_SIZE)tx_size);
  tran_low_t coeff[MAX_TX_SQUARE];

  uint8_t levels_buf[2][TX_PAD_2D];
  alignas(4) uint8_t levels_unaligned_buf[TX_PAD_2D + 2];
  uint8_t *const levels0 = set_levels(levels_buf[0], height);
  uint8_t *const levels1 = set_levels(levels_buf[1], height);
  uint8_t *const levels_unaligned =
      set_levels(levels_unaligned_buf + 1, height);

  constexpr int kMaxHighBdCoeff = (1 << 18) - 1;
  constexpr tran_low_t kExtremeCoeffs[16] = { 0,
                                              127,
                                              -127,
                                              128,
                                              -128,
                                              INT16_MAX,
                                              -INT16_MAX,
                                              32768,
                                              65535,
                                              65536,
                                              kMaxHighBdCoeff,
                                              INT32_MAX,
                                              -1,
                                              -255,
                                              -16384,
                                              -32767 };
#if HAVE_NEON
  // Modes 2-3 exercise negative coeffs <= INT16_MIN. Gate on NEON because x86
  // av1_txb_init_levels_{sse4_1,avx2} pass _mm_abs_epi16(INT16_MIN) into signed
  // _mm_packs_epi16, saturating to -128 instead of 127.
  constexpr tran_low_t kNegExtremeCoeffs[16] = {
    -1,     -127,   -128,      -255,           -INT16_MAX,       INT16_MIN,
    -32769, -65535, -65536,    -131072,        -kMaxHighBdCoeff, -INT32_MAX,
    32768,  65536,  INT32_MAX, kMaxHighBdCoeff
  };
  const int num_modes = (test_func == &av1_txb_init_levels_neon) ? 4 : 2;
#else
  const int num_modes = 2;
#endif

  ACMRandom rnd(ACMRandom::DeterministicSeed());
  for (int mode = 0; mode < num_modes; ++mode) {
    for (int i = 0; i < width * height; ++i) {
      if (mode == 0) {
        coeff[i] = kExtremeCoeffs[i % 16];
      } else if (mode == 1) {
        const uint32_t hi = rnd.Rand16() & 7;
        const uint32_t lo = rnd.Rand16();
        const uint32_t mag = (hi << 16) | lo;
        const uint32_t span =
            static_cast<uint32_t>(kMaxHighBdCoeff) + INT16_MAX + 1u;
        coeff[i] = static_cast<tran_low_t>(mag % span) - INT16_MAX;
#if HAVE_NEON
      } else if (mode == 2) {
        coeff[i] = kNegExtremeCoeffs[i % 16];
      } else {
        const uint32_t hi = rnd.Rand16() & 7;
        const uint32_t lo = rnd.Rand16();
        const uint32_t mag = (hi << 16) | lo;
        coeff[i] = static_cast<tran_low_t>(mag) - kMaxHighBdCoeff;
#endif
      }
    }
    const uint8_t canary = rnd.Rand8();
    levels_unaligned_buf[0] = canary;
    levels_unaligned_buf[TX_PAD_2D + 1] = canary;
    for (int i = 0; i < TX_PAD_2D; ++i) {
      const uint8_t init_byte = rnd.Rand8();
      levels_buf[0][i] = init_byte;
      levels_buf[1][i] = init_byte;
      levels_unaligned_buf[i + 1] = init_byte;
    }
    av1_txb_init_levels_c(coeff, width, height, levels0);
    API_REGISTER_STATE_CHECK(test_func(coeff, width, height, levels1));
    API_REGISTER_STATE_CHECK(test_func(coeff, width, height, levels_unaligned));
    ASSERT_EQ(canary, levels_unaligned_buf[0]);
    ASSERT_EQ(canary, levels_unaligned_buf[TX_PAD_2D + 1]);
    const int stride = height + TX_PAD_HOR;
    for (int c = 0; c < width + TX_PAD_VER; ++c) {
      for (int r = 0; r < stride; ++r) {
        ASSERT_EQ(levels_buf[0][c * stride + r], levels_buf[1][c * stride + r])
            << "mode " << mode << " [" << c << "," << r << "] " << width << "x"
            << height;
        ASSERT_EQ(levels_buf[0][c * stride + r],
                  levels_unaligned_buf[1 + c * stride + r])
            << "unaligned mode " << mode << " [" << c << "," << r << "] "
            << width << "x" << height;
      }
    }
    int tail_start = (width + TX_PAD_VER) * stride + TX_PAD_END;
#if HAVE_NEON
    if (test_func == &av1_txb_init_levels_neon) tail_start -= TX_PAD_END;
#endif
    for (int i = tail_start; i < TX_PAD_2D; ++i) {
      ASSERT_EQ(levels_buf[0][i], levels_buf[1][i]) << "tail " << i;
      ASSERT_EQ(levels_buf[0][i], levels_unaligned_buf[1 + i]) << "tail " << i;
    }
  }
}

TEST_P(EncodeTxbInitLevelTest, match) {
  RunTest(GET_PARAM(0), GET_PARAM(1), 0);
}

TEST_P(EncodeTxbInitLevelTest, MatchHighBitdepth) {
  RunHighBdTest(GET_PARAM(0), GET_PARAM(1));
}

TEST_P(EncodeTxbInitLevelTest, DISABLED_Speed) {
  RunTest(GET_PARAM(0), GET_PARAM(1), 1);
}

#if HAVE_SSE4_1
INSTANTIATE_TEST_SUITE_P(
    SSE4_1, EncodeTxbInitLevelTest,
    ::testing::Combine(::testing::Values(&av1_txb_init_levels_sse4_1),
                       ::testing::Range(0, static_cast<int>(TX_SIZES_ALL), 1)));
#endif
#if HAVE_AVX2
INSTANTIATE_TEST_SUITE_P(
    AVX2, EncodeTxbInitLevelTest,
    ::testing::Combine(::testing::Values(&av1_txb_init_levels_avx2),
                       ::testing::Range(0, static_cast<int>(TX_SIZES_ALL), 1)));
#endif
#if CONFIG_HIGHWAY && HAVE_AVX512
INSTANTIATE_TEST_SUITE_P(
    AVX512, EncodeTxbInitLevelTest,
    ::testing::Combine(::testing::Values(&av1_txb_init_levels_avx512),
                       ::testing::Range(0, static_cast<int>(TX_SIZES_ALL), 1)));
#endif
#if HAVE_NEON
INSTANTIATE_TEST_SUITE_P(
    NEON, EncodeTxbInitLevelTest,
    ::testing::Combine(::testing::Values(&av1_txb_init_levels_neon),
                       ::testing::Range(0, static_cast<int>(TX_SIZES_ALL), 1)));
#endif
}  // namespace
