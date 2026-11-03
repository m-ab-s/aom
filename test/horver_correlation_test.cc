/*
 * Copyright (c) 2018, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <tuple>

#include "gtest/gtest.h"

#include "test/acm_random.h"
#include "test/register_state_check.h"
#include "test/util.h"

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"
#include "config/av1_rtcd.h"

#include "aom/aom_integer.h"

using libaom_test::ACMRandom;

namespace {
using HorverFunc = void (*)(const int16_t *diff, int stride, int w, int h,
                            float *hcorr, float *vcorr);

using HorverTestParam = std::tuple<const HorverFunc>;

class HorverTest : public ::testing::TestWithParam<HorverTestParam> {
 public:
  void SetUp() override {
    data_buf_ = (int16_t *)aom_malloc(MAX_SB_SQUARE * sizeof(int16_t));
    ASSERT_NE(data_buf_, nullptr);
    target_func_ = GET_PARAM(0);
  }
  void TearDown() override { aom_free(data_buf_); }
  void RunHorverTest();
  void RunHorverTest_ExtremeValues();
  void RunHorverSpeedTest(int run_times);

 private:
  HorverFunc target_func_;
  ACMRandom rng_;
  int16_t *data_buf_;
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(HorverTest);

void HorverTest::RunHorverTest() {
  for (int block_size = 0; block_size < BLOCK_SIZES_ALL; block_size++) {
    const int w = block_size_wide[block_size];
    const int h = block_size_high[block_size];
    for (int iter = 0; iter < 1000 && !HasFatalFailure(); ++iter) {
      float hcorr_ref = 0.0, vcorr_ref = 0.0;
      float hcorr_test = 0.0, vcorr_test = 0.0;

      for (int i = 0; i < MAX_SB_SQUARE; ++i) {
        data_buf_[i] = (rng_.Rand16() % (1 << 12)) - (1 << 11);
      }

      av1_get_horver_correlation_full_c(data_buf_, MAX_SB_SIZE, w, h,
                                        &hcorr_ref, &vcorr_ref);

      target_func_(data_buf_, MAX_SB_SIZE, w, h, &hcorr_test, &vcorr_test);

      ASSERT_LE(fabs(hcorr_ref - hcorr_test), 1e-6)
          << "hcorr incorrect (" << w << "x" << h << ")";
      ASSERT_LE(fabs(vcorr_ref - vcorr_test), 1e-6)
          << "vcorr incorrect (" << w << "x" << h << ")";
    }
    //    printf("(%3dx%-3d) passed\n", w, h);
  }
}

void HorverTest::RunHorverSpeedTest(int run_times) {
  for (int i = 0; i < MAX_SB_SQUARE; ++i) {
    data_buf_[i] = rng_.Rand16() % (1 << 12);
  }

  for (int block_size = 0; block_size < BLOCK_SIZES_ALL; block_size++) {
    const int w = block_size_wide[block_size];
    const int h = block_size_high[block_size];
    float hcorr_ref = 0.0, vcorr_ref = 0.0;
    float hcorr_test = 0.0, vcorr_test = 0.0;

    aom_usec_timer timer;
    aom_usec_timer_start(&timer);
    for (int i = 0; i < run_times; ++i) {
      av1_get_horver_correlation_full_c(data_buf_, MAX_SB_SIZE, w, h,
                                        &hcorr_ref, &vcorr_ref);
    }
    aom_usec_timer_mark(&timer);
    const double time1 = static_cast<double>(aom_usec_timer_elapsed(&timer));
    aom_usec_timer_start(&timer);
    for (int i = 0; i < run_times; ++i) {
      target_func_(data_buf_, MAX_SB_SIZE, w, h, &hcorr_test, &vcorr_test);
    }
    aom_usec_timer_mark(&timer);
    const double time2 = static_cast<double>(aom_usec_timer_elapsed(&timer));

    printf("%3dx%-3d:%7.2f/%7.2fns (%3.2f)\n", w, h, time1, time2,
           time1 / time2);
  }
}

void HorverTest::RunHorverTest_ExtremeValues() {
  for (int i = 0; i < MAX_SB_SQUARE; ++i) {
    // Most of get_horver_test is squaring and summing, so simply saturating
    // the whole buffer is mostly likely to cause an overflow.
    data_buf_[i] = (1 << 12) - 1;
  }

  for (int block_size = 0; block_size < BLOCK_SIZES_ALL; block_size++) {
    const int w = block_size_wide[block_size];
    const int h = block_size_high[block_size];
    float hcorr_ref = 0.0, vcorr_ref = 0.0;
    float hcorr_test = 0.0, vcorr_test = 0.0;

    av1_get_horver_correlation_full_c(data_buf_, MAX_SB_SIZE, w, h, &hcorr_ref,
                                      &vcorr_ref);
    target_func_(data_buf_, MAX_SB_SIZE, w, h, &hcorr_test, &vcorr_test);

    ASSERT_LE(fabs(hcorr_ref - hcorr_test), 1e-6) << "hcorr incorrect";
    ASSERT_LE(fabs(vcorr_ref - vcorr_test), 1e-6) << "vcorr incorrect";
  }
}

TEST_P(HorverTest, RandomValues) { RunHorverTest(); }

TEST_P(HorverTest, ExtremeValues) { RunHorverTest_ExtremeValues(); }

TEST_P(HorverTest, DISABLED_Speed) { RunHorverSpeedTest(100000); }

#if HAVE_SSE4_1
INSTANTIATE_TEST_SUITE_P(
    SSE4_1, HorverTest,
    ::testing::Values(av1_get_horver_correlation_full_sse4_1));
#endif  // HAVE_SSE4_1

#if HAVE_NEON
INSTANTIATE_TEST_SUITE_P(
    NEON, HorverTest, ::testing::Values(av1_get_horver_correlation_full_neon));
#endif  // HAVE_NEON

#if HAVE_AVX2
INSTANTIATE_TEST_SUITE_P(
    AVX2, HorverTest, ::testing::Values(av1_get_horver_correlation_full_avx2));
#endif  // HAVE_AVX2

using EnergyDistFunc = void (*)(const int16_t *diff, int stride, int bw, int bh,
                                float *hordist, float *verdist);

class EnergyDistributionTest : public ::testing::TestWithParam<EnergyDistFunc> {
 public:
  void SetUp() override {
    data_buf_ = (int16_t *)aom_malloc(MAX_SB_SQUARE * sizeof(int16_t));
    ASSERT_NE(data_buf_, nullptr);
    target_func_ = GetParam();
  }
  void TearDown() override { aom_free(data_buf_); }
  void RunRandomTest();
  void RunZeroTest();
  void RunExtremeValuesTest();

 private:
  EnergyDistFunc target_func_;
  ACMRandom rng_;
  int16_t *data_buf_;
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(EnergyDistributionTest);

void EnergyDistributionTest::RunRandomTest() {
  for (int tx_size = 0; tx_size < TX_SIZES_ALL; ++tx_size) {
    const int bw = tx_size_wide[tx_size];
    const int bh = tx_size_high[tx_size];
    if (bw > 32 || bh > 32) continue;
    const int esq_w = bw <= 8 ? bw : bw / 2;
    const int esq_h = bh <= 8 ? bh : bh / 2;
    for (int iter = 0; iter < 500 && !HasFatalFailure(); ++iter) {
      float hdist_ref[16] = { 0 }, vdist_ref[16] = { 0 };
      float hdist_test[16] = { 0 }, vdist_test[16] = { 0 };

      for (int i = 0; i < MAX_SB_SQUARE; ++i) {
        data_buf_[i] = (rng_.Rand16() % (1 << 13)) - (1 << 12);
      }

      av1_get_energy_distribution_finer_c(data_buf_, MAX_SB_SIZE, bw, bh,
                                          hdist_ref, vdist_ref);
      API_REGISTER_STATE_CHECK(
          target_func_(data_buf_, MAX_SB_SIZE, bw, bh, hdist_test, vdist_test));

      for (int j = 0; j < esq_w - 1; ++j) {
        ASSERT_FLOAT_EQ(hdist_ref[j], hdist_test[j])
            << "hordist mismatch at " << j << " (" << bw << "x" << bh << ")";
      }
      for (int i = 0; i < esq_h - 1; ++i) {
        ASSERT_FLOAT_EQ(vdist_ref[i], vdist_test[i])
            << "verdist mismatch at " << i << " (" << bw << "x" << bh << ")";
      }
    }
  }
}

void EnergyDistributionTest::RunZeroTest() {
  memset(data_buf_, 0, MAX_SB_SQUARE * sizeof(int16_t));
  for (int tx_size = 0; tx_size < TX_SIZES_ALL; ++tx_size) {
    const int bw = tx_size_wide[tx_size];
    const int bh = tx_size_high[tx_size];
    if (bw > 32 || bh > 32) continue;
    const int esq_w = bw <= 8 ? bw : bw / 2;
    const int esq_h = bh <= 8 ? bh : bh / 2;
    float hdist_ref[16] = { 0 }, vdist_ref[16] = { 0 };
    float hdist_test[16] = { 0 }, vdist_test[16] = { 0 };

    av1_get_energy_distribution_finer_c(data_buf_, MAX_SB_SIZE, bw, bh,
                                        hdist_ref, vdist_ref);
    API_REGISTER_STATE_CHECK(
        target_func_(data_buf_, MAX_SB_SIZE, bw, bh, hdist_test, vdist_test));

    for (int j = 0; j < esq_w - 1; ++j) {
      ASSERT_FLOAT_EQ(hdist_ref[j], hdist_test[j]);
    }
    for (int i = 0; i < esq_h - 1; ++i) {
      ASSERT_FLOAT_EQ(vdist_ref[i], vdist_test[i]);
    }
  }
}

void EnergyDistributionTest::RunExtremeValuesTest() {
  for (int i = 0; i < MAX_SB_SQUARE; ++i) {
    data_buf_[i] = (1 << 12) - 1;
  }
  for (int tx_size = 0; tx_size < TX_SIZES_ALL; ++tx_size) {
    const int bw = tx_size_wide[tx_size];
    const int bh = tx_size_high[tx_size];
    if (bw > 32 || bh > 32) continue;
    const int esq_w = bw <= 8 ? bw : bw / 2;
    const int esq_h = bh <= 8 ? bh : bh / 2;
    float hdist_ref[16] = { 0 }, vdist_ref[16] = { 0 };
    float hdist_test[16] = { 0 }, vdist_test[16] = { 0 };

    av1_get_energy_distribution_finer_c(data_buf_, MAX_SB_SIZE, bw, bh,
                                        hdist_ref, vdist_ref);
    API_REGISTER_STATE_CHECK(
        target_func_(data_buf_, MAX_SB_SIZE, bw, bh, hdist_test, vdist_test));

    for (int j = 0; j < esq_w - 1; ++j) {
      ASSERT_FLOAT_EQ(hdist_ref[j], hdist_test[j]);
    }
    for (int i = 0; i < esq_h - 1; ++i) {
      ASSERT_FLOAT_EQ(vdist_ref[i], vdist_test[i]);
    }
  }
}

TEST_P(EnergyDistributionTest, RandomValues) { RunRandomTest(); }

TEST_P(EnergyDistributionTest, ZeroValues) { RunZeroTest(); }

TEST_P(EnergyDistributionTest, ExtremeValues) { RunExtremeValuesTest(); }

#if HAVE_AVX2
INSTANTIATE_TEST_SUITE_P(
    AVX2, EnergyDistributionTest,
    ::testing::Values(av1_get_energy_distribution_finer_avx2));
#endif  // HAVE_AVX2

}  // namespace
