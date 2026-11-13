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
#include <vector>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"

#include "aom_dsp/prob.h"
#include "aom_ports/aom_timer.h"
#include "av1/common/entropy.h"
#include "av1/encoder/cost.h"
#include "test/acm_random.h"
#include "gtest/gtest.h"

namespace {

using libaom_test::ACMRandom;

using Cdf4BatchFn = void (*)(int *costs, int costs_stride, const uint16_t *cdf,
                             int cdf_stride, int n);

// Fixture that fills av1_cost_lut_low once before exercising the cost helpers.
class CostTokensFromCdf4Test : public ::testing::Test {
 protected:
  void SetUp() override { av1_init_cost_luts(); }
};

// Builds a valid stored 4-symbol CDF from monotonic cumulative probs.
void BuildCdf4(ACMRandom *rnd, aom_cdf_prob cdf[5]) {
  uint16_t v[3];
  for (int k = 0; k < 3; ++k) v[k] = 1 + (rnd->Rand16() % (CDF_PROB_TOP - 1));
  // Keep it monotonic; leave duplicates in so the EC_MIN_PROB clamp gets
  // exercised.
  std::sort(v, v + 3);
  cdf[0] = AOM_ICDF(v[0]);
  cdf[1] = AOM_ICDF(v[1]);
  cdf[2] = AOM_ICDF(v[2]);
  cdf[3] = AOM_ICDF(CDF_PROB_TOP);
  cdf[4] = 0;  // CDF adaptation counter slot; not read by the cost path.
}

// Builds a valid stored 2-symbol CDF.
void BuildCdf2(ACMRandom *rnd, aom_cdf_prob cdf[3]) {
  cdf[0] = AOM_ICDF(1 + (rnd->Rand16() % (CDF_PROB_TOP - 1)));
  cdf[1] = AOM_ICDF(CDF_PROB_TOP);
  cdf[2] = 0;  // CDF adaptation counter slot; not read by the cost path.
}

// The hybrid LUT+arithmetic path must equal the pure-arithmetic
// av1_cost_symbol for every p15, including the COST_LUT_LOW boundary and the
// fallback range.
TEST_F(CostTokensFromCdf4Test, HybridMatchesArithmetic) {
  for (int p = 1; p < CDF_PROB_TOP; ++p)
    ASSERT_EQ(av1_cost_symbol((aom_cdf_prob)p), av1_cost_symbol_lut(p))
        << "p15=" << p;
}

// The N=4 specialization must be bit-identical to the generic path.
TEST_F(CostTokensFromCdf4Test, MatchesGenericRandom) {
  ACMRandom rnd(ACMRandom::DeterministicSeed());
  for (int iter = 0; iter < 200000; ++iter) {
    aom_cdf_prob cdf[5];
    BuildCdf4(&rnd, cdf);
    int ref[4], spec[4];
    av1_cost_tokens_from_cdf(ref, cdf, nullptr);
    av1_cost_tokens_from_cdf_4(spec, cdf);
    for (int i = 0; i < 4; ++i)
      ASSERT_EQ(ref[i], spec[i]) << "iter=" << iter << " i=" << i;
  }
}

// Degenerate CDFs (near-zero probs) must clamp identically.
TEST_F(CostTokensFromCdf4Test, MatchesGenericEdge) {
  const aom_cdf_prob edge[][5] = {
    { AOM_ICDF(1), AOM_ICDF(2), AOM_ICDF(3), AOM_ICDF(CDF_PROB_TOP), 0 },
    { AOM_ICDF(1), AOM_ICDF(1), AOM_ICDF(1), AOM_ICDF(CDF_PROB_TOP), 0 },
    { AOM_ICDF(CDF_PROB_TOP - 3), AOM_ICDF(CDF_PROB_TOP - 2),
      AOM_ICDF(CDF_PROB_TOP - 1), AOM_ICDF(CDF_PROB_TOP), 0 },
    { AOM_ICDF(8192), AOM_ICDF(16384), AOM_ICDF(24576), AOM_ICDF(CDF_PROB_TOP),
      0 },
  };
  for (const auto &cdf : edge) {
    int ref[4], spec[4];
    av1_cost_tokens_from_cdf(ref, cdf, nullptr);
    av1_cost_tokens_from_cdf_4(spec, cdf);
    for (int i = 0; i < 4; ++i) ASSERT_EQ(ref[i], spec[i]) << "i=" << i;
  }
}

// Runs `fn` for a range of context counts and output strides and checks each
// filled entry against the generic scalar path. Exercises the SIMD main loop
// plus every tail length (n % 8) and both output strides (8 = base_cost,
// 4 = br_rate).
void CheckBatchAgainstGeneric(Cdf4BatchFn fn) {
  ACMRandom rnd(ACMRandom::DeterministicSeed());
  const int kCdfStride = CDF_SIZE(4);  // 5
  for (const int costs_stride : { 4, 8 }) {
    for (int n = 1; n <= 45; ++n) {
      std::vector<aom_cdf_prob> cdf(n * kCdfStride);
      for (int k = 0; k < n; ++k) BuildCdf4(&rnd, &cdf[k * kCdfStride]);

      std::vector<int> out(n * costs_stride, -1);
      std::vector<int> ref(n * 4);
      for (int k = 0; k < n; ++k)
        av1_cost_tokens_from_cdf(&ref[k * 4], &cdf[k * kCdfStride], nullptr);

      fn(out.data(), costs_stride, cdf.data(), kCdfStride, n);

      for (int k = 0; k < n; ++k)
        for (int i = 0; i < 4; ++i)
          ASSERT_EQ(ref[k * 4 + i], out[k * costs_stride + i])
              << "stride=" << costs_stride << " n=" << n << " k=" << k
              << " i=" << i;
    }
  }
}

TEST_F(CostTokensFromCdf4Test, BatchCMatchesGeneric) {
  CheckBatchAgainstGeneric(av1_cost_tokens_from_cdf_4_batch_c);
}

#if HAVE_AVX2
TEST_F(CostTokensFromCdf4Test, BatchAVX2MatchesGeneric) {
  CheckBatchAgainstGeneric(av1_cost_tokens_from_cdf_4_batch_avx2);
}
#endif  // HAVE_AVX2

// Same as CheckBatchAgainstGeneric but for the N=2 batch primitive.
void CheckBatch2AgainstGeneric(Cdf4BatchFn fn) {
  ACMRandom rnd(ACMRandom::DeterministicSeed());
  const int kCdfStride = CDF_SIZE(2);  // 3
  for (const int costs_stride : { 2, 4 }) {
    for (int n = 1; n <= 45; ++n) {
      std::vector<aom_cdf_prob> cdf(n * kCdfStride);
      for (int k = 0; k < n; ++k) BuildCdf2(&rnd, &cdf[k * kCdfStride]);

      std::vector<int> out(n * costs_stride, -1);
      std::vector<int> ref(n * 2);
      for (int k = 0; k < n; ++k)
        av1_cost_tokens_from_cdf(&ref[k * 2], &cdf[k * kCdfStride], nullptr);

      fn(out.data(), costs_stride, cdf.data(), kCdfStride, n);

      for (int k = 0; k < n; ++k)
        for (int i = 0; i < 2; ++i)
          ASSERT_EQ(ref[k * 2 + i], out[k * costs_stride + i])
              << "stride=" << costs_stride << " n=" << n << " k=" << k
              << " i=" << i;
    }
  }
}

TEST_F(CostTokensFromCdf4Test, Batch2CMatchesGeneric) {
  CheckBatch2AgainstGeneric(av1_cost_tokens_from_cdf_2_batch_c);
}

#if HAVE_AVX2
TEST_F(CostTokensFromCdf4Test, Batch2AVX2MatchesGeneric) {
  CheckBatch2AgainstGeneric(av1_cost_tokens_from_cdf_2_batch_avx2);
}
#endif  // HAVE_AVX2

#if HAVE_AVX2
// Times the C vs AVX2 batch primitive on the real av1_fill_coeff_costs call
// shapes and prints ns/call and the speedup. DISABLED_ so it runs only on
// request (--gtest_also_run_disabled_tests).
void RunBatchSpeed(const char *name, Cdf4BatchFn c_fn, Cdf4BatchFn simd_fn,
                   int cdf_stride, int syms, const char *label, int n,
                   int costs_stride) {
  ACMRandom rnd(ACMRandom::DeterministicSeed());
  std::vector<aom_cdf_prob> cdf(n * cdf_stride);
  for (int k = 0; k < n; ++k) {
    if (syms == 4)
      BuildCdf4(&rnd, &cdf[k * cdf_stride]);
    else
      BuildCdf2(&rnd, &cdf[k * cdf_stride]);
  }
  std::vector<int> out(n * costs_stride);

  const Cdf4BatchFn funcs[2] = { c_fn, simd_fn };
  const int kLoops = 1000000;
  double ns[2];
  for (int i = 0; i < 2; ++i) {
    aom_usec_timer timer;
    aom_usec_timer_start(&timer);
    for (int j = 0; j < kLoops; ++j)
      funcs[i](out.data(), costs_stride, cdf.data(), cdf_stride, n);
    aom_usec_timer_mark(&timer);
    ns[i] =
        1000.0 * static_cast<double>(aom_usec_timer_elapsed(&timer)) / kLoops;
  }
  printf("%s %-9s n=%2d stride=%d: c %7.1f ns  avx2 %7.1f ns  (%4.2fx)\n", name,
         label, n, costs_stride, ns[0], ns[1], ns[0] / ns[1]);
}

TEST_F(CostTokensFromCdf4Test, DISABLED_Batch4Speed) {
  RunBatchSpeed("cdf_4_batch", av1_cost_tokens_from_cdf_4_batch_c,
                av1_cost_tokens_from_cdf_4_batch_avx2, CDF_SIZE(4), 4,
                "base_cost", SIG_COEF_CONTEXTS, 8);
  RunBatchSpeed("cdf_4_batch", av1_cost_tokens_from_cdf_4_batch_c,
                av1_cost_tokens_from_cdf_4_batch_avx2, CDF_SIZE(4), 4,
                "br_rate", LEVEL_CONTEXTS, BR_CDF_SIZE);
}

TEST_F(CostTokensFromCdf4Test, DISABLED_Batch2Speed) {
  RunBatchSpeed("cdf_2_batch", av1_cost_tokens_from_cdf_2_batch_c,
                av1_cost_tokens_from_cdf_2_batch_avx2, CDF_SIZE(2), 2,
                "txb_skip", TXB_SKIP_CONTEXTS, 2);
  RunBatchSpeed("cdf_2_batch", av1_cost_tokens_from_cdf_2_batch_c,
                av1_cost_tokens_from_cdf_2_batch_avx2, CDF_SIZE(2), 2,
                "eob_extra", EOB_COEF_CONTEXTS, 2);
}
#endif  // HAVE_AVX2

}  // namespace
