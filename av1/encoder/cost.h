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

#ifndef AOM_AV1_ENCODER_COST_H_
#define AOM_AV1_ENCODER_COST_H_

#include "aom_dsp/prob.h"
#include "aom/aom_integer.h"

#ifdef __cplusplus
extern "C" {
#endif

extern const uint16_t av1_prob_cost[129];

// The factor to scale from cost in bits to cost in av1_prob_cost units.
#define AV1_PROB_COST_SHIFT 9

// Cost of coding an n bit literal, using 128 (i.e. 50%) probability
// for each bit.
#define av1_cost_literal(n) ((n) * (1 << AV1_PROB_COST_SHIFT))

static inline int av1_cost_symbol_clamped(int p15) {
  assert(0 < p15 && p15 < CDF_PROB_TOP);
  const int shift = CDF_PROB_BITS - 1 - get_msb(p15);
  // Since CDF_PROB_TOP == 32768 and av1_prob_cost has 129 entries (including a
  // duplicate 128th entry for prob_idx == 128), get_prob(p15 << shift, 32768) -
  // 128 simplifies to ((p15 << shift) - 16320) >> 7.
  const int prob_idx = ((p15 << shift) - 16320) >> 7;
  return av1_prob_cost[prob_idx] + av1_cost_literal(shift);
}

// Calculate the cost of a symbol with probability p15 / 2^15
static inline int av1_cost_symbol(aom_cdf_prob p15) {
  // p15 can be out of range [1, CDF_PROB_TOP - 1]. Clamping it, so that the
  // following cost calculation works correctly. Otherwise, if p15 =
  // CDF_PROB_TOP, shift would be -1, and "p15 << shift" would be wrong.
  const int clamped = clamp((int)p15, 1, CDF_PROB_TOP - 1);
  return av1_cost_symbol_clamped(clamped);
}

void av1_cost_tokens_from_cdf(int *costs, const aom_cdf_prob *cdf,
                              const int *inv_map);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_ENCODER_COST_H_
