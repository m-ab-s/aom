/*
 *  Copyright (c) 2023, Alliance for Open Media. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <arm_neon.h>
#include <assert.h>

#include "config/aom_config.h"

#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/arm/dot_sve.h"
#include "aom_dsp/arm/mem_neon.h"

int64_t av1_block_error_sve(const tran_low_t *coeff, const tran_low_t *dqcoeff,
                            intptr_t block_size, int64_t *ssz) {
  int64x2_t error[2] = { vdupq_n_s64(0), vdupq_n_s64(0) };
  int64x2_t sqcoeff[2] = { vdupq_n_s64(0), vdupq_n_s64(0) };

  assert(block_size >= 16);
  assert((block_size % 16) == 0);

  do {
    const int16x8_t c0 = load_tran_low_to_s16q(coeff);
    const int16x8_t c1 = load_tran_low_to_s16q(coeff + 8);
    const int16x8_t d0 = load_tran_low_to_s16q(dqcoeff);
    const int16x8_t d1 = load_tran_low_to_s16q(dqcoeff + 8);

    const int16x8_t diff0 = vsubq_s16(c0, d0);
    const int16x8_t diff1 = vsubq_s16(c1, d1);

    error[0] = aom_sdotq_s16(error[0], diff0, diff0);
    error[1] = aom_sdotq_s16(error[1], diff1, diff1);
    sqcoeff[0] = aom_sdotq_s16(sqcoeff[0], c0, c0);
    sqcoeff[1] = aom_sdotq_s16(sqcoeff[1], c1, c1);

    coeff += 16;
    dqcoeff += 16;
    block_size -= 16;
  } while (block_size != 0);

  *ssz = vaddvq_s64(vaddq_s64(sqcoeff[0], sqcoeff[1]));
  return vaddvq_s64(vaddq_s64(error[0], error[1]));
}
