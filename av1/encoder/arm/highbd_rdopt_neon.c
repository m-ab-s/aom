/*
 * Copyright (c) 2023, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <assert.h>
#include <arm_neon.h>

#include "config/av1_rtcd.h"
#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/arm/sum_neon.h"

// Compute squared error and squared coefficient sum for 8 highbd coefficients,
// initializing the dual 64-bit accumulators directly via vmull_u32 / vmull_s32
// on the first 4 coefficients without zero-register setup.
static AOM_FORCE_INLINE void highbd_block_error_8_init_neon(
    const tran_low_t *coeff, const tran_low_t *dqcoeff, uint64x2_t *err_u64_0,
    uint64x2_t *err_u64_1, int64x2_t *ssz_s64_0, int64x2_t *ssz_s64_1) {
  const int32x4_t c0 = vld1q_s32(coeff);
  const int32x4_t c1 = vld1q_s32(coeff + 4);
  const int32x4_t d0 = vld1q_s32(dqcoeff);
  const int32x4_t d1 = vld1q_s32(dqcoeff + 4);

  const uint32x4_t diff0 = vreinterpretq_u32_s32(vabdq_s32(c0, d0));
  const uint32x4_t diff1 = vreinterpretq_u32_s32(vabdq_s32(c1, d1));

  *err_u64_0 = vmull_u32(vget_low_u32(diff0), vget_low_u32(diff0));
  *err_u64_1 = vmull_u32(vget_high_u32(diff0), vget_high_u32(diff0));
  *err_u64_0 = vmlal_u32(*err_u64_0, vget_low_u32(diff1), vget_low_u32(diff1));
  *err_u64_1 =
      vmlal_u32(*err_u64_1, vget_high_u32(diff1), vget_high_u32(diff1));

  *ssz_s64_0 = vmull_s32(vget_low_s32(c0), vget_low_s32(c0));
  *ssz_s64_1 = vmull_s32(vget_high_s32(c0), vget_high_s32(c0));
  *ssz_s64_0 = vmlal_s32(*ssz_s64_0, vget_low_s32(c1), vget_low_s32(c1));
  *ssz_s64_1 = vmlal_s32(*ssz_s64_1, vget_high_s32(c1), vget_high_s32(c1));
}

// Accumulate 8 highbd coefficients into dual 64-bit accumulators (10 Q-regs
// total, fitting within the 16-register ARMv7 NEON budget without spills).
static AOM_FORCE_INLINE void highbd_block_error_8_accum_neon(
    const tran_low_t *coeff, const tran_low_t *dqcoeff, uint64x2_t *err_u64_0,
    uint64x2_t *err_u64_1, int64x2_t *ssz_s64_0, int64x2_t *ssz_s64_1) {
  const int32x4_t c0 = vld1q_s32(coeff);
  const int32x4_t c1 = vld1q_s32(coeff + 4);
  const int32x4_t d0 = vld1q_s32(dqcoeff);
  const int32x4_t d1 = vld1q_s32(dqcoeff + 4);

  const uint32x4_t diff0 = vreinterpretq_u32_s32(vabdq_s32(c0, d0));
  const uint32x4_t diff1 = vreinterpretq_u32_s32(vabdq_s32(c1, d1));

  *err_u64_0 = vmlal_u32(*err_u64_0, vget_low_u32(diff0), vget_low_u32(diff0));
  *err_u64_1 =
      vmlal_u32(*err_u64_1, vget_high_u32(diff0), vget_high_u32(diff0));
  *err_u64_0 = vmlal_u32(*err_u64_0, vget_low_u32(diff1), vget_low_u32(diff1));
  *err_u64_1 =
      vmlal_u32(*err_u64_1, vget_high_u32(diff1), vget_high_u32(diff1));

  *ssz_s64_0 = vmlal_s32(*ssz_s64_0, vget_low_s32(c0), vget_low_s32(c0));
  *ssz_s64_1 = vmlal_s32(*ssz_s64_1, vget_high_s32(c0), vget_high_s32(c0));
  *ssz_s64_0 = vmlal_s32(*ssz_s64_0, vget_low_s32(c1), vget_low_s32(c1));
  *ssz_s64_1 = vmlal_s32(*ssz_s64_1, vget_high_s32(c1), vget_high_s32(c1));
}

int64_t av1_highbd_block_error_neon(const tran_low_t *coeff,
                                    const tran_low_t *dqcoeff,
                                    intptr_t block_size, int64_t *ssz, int bd) {
  assert(block_size >= 16);
  assert((block_size % 16) == 0);
  assert(bd == 8 || bd == 10 || bd == 12);

  uint64x2_t err_u64_0, err_u64_1;
  int64x2_t ssz_s64_0, ssz_s64_1;
  highbd_block_error_8_init_neon(coeff, dqcoeff, &err_u64_0, &err_u64_1,
                                 &ssz_s64_0, &ssz_s64_1);
  highbd_block_error_8_accum_neon(coeff + 8, dqcoeff + 8, &err_u64_0,
                                  &err_u64_1, &ssz_s64_0, &ssz_s64_1);

  for (block_size -= 16; block_size != 0; block_size -= 16) {
    coeff += 16;
    dqcoeff += 16;
    highbd_block_error_8_accum_neon(coeff, dqcoeff, &err_u64_0, &err_u64_1,
                                    &ssz_s64_0, &ssz_s64_1);
    highbd_block_error_8_accum_neon(coeff + 8, dqcoeff + 8, &err_u64_0,
                                    &err_u64_1, &ssz_s64_0, &ssz_s64_1);
  }

  const int shift = 2 * (bd - 8);
  const int rounding = (1 << shift) >> 1;
  const uint64x2_t err_u64 = vaddq_u64(err_u64_0, err_u64_1);
  const int64x2_t ssz_s64 = vaddq_s64(ssz_s64_0, ssz_s64_1);
  *ssz = (horizontal_add_s64x2(ssz_s64) + rounding) >> shift;
  return ((int64_t)horizontal_add_u64x2(err_u64) + rounding) >> shift;
}
