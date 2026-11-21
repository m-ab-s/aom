/*
 * Copyright (c) 2019, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <arm_neon.h>
#include <assert.h>
#include <stdint.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"

#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/arm/mem_neon.h"
#include "aom_dsp/arm/sum_neon.h"
#include "aom_ports/mem.h"

int64_t av1_block_error_neon(const tran_low_t *coeff, const tran_low_t *dqcoeff,
                             intptr_t block_size, int64_t *ssz) {
  uint64x2_t err_u64 = vdupq_n_u64(0);
  int64x2_t ssz_s64 = vdupq_n_s64(0);

  assert(block_size >= 16);
  assert((block_size % 16) == 0);

  do {
    const int16x8_t c0 = load_tran_low_to_s16q(coeff);
    const int16x8_t c1 = load_tran_low_to_s16q(coeff + 8);
    const int16x8_t d0 = load_tran_low_to_s16q(dqcoeff);
    const int16x8_t d1 = load_tran_low_to_s16q(dqcoeff + 8);

    const uint16x8_t diff0 = vreinterpretq_u16_s16(vabdq_s16(c0, d0));
    const uint16x8_t diff1 = vreinterpretq_u16_s16(vabdq_s16(c1, d1));

    // By operating on unsigned integers we can store up to 4 squared diff in a
    // 32-bit element before having to widen to 64 bits.
    uint32x4_t err = vmull_u16(vget_low_u16(diff0), vget_low_u16(diff0));
    err = vmlal_u16(err, vget_high_u16(diff0), vget_high_u16(diff0));
    err = vmlal_u16(err, vget_low_u16(diff1), vget_low_u16(diff1));
    err = vmlal_u16(err, vget_high_u16(diff1), vget_high_u16(diff1));
    err_u64 = vpadalq_u32(err_u64, err);

    // We can't do the same here as we're operating on signed integers, so we
    // can only accumulate 2 squares.
    int32x4_t ssz0 = vmull_s16(vget_low_s16(c0), vget_low_s16(c0));
    ssz0 = vmlal_s16(ssz0, vget_high_s16(c0), vget_high_s16(c0));
    ssz_s64 = vpadalq_s32(ssz_s64, ssz0);

    int32x4_t ssz1 = vmull_s16(vget_low_s16(c1), vget_low_s16(c1));
    ssz1 = vmlal_s16(ssz1, vget_high_s16(c1), vget_high_s16(c1));
    ssz_s64 = vpadalq_s32(ssz_s64, ssz1);

    coeff += 16;
    dqcoeff += 16;
    block_size -= 16;
  } while (block_size != 0);

  *ssz = horizontal_add_s64x2(ssz_s64);
  return (int64_t)horizontal_add_u64x2(err_u64);
}

int64_t av1_block_error_lp_neon(const int16_t *coeff, const int16_t *dqcoeff,
                                intptr_t block_size) {
  uint64x2_t err_u64 = vdupq_n_u64(0);

  assert(block_size >= 16);
  assert((block_size % 16) == 0);

  do {
    const int16x8_t c0 = vld1q_s16(coeff);
    const int16x8_t c1 = vld1q_s16(coeff + 8);
    const int16x8_t d0 = vld1q_s16(dqcoeff);
    const int16x8_t d1 = vld1q_s16(dqcoeff + 8);

    const uint16x8_t diff0 = vreinterpretq_u16_s16(vabdq_s16(c0, d0));
    const uint16x8_t diff1 = vreinterpretq_u16_s16(vabdq_s16(c1, d1));

    // By operating on unsigned integers we can store up to 4 squared diff in a
    // 32-bit element before having to widen to 64 bits.
    uint32x4_t err = vmull_u16(vget_low_u16(diff0), vget_low_u16(diff0));
    err = vmlal_u16(err, vget_high_u16(diff0), vget_high_u16(diff0));
    err = vmlal_u16(err, vget_low_u16(diff1), vget_low_u16(diff1));
    err = vmlal_u16(err, vget_high_u16(diff1), vget_high_u16(diff1));
    err_u64 = vpadalq_u32(err_u64, err);

    coeff += 16;
    dqcoeff += 16;
    block_size -= 16;
  } while (block_size != 0);

  return (int64_t)horizontal_add_u64x2(err_u64);
}

static AOM_FORCE_INLINE int any_nz_s32(int32x4_t v) {
#if AOM_ARCH_AARCH64
  return vmaxvq_u32(vreinterpretq_u32_s32(v)) != 0;
#else
  const uint64x2_t u64 = vreinterpretq_u64_s32(v);
  return (vgetq_lane_u64(u64, 0) | vgetq_lane_u64(u64, 1)) != 0;
#endif
}

static AOM_FORCE_INLINE uint64x2_t sum_sq_qm_4_u64(int32x4_t v, int32x4_t wt) {
  const int32x4_t prod = vmulq_s32(v, wt);
  const uint64x2_t sq_lo =
      vreinterpretq_u64_s64(vmull_s32(vget_low_s32(prod), vget_low_s32(prod)));
  const uint64x2_t sq_hi = vreinterpretq_u64_s64(
      vmull_s32(vget_high_s32(prod), vget_high_s32(prod)));
  return vrsraq_n_u64(vrshrq_n_u64(sq_lo, 2 * AOM_QM_BITS), sq_hi,
                      2 * AOM_QM_BITS);
}

static inline void acc_block_error_qm_8(int32x4_t c0, int32x4_t c1,
                                        int32x4_t d0, int32x4_t d1,
                                        int32x4_t d_or, uint8x8_t wt_u8,
                                        uint64x2_t *sse0, uint64x2_t *sse1,
                                        uint64x2_t *ssz0, uint64x2_t *ssz1) {
  const uint16x8_t wt_u16 = vmovl_u8(wt_u8);
  const int32x4_t wt0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(wt_u16)));
  const int32x4_t wt1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(wt_u16)));
  const uint64x2_t sq0 = sum_sq_qm_4_u64(c0, wt0);
  const uint64x2_t sq1 = sum_sq_qm_4_u64(c1, wt1);
  *ssz0 = vaddq_u64(*ssz0, sq0);
  *ssz1 = vaddq_u64(*ssz1, sq1);

  if (!any_nz_s32(d_or)) {
    *sse0 = vaddq_u64(*sse0, sq0);
    *sse1 = vaddq_u64(*sse1, sq1);
  } else {
    *sse0 = vaddq_u64(*sse0, sum_sq_qm_4_u64(vsubq_s32(c0, d0), wt0));
    *sse1 = vaddq_u64(*sse1, sum_sq_qm_4_u64(vsubq_s32(c1, d1), wt1));
  }
}

int64_t av1_block_error_qm_neon(const tran_low_t *coeff,
                                const tran_low_t *dqcoeff, intptr_t block_size,
                                const qm_val_t *qmatrix, const int16_t *scan,
                                int64_t *ssz, int bd) {
  assert(coeff != NULL);
  assert(dqcoeff != NULL);
  assert(qmatrix != NULL);
  assert(scan != NULL);
  assert(ssz != NULL);
  assert(block_size >= 16);
  assert((block_size & (block_size - 1)) == 0);
  assert(bd == 8 || bd == 10 || bd == 12);

  uint64x2_t sse0 = vdupq_n_u64(0);
  uint64x2_t sse1 = vdupq_n_u64(0);
  uint64x2_t ssz0 = vdupq_n_u64(0);
  uint64x2_t ssz1 = vdupq_n_u64(0);

#if AOM_ARCH_AARCH64
  // For block_size <= 64 (4x4, 4x8/8x4, 8x8, 4x16/16x4), block_size equals
  // av1_get_max_eob(tx_size) == tx area, so qmatrix has block_size bytes and
  // every scan[i] satisfies 0 <= scan[i] < block_size, allowing in-register
  // vqtbl lookups instead of 8 scalar vld1_lane_u8 gathers.
  if (block_size <= 64) {
    if (block_size == 16) {
      const uint8x16_t qm = vld1q_u8(qmatrix);
      do {
        const int32x4_t c0 = vld1q_s32(coeff);
        const int32x4_t c1 = vld1q_s32(coeff + 4);
        const int32x4_t d0 = vld1q_s32(dqcoeff);
        const int32x4_t d1 = vld1q_s32(dqcoeff + 4);
        const int32x4_t c_or = vorrq_s32(c0, c1);
        const int32x4_t d_or = vorrq_s32(d0, d1);
        if (any_nz_s32(vorrq_s32(c_or, d_or))) {
          const uint16x8_t scan_u16 = vreinterpretq_u16_s16(vld1q_s16(scan));
          assert(vmaxvq_u16(scan_u16) < 16);
          const uint8x8_t wt_u8 = vqtbl1_u8(qm, vmovn_u16(scan_u16));
          acc_block_error_qm_8(c0, c1, d0, d1, d_or, wt_u8, &sse0, &sse1, &ssz0,
                               &ssz1);
        }
        coeff += 8;
        dqcoeff += 8;
        scan += 8;
        block_size -= 8;
      } while (block_size != 0);
    } else if (block_size == 32) {
      const uint8x16x2_t qm = { { vld1q_u8(qmatrix), vld1q_u8(qmatrix + 16) } };
      do {
        const int32x4_t c0 = vld1q_s32(coeff);
        const int32x4_t c1 = vld1q_s32(coeff + 4);
        const int32x4_t d0 = vld1q_s32(dqcoeff);
        const int32x4_t d1 = vld1q_s32(dqcoeff + 4);
        const int32x4_t c_or = vorrq_s32(c0, c1);
        const int32x4_t d_or = vorrq_s32(d0, d1);
        if (any_nz_s32(vorrq_s32(c_or, d_or))) {
          const uint16x8_t scan_u16 = vreinterpretq_u16_s16(vld1q_s16(scan));
          assert(vmaxvq_u16(scan_u16) < 32);
          const uint8x8_t wt_u8 = vqtbl2_u8(qm, vmovn_u16(scan_u16));
          acc_block_error_qm_8(c0, c1, d0, d1, d_or, wt_u8, &sse0, &sse1, &ssz0,
                               &ssz1);
        }
        coeff += 8;
        dqcoeff += 8;
        scan += 8;
        block_size -= 8;
      } while (block_size != 0);
    } else {
      assert(block_size == 64);
      const uint8x16x4_t qm = { { vld1q_u8(qmatrix), vld1q_u8(qmatrix + 16),
                                  vld1q_u8(qmatrix + 32),
                                  vld1q_u8(qmatrix + 48) } };
      do {
        const int32x4_t c0 = vld1q_s32(coeff);
        const int32x4_t c1 = vld1q_s32(coeff + 4);
        const int32x4_t d0 = vld1q_s32(dqcoeff);
        const int32x4_t d1 = vld1q_s32(dqcoeff + 4);
        const int32x4_t c_or = vorrq_s32(c0, c1);
        const int32x4_t d_or = vorrq_s32(d0, d1);
        if (any_nz_s32(vorrq_s32(c_or, d_or))) {
          const uint16x8_t scan_u16 = vreinterpretq_u16_s16(vld1q_s16(scan));
          assert(vmaxvq_u16(scan_u16) < 64);
          const uint8x8_t wt_u8 = vqtbl4_u8(qm, vmovn_u16(scan_u16));
          acc_block_error_qm_8(c0, c1, d0, d1, d_or, wt_u8, &sse0, &sse1, &ssz0,
                               &ssz1);
        }
        coeff += 8;
        dqcoeff += 8;
        scan += 8;
        block_size -= 8;
      } while (block_size != 0);
    }
  } else
#endif  // AOM_ARCH_AARCH64
  {
    do {
      const int32x4_t c0 = vld1q_s32(coeff);
      const int32x4_t c1 = vld1q_s32(coeff + 4);
      const int32x4_t d0 = vld1q_s32(dqcoeff);
      const int32x4_t d1 = vld1q_s32(dqcoeff + 4);
      const int32x4_t c_or = vorrq_s32(c0, c1);
      const int32x4_t d_or = vorrq_s32(d0, d1);
      if (any_nz_s32(vorrq_s32(c_or, d_or))) {
        uint8x8_t wt_u8 = vld1_dup_u8(&qmatrix[scan[0]]);
        wt_u8 = vld1_lane_u8(&qmatrix[scan[1]], wt_u8, 1);
        wt_u8 = vld1_lane_u8(&qmatrix[scan[2]], wt_u8, 2);
        wt_u8 = vld1_lane_u8(&qmatrix[scan[3]], wt_u8, 3);
        wt_u8 = vld1_lane_u8(&qmatrix[scan[4]], wt_u8, 4);
        wt_u8 = vld1_lane_u8(&qmatrix[scan[5]], wt_u8, 5);
        wt_u8 = vld1_lane_u8(&qmatrix[scan[6]], wt_u8, 6);
        wt_u8 = vld1_lane_u8(&qmatrix[scan[7]], wt_u8, 7);
        acc_block_error_qm_8(c0, c1, d0, d1, d_or, wt_u8, &sse0, &sse1, &ssz0,
                             &ssz1);
      }
      coeff += 8;
      dqcoeff += 8;
      scan += 8;
      block_size -= 8;
    } while (block_size != 0);
  }

  const int shift = 2 * (bd - 8);
  const int64_t sse = ROUND_POWER_OF_TWO_64(
      (int64_t)horizontal_add_u64x2(vaddq_u64(sse0, sse1)), shift);
  *ssz = ROUND_POWER_OF_TWO_64(
      (int64_t)horizontal_add_u64x2(vaddq_u64(ssz0, ssz1)), shift);
  return sse;
}
