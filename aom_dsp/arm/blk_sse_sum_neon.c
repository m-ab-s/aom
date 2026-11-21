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

#include <arm_neon.h>
#include <assert.h>

#include "config/aom_dsp_rtcd.h"
#include "config/aom_config.h"

#include "aom_dsp/arm/mem_neon.h"
#include "aom_dsp/arm/sum_neon.h"

static inline void get_blk_sse_sum_4xh_neon(const int16_t *data, int stride,
                                            int bh, int *x_sum,
                                            int64_t *x2_sum) {
  int i = bh;
  int32x4_t sum = vdupq_n_s32(0);
  int32x4_t sse = vdupq_n_s32(0);

  do {
    int16x8_t d = vcombine_s16(vld1_s16(data), vld1_s16(data + stride));

    sum = vpadalq_s16(sum, d);

    sse = vmlal_s16(sse, vget_low_s16(d), vget_low_s16(d));
    sse = vmlal_s16(sse, vget_high_s16(d), vget_high_s16(d));

    data += 2 * stride;
    i -= 2;
  } while (i != 0);

  *x_sum = horizontal_add_s32x4(sum);
  *x2_sum = horizontal_long_add_s32x4(sse);
}

static inline void get_blk_sse_sum_8xh_neon(const int16_t *data, int stride,
                                            int bh, int *x_sum,
                                            int64_t *x2_sum) {
  int i = bh;
  int32x4_t sum = vdupq_n_s32(0);
  int32x4_t sse = vdupq_n_s32(0);

  // Input is 12-bit wide, so we can add up to 127 squared elements in a signed
  // 32-bits element. Since we're accumulating into an int32x4_t and the maximum
  // value for bh is 32, we don't have to worry about sse overflowing.

  do {
    int16x8_t d = vld1q_s16(data);

    sum = vpadalq_s16(sum, d);

    sse = vmlal_s16(sse, vget_low_s16(d), vget_low_s16(d));
    sse = vmlal_s16(sse, vget_high_s16(d), vget_high_s16(d));

    data += stride;
  } while (--i != 0);

  *x_sum = horizontal_add_s32x4(sum);
  *x2_sum = horizontal_long_add_s32x4(sse);
}

// Input is a 12-bit pixel difference in [-4095, 4095], so vaddq_s16(d0, d1)
// stays in [-8190, 8190] without int16_t overflow, each int32_t lane of
// sse[0..1] holds up to 128 squared elements without signed overflow
// (128 * 4095^2 = 2,146,435,200 <= INT32_MAX), and summing a lane pair as
// uint32_t stays <= 4,292,870,400 <= UINT32_MAX.
static inline void init_blk_sse_sum_16_neon(const int16_t *data, int32x4_t *sum,
                                            int32x4_t sse[2]) {
  const int16x8_t d0 = vld1q_s16(data);
  const int16x8_t d1 = vld1q_s16(data + 8);

  *sum = vpaddlq_s16(vaddq_s16(d0, d1));

  sse[0] = vmull_s16(vget_low_s16(d0), vget_low_s16(d0));
  sse[1] = vmull_s16(vget_high_s16(d0), vget_high_s16(d0));
  sse[0] = vmlal_s16(sse[0], vget_low_s16(d1), vget_low_s16(d1));
  sse[1] = vmlal_s16(sse[1], vget_high_s16(d1), vget_high_s16(d1));
}

static inline void accumulate_blk_sse_sum_16_neon(const int16_t *data,
                                                  int32x4_t *sum,
                                                  int32x4_t sse[2]) {
  const int16x8_t d0 = vld1q_s16(data);
  const int16x8_t d1 = vld1q_s16(data + 8);

  *sum = vpadalq_s16(*sum, vaddq_s16(d0, d1));

  sse[0] = vmlal_s16(sse[0], vget_low_s16(d0), vget_low_s16(d0));
  sse[1] = vmlal_s16(sse[1], vget_high_s16(d0), vget_high_s16(d0));
  sse[0] = vmlal_s16(sse[0], vget_low_s16(d1), vget_low_s16(d1));
  sse[1] = vmlal_s16(sse[1], vget_high_s16(d1), vget_high_s16(d1));
}

static inline uint32x4_t combine_sse_s32x4_pair_neon(const int32x4_t sse[2]) {
  return vaddq_u32(vreinterpretq_u32_s32(sse[0]),
                   vreinterpretq_u32_s32(sse[1]));
}

static inline int64_t sum_sse_s32x4_pair_neon(const int32x4_t sse[2]) {
  return (int64_t)horizontal_long_add_u32x4(combine_sse_s32x4_pair_neon(sse));
}

static inline void get_blk_sse_sum_16xh_neon(const int16_t *data, int stride,
                                             int bh, int *x_sum,
                                             int64_t *x2_sum) {
  assert(bh > 0 && bh <= 64);
  int32x4_t sum;
  int32x4_t sse_s32[2];
  init_blk_sse_sum_16_neon(data, &sum, sse_s32);

  while (--bh != 0) {
    data += stride;
    accumulate_blk_sse_sum_16_neon(data, &sum, sse_s32);
  }

  *x_sum = horizontal_add_s32x4(sum);
  *x2_sum = sum_sse_s32x4_pair_neon(sse_s32);
}

static inline void get_blk_sse_sum_32xh_neon(const int16_t *data, int stride,
                                             int bh, int *x_sum,
                                             int64_t *x2_sum) {
  assert(bh > 0 && bh <= 32);
  int32x4_t sum;
  int32x4_t sse_s32[2];
  init_blk_sse_sum_16_neon(data, &sum, sse_s32);
  accumulate_blk_sse_sum_16_neon(data + 16, &sum, sse_s32);

  while (--bh != 0) {
    data += stride;
    accumulate_blk_sse_sum_16_neon(data, &sum, sse_s32);
    accumulate_blk_sse_sum_16_neon(data + 16, &sum, sse_s32);
  }

  *x_sum = horizontal_add_s32x4(sum);
  *x2_sum = sum_sse_s32x4_pair_neon(sse_s32);
}

static inline void blk_sse_sum_rows_neon(const int16_t **data, int stride,
                                         int bw, int rows, int32x4_t *sum,
                                         int32x4_t sse[2]) {
  init_blk_sse_sum_16_neon(*data, sum, sse);
  for (int j = 16; j != bw; j += 16) {
    accumulate_blk_sse_sum_16_neon(*data + j, sum, sse);
  }

  while (--rows != 0) {
    *data += stride;
    int j = bw;
    const int16_t *data_ptr = *data;
    do {
      accumulate_blk_sse_sum_16_neon(data_ptr, sum, sse);
      data_ptr += 16;
      j -= 16;
    } while (j != 0);
  }
}

static inline void get_blk_sse_sum_large_neon(const int16_t *data, int stride,
                                              int bw, int bh, int *x_sum,
                                              int64_t *x2_sum) {
  // Each lane of sse_s32[0..1] accumulates bw / 8 squared elements per row, so
  // up to (128 * 8) / bw rows fit before widening to 64 bits.
  const int max_elems = 128 * 8;
  assert(bw >= 16 && bw <= max_elems && (bw & 15) == 0);
  assert(bh > 0);

  const int i_limit = max_elems / bw;
  const int first_rows = bh > i_limit ? i_limit : bh;
  int32x4_t sum;
  int32x4_t sse_s32[2];
  blk_sse_sum_rows_neon(&data, stride, bw, first_rows, &sum, sse_s32);
  bh -= first_rows;

  if (bh == 0) {
    *x_sum = horizontal_add_s32x4(sum);
    *x2_sum = sum_sse_s32x4_pair_neon(sse_s32);
    return;
  }

  uint64x2_t sse = vpaddlq_u32(combine_sse_s32x4_pair_neon(sse_s32));
  do {
    data += stride;
    const int rows = bh > i_limit ? i_limit : bh;
    int32x4_t chunk_sum;
    blk_sse_sum_rows_neon(&data, stride, bw, rows, &chunk_sum, sse_s32);
    sum = vaddq_s32(sum, chunk_sum);
    sse = vpadalq_u32(sse, combine_sse_s32x4_pair_neon(sse_s32));
    bh -= rows;
  } while (bh != 0);

  *x_sum = horizontal_add_s32x4(sum);
  *x2_sum = (int64_t)horizontal_add_u64x2(sse);
}

void aom_get_blk_sse_sum_neon(const int16_t *data, int stride, int bw, int bh,
                              int *x_sum, int64_t *x2_sum) {
  if (bw == 4) {
    get_blk_sse_sum_4xh_neon(data, stride, bh, x_sum, x2_sum);
  } else if (bw == 8) {
    get_blk_sse_sum_8xh_neon(data, stride, bh, x_sum, x2_sum);
  } else if (bw == 16 && bh <= 64) {
    get_blk_sse_sum_16xh_neon(data, stride, bh, x_sum, x2_sum);
  } else if (bw == 32 && bh <= 32) {
    get_blk_sse_sum_32xh_neon(data, stride, bh, x_sum, x2_sum);
  } else {
    get_blk_sse_sum_large_neon(data, stride, bw, bh, x_sum, x2_sum);
  }
}
