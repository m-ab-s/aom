/*
 * Copyright (c) 2023, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <arm_neon.h>

#include "aom_dsp/arm/dot_sve.h"
#include "aom_dsp/arm/mem_neon.h"
#include "config/aom_dsp_rtcd.h"

static INLINE uint64_t aom_sum_sse_2d_i16_4xh_sve(const int16_t *src,
                                                  int stride, int height,
                                                  int *sum) {
  int64x2_t sse = vdupq_n_s32(0);
  int32x4_t sum_s32 = vdupq_n_s32(0);

  do {
    int16x8_t s = vcombine_s16(vld1_s16(src), vld1_s16(src + stride));

    sse = aom_sdotq_s16(sse, s, s);

    sum_s32 = vpadalq_s16(sum_s32, s);

    src += 2 * stride;
    height -= 2;
  } while (height != 0);

  *sum += vaddvq_s32(sum_s32);
  return vaddvq_s64(sse);
}

static INLINE uint64_t aom_sum_sse_2d_i16_8xh_sve(const int16_t *src,
                                                  int stride, int height,
                                                  int *sum) {
  int64x2_t sse[2] = { vdupq_n_s64(0), vdupq_n_s64(0) };
  int32x4_t sum_acc[2] = { vdupq_n_s32(0), vdupq_n_s32(0) };

  do {
    int16x8_t s0 = vld1q_s16(src);
    int16x8_t s1 = vld1q_s16(src + stride);

    sse[0] = aom_sdotq_s16(sse[0], s0, s0);
    sse[1] = aom_sdotq_s16(sse[1], s1, s1);

    sum_acc[0] = vpadalq_s16(sum_acc[0], s0);
    sum_acc[1] = vpadalq_s16(sum_acc[1], s1);

    src += 2 * stride;
    height -= 2;
  } while (height != 0);

  *sum += vaddvq_s32(vaddq_s32(sum_acc[0], sum_acc[1]));
  return vaddvq_s64(vaddq_s64(sse[0], sse[1]));
}

static INLINE uint64_t aom_sum_sse_2d_i16_16xh_sve(const int16_t *src,
                                                   int stride, int width,
                                                   int height, int *sum) {
  int64x2_t sse[2] = { vdupq_n_s64(0), vdupq_n_s64(0) };
  int32x4_t sum_acc[2] = { vdupq_n_s32(0), vdupq_n_s32(0) };

  do {
    int w = 0;
    do {
      int16x8_t s0 = vld1q_s16(src + w);
      int16x8_t s1 = vld1q_s16(src + w + 8);

      sse[0] = aom_sdotq_s16(sse[0], s0, s0);
      sse[1] = aom_sdotq_s16(sse[1], s1, s1);

      sum_acc[0] = vpadalq_s16(sum_acc[0], s0);
      sum_acc[1] = vpadalq_s16(sum_acc[1], s1);

      w += 16;
    } while (w < width);

    src += stride;
  } while (--height != 0);

  *sum += vaddvq_s32(vaddq_s32(sum_acc[0], sum_acc[1]));
  return vaddvq_s64(vaddq_s64(sse[0], sse[1]));
}

uint64_t aom_sum_sse_2d_i16_sve(const int16_t *src, int stride, int width,
                                int height, int *sum) {
  uint64_t sse;

  if (width == 4) {
    sse = aom_sum_sse_2d_i16_4xh_sve(src, stride, height, sum);
  } else if (width == 8) {
    sse = aom_sum_sse_2d_i16_8xh_sve(src, stride, height, sum);
  } else if (width % 16 == 0) {
    sse = aom_sum_sse_2d_i16_16xh_sve(src, stride, width, height, sum);
  } else {
    sse = aom_sum_sse_2d_i16_c(src, stride, width, height, sum);
  }

  return sse;
}
