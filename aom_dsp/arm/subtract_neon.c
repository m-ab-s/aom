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

#include <arm_neon.h>
#include <assert.h>

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"

#include "aom/aom_integer.h"
#include "aom_dsp/arm/mem_neon.h"
#include "aom_ports/mem.h"

void aom_subtract_block_neon(int rows, int cols, int16_t *diff,
                             ptrdiff_t diff_stride, const uint8_t *src,
                             ptrdiff_t src_stride, const uint8_t *pred,
                             ptrdiff_t pred_stride) {
  assert(rows >= 4 && (rows & 3) == 0);
  assert(cols >= 4 && (cols & (cols - 1)) == 0);
  int r = rows;
  if (cols > 16) {
    if (cols == 32) {
      do {
        const uint8x16_t s0_0 = vld1q_u8(&src[0]);
        const uint8x16_t s0_1 = vld1q_u8(&src[16]);
        const uint8x16_t p0_0 = vld1q_u8(&pred[0]);
        const uint8x16_t p0_1 = vld1q_u8(&pred[16]);
        const uint8x16_t s1_0 = vld1q_u8(&src[src_stride + 0]);
        const uint8x16_t s1_1 = vld1q_u8(&src[src_stride + 16]);
        const uint8x16_t p1_0 = vld1q_u8(&pred[pred_stride + 0]);
        const uint8x16_t p1_1 = vld1q_u8(&pred[pred_stride + 16]);
        const int16x8_t d00 = vreinterpretq_s16_u16(
            vsubl_u8(vget_low_u8(s0_0), vget_low_u8(p0_0)));
        const int16x8_t d01 = vreinterpretq_s16_u16(
            vsubl_u8(vget_high_u8(s0_0), vget_high_u8(p0_0)));
        const int16x8_t d02 = vreinterpretq_s16_u16(
            vsubl_u8(vget_low_u8(s0_1), vget_low_u8(p0_1)));
        const int16x8_t d03 = vreinterpretq_s16_u16(
            vsubl_u8(vget_high_u8(s0_1), vget_high_u8(p0_1)));
        int16_t *diff1 = diff + diff_stride;
        const int16x8_t d10 = vreinterpretq_s16_u16(
            vsubl_u8(vget_low_u8(s1_0), vget_low_u8(p1_0)));
        const int16x8_t d11 = vreinterpretq_s16_u16(
            vsubl_u8(vget_high_u8(s1_0), vget_high_u8(p1_0)));
        const int16x8_t d12 = vreinterpretq_s16_u16(
            vsubl_u8(vget_low_u8(s1_1), vget_low_u8(p1_1)));
        const int16x8_t d13 = vreinterpretq_s16_u16(
            vsubl_u8(vget_high_u8(s1_1), vget_high_u8(p1_1)));
        vst1q_s16(&diff[0], d00);
        vst1q_s16(&diff[8], d01);
        vst1q_s16(&diff[16], d02);
        vst1q_s16(&diff[24], d03);
        vst1q_s16(&diff1[0], d10);
        vst1q_s16(&diff1[8], d11);
        vst1q_s16(&diff1[16], d12);
        vst1q_s16(&diff1[24], d13);
        diff += diff_stride << 1;
        pred += pred_stride << 1;
        src += src_stride << 1;
        r -= 2;
      } while (r != 0);
    } else {
      do {
        int c = 0;
        do {
          const uint8x16_t v_src_00 = vld1q_u8(&src[c + 0]);
          const uint8x16_t v_src_16 = vld1q_u8(&src[c + 16]);
          const uint8x16_t v_pred_00 = vld1q_u8(&pred[c + 0]);
          const uint8x16_t v_pred_16 = vld1q_u8(&pred[c + 16]);
          const uint16x8_t v_diff_lo_00 =
              vsubl_u8(vget_low_u8(v_src_00), vget_low_u8(v_pred_00));
          const uint16x8_t v_diff_hi_00 =
              vsubl_u8(vget_high_u8(v_src_00), vget_high_u8(v_pred_00));
          const uint16x8_t v_diff_lo_16 =
              vsubl_u8(vget_low_u8(v_src_16), vget_low_u8(v_pred_16));
          const uint16x8_t v_diff_hi_16 =
              vsubl_u8(vget_high_u8(v_src_16), vget_high_u8(v_pred_16));
          vst1q_s16(&diff[c + 0], vreinterpretq_s16_u16(v_diff_lo_00));
          vst1q_s16(&diff[c + 8], vreinterpretq_s16_u16(v_diff_hi_00));
          vst1q_s16(&diff[c + 16], vreinterpretq_s16_u16(v_diff_lo_16));
          vst1q_s16(&diff[c + 24], vreinterpretq_s16_u16(v_diff_hi_16));
          c += 32;
        } while (c < cols);
        diff += diff_stride;
        pred += pred_stride;
        src += src_stride;
      } while (--r != 0);
    }
  } else if (cols > 8) {
    do {
      uint8x16_t v_src0, v_src1, v_src2, v_src3;
      uint8x16_t v_pred0, v_pred1, v_pred2, v_pred3;
      load_u8_16x4(src, src_stride, &v_src0, &v_src1, &v_src2, &v_src3);
      load_u8_16x4(pred, pred_stride, &v_pred0, &v_pred1, &v_pred2, &v_pred3);
      const int16x8_t v_diff0_lo = vreinterpretq_s16_u16(
          vsubl_u8(vget_low_u8(v_src0), vget_low_u8(v_pred0)));
      const int16x8_t v_diff0_hi = vreinterpretq_s16_u16(
          vsubl_u8(vget_high_u8(v_src0), vget_high_u8(v_pred0)));
      const int16x8_t v_diff1_lo = vreinterpretq_s16_u16(
          vsubl_u8(vget_low_u8(v_src1), vget_low_u8(v_pred1)));
      const int16x8_t v_diff1_hi = vreinterpretq_s16_u16(
          vsubl_u8(vget_high_u8(v_src1), vget_high_u8(v_pred1)));
      const int16x8_t v_diff2_lo = vreinterpretq_s16_u16(
          vsubl_u8(vget_low_u8(v_src2), vget_low_u8(v_pred2)));
      const int16x8_t v_diff2_hi = vreinterpretq_s16_u16(
          vsubl_u8(vget_high_u8(v_src2), vget_high_u8(v_pred2)));
      const int16x8_t v_diff3_lo = vreinterpretq_s16_u16(
          vsubl_u8(vget_low_u8(v_src3), vget_low_u8(v_pred3)));
      const int16x8_t v_diff3_hi = vreinterpretq_s16_u16(
          vsubl_u8(vget_high_u8(v_src3), vget_high_u8(v_pred3)));
      vst1q_s16(diff, v_diff0_lo);
      vst1q_s16(diff + 8, v_diff0_hi);
      diff += diff_stride;
      vst1q_s16(diff, v_diff1_lo);
      vst1q_s16(diff + 8, v_diff1_hi);
      diff += diff_stride;
      vst1q_s16(diff, v_diff2_lo);
      vst1q_s16(diff + 8, v_diff2_hi);
      diff += diff_stride;
      vst1q_s16(diff, v_diff3_lo);
      vst1q_s16(diff + 8, v_diff3_hi);
      diff += diff_stride;
      pred += 4 * pred_stride;
      src += 4 * src_stride;
      r -= 4;
    } while (r != 0);
  } else if (cols > 4) {
    do {
      uint8x8_t v_src0, v_src1, v_src2, v_src3;
      uint8x8_t v_pred0, v_pred1, v_pred2, v_pred3;
      load_u8_8x4(src, src_stride, &v_src0, &v_src1, &v_src2, &v_src3);
      load_u8_8x4(pred, pred_stride, &v_pred0, &v_pred1, &v_pred2, &v_pred3);
      const int16x8_t v_diff0 =
          vreinterpretq_s16_u16(vsubl_u8(v_src0, v_pred0));
      const int16x8_t v_diff1 =
          vreinterpretq_s16_u16(vsubl_u8(v_src1, v_pred1));
      const int16x8_t v_diff2 =
          vreinterpretq_s16_u16(vsubl_u8(v_src2, v_pred2));
      const int16x8_t v_diff3 =
          vreinterpretq_s16_u16(vsubl_u8(v_src3, v_pred3));
      store_s16_8x4(diff, diff_stride, v_diff0, v_diff1, v_diff2, v_diff3);
      diff += 4 * diff_stride;
      pred += 4 * pred_stride;
      src += 4 * src_stride;
      r -= 4;
    } while (r != 0);
  } else {
    assert(cols == 4);
    do {
      uint8x8_t v_src_01, v_src_23, v_pred_01, v_pred_23;
      load_unaligned_u8_4x4(src, src_stride, &v_src_01, &v_src_23);
      load_unaligned_u8_4x4(pred, pred_stride, &v_pred_01, &v_pred_23);
      const int16x8_t v_diff_01 =
          vreinterpretq_s16_u16(vsubl_u8(v_src_01, v_pred_01));
      const int16x8_t v_diff_23 =
          vreinterpretq_s16_u16(vsubl_u8(v_src_23, v_pred_23));
      store_s16_4x4(diff, diff_stride, vget_low_s16(v_diff_01),
                    vget_high_s16(v_diff_01), vget_low_s16(v_diff_23),
                    vget_high_s16(v_diff_23));
      diff += diff_stride << 2;
      pred += pred_stride << 2;
      src += src_stride << 2;
      r -= 4;
    } while (r != 0);
  }
}

#if CONFIG_AV1_HIGHBITDEPTH
void aom_highbd_subtract_block_neon(int rows, int cols, int16_t *diff,
                                    ptrdiff_t diff_stride, const uint8_t *src8,
                                    ptrdiff_t src_stride, const uint8_t *pred8,
                                    ptrdiff_t pred_stride) {
  uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  uint16_t *pred = CONVERT_TO_SHORTPTR(pred8);

  if (cols > 16) {
    int r = rows;
    do {
      int c = 0;
      do {
        const uint16x8_t v_src_00 = vld1q_u16(&src[c + 0]);
        const uint16x8_t v_pred_00 = vld1q_u16(&pred[c + 0]);
        const uint16x8_t v_diff_00 = vsubq_u16(v_src_00, v_pred_00);
        const uint16x8_t v_src_08 = vld1q_u16(&src[c + 8]);
        const uint16x8_t v_pred_08 = vld1q_u16(&pred[c + 8]);
        const uint16x8_t v_diff_08 = vsubq_u16(v_src_08, v_pred_08);
        const uint16x8_t v_src_16 = vld1q_u16(&src[c + 16]);
        const uint16x8_t v_pred_16 = vld1q_u16(&pred[c + 16]);
        const uint16x8_t v_diff_16 = vsubq_u16(v_src_16, v_pred_16);
        const uint16x8_t v_src_24 = vld1q_u16(&src[c + 24]);
        const uint16x8_t v_pred_24 = vld1q_u16(&pred[c + 24]);
        const uint16x8_t v_diff_24 = vsubq_u16(v_src_24, v_pred_24);
        vst1q_s16(&diff[c + 0], vreinterpretq_s16_u16(v_diff_00));
        vst1q_s16(&diff[c + 8], vreinterpretq_s16_u16(v_diff_08));
        vst1q_s16(&diff[c + 16], vreinterpretq_s16_u16(v_diff_16));
        vst1q_s16(&diff[c + 24], vreinterpretq_s16_u16(v_diff_24));
        c += 32;
      } while (c < cols);
      diff += diff_stride;
      pred += pred_stride;
      src += src_stride;
    } while (--r != 0);
  } else if (cols > 8) {
    int r = rows;
    while (r >= 2) {
      const uint16x8_t v_src_00 = vld1q_u16(&src[0]);
      const uint16x8_t v_pred_00 = vld1q_u16(&pred[0]);
      const uint16x8_t v_src_08 = vld1q_u16(&src[8]);
      const uint16x8_t v_pred_08 = vld1q_u16(&pred[8]);
      const uint16x8_t v_src_10 = vld1q_u16(&src[src_stride + 0]);
      const uint16x8_t v_pred_10 = vld1q_u16(&pred[pred_stride + 0]);
      const uint16x8_t v_src_18 = vld1q_u16(&src[src_stride + 8]);
      const uint16x8_t v_pred_18 = vld1q_u16(&pred[pred_stride + 8]);
      vst1q_s16(&diff[0],
                vreinterpretq_s16_u16(vsubq_u16(v_src_00, v_pred_00)));
      vst1q_s16(&diff[8],
                vreinterpretq_s16_u16(vsubq_u16(v_src_08, v_pred_08)));
      vst1q_s16(&diff[diff_stride + 0],
                vreinterpretq_s16_u16(vsubq_u16(v_src_10, v_pred_10)));
      vst1q_s16(&diff[diff_stride + 8],
                vreinterpretq_s16_u16(vsubq_u16(v_src_18, v_pred_18)));
      diff += diff_stride << 1;
      pred += pred_stride << 1;
      src += src_stride << 1;
      r -= 2;
    }
    if (r != 0) {
      const uint16x8_t v_src_00 = vld1q_u16(&src[0]);
      const uint16x8_t v_pred_00 = vld1q_u16(&pred[0]);
      const uint16x8_t v_src_08 = vld1q_u16(&src[8]);
      const uint16x8_t v_pred_08 = vld1q_u16(&pred[8]);
      vst1q_s16(&diff[0],
                vreinterpretq_s16_u16(vsubq_u16(v_src_00, v_pred_00)));
      vst1q_s16(&diff[8],
                vreinterpretq_s16_u16(vsubq_u16(v_src_08, v_pred_08)));
    }
  } else if (cols > 4) {
    int r = rows;
    do {
      const uint16x8_t v_src_r0 = vld1q_u16(&src[0]);
      const uint16x8_t v_src_r1 = vld1q_u16(&src[src_stride]);
      const uint16x8_t v_pred_r0 = vld1q_u16(&pred[0]);
      const uint16x8_t v_pred_r1 = vld1q_u16(&pred[pred_stride]);
      const uint16x8_t v_diff_r0 = vsubq_u16(v_src_r0, v_pred_r0);
      const uint16x8_t v_diff_r1 = vsubq_u16(v_src_r1, v_pred_r1);
      vst1q_s16(&diff[0], vreinterpretq_s16_u16(v_diff_r0));
      vst1q_s16(&diff[diff_stride], vreinterpretq_s16_u16(v_diff_r1));
      diff += diff_stride << 1;
      pred += pred_stride << 1;
      src += src_stride << 1;
      r -= 2;
    } while (r != 0);
  } else {
    int r = rows;
    while (r >= 4) {
      const uint16x8_t s01 =
          vcombine_u16(vld1_u16(&src[0]), vld1_u16(&src[src_stride]));
      const uint16x8_t p01 =
          vcombine_u16(vld1_u16(&pred[0]), vld1_u16(&pred[pred_stride]));
      const uint16x8_t s23 = vcombine_u16(vld1_u16(&src[2 * src_stride]),
                                          vld1_u16(&src[3 * src_stride]));
      const uint16x8_t p23 = vcombine_u16(vld1_u16(&pred[2 * pred_stride]),
                                          vld1_u16(&pred[3 * pred_stride]));
      const int16x8_t d01 = vreinterpretq_s16_u16(vsubq_u16(s01, p01));
      const int16x8_t d23 = vreinterpretq_s16_u16(vsubq_u16(s23, p23));
      vst1_s16(&diff[0], vget_low_s16(d01));
      vst1_s16(&diff[diff_stride], vget_high_s16(d01));
      vst1_s16(&diff[2 * diff_stride], vget_low_s16(d23));
      vst1_s16(&diff[3 * diff_stride], vget_high_s16(d23));
      diff += diff_stride << 2;
      pred += pred_stride << 2;
      src += src_stride << 2;
      r -= 4;
    }
    while (r >= 2) {
      const uint16x4_t v_src_r0 = vld1_u16(&src[0]);
      const uint16x4_t v_src_r1 = vld1_u16(&src[src_stride]);
      const uint16x4_t v_pred_r0 = vld1_u16(&pred[0]);
      const uint16x4_t v_pred_r1 = vld1_u16(&pred[pred_stride]);
      const uint16x4_t v_diff_r0 = vsub_u16(v_src_r0, v_pred_r0);
      const uint16x4_t v_diff_r1 = vsub_u16(v_src_r1, v_pred_r1);
      vst1_s16(&diff[0], vreinterpret_s16_u16(v_diff_r0));
      vst1_s16(&diff[diff_stride], vreinterpret_s16_u16(v_diff_r1));
      diff += diff_stride << 1;
      pred += pred_stride << 1;
      src += src_stride << 1;
      r -= 2;
    }
  }
}
#endif  // CONFIG_AV1_HIGHBITDEPTH
