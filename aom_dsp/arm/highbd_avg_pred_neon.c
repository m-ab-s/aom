/*
 * Copyright (c) 2023 The WebM project authors. All Rights Reserved.
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
#include <assert.h>

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"

#include "aom_dsp/arm/mem_neon.h"
#include "aom_dsp/blend.h"

void aom_highbd_comp_avg_pred_neon(uint8_t *comp_pred8, const uint8_t *pred8,
                                   int width, int height, const uint8_t *ref8,
                                   int ref_stride) {
  const uint16_t *pred = CONVERT_TO_SHORTPTR(pred8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  uint16_t *comp_pred = CONVERT_TO_SHORTPTR(comp_pred8);

  int i = height;
  if (width > 8) {
    do {
      int j = 0;
      do {
        const uint16x8_t p = vld1q_u16(pred + j);
        const uint16x8_t r = vld1q_u16(ref + j);

        uint16x8_t avg = vrhaddq_u16(p, r);
        vst1q_u16(comp_pred + j, avg);

        j += 8;
      } while (j < width);

      comp_pred += width;
      pred += width;
      ref += ref_stride;
    } while (--i != 0);
  } else if (width == 8) {
    do {
      const uint16x8_t p = vld1q_u16(pred);
      const uint16x8_t r = vld1q_u16(ref);

      uint16x8_t avg = vrhaddq_u16(p, r);
      vst1q_u16(comp_pred, avg);

      comp_pred += width;
      pred += width;
      ref += ref_stride;
    } while (--i != 0);
  } else {
    assert(width == 4);
    do {
      const uint16x4_t p = vld1_u16(pred);
      const uint16x4_t r = vld1_u16(ref);

      uint16x4_t avg = vrhadd_u16(p, r);
      vst1_u16(comp_pred, avg);

      comp_pred += width;
      pred += width;
      ref += ref_stride;
    } while (--i != 0);
  }
}

void aom_highbd_comp_mask_pred_neon(uint8_t *comp_pred8, const uint8_t *pred8,
                                    int width, int height, const uint8_t *ref8,
                                    int ref_stride, const uint8_t *mask,
                                    int mask_stride, int invert_mask) {
  uint16_t *pred = CONVERT_TO_SHORTPTR(pred8);
  uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  uint16_t *comp_pred = CONVERT_TO_SHORTPTR(comp_pred8);

  const uint16_t *src0 = invert_mask ? pred : ref;
  const uint16_t *src1 = invert_mask ? ref : pred;
  const int src_stride0 = invert_mask ? width : ref_stride;
  const int src_stride1 = invert_mask ? ref_stride : width;

  if (width >= 8) {
    const uint16x8_t max_alpha = vdupq_n_u16(AOM_BLEND_A64_MAX_ALPHA);
    do {
      int j = 0;

      do {
        const uint16x8_t s0 = vld1q_u16(src0 + j);
        const uint16x8_t s1 = vld1q_u16(src1 + j);
        const uint16x8_t m0 = vmovl_u8(vld1_u8(mask + j));

        uint16x8_t m0_inv = vsubq_u16(max_alpha, m0);

        uint32x4_t blend_u32_lo = vmull_u16(vget_low_u16(s0), vget_low_u16(m0));
        uint32x4_t blend_u32_hi =
            vmull_u16(vget_high_u16(s0), vget_high_u16(m0));

        blend_u32_lo =
            vmlal_u16(blend_u32_lo, vget_low_u16(s1), vget_low_u16(m0_inv));
        blend_u32_hi =
            vmlal_u16(blend_u32_hi, vget_high_u16(s1), vget_high_u16(m0_inv));

        uint16x4_t blend_u16_lo =
            vrshrn_n_u32(blend_u32_lo, AOM_BLEND_A64_ROUND_BITS);
        uint16x4_t blend_u16_hi =
            vrshrn_n_u32(blend_u32_hi, AOM_BLEND_A64_ROUND_BITS);
        uint16x8_t blend_u16 = vcombine_u16(blend_u16_lo, blend_u16_hi);

        vst1q_u16(comp_pred + j, blend_u16);

        j += 8;
      } while (j < width);

      src0 += src_stride0;
      src1 += src_stride1;
      mask += mask_stride;
      comp_pred += width;
    } while (--height != 0);
  } else {
    assert(width == 4);
    const uint16x4_t max_alpha = vdup_n_u16(AOM_BLEND_A64_MAX_ALPHA);

    do {
      const uint16x4_t s0 = vld1_u16(src0);
      const uint16x4_t s1 = vld1_u16(src1);
      const uint16x4_t m0 = vget_low_u16(vmovl_u8(load_unaligned_u8_4x1(mask)));

      uint16x4_t m0_inv = vsub_u16(max_alpha, m0);
      uint32x4_t blend_u32 = vmull_u16(s0, m0);
      blend_u32 = vmlal_u16(blend_u32, s1, m0_inv);

      uint16x4_t blend_u16 = vrshrn_n_u32(blend_u32, AOM_BLEND_A64_ROUND_BITS);

      vst1_u16(comp_pred, blend_u16);

      src0 += src_stride0;
      src1 += src_stride1;
      mask += mask_stride;
      comp_pred += 4;
    } while (--height != 0);
  }
}
