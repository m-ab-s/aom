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
