/*
 *  Copyright (c) 2023 The WebM project authors. All Rights Reserved.
 *  Copyright (c) 2023, Alliance for Open Media. All Rights Reserved.
 *
 *  This source code is subject to the terms of the BSD 2 Clause License and
 *  the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 *  was not distributed with this source code in the LICENSE file, you can
 *  obtain it at www.aomedia.org/license/software. If the Alliance for Open
 *  Media Patent License 1.0 was not distributed with this source code in the
 *  PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <arm_neon.h>

#include "config/aom_dsp_rtcd.h"
#include "aom/aom_integer.h"
#include "aom_dsp/arm/mem_neon.h"
#include "aom_dsp/arm/sum_neon.h"
#include "aom_ports/mem.h"

uint32_t aom_highbd_avg_4x4_neon(const uint8_t *a, int a_stride) {
  const uint16_t *a_ptr = CONVERT_TO_SHORTPTR(a);
  uint16x4_t sum, a0, a1, a2, a3;

  load_u16_4x4(a_ptr, a_stride, &a0, &a1, &a2, &a3);

  sum = vadd_u16(a0, a1);
  sum = vadd_u16(sum, a2);
  sum = vadd_u16(sum, a3);

  return (horizontal_add_u16x4(sum) + (1 << 3)) >> 4;
}

uint32_t aom_highbd_avg_8x8_neon(const uint8_t *a, int a_stride) {
  const uint16_t *a_ptr = CONVERT_TO_SHORTPTR(a);
  uint16x8_t sum, a0, a1, a2, a3, a4, a5, a6, a7;

  load_u16_8x8(a_ptr, a_stride, &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7);

  sum = vaddq_u16(a0, a1);
  sum = vaddq_u16(sum, a2);
  sum = vaddq_u16(sum, a3);
  sum = vaddq_u16(sum, a4);
  sum = vaddq_u16(sum, a5);
  sum = vaddq_u16(sum, a6);
  sum = vaddq_u16(sum, a7);

  return (horizontal_add_u16x8(sum) + (1 << 5)) >> 6;
}
