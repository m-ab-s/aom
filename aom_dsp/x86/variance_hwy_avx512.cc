/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#define HWY_BASELINE_TARGETS HWY_AVX3_DL
#define HWY_BROKEN_32BIT 0

#include "aom_dsp/variance_hwy.h"

FOR_EACH_VAR_BLOCK_SIZE(FVAR, avx512)
FMSE16X16(avx512)
FGET_BLK_SSE_SUM(avx512)
FCALC_VAR_STAT(avx512)
