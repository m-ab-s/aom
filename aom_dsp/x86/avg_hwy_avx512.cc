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

#include "aom_dsp/avg_hwy.h"

MAKE_HADAMARD(16, 16, avx512)
MAKE_HADAMARD(32, 32, avx512)
MAKE_HADAMARD_LP(16, 16, avx512)
MAKE_SATD_INT_PRO_FUNCS(avx512)

MAKE_HIGHBD_HADAMARD(16, 16, avx512)
MAKE_HIGHBD_HADAMARD(32, 32, avx512)
