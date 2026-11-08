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

#include "av1/encoder/wedge_utils_hwy.h"

MAKE_AV1_WEDGE_SSE_FROM_RESIDUALS(avx512)
MAKE_AV1_WEDGE_SIGN_FROM_RESIDUALS(avx512)
MAKE_AV1_WEDGE_COMPUTE_DELTA_SQUARES(avx512)
