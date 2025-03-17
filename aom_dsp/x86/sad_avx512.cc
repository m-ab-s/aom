/*
 * Copyright (c) 2025, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include "aom_dsp/sad_hwy.h"

FOR_EACH_SAD_BLOCK_SIZE(FSAD, avx512)
FOR_EACH_SAD_BLOCK_SIZE(FSADSKIP, avx512)
FOR_EACH_SAD_BLOCK_SIZE(FSADAVG, avx512)
FOR_EACH_SAD_BLOCK_SIZE(FSAD4D, avx512)
