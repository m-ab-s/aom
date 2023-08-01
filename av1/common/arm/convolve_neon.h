/*
 *  Copyright (c) 2018, Alliance for Open Media. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef AOM_AV1_COMMON_ARM_CONVOLVE_NEON_H_
#define AOM_AV1_COMMON_ARM_CONVOLVE_NEON_H_

#include <arm_neon.h>

#include "config/aom_config.h"

#define HORIZ_EXTRA_ROWS ((SUBPEL_TAPS + 7) & ~0x07)

// clang versions < 16 did not include the dotprod feature for Arm architecture
// versions that should have it by default, e.g., armv8.6-a.
#if AOM_ARCH_AARCH64 && \
    (defined(__ARM_FEATURE_DOTPROD) || defined(__ARM_FEATURE_MATMUL_INT8))

DECLARE_ALIGNED(16, static const uint8_t, dot_prod_permute_tbl[48]) = {
  0, 1, 2,  3,  1, 2,  3,  4,  2,  3,  4,  5,  3,  4,  5,  6,
  4, 5, 6,  7,  5, 6,  7,  8,  6,  7,  8,  9,  7,  8,  9,  10,
  8, 9, 10, 11, 9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14
};

#endif  // AOM_ARCH_AARCH64 && defined(__ARM_FEATURE_DOTPROD)

#endif  // AOM_AV1_COMMON_ARM_CONVOLVE_NEON_H_
