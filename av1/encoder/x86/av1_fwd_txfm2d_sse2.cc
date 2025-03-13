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

#include "av1/encoder/av1_fwd_txfm2d_hwy.h"

#if AOM_ARCH_X86
// Include top-level function only for 32-bit x86, to support Valgrind. For
// normal use, we require SSE4.1, so av1_lowbd_fwd_txfm_sse4_1 will be used
// instead of this function. However, 32-bit Valgrind does not support SSE4.1,
// so we include a fallback to SSE2 to improve performance
MAKE_LOWBD_TXFM2D_DISPATCH(sse2)
#endif  // AOM_ARCH_X86
