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

FOR_EACH_TXFM2D(MAKE_HIGHBD_TXFM2D, sse4_1)
MAKE_LOWBD_TXFM2D(4, 4, sse4_1)
MAKE_LOWBD_TXFM2D(8, 8, sse4_1)
MAKE_LOWBD_TXFM2D(16, 16, sse4_1)
MAKE_LOWBD_TXFM2D(32, 32, sse4_1)
MAKE_LOWBD_TXFM2D(4, 8, sse4_1)
MAKE_LOWBD_TXFM2D(8, 16, sse4_1)
MAKE_LOWBD_TXFM2D(32, 16, sse4_1)
MAKE_LOWBD_TXFM2D(4, 16, sse4_1)
MAKE_LOWBD_TXFM2D(16, 4, sse4_1)
MAKE_LOWBD_TXFM2D(8, 32, sse4_1)
MAKE_LOWBD_TXFM2D(32, 8, sse4_1)
MAKE_LOWBD_TXFM2D(64, 16, sse4_1)
