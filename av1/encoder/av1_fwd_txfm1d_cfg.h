/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#ifndef AOM_AV1_ENCODER_AV1_FWD_TXFM1D_CFG_H_
#define AOM_AV1_ENCODER_AV1_FWD_TXFM1D_CFG_H_
#include "av1/common/enums.h"
#include "av1/encoder/av1_fwd_txfm1d.h"

static const int8_t av1_fwd_txfm_shift_ls[TX_SIZES_ALL][3] = {
  { 2, 0, 0 },    // TX_4X4
  { 2, -1, 0 },   // TX_8X8
  { 2, -2, 0 },   // TX_16X16
  { 2, -4, 0 },   // TX_32X32
  { 0, -2, -2 },  // TX_64X64
  { 2, -1, 0 },   // TX_4X8
  { 2, -1, 0 },   // TX_8X4
  { 2, -2, 0 },   // TX_8X16
  { 2, -2, 0 },   // TX_16X8
  { 2, -4, 0 },   // TX_16X32
  { 2, -4, 0 },   // TX_32X16
  { 0, -2, -2 },  // TX_32X64
  { 2, -4, -2 },  // TX_64X32
  { 2, -1, 0 },   // TX_4X16
  { 2, -1, 0 },   // TX_16X4
  { 2, -2, 0 },   // TX_8X32
  { 2, -2, 0 },   // TX_32X8
  { 0, -2, 0 },   // TX_16X64
  { 2, -4, 0 },   // TX_64X16
};

static const int8_t av1_fwd_cos_bit_col[5][5] = { { 13, 13, 13, 0, 0 },
                                                  { 13, 13, 13, 12, 0 },
                                                  { 13, 13, 13, 12, 13 },
                                                  { 0, 13, 13, 12, 13 },
                                                  { 0, 0, 13, 12, 13 } };

static const int8_t av1_fwd_cos_bit_row[5][5] = { { 13, 13, 12, 0, 0 },
                                                  { 13, 13, 13, 12, 0 },
                                                  { 13, 13, 12, 13, 12 },
                                                  { 0, 12, 13, 12, 11 },
                                                  { 0, 0, 12, 11, 10 } };

#endif  // AOM_AV1_ENCODER_AV1_FWD_TXFM1D_CFG_H_
