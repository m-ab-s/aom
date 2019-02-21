/*
 * Copyright (c) 2019, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#ifndef ADDITIONHANDLE_FRAME
#define ADDITIONHANDLE_FRAME

#include <stdint.h>

#include "av1/common/onyxc_int.h"

#include "av1/encoder/blockd.h"

extern "C" void additionHandle_frame(AV1_COMP *cpi, AV1_COMMON *cm,
                                     FRAME_TYPE frame_type);
extern "C" void additionHandle_blocks(AV1_COMP *cpi, AV1_COMMON *cm,
                                      FRAME_TYPE frame_type);

extern "C" uint8_t **blocks_to_cnn_secondly(uint8_t *pBuffer_y, int height,
                                            int width, int stride,
                                            FRAME_TYPE frame_type);

#endif  // ADDITIONHANDLE_FRAME
