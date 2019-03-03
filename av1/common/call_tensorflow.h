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

#include <stdint.h>

#include "av1/common/blockd.h"

#ifndef CALL_TENSORFLOW
#define CALL_TENSORFLOW

int init_python();
int finish_python();
uint8_t **call_tensorflow(uint8_t *ppp, int height, int width, int stride,
                          FRAME_TYPE frame_type);
void block_call_tensorflow(uint8_t **buf, uint8_t *ppp, int cur_buf_height,
                           int cur_buf_width, int stride,
                           FRAME_TYPE frame_type);
uint16_t **call_tensorflow_hbd(uint16_t *ppp, int height, int width, int stride,
                               FRAME_TYPE frame_type);
uint16_t **block_call_tensorflow_hbd(uint16_t *ppp, int cur_buf_height,
                                     int cur_buf_width, int stride,
                                     FRAME_TYPE frame_type);

#endif  // CALL_TENSORFLOW
