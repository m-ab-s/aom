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

#ifndef AOM_AV1_COMMON_CNN_H_
#define AOM_AV1_COMMON_CNN_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "config/av1_rtcd.h"

#define CNN_MAX_HIDDEN_LAYERS 25
#define CNN_MAX_LAYERS (CNN_MAX_HIDDEN_LAYERS + 1)
#define CNN_MAX_CHANNELS 64

struct CNN_LAYER_CONFIG {
  int in_channels;
  int filter_width;
  int filter_height;
  int out_channels;
  int skip_width;
  int skip_height;
  float *weights;  // array of length filter_width x filter_height x in_channels
                   // x out_channels
  float *bias;     // array of length out_channels
};

struct CNN_CONFIG {
  int num_layers;
  CNN_LAYER_CONFIG layer_config[CNN_MAX_LAYERS];
};

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_CNN_H_
