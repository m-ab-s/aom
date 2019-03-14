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

#include <math.h>

#include "config/av1_rtcd.h"

struct AV1Common;

#define CNN_MAX_HIDDEN_LAYERS 25
#define CNN_MAX_LAYERS (CNN_MAX_HIDDEN_LAYERS + 1)
#define CNN_MAX_CHANNELS 64

enum {
  PADDING_SAME_ZERO,       // tensorflow's SAME padding with pixels outside
                           // the image area assumed to be 0 (default)
  PADDING_SAME_REPLICATE,  // tensorflow's SAME padding with pixels outside
                           // the image area replicated from closest edge
  PADDING_VALID            // tensorflow's VALID padding
} UENUM1BYTE(PADDING_TYPE);

enum { NONE, RELU, SOFTSIGN } UENUM1BYTE(ACTIVATION);

struct CNN_LAYER_CONFIG {
  int in_channels;
  int filter_width;
  int filter_height;
  int out_channels;
  int skip_width;
  int skip_height;
  float *weights;  // array of length filter_height x filter_width x in_channels
                   // x out_channels where the inner-most scan is out_channels
                   // and the outer most scan is filter_height.
  float *bias;     // array of length out_channels
  PADDING_TYPE pad;       // padding type
  ACTIVATION activation;  // the activation function to use after convolution
  int input_copy;  // copy the input tensor to the current layer and store for
                   // future use as a skip addition layer
  int output_add;  // add previously stored tensor to the output tensor of the
                   // current layer
};

struct CNN_CONFIG {
  int num_layers;  // number of CNN layers ( = number of hidden layers + 1)
  int is_residue;  // whether the output activation is a residue
  int ext_width, ext_height;  // extension horizontally and vertically
  int strict_bounds;          // whether the input bounds are strict or not.
                              // If strict, the extension area is filled by
                              // replication; if not strict, image data is
                              // assumed available beyond the bounds.
  CNN_LAYER_CONFIG layer_config[CNN_MAX_LAYERS];
};

void av1_restore_cnn(uint8_t *dgd, int width, int height, int stride,
                     const CNN_CONFIG *cnn_config);
void av1_restore_cnn_highbd(uint16_t *dgd, int width, int height, int stride,
                            const CNN_CONFIG *cnn_config, int bit_depth);
void av1_restore_cnn_plane(struct AV1Common *cm, const CNN_CONFIG *cnn_config,
                           int plane);

static INLINE float softsign(float x) { return x / (fabsf(x) + 1); }

static INLINE float relu(float x) { return (x < 0) ? 0 : x; }

static INLINE float identity(float x) { return x; }

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_CNN_H_
