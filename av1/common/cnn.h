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
#define CNN_MAX_BRANCHES 4

enum {
  PADDING_SAME_ZERO,       // tensorflow's SAME padding with pixels outside
                           // the image area assumed to be 0 (default)
  PADDING_SAME_REPLICATE,  // tensorflow's SAME padding with pixels outside
                           // the image area replicated from closest edge
  PADDING_VALID            // tensorflow's VALID padding
} UENUM1BYTE(PADDING_TYPE);

enum { NONE, RELU, SOFTSIGN } UENUM1BYTE(ACTIVATION);

// Types of combining branches with output of current layer:
// BRANCH_NOC: no branch combining
// BRANCH_ADD: Add previously stored branch tensor to output of layer
// BRANCH_CAT: Concatenate branch tensor to output of layer
enum { BRANCH_NOC, BRANCH_ADD, BRANCH_CAT } UENUM1BYTE(BRANCH_COMBINE);

struct CNN_LAYER_CONFIG {
  int branch;      // branch index in [0, CNN_MAX_BRANCHES - 1], where
                   // 0 refers to the primary branch.
  int deconvolve;  // whether this is a deconvolution layer.
                   // 0: If skip_width or skip_height are > 1, then we
                   // reduce resolution
                   // 1: If skip_width or skip_height are > 1, then we
                   // increase resolution
  int in_channels;
  int filter_width;
  int filter_height;
  int out_channels;
  int skip_width;
  int skip_height;
  int maxpool;     // whether to use maxpool or not (only effective when
                   // skip width or skip_height are > 1)
  float *weights;  // array of length filter_height x filter_width x in_channels
                   // x out_channels where the inner-most scan is out_channels
                   // and the outer most scan is filter_height.
  float *bias;     // array of length out_channels
  PADDING_TYPE pad;       // padding type
  ACTIVATION activation;  // the activation function to use after convolution
  int input_to_branch;  // copy the input tensor to the current layer and store
                        // for future use as secondary branch with the given
                        // number. The branch number can be [1, CNN_MAX_BRANCHES
                        // - 1].
  BRANCH_COMBINE branch_combine_type;
  int branch_to_combine;  // index of the branch to combine with if
                          // branch_combine_type != BRANCH_NOC
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

void av1_find_cnn_output_size(int in_width, int in_height,
                              const CNN_CONFIG *cnn_config, int *out_width,
                              int *out_height);
void av1_restore_cnn(uint8_t *dgd, int width, int height, int stride,
                     const CNN_CONFIG *cnn_config);
void av1_restore_cnn_highbd(uint16_t *dgd, int width, int height, int stride,
                            const CNN_CONFIG *cnn_config, int bit_depth);
void av1_restore_cnn_plane(struct AV1Common *cm, const CNN_CONFIG *cnn_config,
                           int plane);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_CNN_H_
