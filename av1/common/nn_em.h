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

#ifndef AOM_AV1_COMMON_NN_EM_H_
#define AOM_AV1_COMMON_NN_EM_H_

#include "config/aom_config.h"

#include "aom_ports/mem.h"

#if CONFIG_INTRA_ENTROPY
enum { ACTN_NONE, ACTN_RELU, ACTN_SIGMOID } UENUM1BYTE(ACTN);
enum { SOFTMAX_CROSS_ENTROPY_LOSS } UENUM1BYTE(LOSS_F);

#if CONFIG_USE_SMALL_MODEL
#define KF_Y_MODE_FEATURES 26
#else
#define KF_Y_MODE_FEATURES 72
#endif
#define UV_INTRA_MODE_FEATURES 14

#define EM_MAX_HLAYERS 1
#define EM_MAX_NODES 128

// Fully-connectedly layer configuration
typedef struct FC_LAYER_EM {
  int num_inputs;   // Number of input nodes, i.e. features.
  int num_outputs;  // Number of output nodes.

  float weights[EM_MAX_NODES * EM_MAX_NODES];  // Weight parameters.
  float bias[EM_MAX_NODES];                    // Bias parameters.
  ACTN activation;                             // Activation function.

  float output[EM_MAX_NODES];             // The output array.
  float dY[EM_MAX_NODES];                 // Gradient of outputs
  float dW[EM_MAX_NODES * EM_MAX_NODES];  // Gradient of weights.
  float db[EM_MAX_NODES];                 // Gradient of bias
} FC_LAYER_EM;

// NN configure structure for entropy mode (EM)
typedef struct NN_CONFIG_EM {
  float lr;                               // learning rate
  int num_hidden_layers;                  // Number of hidden layers, max = 10.
  float feature[EM_MAX_NODES];            // Input feature
  FC_LAYER_EM layer[EM_MAX_HLAYERS + 1];  // The layer array
  int num_logits;                         // Number of output nodes.
  float output[EM_MAX_NODES];             // Output
  LOSS_F loss;                            // Loss function
} NN_CONFIG_EM;

// Calculate prediction based on the given input features and neural net config.
// Assume there are no more than NN_MAX_NODES_PER_LAYER nodes in each hidden
// layer.
void av1_nn_predict_em(const float *features, NN_CONFIG_EM *nn_config,
                       float *output);

// Back propagation on the given NN model.
void av1_nn_backprop_em(NN_CONFIG_EM *nn_config, const int label);

// Update the weights via gradient descent.
// mu: learning rate, usually chosen from 0.01~0.0001.
void av1_nn_update_em(NN_CONFIG_EM *nn_config, float mu);
#endif  // CONFIG_INTRA_ENTROPY

#endif  // AOM_AV1_COMMON_NN_EM_H_
