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
#include "av1/common/enums.h"

#include "aom_ports/mem.h"

#if CONFIG_INTRA_ENTROPY
enum { ACTN_NONE, ACTN_RELU, ACTN_SIGMOID } UENUM1BYTE(ACTN);
enum { SOFTMAX_CROSS_ENTROPY_LOSS } UENUM1BYTE(LOSS_F);

#define EM_MAX_HLAYERS (0)
#define EM_MAX_SPARSE_FEATURES (2)

// Fully-connectedly layer.
typedef struct FC_LAYER_EM {
  ACTN activation;  // Activation function.
  int num_inputs;   // Number of input nodes, i.e. features.
  int num_outputs;  // Number of output nodes.
  float *weights;   // Weight parameters.
  float *bias;      // Bias parameters.
  float *output;    // The output array.
  float *dy;        // Gradient of outputs
  float *dw;        // Gradient of weights.
  float *db;        // Gradient of bias
} FC_LAYER_EM;

// Fully-connectedly input layer.
typedef struct FC_INPUT_LAYER_EM {
  ACTN activation;
  int num_sparse_inputs;
  int sparse_input_size[EM_MAX_SPARSE_FEATURES];
  int num_dense_inputs;
  int num_outputs;
  float *sparse_weights[EM_MAX_SPARSE_FEATURES];
  float *dense_weights;
  float *bias;
  float *output;
  float *dy;
  float *db;
  float *dw_dense;
  float *dw_sparse[EM_MAX_SPARSE_FEATURES];
} FC_INPUT_LAYER_EM;

// NN configure structure for entropy mode (EM)
typedef struct NN_CONFIG_EM {
  float lr;               // learning rate
  int num_hidden_layers;  // Number of hidden layers, max = 10.
  FC_INPUT_LAYER_EM input_layer;
  FC_LAYER_EM layer[EM_MAX_HLAYERS];  // The layer array
  int num_logits;                     // Number of output nodes.
  LOSS_F loss;                        // Loss function
  float *output;
  int *sparse_features;
  float *dense_features;
} NN_CONFIG_EM;

// Calculate prediction based on the given input features and neural net config.
// Assume there are no more than NN_MAX_NODES_PER_LAYER nodes in each hidden
// layer.
void av1_nn_predict_em(NN_CONFIG_EM *nn_config);

// Back propagation on the given NN model.
void av1_nn_backprop_em(NN_CONFIG_EM *nn_config, const int label);

// Update the weights via gradient descent.
// mu: learning rate, usually chosen from 0.01~0.0001.
void av1_nn_update_em(NN_CONFIG_EM *nn_config, float mu);
#endif  // CONFIG_INTRA_ENTROPY
#endif  // AOM_AV1_COMMON_NN_EM_H_
