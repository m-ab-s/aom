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

#include "config/aom_config.h"
#include "config/av1_rtcd.h"

#include "av1/common/nn_em.h"

#if CONFIG_INTRA_ENTROPY
// Applies the ReLu activation to one fc layer
// output[i] = Max(input[i],0.0f)
static void nn_relu(float *input, int num_outputs) {
  for (int i = 0; i < num_outputs; ++i) {
    input[i] = AOMMAX(input[i], 0.0f);
  }
}

// Applies the Sigmoid activation to one fc layer
// output[i] = 1/(1+exp(input[i]))
static void nn_sigmoid(float *input, int num_outputs) {
  for (int i = 0; i < num_outputs; ++i) {
    const float tmp = AOMMIN(AOMMAX(input[i], -10.0f), 10.0f);
    input[i] = 1.0f / (1.0f + expf(-tmp));
  }
}

// Forward prediction in one fc layer, used in function av1_nn_predict_V2
void av1_nn_fc_forward_c(FC_LAYER_EM *layer, const float *input,
                         float *output) {
  const float *weights = layer->weights;
  const float *bias = layer->bias;
  assert(layer->num_outputs < EM_MAX_NODES);
  // fc
  for (int node = 0; node < layer->num_outputs; ++node) {
    float val = bias[node];
    for (int i = 0; i < layer->num_inputs; ++i) val += weights[i] * input[i];
    output[node] = val;
    weights += layer->num_inputs;
  }

  // activation
  switch (layer->activation) {
    case ACTN_NONE:  // Do Nothing;
      break;
    case ACTN_RELU: nn_relu(output, layer->num_outputs); break;
    case ACTN_SIGMOID: nn_sigmoid(output, layer->num_outputs); break;
    default: assert(0 && "Unknown activation");  // Unknown activation
  }
}

void av1_nn_input_forward(FC_INPUT_LAYER_EM *layer, const int *sparse_features,
                          const float *dense_features) {
  float *output = layer->output;
  const int output_size = layer->num_outputs;
  const int has_sparse = layer->num_sparse_inputs > 0;
  const int has_dense = layer->num_dense_inputs > 0;
  assert(output_size < EM_MAX_NODES);

  const float *bias = layer->bias;
  for (int out_idx = 0; out_idx < output_size; out_idx++) {
    output[out_idx] = bias[out_idx];
  }

  // Handle sparse layer
  if (has_sparse) {
    const float(*sparse_weights)[EM_MAX_WEIGHT_SIZE] = layer->sparse_weights;
    for (int sparse_idx = 0; sparse_idx < layer->num_sparse_inputs;
         sparse_idx++) {
      const float *weight_ptr = sparse_weights[sparse_idx] +
                                sparse_features[sparse_idx] * output_size;
      for (int out_idx = 0; out_idx < output_size; out_idx++) {
        output[out_idx] += weight_ptr[out_idx];
      }
    }
  }

  // fc
  if (has_dense) {
    const float *dense_weights = layer->dense_weights;
    for (int node = 0; node < layer->num_outputs; ++node) {
      float val = 0.0f;
      for (int i = 0; i < layer->num_dense_inputs; ++i) {
        val += dense_weights[i] * dense_features[i];
      }
      output[node] += val;
      dense_weights += layer->num_dense_inputs;
    }
  }

  // activation
  switch (layer->activation) {
    case ACTN_NONE:  // Do Nothing;
      break;
    case ACTN_RELU: nn_relu(output, layer->num_outputs); break;
    case ACTN_SIGMOID: nn_sigmoid(output, layer->num_outputs); break;
    default: assert(0 && "Unknown activation");  // Unknown activation
  }
}

void av1_nn_predict_em(NN_CONFIG_EM *nn_config) {
  const int *sparse_features = nn_config->sparse_features;
  const float *dense_features = nn_config->dense_features;
  const int num_layers = nn_config->num_hidden_layers;
  assert(num_layers <= EM_MAX_HLAYERS);

  // Propagate input layers
  av1_nn_input_forward(&nn_config->input_layer, sparse_features,
                       dense_features);
  float *input_nodes = nn_config->input_layer.output;

  // Propagate the layers.
  int num_inputs = nn_config->layer[0].num_inputs;
  for (int i = 0; i < num_layers; ++i) {
    assert(num_inputs == nn_config->layer[i].num_inputs);
    av1_nn_fc_forward(nn_config->layer + i, input_nodes,
                      nn_config->layer[i].output);
    input_nodes = nn_config->layer[i].output;
    num_inputs = nn_config->layer[i].num_outputs;
  }

  // Final layer
  assert(num_inputs == nn_config->num_logits);
  (void)num_inputs;
  switch (nn_config->loss) {
    case SOFTMAX_CROSS_ENTROPY_LOSS:
      if (nn_config->num_logits == 1) {
        // sigmoid
        const float tmp = AOMMIN(AOMMAX(input_nodes[0], -10.0f), 10.0f);
        nn_config->output[0] = 1.0f / (1.0f + expf(-tmp));
      } else {
        // softmax
        av1_nn_softmax_em(input_nodes, nn_config->output,
                          nn_config->num_logits);
      }
      break;
    default:
      av1_copy_array(nn_config->output, input_nodes, nn_config->num_logits);
  }
}

/***************************Backprop for gradient******************************/
// Backprop for ReLU activation
static void nn_relu_back(float *dX_out, const float *dY, const float *output,
                         int num_outputs) {
  for (int i = 0; i < num_outputs; ++i)
    dX_out[i] = output[i] > 0.0f ? dY[i] : 0.0f;
}

// Backprop for sigmoid activation
static void nn_sigmoid_back(float *dX_out, const float *dY, const float *output,
                            int num_outputs) {
  for (int i = 0; i < num_outputs; ++i)
    dX_out[i] = dY[i] * output[i] * (1 - output[i]);  // dX=dY*sigmoid(X)
}

// Backprop for softmax cross entropy loss
static void nn_softmax_cross_entropy_loss_back(float *dX_out,
                                               const float *output,
                                               const int num_outputs,
                                               const int label) {
  if (num_outputs == 1) {
    // sigmoid
    assert(label < 2);  // label [0,1]
    dX_out[0] = output[0] - (float)label;
  } else {
    // softmax
    assert(num_outputs > label);  // label [0,1,... num_logits-1]
    av1_copy_array(dX_out, output, num_outputs);
    dX_out[label] -= 1;
  }
}

// Backprop in one fc layer, used in function av1_nn_backprop
static void nn_fc_backward(const float *X, float *dX_out, FC_LAYER_EM *layer) {
  // backprop on activation
  float dY_fc[EM_MAX_NODES] = { 0.0f };  // dY for fc
  switch (layer->activation) {
    case ACTN_NONE:  // no activation, dY_fc <-- dY
      av1_copy_array(dY_fc, layer->dY, layer->num_outputs);
      break;
    case ACTN_RELU:
      nn_relu_back(dY_fc, layer->dY, layer->output, layer->num_outputs);
      break;
    case ACTN_SIGMOID:
      nn_sigmoid_back(dY_fc, layer->dY, layer->output, layer->num_outputs);
      break;
    default: assert(0 && "Unknown activation");  // Unknown activation
  }

  // backprop on fc
  // gradient of W, b
  float *dW = layer->dW;
  float *db = layer->db;
  for (int j = 0; j < layer->num_outputs; ++j) {
    for (int i = 0; i < layer->num_inputs; ++i) dW[i] += dY_fc[j] * X[i];
    db[j] += dY_fc[j];
    dW += layer->num_inputs;
  }

  // gradient of the input, i.e., the output of last layer
  if (dX_out) {
    for (int i = 0; i < layer->num_inputs; ++i) {
      float *w = layer->weights + i;
      float val = 0.0f;
      for (int j = 0; j < layer->num_outputs; ++j) {
        val += dY_fc[j] * w[j * layer->num_inputs];
      }
      dX_out[i] = val;
    }
  }
}

static void nn_fc_input_backward(const int *sparse_features,
                                 const float *dense_features,
                                 FC_INPUT_LAYER_EM *layer) {
  const int num_sparse = layer->num_sparse_inputs;
  const int num_dense = layer->num_dense_inputs;
  const int num_out = layer->num_outputs;
  const int has_sparse = num_sparse > 0;
  const int has_dense = num_dense > 0;

  // backprop on activation
  const float *dY_fc = NULL;
  float dY_buffer[EM_MAX_NODES] = { 0.0f };  // dY for fc
  switch (layer->activation) {
    case ACTN_NONE:  // no activation, dY_fc <-- dY
      dY_fc = layer->dY;
      break;
    case ACTN_RELU:
      nn_relu_back(dY_buffer, layer->dY, layer->output, layer->num_outputs);
      dY_fc = dY_buffer;
      break;
    case ACTN_SIGMOID:
      nn_sigmoid_back(dY_buffer, layer->dY, layer->output, layer->num_outputs);
      dY_fc = dY_buffer;
      break;
    default: assert(0 && "Unknown activation");  // Unknown activation
  }

  // Handle bias
  float *db = layer->db;
  for (int j = 0; j < num_out; ++j) {
    db[j] = dY_fc[j];
  }

  // Handle sparse
  float(*dW_sparse)[EM_MAX_WEIGHT_SIZE] = layer->dW_sparse;
  if (has_sparse) {
    for (int s_idx = 0; s_idx < num_sparse; s_idx++) {
      const int non_zero_idx = sparse_features[s_idx];

      for (int j = 0; j < num_out; ++j) {
        dW_sparse[s_idx][non_zero_idx * num_out + j] = dY_fc[j];
      }
    }
  }

  // Handle dense
  float *dW_dense = layer->dW_dense;
  if (has_dense) {
    for (int j = 0; j < num_out; ++j) {
      for (int i = 0; i < num_dense; ++i) {
        dW_dense[i] = dY_fc[j] * dense_features[i];
      }
      dW_dense += num_dense;
    }
  }
}

void av1_nn_backprop_em(NN_CONFIG_EM *nn_config, const int label) {
  // loss layer
  const int num_layers = nn_config->num_hidden_layers;
  float *prev_dY = num_layers > 0 ? nn_config->layer[num_layers - 1].dY
                                  : nn_config->input_layer.dY;

  switch (nn_config->loss) {
    case SOFTMAX_CROSS_ENTROPY_LOSS:
      nn_softmax_cross_entropy_loss_back(prev_dY, nn_config->output,
                                         nn_config->num_logits, label);
      break;
    default: assert(0 && "Unknown loss");  // Unknown loss
  }

  // hidden fc layer
  float *prev_Y = nn_config->input_layer.output;
  for (int layer_idx = num_layers - 1; layer_idx >= 0; layer_idx++) {
    if (layer_idx == 0) {
      prev_dY = nn_config->input_layer.dY;
      prev_Y = nn_config->input_layer.dY;
    } else {
      FC_LAYER_EM *last_layer = &nn_config->layer[layer_idx - 1];
      prev_dY = last_layer->dY;
      prev_Y = last_layer->output;
    }
    nn_fc_backward(prev_Y, prev_dY, &nn_config->layer[layer_idx]);
  }

  nn_fc_input_backward(nn_config->sparse_features, nn_config->dense_features,
                       &nn_config->input_layer);
}

static void update_input_layer(NN_CONFIG_EM *nn_config, float mu) {
  FC_INPUT_LAYER_EM *input_layer = &nn_config->input_layer;
  const int num_sparse = input_layer->num_sparse_inputs;
  const int num_dense = input_layer->num_dense_inputs;
  const int num_out = input_layer->num_outputs;
  const int has_sparse = num_sparse > 0;
  const int has_dense = num_dense > 0;

  float *b = input_layer->bias;
  float *db = input_layer->db;
  for (int j = 0; j < num_out; ++j) {
    b[j] -= mu * db[j];
  }

  // Handle sparse
  if (has_sparse) {
    float(*dW_sparse)[EM_MAX_WEIGHT_SIZE] = input_layer->dW_sparse;
    float(*W_sparse)[EM_MAX_WEIGHT_SIZE] = input_layer->sparse_weights;

    for (int s_idx = 0; s_idx < num_sparse; s_idx++) {
      const int non_zero_idx = nn_config->sparse_features[s_idx];
      const int sparse_size = input_layer->sparse_input_size[s_idx];
      if (non_zero_idx == sparse_size - 1) {
        continue;
      }
      for (int j = 0; j < num_out; j++) {
        W_sparse[s_idx][non_zero_idx * num_out + j] -=
            mu * dW_sparse[s_idx][non_zero_idx * num_out + j];
      }
    }
  }

  if (has_dense) {
    float *dW_dense = input_layer->dW_dense;
    float *W_dense = input_layer->dense_weights;
    for (int j = 0; j < num_dense * num_out; ++j) {
      W_dense[j] -= mu * dW_dense[j];
    }
  }
}

void av1_nn_update_em(NN_CONFIG_EM *nn_config, float mu) {
  const int num_layers = nn_config->num_hidden_layers;

  // Update the weights
  for (int i = 0; i < num_layers; ++i) {
    FC_LAYER_EM *layer = nn_config->layer + i;
    for (int j = 0; j < layer->num_inputs * layer->num_outputs; ++j) {
      layer->weights[j] -= mu * layer->dW[j];
      layer->dW[j] = 0.f;
    }
    for (int j = 0; j < layer->num_outputs; ++j) {
      layer->bias[j] -= mu * layer->db[j];
      layer->db[j] = 0.f;
    }
  }

  // Input layer
  update_input_layer(nn_config, mu);
}

void av1_nn_softmax_em_c(const float *input, float *output, int n) {
  // Softmax function is invariant to adding the same constant
  // to all input values, so we subtract the maximum input to avoid
  // possible overflow.
  float max_inp = input[0];
  for (int i = 1; i < n; i++) max_inp = AOMMAX(max_inp, input[i]);
  float sum_out = 0.0f;
  for (int i = 0; i < n; i++) {
    // Clamp to range [-10.0, 0.0] to prevent FE_UNDERFLOW errors.
    const float normalized_input = AOMMAX(input[i] - max_inp, -10.0f);
    output[i] = (float)exp(normalized_input);
    sum_out += output[i];
  }
  for (int i = 0; i < n; i++) output[i] /= sum_out;
}
#endif  // CONFIG_INTRA_ENTROPY
