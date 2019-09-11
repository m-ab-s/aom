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
void av1_nn_fc_forward_c(const float *input, FC_LAYER_EM *layer,
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

void av1_nn_predict_em(const float *feature, NN_CONFIG_EM *nn_config,
                       float *output) {
  const float *input_nodes = feature;
  const int num_layers = nn_config->num_hidden_layers;
  assert(num_layers <= EM_MAX_HLAYERS);
  // Copy the input feature to the buffer
  av1_copy_array(nn_config->feature, feature, nn_config->layer[0].num_inputs);

  // Propagate the layers.
  int num_inputs = nn_config->layer[0].num_inputs;
  for (int i = 0; i <= num_layers; ++i) {
    assert(num_inputs == nn_config->layer[i].num_inputs);
    av1_nn_fc_forward(input_nodes, nn_config->layer + i,
                      nn_config->layer[i].output);
    input_nodes = nn_config->layer[i].output;
    num_inputs = nn_config->layer[i].num_outputs;
  }
  (void)num_inputs;

  // Final layer
  assert(nn_config->layer[num_layers].num_outputs == nn_config->num_logits);
  FC_LAYER_EM *layer = nn_config->layer + num_layers;
  switch (nn_config->loss) {
    case SOFTMAX_CROSS_ENTROPY_LOSS:
      if (nn_config->num_logits == 1) {
        // sigmoid
        const float tmp = AOMMIN(AOMMAX(layer->output[0], -10.0f), 10.0f);
        nn_config->output[0] = 1.0f / (1.0f + expf(-tmp));
      } else {
        // softmax
        av1_nn_softmax_em(layer->output, nn_config->output, layer->num_outputs);
      }
      break;
    default:
      av1_copy_array(nn_config->output, input_nodes, nn_config->num_logits);
  }
  // Copy the final layer output
  av1_copy_array(output, nn_config->output, nn_config->num_logits);
}

/***************************Backprop for gradient******************************/
// Backprop for ReLU activation
static void nn_relu_back(float *dX_out, FC_LAYER_EM *layer) {
  const float *dY = layer->dY;
  for (int i = 0; i < layer->num_outputs; ++i)
    dX_out[i] = layer->output[i] > 0.0f ? dY[i] : 0.0f;
}

// Backprop for sigmoid activation
static void nn_sigmoid_back(float *dX_out, FC_LAYER_EM *layer) {
  const float *dY = layer->dY;
  for (int i = 0; i < layer->num_outputs; ++i)
    dX_out[i] =
        dY[i] * layer->output[i] * (1 - layer->output[i]);  // dX=dY*sigmoid(X)
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
    case ACTN_RELU: nn_relu_back(dY_fc, layer); break;
    case ACTN_SIGMOID: nn_sigmoid_back(dY_fc, layer); break;
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

void av1_nn_backprop_em(NN_CONFIG_EM *nn_config, const int label) {
  const int num_layers = nn_config->num_hidden_layers;

  // loss layer
  switch (nn_config->loss) {
    case SOFTMAX_CROSS_ENTROPY_LOSS:
      nn_softmax_cross_entropy_loss_back(nn_config->layer[num_layers].dY,
                                         nn_config->output,
                                         nn_config->num_logits, label);
      break;
    default: assert(0 && "Unknown loss");  // Unknown loss
  }

  // hidden fc layer
  FC_LAYER_EM *layer_ptr = nn_config->layer + num_layers;
  for (int i = 0; i < num_layers; ++i) {
    nn_fc_backward(layer_ptr[-1].output, layer_ptr[-1].dY, layer_ptr);
    layer_ptr -= 1;
  }

  nn_fc_backward(nn_config->feature, NULL,
                 layer_ptr);  // first layer (no dX for feature)
}

void av1_nn_update_em(NN_CONFIG_EM *nn_config, float mu) {
  const int num_layers = nn_config->num_hidden_layers;

  // Update the weights
  for (int i = 0; i <= num_layers; ++i) {
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
