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

#include <emmintrin.h>
#include <stdbool.h>
#include <assert.h>
#include <smmintrin.h>

#include "config/av1_rtcd.h"

#include "aom_ports/system_state.h"

#include "av1/common/entropymode.h"

#if CONFIG_INTRA_ENTROPY

static INLINE __m128 zero_hi_n_floats(int n) {
  static const float mask_array[8] = { -0.0f, -0.0f, -0.0f, -0.0f,
                                       0.0f,  0.0f,  0.0f,  0.0f };
  return _mm_loadu_ps(mask_array + n);
}

static INLINE __m128 set_hi_n_floats(int n) {
  static const float mask_array[8] = { 0.0f,  0.0f,  0.0f,  0.0f,
                                       -0.0f, -0.0f, -0.0f, -0.0f };
  return _mm_loadu_ps(mask_array + n);
}

static void nn_propagate_nto4(const float *const inputs,
                              const float *const weights, __m128 *const outputs,
                              int num_inputs) {
  const int input_rem = num_inputs % 4;
  const int input_whole = num_inputs - input_rem;

  __m128 accum_0 = _mm_setzero_ps();
  __m128 accum_1 = _mm_setzero_ps();
  __m128 accum_2 = _mm_setzero_ps();
  __m128 accum_3 = _mm_setzero_ps();

  int input_node;
  for (input_node = 0; input_node < input_whole; input_node += 4) {
    const __m128 inputs128 = _mm_loadu_ps(&inputs[input_node]);

    const __m128 weight0 = _mm_loadu_ps(&weights[input_node]);
    const __m128 mul0 = _mm_mul_ps(weight0, inputs128);
    accum_0 = _mm_add_ps(accum_0, mul0);

    const __m128 weight1 = _mm_loadu_ps(&weights[input_node + num_inputs]);
    const __m128 mul1 = _mm_mul_ps(weight1, inputs128);
    accum_1 = _mm_add_ps(accum_1, mul1);

    const __m128 weight2 = _mm_loadu_ps(&weights[input_node + 2 * num_inputs]);
    const __m128 mul2 = _mm_mul_ps(weight2, inputs128);
    accum_2 = _mm_add_ps(accum_2, mul2);

    const __m128 weight3 = _mm_loadu_ps(&weights[input_node + 3 * num_inputs]);
    const __m128 mul3 = _mm_mul_ps(weight3, inputs128);
    accum_3 = _mm_add_ps(accum_3, mul3);
  }

  if (input_rem != 0) {
    __m128 inputs128 = _mm_loadu_ps(&inputs[input_whole]);
    const __m128 mask = zero_hi_n_floats(4 - input_rem);
    inputs128 = _mm_blendv_ps(_mm_setzero_ps(), inputs128, mask);

    const __m128 weight0 = _mm_loadu_ps(&weights[input_node]);
    const __m128 mul0 = _mm_mul_ps(weight0, inputs128);
    accum_0 = _mm_add_ps(accum_0, mul0);

    const __m128 weight1 = _mm_loadu_ps(&weights[input_node + num_inputs]);
    const __m128 mul1 = _mm_mul_ps(weight1, inputs128);
    accum_1 = _mm_add_ps(accum_1, mul1);

    const __m128 weight2 = _mm_loadu_ps(&weights[input_node + 2 * num_inputs]);
    const __m128 mul2 = _mm_mul_ps(weight2, inputs128);
    accum_2 = _mm_add_ps(accum_2, mul2);

    const __m128 weight3 = _mm_loadu_ps(&weights[input_node + 3 * num_inputs]);
    const __m128 mul3 = _mm_mul_ps(weight3, inputs128);
    accum_3 = _mm_add_ps(accum_3, mul3);
  }

  const __m128 accum_01 = _mm_hadd_ps(accum_0, accum_1);
  const __m128 accum_23 = _mm_hadd_ps(accum_2, accum_3);
  const __m128 accum_0123 = _mm_hadd_ps(accum_01, accum_23);
  *outputs = _mm_add_ps(*outputs, accum_0123);
}

static void nn_relu(float *input, int num_outputs) {
  for (int i = 0; i < num_outputs; ++i) {
    input[i] = AOMMAX(input[i], 0.0f);
  }
}

static void nn_sigmoid(float *input, int num_outputs) {
  for (int i = 0; i < num_outputs; ++i) {
    const float tmp = AOMMIN(AOMMAX(input[i], -10.0f), 10.0f);
    input[i] = 1.0f / (1.0f + expf(-tmp));
  }
}

// Note: We assume that output is padded to multiples of 4 to make storing
// easier
void av1_nn_fc_forward_sse4_1(const float *input, FC_LAYER_EM *layer,
                              float *output) {
  const float *weights = layer->weights;
  const float *bias = layer->bias;
  const int num_inputs = layer->num_inputs;
  const int num_outputs = layer->num_outputs;
  const int output_rem = num_outputs % 4;
  const int output_whole = num_outputs - output_rem;
  assert(layer->num_outputs < EM_MAX_NODES);

  // fc
  int output_node = 0;
  for (output_node = 0; output_node < output_whole; output_node += 4) {
    __m128 output_reg = _mm_loadu_ps(&bias[output_node]);
    nn_propagate_nto4(input, &weights[output_node * num_inputs], &output_reg,
                      num_inputs);

    _mm_storeu_ps(&output[output_node], output_reg);
  }

  if (output_rem != 0) {
    __m128 output_reg = _mm_loadu_ps(&bias[output_whole]);
    nn_propagate_nto4(input, &weights[output_whole * num_inputs], &output_reg,
                      num_inputs);
    _mm_storeu_ps(&output[output_whole], output_reg);
  }
  aom_clear_system_state();

  switch (layer->activation) {
    case ACTN_NONE:  // Do Nothing;
      break;
    case ACTN_RELU: nn_relu(output, layer->num_outputs); break;
    case ACTN_SIGMOID: nn_sigmoid(output, layer->num_outputs); break;
    default: assert(0 && "Unknown activation");  // Unknown activation
  }
}

// Note: We assume that output is padded to multiples of 4 to make storing
// easier
void av1_nn_softmax_em_sse4_1(const float *input, float *output, int n) {
  const int rem = n % 4;
  const int whole = n - rem;
  assert(layer->num_outputs < EM_MAX_NODES);
  const __m128 neg_inf = _mm_set1_ps(-INFINITY);
  const __m128 mask = set_hi_n_floats(4 - rem);
  int node = 0;

  __m128 max_val = _mm_set1_ps(-INFINITY);
  for (node = 0; node < whole; node += 4) {
    const __m128 input_reg = _mm_loadu_ps(&input[node]);
    max_val = _mm_max_ps(max_val, input_reg);
  }

  if (rem != 0) {
    const __m128 input_reg =
        _mm_blendv_ps(_mm_loadu_ps(&input[node]), neg_inf, mask);
    max_val = _mm_max_ps(max_val, input_reg);
  }

  max_val = _mm_max_ps(max_val, _mm_movehl_ps(max_val, max_val));
  const __m128 shift_max =
      _mm_castsi128_ps(_mm_bsrli_si128(_mm_castps_si128(max_val), 4));
  max_val = _mm_max_ps(max_val, shift_max);
  max_val = _mm_shuffle_ps(max_val, max_val, 0);

  // Exponentiate
  aom_clear_system_state();
  float sum = 0.0f;
  float max_flt = _mm_cvtss_f32(max_val);
  for (node = 0; node < n; node += 1) {
    const float normalized_input = AOMMAX(input[node] - max_flt, -10.0f);
    output[node] = expf(normalized_input);
    sum += output[node];
  }

  __m128 sum_exp = _mm_set1_ps(sum);

  // Divide
  for (node = 0; node < whole; node += 4) {
    const __m128 input_reg = _mm_loadu_ps(&output[node]);
    const __m128 output_reg = _mm_div_ps(input_reg, sum_exp);
    _mm_storeu_ps(&output[node], output_reg);
  }

  if (rem != 0) {
    const __m128 input_reg = _mm_loadu_ps(&output[node]);
    const __m128 output_reg = _mm_div_ps(input_reg, sum_exp);
    _mm_storeu_ps(&output[node], output_reg);
  }
}

#endif  // CONFIG_INTRA_ENTROPY
