/*
 * Copyright (c) 2020, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <stdbool.h>
#include <assert.h>
#include <arm_neon.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"
#include "aom_dsp/aom_dsp_common.h"
#include "av1/encoder/ml.h"

static void nn_activate8(float32x4_t *out_h, float32x4_t *out_l,
                         const float32x4_t *zero) {
  *out_h = vmaxq_f32(*out_h, *zero);
  *out_l = vmaxq_f32(*out_l, *zero);
}

static void nn_activate4(float32x4_t *x, const float32x4_t *zero) {
  *x = vmaxq_f32(*x, *zero);
}

#define CLAMP_0(x) (x = x > 0 ? x : 0)

static void nn_propagate_8to1(int num_inputs, const float *const inputs,
                              const float *const weights,
                              const float *layer_bias,
                              float *const output_nodes, bool output_layer) {
  const float32x4_t zero = vdupq_n_f32(0);
  float32x4_t vadd = zero;
  float total = *layer_bias;

  for (int in = 0; in < num_inputs; in += 8) {
    const float32x4_t inputs_h = vld1q_f32(&inputs[in + 4]);
    const float32x4_t inputs_l = vld1q_f32(&inputs[in]);

    const float32x4_t weights_h = vld1q_f32(&weights[in + 4]);
    const float32x4_t weights_l = vld1q_f32(&weights[in]);

    vadd = vmlaq_f32(vadd, inputs_h, weights_h);
    vadd = vmlaq_f32(vadd, inputs_l, weights_l);
  }
#if AOM_ARCH_AARCH64
  total += vaddvq_f32(vadd);
#else
  float32x2_t vadd_lo = vadd_f32(vget_low_f32(vadd), vget_high_f32(vadd));
  vadd_lo = vpadd_f32(vadd_lo, vadd_lo);
  total += vget_lane_f32(vadd_lo, 0);
#endif

  if (!output_layer) CLAMP_0(total);
  *output_nodes = total;
}

static void nn_propagate_xto1(int num_inputs, const float *const inputs,
                              const float *const weights,
                              const float *layer_bias,
                              float *const output_nodes, bool output_layer) {
  assert(num_inputs > 8 && (num_inputs & 3) != 0);
  float32x4_t vadd = vdupq_n_f32(0);

  float total = *layer_bias;
  int j = num_inputs;
  int in = 0;
  do {
    const float32x4_t inputs_h = vld1q_f32(&inputs[in + 4]);
    const float32x4_t inputs_l = vld1q_f32(&inputs[in]);

    const float32x4_t weights_h = vld1q_f32(&weights[in + 4]);
    const float32x4_t weights_l = vld1q_f32(&weights[in]);

    vadd = vmlaq_f32(vadd, inputs_h, weights_h);
    vadd = vmlaq_f32(vadd, inputs_l, weights_l);
    in += 8;
    j -= 8;
  } while (j >= 8);

#if AOM_ARCH_AARCH64
  total += vaddvq_f32(vadd);

#else
  float32x2_t vadd_lo = vadd_f32(vget_low_f32(vadd), vget_high_f32(vadd));
  vadd_lo = vpadd_f32(vadd_lo, vadd_lo);
  total += vget_lane_f32(vadd_lo, 0);
#endif
  do {
    total += weights[in] * inputs[in];
    in++;
  } while (in != num_inputs);

  if (!output_layer) CLAMP_0(total);
  *output_nodes = total;
}

static void nn_propagate_xto4(int num_inputs, const float *const inputs,
                              const float *const weights,
                              const float *layer_bias,
                              float *const output_nodes, bool output_layer) {
  assert(num_inputs > 8 && (num_inputs & 3) != 0);
  const float32x4_t zero = vdupq_n_f32(0);
  float32x4_t vadd0 = zero;
  float32x4_t vadd1 = zero;
  float32x4_t vadd2 = zero;
  float32x4_t vadd3 = zero;
  const float *const w0 = weights;
  const float *const w1 = weights + num_inputs;
  const float *const w2 = weights + 2 * num_inputs;
  const float *const w3 = weights + 3 * num_inputs;
  int j = num_inputs;
  int in = 0;

  do {
    const float32x4_t inputs_h = vld1q_f32(&inputs[in + 4]);
    const float32x4_t inputs_l = vld1q_f32(&inputs[in]);
    const float32x4_t w0_h = vld1q_f32(&w0[in + 4]);
    const float32x4_t w0_l = vld1q_f32(&w0[in]);
    const float32x4_t w1_h = vld1q_f32(&w1[in + 4]);
    const float32x4_t w1_l = vld1q_f32(&w1[in]);
    const float32x4_t w2_h = vld1q_f32(&w2[in + 4]);
    const float32x4_t w2_l = vld1q_f32(&w2[in]);
    const float32x4_t w3_h = vld1q_f32(&w3[in + 4]);
    const float32x4_t w3_l = vld1q_f32(&w3[in]);

    vadd0 = vmlaq_f32(vadd0, inputs_h, w0_h);
    vadd1 = vmlaq_f32(vadd1, inputs_h, w1_h);
    vadd2 = vmlaq_f32(vadd2, inputs_h, w2_h);
    vadd3 = vmlaq_f32(vadd3, inputs_h, w3_h);
    vadd0 = vmlaq_f32(vadd0, inputs_l, w0_l);
    vadd1 = vmlaq_f32(vadd1, inputs_l, w1_l);
    vadd2 = vmlaq_f32(vadd2, inputs_l, w2_l);
    vadd3 = vmlaq_f32(vadd3, inputs_l, w3_l);
    in += 8;
    j -= 8;
  } while (j >= 8);

  if (j > 3) {
    const float32x4_t in_l = vld1q_f32(&inputs[in]);
    vadd0 = vmlaq_f32(vadd0, in_l, vld1q_f32(&w0[in]));
    vadd1 = vmlaq_f32(vadd1, in_l, vld1q_f32(&w1[in]));
    vadd2 = vmlaq_f32(vadd2, in_l, vld1q_f32(&w2[in]));
    vadd3 = vmlaq_f32(vadd3, in_l, vld1q_f32(&w3[in]));
    in += 4;
  }

#if AOM_ARCH_AARCH64
  const float32x4_t p0 = vpaddq_f32(vadd0, vadd1);
  const float32x4_t p1 = vpaddq_f32(vadd2, vadd3);
  float32x4_t outputs = vaddq_f32(vld1q_f32(layer_bias), vpaddq_f32(p0, p1));
#else
  const float32x4_t p0 =
      vcombine_f32(vpadd_f32(vget_low_f32(vadd0), vget_high_f32(vadd0)),
                   vpadd_f32(vget_low_f32(vadd1), vget_high_f32(vadd1)));
  const float32x4_t p1 =
      vcombine_f32(vpadd_f32(vget_low_f32(vadd2), vget_high_f32(vadd2)),
                   vpadd_f32(vget_low_f32(vadd3), vget_high_f32(vadd3)));
  const float32x4_t sum =
      vcombine_f32(vpadd_f32(vget_low_f32(p0), vget_high_f32(p0)),
                   vpadd_f32(vget_low_f32(p1), vget_high_f32(p1)));
  float32x4_t outputs = vaddq_f32(vld1q_f32(layer_bias), sum);
#endif

  do {
    float32x4_t w_tail = vld1q_dup_f32(&w0[in]);
    w_tail = vld1q_lane_f32(&w1[in], w_tail, 1);
    w_tail = vld1q_lane_f32(&w2[in], w_tail, 2);
    w_tail = vld1q_lane_f32(&w3[in], w_tail, 3);
    outputs = vmlaq_n_f32(outputs, w_tail, inputs[in]);
    in++;
  } while (in != num_inputs);

  if (!output_layer) nn_activate4(&outputs, &zero);
  vst1q_f32(output_nodes, outputs);
}

static void nn_propagate_xto8(int num_inputs, const float *const inputs,
                              const float *const weights,
                              const float *layer_bias,
                              float *const output_nodes, bool output_layer) {
  assert(num_inputs > 8 && (num_inputs & 3) != 0);
  const float32x4_t zero = vdupq_n_f32(0);
  float32x4_t vadd0 = zero;
  float32x4_t vadd1 = zero;
  float32x4_t vadd2 = zero;
  float32x4_t vadd3 = zero;
  float32x4_t vadd4 = zero;
  float32x4_t vadd5 = zero;
  float32x4_t vadd6 = zero;
  float32x4_t vadd7 = zero;
  const float *const w0 = weights;
  const float *const w2 = weights + 2 * num_inputs;
  const float *const w4 = weights + 4 * num_inputs;
  const float *const w6 = weights + 6 * num_inputs;
  int j = num_inputs;
  int in = 0;

  do {
    const float32x4_t in_l = vld1q_f32(&inputs[in]);
    const float32x4_t in_h = vld1q_f32(&inputs[in + 4]);
    vadd0 = vmlaq_f32(vadd0, in_l, vld1q_f32(&w0[in]));
    vadd1 = vmlaq_f32(vadd1, in_l, vld1q_f32(&w0[in + num_inputs]));
    vadd2 = vmlaq_f32(vadd2, in_l, vld1q_f32(&w2[in]));
    vadd3 = vmlaq_f32(vadd3, in_l, vld1q_f32(&w2[in + num_inputs]));
    vadd4 = vmlaq_f32(vadd4, in_l, vld1q_f32(&w4[in]));
    vadd5 = vmlaq_f32(vadd5, in_l, vld1q_f32(&w4[in + num_inputs]));
    vadd6 = vmlaq_f32(vadd6, in_l, vld1q_f32(&w6[in]));
    vadd7 = vmlaq_f32(vadd7, in_l, vld1q_f32(&w6[in + num_inputs]));
    vadd0 = vmlaq_f32(vadd0, in_h, vld1q_f32(&w0[in + 4]));
    vadd1 = vmlaq_f32(vadd1, in_h, vld1q_f32(&w0[in + num_inputs + 4]));
    vadd2 = vmlaq_f32(vadd2, in_h, vld1q_f32(&w2[in + 4]));
    vadd3 = vmlaq_f32(vadd3, in_h, vld1q_f32(&w2[in + num_inputs + 4]));
    vadd4 = vmlaq_f32(vadd4, in_h, vld1q_f32(&w4[in + 4]));
    vadd5 = vmlaq_f32(vadd5, in_h, vld1q_f32(&w4[in + num_inputs + 4]));
    vadd6 = vmlaq_f32(vadd6, in_h, vld1q_f32(&w6[in + 4]));
    vadd7 = vmlaq_f32(vadd7, in_h, vld1q_f32(&w6[in + num_inputs + 4]));
    in += 8;
    j -= 8;
  } while (j >= 8);

  if (j > 3) {
    const float32x4_t in_l = vld1q_f32(&inputs[in]);
    vadd0 = vmlaq_f32(vadd0, in_l, vld1q_f32(&w0[in]));
    vadd1 = vmlaq_f32(vadd1, in_l, vld1q_f32(&w0[in + num_inputs]));
    vadd2 = vmlaq_f32(vadd2, in_l, vld1q_f32(&w2[in]));
    vadd3 = vmlaq_f32(vadd3, in_l, vld1q_f32(&w2[in + num_inputs]));
    vadd4 = vmlaq_f32(vadd4, in_l, vld1q_f32(&w4[in]));
    vadd5 = vmlaq_f32(vadd5, in_l, vld1q_f32(&w4[in + num_inputs]));
    vadd6 = vmlaq_f32(vadd6, in_l, vld1q_f32(&w6[in]));
    vadd7 = vmlaq_f32(vadd7, in_l, vld1q_f32(&w6[in + num_inputs]));
    in += 4;
  }

#if AOM_ARCH_AARCH64
  const float32x4_t p0 = vpaddq_f32(vadd0, vadd1);
  const float32x4_t p1 = vpaddq_f32(vadd2, vadd3);
  const float32x4_t p2 = vpaddq_f32(vadd4, vadd5);
  const float32x4_t p3 = vpaddq_f32(vadd6, vadd7);
  float32x4_t out_l = vaddq_f32(vld1q_f32(layer_bias), vpaddq_f32(p0, p1));
  float32x4_t out_h = vaddq_f32(vld1q_f32(&layer_bias[4]), vpaddq_f32(p2, p3));
#else
  const float32x4_t p0 =
      vcombine_f32(vpadd_f32(vget_low_f32(vadd0), vget_high_f32(vadd0)),
                   vpadd_f32(vget_low_f32(vadd1), vget_high_f32(vadd1)));
  const float32x4_t p1 =
      vcombine_f32(vpadd_f32(vget_low_f32(vadd2), vget_high_f32(vadd2)),
                   vpadd_f32(vget_low_f32(vadd3), vget_high_f32(vadd3)));
  const float32x4_t p2 =
      vcombine_f32(vpadd_f32(vget_low_f32(vadd4), vget_high_f32(vadd4)),
                   vpadd_f32(vget_low_f32(vadd5), vget_high_f32(vadd5)));
  const float32x4_t p3 =
      vcombine_f32(vpadd_f32(vget_low_f32(vadd6), vget_high_f32(vadd6)),
                   vpadd_f32(vget_low_f32(vadd7), vget_high_f32(vadd7)));
  const float32x4_t sum_l =
      vcombine_f32(vpadd_f32(vget_low_f32(p0), vget_high_f32(p0)),
                   vpadd_f32(vget_low_f32(p1), vget_high_f32(p1)));
  const float32x4_t sum_h =
      vcombine_f32(vpadd_f32(vget_low_f32(p2), vget_high_f32(p2)),
                   vpadd_f32(vget_low_f32(p3), vget_high_f32(p3)));
  float32x4_t out_l = vaddq_f32(vld1q_f32(layer_bias), sum_l);
  float32x4_t out_h = vaddq_f32(vld1q_f32(&layer_bias[4]), sum_h);
#endif

  do {
    const float inp = inputs[in];
    float32x4_t wl = vld1q_dup_f32(&w0[in]);
    wl = vld1q_lane_f32(&w0[in + num_inputs], wl, 1);
    wl = vld1q_lane_f32(&w2[in], wl, 2);
    wl = vld1q_lane_f32(&w2[in + num_inputs], wl, 3);
    float32x4_t wh = vld1q_dup_f32(&w4[in]);
    wh = vld1q_lane_f32(&w4[in + num_inputs], wh, 1);
    wh = vld1q_lane_f32(&w6[in], wh, 2);
    wh = vld1q_lane_f32(&w6[in + num_inputs], wh, 3);
    out_l = vmlaq_n_f32(out_l, wl, inp);
    out_h = vmlaq_n_f32(out_h, wh, inp);
    in++;
  } while (in != num_inputs);

  if (!output_layer) nn_activate8(&out_h, &out_l, &zero);
  vst1q_f32(output_nodes, out_l);
  vst1q_f32(&output_nodes[4], out_h);
}

static void nn_propagate_xsto1(int num_inputs, const float *const inputs,
                               const float *const weights,
                               const float *layer_bias,
                               float *const output_nodes, bool output_layer) {
  assert(num_inputs > 4 && num_inputs < 8);
  float total = *layer_bias;
#if AOM_ARCH_AARCH64
  const float32x4_t v_inputs = vld1q_f32(inputs);
  const float32x4_t v_weights = vld1q_f32(weights);
  const float32x4_t vadd = vmulq_f32(v_inputs, v_weights);
  total += vaddvq_f32(vadd);
  int in = 4;
#else
  int in = 0;
#endif
  do {
    total += weights[in] * inputs[in];
    in++;
  } while (in != num_inputs);

  if (!output_layer) CLAMP_0(total);
  *output_nodes = total;
}

static void nn_propagate_xsto4(int num_inputs, const float *const inputs,
                               const float *const weights,
                               const float *layer_bias,
                               float *const output_nodes, bool output_layer) {
  assert(num_inputs > 4 && num_inputs < 8);
  const float32x4_t zero = vdupq_n_f32(0);
  const float32x4_t v_inputs = vld1q_f32(inputs);
  const float *const w0 = weights;
  const float *const w1 = weights + num_inputs;
  const float *const w2 = weights + 2 * num_inputs;
  const float *const w3 = weights + 3 * num_inputs;
  const float32x4_t mul0 = vmulq_f32(v_inputs, vld1q_f32(w0));
  const float32x4_t mul1 = vmulq_f32(v_inputs, vld1q_f32(w1));
  const float32x4_t mul2 = vmulq_f32(v_inputs, vld1q_f32(w2));
  const float32x4_t mul3 = vmulq_f32(v_inputs, vld1q_f32(w3));
#if AOM_ARCH_AARCH64
  const float32x4_t p0 = vpaddq_f32(mul0, mul1);
  const float32x4_t p1 = vpaddq_f32(mul2, mul3);
  float32x4_t outputs = vaddq_f32(vld1q_f32(layer_bias), vpaddq_f32(p0, p1));
#else
  const float32x4_t p0 =
      vcombine_f32(vpadd_f32(vget_low_f32(mul0), vget_high_f32(mul0)),
                   vpadd_f32(vget_low_f32(mul1), vget_high_f32(mul1)));
  const float32x4_t p1 =
      vcombine_f32(vpadd_f32(vget_low_f32(mul2), vget_high_f32(mul2)),
                   vpadd_f32(vget_low_f32(mul3), vget_high_f32(mul3)));
  const float32x4_t sum =
      vcombine_f32(vpadd_f32(vget_low_f32(p0), vget_high_f32(p0)),
                   vpadd_f32(vget_low_f32(p1), vget_high_f32(p1)));
  float32x4_t outputs = vaddq_f32(vld1q_f32(layer_bias), sum);
#endif
  int in = 4;
  do {
    float32x4_t w_tail = vld1q_dup_f32(&w0[in]);
    w_tail = vld1q_lane_f32(&w1[in], w_tail, 1);
    w_tail = vld1q_lane_f32(&w2[in], w_tail, 2);
    w_tail = vld1q_lane_f32(&w3[in], w_tail, 3);
    outputs = vmlaq_n_f32(outputs, w_tail, inputs[in]);
    in++;
  } while (in != num_inputs);
  if (!output_layer) nn_activate4(&outputs, &zero);
  vst1q_f32(output_nodes, outputs);
}

static void nn_propagate_4to1(int num_inputs, const float *const inputs,
                              const float *const weights,
                              const float *layer_bias,
                              float *const output_nodes, bool output_layer) {
  const float32x4_t zero = vdupq_n_f32(0);
  float32x4_t vadd = zero;
  float total = *layer_bias;

  for (int in = 0; in < num_inputs; in += 4) {
    const float32x4_t v_inputs = vld1q_f32(&inputs[in]);
    const float32x4_t v_weights = vld1q_f32(&weights[in]);
    vadd = vmlaq_f32(vadd, v_inputs, v_weights);
  }

#if AOM_ARCH_AARCH64
  total += vaddvq_f32(vadd);
#else
  float32x2_t vadd_lo = vadd_f32(vget_low_f32(vadd), vget_high_f32(vadd));
  vadd_lo = vpadd_f32(vadd_lo, vadd_lo);
  total += vget_lane_f32(vadd_lo, 0);
#endif

  if (!output_layer) CLAMP_0(total);
  *output_nodes = total;
}

static void nn_propagate_4to4(int num_inputs, const float *const inputs,
                              const float *const weights,
                              const float *layer_bias,
                              float *const output_nodes, bool output_layer) {
  float32x4_t outputs = vld1q_f32(layer_bias);
  const float32x4_t zero = vdupq_n_f32(0);

  float32x4_t mul0[2] = { zero, zero };
  float32x4_t mul1[2] = { zero, zero };
  for (int in = 0; in < num_inputs; in += 4) {
    const float32x4_t v_input = vld1q_f32(&inputs[in]);

    for (int i = 0; i < 2; i++) {
      const float32x4_t weight0 = vld1q_f32(&weights[in + 2 * i * num_inputs]);
      mul0[i] = vmlaq_f32(mul0[i], weight0, v_input);
      const float32x4_t weight1 =
          vld1q_f32(&weights[in + (2 * i + 1) * num_inputs]);
      mul1[i] = vmlaq_f32(mul1[i], weight1, v_input);
    }
  }
  for (int i = 0; i < 2; i++)
#if AOM_ARCH_AARCH64
    mul0[i] = vpaddq_f32(mul0[i], mul1[i]);
  const float32x4_t hh = vpaddq_f32(mul0[0], mul0[1]);
#else
    mul0[i] =
        vcombine_f32(vpadd_f32(vget_low_f32(mul0[i]), vget_high_f32(mul0[i])),
                     vpadd_f32(vget_low_f32(mul1[i]), vget_high_f32(mul1[i])));
  const float32x4_t hh =
      vcombine_f32(vpadd_f32(vget_low_f32(mul0[0]), vget_high_f32(mul0[0])),
                   vpadd_f32(vget_low_f32(mul0[1]), vget_high_f32(mul0[1])));
#endif

  outputs = vaddq_f32(outputs, hh);
  if (!output_layer) nn_activate4(&outputs, &zero);
  vst1q_f32(output_nodes, outputs);
}

static void nn_propagate_4to8(const int num_inputs, const float *const inputs,
                              const float *const weights,
                              const float *layer_bias,
                              float *const output_nodes, bool output_layer) {
  float32x4_t out_h = vld1q_f32(&layer_bias[4]);
  float32x4_t out_l = vld1q_f32(layer_bias);
  const float32x4_t zero = vdupq_n_f32(0);
  float32x4_t mul0[4] = { zero, zero, zero, zero };
  float32x4_t mul1[4] = { zero, zero, zero, zero };

  assert(num_inputs >= 4 && (num_inputs & 3) == 0);
  const float *const w0 = weights;
  const float *const w2 = weights + 2 * num_inputs;
  const float *const w4 = weights + 4 * num_inputs;
  const float *const w6 = weights + 6 * num_inputs;
  int in = 0;
  do {
    const float32x4_t v_input = vld1q_f32(&inputs[in]);
    mul0[0] = vmlaq_f32(mul0[0], v_input, vld1q_f32(&w0[in]));
    mul1[0] = vmlaq_f32(mul1[0], v_input, vld1q_f32(&w0[in + num_inputs]));
    mul0[1] = vmlaq_f32(mul0[1], v_input, vld1q_f32(&w2[in]));
    mul1[1] = vmlaq_f32(mul1[1], v_input, vld1q_f32(&w2[in + num_inputs]));
    mul0[2] = vmlaq_f32(mul0[2], v_input, vld1q_f32(&w4[in]));
    mul1[2] = vmlaq_f32(mul1[2], v_input, vld1q_f32(&w4[in + num_inputs]));
    mul0[3] = vmlaq_f32(mul0[3], v_input, vld1q_f32(&w6[in]));
    mul1[3] = vmlaq_f32(mul1[3], v_input, vld1q_f32(&w6[in + num_inputs]));
    in += 4;
  } while (in != num_inputs);
  for (int i = 0; i < 4; i++)
#if AOM_ARCH_AARCH64
    mul0[i] = vpaddq_f32(mul0[i], mul1[i]);
  const float32x4_t hh0 = vpaddq_f32(mul0[0], mul0[1]);
  const float32x4_t hh1 = vpaddq_f32(mul0[2], mul0[3]);
#else
    mul0[i] =
        vcombine_f32(vpadd_f32(vget_low_f32(mul0[i]), vget_high_f32(mul0[i])),
                     vpadd_f32(vget_low_f32(mul1[i]), vget_high_f32(mul1[i])));
  const float32x4_t hh0 =
      vcombine_f32(vpadd_f32(vget_low_f32(mul0[0]), vget_high_f32(mul0[0])),
                   vpadd_f32(vget_low_f32(mul0[1]), vget_high_f32(mul0[1])));
  const float32x4_t hh1 =
      vcombine_f32(vpadd_f32(vget_low_f32(mul0[2]), vget_high_f32(mul0[2])),
                   vpadd_f32(vget_low_f32(mul0[3]), vget_high_f32(mul0[3])));
#endif

  out_h = vaddq_f32(out_h, hh1);
  out_l = vaddq_f32(out_l, hh0);

  if (!output_layer) nn_activate8(&out_h, &out_l, &zero);
  vst1q_f32(&output_nodes[4], out_h);
  vst1q_f32(output_nodes, out_l);
}

static void nn_propagate_8to4(const int num_inputs, const float *const inputs,
                              const float *const weights,
                              const float *layer_bias,
                              float *const output_nodes, bool output_layer) {
  float32x4_t outputs = vld1q_f32(layer_bias);
  const float32x4_t zero = vdupq_n_f32(0);
  float32x4_t add[4] = { zero, zero, zero, zero };
  for (int in = 0; in < num_inputs; in += 8) {
    const float32x4_t inputs_l = vld1q_f32(&inputs[in]);
    const float32x4_t inputs_h = vld1q_f32(&inputs[in + 4]);

    for (int i = 0; i < 4; i++) {
      const float32x4_t weight_l = vld1q_f32(&weights[in + i * num_inputs]);
      const float32x4_t weight_h = vld1q_f32(&weights[in + i * num_inputs + 4]);
      add[i] = vmlaq_f32(add[i], inputs_l, weight_l);
      add[i] = vmlaq_f32(add[i], inputs_h, weight_h);
    }
  }
#if AOM_ARCH_AARCH64
  const float32x4_t hadd_h = vpaddq_f32(add[2], add[3]);
  const float32x4_t hadd_l = vpaddq_f32(add[0], add[1]);
  const float32x4_t haddhadd = vpaddq_f32(hadd_l, hadd_h);
#else
  const float32x4_t hadd_h =
      vcombine_f32(vpadd_f32(vget_low_f32(add[2]), vget_high_f32(add[2])),
                   vpadd_f32(vget_low_f32(add[3]), vget_high_f32(add[3])));
  const float32x4_t hadd_l =
      vcombine_f32(vpadd_f32(vget_low_f32(add[0]), vget_high_f32(add[0])),
                   vpadd_f32(vget_low_f32(add[1]), vget_high_f32(add[1])));
  const float32x4_t haddhadd =
      vcombine_f32(vpadd_f32(vget_low_f32(hadd_l), vget_high_f32(hadd_l)),
                   vpadd_f32(vget_low_f32(hadd_h), vget_high_f32(hadd_h)));
#endif

  outputs = vaddq_f32(outputs, haddhadd);
  if (!output_layer) nn_activate4(&outputs, &zero);
  vst1q_f32(output_nodes, outputs);
}

static AOM_FORCE_INLINE void nn_predict_layer_neon(
    const float *const input_nodes, const float *const layer_weights,
    const float *const layer_bias, int num_inputs, int num_outputs,
    bool output_layer, float *const output_nodes) {
  assert(num_inputs > 0 && num_outputs > 0);
  if (num_inputs % 4 == 0 && num_outputs % 8 == 0) {
    int out = 0;
    do {
      nn_propagate_4to8(num_inputs, input_nodes,
                        &layer_weights[out * num_inputs], &layer_bias[out],
                        &output_nodes[out], output_layer);
      out += 8;
    } while (out != num_outputs);
  } else if (num_inputs % 8 == 0 && num_outputs % 4 == 0) {
    int out = 0;
    do {
      nn_propagate_8to4(num_inputs, input_nodes,
                        &layer_weights[out * num_inputs], &layer_bias[out],
                        &output_nodes[out], output_layer);
      out += 4;
    } while (out != num_outputs);
  } else if (num_inputs % 4 == 0 && num_outputs % 4 == 0) {
    int out = 0;
    do {
      nn_propagate_4to4(num_inputs, input_nodes,
                        &layer_weights[out * num_inputs], &layer_bias[out],
                        &output_nodes[out], output_layer);
      out += 4;
    } while (out != num_outputs);
  } else if (num_inputs % 8 == 0) {
    int out = 0;
    do {
      nn_propagate_8to1(num_inputs, input_nodes,
                        &layer_weights[out * num_inputs], &layer_bias[out],
                        &output_nodes[out], output_layer);
      out++;
    } while (out != num_outputs);
  } else if (num_inputs % 4 == 0) {
    int out = 0;
    do {
      nn_propagate_4to1(num_inputs, input_nodes,
                        &layer_weights[out * num_inputs], &layer_bias[out],
                        &output_nodes[out], output_layer);
      out++;
    } while (out != num_outputs);
  } else if (num_inputs > 8) {
    int out = 0;
    for (; out + 8 <= num_outputs; out += 8) {
      nn_propagate_xto8(num_inputs, input_nodes,
                        &layer_weights[out * num_inputs], &layer_bias[out],
                        &output_nodes[out], output_layer);
    }
    for (; out + 4 <= num_outputs; out += 4) {
      nn_propagate_xto4(num_inputs, input_nodes,
                        &layer_weights[out * num_inputs], &layer_bias[out],
                        &output_nodes[out], output_layer);
    }
    for (; out < num_outputs; out++) {
      nn_propagate_xto1(num_inputs, input_nodes,
                        &layer_weights[out * num_inputs], &layer_bias[out],
                        &output_nodes[out], output_layer);
    }
  } else if (num_inputs >= 4) {
    int out = 0;
    for (; out + 4 <= num_outputs; out += 4) {
      nn_propagate_xsto4(num_inputs, input_nodes,
                         &layer_weights[out * num_inputs], &layer_bias[out],
                         &output_nodes[out], output_layer);
    }
    for (; out < num_outputs; out++) {
      nn_propagate_xsto1(num_inputs, input_nodes,
                         &layer_weights[out * num_inputs], &layer_bias[out],
                         &output_nodes[out], output_layer);
    }
  } else {
    int node = 0;
    do {
      float val = layer_bias[node];
      int i = 0;
      do {
        val += layer_weights[node * num_inputs + i] * input_nodes[i];
        ++i;
      } while (i != num_inputs);
      if (!output_layer) {
        // ReLU as activation function.
        val = val > 0.0f ? val : 0.0f;  // Could use AOMMAX().
      }
      output_nodes[node] = val;
      ++node;
    } while (node != num_outputs);
  }
}

// Avoid inlining the multi-layer hidden-buffer stack frame into the
// 0-hidden-layer fast path of av1_nn_predict_neon.
static AOM_NOINLINE void nn_predict_hidden_neon(
    const float *input_nodes, const NN_CONFIG *const nn_config,
    float *const output) {
  float buf[2][NN_MAX_NODES_PER_LAYER];
  int buf_index = 0;
  int num_inputs = nn_config->num_inputs;
  // Hidden layers, except the final iteration is the output layer.
  for (int layer = 0; layer <= nn_config->num_hidden_layers; layer++) {
    const float *layer_weights = nn_config->weights[layer];
    const float *layer_bias = nn_config->bias[layer];
    bool output_layer = (layer == nn_config->num_hidden_layers);
    float *const output_nodes = output_layer ? output : buf[buf_index];
    const int num_outputs = output_layer ? nn_config->num_outputs
                                         : nn_config->num_hidden_nodes[layer];

    nn_predict_layer_neon(input_nodes, layer_weights, layer_bias, num_inputs,
                          num_outputs, output_layer, output_nodes);
    input_nodes = output_nodes;
    num_inputs = num_outputs;
    buf_index = 1 - buf_index;
  }
}

// Calculate prediction based on the given input features and neural net config.
// Assume there are no more than NN_MAX_NODES_PER_LAYER nodes in each hidden
// layer.
void av1_nn_predict_neon(const float *input_nodes,
                         const NN_CONFIG *const nn_config, int reduce_prec,
                         float *const output) {
  if (nn_config->num_hidden_layers == 0) {
    nn_predict_layer_neon(input_nodes, nn_config->weights[0],
                          nn_config->bias[0], nn_config->num_inputs,
                          nn_config->num_outputs, true, output);
  } else {
    nn_predict_hidden_neon(input_nodes, nn_config, output);
  }
  if (reduce_prec) av1_nn_output_prec_reduce(output, nn_config->num_outputs);
}

static inline float32x4_t approx_exp_neon(float32x4_t y) {
  const float32x4_t a = vdupq_n_f32((1 << 23) / 0.69314718056f);
  const int32x4_t bc = vdupq_n_s32((127 << 23) - 60801);
  const int32x4_t exp_i32 = vaddq_s32(vcvtq_s32_f32(vmulq_f32(y, a)), bc);
  return vreinterpretq_f32_s32(exp_i32);
}

void av1_nn_fast_softmax_16_neon(const float *input, float *output) {
  const float32x4_t clipper = vdupq_n_f32(-10.0f);
  const float32x4_t in0 = vld1q_f32(&input[0]);
  const float32x4_t in1 = vld1q_f32(&input[4]);
  const float32x4_t in2 = vld1q_f32(&input[8]);
  const float32x4_t in3 = vld1q_f32(&input[12]);

  const float32x4_t m4 = vmaxq_f32(vmaxq_f32(in0, in1), vmaxq_f32(in2, in3));
#if AOM_ARCH_AARCH64
  const float max_input = vmaxvq_f32(m4);
#else
  float32x2_t m2 = vmax_f32(vget_low_f32(m4), vget_high_f32(m4));
  m2 = vpmax_f32(m2, m2);
  const float max_input = vget_lane_f32(m2, 0);
#endif
  const float32x4_t v_max = vdupq_n_f32(max_input);

  const float32x4_t exp0 =
      approx_exp_neon(vmaxq_f32(vsubq_f32(in0, v_max), clipper));
  const float32x4_t exp1 =
      approx_exp_neon(vmaxq_f32(vsubq_f32(in1, v_max), clipper));
  const float32x4_t exp2 =
      approx_exp_neon(vmaxq_f32(vsubq_f32(in2, v_max), clipper));
  const float32x4_t exp3 =
      approx_exp_neon(vmaxq_f32(vsubq_f32(in3, v_max), clipper));

  // Accumulate sequentially to guarantee 0-ULP parity with
  // av1_nn_fast_softmax_16_c and 100% SHA-256 bitstream identity.
  float sum_out = vgetq_lane_f32(exp0, 0) + vgetq_lane_f32(exp0, 1);
  sum_out += vgetq_lane_f32(exp0, 2);
  sum_out += vgetq_lane_f32(exp0, 3);
  sum_out += vgetq_lane_f32(exp1, 0);
  sum_out += vgetq_lane_f32(exp1, 1);
  sum_out += vgetq_lane_f32(exp1, 2);
  sum_out += vgetq_lane_f32(exp1, 3);
  sum_out += vgetq_lane_f32(exp2, 0);
  sum_out += vgetq_lane_f32(exp2, 1);
  sum_out += vgetq_lane_f32(exp2, 2);
  sum_out += vgetq_lane_f32(exp2, 3);
  sum_out += vgetq_lane_f32(exp3, 0);
  sum_out += vgetq_lane_f32(exp3, 1);
  sum_out += vgetq_lane_f32(exp3, 2);
  sum_out += vgetq_lane_f32(exp3, 3);

#if AOM_ARCH_AARCH64
  const float32x4_t v_sum = vdupq_n_f32(sum_out);
  vst1q_f32(&output[0], vdivq_f32(exp0, v_sum));
  vst1q_f32(&output[4], vdivq_f32(exp1, v_sum));
  vst1q_f32(&output[8], vdivq_f32(exp2, v_sum));
  vst1q_f32(&output[12], vdivq_f32(exp3, v_sum));
#else
  vst1q_f32(&output[0], exp0);
  vst1q_f32(&output[4], exp1);
  vst1q_f32(&output[8], exp2);
  vst1q_f32(&output[12], exp3);
  int i = 0;
  do {
    output[i] /= sum_out;
    ++i;
  } while (i != 16);
#endif
}
