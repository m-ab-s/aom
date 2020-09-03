/*
 * Copyright (c) 2020, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <cassert>
#include <memory>

#include "aom_dsp/aom_dsp_common.h"
#include "av1/common/interintra_ml.h"
#include "av1/common/interintra_ml_model.h"
#include "av1/common/reconintra.h"
#include "common/tf_lite_includes.h"

namespace {

void add_resolver_builtins(::tflite::MutableOpResolver *resolver) {
  resolver->AddBuiltin(::tflite::BuiltinOperator_ADD,
                       ::tflite::ops::builtin::Register_ADD());
  resolver->AddBuiltin(::tflite::BuiltinOperator_CAST,
                       ::tflite::ops::builtin::Register_CAST());
  resolver->AddBuiltin(::tflite::BuiltinOperator_CONCATENATION,
                       ::tflite::ops::builtin::Register_CONCATENATION());
  resolver->AddBuiltin(::tflite::BuiltinOperator_CONV_2D,
                       ::tflite::ops::builtin::Register_CONV_2D());
  resolver->AddBuiltin(::tflite::BuiltinOperator_EQUAL,
                       ::tflite::ops::builtin::Register_EQUAL());
  resolver->AddBuiltin(::tflite::BuiltinOperator_FILL,
                       ::tflite::ops::builtin::Register_FILL());
  resolver->AddBuiltin(::tflite::BuiltinOperator_GATHER,
                       ::tflite::ops::builtin::Register_GATHER());
  resolver->AddBuiltin(::tflite::BuiltinOperator_IF,
                       ::tflite::ops::builtin::Register_IF());
  resolver->AddBuiltin(::tflite::BuiltinOperator_LEAKY_RELU,
                       ::tflite::ops::builtin::Register_LEAKY_RELU());
  resolver->AddBuiltin(::tflite::BuiltinOperator_LESS,
                       ::tflite::ops::builtin::Register_LESS());
  resolver->AddBuiltin(::tflite::BuiltinOperator_LOGICAL_AND,
                       ::tflite::ops::builtin::Register_LOGICAL_AND());
  resolver->AddBuiltin(::tflite::BuiltinOperator_RESHAPE,
                       ::tflite::ops::builtin::Register_RESHAPE());
  resolver->AddBuiltin(::tflite::BuiltinOperator_SHAPE,
                       ::tflite::ops::builtin::Register_SHAPE());
  resolver->AddBuiltin(::tflite::BuiltinOperator_SLICE,
                       ::tflite::ops::builtin::Register_SLICE());
  resolver->AddBuiltin(::tflite::BuiltinOperator_STRIDED_SLICE,
                       ::tflite::ops::builtin::Register_STRIDED_SLICE());
  resolver->AddBuiltin(::tflite::BuiltinOperator_TRANSPOSE,
                       ::tflite::ops::builtin::Register_TRANSPOSE());
  resolver->AddBuiltin(::tflite::BuiltinOperator_UNPACK,
                       ::tflite::ops::builtin::Register_UNPACK(), 3, 3);
  resolver->AddBuiltin(::tflite::BuiltinOperator_WHILE,
                       ::tflite::ops::builtin::Register_WHILE());
}

// Returns the error reporter (initialized statically). Assumes
// entire program is single threaded.
tflite::ErrorReporter *get_reporter() {
  static tflite::ErrorReporter *reporter_ = tflite::DefaultErrorReporter();
  return reporter_;
}

// Initialize the interpreter (only used for static initialization).
tflite::Interpreter *init_interpreter_() {
  auto model = tflite::GetModel(decode_13759197_5_tflite_data);
  tflite::MutableOpResolver resolver;
  add_resolver_builtins(&resolver);
  tflite::InterpreterBuilder builder(model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ErrorReporter *reporter = get_reporter();
  if (builder(&interpreter) != kTfLiteOk) {
    reporter->Report("Builder failed");
    return nullptr;
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    reporter->Report("Allocating tensors failed");
    return nullptr;
  }

  if (interpreter->inputs().size() != 4) {
    reporter->Report("Wrong number of inputs");
    return nullptr;
  }

  if (interpreter->outputs().size() != 1) {
    reporter->Report("Wrong number of outputs");
    return nullptr;
  }

  return interpreter.release();
}

// Get the interpreter (initialized statically). Assumes entire program
// is single threaded.
tflite::Interpreter *get_interpreter() {
  // Assumes entire program is single-threaded.
  static tflite::Interpreter *interpreter_ = init_interpreter_();
  return interpreter_;
}

// Copy a blank square into the region. Needed as default behavior if
// the interintra ML model does not support a particular use case.
void copy_blank_square(uint8_t *dst, int stride, BLOCK_SIZE bsize,
                       bool is_hbd) {
  const int bw = block_size_wide[bsize];
  const int bh = block_size_high[bsize];
  for (int j = 0; j < bh; ++j) {
    av1_bd_memset(dst + j * stride, 0, bw, is_hbd);
  }
}

void superscale_pred(uint8_t dst[400], const uint8_t *pred, int stride) {
  const int dst_stride = 20;
  for (int j = 0; j < 20; j += 2) {
    for (int i = 0; i < 20; i += 2) {
      int scaled_i = i / 2;
      int scaled_j = j / 2;
      dst[i + j * dst_stride] = pred[scaled_i + scaled_j * stride];
      dst[i + j * dst_stride + 1] = pred[scaled_i + scaled_j * stride];
      dst[i + (j + 1) * dst_stride] = pred[scaled_i + scaled_j * stride];
      dst[i + (j + 1) * dst_stride + 1] = pred[scaled_i + scaled_j * stride];
    }
  }
}

// Load the inputs (inter-predictor + border, intra-predictor border)
// into the interpreter.
void load_inputs(tflite::Interpreter *interpreter, INTERINTRA_MODE mode,
                 BLOCK_SIZE bsize, const uint8_t *inter_pred, int inter_stride,
                 const uint8_t *intra_pred, int intra_stride) {
  const int bw = block_size_wide[bsize];
  const int bh = block_size_high[bsize];

  // Load the inter-predictor and border.
  float *inter_input = interpreter->typed_input_tensor<float>(0);
  // Border region starts at a negative offset.
  inter_pred -= INTERINTRA_ML_BORDER * (1 + inter_stride);
  for (int j = 0; j < bh + INTERINTRA_ML_BORDER; ++j) {
    std::copy_n(inter_pred + j * inter_stride, bw + INTERINTRA_ML_BORDER,
                inter_input + j * (bw + INTERINTRA_ML_BORDER));
  }

  // Load the top-part of the intra-predictor border.
  float *intra_top_input = interpreter->typed_input_tensor<float>(1);
  intra_pred -= INTERINTRA_ML_BORDER * (1 + intra_stride);
  for (int j = 0; j < INTERINTRA_ML_BORDER; ++j) {
    std::copy_n(intra_pred + j * intra_stride, bw + INTERINTRA_ML_BORDER,
                intra_top_input + j * (bw + INTERINTRA_ML_BORDER));
  }

  // Load the left columns of the intra-predictor border.
  float *intra_left_input = interpreter->typed_input_tensor<float>(2);
  for (int j = 0; j < bh; ++j) {
    std::copy_n(intra_pred + (j + INTERINTRA_ML_BORDER) * intra_stride,
                INTERINTRA_ML_BORDER,
                intra_left_input + j * INTERINTRA_ML_BORDER);
  }

  int *mode_input = interpreter->typed_input_tensor<int>(3);
  *mode_input = mode - II_ML_PRED0 + 1;  // Normalize so 1 is the first mode.
}

// Copy the output of the interpreter into the destination buffer. If
// subsample == true, takes a weighted average of 2x2 blocks for each point
// (creating an 8x8 block).
void copy_to_output(tflite::Interpreter *interpreter, BLOCK_SIZE bsize,
                    uint8_t *comp_pred, int comp_stride, bool subsample) {
  const int bw = block_size_wide[bsize];
  const int bh = block_size_high[bsize];
  float *output = interpreter->typed_output_tensor<float>(0);

  for (int j = 0; j < bh; ++j) {
    for (int i = 0; i < bw; ++i) {
      if (!subsample) {
        comp_pred[i + j * comp_stride] =
            // + 0.5 to round to nearest integer when casting to uint8.
            static_cast<uint8_t>(fclamp(output[i + j * bw] + 0.5f, 0, 255));
        continue;
      }
      // Weighted average.
      const int scaled_i = 2 * i;
      const int scaled_j = 2 * j;
      const int output_stride = 16;
      float total = output[scaled_i + output_stride * scaled_j] +
                    output[scaled_i + output_stride * scaled_j + 1] +
                    output[scaled_i + output_stride * (scaled_j + 1)] +
                    output[scaled_i + output_stride * (scaled_j + 1) + 1] +
                    2.0f;  // +2 to round to nearest int when dividing by 4.
      comp_pred[j * comp_stride + i] =
          static_cast<uint8_t>(fclamp(total / 4.0f, 0, 255));
    }
  }
}

void scale_load_inputs(tflite::Interpreter *interpreter, INTERINTRA_MODE mode,
                       const uint8_t *inter_pred, int inter_stride,
                       const uint8_t *intra_pred, int intra_stride) {
  uint8_t scaled_inter_pred[400];
  const int scaled_inter_stride = 20;
  assert(INTERINTRA_ML_BORDER % 2 == 0);
  superscale_pred(scaled_inter_pred,
                  inter_pred - INTERINTRA_ML_BORDER * inter_stride / 2 -
                      INTERINTRA_ML_BORDER / 2,
                  inter_stride);

  uint8_t scaled_intra_pred[400];
  const int scaled_intra_stride = 20;
  superscale_pred(scaled_intra_pred,
                  intra_pred - INTERINTRA_ML_BORDER * intra_stride / 2 -
                      INTERINTRA_ML_BORDER / 2,
                  intra_stride);
  load_inputs(interpreter, mode, BLOCK_16X16,
              scaled_inter_pred + INTERINTRA_ML_BORDER * scaled_inter_stride +
                  INTERINTRA_ML_BORDER,
              scaled_inter_stride,
              scaled_intra_pred + INTERINTRA_ML_BORDER * scaled_intra_stride +
                  INTERINTRA_ML_BORDER,
              scaled_intra_stride);
}

}  // namespace

void av1_combine_interintra_ml(INTERINTRA_MODE mode, BLOCK_SIZE plane_bsize,
                               uint8_t *comp_pred, int comp_stride,
                               const uint8_t *inter_pred, int inter_stride,
                               const uint8_t *intra_pred, int intra_stride,
                               int border) {
  (void)border;
  assert(border >= INTERINTRA_ML_BORDER);
  if (plane_bsize != BLOCK_16X16 && plane_bsize != BLOCK_8X8) {
    // Not yet implemented. Just copy a blank square into the predictor.
    copy_blank_square(comp_pred, comp_stride, plane_bsize, false);
    return;
  }
  tflite::Interpreter *interpreter = get_interpreter();
  if (plane_bsize == BLOCK_16X16) {
    load_inputs(interpreter, mode, plane_bsize, inter_pred, inter_stride,
                intra_pred, intra_stride);
  } else {
    assert(plane_bsize == BLOCK_8X8);
    scale_load_inputs(interpreter, mode, inter_pred, inter_stride, intra_pred,
                      intra_stride);
  }
  auto status = interpreter->Invoke();
  if (status != kTfLiteOk) {
    tflite::ErrorReporter *reporter = get_reporter();
    reporter->Report("Failed to run inference");
    assert(false);
  }

  const bool subsample = plane_bsize == BLOCK_8X8;
  copy_to_output(interpreter, plane_bsize, comp_pred, comp_stride, subsample);
}

void av1_combine_interintra_ml_highbd(
    INTERINTRA_MODE mode, BLOCK_SIZE plane_bsize, uint8_t *comp_pred8,
    int comp_stride, const uint8_t *inter_pred8, int inter_stride,
    const uint8_t *intra_pred8, int intra_stride, int bd, int border) {
  (void)mode;
  (void)inter_pred8;
  (void)inter_stride;
  (void)intra_pred8;
  (void)intra_stride;
  (void)bd;
  (void)border;
  assert(border >= INTERINTRA_ML_BORDER);
  // Not yet implemented. Just copy a blank square into the predictor.
  copy_blank_square(comp_pred8, comp_stride, plane_bsize, true);
}
