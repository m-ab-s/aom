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
#include "av1/common/reconinter.h"
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

// Copy the output of the interpreter into the destination buffer.
void copy_to_output(tflite::Interpreter *interpreter, BLOCK_SIZE bsize,
                    uint8_t *comp_pred, int comp_stride) {
  const int bw = block_size_wide[bsize];
  const int bh = block_size_high[bsize];
  float *output = interpreter->typed_output_tensor<float>(0);

  for (int j = 0; j < bh; ++j) {
    for (int i = 0; i < bw; ++i) {
      comp_pred[i + j * comp_stride] =
          // + 0.5 to round to nearest integer when casting to uint8.
          static_cast<uint8_t>(fclamp(output[i + j * bw] + 0.5f, 0, 255));
    }
  }
}

// Copy the inter-predictor for chroma prediction. Since only 16x16 blocks
// are supported, the chroma is copied if it is 8x8.
void copy_inter(const uint8_t *inter_pred, int inter_stride, uint8_t *comp_pred,
                int comp_stride) {
  for (int i = 0; i < 8; ++i) {
    const uint8_t *inter_row = inter_pred + i * inter_stride;
    uint8_t *comp_row = comp_pred + i * comp_stride;
    memcpy(comp_row, inter_row, 8);
  }
}

}  // namespace

bool is_interintra_ml_supported(const MACROBLOCKD *xd, bool wedge) {
  // Not supported in wedge mode.
  if (wedge) {
    return false;
  }
  // Only supported for block-sizes of 16x16.
  const BLOCK_SIZE bsize = xd->mi[0]->sb_type;
  if (bsize != BLOCK_16X16) {
    return false;
  }
  // build-for-obmc is just used to check whether this is a sub-8x8 block or
  // not. Any value will do for it, since block size must be 16x16.
  const bool build_for_obmc = true;
  int border = av1_calc_border(xd, AOM_PLANE_Y, build_for_obmc);
  border = AOMMIN(border, av1_calc_border(xd, AOM_PLANE_U, build_for_obmc));
  border = AOMMIN(border, av1_calc_border(xd, AOM_PLANE_V, build_for_obmc));
  return border >= INTERINTRA_ML_BORDER;
}

void av1_combine_interintra_ml(INTERINTRA_MODE mode, BLOCK_SIZE plane_bsize,
                               uint8_t *comp_pred, int comp_stride,
                               const uint8_t *inter_pred, int inter_stride,
                               const uint8_t *intra_pred, int intra_stride,
                               int border) {
  (void)border;
  assert(border >= INTERINTRA_ML_BORDER);
  assert(plane_bsize == BLOCK_16X16 || plane_bsize == BLOCK_8X8);
  if (plane_bsize == BLOCK_8X8) {
    copy_inter(inter_pred, inter_stride, comp_pred, comp_stride);
    return;
  }

  tflite::Interpreter *interpreter = get_interpreter();
  load_inputs(interpreter, mode, plane_bsize, inter_pred, inter_stride,
              intra_pred, intra_stride);
  auto status = interpreter->Invoke();
  if (status != kTfLiteOk) {
    tflite::ErrorReporter *reporter = get_reporter();
    reporter->Report("Failed to run inference");
    assert(false);
  }
  copy_to_output(interpreter, plane_bsize, comp_pred, comp_stride);
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
