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

#include <algorithm>
#include <array>
#include <cassert>
#include <memory>

#include "aom_dsp/aom_dsp_common.h"
#include "av1/common/enums.h"
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
  resolver->AddBuiltin(::tflite::BuiltinOperator_PAD,
                       ::tflite::ops::builtin::Register_PAD());
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

const unsigned char *get_serialized_tflite_model(BLOCK_SIZE bsize) {
  switch (bsize) {
    case BLOCK_8X8: return decode_19752907_001_8x8_tflite_data;
    case BLOCK_8X16: return decode_19752907_003_8x16_tflite_data;
    case BLOCK_16X8: return decode_19752907_002_16x8_tflite_data;
    case BLOCK_16X16: return decode_19752907_004_16x16_tflite_data;
    case BLOCK_16X32: return decode_19752907_006_16x32_tflite_data;
    case BLOCK_32X16: return decode_19752907_005_32x16_tflite_data;
    case BLOCK_32X32: return decode_19752907_007_32x32_tflite_data;
    default: return nullptr;
  }
}

// This list depends on the model. Check each time when we replace model.
static std::array<BLOCK_SIZE, 7> kSupportedSizes = { BLOCK_8X8,   BLOCK_8X16,
                                                     BLOCK_16X8,  BLOCK_16X16,
                                                     BLOCK_16X32, BLOCK_32X16,
                                                     BLOCK_32X32 };

// Initialize the interpreter (only used for static initialization).
tflite::Interpreter **init_interpreter_() {
  static tflite::Interpreter *interpreter_[BLOCK_SIZES_ALL] = { nullptr };

  for (size_t i = 0; i < kSupportedSizes.size(); ++i) {
    auto model =
        tflite::GetModel(get_serialized_tflite_model(kSupportedSizes[i]));
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

    if (interpreter->inputs().size() != 5) {
      reporter->Report("Wrong number of inputs");
      return nullptr;
    }

    if (interpreter->outputs().size() != 1) {
      reporter->Report("Wrong number of outputs");
      return nullptr;
    }

    interpreter_[kSupportedSizes[i]] = interpreter.release();
  }

  return &interpreter_[0];
}

// Get the interpreter (initialized statically). Assumes entire program
// is single threaded.
tflite::Interpreter *get_interpreter(BLOCK_SIZE bsize) {
  // Assumes entire program is single-threaded.
  static tflite::Interpreter **interpreter = init_interpreter_();
  return interpreter[bsize];
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
                 BLOCK_SIZE bsize, int plane, const uint8_t *inter_pred,
                 int inter_stride, const uint8_t *intra_pred, int intra_stride,
                 int tflite_input_wide) {
  const int bw = block_size_wide[bsize];
  const int bh = block_size_high[bsize];
  const int tw = tflite_input_wide;

  // Load the inter-predictor and border.
  float *inter_input = interpreter->typed_input_tensor<float>(0);
  // Border region starts at a negative offset.
  inter_pred -= INTERINTRA_ML_BORDER * (1 + inter_stride);
  for (int j = 0; j < bh + INTERINTRA_ML_BORDER; ++j) {
    std::copy_n(inter_pred + j * inter_stride, INTERINTRA_ML_BORDER + bw,
                inter_input + j * (INTERINTRA_ML_BORDER + tw));
  }

  // Load the top-part of the intra-predictor border.
  float *intra_top_input = interpreter->typed_input_tensor<float>(1);
  intra_pred -= INTERINTRA_ML_BORDER * (1 + intra_stride);
  for (int j = 0; j < INTERINTRA_ML_BORDER; ++j) {
    std::copy_n(intra_pred + j * intra_stride, INTERINTRA_ML_BORDER + bw,
                intra_top_input + j * (INTERINTRA_ML_BORDER + tw));
  }

  // Load the left columns of the intra-predictor border.
  float *intra_left_input = interpreter->typed_input_tensor<float>(2);
  for (int j = 0; j < bh; ++j) {
    std::copy_n(intra_pred + (j + INTERINTRA_ML_BORDER) * intra_stride,
                INTERINTRA_ML_BORDER,
                intra_left_input + j * INTERINTRA_ML_BORDER);
  }

  int32_t *mode_input = interpreter->typed_input_tensor<int>(3);
  *mode_input = mode;

  int32_t *plane_input = interpreter->typed_input_tensor<int>(4);
  *plane_input = plane;
}

// Copy the output of the interpreter into the destination buffer.
void copy_to_output(tflite::Interpreter *interpreter, BLOCK_SIZE bsize,
                    uint8_t *comp_pred, int comp_stride,
                    int tflite_output_wide) {
  const int bw = block_size_wide[bsize];
  const int bh = block_size_high[bsize];
  const int tw = tflite_output_wide;
  float *output = interpreter->typed_output_tensor<float>(0);

  for (int j = 0; j < bh; ++j) {
    for (int i = 0; i < bw; ++i) {
      comp_pred[i + j * comp_stride] =
          // + 0.5 to round to nearest integer when casting to uint8.
          static_cast<uint8_t>(fclamp(output[i + j * tw] + 0.5f, 0, 255));
    }
  }
}

}  // namespace

bool is_interintra_ml_supported(const MACROBLOCKD *xd, bool wedge) {
  const BLOCK_SIZE bsize = xd->mi[0]->sb_type;
  // Not supported in wedge mode, but wedge bit is only valid if the
  // block size supports the wedge case.
  if (wedge && is_interintra_wedge_used(bsize)) {
    return false;
  }
  // ML models have limited list of supported block-sizes.
  if (std::find(kSupportedSizes.begin(), kSupportedSizes.end(), bsize) ==
      kSupportedSizes.end()) {
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

void av1_combine_interintra_ml(INTERINTRA_MODE mode, BLOCK_SIZE bsize,
                               BLOCK_SIZE plane_bsize, int plane,
                               uint8_t *comp_pred, int comp_stride,
                               const uint8_t *inter_pred, int inter_stride,
                               const uint8_t *intra_pred, int intra_stride,
                               int border) {
  (void)border;
  assert(border >= INTERINTRA_ML_BORDER);
  if (std::find(kSupportedSizes.begin(), kSupportedSizes.end(), bsize) ==
      kSupportedSizes.end()) {
    // Not yet implemented. Just copy a blank square into the predictor.
    copy_blank_square(comp_pred, comp_stride, plane_bsize, false);
    return;
  }
  tflite::Interpreter *interpreter = get_interpreter(bsize);
  assert(interpreter != nullptr);
  load_inputs(interpreter, mode, plane_bsize, plane, inter_pred, inter_stride,
              intra_pred, intra_stride, block_size_wide[bsize]);
  auto status = interpreter->Invoke();
  if (status != kTfLiteOk) {
    tflite::ErrorReporter *reporter = get_reporter();
    reporter->Report("Failed to run inference");
    assert(false);
  }

  copy_to_output(interpreter, plane_bsize, comp_pred, comp_stride,
                 block_size_wide[bsize]);
}

void av1_combine_interintra_ml_highbd(INTERINTRA_MODE mode, BLOCK_SIZE bsize,
                                      BLOCK_SIZE plane_bsize, int plane,
                                      uint8_t *comp_pred8, int comp_stride,
                                      const uint8_t *inter_pred8,
                                      int inter_stride,
                                      const uint8_t *intra_pred8,
                                      int intra_stride, int bd, int border) {
  (void)mode;
  (void)bsize;
  (void)plane;
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
