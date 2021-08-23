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

#include <vector>

#include "av1/common/av1_common_int.h"
#include "av1/common/cnn_tflite.h"
#include "av1/tflite_models/op_registrations.h"
#include "av1/tflite_models/intra_frame_model/uv_qp12.h"
#include "av1/tflite_models/intra_frame_model/uv_qp22.h"
#include "av1/tflite_models/intra_frame_model/uv_qp32.h"
#include "av1/tflite_models/intra_frame_model/uv_qp43.h"
#include "av1/tflite_models/intra_frame_model/uv_qp53.h"
#include "av1/tflite_models/intra_frame_model/uv_qp63.h"
#include "av1/tflite_models/intra_frame_model/qp0_90.h"
#include "av1/tflite_models/intra_frame_model/qp91_120.h"
#include "av1/tflite_models/intra_frame_model/qp121_145.h"
#include "av1/tflite_models/intra_frame_model/qp146_175.h"
#include "av1/tflite_models/intra_frame_model/qp176_205.h"
#include "av1/tflite_models/intra_frame_model/qp206_255.h"
#include "av1/tflite_models/inter_frame_model/uv_qp0_67.h"
#include "av1/tflite_models/inter_frame_model/uv_qp68_107.h"
#include "av1/tflite_models/inter_frame_model/uv_qp108_147.h"
#include "av1/tflite_models/inter_frame_model/uv_qp148_191.h"
#include "av1/tflite_models/inter_frame_model/uv_qp192_231.h"
#include "av1/tflite_models/inter_frame_model/uv_qp232_255.h"
#include "av1/tflite_models/inter_frame_model/qp0_90.h"
#include "av1/tflite_models/inter_frame_model/qp91_120.h"
#include "av1/tflite_models/inter_frame_model/qp121_145.h"
#include "av1/tflite_models/inter_frame_model/qp146_175.h"
#include "av1/tflite_models/inter_frame_model/qp176_205.h"
#include "av1/tflite_models/inter_frame_model/qp206_255.h"
#include "common/tf_lite_includes.h"

#if CONFIG_CNN_RESTORATION
// Returns the TF-lite model based on the qindex.
static const unsigned char *get_intra_model_from_qindex(int qindex,
                                                        int is_luma) {
  if (qindex <= MIN_CNN_Q_INDEX) {
    assert(0);
    return nullptr;
  }

  if (is_luma) {
    if (qindex < 91) {
      return qp0_90_model_tflite_data;
    } else if (qindex < 121) {
      return qp91_120_model_tflite_data;
    } else if (qindex < 146) {
      return qp121_145_model_tflite_data;
    } else if (qindex < 176) {
      return qp146_175_model_tflite_data;
    } else if (qindex < 206) {
      return qp176_205_model_tflite_data;
    } else {
      return qp206_255_model_tflite_data;
    }
  } else {
    if (qindex < 68) {
      return uv_qp12_model_tflite_data;
    } else if (qindex < 108) {
      return uv_qp22_model_tflite_data;
    } else if (qindex < 148) {
      return uv_qp32_model_tflite_data;
    } else if (qindex < 192) {
      return uv_qp43_model_tflite_data;
    } else if (qindex < 232) {
      return uv_qp53_model_tflite_data;
    } else {
      return uv_qp63_model_tflite_data;
    }
  }
}

// Returns the TF-lite model based on the qindex.
static const unsigned char *get_inter_model_from_qindex(int qindex,
                                                        int is_luma) {
  if (qindex <= MIN_CNN_Q_INDEX) {
    assert(0);
    return nullptr;
  }

  if (is_luma) {
    if (qindex < 91) {
      return qp0_90_inter_model_tflite_data;
    } else if (qindex < 121) {
      return qp91_120_inter_model_tflite_data;
    } else if (qindex < 146) {
      return qp121_145_inter_model_tflite_data;
    } else if (qindex < 176) {
      return qp146_175_inter_model_tflite_data;
    } else if (qindex < 206) {
      return qp176_205_inter_model_tflite_data;
    } else {
      return qp206_255_inter_model_tflite_data;
    }
  } else {
    if (qindex < 68) {
      return uv_qp0_67_inter_model_tflite_data;
    } else if (qindex < 108) {
      return uv_qp68_107_inter_model_tflite_data;
    } else if (qindex < 148) {
      return uv_qp108_147_inter_model_tflite_data;
    } else if (qindex < 192) {
      return uv_qp148_191_inter_model_tflite_data;
    } else if (qindex < 232) {
      return uv_qp192_231_inter_model_tflite_data;
    } else {
      return uv_qp232_255_inter_model_tflite_data;
    }
  }
}

static TfLiteDelegate *get_tflite_xnnpack_delegate(int num_threads) {
  TfLiteXNNPackDelegateOptions xnnpack_options =
      TfLiteXNNPackDelegateOptionsDefault();
  xnnpack_options.num_threads = AOMMAX(num_threads, 1);
  return TfLiteXNNPackDelegateCreate(&xnnpack_options);
}

// Builds and returns the TFlite interpreter.
static std::unique_ptr<tflite::Interpreter> get_tflite_interpreter(
    int qindex, int width, int height, int num_threads, int is_intra_only,
    int is_luma, TfLiteDelegate *xnnpack_delegate) {
  const unsigned char *const model_tflite_data =
      is_intra_only ? get_intra_model_from_qindex(qindex, is_luma)
                    : get_inter_model_from_qindex(qindex, is_luma);
  auto model = tflite::GetModel(model_tflite_data);
  tflite::MutableOpResolver resolver;
  RegisterSelectedOpsAllQps(&resolver);
  tflite::InterpreterBuilder builder(model, resolver);
  // TODO(urvang): Investigate if caching the interpreter object provides
  // further speed-up. May still have to re-build the interpreter if qindex
  // changes.
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  interpreter->SetNumThreads(AOMMAX(num_threads, 1));
  tflite::ErrorReporter *reporter = tflite::DefaultErrorReporter();

  // Dimension order: batch_size, height, width, num_channels.
  // Note: height comes before width here!
  const std::vector<int> in_out_dims = { 1, height, width, 1 };
  // We only need to resize the input tensor. All other tensors (including
  // output tensor) will be resized automatically.
  if (interpreter->ResizeInputTensor(interpreter->inputs()[0], in_out_dims) !=
      kTfLiteOk) {
    reporter->Report("Failed at input tensor resize");
    return nullptr;
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    reporter->Report("Failed at tensor allocation");
    return nullptr;
  }

  if (interpreter->ModifyGraphWithDelegate(xnnpack_delegate) != kTfLiteOk) {
    reporter->Report("Failed at modifying graph with XNNPack delegate");
    return nullptr;
  }

  return interpreter;
}

extern "C" int av1_restore_cnn_img_tflite(int qindex, const uint8_t *dgd,
                                          int width, int height, int dgd_stride,
                                          uint8_t *rst, int rst_stride,
                                          int num_threads, int is_intra_only,
                                          int is_luma) {
  TfLiteDelegate *xnnpack_delegate = get_tflite_xnnpack_delegate(num_threads);
  std::unique_ptr<tflite::Interpreter> interpreter =
      get_tflite_interpreter(qindex, width, height, num_threads, is_intra_only,
                             is_luma, xnnpack_delegate);

  // Prepare input.
  const float max_val = 255.0f;
  const int in_stride = width;
  auto input = interpreter->typed_input_tensor<float>(0);
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      input[r * in_stride + c] =
          static_cast<float>(dgd[r * dgd_stride + c]) / max_val;
      assert(input[r * in_stride + c] >= 0.0f);
      assert(input[r * in_stride + c] <= 1.0f);
    }
  }

  // Invoke TFlite inference.
  tflite::ErrorReporter *reporter = tflite::DefaultErrorReporter();
  auto status = interpreter->Invoke();
  if (status != kTfLiteOk) {
    reporter->Report("Failed at interpreter invocation");
    return 0;
  }

  // Use the output to restore 'dgd' and store in 'rst'.
  const auto output = interpreter->typed_output_tensor<float>(0);
  const int out_stride = width;
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      const int residue =
          static_cast<int>(output[r * out_stride + c] * max_val + 0.5);
      rst[r * rst_stride + c] = clip_pixel(dgd[r * dgd_stride + c] + residue);
    }
  }

  // IMPORTANT: release the interpreter before destroying the delegate.
  interpreter.reset();
  TfLiteXNNPackDelegateDelete(xnnpack_delegate);

  return 1;
}

extern "C" int av1_restore_cnn_img_tflite_highbd(
    int qindex, const uint16_t *dgd, int width, int height, int dgd_stride,
    uint16_t *rst, int rst_stride, int num_threads, int bit_depth,
    int is_intra_only, int is_luma) {
  TfLiteDelegate *xnnpack_delegate = get_tflite_xnnpack_delegate(num_threads);
  std::unique_ptr<tflite::Interpreter> interpreter =
      get_tflite_interpreter(qindex, width, height, num_threads, is_intra_only,
                             is_luma, xnnpack_delegate);

  // Prepare input.
  const auto max_val = static_cast<float>((1 << bit_depth) - 1);
  const int in_stride = width;
  auto input = interpreter->typed_input_tensor<float>(0);
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      input[r * in_stride + c] =
          static_cast<float>(dgd[r * dgd_stride + c]) / max_val;
      assert(input[r * in_stride + c] >= 0.0f);
      assert(input[r * in_stride + c] <= 1.0f);
    }
  }

  // Invoke TFlite inference.
  tflite::ErrorReporter *reporter = tflite::DefaultErrorReporter();
  auto status = interpreter->Invoke();
  if (status != kTfLiteOk) {
    reporter->Report("Failed at interpreter invocation");
    return 0;
  }

  // Use the output to restore 'dgd' and store in 'rst'.
  const auto output = interpreter->typed_output_tensor<float>(0);
  const int out_stride = width;
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      const int residue =
          static_cast<int>(output[r * out_stride + c] * max_val + 0.5);
      rst[r * rst_stride + c] =
          clip_pixel_highbd(dgd[r * dgd_stride + c] + residue, bit_depth);
    }
  }

  // IMPORTANT: release the interpreter before destroying the delegate.
  interpreter.reset();
  TfLiteXNNPackDelegateDelete(xnnpack_delegate);

  return 1;
}

extern "C" void av1_restore_cnn_tflite(const AV1_COMMON *cm, int num_threads,
                                       const int apply_cnn[MAX_MB_PLANE]) {
  YV12_BUFFER_CONFIG *buf = &cm->cur_frame->buf;
  const int is_intra_only = frame_is_intra_only(cm);
  for (int plane = 0; plane < av1_num_planes(cm); ++plane) {
    if (!apply_cnn[plane]) continue;
    const int is_luma = (plane == AOM_PLANE_Y);
    if (cm->seq_params.use_highbitdepth) {
      switch (plane) {
        case AOM_PLANE_Y:
          av1_restore_cnn_img_tflite_highbd(
              cm->quant_params.base_qindex, CONVERT_TO_SHORTPTR(buf->y_buffer),
              buf->y_crop_width, buf->y_crop_height, buf->y_stride,
              CONVERT_TO_SHORTPTR(buf->y_buffer), buf->y_stride, num_threads,
              cm->seq_params.bit_depth, is_intra_only, is_luma);
          break;
        case AOM_PLANE_U:
          av1_restore_cnn_img_tflite_highbd(
              cm->quant_params.base_qindex, CONVERT_TO_SHORTPTR(buf->u_buffer),
              buf->uv_crop_width, buf->uv_crop_height, buf->uv_stride,
              CONVERT_TO_SHORTPTR(buf->u_buffer), buf->uv_stride, num_threads,
              cm->seq_params.bit_depth, is_intra_only, is_luma);
          break;
        case AOM_PLANE_V:
          av1_restore_cnn_img_tflite_highbd(
              cm->quant_params.base_qindex, CONVERT_TO_SHORTPTR(buf->v_buffer),
              buf->uv_crop_width, buf->uv_crop_height, buf->uv_stride,
              CONVERT_TO_SHORTPTR(buf->v_buffer), buf->uv_stride, num_threads,
              cm->seq_params.bit_depth, is_intra_only, is_luma);
          break;
        default: assert(0 && "Invalid plane index");
      }
    } else {
      assert(cm->seq_params.bit_depth == 8);
      switch (plane) {
        case AOM_PLANE_Y:
          av1_restore_cnn_img_tflite(
              cm->quant_params.base_qindex, buf->y_buffer, buf->y_crop_width,
              buf->y_crop_height, buf->y_stride, buf->y_buffer, buf->y_stride,
              num_threads, is_intra_only, is_luma);
          break;
        case AOM_PLANE_U:
          av1_restore_cnn_img_tflite(
              cm->quant_params.base_qindex, buf->u_buffer, buf->uv_crop_width,
              buf->uv_crop_height, buf->uv_stride, buf->u_buffer,
              buf->uv_stride, num_threads, is_intra_only, is_luma);
          break;
        case AOM_PLANE_V:
          av1_restore_cnn_img_tflite(
              cm->quant_params.base_qindex, buf->v_buffer, buf->uv_crop_width,
              buf->uv_crop_height, buf->uv_stride, buf->v_buffer,
              buf->uv_stride, num_threads, is_intra_only, is_luma);
          break;
        default: assert(0 && "Invalid plane index");
      }
    }
  }
}
#endif  // CONFIG_CNN_RESTORATION
