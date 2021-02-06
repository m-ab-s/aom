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

#include "av1/common/cnn_tflite.h"
#include "av1/common/onyxc_int.h"
#include "av1/tflite_models/op_registrations.h"
#include "av1/tflite_models/intra_frame_model/uv_qp22.h"
#include "av1/tflite_models/intra_frame_model/uv_qp32.h"
#include "av1/tflite_models/intra_frame_model/uv_qp43.h"
#include "av1/tflite_models/intra_frame_model/uv_qp53.h"
#include "av1/tflite_models/intra_frame_model/uv_qp63.h"
#include "av1/tflite_models/intra_frame_model/qp22.h"
#include "av1/tflite_models/intra_frame_model/qp32.h"
#include "av1/tflite_models/intra_frame_model/qp43.h"
#include "av1/tflite_models/intra_frame_model/qp53.h"
#include "av1/tflite_models/intra_frame_model/qp63.h"
#include "av1/tflite_models/inter_frame_model/uv_qp68_107.h"
#include "av1/tflite_models/inter_frame_model/uv_qp108_147.h"
#include "av1/tflite_models/inter_frame_model/uv_qp148_191.h"
#include "av1/tflite_models/inter_frame_model/uv_qp192_231.h"
#include "av1/tflite_models/inter_frame_model/uv_qp232_255.h"
#include "av1/tflite_models/inter_frame_model/qp68_107.h"
#include "av1/tflite_models/inter_frame_model/qp108_147.h"
#include "av1/tflite_models/inter_frame_model/qp148_191.h"
#include "av1/tflite_models/inter_frame_model/qp192_231.h"
#include "av1/tflite_models/inter_frame_model/qp232_255.h"

#if CONFIG_CNN_CRLC_GUIDED
#include "av1/tflite_models/crlc_model/qp12_crlc.h"
#include "av1/tflite_models/crlc_model/qp22_crlc.h"
#include "av1/tflite_models/crlc_model/qp28_crlc.h"
#include "av1/tflite_models/crlc_model/qp33_crlc.h"
#include "av1/tflite_models/crlc_model/qp43_crlc.h"
#include "av1/tflite_models/crlc_model/qp53_crlc.h"
#include "av1/tflite_models/crlc_model/qp63_crlc.h"
#endif  // CONFIG_CNN_CRLC_GUIDED

#if CONFIG_NN_RECON
#include "av1/tflite_models/intra_txfm_recon_model/tx16x16.h"
#endif  // CONFIG_NN_RECON
#include "common/tf_lite_includes.h"

#if CONFIG_CNN_RESTORATION || CONFIG_LOOP_RESTORE_CNN
// Returns the TF-lite model based on the qindex.
static const unsigned char *get_intra_model_from_qindex(int qindex,
                                                        int is_luma) {
  if (qindex <= MIN_CNN_Q_INDEX) {
    assert(0);
    return nullptr;
  }

  if (is_luma) {
#if CONFIG_CNN_CRLC_GUIDED
    int QP = qindex / 4;
    if (QP < 17) {
      return qp12_crlc_model_tflite_data;
    } else if (QP < 27) {
      return qp22_crlc_model_tflite_data;
    } else if (QP < 31) {
      return qp28_crlc_model_tflite_data;
    } else if (QP < 37) {
      return qp33_crlc_model_tflite_data;
    } else if (QP < 47) {
      return qp43_crlc_model_tflite_data;
    } else if (QP < 57) {
      return qp53_crlc_model_tflite_data;
    } else {
      return qp63_crlc_model_tflite_data;
    }
#else
    if (qindex < 108) {
      return qp22_model_tflite_data;
    } else if (qindex < 148) {
      return qp32_model_tflite_data;
    } else if (qindex < 192) {
      return qp43_model_tflite_data;
    } else if (qindex < 232) {
      return qp53_model_tflite_data;
    } else {
      return qp63_model_tflite_data;
    }
#endif  // CONFIG_CNN_CRLC_GUIDED
  } else {
    if (qindex < 108) {
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
#if CONFIG_CNN_CRLC_GUIDED
    int QP = qindex / 4;
    if (QP < 17) {
      return qp12_crlc_model_tflite_data;
    } else if (QP < 27) {
      return qp22_crlc_model_tflite_data;
    } else if (QP < 31) {
      return qp28_crlc_model_tflite_data;
    } else if (QP < 37) {
      return qp33_crlc_model_tflite_data;
    } else if (QP < 47) {
      return qp43_crlc_model_tflite_data;
    } else if (QP < 57) {
      return qp53_crlc_model_tflite_data;
    } else {
      return qp63_crlc_model_tflite_data;
    }
#else
    if (qindex < 108) {
      return qp68_107_inter_model_tflite_data;
    } else if (qindex < 148) {
      return qp108_147_inter_model_tflite_data;
    } else if (qindex < 192) {
      return qp148_191_inter_model_tflite_data;
    } else if (qindex < 232) {
      return qp192_231_inter_model_tflite_data;
    } else {
      return qp232_255_inter_model_tflite_data;
    }
#endif  // CONFIG_CNN_CRLC_GUIDED
  } else {
    if (qindex < 108) {
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
                                       int plane_from, int plane_to) {
  YV12_BUFFER_CONFIG *buf = &cm->cur_frame->buf;
  const int is_intra_only = frame_is_intra_only(cm);
  for (int plane = plane_from; plane <= plane_to; ++plane) {
    const int is_luma = (plane == AOM_PLANE_Y);
    if (cm->seq_params.use_highbitdepth) {
      switch (plane) {
        case AOM_PLANE_Y:
          av1_restore_cnn_img_tflite_highbd(
              cm->base_qindex, CONVERT_TO_SHORTPTR(buf->y_buffer),
              buf->y_crop_width, buf->y_crop_height, buf->y_stride,
              CONVERT_TO_SHORTPTR(buf->y_buffer), buf->y_stride, num_threads,
              cm->seq_params.bit_depth, is_intra_only, is_luma);
          break;
        case AOM_PLANE_U:
          av1_restore_cnn_img_tflite_highbd(
              cm->base_qindex, CONVERT_TO_SHORTPTR(buf->u_buffer),
              buf->uv_crop_width, buf->uv_crop_height, buf->uv_stride,
              CONVERT_TO_SHORTPTR(buf->u_buffer), buf->uv_stride, num_threads,
              cm->seq_params.bit_depth, is_intra_only, is_luma);
          break;
        case AOM_PLANE_V:
          av1_restore_cnn_img_tflite_highbd(
              cm->base_qindex, CONVERT_TO_SHORTPTR(buf->v_buffer),
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
              cm->base_qindex, buf->y_buffer, buf->y_crop_width,
              buf->y_crop_height, buf->y_stride, buf->y_buffer, buf->y_stride,
              num_threads, is_intra_only, is_luma);
          break;
        case AOM_PLANE_U:
          av1_restore_cnn_img_tflite(
              cm->base_qindex, buf->u_buffer, buf->uv_crop_width,
              buf->uv_crop_height, buf->uv_stride, buf->u_buffer,
              buf->uv_stride, num_threads, is_intra_only, is_luma);
          break;
        case AOM_PLANE_V:
          av1_restore_cnn_img_tflite(
              cm->base_qindex, buf->v_buffer, buf->uv_crop_width,
              buf->uv_crop_height, buf->uv_stride, buf->v_buffer,
              buf->uv_stride, num_threads, is_intra_only, is_luma);
          break;
        default: assert(0 && "Invalid plane index");
      }
    }
  }
}
#endif  // CONFIG_CNN_RESTORATION || CONFIG_LOOP_RESTORE_CNN

#if CONFIG_CNN_CRLC_GUIDED

extern "C" int av1_restore_cnn_guided_img_tflite(
    int qindex, const uint8_t *dgd, int width, int height, int dgd_stride,
    uint8_t *rst, int rst_stride, int num_threads, int is_intra_only,
    int is_luma, const uint8_t *src, int src_stride, CRLCInfo *ci,
    int frameType) {
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

  uint8_t **sub_dgr = new uint8_t *[height];
  for (int i = 0; i < height; i++) {
    sub_dgr[i] = new uint8_t[width];
  }
  uint8_t **sub_src = new uint8_t *[height];
  for (int i = 0; i < height; i++) {
    sub_src[i] = new uint8_t[width];
  }
  int **sub_r = new int *[height];
  for (int i = 0; i < height; i++) {
    sub_r[i] = new int[width];
  }
  // channel 0
  double **r0 = new double *[height];
  for (int i = 0; i < height; i++) {
    r0[i] = new double[width];
  }
  // channel 1
  double **r1 = new double *[height];
  for (int i = 0; i < height; i++) {
    r1[i] = new double[width];
  }

  uint8_t **repic = new uint8_t *[height];
  for (int i = 0; i < height; i++) {
    repic[i] = new uint8_t[width];
  }

  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      // reconstruct image
      sub_dgr[r][c] = dgd[r * dgd_stride + c];
      // src img
      sub_src[r][c] = src[r * src_stride + c];
      // src img-reconstruct image
      sub_r[r][c] = sub_src[r][c] - sub_dgr[r][c];
      // from tflite get channel 0
      r0[r][c] = output[r * 2 * out_stride + c * 2] * max_val;
      // from tflite get channel 1
      r1[r][c] = output[r * 2 * out_stride + c * 2 + 1] * max_val;
    }
  }
  int scale, A0_min, A1_min;
  int qp = qindex / 4;
  if (qp < 17) {
    scale = 16384;
    A0_min = -7;
    A1_min = -5;
  } else if (17 <= qp && qp < 27) {
    scale = 16384;
    A0_min = -12;
    A1_min = -7;
  } else if (27 <= qp && qp < 31) {
    scale = 8192;
    A0_min = -12;
    A1_min = -3;
  } else if (31 <= qp && qp < 37) {
    scale = 8192;
    A0_min = -13;
    A1_min = -10;
  } else if (37 <= qp && qp < 47) {
    scale = 4192;
    A0_min = -13;
    A1_min = -10;
  } else if (47 <= qp && qp < 57) {
    scale = 2046;
    A0_min = -13;
    A1_min = -10;
  } else if (qp > 56) {
    scale = 2046;
    A0_min = -15;
    A1_min = -6;
  }

  int blockSize = frameType;
  int cols = int(ceil(double(height) / blockSize));
  int rows = int(ceil(double(width) / blockSize));
  int number_crlc = cols * rows;
  int *A = new int[(int)number_crlc * 2];
  int index_A = 0;
  int start_row = 0;
  int end_row = 0;
  int start_clow = 0;
  int end_clow = 0;
  int testnum = 10;
  for (int i = 0; i < cols; i++) {
    for (int j = 0; j < rows; j++) {
      if (i == cols - 1) {
        start_clow = height - blockSize;
        end_clow = height;
      } else {
        start_clow = i * blockSize;
        end_clow = (i + 1) * blockSize;
      }
      if (j == rows - 1) {
        start_row = width - blockSize;
        end_row = width;
      } else {
        start_row = j * blockSize;
        end_row = (j + 1) * blockSize;
      }
      if (width < blockSize) {
        start_row = 0;
        end_row = width;
      }
      if (height < blockSize) {
        start_clow = 0;
        end_clow = height;
      }
      int lenth_clows = end_clow - start_clow;
      int lenth_rows = end_row - start_row;

      int lenth = lenth_clows * lenth_rows;
      int *sub_r_flatten = new int[lenth];
      int k = 0;
      for (int i = start_clow; i < end_clow; i++) {
        for (int j = start_row; j < end_row; j++) {
          sub_r_flatten[k] = sub_r[i][j];
          k = k + 1;
        }
      }

      double *sub_r0 = new double[lenth];

      int k_r0 = 0;
      for (int i = start_clow; i < end_clow; i++) {
        for (int j = start_row; j < end_row; j++) {
          sub_r0[k_r0] = r0[i][j];
          k_r0++;
        }
      }

      double *sub_r1 = new double[lenth];
      int k_r1 = 0;
      for (int i = start_clow; i < end_clow; i++) {
        for (int j = start_row; j < end_row; j++) {
          sub_r1[k_r1] = r1[i][j];
          k_r1++;
        }
      }

      double **R = new double *[lenth];
      for (int i = 0; i < lenth; i++) {
        R[i] = new double[2];
      }

      for (int i = 0; i < lenth; i++) {
        for (int j = 0; j < 2; j++) {
          if (j == 0) {
            R[i][j] = sub_r0[i];
          }
          if (j == 1) {
            R[i][j] = sub_r1[i];
          }
        }
      }

      double **R_T = new double *[2];
      for (int i = 0; i < 2; i++) {
        R_T[i] = new double[lenth];
      }

      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < lenth; j++) {
          if (i == 0) {
            R_T[i][j] = sub_r0[j];
          }
          if (i == 1) {
            R_T[i][j] = sub_r1[j];
          }
        }
      }

      double **R_TDotR = new double *[2];
      for (int i = 0; i < 2; i++) {
        R_TDotR[i] = new double[2];
      }

      R_TDotR[0][0] = 0;
      for (int i = 0; i < lenth; i++) {
        R_TDotR[0][0] += R_T[0][i] * R[i][0];
      }
      R_TDotR[0][1] = 0;
      for (int i = 0; i < lenth; i++) {
        R_TDotR[0][1] += R_T[0][i] * R[i][1];
      }
      R_TDotR[1][0] = 0;
      for (int i = 0; i < lenth; i++) {
        R_TDotR[1][0] += R_T[1][i] * R[i][0];
      }
      R_TDotR[1][1] = 0;
      for (int i = 0; i < lenth; i++) {
        R_TDotR[1][1] += R_T[1][i] * R[i][1];
      }

      double value_R_TDotR =
          R_TDotR[0][0] * R_TDotR[1][1] - R_TDotR[0][1] * R_TDotR[1][0];
      double a00 = R_TDotR[1][1] / value_R_TDotR;
      double a01 = -1 * R_TDotR[0][1] / value_R_TDotR;
      double a10 = -1 * R_TDotR[1][0] / value_R_TDotR;
      double a11 = R_TDotR[0][0] / value_R_TDotR;

      double **R_TDotR_inver = new double *[2];
      for (int i = 0; i < 2; i++) {
        R_TDotR_inver[i] = new double[2];
      }
      R_TDotR_inver[0][0] = a00;
      R_TDotR_inver[0][1] = a01;
      R_TDotR_inver[1][0] = a10;
      R_TDotR_inver[1][1] = a11;

      double **mid = new double *[2];
      for (int i = 0; i < 2; i++) {
        mid[i] = new double[lenth];
      }
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < lenth; j++) {
          if (i == 0) {
            mid[i][j] = R_TDotR_inver[0][0] * R_T[0][j] +
                        R_TDotR_inver[0][1] * R_T[1][j];
          }
          if (i == 1) {
            mid[i][j] = R_TDotR_inver[1][0] * R_T[0][j] +
                        R_TDotR_inver[1][1] * R_T[1][j];
          }
        }
      }

      double A0 = 0;
      double A1 = 0;
      for (int i = 0; i < lenth; i++) {
        A0 += mid[0][i] * sub_r_flatten[i];
        A1 += mid[1][i] * sub_r_flatten[i];
      }
      A0 = A0 * scale;
      A1 = A1 * scale;

      A0 = int(round(A0));
      A1 = int(round(A1));
      if (A0 < A0_min) {
        A0 = A0_min;
      }
      if (A0 > A0_min + 15) {
        A0 = A0_min + 15;
      }
      A[index_A] = int(A0);
      index_A = index_A + 1;
      if (A1 < A1_min) {
        A1 = A1_min;
      }
      if (A1 > A1_min + 15) {
        A1 = A1_min + 15;
      }
      A[index_A] = int(A1);
      index_A = index_A + 1;
      for (int i = start_clow; i < end_clow; i++) {
        for (int j = start_row; j < end_row; j++) {
          repic[i][j] = int(round(sub_dgr[i][j] + A0 * r0[i][j] / scale +
                                  A1 * r1[i][j] / scale));
          repic[i][j] = clip_pixel(repic[i][j]);
        }
      }
    }
  }
  ci->num_crlc_unit = (int)number_crlc;
  for (int i = 0; i < number_crlc * 2; i++) {
    if (i % 2 == 0) {
      if (A[i] < A0_min) {
        A[i] = A0_min;
      }
      if (A[i] > A0_min + 15) {
        A[i] = A0_min + 15;
      }
    } else {
      if (A[i] < A1_min) {
        A[i] = A1_min;
      }
      if (A[i] > A1_min + 15) {
        A[i] = A1_min + 15;
      }
    }
    ci->unit_info[i / 2].xqd[i % 2] = (int)A[i];
  }

  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      rst[r * rst_stride + c] = clip_pixel(repic[r][c]);
    }
  }
  return 1;
}

extern "C" int av1_restore_cnn_guided_img_tflite_highbd(
    int qindex, const uint16_t *dgd, int width, int height, int dgd_stride,
    uint16_t *rst, int rst_stride, int num_threads, int bit_depth,
    int is_intra_only, int is_luma, const uint16_t *src, int src_stride,
    CRLCInfo *ci, int frameType) {
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

  uint16_t **sub_dgr = new uint16_t *[height];
  for (int i = 0; i < height; i++) {
    sub_dgr[i] = new uint16_t[width];
  }
  uint16_t **sub_src = new uint16_t *[height];
  for (int i = 0; i < height; i++) {
    sub_src[i] = new uint16_t[width];
  }
  int **sub_r = new int *[height];
  for (int i = 0; i < height; i++) {
    sub_r[i] = new int[width];
  }
  // channel 0
  double **r0 = new double *[height];
  for (int i = 0; i < height; i++) {
    r0[i] = new double[width];
  }
  // channel 1
  double **r1 = new double *[height];
  for (int i = 0; i < height; i++) {
    r1[i] = new double[width];
  }

  uint16_t **repic = new uint16_t *[height];
  for (int i = 0; i < height; i++) {
    repic[i] = new uint16_t[width];
  }

  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      // reconstruct image
      sub_dgr[r][c] = dgd[r * dgd_stride + c];
      // src img
      sub_src[r][c] = src[r * src_stride + c];
      // src img-reconstruct image
      sub_r[r][c] = sub_src[r][c] - sub_dgr[r][c];
      // from tflite get channel 0
      r0[r][c] = output[r * 2 * out_stride + c * 2] * max_val;
      // from tflite get channel 1
      r1[r][c] = output[r * 2 * out_stride + c * 2 + 1] * max_val;
    }
  }
  int scale, A0_min, A1_min;
  int qp = qindex / 4;
  if (qp < 17) {
    scale = 16384;
    A0_min = -7;
    A1_min = -5;
  } else if (17 <= qp && qp < 27) {
    scale = 16384;
    A0_min = -12;
    A1_min = -7;
  } else if (27 <= qp && qp < 31) {
    scale = 8192;
    A0_min = -12;
    A1_min = -3;
  } else if (31 <= qp && qp < 37) {
    scale = 8192;
    A0_min = -13;
    A1_min = -10;
  } else if (37 <= qp && qp < 47) {
    scale = 4192;
    A0_min = -13;
    A1_min = -10;
  } else if (47 <= qp && qp < 57) {
    scale = 2046;
    A0_min = -13;
    A1_min = -10;
  } else if (qp > 56) {
    scale = 2046;
    A0_min = -15;
    A1_min = -6;
  }
  int blockSize = frameType;
  int cols = int(ceil(double(height) / blockSize));
  int rows = int(ceil(double(width) / blockSize));
  int number_crlc = cols * rows;
  int *A = new int[(int)number_crlc * 2];
  int index_A = 0;
  int start_row = 0;
  int end_row = 0;
  int start_clow = 0;
  int end_clow = 0;
  int testnum = 10;
  for (int i = 0; i < cols; i++) {
    for (int j = 0; j < rows; j++) {
      if (i == cols - 1) {
        start_clow = height - blockSize;
        end_clow = height;
      } else {
        start_clow = i * blockSize;
        end_clow = (i + 1) * blockSize;
      }
      if (j == rows - 1) {
        start_row = width - blockSize;
        end_row = width;
      } else {
        start_row = j * blockSize;
        end_row = (j + 1) * blockSize;
      }
      if (width < blockSize) {
        start_row = 0;
        end_row = width;
      }
      if (height < blockSize) {
        start_clow = 0;
        end_clow = height;
      }
      int lenth_clows = end_clow - start_clow;
      int lenth_rows = end_row - start_row;

      int lenth = lenth_clows * lenth_rows;
      int *sub_r_flatten = new int[lenth];
      int k = 0;
      for (int i = start_clow; i < end_clow; i++) {
        for (int j = start_row; j < end_row; j++) {
          sub_r_flatten[k] = sub_r[i][j];
          k = k + 1;
        }
      }

      double *sub_r0 = new double[lenth];

      int k_r0 = 0;
      for (int i = start_clow; i < end_clow; i++) {
        for (int j = start_row; j < end_row; j++) {
          sub_r0[k_r0] = r0[i][j];
          k_r0++;
        }
      }

      double *sub_r1 = new double[lenth];
      int k_r1 = 0;
      for (int i = start_clow; i < end_clow; i++) {
        for (int j = start_row; j < end_row; j++) {
          sub_r1[k_r1] = r1[i][j];
          k_r1++;
        }
      }

      double **R = new double *[lenth];
      for (int i = 0; i < lenth; i++) {
        R[i] = new double[2];
      }

      for (int i = 0; i < lenth; i++) {
        for (int j = 0; j < 2; j++) {
          if (j == 0) {
            R[i][j] = sub_r0[i];
          }
          if (j == 1) {
            R[i][j] = sub_r1[i];
          }
        }
      }

      double **R_T = new double *[2];
      for (int i = 0; i < 2; i++) {
        R_T[i] = new double[lenth];
      }

      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < lenth; j++) {
          if (i == 0) {
            R_T[i][j] = sub_r0[j];
          }
          if (i == 1) {
            R_T[i][j] = sub_r1[j];
          }
        }
      }

      double **R_TDotR = new double *[2];
      for (int i = 0; i < 2; i++) {
        R_TDotR[i] = new double[2];
      }

      R_TDotR[0][0] = 0;
      for (int i = 0; i < lenth; i++) {
        R_TDotR[0][0] += R_T[0][i] * R[i][0];
      }
      R_TDotR[0][1] = 0;
      for (int i = 0; i < lenth; i++) {
        R_TDotR[0][1] += R_T[0][i] * R[i][1];
      }
      R_TDotR[1][0] = 0;
      for (int i = 0; i < lenth; i++) {
        R_TDotR[1][0] += R_T[1][i] * R[i][0];
      }
      R_TDotR[1][1] = 0;
      for (int i = 0; i < lenth; i++) {
        R_TDotR[1][1] += R_T[1][i] * R[i][1];
      }

      double value_R_TDotR =
          R_TDotR[0][0] * R_TDotR[1][1] - R_TDotR[0][1] * R_TDotR[1][0];
      double a00 = R_TDotR[1][1] / value_R_TDotR;
      double a01 = -1 * R_TDotR[0][1] / value_R_TDotR;
      double a10 = -1 * R_TDotR[1][0] / value_R_TDotR;
      double a11 = R_TDotR[0][0] / value_R_TDotR;

      double **R_TDotR_inver = new double *[2];
      for (int i = 0; i < 2; i++) {
        R_TDotR_inver[i] = new double[2];
      }
      R_TDotR_inver[0][0] = a00;
      R_TDotR_inver[0][1] = a01;
      R_TDotR_inver[1][0] = a10;
      R_TDotR_inver[1][1] = a11;

      double **mid = new double *[2];
      for (int i = 0; i < 2; i++) {
        mid[i] = new double[lenth];
      }
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < lenth; j++) {
          if (i == 0) {
            mid[i][j] = R_TDotR_inver[0][0] * R_T[0][j] +
                        R_TDotR_inver[0][1] * R_T[1][j];
          }
          if (i == 1) {
            mid[i][j] = R_TDotR_inver[1][0] * R_T[0][j] +
                        R_TDotR_inver[1][1] * R_T[1][j];
          }
        }
      }

      double A0 = 0;
      double A1 = 0;
      for (int i = 0; i < lenth; i++) {
        A0 += mid[0][i] * sub_r_flatten[i];
        A1 += mid[1][i] * sub_r_flatten[i];
      }
      A0 = A0 * scale;
      A1 = A1 * scale;
      A0 = int(round(A0));
      A1 = int(round(A1));
      if (A0 < A0_min) {
        A0 = A0_min;
      }
      if (A0 > A0_min + 15) {
        A0 = A0_min + 15;
      }
      A[index_A] = int(A0);
      index_A = index_A + 1;
      if (A1 < A1_min) {
        A1 = A1_min;
      }
      if (A1 > A1_min + 15) {
        A1 = A1_min + 15;
      }
      A[index_A] = int(A1);
      index_A = index_A + 1;

      for (int i = start_clow; i < end_clow; i++) {
        for (int j = start_row; j < end_row; j++) {
          repic[i][j] = int(round(sub_dgr[i][j] + A0 * r0[i][j] / scale +
                                  A1 * r1[i][j] / scale));
          repic[i][j] = clip_pixel(repic[i][j]);
        }
      }
    }
  }
  ci->num_crlc_unit = (int)number_crlc;
  for (int i = 0; i < number_crlc * 2; i++) {
    if (i % 2 == 0) {
      if (A[i] < A0_min) {
        A[i] = A0_min;
      }
      if (A[i] > A0_min + 15) {
        A[i] = A0_min + 15;
      }
    } else {
      if (A[i] < A1_min) {
        A[i] = A1_min;
      }
      if (A[i] > A1_min + 15) {
        A[i] = A1_min + 15;
      }
    }
    ci->unit_info[i / 2].xqd[i % 2] = (int)A[i];
  }

  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      rst[r * rst_stride + c] = clip_pixel_highbd(repic[r][c], bit_depth);
    }
  }
  return 1;
}

extern "C" int av1_restore_cnn_guided_decode_img_tflite(
    int qindex, const uint8_t *dgd, int width, int height, int dgd_stride,
    uint8_t *rst, int rst_stride, int num_threads, int is_intra_only,
    int is_luma, CRLCInfo *ci, int frameType) {
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

  uint8_t **sub_dgr = new uint8_t *[height];
  for (int i = 0; i < height; i++) {
    sub_dgr[i] = new uint8_t[width];
  }

  double **r0 = new double *[height];
  for (int i = 0; i < height; i++) {
    r0[i] = new double[width];
  }

  double **r1 = new double *[height];
  for (int i = 0; i < height; i++) {
    r1[i] = new double[width];
  }

  uint8_t **repic = new uint8_t *[height];
  for (int i = 0; i < height; i++) {
    repic[i] = new uint8_t[width];
  }

  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      // sub_dgr[r][c] = dgd[r * in_stride + c];
      sub_dgr[r][c] = dgd[r * dgd_stride + c];
      r0[r][c] = output[r * 2 * out_stride + c * 2] * max_val;
      r1[r][c] = output[r * 2 * out_stride + c * 2 + 1] * max_val;
    }
  }
  int scale;
  int qp = qindex / 4;
  if (qp < 17) {
    scale = 16384;
  } else if (17 <= qp && qp < 27) {
    scale = 16384;
  } else if (27 <= qp && qp < 31) {
    scale = 8192;
  } else if (31 <= qp && qp < 37) {
    scale = 8192;
  } else if (37 <= qp && qp < 47) {
    scale = 4192;
  } else if (47 <= qp && qp < 57) {
    scale = 2046;
  } else {
    scale = 2046;
  }
  int blockSize = frameType;
  double cols = ceil(double(height) / blockSize);
  double rows = ceil(double(width) / blockSize);
  double number_crlc = cols * rows;
  int *A = new int[(int)number_crlc * 2];
  int index_A = 0;
  int start_row = 0;
  int end_row = 0;
  int start_clow = 0;
  int end_clow = 0;
  int num_block = 0;
  int testnum = 10;

  for (int i = 0; i < cols; i++) {
    for (int j = 0; j < rows; j++) {
      if (i == cols - 1) {
        start_clow = height - blockSize;
        end_clow = height;
      } else {
        start_clow = i * blockSize;
        end_clow = (i + 1) * blockSize;
      }

      if (j == rows - 1) {
        start_row = width - blockSize;
        end_row = width;
      } else {
        start_row = j * blockSize;
        end_row = (j + 1) * blockSize;
      }

      if (width < blockSize) {
        start_row = 0;
        end_row = width;
      }

      if (height < blockSize) {
        start_clow = 0;
        end_clow = height;
      }

      int lenth_clows = end_clow - start_clow;
      int lenth_rows = end_row - start_row;

      int lenth = lenth_clows * lenth_rows;
      int *sub_r_flatten = new int[lenth];
      int k = 0;

      double A0 = ci->unit_info[num_block].xqd[0];
      double A1 = ci->unit_info[num_block].xqd[1];

      num_block++;
      for (int i = start_clow; i < end_clow; i++) {
        for (int j = start_row; j < end_row; j++) {
          repic[i][j] = int(round(sub_dgr[i][j] + A0 * r0[i][j] / scale +
                                  A1 * r1[i][j] / scale));
          repic[i][j] = clip_pixel(repic[i][j]);
        }
      }
    }
  }
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      rst[r * rst_stride + c] = clip_pixel(repic[r][c]);
    }
  }
  return 1;
}

extern "C" int av1_restore_cnn_guided_decode_img_tflite_highbd(
    int qindex, const uint16_t *dgd, int width, int height, int dgd_stride,
    uint16_t *rst, int rst_stride, int num_threads, int bit_depth,
    int is_intra_only, int is_luma, CRLCInfo *ci, int frameType) {
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

  uint16_t **sub_dgr = new uint16_t *[height];
  for (int i = 0; i < height; i++) {
    sub_dgr[i] = new uint16_t[width];
  }

  double **r0 = new double *[height];
  for (int i = 0; i < height; i++) {
    r0[i] = new double[width];
  }

  double **r1 = new double *[height];
  for (int i = 0; i < height; i++) {
    r1[i] = new double[width];
  }

  uint16_t **repic = new uint16_t *[height];
  for (int i = 0; i < height; i++) {
    repic[i] = new uint16_t[width];
  }

  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      // sub_dgr[r][c] = dgd[r * in_stride + c];
      sub_dgr[r][c] = dgd[r * dgd_stride + c];
      r0[r][c] = output[r * 2 * out_stride + c * 2] * max_val;
      r1[r][c] = output[r * 2 * out_stride + c * 2 + 1] * max_val;
    }
  }
  int scale;
  int qp = qindex / 4;
  if (qp < 17) {
    scale = 16384;
  } else if (17 <= qp && qp < 27) {
    scale = 16384;
  } else if (27 <= qp && qp < 31) {
    scale = 8192;
  } else if (31 <= qp && qp < 37) {
    scale = 8192;
  } else if (37 <= qp && qp < 47) {
    scale = 4192;
  } else if (47 <= qp && qp < 57) {
    scale = 2046;
  } else {
    scale = 2046;
  }
  int blockSize = frameType;
  double cols = ceil(double(height) / blockSize);
  double rows = ceil(double(width) / blockSize);
  double number_crlc = cols * rows;
  int *A = new int[(int)number_crlc * 2];
  int index_A = 0;
  int start_row = 0;
  int end_row = 0;
  int start_clow = 0;
  int end_clow = 0;
  int num_block = 0;
  int testnum = 10;

  for (int i = 0; i < cols; i++) {
    for (int j = 0; j < rows; j++) {
      if (i == cols - 1) {
        start_clow = height - blockSize;
        end_clow = height;
      } else {
        start_clow = i * blockSize;
        end_clow = (i + 1) * blockSize;
      }

      if (j == rows - 1) {
        start_row = width - blockSize;
        end_row = width;
      } else {
        start_row = j * blockSize;
        end_row = (j + 1) * blockSize;
      }

      if (width < blockSize) {
        start_row = 0;
        end_row = width;
      }

      if (height < blockSize) {
        start_clow = 0;
        end_clow = height;
      }

      int lenth_clows = end_clow - start_clow;
      int lenth_rows = end_row - start_row;

      int lenth = lenth_clows * lenth_rows;
      int *sub_r_flatten = new int[lenth];
      int k = 0;

      double A0 = ci->unit_info[num_block].xqd[0];
      double A1 = ci->unit_info[num_block].xqd[1];

      num_block++;
      for (int i = start_clow; i < end_clow; i++) {
        for (int j = start_row; j < end_row; j++) {
          repic[i][j] = int(round(sub_dgr[i][j] + A0 * r0[i][j] / scale +
                                  A1 * r1[i][j] / scale));
          repic[i][j] = clip_pixel(repic[i][j]);
        }
      }
    }
  }
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      rst[r * rst_stride + c] = clip_pixel(repic[r][c]);
    }
  }
  return 1;
}

extern "C" void av1_restore_cnn_guided_tflite(AV1_COMMON *cm, int num_threads,
                                              YV12_BUFFER_CONFIG *source_frame,
                                              int plane_from, int plane_to) {
  YV12_BUFFER_CONFIG *buf = &cm->cur_frame->buf;
  const int is_intra_only = frame_is_intra_only(cm);
  for (int plane = plane_from; plane <= plane_to; ++plane) {
    const int is_luma = (plane == AOM_PLANE_Y);
    if (cm->seq_params.use_highbitdepth) {
      switch (plane) {
        case AOM_PLANE_Y:
          av1_restore_cnn_guided_img_tflite_highbd(
              cm->base_qindex, CONVERT_TO_SHORTPTR(buf->y_buffer),
              buf->y_crop_width, buf->y_crop_height, buf->y_stride,
              CONVERT_TO_SHORTPTR(buf->y_buffer), buf->y_stride, num_threads,
              cm->seq_params.bit_depth, is_intra_only, is_luma,
              CONVERT_TO_SHORTPTR(source_frame->y_buffer),
              source_frame->y_stride, &cm->crlc_info[0],
              cm->crlc_info->crlc_unit_size);
          break;
        case AOM_PLANE_U:
          av1_restore_cnn_img_tflite_highbd(
              cm->base_qindex, CONVERT_TO_SHORTPTR(buf->u_buffer),
              buf->uv_crop_width, buf->uv_crop_height, buf->uv_stride,
              CONVERT_TO_SHORTPTR(buf->u_buffer), buf->uv_stride, num_threads,
              cm->seq_params.bit_depth, is_intra_only, is_luma);
          break;
        case AOM_PLANE_V:
          av1_restore_cnn_img_tflite_highbd(
              cm->base_qindex, CONVERT_TO_SHORTPTR(buf->v_buffer),
              buf->uv_crop_width, buf->uv_crop_height, buf->uv_stride,
              CONVERT_TO_SHORTPTR(buf->u_buffer), buf->uv_stride, num_threads,
              cm->seq_params.bit_depth, is_intra_only, is_luma);
          break;
        default: assert(0 && "Invalid plane index");
      }
    } else {
      assert(cm->seq_params.bit_depth == 8);
      switch (plane) {
        case AOM_PLANE_Y:
          av1_restore_cnn_guided_img_tflite(
              cm->base_qindex, buf->y_buffer, buf->y_crop_width,
              buf->y_crop_height, buf->y_stride, buf->y_buffer, buf->y_stride,
              num_threads, is_intra_only, is_luma, source_frame->y_buffer,
              source_frame->y_stride, &cm->crlc_info[0],
              cm->crlc_info->crlc_unit_size);
          break;
        case AOM_PLANE_U:
          av1_restore_cnn_img_tflite(
              cm->base_qindex, buf->u_buffer, buf->uv_crop_width,
              buf->uv_crop_height, buf->uv_stride, buf->u_buffer,
              buf->uv_stride, num_threads, is_intra_only, is_luma);
          break;
        case AOM_PLANE_V:
          av1_restore_cnn_img_tflite(
              cm->base_qindex, buf->v_buffer, buf->uv_crop_width,
              buf->uv_crop_height, buf->uv_stride, buf->v_buffer,
              buf->uv_stride, num_threads, is_intra_only, is_luma);
          break;
        default: assert(0 && "Invalid plane index");
      }
    }
  }
}
extern "C" void av1_restore_cnn_guided_decode_tflite(AV1_COMMON *cm,
                                                     int num_threads,
                                                     int plane_from,
                                                     int plane_to) {
  YV12_BUFFER_CONFIG *buf = &cm->cur_frame->buf;
  const int is_intra_only = frame_is_intra_only(cm);
  for (int plane = plane_from; plane <= plane_to; ++plane) {
    const int is_luma = (plane == AOM_PLANE_Y);
    if (cm->seq_params.use_highbitdepth) {
      switch (plane) {
        case AOM_PLANE_Y:
          av1_restore_cnn_guided_decode_img_tflite_highbd(
              cm->base_qindex, CONVERT_TO_SHORTPTR(buf->y_buffer),
              buf->y_crop_width, buf->y_crop_height, buf->y_stride,
              CONVERT_TO_SHORTPTR(buf->y_buffer), buf->y_stride, num_threads,
              cm->seq_params.bit_depth, is_intra_only, is_luma,
              &cm->crlc_info[0], cm->crlc_info->crlc_unit_size);
          break;
        case AOM_PLANE_U:
          av1_restore_cnn_img_tflite_highbd(
              cm->base_qindex, CONVERT_TO_SHORTPTR(buf->u_buffer),
              buf->uv_crop_width, buf->uv_crop_height, buf->uv_stride,
              CONVERT_TO_SHORTPTR(buf->u_buffer), buf->uv_stride, num_threads,
              cm->seq_params.bit_depth, is_intra_only, is_luma);
          break;
        case AOM_PLANE_V:
          av1_restore_cnn_img_tflite_highbd(
              cm->base_qindex, CONVERT_TO_SHORTPTR(buf->v_buffer),
              buf->uv_crop_width, buf->uv_crop_height, buf->uv_stride,
              CONVERT_TO_SHORTPTR(buf->u_buffer), buf->uv_stride, num_threads,
              cm->seq_params.bit_depth, is_intra_only, is_luma);
          break;
        default: assert(0 && "Invalid plane index");
      }
    } else {
      assert(cm->seq_params.bit_depth == 8);
      switch (plane) {
        case AOM_PLANE_Y:
          av1_restore_cnn_guided_decode_img_tflite(
              cm->base_qindex, buf->y_buffer, buf->y_crop_width,
              buf->y_crop_height, buf->y_stride, buf->y_buffer, buf->y_stride,
              num_threads, is_intra_only, is_luma, &cm->crlc_info[0],
              cm->crlc_info->crlc_unit_size);
          break;
        case AOM_PLANE_U:
          av1_restore_cnn_img_tflite(
              cm->base_qindex, buf->u_buffer, buf->uv_crop_width,
              buf->uv_crop_height, buf->uv_stride, buf->u_buffer,
              buf->uv_stride, num_threads, is_intra_only, is_luma);
          break;
        case AOM_PLANE_V:
          av1_restore_cnn_img_tflite(
              cm->base_qindex, buf->v_buffer, buf->uv_crop_width,
              buf->uv_crop_height, buf->uv_stride, buf->v_buffer,
              buf->uv_stride, num_threads, is_intra_only, is_luma);
          break;
        default: assert(0 && "Invalid plane index");
      }
    }
  }
}

#endif  // CONFIG_CNN_CRLC_GUIDED

#if CONFIG_NN_RECON
// Builds and returns the TFlite interpreter.
static std::unique_ptr<tflite::Interpreter> get_nn_recon_tflite_interpreter(
    int width, int height, int num_threads) {
  auto model = tflite::GetModel(tx16x16_tflite);
  tflite::MutableOpResolver resolver;
  RegisterSelectedOpsAllQps(&resolver);
  tflite::InterpreterBuilder builder(model, resolver);

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
  return interpreter;
}

extern "C" int av1_cnn_recon_tflite(uint8_t *dst, int dst_stride, int height,
                                    int width) {
  const int num_threads = 1;
  std::unique_ptr<tflite::Interpreter> interpreter =
      get_nn_recon_tflite_interpreter(width, height, num_threads);

  // Prepare input.
  const int in_stride = width;
  auto input = interpreter->typed_input_tensor<float>(0);
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      input[r * in_stride + c] = static_cast<float>(dst[r * dst_stride + c]);
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
      const int residue = static_cast<int>(output[r * out_stride + c] + 0.5);
      dst[r * dst_stride + c] = clip_pixel(dst[r * dst_stride + c] + residue);
    }
  }
  return 1;
}
#endif  // CONFIG_NN_RECON
