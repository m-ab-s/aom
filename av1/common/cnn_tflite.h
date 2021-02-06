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

#ifndef AOM_AV1_COMMON_CNN_TFLITE_H
#define AOM_AV1_COMMON_CNN_TFLITE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "av1/common/onyxc_int.h"
#include "av1/common/resize.h"
#include "av1/encoder/ratectrl.h"

// Minimum base_qindex needed to run cnn.
#define MIN_CNN_Q_INDEX 67

// Returns true if we are allowed to use CNN for restoration.
static INLINE int av1_use_cnn(const AV1_COMMON *cm) {
  return ((cm->base_qindex > MIN_CNN_Q_INDEX) && !av1_superres_scaled(cm));
}

// Returns true if we are allowed to use CNN for restoration.
static INLINE int av1_use_cnn_encode(const AV1_COMMON *cm,
                                     FRAME_UPDATE_TYPE update_type) {
  const bool is_overlay_update =
      (update_type == OVERLAY_UPDATE || update_type == INTNL_OVERLAY_UPDATE);

  return av1_use_cnn(cm) && !is_overlay_update;
}

// Restores image in 'dgd' with a CNN model using TFlite and stores output in
// 'rst'. Returns true on success.
int av1_restore_cnn_img_tflite(int qindex, const uint8_t *dgd, int width,
                               int height, int dgd_stride, uint8_t *rst,
                               int rst_stride, int num_threads,
                               int is_intra_only, int is_luma);

// Same as 'av1_restore_cnn_img_tflite' for highbd.
int av1_restore_cnn_img_tflite_highbd(int qindex, const uint16_t *dgd,
                                      int width, int height, int dgd_stride,
                                      uint16_t *rst, int rst_stride,
                                      int num_threads, int bit_depth,
                                      int is_intra_only, int is_luma);

struct AV1Common;

// Restore current frame buffer in 'cm' in-place with a CNN model using TFlite.
void av1_restore_cnn_tflite(const struct AV1Common *cm, int num_threads,
                            int plane_from, int plane_to);

#if CONFIG_CNN_CRLC_GUIDED
int av1_restore_cnn_guided_img_tflite(int qindex, const uint8_t *dgd, int width,
                                      int height, int dgd_stride, uint8_t *rst,
                                      int rst_stride, int num_threads,
                                      int is_intra_only, int is_luma,
                                      const uint8_t *src, int src_stride,
                                      CRLCInfo *ci, int frameType);

int av1_restore_cnn_guided_img_tflite_highbd(
    int qindex, const uint16_t *dgd, int width, int height, int dgd_stride,
    uint16_t *rst, int rst_stride, int num_threads, int bit_depth,
    int is_intra_only, int is_luma, const uint16_t *src, int src_stride,
    CRLCInfo *ci, int frameType);

void av1_restore_cnn_guided_tflite(AV1_COMMON *cm, int num_threads,
                                   YV12_BUFFER_CONFIG *source_frame,
                                   int plane_from, int plane_to);

int av1_restore_cnn_guided_decode_img_tflite(int qindex, const uint8_t *dgd,
                                             int width, int height,
                                             int dgd_stride, uint8_t *rst,
                                             int rst_stride, int num_threads,
                                             int is_intra_only, int is_luma,
                                             CRLCInfo *ci, int frameType);

int av1_restore_cnn_guided_decode_img_tflite_highbd(
    int qindex, const uint16_t *dgd, int width, int height, int dgd_stride,
    uint16_t *rst, int rst_stride, int num_threads, int bit_depth,
    int is_intra_only, int is_luma, CRLCInfo *ci, int frameType);

void av1_restore_cnn_guided_decode_tflite(AV1_COMMON *cm, int num_threads,
                                          int plane_from, int plane_to);
#endif  // CONFIG_CNN_CRLC_GUIDED

// Uses CNN model for txfm reconstruction
int av1_cnn_recon_tflite(uint8_t *dst, int dst_stride, int height, int width);

#ifdef __cplusplus
}
#endif

#endif  // AOM_AV1_COMMON_CNN_TFLITE_H
