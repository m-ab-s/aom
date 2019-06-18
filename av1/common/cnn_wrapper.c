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

#include <stdint.h>
#include <stdio.h>

#include "av1/common/cnn.h"
#include "av1/common/cnn_wrapper.h"
#include "av1/common/onyxc_int.h"

#include "av1/models/intra_frame_model/qp22.h"
#include "av1/models/intra_frame_model/qp32.h"
#include "av1/models/intra_frame_model/qp43.h"
#include "av1/models/intra_frame_model/qp53.h"
#include "av1/models/intra_frame_model/qp63.h"

const CNN_CONFIG *av1_get_cnn_config_from_qindex(int qindex,
                                                 FRAME_TYPE frame_type) {
  (void)frame_type;
  if (qindex <= MIN_CNN_Q_INDEX) {
    return NULL;
  } else if (qindex < 108) {
    return &intra_frame_model_qp22;
  } else if (qindex < 148) {
    return &intra_frame_model_qp32;
  } else if (qindex < 192) {
    return &intra_frame_model_qp43;
  } else if (qindex < 232) {
    return &intra_frame_model_qp53;
  } else {
    return &intra_frame_model_qp63;
  }
}

static void restore_cnn_plane_wrapper(AV1_COMMON *cm, int plane,
                                      const CNN_THREAD_DATA *thread_data) {
  const CNN_CONFIG *config = av1_get_cnn_config_from_qindex(
      cm->base_qindex, cm->current_frame.frame_type);
  if (config) {
    av1_restore_cnn_plane(cm, config, plane, thread_data);
  }
}

void av1_encode_restore_cnn(AV1_COMMON *cm, AVxWorker *workers,
                            int num_workers) {
  // TODO(logangw): Add mechanism to restore AOM_PLANE_U and AOM_PLANE_V.
  const CNN_THREAD_DATA thread_data = { num_workers, workers };
  restore_cnn_plane_wrapper(cm, AOM_PLANE_Y, &thread_data);
}

void av1_decode_restore_cnn(AV1_COMMON *cm, AVxWorker *workers,
                            int num_workers) {
  const CNN_THREAD_DATA thread_data = { num_workers, workers };
  restore_cnn_plane_wrapper(cm, AOM_PLANE_Y, &thread_data);
}
