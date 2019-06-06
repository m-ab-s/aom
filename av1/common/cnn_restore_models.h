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

#ifndef AOM_AV1_COMMON_CNN_RESTORE_MODELS_H_
#define AOM_AV1_COMMON_CNN_RESTORE_MODELS_H_

#include "av1/common/cnn.h"
#include "av1/common/onyxc_int.h"
#include "av1/common/resize.h"

#include "av1/models/intra_frame_model/qp22.h"
#include "av1/models/intra_frame_model/qp32.h"
#include "av1/models/intra_frame_model/qp43.h"
#include "av1/models/intra_frame_model/qp53.h"
#include "av1/models/intra_frame_model/qp63.h"

#ifdef __cplusplus
extern "C" {
#endif

// Minimum base_qindex needed to run cnn.
#define MIN_CNN_Q_INDEX 100

static INLINE int av1_use_cnn(const AV1_COMMON *cm) {
  return ((cm->base_qindex > MIN_CNN_Q_INDEX) && !av1_superres_scaled(cm));
}

#ifdef __cplusplus
}
#endif

#endif  // AOM_AV1_COMMON_CNN_RESTORE_MODELS_H_
