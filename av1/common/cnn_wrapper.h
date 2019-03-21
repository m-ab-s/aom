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

#ifndef AOM_AV1_CNN_WRAPPER_H_
#define AOM_AV1_CNN_WRAPPER_H_

#include "av1/common/onyxc_int.h"
#include "av1/common/resize.h"

#ifdef __cplusplus
extern "C" {
#endif

// Minimum base_qindex needed to run CNN_RESTORATION.
#define MIN_CNN_Q_INDEX 100

static INLINE int av1_use_cnn(const AV1_COMMON *cm) {
  return (cm->base_qindex > MIN_CNN_Q_INDEX) && !av1_superres_scaled(cm);
}

// Wrap a call to av1_restore_cnn_plane that
// restores AOM_PLANE_Y using the model provided by trial_model.h.
void av1_restore_cnn_plane_Y_wrapper(AV1_COMMON *cm);

#ifdef __cplusplus
}
#endif
#endif  // AOM_AV1_CNN_WRAPPER_H_
