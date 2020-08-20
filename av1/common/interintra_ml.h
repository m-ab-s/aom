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

#ifndef AOM_AV1_COMMON_INTERINTRA_ML_H_
#define AOM_AV1_COMMON_INTERINTRA_ML_H_

#include <stddef.h>
#include "av1/common/enums.h"

#ifdef __cplusplus
extern "C" {
#endif

#define INTERINTRA_ML_BORDER 4

// Entry point. Checks if the right block size is passed in.
// Invokes the ML model and stores the output in comp_pred. Note
// that border must be greater than or equal to INTERINTRA_ML_BORDER,
// and represents the amount of border built for the interpredictor
// and intrapredictor (only INTERINTRA_ML_BORDER will be used).
void av1_combine_interintra_ml(INTERINTRA_MODE mode, BLOCK_SIZE plane_bsize,
                               uint8_t *comp_pred, int comp_stride,
                               const uint8_t *inter_pred, int inter_stride,
                               const uint8_t *intra_pred, int intra_stride,
                               int border);

// High bit-depth version of the entry point.
void av1_combine_interintra_ml_highbd(
    INTERINTRA_MODE mode, BLOCK_SIZE plane_bsize, uint8_t *comp_pred8,
    int comp_stride, const uint8_t *inter_pred8, int inter_stride,
    const uint8_t *intra_pred8, int intra_stride, int bd, int border);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_INTERINTRA_ML_H_
