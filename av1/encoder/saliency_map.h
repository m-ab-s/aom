/*
 * Copyright (c) 2023, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#ifndef AOM_AV1_ENCODER_SALIENCY_MAP_H_
#define AOM_AV1_ENCODER_SALIENCY_MAP_H_

typedef struct saliency_feature_map {
  double *buf;  // stores values of the map in 1D array
  int height;
  int width;
} saliency_feature_map;

void av1_set_saliency_map(AV1_COMP *cpi);

#endif  // AOM_AV1_ENCODER_SALIENCY_MAP_H_