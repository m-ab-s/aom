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

#ifndef AOM_AV1_ENCODER_MV_PREC_H_
#define AOM_AV1_ENCODER_MV_PREC_H_

#include "av1/encoder/encoder.h"
#include "av1/encoder/speed_features.h"

// Q threshold for high precision mv.
#define HIGH_PRECISION_MV_QTHRESH 128
#if !CONFIG_REALTIME_ONLY
void av1_collect_mv_stats(AV1_COMP *cpi, int current_q);

static AOM_INLINE int av1_frame_allows_smart_mv(const AV1_COMP *cpi) {
  const int gf_group_index = cpi->gf_group.index;
  const int gf_update_type = cpi->gf_group.update_type[gf_group_index];
  return !frame_is_intra_only(&cpi->common) &&
         !(gf_update_type == INTNL_OVERLAY_UPDATE ||
           gf_update_type == KFFLT_OVERLAY_UPDATE ||
           gf_update_type == OVERLAY_UPDATE);
}
#endif  // !CONFIG_REALTIME_ONLY

static AOM_INLINE void av1_set_high_precision_mv(AV1_COMP *cpi,
                                                 MvSubpelPrecision precision) {
  FeatureFlags *features = &cpi->common.features;
  features->fr_mv_precision = precision;
#if CONFIG_FLEX_MVRES
  features->use_sb_mv_precision = 0;
#endif  // CONFIG_FLEX_MVRES
}

void av1_pick_and_set_high_precision_mv(AV1_COMP *cpi, int qindex);

#endif  // AOM_AV1_ENCODER_MV_PREC_H_
