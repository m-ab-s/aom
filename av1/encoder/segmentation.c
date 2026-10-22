/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <limits.h>

#include "aom_mem/aom_mem.h"

#include "av1/common/pred_common.h"
#include "av1/common/tile_common.h"

#include "av1/encoder/cost.h"
#include "av1/encoder/segmentation.h"

void av1_enable_segmentation(struct segmentation *seg) {
  seg->enabled = 1;
  seg->update_map = 1;
  seg->update_data = 1;
  seg->temporal_update = 0;
}

void av1_disable_segmentation(struct segmentation *seg) {
  seg->enabled = 0;
  seg->update_map = 0;
  seg->update_data = 0;
  seg->temporal_update = 0;
}

void av1_reuse_ref_segment_features(AV1_COMMON *cm) {
  struct segmentation *const seg = &cm->seg;
  const RefCntBuffer *const ref = cm->prev_frame;
  // Frames without a primary reference frame always send the features.
  if (!seg->update_data || cm->features.primary_ref_frame == PRIMARY_REF_NONE ||
      ref == NULL || !ref->seg.enabled) {
    return;
  }
  for (int i = 0; i < MAX_SEGMENTS; i++) {
    if (seg->feature_mask[i] != ref->seg.feature_mask[i]) return;
    for (int j = 0; j < SEG_LVL_MAX; j++) {
      if (segfeature_active(seg, i, j) &&
          get_segdata(seg, i, j) != get_segdata(&ref->seg, i, j)) {
        return;
      }
    }
  }
  seg->update_data = 0;
}

void av1_disable_segfeature(struct segmentation *seg, int segment_id,
                            SEG_LVL_FEATURES feature_id) {
  seg->feature_mask[segment_id] &= ~(1u << feature_id);
}

void av1_clear_segdata(struct segmentation *seg, int segment_id,
                       SEG_LVL_FEATURES feature_id) {
  seg->feature_data[segment_id][feature_id] = 0;
}

void av1_reset_segment_features(AV1_COMMON *cm) {
  struct segmentation *seg = &cm->seg;

  // Set up default state for MB feature flags
  seg->enabled = 0;
  seg->update_map = 0;
  seg->update_data = 0;
  av1_clearall_segfeatures(seg);
}
