/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */
#include "av1/encoder/encoder.h"

static AOM_INLINE void av1_set_mv_precision(AV1_COMP *cpi,
                                            MvSubpelPrecision precision,
                                            int cur_frame_force_integer_mv) {
  MACROBLOCK *const x = &cpi->td.mb;
  x->nmvcost[0][0] = &x->nmv_costs[0][0][MV_MAX];
  x->nmvcost[0][1] = &x->nmv_costs[0][1][MV_MAX];
  x->nmvcost[1][0] = &x->nmv_costs[1][0][MV_MAX];
  x->nmvcost[1][1] = &x->nmv_costs[1][1][MV_MAX];
  x->nmvcost[2][0] = &x->nmv_costs[2][0][MV_MAX];
  x->nmvcost[2][1] = &x->nmv_costs[2][1][MV_MAX];
  x->nmvcost[3][0] = &x->nmv_costs[3][0][MV_MAX];
  x->nmvcost[3][1] = &x->nmv_costs[3][1][MV_MAX];

  if (cur_frame_force_integer_mv) {
    cpi->common.fr_mv_precision = MV_SUBPEL_NONE;
  } else {
    cpi->common.fr_mv_precision = cpi->sf.high_precision_mv_usage == QTR_ONLY
                                      ? MV_SUBPEL_QTR_PRECISION
                                      : precision;
  }

#if CONFIG_SB_FLEX_MVRES
  AV1_COMMON *cm = &cpi->common;
  if (frame_is_intra_only(cm) || cm->fr_mv_precision == MV_SUBPEL_NONE) {
    cm->use_sb_mv_precision = 0;
  } else {
    cm->use_sb_mv_precision = 1;
  }
#endif  // CONFIG_SB_FLEX_MVRES
}

void av1_pick_and_set_high_precision_mv(AV1_COMP *cpi, int q);
