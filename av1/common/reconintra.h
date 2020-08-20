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

#ifndef AOM_AV1_COMMON_RECONINTRA_H_
#define AOM_AV1_COMMON_RECONINTRA_H_

#include <stdlib.h>

#include "aom/aom_integer.h"
#include "av1/common/blockd.h"
#include "av1/common/onyxc_int.h"

#ifdef __cplusplus
extern "C" {
#endif

// Calculate the number of rows / columns available in the reference
// frame.
int av1_intra_top_available(const MACROBLOCKD *xd, int plane);
int av1_intra_left_available(const MACROBLOCKD *xd, int plane);

// Calculate the number of rows / columns unavailable in the reference
// frame. av1_intra_right_unavailable indicates the number of columns
// unavailable on the top of the block (starting from the right),
// av1_intra_bottom_unavailable indicates the number of rows unavailable
// to the left of the block (starting from bottom).
int av1_intra_right_unavailable(const MACROBLOCKD *xd, int plane,
                                TX_SIZE tx_size);
int av1_intra_bottom_unavailable(const MACROBLOCKD *xd, int plane,
                                 TX_SIZE tx_size);

// Equivalent to memmove, but looks at the bit-depth and converts the
// pointer to dst16 (and the amount of data moved) if in high bitdepth mode.
void av1_bd_memmove(uint8_t *dst, const uint8_t *ref, size_t n, bool is_hbd);

// Equivalent to memset, but looks at the bit-depth and copies the value
// every uint16_t space if in high bitdepth mode.
void av1_bd_memset(uint8_t *dst, int c, size_t n, bool is_hbd);

// Extends the intra-predictor to have a border region consisting of the
// reference frame; border region is to the top-left and assumed to be
// offset negatively from the passed in pointer (which points to the start
// of the regular intra-prediction). Note that it is possible for the
// bit-depth to be 8 and is_hbd to be true, if high-bitdepth pipeline
// is forced on.
void av1_extend_intra_border(const uint8_t *ref, int ref_stride, uint8_t *dst,
                             int dst_stride, int top_rows_available,
                             int right_cols_unavailable,
                             int left_cols_available,
                             int bottom_rows_unavailable, int width, int height,
                             int border, aom_bit_depth_t bd, bool is_hbd);
void av1_init_intra_predictors(void);
void av1_predict_intra_block_facade(const AV1_COMMON *cm, MACROBLOCKD *xd,
                                    int plane, int blk_col, int blk_row,
                                    TX_SIZE tx_size);
void av1_predict_intra_block(const AV1_COMMON *cm, const MACROBLOCKD *xd,
                             int wpx, int hpx, TX_SIZE tx_size,
                             PREDICTION_MODE mode, int angle_delta,
                             int use_palette,
                             FILTER_INTRA_MODE filter_intra_mode,
#if CONFIG_ADAPT_FILTER_INTRA
                             ADAPT_FILTER_INTRA_MODE adapt_filter_intra_mode,
#endif
#if CONFIG_DERIVED_INTRA_MODE
                             int use_derived_intra_mode,
#endif  // CONFIG_DERIVED_INTRA_MODE
                             const uint8_t *ref, int ref_stride, uint8_t *dst,
                             int dst_stride, int col_off, int row_off,
                             int plane);

// Mapping of interintra to intra mode for use in the intra component
static const PREDICTION_MODE interintra_to_intra_mode[INTERINTRA_MODES] = {
  DC_PRED,
  V_PRED,
  H_PRED,
  SMOOTH_PRED,
#if CONFIG_ILLUM_MCOMP
  DC_PRED
#endif  // CONFIG_ILLUM_MCOMP
#if CONFIG_INTERINTRA_ML
      // The intra-prediction is not used directly by the ML models. Use
      // simplest intra-predictor for speed.
      DC_PRED,
  DC_PRED,
  DC_PRED,
  DC_PRED,
  DC_PRED,
  DC_PRED,
  DC_PRED,
  DC_PRED,
  DC_PRED,
  DC_PRED
#endif  // CONFIG_INTERINTRA_ML
};

// Mapping of intra mode to the interintra mode
static const INTERINTRA_MODE intra_to_interintra_mode[INTRA_MODES] = {
  II_DC_PRED, II_V_PRED, II_H_PRED, II_V_PRED,      II_SMOOTH_PRED, II_V_PRED,
  II_H_PRED,  II_H_PRED, II_V_PRED, II_SMOOTH_PRED, II_SMOOTH_PRED
};

#define FILTER_INTRA_SCALE_BITS 4

static INLINE int av1_use_angle_delta(BLOCK_SIZE bsize) {
  return bsize >= BLOCK_8X8;
}

static INLINE int av1_allow_intrabc(const AV1_COMMON *const cm) {
  return frame_is_intra_only(cm) && cm->allow_screen_content_tools &&
         cm->allow_intrabc;
}

static INLINE int av1_filter_intra_allowed_bsize(const AV1_COMMON *const cm,
                                                 BLOCK_SIZE bs) {
  if (!cm->seq_params.enable_filter_intra || bs == BLOCK_INVALID) return 0;

  return block_size_wide[bs] <= 32 && block_size_high[bs] <= 32;
}

static INLINE int av1_filter_intra_allowed(const AV1_COMMON *const cm,
                                           const MB_MODE_INFO *mbmi) {
  return mbmi->mode == DC_PRED &&
         mbmi->palette_mode_info.palette_size[0] == 0 &&
         av1_filter_intra_allowed_bsize(cm, mbmi->sb_type);
}

#if CONFIG_ADAPT_FILTER_INTRA
static INLINE int av1_adapt_filter_intra_allowed_bsize(
    const AV1_COMMON *const cm, BLOCK_SIZE bs) {
  if (!cm->seq_params.enable_adapt_filter_intra || bs == BLOCK_INVALID)
    return 0;

  return block_size_wide[bs] <= 128 && block_size_high[bs] <= 128;
}

static INLINE int av1_adapt_filter_intra_allowed(const AV1_COMMON *const cm,
                                                 const MB_MODE_INFO *mbmi) {
  return mbmi->mode == DC_PRED &&
         mbmi->palette_mode_info.palette_size[0] == 0 &&
         mbmi->filter_intra_mode_info.use_filter_intra == 0 &&
         av1_adapt_filter_intra_allowed_bsize(cm, mbmi->sb_type);
}
#endif  // CONFIG_ADAPT_FILTER_INTRA

extern const int8_t av1_filter_intra_taps[FILTER_INTRA_MODES][8][8];

static const int16_t dr_intra_derivative[90] = {
  // More evenly spread out angles and limited to 10-bit
  // Values that are 0 will never be used
  //                    Approx angle
  0,    0, 0,        //
  1023, 0, 0,        // 3, ...
  547,  0, 0,        // 6, ...
  372,  0, 0, 0, 0,  // 9, ...
  273,  0, 0,        // 14, ...
  215,  0, 0,        // 17, ...
  178,  0, 0,        // 20, ...
  151,  0, 0,        // 23, ... (113 & 203 are base angles)
  132,  0, 0,        // 26, ...
  116,  0, 0,        // 29, ...
  102,  0, 0, 0,     // 32, ...
  90,   0, 0,        // 36, ...
  80,   0, 0,        // 39, ...
  71,   0, 0,        // 42, ...
  64,   0, 0,        // 45, ... (45 & 135 are base angles)
  57,   0, 0,        // 48, ...
  51,   0, 0,        // 51, ...
  45,   0, 0, 0,     // 54, ...
  40,   0, 0,        // 58, ...
  35,   0, 0,        // 61, ...
  31,   0, 0,        // 64, ...
  27,   0, 0,        // 67, ... (67 & 157 are base angles)
  23,   0, 0,        // 70, ...
  19,   0, 0,        // 73, ...
  15,   0, 0, 0, 0,  // 76, ...
  11,   0, 0,        // 81, ...
  7,    0, 0,        // 84, ...
  3,    0, 0,        // 87, ...
};

// Get the shift (up-scaled by 256) in X w.r.t a unit change in Y.
// If angle > 0 && angle < 90, dx = -((int)(256 / t));
// If angle > 90 && angle < 180, dx = (int)(256 / t);
// If angle > 180 && angle < 270, dx = 1;
static INLINE int av1_get_dx(int angle) {
#if CONFIG_DERIVED_INTRA_MODE
  if ((angle > 0 && angle < 90) || (angle > 90 && angle < 180)) {
    if (angle > 90) angle = 180 - angle;
    int dx = dr_intra_derivative[angle];
    if (!dx) dx = (int)round(64 / tan(angle * PI / 180));
    return dx;
  } else {
    return 1;
  }
#endif  // CONFIG_DERIVED_INTRA_MODE

  if (angle > 0 && angle < 90) {
    return dr_intra_derivative[angle];
  } else if (angle > 90 && angle < 180) {
    return dr_intra_derivative[180 - angle];
  } else {
    // In this case, we are not really going to use dx. We may return any value.
    return 1;
  }
}

// Get the shift (up-scaled by 256) in Y w.r.t a unit change in X.
// If angle > 0 && angle < 90, dy = 1;
// If angle > 90 && angle < 180, dy = (int)(256 * t);
// If angle > 180 && angle < 270, dy = -((int)(256 * t));
static INLINE int av1_get_dy(int angle) {
#if CONFIG_DERIVED_INTRA_MODE
  if ((angle > 90 && angle < 180) || (angle > 180 && angle < 270)) {
    if (angle > 90 && angle < 180) {
      angle = angle - 90;
    } else {
      angle = 270 - angle;
    }
    int dy = dr_intra_derivative[angle];
    if (!dy) dy = (int)round(64 / tan(angle * PI / 180));
    return dy;
  } else {
    return 1;
  }
#endif  // CONFIG_DERIVED_INTRA_MODE

  if (angle > 90 && angle < 180) {
    return dr_intra_derivative[angle - 90];
  } else if (angle > 180 && angle < 270) {
    return dr_intra_derivative[270 - angle];
  } else {
    // In this case, we are not really going to use dy. We may return any value.
    return 1;
  }
}

static INLINE int av1_use_intra_edge_upsample(int bs0, int bs1, int delta,
                                              int type) {
  const int d = abs(delta);
  const int blk_wh = bs0 + bs1;
  if (d == 0 || d >= 40) return 0;
  return type ? (blk_wh <= 8) : (blk_wh <= 16);
}
#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // AOM_AV1_COMMON_RECONINTRA_H_
