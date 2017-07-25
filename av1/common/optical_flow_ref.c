/*
 * Copyright (c) 2017, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include "av1/common/optical_flow_ref.h"
#include <float.h>
#include <math.h>
#include <time.h>
#include "./aom_config.h"
#include "aom_mem/aom_mem.h"
#include "aom_scale/aom_scale.h"
#include "av1/common/alloccommon.h"
#include "av1/common/onyxc_int.h"
#include "av1/common/sparse_linear_solver.h"

#if CONFIG_OPFL
double timeinit, timesub, timesolve, timeder;  // global timer for debug purpose
static int optical_flow_warp_filter[16][8] = {
    {0, 0, 0, 128, 0, 0, 0, 0},      {0, 2, -6, 126, 8, -2, 0, 0},
    {0, 2, -10, 122, 18, -4, 0, 0},  {0, 2, -12, 116, 28, -8, 2, 0},
    {0, 2, -14, 110, 38, -10, 2, 0}, {0, 2, -14, 102, 48, -12, 2, 0},
    {0, 2, -16, 94, 58, -12, 2, 0},  {0, 2, -14, 84, 66, -12, 2, 0},
    {0, 2, -14, 76, 76, -14, 2, 0},  {0, 2, -12, 66, 84, -14, 2, 0},
    {0, 2, -12, 58, 94, -16, 2, 0},  {0, 2, -12, 48, 102, -14, 2, 0},
    {0, 2, -10, 38, 110, -14, 2, 0}, {0, 2, -8, 28, 116, -12, 2, 0},
    {0, 0, -4, 18, 122, -10, 2, 0},  {0, 0, -2, 8, 126, -6, 2, 0}};

/*
 * Use optical flow method to interpolate a reference frame.
 *
 * Input:
 * ref0, ref1: the reference buffers
 * mv_left: motion field from ref0 to dst
 * mv_right: motion field from ref1 to dst
 * dst_pos: the ratio of the dst frame position (e.g. if ref0 @ time 0,
 *          ref1 @ time 3, dst @ time 1, then dst_pos = 0.333)
 * filename: the yuv file for debug purpose. Use null if no need.
 *
 * Output:
 * dst: pointer to the interpolated frame buffer
 */
void optical_flow_get_ref(YV12_BUFFER_CONFIG *ref0, YV12_BUFFER_CONFIG *ref1,
                          int_mv *mv_left, int_mv *mv_right,
                          YV12_BUFFER_CONFIG *dst, double dst_pos,
                          char *filename) {
  int width, height;
  width = dst->y_width;
  height = dst->y_height;

  timeinit = 0;
  timesolve = 0;
  timesub = 0;
  timeder = 0;

  // Use zero MV if no MV is passed in
  int allocl = 0, allocr = 0;
  if (mv_left == NULL) {
    mv_left = aom_calloc(width / 4 * height / 4, sizeof(int_mv));
    allocl = 1;
    for (int i = 0; i < width / 4 * height / 4; i++) mv_left[i].as_int = 0;
  }
  if (mv_right == NULL) {
    mv_right = aom_calloc(width / 4 * height / 4, sizeof(int_mv));
    allocr = 1;
    for (int i = 0; i < width / 4 * height / 4; i++) mv_right[i].as_int = 0;
  }

  // initialize buffers
  DB_MV *mf_last[3];
  DB_MV *mf_new[3];
  DB_MV *mf_med[3];  // for motion field after median filter
  YV12_BUFFER_CONFIG *ref0_ds[3];
  YV12_BUFFER_CONFIG *ref1_ds[3];
  int wid = width, hgt = height;
  // allocate and downscale frames for each level
  ref0_ds[0] = ref0;
  ref1_ds[0] = ref1;
  for (int l = 0; l < 3; l++) {
    mf_last[l] = aom_calloc(
        (wid + 2 * AVG_MF_BORDER) * (hgt + 2 * AVG_MF_BORDER), sizeof(DB_MV));
    mf_new[l] = aom_calloc(
        (wid + 2 * AVG_MF_BORDER) * (hgt + 2 * AVG_MF_BORDER), sizeof(DB_MV));
    mf_med[l] = aom_calloc(
        (wid + 2 * AVG_MF_BORDER) * (hgt + 2 * AVG_MF_BORDER), sizeof(DB_MV));
    if (l != 0) {
      ref0_ds[l] = aom_calloc(1, sizeof(YV12_BUFFER_CONFIG));
      ref1_ds[l] = aom_calloc(1, sizeof(YV12_BUFFER_CONFIG));
      aom_alloc_frame_buffer(ref0_ds[l], wid, hgt, 1, 1, 0,
                             AOM_BORDER_IN_PIXELS, 0);
      aom_alloc_frame_buffer(ref1_ds[l], wid, hgt, 1, 1, 0,
                             AOM_BORDER_IN_PIXELS, 0);
    }
    wid /= 2;
    hgt /= 2;
  }
  uint8_t temp_buffer[256 * 8];
  for (int i = 0; i < ref0->y_height; i++) {
    for (int j = 0; j < ref0->y_width; j++) {
      ref0_ds[0]->y_buffer[i * ref0->y_stride + j] =
          ref0->y_buffer[i * ref0->y_stride + j];
      ref1_ds[0]->y_buffer[i * ref0->y_stride + j] =
          ref1->y_buffer[i * ref0->y_stride + j];
    }
  }
  for (int l = 1; l < 3; l++) {
    aom_scale_frame(ref0_ds[l - 1], ref0_ds[l], temp_buffer, 8, 2, 1, 2, 1, 0);
    aom_scale_frame(ref1_ds[l - 1], ref1_ds[l], temp_buffer, 8, 2, 1, 2, 1, 0);
  }

  // create a single motion field (current method) in the w/4 h/4 scale level
  create_motion_field(mv_left, mv_right, mf_last[2], width, height,
                      width / 4 + 2 * AVG_MF_BORDER);

  // temporary buffers for MF median filtering
  double mv_r[25], mv_c[25], left[25], right[25];
  // estimate optical flow at each level
  for (int l = 2; l >= 0; l--) {
    wid = width >> l;
    hgt = height >> l;
    int mvstr = wid + 2 * AVG_MF_BORDER;
    // use optical flow to refine our motion field
    refine_motion_field(ref0_ds[l], ref1_ds[l], mf_last[l], mf_new[l], l,
                        dst_pos);
    DB_MV *mf_start_new = mf_new[l] + AVG_MF_BORDER * mvstr + AVG_MF_BORDER;
    DB_MV *mf_start_med = mf_med[l] + AVG_MF_BORDER * mvstr + AVG_MF_BORDER;
    // pad for median filter (may not need)
    pad_motion_field_border(mf_start_new, wid, hgt, mvstr);
    DB_MV tempmv;
    for (int i = 0; i < hgt; i++) {
      for (int j = 0; j < wid; j++) {
        if (USE_MEDIAN_FILTER) {
          tempmv = median_2D_MV_5x5(mf_start_new + i * mvstr + j, mvstr, mv_r,
                                    mv_c, left, right);
          mf_start_med[i * mvstr + j].row = tempmv.row;
          mf_start_med[i * mvstr + j].col = tempmv.col;
        } else {
          mf_start_med[i * mvstr + j].row = mf_start_new[i * mvstr + j].row;
          mf_start_med[i * mvstr + j].col = mf_start_new[i * mvstr + j].col;
        }
      }
    }
    pad_motion_field_border(mf_start_med, wid, hgt, mvstr);

    if (l != 0) {
      // upscale mv to the next lower level
      int mvstr_next = wid * 2 + 2 * AVG_MF_BORDER;
      DB_MV *mf_start_next =
          mf_last[l - 1] + AVG_MF_BORDER * mvstr_next + AVG_MF_BORDER;
      upscale_mv_by_2(mf_start_med, wid, hgt, mvstr, mf_start_next, mvstr_next);
    }
  }

  // interpolate to get our reference frame
  interp_optical_flow(ref0, ref1, mf_med[0], dst, dst_pos);

  // TODO(bohan): dump the frames for now for debug
  if (filename) {
    write_image_opfl(ref0, filename);
    write_image_opfl(dst, filename);
    write_image_opfl(ref1, filename);
  }

  // free buffers
  for (int l = 0; l < 3; l++) {
    aom_free(mf_last[l]);
    aom_free(mf_new[l]);
    aom_free(mf_med[l]);
    if (l != 0) {
      aom_free_frame_buffer(ref0_ds[l]);
      aom_free_frame_buffer(ref1_ds[l]);
      aom_free(ref0_ds[l]);
      aom_free(ref1_ds[l]);
    }
  }
  if (allocl) {
    aom_free(mv_left);
  }
  if (allocr) {
    aom_free(mv_right);
  }

  // TODO(bohan): output time usage for debug for now.
  printf("\n");
  printf("der time: %.4f, sub time: %.4f, init time: %.4f, solve time: %.4f\n",
         timeder, timesub, timeinit, timesolve);
  fflush(stdout);
  return;
}

/*
 * use optical flow method to calculate motion field of a specific level.
 *
 * Input:
 * ref0, ref1: reference frames
 * mf_last: initial motion field
 * level: current scale level, 0 = original, 1 = 0.5, 2 = 0.25, etc.
 * dstpos: dst frame position
 *
 * Output:
 * mf_new: pointer to calculated motion field
 */
void refine_motion_field(YV12_BUFFER_CONFIG *ref0, YV12_BUFFER_CONFIG *ref1,
                         DB_MV *mf_last, DB_MV *mf_new, int level,
                         double dstpos) {
  int count = 0;
  double last_cost = DBL_MAX;
  double new_cost = last_cost;
  int width = ref0->y_width, height = ref0->y_height;
  int mvstr = width + 2 * AVG_MF_BORDER;
  double as_scale_factor = 1;  // annealing factor for laplacian multiplier
  // iteratively warp and estimate motion field
  while (count < MAX_ITER_OPTICAL_FLOW + 2 - level) {
#if FAST_OPTICAL_FLOW
    if (level == 2) {
      new_cost = iterate_update_mv(ref0, ref1, mf_last, mf_new, level, dstpos,
                                   as_scale_factor);
    } else {
      new_cost = iterate_update_mv_fast(ref0, ref1, mf_last, mf_new, level,
                                        dstpos, as_scale_factor);
    }
#else
    new_cost = iterate_update_mv(ref0, ref1, mf_last, mf_new, level, dstpos,
                                 as_scale_factor);
#endif
    // prepare for the next iteration
    DB_MV *mv_start = mf_last + AVG_MF_BORDER * mvstr + AVG_MF_BORDER;
    DB_MV *mv_start_new = mf_new + AVG_MF_BORDER * mvstr + AVG_MF_BORDER;
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        mv_start[i * mvstr + j].row += mv_start_new[i * mvstr + j].row;
        mv_start[i * mvstr + j].col += mv_start_new[i * mvstr + j].col;
        mv_start_new[i * mvstr + j].row = mv_start[i * mvstr + j].row;
        mv_start_new[i * mvstr + j].col = mv_start[i * mvstr + j].col;
      }
    }
    if (new_cost >= last_cost) {
      break;
    }
    last_cost = new_cost;
    count++;
    as_scale_factor *= OPFL_ANNEAL_FACTOR;
  }
  return;
}
#endif  // CONFIG_OPFL
