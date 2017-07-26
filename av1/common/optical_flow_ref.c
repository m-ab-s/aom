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

/*
 * Update motion field at each iteration by solving linear equations directly.
 *
 * Input:
 * ref0, ref1: reference frames
 * mf_last: initial motion field
 * level: current scale level, 0 = original, 1 = 0.5, 2 = 0.25, etc.
 * dstpos: dst frame position
 * scale: scale the laplacian multiplier to perform "annealing"
 *
 * Output:
 * mf_new: pointer to calculated motion field
 */
double iterate_update_mv(YV12_BUFFER_CONFIG *ref0, YV12_BUFFER_CONFIG *ref1,
                         DB_MV *mf_last, DB_MV *mf_new, int level,
                         double dstpos, double scale) {
  double *Ex, *Ey, *Et;
  double a_squared = OF_A_SQUARED * scale;
  double cost = 0;

  //  uint8_t *y0 = ref0->y_buffer;
  int width = ref0->y_width, height = ref0->y_height;
  int mvstr = width + 2 * AVG_MF_BORDER;
  DB_MV *mv_start = mf_last + AVG_MF_BORDER * mvstr + AVG_MF_BORDER;
  DB_MV *mv_start_new = mf_new + AVG_MF_BORDER * mvstr + AVG_MF_BORDER;
  int i, j;

  // allocate buffers
  Ex = aom_calloc(width * height, sizeof(double));
  Ey = aom_calloc(width * height, sizeof(double));
  Et = aom_calloc(width * height, sizeof(double));
  YV12_BUFFER_CONFIG *buffer0 = aom_calloc(1, sizeof(YV12_BUFFER_CONFIG));
  YV12_BUFFER_CONFIG *buffer1 = aom_calloc(1, sizeof(YV12_BUFFER_CONFIG));
  aom_alloc_frame_buffer(buffer0, width, height, 1, 1, 0, AOM_BORDER_IN_PIXELS,
                         0);
  aom_alloc_frame_buffer(buffer1, width, height, 1, 1, 0, AOM_BORDER_IN_PIXELS,
                         0);
  int *row_pos = aom_calloc(width * height * 12, sizeof(int));
  int *col_pos = aom_calloc(width * height * 12, sizeof(int));
  double *values = aom_calloc(width * height * 12, sizeof(double));
  double *mv_vec = aom_calloc(width * height * 2, sizeof(double));
  double *b = aom_calloc(width * height * 2, sizeof(double));
  double *last_mv_vec = mv_vec;

  clock_t start, end;
  start = clock();
  // warp ref1 back to ref0 => buffer
  if (level == 0)
    warp_optical_flow_fwd(ref0, ref1, mv_start, mvstr, buffer0, dstpos);
  else
    warp_optical_flow_fwd_bilinear(ref0, ref1, mv_start, mvstr, buffer0,
                                   dstpos);
  if (level == 0)
    warp_optical_flow_back(ref1, ref0, mv_start, mvstr, buffer1, 1 - dstpos);
  else
    warp_optical_flow_back_bilinear(ref1, ref0, mv_start, mvstr, buffer1,
                                    1 - dstpos);
  end = clock();
  timesub += (double)(end - start) / CLOCKS_PER_SEC;

  start = clock();
  // Calculate partial derivatives
  opfl_get_derivatives(Ex, Ey, Et, buffer0, buffer1, dstpos);
  end = clock();
  timeder += (double)(end - start) / CLOCKS_PER_SEC;

  start = clock();
  // construct and solve A*mv_vec = b
  SPARSE_MTX A, M, F;
  for (j = 0; j < width; j++) {
    for (i = 0; i < height; i++) {
      last_mv_vec[j * height + i] = mv_start[i * mvstr + j].row;
    }
  }
  for (j = 0; j < width; j++) {
    for (i = 0; i < height; i++) {
      last_mv_vec[width * height + j * height + i] =
          mv_start[i * mvstr + j].col;
    }
  }
  // get laplacian filter matrix F; Currently using:
  //    0,  1, 0
  //    1, -4, 1
  //    0,  1, 0
  // M = [F 0; 0 F].
  int c = 0;
  for (j = 0; j < width; j++) {
    for (i = 0; i < height; i++) {
      int center = -4, up = 1, low = 1, left = 1, right = 1;
      if (i == 0) {
        center = center + up;
        up = 0;
      } else if (i == height - 1) {
        center = center + low;
        low = 0;
      }
      if (j == 0) {
        center = center + left;
        left = 0;
      } else if (j == width - 1) {
        center = center + right;
        right = 0;
      }

      if (up) row_pos[c] = j * height + i;
      col_pos[c] = j * height + i - 1;
      values[c] = up;
      c++;
      if (left) row_pos[c] = j * height + i;
      col_pos[c] = (j - 1) * height + i;
      values[c] = left;
      c++;
      row_pos[c] = j * height + i;
      col_pos[c] = j * height + i;
      values[c] = center;
      c++;
      if (right) row_pos[c] = j * height + i;
      col_pos[c] = (j + 1) * height + i;
      values[c] = right;
      c++;
      if (low) row_pos[c] = j * height + i;
      col_pos[c] = j * height + i + 1;
      values[c] = low;
      c++;
    }
  }
  init_sparse_mtx(row_pos, col_pos, values, c, height * width, height * width,
                  &F);
  init_combine_sparse_mtx(&F, &F, &M, 0, 0, height * width, height * width,
                          2 * height * width, 2 * height * width);
  constant_multiply_sparse_matrix(&M, a_squared);
  // construct A
  int offset = height * width;
  c = 0;
  for (j = 0; j < width; j++) {
    for (i = 0; i < height; i++) {
      int center = -4, up = 1, low = 1, left = 1, right = 1;
      if (i == 0) {
        center = center + up;
        up = 0;
      } else if (i == height - 1) {
        center = center + low;
        low = 0;
      }
      if (j == 0) {
        center = center + left;
        left = 0;
      } else if (j == width - 1) {
        center = center + right;
        right = 0;
      }

      if (up) {
        row_pos[c] = j * height + i;
        col_pos[c] = j * height + i - 1;
        values[c] = -a_squared * (double)up;
        c++;
      }
      if (left) {
        row_pos[c] = j * height + i;
        col_pos[c] = (j - 1) * height + i;
        values[c] = -a_squared * (double)left;
        c++;
      }

      row_pos[c] = j * height + i;
      col_pos[c] = j * height + i;
      values[c] =
          Ex[i * width + j] * Ex[i * width + j] - a_squared * (double)center;
      c++;

      if (right) {
        row_pos[c] = j * height + i;
        col_pos[c] = (j + 1) * height + i;
        values[c] = -a_squared * (double)right;
        c++;
      }
      if (low) {
        row_pos[c] = j * height + i;
        col_pos[c] = j * height + i + 1;
        values[c] = -a_squared * (double)low;
        c++;
      }
    }
  }
  for (j = 0; j < width; j++) {
    for (i = 0; i < height; i++) {
      int center = -4, up = 1, low = 1, left = 1, right = 1;
      if (i == 0) {
        center = center + up;
        up = 0;
      } else if (i == height - 1) {
        center = center + low;
        low = 0;
      }
      if (j == 0) {
        center = center + left;
        left = 0;
      } else if (j == width - 1) {
        center = center + right;
        right = 0;
      }

      if (up) {
        row_pos[c] = offset + j * height + i;
        col_pos[c] = offset + j * height + i - 1;
        values[c] = -a_squared * (double)up;
        c++;
      }
      if (left) {
        row_pos[c] = offset + j * height + i;
        col_pos[c] = offset + (j - 1) * height + i;
        values[c] = -a_squared * (double)left;
        c++;
      }

      row_pos[c] = offset + j * height + i;
      col_pos[c] = offset + j * height + i;
      values[c] =
          Ey[i * width + j] * Ey[i * width + j] - a_squared * (double)center;
      c++;

      if (right) {
        row_pos[c] = offset + j * height + i;
        col_pos[c] = offset + (j + 1) * height + i;
        values[c] = -a_squared * (double)right;
        c++;
      }
      if (low) {
        row_pos[c] = offset + j * height + i;
        col_pos[c] = offset + j * height + i + 1;
        values[c] = -a_squared * (double)low;
        c++;
      }
    }
  }
  for (j = 0; j < width; j++) {
    for (i = 0; i < height; i++) {
      row_pos[c] = offset + j * height + i;
      col_pos[c] = j * height + i;
      values[c] = Ex[i * width + j] * Ey[i * width + j];
      c++;
    }
  }
  for (j = 0; j < width; j++) {
    for (i = 0; i < height; i++) {
      row_pos[c] = j * height + i;
      col_pos[c] = offset + j * height + i;
      values[c] = Ex[i * width + j] * Ey[i * width + j];
      c++;
    }
  }
  init_sparse_mtx(row_pos, col_pos, values, c, 2 * width * height,
                  2 * width * height, &A);
  // get b
  mtx_vect_multi_right(&M, last_mv_vec, b, 2 * height * width);
  for (j = 0; j < width; j++) {
    for (i = 0; i < height; i++) {
      b[j * height + i] =
          b[j * height + i] - Et[i * width + j] * Ex[i * width + j];
    }
  }
  for (j = 0; j < width; j++) {
    for (i = 0; i < height; i++) {
      b[j * height + i + height * width] =
          b[j * height + i + height * width] -
          Et[i * width + j] * Ey[i * width + j];
    }
  }
  end = clock();
  timeinit += (double)(end - start) / CLOCKS_PER_SEC;

  start = clock();
  // solve
  conjugate_gradient_sparse(&A, b, 2 * width * height, mv_vec);
  end = clock();
  timesolve += (double)(end - start) / CLOCKS_PER_SEC;

  // reshape motion field to 2D
  cost = 0;
  for (j = 0; j < width; j++) {
    for (i = 0; i < height; i++) {
      mv_start_new[i * mvstr + j].row = mv_vec[j * height + i];
      cost += mv_vec[j * height + i] * mv_vec[j * height + i];
    }
  }
  for (j = 0; j < width; j++) {
    for (i = 0; i < height; i++) {
      mv_start_new[i * mvstr + j].col = mv_vec[width * height + j * height + i];
      cost += mv_vec[width * height + j * height + i] *
              mv_vec[width * height + j * height + i];
    }
  }

  // free buffers
  aom_free(Et);
  aom_free(Ex);
  aom_free(Ey);
  aom_free_frame_buffer(buffer0);
  aom_free(buffer0);
  aom_free_frame_buffer(buffer1);
  aom_free(buffer1);
  aom_free(row_pos);
  aom_free(col_pos);
  aom_free(values);
  free_sparse_mtx_elems(&A);
  free_sparse_mtx_elems(&F);
  free_sparse_mtx_elems(&M);
  aom_free(mv_vec);
  aom_free(b);

  cost = sqrt(cost);  // 2 norm
  return cost;
}

/*
 * Update motion field at each iteration by a fast iterative method.
 *
 * Input:
 * ref0, ref1: reference frames
 * mf_last: initial motion field
 * level: current scale level, 0 = original, 1 = 0.5, 2 = 0.25, etc.
 * dstpos: dst frame position
 * scale: scale the laplacian multiplier to perform "annealing"
 *
 * Output:
 * mf_new: pointer to calculated motion field
 */
double iterate_update_mv_fast(YV12_BUFFER_CONFIG *ref0,
                              YV12_BUFFER_CONFIG *ref1, DB_MV *mf_last,
                              DB_MV *mf_new, int level, double dstpos,
                              double scale) {
  double *Ex, *Ey, *Et;
  double a_squared = OF_A_SQUARED * scale;
  double cost = 0;

  //  uint8_t *y0 = ref0->y_buffer;
  int width = ref0->y_width, height = ref0->y_height;
  int mvstr = width + 2 * AVG_MF_BORDER;
  DB_MV *mv_start = mf_last + AVG_MF_BORDER * mvstr + AVG_MF_BORDER;
  DB_MV *mv_start_new = mf_new + AVG_MF_BORDER * mvstr + AVG_MF_BORDER;
  int i, j;

  // allocate buffers
  Ex = aom_calloc(width * height, sizeof(double));
  Ey = aom_calloc(width * height, sizeof(double));
  Et = aom_calloc(width * height, sizeof(double));
  YV12_BUFFER_CONFIG *buffer0 = aom_calloc(1, sizeof(YV12_BUFFER_CONFIG));
  YV12_BUFFER_CONFIG *buffer1 = aom_calloc(1, sizeof(YV12_BUFFER_CONFIG));
  aom_alloc_frame_buffer(buffer0, width, height, 1, 1, 0, AOM_BORDER_IN_PIXELS,
                         0);
  aom_alloc_frame_buffer(buffer1, width, height, 1, 1, 0, AOM_BORDER_IN_PIXELS,
                         0);

  clock_t start, end;
  start = clock();
  // warp ref1 back to ref0 => buffer
  if (level == 0)
    warp_optical_flow_fwd(ref0, ref1, mv_start, mvstr, buffer0, dstpos);
  else
    warp_optical_flow_fwd_bilinear(ref0, ref1, mv_start, mvstr, buffer0,
                                   dstpos);
  if (level == 0)
    warp_optical_flow_back(ref1, ref0, mv_start, mvstr, buffer1, 1 - dstpos);
  else
    warp_optical_flow_back_bilinear(ref1, ref0, mv_start, mvstr, buffer1,
                                    1 - dstpos);
  end = clock();
  timesub += (double)(end - start) / CLOCKS_PER_SEC;

  start = clock();
  // Calculate partial derivatives
  opfl_get_derivatives(Ex, Ey, Et, buffer0, buffer1, dstpos);
  end = clock();
  timeder += (double)(end - start) / CLOCKS_PER_SEC;

  // iterative solver
  start = clock();
  DB_MV *tempmv;
  DB_MV *bufmv = aom_calloc(height * mvstr, sizeof(DB_MV));
  DB_MV *lp_last = aom_calloc(height * mvstr, sizeof(DB_MV));
  double *denorm = aom_calloc(height * width, sizeof(double));
  DB_MV avg;
  // get the laplacian of initial motion field
  int i0, i1, j0, j1;
  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      bufmv[i * mvstr + j].row = 0;
      bufmv[i * mvstr + j].col = 0;
      lp_last[i * mvstr + j].row = 0;
      lp_last[i * mvstr + j].col = 0;
      if (i == 0) {
        i0 = 0;
        i1 = i + 1;
      } else if (i == height - 1) {
        i1 = height - 1;
        i0 = i - 1;
      } else {
        i0 = i - 1;
        i1 = i + 1;
      }
      if (j == 0) {
        j0 = 0;
        j1 = i + 1;
      } else if (j == width - 1) {
        j1 = width - 1;
        j0 = j - 1;
      } else {
        j0 = j - 1;
        j1 = j + 1;
      }
      lp_last[i * mvstr + j].row += 0.25 * mv_start[i0 * mvstr + j].row +
                                    0.25 * mv_start[i1 * mvstr + j].row +
                                    0.25 * mv_start[i * mvstr + j0].row +
                                    0.25 * mv_start[i * mvstr + j1].row;
      lp_last[i * mvstr + j].row -= mv_start[i * mvstr + j].row;
      lp_last[i * mvstr + j].col += 0.25 * mv_start[i0 * mvstr + j].col +
                                    0.25 * mv_start[i1 * mvstr + j].col +
                                    0.25 * mv_start[i * mvstr + j0].col +
                                    0.25 * mv_start[i * mvstr + j1].col;
      lp_last[i * mvstr + j].col -= mv_start[i * mvstr + j].col;
      denorm[i * width + j] = 16 * a_squared +
                              Ex[i * width + j] * Ex[i * width + j] +
                              Ey[i * width + j] * Ey[i * width + j];
    }
  }
  // calculate the motion field
  for (int k = 0; k < MAX_ITER_FAST_OPFL; k++) {
    for (i = 0; i < height; i++) {
      for (j = 0; j < width; j++) {
        avg.row = 0;
        avg.col = 0;
        if (i == 0) {
          i0 = 0;
          i1 = i + 1;
        } else if (i == height - 1) {
          i1 = height - 1;
          i0 = i - 1;
        } else {
          i0 = i - 1;
          i1 = i + 1;
        }
        if (j == 0) {
          j0 = 0;
          j1 = i + 1;
        } else if (j == width - 1) {
          j1 = width - 1;
          j0 = j - 1;
        } else {
          j0 = j - 1;
          j1 = j + 1;
        }
        avg.row += 0.25 * bufmv[i0 * mvstr + j].row +
                   0.25 * bufmv[i1 * mvstr + j].row +
                   0.25 * bufmv[i * mvstr + j0].row +
                   0.25 * bufmv[i * mvstr + j1].row;
        avg.row += lp_last[i * mvstr + j].row;
        avg.col += 0.25 * bufmv[i0 * mvstr + j].col +
                   0.25 * bufmv[i1 * mvstr + j].col +
                   0.25 * bufmv[i * mvstr + j0].col +
                   0.25 * bufmv[i * mvstr + j1].col;
        avg.col += lp_last[i * mvstr + j].col;
        mv_start_new[i * mvstr + j].row =
            avg.row - Ex[i * width + j] *
                          (Ex[i * width + j] * avg.row +
                           Ey[i * width + j] * avg.col + Et[i * width + j]) /
                          denorm[i * width + j];
        mv_start_new[i * mvstr + j].col =
            avg.col - Ey[i * width + j] *
                          (Ex[i * width + j] * avg.row +
                           Ey[i * width + j] * avg.col + Et[i * width + j]) /
                          denorm[i * width + j];
      }
    }
    if (k < MAX_ITER_FAST_OPFL - 1) {
      tempmv = bufmv;
      bufmv = mv_start_new;
      mv_start_new = tempmv;
    }
  }

  aom_free(bufmv);
  aom_free(lp_last);
  aom_free(denorm);
  end = clock();
  timesolve += (double)(end - start) / CLOCKS_PER_SEC;

  // reshape motion field to 2D
  cost = 0;
  for (j = 0; j < width; j++) {
    for (i = 0; i < height; i++) {
      cost += mv_start_new[i * mvstr + j].row * mv_start_new[i * mvstr + j].row;
    }
  }
  for (j = 0; j < width; j++) {
    for (i = 0; i < height; i++) {
      cost += mv_start_new[i * mvstr + j].col * mv_start_new[i * mvstr + j].col;
    }
  }

  // free buffers
  aom_free(Et);
  aom_free(Ex);
  aom_free(Ey);
  aom_free_frame_buffer(buffer0);
  aom_free(buffer0);
  aom_free_frame_buffer(buffer1);
  aom_free(buffer1);

  cost = sqrt(cost);  // 2 norm
  return cost;
}

void opfl_get_derivatives(double *Ex, double *Ey, double *Et,
                          YV12_BUFFER_CONFIG *buffer0,
                          YV12_BUFFER_CONFIG *buffer1, double dstpos) {
  int lh = DERIVATIVE_FILTER_LENGTH;
  int hleft = (lh - 1) / 2;
  double filter[DERIVATIVE_FILTER_LENGTH] = {
      -1.0 / 60, 9.0 / 60, -45.0 / 60, 0, 45.0 / 60, -9.0 / 60, 1.0 / 60};
  int idx, i, j;
  int width = buffer0->y_width;
  int height = buffer0->y_height;
  int stride = buffer0->y_stride;
  // horizontal derivative filter
  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      Ex[i * width + j] = 0;
      for (int k = 0; k < lh; k++) {
        idx = j + k - hleft;
        if (idx < 0)
          idx = 0;
        else if (idx > width - 1)
          idx = width - 1;
        Ex[i * width + j] +=
            filter[k] * (double)(buffer0->y_buffer[i * stride + idx]) *
                (1 - dstpos) +
            filter[k] * (double)(buffer1->y_buffer[i * stride + idx]) * dstpos;
      }
    }
  }
  // vertical derivative filter
  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      Ey[i * width + j] = 0;
      for (int k = 0; k < lh; k++) {
        idx = i + k - hleft;
        if (idx < 0)
          idx = 0;
        else if (idx > height - 1)
          idx = height - 1;
        Ey[i * width + j] +=
            filter[k] * (double)(buffer0->y_buffer[idx * stride + j]) *
                (1 - dstpos) +
            filter[k] * (double)(buffer1->y_buffer[idx * stride + j]) * dstpos;
      }
    }
  }
  // time derivative
  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      Et[i * width + j] = (double)(buffer1->y_buffer[i * stride + j]) -
                          (double)(buffer0->y_buffer[i * stride + j]);
    }
  }
}

/*
 * Warp the Y component of src to dst according to the motion field
 * Motion field points back from dst to src
 *
 * Input:
 * src: frame to be warped
 * ref: when src not available, use ref and assume 0 motion field
 * mv_start: points to the motion field (to the start of frame, skips border)
 * mvstr: motion field stride
 * dstpos: distance from src to dst
 *
 * Output:
 * dst: pointer to the warped frame
 */
void warp_optical_flow_back(YV12_BUFFER_CONFIG *src, YV12_BUFFER_CONFIG *ref,
                            DB_MV *mf_start, int mvstr, YV12_BUFFER_CONFIG *dst,
                            double dstpos) {
  int width = src->y_width;
  int height = src->y_height;
  int stride = src->y_stride;
  uint8_t *srcy = src->y_buffer;
  uint8_t *dsty = dst->y_buffer;
  uint8_t *refy = ref->y_buffer;

  double ii, jj, di, dj;
  int i0, j0;

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      ii = i + mf_start[i * mvstr + j].col * dstpos;
      jj = j + mf_start[i * mvstr + j].row * dstpos;
      i0 = floor(ii);
      di = ii - i0;
      j0 = floor(jj);
      dj = jj - j0;
      if (i0 < 0 || i0 > height - 1 || j0 < 0 || j0 > width - 1) {
        dsty[i * stride + j] = refy[i * stride + j];
        continue;
      }
      dsty[i * stride + j] =
          get_sub_pel_y(srcy + i0 * stride + j0, stride, di, dj);
    }
  }
}

/*
 * Warp the Y component of src to dst using bilinear method
 * Motion field points back from dst to src
 */
void warp_optical_flow_back_bilinear(YV12_BUFFER_CONFIG *src,
                                     YV12_BUFFER_CONFIG *ref, DB_MV *mf_start,
                                     int mvstr, YV12_BUFFER_CONFIG *dst,
                                     double dstpos) {
  int width = src->y_width;
  int height = src->y_height;
  int stride = src->y_stride;
  uint8_t *srcy = src->y_buffer;
  uint8_t *dsty = dst->y_buffer;
  uint8_t *refy = ref->y_buffer;

  double ii, jj, di, dj, temp;
  int i0, j0, i1, j1;

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      ii = i + mf_start[i * mvstr + j].col * dstpos;
      jj = j + mf_start[i * mvstr + j].row * dstpos;
      i0 = floor(ii);
      di = 1 - ii + i0;
      i1 = i0 + 1;
      j0 = floor(jj);
      dj = 1 - jj + j0;
      j1 = j0 + 1;
      if (i0 < 0 || i0 > height - 1 || j0 < 0 || j0 > width - 1) {
        dsty[i * stride + j] = refy[i * stride + j];
        continue;
      }
      temp = di * dj * (double)srcy[i0 * stride + j0] +
             di * (1 - dj) * (double)srcy[i0 * stride + j1] +
             (1 - di) * dj * (double)srcy[i1 * stride + j0] +
             (1 - di) * (1 - dj) * (double)srcy[i1 * stride + j1];
      dsty[i * stride + j] = (uint8_t)(temp + 0.5);
    }
  }
}

/*
 * Warp the Y component of src to dst
 * Motion field points forward from src to dst
 */
void warp_optical_flow_fwd(YV12_BUFFER_CONFIG *src, YV12_BUFFER_CONFIG *ref,
                           DB_MV *mf_start, int mvstr, YV12_BUFFER_CONFIG *dst,
                           double dstpos) {
  int width = src->y_width;
  int height = src->y_height;
  int stride = src->y_stride;
  uint8_t *srcy = src->y_buffer;
  uint8_t *dsty = dst->y_buffer;
  uint8_t *refy = ref->y_buffer;

  double ii, jj, di, dj;
  int i0, j0;

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      ii = i - mf_start[i * mvstr + j].col * dstpos;
      jj = j - mf_start[i * mvstr + j].row * dstpos;
      i0 = floor(ii);
      di = ii - i0;
      j0 = floor(jj);
      dj = jj - j0;
      if (i0 < 0 || i0 > height - 1 || j0 < 0 || j0 > width - 1) {
        dsty[i * stride + j] = refy[i * stride + j];
        continue;
      }
      dsty[i * stride + j] =
          get_sub_pel_y(srcy + i0 * stride + j0, stride, di, dj);
    }
  }
}

/*
 * Warp the Y component of src to dst using bilinear method
 * Motion field points forward from src to dst
 */
void warp_optical_flow_fwd_bilinear(YV12_BUFFER_CONFIG *src,
                                    YV12_BUFFER_CONFIG *ref, DB_MV *mf_start,
                                    int mvstr, YV12_BUFFER_CONFIG *dst,
                                    double dstpos) {
  int width = src->y_width;
  int height = src->y_height;
  int stride = src->y_stride;
  uint8_t *srcy = src->y_buffer;
  uint8_t *dsty = dst->y_buffer;
  uint8_t *refy = ref->y_buffer;

  double ii, jj, di, dj, temp;
  int i0, j0, i1, j1;

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      ii = i - mf_start[i * mvstr + j].col * dstpos;
      jj = j - mf_start[i * mvstr + j].row * dstpos;
      i0 = floor(ii);
      di = 1 - ii + i0;
      i1 = i0 + 1;
      j0 = floor(jj);
      dj = 1 - jj + j0;
      j1 = j0 + 1;
      if (i0 < 0 || i0 > height - 1 || j0 < 0 || j0 > width - 1) {
        dsty[i * stride + j] = refy[i * stride + j];
        continue;
      }
      temp = di * dj * (double)srcy[i0 * stride + j0] +
             di * (1 - dj) * (double)srcy[i0 * stride + j1] +
             (1 - di) * dj * (double)srcy[i1 * stride + j0] +
             (1 - di) * (1 - dj) * (double)srcy[i1 * stride + j1];
      dsty[i * stride + j] = (uint8_t)(temp + 0.5);
    }
  }
}

/*
 * Interpolate references according to the motion field
 *
 * Input:
 * src0, src1: the reference frames
 * mf_start: the motion field start pointer
 * mvstr: motion field stride
 * dstpos: position of the interpolated frame
 * method: the blend method to be used
 *
 * Output:
 * dst: pointer to the interpolated frame
 */
void warp_optical_flow(YV12_BUFFER_CONFIG *src0, YV12_BUFFER_CONFIG *src1,
                       DB_MV *mf_start, int mvstr, YV12_BUFFER_CONFIG *dst,
                       double dstpos, OPFL_BLEND_METHOD method) {
  if (method == OPFL_DIFF_SELECT) {
    warp_optical_flow_diff_select(src0, src1, mf_start, mvstr, dst, dstpos);
    return;
  }
  int width = src0->y_width;
  int height = src0->y_height;
  int stride = src0->y_stride;
  int uvstride = src0->uv_stride;

  uint8_t *src0y = src0->y_buffer;
  uint8_t *src1y = src1->y_buffer;
  uint8_t *dsty = dst->y_buffer;
  uint8_t *src0u = src0->u_buffer;
  uint8_t *src0v = src0->v_buffer;
  uint8_t *src1u = src1->u_buffer;
  uint8_t *src1v = src1->v_buffer;
  uint8_t *dstu = dst->u_buffer;
  uint8_t *dstv = dst->v_buffer;

  double ii0, jj0, di0, dj0, di0uv, dj0uv;
  double ii1, jj1, di1, dj1, di1uv, dj1uv;
  int i0, j0;
  int i1, j1;
  double dstpel_y, dstpel_u, dstpel_v;
  double dstpel_y0 = 0, dstpel_y1 = 0;
  double pos;
  int do_uv;
  int use0, use1, inside0, inside1;
  int nearest = ((dstpos <= 0.5) ? 0 : 1);

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      pos = dstpos;
      ii0 = i - mf_start[i * mvstr + j].col * dstpos;
      jj0 = j - mf_start[i * mvstr + j].row * dstpos;
      ii1 = i + mf_start[i * mvstr + j].col * dstpos;
      jj1 = j + mf_start[i * mvstr + j].row * dstpos;
      i0 = floor(ii0);
      di0 = ii0 - i0;
      j0 = floor(jj0);
      dj0 = jj0 - j0;
      i1 = floor(ii1);
      di1 = ii1 - i1;
      j1 = floor(jj1);
      dj1 = jj1 - j1;

      do_uv = i % 2 == 0 && j % 2 == 0;  // TODO(bohan) only considering 420 now
      if (do_uv) {
        di0uv = di0 / 2 + 0.5 * ((i0 % 2 + 2) % 2);
        dj0uv = dj0 / 2 + 0.5 * ((j0 % 2 + 2) % 2);
        di1uv = di1 / 2 + 0.5 * ((i1 % 2 + 2) % 2);
        dj1uv = dj1 / 2 + 0.5 * ((j1 % 2 + 2) % 2);
      }

      // Check availability of the references.
      // If one ref is outside then do not use it.
      inside0 = (i0 >= 0 && i0 < height - 1 && j0 >= 0 && j0 < width - 1);
      inside1 = (i1 >= 0 && i1 < height - 1 && j1 >= 0 && j1 < width - 1);
      use0 = inside0 == inside1 || inside0;
      use1 = inside0 == inside1 || inside1;

      // If use nearest single method, then use only one reference
      if (method == OPFL_NEAREST_SINGLE) {
        if (use0 && use1) {
          use0 = (nearest == 0);
          use1 = (nearest == 1);
        }
      }

      // calculate subpel Y refs
      if (use0) {
        dstpel_y0 =
            (double)get_sub_pel_y(src0y + i0 * stride + j0, stride, di0, dj0);
      }
      if (use1) {
        dstpel_y1 =
            (double)get_sub_pel_y(src1y + i1 * stride + j1, stride, di1, dj1);
      }

      // If use diff single method, check the pixels when the refs do not agree
      if (method == OPFL_DIFF_SINGLE) {
        if (use0 && use1) {
          if (fabs(dstpel_y0 - dstpel_y1) > OPTICAL_FLOW_DIFF_THRES) {
            use0 = (nearest == 0);
            use1 = (nearest == 1);
          }
        }
      }

      if (use0 && !use1)
        pos = 0;
      else if (!use0 && use1)
        pos = 1;

      // blend
      dstpel_y = 0;
      dstpel_u = 0;
      dstpel_v = 0;
      if (use0) {
        dstpel_y += dstpel_y0 * (1 - pos);
        if (do_uv) {
          dstpel_u +=
              (double)get_sub_pel_uv(
                  src0u + (int)floor(ii0 / 2) * uvstride + (int)floor(jj0 / 2),
                  uvstride, di0uv, dj0uv) *
              (1 - pos);
          dstpel_v +=
              (double)get_sub_pel_uv(
                  src0v + (int)floor(ii0 / 2) * uvstride + (int)floor(jj0 / 2),
                  uvstride, di0uv, dj0uv) *
              (1 - pos);
        }
      }
      if (use1) {
        dstpel_y += dstpel_y1 * pos;
        if (do_uv) {
          dstpel_u +=
              (double)get_sub_pel_uv(
                  src1u + (int)floor(ii1 / 2) * uvstride + (int)floor(jj1 / 2),
                  uvstride, di1uv, dj1uv) *
              pos;
          dstpel_v +=
              (double)get_sub_pel_uv(
                  src1v + (int)floor(ii1 / 2) * uvstride + (int)floor(jj1 / 2),
                  uvstride, di1uv, dj1uv) *
              pos;
        }
      }
      dsty[i * stride + j] = dstpel_y + 0.5;
      if (do_uv) {
        dstu[i / 2 * uvstride + j / 2] = dstpel_u + 0.5;
        dstv[i / 2 * uvstride + j / 2] = dstpel_v + 0.5;
      }
    }
  }
}

/*
 * Interpolate references according to the motion field
 * when the refs do not agree, use more advanced selection
 *
 * Input:
 * src0, src1: the reference frames
 * mf_start: the motion field start pointer
 * mvstr: motion field stride
 * dstpos: position of the interpolated frame
 *
 * Output:
 * dst: pointer to the interpolated frame
 */
void warp_optical_flow_diff_select(YV12_BUFFER_CONFIG *src0,
                                   YV12_BUFFER_CONFIG *src1, DB_MV *mf_start,
                                   int mvstr, YV12_BUFFER_CONFIG *dst,
                                   double dstpos) {
  int width = src0->y_width;
  int height = src0->y_height;
  int stride = src0->y_stride;
  int uvstride = src0->uv_stride;

  uint8_t *src0y = src0->y_buffer;
  uint8_t *src1y = src1->y_buffer;
  uint8_t *dsty = dst->y_buffer;
  uint8_t *src0u = src0->u_buffer;
  uint8_t *src0v = src0->v_buffer;
  uint8_t *src1u = src1->u_buffer;
  uint8_t *src1v = src1->v_buffer;
  uint8_t *dstu = dst->u_buffer;
  uint8_t *dstv = dst->v_buffer;

  double ii0, jj0, di0, dj0, di0uv, dj0uv;
  double ii1, jj1, di1, dj1, di1uv, dj1uv;
  int i0, j0;
  int i1, j1;
  double dstpel_y, dstpel_u, dstpel_v;
  double pos;
  int do_uv;

  double *used0 = aom_calloc(height * width, sizeof(double));
  double *used1 = aom_calloc(height * width, sizeof(double));
  // refid: -1=unset; 0=ref0; 1=ref1; 2=both
  int *refid = aom_calloc(height * width, sizeof(int));
  int *refid_mode = aom_calloc(height * width, sizeof(int));
  double *dstpel_y0 = aom_calloc(height * width, sizeof(double));
  double *dstpel_y1 = aom_calloc(height * width, sizeof(double));

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      used0[i * width + j] = 0;
      used1[i * width + j] = 0;
      refid[i * width + j] = -1;
    }
  }
  // first check all pixels to see if the refs agree
  // also note down which pixels in refs are used for reference
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      pos = dstpos;
      ii0 = i - mf_start[i * mvstr + j].col * dstpos;
      jj0 = j - mf_start[i * mvstr + j].row * dstpos;
      ii1 = i + mf_start[i * mvstr + j].col * dstpos;
      jj1 = j + mf_start[i * mvstr + j].row * dstpos;
      i0 = floor(ii0);
      di0 = ii0 - i0;
      j0 = floor(jj0);
      dj0 = jj0 - j0;
      i1 = floor(ii1);
      di1 = ii1 - i1;
      j1 = floor(jj1);
      dj1 = jj1 - j1;

      int inside0 = (i0 >= 0 && i0 < height - 1 && j0 >= 0 && j0 < width - 1);
      int inside1 = (i1 >= 0 && i1 < height - 1 && j1 >= 0 && j1 < width - 1);
      int use0 = inside0 == inside1 || inside0;
      int use1 = inside0 == inside1 || inside1;
      if (inside0 && !inside1)
        pos = 0;
      else if (inside1 && !inside0)
        pos = 1;

      dstpel_y0[i * width + j] =
          (double)get_sub_pel_y(src0y + i0 * stride + j0, stride, di0, dj0);
      dstpel_y1[i * width + j] =
          (double)get_sub_pel_y(src1y + i1 * stride + j1, stride, di1, dj1);
      // check if the refs are similar
      if (inside0 && inside1) {
        if (fabs(dstpel_y0[i * width + j] - dstpel_y1[i * width + j]) >
            OPTICAL_FLOW_DIFF_THRES) {
          continue;
        }
      }
      if (inside0 || inside1) {
        used0[(i0)*width + j0] += use0 * (1 - di0) * (1 - dj0);
        used0[(i0 + 1) * width + j0] += use0 * (di0) * (1 - dj0);
        used0[(i0)*width + j0 + 1] += use0 * (1 - di0) * (dj0);
        used0[(i0 + 1) * width + j0 + 1] += use0 * (di0) * (dj0);
        used1[(i1)*width + j1] += use1 * (1 - di1) * (1 - dj1);
        used1[(i1 + 1) * width + j1] += use1 * (di1) * (1 - dj1);
        used1[(i1)*width + j1 + 1] += use1 * (1 - di1) * (dj1);
        used1[(i1 + 1) * width + j1 + 1] += use1 * (di1) * (dj1);
      }  // ignore when both out of bound
      if (use0 && use1)
        refid[i * width + j] = 2;
      else if (use0 && !use1)
        refid[i * width + j] = 0;
      else if (!use0 && use1)
        refid[i * width + j] = 1;
      else
        assert(0);
    }
  }

  // calculate how many time each pixel is referenced
  double totalused0, totalused1;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if (refid[i * width + j] >= 0) continue;
      totalused0 = 0;
      totalused1 = 0;
      pos = dstpos;
      ii0 = i - mf_start[i * mvstr + j].col * dstpos;
      jj0 = j - mf_start[i * mvstr + j].row * dstpos;
      ii1 = i + mf_start[i * mvstr + j].col * dstpos;
      jj1 = j + mf_start[i * mvstr + j].row * dstpos;
      i0 = floor(ii0);
      di0 = ii0 - i0;
      j0 = floor(jj0);
      dj0 = jj0 - j0;
      i1 = floor(ii1);
      di1 = ii1 - i1;
      j1 = floor(jj1);
      dj1 = jj1 - j1;

      int inside0 = (i0 >= 0 && i0 < height - 1 && j0 >= 0 && j0 < width - 1);
      int inside1 = (i1 >= 0 && i1 < height - 1 && j1 >= 0 && j1 < width - 1);
      if (inside0 && !inside1)
        pos = 0;
      else if (inside1 && !inside0)
        pos = 1;

      assert(inside1 && inside0);

      totalused0 += used0[(i0)*width + j0] * (1 - di0) * (1 - dj0);
      totalused0 += used0[(i0 + 1) * width + j0] * (di0) * (1 - dj0);
      totalused0 += used0[(i0)*width + j0 + 1] * (1 - di0) * (dj0);
      totalused0 += used0[(i0 + 1) * width + j0 + 1] * (di0) * (dj0);
      totalused1 += used1[(i1)*width + j1] * (1 - di1) * (1 - dj1);
      totalused1 += used1[(i1 + 1) * width + j1] * (di1) * (1 - dj1);
      totalused1 += used1[(i1)*width + j1 + 1] * (1 - di1) * (dj1);
      totalused1 += used1[(i1 + 1) * width + j1 + 1] * (di1) * (dj1);

      // if one ref pixel has been referenced by other pixels more than the
      // other ref by a threshold, then decide there is occlusion
      if (totalused0 < (totalused0 + totalused1) * OPTICAL_FLOW_REF_THRES) {
        refid[i * width + j] = 0;
      } else if (totalused1 <
                 (totalused0 + totalused1) * OPTICAL_FLOW_REF_THRES) {
        refid[i * width + j] = 1;
      } else {
        refid[i * width + j] = ((dstpos <= 0.5) ? 0 : 1);
      }
    }
  }

  // All refs for each pixel have been decided now.
  // Do mode filter to get rid of outliers
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if (i == 0 || i == height - 1 || j == 0 || j == width - 1) {
        refid_mode[i * width + j] = refid[i * width + j];
        continue;
      }
      refid_mode[i * width + j] =
          ref_mode_filter_3x3(refid + i * width + j, width, dstpos);
    }
  }

  // blend according to the refs selected
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int use0, use1;
      pos = dstpos;
      if (refid_mode[i * width + j] == 0) {
        use0 = 1;
        use1 = 0;
        pos = 0;
      } else if (refid_mode[i * width + j] == 1) {
        use0 = 0;
        use1 = 1;
        pos = 1;
      } else if (refid_mode[i * width + j] == 2) {
        use0 = 1;
        use1 = 1;
      } else {
        assert(0);
      }
      ii0 = i - mf_start[i * mvstr + j].col * dstpos;
      jj0 = j - mf_start[i * mvstr + j].row * dstpos;
      ii1 = i + mf_start[i * mvstr + j].col * dstpos;
      jj1 = j + mf_start[i * mvstr + j].row * dstpos;
      i0 = floor(ii0);
      di0 = ii0 - i0;
      j0 = floor(jj0);
      dj0 = jj0 - j0;
      i1 = floor(ii1);
      di1 = ii1 - i1;
      j1 = floor(jj1);
      dj1 = jj1 - j1;

      do_uv =
          (i % 2 == 0) && (j % 2 == 0);  // TODO(bohan) only considering 420 now
      if (do_uv) {
        di0uv = di0 / 2 + 0.5 * ((i0 % 2 + 2) % 2);
        dj0uv = dj0 / 2 + 0.5 * ((j0 % 2 + 2) % 2);
        di1uv = di1 / 2 + 0.5 * ((i1 % 2 + 2) % 2);
        dj1uv = dj1 / 2 + 0.5 * ((j1 % 2 + 2) % 2);
      }

      dstpel_y = 0;
      dstpel_u = 0;
      dstpel_v = 0;
      if (use0) {
        dstpel_y += dstpel_y0[i * width + j] * (1 - pos);
        if (do_uv) {
          dstpel_u +=
              (double)get_sub_pel_uv(
                  src0u + (int)floor(ii0 / 2) * uvstride + (int)floor(jj0 / 2),
                  uvstride, di0uv, dj0uv) *
              (1 - pos);
          dstpel_v +=
              (double)get_sub_pel_uv(
                  src0v + (int)floor(ii0 / 2) * uvstride + (int)floor(jj0 / 2),
                  uvstride, di0uv, dj0uv) *
              (1 - pos);
        }
      }
      if (use1) {
        dstpel_y += dstpel_y1[i * width + j] * pos;
        if (do_uv) {
          dstpel_u +=
              (double)get_sub_pel_uv(
                  src1u + (int)floor(ii1 / 2) * uvstride + (int)floor(jj1 / 2),
                  uvstride, di1uv, dj1uv) *
              pos;
          dstpel_v +=
              (double)get_sub_pel_uv(
                  src1v + (int)floor(ii1 / 2) * uvstride + (int)floor(jj1 / 2),
                  uvstride, di1uv, dj1uv) *
              pos;
        }
      }
      dsty[i * stride + j] = dstpel_y + 0.5;
      if (do_uv) {
        dstu[i / 2 * uvstride + j / 2] = dstpel_u + 0.5;
        dstv[i / 2 * uvstride + j / 2] = dstpel_v + 0.5;
      }
    }
  }

  aom_free(refid);
  aom_free(refid_mode);
  aom_free(used0);
  aom_free(used1);
  aom_free(dstpel_y0);
  aom_free(dstpel_y1);
}

/*
 * Subpel filter for y pixels. Round the motion field to 1/8 precision.
 *
 * Input:
 * src: the source pixel at the integer location.
 * stride: source stride
 * di: subpel location in height
 * dj: subpel location in width
 *
 * Output:
 * interpolated pixel
 */
uint8_t get_sub_pel_y(uint8_t *src, int stride, double di, double dj) {
  // TODO(bohan) use locally defined filter arrays for now
  int yidx = di * 8 + 0.5;
  int xidx = dj * 8 + 0.5;
  yidx *= 2;
  xidx *= 2;
  if (yidx == 16) {
    yidx = 0;
    src += stride;
  }
  if (xidx == 16) {
    xidx = 0;
    src += 1;
  }
  assert(xidx <= 14 && xidx >= 0);
  assert(yidx <= 14 && yidx >= 0);
  int y[8];
  for (int i = -3; i < 5; i++) {
    y[i + 3] = 0;
    for (int j = -3; j < 5; j++) {
      y[i + 3] += src[i * stride + j] * optical_flow_warp_filter[xidx][j + 3];
    }
    y[i + 3] = (y[i + 3] + (1 << 6)) >> 7;
  }
  int x = 0;
  for (int i = 0; i < 8; i++) {
    x += y[i] * optical_flow_warp_filter[yidx][i];
  }
  x = (x + (1 << 6)) >> 7;
  if (x > 255)
    x = 255;
  else if (x < 0)
    x = 0;
  return (uint8_t)x;
}

/*
 * Subpel filter for u/v pixels. Round the motion field to 1/16 precision.
 *
 * Input:
 * src: the source pixel at the integer location.
 * stride: source stride
 * di: subpel location in height
 * dj: subpel location in width
 *
 * Output:
 * interpolated pixel
 */
uint8_t get_sub_pel_uv(uint8_t *src, int stride, double di, double dj) {
  // TODO(bohan) now only care for YUV 420
  int yidx = di * 16 + 0.5;
  int xidx = dj * 16 + 0.5;
  if (yidx == 16) {
    yidx = 0;
    src += stride;
  }
  if (xidx == 16) {
    xidx = 0;
    src += 1;
  }
  assert(xidx <= 15 && xidx >= 0);
  assert(yidx <= 15 && yidx >= 0);

  int y[8];
  for (int i = -3; i < 5; i++) {
    y[i + 3] = 0;
    for (int j = -3; j < 5; j++) {
      y[i + 3] += src[i * stride + j] * optical_flow_warp_filter[xidx][j + 3];
    }
    y[i + 3] = (y[i + 3] + (1 << 6)) >> 7;
  }
  int x = 0;
  for (int i = 0; i < 8; i++) {
    x += y[i] * optical_flow_warp_filter[yidx][i];
  }
  x = (x + (1 << 6)) >> 7;
  if (x > 255)
    x = 255;
  else if (x < 0)
    x = 0;
  return (uint8_t)x;
}

/*
 * Blend function which calls the real blend methods.
 * Kept as caller where we may make high level changes or pre-process
 */
void interp_optical_flow(YV12_BUFFER_CONFIG *ref0, YV12_BUFFER_CONFIG *ref1,
                         DB_MV *mf, YV12_BUFFER_CONFIG *dst, double dst_pos) {
  // blend here
  int mvstr = dst->y_width + 2 * AVG_MF_BORDER;
  DB_MV *mf_start = mf + AVG_MF_BORDER * mvstr + AVG_MF_BORDER;

  warp_optical_flow(ref0, ref1, mf_start, mvstr, dst, dst_pos,
                    OPFL_DIFF_SELECT);
  return;
}

/*
 * Use the initial motion vectors to create the initial motion field.
 * Currently, use the MVs of 4x4 to directly initialize the motion field
 * of the frame downscaled by 4.
 *
 * Input:
 * mv_left: motion vector points from ref0 to the current frame
 * mv_right: motion vector points from the current frame to ref1
 * width, height: frame width and height
 * mfstr: stride of the motion field buffer
 *
 * Output:
 * mf: pointer to the created motion field
 */
void create_motion_field(int_mv *mv_left, int_mv *mv_right, DB_MV *mf,
                         int width, int height, int mfstr) {
  // since the motion field is just used as initialization for now,
  // just simply use the summation of the two
  int stride = mfstr;
  DB_MV *mf_start = mf + AVG_MF_BORDER * stride + AVG_MF_BORDER;
  int idx;
  for (int h = 0; h < height / 4; h++) {
    for (int w = 0; w < width / 4; w++) {
      // mv_left, mv_right are based on 4x4 block
      idx = h * width / 4 + w;
      mf[h * mfstr + w].row =
          (double)(mv_left[idx].as_mv.row + mv_right[idx].as_mv.row) / 8.0;
      mf[h * mfstr + w].col =
          (double)(mv_left[idx].as_mv.col + mv_right[idx].as_mv.col) / 8.0;
    }
  }
  // Pad the motion field border
  pad_motion_field_border(mf_start, width / 4, height / 4, mfstr);
  return;
}

/*
 * Upscale the motion field by 2.
 * Currently simply copy the nearest mv
 *
 * Input:
 * src: the source motion field
 * srcw, srch, srcs: source width, height, stride
 * dsts: destination mf stride
 *
 * Output;
 * dst: pointer to the upscaled mf buffer
 */
void upscale_mv_by_2(DB_MV *src, int srcw, int srch, int srcs, DB_MV *dst,
                     int dsts) {
  for (int i = 0; i < srch; i++) {
    for (int j = 0; j < srcw; j++) {
      for (int y = 0; y < 2; y++) {
        for (int x = 0; x < 2; x++) {
          dst[(i * 2 + y) * dsts + j * 2 + x].row = src[i * srcs + j].row * 2;
          dst[(i * 2 + y) * dsts + j * 2 + x].col = src[i * srcs + j].col * 2;
        }
      }
    }
  }
  pad_motion_field_border(dst, srcw * 2, srch * 2, dsts);
}

/*
 * Pad the motion field border to prepare for median filter
 */
void pad_motion_field_border(DB_MV *mf_start, int width, int height,
                             int stride) {
  assert(stride == width + 2 * AVG_MF_BORDER);
  // upper
  for (int i = -AVG_MF_BORDER; i < 0; i++) {
    memcpy(mf_start + i * stride, mf_start, sizeof(DB_MV) * width);
  }
  // lower
  for (int i = height; i < height + AVG_MF_BORDER; i++) {
    memcpy(mf_start + i * stride, mf_start + (height - 1) * stride,
           sizeof(DB_MV) * width);
  }
  // left
  for (int i = -AVG_MF_BORDER; i < height + AVG_MF_BORDER; i++) {
    for (int j = -AVG_MF_BORDER; j < 0; j++) {
      mf_start[i * stride + j] = mf_start[i * stride];
    }
  }
  // right
  for (int i = -AVG_MF_BORDER; i < height + AVG_MF_BORDER; i++) {
    for (int j = width; j < width + AVG_MF_BORDER; j++) {
      mf_start[i * stride + j] = mf_start[i * stride + width - 1];
    }
  }
}

/*
 * Median filter the horizontal and vertical components of the motion field
 */
// TODO(bohan): can get rid of mv_r and mv_c, just use one temp buf
DB_MV median_2D_MV_5x5(DB_MV *center, int mvstr, double *mv_r, double *mv_c,
                       double *left, double *right) {
  DB_MV medMV;
  int c = 0;
  for (int i = -2; i < 3; i++) {
    for (int j = -2; j < 3; j++) {
      mv_r[c] = center[i * mvstr + j].row;
      mv_c[c] = center[i * mvstr + j].col;
      c++;
    }
  }
  medMV.row = iter_median_double(mv_r, left, right, 25, 12);
  medMV.col = iter_median_double(mv_c, left, right, 25, 12);
  return medMV;
}

/*
 * Median filter double arrays iteratively
 */
double iter_median_double(double *x, double *left, double *right, int length,
                          int mididx) {
  int pivot = length / 2;
  int ll = 0, rl = 0;
  for (int i = 0; i < length; i++) {
    if (i == pivot) continue;
    if (x[i] <= x[pivot]) {
      left[ll] = x[i];
      ll++;
    } else {
      right[rl] = x[i];
      rl++;
    }
  }
  if (mididx == ll)
    return x[pivot];
  else if (mididx < ll)
    return iter_median_double(left, x, right, ll, mididx);
  else
    return iter_median_double(right, left, x, rl, mididx - ll - 1);
}

/*
 * Do mode filter to the reference selections
 */
int ref_mode_filter_3x3(int *center, int stride, double dstpos) {
  int ref_id_count[3] = {0};
  for (int i = -1; i < 2; i++) {
    for (int j = -1; j < 2; j++) {
      assert(center[i * stride + j] >= 0);
      ref_id_count[center[i * stride + j]]++;
    }
  }
  if (ref_id_count[2] >= ref_id_count[1] &&
      ref_id_count[2] >= ref_id_count[0]) {
    return 2;
  } else if (ref_id_count[1] == ref_id_count[0]) {
    return ((dstpos <= 0.5) ? 0 : 1);
  } else {
    return ((ref_id_count[0] > ref_id_count[1]) ? 0 : 1);
  }
}

/*
 * Write YUV for debug purpose
 */
int write_image_opfl(const YV12_BUFFER_CONFIG *const ref_buf, char *file_name) {
  int h;
  FILE *f_ref = NULL;

  if (ref_buf == NULL) {
    printf("Frame data buffer is NULL.\n");
    return AOM_CODEC_MEM_ERROR;
  }
  if ((f_ref = fopen(file_name, "ab")) == NULL) {
    printf("Unable to open file %s to write.\n", file_name);
    return AOM_CODEC_MEM_ERROR;
  }
  // --- Y ---
  for (h = 0; h < ref_buf->y_height; ++h) {
    fwrite(&ref_buf->y_buffer[h * ref_buf->y_stride], 1, ref_buf->y_width,
           f_ref);
  }
  // --- U ---
  for (h = 0; h < (ref_buf->uv_height); ++h) {
    fwrite(&ref_buf->u_buffer[h * ref_buf->uv_stride], 1, ref_buf->uv_width,
           f_ref);
  }
  // --- V ---
  for (h = 0; h < (ref_buf->uv_height); ++h) {
    fwrite(&ref_buf->v_buffer[h * ref_buf->uv_stride], 1, ref_buf->uv_width,
           f_ref);
  }
  fclose(f_ref);
  return AOM_CODEC_OK;
}
#endif  // CONFIG_OPFL
