/*
 * Copyright (c) 2021, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <math.h>
#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <inttypes.h>
#include <malloc.h>
#include <string.h>
#include <png.h>
#include <time.h>
#include "av1/common/spherical_pred.h"
#include "av1/common/common.h"
#include "av1/common/filter.h"
#include "third_party/googletest/src/googletest/include/gtest/gtest.h"

namespace {

// The following code reads data from y4m file
// The original code is not from Google, so I will state what interface the rest
// of the test code uses.
typedef void *handle_t;

typedef struct {
  /* plane count
   * plane stride x4
   * plane data x4
   */
} image_t;

typedef struct {
  /* pts of picture */
  /* image_t, raw data */
} picture_t;

typedef struct {
  /* frame width, height, and frame count
   */
} config_t;

/* Most of this is from x264 */

/* YUV4MPEG2 raw 420 yuv file operation */
typedef struct {
  /* Basic information filled by the reading functions
  file pointer, frame width, height, fps, etc.*/
} y4m_input_t;

// Given a filename, read y4m file header, fill the structs
int open_file_y4m(char *filename, handle_t *handle, config_t *config) {
  return 0;
}

// Read the raw data of frame #framenum to pic
int read_frame_y4m(handle_t handle, picture_t *pic, int framenum) { return 0; }

// Close file
int close_file_y4m(handle_t handle) { return 0; }

#define DIFF_THRESHOLD 0.00001
constexpr double pi = 3.141592653589793238462643383279502884;

typedef struct {
  double x;
  double y;
} PlaneMV;

// Warp the plane coordinate back if out of border
// Flip x if y out of border
void warp_coordinate(int frame_width, int frame_height, int *x, int *y) {
  const int height = frame_height - 1;
  const int width = frame_width - 1;

  if (*y > height || *y < 0) {
    *y = *y < 0 ? -(1 + *y % height) : frame_height - *y % height;
    *x = *x + frame_width / 2;
  }

  if (*x > width || *x < 0) {
    *x = *x % frame_width;
    *x = *x > 0 ? *x : frame_width + *x;
  }
}

// Warp sub-pixel plane coordinate
void warp_coordinate_interp(int frame_width, int frame_height, double *x,
                            double *y) {
  const int height = frame_height - 1;
  const int width = frame_width - 1;

  if (*y > height || *y < 0) {
    *y = *y < 0 ? -fmod(*y, height) : frame_height - fmod(*y, height);
    *x = *x + frame_width / 2;
  }

  if (*x > width || *x < 0) {
    *x = fmod(*x, frame_width);
    *x = *x > 0 ? *x : frame_width + *x;
  }
}

// Some of the functions are static in the codebase. So I copied them.
static int get_sad_of_blocks(const uint8_t *cur_block,
                             const uint8_t *pred_block, int block_width,
                             int block_height, int cur_block_stride,
                             int pred_block_stride) {
  assert(block_width > 0 && block_height > 0 && cur_block_stride >= 0 &&
         pred_block_stride >= 0);
  int pos_curr;
  int pos_pred;
  int ret_sad = 0;

  for (int idx_y = 0; idx_y < block_height; idx_y++) {
    for (int idx_x = 0; idx_x < block_width; idx_x++) {
      pos_curr = idx_x + idx_y * cur_block_stride;
      pos_pred = idx_x + idx_y * pred_block_stride;

      ret_sad += abs((int)cur_block[pos_curr] - (int)pred_block[pos_pred]);
    }
  }

  return ret_sad;
}

/*!\brief Calculate the filter kernel index based on the subpixel coordination
 * \param[in]   subpel_pos  The subpixel coordination
 * \param[out]  pos_ref     The integer coordination to which the filter kernel
 *                          center should align to. If the subpixel coordination
 *                          is with in the range of (0, -0.0625) to the pixel on
 *                          its right, we choose that pixel. Otherwise, we
 *                          choose the pixel on the left of the subpixel.
 * \return                  The kernel index. It is doubled since we only use
 *                          #(0, 2, 4, ... 14) kernels
 */
int cal_kernel_idx(double subpel_pos, int *pos_ref) {
  double diff = subpel_pos - floor(subpel_pos);
  int idx = (int)round(diff / 0.125);

  if (idx >= SUBPEL_SHIFTS / 2) {
    idx = idx % (SUBPEL_SHIFTS / 2);
    *pos_ref = (int)ceil(subpel_pos);
  } else {
    *pos_ref = (int)floor(subpel_pos);
  }

  return idx * 2;
}

// Get the filtered sums of the rows int the 8x8 block centered at the
// input location
void get_interp_x_arr(int x, int y, const uint8_t *ref_frame,
                      int ref_frame_stride, int frame_width, int frame_height,
                      const InterpKernel *kernel, int filter_tap,
                      int x_kernel_idx, double *x_sum_arr) {
  int fo = filter_tap / 2 - 1;
  int cur_x;
  int cur_y;
  double coeff = 0;
  double sum = 0;

  for (int i = 0; i < 8; i++) {
    cur_y = y - fo + i;
    cur_y = AOMMAX(cur_y, 0);
    cur_y = AOMMIN(cur_y, frame_height - 1);

    sum = 0;

    for (int k = 0; k < filter_tap; k++) {
      coeff = kernel[x_kernel_idx][k] / 128.0;

      cur_x = x - fo + k;
      cur_x = cur_x % frame_width;
      cur_x = cur_x >= 0 ? cur_x : frame_width + cur_x;

      sum += ref_frame[cur_x + cur_y * ref_frame_stride] * coeff;
    }

    x_sum_arr[i] = sum;
  }  // for i
}

void av1_get_pred_plane(int block_x, int block_y, int block_width,
                        int block_height, double delta_x, double delta_y,
                        const uint8_t *cur_frame, const uint8_t *ref_frame,
                        int ref_frame_stride, int frame_width, int frame_height,
                        int block_stride, uint8_t *cur_block,
                        uint8_t *pred_block) {
  assert(ref_frame != NULL && frame_width > 0 && frame_height > 0 &&
         ref_frame_stride >= frame_width);
  assert(cur_block != NULL && pred_block != NULL && block_width > 0 &&
         block_height > 0 && block_stride >= block_width);

  int pos_block;  // Offset in pred_block buffer
  int pos_ref;    // Offset in ref_frame buffer
  int pos_cur;
  int x_current;  // X coordiante in current frame
  int y_current;  // Y coordiante in currrent frame
  int x_ref;      // X coordinate in reference frame
  int y_ref;      // Y coordiante in reference frame

  for (int idx_y = 0; idx_y < block_height; idx_y++) {
    for (int idx_x = 0; idx_x < block_width; idx_x++) {
      x_current = idx_x + block_x;
      y_current = idx_y + block_y;
      x_ref = x_current + delta_x;
      y_ref = y_current + delta_y;

      warp_coordinate(frame_width, frame_height, &x_current, &y_current);
      warp_coordinate(frame_width, frame_height, &x_ref, &y_ref);

      pos_block = idx_x + idx_y * block_stride;
      pos_cur = x_current + y_current * ref_frame_stride;
      pos_ref = x_ref + y_ref * ref_frame_stride;

      cur_block[pos_block] = cur_frame[pos_cur];
      pred_block[pos_block] = ref_frame[pos_ref];
    }
  }
}

void get_interp_x_arr_plane(int x, int y, const uint8_t *ref_frame,
                            int ref_frame_stride, int frame_width,
                            int frame_height, const InterpKernel *kernel,
                            int filter_tap, int x_kernel_idx,
                            double *x_sum_arr) {
  int fo = filter_tap / 2 - 1;
  int cur_x;
  int cur_y;
  double coeff = 0;
  double sum = 0;

  for (int i = 0; i < 8; i++) {
    cur_y = y - fo + i;
    cur_y = AOMMAX(cur_y, 0);
    cur_y = AOMMIN(cur_y, frame_height - 1);

    sum = 0;

    for (int k = 0; k < filter_tap; k++) {
      coeff = kernel[x_kernel_idx][k] / 128.0;

      cur_x = x - fo + k;
      cur_x = cur_x % frame_width;
      cur_x = cur_x >= 0 ? cur_x : frame_width + cur_x;

      sum += ref_frame[cur_x + cur_y * ref_frame_stride] * coeff;
    }

    x_sum_arr[i] = sum;
  }  // for i
}

uint8_t interp_pixel_plane(const uint8_t *ref_frame, int ref_frame_stride,
                           int frame_width, int frame_height, double subpel_x,
                           double subpel_y, const InterpKernel *kernel,
                           int filter_tap) {
  double x_sum_arr[8];
  double sum = 0;
  double coeff = 0;
  int x_kernel_idx;
  int y_kernel_idx;
  int x_ref;
  int y_ref;
  int pixel;

  x_kernel_idx = cal_kernel_idx(subpel_x, &x_ref);
  y_kernel_idx = cal_kernel_idx(subpel_y, &y_ref);

  get_interp_x_arr_plane(x_ref, y_ref, ref_frame, ref_frame_stride, frame_width,
                         frame_height, kernel, filter_tap, x_kernel_idx,
                         x_sum_arr);

  // Perform filtering on the column
  sum = 0;
  for (int k = 0; k < filter_tap; k++) {
    coeff = kernel[y_kernel_idx][k] / 128.0;
    sum += x_sum_arr[k] * coeff;
  }

  pixel = (int)round(sum);
  pixel = clamp(pixel, 0, 255);
  return (uint8_t)pixel;
}

void av1_get_pred_plane_interp_bilinear(
    int block_x, int block_y, int block_width, int block_height, double delta_x,
    double delta_y, const uint8_t *cur_frame, const uint8_t *ref_frame,
    int ref_frame_stride, int frame_width, int frame_height, int block_stride,
    uint8_t *cur_block, uint8_t *pred_block) {
  assert(ref_frame != NULL && frame_width > 0 && frame_height > 0 &&
         ref_frame_stride >= frame_width);
  assert(cur_block != NULL && pred_block != NULL && block_width > 0 &&
         block_height > 0 && block_stride >= block_width);

  int pos_block;  // Offset in pred_block buffer
  int pos_ref;    // Offset in ref_frame buffer
  int pos_cur;
  int x_current;  // X coordiante in current frame
  int y_current;  // Y coordiante in currrent frame
  double x_ref;   // X coordinate in reference frame
  double y_ref;   // Y coordiante in reference frame
  const int filter_tap = 8;

  for (int idx_y = 0; idx_y < block_height; idx_y++) {
    for (int idx_x = 0; idx_x < block_width; idx_x++) {
      x_current = idx_x + block_x;
      y_current = idx_y + block_y;
      x_ref = x_current + delta_x;
      y_ref = y_current + delta_y;

      warp_coordinate(frame_width, frame_height, &x_current, &y_current);
      warp_coordinate_interp(frame_width, frame_height, &x_ref, &y_ref);

      pos_block = idx_x + idx_y * block_stride;
      pos_cur = x_current + y_current * ref_frame_stride;
      // pos_ref = (int)floor(x_ref) + y_ref * ref_frame_stride;

      cur_block[pos_block] = cur_frame[pos_cur];
      pred_block[pos_block] = interp_pixel_plane(
          ref_frame, ref_frame_stride, frame_width, frame_height, x_ref, y_ref,
          av1_bilinear_filters, filter_tap);
    }
  }
}

void av1_get_pred_plane_interp_sub_pel_8(
    int block_x, int block_y, int block_width, int block_height, double delta_x,
    double delta_y, const uint8_t *cur_frame, const uint8_t *ref_frame,
    int ref_frame_stride, int frame_width, int frame_height, int block_stride,
    uint8_t *cur_block, uint8_t *pred_block) {
  assert(ref_frame != NULL && frame_width > 0 && frame_height > 0 &&
         ref_frame_stride >= frame_width);
  assert(cur_block != NULL && pred_block != NULL && block_width > 0 &&
         block_height > 0 && block_stride >= block_width);

  int pos_block;  // Offset in pred_block buffer
  int pos_ref;    // Offset in ref_frame buffer
  int pos_cur;
  int x_current;  // X coordiante in current frame
  int y_current;  // Y coordiante in currrent frame
  double x_ref;   // X coordinate in reference frame
  double y_ref;   // Y coordiante in reference frame
  const int filter_tap = 8;

  for (int idx_y = 0; idx_y < block_height; idx_y++) {
    for (int idx_x = 0; idx_x < block_width; idx_x++) {
      x_current = idx_x + block_x;
      y_current = idx_y + block_y;
      x_ref = x_current + delta_x;
      y_ref = y_current + delta_y;

      warp_coordinate(frame_width, frame_height, &x_current, &y_current);
      warp_coordinate_interp(frame_width, frame_height, &x_ref, &y_ref);

      pos_block = idx_x + idx_y * block_stride;
      pos_cur = x_current + y_current * ref_frame_stride;
      // pos_ref = (int)floor(x_ref) + y_ref * ref_frame_stride;

      cur_block[pos_block] = cur_frame[pos_cur];
      pred_block[pos_block] = interp_pixel_plane(
          ref_frame, ref_frame_stride, frame_width, frame_height, x_ref, y_ref,
          av1_sub_pel_filters_8, filter_tap);
    }
  }
}

void av1_get_pred_plane_interp_sub_pel_8sharp(
    int block_x, int block_y, int block_width, int block_height, double delta_x,
    double delta_y, const uint8_t *cur_frame, const uint8_t *ref_frame,
    int ref_frame_stride, int frame_width, int frame_height, int block_stride,
    uint8_t *cur_block, uint8_t *pred_block) {
  assert(ref_frame != NULL && frame_width > 0 && frame_height > 0 &&
         ref_frame_stride >= frame_width);
  assert(cur_block != NULL && pred_block != NULL && block_width > 0 &&
         block_height > 0 && block_stride >= block_width);

  int pos_block;  // Offset in pred_block buffer
  int pos_ref;    // Offset in ref_frame buffer
  int pos_cur;
  int x_current;  // X coordiante in current frame
  int y_current;  // Y coordiante in currrent frame
  double x_ref;   // X coordinate in reference frame
  double y_ref;   // Y coordiante in reference frame
  const int filter_tap = 8;

  for (int idx_y = 0; idx_y < block_height; idx_y++) {
    for (int idx_x = 0; idx_x < block_width; idx_x++) {
      x_current = idx_x + block_x;
      y_current = idx_y + block_y;
      x_ref = x_current + delta_x;
      y_ref = y_current + delta_y;

      warp_coordinate(frame_width, frame_height, &x_current, &y_current);
      warp_coordinate_interp(frame_width, frame_height, &x_ref, &y_ref);

      pos_block = idx_x + idx_y * block_stride;
      pos_cur = x_current + y_current * ref_frame_stride;
      // pos_ref = (int)floor(x_ref) + y_ref * ref_frame_stride;

      cur_block[pos_block] = cur_frame[pos_cur];
      pred_block[pos_block] = interp_pixel_plane(
          ref_frame, ref_frame_stride, frame_width, frame_height, x_ref, y_ref,
          av1_sub_pel_filters_8sharp, filter_tap);
    }
  }
}

void av1_get_pred_plane_interp_sub_pel_8smooth(
    int block_x, int block_y, int block_width, int block_height, double delta_x,
    double delta_y, const uint8_t *cur_frame, const uint8_t *ref_frame,
    int ref_frame_stride, int frame_width, int frame_height, int block_stride,
    uint8_t *cur_block, uint8_t *pred_block) {
  assert(ref_frame != NULL && frame_width > 0 && frame_height > 0 &&
         ref_frame_stride >= frame_width);
  assert(cur_block != NULL && pred_block != NULL && block_width > 0 &&
         block_height > 0 && block_stride >= block_width);

  int pos_block;  // Offset in pred_block buffer
  int pos_ref;    // Offset in ref_frame buffer
  int pos_cur;
  int x_current;  // X coordiante in current frame
  int y_current;  // Y coordiante in currrent frame
  double x_ref;   // X coordinate in reference frame
  double y_ref;   // Y coordiante in reference frame
  const int filter_tap = 8;

  for (int idx_y = 0; idx_y < block_height; idx_y++) {
    for (int idx_x = 0; idx_x < block_width; idx_x++) {
      x_current = idx_x + block_x;
      y_current = idx_y + block_y;
      x_ref = x_current + delta_x;
      y_ref = y_current + delta_y;

      warp_coordinate(frame_width, frame_height, &x_current, &y_current);
      warp_coordinate_interp(frame_width, frame_height, &x_ref, &y_ref);

      pos_block = idx_x + idx_y * block_stride;
      pos_cur = x_current + y_current * ref_frame_stride;
      // pos_ref = (int)floor(x_ref) + y_ref * ref_frame_stride;

      cur_block[pos_block] = cur_frame[pos_cur];
      pred_block[pos_block] = interp_pixel_plane(
          ref_frame, ref_frame_stride, frame_width, frame_height, x_ref, y_ref,
          av1_sub_pel_filters_8smooth, filter_tap);
    }
  }
}

// Given two index maps, calculate SAD on plane
static int sad_of_blocks_from_idx(const uint8_t *cur_frame,
                                  const uint8_t *ref_frame, int *cur_block_idx,
                                  int *pred_block_idx, int block_width,
                                  int block_height, int cur_block_stride,
                                  int pred_block_stride) {
  assert(block_width > 0 && block_height > 0 && cur_block_stride >= 0 &&
         pred_block_stride >= 0);
  int pos_curr;
  int pos_pred;
  int ret_sad = 0;

  for (int idx_y = 0; idx_y < block_height; idx_y++) {
    for (int idx_x = 0; idx_x < block_width; idx_x++) {
      pos_curr = idx_x + idx_y * cur_block_stride;
      pos_pred = idx_x + idx_y * pred_block_stride;

      ret_sad += abs(cur_frame[cur_block_idx[pos_curr]] -
                     ref_frame[pred_block_idx[pos_pred]]);
    }
  }

  return ret_sad;
}

int av1_motion_search_brute_force_plane(
    int block_x, int block_y, int block_width, int block_height,
    const uint8_t *cur_frame, const uint8_t *ref_frame, int frame_stride,
    int frame_width, int frame_height, int search_range, PlaneMV *best_mv) {
  assert(cur_frame != NULL && ref_frame != NULL && best_mv != NULL);
  assert(block_width > 0 && block_height > 0 && block_width <= 128 &&
         block_height <= 128 && block_x >= 0 && block_y >= 0 &&
         frame_width > 0 && frame_height > 0);
  assert(search_range > 0);

  const double search_step = 1;

  int temp_sad;
  int best_sad;
  int delta_x;
  int delta_y;

  uint8_t cur_block[128 * 128];
  uint8_t pred_block[128 * 128];
  const int pred_block_stride = 128;

  av1_get_pred_plane(block_x, block_y, block_width, block_height, 0, 0,
                     cur_frame, ref_frame, frame_stride, frame_width,
                     frame_height, pred_block_stride, cur_block, pred_block);
  best_sad = get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               pred_block_stride, pred_block_stride);
  best_mv->x = 0;
  best_mv->y = 0;

  for (int i = -search_range; i <= search_range; i++) {
    delta_y = i * search_step;

    for (int j = -search_range; j <= search_range; j++) {
      delta_x = j * search_step;

      av1_get_pred_plane(block_x, block_y, block_width, block_height, delta_x,
                         delta_y, cur_frame, ref_frame, frame_stride,
                         frame_width, frame_height, pred_block_stride,
                         cur_block, pred_block);

      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            pred_block_stride, pred_block_stride);

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv->x = delta_x;
        best_mv->y = delta_y;
      }
    }  // for j
  }    // for i

  return best_sad;
}

int av1_motion_search_brute_force_plane_interp_bilinear(
    int block_x, int block_y, int block_width, int block_height,
    const uint8_t *cur_frame, const uint8_t *ref_frame, int frame_stride,
    int frame_width, int frame_height, int search_range, PlaneMV *start_mv,
    PlaneMV *best_mv) {
  assert(cur_frame != NULL && ref_frame != NULL && best_mv != NULL);
  assert(block_width > 0 && block_height > 0 && block_width <= 128 &&
         block_height <= 128 && block_x >= 0 && block_y >= 0 &&
         frame_width > 0 && frame_height > 0);
  assert(search_range > 0);

  const double search_step = 0.125;

  int temp_sad;
  int best_sad;
  double delta_x;
  double delta_y;

  uint8_t cur_block[128 * 128];
  uint8_t pred_block[128 * 128];
  const int pred_block_stride = 128;

  av1_get_pred_plane_interp_bilinear(
      block_x, block_y, block_width, block_height, start_mv->x, start_mv->y,
      cur_frame, ref_frame, frame_stride, frame_width, frame_height,
      pred_block_stride, cur_block, pred_block);
  best_sad = get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               pred_block_stride, pred_block_stride);
  best_mv->x = 0;
  best_mv->y = 0;

  for (int i = -8; i <= 8; i++) {
    delta_y = start_mv->y + i * search_step;

    for (int j = -8; j <= 8; j++) {
      delta_x = start_mv->x + j * search_step;

      av1_get_pred_plane_interp_bilinear(
          block_x, block_y, block_width, block_height, delta_x, delta_y,
          cur_frame, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, cur_block, pred_block);

      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            pred_block_stride, pred_block_stride);

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv->x = delta_x;
        best_mv->y = delta_y;
      }
    }  // for j
  }    // for i

  return best_sad;
}

int av1_motion_search_brute_force_plane_interp_sub_pel_8(
    int block_x, int block_y, int block_width, int block_height,
    const uint8_t *cur_frame, const uint8_t *ref_frame, int frame_stride,
    int frame_width, int frame_height, int search_range, PlaneMV *start_mv,
    PlaneMV *best_mv) {
  assert(cur_frame != NULL && ref_frame != NULL && best_mv != NULL);
  assert(block_width > 0 && block_height > 0 && block_width <= 128 &&
         block_height <= 128 && block_x >= 0 && block_y >= 0 &&
         frame_width > 0 && frame_height > 0);
  assert(search_range > 0);

  const double search_step = 0.125;

  int temp_sad;
  int best_sad;
  double delta_x;
  double delta_y;

  uint8_t cur_block[128 * 128];
  uint8_t pred_block[128 * 128];
  const int pred_block_stride = 128;

  av1_get_pred_plane_interp_bilinear(
      block_x, block_y, block_width, block_height, start_mv->x, start_mv->y,
      cur_frame, ref_frame, frame_stride, frame_width, frame_height,
      pred_block_stride, cur_block, pred_block);
  best_sad = get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               pred_block_stride, pred_block_stride);
  best_mv->x = 0;
  best_mv->y = 0;

  for (int i = -8; i <= 8; i++) {
    delta_y = start_mv->y + i * search_step;

    for (int j = -8; j <= 8; j++) {
      delta_x = start_mv->x + j * search_step;

      av1_get_pred_plane_interp_sub_pel_8(
          block_x, block_y, block_width, block_height, delta_x, delta_y,
          cur_frame, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, cur_block, pred_block);

      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            pred_block_stride, pred_block_stride);

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv->x = delta_x;
        best_mv->y = delta_y;
      }
    }  // for j
  }    // for i

  return best_sad;
}

int av1_motion_search_brute_force_plane_interp_sub_pel_8smooth(
    int block_x, int block_y, int block_width, int block_height,
    const uint8_t *cur_frame, const uint8_t *ref_frame, int frame_stride,
    int frame_width, int frame_height, int search_range, PlaneMV *start_mv,
    PlaneMV *best_mv) {
  assert(cur_frame != NULL && ref_frame != NULL && best_mv != NULL);
  assert(block_width > 0 && block_height > 0 && block_width <= 128 &&
         block_height <= 128 && block_x >= 0 && block_y >= 0 &&
         frame_width > 0 && frame_height > 0);
  assert(search_range > 0);

  const double search_step = 0.125;

  int temp_sad;
  int best_sad;
  double delta_x;
  double delta_y;

  uint8_t cur_block[128 * 128];
  uint8_t pred_block[128 * 128];
  const int pred_block_stride = 128;

  av1_get_pred_plane_interp_bilinear(
      block_x, block_y, block_width, block_height, start_mv->x, start_mv->y,
      cur_frame, ref_frame, frame_stride, frame_width, frame_height,
      pred_block_stride, cur_block, pred_block);
  best_sad = get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               pred_block_stride, pred_block_stride);
  best_mv->x = 0;
  best_mv->y = 0;

  for (int i = -8; i <= 8; i++) {
    delta_y = start_mv->y + i * search_step;

    for (int j = -8; j <= 8; j++) {
      delta_x = start_mv->x + j * search_step;

      av1_get_pred_plane_interp_sub_pel_8smooth(
          block_x, block_y, block_width, block_height, delta_x, delta_y,
          cur_frame, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, cur_block, pred_block);

      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            pred_block_stride, pred_block_stride);

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv->x = delta_x;
        best_mv->y = delta_y;
      }
    }  // for j
  }    // for i

  return best_sad;
}

int av1_motion_search_brute_force_plane_interp_sub_pel_8sharp(
    int block_x, int block_y, int block_width, int block_height,
    const uint8_t *cur_frame, const uint8_t *ref_frame, int frame_stride,
    int frame_width, int frame_height, int search_range, PlaneMV *start_mv,
    PlaneMV *best_mv) {
  assert(cur_frame != NULL && ref_frame != NULL && best_mv != NULL);
  assert(block_width > 0 && block_height > 0 && block_width <= 128 &&
         block_height <= 128 && block_x >= 0 && block_y >= 0 &&
         frame_width > 0 && frame_height > 0);
  assert(search_range > 0);

  const double search_step = 0.125;

  int temp_sad;
  int best_sad;
  double delta_x;
  double delta_y;

  uint8_t cur_block[128 * 128];
  uint8_t pred_block[128 * 128];
  const int pred_block_stride = 128;

  av1_get_pred_plane_interp_bilinear(
      block_x, block_y, block_width, block_height, start_mv->x, start_mv->y,
      cur_frame, ref_frame, frame_stride, frame_width, frame_height,
      pred_block_stride, cur_block, pred_block);
  best_sad = get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               pred_block_stride, pred_block_stride);
  best_mv->x = 0;
  best_mv->y = 0;

  for (int i = -8; i <= 8; i++) {
    delta_y = start_mv->y + i * search_step;

    for (int j = -8; j <= 8; j++) {
      delta_x = start_mv->x + j * search_step;

      av1_get_pred_plane_interp_sub_pel_8sharp(
          block_x, block_y, block_width, block_height, delta_x, delta_y,
          cur_frame, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, cur_block, pred_block);

      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            pred_block_stride, pred_block_stride);

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv->x = delta_x;
        best_mv->y = delta_y;
      }
    }  // for j
  }    // for i

  return best_sad;
}

int av1_motion_search_brute_force_plane_interp_filter_search(
    int block_x, int block_y, int block_width, int block_height,
    const uint8_t *cur_frame, const uint8_t *ref_frame, int frame_stride,
    int frame_width, int frame_height, int search_range, PlaneMV *start_mv,
    PlaneMV *best_mv) {
  assert(cur_frame != NULL && ref_frame != NULL && best_mv != NULL);
  assert(block_width > 0 && block_height > 0 && block_width <= 128 &&
         block_height <= 128 && block_x >= 0 && block_y >= 0 &&
         frame_width > 0 && frame_height > 0);
  assert(search_range > 0);

  const double search_step = 0.125;

  int temp_sad;
  int best_sad;
  double delta_x;
  double delta_y;

  uint8_t cur_block[128 * 128];
  uint8_t pred_block[128 * 128];
  const int pred_block_stride = 128;

  av1_get_pred_plane_interp_bilinear(
      block_x, block_y, block_width, block_height, start_mv->x, start_mv->y,
      cur_frame, ref_frame, frame_stride, frame_width, frame_height,
      pred_block_stride, cur_block, pred_block);
  best_sad = get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               pred_block_stride, pred_block_stride);
  best_mv->x = 0;
  best_mv->y = 0;

  for (int i = -8; i <= 8; i++) {
    delta_y = start_mv->y + i * search_step;

    for (int j = -8; j <= 8; j++) {
      delta_x = start_mv->x + j * search_step;

      av1_get_pred_plane_interp_bilinear(
          block_x, block_y, block_width, block_height, delta_x, delta_y,
          cur_frame, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, cur_block, pred_block);
      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            pred_block_stride, pred_block_stride);

      av1_get_pred_plane_interp_sub_pel_8(
          block_x, block_y, block_width, block_height, delta_x, delta_y,
          cur_frame, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, cur_block, pred_block);
      temp_sad =
          AOMMIN(temp_sad, get_sad_of_blocks(cur_block, pred_block, block_width,
                                             block_height, pred_block_stride,
                                             pred_block_stride));

      av1_get_pred_plane_interp_sub_pel_8sharp(
          block_x, block_y, block_width, block_height, delta_x, delta_y,
          cur_frame, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, cur_block, pred_block);
      temp_sad =
          AOMMIN(temp_sad, get_sad_of_blocks(cur_block, pred_block, block_width,
                                             block_height, pred_block_stride,
                                             pred_block_stride));

      av1_get_pred_plane_interp_sub_pel_8smooth(
          block_x, block_y, block_width, block_height, delta_x, delta_y,
          cur_frame, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, cur_block, pred_block);
      temp_sad =
          AOMMIN(temp_sad, get_sad_of_blocks(cur_block, pred_block, block_width,
                                             block_height, pred_block_stride,
                                             pred_block_stride));

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv->x = delta_x;
        best_mv->y = delta_y;
      }
    }  // for j
  }    // for i

  return best_sad;
}

/* Update Large Diamond Search Pattern shperical motion vectors based on the
 * center
 *                    ldsp_mv[1]
 *                    /        \
 *          ldsp_mv[8]          ldsp_mv[2]
 *                 /              \
 *       ldsp_mv[7]   ldsp_mv[0]   ldsp_mv[3]
 *                 \              /
 *         ldsp_mv[6]            ldsp_mv[4]
 *                   \          /
 *                    ldsp_mv[5]
 */
static void update_plane_mv_ldsp(PlaneMV ldsp_mv[9], double search_step) {
  ldsp_mv[1].y = ldsp_mv[0].y - 2 * search_step;
  ldsp_mv[1].x = ldsp_mv[0].x;

  ldsp_mv[2].y = ldsp_mv[0].y - search_step;
  ldsp_mv[2].x = ldsp_mv[0].x + search_step;

  ldsp_mv[3].y = ldsp_mv[0].y;
  ldsp_mv[3].x = ldsp_mv[0].x + 2 * search_step;

  ldsp_mv[4].y = ldsp_mv[0].y + search_step;
  ldsp_mv[4].x = ldsp_mv[0].x + search_step;

  ldsp_mv[5].y = ldsp_mv[0].y + 2 * search_step;
  ldsp_mv[5].x = ldsp_mv[0].x;

  ldsp_mv[6].y = ldsp_mv[0].y + search_step;
  ldsp_mv[6].x = ldsp_mv[0].x - search_step;

  ldsp_mv[7].y = ldsp_mv[0].y;
  ldsp_mv[7].x = ldsp_mv[0].x - 2 * search_step;

  ldsp_mv[8].y = ldsp_mv[0].y - search_step;
  ldsp_mv[8].x = ldsp_mv[0].x - search_step;
}

/* Small Diamond Search Pattern on shpere
 *                     sdsp_mv[1]
 *                  /              \
 *         sdsp_mv[4]  sdsp_mv[0]  sdsp_mv[2]
 *                  \              /
 *                     sdsp_mv[3]
 */
static void update_plane_mv_sdsp(PlaneMV sdsp_mv[5], double search_step) {
  sdsp_mv[1].y = sdsp_mv[0].y - search_step;
  sdsp_mv[1].x = sdsp_mv[0].x;

  sdsp_mv[2].y = sdsp_mv[0].y;
  sdsp_mv[2].x = sdsp_mv[0].x + search_step;

  sdsp_mv[3].y = sdsp_mv[0].y + search_step;
  sdsp_mv[3].x = sdsp_mv[0].x;

  sdsp_mv[4].y = sdsp_mv[0].y;
  sdsp_mv[4].x = sdsp_mv[0].x - search_step;
}

int av1_motion_search_diamond_plane(int block_x, int block_y, int block_width,
                                    int block_height, const uint8_t *cur_frame,
                                    const uint8_t *ref_frame, int frame_stride,
                                    int frame_width, int frame_height,
                                    int search_range, PlaneMV *best_mv) {
  assert(cur_frame != NULL && ref_frame != NULL && best_mv != NULL);
  assert(block_width > 0 && block_height > 0 && block_width <= 128 &&
         block_height <= 128 && block_x >= 0 && block_y >= 0 &&
         frame_width > 0 && frame_height > 0);
  assert(search_range > 0);

  // Large Diamond Search Pattern on shpere
  PlaneMV ldsp_mv[9];
  // Small Diamond Search Pattern on shpere
  PlaneMV sdsp_mv[5];

  int search_step = block_height / 5;

  uint8_t cur_block[128 * 128];
  uint8_t pred_block[128 * 128];
  const int pred_block_stride = 128;

  int max_range_y = block_y + search_range;
  int min_range_y = block_y - search_range;
  int max_range_x = block_x + search_range;
  int min_range_x = block_x - search_range;

  int temp_sad;
  int best_sad;
  av1_get_pred_plane(block_x, block_y, block_width, block_height, 0, 0,
                     cur_frame, ref_frame, frame_stride, frame_width,
                     frame_height, pred_block_stride, cur_block, pred_block);
  best_sad = get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               pred_block_stride, pred_block_stride);

  int best_mv_idx = 0;
  ldsp_mv[0].y = 0;
  ldsp_mv[0].x = 0;

  do {
    update_plane_mv_ldsp(ldsp_mv, search_step);

    for (int i = 0; i < 9; i++) {
      av1_get_pred_plane(block_x, block_y, block_width, block_height,
                         ldsp_mv[i].x, ldsp_mv[i].y, cur_frame, ref_frame,
                         frame_stride, frame_width, frame_height,
                         pred_block_stride, cur_block, pred_block);

      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            pred_block_stride, pred_block_stride);

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv_idx = i;
      }
    }  // for

    if (best_mv_idx == 0) {
      if (search_step > 1) {
        search_step *= 0.5;
        continue;
      } else {
        break;
      }
    } else {
      ldsp_mv[0].y = ldsp_mv[best_mv_idx].y;
      ldsp_mv[0].x = ldsp_mv[best_mv_idx].x;
      best_mv_idx = 0;
    }
  } while (block_y + ldsp_mv[5].y <= max_range_y &&
           block_y + ldsp_mv[1].y >= min_range_y &&
           block_x + ldsp_mv[3].x <= max_range_x &&
           block_x + ldsp_mv[7].x >= min_range_x);

  sdsp_mv[0].y = ldsp_mv[best_mv_idx].y;
  sdsp_mv[0].x = ldsp_mv[best_mv_idx].x;
  best_mv_idx = 0;
  search_step = 1;

  update_plane_mv_sdsp(sdsp_mv, search_step);
  for (int i = 0; i < 5; i++) {
    av1_get_pred_plane(block_x, block_y, block_width, block_height,
                       sdsp_mv[i].x, sdsp_mv[i].y, cur_frame, ref_frame,
                       frame_stride, frame_width, frame_height,
                       pred_block_stride, cur_block, pred_block);

    temp_sad =
        get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                          pred_block_stride, pred_block_stride);

    if (temp_sad < best_sad) {
      best_sad = temp_sad;
      best_mv_idx = i;
    }
  }  // for

  best_mv->y = sdsp_mv[best_mv_idx].y;
  best_mv->x = sdsp_mv[best_mv_idx].x;

  return best_sad;
}

int av1_motion_search_diamond_plane_interp_bilinear(
    int block_x, int block_y, int block_width, int block_height,
    const uint8_t *cur_frame, const uint8_t *ref_frame, int frame_stride,
    int frame_width, int frame_height, int search_range, PlaneMV *best_mv) {
  assert(cur_frame != NULL && ref_frame != NULL && best_mv != NULL);
  assert(block_width > 0 && block_height > 0 && block_width <= 128 &&
         block_height <= 128 && block_x >= 0 && block_y >= 0 &&
         frame_width > 0 && frame_height > 0);
  assert(search_range > 0);

  // Large Diamond Search Pattern on shpere
  PlaneMV ldsp_mv[9];
  // Small Diamond Search Pattern on shpere
  PlaneMV sdsp_mv[5];

  int search_step = block_height / 5;

  uint8_t cur_block[128 * 128];
  uint8_t pred_block[128 * 128];
  const int pred_block_stride = 128;

  int max_range_y = block_y + search_range;
  int min_range_y = block_y - search_range;
  int max_range_x = block_x + search_range;
  int min_range_x = block_x - search_range;

  int temp_sad;
  int best_sad;
  av1_get_pred_plane_interp_bilinear(block_x, block_y, block_width,
                                     block_height, 0, 0, cur_frame, ref_frame,
                                     frame_stride, frame_width, frame_height,
                                     pred_block_stride, cur_block, pred_block);
  best_sad = get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               pred_block_stride, pred_block_stride);

  int best_mv_idx = 0;
  ldsp_mv[0].y = 0;
  ldsp_mv[0].x = 0;

  const double min_search_step = 0.125;

  do {
    update_plane_mv_ldsp(ldsp_mv, search_step);

    for (int i = 0; i < 9; i++) {
      av1_get_pred_plane_interp_bilinear(
          block_x, block_y, block_width, block_height, ldsp_mv[i].x,
          ldsp_mv[i].y, cur_frame, ref_frame, frame_stride, frame_width,
          frame_height, pred_block_stride, cur_block, pred_block);

      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            pred_block_stride, pred_block_stride);

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv_idx = i;
      }
    }  // for

    if (best_mv_idx == 0) {
      if (search_step > min_search_step) {
        search_step *= 0.5;
        continue;
      } else {
        break;
      }
    } else {
      ldsp_mv[0].y = ldsp_mv[best_mv_idx].y;
      ldsp_mv[0].x = ldsp_mv[best_mv_idx].x;
      best_mv_idx = 0;
    }
  } while (block_y + ldsp_mv[5].y <= max_range_y &&
           block_y + ldsp_mv[1].y >= min_range_y &&
           block_x + ldsp_mv[3].x <= max_range_x &&
           block_x + ldsp_mv[7].x >= min_range_x);

  sdsp_mv[0].y = ldsp_mv[best_mv_idx].y;
  sdsp_mv[0].x = ldsp_mv[best_mv_idx].x;
  best_mv_idx = 0;

  update_plane_mv_sdsp(sdsp_mv, search_step);
  for (int i = 0; i < 5; i++) {
    av1_get_pred_plane_interp_bilinear(
        block_x, block_y, block_width, block_height, sdsp_mv[i].x, sdsp_mv[i].y,
        cur_frame, ref_frame, frame_stride, frame_width, frame_height,
        pred_block_stride, cur_block, pred_block);

    temp_sad =
        get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                          pred_block_stride, pred_block_stride);

    if (temp_sad < best_sad) {
      best_sad = temp_sad;
      best_mv_idx = i;
    }
  }  // for

  best_mv->y = sdsp_mv[best_mv_idx].y;
  best_mv->x = sdsp_mv[best_mv_idx].x;

  return best_sad;
}

int av1_motion_search_diamond_plane_interp_sub_pel_8(
    int block_x, int block_y, int block_width, int block_height,
    const uint8_t *cur_frame, const uint8_t *ref_frame, int frame_stride,
    int frame_width, int frame_height, int search_range, PlaneMV *best_mv) {
  assert(cur_frame != NULL && ref_frame != NULL && best_mv != NULL);
  assert(block_width > 0 && block_height > 0 && block_width <= 128 &&
         block_height <= 128 && block_x >= 0 && block_y >= 0 &&
         frame_width > 0 && frame_height > 0);
  assert(search_range > 0);

  // Large Diamond Search Pattern on shpere
  PlaneMV ldsp_mv[9];
  // Small Diamond Search Pattern on shpere
  PlaneMV sdsp_mv[5];

  int search_step = block_height / 5;

  uint8_t cur_block[128 * 128];
  uint8_t pred_block[128 * 128];
  const int pred_block_stride = 128;

  int max_range_y = block_y + search_range;
  int min_range_y = block_y - search_range;
  int max_range_x = block_x + search_range;
  int min_range_x = block_x - search_range;

  int temp_sad;
  int best_sad;
  av1_get_pred_plane_interp_sub_pel_8(block_x, block_y, block_width,
                                      block_height, 0, 0, cur_frame, ref_frame,
                                      frame_stride, frame_width, frame_height,
                                      pred_block_stride, cur_block, pred_block);
  best_sad = get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               pred_block_stride, pred_block_stride);

  int best_mv_idx = 0;
  ldsp_mv[0].y = 0;
  ldsp_mv[0].x = 0;

  const double min_search_step = 0.125;

  do {
    update_plane_mv_ldsp(ldsp_mv, search_step);

    for (int i = 0; i < 9; i++) {
      av1_get_pred_plane_interp_sub_pel_8(
          block_x, block_y, block_width, block_height, ldsp_mv[i].x,
          ldsp_mv[i].y, cur_frame, ref_frame, frame_stride, frame_width,
          frame_height, pred_block_stride, cur_block, pred_block);

      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            pred_block_stride, pred_block_stride);

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv_idx = i;
      }
    }  // for

    if (best_mv_idx == 0) {
      if (search_step > min_search_step) {
        search_step *= 0.5;
        continue;
      } else {
        break;
      }
    } else {
      ldsp_mv[0].y = ldsp_mv[best_mv_idx].y;
      ldsp_mv[0].x = ldsp_mv[best_mv_idx].x;
      best_mv_idx = 0;
    }
  } while (block_y + ldsp_mv[5].y <= max_range_y &&
           block_y + ldsp_mv[1].y >= min_range_y &&
           block_x + ldsp_mv[3].x <= max_range_x &&
           block_x + ldsp_mv[7].x >= min_range_x);

  sdsp_mv[0].y = ldsp_mv[best_mv_idx].y;
  sdsp_mv[0].x = ldsp_mv[best_mv_idx].x;
  best_mv_idx = 0;

  update_plane_mv_sdsp(sdsp_mv, search_step);
  for (int i = 0; i < 5; i++) {
    av1_get_pred_plane_interp_sub_pel_8(
        block_x, block_y, block_width, block_height, sdsp_mv[i].x, sdsp_mv[i].y,
        cur_frame, ref_frame, frame_stride, frame_width, frame_height,
        pred_block_stride, cur_block, pred_block);

    temp_sad =
        get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                          pred_block_stride, pred_block_stride);

    if (temp_sad < best_sad) {
      best_sad = temp_sad;
      best_mv_idx = i;
    }
  }  // for

  best_mv->y = sdsp_mv[best_mv_idx].y;
  best_mv->x = sdsp_mv[best_mv_idx].x;

  return best_sad;
}

int av1_motion_search_diamond_plane_interp_sub_pel_8sharp(
    int block_x, int block_y, int block_width, int block_height,
    const uint8_t *cur_frame, const uint8_t *ref_frame, int frame_stride,
    int frame_width, int frame_height, int search_range, PlaneMV *best_mv) {
  assert(cur_frame != NULL && ref_frame != NULL && best_mv != NULL);
  assert(block_width > 0 && block_height > 0 && block_width <= 128 &&
         block_height <= 128 && block_x >= 0 && block_y >= 0 &&
         frame_width > 0 && frame_height > 0);
  assert(search_range > 0);

  // Large Diamond Search Pattern on shpere
  PlaneMV ldsp_mv[9];
  // Small Diamond Search Pattern on shpere
  PlaneMV sdsp_mv[5];

  int search_step = block_height / 5;

  uint8_t cur_block[128 * 128];
  uint8_t pred_block[128 * 128];
  const int pred_block_stride = 128;

  int max_range_y = block_y + search_range;
  int min_range_y = block_y - search_range;
  int max_range_x = block_x + search_range;
  int min_range_x = block_x - search_range;

  int temp_sad;
  int best_sad;
  av1_get_pred_plane_interp_sub_pel_8sharp(
      block_x, block_y, block_width, block_height, 0, 0, cur_frame, ref_frame,
      frame_stride, frame_width, frame_height, pred_block_stride, cur_block,
      pred_block);
  best_sad = get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               pred_block_stride, pred_block_stride);

  int best_mv_idx = 0;
  ldsp_mv[0].y = 0;
  ldsp_mv[0].x = 0;

  const double min_search_step = 0.125;

  do {
    update_plane_mv_ldsp(ldsp_mv, search_step);

    for (int i = 0; i < 9; i++) {
      av1_get_pred_plane_interp_sub_pel_8sharp(
          block_x, block_y, block_width, block_height, ldsp_mv[i].x,
          ldsp_mv[i].y, cur_frame, ref_frame, frame_stride, frame_width,
          frame_height, pred_block_stride, cur_block, pred_block);

      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            pred_block_stride, pred_block_stride);

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv_idx = i;
      }
    }  // for

    if (best_mv_idx == 0) {
      if (search_step > min_search_step) {
        search_step *= 0.5;
        continue;
      } else {
        break;
      }
    } else {
      ldsp_mv[0].y = ldsp_mv[best_mv_idx].y;
      ldsp_mv[0].x = ldsp_mv[best_mv_idx].x;
      best_mv_idx = 0;
    }
  } while (block_y + ldsp_mv[5].y <= max_range_y &&
           block_y + ldsp_mv[1].y >= min_range_y &&
           block_x + ldsp_mv[3].x <= max_range_x &&
           block_x + ldsp_mv[7].x >= min_range_x);

  sdsp_mv[0].y = ldsp_mv[best_mv_idx].y;
  sdsp_mv[0].x = ldsp_mv[best_mv_idx].x;
  best_mv_idx = 0;

  update_plane_mv_sdsp(sdsp_mv, search_step);
  for (int i = 0; i < 5; i++) {
    av1_get_pred_plane_interp_sub_pel_8sharp(
        block_x, block_y, block_width, block_height, sdsp_mv[i].x, sdsp_mv[i].y,
        cur_frame, ref_frame, frame_stride, frame_width, frame_height,
        pred_block_stride, cur_block, pred_block);

    temp_sad =
        get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                          pred_block_stride, pred_block_stride);

    if (temp_sad < best_sad) {
      best_sad = temp_sad;
      best_mv_idx = i;
    }
  }  // for

  best_mv->y = sdsp_mv[best_mv_idx].y;
  best_mv->x = sdsp_mv[best_mv_idx].x;

  return best_sad;
}

int av1_motion_search_diamond_plane_interp_sub_pel_8smooth(
    int block_x, int block_y, int block_width, int block_height,
    const uint8_t *cur_frame, const uint8_t *ref_frame, int frame_stride,
    int frame_width, int frame_height, int search_range, PlaneMV *best_mv) {
  assert(cur_frame != NULL && ref_frame != NULL && best_mv != NULL);
  assert(block_width > 0 && block_height > 0 && block_width <= 128 &&
         block_height <= 128 && block_x >= 0 && block_y >= 0 &&
         frame_width > 0 && frame_height > 0);
  assert(search_range > 0);

  // Large Diamond Search Pattern on shpere
  PlaneMV ldsp_mv[9];
  // Small Diamond Search Pattern on shpere
  PlaneMV sdsp_mv[5];

  int search_step = block_height / 5;

  uint8_t cur_block[128 * 128];
  uint8_t pred_block[128 * 128];
  const int pred_block_stride = 128;

  int max_range_y = block_y + search_range;
  int min_range_y = block_y - search_range;
  int max_range_x = block_x + search_range;
  int min_range_x = block_x - search_range;

  int temp_sad;
  int best_sad;
  av1_get_pred_plane_interp_sub_pel_8smooth(
      block_x, block_y, block_width, block_height, 0, 0, cur_frame, ref_frame,
      frame_stride, frame_width, frame_height, pred_block_stride, cur_block,
      pred_block);
  best_sad = get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               pred_block_stride, pred_block_stride);

  int best_mv_idx = 0;
  ldsp_mv[0].y = 0;
  ldsp_mv[0].x = 0;

  const double min_search_step = 0.125;

  do {
    update_plane_mv_ldsp(ldsp_mv, search_step);

    for (int i = 0; i < 9; i++) {
      av1_get_pred_plane_interp_sub_pel_8smooth(
          block_x, block_y, block_width, block_height, ldsp_mv[i].x,
          ldsp_mv[i].y, cur_frame, ref_frame, frame_stride, frame_width,
          frame_height, pred_block_stride, cur_block, pred_block);

      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            pred_block_stride, pred_block_stride);

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv_idx = i;
      }
    }  // for

    if (best_mv_idx == 0) {
      if (search_step > min_search_step) {
        search_step *= 0.5;
        continue;
      } else {
        break;
      }
    } else {
      ldsp_mv[0].y = ldsp_mv[best_mv_idx].y;
      ldsp_mv[0].x = ldsp_mv[best_mv_idx].x;
      best_mv_idx = 0;
    }
  } while (block_y + ldsp_mv[5].y <= max_range_y &&
           block_y + ldsp_mv[1].y >= min_range_y &&
           block_x + ldsp_mv[3].x <= max_range_x &&
           block_x + ldsp_mv[7].x >= min_range_x);

  sdsp_mv[0].y = ldsp_mv[best_mv_idx].y;
  sdsp_mv[0].x = ldsp_mv[best_mv_idx].x;
  best_mv_idx = 0;

  update_plane_mv_sdsp(sdsp_mv, search_step);
  for (int i = 0; i < 5; i++) {
    av1_get_pred_plane_interp_sub_pel_8smooth(
        block_x, block_y, block_width, block_height, sdsp_mv[i].x, sdsp_mv[i].y,
        cur_frame, ref_frame, frame_stride, frame_width, frame_height,
        pred_block_stride, cur_block, pred_block);

    temp_sad =
        get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                          pred_block_stride, pred_block_stride);

    if (temp_sad < best_sad) {
      best_sad = temp_sad;
      best_mv_idx = i;
    }
  }  // for

  best_mv->y = sdsp_mv[best_mv_idx].y;
  best_mv->x = sdsp_mv[best_mv_idx].x;

  return best_sad;
}

int av1_motion_search_diamond_plane_interp_filter_search(
    int block_x, int block_y, int block_width, int block_height,
    const uint8_t *cur_frame, const uint8_t *ref_frame, int frame_stride,
    int frame_width, int frame_height, int search_range, PlaneMV *best_mv) {
  assert(cur_frame != NULL && ref_frame != NULL && best_mv != NULL);
  assert(block_width > 0 && block_height > 0 && block_width <= 128 &&
         block_height <= 128 && block_x >= 0 && block_y >= 0 &&
         frame_width > 0 && frame_height > 0);
  assert(search_range > 0);

  // Large Diamond Search Pattern on shpere
  PlaneMV ldsp_mv[9];
  // Small Diamond Search Pattern on shpere
  PlaneMV sdsp_mv[5];

  int search_step = block_height / 5;

  uint8_t cur_block[128 * 128];
  uint8_t pred_block[128 * 128];
  const int pred_block_stride = 128;

  int max_range_y = block_y + search_range;
  int min_range_y = block_y - search_range;
  int max_range_x = block_x + search_range;
  int min_range_x = block_x - search_range;

  int temp_sad;
  int best_sad;
  av1_get_pred_plane_interp_bilinear(block_x, block_y, block_width,
                                     block_height, 0, 0, cur_frame, ref_frame,
                                     frame_stride, frame_width, frame_height,
                                     pred_block_stride, cur_block, pred_block);
  best_sad = get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               pred_block_stride, pred_block_stride);

  int best_mv_idx = 0;
  ldsp_mv[0].y = 0;
  ldsp_mv[0].x = 0;

  const double min_search_step = 0.125;

  do {
    update_plane_mv_ldsp(ldsp_mv, search_step);

    for (int i = 0; i < 9; i++) {
      av1_get_pred_plane_interp_bilinear(
          block_x, block_y, block_width, block_height, ldsp_mv[i].x,
          ldsp_mv[i].y, cur_frame, ref_frame, frame_stride, frame_width,
          frame_height, pred_block_stride, cur_block, pred_block);

      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            pred_block_stride, pred_block_stride);

      av1_get_pred_plane_interp_sub_pel_8(
          block_x, block_y, block_width, block_height, ldsp_mv[i].x,
          ldsp_mv[i].y, cur_frame, ref_frame, frame_stride, frame_width,
          frame_height, pred_block_stride, cur_block, pred_block);
      temp_sad =
          AOMMIN(temp_sad, get_sad_of_blocks(cur_block, pred_block, block_width,
                                             block_height, pred_block_stride,
                                             pred_block_stride));

      av1_get_pred_plane_interp_sub_pel_8sharp(
          block_x, block_y, block_width, block_height, ldsp_mv[i].x,
          ldsp_mv[i].y, cur_frame, ref_frame, frame_stride, frame_width,
          frame_height, pred_block_stride, cur_block, pred_block);
      temp_sad =
          AOMMIN(temp_sad, get_sad_of_blocks(cur_block, pred_block, block_width,
                                             block_height, pred_block_stride,
                                             pred_block_stride));

      av1_get_pred_plane_interp_sub_pel_8smooth(
          block_x, block_y, block_width, block_height, ldsp_mv[i].x,
          ldsp_mv[i].y, cur_frame, ref_frame, frame_stride, frame_width,
          frame_height, pred_block_stride, cur_block, pred_block);
      temp_sad =
          AOMMIN(temp_sad, get_sad_of_blocks(cur_block, pred_block, block_width,
                                             block_height, pred_block_stride,
                                             pred_block_stride));

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv_idx = i;
      }
    }  // for

    if (best_mv_idx == 0) {
      if (search_step > min_search_step) {
        search_step *= 0.5;
        continue;
      } else {
        break;
      }
    } else {
      ldsp_mv[0].y = ldsp_mv[best_mv_idx].y;
      ldsp_mv[0].x = ldsp_mv[best_mv_idx].x;
      best_mv_idx = 0;
    }
  } while (block_y + ldsp_mv[5].y <= max_range_y &&
           block_y + ldsp_mv[1].y >= min_range_y &&
           block_x + ldsp_mv[3].x <= max_range_x &&
           block_x + ldsp_mv[7].x >= min_range_x);

  sdsp_mv[0].y = ldsp_mv[best_mv_idx].y;
  sdsp_mv[0].x = ldsp_mv[best_mv_idx].x;
  best_mv_idx = 0;

  update_plane_mv_sdsp(sdsp_mv, search_step);
  for (int i = 0; i < 5; i++) {
    av1_get_pred_plane_interp_bilinear(
        block_x, block_y, block_width, block_height, sdsp_mv[i].x, sdsp_mv[i].y,
        cur_frame, ref_frame, frame_stride, frame_width, frame_height,
        pred_block_stride, cur_block, pred_block);

    temp_sad =
        get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                          pred_block_stride, pred_block_stride);

    av1_get_pred_plane_interp_sub_pel_8(
        block_x, block_y, block_width, block_height, sdsp_mv[i].x, sdsp_mv[i].y,
        cur_frame, ref_frame, frame_stride, frame_width, frame_height,
        pred_block_stride, cur_block, pred_block);
    temp_sad =
        AOMMIN(temp_sad, get_sad_of_blocks(cur_block, pred_block, block_width,
                                           block_height, pred_block_stride,
                                           pred_block_stride));

    av1_get_pred_plane_interp_sub_pel_8sharp(
        block_x, block_y, block_width, block_height, sdsp_mv[i].x, sdsp_mv[i].y,
        cur_frame, ref_frame, frame_stride, frame_width, frame_height,
        pred_block_stride, cur_block, pred_block);
    temp_sad =
        AOMMIN(temp_sad, get_sad_of_blocks(cur_block, pred_block, block_width,
                                           block_height, pred_block_stride,
                                           pred_block_stride));

    av1_get_pred_plane_interp_sub_pel_8smooth(
        block_x, block_y, block_width, block_height, sdsp_mv[i].x, sdsp_mv[i].y,
        cur_frame, ref_frame, frame_stride, frame_width, frame_height,
        pred_block_stride, cur_block, pred_block);
    temp_sad =
        AOMMIN(temp_sad, get_sad_of_blocks(cur_block, pred_block, block_width,
                                           block_height, pred_block_stride,
                                           pred_block_stride));

    if (temp_sad < best_sad) {
      best_sad = temp_sad;
      best_mv_idx = i;
    }
  }  // for

  best_mv->y = sdsp_mv[best_mv_idx].y;
  best_mv->x = sdsp_mv[best_mv_idx].x;

  return best_sad;
}

uint8_t interp_pixel_erp(const uint8_t *ref_frame, int ref_frame_stride,
                         int frame_width, int frame_height, double subpel_x,
                         double subpel_y, const InterpKernel *kernel,
                         int filter_tap) {
  double x_sum_arr[8];
  double sum = 0;
  double coeff = 0;
  int x_kernel_idx;
  int y_kernel_idx;
  int x_ref;
  int y_ref;
  int pixel;

  x_kernel_idx = cal_kernel_idx(subpel_x, &x_ref);
  y_kernel_idx = cal_kernel_idx(subpel_y, &y_ref);

  get_interp_x_arr(x_ref, y_ref, ref_frame, ref_frame_stride, frame_width,
                   frame_height, kernel, filter_tap, x_kernel_idx, x_sum_arr);

  // Perform filtering on the column
  sum = 0;
  for (int k = 0; k < filter_tap; k++) {
    coeff = kernel[y_kernel_idx][k] / 128.0;
    sum += x_sum_arr[k] * coeff;
  }

  pixel = (int)round(sum);
  pixel = clamp(pixel, 0, 255);
  return (uint8_t)pixel;
}

void av1_get_pred_erp_interp(int block_x, int block_y, int block_width,
                             int block_height, double delta_phi,
                             double delta_theta, const uint8_t *ref_frame,
                             int ref_frame_stride, int frame_width,
                             int frame_height, int pred_block_stride,
                             uint8_t *pred_block) {
  assert(ref_frame != NULL && frame_width > 0 && frame_height > 0 &&
         ref_frame_stride >= frame_width);
  assert(pred_block != NULL && block_width > 0 && block_height > 0 &&
         pred_block_stride >= block_width);

  int pos_pred;         // Offset in pred_block buffer
  double x_current;     // X coordiante in current frame
  double y_current;     // Y coordiante in currrent frame
  double subpel_x_ref;  // X coordinate in reference frame
  double subpel_y_ref;  // Y coordiante in reference frame

  PolarVector block_polar;
  CartesianVector block_v;
  CartesianVector block_v_prime;
  CartesianVector k;  // The axis that the block will rotate along
  block_polar.r = 1.0;
  av1_plane_to_sphere_erp(block_x, block_y, frame_width, frame_height,
                          &block_polar.phi, &block_polar.theta);
  block_polar.phi += 0.5 * PI;
  av1_sphere_polar_to_carte(&block_polar, &block_v);

  block_polar.phi += delta_phi;
  block_polar.theta += delta_theta;
  av1_sphere_polar_to_carte(&block_polar, &block_v_prime);

  av1_carte_vectors_cross_product(&block_v, &block_v_prime, &k);
  av1_normalize_carte_vector(&k);

  double product = av1_carte_vectors_dot_product(&block_v, &block_v_prime);
  // Avoid floating point precision issues
  product = AOMMAX(product, -1.0);
  product = AOMMIN(product, 1.0);
  double alpha = acos(product);

  CartesianVector v;
  CartesianVector v_prime;
  PolarVector v_polar;
  PolarVector v_prime_polar;
  v_polar.r = 1.0;
  v_prime_polar.r = 1.0;
  double k_dot_v;

  const int filter_tap = 8;

  for (int idx_y = 0; idx_y < block_height; idx_y++) {
    for (int idx_x = 0; idx_x < block_width; idx_x++) {
      x_current = idx_x + block_x;
      y_current = idx_y + block_y;

      av1_plane_to_sphere_erp(x_current, y_current, frame_width, frame_height,
                              &v_polar.phi, &v_polar.theta);
      v_polar.phi += 0.5 * PI;
      av1_sphere_polar_to_carte(&v_polar, &v);
      k_dot_v = av1_carte_vectors_dot_product(&k, &v);

      v_prime.x = k.x * k_dot_v * (1 - cos(alpha)) + v.x * cos(alpha) +
                  (k.y * v.z - k.z * v.y) * sin(alpha);
      v_prime.y = k.y * k_dot_v * (1 - cos(alpha)) + v.y * cos(alpha) +
                  (k.z * v.x - k.x * v.z) * sin(alpha);
      v_prime.z = k.z * k_dot_v * (1 - cos(alpha)) + v.z * cos(alpha) +
                  (k.x * v.y - k.y * v.x) * sin(alpha);

      av1_sphere_carte_to_polar(&v_prime, &v_prime_polar);
      v_prime_polar.phi -= 0.5 * PI;
      av1_sphere_to_plane_erp(v_prime_polar.phi, v_prime_polar.theta,
                              frame_width, frame_height, &subpel_x_ref,
                              &subpel_y_ref);

      pos_pred = idx_x + idx_y * pred_block_stride;
      pred_block[pos_pred] = interp_pixel_erp(
          ref_frame, ref_frame_stride, frame_width, frame_height, subpel_x_ref,
          subpel_y_ref, av1_sub_pel_filters_8, filter_tap);
    }
  }  // for idx_y
}

void av1_get_pred_erp_interp_bilinear(int block_x, int block_y, int block_width,
                                      int block_height, double delta_phi,
                                      double delta_theta,
                                      const uint8_t *ref_frame,
                                      int ref_frame_stride, int frame_width,
                                      int frame_height, int pred_block_stride,
                                      uint8_t *pred_block) {
  assert(ref_frame != NULL && frame_width > 0 && frame_height > 0 &&
         ref_frame_stride >= frame_width);
  assert(pred_block != NULL && block_width > 0 && block_height > 0 &&
         pred_block_stride >= block_width);

  int pos_pred;         // Offset in pred_block buffer
  double x_current;     // X coordiante in current frame
  double y_current;     // Y coordiante in currrent frame
  double subpel_x_ref;  // X coordinate in reference frame
  double subpel_y_ref;  // Y coordiante in reference frame

  PolarVector block_polar;
  CartesianVector block_v;
  CartesianVector block_v_prime;
  CartesianVector k;  // The axis that the block will rotate along
  block_polar.r = 1.0;
  av1_plane_to_sphere_erp(block_x, block_y, frame_width, frame_height,
                          &block_polar.phi, &block_polar.theta);
  block_polar.phi += 0.5 * PI;
  av1_sphere_polar_to_carte(&block_polar, &block_v);

  block_polar.phi += delta_phi;
  block_polar.theta += delta_theta;
  av1_sphere_polar_to_carte(&block_polar, &block_v_prime);

  av1_carte_vectors_cross_product(&block_v, &block_v_prime, &k);
  av1_normalize_carte_vector(&k);

  double product = av1_carte_vectors_dot_product(&block_v, &block_v_prime);
  // Avoid floating point precision issues
  product = AOMMAX(product, -1.0);
  product = AOMMIN(product, 1.0);
  double alpha = acos(product);

  CartesianVector v;
  CartesianVector v_prime;
  PolarVector v_polar;
  PolarVector v_prime_polar;
  v_polar.r = 1.0;
  v_prime_polar.r = 1.0;
  double k_dot_v;

  const int filter_tap = 8;

  for (int idx_y = 0; idx_y < block_height; idx_y++) {
    for (int idx_x = 0; idx_x < block_width; idx_x++) {
      x_current = idx_x + block_x;
      y_current = idx_y + block_y;

      av1_plane_to_sphere_erp(x_current, y_current, frame_width, frame_height,
                              &v_polar.phi, &v_polar.theta);
      v_polar.phi += 0.5 * PI;
      av1_sphere_polar_to_carte(&v_polar, &v);
      k_dot_v = av1_carte_vectors_dot_product(&k, &v);

      v_prime.x = k.x * k_dot_v * (1 - cos(alpha)) + v.x * cos(alpha) +
                  (k.y * v.z - k.z * v.y) * sin(alpha);
      v_prime.y = k.y * k_dot_v * (1 - cos(alpha)) + v.y * cos(alpha) +
                  (k.z * v.x - k.x * v.z) * sin(alpha);
      v_prime.z = k.z * k_dot_v * (1 - cos(alpha)) + v.z * cos(alpha) +
                  (k.x * v.y - k.y * v.x) * sin(alpha);

      av1_sphere_carte_to_polar(&v_prime, &v_prime_polar);
      v_prime_polar.phi -= 0.5 * PI;
      av1_sphere_to_plane_erp(v_prime_polar.phi, v_prime_polar.theta,
                              frame_width, frame_height, &subpel_x_ref,
                              &subpel_y_ref);

      pos_pred = idx_x + idx_y * pred_block_stride;
      pred_block[pos_pred] = interp_pixel_erp(
          ref_frame, ref_frame_stride, frame_width, frame_height, subpel_x_ref,
          subpel_y_ref, av1_bilinear_filters, filter_tap);
    }
  }  // for idx_y
}

void av1_get_pred_erp_interp_sub_pel_8(int block_x, int block_y,
                                       int block_width, int block_height,
                                       double delta_phi, double delta_theta,
                                       const uint8_t *ref_frame,
                                       int ref_frame_stride, int frame_width,
                                       int frame_height, int pred_block_stride,
                                       uint8_t *pred_block) {
  assert(ref_frame != NULL && frame_width > 0 && frame_height > 0 &&
         ref_frame_stride >= frame_width);
  assert(pred_block != NULL && block_width > 0 && block_height > 0 &&
         pred_block_stride >= block_width);

  int pos_pred;         // Offset in pred_block buffer
  double x_current;     // X coordiante in current frame
  double y_current;     // Y coordiante in currrent frame
  double subpel_x_ref;  // X coordinate in reference frame
  double subpel_y_ref;  // Y coordiante in reference frame

  PolarVector block_polar;
  CartesianVector block_v;
  CartesianVector block_v_prime;
  CartesianVector k;  // The axis that the block will rotate along
  block_polar.r = 1.0;
  av1_plane_to_sphere_erp(block_x, block_y, frame_width, frame_height,
                          &block_polar.phi, &block_polar.theta);
  block_polar.phi += 0.5 * PI;
  av1_sphere_polar_to_carte(&block_polar, &block_v);

  block_polar.phi += delta_phi;
  block_polar.theta += delta_theta;
  av1_sphere_polar_to_carte(&block_polar, &block_v_prime);

  av1_carte_vectors_cross_product(&block_v, &block_v_prime, &k);
  av1_normalize_carte_vector(&k);

  double product = av1_carte_vectors_dot_product(&block_v, &block_v_prime);
  // Avoid floating point precision issues
  product = AOMMAX(product, -1.0);
  product = AOMMIN(product, 1.0);
  double alpha = acos(product);

  CartesianVector v;
  CartesianVector v_prime;
  PolarVector v_polar;
  PolarVector v_prime_polar;
  v_polar.r = 1.0;
  v_prime_polar.r = 1.0;
  double k_dot_v;

  const int filter_tap = 8;

  for (int idx_y = 0; idx_y < block_height; idx_y++) {
    for (int idx_x = 0; idx_x < block_width; idx_x++) {
      x_current = idx_x + block_x;
      y_current = idx_y + block_y;

      av1_plane_to_sphere_erp(x_current, y_current, frame_width, frame_height,
                              &v_polar.phi, &v_polar.theta);
      v_polar.phi += 0.5 * PI;
      av1_sphere_polar_to_carte(&v_polar, &v);
      k_dot_v = av1_carte_vectors_dot_product(&k, &v);

      v_prime.x = k.x * k_dot_v * (1 - cos(alpha)) + v.x * cos(alpha) +
                  (k.y * v.z - k.z * v.y) * sin(alpha);
      v_prime.y = k.y * k_dot_v * (1 - cos(alpha)) + v.y * cos(alpha) +
                  (k.z * v.x - k.x * v.z) * sin(alpha);
      v_prime.z = k.z * k_dot_v * (1 - cos(alpha)) + v.z * cos(alpha) +
                  (k.x * v.y - k.y * v.x) * sin(alpha);

      av1_sphere_carte_to_polar(&v_prime, &v_prime_polar);
      v_prime_polar.phi -= 0.5 * PI;
      av1_sphere_to_plane_erp(v_prime_polar.phi, v_prime_polar.theta,
                              frame_width, frame_height, &subpel_x_ref,
                              &subpel_y_ref);

      pos_pred = idx_x + idx_y * pred_block_stride;
      pred_block[pos_pred] = interp_pixel_erp(
          ref_frame, ref_frame_stride, frame_width, frame_height, subpel_x_ref,
          subpel_y_ref, av1_sub_pel_filters_8, filter_tap);
    }
  }  // for idx_y
}

void av1_get_pred_erp_interp_sub_pel_8sharp(
    int block_x, int block_y, int block_width, int block_height,
    double delta_phi, double delta_theta, const uint8_t *ref_frame,
    int ref_frame_stride, int frame_width, int frame_height,
    int pred_block_stride, uint8_t *pred_block) {
  assert(ref_frame != NULL && frame_width > 0 && frame_height > 0 &&
         ref_frame_stride >= frame_width);
  assert(pred_block != NULL && block_width > 0 && block_height > 0 &&
         pred_block_stride >= block_width);

  int pos_pred;         // Offset in pred_block buffer
  double x_current;     // X coordiante in current frame
  double y_current;     // Y coordiante in currrent frame
  double subpel_x_ref;  // X coordinate in reference frame
  double subpel_y_ref;  // Y coordiante in reference frame

  PolarVector block_polar;
  CartesianVector block_v;
  CartesianVector block_v_prime;
  CartesianVector k;  // The axis that the block will rotate along
  block_polar.r = 1.0;
  av1_plane_to_sphere_erp(block_x, block_y, frame_width, frame_height,
                          &block_polar.phi, &block_polar.theta);
  block_polar.phi += 0.5 * PI;
  av1_sphere_polar_to_carte(&block_polar, &block_v);

  block_polar.phi += delta_phi;
  block_polar.theta += delta_theta;
  av1_sphere_polar_to_carte(&block_polar, &block_v_prime);

  av1_carte_vectors_cross_product(&block_v, &block_v_prime, &k);
  av1_normalize_carte_vector(&k);

  double product = av1_carte_vectors_dot_product(&block_v, &block_v_prime);
  // Avoid floating point precision issues
  product = AOMMAX(product, -1.0);
  product = AOMMIN(product, 1.0);
  double alpha = acos(product);

  CartesianVector v;
  CartesianVector v_prime;
  PolarVector v_polar;
  PolarVector v_prime_polar;
  v_polar.r = 1.0;
  v_prime_polar.r = 1.0;
  double k_dot_v;

  const int filter_tap = 8;

  for (int idx_y = 0; idx_y < block_height; idx_y++) {
    for (int idx_x = 0; idx_x < block_width; idx_x++) {
      x_current = idx_x + block_x;
      y_current = idx_y + block_y;

      av1_plane_to_sphere_erp(x_current, y_current, frame_width, frame_height,
                              &v_polar.phi, &v_polar.theta);
      v_polar.phi += 0.5 * PI;
      av1_sphere_polar_to_carte(&v_polar, &v);
      k_dot_v = av1_carte_vectors_dot_product(&k, &v);

      v_prime.x = k.x * k_dot_v * (1 - cos(alpha)) + v.x * cos(alpha) +
                  (k.y * v.z - k.z * v.y) * sin(alpha);
      v_prime.y = k.y * k_dot_v * (1 - cos(alpha)) + v.y * cos(alpha) +
                  (k.z * v.x - k.x * v.z) * sin(alpha);
      v_prime.z = k.z * k_dot_v * (1 - cos(alpha)) + v.z * cos(alpha) +
                  (k.x * v.y - k.y * v.x) * sin(alpha);

      av1_sphere_carte_to_polar(&v_prime, &v_prime_polar);
      v_prime_polar.phi -= 0.5 * PI;
      av1_sphere_to_plane_erp(v_prime_polar.phi, v_prime_polar.theta,
                              frame_width, frame_height, &subpel_x_ref,
                              &subpel_y_ref);

      pos_pred = idx_x + idx_y * pred_block_stride;
      pred_block[pos_pred] = interp_pixel_erp(
          ref_frame, ref_frame_stride, frame_width, frame_height, subpel_x_ref,
          subpel_y_ref, av1_sub_pel_filters_8sharp, filter_tap);
    }
  }  // for idx_y
}

void av1_get_pred_erp_interp_sub_pel_8smooth(
    int block_x, int block_y, int block_width, int block_height,
    double delta_phi, double delta_theta, const uint8_t *ref_frame,
    int ref_frame_stride, int frame_width, int frame_height,
    int pred_block_stride, uint8_t *pred_block) {
  assert(ref_frame != NULL && frame_width > 0 && frame_height > 0 &&
         ref_frame_stride >= frame_width);
  assert(pred_block != NULL && block_width > 0 && block_height > 0 &&
         pred_block_stride >= block_width);

  int pos_pred;         // Offset in pred_block buffer
  double x_current;     // X coordiante in current frame
  double y_current;     // Y coordiante in currrent frame
  double subpel_x_ref;  // X coordinate in reference frame
  double subpel_y_ref;  // Y coordiante in reference frame

  PolarVector block_polar;
  CartesianVector block_v;
  CartesianVector block_v_prime;
  CartesianVector k;  // The axis that the block will rotate along
  block_polar.r = 1.0;
  av1_plane_to_sphere_erp(block_x, block_y, frame_width, frame_height,
                          &block_polar.phi, &block_polar.theta);
  block_polar.phi += 0.5 * PI;
  av1_sphere_polar_to_carte(&block_polar, &block_v);

  block_polar.phi += delta_phi;
  block_polar.theta += delta_theta;
  av1_sphere_polar_to_carte(&block_polar, &block_v_prime);

  av1_carte_vectors_cross_product(&block_v, &block_v_prime, &k);
  av1_normalize_carte_vector(&k);

  double product = av1_carte_vectors_dot_product(&block_v, &block_v_prime);
  // Avoid floating point precision issues
  product = AOMMAX(product, -1.0);
  product = AOMMIN(product, 1.0);
  double alpha = acos(product);

  CartesianVector v;
  CartesianVector v_prime;
  PolarVector v_polar;
  PolarVector v_prime_polar;
  v_polar.r = 1.0;
  v_prime_polar.r = 1.0;
  double k_dot_v;

  const int filter_tap = 8;

  for (int idx_y = 0; idx_y < block_height; idx_y++) {
    for (int idx_x = 0; idx_x < block_width; idx_x++) {
      x_current = idx_x + block_x;
      y_current = idx_y + block_y;

      av1_plane_to_sphere_erp(x_current, y_current, frame_width, frame_height,
                              &v_polar.phi, &v_polar.theta);
      v_polar.phi += 0.5 * PI;
      av1_sphere_polar_to_carte(&v_polar, &v);
      k_dot_v = av1_carte_vectors_dot_product(&k, &v);

      v_prime.x = k.x * k_dot_v * (1 - cos(alpha)) + v.x * cos(alpha) +
                  (k.y * v.z - k.z * v.y) * sin(alpha);
      v_prime.y = k.y * k_dot_v * (1 - cos(alpha)) + v.y * cos(alpha) +
                  (k.z * v.x - k.x * v.z) * sin(alpha);
      v_prime.z = k.z * k_dot_v * (1 - cos(alpha)) + v.z * cos(alpha) +
                  (k.x * v.y - k.y * v.x) * sin(alpha);

      av1_sphere_carte_to_polar(&v_prime, &v_prime_polar);
      v_prime_polar.phi -= 0.5 * PI;
      av1_sphere_to_plane_erp(v_prime_polar.phi, v_prime_polar.theta,
                              frame_width, frame_height, &subpel_x_ref,
                              &subpel_y_ref);

      pos_pred = idx_x + idx_y * pred_block_stride;
      pred_block[pos_pred] = interp_pixel_erp(
          ref_frame, ref_frame_stride, frame_width, frame_height, subpel_x_ref,
          subpel_y_ref, av1_sub_pel_filters_8smooth, filter_tap);
    }
  }  // for idx_y
}

int av1_motion_search_brute_force_erp_interp(
    int block_x, int block_y, int block_width, int block_height,
    const uint8_t *cur_frame, const uint8_t *ref_frame, int frame_stride,
    int frame_width, int frame_height, int search_range, SphereMV *best_mv) {
  assert(cur_frame != NULL && ref_frame != NULL && best_mv != NULL);
  assert(block_width > 0 && block_height > 0 && block_width <= 128 &&
         block_height <= 128 && block_x >= 0 && block_y >= 0 &&
         frame_width > 0 && frame_height > 0);
  assert(search_range > 0);

  const double search_step_phi = PI / frame_height;
  const double search_step_theta = 2 * PI / frame_width;

  double delta_phi;
  double delta_theta;
  int temp_sad;
  int best_sad;

  const uint8_t *cur_block = &cur_frame[block_x + block_y * frame_stride];
  uint8_t pred_block[128 * 128];
  const int pred_block_stride = 128;

  av1_get_pred_erp_interp(block_x, block_y, block_width, block_height, 0, 0,
                          ref_frame, frame_stride, frame_width, frame_height,
                          pred_block_stride, pred_block);
  best_sad = get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               frame_stride, pred_block_stride);
  best_mv->phi = 0;
  best_mv->theta = 0;

  for (int i = -search_range; i <= search_range; i++) {
    delta_phi = i * search_step_phi;

    for (int j = -search_range; j <= search_range; j++) {
      delta_theta = j * search_step_theta;

      // cal_delta_theta(block_x, block_y, 10, frame_width, frame_height,
      //                 delta_phi, &delta_theta);

      av1_get_pred_erp_interp(block_x, block_y, block_width, block_height,
                              delta_phi, delta_theta, ref_frame, frame_stride,
                              frame_width, frame_height, pred_block_stride,
                              pred_block);

      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            frame_stride, pred_block_stride);

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv->phi = delta_phi;
        best_mv->theta = delta_theta;
      }
    }  // for j
  }    // for i

  return best_sad;
}

/* Update Large Diamond Search Pattern shperical motion vectors based on the
 * center
 *                    ldsp_mv[1]
 *                    /        \
 *          ldsp_mv[8]          ldsp_mv[2]
 *                 /              \
 *       ldsp_mv[7]   ldsp_mv[0]   ldsp_mv[3]
 *                 \              /
 *         ldsp_mv[6]            ldsp_mv[4]
 *                   \          /
 *                    ldsp_mv[5]
 */
static void update_sphere_mv_ldsp(SphereMV ldsp_mv[9], double search_step_phi,
                                  double search_step_theta) {
  ldsp_mv[1].phi = ldsp_mv[0].phi - 2 * search_step_phi;
  ldsp_mv[1].theta = ldsp_mv[0].theta;

  ldsp_mv[2].phi = ldsp_mv[0].phi - search_step_phi;
  ldsp_mv[2].theta = ldsp_mv[0].theta + search_step_theta;

  ldsp_mv[3].phi = ldsp_mv[0].phi;
  ldsp_mv[3].theta = ldsp_mv[0].theta + 2 * search_step_theta;

  ldsp_mv[4].phi = ldsp_mv[0].phi + search_step_phi;
  ldsp_mv[4].theta = ldsp_mv[0].theta + search_step_theta;

  ldsp_mv[5].phi = ldsp_mv[0].phi + 2 * search_step_phi;
  ldsp_mv[5].theta = ldsp_mv[0].theta;

  ldsp_mv[6].phi = ldsp_mv[0].phi + search_step_phi;
  ldsp_mv[6].theta = ldsp_mv[0].theta - search_step_theta;

  ldsp_mv[7].phi = ldsp_mv[0].phi;
  ldsp_mv[7].theta = ldsp_mv[0].theta - 2 * search_step_theta;

  ldsp_mv[8].phi = ldsp_mv[0].phi - search_step_phi;
  ldsp_mv[8].theta = ldsp_mv[0].theta - search_step_theta;
}

/* Small Diamond Search Pattern on shpere
 *                     sdsp_mv[1]
 *                  /              \
 *         sdsp_mv[4]  sdsp_mv[0]  sdsp_mv[2]
 *                  \              /
 *                     sdsp_mv[3]
 */
static void update_sphere_mv_sdsp(SphereMV sdsp_mv[5], double search_step_phi,
                                  double search_step_theta) {
  sdsp_mv[1].phi = sdsp_mv[0].phi - search_step_phi;
  sdsp_mv[1].theta = sdsp_mv[0].theta;

  sdsp_mv[2].phi = sdsp_mv[0].phi;
  sdsp_mv[2].theta = sdsp_mv[0].theta + search_step_theta;

  sdsp_mv[3].phi = sdsp_mv[0].phi + search_step_phi;
  sdsp_mv[3].theta = sdsp_mv[0].theta;

  sdsp_mv[4].phi = sdsp_mv[0].phi;
  sdsp_mv[4].theta = sdsp_mv[0].theta - search_step_theta;
}

int av1_motion_search_diamond_erp_interp(
    int block_x, int block_y, int block_width, int block_height,
    const uint8_t *cur_frame, const uint8_t *ref_frame, int frame_stride,
    int frame_width, int frame_height, int search_range,
    const SphereMV *start_mv, SphereMV *best_mv) {
  assert(cur_frame != NULL && ref_frame != NULL && best_mv != NULL);
  assert(block_width > 0 && block_height > 0 && block_width <= 128 &&
         block_height <= 128 && block_x >= 0 && block_y >= 0 &&
         frame_width > 0 && frame_height > 0);
  assert(search_range > 0);

  // Large Diamond Search Pattern on shpere
  SphereMV ldsp_mv[9];
  // Small Diamond Search Pattern on shpere
  SphereMV sdsp_mv[5];

  double search_step_phi = 0.125 * PI / frame_height;
  double search_step_theta = 0.125 * 2 * PI / frame_width;

  const uint8_t *cur_block = &cur_frame[block_x + block_y * frame_stride];
  uint8_t pred_block[128 * 128];
  const int pred_block_stride = 128;

  double start_phi;
  double start_theta;
  av1_plane_to_sphere_erp(block_x, block_y, frame_width, frame_height,
                          &start_phi, &start_theta);

  double max_range_phi = start_phi + search_range * PI / frame_height;
  double min_range_phi = start_phi - search_range * PI / frame_height;
  double max_range_theta = start_theta + search_range * 2 * PI / frame_height;
  double min_range_theta = start_theta - search_range * 2 * PI / frame_height;

  int best_mv_idx = 0;
  ldsp_mv[0].phi = start_mv->phi;
  ldsp_mv[0].theta = start_mv->theta;

  int temp_sad;
  int best_sad;
  av1_get_pred_erp_interp(block_x, block_y, block_width, block_height,
                          ldsp_mv[0].phi, ldsp_mv[0].theta, ref_frame,
                          frame_stride, frame_width, frame_height,
                          pred_block_stride, pred_block);
  best_sad = get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               frame_stride, pred_block_stride);

  do {
    update_sphere_mv_ldsp(ldsp_mv, search_step_phi, search_step_theta);

    for (int i = 0; i < 9; i++) {
      // cal_delta_theta(block_x, block_y, 10, frame_width, frame_height,
      //                 ldsp_mv[i].phi, &ldsp_mv[i].theta);
      av1_get_pred_erp_interp(block_x, block_y, block_width, block_height,
                              ldsp_mv[i].phi, ldsp_mv[i].theta, ref_frame,
                              frame_stride, frame_width, frame_height,
                              pred_block_stride, pred_block);

      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            frame_stride, pred_block_stride);

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv_idx = i;
      }
    }  // for

    if (best_mv_idx == 0) {
      if (search_step_phi > PI / frame_height &&
          search_step_theta > 2 * PI / frame_height) {
        search_step_phi *= 0.5;
        search_step_theta *= 0.5;
        continue;
      } else {
        break;
      }
    } else {
      ldsp_mv[0].phi = ldsp_mv[best_mv_idx].phi;
      ldsp_mv[0].theta = ldsp_mv[best_mv_idx].theta;
      best_mv_idx = 0;
    }
  } while (start_phi + ldsp_mv[5].phi <= max_range_phi &&
           start_phi + ldsp_mv[1].phi >= min_range_phi &&
           start_theta + ldsp_mv[3].theta <= max_range_theta &&
           start_theta + ldsp_mv[7].theta >= min_range_theta);

  sdsp_mv[0].phi = ldsp_mv[best_mv_idx].phi;
  sdsp_mv[0].theta = ldsp_mv[best_mv_idx].theta;
  best_mv_idx = 0;
  search_step_phi = PI / frame_height;
  search_step_theta = 2 * PI / frame_width;

  update_sphere_mv_sdsp(sdsp_mv, search_step_phi, search_step_theta);
  for (int i = 0; i < 5; i++) {
    av1_get_pred_erp(block_x, block_y, block_width, block_height,
                     sdsp_mv[i].phi, sdsp_mv[i].theta, ref_frame, frame_stride,
                     frame_width, frame_height, pred_block_stride, pred_block);

    temp_sad = get_sad_of_blocks(cur_block, pred_block, block_width,
                                 block_height, frame_stride, pred_block_stride);

    if (temp_sad < best_sad) {
      best_sad = temp_sad;
      best_mv_idx = i;
    }
  }  // for

  best_mv->phi = sdsp_mv[best_mv_idx].phi;
  best_mv->theta = sdsp_mv[best_mv_idx].theta;

  return best_sad;
}

int av1_motion_search_diamond_erp_interp_bilinear(
    int block_x, int block_y, int block_width, int block_height,
    const uint8_t *cur_frame, const uint8_t *ref_frame, int frame_stride,
    int frame_width, int frame_height, int search_range,
    const SphereMV *start_mv, SphereMV *best_mv) {
  assert(cur_frame != NULL && ref_frame != NULL && best_mv != NULL);
  assert(block_width > 0 && block_height > 0 && block_width <= 128 &&
         block_height <= 128 && block_x >= 0 && block_y >= 0 &&
         frame_width > 0 && frame_height > 0);
  assert(search_range > 0);

  // Large Diamond Search Pattern on shpere
  SphereMV ldsp_mv[9];
  // Small Diamond Search Pattern on shpere
  SphereMV sdsp_mv[5];

  double search_step_phi =
      AOMMIN(0.25 * block_height, 0.25 * search_range) * PI / frame_height;
  double search_step_theta =
      AOMMIN(0.25 * block_width, 0.25 * search_range) * 2 * PI / frame_width;

  // printf("Diamond ERP interp search step phi: %f, theta: %f\n",
  // search_step_phi,
  //        search_step_theta);

  const uint8_t *cur_block = &cur_frame[block_x + block_y * frame_stride];
  uint8_t pred_block[128 * 128];
  const int pred_block_stride = 128;

  double start_phi;
  double start_theta;
  av1_plane_to_sphere_erp(block_x, block_y, frame_width, frame_height,
                          &start_phi, &start_theta);

  double max_range_phi =
      start_phi + start_mv->phi + search_range * PI / frame_height;
  double min_range_phi =
      start_phi + start_mv->theta - search_range * PI / frame_height;
  double max_range_theta = start_theta + search_range * 2 * PI / frame_height;
  double min_range_theta = start_theta - search_range * 2 * PI / frame_height;

  int best_mv_idx = 0;
  ldsp_mv[0].phi = start_mv->phi;
  ldsp_mv[0].theta = start_mv->theta;

  int temp_sad;
  int best_sad;
  av1_get_pred_erp_interp_bilinear(block_x, block_y, block_width, block_height,
                                   ldsp_mv[0].phi, ldsp_mv[0].theta, ref_frame,
                                   frame_stride, frame_width, frame_height,
                                   pred_block_stride, pred_block);
  best_sad = get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               frame_stride, pred_block_stride);

  update_sphere_mv_ldsp(ldsp_mv, search_step_phi, search_step_theta);

  const double min_scale = 0.125;

  while (start_phi + ldsp_mv[5].phi <= max_range_phi &&
         start_phi + ldsp_mv[1].phi >= min_range_phi &&
         start_theta + ldsp_mv[3].theta <= max_range_theta &&
         start_theta + ldsp_mv[7].theta >= min_range_theta) {
    for (int i = 0; i < 9; i++) {
      av1_get_pred_erp_interp_bilinear(
          block_x, block_y, block_width, block_height, ldsp_mv[i].phi,
          ldsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);

      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            frame_stride, pred_block_stride);

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv_idx = i;
      }
    }  // for

    if (best_mv_idx == 0) {
      if (search_step_phi > min_scale * PI / frame_height &&
          search_step_theta > min_scale * 2 * PI / frame_height) {
        search_step_phi *= 0.5;
        search_step_theta *= 0.5;
      } else {
        break;
      }
    } else {
      ldsp_mv[0].phi = ldsp_mv[best_mv_idx].phi;
      ldsp_mv[0].theta = ldsp_mv[best_mv_idx].theta;
      best_mv_idx = 0;
    }
    update_sphere_mv_ldsp(ldsp_mv, search_step_phi, search_step_theta);
  }

  sdsp_mv[0].phi = ldsp_mv[0].phi;
  sdsp_mv[0].theta = ldsp_mv[0].theta;
  best_mv_idx = 0;
  search_step_phi = min_scale * PI / frame_height;
  search_step_theta = min_scale * 2 * PI / frame_width;
  update_sphere_mv_sdsp(sdsp_mv, search_step_phi, search_step_theta);
  if (start_phi + sdsp_mv[3].phi <= max_range_phi &&
      start_phi + sdsp_mv[1].phi >= min_range_phi &&
      start_theta + sdsp_mv[2].theta <= max_range_theta &&
      start_theta + sdsp_mv[4].theta >= min_range_theta) {
    for (int i = 0; i < 5; i++) {
      av1_get_pred_erp_interp_bilinear(
          block_x, block_y, block_width, block_height, sdsp_mv[i].phi,
          sdsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);

      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            frame_stride, pred_block_stride);

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv_idx = i;
      }
    }  // for
  }

  best_mv->phi = sdsp_mv[best_mv_idx].phi;
  best_mv->theta = sdsp_mv[best_mv_idx].theta;

  return best_sad;
}

int av1_motion_search_diamond_erp_interp_sub_pel_8(
    int block_x, int block_y, int block_width, int block_height,
    const uint8_t *cur_frame, const uint8_t *ref_frame, int frame_stride,
    int frame_width, int frame_height, int search_range,
    const SphereMV *start_mv, SphereMV *best_mv) {
  assert(cur_frame != NULL && ref_frame != NULL && best_mv != NULL);
  assert(block_width > 0 && block_height > 0 && block_width <= 128 &&
         block_height <= 128 && block_x >= 0 && block_y >= 0 &&
         frame_width > 0 && frame_height > 0);
  assert(search_range > 0);

  // Large Diamond Search Pattern on shpere
  SphereMV ldsp_mv[9];
  // Small Diamond Search Pattern on shpere
  SphereMV sdsp_mv[5];

  double search_step_phi =
      AOMMIN(0.25 * block_height, 0.25 * search_range) * PI / frame_height;
  double search_step_theta =
      AOMMIN(0.25 * block_width, 0.25 * search_range) * 2 * PI / frame_width;

  // printf("Diamond ERP interp search step phi: %f, theta: %f\n",
  // search_step_phi,
  //        search_step_theta);

  const uint8_t *cur_block = &cur_frame[block_x + block_y * frame_stride];
  uint8_t pred_block[128 * 128];
  const int pred_block_stride = 128;

  double start_phi;
  double start_theta;
  av1_plane_to_sphere_erp(block_x, block_y, frame_width, frame_height,
                          &start_phi, &start_theta);

  double max_range_phi =
      start_phi + start_mv->phi + search_range * PI / frame_height;
  double min_range_phi =
      start_phi + start_mv->theta - search_range * PI / frame_height;
  double max_range_theta = start_theta + search_range * 2 * PI / frame_height;
  double min_range_theta = start_theta - search_range * 2 * PI / frame_height;

  int best_mv_idx = 0;
  ldsp_mv[0].phi = start_mv->phi;
  ldsp_mv[0].theta = start_mv->theta;

  int temp_sad;
  int best_sad;
  av1_get_pred_erp_interp_sub_pel_8(block_x, block_y, block_width, block_height,
                                    ldsp_mv[0].phi, ldsp_mv[0].theta, ref_frame,
                                    frame_stride, frame_width, frame_height,
                                    pred_block_stride, pred_block);
  best_sad = get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               frame_stride, pred_block_stride);

  update_sphere_mv_ldsp(ldsp_mv, search_step_phi, search_step_theta);

  const double min_scale = 0.125;

  while (start_phi + ldsp_mv[5].phi <= max_range_phi &&
         start_phi + ldsp_mv[1].phi >= min_range_phi &&
         start_theta + ldsp_mv[3].theta <= max_range_theta &&
         start_theta + ldsp_mv[7].theta >= min_range_theta) {
    for (int i = 0; i < 9; i++) {
      av1_get_pred_erp_interp_sub_pel_8(
          block_x, block_y, block_width, block_height, ldsp_mv[i].phi,
          ldsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);

      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            frame_stride, pred_block_stride);

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv_idx = i;
      }
    }  // for

    if (best_mv_idx == 0) {
      if (search_step_phi > min_scale * PI / frame_height &&
          search_step_theta > min_scale * 2 * PI / frame_height) {
        search_step_phi *= 0.5;
        search_step_theta *= 0.5;
      } else {
        break;
      }
    } else {
      ldsp_mv[0].phi = ldsp_mv[best_mv_idx].phi;
      ldsp_mv[0].theta = ldsp_mv[best_mv_idx].theta;
      best_mv_idx = 0;
    }
    update_sphere_mv_ldsp(ldsp_mv, search_step_phi, search_step_theta);
  }

  sdsp_mv[0].phi = ldsp_mv[0].phi;
  sdsp_mv[0].theta = ldsp_mv[0].theta;
  best_mv_idx = 0;
  search_step_phi = min_scale * PI / frame_height;
  search_step_theta = min_scale * 2 * PI / frame_width;
  update_sphere_mv_sdsp(sdsp_mv, search_step_phi, search_step_theta);
  if (start_phi + sdsp_mv[3].phi <= max_range_phi &&
      start_phi + sdsp_mv[1].phi >= min_range_phi &&
      start_theta + sdsp_mv[2].theta <= max_range_theta &&
      start_theta + sdsp_mv[4].theta >= min_range_theta) {
    for (int i = 0; i < 5; i++) {
      av1_get_pred_erp_interp_sub_pel_8(
          block_x, block_y, block_width, block_height, sdsp_mv[i].phi,
          sdsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);

      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            frame_stride, pred_block_stride);

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv_idx = i;
      }
    }  // for
  }

  best_mv->phi = sdsp_mv[best_mv_idx].phi;
  best_mv->theta = sdsp_mv[best_mv_idx].theta;

  return best_sad;
}

int av1_motion_search_diamond_erp_interp_sub_pel_8sharp(
    int block_x, int block_y, int block_width, int block_height,
    const uint8_t *cur_frame, const uint8_t *ref_frame, int frame_stride,
    int frame_width, int frame_height, int search_range,
    const SphereMV *start_mv, SphereMV *best_mv) {
  assert(cur_frame != NULL && ref_frame != NULL && best_mv != NULL);
  assert(block_width > 0 && block_height > 0 && block_width <= 128 &&
         block_height <= 128 && block_x >= 0 && block_y >= 0 &&
         frame_width > 0 && frame_height > 0);
  assert(search_range > 0);

  // Large Diamond Search Pattern on shpere
  SphereMV ldsp_mv[9];
  // Small Diamond Search Pattern on shpere
  SphereMV sdsp_mv[5];

  double search_step_phi =
      AOMMIN(0.25 * block_height, 0.25 * search_range) * PI / frame_height;
  double search_step_theta =
      AOMMIN(0.25 * block_width, 0.25 * search_range) * 2 * PI / frame_width;

  // printf("Diamond ERP interp search step phi: %f, theta: %f\n",
  // search_step_phi,
  //        search_step_theta);

  const uint8_t *cur_block = &cur_frame[block_x + block_y * frame_stride];
  uint8_t pred_block[128 * 128];
  const int pred_block_stride = 128;

  double start_phi;
  double start_theta;
  av1_plane_to_sphere_erp(block_x, block_y, frame_width, frame_height,
                          &start_phi, &start_theta);

  double max_range_phi =
      start_phi + start_mv->phi + search_range * PI / frame_height;
  double min_range_phi =
      start_phi + start_mv->theta - search_range * PI / frame_height;
  double max_range_theta = start_theta + search_range * 2 * PI / frame_height;
  double min_range_theta = start_theta - search_range * 2 * PI / frame_height;

  int best_mv_idx = 0;
  ldsp_mv[0].phi = start_mv->phi;
  ldsp_mv[0].theta = start_mv->theta;

  int temp_sad;
  int best_sad;
  av1_get_pred_erp_interp_sub_pel_8sharp(
      block_x, block_y, block_width, block_height, ldsp_mv[0].phi,
      ldsp_mv[0].theta, ref_frame, frame_stride, frame_width, frame_height,
      pred_block_stride, pred_block);
  best_sad = get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               frame_stride, pred_block_stride);

  update_sphere_mv_ldsp(ldsp_mv, search_step_phi, search_step_theta);

  const double min_scale = 0.125;

  while (start_phi + ldsp_mv[5].phi <= max_range_phi &&
         start_phi + ldsp_mv[1].phi >= min_range_phi &&
         start_theta + ldsp_mv[3].theta <= max_range_theta &&
         start_theta + ldsp_mv[7].theta >= min_range_theta) {
    for (int i = 0; i < 9; i++) {
      av1_get_pred_erp_interp_sub_pel_8sharp(
          block_x, block_y, block_width, block_height, ldsp_mv[i].phi,
          ldsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);

      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            frame_stride, pred_block_stride);

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv_idx = i;
      }
    }  // for

    if (best_mv_idx == 0) {
      if (search_step_phi > min_scale * PI / frame_height &&
          search_step_theta > min_scale * 2 * PI / frame_height) {
        search_step_phi *= 0.5;
        search_step_theta *= 0.5;
      } else {
        break;
      }
    } else {
      ldsp_mv[0].phi = ldsp_mv[best_mv_idx].phi;
      ldsp_mv[0].theta = ldsp_mv[best_mv_idx].theta;
      best_mv_idx = 0;
    }
    update_sphere_mv_ldsp(ldsp_mv, search_step_phi, search_step_theta);
  }

  sdsp_mv[0].phi = ldsp_mv[0].phi;
  sdsp_mv[0].theta = ldsp_mv[0].theta;
  best_mv_idx = 0;
  search_step_phi = min_scale * PI / frame_height;
  search_step_theta = min_scale * 2 * PI / frame_width;
  update_sphere_mv_sdsp(sdsp_mv, search_step_phi, search_step_theta);
  if (start_phi + sdsp_mv[3].phi <= max_range_phi &&
      start_phi + sdsp_mv[1].phi >= min_range_phi &&
      start_theta + sdsp_mv[2].theta <= max_range_theta &&
      start_theta + sdsp_mv[4].theta >= min_range_theta) {
    for (int i = 0; i < 5; i++) {
      av1_get_pred_erp_interp_sub_pel_8sharp(
          block_x, block_y, block_width, block_height, sdsp_mv[i].phi,
          sdsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);

      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            frame_stride, pred_block_stride);

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv_idx = i;
      }
    }  // for
  }

  best_mv->phi = sdsp_mv[best_mv_idx].phi;
  best_mv->theta = sdsp_mv[best_mv_idx].theta;

  return best_sad;
}

int av1_motion_search_diamond_erp_interp_sub_pel_8smooth(
    int block_x, int block_y, int block_width, int block_height,
    const uint8_t *cur_frame, const uint8_t *ref_frame, int frame_stride,
    int frame_width, int frame_height, int search_range,
    const SphereMV *start_mv, SphereMV *best_mv) {
  assert(cur_frame != NULL && ref_frame != NULL && best_mv != NULL);
  assert(block_width > 0 && block_height > 0 && block_width <= 128 &&
         block_height <= 128 && block_x >= 0 && block_y >= 0 &&
         frame_width > 0 && frame_height > 0);
  assert(search_range > 0);

  // Large Diamond Search Pattern on shpere
  SphereMV ldsp_mv[9];
  // Small Diamond Search Pattern on shpere
  SphereMV sdsp_mv[5];

  double search_step_phi =
      AOMMIN(0.25 * block_height, 0.25 * search_range) * PI / frame_height;
  double search_step_theta =
      AOMMIN(0.25 * block_width, 0.25 * search_range) * 2 * PI / frame_width;

  // printf("Diamond ERP interp search step phi: %f, theta: %f\n",
  // search_step_phi,
  //        search_step_theta);

  const uint8_t *cur_block = &cur_frame[block_x + block_y * frame_stride];
  uint8_t pred_block[128 * 128];
  const int pred_block_stride = 128;

  double start_phi;
  double start_theta;
  av1_plane_to_sphere_erp(block_x, block_y, frame_width, frame_height,
                          &start_phi, &start_theta);

  double max_range_phi =
      start_phi + start_mv->phi + search_range * PI / frame_height;
  double min_range_phi =
      start_phi + start_mv->theta - search_range * PI / frame_height;
  double max_range_theta = start_theta + search_range * 2 * PI / frame_height;
  double min_range_theta = start_theta - search_range * 2 * PI / frame_height;

  int best_mv_idx = 0;
  ldsp_mv[0].phi = start_mv->phi;
  ldsp_mv[0].theta = start_mv->theta;

  int temp_sad;
  int best_sad;
  av1_get_pred_erp_interp_sub_pel_8smooth(
      block_x, block_y, block_width, block_height, ldsp_mv[0].phi,
      ldsp_mv[0].theta, ref_frame, frame_stride, frame_width, frame_height,
      pred_block_stride, pred_block);
  best_sad = get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               frame_stride, pred_block_stride);

  update_sphere_mv_ldsp(ldsp_mv, search_step_phi, search_step_theta);

  const double min_scale = 0.125;

  while (start_phi + ldsp_mv[5].phi <= max_range_phi &&
         start_phi + ldsp_mv[1].phi >= min_range_phi &&
         start_theta + ldsp_mv[3].theta <= max_range_theta &&
         start_theta + ldsp_mv[7].theta >= min_range_theta) {
    for (int i = 0; i < 9; i++) {
      av1_get_pred_erp_interp_sub_pel_8smooth(
          block_x, block_y, block_width, block_height, ldsp_mv[i].phi,
          ldsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);

      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            frame_stride, pred_block_stride);

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv_idx = i;
      }
    }  // for

    if (best_mv_idx == 0) {
      if (search_step_phi > min_scale * PI / frame_height &&
          search_step_theta > min_scale * 2 * PI / frame_height) {
        search_step_phi *= 0.5;
        search_step_theta *= 0.5;
      } else {
        break;
      }
    } else {
      ldsp_mv[0].phi = ldsp_mv[best_mv_idx].phi;
      ldsp_mv[0].theta = ldsp_mv[best_mv_idx].theta;
      best_mv_idx = 0;
    }
    update_sphere_mv_ldsp(ldsp_mv, search_step_phi, search_step_theta);
  }

  sdsp_mv[0].phi = ldsp_mv[0].phi;
  sdsp_mv[0].theta = ldsp_mv[0].theta;
  best_mv_idx = 0;
  search_step_phi = min_scale * PI / frame_height;
  search_step_theta = min_scale * 2 * PI / frame_width;
  update_sphere_mv_sdsp(sdsp_mv, search_step_phi, search_step_theta);
  if (start_phi + sdsp_mv[3].phi <= max_range_phi &&
      start_phi + sdsp_mv[1].phi >= min_range_phi &&
      start_theta + sdsp_mv[2].theta <= max_range_theta &&
      start_theta + sdsp_mv[4].theta >= min_range_theta) {
    for (int i = 0; i < 5; i++) {
      av1_get_pred_erp_interp_sub_pel_8smooth(
          block_x, block_y, block_width, block_height, sdsp_mv[i].phi,
          sdsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);

      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            frame_stride, pred_block_stride);

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv_idx = i;
      }
    }  // for
  }

  best_mv->phi = sdsp_mv[best_mv_idx].phi;
  best_mv->theta = sdsp_mv[best_mv_idx].theta;

  return best_sad;
}

int av1_motion_search_diamond_erp_interp_filter_search(
    int block_x, int block_y, int block_width, int block_height,
    const uint8_t *cur_frame, const uint8_t *ref_frame, int frame_stride,
    int frame_width, int frame_height, int search_range,
    const SphereMV *start_mv, SphereMV *best_mv) {
  assert(cur_frame != NULL && ref_frame != NULL && best_mv != NULL);
  assert(block_width > 0 && block_height > 0 && block_width <= 128 &&
         block_height <= 128 && block_x >= 0 && block_y >= 0 &&
         frame_width > 0 && frame_height > 0);
  assert(search_range > 0);

  // Large Diamond Search Pattern on shpere
  SphereMV ldsp_mv[9];
  // Small Diamond Search Pattern on shpere
  SphereMV sdsp_mv[5];

  double search_step_phi =
      AOMMIN(0.25 * block_height, 0.25 * search_range) * PI / frame_height;
  double search_step_theta =
      AOMMIN(0.25 * block_width, 0.25 * search_range) * 2 * PI / frame_width;

  // printf("Diamond ERP interp search step phi: %f, theta: %f\n",
  // search_step_phi,
  //        search_step_theta);

  const uint8_t *cur_block = &cur_frame[block_x + block_y * frame_stride];
  uint8_t pred_block[128 * 128];
  const int pred_block_stride = 128;

  double start_phi;
  double start_theta;
  av1_plane_to_sphere_erp(block_x, block_y, frame_width, frame_height,
                          &start_phi, &start_theta);

  double max_range_phi =
      start_phi + start_mv->phi + search_range * PI / frame_height;
  double min_range_phi =
      start_phi + start_mv->theta - search_range * PI / frame_height;
  double max_range_theta = start_theta + search_range * 2 * PI / frame_height;
  double min_range_theta = start_theta - search_range * 2 * PI / frame_height;

  int best_mv_idx = 0;
  ldsp_mv[0].phi = start_mv->phi;
  ldsp_mv[0].theta = start_mv->theta;

  int temp_sad;
  int best_sad;
  av1_get_pred_erp_interp_bilinear(block_x, block_y, block_width, block_height,
                                   ldsp_mv[0].phi, ldsp_mv[0].theta, ref_frame,
                                   frame_stride, frame_width, frame_height,
                                   pred_block_stride, pred_block);
  best_sad = get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               frame_stride, pred_block_stride);

  av1_get_pred_erp_interp_sub_pel_8(block_x, block_y, block_width, block_height,
                                    ldsp_mv[0].phi, ldsp_mv[0].theta, ref_frame,
                                    frame_stride, frame_width, frame_height,
                                    pred_block_stride, pred_block);
  best_sad =
      AOMMIN(best_sad,
             get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               frame_stride, pred_block_stride));

  av1_get_pred_erp_interp_sub_pel_8sharp(
      block_x, block_y, block_width, block_height, ldsp_mv[0].phi,
      ldsp_mv[0].theta, ref_frame, frame_stride, frame_width, frame_height,
      pred_block_stride, pred_block);
  best_sad =
      AOMMIN(best_sad,
             get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               frame_stride, pred_block_stride));

  av1_get_pred_erp_interp_sub_pel_8smooth(
      block_x, block_y, block_width, block_height, ldsp_mv[0].phi,
      ldsp_mv[0].theta, ref_frame, frame_stride, frame_width, frame_height,
      pred_block_stride, pred_block);
  best_sad =
      AOMMIN(best_sad,
             get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               frame_stride, pred_block_stride));

  update_sphere_mv_ldsp(ldsp_mv, search_step_phi, search_step_theta);

  const double min_scale = 0.125;

  while (start_phi + ldsp_mv[5].phi <= max_range_phi &&
         start_phi + ldsp_mv[1].phi >= min_range_phi &&
         start_theta + ldsp_mv[3].theta <= max_range_theta &&
         start_theta + ldsp_mv[7].theta >= min_range_theta) {
    for (int i = 0; i < 9; i++) {
      av1_get_pred_erp_interp_bilinear(
          block_x, block_y, block_width, block_height, ldsp_mv[i].phi,
          ldsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);
      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            frame_stride, pred_block_stride);

      av1_get_pred_erp_interp_sub_pel_8(
          block_x, block_y, block_width, block_height, ldsp_mv[i].phi,
          ldsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);
      temp_sad =
          AOMMIN(temp_sad, get_sad_of_blocks(cur_block, pred_block, block_width,
                                             block_height, frame_stride,
                                             pred_block_stride));

      av1_get_pred_erp_interp_sub_pel_8sharp(
          block_x, block_y, block_width, block_height, ldsp_mv[i].phi,
          ldsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);
      temp_sad =
          AOMMIN(temp_sad, get_sad_of_blocks(cur_block, pred_block, block_width,
                                             block_height, frame_stride,
                                             pred_block_stride));

      av1_get_pred_erp_interp_sub_pel_8smooth(
          block_x, block_y, block_width, block_height, ldsp_mv[i].phi,
          ldsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);
      temp_sad =
          AOMMIN(temp_sad, get_sad_of_blocks(cur_block, pred_block, block_width,
                                             block_height, frame_stride,
                                             pred_block_stride));

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv_idx = i;
      }
    }  // for

    if (best_mv_idx == 0) {
      if (search_step_phi > min_scale * PI / frame_height &&
          search_step_theta > min_scale * 2 * PI / frame_height) {
        search_step_phi *= 0.5;
        search_step_theta *= 0.5;
      } else {
        break;
      }
    } else {
      ldsp_mv[0].phi = ldsp_mv[best_mv_idx].phi;
      ldsp_mv[0].theta = ldsp_mv[best_mv_idx].theta;
      best_mv_idx = 0;
    }
    update_sphere_mv_ldsp(ldsp_mv, search_step_phi, search_step_theta);
  }

  sdsp_mv[0].phi = ldsp_mv[0].phi;
  sdsp_mv[0].theta = ldsp_mv[0].theta;
  best_mv_idx = 0;
  search_step_phi = min_scale * PI / frame_height;
  search_step_theta = min_scale * 2 * PI / frame_width;
  update_sphere_mv_sdsp(sdsp_mv, search_step_phi, search_step_theta);
  if (start_phi + sdsp_mv[3].phi <= max_range_phi &&
      start_phi + sdsp_mv[1].phi >= min_range_phi &&
      start_theta + sdsp_mv[2].theta <= max_range_theta &&
      start_theta + sdsp_mv[4].theta >= min_range_theta) {
    for (int i = 0; i < 5; i++) {
      av1_get_pred_erp_interp_bilinear(
          block_x, block_y, block_width, block_height, sdsp_mv[i].phi,
          sdsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);
      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            frame_stride, pred_block_stride);

      av1_get_pred_erp_interp_sub_pel_8(
          block_x, block_y, block_width, block_height, sdsp_mv[i].phi,
          sdsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);
      temp_sad =
          AOMMIN(temp_sad, get_sad_of_blocks(cur_block, pred_block, block_width,
                                             block_height, frame_stride,
                                             pred_block_stride));

      av1_get_pred_erp_interp_sub_pel_8sharp(
          block_x, block_y, block_width, block_height, sdsp_mv[i].phi,
          sdsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);
      temp_sad =
          AOMMIN(temp_sad, get_sad_of_blocks(cur_block, pred_block, block_width,
                                             block_height, frame_stride,
                                             pred_block_stride));

      av1_get_pred_erp_interp_sub_pel_8smooth(
          block_x, block_y, block_width, block_height, sdsp_mv[i].phi,
          sdsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);
      temp_sad =
          AOMMIN(temp_sad, get_sad_of_blocks(cur_block, pred_block, block_width,
                                             block_height, frame_stride,
                                             pred_block_stride));

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv_idx = i;
      }
    }  // for
  }

  best_mv->phi = sdsp_mv[best_mv_idx].phi;
  best_mv->theta = sdsp_mv[best_mv_idx].theta;

  return best_sad;
}

void av1_get_pred_erp(int block_x, int block_y, int block_width,
                      int block_height, double delta_phi, double delta_theta,
                      const uint8_t *ref_frame, int ref_frame_stride,
                      int frame_width, int frame_height, int pred_block_stride,
                      uint8_t *pred_block) {
  assert(ref_frame != NULL && frame_width > 0 && frame_height > 0 &&
         ref_frame_stride >= frame_width);
  assert(pred_block != NULL && block_width > 0 && block_height > 0 &&
         pred_block_stride >= block_width);

  int pos_pred;      // Offset in pred_block buffer
  int pos_ref;       // Offset in ref_frame buffer
  double x_current;  // X coordiante in current frame
  double y_current;  // Y coordiante in currrent frame
  double x_ref;      // X coordinate in reference frame
  double y_ref;      // Y coordiante in reference frame

  PolarVector block_polar;
  CartesianVector block_v;
  CartesianVector block_v_prime;
  CartesianVector k;  // The axis that the block will rotate along
  block_polar.r = 1.0;
  av1_plane_to_sphere_erp(block_x, block_y, frame_width, frame_height,
                          &block_polar.phi, &block_polar.theta);
  block_polar.phi += 0.5 * PI;
  av1_sphere_polar_to_carte(&block_polar, &block_v);

  block_polar.phi += delta_phi;
  block_polar.theta += delta_theta;
  av1_sphere_polar_to_carte(&block_polar, &block_v_prime);

  av1_carte_vectors_cross_product(&block_v, &block_v_prime, &k);
  av1_normalize_carte_vector(&k);

  double product = av1_carte_vectors_dot_product(&block_v, &block_v_prime);
  // Avoid floating point precision issues
  product = AOMMAX(product, -1.0);
  product = AOMMIN(product, 1.0);
  double alpha = acos(product);

  CartesianVector v;
  CartesianVector v_prime;
  PolarVector v_polar;
  PolarVector v_prime_polar;
  v_polar.r = 1.0;
  v_prime_polar.r = 1.0;
  double k_dot_v;

  for (int idx_y = 0; idx_y < block_height; idx_y++) {
    for (int idx_x = 0; idx_x < block_width; idx_x++) {
      x_current = idx_x + block_x;
      y_current = idx_y + block_y;

      av1_plane_to_sphere_erp(x_current, y_current, frame_width, frame_height,
                              &v_polar.phi, &v_polar.theta);
      v_polar.phi += 0.5 * PI;
      av1_sphere_polar_to_carte(&v_polar, &v);
      k_dot_v = av1_carte_vectors_dot_product(&k, &v);

      v_prime.x = k.x * k_dot_v * (1 - cos(alpha)) + v.x * cos(alpha) +
                  (k.y * v.z - k.z * v.y) * sin(alpha);
      v_prime.y = k.y * k_dot_v * (1 - cos(alpha)) + v.y * cos(alpha) +
                  (k.z * v.x - k.x * v.z) * sin(alpha);
      v_prime.z = k.z * k_dot_v * (1 - cos(alpha)) + v.z * cos(alpha) +
                  (k.x * v.y - k.y * v.x) * sin(alpha);

      av1_sphere_carte_to_polar(&v_prime, &v_prime_polar);
      v_prime_polar.phi -= 0.5 * PI;
      av1_sphere_to_plane_erp(v_prime_polar.phi, v_prime_polar.theta,
                              frame_width, frame_height, &x_ref, &y_ref);

      x_ref = round(x_ref);
      x_ref = (int)x_ref == frame_width ? 0 : x_ref;
      y_ref = round(y_ref);
      y_ref = (int)y_ref == frame_height ? frame_height - 1 : y_ref;

      pos_pred = idx_x + idx_y * pred_block_stride;
      pos_ref = (int)x_ref + (int)y_ref * ref_frame_stride;
      pred_block[pos_pred] = ref_frame[pos_ref];
    }
  }
}

/* Calculate the scale of theta based on the current phi.
 * Makes the search/filter pattern on the latitude sparse as the position moves
 * toward polars. Will be used for adjusting searching step, searching range,
 * and filtering.
 */
static double cal_theta_scale(double phi) {
  const double max_scale = 10.0;

  if (cos(phi) > 0.00001) {
    return AOMMIN(max_scale, 1 / cos(phi));
  } else {
    return max_scale;
  }
}

static double cal_range_theta(int search_range, int frame_width, double phi) {
  return search_range * cal_theta_scale(phi) * 2 * PI / frame_width;
}

/* Update Large Diamond Search Pattern shperical motion vectors based on the
 * center
 *                    ldsp_mv[1]
 *                    /        \
 *          ldsp_mv[8]          ldsp_mv[2]
 *                 /              \
 *       ldsp_mv[7]   ldsp_mv[0]   ldsp_mv[3]
 *                 \              /
 *         ldsp_mv[6]            ldsp_mv[4]
 *                   \          /
 *                    ldsp_mv[5]
 */
static void update_sphere_mv_ldsp_interp_scale(SphereMV ldsp_mv[9],
                                               double search_step_phi,
                                               double search_step_theta,
                                               double block_phi) {
  double magnified_step_theta;

  ldsp_mv[1].phi = ldsp_mv[0].phi - 2 * search_step_phi;
  ldsp_mv[1].theta = ldsp_mv[0].theta;

  ldsp_mv[2].phi = ldsp_mv[0].phi - search_step_phi;
  magnified_step_theta =
      search_step_theta * cal_theta_scale(block_phi + ldsp_mv[2].phi);
  ldsp_mv[2].theta = ldsp_mv[0].theta + magnified_step_theta;

  ldsp_mv[3].phi = ldsp_mv[0].phi;
  magnified_step_theta =
      search_step_theta * cal_theta_scale(block_phi + ldsp_mv[3].phi);
  ldsp_mv[3].theta = ldsp_mv[0].theta + 2 * magnified_step_theta;

  ldsp_mv[4].phi = ldsp_mv[0].phi + search_step_phi;
  magnified_step_theta =
      search_step_theta * cal_theta_scale(block_phi + ldsp_mv[4].phi);
  ldsp_mv[4].theta = ldsp_mv[0].theta + magnified_step_theta;

  ldsp_mv[5].phi = ldsp_mv[0].phi + 2 * search_step_phi;
  ldsp_mv[5].theta = ldsp_mv[0].theta;

  ldsp_mv[6].phi = ldsp_mv[0].phi + search_step_phi;
  magnified_step_theta =
      search_step_theta * cal_theta_scale(block_phi + ldsp_mv[6].phi);
  ldsp_mv[6].theta = ldsp_mv[0].theta - magnified_step_theta;

  ldsp_mv[7].phi = ldsp_mv[0].phi;
  magnified_step_theta =
      search_step_theta * cal_theta_scale(block_phi + ldsp_mv[7].phi);
  ldsp_mv[7].theta = ldsp_mv[0].theta - 2 * magnified_step_theta;

  ldsp_mv[8].phi = ldsp_mv[0].phi - search_step_phi;
  magnified_step_theta =
      search_step_theta * cal_theta_scale(block_phi + ldsp_mv[8].phi);
  ldsp_mv[8].theta = ldsp_mv[0].theta - magnified_step_theta;
}

/* Small Diamond Search Pattern on shpere
 *                     sdsp_mv[1]
 *                  /              \
 *         sdsp_mv[4]  sdsp_mv[0]  sdsp_mv[2]
 *                  \              /
 *                     sdsp_mv[3]
 */
static void update_sphere_mv_sdsp_interp_scale(SphereMV sdsp_mv[5],
                                               double search_step_phi,
                                               double search_step_theta,
                                               double block_phi) {
  double magnified_step_theta;

  sdsp_mv[1].phi = sdsp_mv[0].phi - search_step_phi;
  sdsp_mv[1].theta = sdsp_mv[0].theta;

  sdsp_mv[2].phi = sdsp_mv[0].phi;
  magnified_step_theta =
      search_step_theta * cal_theta_scale(block_phi + sdsp_mv[2].phi);
  sdsp_mv[2].theta = sdsp_mv[0].theta + magnified_step_theta;

  sdsp_mv[3].phi = sdsp_mv[0].phi + search_step_phi;
  sdsp_mv[3].theta = sdsp_mv[0].theta;

  sdsp_mv[4].phi = sdsp_mv[0].phi;
  magnified_step_theta =
      search_step_theta * cal_theta_scale(block_phi + sdsp_mv[4].phi);
  sdsp_mv[4].theta = sdsp_mv[0].theta - magnified_step_theta;
}

int av1_motion_search_diamond_erp_interp_bilinear_scale(
    int block_x, int block_y, int block_width, int block_height,
    const uint8_t *cur_frame, const uint8_t *ref_frame, int frame_stride,
    int frame_width, int frame_height, int search_range,
    const SphereMV *start_mv, SphereMV *best_mv) {
  assert(cur_frame != NULL && ref_frame != NULL && best_mv != NULL);
  assert(block_width > 0 && block_height > 0 && block_width <= 128 &&
         block_height <= 128 && block_x >= 0 && block_y >= 0 &&
         frame_width > 0 && frame_height > 0);
  assert(search_range > 0);

  // Large Diamond Search Pattern on shpere
  SphereMV ldsp_mv[9];
  // Small Diamond Search Pattern on shpere
  SphereMV sdsp_mv[5];

  double search_step_phi =
      AOMMIN(0.25 * block_height, 0.25 * search_range) * PI / frame_height;
  double search_step_theta =
      AOMMIN(0.25 * block_width, 0.25 * search_range) * 2 * PI / frame_width;

  const uint8_t *cur_block = &cur_frame[block_x + block_y * frame_stride];
  uint8_t pred_block[128 * 128];
  const int pred_block_stride = 128;

  double start_phi;
  double start_theta;
  av1_plane_to_sphere_erp(block_x, block_y, frame_width, frame_height,
                          &start_phi, &start_theta);

  double max_range_phi =
      start_phi + start_mv->phi + search_range * PI / frame_height;
  double min_range_phi =
      start_phi + start_mv->phi - search_range * PI / frame_height;

  int best_mv_idx = 0;
  ldsp_mv[0].phi = start_mv->phi;
  ldsp_mv[0].theta = start_mv->theta;

  int temp_sad;
  int best_sad;
  av1_get_pred_erp_interp_bilinear(block_x, block_y, block_width, block_height,
                                   ldsp_mv[0].phi, ldsp_mv[0].theta, ref_frame,
                                   frame_stride, frame_width, frame_height,
                                   pred_block_stride, pred_block);
  best_sad = get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               frame_stride, pred_block_stride);
  update_sphere_mv_ldsp_interp_scale(ldsp_mv, search_step_phi,
                                     search_step_theta, start_phi);

  const double min_scale = 0.125;

  while (start_phi + ldsp_mv[5].phi <= max_range_phi &&
         start_phi + ldsp_mv[1].phi >= min_range_phi &&
         start_theta + ldsp_mv[3].theta <=
             start_theta + start_mv->theta +
                 cal_range_theta(search_range, frame_width,
                                 start_phi + ldsp_mv[3].phi) &&
         start_theta + ldsp_mv[7].theta >=
             start_theta + start_mv->theta -
                 cal_range_theta(search_range, frame_width,
                                 start_phi + ldsp_mv[7].phi)) {
    for (int i = 0; i < 9; i++) {
      av1_get_pred_erp_interp_bilinear(
          block_x, block_y, block_width, block_height, ldsp_mv[i].phi,
          ldsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);

      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            frame_stride, pred_block_stride);

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv_idx = i;
      }
    }  // for

    if (best_mv_idx == 0) {
      if (search_step_phi > min_scale * PI / frame_height &&
          search_step_theta > min_scale * 2 * PI / frame_height) {
        search_step_phi *= 0.5;
        search_step_theta *= 0.5;
      } else {
        break;
      }
    } else {
      ldsp_mv[0].phi = ldsp_mv[best_mv_idx].phi;
      ldsp_mv[0].theta = ldsp_mv[best_mv_idx].theta;
      best_mv_idx = 0;
    }
    update_sphere_mv_ldsp_interp_scale(ldsp_mv, search_step_phi,
                                       search_step_theta, start_phi);
  }

  sdsp_mv[0].phi = ldsp_mv[0].phi;
  sdsp_mv[0].theta = ldsp_mv[0].theta;
  best_mv_idx = 0;
  search_step_phi = min_scale * PI / frame_height;
  search_step_theta = min_scale * 2 * PI / frame_width;
  update_sphere_mv_sdsp_interp_scale(sdsp_mv, search_step_phi,
                                     search_step_theta, start_phi);
  if (start_phi + sdsp_mv[3].phi <= max_range_phi &&
      start_phi + sdsp_mv[1].phi >= min_range_phi &&
      start_theta + sdsp_mv[2].theta <=
          start_theta + start_mv->theta +
              cal_range_theta(search_range, frame_width,
                              start_phi + sdsp_mv[2].phi) &&
      start_theta + sdsp_mv[4].theta >=
          start_theta + start_mv->theta +
              cal_range_theta(search_range, frame_width,
                              start_phi + sdsp_mv[4].phi)) {
    for (int i = 0; i < 5; i++) {
      av1_get_pred_erp_interp_bilinear(
          block_x, block_y, block_width, block_height, sdsp_mv[i].phi,
          sdsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);

      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            frame_stride, pred_block_stride);

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv_idx = i;
      }
    }  // for
  }

  best_mv->phi = sdsp_mv[best_mv_idx].phi;
  best_mv->theta = sdsp_mv[best_mv_idx].theta;

  return best_sad;
}

int av1_motion_search_diamond_erp_interp_sub_pel_8_scale(
    int block_x, int block_y, int block_width, int block_height,
    const uint8_t *cur_frame, const uint8_t *ref_frame, int frame_stride,
    int frame_width, int frame_height, int search_range,
    const SphereMV *start_mv, SphereMV *best_mv) {
  assert(cur_frame != NULL && ref_frame != NULL && best_mv != NULL);
  assert(block_width > 0 && block_height > 0 && block_width <= 128 &&
         block_height <= 128 && block_x >= 0 && block_y >= 0 &&
         frame_width > 0 && frame_height > 0);
  assert(search_range > 0);

  // Large Diamond Search Pattern on shpere
  SphereMV ldsp_mv[9];
  // Small Diamond Search Pattern on shpere
  SphereMV sdsp_mv[5];

  double search_step_phi =
      AOMMIN(0.25 * block_height, 0.25 * search_range) * PI / frame_height;
  double search_step_theta =
      AOMMIN(0.25 * block_width, 0.25 * search_range) * 2 * PI / frame_width;

  const uint8_t *cur_block = &cur_frame[block_x + block_y * frame_stride];
  uint8_t pred_block[128 * 128];
  const int pred_block_stride = 128;

  double start_phi;
  double start_theta;
  av1_plane_to_sphere_erp(block_x, block_y, frame_width, frame_height,
                          &start_phi, &start_theta);

  double max_range_phi =
      start_phi + start_mv->phi + search_range * PI / frame_height;
  double min_range_phi =
      start_phi + start_mv->phi - search_range * PI / frame_height;

  int best_mv_idx = 0;
  ldsp_mv[0].phi = start_mv->phi;
  ldsp_mv[0].theta = start_mv->theta;

  int temp_sad;
  int best_sad;
  av1_get_pred_erp_interp_sub_pel_8(block_x, block_y, block_width, block_height,
                                    ldsp_mv[0].phi, ldsp_mv[0].theta, ref_frame,
                                    frame_stride, frame_width, frame_height,
                                    pred_block_stride, pred_block);
  best_sad = get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               frame_stride, pred_block_stride);
  update_sphere_mv_ldsp_interp_scale(ldsp_mv, search_step_phi,
                                     search_step_theta, start_phi);

  const double min_scale = 0.125;

  while (start_phi + ldsp_mv[5].phi <= max_range_phi &&
         start_phi + ldsp_mv[1].phi >= min_range_phi &&
         start_theta + ldsp_mv[3].theta <=
             start_theta + start_mv->theta +
                 cal_range_theta(search_range, frame_width,
                                 start_phi + ldsp_mv[3].phi) &&
         start_theta + ldsp_mv[7].theta >=
             start_theta + start_mv->theta -
                 cal_range_theta(search_range, frame_width,
                                 start_phi + ldsp_mv[7].phi)) {
    for (int i = 0; i < 9; i++) {
      av1_get_pred_erp_interp_sub_pel_8(
          block_x, block_y, block_width, block_height, ldsp_mv[i].phi,
          ldsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);

      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            frame_stride, pred_block_stride);

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv_idx = i;
      }
    }  // for

    if (best_mv_idx == 0) {
      if (search_step_phi > min_scale * PI / frame_height &&
          search_step_theta > min_scale * 2 * PI / frame_height) {
        search_step_phi *= 0.5;
        search_step_theta *= 0.5;
      } else {
        break;
      }
    } else {
      ldsp_mv[0].phi = ldsp_mv[best_mv_idx].phi;
      ldsp_mv[0].theta = ldsp_mv[best_mv_idx].theta;
      best_mv_idx = 0;
    }
    update_sphere_mv_ldsp_interp_scale(ldsp_mv, search_step_phi,
                                       search_step_theta, start_phi);
  }

  sdsp_mv[0].phi = ldsp_mv[0].phi;
  sdsp_mv[0].theta = ldsp_mv[0].theta;
  best_mv_idx = 0;
  search_step_phi = min_scale * PI / frame_height;
  search_step_theta = min_scale * 2 * PI / frame_width;
  update_sphere_mv_sdsp_interp_scale(sdsp_mv, search_step_phi,
                                     search_step_theta, start_phi);
  if (start_phi + sdsp_mv[3].phi <= max_range_phi &&
      start_phi + sdsp_mv[1].phi >= min_range_phi &&
      start_theta + sdsp_mv[2].theta <=
          start_theta + start_mv->theta +
              cal_range_theta(search_range, frame_width,
                              start_phi + sdsp_mv[2].phi) &&
      start_theta + sdsp_mv[4].theta >=
          start_theta + start_mv->theta +
              cal_range_theta(search_range, frame_width,
                              start_phi + sdsp_mv[4].phi)) {
    for (int i = 0; i < 5; i++) {
      av1_get_pred_erp_interp_sub_pel_8(
          block_x, block_y, block_width, block_height, sdsp_mv[i].phi,
          sdsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);

      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            frame_stride, pred_block_stride);

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv_idx = i;
      }
    }  // for
  }

  best_mv->phi = sdsp_mv[best_mv_idx].phi;
  best_mv->theta = sdsp_mv[best_mv_idx].theta;

  return best_sad;
}

int av1_motion_search_diamond_erp_interp_sub_pel_8sharp_scale(
    int block_x, int block_y, int block_width, int block_height,
    const uint8_t *cur_frame, const uint8_t *ref_frame, int frame_stride,
    int frame_width, int frame_height, int search_range,
    const SphereMV *start_mv, SphereMV *best_mv) {
  assert(cur_frame != NULL && ref_frame != NULL && best_mv != NULL);
  assert(block_width > 0 && block_height > 0 && block_width <= 128 &&
         block_height <= 128 && block_x >= 0 && block_y >= 0 &&
         frame_width > 0 && frame_height > 0);
  assert(search_range > 0);

  // Large Diamond Search Pattern on shpere
  SphereMV ldsp_mv[9];
  // Small Diamond Search Pattern on shpere
  SphereMV sdsp_mv[5];

  double search_step_phi =
      AOMMIN(0.25 * block_height, 0.25 * search_range) * PI / frame_height;
  double search_step_theta =
      AOMMIN(0.25 * block_width, 0.25 * search_range) * 2 * PI / frame_width;

  const uint8_t *cur_block = &cur_frame[block_x + block_y * frame_stride];
  uint8_t pred_block[128 * 128];
  const int pred_block_stride = 128;

  double start_phi;
  double start_theta;
  av1_plane_to_sphere_erp(block_x, block_y, frame_width, frame_height,
                          &start_phi, &start_theta);

  double max_range_phi =
      start_phi + start_mv->phi + search_range * PI / frame_height;
  double min_range_phi =
      start_phi + start_mv->phi - search_range * PI / frame_height;

  int best_mv_idx = 0;
  ldsp_mv[0].phi = start_mv->phi;
  ldsp_mv[0].theta = start_mv->theta;

  int temp_sad;
  int best_sad;
  av1_get_pred_erp_interp_sub_pel_8sharp(
      block_x, block_y, block_width, block_height, ldsp_mv[0].phi,
      ldsp_mv[0].theta, ref_frame, frame_stride, frame_width, frame_height,
      pred_block_stride, pred_block);
  best_sad = get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               frame_stride, pred_block_stride);
  update_sphere_mv_ldsp_interp_scale(ldsp_mv, search_step_phi,
                                     search_step_theta, start_phi);

  const double min_scale = 0.125;

  while (start_phi + ldsp_mv[5].phi <= max_range_phi &&
         start_phi + ldsp_mv[1].phi >= min_range_phi &&
         start_theta + ldsp_mv[3].theta <=
             start_theta + start_mv->theta +
                 cal_range_theta(search_range, frame_width,
                                 start_phi + ldsp_mv[3].phi) &&
         start_theta + ldsp_mv[7].theta >=
             start_theta + start_mv->theta -
                 cal_range_theta(search_range, frame_width,
                                 start_phi + ldsp_mv[7].phi)) {
    for (int i = 0; i < 9; i++) {
      av1_get_pred_erp_interp_sub_pel_8sharp(
          block_x, block_y, block_width, block_height, ldsp_mv[i].phi,
          ldsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);

      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            frame_stride, pred_block_stride);

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv_idx = i;
      }
    }  // for

    if (best_mv_idx == 0) {
      if (search_step_phi > min_scale * PI / frame_height &&
          search_step_theta > min_scale * 2 * PI / frame_height) {
        search_step_phi *= 0.5;
        search_step_theta *= 0.5;
      } else {
        break;
      }
    } else {
      ldsp_mv[0].phi = ldsp_mv[best_mv_idx].phi;
      ldsp_mv[0].theta = ldsp_mv[best_mv_idx].theta;
      best_mv_idx = 0;
    }
    update_sphere_mv_ldsp_interp_scale(ldsp_mv, search_step_phi,
                                       search_step_theta, start_phi);
  }

  sdsp_mv[0].phi = ldsp_mv[0].phi;
  sdsp_mv[0].theta = ldsp_mv[0].theta;
  best_mv_idx = 0;
  search_step_phi = min_scale * PI / frame_height;
  search_step_theta = min_scale * 2 * PI / frame_width;
  update_sphere_mv_sdsp_interp_scale(sdsp_mv, search_step_phi,
                                     search_step_theta, start_phi);
  if (start_phi + sdsp_mv[3].phi <= max_range_phi &&
      start_phi + sdsp_mv[1].phi >= min_range_phi &&
      start_theta + sdsp_mv[2].theta <=
          start_theta + start_mv->theta +
              cal_range_theta(search_range, frame_width,
                              start_phi + sdsp_mv[2].phi) &&
      start_theta + sdsp_mv[4].theta >=
          start_theta + start_mv->theta +
              cal_range_theta(search_range, frame_width,
                              start_phi + sdsp_mv[4].phi)) {
    for (int i = 0; i < 5; i++) {
      av1_get_pred_erp_interp_sub_pel_8sharp(
          block_x, block_y, block_width, block_height, sdsp_mv[i].phi,
          sdsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);

      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            frame_stride, pred_block_stride);

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv_idx = i;
      }
    }  // for
  }

  best_mv->phi = sdsp_mv[best_mv_idx].phi;
  best_mv->theta = sdsp_mv[best_mv_idx].theta;

  return best_sad;
}

int av1_motion_search_diamond_erp_interp_sub_pel_8smooth_scale(
    int block_x, int block_y, int block_width, int block_height,
    const uint8_t *cur_frame, const uint8_t *ref_frame, int frame_stride,
    int frame_width, int frame_height, int search_range,
    const SphereMV *start_mv, SphereMV *best_mv) {
  assert(cur_frame != NULL && ref_frame != NULL && best_mv != NULL);
  assert(block_width > 0 && block_height > 0 && block_width <= 128 &&
         block_height <= 128 && block_x >= 0 && block_y >= 0 &&
         frame_width > 0 && frame_height > 0);
  assert(search_range > 0);

  // Large Diamond Search Pattern on shpere
  SphereMV ldsp_mv[9];
  // Small Diamond Search Pattern on shpere
  SphereMV sdsp_mv[5];

  double search_step_phi =
      AOMMIN(0.25 * block_height, 0.25 * search_range) * PI / frame_height;
  double search_step_theta =
      AOMMIN(0.25 * block_width, 0.25 * search_range) * 2 * PI / frame_width;

  const uint8_t *cur_block = &cur_frame[block_x + block_y * frame_stride];
  uint8_t pred_block[128 * 128];
  const int pred_block_stride = 128;

  double start_phi;
  double start_theta;
  av1_plane_to_sphere_erp(block_x, block_y, frame_width, frame_height,
                          &start_phi, &start_theta);

  double max_range_phi =
      start_phi + start_mv->phi + search_range * PI / frame_height;
  double min_range_phi =
      start_phi + start_mv->phi - search_range * PI / frame_height;

  int best_mv_idx = 0;
  ldsp_mv[0].phi = start_mv->phi;
  ldsp_mv[0].theta = start_mv->theta;

  int temp_sad;
  int best_sad;
  av1_get_pred_erp_interp_sub_pel_8sharp(
      block_x, block_y, block_width, block_height, ldsp_mv[0].phi,
      ldsp_mv[0].theta, ref_frame, frame_stride, frame_width, frame_height,
      pred_block_stride, pred_block);
  best_sad = get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               frame_stride, pred_block_stride);
  update_sphere_mv_ldsp_interp_scale(ldsp_mv, search_step_phi,
                                     search_step_theta, start_phi);

  const double min_scale = 0.125;

  while (start_phi + ldsp_mv[5].phi <= max_range_phi &&
         start_phi + ldsp_mv[1].phi >= min_range_phi &&
         start_theta + ldsp_mv[3].theta <=
             start_theta + start_mv->theta +
                 cal_range_theta(search_range, frame_width,
                                 start_phi + ldsp_mv[3].phi) &&
         start_theta + ldsp_mv[7].theta >=
             start_theta + start_mv->theta -
                 cal_range_theta(search_range, frame_width,
                                 start_phi + ldsp_mv[7].phi)) {
    for (int i = 0; i < 9; i++) {
      av1_get_pred_erp_interp_sub_pel_8smooth(
          block_x, block_y, block_width, block_height, ldsp_mv[i].phi,
          ldsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);

      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            frame_stride, pred_block_stride);

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv_idx = i;
      }
    }  // for

    if (best_mv_idx == 0) {
      if (search_step_phi > min_scale * PI / frame_height &&
          search_step_theta > min_scale * 2 * PI / frame_height) {
        search_step_phi *= 0.5;
        search_step_theta *= 0.5;
      } else {
        break;
      }
    } else {
      ldsp_mv[0].phi = ldsp_mv[best_mv_idx].phi;
      ldsp_mv[0].theta = ldsp_mv[best_mv_idx].theta;
      best_mv_idx = 0;
    }
    update_sphere_mv_ldsp_interp_scale(ldsp_mv, search_step_phi,
                                       search_step_theta, start_phi);
  }

  sdsp_mv[0].phi = ldsp_mv[0].phi;
  sdsp_mv[0].theta = ldsp_mv[0].theta;
  best_mv_idx = 0;
  search_step_phi = min_scale * PI / frame_height;
  search_step_theta = min_scale * 2 * PI / frame_width;
  update_sphere_mv_sdsp_interp_scale(sdsp_mv, search_step_phi,
                                     search_step_theta, start_phi);
  if (start_phi + sdsp_mv[3].phi <= max_range_phi &&
      start_phi + sdsp_mv[1].phi >= min_range_phi &&
      start_theta + sdsp_mv[2].theta <=
          start_theta + start_mv->theta +
              cal_range_theta(search_range, frame_width,
                              start_phi + sdsp_mv[2].phi) &&
      start_theta + sdsp_mv[4].theta >=
          start_theta + start_mv->theta +
              cal_range_theta(search_range, frame_width,
                              start_phi + sdsp_mv[4].phi)) {
    for (int i = 0; i < 5; i++) {
      av1_get_pred_erp_interp_sub_pel_8smooth(
          block_x, block_y, block_width, block_height, sdsp_mv[i].phi,
          sdsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);

      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            frame_stride, pred_block_stride);

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv_idx = i;
      }
    }  // for
  }

  best_mv->phi = sdsp_mv[best_mv_idx].phi;
  best_mv->theta = sdsp_mv[best_mv_idx].theta;

  return best_sad;
}

int av1_motion_search_diamond_erp_interp_filter_search_scale(
    int block_x, int block_y, int block_width, int block_height,
    const uint8_t *cur_frame, const uint8_t *ref_frame, int frame_stride,
    int frame_width, int frame_height, int search_range,
    const SphereMV *start_mv, SphereMV *best_mv) {
  assert(cur_frame != NULL && ref_frame != NULL && best_mv != NULL);
  assert(block_width > 0 && block_height > 0 && block_width <= 128 &&
         block_height <= 128 && block_x >= 0 && block_y >= 0 &&
         frame_width > 0 && frame_height > 0);
  assert(search_range > 0);

  // Large Diamond Search Pattern on shpere
  SphereMV ldsp_mv[9];
  // Small Diamond Search Pattern on shpere
  SphereMV sdsp_mv[5];

  double search_step_phi =
      AOMMIN(0.25 * block_height, 0.25 * search_range) * PI / frame_height;
  double search_step_theta =
      AOMMIN(0.25 * block_width, 0.25 * search_range) * 2 * PI / frame_width;

  const uint8_t *cur_block = &cur_frame[block_x + block_y * frame_stride];
  uint8_t pred_block[128 * 128];
  const int pred_block_stride = 128;

  double start_phi;
  double start_theta;
  av1_plane_to_sphere_erp(block_x, block_y, frame_width, frame_height,
                          &start_phi, &start_theta);

  double max_range_phi =
      start_phi + start_mv->phi + search_range * PI / frame_height;
  double min_range_phi =
      start_phi + start_mv->phi - search_range * PI / frame_height;

  int best_mv_idx = 0;
  ldsp_mv[0].phi = start_mv->phi;
  ldsp_mv[0].theta = start_mv->theta;

  int temp_sad;
  int best_sad;
  av1_get_pred_erp_interp_bilinear(block_x, block_y, block_width, block_height,
                                   ldsp_mv[0].phi, ldsp_mv[0].theta, ref_frame,
                                   frame_stride, frame_width, frame_height,
                                   pred_block_stride, pred_block);
  best_sad = get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               frame_stride, pred_block_stride);

  av1_get_pred_erp_interp_sub_pel_8(block_x, block_y, block_width, block_height,
                                    ldsp_mv[0].phi, ldsp_mv[0].theta, ref_frame,
                                    frame_stride, frame_width, frame_height,
                                    pred_block_stride, pred_block);
  best_sad =
      AOMMIN(best_sad,
             get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               frame_stride, pred_block_stride));

  av1_get_pred_erp_interp_sub_pel_8sharp(
      block_x, block_y, block_width, block_height, ldsp_mv[0].phi,
      ldsp_mv[0].theta, ref_frame, frame_stride, frame_width, frame_height,
      pred_block_stride, pred_block);
  best_sad =
      AOMMIN(best_sad,
             get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               frame_stride, pred_block_stride));

  av1_get_pred_erp_interp_sub_pel_8smooth(
      block_x, block_y, block_width, block_height, ldsp_mv[0].phi,
      ldsp_mv[0].theta, ref_frame, frame_stride, frame_width, frame_height,
      pred_block_stride, pred_block);
  best_sad =
      AOMMIN(best_sad,
             get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               frame_stride, pred_block_stride));

  update_sphere_mv_ldsp_interp_scale(ldsp_mv, search_step_phi,
                                     search_step_theta, start_phi);

  const double min_scale = 0.125;

  while (start_phi + ldsp_mv[5].phi <= max_range_phi &&
         start_phi + ldsp_mv[1].phi >= min_range_phi &&
         start_theta + ldsp_mv[3].theta <=
             start_theta + start_mv->theta +
                 cal_range_theta(search_range, frame_width,
                                 start_phi + ldsp_mv[3].phi) &&
         start_theta + ldsp_mv[7].theta >=
             start_theta + start_mv->theta -
                 cal_range_theta(search_range, frame_width,
                                 start_phi + ldsp_mv[7].phi)) {
    for (int i = 0; i < 9; i++) {
      av1_get_pred_erp_interp_bilinear(
          block_x, block_y, block_width, block_height, ldsp_mv[i].phi,
          ldsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);
      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            frame_stride, pred_block_stride);

      av1_get_pred_erp_interp_sub_pel_8(
          block_x, block_y, block_width, block_height, ldsp_mv[i].phi,
          ldsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);
      temp_sad =
          AOMMIN(temp_sad, get_sad_of_blocks(cur_block, pred_block, block_width,
                                             block_height, frame_stride,
                                             pred_block_stride));

      av1_get_pred_erp_interp_sub_pel_8sharp(
          block_x, block_y, block_width, block_height, ldsp_mv[i].phi,
          ldsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);
      temp_sad =
          AOMMIN(temp_sad, get_sad_of_blocks(cur_block, pred_block, block_width,
                                             block_height, frame_stride,
                                             pred_block_stride));

      av1_get_pred_erp_interp_sub_pel_8smooth(
          block_x, block_y, block_width, block_height, ldsp_mv[i].phi,
          ldsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);
      temp_sad =
          AOMMIN(temp_sad, get_sad_of_blocks(cur_block, pred_block, block_width,
                                             block_height, frame_stride,
                                             pred_block_stride));

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv_idx = i;
      }
    }  // for

    if (best_mv_idx == 0) {
      if (search_step_phi > min_scale * PI / frame_height &&
          search_step_theta > min_scale * 2 * PI / frame_height) {
        search_step_phi *= 0.5;
        search_step_theta *= 0.5;
      } else {
        break;
      }
    } else {
      ldsp_mv[0].phi = ldsp_mv[best_mv_idx].phi;
      ldsp_mv[0].theta = ldsp_mv[best_mv_idx].theta;
      best_mv_idx = 0;
    }
    update_sphere_mv_ldsp_interp_scale(ldsp_mv, search_step_phi,
                                       search_step_theta, start_phi);
  }

  sdsp_mv[0].phi = ldsp_mv[0].phi;
  sdsp_mv[0].theta = ldsp_mv[0].theta;
  best_mv_idx = 0;
  search_step_phi = min_scale * PI / frame_height;
  search_step_theta = min_scale * 2 * PI / frame_width;
  update_sphere_mv_sdsp_interp_scale(sdsp_mv, search_step_phi,
                                     search_step_theta, start_phi);
  if (start_phi + sdsp_mv[3].phi <= max_range_phi &&
      start_phi + sdsp_mv[1].phi >= min_range_phi &&
      start_theta + sdsp_mv[2].theta <=
          start_theta + start_mv->theta +
              cal_range_theta(search_range, frame_width,
                              start_phi + sdsp_mv[2].phi) &&
      start_theta + sdsp_mv[4].theta >=
          start_theta + start_mv->theta +
              cal_range_theta(search_range, frame_width,
                              start_phi + sdsp_mv[4].phi)) {
    for (int i = 0; i < 5; i++) {
      av1_get_pred_erp_interp_bilinear(
          block_x, block_y, block_width, block_height, sdsp_mv[i].phi,
          sdsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);
      temp_sad =
          get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                            frame_stride, pred_block_stride);

      av1_get_pred_erp_interp_sub_pel_8(
          block_x, block_y, block_width, block_height, sdsp_mv[i].phi,
          sdsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);
      temp_sad =
          AOMMIN(temp_sad, get_sad_of_blocks(cur_block, pred_block, block_width,
                                             block_height, frame_stride,
                                             pred_block_stride));

      av1_get_pred_erp_interp_sub_pel_8sharp(
          block_x, block_y, block_width, block_height, sdsp_mv[i].phi,
          sdsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);
      temp_sad =
          AOMMIN(temp_sad, get_sad_of_blocks(cur_block, pred_block, block_width,
                                             block_height, frame_stride,
                                             pred_block_stride));

      av1_get_pred_erp_interp_sub_pel_8smooth(
          block_x, block_y, block_width, block_height, sdsp_mv[i].phi,
          sdsp_mv[i].theta, ref_frame, frame_stride, frame_width, frame_height,
          pred_block_stride, pred_block);
      temp_sad =
          AOMMIN(temp_sad, get_sad_of_blocks(cur_block, pred_block, block_width,
                                             block_height, frame_stride,
                                             pred_block_stride));

      if (temp_sad < best_sad) {
        best_sad = temp_sad;
        best_mv_idx = i;
      }
    }  // for
  }

  best_mv->phi = sdsp_mv[best_mv_idx].phi;
  best_mv->theta = sdsp_mv[best_mv_idx].theta;

  return best_sad;
}

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

// Given pixels in y, write a bmp file with indices in highlight higlighted
void write_bmp(char *filename, int width, int height, uint8_t *y,
               int *highlight, int block_width, int block_height,
               int block_stride) {
  int Y;
  int length = 3 * width * height;

  char tag[] = { 'B', 'M' };
  int header[] = {
    0,                                       // File size... update at the end.
    0,        0x36, 0x28, width,    height,  // Image dimensions in pixels

    0x180001, 0,    0,    0x002e23, 0x002e23, 0, 0,
  };
  // Update file size: just the sum of the sizes of the arrays
  // we write to disk.
  header[0] = sizeof(tag) + sizeof(header) + length;

  FILE *fp = fopen(filename, "w+");
  fwrite(&tag, sizeof(tag), 1, fp);
  fwrite(&header, sizeof(header), 1, fp);

  uint8_t *lume = (uint8_t *)malloc(width * height * sizeof(*y));

  for (int i = 0; i < width * height; i++) {
    lume[i] = y[i] * 0.5;
  }

  for (int j = 0; j < block_height; j++) {
    for (int i = 0; i < block_width; i++) {
      lume[highlight[j * block_stride + i]] *= 2;
    }
  }

  for (int j = height - 1; j >= 0; j--) {
    for (int i = 0; i < width; i++) {
      Y = lume[i + j * width];

      fwrite(&Y, sizeof(tag[0]), 1, fp);
      fwrite(&Y, sizeof(tag[0]), 1, fp);
      fwrite(&Y, sizeof(tag[0]), 1, fp);
    }
  }

  fclose(fp);
  free(lume);
}

// Given block coordiante and ERP motion vector, get the indices of pixels in
// the block and the reference block. The result is for write_bmp.
void get_pred_block_idx(int block_x, int block_y, int block_width,
                        int block_height, double delta_phi, double delta_theta,
                        int frame_stride, int frame_width, int frame_height,
                        int pred_block_stride, int block[], int pred_block[]) {
  for (int j = 0; j < block_height; j++) {
    for (int i = 0; i < block_width; i++) {
      block[j * pred_block_stride + i] =
          block_x + i + (block_y + j) * frame_stride;
    }
  }

  int pos_pred;      // Offset in pred_block buffer
  int pos_ref;       // Offset in ref_frame buffer
  double x_current;  // X coordiante in current frame
  double y_current;  // Y coordiante in currrent frame
  double x_ref;      // X coordinate in reference frame
  double y_ref;      // Y coordiante in reference frame

  PolarVector block_polar;
  CartesianVector block_v;
  CartesianVector blcok_v_prime;
  CartesianVector k;  // The axis that the block will rotate along
  block_polar.r = 1.0;
  av1_plane_to_sphere_erp(block_x, block_y, frame_width, frame_height,
                          &block_polar.phi, &block_polar.theta);
  block_polar.phi += 0.5 * pi;
  av1_sphere_polar_to_carte(&block_polar, &block_v);

  block_polar.phi += delta_phi;
  block_polar.theta += delta_theta;
  av1_sphere_polar_to_carte(&block_polar, &blcok_v_prime);

  av1_carte_vectors_cross_product(&block_v, &blcok_v_prime, &k);
  av1_normalize_carte_vector(&k);

  double product = av1_carte_vectors_dot_product(&block_v, &blcok_v_prime);
  // Avoid floating point precision issues
  product = MAX(product, -1.0);
  product = MIN(product, 1.0);
  double alpha = acos(product);

  CartesianVector v;
  CartesianVector v_prime;
  PolarVector v_polar;
  PolarVector v_prime_polar;
  v_polar.r = 1.0;
  v_prime_polar.r = 1.0;
  double k_dot_v;

  for (int idx_y = 0; idx_y < block_height; idx_y++) {
    for (int idx_x = 0; idx_x < block_width; idx_x++) {
      x_current = idx_x + block_x;
      y_current = idx_y + block_y;

      av1_plane_to_sphere_erp(x_current, y_current, frame_width, frame_height,
                              &v_polar.phi, &v_polar.theta);
      v_polar.phi += 0.5 * pi;
      av1_sphere_polar_to_carte(&v_polar, &v);
      k_dot_v = av1_carte_vectors_dot_product(&k, &v);

      v_prime.x = k.x * k_dot_v * (1 - cos(alpha)) + v.x * cos(alpha) +
                  (k.y * v.z - k.z * v.y) * sin(alpha);
      v_prime.y = k.y * k_dot_v * (1 - cos(alpha)) + v.y * cos(alpha) +
                  (k.z * v.x - k.x * v.z) * sin(alpha);
      v_prime.z = k.z * k_dot_v * (1 - cos(alpha)) + v.z * cos(alpha) +
                  (k.x * v.y - k.y * v.x) * sin(alpha);

      av1_sphere_carte_to_polar(&v_prime, &v_prime_polar);
      v_prime_polar.phi -= 0.5 * pi;
      // v_prime_polar.theta -= pi;
      av1_sphere_to_plane_erp(v_prime_polar.phi, v_prime_polar.theta,
                              frame_width, frame_height, &x_ref, &y_ref);

      x_ref = round(x_ref);
      x_ref = x_ref == frame_width ? 0 : x_ref;
      y_ref = round(y_ref);
      y_ref = y_ref == frame_height ? frame_height - 1 : y_ref;

      pos_pred = idx_x + idx_y * pred_block_stride;
      pos_ref = (int)x_ref + (int)y_ref * frame_stride;
      pred_block[pos_pred] = pos_ref;
    }
  }
}

static char *filepaths[] = {
  "360test.y4m",
  "/usr/local/google/home/yaoyaogoogle/Downloads/AerialCity_960P.y4m",
  "/usr/local/google/home/yaoyaogoogle/Downloads/Balboa_960P.y4m",
  "/usr/local/google/home/yaoyaogoogle/Downloads/BranCastle_960P.y4m",
  "/usr/local/google/home/yaoyaogoogle/Downloads/Broadway_960P.y4m",
  "/usr/local/google/home/yaoyaogoogle/Downloads/Gaslamp_960P.y4m",
  "/usr/local/google/home/yaoyaogoogle/Downloads/Harbor_960P.y4m",
  "/usr/local/google/home/yaoyaogoogle/Downloads/KiteFlite_960P.y4m",
  "/usr/local/google/home/yaoyaogoogle/Downloads/Landing_960P.y4m",
  "/usr/local/google/home/yaoyaogoogle/Downloads/PoleVault_le_960P.y4m",
  "/usr/local/google/home/yaoyaogoogle/Downloads/Trolley_960P.y4m",
};

static char *output_filepaths[] = {
  "./whole_frame_analyze/360test/",  "./whole_frame_analyze/AerialCity/",
  "./whole_frame_analyze/Balboa/",   "./whole_frame_analyze/BranCastle/",
  "./whole_frame_analyze/Broadway/", "./whole_frame_analyze/Gaslamp/",
  "./whole_frame_analyze/Harbor/",   "./whole_frame_analyze/KiteFlite/",
  "./whole_frame_analyze/Landing/",  "./whole_frame_analyze/PoleVault/",
  "./whole_frame_analyze/Trolley/",
};

typedef enum {
  TEST360,
  ARIALCITY960,
  BALBOA960,
  BRANCASTLE960,
  BROADWAY960,
  GASLAMP960,
  HARBOR960,
  KITEFLITE960,
  LANDING960,
  POLEVAULT960,
  TROLLEY960,
} FILENAME;

typedef struct {
  FILENAME filename;
  int cur_frame_no;
  int ref_frame_no;
  int block_x;
  int block_y;
  int block_width;
  int block_height;
  int search_range;
} TestConfig;

// Given block and frame info, highlight the motion search result
static void motion_search_test(const TestConfig *test_config) {
  y4m_input_t *handle = NULL;
  config_t config;
  ASSERT_EQ(0, open_file_y4m(filepaths[test_config->filename],
                             (handle_t *)(&handle), &config));

  picture_t cur_pic;
  picture_t ref_pic;
  uint8_t temp;
  cur_pic.img.plane[0] =
      (uint8_t *)malloc(handle->width * handle->height * sizeof(temp));
  cur_pic.img.plane[1] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));
  cur_pic.img.plane[2] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));
  ref_pic.img.plane[0] =
      (uint8_t *)malloc(handle->width * handle->height * sizeof(temp));
  ref_pic.img.plane[1] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));
  ref_pic.img.plane[2] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));

  read_frame_y4m((handle_t)handle, &cur_pic, test_config->cur_frame_no);
  read_frame_y4m((handle_t)handle, &ref_pic, test_config->ref_frame_no);

  SphereMV start_sphere_mv;
  start_sphere_mv.phi = 0;
  start_sphere_mv.theta = 0;
  SphereMV best_sphere_mv;
  PlaneMV best_plane_mv;

  double ref_block_x;
  double ref_block_y;
  double block_phi;
  double block_theta;

  int pred_block[128 * 128];
  int cur_block[128 * 128];
  const int block_stride = 128;

  clock_t start, end;
  double cpu_time_used;
  int sad;

  char *filename = basename(filepaths[test_config->filename]);
  char output[80];

  printf("File path: %s\n", filepaths[test_config->filename]);
  printf("Current block: (%d, %d), size: (%d, %d), in frame %d\n",
         test_config->block_x, test_config->block_y, test_config->block_width,
         test_config->block_height, test_config->cur_frame_no);

  start = clock();
  sad = av1_motion_search_brute_force_plane(
      test_config->block_x, test_config->block_y, test_config->block_width,
      test_config->block_height, cur_pic.img.plane[0], ref_pic.img.plane[0],
      handle->width, handle->width, handle->height, test_config->search_range,
      &best_plane_mv);
  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf(
      "Plane bruteforce. Best SAD: %d at (%d, %d) in frame %d. Processing "
      "time: %f second(s)\n",
      sad, test_config->block_x + best_plane_mv.x,
      test_config->block_y + best_plane_mv.y, test_config->ref_frame_no,
      cpu_time_used);

  start = clock();
  sad = av1_motion_search_diamond_plane(
      test_config->block_x, test_config->block_y, test_config->block_width,
      test_config->block_height, cur_pic.img.plane[0], ref_pic.img.plane[0],
      handle->width, handle->width, handle->height, test_config->search_range,
      &best_plane_mv);
  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf(
      "Plane diamond. Best SAD: %d at (%d, %d) in frame %d. Processing time: "
      "%f second(s)\n",
      sad, test_config->block_x + best_plane_mv.x,
      test_config->block_y + best_plane_mv.y, test_config->ref_frame_no,
      cpu_time_used);

  av1_plane_to_sphere_erp(test_config->block_x, test_config->block_y,
                          handle->width, handle->height, &block_phi,
                          &block_theta);

  start = clock();
  sad = av1_motion_search_brute_force_erp(
      test_config->block_x, test_config->block_y, test_config->block_width,
      test_config->block_height, cur_pic.img.plane[0], ref_pic.img.plane[0],
      handle->width, handle->width, handle->height, test_config->search_range,
      &best_sphere_mv);
  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  av1_sphere_to_plane_erp(block_phi + best_sphere_mv.phi,
                          block_theta + best_sphere_mv.theta, handle->width,
                          handle->height, &ref_block_x, &ref_block_y);
  printf(
      "Sphere bruteforce. Best SAD: %d at Phi: %f, and Theta: %f (%f, %f) in"
      " frame %d. Processing time: % f second(s)\n ",
      sad, best_sphere_mv.phi, best_sphere_mv.theta, ref_block_x, ref_block_y,
      test_config->ref_frame_no, cpu_time_used);

  get_pred_block_idx(test_config->block_x, test_config->block_y,
                     test_config->block_width, test_config->block_height,
                     best_sphere_mv.phi, best_sphere_mv.theta, handle->width,
                     handle->width, handle->height, block_stride, cur_block,
                     pred_block);

  int sad_bruteforce_erp = sad_of_blocks_from_idx(
      cur_pic.img.plane[0], ref_pic.img.plane[0], cur_block, pred_block,
      test_config->block_width, test_config->block_height, block_stride,
      block_stride);
  snprintf(output, sizeof(output), "%.*s_%d.bmp", strlen(filename) - 4,
           filename, test_config->cur_frame_no);
  write_bmp(output, handle->width, handle->height, cur_pic.img.plane[0],
            cur_block, test_config->block_width, test_config->block_height,
            block_stride);
  snprintf(output, sizeof(output), "%.*s_%d_to_%d_bruteforce_erp.bmp",
           strlen(filename) - 4, filename, test_config->cur_frame_no,
           test_config->ref_frame_no);
  write_bmp(output, handle->width, handle->height, ref_pic.img.plane[0],
            pred_block, test_config->block_width, test_config->block_height,
            block_stride);

  start = clock();
  sad = av1_motion_search_diamond_erp(
      test_config->block_x, test_config->block_y, test_config->block_width,
      test_config->block_height, cur_pic.img.plane[0], ref_pic.img.plane[0],
      handle->width, handle->width, handle->height, test_config->search_range,
      &start_sphere_mv, &best_sphere_mv);
  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  av1_sphere_to_plane_erp(block_phi + best_sphere_mv.phi,
                          block_theta + best_sphere_mv.theta, handle->width,
                          handle->height, &ref_block_x, &ref_block_y);
  printf(
      "Sphere diamond. Best SAD: %d at Phi: %f, and Theta: %f (%f, %f) in "
      "frame %d. Processing time: "
      "%f second(s)\n\n",
      sad, best_sphere_mv.phi, best_sphere_mv.theta, ref_block_x, ref_block_y,
      test_config->ref_frame_no, cpu_time_used);

  get_pred_block_idx(test_config->block_x, test_config->block_y,
                     test_config->block_width, test_config->block_height,
                     best_sphere_mv.phi, best_sphere_mv.theta, handle->width,
                     handle->width, handle->height, block_stride, cur_block,
                     pred_block);
  snprintf(output, sizeof(output), "%.*s_%d_to_%d_diamond_erp.bmp",
           strlen(filename) - 4, filename, test_config->cur_frame_no,
           test_config->ref_frame_no);
  write_bmp(output, handle->width, handle->height, ref_pic.img.plane[0],
            pred_block, test_config->block_width, test_config->block_height,
            block_stride);

  free(cur_pic.img.plane[0]);
  free(cur_pic.img.plane[1]);
  free(cur_pic.img.plane[2]);
  free(ref_pic.img.plane[0]);
  free(ref_pic.img.plane[1]);
  free(ref_pic.img.plane[2]);

  close_file_y4m((handle_t)handle);
}

TEST(SphericalMappingTest, EquiMotionSearchTest) {
  // TestConfig configs[] = {
  //   { ARIALCITY960, 60, 65, 1637, 665, 32, 32, 50 },
  //   { BALBOA960, 180, 185, 514, 187, 128, 128, 50 },
  //   { BRANCASTLE960, 60, 65, 914, 398, 64, 64, 50 },
  //   { BROADWAY960, 80, 85, 1518, 570, 64, 64, 50 },
  //   { GASLAMP960, 210, 215, 1150, 500, 64, 64, 50 },
  //   { HARBOR960, 1, 5, 50, 490, 64, 64, 50 },
  //   { KITEFLITE960, 15, 20, 520, 425, 32, 32, 50 },
  //   { LANDING960, 90, 95, 828, 212, 128, 128, 50 },
  //   { POLEVAULT960, 30, 35, 328, 800, 128, 128, 50 },
  //   { TROLLEY960, 90, 95, 1649, 387, 128, 128, 50 },
  // };

  // motion_search_test(&configs[9]);

  // TestConfig configs[] = {
  //   { ARIALCITY960, 60, 65, 0, 0, 32, 32, 50 },
  //   { BALBOA960, 180, 185, 0, 0, 128, 128, 50 },
  //   { BRANCASTLE960, 60, 65, 0, 0, 64, 64, 50 },
  //   { BROADWAY960, 80, 85, 0, 0, 64, 64, 50 },
  //   { GASLAMP960, 210, 215, 0, 0, 64, 64, 50 },
  //   { HARBOR960, 1, 5, 0, 0, 64, 64, 50 },
  //   { KITEFLITE960, 15, 20, 0, 0, 32, 32, 50 },
  //   { LANDING960, 90, 95, 0, 0, 128, 128, 50 },
  //   { POLEVAULT960, 30, 35, 0, 0, 128, 128, 50 },
  //   { TROLLEY960, 90, 95, 0, 0, 128, 128, 50 },
  // };

  // TestConfig configs[] = {
  //   { ARIALCITY960, 60, 60, 1637, 665, 32, 32, 50 },
  //   { BALBOA960, 180, 180, 514, 187, 128, 128, 50 },
  //   { BRANCASTLE960, 60, 60, 914, 398, 64, 64, 50 },
  //   { BROADWAY960, 80, 80, 1518, 570, 64, 64, 50 },
  //   { GASLAMP960, 210, 210, 1150, 500, 64, 64, 50 },
  //   { HARBOR960, 1, 1, 50, 490, 64, 64, 50 },
  //   { KITEFLITE960, 15, 15, 520, 425, 32, 32, 50 },
  //   { LANDING960, 90, 90, 828, 212, 128, 128, 50 },
  //   { POLEVAULT960, 30, 30, 328, 800, 128, 128, 50 },
  //   { TROLLEY960, 90, 90, 1649, 387, 128, 128, 50 },
  // };

  // TestConfig configs[] = {
  //   { ARIALCITY960, 60, 60, 0, 0, 128, 128, 20 },
  //   { ARIALCITY960, 60, 60, 1920 / 4, 0, 32, 32, 20 },
  //   { ARIALCITY960, 60, 60, 1920 / 2, 0, 32, 32, 20 },
  //   { ARIALCITY960, 60, 60, 1920 / 4 * 3, 0, 32, 32, 20 },
  //   { ARIALCITY960, 60, 60, 1920 - 128, 0, 32, 32, 20 },
  //   { ARIALCITY960, 60, 60, 0, 960 / 2, 32, 32, 20 },
  //   { ARIALCITY960, 60, 60, 1920 / 2, 960 / 2, 32, 32, 20 },
  //   { ARIALCITY960, 60, 60, 1920 - 128, 960 / 2, 32, 32, 20 },
  //   { ARIALCITY960, 60, 60, 0, 960 - 128, 32, 32, 20 },
  //   { ARIALCITY960, 60, 60, 1920 / 2, 960 - 128, 32, 32, 20 },
  //   { ARIALCITY960, 60, 60, 1920 - 128, 960 - 128, 32, 32, 20 },
  //   { ARIALCITY960, 60, 60, 0, 0, 32, 32, 20 },
  //   { ARIALCITY960, 60, 60, 0, 1, 32, 32, 20 },
  //   { ARIALCITY960, 60, 60, 0, 2, 32, 32, 20 },
  //   { ARIALCITY960, 60, 60, 0, 923, 32, 32, 20 },
  //   { ARIALCITY960, 60, 60, 0, 924, 32, 32, 20 },
  //   { ARIALCITY960, 60, 60, 0, 952, 8, 8, 20 },
  // };

  TestConfig configs[] = {
    { ARIALCITY960, 60, 60, 0, 0, 128, 128, 20 },
    { BALBOA960, 60, 60, 0, 0, 128, 128, 20 },
    { BRANCASTLE960, 60, 60, 0, 0, 128, 128, 20 },
    { BROADWAY960, 60, 60, 0, 0, 128, 128, 20 },
    { GASLAMP960, 60, 60, 0, 0, 128, 128, 20 },
    { HARBOR960, 60, 60, 0, 0, 128, 128, 20 },
    { KITEFLITE960, 60, 60, 0, 0, 128, 128, 20 },
    { LANDING960, 60, 60, 0, 0, 128, 128, 20 },
    { POLEVAULT960, 60, 60, 0, 0, 128, 128, 20 },
    { TROLLEY960, 60, 60, 0, 0, 128, 128, 20 },
  };

  for (int i = 0; i < 10; i++) {
    motion_search_test(&configs[i]);
  }

  // motion_search_test(&configs[0]);
  // motion_search_test(&configs[16]);
}

// Motion search for multiple blocks in the frame. Print average SAD and other
// info.
static void whole_frame_motion_search_test(const TestConfig *test_config) {
  y4m_input_t *handle = NULL;
  config_t config;
  ASSERT_EQ(0, open_file_y4m(filepaths[test_config->filename],
                             (handle_t *)(&handle), &config));

  picture_t cur_pic;
  picture_t ref_pic;
  uint8_t temp;
  cur_pic.img.plane[0] =
      (uint8_t *)malloc(handle->width * handle->height * sizeof(temp));
  cur_pic.img.plane[1] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));
  cur_pic.img.plane[2] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));
  ref_pic.img.plane[0] =
      (uint8_t *)malloc(handle->width * handle->height * sizeof(temp));
  ref_pic.img.plane[1] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));
  ref_pic.img.plane[2] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));

  read_frame_y4m((handle_t)handle, &cur_pic, test_config->cur_frame_no);
  read_frame_y4m((handle_t)handle, &ref_pic, test_config->ref_frame_no);

  SphereMV start_sphere_mv;
  start_sphere_mv.phi = 0;
  start_sphere_mv.theta = 0;
  SphereMV best_sphere_mv;
  PlaneMV best_plane_mv;

  int block_x;
  int block_y;
  int block_step_x = handle->width / 50;
  int block_step_y = handle->height / 50;

  int avg_sad_plane_bruteforce = 0;
  int avg_sad_plane_diamond = 0;
  int avg_sad_erp_bruteforce_p2p = 0;
  int avg_sad_erp_bruteforce_interp = 0;
  int avg_sad_erp_diamond_interp = 0;
  int avg_sad_erp_diamond_p2p = 0;

  int sad_plane_bruteforce = 0;
  int sad_plane_diamond = 0;
  int sad_erp_bruteforce_p2p = 0;
  int sad_erp_bruteforce_interp = 0;
  int sad_erp_diamond_interp = 0;
  int sad_erp_diamond_p2p = 0;

  int block_count = 0;
  int block_count_diamond_interp_better = 0;
  int block_count_bruteforce_interp_better = 0;
  int block_count_bruteforce_erp_better = 0;
  int block_count_diamond_erp_better = 0;

  clock_t start, end;
  double cpu_time_used;

  double block_phi;
  double block_theta;
  double plane_phi;
  double plane_theta;

  printf("File path: %s\n", filepaths[test_config->filename]);
  printf("Current frame: %d, reference frame: %d\n", test_config->cur_frame_no,
         test_config->ref_frame_no);
  printf("Block size: %dx%d, search range: %d\n", test_config->block_width,
         test_config->block_height, test_config->search_range);

  start = clock();
  for (block_y = 0; block_y + test_config->block_height < handle->height;
       block_y += block_step_y) {
    for (block_x = 0; block_x + test_config->block_width < handle->width;
         block_x += block_step_x) {
      sad_plane_bruteforce = av1_motion_search_brute_force_plane(
          block_x, block_y, test_config->block_width, test_config->block_height,
          cur_pic.img.plane[0], ref_pic.img.plane[0], handle->width,
          handle->width, handle->height, test_config->search_range,
          &best_plane_mv);

      av1_plane_to_sphere_erp(block_x + best_plane_mv.x,
                              block_y + best_plane_mv.y, handle->width,
                              handle->height, &plane_phi, &plane_theta);

      av1_plane_to_sphere_erp(block_x, block_y, handle->width, handle->height,
                              &block_phi, &block_theta);

      start_sphere_mv.phi = plane_phi - block_phi;
      start_sphere_mv.theta = plane_theta - block_theta;

      sad_plane_diamond = av1_motion_search_diamond_plane(
          block_x, block_y, test_config->block_width, test_config->block_height,
          cur_pic.img.plane[0], ref_pic.img.plane[0], handle->width,
          handle->width, handle->height, test_config->search_range,
          &best_plane_mv);

      sad_erp_bruteforce_p2p = av1_motion_search_brute_force_erp(
          block_x, block_y, test_config->block_width, test_config->block_height,
          cur_pic.img.plane[0], ref_pic.img.plane[0], handle->width,
          handle->width, handle->height, test_config->search_range,
          &best_sphere_mv);

      sad_erp_bruteforce_interp = av1_motion_search_brute_force_erp_interp(
          block_x, block_y, test_config->block_width, test_config->block_height,
          cur_pic.img.plane[0], ref_pic.img.plane[0], handle->width,
          handle->width, handle->height, test_config->search_range,
          &best_sphere_mv);

      sad_erp_diamond_p2p = av1_motion_search_diamond_erp(
          block_x, block_y, test_config->block_width, test_config->block_height,
          cur_pic.img.plane[0], ref_pic.img.plane[0], handle->width,
          handle->width, handle->height, test_config->search_range,
          &start_sphere_mv, &best_sphere_mv);

      sad_erp_diamond_interp = av1_motion_search_diamond_erp_interp(
          block_x, block_y, test_config->block_width, test_config->block_height,
          cur_pic.img.plane[0], ref_pic.img.plane[0], handle->width,
          handle->width, handle->height, test_config->search_range,
          &start_sphere_mv, &best_sphere_mv);

      block_count++;

      if (sad_erp_diamond_interp < sad_erp_diamond_p2p) {
        block_count_diamond_interp_better++;
      }

      if (sad_erp_bruteforce_interp < sad_erp_bruteforce_p2p) {
        block_count_bruteforce_interp_better++;
      }

      if (sad_erp_bruteforce_p2p < sad_plane_bruteforce) {
        block_count_bruteforce_erp_better++;
      }

      if (sad_erp_diamond_p2p < sad_plane_diamond) {
        block_count_diamond_erp_better++;
      }

      avg_sad_plane_bruteforce += sad_plane_bruteforce;
      avg_sad_plane_diamond += sad_plane_diamond;
      avg_sad_erp_diamond_p2p += sad_erp_diamond_p2p;
      avg_sad_erp_bruteforce_interp += sad_erp_bruteforce_interp;
      avg_sad_erp_bruteforce_p2p += sad_erp_bruteforce_p2p;
      avg_sad_erp_diamond_interp += sad_erp_diamond_interp;
    }
  }  // for idx_y

  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

  printf("Processing time: %f second(s)\n ", cpu_time_used);
  printf("Total block count: %d\n", block_count);

  avg_sad_plane_bruteforce /= block_count;
  avg_sad_plane_diamond /= block_count;
  avg_sad_erp_diamond_p2p /= block_count;
  avg_sad_erp_bruteforce_interp /= block_count;
  avg_sad_erp_bruteforce_p2p /= block_count;
  avg_sad_erp_diamond_interp /= block_count;

  printf("Average plane bruteforce SAD: %d\n", avg_sad_plane_bruteforce);
  printf("Average plane diamond SAD: %d\n", avg_sad_plane_diamond);
  printf("Pixel to pixel average equirectangle bruteforce SAD: %d\n",
         avg_sad_erp_bruteforce_p2p);
  printf("Interp average equirectangle bruteforce SAD: %d\n",
         avg_sad_erp_bruteforce_interp);
  printf("Pixel to pixel average equirectangle diamond SAD: %d\n",
         avg_sad_erp_diamond_p2p);
  printf("Interp average equirectangle diamond SAD: %d\n",
         avg_sad_erp_diamond_interp);

  printf("Gain interp vs plane bruteforce: %f%%\n",
         100.0 * (avg_sad_plane_bruteforce - avg_sad_erp_bruteforce_interp) /
             avg_sad_plane_bruteforce);

  printf("Gain interp vs pixel to pixel bruteforce: %f%%\n",
         100.0 * (avg_sad_erp_bruteforce_p2p - avg_sad_erp_bruteforce_interp) /
             avg_sad_erp_bruteforce_p2p);

  printf("Gain interp vs plane diamond: %f%%\n",
         100.0 * (avg_sad_plane_bruteforce - avg_sad_erp_diamond_interp) /
             avg_sad_plane_bruteforce);

  printf(
      "%d (%f%%) blocks has less SAD with diamond ERP p2p comparing to "
      "burteforce diamond p2p\n",
      block_count_diamond_erp_better,
      100.0 * block_count_diamond_erp_better / block_count);
  printf(
      "%d (%f%%) blocks has less SAD with burteforce ERP p2p comparing to "
      "burteforce plane p2p\n",
      block_count_bruteforce_erp_better,
      100.0 * block_count_bruteforce_erp_better / block_count);
  printf(
      "%d (%f%%) blocks has less SAD with burteforce ERP interpolation "
      "comparing to "
      "burteforce ERP p2p\n",
      block_count_bruteforce_interp_better,
      100.0 * block_count_bruteforce_interp_better / block_count);
  printf(
      "%d (%f%%) blocks has less SAD with diamond ERP interpolation comparing "
      "to "
      "diamond ERP p2p\n",
      block_count_diamond_interp_better,
      100.0 * block_count_diamond_interp_better / block_count);
  printf("Gain interp vs pixel to pixel diamond: %f%%\n\n",
         100.0 * (avg_sad_erp_diamond_p2p - avg_sad_erp_diamond_interp) /
             avg_sad_erp_diamond_p2p);

  free(cur_pic.img.plane[0]);
  free(cur_pic.img.plane[1]);
  free(cur_pic.img.plane[2]);
  free(ref_pic.img.plane[0]);
  free(ref_pic.img.plane[1]);
  free(ref_pic.img.plane[2]);

  close_file_y4m((handle_t)handle);
}

TEST(SphericalMappingTest, EquiWholeFrameTest) {
  TestConfig configs[] = {
    { ARIALCITY960, 60, 65, 1637, 665, 32, 32, 20 },
    { BALBOA960, 180, 185, 514, 187, 32, 32, 20 },
    { BRANCASTLE960, 60, 65, 914, 398, 32, 32, 20 },
    { BROADWAY960, 80, 85, 1518, 570, 32, 32, 20 },
    { GASLAMP960, 210, 215, 1150, 500, 32, 32, 20 },
    { HARBOR960, 1, 5, 50, 490, 32, 32, 20 },
    { KITEFLITE960, 15, 20, 520, 425, 32, 32, 20 },
    { LANDING960, 90, 95, 828, 212, 32, 32, 20 },
    { POLEVAULT960, 30, 35, 328, 800, 32, 32, 20 },
    { TROLLEY960, 90, 95, 1649, 387, 32, 32, 20 },
  };

  // TestConfig configs[] = {
  //   { ARIALCITY960, 60, 60, 1637, 665, 32, 32, 20 },
  //   { BALBOA960, 180, 180, 514, 187, 32, 32, 20 },
  //   { BRANCASTLE960, 60, 60, 914, 398, 32, 32, 20 },
  //   { BROADWAY960, 80, 80, 1518, 570, 32, 32, 20 },
  //   { GASLAMP960, 210, 210, 1150, 500, 32, 32, 20 },
  //   { HARBOR960, 1, 1, 50, 490, 32, 32, 20 },
  //   { KITEFLITE960, 15, 15, 520, 425, 32, 32, 20 },
  //   { LANDING960, 90, 90, 828, 212, 32, 32, 20 },
  //   { POLEVAULT960, 30, 30, 328, 800, 32, 32, 20 },
  //   { TROLLEY960, 90, 90, 1649, 387, 32, 32, 20 },
  // };

  whole_frame_motion_search_test(&configs[0]);
  // for (int i = 0; i < 10; i++) {
  //   whole_frame_motion_search_test(&configs[i]);
  // }
}

static void load_frames(y4m_input_t *handle, const TestConfig *test_config,
                        picture_t *cur_pic, picture_t *ref_pic) {
  config_t config;
  ASSERT_EQ(0, open_file_y4m(filepaths[test_config->filename],
                             (handle_t *)(&handle), &config));

  uint8_t temp;
  cur_pic->img.plane[0] =
      (uint8_t *)malloc(handle->width * handle->height * sizeof(temp));
  cur_pic->img.plane[1] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));
  cur_pic->img.plane[2] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));
  ref_pic->img.plane[0] =
      (uint8_t *)malloc(handle->width * handle->height * sizeof(temp));
  ref_pic->img.plane[1] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));
  ref_pic->img.plane[2] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));

  read_frame_y4m((handle_t)handle, cur_pic, test_config->cur_frame_no);
  read_frame_y4m((handle_t)handle, ref_pic, test_config->ref_frame_no);
}

static void free_frames(y4m_input_t *handle, picture_t *cur_pic,
                        picture_t *ref_pic) {
  free(cur_pic->img.plane[0]);
  free(cur_pic->img.plane[1]);
  free(cur_pic->img.plane[2]);
  free(ref_pic->img.plane[0]);
  free(ref_pic->img.plane[1]);
  free(ref_pic->img.plane[2]);

  close_file_y4m((handle_t)handle);
}

typedef struct {
  int block_count;

  int avg_sad_plane_bruteforce;
  int avg_sad_plane_diamond;
  int avg_sad_erp_bruteforce_p2p;
  int avg_sad_erp_bruteforce_interp;
  int avg_sad_erp_diamond_interp;
  int avg_sad_erp_diamond_p2p;

  int cnt_erp_d_interp_beat_erp_d_p2p;
  int cnt_erp_d_p2p_beat_erp_d_interp;
  int cnt_erp_d_interp_equ_erp_d_p2p;

  int cnt_erp_d_interp_beat_plane_d;
  int cnt_plane_d_beat_erp_d_interp;
  int cnt_erp_d_interp_equ_plane_d;

  int cnt_erp_d_interp_beat_plane_br;
  int cnt_plane_br_beat_erp_d_interp;
  int cnt_erp_d_interp_equ_plane_br;

  int cnt_erp_d_p2p_beat_plane_d;
  int cnt_plane_d_beat_erp_d_p2p;
  int cnt_erp_d_p2p_equ_plane_d;

  int cnt_erp_d_p2p_beat_plane_br;
  int cnt_plane_br_beat_erp_d_p2p;
  int cnt_erp_d_p2p_equ_plane_br;

  int cnt_plane_d_equ_plane_br;
} AnalyzeData;

static void init_analyze_data(AnalyzeData *data) {
  data->block_count = 0;

  data->avg_sad_plane_bruteforce = 0;
  data->avg_sad_plane_diamond = 0;
  data->avg_sad_erp_bruteforce_p2p = 0;
  data->avg_sad_erp_bruteforce_interp = 0;
  data->avg_sad_erp_diamond_interp = 0;
  data->avg_sad_erp_diamond_p2p = 0;

  data->cnt_erp_d_interp_beat_erp_d_p2p = 0;
  data->cnt_erp_d_p2p_beat_erp_d_interp = 0;
  data->cnt_erp_d_interp_equ_erp_d_p2p = 0;

  data->cnt_erp_d_interp_beat_plane_d = 0;
  data->cnt_plane_d_beat_erp_d_interp = 0;
  data->cnt_erp_d_interp_equ_plane_d = 0;

  data->cnt_erp_d_interp_beat_plane_br = 0;
  data->cnt_plane_br_beat_erp_d_interp = 0;

  data->cnt_erp_d_p2p_beat_plane_d = 0;
  data->cnt_plane_d_beat_erp_d_p2p = 0;

  data->cnt_erp_d_p2p_beat_plane_br = 0;
  data->cnt_plane_br_beat_erp_d_p2p = 0;

  data->cnt_plane_d_equ_plane_br = 0;
}

typedef struct {
  int sad_plane_bruteforce;
  int sad_plane_diamond;
  int sad_erp_bruteforce_p2p;
  int sad_erp_bruteforce_interp;
  int sad_erp_diamond_interp;
  int sad_erp_diamond_p2p;
} SearchResult;

typedef enum {
  SAD_HIGHER,
  SAD_LOWER,
  SAD_EQU,
} MarkColor;

// Planed to highlight all the blocks in the frame
// But might be hard to read. Not used.
static void open_block_map(const char *filepath, int width, int height,
                           FILE *fp) {
  int length = 3 * width * height;

  char tag[] = { 'B', 'M' };
  int header[] = {
    0,                                       // File size... update at the end.
    0,        0x36, 0x28, width,    height,  // Image dimensions in pixels

    0x180001, 0,    0,    0x002e23, 0x002e23, 0, 0,
  };
  // Update file size: just the sum of the sizes of the arrays
  // we write to disk.
  header[0] = sizeof(tag) + sizeof(header) + length;

  fp = fopen(filepath, "w+");
  fwrite(&tag, sizeof(tag), 1, fp);
  fwrite(&header, sizeof(header), 1, fp);
}

static void write_block_map(FILE *fp, uint8_t *r, uint8_t *g, uint8_t *b,
                            int width, int height) {
  int R;
  int G;
  int B;

  for (int j = height - 1; j >= 0; j--) {
    for (int i = 0; i < width; i++) {
      fwrite(&r[i + j * width], sizeof(*r), 1, fp);
      fwrite(&g[i + j * width], sizeof(*r), 1, fp);
      fwrite(&b[i + j * width], sizeof(*r), 1, fp);
    }
  }
}

static void close_block_map(FILE *fp) { fclose(fp); }

// The following 10 functions write detailed search results, including reference
// block coordiantes, pixels, filter kernel, and filter result, into text files.
static void get_blk_coord_plane(int block_x, int block_y, int block_width,
                                int block_height, int coord_stride,
                                int frame_width, int frame_height, int *coord_x,
                                int *coord_y) {
  int x_current;
  int y_current;
  int pos;

  for (int idx_y = 0; idx_y < block_height; idx_y++) {
    for (int idx_x = 0; idx_x < block_width; idx_x++) {
      x_current = idx_x + block_x;
      y_current = idx_y + block_y;

      warp_coordinate(frame_width, frame_height, &x_current, &y_current);

      pos = idx_x + idx_y * coord_stride;
      coord_x[pos] = x_current;
      coord_y[pos] = y_current;
    }
  }
}

static void write_plane_data(const char *data_origin,
                             const TestConfig *test_config,
                             const uint8_t *frame, int frame_stride,
                             int frame_width, int frame_height, int block_x,
                             int block_y, int block_cnt, int sad) {
  char filepath[80];
  snprintf(filepath, sizeof(filepath), "%s%04d_%d_%d_%s_%d.txt",
           output_filepaths[test_config->filename], block_cnt, block_x, block_y,
           data_origin, sad);
  FILE *fp = fopen(filepath, "w");

  fputs(filepaths[test_config->filename], fp);
  fputs("\n\n", fp);

  char block_info[80];
  snprintf(block_info, sizeof(block_info),
           "#%d block at (%d, %d), size %d x %d, SAD: %d\n\n", block_cnt,
           block_x, block_y, test_config->block_width,
           test_config->block_height, sad);
  fputs(block_info, fp);

  const int coord_stride = 128;

  fputs("Coordinates:\n", fp);
  int coord_x[128 * 128];
  int coord_y[128 * 128];
  char coord_buf[14];
  int pos_coord;
  get_blk_coord_plane(block_x, block_y, test_config->block_width,
                      test_config->block_height, coord_stride, frame_width,
                      frame_height, coord_x, coord_y);
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      snprintf(coord_buf, sizeof(coord_buf), "(%4d, %4d) ", coord_x[pos_coord],
               coord_y[pos_coord]);
      fputs(coord_buf, fp);
    }
    fputs("\n", fp);
  }

  fputs("\nPixel:\n", fp);
  char pixel_buf[6];
  int pos_frame;
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    fputs("[", fp);
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      pos_frame = coord_x[pos_coord] + coord_y[pos_coord] * frame_stride;
      snprintf(pixel_buf, sizeof(pixel_buf), "%3d, ", frame[pos_frame]);
      fputs(pixel_buf, fp);
    }
    fputs("]\n", fp);
  }

  fclose(fp);
}

static void get_blk_coord_erp(int block_x, int block_y, int block_width,
                              int block_height, int coord_stride,
                              int frame_width, int frame_height,
                              const SphereMV *sphere_mv, int *coord_x,
                              int *coord_y) {
  double x_current;  // X coordiante in current frame
  double y_current;  // Y coordiante in currrent frame
  double x_ref;      // X coordinate in reference frame
  double y_ref;      // Y coordiante in reference frame
  int pos;

  PolarVector block_polar;
  CartesianVector block_v;
  CartesianVector blcok_v_prime;
  CartesianVector k;  // The axis that the block will rotate along
  block_polar.r = 1.0;
  av1_plane_to_sphere_erp(block_x, block_y, frame_width, frame_height,
                          &block_polar.phi, &block_polar.theta);
  block_polar.phi += 0.5 * pi;
  av1_sphere_polar_to_carte(&block_polar, &block_v);

  block_polar.phi += sphere_mv->phi;
  block_polar.theta += sphere_mv->theta;
  av1_sphere_polar_to_carte(&block_polar, &blcok_v_prime);

  av1_carte_vectors_cross_product(&block_v, &blcok_v_prime, &k);
  av1_normalize_carte_vector(&k);

  double product = av1_carte_vectors_dot_product(&block_v, &blcok_v_prime);
  // Avoid floating point precision issues
  product = MAX(product, -1.0);
  product = MIN(product, 1.0);
  double alpha = acos(product);

  CartesianVector v;
  CartesianVector v_prime;
  PolarVector v_polar;
  PolarVector v_prime_polar;
  v_polar.r = 1.0;
  v_prime_polar.r = 1.0;
  double k_dot_v;

  for (int idx_y = 0; idx_y < block_height; idx_y++) {
    for (int idx_x = 0; idx_x < block_width; idx_x++) {
      x_current = idx_x + block_x;
      y_current = idx_y + block_y;

      av1_plane_to_sphere_erp(x_current, y_current, frame_width, frame_height,
                              &v_polar.phi, &v_polar.theta);
      v_polar.phi += 0.5 * pi;
      av1_sphere_polar_to_carte(&v_polar, &v);
      k_dot_v = av1_carte_vectors_dot_product(&k, &v);

      v_prime.x = k.x * k_dot_v * (1 - cos(alpha)) + v.x * cos(alpha) +
                  (k.y * v.z - k.z * v.y) * sin(alpha);
      v_prime.y = k.y * k_dot_v * (1 - cos(alpha)) + v.y * cos(alpha) +
                  (k.z * v.x - k.x * v.z) * sin(alpha);
      v_prime.z = k.z * k_dot_v * (1 - cos(alpha)) + v.z * cos(alpha) +
                  (k.x * v.y - k.y * v.x) * sin(alpha);

      av1_sphere_carte_to_polar(&v_prime, &v_prime_polar);
      v_prime_polar.phi -= 0.5 * pi;
      // v_prime_polar.theta -= pi;
      av1_sphere_to_plane_erp(v_prime_polar.phi, v_prime_polar.theta,
                              frame_width, frame_height, &x_ref, &y_ref);

      x_ref = round(x_ref);
      x_ref = (int)x_ref == frame_width ? 0 : x_ref;
      y_ref = round(y_ref);
      y_ref = (int)y_ref == frame_height ? frame_height - 1 : y_ref;

      pos = idx_x + idx_y * coord_stride;
      coord_x[pos] = (int)x_ref;
      coord_y[pos] = (int)y_ref;
    }
  }  // for idx_y
}

static void write_erp_data(const char *data_origin,
                           const TestConfig *test_config, const uint8_t *frame,
                           int frame_stride, int frame_width, int frame_height,
                           int block_x, int block_y, int block_cnt,
                           SphereMV *sphere_mv, int sad) {
  char filepath[80];
  snprintf(filepath, sizeof(filepath), "%s%04d_%d_%d_%s_%d.txt",
           output_filepaths[test_config->filename], block_cnt, block_x, block_y,
           data_origin, sad);
  FILE *fp = fopen(filepath, "w");

  fputs(filepaths[test_config->filename], fp);
  fputs("\n\n", fp);

  char block_info[80];
  snprintf(block_info, sizeof(block_info),
           "#%d block at (%d, %d), size %d x %d, SAD: %d\n\n", block_cnt,
           block_x, block_y, test_config->block_width,
           test_config->block_height, sad);
  fputs(block_info, fp);

  const int coord_stride = 128;

  fputs("Coordinates:\n", fp);
  int coord_x[128 * 128];
  int coord_y[128 * 128];
  char coord_buf[14];
  int pos_coord;
  get_blk_coord_erp(block_x, block_y, test_config->block_width,
                    test_config->block_height, coord_stride, frame_width,
                    frame_height, sphere_mv, coord_x, coord_y);
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      snprintf(coord_buf, sizeof(coord_buf), "(%4d, %4d) ", coord_x[pos_coord],
               coord_y[pos_coord]);
      fputs(coord_buf, fp);
    }
    fputs("\n", fp);
  }

  fputs("\nPixel:\n", fp);
  char pixel_buf[6];
  int pos_frame;
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    fputs("[", fp);
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      pos_frame = coord_x[pos_coord] + coord_y[pos_coord] * frame_stride;
      snprintf(pixel_buf, sizeof(pixel_buf), "%3d, ", frame[pos_frame]);
      fputs(pixel_buf, fp);
    }
    fputs("]\n", fp);
  }

  fclose(fp);
}

static void get_blk_coord_erp_interp(int block_x, int block_y, int block_width,
                                     int block_height, int coord_stride,
                                     int frame_width, int frame_height,
                                     const SphereMV *sphere_mv, double *coord_x,
                                     double *coord_y) {
  double x_current;  // X coordiante in current frame
  double y_current;  // Y coordiante in currrent frame
  double subpel_x;   // X coordinate in reference frame
  double subpel_y;   // Y coordiante in reference frame
  int pos;

  PolarVector block_polar;
  CartesianVector block_v;
  CartesianVector blcok_v_prime;
  CartesianVector k;  // The axis that the block will rotate along
  block_polar.r = 1.0;
  av1_plane_to_sphere_erp(block_x, block_y, frame_width, frame_height,
                          &block_polar.phi, &block_polar.theta);
  block_polar.phi += 0.5 * pi;
  av1_sphere_polar_to_carte(&block_polar, &block_v);

  block_polar.phi += sphere_mv->phi;
  block_polar.theta += sphere_mv->theta;
  av1_sphere_polar_to_carte(&block_polar, &blcok_v_prime);

  av1_carte_vectors_cross_product(&block_v, &blcok_v_prime, &k);
  av1_normalize_carte_vector(&k);

  double product = av1_carte_vectors_dot_product(&block_v, &blcok_v_prime);
  // Avoid floating point precision issues
  product = MAX(product, -1.0);
  product = MIN(product, 1.0);
  double alpha = acos(product);

  CartesianVector v;
  CartesianVector v_prime;
  PolarVector v_polar;
  PolarVector v_prime_polar;
  v_polar.r = 1.0;
  v_prime_polar.r = 1.0;
  double k_dot_v;

  for (int idx_y = 0; idx_y < block_height; idx_y++) {
    for (int idx_x = 0; idx_x < block_width; idx_x++) {
      x_current = idx_x + block_x;
      y_current = idx_y + block_y;

      av1_plane_to_sphere_erp(x_current, y_current, frame_width, frame_height,
                              &v_polar.phi, &v_polar.theta);
      v_polar.phi += 0.5 * pi;
      av1_sphere_polar_to_carte(&v_polar, &v);
      k_dot_v = av1_carte_vectors_dot_product(&k, &v);

      v_prime.x = k.x * k_dot_v * (1 - cos(alpha)) + v.x * cos(alpha) +
                  (k.y * v.z - k.z * v.y) * sin(alpha);
      v_prime.y = k.y * k_dot_v * (1 - cos(alpha)) + v.y * cos(alpha) +
                  (k.z * v.x - k.x * v.z) * sin(alpha);
      v_prime.z = k.z * k_dot_v * (1 - cos(alpha)) + v.z * cos(alpha) +
                  (k.x * v.y - k.y * v.x) * sin(alpha);

      av1_sphere_carte_to_polar(&v_prime, &v_prime_polar);
      v_prime_polar.phi -= 0.5 * pi;
      // v_prime_polar.theta -= pi;
      av1_sphere_to_plane_erp(v_prime_polar.phi, v_prime_polar.theta,
                              frame_width, frame_height, &subpel_x, &subpel_y);

      // subpel_x = round(subpel_x);
      // subpel_x = subpel_x == frame_width ? 0 : subpel_x;
      // subpel_y = round(subpel_y);
      // subpel_y = subpel_y == frame_height ? frame_height - 1 : subpel_y;

      pos = idx_x + idx_y * coord_stride;
      coord_x[pos] = subpel_x;
      coord_y[pos] = subpel_y;
    }
  }  // for idx_y
}

static void write_erp_interp_data(const char *data_origin,
                                  const TestConfig *test_config,
                                  const uint8_t *frame, int frame_stride,
                                  int frame_width, int frame_height,
                                  int block_x, int block_y, int block_cnt,
                                  SphereMV *sphere_mv, int sad) {
  char filepath[80];
  snprintf(filepath, sizeof(filepath), "%s%04d_%d_%d_%s_%d.txt",
           output_filepaths[test_config->filename], block_cnt, block_x, block_y,
           data_origin, sad);
  FILE *fp = fopen(filepath, "w");

  fputs(filepaths[test_config->filename], fp);
  fputs("\n\n", fp);

  char block_info[80];
  snprintf(block_info, sizeof(block_info),
           "#%d block at (%d, %d), size %d x %d, SAD: %d\n\n", block_cnt,
           block_x, block_y, test_config->block_width,
           test_config->block_height, sad);
  fputs(block_info, fp);

  const int coord_stride = 128;

  fputs("Sub-pixel coordinates:\n", fp);
  double coord_x[128 * 128];
  double coord_y[128 * 128];
  char coord_buf[24];
  int pos_coord;
  get_blk_coord_erp_interp(block_x, block_y, test_config->block_width,
                           test_config->block_height, coord_stride, frame_width,
                           frame_height, sphere_mv, coord_x, coord_y);
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      snprintf(coord_buf, sizeof(coord_buf), "(%4.4f, %4.4f) ",
               coord_x[pos_coord], coord_y[pos_coord]);
      fputs(coord_buf, fp);
    }
    fputs("\n", fp);
  }

  int int_coord_x[128 * 128];
  int int_coord_y[128 * 128];
  int kernel_x[128 * 128];
  int kernel_y[128 * 128];
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      kernel_x[pos_coord] =
          cal_kernel_idx(coord_x[pos_coord], &int_coord_x[pos_coord]);
      kernel_y[pos_coord] =
          cal_kernel_idx(coord_y[pos_coord], &int_coord_y[pos_coord]);
    }
  }

  fputs("\nInteger coordinates (interpolation center):\n", fp);
  char int_coord_buf[14];
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      snprintf(int_coord_buf, sizeof(int_coord_buf), "(%4d, %4d) ",
               int_coord_x[pos_coord], int_coord_y[pos_coord]);
      fputs(int_coord_buf, fp);
    }
    fputs("\n", fp);
  }

  fputs("\nPixel at interpolation centers:\n", fp);
  char pixel_buf[6];
  int pos_frame;
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    fputs("[", fp);
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      pos_frame =
          int_coord_x[pos_coord] + int_coord_y[pos_coord] * frame_stride;
      snprintf(pixel_buf, sizeof(pixel_buf), "%3d, ", frame[pos_frame]);
      fputs(pixel_buf, fp);
    }
    fputs("]\n", fp);
  }

  fputs("\nInterpolation kernels:\n", fp);
  char kernel_buf[8];
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      snprintf(kernel_buf, sizeof(kernel_buf), "(%1d, %1d) ",
               kernel_x[pos_coord], kernel_y[pos_coord]);
      fputs(kernel_buf, fp);
    }
    fputs("\n", fp);
  }

  fputs("\nInterpolation results:\n", fp);
  uint8_t result;
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    fputs("[", fp);
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      result = interp_pixel_plane(frame, frame_stride, frame_width,
                                  frame_height, coord_x[pos_coord],
                                  coord_y[pos_coord], av1_sub_pel_filters_8, 8);
      snprintf(pixel_buf, sizeof(pixel_buf), "%3d, ", result);
      fputs(pixel_buf, fp);
    }
    fputs("]\n", fp);
  }

  fclose(fp);
}

static void write_erp_interp_data_bilinear(
    const char *data_origin, const TestConfig *test_config,
    const uint8_t *frame, int frame_stride, int frame_width, int frame_height,
    int block_x, int block_y, int block_cnt, SphereMV *sphere_mv, int sad) {
  char filepath[80];
  snprintf(filepath, sizeof(filepath), "%s%04d_%d_%d_%s_%d.txt",
           output_filepaths[test_config->filename], block_cnt, block_x, block_y,
           data_origin, sad);
  FILE *fp = fopen(filepath, "w");

  fputs(filepaths[test_config->filename], fp);
  fputs("\n\n", fp);

  char block_info[80];
  snprintf(block_info, sizeof(block_info),
           "#%d block at (%d, %d), size %d x %d, SAD: %d\n\n", block_cnt,
           block_x, block_y, test_config->block_width,
           test_config->block_height, sad);
  fputs(block_info, fp);

  const int coord_stride = 128;

  fputs("Sub-pixel coordinates:\n", fp);
  double coord_x[128 * 128];
  double coord_y[128 * 128];
  char coord_buf[24];
  int pos_coord;
  get_blk_coord_erp_interp(block_x, block_y, test_config->block_width,
                           test_config->block_height, coord_stride, frame_width,
                           frame_height, sphere_mv, coord_x, coord_y);
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      snprintf(coord_buf, sizeof(coord_buf), "(%4.4f, %4.4f) ",
               coord_x[pos_coord], coord_y[pos_coord]);
      fputs(coord_buf, fp);
    }
    fputs("\n", fp);
  }

  int int_coord_x[128 * 128];
  int int_coord_y[128 * 128];
  int kernel_x[128 * 128];
  int kernel_y[128 * 128];
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      kernel_x[pos_coord] =
          cal_kernel_idx(coord_x[pos_coord], &int_coord_x[pos_coord]);
      kernel_y[pos_coord] =
          cal_kernel_idx(coord_y[pos_coord], &int_coord_y[pos_coord]);
    }
  }

  fputs("\nInteger coordinates (interpolation center):\n", fp);
  char int_coord_buf[14];
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      snprintf(int_coord_buf, sizeof(int_coord_buf), "(%4d, %4d) ",
               int_coord_x[pos_coord], int_coord_y[pos_coord]);
      fputs(int_coord_buf, fp);
    }
    fputs("\n", fp);
  }

  fputs("\nPixel at interpolation centers:\n", fp);
  char pixel_buf[6];
  int pos_frame;
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    fputs("[", fp);
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      pos_frame =
          int_coord_x[pos_coord] + int_coord_y[pos_coord] * frame_stride;
      snprintf(pixel_buf, sizeof(pixel_buf), "%3d, ", frame[pos_frame]);
      fputs(pixel_buf, fp);
    }
    fputs("]\n", fp);
  }

  fputs("\nInterpolation kernels:\n", fp);
  char kernel_buf[8];
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      snprintf(kernel_buf, sizeof(kernel_buf), "(%1d, %1d) ",
               kernel_x[pos_coord], kernel_y[pos_coord]);
      fputs(kernel_buf, fp);
    }
    fputs("\n", fp);
  }

  fputs("\nInterpolation results:\n", fp);
  uint8_t result;
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    fputs("[", fp);
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      result = interp_pixel_plane(frame, frame_stride, frame_width,
                                  frame_height, coord_x[pos_coord],
                                  coord_y[pos_coord], av1_bilinear_filters, 8);
      snprintf(pixel_buf, sizeof(pixel_buf), "%3d, ", result);
      fputs(pixel_buf, fp);
    }
    fputs("]\n", fp);
  }

  fclose(fp);
}

static void write_erp_interp_data_sub_pel_8(
    const char *data_origin, const TestConfig *test_config,
    const uint8_t *frame, int frame_stride, int frame_width, int frame_height,
    int block_x, int block_y, int block_cnt, SphereMV *sphere_mv, int sad) {
  char filepath[80];
  snprintf(filepath, sizeof(filepath), "%s%04d_%d_%d_%s_%d.txt",
           output_filepaths[test_config->filename], block_cnt, block_x, block_y,
           data_origin, sad);
  FILE *fp = fopen(filepath, "w");

  fputs(filepaths[test_config->filename], fp);
  fputs("\n\n", fp);

  char block_info[80];
  snprintf(block_info, sizeof(block_info),
           "#%d block at (%d, %d), size %d x %d, SAD: %d\n\n", block_cnt,
           block_x, block_y, test_config->block_width,
           test_config->block_height, sad);
  fputs(block_info, fp);

  const int coord_stride = 128;

  fputs("Sub-pixel coordinates:\n", fp);
  double coord_x[128 * 128];
  double coord_y[128 * 128];
  char coord_buf[24];
  int pos_coord;
  get_blk_coord_erp_interp(block_x, block_y, test_config->block_width,
                           test_config->block_height, coord_stride, frame_width,
                           frame_height, sphere_mv, coord_x, coord_y);
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      snprintf(coord_buf, sizeof(coord_buf), "(%4.4f, %4.4f) ",
               coord_x[pos_coord], coord_y[pos_coord]);
      fputs(coord_buf, fp);
    }
    fputs("\n", fp);
  }

  int int_coord_x[128 * 128];
  int int_coord_y[128 * 128];
  int kernel_x[128 * 128];
  int kernel_y[128 * 128];
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      kernel_x[pos_coord] =
          cal_kernel_idx(coord_x[pos_coord], &int_coord_x[pos_coord]);
      kernel_y[pos_coord] =
          cal_kernel_idx(coord_y[pos_coord], &int_coord_y[pos_coord]);
    }
  }

  fputs("\nInteger coordinates (interpolation center):\n", fp);
  char int_coord_buf[14];
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      snprintf(int_coord_buf, sizeof(int_coord_buf), "(%4d, %4d) ",
               int_coord_x[pos_coord], int_coord_y[pos_coord]);
      fputs(int_coord_buf, fp);
    }
    fputs("\n", fp);
  }

  fputs("\nPixel at interpolation centers:\n", fp);
  char pixel_buf[6];
  int pos_frame;
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    fputs("[", fp);
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      pos_frame =
          int_coord_x[pos_coord] + int_coord_y[pos_coord] * frame_stride;
      snprintf(pixel_buf, sizeof(pixel_buf), "%3d, ", frame[pos_frame]);
      fputs(pixel_buf, fp);
    }
    fputs("]\n", fp);
  }

  fputs("\nInterpolation kernels:\n", fp);
  char kernel_buf[8];
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      snprintf(kernel_buf, sizeof(kernel_buf), "(%1d, %1d) ",
               kernel_x[pos_coord], kernel_y[pos_coord]);
      fputs(kernel_buf, fp);
    }
    fputs("\n", fp);
  }

  fputs("\nInterpolation results:\n", fp);
  uint8_t result;
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    fputs("[", fp);
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      result = interp_pixel_plane(frame, frame_stride, frame_width,
                                  frame_height, coord_x[pos_coord],
                                  coord_y[pos_coord], av1_sub_pel_filters_8, 8);
      snprintf(pixel_buf, sizeof(pixel_buf), "%3d, ", result);
      fputs(pixel_buf, fp);
    }
    fputs("]\n", fp);
  }

  fclose(fp);
}

static void write_erp_interp_data_sub_pel_8sharp(
    const char *data_origin, const TestConfig *test_config,
    const uint8_t *frame, int frame_stride, int frame_width, int frame_height,
    int block_x, int block_y, int block_cnt, SphereMV *sphere_mv, int sad) {
  char filepath[80];
  snprintf(filepath, sizeof(filepath), "%s%04d_%d_%d_%s_%d.txt",
           output_filepaths[test_config->filename], block_cnt, block_x, block_y,
           data_origin, sad);
  FILE *fp = fopen(filepath, "w");

  fputs(filepaths[test_config->filename], fp);
  fputs("\n\n", fp);

  char block_info[80];
  snprintf(block_info, sizeof(block_info),
           "#%d block at (%d, %d), size %d x %d, SAD: %d\n\n", block_cnt,
           block_x, block_y, test_config->block_width,
           test_config->block_height, sad);
  fputs(block_info, fp);

  const int coord_stride = 128;

  fputs("Sub-pixel coordinates:\n", fp);
  double coord_x[128 * 128];
  double coord_y[128 * 128];
  char coord_buf[24];
  int pos_coord;
  get_blk_coord_erp_interp(block_x, block_y, test_config->block_width,
                           test_config->block_height, coord_stride, frame_width,
                           frame_height, sphere_mv, coord_x, coord_y);
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      snprintf(coord_buf, sizeof(coord_buf), "(%4.4f, %4.4f) ",
               coord_x[pos_coord], coord_y[pos_coord]);
      fputs(coord_buf, fp);
    }
    fputs("\n", fp);
  }

  int int_coord_x[128 * 128];
  int int_coord_y[128 * 128];
  int kernel_x[128 * 128];
  int kernel_y[128 * 128];
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      kernel_x[pos_coord] =
          cal_kernel_idx(coord_x[pos_coord], &int_coord_x[pos_coord]);
      kernel_y[pos_coord] =
          cal_kernel_idx(coord_y[pos_coord], &int_coord_y[pos_coord]);
    }
  }

  fputs("\nInteger coordinates (interpolation center):\n", fp);
  char int_coord_buf[14];
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      snprintf(int_coord_buf, sizeof(int_coord_buf), "(%4d, %4d) ",
               int_coord_x[pos_coord], int_coord_y[pos_coord]);
      fputs(int_coord_buf, fp);
    }
    fputs("\n", fp);
  }

  fputs("\nPixel at interpolation centers:\n", fp);
  char pixel_buf[6];
  int pos_frame;
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    fputs("[", fp);
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      pos_frame =
          int_coord_x[pos_coord] + int_coord_y[pos_coord] * frame_stride;
      snprintf(pixel_buf, sizeof(pixel_buf), "%3d, ", frame[pos_frame]);
      fputs(pixel_buf, fp);
    }
    fputs("]\n", fp);
  }

  fputs("\nInterpolation kernels:\n", fp);
  char kernel_buf[8];
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      snprintf(kernel_buf, sizeof(kernel_buf), "(%1d, %1d) ",
               kernel_x[pos_coord], kernel_y[pos_coord]);
      fputs(kernel_buf, fp);
    }
    fputs("\n", fp);
  }

  fputs("\nInterpolation results:\n", fp);
  uint8_t result;
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    fputs("[", fp);
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      result = interp_pixel_plane(
          frame, frame_stride, frame_width, frame_height, coord_x[pos_coord],
          coord_y[pos_coord], av1_sub_pel_filters_8sharp, 8);
      snprintf(pixel_buf, sizeof(pixel_buf), "%3d, ", result);
      fputs(pixel_buf, fp);
    }
    fputs("]\n", fp);
  }

  fclose(fp);
}

static void write_erp_interp_data_sub_pel_8smooth(
    const char *data_origin, const TestConfig *test_config,
    const uint8_t *frame, int frame_stride, int frame_width, int frame_height,
    int block_x, int block_y, int block_cnt, SphereMV *sphere_mv, int sad) {
  char filepath[80];
  snprintf(filepath, sizeof(filepath), "%s%04d_%d_%d_%s_%d.txt",
           output_filepaths[test_config->filename], block_cnt, block_x, block_y,
           data_origin, sad);
  FILE *fp = fopen(filepath, "w");

  fputs(filepaths[test_config->filename], fp);
  fputs("\n\n", fp);

  char block_info[80];
  snprintf(block_info, sizeof(block_info),
           "#%d block at (%d, %d), size %d x %d, SAD: %d\n\n", block_cnt,
           block_x, block_y, test_config->block_width,
           test_config->block_height, sad);
  fputs(block_info, fp);

  const int coord_stride = 128;

  fputs("Sub-pixel coordinates:\n", fp);
  double coord_x[128 * 128];
  double coord_y[128 * 128];
  char coord_buf[24];
  int pos_coord;
  get_blk_coord_erp_interp(block_x, block_y, test_config->block_width,
                           test_config->block_height, coord_stride, frame_width,
                           frame_height, sphere_mv, coord_x, coord_y);
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      snprintf(coord_buf, sizeof(coord_buf), "(%4.4f, %4.4f) ",
               coord_x[pos_coord], coord_y[pos_coord]);
      fputs(coord_buf, fp);
    }
    fputs("\n", fp);
  }

  int int_coord_x[128 * 128];
  int int_coord_y[128 * 128];
  int kernel_x[128 * 128];
  int kernel_y[128 * 128];
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      kernel_x[pos_coord] =
          cal_kernel_idx(coord_x[pos_coord], &int_coord_x[pos_coord]);
      kernel_y[pos_coord] =
          cal_kernel_idx(coord_y[pos_coord], &int_coord_y[pos_coord]);
    }
  }

  fputs("\nInteger coordinates (interpolation center):\n", fp);
  char int_coord_buf[14];
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      snprintf(int_coord_buf, sizeof(int_coord_buf), "(%4d, %4d) ",
               int_coord_x[pos_coord], int_coord_y[pos_coord]);
      fputs(int_coord_buf, fp);
    }
    fputs("\n", fp);
  }

  fputs("\nPixel at interpolation centers:\n", fp);
  char pixel_buf[6];
  int pos_frame;
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    fputs("[", fp);
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      pos_frame =
          int_coord_x[pos_coord] + int_coord_y[pos_coord] * frame_stride;
      snprintf(pixel_buf, sizeof(pixel_buf), "%3d, ", frame[pos_frame]);
      fputs(pixel_buf, fp);
    }
    fputs("]\n", fp);
  }

  fputs("\nInterpolation kernels:\n", fp);
  char kernel_buf[8];
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      snprintf(kernel_buf, sizeof(kernel_buf), "(%1d, %1d) ",
               kernel_x[pos_coord], kernel_y[pos_coord]);
      fputs(kernel_buf, fp);
    }
    fputs("\n", fp);
  }

  fputs("\nInterpolation results:\n", fp);
  uint8_t result;
  for (int idx_y = 0; idx_y < test_config->block_height; idx_y++) {
    fputs("[", fp);
    for (int idx_x = 0; idx_x < test_config->block_width; idx_x++) {
      pos_coord = idx_x + idx_y * coord_stride;
      result = interp_pixel_plane(
          frame, frame_stride, frame_width, frame_height, coord_x[pos_coord],
          coord_y[pos_coord], av1_sub_pel_filters_8smooth, 8);
      snprintf(pixel_buf, sizeof(pixel_buf), "%3d, ", result);
      fputs(pixel_buf, fp);
    }
    fputs("]\n", fp);
  }

  fclose(fp);
}

// Do motion searches and write the result into txt files
void do_motion_searches(int block_x, int block_y, int block_width,
                        int block_height, const uint8_t *cur_frame,
                        const uint8_t *ref_frame, int frame_stride,
                        int frame_width, int frame_height, int search_range,
                        int block_cnt, const TestConfig *test_config,
                        SearchResult *result, double *time_used) {
  SphereMV start_sphere_mv;
  start_sphere_mv.phi = 0;
  start_sphere_mv.theta = 0;
  SphereMV best_sphere_mv;
  PlaneMV best_plane_mv;

  double block_phi;
  double block_theta;
  double plane_phi;
  double plane_theta;

  mkdir(output_filepaths[test_config->filename], S_IRWXU | S_IRWXG | S_IROTH);

  write_plane_data("cur", test_config, cur_frame, frame_stride, frame_width,
                   frame_height, block_x, block_y, block_cnt, 0);

  result->sad_plane_bruteforce = av1_motion_search_brute_force_plane(
      block_x, block_y, block_width, block_height, cur_frame, ref_frame,
      frame_stride, frame_width, frame_height, search_range, &best_plane_mv);

  write_plane_data("pl_brt", test_config, ref_frame, frame_stride, frame_width,
                   frame_height, block_x + best_plane_mv.x,
                   block_y + best_plane_mv.y, block_cnt,
                   result->sad_plane_bruteforce);

  // Use the result of plane bruteforce as the start of ERP search
  av1_plane_to_sphere_erp(block_x + best_plane_mv.x, block_y + best_plane_mv.y,
                          frame_width, frame_height, &plane_phi, &plane_theta);
  av1_plane_to_sphere_erp(block_x, block_y, frame_width, frame_height,
                          &block_phi, &block_theta);
  start_sphere_mv.phi = plane_phi - block_phi;
  start_sphere_mv.theta = plane_theta - block_theta;

  result->sad_plane_diamond = av1_motion_search_diamond_plane(
      block_x, block_y, block_width, block_height, cur_frame, ref_frame,
      frame_stride, frame_width, frame_height, search_range, &best_plane_mv);

  write_plane_data("pl_dia", test_config, ref_frame, frame_stride, frame_width,
                   frame_height, block_x + best_plane_mv.x,
                   block_y + best_plane_mv.y, block_cnt,
                   result->sad_plane_diamond);

  // result->sad_erp_bruteforce_p2p = av1_motion_search_brute_force_erp(
  //     block_x, block_y, block_width, block_height, cur_frame, ref_frame,
  //     frame_stride, frame_width, frame_height, search_range,
  //     &best_sphere_mv);

  // result->sad_erp_bruteforce_interp =
  // av1_motion_search_brute_force_erp_interp(
  //     block_x, block_y, block_width, block_height, cur_frame, ref_frame,
  //     frame_stride, frame_width, frame_height, search_range,
  //     &best_sphere_mv);

  result->sad_erp_diamond_p2p = av1_motion_search_diamond_erp(
      block_x, block_y, block_width, block_height, cur_frame, ref_frame,
      frame_stride, frame_width, frame_height, search_range, &start_sphere_mv,
      &best_sphere_mv);

  write_erp_data("erp_dia", test_config, ref_frame, frame_stride, frame_width,
                 frame_height, block_x, block_y, block_cnt, &best_sphere_mv,
                 result->sad_erp_diamond_p2p);

  clock_t start, end;
  // double cpu_time_used;
  start = clock();
  result->sad_erp_diamond_interp = av1_motion_search_diamond_erp_interp(
      block_x, block_y, block_width, block_height, cur_frame, ref_frame,
      frame_stride, frame_width, frame_height, search_range, &start_sphere_mv,
      &best_sphere_mv);

  end = clock();
  *time_used += ((double)(end - start)) / CLOCKS_PER_SEC;
  // printf("Processing time: %f second(s)\n ", time_used);

  write_erp_interp_data("erp_dia_interp", test_config, ref_frame, frame_stride,
                        frame_width, frame_height, block_x, block_y, block_cnt,
                        &best_sphere_mv, result->sad_erp_diamond_interp);
}

static void update_analyze_data(const SearchResult *result, AnalyzeData *data) {
  if (result->sad_erp_diamond_interp < result->sad_erp_diamond_p2p) {
    data->cnt_erp_d_interp_beat_erp_d_p2p++;
  } else if (result->sad_erp_diamond_interp > result->sad_erp_diamond_p2p) {
    data->cnt_erp_d_p2p_beat_erp_d_interp++;
  } else {
    data->cnt_erp_d_interp_equ_erp_d_p2p++;
  }

  if (result->sad_erp_diamond_interp < result->sad_plane_diamond) {
    data->cnt_erp_d_interp_beat_plane_d++;
  } else if (result->sad_erp_diamond_interp > result->sad_plane_diamond) {
    data->cnt_plane_d_beat_erp_d_interp++;
  } else {
    data->cnt_erp_d_interp_equ_plane_d++;
  }

  if (result->sad_erp_diamond_interp < result->sad_plane_bruteforce) {
    data->cnt_erp_d_interp_beat_plane_br++;
  } else if (result->sad_erp_diamond_interp > result->sad_plane_diamond) {
    data->cnt_plane_br_beat_erp_d_interp++;
  } else {
    data->cnt_erp_d_interp_equ_plane_d++;
  }

  if (result->sad_erp_diamond_p2p < result->sad_plane_diamond) {
    data->cnt_erp_d_p2p_beat_plane_d++;
  } else if (result->sad_erp_diamond_p2p > result->sad_plane_diamond) {
    data->cnt_plane_d_beat_erp_d_p2p++;
  } else {
    data->cnt_erp_d_p2p_equ_plane_d++;
  }

  if (result->sad_erp_diamond_p2p < result->sad_plane_bruteforce) {
    data->cnt_erp_d_p2p_beat_plane_br++;
  } else if (result->sad_erp_diamond_p2p > result->sad_plane_bruteforce) {
    data->cnt_plane_br_beat_erp_d_p2p++;
  } else {
    data->cnt_erp_d_p2p_equ_plane_br++;
  }

  if (result->sad_plane_diamond == result->sad_plane_bruteforce) {
    data->cnt_plane_d_equ_plane_br++;
  }

  data->avg_sad_plane_bruteforce += result->sad_plane_bruteforce;
  data->avg_sad_plane_diamond += result->sad_plane_diamond;
  data->avg_sad_erp_diamond_p2p += result->sad_erp_diamond_p2p;
  data->avg_sad_erp_bruteforce_interp += result->sad_erp_bruteforce_interp;
  data->avg_sad_erp_bruteforce_p2p += result->sad_erp_bruteforce_p2p;
  data->avg_sad_erp_diamond_interp += result->sad_erp_diamond_interp;
}

static void update_avg_analyze_data(AnalyzeData *data) {
  data->avg_sad_plane_bruteforce /= data->block_count;
  data->avg_sad_plane_diamond /= data->block_count;
  data->avg_sad_erp_diamond_p2p /= data->block_count;
  data->avg_sad_erp_bruteforce_interp /= data->block_count;
  data->avg_sad_erp_bruteforce_p2p /= data->block_count;
  data->avg_sad_erp_diamond_interp /= data->block_count;
}

static void whole_frame_motion_search_analyze(const TestConfig *test_config) {
  y4m_input_t *handle = NULL;
  config_t config;
  ASSERT_EQ(0, open_file_y4m(filepaths[test_config->filename],
                             (handle_t *)(&handle), &config));

  picture_t cur_pic;
  picture_t ref_pic;
  uint8_t temp;
  cur_pic.img.plane[0] =
      (uint8_t *)malloc(handle->width * handle->height * sizeof(temp));
  cur_pic.img.plane[1] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));
  cur_pic.img.plane[2] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));
  ref_pic.img.plane[0] =
      (uint8_t *)malloc(handle->width * handle->height * sizeof(temp));
  ref_pic.img.plane[1] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));
  ref_pic.img.plane[2] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));

  read_frame_y4m((handle_t)handle, &cur_pic, test_config->cur_frame_no);
  read_frame_y4m((handle_t)handle, &ref_pic, test_config->ref_frame_no);

  uint8_t *cur_frame = cur_pic.img.plane[0];
  uint8_t *ref_frame = ref_pic.img.plane[0];

  int block_x;
  int block_y;
  int block_step_x = handle->width / 10;
  int block_step_y = handle->height / 10;

  AnalyzeData data;
  init_analyze_data(&data);

  SearchResult search_result;

  double erp_dia_interp_time = 0;

  // clock_t start, end;
  // double cpu_time_used;

  printf("File path: %s\n", filepaths[test_config->filename]);
  printf("Current frame: %d, reference frame: %d\n", test_config->cur_frame_no,
         test_config->ref_frame_no);
  printf("Block size: %dx%d, search range: %d\n", test_config->block_width,
         test_config->block_height, test_config->search_range);

  // start = clock();
  for (block_y = 0; block_y + test_config->block_height < handle->height;
       block_y += block_step_y) {
    for (block_x = 0; block_x + test_config->block_width < handle->width;
         block_x += block_step_x) {
      do_motion_searches(block_x, block_y, test_config->block_width,
                         test_config->block_height, cur_frame, ref_frame,
                         handle->width, handle->width, handle->height,
                         test_config->search_range, data.block_count + 1,
                         test_config, &search_result, &erp_dia_interp_time);
      data.block_count++;
      update_analyze_data(&search_result, &data);
    }
  }  // for idx_y

  // end = clock();
  // cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Processing time: %f second(s)\n ", erp_dia_interp_time);
  printf("Total block count: %d\n", data.block_count);

  update_avg_analyze_data(&data);

  printf("Average plane bruteforce SAD: %d\n", data.avg_sad_plane_bruteforce);
  printf("Average plane diamond SAD: %d\n", data.avg_sad_plane_diamond);
  printf("Pixel to pixel average equirectangle bruteforce SAD: %d\n",
         data.avg_sad_erp_bruteforce_p2p);
  printf("Interp average equirectangle bruteforce SAD: %d\n",
         data.avg_sad_erp_bruteforce_interp);
  printf("Pixel to pixel average equirectangle diamond SAD: %d\n",
         data.avg_sad_erp_diamond_p2p);
  printf("Interp average equirectangle diamond SAD: %d\n",
         data.avg_sad_erp_diamond_interp);

  printf("Gain ERP diamond p2p  vs plane bruteforce: %f%%\n",
         100.0 *
             (data.avg_sad_plane_bruteforce - data.avg_sad_erp_diamond_p2p) /
             data.avg_sad_plane_bruteforce);
  printf("Gain interp vs plane bruteforce: %f%%\n",
         100.0 *
             (data.avg_sad_plane_bruteforce - data.avg_sad_erp_diamond_interp) /
             data.avg_sad_plane_bruteforce);
  printf("Gain interp vs plane diamond: %f%%\n",
         100.0 *
             (data.avg_sad_plane_diamond - data.avg_sad_erp_diamond_interp) /
             data.avg_sad_plane_diamond);
  printf("Gain interp vs ERP diamond p2p: %f%%\n",
         100.0 *
             (data.avg_sad_erp_diamond_p2p - data.avg_sad_erp_diamond_interp) /
             data.avg_sad_erp_diamond_p2p);
  // printf("Gain interp vs pixel to pixel diamond: %f%%\n",
  //        100.0 *
  //            (data.avg_sad_erp_diamond_p2p -
  //            data.avg_sad_erp_diamond_interp) /
  //            data.avg_sad_erp_diamond_p2p);
  printf(
      "%d (%f%%) blocks has less SAD with burteforce ERP p2p comparing to "
      "burteforce plane p2p\n",
      data.cnt_erp_d_p2p_beat_plane_br,
      100.0 * data.cnt_erp_d_p2p_beat_plane_br / data.block_count);
  printf(
      "%d (%f%%) blocks has less SAD with diamond ERP p2p comparing to "
      "burteforce diamond p2p\n",
      data.cnt_erp_d_p2p_beat_plane_d,
      100.0 * data.cnt_erp_d_p2p_beat_plane_d / data.block_count);
  printf(
      "%d (%f%%) blocks has less SAD with burteforce ERP interpolation "
      "comparing to "
      "plane burteforce p2p\n",
      data.cnt_erp_d_interp_beat_plane_br,
      100.0 * data.cnt_erp_d_interp_beat_plane_br / data.block_count);
  printf(
      "%d (%f%%) blocks has less SAD with diamond ERP interpolation "
      "comparing "
      "to "
      "diamond ERP p2p\n\n",
      data.cnt_erp_d_interp_beat_erp_d_p2p,
      100.0 * data.cnt_erp_d_interp_beat_erp_d_p2p / data.block_count);

  free_frames(handle, &cur_pic, &ref_pic);
}

TEST(SphericalMappingTest, EquiWholeFrameAnalyzeTest) {
  TestConfig configs[] = {
    { ARIALCITY960, 60, 65, 1637, 665, 32, 32, 20 },
    { BALBOA960, 180, 185, 514, 187, 32, 32, 20 },
    { BRANCASTLE960, 60, 65, 914, 398, 32, 32, 20 },
    { BROADWAY960, 80, 85, 1518, 570, 32, 32, 20 },
    { GASLAMP960, 210, 215, 1150, 500, 32, 32, 20 },
    { HARBOR960, 1, 5, 50, 490, 32, 32, 20 },
    { KITEFLITE960, 15, 20, 520, 425, 32, 32, 20 },
    { LANDING960, 90, 95, 828, 212, 32, 32, 20 },
    { POLEVAULT960, 30, 35, 328, 800, 32, 32, 20 },
    { TROLLEY960, 90, 95, 1649, 387, 32, 32, 20 },
  };

  // whole_frame_motion_search_analyze(&configs[4]);
  for (int i = 0; i < 10; i++) {
    whole_frame_motion_search_analyze(&configs[i]);
  }
}

void write_bmp_multi_highlight(char *filename, int width, int height,
                               uint8_t *y, int *highlight1, int *highlight2,
                               int *highlight3, int *highlight4,
                               int *highlight5, int *highlight6,
                               int *highlight7, int *highlight8,
                               int *highlight9, int block_width,
                               int block_height, int block_stride) {
  int Y;
  int length = 3 * width * height;

  char tag[] = { 'B', 'M' };
  int header[] = {
    0,                                       // File size... update at the end.
    0,        0x36, 0x28, width,    height,  // Image dimensions in pixels

    0x180001, 0,    0,    0x002e23, 0x002e23, 0, 0,
  };
  // Update file size: just the sum of the sizes of the arrays
  // we write to disk.
  header[0] = sizeof(tag) + sizeof(header) + length;

  FILE *fp = fopen(filename, "w+");
  fwrite(&tag, sizeof(tag), 1, fp);
  fwrite(&header, sizeof(header), 1, fp);

  uint8_t *lume = (uint8_t *)malloc(width * height * sizeof(*y));

  for (int i = 0; i < width * height; i++) {
    lume[i] = y[i] * 0.5;
  }

  for (int j = 0; j < block_height; j++) {
    for (int i = 0; i < block_width; i++) {
      lume[highlight1[j * block_stride + i]] *= 2;
      lume[highlight2[j * block_stride + i]] *= 2;
      lume[highlight3[j * block_stride + i]] *= 2;
      lume[highlight4[j * block_stride + i]] *= 2;
      lume[highlight5[j * block_stride + i]] *= 2;
      lume[highlight6[j * block_stride + i]] *= 2;
      lume[highlight7[j * block_stride + i]] *= 2;
      lume[highlight8[j * block_stride + i]] *= 2;
      lume[highlight9[j * block_stride + i]] *= 2;
    }
  }

  for (int j = height - 1; j >= 0; j--) {
    for (int i = 0; i < width; i++) {
      Y = lume[i + j * width];

      fwrite(&Y, sizeof(*y), 1, fp);
      fwrite(&Y, sizeof(*y), 1, fp);
      fwrite(&Y, sizeof(*y), 1, fp);
    }
  }

  fclose(fp);
  free(lume);
}

// Verify blocks get distorted on the plane after moving on sphere
TEST(SphericalMappingTest, EquiDistortionTest) {
  char *filename = "360test.y4m";
  y4m_input_t *handle = NULL;
  config_t config;
  ASSERT_EQ(0, open_file_y4m(filename, (handle_t *)(&handle), &config));

  picture_t cur_pic;
  picture_t ref_pic;
  uint8_t temp;
  cur_pic.img.plane[0] =
      (uint8_t *)malloc(handle->width * handle->height * sizeof(temp));
  cur_pic.img.plane[1] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));
  cur_pic.img.plane[2] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));
  ref_pic.img.plane[0] =
      (uint8_t *)malloc(handle->width * handle->height * sizeof(temp));
  ref_pic.img.plane[1] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));
  ref_pic.img.plane[2] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));

  int cur_frame_no = 110;
  read_frame_y4m((handle_t)handle, &cur_pic, cur_frame_no);

  SphereMV best_sphere_mv;
  int block_x = 0;
  int block_y = 0;
  int block_width = 128;
  int block_height = 128;

  int cur_block[128 * 128];
  int up_block[128 * 128];
  int down_block[128 * 128];
  int left_block[128 * 128];
  int right_block[128 * 128];
  int up_left_block[128 * 128];
  int down_left_block[128 * 128];
  int up_right_block[128 * 128];
  int down_right_block[128 * 128];
  const int block_stride = 128;

  get_pred_block_idx(block_x, block_y, block_width, block_height, -0.25 * pi, 0,
                     handle->width, handle->width, handle->height, block_stride,
                     cur_block, up_block);
  get_pred_block_idx(block_x, block_y, block_width, block_height, 0.25 * pi, 0,
                     handle->width, handle->width, handle->height, block_stride,
                     cur_block, down_block);
  get_pred_block_idx(block_x, block_y, block_width, block_height, 0, -0.25 * pi,
                     handle->width, handle->width, handle->height, block_stride,
                     cur_block, left_block);
  get_pred_block_idx(block_x, block_y, block_width, block_height, 0, 0.25 * pi,
                     handle->width, handle->width, handle->height, block_stride,
                     cur_block, right_block);
  get_pred_block_idx(block_x, block_y, block_width, block_height, -0.25 * pi,
                     -0.25 * pi, handle->width, handle->width, handle->height,
                     block_stride, cur_block, up_left_block);
  get_pred_block_idx(block_x, block_y, block_width, block_height, 0.25 * pi,
                     -0.25 * pi, handle->width, handle->width, handle->height,
                     block_stride, cur_block, down_left_block);
  get_pred_block_idx(block_x, block_y, block_width, block_height, -0.25 * pi,
                     0.25 * pi, handle->width, handle->width, handle->height,
                     block_stride, cur_block, up_right_block);
  get_pred_block_idx(block_x, block_y, block_width, block_height, 0.25 * pi,
                     0.25 * pi, handle->width, handle->width, handle->height,
                     block_stride, cur_block, down_right_block);
  write_bmp_multi_highlight(
      "blcok_distortion.bmp", handle->width, handle->height,
      cur_pic.img.plane[0], cur_block, up_block, down_block, left_block,
      right_block, up_left_block, down_left_block, up_right_block,
      down_right_block, block_width, block_height, block_stride);

  free(cur_pic.img.plane[0]);
  free(cur_pic.img.plane[1]);
  free(cur_pic.img.plane[2]);
  free(ref_pic.img.plane[0]);
  free(ref_pic.img.plane[1]);
  free(ref_pic.img.plane[2]);

  close_file_y4m((handle_t)handle);
}

static void row_blocks_motion_search_test(const TestConfig *test_config) {
  y4m_input_t *handle = NULL;
  config_t config;
  ASSERT_EQ(0, open_file_y4m(filepaths[test_config->filename],
                             (handle_t *)(&handle), &config));

  picture_t cur_pic;
  picture_t ref_pic;
  uint8_t temp;
  cur_pic.img.plane[0] =
      (uint8_t *)malloc(handle->width * handle->height * sizeof(temp));
  cur_pic.img.plane[1] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));
  cur_pic.img.plane[2] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));
  ref_pic.img.plane[0] =
      (uint8_t *)malloc(handle->width * handle->height * sizeof(temp));
  ref_pic.img.plane[1] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));
  ref_pic.img.plane[2] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));

  read_frame_y4m((handle_t)handle, &cur_pic, test_config->cur_frame_no);
  read_frame_y4m((handle_t)handle, &ref_pic, test_config->ref_frame_no);

  SphereMV start_sphere_mv;
  start_sphere_mv.phi = 0;
  start_sphere_mv.theta = 0;
  SphereMV best_sphere_mv;
  PlaneMV best_plane_mv;

  int block_x;
  int block_y;
  int block_step_x = handle->width / 5;
  int block_step_y = handle->height / 5;

  int sad_plane_bruteforce = 0;
  int sad_plane_diamond = 0;
  int sad_erp_bruteforce_p2p = 0;
  int sad_erp_bruteforce_interp = 0;
  int sad_erp_diamond_interp = 0;
  int sad_erp_diamond_p2p = 0;

  int block_count = 0;

  clock_t start, end;
  double cpu_time_used;

  double block_phi;
  double block_theta;
  double plane_phi;
  double plane_theta;

  printf("File path: %s\n", filepaths[test_config->filename]);
  printf("Current frame: %d, reference frame: %d\n", test_config->cur_frame_no,
         test_config->ref_frame_no);
  printf("Block size: %dx%d, search range: %d\n", test_config->block_width,
         test_config->block_height, test_config->search_range);
  printf(
      "(block_x, block_y): SAD plane bruteforce, SAD ERP bruteforce(gain), "
      "SAD "
      "ERP diamond(gain)\n");

  for (block_y = 0; block_y + test_config->block_height < handle->height;
       block_y += block_step_y) {
    for (block_x = 0; block_x + test_config->block_width < handle->width;
         block_x += block_step_x) {
      sad_plane_bruteforce = 0;
      sad_plane_diamond = 0;
      sad_erp_bruteforce_p2p = 0;
      sad_erp_bruteforce_interp = 0;
      sad_erp_diamond_interp = 0;
      sad_erp_diamond_p2p = 0;

      // start = clock();

      sad_plane_bruteforce = av1_motion_search_brute_force_plane(
          block_x, block_y, test_config->block_width, test_config->block_height,
          cur_pic.img.plane[0], ref_pic.img.plane[0], handle->width,
          handle->width, handle->height, test_config->search_range,
          &best_plane_mv);

      // av1_plane_to_sphere_erp(block_x + best_plane_mv.x,
      //                         block_y + best_plane_mv.y, handle->width,
      //                         handle->height, &plane_phi, &plane_theta);

      // av1_plane_to_sphere_erp(block_x, block_y, handle->width,
      // handle->height,
      //                         &block_phi, &block_theta);

      // start_sphere_mv.phi = plane_phi - block_phi;
      // start_sphere_mv.theta = plane_theta - block_theta;

      // avg_sad_plane_diamond += av1_motion_search_diamond_plane(
      //     block_x, block_y, test_config->block_width,
      //     test_config->block_height, cur_pic.img.plane[0],
      //     ref_pic.img.plane[0], handle->width, handle->width,
      //     handle->height, test_config->search_range, &best_plane_mv);

      // sad_erp_bruteforce_p2p += av1_motion_search_brute_force_erp(
      //     block_x, block_y, test_config->block_width,
      //     test_config->block_height, cur_pic.img.plane[0],
      //     ref_pic.img.plane[0], handle->width, handle->width,
      //     handle->height, test_config->search_range, &best_sphere_mv);

      // sad_erp_bruteforce_interp =
      // av1_motion_search_brute_force_erp_interp(
      //     block_x, block_y, test_config->block_width,
      //     test_config->block_height, cur_pic.img.plane[0],
      //     ref_pic.img.plane[0], handle->width, handle->width,
      //     handle->height, test_config->search_range, &best_sphere_mv);

      // sad_erp_diamond_p2p += av1_motion_search_diamond_erp(
      //     block_x, block_y, test_config->block_width,
      //     test_config->block_height, cur_pic.img.plane[0],
      //     ref_pic.img.plane[0], handle->width, handle->width,
      //     handle->height, test_config->search_range, &start_sphere_mv,
      //     &best_sphere_mv);

      // sad_erp_diamond_interp = av1_motion_search_diamond_erp_interp(
      //     block_x, block_y, test_config->block_width,
      //     test_config->block_height, cur_pic.img.plane[0],
      //     ref_pic.img.plane[0], handle->width, handle->width,
      //     handle->height, test_config->search_range, &start_sphere_mv,
      //     &best_sphere_mv);

      // end = clock();
      // cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

      // printf("Processing time: %f second(s)\n ", cpu_time_used);

      printf("(%d, %d): %d, %d(%f%%), %d(%f%%)\t", block_x, block_y,
             sad_plane_bruteforce, sad_erp_bruteforce_interp,
             100.0 * (sad_plane_bruteforce - sad_erp_bruteforce_interp) /
                 sad_plane_bruteforce,
             sad_erp_diamond_interp,
             100.0 * (sad_plane_bruteforce - sad_erp_diamond_interp) /
                 sad_plane_bruteforce);

      block_count++;
    }
    printf("\n");
  }

  printf("Total block count: %d\n", block_count);

  free(cur_pic.img.plane[0]);
  free(cur_pic.img.plane[1]);
  free(cur_pic.img.plane[2]);
  free(ref_pic.img.plane[0]);
  free(ref_pic.img.plane[1]);
  free(ref_pic.img.plane[2]);

  close_file_y4m((handle_t)handle);
}

TEST(SphericalMappingTest, EquiBlocksTest) {
  TestConfig configs[] = {
    { ARIALCITY960, 60, 65, 1637, 665, 32, 32, 20 },
    { BALBOA960, 180, 185, 514, 187, 128, 128, 20 },
    { BRANCASTLE960, 60, 65, 914, 398, 64, 64, 20 },
    { BROADWAY960, 80, 85, 1518, 570, 64, 64, 20 },
    { GASLAMP960, 210, 215, 1150, 500, 64, 64, 20 },
    { HARBOR960, 1, 5, 50, 490, 64, 64, 20 },
    { KITEFLITE960, 15, 20, 520, 425, 32, 32, 20 },
    { LANDING960, 90, 95, 828, 212, 128, 128, 20 },
    { POLEVAULT960, 30, 35, 328, 800, 128, 128, 20 },
    { TROLLEY960, 90, 95, 1649, 387, 128, 128, 20 },
  };

  for (int i = 0; i < 10; i++) {
    row_blocks_motion_search_test(&configs[i]);
    printf("\n");
  }
}

// The rest code is for data and graphs in the final report
typedef struct {
  int avg_sad_plane_bruteforce_p2p;
  int avg_sad_plane_bruteforce_interp_bilinear;
  int avg_sad_plane_bruteforce_interp_sub_pel_8;
  int avg_sad_plane_bruteforce_sub_pel_8sharp;
  int avg_sad_plane_bruteforce_sub_pel_8smooth;
  int avg_sad_plane_bruteforce_filter_search;
  int avg_sad_plane_diamond_p2p;
  int avg_sad_plane_diamond_interp_bilinear;
  int avg_sad_plane_diamond_interp_sub_pel_8;
  int avg_sad_plane_diamond_interp_sub_pel_8sharp;
  int avg_sad_plane_diamond_interp_sub_pel_8smooth;
  int avg_sad_plane_diamond_interp_filter_search;
  // int avg_sad_erp_bruteforce_p2p;
  // int avg_sad_erp_bruteforce_interp;
  int avg_sad_erp_diamond_p2p;
  int avg_sad_erp_diamond_interp_interp_bilinear;
  int avg_sad_erp_diamond_interp_interp_sub_pel_8;
  int avg_sad_erp_diamond_interp_sub_pel_8sharp;
  int avg_sad_erp_diamond_interp_sub_pel_8smooth;
  int avg_sad_erp_diamond_interp_filter_search;
  int avg_sad_erp_diamond_interp_bilinear_scale;
  int avg_sad_erp_diamond_interp_sub_pel_8_scale;
  int avg_sad_erp_diamond_interp_sub_pel_8sharp_scale;
  int avg_sad_erp_diamond_interp_sub_pel_8smooth_scale;
  int avg_sad_erp_diamond_interp_filter_search_scale;

  int *sad_plane_bruteforce_p2p;
  int *sad_plane_bruteforce_interp_bilinear;
  int *sad_plane_bruteforce_interp_sub_pel_8;
  int *sad_plane_bruteforce_sub_pel_8sharp;
  int *sad_plane_bruteforce_sub_pel_8smooth;
  int *sad_plane_bruteforce_filter_search;
  int *sad_plane_diamond_p2p;
  int *sad_plane_diamond_interp_bilinear;
  int *sad_plane_diamond_interp_sub_pel_8;
  int *sad_plane_diamond_interp_sub_pel_8sharp;
  int *sad_plane_diamond_interp_sub_pel_8smooth;
  int *sad_plane_diamond_interp_filter_search;
  // int *sad_erp_bruteforce_p2p;
  // int *sad_erp_bruteforce_interp;
  int *sad_erp_diamond_p2p;
  int *sad_erp_diamond_interp_interp_bilinear;
  int *sad_erp_diamond_interp_interp_sub_pel_8;
  int *sad_erp_diamond_interp_sub_pel_8sharp;
  int *sad_erp_diamond_interp_sub_pel_8smooth;
  int *sad_erp_diamond_interp_filter_search;
  int *sad_erp_diamond_interp_bilinear_scale;
  int *sad_erp_diamond_interp_sub_pel_8_scale;
  int *sad_erp_diamond_interp_sub_pel_8sharp_scale;
  int *sad_erp_diamond_interp_sub_pel_8smooth_scale;
  int *sad_erp_diamond_interp_filter_search_scale;

  int row_cnt;
  int col_cnt;

  double time_erp_diamond_interp_interp_bilinear;
  double time_erp_diamond_interp_interp_sub_pel_8;
  double time_erp_diamond_interp_sub_pel_8sharp;
  double time_erp_diamond_interp_sub_pel_8smooth;
  double time_erp_diamond_interp_filter_search;
  double time_erp_diamond_interp_bilinear_scale;
  double time_erp_diamond_interp_sub_pel_8_scale;
  double time_erp_diamond_interp_sub_pel_8sharp_scale;
  double time_erp_diamond_interp_sub_pel_8smooth_scale;
  double time_erp_diamond_interp_filter_search_scale;
} FinalSearchResult;

static void init_final_search_result(FinalSearchResult *result, int row_cnt,
                                     int col_cnt) {
  result->row_cnt = row_cnt;
  result->col_cnt = col_cnt;

  result->avg_sad_plane_bruteforce_p2p = 0;
  result->avg_sad_plane_bruteforce_interp_bilinear = 0;
  result->avg_sad_plane_bruteforce_interp_sub_pel_8 = 0;
  result->avg_sad_plane_bruteforce_sub_pel_8sharp = 0;
  result->avg_sad_plane_bruteforce_sub_pel_8smooth = 0;
  result->avg_sad_plane_bruteforce_filter_search = 0;
  result->avg_sad_plane_diamond_p2p = 0;
  result->avg_sad_plane_diamond_interp_bilinear = 0;
  result->avg_sad_plane_diamond_interp_sub_pel_8 = 0;
  result->avg_sad_plane_diamond_interp_sub_pel_8sharp = 0;
  result->avg_sad_plane_diamond_interp_sub_pel_8smooth = 0;
  result->avg_sad_plane_diamond_interp_filter_search = 0;
  // result->avg_sad_erp_bruteforce_p2p=0;
  // result->avg_sad_erp_bruteforce_interp=0;
  result->avg_sad_erp_diamond_p2p = 0;
  result->avg_sad_erp_diamond_interp_interp_bilinear = 0;
  result->avg_sad_erp_diamond_interp_interp_sub_pel_8 = 0;
  result->avg_sad_erp_diamond_interp_sub_pel_8sharp = 0;
  result->avg_sad_erp_diamond_interp_sub_pel_8smooth = 0;
  result->avg_sad_erp_diamond_interp_filter_search = 0;
  result->avg_sad_erp_diamond_interp_bilinear_scale = 0;
  result->avg_sad_erp_diamond_interp_sub_pel_8_scale = 0;
  result->avg_sad_erp_diamond_interp_sub_pel_8sharp_scale = 0;
  result->avg_sad_erp_diamond_interp_sub_pel_8smooth_scale = 0;
  result->avg_sad_erp_diamond_interp_filter_search_scale = 0;

  result->time_erp_diamond_interp_interp_bilinear = 0;
  result->time_erp_diamond_interp_interp_sub_pel_8 = 0;
  result->time_erp_diamond_interp_sub_pel_8sharp = 0;
  result->time_erp_diamond_interp_sub_pel_8smooth = 0;
  result->time_erp_diamond_interp_filter_search = 0;
  result->time_erp_diamond_interp_bilinear_scale = 0;
  result->time_erp_diamond_interp_sub_pel_8_scale = 0;
  result->time_erp_diamond_interp_sub_pel_8sharp_scale = 0;
  result->time_erp_diamond_interp_sub_pel_8smooth_scale = 0;
  result->time_erp_diamond_interp_filter_search_scale = 0;

  result->sad_plane_bruteforce_p2p =
      (int *)malloc(row_cnt * col_cnt * sizeof(row_cnt));
  result->sad_plane_bruteforce_interp_bilinear =
      (int *)malloc(row_cnt * col_cnt * sizeof(row_cnt));
  result->sad_plane_bruteforce_interp_sub_pel_8 =
      (int *)malloc(row_cnt * col_cnt * sizeof(row_cnt));
  result->sad_plane_bruteforce_sub_pel_8sharp =
      (int *)malloc(row_cnt * col_cnt * sizeof(row_cnt));
  result->sad_plane_bruteforce_sub_pel_8smooth =
      (int *)malloc(row_cnt * col_cnt * sizeof(row_cnt));
  result->sad_plane_bruteforce_filter_search =
      (int *)malloc(row_cnt * col_cnt * sizeof(row_cnt));
  result->sad_plane_diamond_p2p =
      (int *)malloc(row_cnt * col_cnt * sizeof(row_cnt));
  result->sad_plane_diamond_interp_bilinear =
      (int *)malloc(row_cnt * col_cnt * sizeof(row_cnt));
  result->sad_plane_diamond_interp_sub_pel_8 =
      (int *)malloc(row_cnt * col_cnt * sizeof(row_cnt));
  result->sad_plane_diamond_interp_sub_pel_8sharp =
      (int *)malloc(row_cnt * col_cnt * sizeof(row_cnt));
  result->sad_plane_diamond_interp_sub_pel_8smooth =
      (int *)malloc(row_cnt * col_cnt * sizeof(row_cnt));
  result->sad_plane_diamond_interp_filter_search =
      (int *)malloc(row_cnt * col_cnt * sizeof(row_cnt));
  // result->sad_erp_bruteforce_p2p =
  // (int *)malloc(row_cnt * col_cnt * sizeof(int));
  // result->sad_erp_bruteforce_interp =
  // (int *)malloc(row_cnt * col_cnt * sizeof(int));
  result->sad_erp_diamond_p2p =
      (int *)malloc(row_cnt * col_cnt * sizeof(row_cnt));
  result->sad_erp_diamond_interp_interp_bilinear =
      (int *)malloc(row_cnt * col_cnt * sizeof(row_cnt));
  result->sad_erp_diamond_interp_interp_sub_pel_8 =
      (int *)malloc(row_cnt * col_cnt * sizeof(row_cnt));
  result->sad_erp_diamond_interp_sub_pel_8sharp =
      (int *)malloc(row_cnt * col_cnt * sizeof(row_cnt));
  result->sad_erp_diamond_interp_sub_pel_8smooth =
      (int *)malloc(row_cnt * col_cnt * sizeof(row_cnt));
  result->sad_erp_diamond_interp_filter_search =
      (int *)malloc(row_cnt * col_cnt * sizeof(row_cnt));
  result->sad_erp_diamond_interp_bilinear_scale =
      (int *)malloc(row_cnt * col_cnt * sizeof(row_cnt));
  result->sad_erp_diamond_interp_sub_pel_8_scale =
      (int *)malloc(row_cnt * col_cnt * sizeof(row_cnt));
  result->sad_erp_diamond_interp_sub_pel_8sharp_scale =
      (int *)malloc(row_cnt * col_cnt * sizeof(row_cnt));
  result->sad_erp_diamond_interp_sub_pel_8smooth_scale =
      (int *)malloc(row_cnt * col_cnt * sizeof(row_cnt));
  result->sad_erp_diamond_interp_filter_search_scale =
      (int *)malloc(row_cnt * col_cnt * sizeof(row_cnt));
}

static void cal_avg_sad(FinalSearchResult *result) {
  int blk_cnt = result->row_cnt * result->col_cnt;
  for (int i = 0; i < blk_cnt; i++) {
    result->avg_sad_plane_bruteforce_p2p += result->sad_plane_bruteforce_p2p[i];
    result->avg_sad_plane_bruteforce_interp_bilinear +=
        result->sad_plane_bruteforce_interp_bilinear[i];
    result->avg_sad_plane_bruteforce_interp_sub_pel_8 +=
        result->sad_plane_bruteforce_interp_sub_pel_8[i];
    result->avg_sad_plane_bruteforce_sub_pel_8sharp +=
        result->sad_plane_bruteforce_sub_pel_8sharp[i];
    result->avg_sad_plane_bruteforce_sub_pel_8smooth +=
        result->sad_plane_bruteforce_sub_pel_8smooth[i];
    result->avg_sad_plane_bruteforce_filter_search +=
        result->sad_plane_bruteforce_filter_search[i];

    result->avg_sad_plane_diamond_p2p += result->sad_plane_diamond_p2p[i];
    result->avg_sad_plane_diamond_interp_bilinear +=
        result->sad_plane_diamond_interp_bilinear[i];
    result->avg_sad_plane_diamond_interp_sub_pel_8 +=
        result->sad_plane_diamond_interp_sub_pel_8[i];
    result->avg_sad_plane_diamond_interp_sub_pel_8sharp +=
        result->sad_plane_diamond_interp_sub_pel_8sharp[i];
    result->avg_sad_plane_diamond_interp_sub_pel_8smooth +=
        result->sad_plane_diamond_interp_sub_pel_8smooth[i];
    result->avg_sad_plane_diamond_interp_filter_search +=
        result->sad_plane_diamond_interp_filter_search[i];

    result->avg_sad_erp_diamond_p2p += result->sad_erp_diamond_p2p[i];
    result->avg_sad_erp_diamond_interp_interp_bilinear +=
        result->sad_erp_diamond_interp_interp_bilinear[i];
    result->avg_sad_erp_diamond_interp_interp_sub_pel_8 +=
        result->sad_erp_diamond_interp_interp_sub_pel_8[i];
    result->avg_sad_erp_diamond_interp_sub_pel_8sharp +=
        result->sad_erp_diamond_interp_sub_pel_8sharp[i];
    result->avg_sad_erp_diamond_interp_sub_pel_8smooth +=
        result->sad_erp_diamond_interp_sub_pel_8smooth[i];
    result->avg_sad_erp_diamond_interp_filter_search +=
        result->sad_erp_diamond_interp_filter_search[i];

    result->avg_sad_erp_diamond_interp_bilinear_scale +=
        result->sad_erp_diamond_interp_bilinear_scale[i];
    result->avg_sad_erp_diamond_interp_sub_pel_8_scale +=
        result->sad_erp_diamond_interp_sub_pel_8_scale[i];
    result->avg_sad_erp_diamond_interp_sub_pel_8sharp_scale +=
        result->sad_erp_diamond_interp_sub_pel_8sharp_scale[i];
    result->avg_sad_erp_diamond_interp_sub_pel_8smooth_scale +=
        result->sad_erp_diamond_interp_sub_pel_8smooth_scale[i];
    result->avg_sad_erp_diamond_interp_filter_search_scale +=
        result->sad_erp_diamond_interp_filter_search_scale[i];
  }  // for

  result->avg_sad_plane_bruteforce_p2p /= blk_cnt;
  result->avg_sad_plane_bruteforce_interp_bilinear /= blk_cnt;
  result->avg_sad_plane_bruteforce_interp_sub_pel_8 /= blk_cnt;
  result->avg_sad_plane_bruteforce_sub_pel_8sharp /= blk_cnt;
  result->avg_sad_plane_bruteforce_sub_pel_8smooth /= blk_cnt;
  result->avg_sad_plane_bruteforce_filter_search /= blk_cnt;

  result->avg_sad_plane_diamond_p2p /= blk_cnt;
  result->avg_sad_plane_diamond_interp_bilinear /= blk_cnt;
  result->avg_sad_plane_diamond_interp_sub_pel_8 /= blk_cnt;
  result->avg_sad_plane_diamond_interp_sub_pel_8sharp /= blk_cnt;
  result->avg_sad_plane_diamond_interp_sub_pel_8smooth /= blk_cnt;
  result->avg_sad_plane_diamond_interp_filter_search /= blk_cnt;

  result->avg_sad_erp_diamond_p2p /= blk_cnt;
  result->avg_sad_erp_diamond_interp_interp_bilinear /= blk_cnt;
  result->avg_sad_erp_diamond_interp_interp_sub_pel_8 /= blk_cnt;
  result->avg_sad_erp_diamond_interp_sub_pel_8sharp /= blk_cnt;
  result->avg_sad_erp_diamond_interp_sub_pel_8smooth /= blk_cnt;
  result->avg_sad_erp_diamond_interp_filter_search /= blk_cnt;

  result->avg_sad_erp_diamond_interp_bilinear_scale /= blk_cnt;
  result->avg_sad_erp_diamond_interp_sub_pel_8_scale /= blk_cnt;
  result->avg_sad_erp_diamond_interp_sub_pel_8sharp_scale /= blk_cnt;
  result->avg_sad_erp_diamond_interp_sub_pel_8smooth_scale /= blk_cnt;
  result->avg_sad_erp_diamond_interp_filter_search_scale /= blk_cnt;
}

static char *final_results_filepaths[] = {
  "./final_analyze/360test/",  "./final_analyze/AerialCity/",
  "./final_analyze/Balboa/",   "./final_analyze/BranCastle/",
  "./final_analyze/Broadway/", "./final_analyze/Gaslamp/",
  "./final_analyze/Harbor/",   "./final_analyze/KiteFlite/",
  "./final_analyze/Landing/",  "./final_analyze/PoleVault/",
  "./final_analyze/Trolley/",
};

static void write_sad_map(FILE *fp, char *label, int *map, int row_cnt,
                          int col_cnt) {
  fprintf(fp, "#        %s      #\n\n", label);

  for (int i = 0; i < row_cnt; i++) {
    for (int j = 0; j < col_cnt; j++) {
      fprintf(fp, "%d", map[j + i * col_cnt]);
      if (j < col_cnt - 1) {
        fprintf(fp, ", ");
      }
    }  // for j
    fprintf(fp, "\n");
  }  // for i
  fprintf(fp, "\n");
}

static void write_final_search_result(const TestConfig *test_config,
                                      const FinalSearchResult *result) {
  mkdir(final_results_filepaths[test_config->filename],
        S_IRWXU | S_IRWXG | S_IROTH);

  char filepath[80];
  snprintf(filepath, sizeof(filepath), "%sresults.csv",
           final_results_filepaths[test_config->filename]);
  FILE *fp = fopen(filepath, "w");

  fprintf(fp, "%s\n", filepaths[test_config->filename]);
  fprintf(fp,
          "Block at (%d, %d), size %d x %d, current frame: %d, reference "
          "frame: %d\n\n",
          test_config->block_x, test_config->block_y, test_config->block_width,
          test_config->block_height, test_config->cur_frame_no,
          test_config->ref_frame_no);

  fprintf(fp, "************************\n");
  fprintf(fp, "*     Average SADs:    *\n");
  fprintf(fp, "************************\n\n");
  fprintf(
      fp,
      "Plane Bruteforce P2P, Plane Bruteforce Bilinear, Plane Bruteforce 8 "
      "Tap, Plane Bruteforce 8 Tap Sharp, Plane Bruteforce 8 Tap Smooth, "
      "Plane Bruteforce Filter Search, "
      "Plane Diamond P2P, Plane Diamond Bilinear, Plane Diamond 8 "
      "Tap, Plane Diamond 8 Tap Sharp, Plane Diamond 8 Tap Smooth, "
      "Plane Diamond Filter Search, "
      "ERP Diamond P2P, ERP Diamond Bilinear, ERP Diamond 8 "
      "Tap, ERP Diamond 8 Tap Sharp, ERP Diamond 8 Tap Smooth, "
      "ERP Diamond Filter Search, "
      "ERP Scaled Diamond Bilinear, ERP Scaled Diamond 8 "
      "Tap, ERP Scaled Diamond 8 Tap Sharp, ERP Scaled Diamond 8 Tap Smooth, "
      "ERP Scaled Diamond Filter Search\n");
  fprintf(fp,
          "%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, "
          "%d, %d, %d, %d, %d, %d",
          result->avg_sad_plane_bruteforce_p2p,
          result->avg_sad_plane_bruteforce_interp_bilinear,
          result->avg_sad_plane_bruteforce_interp_sub_pel_8,
          result->avg_sad_plane_bruteforce_sub_pel_8sharp,
          result->avg_sad_plane_bruteforce_sub_pel_8smooth,
          result->avg_sad_plane_bruteforce_filter_search,
          result->avg_sad_plane_diamond_p2p,
          result->avg_sad_plane_diamond_interp_bilinear,
          result->avg_sad_plane_diamond_interp_sub_pel_8,
          result->avg_sad_plane_diamond_interp_sub_pel_8sharp,
          result->avg_sad_plane_diamond_interp_sub_pel_8smooth,
          result->avg_sad_plane_diamond_interp_filter_search,
          result->avg_sad_erp_diamond_p2p,
          result->avg_sad_erp_diamond_interp_interp_bilinear,
          result->avg_sad_erp_diamond_interp_interp_sub_pel_8,
          result->avg_sad_erp_diamond_interp_sub_pel_8sharp,
          result->avg_sad_erp_diamond_interp_sub_pel_8smooth,
          result->avg_sad_erp_diamond_interp_filter_search,
          result->avg_sad_erp_diamond_interp_bilinear_scale,
          result->avg_sad_erp_diamond_interp_sub_pel_8_scale,
          result->avg_sad_erp_diamond_interp_sub_pel_8sharp_scale,
          result->avg_sad_erp_diamond_interp_sub_pel_8smooth_scale,
          result->avg_sad_erp_diamond_interp_filter_search_scale);
  fprintf(fp, "\n\n");

  fprintf(fp, "************************\n");
  fprintf(fp, "*     Process time:    *\n");
  fprintf(fp, "************************\n\n");
  fprintf(
      fp,
      "ERP Diamond Bilinear, ERP Diamond 8 "
      "Tap, ERP Diamond 8 Tap Sharp, ERP Diamond 8 Tap Smooth, "
      "ERP Diamond Filter Search, "
      "ERP Scaled Diamond Bilinear, ERP Scaled Diamond 8 "
      "Tap, ERP Scaled Diamond 8 Tap Sharp, ERP Scaled Diamond 8 Tap Smooth, "
      "ERP Scaled Diamond Filter Search\n");
  fprintf(fp, "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f",
          result->time_erp_diamond_interp_interp_bilinear,
          result->time_erp_diamond_interp_interp_sub_pel_8,
          result->time_erp_diamond_interp_sub_pel_8sharp,
          result->time_erp_diamond_interp_sub_pel_8smooth,
          result->time_erp_diamond_interp_filter_search,
          result->time_erp_diamond_interp_bilinear_scale,
          result->time_erp_diamond_interp_sub_pel_8_scale,
          result->time_erp_diamond_interp_sub_pel_8sharp_scale,
          result->time_erp_diamond_interp_sub_pel_8smooth_scale,
          result->time_erp_diamond_interp_filter_search_scale);
  fprintf(fp, "\n\n");

  fprintf(fp, "########################\n");
  fprintf(fp, "#        SAD map:      #\n");
  fprintf(fp, "########################\n\n");

  write_sad_map(fp, "Plane Bruteforce P2P", result->sad_plane_bruteforce_p2p,
                result->row_cnt, result->col_cnt);
  write_sad_map(fp, "Plane Bruteforce Bilinear",
                result->sad_plane_bruteforce_interp_bilinear, result->row_cnt,
                result->col_cnt);
  write_sad_map(fp, "Plane Bruteforce 8 Tap",
                result->sad_plane_bruteforce_interp_sub_pel_8, result->row_cnt,
                result->col_cnt);
  write_sad_map(fp, "Plane Bruteforce 8 Tap Sharp",
                result->sad_plane_bruteforce_sub_pel_8sharp, result->row_cnt,
                result->col_cnt);
  write_sad_map(fp, "Plane Bruteforce 8 Tap Smooth",
                result->sad_plane_bruteforce_sub_pel_8smooth, result->row_cnt,
                result->col_cnt);
  write_sad_map(fp, "Plane Bruteforce Filter Search",
                result->sad_plane_bruteforce_filter_search, result->row_cnt,
                result->col_cnt);
  write_sad_map(fp, "Plane Diamond P2P", result->sad_plane_diamond_p2p,
                result->row_cnt, result->col_cnt);
  write_sad_map(fp, "Plane Diamond Bilinear",
                result->sad_plane_diamond_interp_bilinear, result->row_cnt,
                result->col_cnt);
  write_sad_map(fp, "Plane Diamond 8 Tap",
                result->sad_plane_diamond_interp_sub_pel_8, result->row_cnt,
                result->col_cnt);
  write_sad_map(fp, "Plane Diamond 8 Tap Sharp",
                result->sad_plane_diamond_interp_sub_pel_8sharp,
                result->row_cnt, result->col_cnt);
  write_sad_map(fp, "Plane Diamond 8 Tap Smooth",
                result->sad_plane_diamond_interp_sub_pel_8smooth,
                result->row_cnt, result->col_cnt);
  write_sad_map(fp, "Plane Diamond Filter Search",
                result->sad_plane_diamond_interp_filter_search, result->row_cnt,
                result->col_cnt);
  write_sad_map(fp, "ERP Diamond P2P", result->sad_erp_diamond_p2p,
                result->row_cnt, result->col_cnt);
  write_sad_map(fp, "ERP Diamond Bilinear",
                result->sad_erp_diamond_interp_interp_bilinear, result->row_cnt,
                result->col_cnt);
  write_sad_map(fp, "ERP Diamond 8 Tap",
                result->sad_erp_diamond_interp_interp_sub_pel_8,
                result->row_cnt, result->col_cnt);
  write_sad_map(fp, "ERP Diamond 8 Tap Sharp",
                result->sad_erp_diamond_interp_sub_pel_8sharp, result->row_cnt,
                result->col_cnt);
  write_sad_map(fp, "ERP Diamond 8 Tap Smooth",
                result->sad_erp_diamond_interp_sub_pel_8smooth, result->row_cnt,
                result->col_cnt);
  write_sad_map(fp, "ERP Diamond Filter Search",
                result->sad_erp_diamond_interp_filter_search, result->row_cnt,
                result->col_cnt);
  write_sad_map(fp, "ERP Scaled Diamond Bilinear",
                result->sad_erp_diamond_interp_bilinear_scale, result->row_cnt,
                result->col_cnt);
  write_sad_map(fp, "ERP Scaled Diamond 8 Tap",
                result->sad_erp_diamond_interp_sub_pel_8_scale, result->row_cnt,
                result->col_cnt);
  write_sad_map(fp, "ERP Scaled Diamond 8 Tap Sharp",
                result->sad_erp_diamond_interp_sub_pel_8sharp_scale,
                result->row_cnt, result->col_cnt);
  write_sad_map(fp, "ERP Scaled Diamond 8 Tap Smooth",
                result->sad_erp_diamond_interp_sub_pel_8smooth_scale,
                result->row_cnt, result->col_cnt);
  write_sad_map(fp, "ERP Scaled Diamond Filter Search",
                result->sad_erp_diamond_interp_filter_search_scale,
                result->row_cnt, result->col_cnt);

  fclose(fp);
}

static void free_final_search_result(FinalSearchResult *result) {
  free(result->sad_plane_bruteforce_p2p);
  free(result->sad_plane_bruteforce_interp_bilinear);
  free(result->sad_plane_bruteforce_interp_sub_pel_8);
  free(result->sad_plane_bruteforce_sub_pel_8sharp);
  free(result->sad_plane_bruteforce_sub_pel_8smooth);
  free(result->sad_plane_bruteforce_filter_search);
  free(result->sad_plane_diamond_p2p);
  free(result->sad_plane_diamond_interp_bilinear);
  free(result->sad_plane_diamond_interp_sub_pel_8);
  free(result->sad_plane_diamond_interp_sub_pel_8sharp);
  free(result->sad_plane_diamond_interp_sub_pel_8smooth);
  free(result->sad_plane_diamond_interp_filter_search);
  // free(result->sad_erp_bruteforce_p2p);
  // free(result->sad_erp_bruteforce_interp);
  free(result->sad_erp_diamond_p2p);
  free(result->sad_erp_diamond_interp_interp_bilinear);
  free(result->sad_erp_diamond_interp_interp_sub_pel_8);
  free(result->sad_erp_diamond_interp_sub_pel_8sharp);
  free(result->sad_erp_diamond_interp_sub_pel_8smooth);
  free(result->sad_erp_diamond_interp_filter_search);
  free(result->sad_erp_diamond_interp_bilinear_scale);
  free(result->sad_erp_diamond_interp_sub_pel_8_scale);
  free(result->sad_erp_diamond_interp_sub_pel_8sharp_scale);
  free(result->sad_erp_diamond_interp_sub_pel_8smooth_scale);
  free(result->sad_erp_diamond_interp_filter_search_scale);
}

void do_final_motion_searches(int block_x, int block_y, int block_width,
                              int block_height, const uint8_t *cur_frame,
                              const uint8_t *ref_frame, int frame_stride,
                              int frame_width, int frame_height,
                              int search_range, int block_cnt,
                              const TestConfig *test_config,
                              FinalSearchResult *result) {
  SphereMV start_sphere_mv;
  start_sphere_mv.phi = 0;
  start_sphere_mv.theta = 0;
  SphereMV best_sphere_mv;
  PlaneMV best_plane_mv;
  PlaneMV start_plane_mv;

  double block_phi;
  double block_theta;
  double plane_phi;
  double plane_theta;

  clock_t start, end;

  write_plane_data("cur", test_config, cur_frame, frame_stride, frame_width,
                   frame_height, block_x, block_y, block_cnt, 0);

  result->sad_plane_bruteforce_p2p[block_cnt] =
      av1_motion_search_brute_force_plane(
          block_x, block_y, block_width, block_height, cur_frame, ref_frame,
          frame_stride, frame_width, frame_height, search_range,
          &best_plane_mv);

  write_plane_data("pl_brt", test_config, ref_frame, frame_stride, frame_width,
                   frame_height, block_x + best_plane_mv.x,
                   block_y + best_plane_mv.y, block_cnt,
                   result->sad_plane_bruteforce_p2p[block_cnt]);

  // Use the result of plane bruteforce as the start of ERP search
  start_plane_mv.x = best_plane_mv.x;
  start_plane_mv.y = best_plane_mv.y;
  av1_plane_to_sphere_erp(block_x + best_plane_mv.x, block_y + best_plane_mv.y,
                          frame_width, frame_height, &plane_phi, &plane_theta);
  av1_plane_to_sphere_erp(block_x, block_y, frame_width, frame_height,
                          &block_phi, &block_theta);
  start_sphere_mv.phi = plane_phi - block_phi;
  start_sphere_mv.theta = plane_theta - block_theta;

  result->sad_plane_bruteforce_interp_bilinear[block_cnt] =
      av1_motion_search_brute_force_plane_interp_bilinear(
          block_x, block_y, block_width, block_height, cur_frame, ref_frame,
          frame_stride, frame_width, frame_height, search_range,
          &start_plane_mv, &best_plane_mv);
  result->sad_plane_bruteforce_interp_sub_pel_8[block_cnt] =
      av1_motion_search_brute_force_plane_interp_sub_pel_8(
          block_x, block_y, block_width, block_height, cur_frame, ref_frame,
          frame_stride, frame_width, frame_height, search_range,
          &start_plane_mv, &best_plane_mv);
  result->sad_plane_bruteforce_sub_pel_8sharp[block_cnt] =
      av1_motion_search_brute_force_plane_interp_sub_pel_8sharp(
          block_x, block_y, block_width, block_height, cur_frame, ref_frame,
          frame_stride, frame_width, frame_height, search_range,
          &start_plane_mv, &best_plane_mv);
  result->sad_plane_bruteforce_sub_pel_8smooth[block_cnt] =
      av1_motion_search_brute_force_plane_interp_sub_pel_8smooth(
          block_x, block_y, block_width, block_height, cur_frame, ref_frame,
          frame_stride, frame_width, frame_height, search_range,
          &start_plane_mv, &best_plane_mv);
  result->sad_plane_bruteforce_filter_search[block_cnt] =
      av1_motion_search_brute_force_plane_interp_filter_search(
          block_x, block_y, block_width, block_height, cur_frame, ref_frame,
          frame_stride, frame_width, frame_height, search_range,
          &start_plane_mv, &best_plane_mv);

  result->sad_plane_diamond_p2p[block_cnt] = av1_motion_search_diamond_plane(
      block_x, block_y, block_width, block_height, cur_frame, ref_frame,
      frame_stride, frame_width, frame_height, search_range, &best_plane_mv);
  result->sad_plane_diamond_interp_bilinear[block_cnt] =
      av1_motion_search_diamond_plane_interp_bilinear(
          block_x, block_y, block_width, block_height, cur_frame, ref_frame,
          frame_stride, frame_width, frame_height, search_range,
          &best_plane_mv);
  result->sad_plane_diamond_interp_sub_pel_8[block_cnt] =
      av1_motion_search_diamond_plane_interp_sub_pel_8(
          block_x, block_y, block_width, block_height, cur_frame, ref_frame,
          frame_stride, frame_width, frame_height, search_range,
          &best_plane_mv);
  result->sad_plane_diamond_interp_sub_pel_8sharp[block_cnt] =
      av1_motion_search_diamond_plane_interp_sub_pel_8sharp(
          block_x, block_y, block_width, block_height, cur_frame, ref_frame,
          frame_stride, frame_width, frame_height, search_range,
          &best_plane_mv);
  result->sad_plane_diamond_interp_sub_pel_8smooth[block_cnt] =
      av1_motion_search_diamond_plane_interp_sub_pel_8smooth(
          block_x, block_y, block_width, block_height, cur_frame, ref_frame,
          frame_stride, frame_width, frame_height, search_range,
          &best_plane_mv);
  result->sad_plane_diamond_interp_filter_search[block_cnt] =
      av1_motion_search_diamond_plane_interp_filter_search(
          block_x, block_y, block_width, block_height, cur_frame, ref_frame,
          frame_stride, frame_width, frame_height, search_range,
          &best_plane_mv);

  result->sad_erp_diamond_p2p[block_cnt] = av1_motion_search_diamond_erp(
      block_x, block_y, block_width, block_height, cur_frame, ref_frame,
      frame_stride, frame_width, frame_height, search_range, &start_sphere_mv,
      &best_sphere_mv);
  write_erp_data("erp_dia", test_config, ref_frame, frame_stride, frame_width,
                 frame_height, block_x, block_y, block_cnt, &best_sphere_mv,
                 result->sad_erp_diamond_p2p[block_cnt]);

  start = clock();
  result->sad_erp_diamond_interp_interp_bilinear[block_cnt] =
      av1_motion_search_diamond_erp_interp_bilinear(
          block_x, block_y, block_width, block_height, cur_frame, ref_frame,
          frame_stride, frame_width, frame_height, search_range,
          &start_sphere_mv, &best_sphere_mv);
  end = clock();
  result->time_erp_diamond_interp_interp_bilinear +=
      ((double)(end - start)) / CLOCKS_PER_SEC;
  write_erp_interp_data_bilinear(
      "erp_dia_interp_bilinear", test_config, ref_frame, frame_stride,
      frame_width, frame_height, block_x, block_y, block_cnt, &best_sphere_mv,
      result->sad_erp_diamond_interp_interp_bilinear[block_cnt]);
  start = clock();
  result->sad_erp_diamond_interp_interp_sub_pel_8[block_cnt] =
      av1_motion_search_diamond_erp_interp_sub_pel_8(
          block_x, block_y, block_width, block_height, cur_frame, ref_frame,
          frame_stride, frame_width, frame_height, search_range,
          &start_sphere_mv, &best_sphere_mv);
  end = clock();
  result->time_erp_diamond_interp_interp_sub_pel_8 +=
      ((double)(end - start)) / CLOCKS_PER_SEC;
  write_erp_interp_data_sub_pel_8(
      "erp_dia_interp_sub_pel_8", test_config, ref_frame, frame_stride,
      frame_width, frame_height, block_x, block_y, block_cnt, &best_sphere_mv,
      result->sad_erp_diamond_interp_interp_sub_pel_8[block_cnt]);
  start = clock();
  result->sad_erp_diamond_interp_sub_pel_8sharp[block_cnt] =
      av1_motion_search_diamond_erp_interp_sub_pel_8sharp(
          block_x, block_y, block_width, block_height, cur_frame, ref_frame,
          frame_stride, frame_width, frame_height, search_range,
          &start_sphere_mv, &best_sphere_mv);
  end = clock();
  result->time_erp_diamond_interp_sub_pel_8sharp +=
      ((double)(end - start)) / CLOCKS_PER_SEC;
  write_erp_interp_data_sub_pel_8sharp(
      "erp_dia_interp_sub_pel_8sharp", test_config, ref_frame, frame_stride,
      frame_width, frame_height, block_x, block_y, block_cnt, &best_sphere_mv,
      result->sad_erp_diamond_interp_sub_pel_8sharp[block_cnt]);
  start = clock();
  result->sad_erp_diamond_interp_sub_pel_8smooth[block_cnt] =
      av1_motion_search_diamond_erp_interp_sub_pel_8smooth(
          block_x, block_y, block_width, block_height, cur_frame, ref_frame,
          frame_stride, frame_width, frame_height, search_range,
          &start_sphere_mv, &best_sphere_mv);
  end = clock();
  result->time_erp_diamond_interp_sub_pel_8smooth +=
      ((double)(end - start)) / CLOCKS_PER_SEC;
  write_erp_interp_data_sub_pel_8smooth(
      "erp_dia_interp_sub_pel_8smooth", test_config, ref_frame, frame_stride,
      frame_width, frame_height, block_x, block_y, block_cnt, &best_sphere_mv,
      result->sad_erp_diamond_interp_sub_pel_8smooth[block_cnt]);
  start = clock();
  result->sad_erp_diamond_interp_filter_search[block_cnt] =
      av1_motion_search_diamond_erp_interp_filter_search(
          block_x, block_y, block_width, block_height, cur_frame, ref_frame,
          frame_stride, frame_width, frame_height, search_range,
          &start_sphere_mv, &best_sphere_mv);
  end = clock();
  result->time_erp_diamond_interp_filter_search +=
      ((double)(end - start)) / CLOCKS_PER_SEC;
  write_erp_interp_data(
      "erp_dia_interp_filter_search", test_config, ref_frame, frame_stride,
      frame_width, frame_height, block_x, block_y, block_cnt, &best_sphere_mv,
      result->sad_erp_diamond_interp_filter_search[block_cnt]);

  start = clock();
  result->sad_erp_diamond_interp_bilinear_scale[block_cnt] =
      av1_motion_search_diamond_erp_interp_bilinear_scale(
          block_x, block_y, block_width, block_height, cur_frame, ref_frame,
          frame_stride, frame_width, frame_height, search_range,
          &start_sphere_mv, &best_sphere_mv);
  end = clock();
  result->time_erp_diamond_interp_bilinear_scale +=
      ((double)(end - start)) / CLOCKS_PER_SEC;
  write_erp_interp_data_bilinear(
      "erp_dia_interp_bilinear_scale", test_config, ref_frame, frame_stride,
      frame_width, frame_height, block_x, block_y, block_cnt, &best_sphere_mv,
      result->sad_erp_diamond_interp_bilinear_scale[block_cnt]);
  start = clock();
  result->sad_erp_diamond_interp_sub_pel_8_scale[block_cnt] =
      av1_motion_search_diamond_erp_interp_sub_pel_8_scale(
          block_x, block_y, block_width, block_height, cur_frame, ref_frame,
          frame_stride, frame_width, frame_height, search_range,
          &start_sphere_mv, &best_sphere_mv);
  end = clock();
  result->time_erp_diamond_interp_sub_pel_8_scale +=
      ((double)(end - start)) / CLOCKS_PER_SEC;
  write_erp_interp_data_sub_pel_8(
      "erp_dia_interp_sub_pel_8_scale", test_config, ref_frame, frame_stride,
      frame_width, frame_height, block_x, block_y, block_cnt, &best_sphere_mv,
      result->sad_erp_diamond_interp_sub_pel_8_scale[block_cnt]);
  start = clock();
  result->sad_erp_diamond_interp_sub_pel_8sharp_scale[block_cnt] =
      av1_motion_search_diamond_erp_interp_sub_pel_8sharp_scale(
          block_x, block_y, block_width, block_height, cur_frame, ref_frame,
          frame_stride, frame_width, frame_height, search_range,
          &start_sphere_mv, &best_sphere_mv);
  end = clock();
  result->time_erp_diamond_interp_sub_pel_8sharp_scale +=
      ((double)(end - start)) / CLOCKS_PER_SEC;
  write_erp_interp_data_sub_pel_8sharp(
      "erp_dia_interp_sub_pel_8sharp_scale", test_config, ref_frame,
      frame_stride, frame_width, frame_height, block_x, block_y, block_cnt,
      &best_sphere_mv,
      result->sad_erp_diamond_interp_sub_pel_8sharp_scale[block_cnt]);

  start = clock();
  result->sad_erp_diamond_interp_sub_pel_8smooth_scale[block_cnt] =
      av1_motion_search_diamond_erp_interp_sub_pel_8smooth_scale(
          block_x, block_y, block_width, block_height, cur_frame, ref_frame,
          frame_stride, frame_width, frame_height, search_range,
          &start_sphere_mv, &best_sphere_mv);
  end = clock();
  result->time_erp_diamond_interp_sub_pel_8smooth_scale +=
      ((double)(end - start)) / CLOCKS_PER_SEC;
  write_erp_interp_data_sub_pel_8smooth(
      "erp_dia_interp_sub_pel_8smooth_scale", test_config, ref_frame,
      frame_stride, frame_width, frame_height, block_x, block_y, block_cnt,
      &best_sphere_mv,
      result->sad_erp_diamond_interp_sub_pel_8smooth_scale[block_cnt]);
  start = clock();
  result->sad_erp_diamond_interp_filter_search_scale[block_cnt] =
      av1_motion_search_diamond_erp_interp_filter_search_scale(
          block_x, block_y, block_width, block_height, cur_frame, ref_frame,
          frame_stride, frame_width, frame_height, search_range,
          &start_sphere_mv, &best_sphere_mv);
  end = clock();
  result->time_erp_diamond_interp_filter_search_scale +=
      ((double)(end - start)) / CLOCKS_PER_SEC;
  write_erp_interp_data(
      "erp_dia_interp_filter_search_scale", test_config, ref_frame,
      frame_stride, frame_width, frame_height, block_x, block_y, block_cnt,
      &best_sphere_mv,
      result->sad_erp_diamond_interp_filter_search_scale[block_cnt]);
}

static void whole_frame_final_analyze(const TestConfig *test_config) {
  y4m_input_t *handle = NULL;
  config_t config;
  ASSERT_EQ(0, open_file_y4m(filepaths[test_config->filename],
                             (handle_t *)(&handle), &config));

  picture_t cur_pic;
  picture_t ref_pic;
  uint8_t temp;
  cur_pic.img.plane[0] =
      (uint8_t *)malloc(handle->width * handle->height * sizeof(temp));
  cur_pic.img.plane[1] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));
  cur_pic.img.plane[2] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));
  ref_pic.img.plane[0] =
      (uint8_t *)malloc(handle->width * handle->height * sizeof(temp));
  ref_pic.img.plane[1] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));
  ref_pic.img.plane[2] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));

  read_frame_y4m((handle_t)handle, &cur_pic, test_config->cur_frame_no);
  read_frame_y4m((handle_t)handle, &ref_pic, test_config->ref_frame_no);

  uint8_t *cur_frame = cur_pic.img.plane[0];
  uint8_t *ref_frame = ref_pic.img.plane[0];

  int block_x;
  int block_y;
  int row_cnt = 10;
  int col_cnt = 10;
  int block_step_x = handle->width / col_cnt;
  int block_step_y = handle->height / row_cnt;

  // AnalyzeData data;
  // init_analyze_data(&data);

  FinalSearchResult search_result;
  init_final_search_result(&search_result, row_cnt, col_cnt);

  int block_count = 0;

  mkdir(output_filepaths[test_config->filename], S_IRWXU | S_IRWXG | S_IROTH);

  // clock_t start, end;
  // double cpu_time_used;

  printf("File path: %s\n", filepaths[test_config->filename]);
  printf("Current frame: %d, reference frame: %d\n", test_config->cur_frame_no,
         test_config->ref_frame_no);
  printf("Block size: %dx%d, search range: %d\n", test_config->block_width,
         test_config->block_height, test_config->search_range);

  for (block_y = 0; block_y + test_config->block_height < handle->height;
       block_y += block_step_y) {
    for (block_x = 0; block_x + test_config->block_width < handle->width;
         block_x += block_step_x) {
      do_final_motion_searches(
          block_x, block_y, test_config->block_width, test_config->block_height,
          cur_frame, ref_frame, handle->width, handle->width, handle->height,
          test_config->search_range, block_count, test_config, &search_result);
      block_count++;
      printf("%d\n", block_count);
    }
  }  // for idx_y

  cal_avg_sad(&search_result);
  write_final_search_result(test_config, &search_result);

  free_frames(handle, &cur_pic, &ref_pic);
  free_final_search_result(&search_result);
}

TEST(SphericalMappingTest, FinalTest) {
  TestConfig configs[] = {
    { ARIALCITY960, 60, 65, 1637, 665, 32, 32, 20 },
    { BALBOA960, 180, 185, 514, 187, 32, 32, 20 },
    { BRANCASTLE960, 60, 65, 914, 398, 32, 32, 20 },
    { BROADWAY960, 80, 85, 1518, 570, 32, 32, 20 },
    { GASLAMP960, 210, 215, 1150, 500, 32, 32, 20 },
    { HARBOR960, 1, 5, 50, 490, 32, 32, 20 },
    { KITEFLITE960, 15, 20, 520, 425, 32, 32, 20 },
    { LANDING960, 90, 95, 828, 212, 32, 32, 20 },
    { POLEVAULT960, 30, 35, 328, 800, 32, 32, 20 },
    { TROLLEY960, 90, 95, 1649, 387, 32, 32, 20 },
  };

  // whole_frame_final_analyze(&configs[4]);
  // for (int i = 0; i < 10; i++) {
  //   whole_frame_final_analyze(&configs[i]);
  // }

  y4m_input_t *handle = NULL;
  config_t config;
  ASSERT_EQ(0, open_file_y4m(filepaths[configs[4].filename],
                             (handle_t *)(&handle), &config));

  picture_t cur_pic;
  picture_t ref_pic;
  uint8_t temp;
  cur_pic.img.plane[0] =
      (uint8_t *)malloc(handle->width * handle->height * sizeof(temp));
  cur_pic.img.plane[1] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));
  cur_pic.img.plane[2] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));
  ref_pic.img.plane[0] =
      (uint8_t *)malloc(handle->width * handle->height * sizeof(temp));
  ref_pic.img.plane[1] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));
  ref_pic.img.plane[2] =
      (uint8_t *)malloc(handle->width * handle->height / 4 * sizeof(temp));

  read_frame_y4m((handle_t)handle, &cur_pic, configs[4].cur_frame_no);
  read_frame_y4m((handle_t)handle, &ref_pic, configs[4].ref_frame_no);

  uint8_t *cur_frame = cur_pic.img.plane[0];
  uint8_t *ref_frame = ref_pic.img.plane[0];

  int pred_block[128 * 128];
  int cur_block[128 * 128];
  int block_stride = 128;
  int block_x = 1344;
  int block_y = 576;
  int block_width = configs[4].block_width;
  int block_height = configs[4].block_height;
  int frame_stride = handle->width;
  int frame_width = handle->width;
  int frame_height = handle->height;
  int search_range = configs[4].search_range;

  SphereMV start_sphere_mv;
  start_sphere_mv.phi = 0;
  start_sphere_mv.theta = 0;
  SphereMV best_sphere_mv;
  PlaneMV best_plane_mv;
  PlaneMV start_plane_mv;

  double block_phi;
  double block_theta;
  double plane_phi;
  double plane_theta;

  av1_motion_search_brute_force_plane(
      block_x, block_y, block_width, block_height, cur_frame, ref_frame,
      frame_stride, frame_width, frame_height, search_range, &best_plane_mv);

  start_plane_mv.x = best_plane_mv.x;
  start_plane_mv.y = best_plane_mv.y;
  av1_plane_to_sphere_erp(block_x + best_plane_mv.x, block_y + best_plane_mv.y,
                          frame_width, frame_height, &plane_phi, &plane_theta);
  av1_plane_to_sphere_erp(block_x, block_y, frame_width, frame_height,
                          &block_phi, &block_theta);
  start_sphere_mv.phi = plane_phi - block_phi;
  start_sphere_mv.theta = plane_theta - block_theta;

  av1_motion_search_diamond_erp(block_x, block_y, block_width, block_height,
                                cur_frame, ref_frame, frame_stride, frame_width,
                                frame_height, search_range, &start_sphere_mv,
                                &best_sphere_mv);

  get_pred_block_idx(block_x, block_y, block_width, block_height,
                     best_sphere_mv.phi, best_sphere_mv.theta, frame_stride,
                     frame_width, frame_height, block_stride, cur_block,
                     pred_block);

  write_bmp("car_cur.bmp", frame_width, frame_height, cur_pic.img.plane[0],
            cur_block, block_width, block_height, block_stride);
  write_bmp("car_ref_dia_p2p.bmp", frame_width, frame_height,
            ref_pic.img.plane[0], pred_block, block_width, block_height,
            block_stride);

  av1_motion_search_diamond_erp_interp_sub_pel_8(
      block_x, block_y, block_width, block_height, cur_frame, ref_frame,
      frame_stride, frame_width, frame_height, search_range, &start_sphere_mv,
      &best_sphere_mv);

  get_pred_block_idx(block_x, block_y, block_width, block_height,
                     best_sphere_mv.phi, best_sphere_mv.theta, frame_stride,
                     frame_width, frame_height, block_stride, cur_block,
                     pred_block);

  write_bmp("car_ref_dia_subpl.bmp", frame_width, frame_height,
            ref_pic.img.plane[0], pred_block, block_width, block_height,
            block_stride);

  av1_motion_search_diamond_erp_interp_sub_pel_8_scale(
      block_x, block_y, block_width, block_height, cur_frame, ref_frame,
      frame_stride, frame_width, frame_height, search_range, &start_sphere_mv,
      &best_sphere_mv);

  get_pred_block_idx(block_x, block_y, block_width, block_height,
                     best_sphere_mv.phi, best_sphere_mv.theta, frame_stride,
                     frame_width, frame_height, block_stride, cur_block,
                     pred_block);
  write_bmp("car_ref_dia_subpl_scale.bmp", frame_width, frame_height,
            ref_pic.img.plane[0], pred_block, block_width, block_height,
            block_stride);

  free_frames(handle, &cur_pic, &ref_pic);
}

}  // namespace
