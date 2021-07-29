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

#include <stdlib.h>
#include <math.h>

#include "av1/common/common.h"
#include "av1/common/filter.h"
#include "av1/common/spherical_pred.h"
#include "aom_dsp/aom_dsp_common.h"

#if CONFIG_SPHERICAL_PRED

void av1_sphere_to_plane_erp(double phi, double theta, int width, int height,
                             double *x, double *y) {
  double phi_mod = fmod(phi, 2 * PI);
  int is_on_other_hemisphere = ((phi_mod > 0.5 * PI && phi_mod < 1.5 * PI) ||
                                (phi_mod < -0.5 * PI && phi_mod > -1.5 * PI));

  // Flip phi to the destination quadrant
  phi_mod = is_on_other_hemisphere ? PI - phi_mod : phi_mod;
  // Regulate phi back to [-pi/2, pi/2)
  if (phi_mod > 1.5 * PI) {
    phi_mod = phi_mod - 2 * PI;
  } else if (phi_mod < -1.5 * PI) {
    phi_mod = phi_mod + 2 * PI;
  }

  double theta_mod = fmod(is_on_other_hemisphere ? theta + PI : theta, 2 * PI);
  if (theta_mod < -PI) {
    theta_mod = theta_mod + 2 * PI;
  } else if (theta_mod >= PI) {
    theta_mod = theta_mod - 2 * PI;
  }

  *x = theta_mod / PI * width * 0.5 + width * 0.5;

  // No minus sign for y since we only use an imaginary upside-down globe
  *y = phi_mod / (PI * 0.5) * height * 0.5 + height * 0.5;
  *y = AOMMIN(*y, height - 0.00001);
}

void av1_plane_to_sphere_erp(double x, double y, int width, int height,
                             double *phi, double *theta) {
  // Without the 0.0001, the X information of the top row will be discarded. Any
  // points on the top row of the frame will be mapped to the north pole, hence
  // will be mapped back to the center of the top row in the frame.
  y = AOMMAX(y, 0.00001);
  y = AOMMIN(y, height - 1);

  double x_mod = fmod(x, width);
  x_mod = x_mod >= 0 ? x_mod : x_mod + width;

  // Since x_mod is in [0, width), theta is in [-PI, PI)
  *theta = (x_mod - width * 0.5) / width * 2 * PI;
  *phi = (y - height * 0.5) / height * PI;
}

void av1_normalize_carte_vector(CartesianVector *carte) {
  double temp =
      sqrt(carte->x * carte->x + carte->y * carte->y + carte->z * carte->z);

  // k would be (0, 0, 0) if v and v_prime are the same
  if (temp > 0.0001) {
    carte->x = carte->x / temp;
    carte->y = carte->y / temp;
    carte->z = carte->z / temp;
  }
}

void av1_sphere_polar_to_carte(const PolarVector *polar,
                               CartesianVector *carte) {
  assert(polar != NULL && carte != NULL);

  carte->x = polar->r * cos(polar->theta) * sin(polar->phi);
  carte->y = polar->r * sin(polar->theta) * sin(polar->phi);
  carte->z = polar->r * cos(polar->phi);
}

void av1_sphere_carte_to_polar(const CartesianVector *carte,
                               PolarVector *polar) {
  assert(polar != NULL && carte != NULL);

  polar->r =
      sqrt(carte->x * carte->x + carte->y * carte->y + carte->z * carte->z);
  polar->phi = acos(carte->z / polar->r);

  if (carte->x == 0) {
    if (carte->y > 0) {
      polar->theta = 0.5 * PI;
    } else if (carte->y < 0) {
      polar->theta = 1.5 * PI;
    } else {
      polar->theta = 0;
    }
  } else {
    polar->theta = atan(carte->y / carte->x);
    // arctan only gives (-PI/2, PI/2), so we need to adjust the result
    if (carte->x < 0) {
      polar->theta += PI;
    }
  }
}

double av1_carte_vectors_dot_product(const CartesianVector *left,
                                     const CartesianVector *right) {
  assert(left != NULL && right != NULL);

  return left->x * right->x + left->y * right->y + left->z * right->z;
}

void av1_carte_vectors_cross_product(const CartesianVector *left,
                                     const CartesianVector *right,
                                     CartesianVector *result) {
  assert(left != NULL && right != NULL && result != NULL);

  result->x = left->y * right->z - left->z * right->y;
  result->y = left->z * right->x - left->x * right->z;
  result->z = left->x * right->y - left->y * right->x;
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
static int cal_kernel_idx(double subpel_pos, int *pos_ref) {
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
static void get_interp_x_arr(int x, int y, const uint8_t *ref_frame,
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
      cur_x = cur_x > 0 ? cur_x : frame_width + cur_x;

      sum += ref_frame[cur_x + cur_y * ref_frame_stride] * coeff;
    }

    x_sum_arr[i] = sum;
  }  // for i
}

static uint8_t interp_pixel_erp(const uint8_t *ref_frame, int ref_frame_stride,
                                int frame_width, int frame_height,
                                double subpel_x, double subpel_y,
                                const InterpKernel *kernel, int filter_tap) {
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

void av1_get_pred_erp(int block_x, int block_y, int block_width,
                      int block_height, double delta_phi, double delta_theta,
                      const uint8_t *ref_frame, int ref_frame_stride,
                      int frame_width, int frame_height, int pred_block_stride,
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

int av1_motion_search_brute_force_erp(
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

  av1_get_pred_erp(block_x, block_y, block_width, block_height, 0, 0, ref_frame,
                   frame_stride, frame_width, frame_height, pred_block_stride,
                   pred_block);
  best_sad = get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               frame_stride, pred_block_stride);
  best_mv->phi = 0;
  best_mv->theta = 0;

  for (int i = -search_range; i <= search_range; i++) {
    delta_phi = i * search_step_phi;

    for (int j = -search_range; j <= search_range; j++) {
      delta_theta = j * search_step_theta;

      av1_get_pred_erp(block_x, block_y, block_width, block_height, delta_phi,
                       delta_theta, ref_frame, frame_stride, frame_width,
                       frame_height, pred_block_stride, pred_block);

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

int av1_motion_search_diamond_erp(int block_x, int block_y, int block_width,
                                  int block_height, const uint8_t *cur_frame,
                                  const uint8_t *ref_frame, int frame_stride,
                                  int frame_width, int frame_height,
                                  int search_range, const SphereMV *start_mv,
                                  SphereMV *best_mv) {
  assert(cur_frame != NULL && ref_frame != NULL && best_mv != NULL);
  assert(block_width > 0 && block_height > 0 && block_width <= 128 &&
         block_height <= 128 && block_x >= 0 && block_y >= 0 &&
         frame_width > 0 && frame_height > 0);
  assert(search_range > 0);

  // Large Diamond Search Pattern on shpere
  SphereMV ldsp_mv[9];
  // Small Diamond Search Pattern on shpere
  SphereMV sdsp_mv[5];

  double search_step_phi = 0.5 * block_height * PI / frame_height;
  double search_step_theta = 0.5 * block_width * 2 * PI / frame_width;

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
  av1_get_pred_erp(block_x, block_y, block_width, block_height, ldsp_mv[0].phi,
                   ldsp_mv[0].theta, ref_frame, frame_stride, frame_width,
                   frame_height, pred_block_stride, pred_block);
  best_sad = get_sad_of_blocks(cur_block, pred_block, block_width, block_height,
                               frame_stride, pred_block_stride);

  do {
    update_sphere_mv_ldsp(ldsp_mv, search_step_phi, search_step_theta);

    for (int i = 0; i < 9; i++) {
      av1_get_pred_erp(block_x, block_y, block_width, block_height,
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

#endif
