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

#include "av1/common/common.h"
#include "av1/common/spherical_pred.h"
#include "aom_dsp/aom_dsp_common.h"

void av1_sphere_to_plane_erp(double phi, double theta, int width, int height,
                             double *x, double *y) {
  double phi_mod = fmod(phi, 2 * PI);
  int is_on_other_hemisphere = ((phi_mod > 0.5 * PI && phi_mod < 1.5 * PI) ||
                                (phi_mod < -0.5 * PI && phi_mod > -1.5 * PI));

  // Flip phi to the destination quadrant
  phi_mod = is_on_other_hemisphere ? PI - phi_mod : phi_mod;
  // Regualte phi back to [-pi/2, pi/2)
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
  // This should actually be a range related to 1/cos(phi) since x is distorted
  // TODO(yaoyaogoogle): Adjust the width of x according to the interpolation
  // mode
  *x = theta_mod / PI * width * 0.5 + width * 0.5;

  // No minus sign for y since we only use an imaginary upside-down globe
  *y = phi_mod / (PI * 0.5) * height * 0.5 + height * 0.5;
  *y = AOMMIN(*y, height - 0.00001);
}

void av1_plane_to_sphere_erp(double x, double y, int width, int height,
                             double *phi, double *theta) {
  y = AOMMAX(y, 0);
  y = AOMMIN(y, height - 1);

  double x_mod = fmod(x, width);
  x_mod = x_mod >= 0 ? x_mod : x_mod + width;

  // Since x_mod is in [0, width), theta is in [-PI, PI)
  *theta = (x_mod - width * 0.5) / width * 2 * PI;
  *phi = (y - height * 0.5) / height * PI;
}

void av1_get_pred_erp(int block_x, int block_y, int block_width,
                      int block_height, double delta_phi, double delta_theta,
                      uint8_t *ref_frame, int ref_frame_stride, int frame_width,
                      int frame_height, int pred_block_stride,
                      uint8_t *pred_block) {
  assert(ref_frame != NULL && frame_width > 0 && frame_height > 0 &&
         ref_frame_stride >= frame_width);
  assert(pred_block != NULL && block_width > 0 && block_height > 0 &&
         pred_block_stride >= block_width);

  int pos_pred;      // Offset in pred_block buffer
  int pos_ref;       // Offset in ref_frame buffer
  double phi;        // Latitude on the globe
  double theta;      // Longitude on the globe
  double x_current;  // X coordiante in current frame
  double y_current;  // Y coordiante in currrent frame
  double x_ref;      // X coordinate in reference frame
  double y_ref;      // Y coordiante in reference frame

  for (int idx_y = 0; idx_y < block_height; idx_y++) {
    for (int idx_x = 0; idx_x < block_width; idx_x++) {
      x_current = idx_x + block_x;
      y_current = idx_y + block_y;

      av1_plane_to_sphere_erp(x_current, y_current, frame_width, frame_height,
                              &phi, &theta);
      av1_sphere_to_plane_erp(phi + delta_phi, theta + delta_theta, frame_width,
                              frame_height, &x_ref, &y_ref);

      pos_pred = idx_x + idx_y * pred_block_stride;
      x_ref = round(x_ref);
      x_ref = x_ref == frame_width ? 0 : x_ref;
      y_ref = round(y_ref);
      y_ref = y_ref == frame_height ? frame_height - 1 : y_ref;
      pos_ref = (int)x_ref + (int)y_ref * ref_frame_stride;
      pred_block[pos_pred] = ref_frame[pos_ref];
    }
  }
}
