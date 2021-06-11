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

#include <assert.h>
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
  assert(y >= 0 && y < height);

  double x_mod = fmod(x, width);
  x_mod = x_mod >= 0 ? x_mod : x_mod + width;

  // Since x_mod is in [0, width), theta is in [-PI, PI)
  *theta = (x_mod - width * 0.5) / width * 2 * PI;
  *phi = (y - height * 0.5) / height * PI;
}
