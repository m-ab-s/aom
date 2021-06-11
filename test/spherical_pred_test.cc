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

#include "av1/common/spherical_pred.h"
#include "third_party/googletest/src/googletest/include/gtest/gtest.h"

namespace {

#define DIFF_THRESHOLD 0.00001

TEST(SphericalMappingTest, EquiPlaneToGlobeReverseTest) {
  // Check if the mapping from plane to globe is reverseable
  int width = 400;
  int height = 300;

  double x;
  double y;
  double phi;
  double theta;
  double x_from_globe;
  double y_from_globe;

  for (x = 0; x < width; x += 10) {
    for (y = 0; y < height; y += 10) {
      av1_plane_to_sphere_erp(x, y, width, height, &phi, &theta);
      av1_sphere_to_plane_erp(phi, theta, width, height, &x_from_globe,
                              &y_from_globe);
      EXPECT_NEAR(x, x_from_globe, DIFF_THRESHOLD);
      EXPECT_NEAR(y, y_from_globe, DIFF_THRESHOLD);
    }
  }
}

TEST(SphericalMappingTest, EquiGlobeToPlaneReverseTest) {
  // Check if the mapping from plane to globe is reverseable
  int width = 400;
  int height = 300;

  double x;
  double y;
  double phi;
  double theta;
  double phi_from_plane;
  double theta_from_plane;

  const double pi = 3.141592653589793238462643383279502884;

  for (phi = -0.5 * pi; phi <= 0.5 * pi; phi += 0.1) {
    for (theta = -pi; theta <= pi; theta += 0.1) {
      av1_sphere_to_plane_erp(phi, theta, width, height, &x, &y);
      av1_plane_to_sphere_erp(x, y, width, height, &phi_from_plane,
                              &theta_from_plane);
      EXPECT_NEAR(phi, phi_from_plane, DIFF_THRESHOLD);
      EXPECT_NEAR(theta, theta_from_plane, DIFF_THRESHOLD);
    }
  }
}

TEST(SphericalMappingTest, EquiPlaneToGlobeRangeTest) {
  // Check if the mapping from plane to globe is out of range
  int width = 400;
  int height = 300;

  double x;
  double y;
  double phi;
  double theta;

  const double pi = 3.141592653589793238462643383279502884;

  for (x = -2 * width; x <= 2 * width; x += 1) {
    for (y = 0; y < height; y += 1) {
      av1_plane_to_sphere_erp(x, y, width, height, &phi, &theta);
      EXPECT_GE(phi, -0.5 * pi);
      EXPECT_LT(phi, 0.5 * pi);
      EXPECT_GE(theta, -pi);
      EXPECT_LT(theta, pi);
    }
  }
}

TEST(SphericalMappingTest, EquiGlobeToPlaneRangeTest) {
  // Check if the mapping from plane to globe is out of range
  int width = 400;
  int height = 300;

  double x;
  double y;
  double phi;
  double theta;

  const double pi = 3.141592653589793238462643383279502884;

  for (phi = -0.5 * pi * 2; phi <= 0.5 * pi * 2; phi += 0.1) {
    for (theta = -pi * 2; theta <= pi * 2; theta += 0.1) {
      av1_sphere_to_plane_erp(phi, theta, width, height, &x, &y);
      EXPECT_GE(x, 0);
      EXPECT_LT(x, width);
      EXPECT_GE(y, 0);
      EXPECT_LT(y, height);
    }
  }
}

}  // namespace
