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

#if CONFIG_SPHERICAL_PRED
#include "av1/common/spherical_pred.h"
#endif

#include "third_party/googletest/src/googletest/include/gtest/gtest.h"

namespace {

#if CONFIG_SPHERICAL_PRED

#define DIFF_THRESHOLD 0.00001
constexpr double pi = 3.141592653589793238462643383279502884;

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

  for (phi = -0.5 * pi; phi < 0.5 * pi; phi += 0.1) {
    for (theta = -pi; theta < pi; theta += 0.1) {
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

  for (phi = -0.5 * pi * 4; phi <= 0.5 * pi * 4; phi += 0.1) {
    if (fmod(phi, pi) == 0.5 * pi) {
      continue;
    }
    for (theta = -pi * 4; theta <= pi * 4; theta += 0.1) {
      av1_sphere_to_plane_erp(phi, theta, width, height, &x, &y);
      EXPECT_GE(x, 0);
      EXPECT_LT(x, width);
      EXPECT_GE(y, 0);
      EXPECT_LT(y, height);
    }
  }
}

TEST(SphericalMappingTest, EquiPlaneToGlobeAnchorTest) {
  // Check the correctness of mapping for some special anchor points
  int width = 400;
  int height = 300;
  double phi;
  double theta;

  const int case_cnt = 9;

  const double test_x[case_cnt] = { width * 0.5,
                                    0,
                                    width - DIFF_THRESHOLD,
                                    0,
                                    width - DIFF_THRESHOLD,
                                    (width - DIFF_THRESHOLD) * 0.25,
                                    (width - DIFF_THRESHOLD) * 0.75,
                                    (width - DIFF_THRESHOLD) * 0.75,
                                    (width - DIFF_THRESHOLD) * 0.25 };
  const double test_y[case_cnt] = { height * 0.5,
                                    0,
                                    0,
                                    1.0 * height - DIFF_THRESHOLD,
                                    1.0 * height - DIFF_THRESHOLD,
                                    height * 0.25,
                                    height * 0.25,
                                    height * 0.75,
                                    height * 0.75 };
  const double expect_phi[case_cnt] = { 0,          -0.5 * pi, -0.5 * pi,
                                        0.5 * pi,   0.5 * pi,  -0.25 * pi,
                                        -0.25 * pi, 0.25 * pi, 0.25 * pi };
  const double expect_theta[case_cnt] = { 0,        -pi,      pi,
                                          -pi,      pi,       -0.5 * pi,
                                          0.5 * pi, 0.5 * pi, -0.5 * pi };

  for (int i = 0; i < case_cnt; i++) {
    av1_plane_to_sphere_erp(test_x[i], test_y[i], width, height, &phi, &theta);
    EXPECT_NEAR(phi, expect_phi[i], DIFF_THRESHOLD);
    EXPECT_NEAR(theta, expect_theta[i], DIFF_THRESHOLD);
  }
}

TEST(SphericalMappingTest, EquiGlobeToPlaneAnchorTest) {
  // Check the correctness of mapping for some special anchor points
  int width = 400;
  int height = 300;
  double x;
  double y;

  const int case_cnt = 19;

  const double test_phi[case_cnt] = {
    0,          -0.5 * pi,  -0.5 * pi,  0.5 * pi,   0.5 * pi,
    -0.25 * pi, -0.25 * pi, 0.25 * pi,  0.25 * pi,  -0.75 * pi,
    -0.75 * pi, -pi,        -1.25 * pi, -1.75 * pi, 0.75 * pi,
    0.75 * pi,  pi,         1.25 * pi,  1.75 * pi
  };
  const double test_theta[case_cnt] = { 0,
                                        -pi,
                                        pi - 0.0000001,
                                        -pi,
                                        pi - 0.0000001,
                                        -0.5 * pi,
                                        (pi - 0.0000001) * 0.5,
                                        (pi - 0.0000001) * 0.5,
                                        -0.5 * pi,
                                        0,
                                        0.5 * pi,
                                        -0.5 * pi,
                                        0.5 * pi,
                                        -0.5 * pi,
                                        0,
                                        0.5 * pi,
                                        -0.5 * pi,
                                        0.5 * pi,
                                        -0.5 * pi };
  const double expect_x[case_cnt] = { width * 0.5,
                                      0,
                                      1.0 * width,
                                      0,
                                      1.0 * width,
                                      width * 0.25,
                                      width * 0.75,
                                      width * 0.75,
                                      width * 0.25,
                                      0,
                                      width * 0.25,
                                      width * 0.75,
                                      width * 0.25,
                                      width * 0.25,
                                      0,
                                      width * 0.25,
                                      width * 0.75,
                                      width * 0.25,
                                      width * 0.25 };
  const double expect_y[case_cnt] = { height * 0.5,
                                      0,
                                      0,
                                      1.0 * height - DIFF_THRESHOLD,
                                      1.0 * height - DIFF_THRESHOLD,
                                      height * 0.25,
                                      height * 0.25,
                                      height * 0.75,
                                      height * 0.75,
                                      height * 0.25,
                                      height * 0.25,
                                      height * 0.5,
                                      height * 0.75,
                                      height * 0.75,
                                      height * 0.75,
                                      height * 0.75,
                                      height * 0.5,
                                      height * 0.25,
                                      height * 0.25 };

  for (int i = 0; i < case_cnt; i++) {
    av1_sphere_to_plane_erp(test_phi[i], test_theta[i], width, height, &x, &y);
    EXPECT_NEAR(x, expect_x[i], DIFF_THRESHOLD);
    EXPECT_NEAR(y, expect_y[i], DIFF_THRESHOLD);
  }
}

TEST(SphericalMappingTest, EquiRodriguesReverseTest) {
  // Check if the mapping from plane to globe is reverseable through Rodrigues
  // rotation
  int frame_width = 400;
  int frame_height = 300;

  double x;
  double y;
  double x_from_globe;
  double y_from_globe;

  PolarVector polar;
  PolarVector polar_prime;
  CartesianVector k;
  CartesianVector v;
  CartesianVector v_prime;
  double k_dot_v;
  polar.r = 1.0;
  double product;
  double alpha;

  for (x = 0; x < frame_width; x += 10) {
    for (y = 0; y < frame_height; y += 10) {
      av1_plane_to_sphere_erp(x, y, frame_width, frame_height, &polar.phi,
                              &polar.theta);
      polar.phi += 0.5 * pi;
      av1_sphere_polar_to_carte(&polar, &v);
      av1_sphere_polar_to_carte(&polar, &v_prime);
      av1_carte_vectors_cross_product(&v, &v_prime, &k);
      av1_normalize_carte_vector(&k);

      product = av1_carte_vectors_dot_product(&v, &v_prime);
      // Avoid floating point precision issues
      product = AOMMAX(product, -1.0);
      product = AOMMIN(product, 1.0);
      alpha = acos(product);
      k_dot_v = av1_carte_vectors_dot_product(&k, &v);

      v_prime.x = k.x * k_dot_v * (1 - cos(alpha)) + v.x * cos(alpha) +
                  (k.y * v.z - k.z * v.y) * sin(alpha);
      v_prime.y = k.y * k_dot_v * (1 - cos(alpha)) + v.y * cos(alpha) +
                  (k.z * v.x - k.x * v.z) * sin(alpha);
      v_prime.z = k.z * k_dot_v * (1 - cos(alpha)) + v.z * cos(alpha) +
                  (k.x * v.y - k.y * v.x) * sin(alpha);

      av1_sphere_carte_to_polar(&v_prime, &polar_prime);
      polar_prime.phi -= 0.5 * pi;
      av1_sphere_to_plane_erp(polar_prime.phi, polar_prime.theta, frame_width,
                              frame_height, &x_from_globe, &y_from_globe);

      EXPECT_NEAR(x, x_from_globe, DIFF_THRESHOLD);
      EXPECT_NEAR(y, y_from_globe, DIFF_THRESHOLD);
    }
  }  // for x
}

#endif

}  // namespace
