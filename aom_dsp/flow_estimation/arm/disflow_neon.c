/*
 * Copyright (c) 2023, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include "aom_dsp/flow_estimation/disflow.h"

#include <arm_neon.h>
#include <math.h>

#include "aom_dsp/arm/mem_neon.h"
#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"

static INLINE void get_cubic_kernel_dbl(double x, double *kernel) {
  assert(0 <= x && x < 1);
  double x2 = x * x;
  double x3 = x2 * x;
  kernel[0] = -0.5 * x + x2 - 0.5 * x3;
  kernel[1] = 1.0 - 2.5 * x2 + 1.5 * x3;
  kernel[2] = 0.5 * x + 2.0 * x2 - 1.5 * x3;
  kernel[3] = -0.5 * x2 + 0.5 * x3;
}

static INLINE void get_cubic_kernel_int(double x, int *kernel) {
  double kernel_dbl[4];
  get_cubic_kernel_dbl(x, kernel_dbl);

  kernel[0] = (int)rint(kernel_dbl[0] * (1 << DISFLOW_INTERP_BITS));
  kernel[1] = (int)rint(kernel_dbl[1] * (1 << DISFLOW_INTERP_BITS));
  kernel[2] = (int)rint(kernel_dbl[2] * (1 << DISFLOW_INTERP_BITS));
  kernel[3] = (int)rint(kernel_dbl[3] * (1 << DISFLOW_INTERP_BITS));
}

static INLINE int get_cubic_value_int(const int *p, const int *kernel) {
  return kernel[0] * p[0] + kernel[1] * p[1] + kernel[2] * p[2] +
         kernel[3] * p[3];
}

// Compare two regions of width x height pixels, one rooted at position
// (x, y) in src and the other at (x + u, y + v) in ref.
// This function returns the sum of squared pixel differences between
// the two regions.
static INLINE void compute_flow_error(const uint8_t *src, const uint8_t *ref,
                                      int width, int height, int stride, int x,
                                      int y, double u, double v, int16_t *dt) {
  // Split offset into integer and fractional parts, and compute cubic
  // interpolation kernels
  const int u_int = (int)floor(u);
  const int v_int = (int)floor(v);
  const double u_frac = u - floor(u);
  const double v_frac = v - floor(v);

  int h_kernel[4];
  int v_kernel[4];
  get_cubic_kernel_int(u_frac, h_kernel);
  get_cubic_kernel_int(v_frac, v_kernel);

  // Storage for intermediate values between the two convolution directions
  int tmp_[DISFLOW_PATCH_SIZE * (DISFLOW_PATCH_SIZE + 3)];
  int *tmp = tmp_ + DISFLOW_PATCH_SIZE;  // Offset by one row

  // Clamp coordinates so that all pixels we fetch will remain within the
  // allocated border region, but allow them to go far enough out that
  // the border pixels' values do not change.
  // Since we are calculating an 8x8 block, the bottom-right pixel
  // in the block has coordinates (x0 + 7, y0 + 7). Then, the cubic
  // interpolation has 4 taps, meaning that the output of pixel
  // (x_w, y_w) depends on the pixels in the range
  // ([x_w - 1, x_w + 2], [y_w - 1, y_w + 2]).
  //
  // Thus the most extreme coordinates which will be fetched are
  // (x0 - 1, y0 - 1) and (x0 + 9, y0 + 9).
  const int x0 = clamp(x + u_int, -9, width);
  const int y0 = clamp(y + v_int, -9, height);

  // Horizontal convolution
  for (int i = -1; i < DISFLOW_PATCH_SIZE + 2; ++i) {
    const int y_w = y0 + i;
    for (int j = 0; j < DISFLOW_PATCH_SIZE; ++j) {
      const int x_w = x0 + j;
      int arr[4];

      arr[0] = (int)ref[y_w * stride + (x_w - 1)];
      arr[1] = (int)ref[y_w * stride + (x_w + 0)];
      arr[2] = (int)ref[y_w * stride + (x_w + 1)];
      arr[3] = (int)ref[y_w * stride + (x_w + 2)];

      // Apply kernel and round, keeping 6 extra bits of precision.
      //
      // 6 is the maximum allowable number of extra bits which will avoid
      // the intermediate values overflowing an int16_t. The most extreme
      // intermediate value occurs when:
      // * The input pixels are [0, 255, 255, 0]
      // * u_frac = 0.5
      // In this case, the un-scaled output is 255 * 1.125 = 286.875.
      // As an integer with 6 fractional bits, that is 18360, which fits
      // in an int16_t. But with 7 fractional bits it would be 36720,
      // which is too large.
      tmp[i * DISFLOW_PATCH_SIZE + j] = ROUND_POWER_OF_TWO(
          get_cubic_value_int(arr, h_kernel), DISFLOW_INTERP_BITS - 6);
    }
  }

  // Vertical convolution
  for (int i = 0; i < DISFLOW_PATCH_SIZE; ++i) {
    for (int j = 0; j < DISFLOW_PATCH_SIZE; ++j) {
      const int *p = &tmp[i * DISFLOW_PATCH_SIZE + j];
      const int arr[4] = { p[-DISFLOW_PATCH_SIZE], p[0], p[DISFLOW_PATCH_SIZE],
                           p[2 * DISFLOW_PATCH_SIZE] };
      const int result = get_cubic_value_int(arr, v_kernel);

      // Apply kernel and round.
      // This time, we have to round off the 6 extra bits which were kept
      // earlier, but we also want to keep DISFLOW_DERIV_SCALE_LOG2 extra bits
      // of precision to match the scale of the dx and dy arrays.
      const int round_bits = DISFLOW_INTERP_BITS + 6 - DISFLOW_DERIV_SCALE_LOG2;
      const int warped = ROUND_POWER_OF_TWO(result, round_bits);
      const int src_px = src[(x + j) + (y + i) * stride] << 3;
      const int err = warped - src_px;
      dt[i * DISFLOW_PATCH_SIZE + j] = err;
    }
  }
}

static INLINE void sobel_filter_x(const uint8_t *src, int src_stride,
                                  int16_t *dst, int dst_stride) {
  int16_t tmp[DISFLOW_PATCH_SIZE * (DISFLOW_PATCH_SIZE + 2)];

  // Horizontal filter, using kernel {1, 0, -1}.
  const uint8_t *src_start = src - 1 * src_stride - 1;

  for (int i = 0; i < DISFLOW_PATCH_SIZE + 2; i++) {
    uint8x16_t s = vld1q_u8(src_start + i * src_stride);
    uint8x8_t s0 = vget_low_u8(s);
    uint8x8_t s2 = vget_low_u8(vextq_u8(s, s, 2));

    // Given that the kernel is {1, 0, -1} the convolution is a simple
    // subtraction.
    int16x8_t diff = vreinterpretq_s16_u16(vsubl_u8(s0, s2));

    vst1q_s16(tmp + i * DISFLOW_PATCH_SIZE, diff);
  }

  // Vertical filter, using kernel {1, 2, 1}.
  // This kernel can be split into two 2-taps kernels of value {1, 1}.
  // That way we need only 3 add operations to perform the convolution, one of
  // which can be reused for the next line.
  int16x8_t s0 = vld1q_s16(tmp);
  int16x8_t s1 = vld1q_s16(tmp + DISFLOW_PATCH_SIZE);
  int16x8_t sum01 = vaddq_s16(s0, s1);
  for (int i = 0; i < DISFLOW_PATCH_SIZE; i++) {
    int16x8_t s2 = vld1q_s16(tmp + (i + 2) * DISFLOW_PATCH_SIZE);

    int16x8_t sum12 = vaddq_s16(s1, s2);
    int16x8_t sum = vaddq_s16(sum01, sum12);

    vst1q_s16(dst + i * dst_stride, sum);

    sum01 = sum12;
    s1 = s2;
  }
}

static INLINE void sobel_filter_y(const uint8_t *src, int src_stride,
                                  int16_t *dst, int dst_stride) {
  int16_t tmp[DISFLOW_PATCH_SIZE * (DISFLOW_PATCH_SIZE + 2)];

  // Horizontal filter, using kernel {1, 2, 1}.
  // This kernel can be split into two 2-taps kernels of value {1, 1}.
  // That way we need only 3 add operations to perform the convolution.
  const uint8_t *src_start = src - 1 * src_stride - 1;

  for (int i = 0; i < DISFLOW_PATCH_SIZE + 2; i++) {
    uint8x16_t s = vld1q_u8(src_start + i * src_stride);
    uint8x8_t s0 = vget_low_u8(s);
    uint8x8_t s1 = vget_low_u8(vextq_u8(s, s, 1));
    uint8x8_t s2 = vget_low_u8(vextq_u8(s, s, 2));

    uint16x8_t sum01 = vaddl_u8(s0, s1);
    uint16x8_t sum12 = vaddl_u8(s1, s2);
    uint16x8_t sum = vaddq_u16(sum01, sum12);

    vst1q_s16(tmp + i * DISFLOW_PATCH_SIZE, vreinterpretq_s16_u16(sum));
  }

  // Vertical filter, using kernel {1, 0, -1}.
  // Load the whole block at once to avoid redundant loads during convolution.
  int16x8_t t[10];
  load_s16_8x10(tmp, DISFLOW_PATCH_SIZE, &t[0], &t[1], &t[2], &t[3], &t[4],
                &t[5], &t[6], &t[7], &t[8], &t[9]);

  for (int i = 0; i < DISFLOW_PATCH_SIZE; i++) {
    // Given that the kernel is {1, 0, -1} the convolution is a simple
    // subtraction.
    int16x8_t diff = vsubq_s16(t[i], t[i + 2]);

    vst1q_s16(dst + i * dst_stride, diff);
  }
}

// Computes the components of the system of equations used to solve for
// a flow vector.
//
// The flow equations are a least-squares system, derived as follows:
//
// For each pixel in the patch, we calculate the current error `dt`,
// and the x and y gradients `dx` and `dy` of the source patch.
// This means that, to first order, the squared error for this pixel is
//
//    (dt + u * dx + v * dy)^2
//
// where (u, v) are the incremental changes to the flow vector.
//
// We then want to find the values of u and v which minimize the sum
// of the squared error across all pixels. Conveniently, this fits exactly
// into the form of a least squares problem, with one equation
//
//   u * dx + v * dy = -dt
//
// for each pixel.
//
// Summing across all pixels in a square window of size DISFLOW_PATCH_SIZE,
// and absorbing the - sign elsewhere, this results in the least squares system
//
//   M = |sum(dx * dx)  sum(dx * dy)|
//       |sum(dx * dy)  sum(dy * dy)|
//
//   b = |sum(dx * dt)|
//       |sum(dy * dt)|
static INLINE void compute_flow_matrix(const int16_t *dx, int dx_stride,
                                       const int16_t *dy, int dy_stride,
                                       double *M) {
  int tmp[4] = { 0 };

  for (int i = 0; i < DISFLOW_PATCH_SIZE; i++) {
    for (int j = 0; j < DISFLOW_PATCH_SIZE; j++) {
      tmp[0] += dx[i * dx_stride + j] * dx[i * dx_stride + j];
      tmp[1] += dx[i * dx_stride + j] * dy[i * dy_stride + j];
      // Don't compute tmp[2], as it should be equal to tmp[1]
      tmp[3] += dy[i * dy_stride + j] * dy[i * dy_stride + j];
    }
  }

  // Apply regularization
  // We follow the standard regularization method of adding `k * I` before
  // inverting. This ensures that the matrix will be invertible.
  //
  // Setting the regularization strength k to 1 seems to work well here, as
  // typical values coming from the other equations are very large (1e5 to
  // 1e6, with an upper limit of around 6e7, at the time of writing).
  // It also preserves the property that all matrix values are whole numbers,
  // which is convenient for integerized SIMD implementation.
  tmp[0] += 1;
  tmp[3] += 1;

  tmp[2] = tmp[1];

  M[0] = (double)tmp[0];
  M[1] = (double)tmp[1];
  M[2] = (double)tmp[2];
  M[3] = (double)tmp[3];
}

static INLINE void compute_flow_vector(const int16_t *dx, int dx_stride,
                                       const int16_t *dy, int dy_stride,
                                       const int16_t *dt, int dt_stride,
                                       int *b) {
  memset(b, 0, 2 * sizeof(*b));

  for (int i = 0; i < DISFLOW_PATCH_SIZE; i++) {
    for (int j = 0; j < DISFLOW_PATCH_SIZE; j++) {
      b[0] += dx[i * dx_stride + j] * dt[i * dt_stride + j];
      b[1] += dy[i * dy_stride + j] * dt[i * dt_stride + j];
    }
  }
}

// Try to invert the matrix M
// Note: Due to the nature of how a least-squares matrix is constructed, all of
// the eigenvalues will be >= 0, and therefore det M >= 0 as well.
// The regularization term `+ k * I` further ensures that det M >= k^2.
// As mentioned in compute_flow_matrix(), here we use k = 1, so det M >= 1.
// So we don't have to worry about non-invertible matrices here.
static INLINE void invert_2x2(const double *M, double *M_inv) {
  double det = (M[0] * M[3]) - (M[1] * M[2]);
  assert(det >= 1);
  const double det_inv = 1 / det;

  M_inv[0] = M[3] * det_inv;
  M_inv[1] = -M[1] * det_inv;
  M_inv[2] = -M[2] * det_inv;
  M_inv[3] = M[0] * det_inv;
}

void aom_compute_flow_at_point_neon(const uint8_t *src, const uint8_t *ref,
                                    int x, int y, int width, int height,
                                    int stride, double *u, double *v) {
  double M[4];
  double M_inv[4];
  int b[2];
  int16_t dt[DISFLOW_PATCH_SIZE * DISFLOW_PATCH_SIZE];
  int16_t dx[DISFLOW_PATCH_SIZE * DISFLOW_PATCH_SIZE];
  int16_t dy[DISFLOW_PATCH_SIZE * DISFLOW_PATCH_SIZE];

  // Compute gradients within this patch
  const uint8_t *src_patch = &src[y * stride + x];
  sobel_filter_x(src_patch, stride, dx, DISFLOW_PATCH_SIZE);
  sobel_filter_y(src_patch, stride, dy, DISFLOW_PATCH_SIZE);

  compute_flow_matrix(dx, DISFLOW_PATCH_SIZE, dy, DISFLOW_PATCH_SIZE, M);
  invert_2x2(M, M_inv);

  for (int itr = 0; itr < DISFLOW_MAX_ITR; itr++) {
    compute_flow_error(src, ref, width, height, stride, x, y, *u, *v, dt);
    compute_flow_vector(dx, DISFLOW_PATCH_SIZE, dy, DISFLOW_PATCH_SIZE, dt,
                        DISFLOW_PATCH_SIZE, b);

    // Solve flow equations to find a better estimate for the flow vector
    // at this point
    const double step_u = M_inv[0] * b[0] + M_inv[1] * b[1];
    const double step_v = M_inv[2] * b[0] + M_inv[3] * b[1];
    *u += fclamp(step_u * DISFLOW_STEP_SIZE, -2, 2);
    *v += fclamp(step_v * DISFLOW_STEP_SIZE, -2, 2);

    if (fabs(step_u) + fabs(step_v) < DISFLOW_STEP_SIZE_THRESOLD) {
      // Stop iteration when we're close to convergence
      break;
    }
  }
}
