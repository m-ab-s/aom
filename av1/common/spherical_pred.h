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

#ifndef AOM_AV1_COMMON_SPHERICAL_PRED_H_
#define AOM_AV1_COMMON_SPHERICAL_PRED_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

typedef struct {
  double phi;
  double theta;
} SphereMV;

typedef struct {
  double delta_x;
  double delta_y;
} PlaneMV;

typedef struct {
  double r;
  double phi;
  double theta;
} PolarVector;

typedef struct {
  double x;
  double y;
  double z;
} CartesianVector;

/*!\brief Convert equirectangular coordinate to plane
 * \param[in]   phi     The latitude in radian. When phi passes polars,
 *                      theta will go to the other hemisphere (+ pi).
 * \param[in]   theta   The longitude in radian. If it reaches
 *                      (2n + 1) * pi, it will be treated as -pi.
 * \param[in]   width   The width of the frame
 * \param[in]   height  The height of the frame
 * \param[out]  x       The X coordinate on the plane
 * \param[out]  y       The Y coordinate on the plane. It will be slightly
 *                      less than height even when phi = 0.5 * pi.
 */
void av1_sphere_to_plane_erp(double phi, double theta, int width, int height,
                             double *x, double *y);

/*!\brief Convert plane coordinate to equirectangular
 * \param[in]  x       The X coordinate on the plane, will be warpped
 *                     to the other side if < 0 or >= width.
 * \param[in]  y       The Y coordinate on the plane ([0, height))
 * \param[in]  width   The width of the frame
 * \param[in]  height  The height of the frame
 * \param[out] phi     The latitude in radian ([-pi/2, pi/2))
 * \param[out] theta   The longitude in radian ([-pi, pi))
 */
void av1_plane_to_sphere_erp(double x, double y, int width, int height,
                             double *phi, double *theta);

/*!\brief Normalize Cartesian vector.
 * \param[in/out]  carte  The vector to normalize
 */
void av1_normalize_carte_vector(CartesianVector *carte);

/*!\brief Convert polar vector into Cartesian vector.
 * \param[in]   polar  The polar vector
 * \param[out]  carte  The result Cartesian vector
 */
void av1_sphere_polar_to_carte(const PolarVector *polar,
                               CartesianVector *carte);

/*!\brief Convert Cartesian vector into polar vector.
 * \param[in]   carte  The Cartesian vector
 * \param[out]  polar  The result polar vector
 */
void av1_sphere_carte_to_polar(const CartesianVector *carte,
                               PolarVector *polar);

/*!\brief Dot product of two Cartesian vectors.
 * \param[in]  left   The left Cartesian vector
 * \param[in]  right  The right Cartesian vector
 * \return            The product
 */
double av1_carte_vectors_dot_product(const CartesianVector *left,
                                     const CartesianVector *right);

/*!\brief Dot product of two Cartesian vectors.
 * \param[in]   left    The left Cartesian vector
 * \param[in]   right   The right Cartesian vector
 * \param[out]  result  The result Cartesian vector
 */
void av1_carte_vectors_cross_product(const CartesianVector *left,
                                     const CartesianVector *right,
                                     CartesianVector *result);

/*!\brief Given a spherical motion vector (delat_phi, delta_theta) and a
 * block on the current frame, find the corresponding pixels in the
 * reference frame for each pixel in the given block.
 * \param[in]   block_x             Block's upper left corner X on the plane
 * \param[in]   block_y             Block's upper left corner Y on the plane
 * \param[in]   block_width         Width of the block
 * \param[in]   block_height        Height of the block
 * \param[in]   delta_phi           Latitude motion vector, added to the
 *                                  latitude of each pixel in the block
 * \param[in]   delta_theta         Longitude motion vector, added to the
 *                                  longitude of each pixel in the block
 * \param[in]   ref_frame           Reference frame data
 * \param[in]   ref_frame_stride    Stride of reference frame data
 * \param[in]   frame_width         Width of frame
 * \param[in]   frame_height        Height of frame
 * \param[in]   pred_block_stride   Stride of the predicted block
 * \param[out]  pred_block          Predicted block
 */
void av1_get_pred_erp(int block_x, int block_y, int block_width,
                      int block_height, double delta_phi, double delta_theta,
                      const uint8_t *ref_frame, int ref_frame_stride,
                      int frame_width, int frame_height, int pred_block_stride,
                      uint8_t *pred_block);

/*!\brief Spherical motion search for one block in brute force mode
 * \param[in]       block_x             Block's upper left corner X on the plane
 * \param[in]       block_y             Block's upper left corner Y on the plane
 * \param[in]       block_width         Width of the block
 * \param[in]       block_height        Height of the block
 * \param[in]       cur_frame           Current frame data
 * \param[in]       ref_frame           Reference frame data
 * \param[in]       frame_stride        Stride of frame data
 * \param[in]       frame_width         Width of frame
 * \param[in]       frame_height        Height of frame
 * \param[in]       search_range        Range of search
 * \param[out]      best_mv             Best spherical motion vector
 * \return                              Best SAD
 */
int av1_motion_search_brute_force_erp(
    int block_x, int block_y, int block_width, int block_height,
    const uint8_t *cur_frame, const uint8_t *ref_frame, int frame_stride,
    int frame_width, int frame_height, int search_range, SphereMV *best_mv);

/*!\brief Spherical motion search for one block in diamond mode
 * \param[in]       block_x             Block's upper left corner X on the plane
 * \param[in]       block_y             Block's upper left corner Y on the plane
 * \param[in]       block_width         Width of the block
 * \param[in]       block_height        Height of the block
 * \param[in]       cur_frame           Current frame data
 * \param[in]       ref_frame           Reference frame data
 * \param[in]       frame_stride        Stride of frame data
 * \param[in]       frame_width         Width of frame
 * \param[in]       frame_height        Height of frame
 * \param[in]       search_range        Range of search in pixel
 * \param[in]       start_mv            Initial motion vector
 * \param[out]      best_mv             Best spherical motion vector
 * \return                              Best SAD
 */
int av1_motion_search_diamond_erp(int block_x, int block_y, int block_width,
                                  int block_height, const uint8_t *cur_frame,
                                  const uint8_t *ref_frame, int frame_stride,
                                  int frame_width, int frame_height,
                                  int search_range, const SphereMV *start_mv,
                                  SphereMV *best_mv);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_SPHERICAL_PRED_H_
