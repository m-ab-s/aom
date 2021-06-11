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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_SPHERICAL_PRED_H_
