#ifndef AOM_AV1_COMMON_SPHERICAL_PRED_H_
#define AOM_AV1_COMMON_SPHERICAL_PRED_H_

/*!\brief Convert equirectangular coordinate to plane
 * \param[in]   phi     The latitude in radian ([-pi/2, pi/2])
 * \param[in]   theta   The longitude in radian ([-pi, pi])
 * \param[out]  x       The X coordinate on the plane
 * \param[out]  y       The Y coordinate on the plane
 * \param[in]   width   The width of the frame
 * \param[in]  height  The height of the frame
*/
void equ_to_plane(double phi, double theta, double* x, double* y, int width, int height);

/*!\brief Convert plane coordinate to equirectangular
 * \param[in]  x       The X coordinate on the plane
 * \param[in]  y       The Y coordinate on the plane
 * \param[out] phi     The latitude in radian ([-pi/2, pi/2])
 * \param[out] theta   The longitude in radian ([-pi, pi])
 * \param[in]  width   The width of the frame
 * \param[in]  height  The height of the frame
*/
void plane_to_equ(int x, int y, double *phi, double *theta, int width, int height);

#endif // AOM_AV1_COMMON_SPHERICAL_PRED_H_