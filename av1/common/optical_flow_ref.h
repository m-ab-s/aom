/*
 * Copyright (c) 2017, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#ifndef AV1_COMMON_OPTICAL_FLOW_REF_H_
#define AV1_COMMON_OPTICAL_FLOW_REF_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "./aom_config.h"
#include "av1/common/alloccommon.h"
#include "av1/common/onyxc_int.h"

#if CONFIG_OPFL

#define MAX_ITER_OPTICAL_FLOW 3
#define MAX_OPFL_LEVEL 3
#define MAX_ITER_FAST_OPFL 31  // this has to be an odd number for now.
#define AVG_MF_BORDER 16
#define MAX_MV_LENGTH_1D 160
#define DERIVATIVE_FILTER_LENGTH 7
#define OF_A_SQUARED 100
#define USE_MEDIAN_FILTER 1
#define OPTICAL_FLOW_DIFF_THRES 10.0  // Thres to detect pixel difference
#define OPTICAL_FLOW_REF_THRES 0.3    // Thres to determine reference usage
#define OPTICAL_FLOW_TRUST_MV_THRES 0.3  // Thres to trust motion field
#define FAST_OPTICAL_FLOW 0              // Use fast iterative method
#define OPFL_ANNEAL_FACTOR 1.0  // Annealing factor for laplacian multiplier

typedef enum opfl_blend_method {
  OPFL_SIMPLE_BLEND = 0,
  OPFL_NEAREST_SINGLE,
  OPFL_DIFF_SINGLE,
  OPFL_DIFF_SELECT,
} OPFL_BLEND_METHOD;

// motion field struct with double precision
typedef struct db_mv {
  double row;
  double col;
} DB_MV;

int av1_get_opfl_ref(AV1_COMMON *cm);
void optical_flow_get_ref(YV12_BUFFER_CONFIG *ref0, YV12_BUFFER_CONFIG *ref1,
                          int_mv *mv_left, int_mv *mv_right,
                          YV12_BUFFER_CONFIG *dst, double dst_pos);
void refine_motion_field(YV12_BUFFER_CONFIG *ref0, YV12_BUFFER_CONFIG *ref1,
                         DB_MV *mf_last, DB_MV *mf_new, int level,
                         double dstpos);
double iterate_update_mv(YV12_BUFFER_CONFIG *ref0, YV12_BUFFER_CONFIG *ref1,
                         DB_MV *mf_last, DB_MV *mf_new, int level,
                         double dstpos, double scale);
double iterate_update_mv_fast(YV12_BUFFER_CONFIG *ref0,
                              YV12_BUFFER_CONFIG *ref1, DB_MV *mf_last,
                              DB_MV *mf_new, int level, double dstpos,
                              double scale);
void interp_optical_flow(YV12_BUFFER_CONFIG *ref0, YV12_BUFFER_CONFIG *ref1,
                         DB_MV *mf, YV12_BUFFER_CONFIG *dst, double dst_pos);
void warp_optical_flow_back(YV12_BUFFER_CONFIG *src, YV12_BUFFER_CONFIG *ref,
                            DB_MV *mf_start, int mvstr, YV12_BUFFER_CONFIG *dst,
                            double dstpos);
void warp_optical_flow_back_bilinear(YV12_BUFFER_CONFIG *src,
                                     YV12_BUFFER_CONFIG *ref, DB_MV *mf_start,
                                     int mvstr, YV12_BUFFER_CONFIG *dst,
                                     double dstpos);
void warp_optical_flow_fwd(YV12_BUFFER_CONFIG *src, YV12_BUFFER_CONFIG *ref,
                           DB_MV *mf_start, int mvstr, YV12_BUFFER_CONFIG *dst,
                           double dstpos);
void warp_optical_flow_fwd_bilinear(YV12_BUFFER_CONFIG *src,
                                    YV12_BUFFER_CONFIG *ref, DB_MV *mf_start,
                                    int mvstr, YV12_BUFFER_CONFIG *dst,
                                    double dstpos);
void warp_optical_flow(YV12_BUFFER_CONFIG *src0, YV12_BUFFER_CONFIG *src1,
                       DB_MV *mf_start, int mvstr, YV12_BUFFER_CONFIG *dst,
                       double dstpos, OPFL_BLEND_METHOD method);
void warp_optical_flow_diff_select(YV12_BUFFER_CONFIG *src0,
                                   YV12_BUFFER_CONFIG *src1, DB_MV *mf_start,
                                   int mvstr, YV12_BUFFER_CONFIG *dst,
                                   double dstpos);
uint8_t get_sub_pel_y(uint8_t *src, int stride, double di, double dj);
uint8_t get_sub_pel_uv(uint8_t *src, int stride, double di, double dj);
void create_motion_field(int_mv *mv_left, int_mv *mv_right, DB_MV *mf,
                         int width, int height, int mfstr);
void pad_motion_field_border(DB_MV *mf_start, int width, int height,
                             int stride);
DB_MV median_2D_MV_5x5(DB_MV *center, int mvstr, double *mv_r, double *mv_c,
                       double *left, double *right);
double iter_median_double(double *x, double *left, double *right, int length,
                          int mididx);
int ref_mode_filter_3x3(int *center, int stride, double dstpos);
void upscale_mv_by_2(DB_MV *src, int srcw, int srch, int srcs, DB_MV *dst,
                     int dsts);
int write_image_opfl(const YV12_BUFFER_CONFIG *const ref_buf, char *file_name);
void opfl_get_derivatives(double *Ex, double *Ey, double *Et,
                          YV12_BUFFER_CONFIG *buffer0,
                          YV12_BUFFER_CONFIG *buffer1, double dstpos);

#endif  // CONFIG_OPFL

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AV1_COMMON_OPTICAL_FLOW_REF_H_
