/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#ifndef AOM_AV1_COMMON_CONVOLVE_H_
#define AOM_AV1_COMMON_CONVOLVE_H_

#include <stdbool.h>
#include "av1/common/filter.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint16_t CONV_BUF_TYPE;
typedef struct ConvolveParams {
  int do_average;
  int filter_bits;
  CONV_BUF_TYPE *dst;
  int dst_stride;
  int round_0;
  int round_1;
  int plane;
  int is_compound;
  int use_dist_wtd_comp_avg;
  int fwd_offset;
  int bck_offset;
} ConvolveParams;

#define NONSEP_PIXELS_MAX 32
#define NONSEP_COEFFS_MAX 32
#define NONSEP_ROW_ID 0
#define NONSEP_COL_ID 1
#define NONSEP_BUF_POS 2

static INLINE int16_t clip_base(int16_t x, int bit_depth) {
  (void)bit_depth;
  return x;
}

typedef struct NonsepFilterConfig {
  int prec_bits;
  int num_pixels;
  int num_pixels2;
  const int (*config)[3];
  int strict_bounds;
} NonsepFilterConfig;

// Nonseparable convolution
void av1_convolve_nonsep(const uint8_t *dgd, int width, int height, int stride,
                         const NonsepFilterConfig *config,
                         const int16_t *filter, uint8_t *dst, int dst_stride);
void av1_convolve_nonsep_highbd(const uint8_t *dgd, int width, int height,
                                int stride, const NonsepFilterConfig *config,
                                const int16_t *filter, uint8_t *dst,
                                int dst_stride, int bit_depth);

// Nonseparable convolution with dual input planes - used for cross component
// filtering
void av1_convolve_nonsep_dual(const uint8_t *dgd, int width, int height,
                              int stride, const uint8_t *dgd2, int stride2,
                              const NonsepFilterConfig *config,
                              const int16_t *filter, uint8_t *dst,
                              int dst_stride);
void av1_convolve_nonsep_dual_highbd(const uint8_t *dgd, int width, int height,
                                     int stride, const uint8_t *dgd2,
                                     int stride2,
                                     const NonsepFilterConfig *config,
                                     const int16_t *filter, uint8_t *dst,
                                     int dst_stride, int bit_depth);

#if CONFIG_NEW_TX64X64
// Nonseparable classified convolution with different filters used for each
// pixel based on its class as specified by supplied class map.
void av1_convolve_nonsep_cls(const uint8_t *dgd, int width, int height,
                             int stride, const NonsepFilterConfig *nsfilter,
                             const uint8_t *cls, int cls_stride,
                             const int16_t *filter, int filter_stride,
                             uint8_t *dst, int dst_stride);
void av1_convolve_nonsep_cls_highbd(const uint16_t *dgd, int width, int height,
                                    int stride,
                                    const NonsepFilterConfig *nsfilter,
                                    const uint8_t *cls, int cls_stride,
                                    const int16_t *filter, int filter_stride,
                                    uint16_t *dst, int dst_stride,
                                    int bit_depth);
#endif  // CONFIG_NEW_TX64X64

#define ROUND0_BITS 3
#define COMPOUND_ROUND1_BITS 7
#define WIENER_ROUND0_BITS 3

#define WIENER_CLAMP_LIMIT(fb, r0, bd) (1 << ((bd) + 1 + (fb) - (r0)))

typedef void (*aom_convolve_fn_t)(const uint8_t *src, int src_stride,
                                  uint8_t *dst, int dst_stride, int w, int h,
                                  const InterpFilterParams *filter_params_x,
                                  const InterpFilterParams *filter_params_y,
                                  const int subpel_x_qn, const int subpel_y_qn,
                                  ConvolveParams *conv_params);

typedef void (*aom_highbd_convolve_fn_t)(
    const uint16_t *src, int src_stride, uint16_t *dst, int dst_stride, int w,
    int h, const InterpFilterParams *filter_params_x,
    const InterpFilterParams *filter_params_y, const int subpel_x_qn,
    const int subpel_y_qn, ConvolveParams *conv_params, int bd);

struct AV1Common;
struct scale_factors;

void av1_convolve_2d_facade(const uint8_t *src, int src_stride, uint8_t *dst,
                            int dst_stride, int w, int h, int orig_w,
                            int orig_h, int_interpfilters interp_filters,
                            const int subpel_x_qn, int x_step_q4,
                            const int subpel_y_qn, int y_step_q4, int scaled,
                            ConvolveParams *conv_params,
                            const struct scale_factors *sf, int is_intrabc);

static INLINE ConvolveParams get_conv_params_no_round(int do_average, int plane,
                                                      CONV_BUF_TYPE *dst,
                                                      int dst_stride,
                                                      int is_compound, int bd) {
  ConvolveParams conv_params;
  conv_params.do_average = do_average;
  assert(IMPLIES(do_average, is_compound));
  conv_params.filter_bits = FILTER_BITS;
  conv_params.is_compound = is_compound;
  conv_params.round_0 = ROUND0_BITS;
  conv_params.round_1 = is_compound ? COMPOUND_ROUND1_BITS
                                    : 2 * FILTER_BITS - conv_params.round_0;
  const int intbufrange = bd + FILTER_BITS - conv_params.round_0 + 2;
  assert(IMPLIES(bd < 12, intbufrange <= 16));
  if (intbufrange > 16) {
    conv_params.round_0 += intbufrange - 16;
    if (!is_compound) conv_params.round_1 -= intbufrange - 16;
  }
  // TODO(yunqing): The following dst should only be valid while
  // is_compound = 1;
  conv_params.dst = dst;
  conv_params.dst_stride = dst_stride;
  conv_params.plane = plane;
  return conv_params;
}

static INLINE ConvolveParams get_conv_params(int do_average, int plane,
                                             int bd) {
  return get_conv_params_no_round(do_average, plane, NULL, 0, 0, bd);
}

static INLINE ConvolveParams get_conv_params_wiener(int bd, int filter_bits) {
  ConvolveParams conv_params;
  (void)bd;
  conv_params.filter_bits = filter_bits;
  conv_params.do_average = 0;
  conv_params.is_compound = 0;
  conv_params.round_0 = WIENER_ROUND0_BITS;
  conv_params.round_1 = 2 * filter_bits - conv_params.round_0;
  const int intbufrange = bd + filter_bits - conv_params.round_0 + 2;
  assert(IMPLIES(bd < 12, intbufrange <= 16));
  if (intbufrange > 16) {
    conv_params.round_0 += intbufrange - 16;
    conv_params.round_1 -= intbufrange - 16;
  }
  conv_params.dst = NULL;
  conv_params.dst_stride = 0;
  conv_params.plane = 0;
  return conv_params;
}

#if CONFIG_WIENER_SEP_HIPREC
static INLINE ConvolveParams get_conv_params_wiener_hp(int bd,
                                                       int filter_bits) {
  ConvolveParams conv_params;
  (void)bd;
  conv_params.filter_bits = filter_bits;
  conv_params.do_average = 0;
  conv_params.is_compound = 0;
  conv_params.round_0 = WIENER_ROUND0_BITS + filter_bits - FILTER_BITS;
  conv_params.round_1 = 2 * filter_bits - conv_params.round_0;
  const int intbufrange = bd + filter_bits - conv_params.round_0 + 2;
  assert(IMPLIES(bd < 12, intbufrange <= 16));
  if (intbufrange > 16) {
    conv_params.round_0 += intbufrange - 16;
    conv_params.round_1 -= intbufrange - 16;
  }
  conv_params.dst = NULL;
  conv_params.dst_stride = 0;
  conv_params.plane = 0;
  return conv_params;
}
#endif  // CONFIG_WIENER_SEP_HIPREC

void av1_highbd_convolve_2d_facade(
    const uint8_t *src8, int src_stride, uint8_t *dst, int dst_stride, int w,
    int h, int orig_w, int orig_h, int_interpfilters interp_filters,
    const int subpel_x_qn, int x_step_q4, const int subpel_y_qn, int y_step_q4,
    int scaled, ConvolveParams *conv_params, const struct scale_factors *sf,
    int is_intrabc, int bd);

// TODO(sarahparker) This will need to be integerized and optimized
void av1_convolve_2d_sobel_y_c(const uint8_t *src, int src_stride, double *dst,
                               int dst_stride, int w, int h, int dir,
                               double norm);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_CONVOLVE_H_
