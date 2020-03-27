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
#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"
#include "config/aom_scale_rtcd.h"
#include "config/av1_rtcd.h"
#include "av1/encoder/mv_prec.h"

#define MV_PREC_DET_SOBEL_EDGE_THRESH 256
#define MV_PREC_DET_SSE_STATIC_THRESH 1024
#define MV_PREC_DET_BLK_SIZE_BITS 4
#define MV_PREC_DET_BLK_SIZE (1 << MV_PREC_DET_BLK_SIZE_BITS)
#define MV_PREC_DET_THRESH 0.20
#define MV_PREC_DET_THRESH2 0.10
#define MV_PREC_DET_QTHRESH 64  // Q thresh for 1/8-pel in edge based method

#define MV_HIPREC_QTHRESH 128   // Q thresh for 1/8-pel in q-based method
#define MV_HIPREC_QTHRESH2 192  // Q thresh for 1/4-pel in q-based method

// Q thresh for 1/8-pel in edge based method

// Compute edge energy in the frame
static MvSubpelPrecision determine_frame_mv_precision(const AV1_COMP *cpi,
                                                      int q, int use_edges) {
  (void)cpi;
  if (!use_edges) {
    return q < MV_HIPREC_QTHRESH ? MV_SUBPEL_EIGHTH_PRECISION
                                 : MV_SUBPEL_QTR_PRECISION;
  }
  if (q < MV_PREC_DET_QTHRESH) return MV_SUBPEL_EIGHTH_PRECISION;
  const YV12_BUFFER_CONFIG *srcbuf = cpi->source;
  const YV12_BUFFER_CONFIG *refbuf =
      get_ref_frame_yv12_buf(&cpi->common, GOLDEN_FRAME);
  const int bd = cpi->td.mb.e_mbd.bd;
  const int width = srcbuf->y_crop_width;
  const int height = srcbuf->y_crop_height;
  const int stride = srcbuf->y_stride;
  const int ref_stride = refbuf->y_stride;
  int num_blks[2] = { 0, 0 };
  if (srcbuf->flags & YV12_FLAG_HIGHBITDEPTH) {
    const uint16_t *src16 =
        (const uint16_t *)CONVERT_TO_SHORTPTR(srcbuf->y_buffer);
    const uint16_t *ref16 =
        (const uint16_t *)CONVERT_TO_SHORTPTR(refbuf->y_buffer);
    for (int i = 0; i < height - MV_PREC_DET_BLK_SIZE;
         i += MV_PREC_DET_BLK_SIZE) {
      for (int j = 0; j < width - MV_PREC_DET_BLK_SIZE;
           j += MV_PREC_DET_BLK_SIZE) {
        const uint16_t *src = src16 + i * stride + j;
        const uint16_t *ref = ref16 + i * stride + j;
        int64_t sse = aom_highbd_sse(
            CONVERT_TO_BYTEPTR(src), stride, CONVERT_TO_BYTEPTR(ref),
            ref_stride, MV_PREC_DET_BLK_SIZE, MV_PREC_DET_BLK_SIZE);
        int64_t sse_norm = ROUND_POWER_OF_TWO(
            sse, 2 * MV_PREC_DET_BLK_SIZE_BITS + 2 * (bd - 8));
        if (sse_norm < MV_PREC_DET_SSE_STATIC_THRESH) continue;
        int64_t gx = 0, gy = 0, g;
        for (int y = 0; y < MV_PREC_DET_BLK_SIZE; ++y) {
          for (int x = 0; x < MV_PREC_DET_BLK_SIZE; ++x) {
            gx +=
                abs(src[-stride + 1] - src[-stride - 1] +
                    (src[1] - src[-1]) * 2 + src[stride + 1] - src[stride - 1]);
            gy += abs(src[stride - 1] - src[-stride - 1] +
                      (src[stride] - src[-stride]) * 2 + src[stride + 1] -
                      src[-stride + 1]);
            src++;
          }
          src += stride - MV_PREC_DET_BLK_SIZE;
        }
        g = gx * gx + gy * gy;
        // Normalize to per pixel and bit-depth of 8
        g = ROUND_POWER_OF_TWO(
            g, 4 * MV_PREC_DET_BLK_SIZE_BITS + 4 + 2 * (bd - 8));
        ++num_blks[g > MV_PREC_DET_SOBEL_EDGE_THRESH];
      }
    }
  } else {
    const uint8_t *src8 = srcbuf->y_buffer;
    const uint8_t *ref8 = refbuf->y_buffer;
    for (int i = 0; i < height - MV_PREC_DET_BLK_SIZE;
         i += MV_PREC_DET_BLK_SIZE) {
      for (int j = 0; j < width - MV_PREC_DET_BLK_SIZE;
           j += MV_PREC_DET_BLK_SIZE) {
        const uint8_t *src = src8 + i * stride + j;
        const uint8_t *ref = ref8 + i * stride + j;
        int64_t sse = aom_sse(src, stride, ref, ref_stride,
                              MV_PREC_DET_BLK_SIZE, MV_PREC_DET_BLK_SIZE);
        int64_t sse_norm =
            ROUND_POWER_OF_TWO(sse, 2 * MV_PREC_DET_BLK_SIZE_BITS);
        if (sse_norm < MV_PREC_DET_SSE_STATIC_THRESH) continue;
        int64_t gx = 0, gy = 0, g;
        for (int y = 0; y < MV_PREC_DET_BLK_SIZE; ++y) {
          for (int x = 0; x < MV_PREC_DET_BLK_SIZE; ++x) {
            gx +=
                abs(src[-stride + 1] - src[-stride - 1] +
                    (src[1] - src[-1]) * 2 + src[stride + 1] - src[stride - 1]);
            gy += abs(src[stride - 1] - src[-stride - 1] +
                      (src[stride] - src[-stride]) * 2 + src[stride + 1] -
                      src[-stride + 1]);
            src++;
          }
          src += stride - MV_PREC_DET_BLK_SIZE;
        }
        g = gx * gx + gy * gy;
        // Normalize to per pixel and bit-depth of 8
        g = ROUND_POWER_OF_TWO(
            g, 4 * MV_PREC_DET_BLK_SIZE_BITS + 4 + 2 * (bd - 8));
        ++num_blks[g > MV_PREC_DET_SOBEL_EDGE_THRESH];
      }
    }
  }
  if (num_blks[0] + num_blks[1] == 0) return MV_SUBPEL_QTR_PRECISION;
  const double pct_edge_blks =
      100.0 * (double)num_blks[1] / (num_blks[0] + num_blks[1]);
  const double pct_edge_blks_by_q = pct_edge_blks / q;
  // printf("pct_edge_blks = %f [%d + %d], q = %d, ratio = %f\n", pct_edge_blks,
  //        num_blks[0], num_blks[1], q, pct_edge_blks_by_q);
  if (pct_edge_blks_by_q >= MV_PREC_DET_THRESH)
    return MV_SUBPEL_EIGHTH_PRECISION;
  return MV_SUBPEL_QTR_PRECISION;
}

#if CONFIG_FLEX_MVRES && !CONFIG_SB_FLEX_MVRES
#define FLEX_MV_PRECISION_QTHRESH 256  // Reduce to turn off at low quality
static int determine_flex_mv_precision(const AV1_COMP *cpi, int q) {
  return (cpi->common.fr_mv_precision >= MV_SUBPEL_QTR_PRECISION &&
          !is_stat_generation_stage(cpi) && q <= FLEX_MV_PRECISION_QTHRESH);
}
#endif  // CONFIG_FLEX_MVRES && !CONFIG_SB_FLEX_MVRES

void av1_pick_and_set_high_precision_mv(AV1_COMP *cpi, int q) {
  MvSubpelPrecision precision = cpi->common.cur_frame_force_integer_mv
                                    ? MV_SUBPEL_NONE
                                    : determine_frame_mv_precision(cpi, q, 0);
  assert(IMPLIES(!cpi->common.cur_frame_force_integer_mv,
                 precision >= MV_SUBPEL_QTR_PRECISION));
  av1_set_mv_precision(cpi, precision, cpi->common.cur_frame_force_integer_mv);
#if CONFIG_FLEX_MVRES && !CONFIG_SB_FLEX_MVRES
  cpi->common.use_pb_mv_precision = determine_flex_mv_precision(cpi, q);
#endif  // CONFIG_FLEX_MVRES && !CONFIG_SB_FLEX_MVRES
}
