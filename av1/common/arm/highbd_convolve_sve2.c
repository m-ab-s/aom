/*
 * Copyright (c) 2024, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <assert.h>
#include <arm_neon.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"

#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/arm/dot_sve.h"
#include "aom_dsp/arm/mem_neon.h"
#include "aom_ports/mem.h"
#include "av1/common/convolve.h"
#include "av1/common/filter.h"

DECLARE_ALIGNED(16, static const uint16_t, kDotProd12TapTbl[32]) = {
  0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6,
  4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10,
};

static INLINE int16x8_t aom_tbl_s16(int16x8_t s, uint16x8_t tbl) {
  return svget_neonq_s16(svtbl_s16(svset_neonq_s16(svundef_s16(), s),
                                   svset_neonq_u16(svundef_u16(), tbl)));
}

static INLINE int16x8_t aom_tbl2_s16(int16x8_t s0, int16x8_t s1,
                                     uint16x8_t tbl) {
  svint16x2_t samples = svcreate2_s16(svset_neonq_s16(svundef_s16(), s0),
                                      svset_neonq_s16(svundef_s16(), s1));
  return svget_neonq_s16(
      svtbl2_s16(samples, svset_neonq_u16(svundef_u16(), tbl)));
}

static INLINE uint16x4_t convolve12_4_x(
    int16x8_t s0, int16x8_t s1, int16x8_t filter_0_7, int16x8_t filter_4_11,
    const int64x2_t offset, uint16x8x4_t permute_tbl, uint16x4_t max) {
  int16x8_t permuted_samples[6];
  permuted_samples[0] = aom_tbl_s16(s0, permute_tbl.val[0]);
  permuted_samples[1] = aom_tbl_s16(s0, permute_tbl.val[1]);
  permuted_samples[2] = aom_tbl2_s16(s0, s1, permute_tbl.val[2]);
  permuted_samples[3] = aom_tbl2_s16(s0, s1, permute_tbl.val[3]);
  permuted_samples[4] = aom_tbl_s16(s1, permute_tbl.val[0]);
  permuted_samples[5] = aom_tbl_s16(s1, permute_tbl.val[1]);

  int64x2_t sum01 =
      aom_svdot_lane_s16(offset, permuted_samples[0], filter_0_7, 0);
  sum01 = aom_svdot_lane_s16(sum01, permuted_samples[2], filter_0_7, 1);
  sum01 = aom_svdot_lane_s16(sum01, permuted_samples[4], filter_4_11, 1);

  int64x2_t sum23 =
      aom_svdot_lane_s16(offset, permuted_samples[1], filter_0_7, 0);
  sum23 = aom_svdot_lane_s16(sum23, permuted_samples[3], filter_0_7, 1);
  sum23 = aom_svdot_lane_s16(sum23, permuted_samples[5], filter_4_11, 1);

  int32x4_t res0123 = vcombine_s32(vmovn_s64(sum01), vmovn_s64(sum23));
  uint16x4_t res = vqrshrun_n_s32(res0123, FILTER_BITS);

  return vmin_u16(res, max);
}

static INLINE uint16x8_t convolve12_8_x(int16x8_t s0, int16x8_t s1,
                                        int16x8_t s2, int16x8_t filter_0_7,
                                        int16x8_t filter_4_11, int64x2_t offset,
                                        uint16x8x4_t permute_tbl,
                                        uint16x8_t max) {
  int16x8_t permuted_samples[8];
  permuted_samples[0] = aom_tbl_s16(s0, permute_tbl.val[0]);
  permuted_samples[1] = aom_tbl_s16(s0, permute_tbl.val[1]);
  permuted_samples[2] = aom_tbl2_s16(s0, s1, permute_tbl.val[2]);
  permuted_samples[3] = aom_tbl2_s16(s0, s1, permute_tbl.val[3]);
  permuted_samples[4] = aom_tbl_s16(s1, permute_tbl.val[0]);
  permuted_samples[5] = aom_tbl_s16(s1, permute_tbl.val[1]);
  permuted_samples[6] = aom_tbl2_s16(s1, s2, permute_tbl.val[2]);
  permuted_samples[7] = aom_tbl2_s16(s1, s2, permute_tbl.val[3]);

  int64x2_t sum01 =
      aom_svdot_lane_s16(offset, permuted_samples[0], filter_0_7, 0);
  sum01 = aom_svdot_lane_s16(sum01, permuted_samples[2], filter_0_7, 1);
  sum01 = aom_svdot_lane_s16(sum01, permuted_samples[4], filter_4_11, 1);

  int64x2_t sum23 =
      aom_svdot_lane_s16(offset, permuted_samples[1], filter_0_7, 0);
  sum23 = aom_svdot_lane_s16(sum23, permuted_samples[3], filter_0_7, 1);
  sum23 = aom_svdot_lane_s16(sum23, permuted_samples[5], filter_4_11, 1);

  int64x2_t sum45 =
      aom_svdot_lane_s16(offset, permuted_samples[2], filter_0_7, 0);
  sum45 = aom_svdot_lane_s16(sum45, permuted_samples[4], filter_0_7, 1);
  sum45 = aom_svdot_lane_s16(sum45, permuted_samples[6], filter_4_11, 1);

  int64x2_t sum67 =
      aom_svdot_lane_s16(offset, permuted_samples[3], filter_0_7, 0);
  sum67 = aom_svdot_lane_s16(sum67, permuted_samples[5], filter_0_7, 1);
  sum67 = aom_svdot_lane_s16(sum67, permuted_samples[7], filter_4_11, 1);

  int32x4_t sum0123 = vcombine_s32(vmovn_s64(sum01), vmovn_s64(sum23));
  int32x4_t sum4567 = vcombine_s32(vmovn_s64(sum45), vmovn_s64(sum67));

  uint16x8_t res = vcombine_u16(vqrshrun_n_s32(sum0123, FILTER_BITS),
                                vqrshrun_n_s32(sum4567, FILTER_BITS));

  return vminq_u16(res, max);
}

static INLINE void highbd_convolve_x_sr_12tap_sve2(
    const uint16_t *src, int src_stride, uint16_t *dst, int dst_stride,
    int width, int height, const int16_t *y_filter_ptr,
    ConvolveParams *conv_params, int bd) {
  // This shim allows to do only one rounding shift instead of two.
  const int64x2_t offset = vdupq_n_s64(1 << (conv_params->round_0 - 1));

  const int16x8_t y_filter_0_7 = vld1q_s16(y_filter_ptr);
  const int16x8_t y_filter_4_11 = vld1q_s16(y_filter_ptr + 4);

  uint16x8x4_t permute_tbl = vld1q_u16_x4(kDotProd12TapTbl);

  if (width == 4) {
    const uint16x4_t max = vdup_n_u16((1 << bd) - 1);
    const int16_t *s = (const int16_t *)src;

    do {
      int16x8_t s0, s1, s2, s3, s4, s5, s6, s7;
      load_s16_8x4(s, src_stride, &s0, &s2, &s4, &s6);
      load_s16_8x4(s + 8, src_stride, &s1, &s3, &s5, &s7);

      uint16x4_t d0 = convolve12_4_x(s0, s1, y_filter_0_7, y_filter_4_11,
                                     offset, permute_tbl, max);
      uint16x4_t d1 = convolve12_4_x(s2, s3, y_filter_0_7, y_filter_4_11,
                                     offset, permute_tbl, max);
      uint16x4_t d2 = convolve12_4_x(s4, s5, y_filter_0_7, y_filter_4_11,
                                     offset, permute_tbl, max);
      uint16x4_t d3 = convolve12_4_x(s6, s7, y_filter_0_7, y_filter_4_11,
                                     offset, permute_tbl, max);

      store_u16_4x4(dst, dst_stride, d0, d1, d2, d3);

      s += 4 * src_stride;
      dst += 4 * dst_stride;
      height -= 4;
    } while (height != 0);
  } else {
    const uint16x8_t max = vdupq_n_u16((1 << bd) - 1);

    do {
      const int16_t *s = (const int16_t *)src;
      uint16_t *d = dst;
      int w = width;

      do {
        int16x8_t s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11;
        load_s16_8x4(s, src_stride, &s0, &s3, &s6, &s9);
        load_s16_8x4(s + 8, src_stride, &s1, &s4, &s7, &s10);
        load_s16_8x4(s + 16, src_stride, &s2, &s5, &s8, &s11);

        uint16x8_t d0 = convolve12_8_x(s0, s1, s2, y_filter_0_7, y_filter_4_11,
                                       offset, permute_tbl, max);
        uint16x8_t d1 = convolve12_8_x(s3, s4, s5, y_filter_0_7, y_filter_4_11,
                                       offset, permute_tbl, max);
        uint16x8_t d2 = convolve12_8_x(s6, s7, s8, y_filter_0_7, y_filter_4_11,
                                       offset, permute_tbl, max);
        uint16x8_t d3 = convolve12_8_x(s9, s10, s11, y_filter_0_7,
                                       y_filter_4_11, offset, permute_tbl, max);

        store_u16_8x4(d, dst_stride, d0, d1, d2, d3);

        s += 8;
        d += 8;
        w -= 8;
      } while (w != 0);
      src += 4 * src_stride;
      dst += 4 * dst_stride;
      height -= 4;
    } while (height != 0);
  }
}

void av1_highbd_convolve_x_sr_sve2(const uint16_t *src, int src_stride,
                                   uint16_t *dst, int dst_stride, int w, int h,
                                   const InterpFilterParams *filter_params_x,
                                   const int subpel_x_qn,
                                   ConvolveParams *conv_params, int bd) {
  if (w == 2 || h == 2) {
    av1_highbd_convolve_x_sr_c(src, src_stride, dst, dst_stride, w, h,
                               filter_params_x, subpel_x_qn, conv_params, bd);
    return;
  }

  const int x_filter_taps = get_filter_tap(filter_params_x, subpel_x_qn);

  if (x_filter_taps != 12) {
    av1_highbd_convolve_x_sr_neon(src, src_stride, dst, dst_stride, w, h,
                                  filter_params_x, subpel_x_qn, conv_params,
                                  bd);
    return;
  }

  const int horiz_offset = filter_params_x->taps / 2 - 1;
  const int16_t *x_filter_ptr = av1_get_interp_filter_subpel_kernel(
      filter_params_x, subpel_x_qn & SUBPEL_MASK);

  src -= horiz_offset;

  highbd_convolve_x_sr_12tap_sve2(src, src_stride, dst, dst_stride, w, h,
                                  x_filter_ptr, conv_params, bd);
  return;
}
