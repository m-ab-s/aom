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

#include <assert.h>
#include <math.h>
#include <string.h>

#include "config/aom_scale_rtcd.h"

#include "aom/aom_integer.h"
#include "av1/common/ccso.h"
#include "av1/common/reconinter.h"

/* Pad the border of a frame */
void extend_ccso_border(uint16_t *buf, const int d, MACROBLOCKD *xd) {
  int s = xd->plane[0].dst.width + (CCSO_PADDING_SIZE << 1);
  uint16_t *p = &buf[d * s + d];
  int h = xd->plane[0].dst.height;
  int w = xd->plane[0].dst.width;
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < d; x++) {
      *(p - d + x) = p[0];
      p[w + x] = p[w - 1];
    }
    p += s;
  }
  p -= (s + d);
  for (int y = 0; y < d; y++) {
    memcpy(p + (y + 1) * s, p, sizeof(uint16_t) * (w + (d << 1)));
  }
  p -= ((h - 1) * s);
  for (int y = 0; y < d; y++) {
    memcpy(p - (y + 1) * s, p, sizeof(uint16_t) * (w + (d << 1)));
  }
}

/* Derive the quantized index, later it can be used for retriving offset values
 * from the look-up table */
void cal_filter_support(int *rec_luma_idx, const uint16_t *rec_y,
                        const uint8_t quant_step_size, const int inv_quant_step,
                        const int *rec_idx) {
  for (int i = 0; i < 2; i++) {
    int d = rec_y[rec_idx[i]] - rec_y[0];
    if (d > quant_step_size)
      rec_luma_idx[i] = 2;
    else if (d < inv_quant_step)
      rec_luma_idx[i] = 0;
    else
      rec_luma_idx[i] = 1;
  }
}

/* Derive sample locations for CCSO */
void derive_ccso_sample_pos(int *rec_idx, const int ccso_stride,
                            const uint8_t ext_filter_support) {
  // Input sample locations for CCSO
  //         2 1 4
  // 6 o 5 o 3 o 3 o 5 o 6
  //         4 1 2
  if (ext_filter_support == 0) {
    // Sample position 1
    rec_idx[0] = -1 * ccso_stride;
    rec_idx[1] = 1 * ccso_stride;
  } else if (ext_filter_support == 1) {
    // Sample position 2
    rec_idx[0] = -1 * ccso_stride - 1;
    rec_idx[1] = 1 * ccso_stride + 1;
  } else if (ext_filter_support == 2) {
    // Sample position 3
    rec_idx[0] = -1;
    rec_idx[1] = 1;
  } else if (ext_filter_support == 3) {
    // Sample position 4
    rec_idx[0] = 1 * ccso_stride - 1;
    rec_idx[1] = -1 * ccso_stride + 1;
  } else if (ext_filter_support == 4) {
    // Sample position 5
    rec_idx[0] = -3;
    rec_idx[1] = 3;
  } else {  // if(ext_filter_support == 5) {
    // Sample position 6
    rec_idx[0] = -5;
    rec_idx[1] = 5;
  }
}

/* Apply CCSO for one color component (low bit-depth) */
void apply_ccso_filter(AV1_COMMON *cm, MACROBLOCKD *xd, const int plane,
                       const uint16_t *temp_rec_y_buf, uint8_t *rec_yv_8,
                       const int dst_stride, const int8_t *filter_offset,
                       const uint8_t quant_step_size,
                       const uint8_t ext_filter_support) {
  const CommonModeInfoParams *const mi_params = &cm->mi_params;
  const int ccso_stride_ext = xd->plane[0].dst.width + (CCSO_PADDING_SIZE << 1);
  const int pic_height_c = xd->plane[1].dst.height;
  const int pic_width_c = xd->plane[1].dst.width;
  int rec_luma_idx[2];
  const int inv_quant_step = quant_step_size * -1;
  int rec_idx[2];

  derive_ccso_sample_pos(rec_idx, ccso_stride_ext, ext_filter_support);

  const int8_t *offset_buf;
  if (plane > 0) {
    offset_buf = cm->ccso_info.filter_offset[plane - 1];
  } else {
    offset_buf = filter_offset;
  }
  int ccso_stride_ext_idx[1 << CCSO_BLK_SIZE];
  int dst_stride_idx[1 << CCSO_BLK_SIZE];
  for (int i = 0; i < (1 << CCSO_BLK_SIZE); i++) {
    ccso_stride_ext_idx[i] = ccso_stride_ext * i;
    dst_stride_idx[i] = dst_stride * i;
  }
  const int pad_stride =
      CCSO_PADDING_SIZE * ccso_stride_ext + CCSO_PADDING_SIZE;
  const int y_uv_hori_scale = xd->plane[1].subsampling_x;
  const int y_uv_vert_scale = xd->plane[1].subsampling_y;
  for (int y = 0; y < pic_height_c; y += (1 << CCSO_BLK_SIZE)) {
    for (int x = 0; x < pic_width_c; x += (1 << CCSO_BLK_SIZE)) {
      if (plane > 0) {
        const int ccso_blk_idx =
            (1 << CCSO_BLK_SIZE >>
             (MI_SIZE_LOG2 - xd->plane[plane].subsampling_y)) *
                (y >> CCSO_BLK_SIZE) * mi_params->mi_stride +
            (1 << CCSO_BLK_SIZE >>
             (MI_SIZE_LOG2 - xd->plane[plane].subsampling_x)) *
                (x >> CCSO_BLK_SIZE);
        const bool use_ccso =
            (plane == 1) ? mi_params->mi_grid_base[ccso_blk_idx]->ccso_blk_u
                         : mi_params->mi_grid_base[ccso_blk_idx]->ccso_blk_v;
        if (!use_ccso) continue;
      }
      int y_offset;
      int x_offset;
      if (y + (1 << CCSO_BLK_SIZE) >= pic_height_c)
        y_offset = pic_height_c - y;
      else
        y_offset = (1 << CCSO_BLK_SIZE);

      if (x + (1 << CCSO_BLK_SIZE) >= pic_width_c)
        x_offset = pic_width_c - x;
      else
        x_offset = (1 << CCSO_BLK_SIZE);

      for (int y_off = 0; y_off < y_offset; y_off++) {
        for (int x_off = 0; x_off < x_offset; x_off++) {
          cal_filter_support(
              rec_luma_idx,
              &temp_rec_y_buf[((ccso_stride_ext_idx[y_off] << y_uv_vert_scale) +
                               ((x + x_off) << y_uv_hori_scale)) +
                              pad_stride],
              quant_step_size, inv_quant_step, rec_idx);
          int offset_val = offset_buf[(rec_luma_idx[0] << 2) + rec_luma_idx[1]];
          rec_yv_8[dst_stride_idx[y_off] + x + x_off] =
              clamp(offset_val + rec_yv_8[dst_stride_idx[y_off] + x + x_off], 0,
                    (1 << cm->seq_params.bit_depth) - 1);
        }
      }
    }
    temp_rec_y_buf += (ccso_stride_ext << (CCSO_BLK_SIZE + y_uv_vert_scale));
    rec_yv_8 += (dst_stride << CCSO_BLK_SIZE);
  }
}

/* Apply CCSO for one filtering unit using c code (high bit-depth) */
void ccso_filter_block_hbd_c(
    const uint16_t *temp_rec_y_buf, uint16_t *rec_uv_16, const int x,
    const int y, const int pic_width_c, const int pic_height_c,
    int *rec_luma_idx, const int8_t *offset_buf, const int *ccso_stride_idx,
    const int *dst_stride_idx, const int y_uv_hori_scale,
    const int y_uv_vert_scale, const int pad_stride, const int quant_step_size,
    const int inv_quant_step, const int *rec_idx, const int maxval) {
  int y_offset;
  int x_offset;
  if (y + (1 << CCSO_BLK_SIZE) >= pic_height_c)
    y_offset = pic_height_c - y;
  else
    y_offset = (1 << CCSO_BLK_SIZE);

  if (x + (1 << CCSO_BLK_SIZE) >= pic_width_c)
    x_offset = pic_width_c - x;
  else
    x_offset = (1 << CCSO_BLK_SIZE);

  for (int y_off = 0; y_off < y_offset; y_off++) {
    for (int x_off = 0; x_off < x_offset; x_off++) {
      cal_filter_support(
          rec_luma_idx,
          &temp_rec_y_buf[((ccso_stride_idx[y_off] << y_uv_vert_scale) +
                           ((x + x_off) << y_uv_hori_scale)) +
                          pad_stride],
          quant_step_size, inv_quant_step, rec_idx);
      const int offset_val =
          offset_buf[(rec_luma_idx[0] << 2) + rec_luma_idx[1]];
      rec_uv_16[dst_stride_idx[y_off] + x + x_off] = clamp(
          offset_val + rec_uv_16[dst_stride_idx[y_off] + x + x_off], 0, maxval);
    }
  }
}

/* Apply CCSO for one color component (high bit-depth) */
void apply_ccso_filter_hbd(AV1_COMMON *cm, MACROBLOCKD *xd, const int plane,
                           const uint16_t *temp_rec_y_buf, uint16_t *rec_uv_16,
                           const int dst_stride, const int8_t *filter_offset,
                           const uint8_t quant_step_size,
                           const uint8_t ext_filter_support) {
  const CommonModeInfoParams *const mi_params = &cm->mi_params;
  const int ccso_stride_ext = xd->plane[0].dst.width + (CCSO_PADDING_SIZE << 1);
  const int pic_height_c = xd->plane[1].dst.height;
  const int pic_width_c = xd->plane[1].dst.width;
  int rec_luma_idx[2];
  int inv_quant_step = quant_step_size * -1;
  int rec_idx[2];
  const int maxval = (1 << cm->seq_params.bit_depth) - 1;

  derive_ccso_sample_pos(rec_idx, ccso_stride_ext, ext_filter_support);

  const int8_t *offset_buf;
  if (plane > 0) {
    offset_buf = cm->ccso_info.filter_offset[plane - 1];
  } else {
    offset_buf = filter_offset;
  }
  int ccso_stride_ext_idx[1 << CCSO_BLK_SIZE];
  int dst_stride_idx[1 << CCSO_BLK_SIZE];
  for (int i = 0; i < (1 << CCSO_BLK_SIZE); i++) {
    ccso_stride_ext_idx[i] = ccso_stride_ext * i;
    dst_stride_idx[i] = dst_stride * i;
  }
  const int pad_stride =
      CCSO_PADDING_SIZE * ccso_stride_ext + CCSO_PADDING_SIZE;
  const int y_uv_hori_scale = xd->plane[1].subsampling_x;
  const int y_uv_vert_scale = xd->plane[1].subsampling_y;
  for (int y = 0; y < pic_height_c; y += (1 << CCSO_BLK_SIZE)) {
    for (int x = 0; x < pic_width_c; x += (1 << CCSO_BLK_SIZE)) {
      if (plane > 0) {
        const int ccso_blk_idx =
            (1 << CCSO_BLK_SIZE >>
             (MI_SIZE_LOG2 - xd->plane[plane].subsampling_y)) *
                (y >> CCSO_BLK_SIZE) * mi_params->mi_stride +
            (1 << CCSO_BLK_SIZE >>
             (MI_SIZE_LOG2 - xd->plane[plane].subsampling_x)) *
                (x >> CCSO_BLK_SIZE);
        const bool use_ccso =
            (plane == 1) ? mi_params->mi_grid_base[ccso_blk_idx]->ccso_blk_u
                         : mi_params->mi_grid_base[ccso_blk_idx]->ccso_blk_v;
        if (!use_ccso) continue;
      }

      ccso_filter_block_hbd(temp_rec_y_buf, rec_uv_16, x, y, pic_width_c,
                            pic_height_c, rec_luma_idx, offset_buf,
                            ccso_stride_ext_idx, dst_stride_idx,
                            y_uv_hori_scale, y_uv_vert_scale, pad_stride,
                            quant_step_size, inv_quant_step, rec_idx, maxval);
    }
    temp_rec_y_buf += (ccso_stride_ext << (CCSO_BLK_SIZE + y_uv_vert_scale));
    rec_uv_16 += (dst_stride << CCSO_BLK_SIZE);
  }
}

/* Apply CCSO for one frame */
void ccso_frame(YV12_BUFFER_CONFIG *frame, AV1_COMMON *cm, MACROBLOCKD *xd,
                uint16_t *ext_rec_y) {
  const int num_planes = av1_num_planes(cm);
  av1_setup_dst_planes(xd->plane, cm->seq_params.sb_size, frame, 0, 0, 0,
                       num_planes);

  const uint8_t quant_sz[4] = { 16, 8, 32, 64 };
  for (int plane = 1; plane < 3; plane++) {
    const int dst_stride = xd->plane[plane].dst.stride;
    const uint8_t quant_step_size =
        quant_sz[cm->ccso_info.quant_idx[plane - 1]];
    if (cm->ccso_info.ccso_enable[plane - 1]) {
      if (cm->seq_params.use_highbitdepth) {
        apply_ccso_filter_hbd(cm, xd, plane, ext_rec_y,
                              &CONVERT_TO_SHORTPTR(xd->plane[plane].dst.buf)[0],
                              dst_stride, NULL, quant_step_size,
                              cm->ccso_info.ext_filter_support[plane - 1]);
      } else {
        apply_ccso_filter(
            cm, xd, plane, ext_rec_y, &xd->plane[plane].dst.buf[0], dst_stride,
            NULL, quant_step_size, cm->ccso_info.ext_filter_support[plane - 1]);
      }
    }
  }
}
