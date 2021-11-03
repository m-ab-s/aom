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
#include <string.h>
#include <float.h>

#include "config/aom_dsp_rtcd.h"
#include "config/aom_scale_rtcd.h"

#include "aom/aom_integer.h"
#include "aom_ports/system_state.h"
#include "av1/common/av1_common_int.h"
#include "av1/common/reconinter.h"
#include "av1/encoder/encoder.h"
#include "av1/encoder/pickccso.h"

int8_t best_filter_offset[2][16] = { { 0 } };
int8_t final_filter_offset[2][16] = { { 0 } };
int chroma_error[16] = { 0 };
int chroma_count[16] = { 0 };
bool best_filter_enabled[2];
bool final_filter_enabled[2];
uint8_t final_ext_filter_support[2];
uint8_t final_quant_idx[2];
int ccso_stride;
int ccso_stride_ext;
bool *filter_control;
bool *best_filter_control;
bool *final_filter_control;
uint64_t unfiltered_dist_frame = 0;
uint64_t filtered_dist_frame = 0;
uint64_t *unfiltered_dist_block;
uint64_t *training_dist_block;

const int ccso_offset[8] = { -7, -5, -3, -1, 0, 1, 3, 5 };
const uint8_t quant_sz[4] = { 16, 8, 32, 64 };

/* Compute SSE */
void compute_distortion(const uint16_t *org, const int org_stride,
                        const uint8_t *rec8, const uint16_t *rec16,
                        const int rec_stride, const int height, const int width,
                        uint64_t *distortion_buf,
                        const int distortion_buf_stride,
                        uint64_t *total_distortion) {
  int org_stride_idx[1 << CCSO_BLK_SIZE];
  int rec_stride_idx[1 << CCSO_BLK_SIZE];
  for (int i = 0; i < (1 << CCSO_BLK_SIZE); i++) {
    org_stride_idx[i] = org_stride * i;
    rec_stride_idx[i] = rec_stride * i;
  }
  for (int y = 0; y < height; y += (1 << CCSO_BLK_SIZE)) {
    for (int x = 0; x < width; x += (1 << CCSO_BLK_SIZE)) {
      int err;
      uint64_t ssd = 0;
      int y_offset;
      int x_offset;
      if (y + (1 << CCSO_BLK_SIZE) >= height)
        y_offset = height - y;
      else
        y_offset = (1 << CCSO_BLK_SIZE);

      if (x + (1 << CCSO_BLK_SIZE) >= width)
        x_offset = width - x;
      else
        x_offset = (1 << CCSO_BLK_SIZE);

      for (int y_off = 0; y_off < y_offset; y_off++) {
        for (int x_off = 0; x_off < x_offset; x_off++) {
          if (rec8) {
            err = org[org_stride_idx[y_off] + x + x_off] -
                  rec8[rec_stride_idx[y_off] + x + x_off];
          } else {
            err = org[org_stride_idx[y_off] + x + x_off] -
                  rec16[rec_stride_idx[y_off] + x + x_off];
          }
          ssd += err * err;
        }
      }
      distortion_buf[(y >> CCSO_BLK_SIZE) * distortion_buf_stride +
                     (x >> CCSO_BLK_SIZE)] = ssd;
      *total_distortion += ssd;
    }
    org += (org_stride << CCSO_BLK_SIZE);
    if (rec8) {
      rec8 += (rec_stride << CCSO_BLK_SIZE);
    } else {
      rec16 += (rec_stride << CCSO_BLK_SIZE);
    }
  }
}

/* Derive block level on/off for CCSO */
void derive_blk_md(AV1_COMMON *cm, MACROBLOCKD *xd,
                   const uint64_t *unfiltered_dist,
                   const uint64_t *training_dist, bool *m_filter_control,
                   uint64_t *cur_total_dist, int *cur_total_rate,
                   bool *filter_enable, const int rdmult) {
  const CommonModeInfoParams *const mi_params = &cm->mi_params;
  const int ccso_nvfb = ((mi_params->mi_rows >> xd->plane[1].subsampling_y) +
                         (1 << CCSO_BLK_SIZE >> 2) - 1) /
                        (1 << CCSO_BLK_SIZE >> 2);
  const int ccso_nhfb = ((mi_params->mi_cols >> xd->plane[1].subsampling_x) +
                         (1 << CCSO_BLK_SIZE >> 2) - 1) /
                        (1 << CCSO_BLK_SIZE >> 2);
  bool cur_filter_enabled = false;
  int sb_idx = 0;
  const int rate = av1_cost_literal(1);
  for (int y_sb = 0; y_sb < ccso_nvfb; y_sb++) {
    for (int x_sb = 0; x_sb < ccso_nhfb; x_sb++) {
      uint64_t ssd;
      uint64_t best_ssd = UINT64_MAX;
      int best_rate = INT_MAX;
      uint64_t best_cost = UINT64_MAX;
      uint8_t cur_best_filter_control = 0;
      for (int cur_filter_control = 0; cur_filter_control < 2;
           cur_filter_control++) {
        if (!(*filter_enable)) {
          continue;
        }
        if (cur_filter_control == 0) {
          ssd = unfiltered_dist[sb_idx];
        } else {
          ssd = training_dist[sb_idx];
        }
        const uint64_t rd_cost = RDCOST(rdmult, rate, ssd * 16);
        if (rd_cost < best_cost) {
          best_cost = rd_cost;
          best_rate = rate;
          best_ssd = ssd;
          cur_best_filter_control = cur_filter_control;
          m_filter_control[sb_idx] = cur_filter_control;
        }
      }
      if (cur_best_filter_control != 0) {
        cur_filter_enabled = true;
      }
      *cur_total_rate += best_rate;
      *cur_total_dist += best_ssd;
      sb_idx++;
    }
  }
  *filter_enable = cur_filter_enabled;
}

/* Compute the aggregated residual between original and reconstructed sample for
 * each entry of the LUT */
void compute_total_error(MACROBLOCKD *xd, const uint16_t *ext_rec_luma,
                         const uint16_t *org_chroma, const uint16_t *rec_uv_16,
                         const uint8_t quant_step_size,
                         const uint8_t ext_filter_support) {
  const int pic_height_c = xd->plane[1].dst.height;
  const int pic_width_c = xd->plane[1].dst.width;
  int sb_idx = 0;
  int rec_luma_idx[2];
  const int inv_quant_step = quant_step_size * -1;
  int rec_idx[2];
  if (ext_filter_support == 0) {
    rec_idx[0] = -1 * ccso_stride_ext;
    rec_idx[1] = 1 * ccso_stride_ext;
  } else if (ext_filter_support == 1) {
    rec_idx[0] = -1 * ccso_stride_ext - 1;
    rec_idx[1] = 1 * ccso_stride_ext + 1;
  } else if (ext_filter_support == 2) {
    rec_idx[0] = 0 * ccso_stride_ext - 1;
    rec_idx[1] = 0 * ccso_stride_ext + 1;
  } else if (ext_filter_support == 3) {
    rec_idx[0] = 1 * ccso_stride_ext - 1;
    rec_idx[1] = -1 * ccso_stride_ext + 1;
  } else if (ext_filter_support == 4) {
    rec_idx[0] = 0 * ccso_stride_ext - 3;
    rec_idx[1] = 0 * ccso_stride_ext + 3;
  } else {  // if(ext_filter_support == 5) {
    rec_idx[0] = 0 * ccso_stride_ext - 5;
    rec_idx[1] = 0 * ccso_stride_ext + 5;
  }
  int ccso_stride_idx[1 << CCSO_BLK_SIZE];
  int ccso_stride_ext_idx[1 << CCSO_BLK_SIZE];
  for (int i = 0; i < (1 << CCSO_BLK_SIZE); i++) {
    ccso_stride_idx[i] = ccso_stride * i;
    ccso_stride_ext_idx[i] = ccso_stride_ext * i;
  }
  const int pad_stride =
      CCSO_PADDING_SIZE * ccso_stride_ext + CCSO_PADDING_SIZE;
  const int y_uv_hori_scale = xd->plane[1].subsampling_x;
  const int y_uv_vert_scale = xd->plane[1].subsampling_y;
  for (int y = 0; y < pic_height_c; y += (1 << CCSO_BLK_SIZE)) {
    for (int x = 0; x < pic_width_c; x += (1 << CCSO_BLK_SIZE)) {
      const bool skip_filtering = (filter_control[sb_idx]) ? false : true;
      sb_idx++;
      if (skip_filtering) continue;
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
              &ext_rec_luma[((ccso_stride_ext_idx[y_off] << y_uv_vert_scale) +
                             ((x + x_off) << y_uv_hori_scale)) +
                            pad_stride],
              quant_step_size, inv_quant_step, rec_idx);
          chroma_error[(rec_luma_idx[0] << 2) + rec_luma_idx[1]] +=
              org_chroma[ccso_stride_idx[y_off] + x + x_off] -
              rec_uv_16[ccso_stride_idx[y_off] + x + x_off];
          chroma_count[(rec_luma_idx[0] << 2) + rec_luma_idx[1]]++;
        }
      }
    }
    ext_rec_luma += (ccso_stride_ext << (CCSO_BLK_SIZE + y_uv_vert_scale));
    rec_uv_16 += (ccso_stride << CCSO_BLK_SIZE);
    org_chroma += (ccso_stride << CCSO_BLK_SIZE);
  }
}

/* Derive the offset value in the look-up table */
void derive_lut_offset(int8_t *temp_filter_offset) {
  float temp_offset = 0;
  for (int d0 = 0; d0 < CCSO_INPUT_INTERVAL; d0++) {
    for (int d1 = 0; d1 < CCSO_INPUT_INTERVAL; d1++) {
      const int lut_idx_ext = (d0 << 2) + d1;
      if (chroma_count[lut_idx_ext]) {
        temp_offset =
            (float)chroma_error[lut_idx_ext] / chroma_count[lut_idx_ext];

        if ((temp_offset < -7) || (temp_offset >= 5)) {
          temp_filter_offset[lut_idx_ext] = clamp((int)temp_offset, -7, 5);
        } else {
          for (int offset_idx = 0; offset_idx < 7; offset_idx++) {
            if ((temp_offset >= ccso_offset[offset_idx]) &&
                (temp_offset <= ccso_offset[offset_idx + 1])) {
              if (fabs(temp_offset - ccso_offset[offset_idx]) >
                  fabs(temp_offset - ccso_offset[offset_idx + 1])) {
                temp_filter_offset[lut_idx_ext] = ccso_offset[offset_idx + 1];
              } else {
                temp_filter_offset[lut_idx_ext] = ccso_offset[offset_idx];
              }
              break;
            }
          }
        }
      }
    }
  }
}

/* Derive the look-up table for a color component */
void derive_ccso_filter(AV1_COMMON *cm, const int plane, MACROBLOCKD *xd,
                        const uint16_t *org_uv, const uint16_t *ext_rec_y,
                        const uint16_t *rec_uv, int rdmult) {
  const CommonModeInfoParams *const mi_params = &cm->mi_params;
  const int ccso_nvfb =
      ((mi_params->mi_rows >> xd->plane[plane].subsampling_y) +
       (1 << CCSO_BLK_SIZE >> 2) - 1) /
      (1 << CCSO_BLK_SIZE >> 2);
  const int ccso_nhfb =
      ((mi_params->mi_cols >> xd->plane[plane].subsampling_x) +
       (1 << CCSO_BLK_SIZE >> 2) - 1) /
      (1 << CCSO_BLK_SIZE >> 2);
  const int sb_count = ccso_nvfb * ccso_nhfb;
  const int pic_height_c = xd->plane[plane].dst.height;
  const int pic_width_c = xd->plane[plane].dst.width;
  uint16_t *temp_rec_uv_buf;
  unfiltered_dist_frame = 0;
  unfiltered_dist_block = aom_malloc(sizeof(*unfiltered_dist_block) * sb_count);
  memset(unfiltered_dist_block, 0, sizeof(*unfiltered_dist_block) * sb_count);
  training_dist_block = aom_malloc(sizeof(*training_dist_block) * sb_count);
  memset(training_dist_block, 0, sizeof(*training_dist_block) * sb_count);
  filter_control = aom_malloc(sizeof(*filter_control) * sb_count);
  memset(filter_control, 0, sizeof(*filter_control) * sb_count);
  best_filter_control = aom_malloc(sizeof(*best_filter_control) * sb_count);
  memset(best_filter_control, 0, sizeof(*best_filter_control) * sb_count);
  final_filter_control = aom_malloc(sizeof(*final_filter_control) * sb_count);
  memset(final_filter_control, 0, sizeof(*final_filter_control) * sb_count);
  temp_rec_uv_buf = aom_malloc(sizeof(*temp_rec_uv_buf) *
                               xd->plane[0].dst.height * ccso_stride);
  compute_distortion(org_uv, ccso_stride, NULL, rec_uv, ccso_stride,
                     pic_height_c, pic_width_c, unfiltered_dist_block,
                     ccso_nhfb, &unfiltered_dist_frame);
  const uint64_t best_unfiltered_cost =
      RDCOST(rdmult, av1_cost_literal(1), unfiltered_dist_frame * 16);
  uint64_t best_filtered_cost;
  uint64_t final_filtered_cost = UINT64_MAX;
  int8_t filter_offset[16];
  const int total_filter_support = 6;
  const int total_quant_idx = 4;
  uint8_t frame_bits = 1;
  frame_bits += 2;  // quant step size
  frame_bits += 3;  // filter support index
  for (int ext_filter_support = 0; ext_filter_support < total_filter_support;
       ext_filter_support++) {
    for (int quant_idx = 0; quant_idx < total_quant_idx; quant_idx++) {
      best_filtered_cost = UINT64_MAX;
      bool ccso_enable = true;
      bool keep_training = true;
      bool improvement = false;
      uint64_t prev_total_cost = UINT64_MAX;
      int control_idx = 0;
      for (int y = 0; y < ccso_nvfb; y++) {
        for (int x = 0; x < ccso_nhfb; x++) {
          filter_control[control_idx] = 1;
          control_idx++;
        }
      }
      int training_iter_count = 0;
      while (keep_training) {
        improvement = false;
        if (ccso_enable) {
          memset(chroma_error, 0, sizeof(chroma_error));
          memset(chroma_count, 0, sizeof(chroma_count));
          memset(filter_offset, 0, sizeof(filter_offset));
          memcpy(
              temp_rec_uv_buf, rec_uv,
              sizeof(*temp_rec_uv_buf) * xd->plane[0].dst.height * ccso_stride);
          compute_total_error(xd, ext_rec_y, org_uv, temp_rec_uv_buf,
                              quant_sz[quant_idx], ext_filter_support);
          derive_lut_offset(filter_offset);
        }
        memcpy(
            temp_rec_uv_buf, rec_uv,
            sizeof(*temp_rec_uv_buf) * xd->plane[0].dst.height * ccso_stride);
        apply_ccso_filter_hbd(cm, xd, -1, ext_rec_y, temp_rec_uv_buf,
                              ccso_stride, filter_offset, quant_sz[quant_idx],
                              ext_filter_support);
        filtered_dist_frame = 0;
        compute_distortion(org_uv, ccso_stride, NULL, temp_rec_uv_buf,
                           ccso_stride, pic_height_c, pic_width_c,
                           training_dist_block, ccso_nhfb,
                           &filtered_dist_frame);
        uint64_t cur_total_dist = 0;
        int cur_total_rate = 0;
        derive_blk_md(cm, xd, unfiltered_dist_block, training_dist_block,
                      filter_control, &cur_total_dist, &cur_total_rate,
                      &ccso_enable, rdmult);
        if (ccso_enable) {
          const int lut_bits = 9;
          cur_total_rate +=
              av1_cost_literal(lut_bits * 3) + av1_cost_literal(frame_bits);
          const uint64_t cur_total_cost =
              RDCOST(rdmult, cur_total_rate, cur_total_dist * 16);
          if (cur_total_cost < prev_total_cost) {
            prev_total_cost = cur_total_cost;
            improvement = true;
          }
          if (cur_total_cost < best_filtered_cost) {
            best_filtered_cost = cur_total_cost;
            best_filter_enabled[plane - 1] = ccso_enable;
            memcpy(best_filter_offset[plane - 1], filter_offset,
                   sizeof(filter_offset));
            memcpy(best_filter_control, filter_control,
                   sizeof(*filter_control) * sb_count);
          }
        }
        training_iter_count++;
        if (!improvement || training_iter_count > CCSO_MAX_ITERATIONS) {
          keep_training = false;
        }
      }

      if (best_filtered_cost < final_filtered_cost) {
        final_filtered_cost = best_filtered_cost;
        final_filter_enabled[plane - 1] = best_filter_enabled[plane - 1];
        final_quant_idx[plane - 1] = quant_idx;
        final_ext_filter_support[plane - 1] = ext_filter_support;
        memcpy(final_filter_offset[plane - 1], best_filter_offset[plane - 1],
               sizeof(best_filter_offset[plane - 1]));
        memcpy(final_filter_control, best_filter_control,
               sizeof(*best_filter_control) * sb_count);
      }
    }
  }

  if (best_unfiltered_cost < final_filtered_cost) {
    memset(final_filter_control, 0, sizeof(*final_filter_control) * sb_count);
  }
  bool at_least_one_sb_use_ccso = false;
  for (int control_idx2 = 0;
       final_filter_enabled[plane - 1] && control_idx2 < sb_count;
       control_idx2++) {
    if (final_filter_control[control_idx2]) {
      at_least_one_sb_use_ccso = true;
      break;
    }
  }
  cm->ccso_info.ccso_enable[plane - 1] = at_least_one_sb_use_ccso;
  if (at_least_one_sb_use_ccso) {
    for (int y_sb = 0; y_sb < ccso_nvfb; y_sb++) {
      for (int x_sb = 0; x_sb < ccso_nhfb; x_sb++) {
        if (plane == AOM_PLANE_U) {
          mi_params
              ->mi_grid_base[(1 << CCSO_BLK_SIZE >>
                              (MI_SIZE_LOG2 - xd->plane[1].subsampling_y)) *
                                 y_sb * mi_params->mi_stride +
                             (1 << CCSO_BLK_SIZE >>
                              (MI_SIZE_LOG2 - xd->plane[1].subsampling_x)) *
                                 x_sb]
              ->ccso_blk_u = final_filter_control[y_sb * ccso_nhfb + x_sb];
        } else {
          mi_params
              ->mi_grid_base[(1 << CCSO_BLK_SIZE >>
                              (MI_SIZE_LOG2 - xd->plane[2].subsampling_y)) *
                                 y_sb * mi_params->mi_stride +
                             (1 << CCSO_BLK_SIZE >>
                              (MI_SIZE_LOG2 - xd->plane[2].subsampling_x)) *
                                 x_sb]
              ->ccso_blk_v = final_filter_control[y_sb * ccso_nhfb + x_sb];
        }
      }
    }
    memcpy(cm->ccso_info.filter_offset[plane - 1],
           final_filter_offset[plane - 1],
           sizeof(final_filter_offset[plane - 1]));
    cm->ccso_info.quant_idx[plane - 1] = final_quant_idx[plane - 1];
    cm->ccso_info.ext_filter_support[plane - 1] =
        final_ext_filter_support[plane - 1];
  }
  aom_free(unfiltered_dist_block);
  aom_free(training_dist_block);
  aom_free(filter_control);
  aom_free(final_filter_control);
  aom_free(temp_rec_uv_buf);
  aom_free(best_filter_control);
}

/* Derive the look-up table for a frame */
void ccso_search(AV1_COMMON *cm, MACROBLOCKD *xd, int rdmult,
                 const uint16_t *ext_rec_y, uint16_t *rec_uv[2],
                 uint16_t *org_uv[2]) {
  double rdmult_weight =
      clamp_dbl(0.012 * pow(2, 0.0456 * cm->quant_params.base_qindex), 1, 37);
  int64_t rdmult_temp = (int64_t)rdmult * (int64_t)rdmult_weight;
  if (rdmult_temp < INT_MAX) rdmult = (int)rdmult_temp;
  const int num_planes = av1_num_planes(cm);
  av1_setup_dst_planes(xd->plane, &cm->cur_frame->buf, 0, 0, 0, num_planes,
                       NULL);
  ccso_stride = xd->plane[0].dst.width;
  ccso_stride_ext = xd->plane[0].dst.width + (CCSO_PADDING_SIZE << 1);
  derive_ccso_filter(cm, AOM_PLANE_U, xd, org_uv[AOM_PLANE_U - 1], ext_rec_y,
                     rec_uv[AOM_PLANE_U - 1], rdmult);
  derive_ccso_filter(cm, AOM_PLANE_V, xd, org_uv[AOM_PLANE_V - 1], ext_rec_y,
                     rec_uv[AOM_PLANE_V - 1], rdmult);
}
