/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"
#include "config/av1_rtcd.h"

#include "aom_dsp/pyramid.h"
#include "aom_mem/aom_mem.h"

#include "av1/common/av1_common_int.h"
#include "av1/common/enums.h"
#include "av1/common/entropymv.h"
#include "av1/common/reconinter.h"

#include "av1/encoder/encode_strategy.h"
#include "av1/encoder/encodeframe_utils.h"
#include "av1/encoder/encoder.h"
#include "av1/encoder/encoder_utils.h"
#include "av1/encoder/pyramid_me.h"
#include "av1/encoder/ratectrl.h"
#include "av1/encoder/rd.h"
#include "av1/encoder/rdopt.h"
#include "av1/encoder/temporal_filter.h"
#include "av1/encoder/tpl_model.h"

// Calculates the geometry of the pyramid layers for the current frame size,
// along with the total buffer size and the offset of each layer within it.
static void calc_pme_pyramid_layers(const AV1_COMMON *const cm,
                                    PyramidLayer *const layers,
                                    size_t *const buffer_size,
                                    size_t *const layer_offset) {
  // Compute the size and offset of each layer within one buffer.
  *buffer_size = 0;
  const size_t first_px_offset =
      (PYRAMID_ME_PADDING + PYRAMID_ALIGNMENT - 1) & ~(PYRAMID_ALIGNMENT - 1);
  const size_t extra_bytes = first_px_offset - PYRAMID_ME_PADDING;
  *buffer_size += extra_bytes;
  const int min_blk_size = 8;
  int horz_padding = 0;
  int vert_padding = 0;
  if (cm->width % min_blk_size)
    horz_padding = min_blk_size - (cm->width % min_blk_size);
  if (cm->height % min_blk_size)
    vert_padding = min_blk_size - (cm->height % min_blk_size);

  // TODO(apurve.pandey): Level 0 duplicates the luma plane of the source frame,
  // so its allocation can be avoided by pointing level 0 to the source frame.
  for (PYRAMID_LEVEL level = 0; level < MAX_PYR_LEVELS; ++level) {
    const int width = level ? cm->width + horz_padding : cm->width;
    const int height = level ? cm->height + vert_padding : cm->height;
    const int level_width = width >> level;
    const int level_height = height >> level;

    // Padding on each side, plus PYRAMID_ME_RIGHT_SLACK on the right and
    // PYRAMID_ME_BOTTOM_SLACK at the bottom.
    const int padded_width =
        level_width + 2 * PYRAMID_ME_PADDING + PYRAMID_ME_RIGHT_SLACK;
    const int padded_height =
        level_height + 2 * PYRAMID_ME_PADDING + PYRAMID_ME_BOTTOM_SLACK;

    // Align the stride to PYRAMID_ALIGNMENT so that every row is aligned.
    const int level_stride =
        (padded_width + PYRAMID_ALIGNMENT - 1) & ~(PYRAMID_ALIGNMENT - 1);

    const size_t level_alloc_start = *buffer_size;
    layer_offset[level] = level_alloc_start +
                          PYRAMID_ME_PADDING * level_stride +
                          PYRAMID_ME_PADDING;

    *buffer_size += level_stride * padded_height;

    layers[level].buffer = NULL;
    layers[level].width = level_width;
    layers[level].height = level_height;
    layers[level].stride = level_stride;
  }
}

// Allocates a pyramid using the geometry from calc_pme_pyramid_layers().
// Returns AOM_CODEC_OK on success.
static int alloc_pme_pyramid_layers(ImagePyramid *const pyr_layers,
                                    const PyramidLayer *const layers,
                                    const size_t *const layer_offset,
                                    size_t buffer_size) {
  uint8_t *buffer_alloc =
      aom_memalign(PYRAMID_ALIGNMENT, buffer_size * sizeof(*buffer_alloc));
  if (!buffer_alloc) return AOM_CODEC_MEM_ERROR;

  pyr_layers->buffer_alloc = buffer_alloc;
  pyr_layers->layers = aom_malloc(sizeof(*pyr_layers->layers) * MAX_PYR_LEVELS);
  if (!pyr_layers->layers) {
    aom_free(pyr_layers->buffer_alloc);
    pyr_layers->buffer_alloc = NULL;
    return AOM_CODEC_MEM_ERROR;
  }

  pyr_layers->max_levels = MAX_PYR_LEVELS;

  for (PYRAMID_LEVEL level = 0; level < MAX_PYR_LEVELS; ++level) {
    pyr_layers->layers[level] = layers[level];
    pyr_layers->layers[level].buffer = buffer_alloc + layer_offset[level];
  }

  return AOM_CODEC_OK;
}

bool av1_is_pme_allowed(const AV1_COMP *cpi) {
  return cpi->sf.pme_sf.enable_pyramid_me &&
         cpi->oxcf.gf_cfg.lag_in_frames > 1 &&
         !av1_is_resize_needed(&cpi->oxcf);
}

void av1_init_pme_stats(PmeParams *pme_data) {
  pme_data->pme_stats_ready = false;
}

void av1_alloc_pme_buffers(AV1_COMP *cpi) {
  AV1_PRIMARY *const ppi = cpi->ppi;
  PmeParams *const pme_data = &ppi->pme_data;
  if (!av1_is_pme_allowed(cpi)) return;

  const AV1_COMMON *const cm = &cpi->common;
  if (pme_data->stats_pool[0] != NULL) {
    if (pme_data->alloc_width == cm->width &&
        pme_data->alloc_height == cm->height)
      return;
    av1_free_pme_buffers(pme_data);
  }
  const int lag_in_frames = cpi->oxcf.gf_cfg.lag_in_frames;
  av1_init_pme_stats(pme_data);
  pme_data->prev_gop_arf_disp_order = -1;
  pme_data->frame_stats_ptr = &pme_data->pme_frame[REF_FRAMES + 1];
  pme_data->alloc_width = cm->width;
  pme_data->alloc_height = cm->height;
  PyramidLayer layers[MAX_PYR_LEVELS];
  size_t layer_offset[MAX_PYR_LEVELS];
  size_t buffer_size;
  calc_pme_pyramid_layers(cm, layers, &buffer_size, layer_offset);
  const int num_sb_row =
      (cm->mi_params.mi_rows + MI_SIZE_64X64 - 1) / MI_SIZE_64X64;
  const int num_sb_col =
      (cm->mi_params.mi_cols + MI_SIZE_64X64 - 1) / MI_SIZE_64X64;

  for (int frame = 0; frame < REF_FRAMES; ++frame) {
    if (alloc_pme_pyramid_layers(&pme_data->src_pyr_pool[frame], layers,
                                 layer_offset, buffer_size))
      aom_internal_error(&ppi->error, AOM_CODEC_MEM_ERROR,
                         "Failed to allocate src ref pyramid buffer");
  }

  for (int frame = 0; frame < lag_in_frames; ++frame) {
    AOM_CHECK_MEM_ERROR(
        &ppi->error, pme_data->stats_pool[frame],
        aom_calloc(num_sb_row * num_sb_col * INTER_REFS_PER_FRAME,
                   sizeof(pme_data->stats_pool[0][0][0])));

    if (alloc_pme_pyramid_layers(&pme_data->src_pyr_pool[REF_FRAMES + frame],
                                 layers, layer_offset, buffer_size))
      aom_internal_error(&ppi->error, AOM_CODEC_MEM_ERROR,
                         "Failed to allocate src ref pyramid buffer");
  }

  if (alloc_pme_pyramid_layers(&pme_data->prev_gop_arf_src_pyr, layers,
                               layer_offset, buffer_size))
    aom_internal_error(
        &ppi->error, AOM_CODEC_MEM_ERROR,
        "Failed to allocate previous gop src ref pyramid buffer");
}

// Frees the buffers of a pyramid allocated by alloc_pme_pyramid_layers().
static void free_pme_pyramid_layers(ImagePyramid *const pyr) {
  aom_free(pyr->buffer_alloc);
  aom_free(pyr->layers);
  pyr->buffer_alloc = NULL;
  pyr->layers = NULL;
  pyr->max_levels = 0;
  pyr->filled_levels = 0;
}

void av1_free_pme_buffers(PmeParams *pme_data) {
  free_pme_pyramid_layers(&pme_data->prev_gop_arf_src_pyr);
  for (int frame = 0; frame < REF_FRAMES + MAX_LAG_BUFFERS; ++frame)
    free_pme_pyramid_layers(&pme_data->src_pyr_pool[frame]);
  for (int frame = 0; frame < MAX_LAG_BUFFERS; ++frame) {
    aom_free(pme_data->stats_pool[frame]);
    pme_data->stats_pool[frame] = NULL;
  }
  // The frames point to the freed buffers until the next sub-GOP is set up.
  for (int i = 0; i < MAX_LENGTH_PME_FRAME_STATS; ++i) {
    pme_data->pme_frame[i].src_pyr = NULL;
    pme_data->pme_frame[i].stats = NULL;
  }
  av1_init_pme_stats(pme_data);
  pme_data->alloc_width = 0;
  pme_data->alloc_height = 0;
}

// Returns true if the pyramid is allocated for the size of 'frame'.
static inline bool is_pme_pyr_allocated_for_frame(
    const ImagePyramid *pyr, const YV12_BUFFER_CONFIG *frame) {
  return pyr->layers != NULL &&
         pyr->layers[PYR_LEVEL0].width == frame->y_crop_width &&
         pyr->layers[PYR_LEVEL0].height == frame->y_crop_height;
}

void av1_compute_prev_gop_arf_pyramid(AV1_COMP *cpi) {
  if (!av1_is_pme_allowed(cpi)) return;
  PmeParams *const pme_data = &cpi->ppi->pme_data;
  ImagePyramid *const prev_gop_arf_src_pyr = &pme_data->prev_gop_arf_src_pyr;
  pme_data->prev_gop_arf_disp_order = -1;
  prev_gop_arf_src_pyr->filled_levels = 0;
  // The pyramid is computed only if it is allocated for this frame size.
  if (!is_pme_pyr_allocated_for_frame(prev_gop_arf_src_pyr, cpi->source))
    return;
  if (aom_compute_padded_pyramid(cpi->source, cpi->common.seq_params->bit_depth,
                                 MAX_PYR_LEVELS, prev_gop_arf_src_pyr,
                                 PYRAMID_ME_PADDING, PYRAMID_ME_RIGHT_SLACK,
                                 PYRAMID_ME_BOTTOM_SLACK) < 0)
    return;
  pme_data->prev_gop_arf_disp_order =
      (int)cpi->common.current_frame.display_order_hint;
}

// Returns the index of the reference frame at the same temporal distance from
// the current frame as ref_disp_order_idx, in the opposite direction, or -1 if
// there is none.
static inline int get_mirror_rf_idx(const int *const disp_order_idx,
                                    int cur_disp_order_idx,
                                    int ref_disp_order_idx) {
  const int rel_dist =
      av1_encoder_get_relative_dist(cur_disp_order_idx, ref_disp_order_idx);

  for (int rf_idx = 0; rf_idx < INTER_REFS_PER_FRAME; ++rf_idx) {
    const int this_rel_dist = av1_encoder_get_relative_dist(
        cur_disp_order_idx, disp_order_idx[rf_idx]);
    if (this_rel_dist == -rel_dist) return rf_idx;
  }
  return -1;
}

// Clamps the search area of a block at pos, starting at pos + *origin and
// *size positions long, along one axis: the area starts no more than pad
// positions before the frame and no later than its last position, and ends
// inside the frame of frame_size positions.
static inline void clamp_search_area_1d(int pos, int pad, int frame_size,
                                        int *origin, int *size) {
  if (pos + *origin < -pad) *origin = -pad - pos;
  if (pos + *origin > frame_size - 1) *origin = frame_size - 1 - pos;
  if (pos + *origin + *size > frame_size)
    *size = AOMMAX(1, frame_size - pos - *origin);
}

// Clamps the search area of the block at (pos_x, pos_y), see
// clamp_search_area_1d(), and rounds its width down to a multiple of 8, the
// number of positions the SAD kernels evaluate at once.
static inline void clamp_search_area(int pos_x, int pos_y, int pad_x, int pad_y,
                                     int frame_width, int frame_height,
                                     int *origin_x, int *origin_y, int *width,
                                     int *height) {
  clamp_search_area_1d(pos_x, pad_x, frame_width, origin_x, width);
  if (*width >= 8) *width &= ~0x07;
  clamp_search_area_1d(pos_y, pad_y, frame_height, origin_y, height);
}

// Returns a copy of mv_cost with the MV cost disabled, for the searches on the
// SAD alone.
static inline PmeMvCost get_no_mv_cost(const PmeMvCost *const mv_cost) {
  PmeMvCost no_mv_cost = *mv_cost;
  no_mv_cost.params.mv_cost_type = MV_COST_NONE;
  return no_mv_cost;
}

// Searches every full-pel position of the search window at (orig_x, orig_y),
// clamped to the frame, for the block at (pos_x, pos_y), and stores the best
// MV and its cost in pme_stats.
static void pme_motion_search(PyramidLayer *const cur_lyr,
                              PyramidLayer *const ref_lyr,
                              MvStats *const pme_stats, PmeMvCost *mv_cost,
                              int pos_x, int pos_y, int orig_x, int orig_y,
                              int search_window_w, int search_window_h,
                              BLOCK_SIZE bsize, int aligned_bw, int aligned_bh,
                              bool use_alt_rows_for_sad) {
  // Calculate search origin and search window.
  int search_origin_x = orig_x;
  int search_origin_y = orig_y;
  clamp_search_area(pos_x, pos_y, block_size_wide[bsize] - 1,
                    block_size_high[bsize] - 1, cur_lyr->width, cur_lyr->height,
                    &search_origin_x, &search_origin_y, &search_window_w,
                    &search_window_h);

  // Search window based motion search.
  const uint8_t *const cur_buf =
      &cur_lyr->buffer[pos_y * cur_lyr->stride + pos_x];
  const uint8_t *const ref_buf =
      &ref_lyr->buffer[(search_origin_y + pos_y) * ref_lyr->stride +
                       search_origin_x + pos_x];
  int best_cost = INT_MAX;
  FULLPEL_MV best_mv = { INVALID_MV_ROW_COL, INVALID_MV_ROW_COL };
  mv_cost->origin.col = 0;
  mv_cost->origin.row = 0;
  mv_cost->sad_shift = use_alt_rows_for_sad;
  // With sub-sampled SAD only every other row of the block is used.
  const int cur_stride = cur_lyr->stride << use_alt_rows_for_sad;
  const int ref_stride = ref_lyr->stride << use_alt_rows_for_sad;
  const int sad_bh = aligned_bh >> use_alt_rows_for_sad;
  av1_pme_sad_loop_kernel(cur_buf, cur_stride, ref_buf, ref_stride, aligned_bw,
                          sad_bh, &best_cost, &best_mv, ref_lyr->stride,
                          search_window_w, search_window_h, mv_cost);
  // Adjust the MV w.r.t origin of the search window.
  best_mv.col += search_origin_x;
  best_mv.row += search_origin_y;

  pme_stats->cost = best_cost;
  pme_stats->mv.as_fullmv = best_mv;
}

// Calculates scale factor for search window size.
static inline int get_search_window_factor(int cur_disp_order_idx,
                                           int ref_disp_order_idx) {
  const int rel_dist =
      av1_encoder_get_relative_dist(cur_disp_order_idx, ref_disp_order_idx);
  const int abs_rel_dist = abs(rel_dist);
  const int round_offset = !!(abs_rel_dist & 7);
  const int scale_factor = (abs_rel_dist * 5 / 8) + round_offset;

  return scale_factor;
}

// Search window scale weight / denom for the q index.
static inline void get_qp_based_search_area_scale(int qp_index, int *weight,
                                                  int *denom) {
  const int qp = av1_qindex_to_quantizer(qp_index);
  *weight = AOMMAX(10, qp);
  *denom = MAXQ / 4;
  if (qp >= 46) {
    const double expo = -(AOMMAX(40, (double)qp) - 35) / 10;
    double q_weight = exp(expo);
    q_weight = 1.05 - q_weight;
    q_weight *= 10000;

    *weight = (int)q_weight;
    *denom = 10000;
  }
}

// Scales the search window sizes by weight / denom, to at least 8.
static inline void scale_search_area_qp_based(SearchWindowCfg *const sw_cfg,
                                              int weight, int denom) {
  sw_cfg->min_w = AOMMAX(8, DIVIDE_AND_ROUND(sw_cfg->min_w * weight, denom));
  sw_cfg->min_h = AOMMAX(8, DIVIDE_AND_ROUND(sw_cfg->min_h * weight, denom));
  sw_cfg->max_w = AOMMAX(8, DIVIDE_AND_ROUND(sw_cfg->max_w * weight, denom));
  sw_cfg->max_h = AOMMAX(8, DIVIDE_AND_ROUND(sw_cfg->max_h * weight, denom));
}

// Gets the search windows of pyramid ME level 2 and post pyramid ME: the speed
// feature windows scaled by the q index. They do not depend on the block.
static void get_search_windows_for_pme(
    const PYRAMID_ME_SPEED_FEATURES *const pme_sf, int qindex,
    SearchWindowCfg *const lvl2_search_window,
    SearchWindowCfg *const post_search_window) {
  *lvl2_search_window = pme_sf->lvl2_search_window;
  *post_search_window = pme_sf->post_search_window;
  if (!pme_sf->lvl2_qp_based_window_scaling &&
      !pme_sf->post_qp_based_window_scaling)
    return;
  // Same scale for both windows.
  int weight, denom;
  get_qp_based_search_area_scale(qindex, &weight, &denom);
  if (pme_sf->lvl2_qp_based_window_scaling)
    scale_search_area_qp_based(lvl2_search_window, weight, denom);
  if (pme_sf->post_qp_based_window_scaling)
    scale_search_area_qp_based(post_search_window, weight, denom);
}

// Returns the search window of a pre pyramid ME search zone: the minimum size
// scaled by the temporal distance to the reference frame, up to the maximum.
static inline void get_pre_pme_search_window(
    int cur_disp_order_idx, int ref_disp_order_idx,
    const SearchWindowCfg *search_window_cfg, int *search_window_w,
    int *search_window_h) {
  const int scale_factor =
      get_search_window_factor(cur_disp_order_idx, ref_disp_order_idx);

  // Calculate search window size.
  *search_window_w =
      AOMMIN(search_window_cfg->max_w, search_window_cfg->min_w * scale_factor);
  *search_window_h =
      AOMMIN(search_window_cfg->max_h, search_window_cfg->min_h * scale_factor);
}

// Returns the search window of a pyramid ME level 2 search zone: the window
// configuration scaled by the temporal distance and split over the search
// zones.
static inline void get_pme_lvl2_search_window(
    int cur_disp_order_idx, int ref_disp_order_idx,
    const SearchWindowCfg *search_window_cfg, int *search_window_w,
    int *search_window_h) {
  const int scale_factor =
      get_search_window_factor(cur_disp_order_idx, ref_disp_order_idx);

  // TODO(apurve.pandey): Perform (min_w * scale_factor) / PME_SEARCH_ZONE_COLS
  // in single step to avoid rounding error.
  *search_window_w = search_window_cfg->min_w / PME_SEARCH_ZONE_COLS;
  *search_window_w = ALIGN_POWER_OF_TWO(*search_window_w * scale_factor, 4);
  *search_window_w = AOMMIN(
      *search_window_w,
      ALIGN_POWER_OF_TWO(search_window_cfg->max_w / PME_SEARCH_ZONE_COLS, 4));

  *search_window_h = search_window_cfg->min_h / PME_SEARCH_ZONE_ROWS;
  *search_window_h = AOMMIN(*search_window_h * scale_factor,
                            search_window_cfg->max_h / PME_SEARCH_ZONE_ROWS);
}

// Returns the zero MV SAD of the bw x bh block, at most 64x64, at cur_buf and
// ref_buf. With use_alt_rows_for_sad, only every other row is summed and the
// SAD is doubled. With normalize_sad, the SAD of a block smaller than 64x64
// (at the frame edge) is scaled to the 64x64 area.
static inline int get_b64_zero_mv_sad(const uint8_t *cur_buf, int cur_stride,
                                      const uint8_t *ref_buf, int ref_stride,
                                      int bw, int bh, bool use_alt_rows_for_sad,
                                      bool normalize_sad) {
  const bool is_full_block =
      bw == block_size_wide[BLOCK_64X64] && bh == block_size_high[BLOCK_64X64];
  int sad = 0;
  if (is_full_block) {
    // The pyramid is 8-bit for any input bit depth, so the 8-bit SAD is used,
    // not cpi->ppi->fn_ptr, which is high bit depth for high bit depth input.
    sad = use_alt_rows_for_sad
              ? aom_sad_skip_64x64(cur_buf, cur_stride, ref_buf, ref_stride)
              : aom_sad64x64(cur_buf, cur_stride, ref_buf, ref_stride);
  } else {
    // Frame edge block: bw and bh are any multiple of 8 up to 64, which has no
    // fixed size SAD function.
    const int sad_cur_stride = cur_stride << use_alt_rows_for_sad;
    const int sad_ref_stride = ref_stride << use_alt_rows_for_sad;
    const int sad_bh = bh >> use_alt_rows_for_sad;
    for (int y = 0; y < sad_bh; ++y) {
      for (int x = 0; x < bw; ++x) {
        sad += abs(cur_buf[y * sad_cur_stride + x] -
                   ref_buf[y * sad_ref_stride + x]);
      }
    }
    if (normalize_sad) {
      sad = (int)((int64_t)sad * block_size_wide[BLOCK_64X64] *
                  (block_size_high[BLOCK_64X64] >> use_alt_rows_for_sad) /
                  (bw * sad_bh));
    }
    sad <<= use_alt_rows_for_sad;
  }
  return sad;
}

// Compares the SAD of the 64x64 block at the search origin (*orig_x, *orig_y),
// clamped to the frame, with its zero MV SAD, and resets the origin to the
// zero MV if that is not worse.
static inline void check_search_origin(const AV1_COMP *const cpi,
                                       PyramidLayer *const cur_lyr,
                                       PyramidLayer *const ref_lyr, int *orig_x,
                                       int *orig_y, int pos_x, int pos_y,
                                       int aligned_bw, int aligned_bh,
                                       const PmeMvCost *const mv_cost) {
  const BLOCK_SIZE bsize = BLOCK_64X64;
  MvStats origin_mv_stats = { 0 };
  // SADs on every other row, as the search origin SAD below is computed, so
  // that the two are comparable.
  const bool use_alt_rows_for_sad = cpi->sf.pme_sf.use_alt_rows_for_sad;

  // Calculate SAD of zero mv.
  const uint8_t *const cur_buf =
      &cur_lyr->buffer[pos_y * cur_lyr->stride + pos_x];
  const uint8_t *const ref_buf =
      &ref_lyr->buffer[pos_y * ref_lyr->stride + pos_x];
  const int zero_mv_sad =
      get_b64_zero_mv_sad(cur_buf, cur_lyr->stride, ref_buf, ref_lyr->stride,
                          aligned_bw, aligned_bh, use_alt_rows_for_sad,
                          /*normalize_sad=*/false);

  // Calculate SAD of search origin.
  if (pos_x + *orig_x > ref_lyr->width - 1) {
    *orig_x = ref_lyr->width - 1 - pos_x;
  }
  if (pos_y + *orig_y > ref_lyr->height - 1) {
    *orig_y = ref_lyr->height - 1 - pos_y;
  }
  // SAD without MV cost, as it is compared against zero_mv_sad.
  PmeMvCost no_mv_cost = get_no_mv_cost(mv_cost);
  pme_motion_search(cur_lyr, ref_lyr, &origin_mv_stats, &no_mv_cost, pos_x,
                    pos_y, *orig_x, *orig_y, /*search_window_w=*/1,
                    /*search_window_h=*/1, bsize, aligned_bw, aligned_bh,
                    use_alt_rows_for_sad);
  *orig_x = origin_mv_stats.mv.as_fullmv.col;
  *orig_y = origin_mv_stats.mv.as_fullmv.row;
  if (zero_mv_sad <= origin_mv_stats.cost) {
    *orig_x = 0;
    *orig_y = 0;
  }
}

// Calculates the search window of post pyramid ME, and may move the search
// center to MV 0.
static void get_post_pme_search_window(
    const AV1_COMP *const cpi, PyramidLayer *const cur_lyr,
    PyramidLayer *const ref_lyr, int *const orig_x, int *const orig_y,
    const SearchWindowCfg *const search_window_cfg, int *const search_window_w,
    int *const search_window_h, int pos_x, int pos_y, int cur_disp_order_idx,
    int ref_disp_order_idx, int best_pme_sad, int zero_mv_sad, int rf_idx,
    PmeMvCost *mv_cost) {
  const PmeParams *const pme_data = &cpi->ppi->pme_data;
  const MV_REFERENCE_FRAME ref_frame = rf_idx + LAST_FRAME;

  const PYRAMID_ME_SPEED_FEATURES *const pme_sf = &cpi->sf.pme_sf;
  const bool mv_based_search_win = pme_sf->post_mv_based_search_win;
  const bool is_nearest_ref = pme_sf->post_is_nearest_ref;
  const int search_win_thresh = pme_sf->post_search_win_thresh;
  const int pme_early_exit_thresh = pme_sf->early_exit_thresh;
  const int adjust_search_win = pme_sf->post_adjust_search_win;
  const int search_orig_thresh = pme_sf->post_search_orig_thresh;
  int abs_static_sad_thresh = pme_sf->post_abs_static_sad_thresh;
  int abs_pme_sad_thresh = pme_sf->post_abs_pme_sad_thresh;
  const int static_pme_scale = pme_sf->post_static_scale;
  const int low_pme_sad_scale = pme_sf->post_low_sad_scale;

  // Reduce the thresholds when pyramid level 0 is disabled.
  if (!pme_sf->enable_pyr_level[PYR_LEVEL0]) {
    if (pme_sf->enable_pyr_level[PYR_LEVEL1]) {
      abs_static_sad_thresh /= 4;
      abs_pme_sad_thresh /= 4;
    } else {
      abs_static_sad_thresh /= 16;
      abs_pme_sad_thresh /= 16;
    }
  }
  *search_window_w = search_window_cfg->min_w;
  *search_window_h = search_window_cfg->min_h;

  // Calculate search window scale based on temporal distance.
  const int scale_factor =
      get_search_window_factor(cur_disp_order_idx, ref_disp_order_idx);
  *search_window_w =
      AOMMIN(*search_window_w * scale_factor, search_window_cfg->max_w);
  *search_window_h =
      AOMMIN(*search_window_h * scale_factor, search_window_cfg->max_h);

  if (mv_based_search_win && (!is_nearest_ref || ref_frame == LAST_FRAME ||
                              ref_frame == BWDREF_FRAME)) {
    *search_window_w = abs(*orig_x) > search_win_thresh ? (*search_window_w * 2)
                                                        : *search_window_w;
    *search_window_h = abs(*orig_y) > search_win_thresh ? (*search_window_h * 2)
                                                        : *search_window_h;
  }

  // Calculate search window scale based on best MV and SAD.
  int search_window_scale = 1;
  if (adjust_search_win && abs(*orig_x) <= search_orig_thresh &&
      abs(*orig_y) <= search_orig_thresh &&
      best_pme_sad < abs_static_sad_thresh) {
    search_window_scale = static_pme_scale;
  } else if (best_pme_sad < abs_pme_sad_thresh) {
    search_window_scale = low_pme_sad_scale;
  }

  *search_window_w =
      AOMMAX(1, ALIGN_POWER_OF_TWO(*search_window_w / search_window_scale, 3));
  *search_window_h = AOMMAX(3, *search_window_h / search_window_scale);

  if (pme_early_exit_thresh) {
    if (zero_mv_sad < pme_early_exit_thresh / 6) {
      *search_window_w = 1;
      *search_window_h = 1;
    }
  } else if (pme_data->is_frame_used_as_ref && (*orig_x || *orig_y)) {
    check_search_origin(cpi, cur_lyr, ref_lyr, orig_x, orig_y, pos_x, pos_y,
                        pme_data->aligned_bw, pme_data->aligned_bh, mv_cost);
  }
}

// Get last pyramid motion estimation stage enabled as per configuration.
static inline PYRAMID_LEVEL get_last_pyr_level(
    const bool *const enable_pyr_me) {
  // Ensure PYR_LEVEL1 is enabled if PYR_LEVEL0 is disabled. This is to ensure
  // that the pyramid levels are enabled in order.
  assert(IMPLIES(!enable_pyr_me[PYR_LEVEL0], enable_pyr_me[PYR_LEVEL1]));
  // Level 2 must always be enabled.
  assert(enable_pyr_me[PYR_LEVEL2]);

  if (enable_pyr_me[PYR_LEVEL0])
    return PYR_LEVEL0;
  else if (enable_pyr_me[PYR_LEVEL1])
    return PYR_LEVEL1;
  else
    return PYR_LEVEL2;
}

// Returns whether the reference frame precedes the current frame in display
// order.
static inline bool is_fwd_ref_frame(int cur_disp_order_idx,
                                    int ref_disp_order_idx) {
  return cur_disp_order_idx > ref_disp_order_idx;
}

// Returns whether to skip the reference frame: frames at layer depth 1 are
// searched only against forward reference frames.
static inline bool skip_ref_for_layer_depth(int layer_depth,
                                            int cur_disp_order_idx,
                                            int ref_disp_order_idx) {
  return layer_depth == 1 &&
         !is_fwd_ref_frame(cur_disp_order_idx, ref_disp_order_idx);
}

// Searches every position of the search_area_width x search_area_height area
// for the block_width x block_height block, on its SAD shifted left by
// mv_cost->sad_shift plus the MV cost. Updates *best_cost and *best_mv, the
// position in the area, on an improvement.
void av1_pme_sad_loop_kernel_c(const uint8_t *src, int src_stride,
                               const uint8_t *ref, int ref_stride,
                               int block_width, int block_height,
                               int *best_cost, FULLPEL_MV *best_mv,
                               int src_stride_org, int search_area_width,
                               int search_area_height,
                               const PmeMvCost *mv_cost) {
  assert(mv_cost != NULL);
  int search_x_idx;
  int search_y_idx;

  for (search_y_idx = 0; search_y_idx < search_area_height; search_y_idx++) {
    const PmeRowCost pme_row =
        pme_row_cost(mv_cost, mv_cost->origin.row + search_y_idx);
    for (search_x_idx = 0; search_x_idx < search_area_width; search_x_idx++) {
      const uint8_t *const ref_blk = ref + search_x_idx;
      int sad = 0;
      for (int y = 0; y < block_height; ++y) {
        for (int x = 0; x < block_width; ++x) {
          sad += abs(src[y * src_stride + x] - ref_blk[y * ref_stride + x]);
        }
      }

      // The MV cost is never negative, so comparing the bare SAD first is
      // lossless and keeps the entropy cost off the common path.
      sad <<= mv_cost->sad_shift;
      if (sad < *best_cost) {
        const int cost = sad + pme_col_cost(mv_cost, &pme_row,
                                            mv_cost->origin.col + search_x_idx);
        if (cost < *best_cost) {
          *best_cost = cost;
          best_mv->col = search_x_idx;
          best_mv->row = search_y_idx;
        }
      }
    }

    ref += src_stride_org;
  }
}

// Computes the SADs of the four 8x8 blocks and of the 16x16 block at
// search_index consecutive horizontal offsets of ref, updates their running
// bests on SAD plus MV cost, and stores the 16x16 SADs in
// sad16x16_arr[start_pos]. With skip_row set, only the even rows are summed
// and the SADs are doubled.
static void sad_calc_8x8_16x16_core_c(MvStats *best_8x8, MvStats *best_16x16,
                                      uint8_t *src, int src_stride,
                                      uint8_t *ref, int ref_stride, int_mv mv,
                                      int start_pos, int sad16x16_arr[16][8],
                                      int search_index, bool skip_row,
                                      const PmeMvCost *mv_cost) {
  assert(mv_cost != NULL);
  const PmeRowCost pme_row = pme_row_cost(mv_cost, mv.as_fullmv.row);
  const int row_step = 1 << skip_row;  // 1 or 2
  const int rows = 8 >> skip_row;      // 8 or 4
  for (int s_i = 0; s_i < search_index; s_i++) {
    int_mv cand_mv = mv;
    cand_mv.as_fullmv.col = (int16_t)(mv.as_fullmv.col + s_i);
    // SADs of the four 8x8 quadrants of this 16x16.
    int sad8x8[4];
    int k = 0;
    for (int i = 0; i < 16; i += 8) {
      for (int j = 0; j < 16; j += 8) {
        const uint8_t *const src_blk = src + i * src_stride + j;
        const uint8_t *const ref_blk = ref + i * ref_stride + j + s_i;
        int sad = 0;
        for (int y = 0; y < rows; ++y) {
          for (int x = 0; x < 8; ++x) {
            sad += abs(src_blk[y * row_step * src_stride + x] -
                       ref_blk[y * row_step * ref_stride + x]);
          }
        }
        // << skip_row scales the 4-row sum back onto the 8x8 scale.
        sad8x8[k] = sad << skip_row;
        k++;
      }
    }
    // sad16x16_arr[] stores the SAD without MV cost, as
    // av1_pme_sad_calc_32x32_64x64_core() sums it into the 32x32 and 64x64
    // SADs. The MV cost is computed only once a SAD beats the running best, and
    // is shared by every block size at this search offset.
    int mvc = 0, have_mvc = 0;
    for (int b = 0; b < 4; b++) {
      if (sad8x8[b] < best_8x8[b].cost) {
        if (!have_mvc) {
          mvc = pme_col_cost(mv_cost, &pme_row, mv.as_fullmv.col + s_i);
          have_mvc = 1;
        }
        if (sad8x8[b] + mvc < best_8x8[b].cost) {
          best_8x8[b].cost = sad8x8[b] + mvc;
          best_8x8[b].mv.as_int = cand_mv.as_int;
        }
      }
    }

    int sad16x16 = sad16x16_arr[start_pos][s_i] =
        sad8x8[0] + sad8x8[1] + sad8x8[2] + sad8x8[3];

    if (sad16x16 < best_16x16->cost) {
      if (!have_mvc) {
        mvc = pme_col_cost(mv_cost, &pme_row, mv.as_fullmv.col + s_i);
        have_mvc = 1;
      }
      if (sad16x16 + mvc < best_16x16->cost) {
        best_16x16->cost = sad16x16 + mvc;
        best_16x16->mv.as_int = cand_mv.as_int;
      }
    }
  }
}

// Sums the 16x16 SADs of sad16x16_arr into the four 32x32 SADs and the 64x64
// SAD at each of the search_index offsets, and updates their running bests.
void av1_pme_sad_calc_32x32_64x64_core_c(MvStats *best_32x32,
                                         MvStats *best_64x64, int_mv mv,
                                         int sad16x16_arr[16][8],
                                         int search_index,
                                         const PmeMvCost *mv_cost) {
  assert(mv_cost != NULL);
  const PmeRowCost pme_row = pme_row_cost(mv_cost, mv.as_fullmv.row);
  for (int s_i = 0; s_i < search_index; s_i++) {
    int_mv cand_mv = mv;
    cand_mv.as_fullmv.col = (int16_t)(mv.as_fullmv.col + s_i);
    int sad32x32_0, sad32x32_1, sad32x32_2, sad32x32_3, sad64x64;
    int mvc = 0, have_mvc = 0;
// Computes the MV cost of the current offset, once, when first needed.
#define PME_NEED_COST()                                              \
  do {                                                               \
    if (!have_mvc) {                                                 \
      mvc = pme_col_cost(mv_cost, &pme_row, mv.as_fullmv.col + s_i); \
      have_mvc = 1;                                                  \
    }                                                                \
  } while (0)
    sad32x32_0 = sad16x16_arr[0][s_i] + sad16x16_arr[1][s_i] +
                 sad16x16_arr[2][s_i] + sad16x16_arr[3][s_i];
    if (sad32x32_0 < best_32x32[0].cost) PME_NEED_COST();
    if (sad32x32_0 < best_32x32[0].cost &&
        sad32x32_0 + mvc < best_32x32[0].cost) {
      best_32x32[0].cost = sad32x32_0 + mvc;
      best_32x32[0].mv.as_int = cand_mv.as_int;
    }
    sad32x32_1 = sad16x16_arr[4][s_i] + sad16x16_arr[5][s_i] +
                 sad16x16_arr[6][s_i] + sad16x16_arr[7][s_i];
    if (sad32x32_1 < best_32x32[1].cost) PME_NEED_COST();
    if (sad32x32_1 < best_32x32[1].cost &&
        sad32x32_1 + mvc < best_32x32[1].cost) {
      best_32x32[1].cost = sad32x32_1 + mvc;
      best_32x32[1].mv.as_int = cand_mv.as_int;
    }
    sad32x32_2 = sad16x16_arr[8][s_i] + sad16x16_arr[9][s_i] +
                 sad16x16_arr[10][s_i] + sad16x16_arr[11][s_i];
    if (sad32x32_2 < best_32x32[2].cost) PME_NEED_COST();
    if (sad32x32_2 < best_32x32[2].cost &&
        sad32x32_2 + mvc < best_32x32[2].cost) {
      best_32x32[2].cost = sad32x32_2 + mvc;
      best_32x32[2].mv.as_int = cand_mv.as_int;
    }
    sad32x32_3 = sad16x16_arr[12][s_i] + sad16x16_arr[13][s_i] +
                 sad16x16_arr[14][s_i] + sad16x16_arr[15][s_i];
    if (sad32x32_3 < best_32x32[3].cost) PME_NEED_COST();
    if (sad32x32_3 < best_32x32[3].cost &&
        sad32x32_3 + mvc < best_32x32[3].cost) {
      best_32x32[3].cost = sad32x32_3 + mvc;
      best_32x32[3].mv.as_int = cand_mv.as_int;
    }

    sad64x64 = sad32x32_0 + sad32x32_1 + sad32x32_2 + sad32x32_3;
    if (sad64x64 < best_64x64->cost) PME_NEED_COST();
    if (sad64x64 < best_64x64->cost && sad64x64 + mvc < best_64x64->cost) {
      best_64x64->cost = sad64x64 + mvc;
      best_64x64->mv.as_int = cand_mv.as_int;
    }
#undef PME_NEED_COST
  }
}

// Runs sad_calc_8x8_16x16_core_c() on each 16x16 block of a 64x64 block.
// The results are stored in the order of offsets[], in which the four 16x16
// blocks of each 32x32 block are consecutive.
void av1_pme_sad_calc_8x8_16x16_c(MvStats *best_b8, MvStats *best_b16,
                                  uint8_t *src, int src_stride, uint8_t *ref,
                                  int ref_stride, int_mv mv,
                                  int sad16x16_arr[16][8], int search_index,
                                  int skip_row, const PmeMvCost *mv_cost) {
  static const char offsets[16] = { 0, 1, 4,  5,  2,  3,  6,  7,
                                    8, 9, 12, 13, 10, 11, 14, 15 };
  for (int y = 0; y < 4; y++) {
    for (int x = 0; x < 4; x++) {
      const uint32_t block_index = 16 * y * src_stride + 16 * x;
      const uint32_t search_position_index = 16 * y * ref_stride + 16 * x;
      int start_pos = offsets[4 * y + x];
      const uint32_t start_8x8_pos = 4 * start_pos;

      MvStats *best_8x8 = &best_b8[start_8x8_pos];
      MvStats *best_16x16 = &best_b16[start_pos];

      sad_calc_8x8_16x16_core_c(best_8x8, best_16x16, src + block_index,
                                src_stride, ref + search_position_index,
                                ref_stride, mv, start_pos, sad16x16_arr,
                                search_index, skip_row, mv_cost);
    }
  }
}

// Computes the 8x8 to 64x64 SADs of a 64x64 block at search_index offsets,
// and updates their running bests.
static void sad_calc_sub_block_level(PmeSbStats *const this_lvl_sb_stats,
                                     uint8_t *src, int src_stride, uint8_t *ref,
                                     int ref_stride, int_mv mv,
                                     int sad16x16_arr[16][8], int search_index,
                                     int skip_row, const PmeMvCost *mv_cost) {
  // Both stages must use the same search_index: only
  // sad16x16_arr[][0 .. search_index-1] is filled by the 8x8/16x16 stage.
  av1_pme_sad_calc_8x8_16x16(this_lvl_sb_stats->post_pme_stats_b8,
                             this_lvl_sb_stats->post_pme_stats_b16, src,
                             src_stride, ref, ref_stride, mv, sad16x16_arr,
                             search_index, skip_row, mv_cost);
  av1_pme_sad_calc_32x32_64x64_core(this_lvl_sb_stats->post_pme_stats_b32,
                                    &this_lvl_sb_stats->post_pme_stats_b64, mv,
                                    sad16x16_arr, search_index, mv_cost);
}

// Resets the 8x8 to 64x64 block SADs and MVs of a 64x64 block, and its
// 16x16 SADs at each search offset, before the post pyramid ME search.
static inline void init_post_pme_sb_stats(PmeSbStats *const sb_stats,
                                          int sad16x16_arr[16][8]) {
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 8; ++j) sad16x16_arr[i][j] = INT_MAX;
  }
  for (int i = 0; i < 64; ++i) {
    sb_stats->post_pme_stats_b8[i].cost = INT_MAX;
    sb_stats->post_pme_stats_b8[i].mv.as_int = INVALID_MV;
  }
  for (int i = 0; i < 16; ++i) {
    sb_stats->post_pme_stats_b16[i].cost = INT_MAX;
    sb_stats->post_pme_stats_b16[i].mv.as_int = INVALID_MV;
  }
  for (int i = 0; i < 4; ++i) {
    sb_stats->post_pme_stats_b32[i].cost = INT_MAX;
    sb_stats->post_pme_stats_b32[i].mv.as_int = INVALID_MV;
  }
  sb_stats->post_pme_stats_b64.cost = INT_MAX;
  sb_stats->post_pme_stats_b64.mv.as_int = INVALID_MV;
}

// Scales the search area of the post pyramid ME by the variance of the 8x8
// SADs of the 64x64 block at the search center: grown when the variance is
// high, and shrunk when it is low, as the block is then likely to move as a
// whole.
static inline void scale_search_area_by_8x8_var(
    const PmeSbStats *const sb_stats, int mult2_thresh, int div4_thresh,
    int div2_thresh, int *const search_area_width,
    int *const search_area_height) {
  const int mean_dist_8x8 = sb_stats->post_pme_stats_b64.cost / 64;
  int64_t sum_ofsq_dist_8x8 = 0;
  for (int i = 0; i < 64; i++) {
    const int64_t diff = sb_stats->post_pme_stats_b8[i].cost - mean_dist_8x8;
    sum_ofsq_dist_8x8 += diff * diff;
  }
  const int64_t me_8x8_cost_var = sum_ofsq_dist_8x8 / 64;

  if (me_8x8_cost_var > mult2_thresh) {
    *search_area_width =
        ALIGN_POWER_OF_TWO(AOMMAX(1, *search_area_width * 3 / 2), 3);
    *search_area_height = AOMMAX(1, *search_area_height * 3 / 2);
  }

  if (me_8x8_cost_var < div4_thresh) {
    *search_area_width =
        ALIGN_POWER_OF_TWO(AOMMAX(1, *search_area_width >> 2), 3);
    *search_area_height = AOMMAX(3, *search_area_height >> 2);
  } else if (me_8x8_cost_var < div2_thresh) {
    *search_area_width = ALIGN_POWER_OF_TWO(*search_area_width >> 1, 3);
    *search_area_height = AOMMAX(3, *search_area_height >> 1);
  }
}

// Sets up the MV cost context of the pyramid ME of the sub-GOP. The cost is
// measured against the zero MV, as no MV predictor is available. The SAD
// multiplier of the MV rate is set per frame by init_frame_for_pme().
static inline void init_pme_mv_cost(AV1_COMP *cpi) {
  PmeMvCost *const mv_cost = &cpi->ppi->pme_data.mv_cost;
  MV_COST_PARAMS *const params = &mv_cost->params;
  mv_cost->origin.col = 0;
  mv_cost->origin.row = 0;
  mv_cost->sad_shift = 0;
  av1_zero(*params);
  params->mv_cost_type = MV_COST_NONE;
  if (cpi->sf.pme_sf.enable_mv_cost) {
    const MvCosts *const mv_costs = cpi->td.mb.mv_costs;
    params->mv_cost_type = MV_COST_ENTROPY;
    params->mvjcost = mv_costs->nmv_joint_cost;
    params->mvcost[0] = mv_costs->mv_cost_stack[0];
    params->mvcost[1] = mv_costs->mv_cost_stack[1];
  }
}

// Calculates the zero MV SAD of the 64x64 block (level 0) for each reference
// frame.
static void calc_zero_mv_sad(
    AV1_COMP *const cpi, ImagePyramid *cur_pyr,
    ImagePyramid *const ref_pyr[INTER_REFS_PER_FRAME],
    PmeSbStats (*const pme_sb_stats)[INTER_REFS_PER_FRAME], int pos_x,
    int pos_y) {
  const PYRAMID_LEVEL pyr_level = PYR_LEVEL0;
  const BLOCK_SIZE bsize = BLOCK_64X64;
  PyramidLayer *const cur_lyr = &cur_pyr->layers[pyr_level];
  const uint8_t *cur_buf = &cur_lyr->buffer[pos_y * cur_lyr->stride + pos_x];

  PmeParams *const pme_data = &cpi->ppi->pme_data;
  const int max_layer_depth = cpi->ppi->gf_group.max_layer_depth;
  const int layer_depth = cpi->ppi->gf_group.layer_depth[pme_data->frame_idx];
  const PmeFrameStats *const pme_frame =
      &pme_data->frame_stats_ptr[pme_data->frame_idx];
  const int *const ref_disp_order_idx = pme_data->ref_disp_order_idx;
  const int cur_disp_order_idx = pme_frame->display_index;
  int best_sad = INT_MAX;

  const int zero_sad_thresh = cpi->sf.pme_sf.zero_mv_sad_thresh;
  const int zero_sad_safe_lmt_thresh =
      cpi->sf.pme_sf.zero_mv_sad_safe_lmt_thresh;

  // Calculate SAD for 64x64 block
  bool has_fwd_ref = false;
  bool has_bwd_ref = false;
  for (int rf_idx = 0; rf_idx < INTER_REFS_PER_FRAME; ++rf_idx) {
    PmeSbStats *const this_sb_stats = &(*pme_sb_stats)[rf_idx];
    if (ref_pyr[rf_idx] == NULL) continue;

    const int ref_frame = rf_idx + 1;
    if (ref_frame <= GOLDEN_FRAME) has_fwd_ref = true;
    if (ref_frame >= BWDREF_FRAME && ref_frame <= ALTREF_FRAME)
      has_bwd_ref = true;

    if (skip_ref_for_layer_depth(layer_depth, cur_disp_order_idx,
                                 ref_disp_order_idx[rf_idx]))
      continue;

    PyramidLayer *const ref_lyr = &ref_pyr[rf_idx]->layers[pyr_level];
    const uint8_t *ref_buf = &ref_lyr->buffer[pos_y * ref_lyr->stride + pos_x];

    // Calculate block size within frame.
    const int aligned_width = ALIGN_POWER_OF_TWO(ref_lyr->width, 3);
    const int aligned_height = ALIGN_POWER_OF_TWO(ref_lyr->height, 3);
    const int blk_right_edge =
        AOMMIN(pos_x + block_size_wide[bsize], aligned_width);
    const int bw = blk_right_edge - pos_x;
    const int blk_bottom_edge =
        AOMMIN(pos_y + block_size_high[bsize], aligned_height);
    const int bh = blk_bottom_edge - pos_y;

    const int this_sad = get_b64_zero_mv_sad(
        cur_buf, cur_lyr->stride, ref_buf, ref_lyr->stride, bw, bh,
        /*use_alt_rows_for_sad=*/false, /*normalize_sad=*/true);
    this_sb_stats->zero_mv_sad = this_sad;
    if (this_sad < best_sad) best_sad = this_sad;
  }

  // Prune reference frame based on sad.
  if (layer_depth > 1 && best_sad < zero_sad_thresh) {
    for (int rf_idx = 0; rf_idx < INTER_REFS_PER_FRAME; ++rf_idx) {
      if (ref_pyr[rf_idx] == NULL) continue;
      if (rf_idx + 1 == LAST_FRAME) continue;

      PmeSbStats *const this_sb_stats = &(*pme_sb_stats)[rf_idx];
      if ((this_sb_stats->zero_mv_sad - best_sad) * 100 >
          zero_sad_thresh * best_sad)
        this_sb_stats->prune_ref = true;
    }
  }

  if (zero_sad_safe_lmt_thresh && has_fwd_ref && has_bwd_ref &&
      max_layer_depth > 0 && layer_depth >= max_layer_depth &&
      best_sad != INT_MAX) {
    bool enable_ref_frame_pruning = false;
    const int sad_last_ref = (*pme_sb_stats)[LAST_FRAME - 1].zero_mv_sad;
    const int sad_bwd_ref = (*pme_sb_stats)[BWDREF_FRAME - 1].zero_mv_sad;

    if (sad_last_ref != INT_MAX && sad_bwd_ref != INT_MAX &&
        sad_last_ref < zero_sad_safe_lmt_thresh &&
        sad_bwd_ref < zero_sad_safe_lmt_thresh)
      enable_ref_frame_pruning = true;

    if (enable_ref_frame_pruning) {
      for (int rf_idx = 0; rf_idx < INTER_REFS_PER_FRAME; ++rf_idx) {
        if ((rf_idx + 1) == LAST_FRAME || (rf_idx + 1) == BWDREF_FRAME)
          continue;
        (*pme_sb_stats)[rf_idx].prune_ref = true;
      }
    }
  }
}

// Pre pyramid motion estimation performed at 16x16 block size (level 2).
// Called before pyramid_me_level2() to catch large purely vertical or purely
// horizontal motion. It searches two thin search zones centered at MV 0: a tall
// and narrow zone, and a wide and short zone. The best of the two zones is used
// by pyramid_me_level2() to replace its worst search zone. This stage also
// copies the mirrored forward MVs to the backward reference frames at layer
// depth <= 1.
static void pre_pyramid_me(
    AV1_COMP *const cpi, ImagePyramid *cur_pyr,
    ImagePyramid *const ref_pyr[INTER_REFS_PER_FRAME],
    MvStats (*const pre_pme_stats)[INTER_REFS_PER_FRAME][PRE_PME_SEARCH_ZONES],
    PmeSbStats (*const pme_sb_stats)[INTER_REFS_PER_FRAME], int pos_x,
    int pos_y, PmeMvCost *mv_cost) {
  const PYRAMID_LEVEL pyr_level = PYR_LEVEL2;
  const BLOCK_SIZE bsize = BLOCK_16X16;
  PyramidLayer *const cur_lyr = &cur_pyr->layers[pyr_level];

  PmeParams *const pme_data = &cpi->ppi->pme_data;
  const int layer_depth = cpi->ppi->gf_group.layer_depth[pme_data->frame_idx];
  const PmeFrameStats *const pme_frame =
      &pme_data->frame_stats_ptr[pme_data->frame_idx];
  const int *const ref_disp_order_idx = pme_data->ref_disp_order_idx;
  const int cur_disp_order_idx = pme_frame->display_index;
  int best_cost = INT_MAX;

  const PYRAMID_ME_SPEED_FEATURES *const pme_sf = &cpi->sf.pme_sf;
  const SearchWindowCfg *const search_window_cfg = pme_sf->pre_search_window;
  const int pre_pme_early_exit_thresh = pme_sf->early_exit_thresh;
  const int pre_pme_sad_thresh = pme_sf->pre_sad_thresh;
  const int pre_pme_sad_pct = pme_sf->pre_sad_pct;

  pos_x >>= pyr_level;
  pos_y >>= pyr_level;
  const int aligned_bw = pme_data->aligned_bw >> pyr_level;
  const int aligned_bh = pme_data->aligned_bh >> pyr_level;

  // Perform pre-pyramid motion estimation, except against the backward
  // reference frames of the frames at layer depth 1.
  for (int rf_idx = 0; rf_idx < INTER_REFS_PER_FRAME; ++rf_idx) {
    if (ref_pyr[rf_idx] == NULL) continue;

    // Initialize pre PME sad.
    for (int sz_idx = 0; sz_idx < PRE_PME_SEARCH_ZONES; ++sz_idx) {
      (*pre_pme_stats)[rf_idx][sz_idx].mv.as_int = 0;
      (*pre_pme_stats)[rf_idx][sz_idx].cost = INT_MAX;
    }
    if (skip_ref_for_layer_depth(layer_depth, cur_disp_order_idx,
                                 ref_disp_order_idx[rf_idx]))
      continue;

    if ((*pme_sb_stats)[rf_idx].prune_ref) continue;

    if ((*pme_sb_stats)[rf_idx].zero_mv_sad < pre_pme_early_exit_thresh) {
      for (int sz_idx = 0; sz_idx < PRE_PME_SEARCH_ZONES; ++sz_idx)
        (*pre_pme_stats)[rf_idx][sz_idx].cost = 0;
      continue;
    }

    PyramidLayer *const ref_lyr = &ref_pyr[rf_idx]->layers[pyr_level];
    for (int sz_idx = 0; sz_idx < PRE_PME_SEARCH_ZONES; ++sz_idx) {
      MvStats *const this_stats = &(*pre_pme_stats)[rf_idx][sz_idx];
      int search_window_w;
      int search_window_h;
      get_pre_pme_search_window(cur_disp_order_idx, ref_disp_order_idx[rf_idx],
                                &search_window_cfg[sz_idx], &search_window_w,
                                &search_window_h);

      const int orig_x = -(search_window_w >> 1);
      const int orig_y = -(search_window_h >> 1);
      pme_motion_search(cur_lyr, ref_lyr, this_stats, mv_cost, pos_x, pos_y,
                        orig_x, orig_y, search_window_w, search_window_h, bsize,
                        aligned_bw, aligned_bh, pme_sf->use_alt_rows_for_sad);
      if (this_stats->cost < best_cost) {
        best_cost = this_stats->cost;
      }
    }
  }

  // Perform pre-pyramid motion estimation for backward ref frames.
  for (int rf_idx = 0; rf_idx < INTER_REFS_PER_FRAME; ++rf_idx) {
    if (ref_pyr[rf_idx] == NULL) continue;

    if (layer_depth > 1 ||
        is_fwd_ref_frame(cur_disp_order_idx, ref_disp_order_idx[rf_idx]))
      continue;

    // Copy motion data of forward ref frame to backward ref frame.
    const int fwd_rf_idx = get_mirror_rf_idx(
        ref_disp_order_idx, cur_disp_order_idx, ref_disp_order_idx[rf_idx]);
    if (fwd_rf_idx == -1) {
      aom_internal_error(cpi->common.error, AOM_CODEC_ERROR,
                         "No forward reference frame mirrors the backward "
                         "reference frame in pre pyramid ME.");
    }
    for (int search_zone = 0; search_zone < PRE_PME_SEARCH_ZONES;
         ++search_zone) {
      (*pre_pme_stats)[rf_idx][search_zone] =
          (*pre_pme_stats)[fwd_rf_idx][search_zone];
      // Change sign of motion vector.
      (*pre_pme_stats)[rf_idx][search_zone].mv.as_fullmv.col *= -1;
      (*pre_pme_stats)[rf_idx][search_zone].mv.as_fullmv.row *= -1;
    }
  }

  // Prune reference frame based on sad.
  if (layer_depth > 1 && best_cost < pre_pme_sad_thresh) {
    for (int rf_idx = 0; rf_idx < INTER_REFS_PER_FRAME; ++rf_idx) {
      if (ref_pyr[rf_idx] == NULL) continue;

      if (is_fwd_ref_frame(cur_disp_order_idx, ref_disp_order_idx[rf_idx]))
        continue;

      const int best_zone_cost = AOMMIN((*pre_pme_stats)[rf_idx][0].cost,
                                        (*pre_pme_stats)[rf_idx][1].cost);
      if ((best_zone_cost - best_cost) * 100.0 >
          pre_pme_sad_pct * best_zone_cost) {
        (*pme_sb_stats)[rf_idx].prune_ref = true;
      }
    }
  }
}

// Pyramid motion estimation performed at 64x64 block size (level 0).
static void pyramid_me_level0(
    AV1_COMP *const cpi, ImagePyramid *cur_pyr,
    ImagePyramid *const ref_pyr[INTER_REFS_PER_FRAME],
    PmeSbStats (*const pme_sb_stats)[INTER_REFS_PER_FRAME], int pos_x,
    int pos_y, PmeMvCost *mv_cost) {
  PmeParams *const pme_data = &cpi->ppi->pme_data;
  const int layer_depth = cpi->ppi->gf_group.layer_depth[pme_data->frame_idx];
  const PmeFrameStats *const pme_frame =
      &pme_data->frame_stats_ptr[pme_data->frame_idx];
  const int *const ref_disp_order_idx = pme_data->ref_disp_order_idx;
  const int cur_disp_order_idx = pme_frame->display_index;

  const PYRAMID_LEVEL pyr_level = PYR_LEVEL0;
  const BLOCK_SIZE bsize = BLOCK_64X64;
  PyramidLayer *const cur_lyr = &cur_pyr->layers[pyr_level];

  pos_x >>= pyr_level;
  pos_y >>= pyr_level;
  const int aligned_bw = pme_data->aligned_bw >> pyr_level;
  const int aligned_bh = pme_data->aligned_bh >> pyr_level;

  const PYRAMID_ME_SPEED_FEATURES *const pme_sf = &cpi->sf.pme_sf;
  const int search_window_w = pme_sf->lvl0_search_window_w;
  const int search_window_h = pme_sf->lvl0_search_window_h;

  for (int rf_idx = 0; rf_idx < INTER_REFS_PER_FRAME; ++rf_idx) {
    if (ref_pyr[rf_idx] == NULL) continue;

    if (skip_ref_for_layer_depth(layer_depth, cur_disp_order_idx,
                                 ref_disp_order_idx[rf_idx]))
      continue;

    PmeSbStats *const this_sb_stats = &(*pme_sb_stats)[rf_idx];
    if (this_sb_stats->prune_ref) continue;

    PyramidLayer *const ref_lyr = &ref_pyr[rf_idx]->layers[pyr_level];
    for (int sz_idx = 0; sz_idx < PME_SEARCH_ZONES; ++sz_idx) {
      MvStats *const this_pme_stats =
          &this_sb_stats->pme_stats[pyr_level][sz_idx];
      MvStats *lvl1_pme_stats = &this_sb_stats->pme_stats[PYR_LEVEL1][sz_idx];

      // Convert MV precision to match this pyramid level.
      const int orig_x =
          lvl1_pme_stats->mv.as_fullmv.col * 2 - (search_window_w >> 1);
      const int orig_y =
          lvl1_pme_stats->mv.as_fullmv.row * 2 - (search_window_h >> 1);

      pme_motion_search(cur_lyr, ref_lyr, this_pme_stats, mv_cost, pos_x, pos_y,
                        orig_x, orig_y, search_window_w, search_window_h, bsize,
                        aligned_bw, aligned_bh, pme_sf->use_alt_rows_for_sad);
    }  // Search zone
  }  // rf_idx
}

// Pyramid motion estimation performed at 32x32 block size (level 1).
static void pyramid_me_level1(
    AV1_COMP *const cpi, ImagePyramid *cur_pyr,
    ImagePyramid *const ref_pyr[INTER_REFS_PER_FRAME],
    PmeSbStats (*const pme_sb_stats)[INTER_REFS_PER_FRAME], int pos_x,
    int pos_y, PmeMvCost *mv_cost) {
  PmeParams *const pme_data = &cpi->ppi->pme_data;
  const int layer_depth = cpi->ppi->gf_group.layer_depth[pme_data->frame_idx];
  const PmeFrameStats *const pme_frame =
      &pme_data->frame_stats_ptr[pme_data->frame_idx];
  const int *const ref_disp_order_idx = pme_data->ref_disp_order_idx;
  const int cur_disp_order_idx = pme_frame->display_index;

  const PYRAMID_LEVEL pyr_level = PYR_LEVEL1;
  const BLOCK_SIZE bsize = BLOCK_32X32;
  PyramidLayer *const cur_lyr = &cur_pyr->layers[pyr_level];

  pos_x >>= pyr_level;
  pos_y >>= pyr_level;
  const int aligned_bw = pme_data->aligned_bw >> pyr_level;
  const int aligned_bh = pme_data->aligned_bh >> pyr_level;

  const PYRAMID_ME_SPEED_FEATURES *const pme_sf = &cpi->sf.pme_sf;
  const int pme_early_exit_thresh = pme_sf->early_exit_thresh;
  const int search_window_w = pme_sf->lvl1_search_window_w;
  const int search_window_h = pme_sf->lvl1_search_window_h;

  for (int rf_idx = 0; rf_idx < INTER_REFS_PER_FRAME; ++rf_idx) {
    if (ref_pyr[rf_idx] == NULL) continue;

    if (skip_ref_for_layer_depth(layer_depth, cur_disp_order_idx,
                                 ref_disp_order_idx[rf_idx]))
      continue;

    PmeSbStats *const this_sb_stats = &(*pme_sb_stats)[rf_idx];
    MvStats *const this_pme_stats = this_sb_stats->pme_stats[pyr_level];

    for (int sz_idx = 0; sz_idx < PME_SEARCH_ZONES; ++sz_idx) {
      this_pme_stats[sz_idx].mv.as_int = 0;
      this_pme_stats[sz_idx].cost = INT_MAX;
    }

    if (this_sb_stats->prune_ref) continue;
    if (this_sb_stats->zero_mv_sad < pme_early_exit_thresh >> 2) {
      for (int sz_idx = 0; sz_idx < PME_SEARCH_ZONES; ++sz_idx)
        this_pme_stats[sz_idx].cost = 0;
      continue;
    }

    PyramidLayer *const ref_lyr = &ref_pyr[rf_idx]->layers[pyr_level];
    for (int sz_idx = 0; sz_idx < PME_SEARCH_ZONES; ++sz_idx) {
      MvStats *const lvl2_pme_stats =
          &this_sb_stats->pme_stats[PYR_LEVEL2][sz_idx];

      // Convert MV precision to match this pyramid level.
      const int orig_x =
          lvl2_pme_stats->mv.as_fullmv.col * 2 - (search_window_w >> 1);
      const int orig_y =
          lvl2_pme_stats->mv.as_fullmv.row * 2 - (search_window_h >> 1);

      pme_motion_search(cur_lyr, ref_lyr, &this_pme_stats[sz_idx], mv_cost,
                        pos_x, pos_y, orig_x, orig_y, search_window_w,
                        search_window_h, bsize, aligned_bw, aligned_bh,
                        pme_sf->use_alt_rows_for_sad);
    }  // Search zone
  }  // rf_idx
}

// Pyramid motion estimation performed at 16x16 block size (level 2).
// This is the first stage of the coarse-to-fine search. Unlike
// pre_pyramid_me(), it searches a large square area centered at MV 0, split
// into a 2x2 grid of search zones, and keeps one MV per zone as the starting
// point for level 1. The worst zone is then replaced by the best
// pre_pyramid_me() result, so the long vertical or horizontal motion found by
// pre_pyramid_me() is also refined by the next levels.
static void pyramid_me_level2(
    AV1_COMP *const cpi, ImagePyramid *cur_pyr,
    ImagePyramid *const ref_pyr[INTER_REFS_PER_FRAME],
    MvStats (*const pre_pme_stats)[INTER_REFS_PER_FRAME][PRE_PME_SEARCH_ZONES],
    PmeSbStats (*const pme_sb_stats)[INTER_REFS_PER_FRAME], int pos_x,
    int pos_y, const SearchWindowCfg *const lvl2_search_window,
    PmeMvCost *mv_cost) {
  PmeParams *const pme_data = &cpi->ppi->pme_data;
  const int layer_depth = cpi->ppi->gf_group.layer_depth[pme_data->frame_idx];
  const PmeFrameStats *const pme_frame =
      &pme_data->frame_stats_ptr[pme_data->frame_idx];
  const int *const ref_disp_order_idx = pme_data->ref_disp_order_idx;
  const int cur_disp_order_idx = pme_frame->display_index;

  const PYRAMID_LEVEL pyr_level = PYR_LEVEL2;
  const BLOCK_SIZE bsize = BLOCK_16X16;
  PyramidLayer *const cur_lyr = &cur_pyr->layers[pyr_level];

  pos_x >>= pyr_level;
  pos_y >>= pyr_level;
  const int aligned_bw = pme_data->aligned_bw >> pyr_level;
  const int aligned_bh = pme_data->aligned_bh >> pyr_level;

  const PYRAMID_ME_SPEED_FEATURES *const pme_sf = &cpi->sf.pme_sf;
  const int pme_early_exit_thresh = pme_sf->early_exit_thresh;
  for (int rf_idx = 0; rf_idx < INTER_REFS_PER_FRAME; ++rf_idx) {
    if (ref_pyr[rf_idx] == NULL) continue;
    PmeSbStats *const this_sb_stats = &(*pme_sb_stats)[rf_idx];
    MvStats *const this_pme_stats = this_sb_stats->pme_stats[pyr_level];

    for (int sz_idx = 0; sz_idx < PME_SEARCH_ZONES; ++sz_idx) {
      this_pme_stats[sz_idx].mv.as_int = 0;
      this_pme_stats[sz_idx].cost = INT_MAX;
    }

    if (this_sb_stats->prune_ref) continue;
    if (this_sb_stats->zero_mv_sad < pme_early_exit_thresh >> 2) {
      for (int sz_idx = 0; sz_idx < PME_SEARCH_ZONES; ++sz_idx)
        this_pme_stats[sz_idx].cost = 0;
      continue;
    }

    const int best_pre_pme_sz =
        (*pre_pme_stats)[rf_idx][0].cost <= (*pre_pme_stats)[rf_idx][1].cost
            ? 0
            : 1;
    const int best_pre_pme_cost =
        (*pre_pme_stats)[rf_idx][best_pre_pme_sz].cost;

    if (skip_ref_for_layer_depth(layer_depth, cur_disp_order_idx,
                                 ref_disp_order_idx[rf_idx]))
      continue;

    int search_window_w;
    int search_window_h;
    get_pme_lvl2_search_window(cur_disp_order_idx, ref_disp_order_idx[rf_idx],
                               lvl2_search_window, &search_window_w,
                               &search_window_h);

    PyramidLayer *const ref_lyr = &ref_pyr[rf_idx]->layers[pyr_level];
    for (int sz_idx = 0; sz_idx < PME_SEARCH_ZONES; ++sz_idx) {
      // Calculate search origin for this search zone.
      const int sz_x = sz_idx % PME_SEARCH_ZONE_COLS;
      const int sz_y = sz_idx / PME_SEARCH_ZONE_COLS;
      const int orig_offset_x = search_window_w * sz_x;
      const int orig_offset_y = search_window_h * sz_y;
      const int orig_x =
          orig_offset_x - ((search_window_w * PME_SEARCH_ZONE_COLS) >> 1);
      const int orig_y =
          orig_offset_y - ((search_window_h * PME_SEARCH_ZONE_ROWS) >> 1);

      pme_motion_search(cur_lyr, ref_lyr, &this_pme_stats[sz_idx], mv_cost,
                        pos_x, pos_y, orig_x, orig_y, search_window_w,
                        search_window_h, bsize, aligned_bw, aligned_bh,
                        pme_sf->use_alt_rows_for_sad);
    }  // Search zone

    // Get worst search zone based on sad.
    int worst_pme_cost = 0;
    int worst_pme_sz = 0;
    for (int sz_idx = 0; sz_idx < PME_SEARCH_ZONES; ++sz_idx) {
      if (this_pme_stats[sz_idx].cost >= worst_pme_cost) {
        worst_pme_cost = this_pme_stats[sz_idx].cost;
        worst_pme_sz = sz_idx;
      }
    }

    // Replace worst search zone by pre PME.
    if (best_pre_pme_cost < worst_pme_cost) {
      this_pme_stats[worst_pme_sz] = (*pre_pme_stats)[rf_idx][best_pre_pme_sz];
    }
  }  // rf_idx
}

// Post pyramid motion estimation at full resolution (level 0). Searches around
// the best MV of each reference frame, and keeps the best MV and cost of the
// 64x64 block and each of its 32x32, 16x16 and 8x8 sub-blocks.
static void post_pyramid_me(AV1_COMP *const cpi, ImagePyramid *cur_pyr,
                            ImagePyramid *ref_pyr[INTER_REFS_PER_FRAME],
                            PmeSbStats (*pme_sb_stats)[INTER_REFS_PER_FRAME],
                            int best_pme_cost, int pos_x, int pos_y,
                            const SearchWindowCfg *const post_search_window,
                            PmeMvCost *mv_cost) {
  MACROBLOCK *const x = &cpi->td.mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  const PYRAMID_LEVEL pyr_level = PYR_LEVEL0;
  PyramidLayer *const cur_lyr = &cur_pyr->layers[pyr_level];
  struct buf_2d *const cur_y = &x->plane[AOM_PLANE_Y].src;
  struct buf_2d *const ref_y = &xd->plane[AOM_PLANE_Y].pre[0];
  PmeParams *const pme_data = &cpi->ppi->pme_data;
  const int layer_depth = cpi->ppi->gf_group.layer_depth[pme_data->frame_idx];
  const int cur_disp_order_idx =
      pme_data->frame_stats_ptr[pme_data->frame_idx].display_index;
  const int *const ref_disp_order_idx = pme_data->ref_disp_order_idx;
  const PYRAMID_ME_SPEED_FEATURES *const pme_sf = &cpi->sf.pme_sf;
  // Layer depth 0 (key and golden frames) and 1 (ARF) form the base layer.
  const int prune_ref_sad_thresh =
      pme_sf->post_prune_ref_sad_thresh[layer_depth <= 1 ? 0 : 1];
  const int var_8x8_enable = pme_sf->post_var_8x8_enable;
  const int var_8x8_mult2_thresh = pme_sf->post_var_8x8_mult2_thresh;
  const int var_8x8_div4_thresh = pme_sf->post_var_8x8_div4_thresh;
  const int var_8x8_div2_thresh = pme_sf->post_var_8x8_div2_thresh;
  const PYRAMID_LEVEL last_pyr_level =
      get_last_pyr_level(pme_sf->enable_pyr_level);

  for (int rf_idx = 0; rf_idx < INTER_REFS_PER_FRAME; ++rf_idx) {
    PmeSbStats *const this_sb_stats = &(*pme_sb_stats)[rf_idx];
    const MV_REFERENCE_FRAME ref_frame = rf_idx + LAST_FRAME;

    // Reset before any skip, so that skipped references hold no stale stats.
    int sad16x16_arr[16][8];
    init_post_pme_sb_stats(this_sb_stats, sad16x16_arr);

    if (ref_pyr[rf_idx] == NULL) continue;
    if (this_sb_stats->prune_ref) continue;
    if (skip_ref_for_layer_depth(layer_depth, cur_disp_order_idx,
                                 ref_disp_order_idx[rf_idx]))
      continue;

    // Prune references whose SAD is much worse than the best reference.
    if (prune_ref_sad_thresh != INT_MAX && ref_frame != LAST_FRAME &&
        ref_frame != BWDREF_FRAME) {
      if ((this_sb_stats->best_pme_stats.cost - best_pme_cost) * 100 >
          prune_ref_sad_thresh * best_pme_cost) {
        this_sb_stats->prune_ref = true;
        continue;
      }
    }

    PyramidLayer *const ref_lyr = &ref_pyr[rf_idx]->layers[pyr_level];
    const BLOCK_SIZE bsize = BLOCK_64X64;
    const int mi_col = pos_x >> MI_SIZE_LOG2;
    const int mi_row = pos_y >> MI_SIZE_LOG2;

    int x_search_center = this_sb_stats->best_pme_stats.mv.as_fullmv.col;
    int y_search_center = this_sb_stats->best_pme_stats.mv.as_fullmv.row;

    // MVs can be negative, so they are scaled by multiplication.
    x_search_center *= 1 << last_pyr_level;
    y_search_center *= 1 << last_pyr_level;

    const int zero_mv_sad = this_sb_stats->zero_mv_sad;
    // SAD without MV cost, for the raw-SAD thresholds abs_static_sad_thresh
    // and abs_pme_sad_thresh.
    const int best_pme_sad =
        this_sb_stats->best_pme_stats.cost -
        cand_mv_cost(mv_cost, this_sb_stats->best_pme_stats.mv.as_fullmv);
    int search_area_width;
    int search_area_height;
    get_post_pme_search_window(
        cpi, cur_lyr, ref_lyr, &x_search_center, &y_search_center,
        post_search_window, &search_area_width, &search_area_height, pos_x,
        pos_y, cur_disp_order_idx, ref_disp_order_idx[rf_idx], best_pme_sad,
        zero_mv_sad, rf_idx, mv_cost);

    // Setup frame buffer pointers.
    setup_pred_plane(cur_y, bsize, cur_lyr->buffer, cur_lyr->width,
                     cur_lyr->height, cur_lyr->stride, mi_row, mi_col, NULL,
                     xd->plane[AOM_PLANE_Y].subsampling_x,
                     xd->plane[AOM_PLANE_Y].subsampling_y);

    setup_pred_plane(ref_y, bsize, ref_lyr->buffer, ref_lyr->width,
                     ref_lyr->height, ref_lyr->stride, mi_row, mi_col, NULL,
                     xd->plane[AOM_PLANE_Y].subsampling_x,
                     xd->plane[AOM_PLANE_Y].subsampling_y);

    assert(cur_y->width == ref_y->width);
    assert(cur_y->height == ref_y->height);

    int_mv mv;
    if (var_8x8_enable && (search_area_width * search_area_height > 24)) {
      mv.as_fullmv =
          (FULLPEL_MV){ .row = y_search_center, .col = x_search_center };
      // 1x1 search window at the search center: its 8x8 SADs give the initial
      // 8x8 SAD variance, which resizes the search area.
      PmeMvCost no_mv_cost = get_no_mv_cost(mv_cost);
      sad_calc_sub_block_level(
          this_sb_stats, cur_y->buf, cur_lyr->stride,
          &ref_y->buf[y_search_center * ref_lyr->stride + x_search_center],
          ref_lyr->stride, mv, sad16x16_arr, /*search_index=*/1,
          /*skip_row=*/1, &no_mv_cost);

      scale_search_area_by_8x8_var(this_sb_stats, var_8x8_mult2_thresh,
                                   var_8x8_div4_thresh, var_8x8_div2_thresh,
                                   &search_area_width, &search_area_height);
    }
    const int aligned_frame_width = ALIGN_POWER_OF_TWO(cur_lyr->width, 3);
    const int aligned_frame_height = ALIGN_POWER_OF_TWO(cur_lyr->height, 3);
    int sa_origin_x = -(search_area_width >> 1) + x_search_center;
    int sa_origin_y = -(search_area_height >> 1) + y_search_center;
    clamp_search_area(pos_x, pos_y, block_size_wide[BLOCK_64X64] - 1,
                      block_size_high[BLOCK_64X64] - 1, aligned_frame_width,
                      aligned_frame_height, &sa_origin_x, &sa_origin_y,
                      &search_area_width, &search_area_height);
    uint8_t *const search_region =
        &ref_y->buf[sa_origin_y * ref_lyr->stride + sa_origin_x];

    // Each call evaluates up to 8 consecutive horizontal positions: search
    // the window in groups of 8, then the remainder.
    const int search_area_width_rest_8 = search_area_width & 7;
    const int search_area_width_mult_8 =
        search_area_width - search_area_width_rest_8;

    for (int y_search_index = 0; y_search_index < search_area_height;
         y_search_index++) {
      for (int x_search_index = 0; x_search_index < search_area_width;
           x_search_index += 8) {
        const int this_search_index = x_search_index < search_area_width_mult_8
                                          ? 8
                                          : search_area_width_rest_8;
        mv.as_fullmv.col = sa_origin_x + x_search_index;
        mv.as_fullmv.row = sa_origin_y + y_search_index;
        // Searches this range of up to 8 positions, and keeps the best MV of
        // each 8x8 to 64x64 block.
        sad_calc_sub_block_level(
            this_sb_stats, cur_y->buf, cur_lyr->stride,
            &search_region[y_search_index * ref_lyr->stride + x_search_index],
            ref_lyr->stride, mv, sad16x16_arr, this_search_index,
            /*skip_row=*/1, mv_cost);
      }
    }
  }
}

// Stores in best_pme_stats of each reference its best search zone of the last
// enabled pyramid level, with the MV at that level's precision. Returns the
// best of them over all reference frames, with the MV at full resolution.
static MvStats prepare_best_pme_stats(
    AV1_COMP *const cpi, ImagePyramid *ref_pyr[INTER_REFS_PER_FRAME],
    PmeSbStats (*pme_sb_stats)[INTER_REFS_PER_FRAME]) {
  PmeParams *const pme_data = &cpi->ppi->pme_data;
  const int layer_depth = cpi->ppi->gf_group.layer_depth[pme_data->frame_idx];
  const PmeFrameStats *const pme_frame =
      &pme_data->frame_stats_ptr[pme_data->frame_idx];
  const int *const ref_disp_order_idx = pme_data->ref_disp_order_idx;
  const int cur_disp_order_idx = pme_frame->display_index;

  const PYRAMID_LEVEL last_pyr_level =
      get_last_pyr_level(cpi->sf.pme_sf.enable_pyr_level);
  MvStats best_stats = { .cost = INT_MAX, .mv.as_int = 0 };

  for (int rf_idx = 0; rf_idx < INTER_REFS_PER_FRAME; ++rf_idx) {
    if (ref_pyr[rf_idx] == NULL) continue;

    PmeSbStats *const this_sb_stats = &(*pme_sb_stats)[rf_idx];
    MvStats *const best_zone_stats = &this_sb_stats->best_pme_stats;
    best_zone_stats->cost = INT_MAX;
    best_zone_stats->mv.as_int = 0;
    if (this_sb_stats->prune_ref) continue;

    if (skip_ref_for_layer_depth(layer_depth, cur_disp_order_idx,
                                 ref_disp_order_idx[rf_idx]))
      continue;

    for (int sz_idx = 0; sz_idx < PME_SEARCH_ZONES; ++sz_idx) {
      MvStats *const this_pme_stats =
          &this_sb_stats->pme_stats[last_pyr_level][sz_idx];
      if (this_pme_stats->cost < best_zone_stats->cost)
        *best_zone_stats = *this_pme_stats;
    }

    if (best_zone_stats->cost < best_stats.cost) best_stats = *best_zone_stats;
  }
  // Convert mv precision to full resolution.
  best_stats.mv.as_fullmv.col *= 1 << last_pyr_level;
  best_stats.mv.as_fullmv.row *= 1 << last_pyr_level;

  return best_stats;
}

// Runs the pyramid ME stages on each 64x64 block of the frame: zero MV SAD,
// pre pyramid ME, levels 2, 1 and 0, then post pyramid ME at full resolution
// around the best MV found.
static void pyramid_me(AV1_COMP *const cpi, const AV1_COMMON *const cm,
                       ImagePyramid *cur_pyr,
                       ImagePyramid *ref_pyr[INTER_REFS_PER_FRAME],
                       PmeSbStats (*pme_stats)[INTER_REFS_PER_FRAME],
                       int qindex) {
  PyramidLayer *const cur_lyr = cur_pyr->layers;

  const int num_sb_cols =
      (cm->mi_params.mi_cols + MI_SIZE_64X64 - 1) / MI_SIZE_64X64;
  const int num_sb_rows =
      (cm->mi_params.mi_rows + MI_SIZE_64X64 - 1) / MI_SIZE_64X64;
  PmeParams *const pme_data = &cpi->ppi->pme_data;
  const int aligned_frame_width =
      ALIGN_POWER_OF_TWO(cur_lyr[PYR_LEVEL0].width, 3);
  const int aligned_frame_height =
      ALIGN_POWER_OF_TWO(cur_lyr[PYR_LEVEL0].height, 3);
  int sb_idx = 0;
  const PYRAMID_ME_SPEED_FEATURES *const pme_sf = &cpi->sf.pme_sf;
  const bool enable_zero_sad = pme_sf->enable_zero_mv_sad;
  const bool enable_pre_pyr_me = pme_sf->enable_pre_pyr_me;
  const bool *const enable_pyr_me = pme_sf->enable_pyr_level;

  SearchWindowCfg lvl2_search_window;
  SearchWindowCfg post_search_window;
  get_search_windows_for_pme(pme_sf, qindex, &lvl2_search_window,
                             &post_search_window);

  PmeMvCost *const mv_cost = &pme_data->mv_cost;

  for (int sb_row = 0; sb_row < num_sb_rows; ++sb_row) {
    const int pos_y = sb_row * block_size_high[BLOCK_64X64];
    for (int sb_col = 0; sb_col < num_sb_cols; ++sb_col) {
      const int pos_x = sb_col * block_size_wide[BLOCK_64X64];
      // Reset sb stats.
      for (int rf_idx = 0; rf_idx < INTER_REFS_PER_FRAME; ++rf_idx) {
        PmeSbStats *const this_sb_stats = &(pme_stats[sb_idx])[rf_idx];
        this_sb_stats->zero_mv_sad = INT_MAX;
        this_sb_stats->prune_ref = false;
      }

      // Calculate aligned block size.
      const int blk_right_edge =
          AOMMIN(pos_x + block_size_wide[BLOCK_64X64], aligned_frame_width);
      const int blk_bottom_edge =
          AOMMIN(pos_y + block_size_high[BLOCK_64X64], aligned_frame_height);
      pme_data->aligned_bw = blk_right_edge - pos_x;
      pme_data->aligned_bh = blk_bottom_edge - pos_y;

      // Read by the level 2 search even when the pre pyramid ME is disabled.
      MvStats pre_pme_stats[INTER_REFS_PER_FRAME][PRE_PME_SEARCH_ZONES];
      for (int rf_idx = 0; rf_idx < INTER_REFS_PER_FRAME; ++rf_idx) {
        for (int sz_idx = 0; sz_idx < PRE_PME_SEARCH_ZONES; ++sz_idx) {
          pre_pme_stats[rf_idx][sz_idx].mv.as_int = 0;
          pre_pme_stats[rf_idx][sz_idx].cost = INT_MAX;
        }
      }

      if (enable_zero_sad) {
        // Calculate zero mv sad, level 0 - 64x64
        calc_zero_mv_sad(cpi, cur_pyr, ref_pyr, &pme_stats[sb_idx], pos_x,
                         pos_y);
      }

      if (enable_pre_pyr_me) {
        // Pre-pyramid ME, level 2 - 16x16
        pre_pyramid_me(cpi, cur_pyr, ref_pyr, &pre_pme_stats,
                       &pme_stats[sb_idx], pos_x, pos_y, mv_cost);
      }

      if (enable_pyr_me[PYR_LEVEL2]) {
        // Pyramid ME, level 2 - 16x16
        pyramid_me_level2(cpi, cur_pyr, ref_pyr, &pre_pme_stats,
                          &pme_stats[sb_idx], pos_x, pos_y, &lvl2_search_window,
                          mv_cost);
      }
      if (enable_pyr_me[PYR_LEVEL1]) {
        // Pyramid ME, level 1 - 32x32
        pyramid_me_level1(cpi, cur_pyr, ref_pyr, &pme_stats[sb_idx], pos_x,
                          pos_y, mv_cost);
      }
      if (enable_pyr_me[PYR_LEVEL0]) {
        // Pyramid ME, level 0 - 64x64
        pyramid_me_level0(cpi, cur_pyr, ref_pyr, &pme_stats[sb_idx], pos_x,
                          pos_y, mv_cost);
      }

      const MvStats best_pme_stats =
          prepare_best_pme_stats(cpi, ref_pyr, &pme_stats[sb_idx]);

      // Post-pyramid ME, level 0
      post_pyramid_me(cpi, cur_pyr, ref_pyr, &pme_stats[sb_idx],
                      best_pme_stats.cost, pos_x, pos_y, &post_search_window,
                      mv_cost);

      ++sb_idx;
    }  // sb_col
  }  // sb_row
}

// Returns the number of frames of the sub-GOP that are processed.
static inline int get_gop_length(const GF_GROUP *gf_group) {
  int gop_length = AOMMIN(gf_group->size, MAX_PME_FRAME_IDX - 1);
  return gop_length;
}

// Returns true if the frame is not searched: the first frame of a key frame
// sub-GOP, which has no reference frame, and the overlay frames.
static bool skip_frame_for_pme(const AV1_COMP *cpi,
                               const EncodeFrameParams *const frame_params,
                               int frame_idx) {
  const GF_GROUP *const gf_group = &cpi->ppi->gf_group;
  // The first frame of a key frame sub-GOP has no reference frame.
  if (frame_params->frame_type == KEY_FRAME && frame_idx == 0) return true;
  // The frames beyond the sub-GOP are leaf frames, which are always searched.
  if (frame_idx >= get_gop_length(gf_group)) return false;

  const FRAME_UPDATE_TYPE update_type = gf_group->update_type[frame_idx];
  return update_type == INTNL_OVERLAY_UPDATE || update_type == OVERLAY_UPDATE;
}

// Sets up the reference buffers, the frames of the sub-GOP and the frames
// beyond it, with the reference frames of each frame.
static void init_gop_frames_for_pme(
    AV1_COMP *cpi, const EncodeFrameParams *const init_frame_params) {
  const AV1_COMMON *const cm = &cpi->common;
  const GF_GROUP *const gf_group = &cpi->ppi->gf_group;
  PmeParams *const pme_data = &cpi->ppi->pme_data;
  EncodeFrameParams frame_params = *init_frame_params;

  RefFrameMapPair ref_frame_map_pairs[REF_FRAMES];
  init_ref_map_pair(cpi, ref_frame_map_pairs);
  int remapped_ref_idx[REF_FRAMES];
  int ref_picture_map[REF_FRAMES];
  int pframe_qindex = 0;
  int leaf_qindex = 0;
  int num_frames = 0;

  // Reference buffers.
  for (int i = 0; i < REF_FRAMES; ++i) {
    PmeFrameStats *const frame_stats = &pme_data->frame_stats_ptr[-i - 1];
    frame_stats->src = NULL;
    frame_stats->buf_id = NULL;
    frame_stats->src_pyr = NULL;
    frame_stats->display_index = 0;
    if (frame_params.frame_type != KEY_FRAME) {
      const RefCntBuffer *const ref_buf = cm->ref_frame_map[i];
      if ((int)ref_buf->display_order_hint ==
          pme_data->prev_gop_arf_disp_order) {
        // The pyramid is already computed, see
        // av1_compute_prev_gop_arf_pyramid().
        frame_stats->buf_id = &pme_data->prev_gop_arf_src_pyr;
        frame_stats->src_pyr = &pme_data->prev_gop_arf_src_pyr;
      } else {
        frame_stats->src = &ref_buf->buf;
        frame_stats->buf_id = ref_buf;
        frame_stats->src_pyr = &pme_data->src_pyr_pool[i];
      }
      frame_stats->display_index = ref_buf->display_order_hint;
    }
    ref_picture_map[i] = -i - 1;
  }

  // Frames of the sub-GOP. The overlay frames are not searched: they have no
  // PME stats, and their pyramid slot is shared with the next frame.
  const int gop_length = get_gop_length(gf_group);
  int process_frame_count = 0;
  int gf_index;
  for (gf_index = 0; gf_index < gop_length; ++gf_index) {
    PmeFrameStats *const frame_stats = &pme_data->frame_stats_ptr[gf_index];
    const FRAME_UPDATE_TYPE frame_update_type = gf_group->update_type[gf_index];
    const int lookahead_index =
        gf_group->cur_frame_idx[gf_index] + gf_group->arf_src_offset[gf_index];
    frame_params.show_frame = frame_update_type != ARF_UPDATE &&
                              frame_update_type != INTNL_ARF_UPDATE;
    frame_params.show_existing_frame =
        frame_update_type == INTNL_OVERLAY_UPDATE ||
        frame_update_type == OVERLAY_UPDATE;
    frame_params.frame_type = gf_group->frame_type[gf_index];

    if (frame_update_type == LF_UPDATE) {
      pframe_qindex = gf_group->q_val[gf_index];
      leaf_qindex = pframe_qindex;
      if (cpi->oxcf.rc_cfg.mode == AOM_VBR || cpi->oxcf.rc_cfg.mode == AOM_Q) {
        const int bit_depth = cm->seq_params->bit_depth;
        const double q_val = av1_convert_qindex_to_q(pframe_qindex, bit_depth);
        const double qindex_ratio = (double)pframe_qindex / MAXQ;
        const double qstep_ratio = 1.0 - (qindex_ratio * qindex_ratio * 0.5);
        const int delta_qindex =
            av1_compute_qdelta(&cpi->rc, q_val, q_val * qstep_ratio, bit_depth);
        pframe_qindex = CLIP(pframe_qindex + delta_qindex, MIN_PME_Q_INDEX,
                             MAX_PME_Q_INDEX);
      }
    }

    const struct lookahead_entry *buf = av1_lookahead_peek(
        cpi->ppi->lookahead, lookahead_index, cpi->compressor_stage);
    if (buf == NULL) break;
    // The temporally filtered frame if available, else the source frame.
    FRAME_DIFF frame_diff;
    const YV12_BUFFER_CONFIG *tf_buf =
        av1_tf_info_get_filtered_buf(&cpi->ppi->tf_info, gf_index, &frame_diff);
    frame_stats->src = tf_buf != NULL ? tf_buf : &buf->img;
    frame_stats->display_index =
        lookahead_index + cm->current_frame.frame_number;
    frame_stats->src_pyr =
        &pme_data->src_pyr_pool[REF_FRAMES + process_frame_count];
    frame_stats->buf_id = NULL;
    if (frame_update_type != OVERLAY_UPDATE &&
        frame_update_type != INTNL_OVERLAY_UPDATE) {
      frame_stats->buf_id = frame_stats;
      frame_stats->stats = pme_data->stats_pool[process_frame_count];
      ++process_frame_count;
    }

    const int true_disp = (int)frame_stats->display_index;
    av1_get_ref_frames(ref_frame_map_pairs, true_disp, cpi, gf_index, 0,
                       remapped_ref_idx);
    int refresh_mask =
        av1_get_refresh_frame_flags(cpi, &frame_params, frame_update_type,
                                    gf_index, true_disp, ref_frame_map_pairs);
    if (gf_group->is_frame_non_ref[gf_index]) refresh_mask = 0;
    const int refresh_frame_map_index =
        av1_get_refresh_ref_frame_map(refresh_mask);
    if (refresh_frame_map_index < REF_FRAMES &&
        refresh_frame_map_index != INVALID_IDX) {
      ref_frame_map_pairs[refresh_frame_map_index].disp_order =
          AOMMAX(0, true_disp);
      ref_frame_map_pairs[refresh_frame_map_index].pyr_level =
          get_true_pyr_level(gf_group->layer_depth[gf_index], true_disp,
                             gf_group->max_layer_depth);
    }
    for (int i = LAST_FRAME; i <= ALTREF_FRAME; ++i)
      frame_stats->ref_map_index[i - LAST_FRAME] =
          ref_picture_map[remapped_ref_idx[i - LAST_FRAME]];
    if (refresh_mask) ref_picture_map[refresh_frame_map_index] = gf_index;
    ++num_frames;
  }

  // Leaf frames beyond the sub-GOP.
  const int max_extend_frames =
      cpi->oxcf.gf_cfg.lag_in_frames - MAX_GF_INTERVAL;
  const int extend_frame_length =
      AOMMIN(max_extend_frames,
             cpi->rc.frames_to_key - cpi->ppi->p_rc.baseline_gf_interval);
  int extend_frame_count = 0;
  int frame_display_index = gf_group->cur_frame_idx[gop_length - 1] +
                            gf_group->arf_src_offset[gop_length - 1] + 1;
  for (;
       gf_index < MAX_PME_FRAME_IDX && extend_frame_count < extend_frame_length;
       ++gf_index) {
    PmeFrameStats *const frame_stats = &pme_data->frame_stats_ptr[gf_index];
    const FRAME_UPDATE_TYPE frame_update_type = LF_UPDATE;
    frame_params.show_frame = 1;
    frame_params.show_existing_frame = 0;
    frame_params.frame_type = INTER_FRAME;

    const struct lookahead_entry *buf = av1_lookahead_peek(
        cpi->ppi->lookahead, frame_display_index, cpi->compressor_stage);
    if (buf == NULL) break;
    frame_stats->src = &buf->img;
    frame_stats->buf_id = frame_stats;
    frame_stats->stats = pme_data->stats_pool[process_frame_count];
    frame_stats->src_pyr =
        &pme_data->src_pyr_pool[REF_FRAMES + process_frame_count];
    frame_stats->display_index =
        frame_display_index + cm->current_frame.frame_number;
    ++process_frame_count;

#if CONFIG_BITRATE_ACCURACY && CONFIG_THREE_PASS
    if (cpi->oxcf.pass == AOM_RC_SECOND_PASS) {
      if (cpi->oxcf.rc_cfg.mode == AOM_Q) {
        pframe_qindex = cpi->oxcf.rc_cfg.cq_level;
      } else if (cpi->oxcf.rc_cfg.mode == AOM_VBR) {
        pframe_qindex = 128;
      }
    }
#endif  // CONFIG_BITRATE_ACCURACY && CONFIG_THREE_PASS

    const int true_disp = (int)frame_stats->display_index;
    av1_get_ref_frames(ref_frame_map_pairs, true_disp, cpi, gf_index, 0,
                       remapped_ref_idx);
    const int refresh_mask =
        av1_get_refresh_frame_flags(cpi, &frame_params, frame_update_type,
                                    gf_index, true_disp, ref_frame_map_pairs);
    const int refresh_frame_map_index =
        av1_get_refresh_ref_frame_map(refresh_mask);
    if (refresh_frame_map_index < REF_FRAMES &&
        refresh_frame_map_index != INVALID_IDX) {
      ref_frame_map_pairs[refresh_frame_map_index].disp_order =
          AOMMAX(0, true_disp);
      ref_frame_map_pairs[refresh_frame_map_index].pyr_level =
          get_true_pyr_level(gf_group->layer_depth[gf_index], true_disp,
                             gf_group->max_layer_depth);
    }
    for (int i = LAST_FRAME; i <= ALTREF_FRAME; ++i)
      frame_stats->ref_map_index[i - LAST_FRAME] =
          ref_picture_map[remapped_ref_idx[i - LAST_FRAME]];
    frame_stats->ref_map_index[ALTREF_FRAME - LAST_FRAME] = -1;
    frame_stats->ref_map_index[LAST3_FRAME - LAST_FRAME] = -1;
    frame_stats->ref_map_index[BWDREF_FRAME - LAST_FRAME] = -1;
    frame_stats->ref_map_index[ALTREF2_FRAME - LAST_FRAME] = -1;
    if (refresh_mask) ref_picture_map[refresh_frame_map_index] = gf_index;

    ++num_frames;
    ++extend_frame_count;
    ++frame_display_index;
  }

  pme_data->num_frames = num_frames;
  pme_data->pframe_qindex = pframe_qindex;
  pme_data->leaf_qindex = leaf_qindex;
}

// Computes the source pyramids of the frames which are searched or
// referenced.
static void init_pyramids_for_pme(AV1_COMP *cpi,
                                  const EncodeFrameParams *const frame_params) {
  PmeParams *const pme_data = &cpi->ppi->pme_data;
  const int bit_depth = cpi->common.seq_params->bit_depth;
  bool src_pyr_needed_buffer[MAX_LENGTH_PME_FRAME_STATS] = { false };
  bool *const src_pyr_needed = &src_pyr_needed_buffer[REF_FRAMES + 1];

  for (int frame_idx = 0; frame_idx < pme_data->num_frames; ++frame_idx) {
    if (skip_frame_for_pme(cpi, frame_params, frame_idx)) continue;
    const PmeFrameStats *const frame_stats =
        &pme_data->frame_stats_ptr[frame_idx];
    src_pyr_needed[frame_idx] = true;
    for (int idx = 0; idx < INTER_REFS_PER_FRAME; ++idx)
      src_pyr_needed[frame_stats->ref_map_index[idx]] = true;
  }

  for (int frame_idx = -REF_FRAMES; frame_idx < pme_data->num_frames;
       ++frame_idx) {
    PmeFrameStats *const frame_stats = &pme_data->frame_stats_ptr[frame_idx];
    if (!src_pyr_needed[frame_idx] || frame_stats->src_pyr == NULL) continue;
    // The pyramid of the last frame of the previous sub-GOP is computed when
    // that frame is encoded.
    if (frame_stats->src_pyr == &pme_data->prev_gop_arf_src_pyr) continue;
    // A reference coded at another size (e.g. resize or superres enabled
    // earlier in the stream) is not searched.
    if (!is_pme_pyr_allocated_for_frame(frame_stats->src_pyr,
                                        frame_stats->src)) {
      frame_stats->src_pyr = NULL;
      continue;
    }
    // TODO(apurve.pandey): Optimize this pyramid computation
    frame_stats->src_pyr->filled_levels = 0;
    if (aom_compute_padded_pyramid(frame_stats->src, bit_depth, MAX_PYR_LEVELS,
                                   frame_stats->src_pyr, PYRAMID_ME_PADDING,
                                   PYRAMID_ME_RIGHT_SLACK,
                                   PYRAMID_ME_BOTTOM_SLACK) < 0)
      frame_stats->src_pyr = NULL;
  }
}

// Sets up the frame being processed: its reference frames, without the pruned
// ones, and the SAD multiplier of the MV rate. Returns its q index.
static int init_frame_for_pme(AV1_COMP *cpi, int frame_idx) {
  const GF_GROUP *const gf_group = &cpi->ppi->gf_group;
  PmeParams *const pme_data = &cpi->ppi->pme_data;
  const PmeFrameStats *const frame_stats =
      &pme_data->frame_stats_ptr[frame_idx];
  const int gop_length = get_gop_length(gf_group);
  const YV12_BUFFER_CONFIG *ref_frames[INTER_REFS_PER_FRAME];
  const YV12_BUFFER_CONFIG *ref_frames_ordered[INTER_REFS_PER_FRAME];
  uint32_t ref_frame_display_indices[INTER_REFS_PER_FRAME];

  pme_data->frame_idx = frame_idx;
  for (int idx = 0; idx < INTER_REFS_PER_FRAME; ++idx) {
    const PmeFrameStats *const ref =
        &pme_data->frame_stats_ptr[frame_stats->ref_map_index[idx]];
    ref_frames[idx] = ref->buf_id;
    ref_frame_display_indices[idx] = ref->display_index;
    pme_data->ref_src_pyr[idx] = ref->src_pyr;
    pme_data->ref_disp_order_idx[idx] = ref->display_index;
  }

  // Prune the reference frames by the reference frame speed features.
  for (int i = 0; i < INTER_REFS_PER_FRAME; ++i)
    ref_frames_ordered[i] = ref_frames[ref_frame_priority_order[i] - 1];
  int ref_frame_flags =
      get_ref_frame_flags(&cpi->sf, is_one_pass_rt_params(cpi),
                          ref_frames_ordered, cpi->ext_flags.ref_frame_flags);
  enforce_max_ref_frames(cpi, &ref_frame_flags, ref_frame_display_indices,
                         frame_stats->display_index);
  for (int idx = 0; idx < INTER_REFS_PER_FRAME; ++idx) {
    if ((ref_frame_flags & (1 << idx)) == 0) pme_data->ref_src_pyr[idx] = NULL;
  }
  if (frame_idx < gop_length) {
    for (int idx = 0; idx < INTER_REFS_PER_FRAME; ++idx) {
      const MV_REFERENCE_FRAME refs[2] = { idx + 1, NONE_FRAME };
      if (prune_ref_by_selective_ref_frame(cpi, NULL, refs,
                                           ref_frame_display_indices))
        pme_data->ref_src_pyr[idx] = NULL;
    }
  }

  // Q index of the frame.
  int base_qindex = pme_data->pframe_qindex;
  if (cpi->use_ducky_encode) {
    base_qindex = frame_idx < gop_length ? gf_group->q_val[frame_idx]
                                         : pme_data->leaf_qindex;
  }
  if (cpi->sf.pme_sf.enable_mv_cost)
    av1_set_sad_per_bit(cpi, &pme_data->mv_cost.params.sad_per_bit,
                        base_qindex);

  // Whether a frame of the sub-GOP references this frame.
  pme_data->is_frame_used_as_ref = false;
  const int num_gop_frames = AOMMIN(gop_length, pme_data->num_frames);
  for (int idx = 0; idx < num_gop_frames; ++idx) {
    for (int ref_idx = 0; ref_idx < INTER_REFS_PER_FRAME; ++ref_idx) {
      if (pme_data->frame_stats_ptr[idx].ref_map_index[ref_idx] == frame_idx) {
        pme_data->is_frame_used_as_ref = true;
        break;
      }
    }
    if (pme_data->is_frame_used_as_ref) break;
  }
  return base_qindex;
}

// Runs the pyramid motion estimation for all the frames of the sub-GOP, at the
// start of the sub-GOP:
// - estimates the q index of the frames of the sub-GOP,
// - sets up the sub-GOP frames, their reference frames and source pyramids,
// - sets up the MV cost context, once for the sub-GOP,
// - runs pyramid_me() on each frame that is not skipped, filling its PME stats.
// pme_stats_ready is set only if the stats of all those frames are filled;
// otherwise, e.g. when PME is not allowed, the sub-GOP is too long or a pyramid
// is missing, the stats must not be used.
void av1_pyramid_me_setup(AV1_COMP *cpi,
                          const EncodeFrameParams *const frame_params) {
  assert(cpi->gf_frame_index == 0);
  AV1_COMMON *const cm = &cpi->common;
  PmeParams *const pme_data = &cpi->ppi->pme_data;
  av1_init_pme_stats(pme_data);
  if (!av1_is_pme_allowed(cpi)) return;
  // The PME buffers are sized for at most MAX_LENGTH_PME_FRAME_STATS frames.
  if (cpi->ppi->gf_group.size > MAX_LENGTH_PME_FRAME_STATS) return;

  // Q index of the frames of the sub-GOP.
  av1_tpl_preload_rc_estimate(cpi, frame_params);

  init_gop_frames_for_pme(cpi, frame_params);
  init_pyramids_for_pme(cpi, frame_params);

  if (cpi->sf.pme_sf.enable_mv_cost) {
    // MV costs from the current MV probabilities.
    if (frame_params->frame_type == KEY_FRAME) av1_init_mv_probs(cm);
    av1_fill_mv_costs(&cm->fc->nmvc, cm->features.cur_frame_force_integer_mv,
                      cm->features.allow_high_precision_mv,
                      cpi->td.mb.mv_costs);
  }
  init_pme_mv_cost(cpi);

  bool stats_ready = true;
  for (int frame_idx = 0; frame_idx < pme_data->num_frames; ++frame_idx) {
    if (skip_frame_for_pme(cpi, frame_params, frame_idx)) continue;
    const PmeFrameStats *const frame_stats =
        &pme_data->frame_stats_ptr[frame_idx];
    // No pyramid if the frame size differs from the allocated one; the PME
    // stats of the sub-GOP are then not used.
    if (frame_stats->src_pyr == NULL) {
      stats_ready = false;
      break;
    }
    const int qindex = init_frame_for_pme(cpi, frame_idx);
    // TODO(apurve.pandey): Implement multithread support for pyramid_me()
    pyramid_me(cpi, cm, frame_stats->src_pyr, pme_data->ref_src_pyr,
               frame_stats->stats, qindex);
  }
  pme_data->pme_stats_ready = stats_ready;

  // The q index estimate above changes the frame type and show flag.
  cm->current_frame.frame_type = frame_params->frame_type;
  cm->show_frame = frame_params->show_frame;
}
