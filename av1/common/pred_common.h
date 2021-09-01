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

#ifndef AOM_AV1_COMMON_PRED_COMMON_H_
#define AOM_AV1_COMMON_PRED_COMMON_H_

#include "av1/common/av1_common_int.h"
#include "av1/common/blockd.h"
#include "aom_dsp/aom_dsp_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int pyr_level;
  int disp_order;
  int base_qindex;
} RefFrameMapPair;

static const MV_REFERENCE_FRAME
    ref_frame_priority_order[INTER_REFS_PER_FRAME] = {
      LAST_FRAME,    ALTREF_FRAME, BWDREF_FRAME, GOLDEN_FRAME,
      ALTREF2_FRAME, LAST2_FRAME,  LAST3_FRAME,
    };

static INLINE void init_ref_map_pair(
    AV1_COMMON *cm, RefFrameMapPair ref_frame_map_pairs[REF_FRAMES],
    int is_key) {
  if (is_key) {
    memset(ref_frame_map_pairs, -1, sizeof(*ref_frame_map_pairs) * REF_FRAMES);
    return;
  }
  memset(ref_frame_map_pairs, 0, sizeof(*ref_frame_map_pairs) * REF_FRAMES);
  for (int map_idx = 0; map_idx < REF_FRAMES; map_idx++) {
    // Get reference frame buffer
    const RefCntBuffer *const buf = cm->ref_frame_map[map_idx];
    if (ref_frame_map_pairs[map_idx].disp_order == -1) continue;
    if (buf == NULL) {
      ref_frame_map_pairs[map_idx].disp_order = -1;
      ref_frame_map_pairs[map_idx].pyr_level = -1;
      ref_frame_map_pairs[map_idx].base_qindex = -1;
      continue;
    } else if (buf->ref_count > 1) {
      // Once the keyframe is coded, the slots in ref_frame_map will all
      // point to the same frame. In that case, all subsequent pointers
      // matching the current are considered "free" slots. This will find
      // the next occurance of the current pointer if ref_count indicates
      // there are multiple instances of it and mark it as free.
      for (int idx2 = map_idx + 1; idx2 < REF_FRAMES; ++idx2) {
        const RefCntBuffer *const buf2 = cm->ref_frame_map[idx2];
        if (buf2 == buf) {
          ref_frame_map_pairs[idx2].disp_order = -1;
          ref_frame_map_pairs[idx2].pyr_level = -1;
          ref_frame_map_pairs[idx2].base_qindex = -1;
        }
      }
    }
    ref_frame_map_pairs[map_idx].disp_order = (int)buf->display_order_hint;
    ref_frame_map_pairs[map_idx].pyr_level = buf->pyramid_level;
    ref_frame_map_pairs[map_idx].base_qindex = buf->base_qindex;
  }
}

void av1_get_ref_frames(AV1_COMMON *const cm, int cur_frame_disp,
                        RefFrameMapPair ref_frame_map_pairs[REF_FRAMES]);

#if CONFIG_NEW_REF_SIGNALING
void av1_init_new_ref_frame_map(AV1_COMMON *const cm,
                                RefFrameMapPair *ref_frame_map_pairs,
                                int cur_frame_disp);

static INLINE int convert_named_ref_to_ranked_ref_index(
    const NewRefFramesData *const ref_frame_data, int named_idx) {
  // NONE_FRAME and INVALID_IDX are both equal to -1, simplify by using only
  // INVALID_IDX for this purpose
  if (named_idx == NONE_FRAME) return INVALID_IDX;
  if (named_idx == INVALID_IDX) return INVALID_IDX;
  if (named_idx == INTRA_FRAME) return INTRA_FRAME_NRS;
  return ref_frame_data->named_to_ranked_refs[named_idx];
}

static INLINE void convert_named_ref_to_ranked_ref_pair(
    const NewRefFramesData *const ref_frame_data, MV_REFERENCE_FRAME *rf,
    int ordered, MV_REFERENCE_FRAME_NRS *rf_nrs) {
  rf_nrs[0] = convert_named_ref_to_ranked_ref_index(ref_frame_data, rf[0]);
  rf_nrs[1] = convert_named_ref_to_ranked_ref_index(ref_frame_data, rf[1]);
  if (ordered) {
    if (rf_nrs[1] != INVALID_IDX && rf_nrs[1] < rf_nrs[0]) {
      MV_REFERENCE_FRAME_NRS tmp = rf_nrs[1];
      rf_nrs[1] = rf_nrs[0];
      rf_nrs[0] = tmp;
    }
  }
}

static INLINE int convert_ranked_ref_to_named_ref_index(
    const NewRefFramesData *const ref_frame_data, int ranked_idx) {
  if (ranked_idx == INVALID_IDX) return INVALID_IDX;
  if (ranked_idx == INTRA_FRAME_NRS) return INTRA_FRAME;
  return ref_frame_data->ranked_to_named_refs[ranked_idx];
}

// Find the reference that is furthest in the future
static INLINE int get_furthest_future_ref_index(const AV1_COMMON *const cm) {
  int index = INVALID_IDX;
  int ref_disp_order = -1;
  for (int i = 0; i < cm->new_ref_frame_data.n_future_refs; i++) {
    const int ref = cm->new_ref_frame_data.future_refs[i];
    const RefCntBuffer *const buf = get_ref_frame_buf_nrs(cm, ref);
    if (buf == NULL) continue;
    if ((int)buf->display_order_hint > ref_disp_order) {
      index = ref;
      ref_disp_order = (int)buf->display_order_hint;
    }
  }
  return index;
}

static INLINE int get_n_bidir_compound_modes_nrs(NewRefFramesData *ref_data) {
  return ref_data->n_past_refs * ref_data->n_future_refs;
}

static INLINE int get_n_past_unidir_compound_modes_nrs(
    NewRefFramesData *ref_data) {
  return ref_data->n_past_refs * (ref_data->n_past_refs - 1);
}

static INLINE int get_n_future_unidir_compound_modes_nrs(
    NewRefFramesData *ref_data) {
  return ref_data->n_future_refs * (ref_data->n_future_refs - 1);
}

static INLINE int get_n_unidir_compound_modes_nrs(NewRefFramesData *ref_data) {
  const int past_unidir = get_n_past_unidir_compound_modes_nrs(ref_data);
  const int future_unidir = get_n_future_unidir_compound_modes_nrs(ref_data);
  return past_unidir + future_unidir;
}

static INLINE int get_total_compound_modes_nrs(NewRefFramesData *ref_data) {
  const int unidir = get_n_unidir_compound_modes_nrs(ref_data);
  const int bidir = get_n_bidir_compound_modes_nrs(ref_data);
  return unidir + bidir;
}

// Gets directional i.e. past/future ref rank from overall rank
// in dir_refrank[0]/[1] respectively. Returns 0 if found in past
// list, 1 if found in future list, -1 if not found in either (error).
// Note dir_refrank can be NULL, in which case only the direction
// is returned, the ranks are not output.
static INLINE int get_dir_rank(const AV1_COMMON *const cm, int refrank,
                               int *dir_refrank) {
  assert(refrank != INTRA_FRAME_NRS && refrank != INVALID_IDX);
  assert(refrank < cm->new_ref_frame_data.n_total_refs);
  if (dir_refrank) {
    dir_refrank[0] = -1;
    dir_refrank[1] = -1;
  }
  for (int i = 0; i < cm->new_ref_frame_data.n_past_refs; ++i) {
    if (cm->new_ref_frame_data.past_refs[i] == refrank) {
      if (dir_refrank) dir_refrank[0] = i;
      return 0;
    }
  }
  for (int i = 0; i < cm->new_ref_frame_data.n_future_refs; ++i) {
    if (cm->new_ref_frame_data.future_refs[i] == refrank) {
      if (dir_refrank) dir_refrank[1] = i;
      return 1;
    }
  }
  // If refrank has the same distance as a reference return 0 (past)
  // but the dir_refrank[0] is -1
  if (cm->new_ref_frame_data.cur_ref == refrank) return 0;
  return -1;
}
#endif  // CONFIG_NEW_REF_SIGNALING

static INLINE int get_segment_id(const CommonModeInfoParams *const mi_params,
                                 const uint8_t *segment_ids, BLOCK_SIZE bsize,
                                 int mi_row, int mi_col) {
  const int mi_offset = mi_row * mi_params->mi_cols + mi_col;
  const int bw = mi_size_wide[bsize];
  const int bh = mi_size_high[bsize];
  const int xmis = AOMMIN(mi_params->mi_cols - mi_col, bw);
  const int ymis = AOMMIN(mi_params->mi_rows - mi_row, bh);
  int segment_id = MAX_SEGMENTS;

  for (int y = 0; y < ymis; ++y) {
    for (int x = 0; x < xmis; ++x) {
      segment_id = AOMMIN(segment_id,
                          segment_ids[mi_offset + y * mi_params->mi_cols + x]);
    }
  }

  assert(segment_id >= 0 && segment_id < MAX_SEGMENTS);
  return segment_id;
}

static INLINE int av1_get_spatial_seg_pred(const AV1_COMMON *const cm,
                                           const MACROBLOCKD *const xd,
                                           int *cdf_index) {
  int prev_ul = -1;  // top left segment_id
  int prev_l = -1;   // left segment_id
  int prev_u = -1;   // top segment_id
  const int mi_row = xd->mi_row;
  const int mi_col = xd->mi_col;
  const CommonModeInfoParams *const mi_params = &cm->mi_params;
  const uint8_t *seg_map = cm->cur_frame->seg_map;
  if ((xd->up_available) && (xd->left_available)) {
    prev_ul =
        get_segment_id(mi_params, seg_map, BLOCK_4X4, mi_row - 1, mi_col - 1);
  }
  if (xd->up_available) {
    prev_u =
        get_segment_id(mi_params, seg_map, BLOCK_4X4, mi_row - 1, mi_col - 0);
  }
  if (xd->left_available) {
    prev_l =
        get_segment_id(mi_params, seg_map, BLOCK_4X4, mi_row - 0, mi_col - 1);
  }
  // This property follows from the fact that get_segment_id() returns a
  // nonnegative value. This allows us to test for all edge cases with a simple
  // prev_ul < 0 check.
  assert(IMPLIES(prev_ul >= 0, prev_u >= 0 && prev_l >= 0));

  // Pick CDF index based on number of matching/out-of-bounds segment IDs.
  if (prev_ul < 0) /* Edge cases */
    *cdf_index = 0;
  else if ((prev_ul == prev_u) && (prev_ul == prev_l))
    *cdf_index = 2;
  else if ((prev_ul == prev_u) || (prev_ul == prev_l) || (prev_u == prev_l))
    *cdf_index = 1;
  else
    *cdf_index = 0;

  // If 2 or more are identical returns that as predictor, otherwise prev_l.
  if (prev_u == -1)  // edge case
    return prev_l == -1 ? 0 : prev_l;
  if (prev_l == -1)  // edge case
    return prev_u;
  return (prev_ul == prev_u) ? prev_u : prev_l;
}

static INLINE int av1_get_pred_context_seg_id(const MACROBLOCKD *xd) {
  const MB_MODE_INFO *const above_mi = xd->above_mbmi;
  const MB_MODE_INFO *const left_mi = xd->left_mbmi;
  const int above_sip = (above_mi != NULL) ? above_mi->seg_id_predicted : 0;
  const int left_sip = (left_mi != NULL) ? left_mi->seg_id_predicted : 0;

  return above_sip + left_sip;
}

#if !CONFIG_REMOVE_DIST_WTD_COMP
static INLINE int get_comp_index_context(const AV1_COMMON *cm,
                                         const MACROBLOCKD *xd) {
  MB_MODE_INFO *mbmi = xd->mi[0];
  const RefCntBuffer *const bck_buf = get_ref_frame_buf(cm, mbmi->ref_frame[0]);
  const RefCntBuffer *const fwd_buf = get_ref_frame_buf(cm, mbmi->ref_frame[1]);
  int bck_frame_index = 0, fwd_frame_index = 0;
  int cur_frame_index = cm->cur_frame->order_hint;

  if (bck_buf != NULL) bck_frame_index = bck_buf->order_hint;
  if (fwd_buf != NULL) fwd_frame_index = fwd_buf->order_hint;

  int fwd = abs(get_relative_dist(&cm->seq_params.order_hint_info,
                                  fwd_frame_index, cur_frame_index));
  int bck = abs(get_relative_dist(&cm->seq_params.order_hint_info,
                                  cur_frame_index, bck_frame_index));

  const MB_MODE_INFO *const above_mi = xd->above_mbmi;
  const MB_MODE_INFO *const left_mi = xd->left_mbmi;

  int above_ctx = 0, left_ctx = 0;
  const int offset = (fwd == bck);

  if (above_mi != NULL) {
    if (has_second_ref(above_mi))
      above_ctx = above_mi->compound_idx;
    else if (above_mi->ref_frame[0] == ALTREF_FRAME)
      above_ctx = 1;
  }

  if (left_mi != NULL) {
    if (has_second_ref(left_mi))
      left_ctx = left_mi->compound_idx;
    else if (left_mi->ref_frame[0] == ALTREF_FRAME)
      left_ctx = 1;
  }

  return above_ctx + left_ctx + 3 * offset;
}

static INLINE int get_comp_group_idx_context(const AV1_COMMON *cm,
                                             const MACROBLOCKD *xd) {
  (void)cm;
  const MB_MODE_INFO *const above_mi = xd->above_mbmi;
  const MB_MODE_INFO *const left_mi = xd->left_mbmi;
  int above_ctx = 0, left_ctx = 0;

  if (above_mi) {
    if (has_second_ref(above_mi))
      above_ctx = above_mi->comp_group_idx;
    else if (above_mi->ref_frame[0] == ALTREF_FRAME)
      above_ctx = 3;
  }
  if (left_mi) {
    if (has_second_ref(left_mi))
      left_ctx = left_mi->comp_group_idx;
    else if (left_mi->ref_frame[0] == ALTREF_FRAME)
      left_ctx = 3;
  }

  return AOMMIN(5, above_ctx + left_ctx);
}
#else
static INLINE int get_comp_group_idx_context(const AV1_COMMON *cm,
                                             const MACROBLOCKD *xd) {
  (void)cm;
  MB_MODE_INFO *mbmi = xd->mi[0];
  const RefCntBuffer *const bck_buf = get_ref_frame_buf(cm, mbmi->ref_frame[0]);
  const RefCntBuffer *const fwd_buf = get_ref_frame_buf(cm, mbmi->ref_frame[1]);
  int bck_frame_index = 0, fwd_frame_index = 0;
  int cur_frame_index = cm->cur_frame->order_hint;

  if (bck_buf != NULL) bck_frame_index = bck_buf->order_hint;
  if (fwd_buf != NULL) fwd_frame_index = fwd_buf->order_hint;

  int fwd = abs(get_relative_dist(&cm->seq_params.order_hint_info,
                                  fwd_frame_index, cur_frame_index));
  int bck = abs(get_relative_dist(&cm->seq_params.order_hint_info,
                                  cur_frame_index, bck_frame_index));
  const int offset = (fwd == bck);

  const MB_MODE_INFO *const above_mi = xd->above_mbmi;
  const MB_MODE_INFO *const left_mi = xd->left_mbmi;
  int above_ctx = 0, left_ctx = 0;

  if (above_mi) {
    if (has_second_ref(above_mi))
      above_ctx = above_mi->comp_group_idx;
    else if (above_mi->ref_frame[0] == ALTREF_FRAME)
      above_ctx = 2;
  }
  if (left_mi) {
    if (has_second_ref(left_mi))
      left_ctx = left_mi->comp_group_idx;
    else if (left_mi->ref_frame[0] == ALTREF_FRAME)
      left_ctx = 2;
  }
  const int ctxmap[3 * 3] = { 0, 1, 2, 1, 3, 4, 2, 4, 5 };

  return ctxmap[3 * above_ctx + left_ctx] + offset * 6;
}
#endif  // !CONFIG_REMOVE_DIST_WTD_COMP

static INLINE aom_cdf_prob *av1_get_pred_cdf_seg_id(
    struct segmentation_probs *segp, const MACROBLOCKD *xd) {
  return segp->pred_cdf[av1_get_pred_context_seg_id(xd)];
}

static INLINE int av1_get_skip_mode_context(const MACROBLOCKD *xd) {
  const MB_MODE_INFO *const above_mi = xd->above_mbmi;
  const MB_MODE_INFO *const left_mi = xd->left_mbmi;
  const int above_skip_mode = above_mi ? above_mi->skip_mode : 0;
  const int left_skip_mode = left_mi ? left_mi->skip_mode : 0;
  return above_skip_mode + left_skip_mode;
}

static INLINE int av1_get_skip_txfm_context(const MACROBLOCKD *xd) {
  const MB_MODE_INFO *const above_mi = xd->above_mbmi;
  const MB_MODE_INFO *const left_mi = xd->left_mbmi;
  const int above_skip_txfm = above_mi ? above_mi->skip_txfm : 0;
  const int left_skip_txfm = left_mi ? left_mi->skip_txfm : 0;
  return above_skip_txfm + left_skip_txfm;
}

int av1_get_pred_context_switchable_interp(const MACROBLOCKD *xd, int dir);

// Get a list of palette base colors that are used in the above and left blocks,
// referred to as "color cache". The return value is the number of colors in the
// cache (<= 2 * PALETTE_MAX_SIZE). The color values are stored in "cache"
// in ascending order.
int av1_get_palette_cache(const MACROBLOCKD *const xd, int plane,
                          uint16_t *cache);

static INLINE int av1_get_palette_bsize_ctx(BLOCK_SIZE bsize) {
  assert(bsize < BLOCK_SIZES_ALL);
  return num_pels_log2_lookup[bsize] - num_pels_log2_lookup[BLOCK_8X8];
}

static INLINE int av1_get_palette_mode_ctx(const MACROBLOCKD *xd) {
  const MB_MODE_INFO *const above_mi = xd->above_mbmi;
  const MB_MODE_INFO *const left_mi = xd->left_mbmi;
  int ctx = 0;
  if (above_mi) ctx += (above_mi->palette_mode_info.palette_size[0] > 0);
  if (left_mi) ctx += (left_mi->palette_mode_info.palette_size[0] > 0);
  return ctx;
}

int av1_get_intra_inter_context(const MACROBLOCKD *xd);

int av1_get_reference_mode_context(const MACROBLOCKD *xd);

static INLINE aom_cdf_prob *av1_get_reference_mode_cdf(const MACROBLOCKD *xd) {
  return xd->tile_ctx->comp_inter_cdf[av1_get_reference_mode_context(xd)];
}

static INLINE aom_cdf_prob *av1_get_skip_txfm_cdf(const MACROBLOCKD *xd) {
  return xd->tile_ctx->skip_txfm_cdfs[av1_get_skip_txfm_context(xd)];
}

int av1_get_comp_reference_type_context(const MACROBLOCKD *xd);

// == Uni-directional contexts ==

int av1_get_pred_context_uni_comp_ref_p(const MACROBLOCKD *xd);

int av1_get_pred_context_uni_comp_ref_p1(const MACROBLOCKD *xd);

int av1_get_pred_context_uni_comp_ref_p2(const MACROBLOCKD *xd);

static INLINE aom_cdf_prob *av1_get_comp_reference_type_cdf(
    const MACROBLOCKD *xd) {
  const int pred_context = av1_get_comp_reference_type_context(xd);
  return xd->tile_ctx->comp_ref_type_cdf[pred_context];
}

static INLINE aom_cdf_prob *av1_get_pred_cdf_uni_comp_ref_p(
    const MACROBLOCKD *xd) {
  const int pred_context = av1_get_pred_context_uni_comp_ref_p(xd);
  return xd->tile_ctx->uni_comp_ref_cdf[pred_context][0];
}

static INLINE aom_cdf_prob *av1_get_pred_cdf_uni_comp_ref_p1(
    const MACROBLOCKD *xd) {
  const int pred_context = av1_get_pred_context_uni_comp_ref_p1(xd);
  return xd->tile_ctx->uni_comp_ref_cdf[pred_context][1];
}

static INLINE aom_cdf_prob *av1_get_pred_cdf_uni_comp_ref_p2(
    const MACROBLOCKD *xd) {
  const int pred_context = av1_get_pred_context_uni_comp_ref_p2(xd);
  return xd->tile_ctx->uni_comp_ref_cdf[pred_context][2];
}

// == Bi-directional contexts ==

int av1_get_pred_context_comp_ref_p(const MACROBLOCKD *xd);

int av1_get_pred_context_comp_ref_p1(const MACROBLOCKD *xd);

int av1_get_pred_context_comp_ref_p2(const MACROBLOCKD *xd);

int av1_get_pred_context_comp_bwdref_p(const MACROBLOCKD *xd);

int av1_get_pred_context_comp_bwdref_p1(const MACROBLOCKD *xd);

static INLINE aom_cdf_prob *av1_get_pred_cdf_comp_ref_p(const MACROBLOCKD *xd) {
  const int pred_context = av1_get_pred_context_comp_ref_p(xd);
  return xd->tile_ctx->comp_ref_cdf[pred_context][0];
}

static INLINE aom_cdf_prob *av1_get_pred_cdf_comp_ref_p1(
    const MACROBLOCKD *xd) {
  const int pred_context = av1_get_pred_context_comp_ref_p1(xd);
  return xd->tile_ctx->comp_ref_cdf[pred_context][1];
}

static INLINE aom_cdf_prob *av1_get_pred_cdf_comp_ref_p2(
    const MACROBLOCKD *xd) {
  const int pred_context = av1_get_pred_context_comp_ref_p2(xd);
  return xd->tile_ctx->comp_ref_cdf[pred_context][2];
}

static INLINE aom_cdf_prob *av1_get_pred_cdf_comp_bwdref_p(
    const MACROBLOCKD *xd) {
  const int pred_context = av1_get_pred_context_comp_bwdref_p(xd);
  return xd->tile_ctx->comp_bwdref_cdf[pred_context][0];
}

static INLINE aom_cdf_prob *av1_get_pred_cdf_comp_bwdref_p1(
    const MACROBLOCKD *xd) {
  const int pred_context = av1_get_pred_context_comp_bwdref_p1(xd);
  return xd->tile_ctx->comp_bwdref_cdf[pred_context][1];
}

// == Single contexts ==

int av1_get_pred_context_single_ref_p1(const MACROBLOCKD *xd);

int av1_get_pred_context_single_ref_p2(const MACROBLOCKD *xd);

int av1_get_pred_context_single_ref_p3(const MACROBLOCKD *xd);

int av1_get_pred_context_single_ref_p4(const MACROBLOCKD *xd);

int av1_get_pred_context_single_ref_p5(const MACROBLOCKD *xd);

int av1_get_pred_context_single_ref_p6(const MACROBLOCKD *xd);

static INLINE aom_cdf_prob *av1_get_pred_cdf_single_ref_p1(
    const MACROBLOCKD *xd) {
  return xd->tile_ctx
      ->single_ref_cdf[av1_get_pred_context_single_ref_p1(xd)][0];
}
static INLINE aom_cdf_prob *av1_get_pred_cdf_single_ref_p2(
    const MACROBLOCKD *xd) {
  return xd->tile_ctx
      ->single_ref_cdf[av1_get_pred_context_single_ref_p2(xd)][1];
}
static INLINE aom_cdf_prob *av1_get_pred_cdf_single_ref_p3(
    const MACROBLOCKD *xd) {
  return xd->tile_ctx
      ->single_ref_cdf[av1_get_pred_context_single_ref_p3(xd)][2];
}
static INLINE aom_cdf_prob *av1_get_pred_cdf_single_ref_p4(
    const MACROBLOCKD *xd) {
  return xd->tile_ctx
      ->single_ref_cdf[av1_get_pred_context_single_ref_p4(xd)][3];
}
static INLINE aom_cdf_prob *av1_get_pred_cdf_single_ref_p5(
    const MACROBLOCKD *xd) {
  return xd->tile_ctx
      ->single_ref_cdf[av1_get_pred_context_single_ref_p5(xd)][4];
}
static INLINE aom_cdf_prob *av1_get_pred_cdf_single_ref_p6(
    const MACROBLOCKD *xd) {
  return xd->tile_ctx
      ->single_ref_cdf[av1_get_pred_context_single_ref_p6(xd)][5];
}

// Returns a context number for the given MB prediction signal
// The mode info data structure has a one element border above and to the
// left of the entries corresponding to real blocks.
// The prediction flags in these dummy entries are initialized to 0.
static INLINE int get_tx_size_context(const MACROBLOCKD *xd) {
  const MB_MODE_INFO *mbmi = xd->mi[0];
  const MB_MODE_INFO *const above_mbmi = xd->above_mbmi;
  const MB_MODE_INFO *const left_mbmi = xd->left_mbmi;
  const TX_SIZE max_tx_size = max_txsize_rect_lookup[mbmi->sb_type];
  const int max_tx_wide = tx_size_wide[max_tx_size];
  const int max_tx_high = tx_size_high[max_tx_size];
  const int has_above = xd->up_available;
  const int has_left = xd->left_available;

  int above = xd->above_txfm_context[0] >= max_tx_wide;
  int left = xd->left_txfm_context[0] >= max_tx_high;

  if (has_above)
    if (is_inter_block(above_mbmi))
      above = block_size_wide[above_mbmi->sb_type] >= max_tx_wide;

  if (has_left)
    if (is_inter_block(left_mbmi))
      left = block_size_high[left_mbmi->sb_type] >= max_tx_high;

  if (has_above && has_left)
    return (above + left);
  else if (has_above)
    return above;
  else if (has_left)
    return left;
  else
    return 0;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_PRED_COMMON_H_
