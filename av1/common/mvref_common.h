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

#ifndef AOM_AV1_COMMON_MVREF_COMMON_H_
#define AOM_AV1_COMMON_MVREF_COMMON_H_

#include "av1/common/av1_common_int.h"
#include "av1/common/blockd.h"
#include "av1/common/mv.h"
#include "av1/common/pred_common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define USE_NEW_REF_SIGNALING 0

#define MVREF_ROW_COLS 3

// Set the upper limit of the motion vector component magnitude.
// This would make a motion vector fit in 26 bits. Plus 3 bits for the
// reference frame index. A tuple of motion vector can hence be stored within
// 32 bit range for efficient load/store operations.
#define REFMVS_LIMIT ((1 << 12) - 1)

typedef struct position {
  int row;
  int col;
} POSITION;

// clamp_mv_ref
#define MV_BORDER (16 << 3)  // Allow 16 pels in 1/8th pel units

static INLINE void clamp_mv_ref(MV *mv, int bw, int bh, const MACROBLOCKD *xd) {
  const SubpelMvLimits mv_limits = {
    xd->mb_to_left_edge - GET_MV_SUBPEL(bw) - MV_BORDER,
    xd->mb_to_right_edge + GET_MV_SUBPEL(bw) + MV_BORDER,
    xd->mb_to_top_edge - GET_MV_SUBPEL(bh) - MV_BORDER,
    xd->mb_to_bottom_edge + GET_MV_SUBPEL(bh) + MV_BORDER
  };
  clamp_mv(mv, &mv_limits);
}

static INLINE int_mv get_block_mv(const MB_MODE_INFO *candidate, int which_mv) {
  return candidate->mv[which_mv];
}

// Checks that the given mi_row, mi_col and search point
// are inside the borders of the tile.
static INLINE int is_inside(const TileInfo *const tile, int mi_col, int mi_row,
                            const POSITION *mi_pos) {
  return !(mi_row + mi_pos->row < tile->mi_row_start ||
           mi_col + mi_pos->col < tile->mi_col_start ||
           mi_row + mi_pos->row >= tile->mi_row_end ||
           mi_col + mi_pos->col >= tile->mi_col_end);
}

static INLINE int find_valid_row_offset(const TileInfo *const tile, int mi_row,
                                        int row_offset) {
  return clamp(row_offset, tile->mi_row_start - mi_row,
               tile->mi_row_end - mi_row - 1);
}

static INLINE int find_valid_col_offset(const TileInfo *const tile, int mi_col,
                                        int col_offset) {
  return clamp(col_offset, tile->mi_col_start - mi_col,
               tile->mi_col_end - mi_col - 1);
}

static INLINE int8_t get_uni_comp_ref_idx(const MV_REFERENCE_FRAME *const rf) {
  // Single ref pred
  if (rf[1] <= INTRA_FRAME) return -1;

  // Bi-directional comp ref pred
  if ((rf[0] < BWDREF_FRAME) && (rf[1] >= BWDREF_FRAME)) return -1;

  for (int8_t ref_idx = 0; ref_idx < TOTAL_UNIDIR_COMP_REFS; ++ref_idx) {
    if (rf[0] == comp_ref0(ref_idx) && rf[1] == comp_ref1(ref_idx))
      return ref_idx;
  }
  return -1;
}

static INLINE int8_t av1_ref_frame_type(const MV_REFERENCE_FRAME *const rf) {
  if (rf[1] > INTRA_FRAME) {
    const int8_t uni_comp_ref_idx = get_uni_comp_ref_idx(rf);
    if (uni_comp_ref_idx >= 0) {
      assert((REF_FRAMES + FWD_REFS * BWD_REFS + uni_comp_ref_idx) <
             MODE_CTX_REF_FRAMES);
      return REF_FRAMES + FWD_REFS * BWD_REFS + uni_comp_ref_idx;
    } else {
      return REF_FRAMES + FWD_RF_OFFSET(rf[0]) +
             BWD_RF_OFFSET(rf[1]) * FWD_REFS;
    }
  }

  return rf[0];
}

// clang-format off
static MV_REFERENCE_FRAME ref_frame_map[TOTAL_COMP_REFS][2] = {
  { LAST_FRAME, BWDREF_FRAME },  { LAST2_FRAME, BWDREF_FRAME },
  { LAST3_FRAME, BWDREF_FRAME }, { GOLDEN_FRAME, BWDREF_FRAME },

  { LAST_FRAME, ALTREF2_FRAME },  { LAST2_FRAME, ALTREF2_FRAME },
  { LAST3_FRAME, ALTREF2_FRAME }, { GOLDEN_FRAME, ALTREF2_FRAME },

  { LAST_FRAME, ALTREF_FRAME },  { LAST2_FRAME, ALTREF_FRAME },
  { LAST3_FRAME, ALTREF_FRAME }, { GOLDEN_FRAME, ALTREF_FRAME },

  { LAST_FRAME, LAST2_FRAME }, { LAST_FRAME, LAST3_FRAME },
  { LAST_FRAME, GOLDEN_FRAME }, { BWDREF_FRAME, ALTREF_FRAME },

  // NOTE: Following reference frame pairs are not supported to be explicitly
  //       signalled, but they are possibly chosen by the use of skip_mode,
  //       which may use the most recent one-sided reference frame pair.
  { LAST2_FRAME, LAST3_FRAME }, { LAST2_FRAME, GOLDEN_FRAME },
  { LAST3_FRAME, GOLDEN_FRAME }, {BWDREF_FRAME, ALTREF2_FRAME},
  { ALTREF2_FRAME, ALTREF_FRAME }
};
// clang-format on

static INLINE void av1_set_ref_frame(MV_REFERENCE_FRAME *rf,
                                     MV_REFERENCE_FRAME ref_frame_type) {
  if (ref_frame_type >= REF_FRAMES) {
    rf[0] = ref_frame_map[ref_frame_type - REF_FRAMES][0];
    rf[1] = ref_frame_map[ref_frame_type - REF_FRAMES][1];
  } else {
    assert(ref_frame_type > NONE_FRAME);
    rf[0] = ref_frame_type;
    rf[1] = NONE_FRAME;
  }
}

#if CONFIG_NEW_REF_SIGNALING
// Converts a pair of distinct indices (rf) each in [0, n-1],
// to an combined index in [0, n*(n-1)/2].
// The order of the combined index is as follows:
// (0, 1), (0, 2), (0, 3), ..., (0, n-1),
//         (1, 2), (1, 3), ..., (1, n-1),
//                 (2, 3), ..., (2, n-1),
//                         ...
//                              (n-2, n-1)
static INLINE int8_t single2comb(int n, const int8_t *const rf) {
  assert(rf[0] < n && rf[1] < n);
  int8_t rfr[2] = { rf[0], rf[1] };
  if (rf[1] < rf[0]) {
    rfr[0] = rf[1];
    rfr[1] = rf[0];
  }
  int off = n * rfr[0] - rfr[0] * (rfr[0] + 1) / 2;
  int combindex = off + rfr[1] - rfr[0] - 1;
  return combindex;
}

// Converts a combined index in [0, n*(n-1)/2] to a pair of single
// ref indices (rf) each in [0, n-1]. See comment above for order
// of the combined indexing.
static INLINE void comb2single(int n, int8_t combindex, int8_t *rf) {
  assert(combindex < n * (n - 1) / 2);
  int i = n - 1, j = n - 1;
  rf[0] = 0;
  // Starting form n-1, keep reducing the row length by 1 until
  // combindex < i
  while (i <= combindex) {
    rf[0]++;
    j--;
    i += j;
  }
  rf[1] = combindex - i + j + rf[0] + 1;
  assert(rf[1] > rf[0]);
}

static INLINE int8_t
av1_ref_frame_type_nrs(const MV_REFERENCE_FRAME_NRS *const rf) {
  if (rf[0] == INTRA_FRAME_NRS || rf[0] == INVALID_IDX) {
    // Intra or invalid
    return rf[0];
  } else if (rf[1] == INTRA_FRAME_NRS || rf[1] == INVALID_IDX) {
    // single ref
    return rf[0];
  } else {
    // compound ref
    assert(rf[0] < INTER_REFS_PER_FRAME_NRS);
    assert(rf[1] < INTER_REFS_PER_FRAME_NRS);
    return single2comb(INTER_REFS_PER_FRAME_NRS, rf) + INTER_REFS_PER_FRAME_NRS;
  }
}

static INLINE void av1_set_ref_frame_nrs(
    MV_REFERENCE_FRAME_NRS *rf, MV_REFERENCE_FRAME_NRS ref_frame_type) {
  if (ref_frame_type == INTRA_FRAME_NRS ||
      ref_frame_type < INTER_REFS_PER_FRAME_NRS) {
    rf[0] = ref_frame_type;
    rf[1] = INVALID_IDX;
  } else {
    comb2single(INTER_REFS_PER_FRAME_NRS,
                ref_frame_type - INTER_REFS_PER_FRAME_NRS, rf);
  }
  return;
}

static INLINE MV_REFERENCE_FRAME convert_ranked_ref_to_named_ref_type(
    const NewRefFramesData *const ref_frame_data,
    MV_REFERENCE_FRAME_NRS ref_type_nrs) {
  MV_REFERENCE_FRAME_NRS rf_nrs[2];
  av1_set_ref_frame_nrs(rf_nrs, ref_type_nrs);
  MV_REFERENCE_FRAME rf[2];
  convert_ranked_ref_to_named_ref_pair(ref_frame_data, rf_nrs, 0, rf);
  return av1_ref_frame_type(rf);
}

static INLINE MV_REFERENCE_FRAME_NRS convert_named_ref_to_ranked_ref_type(
    const NewRefFramesData *const ref_frame_data, MV_REFERENCE_FRAME ref_type) {
  MV_REFERENCE_FRAME rf[2];
  av1_set_ref_frame(rf, ref_type);
  MV_REFERENCE_FRAME_NRS rf_nrs[2];
  convert_named_ref_to_ranked_ref_pair(ref_frame_data, rf, 0, rf_nrs);
  return av1_ref_frame_type_nrs(rf_nrs);
}
#endif  // CONFIG_NEW_REF_SIGNALING

static uint16_t compound_mode_ctx_map[3][COMP_NEWMV_CTXS] = {
  { 0, 1, 1, 1, 1 },
  { 1, 2, 3, 4, 4 },
  { 4, 4, 5, 6, 7 },
};

static INLINE int16_t av1_mode_context_pristine(
    const int16_t *const mode_context, const MV_REFERENCE_FRAME *const rf) {
  const int8_t ref_frame = av1_ref_frame_type(rf);
  return mode_context[ref_frame];
}

static INLINE int16_t av1_mode_context_analyzer(
    const int16_t *const mode_context, const MV_REFERENCE_FRAME *const rf) {
  const int8_t ref_frame = av1_ref_frame_type(rf);

  if (rf[1] <= INTRA_FRAME) return mode_context[ref_frame];

  const int16_t newmv_ctx = mode_context[ref_frame] & NEWMV_CTX_MASK;
  const int16_t refmv_ctx =
      (mode_context[ref_frame] >> REFMV_OFFSET) & REFMV_CTX_MASK;

  const int16_t comp_ctx = compound_mode_ctx_map[refmv_ctx >> 1][AOMMIN(
      newmv_ctx, COMP_NEWMV_CTXS - 1)];
  return comp_ctx;
}

#if CONFIG_NEW_INTER_MODES
static INLINE uint8_t av1_drl_ctx(const uint16_t *ref_mv_weight,
                                  const int16_t mode_ctx, int ref_idx) {
#if NO_MV_PARSING_DEP
  (void)ref_mv_weight;
  (void)ref_idx;
  return drl_mode_ctx(mode_ctx);
#else
  (void)mode_ctx;
  if (ref_mv_weight[ref_idx] >= REF_CAT_LEVEL &&
      ref_mv_weight[ref_idx + 1] >= REF_CAT_LEVEL)
    return 0;

  if (ref_mv_weight[ref_idx] >= REF_CAT_LEVEL &&
      ref_mv_weight[ref_idx + 1] < REF_CAT_LEVEL)
    return 1;

  if (ref_mv_weight[ref_idx] < REF_CAT_LEVEL &&
      ref_mv_weight[ref_idx + 1] < REF_CAT_LEVEL)
    return 2;
  return 0;
#endif  // NO_MV_PARSING_DEP
}

static INLINE aom_cdf_prob *av1_get_drl_cdf(FRAME_CONTEXT *ec_ctx,
                                            const uint16_t *ref_mv_weight,
                                            const int16_t mode_ctx,
                                            int ref_idx) {
  const int ctx = av1_drl_ctx(ref_mv_weight, mode_ctx, ref_idx);
  switch (ref_idx) {
    case 0: return ec_ctx->drl0_cdf[ctx];
    case 1: return ec_ctx->drl1_cdf[ctx];
    default: return ec_ctx->drl2_cdf[ctx];
  }
}

static INLINE int av1_drl_range(int num_ref_mvs, int max_est_ref_mvs,
                                int max_drl_bits) {
  (void)num_ref_mvs;
  (void)max_est_ref_mvs;
  (void)max_drl_bits;
#if NO_MV_PARSING_DEP
  return AOMMIN(max_est_ref_mvs - 1, max_drl_bits);
#else
  return AOMMIN(num_ref_mvs - 1, max_drl_bits);
#endif  // NO_MV_PARSING_DEP
}

#else

static INLINE uint8_t av1_drl_ctx(const uint16_t *ref_mv_weight, int ref_idx) {
  if (ref_mv_weight[ref_idx] >= REF_CAT_LEVEL &&
      ref_mv_weight[ref_idx + 1] >= REF_CAT_LEVEL)
    return 0;

  if (ref_mv_weight[ref_idx] >= REF_CAT_LEVEL &&
      ref_mv_weight[ref_idx + 1] < REF_CAT_LEVEL)
    return 1;

  if (ref_mv_weight[ref_idx] < REF_CAT_LEVEL &&
      ref_mv_weight[ref_idx + 1] < REF_CAT_LEVEL)
    return 2;

  return 0;
}
#endif  // CONFIG_NEW_INTER_MODES

void av1_setup_frame_buf_refs(AV1_COMMON *cm);
void av1_setup_frame_sign_bias(AV1_COMMON *cm);
void av1_setup_skip_mode_allowed(AV1_COMMON *cm);
void av1_setup_motion_field(AV1_COMMON *cm);
#if !CONFIG_NEW_REF_SIGNALING
void av1_set_frame_refs(AV1_COMMON *const cm, int *remapped_ref_idx,
                        int lst_map_idx, int gld_map_idx);
#endif  // !CONFIG_NEW_REF_SIGNALING

#if CONFIG_NEW_REF_SIGNALING
static INLINE void av1_collect_neighbors_ref_counts_nrs(
    const AV1_COMMON *const cm, MACROBLOCKD *const xd) {
  (void)cm;
  av1_zero(xd->neighbors_ref_counts_nrs);

  uint8_t *const ref_counts = xd->neighbors_ref_counts_nrs;

  const MB_MODE_INFO *const above_mbmi = xd->above_mbmi;
  const MB_MODE_INFO *const left_mbmi = xd->left_mbmi;
  const int above_in_image = xd->up_available;
  const int left_in_image = xd->left_available;

  // Above neighbor
  if (above_in_image && is_inter_block(above_mbmi)) {
    // TODO(sarahparker) Temporary assert, see aomedia:3060
    assert(convert_named_ref_to_ranked_ref_index(&cm->new_ref_frame_data,
                                                 above_mbmi->ref_frame[0]) ==
           above_mbmi->ref_frame_nrs[0]);
    ref_counts[above_mbmi->ref_frame_nrs[0]]++;
    if (has_second_ref(above_mbmi)) {
      // TODO(sarahparker) Temporary assert, see aomedia:3060
      assert(convert_named_ref_to_ranked_ref_index(&cm->new_ref_frame_data,
                                                   above_mbmi->ref_frame[1]) ==
             above_mbmi->ref_frame_nrs[1]);
      ref_counts[above_mbmi->ref_frame_nrs[1]]++;
    }
  }

  // Left neighbor
  if (left_in_image && is_inter_block(left_mbmi)) {
    // TODO(sarahparker) Temporary assert, see aomedia:3060
    assert(convert_named_ref_to_ranked_ref_index(&cm->new_ref_frame_data,
                                                 left_mbmi->ref_frame[0]) ==
           left_mbmi->ref_frame_nrs[0]);
    ref_counts[left_mbmi->ref_frame_nrs[0]]++;
    if (has_second_ref(left_mbmi)) {
      // TODO(sarahparker) Temporary assert, see aomedia:3060
      assert(convert_named_ref_to_ranked_ref_index(&cm->new_ref_frame_data,
                                                   left_mbmi->ref_frame[1]) ==
             left_mbmi->ref_frame_nrs[1]);
      ref_counts[left_mbmi->ref_frame_nrs[1]]++;
    }
  }
}
#endif  // CONFIG_NEW_REF_SIGNALING

static INLINE void av1_collect_neighbors_ref_counts(MACROBLOCKD *const xd) {
  av1_zero(xd->neighbors_ref_counts);

  uint8_t *const ref_counts = xd->neighbors_ref_counts;

  const MB_MODE_INFO *const above_mbmi = xd->above_mbmi;
  const MB_MODE_INFO *const left_mbmi = xd->left_mbmi;
  const int above_in_image = xd->up_available;
  const int left_in_image = xd->left_available;

  // Above neighbor
  if (above_in_image && is_inter_block(above_mbmi)) {
    ref_counts[above_mbmi->ref_frame[0]]++;
    if (has_second_ref(above_mbmi)) {
      ref_counts[above_mbmi->ref_frame[1]]++;
    }
  }

  // Left neighbor
  if (left_in_image && is_inter_block(left_mbmi)) {
    ref_counts[left_mbmi->ref_frame[0]]++;
    if (has_second_ref(left_mbmi)) {
      ref_counts[left_mbmi->ref_frame[1]]++;
    }
  }
}

void av1_copy_frame_mvs(const AV1_COMMON *const cm,
                        const MB_MODE_INFO *const mi, int mi_row, int mi_col,
                        int x_mis, int y_mis);

// The global_mvs output parameter points to an array of REF_FRAMES elements.
// The caller may pass a null global_mvs if it does not need the global_mvs
// output.
void av1_find_mv_refs(const AV1_COMMON *cm, const MACROBLOCKD *xd,
                      MB_MODE_INFO *mi, MV_REFERENCE_FRAME ref_frame,
#if CONFIG_NEW_REF_SIGNALING
                      MV_REFERENCE_FRAME_NRS ref_frame_nrs,
#endif  // CONFIG_NEW_REF_SIGNALING
                      uint8_t ref_mv_count[MODE_CTX_REF_FRAMES],
                      CANDIDATE_MV ref_mv_stack[][MAX_REF_MV_STACK_SIZE],
                      uint16_t ref_mv_weight[][MAX_REF_MV_STACK_SIZE],
                      int_mv mv_ref_list[][MAX_MV_REF_CANDIDATES],
                      int_mv *global_mvs,
#if CONFIG_NEW_REF_SIGNALING
                      int_mv *global_mvs_nrs,
#endif  // CONFIG_NEW_REF_SIGNALING
                      int16_t *mode_context);

// check a list of motion vectors by sad score using a number rows of pixels
// above and a number cols of pixels in the left to select the one with best
// score to use as ref motion vector
void av1_find_best_ref_mvs(int_mv *mvlist, int_mv *nearest_mv, int_mv *near_mv,
                           MvSubpelPrecision precision);

uint8_t av1_selectSamples(MV *mv, int *pts, int *pts_inref, int len,
                          BLOCK_SIZE bsize);
uint8_t av1_findSamples(const AV1_COMMON *cm, MACROBLOCKD *xd, int *pts,
                        int *pts_inref);

#define INTRABC_DELAY_PIXELS 256  //  Delay of 256 pixels
#define INTRABC_DELAY_SB64 (INTRABC_DELAY_PIXELS / 64)

static INLINE void av1_find_ref_dv(int_mv *ref_dv, const TileInfo *const tile,
                                   int mib_size, int mi_row) {
  if (mi_row - mib_size < tile->mi_row_start) {
    ref_dv->as_fullmv.row = 0;
    ref_dv->as_fullmv.col = -MI_SIZE * mib_size - INTRABC_DELAY_PIXELS;
  } else {
    ref_dv->as_fullmv.row = -MI_SIZE * mib_size;
    ref_dv->as_fullmv.col = 0;
  }
  convert_fullmv_to_mv(ref_dv);
}

static INLINE int av1_is_dv_valid(const MV dv, const AV1_COMMON *cm,
                                  const MACROBLOCKD *xd, int mi_row, int mi_col,
                                  BLOCK_SIZE bsize, int mib_size_log2) {
  const int bw = block_size_wide[bsize];
  const int bh = block_size_high[bsize];
  const int SCALE_PX_TO_MV = 8;
  // Disallow subpixel for now
  // SUBPEL_MASK is not the correct scale
  if (((dv.row & (SCALE_PX_TO_MV - 1)) || (dv.col & (SCALE_PX_TO_MV - 1))))
    return 0;

  const TileInfo *const tile = &xd->tile;
  // Is the source top-left inside the current tile?
  const int src_top_edge = mi_row * MI_SIZE * SCALE_PX_TO_MV + dv.row;
  const int tile_top_edge = tile->mi_row_start * MI_SIZE * SCALE_PX_TO_MV;
  if (src_top_edge < tile_top_edge) return 0;
  const int src_left_edge = mi_col * MI_SIZE * SCALE_PX_TO_MV + dv.col;
  const int tile_left_edge = tile->mi_col_start * MI_SIZE * SCALE_PX_TO_MV;
  if (src_left_edge < tile_left_edge) return 0;
  // Is the bottom right inside the current tile?
  const int src_bottom_edge = (mi_row * MI_SIZE + bh) * SCALE_PX_TO_MV + dv.row;
  const int tile_bottom_edge = tile->mi_row_end * MI_SIZE * SCALE_PX_TO_MV;
  if (src_bottom_edge > tile_bottom_edge) return 0;
  const int src_right_edge = (mi_col * MI_SIZE + bw) * SCALE_PX_TO_MV + dv.col;
  const int tile_right_edge = tile->mi_col_end * MI_SIZE * SCALE_PX_TO_MV;
  if (src_right_edge > tile_right_edge) return 0;

  // Special case for sub 8x8 chroma cases, to prevent referring to chroma
  // pixels outside current tile.
  if (xd->is_chroma_ref && av1_num_planes(cm) > 1) {
    const struct macroblockd_plane *const pd = &xd->plane[1];
    if (bw < 8 && pd->subsampling_x)
      if (src_left_edge < tile_left_edge + 4 * SCALE_PX_TO_MV) return 0;
    if (bh < 8 && pd->subsampling_y)
      if (src_top_edge < tile_top_edge + 4 * SCALE_PX_TO_MV) return 0;
  }

  // Is the bottom right within an already coded SB? Also consider additional
  // constraints to facilitate HW decoder.
  const int max_mib_size = 1 << mib_size_log2;
  const int active_sb_row = mi_row >> mib_size_log2;
  const int active_sb64_col = (mi_col * MI_SIZE) >> 6;
  const int sb_size = max_mib_size * MI_SIZE;
  const int src_sb_row = ((src_bottom_edge >> 3) - 1) / sb_size;
  const int src_sb64_col = ((src_right_edge >> 3) - 1) >> 6;
  const int total_sb64_per_row =
      ((tile->mi_col_end - tile->mi_col_start - 1) >> 4) + 1;
  const int active_sb64 = active_sb_row * total_sb64_per_row + active_sb64_col;
  const int src_sb64 = src_sb_row * total_sb64_per_row + src_sb64_col;
  if (src_sb64 >= active_sb64 - INTRABC_DELAY_SB64) return 0;

  // Wavefront constraint: use only top left area of frame for reference.
  const int gradient = 1 + INTRABC_DELAY_SB64 + (sb_size > 64);
  const int wf_offset = gradient * (active_sb_row - src_sb_row);
  if (src_sb_row > active_sb_row ||
      src_sb64_col >= active_sb64_col - INTRABC_DELAY_SB64 + wf_offset)
    return 0;

  return 1;
}

#if CONFIG_REF_MV_BANK
static INLINE int av1_get_column_bank_index(const AV1_COMMON *cm, int mi_col) {
  const int mi_cols = cm->width / MI_SIZE;
  const int mib_size = cm->seq_params.mib_size;
  const int sb_cols = (mi_cols + mib_size - 1) / mib_size;
  // How many SBs a column bank covers.
  const int bank_width = AOMMAX(1, sb_cols / REF_MV_BANK_COLS);
  const int sb_col = mi_col / mib_size;
  return AOMMIN(sb_col / bank_width, REF_MV_BANK_COLS - 1);
}

void av1_update_ref_mv_bank(const AV1_COMMON *const cm, MACROBLOCKD *const xd,
                            const MB_MODE_INFO *const mbmi);
#endif  // CONFIG_REF_MV_BANK

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_MVREF_COMMON_H_
