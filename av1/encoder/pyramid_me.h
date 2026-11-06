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

#ifndef AOM_AV1_ENCODER_PYRAMID_ME_H_
#define AOM_AV1_ENCODER_PYRAMID_ME_H_

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>

#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/pyramid.h"
#include "aom_scale/yv12config.h"
#include "av1/common/entropymv.h"
#include "av1/common/enums.h"
#include "av1/common/mv.h"
#include "av1/encoder/cost.h"
#include "av1/encoder/lookahead.h"
#include "av1/encoder/mcomp.h"

#ifdef __cplusplus
extern "C" {
#endif

/*!\cond */

struct AV1Common;
struct AV1_COMP;
struct EncodeFrameParams;

#define PYRAMID_ME_PADDING 64

// Extra columns allocated and filled to the right of PYRAMID_ME_PADDING.
// The post pyramid ME search area is clamped to the width aligned to 8, so a
// 64x64 block can start at column aligned_width - 1 and read up to 6 columns
// past width + PYRAMID_ME_PADDING - 1.
// The columns replicate the right-edge pixels.
#define PYRAMID_ME_RIGHT_SLACK 8

// Extra rows allocated and filled below PYRAMID_ME_PADDING.
// The post pyramid ME search area is clamped to the height aligned to 8, so a
// 64x64 block can start at row aligned_height - 1 and read up to 6 rows past
// height + PYRAMID_ME_PADDING - 1.
// The rows replicate the bottom-edge pixels.
#define PYRAMID_ME_BOTTOM_SLACK 8

// Number of search zones of the pre pyramid motion estimation.
#define PRE_PME_SEARCH_ZONES 2

#define PME_SEARCH_ZONE_ROWS 2
#define PME_SEARCH_ZONE_COLS 2
#define PME_SEARCH_ZONES (PME_SEARCH_ZONE_ROWS * PME_SEARCH_ZONE_COLS)

// Q index range of the leaf frames, to which the adjusted q index of the
// leaf frames (pframe_qindex) is clipped.
#define MIN_PME_Q_INDEX 120
#define MAX_PME_Q_INDEX 220

// Maximum number of frames processed in a sub-GOP: the frames of the sub-GOP
// and the frames beyond it.
#define MAX_PME_FRAME_IDX (2 * MAX_LAG_BUFFERS)
// Length of the frame arrays: the processed frames and the reference buffers.
#define MAX_LENGTH_PME_FRAME_STATS (MAX_PME_FRAME_IDX + REF_FRAMES + 1)

typedef enum {
  PYR_LEVEL0 = 0,  // full resolution frame
  PYR_LEVEL1 = 1,  // half width and height
  PYR_LEVEL2 = 2,  // quarter width and height
  MAX_PYR_LEVELS
} PYRAMID_LEVEL;

// Minimum and maximum search window size.
typedef struct {
  int min_w;
  int max_w;
  int min_h;
  int max_h;
} SearchWindowCfg;

// A motion vector and its cost: the SAD plus the MV rate cost.
typedef struct MvStats {
  int_mv mv;
  int cost;
} MvStats;

// MV cost context for the pyramid ME search kernels. The kernels select on
// (sad << sad_shift) + cost.
typedef struct PmeMvCost {
  // Only full_ref_mv, mv_cost_type, mvjcost, mvcost and sad_per_bit are used.
  // mv_cost_type is MV_COST_ENTROPY when pme_sf->enable_mv_cost is set,
  // and MV_COST_NONE otherwise, in which case the MV cost is 0.
  MV_COST_PARAMS params;
  // Search-window origin in this pyramid level's full-pel units. The kernels
  // add it to their loop indices to get the candidate MV.
  FULLPEL_MV origin;
  // Left shift applied to the SAD before adding the cost, so that a SAD of
  // sub-sampled rows is scaled to the full block.
  int sad_shift;
} PmeMvCost;

// Row part of the MV cost, computed once per search row: the row cost and the
// two joint costs the column can select between.
typedef struct {
  int rate_row;   // mvcost[0][] for this row's difference from full_ref_mv
  int joint_c0;   // mvjcost[] for this row with a zero column difference
  int joint_cnz;  // mvjcost[] for this row with a non-zero column difference
} PmeRowCost;

// PME stats of a 64x64 block w.r.t. one reference frame.
typedef struct {
  // Best MV and cost of each search zone at each pyramid level.
  MvStats pme_stats[MAX_PYR_LEVELS][PME_SEARCH_ZONES];
  // Post pyramid ME: best MV and cost of the 64x64 block and its sub-blocks.
  // The 16x16 blocks are grouped by 32x32 block, and the 8x8 blocks by 16x16
  // block, each group in raster order.
  MvStats post_pme_stats_b64;
  MvStats post_pme_stats_b32[4];
  MvStats post_pme_stats_b16[16];
  MvStats post_pme_stats_b8[64];
  // Best search zone of the last enabled pyramid level.
  MvStats best_pme_stats;
  // SAD of the 64x64 block at MV 0.
  int zero_mv_sad;
  // Whether the reference frame is pruned for this block.
  bool prune_ref;
} PmeSbStats;

// A frame of the sub-GOP, or a reference buffer, as used by the pyramid motion
// estimation.
typedef struct {
  // Source frame the pyramid is computed from. NULL for the last frame of the
  // previous sub-GOP, whose pyramid is already computed.
  const YV12_BUFFER_CONFIG *src;
  // Identifies the buffer the frame is reconstructed into. Used only to detect
  // reference frames that share a buffer, never dereferenced. NULL if the frame
  // has no buffer of its own.
  const void *buf_id;
  // Source pyramid of the frame.
  ImagePyramid *src_pyr;
  // PME stats of the frame.
  PmeSbStats (*stats)[INTER_REFS_PER_FRAME];
  // Index in PmeParams::frame_stats_ptr of each reference frame type.
  int ref_map_index[INTER_REFS_PER_FRAME];
  // Display order index of the frame.
  uint32_t display_index;
} PmeFrameStats;

// Buffers and state of the pyramid motion estimation.
typedef struct PmeParams {
  // PME stats at 64x64 block granularity of the frames processed in a sub-GOP.
  // stats_pool[i][j][k] stores the stats of the jth block w.r.t. the kth
  // reference frame of the ith processed frame.
  PmeSbStats (*stats_pool[MAX_LAG_BUFFERS])[INTER_REFS_PER_FRAME];

  // Source pyramids: the first REF_FRAMES are for the reference buffers, the
  // rest for the frames processed in a sub-GOP.
  ImagePyramid src_pyr_pool[REF_FRAMES + MAX_LAG_BUFFERS];

  // Source pyramid of the last frame of the previous sub-GOP, and its display
  // order (-1 if the pyramid is not available).
  ImagePyramid prev_gop_arf_src_pyr;
  int prev_gop_arf_disp_order;

  // Frames of the sub-GOP: entries 1 to REF_FRAMES are for the reference
  // buffers, and entry 0 is unused.
  PmeFrameStats pme_frame[MAX_LENGTH_PME_FRAME_STATS];

  // Pointer into pme_frame, indexed by the frame index in the sub-GOP (negative
  // for reference buffers).
  PmeFrameStats *frame_stats_ptr;

  // Number of frames set up in frame_stats_ptr: the frames of the sub-GOP and
  // the frames beyond it.
  int num_frames;

  // Q index of the leaf frames of the sub-GOP: adjusted (pframe_qindex), and as
  // chosen by the rate control (leaf_qindex).
  int pframe_qindex;
  int leaf_qindex;

  // Index in frame_stats_ptr of the frame being processed.
  int frame_idx;

  // MV cost context of all the pyramid ME stages, set up once per sub-GOP by
  // av1_pyramid_me_setup(). Its params.sad_per_bit is set per frame.
  PmeMvCost mv_cost;

  // Source pyramids of the reference frames of the frame being processed.
  // ref_src_pyr[i] is the pyramid of the ith reference frame type.
  ImagePyramid *ref_src_pyr[INTER_REFS_PER_FRAME];

  // Display order index of the reference frames of the frame being processed.
  int ref_disp_order_idx[INTER_REFS_PER_FRAME];

  // Width and height of the 64x64 block being processed, clipped to the frame
  // size aligned to a multiple of 8.
  int aligned_bw;
  int aligned_bh;

  // Whether the frame being processed is used as a reference frame.
  bool is_frame_used_as_ref;

  // Whether the PME stats of the current sub-GOP have been filled by
  // av1_pyramid_me_setup(). They are valid only when set.
  bool pme_stats_ready;

  // Frame size the buffers are allocated for.
  int alloc_width;
  int alloc_height;
} PmeParams;

// Returns the row part of the MV cost of the candidates in row mv_row.
static AOM_FORCE_INLINE PmeRowCost pme_row_cost(const PmeMvCost *cost,
                                                int mv_row) {
  PmeRowCost row_cost = { 0, 0, 0 };
  if (cost->params.mv_cost_type != MV_COST_ENTROPY) return row_cost;
  // mvcost[] covers +-MV_MAX/8 full pel; the pyramid ME does not clamp MVs.
  assert(abs(mv_row - cost->params.full_ref_mv.row) <= MV_MAX / 8);
  const int drow = GET_MV_SUBPEL(mv_row - cost->params.full_ref_mv.row);
  const int jbase = (!!drow) << 1;
  row_cost.rate_row = cost->params.mvcost[0][drow];
  row_cost.joint_c0 = cost->params.mvjcost[jbase];
  row_cost.joint_cnz = cost->params.mvjcost[jbase | 1];
  return row_cost;
}

// Returns the MV cost of the candidate in column mv_col of the row of
// row_cost.
static AOM_FORCE_INLINE int pme_col_cost(const PmeMvCost *cost,
                                         const PmeRowCost *row_cost,
                                         int mv_col) {
  if (cost->params.mv_cost_type != MV_COST_ENTROPY) return 0;
  assert(abs(mv_col - cost->params.full_ref_mv.col) <= MV_MAX / 8);
  const int dcol = GET_MV_SUBPEL(mv_col - cost->params.full_ref_mv.col);
  const int rate = (dcol ? row_cost->joint_cnz : row_cost->joint_c0) +
                   row_cost->rate_row + cost->params.mvcost[1][dcol];
  return ROUND_POWER_OF_TWO((unsigned)rate * cost->params.sad_per_bit,
                            AV1_PROB_COST_SHIFT);
}

// MV cost of a single candidate.
static AOM_FORCE_INLINE int cand_mv_cost(const PmeMvCost *cost, FULLPEL_MV mv) {
  const PmeRowCost row_cost = pme_row_cost(cost, mv.row);
  return pme_col_cost(cost, &row_cost, mv.col);
}

// Returns true if pyramid motion estimation can run: enabled by speed feature,
// with lookahead, and without resize or superres.
bool av1_is_pme_allowed(const struct AV1_COMP *cpi);

// Allocates the pyramid motion estimation buffers, which are shared by the
// frame parallel encoders. Called for every frame, it allocates them once, on
// the first frame that can run pyramid motion estimation, and reallocates them
// only when the frame size changes. Resize and superres do not change it, as
// they disable pyramid motion estimation.
void av1_alloc_pme_buffers(struct AV1_COMP *cpi);

// Marks the pyramid motion estimation stats as not ready, so that they are
// not used until av1_pyramid_me_setup() fills them for a sub-GOP.
void av1_init_pme_stats(PmeParams *pme_data);

// Frees the pyramid motion estimation buffers.
void av1_free_pme_buffers(PmeParams *pme_data);

// Computes the source pyramid of the last frame of the sub-GOP, used as a
// reference by the next sub-GOP. Called once per sub-GOP, when that frame is
// encoded.
void av1_compute_prev_gop_arf_pyramid(struct AV1_COMP *cpi);

/*!\endcond */

/*!\brief Performs pyramid motion estimation for the frames of a sub-GOP.
 *
 *\ingroup high_level_algo
 *
 * Called once at the start of each sub-GOP. Sets up the frames of the sub-GOP,
 * their source pyramids and the MV cost context, and fills the PME stats of
 * each frame that is searched. pme_data.pme_stats_ready tells whether the stats
 * can be used.
 *
 * \param[in]    cpi             Top - level encoder instance structure
 * \param[in]    frame_params    Per frame encoding parameters
 */
void av1_pyramid_me_setup(struct AV1_COMP *cpi,
                          const struct EncodeFrameParams *const frame_params);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_ENCODER_PYRAMID_ME_H_
