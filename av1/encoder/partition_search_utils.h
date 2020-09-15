/*
 * Copyright (c) 2020, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#ifndef AOM_AV1_ENCODER_PARTITION_SEARCH_UTILS_H_
#define AOM_AV1_ENCODER_PARTITION_SEARCH_UTILS_H_

#include "av1/encoder/encodemb.h"
#include "av1/encoder/encoder.h"
#include "av1/encoder/encodeframe_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

#if !CONFIG_REALTIME_ONLY
// Rectangular partition types.
enum { HORZ = 0, VERT, NUM_RECT_PARTS } UENUM1BYTE(RECT_PART_TYPE);

enum { PART_A = 0, PART_B, NUM_AB_TYPES } UENUM1BYTE(AB_PART_TYPE);
#define SUB_PARTITIONS_SPLIT 4
#define SUB_PARTITIONS_RECT 2

// This structure contains block size related
// variables for use in rd_pick_partition().
typedef struct {
  // Half of block width/height to determine block edge.
  int mi_step_w;
  int mi_step_h;

  // Block row and column indices.
  int mi_row;
  int mi_col;

  // Block width of current partition block.
  int width;

  BLOCK_SIZE max_sq_part;
  BLOCK_SIZE min_sq_part;

  // Block width of maximum partition size allowed.
  int max_partition_size_1d;
  // Block width of minimum partition size allowed.
  int min_partition_size_1d;

  int is_le_min_sq_part;
  int is_gt_max_sq_part;

  // Indicates edge blocks in frame.
  int has_rows;
  int has_cols;

  // Block size of current partition.
  BLOCK_SIZE bsize;

  // Chroma subsampling in x and y directions.
  int ss_x;
  int ss_y;
} PartitionBlkParams;

// Structure holding state variables for partition search.
typedef struct {
  // Parameters related to partition block size.
  PartitionBlkParams part_blk_params;

  int is_block_splittable;

  // // RD cost for the current block of given partition type.
  // RD_STATS this_rdc;

  // // RD cost summed across all blocks of partition type.
  // RD_STATS sum_rdc;

  // Array holding partition type cost.
  int tmp_partition_cost[PARTITION_TYPES];
#if CONFIG_EXT_RECUR_PARTITIONS
  int partition_cost_table[EXT_PARTITION_TYPES];
#endif

  // Pointer to partition cost buffer
  int *partition_cost;

  // RD costs for different partition types.
  int64_t none_rd;
  int64_t split_rd[SUB_PARTITIONS_SPLIT];
  // RD costs for rectangular partitions.
  // rect_part_rd[0][i] is the RD cost of ith partition index of PARTITION_HORZ.
  // rect_part_rd[1][i] is the RD cost of ith partition index of PARTITION_VERT.
  int64_t rect_part_rd[NUM_RECT_PARTS][SUB_PARTITIONS_RECT];

#if CONFIG_EXT_RECUR_PARTITIONS
  // New Simple Motion Result for PARTITION_NONE
  SMSPartitionStats none_data;
#endif  // CONFIG_EXT_RECUR_PARTITIONS

  // Flags indicating if the corresponding partition was winner or not.
  // Used to bypass similar blocks during AB partition evaluation.
  int split_ctx_is_ready[2];
  int rect_ctx_is_ready[NUM_RECT_PARTS];

  // Flags to prune/skip particular partition size evaluation.
  int terminate_partition_search;
  int partition_none_allowed;
  int partition_rect_allowed[NUM_RECT_PARTS];
#if CONFIG_EXT_RECUR_PARTITIONS
  int partition_3_allowed[NUM_RECT_PARTS];
#else
  int partition_ab_allowed[NUM_RECT_PARTS][NUM_AB_TYPES];
  int partition_4_allowed[NUM_RECT_PARTS];
#endif  // CONFIG_EXT_RECUR_PARTITIONS
  int do_rectangular_split;
#if !(CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT)
  int do_square_split;
#endif  // !(CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT)
  int prune_rect_part[NUM_RECT_PARTS];

  // Partition plane context index.
  int pl;

  // This flag will be set if best partition is found from the search.
  bool found_best_partition;
} PartitionSearchState;

void av1_init_partition_search_state(PartitionSearchState *search_state,
                                     MACROBLOCK *x, AV1_COMP *const cpi,
                                     const PC_TREE *pc_tree, int mi_row,
                                     int mi_col, BLOCK_SIZE bsize,
                                     BLOCK_SIZE max_sq_part,
                                     BLOCK_SIZE min_sq_part);

void av1_update_picked_ref_frames_mask(MACROBLOCK *const x, int ref_type,
                                       BLOCK_SIZE bsize, int mib_size,
                                       int mi_row, int mi_col);
#endif  // !CONFIG_REALTIME_ONLY

void av1_encode_b(const AV1_COMP *const cpi, TileDataEnc *tile_data,
                  ThreadData *td, TOKENEXTRA **tp, int mi_row, int mi_col,
                  RUN_TYPE dry_run, BLOCK_SIZE bsize, PARTITION_TYPE partition,
                  const PICK_MODE_CONTEXT *const ctx, int *rate);

#if !CONFIG_REALTIME_ONLY
void av1_encode_sb(const AV1_COMP *const cpi, ThreadData *td,
                   TileDataEnc *tile_data, TOKENEXTRA **tp, int mi_row,
                   int mi_col, RUN_TYPE dry_run, BLOCK_SIZE bsize,
                   PC_TREE *pc_tree, PARTITION_TREE *ptree, int *rate);

// Checks to see if a super block is on a horizontal image edge.
// In most cases this is the "real" edge unless there are formatting
// bars embedded in the stream.
int av1_active_h_edge(const AV1_COMP *cpi, int mi_row, int mi_step);

// Checks to see if a super block is on a vertical image edge.
// In most cases this is the "real" edge unless there are formatting
// bars embedded in the stream.
int av1_active_v_edge(const AV1_COMP *cpi, int mi_row, int mi_step);

void av1_encode_superblock(const AV1_COMP *const cpi, TileDataEnc *tile_data,
                           ThreadData *td, TOKENEXTRA **t, RUN_TYPE dry_run,
                           BLOCK_SIZE bsize, int *rate);

void av1_update_state(const AV1_COMP *const cpi, ThreadData *td,
                      const PICK_MODE_CONTEXT *const ctx, int mi_row,
                      int mi_col, BLOCK_SIZE bsize, RUN_TYPE dry_run);
#endif  // !CONFIG_REALTIME_ONLY

void av1_setup_block_rdmult(const struct AV1_COMP *cpi,
                            struct macroblock *const x, int mi_row, int mi_col,
                            BLOCK_SIZE bsize, AQ_MODE aq_mode,
                            MB_MODE_INFO *mbmi);

#if !CONFIG_REALTIME_ONLY
static INLINE int check_is_chroma_size_valid(PARTITION_TYPE partition,
                                             BLOCK_SIZE bsize, int mi_row,
                                             int mi_col, int ss_x, int ss_y,
                                             const PC_TREE *pc_tree) {
  const BLOCK_SIZE subsize = get_partition_subsize(bsize, partition);
  int is_valid = 0;
  if (subsize < BLOCK_SIZES_ALL) {
    CHROMA_REF_INFO tmp_chr_ref_info = { 1, 0, mi_row, mi_col, bsize, bsize };
    set_chroma_ref_info(mi_row, mi_col, 0, subsize, &tmp_chr_ref_info,
                        &pc_tree->chroma_ref_info, bsize, partition, ss_x,
                        ss_y);
    is_valid = get_plane_block_size(tmp_chr_ref_info.bsize_base, ss_x, ss_y) !=
               BLOCK_INVALID;
  }
  return is_valid;
}

// If any of the search_state_rd is INT_MAX, set them to 0 for the ML models
static INLINE void clip_partition_search_state_rd(
    PartitionSearchState *search_state) {
  search_state->rect_part_rd[HORZ][0] =
      (search_state->rect_part_rd[HORZ][0] < INT64_MAX
           ? search_state->rect_part_rd[HORZ][0]
           : 0);
  search_state->rect_part_rd[HORZ][1] =
      (search_state->rect_part_rd[HORZ][1] < INT64_MAX
           ? search_state->rect_part_rd[HORZ][1]
           : 0);
  search_state->rect_part_rd[VERT][0] =
      (search_state->rect_part_rd[VERT][0] < INT64_MAX
           ? search_state->rect_part_rd[VERT][0]
           : 0);
  search_state->rect_part_rd[VERT][1] =
      (search_state->rect_part_rd[VERT][1] < INT64_MAX
           ? search_state->rect_part_rd[VERT][1]
           : 0);
  search_state->split_rd[0] =
      (search_state->split_rd[0] < INT64_MAX ? search_state->split_rd[0] : 0);
  search_state->split_rd[1] =
      (search_state->split_rd[1] < INT64_MAX ? search_state->split_rd[1] : 0);
  search_state->split_rd[2] =
      (search_state->split_rd[2] < INT64_MAX ? search_state->split_rd[2] : 0);
  search_state->split_rd[3] =
      (search_state->split_rd[3] < INT64_MAX ? search_state->split_rd[3] : 0);
}

static INLINE void init_partition_allowed(PartitionSearchState *search_state,
                                          const AV1_COMP *cpi,
                                          const PC_TREE *pc_tree) {
  const PartitionBlkParams *blk_params = &search_state->part_blk_params;
  const BLOCK_SIZE bsize = blk_params->bsize;
  const int mi_row = blk_params->mi_row, mi_col = blk_params->mi_col;
  const int has_rows = blk_params->has_rows, has_cols = blk_params->has_cols;
  const int ss_x = blk_params->ss_x, ss_y = blk_params->ss_y;

  int is_chroma_size_valid_horz = check_is_chroma_size_valid(
      PARTITION_HORZ, bsize, mi_row, mi_col, ss_x, ss_y, pc_tree);

  int is_chroma_size_valid_vert = check_is_chroma_size_valid(
      PARTITION_VERT, bsize, mi_row, mi_col, ss_x, ss_y, pc_tree);

  search_state->terminate_partition_search = 0;
  search_state->partition_none_allowed = has_rows && has_cols;
  search_state->partition_rect_allowed[HORZ] =
#if CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT
      (has_cols || (!has_rows && !has_cols)) &&
      is_partition_valid(bsize, PARTITION_HORZ) &&
#else   // CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT
      blk_params->has_cols && is_partition_valid(bsize, PARTITION_HORZ) &&
#endif  // CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT
      cpi->oxcf.enable_rect_partitions && is_chroma_size_valid_horz;
  search_state->partition_rect_allowed[VERT] =
#if CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT
      (has_rows || (!has_rows && !has_cols)) &&
      is_partition_valid(bsize, PARTITION_VERT) &&
#else   // CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT
      blk_params->has_rows && is_partition_valid(bsize, PARTITION_VERT) &&
#endif  // CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT
      cpi->oxcf.enable_rect_partitions && is_chroma_size_valid_vert;
#if !(CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT)
  search_state->do_square_split =
      search_state->is_block_splittable && is_square_block(bsize);
#endif  // !(CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT)
}
#endif  // !CONFIG_REALTIME_ONLY
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_ENCODER_ENCODEFRAME_UTILS_H_
