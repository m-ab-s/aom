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

#ifndef AOM_AV1_ENCODER_ENCODEFRAME_UTILS_H_
#define AOM_AV1_ENCODER_ENCODEFRAME_UTILS_H_

#include "av1/common/reconinter.h"

#include "av1/encoder/encoder.h"
#include "av1/encoder/partition_strategy.h"
#include "av1/encoder/rdopt.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { PICK_MODE_RD = 0, PICK_MODE_NONRD, PICK_MODE_FAST_NONRD };

typedef struct {
  ENTROPY_CONTEXT a[MAX_MIB_SIZE * MAX_MB_PLANE];
  ENTROPY_CONTEXT l[MAX_MIB_SIZE * MAX_MB_PLANE];
  PARTITION_CONTEXT sa[MAX_MIB_SIZE];
  PARTITION_CONTEXT sl[MAX_MIB_SIZE];
  TXFM_CONTEXT *p_ta;
  TXFM_CONTEXT *p_tl;
  TXFM_CONTEXT ta[MAX_MIB_SIZE];
  TXFM_CONTEXT tl[MAX_MIB_SIZE];
} RD_SEARCH_MACROBLOCK_CONTEXT;

enum {
  SB_SINGLE_PASS,  // Single pass encoding: all ctxs get updated normally
  SB_DRY_PASS,     // First pass of multi-pass: does not update the ctxs
  SB_WET_PASS      // Second pass of multi-pass: finalize and update the ctx
} UENUM1BYTE(SB_MULTI_PASS_MODE);

// This struct is used to store the statistics used by sb-level multi-pass
// encoding. Currently, this is only used to make a copy of the state before we
// perform the first pass
typedef struct SB_FIRST_PASS_STATS {
  RD_SEARCH_MACROBLOCK_CONTEXT x_ctx;
  RD_COUNTS rd_count;

  int split_count;
  FRAME_COUNTS fc;
  InterModeRdModel inter_mode_rd_models[BLOCK_SIZES_ALL];
  int thresh_freq_fact[BLOCK_SIZES_ALL][MAX_MODES];
  int current_qindex;

#if CONFIG_INTERNAL_STATS
  unsigned int mode_chosen_counts[MAX_MODES];
#endif  // CONFIG_INTERNAL_STATS
} SB_FIRST_PASS_STATS;

#if !CONFIG_REALTIME_ONLY
void av1_init_encode_rd_sb(AV1_COMP *cpi, ThreadData *td,
                           const TileDataEnc *tile_data, PC_TREE **pc_root,
                           SIMPLE_MOTION_DATA_TREE *sms_root, RD_STATS *rd_cost,
                           int mi_row, int mi_col, int gather_tpl_data);

void av1_backup_sb_state(SB_FIRST_PASS_STATS *sb_fp_stats, const AV1_COMP *cpi,
                         ThreadData *td, const TileDataEnc *tile_data,
                         int mi_row, int mi_col);

void av1_restore_sb_state(const SB_FIRST_PASS_STATS *sb_fp_stats, AV1_COMP *cpi,
                          ThreadData *td, TileDataEnc *tile_data, int mi_row,
                          int mi_col);

void av1_reset_mbmi(AV1_COMMON *cm, int mi_row, int mi_col);
#endif  // !CONFIG_REALTIME_ONLY

void av1_setup_delta_q(AV1_COMP *const cpi, ThreadData *td, MACROBLOCK *const x,
                       const TileInfo *const tile_info, int mi_row, int mi_col,
                       int num_planes);

void av1_restore_context(const AV1_COMMON *cm, MACROBLOCK *x,
                         const RD_SEARCH_MACROBLOCK_CONTEXT *ctx, int mi_row,
                         int mi_col, BLOCK_SIZE bsize, const int num_planes);

void av1_save_context(const MACROBLOCK *x, RD_SEARCH_MACROBLOCK_CONTEXT *ctx,
                      int mi_row, int mi_col, BLOCK_SIZE bsize,
                      const int num_planes);

void av1_set_offsets_without_segment_id(const struct AV1_COMP *const cpi,
                                        const TileInfo *const tile,
                                        struct macroblock *const x, int mi_row,
                                        int mi_col, BLOCK_SIZE bsize,
                                        const CHROMA_REF_INFO *chr_ref_info);

void av1_enc_set_offsets(const AV1_COMP *const cpi, const TileInfo *const tile,
                         struct macroblock *const x, int mi_row, int mi_col,
                         BLOCK_SIZE bsize, const CHROMA_REF_INFO *chr_ref_info);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_ENCODER_ENCODEFRAME_UTILS_H_
