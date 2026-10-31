/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include "av1/encoder/context_tree.h"
#include "av1/encoder/encoder.h"
#include "av1/encoder/rd.h"
#include <assert.h>

void av1_copy_tree_context(PICK_MODE_CONTEXT *dst_ctx,
                           PICK_MODE_CONTEXT *src_ctx) {
  dst_ctx->mic = src_ctx->mic;
  dst_ctx->mbmi_ext_best = src_ctx->mbmi_ext_best;

  dst_ctx->num_4x4_blk = src_ctx->num_4x4_blk;
  dst_ctx->skippable = src_ctx->skippable;
#if CONFIG_INTERNAL_STATS
  dst_ctx->best_mode_index = src_ctx->best_mode_index;
#endif  // CONFIG_INTERNAL_STATS

  av1_copy_array(dst_ctx->tx_type_map, src_ctx->tx_type_map,
                 src_ctx->num_4x4_blk);

  dst_ctx->rd_stats = src_ctx->rd_stats;
  dst_ctx->rd_mode_is_ready = src_ctx->rd_mode_is_ready;
}

void av1_setup_shared_coeff_buffer(const SequenceHeader *const seq_params,
                                   PC_TREE_SHARED_BUFFERS *shared_bufs,
                                   struct aom_internal_error_info *error) {
  const int num_planes = seq_params->monochrome ? 1 : MAX_MB_PLANE;
  const int max_sb_square_y = 1 << num_pels_log2_lookup[seq_params->sb_size];
  const int max_sb_square_uv = max_sb_square_y >> (seq_params->subsampling_x +
                                                   seq_params->subsampling_y);
  for (int i = 0; i < num_planes; i++) {
    const int max_num_pix =
        (i == AOM_PLANE_Y) ? max_sb_square_y : max_sb_square_uv;
    AOM_CHECK_MEM_ERROR(error, shared_bufs->coeff_buf[i],
                        aom_memalign(32, max_num_pix * sizeof(tran_low_t)));
    AOM_CHECK_MEM_ERROR(error, shared_bufs->qcoeff_buf[i],
                        aom_memalign(32, max_num_pix * sizeof(tran_low_t)));
    AOM_CHECK_MEM_ERROR(error, shared_bufs->dqcoeff_buf[i],
                        aom_memalign(32, max_num_pix * sizeof(tran_low_t)));
  }
}

void av1_free_shared_coeff_buffer(PC_TREE_SHARED_BUFFERS *shared_bufs) {
  for (int i = 0; i < 3; i++) {
    aom_free(shared_bufs->coeff_buf[i]);
    aom_free(shared_bufs->qcoeff_buf[i]);
    aom_free(shared_bufs->dqcoeff_buf[i]);
    shared_bufs->coeff_buf[i] = NULL;
    shared_bufs->qcoeff_buf[i] = NULL;
    shared_bufs->dqcoeff_buf[i] = NULL;
  }
}

PICK_MODE_CONTEXT *av1_alloc_pmc(const struct AV1_COMP *const cpi,
                                 BLOCK_SIZE bsize,
                                 PC_TREE_SHARED_BUFFERS *shared_bufs) {
  const AV1_COMMON *const cm = &cpi->common;
  const int num_planes = av1_num_planes(cm);
  const int num_pix = block_size_wide[bsize] * block_size_high[bsize];
  const int num_blk = num_pix / 16;

  // Allocate PICK_MODE_CONTEXT and all sub-arrays (eobs, txb_entropy_ctx,
  // color_index_map, tx_type_map) in a single 32-byte aligned memory block to
  // avoid multiple aom_memalign/aom_free calls per partition context.
  const size_t ctx_sz = (sizeof(PICK_MODE_CONTEXT) + 31) & ~(size_t)31;
  const size_t eob_sz = (num_blk * sizeof(uint16_t) + 31) & ~(size_t)31;
  const size_t txb_ctx_sz =
      (num_blk * sizeof(ENTROPY_CONTEXT) + 31) & ~(size_t)31;
  const bool use_palette = (num_pix <= MAX_PALETTE_SQUARE) &&
                           cm->features.allow_screen_content_tools;
  const size_t pal_sz =
      use_palette ? ((num_pix * sizeof(uint8_t) + 31) & ~(size_t)31) : 0;
  const size_t tx_type_sz = num_blk * sizeof(uint8_t);

  const size_t total_sz = ctx_sz + (size_t)num_planes * (eob_sz + txb_ctx_sz) +
                          2 * pal_sz + tx_type_sz;
  uint8_t *raw = (uint8_t *)aom_memalign(32, total_sz);
  if (!raw) return NULL;

  PICK_MODE_CONTEXT *ctx = (PICK_MODE_CONTEXT *)raw;
  memset(ctx, 0, sizeof(*ctx));
  ctx->rd_mode_is_ready = 0;
  ctx->num_4x4_blk = num_blk;

  uint8_t *ptr = raw + ctx_sz;
  for (int i = 0; i < num_planes; ++i) {
    ctx->coeff[i] = shared_bufs->coeff_buf[i];
    ctx->qcoeff[i] = shared_bufs->qcoeff_buf[i];
    ctx->dqcoeff[i] = shared_bufs->dqcoeff_buf[i];
    ctx->eobs[i] = (uint16_t *)ptr;
    ptr += eob_sz;
    ctx->txb_entropy_ctx[i] = ptr;
    ptr += txb_ctx_sz;
  }

  if (use_palette) {
    ctx->color_index_map[0] = ptr;
    ptr += pal_sz;
    ctx->color_index_map[1] = ptr;
    ptr += pal_sz;
  } else {
    ctx->color_index_map[0] = NULL;
    ctx->color_index_map[1] = NULL;
  }

  ctx->tx_type_map = ptr;
  memset(ctx->tx_type_map, 0, tx_type_sz);

  av1_invalid_rd_stats(&ctx->rd_stats);

  return ctx;
}

void av1_reset_pmc(PICK_MODE_CONTEXT *ctx) {
  av1_zero_array(ctx->tx_type_map, ctx->num_4x4_blk);
  av1_invalid_rd_stats(&ctx->rd_stats);
}

void av1_free_pmc(PICK_MODE_CONTEXT *ctx, int num_planes) {
  (void)num_planes;
  if (ctx == NULL) return;
  aom_free(ctx);
}

PC_TREE *av1_alloc_pc_tree_node(BLOCK_SIZE bsize) {
  PC_TREE *pc_tree = aom_calloc(1, sizeof(*pc_tree));
  if (pc_tree == NULL) return NULL;

  pc_tree->partitioning = PARTITION_NONE;
  pc_tree->block_size = bsize;

  return pc_tree;
}

#define FREE_PMC_NODE(CTX)         \
  do {                             \
    av1_free_pmc(CTX, num_planes); \
    CTX = NULL;                    \
  } while (0)

void av1_free_pc_tree_recursive(PC_TREE *pc_tree, int num_planes, int keep_best,
                                int keep_none,
                                PARTITION_SEARCH_TYPE partition_search_type) {
  if (pc_tree == NULL) return;

  // Avoid freeing of extended partitions as they are not supported when
  // partition_search_type is VAR_BASED_PARTITION.
  if (partition_search_type == VAR_BASED_PARTITION && !keep_best &&
      !keep_none) {
    FREE_PMC_NODE(pc_tree->none);

    for (int i = 0; i < 2; ++i) {
      FREE_PMC_NODE(pc_tree->horizontal[i]);
      FREE_PMC_NODE(pc_tree->vertical[i]);
    }

#if !defined(NDEBUG) && !CONFIG_REALTIME_ONLY
    for (int i = 0; i < 3; ++i) {
      assert(pc_tree->horizontala[i] == NULL);
      assert(pc_tree->horizontalb[i] == NULL);
      assert(pc_tree->verticala[i] == NULL);
      assert(pc_tree->verticalb[i] == NULL);
    }
    for (int i = 0; i < 4; ++i) {
      assert(pc_tree->horizontal4[i] == NULL);
      assert(pc_tree->vertical4[i] == NULL);
    }
#endif

    for (int i = 0; i < 4; ++i) {
      if (pc_tree->split[i] != NULL) {
        av1_free_pc_tree_recursive(pc_tree->split[i], num_planes, 0, 0,
                                   partition_search_type);
        pc_tree->split[i] = NULL;
      }
    }
    aom_free(pc_tree);
    return;
  }

  const PARTITION_TYPE partition = pc_tree->partitioning;

  if (!keep_none && (!keep_best || (partition != PARTITION_NONE)))
    FREE_PMC_NODE(pc_tree->none);

  for (int i = 0; i < 2; ++i) {
    if (!keep_best || (partition != PARTITION_HORZ))
      FREE_PMC_NODE(pc_tree->horizontal[i]);
    if (!keep_best || (partition != PARTITION_VERT))
      FREE_PMC_NODE(pc_tree->vertical[i]);
  }
#if !CONFIG_REALTIME_ONLY
  for (int i = 0; i < 3; ++i) {
    if (!keep_best || (partition != PARTITION_HORZ_A))
      FREE_PMC_NODE(pc_tree->horizontala[i]);
    if (!keep_best || (partition != PARTITION_HORZ_B))
      FREE_PMC_NODE(pc_tree->horizontalb[i]);
    if (!keep_best || (partition != PARTITION_VERT_A))
      FREE_PMC_NODE(pc_tree->verticala[i]);
    if (!keep_best || (partition != PARTITION_VERT_B))
      FREE_PMC_NODE(pc_tree->verticalb[i]);
  }
  for (int i = 0; i < 4; ++i) {
    if (!keep_best || (partition != PARTITION_HORZ_4))
      FREE_PMC_NODE(pc_tree->horizontal4[i]);
    if (!keep_best || (partition != PARTITION_VERT_4))
      FREE_PMC_NODE(pc_tree->vertical4[i]);
  }
#endif
  if (!keep_best || (partition != PARTITION_SPLIT)) {
    for (int i = 0; i < 4; ++i) {
      if (pc_tree->split[i] != NULL) {
        av1_free_pc_tree_recursive(pc_tree->split[i], num_planes, 0, 0,
                                   partition_search_type);
        pc_tree->split[i] = NULL;
      }
    }
  }

  if (!keep_best && !keep_none) aom_free(pc_tree);
}

int av1_setup_sms_tree(AV1_COMP *const cpi, ThreadData *td) {
  // The structure 'sms_tree' is used to store the simple motion search data for
  // partition pruning in inter frames. Hence, the memory allocations and
  // initializations related to it are avoided for allintra encoding mode.
  if (cpi->oxcf.kf_cfg.key_freq_max == 0) return 0;

  AV1_COMMON *const cm = &cpi->common;
  const int stat_generation_stage = is_stat_generation_stage(cpi);
  const int is_sb_size_128 = cm->seq_params->sb_size == BLOCK_128X128;
  const int tree_nodes =
      av1_get_pc_tree_nodes(is_sb_size_128, stat_generation_stage);
  int sms_tree_index = 0;
  SIMPLE_MOTION_DATA_TREE *this_sms;
  int square_index = 1;
  int nodes;

  aom_free(td->sms_tree);
  td->sms_tree =
      (SIMPLE_MOTION_DATA_TREE *)aom_calloc(tree_nodes, sizeof(*td->sms_tree));
  if (!td->sms_tree) return -1;
  this_sms = &td->sms_tree[0];

  if (!stat_generation_stage) {
    const int leaf_factor = is_sb_size_128 ? 4 : 1;
    const int leaf_nodes = 256 * leaf_factor;

    // Sets up all the leaf nodes in the tree.
    for (sms_tree_index = 0; sms_tree_index < leaf_nodes; ++sms_tree_index) {
      SIMPLE_MOTION_DATA_TREE *const tree = &td->sms_tree[sms_tree_index];
      tree->block_size = square[0];
    }

    // Each node has 4 leaf nodes, fill each block_size level of the tree
    // from leafs to the root.
    for (nodes = leaf_nodes >> 2; nodes > 0; nodes >>= 2) {
      for (int i = 0; i < nodes; ++i) {
        SIMPLE_MOTION_DATA_TREE *const tree = &td->sms_tree[sms_tree_index];
        tree->block_size = square[square_index];
        for (int j = 0; j < 4; j++) tree->split[j] = this_sms++;
        ++sms_tree_index;
      }
      ++square_index;
    }
  } else {
    // Allocation for firstpass/LAP stage
    // TODO(Mufaddal): refactor square_index to use a common block_size macro
    // from firstpass.c
    SIMPLE_MOTION_DATA_TREE *const tree = &td->sms_tree[sms_tree_index];
    square_index = 2;
    tree->block_size = square[square_index];
  }

  // Set up the root node for the largest superblock size
  td->sms_root = &td->sms_tree[tree_nodes - 1];
  return 0;
}

void av1_free_sms_tree(ThreadData *td) {
  aom_free(td->sms_tree);
  td->sms_tree = NULL;
}
