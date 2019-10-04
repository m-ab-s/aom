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

#include "av1/encoder/context_tree.h"
#include "av1/encoder/encoder.h"

static const BLOCK_SIZE square[MAX_SB_SIZE_LOG2 - 1] = {
  BLOCK_4X4, BLOCK_8X8, BLOCK_16X16, BLOCK_32X32, BLOCK_64X64, BLOCK_128X128,
};

void av1_copy_tree_context(PICK_MODE_CONTEXT *dst_ctx,
                           PICK_MODE_CONTEXT *src_ctx) {
  dst_ctx->mic = src_ctx->mic;
  dst_ctx->mbmi_ext = src_ctx->mbmi_ext;

  dst_ctx->num_4x4_blk = src_ctx->num_4x4_blk;
  dst_ctx->skippable = src_ctx->skippable;
  dst_ctx->best_mode_index = src_ctx->best_mode_index;

  memcpy(dst_ctx->blk_skip, src_ctx->blk_skip,
         sizeof(uint8_t) * src_ctx->num_4x4_blk);

  dst_ctx->hybrid_pred_diff = src_ctx->hybrid_pred_diff;
  dst_ctx->comp_pred_diff = src_ctx->comp_pred_diff;
  dst_ctx->single_pred_diff = src_ctx->single_pred_diff;

  dst_ctx->rd_stats = src_ctx->rd_stats;
  dst_ctx->rd_mode_is_ready = src_ctx->rd_mode_is_ready;

  memcpy(dst_ctx->pred_mv, src_ctx->pred_mv, sizeof(MV) * REF_FRAMES);
}

void av1_setup_shared_coeff_buffer(AV1_COMMON *cm,
                                   PC_TREE_SHARED_BUFFERS *shared_bufs) {
  for (int i = 0; i < 3; i++) {
    const int max_num_pix = MAX_SB_SIZE * MAX_SB_SIZE;
    CHECK_MEM_ERROR(cm, shared_bufs->coeff_buf[i],
                    aom_memalign(32, max_num_pix * sizeof(tran_low_t)));
    CHECK_MEM_ERROR(cm, shared_bufs->qcoeff_buf[i],
                    aom_memalign(32, max_num_pix * sizeof(tran_low_t)));
    CHECK_MEM_ERROR(cm, shared_bufs->dqcoeff_buf[i],
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

PICK_MODE_CONTEXT *av1_alloc_pmc(const AV1_COMMON *cm, BLOCK_SIZE bsize,
                                 PC_TREE_SHARED_BUFFERS *shared_bufs) {
  PICK_MODE_CONTEXT *ctx = NULL;
  struct aom_internal_error_info error;

  AOM_CHECK_MEM_ERROR(&error, ctx, aom_calloc(1, sizeof(*ctx)));

  const int num_planes = av1_num_planes(cm);
  const int num_pix = block_size_wide[bsize] * block_size_high[bsize];
  const int num_blk = num_pix / 16;

  ctx->num_4x4_blk = num_blk;
  AOM_CHECK_MEM_ERROR(&error, ctx->blk_skip,
                      aom_calloc(num_blk, sizeof(uint8_t)));
  for (int i = 0; i < num_planes; ++i) {
    ctx->coeff[i] = shared_bufs->coeff_buf[i];
    ctx->qcoeff[i] = shared_bufs->qcoeff_buf[i];
    ctx->dqcoeff[i] = shared_bufs->dqcoeff_buf[i];
    AOM_CHECK_MEM_ERROR(&error, ctx->eobs[i],
                        aom_memalign(32, num_blk * sizeof(*ctx->eobs[i])));
    AOM_CHECK_MEM_ERROR(
        &error, ctx->txb_entropy_ctx[i],
        aom_memalign(32, num_blk * sizeof(*ctx->txb_entropy_ctx[i])));
  }

  if (num_pix <= MAX_PALETTE_SQUARE) {
    for (int i = 0; i < 2; ++i) {
      AOM_CHECK_MEM_ERROR(
          &error, ctx->color_index_map[i],
          aom_memalign(32, num_pix * sizeof(*ctx->color_index_map[i])));
    }
  }

  return ctx;
}

void av1_free_pmc(PICK_MODE_CONTEXT *ctx, int num_planes) {
  if (ctx == NULL) return;

  aom_free(ctx->blk_skip);
  ctx->blk_skip = NULL;
  for (int i = 0; i < num_planes; ++i) {
    ctx->coeff[i] = NULL;
    ctx->qcoeff[i] = NULL;
    ctx->dqcoeff[i] = NULL;
    aom_free(ctx->eobs[i]);
    ctx->eobs[i] = NULL;
    aom_free(ctx->txb_entropy_ctx[i]);
    ctx->txb_entropy_ctx[i] = NULL;
  }

  for (int i = 0; i < 2; ++i) {
    aom_free(ctx->color_index_map[i]);
    ctx->color_index_map[i] = NULL;
  }

  aom_free(ctx);
}

PC_TREE *av1_alloc_pc_tree_node(BLOCK_SIZE bsize, int is_last) {
  PC_TREE *pc_tree = NULL;
  struct aom_internal_error_info error;

  AOM_CHECK_MEM_ERROR(&error, pc_tree, aom_calloc(1, sizeof(*pc_tree)));

  pc_tree->partitioning = PARTITION_NONE;
  pc_tree->block_size = bsize;
  pc_tree->is_last_subblock = is_last;

  pc_tree->none = NULL;
  for (int i = 0; i < 2; ++i) {
    pc_tree->horizontal[i] = NULL;
    pc_tree->vertical[i] = NULL;
  }
#if CONFIG_RECURSIVE_ABPART
  for (int i = 0; i < 2; ++i) {
    pc_tree->horza_split[i] = NULL;
    pc_tree->horzb_split[i] = NULL;
    pc_tree->verta_split[i] = NULL;
    pc_tree->vertb_split[i] = NULL;
  }
  pc_tree->horza_rec = NULL;
  pc_tree->horzb_rec = NULL;
  pc_tree->verta_rec = NULL;
  pc_tree->vertb_rec = NULL;
#else
  for (int i = 0; i < 3; ++i) {
    pc_tree->horizontala[i] = NULL;
    pc_tree->horizontalb[i] = NULL;
    pc_tree->verticala[i] = NULL;
    pc_tree->verticalb[i] = NULL;
  }
#endif  // CONFIG_RECURSIVE_ABPART
#if CONFIG_3WAY_PARTITIONS
  for (int i = 0; i < 3; ++i) {
    pc_tree->horizontal3[i] = NULL;
    pc_tree->vertical3[i] = NULL;
  }
#endif  // CONFIG_3WAY_PARTITIONS
  for (int i = 0; i < 4; ++i) {
#if !CONFIG_3WAY_PARTITIONS
    pc_tree->horizontal4[i] = NULL;
    pc_tree->vertical4[i] = NULL;
#endif  // !CONFIG_3WAY_PARTITIONS
    pc_tree->split[i] = NULL;
  }

  return pc_tree;
}

#define FREE_PMC_NODE(CTX)         \
  do {                             \
    av1_free_pmc(CTX, num_planes); \
    CTX = NULL;                    \
  } while (0)

void av1_free_pc_tree_recursive(PC_TREE *pc_tree, int num_planes, int keep_best,
                                int keep_none) {
  if (pc_tree == NULL) return;

  const PARTITION_TYPE partition = pc_tree->partitioning;

  if (!keep_none && (!keep_best || (partition != PARTITION_NONE)))
    FREE_PMC_NODE(pc_tree->none);

  for (int i = 0; i < 2; ++i) {
    if (!keep_best || (partition != PARTITION_HORZ))
      FREE_PMC_NODE(pc_tree->horizontal[i]);
    if (!keep_best || (partition != PARTITION_VERT))
      FREE_PMC_NODE(pc_tree->vertical[i]);
  }
#if CONFIG_RECURSIVE_ABPART
  if (!keep_best || (partition != PARTITION_HORZ_A))
    FREE_PMC_NODE(pc_tree->horza_rec);
  if (!keep_best || (partition != PARTITION_HORZ_B))
    FREE_PMC_NODE(pc_tree->horzb_rec);
  if (!keep_best || (partition != PARTITION_VERT_A))
    FREE_PMC_NODE(pc_tree->verta_rec);
  if (!keep_best || (partition != PARTITION_VERT_B))
    FREE_PMC_NODE(pc_tree->vertb_rec);
  for (int i = 0; i < 2; ++i) {
    if ((!keep_best || (partition != PARTITION_HORZ_A)) &&
        pc_tree->horza_split[i] != NULL) {
      av1_free_pc_tree_recursive(pc_tree->horza_split[i], num_planes, 0, 0);
      pc_tree->horza_split[i] = NULL;
    }
    if ((!keep_best || (partition != PARTITION_HORZ_B)) &&
        pc_tree->horzb_split[i] != NULL) {
      av1_free_pc_tree_recursive(pc_tree->horzb_split[i], num_planes, 0, 0);
      pc_tree->horzb_split[i] = NULL;
    }
    if ((!keep_best || (partition != PARTITION_VERT_A)) &&
        pc_tree->verta_split[i] != NULL) {
      av1_free_pc_tree_recursive(pc_tree->verta_split[i], num_planes, 0, 0);
      pc_tree->verta_split[i] = NULL;
    }
    if ((!keep_best || (partition != PARTITION_VERT_B)) &&
        pc_tree->vertb_split[i] != NULL) {
      av1_free_pc_tree_recursive(pc_tree->vertb_split[i], num_planes, 0, 0);
      pc_tree->vertb_split[i] = NULL;
    }
  }
#else
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
#endif  // CONFIG_RECURSIVE_ABPART
#if CONFIG_3WAY_PARTITIONS
  for (int i = 0; i < 3; ++i) {
    if (!keep_best || (partition != PARTITION_HORZ_3))
      FREE_PMC_NODE(pc_tree->horizontal3[i]);
    if (!keep_best || (partition != PARTITION_VERT_3))
      FREE_PMC_NODE(pc_tree->vertical3[i]);
  }
#else
  for (int i = 0; i < 4; ++i) {
    if (!keep_best || (partition != PARTITION_HORZ_4))
      FREE_PMC_NODE(pc_tree->horizontal4[i]);
    if (!keep_best || (partition != PARTITION_VERT_4))
      FREE_PMC_NODE(pc_tree->vertical4[i]);
  }
#endif  // CONFIG_3WAY_PARTITIONS

  if (!keep_best || (partition != PARTITION_SPLIT)) {
    for (int i = 0; i < 4; ++i) {
      if (pc_tree->split[i] != NULL) {
        av1_free_pc_tree_recursive(pc_tree->split[i], num_planes, 0, 0);
        // if (keep_best) aom_free(pc_tree->split[i]);
        pc_tree->split[i] = NULL;
      }
    }
  }

  if (!keep_best && !keep_none) aom_free(pc_tree);
}

void av1_setup_sms_tree(AV1_COMMON *cm, ThreadData *td) {
  const int tree_nodes_inc = 1024;
  const int leaf_factor = 4;
  const int leaf_nodes = 256 * leaf_factor;
  const int tree_nodes = tree_nodes_inc + 256 + 64 + 16 + 4 + 1;
  SIMPLE_MOTION_DATA_TREE *this_sms;
  int square_index = 1;
  int nodes;
  int sms_tree_index;

  aom_free(td->sms_tree);
  CHECK_MEM_ERROR(cm, td->sms_tree,
                  aom_calloc(tree_nodes, sizeof(*td->sms_tree)));
  this_sms = &td->sms_tree[0];

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
      for (int j = 0; j < 4; ++j) tree->split[j] = this_sms++;
      ++sms_tree_index;
    }
    ++square_index;
  }

  // Set up the root node for the largest superblock size
  int i = MAX_MIB_SIZE_LOG2 - MIN_MIB_SIZE_LOG2;
  td->sms_root[i] = &td->sms_tree[tree_nodes - 1];
  // Set up the root nodes for the rest of the possible superblock sizes
  while (--i >= 0) td->sms_root[i] = td->sms_root[i + 1]->split[0];
}

void av1_free_sms_tree(ThreadData *td) {
  if (td->sms_tree != NULL) {
    aom_free(td->sms_tree);
    td->sms_tree = NULL;
  }
}
