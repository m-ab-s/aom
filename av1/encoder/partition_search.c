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

#include "aom_ports/system_state.h"

#include "av1/common/blockd.h"
#include "av1/common/enums.h"
#include "av1/common/reconintra.h"
#include "av1/encoder/aq_complexity.h"
#include "av1/encoder/aq_variance.h"
#include "av1/encoder/block.h"
#include "av1/encoder/context_tree.h"
#include "av1/encoder/encoder.h"
#include "av1/encoder/encodeframe.h"
#include "av1/encoder/encodemv.h"
#include "av1/encoder/partition_search.h"
#include "av1/encoder/partition_search_utils.h"
#include "av1/encoder/partition_strategy.h"
#include "av1/encoder/reconinter_enc.h"
#include "av1/encoder/tokenize.h"
#include "av1/encoder/var_based_part.h"

static void pick_sb_modes(AV1_COMP *const cpi, TileDataEnc *tile_data,
                          MACROBLOCK *const x, int mi_row, int mi_col,
                          RD_STATS *rd_cost, PARTITION_TYPE partition,
                          BLOCK_SIZE bsize, PICK_MODE_CONTEXT *ctx,
                          RD_STATS best_rd, int pick_mode_type) {
  AV1_COMMON *const cm = &cpi->common;
  const int num_planes = av1_num_planes(cm);
  TileInfo *const tile_info = &tile_data->tile_info;
  MACROBLOCKD *const xd = &x->e_mbd;
  MB_MODE_INFO *mbmi;
  MB_MODE_INFO *ctx_mbmi = &ctx->mic;
  struct macroblock_plane *const p = x->plane;
  struct macroblockd_plane *const pd = xd->plane;
  const AQ_MODE aq_mode = cpi->oxcf.aq_mode;
  int i;

#if CONFIG_COLLECT_COMPONENT_TIMING
  start_timing(cpi, rd_pick_sb_modes_time);
#endif

  if (best_rd.rdcost < 0) {
    ctx->rd_stats.rdcost = INT64_MAX;
    ctx->rd_stats.skip = 0;
    av1_invalid_rd_stats(rd_cost);
    return;
  }

  aom_clear_system_state();

  av1_enc_set_offsets(cpi, tile_info, x, mi_row, mi_col, bsize,
                      &ctx->chroma_ref_info);

  mbmi = xd->mi[0];

  if (ctx->rd_mode_is_ready) {
    assert(ctx_mbmi->sb_type == bsize);
    assert(ctx_mbmi->partition == partition);
    *mbmi = *ctx_mbmi;
    rd_cost->rate = ctx->rd_stats.rate;
    rd_cost->dist = ctx->rd_stats.dist;
    rd_cost->rdcost = ctx->rd_stats.rdcost;
  } else {
    mbmi->sb_type = bsize;
    mbmi->partition = partition;
#if CONFIG_DSPL_RESIDUAL
    mbmi->dspl_type = DSPL_NONE;
#endif  // CONFIG_DSPL_RESIDUAL
  }

  mbmi->chroma_ref_info = ctx->chroma_ref_info;

#if CONFIG_RD_DEBUG
  mbmi->mi_row = mi_row;
  mbmi->mi_col = mi_col;
#endif

  for (i = 0; i < num_planes; ++i) {
    p[i].coeff = ctx->coeff[i];
    p[i].qcoeff = ctx->qcoeff[i];
    pd[i].dqcoeff = ctx->dqcoeff[i];
    p[i].eobs = ctx->eobs[i];
    p[i].txb_entropy_ctx = ctx->txb_entropy_ctx[i];
  }

  for (i = 0; i < 2; ++i) pd[i].color_index_map = ctx->color_index_map[i];

  if (!ctx->rd_mode_is_ready) {
    ctx->skippable = 0;

    // Set to zero to make sure we do not use the previous encoded frame stats
    mbmi->skip = 0;

    // Reset skip mode flag.
    mbmi->skip_mode = 0;
  }

  x->skip_chroma_rd = !mbmi->chroma_ref_info.is_chroma_ref;

  if (ctx->rd_mode_is_ready) {
    x->skip = ctx->rd_stats.skip;
    *x->mbmi_ext = ctx->mbmi_ext;
    return;
  }

  if (is_cur_buf_hbd(xd)) {
    x->source_variance = av1_high_get_sby_perpixel_variance(
        cpi, &x->plane[0].src, bsize, xd->bd);
  } else {
    x->source_variance =
        av1_get_sby_perpixel_variance(cpi, &x->plane[0].src, bsize);
  }

  // If the threshold for disabling wedge search is zero, it means the feature
  // should not be used. Use a value that will always succeed in the check.
  if (cpi->sf.disable_wedge_search_edge_thresh == 0) {
    x->edge_strength = UINT16_MAX;
    x->edge_strength_x = UINT16_MAX;
    x->edge_strength_y = UINT16_MAX;
  } else {
    EdgeInfo ei =
        av1_get_edge_info(&x->plane[0].src, bsize, is_cur_buf_hbd(xd), xd->bd);
    x->edge_strength = ei.magnitude;
    x->edge_strength_x = ei.x;
    x->edge_strength_y = ei.y;
  }

  // Default initialization of the threshold for R-D optimization of
  // coefficients for mode decision
  x->coeff_opt_dist_threshold =
      get_rd_opt_coeff_thresh(cpi->coeff_opt_dist_threshold, 0, 0);

  // Save rdmult before it might be changed, so it can be restored later.
  const int orig_rdmult = x->rdmult;
  av1_setup_block_rdmult(cpi, x, mi_row, mi_col, bsize, aq_mode, mbmi);
  // Set error per bit for current rdmult
  set_error_per_bit(x, x->rdmult);
  av1_rd_cost_update(x->rdmult, &best_rd);

#if CONFIG_DERIVED_INTRA_MODE
  mbmi->use_derived_intra_mode[0] = 0;
  mbmi->use_derived_intra_mode[1] = 0;
#endif  // CONFIG_DERIVED_INTRA_MODE
#if CONFIG_DERIVED_MV
  mbmi->derived_mv_allowed = mbmi->use_derived_mv = 0;
#endif  // CONFIG_DERIVED_MV

  // Find best coding mode & reconstruct the MB so it is available
  // as a predictor for MBs that follow in the SB
  if (frame_is_intra_only(cm)) {
#if CONFIG_COLLECT_COMPONENT_TIMING
    start_timing(cpi, av1_rd_pick_intra_mode_sb_time);
#endif
    av1_rd_pick_intra_mode_sb(cpi, x, rd_cost, bsize, ctx, best_rd.rdcost);
#if CONFIG_COLLECT_COMPONENT_TIMING
    end_timing(cpi, av1_rd_pick_intra_mode_sb_time);
#endif
  } else {
#if CONFIG_COLLECT_COMPONENT_TIMING
    start_timing(cpi, av1_rd_pick_inter_mode_sb_time);
#endif
    if (segfeature_active(&cm->seg, mbmi->segment_id, SEG_LVL_SKIP)) {
      av1_rd_pick_inter_mode_sb_seg_skip(cpi, tile_data, x, mi_row, mi_col,
                                         rd_cost, bsize, ctx, best_rd.rdcost);
    } else {
      // TODO(kyslov): do the same for pick_intra_mode and
      //               pick_inter_mode_sb_seg_skip
      switch (pick_mode_type) {
#if !CONFIG_REALTIME_ONLY
        case PICK_MODE_RD:
          av1_rd_pick_inter_mode_sb(cpi, tile_data, x, rd_cost, bsize, ctx,
                                    best_rd.rdcost);
          break;
#endif
        case PICK_MODE_NONRD:
          av1_nonrd_pick_inter_mode_sb(cpi, tile_data, x, rd_cost, bsize, ctx,
                                       best_rd.rdcost);
          break;
        case PICK_MODE_FAST_NONRD:
          av1_fast_nonrd_pick_inter_mode_sb(cpi, tile_data, x, rd_cost, bsize,
                                            ctx, best_rd.rdcost);
          break;
        default: assert(0 && "Unknown pick mode type.");
      }
    }
#if CONFIG_COLLECT_COMPONENT_TIMING
    end_timing(cpi, av1_rd_pick_inter_mode_sb_time);
#endif
  }

  // Examine the resulting rate and for AQ mode 2 make a segment choice.
  if ((rd_cost->rate != INT_MAX) && (aq_mode == COMPLEXITY_AQ) &&
      (bsize >= BLOCK_16X16) &&
      (cm->current_frame.frame_type == KEY_FRAME ||
       cpi->refresh_alt_ref_frame || cpi->refresh_alt2_ref_frame ||
       (cpi->refresh_golden_frame && !cpi->rc.is_src_frame_alt_ref))) {
    av1_caq_select_segment(cpi, x, bsize, mi_row, mi_col, rd_cost->rate);
  }

  x->rdmult = orig_rdmult;

  // TODO(jingning) The rate-distortion optimization flow needs to be
  // refactored to provide proper exit/return handle.
  if (rd_cost->rate == INT_MAX) rd_cost->rdcost = INT64_MAX;

  ctx->rd_stats.rate = rd_cost->rate;
  ctx->rd_stats.dist = rd_cost->dist;
  ctx->rd_stats.rdcost = rd_cost->rdcost;

#if CONFIG_COLLECT_COMPONENT_TIMING
  end_timing(cpi, rd_pick_sb_modes_time);
#endif
}

void av1_nonrd_use_partition(AV1_COMP *cpi, ThreadData *td,
                             TileDataEnc *tile_data, MB_MODE_INFO **mib,
                             TOKENEXTRA **tp, int mi_row, int mi_col,
                             BLOCK_SIZE bsize, PC_TREE *pc_tree,
                             PARTITION_TREE *ptree) {
  AV1_COMMON *const cm = &cpi->common;
  TileInfo *const tile_info = &tile_data->tile_info;
  const SPEED_FEATURES *const sf = &cpi->sf;
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  const int ss_x = xd->plane[1].subsampling_x;
  const int ss_y = xd->plane[1].subsampling_y;
  // Only square blocks from 8x8 to 128x128 are supported
  assert(bsize >= BLOCK_8X8 && bsize <= BLOCK_128X128);
  const int bs = mi_size_wide[bsize];
  const int hbs = bs / 2;
  const PARTITION_TYPE partition =
      (bsize >= BLOCK_8X8) ? get_partition(cm, mi_row, mi_col, bsize)
                           : PARTITION_NONE;
  const BLOCK_SIZE subsize = get_partition_subsize(bsize, partition);
  RD_STATS dummy_cost;
  av1_invalid_rd_stats(&dummy_cost);
  RD_STATS invalid_rd;
  av1_invalid_rd_stats(&invalid_rd);

  if (mi_row >= cm->mi_rows || mi_col >= cm->mi_cols) return;

  assert(mi_size_wide[bsize] == mi_size_high[bsize]);

  pc_tree->partitioning = partition;

  assert(ptree);
  ptree->partition = partition;
  ptree->bsize = bsize;
  ptree->mi_row = mi_row;
  ptree->mi_col = mi_col;
  PARTITION_TREE *parent = ptree->parent;
  set_chroma_ref_info(mi_row, mi_col, ptree->index, bsize,
                      &ptree->chroma_ref_info,
                      parent ? &parent->chroma_ref_info : NULL,
                      parent ? parent->bsize : BLOCK_INVALID,
                      parent ? parent->partition : PARTITION_NONE, ss_x, ss_y);

  // int num_splittable_sub_blocks = 0;
  switch (partition) {
    case PARTITION_SPLIT:
      ptree->sub_tree[0] = av1_alloc_ptree_node(ptree, 0);
      ptree->sub_tree[1] = av1_alloc_ptree_node(ptree, 1);
      ptree->sub_tree[2] = av1_alloc_ptree_node(ptree, 2);
      ptree->sub_tree[3] = av1_alloc_ptree_node(ptree, 3);
      break;
#if CONFIG_EXT_RECUR_PARTITIONS
    case PARTITION_HORZ:
    case PARTITION_VERT:
      ptree->sub_tree[0] = av1_alloc_ptree_node(ptree, 0);
      ptree->sub_tree[1] = av1_alloc_ptree_node(ptree, 1);
      break;
#endif  // CONFIG_EXT_RECUR_PARTITIONS
    default: break;
  }

  xd->above_txfm_context = cm->above_txfm_context[tile_info->tile_row] + mi_col;
  xd->left_txfm_context =
      xd->left_txfm_context_buffer + (mi_row & MAX_MIB_MASK);

  switch (partition) {
    case PARTITION_NONE:
      pc_tree->none =
          av1_alloc_pmc(cm, mi_row, mi_col, bsize, pc_tree, PARTITION_NONE, 0,
                        ss_x, ss_y, &td->shared_coeff_buf);
      pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, &dummy_cost,
                    PARTITION_NONE, bsize, pc_tree->none, invalid_rd,
                    sf->use_fast_nonrd_pick_mode ? PICK_MODE_FAST_NONRD
                                                 : PICK_MODE_NONRD);
      av1_encode_b(cpi, tile_data, td, tp, mi_row, mi_col, 0, bsize, partition,
                   pc_tree->none, NULL);
      break;
    case PARTITION_VERT:
#if CONFIG_EXT_RECUR_PARTITIONS
      ptree->sub_tree[0]->partition = PARTITION_NONE;
      ptree->sub_tree[1]->partition = PARTITION_NONE;

      pc_tree->vertical[0] = av1_alloc_pc_tree_node(
          mi_row, mi_col, subsize, pc_tree, PARTITION_VERT, 0, 0, ss_x, ss_y);
      pc_tree->vertical[1] =
          av1_alloc_pc_tree_node(mi_row, mi_col + hbs, subsize, pc_tree,
                                 PARTITION_VERT, 1, 1, ss_x, ss_y);

      av1_nonrd_use_partition(cpi, td, tile_data, mib, tp, mi_row, mi_col,
                              subsize, pc_tree->vertical[0],
                              ptree->sub_tree[0]);
#else
      for (int i = 0; i < 2; ++i) {
        pc_tree->vertical[i] =
            av1_alloc_pmc(cm, mi_row, mi_col + hbs * i, subsize, pc_tree,
                          PARTITION_VERT, i, ss_x, ss_y, &td->shared_coeff_buf);
      }
      pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, &dummy_cost,
                    PARTITION_VERT, subsize, pc_tree->vertical[0], invalid_rd,
                    sf->use_fast_nonrd_pick_mode ? PICK_MODE_FAST_NONRD
                                                 : PICK_MODE_NONRD);
      av1_encode_b(cpi, tile_data, td, tp, mi_row, mi_col, 0, subsize,
                   PARTITION_VERT, pc_tree->vertical[0], NULL);
#endif  // CONFIG_EXT_RECUR_PARTITIONS
      if (mi_col + hbs < cm->mi_cols && bsize > BLOCK_8X8) {
#if CONFIG_EXT_RECUR_PARTITIONS
        av1_nonrd_use_partition(cpi, td, tile_data, mib + hbs, tp, mi_row,
                                mi_col + hbs, subsize, pc_tree->vertical[1],
                                ptree->sub_tree[1]);
#else
        pick_sb_modes(cpi, tile_data, x, mi_row, mi_col + hbs, &dummy_cost,
                      PARTITION_VERT, subsize, pc_tree->vertical[1], invalid_rd,
                      sf->use_fast_nonrd_pick_mode ? PICK_MODE_FAST_NONRD
                                                   : PICK_MODE_NONRD);
        av1_encode_b(cpi, tile_data, td, tp, mi_row, mi_col + hbs, 0, subsize,
                     PARTITION_VERT, pc_tree->vertical[1], NULL);
#endif  // CONFIG_EXT_RECUR_PARTITIONS
      }
      break;
    case PARTITION_HORZ:
#if CONFIG_EXT_RECUR_PARTITIONS
      ptree->sub_tree[0]->partition = PARTITION_NONE;
      ptree->sub_tree[1]->partition = PARTITION_NONE;

      pc_tree->horizontal[0] = av1_alloc_pc_tree_node(
          mi_row, mi_col, subsize, pc_tree, PARTITION_HORZ, 0, 0, ss_x, ss_y);
      pc_tree->horizontal[1] =
          av1_alloc_pc_tree_node(mi_row + hbs, mi_col, subsize, pc_tree,
                                 PARTITION_HORZ, 1, 1, ss_x, ss_y);

      av1_nonrd_use_partition(cpi, td, tile_data, mib, tp, mi_row, mi_col,
                              subsize, pc_tree->horizontal[0],
                              ptree->sub_tree[0]);
#else
      for (int i = 0; i < 2; ++i) {
        pc_tree->horizontal[i] =
            av1_alloc_pmc(cm, mi_row + hbs * i, mi_col, subsize, pc_tree,
                          PARTITION_HORZ, i, ss_x, ss_y, &td->shared_coeff_buf);
      }
      pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, &dummy_cost,
                    PARTITION_HORZ, subsize, pc_tree->horizontal[0], invalid_rd,
                    sf->use_fast_nonrd_pick_mode ? PICK_MODE_FAST_NONRD
                                                 : PICK_MODE_NONRD);
      av1_encode_b(cpi, tile_data, td, tp, mi_row, mi_col, 0, subsize,
                   PARTITION_HORZ, pc_tree->horizontal[0], NULL);
#endif  // CONFIG_EXT_RECUR_PARTITIONS

      if (mi_row + hbs < cm->mi_rows && bsize > BLOCK_8X8) {
#if CONFIG_EXT_RECUR_PARTITIONS
        av1_nonrd_use_partition(cpi, td, tile_data, mib + hbs * cm->mi_stride,
                                tp, mi_row + hbs, mi_col, subsize,
                                pc_tree->horizontal[1], ptree->sub_tree[1]);
#else
        pick_sb_modes(cpi, tile_data, x, mi_row + hbs, mi_col, &dummy_cost,
                      PARTITION_HORZ, subsize, pc_tree->horizontal[1],
                      invalid_rd,
                      sf->use_fast_nonrd_pick_mode ? PICK_MODE_FAST_NONRD
                                                   : PICK_MODE_NONRD);
        av1_encode_b(cpi, tile_data, td, tp, mi_row + hbs, mi_col, 0, subsize,
                     PARTITION_HORZ, pc_tree->horizontal[1], NULL);
#endif  // CONFIG_EXT_RECUR_PARTITIONS
      }
      break;
    case PARTITION_SPLIT:
      for (int i = 0; i < 4; i++) {
        int x_idx = (i & 1) * hbs;
        int y_idx = (i >> 1) * hbs;
        int jj = i >> 1, ii = i & 0x01;

        if ((mi_row + y_idx >= cm->mi_rows) || (mi_col + x_idx >= cm->mi_cols))
          continue;

        pc_tree->split[i] = av1_alloc_pc_tree_node(
            mi_row + y_idx, mi_col + x_idx, subsize, pc_tree, PARTITION_SPLIT,
            i, i == 3, ss_x, ss_y);
        av1_nonrd_use_partition(cpi, td, tile_data,
                                mib + jj * hbs * cm->mi_stride + ii * hbs, tp,
                                mi_row + y_idx, mi_col + x_idx, subsize,
                                pc_tree->split[i], ptree->sub_tree[i]);
      }
      break;
#if CONFIG_EXT_RECUR_PARTITIONS
    case PARTITION_HORZ_3:
    case PARTITION_VERT_3:
#else
    case PARTITION_VERT_A:
    case PARTITION_VERT_B:
    case PARTITION_HORZ_A:
    case PARTITION_HORZ_B:
    case PARTITION_HORZ_4:
    case PARTITION_VERT_4:
#endif  // CONFIG_EXT_RECUR_PARTITIONS
      assert(0 && "Cannot handle extended partition types");
    default: assert(0); break;
  }

  ptree->is_settled = 1;
  if (partition != PARTITION_SPLIT || bsize == BLOCK_8X8)
    update_partition_context(xd, mi_row, mi_col, subsize, bsize);
}

#if !CONFIG_REALTIME_ONLY
static void set_partial_sb_partition(const AV1_COMMON *const cm,
                                     MB_MODE_INFO *mi, int bh_in, int bw_in,
                                     int mi_rows_remaining,
                                     int mi_cols_remaining, BLOCK_SIZE bsize,
                                     MB_MODE_INFO **mib) {
  int bh = bh_in;
  int r, c;
  for (r = 0; r < cm->seq_params.mib_size; r += bh) {
    int bw = bw_in;
    for (c = 0; c < cm->seq_params.mib_size; c += bw) {
      const int index = r * cm->mi_stride + c;
      mib[index] = mi + index;
      mib[index]->sb_type = find_partition_size(
          bsize, mi_rows_remaining - r, mi_cols_remaining - c, &bh, &bw);
    }
  }
}

void av1_set_fixed_partitioning(AV1_COMP *cpi, const TileInfo *const tile,
                                MB_MODE_INFO **mib, int mi_row, int mi_col,
                                BLOCK_SIZE bsize) {
  AV1_COMMON *const cm = &cpi->common;
  const int mi_rows_remaining = tile->mi_row_end - mi_row;
  const int mi_cols_remaining = tile->mi_col_end - mi_col;
  int block_row, block_col;
  MB_MODE_INFO *const mi_upper_left = cm->mi + mi_row * cm->mi_stride + mi_col;
  int bh = mi_size_high[bsize];
  int bw = mi_size_wide[bsize];

  assert((mi_rows_remaining > 0) && (mi_cols_remaining > 0));

  // Apply the requested partition size to the SB if it is all "in image"
  if ((mi_cols_remaining >= cm->seq_params.mib_size) &&
      (mi_rows_remaining >= cm->seq_params.mib_size)) {
    for (block_row = 0; block_row < cm->seq_params.mib_size; block_row += bh) {
      for (block_col = 0; block_col < cm->seq_params.mib_size;
           block_col += bw) {
        int index = block_row * cm->mi_stride + block_col;
        mib[index] = mi_upper_left + index;
        mib[index]->sb_type = bsize;
      }
    }
  } else {
    // Else this is a partial SB.
    set_partial_sb_partition(cm, mi_upper_left, bh, bw, mi_rows_remaining,
                             mi_cols_remaining, bsize, mib);
  }
}

void av1_rd_use_partition(AV1_COMP *cpi, ThreadData *td, TileDataEnc *tile_data,
                          MB_MODE_INFO **mib, TOKENEXTRA **tp, int mi_row,
                          int mi_col, BLOCK_SIZE bsize, int *rate,
                          int64_t *dist, int do_recon, PC_TREE *pc_tree) {
  AV1_COMMON *const cm = &cpi->common;
  const int num_planes = av1_num_planes(cm);
  TileInfo *const tile_info = &tile_data->tile_info;
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  const int ss_x = xd->plane[1].subsampling_x;
  const int ss_y = xd->plane[1].subsampling_y;
  const int bw = mi_size_wide[bsize];
  const int bh = mi_size_high[bsize];
  const int hbw = bw / 2;
  const int hbh = bh / 2;
  const int pl = (bsize >= BLOCK_8X8)
                     ? partition_plane_context(xd, mi_row, mi_col, bsize)
                     : 0;
  const PARTITION_TYPE partition =
      (bsize >= BLOCK_8X8) ? get_partition(cm, mi_row, mi_col, bsize)
                           : PARTITION_NONE;
  const BLOCK_SIZE subsize = get_partition_subsize(bsize, partition);
  RD_SEARCH_MACROBLOCK_CONTEXT x_ctx;
  RD_STATS last_part_rdc, none_rdc, chosen_rdc, invalid_rdc;
  BLOCK_SIZE sub_subsize = BLOCK_4X4;
  int splits_below = 0;
  BLOCK_SIZE bs_type = mib[0]->sb_type;
  int do_partition_search = 1;

  if (pc_tree->none == NULL) {
    pc_tree->none =
        av1_alloc_pmc(cm, mi_row, mi_col, bsize, pc_tree, PARTITION_NONE, 0,
                      ss_x, ss_y, &td->shared_coeff_buf);
  }
  PICK_MODE_CONTEXT *ctx_none = pc_tree->none;

  if (mi_row >= cm->mi_rows || mi_col >= cm->mi_cols) return;

#if !CONFIG_EXT_RECUR_PARTITIONS
  assert(mi_size_wide[bsize] == mi_size_high[bsize]);
#endif

  av1_invalid_rd_stats(&last_part_rdc);
  av1_invalid_rd_stats(&none_rdc);
  av1_invalid_rd_stats(&chosen_rdc);
  av1_invalid_rd_stats(&invalid_rdc);

  pc_tree->partitioning = partition;

  xd->above_txfm_context = cm->above_txfm_context[tile_info->tile_row] + mi_col;
  xd->left_txfm_context =
      xd->left_txfm_context_buffer + (mi_row & MAX_MIB_MASK);
  av1_save_context(x, &x_ctx, mi_row, mi_col, bsize, num_planes);

  if (bsize == BLOCK_16X16 && cpi->vaq_refresh) {
    av1_enc_set_offsets(cpi, tile_info, x, mi_row, mi_col, bsize,
                        &pc_tree->chroma_ref_info);
    x->mb_energy = av1_log_block_var(cpi, x, bsize);
  }

  // Save rdmult before it might be changed, so it can be restored later.
  const int orig_rdmult = x->rdmult;
  av1_setup_block_rdmult(cpi, x, mi_row, mi_col, bsize, NO_AQ, NULL);

  if (do_partition_search &&
      cpi->sf.partition_search_type == SEARCH_PARTITION &&
      cpi->sf.adjust_partitioning_from_last_frame) {
    // Check if any of the sub blocks are further split.
    if (partition == PARTITION_SPLIT && subsize > BLOCK_8X8) {
      sub_subsize = get_partition_subsize(subsize, PARTITION_SPLIT);
      splits_below = 1;
      for (int i = 0; i < 4; i++) {
        int jj = i >> 1, ii = i & 0x01;
        MB_MODE_INFO *this_mi = mib[jj * hbh * cm->mi_stride + ii * hbw];
        if (this_mi && this_mi->sb_type >= sub_subsize) {
          splits_below = 0;
        }
      }
    }

    // If partition is not none try none unless each of the 4 splits are split
    // even further..
    if (partition != PARTITION_NONE && !splits_below &&
        mi_row + hbh < cm->mi_rows && mi_col + hbw < cm->mi_cols) {
      pc_tree->partitioning = PARTITION_NONE;
      pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, &none_rdc,
                    PARTITION_NONE, bsize, ctx_none, invalid_rdc, PICK_MODE_RD);

      if (none_rdc.rate < INT_MAX) {
        none_rdc.rate += x->partition_cost[pl][PARTITION_NONE];
        none_rdc.rdcost = RDCOST(x->rdmult, none_rdc.rate, none_rdc.dist);
      }

      av1_restore_context(cm, x, &x_ctx, mi_row, mi_col, bsize, num_planes);
      mib[0]->sb_type = bs_type;
      pc_tree->partitioning = partition;
    }
  }

  for (int i = 0; i < 4; ++i) {
    pc_tree->split[i] = av1_alloc_pc_tree_node(
        mi_row + (i >> 1) * hbh, mi_col + (i & 1) * hbw, subsize, pc_tree,
        PARTITION_SPLIT, i, i == 3, ss_x, ss_y);
  }
  switch (partition) {
    case PARTITION_NONE:
      pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, &last_part_rdc,
                    PARTITION_NONE, bsize, ctx_none, invalid_rdc, PICK_MODE_RD);
      break;
    case PARTITION_HORZ:
#if CONFIG_EXT_RECUR_PARTITIONS
      pc_tree->horizontal[0] = av1_alloc_pc_tree_node(
          mi_row, mi_col, subsize, pc_tree, PARTITION_HORZ, 0, 0, ss_x, ss_y);
      pc_tree->horizontal[1] =
          av1_alloc_pc_tree_node(mi_row + hbh, mi_col, subsize, pc_tree,
                                 PARTITION_HORZ, 1, 1, ss_x, ss_y);
      av1_rd_use_partition(cpi, td, tile_data, mib, tp, mi_row, mi_col, subsize,
                           &last_part_rdc.rate, &last_part_rdc.dist, 1,
                           pc_tree->horizontal[0]);
#else
      for (int i = 0; i < 2; ++i) {
        pc_tree->horizontal[i] =
            av1_alloc_pmc(cm, mi_row + hbh * i, mi_col, subsize, pc_tree,
                          PARTITION_HORZ, i, ss_x, ss_y, &td->shared_coeff_buf);
      }
      pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, &last_part_rdc,
                    PARTITION_HORZ, subsize, pc_tree->horizontal[0],
                    invalid_rdc, PICK_MODE_RD);
#endif  // CONFIG_EXT_RECUR_PARTITIONS
      if (last_part_rdc.rate != INT_MAX && bsize >= BLOCK_8X8 &&
          mi_row + hbh < cm->mi_rows) {
        RD_STATS tmp_rdc;
        av1_init_rd_stats(&tmp_rdc);
#if CONFIG_EXT_RECUR_PARTITIONS
        av1_rd_use_partition(cpi, td, tile_data, mib + hbh * cm->mi_stride, tp,
                             mi_row + hbh, mi_col, subsize, &tmp_rdc.rate,
                             &tmp_rdc.dist, 0, pc_tree->horizontal[1]);
#else
        const PICK_MODE_CONTEXT *const ctx_h = pc_tree->horizontal[0];
        av1_update_state(cpi, td, ctx_h, mi_row, mi_col, subsize, 1);
        av1_encode_superblock(cpi, tile_data, td, tp, DRY_RUN_NORMAL, subsize,
                              NULL);
        pick_sb_modes(cpi, tile_data, x, mi_row + hbh, mi_col, &tmp_rdc,
                      PARTITION_HORZ, subsize, pc_tree->horizontal[1],
                      invalid_rdc, PICK_MODE_RD);
#endif  // CONFIG_EXT_RECUR_PARTITIONS
        if (tmp_rdc.rate == INT_MAX || tmp_rdc.dist == INT64_MAX) {
          av1_invalid_rd_stats(&last_part_rdc);
          break;
        }
        last_part_rdc.rate += tmp_rdc.rate;
        last_part_rdc.dist += tmp_rdc.dist;
        last_part_rdc.rdcost += tmp_rdc.rdcost;
      }
      break;
    case PARTITION_VERT:
#if CONFIG_EXT_RECUR_PARTITIONS
      pc_tree->vertical[0] = av1_alloc_pc_tree_node(
          mi_row, mi_col, subsize, pc_tree, PARTITION_VERT, 0, 0, ss_x, ss_y);
      pc_tree->vertical[1] =
          av1_alloc_pc_tree_node(mi_row, mi_col + hbw, subsize, pc_tree,
                                 PARTITION_VERT, 1, 1, ss_x, ss_y);
      av1_rd_use_partition(cpi, td, tile_data, mib, tp, mi_row, mi_col, subsize,
                           &last_part_rdc.rate, &last_part_rdc.dist, 1,
                           pc_tree->vertical[0]);
#else
      for (int i = 0; i < 2; ++i) {
        pc_tree->vertical[i] =
            av1_alloc_pmc(cm, mi_row, mi_col + hbw * i, subsize, pc_tree,
                          PARTITION_VERT, i, ss_x, ss_y, &td->shared_coeff_buf);
      }
      pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, &last_part_rdc,
                    PARTITION_VERT, subsize, pc_tree->vertical[0], invalid_rdc,
                    PICK_MODE_RD);
#endif  // CONFIG_EXT_RECUR_PARTITIONS
      if (last_part_rdc.rate != INT_MAX && bsize >= BLOCK_8X8 &&
          mi_col + hbw < cm->mi_cols) {
        RD_STATS tmp_rdc;
        av1_init_rd_stats(&tmp_rdc);

#if CONFIG_EXT_RECUR_PARTITIONS
        av1_rd_use_partition(cpi, td, tile_data, mib + hbw, tp, mi_row,
                             mi_col + hbw, subsize, &tmp_rdc.rate,
                             &tmp_rdc.dist, 0, pc_tree->vertical[1]);
#else
        const PICK_MODE_CONTEXT *const ctx_v = pc_tree->vertical[0];
        av1_update_state(cpi, td, ctx_v, mi_row, mi_col, subsize, 1);
        av1_encode_superblock(cpi, tile_data, td, tp, DRY_RUN_NORMAL, subsize,
                              NULL);
        pick_sb_modes(cpi, tile_data, x, mi_row, mi_col + hbw, &tmp_rdc,
                      PARTITION_VERT, subsize,
                      pc_tree->vertical[bsize > BLOCK_8X8], invalid_rdc,
                      PICK_MODE_RD);
#endif  // CONFIG_EXT_RECUR_PARTITIONS
        if (tmp_rdc.rate == INT_MAX || tmp_rdc.dist == INT64_MAX) {
          av1_invalid_rd_stats(&last_part_rdc);
          break;
        }
        last_part_rdc.rate += tmp_rdc.rate;
        last_part_rdc.dist += tmp_rdc.dist;
        last_part_rdc.rdcost += tmp_rdc.rdcost;
      }
      break;
    case PARTITION_SPLIT:
      last_part_rdc.rate = 0;
      last_part_rdc.dist = 0;
      last_part_rdc.rdcost = 0;
      for (int i = 0; i < 4; i++) {
        int x_idx = (i & 1) * hbw;
        int y_idx = (i >> 1) * hbh;
        int jj = i >> 1, ii = i & 0x01;
        RD_STATS tmp_rdc;
        if ((mi_row + y_idx >= cm->mi_rows) || (mi_col + x_idx >= cm->mi_cols))
          continue;

        av1_init_rd_stats(&tmp_rdc);
        av1_rd_use_partition(
            cpi, td, tile_data, mib + jj * hbh * cm->mi_stride + ii * hbw, tp,
            mi_row + y_idx, mi_col + x_idx, subsize, &tmp_rdc.rate,
            &tmp_rdc.dist, i != 3, pc_tree->split[i]);
        if (tmp_rdc.rate == INT_MAX || tmp_rdc.dist == INT64_MAX) {
          av1_invalid_rd_stats(&last_part_rdc);
          break;
        }
        last_part_rdc.rate += tmp_rdc.rate;
        last_part_rdc.dist += tmp_rdc.dist;
      }
      break;
#if CONFIG_EXT_RECUR_PARTITIONS
    case PARTITION_HORZ_3:
    case PARTITION_VERT_3:
#else
    case PARTITION_VERT_A:
    case PARTITION_VERT_B:
    case PARTITION_HORZ_A:
    case PARTITION_HORZ_B:
    case PARTITION_HORZ_4:
    case PARTITION_VERT_4:
#endif  // CONFIG_EXT_RECUR_PARTITIONS
      assert(0 && "Cannot handle extended partition types");
    default: assert(0); break;
  }

  if (last_part_rdc.rate < INT_MAX) {
    last_part_rdc.rate += x->partition_cost[pl][partition];
    last_part_rdc.rdcost =
        RDCOST(x->rdmult, last_part_rdc.rate, last_part_rdc.dist);
  }

  if (do_partition_search && cpi->sf.adjust_partitioning_from_last_frame &&
      cpi->sf.partition_search_type == SEARCH_PARTITION &&
      partition != PARTITION_SPLIT && bsize > BLOCK_8X8 &&
      (mi_row + bh < cm->mi_rows || mi_row + hbh == cm->mi_rows) &&
      (mi_col + bw < cm->mi_cols || mi_col + hbw == cm->mi_cols)) {
    BLOCK_SIZE split_subsize = get_partition_subsize(bsize, PARTITION_SPLIT);
    chosen_rdc.rate = 0;
    chosen_rdc.dist = 0;

    av1_restore_context(cm, x, &x_ctx, mi_row, mi_col, bsize, num_planes);
    pc_tree->partitioning = PARTITION_SPLIT;

    // Split partition.
    for (int i = 0; i < 4; i++) {
      int x_idx = (i & 1) * hbw;
      int y_idx = (i >> 1) * hbh;
      RD_STATS tmp_rdc;

      if ((mi_row + y_idx >= cm->mi_rows) || (mi_col + x_idx >= cm->mi_cols))
        continue;

      av1_save_context(x, &x_ctx, mi_row, mi_col, bsize, num_planes);
      pc_tree->split[i]->partitioning = PARTITION_NONE;
      if (pc_tree->split[i]->none != NULL)
        pc_tree->split[i]->none =
            av1_alloc_pmc(cm, mi_row + y_idx, mi_col + x_idx, split_subsize,
                          pc_tree->split[i], PARTITION_NONE, 0, ss_x, ss_y,
                          &td->shared_coeff_buf);
      pick_sb_modes(cpi, tile_data, x, mi_row + y_idx, mi_col + x_idx, &tmp_rdc,
                    PARTITION_SPLIT, split_subsize, pc_tree->split[i]->none,
                    invalid_rdc, PICK_MODE_RD);

      av1_restore_context(cm, x, &x_ctx, mi_row, mi_col, bsize, num_planes);
      if (tmp_rdc.rate == INT_MAX || tmp_rdc.dist == INT64_MAX) {
        av1_invalid_rd_stats(&chosen_rdc);
        break;
      }

      chosen_rdc.rate += tmp_rdc.rate;
      chosen_rdc.dist += tmp_rdc.dist;

      if (i != 3)
        av1_encode_sb(cpi, td, tile_data, tp, mi_row + y_idx, mi_col + x_idx,
                      DRY_RUN_NORMAL, split_subsize, pc_tree->split[i], NULL,
                      NULL);

      chosen_rdc.rate += x->partition_cost[pl][PARTITION_NONE];
    }
    if (chosen_rdc.rate < INT_MAX) {
#if !(CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT)
      chosen_rdc.rate += x->partition_cost[pl][PARTITION_SPLIT];
#endif  // !(CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT)
      chosen_rdc.rdcost = RDCOST(x->rdmult, chosen_rdc.rate, chosen_rdc.dist);
    }
  }

  // If last_part is better set the partitioning to that.
  if (last_part_rdc.rdcost < chosen_rdc.rdcost) {
    mib[0]->sb_type = bsize;
    if (bsize >= BLOCK_8X8) pc_tree->partitioning = partition;
    chosen_rdc = last_part_rdc;
  }
  // If none was better set the partitioning to that.
  if (none_rdc.rdcost < chosen_rdc.rdcost) {
    if (bsize >= BLOCK_8X8) pc_tree->partitioning = PARTITION_NONE;
    chosen_rdc = none_rdc;
  }

  av1_restore_context(cm, x, &x_ctx, mi_row, mi_col, bsize, num_planes);

  // We must have chosen a partitioning and encoding or we'll fail later on.
  // No other opportunities for success.
  if (bsize == cm->seq_params.sb_size)
    assert(chosen_rdc.rate < INT_MAX && chosen_rdc.dist < INT64_MAX);

  if (do_recon) {
    if (bsize == cm->seq_params.sb_size) {
      // NOTE: To get estimate for rate due to the tokens, use:
      // int rate_coeffs = 0;
      // encode_sb(cpi, td, tile_data, tp, mi_row, mi_col, DRY_RUN_COSTCOEFFS,
      //           bsize, pc_tree, &rate_coeffs);
      x->cb_offset = 0;
      av1_reset_ptree_in_sbi(xd->sbi);
      av1_encode_sb(cpi, td, tile_data, tp, mi_row, mi_col, OUTPUT_ENABLED,
                    bsize, pc_tree, xd->sbi->ptree_root, NULL);
    } else {
      av1_encode_sb(cpi, td, tile_data, tp, mi_row, mi_col, DRY_RUN_NORMAL,
                    bsize, pc_tree, NULL, NULL);
    }
  }

  *rate = chosen_rdc.rate;
  *dist = chosen_rdc.dist;
  x->rdmult = orig_rdmult;
}
#if !CONFIG_EXT_RECUR_PARTITIONS
// Try searching for an encoding for the given subblock. Returns zero if the
// rdcost is already too high (to tell the caller not to bother searching for
// encodings of further subblocks)
static int rd_try_subblock(AV1_COMP *const cpi, ThreadData *td,
                           TileDataEnc *tile_data, TOKENEXTRA **tp, int is_last,
                           int mi_row, int mi_col, BLOCK_SIZE subsize,
                           RD_STATS best_rdcost, RD_STATS *sum_rdc,
                           PARTITION_TYPE partition,
                           PICK_MODE_CONTEXT *this_ctx) {
  MACROBLOCK *const x = &td->mb;
  const int orig_mult = x->rdmult;
  av1_setup_block_rdmult(cpi, x, mi_row, mi_col, subsize, NO_AQ, NULL);

  av1_rd_cost_update(x->rdmult, &best_rdcost);

  RD_STATS rdcost_remaining;
  av1_rd_stats_subtraction(x->rdmult, &best_rdcost, sum_rdc, &rdcost_remaining);
  RD_STATS this_rdc;
  pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, &this_rdc, partition,
                subsize, this_ctx, rdcost_remaining, PICK_MODE_RD);

  if (this_rdc.rate == INT_MAX) {
    sum_rdc->rdcost = INT64_MAX;
  } else {
    sum_rdc->rate += this_rdc.rate;
    sum_rdc->dist += this_rdc.dist;
    av1_rd_cost_update(x->rdmult, sum_rdc);
  }

  if (sum_rdc->rdcost >= best_rdcost.rdcost) {
    x->rdmult = orig_mult;
    return 0;
  }

  if (!is_last) {
    av1_update_state(cpi, td, this_ctx, mi_row, mi_col, subsize, 1);
    av1_encode_superblock(cpi, tile_data, td, tp, DRY_RUN_NORMAL, subsize,
                          NULL);
  }

  x->rdmult = orig_mult;
  return 1;
}
#endif  // !CONFIG_EXT_RECUR_PARTITIONS

#if CONFIG_EXT_RECUR_PARTITIONS
typedef struct {
  SIMPLE_MOTION_DATA_TREE *sms_tree;
  PC_TREE *pc_tree;
  PICK_MODE_CONTEXT *ctx;
  int mi_row;
  int mi_col;
  BLOCK_SIZE bsize;
  PARTITION_TYPE partition;
  int is_last_subblock;
  int is_splittable;
  int max_sq_part;
  int min_sq_part;
} SUBBLOCK_RDO_DATA;

// Try searching for an encoding for the given subblock. Returns zero if the
// rdcost is already too high (to tell the caller not to bother searching for
// encodings of further subblocks)
static int rd_try_subblock_new(AV1_COMP *const cpi, ThreadData *td,
                               TileDataEnc *tile_data, TOKENEXTRA **tp,
                               SUBBLOCK_RDO_DATA *rdo_data,
                               RD_STATS best_rdcost, RD_STATS *sum_rdc,
                               SB_MULTI_PASS_MODE multi_pass_mode) {
  MACROBLOCK *const x = &td->mb;
  const int orig_mult = x->rdmult;
  const int mi_row = rdo_data->mi_row;
  const int mi_col = rdo_data->mi_col;
  const BLOCK_SIZE bsize = rdo_data->bsize;

  av1_setup_block_rdmult(cpi, x, mi_row, mi_col, bsize, NO_AQ, NULL);

  av1_rd_cost_update(x->rdmult, &best_rdcost);

  RD_STATS rdcost_remaining;
  av1_rd_stats_subtraction(x->rdmult, &best_rdcost, sum_rdc, &rdcost_remaining);
  RD_STATS this_rdc;

  if (rdo_data->is_splittable) {
    if (!av1_rd_pick_partition(cpi, td, tile_data, tp, mi_row, mi_col, bsize,
                               rdo_data->max_sq_part, rdo_data->min_sq_part,
                               &this_rdc, rdcost_remaining, rdo_data->pc_tree,
                               rdo_data->sms_tree, NULL, multi_pass_mode))
      return 0;
  } else {
#if USE_OLD_PREDICTION_MODE
    const BLOCK_SIZE sb_size = cpi->common.seq_params.sb_size;
    SimpleMotionData *sms_data =
        av1_get_sms_data_entry(x->sms_bufs, mi_row, mi_col, bsize, sb_size);
    av1_set_best_mode_cache(x, sms_data->mode_cache);
#endif  // USE_OLD_PREDICTION_MODE
    pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, &this_rdc,
                  rdo_data->partition, bsize, rdo_data->ctx, rdcost_remaining,
                  PICK_MODE_RD);
#if USE_OLD_PREDICTION_MODE
    x->inter_mode_cache = NULL;
    x->reuse_inter_mode_cache_type = 0;
    if (this_rdc.rate != INT_MAX) {
      av1_add_mode_search_context_to_cache(sms_data, rdo_data->ctx);
    }
#endif  // USE_OLD_PREDICTION_MODE
  }

  if (this_rdc.rate == INT_MAX) {
    sum_rdc->rdcost = INT64_MAX;
  } else {
    sum_rdc->rate += this_rdc.rate;
    sum_rdc->dist += this_rdc.dist;
    av1_rd_cost_update(x->rdmult, sum_rdc);
  }

  if (sum_rdc->rdcost >= best_rdcost.rdcost) {
    x->rdmult = orig_mult;
    return 0;
  }

  if (!rdo_data->is_last_subblock && !rdo_data->is_splittable) {
    av1_update_state(cpi, td, rdo_data->ctx, mi_row, mi_col, bsize, 1);
    av1_encode_superblock(cpi, tile_data, td, tp, DRY_RUN_NORMAL, bsize, NULL);
  }

  x->rdmult = orig_mult;
  return 1;
}
#else   // !CONFIG_EXT_RECUR_PARTITIONS
static bool rd_test_partition3(AV1_COMP *const cpi, ThreadData *td,
                               TileDataEnc *tile_data, TOKENEXTRA **tp,
                               PC_TREE *pc_tree, RD_STATS *best_rdc,
                               PICK_MODE_CONTEXT *ctxs[3], int mi_row,
                               int mi_col, BLOCK_SIZE bsize,
                               PARTITION_TYPE partition, int mi_row0,
                               int mi_col0, BLOCK_SIZE subsize0, int mi_row1,
                               int mi_col1, BLOCK_SIZE subsize1, int mi_row2,
                               int mi_col2, BLOCK_SIZE subsize2) {
  const MACROBLOCK *const x = &td->mb;
  const MACROBLOCKD *const xd = &x->e_mbd;
  const int pl = partition_plane_context(xd, mi_row, mi_col, bsize);
  RD_STATS sum_rdc;
  av1_init_rd_stats(&sum_rdc);
  sum_rdc.rate = x->partition_cost[pl][partition];
  sum_rdc.rdcost = RDCOST(x->rdmult, sum_rdc.rate, 0);
  if (!rd_try_subblock(cpi, td, tile_data, tp, 0, mi_row0, mi_col0, subsize0,
                       *best_rdc, &sum_rdc, partition, ctxs[0]))
    return false;

  if (!rd_try_subblock(cpi, td, tile_data, tp, 0, mi_row1, mi_col1, subsize1,
                       *best_rdc, &sum_rdc, partition, ctxs[1]))
    return false;

  if (!rd_try_subblock(cpi, td, tile_data, tp, 1, mi_row2, mi_col2, subsize2,
                       *best_rdc, &sum_rdc, partition, ctxs[2]))
    return false;

  av1_rd_cost_update(x->rdmult, &sum_rdc);
  if (sum_rdc.rdcost >= best_rdc->rdcost) return false;
  sum_rdc.rdcost = RDCOST(x->rdmult, sum_rdc.rate, sum_rdc.dist);
  if (sum_rdc.rdcost >= best_rdc->rdcost) return false;

  *best_rdc = sum_rdc;
  pc_tree->partitioning = partition;
  return true;
}
#endif  // CONFIG_EXT_RECUR_PARTITIONS

static INLINE void prune_partitions_after_none(
    PartitionSearchState *search_state, AV1_COMP *const cpi, MACROBLOCK *x,
    SIMPLE_MOTION_DATA_TREE *sms_tree, const PICK_MODE_CONTEXT *ctx_none,
    const RD_STATS *this_rdc, unsigned int *pb_source_variance) {
  const AV1_COMMON *const cm = &cpi->common;
  MACROBLOCKD *const xd = &x->e_mbd;

  const PartitionBlkParams *blk_params = &search_state->part_blk_params;
  const BLOCK_SIZE bsize = blk_params->bsize;

#if CONFIG_EXT_RECUR_PARTITIONS
  (void)sms_tree;
#else   // !CONFIG_EXT_RECUR_PARTITIONS
  const int mi_row = blk_params->mi_row, mi_col = blk_params->mi_col;
#endif  // CONFIG_EXT_RECUR_PARTITIONS

  if (!frame_is_intra_only(cm) &&
#if CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT
      search_state->do_rectangular_split &&
#else   // CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT
      (search_state->do_square_split || search_state->do_rectangular_split) &&
#endif  // CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT
      !x->e_mbd.lossless[xd->mi[0]->segment_id] && ctx_none->skippable) {
    const int use_ml_based_breakout =
#if CONFIG_EXT_RECUR_PARTITIONS
        is_square_block(bsize) &&
#endif  // CONFIG_EXT_RECUR_PARTITIONS
        bsize <= cpi->sf.use_square_partition_only_threshold &&
        bsize > BLOCK_4X4 && xd->bd == 8;
    if (use_ml_based_breakout) {
      if (av1_ml_predict_breakout(cpi, bsize, x, this_rdc,
                                  *pb_source_variance)) {
#if !(CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT)
        search_state->do_square_split = 0;
#endif  // !(CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT)
        search_state->do_rectangular_split = 0;
      }
    }

    // Adjust dist breakout threshold according to the partition size.
    const int64_t dist_breakout_thr =
        cpi->sf.partition_search_breakout_dist_thr >>
        ((2 * (MAX_SB_SIZE_LOG2 - 2)) -
         (mi_size_wide_log2[bsize] + mi_size_high_log2[bsize]));
    const int rate_breakout_thr = cpi->sf.partition_search_breakout_rate_thr *
                                  num_pels_log2_lookup[bsize];

    // If all y, u, v transform blocks in this partition are skippable,
    // and the dist & rate are within the thresholds, the partition
    // search is terminated for current branch of the partition search
    // tree. The dist & rate thresholds are set to 0 at speed 0 to
    // disable the early termination at that speed.
    if (this_rdc->dist < dist_breakout_thr &&
        this_rdc->rate < rate_breakout_thr) {
#if !(CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT)
      search_state->do_square_split = 0;
#endif  // !(CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT)
      search_state->do_rectangular_split = 0;
    }
  }

#if !CONFIG_EXT_RECUR_PARTITIONS
  if (cpi->sf.simple_motion_search_early_term_none && cm->show_frame &&
      !frame_is_intra_only(cm) && bsize >= BLOCK_16X16 &&
      mi_row + blk_params->mi_step_h < cm->mi_rows &&
      mi_col + blk_params->mi_step_w < cm->mi_cols &&
      this_rdc->rdcost < INT64_MAX && this_rdc->rdcost >= 0 &&
      this_rdc->rate < INT_MAX && this_rdc->rate >= 0 &&
      (search_state->do_square_split || search_state->do_rectangular_split)) {
    av1_simple_motion_search_early_term_none(
        cpi, x, sms_tree, mi_row, mi_col, bsize, this_rdc,
        &search_state->terminate_partition_search);
  }
#endif  // !CONFIG_EXT_RECUR_PARTITIONS
}

static INLINE void search_partition_none(
    PartitionSearchState *search_state, AV1_COMP *const cpi, ThreadData *td,
    TileDataEnc *tile_data, RD_STATS *best_rdc, PC_TREE *pc_tree,
    SIMPLE_MOTION_DATA_TREE *sms_tree, RD_SEARCH_MACROBLOCK_CONTEXT *x_ctx,
    unsigned int *pb_source_variance, int64_t *none_rd, int64_t *part_none_rd) {
  const AV1_COMMON *const cm = &cpi->common;
  MACROBLOCK *const x = &td->mb;
  const int num_planes = av1_num_planes(cm);

#if CONFIG_EXT_RECUR_PARTITIONS
  (void)part_none_rd;
#endif  // CONFIG_EXT_RECUR_PARTITIONS

  const PartitionBlkParams *blk_params = &search_state->part_blk_params;
  const int mi_row = blk_params->mi_row, mi_col = blk_params->mi_col;
  const BLOCK_SIZE bsize = blk_params->bsize;

  RD_STATS this_rdc;

  pc_tree->none =
      av1_alloc_pmc(cm, mi_row, mi_col, bsize, pc_tree, PARTITION_NONE, 0,
                    blk_params->ss_x, blk_params->ss_y, &td->shared_coeff_buf);
  PICK_MODE_CONTEXT *ctx_none = pc_tree->none;

  if (blk_params->is_le_min_sq_part && blk_params->has_rows &&
      blk_params->has_cols)
    search_state->partition_none_allowed = 1;
  if (search_state->terminate_partition_search ||
      !search_state->partition_none_allowed || blk_params->is_gt_max_sq_part) {
    return;
  }

  int pt_cost = 0;
  if (search_state->is_block_splittable) {
    pt_cost = search_state->partition_cost[PARTITION_NONE] < INT_MAX
                  ? search_state->partition_cost[PARTITION_NONE]
                  : 0;
  }
  RD_STATS partition_rdcost;
  av1_init_rd_stats(&partition_rdcost);
  partition_rdcost.rate = pt_cost;
  av1_rd_cost_update(x->rdmult, &partition_rdcost);
  RD_STATS best_remain_rdcost;
  av1_rd_stats_subtraction(x->rdmult, best_rdc, &partition_rdcost,
                           &best_remain_rdcost);
#if CONFIG_COLLECT_PARTITION_STATS
  if (best_remain_rdcost >= 0) {
    partition_attempts[PARTITION_NONE] += 1;
    aom_usec_timer_start(&partition_timer);
    partition_timer_on = 1;
  }
#endif
#if CONFIG_EXT_RECUR_PARTITIONS && USE_OLD_PREDICTION_MODE
  SimpleMotionData *sms_data = av1_get_sms_data_entry(
      x->sms_bufs, mi_row, mi_col, bsize, cm->seq_params.sb_size);
  av1_set_best_mode_cache(x, sms_data->mode_cache);
#endif  // CONFIG_EXT_RECUR_PARTITIONS && USE_OLD_PREDICTION_MODE
  pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, &this_rdc, PARTITION_NONE,
                bsize, ctx_none, best_remain_rdcost, PICK_MODE_RD);
#if CONFIG_EXT_RECUR_PARTITIONS && USE_OLD_PREDICTION_MODE
  x->inter_mode_cache = NULL;
  x->reuse_inter_mode_cache_type = 0;
  if (this_rdc.rate != INT_MAX) {
    av1_add_mode_search_context_to_cache(sms_data, ctx_none);
  }
#endif  // CONFIG_EXT_RECUR_PARTITIONS && USE_OLD_PREDICTION_MODE
  av1_rd_cost_update(x->rdmult, &this_rdc);
#if CONFIG_COLLECT_PARTITION_STATS
  if (partition_timer_on) {
    aom_usec_timer_mark(&partition_timer);
    int64_t time = aom_usec_timer_elapsed(&partition_timer);
    partition_times[PARTITION_NONE] += time;
    partition_timer_on = 0;
  }
#endif
  *pb_source_variance = x->source_variance;
  if (none_rd) *none_rd = this_rdc.rdcost;
  search_state->none_rd = this_rdc.rdcost;

  // Whether the search produced a valid result.
  if (this_rdc.rate != INT_MAX) {
    // Record picked ref frame to prune ref frames for other partition types.
    if (cpi->sf.prune_ref_frame_for_rect_partitions) {
      const int ref_type = av1_ref_frame_type(ctx_none->mic.ref_frame);
      av1_update_picked_ref_frames_mask(
          x, ref_type, bsize, cm->seq_params.mib_size, mi_row, mi_col);
    }

    // Calculate the total cost and update the best partition.
    if (search_state->is_block_splittable) {
      this_rdc.rate += pt_cost;
      this_rdc.rdcost = RDCOST(x->rdmult, this_rdc.rate, this_rdc.dist);
    }

#if !CONFIG_EXT_RECUR_PARTITIONS
    *part_none_rd = this_rdc.rdcost;
#endif  // !CONFIG_EXT_RECUR_PARTITIONS
    if (this_rdc.rdcost < best_rdc->rdcost) {
      // Update if the current partition is the best
      *best_rdc = this_rdc;
      search_state->found_best_partition = true;
      pc_tree->partitioning = PARTITION_NONE;

      prune_partitions_after_none(search_state, cpi, x, sms_tree, ctx_none,
                                  &this_rdc, pb_source_variance);
    }
  }

  av1_restore_context(cm, x, x_ctx, mi_row, mi_col, bsize, num_planes);
}

#if !(CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT)
static INLINE void search_partition_split(
    PartitionSearchState *search_state, AV1_COMP *const cpi, ThreadData *td,
    TileDataEnc *tile_data, TOKENEXTRA **tp, RD_STATS *best_rdc,
    int64_t *part_split_rd, PC_TREE *pc_tree, SIMPLE_MOTION_DATA_TREE *sms_tree,
    RD_SEARCH_MACROBLOCK_CONTEXT *x_ctx, SB_MULTI_PASS_MODE multi_pass_mode) {
  const AV1_COMMON *const cm = &cpi->common;
  MACROBLOCK *const x = &td->mb;
  const int num_planes = av1_num_planes(cm);

  const PartitionBlkParams *blk_params = &search_state->part_blk_params;
  const int mi_row = blk_params->mi_row, mi_col = blk_params->mi_col;
  const BLOCK_SIZE max_sq_part = blk_params->max_sq_part;
  const BLOCK_SIZE min_sq_part = blk_params->min_sq_part;
  const BLOCK_SIZE bsize = blk_params->bsize;
  const BLOCK_SIZE subsize = get_partition_subsize(bsize, PARTITION_SPLIT);

#if CONFIG_EXT_RECUR_PARTITIONS && KEEP_PARTITION_SPLIT
  const int search_split = !search_state->terminate_partition_search &&
                           search_state->do_square_split &&
                           is_partition_valid(bsize, PARTITION_SPLIT);
#elif !CONFIG_EXT_RECUR_PARTITIONS
  const int search_split = (!search_state->terminate_partition_search &&
                            search_state->do_square_split) ||
                           blk_params->is_gt_max_sq_part;
#endif  // CONFIG_EXT_RECUR_PARTITIONS
#if !CONFIG_EXT_RECUR_PARTITIONS || KEEP_PARTITION_SPLIT
  if (!search_split) {
    return;
  }
#endif  // !CONFIF_EXT_RECUR_PARTITIONS || KEEP_PARTITION_SPLIT

  for (int i = 0; i < 4; ++i) {
    const int x_idx = (i & 1) * blk_params->mi_step_w;
    const int y_idx = (i >> 1) * blk_params->mi_step_h;
    pc_tree->split[i] = av1_alloc_pc_tree_node(
        mi_row + y_idx, mi_col + x_idx, subsize, pc_tree, PARTITION_SPLIT, i,
        i == 3, blk_params->ss_x, blk_params->ss_y);
  }

  RD_STATS sum_rdc;
  av1_init_rd_stats(&sum_rdc);
  sum_rdc.rate = search_state->partition_cost[PARTITION_SPLIT];
  sum_rdc.rdcost = RDCOST(x->rdmult, sum_rdc.rate, 0);

  int idx;
  for (idx = 0; idx < 4 && sum_rdc.rdcost < best_rdc->rdcost; ++idx) {
    const int x_idx = (idx & 1) * blk_params->mi_step_w;
    const int y_idx = (idx >> 1) * blk_params->mi_step_h;

    if (mi_row + y_idx >= cm->mi_rows || mi_col + x_idx >= cm->mi_cols)
      continue;

    RD_STATS this_rdc;
    int64_t *p_split_rd = &search_state->split_rd[idx];

    RD_STATS best_remain_rdcost;
    av1_rd_stats_subtraction(x->rdmult, best_rdc, &sum_rdc,
                             &best_remain_rdcost);

    int curr_quad_tree_idx = 0;
    if (frame_is_intra_only(cm) && bsize <= BLOCK_64X64) {
      curr_quad_tree_idx = x->quad_tree_idx;
      x->quad_tree_idx = 4 * curr_quad_tree_idx + idx + 1;
    }
    if (!av1_rd_pick_partition(cpi, td, tile_data, tp, mi_row + y_idx,
                               mi_col + x_idx, subsize, max_sq_part,
                               min_sq_part, &this_rdc, best_remain_rdcost,
                               pc_tree->split[idx],
#if CONFIG_EXT_RECUR_PARTITIONS
                               sms_tree ? sms_tree->split[idx] : NULL,
                               p_split_rd, multi_pass_mode)) {
#else
                               sms_tree->split[idx], p_split_rd,
                               multi_pass_mode)) {
#endif  // CONFIG_EXT_RECUR_PARTITIONS
      av1_invalid_rd_stats(&sum_rdc);
      break;
    }
    if (frame_is_intra_only(cm) && bsize <= BLOCK_64X64) {
      x->quad_tree_idx = curr_quad_tree_idx;
    }

    sum_rdc.rate += this_rdc.rate;
    sum_rdc.dist += this_rdc.dist;
    av1_rd_cost_update(x->rdmult, &sum_rdc);
#if !CONFIG_EXT_RECUR_PARTITIONS
    if (idx <= 1 && (bsize <= BLOCK_8X8 ||
                     pc_tree->split[idx]->partitioning == PARTITION_NONE)) {
      const MB_MODE_INFO *const mbmi = &pc_tree->split[idx]->none->mic;
      const PALETTE_MODE_INFO *const pmi = &mbmi->palette_mode_info;
      // Neither palette mode nor cfl predicted
      if (pmi->palette_size[0] == 0 && pmi->palette_size[1] == 0) {
        if (mbmi->uv_mode != UV_CFL_PRED)
          search_state->split_ctx_is_ready[idx] = 1;
      }
    }
#endif  // !CONFIG_EXT_RECUR_PARTITIONS
  }
#if CONFIG_COLLECT_PARTITION_STATS
  if (partition_timer_on) {
    aom_usec_timer_mark(&partition_timer);
    int64_t time = aom_usec_timer_elapsed(&partition_timer);
    partition_times[PARTITION_SPLIT] += time;
    partition_timer_on = 0;
  }
#endif
  const int reached_last_index = (idx == 4);

  *part_split_rd = sum_rdc.rdcost;
  if (reached_last_index && sum_rdc.rdcost < best_rdc->rdcost) {
    sum_rdc.rdcost = RDCOST(x->rdmult, sum_rdc.rate, sum_rdc.dist);
    if (sum_rdc.rdcost < best_rdc->rdcost) {
      *best_rdc = sum_rdc;
      search_state->found_best_partition = true;
      pc_tree->partitioning = PARTITION_SPLIT;
    }
  } else if (cpi->sf.less_rectangular_check_level > 0) {
    // Skip rectangular partition test when partition type none gives better
    // rd than partition type split.
    if (cpi->sf.less_rectangular_check_level == 2 || idx <= 2) {
      const int partition_none_valid = search_state->none_rd > 0;
      const int partition_none_better = search_state->none_rd < sum_rdc.rdcost;
      search_state->do_rectangular_split &=
          !(partition_none_valid && partition_none_better);
    }
  }

  av1_restore_context(cm, x, x_ctx, mi_row, mi_col, bsize, num_planes);
}
#endif  // !(CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT)

static INLINE void search_partition_horz(PartitionSearchState *search_state,
                                         AV1_COMP *const cpi, ThreadData *td,
                                         TileDataEnc *tile_data,
                                         TOKENEXTRA **tp, RD_STATS *best_rdc,
                                         PC_TREE *pc_tree,
                                         RD_SEARCH_MACROBLOCK_CONTEXT *x_ctx,
                                         SB_MULTI_PASS_MODE multi_pass_mode) {
  const AV1_COMMON *const cm = &cpi->common;
  MACROBLOCK *const x = &td->mb;
  const int num_planes = av1_num_planes(cm);

  const PartitionBlkParams *blk_params = &search_state->part_blk_params;
  const int mi_row = blk_params->mi_row, mi_col = blk_params->mi_col;
#if CONFIG_EXT_RECUR_PARTITIONS
  const TileInfo *const tile_info = &tile_data->tile_info;
  const BLOCK_SIZE max_sq_part = blk_params->max_sq_part;
  const BLOCK_SIZE min_sq_part = blk_params->min_sq_part;
#else
  (void)multi_pass_mode;
#endif  // CONFIG_EXT_RECUR_PARTITIONS
  const BLOCK_SIZE bsize = blk_params->bsize;

  const BLOCK_SIZE subsize = get_partition_subsize(bsize, PARTITION_HORZ);

  RD_STATS sum_rdc, this_rdc;
  av1_init_rd_stats(&sum_rdc);
  assert(IMPLIES(!cpi->oxcf.enable_rect_partitions,
                 !search_state->partition_rect_allowed[HORZ]));

  if (search_state->terminate_partition_search ||
      !search_state->partition_rect_allowed[HORZ] ||
      search_state->prune_rect_part[HORZ] ||
#if CONFIG_EXT_RECUR_PARTITIONS
      !is_partition_valid(bsize, PARTITION_HORZ) ||
#endif  // CONFIG_EXT_RECUR_PARTITIONS
      (!search_state->do_rectangular_split &&
       !av1_active_h_edge(cpi, mi_row, blk_params->mi_step_h)) ||
      blk_params->is_gt_max_sq_part) {
    return;
  }

  const int part_h_rate = search_state->partition_cost[PARTITION_HORZ];
  if (part_h_rate == INT_MAX ||
      RDCOST(x->rdmult, part_h_rate, 0) >= best_rdc->rdcost) {
    return;
  }
#if CONFIG_EXT_RECUR_PARTITIONS
  pc_tree->horizontal[0] =
      av1_alloc_pc_tree_node(mi_row, mi_col, subsize, pc_tree, PARTITION_HORZ,
                             0, 0, blk_params->ss_x, blk_params->ss_y);
  pc_tree->horizontal[1] = av1_alloc_pc_tree_node(
      mi_row + blk_params->mi_step_h, mi_col, subsize, pc_tree, PARTITION_HORZ,
      1, 1, blk_params->ss_x, blk_params->ss_y);

  if (ENABLE_FAST_RECUR_PARTITION && !frame_is_intra_only(cm) &&
      !x->must_find_valid_partition) {
    SMSPartitionStats part_data;
    const SimpleMotionData *up =
        av1_get_sms_data(cpi, tile_info, x, mi_row, mi_col, subsize);
    const SimpleMotionData *down = av1_get_sms_data(
        cpi, tile_info, x, mi_row + blk_params->mi_step_h, mi_col, subsize);
    part_data.sms_data[0] = up;
    part_data.sms_data[1] = down;
    part_data.num_sub_parts = 2;
    part_data.part_rate = part_h_rate;

    if (search_state->none_rd > 0 && search_state->none_rd < INT64_MAX &&
        (mi_row + 2 * blk_params->mi_step_h <= cm->mi_rows) &&
        (mi_col + 2 * blk_params->mi_step_w <= cm->mi_cols) &&
        av1_prune_new_part(&search_state->none_data, &part_data, x->rdmult)) {
      return;
    }
  }
#else
  for (int i = 0; i < 2; ++i) {
    pc_tree->horizontal[i] =
        av1_alloc_pmc(cm, mi_row + blk_params->mi_step_h * i, mi_col, subsize,
                      pc_tree, PARTITION_HORZ, i, blk_params->ss_x,
                      blk_params->ss_y, &td->shared_coeff_buf);
  }
#endif  // CONFIG_EXT_RECUR_PARTITIONS

  // Search first part
  sum_rdc.rate = search_state->partition_cost[PARTITION_HORZ];
  sum_rdc.rdcost = RDCOST(x->rdmult, sum_rdc.rate, 0);
  RD_STATS best_remain_rdcost;
  av1_rd_stats_subtraction(x->rdmult, best_rdc, &sum_rdc, &best_remain_rdcost);
#if CONFIG_EXT_RECUR_PARTITIONS
  av1_rd_pick_partition(cpi, td, tile_data, tp, mi_row, mi_col, subsize,
                        max_sq_part, min_sq_part, &this_rdc, best_remain_rdcost,
                        pc_tree->horizontal[0], NULL, NULL, multi_pass_mode);
#else
  pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, &this_rdc, PARTITION_HORZ,
                subsize, pc_tree->horizontal[0], best_remain_rdcost,
                PICK_MODE_RD);
#endif  // CONFIG_EXT_RECUR_PARTITIONS
  av1_rd_cost_update(x->rdmult, &this_rdc);

  if (this_rdc.rate == INT_MAX) {
    sum_rdc.rdcost = INT64_MAX;
  } else {
    sum_rdc.rate += this_rdc.rate;
    sum_rdc.dist += this_rdc.dist;
    av1_rd_cost_update(x->rdmult, &sum_rdc);
  }
  search_state->rect_part_rd[HORZ][0] = this_rdc.rdcost;

  if (sum_rdc.rdcost < best_rdc->rdcost && blk_params->has_rows) {
#if !CONFIG_EXT_RECUR_PARTITIONS
    const PICK_MODE_CONTEXT *const ctx_h = pc_tree->horizontal[0];
    const MB_MODE_INFO *const mbmi = &pc_tree->horizontal[0]->mic;
    const PALETTE_MODE_INFO *const pmi = &mbmi->palette_mode_info;
    // Neither palette mode nor cfl predicted
    if (pmi->palette_size[0] == 0 && pmi->palette_size[1] == 0) {
      if (mbmi->uv_mode != UV_CFL_PRED)
        search_state->rect_ctx_is_ready[HORZ] = 1;
    }
    av1_update_state(cpi, td, ctx_h, mi_row, mi_col, subsize, 1);
    av1_encode_superblock(cpi, tile_data, td, tp, DRY_RUN_NORMAL, subsize,
                          NULL);
#endif  // !CONFIG_EXT_RECUR_PARTITIONS

    av1_rd_stats_subtraction(x->rdmult, best_rdc, &sum_rdc,
                             &best_remain_rdcost);

    // Search second part
#if CONFIG_EXT_RECUR_PARTITIONS
    av1_rd_pick_partition(
        cpi, td, tile_data, tp, mi_row + blk_params->mi_step_h, mi_col, subsize,
        max_sq_part, min_sq_part, &this_rdc, best_remain_rdcost,
        pc_tree->horizontal[1], NULL, NULL, multi_pass_mode);
#else
    pick_sb_modes(cpi, tile_data, x, mi_row + blk_params->mi_step_h, mi_col,
                  &this_rdc, PARTITION_HORZ, subsize, pc_tree->horizontal[1],
                  best_remain_rdcost, PICK_MODE_RD);
#endif  // CONFIG_EXT_RECUR_PARTITIONS
    av1_rd_cost_update(x->rdmult, &this_rdc);
    search_state->rect_part_rd[HORZ][1] = this_rdc.rdcost;

    if (this_rdc.rate == INT_MAX) {
      sum_rdc.rdcost = INT64_MAX;
    } else {
      sum_rdc.rate += this_rdc.rate;
      sum_rdc.dist += this_rdc.dist;
      av1_rd_cost_update(x->rdmult, &sum_rdc);
    }
  }

  if (sum_rdc.rdcost < best_rdc->rdcost) {
    sum_rdc.rdcost = RDCOST(x->rdmult, sum_rdc.rate, sum_rdc.dist);
    if (sum_rdc.rdcost < best_rdc->rdcost) {
      *best_rdc = sum_rdc;
      search_state->found_best_partition = true;
      pc_tree->partitioning = PARTITION_HORZ;
    }
  }

  av1_restore_context(cm, x, x_ctx, mi_row, mi_col, bsize, num_planes);
}

static INLINE void search_partition_vert(PartitionSearchState *search_state,
                                         AV1_COMP *const cpi, ThreadData *td,
                                         TileDataEnc *tile_data,
                                         TOKENEXTRA **tp, RD_STATS *best_rdc,
                                         PC_TREE *pc_tree,
                                         RD_SEARCH_MACROBLOCK_CONTEXT *x_ctx,
                                         SB_MULTI_PASS_MODE multi_pass_mode) {
  const AV1_COMMON *const cm = &cpi->common;
  MACROBLOCK *const x = &td->mb;
  const int num_planes = av1_num_planes(cm);

  const PartitionBlkParams *blk_params = &search_state->part_blk_params;
  const int mi_row = blk_params->mi_row, mi_col = blk_params->mi_col;
#if CONFIG_EXT_RECUR_PARTITIONS
  const TileInfo *const tile_info = &tile_data->tile_info;
  const BLOCK_SIZE max_sq_part = blk_params->max_sq_part;
  const BLOCK_SIZE min_sq_part = blk_params->min_sq_part;
#else
  (void)multi_pass_mode;
#endif  // CONFIG_EXT_RECUR_PARTITIONS
  const BLOCK_SIZE bsize = blk_params->bsize;

  const BLOCK_SIZE subsize = get_partition_subsize(bsize, PARTITION_VERT);

  RD_STATS sum_rdc, this_rdc;
  av1_init_rd_stats(&sum_rdc);

  assert(IMPLIES(!cpi->oxcf.enable_rect_partitions,
                 !search_state->partition_rect_allowed[VERT]));

  if (search_state->terminate_partition_search ||
      !search_state->partition_rect_allowed[VERT] ||
      search_state->prune_rect_part[VERT] ||
#if CONFIG_EXT_RECUR_PARTITIONS
      !is_partition_valid(bsize, PARTITION_VERT) ||
#endif  // CONFIG_EXT_RECUR_PARTITIONS
      (!search_state->do_rectangular_split &&
       !av1_active_v_edge(cpi, mi_col, blk_params->mi_step_w)) ||
      blk_params->is_gt_max_sq_part) {
    return;
  }

  const int part_v_rate = search_state->partition_cost[PARTITION_VERT];
  if (part_v_rate == INT_MAX ||
      RDCOST(x->rdmult, part_v_rate, 0) >= best_rdc->rdcost) {
    return;
  }
  av1_init_rd_stats(&sum_rdc);
#if CONFIG_EXT_RECUR_PARTITIONS
  pc_tree->vertical[0] =
      av1_alloc_pc_tree_node(mi_row, mi_col, subsize, pc_tree, PARTITION_VERT,
                             0, 0, blk_params->ss_x, blk_params->ss_y);
  pc_tree->vertical[1] = av1_alloc_pc_tree_node(
      mi_row, mi_col + blk_params->mi_step_w, subsize, pc_tree, PARTITION_VERT,
      1, 1, blk_params->ss_x, blk_params->ss_y);

  if (ENABLE_FAST_RECUR_PARTITION && !frame_is_intra_only(cm) &&
      !x->must_find_valid_partition) {
    const SimpleMotionData *left =
        av1_get_sms_data(cpi, tile_info, x, mi_row, mi_col, subsize);
    const SimpleMotionData *right = av1_get_sms_data(
        cpi, tile_info, x, mi_row, mi_col + blk_params->mi_step_w, subsize);

    SMSPartitionStats part_data;
    part_data.sms_data[0] = left;
    part_data.sms_data[1] = right;
    part_data.num_sub_parts = 2;
    part_data.part_rate = part_v_rate;

    if (search_state->none_rd > 0 && search_state->none_rd < INT64_MAX &&
        (mi_row + 2 * blk_params->mi_step_h <= cm->mi_rows) &&
        (mi_col + 2 * blk_params->mi_step_w <= cm->mi_cols) &&
        av1_prune_new_part(&search_state->none_data, &part_data, x->rdmult)) {
      return;
    }
  }
#else
  for (int i = 0; i < 2; ++i) {
    pc_tree->vertical[i] =
        av1_alloc_pmc(cm, mi_row, mi_col + blk_params->mi_step_w * i, subsize,
                      pc_tree, PARTITION_VERT, i, blk_params->ss_x,
                      blk_params->ss_y, &td->shared_coeff_buf);
  }
#endif  // CONFIG_EXT_RECUR_PARTITIONS

  sum_rdc.rate = search_state->partition_cost[PARTITION_VERT];
  sum_rdc.rdcost = RDCOST(x->rdmult, sum_rdc.rate, 0);
  RD_STATS best_remain_rdcost;
  av1_rd_stats_subtraction(x->rdmult, best_rdc, &sum_rdc, &best_remain_rdcost);
#if CONFIG_EXT_RECUR_PARTITIONS
  av1_rd_pick_partition(cpi, td, tile_data, tp, mi_row, mi_col, subsize,
                        max_sq_part, min_sq_part, &this_rdc, best_remain_rdcost,
                        pc_tree->vertical[0], NULL, NULL, multi_pass_mode);
#else
  pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, &this_rdc, PARTITION_VERT,
                subsize, pc_tree->vertical[0], best_remain_rdcost,
                PICK_MODE_RD);
#endif  // CONFIG_EXT_RECUR_PARTITIONS
  av1_rd_cost_update(x->rdmult, &this_rdc);

  if (this_rdc.rate == INT_MAX) {
    sum_rdc.rdcost = INT64_MAX;
  } else {
    sum_rdc.rate += this_rdc.rate;
    sum_rdc.dist += this_rdc.dist;
    av1_rd_cost_update(x->rdmult, &sum_rdc);
  }
  search_state->rect_part_rd[VERT][0] = this_rdc.rdcost;
  if (sum_rdc.rdcost < best_rdc->rdcost && blk_params->has_cols) {
#if !CONFIG_EXT_RECUR_PARTITIONS
    const MB_MODE_INFO *const mbmi = &pc_tree->vertical[0]->mic;
    const PALETTE_MODE_INFO *const pmi = &mbmi->palette_mode_info;
    // Neither palette mode nor cfl predicted
    if (pmi->palette_size[0] == 0 && pmi->palette_size[1] == 0) {
      if (mbmi->uv_mode != UV_CFL_PRED)
        search_state->rect_ctx_is_ready[VERT] = 1;
    }
    av1_update_state(cpi, td, pc_tree->vertical[0], mi_row, mi_col, subsize, 1);
    av1_encode_superblock(cpi, tile_data, td, tp, DRY_RUN_NORMAL, subsize,
                          NULL);
#endif  // !CONFIG_EXT_RECUR_PARTITIONS

    av1_rd_stats_subtraction(x->rdmult, best_rdc, &sum_rdc,
                             &best_remain_rdcost);
#if CONFIG_EXT_RECUR_PARTITIONS
    av1_rd_pick_partition(cpi, td, tile_data, tp, mi_row,
                          mi_col + blk_params->mi_step_w, subsize, max_sq_part,
                          min_sq_part, &this_rdc, best_remain_rdcost,
                          pc_tree->vertical[1], NULL, NULL, multi_pass_mode);
#else
    pick_sb_modes(cpi, tile_data, x, mi_row, mi_col + blk_params->mi_step_w,
                  &this_rdc, PARTITION_VERT, subsize, pc_tree->vertical[1],
                  best_remain_rdcost, PICK_MODE_RD);
#endif  // CONFIG_EXT_RECUR_PARTITIONS
    av1_rd_cost_update(x->rdmult, &this_rdc);
    search_state->rect_part_rd[VERT][1] = this_rdc.rdcost;

    if (this_rdc.rate == INT_MAX) {
      sum_rdc.rdcost = INT64_MAX;
    } else {
      sum_rdc.rate += this_rdc.rate;
      sum_rdc.dist += this_rdc.dist;
      av1_rd_cost_update(x->rdmult, &sum_rdc);
    }
  }

  av1_rd_cost_update(x->rdmult, &sum_rdc);
  if (sum_rdc.rdcost < best_rdc->rdcost) {
    *best_rdc = sum_rdc;
    search_state->found_best_partition = true;
    pc_tree->partitioning = PARTITION_VERT;
  }

  av1_restore_context(cm, x, x_ctx, mi_row, mi_col, bsize, num_planes);
}

// TODO(any): Streamline all the ab_partition searches
#if !CONFIG_EXT_RECUR_PARTITIONS
static INLINE void prune_ab_partitions(AV1_COMP *cpi, PC_TREE *pc_tree,
                                       PartitionSearchState *search_state,
                                       const MACROBLOCK *x,
                                       const RD_STATS *best_rdc,
                                       unsigned int pb_source_variance,
                                       int ext_partition_allowed) {
  const BLOCK_SIZE bsize = search_state->part_blk_params.bsize;
  int horzab_partition_allowed =
      ext_partition_allowed & cpi->oxcf.enable_ab_partitions;
  int vertab_partition_allowed =
      ext_partition_allowed & cpi->oxcf.enable_ab_partitions;

#if CONFIG_DIST_8X8
  if (x->using_dist_8x8) {
    if (block_size_high[bsize] <= 8 || block_size_wide[bsize] <= 8) {
      horzab_partition_allowed = 0;
      vertab_partition_allowed = 0;
    }
  }
#endif

  if (cpi->sf.prune_ext_partition_types_search_level) {
    if (cpi->sf.prune_ext_partition_types_search_level == 1) {
      // TODO(debargha,huisu@google.com): may need to tune the threshold for
      // pb_source_variance.
      horzab_partition_allowed &= (pc_tree->partitioning == PARTITION_HORZ ||
                                   (pc_tree->partitioning == PARTITION_NONE &&
                                    pb_source_variance < 32) ||
                                   pc_tree->partitioning == PARTITION_SPLIT);
      vertab_partition_allowed &= (pc_tree->partitioning == PARTITION_VERT ||
                                   (pc_tree->partitioning == PARTITION_NONE &&
                                    pb_source_variance < 32) ||
                                   pc_tree->partitioning == PARTITION_SPLIT);
    } else {
      horzab_partition_allowed &= (pc_tree->partitioning == PARTITION_HORZ ||
                                   pc_tree->partitioning == PARTITION_SPLIT);
      vertab_partition_allowed &= (pc_tree->partitioning == PARTITION_VERT ||
                                   pc_tree->partitioning == PARTITION_SPLIT);
    }

    clip_partition_search_state_rd(search_state);
  }
  search_state->partition_ab_allowed[HORZ][PART_A] = horzab_partition_allowed;
  search_state->partition_ab_allowed[HORZ][PART_B] = horzab_partition_allowed;
  if (cpi->sf.prune_ext_partition_types_search_level) {
    const int64_t horz_a_rd = search_state->rect_part_rd[HORZ][1] +
                              search_state->split_rd[0] +
                              search_state->split_rd[1];
    const int64_t horz_b_rd = search_state->rect_part_rd[HORZ][0] +
                              search_state->split_rd[2] +
                              search_state->split_rd[3];
    switch (cpi->sf.prune_ext_partition_types_search_level) {
      case 1:
        search_state->partition_ab_allowed[HORZ][PART_A] &=
            (horz_a_rd / 16 * 14 < best_rdc->rdcost);
        search_state->partition_ab_allowed[HORZ][PART_B] &=
            (horz_b_rd / 16 * 14 < best_rdc->rdcost);
        break;
      case 2:
      default:
        search_state->partition_ab_allowed[HORZ][PART_A] &=
            (horz_a_rd / 16 * 15 < best_rdc->rdcost);
        search_state->partition_ab_allowed[HORZ][PART_B] &=
            (horz_b_rd / 16 * 15 < best_rdc->rdcost);
        break;
    }
  }

  search_state->partition_ab_allowed[VERT][PART_A] = vertab_partition_allowed;
  search_state->partition_ab_allowed[VERT][PART_B] = vertab_partition_allowed;
  if (cpi->sf.prune_ext_partition_types_search_level) {
    const int64_t vert_a_rd = search_state->rect_part_rd[VERT][1] +
                              search_state->split_rd[0] +
                              search_state->split_rd[2];
    const int64_t vert_b_rd = search_state->rect_part_rd[VERT][0] +
                              search_state->split_rd[1] +
                              search_state->split_rd[3];
    switch (cpi->sf.prune_ext_partition_types_search_level) {
      case 1:
        search_state->partition_ab_allowed[VERT][PART_A] &=
            (vert_a_rd / 16 * 14 < best_rdc->rdcost);
        search_state->partition_ab_allowed[VERT][PART_B] &=
            (vert_b_rd / 16 * 14 < best_rdc->rdcost);
        break;
      case 2:
      default:
        search_state->partition_ab_allowed[VERT][PART_A] &=
            (vert_a_rd / 16 * 15 < best_rdc->rdcost);
        search_state->partition_ab_allowed[VERT][PART_B] &=
            (vert_b_rd / 16 * 15 < best_rdc->rdcost);
        break;
    }
  }

  if (cpi->sf.ml_prune_ab_partition && ext_partition_allowed &&
#if CONFIG_EXT_RECUR_PARTITIONS
      is_square_block(bsize) &&
#endif  // CONFIG_EXT_RECUR_PARTITIONS
      search_state->partition_rect_allowed[HORZ] &&
      search_state->partition_rect_allowed[VERT]) {
    // TODO(huisu@google.com): x->source_variance may not be the current
    // block's variance. The correct one to use is pb_source_variance. Need to
    // re-train the model to fix it.
    av1_ml_prune_ab_partition(
        bsize, pc_tree->partitioning, get_unsigned_bits(x->source_variance),
        best_rdc->rdcost, search_state->rect_part_rd[HORZ],
        search_state->rect_part_rd[VERT], search_state->split_rd,
        &search_state->partition_ab_allowed[HORZ][PART_A],
        &search_state->partition_ab_allowed[HORZ][PART_B],
        &search_state->partition_ab_allowed[VERT][PART_A],
        &search_state->partition_ab_allowed[VERT][PART_B]);
  }

  search_state->partition_ab_allowed[HORZ][PART_A] &=
      cpi->oxcf.enable_ab_partitions;
  search_state->partition_ab_allowed[HORZ][PART_B] &=
      cpi->oxcf.enable_ab_partitions;
  search_state->partition_ab_allowed[VERT][PART_A] &=
      cpi->oxcf.enable_ab_partitions;
  search_state->partition_ab_allowed[VERT][PART_B] &=
      cpi->oxcf.enable_ab_partitions;
}

static INLINE void search_partition_horza(PartitionSearchState *search_state,
                                          AV1_COMP *const cpi, ThreadData *td,
                                          TileDataEnc *tile_data,
                                          TOKENEXTRA **tp, RD_STATS *best_rdc,
                                          PC_TREE *pc_tree,
                                          RD_SEARCH_MACROBLOCK_CONTEXT *x_ctx) {
  const AV1_COMMON *const cm = &cpi->common;
  MACROBLOCK *const x = &td->mb;
  const int num_planes = av1_num_planes(cm);

  const PartitionBlkParams *blk_params = &search_state->part_blk_params;
  const int mi_row = blk_params->mi_row, mi_col = blk_params->mi_col;
  const BLOCK_SIZE bsize = blk_params->bsize;

  const BLOCK_SIZE rec_subsize = get_partition_subsize(bsize, PARTITION_HORZ_A);
  const BLOCK_SIZE sqr_subsize = get_partition_subsize(bsize, PARTITION_SPLIT);

  if (search_state->terminate_partition_search ||
      !search_state->partition_rect_allowed[HORZ] ||
      !search_state->partition_ab_allowed[HORZ][PART_A] ||
      blk_params->is_gt_max_sq_part) {
    return;
  }

  pc_tree->horizontala[0] = av1_alloc_pmc(
      cm, mi_row, mi_col, sqr_subsize, pc_tree, PARTITION_HORZ_A, 0,
      blk_params->ss_x, blk_params->ss_y, &td->shared_coeff_buf);
  pc_tree->horizontala[1] =
      av1_alloc_pmc(cm, mi_row, mi_col + blk_params->mi_step_w, sqr_subsize,
                    pc_tree, PARTITION_HORZ_A, 1, blk_params->ss_x,
                    blk_params->ss_y, &td->shared_coeff_buf);
  pc_tree->horizontala[2] =
      av1_alloc_pmc(cm, mi_row + blk_params->mi_step_h, mi_col, rec_subsize,
                    pc_tree, PARTITION_HORZ_A, 2, blk_params->ss_x,
                    blk_params->ss_y, &td->shared_coeff_buf);

  pc_tree->horizontala[0]->rd_mode_is_ready = 0;
  pc_tree->horizontala[1]->rd_mode_is_ready = 0;
  pc_tree->horizontala[2]->rd_mode_is_ready = 0;
  if (search_state->split_ctx_is_ready[0]) {
    av1_copy_tree_context(pc_tree->horizontala[0], pc_tree->split[0]->none);
    pc_tree->horizontala[0]->mic.partition = PARTITION_HORZ_A;
    pc_tree->horizontala[0]->rd_mode_is_ready = 1;
    if (search_state->split_ctx_is_ready[1]) {
      av1_copy_tree_context(pc_tree->horizontala[1], pc_tree->split[1]->none);
      pc_tree->horizontala[1]->mic.partition = PARTITION_HORZ_A;
      pc_tree->horizontala[1]->rd_mode_is_ready = 1;
    }
  }

  search_state->found_best_partition |= rd_test_partition3(
      cpi, td, tile_data, tp, pc_tree, best_rdc, pc_tree->horizontala, mi_row,
      mi_col, bsize, PARTITION_HORZ_A, mi_row, mi_col, sqr_subsize, mi_row,
      mi_col + blk_params->mi_step_w, sqr_subsize,
      mi_row + blk_params->mi_step_h, mi_col, rec_subsize);

  av1_restore_context(cm, x, x_ctx, mi_row, mi_col, bsize, num_planes);
}

static INLINE void search_partition_horzb(PartitionSearchState *search_state,
                                          AV1_COMP *const cpi, ThreadData *td,
                                          TileDataEnc *tile_data,
                                          TOKENEXTRA **tp, RD_STATS *best_rdc,
                                          PC_TREE *pc_tree,
                                          RD_SEARCH_MACROBLOCK_CONTEXT *x_ctx) {
  const AV1_COMMON *const cm = &cpi->common;
  MACROBLOCK *const x = &td->mb;
  const int num_planes = av1_num_planes(cm);

  const PartitionBlkParams *blk_params = &search_state->part_blk_params;
  const int mi_row = blk_params->mi_row, mi_col = blk_params->mi_col;
  const BLOCK_SIZE bsize = blk_params->bsize;

  const BLOCK_SIZE rec_subsize = get_partition_subsize(bsize, PARTITION_HORZ_B);
  const BLOCK_SIZE sqr_subsize = get_partition_subsize(bsize, PARTITION_SPLIT);

  if (search_state->terminate_partition_search ||
      !search_state->partition_rect_allowed[HORZ] ||
      !search_state->partition_ab_allowed[HORZ][PART_B] ||
      blk_params->is_gt_max_sq_part) {
    return;
  }

  pc_tree->horizontalb[0] = av1_alloc_pmc(
      cm, mi_row, mi_col, rec_subsize, pc_tree, PARTITION_HORZ_B, 0,
      blk_params->ss_x, blk_params->ss_y, &td->shared_coeff_buf);
  pc_tree->horizontalb[1] =
      av1_alloc_pmc(cm, mi_row + blk_params->mi_step_h, mi_col, sqr_subsize,
                    pc_tree, PARTITION_HORZ_B, 1, blk_params->ss_x,
                    blk_params->ss_y, &td->shared_coeff_buf);
  pc_tree->horizontalb[2] = av1_alloc_pmc(
      cm, mi_row + blk_params->mi_step_h, mi_col + blk_params->mi_step_w,
      sqr_subsize, pc_tree, PARTITION_HORZ_B, 2, blk_params->ss_x,
      blk_params->ss_y, &td->shared_coeff_buf);

  pc_tree->horizontalb[0]->rd_mode_is_ready = 0;
  pc_tree->horizontalb[1]->rd_mode_is_ready = 0;
  pc_tree->horizontalb[2]->rd_mode_is_ready = 0;
  if (search_state->rect_ctx_is_ready[HORZ]) {
    av1_copy_tree_context(pc_tree->horizontalb[0], pc_tree->horizontal[0]);
    pc_tree->horizontalb[0]->mic.partition = PARTITION_HORZ_B;
    pc_tree->horizontalb[0]->rd_mode_is_ready = 1;
  }

  search_state->found_best_partition |= rd_test_partition3(
      cpi, td, tile_data, tp, pc_tree, best_rdc, pc_tree->horizontalb, mi_row,
      mi_col, bsize, PARTITION_HORZ_B, mi_row, mi_col, rec_subsize,
      mi_row + blk_params->mi_step_h, mi_col, sqr_subsize,
      mi_row + blk_params->mi_step_h, mi_col + blk_params->mi_step_w,
      sqr_subsize);

  av1_restore_context(cm, x, x_ctx, mi_row, mi_col, bsize, num_planes);
}

static INLINE void search_partition_verta(PartitionSearchState *search_state,
                                          AV1_COMP *const cpi, ThreadData *td,
                                          TileDataEnc *tile_data,
                                          TOKENEXTRA **tp, RD_STATS *best_rdc,
                                          PC_TREE *pc_tree,
                                          RD_SEARCH_MACROBLOCK_CONTEXT *x_ctx) {
  const AV1_COMMON *const cm = &cpi->common;
  MACROBLOCK *const x = &td->mb;
  const int num_planes = av1_num_planes(cm);

  const PartitionBlkParams *blk_params = &search_state->part_blk_params;
  const int mi_row = blk_params->mi_row, mi_col = blk_params->mi_col;
  const BLOCK_SIZE bsize = blk_params->bsize;

  const BLOCK_SIZE rec_subsize = get_partition_subsize(bsize, PARTITION_VERT_A);
  const BLOCK_SIZE sqr_subsize = get_partition_subsize(bsize, PARTITION_SPLIT);

  if (search_state->terminate_partition_search ||
      !search_state->partition_rect_allowed[VERT] ||
      !search_state->partition_ab_allowed[VERT][PART_A] ||
      blk_params->is_gt_max_sq_part) {
    return;
  }

  pc_tree->verticala[0] = av1_alloc_pmc(
      cm, mi_row, mi_col, sqr_subsize, pc_tree, PARTITION_VERT_A, 0,
      blk_params->ss_x, blk_params->ss_y, &td->shared_coeff_buf);
  pc_tree->verticala[1] =
      av1_alloc_pmc(cm, mi_row + blk_params->mi_step_h, mi_col, sqr_subsize,
                    pc_tree, PARTITION_VERT_A, 1, blk_params->ss_x,
                    blk_params->ss_y, &td->shared_coeff_buf);
  pc_tree->verticala[2] =
      av1_alloc_pmc(cm, mi_row, mi_col + blk_params->mi_step_w, rec_subsize,
                    pc_tree, PARTITION_VERT_A, 2, blk_params->ss_x,
                    blk_params->ss_y, &td->shared_coeff_buf);

  pc_tree->verticala[0]->rd_mode_is_ready = 0;
  pc_tree->verticala[1]->rd_mode_is_ready = 0;
  pc_tree->verticala[2]->rd_mode_is_ready = 0;
  if (search_state->split_ctx_is_ready[0]) {
    av1_copy_tree_context(pc_tree->verticala[0], pc_tree->split[0]->none);
    pc_tree->verticala[0]->mic.partition = PARTITION_VERT_A;
    pc_tree->verticala[0]->rd_mode_is_ready = 1;
  }

  search_state->found_best_partition |= rd_test_partition3(
      cpi, td, tile_data, tp, pc_tree, best_rdc, pc_tree->verticala, mi_row,
      mi_col, bsize, PARTITION_VERT_A, mi_row, mi_col, sqr_subsize,
      mi_row + blk_params->mi_step_h, mi_col, sqr_subsize, mi_row,
      mi_col + blk_params->mi_step_w, rec_subsize);

  av1_restore_context(cm, x, x_ctx, mi_row, mi_col, bsize, num_planes);
}

static INLINE void search_partition_vertb(PartitionSearchState *search_state,
                                          AV1_COMP *const cpi, ThreadData *td,
                                          TileDataEnc *tile_data,
                                          TOKENEXTRA **tp, RD_STATS *best_rdc,
                                          PC_TREE *pc_tree,
                                          RD_SEARCH_MACROBLOCK_CONTEXT *x_ctx) {
  const AV1_COMMON *const cm = &cpi->common;
  MACROBLOCK *const x = &td->mb;
  const int num_planes = av1_num_planes(cm);

  const PartitionBlkParams *blk_params = &search_state->part_blk_params;
  const int mi_row = blk_params->mi_row, mi_col = blk_params->mi_col;
  const BLOCK_SIZE bsize = blk_params->bsize;

  const BLOCK_SIZE rec_subsize = get_partition_subsize(bsize, PARTITION_VERT_B);
  const BLOCK_SIZE sqr_subsize = get_partition_subsize(bsize, PARTITION_SPLIT);

  if (search_state->terminate_partition_search ||
      !search_state->partition_rect_allowed[VERT] ||
      !search_state->partition_ab_allowed[VERT][PART_B] ||
      blk_params->is_gt_max_sq_part) {
    return;
  }

  pc_tree->verticalb[0] = av1_alloc_pmc(
      cm, mi_row, mi_col, rec_subsize, pc_tree, PARTITION_VERT_B, 0,
      blk_params->ss_x, blk_params->ss_y, &td->shared_coeff_buf);
  pc_tree->verticalb[1] =
      av1_alloc_pmc(cm, mi_row, mi_col + blk_params->mi_step_w, sqr_subsize,
                    pc_tree, PARTITION_VERT_B, 1, blk_params->ss_x,
                    blk_params->ss_y, &td->shared_coeff_buf);
  pc_tree->verticalb[2] = av1_alloc_pmc(
      cm, mi_row + blk_params->mi_step_h, mi_col + blk_params->mi_step_w,
      sqr_subsize, pc_tree, PARTITION_VERT_B, 2, blk_params->ss_x,
      blk_params->ss_y, &td->shared_coeff_buf);

  pc_tree->verticalb[0]->rd_mode_is_ready = 0;
  pc_tree->verticalb[1]->rd_mode_is_ready = 0;
  pc_tree->verticalb[2]->rd_mode_is_ready = 0;
  if (search_state->rect_ctx_is_ready[VERT]) {
    av1_copy_tree_context(pc_tree->verticalb[0], pc_tree->vertical[0]);
    pc_tree->verticalb[0]->mic.partition = PARTITION_VERT_B;
    pc_tree->verticalb[0]->rd_mode_is_ready = 1;
  }

  search_state->found_best_partition |= rd_test_partition3(
      cpi, td, tile_data, tp, pc_tree, best_rdc, pc_tree->verticalb, mi_row,
      mi_col, bsize, PARTITION_VERT_B, mi_row, mi_col, rec_subsize, mi_row,
      mi_col + blk_params->mi_step_w, sqr_subsize,
      mi_row + blk_params->mi_step_h, mi_col + blk_params->mi_step_w,
      sqr_subsize);

  av1_restore_context(cm, x, x_ctx, mi_row, mi_col, bsize, num_planes);
}

static INLINE void prune_partition_4(AV1_COMP *cpi, PC_TREE *pc_tree,
                                     PartitionSearchState *search_state,
                                     MACROBLOCK *x, const RD_STATS *best_rdc,
                                     unsigned int pb_source_variance,
                                     int ext_partition_allowed) {
  const PartitionBlkParams *blk_params = &search_state->part_blk_params;

  const BLOCK_SIZE bsize = blk_params->bsize;
  const int mi_row = blk_params->mi_row, mi_col = blk_params->mi_col;
  const int ss_x = blk_params->ss_x, ss_y = blk_params->ss_y;

  // partition4_allowed is 1 if we can use a PARTITION_HORZ_4 or
  // PARTITION_VERT_4 for this block. This is almost the same as
  // ext_partition_allowed, except that we don't allow 128x32 or 32x128
  // blocks, so we require that bsize is not BLOCK_128X128.
  const int partition4_allowed = cpi->oxcf.enable_1to4_partitions &&
                                 ext_partition_allowed &&
                                 bsize != BLOCK_128X128;

  const int is_chroma_size_valid_horz4 = check_is_chroma_size_valid(
      PARTITION_HORZ_4, bsize, mi_row, mi_col, ss_x, ss_y, pc_tree);

  const int is_chroma_size_valid_vert4 = check_is_chroma_size_valid(
      PARTITION_VERT_4, bsize, mi_row, mi_col, ss_x, ss_y, pc_tree);

  search_state->partition_4_allowed[HORZ] =
      partition4_allowed && search_state->partition_rect_allowed[HORZ] &&
      is_chroma_size_valid_horz4;
  search_state->partition_4_allowed[VERT] =
      partition4_allowed && search_state->partition_rect_allowed[VERT] &&
      is_chroma_size_valid_vert4;
  if (cpi->sf.prune_ext_partition_types_search_level == 2) {
    search_state->partition_4_allowed[HORZ] &=
        (pc_tree->partitioning == PARTITION_HORZ ||
         pc_tree->partitioning == PARTITION_HORZ_A ||
         pc_tree->partitioning == PARTITION_HORZ_B ||
         pc_tree->partitioning == PARTITION_SPLIT ||
         pc_tree->partitioning == PARTITION_NONE);
    search_state->partition_4_allowed[VERT] &=
        (pc_tree->partitioning == PARTITION_VERT ||
         pc_tree->partitioning == PARTITION_VERT_A ||
         pc_tree->partitioning == PARTITION_VERT_B ||
         pc_tree->partitioning == PARTITION_SPLIT ||
         pc_tree->partitioning == PARTITION_NONE);
  }
  if (cpi->sf.ml_prune_4_partition && partition4_allowed &&
      search_state->partition_rect_allowed[HORZ] &&
      search_state->partition_rect_allowed[VERT]) {
    av1_ml_prune_4_partition(
        cpi, x, bsize, pc_tree->partitioning, best_rdc->rdcost,
        search_state->rect_part_rd[HORZ], search_state->rect_part_rd[VERT],
        search_state->split_rd, &search_state->partition_4_allowed[HORZ],
        &search_state->partition_4_allowed[VERT], pb_source_variance, mi_row,
        mi_col);
  }

#if CONFIG_DIST_8X8
  if (x->using_dist_8x8) {
    if (block_size_high[bsize] <= 16 || block_size_wide[bsize] <= 16) {
      search_state->partition_4_allowed[HORZ] = 0;
      search_state->partition_4_allowed[VERT] = 0;
    }
  }
#endif

  if (blk_params->width < (blk_params->min_partition_size_1d << 2)) {
    search_state->partition_4_allowed[HORZ] = 0;
    search_state->partition_4_allowed[VERT] = 0;
  }
}

static INLINE void search_partition_horz4(PartitionSearchState *search_state,
                                          AV1_COMP *const cpi, ThreadData *td,
                                          TileDataEnc *tile_data,
                                          TOKENEXTRA **tp, RD_STATS *best_rdc,
                                          PC_TREE *pc_tree,
                                          RD_SEARCH_MACROBLOCK_CONTEXT *x_ctx) {
  const AV1_COMMON *const cm = &cpi->common;
  MACROBLOCK *const x = &td->mb;
  const int num_planes = av1_num_planes(cm);

  const PartitionBlkParams *blk_params = &search_state->part_blk_params;
  const int mi_row = blk_params->mi_row, mi_col = blk_params->mi_col;
  const BLOCK_SIZE bsize = blk_params->bsize;

  assert(IMPLIES(!cpi->oxcf.enable_rect_partitions,
                 !search_state->partition_4_allowed[HORZ]));

  if (search_state->terminate_partition_search ||
      !search_state->partition_4_allowed[HORZ] || !blk_params->has_rows ||
      (!search_state->do_rectangular_split &&
       !av1_active_h_edge(cpi, mi_row, blk_params->mi_step_h)) ||
      blk_params->is_gt_max_sq_part) {
    return;
  }

  RD_STATS sum_rdc;
  av1_init_rd_stats(&sum_rdc);
  BLOCK_SIZE subsize = get_partition_subsize(bsize, PARTITION_HORZ_4);
  const int quarter_step = mi_size_high[bsize] / 4;

  sum_rdc.rate = search_state->partition_cost[PARTITION_HORZ_4];
  sum_rdc.rdcost = RDCOST(x->rdmult, sum_rdc.rate, 0);

  for (int i = 0; i < 4; ++i) {
    pc_tree->horizontal4[i] =
        av1_alloc_pmc(cm, mi_row + i * quarter_step, mi_col, subsize, pc_tree,
                      PARTITION_HORZ_4, i, blk_params->ss_x, blk_params->ss_y,
                      &td->shared_coeff_buf);
  }

  for (int i = 0; i < 4; ++i) {
    const int this_mi_row = mi_row + i * quarter_step;

    if (i > 0 && this_mi_row >= cm->mi_rows) break;

    PICK_MODE_CONTEXT *ctx_this = pc_tree->horizontal4[i];

    ctx_this->rd_mode_is_ready = 0;
    if (!rd_try_subblock(cpi, td, tile_data, tp, (i == 3), this_mi_row, mi_col,
                         subsize, *best_rdc, &sum_rdc, PARTITION_HORZ_4,
                         ctx_this)) {
      av1_invalid_rd_stats(&sum_rdc);
      break;
    }
  }

  av1_rd_cost_update(x->rdmult, &sum_rdc);
  if (sum_rdc.rdcost < best_rdc->rdcost) {
    *best_rdc = sum_rdc;
    search_state->found_best_partition = true;
    pc_tree->partitioning = PARTITION_HORZ_4;
  }

  av1_restore_context(cm, x, x_ctx, mi_row, mi_col, bsize, num_planes);
}

static INLINE void search_partition_vert4(PartitionSearchState *search_state,
                                          AV1_COMP *const cpi, ThreadData *td,
                                          TileDataEnc *tile_data,
                                          TOKENEXTRA **tp, RD_STATS *best_rdc,
                                          PC_TREE *pc_tree,
                                          RD_SEARCH_MACROBLOCK_CONTEXT *x_ctx) {
  const AV1_COMMON *const cm = &cpi->common;
  MACROBLOCK *const x = &td->mb;
  const int num_planes = av1_num_planes(cm);

  const PartitionBlkParams *blk_params = &search_state->part_blk_params;
  const int mi_row = blk_params->mi_row, mi_col = blk_params->mi_col;
  const BLOCK_SIZE bsize = blk_params->bsize;

  assert(IMPLIES(!cpi->oxcf.enable_rect_partitions,
                 !search_state->partition_4_allowed[VERT]));
  if (search_state->terminate_partition_search ||
      !search_state->partition_4_allowed[VERT] || !blk_params->has_cols ||
      (!search_state->do_rectangular_split &&
       !av1_active_v_edge(cpi, mi_row, blk_params->mi_step_h)) ||
      blk_params->is_gt_max_sq_part) {
    return;
  }

  BLOCK_SIZE subsize = get_partition_subsize(bsize, PARTITION_VERT_4);
  RD_STATS sum_rdc;
  av1_init_rd_stats(&sum_rdc);

  const int quarter_step = mi_size_wide[bsize] / 4;

  sum_rdc.rate = search_state->partition_cost[PARTITION_VERT_4];
  sum_rdc.rdcost = RDCOST(x->rdmult, sum_rdc.rate, 0);

  for (int i = 0; i < 4; ++i) {
    pc_tree->vertical4[i] =
        av1_alloc_pmc(cm, mi_row, mi_col + i * quarter_step, subsize, pc_tree,
                      PARTITION_VERT_4, i, blk_params->ss_x, blk_params->ss_y,
                      &td->shared_coeff_buf);
  }

  for (int i = 0; i < 4; ++i) {
    const int this_mi_col = mi_col + i * quarter_step;

    if (i > 0 && this_mi_col >= cm->mi_cols) break;

    PICK_MODE_CONTEXT *ctx_this = pc_tree->vertical4[i];

    ctx_this->rd_mode_is_ready = 0;
    if (!rd_try_subblock(cpi, td, tile_data, tp, (i == 3), mi_row, this_mi_col,
                         subsize, *best_rdc, &sum_rdc, PARTITION_VERT_4,
                         ctx_this)) {
      av1_invalid_rd_stats(&sum_rdc);
      break;
    }
  }

  av1_rd_cost_update(x->rdmult, &sum_rdc);
  if (sum_rdc.rdcost < best_rdc->rdcost) {
    *best_rdc = sum_rdc;
    search_state->found_best_partition = true;
    pc_tree->partitioning = PARTITION_VERT_4;
  }
  av1_restore_context(cm, x, x_ctx, mi_row, mi_col, bsize, num_planes);
}
#else
static INLINE void prune_partition_3(AV1_COMP *cpi, PC_TREE *pc_tree,
                                     PartitionSearchState *search_state,
                                     MACROBLOCK *x, const RD_STATS *best_rdc,
                                     unsigned int pb_source_variance,
                                     int ext_partition_allowed) {
  const PartitionBlkParams *blk_params = &search_state->part_blk_params;
  const BLOCK_SIZE bsize = blk_params->bsize;
  const int mi_row = blk_params->mi_row, mi_col = blk_params->mi_col;

  // partition3_allowed is 1 if we can use a PARTITION_HORZ_3 or
  // PARTITION_VERT_3 for this block. This is almost the same as
  // ext_partition_allowed, except that we don't allow 128x32 or 32x128
  // blocks, so we require that bsize is not BLOCK_128X128.
  int partition3_allowed = cpi->oxcf.enable_1to3_partitions &&
                           ext_partition_allowed && bsize != BLOCK_128X128;

  const int is_chroma_size_valid_horz3 =
      check_is_chroma_size_valid(PARTITION_HORZ_3, bsize, mi_row, mi_col,
                                 blk_params->ss_x, blk_params->ss_y, pc_tree);

  const int is_chroma_size_valid_vert3 =
      check_is_chroma_size_valid(PARTITION_VERT_3, bsize, mi_row, mi_col,
                                 blk_params->ss_x, blk_params->ss_y, pc_tree);

  search_state->partition_3_allowed[HORZ] =
      partition3_allowed && search_state->partition_rect_allowed[HORZ] &&
      is_chroma_size_valid_horz3;
  search_state->partition_3_allowed[VERT] =
      partition3_allowed && search_state->partition_rect_allowed[VERT] &&
      is_chroma_size_valid_vert3;
  if (cpi->sf.prune_ext_partition_types_search_level == 2) {
    search_state->partition_3_allowed[HORZ] &=
        (pc_tree->partitioning == PARTITION_HORZ ||
         pc_tree->partitioning == PARTITION_SPLIT ||
         pc_tree->partitioning == PARTITION_NONE);
    search_state->partition_3_allowed[VERT] &=
        (pc_tree->partitioning == PARTITION_VERT ||
         pc_tree->partitioning == PARTITION_SPLIT ||
         pc_tree->partitioning == PARTITION_NONE);
  }

  partition3_allowed &= (search_state->partition_3_allowed[HORZ] ||
                         search_state->partition_3_allowed[VERT]);

  // TODO(urvang): Rename speed feature, and change behavior / make it work.
  // currently, it's a hack (still dividing into 4 subparts to get score).
  if (cpi->sf.ml_prune_4_partition && partition3_allowed &&
      is_square_block(bsize) && search_state->partition_rect_allowed[HORZ] &&
      search_state->partition_rect_allowed[VERT]) {
    clip_partition_search_state_rd(search_state);
    av1_ml_prune_4_partition(
        cpi, x, bsize, pc_tree->partitioning, best_rdc->rdcost,
        search_state->rect_part_rd[HORZ], search_state->rect_part_rd[VERT],
        search_state->split_rd, &search_state->partition_3_allowed[HORZ],
        &search_state->partition_3_allowed[VERT], pb_source_variance, mi_row,
        mi_col);
  }

#if CONFIG_DIST_8X8
  if (x->using_dist_8x8) {
    if (block_size_high[bsize] <= 16 || block_size_wide[bsize] <= 16) {
      search_state->partition_3_allowed[HORZ] = 0;
      search_state->partition_3_allowed[VERT] = 0;
    }
  }
#endif

  if (blk_params->width < (blk_params->min_partition_size_1d << 2)) {
    search_state->partition_3_allowed[HORZ] = 0;
    search_state->partition_3_allowed[VERT] = 0;
  }
}

static INLINE void search_partition_horz_3(PartitionSearchState *search_state,
                                           AV1_COMP *const cpi, ThreadData *td,
                                           TileDataEnc *tile_data,
                                           TOKENEXTRA **tp, RD_STATS *best_rdc,
                                           PC_TREE *pc_tree,
                                           RD_SEARCH_MACROBLOCK_CONTEXT *x_ctx,
                                           SB_MULTI_PASS_MODE multi_pass_mode) {
  const AV1_COMMON *const cm = &cpi->common;
  TileInfo *const tile_info = &tile_data->tile_info;
  MACROBLOCK *const x = &td->mb;
  const int num_planes = av1_num_planes(cm);

  const PartitionBlkParams *blk_params = &search_state->part_blk_params;
  const int mi_row = blk_params->mi_row, mi_col = blk_params->mi_col;
  const BLOCK_SIZE bsize = blk_params->bsize;

  const BLOCK_SIZE sml_subsize = get_partition_subsize(bsize, PARTITION_HORZ_3);
  const BLOCK_SIZE big_subsize = get_partition_subsize(bsize, PARTITION_HORZ);

  assert(IMPLIES(!cpi->oxcf.enable_rect_partitions,
                 !search_state->partition_3_allowed[HORZ]));

  if (search_state->terminate_partition_search ||
      !search_state->partition_3_allowed[HORZ] || !blk_params->has_rows ||
      !is_partition_valid(bsize, PARTITION_HORZ_3) ||
      !(search_state->do_rectangular_split ||
        av1_active_h_edge(cpi, mi_row, blk_params->mi_step_h)) ||
      blk_params->is_gt_max_sq_part) {
    return;
  }

  const int part_h3_rate = search_state->partition_cost[PARTITION_HORZ_3];
  if (part_h3_rate == INT_MAX ||
      RDCOST(x->rdmult, part_h3_rate, 0) >= best_rdc->rdcost) {
    return;
  }
  RD_STATS sum_rdc;
  av1_init_rd_stats(&sum_rdc);
  const int quarter_step = mi_size_high[bsize] / 4;

  sum_rdc.rate = search_state->partition_cost[PARTITION_HORZ_3];
  sum_rdc.rdcost = RDCOST(x->rdmult, sum_rdc.rate, 0);

  const int step_multipliers[3] = { 0, 1, 2 };
  const BLOCK_SIZE subblock_sizes[3] = { sml_subsize, big_subsize,
                                         sml_subsize };

  pc_tree->horizontal3[0] = av1_alloc_pc_tree_node(
      mi_row, mi_col, subblock_sizes[0], pc_tree, PARTITION_HORZ_3, 0, 0,
      blk_params->ss_x, blk_params->ss_y);
  pc_tree->horizontal3[1] = av1_alloc_pc_tree_node(
      mi_row + quarter_step, mi_col, subblock_sizes[1], pc_tree,
      PARTITION_HORZ_3, 1, 0, blk_params->ss_x, blk_params->ss_y);
  pc_tree->horizontal3[2] = av1_alloc_pc_tree_node(
      mi_row + quarter_step * 3, mi_col, subblock_sizes[2], pc_tree,
      PARTITION_HORZ_3, 2, 1, blk_params->ss_x, blk_params->ss_y);

  // TODO(chiyotsai@google.com): Pruning horz/vert3 gives a significant loss
  // on certain clips (e.g. galleon_cif.y4m). Need to investigate before we
  // we enable it.
  if (ENABLE_FAST_RECUR_PARTITION == 2 && !frame_is_intra_only(cm) &&
      !x->must_find_valid_partition) {
    const SimpleMotionData *up =
        av1_get_sms_data(cpi, tile_info, x, mi_row, mi_col, subblock_sizes[0]);
    const SimpleMotionData *middle = av1_get_sms_data(
        cpi, tile_info, x, mi_row + quarter_step, mi_col, subblock_sizes[1]);
    const SimpleMotionData *down =
        av1_get_sms_data(cpi, tile_info, x, mi_row + 3 * quarter_step, mi_col,
                         subblock_sizes[2]);

    SMSPartitionStats part_data;
    part_data.sms_data[0] = up;
    part_data.sms_data[1] = middle;
    part_data.sms_data[2] = down;
    part_data.num_sub_parts = 3;
    part_data.part_rate = part_h3_rate;

    if (search_state->none_rd > 0 && search_state->none_rd < INT64_MAX &&
        (mi_row + 2 * blk_params->mi_step_h <= cm->mi_rows) &&
        (mi_col + 2 * blk_params->mi_step_w <= cm->mi_cols) &&
        av1_prune_new_part(&search_state->none_data, &part_data, x->rdmult)) {
      return;
    }
  }

  int this_mi_row = mi_row;
  for (int i = 0; i < 3; ++i) {
    this_mi_row += quarter_step * step_multipliers[i];

    if (i > 0 && this_mi_row >= cm->mi_rows) break;

    SUBBLOCK_RDO_DATA rdo_data = { NULL,
                                   pc_tree->horizontal3[i],
                                   NULL,
                                   this_mi_row,
                                   mi_col,
                                   subblock_sizes[i],
                                   PARTITION_HORZ_3,
                                   i == 2,
                                   1,
                                   blk_params->max_sq_part,
                                   blk_params->min_sq_part };

    if (!rd_try_subblock_new(cpi, td, tile_data, tp, &rdo_data, *best_rdc,
                             &sum_rdc, multi_pass_mode)) {
      av1_invalid_rd_stats(&sum_rdc);
      break;
    }
  }

  av1_rd_cost_update(x->rdmult, &sum_rdc);
  if (sum_rdc.rdcost < best_rdc->rdcost) {
    *best_rdc = sum_rdc;
    search_state->found_best_partition = true;
    pc_tree->partitioning = PARTITION_HORZ_3;
  }

  av1_restore_context(cm, x, x_ctx, mi_row, mi_col, bsize, num_planes);
}

static INLINE void search_partition_vert_3(PartitionSearchState *search_state,
                                           AV1_COMP *const cpi, ThreadData *td,
                                           TileDataEnc *tile_data,
                                           TOKENEXTRA **tp, RD_STATS *best_rdc,
                                           PC_TREE *pc_tree,
                                           RD_SEARCH_MACROBLOCK_CONTEXT *x_ctx,
                                           SB_MULTI_PASS_MODE multi_pass_mode) {
  const AV1_COMMON *const cm = &cpi->common;
  TileInfo *const tile_info = &tile_data->tile_info;
  MACROBLOCK *const x = &td->mb;
  const int num_planes = av1_num_planes(cm);

  const PartitionBlkParams *blk_params = &search_state->part_blk_params;
  const int mi_row = blk_params->mi_row, mi_col = blk_params->mi_col;
  const BLOCK_SIZE bsize = blk_params->bsize;

  const BLOCK_SIZE sml_subsize = get_partition_subsize(bsize, PARTITION_VERT_3);
  const BLOCK_SIZE big_subsize = get_partition_subsize(bsize, PARTITION_VERT);

  assert(IMPLIES(!cpi->oxcf.enable_rect_partitions,
                 !search_state->partition_3_allowed[VERT]));

  if (search_state->terminate_partition_search ||
      !search_state->partition_3_allowed[VERT] || !blk_params->has_cols ||
      !is_partition_valid(bsize, PARTITION_VERT_3) ||
      !(search_state->do_rectangular_split ||
        av1_active_v_edge(cpi, mi_row, blk_params->mi_step_h)) ||
      blk_params->is_gt_max_sq_part) {
    return;
  }

  const int part_v3_rate = search_state->partition_cost[PARTITION_VERT_3];
  if (part_v3_rate == INT_MAX ||
      RDCOST(x->rdmult, part_v3_rate, 0) >= best_rdc->rdcost) {
    return;
  }

  RD_STATS sum_rdc;
  av1_init_rd_stats(&sum_rdc);
  const int quarter_step = mi_size_wide[bsize] / 4;

  sum_rdc.rate = search_state->partition_cost[PARTITION_VERT_3];
  sum_rdc.rdcost = RDCOST(x->rdmult, sum_rdc.rate, 0);

  const int step_multipliers[3] = { 0, 1, 2 };
  const BLOCK_SIZE subblock_sizes[3] = { sml_subsize, big_subsize,
                                         sml_subsize };

  pc_tree->vertical3[0] = av1_alloc_pc_tree_node(
      mi_row, mi_col, subblock_sizes[0], pc_tree, PARTITION_VERT_3, 0, 0,
      blk_params->ss_x, blk_params->ss_y);
  pc_tree->vertical3[1] = av1_alloc_pc_tree_node(
      mi_row, mi_col + quarter_step, subblock_sizes[1], pc_tree,
      PARTITION_VERT_3, 1, 0, blk_params->ss_x, blk_params->ss_y);
  pc_tree->vertical3[2] = av1_alloc_pc_tree_node(
      mi_row, mi_col + quarter_step * 3, subblock_sizes[2], pc_tree,
      PARTITION_VERT_3, 2, 1, blk_params->ss_x, blk_params->ss_y);

  // TODO(chiyotsai@google.com): Pruning horz/vert3 gives a significant loss
  // on certain clips (e.g. galleon_cif.y4m). Need to investigate before we
  // we enable it.
  if (ENABLE_FAST_RECUR_PARTITION == 2 && !frame_is_intra_only(cm) &&
      !x->must_find_valid_partition) {
    const SimpleMotionData *left =
        av1_get_sms_data(cpi, tile_info, x, mi_row, mi_col, subblock_sizes[0]);
    const SimpleMotionData *middle = av1_get_sms_data(
        cpi, tile_info, x, mi_row, mi_col + quarter_step, subblock_sizes[1]);
    const SimpleMotionData *right =
        av1_get_sms_data(cpi, tile_info, x, mi_row, mi_col + 3 * quarter_step,
                         subblock_sizes[2]);

    SMSPartitionStats part_data;
    part_data.sms_data[0] = left;
    part_data.sms_data[1] = middle;
    part_data.sms_data[2] = right;
    part_data.num_sub_parts = 3;
    part_data.part_rate = part_v3_rate;

    if (search_state->none_rd > 0 && search_state->none_rd < INT64_MAX &&
        (mi_row + 2 * blk_params->mi_step_h <= cm->mi_rows) &&
        (mi_col + 2 * blk_params->mi_step_w <= cm->mi_cols) &&
        av1_prune_new_part(&search_state->none_data, &part_data, x->rdmult)) {
      return;
    }
  }

  int this_mi_col = mi_col;
  for (int i = 0; i < 3; ++i) {
    this_mi_col += quarter_step * step_multipliers[i];

    if (i > 0 && this_mi_col >= cm->mi_cols) break;

    SUBBLOCK_RDO_DATA rdo_data = { NULL,
                                   pc_tree->vertical3[i],
                                   NULL,
                                   mi_row,
                                   this_mi_col,
                                   subblock_sizes[i],
                                   PARTITION_VERT_3,
                                   i == 2,
                                   1,
                                   blk_params->max_sq_part,
                                   blk_params->min_sq_part };

    if (!rd_try_subblock_new(cpi, td, tile_data, tp, &rdo_data, *best_rdc,
                             &sum_rdc, multi_pass_mode)) {
      av1_invalid_rd_stats(&sum_rdc);
      break;
    }
  }

  av1_rd_cost_update(x->rdmult, &sum_rdc);
  if (sum_rdc.rdcost < best_rdc->rdcost) {
    *best_rdc = sum_rdc;
    search_state->found_best_partition = true;
    pc_tree->partitioning = PARTITION_VERT_3;
  }
  av1_restore_context(cm, x, x_ctx, mi_row, mi_col, bsize, num_planes);
}

#endif  // CONFIG_EXT_RECUR_PARTITIONS

#define PRUNE_WITH_PREV_PARTITION(cur_partition) \
  (prev_partition != PARTITION_INVALID && prev_partition != (cur_partition))

bool av1_rd_pick_partition(AV1_COMP *const cpi, ThreadData *td,
                           TileDataEnc *tile_data, TOKENEXTRA **tp, int mi_row,
                           int mi_col, BLOCK_SIZE bsize, BLOCK_SIZE max_sq_part,
                           BLOCK_SIZE min_sq_part, RD_STATS *rd_cost,
                           RD_STATS best_rdc, PC_TREE *pc_tree,
                           SIMPLE_MOTION_DATA_TREE *sms_tree, int64_t *none_rd,
                           SB_MULTI_PASS_MODE multi_pass_mode) {
  const AV1_COMMON *const cm = &cpi->common;
  const int num_planes = av1_num_planes(cm);
  TileInfo *const tile_info = &tile_data->tile_info;
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  RD_SEARCH_MACROBLOCK_CONTEXT x_ctx;
  const TOKENEXTRA *const tp_orig = *tp;
  PartitionSearchState search_state;
  av1_init_partition_search_state(&search_state, x, cpi, pc_tree, mi_row,
                                  mi_col, bsize, max_sq_part, min_sq_part);
  const PartitionBlkParams *blk_params = &search_state.part_blk_params;
  const PARTITION_TYPE prev_partition =
#if CONFIG_EXT_RECUR_PARTITIONS
      av1_get_prev_partition(cpi, x, mi_row, mi_col, bsize);
#else
      PARTITION_INVALID;
#endif  // CONFIG_EXT_RECUR_PARTITIONS

#if CONFIG_EXT_RECUR_PARTITIONS
  if (sms_tree != NULL)
#endif  // CONFIG_EXT_RECUR_PARTITIONS
    sms_tree->partitioning = PARTITION_NONE;

  if (best_rdc.rdcost < 0) {
    av1_invalid_rd_stats(rd_cost);
    return search_state.found_best_partition;
  }

#if CONFIG_EXT_RECUR_PARTITIONS
  // Check whether there is a counterpart pc_tree node with the same size
  // and the same neighboring context at the same location but from a different
  // partition path. If yes directly copy the RDO decision made for the
  // counterpart.
  PC_TREE *counterpart_block = av1_look_for_counterpart_block(pc_tree);
  if (counterpart_block) {
    if (counterpart_block->rd_cost.rate != INT_MAX) {
      av1_copy_pc_tree_recursive(cm, pc_tree, counterpart_block,
                                 blk_params->ss_x, blk_params->ss_y,
                                 &td->shared_coeff_buf, num_planes);
      *rd_cost = pc_tree->rd_cost;
      assert(bsize != cm->seq_params.sb_size);
      if (bsize == cm->seq_params.sb_size) exit(0);

      if (!pc_tree->is_last_subblock) {
        av1_encode_sb(cpi, td, tile_data, tp, mi_row, mi_col, DRY_RUN_NORMAL,
                      bsize, pc_tree, NULL, NULL);
      }
      return true;
    } else {
      av1_invalid_rd_stats(rd_cost);
      return false;
    }
  }
#endif  // CONFIG_EXT_RECUR_PARTITIONS

  if (frame_is_intra_only(cm) && bsize == BLOCK_64X64) {
    x->quad_tree_idx = 0;
    x->cnn_output_valid = 0;
  }

  if (bsize == cm->seq_params.sb_size) x->must_find_valid_partition = 0;

  if (none_rd) *none_rd = 0;

  (void)*tp_orig;

#if CONFIG_COLLECT_PARTITION_STATS
  int partition_decisions[EXT_PARTITION_TYPES] = { 0 };
  int partition_attempts[EXT_PARTITION_TYPES] = { 0 };
  int64_t partition_times[EXT_PARTITION_TYPES] = { 0 };
  struct aom_usec_timer partition_timer = { 0 };
  int partition_timer_on = 0;
#if CONFIG_COLLECT_PARTITION_STATS == 2
  PartitionStats *part_stats = &cpi->partition_stats;
#endif
#endif

#ifndef NDEBUG
  // Nothing should rely on the default value of this array (which is just
  // leftover from encoding the previous block. Setting it to fixed pattern
  // when debugging.
  // bit 0, 1, 2 are blk_skip of each plane
  // bit 4, 5, 6 are initialization checking of each plane
  memset(x->blk_skip, 0x77, sizeof(x->blk_skip));
#endif  // NDEBUG

#if !CONFIG_EXT_RECUR_PARTITIONS
  assert(block_size_wide[bsize] == block_size_high[bsize]);
#endif  // !CONFIG_EXT_RECUR_PARTITIONS

  av1_enc_set_offsets(cpi, tile_info, x, mi_row, mi_col, bsize,
                      &pc_tree->chroma_ref_info);

  // Save rdmult before it might be changed, so it can be restored later.
  const int orig_rdmult = x->rdmult;
  av1_setup_block_rdmult(cpi, x, mi_row, mi_col, bsize, NO_AQ, NULL);

  av1_rd_cost_update(x->rdmult, &best_rdc);

  if (bsize == BLOCK_16X16 && cpi->vaq_refresh)
    x->mb_energy = av1_log_block_var(cpi, x, bsize);

  xd->above_txfm_context = cm->above_txfm_context[tile_info->tile_row] + mi_col;
  xd->left_txfm_context =
      xd->left_txfm_context_buffer + (mi_row & MAX_MIB_MASK);
  av1_save_context(x, &x_ctx, mi_row, mi_col, bsize, num_planes);

#if !CONFIG_EXT_RECUR_PARTITIONS
  const int try_intra_cnn_split =
      frame_is_intra_only(cm) && cpi->sf.intra_cnn_split &&
      cm->seq_params.sb_size >= BLOCK_64X64 && bsize <= BLOCK_64X64 &&
      bsize >= BLOCK_8X8 && mi_row + mi_size_high[bsize] <= cm->mi_rows &&
      mi_col + mi_size_wide[bsize] <= cm->mi_cols;

  if (try_intra_cnn_split) {
    av1_intra_mode_cnn_partition(&cpi->common, x, bsize, x->quad_tree_idx,
                                 &search_state.partition_none_allowed,
                                 &search_state.partition_rect_allowed[HORZ],
                                 &search_state.partition_rect_allowed[VERT],
                                 &search_state.do_rectangular_split,
                                 &search_state.do_square_split);
  }

  // Use simple_motion_search to prune partitions. This must be done prior to
  // PARTITION_SPLIT to propagate the initial mvs to a smaller blocksize.
  const int try_split_only =
      cpi->sf.simple_motion_search_split && search_state.do_square_split &&
      bsize >= BLOCK_8X8 && mi_row + mi_size_high[bsize] <= cm->mi_rows &&
      mi_col + mi_size_wide[bsize] <= cm->mi_cols && !frame_is_intra_only(cm) &&
      !av1_superres_scaled(cm);

#if CONFIG_EXT_RECUR_PARTITIONS
  if (try_split_only && sms_tree) {
#else
  if (try_split_only) {
#endif  // CONFIG_EXT_RECUR_PARTITIONS
    av1_simple_motion_search_based_split(
        cpi, x, sms_tree, mi_row, mi_col, bsize,
        &search_state.partition_none_allowed,
        &search_state.partition_rect_allowed[HORZ],
        &search_state.partition_rect_allowed[VERT],
        &search_state.do_rectangular_split, &search_state.do_square_split);
  }

  const int try_prune_rect =
      cpi->sf.simple_motion_search_prune_rect && !frame_is_intra_only(cm) &&
      search_state.do_rectangular_split &&
      (search_state.do_square_split || search_state.partition_none_allowed ||
       (search_state.prune_rect_part[HORZ] &&
        search_state.prune_rect_part[VERT])) &&
      (search_state.partition_rect_allowed[HORZ] ||
       search_state.partition_rect_allowed[VERT]) &&
      bsize >= BLOCK_8X8;

#if CONFIG_EXT_RECUR_PARTITIONS
  if (try_prune_rect && sms_tree) {
#else
  if (try_prune_rect) {
#endif  // CONFIG_EXT_RECUR_PARTITIONS
    av1_simple_motion_search_prune_part(
        cpi, x, sms_tree, mi_row, mi_col, bsize,
        &search_state.partition_none_allowed,
        &search_state.partition_rect_allowed[HORZ],
        &search_state.partition_rect_allowed[VERT],
        &search_state.do_square_split, &search_state.do_rectangular_split,
        &search_state.prune_rect_part[HORZ],
        &search_state.prune_rect_part[VERT]);
  }
#endif  // !CONFIG_EXT_RECUR_PARTITIONS

  // Max and min square partition levels are defined as the partition nodes that
  // the recursive function rd_pick_partition() can reach. To implement this:
  // only PARTITION_NONE is allowed if the current node equals min_sq_part,
  // only PARTITION_SPLIT is allowed if the current node exceeds max_sq_part.
  assert(block_size_wide[min_sq_part] == block_size_high[min_sq_part]);
  assert(block_size_wide[max_sq_part] == block_size_high[max_sq_part]);
  assert(min_sq_part <= max_sq_part);
  assert(blk_params->min_partition_size_1d <=
         blk_params->max_partition_size_1d);
  if (blk_params->is_gt_max_sq_part) {
    // If current block size is larger than max, only allow split.
    search_state.partition_none_allowed = 0;
#if CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT
    search_state.partition_rect_allowed[HORZ] = 1;
    search_state.partition_rect_allowed[VERT] = 1;
#else   // CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT
    search_state.partition_rect_allowed[HORZ] = 0;
    search_state.partition_rect_allowed[VERT] = 0;
    search_state.do_square_split = 1;
#endif  // CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT
  } else if (blk_params->is_le_min_sq_part) {
    // If current block size is less or equal to min, only allow none if valid
    // block large enough; only allow split otherwise.
    search_state.partition_rect_allowed[HORZ] = 0;
    search_state.partition_rect_allowed[VERT] = 0;
    // only disable square split when current block is not at the picture
    // boundary. Otherwise, inherit the square split flag from previous logic
#if CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT
    search_state.partition_none_allowed = 1;
#else   // CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT
    if (blk_params->has_rows && blk_params->has_cols)
      search_state.do_square_split = 0;
    search_state.partition_none_allowed = !search_state.do_square_split;
#endif  // CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT
  }

BEGIN_PARTITION_SEARCH:
  if (x->must_find_valid_partition) {
    init_partition_allowed(&search_state, cpi, pc_tree);
#if CONFIG_EXT_RECUR_PARTITIONS
    if (!is_square_block(bsize)) {
      if (!search_state.partition_rect_allowed[HORZ] &&
          !search_state.partition_rect_allowed[VERT] &&
          !search_state.partition_none_allowed) {
        if (block_size_wide[bsize] > block_size_high[bsize])
          search_state.partition_rect_allowed[VERT] = 1;
        else
          search_state.partition_rect_allowed[HORZ] = 1;
      }
    }
#endif  // CONFIG_EXT_RECUR_PARTITIONS
  }

  // Partition block source pixel variance.
  unsigned int pb_source_variance = UINT_MAX;

#if CONFIG_DIST_8X8
  if (x->using_dist_8x8) {
    if (block_size_high[bsize] <= 8) partition_rect_allowed[HORZ] = 0;
    if (block_size_wide[bsize] <= 8) partition_rect_allowed[VERT] = 0;
    if (block_size_high[bsize] <= 8 || block_size_wide[bsize] <= 8)
      do_square_split = 0;
  }
#endif

  // PARTITION_NONE
#if CONFIG_EXT_RECUR_PARTITIONS
  if (ENABLE_FAST_RECUR_PARTITION && !frame_is_intra_only(cm)) {
    const SimpleMotionData *whole =
        av1_get_sms_data(cpi, tile_info, x, mi_row, mi_col, bsize);
    search_state.none_data.sms_data[0] = whole;
    search_state.none_data.num_sub_parts = 1;
    search_state.none_data.part_rate =
        search_state.partition_cost[PARTITION_NONE];
  }
#endif  // CONFIG_EXT_RECUR_PARTITIONS
  int64_t part_none_rd = INT64_MAX;

  if (!PRUNE_WITH_PREV_PARTITION(PARTITION_NONE)) {
    search_partition_none(&search_state, cpi, td, tile_data, &best_rdc, pc_tree,
                          sms_tree, &x_ctx, &pb_source_variance, none_rd,
                          &part_none_rd);
  }

  // PARTITION_SPLIT
#if !(CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT)
  int64_t part_split_rd = INT64_MAX;
  search_partition_split(&search_state, cpi, td, tile_data, tp, &best_rdc,
                         &part_split_rd, pc_tree, sms_tree, &x_ctx,
                         multi_pass_mode);
#endif  // !(CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT)

#if !(CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT)
  if (cpi->sf.ml_early_term_after_part_split_level &&
      !frame_is_intra_only(cm) && !search_state.terminate_partition_search &&
#if CONFIG_EXT_RECUR_PARTITIONS
      sms_tree &&
#endif  // CONFIG_EXT_RECUR_PARTITIONS
      search_state.do_rectangular_split &&
      (search_state.partition_rect_allowed[HORZ] ||
       search_state.partition_rect_allowed[VERT])) {
    av1_ml_early_term_after_split(cpi, x, sms_tree, bsize, best_rdc.rdcost,
                                  part_none_rd, part_split_rd,
                                  search_state.split_rd, mi_row, mi_col,
                                  &search_state.terminate_partition_search);
  }
#endif  // !(CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT)

  if (!cpi->sf.ml_early_term_after_part_split_level &&
      cpi->sf.ml_prune_rect_partition && !frame_is_intra_only(cm) &&
      (search_state.partition_rect_allowed[HORZ] ||
       search_state.partition_rect_allowed[VERT]) &&
      !(search_state.prune_rect_part[HORZ] ||
        search_state.prune_rect_part[VERT]) &&
      !search_state.terminate_partition_search) {
    av1_setup_src_planes(x, cpi->source, mi_row, mi_col, num_planes,
                         &pc_tree->chroma_ref_info);
    av1_ml_prune_rect_partition(cpi, x, bsize, best_rdc.rdcost,
                                search_state.none_rd, search_state.split_rd,
                                &search_state.prune_rect_part[HORZ],
                                &search_state.prune_rect_part[VERT]);
  }

  // PARTITION_HORZ
  if (!PRUNE_WITH_PREV_PARTITION(PARTITION_HORZ)) {
    search_partition_horz(&search_state, cpi, td, tile_data, tp, &best_rdc,
                          pc_tree, &x_ctx, multi_pass_mode);
  }

  // PARTITION_VERT
  if (!PRUNE_WITH_PREV_PARTITION(PARTITION_VERT)) {
    search_partition_vert(&search_state, cpi, td, tile_data, tp, &best_rdc,
                          pc_tree, &x_ctx, multi_pass_mode);
  }

  if (pb_source_variance == UINT_MAX) {
    av1_setup_src_planes(x, cpi->source, mi_row, mi_col, num_planes,
                         &pc_tree->chroma_ref_info);
    if (is_cur_buf_hbd(xd)) {
      pb_source_variance = av1_high_get_sby_perpixel_variance(
          cpi, &x->plane[0].src, bsize, xd->bd);
    } else {
      pb_source_variance =
          av1_get_sby_perpixel_variance(cpi, &x->plane[0].src, bsize);
    }
  }

  assert(IMPLIES(!cpi->oxcf.enable_rect_partitions,
                 !search_state.do_rectangular_split));

  const int ext_partition_allowed = search_state.do_rectangular_split &&
                                    bsize > BLOCK_8X8 && blk_params->has_rows &&
                                    blk_params->has_cols;

#if !CONFIG_EXT_RECUR_PARTITIONS
  // The standard AB partitions are allowed whenever ext-partition-types are
  // allowed
  prune_ab_partitions(cpi, pc_tree, &search_state, x, &best_rdc,
                      pb_source_variance, ext_partition_allowed);

  // PARTITION_HORZ_A
  search_partition_horza(&search_state, cpi, td, tile_data, tp, &best_rdc,
                         pc_tree, &x_ctx);

  // PARTITION_HORZ_B
  search_partition_horzb(&search_state, cpi, td, tile_data, tp, &best_rdc,
                         pc_tree, &x_ctx);

  // PARTITION_VERT_A
  search_partition_verta(&search_state, cpi, td, tile_data, tp, &best_rdc,
                         pc_tree, &x_ctx);

  // PARTITION_VERT_B
  search_partition_vertb(&search_state, cpi, td, tile_data, tp, &best_rdc,
                         pc_tree, &x_ctx);
#endif  // !CONFIG_EXT_RECUR_PARTITIONS

#if CONFIG_EXT_RECUR_PARTITIONS
  prune_partition_3(cpi, pc_tree, &search_state, x, &best_rdc,
                    pb_source_variance, ext_partition_allowed);

  // PARTITION_HORZ_3
  if (!PRUNE_WITH_PREV_PARTITION(PARTITION_HORZ_3)) {
    search_partition_horz_3(&search_state, cpi, td, tile_data, tp, &best_rdc,
                            pc_tree, &x_ctx, multi_pass_mode);
  }

  // PARTITION_VERT_3
  if (!PRUNE_WITH_PREV_PARTITION(PARTITION_VERT_3)) {
    search_partition_vert_3(&search_state, cpi, td, tile_data, tp, &best_rdc,
                            pc_tree, &x_ctx, multi_pass_mode);
  }
#else
  prune_partition_4(cpi, pc_tree, &search_state, x, &best_rdc,
                    pb_source_variance, ext_partition_allowed);

  // PARTITION_HORZ_4
  search_partition_horz4(&search_state, cpi, td, tile_data, tp, &best_rdc,
                         pc_tree, &x_ctx);

  // PARTITION_VERT_4
  search_partition_vert4(&search_state, cpi, td, tile_data, tp, &best_rdc,
                         pc_tree, &x_ctx);
#endif  // CONFIG_EXT_RECUR_PARTITIONS

#if CONFIG_EXT_RECUR_PARTITIONS
  if (sms_tree)
#endif  // CONFIG_EXT_RECUR_PARTITIONS
    sms_tree->partitioning = pc_tree->partitioning;
  if (bsize == cm->seq_params.sb_size && !search_state.found_best_partition) {
    // Did not find a valid partition, go back and search again, with less
    // constraint on which partition types to search.
    x->must_find_valid_partition = 1;
#if CONFIG_COLLECT_PARTITION_STATS == 2
    part_stats->partition_redo += 1;
#endif
    goto BEGIN_PARTITION_SEARCH;
  }

  *rd_cost = best_rdc;
  pc_tree->rd_cost = best_rdc;
  if (!search_state.found_best_partition) {
    av1_invalid_rd_stats(&pc_tree->rd_cost);
  } else {
#if CONFIG_EXT_RECUR_PARTITIONS
    av1_cache_best_partition(x->sms_bufs, mi_row, mi_col, bsize,
                             cm->seq_params.sb_size, pc_tree->partitioning);
#endif  // CONFIG_EXT_RECUR_PARTITIONS
  }

#if CONFIG_COLLECT_PARTITION_STATS
  if (best_rdc.rate < INT_MAX && best_rdc.dist < INT64_MAX) {
    partition_decisions[pc_tree->partitioning] += 1;
  }
#endif

#if CONFIG_COLLECT_PARTITION_STATS == 1
  // If CONFIG_COLLECT_PARTITION_STATS is 1, then print out the stats for each
  // prediction block
  FILE *f = fopen("data.csv", "a");
  fprintf(f, "%d,%d,%d,", bsize, cm->show_frame, frame_is_intra_only(cm));
  for (int idx = 0; idx < EXT_PARTITION_TYPES; idx++) {
    fprintf(f, "%d,", partition_decisions[idx]);
  }
  for (int idx = 0; idx < EXT_PARTITION_TYPES; idx++) {
    fprintf(f, "%d,", partition_attempts[idx]);
  }
  for (int idx = 0; idx < EXT_PARTITION_TYPES; idx++) {
    fprintf(f, "%ld,", partition_times[idx]);
  }
  fprintf(f, "\n");
  fclose(f);
#endif

#if CONFIG_COLLECT_PARTITION_STATS == 2
  // If CONFIG_COLLECTION_PARTITION_STATS is 2, then we print out the stats for
  // the whole clip. So we need to pass the information upstream to the encoder
  const int bsize_idx = av1_get_bsize_idx_for_part_stats(bsize);
  int *agg_attempts = part_stats->partition_attempts[bsize_idx];
  int *agg_decisions = part_stats->partition_decisions[bsize_idx];
  int64_t *agg_times = part_stats->partition_times[bsize_idx];
  for (int idx = 0; idx < EXT_PARTITION_TYPES; idx++) {
    agg_attempts[idx] += partition_attempts[idx];
    agg_decisions[idx] += partition_decisions[idx];
    agg_times[idx] += partition_times[idx];
  }
#endif

#if CONFIG_EXT_RECUR_PARTITIONS
  if (sms_tree)
#endif  // CONFIG_EXT_RECUR_PARTITIONS
    sms_tree->partitioning = pc_tree->partitioning;
  int pc_tree_dealloc = 0;
  if (search_state.found_best_partition) {
#if CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT
    assert(pc_tree->partitioning != PARTITION_SPLIT);
#endif  // CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT
    if (bsize == cm->seq_params.sb_size) {
      const int emit_output = multi_pass_mode != SB_DRY_PASS;
      const RUN_TYPE run_type = emit_output ? OUTPUT_ENABLED : DRY_RUN_NORMAL;

      x->cb_offset = 0;

      av1_reset_ptree_in_sbi(xd->sbi);
      av1_encode_sb(cpi, td, tile_data, tp, mi_row, mi_col, run_type, bsize,
                    pc_tree, xd->sbi->ptree_root, NULL);
      av1_free_pc_tree_recursive(pc_tree, num_planes, 0, 0);
      pc_tree_dealloc = 1;
    } else if (!pc_tree->is_last_subblock) {
      av1_encode_sb(cpi, td, tile_data, tp, mi_row, mi_col, DRY_RUN_NORMAL,
                    bsize, pc_tree, NULL, NULL);
    }
  }

  int keep_tree = 0;
#if CONFIG_EXT_RECUR_PARTITIONS && USE_OLD_PREDICTION_MODE
  keep_tree = 1;
#endif  // CONFIG_EXT_RECUR_PARTITIONS && USE_OLD_PREDICTION_MODE

  if (!pc_tree_dealloc && !keep_tree) {
    av1_free_pc_tree_recursive(pc_tree, num_planes, 1, 1);
  }

  if (bsize == cm->seq_params.sb_size) {
    assert(best_rdc.rate < INT_MAX);
    assert(best_rdc.dist < INT64_MAX);
  } else {
    assert(tp_orig == *tp);
  }

  x->rdmult = orig_rdmult;
  return search_state.found_best_partition;
}
#endif  // !CONFIG_REALTIME_ONLY
