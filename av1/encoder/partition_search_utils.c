
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

#include "av1/common/cfl.h"
#if CONFIG_INTERINTRA_ML
#include "av1/common/interintra_ml.h"
#endif  // CONFIG_INTERINTRA_ML
#include "av1/common/pred_common.h"
#include "av1/common/reconinter.h"
#include "av1/common/reconintra.h"

#include "av1/encoder/aq_variance.h"
#include "av1/encoder/encodemv.h"
#if CONFIG_NN_RECON
#include "av1/common/nn_recon.h"
#endif
#include "av1/encoder/partition_search_utils.h"
#include "av1/encoder/reconinter_enc.h"

#if CONFIG_INTERINTRA_ML_DATA_COLLECT
#include "av1/encoder/interintra_ml_data_collect.h"
#endif  // CONFIG_INTERINTRA_ML_DATA_COLLECT

#if !CONFIG_REALTIME_ONLY
static const FIRSTPASS_STATS *read_one_frame_stats(const TWO_PASS *p, int frm) {
  assert(frm >= 0);
  if (frm < 0 ||
      p->stats_buf_ctx->stats_in_start + frm > p->stats_buf_ctx->stats_in_end) {
    return NULL;
  }

  return &p->stats_buf_ctx->stats_in_start[frm];
}

int av1_active_h_edge(const AV1_COMP *cpi, int mi_row, int mi_step) {
  int top_edge = 0;
  int bottom_edge = cpi->common.mi_rows;
  int is_active_h_edge = 0;

  // For two pass account for any formatting bars detected.
  if (is_stat_consumption_stage_twopass(cpi)) {
    const AV1_COMMON *const cm = &cpi->common;
    const FIRSTPASS_STATS *const this_frame_stats = read_one_frame_stats(
        &cpi->twopass, cm->current_frame.display_order_hint);
    if (this_frame_stats == NULL) return AOM_CODEC_ERROR;

    // The inactive region is specified in MBs not mi units.
    // The image edge is in the following MB row.
    top_edge += (int)(this_frame_stats->inactive_zone_rows * 4);

    bottom_edge -= (int)(this_frame_stats->inactive_zone_rows * 4);
    bottom_edge = AOMMAX(top_edge, bottom_edge);
  }

  if (((top_edge >= mi_row) && (top_edge < (mi_row + mi_step))) ||
      ((bottom_edge >= mi_row) && (bottom_edge < (mi_row + mi_step)))) {
    is_active_h_edge = 1;
  }
  return is_active_h_edge;
}

int av1_active_v_edge(const AV1_COMP *cpi, int mi_col, int mi_step) {
  int left_edge = 0;
  int right_edge = cpi->common.mi_cols;
  int is_active_v_edge = 0;

  // For two pass account for any formatting bars detected.
  if (is_stat_consumption_stage_twopass(cpi)) {
    const AV1_COMMON *const cm = &cpi->common;
    const FIRSTPASS_STATS *const this_frame_stats = read_one_frame_stats(
        &cpi->twopass, cm->current_frame.display_order_hint);
    if (this_frame_stats == NULL) return AOM_CODEC_ERROR;

    // The inactive region is specified in MBs not mi units.
    // The image edge is in the following MB row.
    left_edge += (int)(this_frame_stats->inactive_zone_cols * 4);

    right_edge -= (int)(this_frame_stats->inactive_zone_cols * 4);
    right_edge = AOMMAX(left_edge, right_edge);
  }

  if (((left_edge >= mi_col) && (left_edge < (mi_col + mi_step))) ||
      ((right_edge >= mi_col) && (right_edge < (mi_col + mi_step)))) {
    is_active_v_edge = 1;
  }
  return is_active_v_edge;
}

static INLINE void set_default_partition_cost(
    PartitionSearchState *search_state, MACROBLOCK *const x) {
#if CONFIG_EXT_RECUR_PARTITIONS
  const PartitionBlkParams *blk_params = &search_state->part_blk_params;
  const BLOCK_SIZE bsize = blk_params->bsize;
  const int pl = search_state->pl;
  (void)x;
  if (is_square_block(bsize)) {
    search_state->partition_cost =
        pl >= 0 ? x->partition_cost[pl] : x->partition_cost[0];
  } else {
    int tmp_pl = pl >= 0 ? pl : 0;
    for (PARTITION_TYPE p = PARTITION_NONE; p < EXT_PARTITION_TYPES; ++p) {
      PARTITION_TYPE_REC p_rec = get_symbol_from_partition_rec_block(bsize, p);

      if (p_rec != PARTITION_INVALID_REC)
        search_state->partition_cost_table[p] =
            x->partition_rec_cost[tmp_pl][p_rec];
      else
        search_state->partition_cost_table[p] = INT_MAX;
    }
    search_state->partition_cost = search_state->partition_cost_table;
  }
#else
  search_state->partition_cost = search_state->pl >= 0
                                     ? x->partition_cost[search_state->pl]
                                     : x->partition_cost[0];
#endif  // CONFIG_EXT_RECUR_PARTITIONS
}

static INLINE void set_partition_cost_for_edge_blk(
    PartitionSearchState *search_state, const AV1_COMMON *const cm) {
  const PartitionBlkParams *blk_params = &search_state->part_blk_params;
#if CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT
  const int has_rows = blk_params->has_rows;
  const int has_cols = blk_params->has_cols;
  (void)cm;
  if (!(has_rows && has_cols)) {
    assert(search_state->is_block_splittable && search_state->pl >= 0);
    if (!has_rows && !has_cols) {
      // At the bottom right, horz or vert
      aom_cdf_prob binary_cdf[2] = { 16384, AOM_ICDF(CDF_PROB_TOP) };
      static const int binary_inv_map[2] = { PARTITION_HORZ, PARTITION_VERT };
      av1_cost_tokens_from_cdf(search_state->tmp_partition_cost, binary_cdf,
                               binary_inv_map);
    } else {
      for (int i = 0; i < PARTITION_TYPES; ++i)
        search_state->tmp_partition_cost[i] = 0;
    }
    search_state->partition_cost = search_state->tmp_partition_cost;
  }
#else   // CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT
  const BLOCK_SIZE bsize = blk_params->bsize;
  if (!(blk_params->has_rows && blk_params->has_cols)) {
    assert(search_state->is_block_splittable && search_state->pl >= 0);
    const aom_cdf_prob *partition_cdf = cm->fc->partition_cdf[search_state->pl];
    const int max_cost = av1_cost_symbol(0);
    for (int i = 0; i < PARTITION_TYPES; ++i)
      search_state->tmp_partition_cost[i] = max_cost;
    if (blk_params->has_cols) {
      // At the bottom, the two possibilities are HORZ and SPLIT
      aom_cdf_prob bot_cdf[2];
      partition_gather_vert_alike(bot_cdf, partition_cdf, bsize);
      static const int bot_inv_map[2] = { PARTITION_HORZ, PARTITION_SPLIT };
      av1_cost_tokens_from_cdf(search_state->tmp_partition_cost, bot_cdf,
                               bot_inv_map);
    } else if (blk_params->has_rows) {
      // At the right, the two possibilities are VERT and SPLIT
      aom_cdf_prob rhs_cdf[2];
      partition_gather_horz_alike(rhs_cdf, partition_cdf, bsize);
      static const int rhs_inv_map[2] = { PARTITION_VERT, PARTITION_SPLIT };
      av1_cost_tokens_from_cdf(search_state->tmp_partition_cost, rhs_cdf,
                               rhs_inv_map);
    } else {
      // At the bottom right, we always split
      search_state->tmp_partition_cost[PARTITION_SPLIT] = 0;
    }

    search_state->partition_cost = search_state->tmp_partition_cost;
  }
#endif  // CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT
}

void av1_init_partition_search_state(PartitionSearchState *search_state,
                                     MACROBLOCK *x, AV1_COMP *const cpi,
                                     const PC_TREE *pc_tree, int mi_row,
                                     int mi_col, BLOCK_SIZE bsize,
                                     BLOCK_SIZE max_sq_part,
                                     BLOCK_SIZE min_sq_part) {
  const AV1_COMMON *const cm = &cpi->common;
  const MACROBLOCKD *const xd = &x->e_mbd;

  PartitionBlkParams *blk_params = &search_state->part_blk_params;

  blk_params->bsize = bsize;

  blk_params->mi_row = mi_row;
  blk_params->mi_col = mi_col;

  blk_params->max_sq_part = max_sq_part;
  blk_params->min_sq_part = min_sq_part;
  blk_params->max_partition_size_1d = block_size_wide[max_sq_part];
  blk_params->min_partition_size_1d = block_size_wide[min_sq_part];
  blk_params->width = block_size_wide[bsize];

  blk_params->is_le_min_sq_part =
      blk_params->width <= blk_params->min_partition_size_1d;
  blk_params->is_gt_max_sq_part =
      blk_params->width > blk_params->max_partition_size_1d;

  blk_params->mi_step_w = mi_size_wide[bsize] / 2;
  blk_params->mi_step_h = mi_size_high[bsize] / 2;

  // Override skipping rectangular partition operations for edge blocks
  blk_params->has_rows =
      (blk_params->mi_row + blk_params->mi_step_h < cm->mi_rows);
  blk_params->has_cols =
      (blk_params->mi_col + blk_params->mi_step_w < cm->mi_cols);

  blk_params->ss_x = xd->plane[1].subsampling_x;
  blk_params->ss_y = xd->plane[1].subsampling_y;

  search_state->is_block_splittable = is_partition_point(bsize);
  search_state->pl = search_state->is_block_splittable
                         ? partition_plane_context(xd, mi_row, mi_col, bsize)
                         : 0;

  set_default_partition_cost(search_state, x);

  search_state->none_rd = 0;
  av1_zero(search_state->split_rd);
  av1_zero(search_state->rect_part_rd);

  init_partition_allowed(search_state, cpi, pc_tree);

  // Override partition costs at the edges of the frame in the same
  // way as in read_partition (see decodeframe.c)
  set_partition_cost_for_edge_blk(search_state, cm);

  search_state->do_rectangular_split = cpi->oxcf.enable_rect_partitions;
  search_state->prune_rect_part[HORZ] = 0;
  search_state->prune_rect_part[VERT] = 0;
#if !CONFIG_EXT_RECUR_PARTITIONS
  memset(&search_state->partition_ab_allowed, 0,
         sizeof(search_state->partition_ab_allowed));
#endif  // !CONFIG_EXT_RECUR_PARTITIONS

  if (bsize > cpi->sf.use_square_partition_only_threshold) {
    search_state->partition_rect_allowed[HORZ] &= !blk_params->has_rows;
    search_state->partition_rect_allowed[VERT] &= !blk_params->has_cols;
  }

  search_state->found_best_partition = false;

  av1_zero(search_state->split_ctx_is_ready);
  av1_zero(search_state->rect_ctx_is_ready);

  search_state->found_best_partition = false;

#if CONFIG_EXT_RECUR_PARTITIONS
  init_sms_partition_stats(&search_state->none_data);
#endif  // CONFIG_EXT_RECUR_PARTITIONS
}
#endif  // !CONFIG_REALTIME_ONLY

static int set_deltaq_rdmult(const AV1_COMP *const cpi, MACROBLOCKD *const xd) {
  const AV1_COMMON *const cm = &cpi->common;

  return av1_compute_rd_mult(
      cpi, cm->base_qindex + xd->delta_qindex + cm->y_dc_delta_q);
}

static void set_ssim_rdmult(const AV1_COMP *const cpi, MACROBLOCK *const x,
                            const BLOCK_SIZE bsize, const int mi_row,
                            const int mi_col, int *const rdmult) {
  const AV1_COMMON *const cm = &cpi->common;

  const int bsize_base = BLOCK_16X16;
  const int num_mi_w = mi_size_wide[bsize_base];
  const int num_mi_h = mi_size_high[bsize_base];
  const int num_cols = (cm->mi_cols + num_mi_w - 1) / num_mi_w;
  const int num_rows = (cm->mi_rows + num_mi_h - 1) / num_mi_h;
  const int num_bcols = (mi_size_wide[bsize] + num_mi_w - 1) / num_mi_w;
  const int num_brows = (mi_size_high[bsize] + num_mi_h - 1) / num_mi_h;
  int row, col;
  double num_of_mi = 0.0;
  double geom_mean_of_scale = 0.0;

  assert(cpi->oxcf.tuning == AOM_TUNE_SSIM);

  aom_clear_system_state();
  for (row = mi_row / num_mi_w;
       row < num_rows && row < mi_row / num_mi_w + num_brows; ++row) {
    for (col = mi_col / num_mi_h;
         col < num_cols && col < mi_col / num_mi_h + num_bcols; ++col) {
      const int index = row * num_cols + col;
      geom_mean_of_scale += log(cpi->ssim_rdmult_scaling_factors[index]);
      num_of_mi += 1.0;
    }
  }
  geom_mean_of_scale = exp(geom_mean_of_scale / num_of_mi);

  *rdmult = (int)((double)(*rdmult) * geom_mean_of_scale + 0.5);
  *rdmult = AOMMAX(*rdmult, 0);
  set_error_per_bit(x, *rdmult);
  aom_clear_system_state();
}

static int get_hier_tpl_rdmult(const AV1_COMP *const cpi, MACROBLOCK *const x,
                               const BLOCK_SIZE bsize, const int mi_row,
                               const int mi_col, int orig_rdmult) {
  const AV1_COMMON *const cm = &cpi->common;
  assert(IMPLIES(cpi->gf_group.size > 0,
                 cpi->gf_group.index < cpi->gf_group.size));
  const int tpl_idx = cpi->gf_group.index;
  const TplDepFrame *tpl_frame = &cpi->tpl_frame[tpl_idx];
  MACROBLOCKD *const xd = &x->e_mbd;
  const int deltaq_rdmult = set_deltaq_rdmult(cpi, xd);
  if (cpi->tpl_model_pass == 1) {
    assert(cpi->oxcf.enable_tpl_model == 2);
    return deltaq_rdmult;
  }
  if (tpl_frame->is_valid == 0) return deltaq_rdmult;
  if (!is_frame_tpl_eligible((AV1_COMP *)cpi)) return deltaq_rdmult;
  if (tpl_idx >= MAX_LAG_BUFFERS) return deltaq_rdmult;
  if (cpi->oxcf.superres_mode != SUPERRES_NONE) return deltaq_rdmult;
  if (cpi->oxcf.aq_mode != NO_AQ) return deltaq_rdmult;

  const int bsize_base = BLOCK_16X16;
  const int num_mi_w = mi_size_wide[bsize_base];
  const int num_mi_h = mi_size_high[bsize_base];
  const int num_cols = (cm->mi_cols + num_mi_w - 1) / num_mi_w;
  const int num_rows = (cm->mi_rows + num_mi_h - 1) / num_mi_h;
  const int num_bcols = (mi_size_wide[bsize] + num_mi_w - 1) / num_mi_w;
  const int num_brows = (mi_size_high[bsize] + num_mi_h - 1) / num_mi_h;
  int row, col;
  double base_block_count = 0.0;
  double geom_mean_of_scale = 0.0;
  aom_clear_system_state();
  for (row = mi_row / num_mi_w;
       row < num_rows && row < mi_row / num_mi_w + num_brows; ++row) {
    for (col = mi_col / num_mi_h;
         col < num_cols && col < mi_col / num_mi_h + num_bcols; ++col) {
      const int index = row * num_cols + col;
      geom_mean_of_scale += log(cpi->tpl_sb_rdmult_scaling_factors[index]);
      base_block_count += 1.0;
    }
  }
  geom_mean_of_scale = exp(geom_mean_of_scale / base_block_count);
  int rdmult = (int)((double)orig_rdmult * geom_mean_of_scale + 0.5);
  rdmult = AOMMAX(rdmult, 0);
  set_error_per_bit(x, rdmult);
  aom_clear_system_state();
  if (bsize == cm->seq_params.sb_size) {
    const int rdmult_sb = set_deltaq_rdmult(cpi, xd);
    assert(rdmult_sb == rdmult);
    (void)rdmult_sb;
  }
  return rdmult;
}

static int set_segment_rdmult(const AV1_COMP *const cpi, MACROBLOCK *const x,
                              int8_t segment_id) {
  const AV1_COMMON *const cm = &cpi->common;
  av1_init_plane_quantizers(cpi, x, segment_id);
  aom_clear_system_state();
#if CONFIG_EXTQUANT
  int segment_qindex = av1_get_qindex(&cm->seg, segment_id, cm->base_qindex,
                                      cm->seq_params.bit_depth);
#else
  int segment_qindex = av1_get_qindex(&cm->seg, segment_id, cm->base_qindex);
#endif
  return av1_compute_rd_mult(cpi, segment_qindex + cm->y_dc_delta_q);
}

void av1_setup_block_rdmult(const AV1_COMP *const cpi, MACROBLOCK *const x,
                            int mi_row, int mi_col, BLOCK_SIZE bsize,
                            AQ_MODE aq_mode, MB_MODE_INFO *mbmi) {
  x->rdmult = cpi->rd.RDMULT;

  if (aq_mode != NO_AQ) {
    assert(mbmi != NULL);
    if (aq_mode == VARIANCE_AQ) {
      if (cpi->vaq_refresh) {
        const int energy = bsize <= BLOCK_16X16
                               ? x->mb_energy
                               : av1_log_block_var(cpi, x, bsize);
        mbmi->segment_id = energy;
      }
      x->rdmult = set_segment_rdmult(cpi, x, mbmi->segment_id);
    } else if (aq_mode == COMPLEXITY_AQ) {
      x->rdmult = set_segment_rdmult(cpi, x, mbmi->segment_id);
    } else if (aq_mode == CYCLIC_REFRESH_AQ) {
      // If segment is boosted, use rdmult for that segment.
      if (cyclic_refresh_segment_id_boosted(mbmi->segment_id))
        x->rdmult = av1_cyclic_refresh_get_rdmult(cpi->cyclic_refresh);
    }
  }

  const AV1_COMMON *const cm = &cpi->common;
  if (cm->delta_q_info.delta_q_present_flag) {
    x->rdmult = get_hier_tpl_rdmult(cpi, x, bsize, mi_row, mi_col, x->rdmult);
  }

  if (cpi->oxcf.tuning == AOM_TUNE_SSIM) {
    set_ssim_rdmult(cpi, x, bsize, mi_row, mi_col, &x->rdmult);
  }
}

// Record the ref frames that have been selected by square partition blocks.
void av1_update_picked_ref_frames_mask(MACROBLOCK *const x, int ref_type,
                                       BLOCK_SIZE bsize, int mib_size,
                                       int mi_row, int mi_col) {
  const int sb_size_mask = mib_size - 1;
  const int mi_row_in_sb = mi_row & sb_size_mask;
  const int mi_col_in_sb = mi_col & sb_size_mask;
  const int mi_size_w = mi_size_wide[bsize];
  const int mi_size_h = mi_size_high[bsize];
  for (int i = mi_row_in_sb; i < mi_row_in_sb + mi_size_h; ++i) {
    for (int j = mi_col_in_sb; j < mi_col_in_sb + mi_size_w; ++j) {
      x->picked_ref_frames_mask[i * 32 + j] |= 1 << ref_type;
    }
  }
}

static void update_txfm_count(MACROBLOCK *x, MACROBLOCKD *xd,
                              FRAME_COUNTS *counts, TX_SIZE tx_size, int depth,
                              int blk_row, int blk_col,
                              uint8_t allow_update_cdf) {
  MB_MODE_INFO *mbmi = xd->mi[0];
  const BLOCK_SIZE bsize = mbmi->sb_type;
  const int max_blocks_high = max_block_high(xd, bsize, 0);
  const int max_blocks_wide = max_block_wide(xd, bsize, 0);
  int ctx = txfm_partition_context(xd->above_txfm_context + blk_col,
                                   xd->left_txfm_context + blk_row,
                                   mbmi->sb_type, tx_size);
  const int txb_size_index = av1_get_txb_size_index(bsize, blk_row, blk_col);

  if (blk_row >= max_blocks_high || blk_col >= max_blocks_wide) return;
  assert(tx_size > TX_4X4);
#if CONFIG_NEW_TX_PARTITION
  (void)depth;
  (void)counts;
  TX_SIZE sub_txs[MAX_TX_PARTITIONS] = { 0 };
  get_tx_partition_sizes(mbmi->partition_type[txb_size_index], tx_size,
                         sub_txs);
  // TODO(sarahparker) This assumes all of the tx sizes in the partition scheme
  // are the same size. This will need to be adjusted to deal with the case
  // where they can be different.
  TX_SIZE this_size = sub_txs[0];
  assert(mbmi->inter_tx_size[txb_size_index] == this_size);
  if (mbmi->partition_type[txb_size_index] != TX_PARTITION_NONE)
    ++x->txb_split_count;

  const int is_rect = is_rect_tx(tx_size);
#if CONFIG_ENTROPY_STATS
  ++counts->txfm_partition[is_rect][ctx][mbmi->partition_type[txb_size_index]];
#endif  // CONFIG_ENTROPY_STATS
  if (allow_update_cdf)
    update_cdf(xd->tile_ctx->txfm_partition_cdf[is_rect][ctx],
               mbmi->partition_type[txb_size_index], TX_PARTITION_TYPES);

  mbmi->tx_size = this_size;
  txfm_partition_update(xd->above_txfm_context + blk_col,
                        xd->left_txfm_context + blk_row, this_size, tx_size);
#else  // CONFIG_NEW_TX_PARTITION
  if (depth == MAX_VARTX_DEPTH) {
    // Don't add to counts in this case
    mbmi->tx_size = tx_size;
    txfm_partition_update(xd->above_txfm_context + blk_col,
                          xd->left_txfm_context + blk_row, tx_size, tx_size);
    return;
  }

  const TX_SIZE plane_tx_size = mbmi->inter_tx_size[txb_size_index];
  if (tx_size == plane_tx_size) {
#if CONFIG_ENTROPY_STATS
    ++counts->txfm_partition[ctx][0];
#endif
    if (allow_update_cdf)
      update_cdf(xd->tile_ctx->txfm_partition_cdf[ctx], 0, 2);
    mbmi->tx_size = tx_size;
    txfm_partition_update(xd->above_txfm_context + blk_col,
                          xd->left_txfm_context + blk_row, tx_size, tx_size);
  } else {
    const TX_SIZE sub_txs = sub_tx_size_map[tx_size];
    const int bsw = tx_size_wide_unit[sub_txs];
    const int bsh = tx_size_high_unit[sub_txs];

#if CONFIG_ENTROPY_STATS
    ++counts->txfm_partition[ctx][1];
#endif
    if (allow_update_cdf)
      update_cdf(xd->tile_ctx->txfm_partition_cdf[ctx], 1, 2);
    ++x->txb_split_count;

    if (sub_txs == TX_4X4) {
      mbmi->inter_tx_size[txb_size_index] = TX_4X4;
      mbmi->tx_size = TX_4X4;
      txfm_partition_update(xd->above_txfm_context + blk_col,
                            xd->left_txfm_context + blk_row, TX_4X4, tx_size);
      return;
    }

    for (int row = 0; row < tx_size_high_unit[tx_size]; row += bsh) {
      for (int col = 0; col < tx_size_wide_unit[tx_size]; col += bsw) {
        int offsetr = row;
        int offsetc = col;

        update_txfm_count(x, xd, counts, sub_txs, depth + 1, blk_row + offsetr,
                          blk_col + offsetc, allow_update_cdf);
      }
    }
  }
#endif  // CONFIG_NEW_TX_PARTITION
}

static void tx_partition_count_update(const AV1_COMMON *const cm, MACROBLOCK *x,
                                      BLOCK_SIZE plane_bsize, int mi_row,
                                      int mi_col, FRAME_COUNTS *td_counts,
                                      uint8_t allow_update_cdf) {
  MACROBLOCKD *xd = &x->e_mbd;
  const int mi_width = block_size_wide[plane_bsize] >> tx_size_wide_log2[0];
  const int mi_height = block_size_high[plane_bsize] >> tx_size_high_log2[0];
  const TX_SIZE max_tx_size = get_vartx_max_txsize(xd, plane_bsize, 0);
  const int bh = tx_size_high_unit[max_tx_size];
  const int bw = tx_size_wide_unit[max_tx_size];
  int idx, idy;

  xd->above_txfm_context = cm->above_txfm_context[xd->tile.tile_row] + mi_col;
  xd->left_txfm_context =
      xd->left_txfm_context_buffer + (mi_row & MAX_MIB_MASK);

  for (idy = 0; idy < mi_height; idy += bh)
    for (idx = 0; idx < mi_width; idx += bw)
      update_txfm_count(x, xd, td_counts, max_tx_size, 0, idy, idx,
                        allow_update_cdf);
}

static void set_txfm_context(MACROBLOCKD *xd, TX_SIZE tx_size, int blk_row,
                             int blk_col) {
  MB_MODE_INFO *mbmi = xd->mi[0];
  const BLOCK_SIZE bsize = mbmi->sb_type;
  const int max_blocks_high = max_block_high(xd, bsize, 0);
  const int max_blocks_wide = max_block_wide(xd, bsize, 0);
  const int txb_size_index = av1_get_txb_size_index(bsize, blk_row, blk_col);
  const TX_SIZE plane_tx_size = mbmi->inter_tx_size[txb_size_index];

  if (blk_row >= max_blocks_high || blk_col >= max_blocks_wide) return;

  if (tx_size == plane_tx_size) {
    mbmi->tx_size = tx_size;
    txfm_partition_update(xd->above_txfm_context + blk_col,
                          xd->left_txfm_context + blk_row, tx_size, tx_size);

  } else {
#if CONFIG_NEW_TX_PARTITION
    TX_SIZE sub_txs[MAX_TX_PARTITIONS] = { 0 };
    const int index = av1_get_txb_size_index(bsize, blk_row, blk_col);
    get_tx_partition_sizes(mbmi->partition_type[index], tx_size, sub_txs);
    int cur_partition = 0;
    int bsw = 0, bsh = 0;
    for (int r = 0; r < tx_size_high_unit[tx_size]; r += bsh) {
      for (int c = 0; c < tx_size_wide_unit[tx_size]; c += bsw) {
        const TX_SIZE sub_tx = sub_txs[cur_partition];
        bsw = tx_size_wide_unit[sub_tx];
        bsh = tx_size_high_unit[sub_tx];
        const int offsetr = blk_row + r;
        const int offsetc = blk_col + c;
        if (offsetr >= max_blocks_high || offsetc >= max_blocks_wide) continue;
        mbmi->tx_size = sub_tx;
        txfm_partition_update(xd->above_txfm_context + blk_col,
                              xd->left_txfm_context + blk_row, sub_tx, sub_tx);
        cur_partition++;
      }
    }
#else
    if (tx_size == TX_8X8) {
      mbmi->inter_tx_size[txb_size_index] = TX_4X4;
      mbmi->tx_size = TX_4X4;
      txfm_partition_update(xd->above_txfm_context + blk_col,
                            xd->left_txfm_context + blk_row, TX_4X4, tx_size);
      return;
    }
    const TX_SIZE sub_txs = sub_tx_size_map[tx_size];
    const int bsw = tx_size_wide_unit[sub_txs];
    const int bsh = tx_size_high_unit[sub_txs];
    for (int row = 0; row < tx_size_high_unit[tx_size]; row += bsh) {
      for (int col = 0; col < tx_size_wide_unit[tx_size]; col += bsw) {
        const int offsetr = blk_row + row;
        const int offsetc = blk_col + col;
        if (offsetr >= max_blocks_high || offsetc >= max_blocks_wide) continue;
        set_txfm_context(xd, sub_txs, offsetr, offsetc);
      }
    }
#endif  // CONFIG_NEW_TX_PARTITION
  }
}

static void tx_partition_set_contexts(const AV1_COMMON *const cm,
                                      MACROBLOCKD *xd, BLOCK_SIZE plane_bsize,
                                      int mi_row, int mi_col) {
  const int mi_width = block_size_wide[plane_bsize] >> tx_size_wide_log2[0];
  const int mi_height = block_size_high[plane_bsize] >> tx_size_high_log2[0];
  const TX_SIZE max_tx_size = get_vartx_max_txsize(xd, plane_bsize, 0);
  const int bh = tx_size_high_unit[max_tx_size];
  const int bw = tx_size_wide_unit[max_tx_size];
  int idx, idy;

  xd->above_txfm_context = cm->above_txfm_context[xd->tile.tile_row] + mi_col;
  xd->left_txfm_context =
      xd->left_txfm_context_buffer + (mi_row & MAX_MIB_MASK);

  for (idy = 0; idy < mi_height; idy += bh)
    for (idx = 0; idx < mi_width; idx += bw)
      set_txfm_context(xd, max_tx_size, idy, idx);
}

void av1_encode_superblock(const AV1_COMP *const cpi, TileDataEnc *tile_data,
                           ThreadData *td, TOKENEXTRA **t, RUN_TYPE dry_run,
                           BLOCK_SIZE bsize, int *rate) {
  const AV1_COMMON *const cm = &cpi->common;
  const int num_planes = av1_num_planes(cm);
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  MB_MODE_INFO **mi_4x4 = xd->mi;
  MB_MODE_INFO *mbmi = mi_4x4[0];
  const int seg_skip =
      segfeature_active(&cm->seg, mbmi->segment_id, SEG_LVL_SKIP);
  const int mis = cm->mi_stride;
  const int mi_width = mi_size_wide[bsize];
  const int mi_height = mi_size_high[bsize];
  const int is_inter = is_inter_block(mbmi);
  const int mi_row = xd->mi_row;
  const int mi_col = xd->mi_col;

  if (!is_inter) {
#if CONFIG_DERIVED_INTRA_MODE
    if (mbmi->use_derived_intra_mode[0] || mbmi->use_derived_intra_mode[1]) {
      const int derived_mode = av1_get_derived_intra_mode(xd, bsize, mbmi);
      if (mbmi->use_derived_intra_mode[0]) {
        mbmi->mode = derived_mode;
      }
      if (mbmi->use_derived_intra_mode[1]) {
        mbmi->uv_mode = derived_mode;
      }
    }
#endif  // CONFIG_DERIVED_INTRA_MODE
    xd->cfl.is_chroma_reference = mbmi->chroma_ref_info.is_chroma_ref;
    xd->cfl.store_y = store_cfl_required(cm, xd);
    mbmi->skip = 1;
    for (int plane = 0; plane < num_planes; ++plane) {
      av1_encode_intra_block_plane(cpi, x, bsize, plane, dry_run,
                                   cpi->optimize_seg_arr[mbmi->segment_id]);
    }

    // If there is at least one lossless segment, force the skip for intra
    // block to be 0, in order to avoid the segment_id to be changed by in
    // write_segment_id().
    if (!cpi->common.seg.segid_preskip && cpi->common.seg.update_map &&
        cpi->has_lossless_segment)
      mbmi->skip = 0;

    xd->cfl.store_y = 0;
    if (av1_allow_palette(cm->allow_screen_content_tools, bsize)) {
      for (int plane = 0; plane < AOMMIN(2, num_planes); ++plane) {
        if (mbmi->palette_mode_info.palette_size[plane] > 0) {
          if (!dry_run) {
            av1_tokenize_color_map(x, plane, t, bsize, mbmi->tx_size,
                                   PALETTE_MAP, tile_data->allow_update_cdf,
                                   td->counts);
          } else if (dry_run == DRY_RUN_COSTCOEFFS) {
            rate +=
                av1_cost_color_map(x, plane, bsize, mbmi->tx_size, PALETTE_MAP);
          }
        }
      }
    }

    av1_update_txb_context(cpi, td, dry_run, bsize, rate, mi_row, mi_col,
                           tile_data->allow_update_cdf);
  } else {
    int ref;
    const int is_compound = has_second_ref(mbmi);

    set_ref_ptrs(cm, xd, mbmi->ref_frame[0], mbmi->ref_frame[1]);
    for (ref = 0; ref < 1 + is_compound; ++ref) {
      const YV12_BUFFER_CONFIG *cfg =
          get_ref_frame_yv12_buf(cm, mbmi->ref_frame[ref]);
      assert(IMPLIES(!is_intrabc_block(mbmi), cfg));
      av1_setup_pre_planes(xd, ref, cfg, mi_row, mi_col,
                           xd->block_ref_scale_factors[ref], num_planes,
                           &mbmi->chroma_ref_info);
    }
#if CONFIG_NEW_INTER_MODES && DISABLE_NEW_INTER_MODES_JOINT_ZERO
    assert(av1_check_newmv_joint_nonzero(cm, x));
#endif  // CONFIG_NEW_INTER_MODES && DISABLE_NEW_INTER_MODES_JOINT_ZERO
#if CONFIG_EXT_IBC_MODES
    /* Log IBC Statistics
    if (is_intrabc_block(mbmi) && !dry_run) {
      av1_log_ibc_statistics(mbmi, bsize);
    }
    */
#endif  // CONFIG_EXT_IBC_MODES

#if CONFIG_DERIVED_MV
    assert(mbmi->derived_mv_allowed == av1_derived_mv_allowed(xd, mbmi));
    if (mbmi->derived_mv_allowed && mbmi->use_derived_mv) {
      MV derived_mv[2];
      int need_update = 0;
      for (ref = 0; ref < 1 + is_compound; ++ref) {
        derived_mv[ref] = av1_derive_mv(cm, xd, ref, mbmi, xd->plane[0].dst.buf,
                                        xd->plane[0].dst.stride);
#if CONFIG_DERIVED_MV_NO_PD
        if (mbmi->derived_mv[ref].row != derived_mv[ref].row ||
            mbmi->derived_mv[ref].col != derived_mv[ref].col) {
          need_update = 1;
        }
#else
        if (mbmi->mv[ref].as_mv.row != derived_mv[ref].row ||
            mbmi->mv[ref].as_mv.col != derived_mv[ref].col) {
          need_update = 1;
        }
#endif  // CONFIG_DERIVED_MV_NO_PD
      }

      if (need_update) {
        mbmi->derived_mv[0] = derived_mv[0];
        if (is_compound) mbmi->derived_mv[1] = derived_mv[1];
#if !CONFIG_DERIVED_MV_NO_PD
        mbmi->mv[0].as_mv = derived_mv[0];
        if (is_compound) mbmi->mv[1].as_mv = derived_mv[1];
#endif  // !CONFIG_DERIVED_MV_NO_PD
        // Do not use warped motion because the warped motion coefficients may
        // have become invalid due to the MV change.
        if (mbmi->motion_mode == WARPED_CAUSAL) {
          mbmi->motion_mode = SIMPLE_TRANSLATION;
        }
        // Update the frame MV buffers.
        if (!dry_run && cm->seq_params.order_hint_info.enable_ref_frame_mvs) {
          const int bw = mi_size_wide[bsize];
          const int bh = mi_size_high[bsize];
          const int x_mis = AOMMIN(bw, cm->mi_cols - mi_col);
          const int y_mis = AOMMIN(bh, cm->mi_rows - mi_row);
          av1_copy_frame_mvs(cm, mbmi, mi_row, mi_col, x_mis, y_mis);
        }
      }
    }
#endif  // CONFIG_DERIVED_MV

#if CONFIG_SKIP_INTERP_FILTER
    av1_validate_interp_filter(cm, mbmi);
#endif  // CONFIG_SKIP_INTERP_FILTER

#if CONFIG_INTERINTRA_ML_DATA_COLLECT
    if (dry_run == OUTPUT_ENABLED &&
        av1_interintra_ml_data_collect_valid(x, bsize)) {
      av1_interintra_ml_data_collect(cpi, x, bsize);
    }
#endif  // CONFIG_INTERINTRA_ML_DATA_COLLECT
    av1_enc_build_inter_predictor(cm, xd, mi_row, mi_col, NULL, bsize, 0,
                                  av1_num_planes(cm) - 1);
    if (mbmi->motion_mode == OBMC_CAUSAL) {
      assert(cpi->oxcf.enable_obmc == 1);
      av1_build_obmc_inter_predictors_sb(cm, xd);
    }

#if CONFIG_MISMATCH_DEBUG
    if (dry_run == OUTPUT_ENABLED) {
      for (int plane = 0; plane < num_planes; ++plane) {
        const struct macroblockd_plane *pd = &xd->plane[plane];
        int pixel_c, pixel_r;
        mi_to_pixel_loc(&pixel_c, &pixel_r, mi_col, mi_row, 0, 0,
                        pd->subsampling_x, pd->subsampling_y);
        if (plane && !mbmi->chroma_ref_info.is_chroma_ref) continue;
        mismatch_record_block_pre(pd->dst.buf, pd->dst.stride,
                                  cm->current_frame.order_hint, plane, pixel_c,
                                  pixel_r, pd->width, pd->height,
                                  xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH);
      }
    }
#else
    (void)num_planes;
#endif

#if CONFIG_DSPL_RESIDUAL
    // Set quantizer
    av1_setup_dspl_quantizer(cpi, x, mbmi->segment_id, mbmi->dspl_type);
#endif
    av1_encode_inter_txfm_block(cpi, x, mi_row, mi_col, dry_run);
    av1_tokenize_sb_tx_size(cpi, td, t, dry_run, mi_row, mi_col, bsize, rate,
                            tile_data->allow_update_cdf);
#if CONFIG_DSPL_RESIDUAL
    // Restore quantizer
    av1_setup_dspl_quantizer(cpi, x, mbmi->segment_id, DSPL_NONE);
#endif
  }

#if CONFIG_INTRA_ENTROPY && !CONFIG_USE_SMALL_MODEL
  if (frame_is_intra_only(cm)) {
    av1_get_gradient_hist(xd, mbmi, bsize);
    av1_get_recon_var(xd, mbmi, bsize);
  }
#endif  // CONFIG_INTRA_ENTROPY

  if (!dry_run) {
    if (av1_allow_intrabc(cm) && is_intrabc_block(mbmi)) td->intrabc_used = 1;
    if (cm->tx_mode == TX_MODE_SELECT && !xd->lossless[mbmi->segment_id] &&
        mbmi->sb_type > BLOCK_4X4 && !(is_inter && (mbmi->skip || seg_skip))) {
      if (is_inter) {
        tx_partition_count_update(cm, x, bsize, mi_row, mi_col, td->counts,
                                  tile_data->allow_update_cdf);
      } else {
        const TX_SIZE max_tx_size = max_txsize_rect_lookup[bsize];
        if (mbmi->tx_size != max_tx_size) ++x->txb_split_count;
        if (block_signals_txsize(bsize)) {
          const int tx_size_ctx = get_tx_size_context(xd);
#if CONFIG_NEW_TX_PARTITION
          const int is_rect = is_rect_tx(max_tx_size);
          if (tile_data->allow_update_cdf)
            update_cdf(xd->tile_ctx->tx_size_cdf[is_rect][tx_size_ctx],
                       mbmi->partition_type[0], TX_PARTITION_TYPES_INTRA);
#if CONFIG_ENTROPY_STATS
          ++td->counts
                ->intra_tx_size[is_rect][tx_size_ctx][mbmi->partition_type[0]];
#endif
#else  // CONFIG_NEW_TX_PARTITION
          const int32_t tx_size_cat = bsize_to_tx_size_cat(bsize);
          const int depth = tx_size_to_depth(mbmi->tx_size, bsize);
          const int max_depths = bsize_to_max_depth(bsize);

          if (tile_data->allow_update_cdf)
            update_cdf(xd->tile_ctx->tx_size_cdf[tx_size_cat][tx_size_ctx],
                       depth, max_depths + 1);
#if CONFIG_ENTROPY_STATS
          ++td->counts->intra_tx_size[tx_size_cat][tx_size_ctx][depth];
#endif
#endif  // CONFIG_NEW_TX_PARTITION
        }
      }
      assert(IMPLIES(is_rect_tx(mbmi->tx_size), is_rect_tx_allowed(xd, mbmi)));
    } else {
      int i, j;
      TX_SIZE intra_tx_size;
      // The new intra coding scheme requires no change of transform size
      if (is_inter) {
        if (xd->lossless[mbmi->segment_id]) {
          intra_tx_size = TX_4X4;
        } else {
          intra_tx_size = tx_size_from_tx_mode(bsize, cm->tx_mode);
        }
      } else {
        intra_tx_size = mbmi->tx_size;
      }

      for (j = 0; j < mi_height; j++)
        for (i = 0; i < mi_width; i++)
          if (mi_col + i < cm->mi_cols && mi_row + j < cm->mi_rows)
            mi_4x4[mis * j + i]->tx_size = intra_tx_size;

      if (intra_tx_size != max_txsize_rect_lookup[bsize]) ++x->txb_split_count;
    }
#if CONFIG_NN_RECON
    if (av1_is_block_nn_recon_eligible(cm, mbmi, mbmi->tx_size)) {
      if (tile_data->allow_update_cdf) {
        update_cdf(xd->tile_ctx->use_nn_recon_cdf, mbmi->use_nn_recon, 2 + 1);
      }
    }
#endif  // CONFIG_NN_RECON
#if CONFIG_REF_MV_BANK
    if (is_inter) av1_update_ref_mv_bank(xd, mbmi, cm->seq_params.mib_size);
#endif  // CONFIG_REF_MV_BANK
  }

  if (cm->tx_mode == TX_MODE_SELECT && block_signals_txsize(mbmi->sb_type) &&
      is_inter && !(mbmi->skip || seg_skip) &&
      !xd->lossless[mbmi->segment_id]) {
    if (dry_run) tx_partition_set_contexts(cm, xd, bsize, mi_row, mi_col);
  } else {
    TX_SIZE tx_size = mbmi->tx_size;
    // The new intra coding scheme requires no change of transform size
    if (is_inter) {
      if (xd->lossless[mbmi->segment_id]) {
        tx_size = TX_4X4;
      } else {
        tx_size = tx_size_from_tx_mode(bsize, cm->tx_mode);
      }
    } else {
      tx_size = (bsize > BLOCK_4X4) ? tx_size : TX_4X4;
    }
    mbmi->tx_size = tx_size;
    set_txfm_ctxs(tx_size, xd->n4_w, xd->n4_h,
                  (mbmi->skip || seg_skip) && is_inter_block(mbmi), xd);
  }
  if (is_inter_block(mbmi) && !mbmi->chroma_ref_info.is_chroma_ref &&
      is_cfl_allowed(xd)) {
    cfl_store_block(xd, mbmi->sb_type, mbmi->tx_size);
  }

  av1_mark_block_as_coded(xd, mi_row, mi_col, bsize, cm->seq_params.sb_size);
}

static void update_filter_type_count(FRAME_COUNTS *counts,
                                     const MACROBLOCKD *xd,
                                     const MB_MODE_INFO *mbmi) {
  int dir;
  for (dir = 0; dir < 2; ++dir) {
    const int ctx = av1_get_pred_context_switchable_interp(xd, dir);
    InterpFilter filter = av1_extract_interp_filter(mbmi->interp_filters, dir);
    if (counts) ++counts->switchable_interp[ctx][filter];
  }
}

static void update_filter_type_cdf(const MACROBLOCKD *xd,
                                   const MB_MODE_INFO *mbmi,
                                   int enable_dual_filter) {
  for (int dir = 0; dir < 2; ++dir) {
    if (dir && !enable_dual_filter) break;
#if CONFIG_SKIP_INTERP_FILTER
    if (!av1_mv_has_subpel(mbmi, dir, enable_dual_filter)) continue;
#endif  // CONFIG_SKIP_INTERP_FILTER
    const int ctx = av1_get_pred_context_switchable_interp(xd, dir);
    InterpFilter filter = av1_extract_interp_filter(mbmi->interp_filters, dir);
    update_cdf(xd->tile_ctx->switchable_interp_cdf[ctx], filter,
               SWITCHABLE_FILTERS);
  }
}

static void update_global_motion_used(PREDICTION_MODE mode, BLOCK_SIZE bsize,
                                      const MB_MODE_INFO *mbmi,
                                      RD_COUNTS *rdc) {
  if (mode == GLOBALMV || mode == GLOBAL_GLOBALMV) {
    const int num_4x4s = mi_size_wide[bsize] * mi_size_high[bsize];
    int ref;
    for (ref = 0; ref < 1 + has_second_ref(mbmi); ++ref) {
      rdc->global_motion_used[mbmi->ref_frame[ref]] += num_4x4s;
    }
  }
}

#if CONFIG_FLEX_MVRES
static void update_reduced_mv_precision_used(const AV1_COMMON *const cm,
                                             const MB_MODE_INFO *mbmi,
                                             RD_COUNTS *rdc) {
  if (!is_pb_mv_precision_active(cm, mbmi->mode, mbmi->max_mv_precision))
    return;
  assert(av1_get_mbmi_mv_precision(cm, mbmi) == mbmi->pb_mv_precision);
  rdc->reduced_mv_precision_used[mbmi->max_mv_precision -
                                 mbmi->pb_mv_precision]++;
}
#endif  // CONFIG_FLEX_MVRES

static void reset_tx_size(MACROBLOCK *x, MB_MODE_INFO *mbmi,
                          const TX_MODE tx_mode) {
  MACROBLOCKD *const xd = &x->e_mbd;
  if (xd->lossless[mbmi->segment_id]) {
    mbmi->tx_size = TX_4X4;
  } else if (tx_mode != TX_MODE_SELECT) {
    mbmi->tx_size = tx_size_from_tx_mode(mbmi->sb_type, tx_mode);
  } else {
    BLOCK_SIZE bsize = mbmi->sb_type;
    TX_SIZE min_tx_size = depth_to_tx_size(MAX_TX_DEPTH, bsize);
    mbmi->tx_size = (TX_SIZE)TXSIZEMAX(mbmi->tx_size, min_tx_size);
  }
  if (is_inter_block(mbmi)) {
    memset(mbmi->inter_tx_size, mbmi->tx_size, sizeof(mbmi->inter_tx_size));
  }
  memset(mbmi->txk_type, DCT_DCT, sizeof(mbmi->txk_type[0]) * TXK_TYPE_BUF_LEN);
  av1_zero(x->blk_skip);
  x->skip = 0;
}

void av1_update_state(const AV1_COMP *const cpi, ThreadData *td,
                      const PICK_MODE_CONTEXT *const ctx, int mi_row,
                      int mi_col, BLOCK_SIZE bsize, RUN_TYPE dry_run) {
  int i, x_idx, y;
  const AV1_COMMON *const cm = &cpi->common;
  const int num_planes = av1_num_planes(cm);
  RD_COUNTS *const rdc = &td->rd_counts;
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  struct macroblock_plane *const p = x->plane;
  struct macroblockd_plane *const pd = xd->plane;
  const MB_MODE_INFO *const mi = &ctx->mic;
  MB_MODE_INFO *const mi_addr = xd->mi[0];
  const struct segmentation *const seg = &cm->seg;
  const int bw = mi_size_wide[mi->sb_type];
  const int bh = mi_size_high[mi->sb_type];
  const int mis = cm->mi_stride;
  const int mi_width = mi_size_wide[bsize];
  const int mi_height = mi_size_high[bsize];

  assert(mi->sb_type == bsize);

  *mi_addr = *mi;
  *x->mbmi_ext = ctx->mbmi_ext;

  memcpy(x->blk_skip, ctx->blk_skip, sizeof(x->blk_skip[0]) * ctx->num_4x4_blk);

  x->skip = ctx->rd_stats.skip;

  // If segmentation in use
  if (seg->enabled) {
    // For in frame complexity AQ copy the segment id from the segment map.
    if (cpi->oxcf.aq_mode == COMPLEXITY_AQ) {
      const uint8_t *const map =
          seg->update_map ? cpi->segmentation_map : cm->last_frame_seg_map;
      mi_addr->segment_id =
          map ? get_segment_id(cm, map, bsize, mi_row, mi_col) : 0;
      reset_tx_size(x, mi_addr, cm->tx_mode);
    }
    // Else for cyclic refresh mode update the segment map, set the segment id
    // and then update the quantizer.
    if (cpi->oxcf.aq_mode == CYCLIC_REFRESH_AQ) {
      av1_cyclic_refresh_update_segment(cpi, mi_addr, mi_row, mi_col, bsize,
                                        ctx->rd_stats.rate, ctx->rd_stats.dist,
                                        x->skip);
    }
    if (mi_addr->uv_mode == UV_CFL_PRED && !is_cfl_allowed(xd))
      mi_addr->uv_mode = UV_DC_PRED;
  }

  for (i = 0; i < num_planes; ++i) {
    p[i].coeff = ctx->coeff[i];
    p[i].qcoeff = ctx->qcoeff[i];
    pd[i].dqcoeff = ctx->dqcoeff[i];
    p[i].eobs = ctx->eobs[i];
    p[i].txb_entropy_ctx = ctx->txb_entropy_ctx[i];
  }
  for (i = 0; i < 2; ++i) pd[i].color_index_map = ctx->color_index_map[i];
  // Restore the coding context of the MB to that that was in place
  // when the mode was picked for it
  for (y = 0; y < mi_height; y++)
    for (x_idx = 0; x_idx < mi_width; x_idx++)
      if ((xd->mb_to_right_edge >> (3 + MI_SIZE_LOG2)) + mi_width > x_idx &&
          (xd->mb_to_bottom_edge >> (3 + MI_SIZE_LOG2)) + mi_height > y) {
        xd->mi[x_idx + y * mis] = mi_addr;
      }

  if (cpi->oxcf.aq_mode) av1_init_plane_quantizers(cpi, x, mi_addr->segment_id);

  if (dry_run) return;

#if CONFIG_INTERNAL_STATS
  {
    unsigned int *const mode_chosen_counts =
        (unsigned int *)cpi->mode_chosen_counts;  // Cast const away.
    if (frame_is_intra_only(cm)) {
      static const int kf_mode_index[] = {
        THR_DC /*DC_PRED*/,
        THR_V_PRED /*V_PRED*/,
        THR_H_PRED /*H_PRED*/,
        THR_D45_PRED /*D45_PRED*/,
        THR_D135_PRED /*D135_PRED*/,
        THR_D113_PRED /*D113_PRED*/,
        THR_D157_PRED /*D157_PRED*/,
        THR_D203_PRED /*D203_PRED*/,
        THR_D67_PRED /*D67_PRED*/,
        THR_SMOOTH,   /*SMOOTH_PRED*/
        THR_SMOOTH_V, /*SMOOTH_V_PRED*/
        THR_SMOOTH_H, /*SMOOTH_H_PRED*/
        THR_PAETH /*PAETH_PRED*/,
      };
      ++mode_chosen_counts[kf_mode_index[mi_addr->mode]];
    } else {
      // Note how often each mode chosen as best
      ++mode_chosen_counts[ctx->best_mode_index];
    }
  }
#endif
  if (!frame_is_intra_only(cm)) {
    if (is_inter_block(mi_addr)) {
      // TODO(sarahparker): global motion stats need to be handled per-tile
      // to be compatible with tile-based threading.
      update_global_motion_used(mi_addr->mode, bsize, mi_addr, rdc);
#if CONFIG_FLEX_MVRES
      update_reduced_mv_precision_used(cm, mi_addr, rdc);
#endif  // CONFIG_FLEX_MVRES
    }

    if (cm->interp_filter == SWITCHABLE &&
        mi_addr->motion_mode != WARPED_CAUSAL &&
        !is_nontrans_global_motion(xd, xd->mi[0])) {
      update_filter_type_count(td->counts, xd, mi_addr);
    }

    rdc->comp_pred_diff[SINGLE_REFERENCE] += ctx->single_pred_diff;
    rdc->comp_pred_diff[COMPOUND_REFERENCE] += ctx->comp_pred_diff;
    rdc->comp_pred_diff[REFERENCE_MODE_SELECT] += ctx->hybrid_pred_diff;
  }

  const int x_mis = AOMMIN(bw, cm->mi_cols - mi_col);
  const int y_mis = AOMMIN(bh, cm->mi_rows - mi_row);
  if (cm->seq_params.order_hint_info.enable_ref_frame_mvs)
    av1_copy_frame_mvs(cm, mi, mi_row, mi_col, x_mis, y_mis);
}

#if CONFIG_NEW_INTER_MODES
static void update_drl_index_stats(FRAME_CONTEXT *fc, FRAME_COUNTS *counts,
                                   const AV1_COMMON *cm,
                                   const MB_MODE_INFO *mbmi,
                                   const MB_MODE_INFO_EXT *mbmi_ext,
                                   int16_t mode_ctx, uint8_t allow_update_cdf) {
  (void)cm;
#if !CONFIG_ENTROPY_STATS
  (void)counts;
#endif  // !CONFIG_ENTROPY_STATS
#if CONFIG_DERIVED_MV
  if (mbmi->derived_mv_allowed && mbmi->use_derived_mv) return;
#endif  // CONFIG_DERIVED_MV
  assert(have_drl_index(mbmi->mode));
  uint8_t ref_frame_type = av1_ref_frame_type(mbmi->ref_frame);
#if CONFIG_FLEX_MVRES && ADJUST_DRL_FLEX_MVRES
  if (mbmi->pb_mv_precision < mbmi->max_mv_precision &&
      (mbmi->mode == NEWMV || mbmi->mode == NEW_NEWMV)) {
    assert(mbmi->ref_mv_idx_adj < mbmi_ext->ref_mv_info.ref_mv_count_adj);
    assert(mbmi->ref_mv_idx_adj < MAX_DRL_BITS + 1);
    const int range_adj =
        AOMMIN(mbmi_ext->ref_mv_info.ref_mv_count_adj - 1, MAX_DRL_BITS);
    for (int idx = 0; idx < range_adj; ++idx) {
      aom_cdf_prob *drl_cdf =
          av1_get_drl_cdf(mode_ctx, fc, mbmi->mode,
                          mbmi_ext->ref_mv_info.ref_mv_weight_adj, idx);
      if (allow_update_cdf) update_cdf(drl_cdf, mbmi->ref_mv_idx_adj != idx, 2);
      if (mbmi->ref_mv_idx_adj == idx) break;
    }
    return;
  }
#endif  // CONFIG_FLEX_MVRES && ADJUST_DRL_FLEX_MVRES
  assert(mbmi->ref_mv_idx < MAX_DRL_BITS + 1);
  const int range = AOMMIN(
      mbmi_ext->ref_mv_info.ref_mv_count[ref_frame_type] - 1, MAX_DRL_BITS);
  for (int idx = 0; idx < range; ++idx) {
    aom_cdf_prob *drl_cdf = av1_get_drl_cdf(
        mode_ctx, fc, mbmi->mode,
        mbmi_ext->ref_mv_info.ref_mv_weight[ref_frame_type], idx);
#if CONFIG_ENTROPY_STATS
    int drl_ctx =
        av1_drl_ctx(mbmi_ext->ref_mv_info.ref_mv_weight[ref_frame_type], idx);
    switch (mbmi->ref_mv_idx) {
      case 0: counts->drl0_mode[drl_ctx][mbmi->ref_mv_idx != idx]++; break;
      case 1: counts->drl1_mode[drl_ctx][mbmi->ref_mv_idx != idx]++; break;
      default: counts->drl2_mode[drl_ctx][mbmi->ref_mv_idx != idx]++; break;
    }
#endif  // CONFIG_ENTROPY_STATS
    if (allow_update_cdf) update_cdf(drl_cdf, mbmi->ref_mv_idx != idx, 2);
    if (mbmi->ref_mv_idx == idx) break;
  }
}
#endif  // CONFIG_NEW_INTER_MODES

static void update_inter_mode_stats(FRAME_CONTEXT *fc, FRAME_COUNTS *counts,
                                    PREDICTION_MODE mode, int16_t mode_context,
                                    uint8_t allow_update_cdf) {
  (void)counts;

  int16_t mode_ctx = mode_context & NEWMV_CTX_MASK;
  if (mode == NEWMV) {
#if CONFIG_ENTROPY_STATS
    ++counts->newmv_mode[mode_ctx][0];
#endif
    if (allow_update_cdf) update_cdf(fc->newmv_cdf[mode_ctx], 0, 2);
    return;
  } else {
#if CONFIG_ENTROPY_STATS
    ++counts->newmv_mode[mode_ctx][1];
#endif
    if (allow_update_cdf) update_cdf(fc->newmv_cdf[mode_ctx], 1, 2);

    mode_ctx = (mode_context >> GLOBALMV_OFFSET) & GLOBALMV_CTX_MASK;
    if (mode == GLOBALMV) {
#if CONFIG_ENTROPY_STATS
      ++counts->zeromv_mode[mode_ctx][0];
#endif
      if (allow_update_cdf) update_cdf(fc->zeromv_cdf[mode_ctx], 0, 2);
      return;
    } else {
#if CONFIG_ENTROPY_STATS
      ++counts->zeromv_mode[mode_ctx][1];
#endif
      if (allow_update_cdf) update_cdf(fc->zeromv_cdf[mode_ctx], 1, 2);
#if !CONFIG_NEW_INTER_MODES
      mode_ctx = (mode_context >> REFMV_OFFSET) & REFMV_CTX_MASK;
#if CONFIG_ENTROPY_STATS
      ++counts->refmv_mode[mode_ctx][mode != NEARESTMV];
#endif
      if (allow_update_cdf) {
        update_cdf(fc->refmv_cdf[mode_ctx], mode != NEARESTMV, 2);
      }
#endif  // !CONFIG_NEW_INTER_MODES
    }
  }
}

static void update_palette_cdf(MACROBLOCKD *xd, const MB_MODE_INFO *const mbmi,
                               FRAME_COUNTS *counts, uint8_t allow_update_cdf) {
  FRAME_CONTEXT *fc = xd->tile_ctx;
  const BLOCK_SIZE bsize = mbmi->sb_type;
  const PALETTE_MODE_INFO *const pmi = &mbmi->palette_mode_info;
  const int palette_bsize_ctx = av1_get_palette_bsize_ctx(bsize);

  (void)counts;

  if (mbmi->mode == DC_PRED) {
    const int n = pmi->palette_size[0];
    const int palette_mode_ctx = av1_get_palette_mode_ctx(xd);

#if CONFIG_ENTROPY_STATS
    ++counts->palette_y_mode[palette_bsize_ctx][palette_mode_ctx][n > 0];
#endif
    if (allow_update_cdf)
      update_cdf(fc->palette_y_mode_cdf[palette_bsize_ctx][palette_mode_ctx],
                 n > 0, 2);
    if (n > 0) {
#if CONFIG_ENTROPY_STATS
      ++counts->palette_y_size[palette_bsize_ctx][n - PALETTE_MIN_SIZE];
#endif
      if (allow_update_cdf) {
        update_cdf(fc->palette_y_size_cdf[palette_bsize_ctx],
                   n - PALETTE_MIN_SIZE, PALETTE_SIZES);
      }
    }
  }

  if (mbmi->uv_mode == UV_DC_PRED) {
    const int n = pmi->palette_size[1];
    const int palette_uv_mode_ctx = (pmi->palette_size[0] > 0);

#if CONFIG_ENTROPY_STATS
    ++counts->palette_uv_mode[palette_uv_mode_ctx][n > 0];
#endif
    if (allow_update_cdf)
      update_cdf(fc->palette_uv_mode_cdf[palette_uv_mode_ctx], n > 0, 2);

    if (n > 0) {
#if CONFIG_ENTROPY_STATS
      ++counts->palette_uv_size[palette_bsize_ctx][n - PALETTE_MIN_SIZE];
#endif
      if (allow_update_cdf) {
        update_cdf(fc->palette_uv_size_cdf[palette_bsize_ctx],
                   n - PALETTE_MIN_SIZE, PALETTE_SIZES);
      }
    }
  }
}

static void sum_intra_stats(const AV1_COMMON *const cm, FRAME_COUNTS *counts,
                            MACROBLOCKD *xd, const MB_MODE_INFO *const mbmi,
                            const MB_MODE_INFO *above_mi,
                            const MB_MODE_INFO *left_mi,
#if CONFIG_INTRA_ENTROPY
                            const MB_MODE_INFO *aboveleft_mi,
#endif  // CONFIG_INTRA_ENTROPY
                            const int intraonly, uint8_t allow_update_cdf) {
  FRAME_CONTEXT *fc = xd->tile_ctx;
  const PREDICTION_MODE y_mode = mbmi->mode;
  const UV_PREDICTION_MODE uv_mode = mbmi->uv_mode;
  (void)counts;
  const BLOCK_SIZE bsize = mbmi->sb_type;

  if (intraonly) {
#if CONFIG_ENTROPY_STATS
    const PREDICTION_MODE above = av1_above_block_mode(above_mi);
    const PREDICTION_MODE left = av1_left_block_mode(left_mi);
    const int above_ctx = intra_mode_context[above];
    const int left_ctx = intra_mode_context[left];
    ++counts->kf_y_mode[above_ctx][left_ctx][y_mode];
#endif  // CONFIG_ENTROPY_STATS
    if (allow_update_cdf) {
#if CONFIG_INTRA_ENTROPY
      NN_CONFIG_EM *nn_model = &(fc->intra_y_mode);
      av1_get_intra_block_feature(nn_model->sparse_features,
                                  nn_model->dense_features, above_mi, left_mi,
                                  aboveleft_mi);
      av1_nn_predict_em(nn_model);
      av1_nn_backprop_em(nn_model, y_mode);
      av1_nn_update_em(nn_model, nn_model->lr);
#else
#if CONFIG_DERIVED_INTRA_MODE
      const int is_dr_mode = av1_is_directional_mode(y_mode);
      update_cdf(get_kf_is_dr_mode_cdf(fc, above_mi, left_mi), is_dr_mode, 2);
      if (is_dr_mode) {
        int update_intra_mode;
        if (av1_enable_derived_intra_mode(xd, bsize)) {
          update_intra_mode = !mbmi->use_derived_intra_mode[0];
          if (allow_update_cdf) {
            update_cdf(get_derived_intra_mode_cdf(fc, above_mi, left_mi, 0),
                       mbmi->use_derived_intra_mode[0], 2);
          }
        } else {
          update_intra_mode = 1;
        }
        if (update_intra_mode) {
          update_cdf(get_kf_dr_mode_cdf(fc, above_mi, left_mi),
                     dr_mode_to_index[y_mode], DIRECTIONAL_MODES);
        }
      } else {
        update_cdf(get_kf_none_dr_mode_cdf(fc, above_mi, left_mi),
                   none_dr_mode_to_index[y_mode], NONE_DIRECTIONAL_MODES);
      }
#else
      update_cdf(get_y_mode_cdf(fc, above_mi, left_mi), y_mode, INTRA_MODES);
#endif  // CONFIG_DERIVED_INTRA_MODE
#endif  // CONFIG_INTRA_ENTROPY
    }
  } else {
#if CONFIG_ENTROPY_STATS
    ++counts->y_mode[size_group_lookup[bsize]][y_mode];
#endif  // CONFIG_ENTROPY_STATS
    if (allow_update_cdf) {
#if CONFIG_DERIVED_INTRA_MODE
      const int ctx = size_group_lookup[bsize];
      const int is_dr_mode = av1_is_directional_mode(y_mode);
      update_cdf(fc->bf_is_dr_mode_cdf[ctx], is_dr_mode, 2);
      if (is_dr_mode) {
        if (av1_enable_derived_intra_mode(xd, bsize)) {
          update_cdf(get_derived_intra_mode_cdf(fc, above_mi, left_mi, 0),
                     mbmi->use_derived_intra_mode[0], 2);
        } else {
          assert(!mbmi->use_derived_intra_mode[0]);
        }
        if (!mbmi->use_derived_intra_mode[0]) {
          update_cdf(fc->bf_dr_mode_cdf[ctx], dr_mode_to_index[y_mode],
                     DIRECTIONAL_MODES);
        }
      } else {
        update_cdf(fc->bf_none_dr_mode_cdf[ctx], none_dr_mode_to_index[y_mode],
                   NONE_DIRECTIONAL_MODES);
      }
#else
      update_cdf(fc->y_mode_cdf[size_group_lookup[bsize]], y_mode, INTRA_MODES);
#endif  // CONFIG_DERIVED_INTRA_MODE
    }
  }

  if (av1_filter_intra_allowed(cm, mbmi)) {
    const int use_filter_intra_mode =
        mbmi->filter_intra_mode_info.use_filter_intra;
#if CONFIG_ENTROPY_STATS
    ++counts->filter_intra[mbmi->sb_type][use_filter_intra_mode];
    if (use_filter_intra_mode) {
      ++counts
            ->filter_intra_mode[mbmi->filter_intra_mode_info.filter_intra_mode];
    }
#endif  // CONFIG_ENTROPY_STATS
    if (allow_update_cdf) {
      update_cdf(fc->filter_intra_cdfs[mbmi->sb_type], use_filter_intra_mode,
                 2);
      if (use_filter_intra_mode) {
        update_cdf(fc->filter_intra_mode_cdf,
                   mbmi->filter_intra_mode_info.filter_intra_mode,
                   FILTER_INTRA_MODES);
      }
    }
  }
#if CONFIG_ADAPT_FILTER_INTRA
  if (av1_adapt_filter_intra_allowed(cm, mbmi)) {
    const int use_adapt_filter_intra_mode =
        mbmi->adapt_filter_intra_mode_info.use_adapt_filter_intra;
#if CONFIG_ENTROPY_STATS
    ++counts->adapt_filter_intra[mbmi->sb_type][use_adapt_filter_intra_mode];
    if (use_adapt_filter_intra_mode) {
      ++counts->adapt_filter_intra_mode[mbmi->adapt_filter_intra_mode_info
                                            .adapt_filter_intra_mode];
    }
#endif  // CONFIG_ENTROPY_STATS
    if (allow_update_cdf) {
      update_cdf(fc->adapt_filter_intra_cdfs[mbmi->sb_type],
                 use_adapt_filter_intra_mode, 2);
      if (use_adapt_filter_intra_mode) {
        update_cdf(fc->adapt_filter_intra_mode_cdf,
                   mbmi->adapt_filter_intra_mode_info.adapt_filter_intra_mode,
                   USED_ADAPT_FILTER_INTRA_MODES);
      }
    }
  }
#endif  // CONFIG_ADAPT_FILTER_INTRA
  if (av1_is_directional_mode(mbmi->mode) && av1_use_angle_delta(bsize)) {
#if CONFIG_ENTROPY_STATS
    ++counts->angle_delta[mbmi->mode - V_PRED]
                         [mbmi->angle_delta[PLANE_TYPE_Y] + MAX_ANGLE_DELTA];
#endif
    if (allow_update_cdf) {
#if CONFIG_DERIVED_INTRA_MODE
      if (!mbmi->use_derived_intra_mode[0])
#endif  // CONFIG_DERIVED_INTRA_MODE
      {
        update_cdf(fc->angle_delta_cdf[mbmi->mode - V_PRED],
                   mbmi->angle_delta[PLANE_TYPE_Y] + MAX_ANGLE_DELTA,
                   2 * MAX_ANGLE_DELTA + 1);
      }
    }
  }

  if (!mbmi->chroma_ref_info.is_chroma_ref) return;

#if CONFIG_ENTROPY_STATS
  ++counts->uv_mode[is_cfl_allowed(xd)][y_mode][uv_mode];
#endif  // CONFIG_ENTROPY_STATS
  if (allow_update_cdf) {
#if CONFIG_INTRA_ENTROPY
    NN_CONFIG_EM *nn_model = &(fc->intra_uv_mode);
    av1_get_intra_uv_block_feature(nn_model->sparse_features,
                                   nn_model->dense_features, y_mode,
                                   is_cfl_allowed(xd), above_mi, left_mi);
    av1_nn_predict_em(nn_model);
    av1_nn_backprop_em(nn_model, uv_mode);
    av1_nn_update_em(nn_model, nn_model->lr);
#else
#if CONFIG_DERIVED_INTRA_MODE
    if (av1_enable_derived_intra_mode(xd, bsize)) {
      update_cdf(fc->uv_derived_intra_mode_cdf[mbmi->use_derived_intra_mode[0]],
                 mbmi->use_derived_intra_mode[1], 2);
    }
    if (!mbmi->use_derived_intra_mode[1]) {
      const CFL_ALLOWED_TYPE cfl_allowed = is_cfl_allowed(xd);
      update_cdf(fc->uv_mode_cdf[cfl_allowed][y_mode], uv_mode,
                 UV_INTRA_MODES - !cfl_allowed);
    }
#else
    const CFL_ALLOWED_TYPE cfl_allowed = is_cfl_allowed(xd);
    update_cdf(fc->uv_mode_cdf[cfl_allowed][y_mode], uv_mode,
               UV_INTRA_MODES - !cfl_allowed);
#endif  // CONFIG_DERIVED_INTRA_MODE
#endif  // CONFIG_INTRA_ENTROPY
  }
  if (uv_mode == UV_CFL_PRED) {
    const int8_t joint_sign = mbmi->cfl_alpha_signs;
    const uint8_t idx = mbmi->cfl_alpha_idx;

#if CONFIG_ENTROPY_STATS
    ++counts->cfl_sign[joint_sign];
#endif
    if (allow_update_cdf)
      update_cdf(fc->cfl_sign_cdf, joint_sign, CFL_JOINT_SIGNS);
    if (CFL_SIGN_U(joint_sign) != CFL_SIGN_ZERO) {
      aom_cdf_prob *cdf_u = fc->cfl_alpha_cdf[CFL_CONTEXT_U(joint_sign)];

#if CONFIG_ENTROPY_STATS
      ++counts->cfl_alpha[CFL_CONTEXT_U(joint_sign)][CFL_IDX_U(idx)];
#endif
      if (allow_update_cdf)
        update_cdf(cdf_u, CFL_IDX_U(idx), CFL_ALPHABET_SIZE);
    }
    if (CFL_SIGN_V(joint_sign) != CFL_SIGN_ZERO) {
      aom_cdf_prob *cdf_v = fc->cfl_alpha_cdf[CFL_CONTEXT_V(joint_sign)];

#if CONFIG_ENTROPY_STATS
      ++counts->cfl_alpha[CFL_CONTEXT_V(joint_sign)][CFL_IDX_V(idx)];
#endif
      if (allow_update_cdf)
        update_cdf(cdf_v, CFL_IDX_V(idx), CFL_ALPHABET_SIZE);
    }
  }
  if (av1_is_directional_mode(get_uv_mode(uv_mode)) &&
#if CONFIG_DERIVED_INTRA_MODE
      !mbmi->use_derived_intra_mode[1] &&
#endif  // CONFIG_DERIVED_INTRA_MODE
      av1_use_angle_delta(bsize)) {
#if CONFIG_ENTROPY_STATS
    ++counts->angle_delta[uv_mode - UV_V_PRED]
                         [mbmi->angle_delta[PLANE_TYPE_UV] + MAX_ANGLE_DELTA];
#endif
    if (allow_update_cdf) {
      update_cdf(fc->angle_delta_cdf[uv_mode - UV_V_PRED],
                 mbmi->angle_delta[PLANE_TYPE_UV] + MAX_ANGLE_DELTA,
                 2 * MAX_ANGLE_DELTA + 1);
    }
  }
  if (av1_allow_palette(cm->allow_screen_content_tools, bsize))
    update_palette_cdf(xd, mbmi, counts, allow_update_cdf);
}

static INLINE void update_inter_stats(const AV1_COMMON *const cm,
                                      TileDataEnc *tile_data, ThreadData *td) {
  MACROBLOCK *x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  const MB_MODE_INFO *const mbmi = xd->mi[0];
  const MB_MODE_INFO_EXT *const mbmi_ext = x->mbmi_ext;
  const CurrentFrame *const current_frame = &cm->current_frame;
  const BLOCK_SIZE bsize = mbmi->sb_type;
  FRAME_CONTEXT *fc = xd->tile_ctx;
  const uint8_t allow_update_cdf = tile_data->allow_update_cdf;
  RD_COUNTS *rdc = &td->rd_counts;

  FRAME_COUNTS *const counts = td->counts;

  assert(!frame_is_intra_only(cm));

  if (mbmi->skip_mode) {
    rdc->skip_mode_used_flag = 1;
    if (current_frame->reference_mode == REFERENCE_MODE_SELECT) {
      assert(has_second_ref(mbmi));
      rdc->compound_ref_used_flag = 1;
    }
    set_ref_ptrs(cm, xd, mbmi->ref_frame[0], mbmi->ref_frame[1]);
    return;
  }

  const int inter_block = is_inter_block(mbmi);
  const int seg_ref_active =
      segfeature_active(&cm->seg, mbmi->segment_id, SEG_LVL_REF_FRAME);

  if (!seg_ref_active) {
#if CONFIG_ENTROPY_STATS
    counts->intra_inter[av1_get_intra_inter_context(xd)][inter_block]++;
#endif  // CONFIG_ENTROPY_STATS
    if (allow_update_cdf) {
      update_cdf(fc->intra_inter_cdf[av1_get_intra_inter_context(xd)],
                 inter_block, 2);
    }
    // If the segment reference feature is enabled we have only a single
    // reference frame allowed for the segment so exclude it from
    // the reference frame counts used to work out probabilities.
    if (inter_block) {
      const MV_REFERENCE_FRAME ref0 = mbmi->ref_frame[0];
      const MV_REFERENCE_FRAME ref1 = mbmi->ref_frame[1];

      av1_collect_neighbors_ref_counts(xd);

      if (current_frame->reference_mode == REFERENCE_MODE_SELECT) {
        if (has_second_ref(mbmi))
          // This flag is also updated for 4x4 blocks
          rdc->compound_ref_used_flag = 1;
        if (is_comp_ref_allowed(bsize)) {
#if CONFIG_ENTROPY_STATS
          counts->comp_inter[av1_get_reference_mode_context(xd)]
                            [has_second_ref(mbmi)]++;
#endif  // CONFIG_ENTROPY_STATS
          if (allow_update_cdf) {
            update_cdf(av1_get_reference_mode_cdf(xd), has_second_ref(mbmi), 2);
          }
        }
      }

      int update_single_ref_cdf = 1;
#if CONFIG_MISC_CHANGES
      if (cm->only_one_ref_available) update_single_ref_cdf = 0;
#endif  // CONFIG_MISC_CHANGES
      if (has_second_ref(mbmi)) {
        const COMP_REFERENCE_TYPE comp_ref_type = has_uni_comp_refs(mbmi)
                                                      ? UNIDIR_COMP_REFERENCE
                                                      : BIDIR_COMP_REFERENCE;
        if (allow_update_cdf) {
          update_cdf(av1_get_comp_reference_type_cdf(xd), comp_ref_type,
                     COMP_REFERENCE_TYPES);
        }
#if CONFIG_ENTROPY_STATS
        counts->comp_ref_type[av1_get_comp_reference_type_context(xd)]
                             [comp_ref_type]++;
#endif  // CONFIG_ENTROPY_STATS

        if (comp_ref_type == UNIDIR_COMP_REFERENCE) {
          const int bit = (ref0 == BWDREF_FRAME);
          if (allow_update_cdf)
            update_cdf(av1_get_pred_cdf_uni_comp_ref_p(xd), bit, 2);
#if CONFIG_ENTROPY_STATS
          counts
              ->uni_comp_ref[av1_get_pred_context_uni_comp_ref_p(xd)][0][bit]++;
#endif  // CONFIG_ENTROPY_STATS
          if (!bit) {
            const int bit1 = (ref1 == LAST3_FRAME || ref1 == GOLDEN_FRAME);
            if (allow_update_cdf)
              update_cdf(av1_get_pred_cdf_uni_comp_ref_p1(xd), bit1, 2);
#if CONFIG_ENTROPY_STATS
            counts->uni_comp_ref[av1_get_pred_context_uni_comp_ref_p1(xd)][1]
                                [bit1]++;
#endif  // CONFIG_ENTROPY_STATS
            if (bit1) {
              if (allow_update_cdf) {
                update_cdf(av1_get_pred_cdf_uni_comp_ref_p2(xd),
                           ref1 == GOLDEN_FRAME, 2);
              }
#if CONFIG_ENTROPY_STATS
              counts->uni_comp_ref[av1_get_pred_context_uni_comp_ref_p2(xd)][2]
                                  [ref1 == GOLDEN_FRAME]++;
#endif  // CONFIG_ENTROPY_STATS
            }
          }
        } else {
          const int bit = (ref0 == GOLDEN_FRAME || ref0 == LAST3_FRAME);
          if (allow_update_cdf)
            update_cdf(av1_get_pred_cdf_comp_ref_p(xd), bit, 2);
#if CONFIG_ENTROPY_STATS
          counts->comp_ref[av1_get_pred_context_comp_ref_p(xd)][0][bit]++;
#endif  // CONFIG_ENTROPY_STATS
          if (!bit) {
            if (allow_update_cdf) {
              update_cdf(av1_get_pred_cdf_comp_ref_p1(xd), ref0 == LAST2_FRAME,
                         2);
            }
#if CONFIG_ENTROPY_STATS
            counts->comp_ref[av1_get_pred_context_comp_ref_p1(xd)][1]
                            [ref0 == LAST2_FRAME]++;
#endif  // CONFIG_ENTROPY_STATS
          } else {
            if (allow_update_cdf) {
              update_cdf(av1_get_pred_cdf_comp_ref_p2(xd), ref0 == GOLDEN_FRAME,
                         2);
            }
#if CONFIG_ENTROPY_STATS
            counts->comp_ref[av1_get_pred_context_comp_ref_p2(xd)][2]
                            [ref0 == GOLDEN_FRAME]++;
#endif  // CONFIG_ENTROPY_STATS
          }
          if (allow_update_cdf) {
            update_cdf(av1_get_pred_cdf_comp_bwdref_p(xd), ref1 == ALTREF_FRAME,
                       2);
          }
#if CONFIG_ENTROPY_STATS
          counts->comp_bwdref[av1_get_pred_context_comp_bwdref_p(xd)][0]
                             [ref1 == ALTREF_FRAME]++;
#endif  // CONFIG_ENTROPY_STATS
          if (ref1 != ALTREF_FRAME) {
            if (allow_update_cdf) {
              update_cdf(av1_get_pred_cdf_comp_bwdref_p1(xd),
                         ref1 == ALTREF2_FRAME, 2);
            }
#if CONFIG_ENTROPY_STATS
            counts->comp_bwdref[av1_get_pred_context_comp_bwdref_p1(xd)][1]
                               [ref1 == ALTREF2_FRAME]++;
#endif  // CONFIG_ENTROPY_STATS
          }
        }
      } else if (update_single_ref_cdf) {
        const int bit = (ref0 >= BWDREF_FRAME);
        if (allow_update_cdf)
          update_cdf(av1_get_pred_cdf_single_ref_p1(xd), bit, 2);
#if CONFIG_ENTROPY_STATS
        counts->single_ref[av1_get_pred_context_single_ref_p1(xd)][0][bit]++;
#endif  // CONFIG_ENTROPY_STATS
        if (bit) {
          assert(ref0 <= ALTREF_FRAME);
          if (allow_update_cdf) {
            update_cdf(av1_get_pred_cdf_single_ref_p2(xd), ref0 == ALTREF_FRAME,
                       2);
          }
#if CONFIG_ENTROPY_STATS
          counts->single_ref[av1_get_pred_context_single_ref_p2(xd)][1]
                            [ref0 == ALTREF_FRAME]++;
#endif  // CONFIG_ENTROPY_STATS
          if (ref0 != ALTREF_FRAME) {
            if (allow_update_cdf) {
              update_cdf(av1_get_pred_cdf_single_ref_p6(xd),
                         ref0 == ALTREF2_FRAME, 2);
            }
#if CONFIG_ENTROPY_STATS
            counts->single_ref[av1_get_pred_context_single_ref_p6(xd)][5]
                              [ref0 == ALTREF2_FRAME]++;
#endif  // CONFIG_ENTROPY_STATS
          }
        } else {
          const int bit1 = !(ref0 == LAST2_FRAME || ref0 == LAST_FRAME);
          if (allow_update_cdf)
            update_cdf(av1_get_pred_cdf_single_ref_p3(xd), bit1, 2);
#if CONFIG_ENTROPY_STATS
          counts->single_ref[av1_get_pred_context_single_ref_p3(xd)][2][bit1]++;
#endif  // CONFIG_ENTROPY_STATS
          if (!bit1) {
            if (allow_update_cdf) {
              update_cdf(av1_get_pred_cdf_single_ref_p4(xd), ref0 != LAST_FRAME,
                         2);
            }
#if CONFIG_ENTROPY_STATS
            counts->single_ref[av1_get_pred_context_single_ref_p4(xd)][3]
                              [ref0 != LAST_FRAME]++;
#endif  // CONFIG_ENTROPY_STATS
          } else {
            if (allow_update_cdf) {
              update_cdf(av1_get_pred_cdf_single_ref_p5(xd),
                         ref0 != LAST3_FRAME, 2);
            }
#if CONFIG_ENTROPY_STATS
            counts->single_ref[av1_get_pred_context_single_ref_p5(xd)][4]
                              [ref0 != LAST3_FRAME]++;
#endif  // CONFIG_ENTROPY_STATS
          }
        }
      }

#if CONFIG_DERIVED_MV
      if (mbmi->derived_mv_allowed) {
        update_cdf(fc->use_derived_mv_cdf[has_second_ref(mbmi)][bsize],
                   mbmi->use_derived_mv, 2);
      }
#endif  // CONFIG_DERIVED_MV

      if (cm->seq_params.enable_interintra_compound &&
          is_interintra_allowed(mbmi)) {
        const int bsize_group = size_group_lookup[bsize];
        if (mbmi->ref_frame[1] == INTRA_FRAME) {
#if CONFIG_ENTROPY_STATS
          counts->interintra[bsize_group][1]++;
#endif  // CONFIG_ENTROPY_STATS
          if (allow_update_cdf)
            update_cdf(fc->interintra_cdf[bsize_group], 1, 2);
#if CONFIG_ENTROPY_STATS
          counts->interintra_mode[bsize_group][mbmi->interintra_mode]++;
#endif  // CONFIG_ENTROPY_STATS
          if (allow_update_cdf) {
#if CONFIG_DERIVED_INTRA_MODE
            if (av1_enable_derived_intra_mode(xd, bsize)) {
              update_cdf(get_derived_intra_mode_cdf(fc, xd->above_mbmi,
                                                    xd->left_mbmi, 1),
                         mbmi->use_derived_intra_mode[0], 2);
            }
            if (!mbmi->use_derived_intra_mode[0])
#endif  // CONFIG_DERIVED_INTRA_MODE
            {
#if CONFIG_INTERINTRA_ML
              if (is_interintra_ml_supported(xd, mbmi->use_wedge_interintra)) {
                update_cdf(fc->interintra_ml_mode_cdf[bsize_group],
                           mbmi->interintra_mode, INTERINTRA_MODES);
              } else {
                update_cdf(fc->interintra_mode_cdf[bsize_group],
                           mbmi->interintra_mode, II_ML_PRED0);
              }
#else
              update_cdf(fc->interintra_mode_cdf[bsize_group],
                         mbmi->interintra_mode, INTERINTRA_MODES);
#endif  // CONFIG_INTERINTRA_ML
            }
          }
          if (is_interintra_wedge_used(bsize)) {
#if CONFIG_ENTROPY_STATS
            counts->wedge_interintra[bsize][mbmi->use_wedge_interintra]++;
#endif  // CONFIG_ENTROPY_STATS
            if (allow_update_cdf) {
              update_cdf(fc->wedge_interintra_cdf[bsize],
                         mbmi->use_wedge_interintra, 2);
            }
            if (mbmi->use_wedge_interintra) {
#if CONFIG_ENTROPY_STATS
              counts->wedge_idx[bsize][mbmi->interintra_wedge_index]++;
#endif  // CONFIG_ENTROPY_STATS
              if (allow_update_cdf) {
                update_cdf(fc->wedge_idx_cdf[bsize],
                           mbmi->interintra_wedge_index, 16);
              }
            }
          }
        } else {
#if CONFIG_ENTROPY_STATS
          counts->interintra[bsize_group][0]++;
#endif  // CONFIG_ENTROPY_STATS
          if (allow_update_cdf)
            update_cdf(fc->interintra_cdf[bsize_group], 0, 2);
        }
      }

      set_ref_ptrs(cm, xd, mbmi->ref_frame[0], mbmi->ref_frame[1]);

      if (allow_update_cdf) {
        const MOTION_MODE_SET motion_mode_set =
            cm->switchable_motion_mode
                ? av1_get_allowed_motion_mode_set(xd->global_motion, xd, mbmi,
                                                  cm->allow_warped_motion)
                : ONLY_SIMPLE_TRANSLATION;
        if (motion_mode_set != ONLY_SIMPLE_TRANSLATION) {
          int num_modes = 0;
          aom_cdf_prob *cdf =
              av1_get_motion_mode_cdf(xd, motion_mode_set, &num_modes);
          int symbol = mbmi->motion_mode;
#if CONFIG_EXT_WARP && CONFIG_SUB8X8_WARP
          if (motion_mode_set == ALLOW_WARPED_CAUSAL) {
            symbol = mbmi->motion_mode == WARPED_CAUSAL;
          }
#endif  // CONFIG_EXT_WARP && CONFIG_SUB8X8_WARP
          update_cdf(cdf, symbol, num_modes);
        }
      }

#if CONFIG_OPTFLOW_REFINEMENT
      if (has_second_ref(mbmi) && mbmi->mode <= NEW_NEWMV) {
#else
      if (has_second_ref(mbmi)) {
#endif
        assert(current_frame->reference_mode != SINGLE_REFERENCE &&
               is_inter_compound_mode(mbmi->mode) &&
               mbmi->motion_mode == SIMPLE_TRANSLATION);

        const int masked_compound_used = is_any_masked_compound_used(bsize) &&
                                         cm->seq_params.enable_masked_compound;
        if (masked_compound_used) {
          const int comp_group_idx_ctx = get_comp_group_idx_context(xd);
#if CONFIG_ENTROPY_STATS
          ++counts->comp_group_idx[comp_group_idx_ctx][mbmi->comp_group_idx];
#endif  // CONFIG_ENTROPY_STATS
          if (allow_update_cdf) {
            update_cdf(fc->comp_group_idx_cdf[comp_group_idx_ctx],
                       mbmi->comp_group_idx, 2);
          }
        }

        if (mbmi->comp_group_idx == 0) {
          const int comp_index_ctx = get_comp_index_context(cm, xd);
#if CONFIG_ENTROPY_STATS
          ++counts->compound_index[comp_index_ctx][mbmi->compound_idx];
#endif  // CONFIG_ENTROPY_STATS
          if (allow_update_cdf) {
            update_cdf(fc->compound_index_cdf[comp_index_ctx],
                       mbmi->compound_idx, 2);
          }
        } else {
          assert(masked_compound_used);
          if (is_interinter_compound_used(COMPOUND_WEDGE, bsize)) {
#if CONFIG_ENTROPY_STATS
            ++counts->compound_type[bsize][mbmi->interinter_comp.type -
                                           COMPOUND_WEDGE];
#endif  // CONFIG_ENTROPY_STATS
            if (allow_update_cdf) {
              update_cdf(fc->compound_type_cdf[bsize],
                         mbmi->interinter_comp.type - COMPOUND_WEDGE,
                         MASKED_COMPOUND_TYPES);
            }
          }
        }
      }
      if (mbmi->interinter_comp.type == COMPOUND_WEDGE) {
        if (is_interinter_compound_used(COMPOUND_WEDGE, bsize)) {
#if CONFIG_ENTROPY_STATS
          counts->wedge_idx[bsize][mbmi->interinter_comp.wedge_index]++;
#endif  // CONFIG_ENTROPY_STATS
          if (allow_update_cdf) {
            update_cdf(fc->wedge_idx_cdf[bsize],
                       mbmi->interinter_comp.wedge_index, 16);
          }
        }
      }
    }
  }

  if (allow_update_cdf && inter_block && cm->interp_filter == SWITCHABLE &&
      mbmi->motion_mode != WARPED_CAUSAL &&
      !is_nontrans_global_motion(xd, xd->mi[0])) {
    update_filter_type_cdf(xd, mbmi, cm->seq_params.enable_dual_filter);
  }

  if (inter_block &&
      !segfeature_active(&cm->seg, mbmi->segment_id, SEG_LVL_SKIP)) {
    int16_t mode_ctx;
    const PREDICTION_MODE mode = mbmi->mode;

    mode_ctx =
        av1_mode_context_analyzer(mbmi_ext->mode_context, mbmi->ref_frame);
    if (has_second_ref(mbmi)) {
#if CONFIG_ENTROPY_STATS
      ++counts->inter_compound_mode[mode_ctx][INTER_COMPOUND_OFFSET(mode)];
#endif  // CONFIG_ENTROPY_STATS
      if (allow_update_cdf)
        update_cdf(fc->inter_compound_mode_cdf[mode_ctx],
                   INTER_COMPOUND_OFFSET(mode), INTER_COMPOUND_MODES);
    } else {
      update_inter_mode_stats(fc, counts, mode, mode_ctx, allow_update_cdf);
    }
    const int new_mv = mbmi->mode == NEWMV || mbmi->mode == NEW_NEWMV;

#if CONFIG_NEW_INTER_MODES
    if (have_drl_index(mbmi->mode)) {
      update_drl_index_stats(fc, counts, cm, mbmi, mbmi_ext, mode_ctx,
                             allow_update_cdf);
    }
    if (have_newmv_in_inter_mode(mbmi->mode)) {
#if CONFIG_FLEX_MVRES
      if (allow_update_cdf &&
          is_pb_mv_precision_active(cm, mbmi->mode, mbmi->max_mv_precision)) {
        const int down_ctx = av1_get_pb_mv_precision_down_context(cm, xd);
        int down = mbmi->max_mv_precision - mbmi->pb_mv_precision;
#if DISALLOW_ONE_DOWN_FLEX_MVRES == 2
        assert((down & 1) == 0);
        const int nsymbs = 2;
        down >>= 1;
#elif DISALLOW_ONE_DOWN_FLEX_MVRES == 1
        assert(down != 1);
        const int nsymbs = mbmi->max_mv_precision;
        down -= (down > 0);
#else
        const int nsymbs = mbmi->max_mv_precision + 1;
#endif  // DISALLOW_ONE_DOWN_FLEX_MVRES
        update_cdf(fc->pb_mv_precision_cdf[down_ctx][mbmi->max_mv_precision -
                                                     MV_SUBPEL_QTR_PRECISION],
                   down, nsymbs);
      }
      assert(mbmi->pb_mv_precision == av1_get_mbmi_mv_precision(cm, mbmi));
#endif  // CONFIG_FLEX_MVRES
      if (new_mv) {
        for (int ref = 0; ref < 1 + has_second_ref(mbmi); ++ref) {
          const int_mv ref_mv = av1_get_ref_mv(x, ref);
          av1_update_mv_stats(&mbmi->mv[ref].as_mv, &ref_mv.as_mv, &fc->nmvc,
                              mbmi->pb_mv_precision);
        }
      } else {
        const int ref =
            mbmi->mode == NEAR_NEWMV;  // Find which half of the compound
                                       // reference has NEWMV
        const int_mv ref_mv = av1_get_ref_mv(x, ref);
        av1_update_mv_stats(&mbmi->mv[ref].as_mv, &ref_mv.as_mv, &fc->nmvc,
                            mbmi->pb_mv_precision);
      }
    }
#else
    if (have_newmv_in_inter_mode(mbmi->mode)) {
#if CONFIG_FLEX_MVRES
      assert(mbmi->pb_mv_precision <= mbmi->max_mv_precision);
      assert(mbmi->max_mv_precision == xd->sbi->sb_mv_precision);
      if (allow_update_cdf &&
          is_pb_mv_precision_active(cm, mbmi->mode, mbmi->max_mv_precision)) {
        const int down_ctx = av1_get_pb_mv_precision_down_context(cm, xd);
        int down = mbmi->max_mv_precision - mbmi->pb_mv_precision;
#if DISALLOW_ONE_DOWN_FLEX_MVRES == 2
        assert((down & 1) == 0);
        const int nsymbs = 2;
        down >>= 1;
#elif DISALLOW_ONE_DOWN_FLEX_MVRES == 1
        assert(down != 1);
        const int nsymbs = mbmi->max_mv_precision;
        down -= (down > 0);
#else
        const int nsymbs = mbmi->max_mv_precision + 1;
#endif  // DISALLOW_ONE_DOWN_FLEX_MVRES
        update_cdf(fc->pb_mv_precision_cdf[down_ctx][mbmi->max_mv_precision -
                                                     MV_SUBPEL_QTR_PRECISION],
                   down, nsymbs);
      }
      assert(mbmi->pb_mv_precision == av1_get_mbmi_mv_precision(cm, mbmi));
#endif  // CONFIG_FLEX_MVRES

      if (new_mv) {
        for (int ref = 0; ref < 1 + has_second_ref(mbmi); ++ref) {
          const int_mv ref_mv = av1_get_ref_mv(x, ref);
          av1_update_mv_stats(&mbmi->mv[ref].as_mv, &ref_mv.as_mv, &fc->nmvc,
                              mbmi->pb_mv_precision);
        }
      } else {
        const int ref =
            (mbmi->mode == NEAREST_NEWMV || mbmi->mode == NEAR_NEWMV);
        const int_mv ref_mv = av1_get_ref_mv(x, ref);
        av1_update_mv_stats(&mbmi->mv[ref].as_mv, &ref_mv.as_mv, &fc->nmvc,
                            mbmi->pb_mv_precision);
      }
    }
#endif  // CONFIG_NEW_INTER_MODES
  }
}

static void update_stats(const AV1_COMMON *const cm, TileDataEnc *tile_data,
                         ThreadData *td, int mi_row, int mi_col) {
  MACROBLOCK *x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  const MB_MODE_INFO *const mbmi = xd->mi[0];
  const CurrentFrame *const current_frame = &cm->current_frame;
  const BLOCK_SIZE bsize = mbmi->sb_type;
  FRAME_CONTEXT *fc = xd->tile_ctx;
  const uint8_t allow_update_cdf = tile_data->allow_update_cdf;

  // delta quant applies to both intra and inter
  const int super_block_upper_left =
      ((mi_row & (cm->seq_params.mib_size - 1)) == 0) &&
      ((mi_col & (cm->seq_params.mib_size - 1)) == 0);

  const int seg_ref_active =
      segfeature_active(&cm->seg, mbmi->segment_id, SEG_LVL_REF_FRAME);

  if (current_frame->skip_mode_info.skip_mode_flag && !seg_ref_active &&
      is_comp_ref_allowed(bsize)) {
    const int skip_mode_ctx = av1_get_skip_mode_context(xd);
#if CONFIG_ENTROPY_STATS
    td->counts->skip_mode[skip_mode_ctx][mbmi->skip_mode]++;
#endif  // CONFIG_ENTROPY_STATS
    if (allow_update_cdf) {
      update_cdf(fc->skip_mode_cdfs[skip_mode_ctx], mbmi->skip_mode, 2);
#if CONFIG_DERIVED_MV
      if (mbmi->skip_mode && mbmi->derived_mv_allowed) {
        update_cdf(fc->use_derived_mv_cdf[2][bsize], mbmi->use_derived_mv, 2);
      }
#endif  // CONFIG_DERIVED_MV
    }
  }

  if (!mbmi->skip_mode) {
    if (!seg_ref_active) {
      const int skip_ctx = av1_get_skip_context(xd);
#if CONFIG_ENTROPY_STATS
      td->counts->skip[skip_ctx][mbmi->skip]++;
#if CONFIG_DSPL_RESIDUAL
      if (!mbmi->skip && is_inter_block(mbmi) &&
          block_size_wide[bsize] >= DSPL_MIN_PARTITION_SIDE &&
          block_size_high[bsize] >= DSPL_MIN_PARTITION_SIDE)
        td->counts->dspl_type[mbmi->dspl_type]++;
#endif  // CONFIG_DSPL_RESIDUAL
#endif  // CONFIG_ENTROPY_STATS
      if (allow_update_cdf) update_cdf(fc->skip_cdfs[skip_ctx], mbmi->skip, 2);
#if CONFIG_DSPL_RESIDUAL
      if (allow_update_cdf && !mbmi->skip && is_inter_block(mbmi) &&
          block_size_wide[bsize] >= DSPL_MIN_PARTITION_SIDE &&
          block_size_high[bsize] >= DSPL_MIN_PARTITION_SIDE)
        update_cdf(fc->dspl_type_cdf, mbmi->dspl_type, DSPL_END);
#endif  // CONFIG_DSPL_RESIDUAL
    }
  }

  const DeltaQInfo *const delta_q_info = &cm->delta_q_info;
  if (delta_q_info->delta_q_present_flag &&
      (bsize != cm->seq_params.sb_size || !mbmi->skip) &&
      super_block_upper_left) {
#if CONFIG_ENTROPY_STATS
    const int dq =
        (mbmi->current_qindex - xd->current_qindex) / delta_q_info->delta_q_res;
    const int absdq = abs(dq);
    for (int i = 0; i < AOMMIN(absdq, DELTA_Q_SMALL); ++i) {
      td->counts->delta_q[i][1]++;
    }
    if (absdq < DELTA_Q_SMALL) td->counts->delta_q[absdq][0]++;
#endif  // CONFIG_ENTROPY_STATS
    xd->current_qindex = mbmi->current_qindex;
    if (delta_q_info->delta_lf_present_flag) {
      if (delta_q_info->delta_lf_multi) {
        const int frame_lf_count =
            av1_num_planes(cm) > 1 ? FRAME_LF_COUNT : FRAME_LF_COUNT - 2;
        for (int lf_id = 0; lf_id < frame_lf_count; ++lf_id) {
#if CONFIG_ENTROPY_STATS
          const int delta_lf = (mbmi->delta_lf[lf_id] - xd->delta_lf[lf_id]) /
                               delta_q_info->delta_lf_res;
          const int abs_delta_lf = abs(delta_lf);
          for (int i = 0; i < AOMMIN(abs_delta_lf, DELTA_LF_SMALL); ++i) {
            td->counts->delta_lf_multi[lf_id][i][1]++;
          }
          if (abs_delta_lf < DELTA_LF_SMALL)
            td->counts->delta_lf_multi[lf_id][abs_delta_lf][0]++;
#endif  // CONFIG_ENTROPY_STATS
          xd->delta_lf[lf_id] = mbmi->delta_lf[lf_id];
        }
      } else {
#if CONFIG_ENTROPY_STATS
        const int delta_lf =
            (mbmi->delta_lf_from_base - xd->delta_lf_from_base) /
            delta_q_info->delta_lf_res;
        const int abs_delta_lf = abs(delta_lf);
        for (int i = 0; i < AOMMIN(abs_delta_lf, DELTA_LF_SMALL); ++i) {
          td->counts->delta_lf[i][1]++;
        }
        if (abs_delta_lf < DELTA_LF_SMALL)
          td->counts->delta_lf[abs_delta_lf][0]++;
#endif  // CONFIG_ENTROPY_STATS
        xd->delta_lf_from_base = mbmi->delta_lf_from_base;
      }
    }
  }

  if (!is_inter_block(mbmi)) {
    sum_intra_stats(cm, td->counts, xd, mbmi, xd->above_mbmi, xd->left_mbmi,
#if CONFIG_INTRA_ENTROPY
                    xd->aboveleft_mbmi,
#endif  // CONFIG_INTRA_ENTROPY
                    frame_is_intra_only(cm), allow_update_cdf);
  }

  if (av1_allow_intrabc(cm)) {
    if (allow_update_cdf) {
      update_cdf(fc->intrabc_cdf, is_intrabc_block(mbmi), 2);
#if CONFIG_EXT_IBC_MODES
      if (is_intrabc_block(mbmi)) {
        update_cdf(fc->intrabc_mode_cdf, mbmi->ibc_mode, 8);
        if (cm->ext_ibc_config == CONFIG_EXT_IBC_ALLMODES)
          update_cdf(fc->intrabc_mode_cdf, mbmi->ibc_mode, 8);
        else if (cm->ext_ibc_config == CONFIG_EXT_IBC_TOP5MODES)
          update_cdf(fc->intrabc_mode_cdf, mbmi->ibc_mode, 6);
        else if (cm->ext_ibc_config == CONFIG_EXT_IBC_TOP3MODES)
          update_cdf(fc->intrabc_mode_cdf, mbmi->ibc_mode, 4);
      }
#endif  // CONFIG_EXT_IBC_MODES
    }

#if CONFIG_ENTROPY_STATS
    ++td->counts->intrabc[is_intrabc_block(mbmi)];
#endif  // CONFIG_ENTROPY_STATS
  }

  if (!frame_is_intra_only(cm)) {
    update_inter_stats(cm, tile_data, td);
  }
}

void av1_encode_b(const AV1_COMP *const cpi, TileDataEnc *tile_data,
                  ThreadData *td, TOKENEXTRA **tp, int mi_row, int mi_col,
                  RUN_TYPE dry_run, BLOCK_SIZE bsize, PARTITION_TYPE partition,
                  const PICK_MODE_CONTEXT *const ctx, int *rate) {
  TileInfo *const tile = &tile_data->tile_info;
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *xd = &x->e_mbd;

  assert(bsize == ctx->mic.sb_type);
  av1_set_offsets_without_segment_id(cpi, tile, x, mi_row, mi_col, bsize,
                                     &ctx->chroma_ref_info);
  const int origin_mult = x->rdmult;
  av1_setup_block_rdmult(cpi, x, mi_row, mi_col, bsize, NO_AQ, NULL);
  MB_MODE_INFO *mbmi = xd->mi[0];
  mbmi->partition = partition;
  av1_update_state(cpi, td, ctx, mi_row, mi_col, bsize, dry_run);

  if (!dry_run) {
    x->mbmi_ext->cb_offset = x->cb_offset;
    assert(x->cb_offset <
           (1 << num_pels_log2_lookup[cpi->common.seq_params.sb_size]));
    av1_init_txk_skip_array(&cpi->common, mbmi, mi_row, mi_col, bsize, 0,
                            cpi->common.fEncTxSkipLog);
  }

  av1_encode_superblock(cpi, tile_data, td, tp, dry_run, bsize, rate);

  if (!dry_run) {
    x->cb_offset += block_size_wide[bsize] * block_size_high[bsize];
    if (bsize == cpi->common.seq_params.sb_size && mbmi->skip == 1 &&
        cpi->common.delta_q_info.delta_lf_present_flag) {
      const int frame_lf_count = av1_num_planes(&cpi->common) > 1
                                     ? FRAME_LF_COUNT
                                     : FRAME_LF_COUNT - 2;
      for (int lf_id = 0; lf_id < frame_lf_count; ++lf_id)
        mbmi->delta_lf[lf_id] = xd->delta_lf[lf_id];
      mbmi->delta_lf_from_base = xd->delta_lf_from_base;
    }
    if (has_second_ref(mbmi)) {
      if (mbmi->compound_idx == 0 ||
          mbmi->interinter_comp.type == COMPOUND_AVERAGE)
        mbmi->comp_group_idx = 0;
      else
        mbmi->comp_group_idx = 1;
    }
    update_stats(&cpi->common, tile_data, td, mi_row, mi_col);
  }
  x->rdmult = origin_mult;
}

#if !CONFIG_REALTIME_ONLY
void av1_encode_sb(const AV1_COMP *const cpi, ThreadData *td,
                   TileDataEnc *tile_data, TOKENEXTRA **tp, int mi_row,
                   int mi_col, RUN_TYPE dry_run, BLOCK_SIZE bsize,
                   PC_TREE *pc_tree, PARTITION_TREE *ptree, int *rate) {
  assert(bsize < BLOCK_SIZES_ALL);

  const AV1_COMMON *const cm = &cpi->common;
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  assert(bsize < BLOCK_SIZES_ALL);
  const int hbs_w = mi_size_wide[bsize] / 2;
  const int hbs_h = mi_size_high[bsize] / 2;
  const int qbs_w = mi_size_wide[bsize] / 4;
  const int qbs_h = mi_size_high[bsize] / 4;
  const int is_partition_root = is_partition_point(bsize);
  const int ctx = is_partition_root
                      ? partition_plane_context(xd, mi_row, mi_col, bsize)
                      : -1;
  const PARTITION_TYPE partition = pc_tree->partitioning;
  const BLOCK_SIZE subsize = get_partition_subsize(bsize, partition);
#if !CONFIG_EXT_RECUR_PARTITIONS
  const BLOCK_SIZE bsize2 = get_partition_subsize(bsize, PARTITION_SPLIT);
#endif  // !CONFIG_EXT_RECUR_PARTITIONS

  if (mi_row >= cm->mi_rows || mi_col >= cm->mi_cols) return;
  if (subsize == BLOCK_INVALID) return;
#if CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT
  assert(partition != PARTITION_SPLIT);
#endif  // CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT

  if (!dry_run && ctx >= 0) {
    const int has_rows = (mi_row + hbs_h) < cm->mi_rows;
    const int has_cols = (mi_col + hbs_w) < cm->mi_cols;

#if CONFIG_EXT_RECUR_PARTITIONS
    if (is_square_block(bsize)) {
#endif  // CONFIG_EXT_RECUR_PARTITIONS
      if (has_rows && has_cols) {
#if CONFIG_ENTROPY_STATS
        td->counts->partition[ctx][partition]++;
#endif

        if (tile_data->allow_update_cdf) {
          FRAME_CONTEXT *fc = xd->tile_ctx;
          update_cdf(fc->partition_cdf[ctx], partition,
                     partition_cdf_length(bsize));
        }
      }
#if CONFIG_EXT_RECUR_PARTITIONS
    } else {
      const PARTITION_TYPE_REC p_rec =
          get_symbol_from_partition_rec_block(bsize, partition);
#if CONFIG_ENTROPY_STATS
      td->counts->partition_rec[ctx][p_rec]++;
#endif

      if (tile_data->allow_update_cdf) {
        FRAME_CONTEXT *fc = xd->tile_ctx;
        update_cdf(fc->partition_rec_cdf[ctx], p_rec,
                   partition_rec_cdf_length(bsize));
      }
    }
#endif  // CONFIG_EXT_RECUR_PARTITIONS
  }

  PARTITION_TREE *sub_tree[4] = { NULL, NULL, NULL, NULL };
  if (!dry_run) {
    assert(ptree);

    ptree->partition = partition;
    ptree->bsize = bsize;
    ptree->mi_row = mi_row;
    ptree->mi_col = mi_col;
    PARTITION_TREE *parent = ptree->parent;
    const int ss_x = xd->plane[1].subsampling_x;
    const int ss_y = xd->plane[1].subsampling_y;
    set_chroma_ref_info(
        mi_row, mi_col, ptree->index, bsize, &ptree->chroma_ref_info,
        parent ? &parent->chroma_ref_info : NULL,
        parent ? parent->bsize : BLOCK_INVALID,
        parent ? parent->partition : PARTITION_NONE, ss_x, ss_y);

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
      case PARTITION_HORZ_3:
      case PARTITION_VERT_3:
        ptree->sub_tree[0] = av1_alloc_ptree_node(ptree, 0);
        ptree->sub_tree[1] = av1_alloc_ptree_node(ptree, 1);
        ptree->sub_tree[2] = av1_alloc_ptree_node(ptree, 2);
        break;
#endif  // CONFIG_EXT_RECUR_PARTITIONS
      default: break;
    }
    for (int i = 0; i < 4; ++i) sub_tree[i] = ptree->sub_tree[i];
  }

  switch (partition) {
    case PARTITION_NONE:
      av1_encode_b(cpi, tile_data, td, tp, mi_row, mi_col, dry_run, subsize,
                   partition, pc_tree->none, rate);
      break;
    case PARTITION_VERT:
#if CONFIG_EXT_RECUR_PARTITIONS
      av1_encode_sb(cpi, td, tile_data, tp, mi_row, mi_col, dry_run, subsize,
                    pc_tree->vertical[0], sub_tree[0], rate);
      if (mi_col + hbs_w < cm->mi_cols) {
        av1_encode_sb(cpi, td, tile_data, tp, mi_row, mi_col + hbs_w, dry_run,
                      subsize, pc_tree->vertical[1], sub_tree[1], rate);
      }
#else
      av1_encode_b(cpi, tile_data, td, tp, mi_row, mi_col, dry_run, subsize,
                   partition, pc_tree->vertical[0], rate);
      if (mi_col + hbs_w < cm->mi_cols) {
        av1_encode_b(cpi, tile_data, td, tp, mi_row, mi_col + hbs_w, dry_run,
                     subsize, partition, pc_tree->vertical[1], rate);
      }
#endif  // CONFIG_EXT_RECUR_PARTITIONS
      break;
    case PARTITION_HORZ:
#if CONFIG_EXT_RECUR_PARTITIONS
      av1_encode_sb(cpi, td, tile_data, tp, mi_row, mi_col, dry_run, subsize,
                    pc_tree->horizontal[0], sub_tree[0], rate);
      if (mi_row + hbs_h < cm->mi_rows) {
        av1_encode_sb(cpi, td, tile_data, tp, mi_row + hbs_h, mi_col, dry_run,
                      subsize, pc_tree->horizontal[1], sub_tree[1], rate);
      }
#else
      av1_encode_b(cpi, tile_data, td, tp, mi_row, mi_col, dry_run, subsize,
                   partition, pc_tree->horizontal[0], rate);
      if (mi_row + hbs_h < cm->mi_rows) {
        av1_encode_b(cpi, tile_data, td, tp, mi_row + hbs_h, mi_col, dry_run,
                     subsize, partition, pc_tree->horizontal[1], rate);
      }
#endif  // CONFIG_EXT_RECUR_PARTITIONS

      break;
    case PARTITION_SPLIT:
      av1_encode_sb(cpi, td, tile_data, tp, mi_row, mi_col, dry_run, subsize,
                    pc_tree->split[0], sub_tree[0], rate);
      av1_encode_sb(cpi, td, tile_data, tp, mi_row, mi_col + hbs_w, dry_run,
                    subsize, pc_tree->split[1], sub_tree[1], rate);
      av1_encode_sb(cpi, td, tile_data, tp, mi_row + hbs_h, mi_col, dry_run,
                    subsize, pc_tree->split[2], sub_tree[2], rate);
      av1_encode_sb(cpi, td, tile_data, tp, mi_row + hbs_h, mi_col + hbs_w,
                    dry_run, subsize, pc_tree->split[3], sub_tree[3], rate);
      break;
#if CONFIG_EXT_RECUR_PARTITIONS
    case PARTITION_HORZ_3: {
      const BLOCK_SIZE bsize3 = get_partition_subsize(bsize, PARTITION_HORZ);
      av1_encode_sb(cpi, td, tile_data, tp, mi_row, mi_col, dry_run, subsize,
                    pc_tree->horizontal3[0], sub_tree[0], rate);
      if (mi_row + qbs_h >= cm->mi_rows) break;
      av1_encode_sb(cpi, td, tile_data, tp, mi_row + qbs_h, mi_col, dry_run,
                    bsize3, pc_tree->horizontal3[1], sub_tree[1], rate);
      if (mi_row + 3 * qbs_h >= cm->mi_rows) break;
      av1_encode_sb(cpi, td, tile_data, tp, mi_row + 3 * qbs_h, mi_col, dry_run,
                    subsize, pc_tree->horizontal3[2], sub_tree[2], rate);
      break;
    }
    case PARTITION_VERT_3: {
      const BLOCK_SIZE bsize3 = get_partition_subsize(bsize, PARTITION_VERT);
      av1_encode_sb(cpi, td, tile_data, tp, mi_row, mi_col, dry_run, subsize,
                    pc_tree->vertical3[0], sub_tree[0], rate);
      if (mi_col + qbs_w >= cm->mi_cols) break;
      av1_encode_sb(cpi, td, tile_data, tp, mi_row, mi_col + qbs_w, dry_run,
                    bsize3, pc_tree->vertical3[1], sub_tree[1], rate);
      if (mi_col + 3 * qbs_w >= cm->mi_cols) break;
      av1_encode_sb(cpi, td, tile_data, tp, mi_row, mi_col + 3 * qbs_w, dry_run,
                    subsize, pc_tree->vertical3[2], sub_tree[2], rate);
      break;
    }
#else
    case PARTITION_HORZ_A:
      av1_encode_b(cpi, tile_data, td, tp, mi_row, mi_col, dry_run, bsize2,
                   partition, pc_tree->horizontala[0], rate);
      av1_encode_b(cpi, tile_data, td, tp, mi_row, mi_col + hbs_w, dry_run,
                   bsize2, partition, pc_tree->horizontala[1], rate);
      av1_encode_b(cpi, tile_data, td, tp, mi_row + hbs_h, mi_col, dry_run,
                   subsize, partition, pc_tree->horizontala[2], rate);
      break;
    case PARTITION_HORZ_B:
      av1_encode_b(cpi, tile_data, td, tp, mi_row, mi_col, dry_run, subsize,
                   partition, pc_tree->horizontalb[0], rate);
      av1_encode_b(cpi, tile_data, td, tp, mi_row + hbs_h, mi_col, dry_run,
                   bsize2, partition, pc_tree->horizontalb[1], rate);
      av1_encode_b(cpi, tile_data, td, tp, mi_row + hbs_h, mi_col + hbs_w,
                   dry_run, bsize2, partition, pc_tree->horizontalb[2], rate);
      break;
    case PARTITION_VERT_A:
      av1_encode_b(cpi, tile_data, td, tp, mi_row, mi_col, dry_run, bsize2,
                   partition, pc_tree->verticala[0], rate);
      av1_encode_b(cpi, tile_data, td, tp, mi_row + hbs_h, mi_col, dry_run,
                   bsize2, partition, pc_tree->verticala[1], rate);
      av1_encode_b(cpi, tile_data, td, tp, mi_row, mi_col + hbs_w, dry_run,
                   subsize, partition, pc_tree->verticala[2], rate);
      break;
    case PARTITION_VERT_B:
      av1_encode_b(cpi, tile_data, td, tp, mi_row, mi_col, dry_run, subsize,
                   partition, pc_tree->verticalb[0], rate);
      av1_encode_b(cpi, tile_data, td, tp, mi_row, mi_col + hbs_w, dry_run,
                   bsize2, partition, pc_tree->verticalb[1], rate);
      av1_encode_b(cpi, tile_data, td, tp, mi_row + hbs_h, mi_col + hbs_w,
                   dry_run, bsize2, partition, pc_tree->verticalb[2], rate);
      break;
    case PARTITION_HORZ_4:
      for (int i = 0; i < 4; ++i) {
        int this_mi_row = mi_row + i * qbs_h;
        if (i > 0 && this_mi_row >= cm->mi_rows) break;

        av1_encode_b(cpi, tile_data, td, tp, this_mi_row, mi_col, dry_run,
                     subsize, partition, pc_tree->horizontal4[i], rate);
      }
      break;
    case PARTITION_VERT_4:
      for (int i = 0; i < 4; ++i) {
        int this_mi_col = mi_col + i * qbs_w;
        if (i > 0 && this_mi_col >= cm->mi_cols) break;

        av1_encode_b(cpi, tile_data, td, tp, mi_row, this_mi_col, dry_run,
                     subsize, partition, pc_tree->vertical4[i], rate);
      }
      break;
#endif  // CONFIG_EXT_RECUR_PARTITIONS
    default: assert(0 && "Invalid partition type."); break;
  }

  if (ptree) ptree->is_settled = 1;
  update_ext_partition_context(xd, mi_row, mi_col, subsize, bsize, partition);
}
#endif  // !CONFIG_REALTIME_ONLY
