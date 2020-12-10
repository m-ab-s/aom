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

#include "av1/common/pred_common.h"
#include "av1/common/reconintra.h"

#include "av1/encoder/aq_variance.h"
#include "av1/encoder/encoder.h"
#include "av1/encoder/encodeframe_utils.h"
#include "av1/encoder/partition_strategy.h"
#include "av1/encoder/rdopt.h"
#include "av1/encoder/tpl_model.h"

#if !CONFIG_REALTIME_ONLY
void av1_backup_sb_state(SB_FIRST_PASS_STATS *sb_fp_stats, const AV1_COMP *cpi,
                         ThreadData *td, const TileDataEnc *tile_data,
                         int mi_row, int mi_col) {
  MACROBLOCK *x = &td->mb;
  MACROBLOCKD *xd = &x->e_mbd;
  const TileInfo *tile_info = &tile_data->tile_info;

  const AV1_COMMON *cm = &cpi->common;
  const int num_planes = av1_num_planes(cm);
  const BLOCK_SIZE sb_size = cm->seq_params.sb_size;

  xd->above_txfm_context = cm->above_txfm_context[tile_info->tile_row] + mi_col;
  xd->left_txfm_context =
      xd->left_txfm_context_buffer + (mi_row & MAX_MIB_MASK);
  av1_save_context(x, &sb_fp_stats->x_ctx, mi_row, mi_col, sb_size, num_planes);

  sb_fp_stats->rd_count = cpi->td.rd_counts;
  sb_fp_stats->split_count = cpi->td.mb.txb_split_count;

  sb_fp_stats->fc = *td->counts;

  memcpy(sb_fp_stats->inter_mode_rd_models, tile_data->inter_mode_rd_models,
         sizeof(sb_fp_stats->inter_mode_rd_models));

  memcpy(sb_fp_stats->thresh_freq_fact, x->thresh_freq_fact,
         sizeof(sb_fp_stats->thresh_freq_fact));

  const int mi_idx = get_mi_grid_idx(cm, mi_row, mi_col);
  sb_fp_stats->current_qindex = cm->mi[mi_idx].current_qindex;

#if CONFIG_INTERNAL_STATS
  memcpy(sb_fp_stats->mode_chosen_counts, cpi->mode_chosen_counts,
         sizeof(sb_fp_stats->mode_chosen_counts));
#endif  // CONFIG_INTERNAL_STATS
}

void av1_restore_sb_state(const SB_FIRST_PASS_STATS *sb_fp_stats, AV1_COMP *cpi,
                          ThreadData *td, TileDataEnc *tile_data, int mi_row,
                          int mi_col) {
  MACROBLOCK *x = &td->mb;

  const AV1_COMMON *cm = &cpi->common;
  const int num_planes = av1_num_planes(cm);
  const BLOCK_SIZE sb_size = cm->seq_params.sb_size;

  av1_restore_context(cm, x, &sb_fp_stats->x_ctx, mi_row, mi_col, sb_size,
                      num_planes);

  cpi->td.rd_counts = sb_fp_stats->rd_count;
  cpi->td.mb.txb_split_count = sb_fp_stats->split_count;

  *td->counts = sb_fp_stats->fc;

  memcpy(tile_data->inter_mode_rd_models, sb_fp_stats->inter_mode_rd_models,
         sizeof(sb_fp_stats->inter_mode_rd_models));
  memcpy(x->thresh_freq_fact, sb_fp_stats->thresh_freq_fact,
         sizeof(sb_fp_stats->thresh_freq_fact));

  const int mi_idx = get_mi_grid_idx(cm, mi_row, mi_col);
  cm->mi[mi_idx].current_qindex = sb_fp_stats->current_qindex;

#if CONFIG_INTERNAL_STATS
  memcpy(cpi->mode_chosen_counts, sb_fp_stats->mode_chosen_counts,
         sizeof(sb_fp_stats->mode_chosen_counts));
#endif  // CONFIG_INTERNAL_STATS
}

// Memset the mbmis at the current superblock to 0
void av1_reset_mbmi(AV1_COMMON *cm, int mi_row, int mi_col) {
  const BLOCK_SIZE sb_size = cm->seq_params.sb_size;
  // size of sb in unit of mi_grid (BLOCK_4X4)
  const int sb_size_mi = mi_size_wide[sb_size];
  assert(cm->mi_stride % sb_size_mi == 0 &&
         "mi is not allocated as a multiple of sb!");

  const int mi_rows = mi_size_high[sb_size];
  for (int cur_mi_row = 0; cur_mi_row < mi_rows; cur_mi_row++) {
    const int mi_cols =
        AOMMIN(sb_size_mi, cm->mi_stride - (mi_col + sb_size_mi));
    assert(get_mi_grid_idx(cm, 0, mi_col + mi_cols) < cm->mi_stride);
    const int mi_grid_idx = get_mi_grid_idx(cm, mi_row + cur_mi_row, mi_col);
    memset(&cm->mi_grid_base[mi_grid_idx], 0,
           mi_cols * sizeof(*cm->mi_grid_base));
    memset(&cm->mi[mi_grid_idx], 0, mi_cols * sizeof(*cm->mi));
  }
}
#endif  // !CONFIG_REALTIME_ONLY

// analysis_type 0: Use mc_dep_cost and intra_cost
// analysis_type 1: Use count of best inter predictor chosen
// analysis_type 2: Use cost reduction from intra to inter for best inter
//                  predictor chosen
static int get_q_for_deltaq_objective(AV1_COMP *const cpi, BLOCK_SIZE bsize,
                                      int analysis_type, int mi_row,
                                      int mi_col) {
  AV1_COMMON *const cm = &cpi->common;
  assert(IMPLIES(cpi->gf_group.size > 0,
                 cpi->gf_group.index < cpi->gf_group.size));
  const int tpl_idx = cpi->gf_group.index;
  TplDepFrame *tpl_frame = &cpi->tpl_frame[tpl_idx];
  TplDepStats *tpl_stats = tpl_frame->tpl_stats_ptr;
  int tpl_stride = tpl_frame->stride;
  int64_t intra_cost = 0;
  int64_t mc_dep_cost = 0;
  const int mi_wide = mi_size_wide[bsize];
  const int mi_high = mi_size_high[bsize];

  if (cpi->tpl_model_pass == 1) {
    assert(cpi->oxcf.enable_tpl_model == 2);
    return cm->base_qindex;
  }

  if (tpl_frame->is_valid == 0) return cm->base_qindex;

  if (!is_frame_tpl_eligible(cpi)) return cm->base_qindex;

  if (cpi->gf_group.index >= MAX_LAG_BUFFERS) return cm->base_qindex;

  int64_t mc_count = 0, mc_saved = 0;
  int mi_count = 0;
  const int mi_col_sr =
      av1_coded_to_superres_mi(mi_col, cm->superres_scale_denominator);
  const int mi_col_end_sr = av1_coded_to_superres_mi(
      mi_col + mi_wide, cm->superres_scale_denominator);
  const int mi_cols_sr = av1_pixels_to_mi(cm->superres_upscaled_width);
  const int step = 1 << cpi->tpl_stats_block_mis_log2;
  for (int row = mi_row; row < mi_row + mi_high; row += step) {
    for (int col = mi_col_sr; col < mi_col_end_sr; col += step) {
      if (row >= cm->mi_rows || col >= mi_cols_sr) continue;
      TplDepStats *this_stats =
          &tpl_stats[av1_tpl_ptr_pos(cpi, row, col, tpl_stride)];
      int64_t mc_dep_delta =
          RDCOST(tpl_frame->base_rdmult, this_stats->mc_dep_rate,
                 this_stats->mc_dep_dist);
      intra_cost += this_stats->recrf_dist << RDDIV_BITS;
      mc_dep_cost += (this_stats->recrf_dist << RDDIV_BITS) + mc_dep_delta;
      mc_count += this_stats->mc_count;
      mc_saved += this_stats->mc_saved;
      mi_count++;
    }
  }

  aom_clear_system_state();

  int offset = 0;
  double beta = 1.0;
  if (analysis_type == 0) {
    if (mc_dep_cost > 0 && intra_cost > 0) {
      const double r0 = cpi->rd.r0;
      const double rk = (double)intra_cost / mc_dep_cost;
      beta = (r0 / rk);
      assert(beta > 0.0);
    }
  } else if (analysis_type == 1) {
    const double mc_count_base = (mi_count * cpi->rd.mc_count_base);
    beta = (mc_count + 1.0) / (mc_count_base + 1.0);
    beta = pow(beta, 0.5);
  } else if (analysis_type == 2) {
    const double mc_saved_base = (mi_count * cpi->rd.mc_saved_base);
    beta = (mc_saved + 1.0) / (mc_saved_base + 1.0);
    beta = pow(beta, 0.5);
  }
  offset = (7 * av1_get_deltaq_offset(cpi, cm->base_qindex, beta)) / 8;
  // printf("[%d/%d]: beta %g offset %d\n", pyr_lev_from_top,
  //        cpi->gf_group.pyramid_height, beta, offset);

  aom_clear_system_state();

  const DeltaQInfo *const delta_q_info = &cm->delta_q_info;
  offset = AOMMIN(offset, delta_q_info->delta_q_res * 9 - 1);
  offset = AOMMAX(offset, -delta_q_info->delta_q_res * 9 + 1);
  int qindex = cm->base_qindex + offset;
#if CONFIG_EXTQUANT
  qindex = AOMMIN(qindex, cm->seq_params.bit_depth == AOM_BITS_8
                              ? MAXQ_8_BITS
                              : cm->seq_params.bit_depth == AOM_BITS_10
                                    ? MAXQ_10_BITS
                                    : MAXQ);
#else
  qindex = AOMMIN(qindex, MAXQ);
#endif
  qindex = AOMMAX(qindex, MINQ);

  return qindex;
}

void av1_setup_delta_q(AV1_COMP *const cpi, ThreadData *td, MACROBLOCK *const x,
                       const TileInfo *const tile_info, int mi_row, int mi_col,
                       int num_planes) {
  AV1_COMMON *const cm = &cpi->common;
  MACROBLOCKD *const xd = &x->e_mbd;
  const DeltaQInfo *const delta_q_info = &cm->delta_q_info;
  const BLOCK_SIZE sb_size = cm->seq_params.sb_size;
  const int mib_size = cm->seq_params.mib_size;

  // Delta-q modulation based on variance
  av1_setup_src_planes(x, cpi->source, mi_row, mi_col, num_planes, NULL);

  int current_qindex = cm->base_qindex;
  if (cm->delta_q_info.delta_q_present_flag) {
    if (cpi->oxcf.deltaq_mode == DELTA_Q_PERCEPTUAL) {
      if (DELTA_Q_PERCEPTUAL_MODULATION == 1) {
        const int block_wavelet_energy_level =
            av1_block_wavelet_energy_level(cpi, x, sb_size);
        x->sb_energy_level = block_wavelet_energy_level;
        current_qindex = av1_compute_q_from_energy_level_deltaq_mode(
            cpi, block_wavelet_energy_level);
      } else {
        const int block_var_level = av1_log_block_var(cpi, x, sb_size);
        x->sb_energy_level = block_var_level;
        current_qindex =
            av1_compute_q_from_energy_level_deltaq_mode(cpi, block_var_level);
      }
    } else if (cpi->oxcf.deltaq_mode == DELTA_Q_OBJECTIVE) {
      assert(cpi->oxcf.enable_tpl_model);
      // Setup deltaq based on tpl stats
      current_qindex =
          get_q_for_deltaq_objective(cpi, sb_size, 0, mi_row, mi_col);
    }
  }

  const int qmask = ~(delta_q_info->delta_q_res - 1);
  current_qindex = clamp(current_qindex, delta_q_info->delta_q_res,
                         256 - delta_q_info->delta_q_res);

  const int sign_deltaq_index =
      current_qindex - xd->current_qindex >= 0 ? 1 : -1;

  const int deltaq_deadzone = delta_q_info->delta_q_res / 4;
  int abs_deltaq_index = abs(current_qindex - xd->current_qindex);
  abs_deltaq_index = (abs_deltaq_index + deltaq_deadzone) & qmask;
  current_qindex = xd->current_qindex + sign_deltaq_index * abs_deltaq_index;
  current_qindex = AOMMAX(current_qindex, MINQ + 1);
  assert(current_qindex > 0);

  xd->delta_qindex = current_qindex - cm->base_qindex;

  CHROMA_REF_INFO sb_chr_ref_info = { 1, 0, mi_row, mi_col, sb_size, sb_size };
  av1_enc_set_offsets(cpi, tile_info, x, mi_row, mi_col, sb_size,
                      &sb_chr_ref_info);
  xd->mi[0]->current_qindex = current_qindex;
  av1_init_plane_quantizers(cpi, x, xd->mi[0]->segment_id);

  // keep track of any non-zero delta-q used
  td->deltaq_used |= (xd->delta_qindex != 0);

  if (cm->delta_q_info.delta_q_present_flag && cpi->oxcf.deltalf_mode) {
    const int lfmask = ~(delta_q_info->delta_lf_res - 1);
    const int delta_lf_from_base =
        ((xd->delta_qindex / 2 + delta_q_info->delta_lf_res / 2) & lfmask);

    // pre-set the delta lf for loop filter. Note that this value is set
    // before mi is assigned for each block in current superblock
    for (int j = 0; j < AOMMIN(mib_size, cm->mi_rows - mi_row); j++) {
      for (int k = 0; k < AOMMIN(mib_size, cm->mi_cols - mi_col); k++) {
        cm->mi[(mi_row + j) * cm->mi_stride + (mi_col + k)].delta_lf_from_base =
            (int8_t)clamp(delta_lf_from_base, -MAX_LOOP_FILTER,
                          MAX_LOOP_FILTER);
        const int frame_lf_count =
            av1_num_planes(cm) > 1 ? FRAME_LF_COUNT : FRAME_LF_COUNT - 2;
        for (int lf_id = 0; lf_id < frame_lf_count; ++lf_id) {
          cm->mi[(mi_row + j) * cm->mi_stride + (mi_col + k)].delta_lf[lf_id] =
              (int8_t)clamp(delta_lf_from_base, -MAX_LOOP_FILTER,
                            MAX_LOOP_FILTER);
        }
      }
    }
  }
}

#if !CONFIG_REALTIME_ONLY
static int get_rdmult_delta(AV1_COMP *cpi, BLOCK_SIZE bsize, int analysis_type,
                            int mi_row, int mi_col, int orig_rdmult) {
  AV1_COMMON *const cm = &cpi->common;
  assert(IMPLIES(cpi->gf_group.size > 0,
                 cpi->gf_group.index < cpi->gf_group.size));
  const int tpl_idx = cpi->gf_group.index;
  TplDepFrame *tpl_frame = &cpi->tpl_frame[tpl_idx];
  TplDepStats *tpl_stats = tpl_frame->tpl_stats_ptr;
  int tpl_stride = tpl_frame->stride;
  int64_t intra_cost = 0;
  int64_t mc_dep_cost = 0;
  const int mi_wide = mi_size_wide[bsize];
  const int mi_high = mi_size_high[bsize];

  if (tpl_frame->is_valid == 0) return orig_rdmult;

  if (!is_frame_tpl_eligible(cpi)) return orig_rdmult;

  if (cpi->gf_group.index >= MAX_LAG_BUFFERS) return orig_rdmult;

  int64_t mc_count = 0, mc_saved = 0;
  int mi_count = 0;
  const int mi_col_sr =
      av1_coded_to_superres_mi(mi_col, cm->superres_scale_denominator);
  const int mi_col_end_sr = av1_coded_to_superres_mi(
      mi_col + mi_wide, cm->superres_scale_denominator);
  const int mi_cols_sr = av1_pixels_to_mi(cm->superres_upscaled_width);
  const int step = 1 << cpi->tpl_stats_block_mis_log2;
  for (int row = mi_row; row < mi_row + mi_high; row += step) {
    for (int col = mi_col_sr; col < mi_col_end_sr; col += step) {
      if (row >= cm->mi_rows || col >= mi_cols_sr) continue;
      TplDepStats *this_stats =
          &tpl_stats[av1_tpl_ptr_pos(cpi, row, col, tpl_stride)];
      int64_t mc_dep_delta =
          RDCOST(tpl_frame->base_rdmult, this_stats->mc_dep_rate,
                 this_stats->mc_dep_dist);
      intra_cost += this_stats->recrf_dist << RDDIV_BITS;
      mc_dep_cost += (this_stats->recrf_dist << RDDIV_BITS) + mc_dep_delta;
      mc_count += this_stats->mc_count;
      mc_saved += this_stats->mc_saved;
      mi_count++;
    }
  }

  aom_clear_system_state();

  double beta = 1.0;
  if (analysis_type == 0) {
    if (mc_dep_cost > 0 && intra_cost > 0) {
      const double r0 = cpi->rd.r0;
      const double rk = (double)intra_cost / mc_dep_cost;
      beta = (r0 / rk);
    }
  } else if (analysis_type == 1) {
    const double mc_count_base = (mi_count * cpi->rd.mc_count_base);
    beta = (mc_count + 1.0) / (mc_count_base + 1.0);
    beta = pow(beta, 0.5);
  } else if (analysis_type == 2) {
    const double mc_saved_base = (mi_count * cpi->rd.mc_saved_base);
    beta = (mc_saved + 1.0) / (mc_saved_base + 1.0);
    beta = pow(beta, 0.5);
  }

  int rdmult = av1_get_adaptive_rdmult(cpi, beta);

  aom_clear_system_state();

  rdmult = AOMMIN(rdmult, orig_rdmult * 3 / 2);
  rdmult = AOMMAX(rdmult, orig_rdmult * 1 / 2);

  rdmult = AOMMAX(1, rdmult);

  return rdmult;
}

static AOM_INLINE void adjust_rdmult_tpl_model(AV1_COMP *cpi, MACROBLOCK *x,
                                               int mi_row, int mi_col) {
  const BLOCK_SIZE sb_size = cpi->common.seq_params.sb_size;
  const int orig_rdmult = cpi->rd.RDMULT;

  if (cpi->tpl_model_pass == 1) {
    assert(cpi->oxcf.enable_tpl_model == 2);
    x->rdmult = orig_rdmult;
    return;
  }

  assert(IMPLIES(cpi->gf_group.size > 0,
                 cpi->gf_group.index < cpi->gf_group.size));
  const int gf_group_index = cpi->gf_group.index;
  if (cpi->oxcf.enable_tpl_model && cpi->oxcf.aq_mode == NO_AQ &&
      cpi->oxcf.deltaq_mode == NO_DELTA_Q && gf_group_index > 0 &&
      cpi->gf_group.update_type[gf_group_index] == ARF_UPDATE) {
    const int dr =
        get_rdmult_delta(cpi, sb_size, 0, mi_row, mi_col, orig_rdmult);
    x->rdmult = dr;
  }
}

static void reset_simple_motion_tree_partition(
    SIMPLE_MOTION_DATA_TREE *sms_tree, BLOCK_SIZE bsize) {
  sms_tree->partitioning = PARTITION_NONE;

  if (bsize >= BLOCK_8X8) {
    BLOCK_SIZE subsize = get_partition_subsize(bsize, PARTITION_SPLIT);
    for (int idx = 0; idx < 4; ++idx)
      reset_simple_motion_tree_partition(sms_tree->split[idx], subsize);
  }
}

// This function initializes the stats for encode_rd_sb.
void av1_init_encode_rd_sb(AV1_COMP *cpi, ThreadData *td,
                           const TileDataEnc *tile_data, PC_TREE **pc_root,
                           SIMPLE_MOTION_DATA_TREE *sms_root, RD_STATS *rd_cost,
                           int mi_row, int mi_col, int gather_tpl_data) {
  const AV1_COMMON *cm = &cpi->common;
  const BLOCK_SIZE sb_size = cm->seq_params.sb_size;
  const TileInfo *tile_info = &tile_data->tile_info;
  MACROBLOCK *x = &td->mb;

  if (!*pc_root) {
    const MACROBLOCKD *xd = &x->e_mbd;
    const int ss_x = xd->plane[1].subsampling_x;
    const int ss_y = xd->plane[1].subsampling_y;
    *pc_root = av1_alloc_pc_tree_node(mi_row, mi_col, sb_size, NULL,
                                      PARTITION_NONE, 0, 1, ss_x, ss_y);
  }

  x->sb_energy_level = 0;
  x->cnn_output_valid = 0;
  if (gather_tpl_data) {
    if (cm->delta_q_info.delta_q_present_flag) {
      const int num_planes = av1_num_planes(cm);
      av1_setup_delta_q(cpi, td, x, tile_info, mi_row, mi_col, num_planes);
      av1_tpl_rdmult_setup_sb(cpi, x, sb_size, mi_row, mi_col);
    }
    if (cpi->oxcf.enable_tpl_model) {
      adjust_rdmult_tpl_model(cpi, x, mi_row, mi_col);
    }
  }

  // Reset hash state for transform/mode rd hash information
  reset_hash_records(x, 0);
  av1_zero(x->picked_ref_frames_mask);
  av1_zero(x->pred_mv);
  av1_invalid_rd_stats(rd_cost);

  const SPEED_FEATURES *sf = &cpi->sf;
  const int use_simple_motion_search =
      (sf->simple_motion_search_split || sf->simple_motion_search_prune_rect ||
       sf->simple_motion_search_early_term_none ||
       sf->ml_early_term_after_part_split_level) &&
      !frame_is_intra_only(cm);
  if (use_simple_motion_search) {
    reset_simple_motion_tree_partition(sms_root, sb_size);
    init_simple_motion_search_mvs(sms_root);
  }

#if CONFIG_EXT_RECUR_PARTITIONS
  av1_init_sms_data_bufs(x->sms_bufs);
#endif  // CONFIG_EXT_RECUR_PARTITIONS
}

#endif  // !CONFIG_REALTIME_ONLY

void av1_set_offsets_without_segment_id(const AV1_COMP *const cpi,
                                        const TileInfo *const tile,
                                        MACROBLOCK *const x, int mi_row,
                                        int mi_col, BLOCK_SIZE bsize,
                                        const CHROMA_REF_INFO *chr_ref_info) {
  const AV1_COMMON *const cm = &cpi->common;
  const int num_planes = chr_ref_info ? av1_num_planes(cm) : 1;
  MACROBLOCKD *const xd = &x->e_mbd;
  assert(bsize < BLOCK_SIZES_ALL);
  const int mi_width = mi_size_wide[bsize];
  const int mi_height = mi_size_high[bsize];

  set_mode_info_offsets(cpi, x, xd, mi_row, mi_col);
  set_skip_context(xd, mi_row, mi_col, num_planes, chr_ref_info);
  xd->above_txfm_context = cm->above_txfm_context[tile->tile_row] + mi_col;
  xd->left_txfm_context =
      xd->left_txfm_context_buffer + (mi_row & MAX_MIB_MASK);

  // Set up destination pointers.
  av1_setup_dst_planes(xd->plane, &cm->cur_frame->buf, mi_row, mi_col, 0,
                       num_planes, chr_ref_info);

  // Set up limit values for MV components.
  // Mv beyond the range do not produce new/different prediction block.
  x->mv_limits.row_min =
      -(((mi_row + mi_height) * MI_SIZE) + AOM_INTERP_EXTEND);
  x->mv_limits.col_min = -(((mi_col + mi_width) * MI_SIZE) + AOM_INTERP_EXTEND);
  x->mv_limits.row_max = (cm->mi_rows - mi_row) * MI_SIZE + AOM_INTERP_EXTEND;
  x->mv_limits.col_max = (cm->mi_cols - mi_col) * MI_SIZE + AOM_INTERP_EXTEND;

  set_plane_n4(xd, bsize, num_planes, chr_ref_info);

  // Set up distance of MB to edge of frame in 1/8th pel units.
#if !CONFIG_EXT_RECUR_PARTITIONS
  assert(!(mi_col & (mi_width - 1)) && !(mi_row & (mi_height - 1)));
#endif  // !CONFIG_EXT_RECUR_PARTITIONS
  set_mi_row_col(xd, tile, mi_row, mi_height, mi_col, mi_width, cm->mi_rows,
                 cm->mi_cols, chr_ref_info);

  // Set up source buffers.
  av1_setup_src_planes(x, cpi->source, mi_row, mi_col, num_planes,
                       chr_ref_info);

  // required by av1_append_sub8x8_mvs_for_idx() and av1_find_best_ref_mvs()
  xd->tile = *tile;

  xd->cfl.mi_row = mi_row;
  xd->cfl.mi_col = mi_col;
}

void av1_enc_set_offsets(const AV1_COMP *const cpi, const TileInfo *const tile,
                         MACROBLOCK *const x, int mi_row, int mi_col,
                         BLOCK_SIZE bsize, CHROMA_REF_INFO *chr_ref_info) {
  const AV1_COMMON *const cm = &cpi->common;
  const struct segmentation *const seg = &cm->seg;
  MACROBLOCKD *const xd = &x->e_mbd;
  MB_MODE_INFO *mbmi;

  av1_set_offsets_without_segment_id(cpi, tile, x, mi_row, mi_col, bsize,
                                     chr_ref_info);

  // Setup segment ID.
  mbmi = xd->mi[0];
  mbmi->segment_id = 0;
  if (seg->enabled) {
    if (seg->enabled && !cpi->vaq_refresh) {
      const uint8_t *const map =
          seg->update_map ? cpi->segmentation_map : cm->last_frame_seg_map;
      mbmi->segment_id =
          map ? get_segment_id(cm, map, bsize, mi_row, mi_col) : 0;
    }
    av1_init_plane_quantizers(cpi, x, mbmi->segment_id);
  }
}

void av1_save_context(const MACROBLOCK *x, RD_SEARCH_MACROBLOCK_CONTEXT *ctx,
                      int mi_row, int mi_col, BLOCK_SIZE bsize,
                      const int num_planes) {
  const MACROBLOCKD *xd = &x->e_mbd;
  int p;
  const int num_4x4_blocks_wide =
      block_size_wide[bsize] >> tx_size_wide_log2[0];
  const int num_4x4_blocks_high =
      block_size_high[bsize] >> tx_size_high_log2[0];
  int mi_width = mi_size_wide[bsize];
  int mi_height = mi_size_high[bsize];

  // buffer the above/left context information of the block in search.
  for (p = 0; p < num_planes; ++p) {
    int tx_col = mi_col;
    int tx_row = mi_row & MAX_MIB_MASK;
    memcpy(ctx->a + num_4x4_blocks_wide * p,
           xd->above_context[p] + (tx_col >> xd->plane[p].subsampling_x),
           (sizeof(ENTROPY_CONTEXT) * num_4x4_blocks_wide) >>
               xd->plane[p].subsampling_x);
    memcpy(ctx->l + num_4x4_blocks_high * p,
           xd->left_context[p] + (tx_row >> xd->plane[p].subsampling_y),
           (sizeof(ENTROPY_CONTEXT) * num_4x4_blocks_high) >>
               xd->plane[p].subsampling_y);
  }
  memcpy(ctx->sa, xd->above_seg_context + mi_col,
         sizeof(*xd->above_seg_context) * mi_width);
  memcpy(ctx->sl, xd->left_seg_context + (mi_row & MAX_MIB_MASK),
         sizeof(xd->left_seg_context[0]) * mi_height);
  memcpy(ctx->ta, xd->above_txfm_context,
         sizeof(*xd->above_txfm_context) * mi_width);
  memcpy(ctx->tl, xd->left_txfm_context,
         sizeof(*xd->left_txfm_context) * mi_height);
  ctx->p_ta = xd->above_txfm_context;
  ctx->p_tl = xd->left_txfm_context;
}

void av1_restore_context(const AV1_COMMON *cm, MACROBLOCK *x,
                         const RD_SEARCH_MACROBLOCK_CONTEXT *ctx, int mi_row,
                         int mi_col, BLOCK_SIZE bsize, const int num_planes) {
  MACROBLOCKD *xd = &x->e_mbd;
  int p;
  const int num_4x4_blocks_wide =
      block_size_wide[bsize] >> tx_size_wide_log2[0];
  const int num_4x4_blocks_high =
      block_size_high[bsize] >> tx_size_high_log2[0];
  int mi_width = mi_size_wide[bsize];
  int mi_height = mi_size_high[bsize];
  for (p = 0; p < num_planes; p++) {
    int tx_col = mi_col;
    int tx_row = mi_row & MAX_MIB_MASK;
    memcpy(xd->above_context[p] + (tx_col >> xd->plane[p].subsampling_x),
           ctx->a + num_4x4_blocks_wide * p,
           (sizeof(ENTROPY_CONTEXT) * num_4x4_blocks_wide) >>
               xd->plane[p].subsampling_x);
    memcpy(xd->left_context[p] + (tx_row >> xd->plane[p].subsampling_y),
           ctx->l + num_4x4_blocks_high * p,
           (sizeof(ENTROPY_CONTEXT) * num_4x4_blocks_high) >>
               xd->plane[p].subsampling_y);
  }
  memcpy(xd->above_seg_context + mi_col, ctx->sa,
         sizeof(*xd->above_seg_context) * mi_width);
  memcpy(xd->left_seg_context + (mi_row & MAX_MIB_MASK), ctx->sl,
         sizeof(xd->left_seg_context[0]) * mi_height);
  xd->above_txfm_context = ctx->p_ta;
  xd->left_txfm_context = ctx->p_tl;
  memcpy(xd->above_txfm_context, ctx->ta,
         sizeof(*xd->above_txfm_context) * mi_width);
  memcpy(xd->left_txfm_context, ctx->tl,
         sizeof(*xd->left_txfm_context) * mi_height);

  av1_mark_block_as_not_coded(xd, mi_row, mi_col, bsize,
                              cm->seq_params.sb_size);
}
