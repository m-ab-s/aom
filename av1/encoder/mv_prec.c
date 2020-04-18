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
#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"
#include "config/aom_scale_rtcd.h"
#include "config/av1_rtcd.h"

#include "aom_ports/system_state.h"

#include "av1/encoder/encodemv.h"
#if !CONFIG_REALTIME_ONLY
#include "av1/encoder/misc_model_weights.h"
#endif  // !CONFIG_REALTIME_ONLY
#include "av1/encoder/mv_prec.h"

#define MV_PREC_DET_SOBEL_EDGE_THRESH 256
#define MV_PREC_DET_SSE_STATIC_THRESH 1024
#define MV_PREC_DET_BLK_SIZE_BITS 4
#define MV_PREC_DET_BLK_SIZE (1 << MV_PREC_DET_BLK_SIZE_BITS)
#define MV_PREC_DET_THRESH 0.20
#define MV_PREC_DET_THRESH2 0.10
#define MV_PREC_DET_QTHRESH 64  // Q thresh for 1/8-pel in edge based method

#define MV_HIPREC_QTHRESH 128   // Q thresh for 1/8-pel in q-based method
#define MV_HIPREC_QTHRESH2 192  // Q thresh for 1/4-pel in q-based method

// Q thresh for 1/8-pel in edge based method
#if !CONFIG_REALTIME_ONLY
static AOM_INLINE int_mv get_ref_mv_for_mv_stats(
    const MB_MODE_INFO *mbmi, const MB_MODE_INFO_EXT *mbmi_ext, int ref_idx) {
  int ref_mv_idx = mbmi->ref_mv_idx;
  if (mbmi->mode == NEAR_NEWMV || mbmi->mode == NEW_NEARMV) {
    assert(has_second_ref(mbmi));
    ref_mv_idx += 1;
  }

  const MV_REFERENCE_FRAME *ref_frames = mbmi->ref_frame;
  const int8_t ref_frame_type = av1_ref_frame_type(ref_frames);
  const CANDIDATE_MV *curr_ref_mv_stack =
      mbmi_ext->ref_mv_stack[ref_frame_type];

  if (ref_frames[1] > INTRA_FRAME) {
    assert(ref_idx == 0 || ref_idx == 1);
    return ref_idx ? curr_ref_mv_stack[ref_mv_idx].comp_mv
                   : curr_ref_mv_stack[ref_mv_idx].this_mv;
  }

  assert(ref_idx == 0);
  return ref_mv_idx < mbmi_ext->ref_mv_count[ref_frame_type]
             ? curr_ref_mv_stack[ref_mv_idx].this_mv
             : mbmi_ext->global_mvs[ref_frame_type];
}

static AOM_INLINE int get_symbol_cost(const aom_cdf_prob *cdf, int symbol) {
  const aom_cdf_prob cur_cdf = AOM_ICDF(cdf[symbol]);
  const aom_cdf_prob prev_cdf = symbol ? AOM_ICDF(cdf[symbol - 1]) : 0;
  const aom_cdf_prob p15 = AOMMAX(cur_cdf - prev_cdf, EC_MIN_PROB);

  return av1_cost_symbol(p15);
}

static AOM_INLINE int keep_one_comp_stat(MV_STATS *mv_stats, int comp_val,
                                         int comp_idx, const AV1_COMP *cpi,
                                         int *rates) {
  assert(comp_val != 0 && "mv component should not have zero value!");
  const int sign = comp_val < 0;
  const int mag = sign ? -comp_val : comp_val;
  const int mag_minus_1 = mag - 1;
  int offset;
  const int mv_class = av1_get_mv_class(mag_minus_1, &offset);
  const int int_part = offset >> 3;         // int mv data
  const int frac_part = (offset >> 1) & 3;  // fractional mv data
  const int high_part = offset & 1;         // high precision mv data
  const int use_hp = cpi->common.fr_mv_precision > MV_SUBPEL_QTR_PRECISION;
  int r_idx = 0;

  const MACROBLOCK *const x = &cpi->td.mb;
  const MACROBLOCKD *const xd = &x->e_mbd;
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
  nmv_context *nmvc = &ec_ctx->nmvc;
  nmv_component *mvcomp_ctx = nmvc->comps;
  nmv_component *cur_mvcomp_ctx = &mvcomp_ctx[comp_idx];
  aom_cdf_prob *sign_cdf = cur_mvcomp_ctx->sign_cdf;
  aom_cdf_prob *class_cdf = cur_mvcomp_ctx->classes_cdf;
  aom_cdf_prob *class0_cdf = cur_mvcomp_ctx->class0_cdf;
  aom_cdf_prob(*bits_cdf)[3] = cur_mvcomp_ctx->bits_cdf;
  aom_cdf_prob *high_part_cdf =
      mv_class ? (cur_mvcomp_ctx->hp_cdf) : (cur_mvcomp_ctx->class0_hp_cdf);

  const int sign_rate = get_symbol_cost(sign_cdf, sign);
  rates[r_idx++] = sign_rate;
  update_cdf(sign_cdf, sign, 2);

  const int class_rate = get_symbol_cost(class_cdf, mv_class);
  rates[r_idx++] = class_rate;
  update_cdf(class_cdf, mv_class, MV_CLASSES);

  int int_bit_rate = 0;
  if (mv_class == MV_CLASS_0) {
    int_bit_rate = get_symbol_cost(class0_cdf, int_part);
    update_cdf(class0_cdf, int_part, CLASS0_SIZE);
  } else {
    const int n = mv_class + CLASS0_BITS - 1;  // number of bits
    for (int i = 0; i < n; ++i) {
      int_bit_rate += get_symbol_cost(bits_cdf[i], (int_part >> i) & 1);
      update_cdf(bits_cdf[i], (int_part >> i) & 1, 2);
    }
  }
  rates[r_idx++] = int_bit_rate;
#if CONFIG_FLEX_MVRES
  aom_cdf_prob *frac_part_cdf =
      mv_class ? (cur_mvcomp_ctx->fp_cdf[0])
               : (cur_mvcomp_ctx->class0_fp_cdf[int_part][0]);
  int frac_part_rate_hpel = 0, frac_part_rate_qpel = 0;
  frac_part_rate_hpel = get_symbol_cost(frac_part_cdf, frac_part);
  rates[r_idx++] = frac_part_rate_hpel;
  update_cdf(frac_part_cdf, frac_part >> 1, 2);

  if (cpi->common.fr_mv_precision > MV_SUBPEL_HALF_PRECISION) {
    frac_part_cdf =
        mv_class
            ? (cur_mvcomp_ctx->fp_cdf[1 + (frac_part >> 1)])
            : (cur_mvcomp_ctx->class0_fp_cdf[int_part][1 + (frac_part >> 1)]);
    frac_part_rate_qpel = get_symbol_cost(frac_part_cdf, frac_part);
    rates[r_idx++] = frac_part_rate_qpel;
    update_cdf(frac_part_cdf, frac_part & 1, 2);
  }

#else
  aom_cdf_prob *frac_part_cdf = mv_class
                                    ? (cur_mvcomp_ctx->fp_cdf)
                                    : (cur_mvcomp_ctx->class0_fp_cdf[int_part]);
  const int frac_part_rate = get_symbol_cost(frac_part_cdf, frac_part);
  rates[r_idx++] = frac_part_rate;
  update_cdf(frac_part_cdf, frac_part, MV_FP_SIZE);
#endif

  const int high_part_rate =
      use_hp ? get_symbol_cost(high_part_cdf, high_part) : 0;
  if (use_hp) {
    update_cdf(high_part_cdf, high_part, 2);
  }
  rates[r_idx++] = high_part_rate;

  mv_stats->last_bit_zero += !high_part;
  mv_stats->last_bit_nonzero += high_part;

#if CONFIG_FLEX_MVRES
  const int total_rate =
      (sign_rate + class_rate + int_bit_rate + frac_part_rate_hpel +
       frac_part_rate_qpel + high_part_rate);
#else
  const int total_rate =
      (sign_rate + class_rate + int_bit_rate + frac_part_rate + high_part_rate);
#endif
  return total_rate;
}

static AOM_INLINE void keep_one_mv_stat(MV_STATS *mv_stats, const MV *ref_mv,
                                        const MV *cur_mv, const AV1_COMP *cpi) {
  const MACROBLOCK *const x = &cpi->td.mb;
  const MACROBLOCKD *const xd = &x->e_mbd;
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
  nmv_context *nmvc = &ec_ctx->nmvc;
  aom_cdf_prob *joint_cdf = nmvc->joints_cdf;
  const int use_hp = cpi->common.fr_mv_precision > MV_SUBPEL_QTR_PRECISION;

  const MV diff = { cur_mv->row - ref_mv->row, cur_mv->col - ref_mv->col };
  const int mv_joint = av1_get_mv_joint(&diff);
  // TODO(chiyotsai@google.com): Estimate hp_diff when we are using lp
  const MV hp_diff = diff;
  const int hp_mv_joint = av1_get_mv_joint(&hp_diff);
  const MV truncated_diff = { (diff.row / 2) * 2, (diff.col / 2) * 2 };
  const MV lp_diff = use_hp ? truncated_diff : diff;
  const int lp_mv_joint = av1_get_mv_joint(&lp_diff);

  aom_clear_system_state();
  const int mv_joint_rate = get_symbol_cost(joint_cdf, mv_joint);
  const int hp_mv_joint_rate = get_symbol_cost(joint_cdf, hp_mv_joint);
  const int lp_mv_joint_rate = get_symbol_cost(joint_cdf, lp_mv_joint);

  update_cdf(joint_cdf, mv_joint, MV_JOINTS);

  mv_stats->total_mv_rate += mv_joint_rate;
  mv_stats->hp_total_mv_rate += hp_mv_joint_rate;
  mv_stats->lp_total_mv_rate += lp_mv_joint_rate;
  mv_stats->mv_joint_count[mv_joint]++;

  for (int comp_idx = 0; comp_idx < 2; comp_idx++) {
    const int comp_val = comp_idx ? diff.col : diff.row;
    const int hp_comp_val = comp_idx ? hp_diff.col : hp_diff.row;
    const int lp_comp_val = comp_idx ? lp_diff.col : lp_diff.row;
    int rates[6];
    av1_zero_array(rates, 6);

    const int comp_rate =
        comp_val ? keep_one_comp_stat(mv_stats, comp_val, comp_idx, cpi, rates)
                 : 0;
    // TODO(chiyotsai@google.com): Properly get hp rate when use_hp is false
#if CONFIG_FLEX_MVRES
    const int hp_rate = hp_comp_val ? rates[0] + rates[1] + rates[2] +
                                          rates[3] + rates[4] + rates[5]
                                    : 0;
    const int lp_rate =
        lp_comp_val ? rates[0] + rates[1] + rates[2] + rates[3] + rates[4] : 0;
#else
    const int hp_rate =
        hp_comp_val ? rates[0] + rates[1] + rates[2] + rates[3] + rates[4] : 0;
    const int lp_rate =
        lp_comp_val ? rates[0] + rates[1] + rates[2] + rates[3] : 0;
#endif
    mv_stats->total_mv_rate += comp_rate;
    mv_stats->hp_total_mv_rate += hp_rate;
    mv_stats->lp_total_mv_rate += lp_rate;
  }
}

static AOM_INLINE void collect_mv_stats_b(MV_STATS *mv_stats,
                                          const AV1_COMP *cpi, int mi_row,
                                          int mi_col) {
  const AV1_COMMON *cm = &cpi->common;

  if (mi_row >= cm->mi_rows || mi_col >= cm->mi_cols) {
    return;
  }

  const MB_MODE_INFO *mbmi = cm->mi_grid_base[mi_row * cm->mi_stride + mi_col];
  const MB_MODE_INFO_EXT *mbmi_ext =
      cpi->mbmi_ext_base + (mi_row * cm->mi_cols + mi_col);

  if (!is_inter_block(mbmi)) {
    mv_stats->intra_count++;
    return;
  }
  mv_stats->inter_count++;

  const PREDICTION_MODE mode = mbmi->mode;
  const int is_compound = has_second_ref(mbmi);

  if (mode == NEWMV || mode == NEW_NEWMV) {
    // All mvs are new
    for (int ref_idx = 0; ref_idx < 1 + is_compound; ++ref_idx) {
      const MV ref_mv = get_ref_mv_for_mv_stats(mbmi, mbmi_ext, ref_idx).as_mv;
      const MV cur_mv = mbmi->mv[ref_idx].as_mv;
      keep_one_mv_stat(mv_stats, &ref_mv, &cur_mv, cpi);
    }
#if CONFIG_NEW_INTER_MODES
  } else if (mode == NEAR_NEWMV || mode == NEW_NEARMV) {
#else
  } else if (mode == NEAREST_NEWMV || mode == NEAR_NEWMV ||
             mode == NEW_NEARESTMV || mode == NEW_NEARMV) {
#endif
    // has exactly one new_mv
    mv_stats->default_mvs += 1;
#if CONFIG_NEW_INTER_MODES
    const int ref_idx = (mode == NEAR_NEWMV);
#else
    const int ref_idx = (mode == NEAREST_NEWMV || mode == NEAR_NEWMV);
#endif
    const MV ref_mv = get_ref_mv_for_mv_stats(mbmi, mbmi_ext, ref_idx).as_mv;
    const MV cur_mv = mbmi->mv[ref_idx].as_mv;

    keep_one_mv_stat(mv_stats, &ref_mv, &cur_mv, cpi);
  } else {
    // No new_mv
    mv_stats->default_mvs += 1 + is_compound;
  }

  // Add texture information
  const BLOCK_SIZE bsize = mbmi->sb_type;
  const int num_rows = block_size_high[bsize];
  const int num_cols = block_size_wide[bsize];
  const int y_stride = cpi->source->y_stride;
  const int px_row = 4 * mi_row, px_col = 4 * mi_col;
  const int buf_is_hbd = cpi->source->flags & YV12_FLAG_HIGHBITDEPTH;
  const int bd = cm->seq_params.bit_depth;
  if (buf_is_hbd) {
    uint16_t *source_buf =
        CONVERT_TO_SHORTPTR(cpi->source->y_buffer) + px_row * y_stride + px_col;
    for (int row = 0; row < num_rows - 1; row++) {
      for (int col = 0; col < num_cols - 1; col++) {
        const int offset = row * y_stride + col;
        const int horz_diff =
            abs(source_buf[offset + 1] - source_buf[offset]) >> (bd - 8);
        const int vert_diff =
            abs(source_buf[offset + y_stride] - source_buf[offset]) >> (bd - 8);
        mv_stats->horz_text += horz_diff;
        mv_stats->vert_text += vert_diff;
        mv_stats->diag_text += horz_diff * vert_diff;
      }
    }
  } else {
    uint8_t *source_buf = cpi->source->y_buffer + px_row * y_stride + px_col;
    for (int row = 0; row < num_rows - 1; row++) {
      for (int col = 0; col < num_cols - 1; col++) {
        const int offset = row * y_stride + col;
        const int horz_diff = abs(source_buf[offset + 1] - source_buf[offset]);
        const int vert_diff =
            abs(source_buf[offset + y_stride] - source_buf[offset]);
        mv_stats->horz_text += horz_diff;
        mv_stats->vert_text += vert_diff;
        mv_stats->diag_text += horz_diff * vert_diff;
      }
    }
  }
}

// Split block
static AOM_INLINE void collect_mv_stats_sb(MV_STATS *mv_stats,
                                           const AV1_COMP *cpi, int mi_row,
                                           int mi_col, BLOCK_SIZE bsize) {
  assert(bsize < BLOCK_SIZES_ALL);
  const AV1_COMMON *cm = &cpi->common;

  if (mi_row >= cm->mi_rows || mi_col >= cm->mi_cols) return;

  const PARTITION_TYPE partition = get_partition(cm, mi_row, mi_col, bsize);
  const BLOCK_SIZE subsize = get_partition_subsize(bsize, partition);

  const int hbs = mi_size_wide[bsize] / 2;
  const int qbs = mi_size_wide[bsize] / 4;

  switch (partition) {
    case PARTITION_NONE:
      collect_mv_stats_b(mv_stats, cpi, mi_row, mi_col);
      break;
    case PARTITION_HORZ:
      collect_mv_stats_b(mv_stats, cpi, mi_row, mi_col);
      collect_mv_stats_b(mv_stats, cpi, mi_row + hbs, mi_col);
      break;
    case PARTITION_VERT:
      collect_mv_stats_b(mv_stats, cpi, mi_row, mi_col);
      collect_mv_stats_b(mv_stats, cpi, mi_row, mi_col + hbs);
      break;
    case PARTITION_SPLIT:
      collect_mv_stats_sb(mv_stats, cpi, mi_row, mi_col, subsize);
      collect_mv_stats_sb(mv_stats, cpi, mi_row, mi_col + hbs, subsize);
      collect_mv_stats_sb(mv_stats, cpi, mi_row + hbs, mi_col, subsize);
      collect_mv_stats_sb(mv_stats, cpi, mi_row + hbs, mi_col + hbs, subsize);
      break;
#if !CONFIG_EXT_RECUR_PARTITIONS
    case PARTITION_HORZ_A:
      collect_mv_stats_b(mv_stats, cpi, mi_row, mi_col);
      collect_mv_stats_b(mv_stats, cpi, mi_row, mi_col + hbs);
      collect_mv_stats_b(mv_stats, cpi, mi_row + hbs, mi_col);
      break;
    case PARTITION_HORZ_B:
      collect_mv_stats_b(mv_stats, cpi, mi_row, mi_col);
      collect_mv_stats_b(mv_stats, cpi, mi_row + hbs, mi_col);
      collect_mv_stats_b(mv_stats, cpi, mi_row + hbs, mi_col + hbs);
      break;
    case PARTITION_VERT_A:
      collect_mv_stats_b(mv_stats, cpi, mi_row, mi_col);
      collect_mv_stats_b(mv_stats, cpi, mi_row + hbs, mi_col);
      collect_mv_stats_b(mv_stats, cpi, mi_row, mi_col + hbs);
      break;
    case PARTITION_VERT_B:
      collect_mv_stats_b(mv_stats, cpi, mi_row, mi_col);
      collect_mv_stats_b(mv_stats, cpi, mi_row, mi_col + hbs);
      collect_mv_stats_b(mv_stats, cpi, mi_row + hbs, mi_col + hbs);
      break;
#endif  // !CONFIG_EXT_RECUR_PARTITIONS
#if CONFIG_EXT_RECUR_PARTITIONS
    case PARTITION_HORZ_3: {
      collect_mv_stats_b(mv_stats, cpi, mi_row, mi_col);
      collect_mv_stats_b(mv_stats, cpi, mi_row + qbs, mi_col);
      collect_mv_stats_b(mv_stats, cpi, mi_row + 3 * qbs, mi_col);
      break;
    }
    case PARTITION_VERT_3: {
      collect_mv_stats_b(mv_stats, cpi, mi_row, mi_col);
      collect_mv_stats_b(mv_stats, cpi, mi_row, mi_col + qbs);
      collect_mv_stats_b(mv_stats, cpi, mi_row, mi_col + 3 * qbs);
      break;
    }
#else
    case PARTITION_HORZ_4:
      for (int i = 0; i < 4; ++i) {
        const int this_mi_row = mi_row + i * qbs;
        collect_mv_stats_b(mv_stats, cpi, this_mi_row, mi_col);
      }
      break;
    case PARTITION_VERT_4:
      for (int i = 0; i < 4; ++i) {
        const int this_mi_col = mi_col + i * qbs;
        collect_mv_stats_b(mv_stats, cpi, mi_row, this_mi_col);
      }
      break;
#endif  // CONFIG_EXT_RECUR_PARTITIONS
    default: assert(0);
  }
}

static AOM_INLINE void collect_mv_stats_tile(MV_STATS *mv_stats,
                                             const AV1_COMP *cpi,
                                             const TileInfo *tile_info) {
  const AV1_COMMON *cm = &cpi->common;
  const int mi_row_start = tile_info->mi_row_start;
  const int mi_row_end = tile_info->mi_row_end;
  const int mi_col_start = tile_info->mi_col_start;
  const int mi_col_end = tile_info->mi_col_end;
  const int sb_size_mi = cm->seq_params.mib_size;
  BLOCK_SIZE sb_size = cm->seq_params.sb_size;
  for (int mi_row = mi_row_start; mi_row < mi_row_end; mi_row += sb_size_mi) {
    for (int mi_col = mi_col_start; mi_col < mi_col_end; mi_col += sb_size_mi) {
      collect_mv_stats_sb(mv_stats, cpi, mi_row, mi_col, sb_size);
    }
  }
}

void av1_collect_mv_stats(AV1_COMP *cpi, int current_q) {
  MV_STATS *mv_stats = &cpi->mv_stats;
  const AV1_COMMON *cm = &cpi->common;
  const int tile_cols = cm->tile_cols;
  const int tile_rows = cm->tile_rows;

  for (int tile_row = 0; tile_row < tile_rows; tile_row++) {
    TileInfo tile_info;
    av1_tile_set_row(&tile_info, cm, tile_row);
    for (int tile_col = 0; tile_col < tile_cols; tile_col++) {
      const int tile_idx = tile_row * tile_cols + tile_col;
      av1_tile_set_col(&tile_info, cm, tile_col);
      cpi->tile_data[tile_idx].tctx = *cm->fc;
      cpi->td.mb.e_mbd.tile_ctx = &cpi->tile_data[tile_idx].tctx;
      collect_mv_stats_tile(mv_stats, cpi, &tile_info);
    }
  }

  mv_stats->q = current_q;
  mv_stats->order = cpi->common.current_frame.order_hint;
  mv_stats->valid = 1;
}

static AOM_INLINE int get_smart_mv_prec(AV1_COMP *cpi, const MV_STATS *mv_stats,
                                        int current_q) {
  const AV1_COMMON *cm = &cpi->common;
  const int order_hint = cpi->common.current_frame.order_hint;
  const int order_diff = order_hint - mv_stats->order;
  aom_clear_system_state();
  const float area = cm->width * cm->height;
  float features[MV_PREC_FEATURE_SIZE] = {
    current_q,
    mv_stats->q,
    order_diff,
    mv_stats->inter_count / area,
    mv_stats->intra_count / area,
    mv_stats->default_mvs / area,
    mv_stats->mv_joint_count[0] / area,
    mv_stats->mv_joint_count[1] / area,
    mv_stats->mv_joint_count[2] / area,
    mv_stats->mv_joint_count[3] / area,
    mv_stats->last_bit_zero / area,
    mv_stats->last_bit_nonzero / area,
    mv_stats->total_mv_rate / area,
    mv_stats->hp_total_mv_rate / area,
    mv_stats->lp_total_mv_rate / area,
    mv_stats->horz_text / area,
    mv_stats->vert_text / area,
    mv_stats->diag_text / area,
  };

  for (int f_idx = 0; f_idx < MV_PREC_FEATURE_SIZE; f_idx++) {
    features[f_idx] =
        (features[f_idx] - av1_mv_prec_mean[f_idx]) / av1_mv_prec_std[f_idx];
  }
  float score = 0.0f;

  av1_nn_predict(features, &av1_mv_prec_dnn_config, 1, &score);

  const int use_high_hp = score >= 0.0f;
  return use_high_hp;
}
#endif  // !CONFIG_REALTIME_ONLY
// Compute edge energy in the frame
static MvSubpelPrecision determine_frame_mv_precision(const AV1_COMP *cpi,
                                                      int q, int use_edges) {
  (void)cpi;
  if (!use_edges) {
    return q < MV_HIPREC_QTHRESH ? MV_SUBPEL_EIGHTH_PRECISION
                                 : MV_SUBPEL_QTR_PRECISION;
  }
  if (q < MV_PREC_DET_QTHRESH) return MV_SUBPEL_EIGHTH_PRECISION;
  const YV12_BUFFER_CONFIG *srcbuf = cpi->source;
  const YV12_BUFFER_CONFIG *refbuf =
      get_ref_frame_yv12_buf(&cpi->common, GOLDEN_FRAME);
  const int bd = cpi->td.mb.e_mbd.bd;
  const int width = srcbuf->y_crop_width;
  const int height = srcbuf->y_crop_height;
  const int stride = srcbuf->y_stride;
  const int ref_stride = refbuf->y_stride;
  int num_blks[2] = { 0, 0 };
  if (srcbuf->flags & YV12_FLAG_HIGHBITDEPTH) {
    const uint16_t *src16 =
        (const uint16_t *)CONVERT_TO_SHORTPTR(srcbuf->y_buffer);
    const uint16_t *ref16 =
        (const uint16_t *)CONVERT_TO_SHORTPTR(refbuf->y_buffer);
    for (int i = 0; i < height - MV_PREC_DET_BLK_SIZE;
         i += MV_PREC_DET_BLK_SIZE) {
      for (int j = 0; j < width - MV_PREC_DET_BLK_SIZE;
           j += MV_PREC_DET_BLK_SIZE) {
        const uint16_t *src = src16 + i * stride + j;
        const uint16_t *ref = ref16 + i * stride + j;
        int64_t sse = aom_highbd_sse(
            CONVERT_TO_BYTEPTR(src), stride, CONVERT_TO_BYTEPTR(ref),
            ref_stride, MV_PREC_DET_BLK_SIZE, MV_PREC_DET_BLK_SIZE);
        int64_t sse_norm = ROUND_POWER_OF_TWO(
            sse, 2 * MV_PREC_DET_BLK_SIZE_BITS + 2 * (bd - 8));
        if (sse_norm < MV_PREC_DET_SSE_STATIC_THRESH) continue;
        int64_t gx = 0, gy = 0, g;
        for (int y = 0; y < MV_PREC_DET_BLK_SIZE; ++y) {
          for (int x = 0; x < MV_PREC_DET_BLK_SIZE; ++x) {
            gx +=
                abs(src[-stride + 1] - src[-stride - 1] +
                    (src[1] - src[-1]) * 2 + src[stride + 1] - src[stride - 1]);
            gy += abs(src[stride - 1] - src[-stride - 1] +
                      (src[stride] - src[-stride]) * 2 + src[stride + 1] -
                      src[-stride + 1]);
            src++;
          }
          src += stride - MV_PREC_DET_BLK_SIZE;
        }
        g = gx * gx + gy * gy;
        // Normalize to per pixel and bit-depth of 8
        g = ROUND_POWER_OF_TWO(
            g, 4 * MV_PREC_DET_BLK_SIZE_BITS + 4 + 2 * (bd - 8));
        ++num_blks[g > MV_PREC_DET_SOBEL_EDGE_THRESH];
      }
    }
  } else {
    const uint8_t *src8 = srcbuf->y_buffer;
    const uint8_t *ref8 = refbuf->y_buffer;
    for (int i = 0; i < height - MV_PREC_DET_BLK_SIZE;
         i += MV_PREC_DET_BLK_SIZE) {
      for (int j = 0; j < width - MV_PREC_DET_BLK_SIZE;
           j += MV_PREC_DET_BLK_SIZE) {
        const uint8_t *src = src8 + i * stride + j;
        const uint8_t *ref = ref8 + i * stride + j;
        int64_t sse = aom_sse(src, stride, ref, ref_stride,
                              MV_PREC_DET_BLK_SIZE, MV_PREC_DET_BLK_SIZE);
        int64_t sse_norm =
            ROUND_POWER_OF_TWO(sse, 2 * MV_PREC_DET_BLK_SIZE_BITS);
        if (sse_norm < MV_PREC_DET_SSE_STATIC_THRESH) continue;
        int64_t gx = 0, gy = 0, g;
        for (int y = 0; y < MV_PREC_DET_BLK_SIZE; ++y) {
          for (int x = 0; x < MV_PREC_DET_BLK_SIZE; ++x) {
            gx +=
                abs(src[-stride + 1] - src[-stride - 1] +
                    (src[1] - src[-1]) * 2 + src[stride + 1] - src[stride - 1]);
            gy += abs(src[stride - 1] - src[-stride - 1] +
                      (src[stride] - src[-stride]) * 2 + src[stride + 1] -
                      src[-stride + 1]);
            src++;
          }
          src += stride - MV_PREC_DET_BLK_SIZE;
        }
        g = gx * gx + gy * gy;
        // Normalize to per pixel and bit-depth of 8
        g = ROUND_POWER_OF_TWO(
            g, 4 * MV_PREC_DET_BLK_SIZE_BITS + 4 + 2 * (bd - 8));
        ++num_blks[g > MV_PREC_DET_SOBEL_EDGE_THRESH];
      }
    }
  }
  if (num_blks[0] + num_blks[1] == 0) return MV_SUBPEL_QTR_PRECISION;
  const double pct_edge_blks =
      100.0 * (double)num_blks[1] / (num_blks[0] + num_blks[1]);
  const double pct_edge_blks_by_q = pct_edge_blks / q;
  // printf("pct_edge_blks = %f [%d + %d], q = %d, ratio = %f\n", pct_edge_blks,
  //        num_blks[0], num_blks[1], q, pct_edge_blks_by_q);
  if (pct_edge_blks_by_q >= MV_PREC_DET_THRESH)
    return MV_SUBPEL_EIGHTH_PRECISION;
  return MV_SUBPEL_QTR_PRECISION;
}

#if CONFIG_FLEX_MVRES
#define FLEX_MV_PRECISION_QTHRESH 256  // Reduce to turn off at low quality
static int determine_pb_flex_mv_precision(const AV1_COMP *cpi, int q) {
  return (cpi->common.fr_mv_precision >= MV_SUBPEL_QTR_PRECISION &&
          !is_stat_generation_stage(cpi) && q <= FLEX_MV_PRECISION_QTHRESH);
}
#endif  // CONFIG_FLEX_MVRES

void av1_pick_and_set_high_precision_mv(AV1_COMP *cpi, int q) {
  MvSubpelPrecision precision = cpi->common.cur_frame_force_integer_mv
                                    ? MV_SUBPEL_NONE
                                    : determine_frame_mv_precision(cpi, q, 0);

  assert(IMPLIES(!cpi->common.cur_frame_force_integer_mv,
                 precision >= MV_SUBPEL_QTR_PRECISION));

  if (cpi->sf.high_precision_mv_usage == QTR_ONLY)
    precision = MV_SUBPEL_QTR_PRECISION;

#if !CONFIG_REALTIME_ONLY
  else if (cpi->sf.high_precision_mv_usage == LAST_MV_DATA &&
           av1_frame_allows_smart_mv(cpi) && cpi->mv_stats.valid) {
    if (get_smart_mv_prec(cpi, &cpi->mv_stats, q))
      precision = MV_SUBPEL_EIGHTH_PRECISION;
  }
#endif  // !CONFIG_REALTIME_ONLY

  av1_set_mv_precision(cpi, precision, cpi->common.cur_frame_force_integer_mv);

#if CONFIG_FLEX_MVRES
  cpi->common.use_sb_mv_precision = ENABLE_SB_RES;
  cpi->common.use_pb_mv_precision =
      ENABLE_PB_RES ? determine_pb_flex_mv_precision(cpi, q) : 0;
#endif  // CONFIG_FLEX_MVRES
}
