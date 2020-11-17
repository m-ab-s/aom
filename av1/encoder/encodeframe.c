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

#include <limits.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"
#include "config/av1_rtcd.h"

#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/binary_codes_writer.h"
#include "aom_ports/mem.h"
#include "aom_ports/aom_timer.h"
#include "aom_ports/system_state.h"

#if CONFIG_MISMATCH_DEBUG
#include "aom_util/debug_util.h"
#endif  // CONFIG_MISMATCH_DEBUG

#include "av1/common/cfl.h"
#include "av1/common/common.h"
#include "av1/common/entropy.h"
#include "av1/common/entropymode.h"
#include "av1/common/idct.h"
#include "av1/common/mv.h"
#include "av1/common/mvref_common.h"
#include "av1/common/pred_common.h"
#include "av1/common/quant_common.h"
#include "av1/common/reconintra.h"
#include "av1/common/reconinter.h"
#include "av1/common/seg_common.h"
#include "av1/common/tile_common.h"
#include "av1/common/warped_motion.h"

#include "av1/encoder/aq_complexity.h"
#include "av1/encoder/aq_cyclicrefresh.h"
#include "av1/encoder/aq_variance.h"
#include "av1/encoder/corner_detect.h"
#include "av1/encoder/global_motion.h"
#include "av1/encoder/encodeframe.h"
#include "av1/encoder/encodemb.h"
#include "av1/encoder/encodemv.h"
#include "av1/encoder/encodetxb.h"
#include "av1/encoder/ethread.h"
#include "av1/encoder/extend.h"
#include "av1/encoder/ml.h"
#include "av1/encoder/partition_search.h"
#include "av1/encoder/partition_strategy.h"
#if !CONFIG_REALTIME_ONLY
#include "av1/encoder/partition_model_weights.h"
#endif
#include "av1/encoder/rd.h"
#include "av1/encoder/rdopt.h"
#include "av1/encoder/reconinter_enc.h"
#include "av1/encoder/segmentation.h"
#include "av1/encoder/tokenize.h"
#include "av1/encoder/tpl_model.h"
#include "av1/encoder/var_based_part.h"
#include "av1/encoder/tpl_model.h"

// This is used as a reference when computing the source variance for the
//  purposes of activity masking.
// Eventually this should be replaced by custom no-reference routines,
//  which will be faster.
const uint8_t AV1_VAR_OFFS[MAX_SB_SIZE] = {
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128
};

static const uint16_t AV1_HIGH_VAR_OFFS_8[MAX_SB_SIZE] = {
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128
};

static const uint16_t AV1_HIGH_VAR_OFFS_10[MAX_SB_SIZE] = {
  128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4,
  128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4,
  128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4,
  128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4,
  128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4,
  128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4,
  128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4,
  128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4,
  128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4,
  128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4,
  128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4,
  128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4,
  128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4,
  128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4,
  128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4,
  128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4
};

static const uint16_t AV1_HIGH_VAR_OFFS_12[MAX_SB_SIZE] = {
  128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16,
  128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16,
  128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16,
  128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16,
  128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16,
  128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16,
  128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16,
  128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16,
  128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16,
  128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16,
  128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16,
  128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16,
  128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16,
  128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16,
  128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16,
  128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16,
  128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16,
  128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16,
  128 * 16, 128 * 16
};

unsigned int av1_get_sby_perpixel_variance(const AV1_COMP *cpi,
                                           const struct buf_2d *ref,
                                           BLOCK_SIZE bs) {
  unsigned int sse;
  const unsigned int var =
      cpi->fn_ptr[bs].vf(ref->buf, ref->stride, AV1_VAR_OFFS, 0, &sse);
  return ROUND_POWER_OF_TWO(var, num_pels_log2_lookup[bs]);
}

unsigned int av1_high_get_sby_perpixel_variance(const AV1_COMP *cpi,
                                                const struct buf_2d *ref,
                                                BLOCK_SIZE bs, int bd) {
  unsigned int var, sse;
  switch (bd) {
    case 10:
      var =
          cpi->fn_ptr[bs].vf(ref->buf, ref->stride,
                             CONVERT_TO_BYTEPTR(AV1_HIGH_VAR_OFFS_10), 0, &sse);
      break;
    case 12:
      var =
          cpi->fn_ptr[bs].vf(ref->buf, ref->stride,
                             CONVERT_TO_BYTEPTR(AV1_HIGH_VAR_OFFS_12), 0, &sse);
      break;
    case 8:
    default:
      var =
          cpi->fn_ptr[bs].vf(ref->buf, ref->stride,
                             CONVERT_TO_BYTEPTR(AV1_HIGH_VAR_OFFS_8), 0, &sse);
      break;
  }
  return ROUND_POWER_OF_TWO(var, num_pels_log2_lookup[bs]);
}

#if !CONFIG_REALTIME_ONLY
static unsigned int get_sby_perpixel_diff_variance(const AV1_COMP *const cpi,
                                                   const struct buf_2d *ref,
                                                   int mi_row, int mi_col,
                                                   BLOCK_SIZE bs) {
  unsigned int sse, var;
  uint8_t *last_y;
  const YV12_BUFFER_CONFIG *last =
      get_ref_frame_yv12_buf(&cpi->common, LAST_FRAME);

  assert(last != NULL);
  last_y =
      &last->y_buffer[mi_row * MI_SIZE * last->y_stride + mi_col * MI_SIZE];
  var = cpi->fn_ptr[bs].vf(ref->buf, ref->stride, last_y, last->y_stride, &sse);
  return ROUND_POWER_OF_TWO(var, num_pels_log2_lookup[bs]);
}

static BLOCK_SIZE get_rd_var_based_fixed_partition(AV1_COMP *cpi, MACROBLOCK *x,
                                                   int mi_row, int mi_col) {
  unsigned int var = get_sby_perpixel_diff_variance(
      cpi, &x->plane[0].src, mi_row, mi_col, BLOCK_64X64);
  if (var < 8)
    return BLOCK_64X64;
  else if (var < 128)
    return BLOCK_32X32;
  else if (var < 2048)
    return BLOCK_16X16;
  else
    return BLOCK_8X8;
}
#endif  // !CONFIG_REALTIME_ONLY

void av1_setup_src_planes(MACROBLOCK *x, const YV12_BUFFER_CONFIG *src,
                          int mi_row, int mi_col, const int num_planes,
                          const CHROMA_REF_INFO *chr_ref_info) {
  // Set current frame pointer.
  x->e_mbd.cur_buf = src;

  // We use AOMMIN(num_planes, MAX_MB_PLANE) instead of num_planes to quiet
  // the static analysis warnings.
  for (int i = 0; i < AOMMIN(num_planes, MAX_MB_PLANE); i++) {
    const int is_uv = i > 0;
    setup_pred_plane(&x->plane[i].src, src->buffers[i], src->crop_widths[is_uv],
                     src->crop_heights[is_uv], src->strides[is_uv], mi_row,
                     mi_col, NULL, x->e_mbd.plane[i].subsampling_x,
                     x->e_mbd.plane[i].subsampling_y, i > 0, chr_ref_info);
  }
}

#define AVG_CDF_WEIGHT_LEFT 3
#define AVG_CDF_WEIGHT_TOP_RIGHT 1

static void avg_cdf_symbol(aom_cdf_prob *cdf_ptr_left, aom_cdf_prob *cdf_ptr_tr,
                           int num_cdfs, int cdf_stride, int nsymbs,
                           int wt_left, int wt_tr) {
  for (int i = 0; i < num_cdfs; i++) {
    for (int j = 0; j <= nsymbs; j++) {
      cdf_ptr_left[i * cdf_stride + j] =
          (aom_cdf_prob)(((int)cdf_ptr_left[i * cdf_stride + j] * wt_left +
                          (int)cdf_ptr_tr[i * cdf_stride + j] * wt_tr +
                          ((wt_left + wt_tr) / 2)) /
                         (wt_left + wt_tr));
      assert(cdf_ptr_left[i * cdf_stride + j] >= 0 &&
             cdf_ptr_left[i * cdf_stride + j] < CDF_PROB_TOP);
    }
  }
}

#define AVERAGE_CDF(cname_left, cname_tr, nsymbs) \
  AVG_CDF_STRIDE(cname_left, cname_tr, nsymbs, CDF_SIZE(nsymbs))

#define AVG_CDF_STRIDE(cname_left, cname_tr, nsymbs, cdf_stride)           \
  do {                                                                     \
    aom_cdf_prob *cdf_ptr_left = (aom_cdf_prob *)cname_left;               \
    aom_cdf_prob *cdf_ptr_tr = (aom_cdf_prob *)cname_tr;                   \
    int array_size = (int)sizeof(cname_left) / sizeof(aom_cdf_prob);       \
    int num_cdfs = array_size / cdf_stride;                                \
    avg_cdf_symbol(cdf_ptr_left, cdf_ptr_tr, num_cdfs, cdf_stride, nsymbs, \
                   wt_left, wt_tr);                                        \
  } while (0)

static void avg_nmv(nmv_context *nmv_left, nmv_context *nmv_tr, int wt_left,
                    int wt_tr) {
  AVERAGE_CDF(nmv_left->joints_cdf, nmv_tr->joints_cdf, 4);
  for (int i = 0; i < 2; i++) {
    AVERAGE_CDF(nmv_left->comps[i].classes_cdf, nmv_tr->comps[i].classes_cdf,
                MV_CLASSES);
#if CONFIG_FLEX_MVRES
    AVERAGE_CDF(nmv_left->comps[i].class0_fp_cdf,
                nmv_tr->comps[i].class0_fp_cdf, 2);
    AVERAGE_CDF(nmv_left->comps[i].fp_cdf, nmv_tr->comps[i].fp_cdf, 2);
#else
    AVERAGE_CDF(nmv_left->comps[i].class0_fp_cdf,
                nmv_tr->comps[i].class0_fp_cdf, MV_FP_SIZE);
    AVERAGE_CDF(nmv_left->comps[i].fp_cdf, nmv_tr->comps[i].fp_cdf, MV_FP_SIZE);
#endif  // CONFIG_FLEX_MVRES
    AVERAGE_CDF(nmv_left->comps[i].sign_cdf, nmv_tr->comps[i].sign_cdf, 2);
    AVERAGE_CDF(nmv_left->comps[i].class0_hp_cdf,
                nmv_tr->comps[i].class0_hp_cdf, 2);
    AVERAGE_CDF(nmv_left->comps[i].hp_cdf, nmv_tr->comps[i].hp_cdf, 2);
    AVERAGE_CDF(nmv_left->comps[i].class0_cdf, nmv_tr->comps[i].class0_cdf,
                CLASS0_SIZE);
    AVERAGE_CDF(nmv_left->comps[i].bits_cdf, nmv_tr->comps[i].bits_cdf, 2);
  }
}

// In case of row-based multi-threading of encoder, since we always
// keep a top - right sync, we can average the top - right SB's CDFs and
// the left SB's CDFs and use the same for current SB's encoding to
// improve the performance. This function facilitates the averaging
// of CDF and used only when row-mt is enabled in encoder.
static void avg_cdf_symbols(FRAME_CONTEXT *ctx_left, FRAME_CONTEXT *ctx_tr,
                            int wt_left, int wt_tr) {
  AVERAGE_CDF(ctx_left->txb_skip_cdf, ctx_tr->txb_skip_cdf, 2);
  AVERAGE_CDF(ctx_left->eob_extra_cdf, ctx_tr->eob_extra_cdf, 2);
  AVERAGE_CDF(ctx_left->dc_sign_cdf, ctx_tr->dc_sign_cdf, 2);
  AVERAGE_CDF(ctx_left->eob_flag_cdf16, ctx_tr->eob_flag_cdf16, 5);
  AVERAGE_CDF(ctx_left->eob_flag_cdf32, ctx_tr->eob_flag_cdf32, 6);
  AVERAGE_CDF(ctx_left->eob_flag_cdf64, ctx_tr->eob_flag_cdf64, 7);
  AVERAGE_CDF(ctx_left->eob_flag_cdf128, ctx_tr->eob_flag_cdf128, 8);
  AVERAGE_CDF(ctx_left->eob_flag_cdf256, ctx_tr->eob_flag_cdf256, 9);
  AVERAGE_CDF(ctx_left->eob_flag_cdf512, ctx_tr->eob_flag_cdf512, 10);
  AVERAGE_CDF(ctx_left->eob_flag_cdf1024, ctx_tr->eob_flag_cdf1024, 11);
  AVERAGE_CDF(ctx_left->coeff_base_eob_cdf, ctx_tr->coeff_base_eob_cdf, 3);
  AVERAGE_CDF(ctx_left->coeff_base_cdf, ctx_tr->coeff_base_cdf, 4);
  AVERAGE_CDF(ctx_left->coeff_br_cdf, ctx_tr->coeff_br_cdf, BR_CDF_SIZE);
  AVERAGE_CDF(ctx_left->newmv_cdf, ctx_tr->newmv_cdf, 2);
  AVERAGE_CDF(ctx_left->zeromv_cdf, ctx_tr->zeromv_cdf, 2);
#if CONFIG_NEW_INTER_MODES
  AVERAGE_CDF(ctx_left->drl0_cdf, ctx_tr->drl0_cdf, 2);
  AVERAGE_CDF(ctx_left->drl1_cdf, ctx_tr->drl1_cdf, 2);
  AVERAGE_CDF(ctx_left->drl2_cdf, ctx_tr->drl2_cdf, 2);
#else
  AVERAGE_CDF(ctx_left->refmv_cdf, ctx_tr->refmv_cdf, 2);
  AVERAGE_CDF(ctx_left->drl_cdf, ctx_tr->drl_cdf, 2);
#endif  // CONFIG_NEW_INTER_MODES
  AVERAGE_CDF(ctx_left->inter_compound_mode_cdf,
              ctx_tr->inter_compound_mode_cdf, INTER_COMPOUND_MODES);
  AVERAGE_CDF(ctx_left->compound_type_cdf, ctx_tr->compound_type_cdf,
              MASKED_COMPOUND_TYPES);
  AVERAGE_CDF(ctx_left->wedge_idx_cdf, ctx_tr->wedge_idx_cdf, 16);
  AVERAGE_CDF(ctx_left->interintra_cdf, ctx_tr->interintra_cdf, 2);
  AVERAGE_CDF(ctx_left->wedge_interintra_cdf, ctx_tr->wedge_interintra_cdf, 2);
  AVERAGE_CDF(ctx_left->interintra_mode_cdf, ctx_tr->interintra_mode_cdf,
              INTERINTRA_MODES);
  AVERAGE_CDF(ctx_left->motion_mode_cdf, ctx_tr->motion_mode_cdf, MOTION_MODES);
#if CONFIG_EXT_WARP && CONFIG_SUB8X8_WARP
  AVERAGE_CDF(ctx_left->warp_cdf, ctx_tr->warp_cdf, 2);
#endif  // CONFIG_EXT_WARP && CONFIG_SUB8X8_WARP
  AVERAGE_CDF(ctx_left->obmc_cdf, ctx_tr->obmc_cdf, 2);
  AVERAGE_CDF(ctx_left->palette_y_size_cdf, ctx_tr->palette_y_size_cdf,
              PALETTE_SIZES);
  AVERAGE_CDF(ctx_left->palette_uv_size_cdf, ctx_tr->palette_uv_size_cdf,
              PALETTE_SIZES);
  for (int j = 0; j < PALETTE_SIZES; j++) {
    int nsymbs = j + PALETTE_MIN_SIZE;
    AVG_CDF_STRIDE(ctx_left->palette_y_color_index_cdf[j],
                   ctx_tr->palette_y_color_index_cdf[j], nsymbs,
                   CDF_SIZE(PALETTE_COLORS));
    AVG_CDF_STRIDE(ctx_left->palette_uv_color_index_cdf[j],
                   ctx_tr->palette_uv_color_index_cdf[j], nsymbs,
                   CDF_SIZE(PALETTE_COLORS));
  }
  AVERAGE_CDF(ctx_left->palette_y_mode_cdf, ctx_tr->palette_y_mode_cdf, 2);
  AVERAGE_CDF(ctx_left->palette_uv_mode_cdf, ctx_tr->palette_uv_mode_cdf, 2);
  AVERAGE_CDF(ctx_left->comp_inter_cdf, ctx_tr->comp_inter_cdf, 2);
  AVERAGE_CDF(ctx_left->single_ref_cdf, ctx_tr->single_ref_cdf, 2);
  AVERAGE_CDF(ctx_left->comp_ref_type_cdf, ctx_tr->comp_ref_type_cdf, 2);
  AVERAGE_CDF(ctx_left->uni_comp_ref_cdf, ctx_tr->uni_comp_ref_cdf, 2);
  AVERAGE_CDF(ctx_left->comp_ref_cdf, ctx_tr->comp_ref_cdf, 2);
  AVERAGE_CDF(ctx_left->comp_bwdref_cdf, ctx_tr->comp_bwdref_cdf, 2);
#if CONFIG_NEW_TX_PARTITION
  // Square blocks
  AVERAGE_CDF(ctx_left->txfm_partition_cdf[0], ctx_tr->txfm_partition_cdf[0],
              TX_PARTITION_TYPES);
  // Rectangular blocks
  AVERAGE_CDF(ctx_left->txfm_partition_cdf[1], ctx_tr->txfm_partition_cdf[1],
              TX_PARTITION_TYPES);
#else   // CONFIG_NEW_TX_PARTITION
  AVERAGE_CDF(ctx_left->txfm_partition_cdf, ctx_tr->txfm_partition_cdf, 2);
#endif  // CONFIG_NEW_TX_PARTITION
  AVERAGE_CDF(ctx_left->compound_index_cdf, ctx_tr->compound_index_cdf, 2);
  AVERAGE_CDF(ctx_left->comp_group_idx_cdf, ctx_tr->comp_group_idx_cdf, 2);
  AVERAGE_CDF(ctx_left->skip_mode_cdfs, ctx_tr->skip_mode_cdfs, 2);
  AVERAGE_CDF(ctx_left->skip_cdfs, ctx_tr->skip_cdfs, 2);
#if CONFIG_DSPL_RESIDUAL
  AVERAGE_CDF(ctx_left->dspl_type_cdf, ctx_tr->dspl_type_cdf, DSPL_END);
#endif  // CONFIG_DSPL_RESIDUAL
  AVERAGE_CDF(ctx_left->intra_inter_cdf, ctx_tr->intra_inter_cdf, 2);
  avg_nmv(&ctx_left->nmvc, &ctx_tr->nmvc, wt_left, wt_tr);
  avg_nmv(&ctx_left->ndvc, &ctx_tr->ndvc, wt_left, wt_tr);
  AVERAGE_CDF(ctx_left->intrabc_cdf, ctx_tr->intrabc_cdf, 2);
#if CONFIG_EXT_IBC_MODES
  AVERAGE_CDF(ctx_left->intrabc_mode_cdf, ctx_tr->intrabc_mode_cdf, 8);
  // AVERAGE_CDF(ctx_left->intrabc_mode_cdf, ctx_tr->intrabc_mode_cdf, 6);
  // AVERAGE_CDF(ctx_left->intrabc_mode_cdf, ctx_tr->intrabc_mode_cdf, 4);
#endif  // CONFIG_EXT_IBC_MODES
#if CONFIG_DERIVED_INTRA_MODE
  AVERAGE_CDF(ctx_left->derived_intra_mode_cdf, ctx_tr->derived_intra_mode_cdf,
              2);
  AVERAGE_CDF(ctx_left->uv_derived_intra_mode_cdf,
              ctx_tr->uv_derived_intra_mode_cdf, 2);
#endif  // CONFIG_DERIVED_INTRA_MODE
#if CONFIG_DERIVED_MV
  AVERAGE_CDF(ctx_left->use_derived_mv_cdf, ctx_tr->use_derived_mv_cdf, 2);
#endif  // CONFIG_DERIVED_MV
  AVERAGE_CDF(ctx_left->seg.tree_cdf, ctx_tr->seg.tree_cdf, MAX_SEGMENTS);
  AVERAGE_CDF(ctx_left->seg.pred_cdf, ctx_tr->seg.pred_cdf, 2);
  AVERAGE_CDF(ctx_left->seg.spatial_pred_seg_cdf,
              ctx_tr->seg.spatial_pred_seg_cdf, MAX_SEGMENTS);
  AVERAGE_CDF(ctx_left->filter_intra_cdfs, ctx_tr->filter_intra_cdfs, 2);
  AVERAGE_CDF(ctx_left->filter_intra_mode_cdf, ctx_tr->filter_intra_mode_cdf,
              FILTER_INTRA_MODES);
#if CONFIG_LOOP_RESTORE_CNN
  AVERAGE_CDF(ctx_left->switchable_restore_cdf[0],
              ctx_tr->switchable_restore_cdf[0], RESTORE_SWITCHABLE_TYPES - 1);
  AVERAGE_CDF(ctx_left->switchable_restore_cdf[1],
              ctx_tr->switchable_restore_cdf[1], RESTORE_SWITCHABLE_TYPES);
#else
  AVERAGE_CDF(ctx_left->switchable_restore_cdf, ctx_tr->switchable_restore_cdf,
              RESTORE_SWITCHABLE_TYPES);
#endif  // CONFIG_LOOP_RESTORE_CNN
  AVERAGE_CDF(ctx_left->wiener_restore_cdf, ctx_tr->wiener_restore_cdf, 2);
  AVERAGE_CDF(ctx_left->sgrproj_restore_cdf, ctx_tr->sgrproj_restore_cdf, 2);
#if CONFIG_WIENER_NONSEP
  AVERAGE_CDF(ctx_left->wiener_nonsep_restore_cdf,
              ctx_tr->wiener_nonsep_restore_cdf, 2);
#endif  // CONFIG_WIENER_NONSEP
#if CONFIG_RST_MERGECOEFFS
  AVERAGE_CDF(ctx_left->merged_param_cdf, ctx_tr->merged_param_cdf, 2);
#endif  // CONFIG_RST_MERGECEOFFS
#if CONFIG_DERIVED_INTRA_MODE
  AVERAGE_CDF(ctx_left->bf_is_dr_mode_cdf, ctx_tr->bf_is_dr_mode_cdf, 2);
  AVERAGE_CDF(ctx_left->bf_dr_mode_cdf, ctx_tr->bf_dr_mode_cdf,
              DIRECTIONAL_MODES);
  AVERAGE_CDF(ctx_left->bf_none_dr_mode_cdf, ctx_tr->bf_none_dr_mode_cdf,
              NONE_DIRECTIONAL_MODES);
#else
  AVERAGE_CDF(ctx_left->y_mode_cdf, ctx_tr->y_mode_cdf, INTRA_MODES);
#endif  // CONFIG_DERIVED_INTRA_MODE
#if !CONFIG_INTRA_ENTROPY
  AVG_CDF_STRIDE(ctx_left->uv_mode_cdf[0], ctx_tr->uv_mode_cdf[0],
                 UV_INTRA_MODES - 1, CDF_SIZE(UV_INTRA_MODES));
  AVERAGE_CDF(ctx_left->uv_mode_cdf[1], ctx_tr->uv_mode_cdf[1], UV_INTRA_MODES);
#if CONFIG_DERIVED_INTRA_MODE
  AVERAGE_CDF(ctx_left->kf_is_dr_mode_cdf, ctx_tr->kf_is_dr_mode_cdf, 2);
  AVERAGE_CDF(ctx_left->kf_dr_mode_cdf, ctx_tr->kf_dr_mode_cdf,
              DIRECTIONAL_MODES);
  AVERAGE_CDF(ctx_left->kf_none_dr_mode_cdf, ctx_tr->kf_none_dr_mode_cdf,
              NONE_DIRECTIONAL_MODES);
#else
  AVERAGE_CDF(ctx_left->kf_y_cdf, ctx_tr->kf_y_cdf, INTRA_MODES);
#endif  // CONFIG_DERIVED_INTRA_MODE
#endif  // !CONFIG_INTRA_ENTROPY
  for (int i = 0; i < PARTITION_CONTEXTS; i++) {
    if (i < 4) {
      AVG_CDF_STRIDE(ctx_left->partition_cdf[i], ctx_tr->partition_cdf[i], 4,
                     CDF_SIZE(10));
    } else if (i < 16) {
      AVERAGE_CDF(ctx_left->partition_cdf[i], ctx_tr->partition_cdf[i], 10);
    } else {
      AVG_CDF_STRIDE(ctx_left->partition_cdf[i], ctx_tr->partition_cdf[i], 8,
                     CDF_SIZE(10));
    }
  }
#if CONFIG_EXT_RECUR_PARTITIONS
  for (int i = 0; i < PARTITION_CONTEXTS_REC; ++i) {
    AVERAGE_CDF(ctx_left->partition_rec_cdf[i], ctx_tr->partition_rec_cdf[i],
                4);
  }
#endif  // CONFIG_EXT_RECUR_PARTITIONS
  AVERAGE_CDF(ctx_left->switchable_interp_cdf, ctx_tr->switchable_interp_cdf,
              SWITCHABLE_FILTERS);
#if CONFIG_FLEX_MVRES
  for (int p = MV_SUBPEL_QTR_PRECISION; p < MV_SUBPEL_PRECISIONS; ++p) {
    for (int j = 0; j < MV_PREC_DOWN_CONTEXTS; ++j) {
#if DISALLOW_ONE_DOWN_FLEX_MVRES == 2
      AVG_CDF_STRIDE(
          ctx_left->pb_mv_precision_cdf[j][p - MV_SUBPEL_QTR_PRECISION],
          ctx_tr->pb_mv_precision_cdf[j][p - MV_SUBPEL_QTR_PRECISION], 2,
          CDF_SIZE(2));
#elif DISALLOW_ONE_DOWN_FLEX_MVRES == 1
      AVG_CDF_STRIDE(
          ctx_left->pb_mv_precision_cdf[j][p - MV_SUBPEL_QTR_PRECISION],
          ctx_tr->pb_mv_precision_cdf[j][p - MV_SUBPEL_QTR_PRECISION], p,
          CDF_SIZE(MV_SUBPEL_PRECISIONS - 1));
#else
      AVG_CDF_STRIDE(
          ctx_left->pb_mv_precision_cdf[j][p - MV_SUBPEL_QTR_PRECISION],
          ctx_tr->pb_mv_precision_cdf[j][p - MV_SUBPEL_QTR_PRECISION], p + 1,
          CDF_SIZE(MV_SUBPEL_PRECISIONS));
#endif  // DISALLOW_ONE_DOWN_FLEX_MVRES
    }
  }
#endif  // CONFIG_FLEX_MVRES
  AVERAGE_CDF(ctx_left->angle_delta_cdf, ctx_tr->angle_delta_cdf,
              2 * MAX_ANGLE_DELTA + 1);
#if CONFIG_NEW_TX_PARTITION
  AVERAGE_CDF(ctx_left->tx_size_cdf[0], ctx_tr->tx_size_cdf[0],
              TX_PARTITION_TYPES_INTRA);
  AVERAGE_CDF(ctx_left->tx_size_cdf[1], ctx_tr->tx_size_cdf[1],
              TX_PARTITION_TYPES_INTRA);
#else
  AVG_CDF_STRIDE(ctx_left->tx_size_cdf[0], ctx_tr->tx_size_cdf[0], MAX_TX_DEPTH,
                 CDF_SIZE(MAX_TX_DEPTH + 1));
  AVERAGE_CDF(ctx_left->tx_size_cdf[1], ctx_tr->tx_size_cdf[1],
              MAX_TX_DEPTH + 1);
  AVERAGE_CDF(ctx_left->tx_size_cdf[2], ctx_tr->tx_size_cdf[2],
              MAX_TX_DEPTH + 1);
  AVERAGE_CDF(ctx_left->tx_size_cdf[3], ctx_tr->tx_size_cdf[3],
              MAX_TX_DEPTH + 1);
#endif  // CONFIG_NEW_TX_PARTITION
#if CONFIG_NN_RECON
  AVERAGE_CDF(ctx_left->use_nn_recon_cdf, ctx_tr->use_nn_recon_cdf, 2 + 1);
#endif  // CONFIG_NN_RECON
  AVERAGE_CDF(ctx_left->delta_q_cdf, ctx_tr->delta_q_cdf, DELTA_Q_PROBS + 1);
  AVERAGE_CDF(ctx_left->delta_lf_cdf, ctx_tr->delta_lf_cdf, DELTA_LF_PROBS + 1);
  for (int i = 0; i < FRAME_LF_COUNT; i++) {
    AVERAGE_CDF(ctx_left->delta_lf_multi_cdf[i], ctx_tr->delta_lf_multi_cdf[i],
                DELTA_LF_PROBS + 1);
  }
  AVG_CDF_STRIDE(ctx_left->intra_ext_tx_cdf[1], ctx_tr->intra_ext_tx_cdf[1], 7,
                 CDF_SIZE(TX_TYPES));
  AVG_CDF_STRIDE(ctx_left->intra_ext_tx_cdf[2], ctx_tr->intra_ext_tx_cdf[2], 5,
                 CDF_SIZE(TX_TYPES));
  AVG_CDF_STRIDE(ctx_left->inter_ext_tx_cdf[1], ctx_tr->inter_ext_tx_cdf[1], 16,
                 CDF_SIZE(TX_TYPES));
  AVG_CDF_STRIDE(ctx_left->inter_ext_tx_cdf[2], ctx_tr->inter_ext_tx_cdf[2], 12,
                 CDF_SIZE(TX_TYPES));
  AVG_CDF_STRIDE(ctx_left->inter_ext_tx_cdf[3], ctx_tr->inter_ext_tx_cdf[3], 2,
                 CDF_SIZE(TX_TYPES));
  AVERAGE_CDF(ctx_left->cfl_sign_cdf, ctx_tr->cfl_sign_cdf, CFL_JOINT_SIGNS);
  AVERAGE_CDF(ctx_left->cfl_alpha_cdf, ctx_tr->cfl_alpha_cdf,
              CFL_ALPHABET_SIZE);
}

static AOM_INLINE void encode_nonrd_sb(AV1_COMP *cpi, ThreadData *td,
                                       TileDataEnc *tile_data, TOKENEXTRA **tp,
                                       const int mi_row, const int mi_col,
                                       CHROMA_REF_INFO *sb_chr_ref_info) {
  AV1_COMMON *const cm = &cpi->common;
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *xd = &x->e_mbd;
  MB_MODE_INFO **mi = cm->mi_grid_base + get_mi_grid_idx(cm, mi_row, mi_col);

  const TileInfo *const tile_info = &tile_data->tile_info;
  const int num_planes = av1_num_planes(cm);
  const int ss_x = xd->plane[1].subsampling_x;
  const int ss_y = xd->plane[1].subsampling_y;
  const BLOCK_SIZE sb_size = cm->seq_params.sb_size;

  av1_setup_delta_q(cpi, td, x, tile_info, mi_row, mi_col, num_planes);
  av1_set_offsets_without_segment_id(cpi, tile_info, x, mi_row, mi_col, sb_size,
                                     sb_chr_ref_info);
  av1_choose_var_based_partitioning(cpi, tile_info, x, mi_row, mi_col);
  td->mb.cb_offset = 0;
  PC_TREE *pc_root = av1_alloc_pc_tree_node(mi_row, mi_col, sb_size, NULL,
                                            PARTITION_NONE, 0, 1, ss_x, ss_y);
  av1_reset_ptree_in_sbi(xd->sbi);
  xd->sbi->sb_mv_precision = cm->fr_mv_precision;
  av1_nonrd_use_partition(cpi, td, tile_data, mi, tp, mi_row, mi_col, sb_size,
                          pc_root, xd->sbi->ptree_root);
  av1_free_pc_tree_recursive(pc_root, num_planes, 0, 0);
}

#if !CONFIG_REALTIME_ONLY
static AOM_INLINE void encode_rd_sb(AV1_COMP *cpi, ThreadData *td,
                                    TileDataEnc *tile_data, TOKENEXTRA **tp,
                                    const int mi_row, const int mi_col,
                                    const int seg_skip,
                                    CHROMA_REF_INFO *sb_chr_ref_info) {
  AV1_COMMON *const cm = &cpi->common;
  MACROBLOCK *const x = &td->mb;
#if !(CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT)
  const SPEED_FEATURES *const sf = &cpi->sf;
#endif  // !(CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT)
  const TileInfo *const tile_info = &tile_data->tile_info;
  MB_MODE_INFO **mi = cm->mi_grid_base + get_mi_grid_idx(cm, mi_row, mi_col);
  const BLOCK_SIZE sb_size = cm->seq_params.sb_size;
  const int num_planes = av1_num_planes(cm);
  const int mib_size_log2 = cm->seq_params.mib_size_log2;
  SB_INFO *sbi = x->e_mbd.sbi;

  SIMPLE_MOTION_DATA_TREE *sms_root =
      td->sms_root[mib_size_log2 - MIN_MIB_SIZE_LOG2];
  PC_TREE *pc_root = NULL;

  int dummy_rate;
  int64_t dummy_dist;
  RD_STATS dummy_rdc;

#if CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT
  (void)seg_skip;
#endif  // CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT
#if CONFIG_EXT_RECUR_PARTITIONS
  x->sms_bufs = td->sms_bufs;
#endif  // CONFIG_EXT_RECUR_PARTITIONS

  av1_init_encode_rd_sb(cpi, td, tile_data, &pc_root, sms_root, &dummy_rdc,
                        mi_row, mi_col, 1);

#if CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT

  // TODO(yuec): incorporate the new partition syntax into rd_use_partition()
  if (0) {
#else
  if (sf->partition_search_type == FIXED_PARTITION || seg_skip) {
    av1_enc_set_offsets(cpi, tile_info, x, mi_row, mi_col, sb_size,
                        sb_chr_ref_info);
    const BLOCK_SIZE bsize = seg_skip ? sb_size : sf->always_this_block_size;
    av1_set_fixed_partitioning(cpi, tile_info, mi, mi_row, mi_col, bsize);
    av1_rd_use_partition(cpi, td, tile_data, mi, tp, mi_row, mi_col, sb_size,
                         &dummy_rate, &dummy_dist, 1, pc_root);
    av1_free_pc_tree_recursive(pc_root, num_planes, 0, 0);
  } else if (cpi->partition_search_skippable_frame) {
#endif  // CONFIG_EXT_RECUR_PARTITIONS && !KEEP_PARTITION_SPLIT
    av1_enc_set_offsets(cpi, tile_info, x, mi_row, mi_col, sb_size,
                        sb_chr_ref_info);
    const BLOCK_SIZE bsize =
        get_rd_var_based_fixed_partition(cpi, x, mi_row, mi_col);
    av1_set_fixed_partitioning(cpi, tile_info, mi, mi_row, mi_col, bsize);
    av1_rd_use_partition(cpi, td, tile_data, mi, tp, mi_row, mi_col, sb_size,
                         &dummy_rate, &dummy_dist, 1, pc_root);
    av1_free_pc_tree_recursive(pc_root, num_planes, 0, 0);
  } else {
    // Note: since rd_pick_partition searches all blocks recursively under
    // CONFIG_EXT_RECUR_PARTITIONS, we deallocate the memory inside
    // rd_pick_partition instead of outside.
#if CONFIG_COLLECT_COMPONENT_TIMING
    start_timing(cpi, rd_pick_partition_time);
#endif
    BLOCK_SIZE max_sq_size = BLOCK_INVALID, min_sq_size = BLOCK_INVALID;
    av1_get_max_min_partition_size(cpi, td, &max_sq_size, &min_sq_size, mi_row,
                                   mi_col);

    const int num_passes = cpi->oxcf.sb_multipass_unit_test ? 2 : 1;

    sbi->sb_mv_precision = cm->fr_mv_precision;
#if CONFIG_FLEX_MVRES
    MvSubpelPrecision best_prec = cm->fr_mv_precision;
    if (cm->use_sb_mv_precision) {
      SB_FIRST_PASS_STATS sb_fp_stats;
      av1_backup_sb_state(&sb_fp_stats, cpi, td, tile_data, mi_row, mi_col);

      int64_t best_rdc = INT64_MAX;
      if (!frame_is_intra_only(cm)) {
        for (MvSubpelPrecision mv_prec = MV_SUBPEL_NONE;
             mv_prec <= cm->fr_mv_precision; mv_prec++) {
          if (!pc_root) {
            av1_init_encode_rd_sb(cpi, td, tile_data, &pc_root, sms_root,
                                  &dummy_rdc, mi_row, mi_col, 0);
            av1_reset_mbmi(&cpi->common, mi_row, mi_col);
            av1_restore_sb_state(&sb_fp_stats, cpi, td, tile_data, mi_row,
                                 mi_col);
          }

          sbi->sb_mv_precision = mv_prec;

          av1_rd_pick_partition(cpi, td, tile_data, tp, mi_row, mi_col, sb_size,
                                max_sq_size, min_sq_size, &dummy_rdc, dummy_rdc,
                                pc_root, sms_root, NULL, SB_DRY_PASS);
          pc_root = NULL;
          if (dummy_rdc.rdcost < best_rdc) {
            best_rdc = dummy_rdc.rdcost;
            best_prec = mv_prec;
          }
        }
      }

      if (!pc_root) {
        av1_init_encode_rd_sb(cpi, td, tile_data, &pc_root, sms_root,
                              &dummy_rdc, mi_row, mi_col, 0);
        av1_reset_mbmi(&cpi->common, mi_row, mi_col);
        av1_restore_sb_state(&sb_fp_stats, cpi, td, tile_data, mi_row, mi_col);
      }
    }
    sbi->sb_mv_precision = best_prec;
#endif

#if CONFIG_DSPL_RESIDUAL
    sbi->allow_dspl_residual = 0;
    // For non-key frames, we try encoding each superblock first without
    // downsampling and then with downsampling (in DRY_RUN mode to avoid
    // updating contexts). Then based on RDO, we allow or disallow residual
    // downsampling for that superblock and encode the superblock again in wet
    // run mode. This RDO optimization is aimed at tackling sub-optimal
    // partition level RDO decisions as the coding of partitions in a superblock
    // is affected by the coding of their nearby partitions.
    // Note that when allow_dspl_residual == 1 then partitions within this SB
    // are *allowed* to pick the new downsampling option but are not *required*
    // to. This decision is made using partition level RDO.
    if (!frame_is_intra_only(cm)) {
      // Save state
      SB_FIRST_PASS_STATS sb_fp_stats;
      av1_backup_sb_state(&sb_fp_stats, cpi, td, tile_data, mi_row, mi_col);

      PC_TREE *pc_root_dspl[2];
      RD_STATS rd_stats_dspl[2];

      // Try dry pass with downsampling disabled
      sbi->allow_dspl_residual = 0;
      pc_root_dspl[0] = NULL;
      av1_init_encode_rd_sb(cpi, td, tile_data, &pc_root_dspl[0], sms_root,
                            &rd_stats_dspl[0], mi_row, mi_col, 0);
      av1_reset_mbmi(&cpi->common, mi_row, mi_col);
      av1_restore_sb_state(&sb_fp_stats, cpi, td, tile_data, mi_row, mi_col);
      av1_rd_pick_partition(cpi, td, tile_data, tp, mi_row, mi_col, sb_size,
                            max_sq_size, min_sq_size, &rd_stats_dspl[0],
                            rd_stats_dspl[0], pc_root_dspl[0], sms_root, NULL,
                            SB_DRY_PASS);

      // Try dry pass with downsampling enabled
      sbi->allow_dspl_residual = 1;
      pc_root_dspl[1] = NULL;
      av1_init_encode_rd_sb(cpi, td, tile_data, &pc_root_dspl[1], sms_root,
                            &rd_stats_dspl[1], mi_row, mi_col, 0);
      av1_reset_mbmi(&cpi->common, mi_row, mi_col);
      av1_restore_sb_state(&sb_fp_stats, cpi, td, tile_data, mi_row, mi_col);
      av1_rd_pick_partition(cpi, td, tile_data, tp, mi_row, mi_col, sb_size,
                            max_sq_size, min_sq_size, &rd_stats_dspl[1],
                            rd_stats_dspl[1], pc_root_dspl[1], sms_root, NULL,
                            SB_DRY_PASS);

      // Select best downsampling option
      if (rd_stats_dspl[0].rdcost <= rd_stats_dspl[1].rdcost)
        sbi->allow_dspl_residual = 0;
      else
        sbi->allow_dspl_residual = 1;

      // Restore state
      pc_root = NULL;
      av1_init_encode_rd_sb(cpi, td, tile_data, &pc_root, sms_root, &dummy_rdc,
                            mi_row, mi_col, 0);
      av1_reset_mbmi(&cpi->common, mi_row, mi_col);
      av1_restore_sb_state(&sb_fp_stats, cpi, td, tile_data, mi_row, mi_col);
    }
#endif  // CONFIG_DSPL_RESIDUAL

    if (num_passes == 1) {
      av1_rd_pick_partition(cpi, td, tile_data, tp, mi_row, mi_col, sb_size,
                            max_sq_size, min_sq_size, &dummy_rdc, dummy_rdc,
                            pc_root, sms_root, NULL, SB_WET_PASS);
      pc_root = NULL;
    } else {
      // First pass
      SB_FIRST_PASS_STATS sb_fp_stats;
      av1_backup_sb_state(&sb_fp_stats, cpi, td, tile_data, mi_row, mi_col);
      av1_rd_pick_partition(cpi, td, tile_data, tp, mi_row, mi_col, sb_size,
                            max_sq_size, min_sq_size, &dummy_rdc, dummy_rdc,
                            pc_root, sms_root, NULL, SB_DRY_PASS);

      // Second pass
      pc_root = NULL;
      av1_init_encode_rd_sb(cpi, td, tile_data, &pc_root, sms_root, &dummy_rdc,
                            mi_row, mi_col, 0);
      av1_reset_mbmi(&cpi->common, mi_row, mi_col);

      av1_restore_sb_state(&sb_fp_stats, cpi, td, tile_data, mi_row, mi_col);

      av1_rd_pick_partition(cpi, td, tile_data, tp, mi_row, mi_col, sb_size,
                            max_sq_size, min_sq_size, &dummy_rdc, dummy_rdc,
                            pc_root, sms_root, NULL, SB_WET_PASS);
    }
  }
#if CONFIG_COLLECT_COMPONENT_TIMING
  end_timing(cpi, rd_pick_partition_time);
#endif

  // TODO(angiebird): Let inter_mode_rd_model_estimation support multi-tile.
  if (cpi->sf.inter_mode_rd_model_estimation == 1 && cm->tile_cols == 1 &&
      cm->tile_rows == 1) {
    av1_inter_mode_data_fit(tile_data, x->rdmult);
  }
}
#endif  // !CONFIG_REALTIME_ONLY

static AOM_INLINE void set_cost_upd_freq(AV1_COMP *cpi, ThreadData *td,
                                         const TileInfo *const tile_info,
                                         const int mi_row, const int mi_col) {
  AV1_COMMON *const cm = &cpi->common;
  const int num_planes = av1_num_planes(cm);
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;

  switch (cpi->oxcf.coeff_cost_upd_freq) {
    case COST_UPD_TILE:  // Tile level
      if (mi_row != tile_info->mi_row_start) break;
      AOM_FALLTHROUGH_INTENDED;
    case COST_UPD_SBROW:  // SB row level in tile
      if (mi_col != tile_info->mi_col_start) break;
      AOM_FALLTHROUGH_INTENDED;
    case COST_UPD_SB:  // SB level
      av1_fill_coeff_costs(&td->mb, xd->tile_ctx, num_planes);
      break;
    default: assert(0);
  }

  switch (cpi->oxcf.mode_cost_upd_freq) {
    case COST_UPD_TILE:  // Tile level
      if (mi_row != tile_info->mi_row_start) break;
      AOM_FALLTHROUGH_INTENDED;
    case COST_UPD_SBROW:  // SB row level in tile
      if (mi_col != tile_info->mi_col_start) break;
      AOM_FALLTHROUGH_INTENDED;
    case COST_UPD_SB:  // SB level
      av1_fill_mode_rates(cm, x, xd->tile_ctx);
      break;
    default: assert(0);
  }

  switch (cpi->oxcf.mv_cost_upd_freq) {
    case COST_UPD_TILE:  // Tile level
      if (mi_row != tile_info->mi_row_start) break;
      AOM_FALLTHROUGH_INTENDED;
    case COST_UPD_SBROW:  // SB row level in tile
      if (mi_col != tile_info->mi_col_start) break;
      AOM_FALLTHROUGH_INTENDED;
    case COST_UPD_SB:  // SB level
      av1_fill_mv_costs(xd->tile_ctx, cm, x);
      break;
    default: assert(0);
  }
}

static void encode_sb_row(AV1_COMP *cpi, ThreadData *td, TileDataEnc *tile_data,
                          int mi_row, TOKENEXTRA **tp, int use_nonrd_mode) {
  AV1_COMMON *const cm = &cpi->common;
  const TileInfo *const tile_info = &tile_data->tile_info;
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  const SPEED_FEATURES *const sf = &cpi->sf;
  const int sb_cols_in_tile = av1_get_sb_cols_in_tile(cm, tile_data->tile_info);
  const BLOCK_SIZE sb_size = cm->seq_params.sb_size;
  const int mib_size = cm->seq_params.mib_size;
  const int mib_size_log2 = cm->seq_params.mib_size_log2;
  const int sb_row = (mi_row - tile_info->mi_row_start) >> mib_size_log2;

#if CONFIG_COLLECT_COMPONENT_TIMING
  start_timing(cpi, encode_sb_time);
#endif

  // Initialize the left context for the new SB row
  av1_zero_left_context(xd);

  // Reset delta for every tile
  if (mi_row == tile_info->mi_row_start || cpi->row_mt) {
    if (cm->delta_q_info.delta_q_present_flag)
      xd->current_qindex = cm->base_qindex;
    if (cm->delta_q_info.delta_lf_present_flag) {
      av1_reset_loop_filter_delta(xd, av1_num_planes(cm));
    }
  }
  reset_thresh_freq_fact(x);

  // Code each SB in the row
  for (int mi_col = tile_info->mi_col_start, sb_col_in_tile = 0;
       mi_col < tile_info->mi_col_end; mi_col += mib_size, sb_col_in_tile++) {
    (*(cpi->row_mt_sync_read_ptr))(&tile_data->row_mt_sync, sb_row,
                                   sb_col_in_tile);

    av1_reset_is_mi_coded_map(xd, cm->seq_params.mib_size);
    av1_set_sb_info(cm, xd, mi_row, mi_col);

    if (tile_data->allow_update_cdf && (cpi->row_mt == 1) &&
        (tile_info->mi_row_start != mi_row)) {
      if ((tile_info->mi_col_start == mi_col)) {
        // restore frame context of 1st column sb
        memcpy(xd->tile_ctx, x->row_ctx, sizeof(*xd->tile_ctx));
      } else {
        int wt_left = AVG_CDF_WEIGHT_LEFT;
        int wt_tr = AVG_CDF_WEIGHT_TOP_RIGHT;
        if (tile_info->mi_col_end > (mi_col + mib_size))
          avg_cdf_symbols(xd->tile_ctx, x->row_ctx + sb_col_in_tile, wt_left,
                          wt_tr);
        else
          avg_cdf_symbols(xd->tile_ctx, x->row_ctx + sb_col_in_tile - 1,
                          wt_left, wt_tr);
      }
    }

    set_cost_upd_freq(cpi, td, tile_info, mi_row, mi_col);

    xd->cur_frame_force_integer_mv = cm->cur_frame_force_integer_mv;
    td->mb.cb_coef_buff = av1_get_cb_coeff_buffer(cpi, mi_row, mi_col);
    x->source_variance = UINT_MAX;
    x->simple_motion_pred_sse = UINT_MAX;

    const struct segmentation *const seg = &cm->seg;
    int seg_skip = 0;
    if (seg->enabled) {
      const uint8_t *const map =
          seg->update_map ? cpi->segmentation_map : cm->last_frame_seg_map;
      const int segment_id =
          map ? get_segment_id(cm, map, sb_size, mi_row, mi_col) : 0;
      seg_skip = segfeature_active(seg, segment_id, SEG_LVL_SKIP);
    }

    CHROMA_REF_INFO sb_chr_ref_info = {
      1, 0, mi_row, mi_col, sb_size, sb_size
    };

    // Initialize some features for the current superblock
    x->e_mbd.sbi->sb_mv_precision = cm->fr_mv_precision;

    if (!(sf->partition_search_type == FIXED_PARTITION || seg_skip) &&
        !cpi->partition_search_skippable_frame &&
        sf->partition_search_type == VAR_BASED_PARTITION && use_nonrd_mode) {
      // Realtime non-rd path
      encode_nonrd_sb(cpi, td, tile_data, tp, mi_row, mi_col, &sb_chr_ref_info);
    } else {
#if !CONFIG_REALTIME_ONLY
      // Non-realtime rd path
      encode_rd_sb(cpi, td, tile_data, tp, mi_row, mi_col, seg_skip,
                   &sb_chr_ref_info);
#endif  // !CONFIG_REALTIME_ONLY
    }
    if (tile_data->allow_update_cdf && (cpi->row_mt == 1) &&
        (tile_info->mi_row_end > (mi_row + mib_size))) {
      if (sb_cols_in_tile == 1)
        memcpy(x->row_ctx, xd->tile_ctx, sizeof(*xd->tile_ctx));
      else if (sb_col_in_tile >= 1)
        memcpy(x->row_ctx + sb_col_in_tile - 1, xd->tile_ctx,
               sizeof(*xd->tile_ctx));
    }
    (*(cpi->row_mt_sync_write_ptr))(&tile_data->row_mt_sync, sb_row,
                                    sb_col_in_tile, sb_cols_in_tile);
  }

#if CONFIG_COLLECT_COMPONENT_TIMING
  end_timing(cpi, encode_sb_time);
#endif
}

static void init_encode_frame_mb_context(AV1_COMP *cpi) {
  AV1_COMMON *const cm = &cpi->common;
  const int num_planes = av1_num_planes(cm);
  MACROBLOCK *const x = &cpi->td.mb;
  MACROBLOCKD *const xd = &x->e_mbd;

  // Copy data over into macro block data structures.
  CHROMA_REF_INFO sb_chr_ref_info = {
    1, 0, 0, 0, cm->seq_params.sb_size, cm->seq_params.sb_size
  };
  av1_setup_src_planes(x, cpi->source, 0, 0, num_planes, &sb_chr_ref_info);

  av1_setup_block_planes(xd, cm->seq_params.subsampling_x,
                         cm->seq_params.subsampling_y, num_planes);
}

static MV_REFERENCE_FRAME get_frame_type(const AV1_COMP *cpi) {
  if (frame_is_intra_only(&cpi->common)) {
    return INTRA_FRAME;
  } else if ((cpi->rc.is_src_frame_alt_ref && cpi->refresh_golden_frame) ||
             cpi->rc.is_src_frame_internal_arf) {
    // We will not update the golden frame with an internal overlay frame
    return ALTREF_FRAME;
  } else if (cpi->refresh_golden_frame || cpi->refresh_alt2_ref_frame ||
             cpi->refresh_alt_ref_frame) {
    return GOLDEN_FRAME;
  } else {
    return LAST_FRAME;
  }
}

static TX_MODE select_tx_mode(const AV1_COMP *cpi) {
  if (cpi->common.coded_lossless) return ONLY_4X4;
  if (cpi->sf.tx_size_search_method == USE_LARGESTALL)
    return TX_MODE_LARGEST;
  else if (cpi->sf.tx_size_search_method == USE_FULL_RD ||
           cpi->sf.tx_size_search_method == USE_FAST_RD)
    return TX_MODE_SELECT;
  else
    return cpi->common.tx_mode;
}

void av1_alloc_tile_data(AV1_COMP *cpi) {
  AV1_COMMON *const cm = &cpi->common;
  const int tile_cols = cm->tile_cols;
  const int tile_rows = cm->tile_rows;

  if (cpi->tile_data != NULL) aom_free(cpi->tile_data);
  CHECK_MEM_ERROR(
      cm, cpi->tile_data,
      aom_memalign(32, tile_cols * tile_rows * sizeof(*cpi->tile_data)));
  cpi->allocated_tiles = tile_cols * tile_rows;
}

#if CONFIG_EXT_IBC_MODES
void av1_allocate_intrabc_sb(uint16_t **InputBlock, BLOCK_SIZE bsize) {
  uint16_t width = block_size_wide[bsize];
  uint16_t height = block_size_high[bsize];

  (*InputBlock) = (uint16_t *)aom_malloc(width * height * sizeof(uint16_t));
}
#endif  // CONFIG_EXT_IBC_MODES

void av1_init_tile_data(AV1_COMP *cpi) {
  AV1_COMMON *const cm = &cpi->common;
  const int num_planes = av1_num_planes(cm);
  const int tile_cols = cm->tile_cols;
  const int tile_rows = cm->tile_rows;
  int tile_col, tile_row;
  TOKENEXTRA *pre_tok = cpi->tile_tok[0][0];
  TOKENLIST *tplist = cpi->tplist[0][0];
  unsigned int tile_tok = 0;
  int tplist_count = 0;

  for (tile_row = 0; tile_row < tile_rows; ++tile_row) {
    for (tile_col = 0; tile_col < tile_cols; ++tile_col) {
      TileDataEnc *const tile_data =
          &cpi->tile_data[tile_row * tile_cols + tile_col];
      TileInfo *const tile_info = &tile_data->tile_info;
      av1_tile_init(tile_info, cm, tile_row, tile_col);

      cpi->tile_tok[tile_row][tile_col] = pre_tok + tile_tok;
      pre_tok = cpi->tile_tok[tile_row][tile_col];
      tile_tok = allocated_tokens(
          *tile_info, cm->seq_params.mib_size_log2 + MI_SIZE_LOG2, num_planes);
      cpi->tplist[tile_row][tile_col] = tplist + tplist_count;
      tplist = cpi->tplist[tile_row][tile_col];
      tplist_count = av1_get_sb_rows_in_tile(cm, tile_data->tile_info);
      tile_data->allow_update_cdf = !cm->large_scale_tile;
      tile_data->allow_update_cdf =
          tile_data->allow_update_cdf && !cm->disable_cdf_update;
      tile_data->tctx = *cm->fc;
#if CONFIG_INTRA_ENTROPY
      av1_config_entropy_models(&tile_data->tctx);
#endif  // CONFIG_INTRA_ENTROPY
    }
  }
}

void av1_encode_sb_row(AV1_COMP *cpi, ThreadData *td, int tile_row,
                       int tile_col, int mi_row) {
  AV1_COMMON *const cm = &cpi->common;
  const int num_planes = av1_num_planes(cm);
  const int tile_cols = cm->tile_cols;
  TileDataEnc *this_tile = &cpi->tile_data[tile_row * tile_cols + tile_col];
  const TileInfo *const tile_info = &this_tile->tile_info;
  TOKENEXTRA *tok = NULL;
  const int sb_row_in_tile =
      (mi_row - tile_info->mi_row_start) >> cm->seq_params.mib_size_log2;
  const int tile_mb_cols =
      (tile_info->mi_col_end - tile_info->mi_col_start + 2) >> 2;
  const int num_mb_rows_in_sb =
      ((1 << (cm->seq_params.mib_size_log2 + MI_SIZE_LOG2)) + 8) >> 4;

  get_start_tok(cpi, tile_row, tile_col, mi_row, &tok,
                cm->seq_params.mib_size_log2 + MI_SIZE_LOG2, num_planes);
  cpi->tplist[tile_row][tile_col][sb_row_in_tile].start = tok;

  encode_sb_row(cpi, td, this_tile, mi_row, &tok, cpi->sf.use_nonrd_pick_mode);

  cpi->tplist[tile_row][tile_col][sb_row_in_tile].stop = tok;
  cpi->tplist[tile_row][tile_col][sb_row_in_tile].count =
      (unsigned int)(cpi->tplist[tile_row][tile_col][sb_row_in_tile].stop -
                     cpi->tplist[tile_row][tile_col][sb_row_in_tile].start);

  assert(
      (unsigned int)(tok -
                     cpi->tplist[tile_row][tile_col][sb_row_in_tile].start) <=
      get_token_alloc(num_mb_rows_in_sb, tile_mb_cols,
                      cm->seq_params.mib_size_log2 + MI_SIZE_LOG2, num_planes));

  (void)tile_mb_cols;
  (void)num_mb_rows_in_sb;
}

void av1_encode_tile(AV1_COMP *cpi, ThreadData *td, int tile_row,
                     int tile_col) {
  AV1_COMMON *const cm = &cpi->common;
  TileDataEnc *const this_tile =
      &cpi->tile_data[tile_row * cm->tile_cols + tile_col];
  const TileInfo *const tile_info = &this_tile->tile_info;
  int mi_row;

  av1_inter_mode_data_init(this_tile);

  MACROBLOCKD *const xd = &td->mb.e_mbd;
  av1_zero_above_context(cm, xd, tile_info->mi_col_start, tile_info->mi_col_end,
                         tile_row);
  av1_init_above_context(cm, xd, tile_row);
  cfl_init(&xd->cfl, &cm->seq_params);
  av1_crc32c_calculator_init(&td->mb.mb_rd_record.crc_calculator);

#if CONFIG_REF_MV_BANK
  av1_zero(xd->ref_mv_bank_above);
  xd->ref_mv_bank_above_pt = NULL;
  av1_zero(xd->ref_mv_bank_left);
  xd->ref_mv_bank_left_pt = NULL;
#endif  // CONFIG_REF_MV_BANK

  for (mi_row = tile_info->mi_row_start; mi_row < tile_info->mi_row_end;
       mi_row += cm->seq_params.mib_size) {
    av1_encode_sb_row(cpi, td, tile_row, tile_col, mi_row);
  }
}

static void encode_tiles(AV1_COMP *cpi) {
  AV1_COMMON *const cm = &cpi->common;
  const int tile_cols = cm->tile_cols;
  const int tile_rows = cm->tile_rows;
  int tile_col, tile_row;

  if (cpi->tile_data == NULL || cpi->allocated_tiles < tile_cols * tile_rows)
    av1_alloc_tile_data(cpi);

  av1_init_tile_data(cpi);

#if CONFIG_EXT_IBC_MODES
  // Allocate 128x128 scratch buffers for Encode:
  // 1. Source
  MACROBLOCK *const x = &cpi->td.mb;
  av1_allocate_intrabc_sb(&x->ibc_src, BLOCK_128X128);
  // 2. Prediction
  MACROBLOCKD *const xd = &x->e_mbd;
  av1_allocate_intrabc_sb(&xd->ibc_pred, BLOCK_128X128);
#endif  // CONFIG_EXT_IBC_MODES

  for (tile_row = 0; tile_row < tile_rows; ++tile_row) {
    for (tile_col = 0; tile_col < tile_cols; ++tile_col) {
      TileDataEnc *const this_tile =
          &cpi->tile_data[tile_row * cm->tile_cols + tile_col];
      cpi->td.intrabc_used = 0;
      cpi->td.deltaq_used = 0;
      cpi->td.mb.e_mbd.tile_ctx = &this_tile->tctx;
      cpi->td.mb.tile_pb_ctx = &this_tile->tctx;
      av1_encode_tile(cpi, &cpi->td, tile_row, tile_col);
      cpi->intrabc_used |= cpi->td.intrabc_used;
      cpi->deltaq_used |= cpi->td.deltaq_used;
    }
  }

#if CONFIG_EXT_IBC_MODES
  // Free scratch buffers for Encode
  aom_free(x->ibc_src);
  aom_free(xd->ibc_pred);
#endif  // CONFIG_EXT_IBC_MODES
}

#define GLOBAL_TRANS_TYPES_ENC 3  // highest motion model to search
static int gm_get_params_cost(const WarpedMotionParams *gm,
                              const WarpedMotionParams *ref_gm,
                              MvSubpelPrecision precision) {
#if CONFIG_FLEX_MVRES
  const int precision_reduce = MV_SUBPEL_EIGHTH_PRECISION - precision;
#else
  const int precision_reduce =
      AOMMIN(1, MV_SUBPEL_EIGHTH_PRECISION - precision);
#endif  // CONFIG_FLEX_MVRES
  int params_cost = 0;
  int trans_bits, trans_prec_diff;
  switch (gm->wmtype) {
    case AFFINE:
    case ROTZOOM:
      params_cost += aom_count_signed_primitive_refsubexpfin(
          GM_ALPHA_MAX + 1, SUBEXPFIN_K,
          (ref_gm->wmmat[2] >> GM_ALPHA_PREC_DIFF) - (1 << GM_ALPHA_PREC_BITS),
          (gm->wmmat[2] >> GM_ALPHA_PREC_DIFF) - (1 << GM_ALPHA_PREC_BITS));
      params_cost += aom_count_signed_primitive_refsubexpfin(
          GM_ALPHA_MAX + 1, SUBEXPFIN_K,
          (ref_gm->wmmat[3] >> GM_ALPHA_PREC_DIFF),
          (gm->wmmat[3] >> GM_ALPHA_PREC_DIFF));
      if (gm->wmtype >= AFFINE) {
        params_cost += aom_count_signed_primitive_refsubexpfin(
            GM_ALPHA_MAX + 1, SUBEXPFIN_K,
            (ref_gm->wmmat[4] >> GM_ALPHA_PREC_DIFF),
            (gm->wmmat[4] >> GM_ALPHA_PREC_DIFF));
        params_cost += aom_count_signed_primitive_refsubexpfin(
            GM_ALPHA_MAX + 1, SUBEXPFIN_K,
            (ref_gm->wmmat[5] >> GM_ALPHA_PREC_DIFF) -
                (1 << GM_ALPHA_PREC_BITS),
            (gm->wmmat[5] >> GM_ALPHA_PREC_DIFF) - (1 << GM_ALPHA_PREC_BITS));
      }
      AOM_FALLTHROUGH_INTENDED;
    case TRANSLATION:
      trans_bits = (gm->wmtype == TRANSLATION)
                       ? GM_ABS_TRANS_ONLY_BITS - precision_reduce
                       : GM_ABS_TRANS_BITS;
      trans_prec_diff = (gm->wmtype == TRANSLATION)
                            ? GM_TRANS_ONLY_PREC_DIFF + precision_reduce
                            : GM_TRANS_PREC_DIFF;
      params_cost += aom_count_signed_primitive_refsubexpfin(
          (1 << trans_bits) + 1, SUBEXPFIN_K,
          (ref_gm->wmmat[0] >> trans_prec_diff),
          (gm->wmmat[0] >> trans_prec_diff));
      params_cost += aom_count_signed_primitive_refsubexpfin(
          (1 << trans_bits) + 1, SUBEXPFIN_K,
          (ref_gm->wmmat[1] >> trans_prec_diff),
          (gm->wmmat[1] >> trans_prec_diff));
      AOM_FALLTHROUGH_INTENDED;
    case IDENTITY: break;
    default: assert(0);
  }
  return (params_cost << AV1_PROB_COST_SHIFT);
}

static int do_gm_search_logic(SPEED_FEATURES *const sf, int num_refs_using_gm,
                              int frame) {
  (void)num_refs_using_gm;
  (void)frame;
  switch (sf->gm_search_type) {
    case GM_FULL_SEARCH: return 1;
    case GM_REDUCED_REF_SEARCH_SKIP_L2_L3:
      return !(frame == LAST2_FRAME || frame == LAST3_FRAME);
    case GM_REDUCED_REF_SEARCH_SKIP_L2_L3_ARF2:
      return !(frame == LAST2_FRAME || frame == LAST3_FRAME ||
               (frame == ALTREF2_FRAME));
    case GM_DISABLE_SEARCH: return 0;
    default: assert(0);
  }
  return 1;
}

// Set the relative distance of a reference frame w.r.t. current frame
static void set_rel_frame_dist(AV1_COMP *cpi) {
  const AV1_COMMON *const cm = &cpi->common;
  const SPEED_FEATURES *const sf = &cpi->sf;
  const OrderHintInfo *const order_hint_info = &cm->seq_params.order_hint_info;
  MV_REFERENCE_FRAME ref_frame;
  for (ref_frame = LAST_FRAME; ref_frame <= ALTREF_FRAME; ++ref_frame) {
    cpi->ref_relative_dist[ref_frame - LAST_FRAME] = 0;
    if (sf->alt_ref_search_fp) {
      int dist = av1_encoder_get_relative_dist(
          order_hint_info,
          cm->cur_frame->ref_display_order_hint[ref_frame - LAST_FRAME],
          cm->current_frame.display_order_hint);
      cpi->ref_relative_dist[ref_frame - LAST_FRAME] = dist;
    }
  }
}

// Enforce the number of references for each arbitrary frame based on user
// options and speed.
static void enforce_max_ref_frames(AV1_COMP *cpi) {
  MV_REFERENCE_FRAME ref_frame;
  int total_valid_refs = 0;
  for (ref_frame = LAST_FRAME; ref_frame <= ALTREF_FRAME; ++ref_frame) {
    if (cpi->ref_frame_flags & av1_ref_frame_flag_list[ref_frame]) {
      total_valid_refs++;
    }
  }

  const int max_allowed_refs = get_max_allowed_ref_frames(cpi);

  for (int i = 0; i < 4 && total_valid_refs > max_allowed_refs; ++i) {
    const MV_REFERENCE_FRAME ref_frame_to_disable = disable_order[i];

    if (!(cpi->ref_frame_flags &
          av1_ref_frame_flag_list[ref_frame_to_disable])) {
      continue;
    }

    switch (ref_frame_to_disable) {
      case LAST3_FRAME: cpi->ref_frame_flags &= ~AOM_LAST3_FLAG; break;
      case LAST2_FRAME: cpi->ref_frame_flags &= ~AOM_LAST2_FLAG; break;
      case ALTREF2_FRAME: cpi->ref_frame_flags &= ~AOM_ALT2_FLAG; break;
      case GOLDEN_FRAME: cpi->ref_frame_flags &= ~AOM_GOLD_FLAG; break;
      default: assert(0);
    }
    --total_valid_refs;
  }
  assert(total_valid_refs <= max_allowed_refs);
}

static INLINE int av1_refs_are_one_sided(const AV1_COMMON *cm) {
  assert(!frame_is_intra_only(cm));

  int one_sided_refs = 1;
  for (int ref = LAST_FRAME; ref <= ALTREF_FRAME; ++ref) {
    const RefCntBuffer *const buf = get_ref_frame_buf(cm, ref);
    if (buf == NULL) continue;

    const int ref_display_order_hint = buf->display_order_hint;
    if (av1_encoder_get_relative_dist(
            &cm->seq_params.order_hint_info, ref_display_order_hint,
            (int)cm->current_frame.display_order_hint) > 0) {
      one_sided_refs = 0;  // bwd reference
      break;
    }
  }
  return one_sided_refs;
}

static INLINE void get_skip_mode_ref_offsets(const AV1_COMMON *cm,
                                             int ref_order_hint[2]) {
  const SkipModeInfo *const skip_mode_info = &cm->current_frame.skip_mode_info;
  ref_order_hint[0] = ref_order_hint[1] = 0;
  if (!skip_mode_info->skip_mode_allowed) return;

  const RefCntBuffer *const buf_0 =
      get_ref_frame_buf(cm, LAST_FRAME + skip_mode_info->ref_frame_idx_0);
  const RefCntBuffer *const buf_1 =
      get_ref_frame_buf(cm, LAST_FRAME + skip_mode_info->ref_frame_idx_1);
  assert(buf_0 != NULL && buf_1 != NULL);

  ref_order_hint[0] = buf_0->order_hint;
  ref_order_hint[1] = buf_1->order_hint;
}

static int check_skip_mode_enabled(AV1_COMP *const cpi) {
  AV1_COMMON *const cm = &cpi->common;

  av1_setup_skip_mode_allowed(cm);
  if (!cm->current_frame.skip_mode_info.skip_mode_allowed) return 0;

  // Turn off skip mode if the temporal distances of the reference pair to the
  // current frame are different by more than 1 frame.
  const int cur_offset = (int)cm->current_frame.order_hint;
  int ref_offset[2];
  get_skip_mode_ref_offsets(cm, ref_offset);
  const int cur_to_ref0 = get_relative_dist(&cm->seq_params.order_hint_info,
                                            cur_offset, ref_offset[0]);
  const int cur_to_ref1 = abs(get_relative_dist(&cm->seq_params.order_hint_info,
                                                cur_offset, ref_offset[1]));
  if (abs(cur_to_ref0 - cur_to_ref1) > 1) return 0;

  // High Latency: Turn off skip mode if all refs are fwd.
  if (cpi->all_one_sided_refs && cpi->oxcf.lag_in_frames > 0) return 0;

  static const int flag_list[REF_FRAMES] = { 0,
                                             AOM_LAST_FLAG,
                                             AOM_LAST2_FLAG,
                                             AOM_LAST3_FLAG,
                                             AOM_GOLD_FLAG,
                                             AOM_BWD_FLAG,
                                             AOM_ALT2_FLAG,
                                             AOM_ALT_FLAG };
  const int ref_frame[2] = {
    cm->current_frame.skip_mode_info.ref_frame_idx_0 + LAST_FRAME,
    cm->current_frame.skip_mode_info.ref_frame_idx_1 + LAST_FRAME
  };
  if (!(cpi->ref_frame_flags & flag_list[ref_frame[0]]) ||
      !(cpi->ref_frame_flags & flag_list[ref_frame[1]]))
    return 0;

  return 1;
}

// Function to decide if we can skip the global motion parameter computation
// for a particular ref frame
static INLINE int skip_gm_frame(AV1_COMMON *const cm, int ref_frame) {
  if ((ref_frame == LAST3_FRAME || ref_frame == LAST2_FRAME) &&
      cm->global_motion[GOLDEN_FRAME].wmtype != IDENTITY) {
    return get_relative_dist(
               &cm->seq_params.order_hint_info,
               cm->cur_frame->ref_order_hints[ref_frame - LAST_FRAME],
               cm->cur_frame->ref_order_hints[GOLDEN_FRAME - LAST_FRAME]) <= 0;
  }
  return 0;
}

static void set_default_interp_skip_flags(AV1_COMP *cpi) {
  const int num_planes = av1_num_planes(&cpi->common);
  cpi->default_interp_skip_flags = (num_planes == 1)
                                       ? INTERP_SKIP_LUMA_EVAL_CHROMA
                                       : INTERP_SKIP_LUMA_SKIP_CHROMA;
}

static void encode_frame_internal(AV1_COMP *cpi) {
  ThreadData *const td = &cpi->td;
  MACROBLOCK *const x = &td->mb;
  AV1_COMMON *const cm = &cpi->common;
  MACROBLOCKD *const xd = &x->e_mbd;
  RD_COUNTS *const rdc = &cpi->td.rd_counts;
  int i;

  x->min_partition_size = AOMMIN(x->min_partition_size, cm->seq_params.sb_size);
  x->max_partition_size = AOMMIN(x->max_partition_size, cm->seq_params.sb_size);
#if CONFIG_DIST_8X8
  x->using_dist_8x8 = cpi->oxcf.using_dist_8x8;
  x->tune_metric = cpi->oxcf.tuning;
#endif

  if (!cpi->sf.use_nonrd_pick_mode) {
    cm->setup_mi(cm);
  }

  xd->mi = cm->mi_grid_base;
  xd->mi[0] = cm->mi;

  av1_zero(*td->counts);
  av1_zero(rdc->comp_pred_diff);

  // Reset the flag.
  cpi->intrabc_used = 0;
  // Need to disable intrabc when superres is selected
  if (av1_superres_scaled(cm)) {
    cm->allow_intrabc = 0;
  }

  cm->allow_intrabc &= (cpi->oxcf.enable_intrabc);

  if (!is_stat_generation_stage(cpi) && av1_use_hash_me(cm) &&
      !cpi->sf.use_nonrd_pick_mode) {
    // add to hash table
    const int pic_width = cpi->source->y_crop_width;
    const int pic_height = cpi->source->y_crop_height;
    uint32_t *block_hash_values[2][2];
    int8_t *is_block_same[2][3];
    int k, j;

    for (k = 0; k < 2; k++) {
      for (j = 0; j < 2; j++) {
        CHECK_MEM_ERROR(cm, block_hash_values[k][j],
                        aom_malloc(sizeof(uint32_t) * pic_width * pic_height));
      }

      for (j = 0; j < 3; j++) {
        CHECK_MEM_ERROR(cm, is_block_same[k][j],
                        aom_malloc(sizeof(int8_t) * pic_width * pic_height));
      }
    }

#if CONFIG_DEBUG
    cm->cur_frame->hash_table.has_content++;
#endif
    av1_hash_table_create(&cm->cur_frame->hash_table);
    av1_generate_block_2x2_hash_value(cpi->source, block_hash_values[0],
                                      is_block_same[0], &cpi->td.mb);
    av1_generate_block_hash_value(cpi->source, 4, block_hash_values[0],
                                  block_hash_values[1], is_block_same[0],
                                  is_block_same[1], &cpi->td.mb);
    av1_add_to_hash_map_by_row_with_precal_data(
        &cm->cur_frame->hash_table, block_hash_values[1], is_block_same[1][2],
        pic_width, pic_height, 4);
    av1_generate_block_hash_value(cpi->source, 8, block_hash_values[1],
                                  block_hash_values[0], is_block_same[1],
                                  is_block_same[0], &cpi->td.mb);
    av1_add_to_hash_map_by_row_with_precal_data(
        &cm->cur_frame->hash_table, block_hash_values[0], is_block_same[0][2],
        pic_width, pic_height, 8);
    av1_generate_block_hash_value(cpi->source, 16, block_hash_values[0],
                                  block_hash_values[1], is_block_same[0],
                                  is_block_same[1], &cpi->td.mb);
    av1_add_to_hash_map_by_row_with_precal_data(
        &cm->cur_frame->hash_table, block_hash_values[1], is_block_same[1][2],
        pic_width, pic_height, 16);
    av1_generate_block_hash_value(cpi->source, 32, block_hash_values[1],
                                  block_hash_values[0], is_block_same[1],
                                  is_block_same[0], &cpi->td.mb);
    av1_add_to_hash_map_by_row_with_precal_data(
        &cm->cur_frame->hash_table, block_hash_values[0], is_block_same[0][2],
        pic_width, pic_height, 32);
    av1_generate_block_hash_value(cpi->source, 64, block_hash_values[0],
                                  block_hash_values[1], is_block_same[0],
                                  is_block_same[1], &cpi->td.mb);
    av1_add_to_hash_map_by_row_with_precal_data(
        &cm->cur_frame->hash_table, block_hash_values[1], is_block_same[1][2],
        pic_width, pic_height, 64);

    av1_generate_block_hash_value(cpi->source, 128, block_hash_values[1],
                                  block_hash_values[0], is_block_same[1],
                                  is_block_same[0], &cpi->td.mb);
    av1_add_to_hash_map_by_row_with_precal_data(
        &cm->cur_frame->hash_table, block_hash_values[0], is_block_same[0][2],
        pic_width, pic_height, 128);

    for (k = 0; k < 2; k++) {
      for (j = 0; j < 2; j++) {
        aom_free(block_hash_values[k][j]);
      }

      for (j = 0; j < 3; j++) {
        aom_free(is_block_same[k][j]);
      }
    }
  }

  for (i = 0; i < MAX_SEGMENTS; ++i) {
    const int qindex = cm->seg.enabled
#if CONFIG_EXTQUANT
                           ? av1_get_qindex(&cm->seg, i, cm->base_qindex,
                                            cm->seq_params.bit_depth)
#else
                           ? av1_get_qindex(&cm->seg, i, cm->base_qindex)
#endif
                           : cm->base_qindex;
#if CONFIG_EXTQUANT
    xd->lossless[i] =
        qindex == 0 &&
        (cm->y_dc_delta_q - cm->seq_params.base_y_dc_delta_q <= 0) &&
        (cm->u_dc_delta_q - cm->seq_params.base_uv_dc_delta_q <= 0) &&
        cm->u_ac_delta_q <= 0 &&
        (cm->v_dc_delta_q - cm->seq_params.base_uv_dc_delta_q <= 0) &&
        cm->v_ac_delta_q <= 0;
#else
    xd->lossless[i] = qindex == 0 && cm->y_dc_delta_q == 0 &&
                      cm->u_dc_delta_q == 0 && cm->u_ac_delta_q == 0 &&
                      cm->v_dc_delta_q == 0 && cm->v_ac_delta_q == 0;
#endif  // CONFIG_EXTQUANT
    if (xd->lossless[i]) cpi->has_lossless_segment = 1;
    xd->qindex[i] = qindex;
    if (xd->lossless[i]) {
      cpi->optimize_seg_arr[i] = NO_TRELLIS_OPT;
    } else {
      cpi->optimize_seg_arr[i] = cpi->sf.optimize_coefficients;
    }
  }
  cm->coded_lossless = is_coded_lossless(cm, xd);
  cm->all_lossless = cm->coded_lossless && !av1_superres_scaled(cm);

  cm->tx_mode = select_tx_mode(cpi);

  // Fix delta q resolution for the moment
  cm->delta_q_info.delta_q_res = 0;
  if (cpi->oxcf.deltaq_mode == DELTA_Q_OBJECTIVE)
    cm->delta_q_info.delta_q_res = DEFAULT_DELTA_Q_RES_OBJECTIVE;
  else if (cpi->oxcf.deltaq_mode == DELTA_Q_PERCEPTUAL)
    cm->delta_q_info.delta_q_res = DEFAULT_DELTA_Q_RES_PERCEPTUAL;
  // Set delta_q_present_flag before it is used for the first time
  cm->delta_q_info.delta_lf_res = DEFAULT_DELTA_LF_RES;
  cm->delta_q_info.delta_q_present_flag = cpi->oxcf.deltaq_mode != NO_DELTA_Q;

  // Turn off cm->delta_q_info.delta_q_present_flag if objective delta_q is used
  // for ineligible frames. That effectively will turn off row_mt usage.
  // Note objective delta_q and tpl eligible frames are only altref frames
  // currently.
  if (cm->delta_q_info.delta_q_present_flag) {
    if (cpi->oxcf.deltaq_mode == DELTA_Q_OBJECTIVE &&
        !is_frame_tpl_eligible(cpi))
      cm->delta_q_info.delta_q_present_flag = 0;
  }

  // Reset delta_q_used flag
  cpi->deltaq_used = 0;

  cm->delta_q_info.delta_lf_present_flag =
      cm->delta_q_info.delta_q_present_flag && cpi->oxcf.deltalf_mode;
  cm->delta_q_info.delta_lf_multi = DEFAULT_DELTA_LF_MULTI;

  // update delta_q_present_flag and delta_lf_present_flag based on
  // base_qindex
  cm->delta_q_info.delta_q_present_flag &= cm->base_qindex > 0;
  cm->delta_q_info.delta_lf_present_flag &= cm->base_qindex > 0;

  av1_frame_init_quantizer(cpi);
  av1_initialize_rd_consts(cpi);
  av1_initialize_me_consts(cpi, x, cm->base_qindex);

  av1_reset_txk_skip_array(cm);

  init_encode_frame_mb_context(cpi);
  set_default_interp_skip_flags(cpi);
  if (cm->prev_frame)
    cm->last_frame_seg_map = cm->prev_frame->seg_map;
  else
    cm->last_frame_seg_map = NULL;
  if (cm->allow_intrabc || cm->coded_lossless) {
    av1_set_default_ref_deltas(cm->lf.ref_deltas);
    av1_set_default_mode_deltas(cm->lf.mode_deltas);
  } else if (cm->prev_frame) {
    memcpy(cm->lf.ref_deltas, cm->prev_frame->ref_deltas, REF_FRAMES);
    memcpy(cm->lf.mode_deltas, cm->prev_frame->mode_deltas, MAX_MODE_LF_DELTAS);
  }
  memcpy(cm->cur_frame->ref_deltas, cm->lf.ref_deltas, REF_FRAMES);
  memcpy(cm->cur_frame->mode_deltas, cm->lf.mode_deltas, MAX_MODE_LF_DELTAS);

  x->txb_split_count = 0;
#if CONFIG_SPEED_STATS
  x->tx_search_count = 0;
#endif  // CONFIG_SPEED_STATS

#if CONFIG_COLLECT_COMPONENT_TIMING
  start_timing(cpi, av1_compute_global_motion_time);
#endif
  av1_zero(rdc->global_motion_used);
  av1_zero(cpi->gmparams_cost);
#if CONFIG_FLEX_MVRES
  av1_zero(rdc->reduced_mv_precision_used);
#endif  // CONFIG_FLEX_MVRES
  if (cpi->common.current_frame.frame_type == INTER_FRAME && cpi->source &&
      cpi->oxcf.enable_global_motion && !cpi->global_motion_search_done) {
    YV12_BUFFER_CONFIG *ref_buf[REF_FRAMES];
    int frame;
    MotionModel params_by_motion[RANSAC_NUM_MOTIONS];
    for (int m = 0; m < RANSAC_NUM_MOTIONS; m++) {
      memset(&params_by_motion[m], 0, sizeof(params_by_motion[m]));
      params_by_motion[m].inliers =
          aom_malloc(sizeof(*(params_by_motion[m].inliers)) * 2 * MAX_CORNERS);
    }

    const double *params_this_motion;
    int inliers_by_motion[RANSAC_NUM_MOTIONS];
    WarpedMotionParams tmp_wm_params;
    // clang-format off
    static const double kIdentityParams[MAX_PARAMDIM - 1] = {
      0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0
    };
    // clang-format on
    int num_refs_using_gm = 0;
    int num_frm_corners = -1;
    int frm_corners[2 * MAX_CORNERS];
    unsigned char *frm_buffer = cpi->source->y_buffer;
    if (cpi->source->flags & YV12_FLAG_HIGHBITDEPTH) {
      // The frame buffer is 16-bit, so we need to convert to 8 bits for the
      // following code. We cache the result until the frame is released.
      frm_buffer =
          av1_downconvert_frame(cpi->source, cpi->common.seq_params.bit_depth);
    }
    const int segment_map_w =
        (cpi->source->y_width + WARP_ERROR_BLOCK) >> WARP_ERROR_BLOCK_LOG;
    const int segment_map_h =
        (cpi->source->y_height + WARP_ERROR_BLOCK) >> WARP_ERROR_BLOCK_LOG;

    uint8_t *segment_map =
        aom_malloc(sizeof(*segment_map) * segment_map_w * segment_map_h);
    memset(segment_map, 0,
           sizeof(*segment_map) * segment_map_w * segment_map_h);

    for (frame = ALTREF_FRAME; frame >= LAST_FRAME; --frame) {
      const MV_REFERENCE_FRAME ref_frame[2] = { frame, NONE_FRAME };
      ref_buf[frame] = NULL;
      RefCntBuffer *buf = get_ref_frame_buf(cm, frame);
      if (buf != NULL) ref_buf[frame] = &buf->buf;
      int pframe;
      cm->global_motion[frame] = default_warp_params;
      const WarpedMotionParams *ref_params =
          cm->prev_frame ? &cm->prev_frame->global_motion[frame]
                         : &default_warp_params;
      // check for duplicate buffer
      for (pframe = ALTREF_FRAME; pframe > frame; --pframe) {
        if (ref_buf[frame] == ref_buf[pframe]) break;
      }
      if (pframe > frame) {
        memcpy(&cm->global_motion[frame], &cm->global_motion[pframe],
               sizeof(WarpedMotionParams));
      } else if (ref_buf[frame] &&
                 ref_buf[frame]->y_crop_width == cpi->source->y_crop_width &&
                 ref_buf[frame]->y_crop_height == cpi->source->y_crop_height &&
                 do_gm_search_logic(&cpi->sf, num_refs_using_gm, frame) &&
                 !prune_ref_by_selective_ref_frame(
                     cpi, ref_frame, cm->cur_frame->ref_display_order_hint,
                     cm->current_frame.display_order_hint) &&
                 !(cpi->sf.selective_ref_gm && skip_gm_frame(cm, frame))) {
        if (num_frm_corners < 0) {
          // compute interest points using FAST features
          num_frm_corners = av1_fast_corner_detect(
              frm_buffer, cpi->source->y_width, cpi->source->y_height,
              cpi->source->y_stride, frm_corners, MAX_CORNERS);
        }
        TransformationType model;

        aom_clear_system_state();

        // TODO(sarahparker, debargha): Explore do_adaptive_gm_estimation = 1
        const int do_adaptive_gm_estimation = 0;

        const int ref_frame_dist = get_relative_dist(
            &cm->seq_params.order_hint_info, cm->current_frame.order_hint,
            cm->cur_frame->ref_order_hints[frame - LAST_FRAME]);
        const GlobalMotionEstimationType gm_estimation_type =
            cm->seq_params.order_hint_info.enable_order_hint &&
                    abs(ref_frame_dist) <= 2 && do_adaptive_gm_estimation
                ? GLOBAL_MOTION_DISFLOW_BASED
                : GLOBAL_MOTION_FEATURE_BASED;
        for (model = ROTZOOM; model < GLOBAL_TRANS_TYPES_ENC; ++model) {
          int64_t best_warp_error = INT64_MAX;
          // Initially set all params to identity.
          for (i = 0; i < RANSAC_NUM_MOTIONS; ++i) {
            memcpy(params_by_motion[i].params, kIdentityParams,
                   (MAX_PARAMDIM - 1) * sizeof(*(params_by_motion[i].params)));
          }

          av1_compute_global_motion(
              model, frm_buffer, cpi->source->y_width, cpi->source->y_height,
              cpi->source->y_stride, frm_corners, num_frm_corners,
              ref_buf[frame], cpi->common.seq_params.bit_depth,
              gm_estimation_type, inliers_by_motion, params_by_motion,
              RANSAC_NUM_MOTIONS);

          for (i = 0; i < RANSAC_NUM_MOTIONS; ++i) {
            if (inliers_by_motion[i] == 0) continue;

            params_this_motion = params_by_motion[i].params;
            av1_convert_model_to_params(params_this_motion, &tmp_wm_params);

            if (tmp_wm_params.wmtype != IDENTITY) {
              av1_compute_feature_segmentation_map(
                  segment_map, segment_map_w, segment_map_h,
                  params_by_motion[i].inliers, params_by_motion[i].num_inliers);

              const int64_t warp_error = av1_refine_integerized_param(
                  &tmp_wm_params, tmp_wm_params.wmtype, is_cur_buf_hbd(xd),
                  xd->bd, ref_buf[frame]->y_buffer, ref_buf[frame]->y_width,
                  ref_buf[frame]->y_height, ref_buf[frame]->y_stride,
                  cpi->source->y_buffer, cpi->source->y_width,
                  cpi->source->y_height, cpi->source->y_stride, 5,
                  best_warp_error, segment_map, segment_map_w);
              if (warp_error < best_warp_error) {
                best_warp_error = warp_error;
                // Save the wm_params modified by
                // av1_refine_integerized_param() rather than motion index to
                // avoid rerunning refine() below.
                memcpy(&(cm->global_motion[frame]), &tmp_wm_params,
                       sizeof(WarpedMotionParams));
              }
            }
          }
          if (cm->global_motion[frame].wmtype <= AFFINE)
            if (!av1_get_shear_params(&cm->global_motion[frame]))
              cm->global_motion[frame] = default_warp_params;

          if (cm->global_motion[frame].wmtype == TRANSLATION) {
            cm->global_motion[frame].wmmat[0] =
                convert_to_trans_prec(cm->fr_mv_precision,
                                      cm->global_motion[frame].wmmat[0]) *
                GM_TRANS_ONLY_DECODE_FACTOR;
            cm->global_motion[frame].wmmat[1] =
                convert_to_trans_prec(cm->fr_mv_precision,
                                      cm->global_motion[frame].wmmat[1]) *
                GM_TRANS_ONLY_DECODE_FACTOR;
          }

          if (cm->global_motion[frame].wmtype == IDENTITY) continue;

          const int64_t ref_frame_error = av1_segmented_frame_error(
              is_cur_buf_hbd(xd), xd->bd, ref_buf[frame]->y_buffer,
              ref_buf[frame]->y_stride, cpi->source->y_buffer,
              cpi->source->y_width, cpi->source->y_height,
              cpi->source->y_stride, segment_map, segment_map_w);

          if (ref_frame_error == 0) continue;

          // If the best error advantage found doesn't meet the threshold for
          // this motion type, revert to IDENTITY.
          if (!av1_is_enough_erroradvantage(
                  (double)best_warp_error / ref_frame_error,
                  gm_get_params_cost(&cm->global_motion[frame], ref_params,
                                     cm->fr_mv_precision),
                  cpi->sf.gm_erroradv_type)) {
            cm->global_motion[frame] = default_warp_params;
          }
          if (cm->global_motion[frame].wmtype != IDENTITY) break;
        }
        aom_clear_system_state();
      }
      if (cm->global_motion[frame].wmtype != IDENTITY) num_refs_using_gm++;
      cpi->gmparams_cost[frame] =
          gm_get_params_cost(&cm->global_motion[frame], ref_params,
                             cm->fr_mv_precision) +
          cpi->gmtype_cost[cm->global_motion[frame].wmtype] -
          cpi->gmtype_cost[IDENTITY];
    }
    aom_free(segment_map);
    // clear disabled ref_frames
    for (frame = LAST_FRAME; frame <= ALTREF_FRAME; ++frame) {
      const int ref_disabled =
          !(cpi->ref_frame_flags & av1_ref_frame_flag_list[frame]);
      if (ref_disabled && cpi->sf.recode_loop != DISALLOW_RECODE) {
        cpi->gmparams_cost[frame] = 0;
        cm->global_motion[frame] = default_warp_params;
      }
    }
    cpi->global_motion_search_done = 1;
    for (int m = 0; m < RANSAC_NUM_MOTIONS; m++) {
      aom_free(params_by_motion[m].inliers);
    }
  }
  memcpy(cm->cur_frame->global_motion, cm->global_motion,
         REF_FRAMES * sizeof(WarpedMotionParams));
#if CONFIG_COLLECT_COMPONENT_TIMING
  end_timing(cpi, av1_compute_global_motion_time);
#endif

#if CONFIG_COLLECT_COMPONENT_TIMING
  start_timing(cpi, av1_setup_motion_field_time);
#endif
  if (cm->allow_ref_frame_mvs) av1_setup_motion_field(cm);
#if CONFIG_COLLECT_COMPONENT_TIMING
  end_timing(cpi, av1_setup_motion_field_time);
#endif

  cpi->all_one_sided_refs =
      frame_is_intra_only(cm) ? 0 : av1_refs_are_one_sided(cm);

  cm->current_frame.skip_mode_info.skip_mode_flag =
      check_skip_mode_enabled(cpi);

  cpi->row_mt_sync_read_ptr = av1_row_mt_sync_read_dummy;
  cpi->row_mt_sync_write_ptr = av1_row_mt_sync_write_dummy;
  cpi->row_mt = 0;

  if (cpi->oxcf.row_mt && (cpi->oxcf.max_threads > 1)) {
    cpi->row_mt = 1;
    cpi->row_mt_sync_read_ptr = av1_row_mt_sync_read;
    cpi->row_mt_sync_write_ptr = av1_row_mt_sync_write;
    av1_encode_tiles_row_mt(cpi);
  } else {
    if (AOMMIN(cpi->oxcf.max_threads, cm->tile_cols * cm->tile_rows) > 1)
      av1_encode_tiles_mt(cpi);
    else
      encode_tiles(cpi);
  }

  // If intrabc is allowed but never selected, reset the allow_intrabc flag.
  if (cm->allow_intrabc && !cpi->intrabc_used) cm->allow_intrabc = 0;
  if (cm->allow_intrabc) cm->delta_q_info.delta_lf_present_flag = 0;

  if (cm->delta_q_info.delta_q_present_flag && cpi->deltaq_used == 0) {
    cm->delta_q_info.delta_q_present_flag = 0;
  }
}

#define CHECK_PRECOMPUTED_REF_FRAME_MAP 0

#if CONFIG_EXT_IBC_MODES
// static uint16_t FrameNum = -1;

/*
static void av1_setup_ibc_statistics() {

  // FrameNum++;  // Update Frame Count

  // Reset variables
  regular_ibc = ibc_rotation90 = ibc_rotation180 = ibc_rotation270 =
      ibc_mirror0 = ibc_mirror45 = ibc_mirror90 = ibc_mirror135 = 0;

  // Reset variables
  ibc_plus = ibc_blk128x = ibc_blk64x = ibc_blk32x = ibc_blk16x =
  ibc_blk8x = ibc_blk4x = 0;
}
*/
/*
static void av1_write_ibc_statistics() {

  IBCStats = fopen("IBC_Statistics.txt", "a+");
  fprintf(IBCStats, "\nFrame Number : %d\n", FrameNum);

  float totalIBCWinners = regular_ibc + ibc_rotation90 + ibc_rotation180 +
                          ibc_rotation270 + ibc_mirror0 + ibc_mirror45 +
                          ibc_mirror90 + ibc_mirror135;
  float regularIBCPercent = (float)(regular_ibc) / totalIBCWinners * 100;
  fprintf(IBCStats, "Regular IBC : %d\t Percentage Win : %f\n", regular_ibc,
          regularIBCPercent);

  fprintf(IBCStats, "Rotation Mode Winners :\n");
  float ibcR90Percent = (float)(ibc_rotation90) / totalIBCWinners * 100;
  fprintf(IBCStats, "Rotation 90 : %d\t Percentage Win : %f\n", ibc_rotation90,
          ibcR90Percent);
  float ibcR180Percent = (float)(ibc_rotation180) / totalIBCWinners * 100;
  fprintf(IBCStats, "Rotation 180 : %d\t Percentage Win : %f\n",
          ibc_rotation180, ibcR180Percent);
  float ibcR270Percent = (float)(ibc_rotation270) / totalIBCWinners * 100;
  fprintf(IBCStats, "Rotation 270 : %d\t Percentage Win : %f\n",
          ibc_rotation270, ibcR270Percent);

  fprintf(IBCStats, "Mirror Mode Winners :\n");
  float ibcM0Percent = (float)(ibc_mirror0) / totalIBCWinners * 100;
  fprintf(IBCStats, "Mirror 0 : %d\t Percentage Win : %f\n", ibc_mirror0,
          ibcM0Percent);
  float ibcM45Percent = (float)(ibc_mirror45) / totalIBCWinners * 100;
  fprintf(IBCStats, "Mirror 45 : %d\t Percentage Win : %f\n", ibc_mirror45,
          ibcM45Percent);
  float ibcM90Percent = (float)(ibc_mirror90) / totalIBCWinners * 100;
  fprintf(IBCStats, "Mirror 90 : %d\t Percentage Win : %f\n", ibc_mirror90,
          ibcM90Percent);
  float ibcM135Percent = (float)(ibc_mirror135) / totalIBCWinners * 100;
  fprintf(IBCStats, "Mirror 135 : %d\t Percentage Win : %f\n", ibc_mirror135,
          ibcM135Percent);

  float totalIBCWinners = regular_ibc + ibc_plus;

  float regularIBCPercent = (float)(regular_ibc) / totalIBCWinners * 100;
  fprintf(IBCStats, "Regular IBC : %d\t Percentage Win : %f\n", regular_ibc,
          regularIBCPercent);
  float ibcPlusPercent = (float)(ibc_plus) / totalIBCWinners * 100;
  fprintf(IBCStats, "IBC+ : %d\t Percentage Win : %f\n", ibc_plus,
          ibcPlusPercent);

  fprintf(IBCStats, " \n");
  float ibc_blk128x_Percent = (float)(ibc_blk128x) / ibc_plus * 100;
  fprintf(IBCStats,
          "IBC+ Blocks (128x128, 128x64, 64x128) : %d\t Percentage Win :
  %f\n",
          ibc_blk128x, ibc_blk128x_Percent);
  float ibc_blk64x_Percent = (float)(ibc_blk64x) / ibc_plus * 100;
  fprintf(IBCStats,
          "IBC+ Blocks (64x64, 64x32, 32x64, 64x16, 16x64) : %d\t Percentage
  "
          "Win : %f\n", ibc_blk64x, ibc_blk64x_Percent);
  float ibc_blk32x_Percent = (float)(ibc_blk32x) / ibc_plus * 100;
  fprintf(IBCStats,
          "IBC+ Blocks (32x32, 32x16, 16x32, 32x8, 8x32) : %d\t Percentage
  Win "
          ": %f\n", ibc_blk32x, ibc_blk32x_Percent);
  float ibc_blk16x_Percent = (float)(ibc_blk16x) / ibc_plus * 100;
  fprintf(IBCStats,
          "IBC+ Blocks (16x16, 16x8, 8x16, 16x4, 4x16) : %d\t Percentage Win
  "
          ": %f\n", ibc_blk16x, ibc_blk16x_Percent);
  float ibc_blk8x_Percent = (float)(ibc_blk8x) / ibc_plus * 100;
  fprintf(IBCStats,
          "IBC+ Blocks (8x8, 8x4, 4x8) : %d\t Percentage Win "
          ": %f\n", ibc_blk8x, ibc_blk8x_Percent);
  float ibc_blk4x_Percent = (float)(ibc_blk4x) / ibc_plus * 100;
  fprintf(IBCStats,
          "IBC+ Blocks (4x4) : %d\t Percentage Win "
          ": %f\n", ibc_blk4x, ibc_blk4x_Percent);

  fclose(IBCStats);
}
*/
/*
static void av1_log_ibc_statistics(MB_MODE_INFO *mbmi, BLOCK_SIZE bsize) {
  // uint8_t ibcWinMode = mbmi->ibc_mode;

  if (ibcWinMode == ROTATION_0)
    regular_ibc++;
  else if (ibcWinMode == MIRROR_90)
    ibc_mirror90++;
  else if (ibcWinMode == MIRROR_0)
    ibc_mirror0++;
  else if (ibcWinMode == ROTATION_180)
    ibc_rotation180++;
  else if (ibcWinMode == MIRROR_135)
    ibc_mirror135++;
  else if (ibcWinMode == MIRROR_45)
    ibc_mirror45++;
  else if (ibcWinMode == ROTATION_90)
    ibc_rotation90++;
  else if (ibcWinMode == ROTATION_270)
    ibc_rotation270++;

  if (ibcWinMode == ROTATION_0) {
    ++regular_ibc;
  } else {
    ++ibc_plus;
    if (bsize == BLOCK_128X128 || bsize == BLOCK_128X64 ||
        bsize == BLOCK_64X128)
      ++ibc_blk128x;
    else if (bsize == BLOCK_64X64 || bsize == BLOCK_64X32 ||
             bsize == BLOCK_32X64 || bsize == BLOCK_64X16 ||
             bsize == BLOCK_16X64)
      ++ibc_blk64x;
    else if (bsize == BLOCK_32X32 || bsize == BLOCK_32X16 ||
             bsize == BLOCK_16X32 || bsize == BLOCK_32X8 || bsize == BLOCK_8X32)
      ++ibc_blk32x;
    else if (bsize == BLOCK_16X16 || bsize == BLOCK_16X8 ||
             bsize == BLOCK_8X16 || bsize == BLOCK_16X4 || bsize == BLOCK_4X16)
      ++ibc_blk16x;
    else if (bsize == BLOCK_8X8 || bsize == BLOCK_8X4 || bsize == BLOCK_4X8)
      ++ibc_blk8x;
    else if (bsize == BLOCK_4X4)
      ++ibc_blk4x;
  }
}
*/
#endif  // CONFIG_EXT_IBC_MODES

void av1_encode_frame(AV1_COMP *cpi) {
  AV1_COMMON *const cm = &cpi->common;
  CurrentFrame *const current_frame = &cm->current_frame;
  const int num_planes = av1_num_planes(cm);
  // Indicates whether or not to use a default reduced set for ext-tx
  // rather than the potential full set of 16 transforms
  cm->reduced_tx_set_used = cpi->oxcf.reduced_tx_type_set;

  // Make sure segment_id is no larger than last_active_segid.
  if (cm->seg.enabled && cm->seg.update_map) {
    const int mi_rows = cm->mi_rows;
    const int mi_cols = cm->mi_cols;
    const int last_active_segid = cm->seg.last_active_segid;
    uint8_t *map = cpi->segmentation_map;
    for (int mi_row = 0; mi_row < mi_rows; ++mi_row) {
      for (int mi_col = 0; mi_col < mi_cols; ++mi_col) {
        map[mi_col] = AOMMIN(map[mi_col], last_active_segid);
      }
      map += mi_cols;
    }
  }

  av1_setup_frame_buf_refs(cm);
  enforce_max_ref_frames(cpi);
  set_rel_frame_dist(cpi);
  av1_setup_frame_sign_bias(cm);

#if CONFIG_EXT_IBC_MODES
  // av1_setup_ibc_statistics();
#endif  // CONFIG_EXT_IBC_MODES

#if CHECK_PRECOMPUTED_REF_FRAME_MAP
  GF_GROUP *gf_group = &cpi->gf_group;
  // TODO(yuec): The check is disabled on OVERLAY frames for now, because info
  // in cpi->gf_group has been refreshed for the next GOP when the check is
  // performed for OVERLAY frames. Since we have not support inter-GOP ref
  // frame map computation, the precomputed ref map for an OVERLAY frame is all
  // -1 at this point (although it is meaning before gf_group is refreshed).
  if (!frame_is_intra_only(cm) && gf_group->index != 0) {
    const RefCntBuffer *const golden_buf = get_ref_frame_buf(cm, GOLDEN_FRAME);

    if (golden_buf) {
      const int golden_order_hint = golden_buf->order_hint;

      for (int ref = LAST_FRAME; ref < EXTREF_FRAME; ++ref) {
        const RefCntBuffer *const buf = get_ref_frame_buf(cm, ref);
        const int ref_disp_idx_precomputed =
            gf_group->ref_frame_disp_idx[gf_group->index][ref - LAST_FRAME];

        (void)ref_disp_idx_precomputed;

        if (buf != NULL) {
          const int ref_disp_idx =
              get_relative_dist(&cm->seq_params.order_hint_info,
                                buf->order_hint, golden_order_hint);

          if (ref_disp_idx >= 0)
            assert(ref_disp_idx == ref_disp_idx_precomputed);
          else
            assert(ref_disp_idx_precomputed == -1);
        } else {
          assert(ref_disp_idx_precomputed == -1);
        }
      }
    }
  }
#endif

#if CONFIG_MISMATCH_DEBUG
  mismatch_reset_frame(num_planes);
#else
  (void)num_planes;
#endif

  if (cpi->sf.frame_parameter_update) {
    int i;
    RD_OPT *const rd_opt = &cpi->rd;
    RD_COUNTS *const rdc = &cpi->td.rd_counts;

    // This code does a single RD pass over the whole frame assuming
    // either compound, single or hybrid prediction as per whatever has
    // worked best for that type of frame in the past.
    // It also predicts whether another coding mode would have worked
    // better than this coding mode. If that is the case, it remembers
    // that for subsequent frames.
    // It does the same analysis for transform size selection also.
    //
    // TODO(zoeliu): To investigate whether a frame_type other than
    // INTRA/ALTREF/GOLDEN/LAST needs to be specified seperately.
    const MV_REFERENCE_FRAME frame_type = get_frame_type(cpi);
    int64_t *const mode_thrs = rd_opt->prediction_type_threshes[frame_type];
    const int is_alt_ref = frame_type == ALTREF_FRAME;

    /* prediction (compound, single or hybrid) mode selection */
    // NOTE: "is_alt_ref" is true only for OVERLAY/INTNL_OVERLAY frames
    int use_single_ref_only = is_alt_ref || frame_is_intra_only(cm);
#if CONFIG_MISC_CHANGES
    use_single_ref_only = use_single_ref_only || cm->only_one_ref_available;
#endif  // CONFIG_MISC_CHANGES
    if (use_single_ref_only) {
      current_frame->reference_mode = SINGLE_REFERENCE;
    } else {
      current_frame->reference_mode = REFERENCE_MODE_SELECT;
    }

    cm->interp_filter = SWITCHABLE;
    if (cm->large_scale_tile) cm->interp_filter = EIGHTTAP_REGULAR;

    cm->switchable_motion_mode = 1;

    rdc->compound_ref_used_flag = 0;
    rdc->skip_mode_used_flag = 0;

    encode_frame_internal(cpi);

    for (i = 0; i < REFERENCE_MODES; ++i)
      mode_thrs[i] = (mode_thrs[i] + rdc->comp_pred_diff[i] / cm->MBs) / 2;

    if (current_frame->reference_mode == REFERENCE_MODE_SELECT) {
      // Use a flag that includes 4x4 blocks
      if (rdc->compound_ref_used_flag == 0) {
        current_frame->reference_mode = SINGLE_REFERENCE;
#if CONFIG_ENTROPY_STATS
        av1_zero(cpi->td.counts->comp_inter);
#endif  // CONFIG_ENTROPY_STATS
      }
    }
    // Re-check on the skip mode status as reference mode may have been
    // changed.
    SkipModeInfo *const skip_mode_info = &current_frame->skip_mode_info;
    if (frame_is_intra_only(cm) ||
        current_frame->reference_mode == SINGLE_REFERENCE) {
      skip_mode_info->skip_mode_allowed = 0;
      skip_mode_info->skip_mode_flag = 0;
    }
    if (skip_mode_info->skip_mode_flag && rdc->skip_mode_used_flag == 0)
      skip_mode_info->skip_mode_flag = 0;

    if (!cm->large_scale_tile) {
      if (cm->tx_mode == TX_MODE_SELECT && cpi->td.mb.txb_split_count == 0)
        cm->tx_mode = TX_MODE_LARGEST;
    }
  } else {
    encode_frame_internal(cpi);
  }

#if CONFIG_EXT_IBC_MODES
  // av1_write_ibc_statistics();
#endif  // CONFIG_EXT_IBC_MODES
}
