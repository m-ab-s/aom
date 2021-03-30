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

#include <math.h>

#include "aom_ports/system_state.h"

#include "av1/common/av1_common_int.h"
#include "av1/common/blockd.h"

PREDICTION_MODE av1_left_block_mode(const MB_MODE_INFO *left_mi) {
  if (!left_mi) return DC_PRED;
  assert(!is_inter_block(left_mi) || is_intrabc_block(left_mi));
#if CONFIG_DERIVED_INTRA_MODE && DERIVED_INTRA_MODE_NOPD
  if (left_mi->use_derived_intra_mode[0]) return DC_PRED;
#endif  // CONFIG_DERIVED_INTRA_MODE && DERIVED_INTRA_MODE_NOPD
  return left_mi->mode;
}

PREDICTION_MODE av1_above_block_mode(const MB_MODE_INFO *above_mi) {
  if (!above_mi) return DC_PRED;
  assert(!is_inter_block(above_mi) || is_intrabc_block(above_mi));
#if CONFIG_DERIVED_INTRA_MODE && DERIVED_INTRA_MODE_NOPD
  if (above_mi->use_derived_intra_mode[0]) return DC_PRED;
#endif  // CONFIG_DERIVED_INTRA_MODE && DERIVED_INTRA_MODE_NOPD
  return above_mi->mode;
}

void av1_reset_is_mi_coded_map(MACROBLOCKD *xd, int stride) {
  av1_zero(xd->is_mi_coded);
  xd->is_mi_coded_stride = stride;
}

void av1_mark_block_as_coded(MACROBLOCKD *xd, BLOCK_SIZE bsize,
                             BLOCK_SIZE sb_size) {
  const int mi_row = xd->mi_row;
  const int mi_col = xd->mi_col;
  const int sb_mi_size = mi_size_wide[sb_size];
  const int mi_row_offset = mi_row & (sb_mi_size - 1);
  const int mi_col_offset = mi_col & (sb_mi_size - 1);

  for (int r = 0; r < mi_size_high[bsize]; ++r)
    for (int c = 0; c < mi_size_wide[bsize]; ++c) {
      const int pos =
          (mi_row_offset + r) * xd->is_mi_coded_stride + mi_col_offset + c;
      xd->is_mi_coded[pos] = 1;
    }
}

void av1_mark_block_as_not_coded(MACROBLOCKD *xd, int mi_row, int mi_col,
                                 BLOCK_SIZE bsize, BLOCK_SIZE sb_size) {
  const int sb_mi_size = mi_size_wide[sb_size];
  const int mi_row_offset = mi_row & (sb_mi_size - 1);
  const int mi_col_offset = mi_col & (sb_mi_size - 1);

  for (int r = 0; r < mi_size_high[bsize]; ++r) {
    uint8_t *row_ptr =
        &xd->is_mi_coded[(mi_row_offset + r) * xd->is_mi_coded_stride +
                         mi_col_offset];
    memset(row_ptr, 0, mi_size_wide[bsize] * sizeof(xd->is_mi_coded[0]));
  }
}

PARTITION_TREE *av1_alloc_ptree_node(PARTITION_TREE *parent, int index) {
  PARTITION_TREE *ptree = NULL;
  struct aom_internal_error_info error;

  AOM_CHECK_MEM_ERROR(&error, ptree, aom_calloc(1, sizeof(*ptree)));

  ptree->parent = parent;
  ptree->index = index;
  ptree->partition = PARTITION_NONE;
  ptree->is_settled = 0;
  for (int i = 0; i < 4; ++i) ptree->sub_tree[i] = NULL;

  return ptree;
}

void av1_free_ptree_recursive(PARTITION_TREE *ptree) {
  if (ptree == NULL) return;

  for (int i = 0; i < 4; ++i) {
    av1_free_ptree_recursive(ptree->sub_tree[i]);
    ptree->sub_tree[i] = NULL;
  }

  aom_free(ptree);
}

void av1_reset_ptree_in_sbi(SB_INFO *sbi) {
  if (sbi->ptree_root) av1_free_ptree_recursive(sbi->ptree_root);

  sbi->ptree_root = av1_alloc_ptree_node(NULL, 0);
}

void av1_set_entropy_contexts(const MACROBLOCKD *xd,
                              struct macroblockd_plane *pd, int plane,
                              BLOCK_SIZE plane_bsize, TX_SIZE tx_size,
                              int has_eob, int aoff, int loff) {
  ENTROPY_CONTEXT *const a = pd->above_entropy_context + aoff;
  ENTROPY_CONTEXT *const l = pd->left_entropy_context + loff;
  const int txs_wide = tx_size_wide_unit[tx_size];
  const int txs_high = tx_size_high_unit[tx_size];

  // above
  if (has_eob && xd->mb_to_right_edge < 0) {
    const int blocks_wide = max_block_wide(xd, plane_bsize, plane);
    const int above_contexts = AOMMIN(txs_wide, blocks_wide - aoff);
    memset(a, has_eob, sizeof(*a) * above_contexts);
    memset(a + above_contexts, 0, sizeof(*a) * (txs_wide - above_contexts));
  } else {
    memset(a, has_eob, sizeof(*a) * txs_wide);
  }

  // left
  if (has_eob && xd->mb_to_bottom_edge < 0) {
    const int blocks_high = max_block_high(xd, plane_bsize, plane);
    const int left_contexts = AOMMIN(txs_high, blocks_high - loff);
    memset(l, has_eob, sizeof(*l) * left_contexts);
    memset(l + left_contexts, 0, sizeof(*l) * (txs_high - left_contexts));
  } else {
    memset(l, has_eob, sizeof(*l) * txs_high);
  }
}

void av1_reset_entropy_context(MACROBLOCKD *xd, const int num_planes) {
  assert(xd->mi[0]->sb_type < BLOCK_SIZES_ALL);
  const int nplanes = 1 + (num_planes - 1) * xd->is_chroma_ref;
  for (int i = 0; i < nplanes; i++) {
    struct macroblockd_plane *const pd = &xd->plane[i];
    const BLOCK_SIZE plane_bsize = get_mb_plane_block_size(
        xd->mi[0], i, pd->subsampling_x, pd->subsampling_y);
    const int txs_wide = mi_size_wide[plane_bsize];
    const int txs_high = mi_size_high[plane_bsize];
    memset(pd->above_entropy_context, 0, sizeof(ENTROPY_CONTEXT) * txs_wide);
    memset(pd->left_entropy_context, 0, sizeof(ENTROPY_CONTEXT) * txs_high);
  }
}

void av1_reset_loop_filter_delta(MACROBLOCKD *xd, int num_planes) {
  xd->delta_lf_from_base = 0;
  const int frame_lf_count =
      num_planes > 1 ? FRAME_LF_COUNT : FRAME_LF_COUNT - 2;
  for (int lf_id = 0; lf_id < frame_lf_count; ++lf_id) xd->delta_lf[lf_id] = 0;
}

void av1_reset_loop_restoration(MACROBLOCKD *xd, const int num_planes) {
  for (int p = 0; p < num_planes; ++p) {
    set_default_wiener(xd->wiener_info + p);
    set_default_sgrproj(xd->sgrproj_info + p);
  }
}

void av1_setup_block_planes(MACROBLOCKD *xd, int ss_x, int ss_y,
                            const int num_planes) {
  int i;

  for (i = 0; i < num_planes; i++) {
    xd->plane[i].plane_type = get_plane_type(i);
    xd->plane[i].subsampling_x = i ? ss_x : 0;
    xd->plane[i].subsampling_y = i ? ss_y : 0;
  }
  for (i = num_planes; i < MAX_MB_PLANE; i++) {
    xd->plane[i].subsampling_x = 1;
    xd->plane[i].subsampling_y = 1;
  }
}

#if CONFIG_DERIVED_INTRA_MODE
#define BINS 36

static int get_bin_index_angle(int angle) {
  angle = AOMMAX(0, AOMMIN(angle, 179));
  return (angle + 2) / 5;
}

static void get_gradient_hist(const uint8_t *src, int src_stride, int rows,
                              int cols, int *hist) {
  float angle;
  src += src_stride;
  for (int r = 1; r < rows - 1; ++r) {
    for (int c = 1; c < cols - 1; ++c) {
      const uint8_t *above = &src[c - src_stride];
      const uint8_t *below = &src[c + src_stride];
      const uint8_t *left = &src[c - 1];
      const uint8_t *right = &src[c + 1];
      const int dx = (right[-src_stride] + 2 * right[0] + right[src_stride]) -
                     (left[-src_stride] + 2 * left[0] + left[src_stride]);
      const int dy = (below[-1] + 2 * below[0] + below[1]) -
                     (above[-1] + 2 * above[0] + above[1]);
      if (dx == 0 && dy == 0) continue;
      if (dx == 0) {
        angle = 0.0f;
      } else {
        angle = atanf(dy * 1.0f / dx);
      }
      int int_angle = 90 - (int)roundf(180 * angle / (float)PI);
      if (int_angle >= 180) int_angle = 0;
      int_angle = AOMMAX(int_angle, 0);
      const int temp = abs(dx) + abs(dy);
      const int bin_index = get_bin_index_angle(int_angle);
      hist[bin_index] += temp;
      if (bin_index > 0) hist[bin_index - 1] += temp / 2;
      if (bin_index < BINS - 1) hist[bin_index + 1] += temp / 2;
    }
    src += src_stride;
  }
}

static void get_highbd_gradient_hist(const uint8_t *src8, int src_stride,
                                     int rows, int cols, int *hist) {
  float angle;
  uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  src += src_stride;
  for (int r = 1; r < rows - 1; ++r) {
    for (int c = 1; c < cols - 1; ++c) {
      const uint16_t *above = &src[c - src_stride];
      const uint16_t *below = &src[c + src_stride];
      const uint16_t *left = &src[c - 1];
      const uint16_t *right = &src[c + 1];
      const int dx = (right[-src_stride] + 2 * right[0] + right[src_stride]) -
                     (left[-src_stride] + 2 * left[0] + left[src_stride]);
      const int dy = (below[-1] + 2 * below[0] + below[1]) -
                     (above[-1] + 2 * above[0] + above[1]);
      if (dx == 0 && dy == 0) continue;
      if (dx == 0) {
        angle = 0.0f;
      } else {
        angle = atanf(dy * 1.0f / dx);
      }
      int int_angle = 90 - (int)roundf(180 * angle / (float)PI);
      if (int_angle >= 180) int_angle = 0;
      int_angle = AOMMAX(int_angle, 0);
      const int temp = abs(dx) + abs(dy);
      const int bin_index = get_bin_index_angle(int_angle);
      hist[bin_index] += temp;
      if (bin_index > 0) hist[bin_index - 1] += temp / 2;
      if (bin_index < BINS - 1) hist[bin_index + 1] += temp / 2;
    }
    src += src_stride;
  }
}

static void generate_hog(const MACROBLOCKD *xd, int *hist) {
  const int stride = xd->plane[0].dst.stride;
  const uint8_t *buf = xd->plane[0].dst.buf;
  const int bsize = xd->mi[0]->sb_type;
  const int bh = block_size_high[bsize];
  const int bw = block_size_wide[bsize];
  const int rows =
      (xd->mb_to_bottom_edge >= 0) ? bh : ((xd->mb_to_bottom_edge >> 3) + bh);
  const int cols =
      (xd->mb_to_right_edge >= 0) ? bw : ((xd->mb_to_right_edge >> 3) + bw);
  const int lines = 3;

  if (is_cur_buf_hbd(xd)) {
    if (xd->above_mbmi) {
      if (xd->left_mbmi) {
        get_highbd_gradient_hist(buf - lines * stride - lines, stride, lines,
                                 cols + lines, hist);
      } else {
        get_highbd_gradient_hist(buf - lines * stride, stride, lines, cols,
                                 hist);
      }
    }
    if (xd->left_mbmi) {
      get_highbd_gradient_hist(buf - lines, stride, rows, lines, hist);
    }
  } else {
    if (xd->above_mbmi) {
      if (xd->left_mbmi) {
        get_gradient_hist(buf - lines * stride - lines, stride, lines,
                          cols + lines, hist);
      } else {
        get_gradient_hist(buf - lines * stride, stride, lines, cols, hist);
      }
    }
    if (xd->left_mbmi) {
      get_gradient_hist(buf - lines, stride, rows, lines, hist);
    }
  }
}

static int derive_intra_mode_from_hog(const MACROBLOCKD *xd) {
  aom_clear_system_state();

  int hist[BINS] = { 0 };
  generate_hog(xd, hist);

  int max_score = 0;
  int best_idx = 0;
  for (int i = 0; i < BINS; ++i) {
    const int this_score = hist[i];
    if (this_score > max_score) {
      max_score = this_score;
      best_idx = i;
    }
  }

  aom_clear_system_state();

  return best_idx;
}

int av1_enable_derived_intra_mode(const MACROBLOCKD *xd, int bsize) {
  return bsize >= BLOCK_8X8 && xd->mb_to_bottom_edge > 0 &&
         xd->mb_to_right_edge > 0 && (xd->above_mbmi || xd->left_mbmi);
}
#undef BINS

static int get_angle_from_index(int index) { return index * 5; }

static int angle_to_mode(int angle) {
  if (angle < 56) return D45_PRED;
  if (angle < 79) return D67_PRED;
  if (angle < 102) return V_PRED;
  if (angle < 124) return D113_PRED;
  if (angle < 146) return D135_PRED;
  if (angle < 169) return D157_PRED;
  if (angle < 192) return H_PRED;
  return D203_PRED;
}

int av1_get_derived_intra_mode(const MACROBLOCKD *xd, int bsize,
                               int *derived_angle) {
  if (av1_enable_derived_intra_mode(xd, bsize)) {
    const int idx = derive_intra_mode_from_hog(xd);
    int angle = get_angle_from_index(idx);
    if (angle < 36) angle += 180;
    *derived_angle = angle;
    const int mode = angle_to_mode(angle);
    return mode;
  }
  return INTRA_MODES;
}
#endif  // CONFIG_DERIVED_INTRA_MODE
