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

#include "av1/common/blockd.h"
#include "av1/common/onyxc_int.h"

PREDICTION_MODE av1_left_block_mode(const MB_MODE_INFO *left_mi) {
  if (!left_mi) return DC_PRED;
#if CONFIG_DERIVED_INTRA_MODE
  if (left_mi->use_derived_intra_mode[0]) return DC_PRED;
#endif  // CONFIG_DERIVED_INTRA_MODE
  assert(!is_inter_block(left_mi) || is_intrabc_block(left_mi));
  return left_mi->mode;
}

PREDICTION_MODE av1_above_block_mode(const MB_MODE_INFO *above_mi) {
  if (!above_mi) return DC_PRED;
#if CONFIG_DERIVED_INTRA_MODE
  if (above_mi->use_derived_intra_mode[0]) return DC_PRED;
#endif  // CONFIG_DERIVED_INTRA_MODE
  assert(!is_inter_block(above_mi) || is_intrabc_block(above_mi));
  return above_mi->mode;
}

void av1_reset_is_mi_coded_map(MACROBLOCKD *xd, int stride) {
  av1_zero(xd->is_mi_coded);
  xd->is_mi_coded_stride = stride;
}

void av1_mark_block_as_coded(MACROBLOCKD *xd, int mi_row, int mi_col,
                             BLOCK_SIZE bsize, BLOCK_SIZE sb_size) {
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

  for (int i = 0; i < 4; ++i) av1_free_ptree_recursive(ptree->sub_tree[i]);

  aom_free(ptree);
}

void av1_reset_ptree_in_sbi(SB_INFO *sbi) {
  if (sbi->ptree_root) av1_free_ptree_recursive(sbi->ptree_root);

  sbi->ptree_root = av1_alloc_ptree_node(NULL, 0);
}

#if CONFIG_INTRA_ENTROPY
#if !CONFIG_USE_SMALL_MODEL
INLINE static void add_hist_features(const uint64_t *hist, float **features) {
  float total = 0.0f;
  if (hist) {
    for (int i = 0; i < 8; ++i) {
      total += (float)hist[i];
    }
  }

  (*features)[0] = logf(total + 1.0f);
  ++*features;

  float *feature_pt = *features;
  if (total > 0.1f) {
    for (int i = 0; i < 8; ++i) {
      feature_pt[i] = (float)hist[i] / total;
    }
  } else {
    for (int i = 0; i < 8; ++i) {
      feature_pt[i] = 0.125f;
    }
  }
  *features += 8;
}

static void add_onehot(float **features, int num, int n) {
  float *feature_pt = *features;
  memset(feature_pt, 0, sizeof(*feature_pt) * num);
  if (n >= 0 && n < num) feature_pt[n] = 1.0f;
  *features += num;
}

#endif  // !CONFIG_USE_SMALL_MODEL

void av1_get_intra_block_feature(int *sparse_features, float *dense_features,
                                 const MB_MODE_INFO *above_mi,
                                 const MB_MODE_INFO *left_mi,
                                 const MB_MODE_INFO *aboveleft_mi) {
  const MB_MODE_INFO *mbmi_list[3] = { above_mi, left_mi, aboveleft_mi };
  const int num_mbmi = CONFIG_USE_SMALL_MODEL ? 2 : 3;
  for (int i = 0; i < num_mbmi; ++i) {
    const MB_MODE_INFO *mbmi = mbmi_list[i];
    const int data_available = mbmi != NULL;
#if CONFIG_USE_SMALL_MODEL
    *(sparse_features++) = data_available ? mbmi->mode : INTRA_MODES;
#else
    *(dense_features++) = data_available;
    add_onehot(&dense_features, INTRA_MODES, data_available ? mbmi->mode : -1);
    const float var = data_available ? (float)mbmi->y_recon_var : 0.0f;
    *(dense_features++) = logf(var + 1.0f);
    add_hist_features(data_available ? mbmi->y_gradient_hist : NULL,
                      &dense_features);
#endif
  }
#if CONFIG_USE_SMALL_MODEL
  (void)dense_features;
#else
  (void)sparse_features;
#endif
}

void av1_pdf2icdf(float *pdf, aom_cdf_prob *cdf, int nsymbs) {
  float accu = 0.0f;  // AOM_ICDF
  for (int i = 0; i < nsymbs - 1; ++i) {
    accu += pdf[i];
    cdf[i] = AOM_ICDF((aom_cdf_prob)(accu * CDF_PROB_TOP));
  }
  cdf[nsymbs - 1] = 0;
}

void av1_get_intra_uv_block_feature(int *sparse_features, float *features,
                                    PREDICTION_MODE cur_y_mode,
                                    int is_cfl_allowed,
                                    const MB_MODE_INFO *above_mi,
                                    const MB_MODE_INFO *left_mi) {
  *(sparse_features++) = cur_y_mode;
  *(sparse_features++) = !is_cfl_allowed;
  (void)features;
  (void)above_mi;
  (void)left_mi;
}

void av1_get_kf_y_mode_cdf_ml(const MACROBLOCKD *xd, aom_cdf_prob *cdf) {
  NN_CONFIG_EM *nn_model = &(xd->tile_ctx->intra_y_mode);
  av1_get_intra_block_feature(nn_model->sparse_features,
                              nn_model->dense_features, xd->above_mbmi,
                              xd->left_mbmi, xd->aboveleft_mbmi);
  av1_nn_predict_em(nn_model);
  av1_pdf2icdf(nn_model->output, cdf, INTRA_MODES);
}

void av1_get_uv_mode_cdf_ml(const MACROBLOCKD *xd, PREDICTION_MODE y_mode,
                            aom_cdf_prob *cdf) {
  NN_CONFIG_EM *nn_model = &(xd->tile_ctx->intra_uv_mode);
  av1_get_intra_uv_block_feature(
      nn_model->sparse_features, nn_model->dense_features, y_mode,
      is_cfl_allowed(xd), xd->above_mbmi, xd->left_mbmi);
  av1_nn_predict_em(nn_model);
  av1_pdf2icdf(nn_model->output, cdf, UV_INTRA_MODES);
}
#endif  // CONFIG_INTRA_ENTROPY

void av1_set_contexts(const MACROBLOCKD *xd, struct macroblockd_plane *pd,
                      int plane, BLOCK_SIZE plane_bsize, TX_SIZE tx_size,
                      int has_eob, int aoff, int loff) {
  ENTROPY_CONTEXT *const a = pd->above_context + aoff;
  ENTROPY_CONTEXT *const l = pd->left_context + loff;
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
void av1_reset_skip_context(MACROBLOCKD *xd, BLOCK_SIZE bsize,
                            const int num_planes) {
  int i;
  int nplanes;
  assert(bsize < BLOCK_SIZES_ALL);

  nplanes = 1 + (num_planes - 1) * xd->mi[0]->chroma_ref_info.is_chroma_ref;
  for (i = 0; i < nplanes; i++) {
    struct macroblockd_plane *const pd = &xd->plane[i];
    const BLOCK_SIZE bsize_base =
        i ? xd->mi[0]->chroma_ref_info.bsize_base : bsize;
    const BLOCK_SIZE plane_bsize =
        get_plane_block_size(bsize_base, pd->subsampling_x, pd->subsampling_y);
    assert(plane_bsize < BLOCK_SIZES_ALL);
    const int txs_wide = block_size_wide[plane_bsize] >> tx_size_wide_log2[0];
    const int txs_high = block_size_high[plane_bsize] >> tx_size_high_log2[0];
    memset(pd->above_context, 0, sizeof(ENTROPY_CONTEXT) * txs_wide);
    memset(pd->left_context, 0, sizeof(ENTROPY_CONTEXT) * txs_high);
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
#if CONFIG_WIENER_NONSEP
    set_default_wiener_nonsep(xd->wiener_nonsep_info + p);
#endif  // CONFIG_WIENER_NONSEP
#if CONFIG_CNN_CRLC_GUIDED
    set_default_crlc(xd->crlc_unitinfo + p);
#endif  // CONFIG_CNN_CRLC_GUIDED
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

void av1_get_unit_width_height_coeff(const MACROBLOCKD *const xd, int plane,
                                     BLOCK_SIZE plane_bsize, int row_plane,
                                     int col_plane, int *unit_width,
                                     int *unit_height) {
  const int is_inter = is_inter_block(xd->mi[0]);
  const int max_blocks_wide_plane =
      is_inter ? block_size_wide[plane_bsize] >> tx_size_wide_log2[0]
               : max_block_wide(xd, plane_bsize, plane);
  const int max_blocks_high_plane =
      is_inter ? block_size_high[plane_bsize] >> tx_size_high_log2[0]
               : max_block_high(xd, plane_bsize, plane);

  const struct macroblockd_plane *const pd = &xd->plane[plane];
  const BLOCK_SIZE max_unit_bsize_plane =
      get_plane_block_size(BLOCK_64X64, pd->subsampling_x, pd->subsampling_y);
  int mu_blocks_wide_plane =
      block_size_wide[max_unit_bsize_plane] >> tx_size_wide_log2[0];
  int mu_blocks_high_plane =
      block_size_high[max_unit_bsize_plane] >> tx_size_high_log2[0];
  mu_blocks_wide_plane = AOMMIN(max_blocks_wide_plane, mu_blocks_wide_plane);
  mu_blocks_high_plane = AOMMIN(max_blocks_high_plane, mu_blocks_high_plane);
  assert(mu_blocks_wide_plane > 0);
  assert(mu_blocks_high_plane > 0);

  *unit_height =
      AOMMIN(mu_blocks_high_plane + row_plane, max_blocks_high_plane);
  *unit_width = AOMMIN(mu_blocks_wide_plane + col_plane, max_blocks_wide_plane);
  assert(*unit_height > 0);
  assert(*unit_width > 0);
}

#if CONFIG_INTRA_ENTROPY && !CONFIG_USE_SMALL_MODEL
static const int16_t cos_angle[8] = {
  // 45 degrees
  45,
  // 22.5 degrees
  59,
  // 0 degrees
  64,
  // -22.5 degrees,
  59,
  // -45 degrees
  45,
  // -67.5 degrees
  24,
  // -90 degrees
  0,
  // 67.5 degrees
  24,
};

static const int16_t sin_angle[8] = {
  // 45 degrees
  45,
  // 22.5 degrees
  24,
  // 0 degrees
  0,
  // -22.5 degrees,
  -24,
  // -45 degrees
  -45,
  // -67.5 degrees
  -59,
  // -90 degrees
  64,
  // 67.5 degrees
  59,
};

static INLINE uint8_t get_angle_idx(int dx, int dy) {
  int max_response = 0;
  int max_angle = 0;
  for (int angle_idx = 0; angle_idx < 8; angle_idx++) {
    const int cos = cos_angle[angle_idx];
    const int sin = sin_angle[angle_idx];
    const int64_t dot_prod = cos * dx + sin * dy;
    const int64_t response = labs(dot_prod);
    if (response > max_response) {
      max_response = response;
      max_angle = angle_idx;
    }
  }

  return max_angle;
}

void av1_get_gradient_hist_lbd_c(const uint8_t *dst, int stride, int rows,
                                 int cols, uint64_t *hist) {
  dst += stride;
  for (int r = 1; r < rows; ++r) {
    int c;
    for (c = 1; c < cols; c += 1) {
      const int dx = dst[c] - dst[c - 1];
      const int dy = dst[c] - dst[c - stride];
      const int mag = dx * dx + dy * dy;
      const uint8_t index = get_angle_idx(dx, dy);
      hist[index] += mag;
    }
    dst += stride;
  }
}

static void get_highbd_gradient_hist(const uint8_t *dst8, int stride, int rows,
                                     int cols, uint64_t *hist) {
  uint16_t *dst = CONVERT_TO_SHORTPTR(dst8);
  dst += stride;
  for (int r = 1; r < rows; ++r) {
    for (int c = 1; c < cols; ++c) {
      const int dx = dst[c] - dst[c - 1];
      const int dy = dst[c] - dst[c - stride];
      const int mag = dx * dx + dy * dy;
      const uint8_t index = get_angle_idx(dx, dy);
      hist[index] += mag;
    }
    dst += stride;
  }
}

void av1_get_gradient_hist(const MACROBLOCKD *const xd,
                           MB_MODE_INFO *const mbmi, BLOCK_SIZE bsize) {
  const int y_stride = xd->plane[0].dst.stride;
  const uint8_t *y_dst = xd->plane[0].dst.buf;
  const int rows = block_size_high[bsize];
  const int cols = block_size_wide[bsize];
  const int y_block_rows =
      xd->mb_to_bottom_edge >= 0 ? rows : (xd->mb_to_bottom_edge >> 3) + rows;
  const int y_block_cols =
      xd->mb_to_right_edge >= 0 ? cols : (xd->mb_to_right_edge >> 3) + cols;

  av1_zero(mbmi->y_gradient_hist);
  if (is_cur_buf_hbd(xd)) {
    get_highbd_gradient_hist(y_dst, y_stride, y_block_rows, y_block_cols,
                             mbmi->y_gradient_hist);
  } else {
    av1_get_gradient_hist_lbd(y_dst, y_stride, y_block_rows, y_block_cols,
                              mbmi->y_gradient_hist);
  }
}

static int64_t variance(const uint8_t *dst, int stride, int w, int h) {
  int64_t sum = 0;
  int64_t sum_square = 0;
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int v = dst[j];
      sum += v;
      sum_square += v * v;
    }
    dst += stride;
  }
  const int n = w * h;
  const int64_t var = (n * sum_square - sum * sum) / n / n;
  return var < 0 ? 0 : var;
}

static int64_t highbd_variance(const uint8_t *dst8, int stride, int w, int h) {
  const uint16_t *dst = CONVERT_TO_SHORTPTR(dst8);
  int64_t sum = 0;
  int64_t sum_square = 0;
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int v = dst[j];
      sum += v;
      sum_square += v * v;
    }
    dst += stride;
  }
  const int n = w * h;
  const int64_t var = (n * sum_square - sum * sum) / n / n;
  return var < 0 ? 0 : var;
}

void av1_get_recon_var(const MACROBLOCKD *const xd, MB_MODE_INFO *const mbmi,
                       BLOCK_SIZE bsize) {
  const int y_stride = xd->plane[0].dst.stride;
  const uint8_t *y_dst = xd->plane[0].dst.buf;
  const int rows = block_size_high[bsize];
  const int cols = block_size_wide[bsize];
  const int y_block_rows =
      xd->mb_to_bottom_edge >= 0 ? rows : (xd->mb_to_bottom_edge >> 3) + rows;
  const int y_block_cols =
      xd->mb_to_right_edge >= 0 ? cols : (xd->mb_to_right_edge >> 3) + cols;

  if (is_cur_buf_hbd(xd)) {
    mbmi->y_recon_var =
        highbd_variance(y_dst, y_stride, y_block_cols, y_block_rows);
  } else {
    mbmi->y_recon_var = variance(y_dst, y_stride, y_block_cols, y_block_rows);
  }
}
#endif  // CONFIG_INTRA_ENTROPY && !CONFIG_USE_SMALL_MODEL

#if CONFIG_DERIVED_INTRA_MODE
// BIN_WIDTH * BINS should be equal to 180.
#define BINS 36
#define BIN_WIDTH 5
static int get_bin_index_from_angle(int angle) {
  angle = AOMMAX(0, AOMMIN(angle, 179));
  return angle / BIN_WIDTH;
}

static INLINE int get_angle_from_index(int index) {
  return index * BIN_WIDTH + (BIN_WIDTH >> 1);
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
      const int bin_index = get_bin_index_from_angle(int_angle);
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
      const int bin_index = get_bin_index_from_angle(int_angle);
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
      (xd->mb_to_bottom_edge >= 0) ? bh : (xd->mb_to_bottom_edge >> 3) + bh;
  const int cols =
      (xd->mb_to_right_edge >= 0) ? bw : (xd->mb_to_right_edge >> 3) + bw;
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

int av1_enable_derived_intra_mode(const MACROBLOCKD *xd, int bsize) {
  return bsize >= BLOCK_8X8 && (xd->above_mbmi || xd->left_mbmi);
}

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
                               MB_MODE_INFO *mbmi) {
  if (av1_enable_derived_intra_mode(xd, bsize)) {
    int hist[BINS] = { 0 };
    aom_clear_system_state();
    generate_hog(xd, hist);
    aom_clear_system_state();
#if FUSION_MODE
    int total_weight = 1;
    for (int i = 0; i < NUM_DERIVED_INTRA_MODES; ++i) {
      int max_score = 0;
      int best_idx = 0;
      for (int idx = 0; idx < BINS; ++idx) {
        const int this_score = hist[idx];
        if (this_score > max_score) {
          max_score = this_score;
          best_idx = idx;
        }
      }
      int angle = get_angle_from_index(best_idx);
      if (angle < 36) angle += 180;
      mbmi->derived_intra_angles[i] = angle;
      mbmi->derived_intra_weights[i] = max_score;
      total_weight += max_score;
      hist[best_idx] = 0;
    }

    const int scale = 1 << DERIVED_INTRA_FUSION_SHIFT;
    int sub_total_weight = 0;
    for (int i = NUM_DERIVED_INTRA_MODES - 1; i > 0; --i) {
      const int weight = mbmi->derived_intra_weights[i];
      mbmi->derived_intra_weights[i] =
          (weight * scale + (total_weight >> 1)) / total_weight;
      sub_total_weight += mbmi->derived_intra_weights[i];
    }
    mbmi->derived_intra_weights[0] = scale - sub_total_weight;
    mbmi->derived_angle = mbmi->derived_intra_angles[0];
    return angle_to_mode(mbmi->derived_angle);
#else
    int max_score = 0;
    int best_idx = 0;
    for (int i = 0; i < BINS; ++i) {
      const int this_score = hist[i];
      if (this_score > max_score) {
        max_score = this_score;
        best_idx = i;
      }
    }
    int angle = get_angle_from_index(best_idx);
    if (angle < 36) angle += 180;
    mbmi->derived_angle = angle;
    const int mode = angle_to_mode(angle);
    return mode;
#endif
  }
  return INTRA_MODES;
}

#undef BINS
#undef BIN_WIDTH
#endif  // CONFIG_DERIVED_INTRA_MODE

void av1_alloc_txk_skip_array(AV1_COMMON *cm) {
  // allocate based on the MIN_TX_SIZE, which is 4x4 block
  for (int plane = 0; plane < MAX_MB_PLANE; plane++) {
    int w = cm->mi_cols << MI_SIZE_LOG2;
    int h = cm->mi_rows << MI_SIZE_LOG2;
    w = ((w + MAX_SB_SIZE - 1) >> MAX_SB_SIZE_LOG2) << MAX_SB_SIZE_LOG2;
    h = ((h + MAX_SB_SIZE - 1) >> MAX_SB_SIZE_LOG2) << MAX_SB_SIZE_LOG2;
    w >>= ((plane == 0) ? 0 : cm->seq_params.subsampling_x);
    h >>= ((plane == 0) ? 0 : cm->seq_params.subsampling_y);
    int stride = (w + MIN_TX_SIZE - 1) >> MIN_TX_SIZE_LOG2;
    int rows = (h + MIN_TX_SIZE - 1) >> MIN_TX_SIZE_LOG2;
    cm->tx_skip[plane] = aom_calloc(rows * stride, sizeof(uint8_t));
    cm->tx_skip_buf_size[plane] = rows * stride;
  }
}

void av1_dealloc_txk_skip_array(AV1_COMMON *cm) {
  for (int plane = 0; plane < MAX_MB_PLANE; plane++) {
    aom_free(cm->tx_skip[plane]);
    cm->tx_skip[plane] = NULL;
  }
}

void av1_reset_txk_skip_array(AV1_COMMON *cm) {
  // allocate based on the MIN_TX_SIZE, which is 4x4 block
  for (int plane = 0; plane < MAX_MB_PLANE; plane++) {
    int w = cm->mi_cols << MI_SIZE_LOG2;
    int h = cm->mi_rows << MI_SIZE_LOG2;
    w = ((w + MAX_SB_SIZE - 1) >> MAX_SB_SIZE_LOG2) << MAX_SB_SIZE_LOG2;
    h = ((h + MAX_SB_SIZE - 1) >> MAX_SB_SIZE_LOG2) << MAX_SB_SIZE_LOG2;
    w >>= ((plane == 0) ? 0 : cm->seq_params.subsampling_x);
    h >>= ((plane == 0) ? 0 : cm->seq_params.subsampling_y);
    int stride = (w + MIN_TX_SIZE - 1) >> MIN_TX_SIZE_LOG2;
    int rows = (h + MIN_TX_SIZE - 1) >> MIN_TX_SIZE_LOG2;
    memset(cm->tx_skip[plane], 0, rows * stride);
  }
}

void av1_init_txk_skip_array(const AV1_COMMON *cm, MB_MODE_INFO *mbmi,
                             int mi_row, int mi_col, BLOCK_SIZE bsize,
                             uint8_t value, FILE *fLog) {
  for (int plane = 0; plane < MAX_MB_PLANE; plane++) {
    const int is_chroma_ref = plane && mbmi->chroma_ref_info.is_chroma_ref;
    int mi_row_plane =
        is_chroma_ref ? mbmi->chroma_ref_info.mi_row_chroma_base : mi_row;
    int mi_col_plane =
        is_chroma_ref ? mbmi->chroma_ref_info.mi_col_chroma_base : mi_col;
    BLOCK_SIZE bsize_plane =
        is_chroma_ref ? mbmi->chroma_ref_info.bsize_base : bsize;
    int w = ((cm->width + MAX_SB_SIZE - 1) >> MAX_SB_SIZE_LOG2)
            << MAX_SB_SIZE_LOG2;
    w >>= ((plane == 0) ? 0 : cm->seq_params.subsampling_x);
    int stride = (w + MIN_TX_SIZE - 1) >> MIN_TX_SIZE_LOG2;
    int x = (mi_col_plane << MI_SIZE_LOG2) >>
            ((plane == 0) ? 0 : cm->seq_params.subsampling_x);
    int y = (mi_row_plane << MI_SIZE_LOG2) >>
            ((plane == 0) ? 0 : cm->seq_params.subsampling_y);
    int row = y >> MIN_TX_SIZE_LOG2;
    int col = x >> MIN_TX_SIZE_LOG2;
    int blk_w = block_size_wide[bsize_plane] >>
                ((plane == 0) ? 0 : cm->seq_params.subsampling_x);
    int blk_h = block_size_high[bsize_plane] >>
                ((plane == 0) ? 0 : cm->seq_params.subsampling_y);
    blk_w >>= MIN_TX_SIZE_LOG2;
    blk_h >>= MIN_TX_SIZE_LOG2;

    for (int r = 0; r < blk_h; r++) {
      for (int c = 0; c < blk_w; c++) {
        uint32_t idx = (row + r) * stride + col + c;
        assert(idx < cm->tx_skip_buf_size[plane]);
        cm->tx_skip[plane][idx] = value;
      }
    }
  }

  if (fLog) {
    int row = (mi_row << MI_SIZE_LOG2);
    int col = (mi_col << MI_SIZE_LOG2);
    int w = block_size_wide[bsize];
    int h = block_size_high[bsize];
    if (value != 0) {
      fprintf(fLog,
              "\n\tSkipped TxBlock: row = %d, col = %d, blk_width = %d, "
              "blk_height = %d",
              row, col, w, h);
    } else {
      fprintf(
          fLog,
          "\nrow = %d, col = %d, width = %d, height = %d, %s, blk skipped = %d",
          row, col, w, h, is_inter_block(mbmi) ? "INTER" : "INTRA", value);
    }
  }
}

void av1_update_txk_skip_array(const AV1_COMMON *cm, int mi_row, int mi_col,
                               int plane, int blk_row, int blk_col,
                               TX_SIZE tx_size, FILE *fLog) {
  blk_row *= 4;
  blk_col *= 4;
  int w = ((cm->width + MAX_SB_SIZE - 1) >> MAX_SB_SIZE_LOG2)
          << MAX_SB_SIZE_LOG2;
  w >>= ((plane == 0) ? 0 : cm->seq_params.subsampling_x);
  int stride = (w + MIN_TX_SIZE - 1) >> MIN_TX_SIZE_LOG2;
  int tx_w = tx_size_wide[tx_size];
  int tx_h = tx_size_high[tx_size];
  int cols = tx_w >> MIN_TX_SIZE_LOG2;
  int rows = tx_h >> MIN_TX_SIZE_LOG2;
  int x = (mi_col << MI_SIZE_LOG2) >>
          ((plane == 0) ? 0 : cm->seq_params.subsampling_x);
  int y = (mi_row << MI_SIZE_LOG2) >>
          ((plane == 0) ? 0 : cm->seq_params.subsampling_y);
  x = (x + blk_col) >> MIN_TX_SIZE_LOG2;
  y = (y + blk_row) >> MIN_TX_SIZE_LOG2;
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      uint32_t idx = (y + r) * stride + x + c;
      assert(idx < cm->tx_skip_buf_size[plane]);
      cm->tx_skip[plane][idx] = 1;
    }
  }
  if (fLog) {
    fprintf(fLog,
            "\n\tSkipped TxBlock: row = %d, col = %d, tx_width = %d, tx_height"
            "= %d, plane = %d",
            ((mi_row << MI_SIZE_LOG2) + blk_row),
            ((mi_col << MI_SIZE_LOG2) + blk_col), tx_size_wide[tx_size],
            tx_size_high[tx_size], plane);
  }
}

uint8_t av1_get_txk_skip(const AV1_COMMON *cm, int mi_row, int mi_col,
                         int plane, int blk_row, int blk_col) {
  blk_row *= 4;
  blk_col *= 4;
  int w = ((cm->width + MAX_SB_SIZE - 1) >> MAX_SB_SIZE_LOG2)
          << MAX_SB_SIZE_LOG2;
  w >>= ((plane == 0) ? 0 : cm->seq_params.subsampling_x);
  int stride = (w + MIN_TX_SIZE - 1) >> MIN_TX_SIZE_LOG2;
  int x = (mi_col << MI_SIZE_LOG2) >>
          ((plane == 0) ? 0 : cm->seq_params.subsampling_x);
  int y = (mi_row << MI_SIZE_LOG2) >>
          ((plane == 0) ? 0 : cm->seq_params.subsampling_y);
  x = (x + blk_col) >> MIN_TX_SIZE_LOG2;
  y = (y + blk_row) >> MIN_TX_SIZE_LOG2;
  uint32_t idx = y * stride + x;
  assert(idx < cm->tx_skip_buf_size[plane]);
  return cm->tx_skip[plane][idx];
}

#if CONFIG_DBLK_TXSKIP
uint8_t av1_lpf_get_txk_skip(const AV1_COMMON *cm, int px, int py, int plane) {
  int w = ((cm->width + MAX_SB_SIZE - 1) >> MAX_SB_SIZE_LOG2)
          << MAX_SB_SIZE_LOG2;
  w >>= ((plane == 0) ? 0 : cm->seq_params.subsampling_x);
  int stride = (w + MIN_TX_SIZE - 1) >> MIN_TX_SIZE_LOG2;
  px >>= MIN_TX_SIZE_LOG2;
  py >>= MIN_TX_SIZE_LOG2;
  uint32_t idx = py * stride + px;
  assert(idx < cm->tx_skip_buf_size[plane]);
  return cm->tx_skip[plane][idx];
}
#endif
