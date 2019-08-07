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
  assert(!is_inter_block(left_mi) || is_intrabc_block(left_mi));
  return left_mi->mode;
}

PREDICTION_MODE av1_above_block_mode(const MB_MODE_INFO *above_mi) {
  if (!above_mi) return DC_PRED;
  assert(!is_inter_block(above_mi) || is_intrabc_block(above_mi));
  return above_mi->mode;
}

#if CONFIG_INTRA_ENTROPY
const uint64_t *av1_block_mode(const MB_MODE_INFO *mi, PREDICTION_MODE *mode,
                               int64_t *recon_var) {
  if (!mi) {
    *mode = 255;
    *recon_var = 0;
    return NULL;
  } else {
    assert(!is_inter_block(mi) || is_intrabc_block(mi));
    *mode = mi->mode;
    *recon_var = mi->y_recon_var;
    return mi->y_gradient_hist;
  }
}

void av1_get_intra_block_feature(float *features, const MB_MODE_INFO *above_mi,
                                 const MB_MODE_INFO *left_mi,
                                 const MB_MODE_INFO *aboveleft_mi) {
  PREDICTION_MODE above, left, aboveleft;
  int64_t above_var, left_var, al_var;
  const uint64_t *above_hist = av1_block_mode(above_mi, &above, &above_var);
  const uint64_t *left_hist = av1_block_mode(left_mi, &left, &left_var);
  const uint64_t *al_hist = av1_block_mode(aboveleft_mi, &aboveleft, &al_var);

  float hist_total[3] = { 0.f, 0.f, 0.f };
  int pt = 0;

  if (above_hist) {
    for (int i = 0; i < 8; ++i) {
      hist_total[0] += (float)above_hist[i];
    }
  }
  if (hist_total[0] > 0.1f) {
    for (int i = 0; i < 8; ++i) {
      features[pt++] = (float)above_hist[i] / hist_total[0];
    }
  } else {
    // deal with all 0 case
    for (int i = 0; i < 8; ++i) {
      features[pt++] = 0.125f;
    }
  }

  if (left_hist) {
    for (int i = 0; i < 8; ++i) {
      hist_total[1] += (float)left_hist[i];
    }
  }
  if (hist_total[1] > 0.1f) {
    for (int i = 0; i < 8; ++i) {
      features[pt++] = (float)left_hist[i] / hist_total[1];
    }
  } else {
    for (int i = 0; i < 8; ++i) {
      features[pt++] = 0.125f;
    }
  }

  if (al_hist) {
    for (int i = 0; i < 8; ++i) {
      hist_total[2] += (float)al_hist[i];
    }
  }
  if (hist_total[2] > 0.1f) {
    for (int i = 0; i < 8; ++i) {
      features[pt++] = (float)al_hist[i] / hist_total[2];
    }
  } else {
    for (int i = 0; i < 8; ++i) {
      features[pt++] = 0.125f;
    }
  }
#if INTRA_MODEL == 0
  features[pt++] = above_var / 12534.f;
  features[pt++] = left_var / 12534.f;
  features[pt++] = al_var / 12534.f;
#endif  // INTRA_MODEL
#if INTRA_MODEL != 1
  for (int i = 0; i < 5; ++i) {
    features[pt++] =
        (above < INTRA_MODES && intra_mode_context[above] == i) ? 1.0f : 0.0f;
  }
  for (int i = 0; i < 5; ++i) {
    features[pt++] =
        (left < INTRA_MODES && intra_mode_context[left] == i) ? 1.0f : 0.0f;
  }
  for (int i = 0; i < 5; ++i) {
    features[pt++] =
        (aboveleft < INTRA_MODES && intra_mode_context[aboveleft] == i) ? 1.0f
                                                                        : 0.0f;
  }
#endif  // INTRA_MODEL
}

void av1_pdf2cdf(float *pdf, aom_cdf_prob *cdf, int nsymbs) {
  float pre = 1.f, thresh = 0.00001f, vsum = 1.f;
  // Give the probability entries a lower bound, to avoid zero
  for (int i = 0; i < nsymbs; ++i) {
    if (pdf[i] < thresh) {
      vsum += (thresh - pdf[i]);
      pdf[i] = thresh;
    }
  }
  for (int i = 0; i < nsymbs; ++i) {
    pdf[i] /= vsum;
    pre -= pdf[i];
    cdf[i] = (aom_cdf_prob)(pre * CDF_PROB_TOP);
  }
}

void av1_get_intra_uv_block_feature(float *feature, PREDICTION_MODE cur_y_mode,
                                    const MB_MODE_INFO *above_mi,
                                    const MB_MODE_INFO *left_mi) {
  PREDICTION_MODE y_mode[2];
  int64_t y_var[2];
  const uint64_t *above_hist = av1_block_mode(above_mi, y_mode, y_var);
  const uint64_t *left_hist = av1_block_mode(left_mi, y_mode + 1, y_var + 1);

  float hist_total[2] = { 0.f, 0.f };
  int pt = 0;

  if (above_hist) {
    for (int i = 0; i < 8; ++i) {
      hist_total[0] += above_hist[i];
    }
  }
  if (hist_total[0] > 0.1f) {
    for (int i = 0; i < 8; ++i) {
      feature[pt++] = above_hist[i] / hist_total[0];
    }
  } else {
    // deal with all 0 case
    for (int i = 0; i < 8; ++i) {
      feature[pt++] = 0.125f;
    }
  }
  if (left_hist) {
    for (int i = 0; i < 8; ++i) {
      hist_total[1] += left_hist[i];
    }
  }
  if (hist_total[1] > 0.1f) {
    for (int i = 0; i < 8; ++i) {
      feature[pt++] = left_hist[i] / hist_total[1];
    }
  } else {
    // deal with all 0 case
    for (int i = 0; i < 8; ++i) {
      feature[pt++] = 0.125f;
    }
  }

  feature[pt++] = (float)y_var[0] / 12534.f;
  feature[pt++] = (float)y_var[1] / 12534.f;
  for (int i = 0; i < INTRA_MODES; ++i) {
    feature[pt++] = (cur_y_mode == i) ? 1.0f : 0.0f;
  }
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
void av1_reset_skip_context(MACROBLOCKD *xd, int mi_row, int mi_col,
                            BLOCK_SIZE bsize, const int num_planes) {
  int i;
  int nplanes;
  int chroma_ref;
  assert(bsize < BLOCK_SIZES_ALL);

  chroma_ref =
      is_chroma_reference(mi_row, mi_col, bsize, xd->plane[1].subsampling_x,
                          xd->plane[1].subsampling_y);
  nplanes = 1 + (num_planes - 1) * chroma_ref;
  for (i = 0; i < nplanes; i++) {
    struct macroblockd_plane *const pd = &xd->plane[i];
    const BLOCK_SIZE plane_bsize =
        get_plane_block_size(bsize, pd->subsampling_x, pd->subsampling_y);
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

#if CONFIG_INTRA_ENTROPY
// Indices are sign, integer, and fractional part of the gradient value
static const uint8_t gradient_to_angle_bin[2][7][16] = {
  {
      { 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 0, 0, 0, 0 },
      { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1 },
      { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
      { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
      { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
      { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 },
      { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 },
  },
  {
      { 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4 },
      { 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3 },
      { 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 },
      { 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 },
      { 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 },
      { 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2 },
      { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 },
  },
};

static void get_gradient_hist(const uint8_t *dst, int stride, int rows,
                              int cols, uint64_t *hist) {
  dst += stride;
  for (int r = 1; r < rows; ++r) {
    for (int c = 1; c < cols; ++c) {
      int dx = dst[c] - dst[c - 1];
      int dy = dst[c] - dst[c - stride];
      int index;
      const int temp = dx * dx + dy * dy;
      if (dy == 0) {
        index = 2;
      } else {
        const int sn = (dx > 0) ^ (dy > 0);
        dx = abs(dx);
        dy = abs(dy);
        const int remd = (dx % dy) * 16 / dy;
        const int quot = dx / dy;
        index = gradient_to_angle_bin[sn][AOMMIN(quot, 6)][AOMMIN(remd, 15)];
      }
      hist[index] += temp;
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
      int dx = dst[c] - dst[c - 1];
      int dy = dst[c] - dst[c - stride];
      int index;
      const int temp = dx * dx + dy * dy;
      if (dy == 0) {
        index = 2;
      } else {
        const int sn = (dx > 0) ^ (dy > 0);
        dx = abs(dx);
        dy = abs(dy);
        const int remd = (dx % dy) * 16 / dy;
        const int quot = dx / dy;
        index = gradient_to_angle_bin[sn][AOMMIN(quot, 6)][AOMMIN(remd, 15)];
      }
      hist[index] += temp;
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
    get_gradient_hist(y_dst, y_stride, y_block_rows, y_block_cols,
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
#endif  // CONFIG_INTRA_ENTROPY
