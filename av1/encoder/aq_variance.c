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

#include <math.h>
#include <stdlib.h>

#include "aom_dsp/aom_dsp_common.h"
#include "aom_ports/mem.h"

#include "av1/encoder/aq_variance.h"
#include "av1/common/seg_common.h"
#include "av1/encoder/encodeframe.h"
#include "av1/encoder/ratectrl.h"
#include "av1/encoder/rd.h"
#include "av1/encoder/segmentation.h"
#include "av1/encoder/dwt.h"
#include "config/aom_config.h"

#if !CONFIG_REALTIME_ONLY
static const double rate_ratio[MAX_SEGMENTS] = { 2.2, 1.7, 1.3, 1.0,
                                                 0.9, .8,  .7,  .6 };

static const double deltaq_rate_ratio[MAX_SEGMENTS] = { 2.5,  2.0, 1.5, 1.0,
                                                        0.75, 1.0, 1.0, 1.0 };
#define ENERGY_MIN (-4)
#define ENERGY_MAX (1)
#define ENERGY_SPAN (ENERGY_MAX - ENERGY_MIN + 1)
#define ENERGY_IN_BOUNDS(energy) \
  assert((energy) >= ENERGY_MIN && (energy) <= ENERGY_MAX)

static const int segment_id[ENERGY_SPAN] = { 0, 1, 1, 2, 3, 4 };

#define SEGMENT_ID(i) segment_id[(i) - ENERGY_MIN]

void av1_vaq_frame_setup(AV1_COMP *cpi) {
  AV1_COMMON *cm = &cpi->common;
  const RefreshFrameInfo *const refresh_frame = &cpi->refresh_frame;
  const int base_qindex = cm->quant_params.base_qindex;
  struct segmentation *seg = &cm->seg;
  int i;

  int resolution_change =
      cm->prev_frame && (cm->width != cm->prev_frame->width ||
                         cm->height != cm->prev_frame->height);
  int avg_energy = (int)(cpi->twopass_frame.mb_av_energy - 2);
  double avg_ratio;
  if (avg_energy > 7) avg_energy = 7;
  if (avg_energy < 0) avg_energy = 0;
  avg_ratio = rate_ratio[avg_energy];

  if (resolution_change) {
    memset(cpi->enc_seg.map, 0, cm->mi_params.mi_rows * cm->mi_params.mi_cols);
    av1_clearall_segfeatures(seg);
    av1_disable_segmentation(seg);
    return;
  }
  if (frame_is_intra_only(cm) || cm->features.error_resilient_mode ||
      refresh_frame->alt_ref_frame ||
      (refresh_frame->golden_frame && !cpi->rc.is_src_frame_alt_ref)) {
    cpi->vaq_refresh = 1;

    av1_enable_segmentation(seg);
    av1_clearall_segfeatures(seg);

    for (i = 0; i < MAX_SEGMENTS; ++i) {
      // Set up avg segment id to be 1.0 and adjust the other segments around
      // it.
      int qindex_delta =
          av1_compute_qdelta_by_rate(cpi, cm->current_frame.frame_type,
                                     base_qindex, rate_ratio[i] / avg_ratio);

      // We don't allow qindex 0 in a segment if the base value is not 0.
      // Q index 0 (lossless) implies 4x4 encoding only and in AQ mode a segment
      // Q delta is sometimes applied without going back around the rd loop.
      // This could lead to an illegal combination of partition size and q.
      if ((base_qindex != 0) && ((base_qindex + qindex_delta) == 0)) {
        qindex_delta = -base_qindex + 1;
      }

      av1_set_segdata(seg, i, SEG_LVL_ALT_Q, qindex_delta);
      av1_enable_segfeature(seg, i, SEG_LVL_ALT_Q);
    }
  }
}

int av1_log_block_avg(const AV1_COMP *cpi, const MACROBLOCK *x, BLOCK_SIZE bs,
                      int mi_row, int mi_col) {
  // This functions returns the block average of luma block
  unsigned int sum, avg, num_pix;
  int r, c;
  const int pic_w = cpi->common.width;
  const int pic_h = cpi->common.height;
  const int bw = MI_SIZE * mi_size_wide[bs];
  const int bh = MI_SIZE * mi_size_high[bs];
  const uint16_t *x16 = CONVERT_TO_SHORTPTR(x->plane[0].src.buf);

  sum = 0;
  num_pix = 0;
  avg = 0;
  int row = mi_row << MI_SIZE_LOG2;
  int col = mi_col << MI_SIZE_LOG2;
  for (r = row; (r < (row + bh)) && (r < pic_h); r++) {
    for (c = col; (c < (col + bw)) && (c < pic_w); c++) {
      sum += *(x16 + r * x->plane[0].src.stride + c);
      num_pix++;
    }
  }
  if (num_pix != 0) {
    avg = sum / num_pix;
  }
  return avg;
}

#define DEFAULT_E_MIDPOINT 10.0

static unsigned int haar_ac_energy(const MACROBLOCK *x, BLOCK_SIZE bs) {
  const MACROBLOCKD *xd = &x->e_mbd;
  int stride = x->plane[0].src.stride;
  const uint8_t *buf = x->plane[0].src.buf;
  const int num_8x8_cols = block_size_wide[bs] / 8;
  const int num_8x8_rows = block_size_high[bs] / 8;
  const int hbd = is_cur_buf_hbd(xd);

  int64_t var = av1_haar_ac_sad_mxn_uint8_input(buf, stride, hbd, num_8x8_rows,
                                                num_8x8_cols);

  return (unsigned int)((uint64_t)var * 256) >> num_pels_log2_lookup[bs];
}

static double log_block_wavelet_energy(const MACROBLOCK *x, BLOCK_SIZE bs) {
  unsigned int haar_sad = haar_ac_energy(x, bs);
  return log1p(haar_sad);
}

int av1_block_wavelet_energy_level(const AV1_COMP *cpi, const MACROBLOCK *x,
                                   BLOCK_SIZE bs) {
  double energy, energy_midpoint;
  energy_midpoint = (is_stat_consumption_stage_twopass(cpi))
                        ? cpi->twopass_frame.frame_avg_haar_energy
                        : DEFAULT_E_MIDPOINT;
  energy = log_block_wavelet_energy(x, bs) - energy_midpoint;
  return clamp((int)round(energy), ENERGY_MIN, ENERGY_MAX);
}

int av1_compute_q_from_energy_level_deltaq_mode(const AV1_COMP *const cpi,
                                                int block_var_level) {
  int rate_level;
  const AV1_COMMON *const cm = &cpi->common;

  if (DELTA_Q_PERCEPTUAL_MODULATION == 1) {
    ENERGY_IN_BOUNDS(block_var_level);
    rate_level = SEGMENT_ID(block_var_level);
  } else {
    rate_level = block_var_level;
  }
  const int base_qindex = cm->quant_params.base_qindex;
  int qindex_delta =
      av1_compute_qdelta_by_rate(cpi, cm->current_frame.frame_type, base_qindex,
                                 deltaq_rate_ratio[rate_level]);

  if ((base_qindex != 0) && ((base_qindex + qindex_delta) == 0)) {
    qindex_delta = -base_qindex + 1;
  }
  return base_qindex + qindex_delta;
}

// Comparer used by qsort() to order an array of unsigned int from smallest to
// largest.
static int comp_unsigned_int(const void *a, const void *b) {
  unsigned int arg1 = *(const unsigned int *)a;
  unsigned int arg2 = *(const unsigned int *)b;

  return (arg1 > arg2) - (arg1 < arg2);
}

unsigned int av1_get_variance_boost_block_variance(const AV1_COMP *cpi,
                                                   const MACROBLOCK *x) {
#define SUPERBLOCK_SIZE 64
#define SUBBLOCK_SIZE 8
#define SUBBLOCKS_IN_SB_DIM (SUPERBLOCK_SIZE / SUBBLOCK_SIZE)
#define SUBBLOCKS_IN_SB (SUBBLOCKS_IN_SB_DIM * SUBBLOCKS_IN_SB_DIM)
#define SUBBLOCKS_IN_OCTILE (SUBBLOCKS_IN_SB / 8)
  DECLARE_ALIGNED(16, static const uint16_t,
                  av1_highbd_all_zeros[SUBBLOCK_SIZE]) = { 0 };
  DECLARE_ALIGNED(16, static const uint8_t,
                  av1_all_zeros[SUBBLOCK_SIZE]) = { 0 };

  const MACROBLOCKD *xd = &x->e_mbd;
  unsigned int sse;
  // Octile is currently hard-coded and optimized for still pictures. In the
  // future, we might want to expose this as a parameter that can be fine-tuned
  // by the caller.
  // An octile of 5 was chosen because it was found to strike the best balance
  // between quality and consistency. Lower octiles tend to score lower in
  // SSIMU2, while higher octiles tend to harm subjective quality consistency,
  // especially in <1 MP images.
  const int octile = 5;
  const uint8_t *all_zeros = is_cur_buf_hbd(xd)
                                 ? CONVERT_TO_BYTEPTR(av1_highbd_all_zeros)
                                 : av1_all_zeros;
  unsigned int variances[SUBBLOCKS_IN_SB];

  // Calculate subblock variances.
  aom_variance_fn_t vf = cpi->ppi->fn_ptr[BLOCK_8X8].vf;
  for (int subb_i = 0; subb_i < SUBBLOCKS_IN_SB_DIM; subb_i++) {
    int i = subb_i * SUBBLOCK_SIZE;
    for (int subb_j = 0; subb_j < SUBBLOCKS_IN_SB_DIM; subb_j++) {
      int j = subb_j * SUBBLOCK_SIZE;
      // Truncating values to integers (i.e. the 64 term) was found to perform
      // better than rounding, or returning them as doubles.
      variances[subb_i * SUBBLOCKS_IN_SB_DIM + subb_j] =
          vf(x->plane[0].src.buf + i * x->plane[0].src.stride + j,
             x->plane[0].src.stride, all_zeros, 0, &sse) /
          64;
    }
  }

  // Order the 8x8 SB values from smallest to largest variance.
  qsort(variances, SUBBLOCKS_IN_SB, sizeof(unsigned int), comp_unsigned_int);

  // Sample three 8x8 variance values: at the specified octile, previous octile,
  // and next octile. Make sure we use the last subblock in each octile as the
  // representative of the octile.
  assert(octile >= 1 && octile <= 8);
  const int middle_index = octile * SUBBLOCKS_IN_OCTILE - 1;
  const int lower_index =
      AOMMAX(SUBBLOCKS_IN_OCTILE - 1, middle_index - SUBBLOCKS_IN_OCTILE);
  const int upper_index =
      AOMMIN(SUBBLOCKS_IN_SB - 1, middle_index + SUBBLOCKS_IN_OCTILE);

  // Weigh the three variances in a 1:2:1 ratio, with rounding (the +2 term).
  // This allows for smoother delta-q transitions among superblocks with
  // mixed-variance features.
  const unsigned int variance =
      (variances[lower_index] + (variances[middle_index] * 2) +
       variances[upper_index] + 2) /
      4;

  return variance;
}
#endif  // !CONFIG_REALTIME_ONLY

int av1_log_block_var(const AV1_COMP *cpi, const MACROBLOCK *x, BLOCK_SIZE bs) {
  DECLARE_ALIGNED(16, static const uint16_t,
                  av1_highbd_all_zeros[MAX_SB_SIZE]) = { 0 };
  DECLARE_ALIGNED(16, static const uint8_t, av1_all_zeros[MAX_SB_SIZE]) = { 0 };

  // This function returns a score for the blocks local variance as calculated
  // by: sum of the log of the (4x4 variances) of each subblock to the current
  // block (x,bs)
  // * 32 / number of pixels in the block_size.
  // This is used for segmentation because to avoid situations in which a large
  // block with a gentle gradient gets marked high variance even though each
  // subblock has a low variance.   This allows us to assign the same segment
  // number for the same sorts of area regardless of how the partitioning goes.

  const MACROBLOCKD *xd = &x->e_mbd;
  double var = 0;
  unsigned int sse;
  int i, j;

  int right_overflow =
      (xd->mb_to_right_edge < 0) ? ((-xd->mb_to_right_edge) >> 3) : 0;
  int bottom_overflow =
      (xd->mb_to_bottom_edge < 0) ? ((-xd->mb_to_bottom_edge) >> 3) : 0;

  const int bw = MI_SIZE * mi_size_wide[bs] - right_overflow;
  const int bh = MI_SIZE * mi_size_high[bs] - bottom_overflow;

  aom_variance_fn_t vf = cpi->ppi->fn_ptr[BLOCK_4X4].vf;
  for (i = 0; i < bh; i += 4) {
    for (j = 0; j < bw; j += 4) {
      if (is_cur_buf_hbd(xd)) {
        var += log1p(vf(x->plane[0].src.buf + i * x->plane[0].src.stride + j,
                        x->plane[0].src.stride,
                        CONVERT_TO_BYTEPTR(av1_highbd_all_zeros), 0, &sse) /
                     16.0);
      } else {
        var += log1p(vf(x->plane[0].src.buf + i * x->plane[0].src.stride + j,
                        x->plane[0].src.stride, av1_all_zeros, 0, &sse) /
                     16.0);
      }
    }
  }
  // Use average of 4x4 log variance. The range for 8 bit 0 - 9.704121561.
  var /= (bw / 4 * bh / 4);
  if (var > 7) var = 7;

  return (int)(var);
}
