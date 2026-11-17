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

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"
#include "config/av1_rtcd.h"

#include "aom_dsp/aom_dsp_common.h"
#include "aom_mem/aom_mem.h"
#include "aom_ports/aom_once.h"
#include "aom_ports/mem.h"
#include "av1/common/av1_common_int.h"
#include "av1/common/cfl.h"
#include "av1/common/reconintra.h"

enum {
  NEED_LEFT = 1 << 1,
  NEED_ABOVE = 1 << 2,
  NEED_ABOVERIGHT = 1 << 3,
  NEED_ABOVELEFT = 1 << 4,
  NEED_BOTTOMLEFT = 1 << 5,
};

#define INTRA_EDGE_FILT 3
#define INTRA_EDGE_TAPS 5
#define MAX_UPSAMPLE_SZ 16
#define NUM_INTRA_NEIGHBOUR_PIXELS (MAX_TX_SIZE * 2 + 32)

#if CONFIG_AV1_HIGHBITDEPTH
static const uint8_t extend_modes[INTRA_MODES] = {
  NEED_ABOVE | NEED_LEFT,                   // DC
  NEED_ABOVE,                               // V
  NEED_LEFT,                                // H
  NEED_ABOVE | NEED_ABOVERIGHT,             // D45
  NEED_LEFT | NEED_ABOVE | NEED_ABOVELEFT,  // D135
  NEED_LEFT | NEED_ABOVE | NEED_ABOVELEFT,  // D113
  NEED_LEFT | NEED_ABOVE | NEED_ABOVELEFT,  // D157
  NEED_LEFT | NEED_BOTTOMLEFT,              // D203
  NEED_ABOVE | NEED_ABOVERIGHT,             // D67
  NEED_LEFT | NEED_ABOVE,                   // SMOOTH
  NEED_LEFT | NEED_ABOVE,                   // SMOOTH_V
  NEED_LEFT | NEED_ABOVE,                   // SMOOTH_H
  NEED_LEFT | NEED_ABOVE | NEED_ABOVELEFT,  // PAETH
};
#endif  // CONFIG_AV1_HIGHBITDEPTH

// In quadtree Z-order (TL, TR, BL, BR), the top-right neighbor at
// (rm - 1, cm + bw_unit) was coded earlier iff the horizontal split boundary
// at rm is at least as coarse as the vertical boundary at cm + bw_unit, i.e.,
// (rm & -rm) >= ((cm + bw_unit) & -(cm + bw_unit)). PARTITION_VERT_A and
// PARTITION_VERT_B visit the final square split in column-major (TL, BL, TR,
// BR) order, so BL blocks (rm & bh_unit) precede TR and lack top-right.
static inline int has_top_right_in_sb(BLOCK_SIZE sb_size, BLOCK_SIZE bsize,
                                      int mi_row, int mi_col,
                                      PARTITION_TYPE partition) {
  assert((unsigned)bsize < BLOCK_SIZES_ALL);
  assert(sb_size == BLOCK_64X64 || sb_size == BLOCK_128X128);
  const int bw_unit = mi_size_wide[bsize];
  const int bh_unit = mi_size_high[bsize];
  const int sb_mi_size = mi_size_high[sb_size];
  const int rm = (mi_row & (sb_mi_size - 1)) & -bh_unit;
  if (rm == 0) return 1;
  const int cm = (mi_col & (sb_mi_size - 1)) & -bw_unit;
  if (cm + bw_unit >= sb_mi_size) return 0;

  if (bw_unit == bh_unit && bw_unit > 1 &&
      (partition == PARTITION_VERT_A || partition == PARTITION_VERT_B) &&
      (rm & bh_unit)) {
    return 0;
  }
  return (unsigned int)(rm & -rm) >=
         (unsigned int)((cm + bw_unit) & -(cm + bw_unit));
}

// Symmetrically, the bottom-left neighbor at (rm + bh_unit, cm - 1) was coded
// earlier iff the vertical boundary at cm strictly exceeds the horizontal
// boundary at rm + bh_unit: (cm & -cm) > ((rm + bh_unit) & -(rm + bh_unit)).
// Under PARTITION_VERT_A / PARTITION_VERT_B column-major square order, TR
// (!(rm & bh_unit) && (cm & bw_unit)) follows BL and thus has bottom-left.
static inline int has_bottom_left_in_sb(BLOCK_SIZE sb_size, BLOCK_SIZE bsize,
                                        int mi_row, int mi_col,
                                        PARTITION_TYPE partition,
                                        int bottom_left_count_unit, int row_off,
                                        int ss_y) {
  assert((unsigned)bsize < BLOCK_SIZES_ALL);
  assert(sb_size == BLOCK_64X64 || sb_size == BLOCK_128X128);
  const int bw_unit = mi_size_wide[bsize];
  const int bh_unit = mi_size_high[bsize];
  const int sb_mi_size = mi_size_high[sb_size];
  const int rm = (mi_row & (sb_mi_size - 1)) & -bh_unit;
  const int cm = (mi_col & (sb_mi_size - 1)) & -bw_unit;

  if (cm == 0) {
    const int row_off_in_sb = (rm >> ss_y) + row_off;
    const int sb_height_unit = sb_mi_size >> ss_y;
    return row_off_in_sb + bottom_left_count_unit < sb_height_unit;
  }
  if (rm + bh_unit >= sb_mi_size) return 0;

  if (bw_unit == bh_unit && bw_unit > 1 &&
      (partition == PARTITION_VERT_A || partition == PARTITION_VERT_B) &&
      !(rm & bh_unit) && (cm & bw_unit)) {
    return 1;
  }
  return (unsigned int)(cm & -cm) >
         (unsigned int)((rm + bh_unit) & -(rm + bh_unit));
}

// Fast-path availability when (ss_x | ss_y | col_off | row_off) == 0,
// covering both luma blocks and 4:4:4 chroma blocks at zero offset.
static inline int has_top_right_unsubsampled_origin(BLOCK_SIZE sb_size,
                                                    BLOCK_SIZE bsize,
                                                    int mi_row, int mi_col,
                                                    PARTITION_TYPE partition,
                                                    TX_SIZE txsz) {
  assert((unsigned)bsize < BLOCK_SIZES_ALL);
  assert((unsigned)txsz < TX_SIZES_ALL);
  const int bw_unit = mi_size_wide[bsize];
  const int top_right_count_unit = tx_size_wide_unit[txsz];
  if (top_right_count_unit < bw_unit) return 1;
  return has_top_right_in_sb(sb_size, bsize, mi_row, mi_col, partition);
}

static inline int has_bottom_left_unsubsampled_origin(BLOCK_SIZE sb_size,
                                                      BLOCK_SIZE bsize,
                                                      int mi_row, int mi_col,
                                                      PARTITION_TYPE partition,
                                                      TX_SIZE txsz) {
  assert((unsigned)bsize < BLOCK_SIZES_ALL);
  assert((unsigned)txsz < TX_SIZES_ALL);
  const int bh_unit = mi_size_high[bsize];
  const int bottom_left_count_unit = tx_size_high_unit[txsz];
  if (bottom_left_count_unit < bh_unit) return 1;
  return has_bottom_left_in_sb(sb_size, bsize, mi_row, mi_col, partition,
                               bottom_left_count_unit, 0, 0);
}

static int has_top_right(BLOCK_SIZE sb_size, BLOCK_SIZE bsize, int mi_row,
                         int mi_col, int top_available, int right_available,
                         PARTITION_TYPE partition, TX_SIZE txsz, int row_off,
                         int col_off, int ss_x, int ss_y) {
  assert((unsigned)bsize < BLOCK_SIZES_ALL);
  if (!top_available || !right_available) return 0;

  const int bw_unit = mi_size_wide[bsize];
  const int plane_bw_unit = AOMMAX(bw_unit >> ss_x, 1);
  const int top_right_count_unit = tx_size_wide_unit[txsz];

  if (row_off > 0) {  // Just need to check if enough pixels on the right.
    if (block_size_wide[bsize] > block_size_wide[BLOCK_64X64]) {
      // Special case: For 128x128 blocks, the transform unit whose
      // top-right corner is at the center of the block does in fact have
      // pixels available at its top-right corner.
      if (row_off == mi_size_high[BLOCK_64X64] >> ss_y &&
          col_off + top_right_count_unit == mi_size_wide[BLOCK_64X64] >> ss_x) {
        return 1;
      }
      const int plane_bw_unit_64 = mi_size_wide[BLOCK_64X64] >> ss_x;
      const int col_off_64 = col_off & (plane_bw_unit_64 - 1);
      return col_off_64 + top_right_count_unit < plane_bw_unit_64;
    }
    return col_off + top_right_count_unit < plane_bw_unit;
  } else {
    // All top-right pixels are in the block above, which is already available.
    if (col_off + top_right_count_unit < plane_bw_unit) return 1;

    return has_top_right_in_sb(sb_size, bsize, mi_row, mi_col, partition);
  }
}

static int has_bottom_left(BLOCK_SIZE sb_size, BLOCK_SIZE bsize, int mi_row,
                           int mi_col, int bottom_available, int left_available,
                           PARTITION_TYPE partition, TX_SIZE txsz, int row_off,
                           int col_off, int ss_x, int ss_y) {
  assert((unsigned)bsize < BLOCK_SIZES_ALL);
  if (!bottom_available || !left_available) return 0;

  // Special case for 128x* blocks, when col_off is half the block width.
  // This is needed because 128x* superblocks are divided into 64x* blocks in
  // raster order
  if (block_size_wide[bsize] > block_size_wide[BLOCK_64X64] && col_off > 0) {
    const int plane_bw_unit_64 = mi_size_wide[BLOCK_64X64] >> ss_x;
    const int col_off_64 = col_off & (plane_bw_unit_64 - 1);
    if (col_off_64 == 0) {
      // We are at the left edge of top-right or bottom-right 64x* block.
      const int plane_bh_unit_64 = mi_size_high[BLOCK_64X64] >> ss_y;
      const int row_off_64 = row_off & (plane_bh_unit_64 - 1);
      const int plane_bh_unit =
          AOMMIN(mi_size_high[bsize] >> ss_y, plane_bh_unit_64);
      // Check if all bottom-left pixels are in the left 64x* block (which is
      // already coded).
      return row_off_64 + tx_size_high_unit[txsz] < plane_bh_unit;
    }
  }

  if (col_off > 0) {
    // Bottom-left pixels are in the bottom-left block, which is not available.
    return 0;
  } else {
    const int bh_unit = mi_size_high[bsize];
    const int plane_bh_unit = AOMMAX(bh_unit >> ss_y, 1);
    const int bottom_left_count_unit = tx_size_high_unit[txsz];

    // All bottom-left pixels are in the left block, which is already available.
    if (row_off + bottom_left_count_unit < plane_bh_unit) return 1;

    return has_bottom_left_in_sb(sb_size, bsize, mi_row, mi_col, partition,
                                 bottom_left_count_unit, row_off, ss_y);
  }
}

typedef void (*intra_pred_fn)(uint8_t *dst, ptrdiff_t stride,
                              const uint8_t *above, const uint8_t *left);

static intra_pred_fn pred[INTRA_MODES][TX_SIZES_ALL];
static intra_pred_fn dc_pred[2][2][TX_SIZES_ALL];

#if CONFIG_AV1_HIGHBITDEPTH
typedef void (*intra_high_pred_fn)(uint16_t *dst, ptrdiff_t stride,
                                   const uint16_t *above, const uint16_t *left,
                                   int bd);
static intra_high_pred_fn pred_high[INTRA_MODES][TX_SIZES_ALL];
static intra_high_pred_fn dc_pred_high[2][2][TX_SIZES_ALL];
#endif

static void init_intra_predictors_internal(void) {
  assert(NELEMENTS(mode_to_angle_map) == INTRA_MODES);

#if CONFIG_REALTIME_ONLY && !CONFIG_AV1_DECODER
#define INIT_RECTANGULAR(p, type)             \
  p[TX_4X8] = aom_##type##_predictor_4x8;     \
  p[TX_8X4] = aom_##type##_predictor_8x4;     \
  p[TX_8X16] = aom_##type##_predictor_8x16;   \
  p[TX_16X8] = aom_##type##_predictor_16x8;   \
  p[TX_16X32] = aom_##type##_predictor_16x32; \
  p[TX_32X16] = aom_##type##_predictor_32x16; \
  p[TX_32X64] = aom_##type##_predictor_32x64; \
  p[TX_64X32] = aom_##type##_predictor_64x32;
#else
#define INIT_RECTANGULAR(p, type)             \
  p[TX_4X8] = aom_##type##_predictor_4x8;     \
  p[TX_8X4] = aom_##type##_predictor_8x4;     \
  p[TX_8X16] = aom_##type##_predictor_8x16;   \
  p[TX_16X8] = aom_##type##_predictor_16x8;   \
  p[TX_16X32] = aom_##type##_predictor_16x32; \
  p[TX_32X16] = aom_##type##_predictor_32x16; \
  p[TX_32X64] = aom_##type##_predictor_32x64; \
  p[TX_64X32] = aom_##type##_predictor_64x32; \
  p[TX_4X16] = aom_##type##_predictor_4x16;   \
  p[TX_16X4] = aom_##type##_predictor_16x4;   \
  p[TX_8X32] = aom_##type##_predictor_8x32;   \
  p[TX_32X8] = aom_##type##_predictor_32x8;   \
  p[TX_16X64] = aom_##type##_predictor_16x64; \
  p[TX_64X16] = aom_##type##_predictor_64x16;
#endif  // CONFIG_REALTIME_ONLY && !CONFIG_AV1_DECODER

#define INIT_NO_4X4(p, type)                  \
  p[TX_8X8] = aom_##type##_predictor_8x8;     \
  p[TX_16X16] = aom_##type##_predictor_16x16; \
  p[TX_32X32] = aom_##type##_predictor_32x32; \
  p[TX_64X64] = aom_##type##_predictor_64x64; \
  INIT_RECTANGULAR(p, type)

#define INIT_ALL_SIZES(p, type)           \
  p[TX_4X4] = aom_##type##_predictor_4x4; \
  INIT_NO_4X4(p, type)

  INIT_ALL_SIZES(pred[V_PRED], v)
  INIT_ALL_SIZES(pred[H_PRED], h)
  INIT_ALL_SIZES(pred[PAETH_PRED], paeth)
  INIT_ALL_SIZES(pred[SMOOTH_PRED], smooth)
  INIT_ALL_SIZES(pred[SMOOTH_V_PRED], smooth_v)
  INIT_ALL_SIZES(pred[SMOOTH_H_PRED], smooth_h)
  INIT_ALL_SIZES(dc_pred[0][0], dc_128)
  INIT_ALL_SIZES(dc_pred[0][1], dc_top)
  INIT_ALL_SIZES(dc_pred[1][0], dc_left)
  INIT_ALL_SIZES(dc_pred[1][1], dc)
#if CONFIG_AV1_HIGHBITDEPTH
  INIT_ALL_SIZES(pred_high[V_PRED], highbd_v)
  INIT_ALL_SIZES(pred_high[H_PRED], highbd_h)
  INIT_ALL_SIZES(pred_high[PAETH_PRED], highbd_paeth)
  INIT_ALL_SIZES(pred_high[SMOOTH_PRED], highbd_smooth)
  INIT_ALL_SIZES(pred_high[SMOOTH_V_PRED], highbd_smooth_v)
  INIT_ALL_SIZES(pred_high[SMOOTH_H_PRED], highbd_smooth_h)
  INIT_ALL_SIZES(dc_pred_high[0][0], highbd_dc_128)
  INIT_ALL_SIZES(dc_pred_high[0][1], highbd_dc_top)
  INIT_ALL_SIZES(dc_pred_high[1][0], highbd_dc_left)
  INIT_ALL_SIZES(dc_pred_high[1][1], highbd_dc)
#endif
#undef intra_pred_allsizes
}

// Directional prediction, zone 1: 0 < angle < 90
void av1_dr_prediction_z1_c(uint8_t *dst, ptrdiff_t stride, int bw, int bh,
                            const uint8_t *above, const uint8_t *left,
                            int upsample_above, int dx, int dy) {
  int r, c, x, base, shift, val;

  (void)left;
  (void)dy;
  assert(dy == 1);
  assert(dx > 0);

  const int max_base_x = ((bw + bh) - 1) << upsample_above;
  const int frac_bits = 6 - upsample_above;
  const int base_inc = 1 << upsample_above;
  x = dx;
  for (r = 0; r < bh; ++r, dst += stride, x += dx) {
    base = x >> frac_bits;
    shift = ((x << upsample_above) & 0x3F) >> 1;

    if (base >= max_base_x) {
      for (int i = r; i < bh; ++i) {
        memset(dst, above[max_base_x], bw * sizeof(dst[0]));
        dst += stride;
      }
      return;
    }

    for (c = 0; c < bw; ++c, base += base_inc) {
      if (base < max_base_x) {
        val = above[base] * (32 - shift) + above[base + 1] * shift;
        dst[c] = ROUND_POWER_OF_TWO(val, 5);
      } else {
        dst[c] = above[max_base_x];
      }
    }
  }
}

// Directional prediction, zone 2: 90 < angle < 180
void av1_dr_prediction_z2_c(uint8_t *dst, ptrdiff_t stride, int bw, int bh,
                            const uint8_t *above, const uint8_t *left,
                            int upsample_above, int upsample_left, int dx,
                            int dy) {
  assert(dx > 0);
  assert(dy > 0);

  const int min_base_x = -(1 << upsample_above);
  const int min_base_y = -(1 << upsample_left);
  (void)min_base_y;
  const int frac_bits_x = 6 - upsample_above;
  const int frac_bits_y = 6 - upsample_left;

  for (int r = 0; r < bh; ++r) {
    for (int c = 0; c < bw; ++c) {
      int val;
      int y = r + 1;
      int x = (c << 6) - y * dx;
      const int base_x = x >> frac_bits_x;
      if (base_x >= min_base_x) {
        const int shift = ((x * (1 << upsample_above)) & 0x3F) >> 1;
        val = above[base_x] * (32 - shift) + above[base_x + 1] * shift;
        val = ROUND_POWER_OF_TWO(val, 5);
      } else {
        x = c + 1;
        y = (r << 6) - x * dy;
        const int base_y = y >> frac_bits_y;
        assert(base_y >= min_base_y);
        const int shift = ((y * (1 << upsample_left)) & 0x3F) >> 1;
        val = left[base_y] * (32 - shift) + left[base_y + 1] * shift;
        val = ROUND_POWER_OF_TWO(val, 5);
      }
      dst[c] = val;
    }
    dst += stride;
  }
}

// Directional prediction, zone 3: 180 < angle < 270
void av1_dr_prediction_z3_c(uint8_t *dst, ptrdiff_t stride, int bw, int bh,
                            const uint8_t *above, const uint8_t *left,
                            int upsample_left, int dx, int dy) {
  int r, c, y, base, shift, val;

  (void)above;
  (void)dx;

  assert(dx == 1);
  assert(dy > 0);

  const int max_base_y = (bw + bh - 1) << upsample_left;
  const int frac_bits = 6 - upsample_left;
  const int base_inc = 1 << upsample_left;
  y = dy;
  for (c = 0; c < bw; ++c, y += dy) {
    base = y >> frac_bits;
    shift = ((y << upsample_left) & 0x3F) >> 1;

    for (r = 0; r < bh; ++r, base += base_inc) {
      if (base < max_base_y) {
        val = left[base] * (32 - shift) + left[base + 1] * shift;
        dst[r * stride + c] = ROUND_POWER_OF_TWO(val, 5);
      } else {
        for (; r < bh; ++r) dst[r * stride + c] = left[max_base_y];
        break;
      }
    }
  }
}

#if CONFIG_AV1_HIGHBITDEPTH
// Directional prediction, zone 1: 0 < angle < 90
void av1_highbd_dr_prediction_z1_c(uint16_t *dst, ptrdiff_t stride, int bw,
                                   int bh, const uint16_t *above,
                                   const uint16_t *left, int upsample_above,
                                   int dx, int dy, int bd) {
  int r, c, x, base, shift, val;

  (void)left;
  (void)dy;
  (void)bd;
  assert(dy == 1);
  assert(dx > 0);

  const int max_base_x = ((bw + bh) - 1) << upsample_above;
  const int frac_bits = 6 - upsample_above;
  const int base_inc = 1 << upsample_above;
  x = dx;
  for (r = 0; r < bh; ++r, dst += stride, x += dx) {
    base = x >> frac_bits;
    shift = ((x << upsample_above) & 0x3F) >> 1;

    if (base >= max_base_x) {
      for (int i = r; i < bh; ++i) {
        aom_memset16(dst, above[max_base_x], bw);
        dst += stride;
      }
      return;
    }

    for (c = 0; c < bw; ++c, base += base_inc) {
      if (base < max_base_x) {
        val = above[base] * (32 - shift) + above[base + 1] * shift;
        dst[c] = ROUND_POWER_OF_TWO(val, 5);
      } else {
        dst[c] = above[max_base_x];
      }
    }
  }
}

// Directional prediction, zone 2: 90 < angle < 180
void av1_highbd_dr_prediction_z2_c(uint16_t *dst, ptrdiff_t stride, int bw,
                                   int bh, const uint16_t *above,
                                   const uint16_t *left, int upsample_above,
                                   int upsample_left, int dx, int dy, int bd) {
  (void)bd;
  assert(dx > 0);
  assert(dy > 0);

  const int min_base_x = -(1 << upsample_above);
  const int min_base_y = -(1 << upsample_left);
  (void)min_base_y;
  const int frac_bits_x = 6 - upsample_above;
  const int frac_bits_y = 6 - upsample_left;

  for (int r = 0; r < bh; ++r) {
    for (int c = 0; c < bw; ++c) {
      int val;
      int y = r + 1;
      int x = (c << 6) - y * dx;
      const int base_x = x >> frac_bits_x;
      if (base_x >= min_base_x) {
        const int shift = ((x * (1 << upsample_above)) & 0x3F) >> 1;
        val = above[base_x] * (32 - shift) + above[base_x + 1] * shift;
        val = ROUND_POWER_OF_TWO(val, 5);
      } else {
        x = c + 1;
        y = (r << 6) - x * dy;
        const int base_y = y >> frac_bits_y;
        assert(base_y >= min_base_y);
        const int shift = ((y * (1 << upsample_left)) & 0x3F) >> 1;
        val = left[base_y] * (32 - shift) + left[base_y + 1] * shift;
        val = ROUND_POWER_OF_TWO(val, 5);
      }
      dst[c] = val;
    }
    dst += stride;
  }
}

// Directional prediction, zone 3: 180 < angle < 270
void av1_highbd_dr_prediction_z3_c(uint16_t *dst, ptrdiff_t stride, int bw,
                                   int bh, const uint16_t *above,
                                   const uint16_t *left, int upsample_left,
                                   int dx, int dy, int bd) {
  int r, c, y, base, shift, val;

  (void)above;
  (void)dx;
  (void)bd;
  assert(dx == 1);
  assert(dy > 0);

  const int max_base_y = (bw + bh - 1) << upsample_left;
  const int frac_bits = 6 - upsample_left;
  const int base_inc = 1 << upsample_left;
  y = dy;
  for (c = 0; c < bw; ++c, y += dy) {
    base = y >> frac_bits;
    shift = ((y << upsample_left) & 0x3F) >> 1;

    for (r = 0; r < bh; ++r, base += base_inc) {
      if (base < max_base_y) {
        val = left[base] * (32 - shift) + left[base + 1] * shift;
        dst[r * stride + c] = ROUND_POWER_OF_TWO(val, 5);
      } else {
        for (; r < bh; ++r) dst[r * stride + c] = left[max_base_y];
        break;
      }
    }
  }
}

#endif  // CONFIG_AV1_HIGHBITDEPTH

DECLARE_ALIGNED(16, const int8_t,
                av1_filter_intra_taps[FILTER_INTRA_MODES][8][8]) = {
  {
      { -6, 10, 0, 0, 0, 12, 0, 0 },
      { -5, 2, 10, 0, 0, 9, 0, 0 },
      { -3, 1, 1, 10, 0, 7, 0, 0 },
      { -3, 1, 1, 2, 10, 5, 0, 0 },
      { -4, 6, 0, 0, 0, 2, 12, 0 },
      { -3, 2, 6, 0, 0, 2, 9, 0 },
      { -3, 2, 2, 6, 0, 2, 7, 0 },
      { -3, 1, 2, 2, 6, 3, 5, 0 },
  },
  {
      { -10, 16, 0, 0, 0, 10, 0, 0 },
      { -6, 0, 16, 0, 0, 6, 0, 0 },
      { -4, 0, 0, 16, 0, 4, 0, 0 },
      { -2, 0, 0, 0, 16, 2, 0, 0 },
      { -10, 16, 0, 0, 0, 0, 10, 0 },
      { -6, 0, 16, 0, 0, 0, 6, 0 },
      { -4, 0, 0, 16, 0, 0, 4, 0 },
      { -2, 0, 0, 0, 16, 0, 2, 0 },
  },
  {
      { -8, 8, 0, 0, 0, 16, 0, 0 },
      { -8, 0, 8, 0, 0, 16, 0, 0 },
      { -8, 0, 0, 8, 0, 16, 0, 0 },
      { -8, 0, 0, 0, 8, 16, 0, 0 },
      { -4, 4, 0, 0, 0, 0, 16, 0 },
      { -4, 0, 4, 0, 0, 0, 16, 0 },
      { -4, 0, 0, 4, 0, 0, 16, 0 },
      { -4, 0, 0, 0, 4, 0, 16, 0 },
  },
  {
      { -2, 8, 0, 0, 0, 10, 0, 0 },
      { -1, 3, 8, 0, 0, 6, 0, 0 },
      { -1, 2, 3, 8, 0, 4, 0, 0 },
      { 0, 1, 2, 3, 8, 2, 0, 0 },
      { -1, 4, 0, 0, 0, 3, 10, 0 },
      { -1, 3, 4, 0, 0, 4, 6, 0 },
      { -1, 2, 3, 4, 0, 4, 4, 0 },
      { -1, 2, 2, 3, 4, 3, 3, 0 },
  },
  {
      { -12, 14, 0, 0, 0, 14, 0, 0 },
      { -10, 0, 14, 0, 0, 12, 0, 0 },
      { -9, 0, 0, 14, 0, 11, 0, 0 },
      { -8, 0, 0, 0, 14, 10, 0, 0 },
      { -10, 12, 0, 0, 0, 0, 14, 0 },
      { -9, 1, 12, 0, 0, 0, 12, 0 },
      { -8, 0, 0, 12, 0, 1, 11, 0 },
      { -7, 0, 0, 1, 12, 1, 9, 0 },
  },
};

void av1_filter_intra_predictor_c(uint8_t *dst, ptrdiff_t stride,
                                  TX_SIZE tx_size, const uint8_t *above,
                                  const uint8_t *left, int mode) {
  int r, c;
  uint8_t buffer[33][33];
  const int bw = tx_size_wide[tx_size];
  const int bh = tx_size_high[tx_size];

  assert(bw <= 32 && bh <= 32);

  for (r = 0; r < bh; ++r) buffer[r + 1][0] = left[r];
  memcpy(buffer[0], &above[-1], (bw + 1) * sizeof(uint8_t));

  for (r = 1; r < bh + 1; r += 2)
    for (c = 1; c < bw + 1; c += 4) {
      const uint8_t p0 = buffer[r - 1][c - 1];
      const uint8_t p1 = buffer[r - 1][c];
      const uint8_t p2 = buffer[r - 1][c + 1];
      const uint8_t p3 = buffer[r - 1][c + 2];
      const uint8_t p4 = buffer[r - 1][c + 3];
      const uint8_t p5 = buffer[r][c - 1];
      const uint8_t p6 = buffer[r + 1][c - 1];
      for (int k = 0; k < 8; ++k) {
        int r_offset = k >> 2;
        int c_offset = k & 0x03;
        int pr = av1_filter_intra_taps[mode][k][0] * p0 +
                 av1_filter_intra_taps[mode][k][1] * p1 +
                 av1_filter_intra_taps[mode][k][2] * p2 +
                 av1_filter_intra_taps[mode][k][3] * p3 +
                 av1_filter_intra_taps[mode][k][4] * p4 +
                 av1_filter_intra_taps[mode][k][5] * p5 +
                 av1_filter_intra_taps[mode][k][6] * p6;
        // Section 7.11.2.3 specifies the right-hand side of the assignment as
        //   Clip1( Round2Signed( pr, INTRA_FILTER_SCALE_BITS ) ).
        // Since Clip1() clips a negative value to 0, it is safe to replace
        // Round2Signed() with Round2().
        buffer[r + r_offset][c + c_offset] =
            clip_pixel(ROUND_POWER_OF_TWO(pr, FILTER_INTRA_SCALE_BITS));
      }
    }

  for (r = 0; r < bh; ++r) {
    memcpy(dst, &buffer[r + 1][1], bw * sizeof(uint8_t));
    dst += stride;
  }
}

#if CONFIG_AV1_HIGHBITDEPTH
static void highbd_filter_intra_predictor(uint16_t *dst, ptrdiff_t stride,
                                          TX_SIZE tx_size,
                                          const uint16_t *above,
                                          const uint16_t *left, int mode,
                                          int bd) {
  int r, c;
  uint16_t buffer[33][33];
  const int bw = tx_size_wide[tx_size];
  const int bh = tx_size_high[tx_size];

  assert(bw <= 32 && bh <= 32);

  for (r = 0; r < bh; ++r) buffer[r + 1][0] = left[r];
  memcpy(buffer[0], &above[-1], (bw + 1) * sizeof(buffer[0][0]));

  for (r = 1; r < bh + 1; r += 2)
    for (c = 1; c < bw + 1; c += 4) {
      const uint16_t p0 = buffer[r - 1][c - 1];
      const uint16_t p1 = buffer[r - 1][c];
      const uint16_t p2 = buffer[r - 1][c + 1];
      const uint16_t p3 = buffer[r - 1][c + 2];
      const uint16_t p4 = buffer[r - 1][c + 3];
      const uint16_t p5 = buffer[r][c - 1];
      const uint16_t p6 = buffer[r + 1][c - 1];
      for (int k = 0; k < 8; ++k) {
        int r_offset = k >> 2;
        int c_offset = k & 0x03;
        int pr = av1_filter_intra_taps[mode][k][0] * p0 +
                 av1_filter_intra_taps[mode][k][1] * p1 +
                 av1_filter_intra_taps[mode][k][2] * p2 +
                 av1_filter_intra_taps[mode][k][3] * p3 +
                 av1_filter_intra_taps[mode][k][4] * p4 +
                 av1_filter_intra_taps[mode][k][5] * p5 +
                 av1_filter_intra_taps[mode][k][6] * p6;
        // Section 7.11.2.3 specifies the right-hand side of the assignment as
        //   Clip1( Round2Signed( pr, INTRA_FILTER_SCALE_BITS ) ).
        // Since Clip1() clips a negative value to 0, it is safe to replace
        // Round2Signed() with Round2().
        buffer[r + r_offset][c + c_offset] = clip_pixel_highbd(
            ROUND_POWER_OF_TWO(pr, FILTER_INTRA_SCALE_BITS), bd);
      }
    }

  for (r = 0; r < bh; ++r) {
    memcpy(dst, &buffer[r + 1][1], bw * sizeof(dst[0]));
    dst += stride;
  }
}
#endif  // CONFIG_AV1_HIGHBITDEPTH

static inline int is_smooth(const MB_MODE_INFO *mbmi, int plane) {
  static_assert(SMOOTH_V_PRED == SMOOTH_PRED + 1 &&
                    SMOOTH_H_PRED == SMOOTH_PRED + 2 &&
                    UV_SMOOTH_V_PRED == UV_SMOOTH_PRED + 1 &&
                    UV_SMOOTH_H_PRED == UV_SMOOTH_PRED + 2,
                "Smooth prediction modes must be contiguous in enum order");
  if (plane == 0) {
    return (uint32_t)(mbmi->mode - SMOOTH_PRED) <=
           (uint32_t)(SMOOTH_H_PRED - SMOOTH_PRED);
  } else {
    // uv_mode is not set for inter blocks, so need to explicitly
    // detect that case.
    if (is_inter_block(mbmi)) return 0;

    return (uint32_t)(mbmi->uv_mode - UV_SMOOTH_PRED) <=
           (uint32_t)(UV_SMOOTH_H_PRED - UV_SMOOTH_PRED);
  }
}

static inline int get_intra_edge_filter_type(const MACROBLOCKD *xd, int plane) {
  const MB_MODE_INFO *const above =
      (plane == 0) ? xd->above_mbmi : xd->chroma_above_mbmi;
  const MB_MODE_INFO *const left =
      (plane == 0) ? xd->left_mbmi : xd->chroma_left_mbmi;
  return (above && is_smooth(above, plane)) || (left && is_smooth(left, plane));
}

static AOM_FORCE_INLINE int intra_edge_filter_strength(int bs0, int bs1,
                                                       int delta, int type) {
  const int d = abs(delta);
  const int blk_wh = bs0 + bs1;
  if (type == 0) {
    if (blk_wh <= 8) return d >= 56;
    if (blk_wh <= 16) return d >= 40;
    if (blk_wh <= 24) return (d >= 8) + (d >= 16) + (d >= 32);
    if (blk_wh <= 32) return (d >= 1) + (d >= 4) + (d >= 32);
    return (d >= 1) * 3;
  } else {
    if (blk_wh <= 8) return (d >= 40) + (d >= 64);
    if (blk_wh <= 16) return (d >= 20) + (d >= 48);
    if (blk_wh <= 24) return (d >= 4) * 3;
    return (d >= 1) * 3;
  }
}

void av1_filter_intra_edge_c(uint8_t *p, int sz, int strength) {
  if (!strength) return;

  const int kernel[INTRA_EDGE_FILT][INTRA_EDGE_TAPS] = { { 0, 4, 8, 4, 0 },
                                                         { 0, 5, 6, 5, 0 },
                                                         { 2, 4, 4, 4, 2 } };
  const int filt = strength - 1;
  uint8_t edge[129];

  memcpy(edge, p, sz * sizeof(*p));
  for (int i = 1; i < sz; i++) {
    int s = 0;
    for (int j = 0; j < INTRA_EDGE_TAPS; j++) {
      int k = i - 2 + j;
      k = (k < 0) ? 0 : k;
      k = (k > sz - 1) ? sz - 1 : k;
      s += edge[k] * kernel[filt][j];
    }
    s = (s + 8) >> 4;
    p[i] = s;
  }
}

static void filter_intra_edge_corner(uint8_t *p_above, uint8_t *p_left) {
  const int kernel[3] = { 5, 6, 5 };

  int s = (p_left[0] * kernel[0]) + (p_above[-1] * kernel[1]) +
          (p_above[0] * kernel[2]);
  s = (s + 8) >> 4;
  p_above[-1] = s;
  p_left[-1] = s;
}

void av1_upsample_intra_edge_c(uint8_t *p, int sz) {
  // interpolate half-sample positions
  assert(sz <= MAX_UPSAMPLE_SZ);

  uint8_t in[MAX_UPSAMPLE_SZ + 3];
  // copy p[-1..(sz-1)] and extend first and last samples
  in[0] = p[-1];
  in[1] = p[-1];
  for (int i = 0; i < sz; i++) {
    in[i + 2] = p[i];
  }
  in[sz + 2] = p[sz - 1];

  // interpolate half-sample edge positions
  p[-2] = in[0];
  for (int i = 0; i < sz; i++) {
    int s = -in[i] + (9 * in[i + 1]) + (9 * in[i + 2]) - in[i + 3];
    s = clip_pixel((s + 8) >> 4);
    p[2 * i - 1] = s;
    p[2 * i] = in[i + 2];
  }
}

static AOM_FORCE_INLINE void copy_above_row_lbd_partial(
    uint8_t *above_row, const uint8_t *above_ref, int n_copy, int max_n) {
  if (n_copy == 4) {
    memcpy(above_row, above_ref, 4);
  } else if (n_copy == 8) {
    memcpy(above_row, above_ref, 8);
  } else if (n_copy == 16) {
    memcpy(above_row, above_ref, 16);
  } else if (n_copy == 32) {
    memcpy(above_row, above_ref, 32);
  } else {
    assert(n_copy > 0 && n_copy <= max_n);
    (void)max_n;
    int k = 0;
    for (; k + 4 <= n_copy; k += 4) memcpy(above_row + k, above_ref + k, 4);
    for (; k < n_copy; ++k) above_row[k] = above_ref[k];
  }
}

static AOM_FORCE_INLINE void copy_above_row_lbd_64(uint8_t *above_row,
                                                   const uint8_t *above_ref,
                                                   int n_copy) {
  if (n_copy == 64) {
    memcpy(above_row, above_ref, 64);
  } else {
    copy_above_row_lbd_partial(above_row, above_ref, n_copy, 63);
  }
}

static AOM_FORCE_INLINE void copy_above_row_lbd_128(uint8_t *above_row,
                                                    const uint8_t *above_ref,
                                                    int n_copy) {
  if (n_copy == 128) {
    memcpy(above_row, above_ref, 128);
  } else if (n_copy == 64) {
    memcpy(above_row, above_ref, 64);
  } else {
    copy_above_row_lbd_partial(above_row, above_ref, n_copy, 127);
  }
}

static AOM_FORCE_INLINE void fill_intra_edge_lbd_tail(uint8_t *p, uint8_t val,
                                                      int n, int max_n) {
  assert(n > 0 && n <= max_n);
  (void)max_n;
  const uint32_t v32 = (uint32_t)val * 0x01010101U;
  int k = 0;
  for (; k + 4 <= n; k += 4) memcpy(p + k, &v32, 4);
  for (; k < n; ++k) p[k] = val;
}

static AOM_FORCE_INLINE void fill_intra_edge_lbd_64(uint8_t *p, uint8_t val,
                                                    int n) {
  if (n == 4) {
    memset(p, val, 4);
  } else if (n == 8) {
    memset(p, val, 8);
  } else if (n == 16) {
    memset(p, val, 16);
  } else if (n == 32) {
    memset(p, val, 32);
  } else if (n == 64) {
    memset(p, val, 64);
  } else {
    fill_intra_edge_lbd_tail(p, val, n, 64);
  }
}

static AOM_FORCE_INLINE void fill_intra_edge_lbd_128(uint8_t *p, uint8_t val,
                                                     int n) {
  if (n == 128) {
    memset(p, val, 128);
  } else if (n == 64) {
    memset(p, val, 64);
  } else if (n == 32) {
    memset(p, val, 32);
  } else if (n == 16) {
    memset(p, val, 16);
  } else if (n == 8) {
    memset(p, val, 8);
  } else if (n == 4) {
    memset(p, val, 4);
  } else {
    fill_intra_edge_lbd_tail(p, val, n, 128);
  }
}

// Out-of-line strided column gather to keep slow/border paths compact.
static AOM_NOINLINE void gather_left_col_lbd(uint8_t *left_col,
                                             const uint8_t *left_ref,
                                             int ref_stride, int n_copy) {
  assert(n_copy > 0);
#if !CONFIG_BIG_ENDIAN
  const ptrdiff_t s1 = (ptrdiff_t)ref_stride;
  const ptrdiff_t s2 = s1 * 2;
  uint8_t *dst_p = left_col;
  const uint8_t *ref_p = left_ref;
  const int n8 = n_copy >> 3;
  if (n8 > 0) {
    const ptrdiff_t s4 = s1 * 4;
    int i = 0;
    do {
      const uint8_t *const p2 = ref_p + s2;
      const uint32_t lo = ((uint32_t)ref_p[0] | ((uint32_t)ref_p[s1] << 8)) |
                          (((uint32_t)p2[0] | ((uint32_t)p2[s1] << 8)) << 16);
      const uint8_t *const q0 = ref_p + s4;
      const uint8_t *const q2 = q0 + s2;
      const uint32_t hi = ((uint32_t)q0[0] | ((uint32_t)q0[s1] << 8)) |
                          (((uint32_t)q2[0] | ((uint32_t)q2[s1] << 8)) << 16);
      const uint64_t v = (uint64_t)lo | ((uint64_t)hi << 32);
      memcpy(dst_p, &v, 8);
      dst_p += 8;
      ++i;
      if ((i < n8) || (n_copy & 7)) ref_p = q0 + s4;
    } while (i != n8);
  }
  if (n_copy & 4) {
    const uint8_t *const r2 = ref_p + s2;
    const uint32_t v = ((uint32_t)ref_p[0] | ((uint32_t)ref_p[s1] << 8)) |
                       (((uint32_t)r2[0] | ((uint32_t)r2[s1] << 8)) << 16);
    memcpy(dst_p, &v, 4);
    dst_p += 4;
    if (n_copy & 3) ref_p = r2 + s2;
  }
  if (n_copy & 2) {
    dst_p[0] = ref_p[0];
    dst_p[1] = ref_p[s1];
    dst_p += 2;
    if (n_copy & 1) ref_p += s2;
  }
  if (n_copy & 1) {
    dst_p[0] = ref_p[0];
  }
#else
  int i = 0;
  do {
    left_col[i] = left_ref[(ptrdiff_t)i * ref_stride];
  } while (++i != n_copy);
#endif
}

static inline void gather_left_col_8_lbd(uint8_t *dst, const uint8_t *left_ref,
                                         ptrdiff_t s1) {
#if !CONFIG_BIG_ENDIAN
  const ptrdiff_t s2 = s1 * 2;
  const ptrdiff_t s4 = s1 * 4;
  const uint8_t *const p2 = left_ref + s2;
  const uint32_t lo = ((uint32_t)left_ref[0] | ((uint32_t)left_ref[s1] << 8)) |
                      (((uint32_t)p2[0] | ((uint32_t)p2[s1] << 8)) << 16);
  const uint8_t *const q0 = left_ref + s4;
  const uint8_t *const q2 = q0 + s2;
  const uint32_t hi = ((uint32_t)q0[0] | ((uint32_t)q0[s1] << 8)) |
                      (((uint32_t)q2[0] | ((uint32_t)q2[s1] << 8)) << 16);
  const uint64_t v = (uint64_t)lo | ((uint64_t)hi << 32);
  memcpy(dst, &v, 8);
#else
  int i = 0;
  do {
    dst[i] = left_ref[(ptrdiff_t)i * s1];
  } while (++i != 8);
#endif
}

static inline void gather_left_col_pow2_lbd(uint8_t *dst,
                                            const uint8_t *left_ref, int stride,
                                            int n) {
  assert(n == 4 || n == 8 || n == 16 || n == 32 || n == 64);
  const ptrdiff_t s1 = (ptrdiff_t)stride;
  if (n == 4) {
#if !CONFIG_BIG_ENDIAN
    const uint8_t *const p2 = left_ref + s1 * 2;
    const uint32_t v = ((uint32_t)left_ref[0] | ((uint32_t)left_ref[s1] << 8)) |
                       (((uint32_t)p2[0] | ((uint32_t)p2[s1] << 8)) << 16);
    memcpy(dst, &v, 4);
#else
    int i = 0;
    do {
      dst[i] = left_ref[(ptrdiff_t)i * s1];
    } while (++i != 4);
#endif
    return;
  }
  const ptrdiff_t s8 = s1 * 8;
  int r = 0;
  for (;;) {
    gather_left_col_8_lbd(dst + r, left_ref, s1);
    r += 8;
    if (r == n) break;
    left_ref += s8;
  }
}

static AOM_FORCE_INLINE BLOCK_SIZE scale_chroma_bsize(BLOCK_SIZE bsize,
                                                      int subsampling_x,
                                                      int subsampling_y) {
  assert(subsampling_x >= 0 && subsampling_x < 2);
  assert(subsampling_y >= 0 && subsampling_y < 2);
  BLOCK_SIZE bs = bsize;
  switch (bsize) {
    case BLOCK_4X4:
      if (subsampling_x == 1 && subsampling_y == 1)
        bs = BLOCK_8X8;
      else if (subsampling_x == 1)
        bs = BLOCK_8X4;
      else if (subsampling_y == 1)
        bs = BLOCK_4X8;
      break;
    case BLOCK_4X8:
      if (subsampling_x == 1 && subsampling_y == 1)
        bs = BLOCK_8X8;
      else if (subsampling_x == 1)
        bs = BLOCK_8X8;
      else if (subsampling_y == 1)
        bs = BLOCK_4X8;
      break;
    case BLOCK_8X4:
      if (subsampling_x == 1 && subsampling_y == 1)
        bs = BLOCK_8X8;
      else if (subsampling_x == 1)
        bs = BLOCK_8X4;
      else if (subsampling_y == 1)
        bs = BLOCK_8X8;
      break;
    case BLOCK_4X16:
      if (subsampling_x == 1 && subsampling_y == 1)
        bs = BLOCK_8X16;
      else if (subsampling_x == 1)
        bs = BLOCK_8X16;
      else if (subsampling_y == 1)
        bs = BLOCK_4X16;
      break;
    case BLOCK_16X4:
      if (subsampling_x == 1 && subsampling_y == 1)
        bs = BLOCK_16X8;
      else if (subsampling_x == 1)
        bs = BLOCK_16X4;
      else if (subsampling_y == 1)
        bs = BLOCK_16X8;
      break;
    default: break;
  }
  return bs;
}

// Bit layout of packed DR metadata passed to out-of-line builders:
//   [7:0]   tx_size (0 .. TX_SIZES_ALL - 1)
//   [15:8]  plane (0 .. MAX_MB_PLANE - 1)
//   [31:16] p_angle (0 .. 270)
//   [32]    disable_edge_filter (0 or 1)
//   [33]    ss_x (0 or 1)
//   [34]    ss_y (0 or 1)
//   [42:35] sb_size (BLOCK_64X64 or BLOCK_128X128)
//   [50:43] col_off (0 .. 255, MI units)
//   [58:51] row_off (0 .. 255, MI units)
//   [61:59] filter_intra_mode (0 .. FILTER_INTRA_MODES) for generic DR, OR
//           [59] has_edge_px (0 or 1) and [63:60] ext_px (0 .. 8) for 8x8 z1/z3
static AOM_FORCE_INLINE uint64_t pack_dr_facade_meta(
    TX_SIZE tx_size, int plane, int p_angle, int disable_edge_filter, int ss_x,
    int ss_y, BLOCK_SIZE sb_size, int col_off, int row_off,
    FILTER_INTRA_MODE filter_intra_mode) {
  static_assert(TX_SIZES_ALL <= 0x100 && MAX_MB_PLANE <= 0x100 &&
                    BLOCK_SIZES_ALL <= 0x100,
                "DR packed metadata fields must fit in 8 bits");
  static_assert(FILTER_INTRA_MODES <= 0x7,
                "FILTER_INTRA_MODES must fit in 3 bits");
  assert((unsigned)tx_size < TX_SIZES_ALL);
  assert((unsigned)plane < MAX_MB_PLANE);
  assert(p_angle >= 0 && p_angle <= 270);
  assert(sb_size == BLOCK_64X64 || sb_size == BLOCK_128X128);
  assert((unsigned)col_off <= 0xff && (unsigned)row_off <= 0xff);
  assert((unsigned)filter_intra_mode <= FILTER_INTRA_MODES);
  return (uint64_t)(uint8_t)tx_size | ((uint64_t)(uint8_t)plane << 8) |
         ((uint64_t)(uint16_t)p_angle << 16) |
         ((uint64_t)(uint8_t)(disable_edge_filter != 0) << 32) |
         ((uint64_t)(uint8_t)(ss_x != 0) << 33) |
         ((uint64_t)(uint8_t)(ss_y != 0) << 34) |
         ((uint64_t)(uint8_t)sb_size << 35) |
         ((uint64_t)(uint8_t)col_off << 43) |
         ((uint64_t)(uint8_t)row_off << 51) |
         ((uint64_t)(uint8_t)filter_intra_mode << 59);
}

static AOM_FORCE_INLINE uint64_t pack_dr_8x8_meta(
    int plane, int p_angle, int disable_edge_filter, int ss_x, int ss_y,
    BLOCK_SIZE sb_size, int col_off, int row_off, int has_edge_px, int ext_px) {
  assert((unsigned)plane < MAX_MB_PLANE);
  assert(p_angle >= 0 && p_angle <= 270);
  assert(sb_size == BLOCK_64X64 || sb_size == BLOCK_128X128);
  assert((unsigned)col_off <= 0xff && (unsigned)row_off <= 0xff);
  assert(ext_px >= 0 && ext_px <= 8);
  return (uint64_t)(uint8_t)TX_8X8 | ((uint64_t)(uint8_t)plane << 8) |
         ((uint64_t)(uint16_t)p_angle << 16) |
         ((uint64_t)(uint8_t)(disable_edge_filter != 0) << 32) |
         ((uint64_t)(uint8_t)(ss_x != 0) << 33) |
         ((uint64_t)(uint8_t)(ss_y != 0) << 34) |
         ((uint64_t)(uint8_t)sb_size << 35) |
         ((uint64_t)(uint8_t)col_off << 43) |
         ((uint64_t)(uint8_t)row_off << 51) |
         ((uint64_t)(uint8_t)(has_edge_px != 0) << 59) |
         ((uint64_t)(uint8_t)ext_px << 60);
}

// Bit layout of packed DR pixel counts passed to generic DR builders:
//   [15:0]  n_top_px (0 .. MAX_TX_SIZE)
//   [31:16] n_left_px (0 .. MAX_TX_SIZE)
//   [47:32] xr_clamped (0 .. MAX_TX_SIZE)
//   [63:48] yd_clamped (0 .. MAX_TX_SIZE)
static AOM_FORCE_INLINE uint64_t pack_dr_facade_px(int n_top_px, int n_left_px,
                                                   int xr_clamped,
                                                   int yd_clamped) {
  assert(n_top_px >= 0 && n_top_px <= MAX_TX_SIZE);
  assert(n_left_px >= 0 && n_left_px <= MAX_TX_SIZE);
  assert(xr_clamped >= 0 && xr_clamped <= MAX_TX_SIZE);
  assert(yd_clamped >= 0 && yd_clamped <= MAX_TX_SIZE);
  return (uint64_t)(uint16_t)n_top_px | ((uint64_t)(uint16_t)n_left_px << 16) |
         ((uint64_t)(uint16_t)xr_clamped << 32) |
         ((uint64_t)(uint16_t)yd_clamped << 48);
}

// Pack (plane, p_angle, disable_edge_filter) for 8x8 z2 to fit in 6 GP args.
static AOM_FORCE_INLINE uint32_t pack_dr_z2_8x8_meta(int plane, int p_angle,
                                                     int disable_edge_filter) {
  assert((unsigned)plane < MAX_MB_PLANE);
  assert(p_angle > 90 && p_angle < 180);
  return (uint32_t)(uint8_t)plane | ((uint32_t)(uint16_t)p_angle << 8) |
         ((uint32_t)(uint8_t)(disable_edge_filter != 0) << 24);
}

// Pack (mode, tx_size, bit_depth) and (n_top_px, n_left_px) to keep both LBD
// and HBD non-directional builders within 6 GP register arguments.
static AOM_FORCE_INLINE uint32_t pack_non_dir_meta(PREDICTION_MODE mode,
                                                   TX_SIZE tx_size,
                                                   int bit_depth) {
  assert((unsigned)mode < INTRA_MODES);
  assert((unsigned)tx_size < TX_SIZES_ALL);
  assert((unsigned)bit_depth <= 0xff);
  return (uint32_t)(uint8_t)mode | ((uint32_t)(uint8_t)tx_size << 8) |
         ((uint32_t)(uint8_t)bit_depth << 16);
}

static AOM_FORCE_INLINE uint32_t pack_non_dir_px(int n_top_px, int n_left_px) {
  assert(n_top_px >= 0 && n_top_px <= MAX_TX_SIZE);
  assert(n_left_px >= 0 && n_left_px <= MAX_TX_SIZE);
  return (uint32_t)(uint16_t)n_top_px | ((uint32_t)(uint16_t)n_left_px << 16);
}

// Returns 1 iff intra_edge_filter_type (smooth vs non-smooth neighbor) can
// affect intra_edge_filter_strength() or av1_use_intra_edge_upsample():
// - blk_wh > 32: strength is (d >= 1) * 3 and upsample is 0 for both types.
// - blk_wh == 8: upsample (0 < d < 40) matches; strength differs for d >= 40.
// - 8 < blk_wh <= 16: upsample differs for d < 40 (1 vs 0); strength matches
//   (1) for 40 <= d < 48 and differs for d >= 48 (1 vs 2) when edge_px > 0.
// - 16 < blk_wh <= 32: upsample is 0; strength matches (3) for d >= 32 and
//   differs for d < 32 when edge_px > 0.
static inline int need_intra_edge_filter_type_check(int txwpx, int txhpx,
                                                    int p_angle, int n_top_px,
                                                    int n_left_px) {
  const int blk_wh = txwpx + txhpx;
  if (blk_wh > 32) return 0;
  if (blk_wh <= 16) {
    if (blk_wh != 8) {
      const int da = abs(p_angle - 90);
      const int dl = abs(p_angle - 180);
      return (p_angle < 180 && (da < 40 || (n_top_px > 0 && da >= 48))) ||
             (p_angle > 90 && (dl < 40 || (n_left_px > 0 && dl >= 48)));
    }
    return (p_angle < 180 && n_top_px > 0 && abs(p_angle - 90) >= 40) ||
           (p_angle > 90 && n_left_px > 0 && abs(p_angle - 180) >= 40);
  }
  return (p_angle < 180 && n_top_px > 0 && abs(p_angle - 90) < 32) ||
         (p_angle > 90 && n_left_px > 0 && abs(p_angle - 180) < 32);
}

// Shared out-of-line TR/BL availability helpers (2 args -> zero stack spills).
static AOM_NOINLINE int has_top_right_available(const MACROBLOCKD *xd,
                                                uint64_t meta) {
  const TX_SIZE tx_size = (TX_SIZE)(meta & 0xff);
  const int ss_x = (int)((meta >> 33) & 1);
  const int ss_y = (int)((meta >> 34) & 1);
  const BLOCK_SIZE sb_size = (BLOCK_SIZE)((meta >> 35) & 0xff);
  const int col_off = (int)((meta >> 43) & 0xff);
  const int row_off = (int)((meta >> 51) & 0xff);
  const int mi_col = -xd->mb_to_left_edge >> (3 + MI_SIZE_LOG2);
  const int txw = tx_size_wide_unit[tx_size];
  if (mi_col + ((col_off + txw) << ss_x) >= xd->tile.mi_col_end) return 0;
  const int mi_row = -xd->mb_to_top_edge >> (3 + MI_SIZE_LOG2);
  const MB_MODE_INFO *const mbmi = xd->mi[0];
  if ((ss_x | ss_y | col_off | row_off) == 0) {
    return has_top_right_unsubsampled_origin(sb_size, mbmi->bsize, mi_row,
                                             mi_col, mbmi->partition, tx_size);
  }
  BLOCK_SIZE bsize = mbmi->bsize;
  if (ss_x || ss_y) bsize = scale_chroma_bsize(bsize, ss_x, ss_y);
  return has_top_right(sb_size, bsize, mi_row, mi_col, 1, 1, mbmi->partition,
                       tx_size, row_off, col_off, ss_x, ss_y);
}

static AOM_NOINLINE int has_bottom_left_available(const MACROBLOCKD *xd,
                                                  uint64_t meta) {
  const TX_SIZE tx_size = (TX_SIZE)(meta & 0xff);
  const int ss_x = (int)((meta >> 33) & 1);
  const int ss_y = (int)((meta >> 34) & 1);
  const BLOCK_SIZE sb_size = (BLOCK_SIZE)((meta >> 35) & 0xff);
  const int col_off = (int)((meta >> 43) & 0xff);
  const int row_off = (int)((meta >> 51) & 0xff);
  const int mi_row = -xd->mb_to_top_edge >> (3 + MI_SIZE_LOG2);
  const int txh = tx_size_high_unit[tx_size];
  if (mi_row + ((row_off + txh) << ss_y) >= xd->tile.mi_row_end) return 0;
  const int mi_col = -xd->mb_to_left_edge >> (3 + MI_SIZE_LOG2);
  const MB_MODE_INFO *const mbmi = xd->mi[0];
  if ((ss_x | ss_y | col_off | row_off) == 0) {
    return has_bottom_left_unsubsampled_origin(
        sb_size, mbmi->bsize, mi_row, mi_col, mbmi->partition, tx_size);
  }
  BLOCK_SIZE bsize = mbmi->bsize;
  if (ss_x || ss_y) bsize = scale_chroma_bsize(bsize, ss_x, ss_y);
  return has_bottom_left(sb_size, bsize, mi_row, mi_col, 1, 1, mbmi->partition,
                         tx_size, row_off, col_off, ss_x, ss_y);
}

// Separate stack frame for 8x8 zone-1 edge buffers off the facade.
static AOM_NOINLINE void build_dr_z1_8x8_lbd(const MACROBLOCKD *xd,
                                             const uint8_t *ref, int ref_stride,
                                             uint8_t *dst, int dst_stride,
                                             uint64_t meta) {
  const int plane = (int)((meta >> 8) & 0xff);
  const int p_angle = (int)((meta >> 16) & 0xffff);
  assert(p_angle > 0 && p_angle < 90);
  const int disable_edge_filter = (int)((meta >> 32) & 1);
  const int n_left_nonzero = (int)((meta >> 59) & 1);
  const int xr_clamped = AOMMIN((int)((meta >> 60) & 0xf), 8);

  const uint8_t *const above_ref = ref - ref_stride;
  const int n_topright_px =
      (xr_clamped > 0 && has_top_right_available(xd, meta)) ? xr_clamped : 0;

  // 16-byte prefix [-16..-1], 32 upsampled edge samples [0..31], 32-byte tail
  // [32..63] (fits av1_upsample_intra_edge_sse4_1's p[-2..61] writes at sz=16).
  DECLARE_ALIGNED(16, uint8_t, above_data[16 + 32 + 32]);
  uint8_t *const above_row = above_data + 16;
  memset(above_data, 0, 16);
  memset(above_row + 16, 0, 48);
  int upsample_above = 0;

  assert(n_topright_px >= 0 && n_topright_px <= 8);
  if (n_topright_px == 8) {
    memcpy(above_row, above_ref, 16);
  } else {
    memcpy(above_row, above_ref, 8);
    if (n_topright_px > 0) {
      memcpy(above_row + 8, above_ref + 8, (size_t)n_topright_px);
    }
    memset(above_row + 8 + n_topright_px, above_ref[7 + n_topright_px],
           (size_t)(8 - n_topright_px));
  }
  if (!disable_edge_filter) {
    above_row[-1] = n_left_nonzero ? above_ref[-1] : above_ref[0];
    const int d = 90 - p_angle;
    const int type =
        (d < 40 || d >= 48) ? get_intra_edge_filter_type(xd, plane) : 0;
    if (type == 0) {
      if (d < 40) {
        upsample_above = 1;
        av1_upsample_intra_edge(above_row, 16);
      } else {
        av1_filter_intra_edge(above_row - 1, 17, 1);
      }
    } else {
      const int s = (d >= 20) + (d >= 48);
      if (s) av1_filter_intra_edge(above_row - 1, 17, s);
    }
  }
  av1_dr_prediction_z1(dst, dst_stride, 8, 8, above_row, NULL, upsample_above,
                       dr_intra_derivative[p_angle], 1);
}

// Separate stack frame for 8x8 zone-2 edge buffers off the facade.
static AOM_NOINLINE void build_dr_z2_8x8_lbd(const MACROBLOCKD *xd,
                                             const uint8_t *ref, int ref_stride,
                                             uint8_t *dst, int dst_stride,
                                             uint32_t z2_meta) {
  const int plane = (int)(z2_meta & 0xff);
  const int p_angle = (int)((z2_meta >> 8) & 0xffff);
  const int disable_edge_filter = (int)((z2_meta >> 24) & 1);
  assert(p_angle > 90 && p_angle < 180);

  const uint8_t *const above_ref = ref - ref_stride;
  const uint8_t *const left_ref = ref - 1;
  // 16-byte prefix [-16..-1], 8 edge samples, 24-byte tail [8..31] (exact fit
  // for av1_dr_prediction_z2's left[-2..31] / above[-2..31] SIMD over-reads
  // and av1_upsample_intra_edge_sse4_1's p[-2..29] writes at sz=8).
  DECLARE_ALIGNED(16, uint8_t, left_data[16 + 8 + 24]);
  DECLARE_ALIGNED(16, uint8_t, above_data[16 + 8 + 24]);
  uint8_t *const above_row = above_data + 16;
  uint8_t *const left_col = left_data + 16;
  memset(above_data, 0, 16);
  memset(above_row + 8, 0, 24);
  memset(left_data, 0, 16);
  memset(left_col + 8, 0, 24);
  const uint8_t tl = above_ref[-1];

  int upsample_above = 0;
  int upsample_left = 0;
  gather_left_col_pow2_lbd(left_col, left_ref, ref_stride, 8);
  memcpy(above_row, above_ref, 8);
  above_row[-1] = tl;
  left_col[-1] = tl;
  if (!disable_edge_filter) {
    const int da = p_angle - 90;
    const int dl = 180 - p_angle;
    const int type = ((da < 40 || da >= 48) || (dl < 40 || dl >= 48))
                         ? get_intra_edge_filter_type(xd, plane)
                         : 0;
    if (type == 0) {
      if (da >= 40) {
        av1_filter_intra_edge(above_row - 1, 9, 1);
      } else {
        upsample_above = 1;
        av1_upsample_intra_edge(above_row, 8);
      }
      if (dl >= 40) {
        av1_filter_intra_edge(left_col - 1, 9, 1);
      } else {
        upsample_left = 1;
        av1_upsample_intra_edge(left_col, 8);
      }
    } else {
      const int sa = (da >= 20) + (da >= 48);
      if (sa) av1_filter_intra_edge(above_row - 1, 9, sa);
      const int sl = (dl >= 20) + (dl >= 48);
      if (sl) av1_filter_intra_edge(left_col - 1, 9, sl);
    }
  }
  av1_dr_prediction_z2(
      dst, dst_stride, 8, 8, above_row, left_col, upsample_above, upsample_left,
      dr_intra_derivative[180 - p_angle], dr_intra_derivative[p_angle - 90]);
}

// Separate stack frame for 8x8 zone-3 edge buffers off the facade.
static AOM_NOINLINE void build_dr_z3_8x8_lbd(const MACROBLOCKD *xd,
                                             const uint8_t *ref, int ref_stride,
                                             uint8_t *dst, int dst_stride,
                                             uint64_t meta) {
  const int plane = (int)((meta >> 8) & 0xff);
  const int p_angle = (int)((meta >> 16) & 0xffff);
  assert(p_angle > 180 && p_angle < 270);
  const int disable_edge_filter = (int)((meta >> 32) & 1);
  const int n_top_nonzero = (int)((meta >> 59) & 1);
  const int yd_clamped = AOMMIN((int)((meta >> 60) & 0xf), 8);

  const uint8_t *const above_ref = ref - ref_stride;
  const uint8_t *const left_ref = ref - 1;
  const int n_bottomleft_px =
      (yd_clamped > 0 && has_bottom_left_available(xd, meta)) ? yd_clamped : 0;

  // 16-byte prefix [-16..-1], 32 upsampled edge samples [0..31], 32-byte tail
  // [32..63] (fits av1_upsample_intra_edge_sse4_1's p[-2..61] writes at sz=16).
  DECLARE_ALIGNED(16, uint8_t, left_data[16 + 32 + 32]);
  uint8_t *const left_col = left_data + 16;
  memset(left_data, 0, 16);
  memset(left_col + 16, 0, 48);
  int upsample_left = 0;

  assert(n_bottomleft_px >= 0 && n_bottomleft_px <= 8);
  gather_left_col_pow2_lbd(left_col, left_ref, ref_stride, 8);
  if (n_bottomleft_px == 8) {
    gather_left_col_pow2_lbd(left_col + 8, left_ref + (ptrdiff_t)8 * ref_stride,
                             ref_stride, 8);
  } else if (n_bottomleft_px == 0) {
    memset(left_col + 8, left_col[7], 8);
  } else {
    gather_left_col_lbd(left_col + 8, left_ref + (ptrdiff_t)8 * ref_stride,
                        ref_stride, n_bottomleft_px);
    memset(&left_col[8 + n_bottomleft_px], left_col[7 + n_bottomleft_px],
           (size_t)(8 - n_bottomleft_px));
  }
  if (!disable_edge_filter) {
    left_col[-1] = n_top_nonzero ? above_ref[-1] : left_ref[0];
    const int d = p_angle - 180;
    const int type =
        (d < 40 || d >= 48) ? get_intra_edge_filter_type(xd, plane) : 0;
    if (type == 0) {
      if (d < 40) {
        upsample_left = 1;
        av1_upsample_intra_edge(left_col, 16);
      } else {
        av1_filter_intra_edge(left_col - 1, 17, 1);
      }
    } else {
      const int s = (d >= 20) + (d >= 48);
      if (s) av1_filter_intra_edge(left_col - 1, 17, s);
    }
  }
  av1_dr_prediction_z3(dst, dst_stride, 8, 8, NULL, left_col, upsample_left, 1,
                       dr_intra_derivative[270 - p_angle]);
}

// Unified out-of-line builder for non-8x8 DR, V/H borders, and filter-intra.
static AOM_NOINLINE void build_directional_and_filter_intra_predictors(
    const MACROBLOCKD *xd, const uint8_t *ref, int ref_stride, uint8_t *dst,
    int dst_stride, uint64_t meta, uint64_t px_pack) {
  const TX_SIZE tx_size = (TX_SIZE)(meta & 0xff);
  const int plane = (int)((meta >> 8) & 0xff);
  const int p_angle = (int)((meta >> 16) & 0xffff);
  const int disable_edge_filter = (int)((meta >> 32) & 1);
  const BLOCK_SIZE sb_size = (BLOCK_SIZE)((meta >> 35) & 0xff);
  const FILTER_INTRA_MODE filter_intra_mode =
      (FILTER_INTRA_MODE)((meta >> 59) & 0x7);
  (void)sb_size;
  assert(sb_size == BLOCK_64X64 || sb_size == BLOCK_128X128);

  const int n_top_px = (int)(px_pack & 0xffff);
  const int n_left_px = (int)((px_pack >> 16) & 0xffff);
  const int xr_clamped = (int)((px_pack >> 32) & 0xffff);
  const int yd_clamped = (int)((px_pack >> 48) & 0xffff);

  const uint8_t *const above_ref = ref - ref_stride;
  const uint8_t *const left_ref = ref - 1;
  const int txwpx = tx_size_wide[tx_size];
  const int txhpx = tx_size_high[tx_size];
  const int use_filter_intra = filter_intra_mode != FILTER_INTRA_MODES;
  assert(use_filter_intra || (p_angle > 0 && p_angle < 270));
  const int need_above = use_filter_intra || (p_angle < 180);
  const int need_left = use_filter_intra || (p_angle > 90);

  assert(n_top_px >= 0);
  assert(n_left_px >= 0);

  if ((!need_above && n_left_px == 0) || (!need_left && n_top_px == 0)) {
    const uint8_t val = need_left ? ((n_top_px > 0) ? above_ref[0] : 129)
                                  : ((n_left_px > 0) ? left_ref[0] : 127);
    av1_fill_block_lbd(dst, dst_stride, val, txwpx, txhpx);
    return;
  }

  int n_topright_px = -1;
  int n_bottomleft_px = -1;
  if (!use_filter_intra) {
    if (p_angle < 90) {
      n_topright_px = (xr_clamped > 0 && has_top_right_available(xd, meta))
                          ? xr_clamped
                          : 0;
    } else if (p_angle > 180) {
      n_bottomleft_px = (yd_clamped > 0 && has_bottom_left_available(xd, meta))
                            ? yd_clamped
                            : 0;
    }
  }

  DECLARE_ALIGNED(16, uint8_t, left_data[NUM_INTRA_NEIGHBOUR_PIXELS]);
  DECLARE_ALIGNED(16, uint8_t, above_data[NUM_INTRA_NEIGHBOUR_PIXELS]);
  uint8_t *const above_row = above_data + 16;
  uint8_t *const left_col = left_data + 16;

  // NEED_LEFT
  if (need_left) {
    const int num_left_pixels_needed =
        txhpx + (n_bottomleft_px >= 0 ? txwpx : 0);
    if (n_left_px > 0) {
      const int n_left_copy = n_left_px + AOMMAX(n_bottomleft_px, 0);
      gather_left_col_lbd(left_col, left_ref, ref_stride, n_left_copy);
      if (n_left_copy < num_left_pixels_needed) {
        fill_intra_edge_lbd_tail(&left_col[n_left_copy],
                                 left_col[n_left_copy - 1],
                                 num_left_pixels_needed - n_left_copy, 127);
      }
    } else {
      const uint8_t val = (n_top_px > 0) ? above_ref[0] : 129;
      fill_intra_edge_lbd_128(left_col, val, num_left_pixels_needed);
    }
  }

  // NEED_ABOVE
  if (need_above) {
    const int num_top_pixels_needed = txwpx + (n_topright_px >= 0 ? txhpx : 0);
    if (n_top_px > 0) {
      const int n_top_copy = n_top_px + AOMMAX(n_topright_px, 0);
      assert(n_top_copy > 0);
      copy_above_row_lbd_128(above_row, above_ref, n_top_copy);
      if (n_top_copy < num_top_pixels_needed) {
        fill_intra_edge_lbd_tail(&above_row[n_top_copy],
                                 above_row[n_top_copy - 1],
                                 num_top_pixels_needed - n_top_copy, 127);
      }
    } else {
      const uint8_t val = (n_left_px > 0) ? left_ref[0] : 127;
      fill_intra_edge_lbd_128(above_row, val, num_top_pixels_needed);
    }
  }

  // Pure vertical (90 deg) and horizontal (180 deg) directional predictors do
  // not require the top-left corner pixel or intra edge filtering.
  if (!use_filter_intra) {
    if (p_angle == 90) {
      pred[V_PRED][tx_size](dst, dst_stride, above_row, left_col);
      return;
    }
    if (p_angle == 180) {
      // aom_h_predictor_32x8_sse2 loads 16 bytes from left, and 4-row
      // SSE2/SSSE3 H predictors load 8 bytes; zero tail avoids MSan reads.
      if (txhpx <= 8) memset(left_col + txhpx, 0, 8);
      pred[H_PRED][tx_size](dst, dst_stride, above_row, left_col);
      return;
    }
  }

  // Zero the 16-byte prefix, unused span, and SIMD over-read tail for
  // Valgrind/MemorySanitizer.
  const int tail_zero = (txwpx + txhpx == 128) ? 16 : 32;
  uint8_t corner;
  if (n_top_px > 0 && n_left_px > 0) {
    corner = above_ref[-1];
  } else if (n_top_px > 0) {
    corner = above_ref[0];
  } else if (n_left_px > 0) {
    corner = left_ref[0];
  } else {
    corner = 128;
  }
  if (need_above) {
    const int num_top_pixels_needed = txwpx + (n_topright_px >= 0 ? txhpx : 0);
    memset(above_data, 0, 16);
    memset(above_row + num_top_pixels_needed, 0,
           (size_t)((txwpx + txhpx + tail_zero) - num_top_pixels_needed));
    above_row[-1] = corner;
  }
  if (need_left) {
    const int num_left_pixels_needed =
        txhpx + (n_bottomleft_px >= 0 ? txwpx : 0);
    // dr_prediction_z2_WxH_neon unconditionally loads left_col[48..63] when
    // bw == 16 (including TX_16X4 and TX_16X8).
    const int left_zero_end = AOMMAX(txhpx + txwpx + tail_zero, MAX_TX_SIZE);
    memset(left_data, 0, 16);
    memset(left_col + num_left_pixels_needed, 0,
           (size_t)(left_zero_end - num_left_pixels_needed));
    left_col[-1] = corner;
  }

  if (use_filter_intra) {
    av1_filter_intra_predictor(dst, dst_stride, tx_size, above_row, left_col,
                               filter_intra_mode);
    return;
  }

  int upsample_above = 0;
  int upsample_left = 0;
  if (!disable_edge_filter) {
    const int need_right = p_angle < 90;
    const int need_bottom = p_angle > 180;
    const int intra_edge_filter_type =
        need_intra_edge_filter_type_check(txwpx, txhpx, p_angle, n_top_px,
                                          n_left_px)
            ? get_intra_edge_filter_type(xd, plane)
            : 0;
    if (need_above && need_left && (txwpx + txhpx >= 24)) {
      filter_intra_edge_corner(above_row, left_col);
    }
    if (need_above && n_top_px > 0) {
      const int strength = intra_edge_filter_strength(
          txwpx, txhpx, p_angle - 90, intra_edge_filter_type);
      if (strength) {
        const int n_px = n_top_px + 1 + (need_right ? txhpx : 0);
        av1_filter_intra_edge(above_row - 1, n_px, strength);
      }
    }
    if (need_left && n_left_px > 0) {
      const int strength = intra_edge_filter_strength(
          txhpx, txwpx, p_angle - 180, intra_edge_filter_type);
      if (strength) {
        const int n_px = n_left_px + 1 + (need_bottom ? txwpx : 0);
        av1_filter_intra_edge(left_col - 1, n_px, strength);
      }
    }
    upsample_above =
        need_above && av1_use_intra_edge_upsample(txwpx, txhpx, p_angle - 90,
                                                  intra_edge_filter_type);
    if (upsample_above) {
      av1_upsample_intra_edge(above_row, txwpx + (need_right ? txhpx : 0));
    }
    upsample_left =
        need_left && av1_use_intra_edge_upsample(txhpx, txwpx, p_angle - 180,
                                                 intra_edge_filter_type);
    if (upsample_left) {
      av1_upsample_intra_edge(left_col, txhpx + (need_bottom ? txwpx : 0));
    }
  }
  if (p_angle < 90) {
    av1_dr_prediction_z1(dst, dst_stride, txwpx, txhpx, above_row, left_col,
                         upsample_above, dr_intra_derivative[p_angle], 1);
  } else if (p_angle > 180) {
    av1_dr_prediction_z3(dst, dst_stride, txwpx, txhpx, above_row, left_col,
                         upsample_left, 1, dr_intra_derivative[270 - p_angle]);
  } else {
    av1_dr_prediction_z2(dst, dst_stride, txwpx, txhpx, above_row, left_col,
                         upsample_above, upsample_left,
                         dr_intra_derivative[180 - p_angle],
                         dr_intra_derivative[p_angle - 90]);
  }
}

static AOM_FORCE_INLINE uint32_t sum_above_row_lbd(const uint8_t *above_ref,
                                                   int txwpx) {
  uint32_t sum = 0;
  for (int i = 0; i < txwpx; ++i) sum += above_ref[i];
  return sum;
}

// Computes 8-bit DC_PRED directly from the reference boundary pixels without
// staging above_row and left_col buffers on the stack.
void av1_predict_dc_block_lbd_c(const uint8_t *ref, int ref_stride,
                                uint8_t *dst, int dst_stride, TX_SIZE tx_size,
                                int txwpx, int txhpx, int n_top_px,
                                int n_left_px) {
  assert((unsigned int)tx_size < TX_SIZES_ALL);
  assert(txwpx == tx_size_wide[tx_size] && txhpx == tx_size_high[tx_size]);
  assert(n_top_px >= 0 && n_top_px <= txwpx);
  assert(n_left_px >= 0 && n_left_px <= txhpx);
  uint32_t sum_top = 0;
  if (n_top_px > 0) {
    const uint8_t *const above_ref = ref - ref_stride;
    if (LIKELY(n_top_px == txwpx)) {
      sum_top = sum_above_row_lbd(above_ref, txwpx);
    } else {
      for (int i = 0; i < n_top_px; ++i) sum_top += above_ref[i];
      sum_top +=
          (uint32_t)above_ref[n_top_px - 1] * (uint32_t)(txwpx - n_top_px);
    }
  }

  uint32_t sum_left = 0;
  if (n_left_px > 0) {
    const uint8_t *const left_ref = ref - 1;
    if (LIKELY(n_left_px == txhpx)) {
      sum_left = av1_sum_left_col_lbd(left_ref, ref_stride, txhpx);
    } else {
      uint8_t last_val = 0;
      for (int i = 0; i < n_left_px; ++i) {
        last_val = left_ref[(ptrdiff_t)i * ref_stride];
        sum_left += last_val;
      }
      sum_left += (uint32_t)last_val * (uint32_t)(txhpx - n_left_px);
    }
  }

  const uint8_t dc_val = av1_dc_val_from_sums_lbd(
      tx_size, txwpx, txhpx, sum_top, sum_left, n_top_px, n_left_px);
  av1_fill_block_lbd(dst, dst_stride, dc_val, txwpx, txhpx);
}

// Directly broadcasts the top reference row or left reference column into dst
// for pure 90-degree (V_PRED) and 180-degree (H_PRED) blocks with width < 32
// when all boundary pixels are in-bounds. Larger widths route via RTCD
// predictors so AVX2 256-bit predictors are used when available.
static AOM_FORCE_INLINE void v_pred_direct_lbd(uint8_t *dst, int dst_stride,
                                               const uint8_t *above_ref,
                                               int txwpx, int txhpx) {
  assert(txwpx <= 16);
  assert(txhpx >= 4 && (txhpx & 1) == 0);
  if (txwpx == 4) {
    uint32_t v;
    memcpy(&v, above_ref, 4);
    do {
      memcpy(dst, &v, 4);
      memcpy(dst + dst_stride, &v, 4);
      dst += dst_stride * 2;
      txhpx -= 2;
    } while (txhpx != 0);
  } else if (txwpx == 8) {
    uint64_t v;
    memcpy(&v, above_ref, 8);
    do {
      memcpy(dst, &v, 8);
      memcpy(dst + dst_stride, &v, 8);
      dst += dst_stride * 2;
      txhpx -= 2;
    } while (txhpx != 0);
  } else {
    assert(txwpx == 16);
    uint8_t v[16];
    memcpy(v, above_ref, 16);
    do {
      memcpy(dst, v, 16);
      memcpy(dst + dst_stride, v, 16);
      dst += dst_stride * 2;
      txhpx -= 2;
    } while (txhpx != 0);
  }
}

static AOM_FORCE_INLINE void h_pred_direct_lbd(uint8_t *dst, int dst_stride,
                                               const uint8_t *left_ref,
                                               int ref_stride, int txwpx,
                                               int txhpx) {
  assert(txwpx <= 16);
  assert(txhpx >= 4 && (txhpx & 3) == 0);
  if (txwpx == 4) {
    do {
      fill_intra_edge_lbd_64(dst, *left_ref, 4);
      dst += dst_stride;
      left_ref += ref_stride;
    } while (--txhpx != 0);
  } else if (txwpx == 8) {
    do {
      fill_intra_edge_lbd_64(dst, *left_ref, 8);
      dst += dst_stride;
      left_ref += ref_stride;
    } while (--txhpx != 0);
  } else {
    assert(txwpx == 16);
    do {
      fill_intra_edge_lbd_64(dst, *left_ref, 16);
      dst += dst_stride;
      left_ref += ref_stride;
    } while (--txhpx != 0);
  }
}

// This function generates the pred data of a given block for non-directional
// intra prediction modes (i.e., DC, SMOOTH, SMOOTH_H, SMOOTH_V and PAETH).
// Out-of-line so 160B aligned stack buffers stay off the intra facade.
static AOM_NOINLINE void build_non_directional_intra_predictors(
    const uint8_t *ref, int ref_stride, uint8_t *dst, int dst_stride,
    uint32_t mode_tx, uint32_t px_pack) {
  const PREDICTION_MODE mode = (PREDICTION_MODE)(mode_tx & 0xff);
  const TX_SIZE tx_size = (TX_SIZE)((mode_tx >> 8) & 0xff);
  const int n_top_px = (int)(px_pack & 0xffff);
  const int n_left_px = (int)((px_pack >> 16) & 0xffff);
  const uint8_t *above_ref = ref - ref_stride;
  const uint8_t *left_ref = ref - 1;
  const int txwpx = tx_size_wide[tx_size];
  const int txhpx = tx_size_high[tx_size];
  assert(n_top_px >= 0);
  assert(n_left_px >= 0);
  assert(mode == DC_PRED || mode == SMOOTH_PRED || mode == SMOOTH_V_PRED ||
         mode == SMOOTH_H_PRED || mode == PAETH_PRED);

  DECLARE_ALIGNED(16, uint8_t, left_data[16 + MAX_TX_SIZE]);
  DECLARE_ALIGNED(16, uint8_t, above_data[16 + MAX_TX_SIZE]);
  uint8_t *const above_row = above_data + 16;
  uint8_t *const left_col = left_data + 16;

  if (n_left_px == txhpx) {
    gather_left_col_pow2_lbd(left_col, left_ref, ref_stride, txhpx);
  } else if (n_left_px > 0) {
    assert(n_left_px < txhpx);
    gather_left_col_lbd(left_col, left_ref, ref_stride, n_left_px);
    fill_intra_edge_lbd_tail(&left_col[n_left_px], left_col[n_left_px - 1],
                             txhpx - n_left_px, 63);
  } else if (mode != DC_PRED) {
    const uint8_t val = (n_top_px > 0) ? above_ref[0] : 129;
    fill_intra_edge_lbd_64(left_col, val, txhpx);
  }

  if (n_top_px == txwpx) {
    copy_above_row_lbd_64(above_row, above_ref, txwpx);
  } else if (n_top_px > 0) {
    assert(n_top_px < txwpx);
    copy_above_row_lbd_partial(above_row, above_ref, n_top_px, 63);
    fill_intra_edge_lbd_tail(&above_row[n_top_px], above_row[n_top_px - 1],
                             txwpx - n_top_px, 63);
  } else if (mode != DC_PRED) {
    const uint8_t val = (n_left_px > 0) ? left_ref[0] : 127;
    fill_intra_edge_lbd_64(above_row, val, txwpx);
  }

  if (mode == PAETH_PRED) {
    // aom_paeth_predictor_{4x4,8x4}_ssse3 load 8 bytes from left and above
    // before shuffling with a runtime mask; zero bytes 4..7 to avoid
    // uninitialized reads under Valgrind/MemorySanitizer (see aomedia:2752).
    if (txhpx == 4) memset(left_col + 4, 0, 4);
    if (txwpx == 4) memset(above_row + 4, 0, 4);
    if (n_top_px > 0 && n_left_px > 0) {
      above_row[-1] = above_ref[-1];
    } else if (n_top_px > 0) {
      above_row[-1] = above_ref[0];
    } else if (n_left_px > 0) {
      above_row[-1] = left_ref[0];
    } else {
      above_row[-1] = 128;
    }
    left_col[-1] = above_row[-1];
  }

  if (mode == DC_PRED) {
    dc_pred[n_left_px > 0][n_top_px > 0][tx_size](dst, dst_stride, above_row,
                                                  left_col);
  } else {
    pred[mode][tx_size](dst, dst_stride, above_row, left_col);
  }
}

#if CONFIG_AV1_HIGHBITDEPTH
void av1_highbd_filter_intra_edge_c(uint16_t *p, int sz, int strength) {
  if (!strength) return;

  const int kernel[INTRA_EDGE_FILT][INTRA_EDGE_TAPS] = { { 0, 4, 8, 4, 0 },
                                                         { 0, 5, 6, 5, 0 },
                                                         { 2, 4, 4, 4, 2 } };
  const int filt = strength - 1;
  uint16_t edge[129];

  memcpy(edge, p, sz * sizeof(*p));
  for (int i = 1; i < sz; i++) {
    int s = 0;
    for (int j = 0; j < INTRA_EDGE_TAPS; j++) {
      int k = i - 2 + j;
      k = (k < 0) ? 0 : k;
      k = (k > sz - 1) ? sz - 1 : k;
      s += edge[k] * kernel[filt][j];
    }
    s = (s + 8) >> 4;
    p[i] = s;
  }
}

static void highbd_filter_intra_edge_corner(uint16_t *p_above,
                                            uint16_t *p_left) {
  const int kernel[3] = { 5, 6, 5 };

  int s = (p_left[0] * kernel[0]) + (p_above[-1] * kernel[1]) +
          (p_above[0] * kernel[2]);
  s = (s + 8) >> 4;
  p_above[-1] = s;
  p_left[-1] = s;
}

void av1_highbd_upsample_intra_edge_c(uint16_t *p, int sz, int bd) {
  // interpolate half-sample positions
  assert(sz <= MAX_UPSAMPLE_SZ);

  uint16_t in[MAX_UPSAMPLE_SZ + 3];
  // copy p[-1..(sz-1)] and extend first and last samples
  in[0] = p[-1];
  in[1] = p[-1];
  for (int i = 0; i < sz; i++) {
    in[i + 2] = p[i];
  }
  in[sz + 2] = p[sz - 1];

  // interpolate half-sample edge positions
  p[-2] = in[0];
  for (int i = 0; i < sz; i++) {
    int s = -in[i] + (9 * in[i + 1]) + (9 * in[i + 2]) - in[i + 3];
    s = (s + 8) >> 4;
    s = clip_pixel_highbd(s, bd);
    p[2 * i - 1] = s;
    p[2 * i] = in[i + 2];
  }
}

static AOM_FORCE_INLINE void copy_above_row_hbd(uint16_t *above_row,
                                                const uint16_t *above_ref,
                                                int n_copy) {
  if (n_copy == 4) {
    memcpy(above_row, above_ref, 4 * sizeof(uint16_t));
  } else if (n_copy == 8) {
    memcpy(above_row, above_ref, 8 * sizeof(uint16_t));
  } else if (n_copy == 16) {
    memcpy(above_row, above_ref, 16 * sizeof(uint16_t));
  } else if (n_copy == 32) {
    memcpy(above_row, above_ref, 32 * sizeof(uint16_t));
  } else if (n_copy == 64) {
    memcpy(above_row, above_ref, 64 * sizeof(uint16_t));
  } else if (n_copy == 128) {
    memcpy(above_row, above_ref, 128 * sizeof(uint16_t));
  } else {
    memcpy(above_row, above_ref, n_copy * sizeof(uint16_t));
  }
}

// Out-of-line HBD builder so low-bitdepth callers avoid HBD stack frames.
static AOM_NOINLINE void highbd_build_directional_and_filter_intra_predictors(
    const MACROBLOCKD *xd, const uint8_t *ref8, int ref_stride, uint8_t *dst8,
    int dst_stride, uint64_t meta, uint64_t px_pack) {
  const TX_SIZE tx_size = (TX_SIZE)(meta & 0xff);
  const int plane = (int)((meta >> 8) & 0xff);
  const int p_angle = (int)((meta >> 16) & 0xffff);
  const int disable_edge_filter = (int)((meta >> 32) & 1);
  const BLOCK_SIZE sb_size = (BLOCK_SIZE)((meta >> 35) & 0xff);
  const FILTER_INTRA_MODE filter_intra_mode =
      (FILTER_INTRA_MODE)((meta >> 59) & 0x7);
  (void)sb_size;
  assert(sb_size == BLOCK_64X64 || sb_size == BLOCK_128X128);

  const int n_top_px = (int)(px_pack & 0xffff);
  const int n_left_px = (int)((px_pack >> 16) & 0xffff);
  const int xr_clamped = (int)((px_pack >> 32) & 0xffff);
  const int yd_clamped = (int)((px_pack >> 48) & 0xffff);

  int i;
  uint16_t *dst = CONVERT_TO_SHORTPTR(dst8);
  const uint16_t *const ref = CONVERT_TO_SHORTPTR(ref8);
  const int txwpx = tx_size_wide[tx_size];
  const int txhpx = tx_size_high[tx_size];
  const uint16_t *above_ref = ref - ref_stride;
  const uint16_t *left_ref = ref - 1;
  const int use_filter_intra = filter_intra_mode != FILTER_INTRA_MODES;
  assert(use_filter_intra || (p_angle > 0 && p_angle < 270));
  const int bit_depth = xd->bd;
  const int base = 128 << (bit_depth - 8);
  const int need_above = use_filter_intra || (p_angle < 180);
  const int need_left = use_filter_intra || (p_angle > 90);

  assert(n_top_px >= 0);
  assert(n_left_px >= 0);

  if ((!need_above && n_left_px == 0) || (!need_left && n_top_px == 0)) {
    int val;
    if (need_left) {
      val = (n_top_px > 0) ? above_ref[0] : base + 1;
    } else {
      val = (n_left_px > 0) ? left_ref[0] : base - 1;
    }
    for (i = 0; i < txhpx; ++i) {
      aom_memset16(dst, val, txwpx);
      dst += dst_stride;
    }
    return;
  }

  int n_topright_px = -1;
  int n_bottomleft_px = -1;
  if (!use_filter_intra) {
    if (p_angle < 90) {
      n_topright_px = (xr_clamped > 0 && has_top_right_available(xd, meta))
                          ? xr_clamped
                          : 0;
    } else if (p_angle > 180) {
      n_bottomleft_px = (yd_clamped > 0 && has_bottom_left_available(xd, meta))
                            ? yd_clamped
                            : 0;
    }
  }

  DECLARE_ALIGNED(16, uint16_t, left_data[NUM_INTRA_NEIGHBOUR_PIXELS]);
  DECLARE_ALIGNED(16, uint16_t, above_data[NUM_INTRA_NEIGHBOUR_PIXELS]);
  uint16_t *const above_row = above_data + 16;
  uint16_t *const left_col = left_data + 16;

  // NEED_LEFT
  if (need_left) {
    const int num_left_pixels_needed =
        txhpx + (n_bottomleft_px >= 0 ? txwpx : 0);
    if (n_left_px > 0) {
      const int n_left_copy =
          n_left_px + (n_bottomleft_px > 0 ? n_bottomleft_px : 0);
      assert(n_left_copy > 0);
      i = 0;
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && \
    __GNUC__ >= 14
#pragma GCC novector
#endif
      do {
        left_col[i] = left_ref[(ptrdiff_t)i * ref_stride];
      } while (++i != n_left_copy);
      if (i < num_left_pixels_needed)
        aom_memset16(&left_col[i], left_col[i - 1], num_left_pixels_needed - i);
    } else {
      const int val = (n_top_px > 0) ? above_ref[0] : base + 1;
      aom_memset16(left_col, val, num_left_pixels_needed);
    }
  }

  // NEED_ABOVE
  if (need_above) {
    const int num_top_pixels_needed = txwpx + (n_topright_px >= 0 ? txhpx : 0);
    if (n_top_px > 0) {
      const int n_top_copy = n_top_px + (n_topright_px > 0 ? n_topright_px : 0);
      copy_above_row_hbd(above_row, above_ref, n_top_copy);
      if (n_top_copy < num_top_pixels_needed)
        aom_memset16(&above_row[n_top_copy], above_row[n_top_copy - 1],
                     num_top_pixels_needed - n_top_copy);
    } else {
      const int val = (n_left_px > 0) ? left_ref[0] : base - 1;
      aom_memset16(above_row, val, num_top_pixels_needed);
    }
  }

  // Pure vertical (90 deg) and horizontal (180 deg) directional predictors do
  // not require the top-left corner pixel or intra edge filtering.
  if (!use_filter_intra) {
    if (p_angle == 90) {
      pred_high[V_PRED][tx_size](dst, dst_stride, above_row, left_col,
                                 bit_depth);
      return;
    }
    if (p_angle == 180) {
      // aom_highbd_h_predictor_8x4_sse2 loads 16 bytes from left; zero the
      // unused 4-sample tail to avoid uninitialized stack reads.
      if (txhpx == 4) memset(left_col + 4, 0, 4 * sizeof(left_col[0]));
      pred_high[H_PRED][tx_size](dst, dst_stride, above_row, left_col,
                                 bit_depth);
      return;
    }
  }

  const int tail_zero = (txwpx + txhpx == 128) ? 16 : 32;
  uint16_t corner;
  if (n_top_px > 0 && n_left_px > 0) {
    corner = above_ref[-1];
  } else if (n_top_px > 0) {
    corner = above_ref[0];
  } else if (n_left_px > 0) {
    corner = left_ref[0];
  } else {
    corner = (uint16_t)base;
  }
  if (need_above) {
    const int num_top_pixels_needed = txwpx + (n_topright_px >= 0 ? txhpx : 0);
    assert(num_top_pixels_needed >= 0 &&
           num_top_pixels_needed <= txwpx + txhpx);
    memset(above_data, 0, 16 * sizeof(above_data[0]));
    memset(above_row + num_top_pixels_needed, 0,
           (size_t)((txwpx + txhpx + tail_zero) - num_top_pixels_needed) *
               sizeof(above_row[0]));
    above_row[-1] = corner;
  }
  if (need_left) {
    const int num_left_pixels_needed =
        txhpx + (n_bottomleft_px >= 0 ? txwpx : 0);
    assert(num_left_pixels_needed >= 0 &&
           num_left_pixels_needed <= txhpx + txwpx);
    memset(left_data, 0, 16 * sizeof(left_data[0]));
    memset(left_col + num_left_pixels_needed, 0,
           (size_t)((txhpx + txwpx + tail_zero) - num_left_pixels_needed) *
               sizeof(left_col[0]));
    left_col[-1] = corner;
  }

  if (use_filter_intra) {
    highbd_filter_intra_predictor(dst, dst_stride, tx_size, above_row, left_col,
                                  filter_intra_mode, bit_depth);
    return;
  }

  assert(p_angle > 0 && p_angle < 270);
  int upsample_above = 0;
  int upsample_left = 0;
  if (!disable_edge_filter) {
    const int need_right = p_angle < 90;
    const int need_bottom = p_angle > 180;
    const int intra_edge_filter_type =
        need_intra_edge_filter_type_check(txwpx, txhpx, p_angle, n_top_px,
                                          n_left_px)
            ? get_intra_edge_filter_type(xd, plane)
            : 0;
    if (need_above && need_left && (txwpx + txhpx >= 24)) {
      highbd_filter_intra_edge_corner(above_row, left_col);
    }
    if (need_above && n_top_px > 0) {
      const int strength = intra_edge_filter_strength(
          txwpx, txhpx, p_angle - 90, intra_edge_filter_type);
      if (strength) {
        const int n_px = n_top_px + 1 + (need_right ? txhpx : 0);
        av1_highbd_filter_intra_edge(above_row - 1, n_px, strength);
      }
    }
    if (need_left && n_left_px > 0) {
      const int strength = intra_edge_filter_strength(
          txhpx, txwpx, p_angle - 180, intra_edge_filter_type);
      if (strength) {
        const int n_px = n_left_px + 1 + (need_bottom ? txwpx : 0);
        av1_highbd_filter_intra_edge(left_col - 1, n_px, strength);
      }
    }
    upsample_above =
        need_above && av1_use_intra_edge_upsample(txwpx, txhpx, p_angle - 90,
                                                  intra_edge_filter_type);
    if (upsample_above) {
      const int n_px = txwpx + (need_right ? txhpx : 0);
      av1_highbd_upsample_intra_edge(above_row, n_px, bit_depth);
    }
    upsample_left =
        need_left && av1_use_intra_edge_upsample(txhpx, txwpx, p_angle - 180,
                                                 intra_edge_filter_type);
    if (upsample_left) {
      const int n_px = txhpx + (need_bottom ? txwpx : 0);
      av1_highbd_upsample_intra_edge(left_col, n_px, bit_depth);
    }
  }
  if (p_angle < 90) {
    av1_highbd_dr_prediction_z1(dst, dst_stride, txwpx, txhpx, above_row,
                                left_col, upsample_above,
                                dr_intra_derivative[p_angle], 1, bit_depth);
  } else if (p_angle > 180) {
    av1_highbd_dr_prediction_z3(dst, dst_stride, txwpx, txhpx, above_row,
                                left_col, upsample_left, 1,
                                dr_intra_derivative[270 - p_angle], bit_depth);
  } else {
    av1_highbd_dr_prediction_z2(dst, dst_stride, txwpx, txhpx, above_row,
                                left_col, upsample_above, upsample_left,
                                dr_intra_derivative[180 - p_angle],
                                dr_intra_derivative[p_angle - 90], bit_depth);
  }
}

// For HBD encode/decode, this function generates the pred data of a given
// block for non-directional intra prediction modes (i.e., DC, SMOOTH, SMOOTH_H,
// SMOOTH_V and PAETH). Out-of-line so low-bitdepth paths avoid HBD frames.
static AOM_NOINLINE void highbd_build_non_directional_intra_predictors(
    const uint8_t *ref8, int ref_stride, uint8_t *dst8, int dst_stride,
    uint32_t mode_tx_bd, uint32_t px_pack) {
  const PREDICTION_MODE mode = (PREDICTION_MODE)(mode_tx_bd & 0xff);
  const TX_SIZE tx_size = (TX_SIZE)((mode_tx_bd >> 8) & 0xff);
  const int bit_depth = (int)((mode_tx_bd >> 16) & 0xff);
  const int n_top_px = (int)(px_pack & 0xffff);
  const int n_left_px = (int)((px_pack >> 16) & 0xffff);
  int i = 0;
  uint16_t *dst = CONVERT_TO_SHORTPTR(dst8);
  const uint16_t *const ref = CONVERT_TO_SHORTPTR(ref8);
  const int txwpx = tx_size_wide[tx_size];
  const int txhpx = tx_size_high[tx_size];
  int need_left = extend_modes[mode] & NEED_LEFT;
  int need_above = extend_modes[mode] & NEED_ABOVE;
  int need_above_left = extend_modes[mode] & NEED_ABOVELEFT;
  const uint16_t *above_ref = ref - ref_stride;
  const uint16_t *left_ref = ref - 1;
  const int base = 128 << (bit_depth - 8);

  assert(n_top_px >= 0);
  assert(n_left_px >= 0);
  assert(mode == DC_PRED || mode == SMOOTH_PRED || mode == SMOOTH_V_PRED ||
         mode == SMOOTH_H_PRED || mode == PAETH_PRED);

  if ((!need_above && n_left_px == 0) || (!need_left && n_top_px == 0)) {
    int val = 0;
    if (need_left) {
      val = (n_top_px > 0) ? above_ref[0] : base + 1;
    } else {
      val = (n_left_px > 0) ? left_ref[0] : base - 1;
    }
    for (i = 0; i < txhpx; ++i) {
      aom_memset16(dst, val, txwpx);
      dst += dst_stride;
    }
    return;
  }

  DECLARE_ALIGNED(16, uint16_t, left_data[NUM_INTRA_NEIGHBOUR_PIXELS]);
  DECLARE_ALIGNED(16, uint16_t, above_data[NUM_INTRA_NEIGHBOUR_PIXELS]);
  uint16_t *const above_row = above_data + 16;
  uint16_t *const left_col = left_data + 16;

  if (need_left) {
    if (n_left_px > 0) {
      i = 0;
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && \
    __GNUC__ >= 14
#pragma GCC novector
#endif
      do {
        left_col[i] = left_ref[(ptrdiff_t)i * ref_stride];
      } while (++i != n_left_px);
      if (i < txhpx) aom_memset16(&left_col[i], left_col[i - 1], txhpx - i);
    } else {
      const int val = (n_top_px > 0) ? above_ref[0] : base + 1;
      aom_memset16(left_col, val, txhpx);
    }
  }

  if (need_above) {
    if (LIKELY(n_top_px == txwpx)) {
      copy_above_row_hbd(above_row, above_ref, txwpx);
    } else if (n_top_px > 0) {
      copy_above_row_hbd(above_row, above_ref, n_top_px);
      aom_memset16(&above_row[n_top_px], above_row[n_top_px - 1],
                   txwpx - n_top_px);
    } else {
      const int val = (n_left_px > 0) ? left_ref[0] : base - 1;
      aom_memset16(above_row, val, txwpx);
    }
  }

  if (need_above_left) {
    if (n_top_px > 0 && n_left_px > 0) {
      above_row[-1] = above_ref[-1];
    } else if (n_top_px > 0) {
      above_row[-1] = above_ref[0];
    } else if (n_left_px > 0) {
      above_row[-1] = left_ref[0];
    } else {
      above_row[-1] = base;
    }
    left_col[-1] = above_row[-1];
  }

  if (mode == DC_PRED) {
    dc_pred_high[n_left_px > 0][n_top_px > 0][tx_size](
        dst, dst_stride, above_row, left_col, bit_depth);
  } else {
    pred_high[mode][tx_size](dst, dst_stride, above_row, left_col, bit_depth);
  }
}
#endif  // CONFIG_AV1_HIGHBITDEPTH

// Out-of-line rare palette path to keep the intra facade stackless.
static AOM_NOINLINE void predict_palette_block(const MACROBLOCKD *xd, int wpx,
                                               uint8_t *dst, int dst_stride,
                                               int plane, int txwpx, int txhpx,
                                               int x, int y, bool is_hbd) {
  const MB_MODE_INFO *const mbmi = xd->mi[0];
  const uint8_t *map_row = xd->plane[plane != 0].color_index_map +
                           xd->color_index_map_offset[plane != 0] +
                           (ptrdiff_t)y * wpx + x;
  const uint16_t *const palette =
      mbmi->palette_mode_info.palette_colors + plane * PALETTE_MAX_SIZE;
  assert(txhpx > 0 && txwpx > 0);
#if CONFIG_AV1_HIGHBITDEPTH
  if (is_hbd) {
    uint16_t *dst16 = CONVERT_TO_SHORTPTR(dst);
    int r = txhpx;
    for (;;) {
      int c = 0;
      do {
        dst16[c] = palette[map_row[c]];
      } while (++c != txwpx);
      if (--r == 0) break;
      dst16 += dst_stride;
      map_row += wpx;
    }
    return;
  }
#else
  (void)is_hbd;
#endif  // CONFIG_AV1_HIGHBITDEPTH
  int r = txhpx;
  for (;;) {
    int c = 0;
    do {
      dst[c] = (uint8_t)palette[map_row[c]];
    } while (++c != txwpx);
    if (--r == 0) break;
    dst += dst_stride;
    map_row += wpx;
  }
}

// Cross-file alignment invariant:
// - txwpx >= 16: aom_{v,dc*}_predictor_{16,32,64}x*_sse2 use aligned
//   _mm_load_si128 / movdqa loads from `above`, requiring 16-byte alignment.
// - txwpx == 4: aom_v_predictor_4x*_sse2 dereference `*(uint32_t *)above` and
//   ARM NEON load_u8_4x1() requires 4-byte alignment under
//   -fsanitize=alignment.
// - txwpx == 8: SSE2/NEON predictors use unaligned 64-bit loads.
static AOM_FORCE_INLINE int is_above_aligned(const uint8_t *above, int txwpx) {
  const uintptr_t above_align_mask =
      (txwpx >= 16) ? 15u : ((txwpx == 4) ? 3u : 0u);
  return ((uintptr_t)above & above_align_mask) == 0;
}

// Separate stack frame for left_col so interior DC stays off the facade.
static AOM_NOINLINE void predict_dc_interior_lbd(const uint8_t *ref,
                                                 int ref_stride, uint8_t *dst,
                                                 int dst_stride,
                                                 TX_SIZE tx_size) {
  assert(is_above_aligned(ref - ref_stride, tx_size_wide[tx_size]));
  const int txhpx = tx_size_high[tx_size];
  DECLARE_ALIGNED(16, uint8_t, left_col[MAX_TX_SIZE]);
  gather_left_col_pow2_lbd(left_col, ref - 1, ref_stride, txhpx);
  dc_pred[1][1][tx_size](dst, dst_stride, ref - ref_stride, left_col);
}

// Separate stack frame for left_col so filter-intra stays off the facade.
static AOM_NOINLINE void predict_filter_intra_interior_lbd(
    const uint8_t *ref, int ref_stride, uint8_t *dst, int dst_stride,
    TX_SIZE tx_size, FILTER_INTRA_MODE filter_intra_mode) {
  const int txwpx = tx_size_wide[tx_size];
  const int txhpx = tx_size_high[tx_size];
  assert(txwpx <= 32 && txhpx <= 32);
  // 64B matches siblings and avoids Clang 4x unroll bloat and GCC -DNDEBUG
  // -Wstringop-overflow warnings without capping `txhpx`.
  DECLARE_ALIGNED(16, uint8_t, left_col[MAX_TX_SIZE]);
  gather_left_col_pow2_lbd(left_col, ref - 1, ref_stride, txhpx);
  const uint8_t *const above_ref = ref - ref_stride;
  // Stage the 5 valid bytes above[-1..3] defensively for 4-wide blocks so SIMD
  // callers never read past a 4-pixel right tile or frame edge.
  if (txwpx == 4) {
    DECLARE_ALIGNED(16, uint8_t, above_buf[16]) = { 0 };
    memcpy(above_buf, above_ref - 1, 5);
    av1_filter_intra_predictor(dst, dst_stride, tx_size, above_buf + 1,
                               left_col, filter_intra_mode);
    return;
  }
  av1_filter_intra_predictor(dst, dst_stride, tx_size, above_ref, left_col,
                             filter_intra_mode);
}

// Separate stack frame for left_col so H_PRED stays off the facade.
static AOM_NOINLINE void predict_h_interior_lbd(const uint8_t *ref,
                                                int ref_stride, uint8_t *dst,
                                                int dst_stride,
                                                TX_SIZE tx_size) {
  const int txhpx = tx_size_high[tx_size];
  assert(tx_size_wide[tx_size] >= 32);
  assert(txhpx >= 8);
  DECLARE_ALIGNED(16, uint8_t, left_col[MAX_TX_SIZE]);
  // aom_h_predictor_32x8_sse2 loads 16 bytes from `left`; zero bytes 8..15 to
  // avoid MSan reads.
  if (txhpx == 8) memset(left_col + 8, 0, 8);
  gather_left_col_pow2_lbd(left_col, ref - 1, ref_stride, txhpx);
  pred[H_PRED][tx_size](dst, dst_stride, NULL, left_col);
}

static inline uint8_t *get_intra_dst_ptr(const struct macroblockd_plane *pd,
                                         int blk_col, int blk_row) {
  const int dst_stride = pd->dst.stride;
  return &pd->dst
              .buf[((ptrdiff_t)blk_row * dst_stride + blk_col) << MI_SIZE_LOG2];
}

// Dual-source sequence/block dispatch: when `has_seq_params` is true (facade
// callers), `seq_params->sb_size`, `seq_params->enable_intra_edge_filter`, and
// `mbmi->angle_delta` are loaded lazily after the fast paths return, and the
// `*_arg` parameters are ignored. When `has_seq_params` is false
// (`av1_predict_intra_block`), the `*_arg` parameters are used and
// `seq_params` is NULL.
static AOM_FORCE_INLINE void predict_intra_block_internal(
    const MACROBLOCKD *xd, bool is_hbd, bool has_seq_params,
    const SequenceHeader *seq_params, BLOCK_SIZE sb_size_arg,
    int enable_intra_edge_filter_arg, int wpx, int hpx, TX_SIZE tx_size,
    PREDICTION_MODE mode, int angle_delta_arg, int use_palette,
    FILTER_INTRA_MODE filter_intra_mode, const uint8_t *ref, int ref_stride,
    uint8_t *dst, int dst_stride, int col_off, int row_off, int plane) {
  assert((unsigned)tx_size < TX_SIZES_ALL);
  assert((unsigned)plane < MAX_MB_PLANE);
  assert(mode < INTRA_MODES);

  const MB_MODE_INFO *const mbmi = xd->mi[0];
  const int txwpx = tx_size_wide[tx_size];
  const int txhpx = tx_size_high[tx_size];
  const int x = col_off << MI_SIZE_LOG2;
  const int y = row_off << MI_SIZE_LOG2;

  if (UNLIKELY(use_palette)) {
    predict_palette_block(xd, wpx, dst, dst_stride, plane, txwpx, txhpx, x, y,
                          is_hbd);
    return;
  }

  const struct macroblockd_plane *const pd = &xd->plane[plane];
  // Luma subsampling is always 0; branching on `plane == AOM_PLANE_Y` enables
  // constant folding in the luma facade.
  const int ss_x = (plane == AOM_PLANE_Y) ? 0 : pd->subsampling_x;
  const int ss_y = (plane == AOM_PLANE_Y) ? 0 : pd->subsampling_y;
  const int have_top =
      row_off || (ss_y ? xd->chroma_up_available : xd->up_available);
  const int have_left =
      col_off || (ss_x ? xd->chroma_left_available : xd->left_available);

  // Distance between the right edge of this prediction block to
  // the frame right edge
  const int xr = (xd->mb_to_right_edge >> (3 + ss_x)) + wpx - x - txwpx;
  // Distance between the bottom edge of this prediction block to
  // the frame bottom edge
  const int yd = (xd->mb_to_bottom_edge >> (3 + ss_y)) + hpx - y - txhpx;
  const int n_top_px = have_top ? clamp(xr + txwpx, 0, txwpx) : 0;
  const int n_left_px = have_left ? clamp(yd + txhpx, 0, txhpx) : 0;

  if (filter_intra_mode != FILTER_INTRA_MODES) {
    if (!is_hbd && n_top_px == txwpx && n_left_px == txhpx) {
      predict_filter_intra_interior_lbd(ref, ref_stride, dst, dst_stride,
                                        tx_size, filter_intra_mode);
      return;
    }
    const BLOCK_SIZE sb_size =
        has_seq_params ? seq_params->sb_size : sb_size_arg;
    const uint64_t dr_meta =
        pack_dr_facade_meta(tx_size, plane, 0, 0, ss_x, ss_y, sb_size, col_off,
                            row_off, filter_intra_mode);
    const uint64_t dr_px = pack_dr_facade_px(n_top_px, n_left_px, 0, 0);
#if CONFIG_AV1_HIGHBITDEPTH
    if (is_hbd) {
      highbd_build_directional_and_filter_intra_predictors(
          xd, ref, ref_stride, dst, dst_stride, dr_meta, dr_px);
      return;
    }
#endif
    build_directional_and_filter_intra_predictors(xd, ref, ref_stride, dst,
                                                  dst_stride, dr_meta, dr_px);
    return;
  }

  if (!av1_is_directional_mode(mode)) {
#if CONFIG_AV1_HIGHBITDEPTH
    if (is_hbd) {
      highbd_build_non_directional_intra_predictors(
          ref, ref_stride, dst, dst_stride,
          pack_non_dir_meta(mode, tx_size, xd->bd),
          pack_non_dir_px(n_top_px, n_left_px));
      return;
    }
#endif  // CONFIG_AV1_HIGHBITDEPTH
#if HAVE_SSE2
    // Keep RTCD dc_pred for non-SSE2/NEON targets and wide >= 32 AVX2 blocks.
    if (mode == DC_PRED && txwpx < 32) {
      av1_predict_dc_block_lbd(ref, ref_stride, dst, dst_stride, tx_size, txwpx,
                               txhpx, n_top_px, n_left_px);
      return;
    }
#endif  // HAVE_SSE2
    if (mode == DC_PRED && n_top_px == txwpx &&
        is_above_aligned(ref - ref_stride, txwpx)) {
      if (n_left_px == txhpx) {
        predict_dc_interior_lbd(ref, ref_stride, dst, dst_stride, tx_size);
        return;
      }
      if (n_left_px == 0) {
        dc_pred[0][1][tx_size](dst, dst_stride, ref - ref_stride, NULL);
        return;
      }
    }
    build_non_directional_intra_predictors(
        ref, ref_stride, dst, dst_stride, pack_non_dir_meta(mode, tx_size, 0),
        pack_non_dir_px(n_top_px, n_left_px));
    return;
  }

  const int angle_delta =
      has_seq_params ? mbmi->angle_delta[plane != AOM_PLANE_Y] * ANGLE_STEP
                     : angle_delta_arg;
  const int p_angle = mode_to_angle_map[mode] + angle_delta;
  if (!is_hbd) {
    if (p_angle == 90 && LIKELY(n_top_px == txwpx) && txwpx < 32) {
      v_pred_direct_lbd(dst, dst_stride, ref - ref_stride, txwpx, txhpx);
      return;
    }
    if (p_angle == 180 && LIKELY(n_left_px == txhpx) && txwpx < 32) {
      h_pred_direct_lbd(dst, dst_stride, ref - 1, ref_stride, txwpx, txhpx);
      return;
    }
    if (p_angle == 90 && n_top_px == txwpx &&
        is_above_aligned(ref - ref_stride, txwpx)) {
      pred[V_PRED][tx_size](dst, dst_stride, ref - ref_stride, NULL);
      return;
    }
    if (p_angle == 180 && n_left_px == txhpx) {
      predict_h_interior_lbd(ref, ref_stride, dst, dst_stride, tx_size);
      return;
    }
  }
  const int disable_edge_filter =
      !(has_seq_params ? seq_params->enable_intra_edge_filter
                       : enable_intra_edge_filter_arg);
  if (!is_hbd && tx_size == TX_8X8 && p_angle > 90 && p_angle < 180 &&
      n_top_px == 8 && n_left_px == 8) {
    build_dr_z2_8x8_lbd(
        xd, ref, ref_stride, dst, dst_stride,
        pack_dr_z2_8x8_meta(plane, p_angle, disable_edge_filter));
    return;
  }

  const BLOCK_SIZE sb_size = has_seq_params ? seq_params->sb_size : sb_size_arg;
  const int xr_clamped = (n_top_px == txwpx && xr > 0) ? AOMMIN(xr, txwpx) : 0;
  const int yd_clamped = (n_left_px == txhpx && yd > 0) ? AOMMIN(yd, txhpx) : 0;

#if CONFIG_AV1_HIGHBITDEPTH
  if (is_hbd) {
    const uint64_t dr_meta = pack_dr_facade_meta(
        tx_size, plane, p_angle, disable_edge_filter, ss_x, ss_y, sb_size,
        col_off, row_off, FILTER_INTRA_MODES);
    const uint64_t dr_px =
        pack_dr_facade_px(n_top_px, n_left_px, xr_clamped, yd_clamped);
    highbd_build_directional_and_filter_intra_predictors(
        xd, ref, ref_stride, dst, dst_stride, dr_meta, dr_px);
    return;
  }
#endif
  if (p_angle < 90 && tx_size == TX_8X8 && n_top_px == 8) {
    build_dr_z1_8x8_lbd(
        xd, ref, ref_stride, dst, dst_stride,
        pack_dr_8x8_meta(plane, p_angle, disable_edge_filter, ss_x, ss_y,
                         sb_size, col_off, row_off, n_left_px > 0, xr_clamped));
    return;
  }
  if (p_angle > 180 && tx_size == TX_8X8 && n_left_px == 8) {
    build_dr_z3_8x8_lbd(
        xd, ref, ref_stride, dst, dst_stride,
        pack_dr_8x8_meta(plane, p_angle, disable_edge_filter, ss_x, ss_y,
                         sb_size, col_off, row_off, n_top_px > 0, yd_clamped));
    return;
  }
  const uint64_t dr_meta =
      pack_dr_facade_meta(tx_size, plane, p_angle, disable_edge_filter, ss_x,
                          ss_y, sb_size, col_off, row_off, FILTER_INTRA_MODES);
  const uint64_t dr_px =
      pack_dr_facade_px(n_top_px, n_left_px, xr_clamped, yd_clamped);
  build_directional_and_filter_intra_predictors(xd, ref, ref_stride, dst,
                                                dst_stride, dr_meta, dr_px);
}

void av1_predict_intra_block(const MACROBLOCKD *xd, BLOCK_SIZE sb_size,
                             int enable_intra_edge_filter, int wpx, int hpx,
                             TX_SIZE tx_size, PREDICTION_MODE mode,
                             int angle_delta, int use_palette,
                             FILTER_INTRA_MODE filter_intra_mode,
                             const uint8_t *ref, int ref_stride, uint8_t *dst,
                             int dst_stride, int col_off, int row_off,
                             int plane) {
  predict_intra_block_internal(
      xd, is_cur_buf_hbd(xd), false, NULL, sb_size, enable_intra_edge_filter,
      wpx, hpx, tx_size, mode, angle_delta, use_palette, filter_intra_mode, ref,
      ref_stride, dst, dst_stride, col_off, row_off, plane);
}

#if !CONFIG_REALTIME_ONLY || CONFIG_AV1_DECODER
// Cold chroma-from-luma path split out to keep the intra facade minimal.
static AOM_NOINLINE void predict_intra_block_cfl(MACROBLOCKD *xd, int plane,
                                                 int blk_col, int blk_row,
                                                 TX_SIZE tx_size) {
  const MB_MODE_INFO *const mbmi = xd->mi[0];
  const struct macroblockd_plane *const pd = &xd->plane[plane];
  const int dst_stride = pd->dst.stride;
  uint8_t *const dst = get_intra_dst_ptr(pd, blk_col, blk_row);
#if CONFIG_DEBUG
  assert(is_cfl_allowed(xd));
  const BLOCK_SIZE plane_bsize =
      get_plane_block_size(mbmi->bsize, pd->subsampling_x, pd->subsampling_y);
  (void)plane_bsize;
  assert(plane_bsize < BLOCK_SIZES_ALL);
  if (!xd->lossless[mbmi->segment_id]) {
    assert(blk_col == 0);
    assert(blk_row == 0);
    assert(block_size_wide[plane_bsize] == tx_size_wide[tx_size]);
    assert(block_size_high[plane_bsize] == tx_size_high[tx_size]);
  }
#endif
  CFL_CTX *const cfl = &xd->cfl;
  const CFL_PRED_TYPE pred_plane = get_cfl_pred_type(plane);
  if (!cfl->dc_pred_is_cached[pred_plane]) {
    const int use_palette = mbmi->palette_mode_info.palette_size[1] > 0;
    av1_predict_intra_block(xd, BLOCK_INVALID, 0, pd->width, pd->height,
                            tx_size, DC_PRED, 0, use_palette,
                            FILTER_INTRA_MODES, dst, dst_stride, dst,
                            dst_stride, blk_col, blk_row, plane);
    if (cfl->use_dc_pred_cache) {
      cfl_store_dc_pred(xd, dst, pred_plane, tx_size_wide[tx_size]);
      cfl->dc_pred_is_cached[pred_plane] = true;
    }
  } else {
    cfl_load_dc_pred(xd, dst, dst_stride, tx_size, pred_plane);
  }
  av1_cfl_predict_block(xd, dst, dst_stride, tx_size, plane);
}
#endif  // !CONFIG_REALTIME_ONLY || CONFIG_AV1_DECODER

// Out-of-line palette dispatch shared between luma and chroma facades.
static AOM_NOINLINE void predict_intra_block_facade_palette(
    const MACROBLOCKD *xd, int plane, int blk_col, int blk_row,
    TX_SIZE tx_size) {
  const struct macroblockd_plane *const pd = &xd->plane[plane];
  const int dst_stride = pd->dst.stride;
  uint8_t *const dst = get_intra_dst_ptr(pd, blk_col, blk_row);
  predict_palette_block(xd, pd->width, dst, dst_stride, plane,
                        tx_size_wide[tx_size], tx_size_high[tx_size],
                        blk_col << MI_SIZE_LOG2, blk_row << MI_SIZE_LOG2,
                        is_cur_buf_hbd(xd) != 0);
}

// Out-of-line filter-intra dispatch so the hot luma facade drops filter-intra
// branches and register pressure (especially in realtime where it is unused).
static AOM_NOINLINE void predict_intra_block_facade_filter_intra(
    const AV1_COMMON *cm, const MACROBLOCKD *xd, int blk_col, int blk_row,
    TX_SIZE tx_size) {
  assert(!is_cur_buf_hbd(xd));
  const FILTER_INTRA_MODE filter_intra_mode =
      xd->mi[0]->filter_intra_mode_info.filter_intra_mode;
  assert((unsigned)filter_intra_mode < FILTER_INTRA_MODES);
  const struct macroblockd_plane *const pd = &xd->plane[AOM_PLANE_Y];
  const int dst_stride = pd->dst.stride;
  uint8_t *const dst = get_intra_dst_ptr(pd, blk_col, blk_row);
  const int txwpx = tx_size_wide[tx_size];
  const int txhpx = tx_size_high[tx_size];
  const int x = blk_col << MI_SIZE_LOG2;
  const int y = blk_row << MI_SIZE_LOG2;
  const int n_top_px =
      (blk_row || xd->up_available)
          ? clamp((xd->mb_to_right_edge >> 3) + pd->width - x, 0, txwpx)
          : 0;
  const int n_left_px =
      (blk_col || xd->left_available)
          ? clamp((xd->mb_to_bottom_edge >> 3) + pd->height - y, 0, txhpx)
          : 0;
  if (n_top_px == txwpx && n_left_px == txhpx) {
    predict_filter_intra_interior_lbd(dst, dst_stride, dst, dst_stride, tx_size,
                                      filter_intra_mode);
    return;
  }
  const uint64_t dr_meta = pack_dr_facade_meta(tx_size, AOM_PLANE_Y, 0, 0, 0, 0,
                                               cm->seq_params->sb_size, blk_col,
                                               blk_row, filter_intra_mode);
  const uint64_t dr_px = pack_dr_facade_px(n_top_px, n_left_px, 0, 0);
  build_directional_and_filter_intra_predictors(xd, dst, dst_stride, dst,
                                                dst_stride, dr_meta, dr_px);
}

#if CONFIG_AV1_HIGHBITDEPTH
// Out-of-line HBD facade path to keep low-bitdepth dispatch register-light.
static AOM_NOINLINE void predict_intra_block_facade_hbd(const AV1_COMMON *cm,
                                                        MACROBLOCKD *xd,
                                                        int plane, int blk_col,
                                                        int blk_row,
                                                        TX_SIZE tx_size) {
  const MB_MODE_INFO *const mbmi = xd->mi[0];
#if !CONFIG_REALTIME_ONLY || CONFIG_AV1_DECODER
  if (plane != AOM_PLANE_Y && mbmi->uv_mode == UV_CFL_PRED) {
    predict_intra_block_cfl(xd, plane, blk_col, blk_row, tx_size);
    return;
  }
#endif  // !CONFIG_REALTIME_ONLY || CONFIG_AV1_DECODER
  if (mbmi->palette_mode_info.palette_size[plane != 0] > 0) {
    predict_intra_block_facade_palette(xd, plane, blk_col, blk_row, tx_size);
    return;
  }
  const struct macroblockd_plane *const pd = &xd->plane[plane];
  const int dst_stride = pd->dst.stride;
  uint8_t *const dst = get_intra_dst_ptr(pd, blk_col, blk_row);
  const PREDICTION_MODE mode =
      (plane == AOM_PLANE_Y) ? mbmi->mode : get_uv_mode(mbmi->uv_mode);
  const FILTER_INTRA_MODE filter_intra_mode =
      (plane == AOM_PLANE_Y && mbmi->filter_intra_mode_info.use_filter_intra)
          ? mbmi->filter_intra_mode_info.filter_intra_mode
          : FILTER_INTRA_MODES;
  predict_intra_block_internal(xd, true, true, cm->seq_params, BLOCK_INVALID, 0,
                               pd->width, pd->height, tx_size, mode, 0, 0,
                               filter_intra_mode, dst, dst_stride, dst,
                               dst_stride, blk_col, blk_row, plane);
}
#endif  // CONFIG_AV1_HIGHBITDEPTH

// Out-of-line chroma dispatch to keep the hot luma facade compact.
static AOM_NOINLINE void predict_intra_block_facade_chroma(
    const AV1_COMMON *cm, MACROBLOCKD *xd, int plane, int blk_col, int blk_row,
    TX_SIZE tx_size) {
  assert(!is_cur_buf_hbd(xd));
  const MB_MODE_INFO *const mbmi = xd->mi[0];
#if !CONFIG_REALTIME_ONLY || CONFIG_AV1_DECODER
  if (mbmi->uv_mode == UV_CFL_PRED) {
    predict_intra_block_cfl(xd, plane, blk_col, blk_row, tx_size);
    return;
  }
#endif  // !CONFIG_REALTIME_ONLY || CONFIG_AV1_DECODER
  if (mbmi->palette_mode_info.palette_size[1] > 0) {
    predict_intra_block_facade_palette(xd, plane, blk_col, blk_row, tx_size);
    return;
  }
  const struct macroblockd_plane *const pd = &xd->plane[plane];
  const int dst_stride = pd->dst.stride;
  uint8_t *const dst = get_intra_dst_ptr(pd, blk_col, blk_row);
  predict_intra_block_internal(
      xd, false, true, cm->seq_params, BLOCK_INVALID, 0, pd->width, pd->height,
      tx_size, get_uv_mode(mbmi->uv_mode), 0, 0, FILTER_INTRA_MODES, dst,
      dst_stride, dst, dst_stride, blk_col, blk_row, plane);
}

void av1_predict_intra_block_facade(const AV1_COMMON *cm, MACROBLOCKD *xd,
                                    int plane, int blk_col, int blk_row,
                                    TX_SIZE tx_size) {
  assert((unsigned)tx_size < TX_SIZES_ALL);
  assert((unsigned)plane < MAX_MB_PLANE);
#if CONFIG_AV1_HIGHBITDEPTH
  if (is_cur_buf_hbd(xd)) {
    predict_intra_block_facade_hbd(cm, xd, plane, blk_col, blk_row, tx_size);
    return;
  }
#endif  // CONFIG_AV1_HIGHBITDEPTH
  if (plane != AOM_PLANE_Y) {
    predict_intra_block_facade_chroma(cm, xd, plane, blk_col, blk_row, tx_size);
    return;
  }

  const MB_MODE_INFO *const mbmi = xd->mi[0];
  if (mbmi->palette_mode_info.palette_size[0] > 0) {
    predict_intra_block_facade_palette(xd, AOM_PLANE_Y, blk_col, blk_row,
                                       tx_size);
    return;
  }
  if (mbmi->filter_intra_mode_info.use_filter_intra) {
    predict_intra_block_facade_filter_intra(cm, xd, blk_col, blk_row, tx_size);
    return;
  }

  const struct macroblockd_plane *const pd = &xd->plane[AOM_PLANE_Y];
  const int dst_stride = pd->dst.stride;
  uint8_t *const dst = get_intra_dst_ptr(pd, blk_col, blk_row);
  predict_intra_block_internal(xd, false, true, cm->seq_params, BLOCK_INVALID,
                               0, pd->width, pd->height, tx_size, mbmi->mode, 0,
                               0, FILTER_INTRA_MODES, dst, dst_stride, dst,
                               dst_stride, blk_col, blk_row, AOM_PLANE_Y);
}

void av1_init_intra_predictors(void) {
  aom_once(init_intra_predictors_internal);
}
