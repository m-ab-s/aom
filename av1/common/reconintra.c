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

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"
#include "config/av1_rtcd.h"

#include "aom_dsp/aom_dsp_common.h"
#include "aom_mem/aom_mem.h"
#include "aom_ports/aom_once.h"
#include "aom_ports/mem.h"
#include "aom_ports/system_state.h"
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
#if CONFIG_MRLS
#define NUM_INTRA_NEIGHBOUR_PIXELS (MAX_TX_SIZE * 2 + 64)
#else
#define NUM_INTRA_NEIGHBOUR_PIXELS (MAX_TX_SIZE * 2 + 32)
#endif

static const uint8_t extend_modes[INTRA_MODES] = {
  NEED_ABOVE | NEED_LEFT
#if CONFIG_ORIP
      | NEED_ABOVELEFT
#endif
  ,                                         // DC
  NEED_ABOVE,                               // V
  NEED_LEFT,                                // H
  NEED_ABOVE | NEED_ABOVERIGHT,             // D45
  NEED_LEFT | NEED_ABOVE | NEED_ABOVELEFT,  // D135
  NEED_LEFT | NEED_ABOVE | NEED_ABOVELEFT,  // D113
  NEED_LEFT | NEED_ABOVE | NEED_ABOVELEFT,  // D157
  NEED_LEFT | NEED_BOTTOMLEFT,              // D203
  NEED_ABOVE | NEED_ABOVERIGHT,             // D67
  NEED_LEFT | NEED_ABOVE
#if CONFIG_ORIP
      | NEED_ABOVELEFT
#endif
  ,                                         // SMOOTH
  NEED_LEFT | NEED_ABOVE,                   // SMOOTH_V
  NEED_LEFT | NEED_ABOVE,                   // SMOOTH_H
  NEED_LEFT | NEED_ABOVE | NEED_ABOVELEFT,  // PAETH
};

#if CONFIG_EXT_RECUR_PARTITIONS
static int has_top_right(const AV1_COMMON *cm, const MACROBLOCKD *xd,
                         BLOCK_SIZE bsize, int mi_row, int mi_col,
                         int top_available, int right_available, TX_SIZE txsz,
                         int row_off, int col_off, int ss_x, int ss_y,
                         int px_to_right_edge, int *px_top_right,
                         int is_bsize_altered_for_chroma) {
  if (!top_available || !right_available) return 0;

  const int bw_unit = mi_size_wide[bsize];
  const int plane_bw_unit = AOMMAX(bw_unit >> ss_x, 1);
  const int top_right_count_unit = tx_size_wide_unit[txsz];
  const int px_tr_common = AOMMIN(tx_size_wide[txsz], px_to_right_edge);

  if (px_tr_common <= 0) return 0;

  *px_top_right = px_tr_common;

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
      const int col_off_64 = col_off % plane_bw_unit_64;
      return col_off_64 + top_right_count_unit < plane_bw_unit_64;
    }
    return col_off + top_right_count_unit < plane_bw_unit;
  } else {
    // All top-right pixels are in the block above, which is already available.
    if (col_off + top_right_count_unit < plane_bw_unit) return 1;

    // Handle the top-right intra tx block of the coding block
    const int sb_mi_size = mi_size_wide[cm->seq_params.sb_size];
    const int mi_row_aligned =
        is_bsize_altered_for_chroma
            ? xd->mi[0]->chroma_ref_info.mi_row_chroma_base
            : mi_row;
    const int mi_col_aligned =
        is_bsize_altered_for_chroma
            ? xd->mi[0]->chroma_ref_info.mi_col_chroma_base
            : mi_col;
    const int tr_mask_row = (mi_row_aligned & (sb_mi_size - 1)) - 1;
    const int tr_mask_col =
        (mi_col_aligned & (sb_mi_size - 1)) + mi_size_wide[bsize];

    if (tr_mask_row < 0) {
      return 1;
    } else if (tr_mask_col >= sb_mi_size) {
      return 0;
    } else {  // Handle the general case: the top_right mi is in the same SB
      const int tr_offset = tr_mask_row * xd->is_mi_coded_stride + tr_mask_col;
      // As long as the first mi is available, we determine tr is available
#if CONFIG_SDP
      int has_tr = xd->is_mi_coded[av1_get_sdp_idx(xd->tree_type)][tr_offset];
#else
      int has_tr = xd->is_mi_coded[tr_offset];
#endif  // CONFIG_SDP

      // Calculate px_top_right: how many top-right pixels are available. If it
      // is less than tx_size_wide[txsz], px_top_right will be used to
      // determine the location of the last available pixel, which will be used
      // for padding.
      if (has_tr) {
        int mi_tr = 0;
        for (int i = 0; i < top_right_count_unit << ss_x; ++i) {
          if ((tr_mask_col + i) >= sb_mi_size ||
#if CONFIG_SDP
              !xd->is_mi_coded[av1_get_sdp_idx(xd->tree_type)][tr_offset + i]
#else
              !xd->is_mi_coded[tr_offset + i]
#endif  // CONFIG_SDP
          ) {
            break;
          } else {
            mi_tr++;
          }
        }

        *px_top_right = AOMMIN((mi_tr << MI_SIZE_LOG2) >> ss_x, px_tr_common);
      }

      return has_tr;
    }
  }
}
#else
// Tables to store if the top-right reference pixels are available. The flags
// are represented with bits, packed into 8-bit integers. E.g., for the 32x32
// blocks in a 128x128 superblock, the index of the "o" block is 10 (in raster
// order), so its flag is stored at the 3rd bit of the 2nd entry in the table,
// i.e. (table[10 / 8] >> (10 % 8)) & 1.
//       . . . .
//       . . . .
//       . . o .
//       . . . .
static uint8_t has_tr_4x4[128] = {
  255, 255, 255, 255, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
  127, 127, 127, 127, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
  255, 127, 255, 127, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
  127, 127, 127, 127, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
  255, 255, 255, 127, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
  127, 127, 127, 127, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
  255, 127, 255, 127, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
  127, 127, 127, 127, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
};
static uint8_t has_tr_4x8[64] = {
  255, 255, 255, 255, 119, 119, 119, 119, 127, 127, 127, 127, 119,
  119, 119, 119, 255, 127, 255, 127, 119, 119, 119, 119, 127, 127,
  127, 127, 119, 119, 119, 119, 255, 255, 255, 127, 119, 119, 119,
  119, 127, 127, 127, 127, 119, 119, 119, 119, 255, 127, 255, 127,
  119, 119, 119, 119, 127, 127, 127, 127, 119, 119, 119, 119,
};
static uint8_t has_tr_8x4[64] = {
  255, 255, 0, 0, 85, 85, 0, 0, 119, 119, 0, 0, 85, 85, 0, 0,
  127, 127, 0, 0, 85, 85, 0, 0, 119, 119, 0, 0, 85, 85, 0, 0,
  255, 127, 0, 0, 85, 85, 0, 0, 119, 119, 0, 0, 85, 85, 0, 0,
  127, 127, 0, 0, 85, 85, 0, 0, 119, 119, 0, 0, 85, 85, 0, 0,
};
static uint8_t has_tr_8x8[32] = {
  255, 255, 85, 85, 119, 119, 85, 85, 127, 127, 85, 85, 119, 119, 85, 85,
  255, 127, 85, 85, 119, 119, 85, 85, 127, 127, 85, 85, 119, 119, 85, 85,
};
static uint8_t has_tr_8x16[16] = {
  255, 255, 119, 119, 127, 127, 119, 119,
  255, 127, 119, 119, 127, 127, 119, 119,
};
static uint8_t has_tr_16x8[16] = {
  255, 0, 85, 0, 119, 0, 85, 0, 127, 0, 85, 0, 119, 0, 85, 0,
};
static uint8_t has_tr_16x16[8] = {
  255, 85, 119, 85, 127, 85, 119, 85,
};
static uint8_t has_tr_16x32[4] = { 255, 119, 127, 119 };
static uint8_t has_tr_32x16[4] = { 15, 5, 7, 5 };
static uint8_t has_tr_32x32[2] = { 95, 87 };
static uint8_t has_tr_32x64[1] = { 127 };
static uint8_t has_tr_64x32[1] = { 19 };
static uint8_t has_tr_64x64[1] = { 7 };
static uint8_t has_tr_64x128[1] = { 3 };
static uint8_t has_tr_128x64[1] = { 1 };
static uint8_t has_tr_128x128[1] = { 1 };
static uint8_t has_tr_4x16[32] = {
  255, 255, 255, 255, 127, 127, 127, 127, 255, 127, 255,
  127, 127, 127, 127, 127, 255, 255, 255, 127, 127, 127,
  127, 127, 255, 127, 255, 127, 127, 127, 127, 127,
};
static uint8_t has_tr_16x4[32] = {
  255, 0, 0, 0, 85, 0, 0, 0, 119, 0, 0, 0, 85, 0, 0, 0,
  127, 0, 0, 0, 85, 0, 0, 0, 119, 0, 0, 0, 85, 0, 0, 0,
};
static uint8_t has_tr_8x32[8] = {
  255, 255, 127, 127, 255, 127, 127, 127,
};
static uint8_t has_tr_32x8[8] = {
  15, 0, 5, 0, 7, 0, 5, 0,
};
static uint8_t has_tr_16x64[2] = { 255, 127 };
static uint8_t has_tr_64x16[2] = { 3, 1 };

static const uint8_t *const has_tr_tables[BLOCK_SIZES_ALL] = {
  // 4X4
  has_tr_4x4,
  // 4X8,       8X4,            8X8
  has_tr_4x8, has_tr_8x4, has_tr_8x8,
  // 8X16,      16X8,           16X16
  has_tr_8x16, has_tr_16x8, has_tr_16x16,
  // 16X32,     32X16,          32X32
  has_tr_16x32, has_tr_32x16, has_tr_32x32,
  // 32X64,     64X32,          64X64
  has_tr_32x64, has_tr_64x32, has_tr_64x64,
  // 64x128,    128x64,         128x128
  has_tr_64x128, has_tr_128x64, has_tr_128x128,
  // 4x16,      16x4,            8x32
  has_tr_4x16, has_tr_16x4, has_tr_8x32,
  // 32x8,      16x64,           64x16
  has_tr_32x8, has_tr_16x64, has_tr_64x16
};

static uint8_t has_tr_vert_8x8[32] = {
  255, 255, 0, 0, 119, 119, 0, 0, 127, 127, 0, 0, 119, 119, 0, 0,
  255, 127, 0, 0, 119, 119, 0, 0, 127, 127, 0, 0, 119, 119, 0, 0,
};
static uint8_t has_tr_vert_16x16[8] = {
  255, 0, 119, 0, 127, 0, 119, 0,
};
static uint8_t has_tr_vert_32x32[2] = { 15, 7 };
static uint8_t has_tr_vert_64x64[1] = { 3 };

// The _vert_* tables are like the ordinary tables above, but describe the
// order we visit square blocks when doing a PARTITION_VERT_A or
// PARTITION_VERT_B. This is the same order as normal except for on the last
// split where we go vertically (TL, BL, TR, BR). We treat the rectangular block
// as a pair of squares, which means that these tables work correctly for both
// mixed vertical partition types.
//
// There are tables for each of the square sizes. Vertical rectangles (like
// BLOCK_16X32) use their respective "non-vert" table
static const uint8_t *const has_tr_vert_tables[BLOCK_SIZES] = {
  // 4X4
  NULL,
  // 4X8,      8X4,         8X8
  has_tr_4x8, NULL, has_tr_vert_8x8,
  // 8X16,     16X8,        16X16
  has_tr_8x16, NULL, has_tr_vert_16x16,
  // 16X32,    32X16,       32X32
  has_tr_16x32, NULL, has_tr_vert_32x32,
  // 32X64,    64X32,       64X64
  has_tr_32x64, NULL, has_tr_vert_64x64,
  // 64x128,   128x64,      128x128
  has_tr_64x128, NULL, has_tr_128x128
};

static const uint8_t *get_has_tr_table(PARTITION_TYPE partition,
                                       BLOCK_SIZE bsize) {
  const uint8_t *ret = NULL;
  // If this is a mixed vertical partition, look up bsize in orders_vert.
  if (partition == PARTITION_VERT_A || partition == PARTITION_VERT_B) {
    assert(bsize < BLOCK_SIZES);
    ret = has_tr_vert_tables[bsize];
  } else {
    ret = has_tr_tables[bsize];
  }
  assert(ret);
  return ret;
}

static int has_top_right(const AV1_COMMON *cm, BLOCK_SIZE bsize, int mi_row,
                         int mi_col, int top_available, int right_available,
                         PARTITION_TYPE partition, TX_SIZE txsz, int row_off,
                         int col_off, int ss_x, int ss_y) {
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
      const int col_off_64 = col_off % plane_bw_unit_64;
      return col_off_64 + top_right_count_unit < plane_bw_unit_64;
    }
    return col_off + top_right_count_unit < plane_bw_unit;
  } else {
    // All top-right pixels are in the block above, which is already available.
    if (col_off + top_right_count_unit < plane_bw_unit) return 1;

    const int bw_in_mi_log2 = mi_size_wide_log2[bsize];
    const int bh_in_mi_log2 = mi_size_high_log2[bsize];
    const int sb_mi_size = mi_size_high[cm->seq_params.sb_size];
    const int blk_row_in_sb = (mi_row & (sb_mi_size - 1)) >> bh_in_mi_log2;
    const int blk_col_in_sb = (mi_col & (sb_mi_size - 1)) >> bw_in_mi_log2;

    // Top row of superblock: so top-right pixels are in the top and/or
    // top-right superblocks, both of which are already available.
    if (blk_row_in_sb == 0) return 1;

    // Rightmost column of superblock (and not the top row): so top-right pixels
    // fall in the right superblock, which is not available yet.
    if (((blk_col_in_sb + 1) << bw_in_mi_log2) >= sb_mi_size) {
      return 0;
    }

    // General case (neither top row nor rightmost column): check if the
    // top-right block is coded before the current block.
    const int this_blk_index =
        ((blk_row_in_sb + 0) << (MAX_MIB_SIZE_LOG2 - bw_in_mi_log2)) +
        blk_col_in_sb + 0;
    const int idx1 = this_blk_index / 8;
    const int idx2 = this_blk_index % 8;
    const uint8_t *has_tr_table = get_has_tr_table(partition, bsize);
    return (has_tr_table[idx1] >> idx2) & 1;
  }
}
#endif

static int has_bottom_left(const AV1_COMMON *cm, const MACROBLOCKD *xd,
                           BLOCK_SIZE bsize, int mi_row, int mi_col,
                           int bottom_available, int left_available,
                           TX_SIZE txsz, int row_off, int col_off, int ss_x,
                           int ss_y, int px_to_bottom_edge, int *px_bottom_left,
                           int is_bsize_altered_for_chroma) {
  if (!bottom_available || !left_available) return 0;

  const int px_bl_common = AOMMIN(tx_size_high[txsz], px_to_bottom_edge);

  if (px_bl_common <= 0) return 0;

  *px_bottom_left = px_bl_common;

  // Special case for 128x* blocks, when col_off is half the block width.
  // This is needed because 128x* superblocks are divided into 64x* blocks in
  // raster order
  if (block_size_wide[bsize] > block_size_wide[BLOCK_64X64] && col_off > 0) {
    const int plane_bw_unit_64 = mi_size_wide[BLOCK_64X64] >> ss_x;
    const int col_off_64 = col_off % plane_bw_unit_64;
    if (col_off_64 == 0) {
      // We are at the left edge of top-right or bottom-right 64x* block.
      const int plane_bh_unit_64 = mi_size_high[BLOCK_64X64] >> ss_y;
      const int row_off_64 = row_off % plane_bh_unit_64;
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

    // The general case: neither the leftmost column nor the bottom row. The
    // bottom-left mi is in the same SB
    const int sb_mi_size = mi_size_high[cm->seq_params.sb_size];
    const int mi_row_aligned =
        is_bsize_altered_for_chroma
            ? xd->mi[0]->chroma_ref_info.mi_row_chroma_base
            : mi_row;
    const int mi_col_aligned =
        is_bsize_altered_for_chroma
            ? xd->mi[0]->chroma_ref_info.mi_col_chroma_base
            : mi_col;
    const int bl_mask_row =
        (mi_row_aligned & (sb_mi_size - 1)) + mi_size_high[bsize];
    const int bl_mask_col = (mi_col_aligned & (sb_mi_size - 1)) - 1;

    if (bl_mask_col < 0) {
      const int plane_sb_height =
          block_size_high[cm->seq_params.sb_size] >> ss_y;
      const int plane_bottom_row =
          (((mi_row_aligned & (sb_mi_size - 1)) << MI_SIZE_LOG2) +
           block_size_high[bsize]) >>
          ss_y;
      *px_bottom_left =
          AOMMIN(plane_sb_height - plane_bottom_row, px_bl_common);

      return *px_bottom_left > 0;
    } else if (bl_mask_row >= sb_mi_size) {
      return 0;
    } else {
      const int bl_offset = bl_mask_row * xd->is_mi_coded_stride + bl_mask_col;
      // As long as there is one bottom-left mi available, we determine bl is
      // available
#if CONFIG_SDP
      int has_bl = xd->is_mi_coded[av1_get_sdp_idx(xd->tree_type)][bl_offset];
#else
      int has_bl = xd->is_mi_coded[bl_offset];
#endif  // CONFIG_SDP

      // Calculate px_bottom_left: how many bottom-left pixels are available. If
      // it is less than tx_size_high[txsz], px_bottom_left will be used to
      // determine the location of the last available pixel, which will be used
      // for padding.
      if (has_bl) {
        int mi_bl = 0;
        for (int i = 0; i < bottom_left_count_unit << ss_y; ++i) {
          if ((bl_mask_row + i) >= sb_mi_size ||
#if CONFIG_SDP
              !xd->is_mi_coded[av1_get_sdp_idx(xd->tree_type)]
                              [bl_offset + i * xd->is_mi_coded_stride]
#else
              !xd->is_mi_coded[bl_offset + i * xd->is_mi_coded_stride]
#endif  // CONFIG_SDP
          ) {
            break;
          } else {
            mi_bl++;
          }
        }

        *px_bottom_left = AOMMIN((mi_bl << MI_SIZE_LOG2) >> ss_y, px_bl_common);
      }

      return has_bl;
    }
  }
}

typedef void (*intra_pred_fn)(uint8_t *dst, ptrdiff_t stride,
                              const uint8_t *above, const uint8_t *left);

static intra_pred_fn pred[INTRA_MODES][TX_SIZES_ALL];
static intra_pred_fn dc_pred[2][2][TX_SIZES_ALL];

typedef void (*intra_high_pred_fn)(uint16_t *dst, ptrdiff_t stride,
                                   const uint16_t *above, const uint16_t *left,
                                   int bd);
static intra_high_pred_fn pred_high[INTRA_MODES][TX_SIZES_ALL];
static intra_high_pred_fn dc_pred_high[2][2][TX_SIZES_ALL];

static void init_intra_predictors_internal(void) {
  assert(NELEMENTS(mode_to_angle_map) == INTRA_MODES);

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

#define INIT_NO_4X4(p, type)                  \
  p[TX_8X8] = aom_##type##_predictor_8x8;     \
  p[TX_16X16] = aom_##type##_predictor_16x16; \
  p[TX_32X32] = aom_##type##_predictor_32x32; \
  p[TX_64X64] = aom_##type##_predictor_64x64; \
  INIT_RECTANGULAR(p, type)

#define INIT_ALL_SIZES(p, type)           \
  p[TX_4X4] = aom_##type##_predictor_4x4; \
  INIT_NO_4X4(p, type)

  INIT_ALL_SIZES(pred[V_PRED], v);
  INIT_ALL_SIZES(pred[H_PRED], h);
  INIT_ALL_SIZES(pred[PAETH_PRED], paeth);
  INIT_ALL_SIZES(pred[SMOOTH_PRED], smooth);
  INIT_ALL_SIZES(pred[SMOOTH_V_PRED], smooth_v);
  INIT_ALL_SIZES(pred[SMOOTH_H_PRED], smooth_h);
  INIT_ALL_SIZES(dc_pred[0][0], dc_128);
  INIT_ALL_SIZES(dc_pred[0][1], dc_top);
  INIT_ALL_SIZES(dc_pred[1][0], dc_left);
  INIT_ALL_SIZES(dc_pred[1][1], dc);
  INIT_ALL_SIZES(pred_high[V_PRED], highbd_v);
  INIT_ALL_SIZES(pred_high[H_PRED], highbd_h);
  INIT_ALL_SIZES(pred_high[PAETH_PRED], highbd_paeth);
  INIT_ALL_SIZES(pred_high[SMOOTH_PRED], highbd_smooth);
  INIT_ALL_SIZES(pred_high[SMOOTH_V_PRED], highbd_smooth_v);
  INIT_ALL_SIZES(pred_high[SMOOTH_H_PRED], highbd_smooth_h);
  INIT_ALL_SIZES(dc_pred_high[0][0], highbd_dc_128);
  INIT_ALL_SIZES(dc_pred_high[0][1], highbd_dc_top);
  INIT_ALL_SIZES(dc_pred_high[1][0], highbd_dc_left);
  INIT_ALL_SIZES(dc_pred_high[1][1], highbd_dc);
#undef intra_pred_allsizes
}

// Directional prediction, zone 1: 0 < angle < 90
void av1_dr_prediction_z1_c(uint8_t *dst, ptrdiff_t stride, int bw, int bh,
                            const uint8_t *above, const uint8_t *left,
                            int upsample_above, int dx, int dy
#if CONFIG_MRLS
                            ,
                            int mrl_index
#endif
) {
  int r, c, x, base, shift, val;

  (void)left;
  (void)dy;
  assert(dy == 1);
  assert(dx > 0);

#if CONFIG_MRLS
  const int max_base_x = ((bw + bh) - 1 + (mrl_index << 1)) << upsample_above;
#else
  const int max_base_x = ((bw + bh) - 1) << upsample_above;
#endif
  const int frac_bits = 6 - upsample_above;
  const int base_inc = 1 << upsample_above;
#if CONFIG_MRLS
  x = dx * (1 + mrl_index);
#else
  x = dx;
#endif
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
                            int dy
#if CONFIG_MRLS
                            ,
                            int mrl_index
#endif
) {
  assert(dx > 0);
  assert(dy > 0);

#if CONFIG_MRLS
  const int min_base_x = -((1 + mrl_index) << upsample_above);
  const int min_base_y = -((1 + mrl_index) << upsample_left);
#else
  const int min_base_x = -(1 << upsample_above);
  const int min_base_y = -(1 << upsample_left);
#endif
  (void)min_base_y;
  const int frac_bits_x = 6 - upsample_above;
  const int frac_bits_y = 6 - upsample_left;

  for (int r = 0; r < bh; ++r) {
    for (int c = 0; c < bw; ++c) {
      int val;
      int y = r + 1;
#if CONFIG_MRLS
      int x = (c << 6) - (y + mrl_index) * dx;
#else
      int x = (c << 6) - y * dx;
#endif
      const int base_x = x >> frac_bits_x;
      if (base_x >= min_base_x) {
        const int shift = ((x * (1 << upsample_above)) & 0x3F) >> 1;
        val = above[base_x] * (32 - shift) + above[base_x + 1] * shift;
        val = ROUND_POWER_OF_TWO(val, 5);
      } else {
        x = c + 1;
#if CONFIG_MRLS
        y = (r << 6) - (x + mrl_index) * dy;
#else
        y = (r << 6) - x * dy;
#endif
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
                            int upsample_left, int dx, int dy
#if CONFIG_MRLS
                            ,
                            int mrl_index
#endif
) {
  (void)above;
  (void)dx;

  assert(dx == 1);
  assert(dy > 0);

#if CONFIG_MRLS
  const int max_base_y = (bw + bh - 1 + (mrl_index << 1)) << upsample_left;
#else
  const int max_base_y = (bw + bh - 1) << upsample_left;
#endif
  const int frac_bits = 6 - upsample_left;
  const int base_inc = 1 << upsample_left;
#if CONFIG_MRLS
  int y = dy * (1 + mrl_index);
#else
  int y = dy;
#endif
  for (int c = 0; c < bw; ++c, y += dy) {
    int base = y >> frac_bits;
    const int shift = ((y << upsample_left) & 0x3F) >> 1;

    for (int r = 0; r < bh; ++r, base += base_inc) {
      if (base < max_base_y) {
        const int val = left[base] * (32 - shift) + left[base + 1] * shift;
        dst[r * stride + c] = ROUND_POWER_OF_TWO(val, 5);
      } else {
        for (; r < bh; ++r) dst[r * stride + c] = left[max_base_y];
        break;
      }
    }
  }
}

static void dr_predictor(uint8_t *dst, ptrdiff_t stride, TX_SIZE tx_size,
                         const uint8_t *above, const uint8_t *left,
                         int upsample_above, int upsample_left, int angle
#if CONFIG_MRLS
                         ,
                         int mrl_index
#endif
) {
  const int dx = av1_get_dx(angle);
  const int dy = av1_get_dy(angle);
  const int bw = tx_size_wide[tx_size];
  const int bh = tx_size_high[tx_size];
  assert(angle > 0 && angle < 270);

  if (angle > 0 && angle < 90) {
    av1_dr_prediction_z1(dst, stride, bw, bh, above, left, upsample_above, dx,
                         dy
#if CONFIG_MRLS
                         ,
                         mrl_index
#endif
    );
  } else if (angle > 90 && angle < 180) {
    av1_dr_prediction_z2(dst, stride, bw, bh, above, left, upsample_above,
                         upsample_left, dx, dy
#if CONFIG_MRLS
                         ,
                         mrl_index
#endif
    );
  } else if (angle > 180 && angle < 270) {
    av1_dr_prediction_z3(dst, stride, bw, bh, above, left, upsample_left, dx, dy
#if CONFIG_MRLS
                         ,
                         mrl_index
#endif
    );
  } else if (angle == 90) {
    pred[V_PRED][tx_size](dst, stride, above, left);
  } else if (angle == 180) {
    pred[H_PRED][tx_size](dst, stride, above, left);
  }
}

// Directional prediction, zone 1: 0 < angle < 90
void av1_highbd_dr_prediction_z1_c(uint16_t *dst, ptrdiff_t stride, int bw,
                                   int bh, const uint16_t *above,
                                   const uint16_t *left, int upsample_above,
                                   int dx, int dy, int bd
#if CONFIG_MRLS
                                   ,
                                   int mrl_index
#endif
) {
  int r, c, x, base, shift, val;

  (void)left;
  (void)dy;
  (void)bd;
  assert(dy == 1);
  assert(dx > 0);

#if CONFIG_MRLS
  const int max_base_x = ((bw + bh) - 1 + (mrl_index << 1)) << upsample_above;
#else
  const int max_base_x = ((bw + bh) - 1) << upsample_above;
#endif
  const int frac_bits = 6 - upsample_above;
  const int base_inc = 1 << upsample_above;
#if CONFIG_MRLS
  x = dx * (1 + mrl_index);
#else
  x = dx;
#endif
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
                                   int upsample_left, int dx, int dy, int bd
#if CONFIG_MRLS
                                   ,
                                   int mrl_index
#endif
) {
  (void)bd;
  assert(dx > 0);
  assert(dy > 0);

#if CONFIG_MRLS
  const int min_base_x = -(1 << upsample_above) - mrl_index;
  const int min_base_y = -(1 << upsample_left) - mrl_index;
#else
  const int min_base_x = -(1 << upsample_above);
  const int min_base_y = -(1 << upsample_left);
#endif
  (void)min_base_y;
  const int frac_bits_x = 6 - upsample_above;
  const int frac_bits_y = 6 - upsample_left;

  for (int r = 0; r < bh; ++r) {
    for (int c = 0; c < bw; ++c) {
      int val;
      int y = r + 1;
#if CONFIG_MRLS
      int x = (c << 6) - (y + mrl_index) * dx;
#else
      int x = (c << 6) - y * dx;
#endif
      const int base_x = x >> frac_bits_x;
      if (base_x >= min_base_x) {
        const int shift = ((x * (1 << upsample_above)) & 0x3F) >> 1;
        val = above[base_x] * (32 - shift) + above[base_x + 1] * shift;
        val = ROUND_POWER_OF_TWO(val, 5);
      } else {
        x = c + 1;
#if CONFIG_MRLS
        y = (r << 6) - (x + mrl_index) * dy;
#else
        y = (r << 6) - x * dy;
#endif
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
                                   int dx, int dy, int bd
#if CONFIG_MRLS
                                   ,
                                   int mrl_index
#endif
) {
  int r, c, y, base, shift, val;

  (void)above;
  (void)dx;
  (void)bd;
  assert(dx == 1);
  assert(dy > 0);

#if CONFIG_MRLS
  const int max_base_y = ((bw + bh - 1) << upsample_left) + (mrl_index << 1);
#else
  const int max_base_y = (bw + bh - 1) << upsample_left;
#endif
  const int frac_bits = 6 - upsample_left;
  const int base_inc = 1 << upsample_left;
#if CONFIG_MRLS
  y = dy * (1 + mrl_index);
#else
  y = dy;
#endif
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

static void highbd_dr_predictor(uint16_t *dst, ptrdiff_t stride,
                                TX_SIZE tx_size, const uint16_t *above,
                                const uint16_t *left, int upsample_above,
                                int upsample_left, int angle, int bd
#if CONFIG_MRLS
                                ,
                                int mrl_index
#endif
) {
  const int dx = av1_get_dx(angle);
  const int dy = av1_get_dy(angle);
  const int bw = tx_size_wide[tx_size];
  const int bh = tx_size_high[tx_size];
  assert(angle > 0 && angle < 270);

  if (angle > 0 && angle < 90) {
    av1_highbd_dr_prediction_z1(dst, stride, bw, bh, above, left,
                                upsample_above, dx, dy, bd
#if CONFIG_MRLS
                                ,
                                mrl_index
#endif
    );
  } else if (angle > 90 && angle < 180) {
    av1_highbd_dr_prediction_z2(dst, stride, bw, bh, above, left,
                                upsample_above, upsample_left, dx, dy, bd
#if CONFIG_MRLS
                                ,
                                mrl_index
#endif
    );
  } else if (angle > 180 && angle < 270) {
    av1_highbd_dr_prediction_z3(dst, stride, bw, bh, above, left, upsample_left,
                                dx, dy, bd
#if CONFIG_MRLS
                                ,
                                mrl_index
#endif
    );
  } else if (angle == 90) {
    pred_high[V_PRED][tx_size](dst, stride, above, left, bd);
  } else if (angle == 180) {
    pred_high[H_PRED][tx_size](dst, stride, above, left, bd);
  }
}

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

  // The initialization is just for silencing Jenkins static analysis warnings
  for (r = 0; r < bh + 1; ++r)
    memset(buffer[r], 0, (bw + 1) * sizeof(buffer[0][0]));

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
        buffer[r + r_offset][c + c_offset] =
            clip_pixel(ROUND_POWER_OF_TWO_SIGNED(
                av1_filter_intra_taps[mode][k][0] * p0 +
                    av1_filter_intra_taps[mode][k][1] * p1 +
                    av1_filter_intra_taps[mode][k][2] * p2 +
                    av1_filter_intra_taps[mode][k][3] * p3 +
                    av1_filter_intra_taps[mode][k][4] * p4 +
                    av1_filter_intra_taps[mode][k][5] * p5 +
                    av1_filter_intra_taps[mode][k][6] * p6,
                FILTER_INTRA_SCALE_BITS));
      }
    }

  for (r = 0; r < bh; ++r) {
    memcpy(dst, &buffer[r + 1][1], bw * sizeof(uint8_t));
    dst += stride;
  }
}

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

  // The initialization is just for silencing Jenkins static analysis warnings
  for (r = 0; r < bh + 1; ++r)
    memset(buffer[r], 0, (bw + 1) * sizeof(buffer[0][0]));

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
        buffer[r + r_offset][c + c_offset] =
            clip_pixel_highbd(ROUND_POWER_OF_TWO_SIGNED(
                                  av1_filter_intra_taps[mode][k][0] * p0 +
                                      av1_filter_intra_taps[mode][k][1] * p1 +
                                      av1_filter_intra_taps[mode][k][2] * p2 +
                                      av1_filter_intra_taps[mode][k][3] * p3 +
                                      av1_filter_intra_taps[mode][k][4] * p4 +
                                      av1_filter_intra_taps[mode][k][5] * p5 +
                                      av1_filter_intra_taps[mode][k][6] * p6,
                                  FILTER_INTRA_SCALE_BITS),
                              bd);
      }
    }

  for (r = 0; r < bh; ++r) {
    memcpy(dst, &buffer[r + 1][1], bw * sizeof(dst[0]));
    dst += stride;
  }
}

static int is_smooth(const MB_MODE_INFO *mbmi, int plane) {
  if (plane == 0) {
    const PREDICTION_MODE mode = mbmi->mode;
    return (mode == SMOOTH_PRED || mode == SMOOTH_V_PRED ||
            mode == SMOOTH_H_PRED);
  } else {
    // uv_mode is not set for inter blocks, so need to explicitly
    // detect that case.
#if CONFIG_SDP
    if (is_inter_block(mbmi, SHARED_PART)) return 0;
#else
    if (is_inter_block(mbmi)) return 0;
#endif

    const UV_PREDICTION_MODE uv_mode = mbmi->uv_mode;
    return (uv_mode == UV_SMOOTH_PRED || uv_mode == UV_SMOOTH_V_PRED ||
            uv_mode == UV_SMOOTH_H_PRED);
  }
}

static int get_filt_type(const MACROBLOCKD *xd, int plane) {
  int ab_sm, le_sm;

  if (plane == 0) {
    const MB_MODE_INFO *ab = xd->above_mbmi;
    const MB_MODE_INFO *le = xd->left_mbmi;
    ab_sm = ab ? is_smooth(ab, plane) : 0;
    le_sm = le ? is_smooth(le, plane) : 0;
  } else {
    const MB_MODE_INFO *ab = xd->chroma_above_mbmi;
    const MB_MODE_INFO *le = xd->chroma_left_mbmi;
    ab_sm = ab ? is_smooth(ab, plane) : 0;
    le_sm = le ? is_smooth(le, plane) : 0;
  }

  return (ab_sm || le_sm) ? 1 : 0;
}

static int intra_edge_filter_strength(int bs0, int bs1, int delta, int type) {
  const int d = abs(delta);
  int strength = 0;

  const int blk_wh = bs0 + bs1;
  if (type == 0) {
    if (blk_wh <= 8) {
      if (d >= 56) strength = 1;
    } else if (blk_wh <= 12) {
      if (d >= 40) strength = 1;
    } else if (blk_wh <= 16) {
      if (d >= 40) strength = 1;
    } else if (blk_wh <= 24) {
      if (d >= 8) strength = 1;
      if (d >= 16) strength = 2;
      if (d >= 32) strength = 3;
    } else if (blk_wh <= 32) {
      if (d >= 1) strength = 1;
      if (d >= 4) strength = 2;
      if (d >= 32) strength = 3;
    } else {
      if (d >= 1) strength = 3;
    }
  } else {
    if (blk_wh <= 8) {
      if (d >= 40) strength = 1;
      if (d >= 64) strength = 2;
    } else if (blk_wh <= 16) {
      if (d >= 20) strength = 1;
      if (d >= 48) strength = 2;
    } else if (blk_wh <= 24) {
      if (d >= 4) strength = 3;
    } else {
      if (d >= 1) strength = 3;
    }
  }
  return strength;
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

void av1_filter_intra_edge_high_c(uint16_t *p, int sz, int strength) {
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

static void filter_intra_edge_corner_high(uint16_t *p_above, uint16_t *p_left) {
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

void av1_upsample_intra_edge_high_c(uint16_t *p, int sz, int bd) {
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

static void build_intra_predictors_high(
    const MACROBLOCKD *xd, const uint8_t *ref8, int ref_stride, uint8_t *dst8,
    int dst_stride, PREDICTION_MODE mode, int angle_delta,
    FILTER_INTRA_MODE filter_intra_mode, TX_SIZE tx_size,
    int disable_edge_filter, int n_top_px, int n_topright_px, int n_left_px,
    int n_bottomleft_px, int plane
#if CONFIG_MRLS
    ,
    int is_sb_boundary
#endif
#if CONFIG_ORIP
    ,
    const int disable_intra_pred_filter_for_hor_ver_mode,
    const int seq_intra_pred_filter_flag
#endif
) {
  int i;
  uint16_t *dst = CONVERT_TO_SHORTPTR(dst8);
  uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  DECLARE_ALIGNED(16, uint16_t, left_data[NUM_INTRA_NEIGHBOUR_PIXELS]);
  DECLARE_ALIGNED(16, uint16_t, above_data[NUM_INTRA_NEIGHBOUR_PIXELS]);
#if CONFIG_MRLS
  uint16_t *const above_row = above_data + 32;
  uint16_t *const left_col = left_data + 32;
#else
  uint16_t *const above_row = above_data + 16;
  uint16_t *const left_col = left_data + 16;
#endif
  const int txwpx = tx_size_wide[tx_size];
  const int txhpx = tx_size_high[tx_size];
  int need_left = extend_modes[mode] & NEED_LEFT;
  int need_above = extend_modes[mode] & NEED_ABOVE;
  int need_above_left = extend_modes[mode] & NEED_ABOVELEFT;
#if CONFIG_MRLS
  const uint8_t mrl_index =
      (plane == PLANE_TYPE_Y && is_inter_block(xd->mi[0]
#if CONFIG_SDP
                                               ,
                                               xd->tree_type
#endif
                                               ) == 0)
          ? xd->mi[0]->mrl_index
          : 0;
  const int above_mrl_idx = is_sb_boundary ? 0 : mrl_index;
  const uint16_t *above_ref = ref - ref_stride * (above_mrl_idx + 1);
  const uint16_t *left_ref = ref - 1 - mrl_index;
#else
  const uint16_t *above_ref = ref - ref_stride;
  const uint16_t *left_ref = ref - 1;
#endif
  int p_angle = 0;
  const int is_dr_mode = av1_is_directional_mode(mode);
  const int use_filter_intra = filter_intra_mode != FILTER_INTRA_MODES;
  int base = 128 << (xd->bd - 8);
  // The left_data, above_data buffers must be zeroed to fix some intermittent
  // valgrind errors. Uninitialized reads in intra pred modules (e.g. width = 4
  // path in av1_highbd_dr_prediction_z2_avx2()) from left_data, above_data are
  // seen to be the potential reason for this issue.
  aom_memset16(left_data, base + 1, NUM_INTRA_NEIGHBOUR_PIXELS);
  aom_memset16(above_data, base - 1, NUM_INTRA_NEIGHBOUR_PIXELS);

  // The default values if ref pixels are not available:
  // base   base-1 base-1 .. base-1 base-1 base-1 base-1 base-1 base-1
  // base+1   A      B  ..     Y      Z
  // base+1   C      D  ..     W      X
  // base+1   E      F  ..     U      V
  // base+1   G      H  ..     S      T      T      T      T      T

#if CONFIG_ORIP
#if CONFIG_MRLS
  int apply_sub_block_based_refinement_filter =
      seq_intra_pred_filter_flag && (mrl_index == 0);
#else
  int apply_sub_block_based_refinement_filter = seq_intra_pred_filter_flag;
#endif
#endif

  if (is_dr_mode) {
    p_angle = mode_to_angle_map[mode] + angle_delta;
    if (p_angle <= 90)
      need_above = 1, need_left = 0, need_above_left = 1;
    else if (p_angle < 180)
      need_above = 1, need_left = 1, need_above_left = 1;
    else
      need_above = 0, need_left = 1, need_above_left = 1;

#if CONFIG_ORIP
    if (apply_sub_block_based_refinement_filter &&
        (p_angle == 90 || p_angle == 180) &&
        !disable_intra_pred_filter_for_hor_ver_mode) {
      need_above = 1;
      need_left = 1;
      need_above_left = 1;
    }
#endif
  }
  if (use_filter_intra) need_left = need_above = need_above_left = 1;

  assert(n_top_px >= 0);
  assert(n_topright_px >= 0);
  assert(n_left_px >= 0);
  assert(n_bottomleft_px >= 0);

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

  // NEED_LEFT
  if (need_left) {
    int need_bottom = extend_modes[mode] & NEED_BOTTOMLEFT;
    if (use_filter_intra) need_bottom = 0;
    if (is_dr_mode) need_bottom = p_angle > 180;
#if CONFIG_MRLS
    const int num_left_pixels_needed =
        txhpx + (need_bottom ? txwpx : 3) + (mrl_index << 1);
#else
    const int num_left_pixels_needed = txhpx + (need_bottom ? txwpx : 0);
#endif
    i = 0;
    if (n_left_px > 0) {
      for (; i < n_left_px; i++) left_col[i] = left_ref[i * ref_stride];
      if (need_bottom && n_bottomleft_px > 0) {
        assert(i == txhpx);
        for (; i < txhpx + n_bottomleft_px; i++)
          left_col[i] = left_ref[i * ref_stride];
      }
      if (i < num_left_pixels_needed)
        aom_memset16(&left_col[i], left_col[i - 1], num_left_pixels_needed - i);
    } else if (n_top_px > 0) {
      aom_memset16(left_col, above_ref[0], num_left_pixels_needed);
    }
  }

  // NEED_ABOVE
  if (need_above) {
    int need_right = extend_modes[mode] & NEED_ABOVERIGHT;
    if (use_filter_intra) need_right = 0;
    if (is_dr_mode) need_right = p_angle < 90;
#if CONFIG_MRLS
    const int num_top_pixels_needed =
        txwpx + (need_right ? txhpx : 0) + (mrl_index << 1);
#else
    const int num_top_pixels_needed = txwpx + (need_right ? txhpx : 0);
#endif
    if (n_top_px > 0) {
      memcpy(above_row, above_ref, n_top_px * sizeof(above_ref[0]));
      i = n_top_px;
      if (need_right && n_topright_px > 0) {
        assert(n_top_px == txwpx);
        memcpy(above_row + txwpx, above_ref + txwpx,
               n_topright_px * sizeof(above_ref[0]));
        i += n_topright_px;
      }
      if (i < num_top_pixels_needed)
        aom_memset16(&above_row[i], above_row[i - 1],
                     num_top_pixels_needed - i);
    } else if (n_left_px > 0) {
      aom_memset16(above_row, left_ref[0], num_top_pixels_needed);
    }
  }

  if (need_above_left) {
#if CONFIG_MRLS
    for (i = 1; i <= mrl_index + 1; i++) {
      if (n_top_px > 0 && n_left_px > 0) {
        above_row[-i] = above_ref[-i];
        if (is_sb_boundary)
          left_col[-i] = left_ref[-ref_stride];
        else
          left_col[-i] = left_ref[-i * ref_stride];

      } else if (n_top_px > 0) {
        above_row[-i] = left_col[-i] = above_ref[0];
      } else if (n_left_px > 0) {
        above_row[-i] = left_col[-i] = left_ref[0];
      } else {
        above_row[-i] = left_col[-i] = base;
      }
    }
#else
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
#endif
  }

  if (use_filter_intra) {
    highbd_filter_intra_predictor(dst, dst_stride, tx_size, above_row, left_col,
                                  filter_intra_mode, xd->bd);
    return;
  }

  if (is_dr_mode) {
    int upsample_above = 0;
    int upsample_left = 0;
#if CONFIG_MRLS
    if (!disable_edge_filter && mrl_index == 0) {
#else
    if (!disable_edge_filter) {
#endif
      const int need_right = p_angle < 90;
      const int need_bottom = p_angle > 180;
      const int filt_type = get_filt_type(xd, plane);
      if (p_angle != 90 && p_angle != 180) {
        const int ab_le = need_above_left ? 1 : 0;
        if (need_above && need_left && (txwpx + txhpx >= 24)) {
          filter_intra_edge_corner_high(above_row, left_col);
        }
        if (need_above && n_top_px > 0) {
          const int strength =
              intra_edge_filter_strength(txwpx, txhpx, p_angle - 90, filt_type);
          const int n_px = n_top_px + ab_le + (need_right ? txhpx : 0);
          av1_filter_intra_edge_high(above_row - ab_le, n_px, strength);
        }
        if (need_left && n_left_px > 0) {
          const int strength = intra_edge_filter_strength(
              txhpx, txwpx, p_angle - 180, filt_type);
          const int n_px = n_left_px + ab_le + (need_bottom ? txwpx : 0);
          av1_filter_intra_edge_high(left_col - ab_le, n_px, strength);
        }
      }
      upsample_above =
          av1_use_intra_edge_upsample(txwpx, txhpx, p_angle - 90, filt_type);
      if (need_above && upsample_above) {
        const int n_px = txwpx + (need_right ? txhpx : 0);
        av1_upsample_intra_edge_high(above_row, n_px, xd->bd);
      }
      upsample_left =
          av1_use_intra_edge_upsample(txhpx, txwpx, p_angle - 180, filt_type);
      if (need_left && upsample_left) {
        const int n_px = txhpx + (need_bottom ? txwpx : 0);
        av1_upsample_intra_edge_high(left_col, n_px, xd->bd);
      }
    }
    highbd_dr_predictor(dst, dst_stride, tx_size, above_row, left_col,
                        upsample_above, upsample_left, p_angle, xd->bd
#if CONFIG_MRLS
                        ,
                        mrl_index
#endif
    );
#if CONFIG_ORIP
    // Apply sub-block based filter for horizontal/vertical intra mode
    apply_sub_block_based_refinement_filter &=
        av1_allow_orip_dir(p_angle, disable_intra_pred_filter_for_hor_ver_mode);
    if (apply_sub_block_based_refinement_filter) {
      av1_apply_orip_4x4subblock_hbd(dst, dst_stride, tx_size, above_row,
                                     left_col, mode, xd->bd);
    }
#endif
    return;
  }

  // predict
  if (mode == DC_PRED) {
    dc_pred_high[n_left_px > 0][n_top_px > 0][tx_size](
        dst, dst_stride, above_row, left_col, xd->bd);
  } else {
    pred_high[mode][tx_size](dst, dst_stride, above_row, left_col, xd->bd);
  }

#if CONFIG_ORIP
  // Apply sub-block based filter for DC/smooth intra mode
  apply_sub_block_based_refinement_filter &=
      av1_allow_orip_smooth_dc(mode, plane);
  if (apply_sub_block_based_refinement_filter) {
    av1_apply_orip_4x4subblock_hbd(dst, dst_stride, tx_size, above_row,
                                   left_col, mode, xd->bd);
  }
#endif
}

static void build_intra_predictors(
    const MACROBLOCKD *xd, const uint8_t *ref, int ref_stride, uint8_t *dst,
    int dst_stride, PREDICTION_MODE mode, int angle_delta,
    FILTER_INTRA_MODE filter_intra_mode, TX_SIZE tx_size,
    int disable_edge_filter, int n_top_px, int n_topright_px, int n_left_px,
    int n_bottomleft_px, int plane
#if CONFIG_MRLS
    ,
    int is_sb_boundary
#endif
#if CONFIG_ORIP
    ,
    const int disable_intra_pred_filter_for_hor_ver_mode,
    const int seq_intra_pred_filter_flag
#endif
) {
  int i;
#if CONFIG_MRLS
  const uint8_t mrl_index =
      (plane == PLANE_TYPE_Y && is_inter_block(xd->mi[0]
#if CONFIG_SDP
                                               ,
                                               xd->tree_type
#endif
                                               ) == 0)
          ? xd->mi[0]->mrl_index
          : 0;
  const int above_mrl_idx = is_sb_boundary ? 0 : mrl_index;
  const uint8_t *above_ref = ref - ref_stride * (above_mrl_idx + 1);
  const uint8_t *left_ref = ref - 1 - mrl_index;
#else
  const uint8_t *above_ref = ref - ref_stride;
  const uint8_t *left_ref = ref - 1;
#endif
  DECLARE_ALIGNED(16, uint8_t, left_data[NUM_INTRA_NEIGHBOUR_PIXELS]);
  DECLARE_ALIGNED(16, uint8_t, above_data[NUM_INTRA_NEIGHBOUR_PIXELS]);
#if CONFIG_MRLS
  uint8_t *const above_row = above_data + 32;
  uint8_t *const left_col = left_data + 32;
#else
  uint8_t *const above_row = above_data + 16;
  uint8_t *const left_col = left_data + 16;
#endif
  const int txwpx = tx_size_wide[tx_size];
  const int txhpx = tx_size_high[tx_size];
  int need_left = extend_modes[mode] & NEED_LEFT;
  int need_above = extend_modes[mode] & NEED_ABOVE;
  int need_above_left = extend_modes[mode] & NEED_ABOVELEFT;
  int p_angle = 0;
  const int is_dr_mode = av1_is_directional_mode(mode);
  const int use_filter_intra = filter_intra_mode != FILTER_INTRA_MODES;
  // The left_data, above_data buffers must be zeroed to fix some intermittent
  // valgrind errors. Uninitialized reads in intra pred modules (e.g. width = 4
  // path in av1_dr_prediction_z1_avx2()) from left_data, above_data are seen to
  // be the potential reason for this issue.
  memset(left_data, 129, NUM_INTRA_NEIGHBOUR_PIXELS);
  memset(above_data, 127, NUM_INTRA_NEIGHBOUR_PIXELS);

  // The default values if ref pixels are not available:
  // 128 127 127 .. 127 127 127 127 127 127
  // 129  A   B  ..  Y   Z
  // 129  C   D  ..  W   X
  // 129  E   F  ..  U   V
  // 129  G   H  ..  S   T   T   T   T   T
  // ..

#if CONFIG_ORIP
#if CONFIG_MRLS
  int apply_sub_block_based_refinement_filter =
      seq_intra_pred_filter_flag && (mrl_index == 0);
#else
  int apply_sub_block_based_refinement_filter = seq_intra_pred_filter_flag;
#endif
#endif

  if (is_dr_mode) {
    p_angle = mode_to_angle_map[mode] + angle_delta;
    if (p_angle <= 90)
      need_above = 1, need_left = 0, need_above_left = 1;
    else if (p_angle < 180)
      need_above = 1, need_left = 1, need_above_left = 1;
    else
      need_above = 0, need_left = 1, need_above_left = 1;

#if CONFIG_ORIP
    if (apply_sub_block_based_refinement_filter &&
        (p_angle == 90 || p_angle == 180) &&
        !disable_intra_pred_filter_for_hor_ver_mode) {
      need_above = 1;
      need_left = 1;
      need_above_left = 1;
    }
#endif
  }
  if (use_filter_intra) need_left = need_above = need_above_left = 1;

  assert(n_top_px >= 0);
  assert(n_topright_px >= 0);
  assert(n_left_px >= 0);
  assert(n_bottomleft_px >= 0);

  if ((!need_above && n_left_px == 0) || (!need_left && n_top_px == 0)) {
    int val;
    if (need_left) {
      val = (n_top_px > 0) ? above_ref[0] : 129;
    } else {
      val = (n_left_px > 0) ? left_ref[0] : 127;
    }
    for (i = 0; i < txhpx; ++i) {
      memset(dst, val, txwpx);
      dst += dst_stride;
    }
    return;
  }

  // NEED_LEFT
  if (need_left) {
    int need_bottom = extend_modes[mode] & NEED_BOTTOMLEFT;
    if (use_filter_intra) need_bottom = 0;
    if (is_dr_mode) need_bottom = p_angle > 180;
#if CONFIG_MRLS
    const int num_left_pixels_needed =
        txhpx + (need_bottom ? txwpx : 3) + (mrl_index << 1);
#else
    const int num_left_pixels_needed = txhpx + (need_bottom ? txwpx : 0);
#endif
    i = 0;
    if (n_left_px > 0) {
      for (; i < n_left_px; i++) left_col[i] = left_ref[i * ref_stride];
      if (need_bottom && n_bottomleft_px > 0) {
        assert(i == txhpx);
        for (; i < txhpx + n_bottomleft_px; i++)
          left_col[i] = left_ref[i * ref_stride];
      }
      if (i < num_left_pixels_needed)
        memset(&left_col[i], left_col[i - 1], num_left_pixels_needed - i);
    } else if (n_top_px > 0) {
      memset(left_col, above_ref[0], num_left_pixels_needed);
    }
  }

  // NEED_ABOVE
  if (need_above) {
    int need_right = extend_modes[mode] & NEED_ABOVERIGHT;
    if (use_filter_intra) need_right = 0;
    if (is_dr_mode) need_right = p_angle < 90;
#if CONFIG_MRLS
    const int num_top_pixels_needed =
        txwpx + (need_right ? txhpx : 0) + (mrl_index << 1);
#else
    const int num_top_pixels_needed = txwpx + (need_right ? txhpx : 0);
#endif
    if (n_top_px > 0) {
      memcpy(above_row, above_ref, n_top_px);
      i = n_top_px;
      if (need_right && n_topright_px > 0) {
        assert(n_top_px == txwpx);
        memcpy(above_row + txwpx, above_ref + txwpx, n_topright_px);
        i += n_topright_px;
      }
      if (i < num_top_pixels_needed)
        memset(&above_row[i], above_row[i - 1], num_top_pixels_needed - i);
    } else if (n_left_px > 0) {
      memset(above_row, left_ref[0], num_top_pixels_needed);
    }
  }

  if (need_above_left) {
#if CONFIG_MRLS
    for (i = 1; i <= mrl_index + 1; i++) {
      if (n_top_px > 0 && n_left_px > 0) {
        above_row[-i] = above_ref[-i];
        if (is_sb_boundary)
          left_col[-i] = left_ref[-ref_stride];
        else
          left_col[-i] = left_ref[-i * ref_stride];
      } else if (n_top_px > 0) {
        above_row[-i] = left_col[-i] = above_ref[0];
      } else if (n_left_px > 0) {
        above_row[-i] = left_col[-i] = left_ref[0];
      } else {
        above_row[-i] = left_col[-i] = 128;
      }
    }
#else
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
#endif
  }

  if (use_filter_intra) {
    av1_filter_intra_predictor(dst, dst_stride, tx_size, above_row, left_col,
                               filter_intra_mode);
    return;
  }

  if (is_dr_mode) {
    int upsample_above = 0;
    int upsample_left = 0;
#if CONFIG_MRLS
    if (!disable_edge_filter && mrl_index == 0) {
#else
    if (!disable_edge_filter) {
#endif
      const int need_right = p_angle < 90;
      const int need_bottom = p_angle > 180;
      const int filt_type = get_filt_type(xd, plane);
      if (p_angle != 90 && p_angle != 180) {
        const int ab_le = need_above_left ? 1 : 0;
        if (need_above && need_left && (txwpx + txhpx >= 24)) {
          filter_intra_edge_corner(above_row, left_col);
        }
        if (need_above && n_top_px > 0) {
          const int strength =
              intra_edge_filter_strength(txwpx, txhpx, p_angle - 90, filt_type);
          const int n_px = n_top_px + ab_le + (need_right ? txhpx : 0);
          av1_filter_intra_edge(above_row - ab_le, n_px, strength);
        }
        if (need_left && n_left_px > 0) {
          const int strength = intra_edge_filter_strength(
              txhpx, txwpx, p_angle - 180, filt_type);
          const int n_px = n_left_px + ab_le + (need_bottom ? txwpx : 0);
          av1_filter_intra_edge(left_col - ab_le, n_px, strength);
        }
      }
      upsample_above =
          av1_use_intra_edge_upsample(txwpx, txhpx, p_angle - 90, filt_type);
      if (need_above && upsample_above) {
        const int n_px = txwpx + (need_right ? txhpx : 0);
        av1_upsample_intra_edge(above_row, n_px);
      }
      upsample_left =
          av1_use_intra_edge_upsample(txhpx, txwpx, p_angle - 180, filt_type);
      if (need_left && upsample_left) {
        const int n_px = txhpx + (need_bottom ? txwpx : 0);
        av1_upsample_intra_edge(left_col, n_px);
      }
    }
    dr_predictor(dst, dst_stride, tx_size, above_row, left_col, upsample_above,
                 upsample_left, p_angle
#if CONFIG_MRLS
                 ,
                 mrl_index
#endif
    );

#if CONFIG_ORIP
    // Apply sub-block based filter for horizontal/vertical intra mode
    apply_sub_block_based_refinement_filter &=
        av1_allow_orip_dir(p_angle, disable_intra_pred_filter_for_hor_ver_mode);
    if (apply_sub_block_based_refinement_filter) {
      av1_apply_orip_4x4subblock(dst, dst_stride, tx_size, above_row, left_col,
                                 mode);
    }
#endif
    return;
  }

  // predict
  if (mode == DC_PRED) {
    dc_pred[n_left_px > 0][n_top_px > 0][tx_size](dst, dst_stride, above_row,
                                                  left_col);
  } else {
    pred[mode][tx_size](dst, dst_stride, above_row, left_col);
  }

#if CONFIG_ORIP
  apply_sub_block_based_refinement_filter &=
      av1_allow_orip_smooth_dc(mode, plane);
  if (apply_sub_block_based_refinement_filter) {
    av1_apply_orip_4x4subblock(dst, dst_stride, tx_size, above_row, left_col,
                               mode);
  }
#endif
}

#define ARITHMETIC_LEFT_SHIFT(x, shift) \
  (((x) >= 0) ? ((x) << (shift)) : (-((-(x)) << (shift))))

void av1_predict_intra_block(
    const AV1_COMMON *cm, const MACROBLOCKD *xd, int wpx, int hpx,
    TX_SIZE tx_size, PREDICTION_MODE mode, int angle_delta, int use_palette,
    FILTER_INTRA_MODE filter_intra_mode, const uint8_t *ref, int ref_stride,
    uint8_t *dst, int dst_stride, int col_off, int row_off, int plane
#if CONFIG_ORIP
    ,
    const int disable_intra_pred_filter_for_hor_ver_mode
#endif
) {
  const MB_MODE_INFO *const mbmi = xd->mi[0];
  const int txwpx = tx_size_wide[tx_size];
  const int txhpx = tx_size_high[tx_size];
  const int x = col_off << MI_SIZE_LOG2;
  const int y = row_off << MI_SIZE_LOG2;

  if (use_palette) {
    int r, c;
    const uint8_t *const map = xd->plane[plane != 0].color_index_map +
                               xd->color_index_map_offset[plane != 0];
    const uint16_t *const palette =
        mbmi->palette_mode_info.palette_colors + plane * PALETTE_MAX_SIZE;
    if (is_cur_buf_hbd(xd)) {
      uint16_t *dst16 = CONVERT_TO_SHORTPTR(dst);
      for (r = 0; r < txhpx; ++r) {
        for (c = 0; c < txwpx; ++c) {
          dst16[r * dst_stride + c] = palette[map[(r + y) * wpx + c + x]];
        }
      }
    } else {
      for (r = 0; r < txhpx; ++r) {
        for (c = 0; c < txwpx; ++c) {
          dst[r * dst_stride + c] =
              (uint8_t)palette[map[(r + y) * wpx + c + x]];
        }
      }
    }
    return;
  }

  const struct macroblockd_plane *const pd = &xd->plane[plane];
  const int txw = tx_size_wide_unit[tx_size];
  const int txh = tx_size_high_unit[tx_size];
  const int ss_x = pd->subsampling_x;
  const int ss_y = pd->subsampling_y;
  const int have_top =
      row_off || (ss_y ? xd->chroma_up_available : xd->up_available);
  const int have_left =
      col_off || (ss_x ? xd->chroma_left_available : xd->left_available);
  const int mi_row = -xd->mb_to_top_edge >> (3 + MI_SIZE_LOG2);
  const int mi_col = -xd->mb_to_left_edge >> (3 + MI_SIZE_LOG2);
#if CONFIG_SDP
  BLOCK_SIZE bsize = mbmi->sb_type[plane > 0];
#else   // CONFIG_SDP
  BLOCK_SIZE bsize = mbmi->sb_type;
#endif  // CONFIG_SDP
  const int mi_wide = mi_size_wide[bsize];
  const int mi_high = mi_size_high[bsize];

  // Distance between the right edge of this prediction block to
  // the tile right edge
  const int xr =
      ARITHMETIC_LEFT_SHIFT(xd->tile.mi_col_end - mi_col - mi_wide, 2 - ss_x) +
      wpx - x - txwpx;
  // Distance between the bottom edge of this prediction block to
  // the tile bottom edge
  const int yd =
      ARITHMETIC_LEFT_SHIFT(xd->tile.mi_row_end - mi_row - mi_high, 2 - ss_y) +
      hpx - y - txhpx;
  const int right_available =
      mi_col + ((col_off + txw) << ss_x) < xd->tile.mi_col_end;
  const int bottom_available =
      (yd > 0) && (mi_row + ((row_off + txh) << ss_y) < xd->tile.mi_row_end);

  const BLOCK_SIZE init_bsize = bsize;
  // force 4x4 chroma component block size.
  if (ss_x || ss_y) {
    bsize = mbmi->chroma_ref_info.bsize_base;
  }

#if CONFIG_EXT_RECUR_PARTITIONS
  int px_top_right = 0;
  const int have_top_right = has_top_right(
      cm, xd, bsize, mi_row, mi_col, have_top, right_available, tx_size,
      row_off, col_off, ss_x, ss_y, xr, &px_top_right, bsize != init_bsize);
#else
  const PARTITION_TYPE partition = mbmi->partition;
  const int have_top_right =
      has_top_right(cm, bsize, mi_row, mi_col, have_top, right_available,
                    partition, tx_size, row_off, col_off, ss_x, ss_y);
#endif

  int px_bottom_left = 0;
  const int have_bottom_left = has_bottom_left(
      cm, xd, bsize, mi_row, mi_col, bottom_available, have_left, tx_size,
      row_off, col_off, ss_x, ss_y, yd, &px_bottom_left, bsize != init_bsize);

  const int disable_edge_filter = !cm->seq_params.enable_intra_edge_filter;

#if CONFIG_MRLS
  const int is_sb_boundary =
      (mi_row % cm->seq_params.mib_size == 0 && row_off == 0) ? 1 : 0;
#endif

  if (is_cur_buf_hbd(xd)) {
    build_intra_predictors_high(
        xd, ref, ref_stride, dst, dst_stride, mode, angle_delta,
        filter_intra_mode, tx_size, disable_edge_filter,
        have_top ? AOMMIN(txwpx, xr + txwpx) : 0,
#if CONFIG_EXT_RECUR_PARTITIONS
        have_top_right ? px_top_right : 0,
#else
        have_top_right ? AOMMIN(txwpx, xr) : 0,
#endif
        have_left ? AOMMIN(txhpx, yd + txhpx) : 0,
        have_bottom_left ? px_bottom_left : 0, plane
#if CONFIG_MRLS
        ,
        is_sb_boundary
#endif
#if CONFIG_ORIP
        ,
        disable_intra_pred_filter_for_hor_ver_mode, cm->seq_params.enable_orip
#endif
    );
    return;
  }

  build_intra_predictors(
      xd, ref, ref_stride, dst, dst_stride, mode, angle_delta,
      filter_intra_mode, tx_size, disable_edge_filter,
      have_top ? AOMMIN(txwpx, xr + txwpx) : 0,
#if CONFIG_EXT_RECUR_PARTITIONS
      have_top_right ? px_top_right : 0,
#else
      have_top_right ? AOMMIN(txwpx, xr) : 0,
#endif
      have_left ? AOMMIN(txhpx, yd + txhpx) : 0,
      have_bottom_left ? px_bottom_left : 0, plane
#if CONFIG_MRLS
      ,
      is_sb_boundary
#endif
#if CONFIG_ORIP
      ,
      disable_intra_pred_filter_for_hor_ver_mode, cm->seq_params.enable_orip
#endif
  );
}

#undef ARITHMETIC_LEFT_SHIFT

void av1_predict_intra_block_facade(const AV1_COMMON *cm, MACROBLOCKD *xd,
                                    int plane, int blk_col, int blk_row,
                                    TX_SIZE tx_size) {
  const MB_MODE_INFO *const mbmi = xd->mi[0];
  struct macroblockd_plane *const pd = &xd->plane[plane];
  const int dst_stride = pd->dst.stride;
  uint8_t *dst = &pd->dst.buf[(blk_row * dst_stride + blk_col) << MI_SIZE_LOG2];
  const PREDICTION_MODE mode =
      (plane == AOM_PLANE_Y) ? mbmi->mode : get_uv_mode(mbmi->uv_mode);
  const int use_palette = mbmi->palette_mode_info.palette_size[plane != 0] > 0;
  const FILTER_INTRA_MODE filter_intra_mode =
      (plane == AOM_PLANE_Y && mbmi->filter_intra_mode_info.use_filter_intra)
          ? mbmi->filter_intra_mode_info.filter_intra_mode
          : FILTER_INTRA_MODES;

#if CONFIG_ORIP
#if CONFIG_SDP
  const BLOCK_SIZE bsize = mbmi->sb_type[AOM_PLANE_Y];
#else
  const BLOCK_SIZE bsize = mbmi->sb_type;
#endif
  const int angle_delta =
      cm->seq_params.enable_orip
          ? ((mbmi->angle_delta[plane != AOM_PLANE_Y] == ANGLE_DELTA_VALUE_ORIP)
                 ? 0
                 : mbmi->angle_delta[plane != AOM_PLANE_Y] * ANGLE_STEP)
          : mbmi->angle_delta[plane != AOM_PLANE_Y] * ANGLE_STEP;
  const int disable_intra_pred_filter_for_hor_ver_mode =
      cm->seq_params.enable_orip
          ? ((mbmi->angle_delta[plane != AOM_PLANE_Y] == ANGLE_DELTA_VALUE_ORIP)
                 ? 1
                 : 0)
          : 1;
  if (cm->seq_params.enable_orip) {
    int violate_orip = disable_intra_pred_filter_for_hor_ver_mode;
    violate_orip &= !av1_signal_orip_for_horver_modes(
        cm, mbmi, plane != AOM_PLANE_Y, bsize);
    assert(violate_orip == 0);
  }

#else
  const int angle_delta = mbmi->angle_delta[plane != AOM_PLANE_Y] * ANGLE_STEP;
#endif

  if (plane != AOM_PLANE_Y && mbmi->uv_mode == UV_CFL_PRED) {
#if CONFIG_DEBUG
    assert(is_cfl_allowed(xd));
#if CONFIG_SDP && CONFIG_EXT_RECUR_PARTITIONS
    const BLOCK_SIZE plane_bsize = get_mb_plane_block_size(
        xd, mbmi, plane, pd->subsampling_x, pd->subsampling_y);
#else
    const BLOCK_SIZE plane_bsize = get_plane_block_size(
#if CONFIG_SDP
        mbmi->sb_type[xd->tree_type == CHROMA_PART], pd->subsampling_x,
        pd->subsampling_y);
#else
        mbmi->chroma_ref_info.bsize_base, pd->subsampling_x, pd->subsampling_y);
#endif  // CONFIG_SDP
#endif  // CONFIG_SDP && CONFIG_EXT_RECUR_PARTITIONS
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
    CFL_PRED_TYPE pred_plane = get_cfl_pred_type(plane);
    if (cfl->dc_pred_is_cached[pred_plane] == 0) {
      av1_predict_intra_block(cm, xd, pd->width, pd->height, tx_size, mode,
                              angle_delta, use_palette, filter_intra_mode, dst,
                              dst_stride, dst, dst_stride, blk_col, blk_row,
#if CONFIG_ORIP
                              plane,
                              disable_intra_pred_filter_for_hor_ver_mode);
#else
                              plane);
#endif

      if (cfl->use_dc_pred_cache) {
        cfl_store_dc_pred(xd, dst, pred_plane, tx_size_wide[tx_size]);
        cfl->dc_pred_is_cached[pred_plane] = 1;
      }
    } else {
      cfl_load_dc_pred(xd, dst, dst_stride, tx_size, pred_plane);
    }
#if CONFIG_SDP
    if (xd->tree_type == CHROMA_PART) {
      const int luma_tx_size =
          av1_get_max_uv_txsize(mbmi->sb_type[PLANE_TYPE_UV], 0, 0);
      cfl_store_tx(xd, blk_row, blk_col, luma_tx_size);
    }
#endif
    cfl_predict_block(xd, dst, dst_stride, tx_size, plane);
    return;
  }

  av1_predict_intra_block(cm, xd, pd->width, pd->height, tx_size, mode,
                          angle_delta, use_palette, filter_intra_mode, dst,
#if CONFIG_ORIP
                          dst_stride, dst, dst_stride, blk_col, blk_row, plane,
                          disable_intra_pred_filter_for_hor_ver_mode);
#else
                          dst_stride, dst, dst_stride, blk_col, blk_row, plane);
#endif
}

void av1_init_intra_predictors(void) {
  aom_once(init_intra_predictors_internal);
}

#if CONFIG_ORIP

DECLARE_ALIGNED(16, const int8_t,
                av1_sub_block_filter_intra_taps_4x4[16][9]) = {
  { 4, 16, 4, 0, 0, 16, 4, 0, 0 }, { 2, 4, 16, 4, 0, 8, 2, 0, 0 },
  { 1, 0, 4, 16, 4, 4, 1, 0, 0 },  { 0, 0, 2, 4, 16, 2, 0, 0, 0 },

  { 2, 8, 2, 0, 0, 4, 16, 4, 0 },  { 0, 2, 8, 2, 0, 2, 8, 2, 0 },
  { 0, 0, 2, 8, 2, 1, 4, 1, 0 },   { 0, 0, 0, 2, 8, 1, 2, 0, 0 },

  { 0, 4, 0, 0, 0, 0, 4, 16, 4 },  { 0, 0, 4, 0, 0, 0, 2, 8, 2 },
  { 0, 0, 1, 4, 1, 0, 1, 4, 1 },   { 0, 0, 0, 2, 4, 0, 0, 4, 0 },

  { 0, 0, 1, 0, 0, 0, 2, 4, 16 },  { 0, 0, 0, 1, 0, 0, 1, 2, 8 },
  { 0, 0, 1, 2, 1, 0, 0, 1, 4 },   { 0, 0, 0, 1, 2, 0, 0, 1, 2 },
};

void av1_apply_orip_4x4subblock(uint8_t *dst, ptrdiff_t stride, TX_SIZE tx_size,
                                const uint8_t *above, const uint8_t *left,
                                PREDICTION_MODE mode) {
  // initialize references for the first row
  uint8_t ref_samples_sb_row[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  uint8_t left_ref_tmp_for_next_sb[5] = { 0, 0, 0, 0, 0 };
  uint8_t ref_samples_sb_col[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  uint8_t top_ref_tmp_for_next_sb[5] = { 0, 0, 0, 0, 0 };

  const int bw = tx_size_wide[tx_size];
  const int bh = tx_size_high[tx_size];
  const int num_vertical_sb = (bh >> 2);
  const int num_top_ref = 5;
  const int num_left_ref = 4;

  uint8_t widthThreshold = (mode == H_PRED) ? 0 : AOMMIN((bw >> 2), 4);
  uint8_t heightThreshold = (mode == V_PRED) ? 0 : AOMMIN((bh >> 2), 4);

  memcpy(&ref_samples_sb_row[0], &above[-1],
         num_top_ref * sizeof(uint8_t));  // copy top reference
  memcpy(&ref_samples_sb_row[num_top_ref], &left[0],
         num_left_ref * sizeof(uint8_t));  // copy left reference
  // initialize references for the column
  if (num_vertical_sb > 1) {
    ref_samples_sb_col[0] = left[3];
    memcpy(&ref_samples_sb_col[1], &dst[3 * stride],
           (num_top_ref - 1) * sizeof(uint8_t));  // copy top reference
    memcpy(&ref_samples_sb_col[5], &left[4],
           num_left_ref * sizeof(uint8_t));  // copy left reference
  }

  // loop to process first row of sub-blocks
  for (int n = 0; n < (bw >> 2); n++) {
    int r_sb = 0;
    int c_sb = (n << 2);

    memcpy(&ref_samples_sb_row[0], &above[c_sb - 1],
           num_top_ref * sizeof(uint8_t));  // copy top reference

    // copy left reference for the next sub-blocks
    for (int q = 0; q < 4; q++)
      left_ref_tmp_for_next_sb[q] = dst[(r_sb + q) * stride + c_sb + 3];
    for (int k = 0; k < 16; ++k) {
      int r_pos = r_sb + (k >> 2);
      int c_pos = c_sb + (k & 0x03);
      if (!(c_pos >= widthThreshold && r_pos >= heightThreshold)) {
        int predvalue = (int)dst[stride * r_pos + c_pos];
        int offset = 0;
        for (int tap = 0; tap < 9; tap++) {
          int diff = (int)ref_samples_sb_row[tap] - predvalue;
          offset += av1_sub_block_filter_intra_taps_4x4[k][tap] * diff;
        }
        offset = (offset + 32) >> 6;
        int filteredpixelValue = predvalue + offset;
        dst[stride * r_pos + c_pos] = clip_pixel(filteredpixelValue);
      }
    }  // End of the subblock

    memcpy(&ref_samples_sb_row[num_top_ref], &left_ref_tmp_for_next_sb[0],
           num_left_ref *
               sizeof(uint8_t));  // copy left reference for the next sub-block
  }

  // process first column
  // loop to process first column of sub-blocks
  if (num_vertical_sb > 1) {
    for (int m = 1; m < num_vertical_sb; m++) {
      int r_sb = (m << 2);
      int c_sb = 0;

      ref_samples_sb_col[0] = left[r_sb - 1];

      memcpy(&ref_samples_sb_col[5], &left[r_sb],
             (num_top_ref - 1) * sizeof(uint8_t));  // copy left reference
      memcpy(&top_ref_tmp_for_next_sb[0], &dst[(r_sb + 3) * stride],
             num_left_ref * sizeof(uint8_t));  // copy top reference

      for (int k = 0; k < 16; ++k) {
        int r_pos = r_sb + (k >> 2);
        int c_pos = c_sb + (k & 0x03);
        if (!(c_pos >= widthThreshold && r_pos >= heightThreshold)) {
          int predvalue = (int)dst[stride * r_pos + c_pos];
          int offset = 0;
          for (int tap = 0; tap < 9; tap++) {
            int diff = (int)ref_samples_sb_col[tap] - predvalue;
            offset += av1_sub_block_filter_intra_taps_4x4[k][tap] * diff;
          }
          offset = (offset + 32) >> 6;
          int filteredpixelValue = predvalue + offset;
          dst[stride * r_pos + c_pos] = clip_pixel(filteredpixelValue);
        }
      }  // End of the subblock
      memcpy(&ref_samples_sb_col[1], &top_ref_tmp_for_next_sb[0],
             (num_top_ref - 1) *
                 sizeof(uint8_t));  // copy top reference for the next sub-block
    }
  }
}

void av1_apply_orip_4x4subblock_hbd(uint16_t *dst, ptrdiff_t stride,
                                    TX_SIZE tx_size, const uint16_t *above,
                                    const uint16_t *left, PREDICTION_MODE mode,
                                    int bd) {
  const int bw = tx_size_wide[tx_size];
  const int bh = tx_size_high[tx_size];

  // initialize references for the first row
  uint16_t ref_samples_sb_row[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  uint16_t left_ref_tmp_for_next_sb[5] = { 0, 0, 0, 0, 0 };
  uint16_t ref_samples_sb_col[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  uint16_t top_ref_tmp_for_next_sb[5] = { 0, 0, 0, 0, 0 };

  const int num_vertical_sb = (bh >> 2);
  const int num_top_ref = 5;
  const int num_left_ref = 4;

  uint8_t widthThreshold = (mode == H_PRED) ? 0 : AOMMIN((bw >> 2), 4);
  uint8_t heightThreshold = (mode == V_PRED) ? 0 : AOMMIN((bh >> 2), 4);

  memcpy(&ref_samples_sb_row[0], &above[-1],
         num_top_ref * sizeof(uint16_t));  // copy top reference
  memcpy(&ref_samples_sb_row[num_top_ref], &left[0],
         num_left_ref * sizeof(uint16_t));  // copy left reference

  // initialize references for the column
  if (num_vertical_sb > 1) {
    ref_samples_sb_col[0] = left[3];
    memcpy(&ref_samples_sb_col[1], &dst[3 * stride],
           (num_top_ref - 1) * sizeof(uint16_t));  // copy top reference
    memcpy(&ref_samples_sb_col[5], &left[4],
           num_left_ref * sizeof(uint16_t));  // copy left reference
  }

  // loop to process first row of sub-blocks
  for (int n = 0; n < (bw >> 2); n++) {
    int r_sb = 0;
    int c_sb = (n << 2);
    memcpy(&ref_samples_sb_row[0], &above[c_sb - 1],
           num_top_ref * sizeof(uint16_t));  // copy top reference

    // copy left reference for the next sub-blocks
    for (int q = 0; q < 4; q++)
      left_ref_tmp_for_next_sb[q] = dst[(r_sb + q) * stride + c_sb + 3];
    for (int k = 0; k < 16; ++k) {
      int r_pos = r_sb + (k >> 2);
      int c_pos = c_sb + (k & 0x03);
      if (!(c_pos >= widthThreshold && r_pos >= heightThreshold)) {
        int predvalue = (int)dst[stride * r_pos + c_pos];
        int offset = 0;
        for (int tap = 0; tap < 9; tap++) {
          int diff = (int)ref_samples_sb_row[tap] - predvalue;
          offset += av1_sub_block_filter_intra_taps_4x4[k][tap] * diff;
        }
        offset = (offset + 32) >> 6;
        int filteredpixelValue = predvalue + offset;
        dst[stride * r_pos + c_pos] = clip_pixel_highbd(filteredpixelValue, bd);
      }
    }  // End of the subblock
    memcpy(&ref_samples_sb_row[num_top_ref], &left_ref_tmp_for_next_sb[0],
           num_left_ref *
               sizeof(uint16_t));  // copy left reference for the next sub-block
  }

  // process first column
  // loop to process first column of sub-blocks
  if (num_vertical_sb > 1) {
    for (int m = 1; m < num_vertical_sb; m++) {
      int r_sb = (m << 2);
      int c_sb = 0;

      ref_samples_sb_col[0] = left[r_sb - 1];
      memcpy(&ref_samples_sb_col[5], &left[r_sb],
             (num_top_ref - 1) * sizeof(uint16_t));  // copy left reference
      memcpy(&top_ref_tmp_for_next_sb[0], &dst[(r_sb + 3) * stride],
             num_left_ref * sizeof(uint16_t));  // copy top reference

      for (int k = 0; k < 16; ++k) {
        int r_pos = r_sb + (k >> 2);
        int c_pos = c_sb + (k & 0x03);
        if (!(c_pos >= widthThreshold && r_pos >= heightThreshold)) {
          int predvalue = (int)dst[stride * r_pos + c_pos];
          int offset = 0;
          for (int tap = 0; tap < 9; tap++) {
            int diff = (int)ref_samples_sb_col[tap] - predvalue;
            offset += av1_sub_block_filter_intra_taps_4x4[k][tap] * diff;
          }
          offset = (offset + 32) >> 6;
          int filteredpixelValue = predvalue + offset;
          dst[stride * r_pos + c_pos] =
              clip_pixel_highbd(filteredpixelValue, bd);
        }
      }  // End of the subblock
      memcpy(
          &ref_samples_sb_col[1], &top_ref_tmp_for_next_sb[0],
          (num_top_ref - 1) *
              sizeof(uint16_t));  // copy top reference for the next sub-block
    }
  }
}
#endif
