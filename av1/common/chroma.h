/*
 * Copyright (c) 2019, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#ifndef AOM_AV1_COMMON_CHROMA_H_
#define AOM_AV1_COMMON_CHROMA_H_

#include <assert.h>
#include <stdbool.h>

#include "config/aom_config.h"

#include "av1/common/common_data.h"
#include "av1/common/enums.h"

#ifdef __cplusplus
extern "C" {
#endif

// Given the chroma block-size, computes the luma block-size. E.g., if the
// chroma block-size is 4x4, and x-subsampling was done but not y-subsampling,
// the luma block-size is 8x4.
static INLINE BLOCK_SIZE scale_chroma_bsize(BLOCK_SIZE bsize,
                                            bool subsampling_x,
                                            bool subsampling_y, int mi_row,
                                            int mi_col) {
#if CONFIG_EXT_RECUR_PARTITIONS
  const int bw = mi_size_wide[bsize];
  const int bh = mi_size_high[bsize];
  const int is_3rd_horz3_16x16_partition =
      (mi_row & 1) && (bw == 4) && (bh == 1);
  const int is_3rd_vert3_16x16_partition =
      (mi_col & 1) && (bw == 1) && (bh == 4);
  const bool is_3way_part =
      is_3rd_horz3_16x16_partition || is_3rd_vert3_16x16_partition;
#else
  (void)mi_row;
  (void)mi_col;
  const bool is_3way_part = false;
#endif  // CONFIG_EXT_RECUR_PARTITIONS

  switch (bsize) {
    case BLOCK_4X4:
      assert(!is_3way_part);
      if (subsampling_x == 1 && subsampling_y == 1)
        return BLOCK_8X8;
      else if (subsampling_x == 1)
        return BLOCK_8X4;
      else if (subsampling_y == 1)
        return BLOCK_4X8;
      return BLOCK_4X4;

    case BLOCK_4X8:
      assert(!is_3way_part);
      if (subsampling_x == 1 && subsampling_y == 1)
        return BLOCK_8X8;
      else if (subsampling_x == 1)
        return BLOCK_8X8;
      else if (subsampling_y == 1)
        return BLOCK_4X8;
      return BLOCK_4X8;

    case BLOCK_8X4:
      assert(!is_3way_part);
      if (subsampling_x == 1 && subsampling_y == 1)
        return BLOCK_8X8;
      else if (subsampling_x == 1)
        return BLOCK_8X4;
      else if (subsampling_y == 1)
        return BLOCK_8X8;
      return BLOCK_8X4;

    case BLOCK_4X16:
      if (subsampling_x == 1 && subsampling_y == 1)
        return is_3way_part ? BLOCK_16X16 : BLOCK_8X16;
      else if (subsampling_x == 1)
        return is_3way_part ? BLOCK_16X16 : BLOCK_8X16;
      return BLOCK_4X16;

    case BLOCK_16X4:
      if (subsampling_x == 1 && subsampling_y == 1)
        return is_3way_part ? BLOCK_16X16 : BLOCK_16X8;
      else if (subsampling_x == 1)
        return BLOCK_16X4;
      else if (subsampling_y == 1)
        return is_3way_part ? BLOCK_16X16 : BLOCK_16X8;
      return BLOCK_16X4;

    default: return bsize;
  }
}

// As some chroma blocks may cover more than one luma block and only the last
// chroma block in such a case is a chroma reference, mi_row and mi_col need to
// be offset for such cases. This function returns the two corresponding
// offsets.
// Note: The offsets returned should be *deducted* from mi_row / mi_col by the
// callers.
static INLINE void get_mi_row_col_offsets(int mi_row, int mi_col, int ss_x,
                                          int ss_y, int bw, int bh,
                                          int *mi_row_offset,
                                          int *mi_col_offset) {
  *mi_row_offset = 0;
  *mi_col_offset = 0;
#if CONFIG_EXT_RECUR_PARTITIONS
  if (ss_x && (mi_col & 0x01)) {
    if (bw == 1 && bh == 4) {
      // Special case: 3rd vertical sub-block of a 16x16 vert3 partition.
      *mi_col_offset = 3;
    } else if (bw == 1) {
      *mi_col_offset = 1;
    } else if (bw == 2 && bh == 4) {
      // Special case: 2nd vertical sub-block of a 16x16 vert3 partition.
      *mi_col_offset = 1;
    }
  }
  if (ss_y && (mi_row & 0x01)) {
    if (bw == 4 && bh == 1) {
      // Special case: 3rd horizontal sub-block of a 16x16 horz3 partition.
      *mi_row_offset = 3;
    } else if (bh == 1) {
      *mi_row_offset = 1;
    } else if (bw == 4 && bh == 2) {
      // Special case: 2rd horizontal sub-block of a 16x16 horz3 partition.
      *mi_row_offset = 1;
    }
  }
#else
  if (ss_x && (mi_col & 0x01) && bw == 1) {
    *mi_col_offset = 1;
  }
  if (ss_y && (mi_row & 0x01) && bh == 1) {
    *mi_row_offset = 1;
  }
#endif  // CONFIG_EXT_RECUR_PARTITIONS
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_CHROMA_H_
