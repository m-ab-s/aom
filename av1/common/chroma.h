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

#include "config/aom_config.h"

#include "av1/common/common_data.h"
#include "av1/common/enums.h"

#ifdef __cplusplus
extern "C" {
#endif

static INLINE BLOCK_SIZE scale_chroma_bsize(BLOCK_SIZE bsize, int subsampling_x,
                                            int subsampling_y, int mi_row,
                                            int mi_col) {
  assert(subsampling_x >= 0 && subsampling_x < 2);
  assert(subsampling_y >= 0 && subsampling_y < 2);
#if CONFIG_3WAY_PARTITIONS
  const int bw = mi_size_wide[bsize];
  const int bh = mi_size_high[bsize];
  const int is_3rd_horz3_16x16_partition =
      (mi_row & 1) && (bw == 4) && (bh == 1);
  const int is_3rd_vert3_16x16_partition =
      (mi_col & 1) && (bw == 1) && (bh == 4);
  const int is_3way_part =
      is_3rd_horz3_16x16_partition || is_3rd_vert3_16x16_partition;
#else
  (void)mi_row;
  (void)mi_col;
  const int is_3way_part = 0;
#endif  // CONFIG_3WAY_PARTITIONS

  BLOCK_SIZE bs = bsize;
  switch (bsize) {
    case BLOCK_4X4:
      assert(!is_3way_part);
      if (subsampling_x == 1 && subsampling_y == 1)
        bs = BLOCK_8X8;
      else if (subsampling_x == 1)
        bs = BLOCK_8X4;
      else if (subsampling_y == 1)
        bs = BLOCK_4X8;
      break;
    case BLOCK_4X8:
      assert(!is_3way_part);
      if (subsampling_x == 1 && subsampling_y == 1)
        bs = BLOCK_8X8;
      else if (subsampling_x == 1)
        bs = BLOCK_8X8;
      else if (subsampling_y == 1)
        bs = BLOCK_4X8;
      break;
    case BLOCK_8X4:
      assert(!is_3way_part);
      if (subsampling_x == 1 && subsampling_y == 1)
        bs = BLOCK_8X8;
      else if (subsampling_x == 1)
        bs = BLOCK_8X4;
      else if (subsampling_y == 1)
        bs = BLOCK_8X8;
      break;
    case BLOCK_4X16:
      if (subsampling_x == 1 && subsampling_y == 1)
        bs = is_3way_part ? BLOCK_16X16 : BLOCK_8X16;
      else if (subsampling_x == 1)
        bs = is_3way_part ? BLOCK_16X16 : BLOCK_8X16;
      else if (subsampling_y == 1)
        bs = BLOCK_4X16;
      break;
    case BLOCK_16X4:
      if (subsampling_x == 1 && subsampling_y == 1)
        bs = is_3way_part ? BLOCK_16X16 : BLOCK_16X8;
      else if (subsampling_x == 1)
        bs = BLOCK_16X4;
      else if (subsampling_y == 1)
        bs = is_3way_part ? BLOCK_16X16 : BLOCK_16X8;
      break;
    default: break;
  }
  return bs;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_CHROMA_H_
