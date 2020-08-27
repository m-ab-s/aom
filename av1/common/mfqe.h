/*
 * Copyright (c) 2020, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#ifndef AOM_AV1_COMMON_MFQE_H_
#define AOM_AV1_COMMON_MFQE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "config/av1_rtcd.h"
#include "config/aom_dsp_rtcd.h"

#include "av1/common/onyxc_int.h"

#define MSE_MAX_BITS 16
#define MSE_MAX (1 << MSE_MAX_BITS)
#define MFQE_MSE_THRESHOLD 0.8  // Threshold for block matching.
#define MFQE_NUM_REFS 3         // Number of reference frames used.
#define MFQE_N_GRID_SEARCH 13   // Number of points in grid search.

#define MFQE_BLOCK_SIZE BLOCK_8X8  // Block size used for MFQE.
#define MFQE_SCALE_SIZE 8          // Scale factor used for MFQE.
#define MFQE_PADDING_SIZE 8        // Padding size for buffers.

// Wrapper struct for only the luma plane.
typedef struct y_buffer_config {
  uint8_t *buffer;       // Pointer to buffer.
  uint8_t *buffer_orig;  // Pointer to buffer, including padding.
  int stride;
  int height;
  int width;
} Y_BUFFER_CONFIG;

// Motion vector information used in MFQE.
typedef struct mv_mfqe {
  MV mv;

  // Subpel quantities of motion vector in x, y directions.
  int subpel_x_qn;
  int subpel_y_qn;

  double alpha;       // Alpha value for weighted blending.
  uint8_t ref_index;  // Index of reference frame.
  uint8_t valid;      // Indicates if motion vector is valid.
} MV_MFQE;

// Constant for zero MV_MFQE, used for initialization.
static const MV_MFQE kZeroMvMFQE = { { 0, 0 }, 0, 0, 0, 0, 0 };

// Row offsets used in the initial grid search for MFQE motion estimation.
static const int16_t grid_search_rows[13] = {
  0, 0, 0, 0, 0, 1, 1, 1, -1, -1, -1, 2, -2,
};

// Column offsets used in the initial grid search for MFQE motion estimation.
static const int16_t grid_search_cols[13] = {
  -2, -1, 0, 1, 2, -1, 0, 1, -1, 0, 1, 0, 0,
};

// Compare two RefCntBuffers based on the base_qindex.
static INLINE int cmpref(const void *a, const void *b) {
  RefCntBuffer *ref1 = *((RefCntBuffer **)a);
  RefCntBuffer *ref2 = *((RefCntBuffer **)b);
  return ref1->base_qindex - ref2->base_qindex;
}

// Calculate the alpha weight value based on the current mse.
static INLINE double get_alpha_weight(double mse) {
  return (MFQE_MSE_THRESHOLD - mse) / (MFQE_MSE_THRESHOLD + 1);
}

// Actually apply In-Loop Multi-Frame Quality Enhancement to the tmp buffer,
// using the reference frames. Perform full-pixel motion search on 8x8 blocks,
// then perform finer-grained search to obtain subpel motion vectors. Finally,
// replace the blocks in current frame by interpolation.
void av1_apply_loop_mfqe(Y_BUFFER_CONFIG *tmp, RefCntBuffer *ref_frames[],
                         BLOCK_SIZE bsize, int scale, int high_bd, int bd);

// Apply In-Loop Multi-Frame Quality Enhancement from the decoder side.
void av1_decode_restore_mfqe(AV1_COMMON *cm, int scale, BLOCK_SIZE bsize);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_MFQE_H_
