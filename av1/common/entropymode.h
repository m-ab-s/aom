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

#ifndef AOM_AV1_COMMON_ENTROPYMODE_H_
#define AOM_AV1_COMMON_ENTROPYMODE_H_

#include "av1/common/entropy.h"
#include "av1/common/entropymv.h"
#include "av1/common/filter.h"
#include "av1/common/seg_common.h"
#include "aom_dsp/aom_filter.h"

#if CONFIG_INTRA_ENTROPY
#include "av1/common/intra_entropy_models.h"
#include "av1/common/nn_em.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define BLOCK_SIZE_GROUPS 4

#define TX_SIZE_CONTEXTS 3

#define INTER_OFFSET(mode) ((mode)-NEARESTMV)
#define INTER_COMPOUND_OFFSET(mode) (uint8_t)((mode)-NEAREST_NEARESTMV)

// Number of possible contexts for a color index.
// As can be seen from av1_get_palette_color_index_context(), the possible
// contexts are (2,0,0), (2,2,1), (3,2,0), (4,1,0), (5,0,0). These are mapped to
// a value from 0 to 4 using 'palette_color_index_context_lookup' table.
#define PALETTE_COLOR_INDEX_CONTEXTS 5

// Palette Y mode context for a block is determined by number of neighboring
// blocks (top and/or left) using a palette for Y plane. So, possible Y mode'
// context values are:
// 0 if neither left nor top block uses palette for Y plane,
// 1 if exactly one of left or top block uses palette for Y plane, and
// 2 if both left and top blocks use palette for Y plane.
#define PALETTE_Y_MODE_CONTEXTS 3

// Palette UV mode context for a block is determined by whether this block uses
// palette for the Y plane. So, possible values are:
// 0 if this block doesn't use palette for Y plane.
// 1 if this block uses palette for Y plane (i.e. Y palette size > 0).
#define PALETTE_UV_MODE_CONTEXTS 2

// Map the number of pixels in a block size to a context
//   64(BLOCK_8X8, BLOCK_4x16, BLOCK_16X4)  -> 0
//  128(BLOCK_8X16, BLOCK_16x8)             -> 1
//   ...
// 4096(BLOCK_64X64)                        -> 6
#define PALATTE_BSIZE_CTXS 7

#define KF_MODE_CONTEXTS 5

struct AV1Common;

typedef struct {
  const int16_t *scan;
  const int16_t *iscan;
} SCAN_ORDER;

#if CONFIG_INTRA_ENTROPY
#define SPARSE_FEATURE_ARRAYS(model_name, idx, weights_size)           \
  float model_name##_input_layer_sparse_##idx##_weights[weights_size]; \
  float model_name##_input_layer_dw_sparse_##idx[weights_size];

#define INPUT_LAYER_ARRAYS(model_name, dense_weights_size, output_size) \
  float model_name##_input_layer_dense_weights[dense_weights_size];     \
  float model_name##_input_layer_bias[output_size];                     \
  float model_name##_input_layer_output[output_size];                   \
  float model_name##_input_layer_dw_dense[dense_weights_size];          \
  float model_name##_input_layer_db[output_size];                       \
  float model_name##_input_layer_dy[output_size];

#define MODEL_ARRAYS(model_name, sparse_input_size, dense_input_size, \
                     output_size)                                     \
  int model_name##_sparse_features[sparse_input_size];                \
  float model_name##_dense_features[dense_input_size];                \
  float model_name##_output[output_size];
#endif  // CONFIG_INTRA_ENTROPY

#if CONFIG_ENTROPY_CONTEXTS
#define EOB_CONTEXTS 8
#endif  // CONFIG_ENTROPY_CONTEXTS

typedef struct frame_contexts {
  aom_cdf_prob txb_skip_cdf[TX_SIZES][TXB_SKIP_CONTEXTS][CDF_SIZE(2)];
  aom_cdf_prob eob_extra_cdf[TX_SIZES][PLANE_TYPES][EOB_COEF_CONTEXTS]
                            [CDF_SIZE(2)];
  aom_cdf_prob dc_sign_cdf[PLANE_TYPES][DC_SIGN_CONTEXTS][CDF_SIZE(2)];
#if CONFIG_ENTROPY_CONTEXTS
  aom_cdf_prob eob_flag_cdf16[EOB_CONTEXTS][PLANE_TYPES][2][CDF_SIZE(5)];
  aom_cdf_prob eob_flag_cdf32[EOB_CONTEXTS][PLANE_TYPES][2][CDF_SIZE(6)];
  aom_cdf_prob eob_flag_cdf64[EOB_CONTEXTS][PLANE_TYPES][2][CDF_SIZE(7)];
  aom_cdf_prob eob_flag_cdf128[EOB_CONTEXTS][PLANE_TYPES][2][CDF_SIZE(8)];
  aom_cdf_prob eob_flag_cdf256[EOB_CONTEXTS][PLANE_TYPES][2][CDF_SIZE(9)];
  aom_cdf_prob eob_flag_cdf512[EOB_CONTEXTS][PLANE_TYPES][2][CDF_SIZE(10)];
  aom_cdf_prob eob_flag_cdf1024[EOB_CONTEXTS][PLANE_TYPES][2][CDF_SIZE(11)];
#else
  aom_cdf_prob eob_flag_cdf16[PLANE_TYPES][2][CDF_SIZE(5)];
  aom_cdf_prob eob_flag_cdf32[PLANE_TYPES][2][CDF_SIZE(6)];
  aom_cdf_prob eob_flag_cdf64[PLANE_TYPES][2][CDF_SIZE(7)];
  aom_cdf_prob eob_flag_cdf128[PLANE_TYPES][2][CDF_SIZE(8)];
  aom_cdf_prob eob_flag_cdf256[PLANE_TYPES][2][CDF_SIZE(9)];
  aom_cdf_prob eob_flag_cdf512[PLANE_TYPES][2][CDF_SIZE(10)];
  aom_cdf_prob eob_flag_cdf1024[PLANE_TYPES][2][CDF_SIZE(11)];
#endif  // CONFIG_ENTROPY_CONTEXTS
  aom_cdf_prob coeff_base_eob_cdf[TX_SIZES][PLANE_TYPES][SIG_COEF_CONTEXTS_EOB]
                                 [CDF_SIZE(3)];
  aom_cdf_prob coeff_base_cdf[TX_SIZES][PLANE_TYPES][SIG_COEF_CONTEXTS]
                             [CDF_SIZE(4)];
  aom_cdf_prob coeff_br_cdf[TX_SIZES][PLANE_TYPES][LEVEL_CONTEXTS]
                           [CDF_SIZE(BR_CDF_SIZE)];

  aom_cdf_prob newmv_cdf[NEWMV_MODE_CONTEXTS][CDF_SIZE(2)];
  aom_cdf_prob zeromv_cdf[GLOBALMV_MODE_CONTEXTS][CDF_SIZE(2)];
  aom_cdf_prob refmv_cdf[REFMV_MODE_CONTEXTS][CDF_SIZE(2)];
  aom_cdf_prob drl_cdf[DRL_MODE_CONTEXTS][CDF_SIZE(2)];

  aom_cdf_prob inter_compound_mode_cdf[INTER_MODE_CONTEXTS]
                                      [CDF_SIZE(INTER_COMPOUND_MODES)];
  aom_cdf_prob compound_type_cdf[BLOCK_SIZES_ALL]
                                [CDF_SIZE(MASKED_COMPOUND_TYPES)];
  aom_cdf_prob wedge_idx_cdf[BLOCK_SIZES_ALL][CDF_SIZE(16)];
  aom_cdf_prob interintra_cdf[BLOCK_SIZE_GROUPS][CDF_SIZE(2)];
  aom_cdf_prob wedge_interintra_cdf[BLOCK_SIZES_ALL][CDF_SIZE(2)];
  aom_cdf_prob interintra_mode_cdf[BLOCK_SIZE_GROUPS]
                                  [CDF_SIZE(INTERINTRA_MODES)];
  aom_cdf_prob motion_mode_cdf[BLOCK_SIZES_ALL][CDF_SIZE(MOTION_MODES)];
  aom_cdf_prob obmc_cdf[BLOCK_SIZES_ALL][CDF_SIZE(2)];
  aom_cdf_prob palette_y_size_cdf[PALATTE_BSIZE_CTXS][CDF_SIZE(PALETTE_SIZES)];
  aom_cdf_prob palette_uv_size_cdf[PALATTE_BSIZE_CTXS][CDF_SIZE(PALETTE_SIZES)];
  aom_cdf_prob palette_y_color_index_cdf[PALETTE_SIZES]
                                        [PALETTE_COLOR_INDEX_CONTEXTS]
                                        [CDF_SIZE(PALETTE_COLORS)];
  aom_cdf_prob palette_uv_color_index_cdf[PALETTE_SIZES]
                                         [PALETTE_COLOR_INDEX_CONTEXTS]
                                         [CDF_SIZE(PALETTE_COLORS)];
  aom_cdf_prob palette_y_mode_cdf[PALATTE_BSIZE_CTXS][PALETTE_Y_MODE_CONTEXTS]
                                 [CDF_SIZE(2)];
  aom_cdf_prob palette_uv_mode_cdf[PALETTE_UV_MODE_CONTEXTS][CDF_SIZE(2)];
  aom_cdf_prob comp_inter_cdf[COMP_INTER_CONTEXTS][CDF_SIZE(2)];
  aom_cdf_prob single_ref_cdf[REF_CONTEXTS][SINGLE_REFS - 1][CDF_SIZE(2)];
  aom_cdf_prob comp_ref_type_cdf[COMP_REF_TYPE_CONTEXTS][CDF_SIZE(2)];
  aom_cdf_prob uni_comp_ref_cdf[UNI_COMP_REF_CONTEXTS][UNIDIR_COMP_REFS - 1]
                               [CDF_SIZE(2)];
  aom_cdf_prob comp_ref_cdf[REF_CONTEXTS][FWD_REFS - 1][CDF_SIZE(2)];
  aom_cdf_prob comp_bwdref_cdf[REF_CONTEXTS][BWD_REFS - 1][CDF_SIZE(2)];
#if CONFIG_NEW_TX_PARTITION
#if CONFIG_NEW_TX_PARTITION_EXT
  aom_cdf_prob txfm_partition_cdf[2][TXFM_PARTITION_CONTEXTS]
                                 [CDF_SIZE(TX_PARTITION_TYPES)];
#else   // CONFIG_NEW_TX_PARTITION_EXT
  aom_cdf_prob txfm_partition_cdf[TXFM_PARTITION_CONTEXTS]
                                 [CDF_SIZE(TX_PARTITION_TYPES)];
#endif  // CONFIG_NEW_TX_PARTITION_EXT
#else   // CONFIG_NEW_TX_PARTITION
  aom_cdf_prob txfm_partition_cdf[TXFM_PARTITION_CONTEXTS][CDF_SIZE(2)];
#endif  // CONFIG_NEW_TX_PARTITION
  aom_cdf_prob compound_index_cdf[COMP_INDEX_CONTEXTS][CDF_SIZE(2)];
  aom_cdf_prob comp_group_idx_cdf[COMP_GROUP_IDX_CONTEXTS][CDF_SIZE(2)];
  aom_cdf_prob skip_mode_cdfs[SKIP_CONTEXTS][CDF_SIZE(2)];
  aom_cdf_prob skip_cdfs[SKIP_CONTEXTS][CDF_SIZE(2)];
  aom_cdf_prob intra_inter_cdf[INTRA_INTER_CONTEXTS][CDF_SIZE(2)];
  nmv_context nmvc;
  nmv_context ndvc;
  aom_cdf_prob intrabc_cdf[CDF_SIZE(2)];
  struct segmentation_probs seg;
  aom_cdf_prob filter_intra_cdfs[BLOCK_SIZES_ALL][CDF_SIZE(2)];
  aom_cdf_prob filter_intra_mode_cdf[CDF_SIZE(FILTER_INTRA_MODES)];
#if CONFIG_ADAPT_FILTER_INTRA
  aom_cdf_prob adapt_filter_intra_cdfs[BLOCK_SIZES_ALL][CDF_SIZE(2)];
  aom_cdf_prob
      adapt_filter_intra_mode_cdf[CDF_SIZE(USED_ADAPT_FILTER_INTRA_MODES)];
#endif  // CONFIG_ADAPT_FILTER_INTRA
  aom_cdf_prob switchable_restore_cdf[CDF_SIZE(RESTORE_SWITCHABLE_TYPES)];
  aom_cdf_prob wiener_restore_cdf[CDF_SIZE(2)];
  aom_cdf_prob sgrproj_restore_cdf[CDF_SIZE(2)];
#if CONFIG_LOOP_RESTORE_CNN
  aom_cdf_prob cnn_restore_cdf[CDF_SIZE(2)];
#endif  // CONFIG_LOOP_RESTORE_CNN
  aom_cdf_prob y_mode_cdf[BLOCK_SIZE_GROUPS][CDF_SIZE(INTRA_MODES)];
  aom_cdf_prob partition_cdf[PARTITION_CONTEXTS][CDF_SIZE(EXT_PARTITION_TYPES)];
  aom_cdf_prob switchable_interp_cdf[SWITCHABLE_FILTER_CONTEXTS]
                                    [CDF_SIZE(SWITCHABLE_FILTERS)];
#if CONFIG_INTRA_ENTROPY
  SPARSE_FEATURE_ARRAYS(
      intra_y_mode, 0,
      EM_Y_SPARSE_FEAT_SIZE_0 *ALIGN_MULTIPLE_OF_FOUR(EM_Y_OUTPUT_SIZE));
  SPARSE_FEATURE_ARRAYS(
      intra_y_mode, 1,
      EM_Y_SPARSE_FEAT_SIZE_1 *ALIGN_MULTIPLE_OF_FOUR(EM_Y_OUTPUT_SIZE));
  INPUT_LAYER_ARRAYS(
      intra_y_mode,
      EM_NUM_Y_DENSE_FEATURES *ALIGN_MULTIPLE_OF_FOUR(EM_Y_OUTPUT_SIZE),
      ALIGN_MULTIPLE_OF_FOUR(EM_Y_OUTPUT_SIZE));
  MODEL_ARRAYS(intra_y_mode, EM_NUM_Y_SPARSE_FEATURES,
               ALIGN_MULTIPLE_OF_FOUR(EM_NUM_Y_DENSE_FEATURES),
               ALIGN_MULTIPLE_OF_FOUR(EM_Y_OUTPUT_SIZE));

  SPARSE_FEATURE_ARRAYS(
      intra_uv_mode, 0,
      EM_UV_SPARSE_FEAT_SIZE_0 *ALIGN_MULTIPLE_OF_FOUR(EM_UV_OUTPUT_SIZE));
  SPARSE_FEATURE_ARRAYS(
      intra_uv_mode, 1,
      EM_UV_SPARSE_FEAT_SIZE_1 *ALIGN_MULTIPLE_OF_FOUR(EM_UV_OUTPUT_SIZE));
  INPUT_LAYER_ARRAYS(intra_uv_mode, 0,
                     ALIGN_MULTIPLE_OF_FOUR(EM_UV_OUTPUT_SIZE));
  MODEL_ARRAYS(intra_uv_mode, EM_NUM_UV_SPARSE_FEATURES, 0,
               ALIGN_MULTIPLE_OF_FOUR(EM_UV_OUTPUT_SIZE));

  NN_CONFIG_EM intra_y_mode;
  NN_CONFIG_EM intra_uv_mode;
#else
  /* kf_y_cdf is discarded after use, so does not require persistent storage.
       However, we keep it with the other CDFs in this struct since it needs to
       be copied to each tile to support parallelism just like the others.
   */
  aom_cdf_prob kf_y_cdf[KF_MODE_CONTEXTS][KF_MODE_CONTEXTS]
                       [CDF_SIZE(INTRA_MODES)];
  aom_cdf_prob uv_mode_cdf[CFL_ALLOWED_TYPES][INTRA_MODES]
                          [CDF_SIZE(UV_INTRA_MODES)];
#endif  // CONFIG_INTRA_ENTROPY

#if CONFIG_FLEX_MVRES
#if DISALLOW_ONE_DOWN_FLEX_MVRES
  aom_cdf_prob
      flex_mv_precision_cdf[MV_SUBPEL_PRECISIONS - MV_SUBPEL_QTR_PRECISION]
                           [CDF_SIZE(MV_SUBPEL_PRECISIONS)];
#else
  aom_cdf_prob flex_mv_precision_cdf[MV_SUBPEL_PRECISIONS -
                                     MV_SUBPEL_QTR_PRECISION][CDF_SIZE(
      MV_SUBPEL_PRECISIONS - DISALLOW_ONE_DOWN_FLEX_MVRES)];
#endif  //
#endif  // CONFIG_FLEX_MVRES

  aom_cdf_prob angle_delta_cdf[DIRECTIONAL_MODES]
                              [CDF_SIZE(2 * MAX_ANGLE_DELTA + 1)];

  aom_cdf_prob tx_size_cdf[MAX_TX_CATS][TX_SIZE_CONTEXTS]
                          [CDF_SIZE(MAX_TX_DEPTH + 1)];
  aom_cdf_prob delta_q_cdf[CDF_SIZE(DELTA_Q_PROBS + 1)];
  aom_cdf_prob delta_lf_multi_cdf[FRAME_LF_COUNT][CDF_SIZE(DELTA_LF_PROBS + 1)];
  aom_cdf_prob delta_lf_cdf[CDF_SIZE(DELTA_LF_PROBS + 1)];
#if CONFIG_MODE_DEP_TX
#if USE_MDTX_INTER
  aom_cdf_prob mdtx_type_inter_cdf[EXT_TX_SIZES][CDF_SIZE(MDTX_TYPES_INTER)];
  aom_cdf_prob use_mdtx_inter_cdf[EXT_TX_SIZES][CDF_SIZE(2)];
#endif
#if USE_MDTX_INTRA
  aom_cdf_prob mdtx_type_intra_cdf[EXT_TX_SIZES][INTRA_MODES]
                                  [CDF_SIZE(MDTX_TYPES_INTRA)];
  aom_cdf_prob use_mdtx_intra_cdf[EXT_TX_SIZES][INTRA_MODES][CDF_SIZE(2)];
#endif
  aom_cdf_prob intra_ext_tx_cdf[EXT_TX_SETS_INTRA][EXT_TX_SIZES][INTRA_MODES]
                               [CDF_SIZE(TX_TYPES_NOMDTX)];
  aom_cdf_prob inter_ext_tx_cdf[EXT_TX_SETS_INTER][EXT_TX_SIZES]
                               [CDF_SIZE(TX_TYPES_NOMDTX)];
#else
  aom_cdf_prob intra_ext_tx_cdf[EXT_TX_SETS_INTRA][EXT_TX_SIZES][INTRA_MODES]
                               [CDF_SIZE(TX_TYPES)];
  aom_cdf_prob inter_ext_tx_cdf[EXT_TX_SETS_INTER][EXT_TX_SIZES]
                               [CDF_SIZE(TX_TYPES)];
#endif  // CONFIG_MODE_DEP_TX
  aom_cdf_prob cfl_sign_cdf[CDF_SIZE(CFL_JOINT_SIGNS)];
  aom_cdf_prob cfl_alpha_cdf[CFL_ALPHA_CONTEXTS][CDF_SIZE(CFL_ALPHABET_SIZE)];
  int initialized;
} FRAME_CONTEXT;

#if CONFIG_MODE_DEP_TX
static const int av1_ext_tx_ind[EXT_TX_SET_TYPES][TX_TYPES_NOMDTX] = {
#else
static const int av1_ext_tx_ind[EXT_TX_SET_TYPES][TX_TYPES] = {
#endif
  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 1, 3, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 1, 5, 6, 4, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0 },
  { 3, 4, 5, 8, 6, 7, 9, 10, 11, 0, 1, 2, 0, 0, 0, 0 },
  { 7, 8, 9, 12, 10, 11, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6 },
};

#if CONFIG_MODE_DEP_TX
static const int av1_ext_tx_inv[EXT_TX_SET_TYPES][TX_TYPES_NOMDTX] = {
#else
static const int av1_ext_tx_inv[EXT_TX_SET_TYPES][TX_TYPES] = {
#endif
  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 9, 0, 3, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 9, 0, 10, 11, 3, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 9, 10, 11, 0, 1, 2, 4, 5, 3, 6, 7, 8, 0, 0, 0, 0 },
  { 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 4, 5, 3, 6, 7, 8 },
};

void av1_set_default_ref_deltas(int8_t *ref_deltas);
void av1_set_default_mode_deltas(int8_t *mode_deltas);
void av1_setup_frame_contexts(struct AV1Common *cm);
void av1_setup_past_independence(struct AV1Common *cm);

// Returns (int)ceil(log2(n)).
// NOTE: This implementation only works for n <= 2^30.
static INLINE int av1_ceil_log2(int n) {
  if (n < 2) return 0;
  int i = 1, p = 2;
  while (p < n) {
    i++;
    p = p << 1;
  }
  return i;
}

// Returns the context for palette color index at row 'r' and column 'c',
// along with the 'color_order' of neighbors and the 'color_idx'.
// The 'color_map' is a 2D array with the given 'stride'.
int av1_get_palette_color_index_context(const uint8_t *color_map, int stride,
                                        int r, int c, int palette_size,
                                        uint8_t *color_order, int *color_idx);

#if CONFIG_INTRA_ENTROPY
#define SETUP_SPARSE_FEATURE_POINTERS(model, idx)     \
  fc->model.input_layer.sparse_weights[idx] =         \
      fc->model##_input_layer_sparse_##idx##_weights; \
  fc->model.input_layer.dw_sparse[idx] =              \
      fc->model##_input_layer_dw_sparse_##idx;

#define SETUP_MODEL_POINTERS(model)                                            \
  fc->model.input_layer.dense_weights = fc->model##_input_layer_dense_weights; \
  fc->model.input_layer.bias = fc->model##_input_layer_bias;                   \
  fc->model.input_layer.output = fc->model##_input_layer_output;               \
  fc->model.input_layer.dy = fc->model##_input_layer_dy;                       \
  fc->model.input_layer.db = fc->model##_input_layer_db;                       \
  fc->model.input_layer.dw_dense = fc->model##_input_layer_dw_dense;           \
  fc->model.sparse_features = fc->model##_sparse_features;                     \
  fc->model.dense_features = fc->model##_dense_features;                       \
  fc->model.output = fc->model##_output;

static INLINE void av1_config_entropy_models(FRAME_CONTEXT *const fc) {
  SETUP_SPARSE_FEATURE_POINTERS(intra_y_mode, 0);
  SETUP_SPARSE_FEATURE_POINTERS(intra_y_mode, 1);
  SETUP_MODEL_POINTERS(intra_y_mode);

  SETUP_SPARSE_FEATURE_POINTERS(intra_uv_mode, 0);
  SETUP_SPARSE_FEATURE_POINTERS(intra_uv_mode, 1);
  SETUP_MODEL_POINTERS(intra_uv_mode);
}
#endif  // CONFIG_INTRA_ENTROPY

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_ENTROPYMODE_H_
