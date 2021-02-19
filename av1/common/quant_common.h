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

#ifndef AOM_AV1_COMMON_QUANT_COMMON_H_
#define AOM_AV1_COMMON_QUANT_COMMON_H_

#include <stdbool.h>
#include "aom/aom_codec.h"
#include "av1/common/seg_common.h"
#include "av1/common/enums.h"
#include "av1/common/entropy.h"

#ifdef __cplusplus
extern "C" {
#endif

#if CONFIG_EXTQUANT
#define MINQ 0
#define QINDEX_BITS 9
#define QINDEX_BITS_UNEXT 8
#define MAXQ_8_BITS 255
#define MAXQ_OFFSET 24
#define MAXQ (255 + 4 * MAXQ_OFFSET)
#define MAXQ_10_BITS (255 + 2 * MAXQ_OFFSET)
#define QINDEX_RANGE (MAXQ - MINQ + 1)
#define QINDEX_RANGE_8_BITS (MAXQ_8_BITS - MINQ + 1)
#define QINDEX_RANGE_10_BITS (MAXQ_10_BITS - MINQ + 1)
#else
#define MINQ 0
#define MAXQ 255
#define QINDEX_BITS 8
#define QINDEX_RANGE (MAXQ - MINQ + 1)
#endif
// Total number of QM sets stored
#define QM_LEVEL_BITS 4
#define NUM_QM_LEVELS (1 << QM_LEVEL_BITS)
/* Range of QMS is between first and last value, with offset applied to inter
 * blocks*/
#define DEFAULT_QM_Y 10
#define DEFAULT_QM_U 11
#define DEFAULT_QM_V 12
#define DEFAULT_QM_FIRST 5
#define DEFAULT_QM_LAST 9

struct AV1Common;
struct CommonQuantParams;
struct macroblockd;

#if CONFIG_FLEX_STEPS
#define MAX_NUM_Q_STEP_INTERVALS 16
#define MAX_NUM_TABLES 8
#define MAX_NUM_Q_STEP_VAL 256

void set_qStep_table_mode_0_1(int qStep_mode, int num_qStep_intervals,
                              int *num_qsteps_in_interval);
void set_qStep_table_mode_2(int qStep_mode, int num_qStep_levels,
                            int *qSteps_level);
void set_qStep_table_mode_3(int qStep_mode, int num_qStep_intervals,
                            int *template_table_idx,
                            int *table_start_region_idx,
                            int *num_qsteps_in_table,
                            int *qSteps_level_in_table);
// void dump_qStep_table(int mode, int expt);  // kk delete
#endif

#if CONFIG_EXTQUANT
int32_t av1_dc_quant_QTX(int qindex, int delta, int base_dc_delta_q,
                         aom_bit_depth_t bit_depth);
int32_t av1_ac_quant_QTX(int qindex, int delta, aom_bit_depth_t bit_depth);
#else
int16_t av1_dc_quant_QTX(int qindex, int delta, aom_bit_depth_t bit_depth);
int16_t av1_ac_quant_QTX(int qindex, int delta, aom_bit_depth_t bit_depth);
#endif

int av1_get_qindex(const struct segmentation *seg, int segment_id,
                   int base_qindex
#if CONFIG_EXTQUANT
                   ,
                   aom_bit_depth_t bit_depth
#endif
);

// Returns true if we are using quantization matrix.
bool av1_use_qmatrix(const struct CommonQuantParams *quant_params,
                     const struct macroblockd *xd, int segment_id);

// Reduce the large number of quantizers to a smaller number of levels for which
// different matrices may be defined
static INLINE int aom_get_qmlevel(int qindex, int first, int last
#if CONFIG_EXTQUANT
                                  ,
                                  aom_bit_depth_t bit_depth
#endif
) {
#if CONFIG_EXTQUANT
  return first + (qindex * (last + 1 - first)) /
                     (bit_depth == AOM_BITS_8
                          ? QINDEX_RANGE_8_BITS
                          : bit_depth == AOM_BITS_10 ? QINDEX_RANGE_10_BITS
                                                     : QINDEX_RANGE);
#else
  return first + (qindex * (last + 1 - first)) / QINDEX_RANGE;
#endif
}

// Initialize all global quant/dequant matrices.
void av1_qm_init(struct CommonQuantParams *quant_params, int num_planes);

// Get global dequant matrix.
const qm_val_t *av1_iqmatrix(const struct CommonQuantParams *quant_params,
                             int qmlevel, int plane, TX_SIZE tx_size);
// Get global quant matrix.
const qm_val_t *av1_qmatrix(const struct CommonQuantParams *quant_params,
                            int qmlevel, int plane, TX_SIZE tx_size);

// Get either local / global dequant matrix as appropriate.
const qm_val_t *av1_get_iqmatrix(const struct CommonQuantParams *quant_params,
                                 const struct macroblockd *xd, int plane,
                                 TX_SIZE tx_size, TX_TYPE tx_type);
// Get either local / global quant matrix as appropriate.
const qm_val_t *av1_get_qmatrix(const struct CommonQuantParams *quant_params,
                                const struct macroblockd *xd, int plane,
                                TX_SIZE tx_size, TX_TYPE tx_type);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_QUANT_COMMON_H_
