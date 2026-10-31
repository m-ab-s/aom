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

#ifndef AOM_AV1_COMMON_IDCT_H_
#define AOM_AV1_COMMON_IDCT_H_

#include <assert.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"

#include "av1/common/blockd.h"
#include "av1/common/common.h"
#include "av1/common/enums.h"
#include "aom_dsp/txfm_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*transform_1d)(const tran_low_t *, tran_low_t *);

typedef struct {
  transform_1d cols, rows;  // vertical and horizontal
} transform_2d;

#define MAX_TX_SCALE 1
static inline int av1_get_tx_scale(const TX_SIZE tx_size) {
  // (pels > 256) + (pels > 1024), where pels = tx_size_2d[tx_size].
  static const uint8_t tx_scale[TX_SIZES_ALL] = { 0, 0, 0, 1, 2, 0, 0, 0, 0, 1,
                                                  1, 2, 2, 0, 0, 0, 0, 1, 1 };
  assert(tx_size < TX_SIZES_ALL);
  return tx_scale[tx_size];
}

static inline void av1_inverse_transform_block(const MACROBLOCKD *xd,
                                               const tran_low_t *dqcoeff,
                                               int plane, TX_TYPE tx_type,
                                               TX_SIZE tx_size, uint8_t *dst,
                                               int stride, int eob,
                                               int reduced_tx_set) {
  if (!eob) return;

  assert(eob <= av1_get_max_eob(tx_size));
  (void)plane;

  TxfmParam txfm_param;
  txfm_param.tx_type = tx_type;
  txfm_param.tx_size = tx_size;
  txfm_param.eob = eob;
  txfm_param.lossless = xd->lossless[xd->mi[0]->segment_id];
  txfm_param.bd = xd->bd;
  txfm_param.is_hbd = is_cur_buf_hbd(xd);
#if !defined(NDEBUG)
  txfm_param.tx_set_type = av1_get_ext_tx_set_type(
      txfm_param.tx_size, is_inter_block(xd->mi[0]), reduced_tx_set);
  assert(av1_ext_tx_used[txfm_param.tx_set_type][txfm_param.tx_type]);
#else
  (void)reduced_tx_set;
#endif

  if (UNLIKELY(txfm_param.is_hbd)) {
    av1_highbd_inv_txfm_add(dqcoeff, dst, stride, &txfm_param);
  } else {
    av1_inv_txfm_add(dqcoeff, dst, stride, &txfm_param);
  }
}
void av1_highbd_iwht4x4_add(const tran_low_t *input, uint8_t *dest, int stride,
                            int eob, int bd);

static inline const int32_t *cast_to_int32(const tran_low_t *input) {
  static_assert(sizeof(int32_t) == sizeof(tran_low_t), "");
  return (const int32_t *)input;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_IDCT_H_
