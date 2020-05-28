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

#ifndef AOM_AV1_COMMON_NN_RECON_H_
#define AOM_AV1_COMMON_NN_RECON_H_

#include "aom/aom_integer.h"

#include "av1/common/cnn_tflite.h"
#include "av1/common/enums.h"
#include "av1/common/onyxc_int.h"

#ifdef __cplusplus
extern "C" {
#endif

static INLINE void av1_cnn_recon(const MACROBLOCKD *xd, uint8_t *dst,
                                 int dst_stride, TX_SIZE tx_size) {
  assert(tx_size == TX_16X16);
  assert(!is_inter_block(xd->mi[0]));
  const int err_val = av1_cnn_recon_tflite(
      dst, dst_stride, tx_size_high[tx_size], tx_size_wide[tx_size]);
  assert(err_val);
  (void)err_val;
  (void)xd;
}

static INLINE int av1_is_block_nn_recon_eligible(const AV1_COMMON *cm,
                                                 const MB_MODE_INFO *mbmi,
                                                 TX_SIZE tx_size) {
  if (cm->tx_mode != TX_MODE_SELECT || !frame_is_intra_only(cm) ||
      is_inter_block(mbmi)) {
    return 0;
  }
  return tx_size == TX_16X16;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_NN_RECON_H_
