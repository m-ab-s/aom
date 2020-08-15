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

#ifndef AOM_AV1_COMMON_SCAN_H_
#define AOM_AV1_COMMON_SCAN_H_

#include "aom/aom_integer.h"
#include "aom_ports/mem.h"

#include "av1/common/enums.h"
#include "av1/common/onyxc_int.h"
#include "av1/common/blockd.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_NEIGHBORS 2

enum {
  SCAN_MODE_ZIG_ZAG,
  SCAN_MODE_COL_DIAG,
  SCAN_MODE_ROW_DIAG,
  SCAN_MODE_COL_1D,
  SCAN_MODE_ROW_1D,
  SCAN_MODES
} UENUM1BYTE(SCAN_MODE);

extern const SCAN_ORDER av1_default_scan_orders[TX_SIZES];
extern const SCAN_ORDER av1_scan_orders[TX_SIZES_ALL][TX_TYPES];

void av1_deliver_eob_threshold(const AV1_COMMON *cm, MACROBLOCKD *xd);

static INLINE const SCAN_ORDER *get_default_scan(TX_SIZE tx_size,
                                                 TX_TYPE tx_type) {
  return &av1_scan_orders[tx_size][tx_type];
}

static INLINE const SCAN_ORDER *get_scan(TX_SIZE tx_size, TX_TYPE tx_type) {
  return get_default_scan(tx_size, tx_type);
}

#if CONFIG_DSPL_RESIDUAL
/*!
 * Scans in an input array into an output array in the order defined by
 * scan_order. The ith element of output dst[i] = src[scan_order->scan[i]]
 *
 * \param src        Pointer to start of input array
 * \param dst        Pointer to start of output array
 * \param eob        Number of elements to scan into the output, e.g., if eob=N
                     then N elements would be written to dst
 * \param scan_order The scan order to use
 */
void scan_array(const tran_low_t *const src, tran_low_t *const dst,
                const int eob, const SCAN_ORDER *const scan_order);

/*!
 * Scans out an input array into an output array in the order defined by
 * scan_order. The ith input element src[i] affects the output as follows:
 * dst[scan_order->scan[i]] = src[i]
 *
 * \param src        Pointer to start of input array
 * \param dst        Pointer to start of output array
 * \param eob        Number of elements to scan into the output, e.g., if eob=N
                     then N elements would be read from src
 * \param scan_order The scan scan order to use
 */
void iscan_array(const tran_low_t *const src, tran_low_t *const dst,
                 const int eob, const SCAN_ORDER *const scan_order);
#endif  // CONFIG_DSPL_RESIDUAL

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_SCAN_H_
