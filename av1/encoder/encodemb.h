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

#ifndef AOM_AV1_ENCODER_ENCODEMB_H_
#define AOM_AV1_ENCODER_ENCODEMB_H_

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"

#include "av1/common/av1_common_int.h"
#include "av1/common/idct.h"
#include "av1/common/scan.h"
#include "av1/common/txb_common.h"
#include "av1/encoder/av1_quantize.h"
#include "av1/encoder/block.h"
#include "av1/encoder/hybrid_fwd_txfm.h"
#include "av1/encoder/tokenize.h"
#ifdef __cplusplus
extern "C" {
#endif

enum {
  AV1_XFORM_QUANT_FP = 0,
  AV1_XFORM_QUANT_B = 1,
  AV1_XFORM_QUANT_DC = 2,
  AV1_XFORM_QUANT_SKIP_QUANT,
  AV1_XFORM_QUANT_TYPES,
} UENUM1BYTE(AV1_XFORM_QUANT);

// TODO(any): Merge OPT_TYPe and TRELLLIS_OPT_TYPE
// Available optimization types to optimize the quantized coefficients.
enum {
  NONE_OPT = 0,            // No optimization.
  TRELLIS_OPT = 1,         // Trellis optimization. See `av1_optimize_b()`.
  DROPOUT_OPT = 2,         // Dropout optimization. See `av1_dropout_qcoeff()`.
  TRELLIS_DROPOUT_OPT = 3  // Perform dropout after trellis optimization.
} UENUM1BYTE(OPT_TYPE);

enum {
  NO_TRELLIS_OPT,          // No trellis optimization
  FULL_TRELLIS_OPT,        // Trellis optimization in all stages
  FINAL_PASS_TRELLIS_OPT,  // Trellis optimization in only the final encode pass
  NO_ESTIMATE_YRD_TRELLIS_OPT  // Disable trellis in estimate_yrd_for_sb
} UENUM1BYTE(TRELLIS_OPT_TYPE);

struct optimize_ctx {
  ENTROPY_CONTEXT ta[MAX_MB_PLANE][MAX_MIB_SIZE];
  ENTROPY_CONTEXT tl[MAX_MB_PLANE][MAX_MIB_SIZE];
};

struct encode_b_args {
  const struct AV1_COMP *cpi;
  MACROBLOCK *x;
  struct optimize_ctx *ctx;
  ENTROPY_CONTEXT *ta;
  ENTROPY_CONTEXT *tl;
  RUN_TYPE dry_run;
  TRELLIS_OPT_TYPE enable_optimize_b;
};

void av1_encode_sb(const struct AV1_COMP *cpi, MACROBLOCK *x, BLOCK_SIZE bsize,
                   RUN_TYPE dry_run);

void av1_foreach_transformed_block_in_plane(
    const MACROBLOCKD *const xd, BLOCK_SIZE plane_bsize, int plane,
    foreach_transformed_block_visitor visit, void *arg);

void av1_encode_sby_pass1(struct AV1_COMP *cpi, MACROBLOCK *x,
                          BLOCK_SIZE bsize);

static inline void av1_setup_xform(const AV1_COMMON *cm, MACROBLOCK *x,
                                   TX_SIZE tx_size, TX_TYPE tx_type,
                                   TxfmParam *txfm_param) {
  MACROBLOCKD *const xd = &x->e_mbd;
  MB_MODE_INFO *const mbmi = xd->mi[0];

  txfm_param->tx_type = tx_type;
  txfm_param->tx_size = tx_size;
  txfm_param->lossless = xd->lossless[mbmi->segment_id];
#if !defined(NDEBUG)
  txfm_param->tx_set_type = av1_get_ext_tx_set_type(
      tx_size, is_inter_block(mbmi), cm->features.reduced_tx_set_used);
#else
  (void)cm;
#endif

  txfm_param->bd = xd->bd;
  txfm_param->is_hbd = is_cur_buf_hbd(xd);
}

static inline void av1_setup_quant(TX_SIZE tx_size, int use_optimize_b,
                                   int xform_quant_idx, int use_quant_b_adapt,
                                   QUANT_PARAM *qparam) {
  qparam->log_scale = av1_get_tx_scale(tx_size);
  qparam->tx_size = tx_size;

  qparam->use_quant_b_adapt = use_quant_b_adapt;
  qparam->use_optimize_b = use_optimize_b;
  qparam->xform_quant_idx = xform_quant_idx;

  qparam->qmatrix = NULL;
  qparam->iqmatrix = NULL;
}

static inline void av1_setup_qmatrix(const CommonQuantParams *quant_params,
                                     const MACROBLOCKD *xd, int plane,
                                     TX_SIZE tx_size, TX_TYPE tx_type,
                                     QUANT_PARAM *qparam) {
  (void)quant_params;
  if (tx_type < IDTX) {
    const struct macroblockd_plane *const pd = &xd->plane[plane];
    const int seg_id = xd->mi[0]->segment_id;
    qparam->qmatrix = pd->seg_qmatrix[seg_id][tx_size];
    qparam->iqmatrix = pd->seg_iqmatrix[seg_id][tx_size];
  } else {
    qparam->qmatrix = NULL;
    qparam->iqmatrix = NULL;
  }
}

void av1_xform_dc_only(MACROBLOCK *x, int plane, int block,
                       TxfmParam *txfm_param, int64_t per_px_mean);

static inline void av1_xform(MACROBLOCK *x, int plane, int block, int blk_row,
                             int blk_col, BLOCK_SIZE plane_bsize,
                             TxfmParam *txfm_param) {
  const struct macroblock_plane *const p = &x->plane[plane];
  const int block_offset = BLOCK_OFFSET(block);
  tran_low_t *const coeff = p->coeff + block_offset;
  const int diff_stride = block_size_wide[plane_bsize];

  const int src_offset = (blk_row * diff_stride + blk_col);
  const int16_t *src_diff = &p->src_diff[src_offset << MI_SIZE_LOG2];

  av1_fwd_txfm(src_diff, coeff, diff_stride, txfm_param);
}

static inline void av1_quant(MACROBLOCK *x, int plane, int block,
                             TxfmParam *txfm_param, const QUANT_PARAM *qparam) {
  const struct macroblock_plane *const p = &x->plane[plane];
  const SCAN_ORDER *const scan_order =
      get_scan(txfm_param->tx_size, txfm_param->tx_type);
  const int block_offset = BLOCK_OFFSET(block);
  tran_low_t *const coeff = p->coeff + block_offset;
  tran_low_t *const qcoeff = p->qcoeff + block_offset;
  tran_low_t *const dqcoeff = p->dqcoeff + block_offset;
  uint16_t *const eob = &p->eobs[block];

  if (LIKELY(qparam->xform_quant_idx != AV1_XFORM_QUANT_SKIP_QUANT)) {
    const int n_coeffs = av1_get_max_eob(txfm_param->tx_size);
    if (LIKELY(!x->seg_skip_block)) {
#if CONFIG_AV1_HIGHBITDEPTH
      if (LIKELY(!txfm_param->is_hbd)) {
        if (LIKELY(qparam->xform_quant_idx == AV1_XFORM_QUANT_FP)) {
          av1_quantize_fp_facade(coeff, n_coeffs, p, qcoeff, dqcoeff, eob,
                                 scan_order, qparam);
        } else if (qparam->xform_quant_idx == AV1_XFORM_QUANT_B) {
          av1_quantize_b_facade(coeff, n_coeffs, p, qcoeff, dqcoeff, eob,
                                scan_order, qparam);
        } else {
          av1_quantize_dc_facade(coeff, n_coeffs, p, qcoeff, dqcoeff, eob,
                                 scan_order, qparam);
        }
      } else {
        if (qparam->xform_quant_idx == AV1_XFORM_QUANT_FP) {
          av1_highbd_quantize_fp_facade(coeff, n_coeffs, p, qcoeff, dqcoeff,
                                        eob, scan_order, qparam);
        } else if (qparam->xform_quant_idx == AV1_XFORM_QUANT_B) {
          av1_highbd_quantize_b_facade(coeff, n_coeffs, p, qcoeff, dqcoeff, eob,
                                       scan_order, qparam);
        } else {
          av1_highbd_quantize_dc_facade(coeff, n_coeffs, p, qcoeff, dqcoeff,
                                        eob, scan_order, qparam);
        }
      }
#else
      if (LIKELY(qparam->xform_quant_idx == AV1_XFORM_QUANT_FP)) {
        av1_quantize_fp_facade(coeff, n_coeffs, p, qcoeff, dqcoeff, eob,
                               scan_order, qparam);
      } else if (qparam->xform_quant_idx == AV1_XFORM_QUANT_B) {
        av1_quantize_b_facade(coeff, n_coeffs, p, qcoeff, dqcoeff, eob,
                              scan_order, qparam);
      } else {
        av1_quantize_dc_facade(coeff, n_coeffs, p, qcoeff, dqcoeff, eob,
                               scan_order, qparam);
      }
#endif
    } else {
      av1_quantize_skip(n_coeffs, qcoeff, dqcoeff, eob);
    }
  }
  if (qparam->use_optimize_b) {
    p->txb_entropy_ctx[block] = 0;
  } else {
    p->txb_entropy_ctx[block] =
        av1_get_txb_entropy_context(qcoeff, scan_order, *eob);
  }
}

static inline void av1_xform_quant(MACROBLOCK *x, int plane, int block,
                                   int blk_row, int blk_col,
                                   BLOCK_SIZE plane_bsize,
                                   TxfmParam *txfm_param,
                                   const QUANT_PARAM *qparam) {
  av1_xform(x, plane, block, blk_row, blk_col, plane_bsize, txfm_param);
  av1_quant(x, plane, block, txfm_param, qparam);
}

// This function can be used as (i) a further optimization to reduce the
// redundancy of quantized coefficients (a.k.a., `qcoeff`) after trellis
// optimization, or (ii) an alternative to trellis optimization in high-speed
// compression mode (e.g., real-time mode under speed-6) due to its LOW time
// complexity. The rational behind is to drop out the may-be redundant quantized
// coefficient which is among a bunch of zeros. NOTE: This algorithm is not as
// accurate as trellis optimization since the hyper-parameters are hard-coded
// instead of dynamic search. More adaptive logic may improve the performance.
// This function should be applied to all or partical block cells.
// Inputs:
//   mb: Pointer to the MACROBLOCK to perform dropout on.
//   plane: Index of the plane to which the target block belongs.
//   block: Index of the target block.
//   tx_size: Transform size of the target block.
//   tx_type: Transform type of the target block. This field is particularly
//            used to find out the scan order of the block.
//   qindex: Quantization index used for target block. In general, all blocks
//           in a same plane share the same quantization index. This field is
//           particularly used to determine how many zeros should be used to
//           drop out a coefficient.
// Returns:
//   Nothing will be returned, but `qcoeff`, `dqcoeff`, `eob`, as well as
//   `txb_entropy_ctx`, which `mb` points to, may be modified by this function.
void av1_dropout_qcoeff(MACROBLOCK *mb, int plane, int block, TX_SIZE tx_size,
                        TX_TYPE tx_type, int qindex);
// Same as above, with the number of zeroes needed before/after a coeff to drop
// it explicitly passed in, instead of being derived from qindex.
void av1_dropout_qcoeff_num(MACROBLOCK *mb, int plane, int block,
                            TX_SIZE tx_size, TX_TYPE tx_type,
                            int dropout_num_before, int dropout_num_after);

void av1_pad_block_border(const MACROBLOCK *x, int rows, int cols,
                          int16_t *diff, ptrdiff_t diff_stride, int plane,
                          BLOCK_SIZE plane_bsize, int blk_col, int blk_row,
                          TX_TYPE tx_type);

static inline void av1_subtract_block(const MACROBLOCK *x, int rows, int cols,
                                      int16_t *diff, ptrdiff_t diff_stride,
                                      const uint8_t *src8, ptrdiff_t src_stride,
                                      const uint8_t *pred8,
                                      ptrdiff_t pred_stride, int plane,
                                      BLOCK_SIZE plane_bsize, int blk_col,
                                      int blk_row, TX_TYPE tx_type,
                                      bool do_border_pad) {
  assert(rows >= 4 && cols >= 4);
#if CONFIG_AV1_HIGHBITDEPTH
  if (UNLIKELY(is_cur_buf_hbd(&x->e_mbd))) {
    aom_highbd_subtract_block(rows, cols, diff, diff_stride, src8, src_stride,
                              pred8, pred_stride);
  } else {
    aom_subtract_block(rows, cols, diff, diff_stride, src8, src_stride, pred8,
                       pred_stride);
  }
#else
  aom_subtract_block(rows, cols, diff, diff_stride, src8, src_stride, pred8,
                     pred_stride);
#endif
  if (UNLIKELY(do_border_pad)) {
    av1_pad_block_border(x, rows, cols, diff, diff_stride, plane, plane_bsize,
                         blk_col, blk_row, tx_type);
  }
}

static inline void av1_subtract_txb(MACROBLOCK *x, int plane,
                                    BLOCK_SIZE plane_bsize, int blk_col,
                                    int blk_row, TX_SIZE tx_size,
                                    TX_TYPE tx_type, bool do_border_pad) {
  struct macroblock_plane *const p = &x->plane[plane];
  const struct macroblockd_plane *const pd = &x->e_mbd.plane[plane];
  const int diff_stride = block_size_wide[plane_bsize];
  const int src_stride = p->src.stride;
  const int dst_stride = pd->dst.stride;
  const int tx1d_width = tx_size_wide[tx_size];
  const int tx1d_height = tx_size_high[tx_size];
  uint8_t *dst = &pd->dst.buf[(blk_row * dst_stride + blk_col) << MI_SIZE_LOG2];
  uint8_t *src = &p->src.buf[(blk_row * src_stride + blk_col) << MI_SIZE_LOG2];
  int16_t *src_diff =
      &p->src_diff[(blk_row * diff_stride + blk_col) << MI_SIZE_LOG2];
  av1_subtract_block(x, tx1d_height, tx1d_width, src_diff, diff_stride, src,
                     src_stride, dst, dst_stride, plane, plane_bsize, blk_col,
                     blk_row, tx_type, do_border_pad);
}

static inline void av1_subtract_plane(MACROBLOCK *x, BLOCK_SIZE plane_bsize,
                                      int plane, bool do_border_pad) {
  struct macroblock_plane *const p = &x->plane[plane];
  const struct macroblockd_plane *const pd = &x->e_mbd.plane[plane];
  assert(plane_bsize < BLOCK_SIZES_ALL);
  const int bw = block_size_wide[plane_bsize];
  const int bh = block_size_high[plane_bsize];

  av1_subtract_block(x, bh, bw, p->src_diff, bw, p->src.buf, p->src.stride,
                     pd->dst.buf, pd->dst.stride, plane, plane_bsize, 0, 0,
                     DCT_DCT, do_border_pad);
}

static inline void av1_set_txb_context(MACROBLOCK *x, int plane, int block,
                                       TX_SIZE tx_size, ENTROPY_CONTEXT *a,
                                       ENTROPY_CONTEXT *l) {
  const uint8_t ctx = x->plane[plane].txb_entropy_ctx[block];
  memset(a, ctx, tx_size_wide_unit[tx_size] * sizeof(*a));
  memset(l, ctx, tx_size_high_unit[tx_size] * sizeof(*l));
}

void av1_encode_intra_block_plane(const struct AV1_COMP *cpi, MACROBLOCK *x,
                                  BLOCK_SIZE bsize, int plane, RUN_TYPE dry_run,
                                  TRELLIS_OPT_TYPE enable_optimize_b);

static inline int is_trellis_used(TRELLIS_OPT_TYPE optimize_b,
                                  RUN_TYPE dry_run) {
  if (optimize_b == NO_TRELLIS_OPT) return false;
  if (optimize_b == FINAL_PASS_TRELLIS_OPT && dry_run != OUTPUT_ENABLED)
    return false;
  return true;
}

// Scaling terms (precision of 12 bits) to perform tx-size specific
// normalization that is used in DCT_DCT forward transform.
// For transform blocks of 1:2 and 2:1       - sqrt(2) normalization is used
// For transform blocks of 1:4 and 4:1       - factor of 2 is used
// For transform blocks TX_8x8 and below     - an additional factor of 2 is used
// For transform blocks max(width,height)=64 - currently not supported

static const uint16_t dc_coeff_scale[TX_SIZES_ALL] = {
  1024, 2048, 4096, 4096, 0,    1448, 1448, 2896, 2896, 2896,
  2896, 0,    0,    2048, 2048, 4096, 4096, 0,    0
};

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_ENCODER_ENCODEMB_H_
