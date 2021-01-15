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

#ifndef AOM_AV1_COMMON_RECONINTER_H_
#define AOM_AV1_COMMON_RECONINTER_H_

#include "av1/common/filter.h"
#include "av1/common/onyxc_int.h"
#include "av1/common/convolve.h"
#include "av1/common/warped_motion.h"
#include "aom/aom_integer.h"

// Work out how many pixels off the edge of a reference frame we're allowed
// to go when forming an inter prediction.
// The outermost row/col of each referernce frame is extended by
// (AOM_BORDER_IN_PIXELS >> subsampling) pixels, but we need to keep
// at least AOM_INTERP_EXTEND pixels within that to account for filtering.
//
// We have to break this up into two macros to keep both clang-format and
// tools/lint-hunks.py happy.
#define AOM_LEFT_TOP_MARGIN_PX(subsampling) \
  ((AOM_BORDER_IN_PIXELS >> subsampling) - AOM_INTERP_EXTEND)
#define AOM_LEFT_TOP_MARGIN_SCALED(subsampling) \
  (AOM_LEFT_TOP_MARGIN_PX(subsampling) << SCALE_SUBPEL_BITS)

#ifdef __cplusplus
extern "C" {
#endif

// Set to (1 << 5) if the 32-ary codebooks are used for any bock size
#define MAX_WEDGE_TYPES (1 << 4)

#define MAX_WEDGE_SIZE_LOG2 5  // 32x32
#define MAX_WEDGE_SIZE (1 << MAX_WEDGE_SIZE_LOG2)
#define MAX_WEDGE_SQUARE (MAX_WEDGE_SIZE * MAX_WEDGE_SIZE)

#define WEDGE_WEIGHT_BITS 6

#define WEDGE_NONE -1

// Angles are with respect to horizontal anti-clockwise
enum {
  WEDGE_HORIZONTAL = 0,
  WEDGE_VERTICAL = 1,
  WEDGE_OBLIQUE27 = 2,
  WEDGE_OBLIQUE63 = 3,
  WEDGE_OBLIQUE117 = 4,
  WEDGE_OBLIQUE153 = 5,
  WEDGE_DIRECTIONS
} UENUM1BYTE(WedgeDirectionType);

// 3-tuple: {direction, x_offset, y_offset}
typedef struct {
  WedgeDirectionType direction;
  int x_offset;
  int y_offset;
} wedge_code_type;

typedef uint8_t *wedge_masks_type[MAX_WEDGE_TYPES];

typedef struct {
  int bits;
  const wedge_code_type *codebook;
  uint8_t *signflip;
  wedge_masks_type *masks;
} wedge_params_type;

extern const wedge_params_type av1_wedge_params_lookup[BLOCK_SIZES_ALL];

typedef struct SubpelParams {
  int xs;
  int ys;
  int subpel_x;
  int subpel_y;
} SubpelParams;

struct build_prediction_ctxt {
  const AV1_COMMON *cm;
  uint8_t **tmp_buf;
  int *tmp_width;
  int *tmp_height;
  int *tmp_stride;
  int mb_to_far_edge;
};

static INLINE int has_scale(int xs, int ys) {
  return xs != SCALE_SUBPEL_SHIFTS || ys != SCALE_SUBPEL_SHIFTS;
}

static INLINE void revert_scale_extra_bits(SubpelParams *sp) {
  sp->subpel_x >>= SCALE_EXTRA_BITS;
  sp->subpel_y >>= SCALE_EXTRA_BITS;
  sp->xs >>= SCALE_EXTRA_BITS;
  sp->ys >>= SCALE_EXTRA_BITS;
  assert(sp->subpel_x < SUBPEL_SHIFTS);
  assert(sp->subpel_y < SUBPEL_SHIFTS);
  assert(sp->xs <= SUBPEL_SHIFTS);
  assert(sp->ys <= SUBPEL_SHIFTS);
}

// If a border is built, it will always be on top-left and of size 16.
// 16 is a limitation due to SIMD requirements of 16-byte alignment.
// TODO(elliottk): if the border proves useful, re-work the code so it
// support building smaller borders. Note that the initial pointer to
// a inter/intra-predictor must be on a 16-byte boundary, and the stride
// must be a multiple of 16.
#define INTERINTRA_PRED_BORDER 16
// 32x32 is the maximum block size allowed for inter-intra modes.
#define MAX_INTERINTRA_BORDER_SB_SQUARE \
  ((INTERINTRA_PRED_BORDER + 32) * (INTERINTRA_PRED_BORDER + 32))

static INLINE void inter_predictor(const uint8_t *src, int src_stride,
                                   uint8_t *dst, int dst_stride,
                                   const SubpelParams *subpel_params, int w,
                                   int h, int orig_w, int orig_h,
                                   ConvolveParams *conv_params,
                                   int_interpfilters interp_filters,
                                   int is_intrabc) {
  assert(conv_params->do_average == 0 || conv_params->do_average == 1);
  const int is_scaled = has_scale(subpel_params->xs, subpel_params->ys);
  assert(IMPLIES(is_intrabc, !is_scaled));
  if (is_scaled) {
    av1_convolve_2d_facade(src, src_stride, dst, dst_stride, w, h, orig_w,
                           orig_h, interp_filters, subpel_params->subpel_x,
                           subpel_params->xs, subpel_params->subpel_y,
                           subpel_params->ys, 1, conv_params, is_intrabc);
  } else {
    SubpelParams sp = *subpel_params;
    revert_scale_extra_bits(&sp);
    av1_convolve_2d_facade(src, src_stride, dst, dst_stride, w, h, orig_w,
                           orig_h, interp_filters, sp.subpel_x, sp.xs,
                           sp.subpel_y, sp.ys, 0, conv_params, is_intrabc);
  }
}

static INLINE void highbd_inter_predictor(const uint8_t *src, int src_stride,
                                          uint8_t *dst, int dst_stride,
                                          const SubpelParams *subpel_params,
                                          int w, int h, int orig_w, int orig_h,
                                          ConvolveParams *conv_params,
                                          int_interpfilters interp_filters,
                                          int is_intrabc, int bd) {
  assert(conv_params->do_average == 0 || conv_params->do_average == 1);
  const int is_scaled = has_scale(subpel_params->xs, subpel_params->ys);
  assert(IMPLIES(is_intrabc, !is_scaled));
  if (is_scaled) {
    av1_highbd_convolve_2d_facade(
        src, src_stride, dst, dst_stride, w, h, orig_w, orig_h, interp_filters,
        subpel_params->subpel_x, subpel_params->xs, subpel_params->subpel_y,
        subpel_params->ys, 1, conv_params, is_intrabc, bd);
  } else {
    SubpelParams sp = *subpel_params;
    revert_scale_extra_bits(&sp);
    av1_highbd_convolve_2d_facade(
        src, src_stride, dst, dst_stride, w, h, orig_w, orig_h, interp_filters,
        sp.subpel_x, sp.xs, sp.subpel_y, sp.ys, 0, conv_params, is_intrabc, bd);
  }
}

void av1_modify_neighbor_predictor_for_obmc(MB_MODE_INFO *mbmi);
int av1_skip_u4x4_pred_in_obmc(int mi_row, int mi_col, BLOCK_SIZE bsize,
                               const struct macroblockd_plane *pd, int dir);

static INLINE int is_interinter_compound_used(COMPOUND_TYPE type,
                                              BLOCK_SIZE sb_type) {
  const int comp_allowed = is_comp_ref_allowed(sb_type);
  switch (type) {
    case COMPOUND_AVERAGE:
    case COMPOUND_DISTWTD:
    case COMPOUND_DIFFWTD: return comp_allowed;
    case COMPOUND_WEDGE:
      return comp_allowed && av1_wedge_params_lookup[sb_type].bits > 0;
    default: assert(0); return 0;
  }
}

static INLINE int is_any_masked_compound_used(BLOCK_SIZE sb_type) {
  COMPOUND_TYPE comp_type;
  int i;
  if (!is_comp_ref_allowed(sb_type)) return 0;
  for (i = 0; i < COMPOUND_TYPES; i++) {
    comp_type = (COMPOUND_TYPE)i;
    if (is_masked_compound_type(comp_type) &&
        is_interinter_compound_used(comp_type, sb_type))
      return 1;
  }
  return 0;
}

static INLINE int get_wedge_bits_lookup(BLOCK_SIZE sb_type) {
  return av1_wedge_params_lookup[sb_type].bits;
}

static INLINE int get_interinter_wedge_bits(BLOCK_SIZE sb_type) {
  const int wbits = av1_wedge_params_lookup[sb_type].bits;
  return (wbits > 0) ? wbits + 1 : 0;
}

static INLINE int is_interintra_wedge_used(BLOCK_SIZE sb_type) {
  return av1_wedge_params_lookup[sb_type].bits > 0;
}

// Makes the inter-predictor. If border is non-zero, builds a top-left
// border based on the inter-predictor;  it is up to the caller to ensure that
// dst / conv_params->dst is large enough to support the region. Note that
// dst / conv_params->dst / src should point to the start of the
// inter-prediction or source, not the border (which is negatively offset).
void av1_make_inter_predictor(
    const uint8_t *src, int src_stride, uint8_t *dst, int dst_stride,
    const SubpelParams *subpel_params, const struct scale_factors *sf, int w,
    int h, ConvolveParams *conv_params, int_interpfilters interp_filters,
    const WarpTypesAllowed *warp_types, int p_col, int p_row, int plane,
    int ref, const MB_MODE_INFO *mi, int build_for_obmc, const MACROBLOCKD *xd,
    int can_use_previous, const int border);

typedef void (*CalcSubpelParamsFunc)(
    MACROBLOCKD *xd, const struct scale_factors *const sf, const MV *const mv,
    int plane, int pre_x, int pre_y, int x, int y, struct buf_2d *const pre_buf,
    int bw, int bh, const WarpTypesAllowed *const warp_types, int ref,
#if CONFIG_OPTFLOW_REFINEMENT
    int use_optflow_ref,
#endif
    const void *const args, uint8_t **pre, SubpelParams *subpel_params,
    int *src_stride);

// Calculate the size of the border region (top-left) that should be used.
// If 0 is returned, no border should be constructed.
int av1_calc_border(const MACROBLOCKD *xd, int plane, int build_for_obmc);

// Note that in the case of a border, dst should already be offset, to allow
// negative offsetting.
void av1_build_inter_predictors(const AV1_COMMON *cm, MACROBLOCKD *xd,
                                int plane, const MB_MODE_INFO *mi,
                                int build_for_obmc, int bw, int bh, int mi_x,
                                int mi_y,
                                CalcSubpelParamsFunc calc_subpel_params_func,
                                const void *const calc_subpel_params_func_args,
                                uint8_t *dst, int dst_stride, int border);

#if CONFIG_OPTFLOW_REFINEMENT
void av1_opfl_mv_refinement_lowbd(const uint8_t *p0, int pstride0,
                                  const uint8_t *p1, int pstride1,
                                  const int16_t *gx0, const int16_t *gy0,
                                  const int16_t *gx1, const int16_t *gy1,
                                  int gstride, int bw, int bh, int d0, int d1,
                                  int grad_prec_bits, int mv_prec_bits,
                                  int *vx0, int *vy0, int *vx1, int *vy1);
void av1_opfl_mv_refinement_highbd(const uint16_t *p0, int pstride0,
                                   const uint16_t *p1, int pstride1,
                                   const int16_t *gx0, const int16_t *gy0,
                                   const int16_t *gx1, const int16_t *gy1,
                                   int gstride, int bw, int bh, int d0, int d1,
                                   int grad_prec_bits, int mv_prec_bits,
                                   int *vx0, int *vy0, int *vx1, int *vy1);

void av1_opfl_mv_refinement4_lowbd(const uint8_t *p0, int pstride0,
                                   const uint8_t *p1, int pstride1,
                                   const int16_t *gx0, const int16_t *gy0,
                                   const int16_t *gx1, const int16_t *gy1,
                                   int gstride, int bw, int bh, int d0, int d1,
                                   int grad_prec_bits, int mv_prec_bits,
                                   int *vx0, int *vy0, int *vx1, int *vy1);
void av1_opfl_mv_refinement4_highbd(const uint16_t *p0, int pstride0,
                                    const uint16_t *p1, int pstride1,
                                    const int16_t *gx0, const int16_t *gy0,
                                    const int16_t *gx1, const int16_t *gy1,
                                    int gstride, int bw, int bh, int d0, int d1,
                                    int grad_prec_bits, int mv_prec_bits,
                                    int *vx0, int *vy0, int *vx1, int *vy1);
#endif  // CONFIG_OPTFLOW_REFINEMENT

// TODO(jkoleszar): yet another mv clamping function :-(
static INLINE MV clamp_mv_to_umv_border_sb(const MACROBLOCKD *xd,
                                           const MV *src_mv, int bw, int bh,
#if CONFIG_OPTFLOW_REFINEMENT
                                           int use_optflow_refinement,
#endif  // CONFIG_OPTFLOW_REFINEMENT
                                           int ss_x, int ss_y) {
  // If the MV points so far into the UMV border that no visible pixels
  // are used for reconstruction, the subpel part of the MV can be
  // discarded and the MV limited to 16 pixels with equivalent results.
  const int spel_left = (AOM_INTERP_EXTEND + bw) << SUBPEL_BITS;
  const int spel_right = spel_left - SUBPEL_SHIFTS;
  const int spel_top = (AOM_INTERP_EXTEND + bh) << SUBPEL_BITS;
  const int spel_bottom = spel_top - SUBPEL_SHIFTS;
#if CONFIG_OPTFLOW_REFINEMENT
  MV clamped_mv;
  if (use_optflow_refinement) {
    // optflow refinement always returns MVs with 1/16 precision so it is not
    // necessary to shift the MV before clamping
    clamped_mv.row = (int16_t)(src_mv->row >> ss_y);
    clamped_mv.col = (int16_t)(src_mv->col >> ss_x);

  } else {
    clamped_mv.row = (int16_t)(src_mv->row * (1 << (1 - ss_y)));
    clamped_mv.col = (int16_t)(src_mv->col * (1 << (1 - ss_x)));
  }
#else
  MV clamped_mv = { (int16_t)(src_mv->row * (1 << (1 - ss_y))),
                    (int16_t)(src_mv->col * (1 << (1 - ss_x))) };
#endif  // CONFIG_OPTFLOW_REFINEMENT
  assert(ss_x <= 1);
  assert(ss_y <= 1);
  clamp_mv(&clamped_mv, xd->mb_to_left_edge * (1 << (1 - ss_x)) - spel_left,
           xd->mb_to_right_edge * (1 << (1 - ss_x)) + spel_right,
           xd->mb_to_top_edge * (1 << (1 - ss_y)) - spel_top,
           xd->mb_to_bottom_edge * (1 << (1 - ss_y)) + spel_bottom);

  return clamped_mv;
}

static INLINE int64_t scaled_buffer_offset(int x_offset, int y_offset,
                                           int stride,
                                           const struct scale_factors *sf) {
  const int x =
      sf ? sf->scale_value_x(x_offset, sf) >> SCALE_EXTRA_BITS : x_offset;
  const int y =
      sf ? sf->scale_value_y(y_offset, sf) >> SCALE_EXTRA_BITS : y_offset;
  return (int64_t)y * stride + x;
}

static INLINE void setup_pred_plane(struct buf_2d *dst, uint8_t *src, int width,
                                    int height, int stride, int mi_row,
                                    int mi_col,
                                    const struct scale_factors *scale,
                                    int subsampling_x, int subsampling_y,
                                    int is_uv,
                                    const CHROMA_REF_INFO *chr_ref_info) {
  int mi_row_chr_base, mi_col_chr_base;
  if (chr_ref_info) {
    mi_row_chr_base = chr_ref_info->mi_row_chroma_base;
    mi_col_chr_base = chr_ref_info->mi_col_chroma_base;
  } else {
    mi_row_chr_base = mi_row;
    mi_col_chr_base = mi_col;
  }

  int mi_row_offset, mi_col_offset;
  if (is_uv) {
    mi_row_offset = mi_row - mi_row_chr_base;
    mi_col_offset = mi_col - mi_col_chr_base;
  } else {
    mi_row_offset = 0;
    mi_col_offset = 0;
  }

  mi_row -= mi_row_offset;
  mi_col -= mi_col_offset;

  const int x = (MI_SIZE * mi_col) >> subsampling_x;
  const int y = (MI_SIZE * mi_row) >> subsampling_y;
  dst->buf = src + scaled_buffer_offset(x, y, stride, scale);
  dst->buf0 = src;
  dst->width = width;
  dst->height = height;
  dst->stride = stride;
}

void av1_setup_dst_planes(struct macroblockd_plane *planes,
                          const YV12_BUFFER_CONFIG *src, int mi_row, int mi_col,
                          const int plane_start, const int plane_end,
                          const CHROMA_REF_INFO *chr_ref_info);

void av1_setup_pre_planes(MACROBLOCKD *xd, int idx,
                          const YV12_BUFFER_CONFIG *src, int mi_row, int mi_col,
                          const struct scale_factors *sf, const int num_planes,
                          const CHROMA_REF_INFO *chr_ref_info);

static INLINE void set_default_interp_filters(
    MB_MODE_INFO *const mbmi, InterpFilter frame_interp_filter) {
  mbmi->interp_filters =
      av1_broadcast_interp_filter(av1_unswitchable_filter(frame_interp_filter));
}

static INLINE int av1_is_interp_needed(const MACROBLOCKD *const xd) {
  const MB_MODE_INFO *const mbmi = xd->mi[0];
  if (mbmi->skip_mode) return 0;
  if (mbmi->motion_mode == WARPED_CAUSAL) return 0;
  if (is_nontrans_global_motion(xd, xd->mi[0])) return 0;
  return 1;
}

void av1_setup_build_prediction_by_above_pred(
    MACROBLOCKD *xd, int rel_mi_col, uint8_t above_mi_width,
    MB_MODE_INFO *above_mbmi, struct build_prediction_ctxt *ctxt,
    const int num_planes);
void av1_setup_build_prediction_by_left_pred(MACROBLOCKD *xd, int rel_mi_row,
                                             uint8_t left_mi_height,
                                             MB_MODE_INFO *left_mbmi,
                                             struct build_prediction_ctxt *ctxt,
                                             const int num_planes);
void av1_build_obmc_inter_prediction(const AV1_COMMON *cm, MACROBLOCKD *xd,
                                     uint8_t *above[MAX_MB_PLANE],
                                     int above_stride[MAX_MB_PLANE],
                                     uint8_t *left[MAX_MB_PLANE],
                                     int left_stride[MAX_MB_PLANE]);

const uint8_t *av1_get_obmc_mask(int length);
void av1_count_overlappable_neighbors(const AV1_COMMON *cm, MACROBLOCKD *xd);

#define MASK_MASTER_SIZE ((MAX_WEDGE_SIZE) << 1)
#define MASK_MASTER_STRIDE (MASK_MASTER_SIZE)

void av1_init_wedge_masks();

static INLINE const uint8_t *av1_get_contiguous_soft_mask(int8_t wedge_index,
                                                          int8_t wedge_sign,
                                                          BLOCK_SIZE sb_type) {
  return av1_wedge_params_lookup[sb_type].masks[wedge_sign][wedge_index];
}

const uint8_t *av1_get_compound_type_mask(
    const INTERINTER_COMPOUND_DATA *const comp_data, BLOCK_SIZE sb_type);

// Allocate a buffer large enough to build a top-left border region. Guarantees
// that the buffer is allocated on a 16-byte boundary and that the stride
// is a multiple of 16. Note that the returned pointer is already offset
// into the allocated buffer, so negative offsetting is acceptable. Use
// av1_free_extended_buffer to unallocate.
void av1_alloc_buf_with_border(uint8_t **buf, int *buf_stride, int border,
                               bool is_hbd);

void av1_free_buf_with_border(uint8_t *buf, int buf_stride, int border,
                              bool is_hbd);

// build interintra_predictors for one plane
void av1_build_interintra_predictors_sbp(const AV1_COMMON *cm, MACROBLOCKD *xd,
                                         uint8_t *pred, int stride,
                                         const BUFFER_SET *ctx, int plane,
                                         BLOCK_SIZE bsize, int border);

// If border is specified, then the destination pointer is treated as having
// an extended region to the top and left, which can be accessed via
// a negative offset to the destination.
void av1_build_intra_predictors_for_interintra(
    const AV1_COMMON *cm, MACROBLOCKD *xd, BLOCK_SIZE bsize, int plane,
    const BUFFER_SET *ctx, uint8_t *dst, int dst_stride, int border);

// If the inter-intra mode is one that requires an extended region, then
// inter_pred and intra_pred should point to the start of the inter/intra
// predictor, *not* the border. The beginning of the border can be accessed
// by offsetting with (-border * stride - border). Note that the code assumes
// the border is along the top-left. If there is no extended region, border
// should be 0.
void av1_combine_interintra(MACROBLOCKD *xd, BLOCK_SIZE bsize, int plane,
                            const uint8_t *inter_pred, int inter_stride,
                            const uint8_t *intra_pred, int intra_stride,
                            int border);

void av1_dist_wtd_comp_weight_assign(const AV1_COMMON *cm,
                                     const MB_MODE_INFO *mbmi, int order_idx,
                                     int *fwd_offset, int *bck_offset,
                                     int *use_dist_wtd_comp_avg,
                                     int is_compound);
int av1_allow_warp(const MB_MODE_INFO *const mbmi,
                   const WarpTypesAllowed *const warp_types,
                   const WarpedMotionParams *const gm_params,
                   int build_for_obmc, const struct scale_factors *const sf,
                   WarpedMotionParams *final_warp_params);

#if CONFIG_DERIVED_MV
int av1_derived_mv_allowed(MACROBLOCKD *const xd, MB_MODE_INFO *const mbmi);

MV av1_derive_mv(const AV1_COMMON *const cm, MACROBLOCKD *xd, int ref,
                 MB_MODE_INFO *mbmi, uint8_t *recon_buf, int recon_stride);
#endif  // CONFIG_DERIVED_MV

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_RECONINTER_H_
