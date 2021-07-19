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

#include "av1/common/av1_common_int.h"
#include "av1/common/convolve.h"
#include "av1/common/filter.h"
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

#define MAX_WEDGE_TYPES 16

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
  int wedge_types;
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
  void *dcb;  // Decoder-only coding block.
};

typedef enum InterPredMode {
  TRANSLATION_PRED,
  WARP_PRED,
} InterPredMode;

typedef enum InterCompMode {
  UNIFORM_SINGLE,
  UNIFORM_COMP,
  MASK_COMP,
} InterCompMode;

typedef struct InterPredParams {
  InterPredMode mode;
  InterCompMode comp_mode;
  WarpedMotionParams warp_params;
  ConvolveParams conv_params;
  const InterpFilterParams *interp_filter_params[2];
  int block_width;
  int block_height;
#if CONFIG_OPTFLOW_REFINEMENT
  // In optical flow refinement, block_width and block_height will pass the
  // subblock size into av1_make_inter_predictor, while orig_w and orig_h
  // keep the original block size that is needed by calc_subpel_params_func
  int orig_width;
  int orig_height;
#endif  // CONFIG_OPTFLOW_REFINEMENT
  int pix_row;
  int pix_col;
  struct buf_2d ref_frame_buf;
  int subsampling_x;
  int subsampling_y;
  const struct scale_factors *scale_factors;
  int bit_depth;
  int use_hbd_buf;
  INTERINTER_COMPOUND_DATA mask_comp;
  BLOCK_SIZE sb_type;
  int is_intrabc;
} InterPredParams;

#if CONFIG_OPTFLOW_REFINEMENT

// Apply bilinear and bicubic interpolation for subpel gradient to avoid
// calls of build_one_inter_predictor function. Bicubic interpolation
// brings better quality but the speed results are neutral. As such, bilinear
// interpolation is used by default for a better trade-off between quality
// and complexity.
#define OPFL_BILINEAR_GRAD 0
#define OPFL_BICUBIC_GRAD 0

// Use downsampled gradient arrays to compute MV offsets
#define OPFL_DOWNSAMP_QUINCUNX 0

// Delta to use for computing gradients in bits, with 0 referring to
// integer-pel. The actual delta value used from the 1/8-pel original MVs
// is 2^(3 - SUBPEL_GRAD_DELTA_BITS). The max value of this macro is 3.
#define SUBPEL_GRAD_DELTA_BITS 3

// Bilinear and bicubic coefficients. Note that, at boundary, we apply
// coefficients that are doubled because spatial distance between the two
// interpolated pixels is halved. In other words, instead of computing
//   coeff * (v[delta] - v[-delta]) / (2 * delta),
// we are practically computing
//   coeff * (v[delta] - v[0]) / (2 * delta).
// Thus, coeff is doubled to get a better gradient quality.
#if OPFL_BILINEAR_GRAD
static const int bilinear_bits = 3;
static const int32_t coeffs_bilinear[4][2] = {
  { 8, 16 },  // delta = 1 (SUBPEL_GRAD_DELTA_BITS = 0)
  { 4, 8 },   // delta = 0.5 (SUBPEL_GRAD_DELTA_BITS = 1)
  { 2, 4 },   // delta = 0.25 (SUBPEL_GRAD_DELTA_BITS = 2)
  { 1, 2 },   // delta = 0.125 (SUBPEL_GRAD_DELTA_BITS = 3)
};
#endif

#if OPFL_BICUBIC_GRAD
static const int bicubic_bits = 7;
static const int32_t coeffs_bicubic[4][2][2] = {
  { { 128, 256 }, { 0, 0 } },    // delta = 1 (SUBPEL_GRAD_DELTA_BITS = 0)
  { { 80, 160 }, { -8, -16 } },  // delta = 0.5 (SUBPEL_GRAD_DELTA_BITS = 1)
  { { 42, 84 }, { -5, -10 } },   // delta = 0.25 (SUBPEL_GRAD_DELTA_BITS = 2)
  { { 21, 42 }, { -3, -6 } },    // delta = 0.125 (SUBPEL_GRAD_DELTA_BITS = 3)
};
#endif
#endif  // CONFIG_OPTFLOW_REFINEMENT

void av1_init_inter_params(InterPredParams *inter_pred_params, int block_width,
                           int block_height, int pix_row, int pix_col,
                           int subsampling_x, int subsampling_y, int bit_depth,
                           int use_hbd_buf, int is_intrabc,
                           const struct scale_factors *sf,
                           const struct buf_2d *ref_buf,
#if CONFIG_REMOVE_DUAL_FILTER
                           InterpFilter interp_filter
#else
                           int_interpfilters interp_filters
#endif  // CONFIG_REMOVE_DUAL_FILTER
);

void av1_init_comp_mode(InterPredParams *inter_pred_params);

void av1_init_warp_params(InterPredParams *inter_pred_params,
                          const WarpTypesAllowed *warp_types, int ref,
                          const MACROBLOCKD *xd, MB_MODE_INFO *mi);

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

static INLINE void inter_predictor(
    const uint8_t *src, int src_stride, uint8_t *dst, int dst_stride,
    const SubpelParams *subpel_params, int w, int h,
    ConvolveParams *conv_params, const InterpFilterParams *interp_filters[2]) {
  assert(conv_params->do_average == 0 || conv_params->do_average == 1);
  const int is_scaled = has_scale(subpel_params->xs, subpel_params->ys);
  if (is_scaled) {
    av1_convolve_2d_facade(src, src_stride, dst, dst_stride, w, h,
                           interp_filters, subpel_params->subpel_x,
                           subpel_params->xs, subpel_params->subpel_y,
                           subpel_params->ys, 1, conv_params);
  } else {
    SubpelParams sp = *subpel_params;
    revert_scale_extra_bits(&sp);
    av1_convolve_2d_facade(src, src_stride, dst, dst_stride, w, h,
                           interp_filters, sp.subpel_x, sp.xs, sp.subpel_y,
                           sp.ys, 0, conv_params);
  }
}

static INLINE void highbd_inter_predictor(
    const uint8_t *src, int src_stride, uint8_t *dst, int dst_stride,
    const SubpelParams *subpel_params, int w, int h,
    ConvolveParams *conv_params, const InterpFilterParams *interp_filters[2],
    int bd) {
  assert(conv_params->do_average == 0 || conv_params->do_average == 1);
  const int is_scaled = has_scale(subpel_params->xs, subpel_params->ys);
  if (is_scaled) {
    av1_highbd_convolve_2d_facade(src, src_stride, dst, dst_stride, w, h,
                                  interp_filters, subpel_params->subpel_x,
                                  subpel_params->xs, subpel_params->subpel_y,
                                  subpel_params->ys, 1, conv_params, bd);
  } else {
    SubpelParams sp = *subpel_params;
    revert_scale_extra_bits(&sp);
    av1_highbd_convolve_2d_facade(src, src_stride, dst, dst_stride, w, h,
                                  interp_filters, sp.subpel_x, sp.xs,
                                  sp.subpel_y, sp.ys, 0, conv_params, bd);
  }
}

void av1_modify_neighbor_predictor_for_obmc(MB_MODE_INFO *mbmi);
int av1_skip_u4x4_pred_in_obmc(BLOCK_SIZE bsize,
                               const struct macroblockd_plane *pd, int dir);

static INLINE int is_interinter_compound_used(COMPOUND_TYPE type,
                                              BLOCK_SIZE sb_type) {
  const int comp_allowed = is_comp_ref_allowed(sb_type);
  switch (type) {
    case COMPOUND_AVERAGE:
#if !CONFIG_REMOVE_DIST_WTD_COMP
    case COMPOUND_DISTWTD:
#endif  // !CONFIG_REMOVE_DIST_WTD_COMP
    case COMPOUND_DIFFWTD: return comp_allowed;
    case COMPOUND_WEDGE:
      return comp_allowed && av1_wedge_params_lookup[sb_type].wedge_types > 0;
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

static INLINE int get_wedge_types_lookup(BLOCK_SIZE sb_type) {
  return av1_wedge_params_lookup[sb_type].wedge_types;
}

static INLINE int av1_is_wedge_used(BLOCK_SIZE sb_type) {
  return av1_wedge_params_lookup[sb_type].wedge_types > 0;
}

void av1_make_inter_predictor(const uint8_t *src, int src_stride, uint8_t *dst,
                              int dst_stride,
                              InterPredParams *inter_pred_params,
                              const SubpelParams *subpel_params);

typedef void (*CalcSubpelParamsFunc)(const MV *const src_mv,
                                     InterPredParams *const inter_pred_params,
                                     MACROBLOCKD *xd, int mi_x, int mi_y,
                                     int ref,
#if CONFIG_OPTFLOW_REFINEMENT
                                     int use_optflow_refinement,
#endif  // CONFIG_OPTFLOW_REFINEMENT
                                     uint8_t **mc_buf, uint8_t **pre,
                                     SubpelParams *subpel_params,
                                     int *src_stride);

void av1_build_one_inter_predictor(
    uint8_t *dst, int dst_stride, const MV *const src_mv,
    InterPredParams *inter_pred_params, MACROBLOCKD *xd, int mi_x, int mi_y,
    int ref, uint8_t **mc_buf, CalcSubpelParamsFunc calc_subpel_params_func);

void av1_build_inter_predictors(const AV1_COMMON *cm, MACROBLOCKD *xd,
                                int plane, MB_MODE_INFO *mi, int build_for_obmc,
                                int bw, int bh, int mi_x, int mi_y,
                                uint8_t **mc_buf,
                                CalcSubpelParamsFunc calc_subpel_params_func);

#if CONFIG_OPTFLOW_REFINEMENT
// This parameter k=OPFL_DIST_RATIO_THR is used to prune MV refinement for the
// case where d0 and d1 are very different. Assuming a = max(|d0|, |d1|) and
// b = min(|d0|, |d1|), MV refinement will only be allowed only if a/b <= k.
// If k is set to 0, refinement will always be enabled.
// If k is set to 1, refinement will only be enabled when |d0|=|d1|.
#define OPFL_DIST_RATIO_THR 0
// Always assume d0 = d1 in optical flow refinement.
#define OPFL_EQUAL_DIST_ASSUMED 0

// Apply regularized least squares (RLS). The RLS parameter is bw * bh * 2^(b-4)
// where b = OPFL_RLS_PARAM_BITS.
#define OPFL_REGULARIZED_LS 1
#define OPFL_RLS_PARAM_BITS 4

// Number of bits allowed for covariance matrix elements (su2, sv2, suv, suw
// and svw) so that D, Px, and Py does not cause overflow issue in int64_t.
// Its value must be <= (64 - mv_prec_bits - grad_prec_bits) / 2.
#define OPFL_COV_CLAMP_BITS 30
#define OPFL_COV_CLAMP_VAL (1 << OPFL_COV_CLAMP_BITS)

// Precision of refined MV returned, 0 being integer pel. For now, only 1/8 or
// 1/16-pel can be used.
#define MV_REFINE_PREC_BITS 4  // (1/16-pel)
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
static INLINE int is_opfl_refine_allowed(const AV1_COMMON *cm,
                                         const MB_MODE_INFO *mbmi) {
  if (!mbmi->ref_frame[1]) return 0;
  const unsigned int cur_index = cm->cur_frame->order_hint;
  const RefCntBuffer *const ref0 = get_ref_frame_buf(cm, mbmi->ref_frame[0]);
  const RefCntBuffer *const ref1 = get_ref_frame_buf(cm, mbmi->ref_frame[1]);
  const int d0 = cur_index - ref0->order_hint;
  const int d1 = cur_index - ref1->order_hint;
  if (!((d0 <= 0) ^ (d1 <= 0))) return 0;

  return OPFL_DIST_RATIO_THR == 0 ||
         (AOMMAX(abs(d0), abs(d1)) <=
          OPFL_DIST_RATIO_THR * AOMMIN(abs(d0), abs(d1)));
}

#define OPTFLOW_INTEGER_MULT_DIVIDE 1
static INLINE int32_t divide_and_round_signed(int64_t P, int64_t D) {
#if OPTFLOW_INTEGER_MULT_DIVIDE
  if (llabs(D) == 1) return (int32_t)(D < 0 ? -P : P);
  static const int optflow_prec_bits = 16;
  int16_t shift;
  const int signD = (D < 0 ? -1 : 1);
  uint16_t iD = resolve_divisor_64(llabs(D), &shift);
  shift -= optflow_prec_bits;
  if (shift < 0) {
    iD <<= (-shift);
    shift = 0;
  }
  const int32_t v = (int32_t)ROUND_POWER_OF_TWO_SIGNED_64(
      P * (int64_t)iD * signD, optflow_prec_bits + shift);
#ifndef NDEBUG
  int32_t v0 = (int32_t)DIVIDE_AND_ROUND_SIGNED(P, D);
  if (abs(v0 - v) > 1 &&
      abs(v0) <= 64) {  // check if error is at most 1 at usable values of v0
    printf("Warning: D = %" PRId64 ", iD = %d, shift = %d, v0 = %d, v = %d\n",
           D, iD, shift, v0, v);
  }
#endif  // NDEBUG
#else
  const int32_t v = (int32_t)DIVIDE_AND_ROUND_SIGNED(P, D);
#endif  // OPTFLOW_INTEGER_MULT_DIVIDE
  return v;
}

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
    clamped_mv.row = (int16_t)ROUND_POWER_OF_TWO_SIGNED(
        src_mv->row * (1 << SUBPEL_BITS), MV_REFINE_PREC_BITS + ss_y);
    clamped_mv.col = (int16_t)ROUND_POWER_OF_TWO_SIGNED(
        src_mv->col * (1 << SUBPEL_BITS), MV_REFINE_PREC_BITS + ss_x);
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
  const SubpelMvLimits mv_limits = {
    xd->mb_to_left_edge * (1 << (1 - ss_x)) - spel_left,
    xd->mb_to_right_edge * (1 << (1 - ss_x)) + spel_right,
    xd->mb_to_top_edge * (1 << (1 - ss_y)) - spel_top,
    xd->mb_to_bottom_edge * (1 << (1 - ss_y)) + spel_bottom
  };

  clamp_mv(&clamped_mv, &mv_limits);

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
                                    const CHROMA_REF_INFO *chr_ref_info) {
  // Offset the buffer pointer
  /*  if (subsampling_y && (mi_row & 0x01) && (mi_size_high[bsize] == 1))
      mi_row -= 1;
    if (subsampling_x && (mi_col & 0x01) && (mi_size_wide[bsize] == 1))
      mi_col -= 1;*/
  if (chr_ref_info && (subsampling_x || subsampling_y)) {
    mi_row = chr_ref_info->mi_row_chroma_base;
    mi_col = chr_ref_info->mi_col_chroma_base;
  }

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
#if CONFIG_OPTFLOW_REFINEMENT
#if CONFIG_REMOVE_DUAL_FILTER
  mbmi->interp_fltr = mbmi->mode > NEW_NEWMV
                          ? MULTITAP_SHARP
                          : av1_unswitchable_filter(frame_interp_filter);
#else
  mbmi->interp_filters =
      mbmi->mode > NEW_NEWMV
          ? av1_broadcast_interp_filter(av1_unswitchable_filter(MULTITAP_SHARP))
          : av1_broadcast_interp_filter(
                av1_unswitchable_filter(frame_interp_filter));
#endif  // CONFIG_REMOVE_DUAL_FILTER
#else
#if CONFIG_REMOVE_DUAL_FILTER
  mbmi->interp_fltr = av1_unswitchable_filter(frame_interp_filter);
#else
  mbmi->interp_filters =
      av1_broadcast_interp_filter(av1_unswitchable_filter(frame_interp_filter));
#endif  // CONFIG_REMOVE_DUAL_FILTER
#endif  // CONFIG_OPTFLOW_REFINEMENT
}

static INLINE int av1_is_interp_needed(const MACROBLOCKD *const xd) {
  const MB_MODE_INFO *const mbmi = xd->mi[0];
  if (mbmi->skip_mode) return 0;
  if (mbmi->motion_mode == WARPED_CAUSAL) return 0;
  if (is_nontrans_global_motion(xd, xd->mi[0])) return 0;
  return 1;
}

// Sets up buffers 'dst_buf1' and 'dst_buf2' from relevant buffers in 'xd' for
// subsequent use in OBMC prediction.
void av1_setup_obmc_dst_bufs(MACROBLOCKD *xd, uint8_t **dst_buf1,
                             uint8_t **dst_buf2);

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

#if !CONFIG_REMOVE_DIST_WTD_COMP
void av1_dist_wtd_comp_weight_assign(const AV1_COMMON *cm,
                                     const MB_MODE_INFO *mbmi, int order_idx,
                                     int *fwd_offset, int *bck_offset,
                                     int is_compound);
#endif  // !CONFIG_REMOVE_DIST_WTD_COMP

const uint8_t *av1_get_compound_type_mask(
    const INTERINTER_COMPOUND_DATA *const comp_data, BLOCK_SIZE sb_type);

// build interintra_predictors for one plane
void av1_build_interintra_predictor(const AV1_COMMON *cm, MACROBLOCKD *xd,
                                    uint8_t *pred, int stride,
                                    const BUFFER_SET *ctx, int plane,
                                    BLOCK_SIZE bsize);

void av1_build_intra_predictors_for_interintra(const AV1_COMMON *cm,
                                               MACROBLOCKD *xd, int plane,
                                               const BUFFER_SET *ctx,
                                               uint8_t *dst, int dst_stride);

void av1_combine_interintra(MACROBLOCKD *xd, BLOCK_SIZE bsize, int plane,
                            const uint8_t *inter_pred, int inter_stride,
                            const uint8_t *intra_pred, int intra_stride);

int av1_allow_warp(const MB_MODE_INFO *const mbmi,
#if CONFIG_EXT_ROTATION
                   const MACROBLOCKD *xd,
#endif  // CONFIG_EXT_ROTATION
                   const WarpTypesAllowed *const warp_types,
                   const WarpedMotionParams *const gm_params,
                   int build_for_obmc, const struct scale_factors *const sf,
                   WarpedMotionParams *final_warp_params);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_RECONINTER_H_
