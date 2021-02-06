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

#ifndef AOM_AV1_COMMON_RESTORATION_H_
#define AOM_AV1_COMMON_RESTORATION_H_

#include "aom_ports/mem.h"
#include "config/aom_config.h"

#include "av1/common/blockd.h"
#include "av1/common/enums.h"

#if CONFIG_RST_MERGECOEFFS
#include "third_party/vector/vector.h"
#endif  // CONFIG_RST_MERGECOEFFS

#ifdef __cplusplus
extern "C" {
#endif

// Border for Loop restoration buffer
#define AOM_RESTORATION_FRAME_BORDER 32
#define CLIP(x, lo, hi) ((x) < (lo) ? (lo) : (x) > (hi) ? (hi) : (x))
#define RINT(x) ((x) < 0 ? (int)((x)-0.5) : (int)((x) + 0.5))

#define RESTORATION_PROC_UNIT_SIZE 64

// Filter tile grid offset upwards compared to the superblock grid
#define RESTORATION_UNIT_OFFSET 8

#define SGRPROJ_BORDER_VERT 3  // Vertical border used for Sgr
#define SGRPROJ_BORDER_HORZ 3  // Horizontal border used for Sgr

#define WIENER_BORDER_VERT 2  // Vertical border used for Wiener
#define WIENER_HALFWIN 3
#define WIENER_BORDER_HORZ (WIENER_HALFWIN)  // Horizontal border for Wiener

// RESTORATION_BORDER_VERT determines line buffer requirement for LR.
// Should be set at the max of SGRPROJ_BORDER_VERT and WIENER_BORDER_VERT.
// Note the line buffer needed is twice the value of this macro.
#if SGRPROJ_BORDER_VERT >= WIENER_BORDER_VERT
#define RESTORATION_BORDER_VERT (SGRPROJ_BORDER_VERT)
#else
#define RESTORATION_BORDER_VERT (WIENER_BORDER_VERT)
#endif  // SGRPROJ_BORDER_VERT >= WIENER_BORDER_VERT

#if SGRPROJ_BORDER_HORZ >= WIENER_BORDER_HORZ
#define RESTORATION_BORDER_HORZ (SGRPROJ_BORDER_HORZ)
#else
#define RESTORATION_BORDER_HORZ (WIENER_BORDER_HORZ)
#endif  // SGRPROJ_BORDER_VERT >= WIENER_BORDER_VERT

// How many border pixels do we need for each processing unit?
#define RESTORATION_BORDER 3

// How many rows of deblocked pixels do we save above/below each processing
// stripe?
#define RESTORATION_CTX_VERT 2

// Additional pixels to the left and right in above/below buffers
// It is RESTORATION_BORDER_HORZ rounded up to get nicer buffer alignment
#define RESTORATION_EXTRA_HORZ 4

// Pad up to 20 more (may be much less is needed)
#define RESTORATION_PADDING 20
#define RESTORATION_PROC_UNIT_PELS                             \
  ((RESTORATION_PROC_UNIT_SIZE + RESTORATION_BORDER_HORZ * 2 + \
    RESTORATION_PADDING) *                                     \
   (RESTORATION_PROC_UNIT_SIZE + RESTORATION_BORDER_VERT * 2 + \
    RESTORATION_PADDING))

#define RESTORATION_UNITSIZE_MAX 256
#define RESTORATION_UNITPELS_HORZ_MAX \
  (RESTORATION_UNITSIZE_MAX * 3 / 2 + 2 * RESTORATION_BORDER_HORZ + 16)
#define RESTORATION_UNITPELS_VERT_MAX                                \
  ((RESTORATION_UNITSIZE_MAX * 3 / 2 + 2 * RESTORATION_BORDER_VERT + \
    RESTORATION_UNIT_OFFSET))
#define RESTORATION_UNITPELS_MAX \
  (RESTORATION_UNITPELS_HORZ_MAX * RESTORATION_UNITPELS_VERT_MAX)

// Two 32-bit buffers needed for the restored versions from two filters
// TODO(debargha, rupert): Refactor to not need the large tilesize to be stored
// on the decoder side.
#define SGRPROJ_TMPBUF_SIZE (RESTORATION_UNITPELS_MAX * 2 * sizeof(int32_t))

#define SGRPROJ_EXTBUF_SIZE (0)
#define SGRPROJ_PARAMS_BITS 4
#define SGRPROJ_PARAMS (1 << SGRPROJ_PARAMS_BITS)

// Precision bits for projection
#define SGRPROJ_PRJ_BITS 7
// Restoration precision bits generated higher than source before projection
#define SGRPROJ_RST_BITS 4
// Internal precision bits for core selfguided_restoration
#define SGRPROJ_SGR_BITS 8
#define SGRPROJ_SGR (1 << SGRPROJ_SGR_BITS)

#define SGRPROJ_PRJ_MIN0 (-(1 << SGRPROJ_PRJ_BITS) * 3 / 4)
#define SGRPROJ_PRJ_MAX0 (SGRPROJ_PRJ_MIN0 + (1 << SGRPROJ_PRJ_BITS) - 1)
#define SGRPROJ_PRJ_MIN1 (-(1 << SGRPROJ_PRJ_BITS) / 4)
#define SGRPROJ_PRJ_MAX1 (SGRPROJ_PRJ_MIN1 + (1 << SGRPROJ_PRJ_BITS) - 1)

#define SGRPROJ_PRJ_SUBEXP_K 4

#define SGRPROJ_BITS (SGRPROJ_PRJ_BITS * 2 + SGRPROJ_PARAMS_BITS)

#define MAX_RADIUS 2  // Only 1, 2, 3 allowed
#define MAX_NELEM ((2 * MAX_RADIUS + 1) * (2 * MAX_RADIUS + 1))
#define SGRPROJ_MTABLE_BITS 20
#define SGRPROJ_RECIP_BITS 12

#define WIENER_HALFWIN1 (WIENER_HALFWIN + 1)
#define WIENER_WIN (2 * WIENER_HALFWIN + 1)
#define WIENER_WIN2 ((WIENER_WIN) * (WIENER_WIN))
#define WIENER_TMPBUF_SIZE (0)
#define WIENER_EXTBUF_SIZE (0)

// If WIENER_WIN_CHROMA == WIENER_WIN - 2, that implies 5x5 filters are used for
// chroma. To use 7x7 for chroma set WIENER_WIN_CHROMA to WIENER_WIN.
#define WIENER_WIN_CHROMA (WIENER_WIN - 2)
#define WIENER_HALFWIN_CHROMA (WIENER_HALFWIN - 1)
#define WIENER_WIN_REDUCED (WIENER_WIN - 2)
#define WIENER_WIN2_CHROMA ((WIENER_WIN_CHROMA) * (WIENER_WIN_CHROMA))

#if CONFIG_WIENER_SEP_HIPREC
#define WIENER_FILT_PREC_BITS (FILTER_BITS + 1)
#else
#define WIENER_FILT_PREC_BITS (FILTER_BITS)
#endif  // CONFIG_WIENER_SEP_HIPREC

#define WIENER_FILT_STEP (1 << WIENER_FILT_PREC_BITS)

// Central values for the taps
#define WIENER_FILT_TAP0_MIDV (3 * (1 << (WIENER_FILT_PREC_BITS - FILTER_BITS)))
#define WIENER_FILT_TAP1_MIDV \
  (-7 * (1 << (WIENER_FILT_PREC_BITS - FILTER_BITS)))
#define WIENER_FILT_TAP2_MIDV \
  (15 * (1 << (WIENER_FILT_PREC_BITS - FILTER_BITS)))
#define WIENER_FILT_TAP3_MIDV                                              \
  (WIENER_FILT_STEP - 2 * (WIENER_FILT_TAP0_MIDV + WIENER_FILT_TAP1_MIDV + \
                           WIENER_FILT_TAP2_MIDV))

#define WIENER_FILT_TAP0_BITS (4 + (WIENER_FILT_PREC_BITS - FILTER_BITS))
#define WIENER_FILT_TAP1_BITS (5 + (WIENER_FILT_PREC_BITS - FILTER_BITS))
#define WIENER_FILT_TAP2_BITS (6 + (WIENER_FILT_PREC_BITS - FILTER_BITS))

#define WIENER_FILT_BITS \
  ((WIENER_FILT_TAP0_BITS + WIENER_FILT_TAP1_BITS + WIENER_FILT_TAP2_BITS) * 2)

#define WIENER_FILT_TAP0_MINV \
  (WIENER_FILT_TAP0_MIDV - (1 << WIENER_FILT_TAP0_BITS) / 2)
#define WIENER_FILT_TAP1_MINV \
  (WIENER_FILT_TAP1_MIDV - (1 << WIENER_FILT_TAP1_BITS) / 2)
#define WIENER_FILT_TAP2_MINV \
  (WIENER_FILT_TAP2_MIDV - (1 << WIENER_FILT_TAP2_BITS) / 2)

#define WIENER_FILT_TAP0_MAXV \
  (WIENER_FILT_TAP0_MIDV - 1 + (1 << WIENER_FILT_TAP0_BITS) / 2)
#define WIENER_FILT_TAP1_MAXV \
  (WIENER_FILT_TAP1_MIDV - 1 + (1 << WIENER_FILT_TAP1_BITS) / 2)
#define WIENER_FILT_TAP2_MAXV \
  (WIENER_FILT_TAP2_MIDV - 1 + (1 << WIENER_FILT_TAP2_BITS) / 2)

#define WIENER_FILT_TAP0_SUBEXP_K (1 + (WIENER_FILT_PREC_BITS - FILTER_BITS))
#define WIENER_FILT_TAP1_SUBEXP_K (2 + (WIENER_FILT_PREC_BITS - FILTER_BITS))
#define WIENER_FILT_TAP2_SUBEXP_K (3 + (WIENER_FILT_PREC_BITS - FILTER_BITS))

#if CONFIG_WIENER_NONSEP
#define WIENERNS_PREC_BITS_Y 8
#define WIENERNS_PREC_BITS_UV 7

#if CONFIG_WIENER_NONSEP_CROSS_FILT
#define WIENERNS_UV_BRD 2  // Max offset for luma used for chorma
#else
#define WIENERNS_UV_BRD 0  // Max offset for luma used for chorma
#endif                     // CONFIG_WIENER_NONSEP_CROSS_FILT

// Apply masking for nonseparable Wiener restoration
#define WIENER_NONSEP_MASK 0
#if WIENER_NONSEP_MASK
// Skip a pixel if all pixels in an NxN window around it have zero transform
// coefficients
#define LOOKAROUND_WIN 1
#endif  // WIENER_NONSEP_MASK

#define WIENERNS_MAX 20

#define WIENERNS_ROW_ID 0
#define WIENERNS_COL_ID 1
#define WIENERNS_BUF_POS 2

#define WIENERNS_BIT_ID 0
#define WIENERNS_MIN_ID 1
#define WIENERNS_SUBEXP_K_ID 2
#define WIENERNS_STEP_ID 3
extern const int wienerns_prec_bits_y;
extern const int wienerns_prec_bits_uv;
extern const int wienerns_y_pixel;  // Number of pixels used for filtering luma
extern const int wienerns_uv_from_uv_pixel;  // Number of pixels used for
                                             // filtering uv from uv only
extern const int wienerns_y;   // Number of luma coefficients in all
extern const int wienerns_uv;  // Number of chroma coefficients in all
extern const int wienerns_config_y[][3];
extern const int wienerns_config_uv_from_uv[][3];
#if CONFIG_WIENER_NONSEP_CROSS_FILT
extern const int wienerns_uv_from_y_pixel;  // Number of pixels used for
                                            // filtering uv from y
extern const int wienerns_config_uv_from_y[][3];
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT
extern const int wienerns_coeff_y[][3];
extern const int wienerns_coeff_uv[][3];
#endif  // CONFIG_WIENER_NONSEP

// Max of SGRPROJ_TMPBUF_SIZE, DOMAINTXFMRF_TMPBUF_SIZE, WIENER_TMPBUF_SIZE
#define RESTORATION_TMPBUF_SIZE (SGRPROJ_TMPBUF_SIZE)

// Max of SGRPROJ_EXTBUF_SIZE, WIENER_EXTBUF_SIZE
#define RESTORATION_EXTBUF_SIZE (WIENER_EXTBUF_SIZE)

// Check the assumptions of the existing code
#if SUBPEL_TAPS != WIENER_WIN + 1
#error "Wiener filter currently only works if SUBPEL_TAPS == WIENER_WIN + 1"
#endif

#if CONFIG_WIENER_SEP_HIPREC
#if WIENER_FILT_PREC_BITS < 7
#error "Wiener filter currently only works if WIENER_FILT_PREC_BITS >= 7"
#endif
#else
#if WIENER_FILT_PREC_BITS != 7
#error "Wiener filter currently only works if WIENER_FILT_PREC_BITS == 7"
#endif
#endif  // CONFIG_WIENER_SEP_HIPREC

#define LR_TILE_ROW 0
#define LR_TILE_COL 0
#define LR_TILE_COLS 1

typedef struct {
  int r[2];  // radii
  int s[2];  // sgr parameters for r[0] and r[1], based on GenSgrprojVtable()
} sgr_params_type;

typedef struct {
  RestorationType restoration_type;
  WienerInfo wiener_info;
  SgrprojInfo sgrproj_info;
#if CONFIG_LOOP_RESTORE_CNN
  CNNInfo cnn_info;
#endif  // CONFIG_LOOP_RESTORE_CNN
#if CONFIG_WIENER_NONSEP
  WienerNonsepInfo wiener_nonsep_info;
#if WIENER_NONSEP_MASK
  // pointer to tx_skip array at the first pixel of the current RU
  const uint8_t *txskip_mask;
  int mask_stride;
  int mask_height;
  int v_start;
  int h_start;
#endif  // WIENER_NONSEP_MASK
#if CONFIG_WIENER_NONSEP_CROSS_FILT
  const uint8_t *luma;
  int luma_stride;
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT
  int plane;
#endif  // CONFIG_WIENER_NONSEP
} RestorationUnitInfo;

// A restoration line buffer needs space for two lines plus a horizontal filter
// margin of RESTORATION_EXTRA_HORZ on each side.
#define RESTORATION_LINEBUFFER_WIDTH \
  (RESTORATION_UNITSIZE_MAX * 3 / 2 + 2 * RESTORATION_EXTRA_HORZ)

// Similarly, the column buffers (used when we're at a vertical tile edge
// that we can't filter across) need space for one processing unit's worth
// of pixels, plus the top/bottom border width
#define RESTORATION_COLBUFFER_HEIGHT \
  (RESTORATION_PROC_UNIT_SIZE + 2 * RESTORATION_BORDER)

typedef struct {
  // Temporary buffers to save/restore 3 lines above/below the restoration
  // stripe.
  uint16_t tmp_save_above[RESTORATION_BORDER][RESTORATION_LINEBUFFER_WIDTH];
  uint16_t tmp_save_below[RESTORATION_BORDER][RESTORATION_LINEBUFFER_WIDTH];
} RestorationLineBuffers;

typedef struct {
  uint8_t *stripe_boundary_above;
  uint8_t *stripe_boundary_below;
  int stripe_boundary_stride;
  int stripe_boundary_size;
} RestorationStripeBoundaries;

typedef struct {
  RestorationType frame_restoration_type;
  int restoration_unit_size;

  // Fields below here are allocated and initialised by
  // av1_alloc_restoration_struct. (horz_)units_per_tile give the number of
  // restoration units in (one row of) the largest tile in the frame. The data
  // in unit_info is laid out with units_per_tile entries for each tile, which
  // have stride horz_units_per_tile.
  //
  // Even if there are tiles of different sizes, the data in unit_info is laid
  // out as if all tiles are of full size.
  int units_per_tile;
  int vert_units_per_tile, horz_units_per_tile;
  RestorationUnitInfo *unit_info;
  RestorationStripeBoundaries boundaries;
  int optimized_lr;
} RestorationInfo;

#if CONFIG_CNN_CRLC_GUIDED
typedef struct {
  int crlc_unit_size;
  int units_per_tile;
  int num_crlc_unit;
  int vert_units_per_tile, horz_units_per_tile;
  CRLCUnitInfo *unit_info;
} CRLCInfo;
#endif  // CONFIG_CNN_CRLC_GUIDED

static INLINE void set_default_sgrproj(SgrprojInfo *sgrproj_info) {
  sgrproj_info->xqd[0] = (SGRPROJ_PRJ_MIN0 + SGRPROJ_PRJ_MAX0) / 2;
  sgrproj_info->xqd[1] = (SGRPROJ_PRJ_MIN1 + SGRPROJ_PRJ_MAX1) / 2;
}

#if CONFIG_CNN_CRLC_GUIDED
static INLINE void set_default_crlc(CRLCUnitInfo *cui) {
  cui->xqd[0] = 16 / 2;
  cui->xqd[1] = 16 / 2;
}
#endif  // CONFIG_CNN_CRLC_GUIDED

static INLINE void set_default_wiener(WienerInfo *wiener_info) {
  wiener_info->vfilter[0] = wiener_info->hfilter[0] = WIENER_FILT_TAP0_MIDV;
  wiener_info->vfilter[1] = wiener_info->hfilter[1] = WIENER_FILT_TAP1_MIDV;
  wiener_info->vfilter[2] = wiener_info->hfilter[2] = WIENER_FILT_TAP2_MIDV;
  wiener_info->vfilter[WIENER_HALFWIN] = wiener_info->hfilter[WIENER_HALFWIN] =
      -2 *
      (WIENER_FILT_TAP2_MIDV + WIENER_FILT_TAP1_MIDV + WIENER_FILT_TAP0_MIDV);
  wiener_info->vfilter[4] = wiener_info->hfilter[4] = WIENER_FILT_TAP2_MIDV;
  wiener_info->vfilter[5] = wiener_info->hfilter[5] = WIENER_FILT_TAP1_MIDV;
  wiener_info->vfilter[6] = wiener_info->hfilter[6] = WIENER_FILT_TAP0_MIDV;
}

#if CONFIG_RST_MERGECOEFFS
static INLINE int check_wiener_eq(const WienerInfo *info,
                                  const WienerInfo *ref) {
  return !memcmp(info->vfilter, ref->vfilter,
                 WIENER_HALFWIN * sizeof(info->vfilter[0])) &&
         !memcmp(info->hfilter, ref->hfilter,
                 WIENER_HALFWIN * sizeof(info->hfilter[0]));
}
static INLINE int check_sgrproj_eq(const SgrprojInfo *info,
                                   const SgrprojInfo *ref) {
  if (!memcmp(info, ref, sizeof(*info))) return 1;
  return 0;
}
#endif  // CONFIG_RST_MERGECOEFFS

#if CONFIG_WIENER_NONSEP
static INLINE void set_default_wiener_nonsep(WienerNonsepInfo *wienerns_info) {
  for (int i = 0; i < wienerns_y; ++i) {
    wienerns_info->nsfilter[i] = wienerns_coeff_y[i][WIENERNS_MIN_ID];
  }
  for (int i = wienerns_y; i < wienerns_y + wienerns_uv; ++i) {
    wienerns_info->nsfilter[i] =
        wienerns_coeff_uv[i - wienerns_y][WIENERNS_MIN_ID];
  }
}

#if CONFIG_RST_MERGECOEFFS
static INLINE int check_wienerns_eq(int chroma, const WienerNonsepInfo *info,
                                    const WienerNonsepInfo *ref) {
  if (!chroma) {
    if (!memcmp(info->nsfilter, ref->nsfilter,
                wienerns_y * sizeof(*info->nsfilter)))
      return 1;
  } else {
    if (!memcmp(&info->nsfilter[wienerns_y], &ref->nsfilter[wienerns_y],
                wienerns_uv * sizeof(*info->nsfilter)))
      return 1;
  }
  return 0;
}
#endif  // CONFIG_RST_MERGECOEFFS

#if CONFIG_WIENER_NONSEP_CROSS_FILT
uint8_t *wienerns_copy_luma(const uint8_t *dgd, int height_y, int width_y,
                            int in_stride, uint8_t **luma, int height_uv,
                            int width_uv, int border, int out_stride);
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT

#endif  // CONFIG_WIENER_NONSEP

typedef struct {
  int h_start, h_end, v_start, v_end;
} RestorationTileLimits;

#if CONFIG_RST_MERGECOEFFS
typedef struct RstUnitSnapshot {
  RestorationTileLimits limits;
  int rest_unit_idx;  // update filter value and sse as needed
  int64_t current_sse;
  int64_t current_bits;
  int64_t merge_sse;
  int64_t merge_bits;
  // Wiener filter info
  int64_t M[WIENER_WIN2];
  int64_t H[WIENER_WIN2 * WIENER_WIN2];
  WienerInfo ref_wiener;
#if CONFIG_WIENER_NONSEP
  // Nonseparable Wiener filter info
  double A[WIENERNS_MAX * WIENERNS_MAX];
  double b[WIENERNS_MAX];
  WienerNonsepInfo ref_wiener_nonsep;
#endif  // CONFIG_WIENER_NONSEP
  // Sgrproj filter info
  SgrprojInfo unit_sgrproj;
  SgrprojInfo ref_sgrproj;
} RstUnitSnapshot;
#endif  // CONFIG_RST_MERGECOEFFS

typedef void (*rest_unit_visitor_t)(const RestorationTileLimits *limits,
                                    const AV1PixelRect *tile_rect,
                                    int rest_unit_idx, void *priv,
                                    int32_t *tmpbuf,
                                    RestorationLineBuffers *rlbs);

typedef struct FilterFrameCtxt {
  const RestorationInfo *rsi;
  int tile_stripe0;
  int ss_x, ss_y;
  int highbd, bit_depth;
  uint8_t *data8, *dst8;
  int data_stride, dst_stride;
  AV1PixelRect tile_rect;
  int base_qindex;
  FRAME_TYPE frame_type;
#if CONFIG_LOOP_RESTORE_CNN
  bool is_luma;
#endif  // CONFIG_LOOP_RESTORE_CNN
#if CONFIG_WIENER_NONSEP
  int plane;
#if WIENER_NONSEP_MASK
  // pointer to tx_skip array at the first pixel of the plane
  const uint8_t *txskip_mask;
  int mask_stride;
  int mask_height;
#endif  // WIENER_NONSEP_MASK
#if CONFIG_WIENER_NONSEP_CROSS_FILT
  const uint8_t *luma;
  int luma_stride;
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT
#endif  // CONFIG_WIENER_NONSEP
} FilterFrameCtxt;

typedef struct AV1LrStruct {
  rest_unit_visitor_t on_rest_unit;
  FilterFrameCtxt ctxt[MAX_MB_PLANE];
  YV12_BUFFER_CONFIG *frame;
  YV12_BUFFER_CONFIG *dst;
} AV1LrStruct;

extern const sgr_params_type av1_sgr_params[SGRPROJ_PARAMS];
extern int sgrproj_mtable[SGRPROJ_PARAMS][2];
extern const int32_t av1_x_by_xplus1[256];
extern const int32_t av1_one_by_x[MAX_NELEM];

void av1_alloc_restoration_struct(struct AV1Common *cm, RestorationInfo *rsi,
                                  int is_uv);
void av1_free_restoration_struct(RestorationInfo *rst_info);

#if CONFIG_CNN_CRLC_GUIDED
void av1_alloc_CRLC_struct(struct AV1Common *cm, CRLCInfo *ci, int is_uv);
void av1_free_CRLC_struct(CRLCInfo *crlc_info);
#endif  // CONFIG_CNN_CRLC_GUIDED

void av1_extend_frame(uint8_t *data, int width, int height, int stride,
                      int border_horz, int border_vert, int highbd);
void av1_decode_xq(const int *xqd, int *xq, const sgr_params_type *params);

// Filter a single loop restoration unit.
//
// limits is the limits of the unit. rui gives the mode to use for this unit
// and its coefficients. If striped loop restoration is enabled, rsb contains
// deblocked pixels to use for stripe boundaries; rlbs is just some space to
// use as a scratch buffer. tile_rect gives the limits of the tile containing
// this unit. tile_stripe0 is the index of the first stripe in this tile.
//
// ss_x and ss_y are flags which should be 1 if this is a plane with
// horizontal/vertical subsampling, respectively. highbd is a flag which should
// be 1 in high bit depth mode, in which case bit_depth is the bit depth.
//
// data8 is the frame data (pointing at the top-left corner of the frame, not
// the restoration unit) and stride is its stride. dst8 is the buffer where the
// results will be written and has stride dst_stride. Like data8, dst8 should
// point at the top-left corner of the frame.
//
// Finally tmpbuf is a scratch buffer used by the sgrproj filter which should
// be at least SGRPROJ_TMPBUF_SIZE big.
void av1_loop_restoration_filter_unit(const RestorationTileLimits *limits,
                                      RestorationUnitInfo *rui,
                                      const RestorationStripeBoundaries *rsb,
#if CONFIG_LOOP_RESTORE_CNN
                                      int restoration_unit_size,
#endif  // CONFIG_LOOP_RESTORE_CNN
                                      RestorationLineBuffers *rlbs,
                                      const AV1PixelRect *tile_rect,
                                      int tile_stripe0, int ss_x, int ss_y,
                                      int highbd, int bit_depth, uint8_t *data8,
                                      int stride, uint8_t *dst8, int dst_stride,
                                      int32_t *tmpbuf, int optimized_lr);

void av1_loop_restoration_filter_frame(YV12_BUFFER_CONFIG *frame,
                                       struct AV1Common *cm, int optimized_lr,
                                       void *lr_ctxt);
void av1_loop_restoration_precal();

typedef void (*rest_tile_start_visitor_t)(int tile_row, int tile_col,
                                          void *priv);
struct AV1LrSyncData;

typedef void (*sync_read_fn_t)(void *const lr_sync, int r, int c, int plane);

typedef void (*sync_write_fn_t)(void *const lr_sync, int r, int c,
                                const int sb_cols, int plane);

// Call on_rest_unit for each loop restoration unit in the plane.
void av1_foreach_rest_unit_in_plane(const struct AV1Common *cm, int plane,
                                    rest_unit_visitor_t on_rest_unit,
                                    void *priv, AV1PixelRect *tile_rect,
                                    int32_t *tmpbuf,
                                    RestorationLineBuffers *rlbs);

// Return 1 iff the block at mi_row, mi_col with size bsize is a
// top-level superblock containing the top-left corner of at least one
// loop restoration unit.
//
// If the block is a top-level superblock, the function writes to
// *rcol0, *rcol1, *rrow0, *rrow1. The rectangle of restoration unit
// indices given by [*rcol0, *rcol1) x [*rrow0, *rrow1) are relative
// to the current tile, whose starting index is returned as
// *tile_tl_idx.
int av1_loop_restoration_corners_in_sb(const struct AV1Common *cm, int plane,
                                       int mi_row, int mi_col, BLOCK_SIZE bsize,
                                       int *rcol0, int *rcol1, int *rrow0,
                                       int *rrow1);
#if CONFIG_CNN_CRLC_GUIDED
int av1_CRLC_corners_in_sb(const struct AV1Common *cm, int plane, int mi_row,
                           int mi_col, BLOCK_SIZE bsize, int *rcol0, int *rcol1,
                           int *rrow0, int *rrow1);
#endif  // CONFIG_CNN_CRLC_GUIDED

void av1_loop_restoration_save_boundary_lines(const YV12_BUFFER_CONFIG *frame,
                                              struct AV1Common *cm,
                                              int after_cdef);
void av1_loop_restoration_filter_frame_init(AV1LrStruct *lr_ctxt,
                                            YV12_BUFFER_CONFIG *frame,
                                            struct AV1Common *cm,
                                            int optimized_lr, int num_planes);
void av1_loop_restoration_copy_planes(AV1LrStruct *loop_rest_ctxt,
                                      struct AV1Common *cm, int num_planes);
void av1_foreach_rest_unit_in_row(
    RestorationTileLimits *limits, const AV1PixelRect *tile_rect,
    rest_unit_visitor_t on_rest_unit, int row_number, int unit_size,
    int unit_idx0, int hunits_per_tile, int vunits_per_tile, int plane,
    void *priv, int32_t *tmpbuf, RestorationLineBuffers *rlbs,
    sync_read_fn_t on_sync_read, sync_write_fn_t on_sync_write,
    struct AV1LrSyncData *const lr_sync);
AV1PixelRect av1_whole_frame_rect(const struct AV1Common *cm, int is_uv);
int av1_lr_count_units_in_tile(int unit_size, int tile_size);
void av1_lr_sync_read_dummy(void *const lr_sync, int r, int c, int plane);
void av1_lr_sync_write_dummy(void *const lr_sync, int r, int c,
                             const int sb_cols, int plane);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_RESTORATION_H_
