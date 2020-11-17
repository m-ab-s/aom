#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "av1/common/reconinter.h"
#include "av1/common/reconintra.h"
#include "av1/encoder/interintra_ml_data_collect.h"
#include "av1/encoder/reconinter_enc.h"

#define BORDER_SIZE 4  // Only generate data with a border of 4 pixels.

// Static references to the currently captured Y/U/V planes. Held in
// memory until we can determine if this block is a skip-block (if so,
// the data is not written out).
static IIMLPlaneInfo *INFO_Y = NULL;
static IIMLPlaneInfo *INFO_U = NULL;
static IIMLPlaneInfo *INFO_V = NULL;

// Static reference to the file where the data is being stored.
static FILE *FP = NULL;

// Clean-up function called on program exit. Closes file descriptor,
// flushing writes.
static void cleanup_fp() {
  if (FP != NULL) {
    int r = fclose(FP);
    assert(r == 0);
    (void)r;
  }
}

static uint8_t read_interintra_bias() {
  const char *bias = getenv("INTERINTRA_BIAS");
  if (bias == NULL) {
    return 100;
  }
  int i = atoi(bias);
  if (i < 0 || i > 100) {
    fprintf(stderr, "INTERINTRA_BIAS must be between 0 and 100.\n");
    exit(1);
  }
  return (uint8_t)i;
}

static uint8_t INTERINTRA_BIAS = 255;

uint8_t av1_interintra_bias() {
  if (INTERINTRA_BIAS == 255) {
    INTERINTRA_BIAS = read_interintra_bias();
  }
  return INTERINTRA_BIAS;
}

// Function called on every invocation of data collection. Initializes
// the file output and registers the cleanup function (if not already done).
// The output file name is "interintra_ml_data_collect.bin" by default.
// The file name can be changed by setting INTERINTRA_ML_DATA_COLLECT
// environment variable.
static void init_first_run() {
  if (FP != NULL) {
    return;
  }
  const char *filename = getenv("INTERINTRA_ML_DATA_COLLECT");
  if (filename == NULL) {
    filename = "interintra_ml_data_collect.bin";
  }
  FP = fopen(filename, "wb");
  assert(FP != NULL);
  int r = atexit(cleanup_fp);
  assert(r == 0);
  (void)r;
}

// Copy the value from the source into the destination, at the given offsets.
// If both source and destination are high-bitdepth, will copy. Otherwise,
// if source is high-bitdepth but destination is lower, will only copy the
// first 8 bits.
static void copy_value(uint8_t *dst, int dst_offset, const uint8_t *src,
                       int src_offset, int bitdepth, bool is_hbd) {
  if (bitdepth > 8) {
    assert(is_hbd);  // If the IIMLPlaneInfo is high-bitdepth, then so must the
    // AV1 data structures.
    uint16_t *dst16 = (uint16_t *)dst;
    uint16_t *src16 = CONVERT_TO_SHORTPTR(src);
    dst16[dst_offset] = src16[src_offset];
  } else if (bitdepth == 8 && !is_hbd) {
    // A pure copy.
    dst[dst_offset] = src[src_offset];
  } else {
    // There is no case when bitdepth > 8 and !is_hbd.
    assert(bitdepth == 8 && is_hbd);
    uint16_t *src16 = CONVERT_TO_SHORTPTR(src);
    assert(src16[src_offset] <= 255);
    dst[dst_offset] = (uint8_t)src16[src_offset];
  }
}

static void copy_source_image(IIMLPlaneInfo *info, MACROBLOCK *const x) {
  // Overallocate, memory is not an issue.
  info->source_image = malloc(info->width * info->height * sizeof(uint16_t));
  assert(info->source_image != NULL);

  const struct buf_2d *ref = &x->plane[info->plane].src;
  const int stride = ref->stride;
  const uint8_t *buf = ref->buf0 + info->y * stride + info->x;
  const bool is_hbd = is_cur_buf_hbd(&x->e_mbd);

  for (int j = 0; j < info->height; ++j) {
    for (int i = 0; i < info->width; ++i) {
      copy_value(info->source_image, j * info->width + i, buf, j * stride + i,
                 info->bitdepth, is_hbd);
    }
  }
}

static void copy_intrapred_lshape(IIMLPlaneInfo *info, MACROBLOCK *const x) {
  info->intrapred_lshape =
      malloc(sizeof(uint16_t) * ((info->width + info->border) * info->border +
                                 info->border * info->height));
  assert(info->intrapred_lshape != NULL);
  MACROBLOCKD *const xd = &x->e_mbd;
  const struct macroblockd_plane *const pd = &xd->plane[info->plane];
  const struct buf_2d *ref = &pd->dst;
  const int stride = ref->stride;
  uint8_t *buf = ref->buf0 + info->y * stride + info->x;
  av1_interintra_ml_data_collect_copy_intrapred_lshape(
      info->intrapred_lshape, buf, stride, info->width, info->height,
      info->border, info->bitdepth, is_cur_buf_hbd(xd));
}

void av1_interintra_ml_data_collect_copy_intrapred_lshape(
    uint8_t *dst, const uint8_t *src, int src_stride, int width, int height,
    int border, int bitdepth, bool is_src_hbd) {
  // Point back at the start of the L-region.
  src -= (border * src_stride + border);
  int info_i = 0;
  // Copy over the top part.
  for (int j = 0; j < border; ++j) {
    for (int i = 0; i < border + width; ++i) {
      copy_value(dst, info_i++, src, j * src_stride + i, bitdepth, is_src_hbd);
    }
  }

  // Copy over the side pixels.
  for (int j = border; j < border + height; ++j) {
    for (int i = 0; i < border; ++i) {
      copy_value(dst, info_i++, src, j * src_stride + i, bitdepth, is_src_hbd);
    }
  }
}

static void copy_interpred(IIMLPlaneInfo *info, const AV1_COMP *const cpi,
                           MACROBLOCK *const x) {
  info->interpred = malloc(sizeof(uint16_t) * (info->width + info->border) *
                           (info->height + info->border));

  const AV1_COMMON *const cm = &cpi->common;
  MACROBLOCKD *const xd = &x->e_mbd;

  uint8_t *orig_buf = xd->plane[info->plane].dst.buf;
  int orig_stride = xd->plane[info->plane].dst.stride;

  uint8_t *interpred;
  int interpred_stride;
  const int border =
      av1_calc_border(xd, AOM_PLANE_Y, false /* build for obmc */);
  assert(border >= BORDER_SIZE);
  av1_alloc_buf_with_border(&interpred, &interpred_stride, border,
                            is_cur_buf_hbd(xd));

  xd->plane[info->plane].dst.buf = interpred;
  xd->plane[info->plane].dst.stride = interpred_stride;

  av1_enc_build_border_only_inter_predictor(cm, xd, xd->mi_row, xd->mi_col,
                                            info->plane);

  int info_i = 0;
  for (int j = -1 * info->border; j < info->height; ++j) {
    for (int i = -1 * info->border; i < info->width; ++i) {
      copy_value(info->interpred, info_i++, interpred, j * interpred_stride + i,
                 info->bitdepth, is_cur_buf_hbd(xd));
    }
  }
  xd->plane[info->plane].dst.buf = orig_buf;
  xd->plane[info->plane].dst.stride = orig_stride;
  av1_free_buf_with_border(interpred, interpred_stride, border,
                           is_cur_buf_hbd(xd));
}

static void copy_predictor(IIMLPlaneInfo *info, const AV1_COMP *const cpi,
                           MACROBLOCK *const x, BLOCK_SIZE bsize) {
  info->predictor = malloc(sizeof(uint16_t) * info->width * info->height);

  const AV1_COMMON *const cm = &cpi->common;
  MACROBLOCKD *const xd = &x->e_mbd;

  uint8_t *orig_buf = xd->plane[info->plane].dst.buf;
  int orig_stride = xd->plane[info->plane].dst.stride;

  uint8_t *predictor;
  int predictor_stride;
  const int border =
      av1_calc_border(xd, AOM_PLANE_Y, false /* build for obmc */);
  assert(border >= BORDER_SIZE);
  av1_alloc_buf_with_border(&predictor, &predictor_stride, border,
                            is_cur_buf_hbd(xd));

  xd->plane[info->plane].dst.buf = predictor;
  xd->plane[info->plane].dst.stride = predictor_stride;

  av1_enc_build_inter_predictor(cm, xd, xd->mi_row, xd->mi_col, NULL, bsize,
                                info->plane, info->plane);
  int info_i = 0;
  for (int j = 0; j < info->height; ++j) {
    for (int i = 0; i < info->width; ++i) {
      copy_value(info->predictor, info_i++, predictor, j * predictor_stride + i,
                 info->bitdepth, is_cur_buf_hbd(xd));
    }
  }
  xd->plane[info->plane].dst.buf = orig_buf;
  xd->plane[info->plane].dst.stride = orig_stride;
  av1_free_buf_with_border(predictor, predictor_stride, border,
                           is_cur_buf_hbd(xd));
}

// Initializes the IIMLPlaneInfo structure.
static IIMLPlaneInfo *init_plane_info(const AV1_COMP *const cpi,
                                      MACROBLOCK *const x, BLOCK_SIZE bsize,
                                      int plane) {
  MACROBLOCKD *const xd = &x->e_mbd;
  MB_MODE_INFO *mbmi = xd->mi[0];
  (void)mbmi;  // Used for asserts.
  const AV1_COMMON *const cm = &cpi->common;
  const struct macroblockd_plane *const pd = &xd->plane[plane];
  const BLOCK_SIZE plane_bsize =
      get_plane_block_size(bsize, pd->subsampling_x, pd->subsampling_y);

  assert(is_inter_block(mbmi));
  assert(!has_second_ref(mbmi));
  assert(!is_intrabc_block(mbmi));
  // Data is packed differently for smaller block sizes. Not supported.
  assert(block_size_wide[plane_bsize] >= 4);
  assert(block_size_high[plane_bsize] >= 4);

  IIMLPlaneInfo *info = malloc(sizeof(IIMLPlaneInfo));
  assert(info != NULL);

  info->width = block_size_wide[plane_bsize];
  info->height = block_size_high[plane_bsize];
  info->plane = plane;
  info->bitdepth = (uint8_t)xd->bd;
  info->border = BORDER_SIZE;
  info->x = (xd->mi_col * MI_SIZE) >> pd->subsampling_x;
  info->y = (xd->mi_row * MI_SIZE) >> pd->subsampling_y;
  info->frame_order_hint = (int)cm->current_frame.order_hint;

  RefCntBuffer *refbuf = get_ref_frame_buf(cm, xd->mi[0]->ref_frame[0]);
  assert(refbuf != NULL);
  info->ref_q = refbuf->base_qindex;
  info->base_q = cm->base_qindex;
  info->lambda = av1_compute_rd_mult_based_on_qindex(cpi, cm->base_qindex);
  assert(info->lambda > 0);
  assert(info->lambda < 10 * 1000 * 1000);

  copy_source_image(info, x);
  copy_intrapred_lshape(info, x);
  copy_interpred(info, cpi, x);
  info->prediction_type = is_interintra_pred(mbmi) ? 2 : 1;
  copy_predictor(info, cpi, x, bsize);
  info->interintra_bias = av1_interintra_bias();
  return info;
}

void write_uint8_and_advance(uint8_t **buf, uint8_t b) {
  **buf = b;
  (*buf)++;
}

void write_int32_and_advance(uint8_t **buf, int32_t value) {
  assert(value >= 0);
  for (size_t i = 0; i < sizeof(value); ++i) {
    // Little-endian order.
    uint8_t byte = 0xff & value;
    write_uint8_and_advance(buf, byte);
    value >>= 8;
  }
}

void write_buffer_and_advance(uint8_t **buf, uint8_t *src, size_t size,
                              int bitdepth) {
  if (bitdepth == 8) {
    for (size_t i = 0; i < size; ++i) {
      write_uint8_and_advance(buf, src[i]);
    }
    return;
  }

  uint16_t *src16 = (uint16_t *)src;
  for (size_t i = 0; i < size; ++i) {
    // Little-endian order.
    uint8_t b1 = (uint8_t)(0xff & src16[i]);
    uint8_t b2 = (uint8_t)(0xff & (src16[i] >> 8));
    write_uint8_and_advance(buf, b1);
    write_uint8_and_advance(buf, b2);
  }
}

void av1_interintra_ml_data_collect_serialize(IIMLPlaneInfo *info,
                                              uint8_t **buf, size_t *buf_size) {
  const int bytes_per_pixel = info->bitdepth == 8 ? 1 : 2;
  const size_t source_size = info->width * info->height;
  const size_t intrapred_lshape_size =
      ((info->border + info->width) * info->border +
       info->border * info->height);
  const size_t interpred_size =
      ((info->border + info->width) * (info->border + info->height));
  *buf_size =
      1 /* width */ + 1 /* height */ + 1 /* plane */ + 1 /* bitdepth */ +
      1 /* border */ + sizeof(int32_t) /* x */ + sizeof(int32_t) /* y */ +
      sizeof(int32_t) /* frame order hint */ + sizeof(int32_t) /* lambda */ +
      1 /* base_q */ + bytes_per_pixel * source_size /* source image */ +
      bytes_per_pixel * intrapred_lshape_size /* reconstruction border */ +
      1 /* prediction_type */ + bytes_per_pixel * source_size /* predictor */ +
      1 /* ref_q */ +
      bytes_per_pixel * interpred_size /* inter-predictor + border */ +
      1 /* interintra_bias */;
  *buf = malloc(*buf_size);
  assert(*buf != NULL);
  uint8_t *start = *buf;

  write_uint8_and_advance(buf, info->width);
  write_uint8_and_advance(buf, info->height);
  write_uint8_and_advance(buf, info->plane);
  write_uint8_and_advance(buf, info->bitdepth);
  write_uint8_and_advance(buf, info->border);
  write_int32_and_advance(buf, info->x);
  write_int32_and_advance(buf, info->y);
  write_int32_and_advance(buf, info->frame_order_hint);
  write_int32_and_advance(buf, info->lambda);
  write_uint8_and_advance(buf, info->base_q);
  write_buffer_and_advance(buf, info->source_image, source_size,
                           info->bitdepth);
  write_buffer_and_advance(buf, info->intrapred_lshape, intrapred_lshape_size,
                           info->bitdepth);
  write_uint8_and_advance(buf, info->prediction_type);
  write_buffer_and_advance(buf, info->predictor, source_size, info->bitdepth);
  write_uint8_and_advance(buf, info->ref_q);
  write_buffer_and_advance(buf, info->interpred, interpred_size,
                           info->bitdepth);
  write_uint8_and_advance(buf, info->interintra_bias);
  assert(start + *buf_size == *buf);
  *buf = start;
}

void av1_interintra_ml_data_collect(const AV1_COMP *const cpi,
                                    MACROBLOCK *const x, BLOCK_SIZE bsize) {
  init_first_run();
  assert(INFO_Y == NULL);
  assert(INFO_U == NULL);
  assert(INFO_V == NULL);
  INFO_Y = init_plane_info(cpi, x, bsize, AOM_PLANE_Y);
  assert(INFO_Y != NULL);

  // Sometimes the chroma is derived from the luma. Check if chroma is
  // available.
  MACROBLOCKD *const xd = &x->e_mbd;
  if (xd->mi[0]->chroma_ref_info.is_chroma_ref) {
    INFO_U = init_plane_info(cpi, x, bsize, AOM_PLANE_U);
    INFO_V = init_plane_info(cpi, x, bsize, AOM_PLANE_V);
    assert(INFO_U != NULL);
    assert(INFO_V != NULL);
  }
}

// Write out the three planes to disk.
void av1_interintra_ml_data_collect_finalize() {
  uint8_t *buf;
  size_t buf_size;

  // If luma is not present, then neither is chroma.
  if (INFO_Y == NULL) {
    assert(INFO_U == NULL);
    assert(INFO_V == NULL);
    return;
  }
  av1_interintra_ml_data_collect_serialize(INFO_Y, &buf, &buf_size);
  size_t written = fwrite(buf, sizeof(*buf), buf_size, FP);
  assert(written == buf_size);
  (void)written;
  free(buf);

  if (INFO_U != NULL) {
    av1_interintra_ml_data_collect_serialize(INFO_U, &buf, &buf_size);
    written = fwrite(buf, sizeof(*buf), buf_size, FP);
    assert(written == buf_size);
    free(buf);
  }

  if (INFO_V != NULL) {
    av1_interintra_ml_data_collect_serialize(INFO_V, &buf, &buf_size);
    written = fwrite(buf, sizeof(*buf), buf_size, FP);
    assert(written == buf_size);
    free(buf);
  }

  av1_interintra_ml_data_collect_abandon();
  assert(INFO_Y == NULL);
  assert(INFO_U == NULL);
  assert(INFO_V == NULL);
}

static void destroy_info(IIMLPlaneInfo *info) {
  if (info != NULL) {
    free(info->source_image);
    free(info->intrapred_lshape);
    free(info->interpred);
    free(info);
  }
}

void av1_interintra_ml_data_collect_abandon() {
  if (INFO_Y == NULL) {
    assert(INFO_U == NULL);
    assert(INFO_V == NULL);
    return;
  }
  destroy_info(INFO_Y);
  destroy_info(INFO_U);
  destroy_info(INFO_V);
  INFO_Y = NULL;
  INFO_U = NULL;
  INFO_V = NULL;
}

static int calc_intra_border(MACROBLOCK *const x, BLOCK_SIZE bsize) {
  MACROBLOCKD *const xd = &x->e_mbd;
  int border = BORDER_SIZE;
  // For each plane, check how much is available on the top and left,
  // shrinking the border if needed.
  for (int plane = AOM_PLANE_Y; plane <= AOM_PLANE_V; ++plane) {
    border = AOMMIN(border, av1_intra_top_available(xd, plane));
    border = AOMMIN(border, av1_intra_left_available(xd, plane));

    // If either the bottom or the right side of the block is partially
    // unavailable, set the entire border to 0 -- we cannot extract the
    // full border.
    const struct macroblockd_plane *const pd = &xd->plane[plane];
    const BLOCK_SIZE plane_bsize =
        get_plane_block_size(bsize, pd->subsampling_x, pd->subsampling_y);
    const TX_SIZE tx_size = max_txsize_rect_lookup[plane_bsize];
    if (av1_intra_bottom_unavailable(xd, plane, tx_size) ||
        av1_intra_right_unavailable(xd, plane, tx_size)) {
      return 0;
    }
  }
  return border;
}

bool av1_interintra_ml_data_collect_valid(MACROBLOCK *const x,
                                          BLOCK_SIZE bsize) {
  MACROBLOCKD *const xd = &x->e_mbd;
  const int inter_border =
      av1_calc_border(xd, AOM_PLANE_Y, false /* build for obmc */);
  const int intra_border = calc_intra_border(x, bsize);
  // Only process 8x8 or larger blocks -- if a dimension is 4, then the
  // chroma dimension can be 2, which has a different packing structure.
  return inter_border >= BORDER_SIZE && intra_border >= BORDER_SIZE &&
         !x->skip && block_size_wide[bsize] >= 8 && block_size_high[bsize] >= 8;
}
