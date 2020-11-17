#ifndef AOM_AV1_ENCODER_INTERINTRA_ML_DATA_COLLECT_H_
#define AOM_AV1_ENCODER_INTERINTRA_ML_DATA_COLLECT_H_

#include <stdint.h>
#include "av1/encoder/block.h"
#include "av1/encoder/encoder.h"

#ifdef __cplusplus
extern "C" {
#endif

// Data collection for the interintra ML training data. The program must be
// run in single-threaded mode.

// Build the inter-predictor (with border), and record it, along with the
// intra-predictor and source image. Note that only a subset of the border
// may end up being recorded.
void av1_interintra_ml_data_collect(const AV1_COMP *const cpi,
                                    MACROBLOCK *const x, BLOCK_SIZE bsize);

// After the data is collected, it is not immediately written out, since the
// encoder may later decide to save it as a skip block. This function
// indicates that the data is safe to write out (not a skip block).
void av1_interintra_ml_data_collect_finalize();

// The block ended up being skipped. Abandon the previously collected data.
void av1_interintra_ml_data_collect_abandon();

// Is this block a valid block for collecting interintra ML data?
bool av1_interintra_ml_data_collect_valid(MACROBLOCK *const x,
                                          BLOCK_SIZE bsize);

// RD-cost of inter-intra modes should be multiplied by this number (and divided
// by 100) to bias the results into picking inter-intra modes more often.
uint8_t av1_interintra_bias();

// These functions / data structures are internal implementation details, but
// are exposed for testing.

// The data structure encoding the information we wish to write out,
// for a given plane.
typedef struct IIMLPlaneInfo {
  // Width and height of the block.
  uint8_t width;
  uint8_t height;
  // Plane, one of AOM_PLANE_Y, AOM_PLANE_U, AOM_PLANE_V.
  uint8_t plane;
  // Number of bits per pixel, either 8, 10, or 12.
  uint8_t bitdepth;
  // The border width recorded for this block.
  uint8_t border;
  // The location of this block (in pixels) in the video.
  int x;
  int y;
  int frame_order_hint;
  // Information about the rate-cost calculation.
  int32_t lambda;
  uint8_t ref_q;
  uint8_t base_q;

  // The original source block. Size is width * height. If bit-depth
  // is 10 or 12, it is actually a uint16_t pointer.
  uint8_t *source_image;

  // The top (border + width) * (border) pixels, followed by
  // the (border * height) left pixels, in raster scan order, of
  // the top-left border of the intra-predictor. If bit-depth is
  // 10 or 12, actually a uint16_t pointer.
  uint8_t *intrapred_lshape;

  // The (border + width) * (border + height) inter-predictor.
  // The inter-predictor starts at the (width * border + border) offset.
  // If bit-depth is 10 or 12, actually a uint16_t pointer.
  uint8_t *interpred;
  // 0 == intra, 1 == inter, 2 == inter-intra; currently only inter and
  // inter-intra are output.
  uint8_t prediction_type;
  // The (width * height) final predictor. If bit-depth is 10 or 12, actually
  // a uint16_t pointer.
  uint8_t *predictor;
  // If an inter-intra-predictor, this value represents a bias -- the percent
  // that the RD-cost for inter-intra-predictors was shifted down by, to
  // bias the data into using the inter-intra predictor more often. E.g.,
  // 90 means that the RD-cost is multiplied by 0.9, 80 means it is multiplied
  // by 0.8, etc.
  uint8_t interintra_bias;
} IIMLPlaneInfo;

// Serialize the plane info data structure into a byte array. Allocates
// memory into buf (the caller must free it afterward).
void av1_interintra_ml_data_collect_serialize(IIMLPlaneInfo *info,
                                              uint8_t **buf, size_t *buf_size);

// Copy over the top and left pixels (in an L shape) from the top-left
// corner, into the buffer. Note that source points to the start of the
// intrapred block (the border is at a negative offset).
void av1_interintra_ml_data_collect_copy_intrapred_lshape(
    uint8_t *dst, const uint8_t *src, int src_stride, int width, int height,
    int border, int bitdepth, bool is_src_hbd);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_ENCODER_INTERINTRA_ML_DATA_COLLECT_H_
