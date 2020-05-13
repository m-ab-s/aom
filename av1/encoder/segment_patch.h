#ifndef AOM_AV1_ENCODER_SEGMENT_PATCH_H
#define AOM_AV1_ENCODER_SEGMENT_PATCH_H

#ifdef __cplusplus
extern "C" {
#endif

#include "aom/aom_integer.h"

// Struct for parameters related to segmentation.
typedef struct {
  float sigma;   //  Parameter used for gaussian smoothing of input image.
  float k;       //  Threshold: larger values result in larger components.
  int min_size;  // Minimum component size enforced by post-processing.
} Av1SegmentParams;

// Get reasonable defaults for segmentation parameters.
void av1_get_default_segment_params(Av1SegmentParams *params);

// Get image segments.
// Inputs:
// - input: input image (Y plane)
// - width: input image width
// - height: input image height
// - stride: input image stride
// - seg_params: segmentation parameters
// Outputs:
// - output: segmented output image (Y plane)
// - num_components: Number of connected components (segments) in the output.
// Note: Assumes output array is already allocated with size = width * height.
void av1_get_segments(const uint8_t *input, int width, int height, int stride,
                      const Av1SegmentParams *seg_params, uint8_t *output,
                      int *num_components);

// Amend mask with values {0,1} to one with values {0,64}.
// Input/output:
// - mask: Binary mask that is modified in-place.
// Inputs:
// - w: mask width and also stride
// - h: mask height
void av1_extend_binary_mask_range(uint8_t *const mask, int w, int h);

// Applies box blur on 'mask' using an averaging filter.
// Input/output:
// - mask: Binary mask with extended range that is modified in-place.
// Inputs:
// - w: mask width and also stride
// - h: mask height
void av1_apply_box_blur(uint8_t *const mask, int w, int h);

#define DUMP_SEGMENT_MASKS 0

#if DUMP_SEGMENT_MASKS
// Dump raw Y plane to a YUV file.
// Can be viewed as follows, for example:
// ffplay -f rawvideo -pixel_format gray -video_size wxh -i <filename>
extern "C" void av1_dump_raw_y_plane(const uint8_t *y, int width, int height,
                                     int stride, const char *filename);
#endif  // DUMP_SEGMENT_MASKS

#ifdef __cplusplus
}
#endif

#endif  // AOM_AV1_ENCODER_SEGMENT_PATCH_H
