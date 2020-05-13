#include <assert.h>
#include <unordered_map>

#include "aom_mem/aom_mem.h"
#include "av1/encoder/segment_patch.h"
#include "third_party/segment/segment-image.h"

using std::unordered_map;

extern "C" void av1_get_default_segment_params(Av1SegmentParams *params) {
  params->sigma = 0.5;
  params->k = 500;
  params->min_size = 50;
}

// Convert Y image to RGB image by copying Y channel to all 3 RGB channels.
static image<rgb> *y_to_rgb(const uint8_t *const input, int width, int height,
                            int stride) {
  image<rgb> *output = new image<rgb>(width, height, false);

  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      const int y_channel = input[j * stride + i];
      imRef(output, i, j).r = y_channel;
      imRef(output, i, j).g = y_channel;
      imRef(output, i, j).b = y_channel;
    }
  }
  return output;
}

// Convert RGB image to an image with segment indices, by picking a segment
// number for each unique RGB color.
void rgb_to_segment_index(const image<rgb> *input, uint8_t *const output) {
  const int width = input->width();
  const int height = input->height();
  const int stride = width;
  // Hash-map for RGB color to an index.
  unordered_map<uint32_t, uint8_t> color_to_idx;
  int next_idx = 0;
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      const unsigned int r_channel = imRef(input, i, j).r;
      const unsigned int g_channel = imRef(input, i, j).g;
      const unsigned int b_channel = imRef(input, i, j).b;
      const uint32_t color = (r_channel << 16) + (g_channel << 8) + b_channel;
      if (!color_to_idx.count(color)) {
        // TODO(urvang): Return error if this happens?
        assert(next_idx < 256);
        color_to_idx[color] = next_idx;
        ++next_idx;
      }
      output[j * stride + i] = color_to_idx[color];
    }
  }
}

extern "C" void av1_get_segments(const uint8_t *input, int width, int height,
                                 int stride, const Av1SegmentParams *seg_params,
                                 uint8_t *output, int *num_components) {
  image<rgb> *input_rgb = y_to_rgb(input, width, height, stride);
  image<rgb> *output_rgb =
      segment_image(input_rgb, seg_params->sigma, seg_params->k,
                    seg_params->min_size, num_components);
  rgb_to_segment_index(output_rgb, output);
}

// Amend mask with values {0,1} to one with values {0,64}.
extern "C" void av1_extend_binary_mask_range(uint8_t *const mask, int w,
                                             int h) {
  for (int r = 0; r < h; ++r) {
    for (int c = 0; c < w; ++c) {
      const int idx = r * w + c;
      if (mask[idx] == 1) mask[idx] = 64;
    }
  }
}

#define BLUR_KERNEL 7  // Box blur kernel size
#define BLUR_HALF_KERNEL ((BLUR_KERNEL - 1) / 2)
#define BLUR_BORDER BLUR_HALF_KERNEL  // Padding needed in each direction.
extern "C" void av1_apply_box_blur(uint8_t *const mask, int w, int h) {
  // Pad as needed in each of the 4 directions.
  const int padded_w = (w + BLUR_BORDER * 2);
  const int padded_h = (h + BLUR_BORDER * 2);
  uint8_t *const input_mem =
      (uint8_t *)aom_malloc(padded_w * padded_h * sizeof(*input_mem));
  uint8_t *const input = input_mem + padded_w * BLUR_BORDER;
  for (int r = 0; r < h; ++r) {
    const uint8_t *const src = mask + r * w;
    uint8_t *const dst = input + r * padded_w + BLUR_BORDER;
    memcpy(dst, src, w * sizeof(*mask));
    for (int c = -BLUR_BORDER; c < 0; ++c) {
      dst[c] = dst[0];
    }
    for (int c = w; c < w + BLUR_BORDER; ++c) {
      dst[c] = dst[w - 1];
    }
  }
  for (int r = -BLUR_BORDER; r < 0; ++r) {
    memcpy(&input[r * padded_w], input, padded_w * sizeof(*input));
  }
  for (int r = h; r < h + BLUR_BORDER; ++r) {
    memcpy(&input[r * padded_w], &input[(h - 1) * padded_w],
           padded_w * sizeof(*input));
  }

  // 1D filter in horizontal direction.
  double *const temp_mem =
      (double *)aom_malloc(w * padded_h * sizeof(*temp_mem));
  double *const temp = temp_mem + w * BLUR_BORDER;
  for (int r = -BLUR_BORDER; r < h + BLUR_BORDER; ++r) {
    const uint8_t *const src = input + r * padded_w + BLUR_BORDER;
    double *const dst = temp + r * w;
    // Simple average computation for 0th column.
    double sum = 0;
    for (int c = -BLUR_HALF_KERNEL; c <= BLUR_HALF_KERNEL; ++c) {
      sum += src[c];
    }
    dst[0] = sum / BLUR_KERNEL;
    // Intelligent average computation for rest of the columns.
    for (int c = 1; c < w; ++c) {
      sum -= src[c - BLUR_HALF_KERNEL - 1];
      sum += src[c + BLUR_HALF_KERNEL];
      dst[c] = sum / BLUR_KERNEL;
    }
  }
  aom_free(input_mem);

  // 1D filter in vertical direction.
  for (int c = 0; c < w; ++c) {
    const double *const src = temp + c;
    uint8_t *const dst = mask + c;
    // Simple average computation for 0th row.
    double sum = 0;
    for (int r = -BLUR_HALF_KERNEL; r <= BLUR_HALF_KERNEL; ++r) {
      sum += src[r * w];
    }
    dst[0] = (uint8_t)round(sum / BLUR_KERNEL);
    // Intelligent average computation for rest of the rows.
    for (int r = 1; r < h; ++r) {
      sum -= src[(r - BLUR_HALF_KERNEL - 1) * w];
      sum += src[(r + BLUR_HALF_KERNEL) * w];
      dst[r * w] = (uint8_t)round(sum / BLUR_KERNEL);
    }
  }
  aom_free(temp_mem);
}
#undef BLUR_KERNEL
#undef BLUR_HALF_KERNEL
#undef BLUR_BORDER

#if DUMP_SEGMENT_MASKS
extern "C" void av1_dump_raw_y_plane(const uint8_t *y, int width, int height,
                                     int stride, const char *filename) {
  FILE *f_out = fopen(filename, "wb");
  if (f_out == NULL) {
    fprintf(stderr, "Unable to open file %s to write.\n", filename);
    return;
  }

  for (int r = 0; r < height; ++r) {
    fwrite(&y[r * stride], sizeof(*y), width, f_out);
  }

  fclose(f_out);
}
#endif  // DUMP_SEGMENT_MASKS
