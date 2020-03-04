#include <assert.h>
#include <unordered_map>

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
