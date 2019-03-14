/*
 * Copyright (c) 2019, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <assert.h>
#include <math.h>

#include "aom_dsp/aom_dsp_common.h"
#include "av1/common/cnn.h"
#include "av1/common/onyxc_int.h"

#define CLAMPINDEX(a, hi) ((a) < 0 ? 0 : ((a) >= (hi) ? ((hi)-1) : (a)))

void av1_cnn_convolve_c(const float **input, int in_width, int in_height,
                        int in_stride, const CNN_LAYER_CONFIG *layer_config,
                        float **output, int out_stride) {
  assert(layer_config->filter_height & 1);
  assert(layer_config->filter_width & 1);
  const int cstep = layer_config->in_channels * layer_config->out_channels;

  const int filter_height_half = layer_config->filter_height >> 1;
  const int filter_width_half = layer_config->filter_width >> 1;

  float (*activation)(float) = identity;
  switch (layer_config->activation) {
    case RELU: activation = relu; break;
    case SOFTSIGN: activation = softsign; break;
  }

  switch (layer_config->pad) {
    case PADDING_SAME_ZERO:
      for (int i = 0; i < layer_config->out_channels; ++i) {
        for (int h = 0, u = 0; h < in_height;
             h += layer_config->skip_height, ++u) {
          for (int w = 0, v = 0; w < in_width;
               w += layer_config->skip_width, ++v) {
            float sum = layer_config->bias[i];
            for (int k = 0; k < layer_config->in_channels; ++k) {
              int off = k * layer_config->out_channels + i;
              for (int l = 0; l < layer_config->filter_height; ++l) {
                const int ii = h + l - filter_height_half;
                for (int m = 0; m < layer_config->filter_width;
                     ++m, off += cstep) {
                  const int jj = w + m - filter_width_half;
                  if (ii < 0 || ii >= in_height || jj < 0 || jj >= in_width)
                    continue;
                  sum += layer_config->weights[off] *
                         input[k][ii * in_stride + jj];
                }
              }
            }
            output[i][u * out_stride + v] = activation(sum);
          }
        }
      }
      break;
    case PADDING_SAME_REPLICATE:
      for (int i = 0; i < layer_config->out_channels; ++i) {
        for (int h = 0, u = 0; h < in_height;
             h += layer_config->skip_height, ++u) {
          for (int w = 0, v = 0; w < in_width;
               w += layer_config->skip_width, ++v) {
            float sum = layer_config->bias[i];
            for (int k = 0; k < layer_config->in_channels; ++k) {
              int off = k * layer_config->out_channels + i;
              for (int l = 0; l < layer_config->filter_height; ++l) {
                const int ii =
                    CLAMPINDEX(h + l - filter_height_half, in_height);
                for (int m = 0; m < layer_config->filter_width; ++m) {
                  const int jj =
                      CLAMPINDEX(w + m - filter_width_half, in_width);
                  sum += layer_config->weights[off] *
                         input[k][ii * in_stride + jj];
                  off += cstep;
                }
              }
            }
            output[i][u * out_stride + v] = activation(sum);
          }
        }
      }
      break;
    case PADDING_VALID:
      for (int i = 0; i < layer_config->out_channels; ++i) {
        for (int h = filter_height_half, u = 0;
             h <
             in_height - layer_config->filter_height + filter_height_half + 1;
             h += layer_config->skip_height, ++u) {
          for (int w = filter_width_half, v = 0;
               w <
               in_width - layer_config->filter_width + filter_width_half + 1;
               w += layer_config->skip_width, ++v) {
            float sum = layer_config->bias[i];
            for (int k = 0; k < layer_config->in_channels; ++k) {
              int off = k * layer_config->out_channels + i;
              for (int l = 0; l < layer_config->filter_height; ++l) {
                const int ii = h + l - filter_height_half;
                for (int m = 0; m < layer_config->filter_width;
                     ++m, off += cstep) {
                  const int jj = w + m - filter_width_half;
                  assert(ii >= 0 && ii < in_height && jj >= 0 && jj < in_width);
                  sum += layer_config->weights[off] *
                         input[k][ii * in_stride + jj];
                }
              }
            }
            output[i][u * out_stride + v] = activation(sum);
          }
        }
      }
      break;
    default: assert(0 && "Unknown padding type");
  }
}

static void find_layer_output_size(int in_width, int in_height,
                                   const CNN_LAYER_CONFIG *layer_config,
                                   int *out_width, int *out_height) {
  switch (layer_config->pad) {
    case PADDING_SAME_ZERO:
    case PADDING_SAME_REPLICATE:
      *out_width =
          (in_width + layer_config->skip_width - 1) / layer_config->skip_width;
      *out_height = (in_height + layer_config->skip_height - 1) /
                    layer_config->skip_height;
      break;
    case PADDING_VALID:
      *out_width =
          (in_width - layer_config->filter_width + layer_config->skip_width) /
          layer_config->skip_width;
      *out_height = (in_height - layer_config->filter_height +
                     layer_config->skip_height) /
                    layer_config->skip_height;
      break;
    default: assert(0 && "Unknown padding type");
  }
}

static void find_cnn_output_size(int in_width, int in_height,
                                 const CNN_CONFIG *cnn_config, int *out_width,
                                 int *out_height) {
  int i_width = in_width + cnn_config->ext_width * 2;
  int i_height = in_height + cnn_config->ext_height * 2;
  for (int i = 0; i < cnn_config->num_layers; ++i) {
    int o_width = 0, o_height = 0;
    find_layer_output_size(i_width, i_height, &cnn_config->layer_config[i],
                           &o_width, &o_height);
    i_width = o_width;
    i_height = o_height;
  }
  *out_width = i_width;
  *out_height = i_height;
}

void av1_cnn_predict_c(const float *input, int in_width, int in_height,
                       int in_stride, const CNN_CONFIG *cnn_config,
                       float *output, int out_stride) {
  const float *buf1[CNN_MAX_CHANNELS] = { 0 };
  float *buf2[CNN_MAX_CHANNELS] = { 0 };
  int allocsize1 = 0, allocsize2 = 0;
  int i_width, i_height, i_stride;
  int o_width = 0, o_height = 0, o_stride;

  for (int layer = 0; layer < cnn_config->num_layers; ++layer) {
    if (layer == 0) {  // First layer
      buf1[0] = input;
      assert(cnn_config->layer_config[layer].in_channels == 1);
      if (cnn_config->num_layers == 1) {  // single layer case
        assert(cnn_config->layer_config[layer].out_channels == 1);
        o_stride = out_stride;
        buf2[0] = output;
      } else {  // more than one layer case
        find_layer_output_size(in_width, in_height,
                               &cnn_config->layer_config[layer], &o_width,
                               &o_height);
        o_stride = o_width;
        allocsize2 =
            o_width * o_height * cnn_config->layer_config[layer].out_channels;
        buf2[0] = (float *)aom_malloc(allocsize2 * sizeof(*buf2[0]));
        for (int c = 1; c < cnn_config->layer_config[layer].out_channels; ++c)
          buf2[c] = &buf2[0][c * o_width * o_height];
      }
      av1_cnn_convolve(buf1, in_width, in_height, in_stride,
                       &cnn_config->layer_config[layer], buf2, o_stride);
      buf1[0] = NULL;
      if (buf2[0] == output) buf2[0] = NULL;
    } else {  // Non-first layer
      assert(cnn_config->layer_config[layer].in_channels ==
             cnn_config->layer_config[layer - 1].out_channels);

      // Swap the buf1 and buf2 and also the corresponding allocsizes.
      float *tmp_buf1 = (float *)buf1[0];
      memcpy(buf1, buf2,
             sizeof(*buf1) * cnn_config->layer_config[layer - 1].out_channels);
      buf2[0] = tmp_buf1;
      int tmpallocsize = allocsize1;
      allocsize1 = allocsize2;
      allocsize2 = tmpallocsize;

      i_width = o_width;
      i_height = o_height;
      i_stride = o_stride;
      if (layer < cnn_config->num_layers - 1) {  // Non-last layer
        find_layer_output_size(i_width, i_height,
                               &cnn_config->layer_config[layer], &o_width,
                               &o_height);
        o_stride = o_width;
        const int newallocsize =
            o_width * o_height * cnn_config->layer_config[layer].out_channels;
        // free and reallocate if the previous allocsize is smaller than
        // needed
        if (allocsize2 < newallocsize) {
          aom_free(buf2[0]);
          buf2[0] = (float *)aom_malloc(newallocsize * sizeof(*buf2[0]));
          allocsize2 = newallocsize;
        }
        for (int c = 1; c < cnn_config->layer_config[layer].out_channels; ++c)
          buf2[c] = &buf2[0][c * o_width * o_height];
      } else {  // Last layer
        assert(cnn_config->layer_config[layer].out_channels == 1);
        o_stride = out_stride;
        aom_free(buf2[0]);
        allocsize2 = 0;
        buf2[0] = output;
      }
      av1_cnn_convolve(buf1, i_width, i_height, i_stride,
                       &cnn_config->layer_config[layer], buf2, o_stride);
      // free for last layer
      if (layer == cnn_config->num_layers - 1) aom_free((void *)buf1[0]);
    }
  }
}

void av1_restore_cnn(uint8_t *dgd, int width, int height, int stride,
                     const CNN_CONFIG *cnn_config) {
  const float max_val = 255.0;
  int out_width, out_height;
  find_cnn_output_size(width, height, cnn_config, &out_width, &out_height);
  assert(out_width == width);
  assert(out_height == height);

  int in_width = width + 2 * cnn_config->ext_width;
  int in_height = height + 2 * cnn_config->ext_height;
  float *input_ = (float *)aom_malloc(in_width * in_height * sizeof(*input_));
  const int in_stride = in_width;
  float *input =
      input_ + cnn_config->ext_height * in_stride + cnn_config->ext_width;

  float *output = (float *)aom_malloc(width * height * sizeof(*output));
  const int out_stride = width;
  if (cnn_config->strict_bounds) {
    for (int i = 0; i < height; ++i)
      for (int j = 0; j < width; ++j)
        input[i * in_stride + j] = (float)dgd[i * stride + j] / max_val;
    // extend left and right
    for (int i = 0; i < height; ++i) {
      for (int j = -cnn_config->ext_width; j < 0; ++j)
        input[i * in_stride + j] = input[i * in_stride];
      for (int j = width; j < width + cnn_config->ext_width; ++j)
        input[i * in_stride + j] = input[i * in_stride + width - 1];
    }
    // extend top and bottom
    for (int i = -cnn_config->ext_height; i < 0; ++i)
      memcpy(&input[i * in_stride - cnn_config->ext_width],
             &input[-cnn_config->ext_width], in_width * sizeof(*input));
    for (int i = height; i < height + cnn_config->ext_height; ++i)
      memcpy(&input[i * in_stride - cnn_config->ext_width],
             &input[(height - 1) * in_stride - cnn_config->ext_width],
             in_width * sizeof(*input));
  } else {
    for (int i = -cnn_config->ext_height; i < height + cnn_config->ext_height;
         ++i)
      for (int j = -cnn_config->ext_width; j < width + cnn_config->ext_width;
           ++j)
        input[i * in_stride + j] = (float)dgd[i * stride + j] / max_val;
  }

  av1_cnn_predict(input_, in_width, in_height, in_stride, cnn_config, output,
                  out_stride);

  if (cnn_config->is_residue) {
    for (int i = 0; i < height; ++i)
      for (int j = 0; j < width; ++j) {
        const int residue = (int)(output[i * out_stride + j] * max_val + 0.5);
        dgd[i * stride + j] = clip_pixel(dgd[i * stride + j] + residue);
      }
  } else {
    for (int i = 0; i < height; ++i)
      for (int j = 0; j < width; ++j)
        dgd[i * stride + j] =
            clip_pixel((int)(output[i * out_stride + j] * max_val + 0.5));
  }

  aom_free(input_);
  aom_free(output);
}

void av1_restore_cnn_highbd(uint16_t *dgd, int width, int height, int stride,
                            const CNN_CONFIG *cnn_config, int bit_depth) {
  const float max_val = (float)((1 << bit_depth) - 1);
  int out_width, out_height;
  find_cnn_output_size(width, height, cnn_config, &out_width, &out_height);
  assert(out_width == width);
  assert(out_height == height);

  int in_width = width + 2 * cnn_config->ext_width;
  int in_height = height + 2 * cnn_config->ext_height;
  float *input_ = (float *)aom_malloc(in_width * in_height * sizeof(*input_));
  const int in_stride = in_width;
  float *input =
      input_ + cnn_config->ext_height * in_stride + cnn_config->ext_width;

  float *output = (float *)aom_malloc(width * height * sizeof(*output));
  const int out_stride = width;
  if (cnn_config->strict_bounds) {
    for (int i = 0; i < height; ++i)
      for (int j = 0; j < width; ++j)
        input[i * in_stride + j] = (float)dgd[i * stride + j] / max_val;
    // extend left and right
    for (int i = 0; i < height; ++i) {
      for (int j = -cnn_config->ext_width; j < 0; ++j)
        input[i * in_stride + j] = input[i * in_stride];
      for (int j = width; j < width + cnn_config->ext_width; ++j)
        input[i * in_stride + j] = input[i * in_stride + width - 1];
    }
    // extend top and bottom
    for (int i = -cnn_config->ext_height; i < 0; ++i)
      memcpy(&input[i * in_stride - cnn_config->ext_width],
             &input[-cnn_config->ext_width], in_width * sizeof(*input));
    for (int i = height; i < height + cnn_config->ext_height; ++i)
      memcpy(&input[i * in_stride - cnn_config->ext_width],
             &input[(height - 1) * in_stride - cnn_config->ext_width],
             in_width * sizeof(*input));
  } else {
    for (int i = -cnn_config->ext_height; i < height + cnn_config->ext_height;
         ++i)
      for (int j = -cnn_config->ext_width; j < width + cnn_config->ext_width;
           ++j)
        input[i * in_stride + j] = (float)dgd[i * stride + j] / max_val;
  }

  av1_cnn_predict(input, width, height, in_stride, cnn_config, output,
                  out_stride);

  if (cnn_config->is_residue) {
    for (int i = 0; i < height; ++i)
      for (int j = 0; j < width; ++j) {
        const int residue = (int)(output[i * out_stride + j] * max_val + 0.5);
        dgd[i * stride + j] +=
            clip_pixel_highbd(dgd[i * stride + j] + residue, bit_depth);
      }
  } else {
    for (int i = 0; i < height; ++i)
      for (int j = 0; j < width; ++j)
        dgd[i * stride + j] = clip_pixel_highbd(
            (int)(output[i * out_stride + j] * max_val + 0.5), bit_depth);
  }

  aom_free(input_);
  aom_free(output);
}

void av1_restore_cnn_plane(AV1_COMMON *cm, const CNN_CONFIG *cnn_config,
                           int plane) {
  YV12_BUFFER_CONFIG *buf = &cm->cur_frame->buf;
  if (cm->seq_params.use_highbitdepth) {
    switch (plane) {
      case AOM_PLANE_Y:
        av1_restore_cnn_highbd(CONVERT_TO_SHORTPTR(buf->y_buffer),
                               buf->y_crop_width, buf->y_crop_height,
                               buf->y_stride, cnn_config,
                               cm->seq_params.bit_depth);
        break;
      case AOM_PLANE_U:
        av1_restore_cnn_highbd(CONVERT_TO_SHORTPTR(buf->u_buffer),
                               buf->uv_crop_width, buf->uv_crop_height,
                               buf->uv_stride, cnn_config,
                               cm->seq_params.bit_depth);
        break;
      case AOM_PLANE_V:
        av1_restore_cnn_highbd(CONVERT_TO_SHORTPTR(buf->v_buffer),
                               buf->uv_crop_width, buf->uv_crop_height,
                               buf->uv_stride, cnn_config,
                               cm->seq_params.bit_depth);
        break;
      default: assert(0 && "Invalid plane index");
    }
  } else {
    assert(cm->seq_params.bit_depth == 8);
    switch (plane) {
      case AOM_PLANE_Y:
        av1_restore_cnn(buf->y_buffer, buf->y_crop_width, buf->y_crop_height,
                        buf->y_stride, cnn_config);
        break;
      case AOM_PLANE_U:
        av1_restore_cnn(buf->u_buffer, buf->uv_crop_width, buf->uv_crop_height,
                        buf->uv_stride, cnn_config);
        break;
      case AOM_PLANE_V:
        av1_restore_cnn(buf->v_buffer, buf->uv_crop_width, buf->uv_crop_height,
                        buf->uv_stride, cnn_config);
        break;
      default: assert(0 && "Invalid plane index");
    }
  }
}
