/*
 * Copyright (c) 2020, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include "av1/common/mfqe.h"
#include "av1/common/resize.h"
#include "av1/encoder/rdopt.h"

#define MFQE_GAUSSIAN_BLOCK 32  // Size of blocks to apply Gaussian Blur.

// 8-tap Gaussian convolution filter with sigma = 1.3, sums to 128,
// all co-efficients must be even.
DECLARE_ALIGNED(16, static const int16_t, gauss_filter[8]) = { 2,  12, 30, 40,
                                                               30, 12, 2,  0 };

// Apply gaussian blur to the block in src and save the result to dst.
void gaussian_blur(const uint8_t *src, int src_stride, int w, int h,
                   uint8_t *dst, bool high_bd, int bd) {
  ConvolveParams conv_params = get_conv_params(0, 0, bd);
  InterpFilterParams filter = { .filter_ptr = gauss_filter,
                                .taps = 8,
                                .subpel_shifts = 0,
                                .interp_filter = EIGHTTAP_REGULAR };
  // Requirements from the vector-optimized implementations.
  assert(h % 4 == 0);
  assert(w % 8 == 0);
  // Because we use an eight tap filter, the stride should be at least 7 + w.
  assert(src_stride >= w + 7);
  if (high_bd) {
    av1_highbd_convolve_2d_sr(CONVERT_TO_SHORTPTR(src), src_stride,
                              CONVERT_TO_SHORTPTR(dst), src_stride, w, h,
                              &filter, &filter, 0, 0, &conv_params, bd);
  } else {
    av1_convolve_2d_sr(src, src_stride, dst, src_stride, w, h, &filter, &filter,
                       0, 0, &conv_params);
  }
}

// Apply Gaussian Blur on a single plane and store the result in destination.
static void mfqe_gaussian_blur(uint8_t *src, uint8_t *dst, int stride, int h,
                               int w, int high_bd, int bd) {
  for (int col = 0; col < h; col += MFQE_GAUSSIAN_BLOCK) {
    for (int row = 0; row < w; row += MFQE_GAUSSIAN_BLOCK) {
      int w_size = AOMMIN(MFQE_GAUSSIAN_BLOCK, w - row);
      int h_size = AOMMIN(MFQE_GAUSSIAN_BLOCK, h - col);
      uint8_t *src_ptr = src + col * stride + row;
      uint8_t *dst_ptr = dst + col * stride + row;

      // Gaussian blur can only operate on blocks with width % 8 == 0 and
      // height % 4 == 0. If there are smaller sized blocks, break out of loop.
      if (w_size % 8 != 0 || h_size % 4 != 0) break;
      gaussian_blur(src_ptr, stride, w_size, h_size, dst_ptr, high_bd, bd);
    }
  }
}

// Resize a single plane with the given resize factor and store in destination.
static void mfqe_resize_plane(const uint8_t *src, uint8_t *dst, int stride,
                              int h, int w, int resize_factor, int highbd,
                              int bd) {
  int dst_stride = stride * resize_factor;
  int dst_height = h * resize_factor;
  int dst_width = w * resize_factor;

  if (highbd)
    av1_highbd_resize_plane(src, h, w, stride, dst, dst_height, dst_width,
                            dst_stride, bd);
  else
    av1_resize_plane(src, h, w, stride, dst, dst_height, dst_width, dst_stride);
}

// Calcuate the sum of squared errors between two blocks in buffers.
static int64_t aom_sse_lowbd(const uint8_t *a, int a_stride, const uint8_t *b,
                             int b_stride, int width, int height) {
  int y, x;
  int64_t sse = 0;

  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
      const int32_t diff = abs(a[x] - b[x]);
      sse += diff * diff;
    }

    a += a_stride;
    b += b_stride;
  }
  return sse;
}

// Calcuate the sum of squared errors between two blocks in buffers.
static int64_t aom_sse_highbd(const uint8_t *a8, int a_stride,
                              const uint8_t *b8, int b_stride, int width,
                              int height) {
  int y, x;
  int64_t sse = 0;
  uint16_t *a = CONVERT_TO_SHORTPTR(a8);
  uint16_t *b = CONVERT_TO_SHORTPTR(b8);
  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
      const int32_t diff = (int32_t)(a[x]) - (int32_t)(b[x]);
      sse += diff * diff;
    }

    a += a_stride;
    b += b_stride;
  }
  return sse;
}

// Return the mean squared error between the given blocks in two buffers. If
// the row and column parameters are not valid indices, return MSE_MAX.
static double get_mse_block(Y_BUFFER_CONFIG buf1, Y_BUFFER_CONFIG buf2,
                            int16_t mb_row_1, int16_t mb_col_1,
                            int16_t mb_row_2, int16_t mb_col_2,
                            BLOCK_SIZE bsize, int highbd) {
  int block_w = block_size_wide[bsize];
  int block_h = block_size_high[bsize];

  // Check if rows and columns are valid, return MSE_MAX if not.
  if ((mb_row_1 < 0) || (mb_col_1 < 0) || (mb_row_2 < 0) || (mb_col_2 < 0) ||
      (mb_row_1 >= buf1.height - block_h) ||
      (mb_row_2 >= buf2.height - block_h) ||
      (mb_col_1 >= buf1.width - block_w) || (mb_col_2 >= buf2.width - block_w))
    return MSE_MAX;

  uint8_t *a = buf1.buffer + buf1.stride * mb_row_1 + mb_col_1;
  uint8_t *b = buf2.buffer + buf2.stride * mb_row_2 + mb_col_2;
  int64_t sse;

  if (highbd)
    sse = aom_sse_highbd(a, buf1.stride, b, buf2.stride, block_w, block_h);
  else
    sse = aom_sse_lowbd(a, buf1.stride, b, buf2.stride, block_w, block_h);

  // Divide the sum of squared errors by the number of pixels in the block.
  double mse = ((double)sse) / (block_w * block_h);
  return mse;
}

// Perform initial diamond search to obtain the full-pixel motion vector for
// every block in the current frame, going through the points specified by the
// grid_search_rows and grid_search_cols defined in the header file.
static void full_pix_diamond_search(MV_MFQE *mvr, int16_t mb_row,
                                    int16_t mb_col, Y_BUFFER_CONFIG cur,
                                    Y_BUFFER_CONFIG refs[], BLOCK_SIZE bsize,
                                    int highbd) {
  MV_MFQE mvr_best = *mvr;
  double this_mse;
  double best_mse = MSE_MAX;
  int block_w = block_size_wide[bsize];
  int block_h = block_size_high[bsize];

  for (int ref_index = 0; ref_index < MFQE_NUM_REFS; ++ref_index) {
    for (int point = 0; point < MFQE_N_GRID_SEARCH; ++point) {
      int16_t dr = grid_search_rows[point];
      int16_t dc = grid_search_cols[point];
      int16_t mb_col_ref = mb_col + mvr->mv.col + dc * block_w;
      int16_t mb_row_ref = mb_row + mvr->mv.row + dr * block_h;

      this_mse = get_mse_block(cur, refs[ref_index], mb_row, mb_col, mb_row_ref,
                               mb_col_ref, bsize, highbd);

      // Store the motion vector with lowest mean squared error.
      if (this_mse < best_mse) {
        best_mse = this_mse;
        mvr_best.mv.col = mvr->mv.col + dc * block_w;
        mvr_best.mv.row = mvr->mv.row + dr * block_h;
        mvr_best.ref_index = ref_index;
      }
    }
  }
  *mvr = mvr_best;
}

// Perform full pixel motion vector search in the low frequency version of the
// current frame and reference frames.
static void full_pixel_search(MV_MFQE *mvr, int16_t mb_row, int16_t mb_col,
                              Y_BUFFER_CONFIG cur, Y_BUFFER_CONFIG refs[],
                              BLOCK_SIZE bsize, int highbd) {
  MV_MFQE mvr_init = *mvr;
  // Perform diamond search to obtain the initial motion vector.
  full_pix_diamond_search(&mvr_init, mb_row, mb_col, cur, refs, bsize, highbd);

  mvr->valid = 0;
  mvr->ref_index = mvr_init.ref_index;

  int mb_row_ref;
  int mb_col_ref;
  int search_size_w = block_size_wide[bsize] / 2;
  int search_size_h = block_size_high[bsize] / 2;
  double this_mse;
  double best_mse = MSE_MAX;

  for (int dr = -search_size_h; dr <= search_size_h; ++dr) {
    for (int dc = -search_size_w; dc <= search_size_w; ++dc) {
      mb_row_ref = mb_row + mvr_init.mv.row + dr;
      mb_col_ref = mb_col + mvr_init.mv.col + dc;

      this_mse = get_mse_block(cur, refs[mvr_init.ref_index], mb_row, mb_col,
                               mb_row_ref, mb_col_ref, bsize, highbd);

      // The motion vector points to the block with lowest mean squared error.
      if (this_mse < best_mse && this_mse < MFQE_MSE_THRESHOLD) {
        best_mse = this_mse;
        mvr->valid = 1;
        mvr->mv.row = mvr_init.mv.row + dr;
        mvr->mv.col = mvr_init.mv.col + dc;
        mvr->alpha = get_alpha_weight(this_mse);
      }
    }
  }
}

static BLOCK_SIZE scale_block_4X4(int scale) {
  if (scale == 1) return BLOCK_4X4;
  if (scale == 2) return BLOCK_8X8;
  if (scale == 4) return BLOCK_16X16;
  if (scale == 8) return BLOCK_32X32;
  return BLOCK_4X4;
}

static BLOCK_SIZE scale_block_8X8(int scale) {
  if (scale == 1) return BLOCK_8X8;
  if (scale == 2) return BLOCK_16X16;
  if (scale == 4) return BLOCK_32X32;
  if (scale == 8) return BLOCK_64X64;
  return BLOCK_8X8;
}

static BLOCK_SIZE scale_block_16X16(int scale) {
  if (scale == 1) return BLOCK_16X16;
  if (scale == 2) return BLOCK_32X32;
  if (scale == 4) return BLOCK_64X64;
  if (scale == 8) return BLOCK_128X128;
  return BLOCK_16X16;
}

// Scale up the given BLOCK_SIZE by the resizing factor.
static BLOCK_SIZE scale_block_size(BLOCK_SIZE bsize, int scale) {
  // Currently, only supports 4X4, 8X8, and 16X16 blocks.
  assert(bsize == BLOCK_4X4 || bsize == BLOCK_8X8 || bsize == BLOCK_16X16);

  if (bsize == BLOCK_4X4) return scale_block_4X4(scale);
  if (bsize == BLOCK_8X8) return scale_block_8X8(scale);
  return scale_block_16X16(scale);
}

// Perform finer-grained motion vector search at subpel level, then save the
// updated motion vector in MV_MFQE.
static void sub_pixel_search(MV_MFQE *mvr, int16_t mb_row, int16_t mb_col,
                             Y_BUFFER_CONFIG cur, Y_BUFFER_CONFIG refs[],
                             BLOCK_SIZE bsize, int resize_factor, int highbd) {
  mb_row *= resize_factor;
  mb_col *= resize_factor;
  bsize = scale_block_size(bsize, resize_factor);

  int search_size = resize_factor / 2;
  int mb_row_ref;
  int mb_col_ref;

  double this_mse;
  double best_mse = MSE_MAX;
  int best_subpel_x_qn = 0;
  int best_subpel_y_qn = 0;

  // Search for the nearby search_size pixels in subpel accuracy, which is half
  // the size of the scale. For example, if the image is scaled by a factor of
  // 8, the algorithm will search the adjacent 4 pixels in the resized image.
  for (int dr = -search_size; dr <= search_size; ++dr) {
    for (int dc = -search_size; dc <= search_size; ++dc) {
      mb_row_ref = mb_row + mvr->mv.row * resize_factor + dr;
      mb_col_ref = mb_col + mvr->mv.col * resize_factor + dc;

      this_mse = get_mse_block(cur, refs[mvr->ref_index], mb_row, mb_col,
                               mb_row_ref, mb_col_ref, bsize, highbd);

      if (this_mse < best_mse) {
        best_mse = this_mse;
        best_subpel_x_qn = dr;
        best_subpel_y_qn = dc;
      }
    }
  }

  mvr->subpel_x_qn = best_subpel_x_qn;
  mvr->subpel_y_qn = best_subpel_y_qn;
}

// Replace the block in current frame using the block from the reference frame,
// using subpel motion vectors and interpolating the reference block. The block
// is replaced via weighted blending, using the alpha value calculated from
// the mean squared error between the current block and reference block.
static void replace_block_alpha(Y_BUFFER_CONFIG tmp, Y_BUFFER_CONFIG refs_sub[],
                                MV_MFQE *mvr, int16_t mb_row, int16_t mb_col,
                                BLOCK_SIZE bsize, int resize_factor,
                                uint8_t *swap_block) {
  (void)swap_block;

  int16_t mb_row_ref =
      (mb_row + mvr->mv.row) * resize_factor + mvr->subpel_x_qn;
  int16_t mb_col_ref =
      (mb_col + mvr->mv.col) * resize_factor + mvr->subpel_y_qn;

  Y_BUFFER_CONFIG ref = refs_sub[mvr->ref_index];
  uint8_t *src = ref.buffer + mb_row_ref * ref.stride + mb_col_ref;
  uint8_t *dst = tmp.buffer + mb_row * tmp.stride + mb_col;

  int src_stride = ref.stride;
  int dst_stride = tmp.stride;
  int block_w = block_size_wide[bsize];
  int block_h = block_size_high[bsize];

  ConvolveParams conv_params = get_conv_params(0, AOM_PLANE_Y, 8);
  int_interpfilters interp_filters;
  interp_filters.as_filters.x_filter = EIGHTTAP_REGULAR;
  interp_filters.as_filters.y_filter = EIGHTTAP_REGULAR;

  av1_convolve_2d_facade(src, src_stride, dst, dst_stride, block_w, block_h,
                         block_w, block_h, interp_filters, mvr->subpel_x_qn, 0,
                         mvr->subpel_y_qn, 0, 0, &conv_params, 0);
}

// High bitdepth version of replace_block_alpha, using the same method.
static void replace_block_alpha_highbd(Y_BUFFER_CONFIG tmp,
                                       Y_BUFFER_CONFIG refs_sub[], MV_MFQE *mvr,
                                       int16_t mb_row, int16_t mb_col,
                                       BLOCK_SIZE bsize, int resize_factor,
                                       uint16_t *swap_block, int bd) {
  int16_t mb_row_ref =
      (mb_row + mvr->mv.row) * resize_factor + mvr->subpel_x_qn;
  int16_t mb_col_ref =
      (mb_col + mvr->mv.col) * resize_factor + mvr->subpel_y_qn;

  Y_BUFFER_CONFIG ref = refs_sub[mvr->ref_index];
  uint8_t *src = ref.buffer + mb_row_ref * ref.stride + mb_col_ref;
  uint8_t *dst = tmp.buffer + mb_row * tmp.stride + mb_col;

  int src_stride = ref.stride;
  int dst_stride = tmp.stride;
  int block_w = block_size_wide[bsize];
  int block_h = block_size_high[bsize];

  av1_highbd_resize_plane(src, block_h * resize_factor, block_w * resize_factor,
                          src_stride, CONVERT_TO_BYTEPTR(swap_block), block_h,
                          block_w, dst_stride, bd);

  uint16_t *swap_ptr = swap_block;
  uint16_t *dst_ptr = CONVERT_TO_SHORTPTR(dst);
  double alpha = mvr->alpha;
  double beta = 1 - alpha;

  for (int row = 0; row < block_h; ++row) {
    for (int col = 0; col < block_w; ++col) {
      double p1 = (double)swap_ptr[col];
      double p2 = (double)dst_ptr[col];
      double p = floor(alpha * p1 + beta * p2 + 0.5);

      p = p > 255 ? 255 : (p < 0 ? 0 : p);  // Clamp the value of pixel.
      dst_ptr[col] = (uint16_t)p;
    }
    swap_ptr += dst_stride;
    dst_ptr += dst_stride;
  }
}

// Dynamically allocate memory for a single buffer.
static void mfqe_alloc_buf(Y_BUFFER_CONFIG *buf, int stride, int h, int w,
                           int resize_factor) {
  buf->stride = stride * resize_factor;
  buf->height = h * resize_factor;
  buf->width = w * resize_factor;

  // The buffer adds padding to the top and bottom for tap filters to be used
  // in Gaussian Blur and resizing. buffer points to the start of the frame and
  // buffer_orig points to the originally allocated buffer including padding.
  int buf_bytes = buf->stride * (buf->height + 2 * MFQE_PADDING_SIZE);
  buf->buffer_orig = aom_memalign(32, sizeof(uint8_t) * buf_bytes);
  memset(buf->buffer_orig, 0, sizeof(uint8_t) * buf_bytes);
  buf->buffer = buf->buffer_orig + buf->stride * MFQE_PADDING_SIZE;
}

// Dynamically allocate memory for a single buffer in high bitdepth version.
static void mfqe_alloc_buf_highbd(Y_BUFFER_CONFIG *buf, int stride, int h,
                                  int w, int resize_factor) {
  buf->stride = stride * resize_factor;
  buf->height = h * resize_factor;
  buf->width = w * resize_factor;

  // The buffer adds padding to the top and bottom for tap filters to be used
  // in Gaussian Blur and resizing. buffer points to the start of the frame and
  // buffer_orig points to the originally allocated buffer including padding.
  int buf_bytes = buf->stride * (buf->height + 2 * MFQE_PADDING_SIZE);
  uint16_t *buffer_orig = aom_memalign(32, sizeof(uint16_t) * buf_bytes);
  memset(buffer_orig, 0, sizeof(uint16_t) * buf_bytes);
  uint16_t *buffer = buffer_orig + buf->stride * MFQE_PADDING_SIZE;

  buf->buffer_orig = CONVERT_TO_BYTEPTR(buffer_orig);
  buf->buffer = CONVERT_TO_BYTEPTR(buffer);
}

// Dynamically allocate memory to be used for av1_apply_loop_mfqe.
static void mfqe_mem_alloc(Y_BUFFER_CONFIG *tmp, RefCntBuffer *ref_frames[],
                           Y_BUFFER_CONFIG *tmp_low, Y_BUFFER_CONFIG *tmp_sub,
                           Y_BUFFER_CONFIG *refs_low, Y_BUFFER_CONFIG *refs_sub,
                           int resize_factor) {
  mfqe_alloc_buf(tmp_low, tmp->stride, tmp->height, tmp->width, 1);
  mfqe_alloc_buf(tmp_sub, tmp->stride, tmp->height, tmp->width, resize_factor);

  YV12_BUFFER_CONFIG *ref;
  for (int i = 0; i < MFQE_NUM_REFS; i++) {
    ref = &ref_frames[i]->buf;
    mfqe_alloc_buf(&refs_low[i], ref->y_stride, ref->y_height, ref->y_width, 1);
    mfqe_alloc_buf(&refs_sub[i], ref->y_stride, ref->y_height, ref->y_width,
                   resize_factor);
  }
}

// Dynamically allocate memory to be used for av1_apply_loop_mfqe.
static void mfqe_mem_alloc_highbd(
    Y_BUFFER_CONFIG *tmp, RefCntBuffer *ref_frames[], Y_BUFFER_CONFIG *tmp_low,
    Y_BUFFER_CONFIG *tmp_sub, Y_BUFFER_CONFIG *refs_low,
    Y_BUFFER_CONFIG *refs_sub, int resize_factor) {
  mfqe_alloc_buf_highbd(tmp_low, tmp->stride, tmp->height, tmp->width, 1);
  mfqe_alloc_buf_highbd(tmp_sub, tmp->stride, tmp->height, tmp->width,
                        resize_factor);

  YV12_BUFFER_CONFIG *ref;
  for (int i = 0; i < MFQE_NUM_REFS; i++) {
    ref = &ref_frames[i]->buf;
    mfqe_alloc_buf_highbd(&refs_low[i], ref->y_stride, ref->y_height,
                          ref->y_width, 1);
    mfqe_alloc_buf_highbd(&refs_sub[i], ref->y_stride, ref->y_height,
                          ref->y_width, resize_factor);
  }
}

// Free all of the dynamically allocated memory inside av1_apply_loop_mfqe.
static void mfqe_mem_free(Y_BUFFER_CONFIG *tmp_low, Y_BUFFER_CONFIG *tmp_sub,
                          Y_BUFFER_CONFIG *refs_low,
                          Y_BUFFER_CONFIG *refs_sub) {
  aom_free(tmp_low->buffer_orig);
  aom_free(tmp_sub->buffer_orig);

  for (int i = 0; i < MFQE_NUM_REFS; i++) {
    aom_free(refs_low[i].buffer_orig);
    aom_free(refs_sub[i].buffer_orig);
  }
}

// Free all of the dynamically allocated memory inside av1_apply_loop_mfqe.
static void mfqe_mem_free_highbd(Y_BUFFER_CONFIG *tmp_low,
                                 Y_BUFFER_CONFIG *tmp_sub,
                                 Y_BUFFER_CONFIG *refs_low,
                                 Y_BUFFER_CONFIG *refs_sub) {
  aom_free(CONVERT_TO_SHORTPTR(tmp_low->buffer_orig));
  aom_free(CONVERT_TO_SHORTPTR(tmp_sub->buffer_orig));

  for (int i = 0; i < MFQE_NUM_REFS; i++) {
    aom_free(CONVERT_TO_SHORTPTR(refs_low[i].buffer_orig));
    aom_free(CONVERT_TO_SHORTPTR(refs_sub[i].buffer_orig));
  }
}

// Apply In-Loop Multi-Frame Quality Enhancement for low bitdepth version.
static void apply_loop_mfqe_lowbd(Y_BUFFER_CONFIG *tmp,
                                  RefCntBuffer *ref_frames[], BLOCK_SIZE bsize,
                                  int resize_factor, int bd) {
  Y_BUFFER_CONFIG tmp_low;
  Y_BUFFER_CONFIG tmp_sub;

  // Contains blurred versions of reference frames.
  Y_BUFFER_CONFIG refs_low[MFQE_NUM_REFS];
  // Contains resized versions of reference frames.
  Y_BUFFER_CONFIG refs_sub[MFQE_NUM_REFS];

  mfqe_mem_alloc(tmp, ref_frames, &tmp_low, &tmp_sub, refs_low, refs_sub,
                 resize_factor);

  mfqe_gaussian_blur(tmp->buffer, tmp_low.buffer, tmp->stride, tmp->height,
                     tmp->width, 0, bd);
  mfqe_resize_plane(tmp->buffer, tmp_sub.buffer, tmp->stride, tmp->height,
                    tmp->width, resize_factor, 0, bd);

  YV12_BUFFER_CONFIG *ref;
  for (int i = 0; i < MFQE_NUM_REFS; i++) {
    ref = &ref_frames[i]->buf;
    mfqe_gaussian_blur(ref->y_buffer, refs_low[i].buffer, ref->y_stride,
                       ref->y_height, ref->y_width, 0, bd);
    mfqe_resize_plane(ref->y_buffer, refs_sub[i].buffer, ref->y_stride,
                      ref->y_height, ref->y_width, resize_factor, 0, bd);
  }

  MV_MFQE mvr;
  int block_w = block_size_wide[bsize];
  int block_h = block_size_high[bsize];

  int block_bytes = (block_h + 2 * MFQE_PADDING_SIZE) * tmp->stride;
  uint8_t *block_orig = aom_memalign(32, sizeof(uint8_t) * block_bytes);
  uint8_t *swap_block = block_orig + MFQE_PADDING_SIZE * tmp->stride;

  for (int16_t mb_row = 0; mb_row < tmp->width; mb_row += block_w) {
    for (int16_t mb_col = 0; mb_col < tmp->height; mb_col += block_h) {
      mvr = kZeroMvMFQE;
      memset(block_orig, 0, sizeof(uint8_t) * block_bytes);

      full_pixel_search(&mvr, mb_row, mb_col, tmp_low, refs_low, bsize, 0);

      if (!mvr.valid) continue;  // Pass if mse is larger than threshold.
      sub_pixel_search(&mvr, mb_row, mb_col, tmp_sub, refs_sub, bsize,
                       resize_factor, 0);

      replace_block_alpha(*tmp, refs_sub, &mvr, mb_row, mb_col, bsize,
                          resize_factor, swap_block);
    }
  }

  // Free all of the dynamically allocated resources.
  aom_free(block_orig);
  mfqe_mem_free(&tmp_low, &tmp_sub, refs_low, refs_sub);
}

// Apply In-Loop Multi-Frame Quality Enhancement for high bitdepth version.
static void apply_loop_mfqe_highbd(Y_BUFFER_CONFIG *tmp,
                                   RefCntBuffer *ref_frames[], BLOCK_SIZE bsize,
                                   int resize_factor, int bd) {
  Y_BUFFER_CONFIG tmp_low;
  Y_BUFFER_CONFIG tmp_sub;

  // Contains blurred versions of reference frames.
  Y_BUFFER_CONFIG refs_low[MFQE_NUM_REFS];
  // Contains resized versions of reference frames.
  Y_BUFFER_CONFIG refs_sub[MFQE_NUM_REFS];

  mfqe_mem_alloc_highbd(tmp, ref_frames, &tmp_low, &tmp_sub, refs_low, refs_sub,
                        resize_factor);

  mfqe_gaussian_blur(tmp->buffer, tmp_low.buffer, tmp->stride, tmp->height,
                     tmp->width, 1, bd);
  mfqe_resize_plane(tmp->buffer, tmp_sub.buffer, tmp->stride, tmp->height,
                    tmp->width, resize_factor, 1, bd);

  YV12_BUFFER_CONFIG *ref;
  for (int i = 0; i < MFQE_NUM_REFS; i++) {
    ref = &ref_frames[i]->buf;
    mfqe_gaussian_blur(ref->y_buffer, refs_low[i].buffer, ref->y_stride,
                       ref->y_height, ref->y_width, 1, bd);
    mfqe_resize_plane(ref->y_buffer, refs_sub[i].buffer, ref->y_stride,
                      ref->y_height, ref->y_width, resize_factor, 1, bd);
  }

  MV_MFQE mvr;
  int block_w = block_size_wide[bsize];
  int block_h = block_size_high[bsize];
  int16_t num_cols = tmp->width / block_w;
  int16_t num_rows = tmp->height / block_h;

  int block_bytes = (block_h + 2 * MFQE_PADDING_SIZE) * tmp->stride;
  uint16_t *block_orig = aom_memalign(32, sizeof(uint16_t) * block_bytes);
  uint16_t *swap_block = block_orig + MFQE_PADDING_SIZE * tmp->stride;

  for (int16_t mb_row = 0; mb_row < num_rows; ++mb_row) {
    for (int16_t mb_col = 0; mb_col < num_cols; ++mb_col) {
      mvr = kZeroMvMFQE;
      memset(block_orig, 0, sizeof(uint16_t) * block_bytes);

      full_pixel_search(&mvr, mb_row, mb_col, tmp_low, refs_low, bsize, 1);

      if (!mvr.valid) continue;  // Pass if mse is larger than threshold.
      sub_pixel_search(&mvr, mb_row, mb_col, tmp_sub, refs_sub, bsize,
                       resize_factor, 1);

      replace_block_alpha_highbd(*tmp, refs_sub, &mvr, mb_row, mb_col, bsize,
                                 resize_factor, swap_block, bd);
    }
  }

  // Free all of the dynamically allocated resources.
  aom_free(block_orig);
  mfqe_mem_free_highbd(&tmp_low, &tmp_sub, refs_low, refs_sub);
}

void av1_apply_loop_mfqe(Y_BUFFER_CONFIG *tmp, RefCntBuffer *ref_frames[],
                         BLOCK_SIZE bsize, int resize_factor, int high_bd,
                         int bd) {
  if (high_bd)
    apply_loop_mfqe_highbd(tmp, ref_frames, bsize, resize_factor, bd);
  else
    apply_loop_mfqe_lowbd(tmp, ref_frames, bsize, resize_factor, bd);
}

// Copy the buffer from source to destination for a single plane.
static void copy_single_plane_lowbd(const uint8_t *src_buf, uint8_t *dst_buf,
                                    int src_stride, int dst_stride, int h,
                                    int w) {
  for (int row = 0; row < h; row++) {
    memcpy(dst_buf, src_buf, w);
    src_buf += src_stride;
    dst_buf += dst_stride;
  }
}

// Copy the buffer from source to destination for a single plane.
static void copy_single_plane_highbd(const uint8_t *src_buf, uint8_t *dst_buf,
                                     int src_stride, int dst_stride, int h,
                                     int w) {
  uint16_t *src_ptr = CONVERT_TO_SHORTPTR(src_buf);
  uint16_t *dst_ptr = CONVERT_TO_SHORTPTR(dst_buf);

  for (int row = 0; row < h; row++) {
    memcpy(dst_ptr, src_ptr, w * sizeof(*src_ptr));
    src_ptr += src_stride;
    dst_ptr += dst_stride;
  }
}

// Compute the mean squared error between two frames, just for a single plane.
static double get_mse_frame(uint8_t *buf1, uint8_t *buf2, int stride1,
                            int stride2, int w, int h, int highbd) {
  uint64_t sse;
  if (highbd)
    sse = aom_sse_highbd(buf1, stride1, buf2, stride2, w, h);
  else
    sse = aom_sse_lowbd(buf1, stride1, buf2, stride2, w, h);

  double mse = ((double)sse) / (w * h);
  return mse;
}

// Apply In-Loop Multi-Frame Quality Enhancement to the y plane of the current
// frame. If MFQE improves the current frame, replace the current y plane with
// the updated buffer. Returns 1 if MFQE is selected, 0 otherwise.
static void search_rest_mfqe_lowbd(const YV12_BUFFER_CONFIG *src,
                                   YV12_BUFFER_CONFIG *cur, AV1_COMMON *cm,
                                   int *use_mfqe) {
  int frame_bytes = cur->y_stride * (cur->y_height + 2 * MFQE_PADDING_SIZE);

  // Buffer to store temporary copy of current frame for MFQE.
  uint8_t *tmpbuf_orig = aom_memalign(32, sizeof(uint8_t) * frame_bytes);
  memset(tmpbuf_orig, 0, sizeof(uint8_t) * frame_bytes);

  uint8_t *tmpbuf = tmpbuf_orig + cur->y_stride * MFQE_PADDING_SIZE;
  copy_single_plane_lowbd(cur->y_buffer, tmpbuf, cur->y_stride, cur->y_stride,
                          cur->y_height, cur->y_width);

  Y_BUFFER_CONFIG tmp = { .buffer = tmpbuf,
                          .stride = cur->y_stride,
                          .height = cur->y_height,
                          .width = cur->y_width };

  RefCntBuffer *ref_frames[ALTREF_FRAME - LAST_FRAME + 1];
  int num_ref_frames = 0;
  MV_REFERENCE_FRAME ref_frame;
  for (ref_frame = LAST_FRAME; ref_frame < ALTREF_FRAME; ++ref_frame) {
    RefCntBuffer *ref = get_ref_frame_buf(cm, ref_frame);
    if (ref) ref_frames[num_ref_frames++] = ref;
  }

  // Return if we have less than 3 available reference frames.
  if (num_ref_frames < MFQE_NUM_REFS) {
    aom_free(tmpbuf_orig);
    return;
  }

  // Assert that pointers to RefCntBuffer are valid, then sort the reference
  // frames based on their base_qindex, from lowest to highest.
  for (int i = 0; i < num_ref_frames; i++) assert(ref_frames[i] != NULL);
  qsort(ref_frames, num_ref_frames, sizeof(ref_frames[0]), cmpref);

  // Perform In-Loop Multi-Frame Quality Enhancement on tmp.
  av1_apply_loop_mfqe(&tmp, ref_frames, MFQE_BLOCK_SIZE, MFQE_SCALE_SIZE, 0,
                      cm->seq_params.bit_depth);

  double mse_prev =
      get_mse_frame(src->y_buffer, cur->y_buffer, src->y_stride, cur->y_stride,
                    src->y_width, src->y_height, 0);
  double mse_curr = get_mse_frame(src->y_buffer, tmp.buffer, src->y_stride,
                                  tmp.stride, src->y_width, src->y_height, 0);

  if (mse_curr < mse_prev) {
    *use_mfqe = 1;
    copy_single_plane_lowbd(tmpbuf, cur->y_buffer, cur->y_stride, cur->y_stride,
                            cur->y_height, cur->y_width);
  }

  aom_free(tmpbuf_orig);
}

static void search_rest_mfqe_highbd(const YV12_BUFFER_CONFIG *src,
                                    YV12_BUFFER_CONFIG *cur, AV1_COMMON *cm,
                                    int *use_mfqe) {
  int frame_bytes = cur->y_stride * (cur->y_height + 2 * MFQE_PADDING_SIZE);

  // Buffer to store temporary copy of current frame for MFQE.
  uint16_t *tmpbuf_orig = aom_memalign(32, sizeof(uint16_t) * frame_bytes);
  memset(tmpbuf_orig, 0, sizeof(uint16_t) * frame_bytes);

  uint16_t *tmpbuf = tmpbuf_orig + cur->y_stride * MFQE_PADDING_SIZE;
  Y_BUFFER_CONFIG tmp = { .buffer = CONVERT_TO_BYTEPTR(tmpbuf),
                          .stride = cur->y_stride,
                          .height = cur->y_height,
                          .width = cur->y_width };

  copy_single_plane_highbd(cur->y_buffer, tmp.buffer, cur->y_stride,
                           cur->y_stride, cur->y_height, cur->y_width);

  RefCntBuffer *ref_frames[ALTREF_FRAME - LAST_FRAME + 1];
  int num_ref_frames = 0;
  MV_REFERENCE_FRAME ref_frame;
  for (ref_frame = LAST_FRAME; ref_frame < ALTREF_FRAME; ++ref_frame) {
    RefCntBuffer *ref = get_ref_frame_buf(cm, ref_frame);
    if (ref) ref_frames[num_ref_frames++] = ref;
  }

  // Return if we have less than 3 available reference frames.
  if (num_ref_frames < MFQE_NUM_REFS) {
    aom_free(tmpbuf_orig);
    return;
  }

  // Assert that pointers to RefCntBuffer are valid, then sort the reference
  // frames based on their base_qindex, from lowest to highest.
  for (int i = 0; i < num_ref_frames; i++) assert(ref_frames[i] != NULL);
  qsort(ref_frames, num_ref_frames, sizeof(ref_frames[0]), cmpref);

  // Perform In-Loop Multi-Frame Quality Enhancement on tmp.
  av1_apply_loop_mfqe(&tmp, ref_frames, MFQE_BLOCK_SIZE, MFQE_SCALE_SIZE, 1,
                      cm->seq_params.bit_depth);

  double mse_prev =
      get_mse_frame(src->y_buffer, cur->y_buffer, src->y_stride, cur->y_stride,
                    src->y_width, src->y_height, 1);
  double mse_curr = get_mse_frame(src->y_buffer, tmp.buffer, src->y_stride,
                                  tmp.stride, src->y_width, src->y_height, 1);

  if (mse_curr < mse_prev) {
    *use_mfqe = 1;
    copy_single_plane_highbd(tmp.buffer, cur->y_buffer, cur->y_stride,
                             cur->y_stride, cur->y_height, cur->y_width);
  }

  aom_free(tmpbuf_orig);
}

// Wrapper function for In-Loop Multi-Frame Quality Enhancement. There are two
// different code paths for low bit depth and high bit depth.
void av1_search_rest_mfqe(const YV12_BUFFER_CONFIG *src,
                          YV12_BUFFER_CONFIG *cur, AV1_COMMON *cm,
                          int *use_mfqe, int high_bd) {
  if (high_bd)
    search_rest_mfqe_highbd(src, cur, cm, use_mfqe);
  else
    search_rest_mfqe_lowbd(src, cur, cm, use_mfqe);
}

// MFQE decoding function in low bitdepth version.
void decode_restore_mfqe_lowbd(AV1_COMMON *cm, BLOCK_SIZE bsize,
                               int resize_factor) {
  YV12_BUFFER_CONFIG *cur = &cm->cur_frame->buf;
  int frame_bytes = cur->y_stride * (cur->y_height + 2 * MFQE_PADDING_SIZE);

  // Buffer to store temporary copy of current frame for MFQE.
  uint8_t *tmpbuf_orig = aom_memalign(32, sizeof(uint8_t) * frame_bytes);
  memset(tmpbuf_orig, 0, sizeof(uint8_t) * frame_bytes);

  uint8_t *tmpbuf = tmpbuf_orig + cur->y_stride * MFQE_PADDING_SIZE;
  copy_single_plane_lowbd(cur->y_buffer, tmpbuf, cur->y_stride, cur->y_stride,
                          cur->y_height, cur->y_width);

  Y_BUFFER_CONFIG cur_frame = { .buffer = tmpbuf,
                                .stride = cur->y_stride,
                                .height = cur->y_height,
                                .width = cur->y_width };

  RefCntBuffer *ref_frames[ALTREF_FRAME - LAST_FRAME + 1];
  int num_ref_frames = 0;
  MV_REFERENCE_FRAME ref_frame;
  for (ref_frame = LAST_FRAME; ref_frame < ALTREF_FRAME; ++ref_frame) {
    RefCntBuffer *ref = get_ref_frame_buf(cm, ref_frame);
    if (ref) ref_frames[num_ref_frames++] = ref;
  }
  assert(num_ref_frames >= MFQE_NUM_REFS);

  // Assert that pointers to RefCntBuffer are valid, then sort the reference
  // frames based on their base_qindex, from lowest to highest.
  for (int i = 0; i < num_ref_frames; i++) assert(ref_frames[i] != NULL);
  qsort(ref_frames, num_ref_frames, sizeof(ref_frames[0]), cmpref);

  // Perform In-Loop Multi-Frame Quality Enhancement on tmp.
  av1_apply_loop_mfqe(&cur_frame, ref_frames, bsize, resize_factor, 0,
                      cm->seq_params.bit_depth);

  copy_single_plane_lowbd(tmpbuf, cur->y_buffer, cur->y_stride, cur->y_stride,
                          cur->y_height, cur->y_width);

  aom_free(tmpbuf_orig);
}

// MFQE decoding function in high bitdepth version.
void decode_restore_mfqe_highbd(AV1_COMMON *cm, BLOCK_SIZE bsize,
                                int resize_factor) {
  YV12_BUFFER_CONFIG *cur = &cm->cur_frame->buf;

  int frame_bytes = cur->y_stride * (cur->y_height + 2 * MFQE_PADDING_SIZE);

  // Buffer to store temporary copy of current frame for MFQE.
  uint16_t *tmpbuf_orig = aom_memalign(32, sizeof(uint16_t) * frame_bytes);
  memset(tmpbuf_orig, 0, sizeof(uint16_t) * frame_bytes);

  uint16_t *tmpbuf = tmpbuf_orig + cur->y_stride * MFQE_PADDING_SIZE;
  Y_BUFFER_CONFIG cur_frame = { .buffer = CONVERT_TO_BYTEPTR(tmpbuf),
                                .stride = cur->y_stride,
                                .height = cur->y_height,
                                .width = cur->y_width };

  copy_single_plane_highbd(cur->y_buffer, cur_frame.buffer, cur->y_stride,
                           cur->y_stride, cur->y_height, cur->y_width);

  RefCntBuffer *ref_frames[ALTREF_FRAME - LAST_FRAME + 1];
  int num_ref_frames = 0;
  MV_REFERENCE_FRAME ref_frame;
  for (ref_frame = LAST_FRAME; ref_frame < ALTREF_FRAME; ++ref_frame) {
    RefCntBuffer *ref = get_ref_frame_buf(cm, ref_frame);
    if (ref) ref_frames[num_ref_frames++] = ref;
  }
  assert(num_ref_frames >= MFQE_NUM_REFS);

  // Assert that pointers to RefCntBuffer are valid, then sort the reference
  // frames based on their base_qindex, from lowest to highest.
  for (int i = 0; i < num_ref_frames; i++) assert(ref_frames[i] != NULL);
  qsort(ref_frames, num_ref_frames, sizeof(ref_frames[0]), cmpref);

  // Perform In-Loop Multi-Frame Quality Enhancement on tmp.
  av1_apply_loop_mfqe(&cur_frame, ref_frames, bsize, resize_factor, 0,
                      cm->seq_params.bit_depth);

  copy_single_plane_highbd(cur_frame.buffer, cur->y_buffer, cur->y_stride,
                           cur->y_stride, cur->y_height, cur->y_width);
  aom_free(tmpbuf_orig);
}

void av1_decode_restore_mfqe(AV1_COMMON *cm, int high_bd) {
  if (high_bd)
    decode_restore_mfqe_highbd(cm, MFQE_BLOCK_SIZE, MFQE_SCALE_SIZE);
  else
    decode_restore_mfqe_lowbd(cm, MFQE_BLOCK_SIZE, MFQE_SCALE_SIZE);
}
