/*
 * Copyright (c) 2018, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <assert.h>
#include <stdbool.h>
#include <string.h>

#include "config/av1_rtcd.h"

#include "av1/encoder/block.h"
#include "av1/encoder/hash.h"
#include "av1/encoder/hash_motion.h"

#define kSrcBits 16
// Each of the 6 supported block sizes (4, 8, 16, 32, 64, 128) has its own set
// of 1 << kSrcBits buckets, indexed by the return value of
// hash_block_size_to_index().
#define kBucketsPerBlockSize (1 << kSrcBits)
#define kMaxCandidatesPerHashBucket 256

static void get_pixels_in_1D_char_array_by_block_2x2(const uint8_t *y_src,
                                                     int stride,
                                                     uint8_t *p_pixels_in1D) {
  const uint8_t *p_pel = y_src;
  int index = 0;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      p_pixels_in1D[index++] = p_pel[j];
    }
    p_pel += stride;
  }
}

static void get_pixels_in_1D_short_array_by_block_2x2(const uint16_t *y_src,
                                                      int stride,
                                                      uint16_t *p_pixels_in1D) {
  const uint16_t *p_pel = y_src;
  int index = 0;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      p_pixels_in1D[index++] = p_pel[j];
    }
    p_pel += stride;
  }
}

// the hash value (hash_value1) consists of two parts, the first 3 bits relate
// to the block size and the remaining 16 bits are the crc values. This
// function is used to get the first 3 bits.
static int hash_block_size_to_index(int block_size) {
  switch (block_size) {
    case 4: return 0;
    case 8: return 1;
    case 16: return 2;
    case 32: return 3;
    case 64: return 4;
    case 128: return 5;
    default: return -1;
  }
}

static uint32_t get_identity_hash_value(const uint8_t a, const uint8_t b,
                                        const uint8_t c, const uint8_t d) {
  // The four input values add up to 32 bits, which is the size of the output.
  // Just pack those values as is.
  return ((uint32_t)a << 24) + ((uint32_t)b << 16) + ((uint32_t)c << 8) +
         ((uint32_t)d);
}

static uint32_t get_xor_hash_value_hbd(const uint16_t a, const uint16_t b,
                                       const uint16_t c, const uint16_t d) {
  uint32_t result;
  // Pack the lower 8 bits of each input value to the 32 bit output, then xor
  // with the upper 8 bits of each input value.
  result = ((uint32_t)(a & 0x00ff) << 24) + ((uint32_t)(b & 0x00ff) << 16) +
           ((uint32_t)(c & 0x00ff) << 8) + ((uint32_t)(d & 0x00ff));
  result ^= ((uint32_t)(a & 0xff00) << 16) + ((uint32_t)(b & 0xff00) << 8) +
            ((uint32_t)(c & 0xff00)) + ((uint32_t)(d & 0xff00) >> 8);
  return result;
}

void av1_hash_table_init(IntraBCHashInfo *intrabc_hash_info) {
  if (!intrabc_hash_info->crc_initialized) {
    av1_crc32c_calculator_init(&intrabc_hash_info->crc_calculator);
    intrabc_hash_info->crc_initialized = 1;
  }
}

static void hash_table_destroy(hash_table *p_hash_table) {
  for (int i = 0; i < NELEMENTS(p_hash_table->block_hashes); i++) {
    aom_free(p_hash_table->bucket_count[i]);
    aom_free(p_hash_table->bucket_start[i]);
    aom_free(p_hash_table->block_hashes[i]);
  }
  memset(p_hash_table, 0, sizeof(*p_hash_table));
}

// Empties the table. The buffers are kept for the next frame.
static void hash_table_clear(hash_table *p_hash_table) {
  for (int i = 0; i < NELEMENTS(p_hash_table->bucket_count); i++) {
    if (p_hash_table->bucket_count[i] == NULL) continue;
    memset(p_hash_table->bucket_count[i], 0,
           kBucketsPerBlockSize * sizeof(*p_hash_table->bucket_count[i]));
  }
}

static void free_tile_tables(IntraBCHashInfo *intrabc_hash_info) {
  for (int i = 0; i < intrabc_hash_info->num_tile_tables; i++) {
    hash_table_destroy(&intrabc_hash_info->tile_tables[i]);
  }
  aom_free(intrabc_hash_info->tile_tables);
  intrabc_hash_info->tile_tables = NULL;
  intrabc_hash_info->num_tile_tables = 0;
}

bool av1_hash_tables_create(IntraBCHashInfo *intrabc_hash_info,
                            int num_tables) {
  if (intrabc_hash_info->num_tile_tables != num_tables) {
    free_tile_tables(intrabc_hash_info);
    intrabc_hash_info->tile_tables = (hash_table *)aom_calloc(
        num_tables, sizeof(*intrabc_hash_info->tile_tables));
    if (intrabc_hash_info->tile_tables == NULL) return false;
    intrabc_hash_info->num_tile_tables = num_tables;
  }
  for (int i = 0; i < num_tables; i++) {
    hash_table_clear(&intrabc_hash_info->tile_tables[i]);
  }
  return true;
}

void av1_hash_tables_destroy(IntraBCHashInfo *intrabc_hash_info) {
  free_tile_tables(intrabc_hash_info);
  for (int i = 0; i < 2; i++) {
    aom_free(intrabc_hash_info->frame_hash_buffer[i]);
    intrabc_hash_info->frame_hash_buffer[i] = NULL;
  }
  intrabc_hash_info->frame_hash_width = 0;
  intrabc_hash_info->frame_hash_height = 0;
}

// Makes room for the buckets of one block size, if it has none yet. Only the
// block sizes that get built have buckets. With hash_max_8x8_intrabc_blocks
// that is just 8x8.
static bool hash_table_reserve_buckets(hash_table *p_hash_table,
                                       int size_index) {
  if (p_hash_table->bucket_count[size_index] != NULL) return true;
  p_hash_table->bucket_count[size_index] = (uint16_t *)aom_calloc(
      kBucketsPerBlockSize, sizeof(*p_hash_table->bucket_count[size_index]));
  p_hash_table->bucket_start[size_index] = (uint32_t *)aom_malloc(
      kBucketsPerBlockSize * sizeof(*p_hash_table->bucket_start[size_index]));
  if (p_hash_table->bucket_count[size_index] == NULL ||
      p_hash_table->bucket_start[size_index] == NULL) {
    aom_free(p_hash_table->bucket_count[size_index]);
    aom_free(p_hash_table->bucket_start[size_index]);
    p_hash_table->bucket_count[size_index] = NULL;
    p_hash_table->bucket_start[size_index] = NULL;
    return false;
  }
  return true;
}

// Makes sure block_hashes[size_index] has room for at least size entries. Its
// old entries are lost even when the function fails.
static bool hash_table_reserve_block_hashes(hash_table *p_hash_table,
                                            int size_index, uint32_t size) {
  if (size <= p_hash_table->block_hashes_size[size_index]) return true;
  aom_free(p_hash_table->block_hashes[size_index]);
  p_hash_table->block_hashes[size_index] =
      aom_malloc(size * sizeof(*p_hash_table->block_hashes[size_index]));
  if (p_hash_table->block_hashes[size_index] == NULL) {
    p_hash_table->block_hashes_size[size_index] = 0;
    return false;
  }
  p_hash_table->block_hashes_size[size_index] = size;
  return true;
}

static inline void add_to_bucket_count(uint16_t *count, int n) {
  *count = AOMMIN(*count + n, kMaxCandidatesPerHashBucket);
}

int32_t av1_hash_table_count(const hash_table *p_hash_table,
                             uint32_t hash_value) {
  const uint16_t *const bucket_count =
      p_hash_table->bucket_count[hash_value >> kSrcBits];
  if (bucket_count == NULL) return 0;
  return bucket_count[hash_value % kBucketsPerBlockSize];
}

const block_hash *av1_hash_get_first_block_hash(const hash_table *p_hash_table,
                                                uint32_t hash_value) {
  assert(av1_hash_table_count(p_hash_table, hash_value) > 0);
  const int size_index = hash_value >> kSrcBits;
  const uint32_t bucket = hash_value % kBucketsPerBlockSize;
  const block_hash *const block_hashes = p_hash_table->block_hashes[size_index];
  return &block_hashes[p_hash_table->bucket_start[size_index][bucket]];
}

// Hashes width 2x2 blocks in a row, the first one at (x, y).
static void hash_2x2_row(const YV12_BUFFER_CONFIG *picture, int x, int y,
                         int width, uint32_t *dst) {
  const int stride = picture->y_stride;
  if (picture->flags & YV12_FLAG_HIGHBITDEPTH) {
    const uint16_t *const top =
        CONVERT_TO_SHORTPTR(picture->y_buffer) + y * stride + x;
    const uint16_t *const bottom = top + stride;
    for (int i = 0; i < width; i++) {
      // For HBD, we either have 40 or 48 bits of input data that the xor hash
      // reduce to 32 bits. We intentionally don't want to "discard" bits to
      // avoid any kind of biasing.
      dst[i] =
          get_xor_hash_value_hbd(top[i], top[i + 1], bottom[i], bottom[i + 1]);
    }
  } else {
    const uint8_t *const top = picture->y_buffer + y * stride + x;
    const uint8_t *const bottom = top + stride;
    for (int i = 0; i < width; i++) {
      // This 2x2 hash isn't used directly as a "key" for the hash table, so
      // we can afford to just copy the 4 8-bit pixel values as a single
      // 32-bit value directly. (i.e. there are no concerns of a lack of
      // uniform distribution)
      dst[i] =
          get_identity_hash_value(top[i], top[i + 1], bottom[i], bottom[i + 1]);
    }
  }
}

// A 4x4 hash at row y reads the 2x2 rows y and y + 2, and an 8x8 hash the 4x4
// rows y and y + 4. The pass keeps only the rows it still needs.
#define kRingRows2x2 3
#define kRingRows4x4 5

bool av1_generate_block_8x8_hash_value(IntraBCHashInfo *intrabc_hash_info,
                                       const YV12_BUFFER_CONFIG *picture,
                                       const IntraBCHashRegion *region,
                                       uint32_t *hash_4x4, uint32_t *hash_8x8,
                                       int pic_block_hash_stride) {
  const int x_start = region->x_start;
  const int y_start = region->y_start;
  const int width = region->x_end - x_start;
  const int height = region->y_end - y_start;
  // Source blocks at the right and bottom borders cannot be part of larger
  // blocks, so each block size has fewer positions than the one below it.
  const int width_4x4 = width - 3;
  const int height_4x4 = height - 3;
  const int width_8x8 = width - 7;
  // Without the 4x4 table only 8x8 hashes are written.
  const int min_size = hash_4x4 != NULL ? 4 : 8;
  if (width < min_size || height < min_size) return true;

  // The 4x4 rows only need the ring when hash_4x4 does not hold them.
  const int ring_rows = kRingRows2x2 + (hash_4x4 != NULL ? 0 : kRingRows4x4);
  uint32_t *const ring =
      (uint32_t *)aom_malloc(sizeof(*ring) * width * ring_rows);
  if (ring == NULL) return false;
  uint32_t *const rows_2x2 = ring;
  // The 4x4 rows go to hash_4x4 instead when the caller needs all of them.
  uint32_t *rows_4x4 = ring + kRingRows2x2 * width;
  int rows_4x4_stride = width;
  int num_rows_4x4 = kRingRows4x4;
  if (hash_4x4 != NULL) {
    rows_4x4 = hash_4x4 + y_start * pic_block_hash_stride + x_start;
    rows_4x4_stride = pic_block_hash_stride;
    num_rows_4x4 = height_4x4;
  }
  uint32_t *const out_8x8 =
      hash_8x8 + y_start * pic_block_hash_stride + x_start;
  CRC32C *const calc = &intrabc_hash_info->crc_calculator;

  hash_2x2_row(picture, x_start, y_start, width - 1, rows_2x2);
  hash_2x2_row(picture, x_start, y_start + 1, width - 1, rows_2x2 + width);
  for (int y = 0; y < height_4x4; y++) {
    uint32_t *const next_2x2 = rows_2x2 + (y + 2) % kRingRows2x2 * width;
    hash_2x2_row(picture, x_start, y_start + y + 2, width - 1, next_2x2);
    uint32_t *const row_4x4 = rows_4x4 + y % num_rows_4x4 * rows_4x4_stride;
    av1_combine_block_hashes(calc, rows_2x2 + y % kRingRows2x2 * width,
                             next_2x2, row_4x4, width_4x4, 2);
    if (y >= 4 && width_8x8 > 0) {
      const uint32_t *const top_4x4 =
          rows_4x4 + (y - 4) % num_rows_4x4 * rows_4x4_stride;
      av1_combine_block_hashes(calc, top_4x4, row_4x4,
                               out_8x8 + (y - 4) * pic_block_hash_stride,
                               width_8x8, 4);
    }
  }
  aom_free(ring);
  return true;
}

void av1_generate_block_hash_value(IntraBCHashInfo *intrabc_hash_info,
                                   const IntraBCHashRegion *region,
                                   int block_size,
                                   const uint32_t *src_pic_block_hash,
                                   uint32_t *dst_pic_block_hash,
                                   int pic_block_hash_stride) {
  const int x_end = region->x_end - block_size + 1;
  const int y_end = region->y_end - block_size + 1;
  if (x_end <= region->x_start || y_end <= region->y_start) return;
  const int sub_size = block_size >> 1;

  // Build up a bigger block from 4 smaller, non-overlapping source block
  // hashes, and compute its hash. Note: source blocks at the right and
  // bottom borders cannot be part of larger blocks, therefore they won't be
  // considered into the block hash value generation process.
  for (int y_pos = region->y_start; y_pos < y_end; y_pos++) {
    const int pos = y_pos * pic_block_hash_stride + region->x_start;
    av1_combine_block_hashes(
        &intrabc_hash_info->crc_calculator, src_pic_block_hash + pos,
        src_pic_block_hash + pos + sub_size * pic_block_hash_stride,
        dst_pic_block_hash + pos, x_end - region->x_start, sub_size);
  }
}

// Adds the blocks of one block size that lie inside region to the table. The
// counts of that block size must be zero on entry, as av1_hash_tables_create()
// leaves them. On return they hold the number of entries in each bucket.
bool av1_add_to_hash_map_by_row_with_precal_data(
    hash_table *p_hash_table, const uint32_t *pic_hash, int pic_hash_stride,
    const IntraBCHashRegion *region, int block_size) {
  const int x_start = region->x_start;
  const int y_start = region->y_start;
  const int x_end = region->x_end - block_size + 1;
  const int y_end = region->y_end - block_size + 1;

  const int size_index = hash_block_size_to_index(block_size);
  assert(size_index >= 0);
  const int crc_mask = kBucketsPerBlockSize - 1;
  int step = block_size;
  int x_offset = 0;
  int y_offset = 0;

  // No block of this size fits in the region.
  if (x_end <= x_start || y_end <= y_start) return true;
  if (!hash_table_reserve_buckets(p_hash_table, size_index)) return false;
  uint16_t *const bucket_count = p_hash_table->bucket_count[size_index];
  uint32_t *const bucket_start = p_hash_table->bucket_start[size_index];

  // The next two loops set bucket_start. The first counts the blocks of each
  // bucket, and the second places the buckets one after another.
  //
  // Place an upper bound each hash table bucket to up to 256 intrabc
  // block candidates, and ignore subsequent ones. Considering more can
  // unnecessarily slow down encoding for virtually no efficiency gain.
  // How many blocks each bucket keeps doesn't depend on the order they are
  // added in, so count them first in raster order. Flat areas give long runs
  // of the same bucket, so count a whole run at once.
  for (int y_pos = y_start; y_pos < y_end; y_pos++) {
    const uint32_t *const row_hash = pic_hash + y_pos * pic_hash_stride;
    int bucket = row_hash[x_start] & crc_mask;
    int run = 0;
    for (int x_pos = x_start; x_pos < x_end; x_pos++) {
      const int curr_bucket = row_hash[x_pos] & crc_mask;
      if (curr_bucket != bucket) {
        add_to_bucket_count(&bucket_count[bucket], run);
        bucket = curr_bucket;
        run = 0;
      }
      run++;
    }
    add_to_bucket_count(&bucket_count[bucket], run);
  }

  // Store the buckets one after another. The counts restart from zero and
  // track how far each bucket is filled.
  uint32_t num_block_hashes = 0;
  for (int i = 0; i <= crc_mask; i++) {
    bucket_start[i] = num_block_hashes;
    num_block_hashes += bucket_count[i];
    bucket_count[i] = 0;
  }
  if (!hash_table_reserve_block_hashes(p_hash_table, size_index,
                                       num_block_hashes)) {
    return false;
  }
  block_hash *const block_hashes = p_hash_table->block_hashes[size_index];

  // Explore the entire region hierarchically to add intrabc candidate blocks to
  // the hash table, by starting with coarser steps (the block size), towards
  // finer-grained steps until every candidate block has been considered.
  // The nested for loop goes through the region column by column.

  // Doing a hierarchical block exploration helps maximize spatial dispersion
  // of the first and foremost candidate blocks while minimizing overlap between
  // them. This is helpful because we only keep up to 256 entries of the
  // same candidate block (located in different places), so we want those
  // entries to cover the biggest area of the image to encode to maximize coding
  // efficiency.

  // This is the coordinate exploration order example for an 8x8 region, with
  // block_size = 4. The top-left corner (x, y) coordinates of each candidate
  // block are shown below. There are 5 * 5 (25) candidate blocks.
  //    x  0  1  2  3  4  5  6  7
  //  y +------------------------
  //  0 |  1 10  5 13  3
  //  1 | 16 22 18 24 20
  //  2 |  7 11  9 14  8
  //  3 | 17 23 19 25 21
  //  4 |  2 12  6 15  4--------+
  //  5 |              | 4 x 4  |
  //  6 |              | block  |
  //  7 |              +--------+

  // Please note that due to the way block exploration works, the smallest step
  // used is 2 (i.e. no two adjacent blocks will be explored consecutively).
  // Also, the exploration is designed to visit each block candidate only once.
  while (step > 1) {
    for (int x_pos = x_start + x_offset; x_pos < x_end; x_pos += step) {
      for (int y_pos = y_start + y_offset; y_pos < y_end; y_pos += step) {
        const int pos = y_pos * pic_hash_stride + x_pos;
        const int bucket = pic_hash[pos] & crc_mask;
        const int count = bucket_count[bucket];
        assert(count <= kMaxCandidatesPerHashBucket);
        if (count == kMaxCandidatesPerHashBucket) continue;

        block_hash *const curr_block_hash =
            &block_hashes[bucket_start[bucket] + count];
        curr_block_hash->x = x_pos;
        curr_block_hash->y = y_pos;
        curr_block_hash->hash_value2 = pic_hash[pos];
        bucket_count[bucket] = count + 1;
      }
    }

    // Adjust offsets and step sizes with this state machine.
    // State 0 is needed because no blocks in pic_hash have been explored,
    // so exploration requires a way to account for blocks with both zero
    // x_offset and zero y_offset.
    // State 0 is always meant to be executed first, but the relative order of
    // states 1, 2 and 3 can be arbitrary, as long as no two adjacent blocks
    // are explored consecutively.
    if (x_offset == 0 && y_offset == 0) {
      // State 0 -> State 1: special case
      // This state transition will only execute when step == block_size
      x_offset = step / 2;
    } else if (x_offset == step / 2 && y_offset == 0) {
      // State 1 -> State 2
      x_offset = 0;
      y_offset = step / 2;
    } else if (x_offset == 0 && y_offset == step / 2) {
      // State 2 -> State 3
      x_offset = step / 2;
    } else {
      assert(x_offset == step / 2 && y_offset == step / 2);
      // State 3 -> State 1: We've fully explored all the coordinates for the
      // current step size, continue by halving the step size
      step /= 2;
      x_offset = step / 2;
      y_offset = 0;
    }
  }

  return true;
}

int av1_hash_is_horizontal_perfect(const YV12_BUFFER_CONFIG *picture,
                                   int block_size, int x_start, int y_start) {
  const int stride = picture->y_stride;
  const uint8_t *p = picture->y_buffer + y_start * stride + x_start;

  if (picture->flags & YV12_FLAG_HIGHBITDEPTH) {
    const uint16_t *p16 = CONVERT_TO_SHORTPTR(p);
    for (int i = 0; i < block_size; i++) {
      for (int j = 1; j < block_size; j++) {
        if (p16[j] != p16[0]) {
          return 0;
        }
      }
      p16 += stride;
    }
  } else {
    for (int i = 0; i < block_size; i++) {
      for (int j = 1; j < block_size; j++) {
        if (p[j] != p[0]) {
          return 0;
        }
      }
      p += stride;
    }
  }

  return 1;
}

int av1_hash_is_vertical_perfect(const YV12_BUFFER_CONFIG *picture,
                                 int block_size, int x_start, int y_start) {
  const int stride = picture->y_stride;
  const uint8_t *p = picture->y_buffer + y_start * stride + x_start;

  if (picture->flags & YV12_FLAG_HIGHBITDEPTH) {
    const uint16_t *p16 = CONVERT_TO_SHORTPTR(p);
    for (int i = 0; i < block_size; i++) {
      for (int j = 1; j < block_size; j++) {
        if (p16[j * stride + i] != p16[i]) {
          return 0;
        }
      }
    }
  } else {
    for (int i = 0; i < block_size; i++) {
      for (int j = 1; j < block_size; j++) {
        if (p[j * stride + i] != p[i]) {
          return 0;
        }
      }
    }
  }
  return 1;
}

void av1_get_block_hash_value(IntraBCHashInfo *intra_bc_hash_info,
                              const uint8_t *y_src, int stride, int block_size,
                              uint32_t *hash_value1, uint32_t *hash_value2,
                              int use_highbitdepth) {
  int add_value = hash_block_size_to_index(block_size);
  assert(add_value >= 0);
  add_value <<= kSrcBits;
  const int crc_mask = (1 << kSrcBits) - 1;

  CRC32C *calc = &intra_bc_hash_info->crc_calculator;
  uint32_t **buf = intra_bc_hash_info->hash_value_buffer;

  // 2x2 subblock hash values in current CU
  int sub_block_in_width = (block_size >> 1);
  if (use_highbitdepth) {
    uint16_t pixel_to_hash[4];
    uint16_t *y16_src = CONVERT_TO_SHORTPTR(y_src);
    for (int y_pos = 0; y_pos < block_size; y_pos += 2) {
      for (int x_pos = 0; x_pos < block_size; x_pos += 2) {
        int pos = (y_pos >> 1) * sub_block_in_width + (x_pos >> 1);
        get_pixels_in_1D_short_array_by_block_2x2(
            y16_src + y_pos * stride + x_pos, stride, pixel_to_hash);
        assert(pos < AOM_BUFFER_SIZE_FOR_BLOCK_HASH);
        // For HBD, we either have 40 or 48 bits of input data that the xor hash
        // reduce to 32 bits. We intentionally don't want to "discard" bits to
        // avoid any kind of biasing.
        buf[0][pos] =
            get_xor_hash_value_hbd(pixel_to_hash[0], pixel_to_hash[1],
                                   pixel_to_hash[2], pixel_to_hash[3]);
      }
    }
  } else {
    uint8_t pixel_to_hash[4];
    for (int y_pos = 0; y_pos < block_size; y_pos += 2) {
      for (int x_pos = 0; x_pos < block_size; x_pos += 2) {
        int pos = (y_pos >> 1) * sub_block_in_width + (x_pos >> 1);
        get_pixels_in_1D_char_array_by_block_2x2(y_src + y_pos * stride + x_pos,
                                                 stride, pixel_to_hash);
        assert(pos < AOM_BUFFER_SIZE_FOR_BLOCK_HASH);
        // This 2x2 hash isn't used directly as a "key" for the hash table, so
        // we can afford to just copy the 4 8-bit pixel values as a single
        // 32-bit value directly. (i.e. there are no concerns of a lack of
        // uniform distribution)
        buf[0][pos] =
            get_identity_hash_value(pixel_to_hash[0], pixel_to_hash[1],
                                    pixel_to_hash[2], pixel_to_hash[3]);
      }
    }
  }

  int src_sub_block_in_width = sub_block_in_width;
  sub_block_in_width >>= 1;

  int src_idx = 0;
  int dst_idx = !src_idx;

  // 4x4 subblock hash values to current block hash values
  uint32_t to_hash[4];
  for (int sub_width = 4; sub_width <= block_size;
       sub_width *= 2, src_idx = !src_idx) {
    dst_idx = !src_idx;

    int dst_pos = 0;
    for (int y_pos = 0; y_pos < sub_block_in_width; y_pos++) {
      for (int x_pos = 0; x_pos < sub_block_in_width; x_pos++) {
        int srcPos = (y_pos << 1) * src_sub_block_in_width + (x_pos << 1);

        assert(srcPos + 1 < AOM_BUFFER_SIZE_FOR_BLOCK_HASH);
        assert(srcPos + src_sub_block_in_width + 1 <
               AOM_BUFFER_SIZE_FOR_BLOCK_HASH);
        assert(dst_pos < AOM_BUFFER_SIZE_FOR_BLOCK_HASH);

        to_hash[0] = buf[src_idx][srcPos];
        to_hash[1] = buf[src_idx][srcPos + 1];
        to_hash[2] = buf[src_idx][srcPos + src_sub_block_in_width];
        to_hash[3] = buf[src_idx][srcPos + src_sub_block_in_width + 1];

        // TODO: bug aomedia:433531610 - serialize input values in a way that's
        // independent of the computer architecture's endianness
        buf[dst_idx][dst_pos] =
            av1_get_crc32c_value(calc, (uint8_t *)to_hash, sizeof(to_hash));
        dst_pos++;
      }
    }

    src_sub_block_in_width = sub_block_in_width;
    sub_block_in_width >>= 1;
  }

  *hash_value1 = (buf[dst_idx][0] & crc_mask) + add_value;
  *hash_value2 = buf[dst_idx][0];
}
