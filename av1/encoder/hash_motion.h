/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#ifndef AOM_AV1_ENCODER_HASH_MOTION_H_
#define AOM_AV1_ENCODER_HASH_MOTION_H_

#include <stdbool.h>

#include "aom/aom_integer.h"
#include "aom_scale/yv12config.h"
#include "av1/encoder/hash.h"
#ifdef __cplusplus
extern "C" {
#endif

// Block size used for force_integer_mv decisions
#define FORCE_INT_MV_DECISION_BLOCK_SIZE 8

// store a block's hash info.
// x and y are the position from the top left of the picture
// hash_value2 is used to store the second hash value
typedef struct _block_hash {
  int16_t x;
  int16_t y;
  uint32_t hash_value2;
} block_hash;

// The rectangle of the frame, in luma pixels, that a hash table takes its
// blocks from. The end values are exclusive.
typedef struct {
  int x_start;
  int y_start;
  int x_end;
  int y_end;
} IntraBCHashRegion;

// Frames with more tiles than this use one table for the whole frame, to keep
// the memory down.
// TODO: bug 576320193 - Let neighboring tiles share a table when there are
// more tiles than this, instead of going back to one table.
#define MAX_INTRABC_HASH_TABLES 16

// A table keeps its buffers from frame to frame. cpi->td.mb owns it and frees
// it with the encoder. The encoder threads build the tables before the tiles
// are encoded, one table at a time per thread. After that they only read them
// through the tile_tables pointer in their own MACROBLOCK, so a table must not
// change while the tiles are encoded.
typedef struct _hash_table {
  // For each block size from 4x4 to 128x128, the number of entries in each
  // bucket and the index of the first one. Each array has 1 << 16 elements,
  // one for each bucket. NULL for the sizes not built.
  uint16_t *bucket_count[6];
  uint32_t *bucket_start[6];
  // The entries of each block size, stored bucket after bucket, and how many
  // entries each array has room for.
  block_hash *block_hashes[6];
  uint32_t block_hashes_size[6];
} hash_table;

struct intrabc_hash_info;

typedef struct intrabc_hash_info {
  // buffer for hash value calculation of a block
  // used only in av1_get_block_hash_value()
  // [two buffers used ping-pong]
  // buffers are AOM_BUFFER_SIZE_FOR_BLOCK_HASH elements long
  uint32_t *hash_value_buffer[2];
  // One table per tile, indexed by tile_row * tile_cols + tile_col. A single
  // table covers the whole frame when there is one tile, or too many.
  hash_table *tile_tables;
  int num_tile_tables;
  // Hash values of every position of the frame, only needed while the tables
  // are built (two buffers, used alternately). Each buffer has
  // frame_hash_width * frame_hash_height elements, the luma size of the source
  // frame they were allocated for. Both sizes are 0 when the buffers are freed.
  uint32_t *frame_hash_buffer[2];
  int frame_hash_width;
  int frame_hash_height;

  CRC32C crc_calculator;
  int crc_initialized;
} IntraBCHashInfo;

void av1_hash_table_init(IntraBCHashInfo *intra_bc_hash_info);
// Makes num_tables empty tables. Their buffers are kept for the next frame
// while the number of tables stays the same.
bool av1_hash_tables_create(IntraBCHashInfo *intra_bc_hash_info,
                            int num_tables);
// Frees the tables and the frame hash buffers.
void av1_hash_tables_destroy(IntraBCHashInfo *intra_bc_hash_info);
int32_t av1_hash_table_count(const hash_table *p_hash_table,
                             uint32_t hash_value);
// Returns the first of the av1_hash_table_count() entries for hash_value.
const block_hash *av1_hash_get_first_block_hash(const hash_table *p_hash_table,
                                                uint32_t hash_value);
// These fill the hash values of the blocks that lie inside region. The buffers
// use the width of the picture as stride.
void av1_generate_block_2x2_hash_value(const YV12_BUFFER_CONFIG *picture,
                                       const IntraBCHashRegion *region,
                                       uint32_t *pic_block_hash);
void av1_generate_block_hash_value(IntraBCHashInfo *intra_bc_hash_info,
                                   const YV12_BUFFER_CONFIG *picture,
                                   const IntraBCHashRegion *region,
                                   int block_size,
                                   const uint32_t *src_pic_block_hash,
                                   uint32_t *dst_pic_block_hash);
bool av1_add_to_hash_map_by_row_with_precal_data(
    hash_table *p_hash_table, const uint32_t *pic_hash, int pic_width,
    const IntraBCHashRegion *region, int block_size);

// check whether the block starts from (x_start, y_start) with the size of
// block_size x block_size has the same color in all rows
int av1_hash_is_horizontal_perfect(const YV12_BUFFER_CONFIG *picture,
                                   int block_size, int x_start, int y_start);
// check whether the block starts from (x_start, y_start) with the size of
// block_size x block_size has the same color in all columns
int av1_hash_is_vertical_perfect(const YV12_BUFFER_CONFIG *picture,
                                 int block_size, int x_start, int y_start);

void av1_get_block_hash_value(IntraBCHashInfo *intra_bc_hash_info,
                              const uint8_t *y_src, int stride, int block_size,
                              uint32_t *hash_value1, uint32_t *hash_value2,
                              int use_highbitdepth);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_ENCODER_HASH_MOTION_H_
