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

#ifndef AOM_AV1_COMMON_ENUMS_H_
#define AOM_AV1_COMMON_ENUMS_H_

#include "config/aom_config.h"

#include "aom/aom_codec.h"
#include "aom/aom_integer.h"
#include "aom_ports/mem.h"

#ifdef __cplusplus
extern "C" {
#endif

#undef MAX_SB_SIZE

// Max superblock size
#define MAX_SB_SIZE_LOG2 7
#define MAX_SB_SIZE (1 << MAX_SB_SIZE_LOG2)
#define MAX_SB_SQUARE (MAX_SB_SIZE * MAX_SB_SIZE)

// Min superblock size
#define MIN_SB_SIZE_LOG2 6

// Pixels per Mode Info (MI) unit
#define MI_SIZE_LOG2 2
#define MI_SIZE (1 << MI_SIZE_LOG2)

// MI-units per max superblock (MI Block - MIB)
#define MAX_MIB_SIZE_LOG2 (MAX_SB_SIZE_LOG2 - MI_SIZE_LOG2)
#define MAX_MIB_SIZE (1 << MAX_MIB_SIZE_LOG2)

// MI-units per min superblock
#define MIN_MIB_SIZE_LOG2 (MIN_SB_SIZE_LOG2 - MI_SIZE_LOG2)

// Mask to extract MI offset within max MIB
#define MAX_MIB_MASK (MAX_MIB_SIZE - 1)

// Maximum number of tile rows and tile columns
#define MAX_TILE_ROWS 64
#define MAX_TILE_COLS 64

#define MAX_VARTX_DEPTH 2

#define MI_SIZE_64X64 (64 >> MI_SIZE_LOG2)
#define MI_SIZE_128X128 (128 >> MI_SIZE_LOG2)

#define MAX_PALETTE_SQUARE (64 * 64)
// Maximum number of colors in a palette.
#define PALETTE_MAX_SIZE 8
// Minimum number of colors in a palette.
#define PALETTE_MIN_SIZE 2

#define FRAME_OFFSET_BITS 5
#define MAX_FRAME_DISTANCE ((1 << FRAME_OFFSET_BITS) - 1)

// 4 frame filter levels: y plane vertical, y plane horizontal,
// u plane, and v plane
#define FRAME_LF_COUNT 4
#define DEFAULT_DELTA_LF_MULTI 0
#define MAX_MODE_LF_DELTAS 2

#define DIST_PRECISION_BITS 4
#define DIST_PRECISION (1 << DIST_PRECISION_BITS)  // 16

#define PROFILE_BITS 3
// The following three profiles are currently defined.
// Profile 0.  8-bit and 10-bit 4:2:0 and 4:0:0 only.
// Profile 1.  8-bit and 10-bit 4:4:4
// Profile 2.  8-bit and 10-bit 4:2:2
//            12-bit  4:0:0, 4:2:2 and 4:4:4
// Since we have three bits for the profiles, it can be extended later.
enum {
  PROFILE_0,
  PROFILE_1,
  PROFILE_2,
  MAX_PROFILES,
} SENUM1BYTE(BITSTREAM_PROFILE);

#define OP_POINTS_CNT_MINUS_1_BITS 5
#define OP_POINTS_IDC_BITS 12

// Note: Some enums use the attribute 'packed' to use smallest possible integer
// type, so that we can save memory when they are used in structs/arrays.

typedef enum ATTRIBUTE_PACKED {
  BLOCK_4X4,
  BLOCK_4X8,
  BLOCK_8X4,
  BLOCK_8X8,
  BLOCK_8X16,
  BLOCK_16X8,
  BLOCK_16X16,
  BLOCK_16X32,
  BLOCK_32X16,
  BLOCK_32X32,
  BLOCK_32X64,
  BLOCK_64X32,
  BLOCK_64X64,
  BLOCK_64X128,
  BLOCK_128X64,
  BLOCK_128X128,
  BLOCK_4X16,
  BLOCK_16X4,
  BLOCK_8X32,
  BLOCK_32X8,
  BLOCK_16X64,
  BLOCK_64X16,
#if CONFIG_FLEX_PARTITION
  BLOCK_4X32,
  BLOCK_32X4,
  BLOCK_8X64,
  BLOCK_64X8,
  BLOCK_4X64,
  BLOCK_64X4,
#endif  // CONFIG_FLEX_PARTITION
  BLOCK_SIZES_ALL,
  BLOCK_SIZES = BLOCK_4X16,
  BLOCK_INVALID = 255,
  BLOCK_LARGEST = (BLOCK_SIZES - 1)
} BLOCK_SIZE;

// 4X4, 8X8, 16X16, 32X32, 64X64, 128X128
#define SQR_BLOCK_SIZES 6

//  Block partition types.  R: Recursive
//
//  NONE          HORZ          VERT          SPLIT
//  +-------+     +-------+     +---+---+     +---+---+
//  |       |     |       |     |   |   |     | R | R |
//  |       |     +-------+     |   |   |     +---+---+
//  |       |     |       |     |   |   |     | R | R |
//  +-------+     +-------+     +---+---+     +---+---+
//
//  HORZ_A        HORZ_B        VERT_A        VERT_B
//  +---+---+     +-------+     +---+---+     +---+---+
//  |   |   |     |       |     |   |   |     |   |   |
//  +---+---+     +---+---+     +---+   |     |   +---+
//  |       |     |   |   |     |   |   |     |   |   |
//  +-------+     +---+---+     +---+---+     +---+---+
//
#if CONFIG_EXT_RECUR_PARTITIONS
//  HORZ_3                 VERT_3
//  +--------------+       +---+------+---+
//  |              |       |   |      |   |
//  +--------------+       |   |      |   |
//  |              |       |   |      |   |
//  |              |       |   |      |   |
//  +--------------+       |   |      |   |
//  |              |       |   |      |   |
//  +--------------+       +---+------+---+
#else
//  HORZ_4        VERT_4
//  +-----+       +-+-+-+
//  +-----+       | | | |
//  +-----+       | | | |
//  +-----+       +-+-+-+
#endif  // CONFIG_EXT_RECUR_PARTITIONS
enum {
  PARTITION_NONE,
  PARTITION_HORZ,
  PARTITION_VERT,
#if CONFIG_EXT_RECUR_PARTITIONS
  PARTITION_HORZ_3,  // 3 horizontal sub-partitions with ratios 4:1, 2:1 and 4:1
  PARTITION_VERT_3,  // 3 vertical sub-partitions with ratios 4:1, 2:1 and 4:1
#else
  PARTITION_SPLIT,
  PARTITION_HORZ_A,  // HORZ split and the top partition is split again
  PARTITION_HORZ_B,  // HORZ split and the bottom partition is split again
  PARTITION_VERT_A,  // VERT split and the left partition is split again
  PARTITION_VERT_B,  // VERT split and the right partition is split again
  PARTITION_HORZ_4,  // 4:1 horizontal partition
  PARTITION_VERT_4,  // 4:1 vertical partition
#endif  // CONFIG_EXT_RECUR_PARTITIONS
  EXT_PARTITION_TYPES,
#if CONFIG_EXT_RECUR_PARTITIONS
  PARTITION_SPLIT = EXT_PARTITION_TYPES,
  PARTITION_TYPES = PARTITION_VERT + 1,
#else
  PARTITION_TYPES = PARTITION_SPLIT + 1,
#endif  // CONFIG_EXT_RECUR_PARTITIONS
  PARTITION_INVALID = 255
} UENUM1BYTE(PARTITION_TYPE);

typedef char PARTITION_CONTEXT;
#define PARTITION_PLOFFSET 4  // number of probability models per block size
#define PARTITION_BLOCK_SIZES 5
#define PARTITION_CONTEXTS (PARTITION_BLOCK_SIZES * PARTITION_PLOFFSET)

#if CONFIG_EXT_RECUR_PARTITIONS
enum {
  PARTITION_NONE_REC,
  PARTITION_LONG_SIDE_REC,
  PARTITION_MULTI_WAY_REC,
  PARTITION_SHORT_SIDE_REC,
  PARTITION_TYPES_REC = PARTITION_SHORT_SIDE_REC + 1,
  PARTITION_INVALID_REC = 255
} UENUM1BYTE(PARTITION_TYPE_REC);

#define PARTITION_BLOCK_SIZES_REC 5  // 128x64, 64x32, 32x16, 16x8, 8x4
#define PARTITION_CONTEXTS_REC (PARTITION_BLOCK_SIZES_REC * PARTITION_PLOFFSET)
#endif  // CONFIG_EXT_RECUR_PARTITIONS

// block transform size
enum {
  TX_4X4,    // 4x4 transform
  TX_8X8,    // 8x8 transform
  TX_16X16,  // 16x16 transform
  TX_32X32,  // 32x32 transform
  TX_64X64,  // 64x64 transform
  TX_4X8,    // 4x8 transform
  TX_8X4,    // 8x4 transform
  TX_8X16,   // 8x16 transform
  TX_16X8,   // 16x8 transform
  TX_16X32,  // 16x32 transform
  TX_32X16,  // 32x16 transform
  TX_32X64,  // 32x64 transform
  TX_64X32,  // 64x32 transform
  TX_4X16,   // 4x16 transform
  TX_16X4,   // 16x4 transform
  TX_8X32,   // 8x32 transform
  TX_32X8,   // 32x8 transform
  TX_16X64,  // 16x64 transform
  TX_64X16,  // 64x16 transform
#if CONFIG_FLEX_PARTITION
  TX_4X32,            // 4x32 transform
  TX_32X4,            // 32x4 transform
  TX_8X64,            // 8x64 transform
  TX_64X8,            // 64x8 transform
  TX_4X64,            // 4x64 transform
  TX_64X4,            // 64x4 transform
#endif                // CONFIG_FLEX_PARTITION
  TX_SIZES_ALL,       // Includes rectangular transforms
  TX_SIZES = TX_4X8,  // Does NOT include rectangular transforms
  TX_SIZES_LARGEST = TX_64X64,
  TX_INVALID = 255  // Invalid transform size
} UENUM1BYTE(TX_SIZE);

#if CONFIG_NEW_TX_PARTITION
//  Baseline transform partition types
//
//  Square:
//  NONE           SPLIT
//  +-------+      +---+---+
//  |       |      |   |   |
//  |       |      +---+---+
//  |       |      |   |   |
//  +-------+      +---+---+
//
//
//  Rectangular:
//  NONE                  SPLIT
//  +--------------+      +-------+-------+
//  |              |      |       |       |
//  |              |      +       +       +
//  |              |      |       |       |
//  +--------------+      +-------+-------+
//
//  Extended transform partition types (square and rect are the same)
//
//  NONE           SPLIT
//  +-------+      +---+---+
//  |       |      |   |   |
//  |       |      +---+---+
//  |       |      |   |   |
//  +-------+      +---+---+
//
//  HORZ                 VERT
//  +-------+      +---+---+
//  |       |      |   |   |
//  +-------+      |   |   |
//  |       |      |   |   |
//  +-------+      +---+---+
//
//  HORZ4              VERT4
//  +-------+      +-+-+-+-+
//  +-------+      | | | | |
//  +-------+      | | | | |
//  +-------+      | | | | |
//  +-------+      +-+-+-+-+
enum {
  TX_PARTITION_NONE,
  TX_PARTITION_SPLIT,
  TX_PARTITION_HORZ,
  TX_PARTITION_VERT,
  TX_PARTITION_HORZ4,
  TX_PARTITION_VERT4,
  TX_PARTITION_TYPES,
  TX_PARTITION_TYPES_INTRA = TX_PARTITION_VERT4 + 1,
} UENUM1BYTE(TX_PARTITION_TYPE);
#endif  // CONFIG_FLEXIBLE_TX

#define TX_SIZE_LUMA_MIN (TX_4X4)
/* We don't need to code a transform size unless the allowed size is at least
   one more than the minimum. */
#define TX_SIZE_CTX_MIN (TX_SIZE_LUMA_MIN + 1)

// Maximum tx_size categories
#define MAX_TX_CATS (TX_SIZES - TX_SIZE_CTX_MIN)
#define MAX_TX_DEPTH 2

#define MAX_TX_SIZE_LOG2 (6)
#define MAX_TX_SIZE (1 << MAX_TX_SIZE_LOG2)
#define MIN_TX_SIZE_LOG2 2
#define MIN_TX_SIZE (1 << MIN_TX_SIZE_LOG2)
#define MAX_TX_SQUARE (MAX_TX_SIZE * MAX_TX_SIZE)

// Pad 4 extra columns to remove horizontal availability check.
#define TX_PAD_HOR_LOG2 2
#define TX_PAD_HOR 4
// Pad 6 extra rows (2 on top and 4 on bottom) to remove vertical availability
// check.
#define TX_PAD_TOP 0
#define TX_PAD_BOTTOM 4
#define TX_PAD_VER (TX_PAD_TOP + TX_PAD_BOTTOM)
// Pad 16 extra bytes to avoid reading overflow in SIMD optimization.
#define TX_PAD_END 16
#define TX_PAD_2D ((32 + TX_PAD_HOR) * (32 + TX_PAD_VER) + TX_PAD_END)

// Number of maxium size transform blocks in the maximum size superblock
#define MAX_TX_BLOCKS_IN_MAX_SB_LOG2 ((MAX_SB_SIZE_LOG2 - MAX_TX_SIZE_LOG2) * 2)
#define MAX_TX_BLOCKS_IN_MAX_SB (1 << MAX_TX_BLOCKS_IN_MAX_SB_LOG2)

// frame transform mode
enum {
  ONLY_4X4,         // use only 4x4 transform
  TX_MODE_LARGEST,  // transform size is the largest possible for pu size
  TX_MODE_SELECT,   // transform specified for each block
  TX_MODES,
} UENUM1BYTE(TX_MODE);

// 1D tx types
enum {
  DCT_1D,
  ADST_1D,
  FLIPADST_1D,
  IDTX_1D,
#if CONFIG_MODE_DEP_INTRA_TX || CONFIG_MODE_DEP_INTER_TX
  MDTX1_1D,
  NSTX,  // this is a placeholder
#endif   // CONFIG_MODE_DEP_INTRA_TX || CONFIG_MODE_DEP_INTER_TX
  TX_TYPES_1D,
} UENUM1BYTE(TX_TYPE_1D);

#if CONFIG_MODE_DEP_INTRA_TX || CONFIG_MODE_DEP_INTER_TX
// If CONFIG_MODE_DEP_NONSEP_SEC_INTRA_TX is 0, apply non-separable transforms
// on 4x4, 4x8, 8x4, and 8x8 blocks. If it is 1, apply non-separable transforms
// on 4x4 blocks, and DCT_DCT with secondary transforms on 4x8, 8x4, and 8x8
// blocks
#define TX_TYPES_NOMDTX 16
#define MDTX_TYPES_INTER 8

#if CONFIG_MODE_DEP_INTRA_TX && CONFIG_MODE_DEP_NONSEP_INTRA_TX
#define MDTX_TYPES_INTRA 4
#define MDTX_DEBUG 0
#else
#define MDTX_TYPES_INTRA 3
#endif  // CONFIG_MODE_DEP_INTRA_TX && CONFIG_MODE_DEP_NONSEP_INTRA_TX
#endif  // CONFIG_MODE_DEP_INTRA_TX || CONFIG_MODE_DEP_INTER_TX

enum {
  DCT_DCT,            // DCT in both horizontal and vertical
  ADST_DCT,           // ADST in vertical, DCT in horizontal
  DCT_ADST,           // DCT in vertical, ADST in horizontal
  ADST_ADST,          // ADST in both directions
  FLIPADST_DCT,       // FLIPADST in vertical, DCT in horizontal
  DCT_FLIPADST,       // DCT in vertical, FLIPADST in horizontal
  FLIPADST_FLIPADST,  // FLIPADST in both directions
  ADST_FLIPADST,      // ADST in vertical, FLIPADST in horizontal
  FLIPADST_ADST,      // FLIPADST in vertical, ADST in horizontal
  IDTX,               // Identity in both directions
  V_DCT,              // DCT in vertical, identity in horizontal
  H_DCT,              // Identity in vertical, DCT in horizontal
  V_ADST,             // ADST in vertical, identity in horizontal
  H_ADST,             // Identity in vertical, ADST in horizontal
  V_FLIPADST,         // FLIPADST in vertical, identity in horizontal
  H_FLIPADST,         // Identity in vertical, FLIPADST in horizontal
#if CONFIG_MODE_DEP_INTRA_TX
  // 3 mode-dependent tx for intra
  MDTX_INTRA_1,  // MDTX in both horizontal and vertical
  MDTX_INTRA_2,  // MDTX in vertical, DCT in horizontal
  MDTX_INTRA_3,  // DCT in vertical, MDTX in horizontal
#if CONFIG_MODE_DEP_NONSEP_INTRA_TX
  MDTX_INTRA_4,  // non-separable MDTX
#endif           // CONFIG_MODE_DEP_NONSEP_INTRA_TX
#endif           // CONFIG_MODE_DEP_INTRA_TX
#if CONFIG_MODE_DEP_INTER_TX
  // 8 mode-dependent tx for inter
  MDTX_INTER_1,  // MDTX in both horizontal and vertical
  MDTX_INTER_2,  // MDTX in vertical, DCT in horizontal
  MDTX_INTER_3,  // DCT in vertical, MDTX in horizontal
  MDTX_INTER_4,  // flipped MDTX in both horizontal and vertical
  MDTX_INTER_5,  // flipped MDTX in vertical, DCT in horizontal
  MDTX_INTER_6,  // DCT in vertical, flipped MDTX in horizontal
  MDTX_INTER_7,  // flipped MDTX in vertical, MDTX in horizontal
  MDTX_INTER_8,  // MDTX in vertical, flipped MDTX in horizontal
#endif           // CONFIG_MODE_DEP_INTER_TX
  TX_TYPES,
} UENUM1BYTE(TX_TYPE);

enum {
  REG_REG,
  REG_SMOOTH,
  REG_SHARP,
  SMOOTH_REG,
  SMOOTH_SMOOTH,
  SMOOTH_SHARP,
  SHARP_REG,
  SHARP_SMOOTH,
  SHARP_SHARP,
} UENUM1BYTE(DUAL_FILTER_TYPE);

enum {
  // DCT only
  EXT_TX_SET_DCTONLY,
  // DCT + Identity only
  EXT_TX_SET_DCT_IDTX,
  // Discrete Trig transforms w/o flip (4) + Identity (1)
  EXT_TX_SET_DTT4_IDTX,
#if CONFIG_MODE_DEP_INTRA_TX
  // Discrete Trig transforms w/o flip (4) + Identity (1) + 1D Hor/vert DCT (2)
  //  + DCT w/ 1 MDTX (2) + MDTX1_MDTX1 (1)
  EXT_TX_SET_DTT4_IDTX_1DDCT_MDTX4,
#else
  // Discrete Trig transforms w/o flip (4) + Identity (1) + 1D Hor/vert DCT (2)
  EXT_TX_SET_DTT4_IDTX_1DDCT,
#endif  // CONFIG_MODE_DEP_INTRA_TX
  // Discrete Trig transforms w/ flip (9) + Identity (1) + 1D Hor/Ver DCT (2)
  EXT_TX_SET_DTT9_IDTX_1DDCT,
#if CONFIG_MODE_DEP_INTER_TX
  // Discrete Trig transforms w/ flip (9) + Identity (1) + 1D Hor/Ver (6)
  //  + DCT w/ 2 MDTXs (4) + 2 MDTXs (4)
  EXT_TX_SET_ALL16_MDTX8,
#else
  // Discrete Trig transforms w/ flip (9) + Identity (1) + 1D Hor/Ver (6)
  EXT_TX_SET_ALL16,
#endif  // CONFIG_MODE_DEP_INTER_TX
  EXT_TX_SET_TYPES
} UENUM1BYTE(TxSetType);

#if CONFIG_MODE_DEP_INTRA_TX || CONFIG_MODE_DEP_INTER_TX
#define IS_2D_TRANSFORM(tx_type) (tx_type < IDTX || tx_type > H_FLIPADST)
#else
#define IS_2D_TRANSFORM(tx_type) (tx_type < IDTX)
#endif  // CONFIG_MODE_DEP_INTRA_TX || CONFIG_MODE_DEP_INTER_TX

#define EXT_TX_SIZES 4       // number of sizes that use extended transforms
#define EXT_TX_SETS_INTER 4  // Sets of transform selections for INTER
#define EXT_TX_SETS_INTRA 3  // Sets of transform selections for INTRA

enum {
  AOM_LAST_FLAG = 1 << 0,
  AOM_LAST2_FLAG = 1 << 1,
  AOM_LAST3_FLAG = 1 << 2,
  AOM_GOLD_FLAG = 1 << 3,
  AOM_BWD_FLAG = 1 << 4,
  AOM_ALT2_FLAG = 1 << 5,
  AOM_ALT_FLAG = 1 << 6,
  AOM_REFFRAME_ALL = (1 << 7) - 1
} UENUM1BYTE(AOM_REFFRAME);

enum {
  UNIDIR_COMP_REFERENCE,
  BIDIR_COMP_REFERENCE,
  COMP_REFERENCE_TYPES,
} UENUM1BYTE(COMP_REFERENCE_TYPE);

enum { PLANE_TYPE_Y, PLANE_TYPE_UV, PLANE_TYPES } UENUM1BYTE(PLANE_TYPE);

#define CFL_ALPHABET_SIZE_LOG2 4
#define CFL_ALPHABET_SIZE (1 << CFL_ALPHABET_SIZE_LOG2)
#define CFL_MAGS_SIZE ((2 << CFL_ALPHABET_SIZE_LOG2) + 1)
#define CFL_IDX_U(idx) (idx >> CFL_ALPHABET_SIZE_LOG2)
#define CFL_IDX_V(idx) (idx & (CFL_ALPHABET_SIZE - 1))

enum { CFL_PRED_U, CFL_PRED_V, CFL_PRED_PLANES } UENUM1BYTE(CFL_PRED_TYPE);

enum {
  CFL_SIGN_ZERO,
  CFL_SIGN_NEG,
  CFL_SIGN_POS,
  CFL_SIGNS
} UENUM1BYTE(CFL_SIGN_TYPE);

enum {
  CFL_DISALLOWED,
  CFL_ALLOWED,
  CFL_ALLOWED_TYPES
} UENUM1BYTE(CFL_ALLOWED_TYPE);

// CFL_SIGN_ZERO,CFL_SIGN_ZERO is invalid
#define CFL_JOINT_SIGNS (CFL_SIGNS * CFL_SIGNS - 1)
// CFL_SIGN_U is equivalent to (js + 1) / 3 for js in 0 to 8
#define CFL_SIGN_U(js) (((js + 1) * 11) >> 5)
// CFL_SIGN_V is equivalent to (js + 1) % 3 for js in 0 to 8
#define CFL_SIGN_V(js) ((js + 1) - CFL_SIGNS * CFL_SIGN_U(js))

// There is no context when the alpha for a given plane is zero.
// So there are 2 fewer contexts than joint signs.
#define CFL_ALPHA_CONTEXTS (CFL_JOINT_SIGNS + 1 - CFL_SIGNS)
#define CFL_CONTEXT_U(js) (js + 1 - CFL_SIGNS)
// Also, the contexts are symmetric under swapping the planes.
#define CFL_CONTEXT_V(js) \
  (CFL_SIGN_V(js) * CFL_SIGNS + CFL_SIGN_U(js) - CFL_SIGNS)

enum {
  PALETTE_MAP,
  COLOR_MAP_TYPES,
} UENUM1BYTE(COLOR_MAP_TYPE);

enum {
  TWO_COLORS,
  THREE_COLORS,
  FOUR_COLORS,
  FIVE_COLORS,
  SIX_COLORS,
  SEVEN_COLORS,
  EIGHT_COLORS,
  PALETTE_SIZES
} UENUM1BYTE(PALETTE_SIZE);

enum {
  PALETTE_COLOR_ONE,
  PALETTE_COLOR_TWO,
  PALETTE_COLOR_THREE,
  PALETTE_COLOR_FOUR,
  PALETTE_COLOR_FIVE,
  PALETTE_COLOR_SIX,
  PALETTE_COLOR_SEVEN,
  PALETTE_COLOR_EIGHT,
  PALETTE_COLORS
} UENUM1BYTE(PALETTE_COLOR);

// Note: All directional predictors must be between V_PRED and D67_PRED (both
// inclusive).
enum {
  DC_PRED,        // Average of above and left pixels
  V_PRED,         // Vertical
  H_PRED,         // Horizontal
  D45_PRED,       // Directional 45  degree
  D135_PRED,      // Directional 135 degree
  D113_PRED,      // Directional 113 degree
  D157_PRED,      // Directional 157 degree
  D203_PRED,      // Directional 203 degree
  D67_PRED,       // Directional 67  degree
  SMOOTH_PRED,    // Combination of horizontal and vertical interpolation
  SMOOTH_V_PRED,  // Vertical interpolation
  SMOOTH_H_PRED,  // Horizontal interpolation
  PAETH_PRED,     // Predict from the direction of smallest gradient
#if !CONFIG_NEW_INTER_MODES
  NEARESTMV,
#endif  // !CONFIG_NEW_INTER_MODES
  NEARMV,
  GLOBALMV,
  NEWMV,
// Compound ref compound modes
#if !CONFIG_NEW_INTER_MODES
  NEAREST_NEARESTMV,
#endif  // !CONFIG_NEW_INTER_MODES
  NEAR_NEARMV,
#if !CONFIG_NEW_INTER_MODES
  NEAREST_NEWMV,
  NEW_NEARESTMV,
#endif  // !CONFIG_NEW_INTER_MODES
  NEAR_NEWMV,
  NEW_NEARMV,
  GLOBAL_GLOBALMV,
  NEW_NEWMV,
#if CONFIG_EXT_COMPOUND
  NEAR_SCALEDMV,
  SCALED_NEARMV,
  NEW_SCALEDMV,
  SCALED_NEWMV,
#endif  // CONFIG_EXT_COMPOUND
  MB_MODE_COUNT,
  INTRA_MODE_START = DC_PRED,
#if CONFIG_NEW_INTER_MODES
  INTRA_MODE_END = NEARMV,
#else
  INTRA_MODE_END = NEARESTMV,
#endif  // CONFIG_NEW_INTER_MODES
  INTRA_MODE_NUM = INTRA_MODE_END - INTRA_MODE_START,
#if CONFIG_NEW_INTER_MODES
  SINGLE_INTER_MODE_START = NEARMV,
  SINGLE_INTER_MODE_END = NEAR_NEARMV,
#else
  SINGLE_INTER_MODE_START = NEARESTMV,
  SINGLE_INTER_MODE_END = NEAREST_NEARESTMV,
#endif  // CONFIG_NEW_INTER_MODES
  SINGLE_INTER_MODE_NUM = SINGLE_INTER_MODE_END - SINGLE_INTER_MODE_START,
#if CONFIG_NEW_INTER_MODES
  COMP_INTER_MODE_START = NEAR_NEARMV,
#else
  COMP_INTER_MODE_START = NEAREST_NEARESTMV,
#endif  // CONFIG_NEW_INTER_MODES
  COMP_INTER_MODE_END = MB_MODE_COUNT,
  COMP_INTER_MODE_NUM = COMP_INTER_MODE_END - COMP_INTER_MODE_START,
#if CONFIG_NEW_INTER_MODES
  INTER_MODE_START = NEARMV,
#else
  INTER_MODE_START = NEARESTMV,
#endif  // CONFIG_NEW_INTER_MODES
  INTER_MODE_END = MB_MODE_COUNT,
  INTRA_MODES = PAETH_PRED + 1,  // PAETH_PRED has to be the last intra mode.
  INTRA_INVALID = MB_MODE_COUNT  // For uv_mode in inter blocks
} UENUM1BYTE(PREDICTION_MODE);

// TODO(ltrudeau) Do we really want to pack this?
// TODO(ltrudeau) Do we match with PREDICTION_MODE?
enum {
  UV_DC_PRED,        // Average of above and left pixels
  UV_V_PRED,         // Vertical
  UV_H_PRED,         // Horizontal
  UV_D45_PRED,       // Directional 45  degree
  UV_D135_PRED,      // Directional 135 degree
  UV_D113_PRED,      // Directional 113 degree
  UV_D157_PRED,      // Directional 157 degree
  UV_D203_PRED,      // Directional 203 degree
  UV_D67_PRED,       // Directional 67  degree
  UV_SMOOTH_PRED,    // Combination of horizontal and vertical interpolation
  UV_SMOOTH_V_PRED,  // Vertical interpolation
  UV_SMOOTH_H_PRED,  // Horizontal interpolation
  UV_PAETH_PRED,     // Predict from the direction of smallest gradient
  UV_CFL_PRED,       // Chroma-from-Luma
  UV_INTRA_MODES,
  UV_MODE_INVALID,  // For uv_mode in inter blocks
} UENUM1BYTE(UV_PREDICTION_MODE);

enum {
  SIMPLE_TRANSLATION,
  OBMC_CAUSAL,    // 2-sided OBMC
  WARPED_CAUSAL,  // 2-sided WARPED
  MOTION_MODES
} UENUM1BYTE(MOTION_MODE);

enum {
  II_DC_PRED,
  II_V_PRED,
  II_H_PRED,
  II_SMOOTH_PRED,
#if CONFIG_ILLUM_MCOMP
  II_ILLUM_MCOMP_PRED,
#endif  // CONFIG_ILLUM_MCOMP
  INTERINTRA_MODES
} UENUM1BYTE(INTERINTRA_MODE);

enum {
  COMPOUND_AVERAGE,
  COMPOUND_DISTWTD,
  COMPOUND_WEDGE,
  COMPOUND_DIFFWTD,
  COMPOUND_TYPES,
  MASKED_COMPOUND_TYPES = 2,
} UENUM1BYTE(COMPOUND_TYPE);

enum {
  FILTER_DC_PRED,
  FILTER_V_PRED,
  FILTER_H_PRED,
  FILTER_D157_PRED,
  FILTER_PAETH_PRED,
  FILTER_INTRA_MODES,
} UENUM1BYTE(FILTER_INTRA_MODE);

// ADAPT_FILTER_INTRA experiment introduces a new group of intra
// prediction modes similar to FILTER_INTRA. Key difference
// is that the new modes adaptively fit the filter weights for
// each transform unit instead of using predefined filter weights
// obtained by offline training. A training region is selected
// within the set of already reconstructed pixels and a least-squares
// problem is solved to obtain the adaptive filter weights both in
// the encoder and decoder.
//
// Provides roughly 1.1% gain on top of FILTER_INTRA. When
// ADAPT_FILTER_INTRA is turned on, the gain from FILTER_INTRA
// drops to below 0.2%.
#if CONFIG_ADAPT_FILTER_INTRA
// Different ADAPT_FILTER_INTRA modes fit different filters
// (either 3 tap or 4 tap). All of them use some subset of
// the 5 neighbor pixels of the one currently being predicted.
// We enumerate neighbor pixels in the clockwise order:
// -------------
// | 2 | 3 | 4 |
// -------------
// | 1 |cur|   |
// -------------
// | 0 |   |   |
// -------------
// For example, ADAPT_FILTER_1_2_3 fits a 3 tap filter that
// uses neighbor pixels with numbers 1,2,3 for prediction.
// ADAPT_FILTER_0_1_2_3_4_LEFT fits a corresponding 4 tap
// filter, but uses only left context of the block for
// training, according to the postfix "_LEFT".
typedef enum {
  ADAPT_FILTER_1_2_3,
  ADAPT_FILTER_0_1_3,
  ADAPT_FILTER_1_3_4,
  ADAPT_FILTER_0_1_2_3_4_LEFT,
  ADAPT_FILTER_1_2_3_4_TOP,
  ADAPT_FILTER_0_2_3,
  ADAPT_FILTER_1_2_4,
  ADAPT_FILTER_INTRA_MODES,
} ADAPT_FILTER_INTRA_MODE;
#define USED_ADAPT_FILTER_INTRA_MODES 7
#endif  // CONFIG_ADAPT_FILTER_INTRA

enum {
  SEQ_LEVEL_2_0,
  SEQ_LEVEL_2_1,
  SEQ_LEVEL_2_2,
  SEQ_LEVEL_2_3,
  SEQ_LEVEL_3_0,
  SEQ_LEVEL_3_1,
  SEQ_LEVEL_3_2,
  SEQ_LEVEL_3_3,
  SEQ_LEVEL_4_0,
  SEQ_LEVEL_4_1,
  SEQ_LEVEL_4_2,
  SEQ_LEVEL_4_3,
  SEQ_LEVEL_5_0,
  SEQ_LEVEL_5_1,
  SEQ_LEVEL_5_2,
  SEQ_LEVEL_5_3,
  SEQ_LEVEL_6_0,
  SEQ_LEVEL_6_1,
  SEQ_LEVEL_6_2,
  SEQ_LEVEL_6_3,
  SEQ_LEVEL_7_0,
  SEQ_LEVEL_7_1,
  SEQ_LEVEL_7_2,
  SEQ_LEVEL_7_3,
  SEQ_LEVELS,
  SEQ_LEVEL_MAX = 31
} UENUM1BYTE(AV1_LEVEL);

#define LEVEL_BITS 5

#define DIRECTIONAL_MODES 8
#if CONFIG_DERIVED_INTRA_MODE
#define NONE_DIRECTIONAL_MODES (INTRA_MODES - DIRECTIONAL_MODES)
#endif  // CONFIG_DERIVED_INTRA_MODE
#define MAX_ANGLE_DELTA 3
#define ANGLE_STEP 3

#define INTER_MODES SINGLE_INTER_MODE_NUM

#define INTER_COMPOUND_MODES COMP_INTER_MODE_NUM

#define SKIP_CONTEXTS 3
#define SKIP_MODE_CONTEXTS 3

#define COMP_INDEX_CONTEXTS 6
#define COMP_GROUP_IDX_CONTEXTS 6

#define NMV_CONTEXTS 3

#define NEWMV_MODE_CONTEXTS 6
#define GLOBALMV_MODE_CONTEXTS 2
#if CONFIG_NEW_INTER_MODES
#define MAX_DRL_BITS 5
#else
// This constant can't be changed to tweak the DRL search size without
// NEW_INTER_MODES
#define MAX_DRL_BITS 2
#define REFMV_MODE_CONTEXTS 6
#endif  // CONFIG_NEW_INTER_MODES
#define DRL_MODE_CONTEXTS 3

#define GLOBALMV_OFFSET 3
#define REFMV_OFFSET 4

#define NEWMV_CTX_MASK ((1 << GLOBALMV_OFFSET) - 1)
#define GLOBALMV_CTX_MASK ((1 << (REFMV_OFFSET - GLOBALMV_OFFSET)) - 1)
#define REFMV_CTX_MASK ((1 << (8 - REFMV_OFFSET)) - 1)

#define COMP_NEWMV_CTXS 5
#define INTER_MODE_CONTEXTS 8

#define DELTA_Q_SMALL 3
#define DELTA_Q_PROBS (DELTA_Q_SMALL)
#define DEFAULT_DELTA_Q_RES_PERCEPTUAL 4
#define DEFAULT_DELTA_Q_RES_OBJECTIVE 4

#define DELTA_LF_SMALL 3
#define DELTA_LF_PROBS (DELTA_LF_SMALL)
#define DEFAULT_DELTA_LF_RES 2

/* Segment Feature Masks */
#define MAX_MV_REF_CANDIDATES 2

#define MAX_REF_MV_STACK_SIZE 8
#define REF_CAT_LEVEL 640

#define INTRA_INTER_CONTEXTS 4
#define COMP_INTER_CONTEXTS 5
#define REF_CONTEXTS 3

#define COMP_REF_TYPE_CONTEXTS 5
#define UNI_COMP_REF_CONTEXTS 3

#define TXFM_PARTITION_CONTEXTS ((TX_SIZES - TX_8X8) * 6 - 3)
typedef uint8_t TXFM_CONTEXT;

// An enum for single reference types (and some derived values).
enum {
  NONE_FRAME = -1,
  INTRA_FRAME,
  LAST_FRAME,
  LAST2_FRAME,
  LAST3_FRAME,
  GOLDEN_FRAME,
  BWDREF_FRAME,
  ALTREF2_FRAME,
  ALTREF_FRAME,
  REF_FRAMES,

  // Extra/scratch reference frame. It may be:
  // - used to update the ALTREF2_FRAME ref (see lshift_bwd_ref_frames()), or
  // - updated from ALTREF2_FRAME ref (see rshift_bwd_ref_frames()).
  EXTREF_FRAME = REF_FRAMES,

  // Number of inter (non-intra) reference types.
  INTER_REFS_PER_FRAME = ALTREF_FRAME - LAST_FRAME + 1,

  // Number of forward (aka past) reference types.
  FWD_REFS = GOLDEN_FRAME - LAST_FRAME + 1,

  // Number of backward (aka future) reference types.
  BWD_REFS = ALTREF_FRAME - BWDREF_FRAME + 1,

  SINGLE_REFS = FWD_REFS + BWD_REFS,
};

#define REF_FRAMES_LOG2 3

// REF_FRAMES for the cm->ref_frame_map array, 1 scratch frame for the new
// frame in cm->cur_frame, INTER_REFS_PER_FRAME for scaled references on the
// encoder in the cpi->scaled_ref_buf array.
#define FRAME_BUFFERS (REF_FRAMES + 1 + INTER_REFS_PER_FRAME)

#define FWD_RF_OFFSET(ref) (ref - LAST_FRAME)
#define BWD_RF_OFFSET(ref) (ref - BWDREF_FRAME)

enum {
  LAST_LAST2_FRAMES,      // { LAST_FRAME, LAST2_FRAME }
  LAST_LAST3_FRAMES,      // { LAST_FRAME, LAST3_FRAME }
  LAST_GOLDEN_FRAMES,     // { LAST_FRAME, GOLDEN_FRAME }
  BWDREF_ALTREF_FRAMES,   // { BWDREF_FRAME, ALTREF_FRAME }
  LAST2_LAST3_FRAMES,     // { LAST2_FRAME, LAST3_FRAME }
  LAST2_GOLDEN_FRAMES,    // { LAST2_FRAME, GOLDEN_FRAME }
  LAST3_GOLDEN_FRAMES,    // { LAST3_FRAME, GOLDEN_FRAME }
  BWDREF_ALTREF2_FRAMES,  // { BWDREF_FRAME, ALTREF2_FRAME }
  ALTREF2_ALTREF_FRAMES,  // { ALTREF2_FRAME, ALTREF_FRAME }
  TOTAL_UNIDIR_COMP_REFS,
  // NOTE: UNIDIR_COMP_REFS is the number of uni-directional reference pairs
  //       that are explicitly signaled.
  UNIDIR_COMP_REFS = BWDREF_ALTREF_FRAMES + 1,
} UENUM1BYTE(UNIDIR_COMP_REF);

#define TOTAL_COMP_REFS (FWD_REFS * BWD_REFS + TOTAL_UNIDIR_COMP_REFS)

#define COMP_REFS (FWD_REFS * BWD_REFS + UNIDIR_COMP_REFS)

// NOTE: A limited number of unidirectional reference pairs can be signalled for
//       compound prediction. The use of skip mode, on the other hand, makes it
//       possible to have a reference pair not listed for explicit signaling.
#define MODE_CTX_REF_FRAMES (REF_FRAMES + TOTAL_COMP_REFS)

// Note: It includes single and compound references. So, it can take values from
// NONE_FRAME to (MODE_CTX_REF_FRAMES - 1). Hence, it is not defined as an enum.
typedef int8_t MV_REFERENCE_FRAME;

enum {
  RESTORE_NONE,
  RESTORE_WIENER,
  RESTORE_SGRPROJ,
#if CONFIG_LOOP_RESTORE_CNN
  RESTORE_CNN,
#endif  // CONFIG_LOOP_RESTORE_CNN
#if CONFIG_WIENER_NONSEP
  RESTORE_WIENER_NONSEP,
#endif  // CONFIG_WIENER_NONSEP
  RESTORE_SWITCHABLE,
  RESTORE_SWITCHABLE_TYPES = RESTORE_SWITCHABLE,
  RESTORE_TYPES = RESTORE_SWITCHABLE + 1,
} UENUM1BYTE(RestorationType);

// Picture prediction structures (0-12 are predefined) in scalability metadata.
enum {
  SCALABILITY_L1T2 = 0,
  SCALABILITY_L1T3 = 1,
  SCALABILITY_L2T1 = 2,
  SCALABILITY_L2T2 = 3,
  SCALABILITY_L2T3 = 4,
  SCALABILITY_S2T1 = 5,
  SCALABILITY_S2T2 = 6,
  SCALABILITY_S2T3 = 7,
  SCALABILITY_L2T1h = 8,
  SCALABILITY_L2T2h = 9,
  SCALABILITY_L2T3h = 10,
  SCALABILITY_S2T1h = 11,
  SCALABILITY_S2T2h = 12,
  SCALABILITY_S2T3h = 13,
  SCALABILITY_SS = 14
} UENUM1BYTE(SCALABILITY_STRUCTURES);

#define SUPERRES_SCALE_BITS 3
#define SUPERRES_SCALE_DENOMINATOR_MIN (SCALE_NUMERATOR + 1)

// In large_scale_tile coding, external references are used.
#define MAX_EXTERNAL_REFERENCES 128
#define MAX_TILES 512

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_ENUMS_H_
