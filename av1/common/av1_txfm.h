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

#ifndef AOM_AV1_COMMON_AV1_TXFM_H_
#define AOM_AV1_COMMON_AV1_TXFM_H_

#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#include "config/aom_config.h"

#include "av1/common/enums.h"
#include "av1/common/blockd.h"
#include "aom/aom_integer.h"
#include "aom_dsp/aom_dsp_common.h"

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(DO_RANGE_CHECK_CLAMP)
#define DO_RANGE_CHECK_CLAMP 0
#endif

extern const int32_t av1_cospi_arr_data[4][64];
extern const int32_t av1_sinpi_arr_data[4][5];

#define MAX_TXFM_STAGE_NUM 12

static const int cos_bit_min = 10;

#define NewSqrt2Bits ((int32_t)12)
// 2^12 * sqrt(2)
static const int32_t NewSqrt2 = 5793;
// 2^12 / sqrt(2)
static const int32_t NewInvSqrt2 = 2896;

static inline const int32_t *cospi_arr(int n) {
  static const int32_t av1_cospi_arr_data_inline[4][64] = {
    { 1024, 1024, 1023, 1021, 1019, 1016, 1013, 1009, 1004, 999, 993, 987, 980,
      972,  964,  955,  946,  936,  926,  915,  903,  891,  878, 865, 851, 837,
      822,  807,  792,  775,  759,  742,  724,  706,  688,  669, 650, 630, 610,
      590,  569,  548,  526,  505,  483,  460,  438,  415,  392, 369, 345, 321,
      297,  273,  249,  224,  200,  175,  150,  125,  100,  75,  50,  25 },
    { 2048, 2047, 2046, 2042, 2038, 2033, 2026, 2018, 2009, 1998, 1987,
      1974, 1960, 1945, 1928, 1911, 1892, 1872, 1851, 1829, 1806, 1782,
      1757, 1730, 1703, 1674, 1645, 1615, 1583, 1551, 1517, 1483, 1448,
      1412, 1375, 1338, 1299, 1260, 1220, 1179, 1138, 1096, 1053, 1009,
      965,  921,  876,  830,  784,  737,  690,  642,  595,  546,  498,
      449,  400,  350,  301,  251,  201,  151,  100,  50 },
    { 4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036, 4017, 3996, 3973,
      3948, 3920, 3889, 3857, 3822, 3784, 3745, 3703, 3659, 3612, 3564,
      3513, 3461, 3406, 3349, 3290, 3229, 3166, 3102, 3035, 2967, 2896,
      2824, 2751, 2675, 2598, 2520, 2440, 2359, 2276, 2191, 2106, 2019,
      1931, 1842, 1751, 1660, 1567, 1474, 1380, 1285, 1189, 1092, 995,
      897,  799,  700,  601,  501,  401,  301,  201,  101 },
    { 8192, 8190, 8182, 8170, 8153, 8130, 8103, 8071, 8035, 7993, 7946,
      7895, 7839, 7779, 7713, 7643, 7568, 7489, 7405, 7317, 7225, 7128,
      7027, 6921, 6811, 6698, 6580, 6458, 6333, 6203, 6070, 5933, 5793,
      5649, 5501, 5351, 5197, 5040, 4880, 4717, 4551, 4383, 4212, 4038,
      3862, 3683, 3503, 3320, 3135, 2948, 2760, 2570, 2378, 2185, 1990,
      1795, 1598, 1401, 1202, 1003, 803,  603,  402,  201 }
  };
  return av1_cospi_arr_data_inline[n - cos_bit_min];
}

static inline const int32_t *sinpi_arr(int n) {
  static const int32_t av1_sinpi_arr_data_inline[4][5] = {
    { 0, 330, 621, 836, 951 },
    { 0, 660, 1241, 1672, 1901 },
    { 0, 1321, 2482, 3344, 3803 },
    { 0, 2642, 4964, 6689, 7606 }
  };
  return av1_sinpi_arr_data_inline[n - cos_bit_min];
}

// The reduced bit-width and permuted arrays are only used in the Arm Neon
// implementations in av1_fwd_txfm2d_neon.c and highbd_fwd_txfm_neon.c for now.
#if HAVE_NEON
// Store cospi/sinpi costants in Q2.13 format.
// See: https://en.wikipedia.org/wiki/Q_(number_format)
extern const int16_t av1_cospi_arr_q13_data[4][128];
extern const int16_t av1_sinpi_arr_q13_data[4][4];

extern const int32_t av1_cospi_arr_s32_data[4][66];

static inline const int16_t *cospi_arr_q13(int n) {
  return av1_cospi_arr_q13_data[n - cos_bit_min];
}

static inline const int16_t *sinpi_arr_q13(int n) {
  return av1_sinpi_arr_q13_data[n - cos_bit_min];
}

static inline const int32_t *cospi_arr_s32(int n) {
  return av1_cospi_arr_s32_data[n - cos_bit_min];
}
#endif  // HAVE_NEON

static inline int32_t range_check_value(int32_t value, int8_t bit) {
#if CONFIG_COEFFICIENT_RANGE_CHECKING
  const int64_t max_value = (1LL << (bit - 1)) - 1;
  const int64_t min_value = -(1LL << (bit - 1));
  if (value < min_value || value > max_value) {
    fprintf(stderr, "coeff out of bit range, value: %d bit %d\n", value, bit);
#if !CONFIG_AV1_ENCODER
    assert(0);
#endif
  }
#endif  // CONFIG_COEFFICIENT_RANGE_CHECKING
#if DO_RANGE_CHECK_CLAMP
  bit = AOMMIN(bit, 31);
  return clamp(value, -(1 << (bit - 1)), (1 << (bit - 1)) - 1);
#endif  // DO_RANGE_CHECK_CLAMP
  (void)bit;
  return value;
}

static inline int64_t range_check_value64(int64_t value, int8_t bit) {
#if CONFIG_COEFFICIENT_RANGE_CHECKING
  const int64_t max_value = (1LL << (bit - 1)) - 1;
  const int64_t min_value = -(1LL << (bit - 1));
  if (value < min_value || value > max_value) {
    fprintf(stderr, "coeff out of bit range, value: %" PRId64 " bit %d\n",
            value, bit);
#if !CONFIG_AV1_ENCODER
    assert(0);
#endif
  }
#endif  // CONFIG_COEFFICIENT_RANGE_CHECKING
#if DO_RANGE_CHECK_CLAMP
  bit = AOMMIN(bit, 63);
  return clamp64(value, -(1LL << (bit - 1)), (1LL << (bit - 1)) - 1);
#endif  // DO_RANGE_CHECK_CLAMP
  (void)bit;
  return value;
}

static inline int32_t round_shift(int64_t value, int bit) {
  assert(bit >= 1);
  return (int32_t)((value + (1ll << (bit - 1))) >> bit);
}

static inline int32_t half_btf(int32_t w0, int32_t in0, int32_t w1, int32_t in1,
                               int bit) {
  int64_t result_64 = (int64_t)(w0 * in0) + (int64_t)(w1 * in1);
  int64_t intermediate = result_64 + (1LL << (bit - 1));
  // NOTE(rachelbarker): The value 'result_64' may not necessarily fit
  // into 32 bits. However, the result of this function is nominally
  // ROUND_POWER_OF_TWO_64(result_64, bit)
  // and that is required to fit into stage_range[stage] many bits
  // (checked by range_check_buf()).
  //
  // Here we've unpacked that rounding operation, and it can be shown
  // that the value of 'intermediate' here *does* fit into 32 bits
  // for any conformant bitstream.
  // The upshot is that, if you do all this calculation using
  // wrapping 32-bit arithmetic instead of (non-wrapping) 64-bit arithmetic,
  // then you'll still get the correct result.
  // To provide a check on this logic, we assert that 'intermediate'
  // would fit into an int32 if range checking is enabled.
#if CONFIG_COEFFICIENT_RANGE_CHECKING
  assert(intermediate >= INT32_MIN && intermediate <= INT32_MAX);
#endif
  return (int32_t)(intermediate >> bit);
}

static inline uint16_t highbd_clip_pixel_add(uint16_t dest, tran_high_t trans,
                                             int bd) {
  return clip_pixel_highbd(dest + (int)trans, bd);
}

#if HAVE_SSE4_1 || HAVE_AVX2 || HAVE_NEON
static inline int get_log_range_out(int bd) {
  const int log_range_out = AOMMAX(16, bd + 6);
  // bd is limited by the bitstream to 12. This assert is to satisfy static
  // analyzers that may assume `log_range_out - 1` is greater than 31.
  assert(log_range_out <= 18);
  return log_range_out;
}

static inline int get_log_range(int bd, int do_cols) {
  const int log_range = AOMMAX(16, bd + (do_cols ? 6 : 8));
  // bd is limited by the bitstream to 12. This assert is to satisfy static
  // analyzers that may assume `log_range - 1` is greater than 31.
  assert(log_range <= 20);
  return log_range;
}
#endif  // HAVE_SSE4_1 || HAVE_AVX2 || HAVE_NEON

typedef void (*TxfmFunc)(const int32_t *input, int32_t *output, int8_t cos_bit,
                         const int8_t *stage_range);

typedef void (*FwdTxfm2dFunc)(const int16_t *input, int32_t *output, int stride,
                              TX_TYPE tx_type, int bd);

enum {
  TXFM_TYPE_DCT4,
  TXFM_TYPE_DCT8,
  TXFM_TYPE_DCT16,
  TXFM_TYPE_DCT32,
  TXFM_TYPE_DCT64,
  TXFM_TYPE_ADST4,
  TXFM_TYPE_ADST8,
  TXFM_TYPE_ADST16,
  TXFM_TYPE_IDENTITY4,
  TXFM_TYPE_IDENTITY8,
  TXFM_TYPE_IDENTITY16,
  TXFM_TYPE_IDENTITY32,
  TXFM_TYPES,
  TXFM_TYPE_INVALID,
} UENUM1BYTE(TXFM_TYPE);

typedef struct TXFM_2D_FLIP_CFG {
  TX_SIZE tx_size;
  int ud_flip;  // flip upside down
  int lr_flip;  // flip left to right
  const int8_t *shift;
  int8_t cos_bit_col;
  int8_t cos_bit_row;
  int8_t stage_range_col[MAX_TXFM_STAGE_NUM];
  int8_t stage_range_row[MAX_TXFM_STAGE_NUM];
  TXFM_TYPE txfm_type_col;
  TXFM_TYPE txfm_type_row;
  int stage_num_col;
  int stage_num_row;
} TXFM_2D_FLIP_CFG;

static inline void get_flip_cfg(TX_TYPE tx_type, int *ud_flip, int *lr_flip) {
  assert((unsigned int)tx_type < TX_TYPES);
  // ud_flip is 1 for FLIPADST_DCT (4), FLIPADST_FLIPADST (6),
  // FLIPADST_ADST (8), V_FLIPADST (14) -> mask 0x4150.
  // lr_flip is 1 for DCT_FLIPADST (5), FLIPADST_FLIPADST (6),
  // ADST_FLIPADST (7), H_FLIPADST (15) -> mask 0x80E0.
  *ud_flip = (0x4150 >> tx_type) & 1;
  *lr_flip = (0x80E0 >> tx_type) & 1;
}

static inline void set_flip_cfg(TX_TYPE tx_type, TXFM_2D_FLIP_CFG *cfg) {
  get_flip_cfg(tx_type, &cfg->ud_flip, &cfg->lr_flip);
}

// Utility function that returns the log of the ratio of the col and row
// sizes.
static inline int get_rect_tx_log_ratio(int col, int row) {
  if (col == row) return 0;
  if (col > row) {
    if (col == row * 2) return 1;
    if (col == row * 4) return 2;
    assert(0 && "Unsupported transform size");
  } else {
    if (row == col * 2) return -1;
    if (row == col * 4) return -2;
    assert(0 && "Unsupported transform size");
  }
  return 0;  // Invalid
}

void av1_gen_fwd_stage_range(int8_t *stage_range_col, int8_t *stage_range_row,
                             const TXFM_2D_FLIP_CFG *cfg, int bd);

void av1_gen_inv_stage_range(int8_t *stage_range_col, int8_t *stage_range_row,
                             const TXFM_2D_FLIP_CFG *cfg, TX_SIZE tx_size,
                             int bd);

void av1_get_fwd_txfm_cfg(TX_TYPE tx_type, TX_SIZE tx_size,
                          TXFM_2D_FLIP_CFG *cfg);
void av1_get_inv_txfm_cfg(TX_TYPE tx_type, TX_SIZE tx_size,
                          TXFM_2D_FLIP_CFG *cfg);
extern const TXFM_TYPE av1_txfm_type_ls[5][TX_TYPES_1D];
extern const int8_t av1_txfm_stage_num_list[TXFM_TYPES];
static inline int get_txw_idx(TX_SIZE tx_size) {
  return tx_size_wide_log2[tx_size] - tx_size_wide_log2[0];
}
static inline int get_txh_idx(TX_SIZE tx_size) {
  return tx_size_high_log2[tx_size] - tx_size_high_log2[0];
}

void av1_range_check_buf(int32_t stage, const int32_t *input,
                         const int32_t *buf, int32_t size, int8_t bit);
#define MAX_TXWH_IDX 5
#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // AOM_AV1_COMMON_AV1_TXFM_H_
