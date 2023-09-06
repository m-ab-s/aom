/*
 * Copyright (c) 2017, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include "config/aom_dsp_rtcd.h"
#include "config/av1_rtcd.h"

#include "av1/common/av1_txfm.h"

// av1_cospi_arr[i][j] = (int)round(cos(PI*j/128) * (1<<(cos_bit_min+i)));
const int32_t av1_cospi_arr_data[4][64] = {
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

// av1_sinpi_arr_data[i][j] = (int)round((sqrt(2) * sin(j*Pi/9) * 2 / 3) * (1
// << (cos_bit_min + i))) modified so that elements j=1,2 sum to element j=4.
const int32_t av1_sinpi_arr_data[4][5] = { { 0, 330, 621, 836, 951 },
                                           { 0, 660, 1241, 1672, 1901 },
                                           { 0, 1321, 2482, 3344, 3803 },
                                           { 0, 2642, 4964, 6689, 7606 } };

// The reduced bit-width arrays are only used in the Arm Neon implementations
// in av1_fwd_txfm2d_neon.c for now.
#if HAVE_NEON
// Constants are stored in groups of four, where symmetrical constants in the
// cospi array are stored adjacent in memory, followed immediately by the same
// constants but negated, i.e.:
//   f(i,j) = (int)round(cos(PI*j/128) * (1<<(cos_bit_min+i))) << (3-i)
//   av1_cospi_arr_q13_data[i][4*j+0] = f(i,j)
//   av1_cospi_arr_q13_data[i][4*j+1] = f(i,63-j)
//   av1_cospi_arr_q13_data[i][4*j+2] = -av1_cospi_arr_q13_data[i][4*j+0]
//   av1_cospi_arr_q13_data[i][4*j+3] = -av1_cospi_arr_q13_data[i][4*j+1]
// See also: https://en.wikipedia.org/wiki/Q_(number_format)
const int16_t
    av1_cospi_arr_q13_data[4][132] = {
      { 8192,  0,     -8192, 0,     8192,  200,   -8192, -200,  8184,  400,
        -8184, -400,  8168,  600,   -8168, -600,  8152,  800,   -8152, -800,
        8128,  1000,  -8128, -1000, 8104,  1200,  -8104, -1200, 8072,  1400,
        -8072, -1400, 8032,  1600,  -8032, -1600, 7992,  1792,  -7992, -1792,
        7944,  1992,  -7944, -1992, 7896,  2184,  -7896, -2184, 7840,  2376,
        -7840, -2376, 7776,  2568,  -7776, -2568, 7712,  2760,  -7712, -2760,
        7640,  2952,  -7640, -2952, 7568,  3136,  -7568, -3136, 7488,  3320,
        -7488, -3320, 7408,  3504,  -7408, -3504, 7320,  3680,  -7320, -3680,
        7224,  3864,  -7224, -3864, 7128,  4040,  -7128, -4040, 7024,  4208,
        -7024, -4208, 6920,  4384,  -6920, -4384, 6808,  4552,  -6808, -4552,
        6696,  4720,  -6696, -4720, 6576,  4880,  -6576, -4880, 6456,  5040,
        -6456, -5040, 6336,  5200,  -6336, -5200, 6200,  5352,  -6200, -5352,
        6072,  5504,  -6072, -5504, 5936,  5648,  -5936, -5648, 5792,  5792,
        -5792, -5792 },
      { 8192,  0,     -8192, 0,     8188,  200,   -8188, -200,  8184,  400,
        -8184, -400,  8168,  604,   -8168, -604,  8152,  804,   -8152, -804,
        8132,  1004,  -8132, -1004, 8104,  1204,  -8104, -1204, 8072,  1400,
        -8072, -1400, 8036,  1600,  -8036, -1600, 7992,  1796,  -7992, -1796,
        7948,  1992,  -7948, -1992, 7896,  2184,  -7896, -2184, 7840,  2380,
        -7840, -2380, 7780,  2568,  -7780, -2568, 7712,  2760,  -7712, -2760,
        7644,  2948,  -7644, -2948, 7568,  3136,  -7568, -3136, 7488,  3320,
        -7488, -3320, 7404,  3504,  -7404, -3504, 7316,  3684,  -7316, -3684,
        7224,  3860,  -7224, -3860, 7128,  4036,  -7128, -4036, 7028,  4212,
        -7028, -4212, 6920,  4384,  -6920, -4384, 6812,  4552,  -6812, -4552,
        6696,  4716,  -6696, -4716, 6580,  4880,  -6580, -4880, 6460,  5040,
        -6460, -5040, 6332,  5196,  -6332, -5196, 6204,  5352,  -6204, -5352,
        6068,  5500,  -6068, -5500, 5932,  5648,  -5932, -5648, 5792,  5792,
        -5792, -5792 },
      { 8192,  0,     -8192, 0,     8190,  202,   -8190, -202,  8182,  402,
        -8182, -402,  8170,  602,   -8170, -602,  8152,  802,   -8152, -802,
        8130,  1002,  -8130, -1002, 8104,  1202,  -8104, -1202, 8072,  1400,
        -8072, -1400, 8034,  1598,  -8034, -1598, 7992,  1794,  -7992, -1794,
        7946,  1990,  -7946, -1990, 7896,  2184,  -7896, -2184, 7840,  2378,
        -7840, -2378, 7778,  2570,  -7778, -2570, 7714,  2760,  -7714, -2760,
        7644,  2948,  -7644, -2948, 7568,  3134,  -7568, -3134, 7490,  3320,
        -7490, -3320, 7406,  3502,  -7406, -3502, 7318,  3684,  -7318, -3684,
        7224,  3862,  -7224, -3862, 7128,  4038,  -7128, -4038, 7026,  4212,
        -7026, -4212, 6922,  4382,  -6922, -4382, 6812,  4552,  -6812, -4552,
        6698,  4718,  -6698, -4718, 6580,  4880,  -6580, -4880, 6458,  5040,
        -6458, -5040, 6332,  5196,  -6332, -5196, 6204,  5350,  -6204, -5350,
        6070,  5502,  -6070, -5502, 5934,  5648,  -5934, -5648, 5792,  5792,
        -5792, -5792 },
      { 8192,  0,     -8192, 0,     8190,  201,   -8190, -201,  8182,  402,
        -8182, -402,  8170,  603,   -8170, -603,  8153,  803,   -8153, -803,
        8130,  1003,  -8130, -1003, 8103,  1202,  -8103, -1202, 8071,  1401,
        -8071, -1401, 8035,  1598,  -8035, -1598, 7993,  1795,  -7993, -1795,
        7946,  1990,  -7946, -1990, 7895,  2185,  -7895, -2185, 7839,  2378,
        -7839, -2378, 7779,  2570,  -7779, -2570, 7713,  2760,  -7713, -2760,
        7643,  2948,  -7643, -2948, 7568,  3135,  -7568, -3135, 7489,  3320,
        -7489, -3320, 7405,  3503,  -7405, -3503, 7317,  3683,  -7317, -3683,
        7225,  3862,  -7225, -3862, 7128,  4038,  -7128, -4038, 7027,  4212,
        -7027, -4212, 6921,  4383,  -6921, -4383, 6811,  4551,  -6811, -4551,
        6698,  4717,  -6698, -4717, 6580,  4880,  -6580, -4880, 6458,  5040,
        -6458, -5040, 6333,  5197,  -6333, -5197, 6203,  5351,  -6203, -5351,
        6070,  5501,  -6070, -5501, 5933,  5649,  -5933, -5649, 5793,  5793,
        -5793, -5793 },
    };

// av1_sinpi_arr_q13_data[i][j] =
//   round((sqrt2 * sin((j+1)*Pi/9) * 2/3) * (1 << (cos_bit_min + i))) << (3-i)
// modified so that elements j=0,1 sum to element j=3.
// See also: https://en.wikipedia.org/wiki/Q_(number_format)
const int16_t av1_sinpi_arr_q13_data[4][4] = { { 2640, 4968, 6688, 7608 },
                                               { 2640, 4964, 6688, 7604 },
                                               { 2642, 4964, 6688, 7606 },
                                               { 2642, 4964, 6689, 7606 } };
#endif  // HAVE_NEON

void av1_round_shift_array_c(int32_t *arr, int size, int bit) {
  int i;
  if (bit == 0) {
    return;
  } else {
    if (bit > 0) {
      for (i = 0; i < size; i++) {
        arr[i] = round_shift(arr[i], bit);
      }
    } else {
      for (i = 0; i < size; i++) {
        arr[i] = (int32_t)clamp64(((int64_t)1 << (-bit)) * arr[i], INT32_MIN,
                                  INT32_MAX);
      }
    }
  }
}

const TXFM_TYPE av1_txfm_type_ls[5][TX_TYPES_1D] = {
  { TXFM_TYPE_DCT4, TXFM_TYPE_ADST4, TXFM_TYPE_ADST4, TXFM_TYPE_IDENTITY4 },
  { TXFM_TYPE_DCT8, TXFM_TYPE_ADST8, TXFM_TYPE_ADST8, TXFM_TYPE_IDENTITY8 },
  { TXFM_TYPE_DCT16, TXFM_TYPE_ADST16, TXFM_TYPE_ADST16, TXFM_TYPE_IDENTITY16 },
  { TXFM_TYPE_DCT32, TXFM_TYPE_INVALID, TXFM_TYPE_INVALID,
    TXFM_TYPE_IDENTITY32 },
  { TXFM_TYPE_DCT64, TXFM_TYPE_INVALID, TXFM_TYPE_INVALID, TXFM_TYPE_INVALID }
};

const int8_t av1_txfm_stage_num_list[TXFM_TYPES] = {
  4,   // TXFM_TYPE_DCT4
  6,   // TXFM_TYPE_DCT8
  8,   // TXFM_TYPE_DCT16
  10,  // TXFM_TYPE_DCT32
  12,  // TXFM_TYPE_DCT64
  7,   // TXFM_TYPE_ADST4
  8,   // TXFM_TYPE_ADST8
  10,  // TXFM_TYPE_ADST16
  1,   // TXFM_TYPE_IDENTITY4
  1,   // TXFM_TYPE_IDENTITY8
  1,   // TXFM_TYPE_IDENTITY16
  1,   // TXFM_TYPE_IDENTITY32
};

void av1_range_check_buf(int32_t stage, const int32_t *input,
                         const int32_t *buf, int32_t size, int8_t bit) {
#if CONFIG_COEFFICIENT_RANGE_CHECKING
  const int64_t max_value = (1LL << (bit - 1)) - 1;
  const int64_t min_value = -(1LL << (bit - 1));

  int in_range = 1;

  for (int i = 0; i < size; ++i) {
    if (buf[i] < min_value || buf[i] > max_value) {
      in_range = 0;
    }
  }

  if (!in_range) {
    fprintf(stderr, "Error: coeffs contain out-of-range values\n");
    fprintf(stderr, "size: %d\n", size);
    fprintf(stderr, "stage: %d\n", stage);
    fprintf(stderr, "allowed range: [%" PRId64 ";%" PRId64 "]\n", min_value,
            max_value);

    fprintf(stderr, "coeffs: ");

    fprintf(stderr, "[");
    for (int j = 0; j < size; j++) {
      if (j > 0) fprintf(stderr, ", ");
      fprintf(stderr, "%d", input[j]);
    }
    fprintf(stderr, "]\n");

    fprintf(stderr, "   buf: ");

    fprintf(stderr, "[");
    for (int j = 0; j < size; j++) {
      if (j > 0) fprintf(stderr, ", ");
      fprintf(stderr, "%d", buf[j]);
    }
    fprintf(stderr, "]\n\n");
  }

  assert(in_range);
#else
  (void)stage;
  (void)input;
  (void)buf;
  (void)size;
  (void)bit;
#endif
}
