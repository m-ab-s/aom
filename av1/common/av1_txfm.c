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
// av1_cospi_arr_q13_data[i][j] =
//   (int)round(cos(PI*j/128) * (1<<(cos_bit_min+i))) << (3-i)
// See also: https://en.wikipedia.org/wiki/Q_(number_format)
const int16_t av1_cospi_arr_q13_data[4][64] = {
  { 8192, 8192, 8184, 8168, 8152, 8128, 8104, 8072, 8032, 7992, 7944,
    7896, 7840, 7776, 7712, 7640, 7568, 7488, 7408, 7320, 7224, 7128,
    7024, 6920, 6808, 6696, 6576, 6456, 6336, 6200, 6072, 5936, 5792,
    5648, 5504, 5352, 5200, 5040, 4880, 4720, 4552, 4384, 4208, 4040,
    3864, 3680, 3504, 3320, 3136, 2952, 2760, 2568, 2376, 2184, 1992,
    1792, 1600, 1400, 1200, 1000, 800,  600,  400,  200 },
  { 8192, 8188, 8184, 8168, 8152, 8132, 8104, 8072, 8036, 7992, 7948,
    7896, 7840, 7780, 7712, 7644, 7568, 7488, 7404, 7316, 7224, 7128,
    7028, 6920, 6812, 6696, 6580, 6460, 6332, 6204, 6068, 5932, 5792,
    5648, 5500, 5352, 5196, 5040, 4880, 4716, 4552, 4384, 4212, 4036,
    3860, 3684, 3504, 3320, 3136, 2948, 2760, 2568, 2380, 2184, 1992,
    1796, 1600, 1400, 1204, 1004, 804,  604,  400,  200 },
  { 8192, 8190, 8182, 8170, 8152, 8130, 8104, 8072, 8034, 7992, 7946,
    7896, 7840, 7778, 7714, 7644, 7568, 7490, 7406, 7318, 7224, 7128,
    7026, 6922, 6812, 6698, 6580, 6458, 6332, 6204, 6070, 5934, 5792,
    5648, 5502, 5350, 5196, 5040, 4880, 4718, 4552, 4382, 4212, 4038,
    3862, 3684, 3502, 3320, 3134, 2948, 2760, 2570, 2378, 2184, 1990,
    1794, 1598, 1400, 1202, 1002, 802,  602,  402,  202 },
  { 8192, 8190, 8182, 8170, 8153, 8130, 8103, 8071, 8035, 7993, 7946,
    7895, 7839, 7779, 7713, 7643, 7568, 7489, 7405, 7317, 7225, 7128,
    7027, 6921, 6811, 6698, 6580, 6458, 6333, 6203, 6070, 5933, 5793,
    5649, 5501, 5351, 5197, 5040, 4880, 4717, 4551, 4383, 4212, 4038,
    3862, 3683, 3503, 3320, 3135, 2948, 2760, 2570, 2378, 2185, 1990,
    1795, 1598, 1401, 1202, 1003, 803,  603,  402,  201 }
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
