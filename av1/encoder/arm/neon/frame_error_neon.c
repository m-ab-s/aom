/*
 * Copyright (c) 2023, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <arm_neon.h>
#include <stdlib.h>

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"

/* clang-format off */
// Error metric used for global motion evaluation.
static const uint16_t error_measure_lut[256] = {
        0,   339,   550,   731,   894,  1045,  1187,  1323,
     1452,  1577,  1698,  1815,  1929,  2040,  2148,  2255,
     2359,  2461,  2562,  2661,  2758,  2854,  2948,  3041,
     3133,  3224,  3314,  3402,  3490,  3577,  3663,  3748,
     3832,  3916,  3998,  4080,  4162,  4242,  4322,  4401,
     4480,  4558,  4636,  4713,  4789,  4865,  4941,  5015,
     5090,  5164,  5237,  5311,  5383,  5456,  5527,  5599,
     5670,  5741,  5811,  5881,  5950,  6020,  6089,  6157,
     6225,  6293,  6361,  6428,  6495,  6562,  6628,  6695,
     6760,  6826,  6891,  6956,  7021,  7086,  7150,  7214,
     7278,  7341,  7405,  7468,  7531,  7593,  7656,  7718,
     7780,  7842,  7903,  7965,  8026,  8087,  8148,  8208,
     8269,  8329,  8389,  8449,  8508,  8568,  8627,  8686,
     8745,  8804,  8862,  8921,  8979,  9037,  9095,  9153,
     9211,  9268,  9326,  9383,  9440,  9497,  9553,  9610,
     9666,  9723,  9779,  9835,  9891,  9947, 10002, 10058,
    10113, 10168, 10224, 10279, 10333, 10388, 10443, 10497,
    10552, 10606, 10660, 10714, 10768, 10822, 10875, 10929,
    10982, 11036, 11089, 11142, 11195, 11248, 11301, 11353,
    11406, 11458, 11511, 11563, 11615, 11667, 11719, 11771,
    11823, 11875, 11926, 11978, 12029, 12080, 12132, 12183,
    12234, 12285, 12335, 12386, 12437, 12487, 12538, 12588,
    12639, 12689, 12739, 12789, 12839, 12889, 12939, 12988,
    13038, 13088, 13137, 13187, 13236, 13285, 13334, 13383,
    13432, 13481, 13530, 13579, 13628, 13676, 13725, 13773,
    13822, 13870, 13918, 13967, 14015, 14063, 14111, 14159,
    14206, 14254, 14302, 14350, 14397, 14445, 14492, 14539,
    14587, 14634, 14681, 14728, 14775, 14822, 14869, 14916,
    14963, 15010, 15056, 15103, 15149, 15196, 15242, 15289,
    15335, 15381, 15427, 15474, 15520, 15566, 15612, 15657,
    15703, 15749, 15795, 15840, 15886, 15932, 15977, 16022,
    16068, 16113, 16158, 16204, 16249, 16294, 16339, 16384,
};
/* clang-format on */

int64_t av1_calc_frame_error_neon(const uint8_t *const ref, int ref_stride,
                                  const uint8_t *const dst, int dst_stride,
                                  int width, int height) {
  int64_t sum_error[4] = { 0, 0, 0, 0 };
  int r = 0;
  int d = 0;

  do {
    int w = width;
    int rr = r;
    int dd = d;

    do {
      uint8x16_t dst_v = vld1q_u8(&dst[dd]);
      uint8x16_t ref_v = vld1q_u8(&ref[rr]);

#if AOM_ARCH_AARCH64
      uint64x2_t abs_v = vreinterpretq_u64_u8(vabdq_u8(dst_v, ref_v));

      uint64_t abs0 = vgetq_lane_u64(abs_v, 0);
      uint64_t abs1 = vgetq_lane_u64(abs_v, 1);

      sum_error[0] += error_measure_lut[(abs0 >> 0) & 0xFF];
      sum_error[1] += error_measure_lut[(abs0 >> 8) & 0xFF];
      sum_error[2] += error_measure_lut[(abs0 >> 16) & 0xFF];
      sum_error[3] += error_measure_lut[(abs0 >> 24) & 0xFF];
      sum_error[0] += error_measure_lut[(abs0 >> 32) & 0xFF];
      sum_error[1] += error_measure_lut[(abs0 >> 40) & 0xFF];
      sum_error[2] += error_measure_lut[(abs0 >> 48) & 0xFF];
      sum_error[3] += error_measure_lut[(abs0 >> 56) & 0xFF];

      sum_error[0] += error_measure_lut[(abs1 >> 0) & 0xFF];
      sum_error[1] += error_measure_lut[(abs1 >> 8) & 0xFF];
      sum_error[2] += error_measure_lut[(abs1 >> 16) & 0xFF];
      sum_error[3] += error_measure_lut[(abs1 >> 24) & 0xFF];
      sum_error[0] += error_measure_lut[(abs1 >> 32) & 0xFF];
      sum_error[1] += error_measure_lut[(abs1 >> 40) & 0xFF];
      sum_error[2] += error_measure_lut[(abs1 >> 48) & 0xFF];
      sum_error[3] += error_measure_lut[(abs1 >> 56) & 0xFF];
#else   // !AOM_ARCH_AARCH64
      uint32x4_t abs_v = vreinterpretq_u32_u8(vabdq_u8(dst_v, ref_v));

      uint32_t abs0 = vgetq_lane_u32(abs_v, 0);
      uint32_t abs1 = vgetq_lane_u32(abs_v, 1);
      uint32_t abs2 = vgetq_lane_u32(abs_v, 2);
      uint32_t abs3 = vgetq_lane_u32(abs_v, 3);

      sum_error[0] += error_measure_lut[(abs0 >> 0) & 0xFF];
      sum_error[1] += error_measure_lut[(abs0 >> 8) & 0xFF];
      sum_error[2] += error_measure_lut[(abs0 >> 16) & 0xFF];
      sum_error[3] += error_measure_lut[(abs0 >> 24) & 0xFF];
      sum_error[0] += error_measure_lut[(abs1 >> 0) & 0xFF];
      sum_error[1] += error_measure_lut[(abs1 >> 8) & 0xFF];
      sum_error[2] += error_measure_lut[(abs1 >> 16) & 0xFF];
      sum_error[3] += error_measure_lut[(abs1 >> 24) & 0xFF];

      sum_error[0] += error_measure_lut[(abs2 >> 0) & 0xFF];
      sum_error[1] += error_measure_lut[(abs2 >> 8) & 0xFF];
      sum_error[2] += error_measure_lut[(abs2 >> 16) & 0xFF];
      sum_error[3] += error_measure_lut[(abs2 >> 24) & 0xFF];
      sum_error[0] += error_measure_lut[(abs3 >> 0) & 0xFF];
      sum_error[1] += error_measure_lut[(abs3 >> 8) & 0xFF];
      sum_error[2] += error_measure_lut[(abs3 >> 16) & 0xFF];
      sum_error[3] += error_measure_lut[(abs3 >> 24) & 0xFF];
#endif  // AOM_ARCH_AARCH64

      dd += 16;
      rr += 16;
      w -= 16;
    } while (w >= 16);

    while (w-- != 0) {
      sum_error[0] += error_measure_lut[abs(dst[dd] - ref[rr])];
      dd++;
      rr++;
    }

    r += ref_stride;
    d += dst_stride;
  } while (--height != 0);

  return sum_error[0] + sum_error[1] + sum_error[2] + sum_error[3];
}
