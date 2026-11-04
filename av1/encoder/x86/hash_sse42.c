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

#include <stdint.h>
#include <nmmintrin.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"

// Byte-boundary alignment issues
#define ALIGN_SIZE 8
#define ALIGN_MASK (ALIGN_SIZE - 1)

#define CALC_CRC(op, crc, type, buf, len) \
  while ((len) >= sizeof(type)) {         \
    (crc) = op((crc), *(type *)(buf));    \
    (len) -= sizeof(type);                \
    buf += sizeof(type);                  \
  }

/**
 * Calculates 32-bit CRC for the input buffer
 * polynomial is 0x11EDC6F41
 * @return A 32-bit unsigned integer representing the CRC
 */
uint32_t av1_get_crc32c_value_sse4_2(void *crc_calculator, const uint8_t *p,
                                     size_t len) {
  (void)crc_calculator;
  const uint8_t *buf = p;
  uint32_t crc = 0xFFFFFFFF;

  // Align the input to the word boundary
  for (; (len > 0) && ((intptr_t)buf & ALIGN_MASK); len--, buf++) {
    crc = _mm_crc32_u8(crc, *buf);
  }

#ifdef __x86_64__
  uint64_t crc64 = crc;
  CALC_CRC(_mm_crc32_u64, crc64, uint64_t, buf, len)
  crc = (uint32_t)crc64;
#endif
  CALC_CRC(_mm_crc32_u32, crc, uint32_t, buf, len)
  CALC_CRC(_mm_crc32_u16, crc, uint16_t, buf, len)
  CALC_CRC(_mm_crc32_u8, crc, uint8_t, buf, len)
  return (crc ^ 0xFFFFFFFF);
}

void av1_combine_block_hashes_sse4_2(void *crc_calculator, const uint32_t *src,
                                     uint32_t *dst, int stride, int width,
                                     int height, int sub_size) {
  (void)crc_calculator;
  for (int y = 0; y < height; y++) {
    const uint32_t *top = src + y * stride;
    const uint32_t *bottom = top + sub_size * stride;
    uint32_t *out = dst + y * stride;
    for (int x = 0; x < width; x++) {
      // Hash the 4 values from registers, in the same byte order as the C
      // version reads them from memory.
#if AOM_ARCH_X86_64
      uint64_t crc = 0xFFFFFFFF;
      crc = _mm_crc32_u64(crc, top[x] | (uint64_t)top[x + sub_size] << 32);
      crc =
          _mm_crc32_u64(crc, bottom[x] | (uint64_t)bottom[x + sub_size] << 32);
      out[x] = (uint32_t)crc ^ 0xFFFFFFFF;
#else
      uint32_t crc = 0xFFFFFFFF;
      crc = _mm_crc32_u32(crc, top[x]);
      crc = _mm_crc32_u32(crc, top[x + sub_size]);
      crc = _mm_crc32_u32(crc, bottom[x]);
      crc = _mm_crc32_u32(crc, bottom[x + sub_size]);
      out[x] = crc ^ 0xFFFFFFFF;
#endif
    }
  }
}
