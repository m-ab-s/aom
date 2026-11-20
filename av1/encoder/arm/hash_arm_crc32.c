/*
 * Copyright (c) 2022, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#else
#include <arm_acle.h>
#endif

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"

#define CRC_LOOP(op, crc, type, buf, len)     \
  do {                                        \
    while ((len) >= sizeof(type)) {           \
      type crc_val_;                          \
      memcpy(&crc_val_, (buf), sizeof(type)); \
      (crc) = op((crc), crc_val_);            \
      (len) -= sizeof(type);                  \
      (buf) += sizeof(type);                  \
    }                                         \
  } while (0)

#define CRC_SINGLE(op, crc, type, buf, len)   \
  do {                                        \
    if ((len) >= sizeof(type)) {              \
      type crc_val_;                          \
      memcpy(&crc_val_, (buf), sizeof(type)); \
      (crc) = op((crc), crc_val_);            \
      (len) -= sizeof(type);                  \
      (buf) += sizeof(type);                  \
    }                                         \
  } while (0)

/* Return 32-bit CRC for the input buffer.
 * Polynomial is 0x1EDC6F41.
 */

uint32_t av1_get_crc32c_value_arm_crc32(void *crc_calculator, const uint8_t *p,
                                        size_t len) {
  (void)crc_calculator;
  assert(p != NULL || len == 0);
  const uint8_t *buf = p;
  uint32_t crc = 0xFFFFFFFF;

#if !AOM_ARCH_AARCH64
  // Align input to 8-byte boundary (only necessary for 32-bit builds.)
  while (len && ((uintptr_t)buf & 7)) {
    crc = __crc32cb(crc, *buf++);
    len--;
  }
#endif

  while (len >= 32) {
    uint64_t v0, v1, v2, v3;
    memcpy(&v0, buf, sizeof(v0));
    memcpy(&v1, buf + 8, sizeof(v1));
    memcpy(&v2, buf + 16, sizeof(v2));
    memcpy(&v3, buf + 24, sizeof(v3));
    crc = __crc32cd(crc, v0);
    crc = __crc32cd(crc, v1);
    crc = __crc32cd(crc, v2);
    crc = __crc32cd(crc, v3);
    buf += 32;
    len -= 32;
  }
  CRC_LOOP(__crc32cd, crc, uint64_t, buf, len);
  CRC_SINGLE(__crc32cw, crc, uint32_t, buf, len);
  CRC_SINGLE(__crc32ch, crc, uint16_t, buf, len);
  CRC_SINGLE(__crc32cb, crc, uint8_t, buf, len);

  return ~crc;
}

// Hash the 4 values from registers, low byte first, matching the C version.
void av1_combine_block_hashes_arm_crc32(void *crc_calculator,
                                        const uint32_t *top,
                                        const uint32_t *bottom, uint32_t *dst,
                                        int width, int sub_size) {
  (void)crc_calculator;
  assert(width >= 0);
  if (width <= 0) return;
  assert(sub_size > 0);
  assert(top != NULL && bottom != NULL && dst != NULL);
  assert(dst != top && dst != bottom);
  assert(dst != top + sub_size && dst != bottom + sub_size);

  const int rem = width & 3;
  for (int x = 0; x < rem; ++x) {
    uint32_t crc = 0xFFFFFFFF;
    crc = __crc32cd(crc, top[x] | ((uint64_t)top[x + sub_size] << 32));
    crc = __crc32cd(crc, bottom[x] | ((uint64_t)bottom[x + sub_size] << 32));
    dst[x] = ~crc;
  }
  int n = width >> 2;
  if (n == 0) return;

  top += rem;
  bottom += rem;
  dst += rem;
  const uint32_t *top_right = top + sub_size;
  const uint32_t *bottom_right = bottom + sub_size;

  do {
    // Process 4 positions per iteration so independent CRC chains overlap.
    uint32_t c0 =
        __crc32cd(0xFFFFFFFF, top[0] | ((uint64_t)top_right[0] << 32));
    uint32_t c1 =
        __crc32cd(0xFFFFFFFF, top[1] | ((uint64_t)top_right[1] << 32));
    uint32_t c2 =
        __crc32cd(0xFFFFFFFF, top[2] | ((uint64_t)top_right[2] << 32));
    uint32_t c3 =
        __crc32cd(0xFFFFFFFF, top[3] | ((uint64_t)top_right[3] << 32));
    c0 = __crc32cd(c0, bottom[0] | ((uint64_t)bottom_right[0] << 32));
    c1 = __crc32cd(c1, bottom[1] | ((uint64_t)bottom_right[1] << 32));
    c2 = __crc32cd(c2, bottom[2] | ((uint64_t)bottom_right[2] << 32));
    c3 = __crc32cd(c3, bottom[3] | ((uint64_t)bottom_right[3] << 32));
    dst[0] = ~c0;
    dst[1] = ~c1;
    dst[2] = ~c2;
    dst[3] = ~c3;
    top += 4;
    top_right += 4;
    bottom += 4;
    bottom_right += 4;
    dst += 4;
  } while (--n != 0);
}
