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

#ifndef AOM_AOM_DSP_BITWRITER_BUFFER_H_
#define AOM_AOM_DSP_BITWRITER_BUFFER_H_

#include <assert.h>

#include "aom/aom_integer.h"

#ifdef __cplusplus
extern "C" {
#endif

struct aom_write_bit_buffer {
  uint8_t *bit_buffer;
  uint32_t bit_offset;
};

int aom_wb_is_byte_aligned(const struct aom_write_bit_buffer *wb);

uint32_t aom_wb_bytes_written(const struct aom_write_bit_buffer *wb);

static inline void aom_wb_write_bit(struct aom_write_bit_buffer *wb, int bit) {
  const uint32_t off = wb->bit_offset;
  const uint32_t p = off >> 3;
  const int q = 7 - (int)(off & 7);
  if (q == 7) {
    wb->bit_buffer[p] = (uint8_t)(bit << 7);
  } else {
    wb->bit_buffer[p] = (uint8_t)((wb->bit_buffer[p] & ~(1 << q)) | (bit << q));
  }
  wb->bit_offset = off + 1;
}

static inline void aom_wb_write_literal(struct aom_write_bit_buffer *wb,
                                        int data, int bits) {
  assert(bits <= 31);
  for (int bit = bits - 1; bit >= 0; bit--) {
    aom_wb_write_bit(wb, (data >> bit) & 1);
  }
}

void aom_wb_write_unsigned_literal(struct aom_write_bit_buffer *wb,
                                   uint32_t data, int bits);

void aom_wb_overwrite_literal(struct aom_write_bit_buffer *wb, int data,
                              int bits);

void aom_wb_write_inv_signed_literal(struct aom_write_bit_buffer *wb, int data,
                                     int bits);

// Writes a variable length unsigned integer. UINT32_MAX is an invalid input.
void aom_wb_write_uvlc(struct aom_write_bit_buffer *wb, uint32_t v);

void aom_wb_write_signed_primitive_refsubexpfin(struct aom_write_bit_buffer *wb,
                                                uint16_t n, uint16_t k,
                                                int16_t ref, int16_t v);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AOM_DSP_BITWRITER_BUFFER_H_
