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

#ifndef AOM_AV1_COMMON_AV1_INV_TXFM1D_H_
#define AOM_AV1_COMMON_AV1_INV_TXFM1D_H_

#include "av1/common/av1_txfm.h"

#if CONFIG_MODE_DEP_INTRA_TX || CONFIG_MODE_DEP_INTER_TX
#include "av1/common/mdtx_bases.h"
#endif  // CONFIG_MODE_DEP_INTRA_TX || CONFIG_MODE_DEP_INTER_TX

#ifdef __cplusplus
extern "C" {
#endif

static INLINE int32_t clamp_value(int32_t value, int8_t bit) {
  if (bit <= 0) return value;  // Do nothing for invalid clamp bit.
  const int64_t max_value = (1LL << (bit - 1)) - 1;
  const int64_t min_value = -(1LL << (bit - 1));
  return (int32_t)clamp64(value, min_value, max_value);
}

static INLINE void clamp_buf(int32_t *buf, int32_t size, int8_t bit) {
  for (int i = 0; i < size; ++i) buf[i] = clamp_value(buf[i], bit);
}

#if CONFIG_MODE_DEP_INTRA_TX || CONFIG_MODE_DEP_INTER_TX
void av1_imdt4(const int32_t *input, int32_t *output, int8_t cos_bit,
               const int8_t *side_info);
void av1_imdt8(const int32_t *input, int32_t *output, int8_t cos_bit,
               const int8_t *side_info);
void av1_imdt16(const int32_t *input, int32_t *output, int8_t cos_bit,
                const int8_t *side_info);
#endif  // CONFIG_MODE_DEP_INTRA_TX || CONFIG_MODE_DEP_INTER_TX
void av1_idct4_new(const int32_t *input, int32_t *output, int8_t cos_bit,
                   const int8_t *stage_range);
void av1_idct8_new(const int32_t *input, int32_t *output, int8_t cos_bit,
                   const int8_t *stage_range);
void av1_idct16_new(const int32_t *input, int32_t *output, int8_t cos_bit,
                    const int8_t *stage_range);
void av1_idct32_new(const int32_t *input, int32_t *output, int8_t cos_bit,
                    const int8_t *stage_range);
void av1_idct64_new(const int32_t *input, int32_t *output, int8_t cos_bit,
                    const int8_t *stage_range);
void av1_iadst4_new(const int32_t *input, int32_t *output, int8_t cos_bit,
                    const int8_t *stage_range);
void av1_iadst8_new(const int32_t *input, int32_t *output, int8_t cos_bit,
                    const int8_t *stage_range);
void av1_iadst16_new(const int32_t *input, int32_t *output, int8_t cos_bit,
                     const int8_t *stage_range);
#if CONFIG_DST7_32x32
void av1_iadst32_new(const int32_t *input, int32_t *output, int8_t cos_bit,
                     const int8_t *stage_range);
#endif
void av1_iidentity4_c(const int32_t *input, int32_t *output, int8_t cos_bit,
                      const int8_t *stage_range);
void av1_iidentity8_c(const int32_t *input, int32_t *output, int8_t cos_bit,
                      const int8_t *stage_range);
void av1_iidentity16_c(const int32_t *input, int32_t *output, int8_t cos_bit,
                       const int8_t *stage_range);
void av1_iidentity32_c(const int32_t *input, int32_t *output, int8_t cos_bit,
                       const int8_t *stage_range);
#if CONFIG_LGT
void av1_iadst4_lgt_intra(const int32_t *input, int32_t *output, int8_t cos_bit,
                          const int8_t *stage_range);
void av1_iadst4_lgt_inter(const int32_t *input, int32_t *output, int8_t cos_bit,
                          const int8_t *stage_range);
void av1_iadst8_lgt_intra(const int32_t *input, int32_t *output, int8_t cos_bit,
                          const int8_t *stage_range);
void av1_iadst8_lgt_inter(const int32_t *input, int32_t *output, int8_t cos_bit,
                          const int8_t *stage_range);
void av1_iadst16_lgt_intra(const int32_t *input, int32_t *output,
                           int8_t cos_bit, const int8_t *stage_range);
void av1_iadst16_lgt_inter(const int32_t *input, int32_t *output,
                           int8_t cos_bit, const int8_t *stage_range);
#endif  // CONFIG_LGT

#ifdef __cplusplus
}
#endif

#endif  // AOM_AV1_COMMON_AV1_INV_TXFM1D_H_
