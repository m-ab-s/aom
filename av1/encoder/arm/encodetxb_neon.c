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

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"

#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/arm/mem_neon.h"
#include "aom_ports/mem.h"
#include "av1/common/txb_common.h"
#include "av1/encoder/encodetxb.h"

static inline void zero_bottom_pad_neon(uint8_t *const pad, const int height) {
  const uint8x16_t zeros_u8 = vdupq_n_u8(0);
  if (height == 4) {
    // TX_PAD_BOTTOM * (4 + TX_PAD_HOR) + TX_PAD_END == 48
    vst1q_u8(pad, zeros_u8);
    vst1q_u8(pad + 16, zeros_u8);
    vst1q_u8(pad + 32, zeros_u8);
  } else if (height == 8) {
    // TX_PAD_BOTTOM * (8 + TX_PAD_HOR) + TX_PAD_END == 64
    vst1q_u8(pad, zeros_u8);
    vst1q_u8(pad + 16, zeros_u8);
    vst1q_u8(pad + 32, zeros_u8);
    vst1q_u8(pad + 48, zeros_u8);
  } else if (height == 16) {
    // TX_PAD_BOTTOM * (16 + TX_PAD_HOR) + TX_PAD_END == 96
    vst1q_u8(pad, zeros_u8);
    vst1q_u8(pad + 16, zeros_u8);
    vst1q_u8(pad + 32, zeros_u8);
    vst1q_u8(pad + 48, zeros_u8);
    vst1q_u8(pad + 64, zeros_u8);
    vst1q_u8(pad + 80, zeros_u8);
  } else {
    // TX_PAD_BOTTOM * (32 + TX_PAD_HOR) + TX_PAD_END == 160
    assert(height == 32);
    vst1q_u8(pad, zeros_u8);
    vst1q_u8(pad + 16, zeros_u8);
    vst1q_u8(pad + 32, zeros_u8);
    vst1q_u8(pad + 48, zeros_u8);
    vst1q_u8(pad + 64, zeros_u8);
    vst1q_u8(pad + 80, zeros_u8);
    vst1q_u8(pad + 96, zeros_u8);
    vst1q_u8(pad + 112, zeros_u8);
    vst1q_u8(pad + 128, zeros_u8);
    vst1q_u8(pad + 144, zeros_u8);
  }
}

static inline int16x8_t load_tran_low_to_s16q_sat(const tran_low_t *const buf) {
  return vcombine_s16(vqmovn_s32(vld1q_s32(buf)),
                      vqmovn_s32(vld1q_s32(buf + 4)));
}

void av1_txb_init_levels_neon(const tran_low_t *const coeff, const int width,
                              const int height, uint8_t *const levels) {
  assert(TX_PAD_TOP == 0 && TX_PAD_HOR == 4 && TX_PAD_BOTTOM == 4 &&
         TX_PAD_END == 16);
  assert(width > 0);

  const int32x4_t zeros = vdupq_n_s32(0);
  uint8_t *ls = levels;
  const tran_low_t *cf = coeff;
  if (height == 4) {
    assert(!(width % 2));
    const int stride = 4 + TX_PAD_HOR;
    int col = width;
    do {
      const int16x8_t coeffAB = load_tran_low_to_s16q_sat(cf);
      const int16x8_t absAB = vqabsq_s16(coeffAB);
      const int8x8_t absABs = vqmovn_s16(absAB);
#if AOM_ARCH_AARCH64
      const int8x16_t absAB8 =
          vcombine_s8(absABs, vreinterpret_s8_s32(vget_low_s32(zeros)));
      const uint8x16_t lsAB =
          vreinterpretq_u8_s32(vzip1q_s32(vreinterpretq_s32_s8(absAB8), zeros));
#else
      const int32x2x2_t absAB8 =
          vzip_s32(vreinterpret_s32_s8(absABs), vget_low_s32(zeros));
      const uint8x16_t lsAB =
          vreinterpretq_u8_s32(vcombine_s32(absAB8.val[0], absAB8.val[1]));
#endif
      vst1q_u8(ls, lsAB);
      ls += (stride << 1);
      cf += 8;
      col -= 2;
    } while (col != 0);
    zero_bottom_pad_neon(ls, 4);
  } else if (height == 8) {
    assert(!(width % 2));
    const int stride = 8 + TX_PAD_HOR;
    int col = width;
    do {
      const int16x8_t coeff0 = load_tran_low_to_s16q_sat(cf);
      const int16x8_t coeff1 = load_tran_low_to_s16q_sat(cf + 8);
      const int16x8_t abs0 = vqabsq_s16(coeff0);
      const int16x8_t abs1 = vqabsq_s16(coeff1);
      const uint8x16_t abs0_u8 = vreinterpretq_u8_s8(vcombine_s8(
          vqmovn_s16(abs0), vreinterpret_s8_s32(vget_low_s32(zeros))));
      const uint8x16_t abs1_u8 = vreinterpretq_u8_s8(vcombine_s8(
          vqmovn_s16(abs1), vreinterpret_s8_s32(vget_low_s32(zeros))));
      vst1q_u8(ls, abs0_u8);
      vst1q_u8(ls + stride, abs1_u8);
      ls += (stride << 1);
      cf += 16;
      col -= 2;
    } while (col != 0);
    zero_bottom_pad_neon(ls, 8);
  } else if (height == 16) {
    const int stride = 16 + TX_PAD_HOR;
    int col = width;
    do {
      const int16x8_t coeffAB = load_tran_low_to_s16q_sat(cf);
      const int16x8_t coeffCD = load_tran_low_to_s16q_sat(cf + 8);
      const int16x8_t absAB = vqabsq_s16(coeffAB);
      const int16x8_t absCD = vqabsq_s16(coeffCD);
      const uint8x16_t absABCD = vreinterpretq_u8_s8(
          vcombine_s8(vqmovn_s16(absAB), vqmovn_s16(absCD)));
      vst1q_u8(ls, absABCD);
      store_u8_4x1(ls + 16, vreinterpret_u8_s32(vget_low_s32(zeros)));
      cf += 16;
      ls += stride;
    } while (--col != 0);
    zero_bottom_pad_neon(ls, 16);
  } else {
    assert(height == 32);
    const int stride = 32 + TX_PAD_HOR;
    int col = width;
    do {
      const int16x8_t coeff0 = load_tran_low_to_s16q_sat(cf);
      const int16x8_t coeff1 = load_tran_low_to_s16q_sat(cf + 8);
      const int16x8_t coeff2 = load_tran_low_to_s16q_sat(cf + 16);
      const int16x8_t coeff3 = load_tran_low_to_s16q_sat(cf + 24);
      const int16x8_t abs0 = vqabsq_s16(coeff0);
      const int16x8_t abs1 = vqabsq_s16(coeff1);
      const int16x8_t abs2 = vqabsq_s16(coeff2);
      const int16x8_t abs3 = vqabsq_s16(coeff3);
      const uint8x16_t abs01 =
          vreinterpretq_u8_s8(vcombine_s8(vqmovn_s16(abs0), vqmovn_s16(abs1)));
      const uint8x16_t abs23 =
          vreinterpretq_u8_s8(vcombine_s8(vqmovn_s16(abs2), vqmovn_s16(abs3)));
      vst1q_u8(ls, abs01);
      vst1q_u8(ls + 16, abs23);
      store_u8_4x1(ls + 32, vreinterpret_u8_s32(vget_low_s32(zeros)));
      cf += 32;
      ls += stride;
    } while (--col != 0);
    zero_bottom_pad_neon(ls, 32);
  }
}

// get_4_nz_map_contexts_2d coefficients:
static const DECLARE_ALIGNED(16, uint8_t, c_4_po_2d[2][16]) = {
  { 0, 1, 6, 6, 1, 6, 6, 21, 6, 6, 21, 21, 6, 21, 21, 21 },
  { 0, 16, 16, 16, 16, 16, 16, 16, 6, 6, 21, 21, 6, 21, 21, 21 }
};

// get_4_nz_map_contexts_hor coefficients:
/* clang-format off */
#define SIG_COEF_CONTEXTS_2D_X4_051010                        \
  (SIG_COEF_CONTEXTS_2D + ((SIG_COEF_CONTEXTS_2D + 5) << 8) + \
  ((SIG_COEF_CONTEXTS_2D + 10) << 16) + ((SIG_COEF_CONTEXTS_2D + 10) << 24))
/* clang-format on */

// get_4_nz_map_contexts_ver coefficients:
static const DECLARE_ALIGNED(16, uint8_t, c_4_po_hor[16]) = {
  SIG_COEF_CONTEXTS_2D + 0,  SIG_COEF_CONTEXTS_2D + 0,
  SIG_COEF_CONTEXTS_2D + 0,  SIG_COEF_CONTEXTS_2D + 0,
  SIG_COEF_CONTEXTS_2D + 5,  SIG_COEF_CONTEXTS_2D + 5,
  SIG_COEF_CONTEXTS_2D + 5,  SIG_COEF_CONTEXTS_2D + 5,
  SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
  SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
  SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
  SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10
};

// get_8_coeff_contexts_2d coefficients, indexed by
// (width >= 8) + (width > 8): 0: width < 8, 1: width == 8, 2: width > 8.
static const DECLARE_ALIGNED(16, uint8_t, c_8_po_2d_all[3][2][16]) = {
  // width < 8
  {
      { 0, 11, 6, 6, 21, 21, 21, 21, 11, 11, 6, 21, 21, 21, 21, 21 },
      { 11, 11, 21, 21, 21, 21, 21, 21, 11, 11, 21, 21, 21, 21, 21, 21 },
  },
  // width == 8
  {
      { 0, 1, 6, 6, 21, 21, 21, 21, 1, 6, 6, 21, 21, 21, 21, 21 },
      { 6, 6, 21, 21, 21, 21, 21, 21, 6, 21, 21, 21, 21, 21, 21, 21 },
  },
  // width > 8
  {
      { 0, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16 },
      { 6, 6, 21, 21, 21, 21, 21, 21, 6, 21, 21, 21, 21, 21, 21, 21 },
  },
};

// get_4_nz_map_contexts_ver coefficients:
static const DECLARE_ALIGNED(16, uint8_t, c_8_po_ver[16]) = {
  SIG_COEF_CONTEXTS_2D + 0,  SIG_COEF_CONTEXTS_2D + 5,
  SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
  SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
  SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
  SIG_COEF_CONTEXTS_2D + 0,  SIG_COEF_CONTEXTS_2D + 5,
  SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
  SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
  SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10
};

// get_16n_coeff_contexts_2d coefficients, indexed by
// (real_width >= real_height) + (real_width > real_height):
// 0: real_width < real_height, 1: real_width == real_height,
// 2: real_width > real_height.
static const DECLARE_ALIGNED(16, uint8_t, c_16_po_2d_all[3][5][16]) = {
  // real_width < real_height
  {
      { 0, 11, 6, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21 },
      { 11, 11, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21 },
      { 11, 11, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21 },
      { 11, 11, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21 },
      { 11, 11, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21 },
  },
  // real_width == real_height
  {
      { 0, 1, 6, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21 },
      { 1, 6, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21 },
      { 6, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21 },
      { 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21 },
      { 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21 },
  },
  // real_width > real_height
  {
      { 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16 },
      { 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16 },
      { 6, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21 },
      { 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21 },
      { 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21 },
  },
};

// get_16n_coeff_contexts_hor coefficients:
static const DECLARE_ALIGNED(16, uint8_t, c_16_po_ver[16]) = {
  SIG_COEF_CONTEXTS_2D + 0,  SIG_COEF_CONTEXTS_2D + 5,
  SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
  SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
  SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
  SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
  SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
  SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10,
  SIG_COEF_CONTEXTS_2D + 10, SIG_COEF_CONTEXTS_2D + 10
};

// end of coefficients declaration area

#if !AOM_ARCH_AARCH64
static inline uint8x16_t load_8bit_4x4_to_1_reg(const uint8_t *const src,
                                                const int byte_stride) {
  return load_unaligned_u8q(src, byte_stride);
}

static inline void load_levels_4x4x5(const uint8_t *const src, const int stride,
                                     const ptrdiff_t *const offsets,
                                     uint8x16_t *const level) {
  level[0] = load_8bit_4x4_to_1_reg(&src[1], stride);
  level[1] = load_8bit_4x4_to_1_reg(&src[stride], stride);
  level[2] = load_8bit_4x4_to_1_reg(&src[offsets[0]], stride);
  level[3] = load_8bit_4x4_to_1_reg(&src[offsets[1]], stride);
  level[4] = load_8bit_4x4_to_1_reg(&src[offsets[2]], stride);
}
#endif  // !AOM_ARCH_AARCH64

static inline uint8x16_t neon_vshr1_2d_u8x16(const uint8x16_t v) {
  return vreinterpretq_u8_u64(vshrq_n_u64(vreinterpretq_u64_u8(v), 8));
}

#if !AOM_ARCH_AARCH64
static inline uint8x16_t get_coeff_contexts_kernel(uint8x16_t *const level) {
  const uint8x16_t const_3 = vdupq_n_u8(3);
  const uint8x16_t const_4 = vdupq_n_u8(4);
  uint8x16_t count;

  count = vminq_u8(level[0], const_3);
  level[1] = vminq_u8(level[1], const_3);
  level[2] = vminq_u8(level[2], const_3);
  level[3] = vminq_u8(level[3], const_3);
  level[4] = vminq_u8(level[4], const_3);
  count = vaddq_u8(count, level[1]);
  count = vaddq_u8(count, level[2]);
  count = vaddq_u8(count, level[3]);
  count = vaddq_u8(count, level[4]);

  count = vrshrq_n_u8(count, 1);
  count = vminq_u8(count, const_4);
  return count;
}
#endif  // !AOM_ARCH_AARCH64

static AOM_FORCE_INLINE void get_4x4_nz_map_contexts_2d(
    const uint8_t *const levels, uint8_t *const coeff_contexts) {
  const uint8x16_t pos_to_offset = vld1q_u8(c_4_po_2d[0]);
  uint8x16_t count;
#if AOM_ARCH_AARCH64
  const uint8x16_t v_0 = vld1q_u8(levels);
  const uint8x16_t v_16 = vld1q_u8(levels + 16);
  const uint8x16_t const_3 = vdupq_n_u8(3);
  const uint8x16_t const_4 = vdupq_n_u8(4);
  const uint32x4_t lo = vreinterpretq_u32_u8(
      vminq_u8(vreinterpretq_u8_u32(vuzp1q_u32(vreinterpretq_u32_u8(v_0),
                                               vreinterpretq_u32_u8(v_16))),
               const_3));
  const uint32x4_t zeros32 = vdupq_n_u32(0);
  const uint32x4_t l0 = vshrq_n_u32(lo, 8);
  const uint32x4_t l2 = vshrq_n_u32(lo, 16);
  const uint32x4_t l1 = vextq_u32(lo, zeros32, 1);
  const uint32x4_t l4 = vextq_u32(lo, zeros32, 2);
  const uint32x4_t l3 = vextq_u32(l0, zeros32, 1);
  count = vaddq_u8(
      vreinterpretq_u8_u32(l0),
      vaddq_u8(vaddq_u8(vreinterpretq_u8_u32(l1), vreinterpretq_u8_u32(l2)),
               vaddq_u8(vreinterpretq_u8_u32(l3), vreinterpretq_u8_u32(l4))));
  count = vminq_u8(vrshrq_n_u8(count, 1), const_4);
#else
  const int stride = 4 + TX_PAD_HOR;
  const ptrdiff_t offsets[3] = { 0 * stride + 2, 1 * stride + 1,
                                 2 * stride + 0 };
  uint8x16_t level[5];
  load_levels_4x4x5(levels, stride, offsets, level);
  count = get_coeff_contexts_kernel(level);
#endif
  count = vaddq_u8(count, pos_to_offset);
  vst1q_u8(coeff_contexts, count);
  coeff_contexts[0] = 0;
}

static AOM_FORCE_INLINE void get_4_nz_map_contexts_2d(
    const uint8_t *levels, const int n_cols, uint8_t *const coeff_contexts) {
  const int stride = 4 + TX_PAD_HOR;
  const uint8x16_t pos_to_offset_large = vdupq_n_u8(21);
  // TX_4X4 is peeled into get_4x4_nz_map_contexts_2d(), so width > 4 here
  // (n_cols may still be 4 for sparse 8x4/16x4 blocks).
  uint8x16_t pos_to_offset = vld1q_u8(c_4_po_2d[1]);
  uint8x16_t count;
  uint8_t *cc = coeff_contexts;
#if AOM_ARCH_AARCH64
  const uint8x16_t const_3 = vdupq_n_u8(3);
  const uint8x16_t const_4 = vdupq_n_u8(4);
  uint32x4_t v_0 = vreinterpretq_u32_u8(vld1q_u8(levels));
#else
  uint8x16_t level[5];
  const ptrdiff_t offsets[3] = { 0 * stride + 2, 1 * stride + 1,
                                 2 * stride + 0 };
#endif

  assert(n_cols > 0 && !(n_cols % 4));

  int col = n_cols;
  do {
#if AOM_ARCH_AARCH64
    const uint32x4_t v_16 = vreinterpretq_u32_u8(vld1q_u8(levels + 16));
    const uint32x4_t v_32 = vreinterpretq_u32_u8(vld1q_u8(levels + 32));
    const uint32x4_t lo = vreinterpretq_u32_u8(
        vminq_u8(vreinterpretq_u8_u32(vuzp1q_u32(v_0, v_16)), const_3));
    const uint32x4_t lo32 = vreinterpretq_u32_u8(
        vminq_u8(vreinterpretq_u8_u32(vuzp1q_u32(v_32, v_32)), const_3));
    const uint32x4_t l0 = vshrq_n_u32(lo, 8);
    const uint32x4_t l2 = vshrq_n_u32(lo, 16);
    const uint32x4_t l1 = vextq_u32(lo, lo32, 1);
    const uint32x4_t l4 = vextq_u32(lo, lo32, 2);
    const uint32x4_t l3 = vextq_u32(l0, vshrq_n_u32(lo32, 8), 1);
    count = vaddq_u8(
        vreinterpretq_u8_u32(l0),
        vaddq_u8(vaddq_u8(vreinterpretq_u8_u32(l1), vreinterpretq_u8_u32(l2)),
                 vaddq_u8(vreinterpretq_u8_u32(l3), vreinterpretq_u8_u32(l4))));
    count = vminq_u8(vrshrq_n_u8(count, 1), const_4);
    v_0 = v_32;
#else
    load_levels_4x4x5(levels, stride, offsets, level);
    count = get_coeff_contexts_kernel(level);
#endif
    count = vaddq_u8(count, pos_to_offset);
    vst1q_u8(cc, count);
    pos_to_offset = pos_to_offset_large;
    levels += 4 * stride;
    cc += 16;
    col -= 4;
  } while (col != 0);

  coeff_contexts[0] = 0;
}

static AOM_FORCE_INLINE void get_4_nz_map_contexts_ver(
    const uint8_t *levels, const int n_cols, uint8_t *coeff_contexts) {
  const int stride = 4 + TX_PAD_HOR;
  const uint8x16_t pos_to_offset =
      vreinterpretq_u8_u32(vdupq_n_u32(SIG_COEF_CONTEXTS_2D_X4_051010));
  uint8x16_t count;
#if AOM_ARCH_AARCH64
  const uint8x16_t const_3 = vdupq_n_u8(3);
  const uint8x16_t const_4 = vdupq_n_u8(4);
  uint32x4_t v_0 = vreinterpretq_u32_u8(vld1q_u8(levels));
#else
  uint8x16_t level[5];
  const ptrdiff_t offsets[3] = { 2, 3, 4 };
#endif

  assert(n_cols > 0 && !(n_cols % 4));

  int col = n_cols;
  do {
#if AOM_ARCH_AARCH64
    const uint32x4_t v_16 = vreinterpretq_u32_u8(vld1q_u8(levels + 16));
    const uint32x4_t v_32 =
        vreinterpretq_u32_u8(vminq_u8(vld1q_u8(levels + 32), const_3));
    const uint32x4_t lo = vreinterpretq_u32_u8(
        vminq_u8(vreinterpretq_u8_u32(vuzp1q_u32(v_0, v_16)), const_3));
    const uint32x4_t l0 = vshrq_n_u32(lo, 8);
    const uint32x4_t l1 = vextq_u32(lo, v_32, 1);
    const uint32x4_t l2 = vshrq_n_u32(lo, 16);
    const uint32x4_t l3 = vshrq_n_u32(lo, 24);
    count =
        vaddq_u8(vaddq_u8(vreinterpretq_u8_u32(l0), vreinterpretq_u8_u32(l1)),
                 vaddq_u8(vreinterpretq_u8_u32(l2), vreinterpretq_u8_u32(l3)));
    count = vminq_u8(vrshrq_n_u8(count, 1), const_4);
    v_0 = v_32;
#else
    load_levels_4x4x5(levels, stride, offsets, level);
    count = get_coeff_contexts_kernel(level);
#endif
    count = vaddq_u8(count, pos_to_offset);
    vst1q_u8(coeff_contexts, count);
    levels += 4 * stride;
    coeff_contexts += 16;
    col -= 4;
  } while (col != 0);
}

static AOM_FORCE_INLINE void get_4_nz_map_contexts_hor(
    const uint8_t *levels, const int n_cols, uint8_t *coeff_contexts) {
  const int stride = 4 + TX_PAD_HOR;
  const uint8x16_t pos_to_offset_large = vdupq_n_u8(SIG_COEF_CONTEXTS_2D + 10);
  uint8x16_t pos_to_offset = vld1q_u8(c_4_po_hor);
  uint8x16_t count;
#if AOM_ARCH_AARCH64
  const uint8x16_t const_3 = vdupq_n_u8(3);
  const uint8x16_t const_4 = vdupq_n_u8(4);
  const uint32x4_t v_0 = vreinterpretq_u32_u8(vld1q_u8(levels));
  const uint32x4_t v_16 = vreinterpretq_u32_u8(vld1q_u8(levels + 16));
  uint32x4_t lo0 = vreinterpretq_u32_u8(
      vminq_u8(vreinterpretq_u8_u32(vuzp1q_u32(v_0, v_16)), const_3));
#else
  uint8x16_t level[5];
  const ptrdiff_t offsets[3] = { 2 * stride, 3 * stride, 4 * stride };
#endif

  assert(n_cols > 0 && !(n_cols % 4));

  int col = n_cols;
  do {
#if AOM_ARCH_AARCH64
    const uint32x4_t v_32 = vreinterpretq_u32_u8(vld1q_u8(levels + 32));
    const uint32x4_t v_48 = vreinterpretq_u32_u8(vld1q_u8(levels + 48));
    const uint32x4_t lo1 = vreinterpretq_u32_u8(
        vminq_u8(vreinterpretq_u8_u32(vuzp1q_u32(v_32, v_48)), const_3));
    const uint32x4_t l0 = vshrq_n_u32(lo0, 8);
    const uint32x4_t l1 = vextq_u32(lo0, lo1, 1);
    const uint32x4_t l2 = vextq_u32(lo0, lo1, 2);
    const uint32x4_t l3 = vextq_u32(lo0, lo1, 3);
    count = vaddq_u8(
        vreinterpretq_u8_u32(l0),
        vaddq_u8(
            vaddq_u8(vreinterpretq_u8_u32(l1), vreinterpretq_u8_u32(l2)),
            vaddq_u8(vreinterpretq_u8_u32(l3), vreinterpretq_u8_u32(lo1))));
    count = vminq_u8(vrshrq_n_u8(count, 1), const_4);
    lo0 = lo1;
#else
    load_levels_4x4x5(levels, stride, offsets, level);
    count = get_coeff_contexts_kernel(level);
#endif
    count = vaddq_u8(count, pos_to_offset);
    vst1q_u8(coeff_contexts, count);
    pos_to_offset = pos_to_offset_large;
    levels += 4 * stride;
    coeff_contexts += 16;
    col -= 4;
  } while (col != 0);
}

static AOM_FORCE_INLINE void get_8_coeff_contexts_2d(const uint8_t *levels,
                                                     const int width,
                                                     const int n_cols,
                                                     uint8_t *coeff_contexts) {
  const int stride = 8 + TX_PAD_HOR;
  uint8_t *cc = coeff_contexts;
  uint8x16_t pos_to_offset[3];

  assert(n_cols > 0 && !(n_cols % 2));

  const int w_idx = (width >= 8) + (width > 8);
  pos_to_offset[0] = vld1q_u8(c_8_po_2d_all[w_idx][0]);
  pos_to_offset[1] = vld1q_u8(c_8_po_2d_all[w_idx][1]);
  pos_to_offset[2] = vdupq_n_u8(21);

  const uint8x16_t const_3 = vdupq_n_u8(3);
  const uint8x16_t const_4 = vdupq_n_u8(4);
  const uint8x16_t v01 =
      vminq_u8(vcombine_u8(vld1_u8(levels), vld1_u8(levels + stride)), const_3);
  const uint8x16_t s01_01 = vaddq_u8(v01, neon_vshr1_2d_u8x16(v01));
  uint8x16_t s12_01 = neon_vshr1_2d_u8x16(s01_01);
  uint8x16_t prev_s01 = s01_01;

  int col = n_cols;
  do {
    const uint8x16_t v23 = vminq_u8(
        vcombine_u8(vld1_u8(levels + 2 * stride), vld1_u8(levels + 3 * stride)),
        const_3);
    const uint8x16_t s01_23 = vaddq_u8(v23, neon_vshr1_2d_u8x16(v23));
    const uint8x16_t s01_12 = vextq_u8(prev_s01, s01_23, 8);
    const uint8x16_t sum = vaddq_u8(s12_01, vaddq_u8(s01_12, v23));
    const uint8x16_t count = vminq_u8(vrshrq_n_u8(sum, 1), const_4);
    vst1q_u8(cc, vaddq_u8(count, pos_to_offset[0]));
    s12_01 = neon_vshr1_2d_u8x16(s01_23);
    prev_s01 = s01_23;
    pos_to_offset[0] = pos_to_offset[1];
    pos_to_offset[1] = pos_to_offset[2];
    levels += 2 * stride;
    cc += 16;
    col -= 2;
  } while (col != 0);

  coeff_contexts[0] = 0;
}

static AOM_FORCE_INLINE void get_8_coeff_contexts_ver(const uint8_t *levels,
                                                      const int n_cols,
                                                      uint8_t *coeff_contexts) {
  const int stride = 8 + TX_PAD_HOR;
  const uint8x16_t pos_to_offset = vld1q_u8(c_8_po_ver);
  const uint8x16_t const_3 = vdupq_n_u8(3);
  const uint8x16_t const_4 = vdupq_n_u8(4);

  assert(n_cols > 0 && !(n_cols % 2));

  uint8x16_t v01 =
      vminq_u8(vcombine_u8(vld1_u8(levels), vld1_u8(levels + stride)), const_3);
  int col = n_cols;
  do {
    const uint8x16_t v23 = vminq_u8(
        vcombine_u8(vld1_u8(levels + 2 * stride), vld1_u8(levels + 3 * stride)),
        const_3);
    const uint8x16_t v12 = vextq_u8(v01, v23, 8);
    const uint8x16_t r1 = neon_vshr1_2d_u8x16(v01);
    const uint8x16_t r2 = neon_vshr1_2d_u8x16(r1);
    const uint8x16_t s12 = vaddq_u8(r1, r2);
    const uint8x16_t s34 =
        vreinterpretq_u8_u64(vshrq_n_u64(vreinterpretq_u64_u8(s12), 16));
    const uint8x16_t sum = vaddq_u8(v12, vaddq_u8(s12, s34));
    const uint8x16_t count = vminq_u8(vrshrq_n_u8(sum, 1), const_4);
    vst1q_u8(coeff_contexts, vaddq_u8(count, pos_to_offset));
    v01 = v23;
    levels += 2 * stride;
    coeff_contexts += 16;
    col -= 2;
  } while (col != 0);
}

static AOM_FORCE_INLINE void get_8_coeff_contexts_hor(const uint8_t *levels,
                                                      const int n_cols,
                                                      uint8_t *coeff_contexts) {
  const int stride = 8 + TX_PAD_HOR;
  const uint8x16_t pos_to_offset_large = vdupq_n_u8(SIG_COEF_CONTEXTS_2D + 10);
  const uint8x16_t const_3 = vdupq_n_u8(3);
  const uint8x16_t const_4 = vdupq_n_u8(4);

  uint8x16_t pos_to_offset = vcombine_u8(vdup_n_u8(SIG_COEF_CONTEXTS_2D + 0),
                                         vdup_n_u8(SIG_COEF_CONTEXTS_2D + 5));

  assert(n_cols > 0 && !(n_cols % 2));

  uint8x16_t v01 =
      vminq_u8(vcombine_u8(vld1_u8(levels), vld1_u8(levels + stride)), const_3);
  uint8x16_t v23 = vminq_u8(
      vcombine_u8(vld1_u8(levels + 2 * stride), vld1_u8(levels + 3 * stride)),
      const_3);
  uint8x16_t s12 = vaddq_u8(vextq_u8(v01, v23, 8), v23);
  int col = n_cols;
  do {
    const uint8x16_t v45 = vminq_u8(
        vcombine_u8(vld1_u8(levels + 4 * stride), vld1_u8(levels + 5 * stride)),
        const_3);
    const uint8x16_t s34 = vaddq_u8(vextq_u8(v23, v45, 8), v45);
    const uint8x16_t r1 = neon_vshr1_2d_u8x16(v01);
    const uint8x16_t sum = vaddq_u8(r1, vaddq_u8(s12, s34));
    const uint8x16_t count = vminq_u8(vrshrq_n_u8(sum, 1), const_4);
    vst1q_u8(coeff_contexts, vaddq_u8(count, pos_to_offset));
    v01 = v23;
    v23 = v45;
    s12 = s34;
    pos_to_offset = pos_to_offset_large;
    levels += 2 * stride;
    coeff_contexts += 16;
    col -= 2;
  } while (col != 0);
}

static AOM_FORCE_INLINE void get_16n_coeff_contexts_2d(
    const uint8_t *levels, const int real_width, const int real_height,
    const int n_cols, const int height, uint8_t *coeff_contexts) {
  uint8_t *cc = coeff_contexts;
  int col = n_cols;
  uint8x16_t pos_to_offset[5];
  uint8x16_t count;

  assert(n_cols > 0);
  assert(!(height % 16));

  const int w_idx = (real_width >= real_height) + (real_width > real_height);
  pos_to_offset[0] = vld1q_u8(c_16_po_2d_all[w_idx][0]);
  pos_to_offset[1] = vld1q_u8(c_16_po_2d_all[w_idx][1]);
  pos_to_offset[2] = vld1q_u8(c_16_po_2d_all[w_idx][2]);
  pos_to_offset[3] = vld1q_u8(c_16_po_2d_all[w_idx][3]);
  pos_to_offset[4] = vld1q_u8(c_16_po_2d_all[w_idx][4]);

  const uint8x16_t const_3 = vdupq_n_u8(3);
  const uint8x16_t const_4 = vdupq_n_u8(4);
  const uint8x16_t zeros = vdupq_n_u8(0);
  if (height == 16) {
    const int stride = 16 + TX_PAD_HOR;
    const uint8x16_t v_c0 = vminq_u8(vld1q_u8(levels), const_3);
    const uint8x16_t v_c1 = vminq_u8(vld1q_u8(levels + stride), const_3);
    const uint8x16_t s01_c0 = vaddq_u8(v_c0, vextq_u8(v_c0, zeros, 1));
    uint8x16_t s12_c0 = vextq_u8(s01_c0, zeros, 1);
    uint8x16_t s01_c1 = vaddq_u8(v_c1, vextq_u8(v_c1, zeros, 1));
    do {
      const uint8x16_t v_c2 = vminq_u8(vld1q_u8(levels + 2 * stride), const_3);
      count = vaddq_u8(s12_c0, vaddq_u8(s01_c1, v_c2));
      count = vminq_u8(vrshrq_n_u8(count, 1), const_4);
      count = vaddq_u8(count, pos_to_offset[0]);
      vst1q_u8(cc, count);
      s12_c0 = vextq_u8(s01_c1, zeros, 1);
      s01_c1 = vaddq_u8(v_c2, vextq_u8(v_c2, zeros, 1));
      pos_to_offset[0] = pos_to_offset[1];
      pos_to_offset[1] = pos_to_offset[2];
      pos_to_offset[2] = pos_to_offset[3];
      pos_to_offset[3] = pos_to_offset[4];
      levels += stride;
      cc += 16;
    } while (--col != 0);
  } else {
    assert(height == 32);
    const int stride = 32 + TX_PAD_HOR;
#if AOM_ARCH_AARCH64
    const uint8x16_t pto_hi_2 = vdupq_n_u8(21);
    uint8x16_t pto_hi_0 =
        (real_width > real_height) ? pos_to_offset[0] : pto_hi_2;
    uint8x16_t pto_hi_1 = pto_hi_0;
    const uint8x16_t v0_c0 = vminq_u8(vld1q_u8(levels), const_3);
    const uint8x16_t v1_c0 = vminq_u8(vld1q_u8(levels + 16), const_3);
    const uint8x16_t v0_c1 = vminq_u8(vld1q_u8(levels + stride), const_3);
    const uint8x16_t v1_c1 = vminq_u8(vld1q_u8(levels + stride + 16), const_3);
    const uint8x16_t s01_lo_c0 = vaddq_u8(v0_c0, vextq_u8(v0_c0, v1_c0, 1));
    const uint8x16_t s01_hi_c0 = vaddq_u8(v1_c0, vextq_u8(v1_c0, zeros, 1));
    uint8x16_t s12_lo_c0 = vextq_u8(s01_lo_c0, s01_hi_c0, 1);
    uint8x16_t s12_hi_c0 = vextq_u8(s01_hi_c0, zeros, 1);
    uint8x16_t s01_lo_c1 = vaddq_u8(v0_c1, vextq_u8(v0_c1, v1_c1, 1));
    uint8x16_t s01_hi_c1 = vaddq_u8(v1_c1, vextq_u8(v1_c1, zeros, 1));
    do {
      const uint8x16_t v0_c2 = vminq_u8(vld1q_u8(levels + 2 * stride), const_3);
      const uint8x16_t v1_c2 =
          vminq_u8(vld1q_u8(levels + 2 * stride + 16), const_3);
      uint8x16_t count_lo = vaddq_u8(s12_lo_c0, vaddq_u8(s01_lo_c1, v0_c2));
      uint8x16_t count_hi = vaddq_u8(s12_hi_c0, vaddq_u8(s01_hi_c1, v1_c2));
      count_lo = vminq_u8(vrshrq_n_u8(count_lo, 1), const_4);
      count_hi = vminq_u8(vrshrq_n_u8(count_hi, 1), const_4);
      vst1q_u8(cc, vaddq_u8(count_lo, pos_to_offset[0]));
      vst1q_u8(cc + 16, vaddq_u8(count_hi, pto_hi_0));
      s12_lo_c0 = vextq_u8(s01_lo_c1, s01_hi_c1, 1);
      s12_hi_c0 = vextq_u8(s01_hi_c1, zeros, 1);
      s01_lo_c1 = vaddq_u8(v0_c2, vextq_u8(v0_c2, v1_c2, 1));
      s01_hi_c1 = vaddq_u8(v1_c2, vextq_u8(v1_c2, zeros, 1));
      pos_to_offset[0] = pos_to_offset[1];
      pos_to_offset[1] = pos_to_offset[2];
      pos_to_offset[2] = pos_to_offset[3];
      pos_to_offset[3] = pos_to_offset[4];
      pto_hi_0 = pto_hi_1;
      pto_hi_1 = pto_hi_2;
      levels += stride;
      cc += 32;
    } while (--col != 0);
#else
    // Sequential 16-row passes avoid exceeding ARMv7's 16 Q-registers.
    const uint8_t *ls = levels;
    const uint8x8_t const_3_d = vget_low_u8(const_3);
    const uint8x16_t v0_c0 = vminq_u8(vld1q_u8(ls), const_3);
    const uint8x8_t v0_c0_r16 = vmin_u8(vld1_u8(ls + 16), const_3_d);
    const uint8x16_t v0_c1 = vminq_u8(vld1q_u8(ls + stride), const_3);
    const uint8x8_t v0_c1_r16 = vmin_u8(vld1_u8(ls + stride + 16), const_3_d);
    const uint8x16_t s01_c0 =
        vaddq_u8(v0_c0, vextq_u8(v0_c0, vcombine_u8(v0_c0_r16, v0_c0_r16), 1));
    const uint8x8_t s01_c0_r16 =
        vadd_u8(v0_c0_r16, vext_u8(v0_c0_r16, v0_c0_r16, 1));
    uint8x16_t s12_c0 =
        vextq_u8(s01_c0, vcombine_u8(s01_c0_r16, s01_c0_r16), 1);
    uint8x16_t s01_c1 =
        vaddq_u8(v0_c1, vextq_u8(v0_c1, vcombine_u8(v0_c1_r16, v0_c1_r16), 1));
    uint8x8_t s01_c1_r16 = vadd_u8(v0_c1_r16, vext_u8(v0_c1_r16, v0_c1_r16, 1));
    do {
      const uint8x16_t v_c2 = vminq_u8(vld1q_u8(ls + 2 * stride), const_3);
      const uint8x8_t v_c2_r16 =
          vmin_u8(vld1_u8(ls + 2 * stride + 16), const_3_d);
      count = vaddq_u8(s12_c0, vaddq_u8(s01_c1, v_c2));
      count = vminq_u8(vrshrq_n_u8(count, 1), const_4);
      vst1q_u8(cc, vaddq_u8(count, pos_to_offset[0]));
      s12_c0 = vextq_u8(s01_c1, vcombine_u8(s01_c1_r16, s01_c1_r16), 1);
      s01_c1 =
          vaddq_u8(v_c2, vextq_u8(v_c2, vcombine_u8(v_c2_r16, v_c2_r16), 1));
      s01_c1_r16 = vadd_u8(v_c2_r16, vext_u8(v_c2_r16, v_c2_r16, 1));
      pos_to_offset[0] = pos_to_offset[1];
      pos_to_offset[1] = pos_to_offset[2];
      pos_to_offset[2] = pos_to_offset[3];
      pos_to_offset[3] = pos_to_offset[4];
      ls += stride;
      cc += 32;
    } while (--col != 0);

    ls = levels + 16;
    cc = coeff_contexts;
    col = n_cols;
    const uint8x16_t pto_hi_2 = vdupq_n_u8(21);
    uint8x16_t pto_hi_0 = vdupq_n_u8((real_width > real_height) ? 16 : 21);
    uint8x16_t pto_hi_1 = pto_hi_0;
    const uint8x16_t v1_c0 = vminq_u8(vld1q_u8(ls), const_3);
    const uint8x16_t v1_c1 = vminq_u8(vld1q_u8(ls + stride), const_3);
    const uint8x16_t s01_hi_c0 = vaddq_u8(v1_c0, vextq_u8(v1_c0, zeros, 1));
    s12_c0 = vextq_u8(s01_hi_c0, zeros, 1);
    s01_c1 = vaddq_u8(v1_c1, vextq_u8(v1_c1, zeros, 1));
    do {
      const uint8x16_t v_c2 = vminq_u8(vld1q_u8(ls + 2 * stride), const_3);
      count = vaddq_u8(s12_c0, vaddq_u8(s01_c1, v_c2));
      count = vminq_u8(vrshrq_n_u8(count, 1), const_4);
      vst1q_u8(cc + 16, vaddq_u8(count, pto_hi_0));
      s12_c0 = vextq_u8(s01_c1, zeros, 1);
      s01_c1 = vaddq_u8(v_c2, vextq_u8(v_c2, zeros, 1));
      pto_hi_0 = pto_hi_1;
      pto_hi_1 = pto_hi_2;
      ls += stride;
      cc += 32;
    } while (--col != 0);
#endif  // AOM_ARCH_AARCH64
  }

  coeff_contexts[0] = 0;
}

static AOM_FORCE_INLINE void get_16n_coeff_contexts_ver(
    const uint8_t *levels, const int n_cols, const int height,
    uint8_t *coeff_contexts) {
  const uint8x16_t pos_to_offset = vld1q_u8(c_16_po_ver);
  const uint8x16_t pos_to_offset_large = vdupq_n_u8(SIG_COEF_CONTEXTS_2D + 10);
  const uint8x16_t const_3 = vdupq_n_u8(3);
  const uint8x16_t const_4 = vdupq_n_u8(4);
  const uint8x16_t zeros = vdupq_n_u8(0);

  assert(n_cols > 0);
  assert(!(height % 16));

  int col = n_cols;
  if (height == 16) {
    const int stride = 16 + TX_PAD_HOR;
    uint8x16_t v_c0 = vminq_u8(vld1q_u8(levels), const_3);
    do {
      const uint8x16_t v_c1 = vminq_u8(vld1q_u8(levels + stride), const_3);
      const uint8x16_t s12 =
          vaddq_u8(vextq_u8(v_c0, zeros, 1), vextq_u8(v_c0, zeros, 2));
      const uint8x16_t s34 = vextq_u8(s12, zeros, 2);
      uint8x16_t count = vaddq_u8(v_c1, vaddq_u8(s12, s34));
      count = vminq_u8(vrshrq_n_u8(count, 1), const_4);
      vst1q_u8(coeff_contexts, vaddq_u8(count, pos_to_offset));
      v_c0 = v_c1;
      levels += stride;
      coeff_contexts += 16;
    } while (--col != 0);
  } else {
    assert(height == 32);
    const int stride = 32 + TX_PAD_HOR;
    uint8x16_t v0_c0 = vminq_u8(vld1q_u8(levels), const_3);
    uint8x16_t v1_c0 = vminq_u8(vld1q_u8(levels + 16), const_3);
    do {
      const uint8x16_t v0_c1 = vminq_u8(vld1q_u8(levels + stride), const_3);
      const uint8x16_t v1_c1 =
          vminq_u8(vld1q_u8(levels + stride + 16), const_3);
      const uint8x16_t s12_lo =
          vaddq_u8(vextq_u8(v0_c0, v1_c0, 1), vextq_u8(v0_c0, v1_c0, 2));
      const uint8x16_t s12_hi =
          vaddq_u8(vextq_u8(v1_c0, zeros, 1), vextq_u8(v1_c0, zeros, 2));
      const uint8x16_t s34_lo = vextq_u8(s12_lo, s12_hi, 2);
      uint8x16_t count_lo = vaddq_u8(v0_c1, vaddq_u8(s12_lo, s34_lo));
      count_lo = vminq_u8(vrshrq_n_u8(count_lo, 1), const_4);
      vst1q_u8(coeff_contexts, vaddq_u8(count_lo, pos_to_offset));

      const uint8x16_t s34_hi = vextq_u8(s12_hi, zeros, 2);
      uint8x16_t count_hi = vaddq_u8(v1_c1, vaddq_u8(s12_hi, s34_hi));
      count_hi = vminq_u8(vrshrq_n_u8(count_hi, 1), const_4);
      vst1q_u8(coeff_contexts + 16, vaddq_u8(count_hi, pos_to_offset_large));

      v0_c0 = v0_c1;
      v1_c0 = v1_c1;
      levels += stride;
      coeff_contexts += 32;
    } while (--col != 0);
  }
}

static AOM_FORCE_INLINE void get_16n_coeff_contexts_hor(
    const uint8_t *levels, const int n_cols, const int height,
    uint8_t *coeff_contexts) {
  const uint8x16_t const_3 = vdupq_n_u8(3);
  const uint8x16_t const_4 = vdupq_n_u8(4);
  uint8x16_t pos_to_offset[3];
  uint8x16_t count;

  assert(n_cols > 0);
  assert(!(height % 16));

  pos_to_offset[0] = vdupq_n_u8(SIG_COEF_CONTEXTS_2D + 0);
  pos_to_offset[1] = vdupq_n_u8(SIG_COEF_CONTEXTS_2D + 5);
  pos_to_offset[2] = vdupq_n_u8(SIG_COEF_CONTEXTS_2D + 10);

  int col = n_cols;
  if (height == 16) {
    const int stride = 16 + TX_PAD_HOR;
    const uint8x16_t zeros = vdupq_n_u8(0);
    uint8x16_t v_c0 = vminq_u8(vld1q_u8(levels), const_3);
    uint8x16_t v_c1 = vminq_u8(vld1q_u8(levels + stride), const_3);
    uint8x16_t v_c2 = vminq_u8(vld1q_u8(levels + 2 * stride), const_3);
    uint8x16_t v_c3 = vminq_u8(vld1q_u8(levels + 3 * stride), const_3);
    do {
      const uint8x16_t v_c4 = vminq_u8(vld1q_u8(levels + 4 * stride), const_3);
      count = vaddq_u8(vextq_u8(v_c0, zeros, 1),
                       vaddq_u8(vaddq_u8(v_c1, v_c2), vaddq_u8(v_c3, v_c4)));
      count = vminq_u8(vrshrq_n_u8(count, 1), const_4);
      vst1q_u8(coeff_contexts, vaddq_u8(count, pos_to_offset[0]));
      v_c0 = v_c1;
      v_c1 = v_c2;
      v_c2 = v_c3;
      v_c3 = v_c4;
      pos_to_offset[0] = pos_to_offset[1];
      pos_to_offset[1] = pos_to_offset[2];
      levels += stride;
      coeff_contexts += 16;
    } while (--col != 0);
  } else {
    assert(height == 32);
    const int stride = 32 + TX_PAD_HOR;
#if AOM_ARCH_AARCH64
    const uint8x16_t zeros = vdupq_n_u8(0);
    uint8x16_t v0_c0 = vminq_u8(vld1q_u8(levels), const_3);
    uint8x16_t v1_c0 = vminq_u8(vld1q_u8(levels + 16), const_3);
    uint8x16_t v0_c1 = vminq_u8(vld1q_u8(levels + stride), const_3);
    uint8x16_t v1_c1 = vminq_u8(vld1q_u8(levels + stride + 16), const_3);
    uint8x16_t v0_c2 = vminq_u8(vld1q_u8(levels + 2 * stride), const_3);
    uint8x16_t v1_c2 = vminq_u8(vld1q_u8(levels + 2 * stride + 16), const_3);
    uint8x16_t v0_c3 = vminq_u8(vld1q_u8(levels + 3 * stride), const_3);
    uint8x16_t v1_c3 = vminq_u8(vld1q_u8(levels + 3 * stride + 16), const_3);
    do {
      const uint8x16_t v0_c4 = vminq_u8(vld1q_u8(levels + 4 * stride), const_3);
      const uint8x16_t v1_c4 =
          vminq_u8(vld1q_u8(levels + 4 * stride + 16), const_3);
      uint8x16_t count_lo =
          vaddq_u8(vextq_u8(v0_c0, v1_c0, 1),
                   vaddq_u8(vaddq_u8(v0_c1, v0_c2), vaddq_u8(v0_c3, v0_c4)));
      uint8x16_t count_hi =
          vaddq_u8(vextq_u8(v1_c0, zeros, 1),
                   vaddq_u8(vaddq_u8(v1_c1, v1_c2), vaddq_u8(v1_c3, v1_c4)));
      count_lo = vminq_u8(vrshrq_n_u8(count_lo, 1), const_4);
      count_hi = vminq_u8(vrshrq_n_u8(count_hi, 1), const_4);
      vst1q_u8(coeff_contexts, vaddq_u8(count_lo, pos_to_offset[0]));
      vst1q_u8(coeff_contexts + 16, vaddq_u8(count_hi, pos_to_offset[0]));
      v0_c0 = v0_c1;
      v1_c0 = v1_c1;
      v0_c1 = v0_c2;
      v1_c1 = v1_c2;
      v0_c2 = v0_c3;
      v1_c2 = v1_c3;
      v0_c3 = v0_c4;
      v1_c3 = v1_c4;
      pos_to_offset[0] = pos_to_offset[1];
      pos_to_offset[1] = pos_to_offset[2];
      levels += stride;
      coeff_contexts += 32;
    } while (--col != 0);
#else
    // Sequential 16-row passes avoid exceeding ARMv7's 16 Q-registers.
    const uint8_t *ls = levels;
    uint8_t *cc = coeff_contexts;
    uint8x16_t v_c0 = vminq_u8(vld1q_u8(ls), const_3);
    uint8x16_t v_c1 = vminq_u8(vld1q_u8(ls + stride), const_3);
    uint8x16_t v_c2 = vminq_u8(vld1q_u8(ls + 2 * stride), const_3);
    uint8x16_t v_c3 = vminq_u8(vld1q_u8(ls + 3 * stride), const_3);
    do {
      const uint8x16_t v_c0_hi = vminq_u8(vld1q_u8(ls + 16), const_3);
      const uint8x16_t v_c4 = vminq_u8(vld1q_u8(ls + 4 * stride), const_3);
      count = vaddq_u8(vextq_u8(v_c0, v_c0_hi, 1),
                       vaddq_u8(vaddq_u8(v_c1, v_c2), vaddq_u8(v_c3, v_c4)));
      count = vminq_u8(vrshrq_n_u8(count, 1), const_4);
      vst1q_u8(cc, vaddq_u8(count, pos_to_offset[0]));
      v_c0 = v_c1;
      v_c1 = v_c2;
      v_c2 = v_c3;
      v_c3 = v_c4;
      pos_to_offset[0] = pos_to_offset[1];
      pos_to_offset[1] = pos_to_offset[2];
      ls += stride;
      cc += 32;
    } while (--col != 0);

    const uint8x16_t zeros = vdupq_n_u8(0);
    pos_to_offset[0] = vdupq_n_u8(SIG_COEF_CONTEXTS_2D + 0);
    pos_to_offset[1] = vdupq_n_u8(SIG_COEF_CONTEXTS_2D + 5);
    ls = levels + 16;
    cc = coeff_contexts;
    col = n_cols;
    v_c0 = vminq_u8(vld1q_u8(ls), const_3);
    v_c1 = vminq_u8(vld1q_u8(ls + stride), const_3);
    v_c2 = vminq_u8(vld1q_u8(ls + 2 * stride), const_3);
    v_c3 = vminq_u8(vld1q_u8(ls + 3 * stride), const_3);
    do {
      const uint8x16_t v_c4 = vminq_u8(vld1q_u8(ls + 4 * stride), const_3);
      count = vaddq_u8(vextq_u8(v_c0, zeros, 1),
                       vaddq_u8(vaddq_u8(v_c1, v_c2), vaddq_u8(v_c3, v_c4)));
      count = vminq_u8(vrshrq_n_u8(count, 1), const_4);
      vst1q_u8(cc + 16, vaddq_u8(count, pos_to_offset[0]));
      v_c0 = v_c1;
      v_c1 = v_c2;
      v_c2 = v_c3;
      v_c3 = v_c4;
      pos_to_offset[0] = pos_to_offset[1];
      pos_to_offset[1] = pos_to_offset[2];
      ls += stride;
      cc += 32;
    } while (--col != 0);
#endif  // AOM_ARCH_AARCH64
  }
}

// Upper bound on (scan[c] >> bhl) + 1 for c < active_eob (rounded up to a
// whole column count) over default_scan_{8x16,8x32,16x16,16x32,32x16,32x32}
// (all TX_CLASS_2D blocks use the default diagonal scan; 8x16/8x32 with
// active_eob <= 28 also use this table), indexed by (active_eob - 1) >> 3 for
// active_eob <= 8 * sizeof(c_diag_cols_by8) (496).
static const uint8_t c_diag_cols_by8[62] = {
  4,  6,  7,  8,  9,  10, 11, 11, 12, 13, 13, 14, 14, 15, 15, 16,
  16, 17, 17, 18, 18, 19, 19, 20, 20, 20, 21, 21, 22, 22, 22, 23,
  23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 26, 27, 27, 27, 27, 28,
  28, 28, 29, 29, 29, 29, 30, 30, 30, 30, 31, 31, 31, 31
};

// Returns a column-granular upper bound on the active column count required
// for coeff_contexts[scan[c]] across 0 <= c < active_eob in TX_CLASS_2D
// (all TX_CLASS_2D transform types use default_scan_*). Contexts outside
// the computed columns remain uninitialized, matching
// av1_get_nz_map_contexts_c.
static inline int nz_diag_n_cols_2d(const int active_eob, const int width,
                                    const int height) {
  assert(active_eob >= 1);
  if (height == 4) {
    // default_scan_{8x4,16x4} enter col 4 at active_eob=14, col 8 at 30, and
    // col 12 at 46, aligning directly with 4-column granularity.
    return AOMMIN(width, 4 + (((active_eob + 2) >> 4) << 2));
  }
  if (height == 8) {
    if (width == 4) return 4;
    // default_scan_{8x8,16x8,32x8} reach cols 2, 4, 6, 8 at active_eob =
    // 6, 15, 28, 44; thereafter +2 cols every 16 coeffs (16x8/32x8).
    if (active_eob <= 5) return 2;
    if (active_eob <= 14) return 4;
    if (active_eob <= 27) return 6;
    if (width == 8 || active_eob <= 43) return 8;
    return AOMMIN(width, 8 + (((active_eob - 28) >> 4) << 1));
  }
  if (active_eob <= 6) return (active_eob <= 3) ? 2 : 3;
  // default_scan_4x16 reaches col 3 (4 cols) at active_eob=7.
  if (width == 4) return 4;
  // default_scan_{8x16,8x32} reach col 7 (8 cols) by active_eob=29.
  if (width == 8 && active_eob > 28) return 8;
  // default_scan_32x16 enters col 16 at active_eob=152, then +1 col per 16
  // coeffs.
  if (height == 16 && active_eob > 151) {
    return AOMMIN(width, 16 + ((active_eob - 136) >> 4));
  }
  // 32x32 reaches col 32 at active_eob=497 (after 8*62=496 coeffs).
  if (active_eob > 8 * (int)sizeof(c_diag_cols_by8)) return width;
  return AOMMIN(width, c_diag_cols_by8[(active_eob - 1) >> 3]);
}

// Note: levels[] must be in the range [0, 127], inclusive; TX_PAD_HOR padding
// lanes at levels[col * stride + height .. col * stride + stride - 1] and
// TX_PAD_BOTTOM trailing columns at
// levels[width * stride .. (width + TX_PAD_BOTTOM) * stride - 1] must be zero,
// and scan must match av1_scan_orders[tx_size][tx_type].scan.
void av1_get_nz_map_contexts_neon(const uint8_t *const levels,
                                  const int16_t *const scan, const int eob,
                                  const TX_SIZE tx_size,
                                  const TX_CLASS tx_class,
                                  int8_t *const coeff_contexts) {
  assert(TX_PAD_HOR == 4 && TX_PAD_BOTTOM == 4);
  assert(eob >= 1);
  assert(tx_size < TX_SIZES_ALL);
  const int last_idx = eob - 1;
  if (!last_idx) {
    coeff_contexts[0] = 0;
    return;
  }
  if (last_idx == 1 && tx_class == TX_CLASS_2D) {
    coeff_contexts[0] = 0;
    coeff_contexts[scan[1]] = 1;
    return;
  }

  uint8_t *const coefficients = (uint8_t *const)coeff_contexts;

  /* coeff_contexts must be 16 byte aligned. */
  assert(!((intptr_t)coeff_contexts & 0xf));

  if (tx_size == TX_4X4 && tx_class == TX_CLASS_2D) {
    assert(eob <= 16);
    get_4x4_nz_map_contexts_2d(levels, coefficients);
    const int pos = scan[last_idx];
    coeff_contexts[pos] = (last_idx <= 2) ? 1 : ((last_idx <= 4) ? 2 : 3);
    return;
  }

  const int width = get_txb_wide(tx_size);
  const int height = get_txb_high(tx_size);
  const int bhl = get_txb_bhl(tx_size);
  assert(eob <= (width << bhl));

  if (tx_class == TX_CLASS_2D) {
    if (height == 4) {
      assert(width > 4);
      const int n_cols = nz_diag_n_cols_2d(last_idx, width, 4);
      get_4_nz_map_contexts_2d(levels, n_cols, coefficients);
    } else {
      const int n_cols = nz_diag_n_cols_2d(last_idx, width, height);
      if (height == 8) {
        get_8_coeff_contexts_2d(levels, width, n_cols, coefficients);
      } else {
        const int real_width = tx_size_wide[tx_size];
        const int real_height = tx_size_high[tx_size];
        get_16n_coeff_contexts_2d(levels, real_width, real_height, n_cols,
                                  height, coefficients);
      }
    }
  } else if (tx_class == TX_CLASS_HORIZ) {
    const int n_cols = ((last_idx - 1) >> bhl) + 1;
    if (height == 4) {
      get_4_nz_map_contexts_hor(levels, (n_cols + 3) & ~3, coefficients);
    } else if (height == 8) {
      get_8_coeff_contexts_hor(levels, (n_cols + 1) & ~1, coefficients);
    } else {
      get_16n_coeff_contexts_hor(levels, n_cols, height, coefficients);
    }
  } else {
    assert(tx_class == TX_CLASS_VERT);
    const int n_cols = AOMMIN(width, last_idx);
    if (height == 4) {
      get_4_nz_map_contexts_ver(levels, (n_cols + 3) & ~3, coefficients);
    } else if (height == 8) {
      get_8_coeff_contexts_ver(levels, (n_cols + 1) & ~1, coefficients);
    } else {
      get_16n_coeff_contexts_ver(levels, n_cols, height, coefficients);
    }
  }

  const int pos = scan[last_idx];
  if (last_idx <= (width << bhl) / 8)
    coeff_contexts[pos] = 1;
  else if (last_idx <= (width << bhl) / 4)
    coeff_contexts[pos] = 2;
  else
    coeff_contexts[pos] = 3;
}
