/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#ifndef AOM_AOM_DSP_SUBTRACT_HWY_H_
#define AOM_AOM_DSP_SUBTRACT_HWY_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"
#include "third_party/highway/hwy/highway.h"

HWY_BEFORE_NAMESPACE();

namespace {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

HWY_ATTR HWY_INLINE void SubtractBlock4xN(
    int rows, int16_t *diff_ptr, ptrdiff_t diff_stride, const uint8_t *src_ptr,
    ptrdiff_t src_stride, const uint8_t *pred_ptr, ptrdiff_t pred_stride) {
  assert(rows >= 0 && rows % 2 == 0);
  constexpr hn::FixedTag<uint8_t, 4> u8x4_tag;
  constexpr hn::FixedTag<uint8_t, 8> u8x8_tag;
  constexpr hn::FixedTag<int16_t, 8> i16x8_tag;
  constexpr hn::FixedTag<int16_t, 4> i16x4_tag;

  for (int r = 0; r < rows; r += 2) {
    const auto s01 =
        hn::Combine(u8x8_tag, hn::LoadU(u8x4_tag, src_ptr + src_stride),
                    hn::LoadU(u8x4_tag, src_ptr));
    const auto p01 =
        hn::Combine(u8x8_tag, hn::LoadU(u8x4_tag, pred_ptr + pred_stride),
                    hn::LoadU(u8x4_tag, pred_ptr));
    const auto d01 =
        hn::Sub(hn::PromoteTo(i16x8_tag, s01), hn::PromoteTo(i16x8_tag, p01));
    hn::StoreU(hn::LowerHalf(i16x4_tag, d01), i16x4_tag, diff_ptr);
    hn::StoreU(hn::UpperHalf(i16x4_tag, d01), i16x4_tag,
               diff_ptr + diff_stride);
    src_ptr += src_stride * 2;
    pred_ptr += pred_stride * 2;
    diff_ptr += diff_stride * 2;
  }
}

HWY_ATTR HWY_INLINE void SubtractBlock8xN(
    int rows, int16_t *diff_ptr, ptrdiff_t diff_stride, const uint8_t *src_ptr,
    ptrdiff_t src_stride, const uint8_t *pred_ptr, ptrdiff_t pred_stride) {
  assert(rows >= 0 && rows % 2 == 0);
  constexpr hn::FixedTag<uint8_t, 8> u8x8_tag;
  constexpr hn::FixedTag<uint8_t, 16> u8x16_tag;
  constexpr hn::FixedTag<int16_t, 16> i16x16_tag;
  constexpr hn::FixedTag<int16_t, 8> i16x8_tag;

  for (int r = 0; r < rows; r += 2) {
    const auto s01 =
        hn::Combine(u8x16_tag, hn::LoadU(u8x8_tag, src_ptr + src_stride),
                    hn::LoadU(u8x8_tag, src_ptr));
    const auto p01 =
        hn::Combine(u8x16_tag, hn::LoadU(u8x8_tag, pred_ptr + pred_stride),
                    hn::LoadU(u8x8_tag, pred_ptr));
    const auto d01 =
        hn::Sub(hn::PromoteTo(i16x16_tag, s01), hn::PromoteTo(i16x16_tag, p01));
    hn::StoreU(hn::LowerHalf(i16x8_tag, d01), i16x8_tag, diff_ptr);
    hn::StoreU(hn::UpperHalf(i16x8_tag, d01), i16x8_tag,
               diff_ptr + diff_stride);
    src_ptr += src_stride * 2;
    pred_ptr += pred_stride * 2;
    diff_ptr += diff_stride * 2;
  }
}

HWY_ATTR HWY_INLINE void SubtractBlock16xN(
    int rows, int16_t *diff_ptr, ptrdiff_t diff_stride, const uint8_t *src_ptr,
    ptrdiff_t src_stride, const uint8_t *pred_ptr, ptrdiff_t pred_stride) {
  assert(rows >= 0 && rows % 2 == 0);
  constexpr hn::FixedTag<uint8_t, 16> u8x16_tag;
  constexpr hn::FixedTag<int16_t, 16> i16x16_tag;
  for (int r = 0; r < rows; r += 2) {
    const auto s0 = hn::PromoteTo(i16x16_tag, hn::LoadU(u8x16_tag, src_ptr));
    const auto p0 = hn::PromoteTo(i16x16_tag, hn::LoadU(u8x16_tag, pred_ptr));
    const auto s1 =
        hn::PromoteTo(i16x16_tag, hn::LoadU(u8x16_tag, src_ptr + src_stride));
    const auto p1 =
        hn::PromoteTo(i16x16_tag, hn::LoadU(u8x16_tag, pred_ptr + pred_stride));
    hn::StoreU(hn::Sub(s0, p0), i16x16_tag, diff_ptr);
    hn::StoreU(hn::Sub(s1, p1), i16x16_tag, diff_ptr + diff_stride);
    src_ptr += src_stride * 2;
    pred_ptr += pred_stride * 2;
    diff_ptr += diff_stride * 2;
  }
}

template <int Cols>
HWY_ATTR HWY_INLINE void SubtractBlockWide(
    int rows, int16_t *diff_ptr, ptrdiff_t diff_stride, const uint8_t *src_ptr,
    ptrdiff_t src_stride, const uint8_t *pred_ptr, ptrdiff_t pred_stride) {
  constexpr int kMaxLanes = (Cols == 128) ? 16 : 32;
  constexpr hn::CappedTag<int16_t, kMaxLanes> i16_tag;
  constexpr hn::Rebind<uint8_t, decltype(i16_tag)> u8_tag;
  constexpr int kLanes = hn::MaxLanes(i16_tag);
  for (int r = 0; r < rows; ++r) {
    HWY_UNROLL(8)
    for (int c = 0; c < Cols; c += kLanes) {
      const auto s = hn::PromoteTo(i16_tag, hn::LoadU(u8_tag, src_ptr + c));
      const auto p = hn::PromoteTo(i16_tag, hn::LoadU(u8_tag, pred_ptr + c));
      hn::StoreU(hn::Sub(s, p), i16_tag, diff_ptr + c);
    }
    src_ptr += src_stride;
    pred_ptr += pred_stride;
    diff_ptr += diff_stride;
  }
}

HWY_ATTR HWY_INLINE void SubtractBlock(int rows, int cols, int16_t *diff_ptr,
                                       ptrdiff_t diff_stride,
                                       const uint8_t *src_ptr,
                                       ptrdiff_t src_stride,
                                       const uint8_t *pred_ptr,
                                       ptrdiff_t pred_stride) {
  switch (cols) {
    case 4:
      SubtractBlock4xN(rows, diff_ptr, diff_stride, src_ptr, src_stride,
                       pred_ptr, pred_stride);
      break;
    case 8:
      SubtractBlock8xN(rows, diff_ptr, diff_stride, src_ptr, src_stride,
                       pred_ptr, pred_stride);
      break;
    case 16:
      SubtractBlock16xN(rows, diff_ptr, diff_stride, src_ptr, src_stride,
                        pred_ptr, pred_stride);
      break;
    case 32:
      SubtractBlockWide<32>(rows, diff_ptr, diff_stride, src_ptr, src_stride,
                            pred_ptr, pred_stride);
      break;
    case 64:
      SubtractBlockWide<64>(rows, diff_ptr, diff_stride, src_ptr, src_stride,
                            pred_ptr, pred_stride);
      break;
    case 128:
      SubtractBlockWide<128>(rows, diff_ptr, diff_stride, src_ptr, src_stride,
                             pred_ptr, pred_stride);
      break;
    default:
      aom_subtract_block_c(rows, cols, diff_ptr, diff_stride, src_ptr,
                           src_stride, pred_ptr, pred_stride);
      break;
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace

HWY_AFTER_NAMESPACE();

#define MAKE_SUBTRACT_BLOCK(suffix)                                          \
  extern "C" void aom_subtract_block_##suffix(                               \
      int rows, int cols, int16_t *diff_ptr, ptrdiff_t diff_stride,          \
      const uint8_t *src_ptr, ptrdiff_t src_stride, const uint8_t *pred_ptr, \
      ptrdiff_t pred_stride);                                                \
  HWY_ATTR void aom_subtract_block_##suffix(                                 \
      int rows, int cols, int16_t *diff_ptr, ptrdiff_t diff_stride,          \
      const uint8_t *src_ptr, ptrdiff_t src_stride, const uint8_t *pred_ptr, \
      ptrdiff_t pred_stride) {                                               \
    HWY_NAMESPACE::SubtractBlock(rows, cols, diff_ptr, diff_stride, src_ptr, \
                                 src_stride, pred_ptr, pred_stride);         \
  }

#endif  // AOM_AOM_DSP_SUBTRACT_HWY_H_
