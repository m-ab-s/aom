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

#ifndef AOM_AV1_ENCODER_AV1_QUANTIZE_HWY_H_
#define AOM_AV1_ENCODER_AV1_QUANTIZE_HWY_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"
#include "config/av1_rtcd.h"

#include "aom/aom_integer.h"
#include "aom_dsp/reduce_sum_hwy.h"
#include "third_party/highway/hwy/highway.h"

HWY_BEFORE_NAMESPACE();

namespace {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

constexpr hn::ScalableTag<int16_t> int16xN_tag;
constexpr hn::ScalableTag<uint16_t> uint16xN_tag;
constexpr hn::ScalableTag<int32_t> int32xN_tag;
constexpr hn::ScalableTag<uint32_t> uint32xN_tag;
constexpr hn::ScalableTag<uint64_t> uint64xN_tag;

constexpr hn::FixedTag<int16_t, 8> int16x8_tag;
constexpr hn::FixedTag<int64_t, 2> int64x2_tag;

using IVec16 = hn::Vec<decltype(int16xN_tag)>;

// Negates elements of `a` where `b < 0`, sets to 0 where `b == 0`,
// and keeps `a` where `b > 0`.
template <typename D>
HWY_ATTR HWY_INLINE hn::Vec<D> Sign16(D d, hn::Vec<D> a, hn::Vec<D> b) {
  const auto zero = hn::Zero(d);
  const auto nz_mask = hn::Ne(b, zero);
  const auto neg_mask = hn::Gt(zero, b);
  const auto a_nz = hn::IfThenElseZero(nz_mask, a);
  return hn::IfThenElse(neg_mask, hn::Sub(zero, a_nz), a_nz);
}

template <int kShift>
HWY_ATTR HWY_INLINE IVec16 MulShift16(IVec16 a, IVec16 b) {
  static_assert(kShift > 0 && kShift < 16, "kShift must be in (0, 16)");
  const auto hi =
      hn::ShiftLeft<16 - kShift>(hn::BitCast(uint16xN_tag, hn::MulHigh(a, b)));
  const auto lo =
      hn::ShiftRight<kShift>(hn::BitCast(uint16xN_tag, hn::Mul(a, b)));
  return hn::BitCast(int16xN_tag, hn::Or(hi, lo));
}

// Broadcasts 128-bit [DC, AC, AC, AC, AC, AC, AC, AC] into full vector
// [DC, AC, AC, ...] where block 0 has [DC, AC...] and blocks 1..K-1 have
// [AC, AC...].
HWY_ATTR HWY_INLINE IVec16 InitOneQp(hn::Vec<decltype(int16x8_tag)> p128) {
  const auto p64 = hn::BitCast(int64x2_tag, p128);
  const auto ac64 = hn::InterleaveUpper(int64x2_tag, p64, p64);
  const auto ac128 = hn::BitCast(int16x8_tag, ac64);
  const auto ac_full =
      hn::BroadcastBlock<0>(hn::ResizeBitCast(int16xN_tag, ac128));
  return hn::InsertBlock<0>(ac_full, p128);
}

// Updates a vector from [DC, AC, AC, ...] to [AC, AC, AC, ...] by broadcasting
// 128-bit block 1 across all blocks (or upper 64-bit half on 128-bit targets).
template <typename D = decltype(int16xN_tag)>
HWY_ATTR HWY_INLINE hn::VFromD<D> UpdateQp(hn::VFromD<D> v) {
  constexpr D d;
  if constexpr (d.MaxBlocks() == 1) {
    const auto v64 = hn::BitCast(int64x2_tag, v);
    return hn::BitCast(d, hn::InterleaveUpper(int64x2_tag, v64, v64));
  } else {
    return hn::BroadcastBlock<1>(v);
  }
}

template <int log_scale>
HWY_ATTR HWY_INLINE void InitQpFp(const int16_t *round_ptr,
                                  const int16_t *quant_ptr,
                                  const int16_t *dequant_ptr, IVec16 *thr,
                                  IVec16 *round, IVec16 *quant,
                                  IVec16 *dequant) {
  static_assert(log_scale >= 0 && log_scale <= 2, "Unexpected log_scale");
  auto round128 = hn::LoadU(int16x8_tag, round_ptr);
  const auto quant128 = hn::LoadU(int16x8_tag, quant_ptr);
  const auto dequant128 = hn::LoadU(int16x8_tag, dequant_ptr);

  if constexpr (log_scale > 0) {
    const auto rnd =
        hn::Set(int16x8_tag, static_cast<int16_t>(1 << (log_scale - 1)));
    round128 = hn::ShiftRight<log_scale>(hn::Add(round128, rnd));
  }

  *round = InitOneQp(round128);
  *quant = InitOneQp(quant128);
  if constexpr (log_scale == 1) {
    *quant = hn::ShiftLeft<1>(*quant);
  }
  *dequant = InitOneQp(dequant128);
  *thr =
      hn::Sub(hn::ShiftRight<1 + log_scale>(*dequant), hn::Set(int16xN_tag, 1));
}

HWY_ATTR HWY_INLINE void UpdateQpFp(IVec16 *thr, IVec16 *round, IVec16 *quant,
                                    IVec16 *dequant) {
  *round = UpdateQp(*round);
  *quant = UpdateQp(*quant);
  *dequant = UpdateQp(*dequant);
  *thr = UpdateQp(*thr);
}

template <int log_scale>
HWY_ATTR HWY_INLINE void LoadBValues(const int16_t *zbin_ptr, IVec16 *zbin,
                                     const int16_t *round_ptr, IVec16 *round,
                                     const int16_t *quant_ptr, IVec16 *quant,
                                     const int16_t *dequant_ptr,
                                     IVec16 *dequant, const int16_t *shift_ptr,
                                     IVec16 *shift) {
  static_assert(log_scale >= 0 && log_scale <= 2, "Unexpected log_scale");
  auto zbin128 = hn::LoadU(int16x8_tag, zbin_ptr);
  auto round128 = hn::LoadU(int16x8_tag, round_ptr);
  const auto quant128 = hn::LoadU(int16x8_tag, quant_ptr);
  const auto dequant128 = hn::LoadU(int16x8_tag, dequant_ptr);
  const auto shift128 = hn::LoadU(int16x8_tag, shift_ptr);

  if constexpr (log_scale > 0) {
    const auto rnd =
        hn::Set(int16x8_tag, static_cast<int16_t>(1 << (log_scale - 1)));
    zbin128 = hn::ShiftRight<log_scale>(hn::Add(zbin128, rnd));
    round128 = hn::ShiftRight<log_scale>(hn::Add(round128, rnd));
  }
  zbin128 = hn::Sub(zbin128, hn::Set(int16x8_tag, 1));

  *zbin = InitOneQp(zbin128);
  *round = InitOneQp(round128);
  *quant = InitOneQp(quant128);
  *dequant = InitOneQp(dequant128);
  *shift = InitOneQp(shift128);
}

HWY_ATTR HWY_INLINE void UpdateQpB(IVec16 *zbin, IVec16 *round, IVec16 *quant,
                                   IVec16 *dequant, IVec16 *shift) {
  *zbin = UpdateQp(*zbin);
  *round = UpdateQp(*round);
  *quant = UpdateQp(*quant);
  *dequant = UpdateQp(*dequant);
  *shift = UpdateQp(*shift);
}

HWY_ATTR HWY_INLINE IVec16 LoadCoefficients(const tran_low_t *coeff_ptr) {
  constexpr intptr_t kStep32 = hn::MaxLanes(int32xN_tag);
  const auto c0 = hn::LoadU(int32xN_tag, coeff_ptr);
  const auto c1 = hn::LoadU(int32xN_tag, coeff_ptr + kStep32);
  return hn::ReorderDemote2To(int16xN_tag, c0, c1);
}

HWY_ATTR HWY_INLINE void StoreCoefficients(IVec16 coeff_vals,
                                           tran_low_t *coeff_ptr) {
  constexpr intptr_t kStep32 = hn::MaxLanes(int32xN_tag);
  const auto coeff_sign = hn::ShiftRight<15>(coeff_vals);
  const auto coeff_lo =
      hn::BitCast(int32xN_tag, hn::InterleaveLower(coeff_vals, coeff_sign));
  const auto coeff_hi = hn::BitCast(
      int32xN_tag, hn::InterleaveUpper(int16xN_tag, coeff_vals, coeff_sign));
  hn::StoreU(coeff_lo, int32xN_tag, coeff_ptr);
  hn::StoreU(coeff_hi, int32xN_tag, coeff_ptr + kStep32);
}

HWY_ATTR HWY_INLINE void StoreZeroCoefficients(tran_low_t *qcoeff_ptr,
                                               tran_low_t *dqcoeff_ptr) {
  constexpr intptr_t kStep32 = hn::MaxLanes(int32xN_tag);
  const auto zero = hn::Zero(int32xN_tag);
  hn::StoreU(zero, int32xN_tag, qcoeff_ptr);
  hn::StoreU(zero, int32xN_tag, qcoeff_ptr + kStep32);
  hn::StoreU(zero, int32xN_tag, dqcoeff_ptr);
  hn::StoreU(zero, int32xN_tag, dqcoeff_ptr + kStep32);
}

template <typename D>
HWY_ATTR HWY_INLINE auto BlockReduceMax(D d, hn::VFromD<D> v) {
  if constexpr (d.MaxBlocks() == 1) {
    return v;
  } else {
    constexpr hn::Half<D> half_d;
    return BlockReduceMax(
        half_d, hn::Max(hn::LowerHalf(half_d, v), hn::UpperHalf(half_d, v)));
  }
}

HWY_ATTR HWY_INLINE uint16_t QuantGatherEob(IVec16 eob) {
  constexpr hn::FixedTag<uint16_t, 8> uint16x8_tag;
  const auto eob128 = BlockReduceMax(int16xN_tag, eob);
  return hn::ReduceMax(uint16x8_tag, hn::BitCast(uint16x8_tag, eob128));
}

// Reorders iscan to match the lane layout produced by ReorderDemote2To(c0, c1)
// where c0 and c1 are 32-bit coefficient vectors.
HWY_ATTR HWY_INLINE IVec16
GetMaxLaneEob(const int16_t *iscan, IVec16 v_eobmax,
              hn::Mask<decltype(int16xN_tag)> nz_mask) {
  const auto v_iscan = hn::LoadU(int16xN_tag, iscan);
  constexpr size_t kLanes64 = hn::MaxLanes(uint64xN_tag);
  static_assert(kLanes64 == 8 || kLanes64 == 4 || kLanes64 == 2,
                "Unexpected vector width");
  IVec16 v_iscan_perm;
  if constexpr (kLanes64 == 8) {
    alignas(
        64) static constexpr uint64_t kPermIdx[8] = { 0, 4, 1, 5, 2, 6, 3, 7 };
    const auto idx = hn::SetTableIndices(uint64xN_tag, kPermIdx);
    v_iscan_perm = hn::BitCast(
        int16xN_tag,
        hn::TableLookupLanes(hn::BitCast(uint64xN_tag, v_iscan), idx));
  } else if constexpr (kLanes64 == 4) {
    alignas(32) static constexpr uint64_t kPermIdx[4] = { 0, 2, 1, 3 };
    const auto idx = hn::SetTableIndices(uint64xN_tag, kPermIdx);
    v_iscan_perm = hn::BitCast(
        int16xN_tag,
        hn::TableLookupLanes(hn::BitCast(uint64xN_tag, v_iscan), idx));
  } else {
    v_iscan_perm = v_iscan;
  }
  const auto v_iscan_plus1 = hn::Add(v_iscan_perm, hn::Set(int16xN_tag, 1));
  const auto v_nz_iscan = hn::IfThenElseZero(nz_mask, v_iscan_plus1);
  return hn::Max(v_eobmax, v_nz_iscan);
}

HWY_ATTR HWY_INLINE void QuantizeLpStep(const int16_t *coeff_ptr,
                                        const int16_t *iscan_ptr,
                                        int16_t *qcoeff_ptr,
                                        int16_t *dqcoeff_ptr, IVec16 round,
                                        IVec16 quant, IVec16 dequant,
                                        IVec16 *eob) {
  const auto coeff = hn::LoadU(int16xN_tag, coeff_ptr);
  const auto abs_coeff = hn::Abs(coeff);
  const auto tmp_rnd = hn::SaturatedAdd(abs_coeff, round);
  const auto abs_qcoeff = hn::MulHigh(tmp_rnd, quant);
  const auto qcoeff = Sign16(int16xN_tag, abs_qcoeff, coeff);
  const auto dqcoeff = hn::Mul(qcoeff, dequant);
  const auto nz_mask = hn::Gt(abs_qcoeff, hn::Zero(int16xN_tag));

  hn::StoreU(qcoeff, int16xN_tag, qcoeff_ptr);
  hn::StoreU(dqcoeff, int16xN_tag, dqcoeff_ptr);

  const auto iscan = hn::LoadU(int16xN_tag, iscan_ptr);
  const auto nz_iscan =
      hn::IfThenElseZero(nz_mask, hn::Add(iscan, hn::Set(int16xN_tag, 1)));
  *eob = hn::Max(*eob, nz_iscan);
}

HWY_ATTR HWY_INLINE void QuantizeLp(const int16_t *coeff_ptr, intptr_t n_coeffs,
                                    const int16_t *round_ptr,
                                    const int16_t *quant_ptr,
                                    int16_t *qcoeff_ptr, int16_t *dqcoeff_ptr,
                                    const int16_t *dequant_ptr,
                                    uint16_t *eob_ptr, const int16_t *scan,
                                    const int16_t *iscan) {
  constexpr intptr_t kStep = hn::MaxLanes(int16xN_tag);
  if (n_coeffs < kStep) {
    av1_quantize_lp_avx2(coeff_ptr, n_coeffs, round_ptr, quant_ptr, qcoeff_ptr,
                         dqcoeff_ptr, dequant_ptr, eob_ptr, scan, iscan);
    return;
  }
  assert(n_coeffs >= kStep && n_coeffs % kStep == 0);
  (void)scan;

  IVec16 eob = hn::Zero(int16xN_tag);
  IVec16 round = InitOneQp(hn::LoadU(int16x8_tag, round_ptr));
  IVec16 quant = InitOneQp(hn::LoadU(int16x8_tag, quant_ptr));
  IVec16 dequant = InitOneQp(hn::LoadU(int16x8_tag, dequant_ptr));

  QuantizeLpStep(coeff_ptr, iscan, qcoeff_ptr, dqcoeff_ptr, round, quant,
                 dequant, &eob);

  if (n_coeffs > kStep) {
    round = UpdateQp(round);
    quant = UpdateQp(quant);
    dequant = UpdateQp(dequant);

    for (intptr_t idx = kStep; idx < n_coeffs; idx += kStep) {
      QuantizeLpStep(coeff_ptr + idx, iscan + idx, qcoeff_ptr + idx,
                     dqcoeff_ptr + idx, round, quant, dequant, &eob);
    }
  }

  *eob_ptr = QuantGatherEob(eob);
}

template <int log_scale>
HWY_ATTR HWY_INLINE void QuantizeFpStep(IVec16 thr, IVec16 round, IVec16 quant,
                                        IVec16 dequant,
                                        const tran_low_t *coeff_ptr,
                                        const int16_t *iscan_ptr,
                                        tran_low_t *qcoeff_ptr,
                                        tran_low_t *dqcoeff_ptr, IVec16 *eob) {
  static_assert(log_scale >= 0 && log_scale <= 2, "Unexpected log_scale");
  const auto coeff = LoadCoefficients(coeff_ptr);
  const auto abs_coeff = hn::Abs(coeff);
  const auto mask = hn::Gt(abs_coeff, thr);

  if (!hn::AllFalse(int16xN_tag, mask)) {
    const auto tmp_rnd =
        hn::IfThenElseZero(mask, hn::SaturatedAdd(abs_coeff, round));
    IVec16 abs_q;
    IVec16 q;
    IVec16 dq;
    hn::Mask<decltype(int16xN_tag)> nz_mask;

    if constexpr (log_scale == 0) {
      abs_q = hn::MulHigh(tmp_rnd, quant);
      q = Sign16(int16xN_tag, abs_q, coeff);
      dq = hn::Mul(q, dequant);
      nz_mask = hn::Gt(abs_q, hn::Zero(int16xN_tag));
    } else if constexpr (log_scale == 1) {
      abs_q = hn::BitCast(int16xN_tag,
                          hn::MulHigh(hn::BitCast(uint16xN_tag, tmp_rnd),
                                      hn::BitCast(uint16xN_tag, quant)));
      q = Sign16(int16xN_tag, abs_q, coeff);
      const auto abs_dq =
          hn::BitCast(int16xN_tag, hn::ShiftRight<1>(hn::BitCast(
                                       uint16xN_tag, hn::Mul(abs_q, dequant))));
      dq = Sign16(int16xN_tag, abs_dq, coeff);
      nz_mask = hn::Gt(abs_q, hn::Zero(int16xN_tag));
    } else {
      abs_q = MulShift16<14>(tmp_rnd, quant);
      const auto abs_dq = MulShift16<2>(abs_q, dequant);
      q = Sign16(int16xN_tag, abs_q, coeff);
      dq = Sign16(int16xN_tag, abs_dq, coeff);
      nz_mask = hn::Ne(dq, hn::Zero(int16xN_tag));
    }

    StoreCoefficients(q, qcoeff_ptr);
    StoreCoefficients(dq, dqcoeff_ptr);
    *eob = GetMaxLaneEob(iscan_ptr, *eob, nz_mask);
  } else {
    StoreZeroCoefficients(qcoeff_ptr, dqcoeff_ptr);
  }
}

template <int log_scale>
HWY_ATTR HWY_INLINE void QuantizeFpTemplate(
    const tran_low_t *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr,
    const int16_t *round_ptr, const int16_t *quant_ptr,
    const int16_t *quant_shift_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr,
    const int16_t *scan_ptr, const int16_t *iscan_ptr) {
  constexpr intptr_t kStep = hn::MaxLanes(int16xN_tag);
  if constexpr (log_scale == 0) {
    if (n_coeffs < kStep) {
      av1_quantize_fp_avx2(coeff_ptr, n_coeffs, zbin_ptr, round_ptr, quant_ptr,
                           quant_shift_ptr, qcoeff_ptr, dqcoeff_ptr,
                           dequant_ptr, eob_ptr, scan_ptr, iscan_ptr);
      return;
    }
  }
  assert(n_coeffs >= kStep && n_coeffs % kStep == 0);
  (void)scan_ptr;
  (void)zbin_ptr;
  (void)quant_shift_ptr;

  IVec16 thr, round, quant, dequant;
  IVec16 eob = hn::Zero(int16xN_tag);

  InitQpFp<log_scale>(round_ptr, quant_ptr, dequant_ptr, &thr, &round, &quant,
                      &dequant);

  QuantizeFpStep<log_scale>(thr, round, quant, dequant, coeff_ptr, iscan_ptr,
                            qcoeff_ptr, dqcoeff_ptr, &eob);

  if (n_coeffs > kStep) {
    UpdateQpFp(&thr, &round, &quant, &dequant);

    for (intptr_t idx = kStep; idx < n_coeffs; idx += kStep) {
      QuantizeFpStep<log_scale>(thr, round, quant, dequant, coeff_ptr + idx,
                                iscan_ptr + idx, qcoeff_ptr + idx,
                                dqcoeff_ptr + idx, &eob);
    }
  }
  *eob_ptr = QuantGatherEob(eob);
}

template <int log_scale>
HWY_ATTR HWY_INLINE void QuantizeBStep(const tran_low_t *coeff_ptr,
                                       tran_low_t *qcoeff_ptr,
                                       tran_low_t *dqcoeff_ptr, IVec16 v_quant,
                                       IVec16 v_dequant, IVec16 v_round,
                                       IVec16 v_zbin, IVec16 v_quant_shift,
                                       const int16_t *iscan_ptr,
                                       IVec16 *v_eobmax) {
  static_assert(log_scale >= 0 && log_scale <= 2, "Unexpected log_scale");
  const auto v_coeff = LoadCoefficients(coeff_ptr);
  const auto v_abs_coeff = hn::Abs(v_coeff);
  const auto zbin_mask = hn::Gt(v_abs_coeff, v_zbin);

  if (hn::AllFalse(int16xN_tag, zbin_mask)) {
    StoreZeroCoefficients(qcoeff_ptr, dqcoeff_ptr);
    return;
  }

  const auto v_tmp_rnd =
      hn::IfThenElseZero(zbin_mask, hn::SaturatedAdd(v_abs_coeff, v_round));
  const auto v_tmp32_a = hn::MulHigh(v_tmp_rnd, v_quant);
  const auto v_tmp32_b = hn::Add(v_tmp32_a, v_tmp_rnd);

  IVec16 v_tmp32;
  IVec16 v_qcoeff;
  IVec16 v_dqcoeff;
  if constexpr (log_scale == 0) {
    v_tmp32 = hn::MulHigh(v_tmp32_b, v_quant_shift);
    v_qcoeff = Sign16(int16xN_tag, v_tmp32, v_coeff);
    v_dqcoeff = hn::Mul(v_qcoeff, v_dequant);
  } else {
    v_tmp32 = MulShift16<16 - log_scale>(v_tmp32_b, v_quant_shift);
    v_dqcoeff =
        Sign16(int16xN_tag, MulShift16<log_scale>(v_tmp32, v_dequant), v_coeff);
    v_qcoeff = Sign16(int16xN_tag, v_tmp32, v_coeff);
  }

  const auto nz_mask = hn::Gt(v_tmp32, hn::Zero(int16xN_tag));
  StoreCoefficients(v_qcoeff, qcoeff_ptr);
  StoreCoefficients(v_dqcoeff, dqcoeff_ptr);
  *v_eobmax = GetMaxLaneEob(iscan_ptr, *v_eobmax, nz_mask);
}

template <int log_scale>
HWY_ATTR HWY_INLINE void QuantizeBTemplate(
    const tran_low_t *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr,
    const int16_t *round_ptr, const int16_t *quant_ptr,
    const int16_t *quant_shift_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr,
    const int16_t *scan, const int16_t *iscan) {
  constexpr intptr_t kStep = hn::MaxLanes(int16xN_tag);
  if constexpr (log_scale == 0) {
    if (n_coeffs < kStep) {
      aom_quantize_b_avx2(coeff_ptr, n_coeffs, zbin_ptr, round_ptr, quant_ptr,
                          quant_shift_ptr, qcoeff_ptr, dqcoeff_ptr, dequant_ptr,
                          eob_ptr, scan, iscan);
      return;
    }
  }
  assert(n_coeffs >= kStep && n_coeffs % kStep == 0);
  (void)scan;

  IVec16 v_zbin, v_round, v_quant, v_dequant, v_quant_shift;
  IVec16 v_eobmax = hn::Zero(int16xN_tag);

  LoadBValues<log_scale>(zbin_ptr, &v_zbin, round_ptr, &v_round, quant_ptr,
                         &v_quant, dequant_ptr, &v_dequant, quant_shift_ptr,
                         &v_quant_shift);

  QuantizeBStep<log_scale>(coeff_ptr, qcoeff_ptr, dqcoeff_ptr, v_quant,
                           v_dequant, v_round, v_zbin, v_quant_shift, iscan,
                           &v_eobmax);

  if (n_coeffs > kStep) {
    UpdateQpB(&v_zbin, &v_round, &v_quant, &v_dequant, &v_quant_shift);

    for (intptr_t idx = kStep; idx < n_coeffs; idx += kStep) {
      QuantizeBStep<log_scale>(coeff_ptr + idx, qcoeff_ptr + idx,
                               dqcoeff_ptr + idx, v_quant, v_dequant, v_round,
                               v_zbin, v_quant_shift, iscan + idx, &v_eobmax);
    }
  }

  *eob_ptr = QuantGatherEob(v_eobmax);
}

HWY_ATTR HWY_INLINE int64_t BlockError(const tran_low_t *coeff,
                                       const tran_low_t *dqcoeff,
                                       intptr_t block_size, int64_t *ssz) {
  constexpr intptr_t kStep = hn::MaxLanes(int16xN_tag);
  if (block_size < kStep) {
    return av1_block_error_avx2(coeff, dqcoeff, block_size, ssz);
  }
  assert(block_size >= kStep && block_size % kStep == 0);

  auto sse_reg = hn::Zero(uint64xN_tag);
  auto ssz_reg = hn::Zero(uint64xN_tag);
  for (intptr_t i = 0; i < block_size; i += kStep) {
    const auto coeff_reg = LoadCoefficients(coeff + i);
    const auto dqcoeff_reg = LoadCoefficients(dqcoeff + i);
    const auto diff = hn::Sub(dqcoeff_reg, coeff_reg);
    const auto err = hn::BitCast(
        uint32xN_tag, hn::WidenMulPairwiseAdd(int32xN_tag, diff, diff));
    const auto sq_coeff =
        hn::BitCast(uint32xN_tag,
                    hn::WidenMulPairwiseAdd(int32xN_tag, coeff_reg, coeff_reg));

    sse_reg = hn::Add(sse_reg, hn::Add(hn::PromoteEvenTo(uint64xN_tag, err),
                                       hn::PromoteOddTo(uint64xN_tag, err)));
    ssz_reg =
        hn::Add(ssz_reg, hn::Add(hn::PromoteEvenTo(uint64xN_tag, sq_coeff),
                                 hn::PromoteOddTo(uint64xN_tag, sq_coeff)));
  }
  constexpr hn::FixedTag<uint64_t, 2> du128;
  const auto sse128 = BlockReduceSum(uint64xN_tag, sse_reg);
  const auto ssz128 = BlockReduceSum(uint64xN_tag, ssz_reg);
  const auto sum = hn::Add(hn::InterleaveLower(du128, sse128, ssz128),
                           hn::InterleaveUpper(du128, sse128, ssz128));
  *ssz = static_cast<int64_t>(hn::ExtractLane(sum, 1));
  return static_cast<int64_t>(hn::ExtractLane(sum, 0));
}

}  // namespace HWY_NAMESPACE
}  // namespace

HWY_AFTER_NAMESPACE();

#define MAKE_AV1_QUANTIZE_LP(suffix)                                         \
  extern "C" HWY_ATTR void av1_quantize_lp_##suffix(                         \
      const int16_t *coeff_ptr, intptr_t n_coeffs, const int16_t *round_ptr, \
      const int16_t *quant_ptr, int16_t *qcoeff_ptr, int16_t *dqcoeff_ptr,   \
      const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan,    \
      const int16_t *iscan) {                                                \
    HWY_NAMESPACE::QuantizeLp(coeff_ptr, n_coeffs, round_ptr, quant_ptr,     \
                              qcoeff_ptr, dqcoeff_ptr, dequant_ptr, eob_ptr, \
                              scan, iscan);                                  \
  }

#define MAKE_AV1_QUANTIZE_FP(suffix)                                           \
  extern "C" HWY_ATTR void av1_quantize_fp_##suffix(                           \
      const tran_low_t *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, \
      const int16_t *round_ptr, const int16_t *quant_ptr,                      \
      const int16_t *quant_shift_ptr, tran_low_t *qcoeff_ptr,                  \
      tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr,  \
      const int16_t *scan, const int16_t *iscan) {                             \
    HWY_NAMESPACE::QuantizeFpTemplate<0>(                                      \
        coeff_ptr, n_coeffs, zbin_ptr, round_ptr, quant_ptr, quant_shift_ptr,  \
        qcoeff_ptr, dqcoeff_ptr, dequant_ptr, eob_ptr, scan, iscan);           \
  }

#define MAKE_AV1_QUANTIZE_FP_WXH(w, h, suffix)                                 \
  extern "C" HWY_ATTR void av1_quantize_fp_##w##x##h##_##suffix(               \
      const tran_low_t *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, \
      const int16_t *round_ptr, const int16_t *quant_ptr,                      \
      const int16_t *quant_shift_ptr, tran_low_t *qcoeff_ptr,                  \
      tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr,  \
      const int16_t *scan, const int16_t *iscan) {                             \
    static_assert(((w) == 32 && (h) == 32) || ((w) == 64 && (h) == 64),        \
                  "Unsupported block size");                                   \
    HWY_NAMESPACE::QuantizeFpTemplate<((w) == 32 ? 1 : 2)>(                    \
        coeff_ptr, n_coeffs, zbin_ptr, round_ptr, quant_ptr, quant_shift_ptr,  \
        qcoeff_ptr, dqcoeff_ptr, dequant_ptr, eob_ptr, scan, iscan);           \
  }

#define MAKE_AOM_QUANTIZE_B(suffix)                                            \
  extern "C" HWY_ATTR void aom_quantize_b_##suffix(                            \
      const tran_low_t *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, \
      const int16_t *round_ptr, const int16_t *quant_ptr,                      \
      const int16_t *quant_shift_ptr, tran_low_t *qcoeff_ptr,                  \
      tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr,  \
      const int16_t *scan, const int16_t *iscan) {                             \
    HWY_NAMESPACE::QuantizeBTemplate<0>(                                       \
        coeff_ptr, n_coeffs, zbin_ptr, round_ptr, quant_ptr, quant_shift_ptr,  \
        qcoeff_ptr, dqcoeff_ptr, dequant_ptr, eob_ptr, scan, iscan);           \
  }

#define MAKE_AOM_QUANTIZE_B_WXH(w, h, suffix)                                  \
  extern "C" HWY_ATTR void aom_quantize_b_##w##x##h##_##suffix(                \
      const tran_low_t *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, \
      const int16_t *round_ptr, const int16_t *quant_ptr,                      \
      const int16_t *quant_shift_ptr, tran_low_t *qcoeff_ptr,                  \
      tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr,  \
      const int16_t *scan, const int16_t *iscan) {                             \
    static_assert(((w) == 32 && (h) == 32) || ((w) == 64 && (h) == 64),        \
                  "Unsupported block size");                                   \
    HWY_NAMESPACE::QuantizeBTemplate<((w) == 32 ? 1 : 2)>(                     \
        coeff_ptr, n_coeffs, zbin_ptr, round_ptr, quant_ptr, quant_shift_ptr,  \
        qcoeff_ptr, dqcoeff_ptr, dequant_ptr, eob_ptr, scan, iscan);           \
  }

#define MAKE_AV1_BLOCK_ERROR(suffix)                                           \
  extern "C" HWY_ATTR int64_t av1_block_error_##suffix(                        \
      const tran_low_t *coeff, const tran_low_t *dqcoeff, intptr_t block_size, \
      int64_t *ssz) {                                                          \
    return HWY_NAMESPACE::BlockError(coeff, dqcoeff, block_size, ssz);         \
  }

#endif  // AOM_AV1_ENCODER_AV1_QUANTIZE_HWY_H_
