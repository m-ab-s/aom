/*
 * Copyright (c) 2001-2016, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#ifndef AOM_AOM_DSP_ENTENC_H_
#define AOM_AOM_DSP_ENTENC_H_
#include <stddef.h>
#include "aom_dsp/entcode.h"
#include "aom_util/endian_inl.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t od_ec_enc_window;

typedef struct od_ec_enc od_ec_enc;

#define OD_MEASURE_EC_OVERHEAD (0)

#if OD_MEASURE_EC_OVERHEAD
#include <math.h>
#if !defined(M_LOG2E)
#define M_LOG2E (1.4426950408889634073599246810019)
#endif
#define OD_LOG2(x) (M_LOG2E * log(x))
#endif  // OD_MEASURE_EC_OVERHEAD

/*The entropy encoder context.*/
struct od_ec_enc {
  /*Buffered output.
    This contains only the raw bits until the final call to od_ec_enc_done(),
     where all the arithmetic-coded data gets prepended to it.*/
  unsigned char *buf;
  /*The size of the buffer.*/
  uint32_t storage;
  /*The offset at which the next entropy-coded byte will be written.*/
  uint32_t offs;
  /*The low end of the current range.*/
  od_ec_enc_window low;
  /*The number of values in the current range.*/
  uint16_t rng;
  /*The number of bits of data in the current value.*/
  int16_t cnt;
  /*Nonzero if an error occurred.*/
  int error;
#if OD_MEASURE_EC_OVERHEAD
  double entropy;
  int nb_symbols;
#endif
};

/*The part of the encoder context that changes with every symbol.
  A caller that writes many symbols in a row can keep a copy of it in local
   variables, so it stays in registers: take it with od_ec_enc_get_state(),
   use the od_ec_encode_*_state() functions and put it back with
   od_ec_enc_set_state() before any other call on the encoder.*/
typedef struct od_ec_enc_state {
  /*The low end of the current range.*/
  od_ec_enc_window low;
  /*The number of values in the current range.*/
  unsigned rng;
  /*The number of bits of data in the current value.*/
  int cnt;
} od_ec_enc_state;

/*See entenc.c for further documentation.*/

void od_ec_enc_init(od_ec_enc *enc, uint32_t size) OD_ARG_NONNULL(1);
void od_ec_enc_reset(od_ec_enc *enc) OD_ARG_NONNULL(1);
void od_ec_enc_clear(od_ec_enc *enc) OD_ARG_NONNULL(1);

void od_ec_encode_bool_q15(od_ec_enc *enc, int val, unsigned f_q15)
    OD_ARG_NONNULL(1);
void od_ec_encode_cdf_q15(od_ec_enc *enc, int s, const uint16_t *cdf, int nsyms)
    OD_ARG_NONNULL(1) OD_ARG_NONNULL(3);

void od_ec_enc_bits(od_ec_enc *enc, uint32_t fl, unsigned ftb)
    OD_ARG_NONNULL(1);

OD_WARN_UNUSED_RESULT unsigned char *od_ec_enc_done(od_ec_enc *enc,
                                                    uint32_t *nbytes)
    OD_ARG_NONNULL(1) OD_ARG_NONNULL(2);

OD_WARN_UNUSED_RESULT int od_ec_enc_tell(const od_ec_enc *enc)
    OD_ARG_NONNULL(1);
OD_WARN_UNUSED_RESULT uint32_t od_ec_enc_tell_frac(const od_ec_enc *enc)
    OD_ARG_NONNULL(1);

// buf is the frame bitbuffer, offs is where carry to be added
static inline void propagate_carry_bwd(unsigned char *buf, uint32_t offs) {
  uint16_t sum, carry = 1;
  do {
    sum = (uint16_t)buf[offs] + 1;
    buf[offs--] = (unsigned char)sum;
    carry = sum >> 8;
  } while (carry);
}

// Convert to big-endian byte order and write data to buffer adding the
// carry-bit
static inline void write_enc_data_to_out_buf(unsigned char *out, uint32_t offs,
                                             uint64_t output, uint64_t carry,
                                             uint32_t *enc_offs,
                                             uint8_t num_bytes_ready) {
  const uint64_t reg = HToBE64(output << ((8 - num_bytes_ready) << 3));
  memcpy(&out[offs], &reg, 8);
  // Propagate carry backwards if exists
  if (carry) {
    assert(offs > 0);
    propagate_carry_bwd(out, offs - 1);
  }
  *enc_offs = offs + num_bytes_ready;
}

OD_WARN_UNUSED_RESULT od_ec_enc_window od_ec_enc_flush(od_ec_enc *enc,
                                                       od_ec_enc_window low,
                                                       int c, int s)
    OD_ARG_NONNULL(1);

static inline od_ec_enc_state od_ec_enc_get_state(const od_ec_enc *enc) {
  od_ec_enc_state st;
  st.low = enc->low;
  st.rng = enc->rng;
  st.cnt = enc->cnt;
  return st;
}

static inline void od_ec_enc_set_state(od_ec_enc *enc,
                                       const od_ec_enc_state *st) {
  enc->low = st->low;
  enc->rng = (uint16_t)st->rng;
  enc->cnt = (int16_t)st->cnt;
}

/*Takes updated low and range values, renormalizes them so that
   32768 <= rng < 65536 (flushing bytes from low to the output buffer if
   necessary), and stores them back in the state.
  low: The new value of low.
  rng: The new value of the range.*/
static AOM_FORCE_INLINE void od_ec_enc_normalize_state(od_ec_enc *enc,
                                                       od_ec_enc_state *st,
                                                       od_ec_enc_window low,
                                                       unsigned rng) {
  assert(rng > 0 && rng <= 65535U);
  /*The number of leading zeros in the 16-bit binary representation of rng.
    get_msb(rng) is at most 15, so the xor is 15 - get_msb(rng).*/
  const int d = 15 ^ get_msb(rng);
  int s = st->cnt + d;
  /*The limit is explained in od_ec_enc_flush().*/
  if (UNLIKELY(s >= 40)) {
    low = od_ec_enc_flush(enc, low, st->cnt, s);
    /*The flush leaves s - 8 * ((s >> 3) + 1) bits.*/
    s = (s & 7) - 8;
  }
  st->low = low << d;
  st->rng = rng << d;
  st->cnt = s;
}

/*Encode a single binary value.
  val: The value to encode (0 or 1).
  f: The probability that the val is one, scaled by 32768.*/
static AOM_FORCE_INLINE void od_ec_encode_bool_q15_state(od_ec_enc *enc,
                                                         od_ec_enc_state *st,
                                                         int val, unsigned f) {
  od_ec_enc_window l;
  unsigned r;
  unsigned v;
  assert(0 < f);
  assert(f < 32768U);
  l = st->low;
  r = st->rng;
  assert(32768U <= r);
  v = ((r >> 8) * (uint32_t)(f >> EC_PROB_SHIFT) >> (7 - EC_PROB_SHIFT));
  v += EC_MIN_PROB;
  if (val) l += r - v;
  r = val ? v : r - v;
  od_ec_enc_normalize_state(enc, st, l, r);
#if OD_MEASURE_EC_OVERHEAD
  enc->entropy -= OD_LOG2((double)(val ? f : (32768 - f)) / 32768.);
  enc->nb_symbols++;
#endif
}

/*Encodes a symbol given a cumulative distribution function (CDF) table in Q15,
   like od_ec_encode_cdf_q15().
  No branch on s, because coefficient levels are hard to predict.
  s: The index of the symbol to encode.
  icdf: 32768 minus the CDF, such that symbol s falls in the range
         [s > 0 ? (32768 - icdf[s - 1]) : 0, 32768 - icdf[s]).
        The values must be monotonically decreasing, and icdf[nsyms - 1] must
         be 0.
  nsyms: The number of symbols in the alphabet.
         This should be at most 16.*/
static AOM_FORCE_INLINE void od_ec_encode_cdf_q15_state(od_ec_enc *enc,
                                                        od_ec_enc_state *st,
                                                        int s,
                                                        const uint16_t *icdf,
                                                        int nsyms) {
  const unsigned r = st->rng;
  assert(s >= 0);
  assert(s < nsyms);
  assert(icdf[nsyms - 1] == OD_ICDF(CDF_PROB_TOP));
  assert(32768U <= r);
  assert(7 - EC_PROB_SHIFT >= 0);
  const int N = nsyms - 1;
  /*icdf[s - 1] for s > 0, and a valid entry to read for s == 0.*/
  const unsigned fl = icdf[s - (s > 0)];
  const unsigned fh = icdf[s];
  const unsigned r8 = r >> 8;
  const unsigned pu =
      r8 * (uint32_t)(fl >> EC_PROB_SHIFT) >> (7 - EC_PROB_SHIFT);
  const unsigned pv =
      r8 * (uint32_t)(fh >> EC_PROB_SHIFT) >> (7 - EC_PROB_SHIFT);
  /*s spans [u, v) below rng, with u = pu + EC_MIN_PROB * (N - s + 1) and
     v = pv + EC_MIN_PROB * (N - s). um is u without their common term. For
     s == 0 (or fl == CDF_PROB_TOP) u is rng, so low stays as it is. Doing
     the select first keeps the chain from rng to the new rng short.*/
  const unsigned rk = r - EC_MIN_PROB * (N - s);
  const unsigned um = (s > 0 && fl < CDF_PROB_TOP) ? pu + EC_MIN_PROB : rk;
  od_ec_enc_normalize_state(enc, st, st->low + (rk - um), um - pv);
#if OD_MEASURE_EC_OVERHEAD
  enc->entropy -= OD_LOG2(
      (double)(OD_ICDF(fh) - OD_ICDF(s > 0 ? fl : OD_ICDF(0))) / CDF_PROB_TOP);
  enc->nb_symbols++;
#endif
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AOM_DSP_ENTENC_H_
