/*
 * Copyright (c) 2020, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>

#include "tools/lanczos/lanczos_resample.h"

/* Shift down with rounding for use when n >= 0, value >= 0 */
#define ROUND_POWER_OF_TWO(value, n) (((value) + (((1 << (n)) >> 1))) >> (n))

/* Shift down with rounding for signed integers, for use when n >= 0 */
#define ROUND_POWER_OF_TWO_SIGNED(value, n)           \
  (((value) < 0) ? -ROUND_POWER_OF_TWO(-(value), (n)) \
                 : ROUND_POWER_OF_TWO((value), (n)))

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

double get_centered_x0(int p, int q) { return (double)(q - p) / (2 * p); }

double get_inverse_x0(int p, int q, double x0) {
  if (x0 == (double)('c')) x0 = get_centered_x0(p, q);
  return -x0 * p / q;
}

static inline int16_t doclip(int16_t x, int low, int high) {
  return (x < low ? low : x > high ? high : x);
}

void show_resample_filter(RationalResampleFilter *rf) {
  printf("------------------\n");
  printf("Resample factor %d / %d\n", rf->p, rf->q);
  printf("Start = %d\n", rf->start);
  printf("Steps = ");
  for (int i = 0; i < rf->p; ++i) {
    printf("%d, ", rf->steps[i]);
  }
  printf("\n");
  printf("Phases = ");
  for (int i = 0; i < rf->p; ++i) {
    printf("%f, ", rf->phases[i]);
  }
  printf("\n");
  printf("Filters [length %d]:\n", rf->length);
  for (int i = 0; i < rf->p; ++i) {
    printf("  { ");
    for (int j = 0; j < rf->length; ++j) printf("%d, ", rf->filter[i][j]);
    printf("  }\n");
  }
  printf("------------------\n");
}

static double sinc(double x) {
  if (fabs(x) < 1e-12) return 1.0;
  return sin(M_PI * x) / (M_PI * x);
}

static double lanczos(double x, int a) {
  const double absx = fabs(x);
  if (absx < (double)a) {
    return sinc(x) * sinc(x / a);
  } else {
    return 0.0;
  }
}

static int get_lanczos_downsampler_filter_length(int p, int q, int a) {
  assert(p < q);
  return 2 * ((a * q + p - 1) / p);
}

static int get_lanczos_upsampler_filter_length(int p, int q, int a) {
  (void)p;
  (void)q;
  assert(p >= q);
  return 2 * a;
}

static void integerize_array(double *x, int len, int bits, int16_t *y) {
  int sumy = 0;
  for (int i = 0; i < len; ++i) {
    y[i] = (int16_t)rint(x[i] * (1 << bits));
    sumy += y[i];
  }
  while (sumy > (1 << bits)) {
    double mx = -65536.0;
    int imx = -1;
    for (int i = 0; i < len; ++i) {
      const double v = (double)y[i] - (x[i] * (1 << bits));
      if (v > mx) {
        mx = v;
        imx = i;
      }
    }
    y[imx] -= 1;
    sumy -= 1;
  }
  while (sumy < (1 << bits)) {
    double mx = 65536.0;
    int imx = -1;
    for (int i = 0; i < len; ++i) {
      const double v = (double)y[i] - (x[i] * (1 << bits));
      if (v < mx) {
        mx = v;
        imx = i;
      }
    }
    y[imx] += 1;
    sumy += 1;
  }
  sumy = 0;
  for (int i = 0; i < len; ++i) {
    sumy += y[i];
  }
  assert(sumy == (1 << bits));
}

static void get_lanczos_downsampler(double x, int p, int q, int a, int bits,
                                    int16_t *ifilter) {
  double filter[MAX_FILTER_LEN] = { 0.0 };
  int tapsby2 = get_lanczos_downsampler_filter_length(p, q, a) / 2;
  assert(tapsby2 * 2 <= MAX_FILTER_LEN);
  double filter_sum = 0;
  for (int i = -tapsby2 + 1; i <= tapsby2; ++i) {
    const double tap = lanczos((i - x) * p / q, a);
    filter[i + tapsby2 - 1] = tap;
    filter_sum += tap;
  }
  assert(filter_sum != 0.0);
  for (int i = -tapsby2 + 1; i <= tapsby2; ++i) {
    filter[i + tapsby2 - 1] /= filter_sum;
  }
  integerize_array(filter, 2 * tapsby2, bits, ifilter);
}

static void get_lanczos_upsampler(double x, int p, int q, int a, int bits,
                                  int16_t *ifilter) {
  double filter[MAX_FILTER_LEN] = { 0.0 };
  int tapsby2 = get_lanczos_upsampler_filter_length(p, q, a) / 2;
  assert(tapsby2 * 2 <= MAX_FILTER_LEN);
  double filter_sum = 0;
  for (int i = -tapsby2 + 1; i <= tapsby2; ++i) {
    const double tap = lanczos(i - x, a);
    filter[i + tapsby2 - 1] = tap;
    filter_sum += tap;
  }
  assert(filter_sum != 0.0);
  for (int i = -tapsby2 + 1; i <= tapsby2; ++i) {
    filter[i + tapsby2 - 1] /= filter_sum;
  }
  integerize_array(filter, 2 * tapsby2, bits, ifilter);
}

static int gcd(int p, int q) {
  int p1 = (p < q ? p : q);
  int q1 = (p1 == p ? q : p);
  while (p1) {
    const int t = p1;
    p1 = q1 % p1;
    q1 = t;
  }
  return q1;
}

void get_resample_filter(int p, int q, int a, double x0, int bits,
                         RationalResampleFilter *rf) {
  double offset[MAX_RATIONAL_FACTOR + 1];
  int intpel[MAX_RATIONAL_FACTOR];
  assert(p > 0 && p <= MAX_RATIONAL_FACTOR);
  assert(q > 0 && q <= MAX_RATIONAL_FACTOR);
  const int g = gcd(p, q);
  assert(g > 0);
  rf->p = p / g;
  rf->q = q / g;
  if (x0 == (double)('c')) x0 = get_centered_x0(rf->p, rf->q);
  rf->filter_bits = bits;
  for (int i = 0; i < rf->p; ++i) {
    offset[i] = (double)rf->q / (double)rf->p * i + x0;
    intpel[i] = (int)floor(offset[i]);
    rf->phases[i] = offset[i] - intpel[i];
  }
  offset[rf->p] = rf->q + x0;
  intpel[rf->p] = (int)floor(offset[rf->p]);

  rf->start = intpel[0];
  for (int i = 0; i < rf->p; ++i) rf->steps[i] = intpel[i + 1] - intpel[i];
  if (rf->p < rf->q) {  // downsampling
    rf->length = get_lanczos_downsampler_filter_length(rf->p, rf->q, a);
    for (int i = 0; i < rf->p; ++i) {
      get_lanczos_downsampler(rf->phases[i], rf->p, rf->q, a, bits,
                              rf->filter[i]);
    }
  } else if (rf->p >= rf->q) {  // upsampling
    rf->length = get_lanczos_upsampler_filter_length(rf->p, rf->q, a);
    for (int i = 0; i < rf->p; ++i) {
      get_lanczos_upsampler(rf->phases[i], rf->p, rf->q, a, bits,
                            rf->filter[i]);
    }
  }
}

int is_resampler_noop(RationalResampleFilter *rf) {
  return (rf->p == 1 && rf->q == 1 && rf->phases[0] == 0.0);
}

void get_resample_filter_inv(int p, int q, int a, double x0, int bits,
                             RationalResampleFilter *rf) {
  double y0 = get_inverse_x0(p, q, x0);
  get_resample_filter(q, p, a, y0, bits, rf);
}

// Assume x buffer is already extended on both sides with x pointing to the
// leftmost pixel, and the extension values are already filled up.
static void resample_1d_core(const int16_t *x, int inlen,
                             RationalResampleFilter *rf, int downshift,
                             ClipProfile *clip, int16_t *y, int outlen) {
  (void)inlen;
  const int tapsby2 = rf->length / 2;
  const int16_t *xext = x;
  xext += rf->start;
  for (int i = 0, p = 0; i < outlen; ++i, p = (p + 1) % rf->p) {
    int sum = 0;
    for (int j = -tapsby2 + 1; j <= tapsby2; ++j) {
      sum += (int)rf->filter[p][j + tapsby2 - 1] * (int)xext[j];
    }
    y[i] = ROUND_POWER_OF_TWO_SIGNED(sum, downshift);
    if (clip) {
      y[i] = (int16_t)clip->issigned ? doclip(y[i], -(1 << (clip->bits - 1)),
                                              (1 << (clip->bits - 1)) - 1)
                                     : doclip(y[i], 0, (1 << clip->bits) - 1);
    }
    xext += rf->steps[p];
  }
}

static void extend_border(int16_t *x, int inlen, int border) {
  for (int i = -border; i < 0; ++i) x[i] = x[0];
  for (int i = 0; i < border; ++i) x[i + inlen] = x[inlen - 1];
}

// Assume x buffer is already extended on both sides with x pointing to the
// leftmost pixel, but the extension values are not filled up.
static void resample_1d_xt(int16_t *x, int inlen, RationalResampleFilter *rf,
                           int downshift, ClipProfile *clip, int16_t *y,
                           int outlen) {
  extend_border(x, inlen, rf->length / 2);
  resample_1d_core(x, inlen, rf, downshift, clip, y, outlen);
}

// Assume a scratch buffer xext of size inlen + rf->length is provided
static void resample_1d_xc(const int16_t *x, int inlen,
                           RationalResampleFilter *rf, int downshift,
                           ClipProfile *clip, int16_t *y, int outlen,
                           int16_t *xext) {
  memcpy(xext, x, sizeof(*x) * inlen);

  resample_1d_xt(xext, inlen, rf, downshift, clip, y, outlen);
}

static void fill_col_to_arr(const int16_t *img, int stride, int len,
                            int16_t *arr) {
  int i;
  const int16_t *iptr = img;
  int16_t *aptr = arr;
  for (i = 0; i < len; ++i, iptr += stride) {
    *aptr++ = *iptr;
  }
}

static void fill_arr_to_col(int16_t *img, int stride, int len,
                            const int16_t *arr) {
  int i;
  int16_t *iptr = img;
  const int16_t *aptr = arr;
  for (i = 0; i < len; ++i, iptr += stride) {
    *iptr = *aptr++;
  }
}

void resample_1d(const int16_t *x, int inlen, RationalResampleFilter *rf,
                 int downshift, ClipProfile *clip, int16_t *y, int outlen) {
  const int tapsby2 = rf->length / 2;
  int16_t *xext_ = (int16_t *)malloc((inlen + rf->length) * sizeof(*x));
  int16_t *xext = xext_ + tapsby2;

  resample_1d_xc(xext, inlen, rf, downshift, clip, y, outlen, xext);

  free(xext_);
}

void resample_2d(const int16_t *x, int inwidth, int inheight, int instride,
                 RationalResampleFilter *rfh, RationalResampleFilter *rfv,
                 int int_extra_bits, ClipProfile *clip, int16_t *y,
                 int outwidth, int outheight, int outstride) {
  if (rfv == NULL || is_resampler_noop(rfv)) {
    resample_horz(x, inwidth, inheight, instride, rfh, clip, y, outwidth,
                  outstride);
    return;
  }
  if (rfh == NULL || is_resampler_noop(rfh)) {
    resample_vert(x, inwidth, inheight, instride, rfv, clip, y, outheight,
                  outstride);
    return;
  }
  int16_t *tmpbuf = (int16_t *)malloc(sizeof(int16_t) * outwidth * inheight);
  const int arrsize =
      outheight + ((inheight + rfv->length > inwidth + rfh->length)
                       ? (inheight + rfv->length)
                       : (inwidth + rfh->length));
  int16_t *tmparr_ = (int16_t *)calloc(arrsize, sizeof(int16_t));
  int16_t *tmparrh = tmparr_ + outheight + rfh->length / 2;
  int16_t *tmparrv = tmparr_ + outheight + rfv->length / 2;
  int16_t *tmparro = tmparr_;
  int tmpstride = outwidth;
  const int downshifth = rfh->filter_bits - int_extra_bits;
  const int downshiftv = rfh->filter_bits + int_extra_bits;
  for (int i = 0; i < inheight; ++i) {
    resample_1d_xc(x + instride * i, inwidth, rfh, downshifth, NULL,
                   tmpbuf + i * tmpstride, outwidth, tmparrh);
  }
  for (int i = 0; i < outwidth; ++i) {
    fill_col_to_arr(tmpbuf + i, outwidth, inheight, tmparrv);
    resample_1d_xt(tmparrv, inheight, rfv, downshiftv, clip, tmparro,
                   outheight);
    fill_arr_to_col(y + i, outstride, outheight, tmparro);
  }
  free(tmpbuf);
  free(tmparr_);
}

void resample_horz(const int16_t *x, int inwidth, int inheight, int instride,
                   RationalResampleFilter *rfh, ClipProfile *clip, int16_t *y,
                   int outwidth, int outstride) {
  const int arrsize = inwidth + rfh->length;
  int16_t *tmparr_ = (int16_t *)calloc(arrsize, sizeof(int16_t));
  int16_t *tmparrh = tmparr_ + rfh->length / 2;
  for (int i = 0; i < inheight; ++i) {
    resample_1d_xc(x + instride * i, inwidth, rfh, rfh->filter_bits, clip,
                   y + i * outstride, outwidth, tmparrh);
  }
  free(tmparr_);
}

void resample_vert(const int16_t *x, int inwidth, int inheight, int instride,
                   RationalResampleFilter *rfv, ClipProfile *clip, int16_t *y,
                   int outheight, int outstride) {
  const int arrsize = outheight + inheight + rfv->length;
  int16_t *tmparr_ = (int16_t *)calloc(arrsize, sizeof(int16_t));
  int16_t *tmparrv = tmparr_ + outheight + rfv->length / 2;
  int16_t *tmparro = tmparr_;
  for (int i = 0; i < inwidth; ++i) {
    fill_col_to_arr(x + i, instride, inheight, tmparrv);
    resample_1d_xt(tmparrv, inheight, rfv, rfv->filter_bits, clip, tmparro,
                   outheight);
    fill_arr_to_col(y + i, outstride, outheight, tmparro);
  }
  free(tmparr_);
}

// Assume x buffer is already extended on both sides with x pointing to the
// leftmost pixel, and the extension values are already filled up.
static void resample_1d_core_in8b(const uint8_t *x, int inlen,
                                  RationalResampleFilter *rf, int downshift,
                                  ClipProfile *clip, int16_t *y, int outlen) {
  (void)inlen;
  const int tapsby2 = rf->length / 2;
  const uint8_t *xext = x;
  xext += rf->start;
  for (int i = 0, p = 0; i < outlen; ++i, p = (p + 1) % rf->p) {
    int sum = 0;
    for (int j = -tapsby2 + 1; j <= tapsby2; ++j) {
      sum += (int)rf->filter[p][j + tapsby2 - 1] * (int)xext[j];
    }
    y[i] = (int16_t)ROUND_POWER_OF_TWO_SIGNED(sum, downshift);
    if (clip) {
      y[i] = (int16_t)clip->issigned ? doclip(y[i], -(1 << (clip->bits - 1)),
                                              (1 << (clip->bits - 1)) - 1)
                                     : doclip(y[i], 0, (1 << clip->bits) - 1);
    }
    xext += rf->steps[p];
  }
}

// Assume x buffer is already extended on both sides with x pointing to the
// leftmost pixel, and the extension values are already filled up.
static void resample_1d_core_8b(const uint8_t *x, int inlen,
                                RationalResampleFilter *rf, int downshift,
                                ClipProfile *clip, uint8_t *y, int outlen) {
  (void)inlen;
  const int tapsby2 = rf->length / 2;
  const uint8_t *xext = x;
  xext += rf->start;
  for (int i = 0, p = 0; i < outlen; ++i, p = (p + 1) % rf->p) {
    int sum = 0;
    for (int j = -tapsby2 + 1; j <= tapsby2; ++j) {
      sum += (int)rf->filter[p][j + tapsby2 - 1] * (int)xext[j];
    }
    y[i] = (uint8_t)ROUND_POWER_OF_TWO_SIGNED(sum, downshift);
    if (clip) {
      y[i] = (uint8_t)clip->issigned ? doclip(y[i], -(1 << (clip->bits - 1)),
                                              (1 << (clip->bits - 1)) - 1)
                                     : doclip(y[i], 0, (1 << clip->bits) - 1);
    }
    xext += rf->steps[p];
  }
}

static void extend_border_8b(uint8_t *x, int inlen, int border) {
  for (int i = -border; i < 0; ++i) x[i] = x[0];
  for (int i = 0; i < border; ++i) x[i + inlen] = x[inlen - 1];
}

static void resample_1d_xt_8b(uint8_t *x, int inlen, RationalResampleFilter *rf,
                              int downshift, ClipProfile *clip, uint8_t *y,
                              int outlen) {
  extend_border_8b(x, inlen, rf->length / 2);
  resample_1d_core_8b(x, inlen, rf, downshift, clip, y, outlen);
}

static void resample_1d_xc_8b(const uint8_t *x, int inlen,
                              RationalResampleFilter *rf, int downshift,
                              ClipProfile *clip, uint8_t *y, int outlen,
                              uint8_t *xext) {
  memcpy(xext, x, inlen * sizeof(*x));

  resample_1d_xt_8b(xext, inlen, rf, downshift, clip, y, outlen);
}

static void resample_1d_xt_in8b(uint8_t *x, int inlen,
                                RationalResampleFilter *rf, int downshift,
                                ClipProfile *clip, int16_t *y, int outlen) {
  extend_border_8b(x, inlen, rf->length / 2);
  resample_1d_core_in8b(x, inlen, rf, downshift, clip, y, outlen);
}

static void resample_1d_xc_in8b(const uint8_t *x, int inlen,
                                RationalResampleFilter *rf, int downshift,
                                ClipProfile *clip, int16_t *y, int outlen,
                                uint8_t *xext) {
  memcpy(xext, x, inlen * sizeof(*x));

  resample_1d_xt_in8b(xext, inlen, rf, downshift, clip, y, outlen);
}

static void fill_col_to_arr_in8b(const uint8_t *img, int stride, int len,
                                 int16_t *arr) {
  int i;
  const uint8_t *iptr = img;
  int16_t *aptr = arr;
  for (i = 0; i < len; ++i, iptr += stride) {
    *aptr++ = (int16_t)(*iptr);
  }
}

static void fill_arr_to_col_out8b(uint8_t *img, int stride, int len,
                                  const int16_t *arr) {
  int i;
  uint8_t *iptr = img;
  const int16_t *aptr = arr;
  for (i = 0; i < len; ++i, iptr += stride) {
    *iptr = (uint8_t)*aptr++;
  }
}

void resample_1d_8b(const uint8_t *x, int inlen, RationalResampleFilter *rf,
                    int downshift, ClipProfile *clip, uint8_t *y, int outlen) {
  const int tapsby2 = rf->length / 2;
  uint8_t *xext_ = (uint8_t *)malloc((inlen + rf->length) * sizeof(*x));
  uint8_t *xext = xext_ + tapsby2;

  resample_1d_xc_8b(x, inlen, rf, downshift, clip, y, outlen, xext);

  free(xext_);
}

void resample_2d_8b(const uint8_t *x, int inwidth, int inheight, int instride,
                    RationalResampleFilter *rfh, RationalResampleFilter *rfv,
                    int int_extra_bits, ClipProfile *clip, uint8_t *y,
                    int outwidth, int outheight, int outstride) {
  if (rfv == NULL || is_resampler_noop(rfv)) {
    resample_horz_8b(x, inwidth, inheight, instride, rfh, clip, y, outwidth,
                     outstride);
    return;
  }
  if (rfh == NULL || is_resampler_noop(rfh)) {
    resample_vert_8b(x, inwidth, inheight, instride, rfv, clip, y, outheight,
                     outstride);
    return;
  }
  int16_t *tmpbuf = (int16_t *)malloc(sizeof(int16_t) * outwidth * inheight);
  const int arrsize =
      outheight + ((inheight + rfv->length > inwidth + rfh->length)
                       ? (inheight + rfv->length)
                       : (inwidth + rfh->length));
  int16_t *tmparr_ = (int16_t *)calloc(arrsize, sizeof(int16_t));
  int16_t *tmparrh = tmparr_ + outheight + rfh->length / 2;
  int16_t *tmparrv = tmparr_ + outheight + rfv->length / 2;
  int16_t *tmparro = tmparr_;
  int tmpstride = outwidth;
  const int downshifth = rfh->filter_bits - int_extra_bits;
  const int downshiftv = rfh->filter_bits + int_extra_bits;
  for (int i = 0; i < inheight; ++i) {
    resample_1d_xc_in8b(x + instride * i, inwidth, rfh, downshifth, NULL,
                        tmpbuf + i * tmpstride, outwidth, (uint8_t *)tmparrh);
  }
  for (int i = 0; i < outwidth; ++i) {
    fill_col_to_arr(tmpbuf + i, outwidth, inheight, tmparrv);
    resample_1d_xt(tmparrv, inheight, rfv, downshiftv, clip, tmparro,
                   outheight);
    fill_arr_to_col_out8b(y + i, outstride, outheight, tmparro);
  }
  free(tmpbuf);
  free(tmparr_);
}

void resample_horz_8b(const uint8_t *x, int inwidth, int inheight, int instride,
                      RationalResampleFilter *rfh, ClipProfile *clip,
                      uint8_t *y, int outwidth, int outstride) {
  const int arrsize = inwidth + rfh->length;
  uint8_t *tmparr_ = (uint8_t *)calloc(arrsize, sizeof(*tmparr_));
  uint8_t *tmparrh = tmparr_ + rfh->length / 2;
  for (int i = 0; i < inheight; ++i) {
    resample_1d_xc_8b(x + instride * i, inwidth, rfh, rfh->filter_bits, clip,
                      y + i * outstride, outwidth, tmparrh);
  }
  free(tmparr_);
}

void resample_vert_8b(const uint8_t *x, int inwidth, int inheight, int instride,
                      RationalResampleFilter *rfv, ClipProfile *clip,
                      uint8_t *y, int outheight, int outstride) {
  const int arrsize = outheight + inheight + rfv->length;
  int16_t *tmparr_ = (int16_t *)calloc(arrsize, sizeof(int16_t));
  int16_t *tmparrv = tmparr_ + outheight + rfv->length / 2;
  int16_t *tmparro = tmparr_;
  for (int i = 0; i < inwidth; ++i) {
    fill_col_to_arr_in8b(x + i, instride, inheight, tmparrv);
    resample_1d_xt(tmparrv, inheight, rfv, rfv->filter_bits, clip, tmparro,
                   outheight);
    fill_arr_to_col_out8b(y + i, outstride, outheight, tmparro);
  }
  free(tmparr_);
}

int get_resampled_output_length(int inlen, int p, int q, int force_even) {
  if (!force_even) {
    // round
    return (inlen * p + q / 2) / q;
  }
  int outlen_floor = inlen * p / q;
  // choose floor or ceil depending on which one is even
  if ((outlen_floor % 2) == 1)
    return outlen_floor + 1;
  else
    return outlen_floor;
}
