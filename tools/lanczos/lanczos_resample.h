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
#include <math.h>

#define MAX_RATIONAL_FACTOR 16
#define MAX_FILTER_LEN 32

typedef enum { EXT_REPEAT = 'R', EXT_SYMMETRIC = 'S' } EXT_TYPE;

typedef struct {
  int p;
  int q;
  int length;
  EXT_TYPE ext_type;
  int start;
  int steps[MAX_RATIONAL_FACTOR];
  int filter_bits;
  int16_t filter[MAX_RATIONAL_FACTOR][MAX_FILTER_LEN];
  double phases[MAX_RATIONAL_FACTOR];
} RationalResampleFilter;

typedef struct {
  int bits;
  int issigned;
} ClipProfile;

double get_centered_x0(int p, int q);

// In the functions below using x0 as an argument,
// x0 is assumed to be in (-1, 1) or 99 (ascii value of 'c') meaning centered
double get_inverse_x0(int p, int q, double x0);

void get_resample_filter(int p, int q, int a, double x0, EXT_TYPE ext_type,
                         int bits, RationalResampleFilter *rf);
void get_resample_filter_inv(int p, int q, int a, double x0, EXT_TYPE ext_type,
                             int bits, RationalResampleFilter *rf);

// whether the resampler filter is a no-op
int is_resampler_noop(RationalResampleFilter *rf);

// 16-bit versions of high-level resampling functions

// Assume no extension of the input x buffer
void resample_1d(const int16_t *x, int inlen, RationalResampleFilter *rf,
                 int downshift, ClipProfile *clip, int16_t *y, int outlen);

void resample_2d(const int16_t *x, int inwidth, int inheight, int instride,
                 RationalResampleFilter *rfh, RationalResampleFilter *rfv,
                 int int_extra_bits, ClipProfile *clip, int16_t *y,
                 int outwidth, int outheight, int outstride);

void resample_horz(const int16_t *x, int inwidth, int inheight, int instride,
                   RationalResampleFilter *rfh, ClipProfile *clip, int16_t *y,
                   int outwidth, int outstride);

void resample_vert(const int16_t *x, int inwidth, int inheight, int instride,
                   RationalResampleFilter *rfv, ClipProfile *clip, int16_t *y,
                   int outheight, int outstride);

// 8-bit versions of high-level resampling functions

// Assume no extension of the input x buffer
void resample_1d_8b(const uint8_t *x, int inlen, RationalResampleFilter *rf,
                    int downshift, ClipProfile *clip, uint8_t *y, int outlen);

void resample_2d_8b(const uint8_t *x, int inwidth, int inheight, int instride,
                    RationalResampleFilter *rfh, RationalResampleFilter *rfv,
                    int int_extra_bits, ClipProfile *clip, uint8_t *y,
                    int outwidth, int outheight, int outstride);

void resample_horz_8b(const uint8_t *x, int inwidth, int inheight, int instride,
                      RationalResampleFilter *rfh, ClipProfile *clip,
                      uint8_t *y, int outwidth, int outstride);

void resample_vert_8b(const uint8_t *x, int inwidth, int inheight, int instride,
                      RationalResampleFilter *rfv, ClipProfile *clip,
                      uint8_t *y, int outheight, int outstride);

void show_resample_filter(RationalResampleFilter *rf);

int get_resampled_output_length(int inlen, int p, int q, int force_even);
