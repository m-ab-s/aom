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
#include <math.h>

#include "tools/lanczos/lanczos_resample.h"

#define DEF_COEFF_PREC_BITS 12

// Usage:
//   lanczos_resample_filter <p>:<q>:<Lanczos_a>[:<x0>] [<filter_bits>]

static void usage_and_exit(char *prog) {
  printf("Usage:\n");
  printf("  %s\n", prog);
  printf("      <p>:<q>:<Lanczos_a>[:<x0>]\n");
  printf("      [<filter_bits>]\n");
  printf("  Notes:\n");
  printf("      <p>/<q> gives the resampling ratio.\n");
  printf("      <Lanczos_a> is Lanczos parameter.\n");
  printf("      <x0> is optional initial offset [default centered].\n");
  printf("          If used, it can be a number in (-1, 1),\n");
  printf("                   or a number in (-1, 1) prefixed by 'i' meaning\n");
  printf("                       using the inverse of the number provided,\n");
  printf("                   or 'c' meaning centered\n");
  printf("      <filter_bits> is the bits for the filter [default 12].\n");
  exit(1);
}

static int parse_rational_factor(char *factor, int *p, int *q, int *a,
                                 double *x0, EXT_TYPE *ext_type) {
  const char delim = ':';
  *p = atoi(factor);
  char *x = strchr(factor, delim);
  if (x == NULL) return 0;
  *q = atoi(&x[1]);
  char *y = strchr(&x[1], delim);
  *a = atoi(&y[1]);
  char *z = strchr(&y[1], delim);
  if (z == NULL)
    *x0 = (double)('c');
  else if (z[1] == 'c' || (z[1] == 'i' && z[2] == 'c'))
    *x0 = (double)('c');
  else if (z[1] == 'i')
    *x0 = get_inverse_x0(*q, *p, atof(&z[2]));
  else
    *x0 = atof(&z[1]);
  if (*p <= 0 || *q <= 0 || *a <= 0) return 0;
  *ext_type = EXT_REPEAT;
  if (z == NULL) return 1;
  char *e = strchr(&z[1], delim);
  if (e == NULL) return 1;
  if (!strcmp(e + 1, "S") || !strcmp(e + 1, "s") || !strcmp(e + 1, "sym"))
    *ext_type = EXT_SYMMETRIC;
  else if (!strcmp(e + 1, "F") || !strcmp(e + 1, "f") || !strcmp(e + 1, "ref"))
    *ext_type = EXT_REFLECT;
  else if (!strcmp(e + 1, "R") || !strcmp(e + 1, "r") || !strcmp(e + 1, "rep"))
    *ext_type = EXT_REPEAT;
  else if (!strcmp(e + 1, "G") || !strcmp(e + 1, "g") || !strcmp(e + 1, "gra"))
    *ext_type = EXT_GRADIENT;
  else
    return 0;
  return 1;
}

int main(int argc, char *argv[]) {
  int bits = DEF_COEFF_PREC_BITS;
  RationalResampleFilter rf;
  if (argc < 2) {
    printf("Not enough arguments\n");
    usage_and_exit(argv[0]);
  }
  if (!strcmp(argv[1], "-help") || !strcmp(argv[1], "-h") ||
      !strcmp(argv[1], "--help") || !strcmp(argv[1], "--h"))
    usage_and_exit(argv[0]);

  int p, q, a;
  EXT_TYPE ext;
  double x0;
  if (!parse_rational_factor(argv[1], &p, &q, &a, &x0, &ext))
    usage_and_exit(argv[0]);
  if (argc > 2) bits = atoi(argv[2]);

  get_resample_filter(p, q, a, x0, ext, bits, &rf);
  show_resample_filter(&rf);
}
