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

#define CFG_MAX_LEN 256
#define CFG_MAX_WORDS 5

#define DEF_COEFF_PREC_BITS 14

// Usage:
//   lanczos_resample_filter <resampling_config> [<filter_bits>]

static void usage_and_exit(char *prog) {
  printf("Usage:\n");
  printf("  %s\n", prog);
  printf("      <resampling_config>\n");
  printf("      [<filter_bits>]\n");
  printf("  Notes:\n");
  printf("      <resampling_config> is of the form:\n");
  printf("          <p>:<q>:<Lanczos_a>[:<x0>:<ext>] where:\n");
  printf("              <p>/<q> gives the resampling ratio.\n");
  printf("              <Lanczos_a> is Lanczos parameter.\n");
  printf("              <x0> is the optional initial offset\n");
  printf("                                 [default: centered]\n");
  printf("                  If used, it can be a number in (-1, 1),\n");
  printf("                  or 'c' meaning centered.\n");
  printf("                      which is a shortcut for x0 = (q-p)/(2p)\n");
  printf("                  or 'd' meaning co-sited chroma with centered\n");
  printf("                      luma for use only on sub-sampled chroma,\n");
  printf("                      which is a shortcut for x0 = (q-p)/(4p)\n");
  printf("                  The field can be prefixed by 'i' meaning\n");
  printf("                      using the inverse of the number provided,\n");
  printf("              <ext> is the optional extension type:\n");
  printf("                    'r' or 'rep' (Repeat)\n");
  printf("                    's' or 'sym' (Symmetric)\n");
  printf("                    'f' or 'ref' (Reflect/Mirror-whole)\n");
  printf("                    'g' or 'gra' (Grafient preserving)\n");
  printf("                                 [default: 'r']\n");
  printf("          If it is desired to provide different config parameters\n");
  printf("          for luma and chroma, the <Lanczos_a> and <x0> fields\n");
  printf("          could be optionally converted to a pair of\n");
  printf("          comma-separated parameters as follows:\n");
  printf("          <p>:<q>:<Lanczos_al>,<lanczos_ac>[:<x0l>,<x0c>:<ext>]\n");
  printf("              where <Lanczos_al> and <lanczos_ac> are\n");
  printf("                        luma and chroma lanczos parameters\n");
  printf("                    <x0l> and <x0c> are\n");
  printf("                        luma and chroma initial offsets\n");
  printf("      <filter_bits> is prec bits for the filter [default 12].\n");
  printf("      Resampling config of 1:1:1:0 is regarded as a no-op\n");
  exit(1);
}

static int split_words(char *buf, char delim, int nmax, char **words) {
  char *y = buf;
  char *x;
  int n = 0;
  while ((x = strchr(y, delim)) != NULL) {
    *x = 0;
    words[n++] = y;
    if (n == nmax) return n;
    y = x + 1;
  }
  words[n++] = y;
  assert(n > 0 && n <= nmax);
  return n;
}

static int parse_rational_config(char *cfg, int *p, int *q, int *a, double *x0,
                                 EXT_TYPE *ext_type) {
  char cfgbuf[CFG_MAX_LEN];
  strncpy(cfgbuf, cfg, CFG_MAX_LEN - 1);

  char *cfgwords[CFG_MAX_WORDS];
  const int ncfgwords = split_words(cfgbuf, ':', CFG_MAX_WORDS, cfgwords);
  if (ncfgwords < 3) return 0;

  *p = atoi(cfgwords[0]);
  *q = atoi(cfgwords[1]);
  if (*p <= 0 || *q <= 0) return 0;

  char *aparams[2];
  const int naparams = split_words(cfgwords[2], ',', 2, aparams);
  assert(naparams > 0);
  for (int k = 0; k < naparams; ++k) {
    a[k] = atoi(aparams[k]);
    if (a[k] <= 0) return 0;
  }
  if (naparams == 1) a[1] = a[0];

  // Set defaults
  x0[0] = x0[1] = (double)('c');
  *ext_type = EXT_REPEAT;

  if (ncfgwords > 3) {
    char *x0params[2];
    const int nx0params = split_words(cfgwords[3], ',', 2, x0params);
    for (int k = 0; k < nx0params; ++k) {
      if (!strcmp(x0params[k], "c") || !strcmp(x0params[k], "ic"))
        x0[k] = (double)('c');
      else if (!strcmp(x0params[k], "d") || !strcmp(x0params[k], "id"))
        x0[k] = (double)('d');
      else if (x0params[k][0] == 'i')
        x0[k] = get_inverse_x0_numeric(*q, *p, atof(&x0params[k][1]));
      else
        x0[k] = atof(&x0params[k][0]);
    }
    if (nx0params == 1) x0[1] = x0[0];
  }
  if (ncfgwords > 4) {
    if (!strcmp(cfgwords[4], "S") || !strcmp(cfgwords[4], "s") ||
        !strcmp(cfgwords[4], "sym"))
      *ext_type = EXT_SYMMETRIC;
    else if (!strcmp(cfgwords[4], "F") || !strcmp(cfgwords[4], "f") ||
             !strcmp(cfgwords[4], "ref"))
      *ext_type = EXT_REFLECT;
    else if (!strcmp(cfgwords[4], "R") || !strcmp(cfgwords[4], "r") ||
             !strcmp(cfgwords[4], "rep"))
      *ext_type = EXT_REPEAT;
    else if (!strcmp(cfgwords[4], "G") || !strcmp(cfgwords[4], "g") ||
             !strcmp(cfgwords[4], "gra"))
      *ext_type = EXT_GRADIENT;
    else
      return 0;
  }
  return 1;
}

int main(int argc, char *argv[]) {
  int bits = DEF_COEFF_PREC_BITS;
  RationalResampleFilter rf[2];
  if (argc < 2) {
    printf("Not enough arguments\n");
    usage_and_exit(argv[0]);
  }
  if (!strcmp(argv[1], "-help") || !strcmp(argv[1], "-h") ||
      !strcmp(argv[1], "--help") || !strcmp(argv[1], "--h"))
    usage_and_exit(argv[0]);

  int p, q, a[2];
  double x0[2];
  EXT_TYPE ext;
  if (!parse_rational_config(argv[1], &p, &q, a, x0, &ext))
    usage_and_exit(argv[0]);
  if (argc > 2) bits = atoi(argv[2]);

  for (int k = 0; k < 2; ++k) {
    if (!get_resample_filter(p, q, a[k], x0[k], ext, 1, bits, &rf[k])) {
      fprintf(stderr, "Cannot generate filter, exiting!\n");
      exit(1);
    }
    printf("------------------\n");
    if (k == 0)
      printf("LUMA Filter:\n");
    else
      printf("CHROMA Filter:\n");
    printf("------------------\n");
    show_resample_filter(&rf[k]);
  }
}
