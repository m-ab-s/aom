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

#define COEFF_PREC_BITS 12
#define INT_EXTRA_PREC_BITS 2

// Usage:
//   lanczos_resample_y4m
//       <y4m_input>
//       <num_frames>
//       <horz_p>:<horz_q>:<Lanczos_horz_a>[:<horz_x0>]
//       <vert_p>:<vert_q>:<Lanczos_vert_a>[:<vert_x0>]
//       <y4m_output>
//       [<outwidth>x<outheight>]

static void usage_and_exit(char *prog) {
  printf("Usage:\n");
  printf("  %s\n", prog);
  printf("      <y4m_input>\n");
  printf("      <num_frames>\n");
  printf("      <horz_p>:<horz_q>:<Lanczos_horz_a>[:<horz_x0>]\n");
  printf("      <vert_p>:<vert_q>:<Lanczos_vert_a>[:<vert_x0>]\n");
  printf("      <y4m_output>\n");
  printf("      [<outwidth>x<outheight>]\n");
  printf("  Notes:\n");
  printf("      <num_frames> is number of frames to be processed\n");
  printf("      <horz_p>/<horz_q> gives the horz resampling ratio.\n");
  printf("      <vert_p>/<vert_q> gives the vert resampling ratio.\n");
  printf("      <Lanczos_horz_a>, <Lanczos_vert_a> are Lanczos parameters.\n");
  printf("      <horz_x0>, <vert_x0> are optional initial offsets\n");
  printf("                                        [default centered].\n");
  printf("          If used, they can be a number in (-1, 1),\n");
  printf("                   or a number in (-1, 1) prefixed by 'i' meaning\n");
  printf("                       using the inverse of the number provided,\n");
  printf("                   or 'c' meaning centered\n");
  printf("      <outwidth>x<outheight> is output video dimensions\n");
  printf("                             only needed in case of upsampling\n");
  exit(1);
}

static int parse_dim(char *v, int *width, int *height) {
  char *x = strchr(v, 'x');
  if (x == NULL) x = strchr(v, 'X');
  if (x == NULL) return 0;
  *width = atoi(v);
  *height = atoi(&x[1]);
  if (*width <= 0 || *height <= 0)
    return 0;
  else
    return 1;
}

static int parse_rational_factor(char *factor, int *p, int *q, int *a,
                                 double *x0) {
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
  if (*p <= 0 || *q <= 0 || *a <= 0)
    return 0;
  else
    return 1;
}

static int split_words(char *buf, char **words) {
  char *y = buf;
  char *x;
  int n = 0;
  while ((x = strchr(y, ' ')) != NULL) {
    *x = 0;
    words[n++] = y;
    y = x + 1;
  }
  words[n++] = y;
  return n;
}

static void join_words(char *dest, int len, char **words, int nwords) {
  (void)len;
  for (int i = 0; i < nwords; ++i) {
    strncat(dest, " ", 1);
    strncat(dest, words[i], strlen(words[i]));
  }
}

static void get_resampled_hdr(char *dest, int len, char **words, int nwords,
                              int width, int height) {
  snprintf(dest, len, "YUV4MPEG2 W%d H%d", width, height);
  join_words(dest, len, words + 3, nwords - 3);
}

static int parse_info(char *hdrwords[], int nhdrwords, int *width, int *height,
                      int *bitdepth, int *subx, int *suby) {
  *bitdepth = 8;
  *subx = 1;
  *suby = 1;
  if (nhdrwords < 4) return 0;
  if (strcmp(hdrwords[0], "YUV4MPEG2")) return 0;
  if (sscanf(hdrwords[1], "W%d", width) != 1) return 0;
  if (sscanf(hdrwords[2], "H%d", height) != 1) return 0;
  if (hdrwords[3][0] != 'F') return 0;
  for (int i = 4; i < nhdrwords; ++i) {
    if (!strncmp(hdrwords[i], "C420", 4)) {
      *subx = 1;
      *suby = 1;
      if (hdrwords[i][4] == 'p') *bitdepth = atoi(&hdrwords[i][5]);
    } else if (!strncmp(hdrwords[i], "C422", 4)) {
      *subx = 1;
      *suby = 0;
      if (hdrwords[i][4] == 'p') *bitdepth = atoi(&hdrwords[i][5]);
    } else if (!strncmp(hdrwords[i], "C444", 4)) {
      *subx = 0;
      *suby = 0;
      if (hdrwords[i][4] == 'p') *bitdepth = atoi(&hdrwords[i][5]);
    }
  }
  return 1;
}

int main(int argc, char *argv[]) {
  RationalResampleFilter horz_rf, vert_rf;
  int ywidth, yheight;
  if (!strcmp(argv[1], "-help") || !strcmp(argv[1], "-h") ||
      !strcmp(argv[1], "--help") || !strcmp(argv[1], "--h"))
    usage_and_exit(argv[0]);
  if (argc < 6) {
    printf("Not enough arguments\n");
    usage_and_exit(argv[0]);
  }
  char *y4m_input = argv[1];
  char *y4m_output = argv[5];

  char hdr[256];
  int nhdrwords;
  char *hdrwords[16];
  FILE *fin = fopen(y4m_input, "rb");
  if (!fgets(hdr, sizeof(hdr), fin)) {
    printf("Invalid y4m file %s\n", y4m_input);
    usage_and_exit(argv[0]);
  }
  // printf("header = %s\n", hdr);
  nhdrwords = split_words(hdr, hdrwords);

  int subx, suby;
  int bitdepth;
  if (!parse_info(hdrwords, nhdrwords, &ywidth, &yheight, &bitdepth, &suby,
                  &subx)) {
    printf("Could not parse header from %s\n", y4m_input);
    usage_and_exit(argv[0]);
  }
  const int bytes_per_pel = (bitdepth + 7) / 8;
  int num_frames = atoi(argv[2]);

  int horz_p, horz_q, vert_p, vert_q;
  int horz_a, vert_a;
  double horz_x0, vert_x0;
  if (!parse_rational_factor(argv[3], &horz_p, &horz_q, &horz_a, &horz_x0))
    usage_and_exit(argv[0]);
  if (!parse_rational_factor(argv[4], &vert_p, &vert_q, &vert_a, &vert_x0))
    usage_and_exit(argv[0]);

  const int uvwidth = subx ? (ywidth + 1) >> 1 : ywidth;
  const int uvheight = suby ? (yheight + 1) >> 1 : yheight;
  const int ysize = ywidth * yheight;
  const int uvsize = uvwidth * uvheight;

  int rywidth = 0, ryheight = 0;
  if (horz_p > horz_q || vert_p > vert_q) {
    // Read output dim if one of the dimensions use upscaling
    if (!parse_dim(argv[6], &rywidth, &ryheight)) usage_and_exit(argv[0]);
  }
  if (horz_p <= horz_q)
    rywidth = get_resampled_output_length(ywidth, horz_p, horz_q, subx);
  if (vert_p <= vert_q)
    ryheight = get_resampled_output_length(yheight, vert_p, vert_q, suby);

  printf("InputSize: %dx%d -> OutputSize: %dx%d\n", ywidth, yheight, rywidth,
         ryheight);

  char rhdr[256];
  get_resampled_hdr(rhdr, 256, hdrwords, nhdrwords, rywidth, ryheight);
  // printf("Resampled header = %s\n", rhdr);
  FILE *fout = fopen(y4m_output, "wb");
  fwrite(rhdr, strlen(rhdr), 1, fout);

  const int ruvwidth = subx ? (rywidth + 1) >> 1 : rywidth;
  const int ruvheight = suby ? (ryheight + 1) >> 1 : ryheight;
  const int rysize = rywidth * ryheight;
  const int ruvsize = ruvwidth * ruvheight;

  const int bits = COEFF_PREC_BITS;
  const int horz_downshift = bits - INT_EXTRA_PREC_BITS;
  const int vert_downshift = bits + INT_EXTRA_PREC_BITS;

  get_resample_filter(horz_p, horz_q, horz_a, horz_x0, bits, &horz_rf);
  // show_resample_filter(&horz_rf);
  get_resample_filter(vert_p, vert_q, vert_a, vert_x0, bits, &vert_rf);
  // show_resample_filter(&vert_rf);

  uint8_t *inbuf =
      (uint8_t *)malloc((ysize + 2 * uvsize) * bytes_per_pel * sizeof(uint8_t));
  uint8_t *outbuf = (uint8_t *)malloc((rysize + 2 * ruvsize) * bytes_per_pel *
                                      sizeof(uint8_t));
  int16_t *src = (int16_t *)malloc((ysize + 2 * uvsize) * sizeof(int16_t));
  int16_t *res = (int16_t *)malloc((rysize + 2 * ruvsize) * sizeof(int16_t));

  ClipProfile clip = { bitdepth, 0 };

  char frametag[] = "FRAME\n";
  for (int n = 0; n < num_frames; ++n) {
    char intag[8];
    if (fread(intag, 6, 1, fin) != 1) break;
    intag[6] = 0;
    if (strcmp(intag, frametag)) {
      printf("could not read frame from %s\n", y4m_input);
      break;
    }
    if (fread(inbuf, (ysize + 2 * uvsize) * bytes_per_pel, 1, fin) != 1) break;
    int16_t *s, *r;
    if (bytes_per_pel == 1) {
      uint8_t *d = inbuf;
      s = src;
      for (int i = 0; i < ysize + 2 * uvsize; ++i) *s++ = (int16_t)(*d++);
    } else {
      uint16_t *d = (uint16_t *)inbuf;
      s = src;
      for (int i = 0; i < ysize + 2 * uvsize; ++i) *s++ = (int16_t)(*d++);
    }
    s = src;
    r = res;
    resample_2d(s, ywidth, yheight, ywidth, &horz_rf, &vert_rf, horz_downshift,
                vert_downshift, &clip, r, rywidth, ryheight, rywidth);
    s += ysize;
    r += rysize;
    resample_2d(s, uvwidth, uvheight, uvwidth, &horz_rf, &vert_rf,
                horz_downshift, vert_downshift, &clip, r, ruvwidth, ruvheight,
                ruvwidth);
    s += uvsize;
    r += ruvsize;
    resample_2d(s, uvwidth, uvheight, uvwidth, &horz_rf, &vert_rf,
                horz_downshift, vert_downshift, &clip, r, ruvwidth, ruvheight,
                ruvwidth);
    if (bytes_per_pel == 1) {
      uint8_t *d = outbuf;
      r = res;
      for (int i = 0; i < rysize + 2 * ruvsize; ++i) *d++ = (uint8_t)(*r++);
    } else {
      uint16_t *d = (uint16_t *)outbuf;
      r = res;
      for (int i = 0; i < rysize + 2 * ruvsize; ++i) *d++ = (uint16_t)(*r++);
    }
    fwrite(frametag, 6, 1, fout);
    fwrite(outbuf, (rysize + 2 * ruvsize) * bytes_per_pel, 1, fout);
  }
  fclose(fin);
  fclose(fout);

  free(inbuf);
  free(outbuf);
  free(src);
  free(res);
}
