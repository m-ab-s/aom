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
//   lanczos_resample_yuv
//       <yuv_input>
//       <width>x<height>
//       <pix_format>
//       <num_frames>
//       <horz_p>:<horz_q>:<Lanczos_horz_a>[:<horz_x0>:<horz_ext>]
//       <vert_p>:<vert_q>:<Lanczos_vert_a>[:<vert_x0>:<vert_ext>]
//       <yuv_output>
//       [<outwidth>x<outheight>]

void usage_and_exit(char *prog) {
  printf("Usage:\n");
  printf("  %s\n", prog);
  printf("      <yuv_input>\n");
  printf("      <width>x<height>\n");
  printf("      <pix_format>\n");
  printf("      <num_frames>\n");
  printf("      <horz_p>:<horz_q>:<Lanczos_horz_a>[:<horz_x0>:<horz_ext>]\n");
  printf("      <vert_p>:<vert_q>:<Lanczos_vert_a>[:<vert_x0>:<vert_ext>]\n");
  printf("      <yuv_output>\n");
  printf("      [<outwidth>x<outheight>]\n");
  printf("  Notes:\n");
  printf("      <width>x<height> is input video dimensions.\n");
  printf("      <pix_format> is one of { yuv420p, yuv420p10, yuv420p12,\n");
  printf("                               yuv422p, yuv422p10, yuv422p12,\n");
  printf("                               yuv444p, yuv444p10, yuv444p12 }\n");
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
  printf("      <horz_ext>, <vert_ext> are optional extension types:\n");
  printf("                   'r' or 'rep' (Repeat)\n");
  printf("                   's' or 'sym' (Symmetric)\n");
  printf("                   'f' or 'ref' (Reflect/Mirror-whole)\n");
  printf("                   'g' or 'gra' (Grafient preserving)\n");
  printf("                                [default: 'r']\n");
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

static int parse_pix_format(char *pix_fmt, int *bitdepth, int *subx,
                            int *suby) {
  *bitdepth = 8;
  if (!strncmp(pix_fmt, "yuv420p", 7)) {
    *subx = *suby = 1;
    if (pix_fmt[7] == 0)
      *bitdepth = 8;
    else if (!strncmp(pix_fmt, "yuv420p10", 9))
      *bitdepth = 10;
    else if (!strncmp(pix_fmt, "yuv420p12", 9))
      *bitdepth = 12;
    else
      *bitdepth = atoi(&pix_fmt[7]);
  } else if (!strncmp(pix_fmt, "yuv422p", 7)) {
    *subx = 1;
    *suby = 0;
    if (pix_fmt[7] == 0)
      *bitdepth = 8;
    else if (!strncmp(pix_fmt, "yuv422p10", 9))
      *bitdepth = 10;
    else if (!strncmp(pix_fmt, "yuv422p12", 9))
      *bitdepth = 12;
    else
      *bitdepth = atoi(&pix_fmt[7]);
  } else if (!strncmp(pix_fmt, "yuv444p", 7)) {
    *subx = 0;
    *suby = 0;
    if (pix_fmt[7] == 0)
      *bitdepth = 8;
    else if (!strncmp(pix_fmt, "yuv444p10", 9))
      *bitdepth = 10;
    else if (!strncmp(pix_fmt, "yuv444p12", 9))
      *bitdepth = 12;
    else
      *bitdepth = atoi(&pix_fmt[7]);
  } else {
    return 0;
  }
  return 1;
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
  RationalResampleFilter horz_rf, vert_rf;
  int ywidth, yheight;
  if (argc < 8) {
    printf("Not enough arguments\n");
    usage_and_exit(argv[0]);
  }
  if (!strcmp(argv[1], "-help") || !strcmp(argv[1], "-h") ||
      !strcmp(argv[1], "--help") || !strcmp(argv[1], "--h"))
    usage_and_exit(argv[0]);
  if (!parse_dim(argv[2], &ywidth, &yheight)) usage_and_exit(argv[0]);

  int subx, suby;
  int bitdepth;
  if (!parse_pix_format(argv[3], &bitdepth, &subx, &suby))
    usage_and_exit(argv[0]);

  const int bytes_per_pel = (bitdepth + 7) / 8;
  int num_frames = atoi(argv[4]);

  int horz_p, horz_q, vert_p, vert_q;
  int horz_a, vert_a;
  EXT_TYPE horz_ext = EXT_REPEAT, vert_ext = EXT_REPEAT;
  double horz_x0, vert_x0;
  if (!parse_rational_factor(argv[5], &horz_p, &horz_q, &horz_a, &horz_x0,
                             &horz_ext))
    usage_and_exit(argv[0]);
  if (!parse_rational_factor(argv[6], &vert_p, &vert_q, &vert_a, &vert_x0,
                             &vert_ext))
    usage_and_exit(argv[0]);

  char *yuv_input = argv[1];
  char *yuv_output = argv[7];

  const int uvwidth = subx ? (ywidth + 1) >> 1 : ywidth;
  const int uvheight = suby ? (yheight + 1) >> 1 : yheight;
  const int ysize = ywidth * yheight;
  const int uvsize = uvwidth * uvheight;

  int rywidth = 0, ryheight = 0;
  if (horz_p > horz_q || vert_p > vert_q) {
    if (argc < 9) {
      printf("Upsampled output dimensions must be provided\n");
      usage_and_exit(argv[0]);
    }
    // Read output dim if one of the dimensions use upscaling
    if (!parse_dim(argv[8], &rywidth, &ryheight)) usage_and_exit(argv[0]);
  }
  if (horz_p <= horz_q)
    rywidth = get_resampled_output_length(ywidth, horz_p, horz_q, subx);
  if (vert_p <= vert_q)
    ryheight = get_resampled_output_length(yheight, vert_p, vert_q, suby);

  printf("InputSize: %dx%d -> OutputSize: %dx%d\n", ywidth, yheight, rywidth,
         ryheight);

  const int ruvwidth = subx ? (rywidth + 1) >> 1 : rywidth;
  const int ruvheight = suby ? (ryheight + 1) >> 1 : ryheight;
  const int rysize = rywidth * ryheight;
  const int ruvsize = ruvwidth * ruvheight;

  const int bits = COEFF_PREC_BITS;
  const int int_extra_bits = INT_EXTRA_PREC_BITS;

  get_resample_filter(horz_p, horz_q, horz_a, horz_x0, horz_ext, bits,
                      &horz_rf);
  // show_resample_filter(&horz_rf);
  get_resample_filter(vert_p, vert_q, vert_a, vert_x0, vert_ext, bits,
                      &vert_rf);
  // show_resample_filter(&vert_rf);

  uint8_t *inbuf =
      (uint8_t *)malloc((ysize + 2 * uvsize) * bytes_per_pel * sizeof(uint8_t));
  uint8_t *outbuf = (uint8_t *)malloc((rysize + 2 * ruvsize) * bytes_per_pel *
                                      sizeof(uint8_t));

  FILE *fin = fopen(yuv_input, "rb");
  FILE *fout = fopen(yuv_output, "wb");

  ClipProfile clip = { bitdepth, 0 };

  for (int n = 0; n < num_frames; ++n) {
    if (fread(inbuf, (ysize + 2 * uvsize) * bytes_per_pel, 1, fin) != 1) break;
    if (bytes_per_pel == 1) {
      uint8_t *s = inbuf;
      uint8_t *r = outbuf;
      resample_2d_8b(s, ywidth, yheight, ywidth, &horz_rf, &vert_rf,
                     int_extra_bits, &clip, r, rywidth, ryheight, rywidth);
      s += ysize;
      r += rysize;
      resample_2d_8b(s, uvwidth, uvheight, uvwidth, &horz_rf, &vert_rf,
                     int_extra_bits, &clip, r, ruvwidth, ruvheight, ruvwidth);
      s += uvsize;
      r += ruvsize;
      resample_2d_8b(s, uvwidth, uvheight, uvwidth, &horz_rf, &vert_rf,
                     int_extra_bits, &clip, r, ruvwidth, ruvheight, ruvwidth);
    } else {
      int16_t *s = (int16_t *)inbuf;
      int16_t *r = (int16_t *)outbuf;
      resample_2d(s, ywidth, yheight, ywidth, &horz_rf, &vert_rf,
                  int_extra_bits, &clip, r, rywidth, ryheight, rywidth);
      s += ysize;
      r += rysize;
      resample_2d(s, uvwidth, uvheight, uvwidth, &horz_rf, &vert_rf,
                  int_extra_bits, &clip, r, ruvwidth, ruvheight, ruvwidth);
      s += uvsize;
      r += ruvsize;
      resample_2d(s, uvwidth, uvheight, uvwidth, &horz_rf, &vert_rf,
                  int_extra_bits, &clip, r, ruvwidth, ruvheight, ruvwidth);
    }
    fwrite(outbuf, (rysize + 2 * ruvsize) * bytes_per_pel, 1, fout);
  }
  fclose(fin);
  fclose(fout);
  free(inbuf);
  free(outbuf);
}
