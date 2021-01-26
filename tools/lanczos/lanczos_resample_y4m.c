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

#define Y4M_HDR_MAX_LEN 256
#define Y4M_HDR_MAX_WORDS 16
#define CFG_MAX_LEN 256
#define CFG_MAX_WORDS 5

#define DEF_EXTRA_PREC_BITS 2
#define DEF_COEFF_PREC_BITS 14
#define DEF_EXT_TYPE (EXT_REPEAT)
#define DEF_WIN_TYPE (WIN_LANCZOS)

// Usage:
//   lanczos_resample_y4m
//       [<Options>]
//       <y4m_input>
//       <num_frames>
//       <horz_resampling_config>
//       <vert_resampling_config>
//       <y4m_output>
//       [<outwidth>x<outheight>]

static void usage_and_exit(char *prog) {
  printf("Usage:\n");
  printf("  %s\n", prog);
  printf("      [<Options>]\n");
  printf("      <y4m_input>\n");
  printf("      <num_frames>\n");
  printf("      <horz_resampling_config>\n");
  printf("      <vert_resampling_config>\n");
  printf("      <y4m_output>\n");
  printf("      [<outwidth>x<outheight>]\n");
  printf("      \n");
  printf("  Notes:\n");
  printf("      <Options> are optional switches prefixed by '-' as follows:\n");
  printf("          -bit:<n>        - providing bits for filter taps\n");
  printf("                                 [default: 14]\n");
  printf("          -ieb:<n>        - providing intermediate extra bits of\n");
  printf("                            prec between horz and vert filtering\n");
  printf("                            clamped to maximum of (15 - bitdepth)\n");
  printf("                                 [default: 2]\n");
  printf("          -ext:<ext_type> - providing the extension type\n");
  printf("               <ext_type> is one of:\n");
  printf("                    'r' or 'rep' (Repeat)\n");
  printf("                    's' or 'sym' (Symmetric)\n");
  printf("                    'f' or 'ref' (Reflect/Mirror-whole)\n");
  printf("                    'g' or 'gra' (Grafient preserving)\n");
  printf("                                 [default: 'r']\n");
  printf("          -win:<win_type> - providing the windowing function type\n");
  printf("               <win_type> is one of:\n");
  printf("                    'lanczos'     (Repeat)\n");
  printf("                    'lanczos_dil' (Symmetric)\n");
  printf("                    'gaussian'    (Gaussian)\n");
  printf("                    'gengaussian' (Generalized Gaussian)\n");
  printf("                    'cosine'      (Cosine)\n");
  printf("                    'hamming      (Hamming)\n");
  printf("                    'blackman     (Blackman)\n");
  printf("                    'kaiser       (Kaiser)\n");
  printf("                                  [default: 'lanczos']\n");
  printf("      \n");
  printf("      <y4m_input> is the input video in Y4M format\n");
  printf("      <y4m_output> is the output video in Y4M format\n");
  printf("      <num_frames> is number of frames to be processed\n");
  printf("      <horz_resampling_config> and <vert_resampling_config>\n");
  printf("              are of the form:\n");
  printf("          <p>:<q>:<Lanczos_a>[:<x0>] where:\n");
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
  printf("          If it is desired to provide different config parameters\n");
  printf("          for luma and chroma, the <Lanczos_a> and <x0> fields\n");
  printf("          could be optionally converted to a pair of\n");
  printf("          comma-separated parameters as follows:\n");
  printf("          <p>:<q>:<Lanczos_al>,<lanczos_ac>[:<x0l>,<x0c>]\n");
  printf("              where <Lanczos_al> and <lanczos_ac> are\n");
  printf("                        luma and chroma lanczos parameters\n");
  printf("                    <x0l> and <x0c> are\n");
  printf("                        luma and chroma initial offsets\n");
  printf("      <outwidth>x<outheight> is output video dimensions\n");
  printf("                             only needed in case of upsampling\n");
  printf("      Resampling config of 1:1:1:0 horizontally or vertically\n");
  printf("          is regarded as a no-op in that direction\n");
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

static void join_words(char *dest, int len, char **words, int nwords) {
  for (int i = 0; i < nwords; ++i) {
    strncat(dest, " ", len - strlen(dest));
    strncat(dest, words[i], len - strlen(dest));
  }
}

static int parse_rational_config(char *cfg, int *p, int *q, int *a,
                                 double *x0) {
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
  return 1;
}

static int get_options(char *argv[], int *bit, int *ebit, EXT_TYPE *ext_type,
                       WIN_TYPE *win_type) {
  int n = 1;
  while (argv[n][0] == '-') {
    if (!strncmp(argv[n], "-bit:", 5)) {
      *bit = atoi(argv[n] + 5);
    } else if (!strncmp(argv[n], "-ieb:", 5)) {
      *ebit = atoi(argv[n] + 5);
    } else if (!strncmp(argv[n], "-ext:", 5)) {
      *ext_type = EXT_REPEAT;
      const char *word = argv[n] + 5;
      if (!strcmp(word, "S") || !strcmp(word, "s") || !strcmp(word, "sym"))
        *ext_type = EXT_SYMMETRIC;
      else if (!strcmp(word, "F") || !strcmp(word, "f") || !strcmp(word, "ref"))
        *ext_type = EXT_REFLECT;
      else if (!strcmp(word, "R") || !strcmp(word, "r") || !strcmp(word, "rep"))
        *ext_type = EXT_REPEAT;
      else if (!strcmp(word, "G") || !strcmp(word, "g") || !strcmp(word, "gra"))
        *ext_type = EXT_GRADIENT;
      else
        fprintf(stderr, "Unknown extension type, using default\n");
    } else if (!strncmp(argv[n], "-win:", 5)) {
      *win_type = WIN_LANCZOS;
      const char *word = argv[n] + 5;
      if (!strcmp(word, "lanczos"))
        *win_type = WIN_LANCZOS;
      else if (!strcmp(word, "lanczos_dil"))
        *win_type = WIN_LANCZOS_DIL;
      else if (!strcmp(word, "gaussian"))
        *win_type = WIN_GAUSSIAN;
      else if (!strcmp(word, "gengaussian"))
        *win_type = WIN_GENGAUSSIAN;
      else if (!strcmp(word, "cosine"))
        *win_type = WIN_COSINE;
      else if (!strcmp(word, "hamming"))
        *win_type = WIN_HAMMING;
      else if (!strcmp(word, "blackman"))
        *win_type = WIN_BLACKMAN;
      else if (!strcmp(word, "kaiser"))
        *win_type = WIN_KAISER;
      else
        fprintf(stderr, "Unknown window type, using default\n");
    }
    n++;
  }
  return n - 1;
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
  int extra_bits = DEF_EXTRA_PREC_BITS;
  int bits = DEF_COEFF_PREC_BITS;
  EXT_TYPE ext = DEF_EXT_TYPE;
  WIN_TYPE win = DEF_WIN_TYPE;

  RationalResampleFilter horz_rf[2], vert_rf[2];
  int ywidth, yheight;

  if (argc < 6) {
    printf("Not enough arguments\n");
    usage_and_exit(argv[0]);
  }
  if (!strcmp(argv[1], "-help") || !strcmp(argv[1], "-h") ||
      !strcmp(argv[1], "--help") || !strcmp(argv[1], "--h"))
    usage_and_exit(argv[0]);
  const int opts = get_options(argv, &bits, &extra_bits, &ext, &win);
  if (argc < 6 + opts) {
    printf("Not enough arguments\n");
    usage_and_exit(argv[0]);
  }
  char *y4m_input = argv[opts + 1];
  char *y4m_output = argv[opts + 5];

  char hdr[Y4M_HDR_MAX_LEN];
  int nhdrwords;
  char *hdrwords[Y4M_HDR_MAX_WORDS];
  FILE *fin = fopen(y4m_input, "rb");
  if (!fgets(hdr, sizeof(hdr), fin)) {
    printf("Invalid y4m file %s\n", y4m_input);
    usage_and_exit(argv[0]);
  }
  // printf("header = %s\n", hdr);
  nhdrwords = split_words(hdr, ' ', Y4M_HDR_MAX_WORDS, hdrwords);

  int subx, suby;
  int bitdepth;
  if (!parse_info(hdrwords, nhdrwords, &ywidth, &yheight, &bitdepth, &suby,
                  &subx)) {
    printf("Could not parse header from %s\n", y4m_input);
    usage_and_exit(argv[0]);
  }
  const int bytes_per_pel = (bitdepth + 7) / 8;
  int num_frames = atoi(argv[opts + 2]);

  int horz_p, horz_q, vert_p, vert_q;
  int horz_a[2], vert_a[2];
  double horz_x0[2], vert_x0[2];
  if (!parse_rational_config(argv[opts + 3], &horz_p, &horz_q, horz_a,
                             horz_x0)) {
    printf("Could not parse horz resampling config\n");
    usage_and_exit(argv[0]);
  }
  if (!parse_rational_config(argv[opts + 4], &vert_p, &vert_q, vert_a,
                             vert_x0)) {
    printf("Could not parse vert resampling config\n");
    usage_and_exit(argv[0]);
  }

  const int uvwidth = subx ? (ywidth + 1) >> 1 : ywidth;
  const int uvheight = suby ? (yheight + 1) >> 1 : yheight;
  const int ysize = ywidth * yheight;
  const int uvsize = uvwidth * uvheight;

  int rywidth = 0, ryheight = 0;
  if (horz_p > horz_q || vert_p > vert_q) {
    if (argc < 7 + opts) {
      printf("Upsampled output dimensions must be provided\n");
      usage_and_exit(argv[0]);
    }
    // Read output dim if one of the dimensions use upscaling
    if (!parse_dim(argv[opts + 6], &rywidth, &ryheight))
      usage_and_exit(argv[0]);
  }
  if (horz_p <= horz_q)
    rywidth = get_resampled_output_length(ywidth, horz_p, horz_q, subx);
  if (vert_p <= vert_q)
    ryheight = get_resampled_output_length(yheight, vert_p, vert_q, suby);

  printf("InputSize: %dx%d -> OutputSize: %dx%d\n", ywidth, yheight, rywidth,
         ryheight);

  char rhdr[Y4M_HDR_MAX_LEN];
  get_resampled_hdr(rhdr, Y4M_HDR_MAX_LEN, hdrwords, nhdrwords, rywidth,
                    ryheight);
  // printf("Resampled header = %s\n", rhdr);
  FILE *fout = fopen(y4m_output, "wb");
  fwrite(rhdr, strlen(rhdr), 1, fout);

  const int ruvwidth = subx ? (rywidth + 1) >> 1 : rywidth;
  const int ruvheight = suby ? (ryheight + 1) >> 1 : ryheight;
  const int rysize = rywidth * ryheight;
  const int ruvsize = ruvwidth * ruvheight;

  for (int k = 0; k < 2; ++k) {
    if (!get_resample_filter(horz_p, horz_q, horz_a[k], horz_x0[k], ext, win,
                             subx, bits, &horz_rf[k])) {
      fprintf(stderr, "Cannot generate filter, exiting!\n");
      exit(1);
    }
    // show_resample_filter(&horz_rf[k]);
    if (!get_resample_filter(vert_p, vert_q, vert_a[k], vert_x0[k], ext, win,
                             suby, bits, &vert_rf[k])) {
      fprintf(stderr, "Cannot generate filter, exiting!\n");
      exit(1);
    }
    // show_resample_filter(&vert_rf[k]);
  }

  uint8_t *inbuf =
      (uint8_t *)malloc((ysize + 2 * uvsize) * bytes_per_pel * sizeof(uint8_t));
  uint8_t *outbuf = (uint8_t *)malloc((rysize + 2 * ruvsize) * bytes_per_pel *
                                      sizeof(uint8_t));

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
    if (bytes_per_pel == 1) {
      uint8_t *s = inbuf;
      uint8_t *r = outbuf;
      resample_2d_8b(s, ywidth, yheight, ywidth, &horz_rf[0], &vert_rf[0],
                     extra_bits, &clip, r, rywidth, ryheight, rywidth);
      s += ysize;
      r += rysize;
      resample_2d_8b(s, uvwidth, uvheight, uvwidth, &horz_rf[1], &vert_rf[1],
                     extra_bits, &clip, r, ruvwidth, ruvheight, ruvwidth);
      s += uvsize;
      r += ruvsize;
      resample_2d_8b(s, uvwidth, uvheight, uvwidth, &horz_rf[1], &vert_rf[1],
                     extra_bits, &clip, r, ruvwidth, ruvheight, ruvwidth);
    } else {
      int16_t *s = (int16_t *)inbuf;
      int16_t *r = (int16_t *)outbuf;
      resample_2d(s, ywidth, yheight, ywidth, &horz_rf[0], &vert_rf[0],
                  extra_bits, &clip, r, rywidth, ryheight, rywidth);
      s += ysize;
      r += rysize;
      resample_2d(s, uvwidth, uvheight, uvwidth, &horz_rf[1], &vert_rf[1],
                  extra_bits, &clip, r, ruvwidth, ruvheight, ruvwidth);
      s += uvsize;
      r += ruvsize;
      resample_2d(s, uvwidth, uvheight, uvwidth, &horz_rf[1], &vert_rf[1],
                  extra_bits, &clip, r, ruvwidth, ruvheight, ruvwidth);
    }
    fwrite(frametag, 6, 1, fout);
    fwrite(outbuf, (rysize + 2 * ruvsize) * bytes_per_pel, 1, fout);
  }
  fclose(fin);
  fclose(fout);
  free(inbuf);
  free(outbuf);
}
