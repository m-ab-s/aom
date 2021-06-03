/*
 * Copyright (c) 2021, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#ifndef AOM_AV1_ENCODER_THIRDPASS_H_
#define AOM_AV1_ENCODER_THIRDPASS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "av1/encoder/firstpass.h"
#include "av1/common/blockd.h"
#include "common/video_reader.h"

// TODO(bohanli): optimize this number
#define MAX_THIRD_PASS_BUF 100

typedef struct {
  int base_q_idx;
  int is_show_existing;
  int is_show_frame;
  FRAME_TYPE frame_type;
  int order_hint;
} THIRD_PASS_FRAME_INFO;

typedef struct {
  char *file;
  AvxVideoReader *reader;
  aom_codec_ctx_t codec;
  THIRD_PASS_FRAME_INFO frame_info[MAX_THIRD_PASS_BUF];
  int num_gop_info_left;
  int last_end_gop;
  const unsigned char *end_frame;
  const unsigned char *frame;
  size_t frame_size;
  int have_frame;
} THIRD_PASS_DEC_CTX;

int av1_set_gop_third_pass(THIRD_PASS_DEC_CTX *ctx, GF_GROUP *gf_group);
void av1_init_thirdpass_ctx(THIRD_PASS_DEC_CTX **ctx, char *file);
void av1_free_thirdpass_ctx(THIRD_PASS_DEC_CTX *ctx);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_ENCODER_THIRDPASS_H_
