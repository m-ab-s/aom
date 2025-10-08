/*
 * Copyright (c) 2025, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include "aom/aom_ext_ratectrl.h"
#include "av1/encoder/av1_ext_ratectrl.h"

aom_codec_err_t av1_extrc_init(AOM_EXT_RATECTRL *ext_ratectrl) {
  if (ext_ratectrl == NULL) {
    return AOM_CODEC_INVALID_PARAM;
  }
  av1_zero(*ext_ratectrl);
  return AOM_CODEC_OK;
}

aom_codec_err_t av1_extrc_create(aom_rc_funcs_t funcs,
                                 aom_rc_config_t ratectrl_config,
                                 AOM_EXT_RATECTRL *ext_ratectrl) {
  aom_rc_status_t rc_status;
  aom_rc_firstpass_stats_t *rc_firstpass_stats;
  if (ext_ratectrl == NULL) {
    return AOM_CODEC_INVALID_PARAM;
  }
  av1_extrc_delete(ext_ratectrl);
  ext_ratectrl->funcs = funcs;
  ext_ratectrl->ratectrl_config = ratectrl_config;
  rc_status = ext_ratectrl->funcs.create_model(ext_ratectrl->funcs.priv,
                                               &ext_ratectrl->ratectrl_config,
                                               &ext_ratectrl->model);
  if (rc_status == AOM_RC_ERROR) {
    return AOM_CODEC_ERROR;
  }
  rc_firstpass_stats = &ext_ratectrl->rc_firstpass_stats;
  rc_firstpass_stats->num_frames = ratectrl_config.show_frame_count;
  rc_firstpass_stats->frame_stats =
      aom_malloc(sizeof(*rc_firstpass_stats->frame_stats) *
                 rc_firstpass_stats->num_frames);
  if (rc_firstpass_stats->frame_stats == NULL) {
    return AOM_CODEC_MEM_ERROR;
  }

  ext_ratectrl->ready = 1;
  return AOM_CODEC_OK;
}

aom_codec_err_t av1_extrc_delete(AOM_EXT_RATECTRL *ext_ratectrl) {
  if (ext_ratectrl == NULL) {
    return AOM_CODEC_INVALID_PARAM;
  }
  if (ext_ratectrl->ready) {
    aom_rc_status_t rc_status =
        ext_ratectrl->funcs.delete_model(ext_ratectrl->model);
    if (rc_status == AOM_RC_ERROR) {
      return AOM_CODEC_ERROR;
    }
    aom_free(ext_ratectrl->rc_firstpass_stats.frame_stats);
  }
  return av1_extrc_init(ext_ratectrl);
}
