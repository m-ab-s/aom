/*
 * Copyright (c) 2019, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <stdint.h>

#include "config/aom_config.h"
#include "config/aom_scale_rtcd.h"

#include "aom/aom_codec.h"
#include "aom/aom_encoder.h"

#include "aom_ports/system_state.h"

#include "av1/common/av1_common_int.h"

#include "av1/encoder/encoder.h"
#include "av1/encoder/firstpass.h"
#include "av1/encoder/gop_structure.h"
#include "av1/encoder/subgop.h"

// Set parameters for frames between 'start' and 'end' (excluding both).
static void set_multi_layer_params(const TWO_PASS *twopass,
                                   GF_GROUP *const gf_group, RATE_CONTROL *rc,
                                   FRAME_INFO *frame_info, int start, int end,
                                   int *cur_frame_idx, int *frame_ind,
                                   int layer_depth) {
  const int num_frames_to_process = end - start - 1;
  assert(num_frames_to_process >= 0);
  if (num_frames_to_process == 0) return;

  // Either we are at the last level of the pyramid, or we don't have enough
  // frames between 'l' and 'r' to create one more level.
  if (layer_depth > gf_group->max_layer_depth_allowed ||
      num_frames_to_process < 3) {
    // Leaf nodes.
    while (++start < end) {
      gf_group->update_type[*frame_ind] = LF_UPDATE;
      gf_group->arf_src_offset[*frame_ind] = 0;
      gf_group->cur_frame_idx[*frame_ind] = *cur_frame_idx;
      gf_group->layer_depth[*frame_ind] = MAX_ARF_LAYERS;
      gf_group->arf_boost[*frame_ind] = av1_calc_arf_boost(
          twopass, rc, frame_info, start, end - start, 0, NULL, NULL);
      gf_group->max_layer_depth =
          AOMMAX(gf_group->max_layer_depth, layer_depth);
      ++(*frame_ind);
      ++(*cur_frame_idx);
    }
  } else {
    const int m = (start + end) / 2;

    // Internal ARF.
    gf_group->update_type[*frame_ind] = INTNL_ARF_UPDATE;
    gf_group->arf_src_offset[*frame_ind] = m - start - 1;
    gf_group->cur_frame_idx[*frame_ind] = *cur_frame_idx;
    gf_group->layer_depth[*frame_ind] = layer_depth;

    // Get the boost factor for intermediate ARF frames.
    gf_group->arf_boost[*frame_ind] = av1_calc_arf_boost(
        twopass, rc, frame_info, m, end - m, m - start, NULL, NULL);
    ++(*frame_ind);

    // Frames displayed before this internal ARF.
    set_multi_layer_params(twopass, gf_group, rc, frame_info, start, m,
                           cur_frame_idx, frame_ind, layer_depth + 1);

    // Overlay for internal ARF.
    gf_group->update_type[*frame_ind] = INTNL_OVERLAY_UPDATE;
    gf_group->arf_src_offset[*frame_ind] = 0;
    gf_group->cur_frame_idx[*frame_ind] = *cur_frame_idx;
    gf_group->arf_boost[*frame_ind] = 0;
    gf_group->layer_depth[*frame_ind] = layer_depth;
    ++(*frame_ind);
    ++(*cur_frame_idx);

    // Frames displayed after this internal ARF.
    set_multi_layer_params(twopass, gf_group, rc, frame_info, m, end,
                           cur_frame_idx, frame_ind, layer_depth + 1);
  }
}

static int find_backward_alt_ref(const SubGOPCfg *subgop_cfg, int cur_idx) {
  int cur_disp_idx = subgop_cfg->step[cur_idx].disp_frame_idx;
  int cur_pyr_level = subgop_cfg->step[cur_idx].pyr_level;
  int bwd_alt_ref_disp_idx = 0;

  for (int i = 0; i < cur_idx; ++i) {
    int disp_idx = subgop_cfg->step[i].disp_frame_idx;
    int pyr_level = subgop_cfg->step[i].pyr_level;
    if (pyr_level < cur_pyr_level && disp_idx < cur_disp_idx) {
      if (disp_idx > bwd_alt_ref_disp_idx) bwd_alt_ref_disp_idx = disp_idx;
    }
  }

  return bwd_alt_ref_disp_idx;
}

static int find_forward_alt_ref(const SubGOPCfg *subgop_cfg, int cur_idx) {
  int cur_disp_idx = subgop_cfg->step[cur_idx].disp_frame_idx;
  int cur_pyr_level = subgop_cfg->step[cur_idx].pyr_level;
  int fwd_alt_ref_disp_idx = subgop_cfg->num_frames;

  for (int i = 0; i < cur_idx; ++i) {
    int disp_idx = subgop_cfg->step[i].disp_frame_idx;
    int pyr_level = subgop_cfg->step[i].pyr_level;
    if (pyr_level < cur_pyr_level && disp_idx > cur_disp_idx) {
      if (disp_idx < fwd_alt_ref_disp_idx) fwd_alt_ref_disp_idx = disp_idx;
    }
  }

  return fwd_alt_ref_disp_idx;
}

static void set_multi_layer_params_from_subgop_cfg(
    const TWO_PASS *twopass, GF_GROUP *const gf_group,
    const SubGOPCfg *subgop_cfg, RATE_CONTROL *rc, FRAME_INFO *frame_info,
    int *cur_frame_idx, int *frame_index) {
  int last_shown_frame = 0;
  int min_pyr_level = MAX_ARF_LAYERS;

  for (int idx = 0; idx < subgop_cfg->num_steps; ++idx) {
    const SubGOPStepCfg *frame = &subgop_cfg->step[idx];

    if (frame->pyr_level < min_pyr_level) min_pyr_level = frame->pyr_level;
  }

  for (int idx = 0; idx < subgop_cfg->num_steps; ++idx) {
    const SubGOPStepCfg *frame = &subgop_cfg->step[idx];
    FRAME_TYPE_CODE type = frame->type_code;
    int pyr_level = frame->pyr_level;
    int disp_idx = frame->disp_frame_idx;
    gf_group->cur_frame_idx[*frame_index] = *cur_frame_idx;

    if (type == FRAME_TYPE_OOO_FILTERED ||
        type == FRAME_TYPE_OOO_UNFILTERED) {  // ARF
      gf_group->update_type[*frame_index] =
          pyr_level == min_pyr_level ? ARF_UPDATE : INTNL_ARF_UPDATE;
      if (pyr_level == min_pyr_level) {
        gf_group->arf_index = *frame_index;
        gf_group->arf_boost[*frame_index] = rc->gfu_boost;
      } else {
        int fwd_arf_disp_idx = find_forward_alt_ref(subgop_cfg, idx);
        int bwd_arf_disp_idx = find_backward_alt_ref(subgop_cfg, idx);
        gf_group->arf_boost[*frame_index] = av1_calc_arf_boost(
            twopass, rc, frame_info, disp_idx, fwd_arf_disp_idx - disp_idx,
            disp_idx - bwd_arf_disp_idx, NULL, NULL);
      }
      gf_group->arf_src_offset[*frame_index] = disp_idx - last_shown_frame - 1;
    } else if (type == FRAME_TYPE_INO_VISIBLE) {  // Leaf
      gf_group->update_type[*frame_index] =
          pyr_level == min_pyr_level ? GF_UPDATE : LF_UPDATE;

      int fwd_arf_disp_idx = find_forward_alt_ref(subgop_cfg, idx);
      gf_group->arf_boost[*frame_index] =
          av1_calc_arf_boost(twopass, rc, frame_info, disp_idx,
                             fwd_arf_disp_idx - disp_idx, 0, NULL, NULL);
      gf_group->arf_src_offset[*frame_index] = 0;
      last_shown_frame = disp_idx;

      (*cur_frame_idx)++;
    } else if (type == FRAME_TYPE_INO_REPEAT ||
               type == FRAME_TYPE_INO_SHOWEXISTING) {  // Overlay
      gf_group->update_type[*frame_index] =
          pyr_level == min_pyr_level ? OVERLAY_UPDATE : INTNL_OVERLAY_UPDATE;
      gf_group->arf_boost[*frame_index] = 0;
      gf_group->arf_src_offset[*frame_index] = 0;
      last_shown_frame = disp_idx;
      (*cur_frame_idx)++;
    }
    gf_group->is_filtered[*frame_index] = (type == FRAME_TYPE_OOO_FILTERED);
    gf_group->layer_depth[*frame_index] = pyr_level;
    gf_group->max_layer_depth = AOMMAX(gf_group->max_layer_depth, pyr_level);

    if (idx != (subgop_cfg->num_steps - 1)) (*frame_index)++;
  }
  for (int idx = 1; idx <= *frame_index; idx++) {
    if (gf_group->layer_depth[idx] == gf_group->max_layer_depth)
      gf_group->layer_depth[idx] = MAX_ARF_LAYERS;
  }
}

static int construct_multi_layer_gf_structure(
    AV1_COMP *cpi, TWO_PASS *twopass, GF_GROUP *const gf_group,
    RATE_CONTROL *rc, FRAME_INFO *const frame_info, int gf_interval,
    FRAME_UPDATE_TYPE first_frame_update_type) {
  int frame_index = 0;
  int cur_frame_index = 0;

  // Keyframe / Overlay frame / Golden frame.
  assert(gf_interval >= 1);
  assert(first_frame_update_type == KF_UPDATE ||
         first_frame_update_type == OVERLAY_UPDATE ||
         first_frame_update_type == GF_UPDATE);

  if (first_frame_update_type == KF_UPDATE &&
      cpi->oxcf.kf_cfg.enable_keyframe_filtering > 1) {
    gf_group->has_overlay_for_key_frame = 1;
    gf_group->update_type[frame_index] = KFFLT_UPDATE;
    gf_group->arf_src_offset[frame_index] = 0;
    gf_group->cur_frame_idx[frame_index] = cur_frame_index;
    gf_group->layer_depth[frame_index] = 0;
    gf_group->max_layer_depth = 0;
    ++frame_index;

    gf_group->update_type[frame_index] = KFFLT_OVERLAY_UPDATE;
    gf_group->arf_src_offset[frame_index] = 0;
    gf_group->cur_frame_idx[frame_index] = cur_frame_index;
    gf_group->layer_depth[frame_index] = 0;
    gf_group->max_layer_depth = 0;
    ++frame_index;
    cur_frame_index++;
  } else {
    gf_group->update_type[frame_index] = first_frame_update_type;
    gf_group->arf_src_offset[frame_index] = 0;
    gf_group->cur_frame_idx[frame_index] = cur_frame_index;
    gf_group->layer_depth[frame_index] =
        first_frame_update_type == OVERLAY_UPDATE ? MAX_ARF_LAYERS + 1 : 0;
    gf_group->max_layer_depth = 0;
    if (gf_group->last_step_prev != NULL &&
        first_frame_update_type == GF_UPDATE) {
      gf_group->layer_depth[frame_index] = gf_group->last_step_prev->pyr_level;
      gf_group->max_layer_depth = gf_group->layer_depth[frame_index];
    }
    ++frame_index;
    ++cur_frame_index;
  }

  // Rest of the frames.
  SubGOPSetCfg *subgop_cfg_set = &cpi->subgop_config_set;
  gf_group->subgop_cfg = NULL;
  gf_group->is_user_specified = 0;
  const SubGOPCfg *subgop_cfg;
  subgop_cfg = av1_find_subgop_config(subgop_cfg_set, gf_interval,
                                      rc->frames_to_key - 1 <= gf_interval + 2,
                                      first_frame_update_type == KF_UPDATE);
  if (subgop_cfg) {
    gf_group->subgop_cfg = subgop_cfg;
    gf_group->is_user_specified = 1;
    set_multi_layer_params_from_subgop_cfg(twopass, gf_group, subgop_cfg, rc,
                                           frame_info, &cur_frame_index,
                                           &frame_index);
  } else {
    // ALTREF.
    const int use_altref = gf_group->max_layer_depth_allowed > 0;

    if (use_altref) {
      gf_group->update_type[frame_index] = ARF_UPDATE;
      gf_group->arf_src_offset[frame_index] = gf_interval - 1;
      gf_group->cur_frame_idx[frame_index] = cur_frame_index;
      gf_group->layer_depth[frame_index] = 1;
      gf_group->arf_boost[frame_index] = cpi->rc.gfu_boost;
      gf_group->max_layer_depth = 1;
      gf_group->arf_index = frame_index;
      ++frame_index;
    } else {
      gf_group->arf_index = -1;
    }
    set_multi_layer_params(twopass, gf_group, rc, frame_info, 0, gf_interval,
                           &cur_frame_index, &frame_index, use_altref + 1);
    // The end frame will be Overlay frame for an ARF GOP; otherwise set it to
    // be GF, for consistency, which will be updated in the next GOP.
    gf_group->update_type[frame_index] =
        use_altref ? OVERLAY_UPDATE : GF_UPDATE;
    gf_group->arf_src_offset[frame_index] = 0;
  }

  return frame_index;
}

#define CHECK_GF_PARAMETER 0
#if CHECK_GF_PARAMETER
void check_frame_params(GF_GROUP *const gf_group, int gf_interval) {
  static const char *update_type_strings[FRAME_UPDATE_TYPES] = {
    "KF_UPDATE",        "LF_UPDATE",      "GF_UPDATE",
    "ARF_UPDATE",       "OVERLAY_UPDATE", "INTNL_OVERLAY_UPDATE",
    "INTNL_ARF_UPDATE", "KFFLT_UPDATE",   "KFFLT_OVERLAY_UPDATE",
  };
  FILE *fid = fopen("GF_PARAMS.txt", "a");

  fprintf(fid, "\ngf_interval = {%d}\n", gf_interval);
  for (int i = 0; i < gf_group->size; ++i) {
    fprintf(fid, "#%2d : %s %d %d %d %d\n", i,
            update_type_strings[gf_group->update_type[i]],
            gf_group->arf_src_offset[i], gf_group->arf_pos_in_gf[i],
            gf_group->arf_update_idx[i], gf_group->pyramid_level[i]);
  }

  fprintf(fid, "number of nodes in each level: \n");
  for (int i = 0; i < gf_group->pyramid_height; ++i) {
    fprintf(fid, "lvl %d: %d ", i, gf_group->pyramid_lvl_nodes[i]);
  }
  fprintf(fid, "\n");
  fclose(fid);
}
#endif  // CHECK_GF_PARAMETER

void av1_gop_setup_structure(AV1_COMP *cpi,
                             const EncodeFrameParams *const frame_params) {
  RATE_CONTROL *const rc = &cpi->rc;
  GF_GROUP *const gf_group = &cpi->gf_group;
  TWO_PASS *const twopass = &cpi->twopass;
  FRAME_INFO *const frame_info = &cpi->frame_info;
  const int key_frame = (frame_params->frame_type == KEY_FRAME);
  const FRAME_UPDATE_TYPE first_frame_update_type =
      key_frame ? KF_UPDATE
                : rc->source_alt_ref_active || (rc->baseline_gf_interval == 1)
                      ? OVERLAY_UPDATE
                      : GF_UPDATE;
  gf_group->is_user_specified = 0;
  gf_group->has_overlay_for_key_frame = 0;
  if (cpi->print_per_frame_stats) {
    printf("baseline_gf_interval = %d\n", rc->baseline_gf_interval);
  }
  gf_group->size = construct_multi_layer_gf_structure(
      cpi, twopass, gf_group, rc, frame_info, rc->baseline_gf_interval,
      first_frame_update_type);
  cpi->rc.level1_qp = -1;  // Set to uninitialized.

#if CHECK_GF_PARAMETER
  check_frame_params(gf_group, rc->baseline_gf_interval);
#endif
}
