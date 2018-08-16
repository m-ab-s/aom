/*
 * Copyright (c) 2017, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include "av1/common/optical_flow_ref.h"
#include <float.h>
#include <math.h>
#include <time.h>
#include "./aom_config.h"
#include "./aom_scale_rtcd.h"
#include "aom_mem/aom_mem.h"
#include "aom_scale/aom_scale.h"
#include "av1/common/alloccommon.h"
#include "av1/common/onyxc_int.h"
#include "av1/common/sparse_linear_solver.h"

#if CONFIG_OPFL
// global timer for debug purpose
double timeinit, timesub, timesolve, timeder, timemed, timetotal;
clock_t startt, endt;
static int optical_flow_warp_filter[16][8] = {
  { 0, 0, 0, 128, 0, 0, 0, 0 },      { 0, 2, -6, 126, 8, -2, 0, 0 },
  { 0, 2, -10, 122, 18, -4, 0, 0 },  { 0, 2, -12, 116, 28, -8, 2, 0 },
  { 0, 2, -14, 110, 38, -10, 2, 0 }, { 0, 2, -14, 102, 48, -12, 2, 0 },
  { 0, 2, -16, 94, 58, -12, 2, 0 },  { 0, 2, -14, 84, 66, -12, 2, 0 },
  { 0, 2, -14, 76, 76, -14, 2, 0 },  { 0, 2, -12, 66, 84, -14, 2, 0 },
  { 0, 2, -12, 58, 94, -16, 2, 0 },  { 0, 2, -12, 48, 102, -14, 2, 0 },
  { 0, 2, -10, 38, 110, -14, 2, 0 }, { 0, 2, -8, 28, 116, -12, 2, 0 },
  { 0, 0, -4, 18, 122, -10, 2, 0 },  { 0, 0, -2, 8, 126, -6, 2, 0 }
};

/*
 * Interpolate the whole opfl reference frame.
 * Should only be called by the encoder.
 *
 * Input:
 * cm: the av1_common pointer.
 *     cm->opfl_ref_frame should already been initialized
 *
 * Output:
 * 1: successfully interpolated
 * 0: reference(s) not available
 */
int av1_get_opfl_ref(AV1_COMMON *cm) {
  OPFL_BUFFER_STRUCT *buf_struct = cm->opfl_buf_struct_ptr;
  OPFL_BLK_INFO blk_info;

  av1_opfl_set_buf(cm, buf_struct);

  if (buf_struct->initialized == 1) {
    int width = buf_struct->ref0_buf[0]->y_width;
    int height = buf_struct->ref0_buf[0]->y_height;

#if FRAME_LEVEL_OPFL
    int blkwidth = width;
    int blkheight = height;
#else
    int blkwidth = OPFL_BLOCK_SIZE;
    int blkheight = OPFL_BLOCK_SIZE;
#endif  // FRAME_LEVEL_OPFL

    double numbh = (double)height / (double)blkheight;
    double numbw = (double)width / (double)blkwidth;

    // for every block in the frame
    for (int i = 0; i < numbh; i++) {
      for (int j = 0; j < numbw; j++) {
        if (i * blkheight >= height) continue;
        if (j * blkwidth >= width) continue;

        blk_info.starth = i * blkheight;
        blk_info.startw = j * blkwidth;
        if (blk_info.starth + blkheight > height) {
          blk_info.blk_height = height - blk_info.starth;
        } else {
          blk_info.blk_height = blkheight;
        }
        if (blk_info.startw + blkwidth > width) {
          blk_info.blk_width = width - blk_info.startw;
        } else {
          blk_info.blk_width = blkwidth;
        }

        blk_info.upbound = 0;
        blk_info.lowerbound = 0;
        blk_info.leftbound = 0;
        blk_info.rightbound = 0;

        if (blk_info.starth == 0) {
          blk_info.upbound = 1;
        }
        if (blk_info.starth + blk_info.blk_height >= height) {
          blk_info.lowerbound = 1;
        }
        if (blk_info.startw == 0) {
          blk_info.leftbound = 1;
        }
        if (blk_info.startw + blk_info.blk_width >= width) {
          blk_info.rightbound = 1;
        }
        av1_optical_flow_get_ref(buf_struct, blk_info);
      }
    }

    // av1_opfl_free_buf(buf_struct);
    return 1;
  } else {
    return 0;
  }
}

/*
 * Initialize and set the buffer for optical flow estimation
 */
void av1_opfl_set_buf(AV1_COMMON *cm, OPFL_BUFFER_STRUCT *buf_struct) {
  startt = clock();
  clock_t starti = clock();

  int cur_offset = cm->frame_offset;
  double dst_pos = -1;

  // find the two nearest bi-directional refs
  // these are the refs between which we do optical flow
  int left_idx = -1, left_offset = -1, right_idx = -1, right_offset = -1;
  int left_chosen = NONE_FRAME, right_chosen = NONE_FRAME;
  opfl_get_closest_refs(cm, &left_idx, &left_offset, &left_chosen, &right_idx,
                        &right_offset, &right_chosen);

  // check all other available bi-direcional ref pairs
  // sorted by distance between refs
  int left_idxs[MAX_NUM_REF_PAIR], left_offsets[MAX_NUM_REF_PAIR],
      left_chosens[MAX_NUM_REF_PAIR], right_idxs[MAX_NUM_REF_PAIR],
      right_offsets[MAX_NUM_REF_PAIR], right_chosens[MAX_NUM_REF_PAIR];
  opfl_select_best_ref_pairs(cm, left_idxs, left_offsets, left_chosens,
                             right_idxs, right_offsets, right_chosens, left_idx,
                             right_idx);

  // if no available refs on both sides, don't do optical flow
  // TODO(bohanli): this should only happen for key frame and altref (?)
  //                If we disable it manually, mismatch happens (why?)
  if (left_idx < 0 || right_idx < 0) {
    buf_struct->initialized = 0;
    return;
  }
  // set buffer ptrs to co-located ref, left ref and right ref
  buf_struct->dst_buf = cm->opfl_ref_frame;
  YV12_BUFFER_CONFIG *left = &(cm->buffer_pool->frame_bufs[left_idx].buf);
  YV12_BUFFER_CONFIG *right = &(cm->buffer_pool->frame_bufs[right_idx].buf);
  buf_struct->ref0_buf[0] = left;
  buf_struct->ref1_buf[0] = right;

  // set params for opfl
  buf_struct->opfl_refs[0] = left_chosen;
  buf_struct->opfl_refs[1] = right_chosen;
  dst_pos = ((double)(cur_offset - left_offset)) /
            ((double)(right_offset - left_offset));
  buf_struct->dst_pos = dst_pos;
  buf_struct->left_offset = left_offset;
  buf_struct->cur_offset = cur_offset;
  buf_struct->right_offset = right_offset;

  int width = buf_struct->ref0_buf[0]->y_width,
      height = buf_struct->ref0_buf[0]->y_height;
  int wid, hgt;

  // allocate buffers
  av1_opfl_alloc_buf(cm, buf_struct);

  // calculate the derivatives
  OPFL_BLK_INFO blk_info;
  blk_info.starth = 0;
  blk_info.startw = 0;
  blk_info.blk_height = height;
  blk_info.blk_width = width;
  blk_info.upbound = 1;
  blk_info.lowerbound = 1;
  blk_info.leftbound = 1;
  blk_info.rightbound = 1;

  // set up initial motion filed
  int_mv *left_mv = aom_calloc(cm->mi_cols * cm->mi_rows, sizeof(int_mv));
  int_mv *right_mv = aom_calloc(cm->mi_cols * cm->mi_rows, sizeof(int_mv));

  for (int i = 0; i < cm->mi_rows; i++) {
    for (int j = 0; j < cm->mi_cols; j++) {
      left_mv[i * cm->mi_cols + j].as_int = INVALID_MV;
      right_mv[i * cm->mi_cols + j].as_int = INVALID_MV;
    }
  }

#if OPFL_EXP_INIT
  // temp mv buffers
  int_mv *left_most_mv = aom_calloc(cm->mi_cols * cm->mi_rows, sizeof(int_mv));
  int_mv *right_most_mv = aom_calloc(cm->mi_cols * cm->mi_rows, sizeof(int_mv));
  int_mv *left_temp_mv = aom_calloc(cm->mi_cols * cm->mi_rows, sizeof(int_mv));
  int_mv *right_temp_mv = aom_calloc(cm->mi_cols * cm->mi_rows, sizeof(int_mv));
  int *is_first_valid = aom_calloc(cm->mi_cols * cm->mi_rows, sizeof(int));

  // get the motion field between the two selected refs
  opfl_set_init_motion(cm, buf_struct, left_idx, right_idx, left_offset,
                       right_offset, left_mv, right_mv);
#if OPFL_DERIVE_INIT_MV
  opfl_derive_init_mv(cm, buf_struct, left_idx, right_idx, left_offset,
                      right_offset, left_mv, right_mv);
#endif
#if OPFL_CHECK_INIT_MV
  // copy the mvs from the nearest refs to a temp buffer
  for (int i = 0; i < cm->mi_rows; i++) {
    for (int j = 0; j < cm->mi_cols; j++) {
      left_most_mv[i * cm->mi_cols + j].as_int =
          left_mv[i * cm->mi_cols + j].as_int;
      right_most_mv[i * cm->mi_cols + j].as_int =
          right_mv[i * cm->mi_cols + j].as_int;
      if (left_mv[i * cm->mi_cols + j].as_int != INVALID_MV &&
          right_mv[i * cm->mi_cols + j].as_int != INVALID_MV) {
        is_first_valid[i * cm->mi_cols + j] = 1;
      } else {
        is_first_valid[i * cm->mi_cols + j] = 0;
      }
    }
  }
  // fill the holes of the mvs for comparison later
  opfl_fill_mv(left_most_mv, cm->mi_cols, cm->mi_rows);
  opfl_fill_mv(right_most_mv, cm->mi_cols, cm->mi_rows);

  // now for the other candidate pairs, find there associated motion vectors
  for (int k = 0; k < 3; k++) {
    if (left_idxs[k] < 0 || right_idxs[k] < 0) break;
    opfl_set_init_motion(cm, buf_struct, left_idxs[k], right_idxs[k],
                         left_offsets[k], right_offsets[k], left_temp_mv,
                         right_temp_mv);
    // for each block, see if the new temp mv is better
    // than what we already have
    opfl_update_init_motion(cm, buf_struct, left_most_mv, right_most_mv,
                            left_offset, right_offset, left_temp_mv,
                            right_temp_mv, left_offsets[k], right_offsets[k],
                            is_first_valid, left_mv, right_mv);
  }
#endif

  aom_free(left_most_mv);
  aom_free(right_most_mv);
  aom_free(left_temp_mv);
  aom_free(right_temp_mv);
  aom_free(is_first_valid);
#else
  // use all available initialization of motion field
  TPL_MV_REF *tpl_mvs_base = cm->cur_frame->tpl_mvs;
  for (int i = 0; i < cm->mi_rows; i++) {
    for (int j = 0; j < cm->mi_cols; j++) {
      for (int k = 0; k < MFMV_STACK_SIZE; k++) {
        if (tpl_mvs_base[i * cm->mi_stride + j]
                .mfmv[left_chosen - LAST_FRAME][k]
                .as_int != INVALID_MV) {
          left_mv[i * cm->mi_cols + j].as_int =
              tpl_mvs_base[i * cm->mi_stride + j]
                  .mfmv[left_chosen - LAST_FRAME][k]
                  .as_int;
          break;
        }
      }
      for (int k = 0; k < MFMV_STACK_SIZE; k++) {
        if (tpl_mvs_base[i * cm->mi_stride + j]
                .mfmv[right_chosen - LAST_FRAME][k]
                .as_int != INVALID_MV) {
          right_mv[i * cm->mi_cols + j].as_int =
              tpl_mvs_base[i * cm->mi_stride + j]
                  .mfmv[right_chosen - LAST_FRAME][k]
                  .as_int;
          break;
        }
      }
    }
  }
#endif

  // copy the initialized motions to each level
  for (int l = 0; l < MAX_OPFL_LEVEL; l++) {
    wid = width >> l;
    hgt = height >> l;

    create_motion_field(left_mv, right_mv, buf_struct->init_mv_buf[l],
#if OPFL_INIT_WT
                        buf_struct->init_mv_wts[l],
#endif
                        width, height, wid, hgt, wid + 2 * AVG_MF_BORDER,
                        dst_pos);

    // fill in possible "holes" in the initialization
    fill_create_motion_field(left_mv, right_mv, buf_struct->init_mv_buf[l],
                             width, height, wid, hgt, wid + 2 * AVG_MF_BORDER);

#if DUMP_OPFL
    if (l == 0) {
      int mvstr = wid + 2 * AVG_MF_BORDER;
      DB_MV *mv_start =
          buf_struct->init_mv_buf[l] + AVG_MF_BORDER * mvstr + AVG_MF_BORDER;

      warp_optical_flow_fwd(buf_struct->ref0_buf[l], buf_struct->ref1_buf[l],
                            mv_start, mvstr, buf_struct->ref0_warped_buf[l],
                            dst_pos, l, !USE_BLK_DERIVATIVE, blk_info);

      warp_optical_flow_back(buf_struct->ref1_buf[l], buf_struct->ref0_buf[l],
                             mv_start, mvstr, buf_struct->ref1_warped_buf[l],
                             1 - dst_pos, l, !USE_BLK_DERIVATIVE, blk_info);
      // int srcstr = buf_struct->ref0_warped_buf[0]->y_stride;
      // for (int i = 0; i < height; i++) {
      //   for (int j = 0; j < width; j++) {
      //     if (left_mv[i/4*wid/4+j/4].as_int == INVALID_MV) {
      //       buf_struct->ref0_warped_buf[0]->y_buffer[i*srcstr+j] = 0;
      //       buf_struct->ref1_warped_buf[0]->y_buffer[i*srcstr+j] = 0;
      //     }
      //   }
      // }

      write_image_opfl(buf_struct->ref0_buf[l], "init_dump.yuv");
      write_image_opfl(buf_struct->ref0_warped_buf[l], "init_dump.yuv");
      write_image_opfl(buf_struct->ref1_warped_buf[l], "init_dump.yuv");
      write_image_opfl(buf_struct->ref1_buf[l], "init_dump.yuv");
    }
#endif
  }

  aom_free(left_mv);
  aom_free(right_mv);

  // warped references according to initialization for future use
  blk_info.starth = 0;
  blk_info.startw = 0;
  for (int l = 0; l < MAX_OPFL_LEVEL; l++) {
    if (USE_BLK_DERIVATIVE && l != 0) {
      continue;
    }
    wid = width >> l;
    hgt = height >> l;
    blk_info.blk_height = hgt;
    blk_info.blk_width = wid;

    int mvstr = wid + 2 * AVG_MF_BORDER;
    DB_MV *mv_start =
        buf_struct->init_mv_buf[l] + AVG_MF_BORDER * mvstr + AVG_MF_BORDER;

    if (l == 0 || USE_BLK_DERIVATIVE) {
      warp_optical_flow_fwd(buf_struct->ref0_buf[l], buf_struct->ref1_buf[l],
                            mv_start, mvstr, buf_struct->ref0_warped_buf[l],
                            dst_pos, l, !USE_BLK_DERIVATIVE, blk_info);
    } else {
      warp_optical_flow_fwd_bilinear(buf_struct->ref0_buf[l],
                                     buf_struct->ref1_buf[l], mv_start, mvstr,
                                     buf_struct->ref0_warped_buf[l], dst_pos, l,
                                     !USE_BLK_DERIVATIVE, blk_info);
    }
    if (l == 0 || USE_BLK_DERIVATIVE) {
      warp_optical_flow_back(buf_struct->ref1_buf[l], buf_struct->ref0_buf[l],
                             mv_start, mvstr, buf_struct->ref1_warped_buf[l],
                             1 - dst_pos, l, !USE_BLK_DERIVATIVE, blk_info);
    } else {
      warp_optical_flow_back_bilinear(
          buf_struct->ref1_buf[l], buf_struct->ref0_buf[l], mv_start, mvstr,
          buf_struct->ref1_warped_buf[l], 1 - dst_pos, l, !USE_BLK_DERIVATIVE,
          blk_info);
    }
    aom_yv12_extend_frame_borders_c(buf_struct->ref0_warped_buf[l]);
    aom_yv12_extend_frame_borders_c(buf_struct->ref1_warped_buf[l]);
  }
  buf_struct->initialized = 1;

  clock_t endi = clock();
  timeinit += (double)(endi - starti) / CLOCKS_PER_SEC;
}

/*
 * Use optical flow method to interpolate a reference frame.
 *
 * Input:
 * buf_struct: containing necessary buffers
 * blk_info: information on the current block (size, location, etc.)
 */
void av1_optical_flow_get_ref(OPFL_BUFFER_STRUCT *buf_struct,
                              OPFL_BLK_INFO blk_info) {
  int width, height, start_h, start_w, wid, hgt, sh, sw;
  width = blk_info.blk_width;
  height = blk_info.blk_height;
  start_h = blk_info.starth;
  start_w = blk_info.startw;

  // temporary buffers for MF median filtering
  double mv_r[25], mv_c[25], left[25], right[25];

  // initialize buffers
  DB_MV **mf_last = buf_struct->mf_last;
  DB_MV **mf_new = buf_struct->mf_new;
  DB_MV **mf_med = buf_struct->mf_med;
  int l = MAX_OPFL_LEVEL - 1;
  wid = width >> l;
  hgt = height >> l;
  int imvstr = buf_struct->ref0_buf[0]->y_width;
  imvstr = (imvstr >> l) + 2 * AVG_MF_BORDER;
  sh = start_h >> l;
  sw = start_w >> l;
  int str = wid + 2 * AVG_MF_BORDER;

  for (int i = 0; i < hgt; i++) {
    for (int j = 0; j < wid; j++) {
      mf_last[l][(i + AVG_MF_BORDER) * str + j + AVG_MF_BORDER] =
          buf_struct->init_mv_buf[l][(i + sh + AVG_MF_BORDER) * imvstr + j +
                                     sw + AVG_MF_BORDER];
    }
  }

  // estimate optical flow at each level
  for (l = MAX_OPFL_LEVEL - 1; l >= 0; l--) {
    wid = width >> l;
    hgt = height >> l;
    sh = start_h >> l;
    sw = start_w >> l;
    int mvstr = wid + 2 * AVG_MF_BORDER;
    imvstr = buf_struct->ref0_buf[0]->y_width;
    imvstr = (imvstr >> l) + 2 * AVG_MF_BORDER;
    // use optical flow to refine our motion field
#if USE_BLK_DERIVATIVE
    refine_motion_field(buf_struct, mf_last[l], mf_new[l], l,
                        buf_struct->dst_pos, 0, blk_info);
#else
    refine_motion_field(buf_struct, mf_last[l], mf_new[l], l,
                        buf_struct->dst_pos, 1, blk_info);
#endif
    DB_MV *mf_start_new = mf_new[l] + AVG_MF_BORDER * mvstr + AVG_MF_BORDER;
    DB_MV *mf_start_med = mf_med[l] + AVG_MF_BORDER * mvstr + AVG_MF_BORDER;

    clock_t startm, endm;
    startm = clock();
    DB_MV *initmv = buf_struct->init_mv_buf[l] + (sh + AVG_MF_BORDER) * imvstr +
                    sw + AVG_MF_BORDER;
    for (int i = 0; i < hgt; i++) {
      for (int j = 0; j < wid; j++) {
        if (USE_MEDIAN_FILTER) {
          int c = 0;
          for (int h = -2; h < 3; h++) {
            for (int w = -2; w < 3; w++) {
              if (i + h < 0 || i + h >= hgt || j + w < 0 || j + w >= wid) {
                // mv_r[c] = initmv[(i + h) * imvstr + j + w].row;
                // mv_c[c] = initmv[(i + h) * imvstr + j + w].col;
                continue;
              } else {
                mv_r[c] = mf_start_new[(i + h) * mvstr + j + w].row;
                mv_c[c] = mf_start_new[(i + h) * mvstr + j + w].col;
              }
              c++;
            }
          }
          mf_start_med[i * mvstr + j].row =
              iter_median_double(mv_r, left, right, c, c / 2);
          mf_start_med[i * mvstr + j].col =
              iter_median_double(mv_c, left, right, c, c / 2);
        } else {
          mf_start_med[i * mvstr + j].row = mf_start_new[i * mvstr + j].row;
          mf_start_med[i * mvstr + j].col = mf_start_new[i * mvstr + j].col;
        }
      }
    }
    endm = clock();
    timemed += (double)(endm - startm) / CLOCKS_PER_SEC;

    if (l != 0) {
      // upscale mv to the next lower level
      int mvstr_next = wid * 2 + 2 * AVG_MF_BORDER;
      DB_MV *mf_start_next =
          mf_last[l - 1] + AVG_MF_BORDER * mvstr_next + AVG_MF_BORDER;
      upscale_mv_by_2(mf_start_med, wid, hgt, mvstr, mf_start_next, mvstr_next);
    } else {
      pad_motion_field_border(mf_start_med, wid, hgt, mvstr);
    }
  }

  // interpolate to get our reference frame
  clock_t start, end;
  start = clock();
  interp_optical_flow(buf_struct->ref0_buf[0], buf_struct->ref1_buf[0],
                      mf_med[0], buf_struct->dst_buf, buf_struct->dst_pos,
                      blk_info);
  end = clock();
  timesub += (double)(end - start) / CLOCKS_PER_SEC;

  // pad border if at border of frame
  int fstr = buf_struct->dst_buf->y_stride;
  int fstruv = buf_struct->dst_buf->uv_stride;
  int border = buf_struct->dst_buf->border;
  int fheight = buf_struct->dst_buf->y_crop_height;
  int fwidth = buf_struct->dst_buf->y_crop_width;
  uint8_t *ydst = buf_struct->dst_buf->y_buffer;
  uint8_t *udst = buf_struct->dst_buf->u_buffer;
  uint8_t *vdst = buf_struct->dst_buf->v_buffer;

  int topb = 0, bottomb = 0, leftb = 0, rightb = 0;
  if (blk_info.upbound) {
    topb = border;
  }
  if (blk_info.lowerbound) {
    bottomb = border + buf_struct->dst_buf->y_height - fheight;
  }
  if (blk_info.leftbound) {
    leftb = border;
  }
  if (blk_info.rightbound) {
    rightb = border + buf_struct->dst_buf->y_width - fwidth;
  }

  extend_plane_opfl(ydst, fstr, fwidth, fheight, topb, leftb, bottomb, rightb);
  extend_plane_opfl(udst, fstruv, fwidth / 2, fheight / 2, topb / 2, leftb / 2,
                    bottomb / 2, rightb / 2);
  extend_plane_opfl(vdst, fstruv, fwidth / 2, fheight / 2, topb / 2, leftb / 2,
                    bottomb / 2, rightb / 2);
  return;
}

/*
 * Allocate buffers in opfl
 */
void av1_opfl_alloc_buf(AV1_COMMON *cm, OPFL_BUFFER_STRUCT *buf_struct) {
  // allocate yuv buffers for opfl use
  int width = buf_struct->ref0_buf[0]->y_width,
      height = buf_struct->ref0_buf[0]->y_height;
  int wid, hgt;
  for (int l = 0; l < MAX_OPFL_LEVEL; l++) {
    wid = width >> l;
    hgt = height >> l;
    if (l == 0) {
      buf_struct->ref0_warped_buf[l] =
          aom_calloc(1, sizeof(YV12_BUFFER_CONFIG));
      buf_struct->ref1_warped_buf[l] =
          aom_calloc(1, sizeof(YV12_BUFFER_CONFIG));
      aom_alloc_frame_buffer(buf_struct->ref0_warped_buf[l], wid, hgt, 1, 1, 0,
                             AOM_BORDER_IN_PIXELS, 0);
      aom_alloc_frame_buffer(buf_struct->ref1_warped_buf[l], wid, hgt, 1, 1, 0,
                             AOM_BORDER_IN_PIXELS, 0);
    }
#if !USE_BLK_DERIVATIVE
    if (l != 0) {
      buf_struct->ref0_buf[l] = aom_calloc(1, sizeof(YV12_BUFFER_CONFIG));
      buf_struct->ref1_buf[l] = aom_calloc(1, sizeof(YV12_BUFFER_CONFIG));
      aom_alloc_frame_buffer(buf_struct->ref0_buf[l], wid, hgt, 1, 1, 0,
                             AOM_BORDER_IN_PIXELS, 1);
      aom_alloc_frame_buffer(buf_struct->ref1_buf[l], wid, hgt, 1, 1, 0,
                             AOM_BORDER_IN_PIXELS, 1);
      buf_struct->ref0_warped_buf[l] =
          aom_calloc(1, sizeof(YV12_BUFFER_CONFIG));
      buf_struct->ref1_warped_buf[l] =
          aom_calloc(1, sizeof(YV12_BUFFER_CONFIG));
      aom_alloc_frame_buffer(buf_struct->ref0_warped_buf[l], wid, hgt, 1, 1, 0,
                             AOM_BORDER_IN_PIXELS, 1);
      aom_alloc_frame_buffer(buf_struct->ref1_warped_buf[l], wid, hgt, 1, 1, 0,
                             AOM_BORDER_IN_PIXELS, 1);
    }
#endif
  }
#if !USE_BLK_DERIVATIVE
  // scale the buffers for pyramid structure
  // TODO(bohan): find out the necessary space for temp_buffer
  uint8_t *temp_buffer = aom_calloc(width * 8, sizeof(uint8_t));
  for (int l = 1; l < MAX_OPFL_LEVEL; l++) {
    aom_scale_frame(buf_struct->ref0_buf[l - 1], buf_struct->ref0_buf[l],
                    temp_buffer, 8, 2, 1, 2, 1, 0);
    aom_scale_frame(buf_struct->ref1_buf[l - 1], buf_struct->ref1_buf[l],
                    temp_buffer, 8, 2, 1, 2, 1, 0);
    aom_yv12_extend_frame_borders_c(buf_struct->ref0_buf[l]);
    aom_yv12_extend_frame_borders_c(buf_struct->ref1_buf[l]);
  }
  aom_free(temp_buffer);
#endif

  // allocate initial motion field buffers
  for (int l = 0; l < MAX_OPFL_LEVEL; l++) {
    wid = width >> l;
    hgt = height >> l;
    buf_struct->init_mv_buf[l] = aom_calloc(
        (wid + 2 * AVG_MF_BORDER) * (hgt + 2 * AVG_MF_BORDER), sizeof(DB_MV));
#if OPFL_INIT_WT
    buf_struct->init_mv_wts[l] = aom_calloc(
        (wid + 2 * AVG_MF_BORDER) * (hgt + 2 * AVG_MF_BORDER), sizeof(double));
#endif
  }

  // allocate motion field buffer for each pyramid level
  int blkwid, blkhgt;
#if FRAME_LEVEL_OPFL
  blkwid = width;
  blkhgt = height;
#else
  blkwid = OPFL_BLOCK_SIZE;
  blkhgt = OPFL_BLOCK_SIZE;
#endif
  for (int l = 0; l < MAX_OPFL_LEVEL; l++) {
    wid = blkwid >> l;
    hgt = blkhgt >> l;
    buf_struct->mf_last[l] = aom_calloc(
        (wid + 2 * AVG_MF_BORDER) * (hgt + 2 * AVG_MF_BORDER), sizeof(DB_MV));
    buf_struct->mf_new[l] = aom_calloc(
        (wid + 2 * AVG_MF_BORDER) * (hgt + 2 * AVG_MF_BORDER), sizeof(DB_MV));
    buf_struct->mf_med[l] = aom_calloc(
        (wid + 2 * AVG_MF_BORDER) * (hgt + 2 * AVG_MF_BORDER), sizeof(DB_MV));

    // allocate buffers
    if (l != 0 && USE_BLK_DERIVATIVE) continue;
    buf_struct->buffer0[l] = aom_calloc(1, sizeof(YV12_BUFFER_CONFIG));
    buf_struct->buffer1[l] = aom_calloc(1, sizeof(YV12_BUFFER_CONFIG));
    aom_alloc_frame_buffer(buf_struct->buffer0[l], wid, hgt, 1, 1, 0,
                           AOM_BORDER_IN_PIXELS, 0);
    aom_alloc_frame_buffer(buf_struct->buffer1[l], wid, hgt, 1, 1, 0,
                           AOM_BORDER_IN_PIXELS, 0);
  }

  // allocate done flag buffer for each blockc
  int wblk, hblk;
  wblk = (width + blkwid - 1) / blkwid;
  hblk = (height + blkhgt - 1) / blkhgt;
  buf_struct->done_flag = aom_calloc(wblk * hblk, sizeof(int));

  // allocate derivative buffers
  // temp derivative buffer for each iter
  buf_struct->Ex = aom_calloc(width * height, sizeof(double));
  buf_struct->Ey = aom_calloc(width * height, sizeof(double));
  buf_struct->Et = aom_calloc(width * height, sizeof(double));
}

/*
 * Free the allocated buffers in buf_struct.
 */
void av1_opfl_free_buf(OPFL_BUFFER_STRUCT *buf_struct) {
  if (buf_struct->initialized != 1) return;

  for (int l = 0; l < MAX_OPFL_LEVEL; l++) {
    if (l == 0) {
      aom_free_frame_buffer(buf_struct->ref0_warped_buf[l]);
      aom_free_frame_buffer(buf_struct->ref1_warped_buf[l]);
      aom_free(buf_struct->ref0_warped_buf[l]);
      aom_free(buf_struct->ref1_warped_buf[l]);
    }
#if !USE_BLK_DERIVATIVE
    if (l != 0) {
      aom_free_frame_buffer(buf_struct->ref0_buf[l]);
      aom_free_frame_buffer(buf_struct->ref1_buf[l]);
      aom_free(buf_struct->ref0_buf[l]);
      aom_free(buf_struct->ref1_buf[l]);
      aom_free_frame_buffer(buf_struct->ref0_warped_buf[l]);
      aom_free_frame_buffer(buf_struct->ref1_warped_buf[l]);
      aom_free(buf_struct->ref0_warped_buf[l]);
      aom_free(buf_struct->ref1_warped_buf[l]);
    }
#endif
  }
  for (int l = 0; l < MAX_OPFL_LEVEL; l++) {
    aom_free(buf_struct->init_mv_buf[l]);
#if OPFL_INIT_WT
    aom_free(buf_struct->init_mv_wts[l]);
#endif
  }
  aom_free(buf_struct->Ex);
  aom_free(buf_struct->Ey);
  aom_free(buf_struct->Et);
  for (int l = 0; l < MAX_OPFL_LEVEL; l++) {
    aom_free(buf_struct->mf_last[l]);
    aom_free(buf_struct->mf_new[l]);
    aom_free(buf_struct->mf_med[l]);

    if (l != 0 && USE_BLK_DERIVATIVE) continue;
    aom_free_frame_buffer(buf_struct->buffer0[l]);
    aom_free_frame_buffer(buf_struct->buffer1[l]);
    aom_free(buf_struct->buffer0[l]);
    aom_free(buf_struct->buffer1[l]);
  }
  aom_free(buf_struct->done_flag);
  endt = clock();
  timetotal += (double)(endt - startt) / CLOCKS_PER_SEC;
#if OPFL_OUTPUT_TIME
  // TODO(bohan): output time usage for debug for now.
  printf(
      "\ninit time: %.4f, der time: %.4f, sub time: %.4f, median time: %.4f, "
      "solve time: %.4f, totaltime: %.4f\n",
      timeinit, timeder, timesub, timemed, timesolve, timetotal);
  fflush(stdout);
#endif
#if DUMP_OPFL
  // TODO(bohan): dump the frames for now for debug
  char filename[20] = "of_dump.yuv";
  write_image_opfl(buf_struct->ref0_buf[0], filename);
  write_image_opfl(buf_struct->dst_buf, filename);
  write_image_opfl(buf_struct->ref1_buf[0], filename);
  char idxfilename[20] = "of_data.txt";
  FILE *f_idx = fopen(idxfilename, "a");
  fprintf(f_idx, "%d %d %d\n", buf_struct->left_offset, buf_struct->cur_offset,
          buf_struct->right_offset);
  fclose(f_idx);
#endif
}

/*
 * get how many motion vector initializations exist between two refs
 */
int get_num_MV_between_refs(AV1_COMMON *cm, int left_idx, int left_offset,
                            int right_idx, int right_offset) {
  int totalNum = 0;
  // process left ref
  MV_REF *mv_ref_base = cm->buffer_pool->frame_bufs[left_idx].mvs;
  int lst_frame_idx = cm->buffer_pool->frame_bufs[left_idx].lst_frame_offset;
  int alt_frame_idx = cm->buffer_pool->frame_bufs[left_idx].alt_frame_offset;
  int gld_frame_idx = cm->buffer_pool->frame_bufs[left_idx].gld_frame_offset;
#if CONFIG_EXT_REFS
  int lst2_frame_idx = cm->buffer_pool->frame_bufs[left_idx].lst2_frame_offset;
  int lst3_frame_idx = cm->buffer_pool->frame_bufs[left_idx].lst3_frame_offset;
  int bwd_frame_idx = cm->buffer_pool->frame_bufs[left_idx].bwd_frame_offset;
#endif
  for (int i = 0; i < cm->mi_rows; i++) {
    for (int j = 0; j < cm->mi_cols; j++) {
      MV_REF *mv_ref = &mv_ref_base[i * cm->mi_cols + j];
      MV_REFERENCE_FRAME ref_frame[2] = { mv_ref->ref_frame[0],
                                          mv_ref->ref_frame[1] };
      if (ref_frame[0] == OPFL_FRAME && ref_frame[1] == NONE_FRAME) {
        ref_frame[0] = mv_ref->opfl_ref_frame[0];
        ref_frame[1] = mv_ref->opfl_ref_frame[1];
      }

      for (int r = 0; r < 2; r++) {
        int ref_offset;
        switch (ref_frame[r]) {
          // case LAST_FRAME: ref_offset = lst_frame_idx; break;
          case ALTREF_FRAME:
            ref_offset = alt_frame_idx;
            break;
            // case GOLDEN_FRAME: ref_offset = gld_frame_idx; break;
#if CONFIG_EXT_REFS
          // case LAST2_FRAME: ref_offset = lst2_frame_idx; break;
          // case LAST3_FRAME: ref_offset = lst3_frame_idx; break;
          case BWDREF_FRAME: ref_offset = bwd_frame_idx; break;
#endif
          default: ref_offset = -2;
        }
        if (ref_offset == right_offset && ref_offset >= 0) {
          totalNum++;
        }
      }
    }
  }
  // process right ref
  mv_ref_base = cm->buffer_pool->frame_bufs[right_idx].mvs;
  lst_frame_idx = cm->buffer_pool->frame_bufs[right_idx].lst_frame_offset;
  alt_frame_idx = cm->buffer_pool->frame_bufs[right_idx].alt_frame_offset;
  gld_frame_idx = cm->buffer_pool->frame_bufs[right_idx].gld_frame_offset;
#if CONFIG_EXT_REFS
  lst2_frame_idx = cm->buffer_pool->frame_bufs[right_idx].lst2_frame_offset;
  lst3_frame_idx = cm->buffer_pool->frame_bufs[right_idx].lst3_frame_offset;
  bwd_frame_idx = cm->buffer_pool->frame_bufs[right_idx].bwd_frame_offset;
#endif
  for (int i = 0; i < cm->mi_rows; i++) {
    for (int j = 0; j < cm->mi_cols; j++) {
      MV_REF *mv_ref = &mv_ref_base[i * cm->mi_cols + j];
      MV_REFERENCE_FRAME ref_frame[2] = { mv_ref->ref_frame[0],
                                          mv_ref->ref_frame[1] };

      if (ref_frame[0] == OPFL_FRAME && ref_frame[1] == NONE_FRAME) {
        ref_frame[0] = mv_ref->opfl_ref_frame[0];
        ref_frame[1] = mv_ref->opfl_ref_frame[1];
      }

      for (int r = 0; r < 2; r++) {
        int ref_offset;
        switch (ref_frame[r]) {
          case LAST_FRAME: ref_offset = lst_frame_idx; break;
          // case ALTREF_FRAME: ref_offset = alt_frame_idx; break;
          case GOLDEN_FRAME: ref_offset = gld_frame_idx; break;
#if CONFIG_EXT_REFS
          case LAST2_FRAME: ref_offset = lst2_frame_idx; break;
          case LAST3_FRAME:
            ref_offset = lst3_frame_idx;
            break;
            // case BWDREF_FRAME: ref_offset = bwd_frame_idx; break;
#endif
          default: ref_offset = -2;
        }
        // only initialize with the same refs!
        if (ref_offset == left_offset && ref_offset >= 0) {
          // calculate mv to right
          totalNum++;
        }
      }
    }
  }
  return totalNum;
}
/*
 * find and sort the best ref pairs by distance
 */
void opfl_select_best_ref_pairs(AV1_COMMON *cm, int *left_idx, int *left_offset,
                                int *left_chosen, int *right_idx,
                                int *right_offset, int *right_chosen,
                                int left_most_idx, int right_most_idx) {
  int cur_offset = cm->frame_offset;

  int alt_buf_idx = cm->frame_refs[ALTREF_FRAME - LAST_FRAME].idx;
  int lst_buf_idx = cm->frame_refs[LAST_FRAME - LAST_FRAME].idx;
  int gld_buf_idx = cm->frame_refs[GOLDEN_FRAME - LAST_FRAME].idx;

#if CONFIG_EXT_REFS
  int lst2_buf_idx = cm->frame_refs[LAST2_FRAME - LAST_FRAME].idx;
  int lst3_buf_idx = cm->frame_refs[LAST3_FRAME - LAST_FRAME].idx;
  int bwd_buf_idx = cm->frame_refs[BWDREF_FRAME - LAST_FRAME].idx;
#endif

  int left_cand_idx[INTER_REFS_PER_FRAME],
      left_cand_offset[INTER_REFS_PER_FRAME];
  int right_cand_idx[INTER_REFS_PER_FRAME],
      right_cand_offset[INTER_REFS_PER_FRAME];
  for (int k = 0; k < INTER_REFS_PER_FRAME; k++) {
    left_cand_idx[k] = -1;
    right_cand_idx[k] = -1;
  }
  int numMV_cand[INTER_REFS_PER_FRAME][INTER_REFS_PER_FRAME];

  int this_offset;
  if (alt_buf_idx >= 0) {
    this_offset = cm->cur_frame->alt_frame_offset;
    if (this_offset > cur_offset) {
      right_cand_idx[ALTREF_FRAME - LAST_FRAME] = alt_buf_idx;
      right_cand_offset[ALTREF_FRAME - LAST_FRAME] = this_offset;
    } else if (this_offset < cur_offset) {
      left_cand_idx[ALTREF_FRAME - LAST_FRAME] = alt_buf_idx;
      left_cand_offset[ALTREF_FRAME - LAST_FRAME] = this_offset;
    }
  }
  if (lst_buf_idx >= 0) {
    this_offset = cm->cur_frame->lst_frame_offset;
    if (this_offset > cur_offset) {
      right_cand_idx[LAST_FRAME - LAST_FRAME] = lst_buf_idx;
      right_cand_offset[LAST_FRAME - LAST_FRAME] = this_offset;
    } else if (this_offset < cur_offset) {
      left_cand_idx[LAST_FRAME - LAST_FRAME] = lst_buf_idx;
      left_cand_offset[LAST_FRAME - LAST_FRAME] = this_offset;
    }
  }
  if (gld_buf_idx >= 0) {
    this_offset = cm->cur_frame->gld_frame_offset;
    if (this_offset > cur_offset) {
      right_cand_idx[GOLDEN_FRAME - LAST_FRAME] = gld_buf_idx;
      right_cand_offset[GOLDEN_FRAME - LAST_FRAME] = this_offset;
    } else if (this_offset < cur_offset) {
      left_cand_idx[GOLDEN_FRAME - LAST_FRAME] = gld_buf_idx;
      left_cand_offset[GOLDEN_FRAME - LAST_FRAME] = this_offset;
    }
  }
#if CONFIG_EXT_REFS
  if (lst2_buf_idx >= 0) {
    this_offset = cm->cur_frame->lst2_frame_offset;
    if (this_offset > cur_offset) {
      right_cand_idx[LAST2_FRAME - LAST_FRAME] = lst2_buf_idx;
      right_cand_offset[LAST2_FRAME - LAST_FRAME] = this_offset;
    } else if (this_offset < cur_offset) {
      left_cand_idx[LAST2_FRAME - LAST_FRAME] = lst2_buf_idx;
      left_cand_offset[LAST2_FRAME - LAST_FRAME] = this_offset;
    }
  }
  if (lst3_buf_idx >= 0) {
    this_offset = cm->cur_frame->lst3_frame_offset;
    if (this_offset > cur_offset) {
      right_cand_idx[LAST3_FRAME - LAST_FRAME] = lst3_buf_idx;
      right_cand_offset[LAST3_FRAME - LAST_FRAME] = this_offset;
    } else if (this_offset < cur_offset) {
      left_cand_idx[LAST3_FRAME - LAST_FRAME] = lst3_buf_idx;
      left_cand_offset[LAST3_FRAME - LAST_FRAME] = this_offset;
    }
  }
  if (bwd_buf_idx >= 0) {
    this_offset = cm->cur_frame->bwd_frame_offset;
    if (this_offset > cur_offset) {
      right_cand_idx[BWDREF_FRAME - LAST_FRAME] = bwd_buf_idx;
      right_cand_offset[BWDREF_FRAME - LAST_FRAME] = this_offset;
    } else if (this_offset < cur_offset) {
      left_cand_idx[BWDREF_FRAME - LAST_FRAME] = bwd_buf_idx;
      left_cand_offset[BWDREF_FRAME - LAST_FRAME] = this_offset;
    }
  }
#endif
  // got all candidates, now calculate the number of mvs available
  // for each pair
  for (int ll = LAST_FRAME - LAST_FRAME; ll < OPFL_FRAME - LAST_FRAME; ll++) {
    for (int rr = LAST_FRAME - LAST_FRAME; rr < OPFL_FRAME - LAST_FRAME; rr++) {
      numMV_cand[ll][rr] = 0;
      if (ll == rr) continue;
      if (left_cand_idx[ll] < 0 || right_cand_idx[rr] < 0) continue;
      if (left_cand_idx[ll] == left_most_idx &&
          right_cand_idx[rr] == right_most_idx)
        continue;
      numMV_cand[ll][rr] =
          get_num_MV_between_refs(cm, left_cand_idx[ll], left_cand_offset[ll],
                                  right_cand_idx[rr], right_cand_offset[rr]);
    }
  }
  // find the ones with most available mvs
  int max_num_MV[MAX_NUM_REF_PAIR];
  for (int k = 0; k < MAX_NUM_REF_PAIR; k++) {
    max_num_MV[k] = 0;
  }
  for (int k = 0; k < MAX_NUM_REF_PAIR; k++) {
    left_idx[k] = -1;
    right_idx[k] = -1;
    for (int ll = LAST_FRAME - LAST_FRAME; ll < OPFL_FRAME - LAST_FRAME; ll++) {
      for (int rr = LAST_FRAME - LAST_FRAME; rr < OPFL_FRAME - LAST_FRAME;
           rr++) {
        if (numMV_cand[ll][rr] == 0) continue;
        int skip = 0;
        for (int kk = 0; kk < k; kk++) {
          if (left_cand_idx[ll] == left_idx[kk] &&
              right_cand_idx[rr] == right_idx[kk]) {
            skip = 1;
            break;
          }
        }
        if (skip > 0) continue;
        if (numMV_cand[ll][rr] > max_num_MV[k]) {
          max_num_MV[k] = numMV_cand[ll][rr];
          left_idx[k] = left_cand_idx[ll];
          left_offset[k] = left_cand_offset[ll];
          left_chosen[k] = ll + LAST_FRAME;
          right_idx[k] = right_cand_idx[rr];
          right_offset[k] = right_cand_offset[rr];
          right_chosen[k] = rr + LAST_FRAME;
        }
      }
    }
    // if (left_idx[k] < 0)
    // break;
  }
  int total_dist[MAX_NUM_REF_PAIR];
  for (int k = 0; k < MAX_NUM_REF_PAIR; k++) {
    if (left_idx[k] >= 0)
      total_dist[k] = -left_offset[k] + right_offset[k];
    else
      total_dist[k] = -1;
  }
  // bubble sort by distance to cur_frame (prefer shorter)
  for (int i = 1; i < MAX_NUM_REF_PAIR; i++) {
    if (total_dist[i] < 0) break;
    for (int j = i - 1; j >= 0; j--) {
      if (total_dist[j + 1] < total_dist[j]) {
        // swap j+1 and j
        int temp;
        temp = left_idx[j];
        left_idx[j] = left_idx[j + 1];
        left_idx[j + 1] = temp;
        temp = left_offset[j];
        left_offset[j] = left_offset[j + 1];
        left_offset[j + 1] = temp;
        temp = left_chosen[j];
        left_chosen[j] = left_chosen[j + 1];
        left_chosen[j + 1] = temp;
        temp = right_idx[j];
        right_idx[j] = right_idx[j + 1];
        right_idx[j + 1] = temp;
        temp = right_offset[j];
        right_offset[j] = right_offset[j + 1];
        right_offset[j + 1] = temp;
        temp = right_chosen[j];
        right_chosen[j] = right_chosen[j + 1];
        right_chosen[j + 1] = temp;
        temp = total_dist[j];
        total_dist[j] = total_dist[j + 1];
        total_dist[j + 1] = temp;
        temp = max_num_MV[j];
        max_num_MV[j] = max_num_MV[j + 1];
        max_num_MV[j + 1] = temp;
      } else {
        break;
      }
    }
  }
  // fix when ref_frames are pointing to the same idx
  // TODO(bohan): any other possiblities?
  for (int i = 0; i < MAX_NUM_REF_PAIR; i++) {
    if (left_idx[i] < 0) continue;
    if (right_chosen[i] == BWDREF_FRAME && bwd_buf_idx == alt_buf_idx) {
      right_chosen[i] = ALTREF_FRAME;
    }
    // if (left_chosen[i] == LAST_FRAME && lst_buf_idx == gld_buf_idx) {
    //   left_chosen[i] = GOLDEN_FRAME;
    // }
  }
}
/*
 * find the closest two-sided refs
 */
void opfl_get_closest_refs(AV1_COMMON *cm, int *left_idx_ptr,
                           int *left_offset_ptr, int *left_chosen_ptr,
                           int *right_idx_ptr, int *right_offset_ptr,
                           int *right_chosen_ptr) {
  int left_idx = -1, left_offset = -1, right_idx = -1, right_offset = -1;
  int left_chosen = NONE_FRAME, right_chosen = NONE_FRAME;

  int alt_buf_idx = cm->frame_refs[ALTREF_FRAME - LAST_FRAME].idx;
  int lst_buf_idx = cm->frame_refs[LAST_FRAME - LAST_FRAME].idx;
  int gld_buf_idx = cm->frame_refs[GOLDEN_FRAME - LAST_FRAME].idx;

#if CONFIG_EXT_REFS
  int lst2_buf_idx = cm->frame_refs[LAST2_FRAME - LAST_FRAME].idx;
  int lst3_buf_idx = cm->frame_refs[LAST3_FRAME - LAST_FRAME].idx;
  int bwd_buf_idx = cm->frame_refs[BWDREF_FRAME - LAST_FRAME].idx;
#endif

  int cur_offset = cm->frame_offset;
  int this_offset;
  if (alt_buf_idx >= 0) {
    this_offset = cm->cur_frame->alt_frame_offset;
    if (this_offset > cur_offset &&
        (right_offset < 0 || this_offset < right_offset)) {
      right_idx = alt_buf_idx;
      right_offset = this_offset;
      right_chosen = ALTREF_FRAME;
    } else if (this_offset < cur_offset && this_offset > left_offset) {
      left_idx = alt_buf_idx;
      left_offset = this_offset;
      left_chosen = ALTREF_FRAME;
    }
  }
  if (lst_buf_idx >= 0) {
    this_offset = cm->cur_frame->lst_frame_offset;
    if (this_offset > cur_offset &&
        (right_offset < 0 || this_offset < right_offset)) {
      right_idx = lst_buf_idx;
      right_offset = this_offset;
      right_chosen = LAST_FRAME;
    } else if (this_offset < cur_offset && this_offset > left_offset) {
      left_idx = lst_buf_idx;
      left_offset = this_offset;
      left_chosen = LAST_FRAME;
    }
  }
  if (gld_buf_idx >= 0) {
    this_offset = cm->cur_frame->gld_frame_offset;
    if (this_offset > cur_offset &&
        (right_offset < 0 || this_offset < right_offset)) {
      right_idx = gld_buf_idx;
      right_offset = this_offset;
      right_chosen = GOLDEN_FRAME;
    } else if (this_offset < cur_offset && this_offset > left_offset) {
      left_idx = gld_buf_idx;
      left_offset = this_offset;
      left_chosen = GOLDEN_FRAME;
    }
  }
#if CONFIG_EXT_REFS
  if (lst2_buf_idx >= 0) {
    this_offset = cm->cur_frame->lst2_frame_offset;
    if (this_offset > cur_offset &&
        (right_offset < 0 || this_offset < right_offset)) {
      right_idx = lst2_buf_idx;
      right_offset = this_offset;
      right_chosen = LAST2_FRAME;
    } else if (this_offset < cur_offset && this_offset > left_offset) {
      left_idx = lst2_buf_idx;
      left_offset = this_offset;
      left_chosen = LAST2_FRAME;
    }
  }
  if (lst3_buf_idx >= 0) {
    this_offset = cm->cur_frame->lst3_frame_offset;
    if (this_offset > cur_offset &&
        (right_offset < 0 || this_offset < right_offset)) {
      right_idx = lst3_buf_idx;
      right_offset = this_offset;
      right_chosen = LAST3_FRAME;
    } else if (this_offset < cur_offset && this_offset > left_offset) {
      left_idx = lst3_buf_idx;
      left_offset = this_offset;
      left_chosen = LAST3_FRAME;
    }
  }
  if (bwd_buf_idx >= 0) {
    this_offset = cm->cur_frame->bwd_frame_offset;
    if (this_offset > cur_offset &&
        (right_offset < 0 || this_offset < right_offset)) {
      right_idx = bwd_buf_idx;
      right_offset = this_offset;
      right_chosen = BWDREF_FRAME;
    } else if (this_offset < cur_offset && this_offset > left_offset) {
      left_idx = bwd_buf_idx;
      left_offset = this_offset;
      left_chosen = BWDREF_FRAME;
    }
  }
#endif
  *left_offset_ptr = left_offset;
  *left_chosen_ptr = left_chosen;
  *left_idx_ptr = left_idx;
  *right_offset_ptr = right_offset;
  *right_chosen_ptr = right_chosen;
  *right_idx_ptr = right_idx;
}
/*
 * find the existing initialization motion vectors between any two reference
 * frames
 */
void opfl_set_init_motion(AV1_COMMON *cm, OPFL_BUFFER_STRUCT *buf_struct,
                          int left_idx, int right_idx, int left_offset,
                          int right_offset, int_mv *left_mv, int_mv *right_mv) {
  int cur_offset = buf_struct->cur_offset;
  double dst_pos = ((double)(cur_offset - left_offset)) /
                   ((double)(right_offset - left_offset));

  for (int i = 0; i < cm->mi_rows; i++) {
    for (int j = 0; j < cm->mi_cols; j++) {
      left_mv[i * cm->mi_cols + j].as_int = INVALID_MV;
      right_mv[i * cm->mi_cols + j].as_int = INVALID_MV;
    }
  }
  // process left ref
  MV_REF *mv_ref_base = cm->buffer_pool->frame_bufs[left_idx].mvs;
  int lst_frame_idx = cm->buffer_pool->frame_bufs[left_idx].lst_frame_offset;
  int alt_frame_idx = cm->buffer_pool->frame_bufs[left_idx].alt_frame_offset;
  int gld_frame_idx = cm->buffer_pool->frame_bufs[left_idx].gld_frame_offset;
#if CONFIG_EXT_REFS
  int lst2_frame_idx = cm->buffer_pool->frame_bufs[left_idx].lst2_frame_offset;
  int lst3_frame_idx = cm->buffer_pool->frame_bufs[left_idx].lst3_frame_offset;
  int bwd_frame_idx = cm->buffer_pool->frame_bufs[left_idx].bwd_frame_offset;
#endif
  for (int i = 0; i < cm->mi_rows; i++) {
    for (int j = 0; j < cm->mi_cols; j++) {
      MV_REF *mv_ref = &mv_ref_base[i * cm->mi_cols + j];

      MV this_mvs[2] = { mv_ref->mv[0].as_mv, mv_ref->mv[1].as_mv };
      MV_REFERENCE_FRAME ref_frame[2] = { mv_ref->ref_frame[0],
                                          mv_ref->ref_frame[1] };
      if (ref_frame[0] == OPFL_FRAME && ref_frame[1] == NONE_FRAME) {
        this_mvs[0] = mv_ref->opfl_ref_mvs[0].as_mv;
        this_mvs[1] = mv_ref->opfl_ref_mvs[1].as_mv;
        ref_frame[0] = mv_ref->opfl_ref_frame[0];
        ref_frame[1] = mv_ref->opfl_ref_frame[1];
      }

      for (int r = 0; r < 2; r++) {
        if (ref_frame[r] == NONE_FRAME || ref_frame[r] == OPFL_FRAME) continue;
        int ref_offset;
        switch (ref_frame[r]) {
          case LAST_FRAME: ref_offset = lst_frame_idx; break;
          case ALTREF_FRAME: ref_offset = alt_frame_idx; break;
          case GOLDEN_FRAME: ref_offset = gld_frame_idx; break;
#if CONFIG_EXT_REFS
          case LAST2_FRAME: ref_offset = lst2_frame_idx; break;
          case LAST3_FRAME: ref_offset = lst3_frame_idx; break;
          case BWDREF_FRAME: ref_offset = bwd_frame_idx; break;
#endif
          default: ref_offset = -1;
        }
        // only initialize with the same refs!
        if (ref_offset == right_offset) {
          // calculate mv to left
          int_mv temp_mv;
          temp_mv.as_mv.row =
              opfl_round_double_2_int(-dst_pos * this_mvs[r].row);
          temp_mv.as_mv.col =
              opfl_round_double_2_int(-dst_pos * this_mvs[r].col);
          int mi_r =
              i - opfl_round_double_2_int(temp_mv.as_mv.row / (8.0 * 4.0));
          int mi_c =
              j - opfl_round_double_2_int(temp_mv.as_mv.col / (8.0 * 4.0));
          if (mi_r < 0 || mi_r >= cm->mi_rows || mi_c < 0 ||
              mi_c >= cm->mi_cols)
            continue;
          left_mv[mi_r * cm->mi_cols + mi_c].as_int = temp_mv.as_int;
          // calculate mv to right
          temp_mv.as_mv.row =
              opfl_round_double_2_int((1 - dst_pos) * this_mvs[r].row);
          temp_mv.as_mv.col =
              opfl_round_double_2_int((1 - dst_pos) * this_mvs[r].col);
          right_mv[mi_r * cm->mi_cols + mi_c].as_int = temp_mv.as_int;
        }
      }
    }
  }
  // process right ref
  mv_ref_base = cm->buffer_pool->frame_bufs[right_idx].mvs;
  lst_frame_idx = cm->buffer_pool->frame_bufs[right_idx].lst_frame_offset;
  alt_frame_idx = cm->buffer_pool->frame_bufs[right_idx].alt_frame_offset;
  gld_frame_idx = cm->buffer_pool->frame_bufs[right_idx].gld_frame_offset;
#if CONFIG_EXT_REFS
  lst2_frame_idx = cm->buffer_pool->frame_bufs[right_idx].lst2_frame_offset;
  lst3_frame_idx = cm->buffer_pool->frame_bufs[right_idx].lst3_frame_offset;
  bwd_frame_idx = cm->buffer_pool->frame_bufs[right_idx].bwd_frame_offset;
#endif
  for (int i = 0; i < cm->mi_rows; i++) {
    for (int j = 0; j < cm->mi_cols; j++) {
      MV_REF *mv_ref = &mv_ref_base[i * cm->mi_cols + j];
      MV this_mvs[2] = { mv_ref->mv[0].as_mv, mv_ref->mv[1].as_mv };
      MV_REFERENCE_FRAME ref_frame[2] = { mv_ref->ref_frame[0],
                                          mv_ref->ref_frame[1] };

      if (ref_frame[0] == OPFL_FRAME && ref_frame[1] == NONE_FRAME) {
        this_mvs[0] = mv_ref->opfl_ref_mvs[0].as_mv;
        this_mvs[1] = mv_ref->opfl_ref_mvs[1].as_mv;
        ref_frame[0] = mv_ref->opfl_ref_frame[0];
        ref_frame[1] = mv_ref->opfl_ref_frame[1];
      }

      for (int r = 0; r < 2; r++) {
        if (ref_frame[r] == NONE_FRAME || ref_frame[r] == OPFL_FRAME) continue;
        int ref_offset;
        switch (ref_frame[r]) {
          case LAST_FRAME: ref_offset = lst_frame_idx; break;
          case ALTREF_FRAME: ref_offset = alt_frame_idx; break;
          case GOLDEN_FRAME: ref_offset = gld_frame_idx; break;
#if CONFIG_EXT_REFS
          case LAST2_FRAME: ref_offset = lst2_frame_idx; break;
          case LAST3_FRAME: ref_offset = lst3_frame_idx; break;
          case BWDREF_FRAME: ref_offset = bwd_frame_idx; break;
#endif
          default: ref_offset = -1;
        }
        // only initialize with the same refs!
        if (ref_offset == left_offset) {
          // calculate mv to right
          int_mv temp_mv;
          temp_mv.as_mv.row =
              opfl_round_double_2_int(-(1 - dst_pos) * this_mvs[r].row);
          temp_mv.as_mv.col =
              opfl_round_double_2_int(-(1 - dst_pos) * this_mvs[r].col);
          int mi_r =
              i - opfl_round_double_2_int(temp_mv.as_mv.row / (8.0 * 4.0));
          int mi_c =
              j - opfl_round_double_2_int(temp_mv.as_mv.col / (8.0 * 4.0));
          if (mi_r < 0 || mi_r >= cm->mi_rows || mi_c < 0 ||
              mi_c >= cm->mi_cols)
            continue;
          right_mv[mi_r * cm->mi_cols + mi_c].as_int = temp_mv.as_int;
          // calculate mv to left
          temp_mv.as_mv.row =
              opfl_round_double_2_int(dst_pos * this_mvs[r].row);
          temp_mv.as_mv.col =
              opfl_round_double_2_int(dst_pos * this_mvs[r].col);
          left_mv[mi_r * cm->mi_cols + mi_c].as_int = temp_mv.as_int;
        }
      }
    }
  }
}

/*
 * This function finds the motion vectors between left_idx and right_idx,
 * and forms the motions pointing from left to right
 * unlike opfl_set_init_motion, this function does not scale and set
 * motion vectors according to dst_pos.
 */
void opfl_find_init_motion(AV1_COMMON *cm, int left_idx, int right_idx,
                           int left_offset, int right_offset, int_mv *left_mv) {
  for (int i = 0; i < cm->mi_rows; i++) {
    for (int j = 0; j < cm->mi_cols; j++) {
      left_mv[i * cm->mi_cols + j].as_int = INVALID_MV;
    }
  }
  // process right to left mv
  MV_REF *mv_ref_base = cm->buffer_pool->frame_bufs[right_idx].mvs;
  int lst_frame_idx = cm->buffer_pool->frame_bufs[right_idx].lst_frame_offset;
  int alt_frame_idx = cm->buffer_pool->frame_bufs[right_idx].alt_frame_offset;
  int gld_frame_idx = cm->buffer_pool->frame_bufs[right_idx].gld_frame_offset;
#if CONFIG_EXT_REFS
  int lst2_frame_idx = cm->buffer_pool->frame_bufs[right_idx].lst2_frame_offset;
  int lst3_frame_idx = cm->buffer_pool->frame_bufs[right_idx].lst3_frame_offset;
  int bwd_frame_idx = cm->buffer_pool->frame_bufs[right_idx].bwd_frame_offset;
#endif
  for (int i = 0; i < cm->mi_rows; i++) {
    for (int j = 0; j < cm->mi_cols; j++) {
      MV_REF *mv_ref = &mv_ref_base[i * cm->mi_cols + j];
      MV this_mvs[2] = { mv_ref->mv[0].as_mv, mv_ref->mv[1].as_mv };
      MV_REFERENCE_FRAME ref_frame[2] = { mv_ref->ref_frame[0],
                                          mv_ref->ref_frame[1] };

      if (ref_frame[0] == OPFL_FRAME && ref_frame[1] == NONE_FRAME) {
        this_mvs[0] = mv_ref->opfl_ref_mvs[0].as_mv;
        this_mvs[1] = mv_ref->opfl_ref_mvs[1].as_mv;
        ref_frame[0] = mv_ref->opfl_ref_frame[0];
        ref_frame[1] = mv_ref->opfl_ref_frame[1];
      }

      for (int r = 0; r < 2; r++) {
        int ref_offset;
        switch (ref_frame[r]) {
          case LAST_FRAME: ref_offset = lst_frame_idx; break;
          case ALTREF_FRAME: ref_offset = alt_frame_idx; break;
          case GOLDEN_FRAME: ref_offset = gld_frame_idx; break;
#if CONFIG_EXT_REFS
          case LAST2_FRAME: ref_offset = lst2_frame_idx; break;
          case LAST3_FRAME: ref_offset = lst3_frame_idx; break;
          case BWDREF_FRAME: ref_offset = bwd_frame_idx; break;
#endif
          default: ref_offset = -1;
        }
        // only initialize with the same refs!
        if (ref_offset == left_offset) {
          // calculate mv to right
          int_mv temp_mv;
          temp_mv.as_mv.row = -this_mvs[r].row;
          temp_mv.as_mv.col = -this_mvs[r].col;
          int mi_r = i - (temp_mv.as_mv.row >> (3 + MI_SIZE_LOG2));
          int mi_c = j - (temp_mv.as_mv.col >> (3 + MI_SIZE_LOG2));
          if (mi_r < 0 || mi_r >= cm->mi_rows || mi_c < 0 ||
              mi_c >= cm->mi_cols)
            continue;
          left_mv[mi_r * cm->mi_cols + mi_c].as_int = temp_mv.as_int;
        }
      }
    }
  }
  // process left to right mv
  mv_ref_base = cm->buffer_pool->frame_bufs[left_idx].mvs;
  lst_frame_idx = cm->buffer_pool->frame_bufs[left_idx].lst_frame_offset;
  alt_frame_idx = cm->buffer_pool->frame_bufs[left_idx].alt_frame_offset;
  gld_frame_idx = cm->buffer_pool->frame_bufs[left_idx].gld_frame_offset;
#if CONFIG_EXT_REFS
  lst2_frame_idx = cm->buffer_pool->frame_bufs[left_idx].lst2_frame_offset;
  lst3_frame_idx = cm->buffer_pool->frame_bufs[left_idx].lst3_frame_offset;
  bwd_frame_idx = cm->buffer_pool->frame_bufs[left_idx].bwd_frame_offset;
#endif
  for (int i = 0; i < cm->mi_rows; i++) {
    for (int j = 0; j < cm->mi_cols; j++) {
      MV_REF *mv_ref = &mv_ref_base[i * cm->mi_cols + j];

      MV this_mvs[2] = { mv_ref->mv[0].as_mv, mv_ref->mv[1].as_mv };
      MV_REFERENCE_FRAME ref_frame[2] = { mv_ref->ref_frame[0],
                                          mv_ref->ref_frame[1] };
      if (ref_frame[0] == OPFL_FRAME && ref_frame[1] == NONE_FRAME) {
        this_mvs[0] = mv_ref->opfl_ref_mvs[0].as_mv;
        this_mvs[1] = mv_ref->opfl_ref_mvs[1].as_mv;
        ref_frame[0] = mv_ref->opfl_ref_frame[0];
        ref_frame[1] = mv_ref->opfl_ref_frame[1];
      }

      for (int r = 0; r < 2; r++) {
        int ref_offset;
        switch (ref_frame[r]) {
          case LAST_FRAME: ref_offset = lst_frame_idx; break;
          case ALTREF_FRAME: ref_offset = alt_frame_idx; break;
          case GOLDEN_FRAME: ref_offset = gld_frame_idx; break;
#if CONFIG_EXT_REFS
          case LAST2_FRAME: ref_offset = lst2_frame_idx; break;
          case LAST3_FRAME: ref_offset = lst3_frame_idx; break;
          case BWDREF_FRAME: ref_offset = bwd_frame_idx; break;
#endif
          default: ref_offset = -1;
        }
        if (ref_offset == right_offset) {
          // calculate mv to right
          int_mv temp_mv;
          temp_mv.as_mv.row = this_mvs[r].row;
          temp_mv.as_mv.col = this_mvs[r].col;
          left_mv[i * cm->mi_cols + j].as_int = temp_mv.as_int;
        }
      }
    }
  }
}
/*
 * This function derives initialization motion vector, for example:
 * assume we want mvs between ref1 and ref2, but there exist mvscale
 * mv1 pointing from ref1 to ref3, and mv2 pointing from ref3 to ref2,
 * then the derived mv = mv1 + mv2
 */
void opfl_derive_init_mv(AV1_COMMON *cm, OPFL_BUFFER_STRUCT *buf_struct,
                         int left_idx, int right_idx, int left_offset,
                         int right_offset, int_mv *left_mv, int_mv *right_mv) {
  int alt_buf_idx = cm->frame_refs[ALTREF_FRAME - LAST_FRAME].idx;
  int lst_buf_idx = cm->frame_refs[LAST_FRAME - LAST_FRAME].idx;
  int gld_buf_idx = cm->frame_refs[GOLDEN_FRAME - LAST_FRAME].idx;

#if CONFIG_EXT_REFS
  int lst2_buf_idx = cm->frame_refs[LAST2_FRAME - LAST_FRAME].idx;
  int lst3_buf_idx = cm->frame_refs[LAST3_FRAME - LAST_FRAME].idx;
  int bwd_buf_idx = cm->frame_refs[BWDREF_FRAME - LAST_FRAME].idx;
#endif

  double dstpos = buf_struct->dst_pos;

  // find candidate refs to try
  int cand_idx[6];
  int cand_offset[6];
  int c = 0;
  if (alt_buf_idx != left_idx && alt_buf_idx != right_idx && alt_buf_idx >= 0) {
    int skip = 0;
    for (int k = 0; k < c; k++) {
      if (cand_idx[k] == alt_buf_idx) skip = 1;
    }
    if (skip == 0) {
      cand_idx[c] = alt_buf_idx;
      cand_offset[c] = cm->cur_frame->alt_frame_offset;
      c++;
    }
  }
  if (lst_buf_idx != left_idx && lst_buf_idx != right_idx && lst_buf_idx >= 0) {
    int skip = 0;
    for (int k = 0; k < c; k++) {
      if (cand_idx[k] == lst_buf_idx) skip = 1;
    }
    if (skip == 0) {
      cand_idx[c] = lst_buf_idx;
      cand_offset[c] = cm->cur_frame->lst_frame_offset;
      c++;
    }
  }
  if (gld_buf_idx != left_idx && gld_buf_idx != right_idx && gld_buf_idx >= 0) {
    int skip = 0;
    for (int k = 0; k < c; k++) {
      if (cand_idx[k] == gld_buf_idx) skip = 1;
    }
    if (skip == 0) {
      cand_idx[c] = gld_buf_idx;
      cand_offset[c] = cm->cur_frame->gld_frame_offset;
      c++;
    }
  }
#if CONFIG_EXT_REFS
  if (lst2_buf_idx != left_idx && lst2_buf_idx != right_idx &&
      lst2_buf_idx >= 0) {
    int skip = 0;
    for (int k = 0; k < c; k++) {
      if (cand_idx[k] == lst2_buf_idx) skip = 1;
    }
    if (skip == 0) {
      cand_idx[c] = lst2_buf_idx;
      cand_offset[c] = cm->cur_frame->lst2_frame_offset;
      c++;
    }
  }
  if (lst3_buf_idx != left_idx && lst3_buf_idx != right_idx &&
      lst3_buf_idx >= 0) {
    int skip = 0;
    for (int k = 0; k < c; k++) {
      if (cand_idx[k] == lst3_buf_idx) skip = 1;
    }
    if (skip == 0) {
      cand_idx[c] = lst3_buf_idx;
      cand_offset[c] = cm->cur_frame->lst3_frame_offset;
      c++;
    }
  }
  if (bwd_buf_idx != left_idx && bwd_buf_idx != right_idx && bwd_buf_idx >= 0) {
    int skip = 0;
    for (int k = 0; k < c; k++) {
      if (cand_idx[k] == bwd_buf_idx) skip = 1;
    }
    if (skip == 0) {
      cand_idx[c] = bwd_buf_idx;
      cand_offset[c] = cm->cur_frame->bwd_frame_offset;
      c++;
    }
  }
#endif
  for (; c < 6; c++) {
    cand_idx[c] = -1;
  }

  int_mv *mv_to_left = aom_calloc(cm->mi_cols * cm->mi_rows, sizeof(int_mv));
  int_mv *mv_to_right = aom_calloc(cm->mi_cols * cm->mi_rows, sizeof(int_mv));
  int_mv cur_mv, temp_mv;
  for (c = 0; c < 6; c++) {
    if (cand_idx[c] < 0) break;
    opfl_find_init_motion(cm, cand_idx[c], left_idx, cand_offset[c],
                          left_offset, mv_to_left);
    opfl_find_init_motion(cm, cand_idx[c], right_idx, cand_offset[c],
                          right_offset, mv_to_right);
    for (int i = 0; i < cm->mi_rows; i++) {
      for (int j = 0; j < cm->mi_cols; j++) {
        if (mv_to_left[i * cm->mi_cols + j].as_int == INVALID_MV ||
            mv_to_right[i * cm->mi_cols + j].as_int == INVALID_MV)
          continue;
        // cur_mv points from left to right
        cur_mv.as_mv.row = mv_to_right[i * cm->mi_cols + j].as_mv.row -
                           mv_to_left[i * cm->mi_cols + j].as_mv.row;
        cur_mv.as_mv.col = mv_to_right[i * cm->mi_cols + j].as_mv.col -
                           mv_to_left[i * cm->mi_cols + j].as_mv.col;
        temp_mv.as_mv.row = opfl_round_double_2_int(-dstpos * cur_mv.as_mv.row);
        temp_mv.as_mv.col = opfl_round_double_2_int(-dstpos * cur_mv.as_mv.col);
        int mi_r = i + opfl_round_double_2_int(
                           (mv_to_left[i * cm->mi_cols + j].as_mv.row -
                            temp_mv.as_mv.row) /
                           (8.0 * 4.0));
        int mi_c = j + opfl_round_double_2_int(
                           (mv_to_left[i * cm->mi_cols + j].as_mv.col -
                            temp_mv.as_mv.col) /
                           (8.0 * 4.0));
        if (mi_r < 0 || mi_r >= cm->mi_rows || mi_c < 0 || mi_c >= cm->mi_cols)
          continue;
        if (left_mv[mi_r * cm->mi_cols + mi_c].as_int == INVALID_MV ||
            right_mv[mi_r * cm->mi_cols + mi_c].as_int == INVALID_MV) {
          left_mv[mi_r * cm->mi_cols + mi_c].as_int = temp_mv.as_int;
          // calculate mv to right
          temp_mv.as_mv.row =
              opfl_round_double_2_int((1 - dstpos) * cur_mv.as_mv.row);
          temp_mv.as_mv.col =
              opfl_round_double_2_int((1 - dstpos) * cur_mv.as_mv.col);
          right_mv[mi_r * cm->mi_cols + mi_c].as_int = temp_mv.as_int;
        }
      }
    }
  }
  aom_free(mv_to_left);
  aom_free(mv_to_right);
}

int opfl_get_4x4_warp_dist(AV1_COMMON *cm, OPFL_BUFFER_STRUCT *buf_struct,
                           int starth, int startw, int_mv left_mv,
                           int_mv right_mv) {
  int dist = 0;
  YV12_BUFFER_CONFIG *r0 = buf_struct->ref0_buf[0];
  YV12_BUFFER_CONFIG *r1 = buf_struct->ref1_buf[0];
  uint8_t *src_y0 = r0->y_buffer;
  uint8_t *src_y1 = r1->y_buffer;
  int srcstride = r0->y_stride;
  int lpix, rpix;
  int yl, xl, yr, xr;
  double dil, djl, dir, djr;

  int width = r0->y_width;
  int height = r0->y_height;

  yl = opfl_floor_double_2_int((double)left_mv.as_mv.row / 8.0);
  xl = opfl_floor_double_2_int((double)left_mv.as_mv.col / 8.0);
  yr = opfl_floor_double_2_int((double)right_mv.as_mv.row / 8.0);
  xr = opfl_floor_double_2_int((double)right_mv.as_mv.col / 8.0);

  dil = (double)left_mv.as_mv.row / 8.0 - yl;
  djl = (double)left_mv.as_mv.col / 8.0 - xl;
  dir = (double)right_mv.as_mv.row / 8.0 - yr;
  djr = (double)right_mv.as_mv.col / 8.0 - xr;

  if (starth + yl < 0) {
    yl = -starth;
  } else if (starth + yl >= height) {
    yl = height - starth;
  }
  if (startw + xl < 0) {
    xl = -startw;
  } else if (startw + xl >= width) {
    xl = width - startw;
  }
  if (starth + yr < 0) {
    yr = -starth;
  } else if (starth + yr >= height) {
    yr = height - starth;
  }
  if (startw + xr < 0) {
    xr = -startw;
  } else if (startw + xr >= width) {
    xr = width - startw;
  }

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      uint8_t *tmpsrc;
      tmpsrc = src_y0 + (starth + i + yl) * srcstride + (startw + j + xl);
      lpix = get_sub_pel_y(tmpsrc, srcstride, dil, djl);
      tmpsrc = src_y1 + (starth + i + yr) * srcstride + (startw + j + xr);
      rpix = get_sub_pel_y(tmpsrc, srcstride, dir, djr);
      dist += (lpix - rpix) * (lpix - rpix);
    }
  }
  return dist;
}

void opfl_update_init_motion(AV1_COMMON *cm, OPFL_BUFFER_STRUCT *buf_struct,
                             int_mv *left_mv, int_mv *right_mv, int left_offset,
                             int right_offset, int_mv *left_cand_mv,
                             int_mv *right_cand_mv, int left_cand_offset,
                             int right_cand_offset, int *is_first_valid,
                             int_mv *left_final_mv, int_mv *right_final_mv) {
  int cur_offset = buf_struct->cur_offset;
  int_mv temp_left_mv, temp_right_mv;
  int dist_ori, dist_new;
  double left_ratio, right_ratio;
  left_ratio = ((double)(cur_offset - left_offset)) /
               ((double)(cur_offset - left_cand_offset));
  right_ratio = ((double)(-cur_offset + right_offset)) /
                ((double)(-cur_offset + right_cand_offset));
  for (int i = 0; i < cm->mi_rows; i++) {
    for (int j = 0; j < cm->mi_cols; j++) {
      if (is_first_valid[i * cm->mi_cols + j] > 0) continue;
      if (left_cand_mv[i * cm->mi_cols + j].as_int == INVALID_MV ||
          right_cand_mv[i * cm->mi_cols + j].as_int == INVALID_MV) {
        continue;
      }
      dist_ori = opfl_get_4x4_warp_dist(cm, buf_struct, i * 4, j * 4,
                                        left_mv[i * cm->mi_cols + j],
                                        right_mv[i * cm->mi_cols + j]);

      temp_left_mv.as_mv.row = opfl_round_double_2_int(
          left_cand_mv[i * cm->mi_cols + j].as_mv.row * left_ratio);
      temp_left_mv.as_mv.col = opfl_round_double_2_int(
          left_cand_mv[i * cm->mi_cols + j].as_mv.col * left_ratio);
      temp_right_mv.as_mv.row = opfl_round_double_2_int(
          right_cand_mv[i * cm->mi_cols + j].as_mv.row * right_ratio);
      temp_right_mv.as_mv.col = opfl_round_double_2_int(
          right_cand_mv[i * cm->mi_cols + j].as_mv.col * right_ratio);
      dist_new = opfl_get_4x4_warp_dist(cm, buf_struct, i * 4, j * 4,
                                        temp_left_mv, temp_right_mv);
      if (dist_ori > dist_new) {
        left_mv[i * cm->mi_cols + j].as_int = temp_left_mv.as_int;
        right_mv[i * cm->mi_cols + j].as_int = temp_right_mv.as_int;
        left_final_mv[i * cm->mi_cols + j].as_int = temp_left_mv.as_int;
        right_final_mv[i * cm->mi_cols + j].as_int = temp_right_mv.as_int;
      }
    }
  }
}
/*
 * use optical flow method to calculate motion field of a specific level.
 *
 * Input:
 * buf_struct: containing buffers of the reference frames
 * mf_last: initial motion field
 * level: current scale level, 0 = original, 1 = 0.5, 2 = 0.25, etc.
 * dstpos: dst frame position
 * usescale: 0->do not scale the original, 1->do scaling of images
 * blk_info: information on the current block
 *
 * Output:
 * mf_new: pointer to calculated motion field
 */
void refine_motion_field(OPFL_BUFFER_STRUCT *buf_struct, DB_MV *mf_last,
                         DB_MV *mf_new, int level, double dstpos, int usescale,
                         OPFL_BLK_INFO blk_info) {
  int count = 0;
  double last_cost = DBL_MAX;
  double new_cost = last_cost;
  int width = blk_info.blk_width, height = blk_info.blk_height;
  width = width >> level;
  height = height >> level;
  int mvstr = width + 2 * AVG_MF_BORDER;
  // annealing factor for laplacian multiplier
  double as_scale_factor = 1 << level;
  // iteratively warp and estimate motion field
  while (count < MAX_ITER_OPTICAL_FLOW) {
    // TODO(bohan): combinations of fast and direct methods might help
#if FAST_OPTICAL_FLOW
    new_cost =
        iterate_update_mv_fast(buf_struct, mf_last, mf_new, level, dstpos,
                               as_scale_factor, usescale, blk_info);
#else
    new_cost = iterate_update_mv(buf_struct, mf_last, mf_new, level, dstpos,
                                 as_scale_factor, usescale, blk_info, count);
#endif
    // prepare for the next iteration
    DB_MV *mv_start = mf_last + AVG_MF_BORDER * mvstr + AVG_MF_BORDER;
    DB_MV *mv_start_new = mf_new + AVG_MF_BORDER * mvstr + AVG_MF_BORDER;
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        mv_start[i * mvstr + j].row += mv_start_new[i * mvstr + j].row;
        mv_start[i * mvstr + j].col += mv_start_new[i * mvstr + j].col;
        if (mv_start[i * mvstr + j].row > MAX_MV_LENGTH_1D)
          mv_start[i * mvstr + j].row = MAX_MV_LENGTH_1D;
        else if (mv_start[i * mvstr + j].row < -MAX_MV_LENGTH_1D)
          mv_start[i * mvstr + j].row = -MAX_MV_LENGTH_1D;
        if (mv_start[i * mvstr + j].col > MAX_MV_LENGTH_1D)
          mv_start[i * mvstr + j].col = MAX_MV_LENGTH_1D;
        else if (mv_start[i * mvstr + j].col < -MAX_MV_LENGTH_1D)
          mv_start[i * mvstr + j].col = -MAX_MV_LENGTH_1D;
        mv_start_new[i * mvstr + j].row = mv_start[i * mvstr + j].row;
        mv_start_new[i * mvstr + j].col = mv_start[i * mvstr + j].col;
      }
    }
    last_cost = new_cost;
    count++;
    as_scale_factor *= OPFL_ANNEAL_FACTOR;
  }
  return;
}

/*
 * Update motion field at each iteration by solving linear equations directly.
 *
 * Input:
 * buf_struct: containing buffers of the reference frames
 * mf_last: initial motion field
 * level: current scale level, 0 = original, 1 = 0.5, 2 = 0.25, etc.
 * dstpos: dst frame position
 * as_scale: scale the laplacian multiplier to perform "annealing"
 * usescale: 0->do not scale the original, 1->do scaling of images
 * blk_info: information on the current block
 *
 * Output:
 * mf_new: pointer to calculated motion field
 */
double iterate_update_mv(OPFL_BUFFER_STRUCT *buf_struct, DB_MV *mf_last,
                         DB_MV *mf_new, int level, double dstpos,
                         double as_scale, int usescale, OPFL_BLK_INFO blk_info,
                         int numWarpedRounds) {
  double *Ex, *Ey, *Et;
  double a_squared = OF_A_SQUARED * as_scale;
  double cost = 0;

  YV12_BUFFER_CONFIG *ref0, *ref1, *buf_init0, *buf_init1;
  int l = level;
  if (!usescale) l = 0;
  ref0 = buf_struct->ref0_buf[l];
  ref1 = buf_struct->ref1_buf[l];
  buf_init0 = buf_struct->ref0_warped_buf[l];
  buf_init1 = buf_struct->ref1_warped_buf[l];

  int y_width = blk_info.blk_width, y_height = blk_info.blk_height;
  int starth = blk_info.starth, startw = blk_info.startw;
  int sh = starth >> level, sw = startw >> level;
  int width = y_width, height = y_height;
  width = width >> level;
  height = height >> level;
  if (usescale) {
    y_width = y_width >> level;
    y_height = y_height >> level;
    starth = starth >> level;
    startw = startw >> level;
  }
  int mvstr = width + 2 * AVG_MF_BORDER;
  DB_MV *mv_start = mf_last + AVG_MF_BORDER * mvstr + AVG_MF_BORDER;
  DB_MV *mv_start_new = mf_new + AVG_MF_BORDER * mvstr + AVG_MF_BORDER;
  int i, j;

  // allocate buffers
  Ex = buf_struct->Ex;
  Ey = buf_struct->Ey;
  Et = buf_struct->Et;
  YV12_BUFFER_CONFIG *buffer0 = buf_struct->buffer0[usescale ? level : 0];
  YV12_BUFFER_CONFIG *buffer1 = buf_struct->buffer1[usescale ? level : 0];

  // TODO(bohan): these buffers can also be moved to the buf_struct
  int *row_pos = aom_calloc(width * height * 12, sizeof(int));
  int *col_pos = aom_calloc(width * height * 12, sizeof(int));
  double *values = aom_calloc(width * height * 12, sizeof(double));
  double *mv_vec = aom_calloc(width * height * 2, sizeof(double));
  double *b = aom_calloc(width * height * 2, sizeof(double));
  double *last_mv_vec = mv_vec;
  double *pixel_weight = aom_calloc(width * height, sizeof(double));
  double *mv_weight = aom_calloc(width * height, sizeof(double));

  int imvstr = buf_struct->ref0_buf[0]->y_width;
  imvstr = (imvstr >> level) + 2 * AVG_MF_BORDER;
  DB_MV *initmv = buf_struct->init_mv_buf[level];
  initmv = initmv + (sh + AVG_MF_BORDER) * imvstr + sw + AVG_MF_BORDER;
#if OPFL_INIT_WT
  double *init_wts = buf_struct->init_mv_wts[level];
  init_wts = init_wts + (sh + AVG_MF_BORDER) * imvstr + sw + AVG_MF_BORDER;
#endif
  int fheight = buf_struct->ref0_buf[0]->y_height;
  int fwidth = buf_struct->ref0_buf[0]->y_width;
  fheight = fheight >> level;
  fwidth = fwidth >> level;

  clock_t starts = clock();
  if (level == 0 || !usescale)
    warp_optical_flow_fwd(ref0, ref1, mv_start, mvstr, buffer0, dstpos, level,
                          usescale, blk_info);
  else
    warp_optical_flow_fwd_bilinear(ref0, ref1, mv_start, mvstr, buffer0, dstpos,
                                   level, usescale, blk_info);
  if (level == 0 || !usescale)
    warp_optical_flow_back(ref1, ref0, mv_start, mvstr, buffer1, 1 - dstpos,
                           level, usescale, blk_info);
  else
    warp_optical_flow_back_bilinear(ref1, ref0, mv_start, mvstr, buffer1,
                                    1 - dstpos, level, usescale, blk_info);
  clock_t ends = clock();
  timesub += (double)(ends - starts) / CLOCKS_PER_SEC;

  clock_t startd = clock();
  // Calculate partial derivatives
  opfl_get_derivatives(Ex, Ey, Et, buffer0, buffer1, buf_init0, buf_init1,
                       dstpos, level, usescale, blk_info);
  clock_t endd = clock();
  timeder += (double)(endd - startd) / CLOCKS_PER_SEC;

  clock_t starti = clock(), endi;
  // construct and solve A*mv_vec = b
  SPARSE_MTX A, M, F;
  for (j = 0; j < width; j++) {
    for (i = 0; i < height; i++) {
      last_mv_vec[j * height + i] = mv_start[i * mvstr + j].col;
    }
  }
  for (j = 0; j < width; j++) {
    for (i = 0; i < height; i++) {
      last_mv_vec[width * height + j * height + i] =
          mv_start[i * mvstr + j].row;
    }
  }
  // check if pointing out of bound
  double boundFactor = 0.01;
  double pixExpFactor = 0.005;
  double mvExpFactor = 0.008;
  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      pixel_weight[i * width + j] = 1;
      mv_weight[i * width + j] = 1;
    }
  }
  double i0, i1, j0, j1;
  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      i0 = (double)(i + starth) -
           (1 - buf_struct->dst_pos) * mv_start[i * mvstr + j].row;
      i1 = (double)(i + starth) +
           (buf_struct->dst_pos) * mv_start[i * mvstr + j].row;
      j0 = (double)(j + startw) -
           (1 - buf_struct->dst_pos) * mv_start[i * mvstr + j].col;
      j1 = (double)(j + startw) +
           (buf_struct->dst_pos) * mv_start[i * mvstr + j].col;
      int is_out = (i0 < 0 || i0 > fheight - 1 || i1 < 0 || i1 > fheight - 1 ||
                    j0 < 0 || j0 > fwidth - 1 || j1 < 0 || j1 > fwidth - 1);
      if (is_out) {
        pixel_weight[i * width + j] = 0.001;
        mv_weight[i * width + j] = boundFactor;
      } else if (level > 2 && numWarpedRounds > 2) {
        pixel_weight[i * width + j] =
            exp(-pixExpFactor * Et[i * width + j] * Et[i * width + j]);
        pixel_weight[i * width + j] = (pixel_weight[i * width + j] > 0.001)
                                          ? pixel_weight[i * width + j]
                                          : 0.001;
        mv_weight[i * width + j] =
            exp(-mvExpFactor * Et[i * width + j] * Et[i * width + j]);
        mv_weight[i * width + j] = (mv_weight[i * width + j] > boundFactor)
                                       ? mv_weight[i * width + j]
                                       : boundFactor;
      }
    }
  }
  // get laplacian filter matrix F; Currently using:
  //    0,  1, 0
  //    1, -4, 1
  //    0,  1, 0
  // M = [F 0; 0 F].
  int c = 0;
  double center, up, low, left, right, before;
  for (j = 0; j < width; j++) {
    for (i = 0; i < height; i++) {
      up = 1;
      low = 1;
      left = 1;
      right = 1;
      if (i == 0) {
        up = 0;
      } else {
        up = mv_weight[(i - 1) * width + j];
      }
      if (i == height - 1) {
        low = 0;
      } else {
        low = mv_weight[(i + 1) * width + j];
      }
      if (j == 0) {
        left = 0;
      } else {
        left = mv_weight[i * width + j - 1];
      }
      if (j == width - 1) {
        right = 0;
      } else {
        right = mv_weight[i * width + j + 1];
      }
#if OPFL_INIT_WT
      before = init_wts[i * imvstr + j];
      center = up + low + left + right + before;
#else
      center = up + low + left + right;
#endif
      // normalize
      up = (up / center) * 4;
      low = (low / center) * 4;
      left = (left / center) * 4;
      right = (right / center) * 4;
      center = -4;
      // up
      if (i != 0) {
        row_pos[c] = j * height + i;
        col_pos[c] = j * height + i - 1;
        values[c] = up;
        c++;
      }
      // left
      if (j != 0) {
        row_pos[c] = j * height + i;
        col_pos[c] = (j - 1) * height + i;
        values[c] = left;
        c++;
      }
      // center
      row_pos[c] = j * height + i;
      col_pos[c] = j * height + i;
      values[c] = center;
      c++;
      // right
      if (j != width - 1) {
        row_pos[c] = j * height + i;
        col_pos[c] = (j + 1) * height + i;
        values[c] = right;
        c++;
      }
      // low
      if (i != height - 1) {
        row_pos[c] = j * height + i;
        col_pos[c] = j * height + i + 1;
        values[c] = low;
        c++;
      }
    }
  }

  init_sparse_mtx(row_pos, col_pos, values, c, height * width, height * width,
                  &F);
  init_combine_sparse_mtx(&F, &F, &M, 0, 0, height * width, height * width,
                          2 * height * width, 2 * height * width);
  constant_multiply_sparse_matrix(&M, a_squared);
  // get b
  mtx_vect_multi_right(&M, last_mv_vec, b, 2 * height * width);

  // construct A and modify b
  int offset = height * width;
  c = 0;
  for (j = 0; j < width; j++) {
    for (i = 0; i < height; i++) {
      up = 1;
      low = 1;
      left = 1;
      right = 1;
      if (i == 0) {
        up = 0;
      } else {
        up = mv_weight[(i - 1) * width + j];
      }
      if (i == height - 1) {
        low = 0;
      } else {
        low = mv_weight[(i + 1) * width + j];
      }
      if (j == 0) {
        left = 0;
      } else {
        left = mv_weight[i * width + j - 1];
      }
      if (j == width - 1) {
        right = 0;
      } else {
        right = mv_weight[i * width + j + 1];
      }
#if OPFL_INIT_WT
      before = init_wts[i * imvstr + j];
      center = up + low + left + right + before;
#else
      center = up + low + left + right;
#endif
      // normalize
      up = (up / center) * 4;
      low = (low / center) * 4;
      left = (left / center) * 4;
      right = (right / center) * 4;
#if OPFL_INIT_WT
      // add the init mv factor to b
      before = (before / center) * 4;
      b[j * height + i] += a_squared * before * initmv[i * imvstr + j].col;
      b[j * height + i + offset] +=
          a_squared * before * initmv[i * imvstr + j].row;
#endif
      center = -4;

      if (i != 0) {
        row_pos[c] = j * height + i;
        col_pos[c] = j * height + i - 1;
        values[c] = -a_squared * up;
        c++;
      }

      if (j != 0) {
        row_pos[c] = j * height + i;
        col_pos[c] = (j - 1) * height + i;
        values[c] = -a_squared * left;
        c++;
      }

      row_pos[c] = j * height + i;
      col_pos[c] = j * height + i;
      // if point out of frame, then derivatives not reliable, do not use
      values[c] =
          pixel_weight[i * width + j] * Ex[i * width + j] * Ex[i * width + j] -
          a_squared * center;
      c++;

      if (j != width - 1) {
        row_pos[c] = j * height + i;
        col_pos[c] = (j + 1) * height + i;
        values[c] = -a_squared * right;
        c++;
      }

      if (i != height - 1) {
        row_pos[c] = j * height + i;
        col_pos[c] = j * height + i + 1;
        values[c] = -a_squared * low;
        c++;
      }
    }
  }
  for (j = 0; j < width; j++) {
    for (i = 0; i < height; i++) {
      up = 1;
      low = 1;
      left = 1;
      right = 1;
      if (i == 0) {
        up = 0;
      } else {
        up = mv_weight[(i - 1) * width + j];
      }
      if (i == height - 1) {
        low = 0;
      } else {
        low = mv_weight[(i + 1) * width + j];
      }
      if (j == 0) {
        left = 0;
      } else {
        left = mv_weight[i * width + j - 1];
      }
      if (j == width - 1) {
        right = 0;
      } else {
        right = mv_weight[i * width + j + 1];
      }
#if OPFL_INIT_WT
      before = init_wts[i * imvstr + j];
      center = up + low + left + right + before;
#else
      center = up + low + left + right;
#endif
      // normalize
      up = (up / center) * 4;
      low = (low / center) * 4;
      left = (left / center) * 4;
      right = (right / center) * 4;
      center = -4;

      if (i != 0) {
        row_pos[c] = offset + j * height + i;
        col_pos[c] = offset + j * height + i - 1;
        values[c] = -a_squared * up;
        c++;
      }

      if (j != 0) {
        row_pos[c] = offset + j * height + i;
        col_pos[c] = offset + (j - 1) * height + i;
        values[c] = -a_squared * left;
        c++;
      }

      row_pos[c] = offset + j * height + i;
      col_pos[c] = offset + j * height + i;
      values[c] =
          pixel_weight[i * width + j] * Ey[i * width + j] * Ey[i * width + j] -
          a_squared * center;
      c++;

      if (j != width - 1) {
        row_pos[c] = offset + j * height + i;
        col_pos[c] = offset + (j + 1) * height + i;
        values[c] = -a_squared * right;
        c++;
      }

      if (i != height - 1) {
        row_pos[c] = offset + j * height + i;
        col_pos[c] = offset + j * height + i + 1;
        values[c] = -a_squared * low;
        c++;
      }
    }
  }

  for (j = 0; j < width; j++) {
    for (i = 0; i < height; i++) {
      row_pos[c] = offset + j * height + i;
      col_pos[c] = j * height + i;

      values[c] =
          pixel_weight[i * width + j] * Ex[i * width + j] * Ey[i * width + j];
      c++;
    }
  }
  for (j = 0; j < width; j++) {
    for (i = 0; i < height; i++) {
      row_pos[c] = j * height + i;
      col_pos[c] = offset + j * height + i;

      values[c] =
          pixel_weight[i * width + j] * Ex[i * width + j] * Ey[i * width + j];
      c++;
    }
  }
  init_sparse_mtx(row_pos, col_pos, values, c, 2 * width * height,
                  2 * width * height, &A);

  // adjust b
  for (j = 0; j < width; j++) {
    for (i = 0; i < height; i++) {
      b[j * height + i] = b[j * height + i] - pixel_weight[i * width + j] *
                                                  Et[i * width + j] *
                                                  Ex[i * width + j];
      b[j * height + i + height * width] =
          b[j * height + i + height * width] -
          pixel_weight[i * width + j] * Et[i * width + j] * Ey[i * width + j];
    }
  }
  endi = clock();
  timeinit += (double)(endi - starti) / CLOCKS_PER_SEC;

  starts = clock();
  conjugate_gradient_sparse(&A, b, 2 * width * height, mv_vec);
  ends = clock();
  timesolve += (double)(ends - starts) / CLOCKS_PER_SEC;

  // reshape motion field to 2D
  cost = 0;
  for (j = 0; j < width; j++) {
    for (i = 0; i < height; i++) {
      mv_start_new[i * mvstr + j].col = mv_vec[j * height + i];
      cost += mv_vec[j * height + i] * mv_vec[j * height + i];
    }
  }
  for (j = 0; j < width; j++) {
    for (i = 0; i < height; i++) {
      mv_start_new[i * mvstr + j].row = mv_vec[width * height + j * height + i];
      cost += mv_vec[width * height + j * height + i] *
              mv_vec[width * height + j * height + i];
    }
  }

  // free buffers
  aom_free(row_pos);
  aom_free(col_pos);
  aom_free(values);
  free_sparse_mtx_elems(&A);
  free_sparse_mtx_elems(&F);
  free_sparse_mtx_elems(&M);
  aom_free(mv_vec);
  aom_free(b);
  aom_free(pixel_weight);
  aom_free(mv_weight);

  cost = sqrt(cost);  // 2 norm
  return cost;
}

/*
 * Update motion field at each iteration by a fast iterative method.
 *
 * Input:
 * buf_struct: containing buffers of the reference frames
 * mf_last: initial motion field
 * level: current scale level, 0 = original, 1 = 0.5, 2 = 0.25, etc.
 * dstpos: dst frame position
 * as_scale: scale the laplacian multiplier to perform "annealing"
 * usescale: 0->do not scale the original, 1->do scaling of images
 * blk_info: information on the current block
 *
 * Output:
 * mf_new: pointer to calculated motion field
 */
double iterate_update_mv_fast(OPFL_BUFFER_STRUCT *buf_struct, DB_MV *mf_last,
                              DB_MV *mf_new, int level, double dstpos,
                              double as_scale, int usescale,
                              OPFL_BLK_INFO blk_info) {
  double *Ex, *Ey, *Et;
  double a_squared = OF_A_SQUARED * as_scale;
  double cost = 0;

  YV12_BUFFER_CONFIG *ref0, *ref1, *buf_init0, *buf_init1;
  int l = level;
  if (!usescale) l = 0;
  ref0 = buf_struct->ref0_buf[l];
  ref1 = buf_struct->ref1_buf[l];
  buf_init0 = buf_struct->ref0_warped_buf[l];
  buf_init1 = buf_struct->ref1_warped_buf[l];

  int y_width = blk_info.blk_width, y_height = blk_info.blk_height;
  int starth = blk_info.starth, startw = blk_info.startw;
  int sh = starth >> level, sw = startw >> level;
  int width = y_width, height = y_height;
  width = width >> level;
  height = height >> level;
  if (usescale) {
    y_width = y_width >> level;
    y_height = y_height >> level;
    starth = starth >> level;
    startw = startw >> level;
  }
  int mvstr = width + 2 * AVG_MF_BORDER;
  DB_MV *mv_start = mf_last + AVG_MF_BORDER * mvstr + AVG_MF_BORDER;
  DB_MV *mv_start_new = mf_new + AVG_MF_BORDER * mvstr + AVG_MF_BORDER;
  int i, j;

  // allocate buffers
  Ex = buf_struct->Ex;
  Ey = buf_struct->Ey;
  Et = buf_struct->Et;
  YV12_BUFFER_CONFIG *buffer0 = buf_struct->buffer0[usescale ? level : 0];
  YV12_BUFFER_CONFIG *buffer1 = buf_struct->buffer1[usescale ? level : 0];

  int imvstr = buf_struct->ref0_buf[0]->y_width;
  imvstr = (imvstr >> level) + 2 * AVG_MF_BORDER;
  DB_MV *initmv = buf_struct->init_mv_buf[level];
  initmv = initmv + (sh + AVG_MF_BORDER) * imvstr + sw + AVG_MF_BORDER;
  int fheight = buf_struct->ref0_buf[0]->y_height;
  int fwidth = buf_struct->ref0_buf[0]->y_width;
  fheight = fheight >> level;
  fwidth = fwidth >> level;

  clock_t starts = clock();
  if (level == 0 || !usescale)
    warp_optical_flow_fwd(ref0, ref1, mv_start, mvstr, buffer0, dstpos, level,
                          usescale, blk_info);
  else
    warp_optical_flow_fwd_bilinear(ref0, ref1, mv_start, mvstr, buffer0, dstpos,
                                   level, usescale, blk_info);
  if (level == 0 || !usescale)
    warp_optical_flow_back(ref1, ref0, mv_start, mvstr, buffer1, 1 - dstpos,
                           level, usescale, blk_info);
  else
    warp_optical_flow_back_bilinear(ref1, ref0, mv_start, mvstr, buffer1,
                                    1 - dstpos, level, usescale, blk_info);
  clock_t ends = clock();
  timesub += (double)(ends - starts) / CLOCKS_PER_SEC;

  clock_t startd = clock();
  // Calculate partial derivatives
  opfl_get_derivatives(Ex, Ey, Et, buffer0, buffer1, buf_init0, buf_init1,
                       dstpos, level, usescale, blk_info);
  clock_t endd = clock();
  timeder += (double)(endd - startd) / CLOCKS_PER_SEC;

  // iterative solver
  starts = clock();
  DB_MV *tempmv;
  DB_MV *bufmv_b =
      aom_calloc((height + 2 * AVG_MF_BORDER) * (width + 2 * AVG_MF_BORDER),
                 sizeof(DB_MV));
  DB_MV *bufmv = bufmv_b + AVG_MF_BORDER * mvstr + AVG_MF_BORDER;
  DB_MV *lp_last = aom_calloc(height * mvstr, sizeof(DB_MV));
  double *denorm = aom_calloc(height * width, sizeof(double));
  DB_MV avg;
  // get the laplacian of initial motion field
  pad_motion_field_border(mv_start, width, height, mvstr);
  int i0, i1, j0, j1;
  for (i = 0; i < height; i++) {
    i0 = i - 1;
    i1 = i + 1;
    for (j = 0; j < width; j++) {
      j0 = j - 1;
      j1 = j + 1;
      bufmv[i * mvstr + j].row = 0;
      bufmv[i * mvstr + j].col = 0;

      lp_last[i * mvstr + j].row = 0.25 * mv_start[i0 * mvstr + j].row +
                                   0.25 * mv_start[i1 * mvstr + j].row +
                                   0.25 * mv_start[i * mvstr + j0].row +
                                   0.25 * mv_start[i * mvstr + j1].row;
      lp_last[i * mvstr + j].row -= mv_start[i * mvstr + j].row;
      lp_last[i * mvstr + j].col = 0.25 * mv_start[i0 * mvstr + j].col +
                                   0.25 * mv_start[i1 * mvstr + j].col +
                                   0.25 * mv_start[i * mvstr + j0].col +
                                   0.25 * mv_start[i * mvstr + j1].col;
      lp_last[i * mvstr + j].col -= mv_start[i * mvstr + j].col;
      denorm[i * width + j] = 16 * a_squared +
                              Ex[i * width + j] * Ex[i * width + j] +
                              Ey[i * width + j] * Ey[i * width + j];
    }
  }
  // calculate the motion field
  for (int k = 0; k < MAX_ITER_FAST_OPFL; k++) {
    pad_motion_field_border(bufmv, width, height, mvstr);
    for (i = 0; i < height; i++) {
      i0 = i - 1;
      i1 = i + 1;
      for (j = 0; j < width; j++) {
        j0 = j - 1;
        j1 = j + 1;
        avg.row = 0.25 * bufmv[i0 * mvstr + j].row +
                  0.25 * bufmv[i1 * mvstr + j].row +
                  0.25 * bufmv[i * mvstr + j0].row +
                  0.25 * bufmv[i * mvstr + j1].row;
        avg.row += lp_last[i * mvstr + j].row;
        avg.col = 0.25 * bufmv[i0 * mvstr + j].col +
                  0.25 * bufmv[i1 * mvstr + j].col +
                  0.25 * bufmv[i * mvstr + j0].col +
                  0.25 * bufmv[i * mvstr + j1].col;
        avg.col += lp_last[i * mvstr + j].col;
        mv_start_new[i * mvstr + j].col =
            avg.col - Ex[i * width + j] *
                          (Ex[i * width + j] * avg.col +
                           Ey[i * width + j] * avg.row + Et[i * width + j]) /
                          denorm[i * width + j];
        mv_start_new[i * mvstr + j].row =
            avg.row - Ey[i * width + j] *
                          (Ex[i * width + j] * avg.col +
                           Ey[i * width + j] * avg.row + Et[i * width + j]) /
                          denorm[i * width + j];
      }
    }
    if (k < MAX_ITER_FAST_OPFL - 1) {
      tempmv = bufmv;
      bufmv = mv_start_new;
      mv_start_new = tempmv;
    }
  }

  aom_free(bufmv_b);
  aom_free(lp_last);
  aom_free(denorm);
  ends = clock();
  timesolve += (double)(ends - starts) / CLOCKS_PER_SEC;

  // reshape motion field to 2D
  cost = 0;
  for (j = 0; j < width; j++) {
    for (i = 0; i < height; i++) {
      cost += mv_start_new[i * mvstr + j].row * mv_start_new[i * mvstr + j].row;
    }
  }
  for (j = 0; j < width; j++) {
    for (i = 0; i < height; i++) {
      cost += mv_start_new[i * mvstr + j].col * mv_start_new[i * mvstr + j].col;
    }
  }

  cost = sqrt(cost);  // 2 norm
  return cost;
}

void opfl_get_derivatives(double *Ex, double *Ey, double *Et,
                          YV12_BUFFER_CONFIG *buffer0,
                          YV12_BUFFER_CONFIG *buffer1,
                          YV12_BUFFER_CONFIG *buffer_init0,
                          YV12_BUFFER_CONFIG *buffer_init1, double dstpos,
                          int level, int usescale, OPFL_BLK_INFO blk_info) {
  int lh = DERIVATIVE_FILTER_LENGTH;
  int hleft = (lh - 1) / 2;
  double filter[DERIVATIVE_FILTER_LENGTH] = { -1.0 / 60, 9.0 / 60,  -45.0 / 60,
                                              0,         45.0 / 60, -9.0 / 60,
                                              1.0 / 60 };
  int idx, i, j;
  int width = blk_info.blk_width;
  int height = blk_info.blk_height;
  int starth = blk_info.starth;
  int startw = blk_info.startw;
  int stride = buffer0->y_stride;
  int istride = buffer_init0->y_stride;

  if (usescale) {
    width = width >> level;
    height = height >> level;
    starth = starth >> level;
    startw = startw >> level;
  }

  uint8_t *buf0i = buffer_init0->y_buffer + starth * istride + startw;
  uint8_t *buf1i = buffer_init1->y_buffer + starth * istride + startw;

  double *tempEx = NULL, *tempEy = NULL, *tempEt = NULL;
  double *oriEx = Ex, *oriEy = Ey, *oriEt = Et;
  if (!usescale && level != 0) {
    tempEx = aom_calloc(width * height, sizeof(double));
    tempEy = aom_calloc(width * height, sizeof(double));
    tempEt = aom_calloc(width * height, sizeof(double));
    Ex = tempEx;
    Ey = tempEy;
    Et = tempEt;
  }

  // horizontal derivative filter
  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      Ex[i * width + j] = 0;
      for (int k = 0; k < lh; k++) {
        idx = j + (k - hleft);
        if ((idx < 0 && blk_info.leftbound != 1) ||
            (idx > width - 1 && blk_info.rightbound != 1)) {
          Ex[i * width + j] +=
              filter[k] * (double)(buf0i[i * istride + idx]) * (1 - dstpos) +
              filter[k] * (double)(buf1i[i * istride + idx]) * dstpos;
        } else {
          if (idx < 0)
            idx = 0;
          else if (idx > width - 1)
            idx = width - 1;
          Ex[i * width + j] +=
              filter[k] * (double)(buffer0->y_buffer[i * stride + idx]) *
                  (1 - dstpos) +
              filter[k] * (double)(buffer1->y_buffer[i * stride + idx]) *
                  dstpos;
        }
      }
    }
  }
  // vertical derivative filter
  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      Ey[i * width + j] = 0;
      for (int k = 0; k < lh; k++) {
        idx = i + (k - hleft);
        if ((idx < 0 && blk_info.upbound != 1) ||
            (idx > height - 1 && blk_info.lowerbound != 1)) {
          Ey[i * width + j] +=
              filter[k] * (double)(buf0i[idx * istride + j]) * (1 - dstpos) +
              filter[k] * (double)(buf1i[idx * istride + j]) * dstpos;
        } else {
          if (idx < 0)
            idx = 0;
          else if (idx > height - 1)
            idx = height - 1;
          Ey[i * width + j] +=
              filter[k] * (double)(buffer0->y_buffer[idx * stride + j]) *
                  (1 - dstpos) +
              filter[k] * (double)(buffer1->y_buffer[idx * stride + j]) *
                  dstpos;
        }
      }
    }
  }
  // time derivative
  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      Et[i * width + j] = (double)(buffer1->y_buffer[i * stride + j]) -
                          (double)(buffer0->y_buffer[i * stride + j]);
    }
  }
  // rescale the derivatives
  if (!usescale && level != 0) {
    Ex = oriEx;
    Ey = oriEy;
    Et = oriEt;
    int s_width = width >> level, s_height = height >> level,
        blk_w = 1 << level;
    for (i = 0; i < s_height; i++) {
      for (j = 0; j < s_width; j++) {
        Ex[i * s_width + j] = 0;
        Ey[i * s_width + j] = 0;
        Et[i * s_width + j] = 0;
        for (int h = 0; h < blk_w; h++) {
          for (int w = 0; w < blk_w; w++) {
            Ex[i * s_width + j] +=
                tempEx[(i * blk_w + h) * width + j * blk_w + w];
            Ey[i * s_width + j] +=
                tempEy[(i * blk_w + h) * width + j * blk_w + w];
            Et[i * s_width + j] +=
                tempEt[(i * blk_w + h) * width + j * blk_w + w];
          }
        }
        Ex[i * s_width + j] /= blk_w;
        Ey[i * s_width + j] /= blk_w;
        Et[i * s_width + j] /= (blk_w * blk_w);
      }
    }
    if (tempEx) aom_free(tempEx);
    if (tempEy) aom_free(tempEy);
    if (tempEt) aom_free(tempEt);
  }
}

/*
 * Warp the Y component of src to dst according to the motion field
 * Motion field points back from dst to src
 */
void warp_optical_flow_back(YV12_BUFFER_CONFIG *src, YV12_BUFFER_CONFIG *ref,
                            DB_MV *mf_start, int mvstr, YV12_BUFFER_CONFIG *dst,
                            double dstpos, int level, int usescale,
                            OPFL_BLK_INFO blk_info) {
  int fwidth = src->y_width, fheight = src->y_height;
  int width = blk_info.blk_width;
  int height = blk_info.blk_height;
  int starth = blk_info.starth;
  int startw = blk_info.startw;
  int srcstride = src->y_stride;
  int dststride = dst->y_stride;
  uint8_t *srcy = src->y_buffer;
  uint8_t *dsty = dst->y_buffer;
  uint8_t *refy = ref->y_buffer;

  double ii, jj, di, dj;
  int i0, j0;

  int blk_w = 1;
  if (!usescale) {
    blk_w = blk_w << level;
  } else {
    width = width >> level;
    height = height >> level;
    starth = starth >> level;
    startw = startw >> level;
  }

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      ii = i + starth +
           mf_start[(i / blk_w) * mvstr + j / blk_w].row * blk_w * dstpos;
      jj = j + startw +
           mf_start[(i / blk_w) * mvstr + j / blk_w].col * blk_w * dstpos;
      i0 = opfl_floor_double_2_int(ii);
      di = ii - i0;
      j0 = opfl_floor_double_2_int(jj);
      dj = jj - j0;
      if (i0 < 0 || i0 > fheight - 1 || j0 < 0 || j0 > fwidth - 1) {
        dsty[i * dststride + j] = refy[(i + starth) * srcstride + j + startw];
        continue;
      }
      dsty[i * dststride + j] =
          get_sub_pel_y(srcy + i0 * srcstride + j0, srcstride, di, dj);
    }
  }
}

/*
 * Warp the Y component of src to dst using bilinear method
 * Motion field points back from dst to src
 */
void warp_optical_flow_back_bilinear(YV12_BUFFER_CONFIG *src,
                                     YV12_BUFFER_CONFIG *ref, DB_MV *mf_start,
                                     int mvstr, YV12_BUFFER_CONFIG *dst,
                                     double dstpos, int level, int usescale,
                                     OPFL_BLK_INFO blk_info) {
  int fwidth = src->y_width, fheight = src->y_height;
  int width = blk_info.blk_width;
  int height = blk_info.blk_height;
  int starth = blk_info.starth;
  int startw = blk_info.startw;
  int srcstride = src->y_stride;
  int dststride = dst->y_stride;
  uint8_t *srcy = src->y_buffer;
  uint8_t *dsty = dst->y_buffer;
  uint8_t *refy = ref->y_buffer;

  double ii, jj, di, dj, temp;
  int i0, j0, i1, j1;

  int blk_w = 1;
  if (!usescale) {
    blk_w = blk_w << level;
  } else {
    width = width >> level;
    height = height >> level;
    starth = starth >> level;
    startw = startw >> level;
  }

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      ii = i + starth +
           mf_start[(i / blk_w) * mvstr + j / blk_w].row * blk_w * dstpos;
      jj = j + startw +
           mf_start[(i / blk_w) * mvstr + j / blk_w].col * blk_w * dstpos;
      i0 = opfl_floor_double_2_int(ii);
      di = 1 - ii + i0;
      i1 = i0 + 1;
      j0 = opfl_floor_double_2_int(jj);
      dj = 1 - jj + j0;
      j1 = j0 + 1;
      if (i0 < 0 || i0 > fheight - 1 || j0 < 0 || j0 > fwidth - 1) {
        dsty[i * dststride + j] = refy[(i + starth) * srcstride + j + startw];
        continue;
      }
      temp = di * dj * (double)srcy[i0 * srcstride + j0] +
             di * (1 - dj) * (double)srcy[i0 * srcstride + j1] +
             (1 - di) * dj * (double)srcy[i1 * srcstride + j0] +
             (1 - di) * (1 - dj) * (double)srcy[i1 * srcstride + j1];
      dsty[i * dststride + j] = (uint8_t)(temp + 0.5);
    }
  }
}

/*
 * Warp the Y component of src to dst
 * Motion field points forward from src to dst
 */
void warp_optical_flow_fwd(YV12_BUFFER_CONFIG *src, YV12_BUFFER_CONFIG *ref,
                           DB_MV *mf_start, int mvstr, YV12_BUFFER_CONFIG *dst,
                           double dstpos, int level, int usescale,
                           OPFL_BLK_INFO blk_info) {
  int fwidth = src->y_width, fheight = src->y_height;
  int width = blk_info.blk_width;
  int height = blk_info.blk_height;
  int starth = blk_info.starth;
  int startw = blk_info.startw;
  int srcstride = src->y_stride;
  int dststride = dst->y_stride;
  uint8_t *srcy = src->y_buffer;
  uint8_t *dsty = dst->y_buffer;
  uint8_t *refy = ref->y_buffer;

  double ii, jj, di, dj;
  int i0, j0;

  int blk_w = 1;
  if (!usescale) {
    blk_w = blk_w << level;
  } else {
    width = width >> level;
    height = height >> level;
    starth = starth >> level;
    startw = startw >> level;
  }

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      ii = i + starth -
           mf_start[(i / blk_w) * mvstr + j / blk_w].row * blk_w * dstpos;
      jj = j + startw -
           mf_start[(i / blk_w) * mvstr + j / blk_w].col * blk_w * dstpos;
      i0 = opfl_floor_double_2_int(ii);
      di = ii - i0;
      j0 = opfl_floor_double_2_int(jj);
      dj = jj - j0;
      if (i0 < 0 || i0 > fheight - 1 || j0 < 0 || j0 > fwidth - 1) {
        dsty[i * dststride + j] = refy[(i + starth) * srcstride + j + startw];
        continue;
      }
      //      printf("%d, %d, %d, %d, %d, %d, %d, %d, %d\n",i, j, dststride,
      //      width, height, fwidth, fheight, startw, starth);
      dsty[i * dststride + j] =
          get_sub_pel_y(srcy + i0 * srcstride + j0, srcstride, di, dj);
    }
  }
}

/*
 * Warp the Y component of src to dst using bilinear method
 * Motion field points forward from src to dst
 */
void warp_optical_flow_fwd_bilinear(YV12_BUFFER_CONFIG *src,
                                    YV12_BUFFER_CONFIG *ref, DB_MV *mf_start,
                                    int mvstr, YV12_BUFFER_CONFIG *dst,
                                    double dstpos, int level, int usescale,
                                    OPFL_BLK_INFO blk_info) {
  int fwidth = src->y_width, fheight = src->y_height;
  int width = blk_info.blk_width;
  int height = blk_info.blk_height;
  int starth = blk_info.starth;
  int startw = blk_info.startw;
  int srcstride = src->y_stride;
  int dststride = dst->y_stride;
  uint8_t *srcy = src->y_buffer;
  uint8_t *dsty = dst->y_buffer;
  uint8_t *refy = ref->y_buffer;

  double ii, jj, di, dj, temp;
  int i0, j0, i1, j1;

  int blk_w = 1;
  if (!usescale) {
    blk_w = blk_w << level;
  } else {
    width = width >> level;
    height = height >> level;
    starth = starth >> level;
    startw = startw >> level;
  }

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      ii = i + starth -
           mf_start[(i / blk_w) * mvstr + j / blk_w].row * blk_w * dstpos;
      jj = j + startw -
           mf_start[(i / blk_w) * mvstr + j / blk_w].col * blk_w * dstpos;
      i0 = opfl_floor_double_2_int(ii);
      di = 1 - ii + i0;
      i1 = i0 + 1;
      j0 = opfl_floor_double_2_int(jj);
      dj = 1 - jj + j0;
      j1 = j0 + 1;
      if (i0 < 0 || i0 > fheight - 1 || j0 < 0 || j0 > fwidth - 1) {
        dsty[i * dststride + j] = refy[(i + starth) * srcstride + j + startw];
        continue;
      }
      temp = di * dj * (double)srcy[i0 * srcstride + j0] +
             di * (1 - dj) * (double)srcy[i0 * srcstride + j1] +
             (1 - di) * dj * (double)srcy[i1 * srcstride + j0] +
             (1 - di) * (1 - dj) * (double)srcy[i1 * srcstride + j1];
      dsty[i * dststride + j] = (uint8_t)(temp + 0.5);
    }
  }
}

/*
 * Interpolate references according to the motion field
 *
 * Input:
 * src0, src1: the reference frames
 * mf_start: the motion field start pointer
 * mvstr: motion field stride
 * dstpos: position of the interpolated frame
 * method: the blend method to be used
 * blk_info: block information
 *
 * Output:
 * dst: pointer to the interpolated frame
 */
void warp_optical_flow(YV12_BUFFER_CONFIG *src0, YV12_BUFFER_CONFIG *src1,
                       DB_MV *mf_start, int mvstr, YV12_BUFFER_CONFIG *dst,
                       double dstpos, OPFL_BLEND_METHOD method,
                       OPFL_BLK_INFO blk_info) {
  if (method == OPFL_DIFF_SELECT) {
    // warp_optical_flow_diff_select(src0, src1, mf_start, mvstr, dst, dstpos);
    warp_optical_flow_bilateral(src0, src1, mf_start, mvstr, dst, dstpos);
    return;
  }
  int fwidth = dst->y_width, fheight = dst->y_height;
  int width = blk_info.blk_width;
  int height = blk_info.blk_height;
  int starth = blk_info.starth;
  int startw = blk_info.startw;
  int starthuv = starth >> 1;
  int startwuv = startw >> 1;
  int stride = src0->y_stride;
  int uvstride = src0->uv_stride;

  uint8_t *src0y = src0->y_buffer + starth * stride + startw;
  uint8_t *src1y = src1->y_buffer + starth * stride + startw;
  uint8_t *dsty = dst->y_buffer + starth * stride + startw;
  uint8_t *src0u = src0->u_buffer + starthuv * uvstride + startwuv;
  uint8_t *src0v = src0->v_buffer + starthuv * uvstride + startwuv;
  uint8_t *src1u = src1->u_buffer + starthuv * uvstride + startwuv;
  uint8_t *src1v = src1->v_buffer + starthuv * uvstride + startwuv;
  uint8_t *dstu = dst->u_buffer + starthuv * uvstride + startwuv;
  uint8_t *dstv = dst->v_buffer + starthuv * uvstride + startwuv;

  double ii0, jj0, di0, dj0, di0uv = 0, dj0uv = 0;
  double ii1, jj1, di1, dj1, di1uv = 0, dj1uv = 0;
  int i0, j0;
  int i1, j1;
  double dstpel_y, dstpel_u, dstpel_v;
  double dstpel_y0 = 0, dstpel_y1 = 0;
  double pos;
  int do_uv;
  int use0, use1, inside0, inside1;
  int nearest = ((dstpos <= 0.5) ? 0 : 1);

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      pos = dstpos;
      ii0 = i - mf_start[i * mvstr + j].row * dstpos;
      jj0 = j - mf_start[i * mvstr + j].col * dstpos;
      ii1 = i + mf_start[i * mvstr + j].row * (1 - dstpos);
      jj1 = j + mf_start[i * mvstr + j].col * (1 - dstpos);
      i0 = opfl_floor_double_2_int(ii0);
      di0 = ii0 - i0;
      j0 = opfl_floor_double_2_int(jj0);
      dj0 = jj0 - j0;
      i1 = opfl_floor_double_2_int(ii1);
      di1 = ii1 - i1;
      j1 = opfl_floor_double_2_int(jj1);
      dj1 = jj1 - j1;

      do_uv = i % 2 == 0 && j % 2 == 0;  // TODO(bohan) only considering 420 now
      if (do_uv) {
        di0uv = di0 / 2 + 0.5 * ((i0 % 2 + 2) % 2);
        dj0uv = dj0 / 2 + 0.5 * ((j0 % 2 + 2) % 2);
        di1uv = di1 / 2 + 0.5 * ((i1 % 2 + 2) % 2);
        dj1uv = dj1 / 2 + 0.5 * ((j1 % 2 + 2) % 2);
      }

      // Check availability of the references.
      // If one ref is outside then do not use it.
      inside0 = (i0 + starth >= 0 && i0 + starth < fheight - 1 &&
                 j0 + startw >= 0 && j0 + startw < fwidth - 1);
      inside1 = (i1 + starth >= 0 && i1 + starth < fheight - 1 &&
                 j1 + startw >= 0 && j1 + startw < fwidth - 1);
      use0 = inside0 == inside1 || inside0;
      use1 = inside0 == inside1 || inside1;

      // If use nearest single method, then use only one reference
      if (method == OPFL_NEAREST_SINGLE) {
        if (use0 && use1) {
          use0 = (nearest == 0);
          use1 = (nearest == 1);
        }
      }

      // calculate subpel Y refs
      if (use0) {
        dstpel_y0 =
            (double)get_sub_pel_y(src0y + i0 * stride + j0, stride, di0, dj0);
      }
      if (use1) {
        dstpel_y1 =
            (double)get_sub_pel_y(src1y + i1 * stride + j1, stride, di1, dj1);
      }

      // If use diff single method, check the pixels when the refs do not agree
      if (method == OPFL_DIFF_SINGLE) {
        if (use0 && use1) {
          if (fabs(dstpel_y0 - dstpel_y1) > OPTICAL_FLOW_DIFF_THRES) {
            use0 = (nearest == 0);
            use1 = (nearest == 1);
          }
        }
      }

      if (use0 && !use1)
        pos = 0;
      else if (!use0 && use1)
        pos = 1;

      // blend
      dstpel_y = 0;
      dstpel_u = 0;
      dstpel_v = 0;
      if (use0) {
        dstpel_y += dstpel_y0 * (1 - pos);
        if (do_uv) {
          dstpel_u +=
              (double)get_sub_pel_uv(
                  src0u + (int)floor(ii0 / 2) * uvstride + (int)floor(jj0 / 2),
                  uvstride, di0uv, dj0uv) *
              (1 - pos);
          dstpel_v +=
              (double)get_sub_pel_uv(
                  src0v + (int)floor(ii0 / 2) * uvstride + (int)floor(jj0 / 2),
                  uvstride, di0uv, dj0uv) *
              (1 - pos);
        }
      }
      if (use1) {
        dstpel_y += dstpel_y1 * pos;
        if (do_uv) {
          dstpel_u +=
              (double)get_sub_pel_uv(
                  src1u + (int)floor(ii1 / 2) * uvstride + (int)floor(jj1 / 2),
                  uvstride, di1uv, dj1uv) *
              pos;
          dstpel_v +=
              (double)get_sub_pel_uv(
                  src1v + (int)floor(ii1 / 2) * uvstride + (int)floor(jj1 / 2),
                  uvstride, di1uv, dj1uv) *
              pos;
        }
      }
      dsty[i * stride + j] = opfl_round_double_2_int(dstpel_y);
      if (do_uv) {
        dstu[i / 2 * uvstride + j / 2] = opfl_round_double_2_int(dstpel_u);
        dstv[i / 2 * uvstride + j / 2] = opfl_round_double_2_int(dstpel_v);
      }
    }
  }
}

/*
 * Interpolate references according to the motion field
 * when the refs do not agree, use more advanced selection
 *
 * Input:
 * src0, src1: the reference frames
 * mf_start: the motion field start pointer
 * mvstr: motion field stride
 * dstpos: position of the interpolated frame
 *
 * Output:
 * dst: pointer to the interpolated frame
 */
// TODO(bohan): this function needs to be updated to be used with
//              block-based methods
void warp_optical_flow_diff_select(YV12_BUFFER_CONFIG *src0,
                                   YV12_BUFFER_CONFIG *src1, DB_MV *mf_start,
                                   int mvstr, YV12_BUFFER_CONFIG *dst,
                                   double dstpos) {
  int width = src0->y_width;
  int height = src0->y_height;
  int stride = src0->y_stride;
  int uvstride = src0->uv_stride;

  uint8_t *src0y = src0->y_buffer;
  uint8_t *src1y = src1->y_buffer;
  uint8_t *dsty = dst->y_buffer;
  uint8_t *src0u = src0->u_buffer;
  uint8_t *src0v = src0->v_buffer;
  uint8_t *src1u = src1->u_buffer;
  uint8_t *src1v = src1->v_buffer;
  uint8_t *dstu = dst->u_buffer;
  uint8_t *dstv = dst->v_buffer;

  double ii0, jj0, di0, dj0, di0uv, dj0uv;
  double ii1, jj1, di1, dj1, di1uv, dj1uv;
  int i0, j0;
  int i1, j1;
  double dstpel_y, dstpel_u, dstpel_v;
  double pos;
  int do_uv;

  double *used0 = aom_calloc(height * width, sizeof(double));
  double *used1 = aom_calloc(height * width, sizeof(double));
  // refid: -1=unset; 0=ref0; 1=ref1; 2=both
  int *refid = aom_calloc(height * width, sizeof(int));
  int *refid_mode = aom_calloc(height * width, sizeof(int));
  double *dstpel_y0 = aom_calloc(height * width, sizeof(double));
  double *dstpel_y1 = aom_calloc(height * width, sizeof(double));

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      used0[i * width + j] = 0;
      used1[i * width + j] = 0;
      refid[i * width + j] = -1;
    }
  }
  // first check all pixels to see if the refs agree
  // also note down which pixels in refs are used for reference
  int disagree_cnt = 0;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      pos = dstpos;
      ii0 = i - mf_start[i * mvstr + j].row * dstpos;
      jj0 = j - mf_start[i * mvstr + j].col * dstpos;
      ii1 = i + mf_start[i * mvstr + j].row * (1 - dstpos);
      jj1 = j + mf_start[i * mvstr + j].col * (1 - dstpos);
      i0 = opfl_floor_double_2_int(ii0);
      di0 = ii0 - i0;
      j0 = opfl_floor_double_2_int(jj0);
      dj0 = jj0 - j0;
      i1 = opfl_floor_double_2_int(ii1);
      di1 = ii1 - i1;
      j1 = opfl_floor_double_2_int(jj1);
      dj1 = jj1 - j1;

      int inside0 = (i0 >= 0 && i0 < height - 1 && j0 >= 0 && j0 < width - 1);
      int inside1 = (i1 >= 0 && i1 < height - 1 && j1 >= 0 && j1 < width - 1);
      int use0 = inside0 == inside1 || inside0;
      int use1 = inside0 == inside1 || inside1;
      if (inside0 && !inside1)
        pos = 0;
      else if (inside1 && !inside0)
        pos = 1;

      dstpel_y0[i * width + j] =
          (double)get_sub_pel_y(src0y + i0 * stride + j0, stride, di0, dj0);
      dstpel_y1[i * width + j] =
          (double)get_sub_pel_y(src1y + i1 * stride + j1, stride, di1, dj1);
      // check if the refs are similar
      if (inside0 && inside1) {
        if (fabs(dstpel_y0[i * width + j] - dstpel_y1[i * width + j]) >
            OPTICAL_FLOW_DIFF_THRES) {
          disagree_cnt++;
          continue;
        }
      }
      if (inside0 || inside1) {
        used0[(i0)*width + j0] += use0 * (1 - di0) * (1 - dj0);
        used0[(i0 + 1) * width + j0] += use0 * (di0) * (1 - dj0);
        used0[(i0)*width + j0 + 1] += use0 * (1 - di0) * (dj0);
        used0[(i0 + 1) * width + j0 + 1] += use0 * (di0) * (dj0);
        used1[(i1)*width + j1] += use1 * (1 - di1) * (1 - dj1);
        used1[(i1 + 1) * width + j1] += use1 * (di1) * (1 - dj1);
        used1[(i1)*width + j1 + 1] += use1 * (1 - di1) * (dj1);
        used1[(i1 + 1) * width + j1 + 1] += use1 * (di1) * (dj1);
      }  // ignore when both out of bound
      if (use0 && use1)
        refid[i * width + j] = 2;
      else if (use0 && !use1)
        refid[i * width + j] = 0;
      else if (!use0 && use1)
        refid[i * width + j] = 1;
      else
        assert(0);
    }
  }

  // Determine if we want to trust the motion field
  int dis_ref_id;
  if (disagree_cnt > width * height * OPTICAL_FLOW_TRUST_MV_THRES) {
    // Do not trust
    dis_ref_id = ((dstpos <= 0.5) ? 0 : 1);
  } else {
    dis_ref_id = 2;
  }

  // calculate how many time each pixel is referenced
  double totalused0, totalused1;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if (dis_ref_id != 2) {
        // since we do not trust the motion field, no need to select
        refid[i * width + j] = dis_ref_id;
        continue;
      } else if (refid[i * width + j] >= 0)
        continue;

      totalused0 = 0;
      totalused1 = 0;
      pos = dstpos;
      ii0 = i - mf_start[i * mvstr + j].row * dstpos;
      jj0 = j - mf_start[i * mvstr + j].col * dstpos;
      ii1 = i + mf_start[i * mvstr + j].row * (1 - dstpos);
      jj1 = j + mf_start[i * mvstr + j].col * (1 - dstpos);
      i0 = opfl_floor_double_2_int(ii0);
      di0 = ii0 - i0;
      j0 = opfl_floor_double_2_int(jj0);
      dj0 = jj0 - j0;
      i1 = opfl_floor_double_2_int(ii1);
      di1 = ii1 - i1;
      j1 = opfl_floor_double_2_int(jj1);
      dj1 = jj1 - j1;

      totalused0 += used0[(i0)*width + j0] * (1 - di0) * (1 - dj0);
      totalused0 += used0[(i0 + 1) * width + j0] * (di0) * (1 - dj0);
      totalused0 += used0[(i0)*width + j0 + 1] * (1 - di0) * (dj0);
      totalused0 += used0[(i0 + 1) * width + j0 + 1] * (di0) * (dj0);
      totalused1 += used1[(i1)*width + j1] * (1 - di1) * (1 - dj1);
      totalused1 += used1[(i1 + 1) * width + j1] * (di1) * (1 - dj1);
      totalused1 += used1[(i1)*width + j1 + 1] * (1 - di1) * (dj1);
      totalused1 += used1[(i1 + 1) * width + j1 + 1] * (di1) * (dj1);

      // if one ref pixel has been referenced by other pixels more than the
      // other ref by a threshold, then decide there is occlusion
      if (totalused0 < (totalused0 + totalused1) * OPTICAL_FLOW_REF_THRES) {
        refid[i * width + j] = 0;
      } else if (totalused1 <
                 (totalused0 + totalused1) * OPTICAL_FLOW_REF_THRES) {
        refid[i * width + j] = 1;
      } else {
        refid[i * width + j] = dis_ref_id;
      }
    }
  }

  // All refs for each pixel have been decided now.
  // Do mode filter to get rid of outliers
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if (i == 0 || i == height - 1 || j == 0 || j == width - 1) {
        refid_mode[i * width + j] = refid[i * width + j];
        continue;
      }
      refid_mode[i * width + j] =
          ref_mode_filter_3x3(refid + i * width + j, width, dstpos);
    }
  }

  // blend according to the refs selected
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int use0 = -1, use1 = -1;
      pos = dstpos;
      if (refid_mode[i * width + j] == 0) {
        use0 = 1;
        use1 = 0;
        pos = 0;
      } else if (refid_mode[i * width + j] == 1) {
        use0 = 0;
        use1 = 1;
        pos = 1;
      } else if (refid_mode[i * width + j] == 2) {
        use0 = 1;
        use1 = 1;
      } else {
        assert(0);
      }
      ii0 = i - mf_start[i * mvstr + j].row * dstpos;
      jj0 = j - mf_start[i * mvstr + j].col * dstpos;
      ii1 = i + mf_start[i * mvstr + j].row * (1 - dstpos);
      jj1 = j + mf_start[i * mvstr + j].col * (1 - dstpos);
      i0 = opfl_floor_double_2_int(ii0);
      di0 = ii0 - i0;
      j0 = opfl_floor_double_2_int(jj0);
      dj0 = jj0 - j0;
      i1 = opfl_floor_double_2_int(ii1);
      di1 = ii1 - i1;
      j1 = opfl_floor_double_2_int(jj1);
      dj1 = jj1 - j1;

      do_uv =
          (i % 2 == 0) && (j % 2 == 0);  // TODO(bohan) only considering 420 now
      if (do_uv) {
        di0uv = di0 / 2 + 0.5 * ((i0 % 2 + 2) % 2);
        dj0uv = dj0 / 2 + 0.5 * ((j0 % 2 + 2) % 2);
        di1uv = di1 / 2 + 0.5 * ((i1 % 2 + 2) % 2);
        dj1uv = dj1 / 2 + 0.5 * ((j1 % 2 + 2) % 2);
      }

      dstpel_y = 0;
      dstpel_u = 0;
      dstpel_v = 0;
      if (use0) {
        dstpel_y += dstpel_y0[i * width + j] * (1 - pos);
        if (do_uv) {
          dstpel_u +=
              (double)get_sub_pel_uv(
                  src0u + (int)floor(ii0 / 2) * uvstride + (int)floor(jj0 / 2),
                  uvstride, di0uv, dj0uv) *
              (1 - pos);
          dstpel_v +=
              (double)get_sub_pel_uv(
                  src0v + (int)floor(ii0 / 2) * uvstride + (int)floor(jj0 / 2),
                  uvstride, di0uv, dj0uv) *
              (1 - pos);
        }
      }
      if (use1) {
        dstpel_y += dstpel_y1[i * width + j] * pos;
        if (do_uv) {
          dstpel_u +=
              (double)get_sub_pel_uv(
                  src1u + (int)floor(ii1 / 2) * uvstride + (int)floor(jj1 / 2),
                  uvstride, di1uv, dj1uv) *
              pos;
          dstpel_v +=
              (double)get_sub_pel_uv(
                  src1v + (int)floor(ii1 / 2) * uvstride + (int)floor(jj1 / 2),
                  uvstride, di1uv, dj1uv) *
              pos;
        }
      }
      dsty[i * stride + j] = opfl_round_double_2_int(dstpel_y);
      if (do_uv) {
        dstu[i / 2 * uvstride + j / 2] = opfl_round_double_2_int(dstpel_u);
        dstv[i / 2 * uvstride + j / 2] = opfl_round_double_2_int(dstpel_v);
      }
    }
  }

  aom_free(refid);
  aom_free(refid_mode);
  aom_free(used0);
  aom_free(used1);
  aom_free(dstpel_y0);
  aom_free(dstpel_y1);
}

/*
 * Interpolate references according to the motion field
 * Use bilateral filters based on the difference of the
 * pixels as well as confidence of initial mf
 *
 * Input:
 * src0, src1: the reference frames
 * mf_start: the motion field start pointer
 * mvstr: motion field stride
 * dstpos: position of the interpolated frame
 *
 * Output:
 * dst: pointer to the interpolated frame
 */
void warp_optical_flow_bilateral(YV12_BUFFER_CONFIG *src0,
                                 YV12_BUFFER_CONFIG *src1, DB_MV *mf_start,
                                 int mvstr, YV12_BUFFER_CONFIG *dst,
                                 double dstpos) {
  int width = src0->y_width;
  int height = src0->y_height;
  int stride = src0->y_stride;
  int uvstride = src0->uv_stride;

  uint8_t *src0y = src0->y_buffer;
  uint8_t *src1y = src1->y_buffer;
  uint8_t *dsty = dst->y_buffer;
  uint8_t *src0u = src0->u_buffer;
  uint8_t *src0v = src0->v_buffer;
  uint8_t *src1u = src1->u_buffer;
  uint8_t *src1v = src1->v_buffer;
  uint8_t *dstu = dst->u_buffer;
  uint8_t *dstv = dst->v_buffer;

  double ii0, jj0, di0, dj0, di0uv, dj0uv;
  double ii1, jj1, di1, dj1, di1uv, dj1uv;
  int i0, j0;
  int i1, j1;
  double dstpel_y, dstpel_u, dstpel_v;
  double pos;
  int do_uv;

  double *used0 = aom_calloc(height * width, sizeof(double));
  double *used1 = aom_calloc(height * width, sizeof(double));
  // refid: -1=unset; 0=ref0; 1=ref1; 2=both
  double *ref0wts = aom_calloc(height * width, sizeof(double));
  double *ref1wts = aom_calloc(height * width, sizeof(double));
  double *dstpel_y0 = aom_calloc(height * width, sizeof(double));
  double *dstpel_y1 = aom_calloc(height * width, sizeof(double));

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      used0[i * width + j] = 0;
      used1[i * width + j] = 0;
    }
  }
  // first note down which pixels in refs are used for reference
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      ii0 = i - mf_start[i * mvstr + j].row * dstpos;
      jj0 = j - mf_start[i * mvstr + j].col * dstpos;
      ii1 = i + mf_start[i * mvstr + j].row * (1 - dstpos);
      jj1 = j + mf_start[i * mvstr + j].col * (1 - dstpos);
      i0 = opfl_floor_double_2_int(ii0);
      di0 = ii0 - i0;
      j0 = opfl_floor_double_2_int(jj0);
      dj0 = jj0 - j0;
      i1 = opfl_floor_double_2_int(ii1);
      di1 = ii1 - i1;
      j1 = opfl_floor_double_2_int(jj1);
      dj1 = jj1 - j1;

      int inside0 = (i0 >= 0 && i0 < height - 1 && j0 >= 0 && j0 < width - 1);
      int inside1 = (i1 >= 0 && i1 < height - 1 && j1 >= 0 && j1 < width - 1);

      dstpel_y0[i * width + j] =
          (double)get_sub_pel_y(src0y + i0 * stride + j0, stride, di0, dj0);
      dstpel_y1[i * width + j] =
          (double)get_sub_pel_y(src1y + i1 * stride + j1, stride, di1, dj1);

      if (inside0) {
        used0[(i0)*width + j0] += (1 - di0) * (1 - dj0);
        used0[(i0 + 1) * width + j0] += (di0) * (1 - dj0);
        used0[(i0)*width + j0 + 1] += (1 - di0) * (dj0);
        used0[(i0 + 1) * width + j0 + 1] += (di0) * (dj0);
      }
      if (inside1) {
        used1[(i1)*width + j1] += (1 - di1) * (1 - dj1);
        used1[(i1 + 1) * width + j1] += (di1) * (1 - dj1);
        used1[(i1)*width + j1 + 1] += (1 - di1) * (dj1);
        used1[(i1 + 1) * width + j1 + 1] += (di1) * (dj1);
      }  // ignore when both out of bound
    }
  }

  // calculate how many time each pixel is referenced
  double totalused0, totalused1;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      totalused0 = 0;
      totalused1 = 0;
      pos = dstpos;
      ii0 = i - mf_start[i * mvstr + j].row * dstpos;
      jj0 = j - mf_start[i * mvstr + j].col * dstpos;
      ii1 = i + mf_start[i * mvstr + j].row * (1 - dstpos);
      jj1 = j + mf_start[i * mvstr + j].col * (1 - dstpos);
      i0 = opfl_floor_double_2_int(ii0);
      di0 = ii0 - i0;
      j0 = opfl_floor_double_2_int(jj0);
      dj0 = jj0 - j0;
      i1 = opfl_floor_double_2_int(ii1);
      di1 = ii1 - i1;
      j1 = opfl_floor_double_2_int(jj1);
      dj1 = jj1 - j1;

      int inside0 = (i0 >= 0 && i0 < height - 1 && j0 >= 0 && j0 < width - 1);
      int inside1 = (i1 >= 0 && i1 < height - 1 && j1 >= 0 && j1 < width - 1);

      totalused0 += used0[(i0)*width + j0] * (1 - di0) * (1 - dj0);
      totalused0 += used0[(i0 + 1) * width + j0] * (di0) * (1 - dj0);
      totalused0 += used0[(i0)*width + j0 + 1] * (1 - di0) * (dj0);
      totalused0 += used0[(i0 + 1) * width + j0 + 1] * (di0) * (dj0);
      totalused1 += used1[(i1)*width + j1] * (1 - di1) * (1 - dj1);
      totalused1 += used1[(i1 + 1) * width + j1] * (di1) * (1 - dj1);
      totalused1 += used1[(i1)*width + j1 + 1] * (1 - di1) * (dj1);
      totalused1 += used1[(i1 + 1) * width + j1 + 1] * (di1) * (dj1);

      // if referenced more, weight should be lower
      // if refs agrees better, weights should be closer to each other
      double diffWts = (dstpel_y0[i * width + j] - dstpel_y1[i * width + j]);
      diffWts = 0.1 * diffWts * diffWts;
      if (diffWts > 2) diffWts = 2;
      if (inside0 == 0 && inside1 == 0) {
        ref0wts[i * width + j] = 1 - pos;
        ref1wts[i * width + j] = pos;
      } else {
        if (inside0)
          ref0wts[i * width + j] = exp(-diffWts * totalused0);
        else
          ref0wts[i * width + j] = 0;
        if (inside1)
          ref1wts[i * width + j] = exp(-diffWts * totalused1);
        else
          ref1wts[i * width + j] = 0;
      }
      double tempSum = ref0wts[i * width + j] + ref1wts[i * width + j];
      ref0wts[i * width + j] /= tempSum;
      ref1wts[i * width + j] /= tempSum;
    }
  }

  // blend according to the refs selected
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      pos = dstpos;

      ii0 = i - mf_start[i * mvstr + j].row * dstpos;
      jj0 = j - mf_start[i * mvstr + j].col * dstpos;
      ii1 = i + mf_start[i * mvstr + j].row * (1 - dstpos);
      jj1 = j + mf_start[i * mvstr + j].col * (1 - dstpos);
      i0 = opfl_floor_double_2_int(ii0);
      di0 = ii0 - i0;
      j0 = opfl_floor_double_2_int(jj0);
      dj0 = jj0 - j0;
      i1 = opfl_floor_double_2_int(ii1);
      di1 = ii1 - i1;
      j1 = opfl_floor_double_2_int(jj1);
      dj1 = jj1 - j1;

      do_uv =
          (i % 2 == 0) && (j % 2 == 0);  // TODO(bohan) only considering 420 now
      if (do_uv) {
        di0uv = di0 / 2 + 0.5 * ((i0 % 2 + 2) % 2);
        dj0uv = dj0 / 2 + 0.5 * ((j0 % 2 + 2) % 2);
        di1uv = di1 / 2 + 0.5 * ((i1 % 2 + 2) % 2);
        dj1uv = dj1 / 2 + 0.5 * ((j1 % 2 + 2) % 2);
      }

      dstpel_y = 0;
      dstpel_u = 0;
      dstpel_v = 0;

      dstpel_y += dstpel_y0[i * width + j] * ref0wts[i * width + j];
      if (do_uv) {
        dstpel_u +=
            (double)get_sub_pel_uv(
                src0u + (int)floor(ii0 / 2) * uvstride + (int)floor(jj0 / 2),
                uvstride, di0uv, dj0uv) *
            ref0wts[i * width + j];
        dstpel_v +=
            (double)get_sub_pel_uv(
                src0v + (int)floor(ii0 / 2) * uvstride + (int)floor(jj0 / 2),
                uvstride, di0uv, dj0uv) *
            ref0wts[i * width + j];
      }

      dstpel_y += dstpel_y1[i * width + j] * ref1wts[i * width + j];
      if (do_uv) {
        dstpel_u +=
            (double)get_sub_pel_uv(
                src1u + (int)floor(ii1 / 2) * uvstride + (int)floor(jj1 / 2),
                uvstride, di1uv, dj1uv) *
            ref1wts[i * width + j];
        dstpel_v +=
            (double)get_sub_pel_uv(
                src1v + (int)floor(ii1 / 2) * uvstride + (int)floor(jj1 / 2),
                uvstride, di1uv, dj1uv) *
            ref1wts[i * width + j];
      }

      dsty[i * stride + j] = opfl_round_double_2_int(dstpel_y);
      if (do_uv) {
        dstu[i / 2 * uvstride + j / 2] = opfl_round_double_2_int(dstpel_u);
        dstv[i / 2 * uvstride + j / 2] = opfl_round_double_2_int(dstpel_v);
      }
    }
  }
  aom_free(ref0wts);
  aom_free(ref1wts);
  aom_free(used0);
  aom_free(used1);
  aom_free(dstpel_y0);
  aom_free(dstpel_y1);
}

/*
 * Subpel filter for y pixels. Round the motion field to 1/8 precision.
 *
 * Input:
 * src: the source pixel at the integer location.
 * stride: source stride
 * di: subpel location in height
 * dj: subpel location in width
 *
 * Output:
 * interpolated pixel
 */
uint8_t get_sub_pel_y(uint8_t *src, int stride, double di, double dj) {
  int yidx = opfl_round_double_2_int(di * 8);
  int xidx = opfl_round_double_2_int(dj * 8);
  yidx *= 2;
  xidx *= 2;
  if (yidx == 16) {
    yidx = 0;
    src += stride;
  }
  if (xidx == 16) {
    xidx = 0;
    src += 1;
  }
  assert(xidx <= 14 && xidx >= 0);
  assert(yidx <= 14 && yidx >= 0);
  int y[8];
  for (int i = -3; i < 5; i++) {
    y[i + 3] = 0;
    for (int j = -3; j < 5; j++) {
      y[i + 3] += src[i * stride + j] * optical_flow_warp_filter[xidx][j + 3];
    }
    y[i + 3] = (y[i + 3] + (1 << 6)) >> 7;
  }
  int x = 0;
  for (int i = 0; i < 8; i++) {
    x += y[i] * optical_flow_warp_filter[yidx][i];
  }
  x = (x + (1 << 6)) >> 7;
  if (x > 255)
    x = 255;
  else if (x < 0)
    x = 0;
  return (uint8_t)x;
}

/*
 * Subpel filter for u/v pixels. Round the motion field to 1/16 precision.
 *
 * Input:
 * src: the source pixel at the integer location.
 * stride: source stride
 * di: subpel location in height
 * dj: subpel location in width
 *
 * Output:
 * interpolated pixel
 */
uint8_t get_sub_pel_uv(uint8_t *src, int stride, double di, double dj) {
  // TODO(bohan) now only care about YUV 420
  int yidx = opfl_round_double_2_int(di * 16);
  int xidx = opfl_round_double_2_int(dj * 16);
  if (yidx == 16) {
    yidx = 0;
    src += stride;
  }
  if (xidx == 16) {
    xidx = 0;
    src += 1;
  }
  assert(xidx <= 15 && xidx >= 0);
  assert(yidx <= 15 && yidx >= 0);

  int y[8];
  for (int i = -3; i < 5; i++) {
    y[i + 3] = 0;
    for (int j = -3; j < 5; j++) {
      y[i + 3] += src[i * stride + j] * optical_flow_warp_filter[xidx][j + 3];
    }
    y[i + 3] = (y[i + 3] + (1 << 6)) >> 7;
  }
  int x = 0;
  for (int i = 0; i < 8; i++) {
    x += y[i] * optical_flow_warp_filter[yidx][i];
  }
  x = (x + (1 << 6)) >> 7;
  if (x > 255)
    x = 255;
  else if (x < 0)
    x = 0;
  return (uint8_t)x;
}

/*
 * Blend function which calls the real blend methods.
 * Kept as caller where we may make high level changes or pre-process
 */
void interp_optical_flow(YV12_BUFFER_CONFIG *ref0, YV12_BUFFER_CONFIG *ref1,
                         DB_MV *mf, YV12_BUFFER_CONFIG *dst, double dst_pos,
                         OPFL_BLK_INFO blk_info) {
  // blend here
  int mvstr = blk_info.blk_width + 2 * AVG_MF_BORDER;
  DB_MV *mf_start = mf + AVG_MF_BORDER * mvstr + AVG_MF_BORDER;

  warp_optical_flow(ref0, ref1, mf_start, mvstr, dst, dst_pos,
                    OPFL_BLEND_METHOD_USED, blk_info);
  return;
}

/*
 * Use the initial motion vectors to create the initial motion field.
 *
 * Input:
 * mv_left: motion vector points from the current frame to ref0
 * mv_right: motion vector points from the current frame to ref1
 * width, height: frame/block width and height
 * mvwid, mvhgt: width and height of the mv at some pyramid scale
 * mfstr: stride of the motion field buffer
 * dstpos: relative location of the current block
 *
 * Output:
 * mf: pointer to the created motion field
 */
void create_motion_field(int_mv *mv_left, int_mv *mv_right, DB_MV *mf,
#if OPFL_INIT_WT
                         double *mv_wts,
#endif
                         int width, int height, int mvwid, int mvhgt, int mfstr,
                         double dstpos) {
  // since the motion field is just used as initialization for now,
  // just simply use the summation of the two
  // TODO(bohan): need to change the function to work for MAX_OPFL_LEVEL > 3
  int stride = mfstr;
  DB_MV *mf_start = mf + AVG_MF_BORDER * stride + AVG_MF_BORDER;
  int idx;
  int blksize = mvwid / (width / 4);
  assert(blksize == mvhgt / (height / 4));
  double tempr, tempc;
  double mvscale = 4 / blksize;
#if OPFL_INIT_WT
  double *wts_start = mv_wts + AVG_MF_BORDER * stride + AVG_MF_BORDER;
  double tempWts;
#endif
  for (int h = 0; h < height / 4; h++) {
    for (int w = 0; w < width / 4; w++) {
      // mv_left, mv_right are based on 4x4 block
      idx = h * width / 4 + w;
      if (mv_left[idx].as_int == INVALID_MV &&
          mv_right[idx].as_int == INVALID_MV) {
        tempr = 0;
        tempc = 0;
#if OPFL_INIT_WT
        tempWts = 0;
#endif
      } else if (mv_left[idx].as_int == INVALID_MV) {
        tempr =
            (double)(mv_right[idx].as_mv.row) / 8.0 / mvscale / (1 - dstpos);
        tempc =
            (double)(mv_right[idx].as_mv.col) / 8.0 / mvscale / (1 - dstpos);
#if OPFL_INIT_WT
        tempWts = 0.0;
#endif
      } else if (mv_right[idx].as_int == INVALID_MV) {
        tempr = (double)(-mv_left[idx].as_mv.row) / 8.0 / mvscale / dstpos;
        tempc = (double)(-mv_left[idx].as_mv.col) / 8.0 / mvscale / dstpos;
#if OPFL_INIT_WT
        tempWts = 0.0;
#endif
      } else {
        tempr = (double)(-mv_left[idx].as_mv.row + mv_right[idx].as_mv.row) /
                8.0 / mvscale;
        tempc = (double)(-mv_left[idx].as_mv.col + mv_right[idx].as_mv.col) /
                8.0 / mvscale;
#if OPFL_INIT_WT
        tempWts = 0.0;
#endif
      }
      for (int i = 0; i < blksize; i++) {
        for (int j = 0; j < blksize; j++) {
          mf_start[(h * blksize + i) * mfstr + w * blksize + j].row = tempr;
          mf_start[(h * blksize + i) * mfstr + w * blksize + j].col = tempc;
#if OPFL_INIT_WT
          double i0, i1, j0, j1;
          i0 = (double)(i + h * blksize) - (dstpos)*tempr;
          i1 = (double)(i + h * blksize) + (1 - dstpos) * tempr;
          j0 = (double)(j + w * blksize) - (dstpos)*tempc;
          j1 = (double)(j + w * blksize) + (1 - dstpos) * tempc;
          int is_out =
              (i0 < 0 || i0 > height - 1 || i1 < 0 || i1 > height - 1 ||
               j0 < 0 || j0 > width - 1 || j1 < 0 || j1 > width - 1);
          if (is_out) {
            wts_start[(h * blksize + i) * mfstr + w * blksize + j] = 0;
          } else {
            wts_start[(h * blksize + i) * mfstr + w * blksize + j] = tempWts;
          }
#endif
        }
      }
    }
  }
  // Pad the motion field border
  // pad_motion_field_border(mf_start, mvwid, mvhgt, mfstr);
  return;
}

/*
 * Fill the initial motion field if there are holes
 *
 * Input:
 * mv_left: motion vector points from the current frame to ref0
 * mv_right: motion vector points from the current frame to ref1
 * width, height: frame/block width and height
 * mvwid, mvhgt: width and height of the mv at some pyramid scale
 * mfstr: stride of the motion field buffer
 * dstpos: relative location of the current block
 *
 * Output:
 * mf: pointer to the created motion field
 */
void fill_create_motion_field(int_mv *mv_left, int_mv *mv_right, DB_MV *mf,
                              int width, int height, int mvwid, int mvhgt,
                              int mfstr) {
  int stride = mfstr;
  DB_MV *mf_start = mf + AVG_MF_BORDER * stride + AVG_MF_BORDER;
  int idx;
  int blksize = mvwid / (width / 4);
  assert(blksize == mvhgt / (height / 4));

  int invalid_cnt = 0;
  DB_MV *tempmv = aom_calloc(mvwid * mvhgt, sizeof(DB_MV));
  // isValid: 0: not valid; 1: already valid; -1:ready for next round
  int *isValid = aom_calloc(height * width / 4 / 4, sizeof(int));
  for (int i = 0; i < mvhgt; i++) {
    for (int j = 0; j < mvwid; j++) {
      tempmv[i * mvwid + j].col = mf_start[i * mfstr + j].col;
      tempmv[i * mvwid + j].row = mf_start[i * mfstr + j].row;
    }
  }

  for (int h = 0; h < height / 4; h++) {
    for (int w = 0; w < width / 4; w++) {
      idx = h * width / 4 + w;
      if (mv_left[idx].as_int == INVALID_MV &&
          mv_right[idx].as_int == INVALID_MV) {
        invalid_cnt++;
        isValid[idx] = 0;
      } else {
        isValid[idx] = 1;
      }
    }
  }
  DB_MV avg;
  int avgcnt;
  while (invalid_cnt > 0 && invalid_cnt != width * height / 4 / 4) {
    for (int h = 0; h < height / 4; h++) {
      for (int w = 0; w < width / 4; w++) {
        idx = h * width / 4 + w;
        if (isValid[idx] == 0) {
          avgcnt = 0;
          avg.col = 0;
          avg.row = 0;
          for (int i = -1; i < 2; i++) {
            for (int j = -1; j < 2; j++) {
              if (i + h >= 0 && i + h < height / 4 && j + w >= 0 &&
                  j + w < width / 4) {
                if (isValid[idx + i * width / 4 + j] > 0) {
                  avg.col +=
                      mf_start[(h + i) * blksize * mfstr + (j + w) * blksize]
                          .col;
                  avg.row +=
                      mf_start[(h + i) * blksize * mfstr + (j + w) * blksize]
                          .row;
                  avgcnt++;
                }
              }
            }
          }
          if (avgcnt != 0) {
            for (int i = 0; i < blksize; i++) {
              for (int j = 0; j < blksize; j++) {
                tempmv[(h * blksize + i) * mvwid + w * blksize + j].row =
                    avg.row / avgcnt;
                tempmv[(h * blksize + i) * mvwid + w * blksize + j].col =
                    avg.col / avgcnt;
              }
            }
            isValid[idx] = -1;
          }
        }
      }
    }
    invalid_cnt = 0;
    for (int h = 0; h < height / 4; h++) {
      for (int w = 0; w < width / 4; w++) {
        idx = h * width / 4 + w;
        if (isValid[idx] == 0) {
          invalid_cnt++;
        } else if (isValid[idx] < 0) {
          isValid[idx] = 1;
          for (int i = 0; i < blksize; i++) {
            for (int j = 0; j < blksize; j++) {
              mf_start[(h * blksize + i) * mfstr + w * blksize + j].row =
                  tempmv[(h * blksize + i) * mvwid + w * blksize + j].row;
              mf_start[(h * blksize + i) * mfstr + w * blksize + j].col =
                  tempmv[(h * blksize + i) * mvwid + w * blksize + j].col;
            }
          }
        }
      }
    }
  }
  aom_free(tempmv);
  aom_free(isValid);
  // Pad the motion field border
  pad_motion_field_border(mf_start, mvwid, mvhgt, mfstr);
}

void opfl_fill_mv(int_mv *pmv, int width, int height) {
  int invalid_cnt = 0;
  int_mv *tempmv = aom_calloc(width * height, sizeof(int_mv));
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if (pmv[i * width + j].as_int == INVALID_MV) {
        invalid_cnt++;
      }
      tempmv[i * width + j].as_int = INVALID_MV;
    }
  }
  int_mv avg;
  int avgcnt;
  while (invalid_cnt > 0 && invalid_cnt != width * height) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        if (pmv[i * width + j].as_int == INVALID_MV) {
          avgcnt = 0;
          avg.as_int = 0;
          for (int h = -1; h < 2; h++) {
            for (int w = -1; w < 2; w++) {
              if (i + h >= 0 && i + h < height && j + w >= 0 && j + w < width) {
                if (pmv[(i + h) * width + j + w].as_int != INVALID_MV) {
                  avg.as_mv.col += pmv[(i + h) * width + j + w].as_mv.col;
                  avg.as_mv.row += pmv[(i + h) * width + j + w].as_mv.row;
                  avgcnt++;
                }
              }
            }
          }
          if (avgcnt != 0) {
            tempmv[i * width + j].as_mv.col =
                opfl_round_double_2_int((double)avg.as_mv.col / avgcnt);
            tempmv[i * width + j].as_mv.row =
                opfl_round_double_2_int((double)avg.as_mv.row / avgcnt);
          }
        }
      }
    }  // for every mv
    invalid_cnt = 0;
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        if (pmv[i * width + j].as_int == INVALID_MV &&
            tempmv[i * width + j].as_int != INVALID_MV) {
          pmv[i * width + j].as_int = tempmv[i * width + j].as_int;
        } else if (pmv[i * width + j].as_int == INVALID_MV) {
          invalid_cnt++;
        }
      }
    }
  }
  aom_free(tempmv);
}
/*
 * Upscale the motion field by 2.
 * Currently simply copy the nearest mv
 *
 * Input:
 * src: the source motion field
 * srcw, srch, srcs: source width, height, stride
 * dsts: destination mf stride
 *
 * Output;
 * dst: pointer to the upscaled mf buffer
 */
void upscale_mv_by_2(DB_MV *src, int srcw, int srch, int srcs, DB_MV *dst,
                     int dsts) {
  for (int i = 0; i < srch; i++) {
    for (int j = 0; j < srcw; j++) {
      for (int y = 0; y < 2; y++) {
        for (int x = 0; x < 2; x++) {
          dst[(i * 2 + y) * dsts + j * 2 + x].row = src[i * srcs + j].row * 2;
          dst[(i * 2 + y) * dsts + j * 2 + x].col = src[i * srcs + j].col * 2;
        }
      }
    }
  }
  pad_motion_field_border(dst, srcw * 2, srch * 2, dsts);
}

/*
 * Pad the motion field border to prepare for median filter
 */
void pad_motion_field_border(DB_MV *mf_start, int width, int height,
                             int stride) {
  assert(stride == width + 2 * AVG_MF_BORDER);
  // upper
  for (int i = -AVG_MF_BORDER; i < 0; i++) {
    memcpy(mf_start + i * stride, mf_start, sizeof(DB_MV) * width);
  }
  // lower
  for (int i = height; i < height + AVG_MF_BORDER; i++) {
    memcpy(mf_start + i * stride, mf_start + (height - 1) * stride,
           sizeof(DB_MV) * width);
  }
  // left
  for (int i = -AVG_MF_BORDER; i < height + AVG_MF_BORDER; i++) {
    for (int j = -AVG_MF_BORDER; j < 0; j++) {
      mf_start[i * stride + j] = mf_start[i * stride];
    }
  }
  // right
  for (int i = -AVG_MF_BORDER; i < height + AVG_MF_BORDER; i++) {
    for (int j = width; j < width + AVG_MF_BORDER; j++) {
      mf_start[i * stride + j] = mf_start[i * stride + width - 1];
    }
  }
}

/*
 * Median filter double arrays iteratively
 */
double iter_median_double(double *x, double *left, double *right, int length,
                          int mididx) {
  int pivot = length / 2;
  int ll = 0, rl = 0;
  for (int i = 0; i < length; i++) {
    if (i == pivot) continue;
    if (x[i] <= x[pivot]) {
      left[ll] = x[i];
      ll++;
    } else {
      right[rl] = x[i];
      rl++;
    }
  }
  if (mididx == ll)
    return x[pivot];
  else if (mididx < ll)
    return iter_median_double(left, x, right, ll, mididx);
  else
    return iter_median_double(right, left, x, rl, mididx - ll - 1);
}

/*
 * Do mode filter to the reference selections
 */
int ref_mode_filter_3x3(int *center, int stride, double dstpos) {
  int ref_id_count[3] = { 0 };
  for (int i = -1; i < 2; i++) {
    for (int j = -1; j < 2; j++) {
      assert(center[i * stride + j] >= 0);
      ref_id_count[center[i * stride + j]]++;
    }
  }
  if (ref_id_count[2] >= ref_id_count[1] &&
      ref_id_count[2] >= ref_id_count[0]) {
    return 2;
  } else if (ref_id_count[1] == ref_id_count[0]) {
    return ((dstpos <= 0.5) ? 0 : 1);
  } else {
    return ((ref_id_count[0] > ref_id_count[1]) ? 0 : 1);
  }
}

/*
 * Write YUV for debug purpose
 */
int write_image_opfl(const YV12_BUFFER_CONFIG *const ref_buf, char *file_name) {
  int h;
  FILE *f_ref = NULL;

  if (ref_buf == NULL) {
    printf("Frame data buffer is NULL.\n");
    return AOM_CODEC_MEM_ERROR;
  }
  if ((f_ref = fopen(file_name, "ab")) == NULL) {
    printf("Unable to open file %s to write.\n", file_name);
    return AOM_CODEC_MEM_ERROR;
  }
  // --- Y ---
  for (h = 0; h < ref_buf->y_height; ++h) {
    fwrite(&ref_buf->y_buffer[h * ref_buf->y_stride], 1, ref_buf->y_width,
           f_ref);
  }
  // --- U ---
  for (h = 0; h < (ref_buf->uv_height); ++h) {
    fwrite(&ref_buf->u_buffer[h * ref_buf->uv_stride], 1, ref_buf->uv_width,
           f_ref);
  }
  // --- V ---
  for (h = 0; h < (ref_buf->uv_height); ++h) {
    fwrite(&ref_buf->v_buffer[h * ref_buf->uv_stride], 1, ref_buf->uv_width,
           f_ref);
  }
  fclose(f_ref);
  return AOM_CODEC_OK;
}

/*
 * Extend pixel planes. This is an exact copy of the original extend_plane
 * function.
 */
void extend_plane_opfl(uint8_t *const src, int src_stride, int width,
                       int height, int extend_top, int extend_left,
                       int extend_bottom, int extend_right) {
  int i;
  const int linesize = extend_left + extend_right + width;

  /* copy the left and right most columns out */
  uint8_t *src_ptr1 = src;
  uint8_t *src_ptr2 = src + width - 1;
  uint8_t *dst_ptr1 = src - extend_left;
  uint8_t *dst_ptr2 = src + width;

  for (i = 0; i < height; ++i) {
    memset(dst_ptr1, src_ptr1[0], extend_left);
    memset(dst_ptr2, src_ptr2[0], extend_right);
    src_ptr1 += src_stride;
    src_ptr2 += src_stride;
    dst_ptr1 += src_stride;
    dst_ptr2 += src_stride;
  }

  /* Now copy the top and bottom lines into each line of the respective
   * borders
   */
  src_ptr1 = src - extend_left;
  src_ptr2 = src + src_stride * (height - 1) - extend_left;
  dst_ptr1 = src + src_stride * -extend_top - extend_left;
  dst_ptr2 = src + src_stride * height - extend_left;

  for (i = 0; i < extend_top; ++i) {
    memcpy(dst_ptr1, src_ptr1, linesize);
    dst_ptr1 += src_stride;
  }

  for (i = 0; i < extend_bottom; ++i) {
    memcpy(dst_ptr2, src_ptr2, linesize);
    dst_ptr2 += src_stride;
  }
}

int opfl_round_double_2_int(double x) {
  if (x >= 0) {
    return (int)(x + 0.5);
  } else {
    return (int)(x - 0.5);
  }
}
int opfl_floor_double_2_int(double x) {
  if (x >= 0) {
    return (int)x;
  } else {
    return (int)x - 1;
  }
}

#endif  // CONFIG_OPFL
