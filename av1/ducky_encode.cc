/*
 * Copyright (c) 2022, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <memory>
#include <numeric>
#include <vector>

#include "config/aom_config.h"

#include "aom/aom_encoder.h"

#include "av1/av1_cx_iface.h"
#include "av1/av1_iface_common.h"
#include "av1/encoder/encoder.h"
#include "av1/encoder/ethread.h"
#include "av1/encoder/firstpass.h"
#include "av1/ducky_encode.h"

#include "common/tools_common.h"

namespace aom {
struct EncoderResource {
  FILE *in_file;
  STATS_BUFFER_CTX *stats_buf_ctx;
  FIRSTPASS_STATS *stats_buffer;
  aom_image_t img;
  AV1_PRIMARY *ppi;
  int lookahead_push_count;
  int encode_frame_count;  // Use in second pass only
};

class DuckyEncode::EncodeImpl {
 public:
  VideoInfo video_info;
  int g_usage;
  enum aom_rc_mode rc_end_usage;
  aom_rational64_t timestamp_ratio;
  std::vector<FIRSTPASS_STATS> stats_list;
  EncoderResource enc_resource;
};

DuckyEncode::DuckyEncode(const VideoInfo &video_info) {
  impl_ptr_ = std::unique_ptr<EncodeImpl>(new EncodeImpl());
  impl_ptr_->video_info = video_info;
  impl_ptr_->g_usage = GOOD;
  impl_ptr_->rc_end_usage = AOM_Q;
  // TODO(angiebird): Set timestamp_ratio properly
  // timestamp_ratio.den = cfg->g_timebase.den;
  // timestamp_ratio.num = (int64_t)cfg->g_timebase.num * TICKS_PER_SEC;
  impl_ptr_->timestamp_ratio = { 1, 1 };
  // TODO(angiebird): How to set ptsvol and duration?
}

DuckyEncode::~DuckyEncode() {}

static AV1EncoderConfig GetEncoderConfig(const VideoInfo &video_info,
                                         int g_usage, aom_enc_pass pass) {
  const aom_codec_iface *codec = aom_codec_av1_cx();
  aom_codec_enc_cfg_t cfg;
  aom_codec_enc_config_default(codec, &cfg, g_usage);
  cfg.g_w = video_info.frame_width;
  cfg.g_h = video_info.frame_height;
  cfg.g_pass = pass;
  // g_timebase is the inverse of frame_rate
  cfg.g_timebase.num = video_info.frame_rate.den;
  cfg.g_timebase.den = video_info.frame_rate.num;
  AV1EncoderConfig oxcf = av1_get_encoder_config(&cfg);
  // TODO(angiebird): Why didn't we init use_highbitdepth in
  // av1_get_encoder_config()?
  oxcf.use_highbitdepth = 0;
  // TODO(angiebird): Let user set the desired speed.
  oxcf.speed = 3;

  // TODO(jingning): Change this to 35 when the baseline rate control
  // logic is in place.
  // Force maximum look ahead buffer to be 19. This will disable the use
  // of maximum 32 GOP length.
  oxcf.gf_cfg.lag_in_frames = 19;

  return oxcf;
}

static STATS_BUFFER_CTX *CreateStatsBufferCtx(int frame_count,
                                              FIRSTPASS_STATS **stats_buffer) {
  STATS_BUFFER_CTX *stats_buf_ctx = new STATS_BUFFER_CTX;
  // +2 is for total_stats and total_left_stats
  *stats_buffer = new FIRSTPASS_STATS[frame_count + 2];
  stats_buf_ctx->stats_in_start = *stats_buffer;
  stats_buf_ctx->stats_in_end = stats_buf_ctx->stats_in_start;
  stats_buf_ctx->stats_in_buf_end = stats_buf_ctx->stats_in_start + frame_count;
  stats_buf_ctx->total_stats = stats_buf_ctx->stats_in_buf_end;
  stats_buf_ctx->total_left_stats =
      stats_buf_ctx->stats_in_start + frame_count + 1;
  av1_twopass_zero_stats(stats_buf_ctx->total_left_stats);
  av1_twopass_zero_stats(stats_buf_ctx->total_stats);
  return stats_buf_ctx;
}

static void DestroyStatsBufferCtx(STATS_BUFFER_CTX **stats_buf_context,
                                  FIRSTPASS_STATS **stats_buffer) {
  (*stats_buf_context)->stats_in_start = nullptr;
  (*stats_buf_context)->stats_in_end = nullptr;
  (*stats_buf_context)->stats_in_buf_end = nullptr;
  (*stats_buf_context)->total_stats = nullptr;
  (*stats_buf_context)->total_left_stats = nullptr;
  delete *stats_buf_context;
  *stats_buf_context = nullptr;
  delete[](*stats_buffer);
  *stats_buffer = nullptr;
}

static FIRSTPASS_STATS ComputeTotalStats(
    const std::vector<FIRSTPASS_STATS> &stats_list) {
  FIRSTPASS_STATS total_stats;
  for (size_t i = 0; i < stats_list.size(); ++i) {
    av1_accumulate_stats(&total_stats, &stats_list[i]);
  }
  return total_stats;
}

static EncoderResource InitEncoder(
    const VideoInfo &video_info, int g_usage, enum aom_rc_mode rc_end_usage,
    aom_enc_pass pass, const std::vector<FIRSTPASS_STATS> *stats_list) {
  EncoderResource enc_resource = {};
  enc_resource.in_file = fopen(video_info.file_path.c_str(), "r");
  enc_resource.lookahead_push_count = 0;
  aom_img_alloc(&enc_resource.img, video_info.img_fmt, video_info.frame_width,
                video_info.frame_height, /*align=*/1);
  AV1EncoderConfig oxcf = GetEncoderConfig(video_info, g_usage, pass);
  oxcf.dec_model_cfg.decoder_model_info_present_flag = 0;
  oxcf.dec_model_cfg.display_model_info_present_flag = 0;
  av1_initialize_enc(g_usage, rc_end_usage);
  AV1_PRIMARY *ppi =
      av1_create_primary_compressor(nullptr,
                                    /*num_lap_buffers=*/0, &oxcf);
  enc_resource.ppi = ppi;

  assert(ppi != nullptr);
  // Turn off ppi->b_calculate_psnr to avoid calling generate_psnr_packet() in
  // av1_post_encode_updates().
  // TODO(angiebird): Modify generate_psnr_packet() to handle the case that
  // cpi->ppi->output_pkt_list = nullptr.
  ppi->b_calculate_psnr = 0;

  aom_codec_err_t res = AOM_CODEC_OK;
  (void)res;
  enc_resource.stats_buf_ctx =
      CreateStatsBufferCtx(video_info.frame_count, &enc_resource.stats_buffer);
  if (pass == AOM_RC_SECOND_PASS) {
    assert(stats_list != nullptr);
    std::copy(stats_list->begin(), stats_list->end(),
              enc_resource.stats_buffer);
    *enc_resource.stats_buf_ctx->total_stats = ComputeTotalStats(*stats_list);
    oxcf.twopass_stats_in.buf = enc_resource.stats_buffer;
    // We need +1 here because av1 encoder assumes
    // oxcf.twopass_stats_in.buf[video_info.frame_count] has the total_stats
    oxcf.twopass_stats_in.sz =
        (video_info.frame_count + 1) * sizeof(enc_resource.stats_buffer[0]);
  } else {
    assert(pass == AOM_RC_FIRST_PASS);
    // We don't use stats_list for AOM_RC_FIRST_PASS.
    assert(stats_list == nullptr);
  }
  ppi->twopass.stats_buf_ctx = enc_resource.stats_buf_ctx;
  BufferPool *buffer_pool = nullptr;
  res = av1_create_context_and_bufferpool(ppi, &ppi->cpi, &buffer_pool, &oxcf,
                                          ENCODE_STAGE, -1);
  // TODO(angiebird): Why didn't we set initial_dimensions in
  // av1_create_compressor()?
  ppi->cpi->initial_dimensions.width = oxcf.frm_dim_cfg.width;
  ppi->cpi->initial_dimensions.height = oxcf.frm_dim_cfg.height;
  // use_ducky_encode is the flag we use to change AV1 behavior
  // slightly based on DuckyEncode's need. We should minimize this kind of
  // change unless it's necessary.
  ppi->cpi->use_ducky_encode = 1;
  assert(res == AOM_CODEC_OK);
  assert(ppi->cpi != nullptr);
  assert(buffer_pool != nullptr);
  const AV1_COMP *cpi = ppi->cpi;
  SequenceHeader *seq_params = ppi->cpi->common.seq_params;

  ppi->seq_params_locked = 1;
  assert(ppi->lookahead == nullptr);

  int lag_in_frames = cpi->oxcf.gf_cfg.lag_in_frames;
  ppi->lookahead = av1_lookahead_init(
      cpi->oxcf.frm_dim_cfg.width, cpi->oxcf.frm_dim_cfg.height,
      seq_params->subsampling_x, seq_params->subsampling_y,
      seq_params->use_highbitdepth, lag_in_frames, cpi->oxcf.border_in_pixels,
      cpi->common.features.byte_alignment,
      /*num_lap_buffers=*/0, /*is_all_intra=*/0,
      cpi->oxcf.tool_cfg.enable_global_motion);
  assert(ppi->lookahead != nullptr);
  return enc_resource;
}

static void FreeEncoder(EncoderResource *enc_resource) {
  fclose(enc_resource->in_file);
  enc_resource->in_file = nullptr;
  aom_img_free(&enc_resource->img);
  DestroyStatsBufferCtx(&enc_resource->stats_buf_ctx,
                        &enc_resource->stats_buffer);
  BufferPool *buffer_pool = enc_resource->ppi->cpi->common.buffer_pool;
  av1_destroy_context_and_bufferpool(enc_resource->ppi->cpi, &buffer_pool);
  av1_remove_primary_compressor(enc_resource->ppi);
  enc_resource->ppi = nullptr;
}

std::vector<FIRSTPASS_STATS> DuckyEncode::ComputeFirstPassStats() {
  aom_enc_pass pass = AOM_RC_FIRST_PASS;
  EncoderResource enc_resource =
      InitEncoder(impl_ptr_->video_info, impl_ptr_->g_usage,
                  impl_ptr_->rc_end_usage, pass, nullptr);
  AV1_PRIMARY *ppi = enc_resource.ppi;
  struct lookahead_ctx *lookahead = ppi->lookahead;
  int frame_count = impl_ptr_->video_info.frame_count;
  FILE *in_file = enc_resource.in_file;
  aom_rational64_t timestamp_ratio = impl_ptr_->timestamp_ratio;
  // TODO(angiebird): Ideally, ComputeFirstPassStats() doesn't output
  // bitstream. Do we need bitstream buffer here?
  std::vector<uint8_t> buf(1000);
  std::vector<FIRSTPASS_STATS> stats_list;
  for (int i = 0; i < frame_count; ++i) {
    assert(!av1_lookahead_full(lookahead));
    if (aom_img_read(&enc_resource.img, in_file)) {
      // TODO(angiebird): Set ts_start/ts_end properly
      int64_t ts_start = enc_resource.lookahead_push_count;
      int64_t ts_end = ts_start + 1;
      YV12_BUFFER_CONFIG sd;
      image2yuvconfig(&enc_resource.img, &sd);
      av1_lookahead_push(lookahead, &sd, ts_start, ts_end,
                         /*use_highbitdepth=*/0, /*flags=*/0);
      ++enc_resource.lookahead_push_count;
      AV1_COMP_DATA cpi_data = {};
      cpi_data.cx_data = buf.data();
      cpi_data.cx_data_sz = buf.size();
      cpi_data.frame_size = 0;
      cpi_data.flush = 1;  // Makes av1_get_compressed_data process a frame
      cpi_data.ts_frame_start = ts_start;
      cpi_data.ts_frame_end = ts_end;
      cpi_data.pop_lookahead = 1;
      cpi_data.timestamp_ratio = &timestamp_ratio;
      // av1_get_compressed_data only generates first pass stats not
      // compresses data
      int res = av1_get_compressed_data(ppi->cpi, &cpi_data);
      (void)res;
      assert(res == static_cast<int>(AOM_CODEC_OK));
      stats_list.push_back(*(ppi->twopass.stats_buf_ctx->stats_in_end - 1));
    }
  }
  av1_end_first_pass(ppi->cpi);

  FreeEncoder(&enc_resource);
  return stats_list;
}

void DuckyEncode::StartEncode(const std::vector<FIRSTPASS_STATS> &stats_list) {
  aom_enc_pass pass = AOM_RC_SECOND_PASS;
  impl_ptr_->stats_list = stats_list;
  impl_ptr_->enc_resource =
      InitEncoder(impl_ptr_->video_info, impl_ptr_->g_usage,
                  impl_ptr_->rc_end_usage, pass, &stats_list);
}

static void DuckyEncodeInfoSetEncodeFrameDecision(
    DuckyEncodeInfo *ducky_encode_info, const EncodeFrameDecision &decision) {
  DuckyEncodeFrameInfo *frame_info = &ducky_encode_info->frame_info;
  frame_info->mode = static_cast<DUCKY_ENCODE_FRAME_MODE>(decision.mode);
  frame_info->q_index = decision.parameters.q_index;
  frame_info->rdmult = decision.parameters.rdmult;
}

static void DuckyEncodeInfoGetEncodeFrameResult(
    const DuckyEncodeInfo *ducky_encode_info, EncodeFrameResult *result) {
  const DuckyEncodeFrameResult &frame_result = ducky_encode_info->frame_result;
  result->q_index = frame_result.q_index;
  result->rdmult = frame_result.rdmult;
  result->rate = frame_result.rate;
  result->dist = frame_result.dist;
  result->psnr = frame_result.psnr;
}

EncodeFrameResult DuckyEncode::EncodeFrame(
    const EncodeFrameDecision &decision) {
  EncodeFrameResult encode_frame_result = {};
  const VideoInfo &video_info = impl_ptr_->video_info;
  // Set bitstream_buf size to a conservatve upperbound.
  // TODO(angiebird): Set bitstream_buf's size optimally.
  encode_frame_result.bitstream_buf = std::vector<uint8_t>(
      video_info.frame_width * video_info.frame_height * 3 * 8, 0);
  AV1_PRIMARY *ppi = impl_ptr_->enc_resource.ppi;
  aom_image_t *img = &impl_ptr_->enc_resource.img;
  AV1_COMP *const cpi = ppi->cpi;
  FILE *in_file = impl_ptr_->enc_resource.in_file;
  struct lookahead_ctx *lookahead = ppi->lookahead;
  while (!av1_lookahead_full(lookahead)) {
    if (aom_img_read(img, in_file)) {
      YV12_BUFFER_CONFIG sd;
      image2yuvconfig(img, &sd);
      int64_t ts_start = impl_ptr_->enc_resource.lookahead_push_count;
      int64_t ts_end = ts_start + 1;
      av1_lookahead_push(lookahead, &sd, ts_start, ts_end,
                         /*use_highbitdepth=*/0, /*flags=*/0);
      ++impl_ptr_->enc_resource.lookahead_push_count;
    } else {
      break;
    }
  }
  AV1_COMP_DATA cpi_data = {};
  cpi_data.cx_data = encode_frame_result.bitstream_buf.data();
  cpi_data.cx_data_sz = encode_frame_result.bitstream_buf.size();
  cpi_data.frame_size = 0;
  cpi_data.flush = 1;
  // ts_frame_start and ts_frame_end are not as important since we are focusing
  // on q mode
  cpi_data.ts_frame_start = impl_ptr_->enc_resource.encode_frame_count;
  cpi_data.ts_frame_end = cpi_data.ts_frame_start + 1;
  cpi_data.pop_lookahead = 1;
  cpi_data.timestamp_ratio = &impl_ptr_->timestamp_ratio;
  ++impl_ptr_->enc_resource.encode_frame_count;

  av1_compute_num_workers_for_mt(cpi);
  av1_init_frame_mt(ppi, cpi);

  DuckyEncodeInfoSetEncodeFrameDecision(&cpi->ducky_encode_info, decision);
  const int status = av1_get_compressed_data(cpi, &cpi_data);
  (void)status;
  assert(status == static_cast<int>(AOM_CODEC_OK));
  encode_frame_result.bitstream_buf.resize(cpi_data.frame_size);
  DuckyEncodeInfoGetEncodeFrameResult(&cpi->ducky_encode_info,
                                      &encode_frame_result);
  av1_post_encode_updates(cpi, &cpi_data);
  if (cpi->common.show_frame) {
    // decrement frames_left counter
    ppi->frames_left = AOMMAX(0, ppi->frames_left - 1);
  }
  return encode_frame_result;
}

void DuckyEncode::EndEncode() { FreeEncoder(&impl_ptr_->enc_resource); }
}  // namespace aom