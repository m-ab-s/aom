/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <csetjmp>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "gtest/gtest.h"

#include "config/aom_config.h"
#include "config/av1_rtcd.h"

#include "aom/aom_image.h"
#include "aom_dsp/aom_dsp_common.h"
#include "aom_mem/aom_mem.h"
#include "aom_ports/mem.h"
#include "aom_scale/yv12config.h"
#include "av1/common/av1_common_int.h"
#include "av1/common/blockd.h"
#include "av1/common/common_data.h"
#include "av1/common/enums.h"
#include "av1/common/reconintra.h"
#include "av1/encoder/allintra_vis.h"
#include "test/acm_random.h"

namespace {

TEST(EncodeInternal, Buganizer558446054) {
  av1_init_intra_predictors();

  MACROBLOCKD xd = {};
  YV12_BUFFER_CONFIG cur_buf = {};
  xd.cur_buf = &cur_buf;
  MB_MODE_INFO mbmi = {};
  mbmi.bsize = BLOCK_8X8;
  mbmi.partition = PARTITION_NONE;
  mbmi.mode = D45_PRED;
  MB_MODE_INFO *mbmi_ptr = &mbmi;
  xd.mi = &mbmi_ptr;
  xd.bd = 8;
  xd.left_available = 1;
  xd.up_available = 1;
  xd.tile.mi_row_end = 100;
  xd.tile.mi_col_end = 100;
  // Set mb_to_bottom_edge and mb_to_right_edge such that
  // yd + txhpx < 0 and xr < 0:
  // yd + txhpx = (mb_to_bottom_edge >> 3) + hpx - y = -16 + 8 - 0 = -8 < 0
  // xr = (mb_to_right_edge >> 3) + wpx - x - txwpx = -5 + 8 - 0 - 4 = -1 < 0
  // xr + txwpx = 3 > 0 (so n_top_px = 3).
  xd.mb_to_bottom_edge = -16 * 8;
  xd.mb_to_right_edge = -5 * 8;

  uint8_t ref_buf[64 * 64];
  memset(ref_buf, 200, sizeof(ref_buf));
  uint8_t dst_buf[64 * 64] = { 0 };
  av1_predict_intra_block(&xd, BLOCK_64X64, /*enable_intra_edge_filter=*/0,
                          /*wpx=*/8, /*hpx=*/8, TX_4X4, D45_PRED,
                          /*angle_delta=*/0, /*use_palette=*/0,
                          FILTER_INTRA_MODES, ref_buf + 64 * 8 + 8, 64, dst_buf,
                          64, /*col_off=*/0, /*row_off=*/0, /*plane=*/0);
  // In Debug builds (-UNDEBUG), reverting clamp() triggers:
  //   Assertion `n_left_px >= 0' failed.
  // In Release builds (-DNDEBUG), reverting clamp() causes n_topright_px < 0
  // (-4 < 0), skipping top-right extension of above_row[3]=200 into
  // above_row[4..7] and leaving dst_buf[3] == 127 instead of 200.
  ASSERT_EQ(dst_buf[3], 200);
}

TEST(EncodeInternal, Buganizer558463892_558589747) {
  std::unique_ptr<AV1_COMP> cpi_test(new AV1_COMP());
  struct aom_internal_error_info error = {};
  if (setjmp(error.jmp)) FAIL();
  error.setjmp = 1;
  cpi_test->common.error = &error;
  cpi_test->frame_info.mi_rows = 16;
  cpi_test->frame_info.mi_cols = 16;
  av1_init_mb_wiener_var_buffer(cpi_test.get());
  ASSERT_NE(cpi_test->mb_weber_stats, nullptr);
  cpi_test->mb_weber_stats[0].satd = 12345;

  // Increase dimensions to 64x64 MI units: mb_weber_stats must be reallocated.
  cpi_test->frame_info.mi_rows = 64;
  cpi_test->frame_info.mi_cols = 64;
  av1_init_mb_wiener_var_buffer(cpi_test.get());
  EXPECT_EQ(cpi_test->mb_weber_stats[0].satd, 0);
  aom_free(cpi_test->mb_weber_stats);
}

TEST(EncodeInternal, Buganizer558417547) {
  std::unique_ptr<AV1_COMP> cpi_test(new AV1_COMP());
  SequenceHeader seq_params = {};
  seq_params.bit_depth = AOM_BITS_8;
  seq_params.sb_size = BLOCK_64X64;
  cpi_test->common.seq_params = &seq_params;
  cpi_test->common.mi_params.mi_rows = 16;
  cpi_test->common.mi_params.mi_cols = 16;
  cpi_test->common.quant_params.base_qindex = 128;
  cpi_test->common.delta_q_info.delta_q_res = 4;
  cpi_test->frame_info.mi_rows = 16;
  cpi_test->frame_info.mi_cols = 16;
  cpi_test->norm_wiener_variance = 100;
  struct aom_internal_error_info error = {};
  if (setjmp(error.jmp)) FAIL();
  error.setjmp = 1;
  cpi_test->common.error = &error;
  av1_init_mb_wiener_var_buffer(cpi_test.get());
  ASSERT_NE(cpi_test->mb_weber_stats, nullptr);

  // Set large SATD and distortion values exceeding INT32_MAX.
  for (int i = 0; i < 16 * 16; ++i) {
    cpi_test->mb_weber_stats[i].satd = 3000000000LL;
    cpi_test->mb_weber_stats[i].distortion = 3000000000LL;
    cpi_test->mb_weber_stats[i].rec_pix_max = 255;
  }

  // Test out-of-bounds / negative mi_row and mi_col without crashing or SIGFPE.
  int q_neg = av1_get_sbq_perceptual_ai(cpi_test.get(), BLOCK_64X64, -16, -16);
  EXPECT_GE(q_neg, 0);
  int q_oob = av1_get_sbq_perceptual_ai(cpi_test.get(), BLOCK_64X64, 100, 100);
  EXPECT_GE(q_oob, 0);

  aom_free(cpi_test->mb_weber_stats);
}

using AlignedU8Ptr = std::unique_ptr<uint8_t[], decltype(&aom_free)>;

AlignedU8Ptr AllocAlignedU8(size_t count) {
  return AlignedU8Ptr(static_cast<uint8_t *>(aom_memalign(16, count)),
                      &aom_free);
}

#if CONFIG_AV1_HIGHBITDEPTH
using AlignedU16Ptr = std::unique_ptr<uint16_t[], decltype(&aom_free)>;

AlignedU16Ptr AllocAlignedU16(size_t count) {
  return AlignedU16Ptr(
      static_cast<uint16_t *>(aom_memalign(16, count * sizeof(uint16_t))),
      &aom_free);
}
#endif  // CONFIG_AV1_HIGHBITDEPTH

int RefHasTopRightInSb(BLOCK_SIZE sb_size, BLOCK_SIZE bsize, int mi_row,
                       int mi_col, PARTITION_TYPE partition) {
  const int bw_unit = mi_size_wide[bsize];
  const int bh_unit = mi_size_high[bsize];
  const int sb_mi_size = mi_size_high[sb_size];
  const int rm = (mi_row & (sb_mi_size - 1)) & -bh_unit;
  if (rm == 0) return 1;
  const int cm = (mi_col & (sb_mi_size - 1)) & -bw_unit;
  if (cm + bw_unit >= sb_mi_size) return 0;

  // Z-order inside SB: walk recursive split bit from coarsest to finest.
  int size = sb_mi_size >> 1;
  while (size >= bw_unit || size >= bh_unit) {
    const int in_bot = (rm & size) != 0;
    const int tr_crosses = ((cm + bw_unit) & size) != 0 && (cm & size) == 0;
    const int tr_at_end = ((cm + bw_unit) & (size - 1)) == 0;
    if (size == bw_unit && size == bh_unit && size > 1 &&
        (partition == PARTITION_VERT_A || partition == PARTITION_VERT_B)) {
      if (in_bot) return 0;
    }
    if (tr_crosses && tr_at_end) {
      // Crossing vertical boundary: top-right is coded iff rm is at a strictly
      // coarser horizontal split.
      return (rm & (size - 1)) == 0;
    }
    if (in_bot && (rm & (size - 1)) == 0) {
      return 1;
    }
    size >>= 1;
  }
  return 0;
}

int RefHasBottomLeftInSb(BLOCK_SIZE sb_size, BLOCK_SIZE bsize, int mi_row,
                         int mi_col, PARTITION_TYPE partition,
                         int bottom_left_count_unit, int row_off, int ss_y) {
  const int bw_unit = mi_size_wide[bsize];
  const int bh_unit = mi_size_high[bsize];
  const int sb_mi_size = mi_size_high[sb_size];
  const int rm = (mi_row & (sb_mi_size - 1)) & -bh_unit;
  const int cm = (mi_col & (sb_mi_size - 1)) & -bw_unit;

  if (cm == 0) {
    const int row_off_in_sb = (rm >> ss_y) + row_off;
    const int sb_height_unit = sb_mi_size >> ss_y;
    return row_off_in_sb + bottom_left_count_unit < sb_height_unit;
  }
  if (rm + bh_unit >= sb_mi_size) return 0;

  // Z-order inside SB: walk split sizes from coarsest to finest.
  int size = sb_mi_size >> 1;
  while (size >= 1) {
    const int in_top = (rm & size) == 0;
    const int in_right = (cm & size) != 0;
    const int bl_crosses_h = ((rm + bh_unit) & size) != 0 && in_top;
    const int bl_crosses_v = in_right && ((cm - 1) & size) == 0;
    if (size == bw_unit && size == bh_unit && size > 1 &&
        (partition == PARTITION_VERT_A || partition == PARTITION_VERT_B) &&
        in_top && in_right) {
      return 1;
    }
    if (bl_crosses_h) return 0;
    if (bl_crosses_v) return 1;
    size >>= 1;
  }
  return 0;
}

int RefHasTopRight(BLOCK_SIZE sb_size, BLOCK_SIZE bsize, int mi_row, int mi_col,
                   int mi_col_end, PARTITION_TYPE partition, TX_SIZE txsz,
                   int row_off, int col_off, int ss_x, int ss_y) {
  const int txw = tx_size_wide_unit[txsz];
  if (mi_col + ((col_off + txw) << ss_x) >= mi_col_end) return 0;
  const int bw_unit = mi_size_wide[bsize];
  const int plane_bw_unit = AOMMAX(bw_unit >> ss_x, 1);
  if (row_off > 0) {
    if (block_size_wide[bsize] > block_size_wide[BLOCK_64X64]) {
      if (row_off == mi_size_high[BLOCK_64X64] >> ss_y &&
          col_off + txw == mi_size_wide[BLOCK_64X64] >> ss_x) {
        return 1;
      }
      const int plane_bw_unit_64 = mi_size_wide[BLOCK_64X64] >> ss_x;
      const int col_off_64 = col_off % plane_bw_unit_64;
      return col_off_64 + txw < plane_bw_unit_64;
    }
    return col_off + txw < plane_bw_unit;
  }
  if (col_off + txw < plane_bw_unit) return 1;
  return RefHasTopRightInSb(sb_size, bsize, mi_row, mi_col, partition);
}

int RefHasBottomLeft(BLOCK_SIZE sb_size, BLOCK_SIZE bsize, int mi_row,
                     int mi_col, int mi_row_end, PARTITION_TYPE partition,
                     TX_SIZE txsz, int row_off, int col_off, int ss_x,
                     int ss_y) {
  const int txh = tx_size_high_unit[txsz];
  if (mi_row + ((row_off + txh) << ss_y) >= mi_row_end) return 0;
  if (block_size_wide[bsize] > block_size_wide[BLOCK_64X64] && col_off > 0) {
    const int plane_bw_unit_64 = mi_size_wide[BLOCK_64X64] >> ss_x;
    const int col_off_64 = col_off % plane_bw_unit_64;
    if (col_off_64 == 0) {
      const int plane_bh_unit_64 = mi_size_high[BLOCK_64X64] >> ss_y;
      const int row_off_64 = row_off % plane_bh_unit_64;
      const int plane_bh_unit =
          AOMMIN(mi_size_high[bsize] >> ss_y, plane_bh_unit_64);
      return row_off_64 + txh < plane_bh_unit;
    }
  }
  if (col_off > 0) return 0;
  const int bh_unit = mi_size_high[bsize];
  const int plane_bh_unit = AOMMAX(bh_unit >> ss_y, 1);
  if (row_off + txh < plane_bh_unit) return 1;
  return RefHasBottomLeftInSb(sb_size, bsize, mi_row, mi_col, partition, txh,
                              row_off, ss_y);
}

int OracleFilterStrength(int bs0, int bs1, int delta, int type) {
  const int d = abs(delta);
  const int blk_wh = bs0 + bs1;
  if (type == 0) {
    if (blk_wh <= 8) return d >= 56;
    if (blk_wh <= 16) return d >= 40;
    if (blk_wh <= 24) return (d >= 8) + (d >= 16) + (d >= 32);
    if (blk_wh <= 32) return (d >= 1) + (d >= 4) + (d >= 32);
    return (d >= 1) * 3;
  }
  if (blk_wh <= 8) return (d >= 40) + (d >= 64);
  if (blk_wh <= 16) return (d >= 20) + (d >= 48);
  if (blk_wh <= 24) return (d >= 4) * 3;
  return (d >= 1) * 3;
}

void OracleFilterEdge(uint8_t *p, int sz, int strength) {
  if (!strength) return;
  static constexpr int kKernel[3][5] = {
    { 0, 4, 8, 4, 0 },
    { 0, 5, 6, 5, 0 },
    { 2, 4, 4, 4, 2 },
  };
  uint8_t edge[129];
  memcpy(edge, p, static_cast<size_t>(sz));
  for (int i = 1; i < sz; ++i) {
    int s = 0;
    for (int j = 0; j < 5; ++j) {
      int k = i - 2 + j;
      if (k < 0) k = 0;
      if (k > sz - 1) k = sz - 1;
      s += edge[k] * kKernel[strength - 1][j];
    }
    p[i] = static_cast<uint8_t>((s + 8) >> 4);
  }
}

void OracleUpsampleEdge(uint8_t *p, int sz) {
  uint8_t in[19];
  in[0] = p[-1];
  in[1] = p[-1];
  for (int i = 0; i < sz; ++i) in[i + 2] = p[i];
  in[sz + 2] = p[sz - 1];
  p[-2] = in[0];
  for (int i = 0; i < sz; ++i) {
    int s = -in[i] + 9 * in[i + 1] + 9 * in[i + 2] - in[i + 3];
    s = (s + 8) >> 4;
    if (s < 0) s = 0;
    if (s > 255) s = 255;
    p[2 * i - 1] = static_cast<uint8_t>(s);
    p[2 * i] = in[i + 2];
  }
}

void OracleDr8x8(uint8_t *dst, int dst_stride, const uint8_t *above_ref,
                 const uint8_t *left_ref, int ref_stride, int p_angle,
                 int enable_filter, int filter_type, int n_top_px,
                 int n_topright_px, int n_left_px, int n_bottomleft_px) {
  uint8_t above_buf[80] = { 0 };
  uint8_t left_buf[80] = { 0 };
  uint8_t *above = above_buf + 16;
  uint8_t *left = left_buf + 16;

  if (n_top_px > 0) {
    const int n_copy = n_top_px + (n_topright_px > 0 ? n_topright_px : 0);
    memcpy(above, above_ref, static_cast<size_t>(n_copy));
    for (int i = n_copy; i < 16; ++i) above[i] = above[n_copy - 1];
  } else {
    memset(above, n_left_px > 0 ? left_ref[0] : 127, 16);
  }
  if (n_left_px > 0) {
    const int n_copy = n_left_px + (n_bottomleft_px > 0 ? n_bottomleft_px : 0);
    for (int i = 0; i < n_copy; ++i) left[i] = left_ref[i * ref_stride];
    for (int i = n_copy; i < 16; ++i) left[i] = left[n_copy - 1];
  } else {
    memset(left, n_top_px > 0 ? above_ref[0] : 129, 16);
  }
  uint8_t tl = 128;
  if (n_top_px > 0 && n_left_px > 0) {
    tl = above_ref[-1];
  } else if (n_top_px > 0) {
    tl = above_ref[0];
  } else if (n_left_px > 0) {
    tl = left_ref[0];
  }
  above[-1] = tl;
  left[-1] = tl;

  int upsample_above = 0;
  int upsample_left = 0;
  if (enable_filter) {
    const int need_above = p_angle < 180;
    const int need_left = p_angle > 90;
    const int need_right = p_angle < 90;
    const int need_bottom = p_angle > 180;
    if (need_above && n_top_px > 0) {
      const int st = OracleFilterStrength(8, 8, p_angle - 90, filter_type);
      OracleFilterEdge(above - 1, n_top_px + 1 + (need_right ? 8 : 0), st);
    }
    if (need_left && n_left_px > 0) {
      const int st = OracleFilterStrength(8, 8, p_angle - 180, filter_type);
      OracleFilterEdge(left - 1, n_left_px + 1 + (need_bottom ? 8 : 0), st);
    }
    if (need_above &&
        av1_use_intra_edge_upsample(8, 8, p_angle - 90, filter_type)) {
      upsample_above = 1;
      OracleUpsampleEdge(above, 8 + (need_right ? 8 : 0));
    }
    if (need_left &&
        av1_use_intra_edge_upsample(8, 8, p_angle - 180, filter_type)) {
      upsample_left = 1;
      OracleUpsampleEdge(left, 8 + (need_bottom ? 8 : 0));
    }
  }

  if (p_angle > 0 && p_angle < 90) {
    const int dx = dr_intra_derivative[p_angle];
    const int max_base = 15 << upsample_above;
    const int frac_bits = 6 - upsample_above;
    const int base_inc = 1 << upsample_above;
    for (int r = 0, x = dx; r < 8; ++r, x += dx) {
      int base = x >> frac_bits;
      const int shift = ((x << upsample_above) & 0x3F) >> 1;
      for (int c = 0; c < 8; ++c, base += base_inc) {
        if (base < max_base) {
          const int v = above[base] * (32 - shift) + above[base + 1] * shift;
          dst[r * dst_stride + c] = static_cast<uint8_t>((v + 16) >> 5);
        } else {
          dst[r * dst_stride + c] = above[max_base];
        }
      }
    }
  } else if (p_angle > 180 && p_angle < 270) {
    const int dy = dr_intra_derivative[270 - p_angle];
    const int max_base = 15 << upsample_left;
    const int frac_bits = 6 - upsample_left;
    const int base_inc = 1 << upsample_left;
    for (int c = 0, y = dy; c < 8; ++c, y += dy) {
      int base = y >> frac_bits;
      const int shift = ((y << upsample_left) & 0x3F) >> 1;
      for (int r = 0; r < 8; ++r, base += base_inc) {
        if (base < max_base) {
          const int v = left[base] * (32 - shift) + left[base + 1] * shift;
          dst[r * dst_stride + c] = static_cast<uint8_t>((v + 16) >> 5);
        } else {
          dst[r * dst_stride + c] = left[max_base];
        }
      }
    }
  } else if (p_angle > 90 && p_angle < 180) {
    const int dx = dr_intra_derivative[180 - p_angle];
    const int dy = dr_intra_derivative[p_angle - 90];
    const int min_bx = -(1 << upsample_above);
    const int frac_x = 6 - upsample_above;
    const int frac_y = 6 - upsample_left;
    for (int r = 0; r < 8; ++r) {
      for (int c = 0; c < 8; ++c) {
        const int x = (c << 6) - (r + 1) * dx;
        const int bx = x >> frac_x;
        if (bx >= min_bx) {
          const int sh = ((x * (1 << upsample_above)) & 0x3F) >> 1;
          const int v = above[bx] * (32 - sh) + above[bx + 1] * sh;
          dst[r * dst_stride + c] = static_cast<uint8_t>((v + 16) >> 5);
        } else {
          const int y = (r << 6) - (c + 1) * dy;
          const int by = y >> frac_y;
          const int sh = ((y * (1 << upsample_left)) & 0x3F) >> 1;
          const int v = left[by] * (32 - sh) + left[by + 1] * sh;
          dst[r * dst_stride + c] = static_cast<uint8_t>((v + 16) >> 5);
        }
      }
    }
  } else {
    ADD_FAILURE() << "Unexpected directional angle: " << p_angle;
  }
}

void OracleFilterIntraLbd(uint8_t *dst, int dst_stride, TX_SIZE tx_size,
                          FILTER_INTRA_MODE fmode, const uint8_t *above_ref,
                          const uint8_t *left_ref, int ref_stride, int n_top_px,
                          int n_left_px) {
  const int txw = tx_size_wide[tx_size];
  const int txh = tx_size_high[tx_size];
  alignas(16) uint8_t above_buf[16 + 64] = {};
  alignas(16) uint8_t left_buf[16 + 64] = {};
  uint8_t *above = above_buf + 16;
  uint8_t *left = left_buf + 16;

  if (n_top_px > 0) {
    memcpy(above, above_ref, static_cast<size_t>(n_top_px));
    for (int i = n_top_px; i < txw; ++i) {
      above[i] = above[n_top_px - 1];
    }
  } else {
    const uint8_t fill = (n_left_px > 0) ? left_ref[0] : 127;
    memset(above, fill, static_cast<size_t>(txw));
  }

  if (n_left_px > 0) {
    for (int i = 0; i < n_left_px; ++i) {
      left[i] = left_ref[i * ref_stride];
    }
    for (int i = n_left_px; i < txh; ++i) {
      left[i] = left[n_left_px - 1];
    }
  } else {
    const uint8_t fill = (n_top_px > 0) ? above_ref[0] : 129;
    memset(left, fill, static_cast<size_t>(txh));
  }

  uint8_t tl = 128;
  if (n_top_px > 0 && n_left_px > 0) {
    tl = above_ref[-1];
  } else if (n_top_px > 0) {
    tl = above_ref[0];
  } else if (n_left_px > 0) {
    tl = left_ref[0];
  }
  above[-1] = tl;
  left[-1] = tl;

  av1_filter_intra_predictor_c(dst, dst_stride, tx_size, above, left,
                               static_cast<int>(fmode));
}

void FillPattern(uint8_t *buf, size_t size, int seed) {
  libaom_test::ACMRandom rnd(seed);
  for (size_t i = 0; i < size; ++i) {
    buf[i] = rnd.Rand8();
  }
}

TEST(EncodeInternal, IntraPredictAlignedVsUnaligned) {
  av1_init_intra_predictors();

  static constexpr TX_SIZE kTxSizes[] = {
    TX_4X4,  TX_4X8, TX_8X8, TX_16X16, TX_32X16,
#if !CONFIG_REALTIME_ONLY || CONFIG_AV1_DECODER
    TX_32X8,
#endif
  };
  static constexpr PREDICTION_MODE kModes[] = {
    V_PRED,    H_PRED,    DC_PRED,     D45_PRED,
    D135_PRED, D203_PRED, SMOOTH_PRED, PAETH_PRED,
  };
  static constexpr int kAlignOffsets[] = { 0, 1, 2, 3, 4, 8, 12 };

  constexpr int kStride = 128;
  AlignedU8Ptr ref_mem = AllocAlignedU8(128 * 128);
  AlignedU8Ptr out_aligned = AllocAlignedU8(64 * 128);
  AlignedU8Ptr aligned_ref = AllocAlignedU8(96 * 128);
  AlignedU8Ptr out_cur = AllocAlignedU8(64 * 128);
  AlignedU8Ptr ref_out = AllocAlignedU8(32 * 128);
  AlignedU8Ptr dst_buf = AllocAlignedU8(64 * 128);
  ASSERT_NE(ref_mem, nullptr);
  ASSERT_NE(out_aligned, nullptr);
  ASSERT_NE(aligned_ref, nullptr);
  ASSERT_NE(out_cur, nullptr);
  ASSERT_NE(ref_out, nullptr);
  ASSERT_NE(dst_buf, nullptr);
  std::unique_ptr<AV1_COMMON> cm(new AV1_COMMON());
  std::unique_ptr<MACROBLOCKD> xd(new MACROBLOCKD());
  FillPattern(ref_mem.get(), 128 * 128,
              libaom_test::ACMRandom::DeterministicSeed());

  for (TX_SIZE tx_size : kTxSizes) {
    const int txw = tx_size_wide[tx_size];
    const int txh = tx_size_high[tx_size];
    for (PREDICTION_MODE mode : kModes) {
      for (int top_avail = 0; top_avail <= 1; ++top_avail) {
        for (int left_avail = 0; left_avail <= 1; ++left_avail) {
          for (int edge_filt = 0; edge_filt <= 1; ++edge_filt) {
            memset(out_aligned.get(), 0, 64 * 128);
            for (int align_off : kAlignOffsets) {
              memset(xd.get(), 0, sizeof(MACROBLOCKD));
              YV12_BUFFER_CONFIG cur_buf = {};
              xd->cur_buf = &cur_buf;
              MB_MODE_INFO mbmi = {};
              mbmi.bsize = BLOCK_64X64;
              mbmi.partition = PARTITION_NONE;
              mbmi.mode = mode;
              MB_MODE_INFO *mbmi_ptr = &mbmi;
              xd->mi = &mbmi_ptr;
              xd->bd = 8;
              xd->up_available = (top_avail != 0);
              xd->left_available = (left_avail != 0);
              xd->tile.mi_row_end = 64;
              xd->tile.mi_col_end = 64;
              xd->mb_to_left_edge = -8 * 8;
              xd->mb_to_top_edge = -8 * 8;
              xd->mb_to_right_edge = 64 * 8;
              xd->mb_to_bottom_edge = 64 * 8;

              // Copy reference window so ref - ref_stride has exact alignment
              // `align_off` mod 16 while holding identical reference pixels.
              memset(aligned_ref.get(), 0, 96 * 128);
              const int base_row = 16;
              const int base_col = 16 + align_off;
              for (int r = -1; r < 64; ++r) {
                memcpy(&aligned_ref[(base_row + r) * kStride + base_col - 1],
                       &ref_mem[(16 + r) * kStride + 16 - 1], 80);
              }
              const uint8_t *ref_ptr =
                  &aligned_ref[base_row * kStride + base_col];
              memset(out_cur.get(), 0, 64 * 128);
              av1_predict_intra_block(xd.get(), BLOCK_64X64, edge_filt, 64, 64,
                                      tx_size, mode, 0, 0, FILTER_INTRA_MODES,
                                      ref_ptr, kStride, out_cur.get(), kStride,
                                      0, 0, AOM_PLANE_Y);

              if (align_off == 0) {
                memcpy(out_aligned.get(), out_cur.get(), 64 * 128);
                if (mode == V_PRED && top_avail) {
                  for (int r = 0; r < txh; ++r) {
                    ASSERT_EQ(memcmp(&out_cur[r * kStride], ref_ptr - kStride,
                                     static_cast<size_t>(txw)),
                              0)
                        << "V_PRED mismatch tx=" << static_cast<int>(tx_size);
                  }
                } else if (mode == H_PRED && left_avail) {
                  for (int r = 0; r < txh; ++r) {
                    for (int c = 0; c < txw; ++c) {
                      ASSERT_EQ(out_cur[r * kStride + c],
                                ref_ptr[r * kStride - 1])
                          << "H_PRED mismatch tx=" << static_cast<int>(tx_size);
                    }
                  }
                } else if (mode == DC_PRED) {
                  int sum = 0;
                  int count = 0;
                  if (top_avail) {
                    for (int c = 0; c < txw; ++c) sum += ref_ptr[-kStride + c];
                    count += txw;
                  }
                  if (left_avail) {
                    for (int r = 0; r < txh; ++r) {
                      sum += ref_ptr[r * kStride - 1];
                    }
                    count += txh;
                  }
                  const uint8_t expected =
                      (count > 0)
                          ? static_cast<uint8_t>((sum + (count >> 1)) / count)
                          : 128;
                  for (int r = 0; r < txh; ++r) {
                    for (int c = 0; c < txw; ++c) {
                      ASSERT_EQ(out_cur[r * kStride + c], expected)
                          << "DC_PRED mismatch tx=" << static_cast<int>(tx_size)
                          << " top=" << top_avail << " left=" << left_avail;
                    }
                  }
                }
              } else {
                for (int r = 0; r < txh; ++r) {
                  ASSERT_EQ(
                      memcmp(&out_aligned[r * kStride], &out_cur[r * kStride],
                             static_cast<size_t>(txw)),
                      0)
                      << "align_off=" << align_off
                      << " tx=" << static_cast<int>(tx_size)
                      << " mode=" << static_cast<int>(mode)
                      << " top=" << top_avail << " left=" << left_avail;
                }
              }
            }
          }
        }
      }
    }
  }

  // Also test av1_predict_intra_block_facade across valid transform-grid
  // blk_col values (multiples of tx_size_wide_unit[tx_size]) so 4-wide tx
  // exercises +0/+4/+8/+12 mod 16, 8-wide exercises +0/+8 mod 16, and 16-wide
  // stays 16-byte aligned as required by x86 SSE2 stores.
  static constexpr TX_SIZE kFacadeTx[] = { TX_4X4, TX_4X8, TX_8X8, TX_16X16 };
  static constexpr PREDICTION_MODE kFacadeModes[] = { V_PRED, H_PRED, DC_PRED,
                                                      PAETH_PRED };
  for (TX_SIZE tx_size : kFacadeTx) {
    const int txw = tx_size_wide[tx_size];
    const int txh = tx_size_high[tx_size];
    const int step_unit = tx_size_wide_unit[tx_size];
    const int blk_row = tx_size_high_unit[tx_size];
    for (PREDICTION_MODE mode : kFacadeModes) {
      memset(ref_out.get(), 0, 32 * 128);
      for (int blk_col = 0; blk_col < 8; blk_col += step_unit) {
        memcpy(dst_buf.get(), ref_mem.get(), 64 * kStride);

        memset(cm.get(), 0, sizeof(AV1_COMMON));
        SequenceHeader seq = {};
        seq.sb_size = BLOCK_64X64;
        seq.enable_intra_edge_filter = 1;
        cm->seq_params = &seq;

        memset(xd.get(), 0, sizeof(MACROBLOCKD));
        YV12_BUFFER_CONFIG cur_buf = {};
        xd->cur_buf = &cur_buf;
        MB_MODE_INFO mbmi = {};
        mbmi.bsize = BLOCK_32X32;
        mbmi.partition = PARTITION_NONE;
        mbmi.mode = mode;
        mbmi.filter_intra_mode_info.filter_intra_mode = FILTER_INTRA_MODES;
        MB_MODE_INFO *mbmi_ptr = &mbmi;
        xd->mi = &mbmi_ptr;
        xd->bd = 8;
        xd->up_available = true;
        xd->left_available = true;
        xd->tile.mi_row_end = 64;
        xd->tile.mi_col_end = 64;
        xd->mb_to_left_edge = -8 * 8;
        xd->mb_to_top_edge = -8 * 8;
        xd->mb_to_right_edge = 128 * 8;
        xd->mb_to_bottom_edge = 128 * 8;
        xd->plane[0].dst.buf = &dst_buf[16 * kStride + 16];
        xd->plane[0].dst.stride = kStride;
        xd->plane[0].width = 32;
        xd->plane[0].height = 32;

        // Normalize reference pixels for this (blk_col, blk_row) so every
        // blk_col sees identical above/left boundary samples.
        uint8_t *blk_dst =
            &xd->plane[0]
                 .dst.buf[(blk_row * kStride + blk_col) << MI_SIZE_LOG2];
        memcpy(blk_dst - kStride - 1, &ref_mem[15 * kStride + 15], 40);
        for (int r = 0; r < txh; ++r) {
          blk_dst[r * kStride - 1] = ref_mem[(16 + r) * kStride + 15];
        }

        av1_predict_intra_block_facade(cm.get(), xd.get(), AOM_PLANE_Y, blk_col,
                                       blk_row, tx_size);
        if (blk_col == 0) {
          for (int r = 0; r < txh; ++r) {
            memcpy(&ref_out[r * kStride], &blk_dst[r * kStride],
                   static_cast<size_t>(txw));
          }
        } else {
          for (int r = 0; r < txh; ++r) {
            ASSERT_EQ(memcmp(&ref_out[r * kStride], &blk_dst[r * kStride],
                             static_cast<size_t>(txw)),
                      0)
                << "facade blk_col=" << blk_col
                << " tx=" << static_cast<int>(tx_size)
                << " mode=" << static_cast<int>(mode);
          }
        }
      }
    }
  }
}

TEST(EncodeInternal, IntraPredictFacadeTopRightBottomLeft) {
  av1_init_intra_predictors();

  struct Case {
    BLOCK_SIZE sb_size;
    BLOCK_SIZE bsize;
    PARTITION_TYPE partition;
    TX_SIZE tx_size;
    int plane;
    int ss_x;
    int ss_y;
    int mi_row;
    int mi_col;
    int row_off;
    int col_off;
    int mi_row_end;
    int mi_col_end;
  };

  // Fields: { sb_size, bsize, partition, tx_size, plane, ss_x, ss_y, mi_row,
  //           mi_col, row_off, col_off, mi_row_end, mi_col_end }
  static constexpr Case kCases[] = {
    // PARTITION_VERT_A / VERT_B square BL blocks must report has_top_right == 0
    // (catches broken `(rm & bh_unit)` in has_top_right_in_sb).
    { BLOCK_64X64, BLOCK_8X8, PARTITION_VERT_A, TX_8X8, 0, 0, 0, 2, 0, 0, 0, 64,
      64 },
    { BLOCK_64X64, BLOCK_8X8, PARTITION_VERT_B, TX_8X8, 0, 0, 0, 2, 4, 0, 0, 64,
      64 },
    { BLOCK_64X64, BLOCK_16X16, PARTITION_VERT_A, TX_16X16, 0, 0, 0, 4, 0, 0, 0,
      64, 64 },
    { BLOCK_128X128, BLOCK_32X32, PARTITION_VERT_B, TX_32X32, 0, 0, 0, 8, 0, 0,
      0, 64, 64 },
    // PARTITION_VERT_A / VERT_B square TR blocks must report
    // has_bottom_left == 1 (catches broken `!(rm & bh_unit) && (cm &
    // bw_unit)`).
    { BLOCK_64X64, BLOCK_8X8, PARTITION_VERT_A, TX_8X8, 0, 0, 0, 0, 2, 0, 0, 64,
      64 },
    { BLOCK_64X64, BLOCK_16X16, PARTITION_VERT_B, TX_16X16, 0, 0, 0, 0, 4, 0, 0,
      64, 64 },
    // Standard Z-order contrast: PARTITION_NONE at same coords flips TR/BL.
    { BLOCK_64X64, BLOCK_8X8, PARTITION_NONE, TX_8X8, 0, 0, 0, 2, 0, 0, 0, 64,
      64 },
    { BLOCK_64X64, BLOCK_8X8, PARTITION_NONE, TX_8X8, 0, 0, 0, 0, 2, 0, 0, 64,
      64 },
    // General Z-order formula case where has_bottom_left_in_sb returns 1.
    { BLOCK_64X64, BLOCK_8X8, PARTITION_NONE, TX_8X8, 0, 0, 0, 0, 4, 0, 0, 64,
      64 },
    // SB top row & rightmost column boundaries.
    { BLOCK_64X64, BLOCK_8X8, PARTITION_NONE, TX_8X8, 0, 0, 0, 0, 14, 0, 0, 64,
      64 },
    { BLOCK_64X64, BLOCK_8X8, PARTITION_NONE, TX_8X8, 0, 0, 0, 2, 14, 0, 0, 64,
      64 },
    // Leftmost SB column BL vs bottommost SB row.
    { BLOCK_64X64, BLOCK_16X16, PARTITION_NONE, TX_16X16, 0, 0, 0, 12, 0, 0, 0,
      64, 64 },
    { BLOCK_64X64, BLOCK_16X16, PARTITION_NONE, TX_16X16, 0, 0, 0, 12, 4, 0, 0,
      64, 64 },
    // Tile boundaries.
    { BLOCK_64X64, BLOCK_8X8, PARTITION_NONE, TX_8X8, 0, 0, 0, 0, 6, 0, 0, 64,
      8 },
    { BLOCK_64X64, BLOCK_8X8, PARTITION_NONE, TX_8X8, 0, 0, 0, 6, 0, 0, 0, 8,
      64 },
    // BLOCK_128X128 special center case & 64x* BL col_off_64 == 0.
    { BLOCK_128X128, BLOCK_128X128, PARTITION_NONE, TX_16X16, 0, 0, 0, 0, 0, 16,
      12, 64, 64 },
    { BLOCK_128X128, BLOCK_128X128, PARTITION_NONE, TX_16X16, 0, 0, 0, 0, 0, 12,
      16, 64, 64 },
    { BLOCK_128X128, BLOCK_128X128, PARTITION_NONE, TX_16X16, 0, 0, 0, 0, 0, 16,
      16, 64, 64 },
    // 4:2:0 chroma subsampled offsets & 4:4:4 chroma origin fast path.
    { BLOCK_64X64, BLOCK_16X16, PARTITION_VERT_A, TX_8X8, 1, 1, 1, 4, 0, 0, 0,
      64, 64 },
    { BLOCK_64X64, BLOCK_16X16, PARTITION_VERT_B, TX_8X8, 1, 1, 1, 0, 4, 0, 0,
      64, 64 },
    { BLOCK_64X64, BLOCK_32X32, PARTITION_NONE, TX_8X8, 1, 1, 1, 0, 0, 2, 2, 64,
      64 },
    { BLOCK_64X64, BLOCK_8X8, PARTITION_VERT_A, TX_8X8, 1, 0, 0, 2, 0, 0, 0, 64,
      64 },
    { BLOCK_64X64, BLOCK_8X8, PARTITION_VERT_A, TX_8X8, 1, 0, 0, 0, 2, 0, 0, 64,
      64 },
  };

  constexpr int kStride = 256;
  AlignedU8Ptr buf = AllocAlignedU8(256 * 256);
  ASSERT_NE(buf, nullptr);
  std::unique_ptr<AV1_COMMON> cm(new AV1_COMMON());
  std::unique_ptr<MACROBLOCKD> xd(new MACROBLOCKD());
  for (int idx = 0; idx < NELEMENTS(kCases); ++idx) {
    const Case &tc = kCases[idx];
    const int txw = tx_size_wide[tc.tx_size];
    const int txh = tx_size_high[tc.tx_size];
    const int x = tc.col_off << MI_SIZE_LOG2;
    const int y = tc.row_off << MI_SIZE_LOG2;

    for (int test_tr = 0; test_tr <= 1; ++test_tr) {
      memset(buf.get(), 10, 256 * 256);
      memset(cm.get(), 0, sizeof(AV1_COMMON));
      SequenceHeader seq = {};
      seq.sb_size = tc.sb_size;
      seq.enable_intra_edge_filter = 0;
      cm->seq_params = &seq;

      memset(xd.get(), 0, sizeof(MACROBLOCKD));
      YV12_BUFFER_CONFIG cur_buf = {};
      xd->cur_buf = &cur_buf;
      MB_MODE_INFO mbmi = {};
      mbmi.bsize = tc.bsize;
      mbmi.partition = tc.partition;
      mbmi.mode = test_tr ? D45_PRED : D203_PRED;
      mbmi.uv_mode = test_tr ? UV_D45_PRED : UV_D203_PRED;
      mbmi.filter_intra_mode_info.filter_intra_mode = FILTER_INTRA_MODES;
      MB_MODE_INFO *mbmi_ptr = &mbmi;
      xd->mi = &mbmi_ptr;
      xd->bd = 8;
      xd->up_available = true;
      xd->left_available = true;
      xd->chroma_up_available = true;
      xd->chroma_left_available = true;
      xd->tile.mi_row_end = tc.mi_row_end;
      xd->tile.mi_col_end = tc.mi_col_end;
      xd->mb_to_left_edge = -(tc.mi_col * MI_SIZE * 8);
      xd->mb_to_top_edge = -(tc.mi_row * MI_SIZE * 8);
      xd->mb_to_right_edge = 512 * 8;
      xd->mb_to_bottom_edge = 512 * 8;
      xd->plane[tc.plane].subsampling_x = tc.ss_x;
      xd->plane[tc.plane].subsampling_y = tc.ss_y;
      xd->plane[tc.plane].width = 128;
      xd->plane[tc.plane].height = 128;
      xd->plane[tc.plane].dst.stride = kStride;
      xd->plane[tc.plane].dst.buf = buf.get() + 16 * kStride + 16;

      uint8_t *blk_dst = xd->plane[tc.plane].dst.buf + y * kStride + x;
      if (test_tr) {
        memset(blk_dst - kStride + txw, 200, static_cast<size_t>(txw));
        av1_predict_intra_block_facade(cm.get(), xd.get(), tc.plane, tc.col_off,
                                       tc.row_off, tc.tx_size);
        const int expected_tr = RefHasTopRight(
            tc.sb_size, tc.bsize, tc.mi_row, tc.mi_col, tc.mi_col_end,
            tc.partition, tc.tx_size, tc.row_off, tc.col_off, tc.ss_x, tc.ss_y);
        EXPECT_EQ(blk_dst[txw - 1], expected_tr ? 200 : 10)
            << "TR mismatch at case " << idx;
        if (idx == 3) {
          EXPECT_EQ(expected_tr, 0);
          EXPECT_EQ(RefHasTopRight(tc.sb_size, tc.bsize, tc.mi_row, tc.mi_col,
                                   tc.mi_col_end, PARTITION_NONE, tc.tx_size,
                                   tc.row_off, tc.col_off, tc.ss_x, tc.ss_y),
                    1);
        }
      } else {
        for (int r = txh; r < 2 * txh; ++r) {
          blk_dst[r * kStride - 1] = 200;
        }
        av1_predict_intra_block_facade(cm.get(), xd.get(), tc.plane, tc.col_off,
                                       tc.row_off, tc.tx_size);
        const int expected_bl = RefHasBottomLeft(
            tc.sb_size, tc.bsize, tc.mi_row, tc.mi_col, tc.mi_row_end,
            tc.partition, tc.tx_size, tc.row_off, tc.col_off, tc.ss_x, tc.ss_y);
        EXPECT_EQ(blk_dst[(txh - 1) * kStride + (txw - 1)],
                  expected_bl ? 200 : 10)
            << "BL mismatch at case " << idx;
        if (idx == 8) {
          EXPECT_EQ(expected_bl, 1);
        }
      }
    }
  }
}

TEST(EncodeInternal, IntraPredict8x8ZonesAndEdgeFilter) {
  av1_init_intra_predictors();

  constexpr int kStride = 128;
  AlignedU8Ptr ref_mem = AllocAlignedU8(64 * 128);
  AlignedU8Ptr dst_buf = AllocAlignedU8(64 * 128);
  ASSERT_NE(ref_mem, nullptr);
  ASSERT_NE(dst_buf, nullptr);
  std::unique_ptr<AV1_COMMON> cm(new AV1_COMMON());
  std::unique_ptr<MACROBLOCKD> xd(new MACROBLOCKD());
  FillPattern(ref_mem.get(), 64 * 128,
              libaom_test::ACMRandom::DeterministicSeed() + 1);

  struct ZoneCase {
    PREDICTION_MODE mode;
    int angle_delta;
  };
  static constexpr ZoneCase kZoneCases[] = {
    { D45_PRED, 0 },   { D45_PRED, -3 }, { D67_PRED, 0 },  { D67_PRED, 3 },
    { D113_PRED, -2 }, { D135_PRED, 0 }, { D157_PRED, 2 }, { D203_PRED, -3 },
    { D203_PRED, 0 },  { D203_PRED, 3 },
  };

  for (const ZoneCase &zc : kZoneCases) {
    const int p_angle =
        mode_to_angle_map[zc.mode] + zc.angle_delta * ANGLE_STEP;
    const int avail_cfgs = (p_angle < 90 || p_angle > 180) ? 2 : 1;
    for (int edge_filt = 0; edge_filt <= 1; ++edge_filt) {
      for (int smooth_cfg = 0; smooth_cfg <= 2; ++smooth_cfg) {
        for (int tr_bl = 0; tr_bl <= 2; ++tr_bl) {
          const int avail_ext = (tr_bl == 0) ? 0 : (tr_bl == 1 ? 4 : 8);
          for (int avail_idx = 0; avail_idx < avail_cfgs; ++avail_idx) {
            const int up_avail = (avail_idx == 1 && p_angle > 180) ? 0 : 1;
            const int left_avail = (avail_idx == 1 && p_angle < 90) ? 0 : 1;

            memcpy(dst_buf.get(), ref_mem.get(), 64 * 128);

            memset(cm.get(), 0, sizeof(AV1_COMMON));
            SequenceHeader seq = {};
            seq.sb_size = BLOCK_64X64;
            seq.enable_intra_edge_filter = static_cast<uint8_t>(edge_filt);
            cm->seq_params = &seq;

            MB_MODE_INFO above_mbmi = {};
            above_mbmi.mode = (smooth_cfg == 1) ? SMOOTH_PRED : DC_PRED;
            MB_MODE_INFO left_mbmi = {};
            left_mbmi.mode = (smooth_cfg == 2) ? SMOOTH_V_PRED : V_PRED;

            memset(xd.get(), 0, sizeof(MACROBLOCKD));
            YV12_BUFFER_CONFIG cur_buf = {};
            xd->cur_buf = &cur_buf;
            MB_MODE_INFO mbmi = {};
            mbmi.bsize = BLOCK_8X8;
            mbmi.partition = PARTITION_NONE;
            mbmi.mode = zc.mode;
            mbmi.angle_delta[0] = static_cast<int8_t>(zc.angle_delta);
            mbmi.filter_intra_mode_info.filter_intra_mode = FILTER_INTRA_MODES;
            MB_MODE_INFO *mbmi_ptr = &mbmi;
            xd->mi = &mbmi_ptr;
            xd->above_mbmi = up_avail ? &above_mbmi : nullptr;
            xd->left_mbmi = left_avail ? &left_mbmi : nullptr;
            xd->bd = 8;
            xd->up_available = (up_avail != 0);
            xd->left_available = (left_avail != 0);
            xd->tile.mi_row_end = 64;
            xd->tile.mi_col_end = 64;
            xd->mb_to_left_edge = -16 * MI_SIZE * 8;
            xd->mb_to_top_edge = -4 * MI_SIZE * 8;
            xd->mb_to_right_edge = avail_ext * 8;
            xd->mb_to_bottom_edge = avail_ext * 8;
            xd->plane[0].width = 8;
            xd->plane[0].height = 8;
            xd->plane[0].dst.stride = kStride;
            xd->plane[0].dst.buf = dst_buf.get() + 16 * kStride + 16;

            const int filter_type = ((up_avail && smooth_cfg == 1) ||
                                     (left_avail && smooth_cfg == 2))
                                        ? 1
                                        : 0;
            const uint8_t *orig_ref = &ref_mem[16 * kStride + 16];
            uint8_t oracle[8 * 8] = {};
            OracleDr8x8(oracle, 8, orig_ref - kStride, orig_ref - 1, kStride,
                        p_angle, edge_filt, filter_type, up_avail ? 8 : 0,
                        (up_avail && p_angle < 90) ? avail_ext : -1,
                        left_avail ? 8 : 0,
                        (left_avail && p_angle > 180) ? avail_ext : -1);

            av1_predict_intra_block_facade(cm.get(), xd.get(), AOM_PLANE_Y, 0,
                                           0, TX_8X8);
            uint8_t *blk_dst = xd->plane[0].dst.buf;
            for (int r = 0; r < 8; ++r) {
              ASSERT_EQ(memcmp(&blk_dst[r * kStride], &oracle[r * 8], 8), 0)
                  << "mode=" << static_cast<int>(zc.mode)
                  << " delta=" << zc.angle_delta << " filt=" << edge_filt
                  << " smooth_cfg=" << smooth_cfg << " ext=" << avail_ext
                  << " up=" << up_avail << " left=" << left_avail;
            }
          }
        }
      }
    }
  }
}

TEST(EncodeInternal, IntraPredictFilterIntraInteriorAndBorder) {
  av1_init_intra_predictors();

  constexpr int kStride = 128;
  AlignedU8Ptr ref_mem = AllocAlignedU8(128 * 128);
  AlignedU8Ptr facade_buf = AllocAlignedU8(128 * 128);
  AlignedU8Ptr direct_out = AllocAlignedU8(64 * 128);
  AlignedU8Ptr oracle_out = AllocAlignedU8(64 * 128);
  ASSERT_NE(ref_mem, nullptr);
  ASSERT_NE(facade_buf, nullptr);
  ASSERT_NE(direct_out, nullptr);
  ASSERT_NE(oracle_out, nullptr);
  std::unique_ptr<AV1_COMMON> cm(new AV1_COMMON());
  std::unique_ptr<MACROBLOCKD> xd(new MACROBLOCKD());
  FillPattern(ref_mem.get(), 128 * 128,
              libaom_test::ACMRandom::DeterministicSeed() + 2);

  static constexpr TX_SIZE kFilterTx[] = {
    TX_4X4,  TX_8X8, TX_16X16, TX_32X32, TX_4X8, TX_8X16,
#if !CONFIG_REALTIME_ONLY || CONFIG_AV1_DECODER
    TX_32X8,
#endif
  };

  for (TX_SIZE tx_size : kFilterTx) {
    const int txw = tx_size_wide[tx_size];
    const int txh = tx_size_high[tx_size];
    for (int fmode = 0; fmode < FILTER_INTRA_MODES; ++fmode) {
      const FILTER_INTRA_MODE fi_mode = static_cast<FILTER_INTRA_MODE>(fmode);
      for (int cfg = 0; cfg < 5; ++cfg) {
        // cfg 0: interior; 1: no above; 2: no left;
        // 3: partial right/bottom; 4: neither above nor left available.
        const int top_avail = (cfg != 1 && cfg != 4);
        const int left_avail = (cfg != 2 && cfg != 4);
        const int wpx = (cfg == 3) ? (txw - 2) : 64;
        const int hpx = (cfg == 3) ? (txh - 2) : 64;
        const int n_top_px = top_avail ? (cfg == 3 ? txw - 2 : txw) : 0;
        const int n_left_px = left_avail ? (cfg == 3 ? txh - 2 : txh) : 0;

        memcpy(facade_buf.get(), ref_mem.get(), 128 * 128);

        memset(cm.get(), 0, sizeof(AV1_COMMON));
        SequenceHeader seq = {};
        seq.sb_size = BLOCK_64X64;
        seq.enable_intra_edge_filter = 1;
        cm->seq_params = &seq;

        memset(xd.get(), 0, sizeof(MACROBLOCKD));
        YV12_BUFFER_CONFIG cur_buf = {};
        xd->cur_buf = &cur_buf;
        MB_MODE_INFO mbmi = {};
        mbmi.bsize = BLOCK_32X32;
        mbmi.partition = PARTITION_NONE;
        mbmi.mode = DC_PRED;
        mbmi.filter_intra_mode_info.use_filter_intra = 1;
        mbmi.filter_intra_mode_info.filter_intra_mode = fi_mode;
        MB_MODE_INFO *mbmi_ptr = &mbmi;
        xd->mi = &mbmi_ptr;
        xd->bd = 8;
        xd->up_available = (top_avail != 0);
        xd->left_available = (left_avail != 0);
        xd->tile.mi_row_end = 64;
        xd->tile.mi_col_end = 64;
        xd->mb_to_left_edge = -8 * 8;
        xd->mb_to_top_edge = -8 * 8;
        xd->mb_to_right_edge = 0;
        xd->mb_to_bottom_edge = 0;
        xd->plane[0].width = wpx;
        xd->plane[0].height = hpx;
        xd->plane[0].dst.stride = kStride;
        xd->plane[0].dst.buf = facade_buf.get() + 16 * kStride + 16;

        memset(oracle_out.get(), 0, 64 * 128);
        OracleFilterIntraLbd(oracle_out.get(), kStride, tx_size, fi_mode,
                             &ref_mem[15 * kStride + 16],
                             &ref_mem[16 * kStride + 15], kStride, n_top_px,
                             n_left_px);

        av1_predict_intra_block_facade(cm.get(), xd.get(), AOM_PLANE_Y, 0, 0,
                                       tx_size);

        memset(direct_out.get(), 0, 64 * 128);
        av1_predict_intra_block(xd.get(), BLOCK_64X64, 0, wpx, hpx, tx_size,
                                DC_PRED, 0, 0, fi_mode,
                                &ref_mem[16 * kStride + 16], kStride,
                                direct_out.get(), kStride, 0, 0, AOM_PLANE_Y);

        for (int r = 0; r < txh; ++r) {
          ASSERT_EQ(memcmp(&xd->plane[0].dst.buf[r * kStride],
                           &oracle_out[r * kStride], static_cast<size_t>(txw)),
                    0)
              << "filter_intra facade vs oracle tx="
              << static_cast<int>(tx_size) << " fmode=" << fmode
              << " cfg=" << cfg;
          ASSERT_EQ(memcmp(&direct_out[r * kStride], &oracle_out[r * kStride],
                           static_cast<size_t>(txw)),
                    0)
              << "filter_intra direct vs oracle tx="
              << static_cast<int>(tx_size) << " fmode=" << fmode
              << " cfg=" << cfg;
        }
      }
    }
  }
}

TEST(EncodeInternal, IntraPredictPartialBorderTx64) {
  av1_init_intra_predictors();

  constexpr int kStride = 256;
  constexpr int kRows = 160;
  AlignedU8Ptr ref_mem = AllocAlignedU8(kRows * kStride);
  AlignedU8Ptr facade_buf = AllocAlignedU8(kRows * kStride);
  AlignedU8Ptr direct_out = AllocAlignedU8(64 * kStride);
  ASSERT_NE(ref_mem, nullptr);
  ASSERT_NE(facade_buf, nullptr);
  ASSERT_NE(direct_out, nullptr);
  std::unique_ptr<AV1_COMMON> cm(new AV1_COMMON());
  std::unique_ptr<MACROBLOCKD> xd(new MACROBLOCKD());
  for (int i = 0; i < kRows * kStride; ++i) {
    ref_mem[i] = static_cast<uint8_t>((i * 53 + 17) & 0xff);
  }

  static constexpr int kWpx[] = { 8, 40, 56, 80, 96 };
  static constexpr int kHpx[] = { 16, 48, 80, 96 };
  static constexpr PREDICTION_MODE kModes[] = { DC_PRED, D45_PRED, D203_PRED };

  for (int wpx : kWpx) {
    for (int hpx : kHpx) {
      for (PREDICTION_MODE mode : kModes) {
        for (int edge_filt = 0; edge_filt <= 1; ++edge_filt) {
          memcpy(facade_buf.get(), ref_mem.get(), kRows * kStride);
          memset(direct_out.get(), 0, 64 * kStride);

          memset(cm.get(), 0, sizeof(AV1_COMMON));
          SequenceHeader seq = {};
          seq.sb_size = BLOCK_64X64;
          seq.enable_intra_edge_filter = static_cast<uint8_t>(edge_filt);
          cm->seq_params = &seq;

          memset(xd.get(), 0, sizeof(MACROBLOCKD));
          YV12_BUFFER_CONFIG cur_buf = {};
          xd->cur_buf = &cur_buf;
          MB_MODE_INFO mbmi = {};
          mbmi.bsize = BLOCK_64X64;
          mbmi.partition = PARTITION_NONE;
          mbmi.mode = mode;
          mbmi.filter_intra_mode_info.filter_intra_mode = FILTER_INTRA_MODES;
          MB_MODE_INFO *mbmi_ptr = &mbmi;
          xd->mi = &mbmi_ptr;
          xd->bd = 8;
          xd->up_available = true;
          xd->left_available = true;
          xd->tile.mi_row_end = 64;
          xd->tile.mi_col_end = 64;
          xd->mb_to_left_edge = -8 * 8;
          xd->mb_to_top_edge = -8 * 8;
          xd->mb_to_right_edge = 0;
          xd->mb_to_bottom_edge = 0;
          xd->plane[0].width = wpx;
          xd->plane[0].height = hpx;
          xd->plane[0].dst.stride = kStride;
          xd->plane[0].dst.buf = facade_buf.get() + 16 * kStride + 16;

          av1_predict_intra_block_facade(cm.get(), xd.get(), AOM_PLANE_Y, 0, 0,
                                         TX_64X64);

          const uint8_t *ref = &ref_mem[16 * kStride + 16];
          av1_predict_intra_block(xd.get(), BLOCK_64X64, edge_filt, wpx, hpx,
                                  TX_64X64, mode, 0, 0, FILTER_INTRA_MODES, ref,
                                  kStride, direct_out.get(), kStride, 0, 0,
                                  AOM_PLANE_Y);

          for (int r = 0; r < 64; ++r) {
            ASSERT_EQ(memcmp(&xd->plane[0].dst.buf[r * kStride],
                             &direct_out[r * kStride], 64),
                      0)
                << "TX_64X64 border facade vs direct wpx=" << wpx
                << " hpx=" << hpx << " mode=" << static_cast<int>(mode)
                << " filt=" << edge_filt << " r=" << r;
          }

          if (mode == DC_PRED) {
            const int n_top = AOMMIN(wpx, 64);
            const int n_left = AOMMIN(hpx, 64);
            int sum = 0;
            for (int c = 0; c < n_top; ++c) sum += ref[-kStride + c];
            sum += (64 - n_top) * ref[-kStride + n_top - 1];
            for (int r = 0; r < n_left; ++r) sum += ref[r * kStride - 1];
            sum += (64 - n_left) * ref[(n_left - 1) * kStride - 1];
            const uint8_t expected_dc = static_cast<uint8_t>((sum + 64) >> 7);
            for (int r = 0; r < 64; ++r) {
              for (int c = 0; c < 64; ++c) {
                ASSERT_EQ(xd->plane[0].dst.buf[r * kStride + c], expected_dc)
                    << "TX_64X64 border DC_PRED wpx=" << wpx << " hpx=" << hpx;
              }
            }
          }
        }
      }
    }
  }
}

#if CONFIG_AV1_HIGHBITDEPTH
TEST(EncodeInternal, IntraPredictHbdFacade) {
  av1_init_intra_predictors();

  constexpr int kStride = 128;
  AlignedU16Ptr buf = AllocAlignedU16(64 * kStride);
  AlignedU16Ptr facade_buf = AllocAlignedU16(64 * kStride);
  AlignedU16Ptr direct_out = AllocAlignedU16(16 * kStride);
  ASSERT_NE(buf, nullptr);
  ASSERT_NE(facade_buf, nullptr);
  ASSERT_NE(direct_out, nullptr);
  std::unique_ptr<AV1_COMMON> cm(new AV1_COMMON());
  std::unique_ptr<MACROBLOCKD> xd(new MACROBLOCKD());
  for (int i = 0; i < 64 * kStride; ++i) {
    buf[i] = static_cast<uint16_t>((i * 37) & 0x3ff);
  }

  static constexpr PREDICTION_MODE kHbdModes[] = {
    DC_PRED, V_PRED, H_PRED, D45_PRED, D135_PRED, D203_PRED, SMOOTH_PRED,
  };
  static constexpr int kHbdCoords[][2] = { { 0, 0 }, { 4, 4 } };

  for (PREDICTION_MODE mode : kHbdModes) {
    const int max_delta_idx = av1_is_directional_mode(mode) ? 1 : 0;
    for (int delta_idx = 0; delta_idx <= max_delta_idx; ++delta_idx) {
      const int angle_delta = (delta_idx == 0) ? 0 : 2;
      for (const auto &coord : kHbdCoords) {
        const int blk_col = coord[0];
        const int blk_row = coord[1];
        for (int edge_filt = 0; edge_filt <= 1; ++edge_filt) {
          memcpy(facade_buf.get(), buf.get(), 64 * kStride * sizeof(uint16_t));
          memset(direct_out.get(), 0, 16 * kStride * sizeof(uint16_t));

          memset(cm.get(), 0, sizeof(AV1_COMMON));
          SequenceHeader seq = {};
          seq.sb_size = BLOCK_64X64;
          seq.enable_intra_edge_filter = static_cast<uint8_t>(edge_filt);
          cm->seq_params = &seq;

          memset(xd.get(), 0, sizeof(MACROBLOCKD));
          YV12_BUFFER_CONFIG cur_buf = {};
          cur_buf.flags = YV12_FLAG_HIGHBITDEPTH;
          xd->cur_buf = &cur_buf;
          MB_MODE_INFO mbmi = {};
          mbmi.bsize = BLOCK_64X64;
          mbmi.partition = PARTITION_NONE;
          mbmi.mode = mode;
          mbmi.angle_delta[0] = static_cast<int8_t>(angle_delta);
          mbmi.filter_intra_mode_info.filter_intra_mode = FILTER_INTRA_MODES;
          MB_MODE_INFO *mbmi_ptr = &mbmi;
          xd->mi = &mbmi_ptr;
          xd->bd = 10;
          xd->up_available = true;
          xd->left_available = true;
          xd->tile.mi_row_end = 64;
          xd->tile.mi_col_end = 64;
          xd->mb_to_left_edge = -8 * 8;
          xd->mb_to_top_edge = -8 * 8;
          xd->mb_to_right_edge = 64 * 8;
          xd->mb_to_bottom_edge = 64 * 8;
          xd->plane[0].width = 64;
          xd->plane[0].height = 64;
          xd->plane[0].dst.stride = kStride;
          xd->plane[0].dst.buf =
              CONVERT_TO_BYTEPTR(facade_buf.get() + 16 * kStride + 16);

          av1_predict_intra_block_facade(cm.get(), xd.get(), AOM_PLANE_Y,
                                         blk_col, blk_row, TX_16X16);

          const uint16_t *ref16 = buf.get() +
                                  (16 + (blk_row << MI_SIZE_LOG2)) * kStride +
                                  16 + (blk_col << MI_SIZE_LOG2);
          av1_predict_intra_block(xd.get(), BLOCK_64X64, edge_filt, 64, 64,
                                  TX_16X16, mode, angle_delta * ANGLE_STEP, 0,
                                  FILTER_INTRA_MODES, CONVERT_TO_BYTEPTR(ref16),
                                  kStride, CONVERT_TO_BYTEPTR(direct_out.get()),
                                  kStride, blk_col, blk_row, AOM_PLANE_Y);

          const uint16_t *f_ptr =
              CONVERT_TO_SHORTPTR(xd->plane[0].dst.buf) +
              ((blk_row * kStride + blk_col) << MI_SIZE_LOG2);
          for (int r = 0; r < 16; ++r) {
            ASSERT_EQ(memcmp(&f_ptr[r * kStride], &direct_out[r * kStride],
                             16 * sizeof(uint16_t)),
                      0)
                << "HBD facade vs direct mode=" << static_cast<int>(mode)
                << " delta=" << angle_delta << " col=" << blk_col
                << " row=" << blk_row << " filt=" << edge_filt;
          }

          if (angle_delta == 0) {
            if (mode == V_PRED) {
              for (int r = 0; r < 16; ++r) {
                ASSERT_EQ(memcmp(&f_ptr[r * kStride], ref16 - kStride,
                                 16 * sizeof(uint16_t)),
                          0)
                    << "HBD V_PRED col=" << blk_col << " row=" << blk_row;
              }
            } else if (mode == H_PRED) {
              for (int r = 0; r < 16; ++r) {
                for (int c = 0; c < 16; ++c) {
                  ASSERT_EQ(f_ptr[r * kStride + c], ref16[r * kStride - 1])
                      << "HBD H_PRED col=" << blk_col << " row=" << blk_row;
                }
              }
            } else if (mode == DC_PRED) {
              int sum = 0;
              for (int c = 0; c < 16; ++c) sum += ref16[-kStride + c];
              for (int r = 0; r < 16; ++r) sum += ref16[r * kStride - 1];
              const uint16_t expected_dc =
                  static_cast<uint16_t>((sum + 16) >> 5);
              for (int r = 0; r < 16; ++r) {
                for (int c = 0; c < 16; ++c) {
                  ASSERT_EQ(f_ptr[r * kStride + c], expected_dc)
                      << "HBD DC_PRED col=" << blk_col << " row=" << blk_row;
                }
              }
            }
          }
        }
      }
    }
  }
}
#endif  // CONFIG_AV1_HIGHBITDEPTH

}  // namespace
