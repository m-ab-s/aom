/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */
#ifndef AOM_AV1_ENCODER_PICKRST_H_
#define AOM_AV1_ENCODER_PICKRST_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "av1/encoder/encoder.h"
#include "aom_ports/system_state.h"

struct yv12_buffer_config;
struct AV1_COMP;

static const uint8_t g_shuffle_stats_data[16] = {
  0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8,
};

static const uint8_t g_shuffle_stats_highbd_data[32] = {
  0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9,
  0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9,
};

static INLINE uint8_t find_average(const uint8_t *src, int h_start, int h_end,
                                   int v_start, int v_end, int stride) {
  uint64_t sum = 0;
  for (int i = v_start; i < v_end; i++) {
    for (int j = h_start; j < h_end; j++) {
      sum += src[i * stride + j];
    }
  }
  uint64_t avg = sum / ((v_end - v_start) * (h_end - h_start));
  return (uint8_t)avg;
}

static INLINE uint16_t find_average_highbd(const uint16_t *src, int h_start,
                                           int h_end, int v_start, int v_end,
                                           int stride) {
  uint64_t sum = 0;
  for (int i = v_start; i < v_end; i++) {
    for (int j = h_start; j < h_end; j++) {
      sum += src[i * stride + j];
    }
  }
  uint64_t avg = sum / ((v_end - v_start) * (h_end - h_start));
  return (uint16_t)avg;
}

void av1_pick_filter_restoration(const YV12_BUFFER_CONFIG *sd,
#if CONFIG_LOOP_RESTORE_CNN
                                 bool allow_restore_cnn_y,
                                 bool allow_restore_cnn_uv,
#endif  // CONFIG_LOOP_RESTORE_CNN
                                 AV1_COMP *cpi);

// Function type to determine edge cost
// info : pointer to unspecified structure type, cast in function, holds any
//  information needed to calculate edge cost
// path : pointer to Vector holding current path to edge represented as int
//  indexes of nodes
// node_idx : node where path ends and edge starts
// max_out_nodes: max outgoing edges from node
// out_edge: outgoing edge we are calculating cost for
// Returns cost of edge.
typedef double (*graph_edge_cost_t)(const void *info, Vector *path,
                                    int node_idx, int max_out_nodes,
                                    int out_edge);

// Exposed for testing purposes.
// src_idx : start of path
// dest_idx : destination of path
// best_path : pointer to Vector storing best path from start to destination
//  as int indexes of nodes
// cost_fn : function to dynamically determine edge cost
// info : pointer to unspecified structure type cast in function, holds any
//  information needed to calculate edge cost
// Returns cost of min-cost path.

// Searching a directed graph for min cost path between node_idx and dest_idx.
// max_out_nodes: number of nodes in graph
// graph: n*n matrix where n is the number of nodes and element
//  graph[n1 * n + n2] is an edge from n1 to n2
double min_cost_path(int src_idx, int dest_idx, int max_out_nodes,
                     const double *graph, Vector *best_path,
                     graph_edge_cost_t cost_fn, const void *info);

// Nodes are organized into subsets of equal size - each subset will have
//  one node of every type.
//       | /A/ |   | /A/ |   | /A/ |   | /A/ |
//  src  | /B/ |   | /B/ |   | /B/ |   | /B/ |  dst
//       | /C/ |   | /C/ |   | /C/ |   | /C/ |
//          1         2         3         4
// Every node has outgoing edges to every node in the next subset, i.e.
//  1A, 1B, and 1C will all have outgoing edges to 2A, 2B, and 2C. We are
//  finding the type in each subset that will give us a min cost path from
//  src to dest.
// max_out_nodes: number of nodes in a subset
// graph: a ((# of subsets * max_out_nodes + 2 nodes for src and dest) *
//  max_out_nodes) matrix where element graph[n * max_out_nodes + e] is an edge
//  from n to the eth node in the next subset. Last subset will only have one
//  outgoing edge from each node to dst, dst has no outgoing edges.
//    A B-Z
//  [ x x x  // src
//    x x x  // subset 1, type A
//    x x x  // subset 1, type B
//     ...
//    x x x  // subset 2, type A
//    x x x  // subset 2, type B
//     ...
//    x I I  // subset x, type A
//    x I I  // subset x, type B
//     ...
//    x I I  // subset x, type Z
//    I I I] // dst
double min_cost_type_path(int src_idx, int dest_idx, int max_out_nodes,
                          const double *graph, Vector *best_path,
                          graph_edge_cost_t cost_fn, const void *info);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_ENCODER_PICKRST_H_
