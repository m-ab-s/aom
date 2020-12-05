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

#include <math.h>
#include "av1/encoder/pickrst.h"
#include "third_party/vector/vector.h"
#include "third_party/googletest/src/googletest/include/gtest/gtest.h"

namespace {

double cost_fn(const void *info, Vector *path, int node_idx, int max_out_nodes,
               int out_edge) {
  (void)path;
  double *graph = (double *)info;
  return graph[node_idx * max_out_nodes + out_edge];
}

TEST(AV1RstMergeCoeffsTest, TestNoSubsetGraphSearch) {
  // Initialize graph.
  double graph[25];
  for (int i = 0; i < 25; ++i) {
    graph[i] = INFINITY;
  }
  graph[1] = -1;
  graph[7] = -2;
  graph[10] = -3;
  graph[15] = -1;
  graph[17] = -1;
  graph[23] = 2;
  // Initialize best_path vector.
  Vector best_path;
  aom_vector_setup(&best_path, 1, sizeof(int));

  double cost = min_cost_path(4,  // start of path
                              1,  // dest of path
                              5,  // max_out, # of nodes
                              graph, &best_path,
                              cost_fn,  // cost function
                              graph);   // information for cost

  // Verify results.
  int correct_path[] = { 4, 3, 2, 0, 1 };
  int i = 0;
  VECTOR_FOR_EACH(&best_path, listed_unit) {
    int node = *(int *)(listed_unit.pointer);
    EXPECT_EQ(correct_path[i], node);
    ++i;
  }
  EXPECT_EQ(cost, -3);
  aom_vector_destroy(&best_path);
}

TEST(AV1RstMergeCoeffsTest, TestNoSubsetStartEqualsEndGraphSearch) {
  // Initialize graph.
  double graph[25];
  for (int i = 0; i < 25; ++i) {
    graph[i] = INFINITY;
  }
  graph[1] = -1;
  graph[7] = -2;
  graph[10] = -3;
  graph[15] = -1;
  graph[17] = -1;
  graph[23] = 2;
  // Initialize best_path vector.
  Vector best_path;
  aom_vector_setup(&best_path, 1, sizeof(int));

  double cost = min_cost_path(4,  // start of path
                              4,  // dest of path
                              5,  // max_out, # of nodes
                              graph, &best_path,
                              cost_fn,  // cost function
                              graph);   // information for cost

  // Verify results.
  int correct_path[] = { 4 };
  int i = 0;
  VECTOR_FOR_EACH(&best_path, listed_unit) {
    int node = *(int *)(listed_unit.pointer);
    EXPECT_EQ(correct_path[i], node);
    ++i;
  }
  EXPECT_EQ(cost, 0);
  aom_vector_destroy(&best_path);
}

TEST(AV1RstMergeCoeffsTest, TestNoSubsetCyclicGraphSearch) {
  // Initialize graph.
  double graph[25];
  for (int i = 0; i < 25; ++i) {
    graph[i] = INFINITY;
  }
  graph[1] = -1;
  graph[2] = -1;  // cycle between 0 and 2 with graph[10]
  graph[3] = 1;
  graph[7] = -2;
  graph[10] = -3;
  graph[17] = -1;
  graph[22] = 2;
  graph[23] = 2;
  // Initialize best_path vector.
  Vector best_path;
  aom_vector_setup(&best_path, 1, sizeof(int));

  double cost = min_cost_path(4,  // start of path
                              3,  // dest of path
                              5,  // max_out, # of nodes
                              graph, &best_path,
                              cost_fn,  // cost function
                              graph);   // information for cost

  // Verify results.
  int correct_path[] = { 4, 2, 0, 3 };
  int i = 0;
  VECTOR_FOR_EACH(&best_path, listed_unit) {
    int node = *(int *)(listed_unit.pointer);
    EXPECT_EQ(correct_path[i], node);
    ++i;
  }
  EXPECT_EQ(cost, 0);
  aom_vector_destroy(&best_path);
}

TEST(AV1RstMergeCoeffsTest, TestSubsetGraphSearch) {
  // Initialize graph.
  // Columns are cost to first, second, third outgoing edge of node.
  double graph[] = {
    8,        8,        8,         // src
    8,        6,        7,         // subset 1, type 1
    8,        6,        7,         // subset 1, type 2
    8,        6,        7,         // subset 1, type 3
    1,        6,        8,         // subset 2, type 1
    8,        6,        8,         // subset 2, type 2
    8,        6,        8,         // subset 2, type 3
    1,        8,        8,         // subset 3, type 1
    8,        8,        8,         // subset 3, type 2
    8,        8,        1,         // subset 3, type 3
    8,        7,        8,         // subset 4, type 1
    8,        7,        8,         // subset 4, type 2
    8,        7,        1,         // subset 4, type 3
    8,        6,        8,         // subset 5, type 1
    8,        6,        8,         // subset 5, type 2
    8,        6,        1,         // subset 5, type 3
    0,        INFINITY, INFINITY,  // subset 6, type 1 - only to dest
    0,        INFINITY, INFINITY,  // subset 6, type 2 - only to dest
    0,        INFINITY, INFINITY,  // subset 6, type 3 - only to dest
    INFINITY, INFINITY, INFINITY   // dest
  };
  // Initialize best_path vector.
  Vector best_path;
  aom_vector_setup(&best_path, 1, sizeof(int));

  double cost = min_cost_type_path(0,   // start of path - src
                                   19,  // dest of path - dest
                                   3,   // max_out, # of types
                                   graph, &best_path,
                                   cost_fn,  // cost function
                                   graph);   // information for cost

  // Verify results.
  int correct_path[] = { 0, 1, 5, 9, 12, 15, 18, 19 };
  int i = 0;
  VECTOR_FOR_EACH(&best_path, listed_unit) {
    int node = *(int *)(listed_unit.pointer);
    EXPECT_EQ(correct_path[i], node);
    ++i;
  }
  EXPECT_EQ(cost, 25);
  aom_vector_destroy(&best_path);
}

}  // namespace
