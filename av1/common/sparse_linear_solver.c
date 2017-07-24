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

#include "av1/common/sparse_linear_solver.h"
#include "./aom_config.h"
#include "av1/common/alloccommon.h"
#include "av1/common/onyxc_int.h"
#include "aom_mem/aom_mem.h"
#include <float.h>

#if CONFIG_OPFL

/*
 * Input:
 * rows: array of row positions
 * cols: array of column positions
 * values: array of element values
 * num_elem: total number of elements in the matrix
 * num_rows: number of rows in the matrix
 * num_cols: number of columns in the matrix
 *
 * Output:
 * sm: pointer to the sparse matrix to be initialized
 */
void init_sparse_mtx(int *rows, int *cols, double *values, int num_elem,
                     int num_rows, int num_cols, SPARSE_MTX *sm) {
  sm->n_elem = num_elem;
  sm->n_rows = num_rows;
  sm->n_cols = num_cols;
  if (num_elem == 0) {
    sm->row_pos = NULL;
    sm->col_pos = NULL;
    sm->value = NULL;
    return;
  }
  sm->row_pos = aom_calloc(num_elem, sizeof(int));
  sm->col_pos = aom_calloc(num_elem, sizeof(int));
  sm->value = aom_calloc(num_elem, sizeof(double));

  memcpy(sm->row_pos, rows, num_elem * sizeof(int));
  memcpy(sm->col_pos, cols, num_elem * sizeof(int));
  memcpy(sm->value, values, num_elem * sizeof(double));
}

/*
 * Combines two sparse matrices (allocating new space).
 *
 * Input:
 * sm1, sm2: matrices to be combined
 * row_offset1, row_offset2: row offset of each matrix in the new matrix
 * col_offset1, col_offset2: column offset of each matrix in the new matrix
 * new_n_rows, new_n_cols: number of rows and columns in the new matrix
 *
 * Output:
 * sm: the combined matrix
 */
void init_combine_sparse_mtx(SPARSE_MTX *sm1, SPARSE_MTX *sm2, SPARSE_MTX *sm,
                             int row_offset1, int col_offset1, int row_offset2,
                             int col_offset2, int new_n_rows, int new_n_cols) {
  sm->n_elem = sm1->n_elem + sm2->n_elem;
  sm->n_cols = new_n_cols;
  sm->n_rows = new_n_rows;

  if (sm->n_elem == 0) {
    sm->row_pos = NULL;
    sm->col_pos = NULL;
    sm->value = NULL;
    return;
  }
  sm->row_pos = aom_calloc(sm->n_elem, sizeof(int));
  sm->col_pos = aom_calloc(sm->n_elem, sizeof(int));
  sm->value = aom_calloc(sm->n_elem, sizeof(double));

  for (int i = 0; i < sm1->n_elem; i++) {
    sm->row_pos[i] = sm1->row_pos[i] + row_offset1;
    sm->col_pos[i] = sm1->col_pos[i] + col_offset1;
  }
  memcpy(sm->value, sm1->value, sm1->n_elem * sizeof(double));
  int n_elem1 = sm1->n_elem;
  for (int i = 0; i < sm2->n_elem; i++) {
    sm->row_pos[n_elem1 + i] = sm2->row_pos[i] + row_offset2;
    sm->col_pos[n_elem1 + i] = sm2->col_pos[i] + col_offset2;
  }
  memcpy(sm->value + n_elem1, sm2->value, sm2->n_elem * sizeof(double));
}

void free_sparse_mtx_elems(SPARSE_MTX *sm) {
  sm->n_cols = 0;
  sm->n_rows = 0;
  if (sm->n_elem != 0) {
    aom_free(sm->row_pos);
    aom_free(sm->col_pos);
    aom_free(sm->value);
  }
  sm->n_elem = 0;
}


#endif  // CONFIG_OPFL
