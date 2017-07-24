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

#ifndef AV1_COMMON_SPARSE_LINEAR_SOLVER_H_
#define AV1_COMMON_SPARSE_LINEAR_SOLVER_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "./aom_config.h"

#if CONFIG_OPFL

#define MAX_CG_SP_ITER 100

typedef struct {
  int n_elem;  // number of non-zero elements
  int n_rows;
  int n_cols;
  // using arrays to represent non-zero elements.
  int *col_pos;
  int *row_pos;  // starts with 0
  double *value;
} SPARSE_MTX;

void init_sparse_mtx(int *rows, int *cols, double *values, int num_elem,
                     int num_rows, int num_cols, SPARSE_MTX *sm);
void init_combine_sparse_mtx(SPARSE_MTX *sm1, SPARSE_MTX *sm2, SPARSE_MTX *sm,
                             int row_offset1, int col_offset1, int row_offset2,
                             int col_offset2, int new_n_rows, int new_n_cols);
void free_sparse_mtx_elems(SPARSE_MTX *sm);

void mtx_vect_multi_right(SPARSE_MTX *sm, double *srcv, double *dstv, int dstl);
double vect_vect_multi(double *src1, int src1l, double *src2);
void constant_multiply_sparse_matrix(SPARSE_MTX *sm, double c);

void conjugate_gradient_sparse(SPARSE_MTX *A, double *b, int bl, double *x);

#endif  // CONFIG_OPFL

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* AV1_COMMON_SPARSE_LINEAR_SOLVER_H_ */
