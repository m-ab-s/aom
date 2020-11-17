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

#ifndef AOM_COMMON_TF_LITE_INCLUDES_H_
#define AOM_COMMON_TF_LITE_INCLUDES_H_

#include <assert.h>

// TensorFlow Lite has several unused parameters that are
// exposed as part of the API. In the AOM build process, this
// will cause failures when -Wunused-parameter is set.
// Since TF Lite is external code, instruct the compiler to
// ignore this warning when including it.
// Note that Clang supports this GCC pragma.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "tensorflow/lite/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/op_resolver.h"

#pragma GCC diagnostic pop

// TensorFlow Lite uses dlsym and dlopen when using delegates,
// e.g., GPU or TPU processing. Static builds of the AOM encoder
// do not support linking with -ldl. Define dummy functions
// to allow linking. Do not use delegation with TensorFlow Lite.

#ifdef __cplusplus
extern "C" {
#endif

void *dlopen(const char *filename, int flags) {
  (void)filename;
  (void)flags;
  assert(0);
  return NULL;
}

void *dlsym(void *handle, const char *symbol) {
  (void)handle;
  (void)symbol;
  assert(0);
  return NULL;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_COMMON_TF_LITE_INCLUDES_H_
