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

// TensorFlow Lite uses dlsym and dlopen when using delegates,
// e.g., GPU or TPU processing. Static builds of the AOM encoder
// do not support linking with -ldl. Define dummy functions
// to allow linking. Do not use delegation with TensorFlow Lite.

#include <cassert>
#include <cstddef>

#include "common/fake_dl.h"

void *dlopen(const char *filename, int flags) {
  (void)filename;
  (void)flags;
  assert(false);
  return NULL;
}

void *dlsym(void *handle, const char *symbol) {
  (void)handle;
  (void)symbol;
  assert(false);
  return NULL;
}
