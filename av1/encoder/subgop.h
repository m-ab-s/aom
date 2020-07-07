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

#ifndef AOM_AV1_ENCODER_SUBGOP_H_

#include "av1/encoder/encoder.h"

#ifdef __cplusplus
extern "C" {
#endif

int av1_process_subgop_config_set(const char *param, SubGOPSetCfg *config_set);
void av1_print_subgop_config_set(SubGOPSetCfg *config_set);

// Finds the ptr to the subgop config with the queried number of
// frames and whether it is the last subgop in a gop.
// If the right number of frames is found but the right
// is_last_subgop is not found then the first config with the
// matching length is returned. If the right number of frames
// is not found, Null is returned.
const SubGOPCfg *av1_find_subgop_config(SubGOPSetCfg *config_set,
                                        int num_frames, int is_last_gop);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_ENCODER_SUBGOP_H_
