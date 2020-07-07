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

#include <stdio.h>
#include <string.h>
#include "av1/encoder/subgop.h"

static char *my_strtok_r(char *str, const char *delim, char **saveptr) {
  if (str == NULL) return NULL;
  if (strlen(str) == 0) return NULL;
  char *ptr = str;
  char *x = strstr(str, delim);
  if (x) {
    *x = 0;
    *saveptr = x + strlen(delim);
  } else {
    *saveptr = NULL;
    return ptr;
  }
  return ptr;
}

static void init_subgop_config_set(SubGOPSetCfg *config_set) {
  memset(config_set, 0, sizeof(*config_set));
  config_set->num_configs = 0;
  for (int i = 0; i < MAX_SUBGOP_CONFIGS; ++i) {
    memset(&config_set->config[i], 0, sizeof(config_set->config[i]));
  }
}

static int process_subgop_step(char *str, SubGOPStepCfg *step) {
  char *ptr;
  step->num_references = 0;
  step->disp_frame_idx = (int8_t)strtol(str, &ptr, 10);
  // Check if no numeric disp idx exist or is negative
  if (ptr == str || step->disp_frame_idx < 0) return 0;
  switch (*ptr) {
    case 'V': step->type_code = FRAME_TYPE_INO_VISIBLE; break;
    case 'R': step->type_code = FRAME_TYPE_INO_REPEAT; break;
    case 'S': step->type_code = FRAME_TYPE_INO_SHOWEXISTING; return 1;
    case 'F': step->type_code = FRAME_TYPE_OOO_FILTERED; break;
    case 'U': step->type_code = FRAME_TYPE_OOO_UNFILTERED; break;
    default: return 0;
  }
  str = ++ptr;
  step->pyr_level = strtol(str, &ptr, 10);
  // Check if no numeric disp idx exist
  if (ptr == str) return 0;
  // Check for character P preceding the references
  if (*ptr != 'P') return 0;
  str = ++ptr;
  char delim[] = "^";
  char *token;
  while ((token = my_strtok_r(str, delim, &str)) != NULL) {
    if (step->num_references >= INTER_REFS_PER_FRAME) return 0;
    step->references[step->num_references] = (int8_t)strtol(token, NULL, 10);
    step->num_references++;
  }
  return 1;
}

static int process_subgop_steps(char *str, SubGOPCfg *config) {
  char delim[] = "/";
  config->num_steps = 0;
  char *token;
  while ((token = my_strtok_r(str, delim, &str)) != NULL) {
    int res = process_subgop_step(token, &config->step[config->num_steps]);
    if (res)
      config->num_steps++;
    else
      return 0;
  }
  return 1;
}

static int process_subgop_config(char *str, SubGOPCfg *config) {
  char delim[] = ":";
  char *token = my_strtok_r(str, delim, &str);
  if (!token) return 0;
  config->num_frames = atoi(token);
  token = my_strtok_r(str, delim, &str);
  if (!token) return 0;
  config->is_last_subgop = atoi(token);
  token = my_strtok_r(str, delim, &str);
  if (!token) return 0;
  return process_subgop_steps(token, config);
}

static int check_subgop_config(SubGOPCfg *config) {
  for (int s = 0; s < config->num_steps; ++s) {
    // check for invalid disp_frame_idx
    if (config->step[s].disp_frame_idx > config->num_frames) return 0;
  }
  return 1;
}

int av1_process_subgop_config_set(const char *param, SubGOPSetCfg *config_set) {
  init_subgop_config_set(config_set);
  if (!param) return 1;
  if (!strlen(param)) return 1;
  const int bufsize = (int)((strlen(param) + 1) * sizeof(*param));
  char *buf = (char *)aom_malloc(bufsize);
  memcpy(buf, param, bufsize);
  char delim[] = ",";

  char *str = buf;
  char *token;
  while ((token = my_strtok_r(str, delim, &str)) != NULL) {
    int res = process_subgop_config(
        token, &config_set->config[config_set->num_configs]);
    if (res) {
      res = check_subgop_config(&config_set->config[config_set->num_configs]);
      if (res)
        config_set->num_configs++;
      else
        return 0;
    } else {
      return 0;
    }
  }
  aom_free(buf);
  return 1;
}

void av1_print_subgop_config_set(SubGOPSetCfg *config_set) {
  if (!config_set->num_configs) return;
  printf("num_configs: %d\n", config_set->num_configs);
  for (int i = 0; i < config_set->num_configs; ++i) {
    printf("config: %d ->\n", i);
    SubGOPCfg *config = &config_set->config[i];
    printf("  num_frames: %d\n", config->num_frames);
    printf("  is_last_subgop: %d\n", config->is_last_subgop);
    printf("  num_steps: %d\n", config->num_steps);
    for (int j = 0; j < config->num_steps; ++j) {
      printf("  step %d -> ", j);
      printf("disp_frame_idx: %d, ", config->step[j].disp_frame_idx);
      printf("type_code: %d, ", config->step[j].type_code);
      printf("pyr_level: %d, ", config->step[j].pyr_level);
      printf("references:");
      for (int r = 0; r < config->step[j].num_references; ++r)
        printf(" %d", config->step[j].references[r]);
      printf("\n");
    }
  }
}

const SubGOPCfg *av1_find_subgop_config(SubGOPSetCfg *config_set,
                                        int num_frames, int is_last_subgop) {
  SubGOPCfg *cfg = NULL;
  int j = -1;
  for (int i = 0; i < config_set->num_configs; ++i) {
    if (config_set->config[i].num_frames == num_frames) {
      if (config_set->config[i].is_last_subgop == is_last_subgop)
        return &config_set->config[i];
      else
        cfg = &config_set->config[j];
    }
  }
  return cfg;
}
