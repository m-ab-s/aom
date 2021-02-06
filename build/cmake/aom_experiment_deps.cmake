#
# Copyright (c) 2017, Alliance for Open Media. All rights reserved
#
# This source code is subject to the terms of the BSD 2 Clause License and the
# Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License was
# not distributed with this source code in the LICENSE file, you can obtain it
# at www.aomedia.org/license/software. If the Alliance for Open Media Patent
# License 1.0 was not distributed with this source code in the PATENTS file, you
# can obtain it at www.aomedia.org/license/patent.
#
if(AOM_BUILD_CMAKE_AOM_EXPERIMENT_DEPS_CMAKE_)
  return()
endif() # AOM_BUILD_CMAKE_AOM_EXPERIMENT_DEPS_CMAKE_
set(AOM_BUILD_CMAKE_AOM_EXPERIMENT_DEPS_CMAKE_ 1)

# Adjusts CONFIG_* CMake variables to address conflicts between active AV2
# experiments.
macro(fix_experiment_configs)

  if(CONFIG_ANALYZER)
    change_config_and_warn(CONFIG_INSPECTION 1 CONFIG_ANALYZER)
  endif()

  if(CONFIG_DIST_8X8 AND CONFIG_MULTITHREAD)
    change_config_and_warn(CONFIG_DIST_8X8 0 CONFIG_MULTITHREAD)
  endif()

  if(CONFIG_LPF_MASK AND CONFIG_FLEX_PARTITION)
    change_config_and_warn(CONFIG_LPF_MASK 0 CONFIG_FLEX_PARTITION)
  endif()

  if(CONFIG_OPTFLOW_REFINEMENT)
    change_config_and_warn(CONFIG_NEW_INTER_MODES 1 CONFIG_OPTFLOW_REFINEMENT)
  endif()

  if(CONFIG_SB_FLEX_MVRES)
    change_config_and_warn(CONFIG_FLEX_MVRES 1 CONFIG_SB_FLEX_MVRES)
  endif()

  if(CONFIG_PB_FLEX_MVRES)
    change_config_and_warn(CONFIG_FLEX_MVRES 1 CONFIG_PB_FLEX_MVRES)
  endif()

  if(CONFIG_LGT32)
    change_config_and_warn(CONFIG_DST_32X32 1 CONFIG_LGT32)
  endif()

  if(CONFIG_MODE_DEP_NONSEP_SEC_INTRA_TX)
    change_config_and_warn(CONFIG_MODE_DEP_NONSEP_INTRA_TX 1
                           CONFIG_MODE_DEP_NONSEP_SEC_INTRA_TX)
  endif()

  if(CONFIG_SUPERRES_TX64_DIRFILTER)
    change_config_and_warn(CONFIG_SUPERRES_TX64 1
                           CONFIG_SUPERRES_TX64_DIRFILTER)
  endif()

  if(CONFIG_WIENER_NONSEP_CROSS_FILT)
    change_config_and_warn(CONFIG_WIENER_NONSEP 1
                           CONFIG_WIENER_NONSEP_CROSS_FILT)
  endif()

  if(CONFIG_NN_RECON)
    change_config_and_warn(CONFIG_TENSORFLOW_LITE 1 CONFIG_NN_RECON)
  endif()

  if(CONFIG_CNN_RESTORATION_SMALL_MODELS)
    if(NOT (CONFIG_CNN_RESTORATION OR CONFIG_LOOP_RESTORE_CNN))
      change_config_and_warn(CONFIG_CNN_RESTORATION 1
                             CONFIG_CNN_RESTORATION_SMALL_MODELS)
    endif()
  endif()

  if(CONFIG_CNN_RESTORATION)
    change_config_and_warn(CONFIG_TENSORFLOW_LITE 1 CONFIG_CNN_RESTORATION)
  endif()

  if(CONFIG_LOOP_RESTORE_CNN)
    change_config_and_warn(CONFIG_TENSORFLOW_LITE 1 CONFIG_LOOP_RESTORE_CNN)
  endif()

  if(CONFIG_CNN_CRLC_GUIDED)
    if(NOT CONFIG_CNN_RESTORATION)
      change_config_and_warn(CONFIG_CNN_RESTORATION 1 CONFIG_CNN_CRLC_GUIDED)
    endif()
  endif()
endmacro()
