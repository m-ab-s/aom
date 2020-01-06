#
# Copyright (c) 2019, Alliance for Open Media. All rights reserved
#
# This source code is subject to the terms of the BSD 2 Clause License and the
# Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License was
# not distributed with this source code in the LICENSE file, you can obtain it
# at www.aomedia.org/license/software. If the Alliance for Open Media Patent
# License 1.0 was not distributed with this source code in the PATENTS file, you
# can obtain it at www.aomedia.org/license/patent.
#
if(AOM_BUILD_CMAKE_TENSORFLOW_LITE_CMAKE_)
  return()
endif() # AOM_BUILD_CMAKE_TENSORFLOW_LITE_CMAKE_
set(AOM_BUILD_CMAKE_TENSORFLOW_LITE_CMAKE_ 1)

include(FindGit)

# Checks if the dependencies on Tensorflow-Lite are already checked out -- if
# not, uses the git submodule command to fetch them.
function(checkout_submodules_)
  # As a quick sanity check, see if at least 1 expected file or directory is
  # present in each submodule. If so, assume they are all checked out (if they
  # are not, then the base directory will be empty).
  if((EXISTS "${AOM_ROOT}/third_party/tensorflow/tensorflow")
     AND (EXISTS
          "${AOM_ROOT}/third_party/tensorflow_aom/ARM_NEON_2_x86_SSE/ReadMe.md")
     AND (EXISTS "${AOM_ROOT}/third_party/tensorflow_aom/abseil-cpp/absl")
     AND (EXISTS "${AOM_ROOT}/third_party/tensorflow_aom/eigen/Eigen")
     AND (EXISTS "${AOM_ROOT}/third_party/tensorflow_aom/farmhash/Makefile.am")
     AND (EXISTS "${AOM_ROOT}/third_party/tensorflow_aom/flatbuffers/BUILD")
     AND (EXISTS "${AOM_ROOT}/third_party/tensorflow_aom/gemmlowp/BUILD"))
    return()
  endif()

  if(NOT GIT_FOUND)
    message(
      FATAL_ERROR
        "Tensorflow-Lite/dependencies not present; " "git could not be found; "
        "please check out submodules with 'git submodule update --init'")
  endif()
  # Note that "git submodule update --init" must be run from inside the git
  # repository; the --git-dir flag does not work.
  message("Checking out Tensorflow-Lite and dependencies")
  execute_process(COMMAND "${GIT_EXECUTABLE}" submodule update --init
                  WORKING_DIRECTORY "${AOM_ROOT}"
                  OUTPUT_VARIABLE submodule_out
                  ERROR_VARIABLE submodule_err
                  RESULT_VARIABLE submodule_result)
  if(NOT ${submodule_result} EQUAL 0)
    message(FATAL_ERROR "Unable to run 'git submodule update --init': "
                        "Return code: " ${submodule_result} ", STDOUT: "
                        ${submodule_out} ", STDERR: " ${submodule_err})
  endif()
endfunction()

function(add_tensorflow_lite_dependency_)
  if(NOT AOM_APP_TARGETS)
    message(FATAL_ERROR "AOM_APP_TARGETS variable must not be empty.")
  endif()
  # Build the library.
  add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/libtensorflow-lite.a"
                     COMMAND "${AOM_ROOT}/third_party/tensorflow_aom/build.sh"
                             "${AOM_ROOT}/third_party/tensorflow"
                             "${CMAKE_CURRENT_BINARY_DIR}/libtensorflow-lite.a"
                     WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
  add_custom_target(tensorflowlite_a ALL
                    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/libtensorflow-lite.a")
  include_directories("${AOM_ROOT}/third_party/tensorflow")
  include_directories(
    "${AOM_ROOT}/third_party/tensorflow_aom/flatbuffers/include/")
  # Add tensorflow-lite as a dependency on libaom.
  add_dependencies(aom tensorflowlite_a)
  target_link_libraries(aom ${AOM_LIB_LINK_TYPE}
                        "${CMAKE_CURRENT_BINARY_DIR}/libtensorflow-lite.a")
  target_link_libraries(aom PRIVATE Threads::Threads)
endfunction()

function(add_tensorflow_lite_example_)
  if(NOT AOM_EXAMPLE_TARGETS)
    message(FATAL_ERROR "AOM_EXAMPLE_TARGETS variable must not be empty.")
  endif()
  add_executable(tf_lite_model
                 "${AOM_ROOT}/examples/tf_lite_model.cc"
                 "${AOM_ROOT}/examples/tf_lite_model_data.cc"
                 "${AOM_ROOT}/examples/tf_lite_ops_registration.cc")
  list(APPEND AOM_EXAMPLE_TARGETS tf_lite_model)
  add_dependencies(tf_lite_model tensorflowlite_a)
  target_link_libraries(
    tf_lite_model
    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/libtensorflow-lite.a")
  target_link_libraries(tf_lite_model PRIVATE Threads::Threads)
endfunction()

# If Tensorflow-Lite should be enabled, adds appropriate build rules / targets.
function(setup_tensorflow_lite)
  if("${AOM_ROOT}" STREQUAL "")
    message(FATAL_ERROR "AOM_ROOT variable must not be empty.")
  endif()
  # Cross-compile is not currently implemented.
  if(CMAKE_TOOLCHAIN_FILE)
    message(WARNING "No cross-compile support for TensorFlow-Lite; disabling")
    set(CONFIG_TENSORFLOW_LITE 0)
  endif()

  # Tensorflow-Lite not configured -- nothing to do.
  if(NOT CONFIG_TENSORFLOW_LITE)
    return()
  endif()
  checkout_submodules_()
  add_tensorflow_lite_dependency_()
  if(ENABLE_EXAMPLES)
    add_tensorflow_lite_example_()
  endif()
endfunction()
