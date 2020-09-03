#
# Copyright (c) 2020, Alliance for Open Media. All rights reserved
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

# Checks if the dependencies on Tensorflow Lite are already checked out -- if
# not, uses the git submodule command to fetch them.
function(checkout_submodules_)
  # As a quick sanity check, see if at least 1 expected file or directory is
  # present in each submodule. If so, assume they are all checked out (if they
  # are not, then the base directory will be empty).
  if(
    (EXISTS "${AOM_ROOT}/third_party/tensorflow/tensorflow")
    AND (EXISTS
         "${AOM_ROOT}/third_party/tensorflow_dependencies/neon_2_sse/ReadMe.md")
    AND (EXISTS "${AOM_ROOT}/third_party/tensorflow_aom/absl/absl")
    AND (EXISTS "${AOM_ROOT}/third_party/tensorflow_aom/eigen/Eigen")
    AND (EXISTS "${AOM_ROOT}/third_party/tensorflow_aom/farmhash/Makefile.am")
    AND (EXISTS "${AOM_ROOT}/third_party/tensorflow_aom/flatbuffers/BUILD")
    AND (EXISTS "${AOM_ROOT}/third_party/tensorflow_aom/fp16/CMakeLists.txt")
    AND (EXISTS "${AOM_ROOT}/third_party/tensorflow_aom/gemmlowp/BUILD")
    AND (EXISTS "${AOM_ROOT}/third_party/tensorflow_aom/ruy/BUILD"))
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
  message("Checking out Tensorflow-Lite submodule")
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
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/libtensorflow-lite.a"
    COMMAND "${AOM_ROOT}/third_party/tensorflow_dependencies/build.pl"
            "${AOM_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
  add_custom_target(tensorflowlite_a ALL
                    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/libtensorflow-lite.a")
  include_directories("${AOM_ROOT}/third_party/tensorflow")
  include_directories(
    "${AOM_ROOT}/third_party/tensorflow_dependencies/flatbuffers/include/")
  # Add tensorflow-lite as a dependency on all AOM applications.
  foreach(aom_app ${AOM_APP_TARGETS})
    add_dependencies(${aom_app} tensorflowlite_a)
    target_link_libraries(
      ${aom_app}
      PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/libtensorflow-lite.a"
              ${AOM_LIB_LINK_TYPE} Threads::Threads
      PRIVATE ${CMAKE_DL_LIBS})
  endforeach()
endfunction()

# If Tensorflow-Lite should be enabled, adds appropriate build rules / targets.
function(setup_tensorflow_lite)
  if("${AOM_ROOT}" STREQUAL "")
    message(FATAL_ERROR "AOM_ROOT variable must not be empty.")
  endif()
  # Cross-compile is not currently implemented.
  if(CMAKE_TOOLCHAIN_FILE)
    message("TOOLCHAIN: ${CMAKE_TOOLCHAIN_FILE}")
    message(WARNING "No cross-compile support for TensorFlow Lite; disabling")
    set(CONFIG_TENSORFLOW_LITE 0 PARENT_SCOPE)
    return()
  endif()
  checkout_submodules_()
  add_tensorflow_lite_dependency_()
endfunction()

function(add_tf_lite_dependency experiment_name)
  # Experiment is not enabled, no need to include TF-Lite in the build.
  if(${${experiment_name}} EQUAL "0")
    return()
  endif()
  # Disable the experiment so Gerrit will not test this case.
  if(CMAKE_TOOLCHAIN_FILE)
    message(WARNING "--- Cross-compile support not implemented for TF-Lite. "
                    "Disabling ${experiment_name}.")
    set(${experiment_name} 0 PARENT_SCOPE)
    return()
  endif()
  if(NOT CONFIG_TENSORFLOW_LITE)
    set(CONFIG_TENSORFLOW_LITE 1 PARENT_SCOPE)
  endif()
endfunction()
