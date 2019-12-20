#!/bin/bash
#
# Copyright (c) 2019, Alliance for Open Media. All rights reserved
#
# This source code is subject to the terms of the BSD 2 Clause License and
# the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
# was not distributed with this source code in the LICENSE file, you can
# obtain it at www.aomedia.org/license/software. If the Alliance for Open
# Media Patent License 1.0 was not distributed with this source code in the
# PATENTS file, you can obtain it at www.aomedia.org/license/patent.
#
###########################################################################
#
# Script to build the static TensorFlow lite library.
#
# Tensorflow's build process generates the static libraries in the same
# directory as the source code. AOM, however, generates binaries/objects/etc.
# in a different directory. This script:
#
# 1.) Copies the Tensorflow code to a temporary directory
# 2.) Copies the necessary dependencies into the temporary directory
# 2.) Compiles it there
# 3.) Copies over the TensorFlow lite static library
#
# Note that we do not use the download_dependencies.sh script directly, as
# it does not perform checksumming on the downloaded code, and it downloads
# directly into the source directory.
set -uxeo pipefail

readonly temp_dir=$(mktemp -d -t tensorflowlite.XXXXXXXX)

# PID of the subprocess that prints "Building TF-Lite...". If it is -1, it
# means that it is either uninitialized or already killed.
repeat_print_pid="-1"

# Delete the temporary directory used for compiling TensorFlow-Lite.
function cleanup() {
  rm -rf "${temp_dir}"
  # If the repeating loop that prints "Building TF-Lite..." is running,
  # stop it.
  if [[ "${repeat_print_pid}" != "-1" ]]; then
    kill "${repeat_print_pid}"
  fi
}

trap cleanup EXIT INT TERM QUIT

# FFT2D is a library used by TensorFlow-Lite, but it is not in a git
# repository. TF-Lite uses a script, "download_dependencies.sh", to retrieve
# it, but does not checksum / validate the download. We use the same URL
# (taken from tensorflow/workspace.bzl) to retrieve it. It is not checked in,
# as the Gerrit style checker complains too much.
function download_fft2d() {
  curl --location --silent "https://storage.googleapis.com/mirror.tensorflow.org/www.kurims.kyoto-u.ac.jp/~ooura/fft2d.tgz" \
  --output "${temp_make_dir}/downloads/fft2d.tgz"
  checksum=$(sha256sum "${temp_make_dir}/downloads/fft2d.tgz" | cut -d ' ' -f 1)
  if [[ ${checksum} != "ada7e99087c4ed477bfdf11413f2ba8db8a840ba9bbf8ac94f4f3972e2a7cec9" ]]; then
    echo "Bad checksum for fft2d.tgz: ${checksum}"
    exit 1
  fi
  tar -C "${temp_make_dir}/downloads" -xzf \
    "${temp_make_dir}/downloads/fft2d.tgz"
}

# TF-Lite's "download_dependencies.sh" script tweaks the Eigen library before
# compilation. Apply similar tweaks here. Note that the other script specifies
# three separate tweaks, but only one of them is relevant.
function tweak_dependencies() {
  sed -i -e 's/static uint64x2_t p2ul_CONJ_XOR = vld1q_u64( p2ul_conj_XOR_DATA );/static uint64x2_t p2ul_CONJ_XOR;/' \
    "${temp_make_dir}/downloads/eigen/Eigen/src/Core/arch/NEON/Complex.h"
}

# Gerrit complains about GCC warnings when building the library. However,
# these warnings are from the Tensorflow libraries. Build the library and
# silence the output (but re-display it in the case of an error).
function build_library() {
  local output=$("${temp_dir}/tensorflow/tensorflow/lite/tools/make/build_lib.sh" 2>&1)
  local rcode=$?
  if [[ ${rcode} -ne 0 ]]; then
    echo "Error building TF-Lite"
    echo "${output}"
    exit ${rcode}
  fi
}

# Every ten seconds, print out that we're building the TF-Lite library.
# Otherwise, it looks like the build is stuck.
function repeat_print_building() {
  while true; do
    echo -n $(date +%H:%M:%S) " "
    echo "Building TF-Lite...";
    sleep 10;
  done
}

if [[ $# -ne 2 ]]; then
  echo "Usage: aom_build.sh /path/to/tensorflow /output/file"
  exit 1
fi

cp -r "$1" "${temp_dir}"
readonly tf_aom_dir=$(dirname "$0")
readonly temp_make_dir="${temp_dir}/tensorflow/tensorflow/lite/tools/make"
mkdir "${temp_make_dir}/downloads"
cp -r "${tf_aom_dir}/ARM_NEON_2_x86_SSE" "${temp_make_dir}/downloads/neon_2_sse"
cp -r "${tf_aom_dir}/abseil-cpp" "${temp_make_dir}/downloads/absl"
cp -r "${tf_aom_dir}/eigen" "${temp_make_dir}/downloads/"
cp -r "${tf_aom_dir}/farmhash" "${temp_make_dir}/downloads/"
cp -r "${tf_aom_dir}/flatbuffers" "${temp_make_dir}/downloads/"
cp -r "${tf_aom_dir}/gemmlowp" "${temp_make_dir}/downloads/"
# Note that fft2d is not in a git repository, so it has been downloaded
# separately.
echo "Downloading FFT2D for TF-Lite"
download_fft2d
# There are specific tweaks applied by TF-Lite: re-apply them here.
tweak_dependencies

set +ex
repeat_print_building &
repeat_print_pid=$!
build_library
kill ${repeat_print_pid}
repeat_print_pid="-1"  # Signal that the print process was killed.
set -ex

cp "${temp_dir}/tensorflow/tensorflow/lite/tools/make/gen/linux_x86_64/lib/libtensorflow-lite.a" "$2"
