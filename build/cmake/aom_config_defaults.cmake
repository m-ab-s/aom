#
# Copyright (c) 2016, Alliance for Open Media. All rights reserved
#
# This source code is subject to the terms of the BSD 2 Clause License and the
# Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License was
# not distributed with this source code in the LICENSE file, you can obtain it
# at www.aomedia.org/license/software. If the Alliance for Open Media Patent
# License 1.0 was not distributed with this source code in the PATENTS file, you
# can obtain it at www.aomedia.org/license/patent.
#
# Defaults for every libaom configuration variable. Here we add all libaom
# config variables to the cmake variable cache, but omit the FORCE parameter to
# allow users to specify values when executing cmake to generate build files.
# Values here are used only if not set by the user.
set(INLINE "" CACHE STRING "Sets INLINE value for current target.")

# CPUs.
set(ARCH_ARM 0 CACHE STRING "Enables ARM architecture.")
set(ARCH_MIPS 0 CACHE STRING "Enables MIPS architecture.")
set(ARCH_PPC 0 CACHE STRING "Enables PPC architecture.")
set(ARCH_X86 0 CACHE STRING "Enables X86 architecture.")
set(ARCH_X86_64 0 CACHE STRING "Enables X86_64 architecture.")

# ARM optimization flags.
set(HAVE_NEON 0 CACHE STRING "Enables NEON intrinsics optimizations.")

# MIPS optimization flags.
set(HAVE_DSPR2 0 CACHE STRING "Enables DSPR2 optimizations.")
set(HAVE_MIPS32 0 CACHE STRING "Enables MIPS32 optimizations.")
set(HAVE_MIPS64 0 CACHE STRING "Enables MIPS64 optimizations. ")
set(HAVE_MSA 0 CACHE STRING "Enables MSA optimizations.")

# PPC optimization flags.
set(HAVE_VSX 0 CACHE STRING "Enables VSX optimizations.")

# x86/x86_64 optimization flags.
set(HAVE_AVX 0 CACHE STRING "Enables AVX optimizations.")
set(HAVE_AVX2 0 CACHE STRING "Enables AVX2 optimizations.")
set(HAVE_MMX 0 CACHE STRING "Enables MMX optimizations. ")
set(HAVE_SSE 0 CACHE STRING "Enables SSE optimizations.")
set(HAVE_SSE2 0 CACHE STRING "Enables SSE2 optimizations.")
set(HAVE_SSE3 0 CACHE STRING "Enables SSE3 optimizations.")
set(HAVE_SSE4_1 0 CACHE STRING "Enables SSE 4.1 optimizations.")
set(HAVE_SSE4_2 0 CACHE STRING "Enables SSE 4.2 optimizations.")
set(HAVE_SSSE3 0 CACHE STRING "Enables SSSE3 optimizations.")

# Flags describing the build environment.
set(HAVE_FEXCEPT 0 CACHE STRING "Internal flag, GNU fenv.h present for target.")
set(HAVE_PTHREAD_H 0 CACHE STRING "Internal flag, target pthread support.")
set(HAVE_UNISTD_H 0 CACHE STRING "Internal flag, unistd.h present for target.")
set(HAVE_WXWIDGETS 0 CACHE STRING "WxWidgets present.")

# Build configuration flags.
set(CONFIG_AV1_DECODER 1 CACHE STRING "Enable AV1 decoder.")
set(CONFIG_AV1_ENCODER 1 CACHE STRING "Enable AV1 encoder.")
set(CONFIG_BIG_ENDIAN 0 CACHE STRING "Internal flag.")
set(CONFIG_GCC 0 CACHE STRING "Building with GCC (detected).")
set(CONFIG_GCOV 0 CACHE STRING "Enable gcov support.")
set(CONFIG_GPROF 0 CACHE STRING "Enable gprof support.")
set(CONFIG_LIBYUV 1 CACHE STRING "Enables libyuv scaling/conversion support.")
set(CONFIG_MSVS 0 CACHE STRING "Building with MS Visual Studio (detected).")
set(CONFIG_MULTITHREAD 0 CACHE STRING "No multithread support.")
set(CONFIG_OS_SUPPORT 0 CACHE STRING "Internal flag.")
set(CONFIG_PIC 0 CACHE STRING "Build with PIC enabled.")
set(CONFIG_RUNTIME_CPU_DETECT 1 CACHE STRING "Runtime CPU detection support.")
set(CONFIG_SHARED 0 CACHE STRING "Build shared libs.")
set(CONFIG_STATIC 1 CACHE STRING "Build static libs.")
set(CONFIG_WEBM_IO 1 CACHE STRING "Enables WebM support.")

# Debugging flags.
set(CONFIG_BITSTREAM_DEBUG 0 CACHE STRING "Bitstream debugging flag.")
set(CONFIG_DEBUG 0 CACHE STRING "Debug build flag.")
set(CONFIG_MISMATCH_DEBUG 0 CACHE STRING "Mismatch debugging flag.")

# AV1 feature flags.
set(CONFIG_ACCOUNTING 0 CACHE STRING "Enables bit accounting.")
set(CONFIG_ANALYZER 0 CACHE STRING "Enables bit stream analyzer.")
set(CONFIG_COEFFICIENT_RANGE_CHECKING 0 CACHE STRING "Coefficient range check.")
set(CONFIG_FILEOPTIONS 1 CACHE STRING "Enables encoder config file support.")
set(CONFIG_INSPECTION 0 CACHE STRING "Enables bitstream inspection.")
set(CONFIG_INTERNAL_STATS 0 CACHE STRING "Enables internal encoder stats.")
set(CONFIG_LOWBITDEPTH 0 CACHE STRING "Enables 8-bit optimized pipeline.")
set(CONFIG_SIZE_LIMIT 0 CACHE STRING "Limit max decode width/height.")
set(CONFIG_SPATIAL_RESAMPLING 1 CACHE STRING "Spatial resampling.")
set(DECODE_HEIGHT_LIMIT 0 CACHE STRING "Set limit for decode height.")
set(DECODE_WIDTH_LIMIT 0 CACHE STRING "Set limit for decode width.")

# AV1 experiment flags.
set(CONFIG_COLLECT_INTER_MODE_RD_STATS 1 CACHE STRING "AV1 experiment flag.")
set(CONFIG_COLLECT_RD_STATS 0 CACHE STRING "AV1 experiment flag.")
set(CONFIG_DIST_8X8 0 CACHE STRING "AV1 experiment flag.")
set(CONFIG_ENTROPY_STATS 0 CACHE STRING "AV1 experiment flag.")
set(CONFIG_FP_MB_STATS 0 CACHE STRING "AV1 experiment flag.")
set(CONFIG_INTER_STATS_ONLY 0 CACHE STRING "AV1 experiment flag.")
set(CONFIG_RD_DEBUG 0 CACHE STRING "AV1 experiment flag.")
