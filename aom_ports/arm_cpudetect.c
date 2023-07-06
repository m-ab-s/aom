/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <stdlib.h>
#include <string.h>
#include "aom_ports/arm.h"
#include "config/aom_config.h"

#ifdef WINAPI_FAMILY
#include <winapifamily.h>
#if !WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
#define getenv(x) NULL
#endif
#endif

static int arm_cpu_env_flags(int *flags) {
  char *env;
  env = getenv("AOM_SIMD_CAPS");
  if (env && *env) {
    *flags = (int)strtol(env, NULL, 0);
    return 0;
  }
  *flags = 0;
  return -1;
}

static int arm_cpu_env_mask(void) {
  char *env;
  env = getenv("AOM_SIMD_CAPS_MASK");
  return env && *env ? (int)strtol(env, NULL, 0) : ~0;
}

#if !CONFIG_RUNTIME_CPU_DETECT || defined(__APPLE__)

int aom_arm_cpu_caps(void) {
  /* This function should actually be a no-op. There is no way to adjust any of
   * these because the RTCD tables do not exist: the functions are called
   * statically */
  int flags;
  int mask;
  if (!arm_cpu_env_flags(&flags)) {
    return flags;
  }
  mask = arm_cpu_env_mask();
#if HAVE_NEON
  flags |= HAS_NEON;
#endif /* HAVE_NEON */
  return flags & mask;
}

#elif defined(_MSC_VER) /* end !CONFIG_RUNTIME_CPU_DETECT || __APPLE__ */
#if HAVE_NEON && !AOM_ARCH_AARCH64
/*For GetExceptionCode() and EXCEPTION_ILLEGAL_INSTRUCTION.*/
#undef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#undef WIN32_EXTRA_LEAN
#define WIN32_EXTRA_LEAN
#include <windows.h>
#endif  // HAVE_NEON && !AOM_ARCH_AARCH64

int aom_arm_cpu_caps(void) {
  int flags;
  int mask;
  if (!arm_cpu_env_flags(&flags)) {
    return flags;
  }
  mask = arm_cpu_env_mask();
#if AOM_ARCH_AARCH64
  return HAS_NEON & mask;
#else
/* MSVC has no inline __asm support for ARM, but it does let you __emit
 *  instructions via their assembled hex code.
 * All of these instructions should be essentially nops.
 */
#if HAVE_NEON
  if (mask & HAS_NEON) {
    __try {
      /*VORR q0,q0,q0*/
      __emit(0xF2200150);
      flags |= HAS_NEON;
    } __except (GetExceptionCode() == EXCEPTION_ILLEGAL_INSTRUCTION) {
      /*Ignore exception.*/
    }
  }
#endif  /* HAVE_NEON */
  return flags & mask;
#endif  // AOM_ARCH_AARCH64
}

#elif defined(__ANDROID__) /* end _MSC_VER */
#include <cpu-features.h>

int aom_arm_cpu_caps(void) {
  int flags;
  int mask;
  uint64_t features;
  if (!arm_cpu_env_flags(&flags)) {
    return flags;
  }
  mask = arm_cpu_env_mask();
  features = android_getCpuFeatures();

#if HAVE_NEON
  if (features & ANDROID_CPU_ARM_FEATURE_NEON) flags |= HAS_NEON;
#endif /* HAVE_NEON */
  return flags & mask;
}

#elif defined(__linux__) /* end __ANDROID__ */

#include <sys/auxv.h>

int aom_arm_cpu_caps(void) {
  int flags;
  if (!arm_cpu_env_flags(&flags)) {
    return flags;
  }
  int mask = arm_cpu_env_mask();
#if AOM_ARCH_AARCH64
  unsigned long hwcap = getauxval(AT_HWCAP);
#if HAVE_NEON
  flags |= HAS_NEON;  // Neon is mandatory in Armv8.0-A.
#endif  // HAVE_NEON
#if HAVE_ARM_CRC32
  if (hwcap & HWCAP_CRC32) flags |= HAS_ARM_CRC32;
#endif  // HAVE_ARM_CRC32
#else   // !AOM_ARCH_AARCH64
  // No runtime feature detection for Armv7 on Linux (yet).
#endif  // AOM_ARCH_AARCH64
  return flags & mask;
}
#else   /* end __linux__ */
#error \
    "Runtime CPU detection selected, but no CPU detection method " \
"available for your platform. Rerun cmake with -DCONFIG_RUNTIME_CPU_DETECT=0."
#endif
