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

#if !CONFIG_RUNTIME_CPU_DETECT

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

#elif defined(__APPLE__) && AOM_ARCH_AARCH64  // end !CONFIG_RUNTIME_CPU_DETECT
#include <stdbool.h>
#include <sys/sysctl.h>

// sysctlbyname() parameter documentation for instruction set characteristics:
// https://developer.apple.com/documentation/kernel/1387446-sysctlbyname/determining_instruction_set_characteristics
static INLINE bool have_feature(const char *feature) {
  int64_t feature_present = 0;
  size_t size = sizeof(feature_present);

  if (sysctlbyname(feature, &feature_present, &size, NULL, 0) != 0) {
    return false;
  }

  return feature_present;
}

int aom_arm_cpu_caps(void) {
  int flags;
  int mask;
  if (!arm_cpu_env_flags(&flags)) {
    return flags;
  }
  mask = arm_cpu_env_mask();

#if HAVE_NEON
  flags |= HAS_NEON;
#endif  // HAVE_NEON
#if HAVE_ARM_CRC32
  if (have_feature("hw.optional.armv8_crc32")) flags |= HAS_ARM_CRC32;
#endif  // HAVE_ARM_CRC32
#if HAVE_NEON_DOTPROD
  if (have_feature("hw.optional.arm.FEAT_DotProd")) flags |= HAS_NEON_DOTPROD;
#endif  // HAVE_NEON_DOTPROD
#if HAVE_NEON_I8MM
  if (have_feature("hw.optional.arm.FEAT_I8MM")) flags |= HAS_NEON_I8MM;
#endif  // HAVE_NEON_I8MM
  return flags & mask;
}

#elif defined(_MSC_VER)  // end __APPLE__ && AOM_ARCH_AARCH64
#undef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#undef WIN32_EXTRA_LEAN
#define WIN32_EXTRA_LEAN

#include <windows.h>

int aom_arm_cpu_caps(void) {
  int flags;
  int mask;
  if (!arm_cpu_env_flags(&flags)) {
    return flags;
  }
  mask = arm_cpu_env_mask();

#if AOM_ARCH_AARCH64
// IsProcessorFeaturePresent() parameter documentation:
// https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-isprocessorfeaturepresent#parameters
#if HAVE_NEON
  flags |= HAS_NEON;  // Neon is mandatory in Armv8.0-A.
#endif  // HAVE_NEON
#if HAVE_ARM_CRC32
  if (IsProcessorFeaturePresent(PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE)) {
    flags |= HAS_ARM_CRC32;
  }
#endif  // HAVE_ARM_CRC32
#if HAVE_NEON_DOTPROD
// Support for PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE was added in Windows SDK
// 20348, supported by Windows 11 and Windows Server 2022.
#if defined(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE)
  if (IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE)) {
    flags |= HAS_NEON_DOTPROD;
  }
#endif  // defined(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE)
#endif  // HAVE_NEON_DOTPROD
// No I8MM feature detection available on Windows at time of writing.
#else   // !AOM_ARCH_AARCH64
#if HAVE_NEON
  // MSVC has no inline __asm support for Arm, but it does let you __emit
  // instructions via their assembled hex code.
  // All of these instructions should be essentially nops.
  if (mask & HAS_NEON) {
    __try {
      // VORR q0,q0,q0
      __emit(0xF2200150);
      flags |= HAS_NEON;
    } __except (GetExceptionCode() == EXCEPTION_ILLEGAL_INSTRUCTION) {
      // Ignore exception.
    }
  }
#endif  // HAVE_NEON
#endif  // AOM_ARCH_AARCH64
  return flags & mask;
}

#elif defined(__ANDROID__) && (__ANDROID_API__ < 18)  // end _MSC_VER
// Use getauxval() when targeting (64-bit) Android with API level >= 18.
// getauxval() is supported since Android API level 18 (Android 4.3.)
// First Android version with 64-bit support was Android 5.x (API level 21).
#include <cpu-features.h>

int aom_arm_cpu_caps(void) {
  int flags;
  int mask;
  if (!arm_cpu_env_flags(&flags)) {
    return flags;
  }
  mask = arm_cpu_env_mask();

#if HAVE_NEON
#if AOM_ARCH_AARCH64
  flags |= HAS_NEON;  // Neon is mandatory in Armv8.0-A.
#else   // !AOM_ARCH_AARCH64
  uint64_t features = android_getCpuFeatures();
  if (features & ANDROID_CPU_ARM_FEATURE_NEON) flags |= HAS_NEON;
#endif  // AOM_ARCH_AARCH64
#endif  // HAVE_NEON
  return flags & mask;
}

#elif defined(__linux__)  // end __ANDROID__ && (__ANDROID_API__ < 18)

#include <sys/auxv.h>

// Define hwcap values ourselves: building with an old auxv header where these
// hwcap values are not defined should not prevent features from being enabled.
#define AOM_AARCH32_HWCAP_NEON (1 << 12)
#define AOM_AARCH64_HWCAP_CRC32 (1 << 7)
#define AOM_AARCH64_HWCAP_ASIMDDP (1 << 20)
#define AOM_AARCH64_HWCAP_SVE (1 << 22)
#define AOM_AARCH64_HWCAP2_I8MM (1 << 13)

int aom_arm_cpu_caps(void) {
  int flags;
  if (!arm_cpu_env_flags(&flags)) {
    return flags;
  }
  int mask = arm_cpu_env_mask();
  unsigned long hwcap = getauxval(AT_HWCAP);
#if AOM_ARCH_AARCH64
  unsigned long hwcap2 = getauxval(AT_HWCAP2);
#if HAVE_NEON
  flags |= HAS_NEON;  // Neon is mandatory in Armv8.0-A.
#endif  // HAVE_NEON
#if HAVE_ARM_CRC32
  if (hwcap & AOM_AARCH64_HWCAP_CRC32) flags |= HAS_ARM_CRC32;
#endif  // HAVE_ARM_CRC32
#if HAVE_NEON_DOTPROD
  if (hwcap & AOM_AARCH64_HWCAP_ASIMDDP) flags |= HAS_NEON_DOTPROD;
#endif  // HAVE_NEON_DOTPROD
#if HAVE_NEON_I8MM
  if (hwcap2 & AOM_AARCH64_HWCAP2_I8MM) flags |= HAS_NEON_I8MM;
#endif  // HAVE_NEON_I8MM
#if HAVE_SVE
  if (hwcap & AOM_AARCH64_HWCAP_SVE) flags |= HAS_SVE;
#endif  // HAVE_SVE
#else   // !AOM_ARCH_AARCH64
#if HAVE_NEON
  if (hwcap & AOM_AARCH32_HWCAP_NEON) flags |= HAS_NEON;
#endif  // HAVE_NEON
#endif  // AOM_ARCH_AARCH64
  return flags & mask;
}
#else   /* end __linux__ */
#error \
    "Runtime CPU detection selected, but no CPU detection method " \
"available for your platform. Rerun cmake with -DCONFIG_RUNTIME_CPU_DETECT=0."
#endif
