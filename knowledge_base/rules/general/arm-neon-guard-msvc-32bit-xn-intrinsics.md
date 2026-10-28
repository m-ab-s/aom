---
name: libaom/arm-neon-guard-msvc-32bit-xn-intrinsics
description: When defining compiler fallback polyfills for ARM Neon multi-vector xN intrinsics, guard with exact major/minor compiler version checks and include 32-bit MSVC
rubric: correctness
severity: blocking
---

**Rule.** When defining compiler fallback polyfills or compatibility wrappers
for ARM Neon multi-vector load/store intrinsics (such as `vld1q_*_xN` or
`vst1q_*_xN`) or backported compiler intrinsics:
1. Guard definitions with exact major and minor compiler version checks (such as `((__GNUC__ << 8) | __GNUC_MINOR__) < 0x805` or `__GNUC__ < 14`) rather than unconditional definitions or coarse major-only checks whenever native support was added in a specific minor/point release. Precede the fallback with a comment documenting the exact compiler version where native support was introduced.
2. Include 32-bit MSVC targets (`(defined(__clang__) || defined(_MSC_VER)) && defined(ARM_32_BIT)` or `defined(_M_ARM)`), as MSVC implements Neon multi-vector intrinsics on AArch64 (`_M_ARM64`) but omits them on 32-bit ARM.

**Why.** Newer compiler releases (such as GCC 14 or backports in GCC 8.5.0)
provide native definitions for Neon xN intrinsics that were absent in older
versions, causing duplicate symbol or conflicting declaration compiler errors if
polyfills are defined unconditionally or guarded only by major version numbers.
Conversely, 32-bit MSVC lacks multi-vector Neon intrinsics that are supported on
64-bit MSVC (AArch64), causing unresolved identifier errors if 32-bit MSVC is
excluded from the fallback definitions.

## Examples

Bad:

```cpp
// 1. Omits 32-bit MSVC or unconditionally defines fallbacks for all GCC versions:
#if defined(__GNUC__) && !defined(__clang__) && defined(ARM_32_BIT)
#define GCC_32_BIT
#endif

#if defined(DEFICIENT_CLANG_32_BIT) || defined(GCC_32_BIT)
static inline int16x8x4_t vld1q_s16_x4(const int16_t *ptr) {
  // ...
}
#endif

// 2. Coarse major-only guard causes conflicts on point release backports (e.g. GCC 8.5.0):
#if __GNUC__ < 9
static INLINE uint16x8x4_t vld1q_u16_x4(const uint16_t *ptr) {
  uint16x8x4_t res = { { vld1q_u16(ptr + 0 * 8), vld1q_u16(ptr + 1 * 8),
                         vld1q_u16(ptr + 2 * 8), vld1q_u16(ptr + 3 * 8) } };
  return res;
}
#endif
```

Good:

```cpp
// 1. Includes 32-bit MSVC and guards GCC with version thresholds:
#if (defined(__clang__) || defined(_MSC_VER)) && defined(ARM_32_BIT)
#define DEFICIENT_CLANG_32_BIT
#endif

#if defined(__GNUC__) && !defined(__clang__) && defined(ARM_32_BIT) && \
    __GNUC__ < 14
#define DEFICIENT_GCC_32_BIT
#endif

#if defined(DEFICIENT_CLANG_32_BIT) || defined(DEFICIENT_GCC_32_BIT)
static inline int16x8x4_t vld1q_s16_x4(const int16_t *ptr) {
  // ...
}
#endif

// 2. Exact major and minor version guard for point release backports:
// vld1q_u16_x4 is defined in GCC from 8.5.0 onwards:
#if ((__GNUC__ << 8) | __GNUC_MINOR__) < 0x805
static INLINE uint16x8x4_t vld1q_u16_x4(const uint16_t *ptr) {
  uint16x8x4_t res = { { vld1q_u16(ptr + 0 * 8), vld1q_u16(ptr + 1 * 8),
                         vld1q_u16(ptr + 2 * 8), vld1q_u16(ptr + 3 * 8) } };
  return res;
}
#endif
```
