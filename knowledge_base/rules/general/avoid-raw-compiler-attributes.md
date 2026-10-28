---
name: libaom/avoid-raw-compiler-attributes
description: When annotating functions or declaring aligned buffers, avoid raw `__attribute__((...))` or `__declspec(...)` syntax; use project portability macros such as `AOM_FORCE_INLINE` or `DECLARE_ALIGNED`
rubric: correctness
severity: blocking
---

**Rule.** Do not use raw GNU/Clang or MSVC compiler attribute syntax such as
`__attribute__((...))`, `__declspec(...)`, or `alignas` directly on functions,
variables, or buffers in cross-platform source files. Use project-defined
portability macros such as `AOM_FORCE_INLINE`, `AOM_INLINE`, `AOM_NO_RETURN`, or
`DECLARE_ALIGNED(n, typ, val)` (from `aom_ports/mem.h`), or guard
compiler-specific attributes with `#if defined(_MSC_VER)` / `#elif
defined(__GNUC__)` checks to ensure cross-compiler compatibility across GCC,
Clang, and Microsoft Visual Studio (MSVC). Ensure alignment boundaries `n`
passed to `DECLARE_ALIGNED` match target SIMD vector widths (e.g., 16 bytes for
SSE/NEON, 32 bytes for AVX2, or 64 bytes for AVX-512).

**Why.** Libaom officially supports cross-platform builds across GCC, Clang, and
Microsoft Visual C++ (MSVC). Raw compiler attributes such as
`__attribute__((...))` or `__declspec(...)` cause syntax errors or build
failures on non-matching toolchains. Project portability macros (like
`AOM_FORCE_INLINE` and `DECLARE_ALIGNED`) encapsulate compiler-specific syntax
portably.

## Examples

Bad:

```cpp
// Bad: Raw GCC/Clang attributes cause build failures under MSVC.
static void __attribute__((noinline))
dr_prediction_z2_Nx4_neon(int N, uint8_t *dst, ptrdiff_t stride);

static __attribute__((always_inline)) int fast_calc(int x) {
  return x * 2;
}

// Bad: Direct compiler alignment attribute breaks cross-platform builds.
__attribute__((aligned(64))) uint16_t dstT[64 * 64];
```

Good:

```cpp
#include "aom_ports/mem.h"

// Good: Use project portability macros or omit non-essential attributes.
static void dr_prediction_z2_Nx4_neon(int N, uint8_t *dst, ptrdiff_t stride);

static AOM_FORCE_INLINE int fast_calc(int x) {
  return x * 2;
}

// Good: DECLARE_ALIGNED ensures portable alignment across GCC, Clang, and MSVC.
DECLARE_ALIGNED(32, uint16_t, dstT[64 * 64]);
```
