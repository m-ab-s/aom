---
name: libaom/aom_dsp/override-hwy-broken-32bit-for-highway-targets
description: When configuring x86 Highway SIMD target macros, define HWY_BASELINE_TARGETS as HWY_AVX3_DL for AVX-512 and override HWY_BROKEN_32BIT instead of HWY_BROKEN_TARGETS
rubric: readability
severity: blocking
---

**Rule.** When configuring Highway SIMD target macros at the top of x86
translation units before including Highway headers, define
`HWY_BASELINE_TARGETS` as `HWY_AVX3_DL` rather than `HWY_AVX3` in AVX-512 files
(`*_hwy_avx512.cc`) (and match the respective target macro such as `HWY_AVX2` or
`HWY_SSE4` for other SIMD units). Additionally, override target-specific
brokenness with `HWY_BROKEN_32BIT` (e.g., `#define HWY_BROKEN_32BIT 0` or
`#define HWY_BROKEN_32BIT (HWY_AVX3 | (HWY_AVX3 - 1))`) instead of globally
resetting compiler filters with `#define HWY_BROKEN_TARGETS 0`.

**Why.** Libaom standardizes on `HWY_AVX3_DL` (AVX-512 with double-pumped
256-bit vectors and AVX-512BW/DQ/VL extensions including VBMI, VBMI2, VNNI,
BITALG) for its AVX-512 Highway targets, ensuring consistent baseline detection,
RTCD dispatch, and compatibility. Selectively defining `HWY_BROKEN_32BIT`
controls 32-bit SIMD target availability without bypassing compiler and
architecture safeguards that `#define HWY_BROKEN_TARGETS 0` would globally
disable.

## Examples

Bad:

```cpp
// Bad: Using plain HWY_AVX3 for AVX-512 or globally disabling all broken target protections
#define HWY_BASELINE_TARGETS HWY_AVX3
#define HWY_BROKEN_TARGETS 0

#include "aom_dsp/sad_hwy.h"
```

Good:

```cpp
// Good: Using HWY_AVX3_DL for AVX-512 files and overriding HWY_BROKEN_32BIT selectively
#define HWY_BASELINE_TARGETS HWY_AVX3_DL
#define HWY_BROKEN_32BIT 0

#include "aom_dsp/sad_hwy.h"
```
