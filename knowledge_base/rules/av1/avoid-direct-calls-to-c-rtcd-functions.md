---
name: libaom/av1/avoid-direct-calls-to-c-rtcd-functions
description: When calling RTCD functions, use base dispatch symbols in production code and matching architecture SIMD kernels in architecture wrappers instead of _c fallbacks
rubric: correctness
severity: should_fix
---

**Rule.** Do not directly invoke unoptimized C fallback (*_c) RTCD or DSP
functions in production code or architecture-specific wrapper implementations:
1. In general production code, call the base RTCD dispatch symbol (without _c or architecture suffixes such as _neon, _sse2, _avx2) to enable runtime CPU detection.
2. In architecture-specific wrapper, dual, or quad SIMD functions (such as *_dual_neon, *_quad_sse2, or *_dual_avx2), invoke the matching architecture-specific SIMD kernels (e.g., *_neon, *_sse2, *_avx2) rather than calling C fallback routines (*_c).

**Why.** In libaom, RTCD (Run-Time CPU Detection) dynamically selects optimized
SIMD implementations at runtime. Directly calling *_c functions in production
code bypasses CPU detection and forces unoptimized C execution. Similarly,
invoking *_c routines inside architecture-specific SIMD wrappers silently
defeats SIMD acceleration and degrades codec performance.

## Examples

Bad:

```cpp
// In general codec code: directly calling _c bypasses RTCD
aom_lpf_horizontal_14_dual_c(s, pitch, lfi->mblim, lfi->lim,
                             lfi->hev_thr, lfin->mblim, lfin->lim,
                             lfin->hev_thr);

// In architecture wrappers: calling _c defeats SIMD acceleration
void aom_lpf_vertical_6_dual_neon(uint8_t *s, int pitch, const uint8_t *blimit0,
                                  const uint8_t *limit0, const uint8_t *thresh0,
                                  const uint8_t *blimit1, const uint8_t *limit1,
                                  const uint8_t *thresh1) {
  aom_lpf_vertical_6_c(s, pitch, blimit0, limit0, thresh0);
  aom_lpf_vertical_6_c(s + 4 * pitch, pitch, blimit1, limit1, thresh1);
}
```

Good:

```cpp
// In general codec code: use base RTCD dispatch symbol
aom_lpf_horizontal_14_dual(s, pitch, lfi->mblim, lfi->lim,
                           lfi->hev_thr, lfin->mblim, lfin->lim,
                           lfin->hev_thr);

// In architecture wrappers: invoke matching architecture SIMD kernels
void aom_lpf_vertical_6_dual_neon(uint8_t *s, int pitch, const uint8_t *blimit0,
                                  const uint8_t *limit0, const uint8_t *thresh0,
                                  const uint8_t *blimit1, const uint8_t *limit1,
                                  const uint8_t *thresh1) {
  aom_lpf_vertical_6_neon(s, pitch, blimit0, limit0, thresh0);
  aom_lpf_vertical_6_neon(s + 4 * pitch, pitch, blimit1, limit1, thresh1);
}
```
