---
name: libaom/av1/use-capability-helper-or-flag-for-optimized-scaler-dispatch
description: When gating dispatch to specialized or optimized routines like frame scalers, use centralized capability helpers and guard platform-specific ratios with architecture macros
rubric: correctness
severity: should_fix
---

**Rule.** When gating dispatch to specialized or optimized routines (such as
frame scalers or resize kernels), control execution using centralized capability
helpers (such as av1_has_optimized_scaler()) or explicit flags rather than
hardcoding dimension checks at call sites. Within capability query helpers,
guard platform-specific parameters, ratios, or block sizes with matching
architecture macros (such as #if HAVE_SSSE3 || HAVE_NEON) so unsupported
platforms do not incorrectly report optimized availability.

**Why.** Hardcoding dimension ratio checks at call sites duplicates logic and
obscures intent. Query helpers that report optimized kernel availability without
checking architecture macros (such as HAVE_SSSE3 or HAVE_NEON) cause unsupported
platforms to take optimized branches that lack SIMD implementations or fall back
incorrectly.

## Examples

Bad:

```cpp
// Bad: Hardcoding dimension ratio checks at call sites
if (((cm->width << 1) == unscaled->y_crop_width &&
     (cm->height << 1) == unscaled->y_crop_height)) {
  av1_resize_and_extend_frame(unscaled, scaled, filter, phase, num_planes);
}

// Bad: Capability helper assumes platform support without architecture guards
static inline bool av1_has_optimized_scaler(int src_w, int src_h,
                                            int dst_w, int dst_h) {
  return (dst_w * 4 == src_w * 3 && dst_h * 4 == src_h * 3);
}
```

Good:

```cpp
// Good: Call sites check centralized capability helper and flag
if (use_optimized_scaler &&
    av1_has_optimized_scaler(unscaled->y_crop_width, unscaled->y_crop_height,
                             scaled->y_crop_width, scaled->y_crop_height)) {
  av1_resize_and_extend_frame(unscaled, scaled, filter, phase, num_planes);
}

// Good: Capability helper guards platform-specific ratios with HAVE_* macros
static inline bool av1_has_optimized_scaler(int src_w, int src_h,
                                            int dst_w, int dst_h) {
  bool has_scaler = false;
#if HAVE_SSSE3 || HAVE_NEON
  has_scaler = has_scaler || (4 * dst_w == 3 * src_w && 4 * dst_h == 3 * src_h);
#endif
  return has_scaler;
}
```
