---
name: libaom/av1/dispatch-8bit-simd-by-runtime-bit-depth
description: When implementing 8-bit SIMD optimizations, dispatch based on runtime bit depth at caller sites rather than disabling SIMD under CONFIG_AV1_HIGHBITDEPTH
rubric: correctness
severity: should_fix
---

**Rule.** Do not disable 8-bit SIMD optimizations or unconditionally fall back
to C reference implementations using compile-time #if CONFIG_AV1_HIGHBITDEPTH
inside SIMD functions. Instead, keep 8-bit SIMD routines enabled and dispatch
between 8-bit and high-bit-depth routines at caller sites based on runtime bit
depth (e.g., cm->seq_params.bit_depth == AOM_BITS_8).

**Why.** In libaom, CONFIG_AV1_HIGHBITDEPTH is a build-time configuration macro
indicating that the compiled binary supports both 8-bit and 10/12-bit video
streams dynamically at runtime. Using #if CONFIG_AV1_HIGHBITDEPTH inside
low-bitdepth SIMD routines disables SIMD acceleration for standard 8-bit streams
in high-bitdepth-enabled builds, causing severe performance regressions.

## Examples

Bad:

```cpp
// Inside 8-bit SIMD implementation (e.g., resize_neon.c or resize_ssse3.c):
#if CONFIG_AV1_HIGHBITDEPTH
  // Disables 8-bit SIMD optimization whenever high-bitdepth build is enabled
  av1_resize_and_extend_frame_c(src, dst, filter, num_planes, phase);
#else
  av1_resize_and_extend_frame_ssse3(src, dst, filter, num_planes, phase);
#endif
```

Good:

```cpp
// In caller dispatch (e.g., av1_scale_if_required()):
#if CONFIG_AV1_HIGHBITDEPTH
  if ((cm->width << 1) == unscaled->y_crop_width &&
      (cm->height << 1) == unscaled->y_crop_height &&
      cm->seq_params.bit_depth == AOM_BITS_8) {
    av1_resize_and_extend_frame(unscaled, scaled, filter, phase, num_planes);
  } else {
    av1_resize_and_extend_frame_nonnormative(
        unscaled, scaled, (int)cm->seq_params.bit_depth, num_planes);
  }
#else
  av1_resize_and_extend_frame(unscaled, scaled, filter, phase, num_planes);
#endif
```
