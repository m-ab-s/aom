---
name: libaom/aom_dsp/name-dsp-functions-by-metrics-and-block-layout
description: When defining DSP helper functions, RTCD primitives, or SIMD kernels, explicitly name their calculated metrics, full 2D block dimensions, and batching layout
rubric: readability
severity: should_fix
---

**Rule.** Name DSP helper functions, RTCD declarations, and SIMD kernels
explicitly after the exact metrics they compute and their precise 2D block
dimensions and multi-block layout. Specifically:
1. **Computed metrics**: Name all calculated outputs accurately (e.g., `get_var_sse_sum` or `get_sse_sum` rather than generic `var` when variance is not computed).
2. **2D Block dimensions**: Specify explicit width-by-height dimensions (e.g., `<name>_<width>x<height>` such as `blend_8x4`, `subtract_4x8`, or `_8x1` for single-row helpers) rather than ambiguous width-only suffixes (e.g., `_w8`).
3. **Batching layout**: Qualify multi-block batching formats (such as `8x8_quad` or `16x16_dual`) consistently across C references, RTCD tables, and architecture-specific SIMD implementations.

**Why.** Inaccurate or truncated names mislead callers and SIMD implementers.
Using generic metric names (like `var` when only SSE and sum are computed)
obscures what values are populated, while omitting height dimensions (like `_w8`
for an 8x4 block) causes ambiguity or naming collisions when introducing
single-line (`_8x1`) or other height variants. Explicit naming ensures
consistent RTCD integration and maintainability across SIMD backends.

## Examples

Bad:

```cpp
// Misleading metric name and ambiguous block layout
void aom_get8x8var_quad_c(const uint8_t *a, int a_stride, const uint8_t *b,
                          int b_stride, uint32_t *sse, int *sum);

// Width-only suffix for a 2D 8x4 block kernel
static INLINE void blend_w8(uint8_t *dst, uint32_t dst_stride,
                            const CONV_BUF_TYPE *src0, uint32_t src0_stride,
                            const CONV_BUF_TYPE *src1, uint32_t src1_stride,
                            int16x8_t mask0, int16x8_t mask1, int16x8_t mask2,
                            int16x8_t mask3, const int16x4_t v_maxval,
                            const uint16x8_t vec_round_offset,
                            const int16x8_t vec_round_bits) {
  // Processes an 8x4 block
}
```

Good:

```cpp
// Accurately names computed metrics and quad-batching layout
void aom_get_sse_sum_8x8_quad_c(const uint8_t *a, int a_stride,
                                const uint8_t *b, int b_stride, uint32_t *sse,
                                int *sum);

// Explicit 2D dimensions match block geometry (8x4)
static INLINE void blend_8x4(uint8_t *dst, uint32_t dst_stride,
                             const CONV_BUF_TYPE *src0, uint32_t src0_stride,
                             const CONV_BUF_TYPE *src1, uint32_t src1_stride,
                             int16x8_t mask0, int16x8_t mask1, int16x8_t mask2,
                             int16x8_t mask3, const int16x4_t v_maxval,
                             const uint16x8_t vec_round_offset,
                             const int16x8_t vec_round_bits) {
  // Explicit 8x4 dimensions match block geometry
}
```
