---
name: libaom/av1/delegate-composite-dsp-to-standalone-primitives
description: When implementing composite DSP or SIMD routines, invoke existing standalone primitives (e.g., aom_comp_avg_pred, aom_dist_wtd_comp_avg_pred) rather than inlining vector loops
rubric: design
severity: should_fix
---

**Rule.** In composite DSP and SIMD routines (such as compound average upsampled
prediction or distance-weighted compound prediction), invoke the corresponding
standalone DSP primitive functions (such as `aom_comp_avg_pred_*`,
`aom_highbd_comp_avg_pred_*`, or `aom_dist_wtd_comp_avg_pred_*`) instead of
duplicating or open-coding vector intrinsic loops. When chaining multi-stage
operations, pass intermediate buffers between the specialized building blocks
and maintain consistent delegation across low-bitdepth and high-bitdepth
implementations.

**Why.** Inlining arithmetic or blending loops across multiple composite
functions duplicates complex SIMD logic, increases maintenance burden, and
creates redundant code across architecture variants. Delegating sub-operations
to dedicated standalone primitives eliminates duplicate vector code, ensures
consistent math, rounding, and bitdepth handling across call sites, reduces
binary size, and allows composite routines to automatically benefit from
optimizations or bug fixes in the underlying building blocks.

## Examples

Bad:

```cpp
// Anti-pattern: Open-coding inline SIMD loops inside composite prediction functions
void aom_highbd_comp_avg_upsampled_pred_neon(...) {
  aom_highbd_upsampled_pred_neon(xd, cm, mi_row, mi_col, mv, comp_pred8, ...);
  for (int i = 0; i < width * height; i += 8) {
    uint16x8_t cp = vqrshlq_u16(vaddq_u16(vld1q_u16(&pred[i]), vld1q_u16(&comp_pred[i])), shift);
    vst1q_u16(&comp_pred[i], cp);
  }
}

void aom_highbd_dist_wtd_comp_avg_upsampled_pred_neon(...) {
  aom_highbd_upsampled_pred_neon(xd, cm, mi_row, mi_col, mv, comp_pred8, ...);
  for (int i = 0; i < width * height; i += 8) {
    // Inlined distance-weighted blending calculation loop
    uint32x4_t cp0 = vmlal_u16(vmull_u16(vget_low_u16(p), bck_offset_u16), vget_low_u16(cp), fwd_offset_u16);
    ...
  }
}
```

Good:

```cpp
// Delegate composite sub-operations to dedicated standalone SIMD primitives
void aom_highbd_comp_avg_upsampled_pred_neon(...) {
  aom_highbd_upsampled_pred_neon(xd, cm, mi_row, mi_col, mv, comp_pred8, width,
                                 height, subpel_x_q3, subpel_y_q3, ref8,
                                 ref_stride, bd, subpel_search);
  aom_highbd_comp_avg_pred_neon(comp_pred8, pred8, width, height, comp_pred8,
                                width);
}

void aom_highbd_dist_wtd_comp_avg_upsampled_pred_neon(...) {
  aom_highbd_upsampled_pred_neon(xd, cm, mi_row, mi_col, mv, comp_pred8, width,
                                 height, subpel_x_q3, subpel_y_q3, ref8,
                                 ref_stride, bd, subpel_search);
  aom_highbd_dist_wtd_comp_avg_pred_neon(comp_pred8, pred8, width, height,
                                         comp_pred8, width, jcp_param);
}
```
