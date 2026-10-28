---
name: libaom/aom_dsp/factor-repetitive-simd-arithmetic-into-inline-helpers
description: When implementing SIMD routines, use existing helper functions (e.g., abs_diff16) or factor repetitive multi-instruction arithmetic sequences across lanes and sub-registers into static inline helpers
rubric: design
severity: should_fix
---

**Rule.** When implementing or modifying SIMD routines, use established inline
helper functions (such as abs_diff16() or abs_diff()) rather than open-coding
equivalent multi-instruction intrinsic sequences (such as
_mm_or_si128(_mm_subs_epu16(a, b), _mm_subs_epu16(b, a))). When repetitive
multi-step arithmetic sequences (such as multiply-accumulate followed by
rounding shifts), mask calculations, or sub-register operations across low/high
halves (such as vget_low_*/vget_high_*, _mm_extract*, or _mm256_extract*) recur
across vector lanes, loop unrolls, or related kernel functions without an
existing helper, factor the shared sequence into a static inline helper
function.

**Why.** Using established inline SIMD helpers and factoring repetitive
multi-instruction sequences into reusable static inline functions clarifies
mathematical intent, reduces visual clutter, and eliminates copy-paste errors
across sub-registers and unrolled lanes without incurring runtime call overhead.

## Examples

Bad:

```cpp
// Open-coded unsigned absolute difference instead of using helper
const __m128i abs_p1p0 =
    _mm_or_si128(_mm_subs_epu16(p[1], p[0]), _mm_subs_epu16(p[0], p[1]));

// Duplicated multi-instruction arithmetic across low and high sub-registers
uint32x4_t m0_lo = vmull_u16(vget_low_u16(m0), vget_low_u16(s0));
m0_lo = vmlal_u16(m0_lo, vget_low_u16(m0_inv), vget_low_u16(s1));
uint16x4_t res_lo = vshrn_n_u32(m0_lo, AOM_BLEND_A64_ROUND_BITS);
uint32x4_t m0_hi = vmull_u16(vget_high_u16(m0), vget_high_u16(s0));
m0_hi = vmlal_u16(m0_hi, vget_high_u16(m0_inv), vget_high_u16(s1));
uint16x4_t res_hi = vshrn_n_u32(m0_hi, AOM_BLEND_A64_ROUND_BITS);
```

Good:

```cpp
// Use existing SIMD helper function
const __m128i abs_p1p0 = abs_diff16(p[1], p[0]);

// Factored repetitive lane/sub-register sequence into a static inline helper
static inline uint16x4_t blend_u16x4(uint16x4_t m, uint16x4_t m_inv,
                                     uint16x4_t a, uint16x4_t b) {
  uint32x4_t acc = vmull_u16(m, a);
  acc = vmlal_u16(acc, m_inv, b);
  return vshrn_n_u32(acc, AOM_BLEND_A64_ROUND_BITS);
}
uint16x4_t res_lo = blend_u16x4(vget_low_u16(m0), vget_low_u16(m0_inv),
                                vget_low_u16(s0), vget_low_u16(s1));
uint16x4_t res_hi = blend_u16x4(vget_high_u16(m0), vget_high_u16(m0_inv),
                                vget_high_u16(s0), vget_high_u16(s1));
```
