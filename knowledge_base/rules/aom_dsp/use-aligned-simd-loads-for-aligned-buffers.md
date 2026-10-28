---
name: libaom/aom_dsp/use-aligned-simd-loads-for-aligned-buffers
description: In SIMD routines, use unaligned loads (like _mm_loadu_si128 or load_unaligned_u8) for unverified buffers, reserving aligned loads strictly for guaranteed-aligned memory
rubric: correctness
severity: blocking
---

**Rule.** In x86 (SSE/AVX) and ARM NEON SIMD routines, use unaligned load
intrinsics or helpers (such as _mm_loadu_si128, _mm256_loadu_si256,
xx_loadu_128, or load_unaligned_u8 / load_unaligned_u8_4x4 / load_unaligned_u8q)
when reading from arbitrary, unverified, offset, or reference frame buffer
pointers. Use alignment-restricted loads or helpers (such as _mm_load_si128,
_mm256_load_si256, or load_u8_4x1 / load_u8_4x2 / load_u8_4x4) strictly when the
source buffer is guaranteed and declared to meet vector/stride alignment (e.g.,
via DECLARE_ALIGNED).

**Why.** Aligned load instructions (e.g., MOVDQA) and alignment-restricted load
helpers trigger general protection faults, hardware exceptions, assertion
failures in debug builds, or undefined behavior when executed on unaligned
addresses or unaligned strides. Conversely, using aligned loads on
guaranteed-aligned buffers documents data layout invariants and avoids unaligned
access penalties on older architectures.

## Examples

Bad:

```cpp
// Bad (x86): Aligned load on arbitrary/unverified reference pointer
__m128i src_line = _mm_load_si128((const __m128i *)ref);

// Bad (ARM NEON): Using load helpers that assume 4-byte pointer/stride alignment
const uint8x8_t r0 = load_u8_4x2(ref, ref_stride);
const uint8x16_t r1 = load_u8_4x4(ref + 2 * ref_stride, ref_stride);
```

Good:

```cpp
// Good (x86): Use unaligned load for unverified pointer; aligned load for DECLARE_ALIGNED buffer
__m128i src_line = _mm_loadu_si128((const __m128i *)ref);
__m128i iscan0 = _mm_load_si128((const __m128i *)iscan_ptr);

// Good (ARM NEON): Use unaligned load helpers safe for arbitrary byte offsets and strides
const uint8x8_t r0 = load_unaligned_u8_4x2(ref, ref_stride);
uint8x8_t r1, r2;
load_unaligned_u8_4x4(ref + 2 * ref_stride, ref_stride, &r1, &r2);
```
