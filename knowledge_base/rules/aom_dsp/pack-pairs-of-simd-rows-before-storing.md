---
name: libaom/aom_dsp/pack-pairs-of-simd-rows-before-storing
description: In x86 SIMD routines handling narrow rows, pack row pairs into 128-bit or 256-bit vectors using high-half loads or yy_loadu2_128 instead of half-utilized registers
rubric: safety
severity: should_fix
---

**Rule.** In x86 SIMD routines processing rows narrower than vector capacity,
pack row pairs into full-width registers rather than leaving vectors
half-utilized:
1. When loading 8-byte (64-bit) rows into 128-bit vectors, use high-half loads (e.g., xx_loadh_64, _mm_loadh_pi, _mm_loadh_pd, loadh_epi64) instead of separate loads and _mm_unpacklo_epi64; on output, pack vector pairs (_mm_packus_epi16(a, b)) and store via paired 64-bit stores (_mm_storel_epi64, _mm_storeh_pi) instead of self-packing.
2. When loading 16-byte (128-bit) rows into 256-bit AVX2 vectors, pack two rows per iteration using yy_loadu2_128 instead of casting single 128-bit loads via _mm256_castsi128_si256.

**Why.** Leaving registers half-utilized (such as casting single 128-bit loads
into 256-bit AVX2 vectors or processing single 64-bit rows in 128-bit vectors)
incurs vector instruction overhead without utilizing execution throughput or
outperforming narrower instruction sets. Packing row pairs halves loop
iterations, arithmetic instructions, and intermediate register usage.

## Examples

Bad:

```cpp
// Bad: Casting 128-bit load into half-utilized AVX2 vector
for (int i = 0; i < N; i++) {
  src_reg = _mm256_castsi128_si256(_mm_load_si128((const __m128i *)src));
  ref0_reg = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)ref0));
  src += src_stride;
}

// Bad: Loading separate 64-bit halves and unpacking, or self-packing output
const __m128i v0 = xx_loadl_64(src + 0 * stride);
const __m128i v1 = xx_loadl_64(src + 1 * stride);
const __m128i v01 = _mm_unpacklo_epi64(v0, v1);
_mm_storel_epi64((__m128i *)(dst + j), _mm_packus_epi16(roundA, roundA));
_mm_storel_epi64((__m128i *)(dst + j + stride), _mm_packus_epi16(roundB, roundB));
```

Good:

```cpp
// Good: Packing two 128-bit rows into 256-bit AVX2 registers with yy_loadu2_128
for (int i = 0; i < N; i += 2) {
  src_reg = yy_loadu2_128(src + src_stride, src);
  ref0_reg = yy_loadu2_128(ref0 + ref_stride, ref0);
  src += 2 * src_stride;
}

// Good: Combining 64-bit rows with high-half load, and packing pairs for store
const __m128i v0 = xx_loadl_64(src + 0 * stride);
const __m128i v01 = xx_loadh_64(v0, src + 1 * stride);
const __m128i round = _mm_packus_epi16(roundA, roundB);
_mm_storel_epi64((__m128i *)(dst + j), round);
_mm_storeh_pi((__m64 *)(dst + j + stride), _mm_castsi128_ps(round));
```
