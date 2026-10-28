---
name: libaom/aom_dsp/pass-all-x86-vectors-as-pointers-over-three-args
description: When defining SIMD helper functions with multiple vector parameters, pass vector types (__m128i, __m256i, __m512i) as pointers rather than by value or mixed conventions
rubric: correctness
severity: should_fix
---

**Rule.** When defining SIMD helper functions that accept vector parameters
(such as `__m128i`, `__m256i`, or `__m512i`), pass vector arguments by pointer
(e.g., `const __m256i *` for read-only inputs or `__m256i *` for mutated/in-out
accumulators) rather than by value. Consistently pass all SIMD vector parameters
as pointers rather than mixing value and pointer parameters, and pass addresses
of array-resident vectors (`&vec[i]`) directly to pointer-based helper
functions.

**Why.** Platform ABIs and calling conventions (such as 32-bit x86 MSVC on
Windows and Windows x64) pass only a limited number of vector parameters in
registers (typically 3 or 4). Passing large SIMD vectors by value or passing
excess parameters spills arguments onto the stack without guaranteed 16- or
32-byte alignment, causing compiler errors (e.g., MSVC C2719),
calling-convention mismatches, and register spills. Passing vector parameters by
pointer avoids alignment hazards and reduces register pressure while inlining
compilers optimize pointer indirection with zero runtime overhead.

## Examples

Bad:

```cpp
// Anti-pattern 1: Passing multiple __m256i vector parameters by value
static INLINE __m256i fold_mul_and_sum_avx2(__m256i partiala, __m256i partialb,
                                            __m256i const1, __m256i const2) {
  const __m256i tmp = _mm256_madd_epi16(partiala, const1);
  return _mm256_add_epi32(tmp, _mm256_madd_epi16(partialb, const2));
}

// Anti-pattern 2: Mixing value and pointer vector parameters
static INLINE __m128i filter_block_2rows(const __m128i a0, const __m128i b0,
                                         const __m128i a1, const __m128i *b1,
                                         const __m128i *filter);

// Anti-pattern 3: Passing array elements by value
res = hadd_four_32_to_64_avx2(sum[0], sum[1], sum[2], sum[3]);
```

Good:

```cpp
// Pattern 1: Pass vector inputs/accumulators as const or mutable pointers
static INLINE __m256i fold_mul_and_sum_avx2(__m256i *partiala, __m256i *partialb,
                                            const __m256i *const1,
                                            const __m256i *const2) {
  const __m256i tmp = _mm256_madd_epi16(*partiala, *const1);
  return _mm256_add_epi32(tmp, _mm256_madd_epi16(*partialb, *const2));
}

// Pattern 2: Consistently pass all vector parameters as pointers
static INLINE __m128i filter_block_2rows(const __m128i *a0, const __m128i *b0,
                                         const __m128i *a1, const __m128i *b1,
                                         const __m128i *filter);

// Pattern 3: Pass memory-resident array elements by pointer address
res = hadd_four_32_to_64_avx2(&sum[0], &sum[1], &sum[2], &sum[3]);
```
