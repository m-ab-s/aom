---
name: libaom/aom_dsp/omit-simd-chunking-loops-when-accumulators-cannot-overflow
description: When implementing SIMD or DSP accumulation kernels, omit chunking loops when accumulators cannot overflow, or hardcode fixed chunking when only 128x128 overflows
rubric: safety
severity: should_fix
---

**Rule.** In SIMD or DSP accumulation kernels (such as variance, SAD, SSE, or
sum of squares), check theoretical maximum sums against accumulator bit-depth
limits across discrete block sizes. If the maximum sum cannot overflow 32-bit
vector registers for any supported size, eliminate chunking loops and accumulate
directly in 32-bit registers with a single post-loop 64-bit reduction. When
accumulator overflow is only possible for the largest discrete block size (e.g.,
128x128), branch directly on that size and use fixed chunk constants instead of
dynamic pixel multiplication and division.

**Why.** Intermediate chunking sub-loops, horizontal reductions, and dynamic
dimension calculations (w * h, limit / w) introduce unnecessary arithmetic,
branching, and register shuffling in hot DSP loops. Determining overflow
boundaries statically against known discrete block sizes maximizes vector
pipeline throughput and avoids runtime division.

## Examples

Bad:

```cpp
// Bad: Unnecessary intermediate chunking and reduction when max sum fits in 32-bit
do {
  __m128i v_acc_q = _mm_setzero_si128();
  const int step = AOMMIN(height - r, 32);
  for (int j = 0; j < step; j += 4) { ... }
  v_acc_q = _mm_add_epi32(v_acc_q, _mm_srli_epi64(v_acc_q, 32));
  sum += (uint64_t)_mm_cvtsi128_si32(v_acc_q);
} while (r < height);

// Bad: Dynamic pixel multiplication and division when only 128x128 can overflow
} else if (w * h <= max_pel_allowed_per_ovf) {
  hbd_obmc_variance_w8n(pre, stride, wsrc, mask, &sse, &sum, w, h);
} else {
  int h_per_ovf = max_pel_allowed_per_ovf / w;
  hbd_obmc_variance_w8n(pre, stride, wsrc, mask, &sse, &sum, w, h_per_ovf);
}
```

Good:

```cpp
// Good: Single continuous accumulation loop with post-loop 64-bit widening
do {
  const __m128i v_acc_d = sum_squares_i16_4x4_sse2(src, stride);
  v_acc_q = _mm_add_epi32(v_acc_q, v_acc_d);
  src += stride << 2;
  r += 4;
} while (r < height);
return (uint64_t)xx_cvtsi128_si64(reduce_to_64(v_acc_q));

// Good: Direct dimension check with fixed constant chunk height
} else if (w < 128 || h < 128) {
  hbd_obmc_variance_w8n(pre, stride, wsrc, mask, &sse, &sum, w, h);
} else {
  assert(w == 128 && h == 128);
  hbd_obmc_variance_w8n(pre, stride, wsrc, mask, &sse, &sum, w, 64);
}
```
