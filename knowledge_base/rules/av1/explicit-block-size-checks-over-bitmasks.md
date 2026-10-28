---
name: libaom/av1/explicit-block-size-checks-over-bitmasks
description: When checking block dimensions or buffer thresholds, use direct relational comparisons (>= 8) or explicit equality (w == 4) instead of bitwise masking or modulo
rubric: readability
severity: should_fix
---

**Rule.** When checking block dimensions, buffer sizes, or value thresholds
(such as w, h, bw, bh, width, height, size, len, or stride) against boundary
conditions or discrete power-of-two sizes, prefer direct relational comparisons
(e.g., bw >= 8, width >= 16) or explicit equality checks (e.g., w == 4) rather
than bitwise masking checks (e.g., !(bw & 0x07), !(w & 0x03), (width & 15) == 0)
or modulo arithmetic (e.g., w % 4). Additionally, when a branch dispatches on a
specific discrete block size in SIMD or DSP kernels, test that size directly
with equality (e.g., w == 4) rather than using relational inequalities like w <=
4 that imply unreachable sub-4 dimensions.

**Why.** Relational comparisons (>=, >, <=, <) and explicit equality checks (==)
directly communicate the intended dimension threshold or discrete block size
requirement, making developer intent clear. Block dimensions in AV1 and DSP
kernels are constrained to discrete power-of-two sizes with a minimum of 4
(e.g., 4, 8, 16, 32, 64, 128); bitwise masking checks like !(w & 0x03) or
inequalities like w <= 4 obscure intent by implying arbitrary multiples of 4 or
non-existent sub-4 dimensions might be encountered. Compilers optimize
relational power-of-two checks as efficiently as bitwise masks.

## Examples

Bad:

```cpp
// Bitwise masking checks for block threshold and discrete size
if (!(bw & 0x07)) {
  process_8x8_block(src, dst);
} else if (!(w & 0x03)) {
  process_block_w4_neon(src, dst, w, h);
}

// Inequality check implying unreachable sub-4 block sizes
if (w <= 4) {
  process_block_w4_neon(src, dst, w, h);
}
```

Good:

```cpp
// Direct relational threshold and explicit discrete size equality check
if (bw >= 8) {
  process_8x8_block(src, dst);
} else if (w == 4) {
  process_block_w4_neon(src, dst, w, h);
}

// Explicit equality check for discrete minimum 4-wide block
if (w == 4) {
  process_block_w4_neon(src, dst, w, h);
}
```
