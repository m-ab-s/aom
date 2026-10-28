---
name: libaom/aom_dsp/load-only-four-bytes-for-four-wide-simd-blocks
description: When loading narrow block rows or reference buffers in x86 SIMD routines, use width-matched loads like xx_loadl_32/64 rather than full-register loads
rubric: safety
severity: should_fix
---

**Rule.** In x86 SIMD routines, use width-matched load helpers or intrinsics
(such as xx_loadl_32 or _mm_cvtsi32_si128 for 4-byte/32-bit rows, and
xx_loadl_64 or _mm_loadl_epi64 for 8-byte/64-bit rows) when reading lines,
reference buffers, or mask rows narrower than full vector registers. Do not use
wider or full-register loads (such as _mm_loadl_epi64 on 4-byte buffers or
xx_loadu_128 / _mm_loadu_si128 on 8-byte rows) on buffers that are only
allocated or valid for the narrower width.

**Why.** Loading 8 or 16 bytes from buffers or row strides that only allocate 4
or 8 valid bytes over-reads past allocated buffer boundaries or row strides.
This causes memory over-read errors, AddressSanitizer violations, or potential
segmentation faults when processing narrow blocks near image or buffer edges.

## Examples

Bad:

```cpp
// 4-byte reference buffer: over-reading 8 bytes with _mm_loadl_epi64
__m128i d = _mm_loadl_epi64((const __m128i *)above);

// 8-byte mask row: over-reading 16 bytes with xx_loadu_128
const __m128i v_ra = xx_loadu_128(mask);
const __m128i v_rb = xx_loadu_128(mask + mask_stride);
```

Good:

```cpp
// 4-byte reference buffer: load exact 4 bytes (32 bits)
__m128i d = _mm_cvtsi32_si128(*(const uint32_t *)above);

// 8-byte mask row: load exact 8 bytes (64 bits)
const __m128i v_ra = xx_loadl_64(mask);
const __m128i v_rb = xx_loadl_64(mask + mask_stride);
```
