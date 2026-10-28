---
name: libaom/cast-subwords-to-unsigned-before-bitwise-packing
description: When bitwise packing sub-word integers or right-shifting SIMD-extracted scalar data, cast operands to unsigned types to prevent arithmetic sign-extension bugs
rubric: correctness
severity: blocking
---

**Rule.** When packing sub-word values (such as two 16-bit integers into a
32-bit scalar or SIMD constant) via bitwise shifts and OR operations, or when
right-shifting packed data extracted from SIMD registers (such as via
_mm_cvtsi128_si32), explicitly cast the operands to unsigned types (such as
uint16_t and uint32_t) before shifting. Do not perform bitwise left-shifts, ORs,
or logical right-shifts on signed integer types.

**Why.** In C and C++, right-shifting a signed integer performs an arithmetic
shift (sign extension), corrupting upper sub-word values when the sign bit is
set. Similarly, casting negative signed values to wider signed types fills upper
bits with ones, corrupting bitwise ORed values, and left-shifting negative
numbers is undefined behavior. Unsigned types guarantee zero-extension and
well-defined logical bitwise shifts.

## Examples

Bad:

```cpp
// Bad: Signed operands cause sign-extension during bitwise packing and shifting
#define pair_set_epi16(a, b) \
  _mm_set1_epi32(((int32_t)((int16_t)(a))) | (((int32_t)(b)) << 16))

int sum32 = _mm_cvtsi128_si32(sum);
sum32 >>= 16;  // Arithmetic shift: sign-extends if bit 31 is set
```

Good:

```cpp
// Good: Unsigned types ensure zero-extension and logical shift semantics
#define pair_set_epi16(a, b) \
  _mm_set1_epi32((int32_t)(((uint16_t)(a)) | (((uint32_t)(b)) << 16)))

uint32_t sum32 = (uint32_t)_mm_cvtsi128_si32(sum);
sum32 >>= 16;  // Logical shift: preserves upper 16 bits without sign extension
```
