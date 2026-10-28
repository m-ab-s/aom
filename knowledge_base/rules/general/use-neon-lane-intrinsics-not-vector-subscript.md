---
name: libaom/use-neon-lane-intrinsics-not-vector-subscript
description: On ARM NEON and SIMD vector types, use standard intrinsics (vadd, vmul, vget_lane, vset_lane, vdup_n, vld1) instead of compiler vector operators, array subscripting, or brace initialization
rubric: correctness
severity: blocking
---

**Rule.** In ARM NEON and SIMD implementations, do not use compiler-specific
vector extensions—such as arithmetic and compound assignment operators (+, -, *,
+=, -=, *=), shift and bitwise operators (<<, >>, &, |, ^), array subscript
indexing (vec[i]), or aggregate/zero brace initialization ({ ... }, { 0 }, { { 0
} })—on vector types (such as uint64x2_t, uint32x4_t, int32x4_t, int16x8_t,
int16x4_t, uint8x16_t, __m128i) or arrays of vectors. Instead, initialize vector
constants and sequential vectors by loading from file-scope static const arrays
with standard load intrinsics (such as vld1[q]_*, _mm_loadu_*) or set intrinsics
(such as _mm_setr_*, vdup[q]_n_*). Use standard intrinsics for arithmetic and
logic (vadd[q]_*, vsub[q]_*, vmul[q]_*, vshl[q]_*, vshr[q]_*, vand[q]_*,
vorr[q]_*, veor[q]_*), lane access (vget[q]_lane_*, vset[q]_lane_*), and leave
overwritten vector arrays uninitialized.

**Why.** In Microsoft Visual C++ (MSVC) for Windows on ARM as well as standard C
SIMD headers, vector types are defined as opaque union or struct types rather
than GCC/Clang vector extension types. Relying on GNU/Clang vector operator
overloads, array subscripting, or brace aggregate initialization breaks
compilation on MSVC. Direct brace initialization of vector constants is
non-portable across C compilers. Additionally, zero-initializing vector arrays
with brace syntax when they are immediately overwritten generates redundant
stores on function entry. Loading constant sequences from file-scope static
const arrays with standard load intrinsics and using standard arithmetic/lane
intrinsics guarantees cross-compiler portability, consistent code generation,
and optimal performance.

## Examples

Bad:

```cpp
// 1. Non-portable brace initialization on vectors & sequential constants:
const int16x4_t iota = { 0, 1, 2, 3 };
const int16x8_t iota_seq = { 0, 1, 2, 3, 4, 5, 6, 7 };
int16x8_t temp_buf[8] = { { 0 } };

// 2. Vector arithmetic, shifts, and compound assignment operators:
const int16x4_t x0123 = vdup_n_s16(x0) + (iota << 6);
uint64x2_t sum = vdupq_n_u64(0);
sum += mse_wxh_16bit(dst, dstride, src, w, w, h);

// 3. Array subscript indexing on NEON vector types:
s[0] = s[2];
ret = vld1_lane_u8(src + idx[0], ret, 0);
```

Good:

```cpp
// 1. Load constants from file-scope static const arrays via vld1_*; leave overwritten buffers uninitialized:
static const int16_t iota_s16[] = { 0, 1, 2, 3 };
static const int16_t iota1_s16[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
const int16x4_t iota = vld1_s16(iota_s16);
const int16x8_t iota_seq = vld1q_s16(iota1_s16);
int16x8_t temp_buf[8];

// 2. Use standard NEON intrinsics for vector arithmetic and shifts:
const int16x4_t x0123 = vadd_s16(vdup_n_s16(x0), vshl_n_s16(iota, 6));
uint64x2_t sum = vdupq_n_u64(0);
sum = vaddq_u64(sum, mse_wxh_16bit(dst, dstride, src, w, w, h));

// 3. Use dedicated NEON lane extraction and insertion intrinsics:
s = vsetq_lane_u16(vgetq_lane_u16(s, 2), s, 0);
ret = vld1_lane_u8(src + vget_lane_s16(vget_low_s16(idx), 0), ret, 0);
```
