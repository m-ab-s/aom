---
name: libaom/aom_dsp/unroll-short-fixed-loops-in-simd-dsp-kernels
description: In SIMD and DSP kernels, unroll short fixed-iteration loops using constant pointer offsets or compile-time immediate intrinsics instead of dynamic indexing or switch wrappers
rubric: safety
severity: should_fix
---

**Rule.** In performance-critical SIMD and DSP kernels, explicitly unroll small
loops with short, fixed iteration counts (e.g., 2–8 iterations or horizontal
vector passes for fixed block widths like 16, 32, 64, or 128) rather than using
loop counters and dynamic indexing. Access sequential vector blocks using
explicit constant pointer offsets (e.g., `ptr + 16`) instead of calculating
dynamic offsets with loop counter indices. Additionally, when extracting or
aligning SIMD vector elements with varying offsets, unroll small loops and
invoke vector extract/align intrinsics (such as ARM NEON `vextq_*` / `vext_*` or
x86 `_mm_alignr_epi8`) directly using compile-time constant literals rather than
writing dynamic wrapper functions with `switch` statements.

**Why.** Small inner loops with fixed iteration counts introduce unnecessary
loop counter increments, condition branches, and dynamic indexing calculations.
Compilers may disable automatic loop unrolling at `-O2` due to code-size
heuristics. Explicit unrolling eliminates loop branch overhead, guarantees
consistent performance across `-O2` and `-O3` optimization levels, and allows
the compiler to better interleave and schedule vector loads and arithmetic
across registers. Furthermore, intrinsics like `vextq_*` and `_mm_alignr_epi8`
require immediate constant byte offsets encoded directly into hardware
instructions; wrapping them in runtime `switch` statements introduces branch
penalties that are completely avoided by unrolling.

## Examples

Bad:

```cpp
// 1. Loop overhead and dynamic indexing for fixed 2-vector block pass (width 32):
for (int j = 0; j < 2; ++j) {
  uint8x16_t s = vld1q_u8(src + j * 16);
  uint8x16_t r = vld1q_u8(ref + j * 16);
  sse += sse_16x1_neon(s, r);
}

// 2. Switch-based wrapper to emulate runtime-variable offset in a loop:
static INLINE uint32x4_t valign(uint32x4_t l, uint32x4_t r, int offset) {
  switch (offset) {
    case 1: return vextq_u32(l, r, 1);
    case 2: return vextq_u32(l, r, 2);
    case 3: return vextq_u32(l, r, 3);
    default: return l;
  }
}
for (int i = 0; i < 4; i++) {
  vmask[i] = valign(k0, kf, 4 - i);
}
```

Good:

```cpp
// 1. Explicitly unroll fixed 2-vector pass with constant pointer offsets:
uint8x16_t s0 = vld1q_u8(src);
uint8x16_t r0 = vld1q_u8(ref);
uint8x16_t s1 = vld1q_u8(src + 16);
uint8x16_t r1 = vld1q_u8(ref + 16);
sse += sse_16x1_neon(s0, r0);
sse += sse_16x1_neon(s1, r1);

// 2. Unroll loop and pass compile-time constant literals directly:
vmask[0] = kf;
vmask[1] = vextq_u32(k0, kf, 3);
vmask[2] = vextq_u32(k0, kf, 2);
vmask[3] = vextq_u32(k0, kf, 1);
```
