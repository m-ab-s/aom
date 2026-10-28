---
name: libaom/av1/maintain-highbd-and-lowbd-counterpart-parity
description: When implementing or updating high-bitdepth or low-bitdepth counterpart routines, keep function signatures, parameterization, buffer sizing, loop constructs, and algorithm logic synchronized
rubric: correctness
severity: should_fix
---

**Rule.** When implementing, modifying, or refactoring high-bitdepth
(`*_highbd`, `highbd_*`, `*_hbd`) or low-bitdepth (8-bit) counterpart functions
across C, DSP, and SIMD implementations, keep their function signatures,
parameterization, and structural implementation logic synchronized. Ensure that:
1. Function signatures, parameters, and algorithmic logic remain in sync; do not retain hardcoded constants or outdated logic in one bit-depth path when the counterpart is updated to support dynamic parameters (such as block sizes or precision controls).
2. Internal structures—including stack buffer dimensions, boundary padding formulas (e.g., `MAX_UPSAMPLE_SZ + 3`), loop constructs (`do-while` vs. `while`), and explanatory comments—mirror their counterpart rather than introducing ad-hoc sizes or divergent loop control when the underlying algorithm and vectorization structure are identical.

**Why.** High-bitdepth and low-bitdepth processing routines mirror each other
across prediction, motion estimation, filtering, and reconstruction pipelines.
Divergence in signatures, parameters, loop control, or buffer sizing causes
feature parity regressions, incorrect processing for high-bitdepth streams,
cross-architecture review difficulties, and memory safety bugs from mismatched
buffer padding.

## Examples

Bad:

```cpp
// Variant 1: Divergent signature and parameterization (hardcoded OF_BSIZE vs flexible n in lowbd)
int opfl_mv_refinement_nxn_highbd(const uint16_t *p0, int pstride0,
                                  const uint16_t *p1, int pstride1,
                                  const int16_t *gx0, const int16_t *gy0,
                                  const int16_t *gx1, const int16_t *gy1,
                                  int gstride, int bw, int bh, int d0, int d1,
                                  int grad_prec_bits, int mv_prec_bits,
                                  int *vx0, int *vy0, int *vx1, int *vy1) {
  assert(bw % OF_BSIZE == 0 && bh % OF_BSIZE == 0);
  for (int i = 0; i < bh; i += OF_BSIZE) {
    for (int j = 0; j < bw; j += OF_BSIZE) {
      av1_opfl_mv_refinement_highbd(..., OF_BSIZE, OF_BSIZE, ...);
    }
  }
}

// Variant 2: Divergent stack buffer sizing and loop construct (diverging from 8-bit version)
void av1_highbd_upsample_intra_edge_neon(uint16_t *p, int sz, int bd) {
  uint16_t edge[141];  // Ad-hoc buffer size diverging from 8-bit formula
  while (sz > 0) {
    sz -= 8;
  }
}
```

Good:

```cpp
// Variant 1: Synchronized signature and flexible parameterization
int opfl_mv_refinement_nxn_highbd(const uint16_t *p0, int pstride0,
                                  const uint16_t *p1, int pstride1,
                                  const int16_t *gx0, const int16_t *gy0,
                                  const int16_t *gx1, const int16_t *gy1,
                                  int gstride, int bw, int bh, int n, int d0,
                                  int d1, int grad_prec_bits, int mv_prec_bits,
                                  int *vx0, int *vy0, int *vx1, int *vy1) {
  assert(bw % n == 0 && bh % n == 0);
  for (int i = 0; i < bh; i += n) {
    for (int j = 0; j < bw; j += n) {
      av1_opfl_mv_refinement_highbd(..., n, n, ...);
    }
  }
}

// Variant 2: Normalized buffer sizing and loop construct matching lowbd counterpart
void av1_highbd_upsample_intra_edge_neon(uint16_t *p, int sz, int bd) {
  uint16_t edge[MAX_UPSAMPLE_SZ + 3];  // Identical padding and size formula
  do {
    sz -= 8;
  } while (sz > 0);
}
```
