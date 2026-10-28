---
name: libaom/av1/hoist-scratch-buffer-allocations
description: Avoid heap allocations in block-level or inner-loop routines by using bounded stack arrays or hoisting scratch buffers to frame or context setup
rubric: safety
severity: should_fix
---

**Rule.** Avoid allocating and freeing dynamic heap memory (such as via
malloc(), calloc(), aom_malloc(), or aom_memalign()) inside per-block,
prediction, search, or inner-loop DSP/SIMD routines. For temporary buffers with
bounded dimensions (e.g., maximum superblock size 128x128), use fixed-size
stack-allocated arrays guarded by dimension assertions (e.g., assert(bw <= 128
&& bh <= 128)). For large, aligned, or reusable scratch buffers (such as SIMD
restoration unit stats buffers), hoist dynamic memory allocations to frame-,
tile-, or search-level context structures (such as pick_lr_ctxt) and pass buffer
pointers into processing kernels.

**Why.** Performing dynamic memory allocation and deallocation inside per-block
or inner-loop functions introduces substantial heap allocator overhead, system
calls, cache penalties, and error-handling failure paths in performance-critical
code paths. Fixed-size stack arrays provide fast, zero-allocation local storage
for bounded block buffers, while hoisting allocations to frame or context setup
amortizes allocation overhead across all blocks and eliminates per-block failure
checks.

## Examples

Bad:

```cpp
// Bad: Dynamic heap allocation for a bounded prediction block
uint8_t *pred_block = (uint8_t *)malloc(bw * bh * sizeof(uint8_t));
if (!pred_block) return;
generate_prediction(pred_block, bw, bh);
free(pred_block);

// Bad: Dynamic scratch memory allocation inside per-unit SIMD routine
int16_t *dgd_avg = (int16_t *)aom_memalign(
    32, sizeof(*dgd_avg) * 6 * RESTORATION_UNITSIZE_MAX * RESTORATION_UNITSIZE_MAX);
// ... compute stats ...
aom_free(dgd_avg);
```

Good:

```cpp
// Good: Fixed-size stack array bounded by maximum superblock dimensions
assert(bw <= 128 && bh <= 128);
uint8_t pred_block[128 * 128];
const int pred_block_stride = 128;
generate_prediction(pred_block, bw, bh, pred_block_stride);

// Good: Hoist scratch allocations to frame or context setup structures
CHECK_MEM_ERROR(cm, cpi->pick_lr_ctxt.dgd_avg,
                (int16_t *)aom_memalign(32, buf_size));
rsc.dgd_avg = cpi->pick_lr_ctxt.dgd_avg;

void av1_compute_stats_avx2(int wiener_win, const uint8_t *dgd, const uint8_t *src,
                            int16_t *dgd_avg, int16_t *src_avg, ...) {
  // Use passed-in scratch buffers directly without allocating
}
```
