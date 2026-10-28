---
name: libaom/av1/terminate-workers-before-freeing-worker-arrays
description: When freeing or reconfiguring AVxWorker thread arrays, terminate active workers before freeing memory, and handle teardown at caller layers rather than inside creation helpers
rubric: design
severity: blocking
---

**Rule.** Before freeing or reallocating AVxWorker thread arrays or their
associated context structures (such as p_mt_info.workers or tile_workers),
explicitly terminate all active worker threads (e.g. calling
av1_terminate_workers() or winterface->end()) and deallocate per-thread data.
Additionally, perform worker teardown and reconfiguration at the caller or codec
interface layer (e.g., av1_cx_iface.c) rather than embedding conditional
teardown inside worker creation helpers (e.g., av1_create_workers()), keeping
creation helpers focused solely on allocation and asserting an uninitialized
state (assert(num_workers == 0)).

**Why.** Freeing or reallocating AVxWorker structures while worker threads are
active leaves background threads accessing deallocated memory and causes
use-after-free crashes or data races. Embedding teardown inside creation helpers
mixes lifecycle responsibilities and risks incomplete cleanup when dependent
context and thread data must be coordinated by the caller.

## Examples

Bad:

```cpp
// Anti-pattern 1: Freeing worker array while threads may still be active
if (p_mt_info->num_workers > 0) {
  aom_free(p_mt_info->workers);
  p_mt_info->workers = NULL;
}

// Anti-pattern 2: Embedding conditional teardown inside creation helper
void av1_create_workers(AV1_PRIMARY *ppi, int num_workers) {
  if (p_mt_info->num_workers > 0) {
    av1_terminate_workers(ppi);
    aom_free(p_mt_info->tile_thr_data);
    p_mt_info->tile_thr_data = NULL;
    aom_free(p_mt_info->workers);
    p_mt_info->workers = NULL;
    p_mt_info->num_workers = 0;
  }
  // Allocate new workers...
}
```

Good:

```cpp
// In caller / codec interface layer (e.g., av1_cx_iface.c):
if (ppi->p_mt_info.num_workers > 0) {
  av1_terminate_workers(ppi);
  free_thread_data(ppi);
  aom_free(ppi->p_mt_info.tile_thr_data);
  ppi->p_mt_info.tile_thr_data = NULL;
  aom_free(ppi->p_mt_info.workers);
  ppi->p_mt_info.workers = NULL;
  ppi->p_mt_info.num_workers = 0;
}
av1_create_workers(ppi, num_workers);

// In ethread.c (creation helper asserts uninitialized state):
void av1_create_workers(AV1_PRIMARY *ppi, int num_workers) {
  assert(p_mt_info->num_workers == 0);
  // Allocate new workers...
}
```
