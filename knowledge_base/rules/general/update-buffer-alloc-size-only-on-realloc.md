---
name: libaom/update-buffer-alloc-size-only-on-realloc
description: When allocating or reallocating buffers, update allocation and dimension tracking fields strictly inside initial setup or reallocation blocks, not in downstream helpers or per-frame routines
rubric: correctness
severity: blocking
---

**Rule.** When allocating, initializing, or dynamically resizing buffers, update
allocation and dimension tracking fields (such as *_alloc_size,
allocated_tile_cols/rows, allocated_tiles, or initial_dimensions) strictly
within the function or conditional block that performs the allocation or initial
setup. Do not set or clear tracking variables in downstream helper functions, do
not reset them to zero upon reallocation, and do not update them unconditionally
in per-frame routines outside setup or reallocation branches.

**Why.** Setting dimension and capacity trackers directly where buffers are
allocated or initialized ensures lifecycle consistency and guarantees error
handlers and resize checks operate on accurate reference state. Updating
trackers outside reallocation blocks or unconditionally in per-frame routines
corrupts baseline dimensions and capacity tracking, causing redundant
reallocations and breaking dynamic resolution changes.

## Examples

Bad:

```cpp
// Anti-pattern 1: Setting allocation dimension trackers in downstream helper rather than allocation function
// In downstream helper row_mt_mem_alloc():
enc_row_mt->allocated_tile_cols = tile_cols;
enc_row_mt->allocated_tile_rows = tile_rows;

// Anti-pattern 2: Updating allocation size unconditionally outside reallocation branch
if (cpi->consec_zero_mv && cpi->consec_zero_mv_alloc_size < current_size) {
  aom_free(cpi->consec_zero_mv);
  CHECK_MEM_ERROR(cm, cpi->consec_zero_mv,
                  aom_calloc(current_size, sizeof(*cpi->consec_zero_mv)));
}
cpi->consec_zero_mv_alloc_size = current_size;

// Anti-pattern 3: Updating baseline dimensions unconditionally in per-frame routine
void av1_check_initial_width(AV1_COMP *cpi, ...) {
  cpi->initial_dimensions.width = cm->width;
  cpi->initial_dimensions.height = cm->height;
  if (!cpi->frame_size_related_setup_done) {
    alloc_compressor_data(cpi);
    cpi->frame_size_related_setup_done = true;
  }
}
```

Good:

```cpp
// Pattern 1: Set dimension tracking directly in allocation function av1_alloc_tile_data()
CHECK_MEM_ERROR(
    cm, cpi->tile_data,
    aom_memalign(32, tile_cols * tile_rows * sizeof(*cpi->tile_data)));
cpi->allocated_tiles = tile_cols * tile_rows;
enc_row_mt->allocated_tile_cols = tile_cols;
enc_row_mt->allocated_tile_rows = tile_rows;

// Pattern 2: Update allocation size strictly inside the reallocation block
if (cpi->consec_zero_mv && cpi->consec_zero_mv_alloc_size < current_size) {
  aom_free(cpi->consec_zero_mv);
  CHECK_MEM_ERROR(cm, cpi->consec_zero_mv,
                  aom_calloc(current_size, sizeof(*cpi->consec_zero_mv)));
  cpi->consec_zero_mv_alloc_size = current_size;
}

// Pattern 3: Update baseline dimensions strictly inside the initial setup block
void av1_check_initial_width(AV1_COMP *cpi, ...) {
  if (!cpi->frame_size_related_setup_done) {
    alloc_compressor_data(cpi);
    cpi->initial_dimensions.width = cm->width;
    cpi->initial_dimensions.height = cm->height;
    cpi->frame_size_related_setup_done = true;
  }
}
```
