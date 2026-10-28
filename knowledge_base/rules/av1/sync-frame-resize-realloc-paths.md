---
name: libaom/av1/sync-frame-resize-realloc-paths
description: When updating buffer reallocation conditions or dimension tracking in encoder routines, synchronize all parallel execution paths (single-thread, tile MT, row MT, first pass) identically
rubric: correctness
severity: should_fix
---

**Rule.** When modifying buffer reallocation checks, dimension tracking, or
state initialization across encoder routines, ensure all parallel execution
paths (single-threaded frame encode in encodeframe.c, tile multi-threading in
av1_encode_tiles_mt, row multi-threading in av1_encode_tiles_row_mt /
av1_row_mt_mem_alloc, and first-pass multi-threading in firstpass.c /
first_pass_row_mt_mem_alloc) and entry points (such as av1_change_config(),
set_size_literal(), or av1_check_initial_width()) are updated consistently.
Verify that reallocation conditions (such as testing exact tile count mismatch
allocated_tiles != tile_cols * tile_rows rather than <) match across all paths,
and ensure dimension updates and setup reset flags (such as initial_dimensions
and frame_size_related_setup_done) are synchronized when triggering compressor
or segmentation map reallocations (alloc_compressor_data(),
realloc_segmentation_maps(), av1_alloc_tile_data()).

**Why.** Encoder logic is split across multiple independent single-threaded and
multi-threaded pathways and passes that manage shared or parallel buffer
allocations. If reallocation conditions diverge (e.g., checking only growth <
instead of any mismatch !=), buffers become desynchronized from multi-threading
synchronization structures (such as row_mt_sync) or skip required reallocations
during dynamic resolution changes, leading to null-pointer dereferences, memory
corruption, or stale configuration state.

## Examples

Bad:

```cpp
// Anti-pattern 1: Reallocation condition divergence across threading paths and passes
// In ethread.c (av1_encode_tiles_row_mt): updated to check mismatch
const bool alloc_tile_data = cpi->allocated_tiles != tile_cols * tile_rows;
// In ethread.c (av1_encode_tiles_mt) and firstpass.c: missed update, still checks <
const bool alloc_tile_data = cpi->allocated_tiles < tile_cols * tile_rows;

// Anti-pattern 2: Omitting setup flag reset in parallel resize entry points
if (cm->width > cpi->data_alloc_width || cm->height > cpi->data_alloc_height) {
  alloc_compressor_data(cpi);
  realloc_segmentation_maps(cpi);
  cpi->initial_dimensions.width = cpi->initial_dimensions.height = 0;
}
```

Good:

```cpp
// Pattern 1: Reallocation condition consistently updated across all parallel paths
// In ethread.c (av1_encode_tiles_row_mt, av1_encode_tiles_mt) and firstpass.c:
const bool alloc_tile_data = cpi->allocated_tiles != tile_cols * tile_rows;
if (alloc_tile_data) av1_alloc_tile_data(cpi);

// Pattern 2: Resize paths synchronized with matching dimension updates and flag resets
if (cm->width > cpi->data_alloc_width || cm->height > cpi->data_alloc_height) {
  alloc_compressor_data(cpi);
  realloc_segmentation_maps(cpi);
  cpi->initial_dimensions.width = cm->width;
  cpi->initial_dimensions.height = cm->height;
  cpi->frame_size_related_setup_done = false;
}
```
