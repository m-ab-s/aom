---
name: libaom/avoid-fatal-exit-in-library-code
description: When handling errors or allocation failures in library functions, return an error status or return early and free allocated resources instead of calling exit() or abort()
rubric: correctness
severity: blocking
---

**Rule.** Do not call `exit()`, `abort()`, `_Exit()`, `quick_exit()`, or other
process-terminating functions upon errors or memory allocation failures in
library code. When an error or allocation failure occurs, release all locally
allocated resources (such as via `aom_free()`) and return an error status (e.g.
`AOM_CODEC_MEM_ERROR` or `0`) or return early to allow caller recovery.

**Why.** libaom is compiled as a shared or static library embedded into external
host applications. Abruptly terminating the process with `exit()` or `abort()`
crashes the host application without allowing graceful error recovery, and
failing to free locally allocated buffers before returning causes memory leaks
on error paths.

## Examples

Bad:

```cpp
uint8_t *src_rgb = (uint8_t *)aom_malloc(buffer_size);
uint8_t *dst_rgb = (uint8_t *)aom_malloc(buffer_size);
if (!src_rgb || !dst_rgb) {
  fprintf(stderr, "Fatal error: calculation failed.\n");
  exit(EXIT_FAILURE);
}

uint16_t *im_block = (uint16_t *)aom_memalign(16, size);
if (!im_block) {
  abort();
}
```

Good:

```cpp
uint8_t *src_rgb = (uint8_t *)aom_malloc(buffer_size);
uint8_t *dst_rgb = (uint8_t *)aom_malloc(buffer_size);
if (!src_rgb || !dst_rgb) {
  aom_free(src_rgb);
  aom_free(dst_rgb);
  return 0;
}

uint16_t *im_block = (uint16_t *)aom_memalign(16, size);
if (!im_block) return;
uint16_t *im_block2 = (uint16_t *)aom_memalign(16, size);
if (!im_block2) {
  aom_free(im_block);
  return;
}
```
