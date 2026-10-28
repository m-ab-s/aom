---
name: libaom/check-memory-allocation-failure
description: Validate dynamic memory allocations with runtime NULL checks or CHECK_MEM_ERROR rather than assert(), release resources on errors, and return fallible status
rubric: correctness
severity: blocking
---

**Rule.** Always verify that return pointers from dynamic memory allocation
functions (aom_malloc, aom_calloc, aom_memalign, aom_realloc, malloc, calloc,
realloc) and opaque resource handles are non-NULL using runtime checks
(CHECK_MEM_ERROR, AOM_CHECK_MEM_ERROR, explicit NULL checks, or abort()) rather
than debug-only assert() or aom_assert(). In codec routines where an error
context (AV1_COMMON *cm or struct aom_internal_error_info *error_info) is
available, wrap allocations in CHECK_MEM_ERROR(cm, ptr, expr) or
AOM_CHECK_MEM_ERROR(error_info, ptr, expr). On allocation failure, validation
error, or early-exit path, explicitly release all locally allocated buffers
(aom_free / free) and destroy acquired handles, or route execution through a
shared cleanup label (goto fail; or goto cleanup;). Additionally, helper
functions that perform dynamic memory allocation or fallible operations must
return a status code (e.g., non-zero on success, zero on error, or an
aom_codec_err_t) rather than returning void, and all call sites must check and
handle the returned status. Never rely on assert() or aom_assert() for
allocation checks since assertions are stripped in release builds when NDEBUG is
defined.

**Why.** Dynamic memory allocation and handle creation functions can return NULL
upon memory exhaustion or invalid sizing. Dereferencing, indexing, or performing
pointer arithmetic on unchecked pointers causes segmentation faults and
undefined behavior. Relying on assert() or aom_assert() leaves release builds
unprotected because assertions are compiled out when NDEBUG is defined. Using
CHECK_MEM_ERROR or AOM_CHECK_MEM_ERROR raises aom_internal_error with
AOM_CODEC_MEM_ERROR safely, while explicit runtime NULL checks combined with
proper cleanup prevent progressive memory and handle leaks along error paths.
Furthermore, returning void from fallible functions causes callers to proceed
with uninitialized or invalid buffers, leading to silent data corruption or
crashes.

## Examples

Bad:

```cpp
// Variant 1: Unchecked allocation or missing cleanup
size_t size = width * height * 3;
uint8_t *src_rgb = (uint8_t *)aom_malloc(size);
uint8_t *dst_rgb = (uint8_t *)aom_malloc(size);
I420ToRGB24Matrix(source->y_buffer, ..., src_rgb, dst_rgb);

// Variant 2: assert() stripped in release builds with NDEBUG
syms = realloc(syms, sizeof(Symbol) * count);
assert(syms != NULL);

// Variant 3: Leaking handle or void function ignoring allocation error
JxlButteraugliApi *api = JxlButteraugliApiCreate(NULL);
JxlButteraugliResult *res = JxlButteraugliCompute(api, ...);
if (!res) return;  // Leaks api handle and caller cannot detect error
```

Good:

```cpp
// Variant 1: Verify allocations, free prior buffers on failure
size_t size = width * height * 3;
uint8_t *src_rgb = (uint8_t *)aom_malloc(size);
uint8_t *dst_rgb = (uint8_t *)aom_malloc(size);
if (!src_rgb || !dst_rgb) {
  aom_free(src_rgb);
  aom_free(dst_rgb);
  return 0;
}

// Variant 2: Explicit runtime check or abort() instead of assert
syms = realloc(syms, sizeof(Symbol) * count);
if (!syms) abort();

// Variant 3: Clean up acquired handle and return fallible status code
JxlButteraugliApi *api = JxlButteraugliApiCreate(NULL);
JxlButteraugliResult *res = JxlButteraugliCompute(api, ...);
if (!res) {
  if (api) JxlButteraugliApiDestroy(api);
  return 0;
}
```
