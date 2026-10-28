---
name: libaom/scale-allocation-by-element-sizeof
description: When allocating dynamic buffers with aom_malloc, aom_memalign, or malloc, multiply element counts by sizeof and pass a single byte-size expression
rubric: safety
severity: blocking
---

**Rule.** When allocating dynamic memory with aom_malloc, aom_memalign, malloc,
or realloc, pass a single total byte size computed by multiplying the element
count by the element size (e.g., count * sizeof(*ptr) or count * sizeof(Type)).
Do not pass an unscaled element count or pass count and size as separate
arguments (use aom_calloc(count, size) when allocating with separate count and
size parameters).

**Why.** aom_malloc, aom_memalign, and malloc accept a single argument
representing total size in bytes rather than element counts. Allocating without
multiplying by sizeof creates an undersized heap buffer, causing out-of-bounds
memory corruption. Passing count and element size as separate arguments to
aom_malloc causes a signature mismatch.

## Examples

Bad:

```cpp
// Incorrect: Allocating an unscaled element count creates an undersized buffer.
buf2[0] = (float *)aom_malloc(newallocsize);

// Incorrect: Passing separate count and size arguments to a single-argument allocator.
CHECK_MEM_ERROR(cm, cpi->consec_zero_mv,
                aom_malloc(current_size, sizeof(*cpi->consec_zero_mv)));
```

Good:

```cpp
// Correct: Scale count by sizeof element to allocate the required bytes.
buf2[0] = (float *)aom_malloc(newallocsize * sizeof(*buf2[0]));

// Correct: Pass a single multiplied byte size expression (or use aom_calloc for separate args).
CHECK_MEM_ERROR(cm, cpi->consec_zero_mv,
                aom_malloc(current_size * sizeof(*cpi->consec_zero_mv)));
```
