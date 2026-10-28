---
name: libaom/av1/size-scratch-buffers-tightly
description: When allocating heap, stack scratch, zero-reference, or SIMD intermediate buffers, size them tightly to required strides, strip widths, or maximum transform dimensions without unused padding.
rubric: safety
severity: should_fix
---

**Rule.** When allocating heap, stack scratch, prediction, zero-reference, or
intermediate SIMD vector buffers, size dimensions tightly based on the actual
required stride and block dimensions (e.g., using fixed subblock constants like
SUBBLOCK_SIZE instead of MAX_SB_SIZE for fixed-size routines, or exact element
counts for local vector arrays). In multi-pass SIMD filtering routines (such as
2D convolution or Wiener filtering) that process blocks in column-wise strips,
size intermediate stack buffers (e.g., im_block) to the active strip width (such
as 8 or 4 columns) rather than the full superblock width (MAX_SB_SIZE).
Additionally, ensure multi-size routines accommodate the maximum supported
transform or block dimension (such as 64x8 vector registers for 64x64
transforms) to prevent out-of-bounds writes. Do not allocate arbitrary oversized
buffers or unused padding dimensions, remove obsolete comments describing former
padding requirements, and document size derivations using named constants rather
than magic numbers.

**Why.** Over-allocating temporary scratch, stack, or zero-reference
buffers—such as allocating full-superblock intermediate buffers ((MAX_SB_SIZE +
taps) * MAX_SB_SIZE) when filtering data is consumed per column strip—causes
stack frame bloat, wastes memory, and degrades data cache locality. Retaining
unused padding or stale layout comments misleads maintainers about memory
safety, stride calculations, and vector access invariants. Conversely,
undersizing intermediate buffers causes out-of-bounds memory accesses and buffer
overflows when handling maximum supported block dimensions.

## Examples

Bad:

```cpp
// Variant 1: Heap scratch buffer with arbitrary magic dimensions
const int offset = 8;
const int dst_buffer_stride = 128;
const int buf_width = 256;
const int buf_height = 256;
CHECK_MEM_ERROR(cm, pred_buffer,
                aom_memalign(32, buf_width * buf_height * sizeof(*pred_buffer)));

// Variant 2: Column-strip SIMD intermediate buffer oversized to full superblock width
DECLARE_ALIGNED(32, int16_t,
                im_block[(MAX_SB_SIZE + MAX_FILTER_TAP - 1) * MAX_SB_SIZE]);

// Variant 3: Undersized intermediate buffer for multi-size transform routines
int16x8_t intermediate[32];  // Bad: Undersized for 64x64 transforms (requires 64*8)
process_txfm_neon(input, intermediate, txfm_size);

// Variant 4: Local SIMD buffer with unused padding or oversized zero-reference
uint32_t frame_sse[(BW + 4) * BH] = { 0 };
DECLARE_ALIGNED(16, static const uint8_t, av1_all_zeros[MAX_SB_SIZE]) = { 0 };
```

Good:

```cpp
// Variant 1: Sized tightly with documented named constants and strides
#define MB_WIENER_PRED_BLOCK_SIZE BLOCK_128X128
#define MB_WIENER_PRED_BUF_STRIDE 128
const int offset = 8;
const int buf_width = block_size_wide[MB_WIENER_PRED_BLOCK_SIZE];
const int buf_height = block_size_high[MB_WIENER_PRED_BLOCK_SIZE] + 2 * offset;
assert(buf_width == MB_WIENER_PRED_BUF_STRIDE);
CHECK_MEM_ERROR(cm, pred_buffer,
                aom_memalign(32, buf_width * buf_height * sizeof(*pred_buffer)));

// Variant 2: Intermediate stack buffer sized tightly to active column strip width (e.g., 8)
DECLARE_ALIGNED(32, int16_t,
                im_block[(MAX_SB_SIZE + MAX_FILTER_TAP) * 8]);

// Variant 3: Dimensioned to accommodate maximum supported 64x64 transform size
int16x8_t intermediate[64 * 8];
process_txfm_neon(input, intermediate, txfm_size);

// Variant 4: Tight unpadded SIMD buffers and fixed subblock constants
uint32_t frame_sse[BW * BH] = { 0 };
#define SUBBLOCK_SIZE 8
DECLARE_ALIGNED(16, static const uint8_t, av1_all_zeros[SUBBLOCK_SIZE]) = { 0 };
```
