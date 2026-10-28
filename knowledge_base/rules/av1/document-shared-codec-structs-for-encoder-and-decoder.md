---
name: libaom/av1/document-shared-codec-structs-for-encoder-and-decoder
description: When documenting fields or reference-counted resources in shared codec structs, cover both encoder and decoder contexts, list all referencing variables, or document pipeline restrictions.
rubric: readability
severity: should_fix
---

**Rule.** In shared and common codec headers (such as av1/common/ or aom/),
ensure doc comments for struct fields accurately reflect both encoder and
decoder contexts (e.g., 'encoded or decoded') when shared across pipelines. When
documenting reference-counted resources (such as RefCntBuffer or BufferPool),
ensure any enumeration of reference-holding pointers or variables exhaustively
covers all referencing fields across both encoder and decoder states (e.g.,
cm->ref_frame_map, cm->cur_frame, cm->scaled_ref_buf, pbi->output_frames). If a
field in a shared structure is restricted to one pipeline (such as encoder-only
or decoder-only) or exists for a non-obvious workaround (e.g.,
display_order_hint), explicitly document its scope restriction, purpose, and
rationale.

**Why.** Shared structures in common headers are accessed across both encoder
and decoder pipelines. Describing a shared field from only one pipeline's
perspective or omitting decoder/encoder reference-holding variables in
reference-counted resource docs obscures resource ownership, making
reference-count leaks, premature releases, and buffer reuse bugs difficult to
audit. Conversely, omitting scope restrictions or rationale for single-pipeline
fields in shared structs leads to confusion about where and why the fields are
maintained.

## Examples

Bad:

```cpp
// In shared header (e.g., av1/common/onyxc_int.h or av1/common/buffer_pool.h):

// Variant 1: Shared field described from only one pipeline's perspective
// Index to the 'new' frame (i.e. the frame currently being encoded) in the
// buffer pool 'cm->buffer_pool'.
int new_fb_idx;

// Variant 2: Reference-holding list misses decoder-specific referencing fields
// For a RefCntBuffer, the following are reference-holding variables:
// - cm->ref_frame_map[]
// - cm->cur_frame
// - cm->scaled_ref_buf[] (encoder only)
int ref_count;

// Variant 3: Scope-restricted field added without scope or rationale comment
unsigned int display_order_hint;
unsigned int ref_display_order_hint[INTER_REFS_PER_FRAME];
```

Good:

```cpp
// In shared header (e.g., av1/common/onyxc_int.h or av1/common/buffer_pool.h):

// Variant 1: Explicitly documents both encoding and decoding contexts
// Index to the 'new' frame (i.e. the frame currently being encoded or
// decoded) in the buffer pool 'cm->buffer_pool'.
int new_fb_idx;

// Variant 2: Exhaustively enumerates both encoder and decoder ref holders
// For a RefCntBuffer, the following are reference-holding variables:
// - cm->ref_frame_map[]
// - cm->cur_frame
// - cm->scaled_ref_buf[] (encoder only)
// - pbi->output_frames[] / pbi->output_frame_index[] (decoder only)
int ref_count;

// Variant 3: Explicitly documents encoder-only scope and rationale
// These variables are used only in encoder to compare absolute display
// order hints and compute relative distance when old frames are referenced.
unsigned int display_order_hint;
unsigned int ref_display_order_hint[INTER_REFS_PER_FRAME];
```
