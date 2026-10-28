---
name: libaom/initialize-all-output-parameters-in-stubs
description: Unconditionally initialize or populate all output pointer parameters at function entry, across all branches and stubs, and via direct boolean expressions without caller pre-initialization
rubric: correctness
severity: blocking
---

**Rule.** When defining or implementing functions with output pointer
parameters, ensure all output parameters are unconditionally initialized and
populated across all execution paths without relying on callers to
pre-initialize dummy values:
1. **Function entry and early returns**: Explicitly initialize or reset all out-parameters (e.g., to 0, NULL, false, or a clean default state) at the start of the function body before processing or early returns, ensuring no output pointer is left unmodified if no data is produced.
2. **Stubs, fallbacks, and callbacks**: In stub, mock, fallback, or alternative implementations matching a function pointer interface (e.g., `av1_return_max_sub_pixel_mv`), explicitly assign valid values to every output pointer parameter (e.g., `*distortion`, `*sse1`, `*bestmv`) rather than omitting them (such as via `(void)param;`) and forcing callers to mask uninitialized reads with dummy values.
3. **Boolean and flag outputs**: Unconditionally assign evaluated boolean expressions directly to the output pointer (e.g., `*use_scaled_ref = ref_is_scaled && scaled_ref;`) rather than conditionally assigning only on true paths (e.g., `if (condition) *out_param = 1;`), which leaves the destination unassigned on false paths.

**Why.** Functions taking output pointer parameters must maintain
self-contained, deterministic contracts without assuming callers pre-initialized
variables. Omitting assignments on early exits, false branches, or in stub
implementations leaves caller stack or heap memory uninitialized, leading to
non-deterministic execution and sanitizer/Valgrind warnings. Masking these
warnings with dummy caller-side initializations hides callee contract violations
and conceals missing data across implementations. Direct unconditional
assignment guarantees deterministic state on all paths while simplifying
branching.

## Examples

Bad:

```cpp
// 1. Incomplete reset at function entry / early exit:
static void get_cx_data(struct stream_state *stream,
                        struct AvxEncoderConfig *global, int *got_data,
                        size_t *frame_size) {
  *got_data = 0;
  // *frame_size is not reset at entry; caller risks reading stale value
  if (!stream) return;
  *frame_size = 100;
}

// 2. Stub/fallback ignoring out-parameters, forcing caller dummy init:
int av1_return_max_sub_pixel_mv(..., MV *bestmv, int *distortion,
                                unsigned int *sse1, ...) {
  (void)distortion;  // leaves *distortion unassigned
  *sse1 = besterr;
  return besterr;
}

// 3. Conditional boolean assignment leaving false path uninitialized:
void check_scaled_reference(int ref_is_scaled, int scaled_ref,
                            bool *use_scaled_ref) {
  if (ref_is_scaled && scaled_ref) {
    *use_scaled_ref = 1;  // unassigned when condition is false
  }
}
```

Good:

```cpp
// 1. Explicitly reset all out-parameters at function entry:
static void get_cx_data(struct stream_state *stream,
                        struct AvxEncoderConfig *global, int *got_data,
                        size_t *frame_size) {
  *got_data = 0;
  *frame_size = 0;
  if (!stream) return;
  *frame_size = 100;
}

// 2. Stub/fallback explicitly populating all out-parameters:
int av1_return_max_sub_pixel_mv(..., MV *bestmv, int *distortion,
                                unsigned int *sse1, ...) {
  *distortion = besterr;
  *sse1 = besterr;
  return besterr;
}

// 3. Unconditionally assign evaluated boolean expression:
void check_scaled_reference(int ref_is_scaled, int scaled_ref,
                            bool *use_scaled_ref) {
  *use_scaled_ref = ref_is_scaled && scaled_ref;
}
```
