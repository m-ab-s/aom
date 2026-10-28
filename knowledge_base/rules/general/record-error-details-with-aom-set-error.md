---
name: libaom/record-error-details-with-aom-set-error
description: When handling codec errors, establish setjmp in interface wrappers and assert active error_info->setjmp upstream, or record details with aom_set_error and return explicitly in non-longjmp paths
rubric: correctness
severity: blocking
---

**Rule.** In codec interface wrapper functions and control handlers (such as
ctrl_* in av1_*x_iface.c), establish the setjmp error-handling buffer around
internal codec calls so internal routines can report errors uniformly via
aom_internal_error(), rather than branching conditionally on error->setjmp. In
internal decoder and encoder functions that rely on aom_internal_error() to
abort execution via longjmp, assert that the error jump buffer is active (e.g.,
assert(cm->error->setjmp) or assert(error_info->setjmp)) upstream at caller
entry points and common execution paths. In codec control functions or other
non-longjmp execution paths without active setjmp targets, record error details
with aom_set_error() and return the error code explicitly rather than calling
aom_internal_error() or returning bare error codes without recorded details.

**Why.** aom_internal_error() only performs a longjmp() when error_info->setjmp
is non-zero; without an active setjmp target, it records error details and
returns silently, causing unexpected fallthrough into subsequent code.
Establishing setjmp at interface boundaries ensures uniform error handling
without internal fallback branching, while asserting error_info->setjmp at
upstream function entry points guarantees missing error-handling boundaries are
caught immediately during testing rather than silently failing on corrupt
bitstreams. In execution paths not bounded by setjmp, calling aom_set_error()
and returning explicitly ensures safe control flow while preserving
human-readable error descriptions on struct aom_internal_error_info.

## Examples

Bad:

```cpp
// Bad: Conditional error reporting fallback in decoder helper
if (cm->error->setjmp) {
  aom_internal_error(cm->error, AOM_CODEC_ERROR, "Invalid param");
} else {
  return AOM_CODEC_ERROR;
}

// Bad: Relying on aom_internal_error() longjmp without asserting active setjmp upstream
aom_codec_err_t av1_set_reference_dec(AV1_COMMON *cm, int idx) {
  YV12_BUFFER_CONFIG *ref_buf = get_ref_frame(cm, idx);
  if (ref_buf == NULL) {
    aom_internal_error(cm->error, AOM_CODEC_ERROR, "No reference frame");
  }
}

// Bad: Returning bare error code without recording error detail
if (ref_buf == NULL) {
  return AOM_CODEC_ERROR;
}
```

Good:

```cpp
// Good: Establish setjmp buffer in interface wrappers (e.g. av1_dx_iface.c)
struct aom_internal_error_info *const error = frame_worker_data->pbi->common.error;
if (setjmp(error->jmp)) {
  error->setjmp = 0;
  return error->error_code;
}
error->setjmp = 1;
av1_decode_frame(pbi, data, data_sz);
error->setjmp = 0;

// Good: Assert setjmp is active when relying on aom_internal_error() longjmp
aom_codec_err_t av1_set_reference_dec(AV1_COMMON *cm, int idx) {
  assert(cm->error->setjmp);
  YV12_BUFFER_CONFIG *ref_buf = get_ref_frame(cm, idx);
  if (ref_buf == NULL) {
    aom_internal_error(cm->error, AOM_CODEC_ERROR, "No reference frame");
  }
}

// Good: In non-longjmp paths, record details with aom_set_error() and return explicitly
if (ref_buf == NULL) {
  aom_set_error(&pbi->error, AOM_CODEC_ERROR, "No reference frame");
  return pbi->error.error_code;
}
```
