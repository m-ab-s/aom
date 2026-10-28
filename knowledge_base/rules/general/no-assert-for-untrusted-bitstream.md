---
name: libaom/no-assert-for-untrusted-bitstream
description: When validating untrusted bitstream syntax, sequence headers, tile parameters, or operating points, use runtime error returns or aom_internal_error rather than assert()
rubric: safety
severity: blocking
---

**Rule.** Do not use assert(), aom_assert(), or debug-only assertions to
validate untrusted bitstream syntax, sequence headers, operating points, tile
parameters, or external stream data. Reject invalid or unsupported bitstream
values through standard runtime control flow, calling aom_internal_error() with
appropriate error codes (such as AOM_CODEC_CORRUPT_FRAME or
AOM_CODEC_UNSUP_BITSTREAM) or returning an error status code explicitly.

**Why.** Bitstreams are external, untrusted inputs. Assertions verify internal
programmer invariants and are stripped in release builds when NDEBUG is defined,
leaving malformed bitstreams unchecked and causing memory corruption or
undefined behavior in production. In debug or fuzzing builds, assertion failures
trigger spurious crashes and denial-of-service failures instead of clean codec
error propagation.

## Examples

Bad:

```cpp
// Asserting on untrusted bitstream headers or operating points:
assert(i == 0 || use_level_7_above);
aom_assert(tile_info->tile_columns <= MAX_TILE_COLS);
```

Good:

```cpp
// Validating untrusted bitstream data with runtime error handling:
if (i != 0 && !use_level_7_above) {
  aom_internal_error(cm->error, AOM_CODEC_CORRUPT_FRAME,
                     "Invalid operating point level configuration");
}
if (tile_info->tile_columns > MAX_TILE_COLS) {
  aom_internal_error(cm->error, AOM_CODEC_UNSUP_BITSTREAM,
                     "Unsupported tile columns count");
}
```
