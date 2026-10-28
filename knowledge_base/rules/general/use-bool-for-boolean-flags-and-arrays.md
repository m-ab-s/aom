---
name: libaom/use-bool-for-boolean-flags-and-arrays
description: When declaring, assigning, or evaluating boolean flags, struct members, arrays, pointers, or predicates, use bool with <stdbool.h>, true/false, direct checks, and inline call-site comments
rubric: readability
severity: blocking
---

**Rule.** In C and dual C/C++ code, use `bool` (explicitly including
`<stdbool.h>` in `.c` and `.h` files rather than relying on transitive includes)
instead of integer types (`int`, `unsigned int`, `uint32_t`, `uint8_t`) for
predicate functions, local variables, flags, struct members, status arrays,
binary status return types, and out-parameters (`bool *` instead of `int *`). Do
not split boolean field types across `#ifdef __cplusplus` guards.

Follow these conventions for boolean usage:
1. **Literals & Conditions**: In assignments, initializations, and returns, use boolean literals `true` and `false` rather than `1` and `0`. Evaluate boolean expressions directly (`if (flag)`, `if (!flag)`) rather than comparing against literals or zero (`== true`, `== false`, `== 1`, `== 0`, `!= 0`).
2. **Explicit Conversions**: Explicitly compare against zero (`!= 0` or `> 0`) when converting bitwise mask checks (e.g., `(flags & MASK) != 0`), comparison function results (e.g., `strcmp(...) != 0`), or multi-valued integer settings/enums to boolean values or function arguments.
3. **Call-Site Annotations & Comments**: Annotate non-obvious boolean literal arguments at call sites with inline parameter-name comments (e.g., `/*param=*/false`). Document ambiguous boolean return value semantics (e.g., `// Returns true on success, false on failure.`) above function declarations in headers, and use `true`/`false` rather than `1`/`0` in code comments.

**Why.** Explicitly including `<stdbool.h>` avoids brittle transitive header
dependencies that break during refactoring. Using standard `bool` with
`true`/`false` enforces strong type safety, clarifies binary intent, avoids
`#ifdef __cplusplus` preprocessor branching, reduces struct memory footprints (1
byte vs. 4 bytes), and prevents pointer type mismatches on output parameters.
Direct conditional checks eliminate redundant comparisons. Explicit conversions
(`!= 0`) clarify intentional truth-value conversions from bitmasks or
multi-level features without implicit truncation. Inline parameter comments
(`/*param=*/false`) make call sites self-documenting without requiring function
signature lookups.

## Examples

Bad:

```cpp
// Bad: missing <stdbool.h>, using int / #ifdef for flags, out-params, comparisons
struct SvcRtcRef {
#if defined(__cplusplus)
  bool is_key_frame;
#else
  int is_key_frame;
#endif
  unsigned int enable_cdef;
  int is_frame_dropped[MAX_STATIC_GF_GROUP_LENGTH];
};

int check_neighbor_blk(const MACROBLOCKD *xd, int *use_scaled_ref) {
  int using_file = strcmp(fn, "-");
  bool is_layer_obu = metadata->flags & AOM_MIF_LAYER_SPECIFIC;
  if (is_layer_obu == 1 || is_screen == false) return 1;
  return 0;
}

diffwtd_mask_neon(mask, 0, src0, src0_stride, src1, src1_stride, h, w);
av1_pick_cdef_from_qp(cm, cpi->sf.rt_sf.skip_cdef_sb, is_screen, avoid_uv);
```

Good:

```cpp
// Good: explicit <stdbool.h>, bool everywhere, true/false, inline comments, != 0
#include <stdbool.h>

struct SvcRtcRef {
  bool is_key_frame;
  bool enable_cdef;
  bool is_frame_dropped[MAX_STATIC_GF_GROUP_LENGTH];
};

// Returns true if neighbor block is valid, false otherwise.
bool check_neighbor_blk(const MACROBLOCKD *xd, bool *use_scaled_ref) {
  const bool using_file = strcmp(fn, "-") != 0;
  const bool is_layer_obu = (metadata->flags & AOM_MIF_LAYER_SPECIFIC) != 0;
  if (is_layer_obu || !is_screen) return true;
  return false;
}

diffwtd_mask_neon(mask, /*inverse=*/false, src0, src0_stride, src1, src1_stride, h, w);
av1_pick_cdef_from_qp(cm, cpi->sf.rt_sf.skip_cdef_sb != 0, is_screen, avoid_uv);
```
