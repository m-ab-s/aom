---
name: libaom/no-top-level-const-on-function-params
description: When declaring or defining functions, omit top-level const qualifiers on pass-by-value scalar parameters and pointer parameters like const int or Type *const
rubric: readability
severity: blocking
---

**Rule.** In function declarations, prototypes, and definitions, do not use
top-level const qualifiers on pass-by-value scalar parameters (such as int,
size_t, ptrdiff_t, uint8_t, tran_low_t, or enum/typedef values like width,
height, stride, log_bw) or on pointer parameters (Type *const param). Pass
scalar parameters by plain unmodified value and pointers as Type *param (or
const Type *param for pointer-to-const data).

**Why.** Top-level const on pass-by-value scalar parameters or pointer
parameters is an internal callee implementation detail that provides no
caller-facing interface guarantee or optimization benefit. It adds visual
clutter to function signatures, causes unnecessary discrepancies between header
declarations, RTCD definitions, and C implementations, and violates readability
conventions (e.g. clang-tidy readability-avoid-const-params-in-decls).

## Examples

Bad:

```cpp
void aom_smooth_h_predictor_32x8_ssse3(uint8_t *LIBAOM_RESTRICT dest,
                                       const ptrdiff_t stride,
                                       const uint8_t *LIBAOM_RESTRICT top_row,
                                       const uint8_t *LIBAOM_RESTRICT left_column);

bool av1_use_qmatrix(const struct CommonQuantParams *quant_params,
                     const struct macroblockd *const xd,
                     const int segment_id);

int aom_vector_var_avx2(const int16_t *ref, const int16_t *src,
                        const int log_bw);
```

Good:

```cpp
void aom_smooth_h_predictor_32x8_ssse3(uint8_t *LIBAOM_RESTRICT dest,
                                       ptrdiff_t stride,
                                       const uint8_t *LIBAOM_RESTRICT top_row,
                                       const uint8_t *LIBAOM_RESTRICT left_column);

bool av1_use_qmatrix(const struct CommonQuantParams *quant_params,
                     const struct macroblockd *xd,
                     int segment_id);

int aom_vector_var_avx2(const int16_t *ref, const int16_t *src,
                        int log_bw);
```
