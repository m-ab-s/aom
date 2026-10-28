---
name: libaom/omit-redundant-narrowing-casts-on-wider-integers
description: When computing or accumulating expressions assigned to int64_t variables, avoid premature (int) truncation casts or redundant widening casts
rubric: correctness
severity: blocking
---

**Rule.** Do not insert narrowing integer casts (such as `(int)` or `(int32_t)`)
when computing expressions assigned to wider integer variables (such as
`int64_t`) or passed to helpers expecting wider integers; cast intermediate
floating-point calculations to `int64_t` rather than `int` to prevent premature
32-bit truncation. Additionally, do not add redundant widening casts (such as
`(int64_t)`) when accumulating, subtracting, or assigning narrower integers into
wider variables, as implicit integer promotion already converts the operands.

**Why.** Narrowing intermediate 64-bit or floating-point expressions with 32-bit
`(int)` casts prematurely truncates values exceeding `INT_MAX`, risking integer
overflow, precision loss, and visual clutter before downstream clamping or
scaling. Conversely, explicit widening casts in compound assignments or
arithmetic with wider variables add unnecessary clutter without altering
promotion semantics.

## Examples

Bad:

```cpp
// Redundant narrowing cast on intermediate 64-bit calculation
int64_t target = (int)(((int64_t)(16 + kf_boost) * rc->avg_frame_bandwidth) >> 4);
target = clamp_iframe_target_size(cpi, target);

// Premature 32-bit cast on intermediate floating-point calculation
int64_t rdmult = (int)((double)rdmult * weight / 128.0);

// Redundant widening cast when accumulating into wider integer
p_rc->bits_off_target -= (int64_t)encoded_frame_size;
```

Good:

```cpp
// Intermediate calculation kept in wider int64_t type
int64_t target = ((int64_t)(16 + kf_boost) * rc->avg_frame_bandwidth) >> 4;
target = clamp_iframe_target_size(cpi, target);

// Cast floating-point result directly to int64_t to prevent truncation
int64_t rdmult = (int64_t)((double)rdmult * weight / 128.0);

// Implicit promotion converts operand cleanly
p_rc->bits_off_target -= encoded_frame_size;
```
