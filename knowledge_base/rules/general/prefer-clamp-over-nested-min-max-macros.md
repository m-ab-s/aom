---
name: libaom/prefer-clamp-over-nested-min-max-macros
description: When constraining values to a range, use clamp(), clamp64(), fclamp(), or clip_pixel() instead of nested AOMMIN/AOMMAX macros, ensuring low <= high for dynamic bounds
rubric: correctness
severity: blocking
---

**Rule.** When constraining integer, 64-bit, floating-point, or pixel values
between lower and upper limits, use standard inline helper functions clamp()
(for int), clamp64() (for int64_t), fclamp() (for double/float), or clip_pixel()
/ clip_pixel_highbd() from aom_dsp/aom_dsp_common.h rather than nesting or
chaining AOMMIN and AOMMAX macros. Additionally, when bounds are computed
dynamically and could invert (low > high), ensure low <= high holds or sanitize
the upper/lower bounds with AOMMAX/AOMMIN prior to calling clamp().

**Why.** AOMMIN and AOMMAX macros evaluate their arguments multiple times,
risking redundant computation and accidental side effects with complex
expressions. The inline helper functions evaluate arguments exactly once,
provide type safety, and improve readability across the codebase. Furthermore,
clamp(val, low, high) is implemented assuming low <= high; if dynamic bounds
invert (low > high), calling clamp() will produce values outside the intended
upper bound.

## Examples

Bad:

```cpp
// Nested macro calls cause redundant evaluations and side effects
const int val = AOMMAX(0, AOMMIN(QINDEX_RANGE - 1, input_val));
const double fval = AOMMAX(0.0, AOMMIN(1.0, rate_factor));

// Chained macros evaluate expressions multiple times
mv_limits->row_min = AOMMAX(mv_limits->row_min, -top_margin);
mv_limits->row_min = AOMMIN(mv_limits->row_min, mv_limits->row_max);

// If dynamic bounds can invert (-top_margin > row_max), clamp returns -top_margin (> row_max)
mv_limits->row_min = clamp(mv_limits->row_min, -top_margin, mv_limits->row_max);
```

Good:

```cpp
// Use inline helpers for concise, single-evaluation bounding
const int val = clamp(input_val, 0, QINDEX_RANGE - 1);
const double fval = fclamp(rate_factor, 0.0, 1.0);

// Sanitize dynamic bounds to guarantee low <= high before clamping
bottom_margin = AOMMAX(bottom_margin, -top_margin);
mv_limits->row_min = AOMMAX(mv_limits->row_min, -top_margin);
mv_limits->row_max = AOMMIN(mv_limits->row_max, bottom_margin);
```
