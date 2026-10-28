---
name: libaom/clamp-floating-point-conversions-to-integer-limits
description: When casting floating-point expressions or 64-bit integers to narrower integer types, clamp values to target bounds using AOMMIN or saturate_cast_double_to_int to avoid overflow
rubric: correctness
severity: blocking
---

**Rule.** When casting floating-point calculations (such as values from ceil(),
floor(), round(), or rate-control formulas) or wider integer variables (such as
int64_t rate-distortion multipliers, frame targets, bit budgets, or
accumulators) to narrower integer types (like int or unsigned int), clamp values
to target integer limits using AOMMIN(..., INT_MAX) or
saturate_cast_double_to_int() before casting. Verify that scaled values or
multi-step accumulations cannot exceed INT_MAX (or [INT_MIN, INT_MAX]) and wrap
or trigger undefined behavior.

**Why.** In C and C++, converting a floating-point value that exceeds the target
integer type's representable range (greater than INT_MAX or less than INT_MIN)
results in undefined behavior. Similarly, casting out-of-range int64_t values
directly to signed 32-bit int causes signed integer overflow and undefined
behavior, leading to negative rate-distortion multipliers, corrupted bit
budgets, or runtime errors. Clamping or saturating prior to casting guarantees
well-defined, range-safe conversions.

## Examples

Bad:

```cpp
// Bad: Casting unbounded floating-point calculation directly to int
rc->max_consec_drop = (int)ceil(cpi->oxcf.rc_cfg.max_consec_drop_time *
                                cpi->framerate / 1000);

// Bad: Direct cast of unbounded wide 64-bit integer / rate calculation to int
int64_t rdmult = av1_compute_rd_mult_based_on_qindex(bit_depth, update_type,
                                                     qindex, tuning);
rdmult = (rdmult * rd_layer_depth_factor[layer_depth]) >> 7;
rdmult += ((rdmult * rd_boost_factor[boost_index]) >> 7);
return (int)rdmult;
```

Good:

```cpp
// Good: Clamping floating-point result or using saturation helper
rc->max_consec_drop = (int)AOMMIN(
    ceil(cpi->oxcf.rc_cfg.max_consec_drop_ms * cpi->framerate / 1000),
    INT_MAX);
rc->max_consec_drop = saturate_cast_double_to_int(
    ceil(cpi->oxcf.rc_cfg.max_consec_drop_ms * cpi->framerate / 1000));

// Good: Clamping 64-bit integer to INT_MAX before casting to int
int64_t rdmult = av1_compute_rd_mult_based_on_qindex(bit_depth, update_type,
                                                     qindex, tuning);
rdmult = (rdmult * rd_layer_depth_factor[layer_depth]) >> 7;
rdmult += ((rdmult * rd_boost_factor[boost_index]) >> 7);
return rdmult > 0 ? (int)AOMMIN(rdmult, INT_MAX) : 1;
```
