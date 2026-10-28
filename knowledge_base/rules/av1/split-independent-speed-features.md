---
name: libaom/av1/split-independent-speed-features
description: When modifying encoder speed features or fixing bugs, split independent feature adjustments and feature re-enablements into separate, focused CLs for isolated benchmarking
rubric: change_scope
severity: should_fix
---

**Rule.** Split independent speed feature tunings, heuristic adjustments, and
bug fixes into separate, focused CLs rather than bundling them together. When
fixing an underlying bug that previously necessitated disabling a speed feature
or optimization, keep the fix self-contained and re-enable or tune the speed
feature in a dedicated follow-up CL with isolated BD-rate and speedup benchmark
results.

**Why.** Bundling multiple unrelated speed feature changes or combining bug
fixes with feature re-enablement obscures performance and quality trade-offs,
complicates code review, introduces behavioral risk, and hinders git bisection
during regression triage.

## Examples

Bad:

```cpp
// BAD: Bundling multiple unrelated speed feature changes in one CL:
sf->gm_sf.num_refinement_steps = 0;
sf->inter_sf.skip_repeated_ref_mv = 1;

// BAD: Re-enabling a disabled speed feature within an underlying bug fix CL:
// (In a CL fixing buffer allocation, prematurely removing a disablement guard):
-  if (cpi->svc.number_spatial_layers > 1) sf->rt_sf.use_rtc_tf = 0;
```

Good:

```cpp
// GOOD: Split unrelated speed feature tunings into dedicated, benchmarked CLs:
sf->gm_sf.num_refinement_steps = 0;

// GOOD: Keep bug fixes isolated; re-enable speed features in a follow-up CL:
// (In the bug fix CL, retain the disablement guard):
if (cpi->svc.number_spatial_layers > 1) sf->rt_sf.use_rtc_tf = 0;
// (In a separate follow-up CL, remove the guard and attach BD-rate benchmarks).
```
