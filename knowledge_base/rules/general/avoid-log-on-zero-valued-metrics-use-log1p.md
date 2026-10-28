---
name: libaom/avoid-log-on-zero-valued-metrics-use-log1p
description: When computing logarithms of non-negative metrics or error statistics that can be zero, use log1p() or an offset instead of bare log() or ternary checks
rubric: correctness
severity: blocking
---

**Rule.** When calculating the logarithm of non-negative metrics, error values,
distortion, variance, or statistics (such as coded_error, intra_error,
log_energy, or frame_avg_wavelet_energy) that can be zero, use log1p(x) or
log1pf(x) (or add an explicit offset such as log(x + 1.0)) rather than calling
bare log() or using ternary/conditional checks against zero. Never pass
unvalidated zeroable quantities directly into log().

**Why.** Evaluating log(0.0) produces -INFINITY or mathematical domain errors,
which corrupt floating-point accumulators and distort downstream rate control
and distortion statistics. Furthermore, ternary zero checks (e.g., `x ? log(x) :
0.0`) produce discontinuities and negative logarithms for fractional values in
(0, 1). Using log1p(x) computes ln(1 + x) accurately, smoothly mapping 0.0 to
0.0 without branching.

## Examples

Bad:

```cpp
// Direct log() produces -INFINITY when input is 0.0; ternary check yields negative logs for (0, 1)
section->log_coded_error += log(frame->coded_error);
section->log_intra_error += frame->intra_error ? log(frame->intra_error) : 0.0;
```

Good:

```cpp
// log1p(x) computes ln(1 + x), safely mapping 0.0 to 0.0 without branching
section->log_coded_error += log1p(frame->coded_error);
section->log_intra_error += log1p(frame->intra_error);
```
