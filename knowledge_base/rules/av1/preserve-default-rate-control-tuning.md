---
name: libaom/av1/preserve-default-rate-control-tuning
description: When modifying rate control or Q-clamping heuristics for specific modes or content types, guard them with explicit conditions like is_screen_content_type and preserve baseline fallbacks
rubric: correctness
severity: should_fix
---

**Rule.** When adding or modifying rate control, Q clamping, or delta-QP
heuristics intended for specific modes or content types (such as
CYCLIC_REFRESH_AQ, SVC, or screen content), explicitly guard the logic with the
appropriate condition (such as cpi->is_screen_content_type). Ensure
mode-specific branches isolate their parameter calculations so baseline tuning
and fallback calculations for standard camera content and unconfigured modes
(such as NO_AQ) remain intact.

**Why.** Rate control adjustments and Q clamping thresholds tuned specifically
for screen content or adaptive quantization modes can cause severe rate
overshoot, undershoot, or quality regressions on standard video if applied
unconditionally or if baseline fallback calculations are accidentally
overwritten.

## Examples

Bad:

```cpp
// Variant 1: Screen-specific CBR adjustment applied unconditionally
if (!cpi->ppi->use_svc) {
  if (p_rc->buffer_level > p_rc->optimal_buffer_level) {
    max_delta_up = AOMMAX(4, max_delta_down);
  } else {
    max_delta_up = AOMMAX(8, max_delta_down);
  }
}

// Variant 2: AQ mode adjustment overwriting baseline screen content fallback
int max_delta_down = AOMMIN(16, AOMMAX(1, rc->q_1_frame / 8));
if (use_cyclic_refresh && cpi->is_screen_content_type) {
  if (cr->sb_index > cr->last_sb_index) {
    max_delta_down = AOMMIN(8, AOMMAX(1, rc->q_1_frame / 32));
  }
}
```

Good:

```cpp
// Variant 1: Explicitly guarded by is_screen_content_type
if (!cpi->ppi->use_svc && cpi->is_screen_content_type) {
  if (p_rc->buffer_level > p_rc->optimal_buffer_level) {
    max_delta_up = AOMMAX(4, max_delta_down);
  } else {
    max_delta_up = AOMMAX(8, max_delta_down);
  }
}

// Variant 2: Explicit branching preserving baseline fallback
if (cpi->oxcf.q_cfg.aq_mode == CYCLIC_REFRESH_AQ &&
    cpi->cyclic_refresh->apply_cyclic_refresh) {
  max_delta_down = AOMMIN(16, AOMMAX(1, rc->q_1_frame / 8));
  if (cpi->is_screen_content_type &&
      cpi->cyclic_refresh->sb_index > cpi->cyclic_refresh->last_sb_index) {
    max_delta_down = AOMMIN(8, AOMMAX(1, rc->q_1_frame / 32));
  }
} else {
  max_delta_down = (cpi->is_screen_content_type)
                       ? AOMMIN(8, AOMMAX(1, rc->q_1_frame / 16))
                       : AOMMIN(16, AOMMAX(1, rc->q_1_frame / 8));
}
```
