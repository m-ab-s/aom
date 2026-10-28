---
name: libaom/av1/document-speed-feature-levels-and-validation-checks
description: When defining or modifying encoder speed features, heuristic flags, or search enums, document affected parameters, mode restrictions, pipeline stages, metric sources, and per-level fallback conditions
rubric: readability
severity: should_fix
---

**Rule.** When defining or modifying encoder speed features, heuristic flags,
thresholds, or search enums in SPEED_FEATURES structs and speed_features.h,
document their exact behavior in comments. Specify the concrete encoder state or
parameter being modified (e.g., rc_bits_per_mb) and the intended objective
(e.g., reducing overshoot) rather than vague summaries. Explicitly document
applicable pipeline stages (e.g., MODE_EVAL vs. WINNER_MODE_EVAL), operating
mode restrictions (such as all-intra mode or keyframes only), interactions with
base speed features, metric sources (e.g., estimated vs. measured RD costs), and
valid values. For multi-level numeric speed features, document each numeric
level with its exact evaluation conditions, which stages or calculations are
skipped (e.g., start MV vs. final MV checks), and any fallback or redo behavior.

**Why.** Vague comments like 'adjust the RC' or underspecified level summaries
obscure algorithmic differences, evaluation timing, fallback re-searches, and
pipeline stage constraints. Clear and comprehensive comments enable maintainers
to safely tune, debug, and understand speed/quality tradeoffs without
reverse-engineering heuristic implementations across multiple files.

## Examples

Bad:

```cpp
// For keyframes in rtc: adjust the RC based on content.
int rc_adjust_keyframe;

// Default Transform search case - used in evaluation of best inter candidates
TX_SEARCH_DEFAULT = 0,
// Transform search in motion mode rd
TX_SEARCH_MOTION_MODE,

// 0: Do not downsample rows
// 1: Use SAD deviation of the even rows to odd rows as the metric to
//    downsample rows
// 2: Downsample rows
int use_downsampled_sad;
```

Good:

```cpp
// For keyframes in rtc: adjust the rc_bits_per_mb, to reduce overshoot.
int rc_adjust_keyframe;

// Default transform search used in evaluation of best inter candidates
// (MODE_EVAL stage) and motion mode winner processing (WINNER_MODE_EVAL stage).
TX_SEARCH_DEFAULT = 0,
// Transform search in motion mode rd during MODE_EVAL stage.
TX_SEARCH_MOTION_MODE,

// 0: Disabled (do not downsample rows)
// 1: Skip SAD calculation of odd rows if SAD deviation for start MV is small.
//    Redo motion search with sf off when SAD deviation is high for final MV.
// 2: Skip SAD calculation of odd rows without start MV testing.
int use_downsampled_sad;
```
