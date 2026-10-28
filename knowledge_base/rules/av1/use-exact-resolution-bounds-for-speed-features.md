---
name: libaom/av1/use-exact-resolution-bounds-for-speed-features
description: When gating encoder features by resolution using AOMMIN(width, height), specify exact short-edge lower and upper bounds (e.g., >= 720 && <= 1080) instead of unbounded, loose, or higher-tier thresholds
rubric: correctness
severity: should_fix
---

**Rule.** When defining resolution-dependent encoder speed features, presets, or
mode conditions using AOMMIN(width, height) (such as AOMMIN(cfg->g_w, cfg->g_h)
or AOMMIN(cm->width, cm->height)), specify explicit, exact short-edge lower and
upper bounds matching tested resolution tiers (such as >= 720 && <= 1080 or >=
608 && <= 1080). Do not use unbounded lower limits (such as >= 720 or
is_720p_or_larger) that inadvertently enable features on higher unverified
resolutions (such as 4K), loose/exclusive intervals (such as > 480 && < 1080)
that miss target boundaries, or higher-tier/long-edge dimensions (such as < 2160
instead of <= 1080 for 1080p content).

**Why.** Encoder speed-versus-quality heuristics and presets are benchmarked on
specific resolution tiers. Because AOMMIN(width, height) resolves to the
short-edge dimension for both landscape (e.g., 1920x1080) and portrait (e.g.,
1080x1920) orientations, resolution bounds must test against the target tier's
short-edge threshold (such as <= 1080 for 1080p). Using higher-tier or long-edge
values (such as < 2160) mistakenly enables features on intermediate classes like
1440p (2560x1440, min dim 1440). Furthermore, omitting upper bounds (>= 720)
allows heuristics to bleed into unverified 4K/UHD resolutions causing
rate-distortion loss, while loose intervals (> 480 && < 1080) miss exact
boundary cases or trigger on unverified resolutions.

## Examples

Bad:

```cpp
// Bad: Unbounded lower check bleeds into 4K/2160p; loose interval misidentifies bounds.
const bool is_720p_or_larger = AOMMIN(cm->width, cm->height) >= 720;
if (!is_vertical_video && is_720p_or_larger) {
  sf->gm_sf.gm_erroradv_tr_level = 1;
}

// Bad: AOMMIN gives short edge (1080 for 1080p); '< 2160' mistakenly includes 1440p.
const int is_low_complexity_decode_mode_supported =
    (cfg->g_usage == AOM_USAGE_GOOD_QUALITY) &&
    (oxcf->speed >= 1 && oxcf->speed <= 3) &&
    (AOMMIN(cfg->g_w, cfg->g_h) >= 608 && AOMMIN(cfg->g_w, cfg->g_h) < 2160);
```

Good:

```cpp
// Good: Explicitly bound both lower and upper dimensions to tested ranges.
const bool is_between_720p_and_1080p = AOMMIN(cm->width, cm->height) >= 720 &&
                                       AOMMIN(cm->width, cm->height) <= 1080;
if (!is_vertical_video && is_between_720p_and_1080p) {
  sf->gm_sf.gm_erroradv_tr_level = 1;
}

// Good: Upper-bounding with short-edge '<= 1080' properly restricts up to 1080p.
const int is_low_complexity_decode_mode_supported =
    (cfg->g_usage == AOM_USAGE_GOOD_QUALITY) &&
    (oxcf->speed >= 1 && oxcf->speed <= 3) &&
    (AOMMIN(cfg->g_w, cfg->g_h) >= 608 && AOMMIN(cfg->g_w, cfg->g_h) <= 1080);
```
