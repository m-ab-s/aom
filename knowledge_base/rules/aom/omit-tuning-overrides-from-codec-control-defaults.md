---
name: libaom/aom/omit-tuning-overrides-from-codec-control-defaults
description: When adding or modifying codec controls and CLI options (ARG_DEF), document usage-mode-specific initialization defaults in headers and help text, but omit tuning-metric overrides
rubric: readability
severity: should_fix
---

**Rule.** When documenting or modifying default values for public codec controls
in API headers (such as `aom/aomcx.h` or `aom/aomdx.h`) and CLI argument
definitions (`ARG_DEF` in `av1/arg_defs.c` or `apps/`):
1. Accurately document baseline encoder initialization defaults across all supported usage modes, clearly distinguishing mode-specific values when they differ (such as `AOM_USAGE_ALL_INTRA`, `AOM_USAGE_REALTIME`, or `AOM_USAGE_GOOD_QUALITY`). When changing or adding usage-mode defaults in encoder initialization code (e.g., `av1/av1_cx_iface.c`), always update the corresponding header and CLI doc comments to match.
2. Do not document dynamic tuning-metric parameter overrides (such as SSIMULACRA 2 or `AOM_TUNE_IQ`) as default values on individual option or codec control definitions; document tuning overrides centrally under the tuning mode enum (`aom_tune_metric`) or `--tune` argument definition.

**Why.** Default values in API headers and CLI help text describe baseline
parameters configured during encoder initialization (`aom_codec_enc_init()`)
across standard usage modes (`AOM_USAGE_*`). Accurately documenting
usage-mode-specific defaults prevents caller confusion about active encoder
behavior when switching between all-intra, realtime, and good-quality modes.
Conversely, tuning metrics dynamically override configuration parameters at
runtime; documenting these overrides under individual options and codec controls
causes clutter and documentation drift, whereas centralizing overrides under the
tuning metric definition keeps option defaults clean and presets easy to find.

## Examples

Bad:

```cpp
// Variant 1: Omits usage-mode distinctions when defaults change in encoder init
// av1/av1_cx_iface.c sets tuning to AOM_TUNE_SSIM for AOM_USAGE_ALL_INTRA, but header states:
/*!\brief Codec control function to set visual tuning
 *
 * The default is AOM_TUNE_PSNR.
 */
AOME_SET_TUNING = 24,

// Variant 2: Documents tuning override as an option default in CLI and header
.qm_min = ARG_DEF(NULL, "qm-min", 1,
                  "Min quant matrix flatness (0..15), default is 5 (4 for "
                  "all intra mode, 2 for SSIMULACRA 2 tune)"),

/*!\brief Codec control to set min quant matrix flatness
 *
 * The default is 5 (or 4 for all-intra, 2 for AOM_TUNE_SSIMULACRA2).
 */
AV1E_SET_QM_MIN = 64,
```

Good:

```cpp
// Variant 1: Accurately reflects defaults across usage modes in header doc
// av1/av1_cx_iface.c sets tuning to AOM_TUNE_SSIM for AOM_USAGE_ALL_INTRA:
/*!\brief Codec control function to set visual tuning
 *
 * The default is AOM_TUNE_PSNR (or AOM_TUNE_SSIM in all-intra mode).
 */
AOME_SET_TUNING = 24,

// Variant 2: Documents usage-mode default and centralizes tuning overrides
.qm_min = ARG_DEF(
    NULL, "qm-min", 1,
    "Min quant matrix flatness (0..15), default is 5 (4 for all intra mode)"),

// In aom/aomcx.h under enum aom_tune_metric:
/*!\brief SSIMULACRA 2 tuning metric.
 * Automatically overrides the following options:
 *   * --qm-min=2
 */
AOM_TUNE_SSIMULACRA2 = 11,
```
