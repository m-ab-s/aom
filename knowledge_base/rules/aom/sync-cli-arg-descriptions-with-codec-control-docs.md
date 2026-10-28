---
name: libaom/aom/sync-cli-arg-descriptions-with-codec-control-docs
description: When adding or updating codec controls, configuration settings, or CLI options, keep documentation, valid values, and terminology consistent across public headers and av1/arg_defs.c
rubric: readability
severity: should_fix
---

**Rule.** When adding, updating, or clarifying codec controls, operational
modes, configuration struct fields (such as in aom_codec_enc_cfg_t), or CLI
options, ensure public API header doc comments (e.g., in aom/aomcx.h,
aom/aomdx.h, or aom/aom_encoder.h) and corresponding command-line argument
definitions (ARG_DEF in av1/arg_defs.c) remain consistent. Document all valid
values, supported modes, behavior details, and precise terminology (such as
input source bit-depth vs. codec bit-depth) across public headers, CLI help
text, and internal struct comments.

**Why.** Public API headers and CLI help strings serve as the primary references
for library callers and CLI users. Inconsistent terminology or omitting newly
supported modes, behaviors, or semantic distinctions across headers, ARG_DEF
help strings, and internal comments creates confusion and leaves features
undiscoverable.

## Examples

Bad:

```cpp
// In aom/aomcx.h (missing newly added deltaq modes 5 and 6):
 * - 4 = use modulation for user rating based perceptual quality optimization
 */
AV1E_SET_DELTAQ_MODE = 107,

// In av1/arg_defs.c (missing mode descriptions or imprecise terminology):
.deltaq_mode =
    ARG_DEF(NULL, "deltaq-mode", 1,
            "Delta qindex mode (0: off, 1: deltaq objective (default), "
            "2: deltaq placeholder, 3: key frame visual quality, 4: user "
            "rating based visual quality optimization); "
            "requires --enable-tpl-model=1"),
.inbitdeptharg = ARG_DEF(NULL, "input-bit-depth", 1, "Bit depth of input");
```

Good:

```cpp
// In aom/aomcx.h (documents all valid modes):
 * - 4 = use modulation for user rating based perceptual quality optimization
 * - 5 = use modulation for HDR video
 * - 6 = use modulation for all intra using Variance Boost
 */
AV1E_SET_DELTAQ_MODE = 107,

// In av1/arg_defs.c (accurate mode descriptions and precise terminology):
.deltaq_mode =
    ARG_DEF(NULL, "deltaq-mode", 1,
            "Delta qindex mode (0: off, 1: deltaq objective (default), "
            "2: deltaq placeholder, 3: key frame visual quality, 4: user "
            "rating based visual quality optimization, 5: HDR video, 6: "
            "Variance Boost); --deltaq-mode=1 requires --enable-tpl-model=1"),
.inbitdeptharg =
    ARG_DEF(NULL, "input-bit-depth", 1, "Bit depth of input source");
```
