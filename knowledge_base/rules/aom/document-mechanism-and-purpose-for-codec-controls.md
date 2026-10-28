---
name: libaom/aom/document-mechanism-and-purpose-for-codec-controls
description: When declaring codec controls, config structs, or enum options in headers, document parameter types, numeric mappings, units, ranges, and restrictions, and use int /* enum_type */ in AOM_CTRL_USE_TYPE
rubric: readability
severity: should_fix
---

**Rule.** When defining public codec controls (`AOME_SET_*`, `AV1E_SET_*`,
`AOMD_GET_*`, `AV1D_GET_*`), configuration flags (`AOM_EFLAG_*`), config structs
(`aom_codec_enc_cfg_t`, `aom_image_t`), tuning enums (`AOM_TUNE_*`), or enum
types and mode options in headers (such as `aom/aomcx.h`, `aom/aomdx.h`, or
internal headers), provide comprehensive Doxygen doc comments (`/*!\brief ...
*/`, `\note`) and format type declarations as follows:

1. **Brief Summary and Parameter Types**:
   - In codec control ID comments, state the expected parameter type directly in the opening `\brief` summary sentence (e.g., `'/*!\brief Codec control to ..., unsigned int parameter.'`). When the control accepts an enum, name the specific enum type (e.g., `'aom_screen_detection_mode parameter'`).
   - Separate subsequent detailed paragraphs, rationale, or option lists from the `\brief` summary line with a blank comment line.
2. **AOM_CTRL_USE_TYPE for Enum Controls**:
   - When declaring type-checking for enum-valued controls with `AOM_CTRL_USE_TYPE`, pass `int` as the type argument rather than `unsigned int` or the enum type name directly, and append an inline comment specifying the expected enum type (e.g., `AOM_CTRL_USE_TYPE(AV1E_SET_MODE, int) /* aom_mode_t */`).
3. **Enumerator Behavior, Option Lists, and Numeric Values**:
   - In option lists for enum controls and when defining enum types (e.g., `DELTA_Q_MODE`), document each enumerator's intended behavior, algorithm variant, and how non-default options differ from standard or default behavior.
   - Document both the numeric integer value and the enum constant identifier (e.g., `'- 1: AOM_SCREEN_DETECTION_STANDARD = standard'`), especially when enum values start at non-zero numbers.
4. **Purpose, Mechanism, and Units**:
   - Describe underlying mechanisms and algorithm effects rather than tautological descriptions. For tuning enums, describe user-facing intent and internal shared logic (e.g., shared `rdmult` calculation).
   - Document concrete units and scaling interpretations (percentages, milliseconds, scaling factors), valid numerical ranges, and default values.
5. **Modulation, Bounds, and Prerequisites**:
   - For controls that dynamically modulate another setting (such as `AV1E_SET_ENABLE_ADAPTIVE_SHARPNESS`), document modulation ranges and interaction with maximum bounds (`AOME_SET_SHARPNESS`).
   - Explicitly document configuration prerequisites (e.g., requiring `g_lag_in_frames == 0`), applicable operating modes or profile restrictions (such as 'for all-intra mode only' or 'real-time only'), and concrete caller data preparation requirements (e.g., 'upshift the frame to the codec bit-depth').

**Why.** Public API and subsystem headers serve as the primary reference for
library consumers and maintainers. In public C headers, enum types are passed as
integers through the variadic `aom_codec_control` interface; specifying `int` in
`AOM_CTRL_USE_TYPE` with a trailing inline comment maintains portable C calling
conventions while clearly identifying the expected enum type for API consumers.
Doxygen treats comment text up to the first blank line as part of the brief
summary; stating parameter or enum types in `\brief` and separating details with
a blank line ensures clean summary tables and enables IDE navigation.
Documenting explicit numeric mappings alongside enum constants clarifies usage
when values start non-zero. Omitting explicit behaviors, mechanisms, units,
prerequisites, bounds interactions, operating mode restrictions, or caller
transformation steps forces developers to inspect internal source code and leads
to misconfigurations or pipeline bugs.

## Examples

Bad:

```cpp
// 1. Missing parameter type in brief sentence, details not separated by blank line:
/*!\brief Set deltaq strength. Valid range: [0, 1000] */
AV1E_SET_DELTAQ_STRENGTH = 148,
AOM_CTRL_USE_TYPE(AV1E_SET_DELTAQ_STRENGTH, int)

// 2. Generic int in brief, missing numeric values, and direct enum/unsigned int in AOM_CTRL_USE_TYPE:
/*!\brief Codec control to set the screen content detection mode, unsigned int parameter.
 * - AOM_SCREEN_DETECTION_STANDARD = standard (default)
 * - AOM_SCREEN_DETECTION_ANTIALIASING_AWARE = anti-aliased text aware */
AV1E_SET_SCREEN_CONTENT_DETECTION_MODE = 171,
AOM_CTRL_USE_TYPE(AV1E_SET_SCREEN_CONTENT_DETECTION_MODE, unsigned int)

// 3. Operating mode restriction omitted on enum values and missing prerequisite on flag:
/*!\brief Calculate PSNR for this frame */
#define AOM_EFLAG_CALCULATE_PSNR (1 << 1)

typedef enum {
  DELTA_Q_OBJECTIVE = 1,          // Modulation to improve objective quality
  DELTA_Q_PERCEPTUAL = 2,         // Modulation to improve perceptual quality
  DELTA_Q_VARIANCE_BOOST = 6,     // Variance Boost style modulation
} DELTA_Q_MODE;
```

Good:

```cpp
// 1. Explicit parameter type in brief, blank line separation, unit (percentage), range, default:
/*!\brief Codec control to set --deltaq-mode strength, unsigned int parameter.
 *
 * Value is a percentage scaling factor. Valid range: [0, 1000]. Default is 100.
 */
AV1E_SET_DELTAQ_STRENGTH = 148,
AOM_CTRL_USE_TYPE(AV1E_SET_DELTAQ_STRENGTH, int)

// 2. Specific enum in brief, numeric values mapped, and int + inline comment in AOM_CTRL_USE_TYPE:
/*!\brief Codec control to set the screen content detection mode, aom_screen_detection_mode parameter.
 *
 * - 1: AOM_SCREEN_DETECTION_STANDARD = standard (default)
 * - 2: AOM_SCREEN_DETECTION_ANTIALIASING_AWARE = anti-aliased text aware
 */
AV1E_SET_SCREEN_CONTENT_DETECTION_MODE = 171,
AOM_CTRL_USE_TYPE(AV1E_SET_SCREEN_CONTENT_DETECTION_MODE,
                  int) /* aom_screen_detection_mode */

// 3. Explicit prerequisite on flag and operating mode restriction on enum values:
/*!\brief Calculate PSNR for this frame, requires g_lag_in_frames to be 0 */
#define AOM_EFLAG_CALCULATE_PSNR (1 << 1)

typedef enum {
  DELTA_Q_OBJECTIVE = 1,          // Modulation to improve objective quality
  DELTA_Q_PERCEPTUAL = 2,         // Modulation to improve perceptual quality
  DELTA_Q_VARIANCE_BOOST = 6,     // Variance Boost modulation for all intra mode only
} DELTA_Q_MODE;
```
