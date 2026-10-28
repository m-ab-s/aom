---
name: libaom/aom/append-new-public-enum-values-at-end
description: When modifying public API enums in aom/*.h, append new enumerators at the end and retain legacy reserved constants as aliases for ABI compatibility
rubric: readability
severity: should_fix
---

**Rule.** In public API headers (aom/*.h), maintain ABI stability when extending
or updating enum definitions: append new enumerators at the end of the enum
(after existing values and before/after sentinels without shifting implicit
integers), and do not insert unnumbered items into the middle. When giving
concrete names to previously reserved or placeholder enumerators, retain the
legacy constant (e.g., *_RESERVED_*) as an alias assigned to the same numeric
value.

**Why.** Public headers define libaom's public C API and ABI contract. Inserting
new unnumbered enumerators into the middle shifts implicit integer values of
subsequent enumerators, and removing existing placeholder or reserved constants
breaks symbol and binary compatibility for external applications compiled
against earlier library versions.

## Examples

Bad:

```cpp
// Shifting values by inserting in the middle:
typedef enum {
  AOM_CONTENT_DEFAULT,
  AOM_CONTENT_SCREEN,
  AOM_CONTENT_ENHANCED,  // Shifts subsequent values like AOM_CONTENT_INVALID!
  AOM_CONTENT_INVALID
} aom_tune_content;

// Removing reserved placeholder breaks ABI:
AOM_CICP_MC_IPT_C2 = 15,
```

Good:

```cpp
// Appending new values at the end:
typedef enum {
  AOM_CONTENT_DEFAULT,
  AOM_CONTENT_SCREEN,
  AOM_CONTENT_INVALID,
  AOM_CONTENT_ENHANCED  // Appended at end to preserve ABI
} aom_tune_content;

// Retaining legacy reserved enumerator as alias:
AOM_CICP_MC_RESERVED_15 = 15, /**< For backward compatibility */
AOM_CICP_MC_IPT_C2 = 15,
```
