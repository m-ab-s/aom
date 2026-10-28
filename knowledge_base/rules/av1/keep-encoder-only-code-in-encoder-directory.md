---
name: libaom/av1/keep-encoder-only-code-in-encoder-directory
description: Keep encoder/decoder logic, state (AV1_COMP, MACROBLOCK), definitions, and macros in av1/encoder/ or av1/decoder/, and never include pipeline headers in av1/common/
rubric: design
severity: should_fix
---

**Rule.** Keep logic, state, definitions, and headers used exclusively by the
encoder or decoder in dedicated pipeline directories and structs rather than in
shared common files:

1. **File and Header Boundaries**: Do not place encoder-only or decoder-only logic, helper functions, macros, constants, lookup tables, enums, structs, or experimental flags (such as rate control, first-pass analysis, search heuristics, RDO, or TPL) in `av1/common/`. Common files (`av1/common/*`) must never `#include` headers from `av1/encoder/` or `av1/decoder/`.
2. **Codec- and Frame-Level State**: Place encoder-only state and configuration parameters (such as motion search history, speed features, rate control) in `AV1_COMP`, `AV1EncoderConfig`, or `ThreadData`, and decoder-only state in `AV1Decoder` or `AV1DecoderConfig`, reserving `AV1_COMMON` (`AV1Common`) strictly for shared codec data.
3. **Block-Level State**: Place encoder-only search, mode evaluation, and rate-distortion state in `MACROBLOCK` (`macroblock`, `av1/encoder/block.h`), reserving `MACROBLOCKD` (`macroblockd`, `av1/common/blockd.h`) strictly for state shared with or required by the decoder.
4. **Shared Code Consolidation**: Pass encoder optimization parameters into shared workers via worker context structs (e.g., `AV1CdefWorkerData`). Consolidate logic shared by both encoder and decoder in `av1/common/` using the `av1_` prefix rather than duplicating implementations across pipelines.

**Why.** libaom supports standalone decoder-only and encoder-only builds while
enforcing strict architectural separation of concerns. Placing encoder-only or
decoder-only logic, macros, constants, lookup tables, or struct fields into
shared locations (`av1/common/`, `AV1_COMMON`, `MACROBLOCKD`) introduces
unwanted dependencies, pollutes shared interfaces, increases memory footprint
for decoder instances, and bloats decoder-only binaries. Including encoder or
decoder headers from `av1/common/` breaks modularity and build isolation.
Conversely, duplicating shared helpers across `av1/encoder/` and `av1/decoder/`
creates maintenance overhead and risks logic divergence.

## Examples

Bad:

```cpp
// Anti-pattern 1: Including encoder or decoder headers in common files
// In av1/common/av1_common_int.h
#include "av1/decoder/decoder.h"
#include "av1/encoder/encoder.h"

// Anti-pattern 2: Placing encoder-only macros, enums, tables, or state in common
#define MAX_FIRSTPASS_ANALYSIS_FRAMES 150
typedef enum { ENCODER_SEARCH_FAST, ENCODER_SEARCH_ACCURATE } ENCODER_SEARCH_MODE;
static const uint16_t av1_derived_intra_tx_used_flag[INTRA_MODES] = { 0x0209, 0x0403 };

typedef struct AV1Common {
  FRAME_TYPE last_frame_type;  // Bad: encoder-only motion search state
} AV1Common;

// Anti-pattern 3: Placing encoder-only block search state in MACROBLOCKD
// In av1/common/blockd.h
typedef struct macroblockd {
  FRAME_CONTEXT *backup_tile_ctx;  // Bad: encoder-only RD search state
} MACROBLOCKD;
```

Good:

```cpp
// Pattern 1: Common headers include only common dependencies
// In av1/common/av1_common_int.h
#include "aom/aom_codec.h"
#include "av1/common/enums.h"

// Pattern 2: Encoder-only macros, enums, tables, and state in av1/encoder/
// In av1/encoder/encoder.h or av1/encoder/firstpass.h
#define MAX_FIRSTPASS_ANALYSIS_FRAMES 150
typedef enum { ENCODER_SEARCH_FAST, ENCODER_SEARCH_ACCURATE } ENCODER_SEARCH_MODE;
static const uint16_t av1_derived_intra_tx_used_flag[INTRA_MODES] = { 0x0209, 0x0403 };

typedef struct AV1_COMP {
  FRAME_TYPE last_frame_type;  // Good: encoder search context
} AV1_COMP;

// Pattern 3: Encoder-only block search state in MACROBLOCK
// In av1/encoder/block.h
typedef struct macroblock {
  FRAME_CONTEXT *backup_tile_ctx;  // Good: encoder-only backup state
} MACROBLOCK;
```
