---
name: libaom/trailing-comment-on-endif-directives
description: 'Annotate #endif and #else on long preprocessor blocks and header guards with matching condition comments formatted without directive keywords, omitting on short blocks'
rubric: readability
severity: blocking
---

**Rule.** Annotate `#endif` and `#else` directives on long or multi-function
preprocessor blocks and header guards with trailing comments accurately matching
the opening condition, macro name, or guard token (e.g., `#endif //
CONFIG_AV1_HIGHBITDEPTH`, `#else // !CONFIG_CTX_ADAPT_LOG_WEIGHT`, `#endif //
AOM_AV1_COMMON_CCSO_H_`). Format closing comments cleanly as `#endif //
<CONDITION>` or `#endif // <MACRO>` without repeating preprocessor directive
keywords like `#if` (e.g., avoid `#endif // #if defined(...)`). When modifying
multi-branch blocks (`#if`/`#elif`/`#endif`), ensure `#endif` comments remain
synchronized with the opening `#if` condition rather than referencing an inner
`#elif`, and ensure `#else` comments accurately reflect the inverted condition.
Maintain consistent directive styles within each file (prefer `#ifdef MACRO` or
`#ifndef MACRO` for standalone identifiers over `#if defined(...)`). Conversely,
omit trailing comments on short, locally visible `#if` blocks or when
expressions would wrap across multiple comment lines.

**Why.** In preprocessor blocks spanning many lines, multiple functions, or
header guards, trailing comments on `#else` and `#endif` clarify directive
boundaries and compilation scopes without requiring readers to scroll back.
Accurately formatted comments without repeated `#if` tokens comply with project
conventions and cpplint rules. Comments referencing inner `#elif` branches or
copy-pasted guard names mislead maintainers about active build configurations,
while commenting short, locally visible blocks adds unnecessary visual clutter.

## Examples

Bad:

```cpp
// Variant 1: Bare #else / #endif on multi-line functions or mismatched header guard
#ifndef AOM_AV1_COMMON_CCSO_H_
#define AOM_AV1_COMMON_CCSO_H_
#if CONFIG_AV1_HIGHBITDEPTH
static void highbd_pred(...) { /* multi-line implementation */ }
#endif
#endif  // AOM_AV1_COMMON_CDEF_H_

// Variant 2: Repeating directive token #if or referencing #elif instead of opening #if
#if defined(CONFIG_AV1_ENCODER)
void encode_frame(void);
#endif  // #if defined(CONFIG_AV1_ENCODER)

#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)
void process_neon(...);
#elif defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
void process_neon(...);
#endif  // defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)

// Variant 3: Unnecessary multi-line comment on short, locally visible block
#if defined(AOM_ADDRESS_SANITIZER) && !defined(NDEBUG)
const size_t kMinStackSize = 1024 * 1024;
#else
const size_t kMinStackSize = 256 * 1024;
#endif  // defined(AOM_ADDRESS_SANITIZER) && !defined(NDEBUG)
```

Good:

```cpp
// Variant 1: Annotated #else / #endif matching opening condition and header guard
#ifndef AOM_AV1_COMMON_CCSO_H_
#define AOM_AV1_COMMON_CCSO_H_
#if CONFIG_AV1_HIGHBITDEPTH
static void highbd_pred(...) { /* multi-line implementation */ }
#endif  // CONFIG_AV1_HIGHBITDEPTH
#endif  // AOM_AV1_COMMON_CCSO_H_

// Variant 2: Clean comment without directive token, synchronized with opening #if
#ifdef CONFIG_AV1_ENCODER
void encode_frame(void);
#endif  // CONFIG_AV1_ENCODER

#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)
void process_neon(...);
#elif defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
void process_neon(...);
#endif  // defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)

// Variant 3: Omit trailing comment on short, locally visible block
#if defined(AOM_ADDRESS_SANITIZER) && !defined(NDEBUG)
const size_t kMinStackSize = 1024 * 1024;
#else
const size_t kMinStackSize = 256 * 1024;
#endif
```
