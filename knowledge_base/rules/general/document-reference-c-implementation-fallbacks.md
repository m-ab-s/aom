---
name: libaom/document-reference-c-implementation-fallbacks
description: When implementing partial SIMD kernels or dispatchers, guard unsupported parameters at entry, fall back to lower-tier SIMD or scalar C, return immediately, and add attributed TODOs
rubric: correctness
severity: blocking
---

**Rule.** When relaxing caller dispatch conditions or implementing SIMD routines
and dispatchers with partial parameter or dimension support (such as minimum
vector step sizes like 4, 8, or 16 pixels, filter tap counts, block dimensions,
or window lengths):
1. Preserve caller guard checks and do not replace conditional dispatch with unconditional SIMD calls when only a subset of configurations is optimized.
2. Inside SIMD kernels, guard unsupported configurations at function entry before allocating intermediate buffers or performing preprocessing calculations, routing execution safely rather than asserting on expanded RTCD inputs.
3. Fall back to available lower-tier SIMD implementations (e.g., SSSE3, SSE4.1, or SSE2 from AVX2/AVX-512) before falling back to the scalar C reference implementation (*_c).
4. When delegating to scalar C or lower-tier fallback functions, immediately return or use an else branch to prevent execution from falling through into vector processing code.
5. Always document fallback constraints with explanatory comments and include attributed TODO comments with tracking issue numbers (e.g., // TODO(aomedia:XXXX): ...) for deferred or missing SIMD optimizations.

**Why.** SIMD intrinsic kernels assume fixed vector widths, specific tap counts,
or aligned block sizes; invoking them on sub-vector dimensions or unsupported
parameter configurations causes buffer overruns, undefined behavior, or
assertion failures. Omitting an immediate return after fallback delegation
causes dangerous fallthrough into vector code, leading to duplicate work and
memory corruption. In higher-tier kernels (e.g., AVX2), delegating unsupported
cases to lower-tier SIMD (e.g., SSSE3) preserves vectorization performance
rather than falling off a scalar performance cliff. Dispatching at function
entry avoids wasted setup computations, and documenting deferred paths with
attributed TODOs ensures missing SIMD coverage is properly tracked.

## Examples

Bad:

```cpp
// Variant 1: Falling back directly to scalar C from AVX2 instead of lower-tier SIMD (SSSE3)
if (filter_params_x->taps == 12) {
  av1_highbd_convolve_x_sr_c(src, src_stride, dst, dst_stride, w, h,
                             filter_params_x, subpel_x_qn, conv_params, bd);
  return;
}

// Variant 2: Missing dimension guard and missing return after fallback (fallthrough bug)
if (filter_params_x->taps > 8) {
  av1_convolve_x_sr_12tap_sse2(src, src_stride, dst, dst_stride, w, h,
                               filter_params_x, subpel_x_qn, conv_params);
}
if (filtered_length < 16) {
  av1_resize_horz_dir_c(input, in_stride, intbuf, height, filtered_length,
                        width2);
  // Missing return: falls through into SSE2 vector processing!
}

// Variant 3: Wasted buffer setup before C fallback, asserting, and missing attributed TODO
void av1_compute_stats_avx2(int wiener_win, const uint8_t *dgd, ...) {
  sub_avg_block_avx2(dgd, ...);  // Wasted work if wiener_win is unsupported
  if (wiener_win == WIENER_WIN) {
    compute_stats_win7_avx2(...);
  } else {
    av1_compute_stats_c(wiener_win, dgd, ...);
  }
}

void av1_apply_temporal_filter_neon(...) {
  // Asserts instead of falling back to C for newly supported 64x64 block size
  assert(block_size == BLOCK_32X32 && "Only support 32x32 block with Neon!");
}
```

Good:

```cpp
// Variant 1: Fall back to lower-tier SIMD (SSSE3) before scalar C reference
if (filter_params_x->taps == 12) {
  av1_highbd_convolve_x_sr_ssse3(src, src_stride, dst, dst_stride, w, h,
                                 filter_params_x, subpel_x_qn, conv_params, bd);
  return;
}

// Variant 2: Guard sub-vector dimensions and return immediately after fallback
if (filter_params_x->taps > 8) {
  if (w < 4) {
    av1_convolve_x_sr_c(src, src_stride, dst, dst_stride, w, h,
                        filter_params_x, subpel_x_qn, conv_params);
  } else {
    av1_convolve_x_sr_12tap_sse2(src, src_stride, dst, dst_stride, w, h,
                                 filter_params_x, subpel_x_qn, conv_params);
  }
}
if (filtered_length < 16) {
  av1_resize_horz_dir_c(input, in_stride, intbuf, height, filtered_length,
                        width2);
  return;
}

// Variant 3: Fall back at entry before work, and document with attributed TODO
void av1_compute_stats_avx2(int wiener_win, const uint8_t *dgd, ...) {
  if (wiener_win != WIENER_WIN && wiener_win != WIENER_WIN_CHROMA) {
    av1_compute_stats_c(wiener_win, dgd, ...);
    return;
  }
  sub_avg_block_avx2(dgd, ...);
  compute_stats_win7_avx2(...);
}

// TODO(aomedia:493082083): Modify temporal filter to support 64x64 block size with Neon.
void av1_apply_temporal_filter_neon(...) {
  if (block_size == BLOCK_64X64) {
    av1_apply_temporal_filter_c(frame_to_filter, mbd, block_size, ...);
    return;
  }
  assert(block_size == BLOCK_32X32 && "Only support 32x32 block with Neon!");
}
```
