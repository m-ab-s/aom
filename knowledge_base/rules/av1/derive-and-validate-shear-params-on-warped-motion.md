---
name: libaom/av1/derive-and-validate-shear-params-on-warped-motion
description: When estimating, constructing, or transforming affine warped motion models (WarpedMotionParams), calculate base matrix parameters before transformations and call av1_get_shear_params() to validate
rubric: correctness
severity: should_fix
---

**Rule.** When computing or updating affine warped motion models
(WarpedMotionParams), ensure base transformation parameters (wmmat) are
estimated or initialized (e.g., via find_affine_int() from sample points or
copied from global motion) before applying additional transformations such as
rotation (av1_warp_rotation()). After constructing or modifying matrix
parameters, call av1_get_shear_params() to derive shear parameters and verify
model validity, handling invalid models by falling back or clearing active
warp/rotation flags.

**Why.** Transformation functions modify wmmat in place; applying them before
model estimation operates on stale or uninitialized matrix values and causes
transformations to be overwritten by find_affine_int(). Furthermore, warped
motion compensation requires valid shear parameters derived from the final
transformation matrix, and omitting av1_get_shear_params() leaves uninitialized
parameters or executes invalid transforms exceeding hardware precision limits.

## Examples

Bad:

```cpp
// Bad: applying rotation before base affine model estimation
#if CONFIG_EXT_ROTATION
  av1_warp_rotation(mbmi, mbmi->rotation, center_x, center_y);
#endif
  if (find_affine_int(np, pts1, pts2, bsize, mvy, mvx, wm_params, mi_row, mi_col))
    return 1;

// Bad: modifying matrix parameters without re-deriving shear parameters
memcpy(&mi->wm_params, &xd->global_motion[mi->ref_frame[ref]], sizeof(WarpedMotionParams));
av1_warp_rotation(mi, mi->rotation, center_x, center_y);
```

Good:

```cpp
// Good: estimate base affine model first, then apply rotation
  if (find_affine_int(np, pts1, pts2, bsize, mvy, mvx, wm_params, mi_row, mi_col))
    return 1;
#if CONFIG_EXT_ROTATION
  av1_warp_rotation(mbmi, mbmi->rotation, center_x, center_y);
#endif

// Good: re-derive shear parameters and validate model
memcpy(&mi->wm_params, &xd->global_motion[mi->ref_frame[ref]], sizeof(WarpedMotionParams));
av1_warp_rotation(mi, mi->rotation, center_x, center_y);
if (!av1_get_shear_params(&mi->wm_params)) {
  mi->rot_flag = 0;
}
```
