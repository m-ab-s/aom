---
name: libaom/av1/use-ppi-number-spatial-layers-for-sequence-config
description: When querying or validating layer counts for sequence configuration, superblock sizing, or codec controls, use ppi layer counts rather than uninitialized cpi->svc fields
rubric: correctness
severity: should_fix
---

**Rule.** Query and validate spatial and temporal layer counts via the primary
compressor instance (ppi->number_spatial_layers, ppi->number_temporal_layers, or
cpi->ppi->... / ctx->ppi->...) rather than sub-mode structures
(cpi->svc.number_spatial_layers or cpi->svc.number_temporal_layers) when
configuring sequence parameters, selecting superblock sizes
(av1_select_sb_size), or bounds-checking layer ID parameters in codec control
handlers (such as ctrl_set_spatial_layer_id or ctrl_set_layer_id). Reserve
cpi->svc layer fields strictly for layer-specific SVC rate control and internal
layer context state.

**Why.** Primary compressor layer counts (ppi->number_spatial_layers and
ppi->number_temporal_layers) are initialized at encoder creation and updated by
both AOME_SET_NUMBER_SPATIAL_LAYERS and AV1E_SET_SVC_PARAMS. In contrast,
sub-mode structures (cpi->svc) are only populated when full SVC parameters are
configured and remain uninitialized or zero during early codec control calls or
non-SVC multi-layer modes (such as progressive image encoding). Referencing
cpi->svc for sequence configuration or control handlers results in incorrect
superblock sizing or premature rejection of valid layer indices.

## Examples

Bad:

```cpp
// 1. Sequence setup / superblock sizing: cpi->svc is unpopulated in non-SVC multi-layer
set_sb_size(cm->seq_params,
            av1_select_sb_size(&cpi->oxcf, cm->width, cm->height,
                               cpi->svc.number_spatial_layers));

// 2. Control handlers: cpi->svc is uninitialized before full SVC setup
static aom_codec_err_t ctrl_set_spatial_layer_id(aom_codec_alg_priv_t *ctx,
                                                 va_list args) {
  const int id = va_arg(args, int);
  if (id < 0 || id >= ctx->ppi->cpi->svc.number_spatial_layers) {
    return AOM_CODEC_INVALID_PARAM;
  }
  ctx->ppi->cpi->common.spatial_layer_id = id;
  return AOM_CODEC_OK;
}
```

Good:

```cpp
// 1. Sequence setup / superblock sizing: use canonical ppi layer counts
set_sb_size(cm->seq_params,
            av1_select_sb_size(&cpi->oxcf, cm->width, cm->height,
                               cpi->ppi->number_spatial_layers));

// 2. Control handlers: bounds-check against initialized primary compressor counts
static aom_codec_err_t ctrl_set_spatial_layer_id(aom_codec_alg_priv_t *ctx,
                                                 va_list args) {
  const int id = va_arg(args, int);
  if (id < 0 || id >= (int)ctx->ppi->number_spatial_layers) {
    return AOM_CODEC_INVALID_PARAM;
  }
  ctx->ppi->cpi->common.spatial_layer_id = id;
  return AOM_CODEC_OK;
}
```
