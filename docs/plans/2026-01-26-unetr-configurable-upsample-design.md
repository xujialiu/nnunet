# UNETR Configurable Upsampling Design

## Overview

Make the upsampling method in UNETR decoder configurable via YAML config, supporting both bilinear interpolation and learnable transposed convolution (deconv).

## Motivation

- Current `model_unetr.py` uses fixed bilinear upsampling in decoder
- `visionfm_unetr.py` uses learnable `ConvTranspose2d`
- Users should be able to choose between these approaches

## Configuration

Add `upsample_mode` to `unetr.yaml` under `decoder` section:

```yaml
model:
  decoder:
    decoder_channels: [512, 256, 128, 64, 32]
    negative_slope: 0.01
    upsample_mode: "deconv"  # Options: "bilinear", "deconv"
```

**Options:**
- `deconv` (default): Learnable `nn.ConvTranspose2d(kernel_size=2, stride=2)`
- `bilinear`: Fixed `nn.Upsample(mode='bilinear', align_corners=False)`

## Implementation

### UNETRDecoder Changes

Add `_make_upsample()` helper method:

```python
def _make_upsample(self, in_channels: int, out_channels: int) -> nn.Module:
    if self.upsample_mode == "deconv":
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    else:  # bilinear
        return nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
```

Update upsampling layers to use helper. Note that channel math differs:
- `bilinear`: Keeps input channels, concat = in + skip
- `deconv`: Changes channels during upsample, concat = out + skip

### Parameter Flow

1. `unetr.yaml` → `decoder.upsample_mode`
2. `nnUNetTrainer_vit.py` passes via `**decoder_kwargs` (no changes needed)
3. `create_segmentation_model()` accepts and forwards param
4. `UNETRSegmentationModel` passes to `UNETRDecoder`

## Files to Modify

| File | Change |
|------|--------|
| `nnunetv2/configs/unetr.yaml` | Add `upsample_mode: "deconv"` |
| `nnunetv2/model/model_unetr.py` | Add param to decoder, model, and factory |

## Backward Compatibility

Default is `deconv`. Existing configs without `upsample_mode` will use deconv upsampling.
