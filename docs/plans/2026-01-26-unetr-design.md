# UNETR 2D Segmentation Head Design

## Overview

Add a UNETR (UNEt TRansformers) semantic segmentation head for 2D images to the nnU-Net ViT codebase. UNETR combines a ViT encoder with a U-Net style CNN decoder using skip connections from multiple transformer layers.

## Files to Create

| File | Purpose |
|------|---------|
| `nnunetv2/model/model_unetr.py` | UNETR model implementation |
| `nnunetv2/configs/unetr.yaml` | Training configuration |

## Architecture

```
Input Image (H, W)
       │
       ▼
   [Pad to patch_size divisible]
       │
       ▼
┌──────────────────────────────────────┐
│         ViT Backbone (frozen/LoRA)   │
│                                      │
│  Layer indices: [2,5,8,11] (base)    │
│            or   [5,11,17,23] (large) │
└──────────────────────────────────────┘
       │
       ├── z1 (early)   ──► DeconvBlock ×4 ──► skip1 (H/1)
       ├── z2           ──► DeconvBlock ×3 ──► skip2 (H/2)
       ├── z3           ──► DeconvBlock ×2 ──► skip3 (H/4)
       └── z4 (deep)    ──► DeconvBlock ×1 ──► bottleneck (H/8)
                                                    │
                                                    ▼
┌──────────────────────────────────────────────────────────────┐
│                     CNN Decoder                               │
│                                                               │
│  Stage 4: Upsample(bottleneck) + Concat(skip3) → ConvBlock   │
│  Stage 3: Upsample + Concat(skip2) → ConvBlock               │
│  Stage 2: Upsample + Concat(skip1) → ConvBlock               │
│  Stage 1: ConvBlock → 1×1 Conv → logits                      │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
   [Crop to original H, W]
       │
       ▼
   Output (num_classes, H, W)
```

## Design Decisions

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| Decoder style | Original UNETR (deconv blocks) | Faithful to paper, proven for medical imaging |
| Normalization | GroupNorm | Matches codebase pattern, works with small batches |
| Activation | LeakyReLU (slope=0.01) | Original UNETR choice |
| Channels | Configurable, default `[512, 256, 128, 64, 32]` | Flexibility for different use cases |
| Variants | Multi-scale only | UNETR is fundamentally multi-scale |
| Feature levels | 4 levels from ViT | Same indices as other models |

## Module Implementations

### DeconvBlock

Upsample 2x with refinement:

```python
class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, negative_slope=0.01):
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels,
                                          kernel_size=2, stride=2)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x):
        x = self.deconv(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
```

### UNETREncoder

Prepares skip connections from ViT features:

- Reshape each feature from (B, N, C) to (B, C, h, w)
- Apply N deconv blocks to reach target resolution:
  - z1: 4 deconv blocks → full resolution
  - z2: 3 deconv blocks → 1/2 resolution
  - z3: 2 deconv blocks → 1/4 resolution
  - z4: 1 deconv block → 1/8 resolution (bottleneck)

### UNETRDecoder

U-Net style merge and upsample:

- Each stage:
  1. Upsample previous features (bilinear 2x)
  2. Concat with skip connection
  3. ConvBlock (Conv + GroupNorm + LeakyReLU) ×2
- Final: 1×1 conv to num_classes

### UNETRSegmentationModel

Main model class with:
- Backbone creation via `create_backbone()`
- Feature extraction via `get_vit_features()`
- Input padding / output cropping
- LoRA integration
- `print_trainable_parameters()` method

## API

### Factory Function

```python
def create_segmentation_model(
    backbone_name: str = "dinov3",
    backbone_size: str = "large",
    num_classes: int = 1,
    checkpoint_path: Optional[str] = None,
    freeze_backbone: bool = False,
    use_lora: bool = True,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[List[str]] = None,
    decoder_channels: List[int] = [512, 256, 128, 64, 32],
    negative_slope: float = 0.01,
) -> nn.Module:
```

### Configuration (unetr.yaml)

```yaml
model:
  model_path: "nnunetv2.model.model_unetr"
  backbone: "dinov3"
  backbone_size: "large"
  decoder:
    decoder_channels: [512, 256, 128, 64, 32]
    negative_slope: 0.01

lora:
  enabled: true
  r: 8
  lora_alpha: 16.0
  target_modules: ["qkv"]
  lora_dropout: 0.05

train:
  initial_lr: 0.001
  weight_decay: 0.0001
  num_epochs: 500
  lr_scheduler: "cosine"
```

## Usage

```bash
python nnUNetv2_train_vit.py DATASET_ID 2d FOLD --config nnunetv2/configs/unetr.yaml
```

## Reused Components

From `backbone.py`:
- `create_backbone()` - ViT backbone creation
- `get_vit_features()` - Multi-level feature extraction
- `LAYER_INDICES` pattern - Feature level selection

From other models:
- LoRA integration pattern
- Input padding / output cropping pattern
- Factory function signature

## References

- Hatamizadeh, A., et al. "UNETR: Transformers for 3D Medical Image Segmentation." WACV 2022.
