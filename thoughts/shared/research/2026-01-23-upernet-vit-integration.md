# Research: Adding UperNet Segmentation Head to ViT Backbone

**Date**: 2026-01-23
**Topic**: UperNet integration with ViT backbone using transformers library

## Research Question

How to add UperNet segmentation head to ViT backbone using `nnUNetv2_train_vit.py` as entry point, leveraging the transformers library's UperNet implementation.

## Summary

The transformers library (v4.57.3) provides `UperNetHead` which can be directly used with ViT backbones. The implementation should follow the existing pattern in `model_segformer_1.py`, using only the `UperNetHead` class (not the full `UperNetForSemanticSegmentation`) combined with the project's existing backbone system.

## Environment

- **Conda environment**: dinov3
- **transformers version**: 4.57.3
- **UperNet classes available**: `UperNetConfig`, `UperNetForSemanticSegmentation`, `UperNetHead`, `UperNetFCNHead`

## Architecture Flow

```
nnUNetv2_train_vit.py
  └── run_training_vit.py::run_training_entry()
        └── nnUNetTrainer_vit (loads YAML config via --config)
              └── initialize()
                    └── Dynamically imports model from config.model.model_path
                          └── Calls create_segmentation_model() factory function
```

## Key Files

| File | Purpose |
|------|---------|
| `nnunetv2/model/model_segformer_1.py:1-312` | **Best template** - SegFormer decoder with ViT backbone |
| `nnunetv2/model/backbone.py:1-197` | `create_backbone()`, `get_vit_features()`, `ProgressiveUpsampler` |
| `nnunetv2/model/primus.py:1-670` | Alternative example with multi-scale features |
| `nnunetv2/training/nnUNetTrainer/vit_config.py:1-54` | YAML config dataclasses |
| `nnunetv2/training/nnUNetTrainer/nnUNetTrainer_vit.py:1-161` | ViT trainer implementation |
| `nnunetv2/configs/segformer.yaml` | Example YAML config |

## Transformers UperNetHead API

### Configuration

```python
from transformers import UperNetConfig
from transformers.models.upernet.modeling_upernet import UperNetHead

config = UperNetConfig(
    num_labels=4,              # Number of segmentation classes
    hidden_size=512,           # Decoder hidden dimension
    pool_scales=[1, 2, 3, 6],  # Pyramid Pooling Module scales
    use_auxiliary_head=False,  # Whether to use auxiliary FCN head
)
```

### UperNetHead Initialization

```python
# in_channels: List of channel dimensions for each of 4 feature levels
# For ViT: all same value since ViT produces same-dim features at each layer
head = UperNetHead(config, in_channels=[1024, 1024, 1024, 1024])
```

### Forward Pass

```python
# Input: List of 4 feature tensors at patch resolution
# Each tensor: [B, embed_dim, H/patch_size, W/patch_size]
features = [feat1, feat2, feat3, feat4]

# Output: [B, num_labels, H/patch_size, W/patch_size]
# Note: Output is at SAME resolution as input features (not full resolution)
logits = head(features)
```

### Tested Example

```python
# ViT Large: embed_dim=1024, patch_size=16
# For 224x224 input: feature maps are 14x14
B, embed_dim, H, W = 2, 1024, 14, 14
features = [torch.randn(B, embed_dim, H, W) for _ in range(4)]

output = head(features)
# output.shape = [2, num_labels, 14, 14]
```

## Multi-Scale Feature Extraction

From existing implementations (`model_segformer_1.py:28-31`, `primus.py:329-332`):

```python
# Layer indices for extracting 4 feature levels from ViT
LAYER_INDICES = {
    12: [2, 5, 8, 11],    # ViT Base (12 transformer blocks)
    24: [5, 11, 17, 23],  # ViT Large (24 transformer blocks)
}

# Usage with backbone.py utilities
from nnunetv2.model.backbone import get_vit_features

features = get_vit_features(
    self.backbone,
    x,
    indices=self.feature_indices,  # e.g., [5, 11, 17, 23] for Large
)
# Returns: List of 4 tensors, each [B, embed_dim, H/patch, W/patch]
```

## Required Implementation Components

### 1. Model Class Structure

Based on `SegFormerSegmentationModel` pattern:

```python
class UperNetSegmentationModel(nn.Module):
    LAYER_INDICES = {
        12: [2, 5, 8, 11],
        24: [5, 11, 17, 23],
    }

    def __init__(
        self,
        backbone_name: str = "dinov3",
        backbone_size: str = "large",
        num_classes: int = 1,
        decoder_hidden_size: int = 512,
        pool_scales: List[int] = [1, 2, 3, 6],
        checkpoint_path: Optional[str] = None,
        freeze_backbone: bool = False,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
    ):
        # 1. Create backbone using create_backbone()
        # 2. Determine feature_indices based on num_layers
        # 3. Create UperNetConfig and UperNetHead
        # 4. Create ProgressiveUpsampler for resolution matching
        # 5. Handle freeze_backbone and use_lora
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Pad input to patch_size multiple
        # 2. Extract multi-scale features via get_vit_features()
        # 3. Pass features through UperNetHead
        # 4. Upsample via ProgressiveUpsampler
        # 5. Crop to original size
        pass

    def print_trainable_parameters(self, detailed: bool = False) -> None:
        # Standard parameter counting
        pass

    def enable_lora(self, ...) -> None:
        # LoRA setup using peft library
        pass

    def disable_lora(self) -> None:
        # LoRA merge and unload
        pass
```

### 2. Factory Function

Must match trainer expectations (see `nnUNetTrainer_vit.py:102-130`):

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
    # Decoder-specific kwargs (from config.model.decoder)
    hidden_size: int = 512,
    pool_scales: List[int] = [1, 2, 3, 6],
    **kwargs,
) -> nn.Module:
    return UperNetSegmentationModel(...)
```

### 3. YAML Configuration

Create `nnunetv2/configs/upernet.yaml`:

```yaml
model:
  model_path: "nnunetv2.model.model_upernet"
  backbone: "dinov3"
  backbone_size: "large"
  checkpoint_path: ""
  decoder:
    hidden_size: 512
    pool_scales: [1, 2, 3, 6]

lora:
  enabled: false
  r: 8
  lora_alpha: 16
  target_modules: ["qkv"]

train:
  initial_lr: 0.001
  weight_decay: 0.00003
  gradient_clip: 12.0
  num_epochs: 1000
  num_iterations_per_epoch: 250
  num_val_iterations_per_epoch: 50
```

## Key Design Decisions

### Why Use Only UperNetHead (Not Full Model)

The full `UperNetForSemanticSegmentation` includes its own backbone loading via `load_backbone()`. This project already has:
- Custom backbone system (`create_backbone` supporting dinov3, dinov2, retfound, visionfm)
- LoRA integration via peft
- Checkpoint loading for foundation models

Using only `UperNetHead` allows reusing the existing infrastructure.

### Resolution Handling

UperNetHead outputs at the same resolution as input features (H/patch_size, W/patch_size). Need `ProgressiveUpsampler` from `backbone.py` to upsample to full resolution:

```python
# UperNetHead output: [B, num_classes, H/16, W/16]
# ProgressiveUpsampler: 2 stages of 2x = 4x total
# Final output: [B, num_classes, H, W] after interpolation
```

### Input Padding

ViT backbones require input dimensions divisible by patch_size. Follow existing pattern:

```python
ps = self.backbone_patch_size  # 16 for dinov3, 14 for dinov2
pad_h = (ps - h % ps) % ps
pad_w = (ps - w % ps) % ps
if pad_h > 0 or pad_w > 0:
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
```

## Usage

After implementation:

```bash
python nnUNetv2_train_vit.py DATASET_ID 2d FOLD --config nnunetv2/configs/upernet.yaml
```

## References

- Existing SegFormer implementation: `nnunetv2/model/model_segformer_1.py`
- Transformers UperNet source: `/home/xujialiu/miniconda3/envs/dinov3/lib/python3.11/site-packages/transformers/models/upernet/modeling_upernet.py`
- UperNet paper: https://arxiv.org/abs/1807.10221
