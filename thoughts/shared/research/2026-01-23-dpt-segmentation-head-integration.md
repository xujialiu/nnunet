---
date: 2026-01-23T12:00:00+08:00
researcher: Claude
git_commit: 8c7c01c0bea98438dc0fe49d66710e307398ce98
branch: add_other_seg_heads
repository: add_other_seg_heads
topic: "Adding DPT Segmentation Head to ViT Backbone"
tags: [research, codebase, dpt, segmentation, vit, decoder]
status: complete
last_updated: 2026-01-23
last_updated_by: Claude
last_updated_note: "Added follow-up research on CLS token extraction from DINOv3/DINOv2"
---

# Research: Adding DPT Segmentation Head to ViT Backbone

**Date**: 2026-01-23
**Researcher**: Claude
**Git Commit**: 8c7c01c0bea98438dc0fe49d66710e307398ce98
**Branch**: add_other_seg_heads
**Repository**: add_other_seg_heads

## Research Question

How to add a DPT (Dense Prediction Transformer) segmentation head to the ViT backbone using `nnUNetv2_train_vit.py` as entry point, using the transformers library DPT implementation.

## Summary

The codebase follows a modular **backbone + decoder** pattern where ViT backbones extract multi-scale features and decoder heads produce segmentation maps. Adding a DPT decoder requires:

1. Creating a new model file (`model_dpt.py`) following the existing pattern
2. Implementing `DPTSegmentationModel` class with the same interface as existing models
3. Exporting a `create_segmentation_model()` factory function
4. Creating a YAML config file (`dpt.yaml`)

The transformers library provides `DPTNeck` and `DPTSemanticSegmentationHead` components that can be extracted and used with custom ViT backbones.

## Detailed Findings

### 1. Existing Model Architecture Pattern

All models in `nnunetv2/model/` follow a consistent pattern:

| File | Model Class | Decoder Type |
|------|-------------|--------------|
| `model_segformer_1.py` | `SegFormerSegmentationModel` | SegformerDecodeHead + ProgressiveUpsampler |
| `primus.py` | `PrimusSegmentationModel` | PatchDecode (transposed convolutions) |
| `primus.py` | `PrimusMultiscaleSegmentationModel` | PatchDecode with multi-scale features |

**Common Interface**:
```python
class XxxSegmentationModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "dinov3",
        backbone_size: str = "large",
        num_classes: int = 1,
        checkpoint_path: Optional[str] = None,
        freeze_backbone: bool = False,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        # Decoder-specific kwargs...
    ):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def enable_lora(...): ...
    def disable_lora(): ...
    def print_trainable_parameters(detailed: bool = False): ...
```

**Factory Function** (required for trainer integration):
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
    **kwargs,  # Decoder-specific params
) -> nn.Module:
```

### 2. Multi-Scale Feature Extraction

The backbone utility (`backbone.py:77-110`) provides `get_vit_features()` to extract intermediate layer features:

```python
LAYER_INDICES = {
    12: [2, 5, 8, 11],    # Base models (12-layer ViT)
    24: [5, 11, 17, 23],  # Large models (24-layer ViT)
}

features = get_vit_features(backbone, x, indices=layer_indices)
# Returns list of 4 tensors: (B, C, H/patch, W/patch)
```

DPT expects exactly 4 feature maps from different layers, which aligns perfectly with this pattern.

### 3. Transformers Library DPT Components

**Verified available in conda dinov3 environment**:
```python
from transformers.models.dpt.modeling_dpt import (
    DPTNeck,                      # Reassemble + Fusion stages
    DPTSemanticSegmentationHead,  # Final segmentation head
    DPTConfig,                    # Configuration
)
```

**DPT Default Config**:
```python
DPTConfig(
    backbone_out_indices=[2, 5, 8, 11],  # Layer indices for features
    neck_hidden_sizes=[96, 192, 384, 768],  # Channel sizes per scale
    fusion_hidden_size=256,  # Unified fusion channels
    reassemble_factors=[4, 2, 1, 0.5],  # Up/downsampling factors
    readout_type="project",  # CLS token handling
    semantic_classifier_dropout=0.1,
)
```

**DPT Architecture Flow**:
```
ViT Backbone (4 layers)
    │
    ▼
DPTNeck
├── DPTReassembleStage (token→spatial, channel projection)
└── DPTFeatureFusionStage (multi-scale fusion with residual refinement)
    │
    ▼
DPTSemanticSegmentationHead
├── Conv(256→256) + BN + ReLU
├── Dropout
├── Conv(256→num_labels)
└── 2x Upsample
    │
    ▼
Segmentation logits
```

### 4. Trainer Integration Flow

The trainer (`nnUNetTrainerNoDeepSupervision_vit`) dynamically loads models:

1. **Config Loading** (`nnUNetTrainer_vit.py:61-86`):
   - Merges YAML with schema defaults via OmegaConf
   - Converts to typed dataclass `ViTnnUNetConfig`

2. **Model Import** (`nnUNetTrainer_vit.py:97-111`):
   ```python
   model_module = importlib.import_module(cfg.model.model_path)
   create_model_fn = model_module.create_segmentation_model
   ```

3. **Model Instantiation** (`nnUNetTrainer_vit.py:113-130`):
   ```python
   decoder_kwargs = dict(cfg.model.decoder) if cfg.model.decoder else {}
   self.network = create_model_fn(
       backbone_name=cfg.model.backbone,
       backbone_size=cfg.model.backbone_size,
       num_classes=self.label_manager.num_segmentation_heads,
       use_lora=cfg.lora.enabled,
       **decoder_kwargs,
   ).to(self.device)
   ```

### 5. Config File Pattern

**Example from segformer.yaml**:
```yaml
model:
  model_path: "nnunetv2.model.model_segformer_1"
  backbone: "dinov3"
  backbone_size: "large"
  checkpoint_path: ""
  decoder:
    hidden_size: 256

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

## Code References

- `nnUNetv2_train_vit.py:13-14` - Entry point importing run_training_entry
- `nnunetv2/run/run_training_vit.py:302-429` - CLI argument parsing
- `nnunetv2/run/run_training_vit.py:34-87` - Trainer instantiation with config
- `nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainerNoDeepSupervision_vit.py:8-19` - Trainer class
- `nnunetv2/training/nnUNetTrainer/nnUNetTrainer_vit.py:61-86` - Config loading
- `nnunetv2/training/nnUNetTrainer/nnUNetTrainer_vit.py:97-130` - Model instantiation
- `nnunetv2/model/backbone.py:35-74` - `create_backbone()` factory
- `nnunetv2/model/backbone.py:77-110` - `get_vit_features()` multi-scale extraction
- `nnunetv2/model/backbone.py:113-197` - `ProgressiveUpsampler` class
- `nnunetv2/model/model_segformer_1.py:27-143` - SegFormer model implementation
- `nnunetv2/model/model_segformer_1.py:197-240` - LoRA integration
- `nnunetv2/model/model_segformer_1.py:258-307` - Factory function
- `nnunetv2/model/primus.py:79-277` - Primus model implementation
- `nnunetv2/model/primus.py:321-547` - Primus multiscale implementation
- `nnunetv2/model/vit_config.py:8-53` - Config schema definitions
- `nnunetv2/configs/segformer.yaml` - Example SegFormer config
- `nnunetv2/configs/primus.yaml` - Example Primus config

## Implementation Blueprint

### New File: `nnunetv2/model/model_dpt.py`

```python
"""DPT (Dense Prediction Transformer) segmentation model for nnUNet."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from transformers import DPTConfig
from transformers.models.dpt.modeling_dpt import (
    DPTNeck,
    DPTSemanticSegmentationHead,
)

from .backbone import create_backbone, get_vit_features

LAYER_INDICES = {
    12: [2, 5, 8, 11],    # Base models (12-layer ViT)
    24: [5, 11, 17, 23],  # Large models (24-layer ViT)
}


class DPTSegmentationModel(nn.Module):
    """DPT decoder with ViT backbone for semantic segmentation."""

    def __init__(
        self,
        backbone_name: str = "dinov3",
        backbone_size: str = "large",
        num_classes: int = 1,
        checkpoint_path: Optional[str] = None,
        freeze_backbone: bool = False,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        # DPT-specific parameters
        fusion_hidden_size: int = 256,
        readout_type: str = "project",  # "project", "add", or "ignore"
    ):
        super().__init__()

        self.num_classes = num_classes
        self.readout_type = readout_type

        # Create backbone
        self.backbone, self.patch_size, _ = create_backbone(
            backbone_name, backbone_size, checkpoint_path
        )

        # Get backbone info
        embed_dim = self.backbone.embed_dim
        num_layers = len(self.backbone.blocks)
        self.layer_indices = LAYER_INDICES.get(num_layers, [2, 5, 8, 11])

        # DPT expects same hidden size for all layers (ViT is uniform)
        neck_hidden_sizes = [embed_dim] * 4

        # Create DPT config
        dpt_config = DPTConfig(
            hidden_size=embed_dim,
            neck_hidden_sizes=neck_hidden_sizes,
            fusion_hidden_size=fusion_hidden_size,
            reassemble_factors=[4, 2, 1, 0.5],
            readout_type=readout_type,
            num_labels=num_classes,
            head_in_index=-1,
            semantic_classifier_dropout=0.1,
            use_batch_norm_in_fusion_residual=True,
        )

        # DPT decoder components
        self.neck = DPTNeck(dpt_config)
        self.head = DPTSemanticSegmentationHead(dpt_config)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Enable LoRA if requested
        if use_lora:
            self.enable_lora(lora_rank, lora_alpha, lora_dropout, lora_target_modules)

    def _get_dpt_features(self, x: torch.Tensor):
        """Extract features in DPT format with real CLS tokens."""
        # Use forward_intermediates with return_prefix_tokens=True
        _, feats = self.backbone.forward_intermediates(
            x,
            indices=self.layer_indices,
            norm=True,
            output_fmt='NCHW',
            return_prefix_tokens=True,
        )

        dpt_feats = []
        patch_h, patch_w = None, None
        for spatial, prefix in feats:
            B, C, H, W = spatial.shape
            patch_h, patch_w = H, W
            # Flatten spatial: (B, C, H, W) -> (B, H*W, C)
            spatial_flat = spatial.flatten(2).transpose(1, 2)
            # Get CLS token (first prefix token, works for DINOv2/DINOv3)
            cls_token = prefix[:, 0:1, :]  # (B, 1, C)
            # Concatenate: [CLS] + [patch tokens]
            feat_with_cls = torch.cat([cls_token, spatial_flat], dim=1)
            dpt_feats.append(feat_with_cls)

        return dpt_feats, patch_h, patch_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Pad to patch_size divisible
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        padded_H, padded_W = x.shape[2], x.shape[3]

        # Extract features with real CLS tokens
        hidden_states_seq, patch_height, patch_width = self._get_dpt_features(x)

        # DPT Neck: reassemble + fusion
        neck_outputs = self.neck(
            hidden_states_seq,
            patch_height=patch_height,
            patch_width=patch_width,
        )

        # DPT Head: final segmentation
        logits = self.head(neck_outputs)

        # Upsample to padded size (head outputs at stride 2)
        logits = F.interpolate(
            logits, size=(padded_H, padded_W), mode="bilinear", align_corners=False
        )

        # Crop to original size
        if pad_h > 0 or pad_w > 0:
            logits = logits[:, :, :H, :W]

        return logits

    def enable_lora(
        self,
        rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
    ) -> None:
        """Enable LoRA fine-tuning on backbone."""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError("peft is required for LoRA. Install with: pip install peft")

        if target_modules is None:
            target_modules = ["qkv"]

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )

        self.backbone = get_peft_model(self.backbone, lora_config)

        # Ensure decoder remains trainable
        for param in self.neck.parameters():
            param.requires_grad = True
        for param in self.head.parameters():
            param.requires_grad = True

    def disable_lora(self) -> None:
        """Merge LoRA weights and remove adapters."""
        try:
            from peft import PeftModel
        except ImportError:
            return

        if isinstance(self.backbone, PeftModel):
            self.backbone = self.backbone.merge_and_unload()

    def print_trainable_parameters(self, detailed: bool = False) -> None:
        """Print trainable parameter statistics."""
        trainable = 0
        total = 0

        for name, param in self.named_parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
                if detailed:
                    print(f"  {name}: {param.numel():,}")

        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


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
    # DPT-specific
    fusion_hidden_size: int = 256,
    readout_type: str = "project",
    **kwargs,
) -> nn.Module:
    """Factory function for DPT segmentation model."""

    if kwargs:
        print(f"Warning: Unused kwargs in create_segmentation_model: {kwargs}")

    return DPTSegmentationModel(
        backbone_name=backbone_name,
        backbone_size=backbone_size,
        num_classes=num_classes,
        checkpoint_path=checkpoint_path,
        freeze_backbone=freeze_backbone,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        fusion_hidden_size=fusion_hidden_size,
        readout_type=readout_type,
    )
```

### New File: `nnunetv2/configs/dpt.yaml`

```yaml
model:
  model_path: "nnunetv2.model.model_dpt"
  backbone: "dinov3"
  backbone_size: "large"
  checkpoint_path: ""
  decoder:
    fusion_hidden_size: 256
    readout_type: "project"  # "project", "add", or "ignore"

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

### Usage Command

```bash
python nnUNetv2_train_vit.py DATASET_ID 2d FOLD --config nnunetv2/configs/dpt.yaml
```

## Key Implementation Notes

1. **CLS Token Handling**: DPT's `DPTNeck` expects a CLS token at position 0 of each feature sequence. Use `forward_intermediates(..., return_prefix_tokens=True)` to get real CLS tokens instead of dummy mean pooling.

2. **Feature Format Conversion**: `forward_intermediates()` returns `(B, C, H, W)` spatial tensors. Convert to `(B, seq_len+1, C)` by flattening spatial dims and prepending CLS token.

3. **Reassemble Factors**: DPT uses `[4, 2, 1, 0.5]` factors to create multi-resolution feature maps from the 4 input features.

4. **Neck Hidden Sizes**: For ViT backbones with uniform embedding dimension (e.g., 1024 for ViT-L), set all `neck_hidden_sizes` to the same value.

5. **Final Upsampling**: DPT head includes 2x upsampling, but additional interpolation may be needed to match input resolution.

## Follow-up Research: CLS Token Strategy (2026-01-23)

### Question
Can DINOv3's feature extraction return actual CLS tokens instead of using dummy mean pooling?

### Answer: YES

The timm models support `return_prefix_tokens=True` in `forward_intermediates()`:

```python
_, feats = model.forward_intermediates(
    x,
    indices=[5, 11, 17, 23],
    norm=True,
    output_fmt='NCHW',
    return_prefix_tokens=True,  # <-- Key parameter
)
# Returns list of (spatial_tensor, prefix_tokens) tuples
```

### Prefix Token Structure

| Model | Model Class | Prefix Tokens | Structure |
|-------|-------------|---------------|-----------|
| DINOv3 | `timm.models.eva.Eva` | 5 | 1 CLS + 4 register tokens |
| DINOv2 | `timm.models.vision_transformer.VisionTransformer` | 1 | 1 CLS only |

### Implementation

```python
def get_dpt_features(model, x, indices):
    """Extract features in DPT format: (B, seq_len+1, C) with real CLS at position 0"""
    _, feats = model.forward_intermediates(
        x, indices=indices, norm=True, output_fmt='NCHW', return_prefix_tokens=True,
    )

    dpt_feats = []
    for spatial, prefix in feats:
        B, C, H, W = spatial.shape
        # Flatten spatial: (B, C, H, W) -> (B, H*W, C)
        spatial_flat = spatial.flatten(2).transpose(1, 2)
        # Get CLS token (first prefix token, works for both DINOv2 and DINOv3)
        cls_token = prefix[:, 0:1, :]  # (B, 1, C)
        # Concatenate: [CLS] + [patch tokens]
        feat_with_cls = torch.cat([cls_token, spatial_flat], dim=1)  # (B, H*W+1, C)
        dpt_feats.append(feat_with_cls)

    return dpt_feats, H, W
```

### Verified Working

Tested complete forward pass:
```
DINOv3 backbone → get_dpt_features() → DPTNeck → DPTSemanticSegmentationHead
Input: (2, 3, 224, 224)
Features: 4 tensors, each (2, 197, 1024) = [CLS] + [14x14 patches]
Neck outputs: [(2, 256, 14, 14), (2, 256, 28, 28), (2, 256, 56, 56), (2, 256, 112, 112)]
Final logits: (2, 4, 224, 224)
```

### Recommendation

Use real CLS tokens via `return_prefix_tokens=True` instead of dummy mean pooling. This preserves the semantic meaning of the CLS token which was trained to aggregate global image information.

## Remaining Open Questions

~~1. **Readout Type Tuning**: The `readout_type` ("project", "add", "ignore") affects how CLS tokens are incorporated. May need experimentation to find best option for medical imaging.~~

**RESOLVED**: Made configurable via `decoder.readout_type` in `dpt.yaml`. Options: "project" (default), "add", "ignore".

~~2. **Batch Norm vs Layer Norm**: DPT uses batch norm in fusion residuals. May need to test with `use_batch_norm_in_fusion_residual=False` for small batch sizes typical in medical imaging.~~

**RESOLVED**: Made configurable via `decoder.use_batch_norm_in_fusion_residual` in `dpt.yaml`. Set to `false` for small batch sizes.
