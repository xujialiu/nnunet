# UperNet ViT Integration Implementation Plan

## Overview

Add UperNet segmentation head to ViT backbone, leveraging the HuggingFace transformers library's `UperNetHead` class. This follows the existing pattern established by `model_segformer_1.py`.

## Current State Analysis

The codebase has a well-established pattern for adding new decoder heads:
- `model_segformer_1.py:20-307` - Template implementation with ViT backbone + HuggingFace decoder head
- `backbone.py:35-196` - Shared utilities (`create_backbone`, `get_vit_features`, `ProgressiveUpsampler`)
- `nnUNetTrainer_vit.py:88-130` - Dynamic model loading via `create_segmentation_model()` factory

### Key Discoveries:
- Feature extraction uses layer indices: 12-block ViT → [2,5,8,11], 24-block ViT → [5,11,17,23] (`model_segformer_1.py:28-31`)
- Input padding to patch_size multiple uses reflect mode (`model_segformer_1.py:114-118`)
- `ProgressiveUpsampler` provides 2-stage 4x upsampling with learned refinement (`backbone.py:113-196`)
- UperNet requires 4 feature levels with same channel dimension for ViT (`UperNetHead(config, in_channels=[1024]*4)`)

## Desired End State

A new model file `nnunetv2/model/model_upernet.py` that:
1. Uses HuggingFace `UperNetHead` with ViT backbone
2. Follows identical API as `model_segformer_1.py`
3. Works with existing trainer and YAML config system
4. Supports all existing backbones (dinov3, dinov2, retfound, visionfm)

**Verification:**
```bash
python nnUNetv2_train_vit.py DATASET_ID 2d FOLD --config nnunetv2/configs/upernet.yaml
```

## What We're NOT Doing

- NOT using full `UperNetForSemanticSegmentation` (includes its own backbone loading)
- NOT adding new backbone types
- NOT modifying existing trainer code
- NOT adding auxiliary FCN head (optional future enhancement)

## Implementation Approach

Mirror `model_segformer_1.py` structure exactly, replacing SegformerDecodeHead with UperNetHead.

---

## Phase 1: Create UperNet Model Module

### Overview
Create `nnunetv2/model/model_upernet.py` with `UperNetSegmentationModel` class and factory function.

### Changes Required:

#### 1. Create Model File
**File**: `nnunetv2/model/model_upernet.py`

```python
"""
UperNet Segmentation Model with ViT backbone.

Uses HuggingFace transformers UperNetHead with multi-scale ViT features.
Architecture follows the UperNet paper: https://arxiv.org/abs/1807.10221
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import UperNetConfig
from transformers.models.upernet.modeling_upernet import UperNetHead

from nnunetv2.model.backbone import (
    ProgressiveUpsampler,
    create_backbone,
    get_vit_features,
)


class UperNetSegmentationModel(nn.Module):
    """
    UperNet segmentation model with ViT backbone.

    Combines a Vision Transformer backbone with UperNetHead decoder for
    semantic segmentation. Supports LoRA fine-tuning via peft library.
    """

    # Layer indices for extracting 4 feature levels from ViT
    LAYER_INDICES = {
        12: [2, 5, 8, 11],    # Base models (12 transformer blocks)
        24: [5, 11, 17, 23],  # Large models (24 transformer blocks)
    }

    def __init__(
        self,
        backbone_name: str = "dinov3",
        backbone_size: str = "large",
        num_classes: int = 1,
        decoder_hidden_size: int = 512,
        pool_scales: Tuple[int, ...] = (1, 2, 3, 6),
        checkpoint_path: Optional[str] = None,
        freeze_backbone: bool = False,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.decoder_hidden_size = decoder_hidden_size

        # Create ViT backbone
        self.backbone, self.backbone_patch_size, _ = create_backbone(
            model_name=backbone_name,
            model_size=backbone_size,
            checkpoint_path=checkpoint_path,
        )

        # Determine feature extraction layers based on backbone depth
        num_layers = len(self.backbone.blocks)
        if num_layers not in self.LAYER_INDICES:
            raise ValueError(
                f"Unsupported ViT depth: {num_layers}. "
                f"Supported: {list(self.LAYER_INDICES.keys())}"
            )
        self.feature_indices = self.LAYER_INDICES[num_layers]

        # Get backbone embedding dimension
        self.embed_dim = self.backbone.embed_dim

        # Configure UperNetHead
        # UperNet expects 4 feature levels with channel dimensions
        upernet_config = UperNetConfig(
            num_labels=num_classes,
            hidden_size=decoder_hidden_size,
            pool_scales=list(pool_scales),
            use_auxiliary_head=False,
        )

        # in_channels: all same for ViT (uniform embedding dimension)
        self.decode_head = UperNetHead(
            upernet_config,
            in_channels=[self.embed_dim] * 4,
        )

        # Progressive upsampler: UperNetHead outputs at H/patch_size resolution
        # Need to upsample to full resolution
        self.upsampler = ProgressiveUpsampler(
            in_channels=num_classes,  # UperNetHead outputs num_labels channels
            num_classes=num_classes,
            num_stages=2,  # 2x * 2x = 4x upsampling
            hidden_dim=decoder_hidden_size // 2,
        )

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Optionally enable LoRA
        if use_lora:
            self.enable_lora(
                rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        _, _, h, w = x.shape

        # Pad input to be divisible by patch_size
        ps = self.backbone_patch_size
        pad_h = (ps - h % ps) % ps
        pad_w = (ps - w % ps) % ps
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        padded_size = x.shape[-2:]

        # Extract multi-scale features from ViT
        features = get_vit_features(
            self.backbone,
            x,
            indices=self.feature_indices,
        )

        # UperNetHead forward: expects list of 4 feature tensors
        # Output: [B, num_labels, H/patch_size, W/patch_size]
        decoder_output = self.decode_head(features)

        # Upsample to full resolution
        logits = self.upsampler(decoder_output, padded_size)

        # Crop to original size
        if pad_h > 0 or pad_w > 0:
            logits = logits[:, :, :h, :w]

        return logits

    def print_trainable_parameters(self, detailed: bool = False) -> None:
        """Print count of trainable vs total parameters."""
        trainable_params = 0
        total_params = 0

        component_params = {
            "backbone": {"trainable": 0, "total": 0},
            "decode_head": {"trainable": 0, "total": 0},
            "upsampler": {"trainable": 0, "total": 0},
        }

        for name, param in self.named_parameters():
            total_params += param.numel()
            component = name.split(".")[0]

            if component in component_params:
                component_params[component]["total"] += param.numel()
                if param.requires_grad:
                    component_params[component]["trainable"] += param.numel()
                    trainable_params += param.numel()
            else:
                if param.requires_grad:
                    trainable_params += param.numel()

        print(f"\nModel Parameter Summary:")
        print(f"{'='*50}")
        print(f"Total parameters:     {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable %:          {100 * trainable_params / total_params:.2f}%")

        if detailed:
            print(f"\n{'Component':<15} {'Trainable':>15} {'Total':>15} {'%':>10}")
            print(f"{'-'*55}")
            for comp, counts in component_params.items():
                pct = (
                    100 * counts["trainable"] / counts["total"]
                    if counts["total"] > 0
                    else 0
                )
                print(
                    f"{comp:<15} {counts['trainable']:>15,} "
                    f"{counts['total']:>15,} {pct:>9.2f}%"
                )

    def enable_lora(
        self,
        rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
    ) -> None:
        """Enable LoRA fine-tuning on the backbone."""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError(
                "peft is required for LoRA. Install with: pip install peft"
            )

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

        # Ensure decoder and upsampler remain trainable
        for param in self.decode_head.parameters():
            param.requires_grad = True
        for param in self.upsampler.parameters():
            param.requires_grad = True

        print(f"LoRA enabled with rank={rank}, alpha={lora_alpha}")
        self.print_trainable_parameters()

    def disable_lora(self) -> None:
        """Merge LoRA weights and disable LoRA."""
        try:
            from peft import PeftModel
        except ImportError:
            return

        if isinstance(self.backbone, PeftModel):
            self.backbone = self.backbone.merge_and_unload()
            print("LoRA weights merged and unloaded")


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
    pool_scales: Tuple[int, ...] = (1, 2, 3, 6),
    **kwargs,
) -> nn.Module:
    """
    Factory function to create UperNet segmentation model.

    This function is called by nnUNetTrainer_vit with parameters from YAML config.

    Args:
        backbone_name: ViT backbone type ("dinov3", "dinov2", "retfound", "visionfm")
        backbone_size: Model size ("base" or "large")
        num_classes: Number of segmentation classes
        checkpoint_path: Optional path to backbone weights
        freeze_backbone: Whether to freeze backbone (overridden if use_lora=True)
        use_lora: Whether to use LoRA fine-tuning
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha scaling
        lora_dropout: LoRA dropout
        lora_target_modules: Target modules for LoRA
        hidden_size: UperNet decoder hidden dimension
        pool_scales: Pyramid Pooling Module scales
        **kwargs: Additional kwargs (ignored with warning)

    Returns:
        UperNetSegmentationModel instance
    """
    if kwargs:
        print(f"Warning: Unknown kwargs passed to create_segmentation_model: {kwargs}")

    return UperNetSegmentationModel(
        backbone_name=backbone_name,
        backbone_size=backbone_size,
        num_classes=num_classes,
        decoder_hidden_size=hidden_size,
        pool_scales=pool_scales,
        checkpoint_path=checkpoint_path,
        freeze_backbone=freeze_backbone,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
    )
```

### Success Criteria:

#### Automated Verification:
- [x] File exists at `nnunetv2/model/model_upernet.py`
- [x] Python syntax is valid: `python -c "from nnunetv2.model.model_upernet import create_segmentation_model"`
- [x] Model instantiates without error (test in Phase 3)

---

## Phase 2: Create YAML Configuration

### Overview
Create `nnunetv2/configs/upernet.yaml` following the existing config pattern.

### Changes Required:

#### 1. Create Config File
**File**: `nnunetv2/configs/upernet.yaml`

```yaml
# UperNet configuration for nnUNet integration
# Architecture: https://arxiv.org/abs/1807.10221
# Uses Pyramid Pooling Module (PPM) and Feature Pyramid Network (FPN)

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
  lora_dropout: 0.05

train:
  initial_lr: 0.001
  weight_decay: 0.00003
  gradient_clip: 12.0
  num_epochs: 1000
  num_iterations_per_epoch: 250
  num_val_iterations_per_epoch: 50
```

### Success Criteria:

#### Automated Verification:
- [x] File exists at `nnunetv2/configs/upernet.yaml`
- [x] YAML syntax is valid: `python -c "from omegaconf import OmegaConf; OmegaConf.load('nnunetv2/configs/upernet.yaml')"`

---

## Phase 3: Verification

### Overview
Verify the implementation works correctly with model instantiation and a forward pass.

### Test Script

```python
# Test script to verify UperNet implementation
import torch
from nnunetv2.model.model_upernet import create_segmentation_model

# Test 1: Model instantiation
print("Test 1: Model instantiation...")
model = create_segmentation_model(
    backbone_name="dinov3",
    backbone_size="large",
    num_classes=4,
    use_lora=False,
)
print("✓ Model instantiated successfully")

# Test 2: Forward pass
print("\nTest 2: Forward pass...")
model.eval()
with torch.no_grad():
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == (2, 4, 224, 224), f"Expected (2, 4, 224, 224), got {output.shape}"
print("✓ Forward pass successful")

# Test 3: Non-divisible input
print("\nTest 3: Non-divisible input size...")
with torch.no_grad():
    x = torch.randn(2, 3, 237, 189)  # Not divisible by 16
    output = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == (2, 4, 237, 189), f"Expected (2, 4, 237, 189), got {output.shape}"
print("✓ Non-divisible input handled correctly")

# Test 4: LoRA
print("\nTest 4: LoRA fine-tuning...")
model_lora = create_segmentation_model(
    backbone_name="dinov3",
    backbone_size="large",
    num_classes=4,
    use_lora=True,
    lora_rank=8,
)
print("✓ LoRA model instantiated successfully")

# Test 5: Parameter counting
print("\nTest 5: Parameter counting...")
model_lora.print_trainable_parameters(detailed=True)

print("\n" + "="*50)
print("All tests passed!")
```

### Success Criteria:

#### Automated Verification:
- [x] Test script runs without errors: `python test_upernet.py`
- [x] Output shape matches input spatial dimensions
- [x] LoRA model instantiates correctly
- [x] Parameter counts are reported

#### Manual Verification:
- [ ] Full training run completes: `python nnUNetv2_train_vit.py DATASET_ID 2d 0 --config nnunetv2/configs/upernet.yaml`
- [ ] Training loss decreases over epochs
- [ ] Validation metrics are computed

---

## Testing Strategy

### Unit Tests:
- Model instantiation with all backbone types (dinov3, dinov2, retfound, visionfm)
- Forward pass with various input sizes
- LoRA enable/disable cycle
- Parameter counting accuracy

### Integration Tests:
- Config loading via OmegaConf
- Factory function invocation from trainer
- Full training loop for 1 epoch

### Manual Testing Steps:
1. Run short training (5 epochs) on a small dataset
2. Verify loss decreases
3. Verify checkpoint saving works
4. Verify inference produces sensible outputs

## Performance Considerations

- UperNetHead has Pyramid Pooling Module which requires adaptive pooling
- Pool scales [1, 2, 3, 6] are standard; larger scales increase memory usage
- `hidden_size=512` is a reasonable default; can be reduced for memory efficiency
- ProgressiveUpsampler adds learned refinement vs. direct bilinear upsampling

## References

- Research document: `thoughts/shared/research/2026-01-23-upernet-vit-integration.md`
- Template implementation: `nnunetv2/model/model_segformer_1.py:20-307`
- Backbone utilities: `nnunetv2/model/backbone.py:35-196`
- Trainer: `nnunetv2/training/nnUNetTrainer/nnUNetTrainer_vit.py:88-130`
- UperNet paper: https://arxiv.org/abs/1807.10221
