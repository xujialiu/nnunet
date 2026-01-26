# DPT Segmentation Head Implementation Plan

## Overview

Add a DPT (Dense Prediction Transformer) segmentation decoder to the ViT backbone system, following the existing model patterns (UperNet, SegFormer, Primus). The implementation uses HuggingFace transformers library's `DPTNeck` and `DPTSemanticSegmentationHead` components.

## Current State Analysis

### Existing Architecture
- All models in `nnunetv2/model/` follow a consistent pattern:
  - `create_segmentation_model()` factory function
  - Class with backbone + decoder + optional upsampler
  - LoRA integration via peft library
  - `LAYER_INDICES` for multi-scale feature extraction

### Key Components Available
- `backbone.py:35-74` - `create_backbone()` factory for ViT backbones
- `backbone.py:77-110` - `get_vit_features()` for multi-scale feature extraction
- `backbone.py:113-196` - `ProgressiveUpsampler` for learned upsampling

### HuggingFace DPT Components
The transformers library provides:
- `DPTConfig` - Configuration class
- `DPTNeck` - Reassemble + Fusion stages
- `DPTSemanticSegmentationHead` - Final segmentation head

## Desired End State

After implementation:
1. New file `nnunetv2/model/model_dpt.py` with `DPTSegmentationModel` class
2. New config `nnunetv2/configs/dpt.yaml`
3. Test config `nnunetv2/configs/dpt_test_dinov3.yaml`
4. Training can be started with: `python nnUNetv2_train_vit.py DATASET_ID 2d FOLD --config nnunetv2/configs/dpt.yaml`

### Verification
- Model instantiates without errors
- Forward pass produces correct output shape
- Training runs successfully for at least 10 iterations
- LoRA can be enabled/disabled

## What We're NOT Doing

- Adding new backbone types (using existing dinov3/dinov2/retfound/visionfm)
- Modifying the trainer (`nnUNetTrainer_vit.py`)
- Adding new dependencies (transformers already available)
- Performance tuning or hyperparameter optimization

## Implementation Approach

The DPT decoder differs from existing decoders in one key aspect: it expects CLS tokens prepended to feature sequences. We'll use `forward_intermediates(..., return_prefix_tokens=True)` to get real CLS tokens from the backbone, rather than using dummy mean pooling.

## Phase 1: Create DPT Model File

### Overview
Create `model_dpt.py` following the exact pattern of `model_upernet.py`.

### Changes Required:

#### 1. New File: `nnunetv2/model/model_dpt.py`

```python
"""
DPT (Dense Prediction Transformer) Segmentation Model with ViT backbone.

Uses HuggingFace transformers DPTNeck and DPTSemanticSegmentationHead.
Architecture follows the DPT paper: https://arxiv.org/abs/2103.13413
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DPTConfig
from transformers.models.dpt.modeling_dpt import (
    DPTNeck,
    DPTSemanticSegmentationHead,
)

from nnunetv2.model.backbone import create_backbone


class DPTSegmentationModel(nn.Module):
    """
    DPT segmentation model with ViT backbone.

    Combines a Vision Transformer backbone with DPT decoder for
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
        fusion_hidden_size: int = 256,
        readout_type: str = "project",
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
        self.fusion_hidden_size = fusion_hidden_size
        self.readout_type = readout_type

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

        # DPT expects same hidden size for all layers (ViT is uniform)
        neck_hidden_sizes = [self.embed_dim] * 4

        # Configure DPT components
        dpt_config = DPTConfig(
            hidden_size=self.embed_dim,
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

    def _get_dpt_features(self, x: torch.Tensor):
        """
        Extract features in DPT format with real CLS tokens.

        DPT expects features as (B, seq_len+1, C) with CLS token at position 0.
        Uses forward_intermediates with return_prefix_tokens=True to get actual CLS tokens.
        """
        _, feats = self.backbone.forward_intermediates(
            x,
            indices=self.feature_indices,
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
        """Forward pass."""
        B, C, H, W = x.shape

        # Pad input to be divisible by patch_size
        ps = self.backbone_patch_size
        pad_h = (ps - H % ps) % ps
        pad_w = (ps - W % ps) % ps
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

    def print_trainable_parameters(self, detailed: bool = False) -> None:
        """Print count of trainable vs total parameters."""
        trainable_params = 0
        total_params = 0

        component_params = {
            "backbone": {"trainable": 0, "total": 0},
            "neck": {"trainable": 0, "total": 0},
            "head": {"trainable": 0, "total": 0},
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

        # Ensure decoder remains trainable
        for param in self.neck.parameters():
            param.requires_grad = True
        for param in self.head.parameters():
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
    fusion_hidden_size: int = 256,
    readout_type: str = "project",
    **kwargs,
) -> nn.Module:
    """
    Factory function to create DPT segmentation model.

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
        fusion_hidden_size: DPT fusion hidden dimension (default 256)
        readout_type: How to handle CLS token ("project", "add", or "ignore")
        **kwargs: Additional kwargs (ignored with warning)

    Returns:
        DPTSegmentationModel instance
    """
    if kwargs:
        print(f"Warning: Unknown kwargs passed to create_segmentation_model: {kwargs}")

    return DPTSegmentationModel(
        backbone_name=backbone_name,
        backbone_size=backbone_size,
        num_classes=num_classes,
        fusion_hidden_size=fusion_hidden_size,
        readout_type=readout_type,
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
- [x] File exists at `nnunetv2/model/model_dpt.py`
- [x] Python syntax is valid: `python -c "from nnunetv2.model.model_dpt import create_segmentation_model"`
- [x] No import errors with transformers DPT components

#### Manual Verification:
- [ ] Code follows the same structure as `model_upernet.py`

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation before proceeding to the next phase.

---

## Phase 2: Create Config Files

### Overview
Create production and test YAML configs for DPT.

### Changes Required:

#### 1. New File: `nnunetv2/configs/dpt.yaml`

```yaml
# DPT (Dense Prediction Transformer) configuration for nnUNet integration
# Architecture: https://arxiv.org/abs/2103.13413
# Uses reassemble + fusion stages with multi-scale features

model:
  model_path: "nnunetv2.model.model_dpt"
  backbone: "dinov3"
  backbone_size: "large"
  checkpoint_path: ""
  decoder:
    fusion_hidden_size: 256
    readout_type: "project"  # Options: "project", "add", "ignore"

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

#### 2. New File: `nnunetv2/configs/dpt_test_dinov3.yaml`

```yaml
# DPT test configuration (minimal iterations for testing)

model:
  model_path: "nnunetv2.model.model_dpt"
  backbone: "dinov3"
  backbone_size: "large"
  checkpoint_path: ""
  decoder:
    fusion_hidden_size: 256
    readout_type: "project"

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
  num_iterations_per_epoch: 10
  num_val_iterations_per_epoch: 10
```

### Success Criteria:

#### Automated Verification:
- [x] File exists at `nnunetv2/configs/dpt.yaml`
- [x] File exists at `nnunetv2/configs/dpt_test_dinov3.yaml`
- [x] YAML syntax is valid: `python -c "import yaml; yaml.safe_load(open('nnunetv2/configs/dpt.yaml'))"`

#### Manual Verification:
- [ ] Config structure matches other configs (segformer.yaml, upernet.yaml)

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation before proceeding to the next phase.

---

## Phase 3: Integration Testing

### Overview
Verify the model can be instantiated and run a forward pass.

### Test Script

Create a temporary test to verify integration:

```python
# Quick test script (run from project root)
import torch
from nnunetv2.model.model_dpt import create_segmentation_model

# Create model
model = create_segmentation_model(
    backbone_name="dinov3",
    backbone_size="large",
    num_classes=4,
    use_lora=False,
)
model.eval()
model.cuda()

# Test forward pass
x = torch.randn(2, 3, 224, 224).cuda()
with torch.no_grad():
    y = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}")  # Expected: (2, 4, 224, 224)

# Test with non-divisible input
x2 = torch.randn(2, 3, 256, 320).cuda()
with torch.no_grad():
    y2 = model(x2)
print(f"Input shape: {x2.shape}")
print(f"Output shape: {y2.shape}")  # Expected: (2, 4, 256, 320)

# Test LoRA
model_lora = create_segmentation_model(
    backbone_name="dinov3",
    backbone_size="large",
    num_classes=4,
    use_lora=True,
    lora_rank=8,
)
model_lora.print_trainable_parameters(detailed=True)
```

### Success Criteria:

#### Automated Verification:
- [x] Model instantiates without errors
- [x] Forward pass produces output with correct shape (B, num_classes, H, W)
- [x] Non-divisible input sizes are handled correctly (padding + cropping)
- [x] LoRA can be enabled without errors

#### Manual Verification:
- [ ] Parameter counts look reasonable
- [ ] LoRA trainable % is significantly lower than full model

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation before proceeding to the next phase.

---

## Phase 4: Training Validation

### Overview
Run a short training session to verify full integration with the trainer.

### Command

```bash
# Using test config with minimal iterations
python nnUNetv2_train_vit.py DATASET_ID 2d 0 --config nnunetv2/configs/dpt_test_dinov3.yaml
```

Replace `DATASET_ID` with an available preprocessed dataset.

### Success Criteria:

#### Automated Verification:
- [ ] Training starts without errors
- [ ] Model is correctly loaded (check printed model info)
- [ ] At least 10 training iterations complete
- [ ] Validation runs without errors

#### Manual Verification:
- [ ] Loss decreases or stays stable (not NaN/Inf)
- [ ] GPU memory usage is reasonable

---

## Testing Strategy

### Unit Tests:
- Model instantiation with different backbones (dinov3, dinov2)
- Forward pass with various input sizes
- LoRA enable/disable

### Integration Tests:
- Config loading via trainer
- Full training loop for 10 iterations

### Manual Testing Steps:
1. Run forward pass test script
2. Start training with test config
3. Verify no errors in first epoch
4. Check tensorboard for loss curves (if applicable)

## Performance Considerations

- **GPU Memory**: DPT decoder adds ~5-10M parameters. Should fit on 24GB GPU with batch size 2-4.
- **Batch Norm**: DPT uses batch norm in fusion residuals. May need `use_batch_norm_in_fusion_residual=False` for very small batch sizes (1-2).
- **Readout Type**: Default "project" works well. "add" may be faster but slightly lower quality.

## References

- Research document: `thoughts/shared/research/2026-01-23-dpt-segmentation-head-integration.md`
- UperNet implementation (reference pattern): `nnunetv2/model/model_upernet.py`
- DPT paper: https://arxiv.org/abs/2103.13413
- HuggingFace DPT: https://huggingface.co/docs/transformers/model_doc/dpt
