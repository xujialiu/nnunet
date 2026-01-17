---
date: 2026-01-17T12:00:00+08:00
researcher: Claude
git_commit: 15e7aa13c7819a3b88bbafe41c97d5e07f153615
branch: use_m2f_dinov3
repository: use_m2f_dinov3
topic: "DINOv3 Mask2Former Integration with nnUNet Pipeline"
tags: [research, codebase, dinov3, mask2former, nnunet, omegaconf, segmentation]
status: complete
last_updated: 2026-01-17
last_updated_by: Claude
---

# Research: DINOv3 Mask2Former Integration with nnUNet Pipeline

**Date**: 2026-01-17T12:00:00+08:00
**Researcher**: Claude
**Git Commit**: 15e7aa13c7819a3b88bbafe41c97d5e07f153615
**Branch**: use_m2f_dinov3
**Repository**: use_m2f_dinov3

## Research Question

How to integrate the DINOv3 Mask2Former segmentation model (from `dinov3/eval/segmentation/train_m2f.py`) into the nnUNet training pipeline (replacing the current model in `nnunetv2/training/nnUNetTrainer/nnUNetTrainer_mymodel.py`) for fair comparison, using OmegaConf for parameter injection.

## Summary

This research documents two distinct segmentation frameworks in the codebase:

1. **DINOv3 Mask2Former** (`dinov3/eval/segmentation/`): A query-based segmentation architecture using a DINOv3 ViT backbone with a deformable attention pixel decoder and masked transformer decoder. Uses OmegaConf dataclasses for configuration.

2. **nnUNet with ViT Model** (`nnunetv2/`): A medical imaging segmentation framework with a custom ViT backbone (also DINOv3-based) using UperNet head and progressive upsampling decoder.

The key architectural differences and integration points are documented below.

---

## Detailed Findings

### 1. DINOv3 Mask2Former Architecture

**Entry Point**: `dinov3/eval/segmentation/run.py:52-66`

```
run_segmentation_with_dinov3()
    └── train_m2f_segmentation() [train_m2f.py:227]
```

**Model Construction**: `dinov3/eval/segmentation/models/__init__.py:76-148`

```python
build_segmentation_decoder(
    backbone_model,                           # DINOv3 ViT backbone
    backbone_out_layers=FOUR_EVEN_INTERVALS,  # [4, 11, 17, 23] for ViT-L
    decoder_type="m2f",
    hidden_dim=2048,
    num_classes=150,
    autocast_dtype=torch.float32,
    lora_enabled=False,
)
```

**Architecture Flow**:
```
Input Image [B, 3, H, W]
    │
    ▼
DINOv3_Adapter [backbone/dinov3_adapter.py:305-489]
    ├── SpatialPriorModule → multi-scale CNN features (c1-c4)
    ├── backbone.get_intermediate_layers() → ViT features at [4,11,17,23]
    ├── InteractionBlocks (deformable attention)
    └── SyncBatchNorm output
    │
    ▼
{"1": [B,D,H/4,W/4], "2": [B,D,H/8,W/8], "3": [B,D,H/16,W/16], "4": [B,D,H/32,W/32]}
    │
    ▼
MSDeformAttnPixelDecoder [heads/pixel_decoder.py:239-414]
    ├── Input projections (1x1 conv + GroupNorm)
    ├── 6-layer transformer encoder with deformable attention
    └── FPN-style upsampling
    │
    ▼
MultiScaleMaskedTransformerDecoder [heads/mask2former_transformer_decoder.py:271-472]
    ├── 100 learnable queries
    ├── 9 decoder layers (cross-attn → self-attn → FFN)
    └── Class/mask embedding heads
    │
    ▼
{
    "pred_logits": [B, 100, num_classes+1],
    "pred_masks": [B, 100, H/4, W/4],
    "aux_outputs": [9 intermediate predictions]
}
```

**Loss Function**: `MaskClassificationLoss` [mask_classification_loss.py]
- Hungarian matching for query-to-target assignment
- Mask loss (BCE + Dice) with point sampling
- Classification cross-entropy loss
- Deep supervision via `aux_outputs`

---

### 2. Current nnUNet Custom Model Architecture

**Entry Point**: `nnUNetv2_train_nodeepsupervision_mymodel.py`
```
run_training_entry() [run_training_nodeepsupervision_mymodel.py:301]
    └── nnUNetTrainerNoDeepSupervision_mymodel [variants/.../nnUNetTrainerNoDeepSupervision_mymodel.py]
        └── initialize() → create_segmentation_model() [nnunetv2/model/model.py:588]
```

**Model Construction**: `nnunetv2/model/model.py:588-642`

```python
create_segmentation_model(
    backbone_name="dinov3",
    backbone_size="large",
    head_type="upernet",
    num_classes=...,
    freeze_backbone=True,
    use_progressive_upsample=True,
    use_lora=True,
    lora_rank=8,
)
```

**Architecture Flow**:
```
Input Image [B, C, H, W]
    │
    ▼
timm.create_model("vit_large_patch16_dinov3.lvd1689m")
    │
    ▼
get_vit_features() → extract at [5, 11, 17, 23]
    │
    ▼
UperNetHead (from HuggingFace transformers)
    │
    ▼
ProgressiveUpsampleDecoder (4-stage 2x upsampling with skip connections)
    │
    ▼
Segmentation output [B, num_classes, H, W]
```

**Loss Function**: `DC_and_CE_loss` (Dice + Cross-Entropy)
- No deep supervision when using NoDeepSupervision trainer

---

### 3. Key Architectural Differences

| Aspect | DINOv3 M2F | nnUNet ViT Model |
|--------|------------|------------------|
| **Backbone source** | Custom DINOv3 ViT via `load_model_and_context()` | timm DINOv3 ViT |
| **Feature extraction** | DINOv3_Adapter with deformable attention | Direct ViT feature extraction |
| **Decoder type** | Query-based Mask2Former | Dense prediction UperNet |
| **Output format** | Dict: pred_logits, pred_masks, aux_outputs | Direct segmentation tensor |
| **Loss** | MaskClassificationLoss (Hungarian + mask/dice/CE) | DC_and_CE_loss |
| **Queries** | 100 learnable queries | N/A (dense prediction) |
| **Deep supervision** | 9 auxiliary outputs | None (NoDeepSupervision) |

---

### 4. OmegaConf Configuration System

**Config Dataclass**: `dinov3/eval/segmentation/config.py:138-158`

```python
@dataclass
class SegmentationConfig:
    model: ModelConfig | None = None       # Backbone config
    bs: int = 2                             # Batch size
    n_gpus: int = 8
    model_dtype: ModelDtype = ModelDtype.FLOAT32
    seed: int = 100
    datasets: DatasetConfig                 # root, train, val
    decoder_head: DecoderConfig             # type, num_classes, hidden_dim
    scheduler: SchedulerConfig              # type, total_iter
    optimizer: OptimizerConfig              # lr, weight_decay
    transforms: TransformConfig             # train/eval transforms
    m2f_train: M2FTrainConfig               # mask/dice/class coefficients
    eval: EvalConfig                        # mode, crop_size, stride
    lora: LoRAConfig                        # enabled, r, alpha
```

**Config Merge Strategy** (`run.py:71-82`):
```python
OmegaConf.merge(
    OmegaConf.structured(SegmentationConfig),  # 1. Dataclass defaults
    OmegaConf.load(base_config_path),           # 2. YAML file values
    OmegaConf.create(eval_args),                # 3. CLI overrides
)
```

**Key M2F Training Hyperparameters** (`M2FTrainConfig`):
```yaml
m2f_train:
  num_points: 12544          # Points sampled for mask loss (112*112)
  oversample_ratio: 3.0
  importance_sample_ratio: 0.75
  mask_coefficient: 5.0
  dice_coefficient: 5.0
  class_coefficient: 2.0
  no_object_coefficient: 0.1
```

---

### 5. Training Loop Comparison

**DINOv3 M2F** (`train_m2f.py:370-396`):
```python
for batch in train_dataloader:
    loss = train_step_m2f(...)
    # Loss = Hungarian matching + mask BCE + dice + class CE
    # aux_outputs provide deep supervision
```

**nnUNet** (`nnUNetTrainer_mymodel.py:1785-1806`):
```python
for epoch in range(self.num_epochs):
    for batch_id in range(self.num_iterations_per_epoch):
        train_outputs.append(self.train_step(next(self.dataloader_train)))
    # Uses iteration-based training with epochs
```

---

### 6. Data Augmentation Differences

**DINOv3 M2F** uses `make_segmentation_train_transforms()`:
- Random resize ratio range
- Random crop
- Horizontal flip
- SemanticToM2FTargets (converts labels to per-class binary masks)

**nnUNet** uses `get_training_transforms()`:
- Spatial transforms (rotation, scaling)
- Gaussian noise/blur
- Brightness/contrast
- Gamma transform
- Mirror (horizontal/vertical/depth flip)

---

### 7. Integration Points for Replacement

To replace the nnUNet model with DINOv3 M2F, the following components need modification:

#### 7.1 Model Creation (`nnUNetTrainer_mymodel.py:294-299`)

Current:
```python
from nnunetv2.model.model import create_segmentation_model
self.network = create_segmentation_model(
    num_classes=self.label_manager.num_segmentation_heads,
    freeze_backbone=True,
).to(self.device)
```

Integration point: Replace with `build_segmentation_decoder()` from DINOv3.

#### 7.2 Loss Function (`nnUNetTrainer_mymodel.py:523-575`)

Current: `DC_and_CE_loss`

Integration point: Replace with `MaskClassificationLoss` and handle auxiliary outputs.

#### 7.3 Training Step (`nnUNetTrainer_mymodel.py:1264-1298`)

Current: Direct forward pass returning tensor.

Integration point: Handle dict output format with `pred_logits`, `pred_masks`, `aux_outputs`.

#### 7.4 Configuration Injection

**Proposed OmegaConf structure for nnUNet integration**:

```yaml
# nnunet_m2f_config.yaml
model:
  type: "m2f"
  backbone: "dinov3"
  backbone_size: "large"
  hidden_dim: 2048
  backbone_out_layers: "FOUR_EVEN_INTERVALS"

lora:
  enabled: true
  r: 8
  lora_alpha: 16
  target_modules: ["qkv", "proj"]

m2f_train:
  num_points: 12544
  mask_coefficient: 5.0
  dice_coefficient: 5.0
  class_coefficient: 2.0
  no_object_coefficient: 0.1

training:
  optimizer: "adamw"
  lr: 1e-4
  weight_decay: 0.05
  scheduler: "WarmupOneCycleLR"
  total_iter: 40000
```

---

## Code References

| File | Line | Description |
|------|------|-------------|
| `dinov3/eval/segmentation/run.py` | 52-66 | Main entry for M2F training |
| `dinov3/eval/segmentation/train_m2f.py` | 227-480 | M2F training loop |
| `dinov3/eval/segmentation/config.py` | 138-158 | SegmentationConfig dataclass |
| `dinov3/eval/segmentation/models/__init__.py` | 76-148 | build_segmentation_decoder() |
| `dinov3/eval/segmentation/models/heads/mask2former_head.py` | 16-97 | Mask2FormerHead class |
| `dinov3/eval/segmentation/models/backbone/dinov3_adapter.py` | 305-489 | DINOv3_Adapter class |
| `nnunetv2/training/nnUNetTrainer/nnUNetTrainer_mymodel.py` | 108-1807 | Custom nnUNet trainer |
| `nnunetv2/model/model.py` | 226-642 | ViTSegmentationModel |
| `nnunetv2/run/run_training_nodeepsupervision_mymodel.py` | 198-420 | Training entry point |

---

## Architecture Diagrams

### DINOv3 Mask2Former Pipeline
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DINOv3 Mask2Former                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input [B,3,H,W]                                                             │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │ DINOv3_Adapter                                                   │        │
│  │  ├── SpatialPriorModule (conv stem → multi-scale CNN)            │        │
│  │  ├── ViT backbone → features at layers [4,11,17,23]              │        │
│  │  ├── InteractionBlocks (deformable attention fusion)             │        │
│  │  └── Output: {"1": 1/4, "2": 1/8, "3": 1/16, "4": 1/32}          │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │ MSDeformAttnPixelDecoder (6 transformer enc layers)              │        │
│  │  └── Output: mask_features + 3 multi-scale features              │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │ MultiScaleMaskedTransformerDecoder (9 layers)                    │        │
│  │  ├── 100 learnable queries                                       │        │
│  │  ├── Cross-attention with multi-scale features                   │        │
│  │  ├── Self-attention among queries                                │        │
│  │  └── Class/mask prediction heads                                 │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│       │                                                                      │
│       ▼                                                                      │
│  Output: {pred_logits: [B,100,C+1], pred_masks: [B,100,H/4,W/4],            │
│           aux_outputs: [9 intermediate predictions]}                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Current nnUNet ViT Pipeline
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           nnUNet ViT Model                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input [B,C,H,W]                                                             │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │ timm ViT backbone (DINOv3 large)                                 │        │
│  │  └── Extract features at layers [5,11,17,23]                     │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │ UperNetHead (HuggingFace transformers)                           │        │
│  │  └── PPM (Pyramid Pooling Module) + FPN                          │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │ ProgressiveUpsampleDecoder (4 stages, 2x each)                   │        │
│  │  └── Skip connections from backbone features                     │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│       │                                                                      │
│       ▼                                                                      │
│  Output: [B, num_classes, H, W]                                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Open Questions

1. **Data format compatibility**: nnUNet uses medical imaging formats (NIfTI) while DINOv3 M2F uses standard image datasets. Need to adapt data loading.

2. **Patch-based training**: nnUNet trains on patches; M2F typically trains on full/cropped images. Need to decide on training strategy.

3. **Evaluation protocol**: nnUNet uses sliding window inference; M2F supports both "slide" and "whole" modes.

4. **Input normalization**: nnUNet uses per-dataset normalization; M2F uses ImageNet normalization by default.

5. **Number of classes**: nnUNet dynamically gets num_classes from dataset.json; M2F expects it in config.

---

## Related Research

- No existing research documents found in thoughts/shared/research/

---

## Appendix: Example OmegaConf Integration Pattern

```python
# Proposed: nnunetv2/training/nnUNetTrainer/nnUNetTrainer_m2f.py

from omegaconf import OmegaConf, MISSING
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class M2FModelConfig:
    type: str = "m2f"
    backbone: str = "dinov3"
    backbone_size: str = "large"
    hidden_dim: int = 2048
    backbone_out_layers: str = "FOUR_EVEN_INTERVALS"

@dataclass
class M2FLoRAConfig:
    enabled: bool = True
    r: int = 8
    lora_alpha: int = 16
    target_modules: List[str] = field(default_factory=lambda: ["qkv", "proj"])
    lora_dropout: float = 0.1

@dataclass
class M2FTrainConfig:
    num_points: int = 12544
    mask_coefficient: float = 5.0
    dice_coefficient: float = 5.0
    class_coefficient: float = 2.0
    no_object_coefficient: float = 0.1

@dataclass
class M2FnnUNetConfig:
    model: M2FModelConfig = field(default_factory=M2FModelConfig)
    lora: M2FLoRAConfig = field(default_factory=M2FLoRAConfig)
    m2f_train: M2FTrainConfig = field(default_factory=M2FTrainConfig)

    # nnUNet-specific overrides
    use_nnunet_augmentation: bool = True
    use_sliding_window_inference: bool = True

# Usage in trainer
class nnUNetTrainer_m2f(nnUNetTrainer_mymodel):
    def __init__(self, ..., config_path: Optional[str] = None):
        super().__init__(...)

        # Load config with OmegaConf
        if config_path:
            yaml_config = OmegaConf.load(config_path)
        else:
            yaml_config = OmegaConf.create({})

        self.m2f_config = OmegaConf.to_object(
            OmegaConf.merge(
                OmegaConf.structured(M2FnnUNetConfig),
                yaml_config,
            )
        )
```
