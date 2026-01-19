# Research: SegFormer Integration into nnUNet Pipeline

**Date**: 2026-01-18
**Git Commit**: 1dbf06defce6a78b45cf0b56bf5250430ac7b8b6
**Branch**: add_segformer
**Repository**: nnunet

## Research Question

How to integrate SegFormer model into nnUNet pipeline for comparable comparison with ViT model, including:
1. Completing model_segformer.py with SegFormer head and factory function
2. Using OmegaConf for parameter injection
3. Writing training script like nnUNetv2_train.py
4. Making minimal changes to nnUNet pipeline

---

## Summary

The nnUNet framework has an established pattern for integrating custom models through trainer subclasses. The existing `nnUNetTrainer_mymodel` and `nnUNetTrainer_m2f` implementations demonstrate two approaches: direct model creation in `initialize()` and OmegaConf-based configuration. SegFormer integration can follow the M2F pattern for full configurability.

---

## Detailed Findings

### 1. Current State of `model_segformer.py`

**Location**: `nnunetv2/model/model_segformer.py`

The file currently contains only backbone-related code copied from `model.py`:
- `create_backbone()` - Creates timm ViT backbones (dinov3, dinov2, retfound, visionfm)
- `get_vit_features()` - Extracts multi-level features from ViT

**Missing components**:
- SegFormer decode head (can use `transformers.SegformerDecodeHead`)
- Factory function `create_segformer_model()`
- SegFormer-specific model class

### 2. Transformers SegFormer Implementation

**Available in dinov3 conda environment**:
```python
from transformers.models.segformer.modeling_segformer import (
    SegformerDecodeHead,
    SegformerForSemanticSegmentation,
)
from transformers import SegformerConfig
```

**SegformerConfig Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_channels` | 3 | Input image channels |
| `num_encoder_blocks` | 4 | Number of encoder stages |
| `depths` | [2,2,2,2] | Layers per stage |
| `hidden_sizes` | [32,64,160,256] | Feature dims per stage |
| `sr_ratios` | [8,4,2,1] | Sequence reduction ratios |
| `decoder_hidden_size` | 256 | Unified decoder channel dim |
| `semantic_loss_ignore_index` | 255 | Ignore label |

**SegformerDecodeHead Architecture** (`modeling_segformer.py:598-653`):
```
4 multi-scale features from encoder
    │
    ▼
Per-stage MLP projection (hidden_sizes[i] → decoder_hidden_size)
    │
    ▼
Bilinear upsample all to Stage 1 resolution (H/4 × W/4)
    │
    ▼
Concatenate in reverse order → Conv2d(1024→256) → BN → ReLU → Dropout
    │
    ▼
Classifier: Conv2d(256→num_labels, 1×1)
    │
    ▼
Logits (B, num_labels, H/4, W/4)
```

### 3. nnUNet Trainer Architecture

**Key integration point**: `initialize()` method in trainer class

**Base pattern** (`nnUNetTrainer.py:283-326`):
```python
def initialize(self):
    self.num_input_channels = determine_num_input_channels(...)
    self.network = self.build_network_architecture(...).to(self.device)
    self.optimizer, self.lr_scheduler = self.configure_optimizers()
    self.loss = self._build_loss()
```

**Custom model pattern** (`nnUNetTrainer_mymodel.py:285-331`):
```python
def initialize(self):
    self.num_input_channels = determine_num_input_channels(...)

    from nnunetv2.model.model import create_segmentation_model
    self.network = create_segmentation_model(
        num_classes=self.label_manager.num_segmentation_heads,
        freeze_backbone=True,
    ).to(self.device)
```

**Dataset properties access**:
- `self.label_manager.num_segmentation_heads` → number of output classes
- `self.num_input_channels` → input channels from dataset
- `self.label_manager.ignore_label` → ignore label value

### 4. OmegaConf Configuration Pattern

**Pattern from M2F integration** (`m2f_config.py` + `nnUNetTrainer_m2f.py`):

**Step 1: Define dataclass configuration**:
```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class SegFormerModelConfig:
    backbone: str = "dinov3"
    backbone_size: str = "large"
    decoder_hidden_size: int = 256
    num_encoder_blocks: int = 4

@dataclass
class SegFormerLoRAConfig:
    enabled: bool = True
    r: int = 8
    lora_alpha: int = 16
    target_modules: List[str] = field(default_factory=lambda: ["qkv"])

@dataclass
class SegFormernnUNetConfig:
    model: SegFormerModelConfig = field(default_factory=SegFormerModelConfig)
    lora: SegFormerLoRAConfig = field(default_factory=SegFormerLoRAConfig)
```

**Step 2: Load config in trainer**:
```python
from omegaconf import OmegaConf

def _load_config(self, config_path: Optional[str]) -> SegFormernnUNetConfig:
    base_config = OmegaConf.structured(SegFormernnUNetConfig)

    if config_path and os.path.exists(config_path):
        yaml_config = OmegaConf.load(config_path)
        config = OmegaConf.merge(base_config, yaml_config)
    else:
        config = base_config

    return OmegaConf.to_object(config)
```

### 5. Training Script Pattern

**Entry point script** (`nnUNetv2_train_nodeepsupervision_mymodel.py`):
```python
from nnunetv2.run.run_training_nodeepsupervision_mymodel import run_training_entry

if __name__ == "__main__":
    sys.exit(run_training_entry())
```

**Run module** (`run_training_nodeepsupervision_mymodel.py`):
- `get_trainer_from_args()` - Loads trainer class dynamically
- `maybe_load_checkpoint()` - Handles checkpoint loading
- `run_training()` - Main training orchestration
- `run_training_entry()` - CLI argument parsing

**Key customizations**:
- Default trainer: `trainer_name: str = "nnUNetTrainerNoDeepSupervision_mymodel"`
- Import statement at top for specific trainer
- Optional extra CLI arguments (e.g., `--m2f_config` in M2F variant)

### 6. Trainer Variant Pattern

**Location**: `nnunetv2/training/nnUNetTrainer/variants/network_architecture/`

**Simple variant** (`nnUNetTrainerNoDeepSupervision_mymodel.py`):
```python
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_mymodel import nnUNetTrainer_mymodel

class nnUNetTrainerNoDeepSupervision_mymodel(nnUNetTrainer_mymodel):
    def __init__(self, plans, configuration, fold, dataset_json, device):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False
```

---

## Implementation Blueprint

### File Structure

```
nnunetv2/
├── model/
│   └── model_segformer.py              # SegFormer model + factory
├── training/nnUNetTrainer/
│   ├── nnUNetTrainer_segformer.py      # Main SegFormer trainer
│   ├── segformer_config.py             # OmegaConf dataclasses
│   └── variants/network_architecture/
│       └── nnUNetTrainerNoDeepSupervision_segformer.py
├── run/
│   └── run_training_segformer.py       # Run module
├── configs/
│   └── segformer_default.yaml          # Default config
└── (root)
    └── nnUNetv2_train_segformer.py     # Entry point
```

### Key Components

**1. `model_segformer.py`**:
```python
from transformers.models.segformer.modeling_segformer import SegformerDecodeHead
from transformers import SegformerConfig

class SegFormerSegmentationModel(nn.Module):
    def __init__(self, backbone_name, backbone_size, num_classes, ...):
        # Create timm backbone
        self.backbone, self.patch_size, _ = create_backbone(...)

        # Create SegFormer decode head
        hidden_sizes = [backbone_embed_dim] * 4  # Adapt from ViT
        config = SegformerConfig(
            hidden_sizes=hidden_sizes,
            num_labels=num_classes,
            decoder_hidden_size=256,
        )
        self.decode_head = SegformerDecodeHead(config)

    def forward(self, x):
        features = get_vit_features(self.backbone, x, indices=self.feature_indices)
        logits = self.decode_head(features)
        return F.interpolate(logits, size=x.shape[-2:], mode='bilinear')

def create_segformer_model(num_classes, backbone_name="dinov3", ...):
    return SegFormerSegmentationModel(...)
```

**2. `segformer_config.py`**:
```python
@dataclass
class SegFormerModelConfig:
    backbone: str = "dinov3"
    backbone_size: str = "large"
    decoder_hidden_size: int = 256

@dataclass
class SegFormerLoRAConfig:
    enabled: bool = True
    r: int = 8
    lora_alpha: int = 16

@dataclass
class SegFormernnUNetConfig:
    model: SegFormerModelConfig = field(default_factory=SegFormerModelConfig)
    lora: SegFormerLoRAConfig = field(default_factory=SegFormerLoRAConfig)
```

**3. `nnUNetTrainer_segformer.py`**:
```python
class nnUNetTrainer_segformer(nnUNetTrainer):
    def __init__(self, ..., segformer_config_path=None):
        self._segformer_config_path = segformer_config_path
        super().__init__(...)
        self.enable_deep_supervision = False
        self.my_init_kwargs["segformer_config_path"] = segformer_config_path
        self.segformer_config = self._load_config(segformer_config_path)

    def initialize(self):
        from nnunetv2.model.model_segformer import create_segformer_model
        self.network = create_segformer_model(
            num_classes=self.label_manager.num_segmentation_heads,
            **self.segformer_config.model.__dict__
        ).to(self.device)
```

---

## Code References

| File | Lines | Description |
|------|-------|-------------|
| `nnunetv2/model/model_segformer.py` | 1-86 | Current backbone code |
| `nnunetv2/model/model.py` | 226-355 | ViTSegmentationModel reference |
| `nnunetv2/model/model.py` | 588-642 | Factory function reference |
| `nnunetv2/training/nnUNetTrainer/nnUNetTrainer_mymodel.py` | 285-331 | initialize() pattern |
| `nnunetv2/training/nnUNetTrainer/m2f_config.py` | 1-50 | OmegaConf config pattern |
| `nnunetv2/training/nnUNetTrainer/nnUNetTrainer_m2f.py` | 103-121 | Config loading pattern |
| `nnunetv2/run/run_training_nodeepsupervision_mymodel.py` | 1-430 | Run module reference |
| `transformers/.../modeling_segformer.py` | 598-653 | SegformerDecodeHead |

---

## Considerations

### SegFormer Decode Head Compatibility

The `SegformerDecodeHead` expects multi-scale features with decreasing spatial resolution:
- Stage 1: H/4 × W/4
- Stage 2: H/8 × W/8
- Stage 3: H/16 × W/16
- Stage 4: H/32 × W/32

ViT backbones produce single-scale features. Options:
1. **Use intermediate layers** - Extract features at different transformer block depths
2. **Tile/reshape** - Apply learned downsampling to create multi-scale features
3. **Skip connections** - Use the existing `ProgressiveUpsampleDecoder` pattern

### LoRA Integration

The existing `ViTSegmentationModel.enable_lora()` method can be reused for SegFormer by wrapping the backbone with `peft.get_peft_model()`.

### Loss Function

For fair comparison with ViT model:
- Use same `DC_and_CE_loss` from nnUNet
- SegFormer outputs at H/4 resolution, needs upsampling before loss computation
