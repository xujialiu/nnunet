# Plan: Refactor ViT Training Entry Point

## Overview

Transform `nnUNetv2_train_segformer.py` into a generalized ViT training entry point (`nnUNetv2_train_vit.py`) that supports multiple segmentation head architectures through config-driven model loading.

## Goals

1. Rename entry point from `nnUNetv2_train_segformer.py` to `nnUNetv2_train_vit.py`
2. Change `--segformer_config` CLI flag to `--config`
3. Introduce config-driven model loading via `model.model_path`
4. Extract shared backbone code into `nnunetv2/model/backbone.py`

## New Config Structure

```yaml
model:
  model_path: "nnunetv2.model.model_segformer_1"  # Python module path
  backbone: "dinov3"
  backbone_size: "large"
  checkpoint_path: ""
  decoder:
    hidden_size: 256

lora:
  enabled: true
  r: 8
  lora_alpha: 16
  target_modules: ["qkv"]
  lora_dropout: 0.05

train:
  initial_lr: 0.001
  weight_decay: 0.00003
  gradient_clip: 12.0
```

## Files to Modify

### 1. Rename Entry Point
- **From**: `nnUNetv2_train_segformer.py`
- **To**: `nnUNetv2_train_vit.py`
- Change import from `run_training_segformer` to `run_training_vit`

### 2. Rename Run Module
- **From**: `nnunetv2/run/run_training_segformer.py`
- **To**: `nnunetv2/run/run_training_vit.py`
- Change `--segformer_config` to `--config`
- Change `segformer_config_path` parameter names to `config_path`
- Update default trainer name from `nnUNetTrainerNoDeepSupervision_segformer` to `nnUNetTrainerNoDeepSupervision_vit`

### 3. Create `nnunetv2/model/backbone.py`
Extract from `model_segformer_1.py`:
- `create_backbone()` function
- `get_vit_features()` function
- `model_configs` dictionary

Contents:
```python
"""ViT backbone creation and feature extraction utilities."""

from typing import Tuple, Optional, List, Union
import torch
import timm

MODEL_CONFIGS = {
    "dinov3": {
        "timm_name": "vit_{size}_patch16_dinov3.lvd1689m",
        "patch_size": 16,
        "pretrained": True,
    },
    "dinov2": {
        "timm_name": "vit_{size}_patch14_dinov2.lvd142m",
        "patch_size": 14,
        "pretrained": True,
    },
    "retfound": {
        "timm_name": "vit_large_patch16_224.mae",
        "patch_size": 16,
        "pretrained": False,
    },
    "visionfm": {
        "timm_name": "vit_base_patch16_224.mae",
        "patch_size": 16,
        "pretrained": False,
    },
}

def create_backbone(
    model_name: str = "dinov3",
    model_size: str = "large",
    checkpoint_path: Optional[str] = None,
) -> Tuple[torch.nn.Module, int, str]:
    """Create ViT backbone from timm."""
    ...

def get_vit_features(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    indices: Optional[Union[int, List[int]]] = None,
) -> List[torch.Tensor]:
    """Extract intermediate features from ViT backbone."""
    ...
```

### 4. Refactor Config System
- **Rename**: `segformer_config.py` → `vit_config.py`
- **Rename classes**:
  - `SegFormerModelConfig` → `ViTModelConfig`
  - `SegFormerLoRAConfig` → `ViTLoRAConfig`
  - `SegFormerTrainConfig` → `ViTTrainConfig`
  - `SegFormernnUNetConfig` → `ViTnnUNetConfig`
- **Add new field**: `model_path: str = "nnunetv2.model.model_segformer_1"` in model config
- **Restructure decoder config**:
  ```python
  @dataclass
  class DecoderConfig:
      hidden_size: int = 256

  @dataclass
  class ViTModelConfig:
      model_path: str = "nnunetv2.model.model_segformer_1"
      backbone: str = "dinov3"
      backbone_size: str = "large"
      checkpoint_path: str = ""
      decoder: DecoderConfig = field(default_factory=DecoderConfig)
  ```

### 5. Refactor Trainer
- **Rename**: `nnUNetTrainer_segformer.py` → `nnUNetTrainer_vit.py`
- **Rename class**: `nnUNetTrainer_segformer` → `nnUNetTrainer_vit`
- **Update `initialize()` method** to use dynamic model loading:
  ```python
  def initialize(self):
      # Dynamic import based on config
      model_module = importlib.import_module(cfg.model.model_path)
      create_model_fn = getattr(model_module, "create_segmentation_model")

      self.network = create_model_fn(
          backbone_name=cfg.model.backbone,
          backbone_size=cfg.model.backbone_size,
          num_classes=self.label_manager.num_segmentation_heads,
          decoder_hidden_size=cfg.model.decoder.hidden_size,
          ...
      )
  ```

### 6. Rename NoDeepSupervision Variant
- **Rename**: `nnUNetTrainerNoDeepSupervision_segformer.py` → `nnUNetTrainerNoDeepSupervision_vit.py`
- **Rename class**: `nnUNetTrainerNoDeepSupervision_segformer` → `nnUNetTrainerNoDeepSupervision_vit`
- Update import to use new trainer base class

### 7. Update `model_segformer_1.py`
- Replace `create_backbone()` and `get_vit_features()` with imports from `backbone.py`
- Keep `SegFormerSegmentationModel` and `create_segformer_model` (specific to this decoder)

### 8. Standardize Model Factory Interface
Each model module (e.g., `model_segformer_1.py`, `model_segformer_2.py`) should export:
```python
def create_segmentation_model(
    backbone_name: str,
    backbone_size: str,
    num_classes: int,
    decoder_hidden_size: int,
    checkpoint_path: Optional[str] = None,
    freeze_backbone: bool = False,
    use_lora: bool = True,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[List[str]] = None,
) -> nn.Module:
    ...
```

## Execution Order

1. **Create `backbone.py`** - Extract shared code first
2. **Update `model_segformer_1.py`** - Use imports from `backbone.py`
3. **Rename config system** - `segformer_config.py` → `vit_config.py` with new structure
4. **Rename trainer** - `nnUNetTrainer_segformer.py` → `nnUNetTrainer_vit.py`
5. **Rename trainer variant** - `nnUNetTrainerNoDeepSupervision_segformer.py` → `nnUNetTrainerNoDeepSupervision_vit.py`
6. **Rename run module** - `run_training_segformer.py` → `run_training_vit.py`
7. **Rename entry point** - `nnUNetv2_train_segformer.py` → `nnUNetv2_train_vit.py`

## Testing

After implementation:
```bash
python nnUNetv2_train_vit.py DATASET_ID 2d FOLD --config path/to/config.yaml
```

With config:
```yaml
model:
  model_path: "nnunetv2.model.model_segformer_1"
  backbone: "dinov3"
  backbone_size: "large"
  checkpoint_path: ""
  decoder:
    hidden_size: 256
```
