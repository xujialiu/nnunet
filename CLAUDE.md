# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nnU-Net v2 fork with Vision Transformer integration for medical image segmentation. Adds ViT backbones (DINOv3, DINOv2, RetFound, VisionFM), multiple decoders (UperNet, SegFormer, Primus), LoRA fine-tuning, and YAML configuration.

## Environment Setup

```bash
conda activate dinov3

export nnUNet_raw="path/to/nnUNet_raw"
export nnUNet_preprocessed="path/to/nnUNet_preprocessed"
export nnUNet_results="path/to/nnUNet_results"
export nnUNet_n_proc_DA=12  # Optional: 12 for RTX 3090, 16-18 for RTX 4090
```

## Essential Commands

```bash
# Standard nnU-Net
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
nnUNetv2_train DATASET_ID CONFIGURATION FOLD  # CONFIGURATION: 2d, 3d_fullres, 3d_lowres
nnUNetv2_predict -i INPUT -o OUTPUT -d DATASET_ID -c CONFIGURATION -f FOLD

# Custom ViT training (recommended)
python nnUNetv2_train_vit.py DATASET_ID 2d FOLD --config nnunetv2/configs/segformer.yaml
# Configs: segformer.yaml, primus.yaml, primus_multiscale.yaml

# Legacy ViT training
python nnUNetv2_train_nodeepsupervision_mymodel.py DATASET_ID 2d FOLD
```

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `nnunetv2/model/` | ViT models: `model.py` (UperNet), `model_segformer_*.py`, `primus.py` |
| `nnunetv2/configs/` | YAML configs for ViT training |
| `nnunetv2/training/nnUNetTrainer/` | Base trainer + variants in `variants/` |
| `nnunetv2/run/` | Training entry points |
| `nnunetv2/inference/` | Sliding window prediction |
| `dinov3/` | DINOv3 framework (self-supervised learning, Mask2Former) |

## ViT Model Architecture

```
ViT Backbone → Multi-level features → [UperNet|SegFormer|Primus] Decoder → Segmentation
```

**Backbones**: DINOv3 (patch16), DINOv2 (patch14, auto-padded), RetFound, VisionFM

**Models**: `ViTSegmentationModel`, `SegFormerSegmentationModel`, `PrimusSegmentationModel`, `PrimusMultiscaleSegmentationModel`

**Trainers**: `nnUNetTrainer_vit` (recommended), `nnUNetTrainer_mymodel`, `nnUNetTrainer_m2f`

## Configuration

### ViT Configs (`nnunetv2/configs/*.yaml`)
```yaml
model:
  model_path: "nnunetv2.model.model_segformer_1"
  backbone: "dinov3"        # dinov3, dinov2, retfound, visionfm
  backbone_size: "large"    # base, large
lora:
  enabled: true
  r: 8
train:
  initial_lr: 0.001
  num_epochs: 500
```

### nnU-Net Plans (`nnUNet_preprocessed/DatasetXXX/`)
- `nnUNetPlans.json` - Auto-generated network config
- `dataset.json` - Dataset metadata
- `splits_final.json` - 5-fold CV splits

## Model API

```python
from nnunetv2.model.model import create_segmentation_model

model = create_segmentation_model(
    model_name="dinov3", model_size="large", num_classes=4,
    use_lora=True, lora_r=16
)
model.print_trainable_parameters()  # Inspect params
```

## Trainer Extension

Override: `build_network_architecture()`, `_build_loss()`, `configure_optimizers()`, `get_training_transforms()`

## Tech Stack

Python 3.9+, PyTorch, timm, transformers, peft, omegaconf, dynamic_network_architectures, batchgenerators, nibabel, SimpleITK

## Documentation

- [docs/architecture.md](docs/architecture.md) - Design patterns, full trainer hierarchy
- [docs/changelog.md](docs/changelog.md) - Project changes
- [documentation/](documentation/) - Official nnU-Net docs
