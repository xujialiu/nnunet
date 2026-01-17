# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nnU-Net v2 fork with custom Vision Transformer integration for medical image segmentation. The framework automatically configures U-Net pipelines from dataset properties while this fork adds ViT backbone support with LoRA fine-tuning.

## Environment Setup

```bash
conda activate dinov3  # Required for all operations

# Required environment variables
export nnUNet_raw="path/to/nnUNet_raw"
export nnUNet_preprocessed="path/to/nnUNet_preprocessed"
export nnUNet_results="path/to/nnUNet_results"

# Optional: Adjust workers for your CPU/GPU ratio
export nnUNet_n_proc_DA=12  # 12 for RTX 3090, 16-18 for RTX 4090
```

**Hardware**: GPU with 10GB+ VRAM for training (RTX 3080/3090/4090 or A100), 6+ CPU cores minimum.

## Essential Commands

### Standard nnU-Net Pipeline
```bash
# Preprocessing
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity

# Training (standard U-Net)
nnUNetv2_train DATASET_ID CONFIGURATION FOLD  # CONFIGURATION: 2d, 3d_fullres, 3d_lowres

# Prediction
nnUNetv2_predict -i INPUT -o OUTPUT -d DATASET_ID -c CONFIGURATION -f FOLD
```

### Custom ViT Model Training
```bash
# No deep supervision variant (recommended for ViT)
python nnUNetv2_train_nodeepsupervision_mymodel.py DATASET_ID 2d FOLD

# These scripts are wrappers calling nnunetv2/run/ modules
```

### Running Tests
```bash
# Prepare dummy test datasets (996-999)
bash nnunetv2/tests/integration_tests/prepare_integration_tests.sh

# Run integration test
bash nnunetv2/tests/integration_tests/run_integration_test.sh DATASET_ID

# DDP test (requires 2 GPUs)
bash nnunetv2/tests/integration_tests/run_integration_test_trainingOnly_DDP.sh DATASET_ID

# Cleanup
python nnunetv2/tests/integration_tests/cleanup_integration_test.py
```

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `nnunetv2/model/` | Custom ViT segmentation model (project addition) |
| `nnunetv2/training/nnUNetTrainer/` | Base trainer + 30+ variants in `variants/` |
| `nnunetv2/run/` | Training entry point modules |
| `nnunetv2/inference/` | Sliding window prediction |
| `nnunetv2/experiment_planning/` | Auto configuration from dataset fingerprint |
| `dinov3/` | DINOv3 self-supervised learning framework (recently added) |
| `dinov3/eval/segmentation/` | Mask2Former decoder training |

## Custom ViT Model (`nnunetv2/model/model.py`)

### Architecture
```
ViT Backbone (DINOv3/DINOv2/RetFound/VisionFM)
    │
    ├─ Multi-level features from transformer blocks
    ▼
UperNetHead (multi-scale aggregation via ASPP)
    │
    ▼
ProgressiveUpsampleDecoder (4-stage 2x upsampling with skip connections)
    │
    ▼
Final segmentation output
```

### Key Functions
```python
from nnunetv2.model.model import create_segmentation_model, create_backbone

# Create model
model = create_segmentation_model(
    model_name="dinov3",           # or "dinov2", "retfound", "visionfm"
    model_size="base",             # or "small", "large", "giant"
    num_classes=4,
    use_lora=True,                 # Enable LoRA fine-tuning
    lora_r=16, lora_alpha=32
)

# Inspect parameters
model.print_trainable_parameters(detailed=True)  # By component
model.print_trainable_layers(show_shapes=True)   # Layer-by-layer

# LoRA control
model.enable_lora()   # Apply LoRA adapters
model.disable_lora()  # Merge and remove adapters
```

### Custom Trainers
- `nnUNetTrainer_mymodel` - ViT with deep supervision
- `nnUNetTrainerNoDeepSupervision_mymodel` - ViT without auxiliary losses (recommended)

## Trainer Hierarchy

Base `nnUNetTrainer` with variants in `nnunetv2/training/nnUNetTrainer/variants/`:

| Category | Variants |
|----------|----------|
| `network_architecture/` | NoDeepSupervision, NoDeepSupervision_mymodel, BN |
| `data_augmentation/` | NoDA, DAOrd0, DA5, NoMirroring, noDummy2DDA |
| `loss/` | CELoss, DiceLoss, TopkLoss |
| `optimizer/`, `lr_schedule/` | SGD, learning rate variants |
| `training_length/` | Xepochs variants |

**Extension pattern**: Override methods like `build_network_architecture()`, `_build_loss()`, `configure_optimizers()`, `get_training_transforms()`.

## Configuration System

Plans files in `nnUNet_preprocessed/DatasetXXX/`:
- `nnUNetPlans.json` - Network architecture, patch/batch sizes (auto-generated)
- `dataset.json` - Dataset metadata and channels
- `splits_final.json` - 5-fold CV splits
- `dataset_fingerprint.json` - Auto-computed dataset properties

Sample configs in `sample/` directory.

## DINOv3 Integration (`dinov3/`)

Recently added self-supervised learning framework with evaluation modules:
- `dinov3/eval/segmentation/train_m2f.py` - Mask2Former decoder training
- `dinov3/eval/segmentation/configs/` - YAML configs for segmentation tasks
- `dinov3/models/` - ViT implementations
- `dinov3/eval/depth/`, `dinov3/eval/detection/` - Other evaluation tasks

## Utility Scripts

- `convert_masks.py` - Colorize segmentation masks for visualization

## Tech Stack

- **Python 3.9+**, **PyTorch** (CUDA/MPS/CPU)
- **timm**, **transformers**, **peft** - ViT backbones, UperNet, LoRA
- **dynamic_network_architectures** - U-Net implementations
- **batchgenerators** v1/v2 - Medical imaging augmentation
- **nibabel**, **SimpleITK** - Medical image I/O

## Additional Documentation

- [docs/architecture.md](docs/architecture.md) - Design patterns, trainer hierarchy
- [docs/changelog.md](docs/changelog.md) - Project changes
- [documentation/](documentation/) - Official nnU-Net docs (dataset format, extending, etc.)
