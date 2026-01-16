# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# nnU-Net v2

Automated medical image segmentation framework. Self-configuring U-Net pipeline that extracts dataset properties and designs optimal configurations without manual intervention.

## Tech Stack
- **Python 3.9+** with **PyTorch** (CUDA/MPS/CPU)
- **dynamic_network_architectures** - U-Net implementations
- **batchgenerators** v1/v2 - Medical imaging augmentation
- **nibabel**, **SimpleITK** - Medical image I/O
- **timm**, **transformers**, **peft** - ViT backbones, UperNet decoder, LoRA fine-tuning

## Key Directories
| Directory | Purpose |
|-----------|---------|
| `nnunetv2/training/nnUNetTrainer/` | Core trainer class and 17+ variants |
| `nnunetv2/experiment_planning/` | Auto configuration planning |
| `nnunetv2/preprocessing/` | Resampling, normalization, cropping |
| `nnunetv2/inference/` | Sliding window prediction |
| `nnunetv2/model/` | Custom ViT model (project fork) |
| `nnUNet_raw/`, `nnUNet_preprocessed/`, `nnUNet_results/` | Data directories |

## Custom ViT Model (Project Fork)
`nnunetv2/model/model.py` contains `ViTSegmentationModel` with:
- **Backbones**: DINOv2, DINOv3, RetFound, VisionFM (via timm)
- **Decoder**: UperNetHead (multi-scale) + ProgressiveUpsampleDecoder
- **LoRA**: Low-rank adaptation for efficient fine-tuning (`use_lora=True`)
- **Trainers**: `nnUNetTrainer_mymodel`, `nnUNetTrainerNoDeepSupervision_mymodel`

## Essential Commands
```bash
conda activate nnunet  # Always use this env
export nnUNet_raw="path/to/nnUNet_raw"
export nnUNet_preprocessed="path/to/nnUNet_preprocessed"
export nnUNet_results="path/to/nnUNet_results"

# Full pipeline
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
nnUNetv2_train DATASET_ID CONFIGURATION FOLD  # e.g., nnUNetv2_train 0 2d 0
python nnUNetv2_train_nodeepsupervision_mymodel.py  # Custom ViT training
nnUNetv2_predict -i INPUT -o OUTPUT -d DATASET_ID -c CONFIGURATION
```

## Additional Documentation
- [Architecture & Patterns](docs/architecture.md) - Design decisions, trainer hierarchy, configuration system
- [Changelog](docs/changelog.md) - Project changes and modifications
- [Official Docs](documentation/) - `how_to_use_nnunet.md`, `dataset_format.md`, `extending_nnunet.md`
