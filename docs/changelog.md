# Changelog

All notable changes to this nnU-Net fork will be documented in this file.

## [2025-01-23]

### Added
- **YAML Configuration System** (`nnunetv2/configs/`)
  - Config-driven model and training setup
  - Configs for SegFormer, Primus, and Primus Multiscale architectures
  - Test configs for all backbone variants (DINOv3, DINOv2, RetFound, VisionFM)
- **SegFormer Model** (`nnunetv2/model/model_segformer_*.py`)
  - Lightweight MLP-based decoder from SegFormer paper
  - Multiple variants (0, 1, 2) with different configurations
- **Primus Model** (`nnunetv2/model/primus.py`)
  - Simple patch decoder from [Primus paper](https://arxiv.org/pdf/2503.01835)
  - `PrimusSegmentationModel` - single layer features
  - `PrimusMultiscaleSegmentationModel` - multi-layer feature aggregation
- **nnUNetTrainer_vit** - New config-driven ViT trainer
  - Dynamic model loading via Python module paths
  - Reads architecture from YAML config files
- **Backbone utilities** (`nnunetv2/model/backbone.py`)
  - Centralized backbone creation and feature extraction
  - `get_vit_features()` for multi-level feature extraction
- Training entry point: `nnUNetv2_train_vit.py`

### Fixed
- **DINOv2 patch size compatibility** - Input padding for patch_size=14 models
  - DINOv2 uses patch_size=14, others use 16
  - Automatic padding/cropping to handle dimension mismatches

### Changed
- Refactored backbone loading into `backbone.py` module
- Updated vit_config.py with comprehensive configuration dataclasses

## [Unreleased - Initial Fork]

### Added
- Custom Vision Transformer model integration (`nnunetv2/model/model.py`)
  - DINOv2, DINOv3, RetFound, VisionFM pretrained backbone support
  - UperNet decoder head for multi-scale aggregation
  - ProgressiveUpsampleDecoder with skip connections
  - LoRA fine-tuning support via PEFT library
  - Dropout regularization throughout decoder
  - Parameter inspection utilities (`print_trainable_parameters`, `print_trainable_layers`)
- DINOv3 framework integration (`dinov3/`)
  - Self-supervised learning infrastructure
  - Mask2Former decoder training (`dinov3/eval/segmentation/`)
  - Detection and depth evaluation modules
- Custom trainer variants:
  - `nnUNetTrainer_mymodel` - ViT-based training
  - `nnUNetTrainerNoDeepSupervision_mymodel` - ViT without deep supervision
  - `nnUNetTrainer_m2f` - Mask2Former decoder trainer
- Training entry point scripts:
  - `nnUNetv2_train.py` - Standard training wrapper
  - `nnUNetv2_train_nodeepsupervision.py` - No deep supervision variant
  - `nnUNetv2_train_nodeepsupervision_mymodel.py` - Custom model variant
  - `nnUNetv2_train_m2f.py` - Mask2Former training
- Utility scripts:
  - `convert_masks.py` - Mask format conversion utility
  - `debugger.py` - Shell script parser for debugging
- Sample configuration files in `sample/` directory
- Retina vessel segmentation datasets (Dataset000-002)

### Changed
- Project configured for local development with conda environment `dinov3`

### Notes
- Based on nnU-Net v2 framework
- See `documentation/changelog.md` for upstream nnU-Net v2 changes from v1
