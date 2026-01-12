# Changelog

All notable changes to this nnU-Net fork will be documented in this file.

## [Unreleased]

### Added
- Custom Vision Transformer model integration (`nnunetv2/model/model.py`)
  - DINOv2, DINOv3, RetFound pretrained backbone support
  - UperNet decoder head for segmentation
- Custom trainer variants:
  - `nnUNetTrainer_mymodel` - ViT-based training
  - `nnUNetTrainerNoDeepSupervision_mymodel` - ViT without deep supervision
- Training entry point scripts:
  - `nnUNetv2_train.py` - Standard training wrapper
  - `nnUNetv2_train_nodeepsupervision.py` - No deep supervision variant
  - `nnUNetv2_train_nodeepsupervision_mymodel.py` - Custom model variant
- Utility scripts:
  - `convert_masks.py` - Mask format conversion utility
  - `debugger.py` - Shell script parser for debugging
- Sample configuration files in `sample/` directory
- Retina vessel segmentation dataset (`Dataset000_semantic_retina_vessel_segmentation`)

### Changed
- Project configured for local development with conda environment `nnunet`

### Notes
- Based on nnU-Net v2 framework
- See `documentation/changelog.md` for upstream nnU-Net v2 changes from v1
