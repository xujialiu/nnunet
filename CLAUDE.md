# nnU-Net v2

Automated medical image segmentation framework. Self-configuring U-Net pipeline that extracts dataset properties and designs optimal configurations without manual intervention.

## Tech Stack
- **Python 3.9+** with **PyTorch** (CUDA/MPS/CPU)
- **dynamic_network_architectures** - U-Net implementations
- **batchgenerators** v1/v2 - Medical imaging augmentation
- **nibabel**, **SimpleITK** - Medical image I/O

## Key Directories
| Directory | Purpose |
|-----------|---------|
| `nnunetv2/training/nnUNetTrainer/` | Core trainer class and 17+ variants |
| `nnunetv2/experiment_planning/` | Auto configuration planning |
| `nnunetv2/preprocessing/` | Resampling, normalization, cropping |
| `nnunetv2/inference/` | Sliding window prediction |
| `nnunetv2/model/` | Custom ViT model (project fork) |
| `nnUNet_raw/`, `nnUNet_preprocessed/`, `nnUNet_results/` | Data directories |

## Essential Commands
```bash
conda activate nnunet  # Always use this env

# Full pipeline
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
nnUNetv2_train DATASET_ID CONFIGURATION FOLD  # e.g., nnUNetv2_train 0 2d 0

# Local scripts: nnUNetv2_train.py, nnUNetv2_train_nodeepsupervision.py, nnUNetv2_train_nodeepsupervision_mymodel.py
nnUNetv2_predict -i INPUT -o OUTPUT -d DATASET_ID -c CONFIGURATION
```

## Environment Variables
```bash
export nnUNet_raw="path/to/nnUNet_raw"
export nnUNet_preprocessed="path/to/nnUNet_preprocessed"
export nnUNet_results="path/to/nnUNet_results"
```

## Additional Documentation
- [Architecture & Patterns](docs/architecture.md) - Design decisions, trainer hierarchy, configuration system
- [Changelog](docs/changelog.md) - Project changes and modifications
- [Official Docs](documentation/) - `how_to_use_nnunet.md`, `dataset_format.md`, `extending_nnunet.md`
