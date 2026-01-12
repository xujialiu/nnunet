# Architecture & Design Patterns

## Multi-Stage Automatic Pipeline

nnU-Net follows a systematic, data-driven approach:

```
Raw Data → Fingerprinting → Planning → Preprocessing → Training → Evaluation → Inference
```

Each stage is decoupled and produces JSON artifacts that drive subsequent stages.

## Trainer Class Hierarchy

Extensible inheritance pattern for customization:

```
nnUNetTrainer (base)
├── nnUNetTrainerNoDeepSupervision
├── nnUNetTrainerBN (batch norm)
├── nnUNetTrainerCELoss
├── nnUNetTrainerDA5 (custom augmentation)
├── nnUNetTrainer_mymodel (custom ViT model)
└── ... (17+ variants)
```

**Extension pattern**: Override specific methods to modify behavior:
- `build_network_architecture()` - Custom network
- `_build_loss()` - Custom loss function
- `configure_optimizers()` - Custom optimizer
- `get_training_transforms()` - Custom augmentation

## Configuration-Driven Architecture

Network architecture defined in JSON plans files, not hardcoded:

```json
{
  "configurations": {
    "2d": {
      "patch_size": [512, 512],
      "batch_size": 12,
      "network_arch_class_name": "PlainConvUNet",
      "conv_kernel_sizes": [[3,3], [3,3], ...],
      "pool_op_kernel_sizes": [[2,2], [2,2], ...]
    }
  }
}
```

Plans location: `nnUNet_preprocessed/DatasetXXX/nnUNetPlans.json`

## Loss Function Composition

Compound losses combining Dice and Cross-Entropy:

```python
# nnunetv2/training/loss/compound_losses.py
DC_and_CE_loss = weight_dice * DiceLoss + weight_ce * CrossEntropyLoss
```

Deep supervision wrapper applies loss to intermediate decoder outputs.

## Data Augmentation Pipeline

Declarative transform composition using batchgenerators v2:

```python
ComposeTransforms([
    SpatialTransform(...),       # Rotation, scaling, elastic
    MirrorTransform(...),        # Spatial mirroring
    GaussianNoiseTransform(...), # Noise injection
    GammaTransform(...),         # Intensity gamma
    DownsampleSegForDSTransform(...),  # Deep supervision targets
])
```

## Image I/O Strategy Pattern

Format-agnostic I/O through abstract base class:

```
BaseReaderWriter (abstract)
├── NibabelIO (NIFTI/NRRD/MHA)
├── SimpleITKIO
├── NaturalImageIO (PNG/BMP/TIF)
└── Tiff3DIO
```

Auto-detection via `determine_reader_writer_from_dataset_json()`.

## Dataset Organization Convention

Datasets identified by 3-digit ID + name:
- `Dataset000_semantic_retina_vessel_segmentation`
- `Dataset001_BrainTumour`

Channel suffix: `{CASE_ID}_{CHANNEL_ID}.{FORMAT}` (e.g., `case_0001_0000.nii.gz`)

## Sliding Window Inference

- Tile input with overlapping patches
- Gaussian weighting reduces boundary artifacts
- Test-time mirroring augmentation supported
- GPU memory-efficient computation

## Key Design Principles

1. **No Manual Configuration** - Automatic hyperparameter selection from data fingerprint
2. **Extensibility** - Trainer variants via inheritance, custom planners, custom architectures
3. **Reproducibility** - 5-fold CV, fixed seeds, deterministic augmentation
4. **Configuration as Data** - JSON plans files are human-readable and editable
5. **Modularity** - Clear separation: preprocessing, training, inference are independent

## Custom Model Integration (Project Fork)

This fork adds Vision Transformer integration:

- `nnunetv2/model/model.py` - ViT backbone with UperNet decoder
- Supports DINOv2, DINOv3, RetFound pretrained weights
- Custom trainers: `nnUNetTrainer_mymodel`, `nnUNetTrainerNoDeepSupervision_mymodel`
