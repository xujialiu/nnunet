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
├── nnUNetTrainer_vit (config-driven ViT - recommended)
├── nnUNetTrainer_mymodel (legacy ViT model)
├── nnUNetTrainer_m2f (Mask2Former decoder)
├── nnUNetTrainerNoDeepSupervision
├── nnUNetTrainerBN (batch norm)
├── nnUNetTrainerCELoss
├── nnUNetTrainerDA5 (custom augmentation)
└── ... (30+ variants in variants/)
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

This fork adds Vision Transformer integration via `nnunetv2/model/`:

### Model Architectures

| Model | File | Decoder | Use Case |
|-------|------|---------|----------|
| `ViTSegmentationModel` | `model.py` | UperNet | Multi-scale with ASPP |
| `SegFormerSegmentationModel` | `model_segformer_*.py` | SegFormer MLP | Lightweight, fast |
| `PrimusSegmentationModel` | `primus.py` | Patch Decode | Simple, direct |
| `PrimusMultiscaleSegmentationModel` | `primus.py` | Multi-layer Patch | Simple + multi-scale |

### General Architecture Pattern
```
Vision Transformer Backbone (frozen or fine-tuned)
         │
         ├─ Extracts multi-level features from transformer blocks
         │  (layers [2,5,8,11] for 12-layer ViT)
         │  (layers [5,11,17,23] for 24-layer ViT)
         ▼
    [Decoder Head] (UperNet | SegFormer | PatchDecode)
         │
         ├─ Multi-scale aggregation (varies by decoder)
         ▼
  ProgressiveUpsampleDecoder (for UperNet/SegFormer)
         │
         ├─ Multi-stage 2x upsampling (eliminates patch artifacts)
         ├─ Skip connections from backbone features
         ├─ GroupNorm + ReLU + Dropout2d between stages
         ▼
    Final Conv (num_classes output)
```

### Supported Backbones

| Backbone | Patch Size | Model ID Pattern |
|----------|------------|------------------|
| DINOv3 | 16 | `vit_{size}_patch16_dinov3.lvd1689m` |
| DINOv2 | 14 | `vit_{size}_patch14_dinov2.lvd142m` |
| RetFound | 16 | Retina-specific |
| VisionFM | 16 | General vision |

**Note**: DINOv2 requires input padding due to patch_size=14. Handled automatically in models.

### YAML Configuration System

Models are configured via YAML files in `nnunetv2/configs/`:

```yaml
model:
  model_path: "nnunetv2.model.model_segformer_1"  # Dynamic module loading
  backbone: "dinov3"
  backbone_size: "large"
  decoder:
    hidden_size: 256

lora:
  enabled: true
  r: 8
  lora_alpha: 16
  target_modules: ["qkv"]

train:
  initial_lr: 0.001
  weight_decay: 0.00003
  num_epochs: 500
```

Configuration classes in `nnunetv2/training/nnUNetTrainer/vit_config.py`:
- `ViTModelConfig` - Model architecture parameters
- `ViTLoRAConfig` - LoRA adapter settings
- `ViTTrainConfig` - Training hyperparameters
- `ViTnnUNetConfig` - Root config combining all

### LoRA Fine-Tuning
Enable with `use_lora=True` to wrap backbone with low-rank adapters:
- Reduces trainable parameters from millions to thousands
- Uses PEFT library for efficient adaptation
- Configure via `lora_r`, `lora_alpha`, `lora_dropout` parameters

### Parameter Inspection
```python
model.print_trainable_parameters()  # Total counts by component
model.print_trainable_layers()      # Detailed layer-by-layer breakdown
```

### Custom Trainers
- `nnUNetTrainer_vit` - Config-driven ViT (recommended)
- `nnUNetTrainer_mymodel` - Legacy ViT training with deep supervision
- `nnUNetTrainerNoDeepSupervision_mymodel` - Legacy ViT without auxiliary losses
- `nnUNetTrainer_m2f` - Mask2Former decoder training
