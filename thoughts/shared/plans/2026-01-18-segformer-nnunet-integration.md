# SegFormer Integration into nnUNet Pipeline - Implementation Plan

## Overview

Integrate SegFormer model into the nnUNet pipeline for comparable comparison with the existing ViT+UperNet model. This follows the established M2F integration pattern with OmegaConf configuration.

## Current State Analysis

### What Exists

- `model_segformer.py` (lines 1-86): Only `create_backbone()` and `get_vit_features()` functions
- `model.py`: Complete `ViTSegmentationModel` with UperNet head (reference implementation)
- `m2f_config.py`: OmegaConf dataclass pattern for configuration
- `nnUNetTrainer_m2f.py`: Trainer with custom config loading pattern
- `nnUNetTrainer_mymodel.py`: Standalone trainer that creates custom models
- `run_training_nodeepsupervision_mymodel.py`: Training script pattern

### What's Missing

- SegFormer decode head and model class in `model_segformer.py`
- `SegFormernnUNetConfig` dataclass configuration
- `nnUNetTrainer_segformer` trainer class
- `nnUNetTrainerNoDeepSupervision_segformer` variant
- `run_training_segformer.py` run module
- `nnUNetv2_train_segformer.py` entry point script

### Key Discoveries

- SegFormer decode head available via `transformers.models.segformer.modeling_segformer.SegformerDecodeHead`
- ViT backbones produce single-scale features; need to replicate to simulate multi-scale for SegFormer head
- M2F pattern stores config path in `my_init_kwargs` for checkpoint serialization (`nnUNetTrainer_m2f.py:95`)
- Trainer `initialize()` imports model lazily to avoid circular imports (`nnUNetTrainer_mymodel.py:296`)

## Desired End State

A complete SegFormer integration that:

1. Creates SegFormer models with ViT backbones (DINOv3/DINOv2/RetFound/VisionFM)
2. Uses OmegaConf for configurable parameters (backbone, decoder size, LoRA)
3. Can be trained via `python nnUNetv2_train_segformer.py DATASET_ID 2d FOLD`
4. Uses the same loss function (`DC_and_CE_loss`) as the ViT model for fair comparison
5. Supports LoRA fine-tuning of the backbone

### Verification

- Training script runs without errors on a test dataset
- Model produces segmentation outputs at input resolution
- Checkpoints save and load correctly (including config path)
- Validation metrics are computed and logged

## What We're NOT Doing

- Deep supervision (SegFormer doesn't naturally support it; output is single-scale)
- Cascaded training support (can be added later if needed)
- Custom loss functions (reuse existing `DC_and_CE_loss`)
- Inference wrapper (direct tensor output, no dict conversion needed unlike M2F)

## Implementation Approach

Follow the M2F integration pattern:

1. Create model file with SegFormer-specific architecture
2. Create OmegaConf config dataclasses
3. Create trainer that loads config and builds model
4. Create run module and entry point script
5. Create NoDeepSupervision variant (default for SegFormer)

---

## Phase 1: Model Implementation (`model_segformer.py`)

### Overview

Complete the `model_segformer.py` file with SegFormer decode head and model class.

### Changes Required:

#### 1. Add SegFormerSegmentationModel Class

**File**: `nnunetv2/model/model_segformer.py`
**Changes**: Add SegFormer model class after existing functions (after line 86)

```python
from transformers.models.segformer.modeling_segformer import SegformerDecodeHead
from transformers import SegformerConfig


class SegFormerSegmentationModel(nn.Module):
    """SegFormer-based segmentation model with ViT backbone.

    Uses HuggingFace SegformerDecodeHead with multi-scale feature simulation
    from ViT backbone.
    """

    # Default layer indices for different ViT depths
    LAYER_INDICES = {
        12: [2, 5, 8, 11],  # Base models
        24: [5, 11, 17, 23],  # Large models
    }

    def __init__(
        self,
        backbone_name: str = "dinov3",
        backbone_size: str = "large",
        num_classes: int = 1,
        decoder_hidden_size: int = 256,
        checkpoint_path: Optional[str] = None,
        freeze_backbone: bool = False,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.decoder_hidden_size = decoder_hidden_size

        # Create backbone
        self.backbone, self.patch_size, _ = create_backbone(
            model_name=backbone_name,
            model_size=backbone_size,
            checkpoint_path=checkpoint_path,
        )

        # Determine feature extraction layers
        num_layers = len(self.backbone.blocks)
        self.feature_indices = self.LAYER_INDICES.get(
            num_layers,
            [
                num_layers // 4 - 1,
                num_layers // 2 - 1,
                3 * num_layers // 4 - 1,
                num_layers - 1,
            ],
        )

        # Get backbone embed dimension
        backbone_embed_dim = self.backbone.embed_dim

        # SegFormer expects 4 feature maps with different channel dimensions
        # ViT produces same-dim features, so we use projections
        hidden_sizes = [backbone_embed_dim] * 4

        # Create SegFormer config
        segformer_config = SegformerConfig(
            hidden_sizes=hidden_sizes,
            num_labels=num_classes,
            decoder_hidden_size=decoder_hidden_size,
        )

        self.decode_head = SegformerDecodeHead(segformer_config)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Enable LoRA if requested
        if use_lora:
            self.enable_lora(
                rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        # Extract multi-level features from ViT
        features = get_vit_features(
            self.backbone,
            x,
            indices=self.feature_indices,
        )

        # SegformerDecodeHead expects list of features
        features_list = list(features)

        # Apply SegFormer decode head
        # Output is at H/4 x W/4 resolution
        logits = self.decode_head(features_list)

        # Upsample to input resolution
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(
                logits,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )

        return logits

    def print_trainable_parameters(self, detailed: bool = False) -> None:
        """Print the number of trainable parameters in the model."""
        def count_params(module: nn.Module) -> Tuple[int, int]:
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total = sum(p.numel() for p in module.parameters())
            return trainable, total

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.parameters())
        percentage = 100 * trainable_params / all_params if all_params > 0 else 0

        print(
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_params:,} || "
            f"trainable%: {percentage:.2f}%"
        )

        if detailed:
            print("-" * 60)
            backbone_trainable, backbone_total = count_params(self.backbone)
            backbone_pct = 100 * backbone_trainable / backbone_total if backbone_total > 0 else 0
            print(f"  backbone:     {backbone_trainable:>12,} / {backbone_total:>12,} ({backbone_pct:.2f}%)")

            head_trainable, head_total = count_params(self.decode_head)
            head_pct = 100 * head_trainable / head_total if head_total > 0 else 0
            print(f"  decode_head:  {head_trainable:>12,} / {head_total:>12,} ({head_pct:.2f}%)")

    def enable_lora(
        self,
        rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
    ) -> None:
        """Enable LoRA for efficient fine-tuning of the backbone."""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError("LoRA requires 'peft' library: pip install peft")

        if target_modules is None:
            target_modules = ["qkv"]

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )

        self.backbone = get_peft_model(self.backbone, lora_config)

        # Ensure decode head remains trainable
        for param in self.decode_head.parameters():
            param.requires_grad = True

        print(f"LoRA enabled with rank={rank}, alpha={lora_alpha}")
        self.print_trainable_parameters(detailed=True)

    def disable_lora(self) -> None:
        """Disable LoRA and merge weights into backbone."""
        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError("LoRA requires 'peft' library: pip install peft")

        if isinstance(self.backbone, PeftModel):
            self.backbone = self.backbone.merge_and_unload()
            print("LoRA disabled and weights merged into backbone")
        else:
            print("LoRA is not enabled on this model")


def create_segformer_model(
    backbone_name: str = "dinov3",
    backbone_size: str = "large",
    num_classes: int = 1,
    decoder_hidden_size: int = 256,
    checkpoint_path: Optional[str] = None,
    freeze_backbone: bool = False,
    use_lora: bool = True,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """Factory function to create SegFormer segmentation models.

    Args:
        backbone_name: One of "dinov3", "dinov2", "retfound", "visionfm"
        backbone_size: "base" or "large"
        num_classes: Number of segmentation classes
        decoder_hidden_size: Hidden dimension for SegFormer decoder (default: 256)
        checkpoint_path: Path to pretrained backbone weights
        freeze_backbone: Whether to freeze backbone weights
        use_lora: Enable LoRA for efficient fine-tuning
        lora_rank: LoRA rank
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout probability for LoRA layers
        lora_target_modules: Module names to apply LoRA to (default: ["qkv"])

    Returns:
        SegFormer segmentation model ready for training
    """
    return SegFormerSegmentationModel(
        backbone_name=backbone_name,
        backbone_size=backbone_size,
        num_classes=num_classes,
        decoder_hidden_size=decoder_hidden_size,
        checkpoint_path=checkpoint_path,
        freeze_backbone=freeze_backbone,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
    )
```

### Success Criteria:

#### Automated Verification:

- [X] Python imports work: `python -c "from nnunetv2.model.model_segformer import create_segformer_model"`
- [X] Model instantiation works: `python -c "from nnunetv2.model.model_segformer import create_segformer_model; m = create_segformer_model(num_classes=4, use_lora=False); print(m)"`
- [X] Forward pass works with dummy input

#### Manual Verification:

- [X] Model architecture printed correctly shows backbone + decode_head components
- [X] Trainable parameter counts are reasonable (~300M total, ~10M trainable with LoRA)

**Implementation Note**: Complete this phase before proceeding.

---

## Phase 2: Configuration (`segformer_config.py`)

### Overview

Create OmegaConf dataclass configuration following M2F pattern.

### Changes Required:

#### 1. Create Configuration File

**File**: `nnunetv2/training/nnUNetTrainer/segformer_config.py`
**Changes**: New file

```python
"""OmegaConf configuration for SegFormer integration with nnUNet."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SegFormerModelConfig:
    """SegFormer model configuration."""
    backbone: str = "dinov3"  # Backbone type: dinov3, dinov2, retfound, visionfm
    backbone_size: str = "large"  # small, base, large
    decoder_hidden_size: int = 256  # SegFormer decoder hidden dimension
    checkpoint_path: str = ""  # Optional path to backbone checkpoint


@dataclass
class SegFormerLoRAConfig:
    """LoRA configuration for backbone fine-tuning."""
    enabled: bool = True
    r: int = 8  # LoRA rank
    lora_alpha: int = 16  # LoRA scaling factor
    target_modules: List[str] = field(default_factory=lambda: ["qkv"])
    lora_dropout: float = 0.05


@dataclass
class SegFormerTrainConfig:
    """Training hyperparameters specific to SegFormer."""
    initial_lr: float = 1e-3  # Higher than default nnUNet
    weight_decay: float = 3e-5
    gradient_clip: float = 12.0


@dataclass
class SegFormernnUNetConfig:
    """Root configuration for SegFormer integration with nnUNet."""
    model: SegFormerModelConfig = field(default_factory=SegFormerModelConfig)
    lora: SegFormerLoRAConfig = field(default_factory=SegFormerLoRAConfig)
    train: SegFormerTrainConfig = field(default_factory=SegFormerTrainConfig)
```

### Success Criteria:

#### Automated Verification:

- [x] Import works: `python -c "from nnunetv2.training.nnUNetTrainer.segformer_config import SegFormernnUNetConfig"`
- [x] OmegaConf conversion works: `python -c "from omegaconf import OmegaConf; from nnunetv2.training.nnUNetTrainer.segformer_config import SegFormernnUNetConfig; c = OmegaConf.structured(SegFormernnUNetConfig); print(OmegaConf.to_yaml(c))"`

---

## Phase 3: Trainer Implementation (`nnUNetTrainer_segformer.py`)

### Overview

Create the main trainer class following M2F pattern.

### Changes Required:

#### 1. Create Trainer File

**File**: `nnunetv2/training/nnUNetTrainer/nnUNetTrainer_segformer.py`
**Changes**: New file (based on `nnUNetTrainer_mymodel.py` with M2F config loading pattern)

Key implementation points:

- Inherit from `nnUNetTrainer_mymodel` (reuse most training logic)
- Add `segformer_config_path` constructor parameter
- Load config via OmegaConf in `_load_segformer_config()` method
- Override `initialize()` to use `create_segformer_model()`
- Set `self.enable_deep_supervision = False` (SegFormer doesn't support it)

```python
"""nnUNet trainer for SegFormer semantic segmentation."""

import os
from pathlib import Path
from typing import Optional

import torch

from omegaconf import OmegaConf

from nnunetv2.training.nnUNetTrainer.segformer_config import SegFormernnUNetConfig
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_mymodel import nnUNetTrainer_mymodel
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels


class nnUNetTrainer_segformer(nnUNetTrainer_mymodel):
    """nnUNet trainer with SegFormer decoder.

    Uses ViT backbone with SegFormer decode head for semantic segmentation.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
        segformer_config_path: Optional[str] = None,
    ):
        # Store config path before calling super().__init__
        self._segformer_config_path = segformer_config_path

        super().__init__(plans, configuration, fold, dataset_json, device)

        # Add config path to init kwargs for checkpoint serialization
        self.my_init_kwargs["segformer_config_path"] = segformer_config_path

        # Load SegFormer config
        self.segformer_config = self._load_segformer_config(segformer_config_path)

        # SegFormer doesn't support deep supervision
        self.enable_deep_supervision = False

        # Override training hyperparameters from config
        self.initial_lr = self.segformer_config.train.initial_lr
        self.weight_decay = self.segformer_config.train.weight_decay

    def _load_segformer_config(self, config_path: Optional[str]) -> SegFormernnUNetConfig:
        """Load SegFormer configuration from YAML or use defaults."""
        base_config = OmegaConf.structured(SegFormernnUNetConfig)

        if config_path and os.path.exists(config_path):
            yaml_config = OmegaConf.load(config_path)
            config = OmegaConf.merge(base_config, yaml_config)
        else:
            # Try default config location
            default_path = (
                Path(__file__).parent.parent.parent / "configs" / "segformer_default.yaml"
            )
            if default_path.exists():
                yaml_config = OmegaConf.load(default_path)
                config = OmegaConf.merge(base_config, yaml_config)
            else:
                config = base_config

        return OmegaConf.to_object(config)

    def initialize(self):
        """Initialize network, optimizer, and loss for SegFormer."""
        if not self.was_initialized:
            self._set_batch_size_and_oversample()

            self.num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json
            )

            # Build SegFormer model
            from nnunetv2.model.model_segformer import create_segformer_model

            cfg = self.segformer_config
            checkpoint_path = cfg.model.checkpoint_path if cfg.model.checkpoint_path else None

            self.network = create_segformer_model(
                backbone_name=cfg.model.backbone,
                backbone_size=cfg.model.backbone_size,
                num_classes=self.label_manager.num_segmentation_heads,
                decoder_hidden_size=cfg.model.decoder_hidden_size,
                checkpoint_path=checkpoint_path,
                freeze_backbone=False,  # LoRA will handle this
                use_lora=cfg.lora.enabled,
                lora_rank=cfg.lora.r,
                lora_alpha=cfg.lora.lora_alpha,
                lora_dropout=cfg.lora.lora_dropout,
                lora_target_modules=list(cfg.lora.target_modules),
            ).to(self.device)

            print(self.network)

            # Optional torch.compile
            if self._do_i_compile():
                self.print_to_log_file("Using torch.compile...")
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()

            # DDP wrapping
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.network
                )
                from torch.nn.parallel import DistributedDataParallel as DDP
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()

            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
            self.was_initialized = True
        else:
            raise RuntimeError("Trainer already initialized")

    def set_deep_supervision_enabled(self, enabled: bool):
        """Override for SegFormer architecture - no deep supervision support."""
        # No-op: SegFormer doesn't use deep supervision
        pass
```

### Success Criteria:

#### Automated Verification:

- [x] Import works: `python -c "from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_segformer import nnUNetTrainer_segformer"`

---

## Phase 4: NoDeepSupervision Variant

### Overview

Create the simple trainer variant that disables deep supervision.

### Changes Required:

#### 1. Create Variant File

**File**: `nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainerNoDeepSupervision_segformer.py`
**Changes**: New file

```python
"""nnUNet trainer for SegFormer without deep supervision."""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_segformer import nnUNetTrainer_segformer
import torch
from typing import Optional


class nnUNetTrainerNoDeepSupervision_segformer(nnUNetTrainer_segformer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
        segformer_config_path: Optional[str] = None,
    ):
        super().__init__(plans, configuration, fold, dataset_json, device, segformer_config_path)
        self.enable_deep_supervision = False
```

### Success Criteria:

#### Automated Verification:

- [x] Import works: `python -c "from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision_segformer import nnUNetTrainerNoDeepSupervision_segformer"`

---

## Phase 5: Run Module and Entry Point

### Overview

Create the training script and run module.

### Changes Required:

#### 1. Create Run Module

**File**: `nnunetv2/run/run_training_segformer.py`
**Changes**: New file (based on `run_training_nodeepsupervision_mymodel.py`)

Key modifications from the base:

- Default trainer: `nnUNetTrainerNoDeepSupervision_segformer`
- Add `--segformer_config` CLI argument
- Pass config path to trainer constructor

#### 2. Create Entry Point Script

**File**: `nnUNetv2_train_segformer.py` (in project root)
**Changes**: New file

```python
#!/usr/bin/env python
"""Entry point for SegFormer training in nnUNet."""

import os
import sys

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

    from nnunetv2.run.run_training_segformer import run_training_entry
    sys.exit(run_training_entry())
```

### Success Criteria:

#### Automated Verification:

- [x] Entry point runs help: `python nnUNetv2_train_segformer.py --help`

---

## Phase 6: Default Configuration File

### Overview

Create a default YAML configuration file.

### Changes Required:

#### 1. Create Config Directory and File

**File**: `nnunetv2/configs/segformer_default.yaml`
**Changes**: New file

```yaml
# SegFormer default configuration for nnUNet integration

model:
  backbone: "dinov3"
  backbone_size: "large"
  decoder_hidden_size: 256
  checkpoint_path: ""

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

### Success Criteria:

#### Automated Verification:

- [x] YAML parses: `python -c "from omegaconf import OmegaConf; OmegaConf.load('nnunetv2/configs/segformer_default.yaml')"`

---

## Testing Strategy

### Unit Tests:

- Model forward pass with various input sizes (224x224, 512x512, etc.)
- Config loading from YAML and defaults
- LoRA enable/disable functionality

### Integration Tests:

- Full training loop on dummy dataset (Dataset996)
- Checkpoint save/load with config preservation
- Validation metrics computation

### Manual Testing Steps:

1. Run preprocessing: `nnUNetv2_plan_and_preprocess -d 996 --verify_dataset_integrity`
2. Run training: `python nnUNetv2_train_segformer.py 996 2d 0`
3. Verify training logs show SegFormer architecture
4. Verify checkpoint contains config path
5. Continue training from checkpoint: `python nnUNetv2_train_segformer.py 996 2d 0 --c`

---

## Performance Considerations

- SegFormer decode head is lightweight (~5M parameters) compared to backbone (~300M)
- LoRA reduces trainable parameters significantly (~10M trainable with LoRA enabled)
- Bilinear upsampling from H/4 resolution is faster than progressive upsampling
- Memory footprint similar to ViT+UperNet model

---

## References

- Original research: `thoughts/shared/research/2026-01-18-segformer-nnunet-integration.md`
- ViT model reference: `nnunetv2/model/model.py:226-643`
- M2F config pattern: `nnunetv2/training/nnUNetTrainer/m2f_config.py`
- M2F trainer pattern: `nnunetv2/training/nnUNetTrainer/nnUNetTrainer_m2f.py:103-121`
- Run module pattern: `nnunetv2/run/run_training_nodeepsupervision_mymodel.py`
