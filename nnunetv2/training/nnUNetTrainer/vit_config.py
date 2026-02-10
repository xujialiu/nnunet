"""OmegaConf configuration for ViT integration with nnUNet."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ViTModelConfig:
    """ViT model configuration."""
    model_path: str = "nnunetv2.model.model_segformer_1"  # Python module path
    backbone: str = "dinov3"  # Backbone type: dinov3, dinov2, retfound, visionfm
    backbone_size: str = "large"  # small, base, large
    checkpoint_path: str = ""  # Optional path to backbone checkpoint
    # Free-form decoder config - each model interprets its own parameters
    decoder: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ViTLoRAConfig:
    """LoRA configuration for backbone fine-tuning."""
    enabled: bool = True
    r: int = 8  # LoRA rank
    lora_alpha: int = 16  # LoRA scaling factor
    target_modules: List[str] = field(default_factory=lambda: ["qkv"])
    lora_dropout: float = 0.05


@dataclass
class ViTTrainConfig:
    """Training hyperparameters specific to ViT models.

    All fields are optional. If not set in config YAML,
    base nnUNetTrainer defaults will be used:
      - initial_lr: 1e-2
      - weight_decay: 3e-5
      - num_epochs: 1000
      - num_iterations_per_epoch: 100
      - num_val_iterations_per_epoch: 50
    """
    initial_lr: Optional[float] = None
    weight_decay: Optional[float] = None
    gradient_clip: Optional[float] = None
    num_epochs: Optional[int] = None
    num_iterations_per_epoch: Optional[int] = None
    num_val_iterations_per_epoch: Optional[int] = None


@dataclass
class ViTnnUNetConfig:
    """Root configuration for ViT integration with nnUNet."""
    model: ViTModelConfig = field(default_factory=ViTModelConfig)
    lora: ViTLoRAConfig = field(default_factory=ViTLoRAConfig)
    train: ViTTrainConfig = field(default_factory=ViTTrainConfig)
