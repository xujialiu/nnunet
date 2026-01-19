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
