"""OmegaConf configuration for Mask2Former integration with nnUNet."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class M2FModelConfig:
    """Mask2Former model configuration."""
    backbone: str = "dinov3"  # Backbone type
    backbone_size: str = "large"  # small, base, large, giant
    hidden_dim: int = 2048  # Hidden dimension for M2F decoder
    backbone_out_layers: str = "FOUR_EVEN_INTERVALS"  # Layer extraction strategy
    autocast_dtype: str = "float32"  # bfloat16, float16, float32


@dataclass
class M2FLoRAConfig:
    """LoRA configuration for backbone fine-tuning."""
    enabled: bool = True
    r: int = 8  # LoRA rank
    lora_alpha: int = 16  # LoRA scaling factor
    target_modules: List[str] = field(default_factory=lambda: ["qkv", "proj"])
    lora_dropout: float = 0.1
    bias: str = "none"
    use_rslora: bool = False


@dataclass
class M2FTrainConfig:
    """Mask2Former training hyperparameters."""
    num_points: int = 12544  # Points sampled for mask loss (112*112)
    oversample_ratio: float = 3.0
    importance_sample_ratio: float = 0.75
    mask_coefficient: float = 5.0  # Mask BCE loss weight
    dice_coefficient: float = 5.0  # Dice loss weight
    class_coefficient: float = 2.0  # Classification CE loss weight
    no_object_coefficient: float = 0.1  # "no object" class weight


@dataclass
class M2FnnUNetConfig:
    """Root configuration for M2F integration with nnUNet."""
    model: M2FModelConfig = field(default_factory=M2FModelConfig)
    lora: M2FLoRAConfig = field(default_factory=M2FLoRAConfig)
    m2f_train: M2FTrainConfig = field(default_factory=M2FTrainConfig)

    # nnUNet-specific settings
    gradient_clip: float = 10.0  # M2F uses lower gradient clip than nnUNet's 12
