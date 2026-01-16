# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from dataclasses import dataclass, field
from enum import Enum
from omegaconf import MISSING
from typing import Any

import torch

from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from dinov3.eval.segmentation.models import BackboneLayersSet
from dinov3.eval.setup import ModelConfig


DEFAULT_MEAN = tuple(mean * 255 for mean in IMAGENET_DEFAULT_MEAN)
DEFAULT_STD = tuple(std * 255 for std in IMAGENET_DEFAULT_STD)


class ModelDtype(Enum):
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"

    @property
    def autocast_dtype(self):
        return {
            ModelDtype.BFLOAT16: torch.bfloat16,
            ModelDtype.FLOAT32: torch.float32,
        }[self]


@dataclass
class OptimizerConfig:
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 1e-2
    gradient_clip: float = 35.0


@dataclass
class SchedulerConfig:
    type: str = "WarmupOneCycleLR"
    total_iter: int = 40_000  # Total number of iterations for training
    constructor_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    root: str = MISSING  # Path to the dataset folder
    train: str = ""  # Dataset descriptor, e.g. "ADE20K:split=TRAIN"
    val: str = ""


@dataclass
class DecoderConfig:
    type: str = "m2f"  # Decoder type must be one of [linear, m2f]
    backbone_out_layers: BackboneLayersSet = BackboneLayersSet.LAST
    use_batchnorm: bool = True
    use_cls_token: bool = False
    use_backbone_norm: bool = True  # Uses the backbone's output normalization on all layers
    num_classes: int = 150  # Number of segmentation classes
    hidden_dim: int = 2048  # Hidden dimension, only used for M2F head
    dropout: float = 0.1  # Dropout ratio in the linear head during training


@dataclass
class TrainConfig:
    diceloss_weight: float = 0.0
    celoss_weight: float = 1.0


@dataclass
class M2FTrainConfig:
    """Mask2Former-specific training hyperparameters.

    Note: no_object_coefficient is passed to MaskClassificationLoss which stores it as eos_coef.
    """
    num_points: int = 12544  # Points sampled for mask loss (112*112)
    oversample_ratio: float = 3.0
    importance_sample_ratio: float = 0.75
    mask_coefficient: float = 5.0
    dice_coefficient: float = 5.0
    class_coefficient: float = 2.0
    no_object_coefficient: float = 0.1


@dataclass
class TrainTransformConfig:
    img_size: Any = None
    random_img_size_ratio_range: tuple[float] | None = None
    crop_size: tuple[int] | None = None
    flip_prob: float = 0.0


@dataclass
class EvalTransformConfig:
    img_size: Any = None
    tta_ratios: tuple[float] = (1.0,)


@dataclass
class TransformConfig:
    train: TrainTransformConfig = field(default_factory=TrainTransformConfig)
    eval: EvalTransformConfig = field(default_factory=EvalTransformConfig)
    mean: tuple[float] = DEFAULT_MEAN
    std: tuple[float] = DEFAULT_STD


@dataclass
class EvalConfig:
    compute_metric_per_image: bool = False
    reduce_zero_label: bool = True  # For ADE20K, ignores 0 label (=background/unlabeled)
    mode: str = "slide"
    crop_size: int | None = None  # Required for slide mode, must be None for whole mode
    stride: int | None = None  # Required for slide mode, must be None for whole mode
    eval_interval: int = 40000
    use_tta: bool = False  # apply test-time augmentation at evaluation time
    max_val_samples: int = 0  # 0 means no limit, useful for smoke tests
    num_visualizations: int = 0  # Number of samples to visualize during validation (0 = no visualization)


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration for backbone fine-tuning."""
    enabled: bool = False  # Whether to enable LoRA
    r: int = 8  # LoRA rank (low-rank dimension)
    lora_alpha: int = 16  # LoRA scaling factor (scale = alpha/r)
    target_modules: list[str] = field(default_factory=lambda: ["qkv", "proj"])
    lora_dropout: float = 0.1  # Dropout probability for LoRA layers
    bias: str = "none"  # Bias training: "none", "all", or "lora_only"
    use_rslora: bool = False  # Use Rank-Stabilized LoRA (scale = alpha/sqrt(r))


@dataclass
class SegmentationConfig:
    model: ModelConfig | None = None  # config of the DINOv3 backbone
    bs: int = 2
    n_gpus: int = 8
    num_workers: int = 6  # number of workers to use / GPU
    model_dtype: ModelDtype = ModelDtype.FLOAT32
    seed: int = 100
    datasets: DatasetConfig = field(default_factory=DatasetConfig)
    metric_to_save: str = "mIoU"  # Name of the metric to save
    decoder_head: DecoderConfig = field(default_factory=DecoderConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    transforms: TransformConfig = field(default_factory=TransformConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    m2f_train: M2FTrainConfig = field(default_factory=M2FTrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    # Additional Parameters
    output_dir: str | None = None
    load_from: str | None = None  # path to .pt checkpoint to resume training from or evaluate from
