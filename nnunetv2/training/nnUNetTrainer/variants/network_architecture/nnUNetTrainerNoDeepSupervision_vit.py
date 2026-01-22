"""nnUNet trainer for ViT models without deep supervision."""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_vit import nnUNetTrainer_vit
import torch
from typing import Optional


class nnUNetTrainerNoDeepSupervision_vit(nnUNetTrainer_vit):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
        config_path: Optional[str] = None,
    ):
        super().__init__(plans, configuration, fold, dataset_json, device, config_path)
        self.enable_deep_supervision = False
