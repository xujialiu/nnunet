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
