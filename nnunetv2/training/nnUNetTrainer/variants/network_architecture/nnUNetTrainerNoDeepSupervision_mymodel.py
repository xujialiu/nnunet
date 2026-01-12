print("nnUNetTrainerNoDeepSupervision_mymodel.py")
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_mymodel import nnUNetTrainer_mymodel
import torch


class nnUNetTrainerNoDeepSupervision_mymodel(nnUNetTrainer_mymodel):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False
