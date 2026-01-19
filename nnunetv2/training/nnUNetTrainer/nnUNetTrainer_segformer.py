"""nnUNet trainer for SegFormer semantic segmentation."""

import os
from pathlib import Path
from typing import Optional

import torch

from omegaconf import OmegaConf

from nnunetv2.training.nnUNetTrainer.segformer_config import SegFormernnUNetConfig
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels


class nnUNetTrainer_segformer(nnUNetTrainer):
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
            from nnunetv2.model.model_segformer_1 import create_segformer_model

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
            self._log_trainable_parameters()

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
