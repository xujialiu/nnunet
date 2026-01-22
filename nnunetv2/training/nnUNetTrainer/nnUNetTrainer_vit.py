"""nnUNet trainer for ViT-based semantic segmentation."""

import importlib
import os
from pathlib import Path
from typing import Optional

import torch

from omegaconf import OmegaConf

from nnunetv2.training.nnUNetTrainer.vit_config import ViTnnUNetConfig
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels


class nnUNetTrainer_vit(nnUNetTrainer):
    """nnUNet trainer with ViT backbone and configurable decoder.

    Uses ViT backbone with configurable segmentation head architecture.
    Model architecture is determined by config's model_path field.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
        config_path: Optional[str] = None,
    ):
        # Store config path before calling super().__init__
        self._config_path = config_path

        super().__init__(plans, configuration, fold, dataset_json, device)

        # Add config path to init kwargs for checkpoint serialization
        self.my_init_kwargs["config_path"] = config_path

        # Load ViT config
        self.vit_config = self._load_vit_config(config_path)

        # ViT models typically don't support deep supervision
        self.enable_deep_supervision = False

        # Override training hyperparameters from config (only if explicitly set)
        # If not set in config, base nnUNetTrainer defaults are used
        if self.vit_config.train.initial_lr is not None:
            self.initial_lr = self.vit_config.train.initial_lr
        if self.vit_config.train.weight_decay is not None:
            self.weight_decay = self.vit_config.train.weight_decay
        if self.vit_config.train.num_epochs is not None:
            self.num_epochs = self.vit_config.train.num_epochs
        if self.vit_config.train.num_iterations_per_epoch is not None:
            self.num_iterations_per_epoch = self.vit_config.train.num_iterations_per_epoch
        if self.vit_config.train.num_val_iterations_per_epoch is not None:
            self.num_val_iterations_per_epoch = self.vit_config.train.num_val_iterations_per_epoch

    def _load_vit_config(self, config_path: Optional[str]) -> ViTnnUNetConfig:
        """Load ViT configuration from YAML file.

        Args:
            config_path: Path to YAML config file. Required.

        Raises:
            ValueError: If config_path is not provided.
            FileNotFoundError: If config file does not exist.
        """
        if not config_path:
            raise ValueError(
                "Config file is required for ViT training. "
                "Please provide --config path/to/config.yaml"
            )

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config file not found: {config_path}"
            )

        base_config = OmegaConf.structured(ViTnnUNetConfig)
        yaml_config = OmegaConf.load(config_path)
        config = OmegaConf.merge(base_config, yaml_config)

        return OmegaConf.to_object(config)

    def initialize(self):
        """Initialize network, optimizer, and loss for ViT model."""
        if not self.was_initialized:
            self._set_batch_size_and_oversample()

            self.num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json
            )

            # Dynamic model loading based on config
            cfg = self.vit_config
            model_module = importlib.import_module(cfg.model.model_path)

            # Get the factory function (standardized name)
            if hasattr(model_module, "create_segmentation_model"):
                create_model_fn = model_module.create_segmentation_model
            elif hasattr(model_module, "create_segformer_model"):
                # Backward compatibility
                create_model_fn = model_module.create_segformer_model
            else:
                raise AttributeError(
                    f"Module {cfg.model.model_path} must have 'create_segmentation_model' or "
                    "'create_segformer_model' function"
                )

            checkpoint_path = cfg.model.checkpoint_path if cfg.model.checkpoint_path else None

            # Pass decoder config as kwargs - each model interprets its own parameters
            decoder_kwargs = dict(cfg.model.decoder) if cfg.model.decoder else {}

            self.network = create_model_fn(
                backbone_name=cfg.model.backbone,
                backbone_size=cfg.model.backbone_size,
                num_classes=self.label_manager.num_segmentation_heads,
                checkpoint_path=checkpoint_path,
                freeze_backbone=False,  # LoRA will handle this
                use_lora=cfg.lora.enabled,
                lora_rank=cfg.lora.r,
                lora_alpha=cfg.lora.lora_alpha,
                lora_dropout=cfg.lora.lora_dropout,
                lora_target_modules=list(cfg.lora.target_modules),
                **decoder_kwargs,
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
        """Override for ViT architecture - no deep supervision support."""
        # No-op: ViT models don't use deep supervision
        pass
