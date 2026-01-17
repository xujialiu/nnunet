"""nnUNet trainer for DINOv3 Mask2Former semantic segmentation."""

import os
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import torch
from torch import autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from omegaconf import OmegaConf

from nnunetv2.training.nnUNetTrainer.m2f_config import M2FnnUNetConfig
from nnunetv2.training.nnUNetTrainer.m2f_utils import (
    DenseToM2FTargetConverter,
    m2f_predictions_to_segmentation,
)
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_m2f(nnUNetTrainer):
    """nnUNet trainer with DINOv3 Mask2Former decoder.

    This trainer replaces the UperNet head with Mask2Former for
    query-based semantic segmentation.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
        m2f_config_path: Optional[str] = None,
    ):
        """Initialize M2F trainer.

        Args:
            plans: nnUNet plans dict
            configuration: Configuration name (e.g., "2d")
            fold: Fold number for cross-validation
            dataset_json: Dataset JSON metadata
            device: PyTorch device
            m2f_config_path: Optional path to M2F YAML config
        """
        # Store m2f_config_path before calling super().__init__
        self._m2f_config_path = m2f_config_path

        super().__init__(plans, configuration, fold, dataset_json, device)

        ### Some hyperparameters for you to fiddle with
        self.initial_lr = 1e-4
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33
        self.probabilistic_oversampling = False
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.num_epochs = 1000
        self.current_epoch = 0
        self.enable_deep_supervision = False  # Disable deep supervision (M2F handles it internally via aux_outputs)

        # Add m2f_config_path to my_init_kwargs (parent uses inspect.signature on child class)
        self.my_init_kwargs["m2f_config_path"] = m2f_config_path

        # Load M2F config
        self.m2f_config = self._load_m2f_config(m2f_config_path)

        # Target converter will be initialized after we know ignore_label
        self._target_converter = None

    def _load_m2f_config(self, config_path: Optional[str]) -> M2FnnUNetConfig:
        """Load M2F configuration from YAML or use defaults."""
        base_config = OmegaConf.structured(M2FnnUNetConfig)

        if config_path and os.path.exists(config_path):
            yaml_config = OmegaConf.load(config_path)
            config = OmegaConf.merge(base_config, yaml_config)
        else:
            # Try default config location
            default_path = (
                Path(__file__).parent.parent.parent / "configs" / "m2f_default.yaml"
            )
            if default_path.exists():
                yaml_config = OmegaConf.load(default_path)
                config = OmegaConf.merge(base_config, yaml_config)
            else:
                config = base_config

        return OmegaConf.to_object(config)

    @property
    def target_converter(self) -> DenseToM2FTargetConverter:
        """Lazy initialization of target converter."""
        if self._target_converter is None:
            ignore_label = self.label_manager.ignore_label
            self._target_converter = DenseToM2FTargetConverter(
                ignore_label=ignore_label if ignore_label is not None else -1
            )
        return self._target_converter

    def initialize(self):
        """Initialize network, optimizer, and loss for M2F."""
        if not self.was_initialized:
            self._set_batch_size_and_oversample()

            # Build M2F model
            self.network = self._build_m2f_network().to(self.device)

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
                self.network = DDP(self.network, device_ids=[self.local_rank])

            # Build M2F loss
            self.loss = self._build_loss()

            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
            self.was_initialized = True
        else:
            raise RuntimeError("Trainer already initialized")

    def _build_m2f_network(self) -> torch.nn.Module:
        """Build Mask2Former network with DINOv3 backbone."""
        from dinov3.eval.segmentation.models import build_segmentation_decoder
        from dinov3.eval.setup import ModelConfig, load_model_and_context
        from dinov3.eval.segmentation.config import LoRAConfig

        num_classes = self.label_manager.num_segmentation_heads

        # Construct hub model name from backbone_size
        # "large" -> "l", "base" -> "b", "small" -> "s", "giant" -> "g"
        # DINOv3 uses patch size 16 (not 14 like DINOv2)
        size_letter = self.m2f_config.model.backbone_size[0]
        hub_name = f"dinov3_vit{size_letter}16"

        model_config = ModelConfig(dino_hub=hub_name)

        # LoRA config conversion
        lora_config = None
        if self.m2f_config.lora.enabled:
            lora_config = LoRAConfig(
                enabled=True,
                r=self.m2f_config.lora.r,
                lora_alpha=self.m2f_config.lora.lora_alpha,
                target_modules=list(self.m2f_config.lora.target_modules),
                lora_dropout=self.m2f_config.lora.lora_dropout,
                bias=self.m2f_config.lora.bias,
                use_rslora=self.m2f_config.lora.use_rslora,
            )

        backbone, _ = load_model_and_context(
            model_config,
            output_dir=str(self.output_folder),
            lora_config=lora_config,
        )

        # Determine autocast dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        autocast_dtype = dtype_map.get(
            self.m2f_config.model.autocast_dtype, torch.float32
        )

        # Build segmentation decoder
        model = build_segmentation_decoder(
            backbone_model=backbone,
            decoder_type="m2f",
            hidden_dim=self.m2f_config.model.hidden_dim,
            num_classes=num_classes,
            autocast_dtype=autocast_dtype,
            lora_enabled=self.m2f_config.lora.enabled,
        )

        return model

    def _build_loss(self):
        """Build MaskClassificationLoss for M2F."""
        from dinov3.eval.segmentation.mask_classification_loss import (
            MaskClassificationLoss,
        )

        num_classes = self.label_manager.num_segmentation_heads
        cfg = self.m2f_config.m2f_train

        loss = MaskClassificationLoss(
            num_points=cfg.num_points,
            oversample_ratio=cfg.oversample_ratio,
            importance_sample_ratio=cfg.importance_sample_ratio,
            mask_coefficient=cfg.mask_coefficient,
            dice_coefficient=cfg.dice_coefficient,
            class_coefficient=cfg.class_coefficient,
            num_labels=num_classes,
            no_object_coefficient=cfg.no_object_coefficient,
        )

        # Move loss to device (contains CrossEntropyLoss with class weights)
        return loss.to(self.device)

    def set_deep_supervision_enabled(self, enabled: bool):
        """Override for M2F architecture.

        M2F handles deep supervision internally via aux_outputs from the
        transformer decoder, so we don't need to set anything on the network.
        """
        # No-op: M2F doesn't use the same deep supervision mechanism as U-Net
        pass

    def train_step(self, batch: dict) -> dict:
        """Training step for M2F model."""
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = target[0]  # Take highest resolution for M2F
        target = target.to(self.device, non_blocking=True)

        # Convert target to M2F format
        m2f_targets = self.target_converter(target)

        self.optimizer.zero_grad(set_to_none=True)

        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            # Forward pass - returns dict
            pred = self.network(data)

            # Compute main loss
            losses = self.loss(
                masks_queries_logits=pred["pred_masks"].float(),
                targets=m2f_targets,
                class_queries_logits=pred["pred_logits"].float(),
            )

            # Compute auxiliary losses
            for i, aux_output in enumerate(pred.get("aux_outputs", [])):
                aux_losses = self.loss(
                    masks_queries_logits=aux_output["pred_masks"].float(),
                    targets=m2f_targets,
                    class_queries_logits=aux_output["pred_logits"].float(),
                )
                for key, value in aux_losses.items():
                    losses[f"{key}_{i}"] = value

            # Aggregate total loss
            loss_total = self.loss.loss_total(
                losses, log_fn=lambda *args, **kwargs: None
            )

        # Backward pass
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss_total).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), self.m2f_config.gradient_clip
            )
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), self.m2f_config.gradient_clip
            )
            self.optimizer.step()

        return {"loss": loss_total.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        """Validation step for M2F model."""
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = target[0]
        target = target.to(self.device, non_blocking=True)

        # Convert target to M2F format
        m2f_targets = self.target_converter(target)

        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            # Forward pass
            pred = self.network(data)

            # Compute loss
            losses = self.loss(
                masks_queries_logits=pred["pred_masks"].float(),
                targets=m2f_targets,
                class_queries_logits=pred["pred_logits"].float(),
            )
            loss_total = self.loss.loss_total(
                losses, log_fn=lambda *args, **kwargs: None
            )

            # Convert predictions to dense segmentation for metrics
            input_size = data.shape[-2:]
            output = m2f_predictions_to_segmentation(
                pred["pred_masks"],
                pred["pred_logits"],
                input_size,
            )

        # Compute Dice metrics (reuse parent's logic)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).float()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(
                output.shape, device=output.device, dtype=torch.float32
            )
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)

        # Handle ignore label
        if self.label_manager.has_ignore_label:
            mask = (target != self.label_manager.ignore_label).float()
            target_for_metrics = target.clone()
            target_for_metrics[target == self.label_manager.ignore_label] = 0
        else:
            mask = None
            target_for_metrics = target

        # Convert target to one-hot for metric computation
        # Use torch.bool dtype for compatibility with get_tp_fp_fn_tn (~y_onehot)
        if target_for_metrics.dim() == 4:
            target_for_metrics = target_for_metrics.squeeze(1)  # [B, H, W]
        target_onehot = torch.zeros(
            output.shape, device=output.device, dtype=torch.bool
        )
        target_onehot.scatter_(1, target_for_metrics.unsqueeze(1).long(), 1)

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot, target_onehot, axes=axes, mask=mask
        )

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()

        if not self.label_manager.has_regions:
            # Remove background class
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            "loss": loss_total.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }
