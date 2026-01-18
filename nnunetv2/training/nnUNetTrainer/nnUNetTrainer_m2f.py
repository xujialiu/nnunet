"""nnUNet trainer for DINOv3 Mask2Former semantic segmentation."""

import os
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import torch
import torch.nn as nn
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


class M2FInferenceWrapper(nn.Module):
    """Wrapper that converts M2F dict output to tensor for inference.

    The standard nnUNet predictor expects the network to return a tensor,
    but M2F returns a dict with pred_masks and pred_logits. This wrapper
    converts the dict output to a dense segmentation tensor.
    """

    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get M2F output (dict with pred_masks, pred_logits, aux_outputs)
        output = self.network(x)

        # Convert to dense segmentation tensor
        input_size = x.shape[-2:]
        segmentation = m2f_predictions_to_segmentation(
            output["pred_masks"],
            output["pred_logits"],
            input_size,
        )
        return segmentation


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
        self.num_iterations_per_epoch = 100  # change to 10 for debug ori 250
        self.num_val_iterations_per_epoch = 50
        self.num_epochs = 200  # for debug ori 1000
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

    def _log_trainable_parameters(self) -> None:
        """Log all trainable parameters with their names and shapes to the log file."""
        trainable_params = []
        frozen_params = []

        for name, param in self.network.named_parameters():
            if param.requires_grad:
                trainable_params.append((name, param.numel(), tuple(param.shape)))
            else:
                frozen_params.append((name, param.numel()))

        total_trainable = sum(p[1] for p in trainable_params)
        total_frozen = sum(p[1] for p in frozen_params)
        total_params = total_trainable + total_frozen

        self.print_to_log_file("=" * 60)
        self.print_to_log_file("TRAINABLE PARAMETERS:")
        self.print_to_log_file("=" * 60)
        for name, numel, shape in trainable_params:
            self.print_to_log_file(f"  {name}: {shape} ({numel:,} params)")
        self.print_to_log_file("-" * 60)
        self.print_to_log_file(
            f"Total trainable: {total_trainable:,} ({100 * total_trainable / total_params:.2f}%)"
        )
        self.print_to_log_file(
            f"Total frozen: {total_frozen:,} ({100 * total_frozen / total_params:.2f}%)"
        )
        self.print_to_log_file(f"Total parameters: {total_params:,}")
        self.print_to_log_file("=" * 60)

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

            self.print_to_log_file(self.network)

            # Log trainable parameters to the log file
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

    def train_step(self, batch: dict, step: int = 0) -> dict:
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

        loss_val = loss_total.detach().cpu().numpy()
        lr = self.optimizer.param_groups[0]["lr"]

        self.print_to_log_file(
            f"step {step}/{self.num_iterations_per_epoch}, loss: {loss_val:.4f}, lr: {lr:.6f}"
        )
        return {"loss": loss_val}

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

    def perform_actual_validation(
        self,
        save_probabilities: bool = False,
        validation_folder_name: str = "validation",
    ):
        """Override to use M2FInferenceWrapper for validation.

        The standard nnUNet predictor expects the network to return a tensor,
        but M2F returns a dict. We wrap the network to convert the output.
        """
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from torch._dynamo import OptimizedModule

        self.set_deep_supervision_enabled(False)
        self.network.eval()

        # Get the actual network (unwrap DDP/compiled if needed)
        if self.is_ddp:
            actual_network = self.network.module
        else:
            actual_network = self.network

        if isinstance(actual_network, OptimizedModule):
            actual_network = actual_network._orig_mod

        # Wrap network for inference
        inference_network = M2FInferenceWrapper(actual_network)

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.manual_initialization(
            inference_network,  # Use wrapped network
            self.plans_manager,
            self.configuration_manager,
            None,
            self.dataset_json,
            self.__class__.__name__,
            self.inference_allowed_mirroring_axes,
        )

        # Call parent's validation logic with our wrapped predictor
        # We need to duplicate the logic here since we can't easily inject the predictor
        import multiprocessing
        import warnings
        from time import sleep
        from os.path import join

        import torch.distributed as dist

        from nnunetv2.utilities.file_path_utilities import (
            check_workers_alive_and_busy,
        )
        from nnunetv2.inference.export_prediction import (
            export_prediction_from_logits,
        )
        from nnunetv2.utilities.label_handling.label_handling import (
            convert_labelmap_to_one_hot,
        )
        from nnunetv2.utilities.helpers import empty_cache
        from batchgenerators.utilities.file_and_folder_operations import (
            maybe_mkdir_p,
        )
        from nnunetv2.paths import nnUNet_preprocessed
        from nnunetv2.configuration import default_num_processes

        with multiprocessing.get_context("spawn").Pool(
            default_num_processes
        ) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, validation_folder_name)
            maybe_mkdir_p(validation_output_folder)

            _, val_keys = self.do_split()
            if self.is_ddp:
                last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1
                val_keys = val_keys[self.local_rank :: dist.get_world_size()]

            dataset_val = self.dataset_class(
                self.preprocessed_dataset_folder,
                val_keys,
                folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            )

            next_stages = self.configuration_manager.next_stage_names

            if next_stages is not None:
                _ = [
                    maybe_mkdir_p(
                        join(self.output_folder_base, "predicted_next_stage", n)
                    )
                    for n in next_stages
                ]

            results = []

            for i, k in enumerate(dataset_val.identifiers):
                proceed = not check_workers_alive_and_busy(
                    segmentation_export_pool, worker_list, results, allowed_num_queued=2
                )
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(
                        segmentation_export_pool,
                        worker_list,
                        results,
                        allowed_num_queued=2,
                    )

                data, _, seg_prev, properties = dataset_val.load_case(k)
                data = data[:]

                if self.is_cascaded:
                    seg_prev = seg_prev[:]
                    data = np.vstack(
                        (
                            data,
                            convert_labelmap_to_one_hot(
                                seg_prev,
                                self.label_manager.foreground_labels,
                                output_dtype=data.dtype,
                            ),
                        )
                    )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                output_filename_truncated = join(validation_output_folder, k)

                prediction = predictor.predict_sliding_window_return_logits(data)
                prediction = prediction.cpu()

                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_logits,
                        (
                            (
                                prediction,
                                properties,
                                self.configuration_manager,
                                self.plans_manager,
                                self.dataset_json,
                                output_filename_truncated,
                                save_probabilities,
                            ),
                        ),
                    )
                )

                if next_stages is not None:
                    for n in next_stages:
                        next_stage_config_manager = (
                            self.plans_manager.get_configuration(n)
                        )
                        expected_preprocessed_folder = join(
                            nnUNet_preprocessed,
                            self.plans_manager.dataset_name,
                            next_stage_config_manager.data_identifier,
                        )
                        dataset_class = infer_dataset_class(
                            expected_preprocessed_folder
                        )
                        try:
                            dataset_next_stage = dataset_class(
                                expected_preprocessed_folder,
                                [k],
                                folder_with_segs_from_previous_stage=None,
                            )
                            _, _, _, properties_next = dataset_next_stage.load_case(k)
                            output_folder_next_stage = join(
                                self.output_folder_base, "predicted_next_stage", n
                            )
                            export_prediction_from_logits(
                                prediction,
                                properties_next,
                                next_stage_config_manager,
                                self.plans_manager,
                                self.dataset_json,
                                join(output_folder_next_stage, k),
                                save_probabilities,
                            )
                        except FileNotFoundError:
                            self.print_to_log_file(
                                f"Skipping {k} for next stage {n}: preprocessed file not found"
                            )

                if self.is_ddp:
                    if i < last_barrier_at_idx:
                        dist.barrier()

            _ = [r.get() for r in results]

        if self.is_ddp:
            dist.barrier()

        if self.local_rank == 0:
            from nnunetv2.evaluation.evaluate_predictions import (
                compute_metrics_on_folder,
            )

            gt_folder = join(
                self.preprocessed_dataset_folder_base, "gt_segmentations"
            )
            metrics = compute_metrics_on_folder(
                gt_folder,
                validation_output_folder,
                join(validation_output_folder, "summary.json"),
                self.plans_manager.image_reader_writer_class(),
                self.dataset_json["file_ending"],
                self.label_manager.foreground_regions
                if self.label_manager.has_regions
                else self.label_manager.foreground_labels,
                self.label_manager.ignore_label,
                chill=True,
            )
            self.print_to_log_file(f"Validation metrics: {metrics}")

        self.set_deep_supervision_enabled(True)
        empty_cache(self.device)

        self.network.train()
