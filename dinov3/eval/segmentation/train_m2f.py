# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
Mask2Former training script for semantic segmentation.

This is a standalone training script for M2F decoder, separate from train.py
which handles linear head training. Follows the same patterns as train.py.
"""

from functools import partial
import logging
import numpy as np
import os
import random

import torch
import torch.distributed as dist

from dinov3.data import DatasetWithEnumeratedTargets, SamplerType, make_data_loader, make_dataset
import dinov3.distributed as distributed
from dinov3.eval.segmentation.eval import evaluate_segmentation_model
from dinov3.eval.segmentation.mask_classification_loss import MaskClassificationLoss
from dinov3.eval.segmentation.metrics import SEGMENTATION_METRICS
from dinov3.eval.segmentation.models import build_segmentation_decoder, log_trainable_parameters
from dinov3.eval.segmentation.schedulers import build_scheduler
from dinov3.eval.segmentation.transforms import (
    make_segmentation_eval_transforms,
    make_segmentation_train_transforms,
)
from dinov3.logging import MetricLogger, SmoothedValue

logger = logging.getLogger("dinov3")


class InfiniteDataloader:
    """Wraps a dataloader to iterate infinitely, incrementing epoch on each cycle."""

    def __init__(self, dataloader: torch.utils.data.DataLoader):
        self.dataloader = dataloader
        self.data_iterator = iter(dataloader)
        self.sampler = dataloader.sampler
        if not hasattr(self.sampler, "epoch"):
            self.sampler.epoch = 0

    def __iter__(self):
        return self

    def __len__(self) -> int:
        return len(self.dataloader)

    def __next__(self):
        try:
            data = next(self.data_iterator)
        except StopIteration:
            self.sampler.epoch += 1
            self.data_iterator = iter(self.dataloader)
            data = next(self.data_iterator)
        return data


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed.
    """
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def collate_m2f_batch(batch):
    """Collate function for M2F training with variable-length targets.

    For semantic segmentation, each image has a different number of classes present,
    so targets have variable N dimension. This function:
    - Stacks images normally
    - Keeps targets as a list of dicts

    Args:
        batch: List of (image, (index, target_dict)) from DatasetWithEnumeratedTargets

    Returns:
        (images, targets): images [B, C, H, W], targets List[dict]
    """
    images = []
    targets = []

    for item in batch:
        # DatasetWithEnumeratedTargets returns: (image, (index, target))
        # We need to extract the target dict from the (index, target) tuple
        image = item[0]
        index_and_target = item[1]
        target = index_and_target[1]  # Extract target dict from (index, target)

        images.append(image)
        targets.append(target)

    return torch.stack(images, dim=0), targets


def validate_m2f(
    segmentation_model: torch.nn.Module,
    val_dataloader,
    device,
    autocast_dtype,
    eval_res,
    eval_stride,
    num_classes,
    global_step,
    metric_to_save,
    current_best_metric_to_save_value,
    max_val_samples: int = 0,  # 0 means no limit
    num_visualizations: int = 0,  # Number of samples to visualize
    output_dir: str | None = None,  # Directory to save visualizations
    reduce_zero_label: bool = True,  # Whether to reduce label 0 (for datasets like ADE20K)
    inference_mode: str = "slide",  # Inference mode: "slide" or "whole"
):
    """Run validation and return metrics.

    Uses evaluate_segmentation_model from eval.py which handles M2F inference correctly.
    """
    new_metric_values_dict = evaluate_segmentation_model(
        segmentation_model,
        val_dataloader,
        device,
        eval_res,
        eval_stride,
        decoder_head_type="m2f",  # Hardcoded for M2F
        num_classes=num_classes,
        autocast_dtype=autocast_dtype,
        max_samples=max_val_samples,
        num_visualizations=num_visualizations,
        output_dir=output_dir,
        global_step=global_step,
        reduce_zero_label=reduce_zero_label,
        inference_mode=inference_mode,
    )
    logger.info(f"Step {global_step}: {new_metric_values_dict}")
    # Put decoder back in train mode (backbone stays in eval mode)
    segmentation_model.module.segmentation_model[1].train()
    is_better = new_metric_values_dict[metric_to_save] > current_best_metric_to_save_value
    return is_better, new_metric_values_dict


def train_step_m2f(
    segmentation_model: torch.nn.Module,
    batch,
    device,
    scaler,
    optimizer,
    optimizer_gradient_clip,
    scheduler,
    criterion: MaskClassificationLoss,
    model_dtype,
    global_step,
):
    """Training step for Mask2Former decoder.

    Handles:
    - Dict outputs from M2F head: {pred_logits, pred_masks, aux_outputs}
    - List[dict] targets from SemanticToM2FTargets transform
    - Deep supervision via auxiliary outputs (9 decoder layers)
    """
    # a) load batch - targets is List[dict] from collate_m2f_batch
    batch_img, targets = batch
    batch_img = batch_img.to(device)
    # Move target tensors to device
    targets = [
        {"masks": t["masks"].to(device), "labels": t["labels"].to(device)}
        for t in targets
    ]
    optimizer.zero_grad(set_to_none=True)

    # b) forward pass
    with torch.autocast("cuda", dtype=model_dtype, enabled=model_dtype is not None):
        pred = segmentation_model(batch_img)
        # pred = {
        #     "pred_logits": [B, 100, num_classes+1],
        #     "pred_masks": [B, 100, H, W],
        #     "aux_outputs": [9 dicts with same structure]
        # }

    # c) compute loss for final output
    # Cast predictions to float32 for loss computation (Hungarian matcher requires consistent dtypes)
    losses = criterion(
        masks_queries_logits=pred["pred_masks"].float(),
        targets=targets,
        class_queries_logits=pred["pred_logits"].float(),
    )

    # d) compute auxiliary losses for deep supervision
    for i, aux_output in enumerate(pred.get("aux_outputs", [])):
        aux_losses = criterion(
            masks_queries_logits=aux_output["pred_masks"].float(),
            targets=targets,
            class_queries_logits=aux_output["pred_logits"].float(),
        )
        for key, value in aux_losses.items():
            losses[f"{key}_{i}"] = value

    # e) aggregate weighted losses using criterion's loss_total method
    # Note: We pass a no-op log function since we log separately
    loss_total = criterion.loss_total(losses, log_fn=lambda *args, **kwargs: None)

    # f) optimization
    if scaler is not None:
        scaler.scale(loss_total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(segmentation_model.module.parameters(), optimizer_gradient_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(segmentation_model.module.parameters(), optimizer_gradient_clip)
        optimizer.step()

    if global_step > 0:  # Inheritance from mmcv scheduler behavior
        scheduler.step()

    return loss_total


def train_m2f_segmentation(backbone, config):
    """Main training function for Mask2Former semantic segmentation.

    Args:
        backbone: DINOv3 backbone model (from load_model_and_context in run.py)
        config: SegmentationConfig with M2F-specific settings

    Similar structure to train_segmentation() but specialized for M2F:
    - Uses MaskClassificationLoss instead of MultiSegmentationLoss
    - Uses collate_m2f_batch for variable-length targets
    - Handles dict outputs from M2F head
    """
    assert config.decoder_head.type == "m2f", f"This script is for M2F training, got {config.decoder_head.type}"

    # 1- Build the segmentation decoder with M2F head
    logger.info("Initializing the M2F segmentation model")
    segmentation_model = build_segmentation_decoder(
        backbone,
        config.decoder_head.backbone_out_layers,
        "m2f",
        num_classes=config.decoder_head.num_classes,
        autocast_dtype=config.model_dtype.autocast_dtype,
        dropout=config.decoder_head.dropout,
        hidden_dim=config.decoder_head.hidden_dim,
        lora_enabled=config.lora.enabled,
    )

    # Log trainable parameters before DDP wrapping
    logger.info(f"LoRA enabled: {config.lora.enabled}")
    log_trainable_parameters(segmentation_model, logger)
    global_device = distributed.get_rank()
    local_device = torch.cuda.current_device()
    segmentation_model = torch.nn.parallel.DistributedDataParallel(
        segmentation_model.to(local_device), device_ids=[local_device]
    )
    model_parameters = filter(lambda p: p.requires_grad, segmentation_model.parameters())
    logger.info(f"Number of trainable parameters: {sum(p.numel() for p in model_parameters)}")

    # 2- Create data transforms + dataloaders
    train_transforms = make_segmentation_train_transforms(
        img_size=config.transforms.train.img_size,
        random_img_size_ratio_range=config.transforms.train.random_img_size_ratio_range,
        crop_size=config.transforms.train.crop_size,
        flip_prob=config.transforms.train.flip_prob,
        reduce_zero_label=config.eval.reduce_zero_label,
        mean=config.transforms.mean,
        std=config.transforms.std,
        convert_to_m2f_format=True,  # M2F-specific: convert labels to per-class binary masks
        num_classes=config.decoder_head.num_classes,
    )
    val_transforms = make_segmentation_eval_transforms(
        img_size=config.transforms.eval.img_size,
        inference_mode=config.eval.mode,
        use_tta=config.eval.use_tta,
        tta_ratios=config.transforms.eval.tta_ratios,
        mean=config.transforms.mean,
        std=config.transforms.std,
        reduce_zero_label=config.eval.reduce_zero_label,
    )

    train_dataset = DatasetWithEnumeratedTargets(
        make_dataset(
            dataset_str=f"{config.datasets.train}:root={config.datasets.root}",
            transforms=train_transforms,
        )
    )
    train_sampler_type = None
    if distributed.is_enabled():
        train_sampler_type = SamplerType.DISTRIBUTED
    init_fn = partial(
        worker_init_fn, num_workers=config.num_workers, rank=global_device, seed=config.seed + global_device
    )
    train_dataloader = InfiniteDataloader(
        make_data_loader(
            dataset=train_dataset,
            batch_size=config.bs,
            num_workers=config.num_workers,
            sampler_type=train_sampler_type,
            shuffle=True,
            persistent_workers=False,
            worker_init_fn=init_fn,
            collate_fn=collate_m2f_batch,  # M2F-specific: handles variable-length targets
        )
    )

    val_dataset = DatasetWithEnumeratedTargets(
        make_dataset(
            dataset_str=f"{config.datasets.val}:root={config.datasets.root}",
            transforms=val_transforms,
        )
    )
    val_sampler_type = None
    if distributed.is_enabled():
        val_sampler_type = SamplerType.DISTRIBUTED
    val_dataloader = make_data_loader(
        dataset=val_dataset,
        batch_size=1,
        num_workers=config.num_workers,
        sampler_type=val_sampler_type,
        drop_last=False,
        shuffle=False,
        persistent_workers=True,
    )

    # 3- Define and create scaler, optimizer, scheduler, loss
    scaler = None
    if config.model_dtype.autocast_dtype is not None:
        scaler = torch.amp.GradScaler("cuda")

    optimizer = torch.optim.AdamW(
        [
            {
                "params": filter(lambda p: p.requires_grad, segmentation_model.parameters()),
                "lr": config.optimizer.lr,
                "betas": (config.optimizer.beta1, config.optimizer.beta2),
                "weight_decay": config.optimizer.weight_decay,
            }
        ]
    )
    scheduler = build_scheduler(
        config.scheduler.type,
        optimizer=optimizer,
        lr=config.optimizer.lr,
        total_iter=config.scheduler.total_iter,
        constructor_kwargs=config.scheduler.constructor_kwargs,
    )

    # M2F-specific loss function
    criterion = MaskClassificationLoss(
        num_points=config.m2f_train.num_points,
        oversample_ratio=config.m2f_train.oversample_ratio,
        importance_sample_ratio=config.m2f_train.importance_sample_ratio,
        mask_coefficient=config.m2f_train.mask_coefficient,
        dice_coefficient=config.m2f_train.dice_coefficient,
        class_coefficient=config.m2f_train.class_coefficient,
        num_labels=config.decoder_head.num_classes,
        no_object_coefficient=config.m2f_train.no_object_coefficient,
    ).to(local_device)  # Move to GPU for empty_weight buffer

    total_iter = config.scheduler.total_iter
    global_step = 0
    global_best_metric_values = {metric: 0.0 for metric in SEGMENTATION_METRICS}

    # 4- Training loop
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("loss", SmoothedValue(window_size=4, fmt="{value:.3f}"))
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.2e}"))
    for batch in metric_logger.log_every(
        train_dataloader,
        10,
        header="Train M2F: ",
        start_iteration=global_step,
        n_iterations=total_iter,
    ):
        if global_step >= total_iter:
            break
        loss = train_step_m2f(
            segmentation_model,
            batch,
            local_device,
            scaler,
            optimizer,
            config.optimizer.gradient_clip,
            scheduler,
            criterion,
            config.model_dtype.autocast_dtype,
            global_step,
        )
        global_step += 1
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        # Periodic validation
        if global_step % config.eval.eval_interval == 0:
            dist.barrier()
            is_better, best_metric_values_dict = validate_m2f(
                segmentation_model,
                val_dataloader,
                local_device,
                config.model_dtype.autocast_dtype,
                config.eval.crop_size,
                config.eval.stride,
                config.decoder_head.num_classes,
                global_step,
                config.metric_to_save,
                global_best_metric_values[config.metric_to_save],
                max_val_samples=config.eval.max_val_samples,
                num_visualizations=config.eval.num_visualizations,
                output_dir=config.output_dir,
                reduce_zero_label=config.eval.reduce_zero_label,
                inference_mode=config.eval.mode,
            )
            if is_better:
                logger.info(f"New best metrics at Step {global_step}: {best_metric_values_dict}")
                global_best_metric_values = best_metric_values_dict

    # Final validation if total_iter not divisible by eval_interval
    if total_iter % config.eval.eval_interval:
        is_better, best_metric_values_dict = validate_m2f(
            segmentation_model,
            val_dataloader,
            local_device,
            config.model_dtype.autocast_dtype,
            config.eval.crop_size,
            config.eval.stride,
            config.decoder_head.num_classes,
            global_step,
            config.metric_to_save,
            global_best_metric_values[config.metric_to_save],
            max_val_samples=config.eval.max_val_samples,
            num_visualizations=config.eval.num_visualizations,
            output_dir=config.output_dir,
            reduce_zero_label=config.eval.reduce_zero_label,
            inference_mode=config.eval.mode,
        )
        if is_better:
            logger.info(f"New best metrics at Step {global_step}: {best_metric_values_dict}")
            global_best_metric_values = best_metric_values_dict

    logger.info("M2F Training is done!")

    # Save final model - save decoder and adapter (not backbone, unless LoRA)
    # Model structure: segmentation_model.segmentation_model = Sequential(adapter, decoder)
    # adapter = segmentation_model.0, decoder = segmentation_model.1
    # We save: all of decoder + adapter's trainable parts (not backbone weights, except LoRA)
    state_dict = segmentation_model.module.state_dict()
    save_dict = {}

    for k, v in state_dict.items():
        # Save decoder head (segmentation_model.1.*)
        if "segmentation_model.1" in k:
            save_dict[k] = v
        # Save adapter interaction layers but NOT frozen backbone (unless LoRA)
        elif "segmentation_model.0" in k:
            if "backbone" not in k:
                # Always save non-backbone adapter parts
                save_dict[k] = v
            elif config.lora.enabled and "lora_" in k:
                # Save LoRA weights from backbone
                save_dict[k] = v

    if config.lora.enabled:
        lora_keys = [k for k in save_dict.keys() if "lora_" in k]
        logger.info(f"Saving LoRA weights: {lora_keys[:5]}...")

    torch.save(
        {
            "model": save_dict,
            "optimizer": optimizer.state_dict(),
            "lora_config": config.lora if config.lora.enabled else None,
        },
        os.path.join(config.output_dir, "model_final.pth"),
    )
    logger.info(f"Final best metrics: {global_best_metric_values}")
    return global_best_metric_values
