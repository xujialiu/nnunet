# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from functools import partial
import logging
import os

import numpy as np
from PIL import Image
import torch

import dinov3.distributed as distributed
from dinov3.data import DatasetWithEnumeratedTargets, SamplerType, make_data_loader, make_dataset
from dinov3.eval.segmentation.inference import make_inference
from dinov3.eval.segmentation.metrics import (
    calculate_intersect_and_union,
    calculate_segmentation_metrics,
    restore_original_labels,
)
from dinov3.eval.segmentation.models import build_segmentation_decoder
from dinov3.eval.segmentation.transforms import make_segmentation_eval_transforms
from dinov3.hub.segmentors import dinov3_vit7b16_ms
from dinov3.logging import MetricLogger

logger = logging.getLogger("dinov3")

RESULTS_FILENAME = "results-semantic-segmentation.csv"
MAIN_METRICS = ["mIoU"]


def save_visualization(pred_mask: torch.Tensor, gt_mask: torch.Tensor, output_path: str):
    """Save prediction and ground truth masks side by side as PNG.

    Args:
        pred_mask: Prediction tensor of shape [H, W] with class indices
        gt_mask: Ground truth tensor of shape [H, W] with class indices
        output_path: Path to save the PNG file
    """
    # Convert to numpy
    pred_np = pred_mask.cpu().numpy().astype(np.uint8)
    gt_np = gt_mask.cpu().numpy().astype(np.uint8)

    # Concatenate horizontally: [GT | Pred]
    concat_np = np.concatenate([gt_np, pred_np], axis=1)

    img = Image.fromarray(concat_np, mode='L')
    img.save(output_path)


def evaluate_segmentation_model(
    segmentation_model: torch.nn.Module,
    test_dataloader,
    device,
    eval_res,
    eval_stride,
    decoder_head_type,
    num_classes,
    autocast_dtype,
    max_samples: int = 0,  # 0 means no limit
    num_visualizations: int = 0,  # Number of samples to visualize
    output_dir: str | None = None,  # Directory to save visualizations
    global_step: int = 0,  # Current training iteration for naming
    reduce_zero_label: bool = True,  # Whether to reduce label 0 (for datasets like ADE20K)
    inference_mode: str = "slide",  # Inference mode: "slide" or "whole"
):
    segmentation_model = segmentation_model.to(device)
    segmentation_model.eval()
    all_metric_values = []
    metric_logger = MetricLogger(delimiter="  ")

    # Create visualization directory if needed
    vis_dir = None
    if num_visualizations > 0 and output_dir is not None:
        vis_dir = os.path.join(output_dir, "visualization", f"iter_{global_step}")
        if distributed.get_rank() == 0:
            os.makedirs(vis_dir, exist_ok=True)

    for sample_idx, (batch_img, (_, gt)) in enumerate(metric_logger.log_every(test_dataloader, 10, header="Validation: ")):
        if max_samples > 0 and sample_idx >= max_samples:
            break
        batch_img = [img.to(device).to(dtype=autocast_dtype) for img in batch_img]
        gt = gt.to(device)[0]
        aggregated_preds = torch.zeros(1, num_classes, gt.shape[-2], gt.shape[-1])
        for img_idx, img in enumerate(batch_img):
            aggregated_preds += make_inference(
                img,
                segmentation_model.module,
                inference_mode=inference_mode,
                decoder_head_type=decoder_head_type,
                rescale_to=gt.shape[-2:],
                n_output_channels=num_classes,
                crop_size=(eval_res, eval_res) if inference_mode == "slide" else None,
                stride=(eval_stride, eval_stride) if inference_mode == "slide" else None,
                apply_horizontal_flip=(img_idx and img_idx >= len(batch_img) / 2),
                output_activation=partial(torch.nn.functional.softmax, dim=1),
            )
        aggregated_preds = (aggregated_preds / len(batch_img)).argmax(dim=1, keepdim=True).to(device)

        # Save visualization for first num_visualizations samples (only on rank 0)
        if sample_idx < num_visualizations and vis_dir is not None and distributed.get_rank() == 0:
            pred_mask = aggregated_preds[0, 0]  # [H, W] with class indices
            gt_mask = gt.clone()  # [H, W] with class indices
            if reduce_zero_label:
                pred_mask = restore_original_labels(pred_mask)
                gt_mask = restore_original_labels(gt_mask)
            vis_path = os.path.join(vis_dir, f"sample_{sample_idx:04d}.png")
            save_visualization(pred_mask, gt_mask, vis_path)
            logger.info(f"Saved visualization to {vis_path}")

        # Labels are already reduced in eval transforms, don't reduce again
        intersect_and_union = calculate_intersect_and_union(
            aggregated_preds[0],
            gt,
            num_classes=num_classes,
            reduce_zero_label=False,  # Already reduced in transforms
        )
        all_metric_values.append(intersect_and_union)
        del img, gt, aggregated_preds, intersect_and_union

    all_metric_values = torch.stack(all_metric_values)
    if distributed.is_enabled():
        all_metric_values = torch.cat(distributed.gather_all_tensors((all_metric_values)))
    final_metrics = calculate_segmentation_metrics(
        all_metric_values,
        metrics=["mIoU", "dice", "fscore"],
    )
    final_metrics = {k: round(v.cpu().item() * 100, 2) for k, v in final_metrics.items()}
    logger.info(str(final_metrics))
    return final_metrics


def test_segmentation(backbone, config):
    # 1- construct a segmentation decoder
    if config.load_from == "dinov3_vit7b16_ms":  # torch hub descriptor
        # Load public m2f head checkpoints
        logger.info("Loading the 7B backbone and the M2F adapter with torchhub")
        segmentation_model = dinov3_vit7b16_ms(autocast_dtype=config.model_dtype.autocast_dtype, check_hash=True)
    else:
        segmentation_model = build_segmentation_decoder(
            backbone,
            config.decoder_head.backbone_out_layers,
            config.decoder_head.type,
            hidden_dim=config.decoder_head.hidden_dim,  # Only used for instantiating a M2F head
            num_classes=config.decoder_head.num_classes,
            autocast_dtype=config.model_dtype.autocast_dtype,
            dropout=config.decoder_head.dropout,
        )
        state_dict = torch.load(config.load_from, map_location="cpu")["model"]
        _, _ = segmentation_model.load_state_dict(state_dict, strict=False)
    device = distributed.get_rank()
    segmentation_model = torch.nn.parallel.DistributedDataParallel(segmentation_model.to(device), device_ids=[device])

    # 2- dataloader for testing
    eval_res = config.eval.crop_size
    eval_stride = config.eval.stride
    transforms = make_segmentation_eval_transforms(
        img_size=config.transforms.eval.img_size,
        inference_mode=config.eval.mode,
        use_tta=config.eval.use_tta,
        tta_ratios=config.transforms.eval.tta_ratios,
        mean=config.transforms.mean,
        std=config.transforms.std,
        reduce_zero_label=config.eval.reduce_zero_label,
    )

    test_dataset = DatasetWithEnumeratedTargets(
        make_dataset(
            dataset_str=f"{config.datasets.val}:root={config.datasets.root}",
            transforms=transforms,
        )
    )

    test_sampler_type = None
    if distributed.is_enabled():
        test_sampler_type = SamplerType.DISTRIBUTED

    test_dataloader = make_data_loader(
        dataset=test_dataset,
        batch_size=1,
        num_workers=6,
        sampler_type=test_sampler_type,
        drop_last=False,
        shuffle=False,
        persistent_workers=True,
    )

    # 3- make inference
    return evaluate_segmentation_model(
        segmentation_model=segmentation_model,
        test_dataloader=test_dataloader,
        device=device,
        eval_res=eval_res,
        eval_stride=eval_stride,
        decoder_head_type=config.decoder_head.type,
        num_classes=config.decoder_head.num_classes,
        autocast_dtype=config.model_dtype.autocast_dtype,
        reduce_zero_label=config.eval.reduce_zero_label,
        inference_mode=config.eval.mode,
    )
