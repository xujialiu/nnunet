"""Utilities for Mask2Former integration with nnUNet."""

import torch
from typing import List, Dict


class DenseToM2FTargetConverter:
    """Convert dense segmentation labels to M2F target format.

    nnUNet uses dense labels: [B, 1, H, W] with class IDs per pixel.
    M2F expects: List[Dict] with {"masks": [N, H, W], "labels": [N]} per batch item.

    For semantic segmentation, each unique class in the image gets one binary mask.
    """

    def __init__(self, ignore_label: int = -1):
        """
        Args:
            ignore_label: Label value to ignore (nnUNet uses -1 for regions, varies by dataset)
        """
        self.ignore_label = ignore_label

    def __call__(self, target: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """Convert batch of dense labels to M2F format.

        Args:
            target: Dense labels [B, 1, H, W] or [B, H, W]

        Returns:
            List of dicts, one per batch item:
                {"masks": [N, H, W], "labels": [N]}
        """
        if target.dim() == 4:
            target = target.squeeze(1)  # [B, H, W]

        batch_targets = []
        for b in range(target.shape[0]):
            single_target = target[b]  # [H, W]
            batch_targets.append(self._convert_single(single_target))

        return batch_targets

    def _convert_single(self, label: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert single dense label to M2F format.

        Args:
            label: Dense label [H, W]

        Returns:
            Dict with "masks" [N, H, W] and "labels" [N]
        """
        # Find unique classes (excluding ignore_label)
        unique_classes = torch.unique(label)
        if self.ignore_label is not None:
            unique_classes = unique_classes[unique_classes != self.ignore_label]

        if len(unique_classes) == 0:
            # Edge case: no valid classes
            H, W = label.shape
            return {
                "masks": torch.zeros((0, H, W), dtype=torch.float32, device=label.device),
                "labels": torch.zeros((0,), dtype=torch.long, device=label.device),
            }

        # Create binary mask for each class
        masks = []
        labels = []
        for class_id in unique_classes:
            mask = (label == class_id).float()
            masks.append(mask)
            labels.append(class_id)

        return {
            "masks": torch.stack(masks, dim=0),  # [N, H, W]
            "labels": torch.stack(labels, dim=0).long(),  # [N]
        }


def m2f_predictions_to_segmentation(
    pred_masks: torch.Tensor,
    pred_logits: torch.Tensor,
    target_size: tuple,
) -> torch.Tensor:
    """Convert M2F predictions to dense segmentation map.

    Args:
        pred_masks: Query mask predictions [B, Q, H_mask, W_mask]
        pred_logits: Query class predictions [B, Q, num_classes+1]
        target_size: Target output size (H, W)

    Returns:
        Dense segmentation [B, num_classes, H, W] as logits
    """
    B, Q, num_classes_plus_1 = pred_logits.shape
    num_classes = num_classes_plus_1 - 1  # Last class is "no object"

    # Softmax over classes (excluding no-object class for segmentation)
    class_probs = torch.softmax(pred_logits[:, :, :-1], dim=-1)  # [B, Q, num_classes]

    # Sigmoid on masks
    mask_probs = torch.sigmoid(pred_masks)  # [B, Q, H_mask, W_mask]

    # Upsample masks to target size
    mask_probs = torch.nn.functional.interpolate(
        mask_probs,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )  # [B, Q, H, W]

    # Combine: for each pixel, aggregate query contributions weighted by mask
    # Result: [B, num_classes, H, W]
    # Method: sum over queries of (mask_prob * class_prob)
    # Reshape for einsum: [B, Q, H, W] x [B, Q, C] -> [B, C, H, W]
    segmentation = torch.einsum("bqhw,bqc->bchw", mask_probs, class_probs)

    return segmentation
