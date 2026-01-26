"""
DPT (Dense Prediction Transformer) Segmentation Model with ViT backbone.

Uses HuggingFace transformers DPTNeck and DPTSemanticSegmentationHead.
Architecture follows the DPT paper: https://arxiv.org/abs/2103.13413
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DPTConfig
from transformers.models.dpt.modeling_dpt import (
    DPTNeck,
    DPTSemanticSegmentationHead,
)

from nnunetv2.model.backbone import create_backbone


class DPTSegmentationModel(nn.Module):
    """
    DPT segmentation model with ViT backbone.

    Combines a Vision Transformer backbone with DPT decoder for
    semantic segmentation. Supports LoRA fine-tuning via peft library.
    """

    # Layer indices for extracting 4 feature levels from ViT
    LAYER_INDICES = {
        12: [2, 5, 8, 11],    # Base models (12 transformer blocks)
        24: [5, 11, 17, 23],  # Large models (24 transformer blocks)
    }

    def __init__(
        self,
        backbone_name: str = "dinov3",
        backbone_size: str = "large",
        num_classes: int = 1,
        fusion_hidden_size: int = 256,
        readout_type: str = "project",
        checkpoint_path: Optional[str] = None,
        freeze_backbone: bool = False,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.fusion_hidden_size = fusion_hidden_size
        self.readout_type = readout_type

        # Create ViT backbone
        self.backbone, self.backbone_patch_size, _ = create_backbone(
            model_name=backbone_name,
            model_size=backbone_size,
            checkpoint_path=checkpoint_path,
        )

        # Determine feature extraction layers based on backbone depth
        num_layers = len(self.backbone.blocks)
        if num_layers not in self.LAYER_INDICES:
            raise ValueError(
                f"Unsupported ViT depth: {num_layers}. "
                f"Supported: {list(self.LAYER_INDICES.keys())}"
            )
        self.feature_indices = self.LAYER_INDICES[num_layers]

        # Get backbone embedding dimension
        self.embed_dim = self.backbone.embed_dim

        # DPT expects same hidden size for all layers (ViT is uniform)
        neck_hidden_sizes = [self.embed_dim] * 4

        # Configure DPT components
        dpt_config = DPTConfig(
            hidden_size=self.embed_dim,
            neck_hidden_sizes=neck_hidden_sizes,
            fusion_hidden_size=fusion_hidden_size,
            reassemble_factors=[4, 2, 1, 0.5],
            readout_type=readout_type,
            num_labels=num_classes,
            head_in_index=-1,
            semantic_classifier_dropout=0.1,
            use_batch_norm_in_fusion_residual=True,
        )

        # DPT decoder components
        self.neck = DPTNeck(dpt_config)
        self.head = DPTSemanticSegmentationHead(dpt_config)

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Optionally enable LoRA
        if use_lora:
            self.enable_lora(
                rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
            )

    def _get_dpt_features(self, x: torch.Tensor):
        """
        Extract features in DPT format with real CLS tokens.

        DPT expects features as (B, seq_len+1, C) with CLS token at position 0.
        Uses forward_intermediates with return_prefix_tokens=True to get actual CLS tokens.
        """
        _, feats = self.backbone.forward_intermediates(
            x,
            indices=self.feature_indices,
            norm=True,
            output_fmt='NCHW',
            return_prefix_tokens=True,
        )

        dpt_feats = []
        patch_h, patch_w = None, None
        for spatial, prefix in feats:
            _, _, H, W = spatial.shape
            patch_h, patch_w = H, W
            # Flatten spatial: (B, C, H, W) -> (B, H*W, C)
            spatial_flat = spatial.flatten(2).transpose(1, 2)
            # Get CLS token (first prefix token, works for DINOv2/DINOv3)
            cls_token = prefix[:, 0:1, :]  # (B, 1, C)
            # Concatenate: [CLS] + [patch tokens]
            feat_with_cls = torch.cat([cls_token, spatial_flat], dim=1)
            dpt_feats.append(feat_with_cls)

        return dpt_feats, patch_h, patch_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        _, _, H, W = x.shape

        # Pad input to be divisible by patch_size
        ps = self.backbone_patch_size
        pad_h = (ps - H % ps) % ps
        pad_w = (ps - W % ps) % ps
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        padded_H, padded_W = x.shape[2], x.shape[3]

        # Extract features with real CLS tokens
        hidden_states_seq, patch_height, patch_width = self._get_dpt_features(x)

        # DPT Neck: reassemble + fusion
        neck_outputs = self.neck(
            hidden_states_seq,
            patch_height=patch_height,
            patch_width=patch_width,
        )

        # DPT Head: final segmentation
        logits = self.head(neck_outputs)

        # Upsample to padded size (head outputs at stride 2)
        logits = F.interpolate(
            logits, size=(padded_H, padded_W), mode="bilinear", align_corners=False
        )

        # Crop to original size
        if pad_h > 0 or pad_w > 0:
            logits = logits[:, :, :H, :W]

        return logits

    def print_trainable_parameters(self, detailed: bool = False) -> None:
        """Print count of trainable vs total parameters."""
        trainable_params = 0
        total_params = 0

        component_params = {
            "backbone": {"trainable": 0, "total": 0},
            "neck": {"trainable": 0, "total": 0},
            "head": {"trainable": 0, "total": 0},
        }

        for name, param in self.named_parameters():
            total_params += param.numel()
            component = name.split(".")[0]

            if component in component_params:
                component_params[component]["total"] += param.numel()
                if param.requires_grad:
                    component_params[component]["trainable"] += param.numel()
                    trainable_params += param.numel()
            else:
                if param.requires_grad:
                    trainable_params += param.numel()

        print(f"\nModel Parameter Summary:")
        print(f"{'='*50}")
        print(f"Total parameters:     {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable %:          {100 * trainable_params / total_params:.2f}%")

        if detailed:
            print(f"\n{'Component':<15} {'Trainable':>15} {'Total':>15} {'%':>10}")
            print(f"{'-'*55}")
            for comp, counts in component_params.items():
                pct = (
                    100 * counts["trainable"] / counts["total"]
                    if counts["total"] > 0
                    else 0
                )
                print(
                    f"{comp:<15} {counts['trainable']:>15,} "
                    f"{counts['total']:>15,} {pct:>9.2f}%"
                )

    def enable_lora(
        self,
        rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
    ) -> None:
        """Enable LoRA fine-tuning on the backbone."""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError(
                "peft is required for LoRA. Install with: pip install peft"
            )

        if target_modules is None:
            target_modules = ["qkv"]

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )

        self.backbone = get_peft_model(self.backbone, lora_config)

        # Ensure decoder remains trainable
        for param in self.neck.parameters():
            param.requires_grad = True
        for param in self.head.parameters():
            param.requires_grad = True

        print(f"LoRA enabled with rank={rank}, alpha={lora_alpha}")
        self.print_trainable_parameters()

    def disable_lora(self) -> None:
        """Merge LoRA weights and disable LoRA."""
        try:
            from peft import PeftModel
        except ImportError:
            return

        if isinstance(self.backbone, PeftModel):
            self.backbone = self.backbone.merge_and_unload()
            print("LoRA weights merged and unloaded")


def create_segmentation_model(
    backbone_name: str = "dinov3",
    backbone_size: str = "large",
    num_classes: int = 1,
    checkpoint_path: Optional[str] = None,
    freeze_backbone: bool = False,
    use_lora: bool = True,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[List[str]] = None,
    # Decoder-specific kwargs (from config.model.decoder)
    fusion_hidden_size: int = 256,
    readout_type: str = "project",
    **kwargs,
) -> nn.Module:
    """
    Factory function to create DPT segmentation model.

    This function is called by nnUNetTrainer_vit with parameters from YAML config.

    Args:
        backbone_name: ViT backbone type ("dinov3", "dinov2", "retfound", "visionfm")
        backbone_size: Model size ("base" or "large")
        num_classes: Number of segmentation classes
        checkpoint_path: Optional path to backbone weights
        freeze_backbone: Whether to freeze backbone (overridden if use_lora=True)
        use_lora: Whether to use LoRA fine-tuning
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha scaling
        lora_dropout: LoRA dropout
        lora_target_modules: Target modules for LoRA
        fusion_hidden_size: DPT fusion hidden dimension (default 256)
        readout_type: How to handle CLS token ("project", "add", or "ignore")
        **kwargs: Additional kwargs (ignored with warning)

    Returns:
        DPTSegmentationModel instance
    """
    if kwargs:
        print(f"Warning: Unknown kwargs passed to create_segmentation_model: {kwargs}")

    return DPTSegmentationModel(
        backbone_name=backbone_name,
        backbone_size=backbone_size,
        num_classes=num_classes,
        fusion_hidden_size=fusion_hidden_size,
        readout_type=readout_type,
        checkpoint_path=checkpoint_path,
        freeze_backbone=freeze_backbone,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
    )
