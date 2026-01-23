from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import SegformerConfig
from transformers.models.segformer.modeling_segformer import SegformerDecodeHead

from nnunetv2.model.backbone import create_backbone, get_vit_features

print("model_segformer_0.py")


# ===================== SegFormer Segmentation Model =====================


class SegFormerSegmentationModel(nn.Module):
    """SegFormer-based segmentation model with ViT backbone.

    Uses HuggingFace SegformerDecodeHead with multi-scale feature simulation
    from ViT backbone.
    """

    # Default layer indices for different ViT depths
    LAYER_INDICES = {
        12: [2, 5, 8, 11],  # Base models
        24: [5, 11, 17, 23],  # Large models
    }

    def __init__(
        self,
        backbone_name: str = "dinov3",
        backbone_size: str = "large",
        num_classes: int = 1,
        decoder_hidden_size: int = 256,
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
        self.decoder_hidden_size = decoder_hidden_size

        # Create backbone
        self.backbone, self.backbone_patch_size, _ = create_backbone(
            model_name=backbone_name,
            model_size=backbone_size,
            checkpoint_path=checkpoint_path,
        )

        # Determine feature extraction layers
        num_layers = len(self.backbone.blocks)
        self.feature_indices = self.LAYER_INDICES.get(
            num_layers,
            [
                num_layers // 4 - 1,
                num_layers // 2 - 1,
                3 * num_layers // 4 - 1,
                num_layers - 1,
            ],
        )

        # Get backbone embed dimension
        backbone_embed_dim = self.backbone.embed_dim

        # SegFormer expects 4 feature maps with different channel dimensions
        # ViT produces same-dim features, so we use projections
        hidden_sizes = [backbone_embed_dim] * 4

        # Create SegFormer config
        segformer_config = SegformerConfig(
            hidden_sizes=hidden_sizes,
            num_labels=num_classes,
            decoder_hidden_size=decoder_hidden_size,
        )

        self.decode_head = SegformerDecodeHead(segformer_config)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Enable LoRA if requested
        if use_lora:
            self.enable_lora(
                rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original size for cropping
        _, _, h, w = x.shape

        # Pad input to be divisible by backbone patch size (e.g., 14 for DINOv2)
        ps = self.backbone_patch_size
        pad_h = (ps - h % ps) % ps
        pad_w = (ps - w % ps) % ps
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        padded_size = x.shape[-2:]

        # Extract multi-level features from ViT
        features = get_vit_features(
            self.backbone,
            x,
            indices=self.feature_indices,
        )

        # SegformerDecodeHead expects list of features
        features_list = list(features)

        # Apply SegFormer decode head
        # Output is at H/4 x W/4 resolution
        logits = self.decode_head(features_list)

        # Upsample to padded resolution
        if logits.shape[-2:] != padded_size:
            logits = F.interpolate(
                logits,
                size=padded_size,
                mode="bilinear",
                align_corners=False,
            )

        # Crop to original size
        if pad_h > 0 or pad_w > 0:
            logits = logits[:, :, :h, :w]

        return logits

    def print_trainable_parameters(self, detailed: bool = False) -> None:
        """Print the number of trainable parameters in the model.

        Args:
            detailed: If True, also print breakdown by component (backbone, decode_head)
        """

        def count_params(module: nn.Module) -> Tuple[int, int]:
            """Count trainable and total parameters in a module."""
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total = sum(p.numel() for p in module.parameters())
            return trainable, total

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.parameters())
        percentage = 100 * trainable_params / all_params if all_params > 0 else 0

        print(
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_params:,} || "
            f"trainable%: {percentage:.2f}%"
        )

        if detailed:
            print("-" * 60)
            backbone_trainable, backbone_total = count_params(self.backbone)
            backbone_pct = (
                100 * backbone_trainable / backbone_total if backbone_total > 0 else 0
            )
            print(
                f"  backbone:     {backbone_trainable:>12,} / {backbone_total:>12,} "
                f"({backbone_pct:.2f}%)"
            )

            head_trainable, head_total = count_params(self.decode_head)
            head_pct = 100 * head_trainable / head_total if head_total > 0 else 0
            print(
                f"  decode_head:  {head_trainable:>12,} / {head_total:>12,} "
                f"({head_pct:.2f}%)"
            )

    def enable_lora(
        self,
        rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
    ) -> None:
        """Enable LoRA (Low-Rank Adaptation) for efficient fine-tuning of the backbone.

        Args:
            rank: LoRA rank (lower = fewer params, higher = more expressive)
            lora_alpha: LoRA scaling factor (alpha/rank determines scaling)
            lora_dropout: Dropout probability for LoRA layers
            target_modules: List of module names to apply LoRA to.
                           Default targets attention qkv projections.
        """
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError(
                "LoRA requires the 'peft' library. Install with: pip install peft"
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

        # Ensure decode head remains trainable
        for param in self.decode_head.parameters():
            param.requires_grad = True

        print(f"LoRA enabled with rank={rank}, alpha={lora_alpha}")
        self.print_trainable_parameters(detailed=True)

    def disable_lora(self) -> None:
        """Disable LoRA and merge weights into backbone."""
        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError(
                "LoRA requires the 'peft' library. Install with: pip install peft"
            )

        if isinstance(self.backbone, PeftModel):
            self.backbone = self.backbone.merge_and_unload()
            print("LoRA disabled and weights merged into backbone")
        else:
            print("LoRA is not enabled on this model")


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
    # Decoder-specific kwargs
    hidden_size: int = 256,
    **kwargs,  # Ignore unknown decoder params for forward compatibility
) -> nn.Module:
    """Factory function to create SegFormer segmentation models.

    Args:
        backbone_name: One of "dinov3", "dinov2", "retfound", "visionfm"
        backbone_size: "base" or "large"
        num_classes: Number of segmentation classes
        checkpoint_path: Path to pretrained backbone weights
        freeze_backbone: Whether to freeze backbone weights
        use_lora: Enable LoRA for efficient fine-tuning
        lora_rank: LoRA rank
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout probability for LoRA layers
        lora_target_modules: Module names to apply LoRA to (default: ["qkv"])
        hidden_size: Hidden dimension for SegFormer decoder (default: 256)
        **kwargs: Additional decoder params (ignored for this model)

    Returns:
        SegFormer segmentation model ready for training
    """
    if kwargs:
        print(f"Warning: Ignoring unknown decoder params: {list(kwargs.keys())}")

    return SegFormerSegmentationModel(
        backbone_name=backbone_name,
        backbone_size=backbone_size,
        num_classes=num_classes,
        decoder_hidden_size=hidden_size,
        checkpoint_path=checkpoint_path,
        freeze_backbone=freeze_backbone,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
    )


# Backward compatibility alias
# create_segformer_model = create_segmentation_model
