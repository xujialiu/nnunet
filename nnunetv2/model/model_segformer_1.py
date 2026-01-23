from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import SegformerConfig
from transformers.models.segformer.modeling_segformer import SegformerDecodeHead

from nnunetv2.model.backbone import create_backbone, get_vit_features


class ProgressiveUpsampler(nn.Module):
    """Progressive upsampling module that replaces direct bilinear interpolation.

    Upsamples through multiple 2x stages with learned convolutions between stages
    to refine features and produce smoother outputs.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_stages: int = 2,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """Initialize progressive upsampler.

        Args:
            in_channels: Number of input channels (from SegFormer decode head)
            num_classes: Number of output segmentation classes
            num_stages: Number of 2x upsampling stages (2 stages = 4x total)
            hidden_dim: Hidden dimension for intermediate layers
            dropout: Dropout probability
        """
        super().__init__()
        self.num_stages = num_stages

        if hidden_dim is None:
            hidden_dim = in_channels

        # Build channel progression
        channels = [in_channels]
        current = hidden_dim
        for _ in range(num_stages):
            channels.append(max(current, 32))
            current = current // 2

        # Build upsampling stages
        self.stages = nn.ModuleList()
        for i in range(num_stages):
            self.stages.append(
                self._make_block(channels[i], channels[i + 1], dropout)
            )

        # Final segmentation head
        self.seg_head = nn.Conv2d(channels[-1], num_classes, kernel_size=1)

    def _get_num_groups(self, num_channels: int) -> int:
        """Get largest divisor of num_channels that's <= 32."""
        for g in [32, 16, 8, 4, 2, 1]:
            if num_channels % g == 0:
                return g
        return 1

    def _make_block(self, in_ch: int, out_ch: int, dropout: float) -> nn.Module:
        num_groups = self._get_num_groups(out_ch)
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """Progressive upsampling to target size.

        Args:
            x: Input tensor at reduced resolution (B, C, H/4, W/4)
            target_size: Desired output size (H, W)

        Returns:
            Segmentation logits at target resolution (B, num_classes, H, W)
        """
        # Progressive 2x upsampling through stages
        for stage in self.stages:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = stage(x)

        # Final interpolation to exact target size if needed
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

        return self.seg_head(x)


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
        # Output features from decode head (not final logits)
        segformer_config = SegformerConfig(
            hidden_sizes=hidden_sizes,
            num_labels=decoder_hidden_size,  # Output features, not logits
            decoder_hidden_size=decoder_hidden_size,
        )

        self.decode_head = SegformerDecodeHead(segformer_config)

        # Progressive upsampler: 2 stages of 2x = 4x total upsampling
        self.upsampler = ProgressiveUpsampler(
            in_channels=decoder_hidden_size,
            num_classes=num_classes,
            num_stages=2,
            hidden_dim=decoder_hidden_size,
        )

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
        decoder_features = self.decode_head(features_list)

        # Progressive upsampling to padded resolution
        logits = self.upsampler(decoder_features, padded_size)

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

            upsampler_trainable, upsampler_total = count_params(self.upsampler)
            upsampler_pct = (
                100 * upsampler_trainable / upsampler_total if upsampler_total > 0 else 0
            )
            print(
                f"  upsampler:    {upsampler_trainable:>12,} / {upsampler_total:>12,} "
                f"({upsampler_pct:.2f}%)"
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

        # Ensure decode head and upsampler remain trainable
        for param in self.decode_head.parameters():
            param.requires_grad = True
        for param in self.upsampler.parameters():
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
