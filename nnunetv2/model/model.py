from typing import Tuple, Optional, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from transformers import UperNetConfig
from transformers.models.upernet.modeling_upernet import UperNetHead


def create_backbone(
    model_name: str = "dinov3",
    model_size: str = "large",
    checkpoint_path: Optional[str] = None,
) -> Tuple[torch.nn.Module, int, str]:
    model_configs = {
        # timm.model.eva
        "dinov3": {
            "timm_name": f"vit_{model_size}_patch16_dinov3.lvd1689m",
            "patch_size": 16,
            "pretrained": True,
        },
        "dinov2": {
            "timm_name": f"vit_{model_size}_patch14_dinov2.lvd142m",
            "patch_size": 14,
            "pretrained": True,
        },
        "retfound": {
            "timm_name": "vit_large_patch16_224.mae",
            "patch_size": 16,
            "pretrained": False,
        },
        "visionfm": {
            "timm_name": "vit_base_patch16_224.mae",
            "patch_size": 16,
            "pretrained": False,
        },
    }

    if model_name not in model_configs:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from {list(model_configs.keys())}"
        )

    config = model_configs[model_name]

    model = timm.create_model(
        config["timm_name"],
        pretrained=config["pretrained"],
        dynamic_img_size=True,
    )

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        msg = model.load_state_dict(checkpoint["model"], strict=False)
        print(f"Loaded checkpoint: {msg}")

    return model, config["patch_size"], model_name


def get_vit_features(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    indices: Optional[Union[int, List[int]]] = None,
) -> List[torch.Tensor]:
    if hasattr(model, "get_intermediate_layers"):
        feats = model.get_intermediate_layers(
            input_tensor,
            n=indices,
            reshape=True,
            norm=True,
        )
    elif hasattr(model, "forward_intermediates"):
        _, feats = model.forward_intermediates(
            input_tensor,
            indices=indices,
            norm=True,
            output_fmt="NCHW",
            return_prefix_tokens=False,
        )
    else:
        raise RuntimeError("Model does not support feature extraction")

    return feats


# ===================== Segmentation Heads =====================


class ProgressiveUpsampleDecoder(nn.Module):
    """Progressive upsampling decoder that eliminates patch artifacts.

    Upsamples through multiple stages (each 2x), with learned convolutions
    between stages to refine features and smooth patch boundaries.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        num_stages: int = 4,
    ):
        super().__init__()
        self.num_stages = num_stages

        if hidden_dim is None:
            hidden_dim = in_channels // 2

        # Channel progression: in -> hidden -> hidden/2 -> hidden/4 -> ...
        channels = [in_channels]
        current = hidden_dim
        for _ in range(num_stages):
            channels.append(max(current, 32))
            current = current // 2

        self.stages = nn.ModuleList()
        for i in range(num_stages):
            self.stages.append(self._make_block(channels[i], channels[i + 1]))

        self.seg_head = nn.Conv2d(channels[-1], num_classes, 1)

    def _get_num_groups(self, num_channels: int) -> int:
        """Get the largest divisor of num_channels that's <= 32."""
        for g in [32, 16, 8, 4, 2, 1]:
            if num_channels % g == 0:
                return g
        return 1

    def _make_block(self, in_ch: int, out_ch: int) -> nn.Module:
        num_groups = self._get_num_groups(out_ch)
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """Progressive upsampling from patch resolution to target size.

        Args:
            x: Input tensor at patch resolution (B, C, H_patch, W_patch)
            target_size: Desired output size (H, W)

        Returns:
            Segmentation logits at target resolution (B, num_classes, H, W)
        """
        # Progressive 2x upsampling for stages 0 to n-2
        for i in range(len(self.stages) - 1):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = self.stages[i](x)

        # Final stage: upsample to exact target size (handles 14 vs 16 patch size)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        x = self.stages[-1](x)

        return self.seg_head(x)


# ===================== Main Segmentation Model =====================


class ViTSegmentationModel(nn.Module):
    """Complete ViT-based segmentation model using HuggingFace transformers UperNetHead."""

    # Default layer indices for different ViT depths
    LAYER_INDICES = {
        12: [2, 5, 8, 11],  # Base models
        24: [5, 11, 17, 23],  # Large models
    }

    def __init__(
        self,
        backbone_name: str = "dinov3",
        backbone_size: str = "large",
        head_type: str = "upernet",
        num_classes: int = 1,
        embed_dim: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        freeze_backbone: bool = False,
        use_progressive_upsample: bool = False,
    ):
        super().__init__()
        self.head_type = head_type
        self.num_classes = num_classes
        self.use_progressive_upsample = use_progressive_upsample

        # Create backbone
        self.backbone, self.patch_size, _ = create_backbone(
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
        in_channels = [backbone_embed_dim] * len(self.feature_indices)

        # Auto-set embed_dim if not provided
        if embed_dim is None:
            embed_dim = backbone_embed_dim // 2

        # Create head using transformers UperNetHead
        if head_type == "upernet":
            if use_progressive_upsample:
                # UperNetHead outputs embed_dim channels, ProgressiveUpsampleDecoder does final classification
                upernet_config = UperNetConfig(
                    hidden_size=embed_dim,
                    num_labels=embed_dim,  # Output features, not classes
                    pool_scales=[1, 2, 3, 6],
                )
                self.head = UperNetHead(upernet_config, in_channels)
                self.decoder = ProgressiveUpsampleDecoder(
                    in_channels=embed_dim,
                    num_classes=num_classes,
                    hidden_dim=embed_dim // 2,
                )
            else:
                # UperNetHead outputs num_classes directly
                upernet_config = UperNetConfig(
                    hidden_size=embed_dim,
                    num_labels=num_classes,
                    pool_scales=[1, 2, 3, 6],
                )
                self.head = UperNetHead(upernet_config, in_channels)
        else:
            raise ValueError(f"Unknown head type: {head_type}. Choose from ['upernet']")

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        # Extract multi-level features
        features = get_vit_features(
            self.backbone,
            x,
            indices=self.feature_indices,
        )

        # Apply transformers UperNetHead (expects list of features)
        output = self.head(list(features))

        # Upsample to input size
        if self.use_progressive_upsample:
            # Use progressive decoder for smoother upsampling
            output = self.decoder(output, input_size)
        else:
            # Simple bilinear upsampling
            if output.shape[-2:] != input_size:
                output = F.interpolate(
                    output,
                    size=input_size,
                    mode="bilinear",
                    align_corners=False,
                )

        return output


def create_segmentation_model(
    backbone_name: str = "dinov3",
    backbone_size: str = "large",
    head_type: str = "upernet",
    num_classes: int = 1,
    embed_dim: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    freeze_backbone: bool = False,
    use_progressive_upsample: bool = True,
) -> nn.Module:
    """Factory function to create segmentation models.

    Uses HuggingFace transformers UperNetHead for the segmentation decoder.

    Args:
        backbone_name: One of "dinov3", "dinov2", "retfound", "visionfm"
        backbone_size: "base" or "large"
        head_type: "upernet" (uses transformers.models.upernet.UperNetHead)
        num_classes: Number of segmentation classes (1 for binary)
        embed_dim: Hidden dimension for decoder (auto-set if None)
        checkpoint_path: Path to pretrained backbone weights
        freeze_backbone: Whether to freeze backbone weights
        use_progressive_upsample: Use progressive upsampling decoder to eliminate patch artifacts

    Returns:
        Segmentation model ready for training
    """
    return ViTSegmentationModel(
        backbone_name=backbone_name,
        backbone_size=backbone_size,
        head_type=head_type,
        num_classes=num_classes,
        embed_dim=embed_dim,
        checkpoint_path=checkpoint_path,
        freeze_backbone=freeze_backbone,
        use_progressive_upsample=use_progressive_upsample,
    )
