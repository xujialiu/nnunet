"""ViT backbone creation and feature extraction utilities."""

from typing import Tuple, Optional, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


MODEL_CONFIGS = {
    "dinov3": {
        "timm_name_template": "vit_{size}_patch16_dinov3.lvd1689m",
        "patch_size": 16,
        "pretrained": True,
    },
    "dinov2": {
        "timm_name_template": "vit_{size}_patch14_dinov2.lvd142m",
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


def create_backbone(
    model_name: str = "dinov3",
    model_size: str = "large",
    checkpoint_path: Optional[str] = None,
) -> Tuple[torch.nn.Module, int, str]:
    """Create ViT backbone from timm.

    Args:
        model_name: One of "dinov3", "dinov2", "retfound", "visionfm"
        model_size: Size variant - "small", "base", or "large"
        checkpoint_path: Optional path to custom weights

    Returns:
        Tuple of (model, patch_size, model_name)
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from {list(MODEL_CONFIGS.keys())}"
        )

    config = MODEL_CONFIGS[model_name]

    # Handle template-based model names (dinov3, dinov2) vs fixed names (retfound, visionfm)
    if "timm_name_template" in config:
        timm_name = config["timm_name_template"].format(size=model_size)
    else:
        timm_name = config["timm_name"]

    model = timm.create_model(
        timm_name,
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
    """Extract intermediate features from ViT backbone.

    Args:
        model: ViT backbone model
        input_tensor: Input image tensor (B, C, H, W)
        indices: Layer indices to extract features from

    Returns:
        List of feature tensors at specified layers
    """
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
            self.stages.append(self._make_block(channels[i], channels[i + 1], dropout))

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
