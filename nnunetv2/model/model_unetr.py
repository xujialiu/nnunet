"""UNETR model for ViT-based 2D segmentation.

Architecture from: Hatamizadeh et al. "UNETR: Transformers for 3D Medical Image Segmentation" WACV 2022
Adapted for 2D segmentation with configurable decoder channels.
"""

from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.model.backbone import create_backbone, get_vit_features


class DeconvBlock(nn.Module):
    """Deconvolution block for 2x upsampling with refinement.

    Structure: ConvTranspose2d (2x) -> Conv2d -> GroupNorm -> LeakyReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        num_groups = self._get_num_groups(out_channels)

        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def _get_num_groups(self, num_channels: int) -> int:
        """Get largest divisor of num_channels that's <= 8."""
        for g in [8, 4, 2, 1]:
            if num_channels % g == 0:
                return g
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ConvBlock(nn.Module):
    """Double convolution block for decoder.

    Structure: (Conv2d -> GroupNorm -> LeakyReLU) x2
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        num_groups = self._get_num_groups(out_channels)

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.act1 = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act2 = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def _get_num_groups(self, num_channels: int) -> int:
        """Get largest divisor of num_channels that's <= 8."""
        for g in [8, 4, 2, 1]:
            if num_channels % g == 0:
                return g
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x
