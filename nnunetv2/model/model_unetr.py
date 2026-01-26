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


class UNETREncoder(nn.Module):
    """Encoder that prepares skip connections from ViT features.

    Takes 4 ViT feature levels and applies deconv blocks to create
    skip connections at different resolutions:
    - z1 (early): 4 deconv blocks -> full resolution
    - z2: 3 deconv blocks -> 1/2 resolution
    - z3: 2 deconv blocks -> 1/4 resolution
    - z4 (deep): 1 deconv block -> 1/8 resolution (bottleneck)
    """

    def __init__(
        self,
        backbone_embed_dim: int,
        decoder_channels: List[int],
        negative_slope: float = 0.01,
    ):
        """
        Args:
            backbone_embed_dim: ViT embedding dimension (e.g., 1024 for large)
            decoder_channels: Channel dimensions [c0, c1, c2, c3, c4] from deep to shallow
                             e.g., [512, 256, 128, 64, 32]
            negative_slope: LeakyReLU negative slope
        """
        super().__init__()

        # z4 (deepest) -> 1 deconv block -> bottleneck at 1/8 resolution
        self.z4_blocks = nn.Sequential(
            DeconvBlock(backbone_embed_dim, decoder_channels[0], negative_slope),
        )

        # z3 -> 2 deconv blocks -> skip at 1/4 resolution
        self.z3_blocks = nn.Sequential(
            DeconvBlock(backbone_embed_dim, decoder_channels[1], negative_slope),
            DeconvBlock(decoder_channels[1], decoder_channels[1], negative_slope),
        )

        # z2 -> 3 deconv blocks -> skip at 1/2 resolution
        self.z2_blocks = nn.Sequential(
            DeconvBlock(backbone_embed_dim, decoder_channels[2], negative_slope),
            DeconvBlock(decoder_channels[2], decoder_channels[2], negative_slope),
            DeconvBlock(decoder_channels[2], decoder_channels[2], negative_slope),
        )

        # z1 (earliest) -> 4 deconv blocks -> skip at full resolution
        self.z1_blocks = nn.Sequential(
            DeconvBlock(backbone_embed_dim, decoder_channels[3], negative_slope),
            DeconvBlock(decoder_channels[3], decoder_channels[3], negative_slope),
            DeconvBlock(decoder_channels[3], decoder_channels[3], negative_slope),
            DeconvBlock(decoder_channels[3], decoder_channels[3], negative_slope),
        )

    def forward(
        self, features: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: List of 4 ViT features [z1, z2, z3, z4] each (B, C, h, w)

        Returns:
            Tuple of (skip1, skip2, skip3, bottleneck) at resolutions
            (H, H/2, H/4, H/8) relative to padded input
        """
        z1, z2, z3, z4 = features

        bottleneck = self.z4_blocks(z4)  # H/8
        skip3 = self.z3_blocks(z3)       # H/4
        skip2 = self.z2_blocks(z2)       # H/2
        skip1 = self.z1_blocks(z1)       # H

        return skip1, skip2, skip3, bottleneck
