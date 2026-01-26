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


class UNETRDecoder(nn.Module):
    """U-Net style decoder with skip connections.

    Progressively upsamples and merges with skip connections:
    bottleneck (H/8) -> merge skip3 (H/4) -> merge skip2 (H/2) -> merge skip1 (H) -> logits
    """

    def __init__(
        self,
        decoder_channels: List[int],
        num_classes: int,
        negative_slope: float = 0.01,
    ):
        """
        Args:
            decoder_channels: Channel dimensions [c0, c1, c2, c3, c4] from deep to shallow
                             e.g., [512, 256, 128, 64, 32]
            num_classes: Number of output segmentation classes
            negative_slope: LeakyReLU negative slope
        """
        super().__init__()

        # Stage 4: bottleneck (c0) + skip3 (c1) -> upsample -> conv -> c1
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv4 = ConvBlock(
            decoder_channels[0] + decoder_channels[1],  # concat channels
            decoder_channels[1],
            negative_slope,
        )

        # Stage 3: c1 + skip2 (c2) -> upsample -> conv -> c2
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = ConvBlock(
            decoder_channels[1] + decoder_channels[2],
            decoder_channels[2],
            negative_slope,
        )

        # Stage 2: c2 + skip1 (c3) -> upsample -> conv -> c3
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = ConvBlock(
            decoder_channels[2] + decoder_channels[3],
            decoder_channels[3],
            negative_slope,
        )

        # Stage 1: c3 -> conv -> c4 -> 1x1 conv -> num_classes
        self.conv1 = ConvBlock(decoder_channels[3], decoder_channels[4], negative_slope)
        self.seg_head = nn.Conv2d(decoder_channels[4], num_classes, kernel_size=1)

    def forward(
        self,
        skip1: torch.Tensor,
        skip2: torch.Tensor,
        skip3: torch.Tensor,
        bottleneck: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            skip1: Skip connection at full resolution (B, c3, H, W)
            skip2: Skip connection at 1/2 resolution (B, c2, H/2, W/2)
            skip3: Skip connection at 1/4 resolution (B, c1, H/4, W/4)
            bottleneck: Bottleneck features at 1/8 resolution (B, c0, H/8, W/8)

        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        # Stage 4: upsample bottleneck and merge with skip3
        x = self.up4(bottleneck)
        x = torch.cat([x, skip3], dim=1)
        x = self.conv4(x)

        # Stage 3: upsample and merge with skip2
        x = self.up3(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.conv3(x)

        # Stage 2: upsample and merge with skip1
        x = self.up2(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.conv2(x)

        # Stage 1: final conv and segmentation head
        x = self.conv1(x)
        logits = self.seg_head(x)

        return logits


# Layer indices for multi-scale feature extraction
LAYER_INDICES = {
    12: [2, 5, 8, 11],    # Base models (12 transformer blocks)
    24: [5, 11, 17, 23],  # Large models (24 transformer blocks)
}


class UNETRSegmentationModel(nn.Module):
    """UNETR segmentation model with ViT backbone.

    Architecture from: Hatamizadeh et al. "UNETR: Transformers for 3D Medical Image Segmentation"
    Adapted for 2D segmentation.
    """

    def __init__(
        self,
        backbone_name: str = "dinov3",
        backbone_size: str = "large",
        num_classes: int = 1,
        decoder_channels: Optional[List[int]] = None,
        negative_slope: float = 0.01,
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

        # Default decoder channels: [512, 256, 128, 64, 32]
        if decoder_channels is None:
            decoder_channels = [512, 256, 128, 64, 32]
        self.decoder_channels = decoder_channels

        # Create backbone
        self.backbone, self.backbone_patch_size, _ = create_backbone(
            model_name=backbone_name,
            model_size=backbone_size,
            checkpoint_path=checkpoint_path,
        )

        # Determine feature extraction layers
        num_layers = len(self.backbone.blocks)
        self.feature_indices = LAYER_INDICES.get(
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

        # UNETR encoder (skip connection preparation)
        self.encoder = UNETREncoder(
            backbone_embed_dim=backbone_embed_dim,
            decoder_channels=decoder_channels,
            negative_slope=negative_slope,
        )

        # UNETR decoder (U-Net style)
        self.decoder = UNETRDecoder(
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            negative_slope=negative_slope,
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

    def enable_lora(
        self,
        rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
    ) -> None:
        """Enable LoRA fine-tuning on the backbone.

        Will be implemented in Task 6.
        """
        raise NotImplementedError("LoRA support will be added in Task 6")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original size for cropping
        _, _, h, w = x.shape

        # Pad input to be divisible by backbone patch size
        ps = self.backbone_patch_size
        pad_h = (ps - h % ps) % ps
        pad_w = (ps - w % ps) % ps
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        # Extract multi-level features from ViT
        features = get_vit_features(
            self.backbone,
            x,
            indices=self.feature_indices,
        )

        # Prepare skip connections via UNETR encoder
        skip1, skip2, skip3, bottleneck = self.encoder(list(features))

        # Decode via UNETR decoder
        logits = self.decoder(skip1, skip2, skip3, bottleneck)

        # Crop to original size
        if pad_h > 0 or pad_w > 0:
            logits = logits[:, :, :h, :w]

        return logits
