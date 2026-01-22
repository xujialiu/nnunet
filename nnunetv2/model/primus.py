"""Primus model for ViT-based segmentation.

Architecture from: https://arxiv.org/pdf/2503.01835
`Primus: Enforcing Attention Usage for 3D Medical Image Segmentation`
"""

import math
from typing import Tuple, Optional, List

import torch
from torch import nn

from dynamic_network_architectures.architectures.abstract_arch import (
    AbstractDynamicNetworkArchitectures,
)
from dynamic_network_architectures.building_blocks.patch_encode_decode import (
    LayerNormNd,
)
from dynamic_network_architectures.initialization.weight_init import InitWeights_He

from nnunetv2.model.backbone import create_backbone, get_vit_features


class PatchDecode(nn.Module):
    """
    Loosely inspired by SAM decoder
    https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py#L53
    """

    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        out_channels: int,
        norm=LayerNormNd,
        activation=nn.GELU,
    ):
        """
        patch size must be 2^x, so 2, 4, 8, 16, 32, etc. Otherwise we die
        """
        super().__init__()
        assert patch_size > 0
        n = int(math.log2(patch_size))

        assert 2**n == patch_size and n >= 1

        ch = [embed_dim]
        for _ in range(n):
            ch.append(ch[-1] // 2)
        ch.append(out_channels)

        stages = []
        for i in range(n):
            stages.append(
                nn.Sequential(
                    nn.ConvTranspose2d(ch[i], ch[i + 1], kernel_size=2, stride=2),
                    norm(ch[i + 1]),
                    activation(),
                )
            )
        stages.append(nn.Conv2d(ch[-2], ch[-1], kernel_size=1))
        self.decode = nn.Sequential(*stages)

    def forward(self, x):
        """
        Expects input of shape (B, embed_dim, px, py)! This will require you to reshape the output of your transformer!
        """
        return self.decode(x)


class Decoder(nn.Module):
    """Dummy decoder class for nnUNet compatibility."""

    def __init__(self):
        super().__init__()
        self.deep_supervision = False


class PrimusSegmentationModel(nn.Module):
    """Primus segmentation model with ViT backbone and simple patch decoder.

    Architecture as proposed in the Primus paper (https://arxiv.org/pdf/2503.01835)
    `Primus: Enforcing Attention Usage for 3D Medical Image Segmentation`

    Consists of a ViT encoder and a simple patch decoder.
    """

    def __init__(
        self,
        backbone_name: str = "dinov3",
        backbone_size: str = "large",
        num_classes: int = 1,
        checkpoint_path: Optional[str] = None,
        freeze_backbone: bool = False,
        embed_dim: Optional[int] = None,
        patch_embed_size: Optional[int] = None,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Create backbone
        self.backbone, backbone_patch_size, _ = create_backbone(
            model_name=backbone_name,
            model_size=backbone_size,
            checkpoint_path=checkpoint_path,
        )

        # Use specified patch_embed_size or default to backbone's patch_size
        self.patch_size = patch_embed_size if patch_embed_size is not None else backbone_patch_size

        # Get backbone embed dimension
        backbone_embed_dim = self.backbone.embed_dim

        # Use specified embed_dim or default to backbone's embed_dim
        decoder_embed_dim = embed_dim if embed_dim is not None else backbone_embed_dim

        # Add projection layer if embed_dim differs from backbone
        if decoder_embed_dim != backbone_embed_dim:
            self.proj = nn.Conv2d(backbone_embed_dim, decoder_embed_dim, kernel_size=1)
        else:
            self.proj = None

        # Simple patch decoder (upsample from patch resolution to original)
        self.up_projection = PatchDecode(
            self.patch_size,
            decoder_embed_dim,
            num_classes,
            norm=decoder_norm,
            activation=decoder_act,
        )

        # Dummy decoder for nnUNet compatibility
        self.decoder = Decoder()

        # Initialize decoder weights
        self.up_projection.apply(InitWeights_He(1e-2))
        if self.proj is not None:
            self.proj.apply(InitWeights_He(1e-2))

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
        # Extract features from the last layer
        feats = get_vit_features(self.backbone, x, indices=1)
        x = feats[0]

        # Project to decoder embed_dim if needed
        if self.proj is not None:
            x = self.proj(x)

        # Decode to original resolution
        dec_out = self.up_projection(x)
        return dec_out

    def print_trainable_parameters(self, detailed: bool = False) -> None:
        """Print the number of trainable parameters in the model."""

        def count_params(module: nn.Module) -> Tuple[int, int]:
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
                f"  backbone:       {backbone_trainable:>12,} / {backbone_total:>12,} "
                f"({backbone_pct:.2f}%)"
            )

            decoder_trainable, decoder_total = count_params(self.up_projection)
            decoder_pct = (
                100 * decoder_trainable / decoder_total if decoder_total > 0 else 0
            )
            print(
                f"  up_projection:  {decoder_trainable:>12,} / {decoder_total:>12,} "
                f"({decoder_pct:.2f}%)"
            )

    def enable_lora(
        self,
        rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
    ) -> None:
        """Enable LoRA for efficient fine-tuning of the backbone."""
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

        # Ensure decoder remains trainable
        for param in self.up_projection.parameters():
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


# Legacy class for backward compatibility
class Primus(AbstractDynamicNetworkArchitectures):
    """Legacy Primus class for backward compatibility.

    Use PrimusSegmentationModel for new code.
    """

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder=None,
    ):
        super().__init__()

        self.up_projection = PatchDecode(
            patch_embed_size,
            embed_dim,
            num_classes,
            norm=decoder_norm,
            activation=decoder_act,
        )

        self.dino_encoder = dino_encoder
        self.decoder = Decoder()
        self.up_projection.apply(InitWeights_He(1e-2))

    def forward(self, x, ret_mask=False):
        indices = 1
        feats = get_vit_features(self.dino_encoder, x, indices=indices)
        x = feats[0]

        dec_out = self.up_projection(x)
        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")


class PrimusMultiscaleSegmentationModel(nn.Module):
    """Primus with multi-scale feature aggregation.

    Similar to ViT-adapter: extracts features from multiple transformer layers
    and concatenates along channel dimension.
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
        checkpoint_path: Optional[str] = None,
        freeze_backbone: bool = False,
        embed_dim: Optional[int] = None,
        patch_embed_size: Optional[int] = None,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        interaction_indices: Optional[List[int]] = None,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Create backbone
        self.backbone, backbone_patch_size, _ = create_backbone(
            model_name=backbone_name,
            model_size=backbone_size,
            checkpoint_path=checkpoint_path,
        )

        # Use specified patch_embed_size or default to backbone's patch_size
        self.patch_size = patch_embed_size if patch_embed_size is not None else backbone_patch_size

        # Get backbone embed dimension and determine layer indices
        backbone_embed_dim = self.backbone.embed_dim
        num_layers = len(self.backbone.blocks)

        if interaction_indices is None:
            self.interaction_indices = self.LAYER_INDICES.get(
                num_layers,
                [
                    num_layers // 4 - 1,
                    num_layers // 2 - 1,
                    3 * num_layers // 4 - 1,
                    num_layers - 1,
                ],
            )
        else:
            self.interaction_indices = interaction_indices

        num_scales = len(self.interaction_indices)

        # Use specified embed_dim or default to backbone's embed_dim
        decoder_embed_dim = embed_dim if embed_dim is not None else backbone_embed_dim

        # Concatenated features dimension from backbone
        concat_dim = backbone_embed_dim * num_scales

        # Add projection layer if embed_dim differs from backbone
        # Project from concat_dim to decoder_embed_dim * num_scales to maintain structure
        if decoder_embed_dim != backbone_embed_dim:
            self.proj = nn.Conv2d(concat_dim, decoder_embed_dim * num_scales, kernel_size=1)
            decoder_input_dim = decoder_embed_dim * num_scales
        else:
            self.proj = None
            decoder_input_dim = concat_dim

        self.up_projection = PatchDecode(
            self.patch_size,
            decoder_input_dim,
            num_classes,
            norm=decoder_norm,
            activation=decoder_act,
        )

        # Dummy decoder for nnUNet compatibility
        self.decoder = Decoder()

        # Initialize decoder weights
        self.up_projection.apply(InitWeights_He(1e-2))
        if self.proj is not None:
            self.proj.apply(InitWeights_He(1e-2))

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
        # Extract multi-scale features
        feats = get_vit_features(self.backbone, x, indices=self.interaction_indices)

        # Concatenate along channel dimension
        hier = torch.cat(list(feats), dim=1)

        # Project to decoder embed_dim if needed
        if self.proj is not None:
            hier = self.proj(hier)

        # Decode to original resolution
        dec_out = self.up_projection(hier)
        return dec_out

    def print_trainable_parameters(self, detailed: bool = False) -> None:
        """Print the number of trainable parameters in the model."""

        def count_params(module: nn.Module) -> Tuple[int, int]:
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
                f"  backbone:       {backbone_trainable:>12,} / {backbone_total:>12,} "
                f"({backbone_pct:.2f}%)"
            )

            decoder_trainable, decoder_total = count_params(self.up_projection)
            decoder_pct = (
                100 * decoder_trainable / decoder_total if decoder_total > 0 else 0
            )
            print(
                f"  up_projection:  {decoder_trainable:>12,} / {decoder_total:>12,} "
                f"({decoder_pct:.2f}%)"
            )

    def enable_lora(
        self,
        rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
    ) -> None:
        """Enable LoRA for efficient fine-tuning of the backbone."""
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

        # Ensure decoder remains trainable
        for param in self.up_projection.parameters():
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


# Legacy class for backward compatibility
class Primus_Multiscale(AbstractDynamicNetworkArchitectures):
    """Legacy Primus_Multiscale class for backward compatibility.

    Use PrimusMultiscaleSegmentationModel for new code.
    """

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder=None,
        interaction_indices=[1, 2, 3, 4],
    ):
        super().__init__()

        self.up_projection = PatchDecode(
            patch_embed_size,
            embed_dim * len(interaction_indices),
            num_classes,
            norm=decoder_norm,
            activation=decoder_act,
        )

        self.dino_encoder = dino_encoder
        self.decoder = Decoder()
        self.up_projection.apply(InitWeights_He(1e-2))
        self.interaction_indices = interaction_indices

    def forward(self, x, ret_mask=False):
        feats = get_vit_features(self.dino_encoder, x, indices=self.interaction_indices)
        hier = torch.cat(list(feats), dim=1)
        dec_out = self.up_projection(hier)
        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")


# ===================== Factory Function =====================


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
    embed_dim: Optional[int] = None,
    patch_embed_size: Optional[int] = None,
    multiscale: bool = False,
    interaction_indices: Optional[List[int]] = None,
    **kwargs,
) -> nn.Module:
    """Factory function to create Primus segmentation models.

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
        embed_dim: Decoder embedding dimension. If None, uses backbone's embed_dim.
                   If different from backbone, a projection layer is added.
        patch_embed_size: Patch size for decoder upsampling. If None, uses backbone's patch_size.
                          Must be power of 2 (2, 4, 8, 16, 32, etc.)
        multiscale: If True, use PrimusMultiscaleSegmentationModel
        interaction_indices: Layer indices for multiscale feature extraction
        **kwargs: Additional params (ignored for forward compatibility)

    Returns:
        Primus segmentation model ready for training
    """
    if kwargs:
        print(f"Warning: Ignoring unknown decoder params: {list(kwargs.keys())}")

    if multiscale:
        return PrimusMultiscaleSegmentationModel(
            backbone_name=backbone_name,
            backbone_size=backbone_size,
            num_classes=num_classes,
            checkpoint_path=checkpoint_path,
            freeze_backbone=freeze_backbone,
            embed_dim=embed_dim,
            patch_embed_size=patch_embed_size,
            interaction_indices=interaction_indices,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
        )
    else:
        return PrimusSegmentationModel(
            backbone_name=backbone_name,
            backbone_size=backbone_size,
            num_classes=num_classes,
            checkpoint_path=checkpoint_path,
            freeze_backbone=freeze_backbone,
            embed_dim=embed_dim,
            patch_embed_size=patch_embed_size,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
        )
