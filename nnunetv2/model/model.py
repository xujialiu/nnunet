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
    Supports skip connections from backbone features for better detail preservation.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        num_stages: int = 4,
        dropout: float = 0.1,
        backbone_channels: Optional[List[int]] = None,
    ):
        """Initialize the progressive upsample decoder.

        Args:
            in_channels: Number of input channels from UperNet head
            num_classes: Number of output segmentation classes
            hidden_dim: Hidden dimension for decoder (auto-set if None)
            num_stages: Number of upsampling stages
            dropout: Dropout probability for regularization
            backbone_channels: List of channel dims from backbone features for skip connections
        """
        super().__init__()
        self.num_stages = num_stages
        self.dropout = dropout
        self.use_skip_connections = backbone_channels is not None

        if hidden_dim is None:
            hidden_dim = in_channels // 2

        # Channel progression: in -> hidden -> hidden/2 -> hidden/4 -> ...
        channels = [in_channels]
        current = hidden_dim
        for _ in range(num_stages):
            channels.append(max(current, 32))
            current = current // 2

        # Skip connection projections (project backbone features to decoder channels)
        # We use the last (num_stages - 1) backbone features as skip connections
        if self.use_skip_connections and backbone_channels:
            self.skip_projections = nn.ModuleList()
            # Skip connections are used for stages 0 to num_stages-2
            # backbone_channels is ordered shallow to deep, we reverse to match decoder order
            reversed_backbone = list(reversed(backbone_channels))
            for i in range(min(num_stages - 1, len(reversed_backbone))):
                # Project backbone channels to match decoder stage output
                self.skip_projections.append(
                    nn.Conv2d(reversed_backbone[i], channels[i + 1], 1, bias=False)
                )
            # Pad with None if backbone has fewer features than decoder stages
            while len(self.skip_projections) < num_stages - 1:
                self.skip_projections.append(None)

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
            nn.Dropout2d(self.dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        target_size: Tuple[int, int],
        backbone_features: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Progressive upsampling from patch resolution to target size.

        Args:
            x: Input tensor at patch resolution (B, C, H_patch, W_patch)
            target_size: Desired output size (H, W)
            backbone_features: Optional list of backbone features for skip connections
                              (ordered from shallow to deep layers)

        Returns:
            Segmentation logits at target resolution (B, num_classes, H, W)
        """
        # Reverse backbone features so deepest is first (matches decoder order)
        skip_features = None
        if self.use_skip_connections and backbone_features is not None:
            skip_features = list(reversed(backbone_features))

        # Progressive 2x upsampling for stages 0 to n-2
        for i in range(len(self.stages) - 1):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = self.stages[i](x)

            # Add skip connection if available
            if skip_features is not None and i < len(self.skip_projections):
                skip_proj = self.skip_projections[i]
                if skip_proj is not None and i < len(skip_features):
                    skip_feat = skip_features[i]
                    # Resize skip feature to match current decoder resolution
                    skip_feat = F.interpolate(
                        skip_feat,
                        size=x.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    skip_feat = skip_proj(skip_feat)
                    x = x + skip_feat

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
        decoder_dropout: float = 0.1,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
    ):
        super().__init__()
        self.head_type = head_type
        self.num_classes = num_classes
        self.use_progressive_upsample = use_progressive_upsample
        self.decoder_dropout = decoder_dropout

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
                    dropout=decoder_dropout,
                    backbone_channels=in_channels,  # For skip connections
                )
            else:
                # Direct class prediction without progressive upsampling
                upernet_config = UperNetConfig(
                    hidden_size=embed_dim,
                    num_labels=num_classes,  # Direct class prediction
                    pool_scales=[1, 2, 3, 6],
                )
                self.head = UperNetHead(upernet_config, in_channels)
        else:
            raise ValueError(f"Unknown head type: {head_type}. Choose from ['upernet']")

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
        input_size = x.shape[-2:]

        # Extract multi-level features
        features = get_vit_features(
            self.backbone,
            x,
            indices=self.feature_indices,
        )

        # Apply transformers UperNetHead (expects list of features)
        features_list = list(features)
        output = self.head(features_list)

        # Upsample to input size
        if self.use_progressive_upsample:
            # Use progressive decoder for smoother upsampling with skip connections
            output = self.decoder(output, input_size, backbone_features=features_list)
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

    def print_trainable_parameters(self, detailed: bool = False) -> None:
        """Print the number of trainable parameters in the model.

        Args:
            detailed: If True, also print breakdown by component (backbone, head, decoder)
        """

        def count_params(module: nn.Module) -> Tuple[int, int]:
            """Count trainable and total parameters in a module."""
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total = sum(p.numel() for p in module.parameters())
            return trainable, total

        trainable_params = 0
        all_params = 0
        for param in self.parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        percentage = 100 * trainable_params / all_params if all_params > 0 else 0

        print(
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_params:,} || "
            f"trainable%: {percentage:.2f}%"
        )

        if detailed:
            print("-" * 60)
            # Backbone
            backbone_trainable, backbone_total = count_params(self.backbone)
            backbone_pct = (
                100 * backbone_trainable / backbone_total if backbone_total > 0 else 0
            )
            print(
                f"  backbone:  {backbone_trainable:>12,} / {backbone_total:>12,} "
                f"({backbone_pct:.2f}%)"
            )

            # Head
            head_trainable, head_total = count_params(self.head)
            head_pct = 100 * head_trainable / head_total if head_total > 0 else 0
            print(
                f"  head:      {head_trainable:>12,} / {head_total:>12,} "
                f"({head_pct:.2f}%)"
            )

            # Decoder (if exists)
            if hasattr(self, "decoder"):
                decoder_trainable, decoder_total = count_params(self.decoder)
                decoder_pct = (
                    100 * decoder_trainable / decoder_total if decoder_total > 0 else 0
                )
                print(
                    f"  decoder:   {decoder_trainable:>12,} / {decoder_total:>12,} "
                    f"({decoder_pct:.2f}%)"
                )

    def print_trainable_layers(
        self,
        show_shapes: bool = False,
        group_by_component: bool = True,
        show_frozen: bool = False,
    ) -> None:
        """Print all trainable layers in the model.

        Args:
            show_shapes: If True, also print parameter shapes and sizes
            group_by_component: If True, group layers by component (backbone, head, decoder)
            show_frozen: If True, also show frozen layers (marked with ❄)
        """

        def format_param(name: str, param: torch.nn.Parameter, trainable: bool) -> str:
            """Format a single parameter line."""
            status = "✓" if trainable else "❄"
            if show_shapes:
                shape_str = str(list(param.shape))
                return f"  {status} {name}: {shape_str} ({param.numel():,} params)"
            else:
                return f"  {status} {name}"

        def print_component_layers(
            comp_name: str, component: nn.Module
        ) -> Tuple[int, int]:
            """Print layers for a component and return counts."""
            trainable_layers = []
            frozen_layers = []

            for name, param in component.named_parameters():
                if param.requires_grad:
                    trainable_layers.append(format_param(name, param, trainable=True))
                else:
                    frozen_layers.append(format_param(name, param, trainable=False))

            # Print header
            total = len(trainable_layers) + len(frozen_layers)
            print(f"\n{'=' * 60}")
            print(
                f"{comp_name.upper()}: {len(trainable_layers)}/{total} layers trainable"
            )
            print("=" * 60)

            # Print trainable layers
            if trainable_layers:
                for layer in trainable_layers:
                    print(layer)

            # Print frozen layers if requested
            if show_frozen and frozen_layers:
                print(f"  --- Frozen ({len(frozen_layers)} layers) ---")
                for layer in frozen_layers:
                    print(layer)

            return len(trainable_layers), len(frozen_layers)

        if group_by_component:
            components = [("backbone", self.backbone), ("head", self.head)]
            if hasattr(self, "decoder"):
                components.append(("decoder", self.decoder))

            total_trainable = 0
            total_frozen = 0

            for comp_name, component in components:
                trainable, frozen = print_component_layers(comp_name, component)
                total_trainable += trainable
                total_frozen += frozen

            # Summary
            print(f"\n{'=' * 60}")
            print(f"SUMMARY: {total_trainable} trainable, {total_frozen} frozen layers")
            print("=" * 60)

        else:
            print("\nTrainable Layers:")
            print("=" * 60)

            trainable_count = 0
            frozen_count = 0

            for name, param in self.named_parameters():
                if param.requires_grad:
                    print(format_param(name, param, trainable=True))
                    trainable_count += 1
                elif show_frozen:
                    print(format_param(name, param, trainable=False))
                    frozen_count += 1

            print("=" * 60)
            if show_frozen:
                print(
                    f"Total: {trainable_count} trainable, {frozen_count} frozen layers"
                )
            else:
                print(f"Total: {trainable_count} trainable layers")

    def enable_lora(
        self,
        rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
    ) -> None:
        """Enable LoRA (Low-Rank Adaptation) for efficient fine-tuning of the backbone.

        This method wraps the backbone with LoRA adapters, significantly reducing
        the number of trainable parameters while maintaining performance.

        Args:
            rank: LoRA rank (lower = fewer params, higher = more expressive)
            lora_alpha: LoRA scaling factor (alpha/rank determines scaling)
            lora_dropout: Dropout probability for LoRA layers
            target_modules: List of module names to apply LoRA to.
                           Default targets attention qkv projections.

        Raises:
            ImportError: If peft library is not installed
        """
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError(
                "LoRA requires the 'peft' library. Install with: pip install peft"
            )

        # Default target modules for ViT attention layers
        if target_modules is None:
            target_modules = ["qkv"]

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )

        # Wrap backbone with LoRA
        self.backbone = get_peft_model(self.backbone, lora_config)

        # Ensure head and decoder remain trainable
        for param in self.head.parameters():
            param.requires_grad = True
        if hasattr(self, "decoder"):
            for param in self.decoder.parameters():
                param.requires_grad = True

        print(f"LoRA enabled with rank={rank}, alpha={lora_alpha}")
        self.print_trainable_parameters(detailed=True)
        self.print_trainable_layers()

    def disable_lora(self) -> None:
        """Disable LoRA and restore the original backbone.

        Merges LoRA weights into the backbone and removes LoRA wrappers.
        """
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
    head_type: str = "upernet",
    num_classes: int = 1,
    embed_dim: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    freeze_backbone: bool = False,
    use_progressive_upsample: bool = True,
    decoder_dropout: float = 0.1,
    use_lora: bool = True,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[List[str]] = None,
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
        decoder_dropout: Dropout probability in progressive decoder (default: 0.1)
        use_lora: Enable LoRA for efficient fine-tuning (requires peft library)
        lora_rank: LoRA rank (lower = fewer params, higher = more expressive)
        lora_alpha: LoRA scaling factor (alpha/rank determines scaling)
        lora_dropout: Dropout probability for LoRA layers
        lora_target_modules: List of module names to apply LoRA to (default: ["qkv"])

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
        decoder_dropout=decoder_dropout,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
    )
