from typing import Tuple, Optional, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from transformers import SegformerConfig
from transformers.models.segformer.modeling_segformer import SegformerDecodeHead


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
        input_size = x.shape[-2:]

        # Upsample input by patch_size so features are at H x W
        upsampled_size = (input_size[0] * self.patch_size, input_size[1] * self.patch_size)
        x_upsampled = F.interpolate(x, size=upsampled_size, mode="bilinear", align_corners=False)

        # Extract multi-level features from ViT (now at H x W resolution)
        features = get_vit_features(
            self.backbone,
            x_upsampled,
            indices=self.feature_indices,
        )

        # Features are now at H x W, decode head outputs at H x W
        features_list = list(features)
        logits = self.decode_head(features_list)

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


def create_segformer_model(
    backbone_name: str = "dinov3",
    backbone_size: str = "large",
    num_classes: int = 1,
    decoder_hidden_size: int = 256,
    checkpoint_path: Optional[str] = None,
    freeze_backbone: bool = False,
    use_lora: bool = True,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """Factory function to create SegFormer segmentation models.

    Args:
        backbone_name: One of "dinov3", "dinov2", "retfound", "visionfm"
        backbone_size: "base" or "large"
        num_classes: Number of segmentation classes
        decoder_hidden_size: Hidden dimension for SegFormer decoder (default: 256)
        checkpoint_path: Path to pretrained backbone weights
        freeze_backbone: Whether to freeze backbone weights
        use_lora: Enable LoRA for efficient fine-tuning
        lora_rank: LoRA rank
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout probability for LoRA layers
        lora_target_modules: Module names to apply LoRA to (default: ["qkv"])

    Returns:
        SegFormer segmentation model ready for training
    """
    return SegFormerSegmentationModel(
        backbone_name=backbone_name,
        backbone_size=backbone_size,
        num_classes=num_classes,
        decoder_hidden_size=decoder_hidden_size,
        checkpoint_path=checkpoint_path,
        freeze_backbone=freeze_backbone,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
    )
