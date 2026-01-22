"""ViT backbone creation and feature extraction utilities."""

from typing import Tuple, Optional, List, Union

import torch
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
