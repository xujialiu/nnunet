# UNETR Configurable Upsampling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make UNETR decoder upsampling method configurable via YAML config, supporting bilinear interpolation and learnable transposed convolution.

**Architecture:** Add `upsample_mode` parameter that flows from YAML config → factory function → model → decoder. The decoder uses a helper method to create either `nn.Upsample` or `nn.ConvTranspose2d` based on the mode.

**Tech Stack:** PyTorch, OmegaConf (YAML parsing handled by trainer)

---

## Task 1: Update UNETRDecoder with upsample_mode

**Files:**
- Modify: `nnunetv2/model/model_unetr.py:172-258`

**Step 1: Add `_make_upsample` helper method to UNETRDecoder**

Add this method inside `UNETRDecoder` class after `__init__`:

```python
def _make_upsample(self, in_channels: int, out_channels: int) -> nn.Module:
    """Create upsampling layer based on mode.

    Args:
        in_channels: Input channels (used for deconv)
        out_channels: Output channels (used for deconv)

    Returns:
        Upsampling module
    """
    if self.upsample_mode == "deconv":
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    else:  # bilinear (default fallback)
        return nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
```

**Step 2: Update UNETRDecoder.__init__ signature and body**

Change the `__init__` method to accept `upsample_mode` and use the helper:

```python
def __init__(
    self,
    decoder_channels: List[int],
    num_classes: int,
    negative_slope: float = 0.01,
    upsample_mode: str = "deconv",
):
    """
    Args:
        decoder_channels: Channel dimensions [c0, c1, c2, c3, c4] from deep to shallow
                         e.g., [512, 256, 128, 64, 32]
        num_classes: Number of output segmentation classes
        negative_slope: LeakyReLU negative slope
        upsample_mode: "deconv" (learnable) or "bilinear" (fixed interpolation)
    """
    super().__init__()
    self.upsample_mode = upsample_mode

    # Calculate concat channels based on upsample mode
    # deconv: upsample changes channels, so concat = out_ch + skip_ch
    # bilinear: upsample keeps channels, so concat = in_ch + skip_ch
    if upsample_mode == "deconv":
        concat4 = decoder_channels[1] + decoder_channels[1]  # up4_out + skip3
        concat3 = decoder_channels[2] + decoder_channels[2]  # up3_out + skip2
        concat2 = decoder_channels[3] + decoder_channels[3]  # up2_out + skip1
    else:
        concat4 = decoder_channels[0] + decoder_channels[1]  # bottleneck + skip3
        concat3 = decoder_channels[1] + decoder_channels[2]  # conv4_out + skip2
        concat2 = decoder_channels[2] + decoder_channels[3]  # conv3_out + skip1

    # Stage 4: bottleneck (c0) + skip3 (c1) -> upsample -> conv -> c1
    self.up4 = self._make_upsample(decoder_channels[0], decoder_channels[1])
    self.conv4 = ConvBlock(concat4, decoder_channels[1], negative_slope)

    # Stage 3: c1 + skip2 (c2) -> upsample -> conv -> c2
    self.up3 = self._make_upsample(decoder_channels[1], decoder_channels[2])
    self.conv3 = ConvBlock(concat3, decoder_channels[2], negative_slope)

    # Stage 2: c2 + skip1 (c3) -> upsample -> conv -> c3
    self.up2 = self._make_upsample(decoder_channels[2], decoder_channels[3])
    self.conv2 = ConvBlock(concat2, decoder_channels[3], negative_slope)

    # Stage 1: c3 -> conv -> c4 -> 1x1 conv -> num_classes
    self.conv1 = ConvBlock(decoder_channels[3], decoder_channels[4], negative_slope)
    self.seg_head = nn.Conv2d(decoder_channels[4], num_classes, kernel_size=1)
```

**Step 3: Verify syntax by importing module**

Run: `cd /data_B/xujialiu/projects/nnunet/nnunet/.worktrees/unetr-upsample && python -c "from nnunetv2.model.model_unetr import UNETRDecoder; print('OK')"`

Expected: `OK`

**Step 4: Commit**

```bash
git add nnunetv2/model/model_unetr.py
git commit -m "feat(unetr): add configurable upsample_mode to UNETRDecoder"
```

---

## Task 2: Update UNETRSegmentationModel to pass upsample_mode

**Files:**
- Modify: `nnunetv2/model/model_unetr.py:268-347`

**Step 1: Update UNETRSegmentationModel.__init__ signature**

Add `upsample_mode` parameter and pass to decoder:

```python
def __init__(
    self,
    backbone_name: str = "dinov3",
    backbone_size: str = "large",
    num_classes: int = 1,
    decoder_channels: Optional[List[int]] = None,
    negative_slope: float = 0.01,
    upsample_mode: str = "deconv",
    checkpoint_path: Optional[str] = None,
    freeze_backbone: bool = False,
    use_lora: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[List[str]] = None,
):
```

**Step 2: Update decoder instantiation**

Change the decoder creation to pass upsample_mode:

```python
# UNETR decoder (U-Net style)
self.decoder = UNETRDecoder(
    decoder_channels=decoder_channels,
    num_classes=num_classes,
    negative_slope=negative_slope,
    upsample_mode=upsample_mode,
)
```

**Step 3: Verify syntax**

Run: `cd /data_B/xujialiu/projects/nnunet/nnunet/.worktrees/unetr-upsample && python -c "from nnunetv2.model.model_unetr import UNETRSegmentationModel; print('OK')"`

Expected: `OK`

**Step 4: Commit**

```bash
git add nnunetv2/model/model_unetr.py
git commit -m "feat(unetr): pass upsample_mode through UNETRSegmentationModel"
```

---

## Task 3: Update create_segmentation_model factory

**Files:**
- Modify: `nnunetv2/model/model_unetr.py:478-531`

**Step 1: Update factory function signature**

Add `upsample_mode` parameter:

```python
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
    # UNETR-specific decoder params
    decoder_channels: Optional[List[int]] = None,
    negative_slope: float = 0.01,
    upsample_mode: str = "deconv",
    **kwargs,
) -> nn.Module:
```

**Step 2: Update docstring**

Add to docstring:

```python
"""Factory function to create UNETR segmentation models.

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
    decoder_channels: Channel dimensions for decoder [c0, c1, c2, c3, c4]
                     Default: [512, 256, 128, 64, 32]
    negative_slope: LeakyReLU negative slope (default: 0.01)
    upsample_mode: "deconv" (learnable) or "bilinear" (fixed). Default: "deconv"
    **kwargs: Additional params (ignored for forward compatibility)

Returns:
    UNETR segmentation model ready for training
"""
```

**Step 3: Update return statement**

Pass upsample_mode to model:

```python
return UNETRSegmentationModel(
    backbone_name=backbone_name,
    backbone_size=backbone_size,
    num_classes=num_classes,
    decoder_channels=decoder_channels,
    negative_slope=negative_slope,
    upsample_mode=upsample_mode,
    checkpoint_path=checkpoint_path,
    freeze_backbone=freeze_backbone,
    use_lora=use_lora,
    lora_rank=lora_rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    lora_target_modules=lora_target_modules,
)
```

**Step 4: Verify factory works**

Run: `cd /data_B/xujialiu/projects/nnunet/nnunet/.worktrees/unetr-upsample && python -c "from nnunetv2.model.model_unetr import create_segmentation_model; print('OK')"`

Expected: `OK`

**Step 5: Commit**

```bash
git add nnunetv2/model/model_unetr.py
git commit -m "feat(unetr): add upsample_mode to create_segmentation_model factory"
```

---

## Task 4: Update YAML config

**Files:**
- Modify: `nnunetv2/configs/unetr.yaml`

**Step 1: Add upsample_mode to decoder section**

Update the YAML file:

```yaml
# UNETR configuration for nnUNet ViT integration
# Based on: Hatamizadeh et al. "UNETR: Transformers for 3D Medical Image Segmentation"

model:
  model_path: "nnunetv2.model.model_unetr"
  backbone: "dinov3"
  backbone_size: "large"
  checkpoint_path: ""
  decoder:
    decoder_channels: [512, 256, 128, 64, 32]
    negative_slope: 0.01
    upsample_mode: "deconv"  # Options: "deconv" (learnable), "bilinear" (fixed)

lora:
  enabled: true
  r: 8
  lora_alpha: 16
  target_modules: ["qkv"]
  lora_dropout: 0.05

train:
  initial_lr: 0.001
  weight_decay: 0.0001
  gradient_clip: 12.0
  num_epochs: 1000
  num_iterations_per_epoch: 250
  num_val_iterations_per_epoch: 50
```

**Step 2: Commit**

```bash
git add nnunetv2/configs/unetr.yaml
git commit -m "feat(config): add upsample_mode option to unetr.yaml"
```

---

## Task 5: Update test configs

**Files:**
- Modify: `nnunetv2/configs/unetr_test_dinov3.yaml`
- Modify: `nnunetv2/configs/unetr_test_dinov2.yaml`

**Step 1: Read current test configs to understand structure**

Check existing test configs and add upsample_mode if decoder section exists.

**Step 2: Add upsample_mode to test configs**

Add `upsample_mode: "deconv"` to the decoder section of each test config.

**Step 3: Commit**

```bash
git add nnunetv2/configs/unetr_test_*.yaml
git commit -m "feat(config): add upsample_mode to unetr test configs"
```

---

## Task 6: Integration test

**Step 1: Test model creation with deconv mode**

Run:
```bash
cd /data_B/xujialiu/projects/nnunet/nnunet/.worktrees/unetr-upsample
python -c "
from nnunetv2.model.model_unetr import create_segmentation_model
import torch

model = create_segmentation_model(
    backbone_name='dinov3',
    backbone_size='large',
    num_classes=4,
    upsample_mode='deconv',
    use_lora=False,
    freeze_backbone=True,
)
print('Deconv mode - Decoder up4:', type(model.decoder.up4).__name__)
x = torch.randn(1, 3, 224, 224)
out = model(x)
print('Output shape:', out.shape)
"
```

Expected:
```
Deconv mode - Decoder up4: ConvTranspose2d
Output shape: torch.Size([1, 4, 224, 224])
```

**Step 2: Test model creation with bilinear mode**

Run:
```bash
python -c "
from nnunetv2.model.model_unetr import create_segmentation_model
import torch

model = create_segmentation_model(
    backbone_name='dinov3',
    backbone_size='large',
    num_classes=4,
    upsample_mode='bilinear',
    use_lora=False,
    freeze_backbone=True,
)
print('Bilinear mode - Decoder up4:', type(model.decoder.up4).__name__)
x = torch.randn(1, 3, 224, 224)
out = model(x)
print('Output shape:', out.shape)
"
```

Expected:
```
Bilinear mode - Decoder up4: Upsample
Output shape: torch.Size([1, 4, 224, 224])
```

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat(unetr): complete configurable upsampling implementation" --allow-empty
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Add `_make_upsample` helper and update `UNETRDecoder` | `model_unetr.py` |
| 2 | Pass `upsample_mode` through `UNETRSegmentationModel` | `model_unetr.py` |
| 3 | Update `create_segmentation_model` factory | `model_unetr.py` |
| 4 | Add `upsample_mode` to `unetr.yaml` | `unetr.yaml` |
| 5 | Update test configs | `unetr_test_*.yaml` |
| 6 | Integration test both modes | - |
