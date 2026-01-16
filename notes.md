# nnU-Net Training vs Validation: Input Sizes and Methods

## 1. Different Input Sizes in Training vs Validation

**Yes, nnU-Net uses different input sizes.**

### Training
Uses a **larger initial patch size** that gets cropped down after augmentation:
- Location: `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:876-887`
```python
dl_tr = nnUNetDataLoader(
    dataset_tr,
    self.batch_size,
    initial_patch_size,  # Larger - provides headroom for augmentation
    self.configuration_manager.patch_size,  # Final size after crop
    ...
)
```

### Validation (during training)
Uses the **same size for both parameters** (no extra padding):
- Location: `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:888-899`
```python
dl_val = nnUNetDataLoader(
    dataset_val,
    self.batch_size,
    self.configuration_manager.patch_size,  # Same as final
    self.configuration_manager.patch_size,  # No pre-padding needed
    ...
)
```

### Why the Difference?
- Training patches are sampled larger to allow spatial augmentations (rotation, scaling) without introducing border artifacts
- After augmentation, patches are cropped to `final_patch_size`
- Validation doesn't need this extra space since no spatial augmentation is applied
- See `nnunetv2/training/dataloading/data_loader.py:52-58` for implementation details

---

## 2. Validation Strategy: Two Different Approaches

| When | Method | Code Location |
|------|--------|---------------|
| During training epochs | Patch-based (fixed patches) | `nnUNetTrainer.py:1311-1389` |
| After training (final) | Sliding window on whole images | `nnUNetTrainer.py:1552-1779` |

### A. During Training Epochs - Patch-Based Validation
- Simple forward pass on batched patches
- Location: `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:1793-1798`
```python
with torch.no_grad():
    self.on_validation_epoch_start()
    val_outputs = []
    for batch_id in range(self.num_val_iterations_per_epoch):
        val_outputs.append(self.validation_step(next(self.dataloader_val)))
    self.on_validation_epoch_end(val_outputs)
```

### B. Final Validation - Sliding Window on Whole Images
- Uses `nnUNetPredictor` with sliding window inference
- Location: `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:1669`
```python
prediction = predictor.predict_sliding_window_return_logits(data)
```

**Sliding window features:**
- Implementation: `nnunetv2/inference/predict_from_raw_data.py:896-968`
- Gaussian weighting for smooth blending of overlapping patches
- Test-time augmentation (mirroring)
- `tile_step_size=0.5` (50% overlap between patches)

---

## Summary Table

| Aspect | Training | Validation (Epochs) | Validation (Final) |
|--------|----------|---------------------|-------------------|
| Patch Size | `initial_patch_size` (larger) | `patch_size` (fixed) | Full image |
| Strategy | Random crop + augmentation | Fixed-size patches | Sliding window |
| Gaussian Weighting | No | No | Yes |
| Test-Time Augmentation | No | No | Yes (mirroring) |
