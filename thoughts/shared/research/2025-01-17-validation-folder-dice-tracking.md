---
date: 2026-01-17T14:21:43+08:00
researcher: Claude
git_commit: fc5dd476417961607d7eec5f21d7daaebc1f1218
branch: use_m2f_dinov3
repository: use_m2f_dinov3
topic: "Understanding validation folder creation and dice score tracking during training"
tags: [research, codebase, nnunet, validation, dice, training]
status: complete
last_updated: 2026-01-17
last_updated_by: Claude
---

# Research: Validation Folder Creation and Dice Score Tracking

**Date**: 2026-01-17T14:21:43+08:00
**Researcher**: Claude
**Git Commit**: fc5dd476417961607d7eec5f21d7daaebc1f1218
**Branch**: use_m2f_dinov3
**Repository**: use_m2f_dinov3

## Research Question

How is the validation folder with `.npz`, `.png`, `.pkl` files created at the end of training, and where is dice score improvement tracked during training? The goal is to understand how to produce `validation_<epoch>` folders whenever dice improves during training.

## Summary

The validation output system consists of two separate mechanisms:

1. **Per-epoch pseudo validation**: Runs during training via `validation_step()` → computes approximate dice on random patches → tracks EMA dice → saves `checkpoint_best.pth` when improved
2. **Full validation**: Runs via `perform_actual_validation()` → produces the `validation/` folder with `.npz`, `.pkl`, and segmentation files → currently only called at end of training

To achieve `validation_<epoch>` folders on dice improvement, the `on_epoch_end()` method needs modification to call `perform_actual_validation()` with a custom output folder name when a new best EMA dice is detected.

## Detailed Findings

### 1. Validation Folder Creation

**Location**: `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:1552-1779`

The `perform_actual_validation()` method handles all validation output:

#### Folder Creation (lines 1597-1598)
```python
validation_output_folder = join(self.output_folder, "validation")
maybe_mkdir_p(validation_output_folder)
```

#### Output Files Generated

| File Type | Condition | Save Location | Code Reference |
|-----------|-----------|---------------|----------------|
| Segmentation (`.nii.gz` or dataset file_ending) | Always | `validation/<case_id>.<file_ending>` | Line 1674-1688 via `export_prediction_from_logits()` |
| `.npz` (softmax probabilities) | Only when `save_probabilities=True` (`--npz` flag) | `validation/<case_id>.npz` | `nnunetv2/inference/export_prediction.py:130-133` |
| `.pkl` (properties dict) | Only when `save_probabilities=True` | `validation/<case_id>.pkl` | `nnunetv2/inference/export_prediction.py:134` |
| `summary.json` (metrics) | Always | `validation/summary.json` | Line 1756-1770 via `compute_metrics_on_folder()` |

#### When Validation is Triggered

Currently only at end of training from run scripts:

- `nnunetv2/run/run_training.py:291`
- `nnunetv2/run/run_training_nodeepsupervision.py:297`
- `nnunetv2/run/run_training_nodeepsupervision_mymodel.py:298`
- `nnunetv2/run/run_training_m2f.py:310`

```python
# run_training.py:284-291
if not only_run_validation:
    nnunet_trainer.run_training()

if val_with_best:
    nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, "checkpoint_best.pth"))
nnunet_trainer.perform_actual_validation(export_validation_probabilities)
```

### 2. Dice Score Tracking During Training

#### Training Loop Structure

**Location**: `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:1781-1802`

```python
def run_training(self):
    self.on_train_start()

    for epoch in range(self.current_epoch, self.num_epochs):
        self.on_epoch_start()
        self.on_train_epoch_start()
        # ... 250 training iterations ...
        self.on_train_epoch_end(train_outputs)

        with torch.no_grad():
            self.on_validation_epoch_start()
            # ... 50 validation iterations ...
            self.on_validation_epoch_end(val_outputs)

        self.on_epoch_end()  # <-- Best dice detection happens here

    self.on_train_end()
```

#### Dice Calculation Flow

1. **Per-batch computation** (`validation_step()`, lines 1311-1389):
   - Computes `tp_hard`, `fp_hard`, `fn_hard` using `get_tp_fp_fn_tn()` from `nnunetv2/training/loss/dice.py:140`

2. **Epoch aggregation** (`on_validation_epoch_end()`, lines 1391-1426):
   ```python
   # Lines 1418-1422
   global_dc_per_class = 2 * tp / (2 * tp + fp + fn)
   mean_fg_dice = np.nanmean(global_dc_per_class)
   self.logger.log("mean_fg_dice", mean_fg_dice, self.current_epoch)
   ```

3. **EMA computation** (automatic in logger, `nnunetv2/training/logging/nnunet_logger.py:55-62`):
   ```python
   ema_fg_dice = previous_ema * 0.9 + 0.1 * new_value
   ```

#### Best Model Detection

**Location**: `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:1463-1471`

```python
def on_epoch_end(self):
    # ... logging code ...

    # handle 'best' checkpointing. ema_fg_dice is computed by the logger
    if (
        self._best_ema is None
        or self.logger.my_fantastic_logging["ema_fg_dice"][-1] > self._best_ema
    ):
        self._best_ema = self.logger.my_fantastic_logging["ema_fg_dice"][-1]
        self.print_to_log_file(
            f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}"
        )
        self.save_checkpoint(join(self.output_folder, "checkpoint_best.pth"))
        # <-- THIS IS WHERE validation_<epoch> generation should be added

    self.current_epoch += 1
```

### 3. Progress PNG Generation

**Location**: `nnunetv2/training/logging/nnunet_logger.py:93-152`

The `progress.png` file is generated at end of each epoch (line 1474 in trainer):
```python
self.logger.plot_progress_png(self.output_folder)
```

This file is saved to `<output_folder>/progress.png`, NOT in the validation folder.

### 4. Related Trainer Variants

The same pattern exists in custom model trainers:

| File | Best Detection Lines | Validation Method Lines |
|------|---------------------|------------------------|
| `nnUNetTrainer.py` | 1463-1471 | 1552-1779 |
| `nnUNetTrainer_mymodel.py` | 1469-1477 | 1558-1785 |

## Code References

- `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:1463-1471` - Best EMA dice detection and checkpoint saving
- `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:1552-1779` - `perform_actual_validation()` method
- `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:1597-1598` - Validation folder creation
- `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:1781-1802` - Main training loop
- `nnunetv2/training/logging/nnunet_logger.py:55-62` - EMA dice calculation
- `nnunetv2/inference/export_prediction.py:130-136` - NPZ/PKL file saving
- `nnunetv2/evaluation/evaluate_predictions.py:143` - `compute_metrics_on_folder()` for summary.json
- `nnunetv2/run/run_training.py:291` - Where `perform_actual_validation()` is called at end

## Architecture Documentation

### Checkpoint Files

| File | When Saved | Purpose |
|------|------------|---------|
| `checkpoint_latest.pth` | Every `save_every` epochs (default 50) | Resume training |
| `checkpoint_best.pth` | When new best EMA dice | Best model for inference |
| `checkpoint_final.pth` | At end of training | Final model state |

### Key Configuration Parameters

```python
self.num_iterations_per_epoch = 250  # Training batches per epoch
self.num_val_iterations_per_epoch = 50  # Validation batches per epoch
self.num_epochs = 1000  # Total training epochs
self.save_every = 50  # Checkpoint frequency
```

## Implementation Strategy

To produce `validation_<epoch>` folders on dice improvement:

### Option 1: Modify `perform_actual_validation()` to accept custom folder name

1. Add parameter: `validation_folder_name: str = "validation"`
2. Change line 1597: `validation_output_folder = join(self.output_folder, validation_folder_name)`

### Option 2: Add call in `on_epoch_end()` after best detection

After line 1471 (checkpoint save), add:
```python
self.perform_actual_validation(
    save_probabilities=True,  # or make configurable
    validation_folder_name=f"validation_{self.current_epoch}"
)
```

### Considerations

- Full validation is expensive (sliding window inference on all val cases)
- May significantly increase training time if dice improves frequently
- Consider adding a minimum epoch interval between validations
- The `save_probabilities` flag controls `.npz`/`.pkl` generation

## Open Questions

1. Should there be a cooldown period between validation runs to avoid excessive computation?
2. Should old `validation_<epoch>` folders be automatically cleaned up to save disk space?
3. Should the feature be controlled by a new command-line flag (e.g., `--val_on_best`)?
