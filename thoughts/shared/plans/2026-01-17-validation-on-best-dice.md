# Validation on Best Dice Implementation Plan

## Overview

Add automatic validation output generation (`validation_<epoch>/` folders with `.npz`, `.pkl`, and segmentation files) whenever the EMA dice score improves during training. This feature will work with both `nnUNetv2_train.py` and `nnUNetv2_train_m2f.py`.

## Current State Analysis

### How It Works Now
1. `on_epoch_end()` detects when EMA dice improves and saves `checkpoint_best.pth`
2. `perform_actual_validation()` creates `validation/` folder with outputs, but only runs at **end of training**
3. The validation folder name is hardcoded as `"validation"`

### Trainer Inheritance
```
nnUNetTrainer (base)
    └── nnUNetTrainer_m2f (inherits, no overrides for target methods)
```

Since `nnUNetTrainer_m2f` inherits from `nnUNetTrainer` without overriding `on_epoch_end()` or `perform_actual_validation()`, we only need to modify the base class.

### Key Discoveries:
- Best dice detection: `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:1463-1471`
- Validation folder creation: `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:1597` (hardcoded `"validation"`)
- Metrics saved to: `validation/summary.json` via `compute_metrics_on_folder()`

## Desired End State

When training with `--npz` flag:
- Every time EMA dice improves, a `validation_<epoch>/` folder is created
- Each folder contains: segmentation files, `.npz`, `.pkl`, and `summary.json`
- Final validation still creates `validation/` folder at end of training
- Works for both `nnUNetv2_train.py` and `nnUNetv2_train_m2f.py`

### Verification:
```bash
# After training, output folder should contain:
ls nnUNet_results/DatasetXXX/.../fold_0/
# checkpoint_best.pth
# checkpoint_final.pth
# validation/          (final validation)
# validation_15/       (epoch 15 - first improvement)
# validation_42/       (epoch 42 - second improvement)
# ...
```

## What We're NOT Doing

- No cooldown period between validations
- No command-line flag to enable/disable (always enabled)
- No automatic cleanup of old validation folders
- Not modifying `nnUNetTrainer_mymodel.py` (user confirmed using standard trainer)

## Implementation Approach

Modify only `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`:
1. Add `validation_folder_name` parameter to `perform_actual_validation()`
2. Add call to validation in `on_epoch_end()` when best dice is detected

---

## Phase 1: Modify `perform_actual_validation()` Signature

### Overview
Add an optional `validation_folder_name` parameter to allow custom output folder names.

### Changes Required:

#### 1. Update method signature and folder creation
**File**: `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`

**Change 1 - Line 1552**: Update method signature
```python
# BEFORE:
def perform_actual_validation(self, save_probabilities: bool = False):

# AFTER:
def perform_actual_validation(self, save_probabilities: bool = False, validation_folder_name: str = "validation"):
```

**Change 2 - Line 1597**: Use the parameter for folder name
```python
# BEFORE:
validation_output_folder = join(self.output_folder, "validation")

# AFTER:
validation_output_folder = join(self.output_folder, validation_folder_name)
```

### Success Criteria:

#### Automated Verification:
- [x] Python syntax is valid: `python -m py_compile nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`
- [x] Existing behavior unchanged: calling `perform_actual_validation(save_probabilities=True)` still creates `validation/` folder (default parameter value ensures this)

---

## Phase 2: Add Validation Call in `on_epoch_end()`

### Overview
Call `perform_actual_validation()` when a new best EMA dice is detected, using `validation_<epoch>` as the folder name.

### Changes Required:

#### 1. Add validation call after best checkpoint save
**File**: `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`

**Location**: Inside the `on_epoch_end()` method, after line 1471 (after `self.save_checkpoint(...)`)

```python
# BEFORE (lines 1462-1471):
        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if (
            self._best_ema is None
            or self.logger.my_fantastic_logging["ema_fg_dice"][-1] > self._best_ema
        ):
            self._best_ema = self.logger.my_fantastic_logging["ema_fg_dice"][-1]
            self.print_to_log_file(
                f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}"
            )
            self.save_checkpoint(join(self.output_folder, "checkpoint_best.pth"))

# AFTER:
        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if (
            self._best_ema is None
            or self.logger.my_fantastic_logging["ema_fg_dice"][-1] > self._best_ema
        ):
            self._best_ema = self.logger.my_fantastic_logging["ema_fg_dice"][-1]
            self.print_to_log_file(
                f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}"
            )
            self.save_checkpoint(join(self.output_folder, "checkpoint_best.pth"))

            # Perform validation and save outputs for this best epoch
            self.print_to_log_file(
                f"Running validation for best epoch {self.current_epoch}..."
            )
            self.perform_actual_validation(
                save_probabilities=True,
                validation_folder_name=f"validation_{self.current_epoch}"
            )
            self.print_to_log_file(
                f"Validation outputs saved to validation_{self.current_epoch}/"
            )
```

### Success Criteria:

#### Automated Verification:
- [x] Python syntax is valid: `python -m py_compile nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`
- [x] Import test passes: `python -c "from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer"`

#### Manual Verification:
- [ ] Run short training and verify `validation_<epoch>/` folders are created when dice improves
- [ ] Verify each folder contains `.npz`, `.pkl`, segmentation files, and `summary.json`

---

## Phase 3: Test with Both Entry Points

### Overview
Verify the feature works with both `nnUNetv2_train.py` and `nnUNetv2_train_m2f.py`.

### Test Commands:

```bash
# Test with standard trainer
export nnUNet_raw="/data_B/xujialiu/projects/nnunet/nnunet/nnUNet_raw"
export nnUNet_preprocessed="/data_B/xujialiu/projects/nnunet/nnunet/nnUNet_preprocessed"
export nnUNet_results="nnUNet_results"

# Short test run (reduce epochs for testing)
CUDA_VISIBLE_DEVICES=0 python nnUNetv2_train.py 0 2d 0 --npz

# Test with m2f trainer
CUDA_VISIBLE_DEVICES=0 python nnUNetv2_train_m2f.py 0 2d 0 --npz
```

### Success Criteria:

#### Manual Verification:
- [ ] `nnUNetv2_train.py`: Creates `validation_<epoch>/` folders on dice improvement
- [ ] `nnUNetv2_train_m2f.py`: Creates `validation_<epoch>/` folders on dice improvement
- [ ] Each `validation_<epoch>/` folder contains expected files
- [ ] Final `validation/` folder still created at end of training
- [ ] Training completes without errors

---

## Testing Strategy

### Manual Testing Steps:
1. Run training with a small dataset for a few epochs
2. Monitor log output for "Running validation for best epoch X..."
3. Check output folder for `validation_<epoch>/` directories
4. Verify contents of validation folders match expected structure

### Expected Output Structure:
```
nnUNet_results/Dataset000_XXX/nnUNetTrainer__nnUNetPlans__2d/fold_0/
├── checkpoint_best.pth
├── checkpoint_final.pth
├── checkpoint_latest.pth
├── progress.png
├── validation/              # Final validation
│   ├── case_0001.nii.gz
│   ├── case_0001.npz
│   ├── case_0001.pkl
│   └── summary.json
├── validation_5/            # Best at epoch 5
│   ├── case_0001.nii.gz
│   ├── case_0001.npz
│   ├── case_0001.pkl
│   └── summary.json
└── validation_23/           # Best at epoch 23
    ├── ...
```

## Performance Considerations

- Full validation runs sliding window inference on all validation cases
- This is computationally expensive (may add significant time per improvement)
- For datasets with many validation cases, each validation run could take several minutes
- Disk usage will increase with multiple validation folders

## References

- Research document: `thoughts/shared/research/2025-01-17-validation-folder-dice-tracking.md`
- Key file: `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:1463-1471` (best dice detection)
- Key file: `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:1552-1779` (perform_actual_validation)
