#!/usr/bin/env python
"""Entry point for ViT-based segmentation training in nnUNet."""

import os
import sys

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

    from nnunetv2.run.run_training_vit import run_training_entry
    sys.exit(run_training_entry())
