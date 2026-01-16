# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from omegaconf import OmegaConf
import os
import sys
from typing import Any

from dinov3.eval.segmentation.config import SegmentationConfig
from dinov3.eval.segmentation.eval import test_segmentation
from dinov3.eval.segmentation.train import train_segmentation
from dinov3.eval.helpers import args_dict_to_dataclass, cli_parser, write_results
from dinov3.eval.setup import load_model_and_context
from dinov3.run.init import job_context


logger = logging.getLogger("dinov3")

RESULTS_FILENAME = "results-semantic-segmentation.csv"
MAIN_METRICS = ["mIoU"]


def validate_eval_config(config: SegmentationConfig):
    """Validate eval config for mode-specific requirements."""
    if config.eval.mode == "slide":
        if config.eval.crop_size is None:
            raise ValueError(
                "eval.crop_size is required when eval.mode='slide'. "
                "Please specify crop_size for sliding window inference."
            )
        if config.eval.stride is None:
            raise ValueError(
                "eval.stride is required when eval.mode='slide'. "
                "Please specify stride for sliding window inference."
            )
    elif config.eval.mode == "whole":
        if config.eval.crop_size is not None:
            raise ValueError(
                "eval.crop_size must not be set when eval.mode='whole'. "
                "Use transforms.eval.img_size to control inference resolution instead."
            )
        if config.eval.stride is not None:
            raise ValueError(
                "eval.stride must not be set when eval.mode='whole'. "
                "Stride is only used for sliding window inference."
            )


def run_segmentation_with_dinov3(
    backbone,
    config,
):
    if config.load_from:
        logger.info("Testing model performance on a pretrained decoder head")
        return test_segmentation(backbone=backbone, config=config)

    # Route to appropriate training script based on decoder type
    if config.decoder_head.type == "m2f":
        from dinov3.eval.segmentation.train_m2f import train_m2f_segmentation
        return train_m2f_segmentation(backbone=backbone, config=config)
    else:
        assert config.decoder_head.type == "linear", f"Training only supports linear or m2f, got {config.decoder_head.type}"
        return train_segmentation(backbone=backbone, config=config)


def benchmark_launcher(eval_args: dict[str, object]) -> dict[str, Any]:
    """Initialization of distributed and logging are preconditions for this method"""
    if "config" in eval_args:  # using a config yaml file, useful for training
        base_config_path = eval_args.pop("config")
        output_dir = eval_args["output_dir"]
        base_config = OmegaConf.load(base_config_path)
        structured_config = OmegaConf.structured(SegmentationConfig)
        dataclass_config: SegmentationConfig = OmegaConf.to_object(
            OmegaConf.merge(
                structured_config,
                base_config,
                OmegaConf.create(eval_args),
            )
        )
    else:  # either using default values, or only adding some args to the command line
        dataclass_config, output_dir = args_dict_to_dataclass(eval_args=eval_args, config_dataclass=SegmentationConfig)
    validate_eval_config(dataclass_config)
    backbone = None
    if dataclass_config.model:
        backbone, _ = load_model_and_context(
            dataclass_config.model,
            output_dir=output_dir,
            lora_config=dataclass_config.lora,
        )
    else:
        assert dataclass_config.load_from == "dinov3_vit7b16_ms"
    logger.info(f"Segmentation Config:\n{OmegaConf.to_yaml(dataclass_config)}")
    segmentation_file_path = os.path.join(output_dir, "segmentation_config.yaml")
    OmegaConf.save(config=dataclass_config, f=segmentation_file_path)
    results_dict = run_segmentation_with_dinov3(backbone=backbone, config=dataclass_config)
    write_results(results_dict, output_dir, RESULTS_FILENAME)
    return results_dict


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    eval_args = cli_parser(argv)
    with job_context(output_dir=eval_args["output_dir"]):
        benchmark_launcher(eval_args=eval_args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
