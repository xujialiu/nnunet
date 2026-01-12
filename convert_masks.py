from pathlib import Path
from skimage import io
import numpy as np
import argparse


def mask_to_color(mask, color_map=None):
    if color_map is None:
        # Default color map for retinal vessel segmentation
        color_map = {
            0: [0, 0, 0],  # Background - black
            1: [255, 0, 0],  # Class 1 - red (e.g., arteries)
            2: [0, 0, 255],  # Class 2 - blue (e.g., veins)
            3: [0, 255, 0],  # Class 3 - green (e.g., other vessels)
        }

    # Create output RGB image
    h, w = mask.shape[:2]
    color_img = np.zeros((h, w, 3), dtype=np.uint8)

    # Map each label to its color
    for label, color in color_map.items():
        color_img[mask == label] = color

    return color_img


def batch_convert_masks(src_path, suffix, dst_path=None, color_map=None):
    src_path = Path(src_path)

    # Ensure suffix starts with '.'
    if not suffix.startswith("."):
        suffix = "." + suffix

    # Create output directory: src_path/../masks_converted or user-specified
    if dst_path is None:
        dst_path = src_path.parent / "masks_converted"
    else:
        dst_path = Path(dst_path)
    dst_path.mkdir(parents=True, exist_ok=True)

    # Find all files with the given suffix
    mask_files = sorted(src_path.glob(f"*{suffix}"))

    print(f"Found {len(mask_files)} files with suffix '{suffix}'")
    print(f"Output directory: {dst_path}")
    print("-" * 50)

    for mask_file in mask_files:
        # Read mask
        mask = io.imread(mask_file)

        # Convert to color
        color_img = mask_to_color(mask, color_map)

        # Save to output directory with same filename
        output_path = dst_path / mask_file.name
        io.imsave(output_path, color_img)

        print(f"Converted: {mask_file.name}")

    print("-" * 50)
    print(f"Done! {len(mask_files)} files saved to: {dst_path}")

    return dst_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert segmentation masks to colorized images"
    )
    parser.add_argument(
        "--src_path",
        type=str,
        help="Source directory containing mask files"
    )
    parser.add_argument(
        "-s", "--suffix",
        type=str,
        default=".png",
        help="File suffix/extension to look for (default: .png)"
    )
    parser.add_argument(
        "-d", "--dst_path",
        type=str,
        default=None,
        help="Output directory (default: src_path/../masks_converted)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_convert_masks(
        src_path=args.src_path,
        suffix=args.suffix,
        dst_path=args.dst_path
    )