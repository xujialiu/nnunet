import argparse
from pathlib import Path

from joblib import Parallel, delayed
from skimage import io
import numpy as np


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


def _process_one(img_file: Path, dst_path: Path, src_path: Path) -> None:
    relative_path = img_file.relative_to(src_path)
    dst_file = dst_path / relative_path

    if dst_file.exists():
        print(f"Skipping {img_file.name}: destination already exists")
        return

    img = io.imread(img_file)

    if img.ndim == 3:
        print(f"Skipping {img_file.name}: not a grayscale mask (shape={img.shape})")
        return

    try:
        converted_img = mask_to_color(img)
    except Exception as e:
        print(f"Error processing {img_file}: {e}")
        return

    dst_file.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(dst_file, converted_img)
    print(f"Saved: {dst_file}")


def convert_images(src_root: str, dst_root: str = None, n_jobs: int = -1) -> None:
    """
    Find all JPG and PNG images in src_root, convert masks to color,
    and save to dst_root preserving directory structure.
    """
    src_path = Path(src_root)

    if dst_root:
        dst_path = Path(dst_root)
    else:
        dst_path = src_path.parent / f"{src_path.name}_converted"

    extensions = (".jpg", ".jpeg", ".png")
    image_files = [f for f in src_path.rglob("*") if f.suffix.lower() in extensions]

    print(f"Found {len(image_files)} images, using n_jobs={n_jobs}")

    Parallel(n_jobs=n_jobs)(
        delayed(_process_one)(img_file, dst_path, src_path)
        for img_file in image_files
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert segmentation masks to color images"
    )
    parser.add_argument(
        "src_folder",
        type=str,
        help="Source folder containing mask images",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output folder (default: <src_folder>_converted)",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=-1,
        help="Number of parallel workers (default: -1, all CPUs)",
    )

    args = parser.parse_args()
    convert_images(args.src_folder, args.output, n_jobs=args.jobs)
