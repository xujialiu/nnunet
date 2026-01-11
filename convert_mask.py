from pathlib import Path
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


def batch_convert_masks(src_path, suffix, color_map=None):
    src_path = Path(src_path)

    # Ensure suffix starts with '.'
    if not suffix.startswith("."):
        suffix = "." + suffix

    # Create output directory: src_path/../masks_converted
    output_dir = src_path.parent / "masks_converted"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all files with the given suffix
    mask_files = sorted(src_path.glob(f"*{suffix}"))

    print(f"Found {len(mask_files)} files with suffix '{suffix}'")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    for mask_file in mask_files:
        # Read mask
        mask = io.imread(mask_file)

        # Convert to color
        color_img = mask_to_color(mask, color_map)

        # Save to output directory with same filename
        output_path = output_dir / mask_file.name
        io.imsave(output_path, color_img)

        print(f"Converted: {mask_file.name}")

    print("-" * 50)
    print(f"Done! {len(mask_files)} files saved to: {output_dir}")

    return output_dir


# Example usage
if __name__ == "__main__":
    src_path = "/data_B/xujialiu/projects/nnunet/nnUNet_results/Dataset000_semantic_retina_vessel_segmentation/nnUNetTrainer__nnUNetPlans__2d/fold_0/validation"

    # Convert all .png files
    batch_convert_masks(src_path, suffix=".png")
