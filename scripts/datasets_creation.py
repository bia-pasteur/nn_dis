"""This is useful to create the train, validation and test folders.

Unifies the two kinds of inputs:
    - *_img.npy files: synthetic images, split by whole file and copied as-is
    - any other file:  loaded through BioImage, reordered to TZYXC, split by
                       frame pairs, with physical pixel sizes preserved
"""

import os
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path

import jsonargparse
import numpy as np
import tifffile
from bioio import BioImage
from sklearn.model_selection import train_test_split

from config import DatasetCreation


def load_order_clean(image_path: str):
    # Store the name of the image for the plots
    image_name = os.path.basename(image_path)
    image_name_without_ext = os.path.splitext(image_name)[0]

    # Load the BioImage image
    bio_img = BioImage(image_path)

    # The expected order is TZYXC
    image = bio_img.get_image_data("TZYXC").astype(np.float32)

    # Extract the physical size of the pixels
    physical_pixel_sizes = bio_img.physical_pixel_sizes

    # Normalize
    image = image / image.max()

    return image, image_name_without_ext, physical_pixel_sizes


def main(dataset_creation: DatasetCreation):
    """
    Main entry for training, evaluation and testing datasets creation.

    Synthetic *_img.npy files are split by whole file and copied as-is.
    Other images are split by frame pairs (t, t + gap) for gap in 1..gap_max.

    For now, removing the images with 4 channels.

    Args:
        dataset_creation (DatasetCreation): Parameters for the datasets
    """
    folder_images = Path(dataset_creation.folder)
    folder = folder_images.parent

    for split_name in ["train", "val", "test"]:
        (folder / split_name).mkdir(parents=True, exist_ok=True)

    # Synthetic (and fractal) images end in .npy
    img_files = sorted(folder_images.glob("*_img.npy"))

    if img_files:
        train_files, temp_files = train_test_split(
            img_files, test_size=0.3, random_state=dataset_creation.seed
        )
        val_files, test_files = train_test_split(
            temp_files, test_size=0.5, random_state=dataset_creation.seed
        )

        def copy_files(files, split_name):
            split_folder = folder / split_name
            for f in files:
                shutil.copy(f, split_folder / f.name)

        copy_files(train_files, "train")
        copy_files(val_files, "val")
        copy_files(test_files, "test")

    # Other images (.tif, .tiff...) are worked using bioio.
    for f in sorted(folder_images.glob("*.tif*")):
        img, name, pps = load_order_clean(f)

        ome_metadata = {"axes": "TCZYX"}
        for dim, value in [("Z", pps.Z), ("Y", pps.Y), ("X", pps.X)]:
            if value is not None:
                ome_metadata[f"PhysicalSize{dim}"] = float(value)
                ome_metadata[f"PhysicalSize{dim}Unit"] = "µm"

        time_frames = img.shape[0]

        frame_pairs = [
            (t, t + gap)
            for t in range(time_frames)
            for gap in range(1, dataset_creation.gap_max + 1)
            if t + gap < time_frames
        ]

        # Split indices into train (70%), and temporary holdout (30%)
        train_idx, tmp_idx = train_test_split(
            frame_pairs, test_size=0.3, random_state=dataset_creation.seed
        )
        # Split the temporary holdout equally into validation (15%) and test (15%)
        val_idx, test_idx = train_test_split(
            tmp_idx, test_size=0.5, random_state=dataset_creation.seed
        )

        # Map each split name to its corresponding frame indices
        split_map = {"train": train_idx, "val": val_idx, "test": test_idx}

        # Iterate over each split and save the corresponding image pairs
        for split_name, pairs in split_map.items():
            for t1, t2 in pairs:
                # Stack the two frames along a time dimension
                pair = img[[t1, t2]]  # shape (2, Z, H, W, C)

                pair_tczyx = np.moveaxis(pair, -1, 1)  # (T,C,Z,Y,X)

                # Skip 4‑channel images (e.g., RGBA)
                if pair_tczyx.shape[1] == 4:
                    continue

                out_name = f"{name}_t{t1:04d}_t{t2:04d}.tiff"
                out_path = folder / split_name / out_name

                tifffile.imwrite(out_path, pair_tczyx, ome=True, metadata=ome_metadata)

                del pair, pair_tczyx


if __name__ == "__main__":
    jsonargparse.auto_cli(main, as_positional=False)
