"""This is useful to create the train, validation and test folders"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path

import jsonargparse
import numpy as np
import tifffile
from sklearn.model_selection import train_test_split

from config import DatasetCreation


def load_order_clean(image_path: str):
    """
    Loads a TIFF image, ensures it follows the TZYXC axis order and normalizes it

    Args:
        image_path (str): Path to the input TIFF image.

    Returns:
        tuple: A tuple containing the normalized image as a numpy
                array and the image's filename without extension.∏
    """
    # Store the name of the image for the plots
    image_name = os.path.basename(image_path)
    image_name_without_ext = os.path.splitext(image_name)[0]

    # Load the TIFF image
    with tifffile.TiffFile(image_path) as tif:
        image = tif.series[0].asarray()  # Read the image array

        # Check the axes order
        axes = tif.series[0].axes

        # The expected order is TZYXC
        expected_axes = ["T", "Z", "Y", "X", "C"]

        if "C" not in axes:
            image = image[..., np.newaxis]  # Add C as the last axis
            axes += "C"

        if "Z" not in axes:
            # Insert a singleton Z-dimension before Y
            y_index = axes.find("Y")
            image = np.expand_dims(image, axis=y_index)
            axes = axes[:y_index] + "Z" + axes[y_index:]

        if axes != expected_axes:
            # Swap axes based on the current axes order
            axis_order = {axis: idx for idx, axis in enumerate(axes)}

            # Create a mapping of the current axes to the target axes
            swap_order = [axis_order[axis] for axis in expected_axes]

            # Swap the axes to match TZYXC
            image = np.transpose(image, swap_order)

    # Normalize the image
    image = image / image.max()

    return image, image_name_without_ext


def main(dataset_creation: DatasetCreation):
    """
    Main entry for training, evaluation and testing datasets
    For now, removing the images with 4 channels

    Args:
        dataset_creation (DatasetCreation): Parameters for the datasets
    """

    folder_images = Path(dataset_creation.folder)
    folder = folder_images.parent

    # Extracting the files
    tiff_files = sorted(folder_images.glob("*.tif*"))

    # Creating the folders
    for split_name in ["train", "val", "test"]:
        (folder / split_name).mkdir(parents=True, exist_ok=True)

    for f in tiff_files:
        # Load the image
        img, name = load_order_clean(f)
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
                pair = img[[t1, t2]]  # shape (2, H, W, C)
                pair_tczyx = np.moveaxis(pair, -1, 1)  # (T,C,Z,Y,X)

                # Skip 4‑channel images (e.g., RGBA)
                if pair_tczyx.shape[1] == 4:
                    continue

                out_name = f"{name}_t{t1:04d}_t{t2:04d}.tiff"
                out_path = folder / split_name / out_name

                tifffile.imwrite(
                    out_path, pair_tczyx, ome=True, metadata={"axes": "TCZYX"}
                )

                del pair, pair_tczyx


if __name__ == "__main__":
    jsonargparse.auto_cli(main, as_positional=False)
