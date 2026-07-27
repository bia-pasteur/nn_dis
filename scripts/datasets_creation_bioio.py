"""This is useful to create the train, validation and test folders"""

import os
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
    """
    Loads a TIFF image via BioImage, reorders to TZYXC, normalizes it,
    and extracts physical pixel sizes from the metadata.

    Returns:
        tuple: (normalized image array, name without ext, PhysicalPixelSizes)
               PhysicalPixelSizes has .Z .Y .X fields in µm (None if missing)
    """
    image_name = os.path.basename(image_path)
    image_name_without_ext = os.path.splitext(image_name)[0]

    bio_img = BioImage(image_path)

    image = bio_img.get_image_data("TZYXC").astype(np.float32)

    physical_pixel_sizes = bio_img.physical_pixel_sizes

    image = image / image.max()

    return image, image_name_without_ext, physical_pixel_sizes


def main(dataset_creation: DatasetCreation):
    folder_images = Path(dataset_creation.folder)
    folder = folder_images.parent

    tiff_files = sorted(folder_images.glob("*.tif*"))

    for split_name in ["train", "val", "test"]:
        (folder / split_name).mkdir(parents=True, exist_ok=True)

    for f in tiff_files:
        img, name, pps = load_order_clean(f)
        time_frames = img.shape[0]

        frame_pairs = [
            (t, t + gap)
            for t in range(time_frames)
            for gap in range(1, dataset_creation.gap_max + 1)
            if t + gap < time_frames
        ]

        train_idx, tmp_idx = train_test_split(
            frame_pairs, test_size=0.3, random_state=dataset_creation.seed
        )
        val_idx, test_idx = train_test_split(
            tmp_idx, test_size=0.5, random_state=dataset_creation.seed
        )

        split_map = {"train": train_idx, "val": val_idx, "test": test_idx}

        ome_metadata = {"axes": "TCZYX"}
        for dim, value in [("Z", pps.Z), ("Y", pps.Y), ("X", pps.X)]:
            if value is not None:
                ome_metadata[f"PhysicalSize{dim}"] = float(value)
                ome_metadata[f"PhysicalSize{dim}Unit"] = "µm"

        for split_name, pairs in split_map.items():
            for t1, t2 in pairs:
                pair = img[[t1, t2]]  # (2, Z, Y, X, C)
                pair_tczyx = np.moveaxis(pair, -1, 1)  # (T, C, Z, Y, X)

                if pair_tczyx.shape[1] == 4:
                    continue

                out_name = f"{name}_t{t1:04d}_t{t2:04d}.tiff"
                out_path = folder / split_name / out_name

                tifffile.imwrite(
                    out_path,
                    pair_tczyx,
                    ome=True,
                    metadata=ome_metadata,  # ← pixel sizes land in OME-XML
                )

                del pair, pair_tczyx


if __name__ == "__main__":
    jsonargparse.auto_cli(main, as_positional=False)
