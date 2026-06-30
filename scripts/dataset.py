"""This is useful to create the datasets"""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import tifffile
import torch
import torchio as tio
from scipy.ndimage import binary_dilation, binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.morphology import disk, remove_small_objects
from torch.utils.data import Dataset

from scripts.utils import reshape_img


def get_transforms():
    """3D transformations for training

    Returns:
        _type_: _description_
    """

    geometric_transforms = tio.Compose(
        [
            tio.RandomFlip(axes=("LR",), flip_probability=0.5),
            tio.RandomAffine(scales=(0.95, 1.05), degrees=5, translation=1, p=0.5),
            tio.RandomElasticDeformation(
                num_control_points=7,
                max_displacement=1,
                locked_borders=2,
            ),
        ],
        include=("source", "target"),
    )
    intensity_transforms = tio.Compose(
        [
            tio.RandomNoise(std=(0.01, 0.05), p=0.3),
            tio.RandomGamma(log_gamma=(-0.2, 0.2), p=0.3),
            tio.RandomBlur(std=(0, 1), p=0.2),
        ],
        include=("source",),
    )

    training_transform = tio.Compose([geometric_transforms, intensity_transforms])

    return training_transform


def compute_mask(img: np.ndarray, max_size: int = 20, radius: int = 9) -> np.ndarray:
    """
    Computes a rough mask around the cell in an image

    Args:
        img (np.ndarray): The image containing a cell
        max_size (int, optional): The maximum size of telements to delete from the mask obtained via thresholding. Defaults to 20.
        radius (int, optional): The radius of the disk to add around the obtained mask. Defaults to 9.

    Returns:
        np.ndarray: The mask
    """
    max_t = np.max(img, axis=0)
    max_z = np.max(max_t, axis=0)
    thresh = threshold_otsu(max_z)
    mask_2d = max_z > thresh
    mask_2d = binary_fill_holes(mask_2d)
    mask_2d = remove_small_objects(mask_2d, max_size=max_size)

    mask_2d = binary_dilation(mask_2d, structure=disk(radius))

    return mask_2d


class TractionDataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, folder: Path, transform: Optional[Callable] = None):
        self.files = list(folder.glob("*.tiff"))
        self.folder = folder

        self.transform = transform

        depths = []
        for f in self.files:
            img = tifffile.imread(f)
            d = img.shape[2]
            depths.append(d)

        self.max_depth = max(depths) if depths else 0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        img_path = self.files[index]
        img_raw = tifffile.imread(img_path)

        q1, q99 = np.quantile(img_raw, [0.01, 0.99])
        img = np.clip((img_raw - q1) / (q99 - q1), 0, 1)

        source_numpy = img[1, 1]
        target_numpy = img[0, 1]

        source = reshape_img(source_numpy, self.max_depth)  # (Z, H, W)
        target = reshape_img(target_numpy, self.max_depth)  # (Z, H, W)

        mask_2d = compute_mask(np.stack([source, target], axis=0))  # (H, W)
        mask_2d = (
            torch.from_numpy(mask_2d).unsqueeze(0).unsqueeze(0)
        )  # (C, Z, H, W) -> (1, 1, H, W)

        source = torch.from_numpy(source).unsqueeze(0)  # (C, Z, H, W) -> (1, Z, H, W)
        target = torch.from_numpy(target).unsqueeze(0)  # (C, Z, H, W) -> (1, Z, H, W)

        subject = tio.Subject(
            source=tio.ScalarImage(tensor=source),
            target=tio.ScalarImage(tensor=target),
        )

        if self.transform is not None:
            subject = self.transform(subject)

        return {
            "source": subject.source.data,
            "target": subject.target.data,
            "mask": mask_2d,
        }
