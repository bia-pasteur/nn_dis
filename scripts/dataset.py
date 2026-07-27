"""This is useful to create the datasets"""

import os
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

    training_transform = tio.Compose(
        [
            tio.RandomFlip(axes=("LR",), flip_probability=0.5),
            tio.RandomAffine(
                scales=(0.9, 1.1),
                degrees=(-180, 180, 0, 0, 0, 0),
                translation=7,
                p=0.5,
            ),
            tio.RandomNoise(std=(0, 0.08), p=0.3),
            # tio.RandomGamma(log_gamma=(-0.7, 0.7), p=0.3),
            tio.RandomBlur(std=(0, 1), p=0.2),
        ],
    )
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
        self.files = sorted(
            [
                *folder.glob("*.tiff"),
                *folder.glob("*.tif"),
                *folder.glob("*.npy"),
            ]
        )
        self.folder = folder

        self.transform = transform

        depths = []
        for f in self.files:
            if f.suffix.lower() in [".tiff", ".tif"]:
                img = tifffile.imread(f)
            elif f.suffix.lower() == ".npy":
                img = np.load(f)
            else:
                continue  # skip unknown file types

            if img.ndim == 4:
                d = img.shape[1]
            elif img.ndim >= 3:
                d = img.shape[2]
            else:
                d = 1  # fallback for 2D images

            depths.append(d)

        self.max_depth = max(depths) if depths else 0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):

        img_path = self.files[index]

        ext = os.path.splitext(img_path)[1].lower()

        if ext == ".npy":
            img_raw = np.load(img_path)
        elif ext in [".tiff", ".tif"]:
            img_raw = tifffile.imread(img_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        # Qunatile normalization
        q1, q99 = np.quantile(img_raw[:, 0], [0.01, 0.99])
        img = np.clip((img_raw[:, 0] - q1) / (q99 - q1), 0, 1)

        # Extract source and target (I2 -> I1 for vxm)
        if img.ndim == 4:
            source_numpy = img[1]
            target_numpy = img[0]
        else:
            source_numpy = img[1]  # 0 for gel, 1 for cell
            target_numpy = img[0]  # 0 for gel, 1 for cell

        source = reshape_img(source_numpy, self.max_depth)  # (Z, H, W)
        target = reshape_img(target_numpy, self.max_depth)  # (Z, H, W)

        mask_2d = compute_mask(np.stack([source, target], axis=0))  # (H, W)
        mask_2d = (
            torch.from_numpy(mask_2d).unsqueeze(0).unsqueeze(0)
        )  # (C, Z, H, W) -> (1, 1, H, W)

        source = torch.from_numpy(source)  # (Z, H, W) -> (Z, H, W)
        target = torch.from_numpy(target)  # (Z, H, W) -> (Z, H, W)

        # Stack to create a scalar image where each temporal frame becomes a channel.
        # This is necessary for augmentations to apply correctly on the two images.
        scalar_img = torch.stack(
            [source, target], dim=0
        )  # (C, Z, H, W) -> (2, Z, H, W)
        subject = tio.Subject(stack=tio.ScalarImage(tensor=scalar_img))

        if self.transform is not None:
            subject = self.transform(subject)

        return {
            "source": subject.stack.data[0:1],  # I2
            "target": subject.stack.data[1:2],  # I1
            "mask": mask_2d,
        }
