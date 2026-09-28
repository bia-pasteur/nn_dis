"""This is useful to create the datasets"""

import os
from pathlib import Path

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

    def __init__(
        self,
        folder: Path,
        transform: callable = None,
        channel: str | None = None,
        mask: bool | None = False,
        path_gt: str | None = None,
    ):
        self.files = sorted(
            [
                *folder.glob("*.tiff"),
                *folder.glob("*.tif"),
                *folder.glob("*.npy"),
            ]
        )
        self.folder = folder
        self.channel = channel
        self.mask = mask
        self.transform = transform
        self.path_gt = path_gt

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

        u_gt = None

        if ext == ".npy":
            img_raw = np.load(img_path)
            mask_gt = img_raw[0] > 0.05
            self.channel = None

            if self.path_gt:
                name_img = img_path.name
                ugt_name = name_img.replace("_img.npy", "_ugt.npy")
                path_ugt_file = self.path_gt / ugt_name

                ugt_raw = np.load(path_ugt_file)

                u_gt = np.stack(
                    [
                        reshape_img(ugt_raw[0], self.max_depth),
                        reshape_img(ugt_raw[1], self.max_depth),
                        reshape_img(ugt_raw[2], self.max_depth),
                    ]
                )

        elif ext in [".tiff", ".tif"]:
            img_raw = tifffile.imread(img_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        # Quantile normalization
        data = (
            img_raw[:, self.channel] if self.channel is not None else img_raw
        )  # Normalize the channel if provided
        q1, q99 = np.quantile(data, [0.01, 0.99])
        img = np.clip((data - q1) / (q99 - q1), 0, 1)  # (2, Z, H, W)

        # Extract reference and moving images (I1 -> I2)
        ref_numpy = img[0]
        moving_numpy = img[1]

        ref = reshape_img(ref_numpy, self.max_depth)  # (Z, H, W)
        moving = reshape_img(moving_numpy, self.max_depth)  # (Z, H, W)

        mask_gt = ref > 0.05

        mask = (
            compute_mask(np.stack([ref, moving], axis=0))
            if self.mask
            else np.ones(ref.shape[1:])
        )
        mask_2d = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        ref = torch.from_numpy(ref)  # (Z, H, W) numpy -> (Z, H, W) torch
        moving = torch.from_numpy(moving)  # (Z, H, W) numpy -> (Z, H, W) torch

        # Stack to create a scalar image where each temporal frame becomes a channel.
        # This is necessary for augmentations to apply correctly on the two images.
        scalar_img = torch.stack([ref, moving], dim=0)  # (C, Z, H, W) -> (2, Z, H, W)
        subject = tio.Subject(stack=tio.ScalarImage(tensor=scalar_img))

        if self.transform is not None:
            subject = self.transform(subject)

        return {
            "ref": subject.stack.data[0:1],  # I1
            "moving": subject.stack.data[1:2],  # I2
            "mask": mask_2d,
            "u_gt": u_gt,
            "mask_gt": mask_gt,
        }
