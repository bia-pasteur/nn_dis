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


def get_spatial_transforms():
    """3D transformations for training

    Returns:
        _type_: _description_
    """

    spatial_transforms = tio.Compose(
        [
            tio.RandomFlip(axes=("LR",), flip_probability=0.3),
            tio.RandomFlip(axes=("AP",), flip_probability=0.3),
            tio.RandomFlip(axes=("IS",), flip_probability=0.3),
            tio.RandomAffine(
                scales=(0.9, 1.1),
                degrees=(-180, 180, 0, 0, 0, 0),
                translation=7,
                p=0.5,
            ),
        ],
    )
    return spatial_transforms


def get_intensity_transforms():
    """3D transformations for training

    Returns:
        _type_: _description_
    """

    intensity_transforms = tio.Compose(
        [
            # tio.RandomNoise(std=(0, 0.08), p=0.3),
            tio.RandomBlur(std=(0, 1), p=0.2),
        ],
    )
    return intensity_transforms


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


# class TractionDataset(Dataset):
#     """_summary_

#     Args:
#         Dataset (_type_): _description_
#     """

#     def __init__(
#         self,
#         folder: Path,
#         transform: callable = None,
#         channel: str | None = None,
#         mask: bool | None = False,
#         path_gt: str | None = None,
#     ):
#         self.files = sorted(
#             [
#                 *folder.glob("*.tiff"),
#                 *folder.glob("*.tif"),
#                 *folder.glob("*.npy"),
#             ]
#         )
#         self.folder = folder
#         self.channel = channel
#         self.mask = mask
#         self.transform = transform
#         self.path_gt = path_gt

#         depths = []
#         for f in self.files:
#             if f.suffix.lower() in [".tiff", ".tif"]:
#                 img = tifffile.imread(f)
#             elif f.suffix.lower() == ".npy":
#                 img = np.load(f)
#             else:
#                 continue  # skip unknown file types

#             if img.ndim == 4:
#                 d = img.shape[1]
#             elif img.ndim >= 3:
#                 d = img.shape[2]
#             else:
#                 d = 1  # fallback for 2D images

#             depths.append(d)

#         self.max_depth = max(depths) if depths else 0

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, index: int):

#         img_path = self.files[index]

#         ext = os.path.splitext(img_path)[1].lower()

#         u_gt = None

#         if ext == ".npy":
#             img_raw = np.load(img_path)
#             self.channel = None

#             if self.path_gt:
#                 name_img = img_path.name
#                 ugt_name = name_img.replace("_img.npy", "_ugt.npy")
#                 path_ugt_file = self.path_gt / ugt_name

#                 ugt_raw = np.load(path_ugt_file)

#                 u_gt = np.stack(
#                     [
#                         reshape_img(ugt_raw[0], self.max_depth),
#                         reshape_img(ugt_raw[1], self.max_depth),
#                         reshape_img(ugt_raw[2], self.max_depth),
#                     ]
#                 )

#         elif ext in [".tiff", ".tif"]:
#             img_raw = tifffile.imread(img_path)
#         else:
#             raise ValueError(f"Unsupported file type: {ext}")

#         # Quantile normalization
#         data = (
#             img_raw[:, self.channel] if self.channel is not None else img_raw
#         )  # Normalize the channel if provided
#         # q1, q99 = np.quantile(data, [0.01, 0.99])
#         # img = np.clip((data - q1) / (q99 - q1), 0, 1)  # (2, Z, H, W)
#         img = np.clip(data / data.max(), 0, 1)

#         # Extract reference and moving images (I1 -> I2)
#         ref_numpy = img[0]
#         moving_numpy = img[1]

#         ref = reshape_img(ref_numpy, self.max_depth)  # (Z, H, W)
#         moving = reshape_img(moving_numpy, self.max_depth)  # (Z, H, W)

#         mask_gt = ref > 0.01

#         mask = (
#             compute_mask(np.stack([ref, moving], axis=0))
#             if self.mask
#             else np.ones(ref.shape[1:])
#         )
#         mask_2d = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

#         ref = torch.from_numpy(ref)  # (Z, H, W) numpy -> (Z, H, W) torch
#         moving = torch.from_numpy(moving)  # (Z, H, W) numpy -> (Z, H, W) torch

#         # Stack to create a scalar image where each temporal frame becomes a channel.
#         # This is necessary for augmentations to apply correctly on the two images.
#         scalar_img = torch.stack([ref, moving], dim=0)  # (C, Z, H, W) -> (2, Z, H, W)
#         subject = tio.Subject(stack=tio.ScalarImage(tensor=scalar_img))

#         if self.transform is not None:
#             subject = self.transform(subject)

#         return {
#             "ref": subject.stack.data[0:1],  # I1
#             "moving": subject.stack.data[1:2],  # I2
#             "mask": mask_2d,
#             "u_gt": u_gt,
#             "mask_gt": mask_gt,
#         }


class TractionDataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """

    def __init__(
        self,
        folder: Path,
        spatial_transforms: callable = None,
        intensity_transforms: callable = None,
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
        self.spatial_transforms = spatial_transforms
        self.intensity_transforms = intensity_transforms
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

        if self.path_gt is not None:
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
            ).astype(np.float32)

    elif ext in [".tiff", ".tif"]:
        img_raw = tifffile.imread(img_path)

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    data = img_raw[:, self.channel] if self.channel is not None else img_raw

    img = np.clip(
        data / (data.max() + 1e-8),
        0,
        1,
    ).astype(np.float32)

    ref_numpy = img[0]
    moving_numpy = img[1]

    ref = reshape_img(ref_numpy, self.max_depth).astype(np.float32)
    moving = reshape_img(moving_numpy, self.max_depth).astype(np.float32)

    mask_gt = ref > 0.01

    mask = (
        compute_mask(np.stack([ref, moving], axis=0))
        if self.mask
        else np.ones(ref.shape[1:], dtype=np.float32)
    )

    mask_2d = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)

    ref = torch.from_numpy(ref).float()
    moving = torch.from_numpy(moving).float()

    if u_gt is None:
        raise ValueError(
            f"No displacement field found for {img_path}. "
            "This dataset augmentation expects a ground-truth flow."
        )

    flow = torch.from_numpy(u_gt).float()

    # Shape: (2, Z, H, W)
    scalar_img = torch.stack([ref, moving], dim=0)

    spatial_subject = tio.Subject(
        stack=tio.ScalarImage(tensor=scalar_img),
        flow=tio.ScalarImage(tensor=flow),
    )

    if self.transform is not None:
        spatial_subject = self.spatial_transform(spatial_subject)

    transformed_img = spatial_subject.stack.data.cpu().numpy().copy()
    transformed_flow = spatial_subject.flow.data.cpu().numpy().copy()

    total_linear = np.eye(3, dtype=np.float32)

    for applied_transform in spatial_subject.history:
        transform_name = applied_transform.__class__.__name__

        # RandomFlip is stored as Flip in TorchIO history
        if transform_name == "Flip":
            axes = applied_transform.axes

            flip_linear = np.eye(3, dtype=np.float32)

            # x / LR
            if 0 in axes or "LR" in axes:
                flip_linear[0, 0] = -1

            # y / AP
            if 1 in axes or "AP" in axes:
                flip_linear[1, 1] = -1

            # z / IS
            if 2 in axes or "IS" in axes:
                flip_linear[2, 2] = -1

            total_linear = flip_linear @ total_linear

        elif transform_name == "Affine":
            sitk_transform = applied_transform.get_affine_transform(
                spatial_subject.stack
            )

            p0 = np.asarray(
                sitk_transform.TransformPoint((0.0, 0.0, 0.0)),
                dtype=np.float32,
            )
            px = np.asarray(
                sitk_transform.TransformPoint((1.0, 0.0, 0.0)),
                dtype=np.float32,
            )
            py = np.asarray(
                sitk_transform.TransformPoint((0.0, 1.0, 0.0)),
                dtype=np.float32,
            )
            pz = np.asarray(
                sitk_transform.TransformPoint((0.0, 0.0, 1.0)),
                dtype=np.float32,
            )

            output_to_input_linear_lps = np.stack(
                [
                    px - p0,
                    py - p0,
                    pz - p0,
                ],
                axis=1,
            )
            input_to_output_linear_lps = np.linalg.inv(output_to_input_linear_lps)

            lps_to_ras = np.diag([-1, -1, 1]).astype(np.float32)

            affine_linear_ras = lps_to_ras @ input_to_output_linear_lps @ lps_to_ras

            total_linear = affine_linear_ras @ total_linear

    transformed_flow = np.einsum(
        "ij,jzhw->izhw",
        total_linear,
        transformed_flow,
    )

    transformed_img = torch.from_numpy(transformed_img).float()
    transformed_flow = torch.from_numpy(transformed_flow).float()

    if self.intensity_transforms is not None:
        intensity_subject = tio.Subject(stack=tio.ScalarImage(tensor=transformed_img))

        intensity_subject = self.intensity_transforms(intensity_subject)

        transformed_img = intensity_subject.stack.data.float()

    return {
        "ref": transformed_img[0:1],  # (1, Z, H, W)
        "moving": transformed_img[1:2],  # (1, Z, H, W)
        "mask": mask_2d,  # (1, 1, H, W)
        "u_gt": transformed_flow,  # (3, Z, H, W)
        "mask_gt": torch.from_numpy(mask_gt).bool(),
    }
