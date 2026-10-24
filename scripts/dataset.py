"""This is useful to create the datasets"""

from pathlib import Path

import numpy as np
import torch
import torchio as tio
from scipy.ndimage import binary_dilation, binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.morphology import disk, remove_small_objects
from torch.utils.data import Dataset

from scripts.image_preprocessing import get_img_spacing, get_max_depths, get_ugt


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
            # tio.RandomAffine(
            #     scales=(0.9, 1.1),
            #     degrees=(-180, 180, 0, 0, 0, 0),
            #     translation=7,
            #     p=0.5,
            # ),
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
            tio.RandomNoise(std=(0, 0.1), p=0.3),
            # tio.RandomBlur(std=(0, 1), p=0.2),
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
        default_spacing: tuple[float, float, float] | None = (4.0, 1.0, 1.0),
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
        self.default_spacing = default_spacing

        self.max_depth = get_max_depths(self.files)

    def __len__(self):
        return len(self.files)

    def _get_spatial_transform_linear_matrix(self, subject: tio.Subject) -> np.ndarray:
        """
        Recover the cumulative linear transformation applied to displacement
        vectors from TorchIO's transform history.

        The returned matrix acts on vectors in (dx, dy, dz) order.
        """
        total_linear_xyz = np.eye(3, dtype=np.float32)

        for applied_transform in subject.history:
            transform_name = applied_transform.__class__.__name__

            if transform_name == "Flip":
                flip_linear_xyz = np.eye(3, dtype=np.float32)
                axes = applied_transform.axes

                # Physical x direction
                if 0 in axes or "LR" in axes:
                    flip_linear_xyz[0, 0] = -1

                # Physical y direction
                if 1 in axes or "AP" in axes:
                    flip_linear_xyz[1, 1] = -1

                # Physical z direction
                if 2 in axes or "IS" in axes:
                    flip_linear_xyz[2, 2] = -1

                total_linear_xyz = flip_linear_xyz @ total_linear_xyz

            elif transform_name == "Affine":
                sitk_transform = applied_transform.get_affine_transform(subject.stack)

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

                # TorchIO/SimpleITK resampling transform maps output -> input.
                output_to_input_linear_lps = np.stack(
                    [px - p0, py - p0, pz - p0],
                    axis=1,
                )

                # Vector transformation should use input -> output.
                input_to_output_linear_lps = np.linalg.inv(output_to_input_linear_lps)

                # Convert SimpleITK LPS coordinates to XYZ/RAS-like coordinates.
                lps_to_ras = np.diag([-1, -1, 1]).astype(np.float32)

                affine_linear_xyz = lps_to_ras @ input_to_output_linear_lps @ lps_to_ras

                total_linear_xyz = affine_linear_xyz @ total_linear_xyz

        return total_linear_xyz

    def _apply_spatial_transforms(
        self,
        image: torch.Tensor,
        flow: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Preserve the original tensors when no spatial transform
        # was requested. This makes the non-augmented case unambiguous.
        if self.spatial_transforms is None:
            return image, flow
        image_tio = image.permute(0, 3, 2, 1).contiguous()
        flow_tio = flow.permute(0, 3, 2, 1).contiguous()

        subject = tio.Subject(
            stack=tio.ScalarImage(tensor=image_tio),
            flow=tio.ScalarImage(tensor=flow_tio),
        )

        subject = self.spatial_transforms(subject)

        transformed_image = subject.stack.data.permute(0, 3, 2, 1).contiguous()
        transformed_flow = subject.flow.data.permute(0, 3, 2, 1).contiguous()

        total_linear_xyz = self._get_spatial_transform_linear_matrix(subject)

        # Dataset convention -> (dz, dy, dx).
        # Rotation/reflection matrices ocnvention -> (dx, dy, dz)
        # Convertion: (dx, dy, dz)
        flow_xyz = transformed_flow[[2, 1, 0]].cpu().numpy()

        flow_xyz = np.einsum(
            "ij,jzhw->izhw",
            total_linear_xyz,
            flow_xyz,
        )

        # Convert back: (dz, dy, dx).
        transformed_flow = torch.from_numpy(flow_xyz[[2, 1, 0]]).float()

        return transformed_image, transformed_flow

    def _apply_intensity_transforms(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        if self.intensity_transforms is None:
            return image

        # (C, Z, H, W) -> (C, W, H, Z)
        image_tio = image.permute(0, 3, 2, 1).contiguous()

        subject = tio.Subject(
            stack=tio.ScalarImage(tensor=image_tio),
        )

        subject = self.intensity_transforms(subject)

        # (C, W, H, Z) -> (C, Z, H, W)
        transformed_image = subject.stack.data.permute(0, 3, 2, 1).contiguous().float()

        assert transformed_image.shape == image.shape

        return transformed_image

    def __getitem__(self, index: int):
        img_path = self.files[index]

        img, spacing = get_img_spacing(
            img_path,
            target_d=self.max_depth,
            default_spacing=self.default_spacing,
        )

        if self.path_gt is None:
            raise ValueError(
                "This dataset requires `path_gt` to load ground-truth flow."
            )

        u_gt = get_ugt(img_path=img_path, path_gt=self.path_gt, target_d=self.max_depth)

        if u_gt is None:
            raise ValueError(
                f"No displacement field found for {img_path}. "
                "This dataset requires a ground-truth flow."
            )

        ref = torch.from_numpy(img[0]).float()
        moving = torch.from_numpy(img[1]).float()
        flow = torch.from_numpy(u_gt).float()

        # Original, pre-augmentation image stack: (2, Z, H, W)
        original_image = torch.stack([ref, moving], dim=0)

        image, flow = self._apply_spatial_transforms(
            image=original_image,
            flow=flow,
        )

        # Mask derived from the geometrically aligned image.
        mask_gt = image[0] > 0.0005

        # Intensity transforms do not affect flow or mask geometry.
        image = self._apply_intensity_transforms(image)

        # This is intentionally the original 2D mask, as in your code.
        mask = (
            compute_mask(np.stack([img[0], img[1]], axis=0))
            if self.mask
            else np.ones(img[0].shape[1:], dtype=np.float32)
        )
        mask_2d = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)

        return {
            "ref": image[0:1],
            "moving": image[1:2],
            "mask": mask_2d,
            "u_gt": flow,
            "mask_gt": mask_gt.bool(),
            "spacing": torch.as_tensor(spacing, dtype=torch.float32),
        }
