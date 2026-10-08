#!/usr/bin/env python

"""
Image registration script using VoxelMorph models.

This script registers a ref image to a moving image using a trained VoxelMorph model,
following the workflow from tests_synthetic.ipynb.

Supports both .npy and .tiff/.tif files (via bioio).
Outputs the warped ref image and displacement field.

Example usage:
    python scripts/register.py \
        --input path/to/image.tiff \
        --model path/to/model.pt \
        --output_dir path/to/results \
        --channel 2 \
        --moving_d 32 --moving_h 128 --moving_w 128 \
        --iterations 7

If you use this code, please cite the appropriate VoxelMorph papers.
"""

import importlib
from pathlib import Path

import numpy as np
import torch

import network.models
from scripts.image_preprocessing import get_img_spacing

importlib.reload(network.models)


def load_model(
    folder_path: str,
    model_type: str = "aniso",
    features: list[int] = [16, 32, 32, 64, 64],
) -> torch.nn.Module:
    """
    Load a trained VoxelMorph model from a checkpoint.

    Args:
        folder_path: Path to the folder containing model checkpoint (.pt file)
        model_type: Either "aniso" for NetworkPairwiseAnisotropic or
                    any other value for NetworkPairwise
        features: Feature channels used in the model

    Returns:
        torch.nn.Module: Loaded model in eval mode
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = f"{folder_path}/checkpoint_epoch200.pt"  # best.pt"

    if model_type == "aniso":
        model = network.models.NetworkPairwiseAnisotropic(
            ndim=3,
            ref_channels=1,
            moving_channels=1,
            features=features,
        ).to(device)
    else:
        model = network.models.NetworkPairwise(
            ndim=3,
            ref_channels=1,
            moving_channels=1,
            features=features,
        ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    return model


def compose_flows(transformer, previous_flow, new_flow):
    return new_flow + transformer(previous_flow, new_flow)


def register(
    img_path: Path,
    folder_path: Path,
    model_type: str = "aniso",
    features: list[int] = [16, 32, 32, 64, 64],
    channel: int | None = None,
    default_spacing: list[float] | None = None,
    target_d: int = 32,
    target_h: int = 128,
    target_w: int = 128,
    multiple: int = 16,
    crop: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None = None,
) -> tuple[np.ndarray, tuple[float, float, float], torch.Tensor, np.ndarray]:
    """
    Register a moving image to a reference image using a trained VoxelMorph model.

    Performs iterative registration by applying the model multiple times to refine the alignment.

    Args:
        img_path: Path to the image file. img[0] is the reference image,
                  img[1] is the moving image.
        folder_path: Path to the folder containing model checkpoint (.pt file)
        model_type: Type of model used.
                    Can be 'iso' for isotropic images or 'aniso' for anisotropic images. Defaults to "aniso".
        features: Feature channels used in the model. Defaults to [16, 32, 32, 64, 64].
        nb_warps: Number of registration refinement iterations. Defaults to 5.
        channel: Channel index to work on. None for images without channels. Defaults to None.
        default_spacing: Default voxel spacing (Z, Y, X) if not in metadata. Defaults to None.
        target_d: Target depth dimension. Defaults to 32.
        target_h: Target height dimension. Defaults to 128.
        target_w: Target width dimension. Defaults to 128.
        multiple: Dimension must be multiple of this value. Defaults to 16.
        crop: Optional cropping ranges as ((d_start, d_end), (h_start, h_end), (w_start, w_end)).
              If None, no cropping is applied. Defaults to None.

    Returns:
        Tuple containing:
            - np.ndarray: Preprocessed image stack (2, D, H, W)
            - Tuple[float, float, float]: Voxel spacing (Z, Y, X)
            - torch.Tensor: Final displacement field tensor
            - np.ndarray: Warped moving image
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(
        folder_path=folder_path, model_type=model_type, features=features
    )

    img, spacing = get_img_spacing(
        path_img=img_path,
        target_d=target_d,
        target_h=target_h,
        target_w=target_w,
        multiple=multiple,
        channel=channel,
        default_spacing=default_spacing,
        crop=crop,
    )

    ref = (
        torch.from_numpy(img[0]).unsqueeze(0).unsqueeze(0).to(device)
    )  # (1, 1, Z, H, W)
    moving = (
        torch.from_numpy(img[1]).unsqueeze(0).unsqueeze(0).to(device)
    )  # (1, 1, Z, H, W)

    displacement, warped_moving = model(ref, moving, return_warped_moving=True)

    return (
        img,
        spacing,
        displacement.detach().numpy(),
        warped_moving.detach().numpy()[0, 0],
    )
