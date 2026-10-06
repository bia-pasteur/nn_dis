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

import os
from pathlib import Path

import numpy as np
import torch
from bioio import BioImage

# Set backends before importing voxelmorph-related modules
os.environ["NEURITE_BACKEND"] = "pytorch"
os.environ["VXM_BACKEND"] = "pytorch"

# Local imports
import importlib

import network.models
from scripts.utils import reshape_img

importlib.reload(network.models)


def get_img_spacing(
    path_img: Path,
    target_d: int = 32,
    target_h: int = 128,
    target_w: int = 128,
    multiple: int = 16,
    channel: int | None = None,
    default_spacing: tuple[float, float, float] | None = None,
    crop: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None = None,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """
    Load and preprocess an image pair for registration.

    Applies quantile normalization and reshapes frames to target dimensions.
    Supports both .npy and .tiff/.tif files (via bioio).

    Args:
        path_img: Path to the image file (.npy or .tiff/.tif)
        target_d: Target depth dimension
        target_h: Target height dimension
        target_w: Target width dimension
        multiple: Dimension must be multiple of this value
        channel: Channel index to extract (for multi-channel images)
        default_spacing: Default voxel spacing (Z, Y, X) if not in metadata
        crop: Optional cropping ranges as ((d_start, d_end), (h_start, h_end), (w_start, w_end)).
              If None, no cropping is applied.

    Returns:
        Preprocessed image stack and updated spacing:
            - np.ndarray: Preprocessed image stack (2, D, H, W)
            - Tuple[float, float, float]: Updated spacing after reshaping (Z, Y, X)
    """
    ext = os.path.splitext(path_img)[1].lower()

    if ext == ".npy":
        img = np.load(path_img)  # expected shape: (T, D, H, W)
        original_spacing = (
            default_spacing if default_spacing is not None else (1.0, 1.0, 1.0)
        )

        if channel is not None:
            raise ValueError(".npy files have no channel dimension support here.")

    else:
        img_raw = BioImage(path_img)
        pps = img_raw.physical_pixel_sizes

        z_spacing = pps.Z if pps.Z is not None else 1.0

        if pps.Y is not None and pps.X is not None:
            y_spacing = pps.Y
            x_spacing = pps.X
        elif pps.Y is not None and pps.X is None:
            y_spacing = pps.Y
            x_spacing = pps.Y
        elif pps.X is not None and pps.Y is None:
            y_spacing = pps.X
            x_spacing = pps.X
        else:  # both None
            y_spacing = 1.0
            x_spacing = 1.0

        original_spacing = (z_spacing, y_spacing, x_spacing)

        if channel is not None:
            img = img_raw.get_image_data("TZYX", C=channel)
        else:
            img = img_raw.data

    if crop is not None:
        (t_start, t_end), (d_start, d_end), (h_start, h_end), (w_start, w_end) = crop
        img = img[t_start:t_end, d_start:d_end, h_start:h_end, w_start:w_end]

    original_shape = img.shape  # (T, D, H, W)
    _, orig_h, orig_w = original_shape[1:]

    img = np.clip(img / img.max(), 0, 1)

    # Reshape frames
    img_0_reshaped = reshape_img(img[0], target_d, target_h, target_w, multiple)
    img_1_reshaped = reshape_img(img[1], target_d, target_h, target_w, multiple)

    # Compute new spacing after reshaping
    new_spacing = (
        original_spacing[0],  # Z unchanged
        original_spacing[1] * (orig_h / target_h),  # Y scaled
        original_spacing[2] * (orig_w / target_w),  # X scaled
    )
    new_spacing = tuple(round(s, 2) for s in new_spacing)

    img_padded = np.stack([img_0_reshaped, img_1_reshaped], axis=0)

    return img_padded, new_spacing


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

    model_path = f"{folder_path}/best.pt"  # best.pt"

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
    nb_warps: int = 5,
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


# def main():
#     parser = argparse.ArgumentParser(
#         description="Register a ref image to a moving image using VoxelMorph."
#     )
#     parser.add_argument(
#         "--input",
#         required=True,
#         type=Path,
#         help="Input image file (.npy or .tiff/.tif) containing both ref and moving frames",
#     )
#     parser.add_argument(
#         "--model",
#         required=True,
#         type=Path,
#         help="Path to trained model checkpoint (.pt file)",
#     )
#     parser.add_argument(
#         "--output_dir", required=True, type=Path, help="Directory to save output files"
#     )
#     parser.add_argument(
#         "--channel",
#         type=int,
#         default=None,
#         help="Channel index to extract (for multi-channel images)",
#     )
#     parser.add_argument(
#         "--model_type",
#         type=str,
#         default="aniso",
#         choices=["aniso", "standard"],
#         help="Model architecture type",
#     )
#     parser.add_argument(
#         "--moving_d", type=int, default=32, help="Target depth dimension"
#     )
#     parser.add_argument(
#         "--moving_h", type=int, default=128, help="Target height dimension"
#     )
#     parser.add_argument(
#         "--moving_w", type=int, default=128, help="Target width dimension"
#     )
#     parser.add_argument(
#         "--iterations",
#         type=int,
#         default=7,
#         help="Number of registration refinement iterations",
#     )
#     parser.add_argument(
#         "--default_spacing",
#         type=float,
#         nargs=3,
#         default=[1.0, 0.13, 0.13],
#         help="Default voxel spacing (Z, Y, X) if not in metadata",
#     )

#     args = parser.parse_args()

#     # Create output directory
#     args.output_dir.mkdir(parents=True, exist_ok=True)

#     # Load and preprocess image
#     print(f"Loading image from {args.input}...")
#     img_padded, spacing = get_img_spacing(
#         path_img=args.input,
#         moving_d=args.moving_d,
#         moving_h=args.moving_h,
#         moving_w=args.moving_w,
#         multiple=16,
#         channel=args.channel,
#         default_spacing=tuple(args.default_spacing),
#     )
#     print(f"Image loaded with spacing: {spacing}")

#     # Load model
#     print(f"Loading model from {args.model}...")
#     model, device = load_model(str(args.model), args.model_type)
#     print(f"Model loaded on device: {device}")

#     # Register images
#     print(f"Running registration with {args.iterations} iterations...")
#     displacement, warped_ref = register_images(
#         img_padded, model, device, iterations=args.iterations
#     )

#     # Convert outputs to numpy
#     displacement_np = displacement.detach().cpu().numpy()
#     warped_np = warped_ref.detach().cpu().numpy()[0, 0]

#     # Save displacement field
#     dis_path = args.output_dir / f"displacement_{args.input.stem}.npy"
#     np.save(dis_path, displacement_np)
#     print(f"Saved displacement field to {dis_path}")

#     # Save warped image
#     warped_path = args.output_dir / f"warped_{args.input.stem}.npy"
#     np.save(warped_path, warped_np)
#     print(f"Saved warped image to {warped_path}")

#     # Save spacing info
#     spacing_path = args.output_dir / f"spacing_{args.input.stem}.txt"
#     with open(spacing_path, "w") as f:
#         f.write(f"spacing: {spacing}\n")
#     print(f"Saved spacing info to {spacing_path}")

#     print("Registration complete!")


# if __name__ == "__main__":
#     main()
