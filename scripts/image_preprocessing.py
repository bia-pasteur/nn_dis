import os
from pathlib import Path

import numpy as np
from bioio import BioImage
from scipy.ndimage import zoom


def reshape_img(
    image: np.ndarray,
    target_d: int = 32,
    target_h: int = 128,
    target_w: int = 128,
    multiple: int = 16,
) -> np.ndarray:
    """
    Pads image depth (D) to max_depth and resizes H, W to target_h and target_w

    Args:
        image (np.ndarray): 3D array (D, H, W)
        max_depth (int): Target depth to pad/center-pad
        target_h (int): target height H
        target_w (int): target width W

    Returns:
        np.ndarray: Reshaped image (target_d, target_h, target_w)
    """
    d = image.shape[0]

    if d < target_d:
        pad_before = (target_d - d) // 2
        pad_after = target_d - d - pad_before
        image = np.pad(
            image,
            ((pad_before, pad_after), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    resized = np.zeros((target_d, target_h, target_w), dtype=np.float32)

    for z in range(target_d):
        scale_h = target_h / image.shape[1]
        scale_w = target_w / image.shape[2]
        resized[z] = zoom(image[z], (scale_h, scale_w), order=1)

    out = resized

    def pad_to_multiple(arr, multiple):
        d, h, w = arr.shape
        pad_d = (multiple - d % multiple) % multiple
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple

        pad_before_d, pad_after_d = pad_d // 2, pad_d - pad_d // 2
        pad_before_h, pad_after_h = pad_h // 2, pad_h - pad_h // 2
        pad_before_w, pad_after_w = pad_w // 2, pad_w - pad_w // 2

        return np.pad(
            arr,
            (
                (pad_before_d, pad_after_d),
                (pad_before_h, pad_after_h),
                (pad_before_w, pad_after_w),
            ),
            mode="constant",
            constant_values=0,
        )

    out = pad_to_multiple(out, multiple)
    return out.astype(np.float32)


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
        img_raw = np.load(path_img)  # expected shape: (T, D, H, W)
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
        img_raw = img_raw.get_image_data("TCZYX")

    img = img_raw[:, channel] if channel is not None else img_raw

    if crop is not None:
        (t_start, t_end), (d_start, d_end), (h_start, h_end), (w_start, w_end) = crop
        img = img[t_start:t_end, d_start:d_end, h_start:h_end, w_start:w_end]

    original_shape = img.shape  # (T, D, H, W)
    _, orig_h, orig_w = original_shape[1:]

    img_norm = np.clip(
        img / (img.max() + 1e-8),
        0,
        1,
    ).astype(np.float32)

    # q_low, q_high = np.quantile(img, [0.01, 0.99])

    # img_norm = np.clip(
    #     (img - q_low) / (q_high - q_low + 1e-8),
    #     0,
    #     1,
    # ).astype(np.float32)

    # Reshape frames
    ref = reshape_img(img_norm[0], target_d, target_h, target_w, multiple)
    moving = reshape_img(img_norm[1], target_d, target_h, target_w, multiple)

    # Compute new spacing after reshaping
    new_spacing = (
        original_spacing[0],  # Z unchanged
        original_spacing[1] * (orig_h / target_h),  # Y scaled
        original_spacing[2] * (orig_w / target_w),  # X scaled
    )
    new_spacing = tuple(round(s, 2) for s in new_spacing)

    return np.stack([ref, moving], axis=0).astype(np.float32), new_spacing


def get_ugt(
    img_path: Path,
    path_gt: Path,
    target_d: int = 32,
    target_h: int = 128,
    target_w: int = 128,
    multiple: int = 16,
):
    name_img = img_path.name
    ugt_name = name_img.replace("_img.npy", "_ugt.npy")
    path_ugt_file = path_gt / ugt_name

    ugt_raw = np.load(path_ugt_file)

    u_gt = np.stack(
        [
            reshape_img(ugt_raw[0], target_d, target_h, target_w, multiple),
            reshape_img(ugt_raw[1], target_d, target_h, target_w, multiple),
            reshape_img(ugt_raw[2], target_d, target_h, target_w, multiple),
        ]
    ).astype(np.float32)

    return u_gt


def get_max_depths(files: list[Path]):
    depths = []
    for f in files:
        if f.suffix.lower() == ".npy":
            img = np.load(f)
        elif f.suffix.lower() in [".tiff", ".tif"]:
            img = BioImage(f)
        else:
            continue  # skip unknown file types

        if img.ndim == 4:
            d = img.shape[1]
        elif img.ndim >= 3:
            d = img.shape[2]
        else:
            d = 1  # fallback for 2D images

        depths.append(d)

    return max(depths) if depths else 0
