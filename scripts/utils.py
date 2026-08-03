"""Useful functions"""

import numpy as np
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


def reshape_disp(
    disp: np.ndarray,
    target_d: int,
    target_h: int = 128,
    target_w: int = 128,
    multiple: int = 16,
) -> np.ndarray:
    """
    Redimensionne la displacement map (C, N, D, H, W)
    C = 3 (composantes x,y,z)
    N = nombre d'exemples (souvent 2)
    """
    C, N, D, H, W = disp.shape
    out = np.zeros((C, N, target_d, target_h, target_w), dtype=np.float32)
    for c in range(C):
        for n in range(N):
            out[c, n] = reshape_img(disp[c, n], target_d, target_h, target_w, multiple)
    return out


def rmse(u: np.ndarray, h: np.ndarray) -> float:
    """
    Computes the Root Mean Square Error (RMSE) between two vector fields.

    Args:
        u (np.ndarray):
            Ground-truth field, shape `(d, H, W, ...)`.
        h (np.ndarray):
            Estimated or predicted field, same shape as `u`.

    Returns:
        float:
            The root mean square error value between `u` and `h`.
    """
    diff = u - h
    mse = np.mean(diff**2)
    return np.sqrt(mse)
