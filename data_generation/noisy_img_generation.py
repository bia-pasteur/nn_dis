import shutil
from pathlib import Path

import jsonargparse
import numpy as np

from config import NoiseParams


def add_noise(
    img: np.ndarray,
    std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add Gaussian noise to an image.

    Args:
        img (np.ndarray): Image.
        std (float): Standard deviation of the added Gaussian nosie.
        rng (np.random.Generator): Random generator for reproducibility.

    Returns:
        np.ndarray: Noisy image.
    """
    return np.clip(img + rng.normal(0.0, std, size=img.shape), 0.0, 1.0)


def main(noise_params: NoiseParams):
    folder_images = Path(noise_params.folder_images)
    folder_noisy_images = Path(noise_params.folder_noisy_images)
    folder_noisy_images.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=noise_params.seed)

    paths_img = sorted(
        path for path in folder_images.iterdir() if path.name.endswith("_img.npy")
    )
    paths_img_selected = rng.choice(
        paths_img, size=noise_params.nb_images, replace=False
    )

    stds_noise = np.linspace(0, noise_params.max_std, noise_params.nb_noisy_images)

    for path_img in paths_img_selected:
        rng_noise = np.random.default_rng(seed=rng.integers(0, 2**32))
        name_cell = path_img.name.replace("_img.npy", "")
        img_cell = np.load(path_img)
        shutil.copyfile(
            folder_images / f"{name_cell}_ugt.npy",
            folder_noisy_images / f"{name_cell}_ugt.npy",
        )

        for i, std_noise in enumerate(stds_noise, start=1):
            img_cell_noisy = add_noise(img_cell, std_noise, rng_noise)
            np.save(
                folder_noisy_images / f"{name_cell}_noise_{i:02d}_img.npy",
                img_cell_noisy,
            )


if __name__ == "__main__":
    jsonargparse.auto_cli(main, as_positional=False)
