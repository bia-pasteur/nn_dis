import os

import jsonargparse
import numpy as np
from scipy.ndimage import map_coordinates

from config import FractalParams
from fractal_generation.utils_fractals import (
    generate_fractal_image,
    generate_pyvista_perlin_displacement,
    generate_random_gaussian_displacement,
)


def create_full_fractal_image(
    seed: int,
    n_functions: int,
    min_singular_value: float,
    max_singular_value: float,
    nb_iterations: int,
    translation_scale: float,
    size_img_x: int,
    size_img_y: int,
    size_img_z: int,
    pad_x: int,
    pad_y: int,
    pad_z: int,
    amplitude_min_dis: float | None = None,
    amplitude_max_dis: float | None = None,
    freq_min_dis: float | None = None,
    freq_max_dis: float | None = None,
    sigma: float | None = None,
    alpha: float | None = None,
):
    perlin_params = [
        amplitude_min_dis,
        amplitude_max_dis,
        freq_min_dis,
        freq_max_dis,
    ]
    gaussian_params = [sigma, alpha]

    has_all_perlin = all(p is not None for p in perlin_params)
    has_any_perlin = any(p is not None for p in perlin_params)

    has_all_gaussian = all(p is not None for p in gaussian_params)
    has_any_gaussian = any(p is not None for p in gaussian_params)

    assert has_all_perlin == has_any_perlin, (
        "Incomplete Perlin parameters! You must pass all 4 parameters: "
        "amplitude_min_dis, amplitude_max_dis, freq_min_dis, freq_max_dis."
    )
    assert has_all_gaussian == has_any_gaussian, (
        "Incomplete Gaussian parameters! You must pass both 'sigma' and 'alpha'."
    )
    rng = np.random.default_rng(seed)

    reference = generate_fractal_image(
        rng=rng,
        n_functions=n_functions,
        min_singular_value=min_singular_value,
        max_singular_value=max_singular_value,
        translation_scale=translation_scale,
        size_img_x=size_img_x,
        size_img_y=size_img_y,
        size_img_z=size_img_z,
        pad_x=pad_x,
        pad_y=pad_y,
        pad_z=pad_z,
        nb_iterations=nb_iterations,
    )

    shape = reference.shape

    if has_all_perlin:
        u = generate_pyvista_perlin_displacement(
            shape=shape,
            amplitude_min=amplitude_min_dis,
            amplitude_max=amplitude_max_dis,
            freq_min=freq_min_dis,
            freq_max=freq_max_dis,
            scale_z=shape[1] / shape[0],
            rng=rng,
        )

    elif has_all_gaussian:
        u = generate_random_gaussian_displacement(
            shape=shape, scale_z=shape[1] / shape[0], sigma=6.0, alpha=100.0, rng=rng
        )

    else:
        raise ValueError(
            "You must specify EITHER all Perlin displacement parameters "
            "OR all Gaussian parameters (sigma, alpha), but NOT both or neither."
        )

    grid_z, grid_y, grid_x = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing="ij",
    )

    new_z = grid_z + u[0]
    new_y = grid_y + u[1]
    new_x = grid_x + u[2]

    indices = [new_z, new_y, new_x]

    moving = map_coordinates(reference, indices, order=3, mode="reflect")

    img = np.stack([reference, moving], axis=0)

    return img, u


def main(fractal_params: FractalParams):
    for seed in fractal_params.seeds:
        for n_functions in fractal_params.nb_functions:
            img, u = create_full_fractal_image(
                seed=seed,
                n_functions=n_functions,
                min_singular_value=fractal_params.min_singular_value,
                max_singular_value=fractal_params.max_singular_value,
                translation_scale=fractal_params.translation_scale,
                nb_iterations=fractal_params.nb_iterations,
                size_img_x=fractal_params.size_img_x,
                size_img_y=fractal_params.size_img_y,
                size_img_z=fractal_params.size_img_z,
                pad_x=fractal_params.pad_x,
                pad_y=fractal_params.pad_y,
                pad_z=fractal_params.pad_z,
                amplitude_min_dis=fractal_params.amplitude_min_dis,
                amplitude_max_dis=fractal_params.amplitude_max_dis,
                freq_min_dis=fractal_params.freq_min_dis,
                freq_max_dis=fractal_params.freq_max_dis,
                sigma=fractal_params.sigma,
                alpha=fractal_params.alpha,
            )

            output_dir = os.path.join(fractal_params.save_dir)

            os.makedirs(output_dir, exist_ok=True)

            np.save(
                os.path.join(output_dir, f"fractal_{seed}_{n_functions}_funcs_ugt.npy"),
                u,
            )
            np.save(
                os.path.join(output_dir, f"fractal_{seed}_{n_functions}_funcs_img.npy"),
                img,
            )


if __name__ == "__main__":
    jsonargparse.auto_cli(main, as_positional=False)
