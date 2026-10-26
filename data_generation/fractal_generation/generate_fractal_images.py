import os

import jsonargparse
import numpy as np
from scipy.interpolate import griddata

from config import FractalParams
from data_generation.fractal_generation.utils_fractals import (
    generate_fractal_image,
    generate_pyvista_perlin_displacement,
    generate_random_gaussian_displacement,
    rigid_and_gaussian_warp,
    rigid_displacement_and_warp,
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
    sigma_gaussian_min: float | None = None,
    sigma_gaussian_max: float | None = None,
    alpha_gaussian_min: float | None = None,
    alpha_gaussian_max: float | None = None,
    alpha_rigid_min: float | None = None,
    alpha_rigid_max: float | None = None,
    beta_rigid_min: float | None = None,
    beta_rigid_max: float | None = None,
    gamma_rigid_min: float | None = None,
    gamma_rigid_max: float | None = None,
    translation_min: float | None = None,
    translation_max: float | None = None,
):
    """
    Create a 3D image of a fractal that deforms.

    The fractal is generated as an iterated function system (IFS) and deformed using
    a displacement field created either by Perlin noise, Gaussian noise, rigid motion
    or rigid motion + Gaussian noise.

    Args:
        seed (int): Seed for reproducibility.
        n_functions (int): Number of affine maps in the IFS.
        min_singular_value (float): Lower bound for the singular values of each affine matrix.
        max_singular_value (float): Upper bound for the singular values of each affine matrix (<1).
        nb_iterations (int): Number of IFS iterations.
        translation_scale (float): Translation component bounds (Uniform(-translation_scale, translation_scale)).
        size_img_x (int): Size of the output volume in x.
        size_img_y (int): Size of the output volume in y.
        size_img_z (int): Size of the output volume in z.
        pad_x (int): Margin in voxels left empty on x edges.
        pad_y (int): Margin in voxels left empty on y edges.
        pad_z (int): Margin in voxels left empty on z edges.
        amplitude_min_dis (float | None, optional): Perlin noise minimum amplitude. Defaults to None.
        amplitude_max_dis (float | None, optional): Perlin noise maximum amplitude. Defaults to None.
        freq_min_dis (float | None, optional): Perlin noise minimum frequency. Defaults to None.
        freq_max_dis (float | None, optional): Perlin noise maximum frequency. Defaults to None.
        sigma_gaussian_min (float | None, optional): Gaussian minimum filter sigma. Defaults to None.
        sigma_gaussian_max (float | None, optional): Gaussian maximum filter sigma. Defaults to None.
        alpha_gaussian_min (float | None, optional): Gaussian minimum magnitude. Defaults to None.
        alpha_gaussian_max (float | None, optional): Gaussian maximum magnitude. Defaults to None.
        alpha_rigid_min (float | None, optional): Rigid minimum rotation angle around z (degrees). Defaults to None.
        alpha_rigid_max (float | None, optional): Rigid maximum rotation angle around z (degrees). Defaults to None.
        beta_rigid_min (float | None, optional): Rigid minimum rotation angle around y (degrees). Defaults to None.
        beta_rigid_max (float | None, optional): Rigid maximum rotation angle around y (degrees). Defaults to None.
        gamma_rigid_min (float | None, optional): Rigid minimum rotation angle around x (degrees). Defaults to None.
        gamma_rigid_max (float | None, optional): Rigid maximum rotation angle around x (degrees). Defaults to None.
        translation_min (float | None, optional): Rigid minimum translation per axis. Defaults to None.
        translation_max (float | None, optional): Rigid maximum translation per axis. Defaults to None.

    Raises:
        ValueError: If neither the Perlin nor the Gaussian displacement
            parameters are fully specified (and no rigid parameters are given).

    Returns:
        tuple[np.ndarray, np.ndarray]: A pair (img, u) where

        - img is a (2, size_img_z, size_img_y, size_img_x) stack of the
          original fractal image and its deformed version
        - u is the (3, size_img_z, size_img_y, size_img_x) displacement
          field (uz, uy, ux), in voxel units.
    """

    rng = np.random.default_rng(seed)

    perlin_params = [
        amplitude_min_dis,
        amplitude_max_dis,
        freq_min_dis,
        freq_max_dis,
    ]
    gaussian_params = [
        sigma_gaussian_min,
        sigma_gaussian_max,
        alpha_gaussian_min,
        alpha_gaussian_max,
    ]
    rigid_params = [
        alpha_rigid_min,
        alpha_rigid_max,
        beta_rigid_min,
        beta_rigid_max,
        gamma_rigid_min,
        gamma_rigid_max,
        translation_min,
        translation_max,
    ]

    has_all_perlin = all(p is not None for p in perlin_params)
    has_any_perlin = any(p is not None for p in perlin_params)

    has_all_gaussian = all(p is not None for p in gaussian_params)
    has_any_gaussian = any(p is not None for p in gaussian_params)

    has_all_rigid = all(p is not None for p in rigid_params)
    has_any_rigid = any(p is not None for p in rigid_params)

    assert has_all_perlin == has_any_perlin, (
        "Incomplete Perlin parameters! You must pass all 4 parameters: "
        "amplitude_min_dis, amplitude_max_dis, freq_min_dis, freq_max_dis."
    )
    assert has_all_gaussian == has_any_gaussian, (
        "Incomplete Gaussian parameters! You must pass both 'sigma' and 'alpha'."
    )
    assert has_all_rigid == has_any_rigid, (
        "Incomplete rigid deformation parameters! You must pass all 4 parameters: "
        "alpha_rigid_min, alpha_rigid_max, beta_rigid_min, beta_rigid_max, gamma_rigid_min, gamma_rigid_max, translation_min, translation_max."
    )

    rng = np.random.default_rng(seed)

    fractal_img = generate_fractal_image(
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

    shape = fractal_img.shape

    if has_all_rigid:
        if has_all_gaussian:
            u = rigid_and_gaussian_warp(
                image=fractal_img,
                alpha_min=alpha_rigid_min,
                alpha_max=alpha_rigid_max,
                beta_min=beta_rigid_min,
                beta_max=beta_rigid_max,
                gamma_min=gamma_rigid_min,
                gamma_max=gamma_rigid_max,
                translation_min=translation_min,
                translation_max=translation_max,
                sigma_gaussian_min=sigma_gaussian_min,
                sigma_gaussian_max=sigma_gaussian_max,
                alpha_gaussian_min=alpha_gaussian_min,
                alpha_gaussian_max=alpha_gaussian_max,
                rng=rng,
                scale_z=shape[1] / shape[0],
                order=3,
                mode="constant",
            )
        else:
            u = rigid_displacement_and_warp(
                image=fractal_img,
                alpha_min=alpha_rigid_min,
                alpha_max=alpha_rigid_max,
                beta_min=beta_rigid_min,
                beta_max=beta_rigid_max,
                gamma_min=gamma_rigid_min,
                gamma_max=gamma_rigid_max,
                translation_min=translation_min,
                translation_max=translation_max,
                rng=rng,
                scale_z=shape[1] / shape[0],
                order=3,
                mode="constant",
            )

    else:
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
                shape=shape,
                sigma_min=sigma_gaussian_min,
                sigma_max=sigma_gaussian_max,
                alpha_min=alpha_gaussian_min,
                alpha_max=alpha_gaussian_max,
                scale_z=shape[1] / shape[0],
                rng=rng,
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

    destination_points = np.column_stack(
        [
            new_z.ravel(),
            new_y.ravel(),
            new_x.ravel(),
        ]
    )

    values = fractal_img.ravel()

    fractal_deformed = griddata(
        points=destination_points,
        values=values,
        xi=(grid_z, grid_y, grid_x),
        method="nearest",
        fill_value=0.0,
    )

    img = np.stack([fractal_img, fractal_deformed], axis=0)

    return img, u


def main(fractal_params: FractalParams):
    for seed in range(fractal_params.seeds):
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
                sigma_gaussian_min=fractal_params.sigma_gaussian_min,
                sigma_gaussian_max=fractal_params.sigma_gaussian_max,
                alpha_gaussian_min=fractal_params.alpha_gaussian_min,
                alpha_gaussian_max=fractal_params.alpha_gaussian_max,
                alpha_rigid_min=fractal_params.alpha_rigid_min,
                alpha_rigid_max=fractal_params.alpha_rigid_max,
                beta_rigid_min=fractal_params.beta_rigid_min,
                beta_rigid_max=fractal_params.beta_rigid_max,
                gamma_rigid_min=fractal_params.gamma_rigid_min,
                gamma_rigid_max=fractal_params.gamma_rigid_max,
                translation_min=fractal_params.translation_min,
                translation_max=fractal_params.translation_max,
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
