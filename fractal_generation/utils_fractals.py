import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter

from fractal_generation.fractals_generation import IFSFunction3D


def random_orthogonal_matrix(rng):
    Q, R = np.linalg.qr(rng.normal(size=(3, 3)))

    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1
    Q = Q * signs

    return Q


def random_affine_matrix(
    rng,
    min_singular_value: float,
    max_singular_value: float,
):

    U = random_orthogonal_matrix(rng)
    V = random_orthogonal_matrix(rng)
    singular_values = rng.uniform(
        min_singular_value,
        max_singular_value,
        size=3,
    )

    S = np.diag(singular_values)

    A = U @ S @ V.T

    return A


def add_random_affine_ifs(
    ifs,
    rng,
    n_functions: int,
    min_singular_value: float,
    max_singular_value: float,
    translation_scale: float,
):
    """
    Add random affine transformations to the IFS.

    Each transform is:

        x -> A x + T

    with singular values of A constrained to be < 1.
    """

    for _ in range(n_functions):
        A = random_affine_matrix(
            rng,
            min_singular_value=min_singular_value,
            max_singular_value=max_singular_value,
        )

        T = rng.uniform(
            -translation_scale,
            translation_scale,
            size=3,
        )

        ifs.set_param(A, T, 1 / n_functions)


def generate_fractal_image(
    rng: np.random.Generator,
    n_functions: int,
    min_singular_value: float,
    max_singular_value: float,
    translation_scale: float,
    size_img_x: int,
    size_img_y: int,
    size_img_z: int,
    pad_x: int,
    pad_y: int,
    pad_z: int,
    nb_iterations: int,
):
    ifs = IFSFunction3D(
        prev_x=0.0,
        prev_y=0.0,
        prev_z=0.0,
        rng=rng,
    )

    add_random_affine_ifs(
        ifs,
        rng=rng,
        n_functions=n_functions,
        min_singular_value=min_singular_value,
        max_singular_value=max_singular_value,
        translation_scale=translation_scale,
    )

    ifs.calculate(nb_iterations=nb_iterations)

    image = ifs.make_intensity_volume_zyx(
        size_img_x=size_img_x,
        size_img_y=size_img_y,
        size_img_z=size_img_z,
        pad_x=pad_x,
        pad_y=pad_y,
        pad_z=pad_z,
    )

    return image


def rotation_matrix(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """
    R = Rz(alpha) @ Ry(beta) @ Rx(gamma)
    """
    return np.array(
        [
            [
                np.cos(alpha) * np.cos(beta),
                np.cos(alpha) * np.sin(beta) * np.sin(gamma)
                - np.sin(alpha) * np.cos(gamma),
                np.cos(alpha) * np.sin(beta) * np.cos(gamma)
                + np.sin(alpha) * np.sin(gamma),
            ],
            [
                np.sin(alpha) * np.cos(beta),
                np.sin(alpha) * np.sin(beta) * np.sin(gamma)
                + np.cos(alpha) * np.cos(gamma),
                np.sin(alpha) * np.sin(beta) * np.cos(gamma)
                - np.cos(alpha) * np.sin(gamma),
            ],
            [
                -np.sin(beta),
                np.cos(beta) * np.sin(gamma),
                np.cos(beta) * np.cos(gamma),
            ],
        ],
        dtype=float,
    )


def generate_pyvista_perlin_displacement(
    shape: list[int],
    amplitude_min: float,
    amplitude_max: float,
    freq_min: float,
    freq_max: float,
    scale_z: float,
    rng: np.random.Generator,
):
    assert len(shape) == 3
    nz, ny, nx = shape

    freq_x, freq_y, freq_z = (
        rng.uniform(freq_min, freq_max),
        rng.uniform(freq_min, freq_max),
        rng.uniform(freq_min, freq_max) * scale_z,
    )

    u_comps = []

    for _ in range(3):
        phase = tuple(rng.uniform(0, 1000, size=3))

        amplitude = rng.uniform(amplitude_min, amplitude_max)

        noise_func = pv.perlin_noise(
            amplitude=amplitude, freq=(freq_x, freq_y, freq_z), phase=phase
        )

        sampled = pv.sample_function(
            noise_func,
            bounds=(0, nx - 1, 0, ny - 1, 0, nz - 1),
            dim=(nx, ny, nz),
        )

        comp = sampled.point_data["scalars"].reshape((nz, ny, nx))
        u_comps.append(comp)

    uz = u_comps[0] / scale_z
    uy = u_comps[1]
    ux = u_comps[2]

    return np.stack([uz, uy, ux], axis=0)


def generate_random_gaussian_displacement(
    shape: list[int],
    sigma: float,
    alpha: float,
    scale_z: float,
    rng: np.random.Generator,
):
    assert len(shape) == 3

    uz = (
        gaussian_filter(rng.standard_normal(shape), sigma=sigma, mode="reflect")
        * alpha
        / scale_z
    )
    uy = (
        gaussian_filter(rng.standard_normal(shape), sigma=sigma, mode="reflect") * alpha
    )
    ux = (
        gaussian_filter(rng.standard_normal(shape), sigma=sigma, mode="reflect") * alpha
    )

    return np.stack([uz, uy, ux], axis=0)
