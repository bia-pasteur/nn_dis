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


def rigid_displacement_and_warp(
    image: np.ndarray,
    alpha_min: float,
    alpha_max: float,
    beta_min: float,
    beta_max: float,
    gamma_min: float,
    gamma_max: float,
    translation_min: float,
    translation_max: float,
    rng: np.random.Generator,
    scale_z: float = 1.0,
    order: int = 1,
    mode: str = "constant",
):
    assert (
        alpha_min < alpha_max
        and beta_min < beta_max
        and gamma_min < gamma_max
        and translation_min < translation_max
    ), "Min values must be inferior to max values, check config file !"

    alpha = rng.uniform(np.deg2rad(alpha_min), np.deg2rad(alpha_max))
    beta = rng.uniform(np.deg2rad(beta_min), np.deg2rad(beta_max))
    gamma = rng.uniform(np.deg2rad(gamma_min), np.deg2rad(gamma_max))

    t = rng.uniform(translation_min, translation_max, size=3)

    assert image.ndim == 3
    nz, ny, nx = image.shape

    R = rotation_matrix(alpha, beta, gamma)

    zz, yy, xx = np.meshgrid(
        np.arange(nz, dtype=float),
        np.arange(ny, dtype=float),
        np.arange(nx, dtype=float),
        indexing="ij",
    )

    x = xx
    y = yy
    z = zz * scale_z

    center = np.array(
        [
            (nx - 1) / 2.0,
            (ny - 1) / 2.0,
            (nz - 1) * scale_z / 2.0,
        ]
    )

    coords_1 = np.stack([x, y, z], axis=0)  # (3, nz, ny, nx)

    centered_1 = coords_1 - center[:, None, None, None]

    coords_2 = (
        np.einsum("ij,jzyx->izyx", R, centered_1)
        + center[:, None, None, None]
        + t[:, None, None, None]
    )

    displacement_physical = coords_2 - coords_1

    ux = displacement_physical[0]
    uy = displacement_physical[1]
    uz = displacement_physical[2] / scale_z

    displacement = np.stack([uz, uy, ux], axis=0)

    return displacement


def generate_pyvista_perlin_displacement(
    shape: list[int],
    amplitude_min: float,
    amplitude_max: float,
    freq_min: float,
    freq_max: float,
    scale_z: float,
    rng: np.random.Generator,
):
    assert amplitude_min < amplitude_max and freq_min < freq_max, (
        "Min values must be inferior to max values, check config file !"
    )
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
    sigma_min: float,
    sigma_max: float,
    alpha_min: float,
    alpha_max: float,
    scale_z: float,
    rng: np.random.Generator,
):
    assert alpha_min < alpha_max and sigma_min < sigma_max, (
        "Min values must be inferior to max values, check config file !"
    )
    assert len(shape) == 3

    sigma = rng.uniform(sigma_min, sigma_max)
    alpha = rng.uniform(alpha_min, alpha_max)

    def smooth_and_normalize() -> np.ndarray:
        field = gaussian_filter(
            rng.standard_normal(shape),
            sigma=sigma,
            mode="reflect",
        )

        field = field / (np.max(np.abs(field)) + 1e-8)
        field = field * alpha

        return field

    uz = smooth_and_normalize() / scale_z
    uy = smooth_and_normalize()
    ux = smooth_and_normalize()

    return np.stack([uz, uy, ux], axis=0)


def rigid_and_gaussian_warp(
    image: np.ndarray,
    rng: np.random.Generator,
    alpha_min: float,
    alpha_max: float,
    beta_min: float,
    beta_max: float,
    gamma_min: float,
    gamma_max: float,
    translation_min: float,
    translation_max: float,
    sigma_gaussian_min: float,
    sigma_gaussian_max: float,
    alpha_gaussian_min: float,
    alpha_gaussian_max: float,
    scale_z: float,
    order: int = 3,
    mode: str = "reflect",
):
    shape = image.shape

    assert (
        alpha_min < alpha_max
        and beta_min < beta_max
        and gamma_min < gamma_max
        and translation_min < translation_max
        and sigma_gaussian_min < sigma_gaussian_max
        and alpha_gaussian_min < alpha_gaussian_max
    ), "Min values must be inferior to max values, check config file !"

    u_rigid = rigid_displacement_and_warp(
        image=image,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        beta_min=beta_min,
        beta_max=beta_max,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        translation_min=translation_min,
        translation_max=translation_max,
        rng=rng,
        scale_z=scale_z,
        order=order,
        mode=mode,
    )

    u_gaussian = generate_random_gaussian_displacement(
        shape=shape,
        sigma_min=sigma_gaussian_min,
        sigma_max=sigma_gaussian_max,
        alpha_min=alpha_gaussian_min,
        alpha_max=alpha_gaussian_max,
        scale_z=scale_z,
        rng=rng,
    )

    u = u_rigid + u_gaussian

    return u
