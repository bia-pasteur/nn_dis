import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter

from data_generation.fractal_generation.fractals_generation import IFSFunction3D


def random_orthogonal_matrix(rng):
    """
    Build a random 3x3 orthogonal matrix.

    Uses the QR decomposition of a standard normal matrix.

    Args:
        rng (np.random.Generator): Random generator for reproducibility.

    Returns:
        np.ndarray: (3, 3) orthogonal matrix.
    """
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
    """
    Build a random affine matrix with controlled singular values.

    A = U S V^T, where U and V are random orthogonal matrices and S holds
    singular values drawn uniformly from [min_singular_value, max_singular_value].

    Args:
        rng (np.random.Generator):Random generator for reproducibility.
        min_singular_value (float): Lower bound for the singular values.
        max_singular_value (float): Upper bound for the singular values.

    Returns:
        np.ndarray: (3, 3) affine matrix.
    """
    assert 0 < min_singular_value <= max_singular_value < 1, (
        "Singular values must satisfy 0 < min <= max < 1 so the IFS map is "
        "contractive; check config file !"
    )

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
    Add n_functions random affine maps to the IFS, with equal probability.

    Args:
        ifs (IFSFunction3D): IFS object to add the maps to.
        rng (np.random.Generator): Random generator for reproducibility.
        n_functions (int): Number of affine maps to add.
        min_singular_value (float): Lower bound for each map's singular values.
        max_singular_value (float): Upper bound for each map's singular values.
        translation_scale (float): Scale of the uniform translation range.
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
    """
    Generate a 3D fractal image as an intensity volume.

    Builds an IFS of random affine maps see add_random_affine_ifs,
    iterates it to produce a point cloud, then rasterizes the points into a
    (z, y, x) volume.

    Args:
        rng (np.random.Generator): Random generator for reproducibility.
        n_functions (int): Number of affine maps in the IFS; each is applied
            with probability 1/n_functions per iteration.
        min_singular_value (float): Lower bound for each map's singular values.
        max_singular_value (float): Upper bound for each map's singular values.
        translation_scale (float): Scale of the uniform translation range of each map.
        size_img_x (int): Size of the output volume along x.
        size_img_y (int): Size of the output volume along y.
        size_img_z (int): Size of the output volume along z.
        pad_x (int): Margin in voxels left empty on each x edge.
        pad_y (int): Margin in voxels left empty on each y edge.
        pad_z (int): Margin in voxels left empty on each z edge.
        nb_iterations (int): Number of IFS iterations.

    Returns:
        np.ndarray: (size_img_z, size_img_y, size_img_x) fractal image.
    """
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
    Build a 3D rotation matrix from Euler angles.

    R = Rz(alpha) @ Ry(beta) @ Rx(gamma)

    Args:
        alpha (float): Rotation angle around z (radians).
        beta (float): Rotation angle around y (radians).
        gamma (float): Rotation angle around x (radians).

    Returns:
        np.ndarray: (3, 3) rotation matrix.
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
):
    """
    Create a displacement field as a random rigid transform.

    Args:
        image (np.ndarray): (nz, ny, nx) image.
        alpha_min (float): Minimum rotation angle around z (degrees).
        alpha_max (float): Maximum rotation angle around z (degrees).
        beta_min (float): Minimum rotation angle around y (degrees).
        beta_max (float): Maximum rotation angle around y (degrees).
        gamma_min (float): Minimum rotation angle around x (degrees).
        gamma_max (float): Maximum rotation angle around x (degrees).
        translation_min (float): Minimum translation per axis (voxels).
        translation_max (float): Maximum translation per axis (voxels).
        rng (np.random.Generator): Random generator for reproducibility.
        scale_z (float, optional): Physical size of a z voxel relative to an x/y voxel. Defaults to 1.0.

    Returns:
        np.ndarray: (3, nz, ny, nx) displacement field (uz, uy, ux), in voxels.
    """
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
    """
    Create a displacement field from Perlin noise.

    One frequency per axis (z scaled by scale_z), three independent noise
    fields (one per component) with random phases and amplitudes.

    Args:
        shape (list[int]): (nz, ny, nx) shape of the output field.
        amplitude_min (float): Minimum displacement amplitude per component.
        amplitude_max (float): Maximum displacement amplitude per component.
        freq_min (float): Minimum noise frequency.
        freq_max (float): Maximum noise frequency.
        scale_z (float): Physical size of a z voxel relative to an x/y voxel.
        rng (np.random.Generator): Random generator for reproducibility.

    Returns:
        np.ndarray: (3, nz, ny, nx) displacement field (uz, uy, ux), in voxels.
    """
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
    """
    Create a displacement field from Gaussian-filtered noise.

    White noise filtered with a Gaussian kernel of sigma, normalized to a max
    absolute value of alpha; three independent fields, one per component.

    Args:
        shape (list[int]): (nz, ny, nx) shape of the output field.
        sigma_min (float): Minimum Gaussian filter sigma.
        sigma_max (float): Maximum Gaussian filter sigma.
        alpha_min (float): Minimum max displacement magnitude per component.
        alpha_max (float): Maximum max displacement magnitude per component.
        scale_z (float): Physical size of a z voxel relative to an x/y voxel.
        rng (np.random.Generator): Random generator for reproducibility.

    Returns:
        np.ndarray: (3, nz, ny, nx) displacement field (uz, uy, ux), in voxels.
    """
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
    """
    Create a displacement field combining a rigid and a Gaussian transform.

    The two fields are computed independently and summed; see
    rigid_displacement_and_warp and generate_random_gaussian_displacement.

    Args:
        image (np.ndarray): (nz, ny, nx) image.
        rng (np.random.Generator): Random generator for reproducibility.
        alpha_min (float): Rigid minimum rotation angle around z (degrees).
        alpha_max (float): Rigid maximum rotation angle around z (degrees).
        beta_min (float): Rigid minimum rotation angle around y (degrees).
        beta_max (float): Rigid maximum rotation angle around y (degrees).
        gamma_min (float): Rigid minimum rotation angle around x (degrees).
        gamma_max (float): Rigid maximum rotation angle around x (degrees).
        translation_min (float): Rigid minimum translation per axis (voxels).
        translation_max (float): Rigid maximum translation per axis (voxels).
        sigma_gaussian_min (float): Gaussian minimum filter sigma.
        sigma_gaussian_max (float): Gaussian maximum filter sigma.
        alpha_gaussian_min (float): Gaussian minimum max displacement magnitude.
        alpha_gaussian_max (float): Gaussian maximum max displacement magnitude.
        scale_z (float): Physical size of a z voxel relative to an x/y voxel.
        order (int, optional): Unused (kept for API compatibility). Defaults to 3.
        mode (str, optional): Unused (kept for API compatibility). Defaults to "reflect".

    Returns:
        np.ndarray: (3, nz, ny, nx) displacement field (uz, uy, ux), in voxels,
        equal to rigid + Gaussian.
    """
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
