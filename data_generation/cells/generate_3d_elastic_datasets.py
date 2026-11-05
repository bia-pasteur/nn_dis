import os
from pathlib import Path

import jsonargparse
import numpy as np
import scipy.ndimage as ndi
import skimage
import tifffile
from scipy.interpolate import griddata

from data_generation.cells.src.config import ThreeDimsSimuParams
from data_generation.cells.src.imaging.generator import (
    create_image_simu_3d,
)
from data_generation.fractals.utils_fractals import (
    rigid_displacement_and_warp,
)


def add_noise(
    img: np.ndarray,
    std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add Gaussian noise to an image.

    Args:
        img (np.ndarray): Image.
        std (float): Standard deviation of the added noise.
        rng (np.random.Generator): Random generator for reproducibility.

    Returns:
        np.ndarray: Noisy image.
    """
    return np.clip(img + rng.normal(0.0, std, size=img.shape), 0.0, 1.0)


def create_3d_elastic_cell_image(
    img: np.ndarray,
    masks: np.ndarray,
    spacing: tuple[float, float, float],
    slices: list[tuple[slice, ...] | None],
    t_end: int,
    num_time_steps: int,
    range_traction: tuple[float, float],
    youngs_modulus: float,
    nu: float,
    name: str,
    rng: np.random.Generator,
    eta: float | None = 0,
    min_gap_factor: float = 0.0,
    n_zones: int | None = None,
    axes: tuple[int, ...] | None = None,
):
    """
    Creates a 3d image of a cell undergoing elastic deformation.
    The cell is obtained using an image of various cells and the mask associated
    with the cell of interest. It is then meshed and deformed by solving a linear
    elastic problem using the finite elements methods. The images of the cell before and after
    deformation are then recreated by getting the intensities of the pixels back from the original image.

    Args:
        img (np.ndarray): The original image of cells
        masks (np.ndarray): The mask of the cells in the image
        slices (List[Optional[Tuple[slice, ...]]]): A list of tuples of slices representing the bounding box of each
                                                    labeled object in the mask array. Each element corresponds to a
                                                    label (1-indexed), and is None if the label is not present in the mask.
        t_end (int): End time of the simulation
        num_time_steps (int): Numbe rof time steps of the simulation
        range_traction (tuple[float, float]): Range of values for the traction applied in the zone
        youngs_modulus (float): Young's modulus of the cell
        nu (float): Poisson's ratio of the cell
        n_zones (int): Number of random traction zones to be created
        name (str): Name to save the image
        rng (np.random.Generator): Random generator for reproducibility
        eta (Optional, float): Viscosity of the cell

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - u_ground_truth (np.ndarray): Displacement field array of shape (2, 1, H, W),
              representing ground-truth displacements between time steps.
            - img_final (np.ndarray): Normalized image (float array in [0, 1]) of shape (2, H, W)
              representing the cell in its initial condition and the deformed cell.
    """

    img_single_cell = np.pad(
        np.where(masks == 1, img, 0)[slices[0][0], slices[0][1], slices[0][2]],
        ((5, 5), (30, 30), (30, 30)),
        "constant",
    )
    mask = img_single_cell != 0
    mask_opened = np.stack(
        [
            skimage.morphology.isotropic_opening(mask[i], radius=10)
            for i in range(mask.shape[0])
        ],
        axis=0,
    )

    img_opened = img_single_cell * mask_opened

    u_list, warped_image_path, _ = create_image_simu_3d(
        img=img_opened,
        spacing=spacing,
        t_end=t_end,
        num_time_steps=num_time_steps,
        youngs_modulus=youngs_modulus,
        nu=nu,
        eta=eta,
        range_traction=range_traction,
        rng=rng,
        name=name,
        min_gap_factor=min_gap_factor,
        n_zones=n_zones,
        axes=axes,
    )

    u_list = u_list.transpose(4, 0, 1, 2, 3)
    u_gt = u_list[:, 1:]

    u_ground_truth = np.zeros_like(u_gt)
    u_ground_truth[0] = u_gt[2] / spacing[0]
    u_ground_truth[1] = u_gt[1] / spacing[1]
    u_ground_truth[2] = u_gt[0] / spacing[2]

    img = tifffile.imread(warped_image_path)
    img = np.clip(img / (img[0].max() + 1e-8), 0, 1).astype(np.float32)
    shape = img.shape
    displacement = rigid_displacement_and_warp(
        image=img[1],
        alpha_min=-10,
        alpha_max=10,
        beta_min=-10,
        beta_max=10,
        gamma_min=-10,
        gamma_max=10,
        translation_min=-15,
        translation_max=15,
        rng=rng,
        scale_z=shape[1] / shape[0],
    )

    shape = img[1].shape
    grid_z, grid_y, grid_x = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing="ij",
    )

    new_z = grid_z + displacement[0]
    new_y = grid_y + displacement[1]
    new_x = grid_x + displacement[2]

    destination_points = np.column_stack([
        new_z.ravel(),
        new_y.ravel(),
        new_x.ravel(),
    ])

    values = img[1].ravel()

    img_deformed = griddata(
        points=destination_points,
        values=values,
        xi=(grid_z, grid_y, grid_x),
        method="nearest",
        fill_value=0.0,
    )

    img_final = np.stack([img[0], img_deformed], axis=0)

    rng_noise = np.random.default_rng(seed=rng.integers(0, 2**32))
    img_final_noisy = add_noise(img_final, std=0.03, rng=rng_noise)

    u_ground_truth = u_ground_truth + displacement[:, None]

    return u_ground_truth, img_final_noisy


def main(three_dims_simu: ThreeDimsSimuParams):
    """
    Main entry for 3D elastic cell images generation.

    Args:
        three_dims_simu (ThreeDimsSimuParams): Parameters for 3D simulation.
    """
    nu = three_dims_simu.nu_for_ym
    master_rng = np.random.default_rng(42)

    output_dir = Path(three_dims_simu.save_path)
    os.makedirs(output_dir, exist_ok=True)

    for i, img_path in enumerate(three_dims_simu.img_paths):
        masks_path = Path(three_dims_simu.masks_paths[i])

        img = tifffile.imread(img_path)
        masks = tifffile.imread(masks_path)
        slices = ndi.find_objects(masks)

        for ym in three_dims_simu.youngs_modulus:
            for _ in range(three_dims_simu.nb_seeds):
                simulation_seed = int(master_rng.integers(0, 2**63))
                rng = np.random.default_rng(simulation_seed)

                name = f"im_{i:02}_E_{ym}_nu_{nu}_seed_{simulation_seed}"

                print(
                    f"\nRunning simulation {name}, "
                    f"E={ym}, nu={nu}, seed={simulation_seed}"
                )

                u_gt, img_final = create_3d_elastic_cell_image(
                    img=img,
                    masks=masks,
                    spacing=three_dims_simu.spacing,
                    slices=slices,
                    t_end=three_dims_simu.t_end,
                    num_time_steps=three_dims_simu.num_time_steps,
                    range_traction=three_dims_simu.range_traction,
                    youngs_modulus=ym,
                    nu=nu,
                    name=name,
                    rng=rng,
                )

                np.save(output_dir / f"{name}_ugt.npy", u_gt)
                np.save(output_dir / f"{name}_img.npy", img_final)


if __name__ == "__main__":
    jsonargparse.auto_cli(main, as_positional=False)
