"""Useful to get the results of the mechanical computations on images stored in a dictionary"""

# pylint: disable=line-too-long
# pylint: disable=trailing-whitespace

import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import torch

from experiments.src.quantities_computation import (
    compute_normals_from_mask_3d,
    compute_traction_3d,
    deformation,
    strain_mask,
    stress_mask,
)
from scripts.image_preprocessing import get_img_spacing, reshape_img
from scripts.register import register

_RMSE_METRIC_KEYS = {
    "disp": "rmse_flows",
    "strain": "rmse_strain",
    "def": "rmse_def",
    "stress": "rmse_stress",
    "traction": "rmse_traction",
}


def _is_isotropic_model(folder_path: str | Path) -> bool:
    """
    Ugly function to state if the model is isotropic by its name
    (train 16)

    Args:
        folder_path (str | Path): Path to the folder containing the model results

    Returns:
        bool: True if the model is isotropic, else False
    """
    name = Path(folder_path).name
    match = re.fullmatch(r"(.+?)_(?:\d+|bis|ter)", name)
    if match and re.search(r"\d$", match.group(1)):
        name = match.group(1)
    return name.endswith("16")


def get_img_dis_model(
    img_path: Path,
    folder_path: Path,
    default_spacing: list[float] = [1.0, 0.13, 0.13],
):
    """
    Get the image, spacing and displacement inferred by the model.

    Args:
        img_path (Path): Path to the image
        folder_path (Path): Path to the folder containing the model results
        default_spacing (list[float], optional): Default voxel size. Defaults to [1.0, 0.13, 0.13].

    Returns:
        _type_: _description_
    """
    if _is_isotropic_model(folder_path):
        model_type = "iso"
        features = [16, 32, 32]
    else:
        model_type = "aniso"
        features = [16, 32, 32, 64, 64]
    img, spacing, dis_model, warped_model, _ = register(
        img_path=img_path,
        folder_path=folder_path,
        model_type=model_type,
        default_spacing=default_spacing,
        features=features,
    )
    dis_model = np.transpose(dis_model, (1, 0, 2, 3, 4))
    mask = img[0] > 0.1
    dis_model_masked = np.stack(
        [dis_model[i] * mask for i in range(dis_model.shape[0])]
    )

    return img, spacing, mask, dis_model, warped_model, dis_model_masked


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute the RMSE of a prediction vs a target

    Args:
        pred (np.ndarray): Prediction
        target (np.ndarray): Target

    Returns:
        float: RMSE
    """
    return np.sqrt(np.mean((pred - target) ** 2))


def _compute_mean_std_null_disp(results: dict) -> dict:
    """
    Compute the average RMSE (and its std) of the null displacement field
    (i.e. predicting zero displacement everywhere) and add it to the results dictionary.
    Useful for verifications.

    Args:
        results (dict): Results dictionary.

    Returns:
        dict: Updated results dictionary.
    """
    null_vals = [
        res["rmse_flows"]["null"]
        for nb, res in results.items()
        if isinstance(nb, int) and res and "null" in res["rmse_flows"]
    ]

    results["mean_rmse_disp"]["null"] = np.mean(null_vals)
    results["std_rmse_disp"]["null"] = np.std(null_vals)

    return results


def erode_anisotropic(
    mask: np.ndarray,
    spacing: Sequence[float],
    depth: float,
) -> np.ndarray:
    """
    Erode a mask by a constant physical depth, accounting for anisotropic voxels.
    Avoids going too deep in the much smaller Z-axis.
    Erosion performed using an ellipsoidal structure with radii
    depth / spacing[i] voxels along axis i.

    Args:
        mask (np.ndarray): Mask of the cell.
        spacing (Sequence[float]): Physical size of a voxel (Z, Y, X).
        depth (float): Physical depth of the erosion, in the same units as
                       spacing.

    Returns:
        np.ndarray: Boolean mask eroded by depth.
    """
    radii = np.maximum(1.0, np.asarray(depth) / np.asarray(spacing))

    ranges = [np.arange(-int(np.ceil(r)), int(np.ceil(r)) + 1) for r in radii]
    grids = np.meshgrid(*ranges, indexing="ij")
    structure = sum((g / r) ** 2 for g, r in zip(grids, radii)) <= 1.0

    return ndi.binary_erosion(mask, structure=structure)


def _initialize_results_image_gt(
    u_gt: np.ndarray,
    mask: np.ndarray,
    mu: float,
    lambda_: float,
    spacing: Sequence[float],
) -> tuple[dict, tuple[float, float, float, float, float, np.ndarray], np.ndarray]:
    """
    Initialize the results dictionary with Ground-Truth values

    Args:
        u_gt (np.ndarray): Ground-Truth displacmeent field
        mask (np.ndarray): Mask of the cell
        mu (float): First Lamé parameter
        lambda_ (float): Second Lamé parameter
        spacing (Sequence[float]): Physical size of a voxel (Z, Y, X) of the
                                   grid the fields are defined on.

    Returns:
        dict, tuple[float, float, float, float, float, np.ndarray], np.ndarray:

    """
    results_im = {}
    strain_gt = strain_mask(u_gt, [1, 1, 1], mask)
    def_gt = deformation(strain_gt)
    stress_gt = stress_mask(strain_gt, mu, lambda_)

    mask_eroded_gt = erode_anisotropic(mask, spacing, 3 * min(spacing))
    inner_boundary_gt = mask_eroded_gt & (~ndi.binary_erosion(mask_eroded_gt))
    normals_gt = compute_normals_from_mask_3d(mask_eroded_gt)
    normals_gt[:, ~inner_boundary_gt] = 0
    traction_gt = compute_traction_3d(stress_gt[:, :, 0], -normals_gt)

    norm_disp = np.sqrt(np.mean(u_gt**2))
    norm_strain = np.sqrt(np.mean(strain_gt**2))
    norm_def = np.sqrt(np.mean(def_gt**2))
    norm_stress = np.sqrt(np.mean(stress_gt**2))
    norm_traction = np.sqrt(np.mean(traction_gt**2))

    results_im["flows"] = {"gt": u_gt}
    results_im["strain"] = {"gt": strain_gt}
    results_im["deformation"] = {"gt": def_gt}
    results_im["stress"] = {"gt": stress_gt}
    results_im["traction"] = {"gt": traction_gt}
    results_im["rmse_flows"] = {"null": norm_disp}
    results_im["rmse_strain"] = {}
    results_im["rmse_def"] = {}
    results_im["rmse_stress"] = {}
    results_im["rmse_traction"] = {}
    results_im["runtime"] = {}

    return (
        results_im,
        (
            norm_disp,
            norm_strain,
            norm_def,
            norm_stress,
            norm_traction,
            normals_gt,
        ),
    )


def _update_results_method(
    results_im: dict,
    method_name: str,
    h: np.ndarray,
    mask: np.ndarray,
    mu: float,
    lambda_: float,
    norms_normals_gt: tuple,
    time_method: float | None = None,
    save_fields: bool = False,
) -> dict:
    """
    Update the results dictionary for one image with values related to one optical flow method.

    Args:
        results_im (dict): Results dictionary.
        method_name (str): Name fo the optical flow method.
        h (np.ndarray): Displacement field obtained using the optical flow method of interest.
        mask (np.ndarray): Mask of the cell in the image of interest.
        time_method (float): Run time of the algorithm.
        mu (float): First Lamé parameter.
        lambda_ (float): Second Lamé parameter.
        norms_normals_gt (tuple): Norms of the Ground-Truth mechanical values for RMSE computation
                                  and normals of the cell for traction computation.

    Returns:
        dict: Updated results dictionary.
    """
    norm_disp, norm_strain, norm_def, norm_stress, norm_traction, normals = (
        norms_normals_gt
    )

    h_mask = h * mask

    rmse_flow = rmse(h_mask, results_im["flows"]["gt"]) / norm_disp

    strain_of = strain_mask(h_mask, [1, 1, 1], mask)
    rmse_strain = rmse(strain_of, results_im["strain"]["gt"]) / norm_strain

    def_of = deformation(strain_of)
    rmse_def = rmse(def_of, results_im["deformation"]["gt"]) / norm_def

    stress_of = stress_mask(strain_of, mu, lambda_)
    rmse_stress = rmse(stress_of, results_im["stress"]["gt"]) / norm_stress

    traction_of = compute_traction_3d(stress_of[:, :, 0], -normals)
    rmse_traction = rmse(traction_of, results_im["traction"]["gt"]) / norm_traction

    if save_fields:
        results_im["flows"][method_name] = h
        results_im["strain"][method_name] = strain_of
        results_im["deformation"][method_name] = def_of
        results_im["stress"][method_name] = stress_of
        results_im["traction"][method_name] = traction_of

    results_im["runtime"][method_name] = time_method if time_method is not None else 0

    results_im["rmse_flows"][method_name] = rmse_flow
    results_im["rmse_strain"][method_name] = rmse_strain
    results_im["rmse_def"][method_name] = rmse_def
    results_im["rmse_stress"][method_name] = rmse_stress
    results_im["rmse_traction"][method_name] = rmse_traction

    return results_im


def _compute_mean_stds_values(
    results: dict,
    method_name: str,
    group_members: list[str] | None = None,
) -> dict:
    """
    Compute the average values and their stds for all mechanical quantities
    for one OF method and add them to the results dictionary.

    If group_members is given, the values of every member method (e.g. the
    trainings of the same config with different seeds) are pooled: the means and
    stds are computed over all the images of all the members. The std between the
    per-member means is also added under the "std_seeds_*" keys.

    Args:
        results (dict): Results dictionary.
        method_name (str): Name of the OF method (or of the seed group).
        group_members (list[str] | None): Names of the methods to pool into
                                          method_name. Defaults to None.

    Returns:
        dict: Updated results dictionary.
    """
    member_names = group_members if group_members is not None else [method_name]
    disp_vals, strain_vals, def_vals, stress_vals, trac_vals, runtime_vals = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for nb, res in results.items():
        if not isinstance(nb, int):
            continue
        if res:
            for member in member_names:
                disp_vals.append(res["rmse_flows"][member])
                strain_vals.append(res["rmse_strain"][member])
                def_vals.append(res["rmse_def"][member])
                stress_vals.append(res["rmse_stress"][member])
                trac_vals.append(res["rmse_traction"][member])
                runtime_vals.append(res["runtime"][member])

    results["mean_rmse_disp"][method_name] = np.mean(disp_vals)
    results["mean_rmse_strain"][method_name] = np.mean(strain_vals)
    results["mean_rmse_def"][method_name] = np.mean(def_vals)
    results["mean_rmse_stress"][method_name] = np.mean(stress_vals)
    results["mean_rmse_traction"][method_name] = np.mean(trac_vals)
    results["mean_runtime"][method_name] = np.mean(runtime_vals)

    results["std_rmse_disp"][method_name] = np.std(disp_vals)
    results["std_rmse_strain"][method_name] = np.std(strain_vals)
    results["std_rmse_def"][method_name] = np.std(def_vals)
    results["std_rmse_stress"][method_name] = np.std(stress_vals)
    results["std_rmse_traction"][method_name] = np.std(trac_vals)
    results["std_runtime"][method_name] = np.std(runtime_vals)

    if group_members is not None:
        for metric, metric_key in _RMSE_METRIC_KEYS.items():
            seed_means = [
                np.mean(
                    [
                        res[metric_key][member]
                        for nb, res in results.items()
                        if isinstance(nb, int) and res
                    ]
                )
                for member in group_members
            ]
            results.setdefault(f"std_seeds_rmse_{metric}", {})[method_name] = np.std(
                seed_means
            )

    return results


def _merge_seed_groups_per_image(
    results: dict,
    seed_groups: dict[str, list[str]],
) -> dict:
    """
    For every image, average the metric values of the methods of each seed group
    and store the averages under the group name, alongside the per-seed values.

    Args:
        results (dict): Results dictionary, whose per-image entries contain the
                        values of every member of the seed groups.
        seed_groups (dict[str, list[str]]): {group_name: [method_names]}.

    Returns:
        dict: Updated results dictionary.
    """
    for nb, results_im in results.items():
        if not isinstance(nb, int) or not results_im:
            continue
        for group_name, members in seed_groups.items():
            for metric in [
                "rmse_flows",
                "rmse_strain",
                "rmse_def",
                "rmse_stress",
                "rmse_traction",
                "runtime",
            ]:
                vals = [
                    results_im[metric][member]
                    for member in members
                    if member in results_im.get(metric, {})
                ]
                if vals:
                    results_im[metric][group_name] = float(np.mean(vals))

    return results


def _get_ugt_name(filename: str, noisy: bool) -> str:
    """
    Get the name of the Ground-Truth displacement file associated with an image.

    For noisy images, the noise level suffix is removed before adding the ugt
    suffix, since all the noisy versions of an image share the same ugt file.

    Args:
        filename (str): Name of the image file.
        noisy (bool): Whether the image is a noisy version of a base image.

    Returns:
        str: Name of the corresponding ugt file.
    """
    if noisy:
        filename = re.sub(r"_noise_\d+", "", filename)
    return re.sub(r"_img\.(npy|tiff|tif)$", "_ugt.npy", filename)


def _compute_rmse_vs_noise(results: dict) -> dict:
    """
    Compute the mean and std of every RMSE quantity for each noise level,
    across all the images sharing this noise level, and add them to the results
    dictionary under the keys "mean_rmse_<metric>_vs_noise" and
    "std_rmse_<metric>_vs_noise". Also store the noise stds under "noise_stds".

    Args:
        results (dict): Results dictionary, whose per-image entries contain a
                        "noise_index" and a "noise_std".

    Returns:
        dict: Updated results dictionary.
    """
    by_level: dict[int, list[dict]] = {}
    for nb, res in results.items():
        if isinstance(nb, int) and res and "noise_index" in res:
            by_level.setdefault(res["noise_index"], []).append(res)

    if not by_level:
        raise ValueError(
            "No noisy image results found: cannot compute the RMSE vs noise."
        )

    levels = sorted(by_level)
    results["noise_indices"] = levels
    results["noise_stds"] = [by_level[level][0]["noise_std"] for level in levels]

    metric_names = {
        metric_key: metric for metric, metric_key in _RMSE_METRIC_KEYS.items()
    }

    per_image_results = [res for images in by_level.values() for res in images]

    for metric_key, metric_name in metric_names.items():
        if not any(metric_key in res for res in per_image_results):
            continue

        methods = sorted(
            {
                m
                for res in per_image_results
                if metric_key in res
                for m in res[metric_key]
            }
        )

        results[f"mean_rmse_{metric_name}_vs_noise"] = {}
        results[f"std_rmse_{metric_name}_vs_noise"] = {}

        for method in methods:
            means, stds = [], []
            for level in levels:
                vals = [
                    res[metric_key][method]
                    for res in by_level[level]
                    if metric_key in res and method in res[metric_key]
                ]
                means.append(np.mean(vals))
                stds.append(np.std(vals))

            results[f"mean_rmse_{metric_name}_vs_noise"][method] = means
            results[f"std_rmse_{metric_name}_vs_noise"][method] = stds

    return results


def _load_of_runtimes(path_of_results: Path) -> dict[str, float]:
    runtimes_df = pd.read_csv(path_of_results / "average_optical_flow_times.csv")
    return dict(zip(runtimes_df["method"], runtimes_df["average_time_seconds"]))


def _normalize_model_folders(
    results_folders: list[str | Path] | dict[str, str | Path | Sequence[str | Path]],
) -> tuple[dict[str, Path], dict[str, list[str]], list[str]]:
    """
    Convert the list (or dict) of model folders into a dict {method_name: folder_path}
    and a mapping of the trainings that were run several times with different seeds.

    - If a list is given, the model name is model_<folder_name>.
    - If a dict is given, the keys are used as model names.

    Returns:
        tuple[dict[str, Path], dict[str, list[str]], list[str]]:
            - {method_name: folder_path} for every folder (seed folders included).
            - {group_name: [method_names]} for the averaged seed groups.
            - Display names of the methods, in input order (seed group members excluded).
    """
    if isinstance(results_folders, (str, Path)):
        results_folders = [results_folders]

    if isinstance(results_folders, list):
        models = {}
        for folder in results_folders:
            folder = Path(folder)
            base_name = f"model_{folder.name}"
            name = base_name
            k = 1
            while name in models:
                name = f"{base_name}_{k}"
                k += 1
            models[name] = folder
        return models, {}, list(models.keys())

    models = {}
    seed_groups: dict[str, list[str]] = {}

    for name, folders in results_folders.items():
        if isinstance(folders, (str, Path)):
            folders = [folders]
        if not folders:
            raise ValueError(f"`results_folders['{name}']` is empty.")

        existing = []
        for folder in folders:
            folder = Path(folder)
            if folder.is_dir():
                existing.append(folder)
            else:
                print(f"WARNING: model folder not found, skipped in '{name}': {folder}")
        if not existing:
            raise ValueError(f"No existing model folder for '{name}': {folders}")

        if len(existing) == 1:
            models[name] = existing[0]
        else:
            seed_groups[name] = []
            for folder in existing:
                member_name = folder.name
                k = 1
                while member_name in models or member_name in results_folders:
                    member_name = f"{folder.name}_{k}"
                    k += 1
                models[member_name] = folder
                seed_groups[name].append(member_name)

    return models, seed_groups, list(results_folders.keys())


def _process_image(
    path_img: Path,
    model_folders: dict[str, Path],
    of_functions: list[str],
    path_of_results: Path,
    folder_img_ugt: Path,
    noise_stds: np.ndarray | None,
    folder_mask_img: str | Path | None,
    of_runtimes: dict[str, float],
    save_fields: bool = False,
) -> dict:
    """
    Compute all the results for a single image, so images can be processed in parallel.

    Args:
        path_img (Path): Path to the image.
        model_folders (dict[str, Path]): Dictionary containing the different folder
        of_functions (list[str]): _description_
        path_of_results (Path): _description_
        folder_img_ugt (Path): _description_
        noise_stds (np.ndarray | None): _description_
        folder_mask_img (str | Path | None): _description_
        of_runtimes (dict[str, float]): _description_
        save_fields (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        dict: The results dictionary for this image.
    """
    # Several images are processed at once, so give each worker process an
    # equal share of the cores instead of letting torch use all of them.
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    filename = path_img.name

    match = re.search(r"E_([\d.]+)_nu_([\d.]+)", filename)
    if not match:
        raise ValueError(f"Cannot find E and nu in filename: {filename}")

    E = float(match.group(1))
    nu = float(match.group(2))
    lambda_ = E * nu / ((1.0 + nu) * (1.0 - (2 * nu)))
    mu = E / (2.0 * (1.0 + nu))

    noise_index = None
    if noise_stds is not None:
        match_noise = re.search(r"_noise_(\d+)_img\.(npy|tiff|tif)$", filename)
        if not match_noise:
            raise ValueError(f"Cannot find the noise level in filename: {filename}")
        noise_index = int(match_noise.group(1))
        if not 1 <= noise_index <= len(noise_stds):
            raise ValueError(
                f"Noise index {noise_index} out of range for {len(noise_stds)} "
                f"noise stds: {filename}"
            )

    results_im = None
    norms_normals_gt = None
    mask_ref = None

    for model_name, model_folder in model_folders.items():
        start_time = time.time()
        img, spacing, mask, h_model, warped_model, h_model_masked = get_img_dis_model(
            path_img, model_folder
        )
        time_model = time.time() - start_time

        if results_im is None:
            mask_ref = mask

            if noise_stds is not None:
                if folder_mask_img is None:
                    raise ValueError()
                img_clean, _ = get_img_spacing(
                    path_img=Path(folder_mask_img) / re.sub(r"_noise_\d+", "", filename)
                )
                mask_ref = img_clean[0] > 0.1

            ugt_name = _get_ugt_name(filename, noisy=noise_stds is not None)
            ugt_raw = np.load(folder_img_ugt / ugt_name)

            ugt_pixels = np.stack([ugt_raw[i] for i in range(3)])

            sy = 128 / ugt_pixels[1, 0].shape[1]
            sx = 128 / ugt_pixels[2, 0].shape[2]
            ugt_pixels[1] *= sy
            ugt_pixels[2] *= sx

            disp_gt = np.stack([reshape_img(ugt_pixels[i, 0], 32) for i in range(3)])

            results_im, norms_normals_gt = _initialize_results_image_gt(
                disp_gt[:, np.newaxis] * mask, mask, mu, lambda_, spacing
            )

        results_im["image"] = img
        results_im["mask"] = mask
        results_im["spacing"] = spacing
        results_im["filename"] = filename
        results_im["E"] = E
        results_im["nu"] = nu
        if noise_index is not None:
            results_im["noise_index"] = noise_index
            results_im["noise_std"] = float(noise_stds[noise_index - 1])

        results_im = _update_results_method(
            results_im=results_im,
            method_name=model_name,
            h=h_model,
            mask=mask_ref,
            time_method=time_model,
            mu=mu,
            lambda_=lambda_,
            norms_normals_gt=norms_normals_gt,
            save_fields=save_fields,
        )

    for method in of_functions:
        h = np.load(path_of_results / f"flow_{method}_{filename}")

        if method == "fista":
            h = np.stack([h[i] / spacing[i] for i in range(h.shape[0])])

        results_im = _update_results_method(
            results_im=results_im,
            method_name=method,
            h=h,
            mask=mask_ref,
            time_method=of_runtimes[method],
            mu=mu,
            lambda_=lambda_,
            norms_normals_gt=norms_normals_gt,
            save_fields=save_fields,
        )

    return results_im


def compute_of_strain_traction(
    folder_img_ugt: str | Path,
    of_functions: list[str],
    results_folders: list[str | Path] | dict[str, str | Path],
    path_of_results: str | Path,
    noise_stds: np.ndarray | None = None,
    folder_mask_img: str | Path | None = None,
    save_fields: bool = False,
    max_workers: int = 4,
) -> dict:
    """
    Get Ground-Truth and compute model-based and optical-flow-based displacement, strain,
    deformation, stress, and traction fields on images.
    Compute average values and their stds for every measured quantity.

    Args:
        folder_img_ugt (str | Path): Folder containing the images and the GT displacements.
                                     If noise_stds is given, the images are noisy versions named
                                     sharing one GT file perbase image.
        of_functions (list[str]): Names of the optical flow methods whose results we want
                                  to add in the comparison.
        results_folders (list | dict): Folders containing the trained models.
            - list: model names are derived from folder names.
            - dict: {model_name: folder_path}, or {model_name: [folder_paths]} when
              a training was run several times with different seeds: the results of
              the folders are then averaged into one model named model_name
              (each folder is also kept as a separate method in the results).
        path_of_results (str | Path): Folder containing the optical flow results and average run time per method.
        noise_stds (np.ndarray | None): Std of the noise of each image, indexed by the
                                        "noise_XX". If None, the images are assumed noiseless.
                                        Defaults to None.
        folder_mask_img (str | Path | None): Folder containing the clean images, required
                                             when noise_stds is given: the mask is  derived from
                                             the clean image. Defaults to None.
        save_fields (bool, optional): Whether we want to save the displacmeent, strain, stress and
                            traction fields in the final dictionary. Defaults to False.
        max_workers (int, optional): Maximum number of parallel worker processes used to process the
                           images, one image per worker. Defaults to 4.

    Returns:
        dict: Dictionary containing, for each image index, the mean/std RMSE of every mechanical
              quantity in the study.
              If save_fields is True, the GT and computed fields are also saved in the dictionary.
              When noise_stds is given, the mean/std RMSE of every quantity for each noise level is
              added to the dictionary.
    """
    folder_img_ugt = Path(folder_img_ugt)
    path_of_results = Path(path_of_results)
    model_folders, seed_groups, display_names = _normalize_model_folders(
        results_folders
    )

    if not model_folders:
        raise ValueError("`results_folders` must contain at least one model folder.")

    results = {}
    of_runtimes = _load_of_runtimes(path_of_results)

    image_paths = sorted(
        p
        for p in folder_img_ugt.iterdir()
        if p.name.endswith(("img.npy", "img.tiff", "img.tif"))
    )

    worker = partial(
        _process_image,
        model_folders=model_folders,
        of_functions=of_functions,
        path_of_results=path_of_results,
        folder_img_ugt=folder_img_ugt,
        noise_stds=noise_stds,
        folder_mask_img=folder_mask_img,
        of_runtimes=of_runtimes,
        save_fields=save_fields,
    )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for nb, results_im in enumerate(executor.map(worker, image_paths)):
            if nb % 10 == 0:
                print(f"{nb} images processed")
            results[nb] = results_im

    if seed_groups:
        _merge_seed_groups_per_image(results, seed_groups)

    for key in [
        "mean_rmse_disp",
        "mean_rmse_strain",
        "mean_rmse_def",
        "mean_rmse_stress",
        "mean_rmse_traction",
        "mean_runtime",
        "std_rmse_disp",
        "std_rmse_strain",
        "std_rmse_def",
        "std_rmse_stress",
        "std_rmse_traction",
        "std_runtime",
        "std_seeds_rmse_disp",
        "std_seeds_rmse_strain",
        "std_seeds_rmse_def",
        "std_seeds_rmse_stress",
        "std_seeds_rmse_traction",
    ]:
        results[key] = {}

    results["model_names"] = display_names
    results["model_folders"] = {k: str(v) for k, v in model_folders.items()}
    results["seed_groups"] = seed_groups
    results["seed_folders"] = {
        group: [str(model_folders[member]) for member in members]
        for group, members in seed_groups.items()
    }

    member_names = [name for name in model_folders if name not in display_names]
    for method in [*display_names, *of_functions, *member_names]:
        _compute_mean_stds_values(
            results, method, group_members=seed_groups.get(method)
        )

    _compute_mean_std_null_disp(results)

    if noise_stds is not None:
        _compute_rmse_vs_noise(results)

    return results


def save_rmse_csv(
    results: dict,
    output_path: str | Path,
    std_source: str = "dataset",
):
    """
    Save the mean RMSEs (and runtimes) of every method to a CSV file.

    Args:
        results (dict): Results dictionary as returned by
                        `compute_of_strain_traction`.
        output_path (str | Path): Path of the CSV file to write.
        std_source (str): Which standard deviations to report next to the
                          means. "dataset" (default) uses the std over all the
                          images of all the members
                          "seeds" uses the std between the per-seed means of
                          each seed group, falling back to the dataset std for
                          methods without seed stds.
    """
    if std_source not in ("dataset", "seeds"):
        raise ValueError(
            f"Unknown std_source: '{std_source}'. Expected 'dataset' or 'seeds'."
        )

    seed_members = {
        member
        for members in results.get("seed_groups", {}).values()
        for member in members
    }

    metrics = [
        key.removeprefix("mean_")
        for key in results
        if isinstance(key, str)
        and (key.startswith("mean_rmse_") or key == "mean_runtime")
        and not key.endswith("_vs_noise")
    ]

    if not metrics:
        raise ValueError("No RMSE or runtime metrics found in results.")

    methods = results[f"mean_{metrics[0]}"].keys()

    rows = []
    for method in methods:
        if method == "null" or method in seed_members:
            continue

        row = {"Method": method}

        for metric in metrics:
            mean = results[f"mean_{metric}"][method]
            std = results.get(f"std_{metric}", {}).get(method)
            if std_source == "seeds":
                std = results.get(f"std_seeds_{metric}", {}).get(method, std)

            row[metric] = (
                f"{mean:.4g} ± {std:.4g}" if std is not None else f"{mean:.4g}"
            )

        rows.append(row)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )


def compute_results_micro_im(
    img_path: str | Path,
    results_folder: str | Path,
    flows_folder: str | Path,
    filename: str,
    algo: list[str],
    E: float,
    nu: float,
    erosion_iterations: int = 1,
    channel: int | None = None,
    default_spacing: list[float] | None = None,
    crop: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None = None,
    model_type: str = "aniso",
) -> dict:
    """
    Get the displacmeent field and mechanical values
    computed with different methods on a microscopy image

    Args:
        img_path (str | Path): Path to the image.
        results_folder (str | Path): Folders containing the trained deep learning model.
        flows_folder (str | Path): Folders containing the computed optical flows.
        filename (str): Name fo the image file.
        algo (list[str]): List of the optical flow algorithm to fetch the results of.
        E (float): Young's modulus.
        nu (float): Poisson's ratio.
        erosion_iterations (int): Number of erosion iterations to perform for estimating the boundary.
        channel: Channel index to work on. None for images without channels. Defaults to None.
        default_spacing: Default voxel spacing (Z, Y, X) if not in metadata. Defaults to None.
        crop: Optional cropping ranges as ((d_start, d_end), (h_start, h_end), (w_start, w_end)).
              If None, no cropping is applied. Defaults to None.
        model_type (str, optional): Type of the model. Defaults to "aniso".

    Returns:
        dict: A dictionary containing the results associated with the image.
    """
    img, spacing, h_model, _, model_runtime = register(
        img_path=img_path,
        folder_path=results_folder,
        model_type=model_type,
        channel=channel,
        default_spacing=default_spacing,
        crop=crop,
    )
    h_model = np.transpose(h_model, (1, 0, 2, 3, 4))

    flows_folder = Path(flows_folder)

    runtime_file = flows_folder / f"time_methods_{Path(filename).stem}.csv"
    if runtime_file.exists():
        runtimes_df = pd.read_csv(runtime_file)
        of_runtimes = dict(zip(runtimes_df["method"], runtimes_df["time_seconds"]))
    else:
        of_runtimes = _load_of_runtimes(flows_folder)

    flows = []
    model_names = []
    flows.append(h_model)
    model_names.append("Model")

    for algorithm in algo:
        algorithm = algorithm.lower()
        h = np.load(flows_folder / f"flow_{algorithm}_{filename}")

        if algorithm == "fista":
            h = np.stack([h[i] / spacing[i] for i in range(h.shape[0])])

        flows.append(h)
        model_names.append(algorithm.upper())

    # Lamé parameters.
    lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    mask = img[0] > 0.1

    mask_eroded = erode_anisotropic(
        mask,
        spacing,
        erosion_iterations * min(spacing),
    )
    inner_boundary = mask_eroded & ~ndi.binary_erosion(mask_eroded)

    normals = compute_normals_from_mask_3d(mask_eroded)
    normals[:, ~inner_boundary] = 0

    results = {
        "image": img,
        "spacing": spacing,
        "flows": [],
        "def": [],
        "tractions": [],
        "model_names": model_names,
        "runtime": {"Model": model_runtime},
    }

    for h in flows:
        h_mask = h * mask
        strain = strain_mask(h_mask, [1, 1, 1], mask)
        deformation_field = deformation(strain)
        stress = stress_mask(strain, mu, lambda_)
        traction = compute_traction_3d(stress[:, :, 0], -normals)

        results["flows"].append(h[:, 0])
        results["def"].append(deformation_field[0])
        results["tractions"].append(traction)

    for algorithm in algo:
        method_name = algorithm.upper()
        results["runtime"][method_name] = of_runtimes.get(algorithm.lower())

    return results
