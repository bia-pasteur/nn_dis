"""Useful to get the results of the mechanical computations on images stored in a dictionary"""

# pylint: disable=line-too-long
# pylint: disable=trailing-whitespace

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi

from quantities_computation import (
    compute_normals_from_mask_3d,
    compute_traction_3d,
    deformation,
    strain_mask,
    stress_mask,
)
from scripts.image_preprocessing import reshape_img
from scripts.register import register


def get_img_dis_model(
    img_path, folder_path, model_type="aniso", default_spacing=[1.0, 0.13, 0.13]
):
    img, spacing, dis_model, warped_model = register(
        img_path=img_path,
        folder_path=folder_path,
        model_type=model_type,
        default_spacing=default_spacing,
    )
    dis_model = np.transpose(dis_model, (1, 0, 2, 3, 4))
    mask = img[0] > 0.1
    dis_model_masked = np.stack(
        [dis_model[i] * mask for i in range(dis_model.shape[0])]
    )

    return img, spacing, mask, dis_model, warped_model, dis_model_masked


def rmse(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2))


def _compute_mean_std_null_disp(results: dict) -> dict:
    """
    Compute the average RMSE (and its std) of the null displacement field
    (i.e. predicting zero displacement everywhere) and add it to the results dictionary.

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


def _initialize_results_image_gt(
    u_gt: np.ndarray, mask: np.ndarray, mu: float, lambda_: float
) -> tuple[dict, tuple[float, float, float, float, float, np.ndarray], np.ndarray]:
    """
    Initialize the results dictionary with Ground-Truth values

    Args:
        u_gt (np.ndarray): Ground-Truth displacmeent field
        mu (float): First Lamé parameter
        lambda_ (float): Second Lamé parameter

    Returns:
        dict, tuple[float, float, float, float, float, np.ndarray], np.ndarray:

    """
    results_im = {}
    strain_gt = strain_mask(u_gt, [1, 1, 1], mask)
    def_gt = deformation(strain_gt)
    stress_gt = stress_mask(strain_gt, mu, lambda_)

    mask_eroded_gt = ndi.binary_erosion(mask, iterations=3)
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


def _compute_mean_stds_values(results: dict, method_name: str) -> dict:
    """
    Compute the average values and their stds for all mechanical quantities
    for one OF method and add them to the results dictionary.

    Args:
        results (dict): Results dictionary.
        method_name (str): Name of the optical flow method.

    Returns:
        dict: Updated results dictionary.
    """
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
            disp_vals.append(res["rmse_flows"][method_name])
            strain_vals.append(res["rmse_strain"][method_name])
            def_vals.append(res["rmse_def"][method_name])
            stress_vals.append(res["rmse_stress"][method_name])
            trac_vals.append(res["rmse_traction"][method_name])
            runtime_vals.append(res["runtime"][method_name])

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
        "rmse_flows": "disp",
        "rmse_strain": "strain",
        "rmse_def": "def",
        "rmse_stress": "stress",
        "rmse_traction": "traction",
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
    results_folders: list[str | Path] | dict[str, str | Path],
) -> dict[str, Path]:
    """
    Convert the list (or dict) of model folders into a dict {model_name: folder_path}.

    - If a dict is given, the keys are used as model names.
    - If a list is given, the model name is "model_<folder_name>".
      Duplicate names are disambiguated with a numeric suffix.
    """
    if isinstance(results_folders, dict):
        return {name: Path(folder) for name, folder in results_folders.items()}

    if isinstance(results_folders, (str, Path)):
        results_folders = [results_folders]

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
    return models


def compute_of_strain_traction(
    folder_img_ugt: str | Path,
    of_functions: list[str],
    results_folders: list[str | Path] | dict[str, str | Path],
    path_of_results: str | Path,
    noise_stds: np.ndarray | None = None,
) -> dict:
    """
    Get Ground-Truth and compute model-based and optical-flow-based displacement, strain,
    deformation, stress, and traction fields on images.
    Compute average values and their stds for every measured quantity.

    Args:
        folder_img_ugt (str | Path): Folder containing the images (*_img.npy / .tif) and
                                     the GT displacements (*_ugt.npy). If `noise_stds`
                                     is given, the images are noisy versions named
                                     *_noise_XX_img.npy sharing one *_ugt.npy file per
                                     base image.
        of_functions (list[str]): Names of the optical flow methods whose results are stored
                                  in `path_of_results` as `flow_<method>_<filename>`.
        results_folders (list | dict): Folders containing the trained models.
            - list: model names are derived from folder names ("model_<folder_name>").
            - dict: {model_name: folder_path}.
        path_of_results (str | Path): Folder containing the optical flow results and
                                      "average_optical_flow_times.csv".
        noise_stds (np.ndarray | None): Std of the noise of each image, indexed by the
                                        "noise_XX" suffix (XX - 1). If None, the images
                                        are assumed noiseless. Defaults to None.

    Returns:
        dict: Dictionary containing, for each image index, the GT and estimated fields
              for every model and OF method, plus the mean/std RMSE of every quantity.
              When `noise_stds` is given, also the mean/std RMSE of every quantity
              for each noise level ("mean_rmse_<metric>_vs_noise").
    """
    folder_img_ugt = Path(folder_img_ugt)
    path_of_results = Path(path_of_results)
    model_folders = _normalize_model_folders(results_folders)

    if not model_folders:
        raise ValueError("`results_folders` must contain at least one model folder.")

    results = {}
    nb = 0
    of_runtimes = _load_of_runtimes(path_of_results)

    for path_img in sorted(folder_img_ugt.iterdir()):
        filename = path_img.name
        if not filename.endswith(("img.npy", "img.tiff", "img.tif")):
            continue

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
                raise ValueError(
                    f"Cannot find the noise level in filename: {filename}"
                )
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
            img, spacing, mask, h_model, warped_model, h_model_masked = (
                get_img_dis_model(path_img, model_folder)
            )
            time_model = time.time() - start_time

            if results_im is None:
                mask_ref = mask

                ugt_name = _get_ugt_name(filename, noisy=noise_stds is not None)
                ugt_raw = np.load(folder_img_ugt / ugt_name)

                ugt_pixels = np.stack([ugt_raw[i] for i in range(3)])

                sy = 128 / ugt_pixels[1, 0].shape[1]
                sx = 128 / ugt_pixels[2, 0].shape[2]
                ugt_pixels[1] *= sy
                ugt_pixels[2] *= sx

                disp_gt = np.stack(
                    [reshape_img(ugt_pixels[i, 0], 32) for i in range(3)]
                )

                results_im, norms_normals_gt = _initialize_results_image_gt(
                    disp_gt[:, np.newaxis] * mask, mask, mu, lambda_
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
            )

        results[nb] = results_im
        nb += 1

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
    ]:
        results[key] = {}

    results["model_names"] = list(model_folders.keys())
    results["model_folders"] = {k: str(v) for k, v in model_folders.items()}

    for method in [*model_folders.keys(), *of_functions]:
        _compute_mean_stds_values(results, method)

    _compute_mean_std_null_disp(results)

    if noise_stds is not None:
        _compute_rmse_vs_noise(results)

    return results


def save_rmse_csv(results: dict, output_path: str | Path):
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
        if method == "null":
            continue

        row = {"Method": method}

        for metric in metrics:
            mean = results[f"mean_{metric}"][method]
            std = results.get(f"std_{metric}", {}).get(method)

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
    erosion_iterations: int = 3,
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
    img, spacing, h_model, _ = register(
        img_path=img_path,
        folder_path=results_folder,
        model_type=model_type,
        channel=channel,
        default_spacing=default_spacing,
        crop=crop,
    )
    h_model = np.transpose(h_model, (1, 0, 2, 3, 4))

    flows_folder = Path(flows_folder)

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
    mask_eroded = ndi.binary_erosion(
        mask,
        iterations=erosion_iterations,
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

    return results
