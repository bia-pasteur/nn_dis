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
    time_method: float,
    mu: float,
    lambda_: float,
    norms_normals_gt: tuple,
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

    rmse_flow = rmse(h_mask, results_im["flows"]["gt"])  # / norm_disp

    strain_of = strain_mask(h_mask, [1, 1, 1], mask)
    rmse_strain = rmse(strain_of, results_im["strain"]["gt"])  # / norm_strain

    def_of = deformation(strain_of)
    rmse_def = rmse(def_of, results_im["deformation"]["gt"])  # / norm_def

    stress_of = stress_mask(strain_of, mu, lambda_)
    rmse_stress = rmse(stress_of, results_im["stress"]["gt"])  # / norm_stress

    traction_of = compute_traction_3d(stress_of[:, :, 0], -normals)
    rmse_traction = rmse(traction_of, results_im["traction"]["gt"])  # / norm_traction

    results_im["flows"][method_name] = h
    results_im["strain"][method_name] = strain_of
    results_im["deformation"][method_name] = def_of
    results_im["stress"][method_name] = stress_of
    results_im["traction"][method_name] = traction_of
    results_im["runtime"][method_name] = time_method

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


def _load_of_runtimes(path_of_results: Path) -> dict[str, float]:
    runtimes_df = pd.read_csv(path_of_results / "average_optical_flow_times.csv")
    return dict(zip(runtimes_df["method"], runtimes_df["average_time_seconds"]))


def compute_of_strain_traction(
    folder_img_ugt: str,
    of_functions: list[str],
    results_folder: str,
    path_of_results: str,
) -> dict:
    """
    Get Ground-Truth and compute optical-flow-based displacement, strain, deformation, stress, and traction fields on images.
    Compute average values and their stds for every measured quantity if more than one image is provided.

    Args:
        images (list[np.ndarray]): Images to analyze.
        displacements (list[np.ndarray]): Ground-Truth displacements on the images.
        mu (float): First Lamé parameter.
        lambda_ (float): Second Lamé parameter.
        of_functions (list[callable]): Optical flow functions to use during the analysis.
        of_params (list[dict]): Parameters for each optical flow function.
        wofv_displacements (list[np.ndarray] | None, optional): wOFV displacement computed using the PIVlab sofware. Defaults to None.
        piv_displacements (list[np.ndarray] | None, optional): PIV displacement computed using the PIVlab sofware. Defaults to None.
        reg_multipliers (list[float] | None, optional): Values to multiply the optical flow parameters.
                                                        Used in the noise experiment. Defaults to None.

    Returns:
        dict: Dictionary containing the Ground-Truth and optical flow based displacement, strain, deformation, stress, and traction fields on images
        as well as the RMSE value for every quantity.
    """
    results = {}
    nb = 0
    of_runtimes = _load_of_runtimes(path_of_results)

    for path_img in sorted(folder_img_ugt.iterdir()):
        filename = path_img.name
        if not filename.endswith(("img.npy", "img.tiff", "img.tif")):
            continue

        match = re.search(r"E_([\d.]+)_nu_([\d.]+)", filename)

        if match:
            E = float(match.group(1))
            nu = float(match.group(2))

        lambda_ = E * nu / ((1.0 + nu) * (1.0 - (2 * nu)))
        mu = E / (2.0 * (1.0 + nu))

        start_time = time.time()
        img, spacing, mask, h_model, warped_model, h_model_masked = get_img_dis_model(
            path_img, results_folder
        )
        time_model = time.time() - start_time

        ugt_name = filename.replace("_img.npy", "_ugt.npy")
        path_ugt = folder_img_ugt / ugt_name

        ugt_raw = np.load(path_ugt)

        ugt_pixels = np.stack(
            [ugt_raw[i] for i in range(3)]
        )  # / spacing[i] for i in range(3)])

        disp_gt = np.stack([reshape_img(ugt_pixels[i, 0], 32) for i in range(3)])

        results_im, norms_normals_gt = _initialize_results_image_gt(
            disp_gt[:, np.newaxis], mask, mu, lambda_
        )

        results_im = _update_results_method(
            results_im=results_im,
            method_name="model",
            h=h_model,
            mask=mask,
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
                mask=mask,
                time_method=of_runtimes[method],
                mu=mu,
                lambda_=lambda_,
                norms_normals_gt=norms_normals_gt,
            )

        results[nb] = results_im

        nb += 1

    results["mean_rmse_disp"] = {}
    results["mean_rmse_strain"] = {}
    results["mean_rmse_def"] = {}
    results["mean_rmse_stress"] = {}
    results["mean_rmse_traction"] = {}
    results["mean_runtime"] = {}
    results["std_rmse_disp"] = {}
    results["std_rmse_strain"] = {}
    results["std_rmse_def"] = {}
    results["std_rmse_stress"] = {}
    results["std_rmse_traction"] = {}
    results["std_runtime"] = {}

    for method in ["model", *of_functions]:
        _compute_mean_stds_values(results, method)

    _compute_mean_std_null_disp(results)

    return results
