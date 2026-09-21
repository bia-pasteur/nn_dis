import csv
import os
from pathlib import Path
from typing import Callable, Dict, Tuple

import jsonargparse
import numpy as np

from config import OpticalFlowParams, SaveParams
from optical_flow.algorithms import fista_of, warp
from scripts.register import get_img_spacing


def rmse_(diff):
    return float(np.sqrt(np.mean(diff**2)))


def save_flow_of(
    algo: Callable,
    params: Dict,
    algo_name: str,
    folder_img: Path,
    save_path: Path,
    target_d: int = 32,
    target_h: int = 128,
    target_w: int = 128,
    multiple: int = 16,
    channel: int = None,
    default_spacing: Tuple[float, float, float] = None,
    crop: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] = None,
):

    metrics_rows = []

    for path_img in folder_img.iterdir():
        ext = os.path.splitext(path_img)[1].lower()
        if ext in [".npy", ".tiff", ".tif"]:
            img_padded, spacing = get_img_spacing(
                path_img=path_img,
                target_d=target_d,
                target_h=target_h,
                target_w=target_w,
                multiple=multiple,
                channel=channel,
                default_spacing=[1.0, 0.13, 0.13],  # default_spacing,
                crop=crop,
            )

            mask_padded = img_padded[0] > 0.1

            if algo_name == "fista":
                print("new fista image")
                dis = algo(img_padded, params, global_flow=True, spacing=spacing)
                dis_ = dis[:, 1:]
                dis_final = np.stack(
                    [dis_[i] * mask_padded for i in range(dis_.shape[0])]
                )
                warped_im = warp(img_padded[1], dis_[:, 0], spacing)

            else:
                dis = algo(img_padded, params, global_flow=True)
                dis_ = dis[:, 1:]
                dis_final = np.stack(
                    [dis_[i] * mask_padded * spacing[i] for i in range(dis_.shape[0])]
                )
                warped_im = warp(img_padded[1], dis_[:, 0])

            diff_im = img_padded[1] - img_padded[0]
            diff_warped = warped_im - img_padded[0]

            norm_I2_I1 = np.linalg.norm(diff_im)
            norm_warped_I1 = np.linalg.norm(diff_warped)

            rmse_I2_I1 = rmse_(diff_im)
            rmse_warped_I1 = rmse_(diff_warped)

            metrics_rows.append(
                {
                    "model": algo_name,
                    "image": path_img.name,
                    "norm_I2_minus_I1": float(norm_I2_I1),
                    "rmse_I2_minus_I1": float(rmse_I2_I1),
                    "norm_warped_minus_I1": float(norm_warped_I1),
                    "rmse_warped_minus_I1": float(rmse_warped_I1),
                }
            )

            if save_path is not None:
                np.save(save_path / f"flow_{algo_name}_{path_img.name}", dis_final)
                np.save(save_path / f"warped_{algo_name}_{path_img.name}", warped_im)
                np.save(save_path / f"init_im_{path_img.name}", img_padded)

        mean_row = {
            "model": algo_name,
            "image": "MEAN",
            "norm_I2_minus_I1": float(
                np.mean([row["norm_I2_minus_I1"] for row in metrics_rows])
            ),
            "rmse_I2_minus_I1": float(
                np.mean([row["rmse_I2_minus_I1"] for row in metrics_rows])
            ),
            "norm_warped_minus_I1": float(
                np.mean([row["norm_warped_minus_I1"] for row in metrics_rows])
            ),
            "rmse_warped_I1": float(
                np.mean([row["rmse_warped_minus_I1"] for row in metrics_rows])
            ),
        }

    mean_row["rmse_warped_minus_I1"] = mean_row.pop("rmse_warped_I1")
    metrics_rows.append(mean_row)

    csv_path = save_path / f"metrics_{algo_name}.csv"
    print(csv_path)

    fieldnames = [
        "model",
        "image",
        "norm_I2_minus_I1",
        "rmse_I2_minus_I1",
        "norm_warped_minus_I1",
        "rmse_warped_minus_I1",
    ]

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)

    return metrics_rows


def main(optical_flow: OpticalFlowParams, save_params: SaveParams):
    """
    Main entry for saving optical flow results.
    Saves the results of fista, Horn-Schunck and TV-L1
    for every image in `folder` in the `save_path` folder

    Args:
        of_params (OpticalFlowParams): Parameters for the OF algorithms
        folder (str): Folder containing the images
        save_path (str): Where to save the results
    """

    fista_params = optical_flow.fista
    hs_params = optical_flow.hs
    tvl1_params = optical_flow.tvl1

    folder_img = Path(save_params.folder)
    save_path = Path(save_params.save_path)

    # _ = save_flow_of(hs_of, hs_params, "hs", folder_img, save_path)
    # _ = save_flow_of(tv_l1, tvl1_params, "tvl1", folder_img, save_path)
    _ = save_flow_of(fista_of, fista_params, "fista", folder_img, save_path)


if __name__ == "__main__":
    jsonargparse.auto_cli(main, as_positional=False)
