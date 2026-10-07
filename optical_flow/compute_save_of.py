import csv
import time
from pathlib import Path
from typing import Callable, Dict, Tuple

import jsonargparse
import numpy as np

from config import OpticalFlowParams, SaveParams
from optical_flow.algorithms import fista_of, hs_of, ilk, tv_l1
from scripts.image_preprocessing import get_img_spacing


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
    default_spacing: Tuple[float, float, float] = (1.0, 0.13, 0.13),
    crop: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] = None,
):
    times_method = []

    for path_img in folder_img.iterdir():
        filename = path_img.name.lower()

        if not filename.endswith(("img.npy", "img.tiff", "img.tif")):
            continue

        img_padded, spacing = get_img_spacing(
            path_img=path_img,
            target_d=target_d,
            target_h=target_h,
            target_w=target_w,
            multiple=multiple,
            channel=channel,
            default_spacing=default_spacing,
            crop=crop,
        )

        if algo_name == "fista":
            start_time = time.time()
            h = algo(img_padded, params, global_flow=True, spacing=spacing)
            time_img = time.time() - start_time

        else:
            start_time = time.time()
            h = algo(img_padded, params, global_flow=True)
            time_img = time.time() - start_time

        times_method.append(time_img)

        if save_path is not None:
            np.save(save_path / f"flow_{algo_name}_{path_img.name}", h[:, 1:])

    return float(np.mean(times_method))


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
    ilk_params = optical_flow.ilk

    folder_img = Path(save_params.folder)
    save_path = Path(save_params.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    average_times = {
        "hs": save_flow_of(hs_of, hs_params, "hs", folder_img, save_path),
        "tvl1": save_flow_of(tv_l1, tvl1_params, "tvl1", folder_img, save_path),
        "fista": save_flow_of(fista_of, fista_params, "fista", folder_img, save_path),
        "ilk": save_flow_of(ilk, ilk_params, "ilk", folder_img, save_path),
    }

    output_file = save_path / "average_optical_flow_times.csv"

    with output_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "average_time_seconds"])

        for method, average_time in average_times.items():
            writer.writerow([method, average_time])


if __name__ == "__main__":
    jsonargparse.auto_cli(main, as_positional=False)
