import copy
import csv
import dataclasses
import itertools
import time
from pathlib import Path

import jsonargparse
import numpy as np

from config import OpticalFlowParamsList
from experiments.src.pipeline_methods import rmse
from optical_flow.algorithms import fista_of, hs_of, ilk, tv_l1
from scripts.image_preprocessing import get_img_spacing, reshape_img

TARGET_D, TARGET_H, TARGET_W = 32, 128, 128


def save_results_to_csv(results: list[dict], csv_name: str, save_dir: Path) -> Path:
    """Appends the results of one method to `<save_dir>/<csv_name>.csv`"""
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_dir / f"{csv_name}.csv"

    if not results:
        return csv_path

    fieldnames = results[0].keys()
    file_exists = csv_path.exists()

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerows(results)

    return csv_path


def build_combos(of_params) -> list:
    """
    Builds every combination of the list-valued fields of a *ParamsList object.
    Each combination is an instance of the same class, with the list-valued
    fields replaced by one value of the grid.

    Args:
        of_params: A FistaParamsList / HSParamsList / TVL1ParamsList / ILKParamsList instance.

    Returns:
        list: One instance per combination of parameters.
    """
    d = dataclasses.asdict(of_params)
    list_fields = {k: v for k, v in d.items() if isinstance(v, list)}
    fixed_fields = {k: v for k, v in d.items() if not isinstance(v, list)}
    cls = type(of_params)

    return [
        cls(**{**fixed_fields, **dict(zip(list_fields, vals))})
        for vals in itertools.product(*list_fields.values())
    ]


def process_case(
    of_func: callable,
    method_name: str,
    combos: list,
    folder_img_ugt: Path,
    global_flow: bool,
    image_name: str | None,
    default_spacing: tuple[float, float, float],
    save_dir: Path,
) -> list[dict]:
    """
    Computes the flow of every image for every combination of parameters and
    evaluates the masked RMSE against the ground-truth displacement.

    Args:
        of_func: Optical flow function
        method_name (str): Name of the optical flow method.
        combos (list): Parameter combinations to test.
        folder_img_ugt (Path): Folder containing the images and the ground-truth displacements.
        global_flow (bool): Global (0 -> t) or local (t -> t+1) flow.
        image_name (str): If given, only processes the image whose filename contains this string. None processes every image.
        default_spacing (tuple): Default voxel spacing (Z, Y, X) if not in metadata. Defaults to None.
        save_dir (Path): Where to save the CSV results.

    Returns:
        list[dict]: One row per (combination, image), with the RMSE and the runtime.
    """
    results = []

    for path_img in sorted(folder_img_ugt.iterdir()):
        if not path_img.name.endswith("_img.npy"):
            continue

        if image_name is not None and image_name not in path_img.name:
            continue

        ugt_path = path_img.with_name(path_img.name.replace("_img", "_ugt"))
        if not ugt_path.exists():
            raise FileNotFoundError(f"Missing ground-truth file: {ugt_path}")

        img, spacing = get_img_spacing(
            path_img=path_img,
            target_d=TARGET_D,
            target_h=TARGET_H,
            target_w=TARGET_W,
            default_spacing=default_spacing,
        )

        ugt_raw = np.load(ugt_path)

        ugt_raw[1] *= TARGET_H / ugt_raw[1].shape[-2]
        ugt_raw[2] *= TARGET_W / ugt_raw[2].shape[-1]
        disp_gt = np.stack(
            [reshape_img(ugt_raw[i, 0], TARGET_D, TARGET_H, TARGET_W) for i in range(3)]
        )

        mask = img[0] > 0.1
        disp_gt_masked = disp_gt * mask
        norm_disp = np.sqrt(np.mean(disp_gt_masked**2))

        for params in combos:
            start_time = time.time()
            if method_name == "fista":
                h = of_func(img, params, global_flow=global_flow, spacing=spacing)
                h = np.stack([h[i] / spacing[i] for i in range(h.shape[0])])
            else:
                h = of_func(img, params, global_flow=global_flow)
            time_combo = time.time() - start_time

            h = h[:, 1] if global_flow else h[:, 0]

            h_mask = h * mask
            rmse_flow = rmse(disp_gt_masked, h_mask) / norm_disp

            results.append(
                {
                    "params": str(params),
                    "image": path_img.name,
                    "rmse": rmse_flow,
                    "time": time_combo,
                }
            )

    csv_path = save_results_to_csv(results, csv_name=method_name, save_dir=save_dir)
    print(f"Results appended to {csv_path}")

    return results


def main(
    optical_flow_list: OpticalFlowParamsList,
    folder_img_ugt: str,
    of_funcs: list[str],
    save_dir: str = "results/fine_tuning",
    image_name: str | None = None,
    default_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
):
    """
    Main entry for fine-tuning

    Args:
        optical_flow_list (OpticalFlowParamsList): Parameters to test for different methods
        folder_img_ugt (str): Folder containing the images and the ground-truth displacements
        of_funcs (list[str]): List of optical flow functions.
        save_dir (str, optional): Directory where the results should be solved. Defaults to "results/fine_tuning".
        image_name (str | None, optional): If given, only processes the image whose filename contains this string. None processes every image.
        default_spacing (tuple[float, float, float], optional): Default voxel spacing (Z, Y, X) if not in metadata. Defaults to (1.0, 0.13, 0.13).

    Raises:
        ValueError: If an unknown optical flow function is given.
    """
    of_methods = {
        "fista": (fista_of, optical_flow_list.fista_list),
        "hs": (hs_of, optical_flow_list.hs_list),
        "tvl1": (tv_l1, optical_flow_list.tvl1_list),
        "ilk": (ilk, optical_flow_list.ilk_list),
    }

    folder_img_ugt = Path(folder_img_ugt)
    save_dir = Path(save_dir)

    for of_func_name in of_funcs:
        print(of_func_name)
        if of_func_name not in of_methods:
            raise ValueError(f"Unknown optical flow method '{of_func_name}'")

        of_func, of_params = of_methods[of_func_name]
        combos = build_combos(of_params)
        print(f"{len(combos)} combinations to test")

        _ = process_case(
            of_func=of_func,
            method_name=of_func_name,
            combos=combos,
            folder_img_ugt=folder_img_ugt,
            global_flow=optical_flow_list.global_flow,
            image_name=image_name,
            default_spacing=copy.deepcopy(default_spacing),
            save_dir=save_dir,
        )


if __name__ == "__main__":
    jsonargparse.auto_cli(main, as_positional=False)
