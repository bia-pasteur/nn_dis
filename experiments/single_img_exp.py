import pickle
from pathlib import Path

import numpy as np

from experiments.pipeline_methods import (
    compute_of_strain_traction,
)
from experiments.plot_functions import show_dis_def_traction

FOLDER_SYNTH_IMG = Path("/Users/josephinelahmani/Desktop/data/single_img")
PATH_OF_RESULTS = Path("/Users/josephinelahmani/Desktop/nn_dis/data/computed_flows")
PATHS_TO_MODELS = {
    "train 14": "/Users/josephinelahmani/Desktop/nn_dis/results/train_14"
}
OF_FUNCTIONS = ["fista", "hs"]
SAVE_DIR_PLOTS = Path("/Users/josephinelahmani/Desktop/nn_dis/results/plots")
SAVE_DIR_DICT = Path("/Users/josephinelahmani/Desktop/nn_dis/results/dict")
BEST_MODEL = "train 14"
SAVE_PATH_DICT = SAVE_DIR_DICT / "results_synthetic.pkl"

ARROW_REF_LENGTH = 10.0


def build_comparison_results(results_im: dict) -> dict:
    def squeeze_singleton(field: np.ndarray) -> np.ndarray:
        if field.ndim == 5 and field.shape[1] == 1:
            return field[:, 0]
        if field.ndim == 4 and field.shape[0] == 1:
            return field[0]
        return field

    methods = [name for name in results_im["flows"] if name != "gt"]

    return {
        "flows": [squeeze_singleton(results_im["flows"][m]) for m in ["gt", *methods]],
        "def": [
            squeeze_singleton(results_im["deformation"][m]) for m in ["gt", *methods]
        ],
        "tractions": [
            squeeze_singleton(results_im["traction"][m]) for m in ["gt", *methods]
        ],
    }


def plot_synthetic_image(results_im: dict) -> None:
    """
    Plot the two cell frames, and the displacement, deformation and traction
    of every method together with the Ground-Truth, for one image.

    Args:
        results_im (dict): Results for one image, as returned by
                           `compute_of_strain_traction` with `save_fields=True`.
    """
    z_size = results_im["spacing"][0] / results_im["spacing"][1]
    plot_results = build_comparison_results(results_im)

    arrow_factor_trac = [0.02] * len(plot_results["tractions"])

    arrow_factor_dis = 1.0

    name = f"synthetic_{results_im['filename'].replace('_img.npy', '')}"

    show_dis_def_traction(
        img=results_im["image"],
        results=plot_results,
        name=name,
        arrow_factor_dis=arrow_factor_dis,
        stride_dis=3,
        z_size=z_size,
        arrow_factor_trac=arrow_factor_trac,
        stride_trac=2,
        save_path=SAVE_DIR_PLOTS,
        display=False,
        colorbar=True,
        zoom=1.5,
    )


def main():
    results_synthetic_images = compute_of_strain_traction(
        folder_img_ugt=FOLDER_SYNTH_IMG,
        of_functions=OF_FUNCTIONS,
        results_folders=PATHS_TO_MODELS,
        path_of_results=PATH_OF_RESULTS,
        save_fields=True,
    )

    SAVE_DIR_DICT.mkdir(parents=True, exist_ok=True)
    with open(SAVE_PATH_DICT, "wb") as f:
        pickle.dump(results_synthetic_images, f)

    for nb, results_im in results_synthetic_images.items():
        if isinstance(nb, int):
            plot_synthetic_image(results_im)


if __name__ == "__main__":
    main()
