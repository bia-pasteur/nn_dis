import pickle
from pathlib import Path

import numpy as np

from experiments.src.pipeline_methods import (
    compute_of_strain_traction,
    save_rmse_csv,
)
from experiments.src.plot_functions import save_rmse_vs_noise

FOLDER_NOISY_IMG = Path("/Users/josephinelahmani/Desktop/data/dataset_3d_of_noise")
PATH_OF_RESULTS = Path(
    "/Users/josephinelahmani/Desktop/nn_dis/data/computed_flows_noise"
)
PATHS_TO_MODELS = {
    "train 14_ter": "/Users/josephinelahmani/Desktop/nn_dis/results/train_14_ter",
}
OF_FUNCTIONS = ["fista", "hs", "tvl1", "ilk"]
SAVE_DIR = "/Users/josephinelahmani/Desktop/nn_dis/results/plots"
CLEAN_IMG = "/Users/josephinelahmani/Desktop/data/dataset_3d"
STDS_NOISE = np.linspace(0.0, 0.2, 5)
BEST_MODEL = "train 14_ter"
SAVE_DIR_DICT = Path("/Users/josephinelahmani/Desktop/nn_dis/results/dict")
SAVE_PATH_DICT = SAVE_DIR_DICT / "results_noisy.pkl"

RECOMPUTE = True
METRIC = ["disp", "strain", "traction"]

if RECOMPUTE:

    def main():
        results_synthetic_images = compute_of_strain_traction(
            folder_img_ugt=FOLDER_NOISY_IMG,
            of_functions=OF_FUNCTIONS,
            results_folders=PATHS_TO_MODELS,
            path_of_results=PATH_OF_RESULTS,
            noise_stds=STDS_NOISE,
            folder_mask_img=Path(CLEAN_IMG),
        )

        SAVE_DIR_DICT.mkdir(parents=True, exist_ok=True)
        with SAVE_PATH_DICT.open("wb") as f:
            pickle.dump(
                results_synthetic_images,
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        save_rmse_csv(
            results=results_synthetic_images,
            output_path=SAVE_DIR_DICT / "rmse_noisy.csv",
        )

        save_rmse_vs_noise(
            results=results_synthetic_images,
            results_dir=Path(SAVE_DIR),
            methods=[
                ("3DeepFlow", BEST_MODEL),
                ("HS", "hs"),
                ("Hessian HS", "fista"),
                ("TV-L1", "tvl1"),
                ("ILK", "ilk"),
            ],
            metric=METRIC,
        )

else:

    def main():
        with SAVE_PATH_DICT.open("rb") as f:
            results_synthetic_images = pickle.load(f)

        save_rmse_csv(
            results=results_synthetic_images,
            output_path=SAVE_DIR_DICT / "rmse_noisy.csv",
        )

        save_rmse_vs_noise(
            results=results_synthetic_images,
            results_dir=Path(SAVE_DIR),
            methods=[
                ("3DeepFlow", BEST_MODEL),
                ("HS", "hs"),
                ("Hessian HS", "fista"),
                ("TV-L1", "tvl1"),
                ("ILK", "ilk"),
            ],
            metric=METRIC,
        )


if __name__ == "__main__":
    main()
