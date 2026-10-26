import pickle
from pathlib import Path

import numpy as np

from experiments.pipeline_methods import (
    compute_of_strain_traction,
    save_rmse_csv,
)
from experiments.plot_functions import save_rmse_vs_noise

FOLDER_NOISY_IMG = Path("/Users/josephinelahmani/Desktop/data/dataset_3d_of_noise")
PATH_OF_RESULTS = Path(
    "/Users/josephinelahmani/Desktop/nn_dis/data/computed_flows_noise"
)
PATHS_TO_MODELS = {  #'train 1': '/Users/josephinelahmani/Desktop/nn_dis/results/train_1',
    #'train 2': '/Users/josephinelahmani/Desktop/nn_dis/results/train_2',
    #'train 3': '/Users/josephinelahmani/Desktop/nn_dis/results/train_3',
    # "train 4": "/Users/josephinelahmani/Desktop/nn_dis/results/train_4",
    # 'train 5': '/Users/josephinelahmani/Desktop/nn_dis/results/train_5',
    # 'train 6': '/Users/josephinelahmani/Desktop/nn_dis/results/train_6',
    # 'train 7': '/Users/josephinelahmani/Desktop/nn_dis/results/train_7',
    # 'train 8': '/Users/josephinelahmani/Desktop/nn_dis/results/train_8',
    # 'train 9': '/Users/josephinelahmani/Desktop/nn_dis/results/train_9',
    # 'train 10': '/Users/josephinelahmani/Desktop/nn_dis/results/train_10',
    # 'train 11': '/Users/josephinelahmani/Desktop/nn_dis/results/train_11',
    "train 14": "/Users/josephinelahmani/Desktop/nn_dis/results/train_14",
    # "train 17": "/Users/josephinelahmani/Desktop/nn_dis/results/train_17",
}
OF_FUNCTIONS = ["fista", "hs", "tvl1", "ilk"]
SAVE_DIR = "/Users/josephinelahmani/Desktop/nn_dis/results/plots"
CLEAN_IMG = "/Users/josephinelahmani/Desktop/data/dataset_3d"
STDS_NOISE = np.linspace(0.0, 0.2, 5)
BEST_MODEL = "train 14"
SAVE_DIR_DICT = Path("/Users/josephinelahmani/Desktop/nn_dis/results/dict")
SAVE_PATH_DICT = SAVE_DIR_DICT / "results_noisy.pkl"

RECOMPUTE = False
METRIC = "stress"

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
                ("Proposed", BEST_MODEL),
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
                ("Proposed", BEST_MODEL),
                ("HS", "hs"),
                ("Hessian HS", "fista"),
                ("TV-L1", "tvl1"),
                ("ILK", "ilk"),
            ],
            metric=METRIC,
        )


if __name__ == "__main__":
    main()
