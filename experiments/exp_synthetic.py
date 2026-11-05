import pickle
from pathlib import Path

from experiments.src.pipeline_methods import (
    compute_of_strain_traction,
    save_rmse_csv,
)
from experiments.src.plot_functions import save_scatter_comparison

FOLDER_SYNTH_IMG = Path("/Users/josephinelahmani/Desktop/data/dataset_3d")
PATH_OF_RESULTS = Path("/Users/josephinelahmani/Desktop/nn_dis/data/computed_flows")
PATHS_TO_MODELS = {
    "train 14_ter": "/Users/josephinelahmani/Desktop/nn_dis/results//train_14_ter"
}
OF_FUNCTIONS = ["fista", "hs", "tvl1", "ilk"]
SAVE_DIR_PLOTS = Path("/Users/josephinelahmani/Desktop/nn_dis/results/plots")
SAVE_DIR_DICT = Path("/Users/josephinelahmani/Desktop/nn_dis/results/dict")
SAVE_PATH_DICT = SAVE_DIR_DICT / "results_synthetic.pkl"

SAVE_SCATTER = True
BEST_MODEL = "train 14_ter"
RECOMPUTE = True


if RECOMPUTE:

    def main():
        results_synthetic_images = compute_of_strain_traction(
            folder_img_ugt=FOLDER_SYNTH_IMG,
            of_functions=OF_FUNCTIONS,
            results_folders=PATHS_TO_MODELS,
            path_of_results=PATH_OF_RESULTS,
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
            output_path=SAVE_DIR_DICT / "rmse_synthetic.csv",
        )

        if SAVE_SCATTER:
            save_scatter_comparison(
                results=results_synthetic_images,
                results_dir=SAVE_DIR_PLOTS,
                methods=[
                    ("3DeepFlow", BEST_MODEL),
                    ("HS", "hs"),
                    ("Hessian HS", "fista"),
                    ("TV-L1", "tvl1"),
                    ("ILK", "ilk"),
                ],
                strain=True,
            )

else:

    def main():
        with SAVE_PATH_DICT.open("rb") as f:
            results_synthetic_images = pickle.load(f)

        save_rmse_csv(
            results=results_synthetic_images,
            output_path=SAVE_DIR_DICT / "rmse_synthetic.csv",
        )

        save_scatter_comparison(
            results=results_synthetic_images,
            results_dir=SAVE_DIR_PLOTS,
            methods=[
                ("3DeepFlow", BEST_MODEL),
                ("HS", "hs"),
                ("Hessian HS", "fista"),
                ("TV-L1", "tvl1"),
                ("ILK", "ilk"),
            ],
            strain=True,
        )


if __name__ == "__main__":
    main()
