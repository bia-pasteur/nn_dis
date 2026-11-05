import pickle
from pathlib import Path

from experiments.src.pipeline_methods import (
    compute_of_strain_traction,
    save_rmse_csv,
)
from experiments.src.plot_functions import save_rmse_vs_train_size

FOLDER_SYNTH_IMG = Path("/Users/josephinelahmani/Desktop/data/dataset_3d")
PATH_OF_RESULTS = Path("/Users/josephinelahmani/Desktop/nn_dis/data/computed_flows")
RESULTS_ROOT = Path("/Users/josephinelahmani/Desktop/nn_dis/results")

PATHS_TO_MODELS = {
    "train_18": [
        RESULTS_ROOT / "train_18",
        RESULTS_ROOT / "train_18_4",
        RESULTS_ROOT / "train_18_5",
    ],
    "train_19": [
        RESULTS_ROOT / "train_19",
        RESULTS_ROOT / "train_19_4",
        RESULTS_ROOT / "train_19_5",
    ],
    "train_22": [
        RESULTS_ROOT / "train_22",
        RESULTS_ROOT / "train_22_4",
        RESULTS_ROOT / "train_22_5",
    ],
    "train_14": [
        RESULTS_ROOT / "train_14_ter",
        RESULTS_ROOT / "train_14_4",
        RESULTS_ROOT / "train_14_5",
    ],
}

TRAIN_SIZES = {
    "train_18": 10,
    "train_19": 100,
    "train_22": 1000,
    "train_14": 630,
}
METRICS = ["disp"]
METHODS = [(name, name) for name in PATHS_TO_MODELS]

SAVE_DIR_DICT = Path("/Users/josephinelahmani/Desktop/nn_dis/results/dict")
SAVE_DIR = "/Users/josephinelahmani/Desktop/nn_dis/results/plots"
RECOMPUTE = True
SAVE_PATH_DICT = SAVE_DIR_DICT / "results_ablation_training.pkl"
OUTPUT_CSV = SAVE_DIR_DICT / "rmse_ablation_training.csv"
PLOTS_DIR = SAVE_DIR


if RECOMPUTE:

    def main():
        results_ablation = compute_of_strain_traction(
            folder_img_ugt=FOLDER_SYNTH_IMG,
            of_functions=[],
            results_folders=PATHS_TO_MODELS,
            path_of_results=PATH_OF_RESULTS,
        )

        SAVE_DIR_DICT.mkdir(parents=True, exist_ok=True)
        with SAVE_PATH_DICT.open("wb") as f:
            pickle.dump(
                results_ablation,
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        save_rmse_csv(
            results=results_ablation,
            output_path=OUTPUT_CSV,
            std_source="seeds",
        )

        save_rmse_vs_train_size(
            results=results_ablation,
            results_dir=SAVE_DIR,
            methods=METHODS,
            train_sizes=TRAIN_SIZES,
            metric=METRICS,
            display=False,
            std_source="seeds",
        )

else:

    def main():
        with SAVE_PATH_DICT.open("rb") as f:
            results_ablation = pickle.load(f)

        save_rmse_csv(
            results=results_ablation,
            output_path=OUTPUT_CSV,
            std_source="seeds",
        )

        save_rmse_vs_train_size(
            results=results_ablation,
            results_dir=SAVE_DIR,
            methods=METHODS,
            train_sizes=TRAIN_SIZES,
            metric=METRICS,
            display=False,
            std_source="seeds",
        )


if __name__ == "__main__":
    main()
