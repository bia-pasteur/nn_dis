import pickle
from pathlib import Path

from experiments.pipeline_methods import (
    compute_of_strain_traction,
    save_rmse_csv,
)
from experiments.plot_functions import save_rmse_vs_train_size

FOLDER_SYNTH_IMG = Path("/Users/josephinelahmani/Desktop/data/dataset_3d")
PATH_OF_RESULTS = Path("/Users/josephinelahmani/Desktop/nn_dis/data/computed_flows")
PATHS_TO_MODELS = {
    "train_18": "/Users/josephinelahmani/Desktop/nn_dis/results/train_18",
    "train_19": "/Users/josephinelahmani/Desktop/nn_dis/results/train_19",
    "train_20": "/Users/josephinelahmani/Desktop/nn_dis/results/train_20",
    "train_21": "/Users/josephinelahmani/Desktop/nn_dis/results/train_21",
    "train_14": "/Users/josephinelahmani/Desktop/nn_dis/results/train_14",
}


TRAIN_SIZES = {
    "train_18": 10,
    "train_19": 100,
    "train_20": 300,
    "train_21": 500,
    "train_14": 630,
}
METRICS = ["disp"]
METHODS = [(name, name) for name in PATHS_TO_MODELS]

SAVE_DIR_DICT = Path("/Users/josephinelahmani/Desktop/nn_dis/results/dict")
RECOMPUTE = False
SAVE_PATH_DICT = SAVE_DIR_DICT / "results_ablation_training.pkl"
OUTPUT_CSV = SAVE_DIR_DICT / "rmse_ablation_training.csv"
PLOTS_DIR = SAVE_DIR_DICT


def plot(results_ablation: dict, metrics: list):
    save_rmse_vs_train_size(
        results=results_ablation,
        results_dir=PLOTS_DIR,
        methods=METHODS,
        train_sizes=TRAIN_SIZES,
        metric=metrics,
        display=True,
    )


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

        save_rmse_csv(results=results_ablation, output_path=OUTPUT_CSV)

        save_rmse_vs_train_size(
            results=results_ablation,
            results_dir=PLOTS_DIR,
            methods=METHODS,
            train_sizes=TRAIN_SIZES,
            metric=METRICS,
            display=True,
        )

else:

    def main():
        with SAVE_PATH_DICT.open("rb") as f:
            results_ablation = pickle.load(f)

        save_rmse_csv(results=results_ablation, output_path=OUTPUT_CSV)
        plot(results_ablation, METRICS)


if __name__ == "__main__":
    main()
