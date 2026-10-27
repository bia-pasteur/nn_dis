import pickle
from pathlib import Path

from experiments.pipeline_methods import (
    compute_of_strain_traction,
    save_rmse_csv,
)

FOLDER_SYNTH_IMG = Path("/Users/josephinelahmani/Desktop/data/dataset_3d")
PATH_OF_RESULTS = Path("/Users/josephinelahmani/Desktop/nn_dis/data/computed_flows")
PATHS_TO_MODELS = {
    "Baseline (train 14)": "/Users/josephinelahmani/Desktop/nn_dis/results/train_14",
    "No reg, alpha=0 (train 5)": "/Users/josephinelahmani/Desktop/nn_dis/results/train_5",
    "Isotropic (train 16)": "/Users/josephinelahmani/Desktop/nn_dis/results/train_16",
    "No supervision (gt_param=0) (train 12)": "/Users/josephinelahmani/Desktop/nn_dis/results/train_12",
    "Long decay, 100-500 (train 4)": "/Users/josephinelahmani/Desktop/nn_dis/results/train_4",
    "No decay, superivison all along (train 7)": "/Users/josephinelahmani/Desktop/nn_dis/results/train_7",
}
SAVE_DIR_DICT = Path("/Users/josephinelahmani/Desktop/nn_dis/results/dict")
RECOMPUTE = True
SAVE_PATH_DICT = SAVE_DIR_DICT / "results_ablation.pkl"
OUTPUT_CSV = SAVE_DIR_DICT / "rmse_ablation.csv"

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

else:

    def main():
        with SAVE_PATH_DICT.open("rb") as f:
            results_ablation = pickle.load(f)

        save_rmse_csv(results=results_ablation, output_path=OUTPUT_CSV)


if __name__ == "__main__":
    main()
