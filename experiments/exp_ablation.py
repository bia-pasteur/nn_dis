import pickle
from pathlib import Path

from experiments.src.pipeline_methods import (
    compute_of_strain_traction,
    save_rmse_csv,
)

FOLDER_SYNTH_IMG = Path("/Users/josephinelahmani/Desktop/data/dataset_3d")
PATH_OF_RESULTS = Path("/Users/josephinelahmani/Desktop/nn_dis/data/computed_flows")
RESULTS_ROOT = Path("/Users/josephinelahmani/Desktop/nn_dis/results")

PATHS_TO_MODELS = {
    "Baseline (train 14)": [
        RESULTS_ROOT / "train_14_ter",
        RESULTS_ROOT / "train_14_4",
        RESULTS_ROOT / "train_14_5",
    ],
    "No reg, alpha=0 (train 5)": [
        RESULTS_ROOT / "train_5",
        RESULTS_ROOT / "train_5_4",
        RESULTS_ROOT / "train_5_5",
    ],
    "Isotropic (train 16)": [
        RESULTS_ROOT / "train_16",
        RESULTS_ROOT / "train_16_4",
        RESULTS_ROOT / "train_16_5",
    ],
    "No supervision (gt_param=0) (train 12)": [
        RESULTS_ROOT / "train_12",
        RESULTS_ROOT / "train_12_4",
        RESULTS_ROOT / "train_12_5",
    ],
    "No data term, (train 23)": [
        RESULTS_ROOT / "train_23",
        RESULTS_ROOT / "train_23_4",
        RESULTS_ROOT / "train_23_5",
    ],
    "No decay, superivison all along (train 7)": [
        RESULTS_ROOT / "train_7",
        RESULTS_ROOT / "train_7_4",
        RESULTS_ROOT / "train_7_5",
    ],
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

        save_rmse_csv(
            results=results_ablation,
            output_path=OUTPUT_CSV,
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


if __name__ == "__main__":
    main()
