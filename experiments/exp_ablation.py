import pickle
from pathlib import Path

import pandas as pd

from experiments.pipeline_methods import compute_of_strain_traction

RESULTS_ROOT = Path("/Users/josephinelahmani/Desktop/nn_dis/results")
FOLDER_TEST_IMG = Path("/Users/josephinelahmani/Desktop/data/dataset_3d")
PATH_OF_RESULTS = Path("/Users/josephinelahmani/Desktop/nn_dis/data/computed_flows")
SAVE_DIR_DICT = RESULTS_ROOT / "dict"
SAVE_PATH_DICT = SAVE_DIR_DICT / "results_ablation.pkl"
SAVE_PATH_CSV = SAVE_DIR_DICT / "rmse_ablation.csv"

ABLATION_RUNS = {
    "10": {"folder": RESULTS_ROOT / "train_18", "checkpoint": "final.pt"},
    "100": {"folder": RESULTS_ROOT / "train_19", "checkpoint": "final.pt"},
    "500": {"folder": RESULTS_ROOT / "train_20", "checkpoint": "final.pt"},
    "630 (all)": {"folder": RESULTS_ROOT / "train_17", "checkpoint": "final.pt"},
}

REFERENCE_OF_FUNCTIONS = ["fista"]

METRICS = {
    "disp": "u",
    "strain": "eps",
    "def": "def",
    "stress": "sigma",
    "traction": "T",
}

def _available_runs() -> dict[str, dict]:
    """Keep only the runs whose checkpoint exists on disk."""
    runs = {}
    for label, run in ABLATION_RUNS.items():
        checkpoint_path = run["folder"] / run["checkpoint"]
        if not checkpoint_path.exists():
            print(f"WARNING: {checkpoint_path} not found, skipping '{label}'.")
            continue
        runs[label] = run

    return runs


def _save_table(results: dict, model_folders: dict) -> pd.DataFrame:
    """Build the ablation table (mean ± std NRMSE per quantity per size)."""
    labels = [*model_folders, *REFERENCE_OF_FUNCTIONS]
    rows = []

    null_disp = results.get("mean_rmse_disp", {}).get("null")
    if null_disp is not None:
        # The null displacement only has an NRMSE in u (no strain, etc.)
        rows.append({"size": "null (u=0)", "u": f"{null_disp:.4g}"})

    for label in labels:
        row = {"size": label}
        for metric, col in METRICS.items():
            mean = results.get(f"mean_rmse_{metric}", {})
            std = results.get(f"std_rmse_{metric}", {})
            if label in mean:
                row[col] = f"{mean[label]:.4g} ± {std[label]:.4g}"
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> pd.DataFrame:
    model_folders = _available_runs()
    if not model_folders and not REFERENCE_OF_FUNCTIONS:
        raise RuntimeError("No ablation run available to evaluate.")

    results = compute_of_strain_traction(
        folder_img_ugt=FOLDER_TEST_IMG,
        of_functions=REFERENCE_OF_FUNCTIONS,
        results_folders=model_folders,
        path_of_results=PATH_OF_RESULTS,
        save_fields=False,
    )

    SAVE_DIR_DICT.mkdir(parents=True, exist_ok=True)
    with SAVE_PATH_DICT.open("wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    table = _save_table(results, model_folders)
    table.to_csv(SAVE_PATH_CSV, index=False, encoding="utf-8-sig")

    print("\n=== Dataset-size ablation (NRMSE, mean ± std over the test set) ===")
    print(table.to_string(index=False))
    print(f"\nSaved to {SAVE_PATH_CSV}")

    return table


if __name__ == "__main__":
    main()
