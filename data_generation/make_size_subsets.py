"""Create dataset-size ablation subsets (train_size_N) from a source train folder.

Subsets are nested: the size-N subset is the first N files of a seeded
permutation, so train_size_10 ⊂ train_size_100 ⊂ train_size_500 ⊂ train.
Only relative symlinks to the source images are created (no duplication);
the ground-truth folder does not need subsetting because `get_ugt` looks
files up by name.
"""

import jsonargparse
import numpy as np
from pathlib import Path

from config import SubsetParams


def make_size_subsets(subset_params: SubsetParams) -> None:
    """Create nested train_size_N symlink folders under `dataset_root`."""
    root = Path(subset_params.dataset_root)
    source = root / subset_params.source_folder

    paths_img = sorted(path for path in source.iterdir() if path.name.endswith("_img.npy"))
    if not paths_img:
        raise FileNotFoundError(f"No *_img.npy files found in {source}")

    rng = np.random.default_rng(seed=subset_params.seed)
    permutation = rng.permutation(len(paths_img))

    all_data = root / "all_data"
    max_size = max(subset_params.sizes)
    if max_size > len(paths_img):
        raise ValueError(
            f"Requested subset of {max_size} images but only {len(paths_img)} available."
        )

    for size in sorted(subset_params.sizes):
        subset_dir = root / f"train_size_{size}"
        subset_dir.mkdir(exist_ok=True)

        selected = [paths_img[i] for i in permutation[:size]]
        for path_img in selected:
            link = subset_dir / path_img.name
            if link.exists() or link.is_symlink():
                continue
            # Relative link so the folder works identically on the cluster
            # (/pasteur/helix/...) and on the macOS mount (/Volumes/projects/...)
            link.symlink_to(Path("..") / subset_params.source_folder / path_img.name)

        # Verify every linked image resolves and has a ground-truth file
        missing_gt = []
        for path_img in selected:
            if not (subset_dir / path_img.name).exists():
                raise FileNotFoundError(f"Broken symlink for {path_img.name}")
            ugt_name = path_img.name.replace("_img.npy", "_ugt.npy")
            if not (all_data / ugt_name).exists():
                missing_gt.append(ugt_name)
        if missing_gt:
            raise FileNotFoundError(
                f"{len(missing_gt)} images in train_size_{size} have no ground truth "
                f"in {all_data}: {missing_gt[:3]}..."
            )

        print(f"train_size_{size}: {size} images (nested, seed={subset_params.seed})")


if __name__ == "__main__":
    jsonargparse.auto_cli(make_size_subsets, as_positional=False)
