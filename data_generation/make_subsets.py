import os
import random
from pathlib import Path

import jsonargparse

from config import SubsetParams


def make_subsets(subset_params: SubsetParams) -> None:
    """
    Create a subset of a given dataset, using symlinks for
    memory.

    Args:
        subset_params (SubsetParams): Parameters of the subset to create.
    """
    root = Path(subset_params.dataset_root)
    source = root / subset_params.source_folder
    sizes = sorted(set(subset_params.sizes))

    files = sorted(path for path in source.iterdir() if path.is_file())

    destinations = {size: root / f"{source.name}_{size}" for size in sizes}

    rng = random.Random(subset_params.seed)
    rng.shuffle(files)

    for size, destination in destinations.items():
        destination.mkdir()

        for source_file in files[:size]:
            link = destination / source_file.name

            target = os.path.relpath(source_file, start=destination)
            link.symlink_to(target)


if __name__ == "__main__":
    jsonargparse.auto_cli(make_subsets, as_positional=False)
