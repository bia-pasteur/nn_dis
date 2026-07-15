import shutil
from pathlib import Path

import jsonargparse
from sklearn.model_selection import train_test_split

from config import DatasetCreation


def main(dataset_creation: DatasetCreation):
    """
    Main entry for training, evaluation and testing datasets creation for synthetic images

    Args:
        dataset_creation (DatasetCreation): Parameters for the datasets
    """

    folder_images = Path(dataset_creation.folder)
    folder = folder_images.parent

    img_files = sorted(folder_images.glob("*_img.npy"))

    for split_name in ["train", "val", "test"]:
        (folder / split_name).mkdir(parents=True, exist_ok=True)

    train_files, temp_files = train_test_split(
        img_files, test_size=0.3, random_state=dataset_creation.seed
    )
    val_files, test_files = train_test_split(
        temp_files, test_size=0.5, random_state=dataset_creation.seed
    )

    def copy_files(files, split_name):
        split_folder = folder / split_name
        for f in files:
            shutil.copy(f, split_folder / f.name)

    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")


if __name__ == "__main__":
    jsonargparse.auto_cli(main, as_positional=False)
