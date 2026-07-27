import os
from pathlib import Path

import jsonargparse
import numpy as np
import tifffile

from config import OpticalFlowParams, SaveParams
from optical_flow.algorithms import fista_of, hs_of, tv_l1
from scripts.utils import reshape_img


def main(optical_flow: OpticalFlowParams, save_params: SaveParams):
    """
    Main entry for saving optical flow results.
    Saves the results of fista, Horn-Schunck and TV-L1
    for every image in `folder` in the `save_path` folder

    Args:
        of_params (OpticalFlowParams): Parameters for the OF algorithms
        folder (str): Folder containing the images
        save_path (str): Where to save the results
    """
    folder = Path(save_params.folder)
    save_path = Path(save_params.save_path)
    for path_img in folder.iterdir():
        ext = os.path.splitext(path_img)[1].lower()
        if ext == ".npy":
            img_raw = np.load(path_img)
        elif ext in [".tiff", ".tif"]:
            img_raw = tifffile.imread(path_img)

        q1, q99 = np.quantile(img_raw[:, 0], [0.01, 0.99])
        img = np.clip((img_raw[:, 0] - q1) / (q99 - q1), 0, 1)

        if img.ndim == 4:
            img_padded = np.stack(
                [reshape_img(img[0], 32), reshape_img(img[1], 32)], axis=0
            )
        else:
            img_padded = np.stack(
                [reshape_img(img[0], 32), reshape_img(img[1], 32)], axis=0
            )
        mask_padded = img_padded[0] > 0.1

        fista_params = optical_flow.fista
        hs_params = optical_flow.hs
        tvl1_params = optical_flow.tvl1

        spacing = fista_params.spacing
        dis_fista = fista_of(img_padded, fista_params, True)
        dis_hs = hs_of(img_padded, hs_params, True)
        dis_tvl1 = tv_l1(img_padded, tvl1_params, True)

        dis_fista_ = dis_fista[:, 1:, :, :, :]
        dis_fista_final = np.stack(
            [dis_fista_[i] * mask_padded for i in range(dis_fista_.shape[0])]
        )

        dis_hs_ = dis_hs[:, 1:, :, :, :]
        dis_hs_final = np.stack(
            [dis_hs_[i] * mask_padded * spacing[i] for i in range(dis_hs_.shape[0])]
        )

        dis_tvl1_ = dis_tvl1[:, 1:, :, :, :]
        dis_tvl1_final = np.stack(
            [dis_tvl1_[i] * mask_padded * spacing[i] for i in range(dis_tvl1_.shape[0])]
        )
        save_path.mkdir(exist_ok=True)
        np.save(save_path / f"flow_fista_beads_{path_img.name}", dis_fista_final)
        np.save(save_path / f"flow_hs_beads_{path_img.name}", dis_hs_final)
        np.save(save_path / f"flow_tvl1_beads_{path_img.name}", dis_tvl1_final)


if __name__ == "__main__":
    jsonargparse.auto_cli(main, as_positional=False)
