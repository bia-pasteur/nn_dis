from pathlib import Path
from types import SimpleNamespace

import napari
import numpy as np
import yaml

from optical_flow.algorithms import fista_of, hs_of, warp
from scripts.register import get_img_spacing, register
from scripts.utils import reshape_img

RESULTS_FOLDER = "/Users/josephinelahmani/helix_mount/nn_dis/results/train_3"
SAVE_PATH = Path("/Users/josephinelahmani/Desktop/nn_dis/data")
BASE_FOLDER = Path("/Users/josephinelahmani/Desktop/nn_dis/data")
NAME_IMG_Z = "E_1000_nu_0.4_nb_zones_1_test_z_img.npy"
NAME_IMG_X = "E_1000_nu_0.4_nb_zones_1_test_x_img.npy"
NAME_IMG_4_ZONES = "E_1000_nu_0.4_nb_zones_4_test_img.npy"


def get_img_ugt_spacing(
    base_folder: Path, name_img: str
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[float, float, float],
]:
    """
    Obtain the reshaped image, the GT displacement field and
    the adapted spacing.

    Args:
        base_folder (Path): Path of the folder containing the image.
        name_img (str): Name of the image in the folder.

    Returns:
        tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
            img (np.ndarray): Image
            u_gt (np.ndarray): GT displacement field
            spacing (tuple[float, float, float]): Spacing (z, y, x)
    """
    ugt_name = name_img.replace("_img.npy", "_ugt.npy")
    path_img = base_folder / name_img

    ugt_raw = np.load(base_folder / ugt_name)

    u_gt = np.stack(
        [
            reshape_img(ugt_raw[0, 0]),
            reshape_img(ugt_raw[1, 0]),
            reshape_img(ugt_raw[2, 0]),
        ]
    )
    img, spacing = get_img_spacing(
        path_img=path_img,
        default_spacing=[1.0, 0.13, 0.13],
    )
    return img, u_gt, spacing


def get_img_save_of(
    base_folder: Path, name_img: str, save_path: Path
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[float, float, float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Obtain the reshaped image, the GT displacement field and
    the adapted spacing. Compute FISTA and HS optical flows
    on the image, mask and save them.

    Args:
        base_folder (Path): Path of the folder containing the image.
        name_img (str): Name of the image in the folder.
        save_path (Path): Path of the folder for saving the displacment fields.

    Returns:
        tuple[ np.ndarray, np.ndarray, tuple[float, float, float], np.ndarray, np.nddaray, np.ndarray, np.ndarray]:
            img (np.ndarray): Image.
            u_gt (np.ndarray): GT displacement field.
            spacing (tuple[float, float, float]): Spacing (z, y, x).
            h_fista (np.ndarray): FISTA displacement field.
            h_fista_masked (np.ndarray): Masked FISTA displacement field.
            h_hs (np.ndarray): HS displacement field.
            h_hs_masked (np.ndarray): Masked HS displacement field.

    """
    img, u_gt, spacing = get_img_ugt_spacing(base_folder, name_img)
    with open(
        "/Users/josephinelahmani/Desktop/nn_dis/configs/optical_flow.yaml", "r"
    ) as f:
        config = yaml.safe_load(f)

    fista_params = SimpleNamespace(**config["optical_flow"]["fista"])
    hs_params = SimpleNamespace(**config["optical_flow"]["hs"])

    h_fista_raw = fista_of(img, fista_params, global_flow=False, spacing=spacing)
    h_fista = np.stack(
        [h_fista_raw[i] / spacing[i] for i in range(h_fista_raw.shape[0])]
    )
    h_hs = hs_of(img, hs_params, global_flow=False)

    mask = img[0] > 0.1  # 0 theshold gives holes in the cell
    h_hs_masked = np.stack([h_hs[i] * mask for i in range(h_hs.shape[0])])
    h_fista_masked = np.stack([h_fista[i] * mask for i in range(h_fista.shape[0])])

    np.save(save_path / f"flow_fista_{name_img}", h_fista)
    np.save(save_path / f"flow_fista_masked_{name_img}", h_fista_masked)
    np.save(save_path / f"flow_hs_{name_img}", h_hs)
    np.save(save_path / f"flow_hs_masked_{name_img}", h_hs_masked)

    return img, u_gt, spacing, h_fista, h_fista_masked, h_hs, h_hs_masked


def get_img_flows(
    base_folder: Path,
    name_img: str,
    save_path: str | None = None,
    results_folder: str | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[float, float, float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    If save_path is not None, go back to get_img_save_of and compute the fields
    with OF algorithms. If results_folder is not None, infer VXM displacement field on the image (masked and not masked).
    Obtain the reshaped image, the GT displacement field and the adapted spacing.
    Load FISTA and HS displacement field (masked and not masked).

    Args:
        base_folder (Path): Path of the folder containing the image.
        name_img (str): Name of the image in the folder.
        save_path (str | None, optional): Path of the folder for saving the displacment fields. Defaults to None.
        results_folder (str | None, optional): _description_. Defaults to None.

    Returns:
        tuple[ np.ndarray, np.ndarray, tuple[float, float, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, ]: _description_
    """
    h_vxm = None
    h_vxm_masked = None
    if save_path is not None:
        img, u_gt, spacing, h_fista, h_fista_masked, h_hs, h_hs_masked = (
            get_img_save_of(base_folder, name_img, save_path)
        )
        if results_folder is not None:
            img, spacing, h_vxm_raw, _ = register(
                img_path=base_folder / name_img,
                folder_path=results_folder,
                model_type="aniso",
                nb_warps=1,
                default_spacing=[1.0, 0.13, 0.13],
            )
    else:
        ugt_name = name_img.replace("_img.npy", "_ugt.npy")
        ugt_raw = np.load(base_folder / ugt_name)
        u_gt = np.stack(
            [
                reshape_img(ugt_raw[0, 0]),
                reshape_img(ugt_raw[1, 0]),
                reshape_img(ugt_raw[2, 0]),
            ]
        )
        if results_folder is not None:
            img, spacing, h_vxm_raw, _ = register(
                img_path=base_folder / name_img,
                folder_path=results_folder,
                model_type="aniso",
                nb_warps=1,
                default_spacing=[1.0, 0.13, 0.13],
            )
            h_vxm = np.transpose(h_vxm_raw, (1, 0, 2, 3, 4))
            mask = img[0] > 0.1
            h_vxm_masked = np.stack([h_vxm[i] * mask for i in range(h_vxm.shape[0])])

        h_fista = np.load(SAVE_PATH / f"flow_fista_{name_img}")
        h_fista_masked = np.load(SAVE_PATH / f"flow_fista_masked_{name_img}")
        h_hs = np.load(SAVE_PATH / f"flow_hs_{name_img}")
        h_hs_masked = np.load(SAVE_PATH / f"flow_hs_masked_{name_img}")

    return (
        img,
        u_gt,
        spacing,
        h_fista,
        h_fista_masked,
        h_hs,
        h_hs_masked,
        h_vxm,
        h_vxm_masked,
    )


def add_flows_viewer(
    flow: np.ndarray,
    flow_name: str,
    ax: int,
    viewer: napari.Viewer,
    spacing: tuple[float, float, float],
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """
    Add a flow visualization on a given ax to a napari viewer.

    Args:
        flow (np.ndarray): The flow.
        flow_name (str): The name of the flow.
        ax (int): The ax index. Can only be 0, 1 or 2 in 3D.
        viewer (napari.Viewer): The existing viewer.
        spacing (tuple[float, float, float]): The spacing or scale of the image (z, y, x).

    Raises:
        ValueError: If ax is higher than the dimension.
    """
    if ax > flow.shape[0]:
        raise ValueError(f"ax can only be 0, 1 or 2 in 3D. Got {ax}")
    img = viewer.add_image(
        flow[ax, 0],
        name=f"{flow_name}, ax={ax}",
        scale=spacing,
        colormap="cividis",
        interpolation3d="nearest",
        contrast_limits=(vmin, vmax) if vmin is not None and vmax is not None else None,
    )
    img.colorbar.visible = True


def get_viewer_flows(
    img: np.ndarray,
    flows: list[np.ndarray],
    flows_name: list[str],
    spacing: tuple[float, float, float],
    axes: list[int],
    vmin: float | None = None,
    vmax: float | None = None,
):
    """
    Create a napari viewer to observe an image and the different displacement
    fields on various axes (x, y and z).

    Args:
        img (np.ndarray): The image.
        flows (list[np.ndarray]): List of the displacement fields to display.
        flows_name (list[str]): Names of the displayed displacement fields.
        spacing (tuple[float, float, float]): Spacing or scale of the image (z, y, x).
        axes (list[int]): Axes to show the displacement fields on.
    """
    viewer = napari.Viewer()
    viewer.add_image(
        img,
        scale=spacing,
        interpolation3d="nearest",
    )

    for i, flow in enumerate(flows):
        for ax in axes:
            if vmin is not None and vmax is not None:
                add_flows_viewer(
                    flow, flows_name[i], ax, viewer, spacing, vmin=vmin, vmax=vmax
                )
            else:
                add_flows_viewer(
                    flow,
                    flows_name[i],
                    ax,
                    viewer,
                    spacing,
                )
    napari.run()


def warped_img(
    img: np.ndarray, h_fista: np.ndarray, h_hs: np.ndarray, h_vxm: np.ndarray
) -> list[np.ndarray]:
    """
    Warp the moving image with all displacement fields, using the warping used in OF methods.

    Args:
        img (np.ndarray): The image.
        h_fista (np.ndarray): FISTA displacement field.
        h_hs (np.ndarray): HS displacement field.
        h_vxm (np.ndarray): VXM displacement field.

    Returns:
        list[np.ndarray]: The tree warped images.
    """
    return (
        warp(img[1], h_fista[:, 0]),
        warp(img[1], h_hs[:, 0]),
        warp(img[1], h_vxm[:, 0]),
    )


def get_viewer_warped_img(
    img: np.ndarray,
    spacing: tuple[float, float, float],
    h_fista: np.ndarray,
    h_hs: np.ndarray,
    h_vxm: np.ndarray,
) -> list[np.ndarray]:
    """
    Get a napari viewer with I1 and the three version of warped_I2 obtained
    with FISTA, HS and VXM displacement fields.

    Args:
        img (np.ndarray): The image.
        spacing (tuple[float, float, float]): The spacing or scale of the image (z, y, x).
        h_fista (np.ndarray): FISTA displacement field.
        h_hs (np.ndarray): HS displacement field.
        h_vxm (np.ndarray): VXM displacement field.
    """
    warped_fista, warped_hs, warped_vxm = warped_img(img, h_fista, h_hs, h_vxm)
    viewer = napari.Viewer()
    for image in [img, warped_fista, warped_hs, warped_vxm]:
        viewer.add_image(
            image,
            scale=spacing,
            interpolation3d="nearest",
        )
    napari.run()


def rmse_(diff):
    return np.sqrt(np.mean(diff**2))


def get_metrics(
    img, u_gt, h_fista, h_hs, h_vxm, h_fista_masked, h_hs_masked, h_vxm_masked
):
    diff_im = img[1] - img[0]
    warped_fista, warped_hs, warped_vxm = warped_img(img, h_fista, h_hs, h_vxm)
    diff_w_f = warped_fista - img[0]
    diff_u_f = h_fista_masked - u_gt[:, np.newaxis]
    diff_w_hs = warped_hs - img[0]
    diff_u_hs = h_hs_masked - u_gt[:, np.newaxis]
    diff_w_vxm = warped_vxm - img[0]
    diff_u_vxm = h_vxm_masked - u_gt[:, np.newaxis]

    print("Norms")
    print("I2 - I1:", np.linalg.norm(diff_im))
    print("warped_fista - I1:", np.linalg.norm(diff_w_f))
    print("h_fista - ugt:", np.linalg.norm(diff_u_f))
    print("warped_hs - I1:", np.linalg.norm(diff_w_hs))
    print("h_hs - ugt:", np.linalg.norm(diff_u_hs))
    print("warped_vxm - I1:", np.linalg.norm(diff_w_vxm))
    print("h_vxm - ugt:", np.linalg.norm(diff_u_vxm))

    print("RMSE")
    print("I2 - I1:", rmse_(diff_im))
    print("warped_fista - I1:", rmse_(diff_w_f))
    print("h_fista - ugt", rmse_(diff_u_f))
    print("warped_hs - I1:", rmse_(diff_w_hs))
    print("h_hs - ugt:", rmse_(diff_u_hs))
    print("warped_vxm - I1:", rmse_(diff_w_vxm))
    print("h_vxm - ugt:", rmse_(diff_u_vxm))


(
    img_x,
    u_gt_x,
    spacing_x,
    h_fista_x,
    h_fista_masked_x,
    h_hs_x,
    h_hs_masked_x,
    h_vxm_x,
    h_vxm_masked_x,
) = get_img_flows(
    BASE_FOLDER,
    NAME_IMG_X,
    results_folder=RESULTS_FOLDER,
    # save_path=SAVE_PATH,
)

get_metrics(
    img_x,
    u_gt_x,
    h_fista_x,
    h_hs_x,
    h_vxm_x,
    h_fista_masked_x,
    h_hs_masked_x,
    h_vxm_masked_x,
)

(
    img_z,
    u_gt_z,
    spacing_z,
    h_fista_z,
    h_fista_masked_z,
    h_hs_z,
    h_hs_masked_z,
    h_vxm_z,
    h_vxm_masked_z,
) = get_img_flows(
    BASE_FOLDER,
    NAME_IMG_Z,
    results_folder=RESULTS_FOLDER,
    # save_path=SAVE_PATH,
)

# get_metrics(
#     img_z,
#     u_gt_z,
#     h_fista_z,
#     h_hs_z,
#     h_vxm_z,
#     h_fista_masked_z,
#     h_hs_masked_z,
#     h_vxm_masked_x,
# )

# (
#     img_4,
#     u_gt_4,
#     spacing_4,
#     h_fista_4,
#     h_fista_masked_4,
#     h_hs_4,
#     h_hs_masked_4,
#     h_vxm_4,
#     h_vxm_masked_4,
# ) = get_img_flows(
#     BASE_FOLDER,
#     NAME_IMG_4_ZONES,
#     results_folder=RESULTS_FOLDER,
# save_path=SAVE_PATH,
# )

flows_x = [
    u_gt_x[:, np.newaxis],
    h_hs_x,
    h_hs_masked_x,
    h_fista_x,
    h_fista_masked_x,
    h_vxm_x,
    h_vxm_masked_x,
]
flows_z = [
    u_gt_z[:, np.newaxis],
    h_hs_z,
    h_hs_masked_z,
    h_fista_z,
    h_fista_masked_z,
    h_vxm_z,
    h_vxm_masked_z,
]
# flows_4 = [
#     u_gt_4[:, np.newaxis],
#     h_hs_4,
#     h_hs_masked_4,
#     h_fista_4,
#     h_fista_masked_4,
#     h_vxm_4,
#     h_vxm_masked_4,
# ]

flows_name = ["GT", "HS", "HS masked", "FISTA", "FISTA masked", "VXM", "VXM masked"]

get_viewer_flows(img_x, flows_x, flows_name, spacing_x, axes=[2])
get_viewer_warped_img(img_x, spacing_x, h_fista_x, h_hs_x, h_vxm_x)

# get_viewer_flows(img_z, flows_z, flows_name, spacing_z, axes=[0])
# get_viewer_warped_img(img_z, spacing_z, h_fista_z, h_hs_z, h_vxm_z)

# get_viewer_flows(img_4, flows_4, flows_name, spacing_4, axes=[0])
# get_viewer_warped_img(img_4, spacing_4, h_fista_4, h_hs_4, h_vxm_4)

# path_img = BASE_FOLDER / NAME_IMG_4_ZONES

# img, spacing = get_img_spacing(
#     path_img=path_img,
#     default_spacing=[1.0, 0.13, 0.13],
# )
# viewer = napari.Viewer()
# viewer.add_image(img, scale=[1.0, 0.13, 0.13])
# napari.run()
