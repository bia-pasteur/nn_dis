from pathlib import Path

from experiments.pipeline_methods import (
    compute_results_micro_im,
)
from experiments.plot_functions import show_cells, show_dis_def_traction

PATH_IMG_MICRO = Path(
    "/Users/josephinelahmani/Desktop/data/3D_img_sudheer/zstack_time_1_MMStack_Pos0.ome-1.tif"
)
CHANNEL = 2
CROP = [(0, 2), (10, 42), (300, 750), (150, 500)]
PATH_OF_RESULTS = Path("/Users/josephinelahmani/Desktop/nn_dis/data/computed_flows")
PATH_TO_MODEL = "/Users/josephinelahmani/Desktop/nn_dis/results/train_14"
OF_ALGOS = ["fista", "hs"]
FILENAME = "img_sudheer.npy"
E = 700
NU = 0.45
EROSION_ITER = 2
SAVE_PATH = "/Users/josephinelahmani/Desktop/nn_dis/results/plots"

results_img_sudheer = compute_results_micro_im(
    img_path=PATH_IMG_MICRO,
    results_folder=PATH_TO_MODEL,
    flows_folder=PATH_OF_RESULTS,
    channel=CHANNEL,
    default_spacing=None,
    crop=CROP,
    filename=FILENAME,
    algo=OF_ALGOS,
    E=E,
    nu=NU,
    erosion_iterations=EROSION_ITER,
)

show_dis_def_traction(
    img=results_img_sudheer["image"],
    results=results_img_sudheer,
    name="micro",
    arrow_factor_dis=2.0,
    stride_dis=3,
    z_size=results_img_sudheer["spacing"][0] / results_img_sudheer["spacing"][1],
    arrow_factor_trac=[0.015, 0.004, 0.004],
    stride_trac=2,
    save_path=SAVE_PATH,
    display=False,
    colorbar=True,
    zoom=1.5,
)

show_cells(
    img=results_img_sudheer["image"],
    name="cell_pos",
    z_size=results_img_sudheer["spacing"][0] / results_img_sudheer["spacing"][1],
    save_path=SAVE_PATH,
    display=False,
    zoom=1.3,
)
