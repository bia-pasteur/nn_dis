"""Configuration files"""

from dataclasses import dataclass
from typing import List


@dataclass
class FractalParams:
    """Parameters for fractal image generation"""

    save_dir: str
    seeds: int
    nb_functions: list[int]
    nb_iterations: int
    min_singular_value: float
    max_singular_value: float
    translation_scale: float
    size_img_x: int
    size_img_y: int
    size_img_z: int
    pad_x: int
    pad_y: int
    pad_z: int
    amplitude_min_dis: float | None
    amplitude_max_dis: float | None
    freq_min_dis: float | None
    freq_max_dis: float | None
    sigma_gaussian_min: float | None
    sigma_gaussian_max: float | None
    alpha_gaussian_min: float | None
    alpha_gaussian_max: float | None
    alpha_rigid_min: float | None
    alpha_rigid_max: float | None
    beta_rigid_min: float | None
    beta_rigid_max: float | None
    gamma_rigid_min: float | None
    gamma_rigid_max: float | None
    translation_min: float | None
    translation_max: float | None


@dataclass
class ThreeDimsSimuParams:
    """Parameters for the elastic cell simulation"""

    img_paths: list[str]
    masks_paths: list[str]
    spacing: tuple[float, float, float]
    t_end: int
    num_time_steps: int
    eta: float
    ym_for_nu: float
    nu_for_ym: float
    range_traction: tuple[float, float]
    youngs_modulus: list[float]
    nu: list[float]
    nb_zones_list: list[int]
    nb_seeds: int
    save_path: str


@dataclass
class NoiseParams:
    """Parameters for noisy image generation"""

    folder_images: str
    folder_noisy_images: str
    seed: int
    max_std: float
    nb_images: int
    nb_noisy_images: int


@dataclass
class SubsetParams:
    """Parameters for creating dataset-size ablation subsets"""

    dataset_root: str
    source_folder: str = "train"
    sizes: List[int] | None = None
    seed: int = 42


@dataclass
class DatasetCreation:
    """Parameters for dataset creation"""

    folder: str
    seed: int
    gap_max: int


@dataclass
class TrainConfig:
    """Parameters for training"""

    output_dir: str
    epochs: int
    workers: int
    batch_size: int
    lr: float
    warp_param: float
    alpha_param: float
    beta_param: float
    gt_param: float
    gpu: int
    features: List[int]
    model_type: str
    flow_multiplier: float
    save_every: int
    train_data: str
    val_data: str
    reduction: str
    gt_decay: bool | None = False
    gt_decay_start_epoch: int | None = None
    gt_decay_end_epoch: int | None = None
    seed: int = 42
    channel: int | None = None
    mask: bool = False
    path_gt: str | None = None


@dataclass
class FistaParams:
    """FISTA parameters"""

    num_iter: int
    num_warp: int
    alpha: float
    beta: float
    eps: float
    num_pyramid: int
    pyramid_downscale: float
    pyramid_min_size: int


@dataclass
class HSParams:
    """Horn Schunck parameters"""

    num_iter: int
    num_warp: int
    alpha: float
    eps: float
    num_pyramid: int
    pyramid_downscale: float
    pyramid_min_size: int
    w: float


@dataclass
class FarnebackParams:
    """Farneback parameters"""

    winSize: int
    pyrScale: float
    numLevels: int
    fastPyramids: bool
    numIters: int
    polyN: int
    polySigma: float
    flags: int


@dataclass
class TVL1Params:
    """TV-L1 parameters"""

    attachment: float
    tightness: float
    num_warp: int
    num_iter: int
    tol: float
    prefilter: bool


@dataclass
class ILKParams:
    """ILK parameters"""

    radius: float
    num_warp: int
    gaussian: bool
    prefilter: bool


@dataclass
class OpticalFlowParams:
    """Optical Flow parameters"""

    global_flow: bool
    fista: FistaParams
    hs: HSParams
    farneback: FarnebackParams
    tvl1: TVL1Params
    ilk: ILKParams


@dataclass
class FistaParamsList:
    """FISTA parameters (list-valued fields generate a grid of combinations)"""

    num_iter: list[int]
    num_warp: list[int]
    alpha: list[float]
    beta: list[float]
    eps: float
    num_pyramid: int
    pyramid_downscale: float
    pyramid_min_size: int


@dataclass
class HSParamsList:
    """Horn-Schunck parameters (list-valued fields generate a grid of combinations)"""

    num_iter: list[int]
    num_warp: list[int]
    alpha: list[float]
    eps: float
    num_pyramid: int
    pyramid_downscale: float
    pyramid_min_size: int
    w: float


@dataclass
class TVL1ParamsList:
    """TV-L1 parameters (list-valued fields generate a grid of combinations)"""

    attachment: list[float]
    tightness: list[float]
    num_warp: list[int]
    num_iter: list[int]
    tol: float
    prefilter: bool


@dataclass
class ILKParamsList:
    """ILK parameters (list-valued fields generate a grid of combinations)"""

    radius: list[float]
    num_warp: list[int]
    gaussian: bool
    prefilter: bool


@dataclass
class OpticalFlowParamsList:
    """Optical Flow parameters"""

    global_flow: bool
    fista_list: FistaParamsList
    hs_list: HSParamsList
    tvl1_list: TVL1ParamsList
    ilk_list: ILKParamsList


@dataclass
class SaveParams:
    """Parameters for saving the results"""

    folder: str
    save_path: str
