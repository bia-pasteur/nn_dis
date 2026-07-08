"""Configuration files"""

from dataclasses import dataclass
from typing import List


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
    alpha_param: float
    beta_param: float
    gpu: int
    features: List[int]
    model_type: str
    flow_multiplier: float
    save_every: int
    train_data: str
    val_data: str
    reduction: str


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
    spacing: List[int] | None = None


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
