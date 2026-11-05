"""Config"""

from dataclasses import dataclass


@dataclass
class ElasticSimuParams:
    """Parameters for the elastic cell simulation"""

    img_paths: list[str]
    masks_paths: list[str]
    t_end: int
    num_time_steps: int
    eta: float
    ym_for_t_nu: float
    t_for_ym_nu: float
    nu_for_ym_t: float
    traction_zone: list[float]
    youngs_modulus: list[float]
    nu: list[float]


@dataclass
class NoiseSimuParams:
    """Parameters for the noisy images"""

    traction_zone: float
    ym: float
    nu: float
    eta: float
    im: str
    noise_stds: list[float]


@dataclass
class GeneralParams:
    """General parameters of the simulations"""

    dataset_dir: str


@dataclass
class ViscoelasticSimuParams:
    """Parameters for the viscoelastic cell simulation"""

    img_paths: list[str]
    masks_paths: list[str]
    t_end: int
    num_time_steps: int
    youngs_modulus: list[float]
    traction_zone: list[float]
    nu: list[float]
    eta: list[float]
    ym_for_t_nu: float
    t_for_ym_nu: float
    nu_for_ym_t: float
    eta_for_ym_t_nu: float
    subparts: dict[str, dict]


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
