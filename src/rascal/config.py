from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

import numpy as np
import numpy.typing as npt
from omegaconf import MISSING

from .atlas import Atlas
from .houghtransform import HoughTransform

"""
output_data = {
    "peaks": self.peaks,
    "spectrum": self.spectrum,
    "num_pix": self.num_pix,
    "effective_pixel": self.effective_pixel,
    "plotting_library": self.plotting_library,
    "seed": self.seed,
    "logger": self.logger,
    "log_level": self.log_level,
    "hide_progress": self.hide_progress,
    "candidate_tolerance": self.candidate_tolerance,
    "constrain_poly": self.constrain_poly,
    "atlas": {
        "config": self.atlas_config,
        "linelist": self.atlas_config["linelist"],
        "vacuum": self.atlas_config["vacuum"],
        "pressure": self.atlas_config["pressure"],
        "temperature": self.atlas_config["temperature"],
        "relative_humidity": self.atlas_config["relative_humidity"],
        "elements": self.atlas_config["elements"],
        "min_atlas_wavelength": self.atlas_config[
            "min_atlas_wavelength"
        ],
        "max_atlas_wavelength": self.atlas_config[
            "max_atlas_wavelength"
        ],
        "min_intensity": self.atlas_config["min_intensity"],
        "min_distance": self.atlas_config["min_distance"],
        "brightest_n_lines": self.atlas_config["brightest_n_lines"],
        "element_list": self.atlas_config["element_list"],
        "wavelength_list": self.atlas_config["wavelength_list"],
        "intensity_list": self.atlas_config["intensity_list"],
    },
    "hough_transform": {
        "num_slopes": self.num_slopes,
        "xbins": self.xbins,
        "ybins": self.ybins,
        "min_wavelength": self.min_wavelength,
        "max_wavelength": self.max_wavelength,
        "range_tolerance": self.range_tolerance,
        "linearity_tolerance": self.linearity_tolerance,
    },
    "ransac": {
        "sample_size": self.sample_size,
        "top_n_candidate": self.top_n_candidate,
        "linear": self.linear,
        "filter_close": self.filter_close,
        "ransac_tolerance": self.ransac_tolerance,
        "candidate_weighted": self.candidate_weighted,
        "hough_weight": self.hough_weight,
        "minimum_matches": self.minimum_matches,
        "minimum_peak_utilisation": self.minimum_peak_utilisation,
        "minimum_fit_error": self.minimum_fit_error,
    },
    "results": {
        "matched_peaks": self.matched_peaks,
        "matched_atlas": self.matched_atlas,
        "fit_coeff": self.fit_coeff,
    },
}

with open(filename, "w+", encoding="ascii") as config_file:
    yaml.dump(output_data, config_file, default_flow_style=False)
"""


@dataclass
class DataConfig:
    filename: str = ""
    num_pix: Optional[int] = None
    contiguous_range: Optional[List[float]] = field(default=None)
    peaks: Optional[List[float]] = field(default=None)
    spectrum: Optional[List[float]] = field(default=None)
    detector_min_wave: float = 3000.0
    detector_max_wave: float = 9000.0
    detector_edge_tolerance: float = 200.0


@dataclass
class HoughConfig:
    num_slopes: int = 2000
    xbins: int = 100
    ybins: int = 100
    range_tolerance: float = 500
    linearity_tolerance: float = 100
    constrain_poly: bool = False
    min_intercept: float = MISSING
    max_intercept: float = MISSING
    min_slope: float = MISSING
    max_slope: float = MISSING


@dataclass
class RansacConfig:
    sample_size: int = 5
    top_n_candidate: int = 5
    max_tries: int = 1000
    linear: bool = True
    use_msac: bool = True
    filter_close: bool = False
    inlier_tolerance: float = 5.0
    candidate_weighted: bool = True
    hough_weight: float = 1.0
    minimum_matches: int = 3
    type: str = "poly"
    degree: int = 4
    progress: bool = False
    rms_tolerance: float = 4.0
    sampler: str = "probabilistic"
    minimum_peak_utilisation: float = 0.0
    minimum_fit_error: float = 1.0e-4


@dataclass
class CalibratorConfig:

    plotting_library: str = "matplotlib"
    seed: Optional[int] = None
    logger_name: str = "rascal"
    log_level: str = "info"
    hide_progress: bool = False

    data: DataConfig = field(default_factory=DataConfig)
    hough: HoughConfig = field(default_factory=HoughConfig)
    ransac: RansacConfig = field(default_factory=RansacConfig)
