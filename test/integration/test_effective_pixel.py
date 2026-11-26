import copy
from functools import partialmethod

import numpy as np

# Suppress tqdm output
from tqdm import tqdm

from rascal.atlas import Atlas
from rascal.calibrator import Calibrator
from rascal.synthetic import SyntheticSpectrum

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

# Create a test spectrum with a simple linear relationship
# between pixels/wavelengths. The intercept is set to
# 100 nm and the gradient is set to 2.
intercept = 100
gradient = 2.0
best_p = [intercept, gradient]
s = SyntheticSpectrum(coefficients=best_p)

# We add a bunch of wavelengths between 200-1200 nm
peaks, waves = s.get_pixels(np.linspace(200, 1200, num=25))

assert len(peaks) > 0


config = {
    "seed": 0,
    "detector": {
        "contiguous_range": [1.0, 344.0, 345.37, 634.37],
        "num_pix": 634,
        "detector_min_wave": 100.0,
        "detector_max_wave": 1400.0,
    },
    "hough": {
        "num_slopes": 2000,
        "range_tolerance": 50.0,
        "xbins": 100,
        "ybins": 100,
    },
    "ransac": {
        "sample_size": 5,
        "top_n_candidate": 10,
        "filter_close": True,
        "minimum_fit_error": 1e-25,
    },
}


def test_effective_pixel_not_affecting_fit_int_peaks():
    a = Atlas(
        element="Test",
        source="manual",
        wavelengths=waves,
        min_wavelength=100.0,
        max_wavelength=1400.0,
    )

    _config = copy.deepcopy(config)
    _config["ransac"]["max_tries"] = 2000
    _config["ransac"]["degree"] = 3

    c = Calibrator(
        peaks=peaks.astype("int"), atlas_lines=a.atlas_lines, config=_config
    )

    res = c.fit()
    assert res["success"]

    matched = c.match_peaks(best_p, refine=True, robust_refit=True)
    assert np.isin(c.matched_peaks, c.peaks_effective).all()
    assert matched["atlas_utilisation"] > 0.0


def test_effective_pixel_not_affecting_fit_perfect_peaks():
    a = Atlas(
        element="Test",
        source="manual",
        wavelengths=waves,
        min_wavelength=100.0,
        max_wavelength=1400.0,
    )

    _config = copy.deepcopy(config)
    _config["ransac"]["max_tries"] = 2000
    _config["ransac"]["degree"] = 3
    c = Calibrator(peaks=peaks, atlas_lines=a.atlas_lines, config=_config)

    res = c.fit()
    assert res["success"]

    matched = c.match_peaks(best_p, refine=True, robust_refit=True)
    assert matched["atlas_utilisation"] > 0.0
    assert np.isin(c.matched_peaks, c.peaks_effective).all()
