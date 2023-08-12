from functools import partialmethod

import numpy as np
import pytest
from matplotlib.font_manager import X11FontDirectories

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

# We add a bunch of wavelegnths between 200-1200 nm
peaks, waves = s.get_pixels(np.linspace(200, 1200, num=25))

assert len(peaks) > 0


# Effective pixel: arbitrarily we'll set the number of pixels to 768 (i.e.
# a max range of around 1500 nm
config = {
    "detector": {
        "contiguous_range": [1.0, 344.0, 345.37, 634.37],
        "num_pix": 634,
    },
    "hough": {
        "num_slopes": 2000,
        "range_tolerance": 100.0,
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
    # Set up the calibrator with the pixel values of our
    # wavelengths
    a = Atlas(
        element="Test",
        source="manual",
        wavelengths=np.arange(10),
        min_wavelength=0,
        max_wavelength=10,
    )

    _config = config.copy()
    _config["ransac"]["max_tries"] = 2000
    _config["ransac"]["degree"] = 3

    c = Calibrator(
        peaks=peaks.astype("int"), atlas_lines=a.atlas_lines, config=_config
    )

    # And let's try and fit...
    res = c.fit()

    assert res

    (
        res["fit_coeff"],
        x_fit,
        y_fit,
        res["rms_residual"],
        res["residual"],
        res["peak_utilisation"],
        res["atlas_utilisation"],
        res["success"],
    ) = c.match_peaks(best_p, refine=True, robust_refit=True).values()
    fit_diff = c.polyval(x_fit, best_p) - y_fit
    rms = np.sqrt(np.sum(fit_diff**2 / len(x_fit)))

    assert np.in1d(c.matched_peaks, c.peaks).all()
    # assert res["peak_utilisation"] > 0.7
    assert res["atlas_utilisation"] > 0.0
    # assert rms < 5.0


def test_effective_pixel_not_affecting_fit_perfect_peaks():
    # Set up the calibrator with the pixel values of our
    # wavelengths
    a = Atlas(
        element="Test",
        source="manual",
        wavelengths=np.arange(10),
        min_wavelength=0,
        max_wavelength=10,
    )
    # Arbitrarily we'll set the number of pixels to 768 (i.e.
    # a max range of around 1500 nm

    _config = config.copy()
    _config["ransac"]["max_tries"] = 2000
    _config["ransac"]["degree"] = 3
    c = Calibrator(peaks=peaks, atlas_lines=a.atlas_lines, config=config)

    # And let's try and fit...
    res = c.fit()

    assert res

    # c.plot_fit(display=False)

    (
        res["fit_coeff"],
        x_fit,
        y_fit,
        res["rms_residual"],
        res["residual"],
        res["peak_utilisation"],
        res["atlas_utilisation"],
        res["success"],
    ) = c.match_peaks(best_p, refine=True, robust_refit=True).values()

    fit_diff = c.polyval(x_fit, best_p) - y_fit
    rms = np.sqrt(np.sum(fit_diff**2 / len(x_fit)))

    # assert res["peak_utilisation"] > 0.7
    assert res["atlas_utilisation"] > 0.0
    # assert rms < 5.0
    assert np.in1d(c.matched_peaks, c.peaks).all()
