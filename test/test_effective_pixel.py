from functools import partialmethod

import numpy as np
import pytest
from rascal.atlas import Atlas
from rascal.calibrator import Calibrator
from rascal.synthetic import SyntheticSpectrum

# Suppress tqdm output
from tqdm import tqdm

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

effective_pixel = np.arange(987).astype("int")
effective_pixel[len(effective_pixel) // 2 :] = (
    effective_pixel[len(effective_pixel) // 2 :] + 53.37
)

assert len(peaks) > 0


def test_effective_pixel_not_affecting_fit_int_peaks():

    # Set up the calibrator with the pixel values of our
    # wavelengths
    c = Calibrator(peaks=peaks.astype("int"))
    a = Atlas()

    # Arbitrarily we'll set the number of pixels to 768 (i.e.
    # a max range of around 1500 nm
    c.set_calibrator_properties(effective_pixel=effective_pixel)

    # Setup the Hough transform parameters
    c.set_hough_properties(
        range_tolerance=100.0, min_wavelength=100.0, max_wavelength=1300.0
    )

    c.set_ransac_properties(minimum_fit_error=1e-25)

    # Add our fake lines as the atlas
    a.add_user_atlas(elements=["Test"] * len(waves), wavelengths=waves)
    c.set_atlas(a)
    assert len(c.atlas.atlas_lines) > 0

    # And let's try and fit...
    res = c.fit(max_tries=2000, fit_deg=3)

    assert res

    c.plot_fit(display=False)

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

    assert res["peak_utilisation"] > 0.7
    assert res["atlas_utilisation"] > 0.0
    assert rms < 5.0
    assert np.in1d(c.matched_peaks, c.peaks).all()


def test_effective_pixel_not_affecting_fit_perfect_peaks():

    # Set up the calibrator with the pixel values of our
    # wavelengths
    c = Calibrator(peaks=peaks)
    a = Atlas()

    # Arbitrarily we'll set the number of pixels to 768 (i.e.
    # a max range of around 1500 nm
    c.set_calibrator_properties(effective_pixel=effective_pixel)

    # Setup the Hough transform parameters
    c.set_hough_properties(
        range_tolerance=100.0, min_wavelength=100.0, max_wavelength=1300.0
    )

    c.set_ransac_properties(minimum_fit_error=1e-25)

    # Add our fake lines as the atlas
    a.add_user_atlas(elements=["Test"] * len(waves), wavelengths=waves)
    c.set_atlas(a)
    assert len(c.atlas.atlas_lines) > 0

    # And let's try and fit...
    res = c.fit(max_tries=2000, fit_deg=3)

    assert res

    c.plot_fit(display=False)

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

    assert res["peak_utilisation"] > 0.7
    assert res["atlas_utilisation"] > 0.0
    assert rms < 5.0
    assert np.in1d(c.matched_peaks, c.peaks).all()
