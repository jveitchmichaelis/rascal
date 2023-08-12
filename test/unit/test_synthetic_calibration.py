import logging
from functools import partialmethod

import numpy as np

# Suppress tqdm output
from tqdm import tqdm

from rascal.atlas import Atlas
from rascal.calibrator import Calibrator
from rascal.synthetic import SyntheticSpectrum

logger = logging.getLogger(__name__)

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

# Create a test spectrum with a simple linear relationship
# between pixels/wavelengths. The intercept is set to
# 100 nm and the gradient is set to 2.
intercept = 3000  # Wavelength at pixel 0
gradient = 10  # A/px
min_wavelength = 3500
max_wavelength = 9000
best_p = [intercept, gradient]
s = SyntheticSpectrum(
    coefficients=best_p,
    min_wavelength=min_wavelength,
    max_wavelength=max_wavelength,
)

# We add a bunch of wavelegnths between 200-1200 nm
peaks, waves = s.get_pixels(
    np.linspace(min_wavelength + 100, max_wavelength - 100, num=25)
)
max_pix, _ = s.get_pixels([max_wavelength])


def test_default():

    atlas = Atlas(
        source="manual",
        wavelengths=waves,
        min_wavelength=min_wavelength,
        max_wavelength=max_wavelength,
        range_tolerance=100.0,
        element="Test",
    )
    assert len(atlas.atlas_lines) == len(waves)

    config = {
        "detector": {
            "contiguous_range": None,
            "num_pix": int(max_pix),
            "detector_min_wave": 3000.0,
            "detector_max_wave": 9000.0,
            "detector_edge_tolerance": 500.0,
        },
        "hough": {
            "num_slopes": 2000,
            "range_tolerance": 100.0,
            "xbins": 100,
            "ybins": 100,
        },
        "ransac": {"minimum_fit_error": 1e-25, "max_tries": 1000},
    }

    # Set up the calibrator with the pixel values of our
    # wavelengths
    c = Calibrator(peaks, atlas_lines=atlas.atlas_lines, config=config)

    c.do_hough_transform(brute_force=False)

    # Check that all the ground truth lines are in the pair list:
    ground_truth = list(zip(peaks.round(2), waves.round(2)))
    for point in ground_truth:
        assert point in c.pairs.round(2)
        # idx = np.where((c.pairs.round(2) == point).all(axis=1))[0][0]

    # And let's try and fit...
    res = c.fit()

    assert res["success"]

    res = c.match_peaks(res["fit_coeff"], refine=False, robust_refit=True)

    assert res["peak_utilisation"] > 0.7
    assert res["atlas_utilisation"] > 0.0
    assert res["rms"] < 5.0


# def test_fitting_with_initial_polynomial():

#     # Set up the calibrator with the pixel values of our
#     # wavelengths
#     c = Calibrator(peaks=peaks)
#     a = Atlas()

#     # Arbitrarily we'll set the number of pixels to 768 (i.e.
#     # a max range of around 1500 nm
#     c.set_calibrator_properties(num_pix=768)

#     # Setup the Hough transform parameters
#     c.set_hough_properties(
#         range_tolerance=100.0, min_wavelength=100.0, max_wavelength=1500.0
#     )

#     c.set_ransac_properties(linear=False, minimum_fit_error=1e-12)

#     # Add our fake lines as the atlas
#     a.add_user_atlas(elements=["Test"] * len(waves), wavelengths=waves)
#     c.set_atlas(a)
#     assert len(c.atlas.atlas_lines) > 0

#     c.do_hough_transform(brute_force=False)

#     # And let's try and fit...
#     res = c.fit(max_tries=500, fit_coeff=best_p)

#     res = c.match_peaks(res['fit_coeff'], refine=False, robust_refit=True)

#     assert res['peak_utilisation'] > 0.7
#     assert res['atlas_utilisation'] > 0.0
#     assert res['rms'] < 5.0
