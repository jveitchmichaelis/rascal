import os
from functools import partialmethod

import numpy as np

# Suppress tqdm output
import pytest
from astropy.io import fits
from scipy.signal import find_peaks

from rascal import util
from rascal.atlas import Atlas
from rascal.calibrator import Calibrator


@pytest.fixture(scope="session")
def base_dir():
    return os.path.dirname(__file__)


@pytest.fixture(scope="session")
def calibrator(base_dir):
    # Line list
    wavelengths = [
        4193.5,
        4385.77,
        4500.98,
        4524.68,
        4582.75,
        4624.28,
        4671.23,
        4697.02,
        4734.15,
        4807.02,
        4921.48,
        5028.28,
        5618.88,
        5823.89,
        5893.29,
        5934.17,
        6182.42,
        6318.06,
        6472.841,
        6595.56,
        6668.92,
        6728.01,
        6827.32,
        6976.18,
        7119.60,
        7257.9,
        7393.8,
        7584.68,
        7642.02,
        7740.31,
        7802.65,
        7887.40,
        7967.34,
        8057.258,
        10000.0,
    ]
    elements = ["Xe"] * len(wavelengths)

    # Load the LT SPRAT data
    spectrum2D = fits.open(
        os.path.join(
            base_dir, "..", "examples/data_lt_sprat/v_a_20190516_57_1_0_1.fits"
        )
    )[0].data

    # Collapse into 1D spectrum between row 110 and 120
    spectrum = np.median(spectrum2D[110:120], axis=0)

    # Identify the peaks
    peaks, _ = find_peaks(
        spectrum, height=300, prominence=150, distance=5, threshold=None
    )
    peaks = util.refine_peaks(spectrum, peaks, window_width=5)

    # Initialise the calibrator
    c = Calibrator(peaks)
    a = Atlas()

    c.set_calibrator_properties(num_pix=1024)
    c.set_hough_properties(
        num_slopes=2000,
        range_tolerance=200.0,
        xbins=100,
        ybins=100,
        min_wavelength=3600.0,
        max_wavelength=8000.0,
    )
    a.add_user_atlas(elements=elements, wavelengths=wavelengths)
    c.set_atlas(a)
    c.atlas.clear()
    assert len(a.atlas_lines) == 0
    a.add_user_atlas(elements=elements, wavelengths=wavelengths)
    c.set_atlas(a)
    c.atlas.remove_atlas_lines_range(9999.0)
    assert len(c.atlas.atlas_lines) == len(wavelengths) - 1
    c.atlas.list()

    return c


@pytest.fixture(scope="session")
def result(calibrator):

    # Run the wavelength calibration
    res = calibrator.fit(max_tries=7500, fit_deg=4, candidate_tolerance=5.0)
    assert res
    rms = res["rms"]
    n_peaks = len(calibrator.matched_peaks)

    assert n_peaks > 10

    return res


def check_fit(calibrator, result):

    n_peaks = len(calibrator.matched_peaks)
    n_atlas = len(calibrator.matched_atlas)

    assert len(calibrator.matched_peaks) >= n_peaks
    assert len(calibrator.matched_atlas) >= n_atlas


def test_match_peaks_robust_refit(calibrator, result):

    # repeat 10 times
    for i in range(10):
        # Refine solution
        _ = calibrator.match_peaks(
            result["fit_coeff"], refine=True, robust_refit=True
        )
        check_fit(calibrator, result)


def test_match_peaks_NOT_robust_refit(calibrator, result):

    fit_coeff = result["fit_coeff"]

    # repeat 10 times
    for i in range(10):
        # Refine solution
        _ = calibrator.match_peaks(fit_coeff, refine=True, robust_refit=False)
        check_fit(calibrator, result)


def test_match_peaks_robust_refit_powell(calibrator, result):

    fit_coeff = result["fit_coeff"]

    # repeat 10 times
    for i in range(10):
        # Refine solution
        _ = calibrator.match_peaks(
            fit_coeff, refine=True, robust_refit=True, method="powell"
        )
        check_fit(calibrator, result)


def test_match_peaks_robust_refit_different_fit_deg(calibrator, result):

    fit_coeff = result["fit_coeff"]

    # repeat 10 times
    for i in range(10):
        # Refine solution
        _ = calibrator.match_peaks(
            fit_coeff,
            refine=True,
            robust_refit=True,
            fit_deg=5,
        )
        check_fit(calibrator, result)


def test_match_peaks_output(calibrator, base_dir):

    calibrator.save_matches(
        filename=os.path.join(base_dir, "test_output", "matches"),
        file_format="csv",
    )
    assert os.path.exists(os.path.join(base_dir, "test_output", "matches.csv"))
    calibrator.save_matches(
        filename=os.path.join(base_dir, "test_output", "matches"),
        file_format="npy",
    )
    assert os.path.exists(os.path.join(base_dir, "test_output", "matches.npy"))
