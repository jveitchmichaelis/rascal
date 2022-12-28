import logging
import os
from functools import partialmethod

import numpy as np
import pkg_resources
from astropy.io import fits
from scipy.signal import find_peaks

# Suppress tqdm output
from tqdm import tqdm

from rascal import util
from rascal.atlas import Atlas
from rascal.calibrator import Calibrator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

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


def run_sprat_calibration(fit_deg):

    # Load the LT SPRAT data
    base_dir = os.path.dirname(__file__)
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

    # Run the wavelength calibration
    res = c.fit(
        max_tries=5000, fit_deg=fit_deg, candidate_tolerance=5.0, use_msac=True
    )

    assert res

    # Refine solution
    res = c.match_peaks(res["fit_coeff"], refine=False, robust_refit=True)

    return res


def test_run_sprat_calibration_with_manual_linelist_file():

    # Load the LT SPRAT data
    base_dir = os.path.dirname(__file__)
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
    a.add(
        elements=["Xe"],
        linelist=pkg_resources.resource_filename(
            "rascal", "arc_lines/nist_clean.csv"
        ),
    )
    c.set_atlas(a)
    c.set_calibrator_properties(num_pix=1024)
    c.set_hough_properties(
        num_slopes=2000,
        range_tolerance=200.0,
        xbins=100,
        ybins=100,
        min_wavelength=3600.0,
        max_wavelength=8000.0,
    )

    # Run the wavelength calibration
    res = c.fit(max_tries=2500, fit_deg=4, candidate_tolerance=5)
    assert res


def test_sprat_calibration():

    logger.info(
        "Test if LT/SPRAT Xe calibration return the order of "
        "polynomial properly."
    )

    for i in range(3, 6):
        res = run_sprat_calibration(fit_deg=i)
        assert len(res["fit_coeff"]) == (i + 1)


def test_sprat_calibration_multirun():

    logger.info("Test the repeatability of LT/SPRAT Xe calibration.")

    n = 10

    c0 = np.zeros(n)
    c1 = np.zeros(n)
    c2 = np.zeros(n)
    c3 = np.zeros(n)
    c4 = np.zeros(n)
    peak_utilisation = np.zeros(n)
    atlas_utilisation = np.zeros(n)
    rms = np.zeros(n)

    for i in range(n):
        res = run_sprat_calibration(4)
        c0[i], c1[i], c2[i], c3[i], c4[i] = res["fit_coeff"]
        peak_utilisation[i] = res["peak_utilisation"]
        atlas_utilisation[i] = res["atlas_utilisation"]
        rms[i] = res["rms"]

    assert np.std(c0) < 500.0
    assert np.std(c1) < 10
    assert np.std(c2) < 1
    assert np.std(c3) < 0.1
    assert np.std(c4) < 0.01
    assert np.std(peak_utilisation) < 10.0
    assert np.std(atlas_utilisation) < 10.0
    assert np.std(rms) < 5.0
