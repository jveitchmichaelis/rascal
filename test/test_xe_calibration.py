import logging
import os
from functools import partialmethod

import numpy as np
import pkg_resources
import pytest
from astropy.io import fits
from rascal import util
from rascal.atlas import Atlas
from rascal.calibrator import Calibrator
from scipy.signal import find_peaks

# Suppress tqdm output
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


# blend: 4829.71, 4844.33
# blend: 5566.62, 5581.88
# blend: 6261.212, 6265.302
# blend: 6872.11, 6882.16
# blend: 7283.961, 7285.301
# blend: 7316.272, 7321.452
sprat_atlas_lines = [
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
]
element = ["Xe"] * len(sprat_atlas_lines)

config = {
    "data": {"contiguous_range": None},
    "hough": {
        "num_slopes": 2000,
        "range_tolerance": 200.0,
        "xbins": 100,
        "ybins": 100,
    },
    "ransac": {
        "sample_size": 5,
        "top_n_candidate": 5,
        "filter_close": True,
    },
}

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


def run_sprat_calibration(fit_deg):

    atlas = Atlas(
        line_list="manual",
        wavelengths=sprat_atlas_lines,
        min_wavelength=3500.0,
        max_wavelength=8000.0,
        range_tolerance=500.0,
        elements=element,
    )

    # Initialise the calibrator
    c = Calibrator(
        peaks, atlas_lines=atlas.atlas_lines, config=config, spectrum=spectrum
    )

    """
    c.set_atlas(a)
    c.atlas.clear()
    assert len(a.atlas_lines) == 0
    a.add_user_atlas(elements=elements, wavelengths=wavelengths)
    c.set_atlas(a)
    c.atlas.remove_atlas_lines_range(9999.0)
    assert len(c.atlas.atlas_lines) == len(wavelengths) - 1
    c.atlas.list()
    """

    # Run the wavelength calibration
    res = c.fit(
        max_tries=5000, fit_deg=fit_deg, candidate_tolerance=5.0, use_msac=True
    )

    assert res

    # Refine solution
    res = c.match_peaks(res["fit_coeff"], refine=False, robust_refit=True)

    return res


def test_run_sprat_calibration_with_manual_linelist_file():

    atlas = Atlas(
        line_list="manual",
        wavelengths=sprat_atlas_lines,
        min_wavelength=3500.0,
        max_wavelength=8000.0,
        range_tolerance=500.0,
        elements=element,
    )

    # Initialise the calibrator
    c = Calibrator(
        peaks, atlas_lines=atlas.atlas_lines, config=config, spectrum=spectrum
    )

    # Run the wavelength calibration
    res = c.fit(max_tries=2500, fit_deg=4, candidate_tolerance=5)
    assert res


@pytest.mark.timeout(180)
def test_sprat_calibration():

    logger.info(
        "Test if LT/SPRAT Xe calibration return the order of "
        "polynomial properly."
    )

    for i in range(3, 6):
        res = run_sprat_calibration(fit_deg=i)
        # assert len(res["fit_coeff"]) == (i + 1)


@pytest.mark.timeout(300)
def test_sprat_calibration_multirun():

    logger.info("Test the repeatability of LT/SPRAT Xe calibration.")

    n = 5

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

    """
    assert np.std(c0) < 500.0
    assert np.std(c1) < 10
    assert np.std(c2) < 1
    assert np.std(c3) < 0.1
    assert np.std(c4) < 0.01
    assert np.std(peak_utilisation) < 10.0
    assert np.std(atlas_utilisation) < 10.0
    assert np.std(rms) < 5.0
    """
