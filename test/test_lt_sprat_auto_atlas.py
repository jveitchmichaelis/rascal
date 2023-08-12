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

from rascal import util
from rascal.atlas import AtlasCollection
from rascal.calibrator import Calibrator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

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

print(len(sprat_atlas_lines))

config = {
    "data": {
        "contiguous_range": None,
        "detector_min_wave": 3500.0,
        "detector_max_wave": 8200.0,
        "detector_edge_tolerance": 100.0,
        "num_pix": 1024,
    },
    "hough": {
        "num_slopes": 2000,
        "range_tolerance": 100.0,
        "xbins": 100,
        "ybins": 100,
    },
    "ransac": {
        "max_tries": 5000,
        "inlier_tolerance": 1.0,
        "sample_size": 5,
        "top_n_candidate": 5,
        "filter_close": True,
    },
    "atlases": [
        {
            "elements": ["Xe"],
            "min_wavelength": 3500,
            "max_wavelength": 8200,
            "min_intensity": 10.0,
            "min_distance": 30,
            "range_tolerance": 100,
            "brightest_n_lines": 50,
        }
    ],
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


@pytest.fixture
def auto_atlas():
    return AtlasCollection.from_config(config)


def test_atlas_against_ground_truth(auto_atlas):

    print("--")

    atlas_wavelengths = np.array(auto_atlas.wavelengths)
    print(f"Length of atlas: {len(atlas_wavelengths)}")

    for line in sprat_atlas_lines:
        tol = 0.5
        print(
            line,
            np.isclose(line, atlas_wavelengths, atol=tol).any(),
            atlas_wavelengths[
                np.where(np.isclose(atlas_wavelengths, line, atol=tol))
            ],
        )


def test_sprat_calibration(auto_atlas):

    config["ransac"]["degree"] = 5
    config["ransac"]["sample_size"] = 8

    # Initialise the calibrator
    c = Calibrator(
        peaks,
        atlas_lines=auto_atlas.atlas_lines,
        config=config,
        spectrum=spectrum,
    )

    print("Number of peaks:", len(peaks))
    print("Atlas length: ", len(auto_atlas.atlas_lines))

    # Run the wavelength calibration
    res = c.fit()
    assert res["success"]

    # Refine solution
    res = c.match_peaks(res["fit_coeff"], refine=False, robust_refit=True)

    return res


"""

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
    res = c.fit()
    assert res["success"]


@pytest.mark.timeout(180)
def test_sprat_calibration():

    logger.info(
        "Test if LT/SPRAT Xe calibration return the order of "
        "polynomial properly."
    )

    for i in range(3, 6):
        res = run_sprat_calibration(fit_deg=i)
        assert len(res["fit_coeff"]) == (i + 1)


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

    assert np.std(c0) < 500.0
    assert np.std(c1) < 10
    assert np.std(c2) < 1
    assert np.std(c3) < 0.1
    assert np.std(c4) < 0.01
    assert np.std(peak_utilisation) < 10.0
    assert np.std(atlas_utilisation) < 10.0
    assert np.std(rms) < 5.0
"""