import logging
import os

from astropy.io import fits
import numpy as np
from scipy.signal import find_peaks

from rascal.atlas import Atlas
from rascal.calibrator import Calibrator
from rascal import util

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
    peaks, _ = find_peaks(spectrum, height=500, distance=5, threshold=None)
    peaks = util.refine_peaks(spectrum, peaks, window_width=5)

    # Initialise the calibrator
    c = Calibrator(peaks)
    a = Atlas()

    c.set_calibrator_properties(num_pix=1024)
    c.set_hough_properties(
        num_slopes=1000,
        range_tolerance=500.0,
        xbins=100,
        ybins=100,
        min_wavelength=3500.0,
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
    best_p, x, y, rms, residual, peak_utilisation, atlas_utilisation = c.fit(
        max_tries=200, fit_deg=fit_deg
    )

    # Refine solution
    (
        best_p,
        x_fit,
        y_fit,
        rms,
        residual,
        peak_utilisation,
        atlas_utilisation,
    ) = c.match_peaks(best_p, refine=False, robust_refit=True)

    fit_diff = c.polyval(x_fit, best_p) - y_fit
    rms = np.sqrt(np.sum(fit_diff**2 / len(x_fit)))

    return best_p, residual, peak_utilisation, atlas_utilisation, rms


def test_sprat_calibration():

    logger.info(
        "Test if LT/SPRAT Xe calibration return the order of "
        "polynomial properly."
    )

    for i in range(3, 6):
        best_p, _, _, _, _ = run_sprat_calibration(fit_deg=i)
        assert len(best_p) == (i + 1)


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
        (
            best_p,
            _,
            peak_utilisation[i],
            atlas_utilisation[i],
            rms[i],
        ) = run_sprat_calibration(4)
        c0[i], c1[i], c2[i], c3[i], c4[i] = best_p

    assert np.std(c0) < 500.0
    assert np.std(c1) < 10
    assert np.std(c2) < 1
    assert np.std(c3) < 0.1
    assert np.std(c4) < 0.01
    assert np.std(peak_utilisation) < 10.0
    assert np.std(atlas_utilisation) < 10.0
    assert np.std(rms) < 5.0
