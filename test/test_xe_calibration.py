import os
import logging

from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
import pytest
from scipy.signal import find_peaks

from rascal.calibrator import Calibrator
from rascal import util
from rascal import models

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


def run_sprat_calibration(polydeg):

    # Load the LT SPRAT data
    base_dir = os.path.dirname(__file__)
    spectrum2D = fits.open(os.path.join(base_dir, '..', 'examples/data_lt_sprat/v_a_20190516_57_1_0_1.fits'))[0].data

    # Collapse into 1D spectrum between row 110 and 120
    spectrum = np.median(spectrum2D[110:120], axis=0)

    # Identify the peaks
    peaks, _ = find_peaks(spectrum, height=500, distance=5, threshold=None)
    peaks = util.refine_peaks(spectrum, peaks, window_width=5)

    # Initialise the calibrator
    c = Calibrator(peaks, num_pix=1024, min_wavelength=3500., max_wavelength=8000.)
    c.set_fit_constraints(num_slopes=10000,
                          range_tolerance=500.,
                          xbins=100,
                          ybins=100,
                          polydeg=polydeg)
    c.add_atlas(elements='Xe')

    # Run the wavelength calibration
    best_p, rms, residual, peak_utilisation = c.fit(max_tries=10000)

    # Refine solution
    # First set is to refine only the 0th and 1st coefficient (i.e. the 2 lowest orders)
    best_p, x_fit, y_fit, residual, peak_utilisation = c.match_peaks(
        best_p,
        delta=best_p[:1] * 0.001,
        tolerance=10.,
        convergence=1e-10,
        method='Nelder-Mead',
        robust_refit=True)

    # Second set is to refine all the coefficients
    best_p, x_fit, y_fit, residual, peak_utilisation = c.match_peaks(
        best_p,
        delta=best_p * 0.001,
        tolerance=10.,
        convergence=1e-10,
        method='Nelder-Mead',
        robust_refit=True)

    fit_diff = c.polyval(x_fit, best_p) - y_fit
    rms = np.sqrt(np.sum(fit_diff**2 / len(x_fit)))

    return best_p, residual, peak_utilisation, rms


def test_sprat_calibration():

    logger.info("Test if LT/SPRAT Xe calibration return the order of polynomial properly.")

    for i in range(3, 6):
        best_p, _, _, _ = run_sprat_calibration(polydeg=i)
        assert (len(best_p) == (i + 1))


def test_sprat_calibration_multirun():

    logger.info("Test the repeatability of LT/SPRAT Xe calibration.")

    n = 10

    c0 = np.zeros(n)
    c1 = np.zeros(n)
    c2 = np.zeros(n)
    c3 = np.zeros(n)
    c4 = np.zeros(n)
    peak_utilisation = np.zeros(n)
    rms = np.zeros(n)

    for i in range(n):
        best_p, _, peak_utilisation[i], rms[i] = run_sprat_calibration(4)
        c0[i], c1[i], c2[i], c3[i], c4[i] = best_p

    assert np.std(c0) < 500.
    assert np.std(c1) < 10
    assert np.std(c2) < 1
    assert np.std(c3) < 0.1
    assert np.std(c4) < 0.01
    assert np.std(peak_utilisation) < 10.
    assert np.std(rms) < 5.



