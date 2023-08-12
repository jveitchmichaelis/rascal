from functools import partialmethod

import numpy as np

# Suppress tqdm output
from tqdm import tqdm

from rascal.atlas import Atlas
from rascal.calibrator import Calibrator

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

np.random.seed(0)

peaks = np.sort(np.random.random(31) * 1000.0)
# Removed the closely spaced peaks
distance_mask = np.isclose(peaks[:-1], peaks[1:], atol=5.0)
distance_mask = np.insert(distance_mask, 0, False)
peaks = peaks[~distance_mask]

# Line list
wavelengths_linear = 3000.0 + 5.0 * peaks
wavelengths_quadratic = 3000.0 + 4 * peaks + 1.0e-3 * peaks**2.0

elements_linear = ["Linear"] * len(wavelengths_linear)
elements_quadratic = ["Quadratic"] * len(wavelengths_quadratic)


def test_linear_fit():
    atlas = Atlas(
        source="manual",
        wavelengths=wavelengths_linear,
        min_wavelength=3500.0,
        max_wavelength=8000.0,
        elements=elements_linear,
    )

    config = {
        "data": {"contiguous_range": None, "num_pix": 1000},
        "hough": {
            "num_slopes": 1000,
            "range_tolerance": 200.0,
            "xbins": 200,
            "ybins": 200,
        },
        "ransac": {
            "minimum_matches": 20,
            "minimum_fit_error": 1e-25,
        },
    }

    # Initialise the calibrator
    c = Calibrator(peaks, atlas_lines=atlas.atlas_lines, config=config)

    c.do_hough_transform(brute_force=False)

    # Run the wavelength calibration
    res = c.fit(max_tries=500, fit_deg=1)
    print(res)
    # Refine solution
    res = c.match_peaks(res["fit_coeff"], refine=False, robust_refit=True)

    assert np.abs(res["fit_coeff"][1] - 5.0) / 5.0 < 0.001
    assert np.abs(res["fit_coeff"][0] - 3000.0) / 3000.0 < 0.001
    assert res["peak_utilisation"] > 0.8
    assert res["atlas_utilisation"] > 0.0
    assert len(c.get_pix_wave_pairs()) == len(peaks)


def test_manual_refit():
    atlas = Atlas(
        source="manual",
        wavelengths=wavelengths_linear,
        min_wavelength=3500.0,
        max_wavelength=8000.0,
        elements=elements_linear,
    )

    config = {
        "data": {"contiguous_range": None, "num_pix": 1000},
        "hough": {
            "num_slopes": 1000,
            "range_tolerance": 500.0,
            "xbins": 200,
            "ybins": 200,
        },
        "ransac": {
            "minimum_matches": 20,
            "minimum_fit_error": 1e-25,
        },
    }

    # Initialise the calibrator
    c = Calibrator(peaks, atlas_lines=atlas.atlas_lines, config=config)

    c.do_hough_transform(brute_force=False)

    # Run the wavelength calibration
    res = c.fit(max_tries=500, fit_deg=1)

    # Refine solution
    res = c.match_peaks(res["fit_coeff"], refine=False, robust_refit=True)
    res_manual = c.manual_refit(res["matched_peaks"], res["matched_atlas"])

    assert np.allclose(res_manual["fit_coeff"], res["fit_coeff"])


def test_manual_refit_remove_points():
    atlas = Atlas(
        source="manual",
        wavelengths=wavelengths_linear,
        min_wavelength=3500.0,
        max_wavelength=8000.0,
        elements=elements_linear,
    )

    config = {
        "data": {"contiguous_range": None, "num_pix": 1000},
        "hough": {
            "num_slopes": 1000,
            "range_tolerance": 500.0,
            "xbins": 200,
            "ybins": 200,
        },
        "ransac": {
            "minimum_matches": 20,
            "minimum_fit_error": 1e-25,
        },
    }

    # Initialise the calibrator
    c = Calibrator(peaks, atlas_lines=atlas.atlas_lines, config=config)

    c.do_hough_transform(brute_force=False)

    # Run the wavelength calibration
    res = c.fit(max_tries=500, fit_deg=1)

    # Refine solution
    res = c.match_peaks(res["fit_coeff"], refine=False, robust_refit=True)

    c.remove_pix_wave_pair(5)

    res_manual = c.manual_refit(res["matched_peaks"], res["matched_atlas"])

    assert np.allclose(res_manual["fit_coeff"], res["fit_coeff"])


def test_manual_refit_add_points():
    atlas = Atlas(
        source="manual",
        wavelengths=wavelengths_linear,
        min_wavelength=3500.0,
        max_wavelength=8000.0,
        elements=elements_linear,
    )

    config = {
        "data": {"contiguous_range": None, "num_pix": 1000},
        "hough": {
            "num_slopes": 1000,
            "range_tolerance": 500.0,
            "xbins": 200,
            "ybins": 200,
        },
        "ransac": {
            "minimum_matches": 20,
            "minimum_fit_error": 1e-25,
        },
    }

    # Initialise the calibrator
    c = Calibrator(peaks, atlas_lines=atlas.atlas_lines, config=config)

    c.do_hough_transform(brute_force=False)

    # Run the wavelength calibration
    res = c.fit(max_tries=500, fit_deg=1)

    # Refine solution
    res = c.match_peaks(res["fit_coeff"], refine=False, robust_refit=True)

    c.add_pix_wave_pair(2000.0, 3000.0 + 4 * 2000.0 + 1.0e-3 * 2000.0**2.0)
    res_manual = c.manual_refit(res["matched_peaks"], res["matched_atlas"])

    assert np.allclose(res_manual["fit_coeff"], res["fit_coeff"])


def test_quadratic_fit():
    atlas = Atlas(
        source="manual",
        wavelengths=wavelengths_quadratic,
        min_wavelength=3500.0,
        max_wavelength=8000.0,
        elements=elements_quadratic,
    )

    config = {
        "data": {"contiguous_range": None, "num_pix": 1000},
        "hough": {
            "num_slopes": 1000,
            "range_tolerance": 500.0,
            "xbins": 200,
            "ybins": 200,
        },
        "ransac": {
            "minimum_matches": 20,
            "minimum_fit_error": 1e-25,
        },
    }

    # Initialise the calibrator
    c = Calibrator(peaks, atlas_lines=atlas.atlas_lines, config=config)

    c.do_hough_transform(brute_force=False)

    # Run the wavelength calibration
    res = c.fit(
        max_tries=2000, fit_tolerance=5.0, candidate_tolerance=2.0, fit_deg=2
    )
    # Refine solution
    res_robust = c.match_peaks(
        res["fit_coeff"], refine=False, robust_refit=True
    )

    assert np.abs(res_robust["fit_coeff"][2] - 1e-3) / 1e-3 < 0.001
    assert np.abs(res_robust["fit_coeff"][1] - 4.0) / 4.0 < 0.001
    assert np.abs(res_robust["fit_coeff"][0] - 3000.0) / 3000.0 < 0.001
    assert res_robust["peak_utilisation"] > 0.7
    assert res_robust["atlas_utilisation"] > 0.5


def test_quadratic_fit_legendre():
    atlas = Atlas(
        source="manual",
        wavelengths=wavelengths_quadratic,
        min_wavelength=3500.0,
        max_wavelength=8000.0,
        elements=elements_quadratic,
    )

    config = {
        "data": {"contiguous_range": None, "num_pix": 1000},
        "hough": {
            "num_slopes": 500,
            "range_tolerance": 200.0,
            "xbins": 100,
            "ybins": 100,
        },
        "ransac": {
            "sample_size": 5,
            "minimum_matches": 10,
            "minimum_fit_error": 1e-25,
        },
    }

    # Initialise the calibrator
    c = Calibrator(peaks, atlas_lines=atlas.atlas_lines, config=config)

    c.do_hough_transform(brute_force=False)

    # Run the wavelength calibration
    res = c.fit(
        max_tries=10000,
        fit_tolerance=5.0,
        candidate_tolerance=2.0,
        fit_deg=2,
        fit_type="legendre",
    )

    # Legendre 2nd order takes the form
    assert np.abs(res["fit_coeff"][1] - 4.0) / 4.0 < 0.001
    assert np.abs(res["fit_coeff"][0] - 3000.0) / 3000.0 < 0.001
    assert res["peak_utilisation"] > 0.6
    assert res["atlas_utilisation"] > 0.5


def test_quadratic_fit_chebyshev():
    atlas = Atlas(
        source="manual",
        wavelengths=wavelengths_quadratic,
        min_wavelength=3500.0,
        max_wavelength=8000.0,
        elements=elements_quadratic,
    )

    config = {
        "data": {"contiguous_range": None, "num_pix": 1000},
        "hough": {
            "num_slopes": 500,
            "range_tolerance": 200.0,
            "xbins": 100,
            "ybins": 100,
        },
        "ransac": {
            "sample_size": 5,
            "minimum_matches": 10,
            "minimum_fit_error": 1e-25,
        },
    }

    # Initialise the calibrator
    c = Calibrator(peaks, atlas_lines=atlas.atlas_lines, config=config)

    c.do_hough_transform(brute_force=False)

    # Run the wavelength calibration
    res = c.fit(
        max_tries=10000,
        fit_tolerance=5.0,
        candidate_tolerance=2.0,
        fit_deg=2,
        fit_type="chebyshev",
    )

    assert np.abs(res["fit_coeff"][1] - 4.0) / 4.0 < 0.001
    assert np.abs(res["fit_coeff"][0] - 3000.0) / 3000.0 < 0.001
    assert res["peak_utilisation"] > 0.6
    assert res["atlas_utilisation"] > 0.5
