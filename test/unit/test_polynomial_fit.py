from functools import partialmethod

import numpy as np

# Suppress tqdm output
from tqdm import tqdm

from rascal.atlas import Atlas
from rascal.calibrator import Calibrator

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def test_linear_fit(peaks, detector_config, wavelengths_linear):
    atlas = Atlas(
        source="manual",
        wavelengths=wavelengths_linear,
        min_wavelength=3000.0,
        max_wavelength=8000.0,
        element="linear",
    )

    config = {
        "seed": 0,
        "detector": detector_config,
        "hough": {
            "num_slopes": 1000,
            "range_tolerance": 200.0,
            "xbins": 200,
            "ybins": 200,
        },
        "ransac": {
            "minimum_matches": 20,
            "minimum_fit_error": 1e-25,
            "max_tries": 500,
            "degree": 1,
            "sample_size": 3
        },
    }

    # Initialise the calibrator
    c = Calibrator(peaks, atlas_lines=atlas.atlas_lines, config=config)

    c.do_hough_transform(brute_force=False)

    # Run the wavelength calibration
    res = c.fit()

    # Refine solution
    res = c.match_peaks(res["fit_coeff"], refine=False, robust_refit=True)
    assert np.abs(res["fit_coeff"][1] - 5.0) / 5.0 < 0.001
    assert np.abs(res["fit_coeff"][0] - 3000.0) / 3000.0 < 0.001
    assert res["fit_coeff"][0] > 0
    assert res["peak_utilisation"] > 0.8
    assert res["atlas_utilisation"] > 0.0
    assert len(c.get_pix_wave_pairs()) == len(peaks)


def test_manual_refit(peaks, detector_config, wavelengths_linear):
    atlas = Atlas(
        source="manual",
        wavelengths=wavelengths_linear,
        min_wavelength=3500.0,
        max_wavelength=8000.0,
        element="linear",
    )

    config = {
        "seed": 0,
        "detector": detector_config,
        "hough": {
            "num_slopes": 1000,
            "range_tolerance": 200.0,
            "xbins": 200,
            "ybins": 200,
        },
        "ransac": {
            "minimum_matches": 20,
            "minimum_fit_error": 1e-25,
            "max_tries": 500,
            "degree": 1,
            "sample_size": 3
        },
    }

    # Initialise the calibrator
    c = Calibrator(peaks, atlas_lines=atlas.atlas_lines, config=config)

    c.do_hough_transform(brute_force=False)

    # Run the wavelength calibration
    res = c.fit()

    # Refine solution
    res = c.match_peaks(res["fit_coeff"], refine=False, robust_refit=True)
    res_manual = c.manual_refit(res["matched_peaks"], res["matched_atlas"])

    assert np.allclose(res_manual["fit_coeff"], res["fit_coeff"])
    assert res["fit_coeff"][0] > 0


def test_manual_refit_remove_points(peaks, detector_config, wavelengths_linear):
    atlas = Atlas(
        source="manual",
        wavelengths=wavelengths_linear,
        min_wavelength=3000.0,
        max_wavelength=8000.0,
        element="linear",
    )

    config = {
        "seed": 0,
        "detector": detector_config,
        "hough": {
            "num_slopes": 1000,
            "range_tolerance": 200.0,
            "xbins": 200,
            "ybins": 200,
        },
        "ransac": {
            "minimum_matches": 20,
            "minimum_fit_error": 1e-25,
            "max_tries": 500,
            "degree": 1,
            "sample_size": 3
        },
    }

    # Initialise the calibrator
    c = Calibrator(peaks, atlas_lines=atlas.atlas_lines, config=config)

    c.do_hough_transform(brute_force=False)

    # Run the wavelength calibration
    res = c.fit()

    # Refine solution
    res = c.match_peaks(res["fit_coeff"], refine=False, robust_refit=True)

    c.remove_pix_wave_pair(5)

    res_manual = c.manual_refit(res["matched_peaks"], res["matched_atlas"])

    assert np.allclose(res_manual["fit_coeff"], res["fit_coeff"])
    assert res["fit_coeff"][0] > 0


def test_manual_refit_add_points(peaks, detector_config, wavelengths_linear):
    atlas = Atlas(
        source="manual",
        wavelengths=wavelengths_linear,
        min_wavelength=3500.0,
        max_wavelength=8000.0,
        element="linear",
    )

    config = {
        "seed": 0,
        "detector": detector_config,
        "hough": {
            "num_slopes": 1000,
            "range_tolerance": 200.0,
            "xbins": 200,
            "ybins": 200,
        },
        "ransac": {
            "minimum_matches": 20,
            "minimum_fit_error": 1e-25,
            "max_tries": 500,
            "degree": 1,
            "sample_size": 3
        },
    }

    # Initialise the calibrator
    c = Calibrator(peaks, atlas_lines=atlas.atlas_lines, config=config)

    c.do_hough_transform(brute_force=False)

    # Run the wavelength calibration
    res = c.fit()

    # Refine solution
    res = c.match_peaks(res["fit_coeff"], refine=False, robust_refit=True)

    c.add_pix_wave_pair(2000.0, 3000.0 + 4 * 2000.0 + 1.0e-3 * 2000.0**2.0)
    res_manual = c.manual_refit(res["matched_peaks"], res["matched_atlas"])

    assert np.allclose(res_manual["fit_coeff"], res["fit_coeff"])
    assert res["fit_coeff"][0] > 0


def test_quadratic_fit(peaks, detector_config, wavelengths_quadratic):
    atlas = Atlas(
        source="manual",
        wavelengths=wavelengths_quadratic,
        min_wavelength=3000.0,
        max_wavelength=8000.0,
        element="quadratic",
    )

    config = {
        "seed": 0,
        "detector": detector_config,
        "hough": {
            "num_slopes": 1000,
            "range_tolerance": 200.0,
            "xbins": 200,
            "ybins": 200,
            "linearity_tolerance": 10.0,
        },
        "ransac": {
            "top_n_candidate": 10,
            "sample_size": 4,
            "minimum_matches": 20,
            "minimum_fit_error": 1e-25,
            "max_tries": 2000,
            "rms_tolerance": 5.0,
            "inlier_tolerance": 2.0,
            "degree": 2,
        },
    }

    # Initialise the calibrator
    c = Calibrator(peaks, atlas_lines=atlas.atlas_lines, config=config)

    c.do_hough_transform(brute_force=False)

    c._get_candidate_points_linear(c.config.hough.linearity_tolerance)
    all_cand_peaks = np.concatenate([cand[0] for cand in c.candidates])
    all_cand_waves = np.concatenate([cand[1] for cand in c.candidates])
    n_recalled = sum(
        np.any(
            np.isclose(all_cand_peaks, p, atol=1e-6)
            & np.isclose(all_cand_waves, wavelengths_quadratic[i], atol=1e-6)
        )
        for i, p in enumerate(peaks)
    )
    candidate_recall = n_recalled / len(peaks)
    assert candidate_recall > 0.9

    # Run the wavelength calibration
    res = c.fit()
    # Refine solution
    res_robust = c.match_peaks(
        res["fit_coeff"], refine=False, robust_refit=True
    )
    assert np.abs(res_robust["fit_coeff"][2] - 1e-3) / 1e-3 < 0.001
    assert np.abs(res_robust["fit_coeff"][1] - 4.0) / 4.0 < 0.001
    assert np.abs(res_robust["fit_coeff"][0] - 3000.0) / 3000.0 < 0.001
    assert res_robust["fit_coeff"][0] > 0
    assert res_robust["peak_utilisation"] > 0.9
    assert res_robust["atlas_utilisation"] > 0.9


def test_quadratic_fit_legendre(peaks, detector_config, wavelengths_quadratic):
    atlas = Atlas(
        source="manual",
        wavelengths=wavelengths_quadratic,
        min_wavelength=3000.0,
        max_wavelength=8000.0,
        element="quadratic",
    )

    config = {
        "seed": 0,
        "detector": detector_config,
        "hough": {
            "num_slopes": 500,
            "range_tolerance": 200.0,
            "xbins": 100,
            "ybins": 100,
            "linearity_tolerance": 100.0,
        },
        "ransac": {
            "sample_size": 4,
            "top_n_candidate": 10,
            "minimum_matches": 10,
            "minimum_fit_error": 1e-25,
            "max_tries": 500,
            "rms_tolerance": 5.0,
            "inlier_tolerance": 2.0,
            "degree": 2,
            "fit_type": "legendre",
        },
    }

    # Initialise the calibrator
    c = Calibrator(peaks, atlas_lines=atlas.atlas_lines, config=config)

    c.do_hough_transform(brute_force=False)

    c._get_candidate_points_linear(c.config.hough.linearity_tolerance)
    all_cand_peaks = np.concatenate([cand[0] for cand in c.candidates])
    all_cand_waves = np.concatenate([cand[1] for cand in c.candidates])
    n_recalled = sum(
        np.any(
            np.isclose(all_cand_peaks, p, atol=1e-6)
            & np.isclose(all_cand_waves, wavelengths_quadratic[i], atol=1e-6)
        )
        for i, p in enumerate(peaks)
    )
    candidate_recall = n_recalled / len(peaks)
    assert candidate_recall > 0.9

    # Run the wavelength calibration
    res = c.fit()

    assert res["fit_coeff"][0] > 0
    assert res["peak_utilisation"] > 0.9
    assert res["atlas_utilisation"] > 0.9


def test_quadratic_fit_chebyshev(peaks, detector_config, wavelengths_quadratic):
    atlas = Atlas(
        source="manual",
        wavelengths=wavelengths_quadratic,
        min_wavelength=3000.0,
        max_wavelength=8000.0,
        element="quadratic",
    )

    config = {
        "seed": 0,
        "detector": detector_config,
        "hough": {
            "num_slopes": 500,
            "range_tolerance": 200.0,
            "xbins": 100,
            "ybins": 100,
            "linearity_tolerance": 100.0,
        },
        "ransac": {
            "sample_size": 4,
            "top_n_candidate": 10,
            "minimum_matches": 10,
            "minimum_fit_error": 1e-25,
            "max_tries": 500,
            "rms_tolerance": 5.0,
            "inlier_tolerance": 2.0,
            "degree": 2,
            "fit_type": "chebyshev",
        },
    }

    # Initialise the calibrator
    c = Calibrator(peaks, atlas_lines=atlas.atlas_lines, config=config)

    c.do_hough_transform(brute_force=False)
    c._get_candidate_points_linear(c.config.hough.linearity_tolerance)
    all_cand_peaks = np.concatenate([cand[0] for cand in c.candidates])
    all_cand_waves = np.concatenate([cand[1] for cand in c.candidates])
    n_recalled = sum(
        np.any(
            np.isclose(all_cand_peaks, p, atol=1e-6)
            & np.isclose(all_cand_waves, wavelengths_quadratic[i], atol=1e-6)
        )
        for i, p in enumerate(peaks)
    )
    candidate_recall = n_recalled / len(peaks)
    assert candidate_recall > 0.9
    # Run the wavelength calibration
    res = c.fit()

    assert res["fit_coeff"][0] > 0
    assert res["peak_utilisation"] > 0.9
    assert res["atlas_utilisation"] > 0.9
