from unittest.mock import patch

import numpy as np
import pytest

from rascal.atlas import Atlas
from rascal.calibrator import Calibrator
from rascal.ransac import RansacSolver, SolveResult


def test_constrain_poly_filters_pairs():
    peaks = np.array([100.0, 500.0, 900.0])
    wavelengths = np.array([3500.0, 5500.0, 7500.0])
    atlas = Atlas(
        source="manual",
        element="Test",
        wavelengths=wavelengths,
        min_wavelength=3000.0,
        max_wavelength=8000.0,
    )
    config = {
        "detector": {
            "num_pix": 1000,
            "detector_min_wave": 3000.0,
            "detector_max_wave": 8000.0,
        },
        "hough": {
            "constrain_poly": True,
            "range_tolerance": 200.0,
            "linearity_tolerance": 100.0,
        },
    }
    c = Calibrator(peaks, atlas_lines=atlas.atlas_lines, config=config)

    # All 9 pairs exist unconstrained; constrained should keep the 3 diagonal
    # pairs and reject extreme off-diagonal pairs like (100, 7500) and (900, 3500)
    assert len(c.pairs) < len(peaks) * len(wavelengths)
    assert len(c.pairs) >= 3
    for p, w in zip(peaks, wavelengths):
        assert any(
            np.isclose(pair[0], p) and np.isclose(pair[1], w)
            for pair in c.pairs
        )


def test_generate_pairs_uses_effective_pixels():
    # Two CCD chips of 500 px each with a 50 px gap between them:
    # chip 1: 0..499, chip 2: 550..1049
    contiguous_range = [0.0, 499.0, 550.0, 1049.0]
    peaks = np.array([100.0, 600.0])  # raw pixel 600 -> effective pixel 650
    wavelengths = np.array([4000.0, 6000.0])
    atlas = Atlas(
        source="manual",
        element="Test",
        wavelengths=wavelengths,
        min_wavelength=3000.0,
        max_wavelength=8000.0,
    )
    config = {
        "detector": {
            "num_pix": 1000,
            "contiguous_range": contiguous_range,
            "detector_min_wave": 3000.0,
            "detector_max_wave": 8000.0,
        },
    }
    c = Calibrator(peaks, atlas_lines=atlas.atlas_lines, config=config)

    pair_pixels = set(np.round(c.pairs[:, 0], 6))
    assert pair_pixels == {100.0, 650.0}


def test_ransac_solver_accepts_fixed_pairs():
    x = np.arange(2, 12, dtype=float)
    y = 3000.0 + 5.0 * x
    fixed_x = np.array([0.0, 1.0])
    fixed_y = np.array([3000.0, 3005.0])

    solver = RansacSolver(
        x,
        y,
        config={"degree": 1, "sample_size": 2, "max_tries": 50},
        fixed_x=fixed_x,
        fixed_y=fixed_y,
    )
    assert solver.solve()
    assert np.allclose(solver.best_result.fit_coeffs, [3000.0, 5.0], atol=1e-2)


def test_calibrator_fit_passes_known_pairs_and_hough(
    peaks, detector_config, wavelengths_linear
):
    atlas = Atlas(
        source="manual",
        element="linear",
        wavelengths=wavelengths_linear,
        min_wavelength=3000.0,
        max_wavelength=8000.0,
    )
    config = {
        "seed": 0,
        "detector": detector_config,
        "hough": {"num_slopes": 500, "xbins": 50, "ybins": 50},
        "ransac": {"degree": 1, "sample_size": 3, "max_tries": 50},
    }
    c = Calibrator(peaks, atlas_lines=atlas.atlas_lines, config=config)
    c.set_known_pairs([peaks[0]], [wavelengths_linear[0]])

    captured = {}
    orig_init = RansacSolver.__init__

    def spy_init(self, *args, **kwargs):
        captured.update(kwargs)
        return orig_init(self, *args, **kwargs)

    with patch.object(RansacSolver, "__init__", spy_init):
        c.fit()

    assert captured.get("hough") is c.hough_transformer
    assert np.array_equal(captured.get("fixed_x"), c.pix_known)
    assert np.array_equal(captured.get("fixed_y"), c.wave_known)


def test_candidate_selection_prioritizes_top_hough_bins(
    peaks, detector_config, wavelengths_linear
):
    # Mix true wavelengths with decoy wavelengths across the detector range
    rng = np.random.default_rng(42)
    decoys = rng.uniform(3100.0, 7900.0, size=30)
    all_waves = np.sort(np.concatenate([wavelengths_linear, decoys]))

    atlas = Atlas(
        source="manual",
        element="linear",
        wavelengths=all_waves,
        min_wavelength=3000.0,
        max_wavelength=8000.0,
    )
    config = {
        "seed": 0,
        "detector": detector_config,
        "hough": {
            "num_slopes": 2000,
            "range_tolerance": 300.0,
            "xbins": 100,
            "ybins": 100,
            "linearity_tolerance": 50.0,
        },
        "ransac": {"top_n_candidate": 5},
    }
    c = Calibrator(peaks, atlas_lines=atlas.atlas_lines, config=config)
    c.do_hough_transform()
    c._get_candidate_points_linear(c.config.hough.linearity_tolerance)
    cp, ca = c._get_most_common_candidates(
        c.candidates, top_n_candidate=1, weighted=True
    )

    recalled = sum(
        np.any(
            np.isclose(cp, p, atol=1e-4)
            & np.isclose(ca, w, atol=1e-4)
        )
        for p, w in zip(peaks, wavelengths_linear)
    )
    assert recalled / len(peaks) > 0.8


def test_fit_valid_uses_max_contiguous_pixel_for_chip_gaps():
    # Two 500-px chips with a 100-px gap -> max(contiguous_pixel) = 1099
    # Linear dispersion: w(x) = 3000 + 5 * x
    # At x=1000 (len(spectrum)): w = 8000; at x=1099 (max contiguous): w = 8495
    # With detector_max_wave=8500 and range_tolerance=200:
    # valid upper range is [8300, 8700] -> x=1099 (8495) is valid, x=1000 (8000) is NOT!
    contiguous_range = [0.0, 499.0, 600.0, 1099.0]
    peaks = np.linspace(50.0, 950.0, 15)
    spectrum = np.zeros(1000)
    atlas = Atlas(
        source="manual",
        element="Test",
        wavelengths=3000.0 + 5.0 * np.linspace(50.0, 1050.0, 15),
        min_wavelength=3000.0,
        max_wavelength=8500.0,
    )
    config = {
        "detector": {
            "num_pix": 1000,
            "contiguous_range": contiguous_range,
            "detector_min_wave": 3000.0,
            "detector_max_wave": 8500.0,
        },
        "hough": {"range_tolerance": 200.0},
        "ransac": {"minimum_matches": 5, "minimum_fit_error": 1e-6},
    }
    c = Calibrator(
        peaks, atlas_lines=atlas.atlas_lines, config=config, spectrum=spectrum
    )

    inliers_x = c.peaks_effective
    inliers_y = 3000.0 + 5.0 * inliers_x
    result = SolveResult(
        fit_coeffs=np.array([3000.0, 5.0]),
        x=inliers_x,
        y=inliers_y,
        residual=np.full(len(inliers_x), 0.1),
        rms_tolerance=4.0,
    )
    assert c._fit_valid(result) is True


def test_msac_cost_prefers_more_inliers_at_rms_tolerance():
    x = np.arange(15, dtype=float)
    y = 3000.0 + 5.0 * x
    solver = RansacSolver(
        x,
        y,
        config={"degree": 1, "rms_tolerance": 4.0, "use_msac": True},
    )

    # Result A: 10 tight inliers + 5 large outliers
    res_a = SolveResult(
        fit_coeffs=np.array([3000.0, 5.0]),
        x=x,
        y=y,
        residual=np.array([0.5] * 10 + [500.0] * 5),
        rms_tolerance=4.0,
    )
    # Result B: only 2 tight inliers + 13 moderate outliers
    res_b = SolveResult(
        fit_coeffs=np.array([3000.0, 5.0]),
        x=x,
        y=y,
        residual=np.array([0.5] * 2 + [20.0] * 13),
        rms_tolerance=4.0,
    )
    assert solver._cost(res_a) < solver._cost(res_b)
