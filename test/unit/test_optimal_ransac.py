import numpy as np
import pytest

from rascal.atlas import Atlas
from rascal.calibrator import Calibrator
from rascal.optimal_ransac import OptimalRansacSolver
from rascal.ransac import RansacSolver


def test_optimal_ransac_clean_polynomial():
    np.random.seed(0)
    x = np.linspace(0, 1000, 25)
    y = 3000.0 + 4.0 * x + 1e-3 * x**2

    solver = OptimalRansacSolver(
        x,
        y,
        config={"degree": 2, "sample_size": 4, "max_tries": 200, "rms_tolerance": 2.0},
    )
    assert solver.solve()
    assert len(solver.best_result.x) == len(x)
    assert np.allclose(solver.best_result.fit_coeffs, [3000.0, 4.0, 1e-3], rtol=1e-3)


def test_optimal_ransac_with_outliers_and_weights():
    np.random.seed(0)
    x_true = np.linspace(0, 1000, 20)
    y_true = 3000.0 + 5.0 * x_true

    # Add 2 decoy wavelengths per peak with low weight
    x_all = []
    y_all = []
    w_all = []
    for xi, yi in zip(x_true, y_true):
        x_all.extend([xi, xi, xi])
        y_all.extend([yi, yi + 120.0, yi - 95.0])
        w_all.extend([50.0, 1.0, 1.0])

    solver = OptimalRansacSolver(
        np.array(x_all),
        np.array(y_all),
        config={"degree": 1, "sample_size": 3, "max_tries": 200, "rms_tolerance": 2.0},
        weights=np.array(w_all),
    )
    assert solver.solve()
    assert len(solver.best_result.x) >= 18
    assert np.allclose(solver.best_result.fit_coeffs, [3000.0, 5.0], rtol=1e-2)


def test_optimal_ransac_with_fixed_pairs():
    np.random.seed(0)
    x = np.arange(2, 14, dtype=float)
    y = 3000.0 + 5.0 * x
    solver = OptimalRansacSolver(
        x,
        y,
        config={"degree": 1, "sample_size": 2, "max_tries": 50},
        fixed_x=[0.0, 1.0],
        fixed_y=[3000.0, 3005.0],
    )
    assert solver.solve()
    assert np.allclose(solver.best_result.fit_coeffs, [3000.0, 5.0], atol=1e-2)


def test_calibrator_default_solver_is_ransac_and_supports_optimal(
    peaks, detector_config, wavelengths_linear
):
    atlas = Atlas(
        source="manual",
        element="linear",
        wavelengths=wavelengths_linear,
        min_wavelength=3000.0,
        max_wavelength=8000.0,
    )
    base_cfg = {
        "seed": 0,
        "detector": detector_config,
        "hough": {"num_slopes": 500, "xbins": 50, "ybins": 50},
        "ransac": {
            "degree": 1,
            "sample_size": 3,
            "max_tries": 100,
            "minimum_fit_error": 1e-25,
        },
    }
    c_default = Calibrator(peaks, atlas_lines=atlas.atlas_lines, config=base_cfg)
    assert c_default.config.ransac.solver == "ransac"

    opt_cfg = {
        **base_cfg,
        "ransac": {**base_cfg["ransac"], "solver": "optimal"},
    }
    c_opt = Calibrator(peaks, atlas_lines=atlas.atlas_lines, config=opt_cfg)
    res = c_opt.fit()
    assert res["success"]
    assert res["peak_utilisation"] > 0.9


def test_calibrator_unknown_solver_raises(
    peaks, detector_config, wavelengths_linear
):
    atlas = Atlas(
        source="manual",
        element="linear",
        wavelengths=wavelengths_linear,
        min_wavelength=3000.0,
        max_wavelength=8000.0,
    )
    cfg = {
        "seed": 0,
        "detector": detector_config,
        "hough": {"num_slopes": 500, "xbins": 50, "ybins": 50},
        "ransac": {"degree": 1, "sample_size": 3, "max_tries": 20, "solver": "invalid"},
    }
    c = Calibrator(peaks, atlas_lines=atlas.atlas_lines, config=cfg)
    with pytest.raises(ValueError, match="Unknown RANSAC solver"):
        c.fit()
