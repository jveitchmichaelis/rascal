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
