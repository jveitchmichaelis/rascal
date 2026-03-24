import numpy as np
import pytest


NUM_PIX = 1000
LINEAR_GRAD = 5


@pytest.fixture
def peaks():
    rng = np.random.RandomState(0)
    raw = np.sort(rng.random(31) * float(NUM_PIX))
    mask = np.isclose(raw[:-1], raw[1:], atol=5.0)
    mask = np.insert(mask, 0, False)
    return raw[~mask]


@pytest.fixture
def detector_config():
    return {
        "contiguous_range": None,
        "num_pix": NUM_PIX,
        "detector_min_wave": 3000,
        "detector_max_wave": 3000 + LINEAR_GRAD * NUM_PIX,
    }


@pytest.fixture
def wavelengths_linear(peaks):
    return 3000.0 + LINEAR_GRAD * peaks


@pytest.fixture
def wavelengths_quadratic(peaks):
    return 3000.0 + 4 * peaks + 1.0e-3 * peaks**2.0
