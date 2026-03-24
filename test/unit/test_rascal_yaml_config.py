import importlib.resources
import os
from functools import partialmethod

import numpy as np
import pytest
import yaml

# Suppress tqdm output
from tqdm import tqdm

from rascal import calibrator
from rascal.atlas import Atlas

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

base_dir = os.path.dirname(__file__)

_user_atlas = Atlas(
    element="Test",
    source="manual",
    wavelengths=np.linspace(3000, 8000, 20),
    min_wavelength=3000.0,
    max_wavelength=8000.0,
)

_peaks = np.linspace(0, 500, 20)

_config = {
    "detector": {
        "detector_min_wave": 3000.0,
        "detector_max_wave": 8000.0,
    }
}


def test_save_rascal_config(tmp_path):
    c = calibrator.Calibrator(
        _peaks, _user_atlas.atlas_lines, config=_config
    )
    c.save_config(str(tmp_path / "subdir" / "test_rascal_config.yaml"))
    assert (tmp_path / "subdir" / "test_rascal_config.yaml").exists()


def test_load_rascal_config_expect_fail_ytype():
    c = calibrator.Calibrator(
        _peaks, _user_atlas.atlas_lines, config=_config
    )
    with pytest.raises(ValueError):
        c.load_config(np.arange(100), y_type="bla")
