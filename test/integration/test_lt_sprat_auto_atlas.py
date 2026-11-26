import logging
import os
from functools import partialmethod

import numpy as np
import pytest
from astropy.io import fits
from scipy.signal import find_peaks

# Suppress tqdm output
from tqdm import tqdm

from rascal import util
from rascal.atlas import AtlasCollection, NistSource
from rascal.calibrator import Calibrator

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

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

config = {
    "seed": 0,
    "detector": {
        "contiguous_range": None,
        "detector_min_wave": 3500.0,
        "detector_max_wave": 8200.0,
        "detector_edge_tolerance": 100.0,
        "num_pix": 1024,
    },
    "hough": {
        "num_slopes": 2000,
        "range_tolerance": 200.0,
        "xbins": 100,
        "ybins": 100,
    },
    "ransac": {
        "max_tries": 5000,
        "inlier_tolerance": 5.0,
        "sample_size": 5,
        "top_n_candidate": 5,
        "filter_close": True,
        "degree": 4,
    },
    "atlases": [
        {
            "element": "Xe I",
            "nist_source": NistSource.NIST_ALL,
            "use_accurate_lines": False,
            "min_wavelength": 3500,
            "max_wavelength": 8200,
            "min_intensity": 50.0,
            "min_distance": 10.0,
            "range_tolerance": 100,
            "brightest_n_lines": 40,
        }
    ],
}

# Load the LT SPRAT data
base_dir = os.path.dirname(os.path.abspath(__file__))
spectrum2D = fits.open(
    os.path.join(
        base_dir,
        "..",
        "..",
        "examples",
        "data_lt_sprat",
        "v_a_20190516_57_1_0_1.fits",
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
    atlas_wavelengths = np.array(auto_atlas.wavelengths)
    assert len(atlas_wavelengths) > 0

    matched = sum(
        np.isclose(line, atlas_wavelengths, atol=1.0).any()
        for line in sprat_atlas_lines
    )
    assert matched >= 10


def test_sprat_calibration(auto_atlas):
    cal_config = {k: v for k, v in config.items() if k != "atlases"}
    c = Calibrator(
        peaks,
        atlas_lines=auto_atlas.atlas_lines,
        config=cal_config,
        spectrum=spectrum,
    )

    res = c.fit()
    assert res["success"]

    res = c.match_peaks(res["fit_coeff"], refine=False, robust_refit=True)
    assert len(res["matched_peaks"]) > 5
