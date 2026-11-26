import os
from functools import partialmethod

import numpy as np
import pytest
from astropy.io import fits
from scipy.signal import find_peaks

# Suppress tqdm output
from tqdm import tqdm

from rascal import util
from rascal.atlas import Atlas
from rascal.calibrator import Calibrator

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

HERE = os.path.dirname(os.path.realpath(__file__))

# Load the LT SPRAT data
fits_file = fits.open(
    os.path.join(
        HERE,
        "..",
        "..",
        "examples",
        "data_lt_sprat",
        "v_a_20190516_57_1_0_1.fits",
    )
)[0]

spectrum2D = fits_file.data

# Collapse into 1D spectrum between row 110 and 120
spectrum = np.median(spectrum2D[110:120], axis=0)

temperature = fits_file.header["REFTEMP"]
pressure = fits_file.header["REFPRES"] * 100.0
relative_humidity = fits_file.header["REFHUMID"]

# Identify the peaks
peaks, _ = find_peaks(
    spectrum, height=300, prominence=150, distance=5, threshold=None
)
peaks = util.refine_peaks(spectrum, peaks, window_width=3)

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

user_atlas = Atlas(
    source="manual",
    wavelengths=sprat_atlas_lines,
    min_wavelength=3800.0,
    max_wavelength=8200.0,
    range_tolerance=200.0,
    element="Xe",
    pressure=pressure,
    temperature=temperature,
    relative_humidity=relative_humidity,
)

config = {
    "seed": 0,
    "log_level": "INFO",
    "detector": {
        "contiguous_range": None,
        "detector_min_wave": 3500.0,
        "detector_max_wave": 8200.0,
        "detector_edge_tolerance": 100.0,
        "num_pix": 1024,
    },
    "hough": {
        "num_slopes": 2500,
        "range_tolerance": 200.0,
        "xbins": 200,
        "ybins": 200,
    },
    "ransac": {
        "max_tries": 5000,
        "inlier_tolerance": 5.0,
        "sample_size": 5,
        "top_n_candidate": 5,
        "filter_close": True,
    },
}


@pytest.fixture(scope="module")
def calibrator():
    c = Calibrator(
        peaks,
        atlas_lines=user_atlas.atlas_lines,
        config=config,
        spectrum=spectrum,
    )
    c.do_hough_transform()
    return c


def test_plot_arc(calibrator, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calibrator.use_matplotlib()

    # auto filename
    calibrator.plot_arc(display=False, fig_type="png", save_fig=True)
    assert (tmp_path / "rascal_arc.png").exists()

    # user provided filename
    calibrator.plot_arc(
        display=False,
        log_spectrum=True,
        fig_type="png",
        save_fig=True,
        filename=str(tmp_path / "test_lt_sprat_arc_matplotlib"),
    )
    assert (tmp_path / "test_lt_sprat_arc_matplotlib.png").exists()

    calibrator.use_plotly()
    assert calibrator.which_plotting_library() == "plotly"

    # auto filename (iframe HTML export)
    calibrator.plot_arc(display=False, fig_type="iframe", save_fig=True)
    assert (tmp_path / "rascal_arc.iframe").exists()

    # user provided filename
    calibrator.plot_arc(
        display=False,
        log_spectrum=True,
        fig_type="iframe",
        save_fig=True,
        filename=str(tmp_path / "test_lt_sprat_arc_plotly"),
    )
    assert (tmp_path / "test_lt_sprat_arc_plotly.iframe").exists()


def test_sprat_manual_atlas_fit_match_peaks_and_create_summary(
    calibrator, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    # Run the wavelength calibration
    res = calibrator.fit()
    assert res["success"]
    assert res["atlas_utilisation"] >= 0.7

    # Plot with plotly
    calibrator.use_plotly()
    calibrator.plot_fit(
        res["fit_coeff"],
        spectrum,
        plot_atlas=True,
        log_spectrum=False,
        display=False,
        save_fig=True,
        fig_type="iframe",
        filename=str(tmp_path / "test_lt_sprat_fit_plotly"),
    )

    res = calibrator.match_peaks(res["fit_coeff"])
    assert res["atlas_utilisation"] > 0.9

    # repeat with matplotlib
    calibrator.use_matplotlib()
    assert calibrator.which_plotting_library() == "matplotlib"

    calibrator.plot_fit(
        res["fit_coeff"],
        spectrum,
        plot_atlas=True,
        log_spectrum=False,
        display=False,
        save_fig=True,
        fig_type="png",
        filename=str(tmp_path / "test_lt_sprat_fit_matplotlib"),
    )
    assert (tmp_path / "test_lt_sprat_fit_matplotlib.png").exists()

    calibrator.plot_fit(
        res["fit_coeff"],
        spectrum,
        plot_atlas=True,
        log_spectrum=True,
        save_fig=True,
        fig_type="png",
        display=False,
        filename=str(tmp_path / "test_lt_sprat_fit_log_matplotlib"),
        return_jsonstring=True,
    )

    calibrator.summary()

    # save with a filename provided
    summary_path = tmp_path / "test_lt_sprat_summary.txt"
    calibrator.save_summary(filename=str(summary_path))
    assert summary_path.exists()

    # save without providing a filename
    out_path = calibrator.save_summary()
    assert os.path.exists(out_path)


def test_plot_hough_space(calibrator, tmp_path):
    calibrator.use_plotly()
    calibrator.plot_search_space(
        display=False,
        fig_type="iframe",
        save_fig=True,
        filename=str(tmp_path / "test_lt_sprat_search_space_plotly"),
    )
    assert (tmp_path / "test_lt_sprat_search_space_plotly.iframe").exists()

    calibrator.use_matplotlib()
    assert calibrator.which_plotting_library() == "matplotlib"

    calibrator.plot_search_space(
        save_fig=True,
        fig_type="png",
        display=False,
        filename=str(tmp_path / "test_lt_sprat_search_space_matplotlib"),
    )
    assert (tmp_path / "test_lt_sprat_search_space_matplotlib.png").exists()
