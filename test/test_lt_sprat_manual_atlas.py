import os
from functools import partialmethod

import numpy as np
import pytest
from astropy.io import fits
from rascal import util
from rascal.atlas import Atlas
from rascal.calibrator import Calibrator
from scipy.signal import find_peaks

# Suppress tqdm output
from tqdm import tqdm

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

HERE = os.path.dirname(os.path.realpath(__file__))

# Load the LT SPRAT data
fits_file = fits.open(
    os.path.join(
        HERE,
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
element = ["Xe"] * len(sprat_atlas_lines)

user_atlas = Atlas(
    line_list="manual",
    wavelengths=sprat_atlas_lines,
    min_wavelength=3500.0,
    max_wavelength=8000.0,
    range_tolerance=500.0,
    elements=element,
    pressure=pressure,
    temperature=temperature,
    relative_humidity=relative_humidity,
)

config = {
    "data": {"contiguous_range": None},
    "hough": {
        "num_slopes": 2000,
        "range_tolerance": 200.0,
        "xbins": 100,
        "ybins": 100,
    },
    "ransac": {
        "sample_size": 5,
        "top_n_candidate": 5,
        "filter_close": True,
    },
}


# Initialise the calibrator
c = Calibrator(
    peaks, atlas_lines=user_atlas.atlas_lines, config=config, spectrum=spectrum
)

c.do_hough_transform(brute_force=True)


def test_plot_arc():

    # auto filename
    c.plot_arc(display=False, fig_type="png+html", save_fig=True)
    os.remove("rascal_arc.png")
    # user provided filename
    c.plot_arc(
        display=False,
        log_spectrum=True,
        fig_type="png+html",
        save_fig=True,
        filename=os.path.join(
            HERE, "test_output", "test_lt_sprat_arc_matplotlib"
        ),
    )

    c.use_plotly()
    assert c.which_plotting_library() == "plotly"

    # auto filename
    c.plot_arc(display=False, fig_type="png+html", save_fig=True)
    os.remove("rascal_arc.png")
    # user provided filename
    c.plot_arc(
        display=False,
        log_spectrum=True,
        fig_type="png+html",
        save_fig=True,
        filename=os.path.join(HERE, "test_output", "test_lt_sprat_arc_plotly"),
    )

    # Not taking the log of the spectrum
    c.plot_arc(
        display=False,
        save_fig=True,
        filename=os.path.join(
            HERE, "test_output", "test_lt_sprat_arc_matplotlib"
        ),
    )

    c.plot_arc(
        display=False,
        save_fig=True,
        fig_type="png+html",
        filename=os.path.join(
            HERE, "test_output", "test_lt_sprat_arc_log_matplotlib"
        ),
    )


def test_sprat_manual_atlas_fit_match_peaks_and_create_summary():

    # Run the wavelength calibration
    res = c.fit(max_tries=5000, candidate_tolerance=5.0)
    assert res["success"]

    # Plot the solution
    c.plot_fit(
        res["fit_coeff"],
        spectrum,
        plot_atlas=True,
        log_spectrum=False,
        display=False,
        save_fig=True,
        fig_type="png+html",
        filename=os.path.join(HERE, "test_output", "test_lt_sprat_fit_plotly"),
    )

    res = c.match_peaks(res["fit_coeff"])

    c.plot_fit(
        res["fit_coeff"],
        spectrum,
        plot_atlas=True,
        log_spectrum=False,
        display=False,
        fig_type="png+html",
        save_fig=True,
        filename=os.path.join(HERE, "test_output", "test_lt_sprat_fit_plotly"),
    )

    # repeat everything with matplotlib
    c.use_matplotlib()
    assert c.which_plotting_library() == "matplotlib"

    # Plot the solution
    c.plot_fit(
        res["fit_coeff"],
        spectrum,
        plot_atlas=True,
        log_spectrum=False,
        display=False,
        save_fig=True,
        fig_type="png+html",
        filename=os.path.join(
            HERE, "test_output", "test_lt_sprat_fit_matplotlib"
        ),
    )

    # Plot the solution and return json
    c.plot_fit(
        res["fit_coeff"],
        spectrum,
        plot_atlas=True,
        log_spectrum=True,
        save_fig=True,
        fig_type="png+html",
        display=False,
        filename=os.path.join(
            HERE, "test_output", "test_lt_sprat_fit_log_matplotlib"
        ),
        return_jsonstring=True,
    )

    c.summary()

    # save with a filename provided
    c.save_summary(
        filename=os.path.join(HERE, "test_output", "test_lt_sprat_summary.txt")
    )

    # save without providing a filename
    out_path = c.save_summary()
    os.remove(out_path)


@pytest.mark.xfail()
def test_plot_hough_soace():
    # Show the parameter space for searching possible solution
    c.plot_search_space(
        display=False,
        fig_type="png+html",
        save_fig=True,
        filename=os.path.join(
            HERE, "test_output", "test_lt_sprat_search_space_plotly"
        ),
    )

    # repeat everything with matplotlib
    c.use_matplotlib()
    assert c.which_plotting_library() == "matplotlib"

    # Show the parameter space for searching possible solution
    c.plot_search_space(
        save_fig=True,
        fig_type="png+html",
        display=False,
        filename=os.path.join(
            HERE, "test_output", "test_lt_sprat_search_space_matplotlib"
        ),
    )

    # Show the parameter space for searching possible solution and return json
    c.plot_search_space(
        save_fig=True,
        fig_type="png+html",
        display=False,
        filename=os.path.join(
            HERE, "test_output", "test_lt_sprat_search_space_matplotlib"
        ),
        return_jsonstring=True,
    )
