import os

import numpy as np
from astropy.io import fits
from scipy.signal import find_peaks

from rascal.atlas import Atlas
from rascal.calibrator import Calibrator
from rascal import util

HERE = os.path.dirname(os.path.realpath(__file__))


def test_sprat_manual_atlas():

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
    peaks, _ = find_peaks(spectrum, height=300, distance=5, threshold=None)
    peaks = util.refine_peaks(spectrum, peaks, window_width=5)

    # Initialise the calibrator
    c = Calibrator(peaks, spectrum=spectrum)
    a = Atlas()

    c.use_plotly()
    assert c.which_plotting_library() == "plotly"

    # auto filename
    c.plot_arc(display=False, fig_type="png+html", save_fig=True)
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

    c.set_hough_properties(
        num_slopes=5000,
        range_tolerance=500.0,
        xbins=100,
        ybins=100,
        min_wavelength=3500.0,
        max_wavelength=8000.0,
    )
    # blend: 4829.71, 4844.33
    # blend: 5566.62, 5581.88
    # blend: 6261.212, 6265.302
    # blend: 6872.11, 6882.16
    # blend: 7283.961, 7285.301
    # blend: 7316.272, 7321.452
    atlas = [
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
    element = ["Xe"] * len(atlas)

    a.add_user_atlas(
        element,
        atlas,
        pressure=pressure,
        temperature=temperature,
        relative_humidity=relative_humidity,
    )
    c.set_atlas(a, constrain_poly=True)

    c.set_ransac_properties(
        sample_size=5, top_n_candidate=5, filter_close=True
    )

    c.do_hough_transform(brute_force=True)

    # Run the wavelength calibration
    best_p, x, y, rms, residual, peak_utilisation, atlas_utilisation = c.fit(
        max_tries=250
    )

    # Plot the solution
    c.plot_fit(
        best_p,
        spectrum,
        plot_atlas=True,
        log_spectrum=False,
        display=False,
        save_fig=True,
        fig_type="png+html",
        tolerance=5.0,
        filename=os.path.join(HERE, "test_output", "test_lt_sprat_fit_plotly"),
    )

    (
        fit_coeff_new,
        peak_matched,
        atlas_matched,
        rms,
        residual,
        peak_utilisation,
        atlas_utilisation,
    ) = c.match_peaks(best_p)

    c.plot_fit(
        fit_coeff_new,
        spectrum,
        plot_atlas=True,
        log_spectrum=False,
        display=False,
        fig_type="png+html",
        save_fig=True,
        filename=os.path.join(
            HERE, "test_output", "test_lt_sprat_fit_matplotlib"
        ),
        tolerance=5.0,
    )

    # Show the parameter space for searching possible solution
    c.plot_search_space(
        display=False,
        fig_type="png+html",
        save_fig=True,
        filename=os.path.join(
            HERE, "test_output", "test_lt_sprat_search_space_matplotlib"
        ),
    )

    print("Stdev error: {} A".format(residual.std()))
    print("Peaks utilisation rate: {}%".format(peak_utilisation * 100))
    print("Atlas utilisation rate: {}%".format(atlas_utilisation * 100))

    c.use_matplotlib()
    assert c.which_plotting_library() == "matplotlib"

    c.plot_arc(
        display=False,
        save_fig=True,
        filename=os.path.join(HERE, "test_output", "test_lt_sprat_arc_plotly"),
    )
    c.plot_arc(
        log_spectrum=True,
        display=False,
        save_fig=True,
        fig_type="png+html",
        filename=os.path.join(
            HERE, "test_output", "test_lt_sprat_arc_log_plotly"
        ),
    )

    # Plot the solution
    c.plot_fit(
        best_p,
        spectrum,
        plot_atlas=True,
        log_spectrum=False,
        tolerance=5.0,
        display=False,
        save_fig=True,
        fig_type="png+html",
        filename=os.path.join(HERE, "test_output", "test_lt_sprat_fit_plotly"),
    )

    # Plot the solution
    c.plot_fit(
        best_p,
        spectrum,
        plot_atlas=True,
        log_spectrum=False,
        tolerance=5.0,
        display=False,
        save_fig=True,
        fig_type="png+html",
        filename=os.path.join(HERE, "test_output", "test_lt_sprat_fit_plotly"),
    )

    # Show the parameter space for searching possible solution
    c.plot_search_space(
        save_fig=True,
        fig_type="png+html",
        display=False,
        filename=os.path.join(HERE, "test_output"),
    )

    c.plot_arc(
        save_fig=True,
        fig_type="png+html",
        display=False,
        filename=os.path.join(HERE, "test_output", "test_lt_sprat_arc_plotly"),
    )

    c.plot_arc(
        save_fig=True,
        fig_type="png+html",
        display=False,
        return_jsonstring=True,
    )

    # Plot the solution
    c.plot_fit(
        best_p,
        spectrum,
        plot_atlas=True,
        log_spectrum=True,
        tolerance=5.0,
        save_fig=True,
        fig_type="png+html",
        display=False,
        filename=os.path.join(
            HERE, "test_output", "test_lt_sprat_fit_log_plotly"
        ),
        return_jsonstring=True,
    )
    # Plot the solution
    c.plot_fit(
        best_p,
        spectrum,
        plot_atlas=True,
        log_spectrum=True,
        tolerance=5.0,
        save_fig=True,
        fig_type="png+html",
        display=False,
        filename=os.path.join(
            HERE, "test_output", "test_lt_sprat_fit_log_plotly"
        ),
    )

    # Show the parameter space for searching possible solution
    c.plot_search_space(
        save_fig=True,
        fig_type="png+html",
        display=False,
        filename=os.path.join(
            HERE, "test_output", "test_lt_sprat_search_space_plotly"
        ),
        return_jsonstring=True,
    )
