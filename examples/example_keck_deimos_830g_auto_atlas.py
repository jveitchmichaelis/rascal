import json
import os

import numpy as np
from scipy.signal import find_peaks

from rascal.atlas import AtlasCollection
from rascal.calibrator import Calibrator
from rascal.util import refine_peaks

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# Load the 1D Spectrum from Pypeit
with open(
    os.path.join(
        base_dir, "data_keck_deimos", "keck_deimos_830g_l_PYPIT.json"
    )
) as f:
    spectrum_json = json.load(f)
spectrum = np.array(spectrum_json["spec"])

# Identify the arc lines
peaks, _ = find_peaks(spectrum, prominence=200, distance=10)
peaks = refine_peaks(spectrum, peaks, window_width=3)

atlas = AtlasCollection(
    atlases=[
        {
            "element": el,
            "min_intensity": 1000.0,
            "pressure": 70000.0,
            "temperature": 285.0,
            "range_tolerance": 500.0,
            "min_wavelength": 6500.0,
            "max_wavelength": 10500.0,
        }
        for el in ["Ne", "Ar", "Kr"]
    ]
)

config = {
    "detector": {
        "detector_min_wave": 6500.0,
        "detector_max_wave": 10500.0,
    },
    "hough": {
        "num_slopes": 5000,
        "range_tolerance": 500.0,
        "linearity_tolerance": 50.0,
        "xbins": 500,
        "ybins": 500,
    },
    "ransac": {
        "sample_size": 5,
        "top_n_candidate": 5,
        "linear": True,
        "filter_close": True,
        "rms_tolerance": 5.0,
        "inlier_tolerance": 5.0,
        "candidate_weighted": True,
        "hough_weight": 1.0,
        "minimum_matches": 10,
        "max_tries": 1000,
    },
}

c = Calibrator(
    peaks, atlas_lines=atlas.atlas_lines, config=config, spectrum=spectrum
)
c.plot_arc(
    display=False,
    save_fig=True,
    filename=os.path.join(output_dir, "keck-deimos-auto-atlas-arc-spectrum"),
)

c.do_hough_transform()

# Run the wavelength calibration
res = c.fit()

if res["success"]:
    # Plot the solution
    c.plot_fit(
        res["fit_coeff"],
        spectrum=spectrum,
        display=False,
        plot_atlas=True,
        log_spectrum=False,
        save_fig=True,
        filename=os.path.join(
            output_dir, "keck-deimos-auto-atlas-wavelength-calibration"
        ),
    )

    # Show the parameter space for searching possible solution
    print("RMS: {}".format(res["rms"]))
    print("Stdev error: {} A".format(res["residual"].std()))
    print("Peaks utilisation rate: {}%".format(res["peak_utilisation"] * 100))
    print("Atlas utilisation rate: {}%".format(res["atlas_utilisation"] * 100))

    c.plot_search_space(
        display=False,
        save_fig=True,
        filename=os.path.join(
            output_dir, "keck-deimos-auto-atlas-search-space"
        ),
    )
else:
    print("Calibration failed to find a valid solution.")
