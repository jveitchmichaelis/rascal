import os

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

from rascal import util
from rascal.atlas import AtlasCollection
from rascal.calibrator import Calibrator

# Load the WHT/ISIS data
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)

spectrum2D = fits.open(
    os.path.join(base_dir, "data_wht_isis", "r2701004_red_arc.fit")
)[1].data.T

# Collapse into 1D spectrum between row 500 and 520
spectrum = np.median(spectrum2D[500:520], axis=0)

plt.figure(1, figsize=(10, 4))
plt.imshow(np.log10(spectrum2D), aspect="auto", origin="lower")
plt.xlabel("Spectral Direction / Pix")
plt.ylabel("Spatial Direction / Pix")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "wht-isis-auto-atlas-arc-image.png"))
plt.close()

# Identify the peaks
peaks, _ = find_peaks(
    spectrum, height=500, prominence=100, distance=5, threshold=None
)
peaks = util.refine_peaks(spectrum, peaks, window_width=3)

atlas = AtlasCollection(
    atlases=[
        {
            "element": el,
            "min_wavelength": 6000.0,
            "max_wavelength": 11000.0,
            "min_intensity": 250.0,
            "min_distance": 15.0,
            "range_tolerance": 500.0,
            "vacuum": False,
            "pressure": 101325.0,
            "temperature": 273.15,
            "relative_humidity": 0.0,
        }
        for el in ["Ne", "Ar", "Cu"]
    ]
)

config = {
    "detector": {
        "detector_min_wave": 7000.0,
        "detector_max_wave": 10500.0,
    },
    "hough": {
        "num_slopes": 10000,
        "xbins": 500,
        "ybins": 500,
        "range_tolerance": 500.0,
        "linearity_tolerance": 50.0,
        "constrain_poly": False,
    },
    "ransac": {
        "sample_size": 5,
        "top_n_candidate": 5,
        "filter_close": True,
        "minimum_matches": 20,
        "inlier_tolerance": 5.0,
        "max_tries": 1000,
    },
}

# Initialise the calibrator
c = Calibrator(
    peaks, atlas_lines=atlas.atlas_lines, config=config, spectrum=spectrum
)
c.plot_arc(
    display=False,
    log_spectrum=True,
    save_fig=True,
    filename=os.path.join(output_dir, "wht-isis-auto-atlas-arc-spectrum"),
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
            output_dir, "wht-isis-auto-atlas-wavelength-calibration"
        ),
    )

    # Show the parameter space for searching possible solution
    c.plot_search_space(
        display=False,
        save_fig=True,
        filename=os.path.join(output_dir, "wht-isis-auto-atlas-search-space"),
    )

    print("RMS: {}".format(res["rms"]))
    print("Stdev error: {} A".format(res["residual"].std()))
    print("Peaks utilisation rate: {}%".format(res["peak_utilisation"] * 100))
    print("Atlas utilisation rate: {}%".format(res["atlas_utilisation"] * 100))
else:
    print("Calibration failed to find a valid solution.")
