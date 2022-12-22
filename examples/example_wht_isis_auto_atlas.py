import os

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

from rascal.calibrator import Calibrator
from rascal.atlas import Atlas
from rascal import util

# Load the WHT/ISIS data
base_dir = os.path.dirname(os.path.abspath(__file__))

spectrum2D = fits.open(
    os.path.join(base_dir, "data_wht_isis", "r2701004_red_arc.fit")
)[1].data.T

# Collapse into 1D spectrum between row 500 and 520
spectrum = np.median(spectrum2D[500:520], axis=0)

plt.ion()
plt.figure(1, figsize=(10, 4))
plt.imshow(np.log10(spectrum2D), aspect="auto", origin="lower")
plt.xlabel("Spectral Direction / Pix")
plt.ylabel("Spatial Direction / Pix")
plt.tight_layout()
plt.savefig(
    os.path.join(base_dir, "output", "wht-isis-auto-atlas-arc-image.png")
)

# Identify the peaks
peaks, _ = find_peaks(
    spectrum, height=500, prominence=100, distance=5, threshold=None
)
peaks = util.refine_peaks(spectrum, peaks, window_width=3)

# Initialise the calibrator
c = Calibrator(peaks, spectrum=spectrum)
c.plot_arc(
    display=False,
    log_spectrum=True,
    save_fig="png",
    filename=os.path.join(
        base_dir, "output", "wht-isis-auto-atlas-arc-spectrum"
    ),
)
c.set_hough_properties(
    num_slopes=10000,
    xbins=500,
    ybins=500,
    min_wavelength=7000.0,
    max_wavelength=10500.0,
    range_tolerance=500.0,
    linearity_tolerance=50,
)
c.set_ransac_properties(
    sample_size=5, top_n_candidate=5, filter_close=True, minimum_matches=20
)
atlas = Atlas(
    ["Ne", "Ar", "Cu"],
    min_atlas_wavelength=6000,
    max_atlas_wavelength=11000,
    min_intensity=250,
    min_distance=15,
    range_tolerance=500.0,
    vacuum=False,
    pressure=101325.0,
    temperature=273.15,
    relative_humidity=0.0,
)
c.set_atlas(atlas, constrain_poly=False)
c.do_hough_transform()

# Run the wavelength calibration
res = c.fit(max_tries=1000, candidate_tolerance=5)

# Plot the solution
c.plot_fit(
    res["fit_coeffs"],
    spectrum,
    display=False,
    plot_atlas=True,
    log_spectrum=False,
    save_fig="png",
    filename=os.path.join(
        base_dir, "output", "wht-isis-auto-atlas-wavelength-calibration"
    ),
)

# Show the parameter space for searching possible solution
c.plot_search_space(
    save_fig="png",
    filename=os.path.join(
        base_dir, "output", "wht-isis-auto-atlas-search-space"
    ),
)

print("RMS: {}".format(res["rms"]))
print("Stdev error: {} A".format(res["residual"].std()))
print("Peaks utilisation rate: {}%".format(res["peak_utilisation"] * 100))
print("Atlas utilisation rate: {}%".format(res["atlas_utilisation"] * 100))
