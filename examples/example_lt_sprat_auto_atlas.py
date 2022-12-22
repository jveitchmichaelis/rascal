import os

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

from rascal import util
from rascal.atlas import Atlas
from rascal.calibrator import Calibrator

# Load the LT SPRAT data
base_dir = os.path.dirname(os.path.abspath(__file__))

fits_file = fits.open(
    os.path.join(base_dir, "data_lt_sprat", "v_a_20190516_57_1_0_1.fits")
)[0]

spectrum2D = fits_file.data

plt.ion()
plt.figure(1, figsize=(10, 4))
plt.imshow(np.log10(spectrum2D), aspect="auto", origin="lower")
plt.xlabel("Spectral Direction / Pix")
plt.ylabel("Spatial Direction / Pix")
plt.tight_layout()
plt.savefig(
    os.path.join(base_dir, "output", "lt-sprat-auto-atlas-arc-image.png")
)


temperature = fits_file.header["REFTEMP"]
pressure = fits_file.header["REFPRES"] * 100.0
relative_humidity = fits_file.header["REFHUMID"]

# Collapse into 1D spectrum between row 110 and 120
spectrum = np.median(spectrum2D[110:120], axis=0)

# Identify the peaks
peaks, _ = find_peaks(spectrum, height=300, prominence=200, distance=5)
peaks = util.refine_peaks(spectrum, peaks, window_width=5)

# Initialise the calibrator
c = Calibrator(peaks, spectrum=spectrum)
c.plot_arc(
    display=False,
    save_fig="png",
    filename=os.path.join(
        base_dir, "output", "lt-sprat-auto-atlas-arc-spectrum"
    ),
)

c.set_hough_properties(
    num_slopes=2000,
    range_tolerance=200.0,
    xbins=100,
    ybins=100,
    min_wavelength=3600.0,
    max_wavelength=8000.0,
)
c.set_ransac_properties(
    sample_size=5, top_n_candidate=5, filter_close=True, minimum_matches=13
)

atlas = Atlas(
    elements=["Xe"],
    min_intensity=50.0,
    min_distance=10,
    min_atlas_wavelength=3600.0,
    max_atlas_wavelength=8000.0,
    range_tolerance=200.0,
    pressure=pressure,
    temperature=temperature,
    relative_humidity=relative_humidity,
)
c.set_atlas(atlas, candidate_tolerance=5.0)

c.do_hough_transform()

# Run the wavelength calibration
res = c.fit(max_tries=2000)

# Plot the solution
c.plot_fit(
    res["fit_coeff"],
    spectrum=spectrum,
    display=False,
    plot_atlas=True,
    log_spectrum=False,
    save_fig="png",
    filename=os.path.join(
        base_dir, "output", "lt-sprat-auto-atlas-wavelength-calibration"
    ),
)

# Show the parameter space for searching possible solution
c.plot_search_space(
    save_fig="png",
    filename=os.path.join(
        base_dir, "output", "lt-sprat-auto-atlas-search-space"
    ),
)

print("RMS: {}".format(res["rms"]))
print("Stdev error: {} A".format(res["residual"].std()))
print("Peaks utilisation rate: {}%".format(res["peak_utilisation"] * 100))
print("Atlas utilisation rate: {}%".format(res["atlas_utilisation"] * 100))
