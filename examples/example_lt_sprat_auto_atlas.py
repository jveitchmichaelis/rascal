import os

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from rascal import util
from rascal.atlas import Atlas
from rascal.calibrator import Calibrator
from scipy.signal import find_peaks

# Load the LT SPRAT data
base_dir = os.path.dirname(os.path.abspath(__file__))

fits_file = fits.open(
    os.path.join(base_dir, "data_lt_sprat", "v_a_20190516_57_1_0_1.fits")
)[0]

spectrum2D = fits_file.data

plt.figure(1, figsize=(10, 4))
plt.imshow(np.log10(spectrum2D), aspect="auto", origin="lower")
plt.xlabel("Spectral Direction / Pix")
plt.ylabel("Spatial Direction / Pix")
plt.tight_layout()
plt.savefig(
    os.path.join(
        base_dir, "output_lt_sprat", "lt-sprat-manual-atlas-arc-image.png"
    )
)

# Collapse into 1D spectrum between row 110 and 120
spectrum = np.median(spectrum2D[110:120], axis=0)

temperature = fits_file.header["REFTEMP"]
pressure = fits_file.header["REFPRES"] * 100.0
relative_humidity = fits_file.header["REFHUMID"]

# Identify the peaks
peaks, _ = find_peaks(
    spectrum, height=200, prominence=100, distance=5, threshold=None
)
peaks = util.refine_peaks(spectrum, peaks, window_width=3)

config = {
    "data": {
        "contiguous_range": None,
        "detector_min_wave": 3500.0,
        "detector_max_wave": 8200.0,
        "detector_edge_tolerance": 200.0,
        "num_pix": 1024,
    },
    "hough": {
        "num_slopes": 2000,
        "range_tolerance": 200.0,
        "xbins": 200,
        "ybins": 200,
    },
    "ransac": {
        "max_tries": 5000,
        "inlier_tolerance": 8.0,
        "sample_size": 8,
        "top_n_candidate": 5,
        "filter_close": False,
    },
}

atlas = Atlas(
    elements=["Xe"],
    min_intensity=50.0,
    min_wavelength=3800.0,
    max_wavelength=8200.0,
    range_tolerance=200.0,
    pressure=pressure,
    temperature=temperature,
    relative_humidity=relative_humidity,
    brightest_n_lines=40,
)

# Initialise the calibrator
c = Calibrator(
    peaks, atlas_lines=atlas.atlas_lines, config=config, spectrum=spectrum
)

# Run the wavelength calibration
res = c.fit()

c.plot_arc(
    display=False,
    save_fig="png",
    filename=os.path.join(
        base_dir, "output_lt_sprat", "lt-sprat-manual-atlas-arc-spectrum"
    ),
)

# Plot the solution
c.plot_fit(
    res["fit_coeff"],
    spectrum=spectrum,
    display=False,
    plot_atlas=True,
    log_spectrum=False,
    save_fig="png",
    filename=os.path.join(
        base_dir,
        "output_lt_sprat",
        "lt-sprat-manual-atlas-wavelength-calibration",
    ),
)

# Show the parameter space for searching possible solution
c.plot_search_space(
    save_fig="png",
    filename=os.path.join(
        base_dir, "output_lt_sprat", "lt-sprat-manual-atlas-search-space"
    ),
)

print("RMS: {}".format(res["rms"]))
print("Stdev error: {} A".format(res["residual"].std()))
print("Peaks utilisation rate: {}%".format(res["peak_utilisation"] * 100))
print("Atlas utilisation rate: {}%".format(res["atlas_utilisation"] * 100))
