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


# blend: 4829.71, 4844.33
# blend: 5566.62, 5581.88
# blend: 6261.212, 6265.302
# blend: 6872.11, 6882.16
# blend: 7283.961, 7285.301
# blend: 7316.272, 7321.452
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

config = {
    "data": {
        "contiguous_range": None,
        "detector_min_wave": 3500.0,
        "detector_max_wave": 8000.0,
        "detector_edge_tolerance": 200.0,
        "num_pix": 1024,
    },
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
        "max_tries": 2500,
    },
}

atlas = Atlas(
    line_list="manual",
    wavelengths=sprat_atlas_lines,
    min_wavelength=3800.0,
    max_wavelength=8000.0,
    range_tolerance=250.0,
    elements=element,
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
