import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.signal import find_peaks

from rascal.atlas import Atlas
from rascal.calibrator import Calibrator
from rascal.util import refine_peaks

atlas_lines = [
    3650.153,
    4046.563,
    4077.831,
    4358.328,
    5460.735,
    5769.598,
    5790.663,
    6682.960,
    6752.834,
    6871.289,
    6965.431,
    7030.251,
    7067.218,
    7147.042,
    7272.936,
    7383.981,
    7503.869,
    7514.652,
    7635.106,
    7723.98,
]

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)

data = fits.open(
    os.path.join(
        base_dir,
        "data_gtc_osiris",
        "0002672523-20200911-OSIRIS-OsirisCalibrationLamp.fits",
    )
)[1]

plt.figure(1, figsize=(10, 4))
plt.imshow(np.log10(data.data.T), aspect="auto", origin="lower")
plt.xlabel("Spectral Direction / Pix")
plt.ylabel("Spatial Direction / Pix")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "gtc-osiris-manual-atlas-arc-image.png"))
plt.close()

spectrum = np.median(data.data.T[550:570], axis=0)

# Identify the arc lines
peaks, _ = find_peaks(
    spectrum, height=1250, prominence=20, distance=3, threshold=None
)
peaks_refined = refine_peaks(spectrum, peaks, window_width=3)

atlas = Atlas(
    source="manual",
    element="HgAr",
    wavelengths=atlas_lines,
    min_wavelength=3500.0,
    max_wavelength=8000.0,
    range_tolerance=500.0,
)

config = {
    "detector": {
        "detector_min_wave": 3500.0,
        "detector_max_wave": 8000.0,
    },
    "hough": {
        "num_slopes": 2000,
        "xbins": 100,
        "ybins": 100,
        "range_tolerance": 500.0,
        "linearity_tolerance": 50.0,
        "constrain_poly": True,
    },
    "ransac": {
        "sample_size": 5,
        "top_n_candidate": 5,
        "linear": True,
        "filter_close": True,
        "rms_tolerance": 10.0,
        "inlier_tolerance": 5.0,
        "candidate_weighted": True,
        "hough_weight": 1.0,
        "minimum_matches": 11,
        "max_tries": 1000,
        "degree": 4,
    },
}

c = Calibrator(
    peaks_refined, atlas_lines=atlas.atlas_lines, config=config, spectrum=spectrum
)

c.plot_arc(
    save_fig=True,
    display=False,
    filename=os.path.join(output_dir, "gtc-osiris-manual-atlas-arc-spectrum"),
)

c.do_hough_transform()

# Run the wavelength calibration
res = c.fit()

if res["success"]:
    # Plot the solution
    c.plot_fit(
        res["fit_coeff"],
        plot_atlas=True,
        log_spectrum=False,
        display=False,
        save_fig=True,
        filename=os.path.join(
            output_dir, "gtc-osiris-manual-atlas-wavelength-calibration"
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
            output_dir, "gtc-osiris-manual-atlas-search-space"
        ),
    )
else:
    print("Calibration failed to find a valid solution.")
