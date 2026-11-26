import os

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

from eval_utils import report_calibration_diagnostics
from rascal import util
from rascal.atlas import AtlasCollection
from rascal.calibrator import Calibrator

# All pixel values here are unbinned
# n for north
# s for south
hamamatsu_width = 2048

# chip gaps
hamamatsu_n_gap = 67
hamamatsu_s_gap = 61


def create_contiguous_range(northsouth, binning):
    """
    The GMOS longslit spectrum spreads over three CCDs. This function creates
    the contiguous pixel ranges adjusted for the chip gaps, which can lead to
    a non-integer shift in the number of pixels.
    """
    binned_width = hamamatsu_width / binning

    if northsouth == "north":
        gap_binned_width = hamamatsu_n_gap / binning
    elif northsouth == "south":
        gap_binned_width = hamamatsu_s_gap / binning
    else:
        raise ValueError('Please choose from "north" or "south".')

    return [
        0.0,
        binned_width - 1,
        binned_width + gap_binned_width,
        binned_width * 2 + gap_binned_width - 1,
        binned_width * 2 + gap_binned_width * 2,
        binned_width * 3 + gap_binned_width * 2 - 1,
    ]


# Load the GMOS data
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)

spectrum2D = fits.open(
    os.path.join(base_dir, "data_gemini_gmos", "N20181115S0215_flattened.fits")
)[0].data

plt.figure(1, figsize=(10, 4))
plt.imshow(np.log10(spectrum2D), aspect="auto", origin="lower")
plt.xlabel("Spectral Direction / Pix")
plt.ylabel("Spatial Direction / Pix")
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "gemini-gmosls-auto-atlas-arc-image.png")
)
plt.close()

# Collapse into 1D spectrum between row 300 and 310
spectrum = np.median(spectrum2D[300:310], axis=0)[::-1]

# Identify the peaks
peaks, _ = find_peaks(
    spectrum, height=1000, prominence=500, distance=5, threshold=None
)
peaks = util.refine_peaks(spectrum, peaks, window_width=3)

# Vacuum wavelengths
atlas = AtlasCollection(
    atlases=[
        {
            "element": el,
            "min_intensity": 25.0,
            "min_distance": 5.0,
            "range_tolerance": 500.0,
            "vacuum": False,
            "pressure": 61700.0,
            "temperature": 276.55,
            "relative_humidity": 4.0,
            "min_wavelength": 5000.0,
            "max_wavelength": 9500.0,
        }
        for el in ["Cu", "Ar"]
    ]
)

config = {
    "detector": {
        "contiguous_range": create_contiguous_range("north", 2),
        "detector_min_wave": 5000.0,
        "detector_max_wave": 9500.0,
    },
    "hough": {
        "num_slopes": 5000,
        "range_tolerance": 500.0,
        "xbins": 200,
        "ybins": 200,
    },
    "ransac": {
        "sample_size": 5,
        "top_n_candidate": 10,
        "minimum_matches": 18,
        "max_tries": 5000,
        "degree": 4,
    },
}

# Initialise the calibrator
c = Calibrator(
    peaks, atlas_lines=atlas.atlas_lines, config=config, spectrum=spectrum
)
c.plot_arc(
    display=False,
    save_fig=True,
    filename=os.path.join(output_dir, "gemini-gmosls-auto-atlas-arc-spectrum"),
)

c.do_hough_transform()

# Run the wavelength calibration
res = c.fit()

if res["success"]:
    # Plot the solution
    c.plot_fit(
        res["fit_coeff"],
        spectrum=spectrum,
        plot_atlas=True,
        log_spectrum=False,
        display=False,
        save_fig=True,
        filename=os.path.join(
            output_dir, "gemini-gmosls-auto-atlas-wavelength-calibration"
        ),
    )

    # Show the parameter space for searching possible solution
    c.plot_search_space(
        display=False,
        save_fig=True,
        filename=os.path.join(
            output_dir, "gemini-gmosls-auto-atlas-search-space"
        ),
    )

    print("RMS: {}".format(res["rms"]))
    print("Stdev error: {} A".format(res["residual"].std()))
    print("Peaks utilisation rate: {}%".format(res["peak_utilisation"] * 100))
    print("Atlas utilisation rate: {}%".format(res["atlas_utilisation"] * 100))
else:
    print("Calibration failed to find a valid solution.")

gmos_gt_coeffs = [
    4.66525885e03,
    1.46720775e00,
    1.63000305e-05,
    1.37474397e-10,
    -1.67109123e-13,
]
report_calibration_diagnostics(c, res, gt_coeffs=gmos_gt_coeffs)
