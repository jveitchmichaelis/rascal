import os

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

from eval_utils import report_calibration_diagnostics
from rascal import util
from rascal.atlas import Atlas
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
    os.path.join(output_dir, "gemini-gmosls-manual-atlas-arc-image.png")
)
plt.close()

# Collapse into 1D spectrum between row 300 and 310
spectrum = np.median(spectrum2D[300:310], axis=0)[::-1]

# Identify the peaks
peaks, _ = find_peaks(
    spectrum, height=1000, prominence=500, distance=5, threshold=None
)
peaks = util.refine_peaks(spectrum, peaks, window_width=5)

# Vacuum wavelengths
# blend: 5143.21509, 5146.74143
# something weird near there, so not used: 8008.359, 8016.990
atlas_lines = [
    4703.632,
    4728.19041,
    4766.19677,
    4807.36348,
    4849.16386,
    4881.22627,
    4890.40721,
    4906.12088,
    4934.58593,
    4966.46490,
    5018.56194,
    5063.44827,
    5163.723,
    5189.191,
    5497.401,
    5560.246,
    5608.290,
    5913.723,
    6754.698,
    6873.185,
    6967.352,
    7032.190,
    7069.167,
    7149.012,
    7274.940,
    7386.014,
    7505.935,
    7516.721,
    7637.208,
    7725.887,
    7893.246,
    7950.362,
    8105.921,
    8117.542,
    8266.794,
    8410.521,
    8426.963,
    8523.783,
    8670.325,
    9125.471,
    9197.161,
    9227.03,
    9356.787,
    9660.435,
    9787.186,
]

atlas = Atlas(
    source="manual",
    element="CuAr",
    wavelengths=atlas_lines,
    min_wavelength=5000.0,
    max_wavelength=9500.0,
    range_tolerance=500.0,
    input_vacuum=True,
    vacuum=False,
    pressure=61700.0,
    temperature=276.55,
    relative_humidity=4.0,
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
        "sample_size": 8,
        "top_n_candidate": 10,
        "max_tries": 1000,
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
    filename=os.path.join(
        output_dir, "gemini-gmosls-manual-atlas-arc-spectrum"
    ),
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
            output_dir, "gemini-gmosls-manual-atlas-wavelength-calibration"
        ),
    )

    # Show the parameter space for searching possible solution
    c.plot_search_space(
        display=False,
        save_fig=True,
        filename=os.path.join(
            output_dir, "gemini-gmosls-manual-atlas-search-space"
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
