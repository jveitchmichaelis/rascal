import os

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from rascal import util
from rascal.atlas import Atlas
from rascal.calibrator import Calibrator
from scipy import interpolate
from scipy.signal import find_peaks

# All pixel values here are unbinned
# n for north
# s for south
hamamatsu_width = 2048

# chip gaps
hamamatsu_n_gap = 67
hamamatsu_s_gap = 61


def create_pixel_array(northsouth, binning):
    """
    The GMOS longslit spectrum spreads over three CCDs. This function creates
    the pixel list corresponding the binned pixels adjusted for the chip gaps,
    which can lead to a non-integer shift in the number of pixels.
    Parameters
    ----------
    northsout: str
        Indicate whether the Gemini north or south, because the chip gaps are
        different.
    binning : numeric
        The binning factor
    Returns
    -------
    pixels: numpy.array
        The pixel values adjusted for chip gaps for the corresponding pixel.
    """

    binned_width = hamamatsu_width / binning

    if northsouth == "north":
        gap_binned_width = hamamatsu_n_gap / binning
    elif northsouth == "south":
        gap_binned_width = hamamatsu_s_gap / binning
    else:
        raise ValueError('Please choose from "north" or "south".')

    pixels = np.concatenate(
        (
            np.arange(binned_width),
            np.arange(
                binned_width + gap_binned_width,
                binned_width * 2 + gap_binned_width,
            ),
            np.arange(
                binned_width * 2 + gap_binned_width * 2,
                binned_width * 3 + gap_binned_width * 2,
            ),
        )
    )

    return pixels


pixels = create_pixel_array("north", 2)
rawpix_to_pix_itp = interpolate.interp1d(np.arange(len(pixels)), pixels)

# Load the GMOS data
base_dir = os.path.dirname(os.path.abspath(__file__))
spectrum2D = fits.open(
    os.path.join(base_dir, "data_gemini_gmos", "N20181115S0215_flattened.fits")
)[0].data

plt.figure(1, figsize=(10, 4))
plt.imshow(np.log10(spectrum2D), aspect="auto", origin="lower")
plt.xlabel("Spectral Direction / Pix")
plt.ylabel("Spatial Direction / Pix")
plt.tight_layout()
plt.savefig(
    os.path.join(
        base_dir, "output", "gemini-gmosls-manual-atlas-arc-image.png"
    )
)

# Collapse into 1D spectrum between row 300 and 310
spectrum = np.median(spectrum2D[300:310], axis=0)[::-1]

# Identify the peaks
peaks, _ = find_peaks(
    spectrum, height=1000, prominence=500, distance=5, threshold=None
)
peaks = util.refine_peaks(spectrum, peaks, window_width=5)

peaks_shifted = rawpix_to_pix_itp(peaks)

# Initialise the calibrator
c = Calibrator(peaks_shifted, spectrum=spectrum)
c.set_calibrator_properties(effective_pixel=pixels)
c.plot_arc(
    pixels,
    display=False,
    save_fig="png",
    filename=os.path.join(
        base_dir, "output", "gemini-gmosls-manual-atlas-arc-spectrum"
    ),
)
c.set_hough_properties(
    num_slopes=5000,
    range_tolerance=500.0,
    xbins=200,
    ybins=200,
    min_wavelength=5000.0,
    max_wavelength=9500.0,
)
c.set_ransac_properties(sample_size=8, top_n_candidate=10)
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

element = ["CuAr"] * len(atlas_lines)

atlas = Atlas(range_tolerance=500.0)
atlas.add_user_atlas(
    elements=element,
    wavelengths=atlas_lines,
    vacuum=True,
    pressure=61700.0,
    temperature=276.55,
    relative_humidity=4.0,
)
c.set_atlas(atlas)

c.do_hough_transform()

# Run the wavelength calibration
res = c.fit(max_tries=1000, fit_deg=4)

# Plot the solution
c.plot_fit(
    res["fit_coeff"],
    spectrum,
    plot_atlas=True,
    log_spectrum=False,
    display=False,
    save_fig="png",
    filename=os.path.join(
        base_dir, "output", "gemini-gmosls-manual-atlas-wavelength-calibration"
    ),
)

# Show the parameter space for searching possible solution
c.plot_search_space(
    save_fig="png",
    filename=os.path.join(
        base_dir, "output", "gemini-gmosls-manual-atlas-search-space"
    ),
)

print("RMS: {}".format(res["rms"]))
print("Stdev error: {} A".format(res["residual"].std()))
print("Peaks utilisation rate: {}%".format(res["peak_utilisation"] * 100))
print("Atlas utilisation rate: {}%".format(res["atlas_utilisation"] * 100))
