import os
import sys

import numpy as np
from astropy.io import fits
from scipy.signal import find_peaks
from scipy import interpolate

from rascal.calibrator import Calibrator
from rascal.atlas import Atlas
from rascal import util

sys.path.append("../../bhtomspec/GMOS")

from gmos_fieldflattening import create_pixel_array

pixels = create_pixel_array("north", 2)
rawpix_to_pix_itp = interpolate.interp1d(np.arange(len(pixels)), pixels)

# Load the GMOS data
base_dir = os.path.dirname(__file__)
spectrum2D = fits.open(
    os.path.join(base_dir, "data_gemini_gmos/N20181115S0215_flattened.fits")
)[0].data

# Collapse into 1D spectrum between row 300 and 310
spectrum = np.median(spectrum2D[300:310], axis=0)[::-1]

# Identify the peaks
peaks, _ = find_peaks(
    spectrum, height=1000, prominence=500, distance=5, threshold=None
)
peaks = util.refine_peaks(spectrum, peaks, window_width=3)

peaks_shifted = rawpix_to_pix_itp(peaks)

# Initialise the calibrator
c = Calibrator(peaks_shifted, spectrum=spectrum)
c.set_calibrator_properties(pixel_list=pixels)
c.plot_arc(pixels)
c.set_hough_properties(
    num_slopes=5000,
    range_tolerance=500.0,
    xbins=200,
    ybins=200,
    min_wavelength=5000.0,
    max_wavelength=9500.0,
)
c.set_ransac_properties(sample_size=5, top_n_candidate=5)
# Vacuum wavelengths

atlas = Atlas(
    elements=["Cu", "Ar"],
    min_intensity=80,
    min_distance=5,
    range_tolerance=500.0,
    pressure=70000.0,
    temperature=280.0,
)
c.set_atlas(atlas)

c.do_hough_transform()

# Run the wavelength calibration
(
    best_p,
    matched_peaks,
    matched_atlas,
    rms,
    residual,
    peak_utilisation,
    atlas_utilisation,
) = c.fit(max_tries=1000, fit_deg=4)

# Plot the solution
c.plot_fit(
    best_p, spectrum, plot_atlas=True, log_spectrum=False, tolerance=5.0
)

# Show the parameter space for searching possible solution
c.plot_search_space()

print("Stdev error: {} A".format(residual.std()))
print("Peaks utilisation rate: {}%".format(peak_utilisation * 100))
print("Atlas utilisation rate: {}%".format(atlas_utilisation * 100))
