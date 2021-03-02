import numpy as np
from astropy.io import fits
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import os
from scipy import interpolate

from rascal.calibrator import Calibrator
from rascal import models
from rascal import util

import sys

sys.path.append('../../bhtomspec/GMOS')

from gmos_fieldflattening import create_pixel_array

pixels = create_pixel_array('north', 2)
rawpix_to_pix_itp = interpolate.interp1d(np.arange(len(pixels)), pixels)

# Load the LT SPRAT data
base_dir = os.path.dirname(__file__)
spectrum2D = fits.open(
    os.path.join(base_dir,
                 'data_gemini_gmos/N20181115S0215_flattened.fits'))[0].data

# Collapse into 1D spectrum between row 110 and 120
spectrum = np.median(spectrum2D[300:310], axis=0)[::-1]

# Identify the peaks
peaks, _ = find_peaks(spectrum, height=500, distance=5, threshold=None)
peaks = util.refine_peaks(spectrum, peaks, window_width=3)

peaks_shifted = rawpix_to_pix_itp(peaks)

# Initialise the calibrator
c = Calibrator(peaks_shifted, spectrum=spectrum)
c.set_calibrator_properties(pixel_list=pixels)
c.plot_arc()
c.set_hough_properties(num_slopes=5000,
                       range_tolerance=500.,
                       xbins=200,
                       ybins=200,
                       min_wavelength=5000.,
                       max_wavelength=9500.)
c.set_ransac_properties(sample_size=8, top_n_candidate=10)
# Vacuum wavelengths
c.add_atlas(elements=['Cu', 'Ar'],
            min_intensity=50,
            min_distance=5,
            pressure=70000.,
            temperature=280.)

c.do_hough_transform()

# Run the wavelength calibration
best_p, rms, residual, peak_utilisation = c.fit(max_tries=1000, fit_deg=4)

# Plot the solution
c.plot_fit(best_p, spectrum, plot_atlas=True, log_spectrum=False, tolerance=5.)

# Show the parameter space for searching possible solution
c.plot_search_space()

print("Stdev error: {} A".format(residual.std()))
print("Peaks utilisation rate: {}%".format(peak_utilisation * 100))
