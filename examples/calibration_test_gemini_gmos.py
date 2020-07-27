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

sys.path.append('../../GMOSfieldflattening')

from gmos_longslit_fieldflattening import create_pixel_array

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
peaks, _ = find_peaks(spectrum, height=1000, distance=5, threshold=None)
peaks = util.refine_peaks(spectrum, peaks, window_width=5)

peaks_shifted = rawpix_to_pix_itp(peaks)

plt.figure(1, figsize=(12, 8))
plt.clf()
plt.plot(pixels, spectrum / spectrum.max())
plt.scatter(peaks_shifted, spectrum[peaks.astype('int')] / spectrum.max())
plt.title('Number of pixels: ' + str(spectrum.shape[0]))
plt.xlabel("Pixel (Spectral Direction)")
plt.ylabel("Normalised Count")
plt.xlim(0, 3072)
plt.grid()
plt.tight_layout()

# Initialise the calibrator
c = Calibrator(peaks_shifted,
               num_pix=len(pixels),
               pixel_list=pixels,
               min_wavelength=5000.,
               max_wavelength=9500.)
c.set_fit_constraints(num_slopes=5000,
                      candidate_thresh=10.,
                      range_tolerance=500.,
                      xbins=200,
                      ybins=200)
c.add_atlas(elements=['Cu', 'Ar'],
            min_intensity=50,
            min_distance=5,
            pressure=70000.,
            temperature=280.)

# Show the parameter space for searching possible solution
c.plot_search_space()

# Run the wavelength calibration
best_p, rms, residual, peak_utilisation = c.fit(max_tries=10000)

# Refine solution
# First set is to refine only the 0th and 1st coefficient (i.e. the 2 lowest orders)
best_p, x_fit, y_fit, residual, peak_utilisation = c.match_peaks(
    best_p,
    n_delta=2,
    tolerance=10.,
    convergence=1e-10,
    method='Nelder-Mead',
    robust_refit=True)
# Second set is to refine all the coefficients
best_p, x_fit, y_fit, residual, peak_utilisation = c.match_peaks(
    best_p,
    tolerance=10.,
    convergence=1e-10,
    method='Nelder-Mead',
    robust_refit=True)

# Plot the solution
c.plot_fit(spectrum, best_p, plot_atlas=True, log_spectrum=False, tolerance=5.)

fit_diff = c.polyval(x_fit, best_p) - y_fit
rms = np.sqrt(np.sum(fit_diff**2 / len(x_fit)))

print("Stdev error: {} A".format(fit_diff.std()))
print("Peaks utilisation rate: {}%".format(peak_utilisation * 100))
