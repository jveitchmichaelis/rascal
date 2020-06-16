import numpy as np
import os
from astropy.io import fits
from scipy.signal import find_peaks
from matplotlib import pyplot as plt

from rascal.calibrator import Calibrator
from rascal import models
from rascal import util

# Load the LT SPRAT data
base_dir = os.path.dirname(__file__)
spectrum2D = fits.open(
    os.path.join(base_dir,
                 'data_int_ids/int20180101_01355922.fits.fz'))[1].data

# Collapse into 1D spectrum between row 110 and 120
spectrum = np.flip(spectrum2D.mean(1), 0)
'''
plt.figure()
plt.plot(spectrum / spectrum.max())
plt.title('Number of pixels: ' + str(spectrum.shape[0]))
plt.xlabel("Pixel (Spectral Direction)")
plt.ylabel("Normalised Count")
plt.xlim(0, len(spectrum))
plt.grid()
plt.tight_layout()
'''
# Identify the peaks
peaks, _ = find_peaks(spectrum, prominence=15, distance=5, threshold=None)
peaks = util.refine_peaks(spectrum, peaks, window_width=5)

# Initialise the calibrator
c = Calibrator(peaks,
               num_pix=len(spectrum),
               min_wavelength=2500.,
               max_wavelength=4600.)
c.set_fit_constraints(num_slopes=5000,
                      range_tolerance=1000.,
                      polydeg=4,
                      xbins=100,
                      ybins=100)
c.add_atlas(elements=['CuNeAr_high'])
'''
# Show the parameter space for searching possible solution
c.plot_search_space()
'''
# Run the wavelength calibration
best_p, rms, residual, peak_utilisation = c.fit(max_tries=2000)

# Refine solution
# First set is to refine only the 0th and 1st coefficient (i.e. the 2 lowest orders)
best_p, x_fit, y_fit, residual, peak_utilisation = c.match_peaks(
    best_p,
    delta=best_p[:1] * 0.001,
    tolerance=10.,
    convergence=1e-10,
    method='Nelder-Mead',
    robust_refit=True)
# Second set is to refine all the coefficients
best_p, x_fit, y_fit, residual, peak_utilisation = c.match_peaks(
    best_p,
    delta=best_p * 0.001,
    tolerance=10.,
    convergence=1e-10,
    method='Nelder-Mead',
    robust_refit=True)

# Plot the solution
c.plot_fit(spectrum, best_p, plot_atlas=True, log_spectrum=False, tolerance=3)

fit_diff = c.polyval(x_fit, best_p) - y_fit
rms = np.sqrt(np.sum(fit_diff**2 / len(x_fit)))

print("Stdev error: {} A".format(fit_diff.std()))
print("Peaks utilisation rate: {}%".format(peak_utilisation * 100))
