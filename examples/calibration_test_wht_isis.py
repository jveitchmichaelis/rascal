import numpy as np
from astropy.io import fits
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import os

from rascal.calibrator import Calibrator
from rascal import models
from rascal import util

# Load the LT SPRAT data
base_dir = os.path.dirname(__file__)
spectrum2D = fits.open(
    os.path.join(base_dir, 'data_wht_isis/r2701004_red_arc.fit'))[1].data.T

# Collapse into 1D spectrum between row 110 and 120
spectrum = np.median(spectrum2D[500:520], axis=0)

plt.figure()
plt.plot(spectrum / spectrum.max())
plt.title('Number of pixels: ' + str(spectrum.shape[0]))
plt.xlabel("Pixel (Spectral Direction)")
plt.ylabel("Normalised Count")
plt.xlim(0, len(spectrum))
plt.grid()
plt.tight_layout()

# Identify the peaks
peaks, _ = find_peaks(spectrum, height=1500, distance=10, threshold=None)
peaks = util.refine_peaks(spectrum, peaks, window_width=5)

# Initialise the calibrator
c = Calibrator(peaks,
               num_pix=len(spectrum),
               min_wavelength=6500.,
               max_wavelength=10500.)
c.set_fit_constraints(num_slopes=5000,
                      range_tolerance=1000.,
                      xbins=200,
                      ybins=200)
c.add_atlas(elements=['Cu', 'Ne', 'Ar'],
            min_intensity=20,
            pressure=90000.,
            temperature=285.)

# Show the parameter space for searching possible solution
#c.plot_search_space()

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
