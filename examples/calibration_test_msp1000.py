import numpy as np
from astropy.io import fits
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm
import os

from rascal.calibrator import Calibrator
from rascal.util import refine_peaks
from rascal import models

# Load the 1D spectrum
base_dir = os.path.dirname(__file__)
spectrum = np.loadtxt(os.path.join(base_dir,'data_msp1000/A620EBA HgCal.mspec'), delimiter=',')[:, 1]

plt.figure()
plt.plot(spectrum / spectrum.max())
plt.title('Number of pixels: ' + str(spectrum.shape[0]))
plt.xlabel("Pixel (Spectral Direction)")
plt.ylabel("Normalised Count")
plt.xlim(0, len(spectrum))
plt.grid()
plt.tight_layout()

# Identify the peaks
peaks, _ = find_peaks(spectrum, prominence=300, distance=15, threshold=None)
peaks_refined = refine_peaks(spectrum, peaks, window_width=3)

# Initialise the calibrator
c = Calibrator(peaks,
               num_pix=len(spectrum),
               min_wavelength=4000.,
               max_wavelength=8750.)

# Ignore bluer Argon lines
c.add_atlas("Hg", include_second_order=True)
c.add_atlas("Ar", min_wavelength=6500)

# Show the parameter space for searching possible solution
c.plot_search_space()

# Run the wavelength calibration
best_p, rms, residual, peak_utilisation = c.fit(max_tries=10000)

# Refine solution
best_p, x_fit, y_fit, residual, peak_utilisation = c.refine_fit(
    best_p,
    delta=best_p * 0.01,
    tolerance=10.,
    convergence=1e-10,
    method='Nelder-Mead',
    robust_refit=True)

# Plot the solution
c.plot_fit(spectrum, best_p, plot_atlas=True, log_spectrum=False, tolerance=5)

fit_diff = c.polyval(x_fit, best_p) - y_fit
rms = np.sqrt(np.sum(fit_diff**2 / len(x_fit)))

print("Stdev error: {} A".format(fit_diff.std()))
print("Peaks utilisation rate: {}%".format(peak_utilisation * 100))
