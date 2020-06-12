import pkg_resources

from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from rascal.calibrator import Calibrator
from rascal.util import refine_peaks

# Load the 1D Spectrum from Pypeit
data_path = pkg_resources.resource_filename(
    "pypeit", "data/arc_lines/reid_arxiv/keck_deimos_830G.fits")
spectrum = fits.open(data_path)[1].data

flux = spectrum['flux']

# Identify the arc lines
peaks, _ = find_peaks(flux, prominence=1000, distance=10)
refined_peaks = refine_peaks(flux, peaks, window_width=3)

intensity_range = max(flux) - min(flux)

# Plot
fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot("110")
ax.plot(flux)

ax.vlines(refined_peaks,
          0,
          max(flux) + 0.05 * intensity_range,
          linestyle='dashed',
          alpha=0.4,
          color='red')
plt.xlabel("Pixel")
plt.ylabel("Intensity (arbitrary)")

# Initialise the calibrator
c = Calibrator(refined_peaks,
               num_pix=len(spectrum),
               min_wavelength=6500,
               max_wavelength=10400)

c.add_atlas(["Ne", "Ar", "Kr"])
c.set_fit_constraints(range_tolerance=500, fit_tolerance=10, polydeg=5)

# Show the parameter space for searching possible solution
c.plot_search_space()

# Run the wavelength calibration
best_p, rms, residual, peak_utilisation = c.fit(max_tries=5000)

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
    robust_refit=True,
    polydeg=7)

# Plot the solution
c.plot_fit(flux, best_p, plot_atlas=True, log_spectrum=False, tolerance=3)

fit_diff = c.polyval(x_fit, best_p) - y_fit
rms = np.sqrt(np.sum(fit_diff**2 / len(x_fit)))

print("Stdev error: {} A".format(fit_diff.std()))
print("Peaks utilisation rate: {}%".format(peak_utilisation * 100))
