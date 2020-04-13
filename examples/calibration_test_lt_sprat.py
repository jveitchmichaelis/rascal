import numpy as np
from astropy.io import fits
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm
import os

from rascal.calibrator import Calibrator
from rascal import models

# Load the LT SPRAT data
base_dir = os.path.dirname(__file__)
spectrum2D = fits.open(os.path.join(base_dir,'data_lt_sprat/v_a_20190516_57_1_0_1.fits'))[0].data

# Collapse into 1D spectrum between row 110 and 120
spectrum = np.median(spectrum2D[110:120], axis=0)

plt.figure()
plt.plot(spectrum / spectrum.max())
plt.title('Number of pixels: ' + str(spectrum.shape[0]))
plt.xlabel("Pixel (Spectral Direction)")
plt.ylabel("Normalised Count")
plt.xlim(0, 1024)
plt.grid()
plt.tight_layout()

# Identify the peaks
peaks, _ = find_peaks(spectrum, height=500, distance=5, threshold=None)

# Initialise the calibrator
c = Calibrator(peaks,
               num_pix=1024,
               min_wavelength=3800.,
               max_wavelength=8200.)
c.set_fit_constraints(num_slopes=5000,
                      range_tolerance=500.,
                      xbins=200,
                      ybins=200)
c.add_atlas(elements='Xe')

# Show the parameter space for searching possible solution
c.plot_search_space()

# Run the wavelength calibration
best_p, rms, residual, peak_utilisation = c.fit(max_tries=10000)

# Refine solution
best_p, x_fit, y_fit, residual, peak_utilisation = c.match_peaks_to_atlas(
    best_p, polydeg=4, tolerance=3)

# Plot the solution
c.plot_fit(spectrum, best_p, plot_atlas=True, log_spectrum=False, tolerance=3)

fit_diff = c.polyval(x_fit, best_p) - y_fit
rms = np.sqrt(np.sum(fit_diff**2 / len(x_fit)))

print("Stdev error: {} A".format(fit_diff.std()))
print("Peaks utilisation rate: {}%".format(peak_utilisation * 100))
