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
    os.path.join(base_dir, 'data_lt_sprat/v_a_20190516_57_1_0_1.fits'))[0].data

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
peaks = util.refine_peaks(spectrum, peaks, window_width=5)

# Initialise the calibrator
c = Calibrator(peaks, num_pix=1024, min_wavelength=3500., max_wavelength=8000.)
c.set_fit_constraints(num_slopes=10000,
                      range_tolerance=500.,
                      xbins=100,
                      ybins=100)

# blend: 4829.71, 4844.33
# blend: 5566.62, 5581.88
# blend: 6261.212, 6265.302
# blend: 6872.11, 6882.16
# blend: 7283.961, 7285.301
# blend: 7316.272, 7321.452
atlas = [
    4193.5, 4385.77, 4500.98, 4524.68, 4582.75, 4624.28, 4671.23, 4697.02,
    4734.15, 4807.02, 4921.48, 5028.28, 5618.88, 5823.89, 5893.29, 5934.17,
    6182.42, 6318.06, 6472.841, 6595.56, 6668.92, 6728.01, 6827.32, 6976.18,
    7119.60, 7257.9, 7393.8, 7584.68, 7642.02, 7740.31, 7802.65, 7887.40,
    7967.34, 8057.258
]
element = ['Xe'] * len(atlas)

c.load_user_atlas(element, atlas)

c.set_peaks(constrain_poly=True)

# Show the parameter space for searching possible solution
c.plot_search_space()

# Run the wavelength calibration
best_p, rms, residual, peak_utilisation = c.fit(max_tries=10000,
                                                sample_size=5,
                                                top_n=20)

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
