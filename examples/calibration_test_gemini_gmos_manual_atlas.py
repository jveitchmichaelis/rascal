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
                      ybins=200,
                      polydeg=4)

# Vacuum wavelengths
# blend: 5143.21509, 5146.74143
# something weird near there, so not used: 8008.359, 8016.990
atlas = [
    4703.632, 4728.19041, 4766.19677, 4807.36348, 4849.16386, 4881.22627, 4890.40721, 4906.12088, 4934.58593, 4966.46490,
    5018.56194, 5063.44827, 5163.723, 5189.191, 5497.401,
    5560.246, 5608.290, 5913.723,
    6754.698, 6873.185, 6967.352,
    7032.190, 7069.167, 7149.012, 7274.940, 7386.014,
    7505.935, 7516.721, 7637.208, 7725.887, 7893.246, 7950.362,
    8105.921, 8117.542, 8266.794, 8410.521, 8426.963,
    8523.783, 8670.325,
    9125.471, 9197.161, 9227.03, 9356.787,
    9660.435, 9787.186
]

element = ['CuAr'] * len(atlas)

c.load_user_atlas(elements=element, wavelengths=atlas, vacuum=True, pressure=61700., temperature=276.55, relative_humidity=4.)

c.set_peaks(constrain_poly=True)

# Show the parameter space for searching possible solution
c.plot_search_space()

# Run the wavelength calibration
best_p, rms, residual, peak_utilisation = c.fit(max_tries=10000)

# Plot the solution
c.plot_fit(spectrum, best_p, plot_atlas=True, log_spectrum=False, tolerance=5.)

fit_diff = c.polyval(x_fit, best_p) - y_fit
rms = np.sqrt(np.sum(fit_diff**2 / len(x_fit)))

print("Stdev error: {} A".format(fit_diff.std()))
print("Peaks utilisation rate: {}%".format(peak_utilisation * 100))
