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

# Identify the peaks
peaks, _ = find_peaks(spectrum, prominence=15, distance=5, threshold=None)
peaks = util.refine_peaks(spectrum, peaks, window_width=5)

# Initialise the calibrator
c = Calibrator(peaks, spectrum=spectrum)
c.plot_arc()
c.set_hough_properties(num_slopes=5000,
                       range_tolerance=500.,
                       xbins=200,
                       ybins=200,
                       min_wavelength=2500.,
                       max_wavelength=4600.)
c.set_ransac_properties(sample_size=8, top_n_candidate=10)
c.add_atlas(elements=['Cu', 'Ne', 'Ar'],
            min_intensity=10,
            pressure=80000.,
            temperature=285.)

c.do_hough_transform()

# Run the wavelength calibration
best_p, rms, residual, peak_utilisation = c.fit(max_tries=100, fit_deg=4)

# Plot the solution
c.plot_fit(best_p, spectrum, plot_atlas=True, log_spectrum=False, tolerance=5.)

# Show the parameter space for searching possible solution
c.plot_search_space()

print("Stdev error: {} A".format(residual.std()))
print("Peaks utilisation rate: {}%".format(peak_utilisation * 100))
