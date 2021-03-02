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

# Identify the peaks
peaks, _ = find_peaks(spectrum,
                      height=500,
                      prominence=10,
                      distance=5,
                      threshold=None)
peaks = util.refine_peaks(spectrum, peaks, window_width=3)

# Initialise the calibrator
c = Calibrator(peaks, spectrum=spectrum)
c.plot_arc(log_spectrum=True)
c.set_hough_properties(num_slopes=5000,
                       range_tolerance=500.,
                       xbins=100,
                       ybins=100,
                       min_wavelength=6500.,
                       max_wavelength=10500.)
c.set_ransac_properties(sample_size=5, top_n_candidate=5, filter_close=True)
c.add_atlas(elements=['Cu', 'Ne', 'Ar'],
            min_intensity=10,
            pressure=90000.,
            temperature=285.)

c.do_hough_transform()

# Run the wavelength calibration
best_p, rms, residual, peak_utilisation = c.fit(max_tries=200)

# Plot the solution
c.plot_fit(best_p, spectrum, plot_atlas=True, log_spectrum=False, tolerance=5.)

# Show the parameter space for searching possible solution
c.plot_search_space()

print("Stdev error: {} A".format(residual.std()))
print("Peaks utilisation rate: {}%".format(peak_utilisation * 100))
