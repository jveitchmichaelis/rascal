import numpy as np
from astropy.io import fits
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import os

from rascal.calibrator import Calibrator
from rascal.util import refine_peaks
from rascal import models

# Load the 1D spectrum
base_dir = os.path.dirname(__file__)
spectrum = np.loadtxt(os.path.join(base_dir,
                                   'data_msp1000/A620EBA HgCal.mspec'),
                      delimiter=',')[:, 1]

# Identify the peaks
peaks, _ = find_peaks(spectrum, prominence=300, distance=15, threshold=None)
peaks_refined = refine_peaks(spectrum, peaks, window_width=5)

# Initialise the calibrator
c = Calibrator(peaks, spectrum=spectrum)
c.plot_arc()
c.set_hough_properties(num_slopes=5000,
                       range_tolerance=500.,
                       xbins=100,
                       ybins=100,
                       min_wavelength=4000.,
                       max_wavelength=8750.)
c.set_ransac_properties(sample_size=5, top_n_candidate=5, filter_close=True)
# Ignore bluer Argon lines
c.add_atlas("Hg")
c.add_atlas("Ar", min_atlas_wavelength=6500)

c.do_hough_transform()

# Run the wavelength calibration
best_p, rms, residual, peak_utilisation = c.fit(max_tries=1000)

# Plot the solution
c.plot_fit(best_p, spectrum, plot_atlas=True, log_spectrum=False, tolerance=5)

# Show the parameter space for searching possible solution
c.plot_search_space()

print("Stdev error: {} A".format(residual.std()))
print("Peaks utilisation rate: {}%".format(peak_utilisation * 100))
