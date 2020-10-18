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
fits_file = fits.open(
    os.path.join(base_dir, 'data_lt_sprat/v_a_20190516_57_1_0_1.fits'))[0]

spectrum2D = fits_file.data

temperature = fits_file.header['REFTEMP']
pressure = fits_file.header['REFPRES'] * 100.
relative_humidity = fits_file.header['REFHUMID']

# Collapse into 1D spectrum between row 110 and 120
spectrum = np.median(spectrum2D[110:120], axis=0)

# Identify the peaks
peaks, _ = find_peaks(spectrum, height=100, prominence=10, distance=5)
peaks = util.refine_peaks(spectrum, peaks, window_width=5)

# Initialise the calibrator
c = Calibrator(peaks, spectrum=spectrum)
c.plot_arc()
c.set_hough_properties(num_slopes=5000,
                       range_tolerance=500.,
                       xbins=100,
                       ybins=100,
                       min_wavelength=3800.,
                       max_wavelength=8200.)
c.set_ransac_properties(sample_size=5, top_n_candidate=5, filter_close=True)
c.add_atlas(elements=["Xe"],
            min_intensity=10.,
            min_distance=5,
            min_atlas_wavelength=3800.,
            max_atlas_wavelength=8200.,
            candidate_tolerance=5.,
            pressure=pressure,
            temperature=temperature,
            relative_humidity=relative_humidity)

c.do_hough_transform()

# Run the wavelength calibration
best_p, rms, residual, peak_utilisation = c.fit(max_tries=250)

# Plot the solution
c.plot_fit(best_p, spectrum, plot_atlas=True, log_spectrum=False, tolerance=5.)

# Show the parameter space for searching possible solution
#c.plot_search_space()

print("Stdev error: {} A".format(residual.std()))
print("Peaks utilisation rate: {}%".format(peak_utilisation * 100))
