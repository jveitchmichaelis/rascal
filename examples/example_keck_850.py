import json

from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from rascal.calibrator import Calibrator
from rascal.util import refine_peaks

# Load the 1D Spectrum from Pypeit
spectrum_json = json.load(
    open('data_keck_deimos/keck_deimos_830g_l_PYPIT.json'))
spectrum = np.array(spectrum_json['spec'])

# Identify the arc lines
peaks, _ = find_peaks(spectrum, prominence=100, distance=10)
peaks = refine_peaks(spectrum, peaks, window_width=3)

c = Calibrator(peaks, spectrum=spectrum)
c.plot_arc()
c.set_hough_properties(num_slopes=5000,
                       range_tolerance=1000.,
                       xbins=200,
                       ybins=200,
                       min_wavelength=6500.,
                       max_wavelength=10500.)
c.set_ransac_properties(sample_size=5, top_n_candidate=10)
c.add_atlas(elements=["Ne", "Ar", "Kr"],
            min_intensity=200.,
            pressure=70000.,
            temperature=285.)
c.do_hough_transform()

# Run the wavelength calibration
best_p, rms, residual, peak_utilisation = c.fit(max_tries=1000)

# Plot the solution
c.plot_fit(best_p, spectrum, plot_atlas=True, log_spectrum=False, tolerance=5.)

# Show the parameter space for searching possible solution
c.plot_search_space()

print("Stdev error: {} A".format(residual.std()))
print("Peaks utilisation rate: {}%".format(peak_utilisation * 100))
