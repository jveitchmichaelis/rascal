import json

import numpy as np
from scipy.signal import find_peaks

from rascal.calibrator import Calibrator
from rascal.util import refine_peaks

# Load the 1D Spectrum from Pypeit
spectrum_json = json.load(
    open('data_keck_deimos/keck_deimos_830g_l_PYPIT.json'))
spectrum = np.array(spectrum_json['spec'])

# Identify the arc lines
peaks, _ = find_peaks(spectrum, prominence=200, distance=10)
peaks = refine_peaks(spectrum, peaks, window_width=3)

c = Calibrator(peaks, spectrum=spectrum)
c.plot_arc(save_fig='png', filename='output/keck-deimos-arc-spectrum')
c.set_hough_properties(num_slopes=10000.,
                       range_tolerance=500.,
                       linearity_tolerance=50,
                       xbins=200,
                       ybins=200,
                       min_wavelength=6500.,
                       max_wavelength=10500.)
c.set_ransac_properties(sample_size=5,
                        top_n_candidate=5,
                        linear=True,
                        filter_close=True,
                        ransac_tolerance=5,
                        candidate_weighted=True,
                        hough_weight=1.0)
c.add_atlas(elements=["Ne", "Ar", "Kr"],
            min_intensity=1000.,
            pressure=70000.,
            temperature=285.)
c.do_hough_transform()

# Run the wavelength calibration
(fit_coeff, matched_peaks, matched_atlas, rms, residual, peak_utilisation,
 atlas_utilisation) = c.fit(max_tries=1000)

# Plot the solution
c.plot_fit(fit_coeff,
           spectrum,
           plot_atlas=True,
           log_spectrum=False,
           tolerance=5.,
           save_fig='png',
           filename='output/keck-deimos-wavelength-calibration')

# Show the parameter space for searching possible solution
print("RMS: {}".format(rms))
print("Stdev error: {} A".format(np.abs(residual).std()))
print("Peaks utilisation rate: {}%".format(peak_utilisation * 100))
print("Atlas utilisation rate: {}%".format(atlas_utilisation * 100))

c.plot_search_space(save_fig='png', filename='output/keck-deimos-search-space')