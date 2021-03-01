import json

from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from rascal.calibrator import Calibrator
from rascal.util import refine_peaks

# Load the 1D Spectrum from Pypeit
data = fits.open('data_eso36_efosc/EFOSC_spec_HeAr227_0005.fits')[0]
spectrum = np.median(data.data.T, axis=0)

# Identify the arc lines
peaks, _ = find_peaks(spectrum, prominence=20, distance=10, threshold=None)
peaks_refined = refine_peaks(spectrum, peaks, window_width=3)

c = Calibrator(peaks_refined, spectrum)
c.set_calibrator_properties(num_pix=len(spectrum),
                            plotting_library='matplotlib',
                            log_level='info')

c.plot_arc()

c.set_hough_properties(num_slopes=5000,
                       range_tolerance=500.,
                       xbins=100,
                       ybins=100,
                       min_wavelength=3500.,
                       max_wavelength=7500.)
c.set_ransac_properties(sample_size=5, top_n_candidate=5)
c.add_atlas(["He"])
c.add_atlas(["Ar"], min_intensity=20)
c.do_hough_transform()

# Run the wavelength calibration
best_p, rms, residual, peak_utilisation = c.fit(max_tries=1000)

c.plot_fit(best_p,
           plot_atlas=True,
           log_spectrum=False,
           tolerance=5.0,
           renderer='notebook')

c.plot_search_space()

rms = np.sqrt(np.sum(residual**2 / len(residual)))

print("Stdev error: {} A".format(np.abs(residual).std()))
print("Peaks utilisation rate: {}%".format(peak_utilisation * 100))