import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.signal import find_peaks

from rascal.calibrator import Calibrator
from rascal.util import refine_peaks

atlas = [
    3650.153, 4046.563, 4077.831, 4358.328, 5460.735, 5769.598, 5790.663,
    6682.960, 6752.834, 6871.289, 6965.431, 7030.251, 7067.218, 7147.042,
    7272.936, 7383.981, 7503.869, 7514.652, 7635.106, 7723.98
]
element = ['HgAr'] * len(atlas)

data = fits.open(
    'data_gtc_osiris/0002672523-20200911-OSIRIS-OsirisCalibrationLamp.fits')[1]

plt.figure(1, figsize=(10, 4))
plt.imshow(np.log10(data.data.T), aspect='auto', origin='lower')
plt.xlabel('Spectral Direction / Pix')
plt.ylabel('Spatial Direction / Pix')
plt.tight_layout()
plt.savefig('output/gtc-osiris-arc-image.png')

spectrum = np.median(data.data.T[550:570], axis=0)

# Identify the arc lines
peaks, _ = find_peaks(spectrum,
                      height=1250,
                      prominence=20,
                      distance=3,
                      threshold=None)
peaks_refined = refine_peaks(spectrum, peaks, window_width=3)

c = Calibrator(peaks_refined, spectrum=spectrum)

c.set_hough_properties(num_slopes=2000,
                       xbins=100,
                       ybins=100,
                       min_wavelength=3500.,
                       max_wavelength=8000.,
                       range_tolerance=500.,
                       linearity_tolerance=50)

c.add_user_atlas(elements=element, wavelengths=atlas, constrain_poly=True)

c.set_ransac_properties(sample_size=5,
                        top_n_candidate=5,
                        linear=True,
                        filter_close=True,
                        ransac_tolerance=5,
                        candidate_weighted=True,
                        hough_weight=1.0)

c.do_hough_transform()

c.plot_arc(save_fig='png',
           filename='output/gtc-osiris-arc-spectrum')

c.add_user_atlas(elements=element, wavelengths=atlas, constrain_poly=True)
c.do_hough_transform()

# Run the wavelength calibration
(fit_coeff, matched_peaks, matched_atlas, rms, residual, peak_utilisation,
 atlas_utilisation) = c.fit(max_tries=200, fit_tolerance=10., fit_deg=4)

# Plot the solution
c.plot_fit(fit_coeff,
           plot_atlas=True,
           log_spectrum=False,
           tolerance=5.,
           save_fig='png',
           filename='output/gtc-osiris-wavelength-calibration')

# Show the parameter space for searching possible solution
print("RMS: {}".format(rms))
print("Stdev error: {} A".format(np.abs(residual).std()))
print("Peaks utilisation rate: {}%".format(peak_utilisation * 100))
print("Atlas utilisation rate: {}%".format(atlas_utilisation * 100))

c.plot_search_space(save_fig='png', filename='output/gtc-osiris-search-space')
