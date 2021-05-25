import numpy as np
import os
from astropy.io import fits
from scipy.signal import find_peaks

from rascal.calibrator import Calibrator
from rascal import util

# Load the LT SPRAT data
if '__file__' in locals():
    base_dir = os.path.dirname(__file__)
else:
    base_dir = os.getcwd()

fits_file = fits.open(
    os.path.join(base_dir, '..',
                 'examples/data_lt_sprat/v_a_20190516_57_1_0_1.fits'))[0]

spectrum2D = fits_file.data

temperature = fits_file.header['REFTEMP']
pressure = fits_file.header['REFPRES'] * 100.
relative_humidity = fits_file.header['REFHUMID']

# Collapse into 1D spectrum between row 110 and 120
spectrum = np.median(spectrum2D[110:120], axis=0)

# Identify the peaks
peaks, _ = find_peaks(spectrum, height=100, prominence=10, distance=5)
peaks = util.refine_peaks(spectrum, peaks, window_width=5)

# Order of polynomial
fit_deg = 4

# Number of repetance
N = 1000

# Using NIST lines
max_tries = [50, 100, 150, 200, 250, 500, 1000, 2000, 5000]
best_p_mt = []
rms_mt = []
residual_mt = []
peak_utilisation_mt = []

for mt in max_tries:

    # Repeat N times
    best_p = []
    rms = []
    residual = []
    peak_utilisation = []

    # Initialise the calibrator
    c = Calibrator(peaks, spectrum=spectrum)
    c.set_hough_properties(num_slopes=5000,
                           range_tolerance=500.,
                           xbins=100,
                           ybins=100,
                           min_wavelength=3800.,
                           max_wavelength=8200.)
    c.set_ransac_properties(sample_size=5,
                            top_n_candidate=5,
                            filter_close=True)
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

    for i in range(N):

        print('max_tries: {}, repetition: {} of 1000'.format(mt, i + 1))

        # Run the wavelength calibration
        solution = c.fit(max_tries=mt, fit_deg=fit_deg, progress=False)
        best_p.append(solution[0])
        rms.append(solution[1])
        residual.append(solution[2])
        peak_utilisation.append(solution[3])

    best_p_mt.append(best_p)
    rms_mt.append(rms)
    residual_mt.append(residual)
    peak_utilisation_mt.append(peak_utilisation)

np.save(os.path.join(base_dir, 'sprat_best_p_auto_mt'), best_p_mt)
np.save(os.path.join(base_dir, 'sprat_rms_auto_mt'), rms_mt)
np.save(os.path.join(base_dir, 'sprat_residual_auto_mt'), residual_mt)
np.save(os.path.join(base_dir, 'sprat_peak_utilisation_auto_mt'), peak_utilisation_mt)
