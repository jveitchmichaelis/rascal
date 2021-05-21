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

atlas = [
    4193.5, 4385.77, 4500.98, 4524.68, 4582.75, 4624.28, 4671.23, 4697.02,
    4734.15, 4807.02, 4921.48, 5028.28, 5618.88, 5823.89, 5893.29, 5934.17,
    6182.42, 6318.06, 6472.841, 6595.56, 6668.92, 6728.01, 6827.32, 6976.18,
    7119.60, 7257.9, 7393.8, 7584.68, 7642.02, 7740.31, 7802.65, 7887.40,
    7967.34, 8057.258
]
element = ['Xe'] * len(atlas)

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
max_tries = [25, 50, 75, 100, 150, 200, 250, 500, 1000, 2000, 5000]
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

    for i in range(N):

        print('max_tries: {}, repetition: {} of 1000'.format(mt, i + 1))
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
        c.add_user_atlas(element,
                        atlas,
                        constrain_poly=True,
                        pressure=pressure,
                        temperature=temperature,
                        relative_humidity=relative_humidity)

        c.do_hough_transform()

        # Run the wavelength calibration
        solution = c.fit(max_tries=mt, fit_deg=fit_deg, progress=False)
        best_p.append(solution[0])
        rms.append(solution[1])
        residual.append(solution[2])
        peak_utilisation.append(solution[3])

        del c
        c = None

    best_p_mt.append(best_p)
    rms_mt.append(rms)
    residual_mt.append(residual)
    peak_utilisation_mt.append(peak_utilisation)

np.save(os.path.join(base_dir, 'best_p_mt'), best_p_mt)
np.save(os.path.join(base_dir, 'rms_mt'), rms_mt)
np.save(os.path.join(base_dir, 'residual_mt'), residual_mt)
np.save(os.path.join(base_dir, 'peak_utilisation_mt'), peak_utilisation_mt)
