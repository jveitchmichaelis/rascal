import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from scipy.signal import find_peaks
from calibrator import Calibrator

def load_calibration_lines(input_file='calibration_lines.csv',
                           elements=["Hg", "Ar", "Xe", "CuNeAr", "Kr"],
                           min_wavelength=100,
                           max_wavelength=1000):
    cal_lines = np.loadtxt(input_file, delimiter=',', dtype='U', skiprows=1)
    wave = cal_lines[:, 0].astype('float')
    element = cal_lines[:, 1]
    # Get lines of the requested elements
    lines = wave[np.isin(element, elements)]
    # Get only lines within the requested wavelength
    mask = (lines > min_wavelength) * (lines < max_wavelength)
    return lines[mask]

atlas = load_calibration_lines(
    "calibration_lines.csv", elements=["Xe"], min_wavelength=300, max_wavelength=900)

spectrum = np.median(fits.open('v_a_20190516_55_1_0_1.fits')[0].data[110:120], axis=0)

peaks, _ = find_peaks(spectrum, distance=10., threshold=10.)

plt.plot(spectrum)
plt.vlines(peaks,
              spectrum[peaks.astype('int')],
              spectrum.max(),
              colors='C1')

c = Calibrator(peaks, atlas)
c.set_fit_constraints(
    min_slope=0.2,
    max_slope=0.8,
    min_intercept=200.,
    max_intercept=500.,
    fit_tolerance=0.5,
    line_fit_thresh=2,
    thresh=5,
    polydeg=5,
    fittype='poly')

# Providing known pixel-wavelength mapping
#c.set_known_pairs([635.6024, 803.5022], [631.806, 711.96])
best_p = c.fit(mode='slow', progress=False)
c.plot_fit(spectrum, best_p)
