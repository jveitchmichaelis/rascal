import numpy as np
import os
from astropy.io import fits
from scipy.signal import find_peaks
from scipy import interpolate

from rascal.calibrator import Calibrator
from rascal import util

# All pixel values here are unbinned
hamamatsu_width = 2048

# chip gaps
hamamatsu_n_gap = 67
hamamatsu_s_gap = 61


def create_pixel_array(northsouth, binning):
    '''
    The GMOS longslit spectrum spreads over three CCDs. This function creates
    the pixel list corresponding the binned pixels adjusted for the chip gaps,
    which can lead to a non-integer shift in the number of pixels.
    Parameters
    ----------
    northsout: str
        Indicate whether the Gemini north or south, because the chip gaps are
        different.
    binning : numeric
        The binning factor
    Returns
    -------
    pixels: numpy.array
        The pixel values adjusted for chip gaps for the corresponding pixel.
    '''

    binned_width = hamamatsu_width / binning

    if northsouth == 'north':
        gap_binned_width = hamamatsu_n_gap / binning
    elif northsouth == 'south':
        gap_binned_width = hamamatsu_s_gap / binning
    else:
        raise ValueError('Please choose from "north" or "south".')

    pixels = np.concatenate(
        (np.arange(binned_width),
         np.arange(binned_width + gap_binned_width,
                   binned_width * 2 + gap_binned_width),
         np.arange(binned_width * 2 + gap_binned_width * 2,
                   binned_width * 3 + gap_binned_width * 2)))

    return pixels


pixels = create_pixel_array('north', 2)
rawpix_to_pix_itp = interpolate.interp1d(np.arange(len(pixels)), pixels)

# Load the LT SPRAT data
if '__file__' in locals():
    base_dir = os.path.dirname(__file__)
else:
    base_dir = os.getcwd()

fits_file = fits.open(
    os.path.join(base_dir, '..',
                 'examples/data_gemini_gmos/N20181115S0215_flattened.fits'))[0]

spectrum2D = fits_file.data

# Collapse into 1D spectrum between row 300 and 310
spectrum = np.median(spectrum2D[300:310], axis=0)[::-1]

temperature = 276.55
pressure = 61700.
relative_humidity = 4.

atlas = [
    4703.632, 4728.19041, 4766.19677, 4807.36348, 4849.16386, 4881.22627,
    4890.40721, 4906.12088, 4934.58593, 4966.46490, 5018.56194, 5063.44827,
    5163.723, 5189.191, 5497.401, 5560.246, 5608.290, 5913.723, 6754.698,
    6873.185, 6967.352, 7032.190, 7069.167, 7149.012, 7274.940, 7386.014,
    7505.935, 7516.721, 7637.208, 7725.887, 7893.246, 7950.362, 8105.921,
    8117.542, 8266.794, 8410.521, 8426.963, 8523.783, 8670.325, 9125.471,
    9197.161, 9227.03, 9356.787, 9660.435, 9787.186
]

element = ['CuAr'] * len(atlas)

# Identify the peaks
peaks, _ = find_peaks(spectrum, height=1000, distance=5)
peaks = util.refine_peaks(spectrum, peaks, window_width=5)

peaks_shifted = rawpix_to_pix_itp(peaks)

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
                           xbins=200,
                           ybins=200,
                           min_wavelength=5000.,
                           max_wavelength=9500.)
    c.set_ransac_properties(sample_size=5,
                            top_n_candidate=5,
                            filter_close=True)
    c.add_user_atlas(element,
                     atlas,
                     constrain_poly=True,
                     vacuum=True,
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

np.save(os.path.join(base_dir, 'gmos_best_p_manual_mt'), best_p_mt)
np.save(os.path.join(base_dir, 'gmos_rms_manual_mt'), rms_mt)
np.save(os.path.join(base_dir, 'gmos_residual_manual_mt'), residual_mt)
np.save(os.path.join(base_dir, 'gmos_peak_utilisation_manual_mt'),
        peak_utilisation_mt)
