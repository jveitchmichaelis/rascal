import os
import logging
import sys

from astropy.io import fits
from matplotlib import pyplot as plt

import numpy as np
import pytest
from scipy.signal import find_peaks
from tqdm.autonotebook import tqdm

from rascal.calibrator import Calibrator
from rascal import models
from rascal import util


# Load the LT SPRAT data
spectrum2D = fits.open('/Users/marcolam/git/GMOSfieldflattening/example/gaia2018cnz/output/N20181115S0215_flattened.fits')[0].data
# Collapse into 1D spectrum between row 150 and 160
spectrum = np.median(spectrum2D[150:160], axis=0)[::-1]


sys.path.append('/Users/marcolam/git/GMOSfieldflattening')

from gmos_longslit_fieldflattening import create_pixel_array


pixel_list = create_pixel_array('north', 2)


# Identify the peaks
peaks, _ = find_peaks(spectrum, height=500, distance=10, threshold=None)
peaks = util.refine_peaks(spectrum, peaks, window_width=5)

# Initialise the calibrator
c = Calibrator(peaks, num_pix=len(pixel_list), pixel_list=pixel_list, min_wavelength=5000., max_wavelength=9500.)
c.set_fit_constraints(num_slopes=2000,
                      range_tolerance=500.,
                      candidate_thresh=10.,
                      fit_tolerance=5,
                      xbins=200,
                      ybins=200)


c.add_atlas('CuAr')

def run_sprat_calibration(polydeg, peaks):
    # Run the wavelength calibration
    best_p, rms, residual, peak_utilisation = c.fit(max_tries=5000,
                      top_n=10)
    # First set is to refine only the 0th and 1st coefficient (i.e. the 2 lowest orders)
    best_p, x_fit, y_fit, residual, peak_utilisation = c.match_peaks(
        best_p,
        n_delta=2,
        tolerance=10.,
        convergence=1e-10,
        method='Nelder-Mead',
        robust_refit=True,
        polydeg=polydeg)
    # Second set is to refine all the coefficients
    best_p, x_fit, y_fit, residual, peak_utilisation = c.match_peaks(
        best_p,
        tolerance=10.,
        convergence=1e-10,
        method='Nelder-Mead',
        robust_refit=True,
        polydeg=polydeg)
    wave = np.polynomial.polynomial.polyval(peaks, best_p)
    return best_p, residual, peak_utilisation, rms, wave


# run n times
n = 10

c0 = np.zeros(n)
c1 = np.zeros(n)
c2 = np.zeros(n)
c3 = np.zeros(n)
c4 = np.zeros(n)
peak_utilisation = np.zeros(n)
rms = np.zeros(n)
waves = np.zeros((n, len(peaks)))

for i in range(n):
    print(i)
    best_p, _, peak_utilisation[i], rms[i], waves[i] = run_sprat_calibration(4, peaks)
    c0[i], c1[i], c2[i], c3[i], c4[i] = best_p


#sort by the wavelength fitted to the largest pixel value
mask_sorted = np.argsort(waves[:,-1])
'''
plt.figure(1)
plt.clf()
plt.scatter(c0[mask_sorted], c1[mask_sorted], c=np.arange(n), s=5)
plt.xlabel('c0')
plt.ylabel('c1')
plt.xlim(min(c0), max(c0))
plt.ylim(min(c1), max(c1))
plt.colorbar()
plt.grid()
plt.tight_layout()

plt.figure(2)
plt.clf()
plt.scatter(c1[mask_sorted], c2[mask_sorted], c=np.arange(n), s=5)
plt.xlabel('c1')
plt.ylabel('c2')
plt.xlim(min(c1), max(c1))
plt.ylim(min(c2), max(c2))
plt.colorbar()
plt.grid()
plt.tight_layout()

plt.figure(3)
plt.clf()
plt.scatter(c2[mask_sorted], c3[mask_sorted], c=np.arange(n), s=5)
plt.xlabel('c2')
plt.ylabel('c3')
plt.xlim(min(c2), max(c2))
plt.ylim(min(c3), max(c3))
plt.colorbar()
plt.grid()
plt.tight_layout()

plt.figure(4)
plt.clf()
plt.scatter(c3[mask_sorted], c4[mask_sorted], c=np.arange(n), s=5)
plt.xlabel('c3')
plt.ylabel('c4')
plt.xlim(min(c3), max(c3))
plt.ylim(min(c4), max(c4))
plt.colorbar()
plt.grid()
plt.tight_layout()
'''
plt.figure(5)
plt.clf()
for i in range(len(peaks)):
    plt.plot(waves[mask_sorted][:,i], color='black', lw=1)

plt.title('hand-picked lines')
plt.grid()
plt.tight_layout()
