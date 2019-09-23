import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import find_peaks
from calibrator import Calibrator
plt.ion()

spectrum = np.median(fits.open('v_a_20190516_55_1_0_1.fits')[0].data[110:130], axis=0)

# pixelscale in unit of A/pix
pix_scale = 9.2
peaks, _ = find_peaks(
  spectrum, distance=3., prominence=np.percentile(spectrum, 20))

plt.figure(1, figsize=(8,8))
plt.clf()
plt.plot(spectrum)
plt.vlines(peaks, spectrum[peaks.astype('int')], spectrum.max()*1.1, colors='C1')
plt.ylim(0, spectrum.max()*1.1)
plt.xlim(0, len(spectrum))

c = Calibrator(peaks, elements=["Xe"], min_wavelength=3000., max_wavelength=9000.)

# thresh (A) :: the individual line fitting tolerance to accept as a valid fitting point
# fit_tolerance (A) :: the RMS
c.set_fit_constraints(
    n_pix=len(spectrum),
    min_intercept=3000.,
    max_intercept=5000.,
    fit_tolerance=pix_scale*0.5,
    thresh=pix_scale*1.,
    polydeg=4
    )

pix=np.array((241., 269., 280., 294., 317., 359., 462.5, 468.4, 477., 530.3, 752., 836., 900., 912., 980.))
wave=np.array((4500.98, 4624.28, 4671.23, 4734.15, 4844.33, 5023.88, 5488.56, 5531.07, 5581.88, 5823.89, 6872.11, 7284.3, 7584.68, 7642.02, 7967.34))

# Providing known pixel-wavelength mapping
c.set_known_pairs(pix, wave)

manual_p = np.polyfit(pix, wave, deg=4)
c.plot_fit(spectrum, c.match_peaks_to_atlas(manual_p)[0], tolerance=pix_scale*1.)
# best solution to date (18-9-2019)
# array([ 9.14196336e-11, -7.08052801e-07,  1.32941263e-03,  3.84120934e+00, 3.50802551e+03])

best_p = c.fit(
    sample_size=20,
    max_tries=10000,
    top_n=10,
    n_slope=30000)
#c.plot_fit(spectrum, best_p_fast, tolerance=pix_scale)
#c.plot_fit(spectrum, best_p, tolerance=pix_scale*1.)
c.plot_fit(spectrum, c.match_peaks_to_atlas(best_p)[0], tolerance=pix_scale*1.)

