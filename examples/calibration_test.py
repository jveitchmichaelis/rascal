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
    thresh=pix_scale*2.,
    polydeg=3
    )

# Providing known pixel-wavelength mapping
#best_p_fast = c.fit(mode='fast', progress=True)

best_p = c.fit(
    sample_size=5,
    max_tries=100,
    top_n=100,
    n_slope=10000)
#c.plot_fit(spectrum, best_p_fast, tolerance=pix_scale)
#c.plot_fit(spectrum, best_p, tolerance=pix_scale*1.)
c.plot_fit(spectrum, c.match_peaks_to_atlas(best_p)[0], tolerance=pix_scale*1.)


coeff = best_p

# Providing known pixel-wavelength mapping
#best_p_fast = c.fit(mode='fast', progress=True)

c.set_fit_constraints(
    n_pix=len(spectrum),
    min_intercept=3000.,
    max_intercept=5000.,
    fit_tolerance=pix_scale*0.5,
    thresh=pix_scale*1.,
    polydeg=5
    )

best_p = c.fit(
    sample_size=10,
    max_tries=2000,
    top_n=10,
    n_slope=10000,
    coeff=coeff
    )
#c.plot_fit(spectrum, best_p_fast, tolerance=pix_scale)
#c.plot_fit(spectrum, best_p, tolerance=pix_scale*1.)
c.plot_fit(spectrum, c.match_peaks_to_atlas(best_p)[0], tolerance=pix_scale*1.)





