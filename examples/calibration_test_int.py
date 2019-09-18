from rascal.calibrator import Calibrator, polyfit_value, getPeaks
from rascal.util import load_calibration_lines

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

spectrum = np.flip(fits.open("int20180101_01355922.fits.fz")[1].data.mean(1), 0)
atlas = load_calibration_lines(elements = ["CuNeAr"], min_wavelength=200)/10.0
atlas = np.loadtxt("../rascal/CuNe_CuAr.txt", delimiter=',')/10.0

peaks, _ = getPeaks(np.log(spectrum), min_dist=30, thres=0.1)

plt.plot(np.log(spectrum))
plt.vlines(peaks, min(np.log(spectrum)), max(np.log(spectrum)))
plt.show()

c = Calibrator(peaks, atlas)
c.set_fit_constraints(min_slope=0.045,
                      max_slope=0.055,
                      min_intercept=250,
                      max_intercept=300,
                      line_fit_thresh=1,
                      fit_tolerance=0.5)

best_p = c.fit(3)
final_p = c.match_peaks_to_atlas(best_p, 0.25)

print(atlas)
print(final_p)
c.plot_fit(spectrum, final_p)
c.match_peaks_to_atlas(final_p)