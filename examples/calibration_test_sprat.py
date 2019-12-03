from rascal.calibrator import Calibrator, polyfit_value
from rascal.util import load_calibration_lines

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from scipy.signal import find_peaks

spectrum = np.median(fits.open("./v_a_20190516_55_1_0_1.fits")[0].data)

atlas = load_calibration_lines(elements = ["Xe"], min_wavelength=300, max_wavelength=850)

plt.plot(spectrum)
plt.show()

peaks, _ = find_peaks(spectrum, height=1000)

print(atlas)
print(peaks)

c = Calibrator(peaks, atlas)
c.set_fit_constraints(min_slope=0.43,
                      max_slope=0.47,
                      min_intercept=300,
                      max_intercept=400,
                      line_fit_thresh=2)
 
best_p = c.fit(3)
print(best_p)
c.plot_fit(spectrum, best_p)

