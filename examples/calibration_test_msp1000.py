from rascal.calibrator import Calibrator, polyfit_value
from rascal.util import load_calibration_lines
from scipy.signal import find_peaks

import numpy as np
import matplotlib.pyplot as plt

spectrum = np.loadtxt("A620EBA HgCal.mspec", delimiter=',')[:,1]
atlas = load_calibration_lines(elements = ["Hg", "Ar"], min_wavelength=400)

print(atlas)

peaks, _ = find_peaks(spectrum, height=600, distance=25, width=1)

expected_slope = 0.16
slope_accuracy = 0.05

expected_intercept = 400
intercept_accuracy = 0.05

c = Calibrator(peaks, atlas)
c.set_fit_constraints(min_slope=expected_slope*(1 - slope_accuracy),
                      max_slope=expected_slope*(1 + slope_accuracy),
                      min_intercept=expected_intercept*(1 - intercept_accuracy),
                      max_intercept=expected_intercept*(1 + intercept_accuracy),
                      line_fit_thresh=5)
 
best_p = c.fit(3)
print("Final fit: ", best_p)
c.plot_fit(spectrum, best_p)

