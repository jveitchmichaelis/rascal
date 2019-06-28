from calibrator import Calibrator, getPeaks, polyfit_value
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def load_calibration_lines(input_file = 'calibration_lines.csv', elements = ["Hg", "Ar", "Xe", "CuNeAr", "Kr"], min_wavelength=350, max_wavelength=800):

    with open(input_file,'r') as calfile:
        data = calfile.readlines()[1:]
    
    cal_lines = [line.replace('\n', '').split(',')[:2] for line in data]

    lines = np.array(np.sort([float(line[0]) for line in cal_lines if line[1] in elements]))

    mask = (lines > min_wavelength)*(lines < max_wavelength)
    
    return lines[mask]

spectrum = fits.open("../r2701003_blue_arc.fit")[1].data.mean(1)
atlas = load_calibration_lines("C:/Users/Josh/Downloads/calibration_lines.csv", elements = ["CuNeAr"], min_wavelength=200)

plt.plot(spectrum)
plt.show()

peaks, _ = getPeaks(np.log(spectrum), min_dist=25, thres=0.3)

c = Calibrator(peaks, atlas)
c.set_fit_constraints(min_slope=0.07,
                      max_slope=0.09,
                      min_intercept=200,
                      max_intercept=300,
                      line_fit_thresh=5)
 
best_p = c.fit(3)
print(best_p)
c.plot_fit(spectrum, best_p)

