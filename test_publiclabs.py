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

spectrum = np.loadtxt("../158352.csv", delimiter=',', usecols=1)
atlas = load_calibration_lines("C:/Users/Josh/Downloads/calibration_lines.csv", elements=["Hg", "Eu", "Tb"], min_wavelength=300,max_wavelength=700)

print(atlas)

#peaks, _ = getPeaks(spectrum, min_dist=25, thres=0.3)
from scipy.signal import find_peaks

spectrum /= spectrum.max()
peaks, _ = find_peaks(spectrum, height=0.2, distance=10, width=(0,15))

plt.plot(spectrum)
plt.vlines(peaks, 0, max(spectrum))
plt.show()

if len(peaks) == 0:
    print("No peaks found, try again!")
    exit(1)

c = Calibrator(peaks, atlas)
c.set_fit_constraints(min_slope=0.8,
                      max_slope=1.2,
                      min_intercept=250,
                      max_intercept=350,
                      line_fit_thresh=5)
 
best_p = c.fit(5)

if best_p is not None:
    c.plot_fit(spectrum, best_p)

