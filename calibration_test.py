from calibrator import Calibrator, getPeaks, polyfit_value
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

def load_calibration_lines(input_file = 'calibration_lines.csv', elements = ["Hg", "Ar", "Xe", "CuNeAr", "Kr"], min_wavelength=350, max_wavelength=800):

    with open(input_file,'r') as calfile:
        data = calfile.readlines()[1:]
    
    cal_lines = [line.replace('\n', '').split(',')[:2] for line in data]

    lines = np.array(np.sort([float(line[0]) for line in cal_lines if line[1] in elements]))

    mask = (lines > min_wavelength)*(lines < max_wavelength)
    
    return lines[mask]

spectrum = np.loadtxt("C:/Users/Josh/Downloads/A620EBA HgCal.mspec", delimiter=',')[:,1]
atlas = load_calibration_lines("C:/Users/Josh/Downloads/calibration_lines.csv", elements = ["Hg", "Ar"], min_wavelength=380)

peaks, _ = getPeaks(np.log(spectrum), min_dist=100, thres=0.5)

c = Calibrator(peaks, atlas)
c.set_fit_constraints(min_slope=0.1,
                      max_slope=0.15,
                      min_intercept=375,
                      max_intercept=425,
                      line_fit_thresh=5)
 
best_p = c.fit()
print(best_p)
c.plot_fit(spectrum, best_p)

