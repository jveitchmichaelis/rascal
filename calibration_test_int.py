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

spectrum = np.flip(fits.open("../int20180101_01355922.fits.fz")[1].data.mean(1), 0)
atlas = load_calibration_lines("C:/Users/Josh/Downloads/calibration_lines.csv", elements = ["CuNeAr"], min_wavelength=200)
atlas = np.loadtxt("../CuNe_CuAr.txt", delimiter=',')/10.0
#print(atlas)

##spectrum -= min(spectrum)

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