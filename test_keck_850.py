from calibrator import Calibrator, getPeaks, polyfit_value
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import json

def load_calibration_lines(input_file = 'calibration_lines.csv', elements = ["Hg", "Ar", "Xe", "CuNeAr", "Kr"], min_wavelength=350, max_wavelength=800):

    with open(input_file,'r') as calfile:
        data = calfile.readlines()[1:]
    
    cal_lines = [line.replace('\n', '').split(',')[:2] for line in data]

    lines = np.array(np.sort([float(line[0]) for line in cal_lines if line[1] in elements]))

    mask = (lines > min_wavelength)*(lines < max_wavelength)
    
    return lines[mask]

with open("./keck_deimos_830g_l_PYPIT.json") as json_file:  
    data = json.load(json_file)
    spectrum = np.array(data["spec"])

atlas = load_calibration_lines("C:/Users/Josh/Downloads/calibration_lines.csv", elements=["Ne", "Ar", "Kr"], min_wavelength=500,max_wavelength=1000)

print(atlas)

#peaks, _ = getPeaks(spectrum, min_dist=25, thres=0.3)
from scipy.signal import find_peaks

spectrum /= spectrum.max()
peaks, _ = find_peaks(spectrum, height=0.1, distance=10, width=(0,40))

plt.plot(spectrum)
plt.vlines(peaks, 0, max(spectrum))
plt.show()

if len(peaks) == 0:
    print("No peaks found, try again!")
    exit(1)

c = Calibrator(peaks, atlas)
c.set_fit_constraints(min_slope=0.043,
                      max_slope=0.053,
                      min_intercept=550,
                      max_intercept=650,
                      line_fit_thresh=2)
 
best_p = c.match_peaks_to_atlas(c.fit(2))

if best_p is not None:
    c.plot_fit(spectrum, best_p)

print("Start wavelength: ", best_p[-1])
print("Centre wavelenght: ", polyfit_value(len(spectrum)/2, best_p[::-1]))
print("Dispersion: ", best_p[-2])
