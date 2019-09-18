import os
import numpy as np 

def load_calibration_lines(input_file = None, elements = ["Hg", "Ar", "Xe", "CuNeAr", "Kr"], min_wavelength=350, max_wavelength=800, transform=None):

    if input_file is None:
        input_file = '{}/calibration_lines.csv'.format(os.path.dirname(__file__))
    
    with open(input_file,'r') as calfile:
        data = calfile.readlines()[1:]
    
    cal_lines = [line.replace('\n', '').split(',')[:2] for line in data]

    lines = np.array(np.sort([float(line[0]) for line in cal_lines if line[1].rstrip() in elements]))

    mask = (lines > min_wavelength)*(lines < max_wavelength)

    if transform == 'a_to_nm':
        lines /= 10.0
    elif transform == 'nm_to_a':
        lines *= 10.0
    
    return lines[mask]