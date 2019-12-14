import numpy as np
import random
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import pkg_resources

"""
def filter_wavelengths(lines, min_wavelength, max_wavlength):
    wavelengths = lines[:,1].astype(np.float32)
    wavelength_mask = (wavelengths >= min_wavelength) * (wavelengths <= max_wavlength)

    return lines[wavelength_mask]

def filter_intensity(lines, min_intensity=0):
    out = []
    for line in lines:
        _, _, intensity = line
        if float(intensity) >= min_intensity:
            out.append(True)
        else:
            out.append(False)
    
    return np.array(out).astype(bool)

def filter_separation(wavelengths, min_separation=0):
    left_dists = np.zeros_like(wavelengths)
    left_dists[1:] = wavelengths[1:]-wavelengths[:-1]

    right_dists = np.zeros_like(wavelengths)
    right_dists[:-1] = wavelengths[1:]-wavelengths[:-1]
    distances = np.minimum(right_dists, left_dists)
    distances[0] = right_dists[0]
    distances[-1] = left_dists[-1]
    
    distance_mask = np.abs(distances) >= min_separation
    
    return distance_mask

def load_calibration_lines(elements=[], min_wavelength=0, max_wavelength=15000, min_distance=10, min_intensity=40):

    if isinstance(elements, str):
        elements = [elements]

    # Element, wavelength, intensity
    file_path = pkg_resources.resource_filename('rascal', 'arc_lines/nist_clean.csv')

    lines = np.loadtxt(file_path,
                       delimiter=',',
                       dtype=">U12")
    
    # Mask elements
    mask = [(l[0] in elements) for l in lines]
    lines = lines[mask]
    
    # Filter wavelengths
    lines = filter_wavelengths(lines, min_wavelength, max_wavelength)

    # Calculate peak separation
    if min_distance > 0:
        distance_mask = filter_separation(lines[:,1].astype('float32'), min_distance)
    else:
        distance_mask = np.ones_like(lines[:,0].astype(bool))

    # Filter intensities
    if min_intensity > 0:
        intensity_mask = filter_intensity(lines, min_intensity)
    else:
        intensity_mask = np.ones_like(lines[:,0]).astype(bool)

    mask = distance_mask * intensity_mask
    
    elements = lines[:,0][mask]
    wavelengths = lines[:,1][mask].astype('float32')
    intensities = lines[:,2][mask].astype('float32')

    # Vacuum to air conversion 
    # Donald Morton (2000, ApJ. Suppl., 130, 403)
    s = 10000/wavelengths
    s2 = s**2

    n = 1 + 0.0000834254 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2)
    wavelengths /= n

    return elements, wavelengths, intensities


"""
def load_calibration_lines(elements,
                               min_wavelength=1000.,
                               max_wavelength=10000.):
        '''
        https://apps.dtic.mil/dtic/tr/fulltext/u2/a105494.pdf
        '''

        if isinstance(elements, str):
            elements = [elements]

        lines = []
        line_elements = []
        line_strengths = []

        for arc in elements:
            file_path = pkg_resources.resource_filename('rascal', 'arc_lines/{}.csv'.format(arc.lower()))

            with open(file_path, 'r') as f:

                f.readline()
                for l in f.readlines():
                    if l[0] == '#':
                        continue
                        
                    data = l.split(',')
                    if len(data) > 2:
                        line, strength, source = data[:3]
                        line_strengths.append(float(strength))
                    else:
                        line, source = data[:2]
                        line_strengths.append(0)
                    
                    lines.append(float(line))
                    line_elements.append(source)
       
        cal_lines = np.array(lines)
        cal_elements = np.array(line_elements)
        cal_strengths = np.array(line_strengths)

        # Get only lines within the requested wavelength
        mask = (cal_lines > min_wavelength) * (cal_lines < max_wavelength)
        return cal_elements[mask], cal_lines[mask], cal_strengths[mask]

def gauss(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))

def refine_peaks(spectrum, peaks, window_width=10):
    refined_peaks = []
    
    spectrum = np.array(spectrum)
    
    for peak in peaks:

        y = spectrum[int(peak)-window_width:int(peak)+window_width]
        y /= y.max()

        x = np.arange(len(y))

        n = len(x)                          
        mean = sum(x*y)/n                   
        sigma = sum(y*(x-mean)**2)/n

        try:
            popt, _ = curve_fit(gauss,x,y,p0=[1,mean,sigma])
            height, centre, width = popt

            if height < 0:
                continue
            refined_peaks.append(peak-window_width+centre)
        except RuntimeError:
            continue
            
    return np.array(refined_peaks)