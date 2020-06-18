import numpy as np
import random
from scipy.optimize import curve_fit
from scipy import asarray as ar
from numpy import exp
import pkg_resources
"""

def vacuum_to_air(wavelengths):
    s = 10000 / wavelengths
    s2 = s**2

    n = 1 + 0.0000834254 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2)
    wavelengths /= n

    return

def load_calibration_lines(elements,
                           min_atlas_wavelength=1000.,
                           max_atlas_wavelength=10000.,
                           include_second_order=False,
                           relative_intensity=1000,
                           pressure=101325,
                           temperature=273.15):
    '''
    https://apps.dtic.mil/dtic/tr/fulltext/u2/a105494.pdf
    '''

    if isinstance(elements, str):
        elements = [elements]

    lines = []
    line_elements = []
    line_strengths = []

    for arc in elements:
        file_path = pkg_resources.resource_filename(
            'rascal', 'arc_lines/{}.csv'.format(arc.lower()))

        with open(file_path, 'r') as f:

            f.readline()
            for l in f.readlines():
                if l[0] == '#':
                    continue

                data = l.rstrip().split(',')
                if len(data) > 2:
                    line, strength, source = data[:3]
                    line_strengths.append(float(strength))
                else:
                    line, source = data[:2]
                    line_strengths.append(0)

                lines.append(float(line))
                line_elements.append(source)

                if include_second_order:
                    wavelength = 2 * lines[-1]

                    lines.append(wavelength)
                    line_elements.append(line_elements[-1] + "_2")
                    line_strengths.append(line_strengths[-1])

    cal_lines = np.array(lines)
    cal_elements = np.array(line_elements)
    cal_strengths = np.array(line_strengths)

    # Get only lines within the requested wavelength
    mask = (cal_lines > min_atlas_wavelength) * (
        cal_lines < max_atlas_wavelength) * (cal_strengths >
                                             relative_intensity)
    return cal_elements[mask], cal_lines[mask], cal_strengths[mask]

"""


def pressure_temperature_to_density(pressure, temperature):
    '''
    Get the air density in unit of amagat from pressure in Pa and temperature
    in K
    '''

    density = (pressure / 101325) * (273.15 / temperature)

    return density


def vacuum_to_air_wavelength(wavelengths, density=1.0):
    '''
    Calculate refractive index of air from Cauchy formula.

    Input: wavelength in Angstrom, density of air in amagat (relative to STP,
    e.g. ~10% decrease per 1000m above sea level).

    refracstp = (n-1) * 1E6
    return n = refracstp / 1E6 + 1

    The IAU standard for conversion from air to vacuum wavelengths is given
    in Morton (1991, ApJS, 77, 119). For vacuum wavelengths (VAC) in
    Angstroms, convert to air wavelength (AIR) via:

    AIR = VAC / (1.0 + n * (2.735182E-4 + 131.4182 / VAC^2 + 2.76249E8 / VAC^4))
    '''

    wl = np.array(wavelengths)
    wl_inv = 1. / wl

    refracstp = 1. + density * (0.000272643 + 131.4182 * wl_inv**2 + 2.76249E8 * wl_inv**4)
    air_wavelengths = refracstp * wl

    return air_wavelengths


def filter_wavelengths(lines, min_atlas_wavelength, max_wavlength):
    wavelengths = lines[:,1].astype(np.float32)
    wavelength_mask = (wavelengths >= min_atlas_wavelength) * (wavelengths <= max_wavlength)

    return lines[wavelength_mask]


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


def filter_intensity(lines, min_intensity=0):
    out = []
    for line in lines:
        _, _, intensity = line
        if float(intensity) >= min_intensity:
            out.append(True)
        else:
            out.append(False)

    return np.array(out).astype(bool)


def load_calibration_lines(elements=[],
                           min_atlas_wavelength=3000,
                           max_atlas_wavelength=15000,
                           min_intensity=10,
                           min_distance=10,
                           vacuum=False,
                           pressure=101325,
                           temperature=273.15):

    if isinstance(elements, str):
        elements = [elements]

    # Element, wavelength, intensity
    file_path = pkg_resources.resource_filename('rascal',
                                                'arc_lines/nist_clean.csv')

    lines = np.loadtxt(file_path, delimiter=',', dtype=">U12")

    # Mask elements
    mask = [(l[0] in elements) for l in lines]
    lines = lines[mask]

    # Filter wavelengths
    lines = filter_wavelengths(lines, min_atlas_wavelength,
                               max_atlas_wavelength)

    # Filter intensities
    if min_intensity > 0:
        intensity_mask = filter_intensity(lines, min_intensity)
    else:
        intensity_mask = np.ones_like(lines[:, 0]).astype(bool)

    elements = lines[:, 0][intensity_mask]
    wavelengths = lines[:, 1][intensity_mask].astype('float32')
    intensities = lines[:, 2][intensity_mask].astype('float32')

    # Calculate peak separation
    if min_distance > 0:
        distance_mask = filter_separation(wavelengths,
                                          min_distance)
    else:
        distance_mask = np.ones_like(wavelengths.astype(bool))

    elements = elements[distance_mask]
    wavelengths = wavelengths[distance_mask]
    intensities = intensities[distance_mask]

    # Vacuum to air conversion
    if not vacuum:
        density = pressure_temperature_to_density(pressure, temperature)
        wavelengths = vacuum_to_air_wavelength(wavelengths, density)

    return elements, wavelengths, intensities


def gauss(x, a, x0, sigma):
    return a * exp(-(x - x0)**2 / (2 * sigma**2))


def refine_peaks(spectrum, peaks, window_width=10):
    refined_peaks = []

    spectrum = np.array(spectrum)

    for peak in peaks:

        y = spectrum[int(peak) - window_width:int(peak) + window_width]
        y /= y.max()

        x = np.arange(len(y))

        n = len(x)
        mean = sum(x * y) / n
        sigma = sum(y * (x - mean)**2) / n

        try:
            popt, _ = curve_fit(gauss, x, y, p0=[1, mean, sigma])
            height, centre, _ = popt

            if height < 0:
                continue

            if centre > len(spectrum) or centre < 0:
                continue

            refined_peaks.append(peak - window_width + centre)
        except RuntimeError:
            continue

    refined_peaks = np.array(refined_peaks)
    mask = (refined_peaks > 0) & (refined_peaks < len(spectrum))

    return refined_peaks[mask]


def derivative(p):
    derv = []
    for i in range(1, len(p)):
        derv.append(i * p[i])
    return derv
