import numpy as np
from scipy.optimize import curve_fit
from numpy import exp
import pkg_resources


def get_vapour_pressure(temperature):
    """
    Appendix A.I of https://emtoolbox.nist.gov/Wavelength/Documentation.asp
    """
    K1 = 1.16705214528E+03
    K2 = -7.24213167032E+05
    K3 = -1.70738469401E+01
    K4 = 1.20208247025E+04
    K5 = -3.23255503223E+06
    K6 = 1.49151086135E+01
    K7 = -4.82326573616E+03
    K8 = 4.05113405421E+05
    K9 = -2.38555575678E-01
    K10 = 6.50175348448E+02
    omega = temperature + K9 / (temperature - K10)
    A = omega**2. + K1 * omega + K2
    B = K3 * omega**2. + K4 * omega + K5
    C = K6 * omega**2. + K7 * omega + K8
    X = -B + np.sqrt(B**2. - 4 * A * C)
    vapour_pressure = 10**6. * (2. * C / X)**4.
    return vapour_pressure


def get_vapour_partial_pressure(relative_humidity, vapour_pressure):
    """
    Appendix A.II of https://emtoolbox.nist.gov/Wavelength/Documentation.asp
    """
    partial_pressure = relative_humidity / 100. * vapour_pressure
    return partial_pressure


def edlen_refraction(wavelengths, temperature, pressure,
                     vapour_partial_pressure):
    """
    Appendix A.IV of https://emtoolbox.nist.gov/Wavelength/Documentation.asp
    """
    A = 8342.54
    B = 2406147.
    C = 15998.
    D = 96095.43
    E = 0.601
    F = 0.00972
    G = 0.003661
    S = 1. / np.array(wavelengths)**2.
    n_s = 1. + 1E-8 * (A + B / (130 - S) + C / (38.9 - S))
    X = (1. + 1E-8 *
         (E - F *
          (temperature - 273.15)) * pressure) / (1. + G *
                                                 (temperature - 273.15))
    n_tp = 1. + pressure * (n_s - 1.) * X / D
    n = n_tp - 1E-10 * (292.75 / temperature) * (
        3.7345 - 0.0401 * S) * vapour_partial_pressure
    return n


def vacuum_to_air_wavelength(wavelengths,
                             temperature=273.15,
                             pressure=101325,
                             relative_humidity=0):
    """

    The conversion follows the Modified Edlén Equations

    https://emtoolbox.nist.gov/Wavelength/Documentation.asp

    pressure drops by ~10% per 1000m above sea level
    temperature depends heavily on the location
    relative humidity is between 0-100, depends heavily on the location


    Parameters
    ----------
    wavelengths: float or numpy.array
        Wavelengths in vacuum
    temperature: float
        In unit of Kelvin
    pressure: float
        In unit of Pa
    relative_humidity: float
        Unitless in percentage (i.e. 0 - 100)

    Returns
    -------
    air wavelengths: float or numpy.array
        The wavelengths in air given the condition
    """

    vapour_pressure = get_vapour_pressure(temperature)
    vapour_partial_pressure = get_vapour_partial_pressure(
        relative_humidity, vapour_pressure)
    return np.array(wavelengths) / edlen_refraction(
        wavelengths, temperature, pressure, vapour_partial_pressure)


def filter_wavelengths(lines, min_atlas_wavelength, max_atlas_wavelength):
    """
    Filters a wavelength list to a minimum and maximum range.

    Parameters
    ----------

    lines: list
        List of input wavelengths
    min_atlas_wavelength: int
        Min wavelength, Ansgtrom
    max_atlas_wavelength: int
        Max wavelength, Angstrom

    Returns
    -------

    lines: list
        Filtered wavelengths within specified range limit

    """

    wavelengths = lines[:, 1].astype(np.float32)
    wavelength_mask = (
        (wavelengths >= min_atlas_wavelength) &
        (wavelengths <= max_atlas_wavelength))

    return lines[wavelength_mask]


def filter_separation(wavelengths, min_separation=0):
    """
    Filters a wavelength list by a separation threshold.

    Parameters
    ----------

    wavelengths: list
        List of input wavelengths
    min_separation: int
        Separation threshold, Ansgtrom

    Returns
    -------

    distance_mask: list
        Mask of values which satisfy the separation criteria

    """

    left_dists = np.zeros_like(wavelengths)
    left_dists[1:] = wavelengths[1:] - wavelengths[:-1]

    right_dists = np.zeros_like(wavelengths)
    right_dists[:-1] = wavelengths[1:] - wavelengths[:-1]
    distances = np.minimum(right_dists, left_dists)
    distances[0] = right_dists[0]
    distances[-1] = left_dists[-1]

    distance_mask = np.abs(distances) >= min_separation

    return distance_mask


def filter_intensity(lines, min_intensity=0):
    """
    Filters a line list by an intensity threshold

    Parameters
    ----------

    lines: list[tuple (str, float, float)]
        A list of input lines where the 2nd parameter
        is intensity
    min_intensity: int
        Intensity threshold

    Returns
    -------

    lines: list
        Filtered line list

    """

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
                           pressure=101325.,
                           temperature=273.15,
                           relative_humidity=0.):
    """
    Load calibration lines from the standard NIST atlas.
    Rascal provides a cleaned set of NIST lines that can be
    used for general purpose calibration. It is recommended
    however that for repeated and robust calibration, the
    user should specify an instrument-specific atlas.

    Provide a wavelength range suitable to your calibration
    source. You can also specify a minimum intensity that
    corresponds to the values listed in the NIST tables.

    If you want air wavelengths (default), you can provide
    atmospheric conditions for your system. In most cases
    the default values of standard temperature and pressure
    should be sufficient.


    Parameters
    ----------
    elements: list
        List of short element names, e.g. He as per NIST
    min_atlas_wavelength: int
        Minimum wavelength to search, Angstrom
    max_atlas_wavelength: int
        Maximum wavelength to search, Angstrom
    min_intensity: int
        Minimum intensity to search, per NIST
    max_intensity: int
        Maximum intensity to search, per NIST
    vacuum: bool
        Return vacuum wavelengths
    pressure: float
        Atmospheric pressure, Pascal
    temperature: float
        Temperature in Kelvin, default room temp
    relative_humidity: float
        Relative humidity, percent

    Returns
    -------
    out: list
        Emission lines corresponding to the parameters specified
    """

    if isinstance(elements, str):
        elements = [elements]

    # Element, wavelength, intensity
    file_path = pkg_resources.resource_filename('rascal',
                                                'arc_lines/nist_clean.csv')

    lines = np.loadtxt(file_path, delimiter=',', dtype=">U12")

    # Mask elements
    mask = [(li[0] in elements) for li in lines]
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
        distance_mask = filter_separation(wavelengths, min_distance)
    else:
        distance_mask = np.ones_like(wavelengths.astype(bool))

    elements = elements[distance_mask]
    wavelengths = wavelengths[distance_mask]
    intensities = intensities[distance_mask]

    # Vacuum to air conversion
    if not vacuum:
        wavelengths = vacuum_to_air_wavelength(wavelengths, temperature,
                                               pressure, relative_humidity)

    return elements, wavelengths, intensities


def gauss(x, a, x0, sigma):
    """
    1D Gaussian

    Parameters
    ----------
    x:
        value or values to evaluate the Gaussian at
    a: float
        Magnitude
    x0: float
        Gaussian centre
    sigma: float
        Standard deviation (spread)

    Returns
    -------
    out: list
        The Gaussian function evaluated at provided x
    """

    return a * exp(-(x - x0)**2 / (2 * sigma**2 + 1e-9))


def refine_peaks(spectrum, peaks, window_width=10, distance=None):
    """
    Refine peak locations in a spectrum from a set of initial estimates.

    This function attempts to fit a Gaussian to each peak in the provided
    list. It returns a list of sub-pixel refined peaks. If two peaks are
    very close, they can be refined to the same location. In this case
    only one of the peaks will be returned - i.e. this function will return
    a unique set of peak locations.

    Parameters
    ----------
    spectrum: list
        Input spectrum (list of intensities)
    peaks: list
        A list of peak locations in pixels
    window_width: int
        Size of window to consider in fit either side of
        initial peak location

    Returns
    -------
    refined_peaks: list
        A list of refined peak locations

    """

    refined_peaks = []

    spectrum = np.array(spectrum)
    length = len(spectrum)

    for peak in peaks:

        y = spectrum[max(0,
                         int(peak) -
                         window_width):min(int(peak) + window_width, length)]
        y /= np.nanmax(y)
        x = np.arange(len(y))

        mask = np.isfinite(y) & ~np.isnan(y)
        n = np.sum(mask)

        if n == 0:
            continue

        mean = np.sum(x * y) / n
        sigma = np.sum(y * (x - mean)**2) / n

        try:

            popt, _ = curve_fit(gauss, x[mask], y[mask], p0=[1, mean, sigma])
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

    # Remove peaks that are within rounding errors from each other from the
    # curve_fit
    distance_mask = np.isclose(refined_peaks[:-1], refined_peaks[1:])
    distance_mask = np.insert(distance_mask, 0, False)

    return refined_peaks[mask & ~distance_mask]


def _derivative(p):
    """
    Compute the derivative of a polynomial function.

    Parameters
    ----------
    p: list
        Polynomial coefficients, in increasing order
        (e.g. 0th coefficient first)

    Returns
    -------
    derv: list
        Derivative coefficients, i * p[i]

    """
    derv = []
    for i in range(1, len(p)):
        derv.append(i * p[i])
    return derv
