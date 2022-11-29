import os

import numpy as np
from scipy.optimize import curve_fit
import pkg_resources


def get_vapour_pressure(temperature):
    """
    Appendix A.I of https://emtoolbox.nist.gov/Wavelength/Documentation.asp
    """
    K1 = 1.16705214528e03
    K2 = -7.24213167032e05
    K3 = -1.70738469401e01
    K4 = 1.20208247025e04
    K5 = -3.23255503223e06
    K6 = 1.49151086135e01
    K7 = -4.82326573616e03
    K8 = 4.05113405421e05
    K9 = -2.38555575678e-01
    K10 = 6.50175348448e02
    omega = temperature + K9 / (temperature - K10)
    A = omega**2.0 + K1 * omega + K2
    B = K3 * omega**2.0 + K4 * omega + K5
    C = K6 * omega**2.0 + K7 * omega + K8
    X = -B + np.sqrt(B**2.0 - 4 * A * C)
    vapour_pressure = 10**6.0 * (2.0 * C / X) ** 4.0
    return vapour_pressure


def get_vapour_partial_pressure(relative_humidity, vapour_pressure):
    """
    Appendix A.II of https://emtoolbox.nist.gov/Wavelength/Documentation.asp
    """
    partial_pressure = relative_humidity / 100.0 * vapour_pressure
    return partial_pressure


def edlen_refraction(
    wavelengths, temperature, pressure, vapour_partial_pressure
):
    """
    Appendix A.IV of https://emtoolbox.nist.gov/Wavelength/Documentation.asp

    Parameters
    ----------
    wavelengths: float
        In unit of Angstrom
    temperature: float
        In unit of Celcius
    pressure: float
        In unit of Pascal

    """

    # Convert to micron for computing variable S
    w = np.array(wavelengths) / 1e4

    t = temperature
    T = temperature + 273.15

    A = 8342.54
    B = 2406147.0
    C = 15998.0
    D = 96095.43
    E = 0.601
    F = 0.00972
    G = 0.003661
    S = np.array(w) ** -2.0
    n_s = 1.0 + 1e-8 * (A + B / (130 - S) + C / (38.9 - S))
    X = (1.0 + 1e-8 * (E - F * t) * pressure) / (1.0 + G * t)
    n_tp = 1.0 + pressure * (n_s - 1.0) * X / D
    n = (
        n_tp
        - 1e-10
        * (292.75 / T)
        * (3.7345 - 0.0401 * S)
        * vapour_partial_pressure
    )
    return n


def vacuum_to_air_wavelength(
    wavelengths, temperature=273.15, pressure=101325, relative_humidity=0
):
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

    # Convert to celcius
    t = temperature - 273.15

    vapour_pressure = get_vapour_pressure(t)
    vapour_partial_pressure = get_vapour_partial_pressure(
        relative_humidity, vapour_pressure
    )
    return np.array(wavelengths) / edlen_refraction(
        wavelengths, t, pressure, vapour_partial_pressure
    )


def air_to_vacuum_wavelength(
    wavelengths, temperature=273.15, pressure=101325, relative_humidity=0
):
    """
    The conversion follows the Modified Edlén Equations
    https://emtoolbox.nist.gov/Wavelength/Documentation.asp
    https://iopscience.iop.org/article/10.1088/0026-1394/35/2/8
    pressure drops by ~10% per 1000m above sea level
    temperature depends heavily on the location
    relative humidity is between 0-100, depends heavily on the location
    Parameters
    ----------
    wavelengths: float or numpy.array
        Wavelengths in vacuum in unit of Angstrom
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

    # Convert to celcius
    t = temperature - 273.15

    # get_vapour_pressure takes temperature in Celcius
    vapour_pressure = get_vapour_pressure(t)
    vapour_partial_pressure = get_vapour_partial_pressure(
        relative_humidity, vapour_pressure
    )
    return np.array(wavelengths) * edlen_refraction(
        wavelengths, t, pressure, vapour_partial_pressure
    )


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

    wavelengths = lines[:, 1].astype(np.float64)

    _, index, _ = np.unique(wavelengths, return_counts=True, return_index=True)

    wavelength_mask = (wavelengths[index] >= min_atlas_wavelength) & (
        wavelengths[index] <= max_atlas_wavelength
    )

    return lines[index][wavelength_mask]


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


def filter_intensity(elements, lines, min_intensity=None):
    """
    Filters a line list by an intensity threshold
    Parameters
    ----------
    lines: list[tuple (str, float, float)]
        A list of input lines where 1st parameter is the name of the element
        the 2nd parameter is the wavelength, the 3rd is the intensities
    min_intensity: float
        Intensity threshold
    Returns
    -------
    lines: list
        Filtered line list
    """

    if min_intensity is None:

        min_intensity_dict = {}

        for i, e in enumerate(elements):

            min_intensity_dict[e] = 0.0

    elif isinstance(min_intensity, (int, float)):

        min_intensity_dict = {}

        for i, e in enumerate(elements):

            min_intensity_dict[e] = float(min_intensity)

    elif isinstance(min_intensity, (list, np.ndarray)):

        assert len(min_intensity) == len(elements), (
            "min_intensity has to be in, float of list/array "
            "the same size as the elements. min_intensity is {}"
            "and elements is {}.".format(min_intensity, elements)
        )

        min_intensity_dict = {}

        for i, e in enumerate(elements):

            min_intensity_dict[e] = min_intensity[i]

    else:

        raise ValueError(
            "min_intensity has to be in, float of list/array "
            "the same size as the elements. min_intensity is {}"
            "and elements is {}.".format(min_intensity, elements)
        )

    out = []

    for line in lines:
        element = line[0]
        intensity = float(line[2])

        if intensity >= min_intensity_dict[element]:
            out.append(True)
        else:
            out.append(False)

    return np.array(out).astype(bool)


def load_calibration_lines(
    elements=[],
    min_atlas_wavelength=3000.0,
    max_atlas_wavelength=15000.0,
    min_intensity=10,
    min_distance=10,
    vacuum=False,
    pressure=101325.0,
    temperature=273.15,
    relative_humidity=0.0,
    linelist="nist",
    brightest_n_lines=None,
):
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
    linelist: str
        Either 'nist' to use the default lines or path to a linelist file.
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
    if isinstance(linelist, str):
        if linelist.lower() == "nist":
            file_path = pkg_resources.resource_filename(
                "rascal", "arc_lines/nist_clean.csv"
            )
            lines = np.loadtxt(file_path, delimiter=",", dtype=">U12")
        elif os.path.exists(linelist):
            lines = np.loadtxt(linelist, delimiter=",", dtype=">U12")
        else:
            raise ValueError(
                f"Unknown string is provided as linelist: {linelist}."
            )
    else:
        raise ValueError("Please provide a valid format of line list.")

    # Mask elements
    mask = [(li[0] in elements) for li in lines]
    lines = lines[mask]

    # update the wavelength limit
    if not vacuum:
        min_atlas_wavelength, max_atlas_wavelength = air_to_vacuum_wavelength(
            (min_atlas_wavelength, max_atlas_wavelength),
            temperature,
            pressure,
            relative_humidity,
        )

    # Filter wavelengths
    lines = filter_wavelengths(
        lines, min_atlas_wavelength, max_atlas_wavelength
    )

    # Filter intensities
    if isinstance(min_intensity, (float, int, list, np.ndarray)):
        intensity_mask = filter_intensity(elements, lines, min_intensity)
    else:
        intensity_mask = np.ones_like(lines[:, 0]).astype(bool)

    element_list = lines[:, 0][intensity_mask]
    wavelength_list = lines[:, 1][intensity_mask].astype("float64")
    intensity_list = lines[:, 2][intensity_mask].astype("float64")

    if brightest_n_lines is not None:
        to_keep = np.argsort(np.array(intensity_list))[::-1][
            :brightest_n_lines
        ]
        element_list = element_list[to_keep]
        intensity_list = intensity_list[to_keep]
        wavelength_list = wavelength_list[to_keep]

    # Calculate peak separation
    if min_distance > 0:
        distance_mask = filter_separation(wavelength_list, min_distance)
    else:
        distance_mask = np.ones_like(wavelength_list.astype(bool))

    element_list = element_list[distance_mask]
    wavelength_list = wavelength_list[distance_mask]
    intensity_list = intensity_list[distance_mask]

    # Vacuum to air conversion
    if not vacuum:
        wavelength_list = vacuum_to_air_wavelength(
            wavelength_list, temperature, pressure, relative_humidity
        )

    return element_list, wavelength_list, intensity_list


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

    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2 + 1e-9))


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

        y = spectrum[
            max(0, int(peak) - window_width) : min(
                int(peak) + window_width, length
            )
        ]
        y /= np.nanmax(y)
        x = np.arange(len(y))

        mask = np.isfinite(y) & ~np.isnan(y)
        n = np.sum(mask)

        if n == 0:
            continue

        mean = np.sum(x * y) / n
        sigma = np.sum(y * (x - mean) ** 2) / n

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
