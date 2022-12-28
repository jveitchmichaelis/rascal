#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Some uncategorised utility functions.

"""

import os
from collections import defaultdict
from typing import Union

import numpy as np
import pkg_resources
from scipy.optimize import curve_fit


def get_vapour_pressure(temperature: float):
    """
    Appendix A.I of https://emtoolbox.nist.gov/Wavelength/Documentation.asp

    Parameters
    ----------
    temperature: float
        In unit of Celcius

    """

    T = temperature + 273.15

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
    # omega takes Celcius
    omega = T + K9 / (T - K10)
    A = omega**2.0 + K1 * omega + K2
    B = K3 * omega**2.0 + K4 * omega + K5
    C = K6 * omega**2.0 + K7 * omega + K8
    X = -B + np.sqrt(B**2.0 - 4.0 * A * C)
    vapour_pressure = 1.0e6 * (2.0 * C / X) ** 4.0
    return vapour_pressure


def get_vapour_partial_pressure(
    relative_humidity: float, vapour_pressure: float
):
    """
    Appendix A.II of https://emtoolbox.nist.gov/Wavelength/Documentation.asp

    Parameters
    ----------
    relative_humidity: float
        In percentage point
    vapour_pressure: float
        In unit of Pascal

    """

    assert (
        relative_humidity >= 0.0 and relative_humidity <= 100.0
    ), "relative_humidity has to be between 0 and 100."

    partial_pressure = relative_humidity / 100.0 * vapour_pressure
    return partial_pressure


def edlen_refraction(
    wavelengths: float,
    temperature: float,
    pressure: float,
    vapour_partial_pressure: float,
):
    """
    Appendix A.IV of https://emtoolbox.nist.gov/Wavelength/Documentation.asp
    https://iopscience.iop.org/article/10.1088/0026-1394/35/2/8

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
    n_s = 1.0 + 1e-8 * (A + B / (130.0 - S) + C / (38.9 - S))
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
    wavelengths: Union[float, np.ndarray],
    temperature: float = 273.15,
    pressure: float = 101325.0,
    relative_humidity: float = 0.0,
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
    return np.array(wavelengths) / edlen_refraction(
        wavelengths, t, pressure, vapour_partial_pressure
    )


def air_to_vacuum_wavelength(
    wavelengths: Union[float, np.ndarray],
    temperature: float = 273.15,
    pressure: float = 101325.0,
    relative_humidity: float = 0.0,
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


def filter_wavelengths(
    lines: Union[list, np.ndarray],
    min_atlas_wavelength: float,
    max_atlas_wavelength: float,
):
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


def filter_distance(wavelengths: float, min_distance: float = 0.0):
    """
    Filters a wavelength list by a distance threshold.

    Parameters
    ----------

    wavelengths: list
        List of input wavelengths
    min_distance: int
        Separation threshold, Ansgtrom

    Returns
    -------

    distance_mask: list
        Mask of values which satisfy the distance criteria

    """

    left_dists = np.zeros_like(wavelengths)
    left_dists[1:] = wavelengths[1:] - wavelengths[:-1]

    right_dists = np.zeros_like(wavelengths)
    right_dists[:-1] = wavelengths[1:] - wavelengths[:-1]
    distances = np.minimum(right_dists, left_dists)
    distances[0] = right_dists[0]
    distances[-1] = left_dists[-1]

    distance_mask = np.abs(distances) >= min_distance

    return distance_mask


def filter_intensity(
    elements: Union[list, np.ndarray],
    lines: Union[list, np.ndarray],
    min_intensity: float = None,
):
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
            "min_intensity has to be in, float of list/array the same "
            + f"size as the elements. min_intensity is {min_intensity} and "
            + f"elements is {elements}."
        )

        min_intensity_dict = {}

        for i, e in enumerate(elements):

            min_intensity_dict[e] = min_intensity[i]

    else:

        raise ValueError(
            "min_intensity has to be in, float of list/array the same "
            + f"size as the elements. min_intensity is {min_intensity} and "
            + f"elements is {elements}."
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
    linelist: str = "nist",
    min_atlas_wavelength: float = 3000.0,
    max_atlas_wavelength: float = 15000.0,
    min_intensity: float = 5.0,
    min_distance: float = 0.0,
    brightest_n_lines: int = None,
    vacuum: bool = False,
    pressure: float = 101325.0,
    temperature: float = 273.15,
    relative_humidity: float = 0.0,
):
    """
    Get calibration lines from the standard NIST atlas to screen.
    Rascal provides a cleaned set of NIST lines that can be used for
    general purpose calibration. It is recommended however that for
    repeated and robust calibration, the user should specify an
    instrument-specific atlas.

    Provide a wavelength range suitable to your calibration source. You
    can also specify a minimum intensity that corresponds to the values
    listed in the NIST tables.

    If you want air wavelengths (default), you can provide atmospheric
    conditions for your system. In most cases the default values of
    standard temperature and pressure should be sufficient.


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
    min_intensity: float
        Minimum intensity to search, per NIST
    min_distance: float
        All ines within this distance from other lines are treated
        as unresolved, all of them get removed from the list.
    brightest_n_lines: int
        Only return the n brightest lines
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
    element_list: list
        Emission line names corresponding to the parameters specified
    wavelength_list: list
        Emission line wavelengths corresponding to the parameters specified
    intensity_list: list
        Emission line intensities corresponding to the parameters specified

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
                "Unknown string is provided as linelist: {linelist}."
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

    # Calculate peak distance
    if min_distance > 0:
        distance_mask = filter_distance(wavelength_list, min_distance)
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


def print_calibration_lines(
    elements: Union[list, np.ndarray] = [],
    linelist: str = "nist",
    min_atlas_wavelength: float = 3000.0,
    max_atlas_wavelength: float = 15000.0,
    min_intensity: float = 5.0,
    min_distance: float = 0.0,
    brightest_n_lines: int = None,
    vacuum: bool = False,
    pressure: float = 101325.0,
    temperature: float = 273.15,
    relative_humidity: float = 0.0,
):
    """
    Print calibration lines from the standard NIST atlas to screen.
    Rascal provides a cleaned set of NIST lines that can be used for
    general purpose calibration. It is recommended however that for
    repeated and robust calibration, the user should specify an
    instrument-specific atlas.

    Provide a wavelength range suitable to your calibration source. You
    can also specify a minimum intensity that corresponds to the values
    listed in the NIST tables.

    If you want air wavelengths (default), you can provide atmospheric
    conditions for your system. In most cases the default values of
    standard temperature and pressure should be sufficient.

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
    min_distance: int
        All ines within this distance from other lines are treated
        as unresolved, all of them get removed from the list.
    brightest_n_lines: int
        Only return the n brightest lines
    vacuum: bool
        Return vacuum wavelengths
    pressure: float
        Atmospheric pressure, Pascal
    temperature: float
        Temperature in Kelvin, default room temp
    relative_humidity: float
        Relative humidity, percent

    """

    elements, lines, intensities = load_calibration_lines(
        elements=elements,
        linelist=linelist,
        min_atlas_wavelength=min_atlas_wavelength,
        max_atlas_wavelength=max_atlas_wavelength,
        min_intensity=min_intensity,
        min_distance=min_distance,
        brightest_n_lines=brightest_n_lines,
        vacuum=vacuum,
        pressure=pressure,
        temperature=temperature,
        relative_humidity=relative_humidity,
    )

    output = ""
    for element, line, intensity in zip(elements, lines, intensities):

        output += (
            f"Element: {element} at {line} Angstrom with intensity "
            + f"{intensity}.{os.linesep}"
        )

    print(output)


def gauss(x: Union[float, np.ndarray], a: float, x0: float, sigma: float):
    """
    1D Gaussian

    Parameters
    ----------
    x: float, np.ndarray
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

    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2 + 1e-16))


def refine_peaks(
    spectrum: Union[list, np.ndarray],
    peaks: Union[list, np.ndarray],
    window_width: float = 10.0,
):
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


def _derivative(p: Union[list, np.ndarray]):
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


def get_duplicate_indices(x: Union[list, np.ndarray]):
    """
    Return the duplicate indices of the input list, x

    Parameters
    ----------
    x: list

    Returns
    -------
    duplicates: list(int)
        Indices of duplicate elements
    """
    # NB iteration_utilities is much faster, but doesn't return indices

    dic = defaultdict(list)

    for idx, element in enumerate(x):
        dic[tuple(element)].append(idx)

    indices = []
    for key in dic:
        if len(dic[key]) > 1:
            indices.extend(dic[key])

    return indices


def _clean_matches(x: list, user_lines: list = []):
    """
    Clean a list of atlas match groups given the following
    constraints:

    1. If there are duplicate match lists, None will be appended to
    provide for the case when neither match is appropriate.
    2. If user lines are provided then we check to see if there
    are any match groups that contain that line. If only one does
    then the other lines in that group are removed.
    3. All match groups should have only unique values

    Parameters
    ----------
    x: list of list(float)

    Returns
    -------
    cleaned: list(set(float))
        A list of sets of matched atlas lines, cleaned as per the
        documentation above.

    """

    # If we enforce values that must appear
    # and they only appear in a single
    # match.
    for value in user_lines:
        val_idx = -1
        count = 0
        for idx, element in enumerate(x):
            if value in element:
                if count > 0:
                    count = 0
                    break
                else:
                    val_idx = idx
                    count = 1

        if count == 1:
            x[val_idx] = [value]

    # Add None to duplicates
    for idx in get_duplicate_indices(x):
        if len(x[idx]) > 0:
            x[idx].append(-1)

    return [set(i) for i in x]


def _make_unique_permutation(x: list, empty_val: Union[int, float] = -1):
    """
    Return all permutations of the list of inputs, subject
    to the constraint that all permutations only contain
    unique values. It is guaranteed that the output
    permutations have the same length as the input x.

    Parameters
    ----------
    x: list of list(float)
        Input list of values to permute. Each value must be a list or
        iterable.
    empty_val: Any
        Special placeholder to use if the input list has an empty element

    Returns
    -------
    permutations: list(list(float))
        A list of sets of matched atlas lines, cleaned as per the
        documentation above.
    """

    permutations = [[]]

    for input_list in x:

        new_permutations = []

        for permutation in permutations:

            if len(input_list) == 0:
                permutation.append(empty_val)
            else:
                for element in input_list:

                    if element not in permutation or element is empty_val:
                        new_permutation = list(permutation)
                        new_permutation.append(element)

                        new_permutations.append(new_permutation)

        if len(new_permutations) > 0:
            permutations = new_permutations

    return permutations
