#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is for generating synthetic arcs to enable testing.

"""

from typing import Union

import numpy as np

from . import models


class SyntheticSpectrum:
    """
    Creates a synthetic spectrum generator which, given a suitable model,
    outputs the expected pixel locations of input wavelengths. This should be
    mainly for testing.

    """

    def __init__(
        self,
        coefficients: Union[list, np.ndarray] = None,
        min_wavelength: float = 200.0,
        max_wavelength: float = 1200.0,
    ):
        """

        Parameters
        ----------
        coefficients: list
            Coefficients for the model
        min_wavelength: float
            Minimum wavelength limit
        max_wavelength: float
            Maximum wavelength limit

        """

        # Default is approx. range of Silicon
        self.set_wavelength_limit(float(min_wavelength), float(max_wavelength))

        if coefficients is not None:

            self.coefficients = coefficients
            self.set_model(self.coefficients)

        else:

            self.model = None
            self.degree = None

    def set_model(self, coefficients: Union[list, np.ndarray]):
        """
        Set the model to fit

        Parameters
        ----------
        coefficients: list, np.ndarray
            polynomial coefficients, from the lowest to the highest order.

        """

        self.degree = len(coefficients) - 1
        self.model = models.polynomial(a=coefficients, degree=self.degree)

    def set_wavelength_limit(
        self, min_wavelength: float = None, max_wavelength: float = None
    ):
        """
        Set a wavelength filter for the 'get_pixels' function.

        """

        if (
            not isinstance(min_wavelength, float)
            and min_wavelength is not None
        ):

            raise TypeError(
                "Please provide a numeric value or None to "
                "retain the min_wavelength."
            )

        else:

            # Placeholder Min/Max
            if min_wavelength is not None:

                new_min_wavelength = min_wavelength

            else:

                new_min_wavelength = self.min_wavelength

        if (
            not isinstance(max_wavelength, float)
            and max_wavelength is not None
        ):

            raise TypeError(
                "Please provide a numeric value or None to "
                "retain the max_wavelength."
            )

        else:
            if max_wavelength is not None:

                new_max_wavelength = max_wavelength

            else:

                new_max_wavelength = self.max_wavelength

        # Check if Max > Min
        if new_max_wavelength > new_min_wavelength:

            self.min_wavelength = new_min_wavelength
            self.max_wavelength = new_max_wavelength

        else:

            raise RuntimeError(
                "Minimum wavelength cannot be larger than "
                "the maximum wavelength."
            )

    def get_pixels(self, wavelengths: Union[list, np.ndarray]):
        """
        Returns a list of pixel locations for the wavelengths provided

        """

        if not isinstance(wavelengths, (list, np.ndarray)):

            raise TypeError("Please provide a list or an numpy array.")

        wavelengths = np.array(wavelengths)
        wavelengths = wavelengths[wavelengths >= self.min_wavelength]
        wavelengths = wavelengths[wavelengths <= self.max_wavelength]

        # Constant function y = c
        if self.degree == 0:
            pixels = self.coefficients[0] * np.ones(len(wavelengths))
        # Linear function y = mx + c
        elif self.degree == 1:
            # x = (y - c) / m
            pixels = (wavelengths - self.coefficients[0]) / self.coefficients[
                1
            ]
        # High order polynomials
        else:
            _p = np.poly1d(self.coefficients)
            pixels = [(_p - w).roots for w in wavelengths]

        return pixels, wavelengths
