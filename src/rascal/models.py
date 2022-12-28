#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model functions for spectral fitting

"""

import copy
from typing import Union

import numpy as np
import scipy.optimize


def polynomial(a: Union[list, np.ndarray], degree: int = 3):
    """
    Returns a lambda function which computes an nth order polynormal:

    f(x, a) = sum_i (a[degree-i] * x**i )

    Parameters
    ----------
    a: list
        Polynomial coefficients
    degree: int
        Polynomial degree

    Returns
    -------
    poly: callable function
        Polynomial function

    """

    assert len(a) == degree + 1

    def poly(x):
        t = a[-1]

        for i in range(1, int(degree + 1)):
            t += a[int(degree - i)] * x**i

        return t

    return poly


def poly_cost_function(
    a: Union[list, np.ndarray],
    x: Union[list, np.ndarray],
    y: Union[list, np.ndarray],
    degree: int,
):
    """
    Polynomial cost function. Returns the absolute
    difference between the target value and
    predicted values.

    Parameters
    ----------
    a: list
        Polynomial coefficients
    x: list
        Values to evaluate polynomial at
    y: list
        Target values for each x
    degree: int
        Polynomial degree

    Returns
    -------
    residual: list
        y - f(x)

    """

    f = polynomial(a, degree)
    return y - f(x)


def normalise_input(x, y):
    """
    Transforms inputs to have unit variance.

    Parameters
    ----------
    x: list
        list of values
    y: list
        list of values

    """

    x_scale = x.std()
    y_scale = y.std()

    x_norm = x / x_scale
    y_norm = y / y_scale

    return x_norm, y_norm


def robust_polyfit(x, y, degree=3, x0=None):
    """
    Perform a robust polyfit given a set of values (x,y).

    Specifically this function performs a least squares
    fit to the given data points using the robust Huber
    loss. Inputs are normalised prior to fitting.

    Parameters
    ----------
    x: list
        Data points
    y: list
        Target data to fit
    degree: int
        Polynomial degree to fit
    x0: list or None
        Initial coefficients

    Returns
    -------
    p: list
        Polynomial coefficients

    """

    x_n, y_n = normalise_input(x, y)

    p_init = copy.copy(x0)
    # Need to normalise the fit function too
    if p_init is not None:
        for i in range(0, degree):
            p_init[i] *= x.std() ** i

        p_init /= y.std() + 1e-9
    else:
        p_init = np.ones(degree + 1)

    res = scipy.optimize.least_squares(
        poly_cost_function,
        p_init,
        args=(x_n, y_n, degree),
        loss="huber",
        diff_step=1e-5,
    )
    p = res.x

    p *= y.std()

    # highest order first
    for i in range(0, degree):
        p[i] /= x.std() ** (degree - i)

    return p[::-1]
