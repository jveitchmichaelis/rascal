import scipy.optimize
import numpy as np
"""
Model functions for spectral fitting
"""


def polynomial(a, degree=3):
    """
    Returns a lambda function which computes an nth order polynormal:

    f(x, a) = sum_i (a[degree-i] * x**i )
    """

    assert (len(a) == degree + 1)

    def poly(x):
        t = a[-1]

        for i in range(1, int(degree + 1)):
            t += a[int(degree - i)] * x**i

        return t

    return poly


def poly_cost_function(a, x, y, degree):
    f = polynomial(a, degree)
    return y - f(x)


def normalise_input(x, y):
    """
    Transforms inputs to have unit variance
    """
    x_scale = x.std()
    y_scale = y.std()

    x_norm = x / x_scale
    y_norm = y / y_scale

    return x_norm, y_norm


def robust_polyfit(x, y, degree=3, x0=None, bounds=None):

    x_n, y_n = normalise_input(x, y)

    # Need to normalise the fit function too
    if x0 is not None:
        for i in range(0, degree):
            x0[i] *= x.std()**(i)

        x0 /= y.std()
    else:
        x0 = np.ones(degree + 1)

    if bounds is None:
        bounds = np.inf * np.ones(degree + 1)

    assert len(bounds) > 0

    res = scipy.optimize.least_squares(poly_cost_function,
                                       x0,
                                       args=(x_n, y_n, degree),
                                       loss='huber',
                                       diff_step=1e-5)
    p = res.x

    p *= y.std()

    # highest order first
    for i in range(0, degree):
        p[i] /= x.std()**(degree - i)

    return p[::-1]


def pprint_coefficients(coeffs):
    expr = "{} ".format(round(coeffs[0], 3))

    if len(coeffs) > 1:
        for i, c in enumerate(coeffs[1:]):
            if i == 0:
                expr += "+ {}*x".format(round(c, 3))
            else:
                expr += "+ {}*x^{}".format(round(c, 3), i+1)

    return expr
