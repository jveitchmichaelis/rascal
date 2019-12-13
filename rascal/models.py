import scipy.optimize
import numpy as np

"""
Model functions for spectral fitting
"""

def poly_cost_function(a, x, y, degree):
    f = polynomial(a, degree)
    return y - f(x)

def linear(a):
    """
    Returns a lambda function which computes:

    f(x, a) =  a[1] + a[0]*x
    """

    assert(len(a) == 2)

    return lambda x: a[1] + a[0]*x

def quadratic(a):
    """
    Returns a lambda function which computes:

    f(x, a) = a[2] + a[1]*x + a[0]*x**2
    """

    assert(len(a == 3))

    return lambda x: a[2] + a[1]*x + a[0]*x**2

def cubic(a):
    """
    Returns a lambda function which computes:

    f(x, a) = a[3] + a[2]*x + a[1]*x**2 + a[0]*x**3
    """

    assert(len(a) == 4)

    return lambda x: a[3] + a[2]*x + a[1]*x**2 + a[0]*x**3

def polynomial(a, degree=3):


    """
    Returns a lambda function which computes an nth order polynormal:

    f(x, a) = sum_i (a[degree-i] * x**i )
    """

    assert(len(a) == degree+1)

    if degree == 1:
        return linear(a)
    elif degree == 2:
        return quadratic(a)
    elif degree == 3:
        return cubic(a)

    def poly(x):
        t = a[-1]

        for i in range(1, int(degree+1)):
            t += a[int(degree-i)]*x**i

        return t
    
    return poly

def normalise_input(x, y):
    """
    Transforms inputs to have unit variance
    """
    x_scale = x.std()
    y_scale = y.std()
    
    x_norm = x/x_scale
    y_norm = y/y_scale
    
    return x_norm, y_norm

def robust_polyfit(x, y, degree=3, x0=None, bounds=None):

        x_n, y_n = normalise_input(x, y)

        # Need to normalise the fit function too
        if x0 is not None:
            for i in range(0, degree):
                x0[i] *= x.std() ** (degree-1)

            x0 /= y.std()
        else:
            x0 = np.ones(degree+1)

        if bounds is None:
            bounds = np.inf*np.ones(degree+1)

        assert len(bounds) > 0 

        res = scipy.optimize.least_squares(poly_cost_function, x0, args=(x_n, y_n, degree), loss='huber', diff_step=1e-5)
        p = res.x

        p *= y.std()

        for i in range(0, degree):
            p[i] /= x.std() ** (degree-i)

        return p