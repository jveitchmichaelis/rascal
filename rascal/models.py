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

        for i in range(1, degree+1):
            t += a[degree-i]*x**i

        return t
    
    return poly