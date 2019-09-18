"""
Model functions for spectral fitting
"""

def linear(a):
    """
    Returns a lambda function which computes:

    f(x, a) =  a[0] + a[1]*x
    """

    assert(len(a) == 2)

    return lambda x: a[0] + a[1]*x

def quadratic(a):
    """
    Returns a lambda function which computes:

    f(x, a) = a[0] + a[1]*x + a[2]*x**2
    """

    assert(len(a == 3))

    return lambda x: a[0] + a[1]*x + a[2]*x**2

def cubic(a):
    """
    Returns a lambda function which computes:

    f(x, a) = a[0] + a[1]*x + a[2]*x**2 + a[3]*x**3
    """

    assert(len(a) == 4)

    return lambda x: a[0] + a[1]*x + a[2]*x**2 + a[3]*x**3

def polynomial(a, degree=3):


    """
    Returns a lambda function which computes an nth order polynormal:

    f(x, a) = sum_i (a[i] * x**i )
    """

    assert(len(a) == degree+1)

    if degree == 1:
        return linear(a)
    elif degree == 2:
        return quadratic(a)
    elif degree == 3:
        return cubic(a)

    def poly(x):
        t = a[0]

        for i in range(1,degree+1):
            t += a[i]*x**i

        return t
    
    return poly