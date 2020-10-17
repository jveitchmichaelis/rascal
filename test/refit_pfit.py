import numpy as np
from numpy.polynomial.polynomial import polyfit as polyfit
from numpy.polynomial.polynomial import polyval as polyval
from scipy.optimize import minimize
from matplotlib.pyplot import *
ion()

x = np.arange(-10, 10, 0.1)
y_noise = (np.random.random(len(x)) - 0.5) * x * 10.
y = x + x**2. + y_noise

x_shifted = x + 2
y_noise_shifted = (np.random.random(len(x)) - 0.5) * x * 10.
y_shifted = x + x**2. + y_noise_shifted

figure(1)
clf()
scatter(x, y, s=5)
scatter(x_shifted, y_shifted, s=5)

pfit = polyfit(x, y, deg=3)

plot(x, polyval(x, pfit))


def adjust_pfit_coefficient(delta, x, y, pfit):
    pfit_new = pfit.copy()
    for i, d in enumerate(delta):
        pfit_new[i] += d
    lsq = np.sum((y - polyval(x, pfit_new))**2.)
    return lsq


delta_trial = [0., 0., 0., 0.]

delta = minimize(adjust_pfit_coefficient,
                 delta_trial,
                 args=(x_shifted, y_shifted, pfit)).x
pfit_new = pfit.copy()
for i, d in enumerate(delta):
    pfit_new[i] += d

plot(x_shifted, polyval(x_shifted, pfit_new))
grid()