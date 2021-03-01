import numpy as np
from scipy import signal
from matplotlib import pyplot as plt


def gaus(x, a, b, x0, sigma):
    """
    Simple Gaussian function.

    Parameters
    ----------
    x: float or 1-d numpy array
        The data to evaluate the Gaussian over
    a: float
        the amplitude
    b: float
        the constant offset
    x0: float
        the center of the Gaussian
    sigma: float
        the width of the Gaussian

    Returns
    -------
    Array or float of same type as input (x).
    """
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + b


nist = np.loadtxt('nist_clean.csv', delimiter=',', dtype='str', skiprows=1)
nist_element = nist[:, 0]
nist_wavelength = nist[:, 1].astype('float')
nist_intensity = nist[:, 2].astype('float')

xe_idx = (nist_element == 'Xe')
xe_w = np.around(nist_wavelength[xe_idx], decimals=2)
xe_i = nist_intensity[xe_idx]

# Generate the equally spaced-wavelength array, and the corresponding intensity
wavelength = np.around(np.arange(min(xe_w) - 500.,
                                 max(xe_w) + 500.01, 0.01),
                       decimals=2)
intensity = np.zeros_like(wavelength)
intensity[np.where(np.isin(wavelength, xe_w))] = xe_i

# Convolve with gaussian, expected from the resolution
min_wavelength = 3500
max_wavelength = 8500
num_pix = 1024

# A per pixel
R = (max_wavelength - min_wavelength) / num_pix
# Nyquist sampling rate (2.3) for CCD at seeing of 1 arcsec
sigma = R * 2.3 * 1.0
x = np.arange(-100, 100.01, 0.01)
gaussian = gaus(x, a=1., b=0., x0=0., sigma=sigma)

# Convolve to simulate the arc spectrum
model_spectrum = signal.convolve(intensity, gaussian, 'same')

plt.figure(1, figsize=(8, 8))
plt.clf()
plt.plot(wavelength, intensity, color='grey', label='NIST values')
plt.plot(wavelength, model_spectrum, label='Convolved Arc')
plt.xlabel('Vacuum Wavelength / A')
plt.ylabel('NIST intensity')
plt.grid()
plt.xlim(3800, 8200)
plt.ylim(0, 1000)
plt.legend()
plt.tight_layout()
