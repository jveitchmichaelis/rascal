import pynverse
import numpy as np
from . import models

class SyntheticSpectrum:
    def __init__(self, coefficients, model_type='cubic', degree=None):
        """
        Creates a synthetic spectrum generator which, given a suitable model,
        outputs the expected pixel locations of input wavelengths.
        
        Args:
            coefficients - list, coefficients for the model
            model_type - str, model type (linear, quadratic, cubic or poly)
            degree = int, if using a general poly model, its degree, default None
            
        It is
        expected that this will be used mainly for model testing, but 
        you can alsus
        """
        self.model = None

        # Default is approx. range of Silicon
        self.min_wavelength = 200
        self.max_wavelength = 12000

        # Model to fit
        if model_type == 'quadratic':
            self.model = models.quadratic(coefficients)
        elif model_type == 'cubic':
            self.model = models.cubic(coefficients)
        elif model_type == 'poly':

            if degree is None:
                raise ValueError("You should specify a polynomial degree.")

            self.model = models.polynomial(coefficients, degree)
        else:
            raise NotImplementedError

    def set_wavelength_limit(self, min_w, max_w):
        """
        Set a wavelength filter for the `get_pixels` function.
        """
        self.min_wavelength = min_w
        self.max_wavelength = max_w

    def get_pixels(self, wavelengths):
        """
        Returns a list of pixel locations for the wavelengths provided
        """


        if self.model is None:
            raise ValueError("Model not initiated")

        wavelengths = np.array(wavelengths)
        wavelengths = wavelengths[wavelengths > self.min_wavelength]
        wavelengths = wavelengths[wavelengths < self.max_wavelength]

        return pynverse.inversefunc(self.model, wavelengths)