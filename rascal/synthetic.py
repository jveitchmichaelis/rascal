import pynverse
import numpy as np
from . import models


class SyntheticSpectrum:
    def __init__(self,
                 coefficients=None,
                 min_wavelength=200.,
                 max_wavelength=1200.):
        """
        Creates a synthetic spectrum generator which, given a suitable model,
        outputs the expected pixel locations of input wavelengths.  It is
        expected that this will be used mainly for model testing.

        Parameters
        ----------
        coefficients: list
            coefficients for the model

        """

        # Default is approx. range of Silicon
        self.set_wavelength_limit(min_wavelength, max_wavelength)

        if coefficients is not None:
            self.coefficients = coefficients
            self.set_model(self.coefficients)

        else:

            self.model = None
            self.degree = None

    def set_model(self, coefficients):
        """
        Set the model to fit
        """

        if isinstance(coefficients, (list, np.ndarray)):

            self.degree = len(coefficients) - 1
            self.model = models.polynomial(a=coefficients, degree=self.degree)

        else:

            raise TypeError('Please provide a list or an numpy array.')

    def set_wavelength_limit(self, min_wavelength=None, max_wavelength=None):
        """
        Set a wavelength filter for the 'get_pixels' function.
        """

        if (not isinstance(min_wavelength, float)
                and min_wavelength is not None):

            raise TypeError('Please provide a numeric value or None to '
                            'retain the min_wavelength.')

        else:

            # Placeholder Min/Max
            if min_wavelength is not None:

                new_min_wavelength = min_wavelength

            else:

                new_min_wavelength = self.min_wavelength

        if (not isinstance(max_wavelength, float)
                and max_wavelength is not None):

            raise TypeError('Please provide a numeric value or None to '
                            'retain the max_wavelength.')

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

            raise RuntimeError('Minimum wavelength cannot be larger than '
                               'the maximum wavelength.')

    def get_pixels(self, wavelengths):
        """
        Returns a list of pixel locations for the wavelengths provided
        """

        if not isinstance(wavelengths, (list, np.ndarray)):

            raise TypeError('Please provide a list or an numpy array.')

        wavelengths = np.array(wavelengths)
        wavelengths = wavelengths[wavelengths > self.min_wavelength]
        wavelengths = wavelengths[wavelengths < self.max_wavelength]

        # Constant function y = c
        if self.degree == 0:
            pixels = self.coefficients[0]*np.ones(len(wavelengths))
        # Linear function y = mx + c
        elif self.degree == 1:
            # x = (y - c) / m
            pixels =\
                (wavelengths - self.coefficients[0]) / self.coefficients[1]
        else:
            pixels = pynverse.inversefunc(self.model, wavelengths)

        return pixels, wavelengths


'''
class RandomSyntheticSpectrum(SyntheticSpectrum):
    def __init__(self,
                 min_wavelength=400,
                 max_wavelength=800,
                 dispersion=0.5,
                 model_type='poly',
                 degree=5):

        x0 = min_wavelength
        x1 = dispersion
        x2 = 0.1 * random.random()

        coefficients = [x0, x1, x2]

        super().__init__(coefficients, model_type, degree)

    def add_atlas(elements, n_lines=30, min_intensity=10, min_distance=10):
        lines = load_calibration_lines(
            elements,
            min_atlas_wavelength=self.min_wavelength,
            max_atlas_wavelength=self.max_wavelength,
            min_intensity=min_intensity,
            min_distance=min_distance,
            vacuum=False,
            pressure=101325.,
            temperature=273.15,
            relative_humidity=0.)

        self.lines = random.choose(lines, n_lines)
'''
