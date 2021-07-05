import numpy as np

from rascal.synthetic import SyntheticSpectrum
from rascal.calibrator import Calibrator


def test_default():

    # Create a test spectrum with a simple linear relationship
    # between pixels/wavelengths. The intercept is set to
    # 100 nm and the dispersion is set to 2.
    intercept = 100
    dispersion = 2.0
    s = SyntheticSpectrum(coefficients=[dispersion, intercept])

    # We add a bunch of wavelegnths between 200-1200 nm
    waves = np.linspace(200, 1200, num=20)

    peaks, waves = s.get_pixels(waves)
    assert len(peaks) > 0

    # Set up the calibrator with the pixel values of our
    # wavelengths
    c = Calibrator(peaks=peaks)

    # Arbitrarily we'll set the number of pixels to 768 (i.e.
    # a max range of around 1500 nm
    c.set_calibrator_properties(num_pix=768)

    # Setup the Hough transform parameters
    c.set_hough_properties(range_tolerance=100.,
                           min_wavelength=100.,
                           max_wavelength=1500.)

    # Add our fake lines as the atlas
    c.add_user_atlas(elements=["Test"] * len(waves), wavelengths=waves)
    assert len(c.atlas) > 0

    # And let's try and fit...
    best_p, rms, residual, peak_utilisation = c.fit(max_tries=500)

    best_p, x_fit, y_fit, residual, peak_utilisation = c.match_peaks(
        best_p, refine=False, robust_refit=True)

    fit_diff = c.polyval(x_fit, best_p) - y_fit
    rms = np.sqrt(np.sum(fit_diff**2 / len(x_fit)))

    assert peak_utilisation > 0.75
    assert rms < 5.
