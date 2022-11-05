import numpy as np

from rascal.atlas import Atlas
from rascal.calibrator import Calibrator
from rascal.synthetic import SyntheticSpectrum


def test_default():

    # Create a test spectrum with a simple linear relationship
    # between pixels/wavelengths. The intercept is set to
    # 100 nm and the gradient is set to 2.
    intercept = 100
    gradient = 2.0
    s = SyntheticSpectrum(coefficients=[intercept, gradient])

    # We add a bunch of wavelegnths between 200-1200 nm
    waves = np.linspace(200, 1200, num=20)

    peaks, waves = s.get_pixels(waves)
    assert len(peaks) > 0

    # Set up the calibrator with the pixel values of our
    # wavelengths
    c = Calibrator(peaks=peaks)
    a = Atlas()

    # Arbitrarily we'll set the number of pixels to 768 (i.e.
    # a max range of around 1500 nm
    c.set_calibrator_properties(num_pix=768)

    # Setup the Hough transform parameters
    c.set_hough_properties(
        range_tolerance=100.0, min_wavelength=100.0, max_wavelength=1500.0
    )

    # Add our fake lines as the atlas
    a.add_user_atlas(elements=["Test"] * len(waves), wavelengths=waves)
    c.set_ransac_properties(minimum_fit_error=1e-12)
    c.set_atlas(a)
    assert len(c.atlas.atlas_lines) > 0

    # And let's try and fit...
    best_p, x, y, rms, residual, peak_utilisation, atlas_utilisation = c.fit(
        max_tries=500
    )

    (
        best_p,
        x_fit,
        y_fit,
        rms,
        residual,
        peak_utilisation,
        atlas_utilisation,
    ) = c.match_peaks(best_p, refine=False, robust_refit=True)

    fit_diff = c.polyval(x_fit, best_p) - y_fit
    rms = np.sqrt(np.sum(fit_diff**2 / len(x_fit)))

    assert peak_utilisation > 0.7
    assert atlas_utilisation > 0.0
    assert rms < 5.0


def test_get_candidate_points_poly():

    # Create a test spectrum with a simple linear relationship
    # between pixels/wavelengths. The intercept is set to
    # 100 nm and the gradient is set to 2.
    intercept = 100
    gradient = 2.0
    best_p = [intercept, gradient]
    s = SyntheticSpectrum(coefficients=best_p)

    # We add a bunch of wavelegnths between 200-1200 nm
    waves = np.linspace(200, 1200, num=25)

    peaks, waves = s.get_pixels(waves)
    assert len(peaks) > 0

    # Set up the calibrator with the pixel values of our
    # wavelengths
    c = Calibrator(peaks=peaks)
    a = Atlas()

    # Arbitrarily we'll set the number of pixels to 768 (i.e.
    # a max range of around 1500 nm
    c.set_calibrator_properties(num_pix=768)

    # Setup the Hough transform parameters
    c.set_hough_properties(
        range_tolerance=100.0, min_wavelength=100.0, max_wavelength=1500.0
    )

    c.set_ransac_properties(linear=False, minimum_fit_error=1e-12)

    # Add our fake lines as the atlas
    a.add_user_atlas(elements=["Test"] * len(waves), wavelengths=waves)
    c.set_atlas(a)
    assert len(c.atlas.atlas_lines) > 0

    # And let's try and fit...
    best_p, x, y, rms, residual, peak_utilisation, atlas_utilisation = c.fit(
        max_tries=500, fit_coeff=best_p
    )

    (
        best_p,
        x_fit,
        y_fit,
        rms,
        residual,
        peak_utilisation,
        atlas_utilisation,
    ) = c.match_peaks(best_p, refine=True, robust_refit=True)

    fit_diff = c.polyval(x_fit, best_p) - y_fit
    rms = np.sqrt(np.sum(fit_diff**2 / len(x_fit)))

    assert peak_utilisation > 0.7
    assert atlas_utilisation > 0.0
    assert rms < 5.0
