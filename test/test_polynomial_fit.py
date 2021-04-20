import numpy as np
from rascal.calibrator import Calibrator

peaks = np.sort(np.random.random(31) * 1000.)

# Line list
wavelengths_linear = 3000. + 5. * peaks
wavelengths_quadratic = 3000. + 4 * peaks + 1.0e-3 * peaks**2.

elements_linear = ['Linear'] * len(wavelengths_linear)
elements_quadratic = ['Quadratic'] * len(wavelengths_quadratic)


def test_linear_fit():

    # Initialise the calibrator
    c = Calibrator(peaks)
    c.set_calibrator_properties(num_pix=1000)
    c.set_hough_properties(num_slopes=1000,
                           range_tolerance=500.,
                           xbins=200,
                           ybins=200,
                           min_wavelength=3000.,
                           max_wavelength=8000.)
    c.add_user_atlas(elements=elements_linear, wavelengths=wavelengths_linear)
    c.set_ransac_properties()

    # Run the wavelength calibration
    best_p, rms, residual, peak_utilisation = c.fit(max_tries=500, fit_deg=1)
    # Refine solution
    best_p, x_fit, y_fit, residual, peak_utilisation = c.match_peaks(
        best_p, refine=False, robust_refit=True)

    assert (best_p[1] > 5. * 0.999) & (best_p[1] < 5. * 1.001)
    assert (best_p[0] > 3000. * 0.999) & (best_p[0] < 3000. * 1.001)


def test_quadratic_fit():

    # Initialise the calibrator
    c = Calibrator(peaks)
    c.set_calibrator_properties(num_pix=1000)
    c.set_hough_properties(num_slopes=1000,
                           range_tolerance=500.,
                           xbins=100,
                           ybins=100,
                           min_wavelength=3000.,
                           max_wavelength=8000.)
    c.add_user_atlas(elements=elements_quadratic,
                     wavelengths=wavelengths_quadratic)
    c.set_ransac_properties()

    # Run the wavelength calibration
    best_p, rms, residual, peak_utilisation = c.fit(max_tries=1000, fit_deg=2)
    # Refine solution
    best_p, x_fit, y_fit, residual, peak_utilisation = c.match_peaks(
        best_p, refine=False, robust_refit=True)

    assert (best_p[2] > 1e-3 * 0.999) & (best_p[2] < 1e-3 * 1.001)
    assert (best_p[1] > 4. * 0.999) & (best_p[1] < 4. * 1.001)
    assert (best_p[0] > 3000. * 0.999) & (best_p[0] < 3000. * 1.001)


def test_quadratic_fit_legendre():

    # Initialise the calibrator
    c = Calibrator(peaks)
    c.set_calibrator_properties(num_pix=1000)
    c.set_hough_properties(num_slopes=500,
                           range_tolerance=200.,
                           xbins=100,
                           ybins=100,
                           min_wavelength=3000.,
                           max_wavelength=8000.)
    c.add_user_atlas(elements=elements_quadratic,
                     wavelengths=wavelengths_quadratic)
    c.set_ransac_properties(sample_size=10)

    # Run the wavelength calibration
    best_p, rms, residual, peak_utilisation = c.fit(max_tries=2000,
                                                    fit_tolerance=5.,
                                                    candidate_tolerance=2.,
                                                    fit_deg=2,
                                                    fit_type='legendre')

    # Legendre 2nd order takes the form
    assert (best_p[2] > 1e-3 * 0.99) & (best_p[2] < 1e-3 * 1.01)
    assert (best_p[1] > 4. * 0.99) & (best_p[1] < 4. * 1.01)
    assert (best_p[0] > 3000. * 0.99) & (best_p[0] < 3000. * 1.01)


def test_quadratic_fit_chebyshev():

    # Initialise the calibrator
    c = Calibrator(peaks)
    c.set_calibrator_properties(num_pix=1000)
    c.set_hough_properties(num_slopes=500,
                           range_tolerance=200.,
                           xbins=100,
                           ybins=100,
                           min_wavelength=3000.,
                           max_wavelength=8000.)
    c.add_user_atlas(elements=elements_quadratic,
                     wavelengths=wavelengths_quadratic)
    c.set_ransac_properties(sample_size=10)

    # Run the wavelength calibration
    best_p, rms, residual, peak_utilisation = c.fit(max_tries=2000,
                                                    fit_tolerance=5.,
                                                    candidate_tolerance=2.,
                                                    fit_deg=2,
                                                    fit_type='chebyshev')
