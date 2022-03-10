import numpy as np
from rascal.atlas import Atlas
from rascal.calibrator import Calibrator

np.random.seed(0)

peaks = np.sort(np.random.random(31) * 1000.0)
# Removed the closely spaced peaks
distance_mask = np.isclose(peaks[:-1], peaks[1:], atol=5.0)
distance_mask = np.insert(distance_mask, 0, False)
peaks = peaks[~distance_mask]

# Line list
wavelengths_linear = 3000.0 + 5.0 * peaks
wavelengths_quadratic = 3000.0 + 4 * peaks + 1.0e-3 * peaks**2.0

elements_linear = ["Linear"] * len(wavelengths_linear)
elements_quadratic = ["Quadratic"] * len(wavelengths_quadratic)


def test_linear_fit():

    # Initialise the calibrator
    c = Calibrator(peaks)
    a = Atlas()
    c.set_calibrator_properties(num_pix=1000)
    c.set_hough_properties(
        num_slopes=1000,
        range_tolerance=500.0,
        xbins=200,
        ybins=200,
        min_wavelength=3000.0,
        max_wavelength=8000.0,
    )
    a.add_user_atlas(elements=elements_linear, wavelengths=wavelengths_linear)
    c.set_atlas(a)
    c.set_ransac_properties(minimum_matches=20, minimum_fit_error=1e-12)
    c.do_hough_transform(brute_force=False)

    # Run the wavelength calibration
    (
        best_p,
        matched_peaks,
        matched_atlas,
        rms,
        residual,
        peak_utilisation,
        atlas_utilisation,
    ) = c.fit(max_tries=500, fit_deg=1)
    # Refine solution
    (
        best_p,
        matched_peaks,
        matched_atlas,
        rms,
        residual,
        peak_utilisation,
        atlas_utilisation,
    ) = c.match_peaks(best_p, refine=False, robust_refit=True)

    assert np.abs(best_p[1] - 5.0) / 5.0 < 0.001
    assert np.abs(best_p[0] - 3000.0) / 3000.0 < 0.001
    assert peak_utilisation > 0.8
    assert atlas_utilisation > 0.0

    assert len(c.get_pix_wave_pairs()) == len(peaks)


def test_manual_refit():

    # Initialise the calibrator
    c = Calibrator(peaks)
    a = Atlas()
    c.set_calibrator_properties(num_pix=1000)
    c.set_hough_properties(
        num_slopes=1000,
        range_tolerance=500.0,
        xbins=200,
        ybins=200,
        min_wavelength=3000.0,
        max_wavelength=8000.0,
    )
    a.add_user_atlas(elements=elements_linear, wavelengths=wavelengths_linear)
    c.set_atlas(a)
    c.set_ransac_properties(minimum_matches=25, minimum_fit_error=1e-12)
    c.do_hough_transform(brute_force=False)

    # Run the wavelength calibration
    (
        best_p,
        matched_peaks,
        matched_atlas,
        rms,
        residual,
        peak_utilisation,
        atlas_utilisation,
    ) = c.fit(max_tries=500, fit_deg=1)

    # Refine solution
    (
        best_p_robust,
        matched_peaks,
        matched_atlas,
        rms,
        residual,
        peak_utilisation,
        atlas_utilisation,
    ) = c.match_peaks(best_p, refine=False, robust_refit=True)

    (
        best_p_manual,
        matched_peaks,
        matched_atlas,
        rms,
        residuals,
    ) = c.manual_refit(matched_peaks, matched_atlas)

    assert np.allclose(best_p_manual, best_p)


def test_manual_refit_remove_points():

    # Initialise the calibrator
    c = Calibrator(peaks)
    a = Atlas()
    c.set_calibrator_properties(num_pix=1000)
    c.set_hough_properties(
        num_slopes=1000,
        range_tolerance=500.0,
        xbins=200,
        ybins=200,
        min_wavelength=3000.0,
        max_wavelength=8000.0,
    )
    a.add_user_atlas(elements=elements_linear, wavelengths=wavelengths_linear)
    c.set_atlas(a)
    c.set_ransac_properties(minimum_matches=25, minimum_fit_error=1e-12)
    c.do_hough_transform(brute_force=False)

    # Run the wavelength calibration
    (
        best_p,
        atched_peaks,
        matched_atlas,
        rms,
        residual,
        peak_utilisation,
        atlas_utilisation,
    ) = c.fit(max_tries=500, fit_deg=1)

    # Refine solution
    (
        best_p,
        matched_peaks,
        matched_atlas,
        rms,
        residual,
        peak_utilisation,
        atlas_utilisation,
    ) = c.match_peaks(best_p, refine=False, robust_refit=True)

    c.remove_pix_wave_pair(5)

    (
        best_p_manual,
        matched_peaks,
        matched_atlas,
        rms,
        residuals,
    ) = c.manual_refit(matched_peaks, matched_atlas)

    assert np.allclose(best_p_manual, best_p)


def test_manual_refit_add_points():

    # Initialise the calibrator
    c = Calibrator(peaks)
    a = Atlas()
    c.set_calibrator_properties(num_pix=1000)
    c.set_hough_properties(
        num_slopes=1000,
        range_tolerance=500.0,
        xbins=200,
        ybins=200,
        min_wavelength=3000.0,
        max_wavelength=8000.0,
    )
    a.add_user_atlas(elements=elements_linear, wavelengths=wavelengths_linear)
    c.set_atlas(a)
    c.set_ransac_properties(minimum_matches=25, minimum_fit_error=1e-12)
    c.do_hough_transform(brute_force=False)

    # Run the wavelength calibration
    (
        best_p,
        atched_peaks,
        matched_atlas,
        rms,
        residual,
        peak_utilisation,
        atlas_utilisation,
    ) = c.fit(max_tries=500, fit_deg=1)

    # Refine solution
    (
        best_p,
        matched_peaks,
        matched_atlas,
        rms,
        residual,
        peak_utilisation,
        atlas_utilisation,
    ) = c.match_peaks(best_p, refine=False, robust_refit=True)

    c.add_pix_wave_pair(2000.0, 3000.0 + 4 * 2000.0 + 1.0e-3 * 2000.0**2.0)
    (
        best_p_manual,
        matched_peaks,
        matched_atlas,
        rms,
        residuals,
    ) = c.manual_refit(matched_peaks, matched_atlas)

    assert np.allclose(best_p_manual, best_p)


def test_quadratic_fit():

    # Initialise the calibrator
    c = Calibrator(peaks)
    a = Atlas()
    c.set_calibrator_properties(num_pix=1000)
    c.set_hough_properties(
        num_slopes=1000,
        range_tolerance=500.0,
        xbins=100,
        ybins=100,
        min_wavelength=3000.0,
        max_wavelength=8000.0,
    )
    a.add_user_atlas(
        elements=elements_quadratic, wavelengths=wavelengths_quadratic
    )
    c.set_atlas(a)
    c.set_ransac_properties(minimum_matches=20, minimum_fit_error=1e-12)
    c.do_hough_transform(brute_force=False)

    # Run the wavelength calibration
    (
        best_p,
        matched_peaks,
        matched_atlas,
        rms,
        residual,
        peak_utilisation,
        atlas_utilisation,
    ) = c.fit(
        max_tries=2000, fit_tolerance=5.0, candidate_tolerance=2.0, fit_deg=2
    )
    # Refine solution
    (
        best_p_robust,
        matched_peaks,
        matched_atlas,
        rms,
        residual,
        peak_utilisation,
        atlas_utilisation,
    ) = c.match_peaks(best_p, refine=False, robust_refit=True)

    assert np.abs(best_p_robust[2] - 1e-3) / 1e-3 < 0.001
    assert np.abs(best_p_robust[1] - 4.0) / 4.0 < 0.001
    assert np.abs(best_p_robust[0] - 3000.0) / 3000.0 < 0.001
    assert peak_utilisation > 0.8
    assert atlas_utilisation > 0.5


def test_quadratic_fit_legendre():

    # Initialise the calibrator
    c = Calibrator(peaks)
    a = Atlas()
    c.set_calibrator_properties(num_pix=1000)
    c.set_hough_properties(
        num_slopes=500,
        range_tolerance=200.0,
        xbins=100,
        ybins=100,
        min_wavelength=3000.0,
        max_wavelength=8000.0,
    )
    a.add_user_atlas(
        elements=elements_quadratic, wavelengths=wavelengths_quadratic
    )
    c.set_atlas(a)
    c.set_ransac_properties(
        sample_size=10, minimum_matches=20, minimum_fit_error=1e-12
    )
    c.do_hough_transform(brute_force=False)

    # Run the wavelength calibration
    (
        best_p,
        matched_peaks,
        matched_atlas,
        rms,
        residual,
        peak_utilisation,
        atlas_utilisation,
    ) = c.fit(
        max_tries=2000,
        fit_tolerance=5.0,
        candidate_tolerance=2.0,
        fit_deg=2,
        fit_type="legendre",
    )

    # Legendre 2nd order takes the form
    assert np.abs(best_p[2] - 1e-3) / 1e-3 < 0.001
    assert np.abs(best_p[1] - 4.0) / 4.0 < 0.001
    assert np.abs(best_p[0] - 3000.0) / 3000.0 < 0.001
    assert peak_utilisation > 0.7
    assert atlas_utilisation > 0.5


def test_quadratic_fit_chebyshev():

    # Initialise the calibrator
    c = Calibrator(peaks)
    a = Atlas()
    c.set_calibrator_properties(num_pix=1000)
    c.set_hough_properties(
        num_slopes=500,
        range_tolerance=200.0,
        xbins=100,
        ybins=100,
        min_wavelength=3000.0,
        max_wavelength=8000.0,
    )
    a.add_user_atlas(
        elements=elements_quadratic, wavelengths=wavelengths_quadratic
    )
    c.set_atlas(a)
    c.set_ransac_properties(
        sample_size=10, minimum_matches=20, minimum_fit_error=1e-12
    )
    c.do_hough_transform(brute_force=False)

    # Run the wavelength calibration
    (
        best_p,
        matched_peaks,
        matched_atlas,
        rms,
        residual,
        peak_utilisation,
        atlas_utilisation,
    ) = c.fit(
        max_tries=2000,
        fit_tolerance=5.0,
        candidate_tolerance=2.0,
        fit_deg=2,
        fit_type="chebyshev",
    )

    assert np.abs(best_p[2] - 1e-3) / 1e-3 < 0.001
    assert np.abs(best_p[1] - 4.0) / 4.0 < 0.001
    assert np.abs(best_p[0] - 3000.0) / 3000.0 < 0.001
    assert peak_utilisation > 0.7
    assert atlas_utilisation > 0.5
