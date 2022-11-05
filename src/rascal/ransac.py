import copy
import itertools
from tqdm.auto import tqdm
import numpy as np
from scipy import interpolate
from . import models
from .util import _derivative


def solve_candidate_ransac(
    calibrator,
    fit_deg,
    fit_coeff,
    max_tries,
    candidate_tolerance,
    brute_force,
    progress,
):
    """
    Use RANSAC to sample the parameter space and give best guess

    Parameters
    ----------
    fit_deg: int
        The order of polynomial.
    fit_coeff: None or 1D numpy array
        Initial polynomial fit fit_coefficients.
    max_tries: int
        Number of trials of polynomial fitting.
    candidate_tolerance: float
        toleranceold  (Angstroms) for considering a point to be an inlier
        during candidate peak/line selection. This should be reasonable
        small as we want to search for candidate points which are
        *locally* linear.
    brute_force: boolean
        Solve all pixel-wavelength combinations with set to True.
    progress: boolean
        Show the progress bar with tdqm if set to True.


    Returns
    -------
    best_p: list
        A list of size fit_deg of the best fit polynomial
        fit_coefficient.
    best_err: float
        Arithmetic mean of the residuals.
    sum(best_inliers): int
        Number of lines fitted within the ransac_tolerance.
    valid_solution: boolean
        False if overfitted.

    """

    if calibrator.linear:

        calibrator._get_candidate_points_linear(candidate_tolerance)

    else:

        calibrator._get_candidate_points_poly(candidate_tolerance)

    (
        calibrator.candidate_peak,
        calibrator.candidate_arc,
    ) = calibrator._get_most_common_candidates(
        calibrator.candidates,
        top_n_candidate=calibrator.top_n_candidate,
        weighted=calibrator.candidate_weighted,
    )

    calibrator.fit_deg = fit_deg

    valid_solution = False
    best_p = None
    best_cost = 1e50
    best_err = 1e50
    best_mask = [False]
    best_residual = None
    best_inliers = 0

    # Note that there may be multiple matches for
    # each peak, that is len(x) > len(np.unique(x))
    x = np.array(calibrator.candidate_peak)
    y = np.array(calibrator.candidate_arc)

    # Filter close wavelengths
    if calibrator.filter_close:

        unique_y = np.unique(y)
        idx = np.argwhere(
            unique_y[1:] - unique_y[0:-1] < 3 * calibrator.ransac_tolerance
        )
        separation_mask = np.argwhere((y == unique_y[idx]).sum(0) == 0)
        y = y[separation_mask].flatten()
        x = x[separation_mask].flatten()

    # If the number of lines is smaller than the number of degree of
    # polynomial fit, return failed fit.
    if len(np.unique(x)) <= calibrator.fit_deg:

        return (best_p, best_err, sum(best_mask), 0, False)

    # Brute force check all combinations. If the request sample_size is
    # the same or larger than the available lines, it is essentially a
    # brute force.
    if brute_force or (calibrator.sample_size >= len(np.unique(x))):

        idx = range(len(x))
        sampler = itertools.combinations(idx, calibrator.sample_size)
        calibrator.sample_size = len(np.unique(x))

    else:

        sampler = range(int(max_tries))

    if progress:

        sampler_list = tqdm(sampler, disable=calibrator.hide_progress)

    else:

        sampler_list = sampler

    peaks = np.sort(np.unique(x))
    idx = range(len(peaks))

    # Build a key(pixel)-value(wavelength) dictionary from the candidates
    candidates = {}

    for p in np.unique(x):

        candidates[p] = y[x == p]

    if calibrator.ht.xedges is not None:

        xbin_size = (calibrator.ht.xedges[1] - calibrator.ht.xedges[0]) / 2.0
        ybin_size = (calibrator.ht.yedges[1] - calibrator.ht.yedges[0]) / 2.0

        if np.isfinite(calibrator.hough_weight):

            twoditp = interpolate.RectBivariateSpline(
                calibrator.ht.xedges[1:] - xbin_size,
                calibrator.ht.yedges[1:] - ybin_size,
                calibrator.ht.hist,
            )

    else:

        twoditp = None

    # Calculate initial error given pre-existing fit
    if fit_coeff is not None:
        err, _, _ = calibrator._match_bijective(candidates, peaks, fit_coeff)
        best_cost = sum(err)
        best_err = np.sqrt(np.mean(err**2.0))

    # The histogram is fixed, so pre-computed outside the loop
    if not brute_force:

        # weight the probability of choosing the sample by the inverse
        # line density
        h = np.histogram(peaks, bins=10)
        prob = 1.0 / h[0][np.digitize(peaks, h[1], right=True) - 1]
        prob = prob / np.sum(prob)

    for sample in sampler_list:

        keep_trying = True
        calibrator.logger.debug(sample)

        while keep_trying:

            should_stop = False

            if brute_force:

                x_hat = x[[sample]]
                y_hat = y[[sample]]

            else:

                # Pick some random peaks
                x_hat = np.random.choice(
                    peaks, calibrator.sample_size, replace=False, p=prob
                )
                y_hat = []

                # Pick a random wavelength for this x
                for _x in x_hat:

                    y_choice = candidates[_x]

                    # Avoid picking a y that's already associated with
                    # another x
                    if not set(y_choice).issubset(set(y_hat)):

                        y_temp = np.random.choice(y_choice)

                        while y_temp in y_hat:

                            y_temp = np.random.choice(y_choice)

                        y_hat.append(y_temp)

                    else:

                        calibrator.logger.debug(
                            "Not possible to draw a unique "
                            "set of atlas wavelengths."
                        )
                        should_stop = True
                        break

            if should_stop:

                break

            # insert user given known pairs
            if calibrator.pix_known is not None:

                x_hat = np.concatenate((x_hat, calibrator.pix_known))
                y_hat = np.concatenate((y_hat, calibrator.wave_known))

            # Try to fit the data.
            # This doesn't need to be robust, it's an exact fit.
            fit_coeffs = calibrator.polyfit(x_hat, y_hat, calibrator.fit_deg)

            # Check the intercept.
            if (fit_coeffs[0] < calibrator.min_intercept) | (
                fit_coeffs[0] > calibrator.max_intercept
            ):

                calibrator.logger.debug("Intercept exceeds bounds.")
                continue

            # Check monotonicity.
            pix_min = peaks[0] - np.ptp(peaks) * 0.2
            pix_max = peaks[-1] + np.ptp(peaks) * 0.2
            calibrator.logger.debug((pix_min, pix_max))

            if not np.all(
                np.diff(
                    calibrator.polyval(
                        np.arange(pix_min, pix_max, 1), fit_coeffs
                    )
                )
                > 0
            ):

                calibrator.logger.debug(
                    "Solution is not monotonically increasing."
                )
                continue

            # Compute error and filter out many-to-one matches
            err, matched_x, matched_y = calibrator._match_bijective(
                candidates, peaks, fit_coeffs
            )

            if len(matched_x) == 0:
                continue

            # use the Hough space density as weights for the cost function
            wave = calibrator.polyval(calibrator.pixel_list, fit_coeffs)
            gradient = calibrator.polyval(
                calibrator.pixel_list, _derivative(fit_coeffs)
            )
            intercept = wave - gradient * calibrator.pixel_list

            # modified cost function weighted by the Hough space density
            if (calibrator.hough_weight is not None) & (twoditp is not None):

                weight = calibrator.hough_weight * np.sum(
                    twoditp(intercept, gradient, grid=False)
                )

            else:

                weight = 1.0

            if calibrator.use_msac:
                # M-SAC Estimator (Torr and Zisserman, 1996)
                err[
                    err > calibrator.ransac_tolerance
                ] = calibrator.ransac_tolerance

                cost = (
                    sum(err)
                    / (len(err) - len(fit_coeffs) + 1)
                    / (weight + 1e-9)
                )
            else:
                cost = 1.0 / (sum(err < calibrator.ransac_tolerance) + 1e-9)

            # If this is potentially a new best fit, then handle that first
            if cost <= best_cost:
                # reject lines outside the rms limit (ransac_tolerance)
                # TODO: should n_inliers be recalculated from the robust
                # fit?
                mask = err < calibrator.ransac_tolerance
                n_inliers = sum(mask)
                matched_peaks = matched_x[mask]
                matched_atlas = matched_y[mask]

                if len(matched_peaks) <= calibrator.fit_deg:

                    calibrator.logger.debug(
                        "Too few good candidates for fitting."
                    )
                    continue

                # Now we do a robust fit
                if calibrator.fit_type == "poly":
                    try:

                        coeffs = models.robust_polyfit(
                            matched_peaks, matched_atlas, calibrator.fit_deg
                        )

                    except np.linalg.LinAlgError:

                        calibrator.logger.warning(
                            "Linear algebra error in robust fit"
                        )
                        continue
                else:
                    coeffs = calibrator.polyfit(
                        matched_peaks, matched_atlas, calibrator.fit_deg
                    )

                # Check ends of fit:
                if calibrator.min_wavelength is not None:

                    min_wavelength_px = calibrator.polyval(0, coeffs)

                    if min_wavelength_px < (
                        calibrator.min_wavelength - calibrator.range_tolerance
                    ) or min_wavelength_px > (
                        calibrator.min_wavelength + calibrator.range_tolerance
                    ):
                        calibrator.logger.debug(
                            "Lower wavelength of fit too small, "
                            "{:1.2f}.".format(min_wavelength_px)
                        )

                        continue

                if calibrator.max_wavelength is not None:

                    if calibrator.spectrum is not None:
                        fit_max_wavelength = len(calibrator.spectrum)
                    else:
                        fit_max_wavelength = calibrator.num_pix

                    max_wavelength_px = calibrator.polyval(
                        fit_max_wavelength, coeffs
                    )

                    if max_wavelength_px > (
                        calibrator.max_wavelength + calibrator.range_tolerance
                    ) or max_wavelength_px < (
                        calibrator.max_wavelength - calibrator.range_tolerance
                    ):
                        calibrator.logger.debug(
                            "Upper wavelength of fit too large, "
                            "{:1.2f}.".format(max_wavelength_px)
                        )

                        continue

                # Get the residual of the fit
                residual = (
                    calibrator.polyval(matched_peaks, coeffs) - matched_atlas
                )
                residual[
                    np.abs(residual) > calibrator.ransac_tolerance
                ] = calibrator.ransac_tolerance

                rms_residual = np.sqrt(np.mean(residual**2))

                # Make sure that we don't accept fits with zero error
                if rms_residual < calibrator.minimum_fit_error:

                    calibrator.logger.debug(
                        "Fit error too small, " "{:1.2f}.".format(best_err)
                    )

                    continue

                # Check that we have enough inliers based on user specified
                # constraints

                if n_inliers < calibrator.minimum_matches:

                    calibrator.logger.debug(
                        "Not enough matched peaks for valid solution, "
                        "user specified {}.".format(calibrator.minimum_matches)
                    )
                    continue

                if n_inliers < calibrator.minimum_peak_utilisation * len(
                    calibrator.peaks
                ):

                    calibrator.logger.debug(
                        "Not enough matched peaks for valid solution, "
                        "user specified {:1.2f} %.".format(
                            100 * calibrator.minimum_matches
                        )
                    )
                    continue

                if (
                    not calibrator.use_msac
                    and n_inliers == best_inliers
                    and rms_residual > best_err
                ):
                    calibrator.logger.info(
                        "Match has same number of inliers, "
                        "but fit error is worse "
                        "({:1.2f} > {:1.2f}) %.".format(rms_residual, best_err)
                    )
                    continue

                # If the best fit is accepted, update the lists
                best_cost = cost
                best_inliers = n_inliers
                best_p = coeffs
                best_err = rms_residual
                best_residual = residual
                calibrator.matched_peaks = list(copy.deepcopy(matched_peaks))
                calibrator.matched_atlas = list(copy.deepcopy(matched_atlas))

                # Sanity check that matching peaks/atlas lines are 1:1
                assert len(np.unique(calibrator.matched_peaks)) == len(
                    calibrator.matched_peaks
                )
                assert len(np.unique(calibrator.matched_atlas)) == len(
                    calibrator.matched_atlas
                )
                assert len(np.unique(calibrator.matched_atlas)) == len(
                    np.unique(calibrator.matched_peaks)
                )

                if progress:

                    sampler_list.set_description(
                        "Most inliers: {:d}, "
                        "best error: {:1.4f}".format(best_inliers, best_err)
                    )

                # Break early if all peaks are matched
                if best_inliers == len(peaks):
                    break

            # If we got this far, then we can continue to the next sample
            keep_trying = False

    # Overfit check
    if best_inliers <= calibrator.fit_deg + 1:

        valid_solution = False

    else:

        valid_solution = True

    # If we totally failed then this can be empty
    assert best_inliers == len(calibrator.matched_peaks)
    assert best_inliers == len(calibrator.matched_atlas)

    assert len(calibrator.matched_atlas) == len(set(calibrator.matched_atlas))

    calibrator.logger.info("Found: {}".format(best_inliers))

    return best_p, best_err, best_residual, best_inliers, valid_solution
