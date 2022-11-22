import copy
import itertools
from tqdm.auto import tqdm
import numpy as np
from scipy import interpolate
from dotmap import DotMap

from . import models
from .util import _derivative
from .sampler import UniformRandomSampler, WeightedRandomSampler

"""

Ransac_properties

"""

_default_config = {
    "sample_size": 5,
    "top_n_candidate": 5,
    "linear": True,
    "filter_close": False,
    "fit_tolerance": 5,
    "candidate_weighted": True,
    "hough_weight": 1.0,
    "minimum_matches": 3,
    "minimum_peak_utilisation": 0.0,
    "minimum_fit_error": 1.0e-4,
    "fit_type": "poly",
    "max_tries": -1,
    "fit_deg": 4,
    "use_msac": True,
    "weight_samples": True,
    "hide_progress": False,
    "polyfit": np.polynomial.polynomial.polyfit,
    "polyval": np.polynomial.polynomial.polyval,
    "hough": None,
}


class RansacSolver:
    def __init__(self, x, y, config=None):

        if config is None:
            config = _default_config

        self.config = DotMap(config)

        self.x = x
        self.y = y
        self.unique_x = np.sort(np.unique(self.x))

        self.setup()

    def setup(self):

        if len(np.unique(self.x)) <= self.config.fit_deg:
            raise ValueError(
                "Fit degree is greater than the provided number of points"
            )

        self.polyfit = self.config.polyfit
        self.polyval = self.config.polyval

        if self.config == "weight_samples":
            self.sampler = WeightedRandomSampler(
                self.x,
                self.y,
                self.config.sample_size,
                n_samples=self.config.max_tries,
            )
        else:
            self.sampler = UniformRandomSampler(
                self.x,
                self.y,
                self.config.sample_size,
                n_samples=self.config.max_tries,
            )

        if self.config.filter_close:
            self._filter_close()

        if self.config.hough is not None:

            ht = self.config.hough

            xbin_size = (ht.xedges[1] - ht.xedges[0]) / 2.0
            ybin_size = (ht.yedges[1] - ht.yedges[0]) / 2.0

            if np.isfinite(ht.hough_weight):

                self.twoditp = interpolate.RectBivariateSpline(
                    ht.xedges[1:] - xbin_size,
                    ht.yedges[1:] - ybin_size,
                    ht.hist,
                )

        else:
            self.twoditp = None

    def _filter_close(self):

        unique_y = np.unique(self.y)

        idx = np.argwhere(
            unique_y[1:] - unique_y[0:-1] < 3 * self.fit_tolerance
        )
        separation_mask = np.argwhere((self.y == unique_y[idx]).sum(0) == 0)

        self.y = self.y[separation_mask].flatten()
        self.x = self.x[separation_mask].flatten()

    def solve(self):

        self.should_stop = False

        # Reset/init best params
        self.valid_solution = False
        self.best_p = None
        self.best_cost = 1e50
        self.best_err = 1e50
        self.best_mask = [False]
        self.best_residual = None
        self.best_inliers = 0

        for sample in self.sampler:

            if self.should_stop:
                break

            x, y = sample

            fit_coeffs = self._fit_sample(x, y)
            err, cost = self._cost(fit_coeffs)
            self._update_cost(err, cost)

    def _match_bijective(self, y_for_x, x, fit_coeff):
        """

        Internal function used to return a list of inliers with a
        one-to-one relationship between peaks and wavelengths. This
        is critical as often we have several potential candidate lines
        for each peak. This function first iterates through each peak
        and selects the wavelength with the smallest error. It then
        iterates through this list and does the same for duplicate
        wavelengths.

        parameters
        ----------
        candidates: dict
            match candidates, internal to ransac

        peaks: list
            list of peaks [px]

        fit_coeff: list
            polynomial fit coefficients

        """

        err = []
        matched_x = []
        matched_y = []

        for x_i in x:

            fit = self.polyval(x_i, fit_coeff)

            # Get closest match for this peak
            errs = np.abs(fit - y_for_x[x_i])
            idx = np.argmin(errs)

            err.append(errs[idx])
            matched_x.append(x_i)
            matched_y.append(y_for_x[x_i][idx])

        err = np.array(err)
        matched_x = np.array(matched_x)
        matched_y = np.array(matched_y)

        # Now we also need to resolve duplicate y's
        filtered_x = []
        filtered_y = []
        filtered_err = []

        for wavelength in np.unique(matched_y):

            mask = matched_y == wavelength
            filtered_y.append(wavelength)

            err_idx = np.argmin(err[mask])
            filtered_x.append(matched_x[mask][err_idx])
            filtered_err.append(err[mask][err_idx])

        # overwrite
        err = np.array(filtered_err)
        matched_x = np.array(filtered_x)
        matched_y = np.array(filtered_y)

        assert len(np.unique(matched_x)) == len(np.unique(matched_y))

        return err, matched_x, matched_y

    def _fit_sample(self, x_hat, y_hat):
        # Try to fit the data.
        # This doesn't need to be robust, it's an exact fit.
        return self.polyfit(x_hat, y_hat, self.config.fit_deg)

    def _fit_valid(self, fit_coeffs):

        # Check the intercept.
        if (fit_coeffs[0] < self.min_intercept) | (
            fit_coeffs[0] > self.max_intercept
        ):

            self.logger.debug("Intercept exceeds bounds.")
            return False

        # Check monotonicity.
        pix_min = peaks[0] - np.ptp(peaks) * 0.2
        num_pix = peaks[-1] + np.ptp(peaks) * 0.2
        self.logger.debug((pix_min, num_pix))

        if not np.all(
            np.diff(self.polyval(np.arange(pix_min, num_pix, 1), fit_coeffs))
            > 0
        ):

            self.logger.debug("Solution is not monotonically increasing.")
            return False

        # Compute error and filter out many-to-one matches
        err, matched_x, matched_y = self._match_bijective(
            candidates, peaks, fit_coeffs
        )

        if len(matched_x) == 0:
            return False

        return valid

    def _cost(self, fit_coeffs):

        # modified cost function weighted by the Hough space density
        if (self.config.hough is not None) & (self.twoditp is not None):

            wave = self.polyval(self.x, fit_coeffs)
            gradient = self.polyval(self.x, _derivative(fit_coeffs))
            intercept = wave - gradient

            weight = self.hough_weight * np.sum(
                self.twoditp(intercept, gradient, grid=False)
            )

        else:

            weight = 1.0

        err, self.matched_x, self.matched_y = self._match_bijective(
            self.sampler.y_for_x, self.unique_x, fit_coeffs
        )

        if self.config.use_msac:
            # M-SAC Estimator (Torr and Zisserman, 1996)
            err[err > self.config.fit_tolerance] = self.config.fit_tolerance

            cost = (
                sum(err) / (len(err) - len(fit_coeffs) + 1) / (weight + 1e-9)
            )
        else:
            cost = 1.0 / (sum(err < self.config.fit_tolerance) + 1e-9)

        return err, cost

    def check_solution(self, fit):
        pass

    def _update_cost(self, err, cost):

        if cost <= self.best_cost:
            # reject lines outside the rms limit (ransac_tolerance)
            # TODO: should n_inliers be recalculated from the robust
            # fit?
            mask = err < self.config.fit_tolerance
            n_inliers = sum(mask)
            matched_peaks = self.matched_x[mask]
            matched_atlas = self.matched_y[mask]

            if len(matched_peaks) <= self.config.fit_deg:

                self.logger.debug("Too few good candidates for fitting.")
                return

            # Now we do a robust fit
            if self.config.fit_type == "poly":
                try:

                    coeffs = models.robust_polyfit(
                        matched_peaks, matched_atlas, self.config.fit_deg
                    )

                except np.linalg.LinAlgError:

                    self.logger.warning("Linear algebra error in robust fit")
                    return
            else:
                coeffs = self.polyfit(
                    matched_peaks, matched_atlas, self.config.fit_deg
                )

            # Check ends of fit:
            if self.config.min_wavelength is not None:

                min_wavelength_px = self.polyval(0, coeffs)

                if min_wavelength_px < (
                    self.config.min_wavelength - self.config.range_tolerance
                ) or min_wavelength_px > (
                    self.config.min_wavelength + self.config.range_tolerance
                ):
                    self.logger.debug(
                        "Lower wavelength of fit too small, "
                        "{:1.2f}.".format(min_wavelength_px)
                    )

                    return

            if self.config.max_wavelength is not None:

                if self.config.spectrum is not None:
                    fit_max_wavelength = len(self.config.spectrum)
                else:
                    fit_max_wavelength = self.config.num_pix

                max_wavelength_px = self.polyval(fit_max_wavelength, coeffs)

                if max_wavelength_px > (
                    self.config.max_wavelength + self.config.range_tolerance
                ) or max_wavelength_px < (
                    self.config.max_wavelength - self.config.range_tolerance
                ):
                    self.logger.debug(
                        "Upper wavelength of fit too large, "
                        "{:1.2f}.".format(max_wavelength_px)
                    )

                    return

            # Get the residual of the fit
            residual = self.polyval(matched_peaks, coeffs) - matched_atlas
            residual[
                np.abs(residual) > self.config.fit_tolerance
            ] = self.config.fit_tolerance

            rms_residual = np.sqrt(np.mean(residual**2))

            # Make sure that we don't accept fits with zero error
            if rms_residual < self.config.minimum_fit_error:

                self.logger.debug(
                    "Fit error too small, " "{:1.2f}.".format(rms_residual)
                )

                return

            # Check that we have enough inliers based on user specified
            # constraints

            if n_inliers < self.config.minimum_matches:

                self.logger.debug(
                    "Not enough matched peaks for valid solution, "
                    "user specified {}.".format(self.config.minimum_matches)
                )
                return

            if n_inliers < self.config.minimum_peak_utilisation * len(
                self.unique_x
            ):

                self.logger.debug(
                    "Not enough matched peaks for valid solution, "
                    "user specified {:1.2f} %.".format(
                        100 * self.config.minimum_peak_utilisation
                    )
                )
                return

            if (
                not self.config.use_msac
                and n_inliers == self.best_inliers
                and rms_residual > self.best_err
            ):
                self.logger.info(
                    "Match has same number of inliers, "
                    "but fit error is worse "
                    "({:1.2f} > {:1.2f}) %.".format(
                        rms_residual, self.best_err
                    )
                )
                return

            # If the best fit is accepted, update the lists
            self.best_cost = cost
            self.best_inliers = n_inliers
            self.best_p = coeffs
            self.best_err = rms_residual
            self.best_residual = residual
            self.matched_peaks = list(copy.deepcopy(matched_peaks))
            self.matched_atlas = list(copy.deepcopy(matched_atlas))

            # Sanity check that matching peaks/atlas lines are 1:1
            assert len(np.unique(self.matched_peaks)) == len(
                self.matched_peaks
            )
            assert len(np.unique(self.matched_atlas)) == len(
                self.matched_atlas
            )
            assert len(np.unique(self.matched_atlas)) == len(
                np.unique(self.matched_peaks)
            )


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

    # Calculate initial error given pre-existing fit
    if fit_coeff is not None:
        err, _, _ = np.zeros()
        best_cost = sum(err)
        best_err = np.sqrt(np.mean(err**2.0))

    # The histogram is fixed, so pre-computed outside the loop
    if not brute_force:

        # weight the probability of choosing the sample by the inverse
        # line density
        pass

    for sample in sampler_list:

        keep_trying = True
        self.logger.debug(sample)

        while keep_trying:

            should_stop = False

            if brute_force:

                x_hat = x[[sample]]
                y_hat = y[[sample]]

            else:
                pass

            if should_stop:

                break

            # insert user given known pairs
            if calibrator.pix_known is not None:

                x_hat = np.concatenate((x_hat, calibrator.pix_known))
                y_hat = np.concatenate((y_hat, calibrator.wave_known))

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

    self.logger.info("Found: {}".format(best_inliers))

    return best_p, best_err, best_residual, best_inliers, valid_solution
