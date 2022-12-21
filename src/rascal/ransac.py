import copy
from tqdm.auto import tqdm
import numpy as np
from scipy import interpolate
from dotmap import DotMap
import logging

from . import models
from .util import _derivative
from .sampler import UniformRandomSampler, WeightedRandomSampler

"""

Ransac_properties

"""

_default_config = {
    "sample_size": 5,
    "filter_close": False,
    "fit_tolerance": 5,
    "hough_weight": 1.0,
    "fit_type": "poly",
    "max_tries": -1,
    "fit_deg": 4,
    "use_msac": True,
    "weight_samples": True,
    "progress": False,
    "polyfit_fn": np.polynomial.polynomial.polyfit,
    "polyval_fn": np.polynomial.polynomial.polyval,
    "fit_valid_fn": lambda x: True,
    "hough": None,
}


class SolveResult:
    def __init__(self, fit_coeffs=None, cost=1e9, x=[], y=[], residual=[]):
        self.fit_coeffs = fit_coeffs
        self.x = x
        self.y = y
        self.cost = cost

        if len(residual) > 0:
            self.residual = residual
            self.rms_residual = np.sqrt(np.mean(self.residual**2))


class RansacSolver:
    def __init__(self, x, y, config=None):

        if config is None:
            config = _default_config

        self.config = DotMap(config, _dynamic=False)

        self.x = x
        self.y = y
        self.unique_x = np.sort(np.unique(self.x))

        self.setup()

    def setup(self):

        self.logger = logging.getLogger("ransac")

        if len(np.unique(self.x)) <= self.config.fit_deg:
            raise ValueError(
                "Fit degree is greater than the provided number of points"
            )

        self.polyfit = self.config.polyfit_fn
        self.polyval = self.config.polyval_fn
        self._fit_valid = self.config.fit_valid_fn

        if self.config == "weight_samples":
            self.logger.debug(f"Using weighted random sampler")
            self.sampler = WeightedRandomSampler(
                self.x,
                self.y,
                self.config.sample_size,
                n_samples=self.config.max_tries,
            )
        else:
            self.logger.debug(f"Using uniform random sampler")
            self.sampler = UniformRandomSampler(
                self.x,
                self.y,
                self.config.sample_size,
                n_samples=self.config.max_tries,
            )

        if self.config.filter_close:
            self.logger.debug(
                f"Filtering close x-values with tolerance {self.config.fit_tolerance}"
            )
            self._filter_close()

        if self.config.hough is not None:
            self.logger.debug(f"Using hough weighting")
            ht = self.config.hough

            xbin_size = (ht.xedges[1] - ht.xedges[0]) / 2.0
            ybin_size = (ht.yedges[1] - ht.yedges[0]) / 2.0

            if np.isfinite(self.config.hough_weight):

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
            unique_y[1:] - unique_y[0:-1] < 3 * self.config.fit_tolerance
        )
        separation_mask = np.argwhere((self.y == unique_y[idx]).sum(0) == 0)

        self.y = self.y[separation_mask].flatten()
        self.x = self.x[separation_mask].flatten()

    def solve(self):

        self.should_stop = False

        # Reset/init best params
        self.valid_solution = False
        self.best_result = SolveResult()

        sample_iter = self.sampler

        if self.config.progress:
            sample_iter = tqdm(sample_iter)

        for sample in sample_iter:

            if self.should_stop:
                break

            x, y = sample

            fit_coeffs = self._fit_sample(x, y)
            self.logger.debug(f"Sample fit coeffs: {fit_coeffs}")

            residual, matched_x, matched_y = self._match_bijective(
                self.sampler.y_for_x, self.unique_x, fit_coeffs
            )

            result = SolveResult(
                fit_coeffs=fit_coeffs,
                residual=residual,
                x=matched_x,
                y=matched_y,
            )

            result.cost = self._cost(result)
            self.logger.debug(f"Fit cost: {result.cost}")
            self.logger.debug(f"Fit error: {result.rms_residual}")

            self._update_best(result)

            if self.config.progress:
                if self.valid_solution:
                    sample_iter.set_description(
                        "Most inliers: {:d}, "
                        "best error: {:1.4f}".format(
                            len(self.best_result.x),
                            self.best_result.rms_residual,
                        )
                    )

        if self.valid_solution:
            self.logger.info(
                "Found {} inliers".format(len(self.best_result.x))
            )

        return self.valid_solution

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

    def _cost(self, result):

        # modified cost function weighted by the Hough space density
        if (self.config.hough is not None) & (self.twoditp is not None):

            wave = self.polyval(self.x, result.fit_coeffs)
            gradient = self.polyval(self.x, _derivative(result.fit_coeffs))
            intercept = wave - gradient

            weight = self.config.hough_weight * np.sum(
                self.twoditp(intercept, gradient, grid=False)
            )

        else:

            weight = 1.0

        if self.config.use_msac:
            # M-SAC Estimator (Torr and Zisserman, 1996)
            result.residual[
                result.residual > self.config.fit_tolerance
            ] = self.config.fit_tolerance

            cost = (
                sum(result.residual)
                / (len(result.residual) - len(result.fit_coeffs) + 1)
                / (weight + 1e-9)
            )
        else:
            cost = 1.0 / (
                sum(result.residual < self.config.fit_tolerance) + 1e-9
            )

        return cost

    def _update_best(self, result):

        if result.cost <= self.best_result.cost:

            mask = result.residual < self.config.fit_tolerance
            n_inliers = sum(mask)
            inliers_x = result.x[mask]
            inliers_y = result.y[mask]

            if len(inliers_x) <= self.config.fit_deg:

                self.logger.debug("Too few good candidates for fitting.")
                return

            # Now we do a robust fit
            if self.config.fit_type == "poly":
                try:
                    coeffs = models.robust_polyfit(
                        inliers_x, inliers_y, self.config.fit_deg
                    )

                except np.linalg.LinAlgError:

                    self.logger.warning("Linear algebra error in robust fit")
                    return
            else:
                coeffs = self.polyfit(
                    inliers_x, inliers_y, self.config.fit_deg
                )

            if self._fit_valid(result):

                # Get the residual of the inliers
                residual = self.polyval(inliers_x, coeffs) - inliers_y
                residual[
                    np.abs(residual) > self.config.fit_tolerance
                ] = self.config.fit_tolerance

                rms_residual = np.sqrt(np.mean(residual**2))

                if (
                    not self.config.use_msac
                    and n_inliers == len(self.best_result.x)
                    and rms_residual > self.best_result.rms_residual
                ):
                    self.logger.info(
                        "Match has same number of inliers, "
                        "but fit error is worse "
                        "({:1.2f} > {:1.2f}) %.".format(
                            rms_residual, self.best_result.rms_residual
                        )
                    )
                    return

                # Overfit
                if n_inliers <= self.config.fit_deg + 1:
                    return

                # Sanity check that matching peaks/atlas lines are 1:1
                assert len(np.unique(inliers_x)) == len(inliers_x)
                assert len(np.unique(inliers_y)) == len(inliers_y)
                assert len(np.unique(inliers_x)) == len(np.unique(inliers_y))

                # Are these still required?
                # assert n_inliers == len(result.x)
                # assert n_inliers == len(result.y)
                # assert len(result.y) == len(set(result.y))

                self.best_result = SolveResult(
                    fit_coeffs=coeffs,
                    cost=result.cost,
                    residual=residual,
                    x=list(copy.deepcopy(inliers_x)),
                    y=list(copy.deepcopy(inliers_y)),
                )

                if n_inliers == len(self.x):
                    self.should_stop = True

                self.valid_solution = True
