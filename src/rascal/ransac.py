#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configure the ransac for the calibrator

"""

import copy
import logging
from typing import Callable, Optional, Union

import numpy as np
from dotmap import DotMap
from omegaconf import OmegaConf
from scipy import interpolate
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

from . import models
from .config import RansacConfig
from .houghtransform import HoughTransform
from .sampler import (
    ProbabilisticSampler,
    UniformRandomSampler,
    WeightedRandomSampler,
)
from .util import _derivative


class SolveResult:
    """
    Josh will write something here.

    """

    def __init__(
        self,
        fit_coeffs: Union[list, np.ndarray] = [],
        cost: float = 1.0e9,
        x: Union[list, np.ndarray] = [],
        y: Union[list, np.ndarray] = [],
        residual: Union[list, np.ndarray] = [],
        rms_tolerance: float = 5.0,
    ):
        """
        Josh will write something here.

        """

        self.fit_coeffs = np.asarray(fit_coeffs)
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.cost = cost
        self.residual = np.asarray(residual)
        self.rms_residual = None
        self.rms_tolerance = rms_tolerance
        self.n_inliers = None
        self.inliers_x = None
        self.inliers_y = None

        if len(residual) > 0:

            self.rms_residual = np.sqrt(np.mean(self.residual**2))
            mask = self.residual < self.rms_tolerance
            self.n_inliers = np.count_nonzero(mask)
            self.inliers_x = self.x[mask]
            self.inliers_y = self.y[mask]


class RansacSolver:
    """
    Josh will write something here.

    """

    def __init__(
        self,
        x: Union[list, np.ndarray],
        y: Union[list, np.ndarray],
        hough: Optional[HoughTransform] = None,
        config: dict = None,
        polyfit_fn: Callable = np.polynomial.polynomial.polyfit,
        polyval_fn: Callable = np.polynomial.polynomial.polyval,
        fit_valid_fn: Callable = lambda x: True,
        fixed_x: Optional[Union[list, np.ndarray]] = None,
        fixed_y: Optional[Union[list, np.ndarray]] = None,
    ):
        """
        Josh will write something here.

        """

        self.config = OmegaConf.structured(RansacConfig)

        if config is not None:
            if isinstance(config, dict):
                self.config = OmegaConf.merge(
                    self.config, OmegaConf.create(config)
                )
            else:
                self.config = OmegaConf.merge(self.config, config)

        self.x = x
        self.y = y
        self.fixed_x = (
            np.asarray(fixed_x, dtype=float).reshape(-1)
            if fixed_x is not None
            else None
        )
        self.fixed_y = (
            np.asarray(fixed_y, dtype=float).reshape(-1)
            if fixed_y is not None
            else None
        )
        if (self.fixed_x is None) != (self.fixed_y is None):
            raise ValueError(
                "Both fixed_x and fixed_y must be provided together."
            )
        if self.fixed_x is not None and len(self.fixed_x) != len(self.fixed_y):
            raise ValueError("fixed_x and fixed_y must have the same length.")
        self.polyfit = polyfit_fn
        self.polyval = polyval_fn
        self.hough = hough
        self._fit_valid = fit_valid_fn
        self.unique_x = np.sort(np.unique(self.x))

        self.setup()

        self.should_stop = False

        # Reset/init best params
        self.valid_solution = False
        self.best_result = SolveResult()

    def setup(self):
        """
        Josh will write something here.

        """

        self.logger = logging.getLogger(__name__)

        if self.config.filter_close:
            self.logger.debug(
                "Filtering close y-values with tolerance "
                + f"{self.config.rms_tolerance}"
            )
            self._filter_close()

        self.unique_x = np.sort(np.unique(self.x))

        if len(self.unique_x) <= self.config.degree:
            raise ValueError(
                f"Fit degree ({self.config.degree}) is greater than the provided number of points ({len(self.unique_x)})"
            )

        if self.config.sampler == "weighted":
            self.logger.debug(f"Using weighted random sampler")
            self.sampler = WeightedRandomSampler(
                self.x,
                self.y,
                self.config.sample_size,
                n_samples=self.config.max_tries,
            )
        elif self.config.sampler == "probabilistic":
            self.logger.debug(f"Using probalistic random sampler")
            self.sampler = ProbabilisticSampler(
                self.x,
                self.y,
                self.config.sample_size,
                n_samples=self.config.max_tries,
            )
        elif self.config.sampler == "uniform":
            self.logger.debug("Using uniform random sampler")
            self.sampler = UniformRandomSampler(
                self.x,
                self.y,
                self.config.sample_size,
                n_samples=self.config.max_tries,
            )
        else:
            raise NotImplementedError

        if self.hough is not None and self.hough.hist is not None:
            self.logger.debug("Using hough weighting")
            ht = self.hough

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

        else:
            self.twoditp = None

    def _filter_close(self):
        """
        Josh will write something here.

        """

        unique_y = np.unique(self.y)

        idx = np.argwhere(
            unique_y[1:] - unique_y[0:-1] < 3 * self.config.rms_tolerance
        )
        separation_mask = np.argwhere((self.y == unique_y[idx]).sum(0) == 0)

        self.y = self.y[separation_mask].flatten()
        self.x = self.x[separation_mask].flatten()

    def solve(self):
        """
        Josh will write something here.

        """

        sample_iter = self.sampler

        if self.config.progress:
            sample_iter = tqdm(sample_iter)

        self.logger.debug(
            f"Starting RANSAC with {self.config.max_tries} tries"
        )
        self.logger.debug(f"Unique x values: {len(self.unique_x)}")

        for sample in sample_iter:

            if self.should_stop:
                break

            x, y = sample

            fit_coeffs = self._fit_sample(x, y)
            self.logger.debug(f"Sample fit coeffs: {fit_coeffs}")

            residual, matched_x, matched_y = self._match_bijective(
                self.sampler.y_for_x, self.unique_x, fit_coeffs
            )

            inlier_mask = residual < self.config.rms_tolerance
            self.sampler.update(matched_x[inlier_mask], matched_y[inlier_mask])

            result = SolveResult(
                fit_coeffs=fit_coeffs,
                residual=residual,
                x=matched_x,
                y=matched_y,
                rms_tolerance=self.config.rms_tolerance,
            )

            result.cost = self._cost(result)
            self.logger.debug(f"Fit cost: {result.cost}")
            self.logger.debug(f"Fit error: {result.rms_residual}")

            self._update_best(result)

            if self.config.progress:

                if self.valid_solution:

                    self.logger.debug(f"Inliers: {len(self.best_result.x):d}")

                    sample_iter.set_description(
                        f"Most inliers: {len(self.best_result.x):d} "
                        + f"best error: {self.best_result.rms_residual:1.4f}"
                    )

        if self.valid_solution:

            self.logger.info(f"Found {len(self.best_result.x)} inliers")

        return self.valid_solution

    def _match_bijective(
        self,
        y_for_x: Union[list, np.ndarray],
        x: Union[list, np.ndarray],
        fit_coeff: Union[list, np.ndarray],
    ):
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
        y_for_x: list, np.ndarray
            list of peaks [px]
        x: list, np.ndarray
            list of wavelengths
        fit_coeff: list
            polynomial fit coefficients

        """

        if self.config.match_method == "greedy":
            return self._match_greedy(y_for_x, x, fit_coeff)
        else:
            return self._match_hungarian(y_for_x, x, fit_coeff)

    def _match_hungarian(
        self,
        y_for_x: Union[list, np.ndarray],
        x: Union[list, np.ndarray],
        fit_coeff: Union[list, np.ndarray],
    ):
        # Collect all unique atlas lines that are candidates for any peak
        all_y = np.unique(np.concatenate([y_for_x[x_i] for x_i in x]))
        y_to_idx = {y: i for i, y in enumerate(all_y)}

        n_peaks = len(x)
        n_atlas = len(all_y)

        # Build cost matrix: cost[i,j] = |polyval(x_i, coeff) - all_y[j]|
        # Non-candidate pairs get a large sentinel value.
        LARGE = 1e9
        cost_matrix = np.full((n_peaks, n_atlas), LARGE)

        for i, x_i in enumerate(x):
            fit = self.polyval(x_i, fit_coeff)
            for y in y_for_x[x_i]:
                j = y_to_idx[y]
                cost_matrix[i, j] = np.abs(fit - y)

        # Globally optimal bijective assignment (Hungarian algorithm)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Drop assignments that were to sentinel (non-candidate) slots
        costs = cost_matrix[row_ind, col_ind]
        valid = costs < LARGE

        matched_x = x[row_ind[valid]]
        matched_y = all_y[col_ind[valid]]
        err = costs[valid]

        assert len(np.unique(matched_x)) == len(np.unique(matched_y))

        return err, matched_x, matched_y

    def _match_greedy(
        self,
        y_for_x: Union[list, np.ndarray],
        x: Union[list, np.ndarray],
        fit_coeff: Union[list, np.ndarray],
    ):
        matched_x = []
        matched_y = []
        err = []
        used_y = set()

        for x_i in x:
            fit = self.polyval(x_i, fit_coeff)
            candidates = [(np.abs(fit - y), y) for y in y_for_x[x_i] if y not in used_y]
            if not candidates:
                continue
            best_err, best_y = min(candidates)
            matched_x.append(x_i)
            matched_y.append(best_y)
            err.append(best_err)
            used_y.add(best_y)

        matched_x = np.asarray(matched_x)
        matched_y = np.asarray(matched_y)
        err = np.asarray(err)

        assert len(np.unique(matched_x)) == len(np.unique(matched_y))

        return err, matched_x, matched_y

    def _fit_sample(
        self, x_hat: Union[list, np.ndarray], y_hat: Union[list, np.ndarray]
    ):
        """
        Fit for a polynomial.

        Parameters
        ----------
        x_hat: list, np.ndarray
            abscissa values
        y_hat: list, np.ndarray
            ordinate values

        """

        # Try to fit the data.
        # This doesn't need to be robust, it's an exact fit.
        if self.fixed_x is not None and len(self.fixed_x) > 0:
            x_hat = np.concatenate([x_hat, self.fixed_x])
            y_hat = np.concatenate([y_hat, self.fixed_y])
        return self.polyfit(x_hat, y_hat, self.config.degree)

    def _cost(self, result: SolveResult):
        """
        Josh will write something here.

        Parameters
        ----------
        results: SolveResult
            ?

        """

        # modified cost function weighted by the Hough space density
        if (self.hough is not None) and (self.twoditp is not None):

            wave = self.polyval(self.x, result.fit_coeffs)
            gradient = self.polyval(self.x, _derivative(result.fit_coeffs))
            intercept = wave - gradient * self.x

            hough_density = np.maximum(
                0.0, self.twoditp(gradient, intercept, grid=False)
            )
            weight = self.config.hough_weight * np.sum(hough_density)

        else:

            weight = 1.0

        if self.config.use_msac:

            # M-SAC Estimator (Torr and Zisserman, 1996)
            clipped = np.minimum(result.residual, self.config.rms_tolerance)
            dof = max(1, len(result.residual) - len(result.fit_coeffs) + 1)
            cost = (np.sum(clipped) + 1e-16) / dof / (weight + 1e-9)

        else:

            cost = 1.0 / (
                sum(result.residual < self.config.inlier_tolerance) + 1e-16
            )

        return cost

    def _update_best(self, result: SolveResult):
        """
        Josh will write something here.

        Parameters
        ----------
        results: SolveResult
            ?

        """

        if result.cost <= self.best_result.cost:

            mask = result.residual < self.config.rms_tolerance
            n_inliers = sum(mask)
            self.logger.debug(
                f"Number of points inlying with an error less than {self.config.rms_tolerance}: {n_inliers}, out of {len(mask)}"
            )
            inliers_x = result.x[mask]
            inliers_y = result.y[mask]

            if len(inliers_x) <= self.config.degree:

                self.logger.debug("Too few good candidates for fitting.")

                return False

            # Now we do a robust fit
            if self.config.fit_type == "poly":
                try:

                    coeffs = models.robust_polyfit(
                        inliers_x, inliers_y, self.config.degree
                    )

                except np.linalg.LinAlgError:

                    self.logger.warning("Linear algebra error in robust fit")
                    return False

            else:
                coeffs = self.polyfit(inliers_x, inliers_y, self.config.degree)

            # Re-expand inliers using the robust fit coefficients - more candidates
            # may now be within tolerance.
            residual_robust, inliers_x, inliers_y = self._match_bijective(
                self.sampler.y_for_x, self.unique_x, coeffs
            )
            mask_robust = residual_robust < self.config.rms_tolerance
            inliers_x = inliers_x[mask_robust]
            inliers_y = inliers_y[mask_robust]
            n_inliers = len(inliers_x)

            # Overfit
            if n_inliers <= self.config.degree + 1:
                self.logger.debug(
                    f"Overfit: number of inliers {n_inliers} is less than what's required to fit: {self.config.degree+1}"
                )
                return False

            # Get the residual of the (re-expanded) inliers
            residual = self.polyval(inliers_x, coeffs) - inliers_y
            residual[
                np.abs(residual) > self.config.rms_tolerance
            ] = self.config.rms_tolerance

            rms_residual = np.sqrt(np.mean(residual**2))

            # Validate using the new coefficients and post-robust inliers
            robust_result = SolveResult(
                fit_coeffs=coeffs,
                residual=np.abs(residual),
                x=inliers_x,
                y=inliers_y,
                rms_tolerance=self.config.rms_tolerance,
            )
            if self._fit_valid(robust_result):

                if (
                    not self.config.use_msac
                    and n_inliers == len(self.best_result.x)
                    and rms_residual > self.best_result.rms_residual
                ):
                    self.logger.info(
                        "Match has same number of inliers, but fit error is "
                        + f"worse ({rms_residual} > "
                        + f"{self.best_result.rms_residual})."
                    )
                    return False

                # Sanity check that matching peaks/atlas lines are 1:1
                assert len(np.unique(inliers_x)) == len(inliers_x)
                assert len(np.unique(inliers_y)) == len(inliers_y)
                assert len(np.unique(inliers_x)) == len(np.unique(inliers_y))

                self.best_result = SolveResult(
                    fit_coeffs=coeffs,
                    cost=result.cost,
                    residual=residual,
                    x=list(copy.deepcopy(inliers_x)),
                    y=list(copy.deepcopy(inliers_y)),
                    rms_tolerance=self.config.rms_tolerance,
                )

                if n_inliers == len(self.x):
                    self.logger.debug(
                        "All x fitted as inliers, breaking early."
                    )
                    self.should_stop = True

                self.valid_solution = True
        else:
            self.logger.debug(
                f"New solution {result.cost} is worse than current best {self.best_result.cost}."
            )
