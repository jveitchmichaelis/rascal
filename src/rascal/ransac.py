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
    ):
        """
        Josh will write something here.

        """

        self.config = OmegaConf.structured(RansacConfig)

        if config is not None:
            if isinstance(config, dict):
                OmegaConf.merge(self.config, OmegaConf.create(config))
            else:
                OmegaConf.merge(self.config, config)

        self.x = x
        self.y = y
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

        if len(np.unique(self.x)) <= self.config.degree:
            raise ValueError(
                f"Fit degree ({self.config.degree}) is greater than the provided number of points ({len(np.unique(self.x))})"
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

        if self.config.filter_close:
            self.logger.debug(
                "Filtering close x-values with tolerance "
                + f"{self.config.rms_tolerance}"
            )
            self._filter_close()

        if self.hough is not None:
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

            self.sampler.update(matched_x, matched_y)

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

        err = []
        matched_x = []
        matched_y = []

        for x_i in x:

            fit = self.polyval(x_i, fit_coeff)

            # Get closest match for this peak
            errs = (fit - y_for_x[x_i]) ** 2
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
        if (self.hough is not None) & (self.twoditp is not None):

            wave = self.polyval(self.x, result.fit_coeffs)
            gradient = self.polyval(self.x, _derivative(result.fit_coeffs))
            intercept = wave - gradient

            # weight = self.config.hough_weight * np.sum(
            #    self.twoditp(intercept, gradient, grid=False)
            # )

        else:

            # weight = 1.0
            pass

        if self.config.use_msac:

            # M-SAC Estimator (Torr and Zisserman, 1996)

            tolerance = 1.96 * result.residual.std()

            result.residual[result.residual > tolerance] = tolerance

            # Remove for now: / (len(result.residual) - len(result.fit_coeffs) + 1)
            # / (weight + 1e-16)

            cost = sum(result.residual) + 1e-16

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
            if self.config.type == "poly":
                try:

                    coeffs = models.robust_polyfit(
                        inliers_x, inliers_y, self.config.degree
                    )

                except np.linalg.LinAlgError:

                    self.logger.warning("Linear algebra error in robust fit")
                    return False

            else:
                coeffs = self.polyfit(inliers_x, inliers_y, self.config.degree)

            if self._fit_valid(result):

                # Get the residual of the inliers
                residual = self.polyval(inliers_x, coeffs) - inliers_y
                residual[
                    np.abs(residual) > self.config.rms_tolerance
                ] = self.config.rms_tolerance

                rms_residual = np.sqrt(np.mean(residual**2))

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

                # Overfit
                if n_inliers <= self.config.degree + 1:
                    self.logger.debug(
                        f"Overfit: number of inliers {n_inliers} is less than what's required to fit: {self.config.degree+1}"
                    )
                    return False

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
