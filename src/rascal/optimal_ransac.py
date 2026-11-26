#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimal RANSAC solver for wavelength calibration.

Python translation of the Optimal RANSAC algorithm by Anders Hast, Johan Nysjo,
and Andrea Marchetti. Original MATLAB source: https://user.it.uu.se/~andha390/code.html

Key additions over standard RANSAC (ransac.py):
  - _rescore: iteratively refit to inliers and re-match all data until set converges
  - _resample: subsample current inliers, refit, rescore; restart if a better set is found
  - _pruneset: iteratively remove worst point with tight threshold acc until all fit
  - ner stopping: halt after ner consecutive iterations that produce the same inlier set

References:
    Hast, A., Nysjo, J., Marchetti, A. (2013). Optimal RANSAC - Towards a
    Repeatable Algorithm for Finding the Optimal Set.

    Fishler, M.A. and Boles, R.C. (1981). Random sample consensus: A paradigm
    for model fitting with applications to image analysis and automated
    cartography. Comm. Assoc. Comp. Mach., 24(6), 381-395.
"""

from typing import Callable, Optional, Union

import numpy as np

from .ransac import RansacSolver, SolveResult


class OptimalRansacSolver(RansacSolver):
    """
    Optimal RANSAC solver for polynomial wavelength calibration.

    Extends RansacSolver with local optimization: after each initial sample,
    iteratively resamples and rescores to find the optimal inlier set, then
    prunes with a tighter threshold for repeatability.

    Parameters
    ----------
    x, y, hough, config, polyfit_fn, polyval_fn, fit_valid_fn
        Same as RansacSolver.
    acc : float, optional
        Tight accuracy threshold used by _pruneset. Points with residual > acc
        are iteratively removed. Defaults to rms_tolerance / 4.
        Pruning is only applied when acc < rms_tolerance.
    ner : int
        Number of Equal (non-improving) iterations before stopping. Higher
        values increase repeatability at the cost of more iterations.
        Default: 3.
    nrtrials : int
        Number of resampling attempts per iteration in _resample. Default: 8.
    maxit : int
        Maximum rescoring iterations per resample attempt. Default: 20.
    low : int, optional
        Minimum inlier count required before local optimization is applied.
        Defaults to degree + 2.
    """

    def __init__(
        self,
        x: Union[list, np.ndarray],
        y: Union[list, np.ndarray],
        hough=None,
        config: dict = None,
        polyfit_fn: Callable = np.polynomial.polynomial.polyfit,
        polyval_fn: Callable = np.polynomial.polynomial.polyval,
        fit_valid_fn: Callable = lambda x: True,
        acc: Optional[float] = None,
        ner: int = 3,
        nrtrials: int = 8,
        maxit: int = 20,
        low: Optional[int] = None,
        fixed_x: list = None,
        fixed_y: list = None,
        weights: Optional[Union[list, np.ndarray]] = None,
    ):
        super().__init__(
            x,
            y,
            hough=hough,
            config=config,
            polyfit_fn=polyfit_fn,
            polyval_fn=polyval_fn,
            fit_valid_fn=fit_valid_fn,
            fixed_x=fixed_x,
            fixed_y=fixed_y,
            weights=weights,
        )
        self.acc = acc if acc is not None else self.config.rms_tolerance / 4.0
        self.ner = ner
        self.nrtrials = nrtrials
        self.maxit = maxit
        self.low = low if low is not None else self.config.degree + 2

    def solve(self):
        """
        Main Optimal RANSAC loop.

        Mirrors optimalRansac.m: each iteration samples a minimal set, fits,
        then locally optimises via _resample/_rescore, prunes with _pruneset,
        and tracks how many consecutive iterations yield the same inlier set.
        Stops when that count reaches ner or max_tries is exhausted.
        """
        sample_iter = self.sampler

        if self.config.progress:
            from tqdm.auto import tqdm
            sample_iter = tqdm(sample_iter)

        self.logger.debug(
            f"Starting Optimal RANSAC: max_tries={self.config.max_tries}, "
            f"ner={self.ner}, acc={self.acc:.3f}, nrtrials={self.nrtrials}"
        )

        nes = 0           # consecutive equal-set count
        maxinl = 0        # best inlier count seen so far
        sinliers = None   # best inlier set as frozenset of (x, y) pairs

        for sample in sample_iter:

            if self.should_stop:
                break

            x_s, y_s = sample

            # --- sampleSet equivalent: fit minimal sample, initial match ---
            try:
                fit_coeffs = self._fit_sample(x_s, y_s)
            except Exception:
                continue

            residual, matched_x, matched_y = self._match_bijective(
                self.sampler.y_for_x, self.unique_x, fit_coeffs
            )

            mask = residual < self.config.rms_tolerance
            inliers_x = matched_x[mask]
            inliers_y = matched_y[mask]
            ninliers = len(inliers_x)
            if ninliers > self.config.sample_size:
                self.sampler.update(inliers_x, inliers_y)

            # --- Local optimisation (only when we have enough inliers) ---
            if ninliers > self.low:

                # resample.m: subsample inliers, refit, rescore until no improvement
                fit_coeffs, inliers_x, inliers_y = self._resample(
                    fit_coeffs, inliers_x, inliers_y
                )
                ninliers = len(inliers_x)

                # pruneset.m: strip borderline inliers using tight threshold
                if self.acc < self.config.rms_tolerance and ninliers > self.low:
                    fit_coeffs, inliers_x, inliers_y = self._pruneset(
                        fit_coeffs, inliers_x, inliers_y
                    )
                    ninliers = len(inliers_x)

            # --- Score and update best ---
            # Re-match the full dataset with the optimised coefficients so that
            # cost is computed consistently with the parent (all matched pairs,
            # not just the pruned subset).
            if ninliers > self.config.degree + 1:
                residual_full, matched_x_full, matched_y_full = self._match_bijective(
                    self.sampler.y_for_x, self.unique_x, fit_coeffs
                )
                result = SolveResult(
                    fit_coeffs=fit_coeffs,
                    residual=residual_full,
                    x=matched_x_full,
                    y=matched_y_full,
                    rms_tolerance=self.config.rms_tolerance,
                )
                result.cost = self._cost(result)
                self._update_best(result)

            # --- ner stopping criterion (mirrors optimalRansac.m lines 144-158) ---
            current_set = frozenset(zip(inliers_x.tolist(), inliers_y.tolist()))

            if maxinl > self.low and maxinl == ninliers:
                if sinliers is not None and current_set == sinliers:
                    nes += 1
                else:
                    nes = 0
                    sinliers = current_set
            elif ninliers > maxinl:
                nes = 0
                maxinl = ninliers
                sinliers = current_set
            elif ninliers == maxinl - 1:
                # Slight degradation is acceptable (matches MATLAB behaviour)
                nes = 0
                maxinl = ninliers
                sinliers = current_set

            if nes >= self.ner:
                self.logger.debug(
                    f"ner stopping: same inlier set seen {nes} consecutive times "
                    f"({ninliers} inliers)"
                )
                break

            if self.config.progress and self.valid_solution:
                sample_iter.set_description(
                    f"Most inliers: {len(self.best_result.x):d} "
                    + f"best error: {self.best_result.rms_residual:1.4f}"
                )

        if self.valid_solution:
            self.logger.info(f"Found {len(self.best_result.x)} inliers")

        return self.valid_solution

    # ------------------------------------------------------------------
    # Core local-optimisation helpers
    # ------------------------------------------------------------------

    def _rescore(self, fit_coeffs, inliers_x, inliers_y):
        """
        Iteratively refit to inliers and re-match all data until converged.

        Mirrors rescore.m. Fits a new model to the current inlier set, then
        re-evaluates the model against the full dataset. Repeats until the
        inlier set stops changing or maxit is reached.

        Parameters
        ----------
        fit_coeffs : np.ndarray
        inliers_x, inliers_y : np.ndarray

        Returns
        -------
        fit_coeffs, inliers_x, inliers_y : np.ndarray
        """
        n_prev = len(inliers_x)
        prev_set = frozenset(zip(inliers_x.tolist(), inliers_y.tolist()))

        for _ in range(self.maxit):

            if len(inliers_x) <= self.config.degree:
                break

            try:
                fit_coeffs = self.polyfit(inliers_x, inliers_y, self.config.degree)
            except Exception:
                break

            # Re-match against full dataset
            residual, matched_x, matched_y = self._match_bijective(
                self.sampler.y_for_x, self.unique_x, fit_coeffs
            )
            mask = residual < self.config.rms_tolerance
            new_x = matched_x[mask]
            new_y = matched_y[mask]
            n_new = len(new_x)

            if n_new <= self.low:
                break

            new_set = frozenset(zip(new_x.tolist(), new_y.tolist()))

            if n_new != n_prev:
                # Set changed in size — keep going
                n_prev = n_new
                prev_set = new_set
                inliers_x, inliers_y = new_x, new_y
            elif new_set == prev_set:
                # Converged: inlier set identical
                inliers_x, inliers_y = new_x, new_y
                break
            else:
                # Same count, different members — keep going
                n_prev = n_new
                prev_set = new_set
                inliers_x, inliers_y = new_x, new_y

        return fit_coeffs, inliers_x, inliers_y

    def _resample(self, fit_coeffs, inliers_x, inliers_y):
        """
        Try random subsets of the current inlier set to find a better model.

        Mirrors resample.m. Randomly draws a subset (size max(s, n/4)) from
        the current inliers, fits, rescores against all data. If a larger
        inlier set is found the process restarts from that set.

        Parameters
        ----------
        fit_coeffs : np.ndarray
        inliers_x, inliers_y : np.ndarray

        Returns
        -------
        fit_coeffs, inliers_x, inliers_y : np.ndarray
        """
        n_inliers = len(inliers_x)
        best_coeffs = fit_coeffs
        best_x = inliers_x.copy()
        best_y = inliers_y.copy()
        best_n = n_inliers

        i = 0
        while i < self.nrtrials:
            i += 1

            sub_size = max(self.config.sample_size, round(n_inliers / 4))
            sub_size = min(sub_size, n_inliers)

            idx = np.random.choice(n_inliers, sub_size, replace=False)
            sub_x = best_x[idx]
            sub_y = best_y[idx]

            try:
                H = self.polyfit(sub_x, sub_y, self.config.degree)
            except Exception:
                continue

            residual, matched_x, matched_y = self._match_bijective(
                self.sampler.y_for_x, self.unique_x, H
            )
            mask = residual < self.config.rms_tolerance
            inl_x = matched_x[mask]
            inl_y = matched_y[mask]

            if len(inl_x) <= self.low:
                continue

            H, inl_x, inl_y = self._rescore(H, inl_x, inl_y)
            n_new = len(inl_x)

            if n_new > best_n:
                best_coeffs = H
                best_x = inl_x
                best_y = inl_y
                best_n = n_new
                n_inliers = n_new
                i = 0  # restart with the improved set

        return best_coeffs, best_x, best_y

    def _pruneset(self, fit_coeffs, inliers_x, inliers_y):
        """
        Iteratively remove the worst-fitting point using tight threshold acc.

        Mirrors pruneset.m. Computes residuals for all current inliers, finds
        the point with the largest residual, and removes it if it exceeds acc
        and the remaining count is still above low. Refits after each removal.
        Stops when no residual exceeds acc or the count hits low.

        Parameters
        ----------
        fit_coeffs : np.ndarray
        inliers_x, inliers_y : np.ndarray

        Returns
        -------
        fit_coeffs, inliers_x, inliers_y : np.ndarray
        """
        put_x = inliers_x.copy()
        put_y = inliers_y.copy()

        while len(put_x) > self.low:
            residuals = np.abs(self.polyval(put_x, fit_coeffs) - put_y)
            worst_idx = int(np.argmax(residuals))

            if residuals[worst_idx] > self.acc and len(put_x) - 1 > self.low:
                put_x = np.delete(put_x, worst_idx)
                put_y = np.delete(put_y, worst_idx)
                try:
                    fit_coeffs = self.polyfit(put_x, put_y, self.config.degree)
                except Exception:
                    break
            else:
                break

        return fit_coeffs, put_x, put_y
