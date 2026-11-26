#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configure the sampler

"""

import bisect
import itertools
import logging
import random
from typing import Union

import numpy as np
import scipy
from tqdm.auto import tqdm


class Sampler:
    """
    Josh will write something here.

    """

    def __init__(
        self,
        x: Union[np.ndarray, list],
        y: Union[np.ndarray, list],
        sample_size: int,
        n_samples: int = -1,
        weights: Union[np.ndarray, list, None] = None,
        min_slope: Union[float, None] = None,
        max_slope: Union[float, None] = None,
    ):
        """
        This is a base class for sampling functions for RANSAC. Users should
        provide a list of x and y, of equal length. Within rascal, these
        lists represent pairs (x_i, y_i) corresponding to potential peak and
        atlas matches. Many-to-one relations e.g. a peak that might map to
        one of several lines should be specified separately
        e.g. x=(1,1,2,3), y=(2,3,4,5).

        Parameters
        ----------
        x: np.ndarray or list of float
            input x
        y: np.ndarray or list of float
            input y
        sample_size: int
            size of sample to return
        n_samples: int
            number of samples to return

        """

        self.logger = logging.getLogger("sampler")

        self.x = np.array(x).reshape(-1)
        self.y = np.array(y).reshape(-1)

        assert len(self.x) == len(self.y)

        if weights is not None:
            self.weights = np.asarray(weights, dtype=np.float64).reshape(-1)
            assert len(self.weights) == len(self.x)
            self.w_for_x = {}
        else:
            self.weights = None
            self.w_for_x = None

        self.sample_size = sample_size
        self.n_samples = n_samples
        self.min_slope = min_slope
        self.max_slope = max_slope

        self.y_for_x = {}
        self.unique_x = np.sort(np.unique(self.x))

        y_combinations = 0
        for p in self.unique_x:
            x_mask = self.x == p
            ys_p = self.y[x_mask]
            uniq_y, inv = np.unique(ys_p, return_inverse=True)
            self.y_for_x[p] = uniq_y
            if self.w_for_x is not None:
                ws_p = np.zeros(len(uniq_y), dtype=np.float64)
                np.add.at(ws_p, inv, self.weights[x_mask])
                self.w_for_x[p] = ws_p
            y_combinations += len(self.y_for_x[p])

        self.maximum_samples = scipy.special.comb(
            y_combinations, self.sample_size, exact=True
        )

        self.existing_samples = {}
        self.sample_x_prob = None
        self._setup()

    def _permutations_for_sample(self, x_sample: Union[list, np.ndarray]):
        """
        Return x/y permutations for a sample

        Parameters
        ----------
        sample: list of x values

        Returns
        -------
        x_sample: np.ndarray
            list of x values

        """

        for y_sample in itertools.product(
            *[self.y_for_x[x] for x in x_sample]
        ):

            # Filter non-monotonic y, can happen when two peaks match
            # to two close calibration lines
            if not np.all(y_sample[1:] > y_sample[:-1], axis=0):
                continue

            yield x_sample, y_sample
            return

        yield None, None

    def _setup(self):
        """
        Internal setup function, should be defined by sub-classes

        """
        self.x_combinations = itertools.combinations(
            self.unique_x, self.sample_size
        )

    def update(self, matched_x, matched_y):
        pass

    def samples(self):
        """
        Josh will write something here.

        """

        self.logger.info(
            f"Generating samples of len {self.sample_size} from pool of "
            + f"{len(self.unique_x)} values"
        )

        if self.n_samples < 0:
            for sample in tqdm(self.x_combinations):
                for x in self._permutations_for_sample(sample):
                    if x[0] is not None:
                        yield x
        else:

            # Sample from the iterator since we know the max number of
            # permutations
            _choices = random.sample(
                population=range(self.maximum_samples), k=self.n_samples
            )
            choices = {}
            for choice in _choices:
                choices[choice] = 1

            for idx, sample in enumerate(self.x_combinations):
                for x in self._permutations_for_sample(sample):
                    if idx in choices:
                        yield x

    def __iter__(self):
        """
        Obtain the next sample. For non-random samplers, this is a
        lazy function.

        Returns
        -------
        sample: tuple, tuple
            sample of x and y values
        """

        yield from self.samples()


class UniformRandomSampler(Sampler):
    """
    Josh will write something here.

    """

    def get_sample(self):
        """
        Simple random sample from population

        Yields
        ------
            x_hat, y_hat: sample

        """

        resample = True
        retries = 20

        while resample and retries > 0:

            retries -= 1

            x_hat = np.sort(
                np.random.choice(
                    self.unique_x,
                    self.sample_size,
                    replace=False,
                    p=self.sample_x_prob,
                )
            )

            y_hat = []
            sampled_y = {}

            # Get a random y for this x
            for idx_k, _x in enumerate(x_hat):

                y_choice = self.y_for_x[_x]

                # No samples yet, free choice
                lo = 0
                hi = len(y_choice)

                if len(y_hat) > 0:

                    # Enforce monotonic increasing (and physical secant slope
                    # bounds when provided) to avoid unphysical jumps before
                    # fitting.
                    if (
                        self.min_slope is not None
                        and self.max_slope is not None
                    ):
                        dx = _x - x_hat[idx_k - 1]
                        y_lo = y_hat[-1] + self.min_slope * dx
                        y_hi = y_hat[-1] + self.max_slope * dx
                        lo = bisect.bisect_left(y_choice, y_lo)
                        hi = bisect.bisect_right(y_choice, y_hi)
                        if lo < len(y_choice) and y_choice[lo] <= y_hat[-1]:
                            lo = bisect.bisect_right(y_choice, y_hat[-1])
                    else:
                        lo = bisect.bisect(y_choice, y_hat[-1])

                    # Pick a random y
                    if lo >= hi:
                        resample = True
                        break

                # Select at random (weighted by Hough consensus if available)
                if self.w_for_x is not None:
                    w_slice = self.w_for_x[_x][lo:hi]
                    w_sum = w_slice.sum()
                    p_y = w_slice / w_sum if w_sum > 0 else None
                    _y = np.random.choice(y_choice[lo:hi], p=p_y)
                else:
                    _y = np.random.choice(y_choice[lo:hi])
                sampled_y[_y] = 1
                y_hat.append(_y)

            if len(y_hat) == len(x_hat):
                sample = tuple([*x_hat, *y_hat])

                # Check we've not used this point before
                if sample not in self.existing_samples:
                    self.existing_samples[sample] = 1
                    yield (x_hat, y_hat)
                    resample = False

    def samples(self):
        """
        Josh will write something here.

        """

        # All unique variations of x with desired sample size
        if self.n_samples > 0:
            self.logger.info(
                f"Generating {self.n_samples} samples of length "
                + f"{self.sample_size}."
            )
            for _ in range(self.n_samples):
                yield from self.get_sample()

        # Brute force
        else:
            super()._setup()
            yield from super().samples()


class WeightedRandomSampler(UniformRandomSampler):
    """
    Josh will write something here.

    """

    def _setup(self, max_bins: int = 10):
        """
        Josh will write something here.

        Parameters
        ----------
        max_bins: int
            ???

        """

        n_bins = min(max_bins, len(self.unique_x))
        hist, bin_edges = np.histogram(self.unique_x, bins=n_bins)
        prob = (
            1.0 / hist[np.digitize(self.unique_x, bin_edges, right=True) - 1]
        )
        self.sample_x_prob = prob / np.sum(prob)


class ProbabilisticSampler(UniformRandomSampler):
    def _setup(self):

        self.sample_count = {}
        self.sample_count_x = {}
        self.pairs = []

        for x in self.unique_x:
            for y in self.y_for_x[x]:
                self.sample_count_x[x] = 1
                self.sample_count[(x, y)] = 1
                self.pairs.append((x, y))

        self.samples_returned = 0

        assert (
            self.n_samples > 0
        ), "Probabilistic sampler requires a fixed number of iterations"

    def update(self, matched_x, matched_y):

        for x in matched_x:
            self.sample_count_x[x] += 1

        self.sample_x_prob = np.ones_like(self.unique_x, dtype=np.float64)
        for i, x in enumerate(self.unique_x):
            self.sample_x_prob[i] = float(self.sample_count_x[x])

        self.sample_x_prob /= self.sample_x_prob.sum()

    """
    def get_sample(self):

        prob = np.zeros(len(self.pairs))

        for i, pair in enumerate(self.pairs):
            prob[i] = self.sample_count[pair]

        prob /= prob.sum()

        idx = np.arange(len(self.pairs))

        sample = np.sort(
            np.random.choice(
                a=idx,
                size=self.sample_size,
                replace=False,
                p=prob,
            )
        )

        x_hat = []
        y_hat = []
        for s in sample:
            _x, _y = self.pairs[s]
            x_hat.append(_x)
            y_hat.append(_y)

        return np.array(x_hat), np.array(y_hat)
    """

    def __iter__(self):
        """

        Obtain the next sample probabilistically.

        Yields
        ------
            sample: tuple, tuple
                sample of x and y values
        """

        while self.samples_returned < self.n_samples:
            self.samples_returned += 1
            yield from self.get_sample()
