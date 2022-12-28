import bisect
import itertools
import logging
import random
from functools import reduce
from operator import mul

import numpy as np
import scipy
from tqdm.auto import tqdm


class Sampler:
    def __init__(self, x, y, sample_size, n_samples=-1):
        """

        This is a base class for sampling functions for RANSAC. Users should
        provide a list of x and y, of equal length. Within rascal, these lists
        represent pairs (x_i, y_i) corresponding to potential peak and atlas
        matches. Many-to-one relations e.g. a peak that might map to one of several
        lines should be specified separately e.g. x=(1,1,2,3), y=(2,3,4,5).

        Parameters
        ----------
            x: (list(float))
                input x
            y: (list(float))
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

        self.sample_size = sample_size
        self.n_samples = n_samples

        self.y_for_x = {}
        self.unique_x = np.sort(np.unique(self.x))

        y_combinations = 0
        for p in self.unique_x:
            self.y_for_x[p] = np.sort(np.unique(self.y[self.x == p]))
            y_combinations += len(self.y_for_x[p])

        self.maximum_samples = scipy.special.comb(
            y_combinations, self.sample_size, exact=True
        )

        self.existing_samples = {}
        self.sample_x_prob = None
        self._setup()

    def _permutations_for_sample(self, x_sample):
        """Return x/y permutations for a sample

        Parameters
        ----------
        sample
            list of x values

        Yields
        ------
            x_sample, y_sample: lists of x and y values
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
        self.logger.info(
            f"Generating samples of len {self.sample_size} from pool of {len(self.unique_x)} values"
        )

        if self.n_samples < 0:
            for sample in tqdm(self.x_combinations):
                for x in self._permutations_for_sample(sample):
                    if x[0] is not None:
                        yield x
        else:

            # Sample from the iterator since we know the max number of permutations
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

        Yields
        ------
            sample: tuple, tuple
                sample of x and y values
        """
        yield from self.samples()


class UniformRandomSampler(Sampler):
    def get_sample(self):
        """Simple random sample from population

        Yields
        ------
            x_hat, y_hat: sample
        """
        resample = True
        retries = 5

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
            for _x in x_hat:

                y_choice = self.y_for_x[_x]

                # No samples yet, free choice

                i = 0

                if len(y_hat) > 0:

                    # Enforce monotonic increasing
                    i = bisect.bisect(y_choice, y_hat[-1])

                    # Pick a random y
                    if i == len(y_choice):
                        resample = True
                        break

                # Select at random
                _y = np.random.choice(y_choice[i:])
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

        # All unique variations of x with desired sample size
        if self.n_samples > 0:
            self.logger.info(
                f"Generating {self.n_samples} samples of length {self.sample_size}"
            )
            for i in range(self.n_samples):
                yield from self.get_sample()

        # Brute force
        else:
            super()._setup()
            yield from super().samples()


class WeightedRandomSampler(UniformRandomSampler):
    def _setup(self, max_bins=10):

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
