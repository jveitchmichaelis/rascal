import itertools
import numpy as np
import logging
import random
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

        for p in self.unique_x:
            self.y_for_x[p] = np.sort(np.unique(self.y[self.x == p]))

        self.x_combinations = itertools.combinations(
            self.unique_x, self.sample_size
        )

        self._setup()

    def _permutations_for_sample(self, sample):

        out = []

        for x in sample:

            # Special case for the first x
            if len(out) == 0:
                for y in self.y_for_x[x]:
                    out.append([[x], [y]])

            # Other x's
            else:
                for o in out:
                    for y in self.y_for_x[x]:
                        if y > o[1][-1]:
                            o[0].append(x)
                            o[1].append(y)
                            break

        for o in out:
            sample = o[0]
            y_sample = o[1]

            if len(y_sample) == self.sample_size:
                yield sample, y_sample

    def _generate_samples(self, max_samples=None):
        """

        Generate samples. This function will return samples that are unique
        and montonically increasing in both x and y.

        Yields
        ------
            sample: tuple, tuple
                sample of x and y values
        """

        # All unique variations of x with desired sample size
        self.logger.info(
            f"Generating samples of len {self.sample_size} from pool of {len(self.unique_x)} values"
        )

        for sample in tqdm(self.x_combinations):
            for x in self._permutations_for_sample(sample):
                yield x

    def _setup(self):
        """

        Internal setup function, should be defined by sub-classes

        """
        self.samples = self._generate_samples()

    def __iter__(self):
        """

        Obtain the next sample. For non-random samplers, this is a
        lazy function.

        Yields
        ------
            sample: tuple, tuple
                sample of x and y values
        """

        for sample in self.samples:
            yield sample


class UniformRandomSampler(Sampler):
    def _select_samples(self):
        """

        Select samples by randomly shuffling the list
        and then taking the first N.

        """
        # Get a subset of the x combinations
        all_x_combinations = [x for x in self.x_combinations]
        random.shuffle(all_x_combinations)
        self.x_combinations = all_x_combinations[: self.n_samples]

    def _setup(self):
        """

        Setup function for random sampling. First enumerates all possible samples.
        This should be relatively fast for typical rascal scenarios, but will slow
        significantly if there are is a dense many-to-one relationship between
        x and y.

        """

        # Fallback to returning all samples
        if self.n_samples == -1:
            self.logger.warning(
                "Returning all samples as number of samples was not provided."
            )

        self._select_samples()
        self.samples = [sample for sample in self._generate_samples()]

        if self.n_samples > len(self.samples):
            self.logger.warning(
                f"Returning all samples as max tries ({self.n_samples})is"
                + "greater than number of combinations ({len(self.samples)})"
            )
        else:
            random.shuffle(self.samples)
            self.samples = self.samples[: self.n_samples]


class WeightedRandomSampler(UniformRandomSampler):
    def _select_samples(self, max_bins=10):
        """

        Select samples by weighting each based on density
        and then randomly sampling with this cost.

        Parameters:
        -----------
            max_bins: int, optional
                Number of bins to use for histogram. Defaults to 10.
        """
        all_x_combinations = [x for x in self.x_combinations]
        random.shuffle(all_x_combinations)
        self.x_combinations = all_x_combinations[: self.n_samples]

        n_bins = min(max_bins, len(self.unique_x))
        hist, bin_edges = np.histogram(self.unique_x, bins=n_bins)
        prob = (
            1.0 / hist[np.digitize(self.unique_x, bin_edges, right=True) - 1]
        )
        self.prob = prob / np.sum(prob)

        prob_map = {}
        for i, x in enumerate(self.unique_x):
            prob_map[x] = self.prob[i]

        weight = []
        for x_hat in self.x_combinations:
            weight.append(sum([prob_map[x] for x in x_hat]))

        try:
            self.x_combinations = np.random.choice(
                self.x_combinations,
                size=self.n_samples,
                replace=False,
                p=weight,
            )
        except ValueError:
            self.logger.error(
                f"Unable to draw {self.n_samples} unique samples from population."
            )
