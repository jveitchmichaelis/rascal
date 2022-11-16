import itertools
import numpy as np
import logging
import random


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
            self.y_for_x[p] = self.y[self.x == p]

        self.samples = self._generate_samples()
        self._setup()

    def _generate_samples(self):
        """

        Generate samples. This function will return samples that are unique
        and montonically increasing in both x and y.

        Yields
        ------
            sample: tuple, tuple
                sample of x and y values
        """

        # All unqiue variations of x with desired sample size
        samples = itertools.combinations(self.unique_x, self.sample_size)

        for sample in samples:

            # Possible y's for this sample
            for y_sample in itertools.product(
                *[self.y_for_x[x] for x in sample]
            ):

                # Filter duplicate y's
                if len(np.unique(y_sample)) != len(y_sample):
                    continue

                # Filter non-monotonic y, can happen when two peaks match
                # to two close calibration lines
                if not np.all(y_sample[1:] >= y_sample[:-1], axis=0):
                    continue

                yield sample, y_sample

    def _setup(self):
        """

        Internal setup function, should be defined by sub-classes

        """
        pass

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
        random.shuffle(self.samples)
        self.samples = self.samples[: self.n_samples]

    def _setup(self):
        """

        Setup function for random sampling. First enumerates all possible samples.
        This should be relatively fast for typical rascal scenarios, but will slow
        significantly if there are is a dense many-to-one relationship between
        x and y.

        """

        # Generate
        self.samples = [sample for sample in self.samples]

        # Fallback to returning all samples
        if self.n_samples == -1:
            self.logger.warning(
                "Returning all samples as number of samples was not provided."
            )
        elif self.n_samples > len(self.samples):
            self.logger.warning(
                f"Returning all samples as max tries ({self.n_samples})is"
                + "greater than number of combinations ({len(self.samples)})"
            )
        else:
            self._select_samples()


class WeightedRandomSampler(UniformRandomSampler):
    def _select_samples(self, max_bins=10):
        """

        Select samples by weighting each based on Hough density
        and then randomly sampling with this cost.

        Parameters:
        -----------
            max_bins: int, optional
                Number of bins to use for Hough histogram. Defaults to 10.
        """
        # Hough density weighting
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
        for sample in self.samples:
            x_hat, _ = sample
            weight.append(sum([prob_map[x] for x in x_hat]))

        try:
            self.samples = np.random.choice(
                self.samples, size=self.n_samples, replace=False, p=weight
            )
        except ValueError:
            self.logger.error(
                f"Unable to draw {self.n_samples} unique samples from population."
            )
