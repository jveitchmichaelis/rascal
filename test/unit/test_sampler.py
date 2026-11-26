import logging
from functools import partialmethod

import pytest

# Suppress tqdm output
from tqdm import tqdm

from rascal.sampler import Sampler, UniformRandomSampler, WeightedRandomSampler

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

logger = logging.getLogger("test_samples")


def test_mismatch_length():
    with pytest.raises(AssertionError):
        x = range(10)
        y = range(5)
        _ = Sampler(x, y, 3)


def test_exhaust_brute_sampler():

    x = range(5)
    y = range(5)
    sample_size = 2

    sampler = Sampler(x, y, sample_size=sample_size, n_samples=-1)

    samples = [x for x in sampler.samples()]

    # Note that because we enforce monotonicity, it is generally
    # dificult to check that the number of samples returned is correct
    # e.g. (0, 1)
    assert len(samples) == sampler.maximum_samples


def test_exhaust_weighted_random_sampler():

    x = range(5)
    y = range(5)

    sampler = WeightedRandomSampler(x, y, sample_size=3)
    samples = [x for x in sampler]

    assert len(samples) > 0


def test_exhaust_uniform_random_sampler():

    x = range(5)
    y = range(5)

    sampler = UniformRandomSampler(x, y, sample_size=3)
    samples = [x for x in sampler]

    assert len(samples) > 0


def test_truncated_random_sample():

    x = [1, 1, 2, 3, 4, 4, 5]
    y = [0, 1, 2, 3, 4, 5, 6]
    n_samples = 10

    sampler = UniformRandomSampler(x, y, sample_size=3, n_samples=n_samples)
    samples = [x for x in sampler]

    assert len(samples) > 0

    sample_set = set()

    for sample in samples:
        x_hat, y_hat = sample
        sample_set.add(tuple([*x_hat, *y_hat]))

    assert len(sample_set) == len(samples)
    assert len(samples) == n_samples


def test_monotonicity():

    x = [0, 0, 1, 1]
    y = [1, 2, 1, 2]

    sampler = Sampler(x, y, sample_size=2)
    samples = [x for x in sampler]

    """
    We expect only one correct sample:

    [0,1], [1,2]
    
    whereas in principle all cominations of
    [0,1] x [1,2 ] are generated

    """

    assert len(samples) == 1


def test_weighted_y_sampling_prefers_high_weight_candidates():
    import numpy as np

    np.random.seed(0)
    # 10 unique x values, each with 1 true high-weight y and 4 low-weight decoy y values
    x = []
    y = []
    weights = []
    for xi in range(10):
        true_y = 100.0 * (xi + 1)
        x.append(xi)
        y.append(true_y)
        weights.append(100.0)
        for d in range(1, 5):
            x.append(xi)
            y.append(true_y + d)
            weights.append(1.0)

    sampler = UniformRandomSampler(
        x, y, sample_size=3, n_samples=50, weights=weights
    )
    samples = list(sampler)
    assert len(samples) > 0
    true_picks = sum(
        1
        for x_hat, y_hat in samples
        for xi, yi in zip(x_hat, y_hat)
        if np.isclose(yi, 100.0 * (xi + 1))
    )
    total_picks = sum(len(x_hat) for x_hat, _ in samples)
    # Uniform would pick true_y ~20% of the time; weighted (100 vs 4) should pick >80%
    assert true_picks / total_picks > 0.8


def test_slope_bounded_sampling_enforces_secant_bounds():
    import numpy as np

    np.random.seed(0)
    # 10 x values spaced by 10 px, true slope = 2.0 A/px (y = 3000 + 2*x),
    # plus decoy y values with huge/tiny slopes relative to neighbors.
    x = []
    y = []
    for xi in range(0, 100, 10):
        true_y = 3000.0 + 2.0 * xi
        for cand_y in [true_y, true_y + 150.0, true_y + 300.0]:
            x.append(float(xi))
            y.append(cand_y)

    min_slope = 1.0
    max_slope = 3.0
    sampler = UniformRandomSampler(
        x,
        y,
        sample_size=4,
        n_samples=50,
        min_slope=min_slope,
        max_slope=max_slope,
    )
    samples = list(sampler)
    assert len(samples) > 0
    for x_hat, y_hat in samples:
        secants = np.diff(y_hat) / np.diff(x_hat)
        assert np.all(secants >= min_slope)
        assert np.all(secants <= max_slope)
