from rascal.sampler import Sampler, WeightedRandomSampler, UniformRandomSampler
import pytest
import logging

logger = logging.getLogger("test_samples")


@pytest.mark.xfail()
def test_mismatch_length():

    x = range(10)
    y = range(5)

    _ = Sampler(x, y, 3)


def test_exhaust_brute_sampler():

    x = range(5)
    y = range(5)
    sample_size = 2

    sampler = Sampler(x, y, sample_size=sample_size)

    samples = [x for x in sampler]

    import math

    assert len(samples) == math.factorial(len(x)) / (
        math.factorial(len(x) - sample_size) * math.factorial(sample_size)
    )


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
        sample_set.add(sample)

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
