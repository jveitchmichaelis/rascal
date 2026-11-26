import random
from functools import partialmethod

import numpy as np

# Suppress tqdm output
from tqdm import tqdm

from rascal.ransac import RansacSolver

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def test_ransac():

    x = np.arange(10)
    y = x**2

    config = {"max_tries": 100}

    solver = RansacSolver(x, y, config=config)

    solver.solve()
    assert solver.valid_solution


def test_ransac_outliers():
    rng = np.random.default_rng(0)
    random.seed(0)

    noise_mag = 5
    n = 15
    inlier_ratio = 0.5
    x = np.arange(n)
    eps = noise_mag * rng.random(n)
    y = x**2 + eps

    assert inlier_ratio <= 1

    for i in range(int((1 - inlier_ratio) * n)):
        y[random.randint(0, n - 1)] *= 2

    solver = RansacSolver(x, y)

    solver.solve()
    assert solver.valid_solution


def test_filter_close_updates_sampler_and_unique_x():
    # Two y values (3005.0, 3006.0) are within 3 * rms_tolerance (12.0) of each other;
    # the remaining y values are spaced by 50.0.
    x = np.arange(10, dtype=float)
    y = np.array(
        [3005.0, 3006.0, 3050.0, 3100.0, 3150.0, 3200.0, 3250.0, 3300.0, 3350.0, 3400.0]
    )
    solver = RansacSolver(
        x,
        y,
        config={"degree": 1, "sample_size": 2, "rms_tolerance": 4.0, "filter_close": True},
    )

    assert len(solver.unique_x) == len(np.unique(solver.x)) == 9
    assert set(solver.sampler.y_for_x.keys()) == set(solver.unique_x)
    assert 3005.0 not in np.concatenate(list(solver.sampler.y_for_x.values()))
