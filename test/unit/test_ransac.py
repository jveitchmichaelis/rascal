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

    noise_mag = 5
    n = 15
    inlier_ratio = 0.5
    x = np.arange(n)
    eps = noise_mag * np.random.random(n)
    y = x**2 + eps

    assert inlier_ratio <= 1

    for i in range(int((1 - inlier_ratio) * n)):
        y[random.randint(0, n - 1)] *= 2

    solver = RansacSolver(x, y)

    solver.solve()
    assert solver.valid_solution
