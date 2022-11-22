from rascal.ransac import RansacSolver
import numpy as np


def test_ransac():

    x = np.arange(10)
    y = x**2

    solver = RansacSolver(x, y)

    solver.solve()
