import random
from functools import partialmethod

import numpy as np

# Suppress tqdm output
from tqdm import tqdm

from rascal.ransac import RansacSolver, SolveResult

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


def test_update_best_validates_post_robust_inliers():
    # Construct points where the pre-robust result claims 8 inliers,
    # but re-matching with the robust fit yields only 5 inliers (< minimum_matches=7).
    x = np.arange(10, dtype=float)
    y = 3000.0 + 5.0 * x
    # Make 5 points match the line and 5 points be alternating outliers (> rms_tolerance=4.0)
    y[5:] += np.array([15.0, -15.0, 15.0, -15.0, 15.0])

    solver = RansacSolver(
        x,
        y,
        config={"degree": 1, "sample_size": 2, "rms_tolerance": 4.0},
        fit_valid_fn=lambda res: len(res.inliers_x) >= 7,
    )

    # Pre-robust result with 8 fake small residuals so pre-robust _fit_valid would pass,
    # while post-robust bijective re-matching only finds 5 true inliers (< 7).
    fake_result = SolveResult(
        fit_coeffs=np.array([3000.0, 5.0]),
        cost=1.0,
        x=x,
        y=y,
        residual=np.array([0.1] * 8 + [50.0] * 2),
        rms_tolerance=4.0,
    )
    solver._update_best(fake_result)
    assert not solver.valid_solution


def test_match_hungarian_outlier_does_not_steal_inlier():
    # Three peaks with true linear relation y = 1000 + 10 * x:
    #   x=0 -> true y=1000, x=1 -> true y=1010, x=2 -> true y=1020
    # Peak x=2 is an outlier whose true line 1020 is NOT in its candidates;
    # its only candidate is 1010 (error 10.0 > rms_tolerance=4.0).
    # Peak x=1 has candidates [1000, 1010], peak x=0 has [1000, 1050].
    # An outlier peak must NEVER steal 1010 from x=1 (or trigger a cascade
    # stealing 1000 from x=0) just to avoid an unmatched sentinel cost.
    x = np.array([0.0, 0.0, 1.0, 1.0, 2.0])
    y = np.array([1000.0, 1050.0, 1000.0, 1010.0, 1010.0])
    solver = RansacSolver(
        x,
        y,
        config={"degree": 1, "sample_size": 2, "rms_tolerance": 4.0},
    )
    coeffs = np.array([1000.0, 10.0])
    err, matched_x, matched_y = solver._match_bijective(
        solver.sampler.y_for_x, solver.unique_x, coeffs
    )
    inlier_mask = err < 4.0
    matched_pairs = set(zip(matched_x[inlier_mask], matched_y[inlier_mask]))
    assert matched_pairs == {(0.0, 1000.0), (1.0, 1010.0)}


def test_hough_weight_zero_disables_hough_weighting():
    from rascal.houghtransform import HoughTransform

    x = np.linspace(0, 100, 12)
    y = 3000.0 + 2.0 * x
    y[-3:] += 50.0  # 3 outliers so M-SAC clipped sum > 0
    ht = HoughTransform()
    ht.set_constraints(min_slope=1.0, max_slope=3.0, min_intercept=2900.0, max_intercept=3100.0)
    ht.generate_hough_points(x, y, num_slopes=50)
    ht.bin_hough_points(xbins=10, ybins=10)

    solver = RansacSolver(
        x,
        y,
        hough=ht,
        config={"degree": 1, "sample_size": 2, "hough_weight": 0.0, "max_tries": 50},
    )
    solver.solve()
    assert solver.valid_solution
    expected_msac = 3 * solver.config.rms_tolerance / (len(x) - 2 + 1)
    assert np.isclose(solver.best_result.cost, expected_msac, rtol=1e-3)


def test_hough_weight_scales_relative_preference():
    from rascal.houghtransform import HoughTransform

    x = np.linspace(0, 100, 12)
    y = 3000.0 + 2.0 * x
    ht = HoughTransform()
    ht.set_constraints(
        min_slope=1.0, max_slope=3.0, min_intercept=2900.0, max_intercept=3100.0
    )
    ht.generate_hough_points(x, y, num_slopes=50)
    ht.bin_hough_points(xbins=10, ybins=10)

    # Two models with identical M-SAC residuals: one aligned with the Hough peak
    # and one shifted away into a lower-density Hough region.
    res_peak = SolveResult(
        fit_coeffs=np.array([3000.0, 2.0]),
        x=x,
        y=y,
        residual=np.full(len(x), 1.0),
        rms_tolerance=4.0,
    )
    res_off = SolveResult(
        fit_coeffs=np.array([3015.0, 2.0]),
        x=x,
        y=y,
        residual=np.full(len(x), 1.0),
        rms_tolerance=4.0,
    )

    ratios = []
    for hw in [0.0, 1.0, 5.0]:
        solver = RansacSolver(
            x, y, hough=ht, config={"degree": 1, "sample_size": 2, "hough_weight": hw}
        )
        ratios.append(solver._cost(res_off) / solver._cost(res_peak))

    # At hw=0 the ratio is 1.0 (pure M-SAC); larger hw must strictly increase
    # the preference for the high-Hough-density model rather than cancelling out,
    # while remaining bounded (no 1e9 explosion when density is low/zero).
    assert np.isclose(ratios[0], 1.0)
    assert ratios[0] < ratios[1] < 10.0
    assert ratios[2] > ratios[1] * 1.1


def test_update_best_uses_post_robust_cost():
    # 10 points on y = 3000 + 5*x. Give _update_best a result with a high
    # pre-robust cost (e.g. from a noisy minimal sample), which after robust
    # refit + re-expansion matches all 10 points with ~0 residual.
    # best_result.cost must be updated to the low post-robust cost (~0),
    # not left at the stale pre-robust cost (2.5), so a worse 6-inlier fit
    # cannot later overwrite it.
    x = np.arange(10, dtype=float)
    y = 3000.0 + 5.0 * x

    solver = RansacSolver(
        x,
        y,
        config={"degree": 1, "sample_size": 2, "rms_tolerance": 4.0},
    )
    good_pre = SolveResult(
        fit_coeffs=np.array([3000.0, 5.0]),
        cost=2.5,
        x=x,
        y=y,
        residual=np.array([0.0] * 10),
        rms_tolerance=4.0,
    )
    solver._update_best(good_pre)
    assert solver.valid_solution
    assert solver.best_result.cost < 1e-6

    # Now try to overwrite with a worse fit whose pre-robust cost (2.0) is
    # below the stale pre-robust cost (2.5) but well above the true post-robust
    # cost (~0). It must be rejected.
    worse = SolveResult(
        fit_coeffs=np.array([3000.0, 5.5]),
        cost=2.0,
        x=x[:6],
        y=y[:6],
        residual=np.abs(3000.0 + 5.5 * x[:6] - y[:6]),
        rms_tolerance=4.0,
    )
    solver._update_best(worse)
    assert len(solver.best_result.x) == 10


def test_sampler_not_updated_without_consensus_beyond_sample_size():
    # 10 non-collinear points where any 2-point linear fit only matches its own
    # 2 sampled points (no consensus beyond sample_size=2). ProbabilisticSampler
    # must not drift its sampling counts on non-consensus minimal samples.
    x = np.arange(10, dtype=float)
    y = 1000.0 + 10.0 * x**2  # strictly convex: no 3 points are collinear
    solver = RansacSolver(
        x,
        y,
        config={
            "degree": 1,
            "sample_size": 2,
            "rms_tolerance": 1.0,
            "max_tries": 20,
            "sampler": "probabilistic",
        },
    )
    np.random.seed(0)
    solver.solve()
    assert all(count == 1 for count in solver.sampler.sample_count_x.values())


def test_update_best_iteratively_reexpands_inliers():
    # Realistic degree-4 polynomial with small measurement noise across [0, 100].
    # An initial partial match on the left half (x <= 45) yields a noisy fit
    # that extrapolates past rms_tolerance at the far right after a single refit,
    # requiring iterative refit + re-expansion to pull in all 21 points.
    rng = np.random.default_rng(42)
    x = np.linspace(0, 100, 21)
    true_coeffs = np.array([3000.0, 2.0, 0.005, -2e-5, 1e-7])
    y = np.polynomial.polynomial.polyval(x, true_coeffs) + rng.normal(0, 0.4, len(x))

    solver = RansacSolver(
        x,
        y,
        config={"degree": 4, "sample_size": 5, "rms_tolerance": 4.0},
    )
    # Initial partial inlier set: only x <= 40 (9 points)
    init_res = np.where(x <= 40.0, 0.5, 50.0)
    pre = SolveResult(
        fit_coeffs=true_coeffs,
        cost=100.0,
        x=x,
        y=y,
        residual=init_res,
        rms_tolerance=4.0,
    )
    solver._update_best(pre)
    assert solver.valid_solution
    assert len(solver.best_result.x) == len(x)


def test_ransac_passes_hough_slope_bounds_and_skips_non_monotonic_fits():
    from rascal.houghtransform import HoughTransform

    x = np.linspace(0, 100, 12)
    y = 3000.0 + 2.0 * x
    ht = HoughTransform()
    ht.set_constraints(
        min_slope=1.5, max_slope=2.5, min_intercept=2900.0, max_intercept=3100.0
    )
    ht.generate_hough_points(x, y, num_slopes=50)
    ht.bin_hough_points(xbins=10, ybins=10)

    solver = RansacSolver(
        x, y, hough=ht, config={"degree": 2, "sample_size": 3, "max_tries": 20}
    )
    assert solver.sampler.min_slope is not None
    assert solver.sampler.max_slope is not None
    assert solver.sampler.min_slope <= 1.5
    assert solver.sampler.max_slope >= 2.5

    # A turning quadratic y = 3000 + 2*x - 0.05*x^2 peaks at x=20 and decreases
    # for x > 20, so _is_monotonic must reject it before bijective matching.
    assert not solver._is_monotonic(np.array([3000.0, 2.0, -0.05]))
    assert solver._is_monotonic(np.array([3000.0, 2.0, 0.005]))
