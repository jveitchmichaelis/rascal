import numpy as np
import os
import pytest

from rascal.calibrator import HoughTransform, Calibrator

HERE = os.path.dirname(os.path.realpath(__file__))

ht = HoughTransform()
ht.set_constraints(
    min_slope=1, max_slope=4, min_intercept=3000.0, max_intercept=5000.0
)
ht.generate_hough_points(
    x=(np.random.random(1000) * 3 + 1),
    y=(np.random.random(1000) * 2000 + 3000),
    num_slopes=1000,
)
ht.bin_hough_points(xbins=100, ybins=100)


def test_save_as_npy():
    ht.save(
        filename=os.path.join(HERE, "test_output", "test_hough_transform_npy"),
        fileformat="npy",
    )


def test_save_as_json():
    ht.save(
        filename=os.path.join(
            HERE, "test_output", "test_hough_transform_json"
        ),
        fileformat="json",
    )


def test_load_npy():
    ht_loaded = HoughTransform()
    ht_loaded.load(
        filename=os.path.join(
            HERE, "test_output", "test_hough_transform_npy.npy"
        ),
        filetype="npy",
    )

    assert (ht_loaded.hough_points == ht.hough_points).all
    assert (ht_loaded.hist == ht.hist).all
    assert (ht_loaded.xedges == ht.xedges).all
    assert (ht_loaded.yedges == ht.yedges).all
    assert ht_loaded.min_slope == ht.min_slope
    assert ht_loaded.max_slope == ht.max_slope
    assert ht_loaded.min_intercept == ht.min_intercept
    assert ht_loaded.max_intercept == ht.max_intercept


def test_load_json():
    ht_loaded = HoughTransform()
    ht_loaded.load(
        filename=os.path.join(
            HERE, "test_output", "test_hough_transform_json.json"
        ),
        filetype="json",
    )

    assert (ht_loaded.hough_points == ht.hough_points).all
    assert (ht_loaded.hist == ht.hist).all
    assert (ht_loaded.xedges == ht.xedges).all
    assert (ht_loaded.yedges == ht.yedges).all
    assert ht_loaded.min_slope == ht.min_slope
    assert ht_loaded.max_slope == ht.max_slope
    assert ht_loaded.min_intercept == ht.min_intercept
    assert ht_loaded.max_intercept == ht.max_intercept


@pytest.mark.xfail()
def test_load_fail():
    ht_fail = HoughTransform()
    ht_fail.load(
        filename=os.path.join(
            HERE, "test_output", "test_hough_transform_json.json"
        ),
        filetype="lalala",
    )


def test_ht_not_saved_to_disk():
    a = ht.save(
        filename=os.path.join(HERE, "test_output", "test_hough_transform_npy"),
        fileformat="npy",
        to_disk=False,
    )
    b = ht.save(
        filename=os.path.join(HERE, "test_output", "test_hough_transform_npy"),
        fileformat="json",
        to_disk=False,
    )
    c, d = ht.save(
        filename=os.path.join(HERE, "test_output", "test_hough_transform_npy"),
        fileformat="npy+json",
        to_disk=False,
    )
    assert a == c
    assert b == d


def test_extending_ht():

    ht2 = HoughTransform()
    ht2.set_constraints(
        min_slope=1, max_slope=4, min_intercept=7000.0, max_intercept=9000.0
    )
    ht2.generate_hough_points(
        x=(np.random.random(1000) * 3 + 1),
        y=(np.random.random(1000) * 2000 + 7000),
        num_slopes=1000,
    )

    ht2.add_hough_points(ht)

    assert set(ht.hough_points[:, 0]).issubset(set(ht2.hough_points[:, 0]))
    assert set(ht.hough_points[:, 1]).issubset(set(ht2.hough_points[:, 1]))

    shape_now = np.shape(ht2.hough_points)

    ht2.add_hough_points(np.ones((100, 2)))
    assert np.shape(ht2.hough_points) == (shape_now[0] + 100, 2)


@pytest.mark.xfail()
def test_extending_ht_expect_fail():

    ht.add_hough_points(np.ones(100))


def test_loading_ht_into_calibrator():
    c = Calibrator(np.arange(10))
    c.load_hough_transform(
        os.path.join(HERE, "test_output", "test_hough_transform_npy")
    )
    c.save_hough_transform(
        os.path.join(HERE, "test_output", "test_hough_transform_npy_2")
    )
