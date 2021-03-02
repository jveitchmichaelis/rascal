import numpy as np
import pytest

from rascal.calibrator import HoughTransform

ht = HoughTransform()
ht.set_constraints(min_slope=1,
                   max_slope=4,
                   min_intercept=3000.,
                   max_intercept=5000.)
ht.generate_hough_points(x=(np.random.random(1000) * 3 + 1),
                         y=(np.random.random(1000) * 2000 + 3000),
                         num_slopes=1000)
ht.bin_hough_points(xbins=100, ybins=100)


def test_save_as_npy():
    ht.save(filename='test/test_hough_transform_npy', fileformat='npy')


def test_save_as_json():
    ht.save(filename='test/test_hough_transform_json', fileformat='json')


def test_load_npy():
    ht_loaded = HoughTransform()
    ht_loaded.load(filename='test/test_hough_transform_npy.npy',
                   filetype='npy')

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
    ht_loaded.load(filename='test/test_hough_transform_json.json',
                   filetype='json')

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
    ht_fail.load(filename='test/test_hough_transform_json.json',
                 filetype='lalala')


def test_hf_not_saved_to_disk():
    a = ht.save(filename='test/test_hough_transform_npy',
                fileformat='npy',
                to_disk=False)
    b = ht.save(filename='test/test_hough_transform_npy',
                fileformat='json',
                to_disk=False)
    c, d = ht.save(filename='test/test_hough_transform_npy',
                   fileformat='npy+json',
                   to_disk=False)
    assert a == c
    assert b == d
