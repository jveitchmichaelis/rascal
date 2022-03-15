import logging
from itertools import combinations

import numpy as np
import pytest

from rascal.calibrator import Calibrator
from rascal.atlas import Atlas

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


@pytest.mark.xfail()
def test_loading_empty_calibrator_expect_fail():
    logger.info("Testing load without peaks.")
    Calibrator()


def test_initialisation():
    logger.info("Testing loading.")
    cal = Calibrator(peaks=np.arange(10))
    assert len(cal.peaks) == 10
    assert cal.atlas is None


def test_load_single_line():
    element_list = ["Hg", "Ar", "Xe", "Kr"]
    atlas = Atlas()

    cal = Calibrator(peaks=np.arange(10))
    for element in element_list:
        logger.info("Testing load single element: {}".format(element))
        atlas.add(elements=element)
        cal.set_atlas(atlas)
        assert len(cal.atlas.atlas_lines) > 0


def test_load_mutliple_lines():
    element_list = ["Hg", "Ar", "Xe", "Kr"]
    atlas = Atlas()

    cal = Calibrator(peaks=np.arange(10))
    for i in range(1, len(element_list) + 1):
        for elements in combinations(element_list, i):
            logger.info("Testing load elements: {}".format(elements))
        atlas.add(elements=element_list)
        cal.set_atlas(atlas)
        assert len(cal.atlas.atlas_lines) > 0


def test_setting_a_known_pair():
    logger.info("Testing adding a known pair.")
    cal = Calibrator(peaks=np.arange(10))
    cal.set_known_pairs(123, 456)
    assert cal.pix_known == 123
    assert cal.wave_known == 456


def test_setting_known_pairs():
    logger.info("Testing adding known pairs.")
    cal = Calibrator(peaks=np.arange(10))
    cal.set_known_pairs([123, 234], [456, 567])
    assert len(cal.pix_known) == 2
    assert len(cal.wave_known) == 2


@pytest.mark.xfail()
def test_setting_a_none_to_known_pairs_expect_fail():
    logger.info("Testing adding None as known pairs.")
    cal = Calibrator(peaks=np.arange(10))
    cal.set_known_pairs([1.0], [None])


@pytest.mark.xfail()
def test_setting_nones_to_known_pairs_expect_fail():
    logger.info("Testing adding None as known pairs.")
    cal = Calibrator(peaks=np.arange(10))
    cal.set_known_pairs([None], [None])
