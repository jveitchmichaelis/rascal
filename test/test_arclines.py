import logging
from itertools import combinations

import numpy as np
import pytest

from rascal.calibrator import Calibrator

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
    assert len(cal.atlas) == 0
    assert len(cal.atlas_elements) == 0


def test_load_single_line():
    element_list = ["Hg", "Ar", "Xe", "Kr"]

    cal = Calibrator(peaks=np.arange(10))
    for element in element_list:
        logger.info("Testing load single element: {}".format(element))
        cal.add_atlas(element)
        assert len(cal.atlas) > 0
        assert len(cal.atlas_elements) > 0


def test_load_mutliple_lines():

    element_list = ["Hg", "Ar", "Xe", "Kr"]

    cal = Calibrator(peaks=np.arange(10))
    for i in range(1, len(element_list) + 1):
        for elements in combinations(element_list, i):
            logger.info("Testing load elements: {}".format(elements))
            cal.add_atlas(elements)
            assert len(cal.atlas) > 0
            assert len(cal.atlas_elements) > 0
