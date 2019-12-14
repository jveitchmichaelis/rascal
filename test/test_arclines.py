from rascal.calibrator import Calibrator
from itertools import combinations
import pytest
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

def test_no_elements():
    logger.info("Testing no elements")
    cal = Calibrator()
    assert len(cal.atlas) > 0
    assert len(cal.atlas_elements) > 0

def test_empty_elements():
    logger.info("Testing load empty element list")
    with pytest.raises(ValueError):
         cal = Calibrator(elements=[])

def test_load_single_line():
    element_list = ["Hg", "Ar", "Xe", "Kr"]

    for element in element_list:
        logger.info("Testing load single element: {}".format(element))
        cal = Calibrator(elements=element)
        assert len(cal.atlas) > 0
        assert len(cal.atlas_elements) > 0

def test_load_mutliple_lines():

    element_list = ["Hg", "Ar", "Xe", "Kr"]

    for i in range(1, len(element_list)+1):
        for elements in combinations(element_list, i):
            logger.info("Testing load elements: {}".format(elements))
            cal = Calibrator(elements=elements)
            assert len(cal.atlas) > 0
            assert len(cal.atlas_elements) > 0