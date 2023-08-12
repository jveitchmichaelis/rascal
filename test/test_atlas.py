import importlib.resources as import_resources
import logging
import os
import time
from functools import partialmethod
from itertools import combinations

import numpy as np
import pytest

# Suppress tqdm output
from tqdm import tqdm

from rascal.atlas import Atlas
from rascal.calibrator import Calibrator
from rascal.util import get_elements

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

HERE = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture
def elements():
    elements = get_elements()
    return elements


def test_load_nist_single(elements):

    min_wavelength = 3000
    max_wavelength = 8000

    for element in elements:
        time_start = time.perf_counter()

        nist_atlas = Atlas(
            elements=[element],
            line_list="nist",
            min_wavelength=min_wavelength,
            max_wavelength=max_wavelength,
            brightest_n_lines=None,
        )

        for line in nist_atlas.get_lines():
            assert line > (
                min_wavelength - nist_atlas.range_tolerance
            ), "Line is less than min wavelength"
            assert line < (
                max_wavelength + nist_atlas.range_tolerance
            ), "Line is greater than max wavelength"

        time_end = time.perf_counter()
        elapsed = time_end - time_start
        assert elapsed < 0.5


def test_load_with_min_intensity():

    min_intensity = 10

    nist_atlas = Atlas(
        elements=["Xe"],
        line_list="nist",
        min_wavelength=4000,
        max_wavelength=8000,
        min_intensity=min_intensity,
        brightest_n_lines=None,
    )

    for i in nist_atlas.get_intensities():
        assert i >= min_intensity, "Line intensity is below minimum"


def test_load_nist_all(elements):

    nist_atlas = Atlas(
        elements=elements,
        line_list="nist",
        min_wavelength=0,
        max_wavelength=8000,
    )

    assert len(nist_atlas) > 0


def test_check_nonzero_length_common_elements_single():

    for element in ["Xe", "Kr", "Ar", "Ne", "Cu"]:
        nist_atlas = Atlas(
            elements=element,
            line_list="nist",
            min_wavelength=4000,
            max_wavelength=8000,
        )

        assert len(nist_atlas) > 0


def test_load_single_line():
    user_atlas = Atlas(
        elements="Test",
        line_list="manual",
        wavelengths=[5.0],
        min_wavelength=0,
        max_wavelength=10,
    )
    cal = Calibrator(peaks=np.arange(10), atlas_lines=user_atlas.atlas_lines)


def test_load_mutliple_lines():
    user_atlas = Atlas(
        elements="Test",
        line_list="manual",
        wavelengths=np.arange(10),
        min_wavelength=0,
        max_wavelength=10,
    )
    cal = Calibrator(peaks=np.arange(10), atlas_lines=user_atlas.atlas_lines)


def test_setting_a_known_pair():
    user_atlas = Atlas(
        elements="Test",
        line_list="manual",
        wavelengths=np.arange(10),
        min_wavelength=0,
        max_wavelength=10,
    )
    logger.info("Testing adding a known pair.")
    cal = Calibrator(peaks=np.arange(10), atlas_lines=user_atlas.atlas_lines)
    cal.set_known_pairs(123, 456)
    assert cal.pix_known == 123
    assert cal.wave_known == 456


def test_setting_known_pairs():
    user_atlas = Atlas(
        elements="Test",
        line_list="manual",
        wavelengths=np.arange(10),
        min_wavelength=0,
        max_wavelength=10,
    )
    logger.info("Testing adding known pairs.")
    cal = Calibrator(peaks=np.arange(10), atlas_lines=user_atlas.atlas_lines)
    cal.set_known_pairs([123, 234], [456, 567])
    assert len(cal.pix_known) == 2
    assert len(cal.wave_known) == 2


@pytest.mark.xfail()
def test_setting_a_none_to_known_pairs_expect_fail():
    user_atlas = Atlas(
        elements="Test",
        line_list="manual",
        wavelengths=np.arange(10),
        min_wavelength=0,
        max_wavelength=10,
    )
    logger.info("Testing adding None as known pairs.")
    cal = Calibrator(peaks=np.arange(10), atlas_lines=user_atlas.atlas_lines)
    cal.set_known_pairs([1.0], [None])


@pytest.mark.xfail()
def test_setting_nones_to_known_pairs_expect_fail():
    user_atlas = Atlas(
        elements="Test",
        line_list="manual",
        wavelengths=np.arange(10),
        min_wavelength=0,
        max_wavelength=10,
    )
    logger.info("Testing adding None as known pairs.")
    cal = Calibrator(peaks=np.arange(10), atlas_lines=user_atlas.atlas_lines)
    cal.set_known_pairs([None], [None])


element_list = ["Hg", "Ar", "Xe", "Kr"]
user_atlas = Atlas(
    elements="Test",
    line_list="manual",
    wavelengths=np.arange(10),
    min_wavelength=0,
    max_wavelength=10,
)

cal = Calibrator(np.arange(10), user_atlas.atlas_lines)

"""
def test_get_summary_executive():
    cal.atlas_summary(mode="executive")


def test_get_summary_full():
    cal.atlas_summary(mode="full")


def test_get_summary_executive_return_string():
    something = cal.atlas_summary(mode="executive", return_string=True)
    assert type(something) == str


def test_get_summary_full_return_string():
    something = cal.atlas_summary(mode="full", return_string=True)
    assert type(something) == str


def test_save_executive_summary():
    filepath = os.path.join(
        HERE, "test_output", "test_save_atlas_executive_summary.txt"
    )
    cal.save_atlas_summary(
        mode="executive",
        filename=filepath,
    )
    assert os.path.exists(filepath)
    os.remove(filepath)


def test_save_full_summary():
    filepath = os.path.join(
        HERE, "test_output", "test_save_atlas_full_summary.txt"
    )
    cal.save_atlas_summary(
        mode="full",
        filename=filepath,
    )
    assert os.path.exists(filepath)
    os.remove(filepath)



def test_save_executive_summary_default():
    output_path = cal.save_atlas_summary(
        mode="executive",
    )
    os.remove(output_path)


def test_save_full_summary_default():
    output_path = cal.save_atlas_summary(
        mode="full",
    )
    os.remove(output_path)
"""
