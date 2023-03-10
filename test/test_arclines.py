import logging
import os
from functools import partialmethod
from itertools import combinations

import numpy as np
import pytest
from rascal.atlas import Atlas
from rascal.calibrator import Calibrator

# Suppress tqdm output
from tqdm import tqdm

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

HERE = os.path.dirname(os.path.realpath(__file__))


"""
def test_loading_empty_calibrator():
    logger.info("Testing load without peaks.")
    Calibrator()


def test_initialisation():
    logger.info("Testing loading.")
    cal = Calibrator(peaks=np.arange(10))
    assert len(cal.peaks) == 10
    assert cal.atlas is None
"""


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
