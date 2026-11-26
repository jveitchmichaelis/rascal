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

from rascal.atlas import Atlas, NistSource
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
            element=element,
            source="nist",
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
        element="Xe",
        source="nist",
        min_wavelength=4000,
        max_wavelength=8000,
        min_intensity=min_intensity,
        brightest_n_lines=None,
    )

    for i in nist_atlas.get_intensities():
        assert i >= min_intensity, "Line intensity is below minimum"


"""
# TODO: Move to Atlas collection
def test_load_nist_all(elements):

    collection = AtlasCo    
    nist_atlas = Atlas(
        element=elements,
        line_list="nist",
        min_wavelength=0,
        max_wavelength=8000,
    )
"""


def test_check_nonzero_length_common_elements_single():

    for element in ["Xe", "Kr", "Ar", "Ne", "Cu"]:
        nist_atlas = Atlas(
            element=element,
            source="nist",
            min_wavelength=4000,
            max_wavelength=8000,
        )

        assert len(nist_atlas) > 0


def test_load_single_line():
    user_atlas = Atlas(
        element="Test",
        source="manual",
        wavelengths=[5.0],
        min_wavelength=0,
        max_wavelength=10,
    )
    cal = Calibrator(
        peaks=np.arange(10),
        atlas_lines=user_atlas.atlas_lines,
        config={"detector": {"detector_min_wave": 0.0, "detector_max_wave": 10.0}},
    )


def test_load_mutliple_lines():
    user_atlas = Atlas(
        element="Test",
        source="manual",
        wavelengths=np.arange(10),
        min_wavelength=0,
        max_wavelength=10,
    )
    cal = Calibrator(
        peaks=np.arange(10),
        atlas_lines=user_atlas.atlas_lines,
        config={"detector": {"detector_min_wave": 0.0, "detector_max_wave": 10.0}},
    )


def test_setting_a_known_pair():
    user_atlas = Atlas(
        element="Test",
        source="manual",
        wavelengths=np.arange(10),
        min_wavelength=0,
        max_wavelength=10,
    )
    logger.info("Testing adding a known pair.")
    cal = Calibrator(
        peaks=np.arange(10),
        atlas_lines=user_atlas.atlas_lines,
        config={"detector": {"detector_min_wave": 0.0, "detector_max_wave": 10.0}},
    )
    cal.set_known_pairs(123, 456)
    assert cal.pix_known == 123
    assert cal.wave_known == 456


def test_setting_known_pairs():
    user_atlas = Atlas(
        element="Test",
        source="manual",
        wavelengths=np.arange(10),
        min_wavelength=0,
        max_wavelength=10,
    )
    logger.info("Testing adding known pairs.")
    cal = Calibrator(
        peaks=np.arange(10),
        atlas_lines=user_atlas.atlas_lines,
        config={"detector": {"detector_min_wave": 0.0, "detector_max_wave": 10.0}},
    )
    cal.set_known_pairs([123, 234], [456, 567])
    assert len(cal.pix_known) == 2
    assert len(cal.wave_known) == 2


def test_setting_a_none_to_known_pairs_expect_fail():
    user_atlas = Atlas(
        element="Test",
        source="manual",
        wavelengths=np.arange(10),
        min_wavelength=0,
        max_wavelength=10,
    )
    logger.info("Testing adding None as known pairs.")
    cal = Calibrator(
        peaks=np.arange(10),
        atlas_lines=user_atlas.atlas_lines,
        config={"detector": {"detector_min_wave": 0.0, "detector_max_wave": 10.0}},
    )
    with pytest.raises(ValueError):
        cal.set_known_pairs([1.0], [None])


def test_setting_nones_to_known_pairs_expect_fail():
    user_atlas = Atlas(
        element="Test",
        source="manual",
        wavelengths=np.arange(10),
        min_wavelength=0,
        max_wavelength=10,
    )
    logger.info("Testing adding None as known pairs.")
    cal = Calibrator(
        peaks=np.arange(10),
        atlas_lines=user_atlas.atlas_lines,
        config={"detector": {"detector_min_wave": 0.0, "detector_max_wave": 10.0}},
    )
    with pytest.raises(ValueError):
        cal.set_known_pairs([None], [None])


def test_load_nist_all_comment_header():
    atlas = Atlas(
        element="Fe I",
        source="nist",
        nist_source=NistSource.NIST_ALL,
        min_wavelength=5900.0,
        max_wavelength=7100.0,
        min_intensity=20.0,
    )
    assert len(atlas) > 0


def test_atlas_brightest_n_lines():
    atlas_all = Atlas(
        element="Xe",
        source="nist",
        min_wavelength=4000.0,
        max_wavelength=8000.0,
        min_intensity=10.0,
        min_distance=0.0,
        brightest_n_lines=None,
    )
    atlas_top = Atlas(
        element="Xe",
        source="nist",
        min_wavelength=4000.0,
        max_wavelength=8000.0,
        min_intensity=10.0,
        min_distance=0.0,
        brightest_n_lines=10,
    )
    assert len(atlas_all) > 10
    assert len(atlas_top) == 10
    # Must be the 10 brightest lines, not the 10 faintest
    expected_min = sorted(atlas_all.get_intensities(), reverse=True)[9]
    assert min(atlas_top.get_intensities()) >= expected_min


def test_atlas_min_distance_filtering():
    atlas = Atlas(
        element="Ar",
        source="nist",
        min_wavelength=4000.0,
        max_wavelength=8000.0,
        min_intensity=10.0,
        min_distance=50.0,
    )
    lines = sorted(atlas.get_lines())
    assert len(lines) > 1
    diffs = np.diff(lines)
    assert np.all(diffs >= 50.0)


def test_atlas_vacuum_consistent_manual_and_nist():
    # NIST: vacuum=True returns longer (vacuum) wavelengths than vacuum=False (air)
    nist_air = Atlas(
        element="He I",
        source="nist",
        min_wavelength=4000.0,
        max_wavelength=7000.0,
        min_distance=0.0,
        vacuum=False,
    )
    nist_vac = Atlas(
        element="He I",
        source="nist",
        min_wavelength=4000.0,
        max_wavelength=7000.0,
        min_distance=0.0,
        vacuum=True,
    )
    assert len(nist_air) == len(nist_vac) > 0
    assert (np.array(nist_vac.get_lines()) > np.array(nist_air.get_lines())).all()

    # Manual: vacuum=True also returns longer (vacuum) wavelengths than vacuum=False (air)
    man_air = Atlas(
        element="Test",
        source="manual",
        wavelengths=[5000.0, 6000.0],
        min_wavelength=4000.0,
        max_wavelength=7000.0,
        vacuum=False,
    )
    man_vac = Atlas(
        element="Test",
        source="manual",
        wavelengths=[5000.0, 6000.0],
        min_wavelength=4000.0,
        max_wavelength=7000.0,
        vacuum=True,
    )
    assert (np.array(man_vac.get_lines()) > np.array(man_air.get_lines())).all()

    # Manual with input_vacuum=True, vacuum=False converts vacuum -> air
    man_vac_to_air = Atlas(
        element="Test",
        source="manual",
        wavelengths=[5000.0, 6000.0],
        min_wavelength=4000.0,
        max_wavelength=7000.0,
        input_vacuum=True,
        vacuum=False,
    )
    assert (np.array(man_vac_to_air.get_lines()) < np.array([5000.0, 6000.0])).all()

    # Manual with input_vacuum=True, vacuum=True leaves vacuum wavelengths unchanged
    man_vac_keep = Atlas(
        element="Test",
        source="manual",
        wavelengths=[5000.0, 6000.0],
        min_wavelength=4000.0,
        max_wavelength=7000.0,
        input_vacuum=True,
        vacuum=True,
    )
    assert np.allclose(man_vac_keep.get_lines(), [5000.0, 6000.0])

