import os
from functools import partialmethod
from unittest.mock import patch

import numpy as np
import pkg_resources
import pytest

# Suppress tqdm output
from tqdm import tqdm

from rascal import util

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

base_dir = os.path.dirname(os.path.abspath(__file__))

pressure = np.array([9, 10, 12, 10, 10, 10, 10, 10]) * 1e4
temperature = np.array([20, 20, 20, 10, 30, 20, 20, 20])
relative_humidity = np.array([0, 0, 0, 0, 0, 25, 50, 75])
# These values are from 1996
# http://jupiter.chem.uoa.gr/thanost/papers/papers4/Metrol_30(1993)155.pdf
elden = np.array(
    [21459.0, 26826.2, 32193.8, 27776.1, 25938.5, 26804.6, 26783.4, 26761.9]
)


def test_edlen_refractive_index():
    for t, p, h, e in zip(temperature, pressure, relative_humidity, elden):
        nm1e8 = (
            util.edlen_refraction(
                6330.0,
                t,
                p,
                util.get_vapour_partial_pressure(
                    h, util.get_vapour_pressure(t)
                ),
            )
            - 1
        ) * 1e8
        # We accept large errors because the set of coefficients for the
        # Edlen equations changes all the time and we onlyl need 2 S.F.
        # accuracy
        assert np.isclose(
            nm1e8, e, rtol=0.1, atol=1000
        ), "{} istaed of {}.".format(nm1e8, e)


def test_vacuum_to_air_wavelength():
    # https://classic.sdss.org/dr7/products/spectra/vacwavelength.html
    #  line      air    vacuum
    # H-beta  4861.363 4862.721
    # [O III] 4958.911 4960.295
    # [O III] 5006.843 5008.239
    # [N II]  6548.05  6549.86
    # H-alpha 6562.801 6564.614
    # [N II]  6583.45  6585.27
    # [S II]  6716.44  6718.29
    # [S II]  6730.82  6732.68
    wave_vacuum = np.array(
        [
            4862.721,
            4960.295,
            5008.239,
            6549.86,
            6564.614,
            6585.27,
            6718.29,
            6732.68,
        ]
    )
    wave_air = np.array(
        [
            4861.363,
            4958.911,
            5006.843,
            6548.05,
            6562.80,
            6583.45,
            6716.44,
            6730.82,
        ]
    )
    assert np.isclose(
        wave_air,
        util.vacuum_to_air_wavelength(
            wave_vacuum, temperature=288.15, pressure=101325
        ),
        atol=0.1,
        rtol=0.01,
    ).all()


def test_load_calibration_lines():
    assert (
        len(util.load_calibration_lines(elements=["He"], min_intensity=5)[0])
        == 28
    )
    assert (
        len(util.load_calibration_lines(elements=["He"], min_intensity=0)[0])
        == 49
    )
    assert (
        len(util.load_calibration_lines(elements=["He"], min_distance=0)[0])
        == 28
    )
    assert (
        len(
            util.load_calibration_lines(
                elements=["He"], min_intensity=0, min_distance=0
            )[0]
        )
        == 49
    )


def test_load_calibration_lines_top_10_only():
    assert (
        len(
            util.load_calibration_lines(
                elements=["He"], min_intensity=10, brightest_n_lines=10
            )[0]
        )
        <= 10
    )
    assert (
        len(
            util.load_calibration_lines(
                elements=["He"], min_intensity=0, brightest_n_lines=10
            )[0]
        )
        <= 10
    )
    assert (
        len(
            util.load_calibration_lines(
                elements=["He"], min_distance=0, brightest_n_lines=10
            )[0]
        )
        <= 10
    )


def test_load_calibration_lines_vacuum_vs_air():
    wave_air = util.load_calibration_lines(elements=["He"], min_intensity=10)[
        1
    ]
    wave_vacuum = util.load_calibration_lines(
        elements=["He"], min_intensity=10, vacuum=True
    )[1]
    assert (np.array(wave_air) < np.array(wave_vacuum)).all()


def test_load_calibration_lines_from_file():
    lines_manual = util.load_calibration_lines(
        elements=["He"],
        linelist=pkg_resources.resource_filename(
            "rascal", "arc_lines/nist_clean.csv"
        ),
    )
    lines = util.load_calibration_lines(elements=["He"])
    assert (lines[1] == lines_manual[1]).all()


@pytest.mark.xfail()
def test_load_calibration_lines_from_unknown_file():
    lines_manual = util.load_calibration_lines(
        elements=["He"],
        linelist="blabla",
    )
    lines = util.load_calibration_lines(elements=["He"])
    assert (lines[1] == lines_manual[1]).all()


@pytest.mark.xfail()
def test_load_calibration_lines_from_unknown_type():
    lines_manual = util.load_calibration_lines(
        elements=["He"],
        linelist=np.ones(10),
    )
    lines = util.load_calibration_lines(elements=["He"])
    assert (lines[1] == lines_manual[1]).all()


def test_print_calibration_lines(capfd):
    util.print_calibration_lines(elements=["He"])
    out, err = capfd.readouterr()
    assert type(out) == str


def test_derivative():
    assert util._derivative([2, 3, 4, 5]) == [3, 8, 15]


def test_filter_multiple_element_linelist_number_min_intensity():
    util.load_calibration_lines(
        elements=["He", "Xe", "Cu", "Ar"], min_intensity=0
    )


def test_filter_multiple_element_linelist_list_min_intensity():
    util.load_calibration_lines(
        elements=["He", "Xe", "Cu", "Ar"],
        min_intensity=[10.0, 50.0, 100.0, 500.0],
    )


@pytest.mark.xfail()
def test_filter_multiple_element_linelist_list_min_intensity_expect_fail():
    util.load_calibration_lines(
        elements=["He", "Xe", "Cu", "Ar"], min_intensity=[10.0, 50.0, 100.0]
    )


def test_filter_multiple_element_linelist_wrong_min_intensity_dtype():
    util.load_calibration_lines(
        elements=["He", "Xe", "Cu", "Ar"], min_intensity="[10.0, 50.0, 100.0]"
    )


@patch("matplotlib.pyplot.show")
def test_plot_calibration_lines(mock_show):
    util.plot_calibration_lines(elements=["He"])


@patch("matplotlib.pyplot.show")
def test_display_plot_calibration_lines_1_element(mock_show):

    util.plot_calibration_lines(
        elements=["He"],
        min_atlas_wavelength=2900,
        max_atlas_wavelength=4500,
        pixel_scale=0.25,
        min_intensity=100.0,
        label=True,
        display=True,
        save_fig=False,
    )


def test_save_plot_calibration_lines_1_element():

    util.plot_calibration_lines(
        elements=["He"],
        min_atlas_wavelength=2900,
        max_atlas_wavelength=4500,
        pixel_scale=0.25,
        min_intensity=100.0,
        label=True,
        display=False,
        save_fig=True,
        filename=os.path.join(
            base_dir, "test_output", "example_CuNeAr_calibration_lines"
        ),
        fig_kwarg={"figsize": (30, 8)},
    )


def test_save_plot_calibration_lines_3_elements():

    util.plot_calibration_lines(
        elements=["Cu", "Ne", "Ar"],
        min_atlas_wavelength=2900,
        max_atlas_wavelength=4500,
        pixel_scale=0.25,
        min_intensity=100.0,
        label=False,
        display=False,
        save_fig=True,
        filename=os.path.join(
            base_dir, "test_output", "example_CuNeAr_calibration_lines"
        ),
        fig_kwarg={"figsize": (30, 8)},
    )
