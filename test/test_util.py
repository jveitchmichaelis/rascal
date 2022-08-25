import pkg_resources
import pytest
from unittest.mock import patch

import numpy as np

from rascal import util


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


def test_get_calibration_lines():
    assert (
        len(util.get_calibration_lines(elements=["He"], min_intensity=5)[0])
        == 12
    )
    assert (
        len(util.get_calibration_lines(elements=["He"], min_intensity=0)[0])
        == 21
    )
    assert (
        len(util.get_calibration_lines(elements=["He"], min_distance=0)[0])
        == 62
    )
    assert (
        len(
            util.get_calibration_lines(
                elements=["He"], min_intensity=0, min_distance=0
            )[0]
        )
        == 112
    )


def test_get_calibration_lines_top_10_only():
    assert (
        len(
            util.get_calibration_lines(
                elements=["He"], min_intensity=10, brightest_n_lines=10
            )[0]
        )
        <= 10
    )
    assert (
        len(
            util.get_calibration_lines(
                elements=["He"], min_intensity=0, brightest_n_lines=10
            )[0]
        )
        <= 10
    )
    assert (
        len(
            util.get_calibration_lines(
                elements=["He"], min_distance=0, brightest_n_lines=10
            )[0]
        )
        <= 10
    )


def test_get_calibration_lines_vacuum_vs_air():
    wave_air = util.get_calibration_lines(elements=["He"], min_intensity=10)[1]
    wave_vacuum = util.get_calibration_lines(
        elements=["He"], min_intensity=10, vacuum=True
    )[1]
    assert (np.array(wave_air) < np.array(wave_vacuum)).all()


def test_get_calibration_lines_from_file():
    lines_manual = util.get_calibration_lines(
        elements=["He"],
        linelist=pkg_resources.resource_filename(
            "rascal", "arc_lines/nist_clean.csv"
        ),
    )
    lines = util.get_calibration_lines(elements=["He"])
    assert (lines[1] == lines_manual[1]).all()


@pytest.mark.xfail()
def test_get_calibration_lines_from_unknown_file():
    lines_manual = util.get_calibration_lines(
        elements=["He"],
        linelist="blabla",
    )
    lines = util.get_calibration_lines(elements=["He"])
    assert (lines[1] == lines_manual[1]).all()


@pytest.mark.xfail()
def test_get_calibration_lines_from_unknown_type():
    lines_manual = util.get_calibration_lines(
        elements=["He"],
        linelist=np.ones(10),
    )
    lines = util.get_calibration_lines(elements=["He"])
    assert (lines[1] == lines_manual[1]).all()


def test_print_calibration_lines(capfd):
    util.print_calibration_lines(elements=["He"])
    out, err = capfd.readouterr()
    assert type(out) == str


@patch("matplotlib.pyplot.show")
def test_plot_calibration_lines(mock_show):
    util.plot_calibration_lines(elements=["He"])


def test_derivative():
    assert util._derivative([2, 3, 4, 5]) == [3, 8, 15]
