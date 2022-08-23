from unittest.mock import patch

import numpy as np
import pytest

from rascal.util import *
from rascal.util import _derivative


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
            edlen_refraction(
                6330.0,
                t,
                p,
                get_vapour_partial_pressure(h, get_vapour_pressure(t)),
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
        vacuum_to_air_wavelength(
            wave_vacuum, temperature=288.15, pressure=101325
        ),
        atol=0.1,
        rtol=0.01,
    ).all()


def test_get_calibration_lines():
    assert (
        len(get_calibration_lines(elements=["He"], min_intensity=10)[0]) == 20
    )
    assert (
        len(get_calibration_lines(elements=["He"], min_intensity=0)[0]) == 33
    )
    assert len(get_calibration_lines(elements=["He"], min_distance=0)[0]) == 25


def test_get_calibration_lines_vacuum_vs_air():
    wave_air = get_calibration_lines(elements=["He"], min_intensity=10)[1]
    wave_vacuum = get_calibration_lines(
        elements=["He"], min_intensity=10, vacuum=True
    )[1]
    assert (np.array(wave_air) < np.array(wave_vacuum)).all()


def test_print_calibration_lines(capfd):
    print_calibration_lines(elements=["He"])
    out, err = capfd.readouterr()
    assert type(out) == str


@patch("matplotlib.pyplot.show")
def test_plot_calibration_lines(mock_show):
    plot_calibration_lines(elements=["He"])


def test_derivative():
    assert _derivative([2, 3, 4, 5]) == [3, 8, 15]
