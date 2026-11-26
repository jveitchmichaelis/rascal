import importlib.resources
import os
from functools import partialmethod
from unittest.mock import patch

import numpy as np
import pytest

# Suppress tqdm output
from tqdm import tqdm

from rascal import plotting, util

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
        # Edlen equations changes all the time and we only need 2 S.F.
        # accuracy
        assert np.isclose(
            nm1e8, e, rtol=0.1, atol=1000
        ), "{} instead of {}.".format(nm1e8, e)


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


def test_air_to_vacuum_roundtrip():
    wave_vacuum = np.array([4000.0, 5500.0, 7000.0, 9000.0])
    wave_air = util.vacuum_to_air_wavelength(
        wave_vacuum, temperature=288.15, pressure=101325
    )
    recovered = util.air_to_vacuum_wavelength(
        wave_air, temperature=288.15, pressure=101325
    )
    assert np.allclose(recovered, wave_vacuum, atol=1e-3)


def test_load_calibration_lines():
    lines_strong = util.load_calibration_lines(
        linelist="nist_strong", element="He I", min_intensity=5
    )
    assert len(lines_strong) == 38
    assert all(
        {"element", "wavelength", "intensity"}.issubset(line.keys())
        for line in lines_strong
    )

    lines_all = util.load_calibration_lines(
        linelist="nist", element="He I", min_intensity=5
    )
    assert len(lines_all) == 112

    lines_dist = util.load_calibration_lines(
        linelist="nist_strong", element="He I", min_distance=50.0
    )
    assert 0 < len(lines_dist) < len(lines_strong)


def test_load_calibration_lines_top_10_only():
    lines = util.load_calibration_lines(
        linelist="nist_strong",
        element="He I",
        min_intensity=10,
        brightest_n_lines=10,
    )
    assert len(lines) <= 10

    lines_zero = util.load_calibration_lines(
        linelist="nist_strong",
        element="He I",
        min_intensity=0,
        brightest_n_lines=10,
    )
    assert len(lines_zero) <= 10


def test_load_calibration_lines_vacuum_vs_air():
    wave_air = [
        line["wavelength"]
        for line in util.load_calibration_lines(
            linelist="nist_strong", element="He I", min_intensity=10
        )
    ]
    wave_vacuum = [
        line["wavelength"]
        for line in util.load_calibration_lines(
            linelist="nist_strong",
            element="He I",
            min_intensity=10,
            vacuum=True,
        )
    ]
    assert (np.array(wave_air) < np.array(wave_vacuum)).all()


def test_load_calibration_lines_from_file():
    lines_manual = util.load_calibration_lines(
        element="He I",
        linelist=str(
            importlib.resources.files("rascal")
            / "arc_lines/nist_clean_He_I.csv"
        ),
    )
    lines = util.load_calibration_lines(element="He I")
    assert [l["wavelength"] for l in lines] == [
        l["wavelength"] for l in lines_manual
    ]


def test_load_calibration_lines_from_unknown_file():
    with pytest.raises(ValueError):
        util.load_calibration_lines(
            element="He I",
            linelist="blabla",
        )


def test_load_calibration_lines_from_unknown_type():
    with pytest.raises(ValueError):
        util.load_calibration_lines(
            element="He I",
            linelist=np.ones(10),
        )


def test_print_calibration_lines(capfd):
    util.print_calibration_lines(elements=["He"])
    out, err = capfd.readouterr()
    assert isinstance(out, str)


def test_derivative():
    assert util._derivative([2, 3, 4, 5]) == [3, 8, 15]


@patch("matplotlib.pyplot.show")
def test_plot_calibration_lines(mock_show):
    plotting.plot_calibration_lines(elements=["He"])


@patch("matplotlib.pyplot.show")
def test_display_plot_calibration_lines_1_element(mock_show):
    plotting.plot_calibration_lines(
        elements=["He"],
        min_atlas_wavelength=2900,
        max_atlas_wavelength=4500,
        pixel_scale=0.25,
        min_intensity=100.0,
        label=True,
        display=True,
        save_fig=False,
    )


def test_save_plot_calibration_lines_1_element(tmp_path):
    plotting.plot_calibration_lines(
        elements=["He"],
        min_atlas_wavelength=2900,
        max_atlas_wavelength=4500,
        pixel_scale=0.25,
        min_intensity=100.0,
        label=True,
        display=False,
        save_fig=True,
        filename=str(tmp_path / "example_CuNeAr_calibration_lines"),
        fig_kwarg={"figsize": (30, 8)},
    )


def test_save_plot_calibration_lines_3_elements(tmp_path):
    plotting.plot_calibration_lines(
        elements=["Cu", "Ne", "Ar"],
        min_atlas_wavelength=2900,
        max_atlas_wavelength=4500,
        pixel_scale=0.25,
        min_intensity=100.0,
        label=False,
        display=False,
        save_fig=True,
        filename=str(tmp_path / "example_CuNeAr_calibration_lines"),
        fig_kwarg={"figsize": (30, 8)},
    )
