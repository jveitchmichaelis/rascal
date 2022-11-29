import os
from unittest.mock import patch

from rascal import plotting

base_dir = os.path.dirname(os.path.abspath(__file__))


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


def test_save_plot_calibration_lines_1_element():

    plotting.plot_calibration_lines(
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

    plotting.plot_calibration_lines(
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
