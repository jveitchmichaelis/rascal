import os

import numpy as np

from rascal import util

base_dir = os.path.dirname(os.path.abspath(__file__))

util.plot_calibration_lines(
    elements=["Cu", "Ne", "Ar"],
    min_atlas_wavelength=2900,
    max_atlas_wavelength=4500,
    pixel_scale=0.25,
    min_intensity=1000.0,
    label=True,
    display=False,
    save_fig=True,
    filename=os.path.join(
        base_dir, "output", "example_CuNeAr_calibration_lines"
    ),
    fig_kwarg={"figsize": (30, 8)},
)
