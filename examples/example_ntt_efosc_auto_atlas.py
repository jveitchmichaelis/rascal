import os

import numpy as np
from astropy.io import fits
from scipy.signal import find_peaks

from eval_utils import report_calibration_diagnostics
from rascal.atlas import AtlasCollection
from rascal.calibrator import Calibrator
from rascal.util import refine_peaks

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# Load the 1D Spectrum
data = fits.open(
    os.path.join(base_dir, "data_eso36_efosc", "EFOSC_spec_HeAr227_0005.fits")
)[0]
spectrum = np.median(data.data.T, axis=0)

# Identify the arc lines
peaks, _ = find_peaks(spectrum, prominence=200, distance=10, threshold=None)
peaks_refined = refine_peaks(spectrum, peaks, window_width=3)

atlas = AtlasCollection(
    atlases=[
        {
            "element": "He",
            "range_tolerance": 100.0,
            "min_wavelength": 3500.0,
            "max_wavelength": 7500.0,
        },
        {
            "element": "Ar",
            "min_intensity": 100.0,
            "range_tolerance": 100.0,
            "min_wavelength": 3500.0,
            "max_wavelength": 7500.0,
        },
    ]
)

config = {
    "plotting_library": "matplotlib",
    "log_level": "info",
    "detector": {
        "num_pix": len(spectrum),
        "detector_min_wave": 3500.0,
        "detector_max_wave": 7500.0,
    },
    "hough": {
        "num_slopes": 2000,
        "range_tolerance": 500.0,
        "xbins": 100,
        "ybins": 100,
    },
    "ransac": {
        "sample_size": 5,
        "top_n_candidate": 5,
        "rms_tolerance": 10.0,
        "inlier_tolerance": 5.0,
        "minimum_matches": 10,
        "max_tries": 1000,
        "degree": 3,
    },
}

c = Calibrator(
    peaks_refined, atlas_lines=atlas.atlas_lines, config=config, spectrum=spectrum
)

c.plot_arc(
    save_fig=True,
    display=False,
    filename=os.path.join(output_dir, "ntt-efosc-auto-atlas-arc-spectrum"),
)

c.set_known_pairs(998.53393246, 7053.9)

c.do_hough_transform()

# Run the wavelength calibration
res = c.fit()

if res["success"]:
    c.plot_fit(
        res["fit_coeff"],
        plot_atlas=True,
        log_spectrum=False,
        display=False,
        save_fig=True,
        filename=os.path.join(
            output_dir, "ntt-efosc-auto-atlas-wavelength-calibration"
        ),
    )

    c.plot_search_space(
        display=False,
        save_fig=True,
        filename=os.path.join(output_dir, "ntt-efosc-auto-atlas-search-space"),
    )

    print("RMS: {}".format(res["rms"]))
    print("Stdev error: {} A".format(res["residual"].std()))
    print("Peaks utilisation rate: {}%".format(res["peak_utilisation"] * 100))
    print("Atlas utilisation rate: {}%".format(res["atlas_utilisation"] * 100))
else:
    print("Calibration failed to find a valid solution.")

report_calibration_diagnostics(c, res)
