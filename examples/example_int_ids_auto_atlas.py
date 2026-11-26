import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.signal import find_peaks

from eval_utils import report_calibration_diagnostics
from rascal.atlas import AtlasCollection
from rascal.calibrator import Calibrator
from rascal.util import refine_peaks

# Load the INT/IDS data
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)

spectrum2D = fits.open(
    os.path.join(base_dir, "data_int_ids", "int20180101_01355922.fits.fz")
)

data = spectrum2D[1].data

plt.figure(1, figsize=(10, 4))
plt.imshow(np.log10(np.array(data.data).T), aspect="auto", origin="lower")
plt.xlabel("Spectral Direction / Pix")
plt.ylabel("Spatial Direction / Pix")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "int-ids-auto-atlas-arc-image.png"))
plt.close()

# Collapse into 1D spectrum
spectrum = np.flip(data.mean(1), 0)

# Identify the peaks
peaks, _meta = find_peaks(spectrum, prominence=10, distance=5, threshold=None)
prominences = _meta["prominences"]

# Get the top 100 lines
peaks = peaks[np.argsort(prominences)[::-1][:100]]
peaks = refine_peaks(spectrum, peaks, window_width=5)

atlas = AtlasCollection(
    atlases=[
        {
            "element": el,
            "min_intensity": 20.0,
            "temperature": spectrum2D[0].header["TEMPTUBE"] + 273.0,
            "range_tolerance": 300.0,
            "min_wavelength": 3000.0,
            "max_wavelength": 4600.0,
        }
        for el in ["Cu", "Ne", "Ar"]
    ]
)

config = {
    "detector": {
        "detector_min_wave": 3000.0,
        "detector_max_wave": 4600.0,
    },
    "hough": {
        "num_slopes": 2000,
        "range_tolerance": 300.0,
        "xbins": 200,
        "ybins": 200,
    },
    "ransac": {
        "sample_size": 5,
        "top_n_candidate": 5,
        "max_tries": 200,
        "degree": 4,
    },
}

# Initialise the calibrator
c = Calibrator(
    peaks, atlas_lines=atlas.atlas_lines, config=config, spectrum=spectrum
)
c.plot_arc(
    save_fig=True,
    display=False,
    filename=os.path.join(output_dir, "int-ids-auto-atlas-arc-spectrum"),
)

c.do_hough_transform()

# Run the wavelength calibration
res = c.fit()

if res["success"]:
    # Plot the solution
    c.plot_fit(
        res["fit_coeff"],
        spectrum=spectrum,
        plot_atlas=True,
        log_spectrum=False,
        display=False,
        save_fig=True,
        filename=os.path.join(
            output_dir, "int-ids-auto-atlas-wavelength-calibration"
        ),
    )

    # Show the parameter space for searching possible solution
    c.plot_search_space(
        display=False,
        save_fig=True,
        filename=os.path.join(output_dir, "int-ids-auto-atlas-search-space"),
    )

    print("RMS: {}".format(res["rms"]))
    print("Stdev error: {} A".format(res["residual"].std()))
    print("Peaks utilisation rate: {}%".format(res["peak_utilisation"] * 100))
    print("Atlas utilisation rate: {}%".format(res["atlas_utilisation"] * 100))
else:
    print("Calibration failed to find a valid solution.")

report_calibration_diagnostics(c, res)
