import os

import numpy as np
from astropy.io import fits
from scipy.signal import find_peaks

from rascal import util
from rascal.atlas import AtlasCollection, NistSource
from rascal.calibrator import Calibrator

# Load the Asiago FeAr data
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)

fits_file = fits.open(os.path.join(base_dir, "IMA101009.fits"))[0]

spectrum2D = fits_file.data

temperature = fits_file.header["TEMP-INT"]
pressure = fits_file.header["ATMPRESS"] * 100.0
relative_humidity = fits_file.header["HUM-INT"]

# Collapse into 1D spectrum between row 100 and 200
spectrum = np.median(spectrum2D[100:200], axis=0)

# Identify the peaks
peaks, _ = find_peaks(spectrum, height=300, prominence=5, distance=5)
peaks = util.refine_peaks(spectrum, peaks, window_width=5)

atlas = AtlasCollection(
    atlases=[
        {
            "element": "Fe I",
            "min_wavelength": 5900.0,
            "max_wavelength": 7100.0,
            "min_intensity": 20.0,
            "nist_source": NistSource.NIST_ALL,
            "pressure": pressure,
            "temperature": temperature,
            "relative_humidity": relative_humidity,
        },
        {
            "element": "Fe II",
            "min_wavelength": 5900.0,
            "max_wavelength": 7100.0,
            "min_intensity": 20.0,
            "nist_source": NistSource.NIST_ALL,
            "pressure": pressure,
            "temperature": temperature,
            "relative_humidity": relative_humidity,
        },
    ]
)

config = {
    "detector": {
        "detector_min_wave": 5800.0,
        "detector_max_wave": 7010.0,
    },
    "hough": {
        "num_slopes": 5000,
        "range_tolerance": 200.0,
        "xbins": 200,
        "ybins": 200,
    },
    "ransac": {
        "sample_size": 5,
        "top_n_candidate": 5,
        "filter_close": True,
        "max_tries": 5000,
    },
}

# Initialise the calibrator
c = Calibrator(
    peaks, atlas_lines=atlas.atlas_lines, config=config, spectrum=spectrum
)
c.plot_arc(
    display=False,
    save_fig=True,
    filename=os.path.join(output_dir, "asiago-fear-arc-spectrum"),
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
        filename=os.path.join(output_dir, "asiago-fear-wavelength-calibration"),
    )
    print("RMS: {}".format(res["rms"]))
    print("Stdev error: {} A".format(res["residual"].std()))
    print("Peaks utilisation rate: {}%".format(res["peak_utilisation"] * 100))
    print("Atlas utilisation rate: {}%".format(res["atlas_utilisation"] * 100))
else:
    print("Calibration failed to find a valid solution.")
