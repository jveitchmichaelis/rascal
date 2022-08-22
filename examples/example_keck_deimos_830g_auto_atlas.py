import json

import numpy as np
from scipy.signal import find_peaks

from rascal.calibrator import Calibrator
from rascal.atlas import Atlas
from rascal.util import refine_peaks

# Load the 1D Spectrum from Pypeit
spectrum_json = json.load(
    open("data_keck_deimos/keck_deimos_830g_l_PYPIT.json")
)
spectrum = np.array(spectrum_json["spec"])

# Identify the arc lines
peaks, _ = find_peaks(spectrum, prominence=200, distance=10)
peaks = refine_peaks(spectrum, peaks, window_width=3)

c = Calibrator(peaks, spectrum=spectrum)
c.plot_arc(
    display=False,
    save_fig="png",
    filename="output/keck-deimos-auto-atlas-arc-spectrum",
)
c.set_hough_properties(
    num_slopes=5000.0,
    range_tolerance=500.0,
    linearity_tolerance=50,
    xbins=500,
    ybins=500,
    min_wavelength=6500.0,
    max_wavelength=10500.0,
)
c.set_ransac_properties(
    sample_size=5,
    top_n_candidate=5,
    linear=True,
    filter_close=True,
    ransac_tolerance=5,
    candidate_weighted=True,
    hough_weight=1.0,
    minimum_matches=10,
)

atlas = Atlas(
    elements=["Ne", "Ar", "Kr"],
    min_intensity=1000.0,
    pressure=70000.0,
    temperature=285.0,
    range_tolerance=500.0,
    min_atlas_wavelength=6500.0,
    max_atlas_wavelength=10500.0,
)
c.set_atlas(atlas)

c.do_hough_transform()

# Run the wavelength calibration
(
    fit_coeff,
    matched_peaks,
    matched_atlas,
    rms,
    residual,
    peak_utilisation,
    atlas_utilisation,
) = c.fit(max_tries=1000, candidate_tolerance=5)

# Plot the solution
c.plot_fit(
    fit_coeff,
    spectrum,
    display=False,
    plot_atlas=True,
    log_spectrum=False,
    tolerance=5.0,
    save_fig="png",
    filename="output/keck-deimos-auto-atlas-wavelength-calibration",
)

# Show the parameter space for searching possible solution
print("RMS: {}".format(rms))
print("Stdev error: {} A".format(np.abs(residual).std()))
print("Peaks utilisation rate: {}%".format(peak_utilisation * 100))
print("Atlas utilisation rate: {}%".format(atlas_utilisation * 100))

c.plot_search_space(
    save_fig="png", filename="output/keck-deimos-auto-atlas-search-space"
)
