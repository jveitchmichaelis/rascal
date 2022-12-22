import os

import numpy as np
from astropy.io import fits
from scipy.signal import find_peaks

from rascal.atlas import Atlas
from rascal.calibrator import Calibrator
from rascal.util import refine_peaks

base_dir = os.path.dirname(os.path.abspath(__file__))

# Load the 1D Spectrum from Pypeit
data = fits.open(
    os.path.join(base_dir, "data_eso36_efosc", "EFOSC_spec_HeAr227_0005.fits")
)[0]
spectrum = np.median(data.data.T, axis=0)

# Identify the arc lines
peaks, _ = find_peaks(spectrum, prominence=200, distance=10, threshold=None)
peaks_refined = refine_peaks(spectrum, peaks, window_width=3)

c = Calibrator(peaks_refined, spectrum)
c.set_calibrator_properties(
    num_pix=len(spectrum), plotting_library="matplotlib", log_level="info"
)

c.plot_arc(
    save_fig="png",
    display=False,
    filename=os.path.join(
        base_dir, "output", "ntt-efosc-auto-atlas-arc-spectrum"
    ),
)

c.set_known_pairs(998.53393246, 7053.9)
c.set_hough_properties(
    num_slopes=2000,
    range_tolerance=500.0,
    xbins=100,
    ybins=100,
    min_wavelength=3500.0,
    max_wavelength=7500.0,
)
c.set_ransac_properties(
    sample_size=5, top_n_candidate=5, ransac_tolerance=10, minimum_matches=10
)

atlas = Atlas(
    ["He"],
    range_tolerance=100,
    min_atlas_wavelength=3500.0,
    max_atlas_wavelength=7500.0,
)
atlas.add(["Ar"], min_intensity=100)

c.set_atlas(atlas)

c.do_hough_transform()

# Run the wavelength calibration
res = c.fit(max_tries=1000, fit_deg=3, candidate_tolerance=5)

c.plot_fit(
    res["fit_coeffs"],
    plot_atlas=True,
    log_spectrum=False,
    display=False,
    save_fig="png",
    filename=os.path.join(
        base_dir, "output", "ntt-efosc-auto-atlas-wavelength-calibration"
    ),
)

c.plot_search_space(
    save_fig="png",
    filename=os.path.join(
        base_dir, "output", "ntt-efosc-auto-atlas-search-space"
    ),
)

print("RMS: {}".format(res["rms"]))
print("Stdev error: {} A".format(res["residual"].std()))
print("Peaks utilisation rate: {}%".format(res["peak_utilisation"] * 100))
print("Atlas utilisation rate: {}%".format(res["atlas_utilisation"] * 100))
