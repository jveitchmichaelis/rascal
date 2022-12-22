import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import os
from scipy.signal import find_peaks

from rascal.calibrator import Calibrator
from rascal.atlas import Atlas
from rascal.util import refine_peaks


base_dir = os.path.dirname(os.path.abspath(__file__))

data = fits.open(
    os.path.join(
        base_dir,
        "data_gtc_osiris",
        "0002672523-20200911-OSIRIS-OsirisCalibrationLamp.fits",
    )
)[1]

plt.figure(1, figsize=(10, 4))
plt.imshow(np.log10(data.data.T), aspect="auto", origin="lower")
plt.xlabel("Spectral Direction / Pix")
plt.ylabel("Spatial Direction / Pix")
plt.tight_layout()

os.makedirs(os.path.join(base_dir, "output"), exist_ok=True)
plt.savefig(
    os.path.join(base_dir, "output", "gtc-osiris-auto-atlas-arc-image.png")
)

spectrum = np.median(data.data.T[550:570], axis=0)

# Identify the arc lines
peaks, _ = find_peaks(
    spectrum, height=1250, prominence=20, distance=3, threshold=None
)
peaks_refined = refine_peaks(spectrum, peaks, window_width=3)

c = Calibrator(peaks_refined, spectrum=spectrum)

c.set_hough_properties(
    num_slopes=2000,
    xbins=100,
    ybins=100,
    min_wavelength=3600.0,
    max_wavelength=7900.0,
    range_tolerance=500.0,
    linearity_tolerance=100,
)

atlas = Atlas(
    elements=["Hg", "Ar"],
    min_intensity=100,
    range_tolerance=200.0,
    min_atlas_wavelength=3600.0,
    max_atlas_wavelength=7900.0,
)
c.set_atlas(atlas)


c.set_ransac_properties(
    sample_size=5,
    top_n_candidate=6,
    linear=True,
    filter_close=True,
    ransac_tolerance=10,
    candidate_weighted=True,
    hough_weight=1.0,
    minimum_matches=10,
)

c.do_hough_transform()

c.plot_arc(
    save_fig="png",
    display=False,
    filename=os.path.join(
        base_dir, "output", "gtc-osiris-auto-atlas-arc-spectrum"
    ),
)

c.do_hough_transform()

# Run the wavelength calibration
res = c.fit(
    max_tries=5000, fit_tolerance=10.0, candidate_tolerance=5.0, fit_deg=4
)

# Plot the solution
c.plot_fit(
    res["fit_coeff"],
    plot_atlas=True,
    log_spectrum=False,
    display=False,
    save_fig="png",
    filename=os.path.join(
        base_dir, "output", "gtc-osiris-auto-atlas-wavelength-calibration"
    ),
)

# Show the parameter space for searching possible solution
print("RMS: {}".format(res["rms"]))
print("Stdev error: {} A".format(res["residual"].std()))
print("Peaks utilisation rate: {}%".format(res["peak_utilisation"] * 100))
print("Atlas utilisation rate: {}%".format(res["atlas_utilisation"] * 100))

c.plot_search_space(
    save_fig="png",
    filename=os.path.join(
        base_dir, "output", "gtc-osiris-auto-atlas-search-space"
    ),
)
