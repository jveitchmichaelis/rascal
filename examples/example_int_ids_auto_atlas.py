import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import os
from scipy.signal import find_peaks

from rascal.calibrator import Calibrator
from rascal.atlas import Atlas
from rascal.util import refine_peaks

# Load the INT/IDS data
base_dir = os.path.dirname(os.path.abspath(__file__))

spectrum2D = fits.open(
    os.path.join(base_dir, "data_int_ids", "int20180101_01355922.fits.fz")
)

data = spectrum2D[1].data

plt.figure(1, figsize=(10, 4))
plt.imshow(np.log10(np.array(data.data).T), aspect="auto", origin="lower")
plt.xlabel("Spectral Direction / Pix")
plt.ylabel("Spatial Direction / Pix")
plt.tight_layout()

os.makedirs(os.path.join(base_dir, "output"), exist_ok=True)
plt.savefig(
    os.path.join(base_dir, "output", "gtc-osiris-manual-atlas-arc-image.png")
)


# Collapse into 1D spectrum between row 110 and 120
spectrum = np.flip(data.mean(1), 0)

# Identify the peaks
peaks, _meta = find_peaks(spectrum, prominence=10, distance=5, threshold=None)
prominences = _meta["prominences"]

# Get the top 50 lines
peaks = peaks[np.argsort(prominences)[::-1][:100]]
peaks = refine_peaks(spectrum, peaks, window_width=5)

# Initialise the calibrator
c = Calibrator(peaks, spectrum=spectrum)
c.plot_arc(
    save_fig="png",
    display=False,
    filename=os.path.join(
        base_dir, "output", "int-ids-auto-atlas-arc-spectrum"
    ),
)
c.set_hough_properties(
    num_slopes=2000,
    range_tolerance=300.0,
    xbins=200,
    ybins=200,
    min_wavelength=3000.0,
    max_wavelength=4600.0,
)
c.set_ransac_properties(sample_size=5, top_n_candidate=5)

atlas = Atlas(
    elements=["Cu", "Ne", "Ar"],
    min_intensity=20,
    temperature=spectrum2D[0].header["TEMPTUBE"] + 273.0,
    range_tolerance=300,
    min_atlas_wavelength=3000.0,
    max_atlas_wavelength=4600.0,
)
c.set_atlas(atlas)

c.do_hough_transform()

# Run the wavelength calibration
(
    best_p,
    matched_peaks,
    matched_atlas,
    rms,
    residual,
    peak_utilisation,
    atlas_utilisation,
) = c.fit(max_tries=200, fit_deg=4)

# Plot the solution
c.plot_fit(
    best_p,
    spectrum,
    plot_atlas=True,
    log_spectrum=False,
    display=False,
    save_fig="png",
    filename=os.path.join(
        base_dir, "output", "int-ids-auto-atlas-wavelength-calibration"
    ),
)

# Show the parameter space for searching possible solution
c.plot_search_space(
    save_fig="png",
    filename=os.path.join(
        base_dir, "output", "int-ids-auto-atlas-search-space"
    ),
)

print("Stdev error: {} A".format(residual.std()))
print("Peaks utilisation rate: {}%".format(peak_utilisation * 100))
print("Atlas utilisation rate: {}%".format(atlas_utilisation * 100))
