import os

import numpy as np
from astropy.io import fits
from rascal import util
from rascal.atlas import Atlas
from rascal.calibrator import Calibrator
from scipy.signal import find_peaks

__file__ = __name__

# Load the LT SPRAT data
base_dir = os.path.dirname(__file__)
fits_file = fits.open(os.path.join(base_dir, "IMA101009.fits"))[0]

spectrum2D = fits_file.data

temperature = fits_file.header["TEMP-INT"]
pressure = fits_file.header["ATMPRESS"] * 100.0
relative_humidity = fits_file.header["HUM-INT"]

# Collapse into 1D spectrum between row 110 and 120
spectrum = np.median(spectrum2D[100:200], axis=0)

# Identify the peaks
peaks, _ = find_peaks(spectrum, height=300, prominence=5, distance=5)
peaks = util.refine_peaks(spectrum, peaks, window_width=5)

# Initialise the calibrator
c = Calibrator(peaks, spectrum=spectrum)
c.plot_arc()
c.set_hough_properties(
    num_slopes=5000,
    range_tolerance=200.0,
    xbins=200,
    ybins=200,
    min_wavelength=5800.0,
    max_wavelength=7010.0,
)
c.set_ransac_properties(sample_size=5, top_n_candidate=5, filter_close=True)

fe1 = np.array(
    list(
        csv.reader(
            open(
                r"C:\Users\cylam\git\rascal\src\rascal\arc_lines\nist_clean_Fe_I.csv"
            ),
            delimiter=",",
            quotechar='"',
            skipinitialspace=True,
        )
    )[1:]
)
fe2 = np.array(
    list(
        csv.reader(
            open(
                r"C:\Users\cylam\git\rascal\src\rascal\arc_lines\nist_clean_Fe_II.csv"
            ),
            delimiter=",",
            quotechar='"',
            skipinitialspace=True,
        )
    )[1:]
)


fe1 = fe1[fe1[:, 2].astype("float") > 20.0]
fe1 = fe1[fe1[:, 1].astype("float") > 5900]
fe1 = fe1[fe1[:, 1].astype("float") < 7100]

fe2 = fe2[fe2[:, 2].astype("float") > 20.0]
fe2 = fe2[fe2[:, 1].astype("float") > 5900]
fe2 = fe2[fe2[:, 1].astype("float") < 7100]


atlas = Atlas(
    pressure=pressure,
    temperature=temperature,
    relative_humidity=relative_humidity,
)
atlas.add_user_atlas(
    fe1[:, 0],
    fe1[:, 1].astype("float"),
    pressure=pressure,
    temperature=temperature,
    relative_humidity=relative_humidity,
)
atlas.add_user_atlas(
    fe2[:, 0],
    fe2[:, 1].astype("float"),
    pressure=pressure,
    temperature=temperature,
    relative_humidity=relative_humidity,
)
c.set_atlas(atlas, candidate_tolerance=5.0)

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
) = c.fit(max_tries=1000000)

# Plot the solution
c.plot_fit(
    best_p, spectrum, plot_atlas=True, log_spectrum=False, tolerance=5.0
)
