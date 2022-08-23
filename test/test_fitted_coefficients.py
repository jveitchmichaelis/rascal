import os

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy import interpolate

from rascal.calibrator import Calibrator
from rascal.atlas import Atlas
from rascal import util

base_dir = os.path.dirname(os.path.abspath(__file__))

gmos_fit_coeff = np.array(
    [
        4.66525885e03,
        1.46720775e00,
        1.63000305e-05,
        1.37474397e-10,
        -1.67109123e-13,
    ]
)
osiris_fit_coeff = np.array(
    [
        3.63546445e03,
        1.45549920e00,
        4.66498922e-04,
        -9.89023065e-08,
        9.00463547e-12,
    ]
)
sprat_fit_coeff = np.array(
    [
        3.48888535e03,
        3.80147651e00,
        1.74826021e-03,
        -1.37695946e-06,
        4.04726228e-10,
    ]
)

# All pixel values here are unbinned
# n for north
# s for south
hamamatsu_width = 2048

# chip gaps
hamamatsu_n_gap = 67
hamamatsu_s_gap = 61


def create_pixel_array(northsouth, binning):
    """
    The GMOS longslit spectrum spreads over three CCDs. This function creates
    the pixel list corresponding the binned pixels adjusted for the chip gaps,
    which can lead to a non-integer shift in the number of pixels.
    Parameters
    ----------
    northsout: str
        Indicate whether the Gemini north or south, because the chip gaps are
        different.
    binning : numeric
        The binning factor
    Returns
    -------
    pixels: numpy.array
        The pixel values adjusted for chip gaps for the corresponding pixel.
    """
    binned_width = hamamatsu_width / binning
    if northsouth == "north":
        gap_binned_width = hamamatsu_n_gap / binning
    elif northsouth == "south":
        gap_binned_width = hamamatsu_s_gap / binning
    else:
        raise ValueError('Please choose from "north" or "south".')
    pixels = np.concatenate(
        (
            np.arange(binned_width),
            np.arange(
                binned_width + gap_binned_width,
                binned_width * 2 + gap_binned_width,
            ),
            np.arange(
                binned_width * 2 + gap_binned_width * 2,
                binned_width * 3 + gap_binned_width * 2,
            ),
        )
    )
    return pixels


def test_gmos_fit():

    pixels = create_pixel_array("north", 2)
    rawpix_to_pix_itp = interpolate.interp1d(np.arange(len(pixels)), pixels)

    spectrum2D = fits.open(
        os.path.join(
            base_dir,
            "..",
            "examples",
            "data_gemini_gmos",
            "N20181115S0215_flattened.fits",
        )
    )[0].data

    # Collapse into 1D spectrum between row 300 and 310
    spectrum = np.median(spectrum2D[300:310], axis=0)[::-1]

    # Identify the peaks
    peaks, _ = find_peaks(
        spectrum, height=1000, prominence=500, distance=5, threshold=None
    )
    peaks = util.refine_peaks(spectrum, peaks, window_width=5)
    peaks_shifted = rawpix_to_pix_itp(peaks)

    # Initialise the calibrator
    c = Calibrator(peaks_shifted, spectrum=spectrum)
    c.set_calibrator_properties(pixel_list=pixels)
    c.set_hough_properties(
        num_slopes=5000,
        range_tolerance=500.0,
        xbins=200,
        ybins=200,
        min_wavelength=5000.0,
        max_wavelength=9500.0,
    )
    c.set_ransac_properties(sample_size=8, top_n_candidate=10)
    # Vacuum wavelengths
    # blend: 5143.21509, 5146.74143
    # something weird near there, so not used: 8008.359, 8016.990
    gmos_atlas_lines = [
        4703.632,
        4728.19041,
        4766.19677,
        4807.36348,
        4849.16386,
        4881.22627,
        4890.40721,
        4906.12088,
        4934.58593,
        4966.46490,
        5018.56194,
        5063.44827,
        5163.723,
        5189.191,
        5497.401,
        5560.246,
        5608.290,
        5913.723,
        6754.698,
        6873.185,
        6967.352,
        7032.190,
        7069.167,
        7149.012,
        7274.940,
        7386.014,
        7505.935,
        7516.721,
        7637.208,
        7725.887,
        7893.246,
        7950.362,
        8105.921,
        8117.542,
        8266.794,
        8410.521,
        8426.963,
        8523.783,
        8670.325,
        9125.471,
        9197.161,
        9227.03,
        9356.787,
        9660.435,
        9787.186,
    ]

    element = ["CuAr"] * len(gmos_atlas_lines)

    atlas = Atlas(range_tolerance=500.0)
    atlas.add_user_atlas(
        elements=element,
        wavelengths=gmos_atlas_lines,
        vacuum=True,
        pressure=61700.0,
        temperature=276.55,
        relative_humidity=4.0,
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
    ) = c.fit(max_tries=1000, fit_deg=4)

    assert np.isclose(c.fit_coeff, gmos_fit_coeff, rtol=0.01).all()


def test_osiris_fit():

    osiris_atlas_lines = [
        3650.153,
        4046.563,
        4077.831,
        4358.328,
        5460.735,
        5769.598,
        5790.663,
        6682.960,
        6752.834,
        6871.289,
        6965.431,
        7030.251,
        7067.218,
        7147.042,
        7272.936,
        7383.981,
        7503.869,
        7514.652,
        7635.106,
        7723.98,
    ]
    element = ["HgAr"] * len(osiris_atlas_lines)

    data = fits.open(
        os.path.join(
            base_dir,
            "..",
            "examples",
            "data_gtc_osiris",
            "0002672523-20200911-OSIRIS-OsirisCalibrationLamp.fits",
        )
    )[1]

    spectrum = np.median(data.data.T[550:570], axis=0)

    # Identify the arc lines
    peaks, _ = find_peaks(
        spectrum, height=1250, prominence=20, distance=3, threshold=None
    )
    peaks_refined = util.refine_peaks(spectrum, peaks, window_width=3)

    c = Calibrator(peaks_refined, spectrum=spectrum)

    c.set_hough_properties(
        num_slopes=2000,
        xbins=100,
        ybins=100,
        min_wavelength=3500.0,
        max_wavelength=8000.0,
        range_tolerance=500.0,
        linearity_tolerance=50,
    )

    atlas = Atlas(range_tolerance=500.0)
    atlas.add_user_atlas(elements=element, wavelengths=osiris_atlas_lines)
    c.set_atlas(atlas, constrain_poly=True)

    c.set_ransac_properties(
        sample_size=5,
        top_n_candidate=5,
        linear=True,
        filter_close=True,
        ransac_tolerance=5,
        candidate_weighted=True,
        hough_weight=1.0,
        minimum_matches=11,
    )

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
    ) = c.fit(max_tries=1000, fit_tolerance=10.0, fit_deg=4)

    assert np.isclose(c.fit_coeff, osiris_fit_coeff, rtol=0.01).all()


def test_sprat_fit():

    fits_file = fits.open(
        os.path.join(
            base_dir,
            "..",
            "examples",
            "data_lt_sprat",
            "v_a_20190516_57_1_0_1.fits",
        )
    )[0]

    spectrum2D = fits_file.data

    # Collapse into 1D spectrum between row 110 and 120
    spectrum = np.median(spectrum2D[110:120], axis=0)

    temperature = fits_file.header["REFTEMP"]
    pressure = fits_file.header["REFPRES"] * 100.0
    relative_humidity = fits_file.header["REFHUMID"]

    # Identify the peaks
    peaks, _ = find_peaks(
        spectrum, height=300, prominence=150, distance=5, threshold=None
    )
    peaks = util.refine_peaks(spectrum, peaks, window_width=5)

    # Initialise the calibrator
    c = Calibrator(peaks, spectrum=spectrum)
    c.set_hough_properties(
        num_slopes=2000,
        range_tolerance=200.0,
        xbins=100,
        ybins=100,
        min_wavelength=3600.0,
        max_wavelength=8000.0,
    )
    c.set_ransac_properties(
        sample_size=5, top_n_candidate=5, filter_close=True
    )
    # blend: 4829.71, 4844.33
    # blend: 5566.62, 5581.88
    # blend: 6261.212, 6265.302
    # blend: 6872.11, 6882.16
    # blend: 7283.961, 7285.301
    # blend: 7316.272, 7321.452
    atlas_lines = [
        4193.5,
        4385.77,
        4500.98,
        4524.68,
        4582.75,
        4624.28,
        4671.23,
        4697.02,
        4734.15,
        4807.02,
        4921.48,
        5028.28,
        5618.88,
        5823.89,
        5893.29,
        5934.17,
        6182.42,
        6318.06,
        6472.841,
        6595.56,
        6668.92,
        6728.01,
        6827.32,
        6976.18,
        7119.60,
        7257.9,
        7393.8,
        7584.68,
        7642.02,
        7740.31,
        7802.65,
        7887.40,
        7967.34,
        8057.258,
    ]
    element = ["Xe"] * len(atlas_lines)

    atlas = Atlas(range_tolerance=200)
    atlas.add_user_atlas(
        element,
        atlas_lines,
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
    ) = c.fit(max_tries=2500)

    assert np.isclose(c.fit_coeff[:3], sprat_fit_coeff[:3], rtol=0.01).all()
