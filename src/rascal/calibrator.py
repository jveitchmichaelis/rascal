import copy
import itertools
import logging
import os
import time

import numpy as np
import yaml
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from tqdm.auto import tqdm

from . import atlas, models, plotting
from .houghtransform import HoughTransform
from .ransac import RansacSolver
from .util import _clean_matches, _make_unique_permutation, gauss


class Calibrator:
    def __init__(
        self,
        peaks=None,
        spectrum=None,
        logger_name="Calibrator",
        log_level="warning",
    ):
        """
        Initialise the calibrator object.

        Parameters
        ----------
        peaks: list
            List of identified arc line pixel values.
        spectrum: list
            The spectral intensity as a function of pixel.

        """

        self.logger = logging.getLogger()
        self.log_level = log_level

        self.matplotlib_imported = False
        self.plotly_imported = False
        self.plot_with_matplotlib = False
        self.plot_with_plotly = False
        self.atlas = None
        self.pix_known = None
        self.wave_known = None
        self.hough_lines = None
        self.hough_points = None
        self.pairs = None
        self.ht = HoughTransform()

        # calibrator_properties
        self.num_pix = None
        self.effective_pixel = None
        self.plotting_library = None
        self.constrain_poly = None

        # hough_properties
        self.num_slopes = None
        self.xbins = None
        self.ybins = None
        self.min_wavelength = None
        self.max_wavelength = None
        self.range_tolerance = None
        self.linearity_tolerance = None

        # ransac_properties
        self.sample_size = None
        self.top_n_candidate = None
        self.linear = None
        self.filter_close = None
        self.ransac_tolerance = None
        self.candidate_weighted = None
        self.hough_weight = None
        self.minimum_matches = None
        self.minimum_peak_utilisation = None
        self.minimum_fit_error = None

        # results
        self.matched_peaks = []
        self.matched_atlas = []
        self.fit_coeff = []
        self.rms = 1e30
        self.residuals = []
        self.peak_utilisation = 0.0
        self.atlas_utilisation = 0.0

        self.success = False
        self.res = {
            "fit_coeff": None,
            "matched_peaks": None,
            "matched_atlas": None,
            "rms": None,
            "residual": None,
            "peak_utilisation": None,
            "atlas_utilisation": None,
            "success": False,
        }

        self.set_logger(logger_name, log_level)
        self.add_data(peaks, spectrum)
        self.set_calibrator_properties()
        self.set_hough_properties()
        self.set_ransac_properties()

    def add_data(self, peaks, spectrum):
        """
        Add the peaks to be solved for wavelength solution. The arc spectrum
        is optional but it would be a useful for visualisation/diagnostics.

        peaks: list
            List of identified arc line pixel values.
        spectrum: list
            The spectral intensity as a function of pixel.

        """
        self.peaks = copy.deepcopy(peaks)
        self.spectrum = copy.deepcopy(spectrum)

        if self.num_pix is None:

            self.set_num_pix(None)

    def set_logger(self, logger_name, log_level):

        # initialise the logger
        self.logger = logging.getLogger(logger_name)
        self.logger.propagate = False
        level = logging.getLevelName(log_level.upper())
        self.logger.setLevel(level)
        self.log_level = level

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] "
            "%(message)s",
            datefmt="%a, %d %b %Y %H:%M:%S",
        )

        if len(self.logger.handlers) == 0:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _generate_pairs(self):
        """
        Generate pixel-wavelength pairs without the allowed regions set by the
        linearity limit. This assumes a relatively linear spectrograph.

        Parameters
        ----------
        candidate_tolerance: float (default: 10)
            toleranceold  (Angstroms) for considering a point to be an inlier
            during candidate peak/line selection. This should be reasonable
            small as we want to search for candidate points which are
            *locally* linear.
        constrain_poly: boolean
            Apply a polygonal constraint on possible peak/atlas pairs

        """

        pairs = [
            pair
            for pair in itertools.product(self.peaks, self.atlas.get_lines())
        ]

        if self.constrain_poly:

            # Remove pairs outside polygon
            valid_area = Delaunay(
                [
                    (0, self.max_intercept + self.candidate_tolerance),
                    (0, self.min_intercept - self.candidate_tolerance),
                    (
                        self.effective_pixel.max(),
                        self.max_wavelength
                        - self.range_tolerance
                        - self.candidate_tolerance,
                    ),
                    (
                        self.effective_pixel.max(),
                        self.max_wavelength
                        + self.range_tolerance
                        + self.candidate_tolerance,
                    ),
                ]
            )

            mask = valid_area.find_simplex(pairs) >= 0
            self.pairs = np.array(pairs)[mask]

        else:

            self.pairs = np.array(pairs)

    def _merge_candidates(self, candidates):
        """
        Merge two candidate lists.

        Parameters
        ----------
        candidates: list
            list containing pixel-wavelength pairs.

        """

        merged = []

        for pairs in candidates:

            for pair in np.array(pairs).T:

                merged.append(pair)

        return np.sort(np.array(merged))

    def _get_most_common_candidates(
        self, candidates, top_n_candidate, weighted
    ):
        """
        Takes a number of candidate pair sets and returns the most common
        pair for each wavelength

        Parameters
        ----------
        candidates: list of list(float, float)
            A list of list of peak/line pairs
        top_n_candidate: int
            Top ranked lines to be fitted.
        weighted: boolean
            If True, the distance from the atlas wavelength will be used to
            compute the probilitiy based on how far it is from the Gaussian
            distribution from the known line.

        """

        peaks = []
        wavelengths = []
        probabilities = []

        for candidate in candidates:

            peaks.extend(candidate[0])
            wavelengths.extend(candidate[1])
            probabilities.extend(candidate[2])

        peaks = np.array(peaks)
        wavelengths = np.array(wavelengths)
        probabilities = np.array(probabilities)

        out_peaks = []
        out_wavelengths = []

        for peak in np.unique(peaks):

            idx = np.where(peaks == peak)

            if len(idx) > 0:

                wavelengths_matched = wavelengths[idx]

                if weighted:

                    counts = probabilities[idx]

                else:

                    counts = np.ones_like(probabilities[idx])

                n = int(
                    min(top_n_candidate, len(np.unique(wavelengths_matched)))
                )

                unique_wavelengths = np.unique(wavelengths_matched)
                aggregated_count = np.zeros_like(unique_wavelengths)
                for j, w in enumerate(unique_wavelengths):

                    idx_j = np.where(wavelengths_matched == w)
                    aggregated_count[j] = np.sum(counts[idx_j])

                out_peaks.extend([peak] * n)
                out_wavelengths.extend(
                    wavelengths_matched[np.argsort(-aggregated_count)[:n]]
                )

        return out_peaks, out_wavelengths

    def _get_candidate_points_linear(self, candidate_tolerance):
        """
        Returns a list of peak/wavelengths pairs which agree with the fit

        (wavelength - gradient * x + intercept) < tolerance

        Note: depending on the tolerance set, one peak may match with
        multiple wavelengths.

        Parameters
        ----------
        candidate_tolerance: float (default: 10)
            tolerance  (Angstroms) for considering a point to be an inlier
            during candidate peak/line selection. This should be reasonable
            small as we want to search for candidate points which are
            *locally* linear.

        """

        # Locate candidate points for these lines fits
        self.candidates = []

        for line in self.hough_lines:

            gradient, intercept = line

            predicted = gradient * self.pairs[:, 0] + intercept
            actual = self.pairs[:, 1]
            diff = np.abs(predicted - actual)
            mask = diff <= candidate_tolerance

            if sum(mask) == 0:
                continue

            # Match the range_tolerance to 1.1775 s.d. to match the FWHM
            # Note that the pairs outside of the range_tolerance were already
            # removed in an earlier stage
            weight = gauss(
                actual[mask],
                1.0,
                predicted[mask],
                (self.range_tolerance + self.linearity_tolerance) * 1.1775,
            )

            self.candidates.append(
                (self.pairs[:, 0][mask], actual[mask], weight)
            )

    def _get_candidate_points_poly(self, candidate_tolerance):
        """
        **EXPERIMENTAL**

        Returns a list of peak/wavelengths pairs which agree with the fit

        (wavelength - gradient * x + intercept) < tolerance

        Note: depending on the candidate_tolerance, one peak may
        match with multiple wavelengths.

        Parameters
        ----------
        candidate_tolerance: float (default: 10)
            toleranceold  (Angstroms) for considering a point to be an inlier
            during candidate peak/line selection. This should be reasonable
            small as we want to search for candidate points which are
            *locally* linear.

        """

        if self.fit_coeff is None:

            raise ValueError(
                "A guess solution for a polynomial fit has to "
                "be provided as fit_coeff in fit() in order to generate "
                "candidates for RANSAC sampling."
            )

        self.candidates = []

        # actual wavelengths
        actual = np.array(self.atlas.get_lines())

        n = len(self.hough_lines)

        delta = (
            np.random.random(n) * self.range_tolerance * 2.0
            - self.range_tolerance
        )

        for d in delta:

            # predicted wavelength
            predicted = self.polyval(self.peaks, self.fit_coeff) + d
            diff = np.abs(actual - predicted)
            mask = diff < candidate_tolerance

            if np.sum(mask) > 0:

                weight = gauss(
                    actual[mask], 1.0, predicted[mask], self.range_tolerance
                )
                self.candidates.append(
                    [self.peaks[mask], actual[mask], weight]
                )

    def _adjust_polyfit(self, delta, fit, tolerance, min_frac):
        """
        **EXPERIMENTAL**

        Parameters
        ----------
        delta: list or numpy.ndarray
            The first n polynomial coefficients to be shifted by delta.
        fit: list or numpy.ndarray
            The polynomial coefficients.
        tolerance: float
            The maximum difference between fit and atlas to be accounted for
            the best fit.
        min_frac: float
            The minimum fraction of lines to be used.

        Return
        ------
        lsq: float
            The least squared value of the fit.

        """

        # x is wavelength
        # x_matched is pixel
        x_matched = []
        # y_matched is wavelength
        y_matched = []
        fit_new = fit.copy()

        atlas_lines = self.atlas.get_lines()

        for i, d in enumerate(delta):

            fit_new[i] += d

        for p in self.peaks:

            x = self.polyval(p, fit_new)
            diff = atlas_lines - x
            diff_abs = np.abs(diff)
            idx = np.argmin(diff_abs)

            if diff_abs[idx] < tolerance:

                x_matched.append(p)
                y_matched.append(atlas_lines[idx])

        x_matched = np.array(x_matched)
        y_matched = np.array(y_matched)

        dof = len(x_matched) - len(fit_new) - 1

        if dof < 1:

            return np.inf

        if len(x_matched) < len(self.peaks) * min_frac:

            return np.inf

        if not np.all(
            np.diff(self.polyval(np.sort(self.effective_pixel), fit_new)) > 0
        ):

            self.logger.info("not monotonic")
            return np.inf

        lsq = (
            np.sum((y_matched - self.polyval(x_matched, fit_new)) ** 2.0) / dof
        )

        return lsq

    def load_config(self, yaml_config, y_type="filepath"):
        """
        Load a yaml configuration file to populate a Calibrator object
        and optionally an Atlas object.

        Parameters
        ----------
        yaml_config : str or pyyaml object
            Filepath or a pyyaml object
        y_type: str
            Specify 'yaml' for loading from file or 'object' to take a pyyaml
            object.

        """

        # Load from file
        if y_type == "filepath":

            with open(yaml_config, "r") as stream:

                config = yaml.safe_load(stream)

        # Load from a pyyaml object
        elif y_type == "object":

            config = yaml_config

        else:

            raise ValueError(
                "Unknown y_type: {}. Please choose from "
                "'filepath' or 'stream'".format(y_type)
            )

        # update logger properties
        self.logger = config["logger"]
        self.log_level = config["log_level"]
        self.set_logger(
            logger_name=self.logger,
            log_level=self.log_level,
        )

        # Add data to the calibrator
        self.peaks = config["peaks"]
        self.spectrum = config["spectrum"]
        self.add_data(self.peaks, self.spectrum)

        # Calibrator Properties
        self.num_pix = config["num_pix"]
        self.effective_pixel = config["effective_pixel"]
        self.plotting_library = config["plotting_library"]
        self.seed = config["seed"]
        self.hide_progress = config["hide_progress"]
        self.set_calibrator_properties(
            num_pix=self.num_pix,
            effective_pixel=self.effective_pixel,
            plotting_library=self.plotting_library,
            seed=self.seed,
            hide_progress=self.hide_progress,
        )

        self.candidate_tolerance = config["candidate_tolerance"]
        self.constrain_poly = config["constrain_poly"]

        # Atlas Properties
        # config["atlas"]:
        #   False - skip this step
        #   True - read from that rascal config
        #   string - treated as a path to an Atlas config file
        self.atlas_config = config["atlas"]
        if not self.atlas_config["config"]:

            self.logger.info("Atlas is not generated from the rascal yaml.")

        elif self.atlas_config["config"]:

            if self.atlas_config["linelist"] == "nist":

                a = atlas.Atlas()
                a.add(
                    elements=self.atlas_config["elements"],
                    linelist=self.atlas_config["linelist"],
                    min_atlas_wavelength=self.atlas_config[
                        "min_atlas_wavelength"
                    ],
                    max_atlas_wavelength=self.atlas_config[
                        "max_atlas_wavelength"
                    ],
                    min_intensity=self.atlas_config["min_intensity"],
                    min_distance=self.atlas_config["min_distance"],
                    brightest_n_lines=self.atlas_config["brightest_n_lines"],
                    vacuum=self.atlas_config["vacuum"],
                    pressure=self.atlas_config["pressure"],
                    temperature=self.atlas_config["temperature"],
                    relative_humidity=self.atlas_config["relative_humidity"],
                )
                self.logger.info(
                    "Atlas is generated from the rascal yaml using Nist lines."
                )

            # This loads the lines directly
            elif self.atlas_config["linelist"] == "user":

                a = atlas.Atlas()
                a.add_user_atlas(
                    elements=self.atlas_config["element_list"],
                    wavelengths=self.atlas_config["wavelength_list"],
                    intensities=self.atlas_config["intensity_list"],
                    vacuum=self.atlas_config["vacuum"],
                    pressure=self.atlas_config["pressure"],
                    temperature=self.atlas_config["temperature"],
                    relative_humidity=self.atlas_config["relative_humidity"],
                )
                self.logger.info(
                    "Atlas is generated from the rascal yaml using user lines."
                )

            # Unknown mode
            else:

                raise ValueError(
                    "Unknown linelist type: {}. Please choose from "
                    "'nist' or 'user'.".format(self.atlas_config["linelist"])
                )

        elif isinstance(self.atlas_config["config"], str):

            a = atlas.Atlas()
            a.load_config(self.atlas_config["config"], y_type="filepath")

        else:

            raise ValueError(
                "Unknown atlas config type: {}. Please choose from 'true', "
                "'false' or a valid path to an Atlas config.".format(
                    self.atlas_config["config"]
                )
            )

        # Only add the atlas to the Calibrator if it were set to True
        if self.atlas_config["config"]:

            self.set_atlas(a, self.candidate_tolerance, self.constrain_poly)

        # Hough transform properties
        hough_config = config["hough_transform"]

        self.num_slopes = hough_config["num_slopes"]
        self.xbins = hough_config["xbins"]
        self.ybins = hough_config["ybins"]
        self.min_wavelength = hough_config["min_wavelength"]
        self.max_wavelength = hough_config["max_wavelength"]
        self.range_tolerance = hough_config["range_tolerance"]
        self.linearity_tolerance = hough_config["linearity_tolerance"]

        self.set_hough_properties(
            num_slopes=self.num_slopes,
            xbins=self.xbins,
            ybins=self.ybins,
            min_wavelength=self.min_wavelength,
            max_wavelength=self.max_wavelength,
            range_tolerance=self.range_tolerance,
            linearity_tolerance=self.linearity_tolerance,
        )

        # RANSAC properties
        ransac_config = config["ransac"]

        self.sample_size = ransac_config["sample_size"]
        self.sampler = ransac_config["sampler"]
        self.top_n_candidate = ransac_config["top_n_candidate"]
        self.linear = ransac_config["linear"]
        self.filter_close = ransac_config["filter_close"]
        self.ransac_tolerance = ransac_config["ransac_tolerance"]
        self.candidate_weighted = ransac_config["candidate_weighted"]
        self.hough_weight = ransac_config["hough_weight"]
        self.minimum_matches = ransac_config["minimum_matches"]
        self.minimum_peak_utilisation = ransac_config[
            "minimum_peak_utilisation"
        ]
        self.minimum_fit_error = ransac_config["minimum_fit_error"]

        self.set_ransac_properties(
            sample_size=self.sample_size,
            top_n_candidate=self.top_n_candidate,
            linear=self.linear,
            filter_close=self.filter_close,
            ransac_tolerance=self.ransac_tolerance,
            candidate_weighted=self.candidate_weighted,
            hough_weight=self.hough_weight,
            minimum_matches=self.minimum_matches,
            minimum_peak_utilisation=self.minimum_peak_utilisation,
            minimum_fit_error=self.minimum_fit_error,
            sampler=self.sampler,
        )

        # Results
        result_config = config["results"]

        self.matched_peaks = result_config["matched_peaks"]
        self.matched_atlas = result_config["matched_atlas"]
        self.fit_coeff = result_config["fit_coeff"]

    def save_config(self, filename):

        output_data = {
            "peaks": self.peaks,
            "spectrum": self.spectrum,
            "num_pix": self.num_pix,
            "effective_pixel": self.effective_pixel,
            "plotting_library": self.plotting_library,
            "seed": self.seed,
            "logger": self.logger,
            "log_level": self.log_level,
            "hide_progress": self.hide_progress,
            "candidate_tolerance": self.candidate_tolerance,
            "constrain_poly": self.constrain_poly,
            "atlas": {
                "config": self.atlas_config,
                "linelist": self.atlas_config["linelist"],
                "vacuum": self.atlas_config["vacuum"],
                "pressure": self.atlas_config["pressure"],
                "temperature": self.atlas_config["temperature"],
                "relative_humidity": self.atlas_config["relative_humidity"],
                "elements": self.atlas_config["elements"],
                "min_atlas_wavelength": self.atlas_config[
                    "min_atlas_wavelength"
                ],
                "max_atlas_wavelength": self.atlas_config[
                    "max_atlas_wavelength"
                ],
                "min_intensity": self.atlas_config["min_intensity"],
                "min_distance": self.atlas_config["min_distance"],
                "brightest_n_lines": self.atlas_config["brightest_n_lines"],
                "element_list": self.atlas_config["element_list"],
                "wavelength_list": self.atlas_config["wavelength_list"],
                "intensity_list": self.atlas_config["intensity_list"],
            },
            "hough_transform": {
                "num_slopes": self.num_slopes,
                "xbins": self.xbins,
                "ybins": self.ybins,
                "min_wavelength": self.min_wavelength,
                "max_wavelength": self.max_wavelength,
                "range_tolerance": self.range_tolerance,
                "linearity_tolerance": self.linearity_tolerance,
            },
            "ransac": {
                "sample_size": self.sample_size,
                "top_n_candidate": self.top_n_candidate,
                "linear": self.linear,
                "filter_close": self.filter_close,
                "ransac_tolerance": self.ransac_tolerance,
                "candidate_weighted": self.candidate_weighted,
                "hough_weight": self.hough_weight,
                "minimum_matches": self.minimum_matches,
                "minimum_peak_utilisation": self.minimum_peak_utilisation,
                "minimum_fit_error": self.minimum_fit_error,
            },
            "results": {
                "matched_peaks": self.matched_peaks,
                "matched_atlas": self.matched_atlas,
                "fit_coeff": self.fit_coeff,
            },
        }

        with open(filename, "w+") as f:

            yaml.dump(output_data, f, default_flow_style=False)

    def which_plotting_library(self):
        """
        Call to show if the Calibrator is using matplotlib or plotly library
        (or neither).

        """

        if self.plot_with_matplotlib:

            self.logger.info("Using matplotlib.")
            return "matplotlib"

        elif self.plot_with_plotly:

            self.logger.info("Using plotly.")
            return "plotly"

        else:

            self.logger.warning("Neither maplotlib nor plotly are imported.")
            return None

    def use_matplotlib(self):
        """
        Call to switch to matplotlib.

        """

        self.plot_with_matplotlib = True
        self.plot_with_plotly = False

    def use_plotly(self):
        """
        Call to switch to plotly.

        """

        self.plot_with_plotly = True
        self.plot_with_matplotlib = False

    def set_calibrator_properties(
        self,
        num_pix=None,
        effective_pixel=None,
        plotting_library=None,
        seed=None,
        hide_progress=False,
    ):
        """
        Initialise the calibrator object.

        Parameters
        ----------
        num_pix: int
            Number of pixels in the spectral axis.
        effective_pixel: list
            pixel value of the of the spectrum, this is only needed if the
            spectrum spans multiple detector arrays.
        plotting_library: string (default: 'matplotlib')
            Choose between matplotlib and plotly.
        seed: int
            Set an optional seed for random number generators. If used,
            this parameter must be set prior to calling RANSAC. Useful
            for deterministic debugging.
        logger_name: string (default: 'Calibrator')
            The name of the logger. It can use an existing logger if a
            matching name is provided.
        log_level: string (default: 'info')
            Choose {critical, error, warning, info, debug, notset}.
        hide_progress: bool (default: False)
            Set to hide tdqm progress bar.

        """

        self.hide_progress = hide_progress

        # set the num_pix
        self.set_num_pix(num_pix)

        # set the effective_pixel
        if effective_pixel is not None:

            self.effective_pixel = np.asarray(effective_pixel)

        elif self.effective_pixel is None and self.num_pix is not None:

            self.effective_pixel = np.arange(self.num_pix)

        else:

            pass

        self.logger.info(
            "effective_pixel is set to {}.".format(effective_pixel)
        )

        if seed is not None:
            np.random.seed(seed)
            self.seed = seed

        # if the plotting library is supplied
        if plotting_library is not None:

            # set the plotting library
            self.plotting_library = plotting_library

        # if the plotting library is not supplied but the calibrator does not
        # know which library to use yet.
        elif self.plotting_library is None:

            self.plotting_library = "matplotlib"

        # everything is good
        else:

            pass

        # check the choice of plotting library is available and used.
        if self.plotting_library == "matplotlib":

            self.use_matplotlib()
            self.logger.info("Plotting with matplotlib.")

        elif self.plotting_library == "plotly":

            self.use_plotly()
            self.logger.info("Plotting with plotly.")

        else:

            self.logger.warning(
                "Unknown plotting_library, please choose from "
                "matplotlib or plotly. Execute use_matplotlib() or "
                "use_plotly() to manually select the library."
            )

    def set_num_pix(self, num_pix):

        if num_pix is not None:

            self.num_pix = num_pix
            self.effective_pixel = np.arange(self.num_pix)

        elif self.num_pix is None:

            try:

                self.num_pix = len(self.spectrum)
                self.effective_pixel = np.arange(self.num_pix)

            except Exception as e:

                self.logger.warning(e)
                self.logger.warning(
                    "Neither num_pix nor spectrum is given, "
                    "it uses 1.1 times max(peaks) as the "
                    "maximum pixel value."
                )
                try:

                    self.num_pix = 1.1 * max(self.peaks)
                    self.effective_pixel = np.arange(self.num_pix)

                except Exception as e2:

                    self.logger.warning(e)
                    self.logger.warning(
                        "num_pix cannot be set, please provide a num_pix, "
                        "or the peaks, so that we can guess the num_pix."
                    )

        else:

            pass

        self.logger.info("num_pix is set to {}.".format(num_pix))

    def set_hough_properties(
        self,
        num_slopes=None,
        xbins=None,
        ybins=None,
        min_wavelength=None,
        max_wavelength=None,
        range_tolerance=None,
        linearity_tolerance=None,
    ):
        """
        parameters
        ----------
        num_slopes: int (default: 2000)
            Number of slopes to consider during Hough transform
        xbins: int (default: 50)
            Number of bins for Hough accumulation
        ybins: int (default: 50)
            Number of bins for Hough accumulation
        min_wavelength: float (default: 3000)
            Minimum wavelength of the spectrum.
        max_wavelength: float (default: 9000)
            Maximum wavelength of the spectrum.
        range_tolerance: float (default: 500)
            Estimation of the error on the provided spectral range
            e.g. 3000-5000 with tolerance 500 will search for
            solutions that may satisfy 2500-5500
        linearity_tolerance: float (default: 100)
            A toleranceold (Ansgtroms) which defines some padding around the
            range tolerance to allow for non-linearity. This should be the
            maximum expected excursion from linearity.

        """

        # set the num_slopes
        if num_slopes is not None:

            self.num_slopes = int(num_slopes)

        elif self.num_slopes is None:

            self.num_slopes = 2000

        else:

            pass

        # set the xbins
        if xbins is not None:

            self.xbins = xbins

        elif self.xbins is None:

            self.xbins = 100

        else:

            pass

        # set the ybins
        if ybins is not None:

            self.ybins = ybins

        elif self.ybins is None:

            self.ybins = 100

        else:

            pass

        # set the min_wavelength
        if min_wavelength is not None:

            self.min_wavelength = min_wavelength

        elif self.min_wavelength is None:

            self.min_wavelength = 3000.0

        else:

            pass

        # set the max_wavelength
        if max_wavelength is not None:

            self.max_wavelength = max_wavelength

        elif self.max_wavelength is None:

            self.max_wavelength = 9000.0

        else:

            pass

        # Set the range_tolerance
        if range_tolerance is not None:

            self.range_tolerance = range_tolerance

        elif self.range_tolerance is None:

            self.range_tolerance = 500

        else:

            pass

        # Set the linearity_tolerance
        if linearity_tolerance is not None:

            self.linearity_tolerance = linearity_tolerance

        elif self.linearity_tolerance is None:

            self.linearity_tolerance = 100

        else:

            pass

        # Start wavelength in the spectrum, +/- some tolerance
        self.min_intercept = self.min_wavelength - self.range_tolerance
        self.max_intercept = self.min_wavelength + self.range_tolerance

        if self.effective_pixel is not None:

            self.min_slope = (
                (
                    self.max_wavelength
                    - self.range_tolerance
                    - self.linearity_tolerance
                )
                - (
                    self.min_intercept
                    + self.range_tolerance
                    + self.linearity_tolerance
                )
            ) / np.ptp(self.effective_pixel)

            self.max_slope = (
                (
                    self.max_wavelength
                    + self.range_tolerance
                    + self.linearity_tolerance
                )
                - (
                    self.min_intercept
                    - self.range_tolerance
                    - self.linearity_tolerance
                )
            ) / np.ptp(self.effective_pixel)

        if self.atlas is not None and self.pairs is not None:

            self._generate_pairs()

    def set_ransac_properties(
        self,
        sample_size=None,
        top_n_candidate=None,
        linear=None,
        filter_close=None,
        ransac_tolerance=None,
        candidate_weighted=None,
        hough_weight=None,
        minimum_matches=None,
        minimum_peak_utilisation=None,
        minimum_fit_error=None,
        sampler=None,
    ):
        """
        Configure the Calibrator. This may require some manual twiddling before
        the calibrator can work efficiently. However, in theory, a large
        max_tries in fit() should provide a good solution in the expense of
        performance (minutes instead of seconds).

        Parameters
        ----------
        sample_size: int (default: 5)
            Number of samples used for fitting, this is automatically
            set to the polynomial degree + 1, but a larger value can
            be specified here.
        top_n_candidate: int (default: 5)
            Top ranked lines to be fitted.
        linear: boolean (default: True)
            True to use the hough transformed gradient, otherwise, use the
            known polynomial.
        filter_close: boolean (default: False)
            Remove the pairs that are out of bounds in the hough space.
        ransac_tolerance: float (default: 5)
            The distance criteria  (Angstroms) to be considered an inlier to a
            fit. This should be close to the size of the expected residuals on
            the final fit (e.g. 1A is typical)
        candidate_weighted: boolean (default: True)
            Set to True to down-weight pairs that are far from the fit.
        hough_weight: float or None (default: 1.0)
            Set to use the hough space to weigh the fit. The theoretical
            optimal weighting is unclear. The larger the value, the heavily it
            relies on the overdensity in the hough space for a good fit.
        minimum_matches: int or None (default: 3)
            Set to only accept fit solutions with a minimum number of
            matches. Setting this will prevent the fitting function from
            accepting spurious low-error fits.
        minimum_peak_utilisation: int or None (default: 0)
            Set to only accept fit solutions with a fraction of matches. This
            option is convenient if you don't want to specify an absolute
            number of atlas lines. Range is 0 - 1 inclusive.
        minimum_fit_error: float or None (default: 1e-4)
            Set to only accept fits with a minimum error. This avoids
            accepting "perfect" fit solutions with zero errors. However
            if you have an extremely good system, you may want to set this
            tolerance lower.

        """

        # Setting the sample_size
        if sample_size is not None:

            self.sample_size = sample_size

        elif self.sample_size is None:

            self.sample_size = 5

        else:

            pass

        if sampler is not None:
            self.sampler = sampler
        else:
            self.sampler = "probabilistic"

        # Set top_n_candidate
        if top_n_candidate is not None:

            self.top_n_candidate = top_n_candidate

        elif self.top_n_candidate is None:

            self.top_n_candidate = 5

        else:

            pass

        # Set linear
        if linear is not None:

            self.linear = linear

        elif self.linear is None:

            self.linear = True

        else:

            pass

        # Set to filter closely spaced lines
        if filter_close is not None:

            self.filter_close = filter_close

        elif self.filter_close is None:

            self.filter_close = False

        else:

            pass

        # Set the ransac_tolerance
        if ransac_tolerance is not None:

            self.ransac_tolerance = ransac_tolerance

        elif self.ransac_tolerance is None:

            self.ransac_tolerance = 5

        else:

            pass

        # Set to weigh the candidate pairs by the density (pixel)
        if candidate_weighted is not None:

            self.candidate_weighted = candidate_weighted

        elif self.candidate_weighted is None:

            self.candidate_weighted = True

        else:

            pass

        # Set the multiplier of the weight of the hough density
        if hough_weight is not None:

            self.hough_weight = hough_weight

        elif self.hough_weight is None:

            self.hough_weight = 1.0

        else:

            pass

        # Set the minimum number of desired matches
        if minimum_matches is not None:

            assert minimum_matches > 0
            self.minimum_matches = minimum_matches

        elif self.minimum_matches is None:

            self.minimum_matches = 3

        else:

            pass

        # Set the minimum utilisation required
        if minimum_peak_utilisation is not None:

            assert (
                minimum_peak_utilisation >= 0
                and minimum_peak_utilisation <= 1.0
            )
            self.minimum_peak_utilisation = minimum_peak_utilisation

        elif self.minimum_peak_utilisation is None:

            self.minimum_peak_utilisation = 0

        else:

            pass

        # Set the minimum fit error
        if minimum_fit_error is not None:

            assert minimum_fit_error >= 0
            self.minimum_fit_error = minimum_fit_error

        elif self.minimum_fit_error is None:

            self.minimum_fit_error = 1e-4

        else:

            pass

    def remove_atlas_lines_range(self, wavelength, tolerance=10):
        """
        Remove arc lines within a certain wavelength range.
        """

        self.atlas.remove_atlas_lines_range(wavelength, tolerance)

    def list_atlas(self):
        """
        List all the lines loaded to the Calibrator.
        """

        self.atlas.list()

    def clear_atlas(self):
        """
        Remove all the lines loaded to the Calibrator.
        """

        self.atlas.clear()

    def set_atlas(self, atlas, candidate_tolerance=10.0, constrain_poly=False):
        """
        Adds an atlas of arc lines to the calibrator

        Parameters
        ----------
        atlas: rascal.Atlas
            Chemical symbol, case insensitive
        candidate_tolerance: float (default: 10)
            toleranceold  (Angstroms) for considering a point to be an inlier
            during candidate peak/line selection. This should be reasonable
            small as we want to search for candidate points which are
            *locally* linear.
        constrain_poly: boolean
            Apply a polygonal constraint on possible peak/atlas pairs
        """

        self.atlas = atlas

        self.candidate_tolerance = candidate_tolerance
        self.constrain_poly = constrain_poly

        # Create a list of all possible pairs of detected peaks and lines
        # from atlas
        if self.peaks is not None:

            self._generate_pairs()

    def atlas_summary(self, mode="executive", return_string=False):
        """
        Return a summary of the content of the Atlas object. The executive
        mode only return basic info. The full mode list items in details.

        Parameters
        ----------
        mode : str, optional
            Mode of summery, choose from "executive" and "full".
            (Default: "executive")

        """

        summary = self.atlas.summary(mode=mode, return_string=return_string)

        if return_string:

            return summary

    def save_atlas_summary(self, mode="full", filename=None):
        """
        Save the summary of the Atlas object, see `summary` for more detail.

        Parameters
        ----------
        mode : str, optional
            Mode of summery, choose from "executive" and "full".
            (Default: "full")
        filename : str, optional
            The export destination path, None will return with filename
            "atlas_summary_YYMMDD_HHMMSS"  (Default: None)

        """

        output_path = self.atlas.save_summary(mode=mode, filename=filename)

        return output_path

    def do_hough_transform(self, brute_force=False):

        if self.pairs is not None and not len(self.pairs) > 0:

            logging.warning("pairs list is empty. Try generating now.")
            self._generate_pairs()

            if not len(self.pairs) > 0:

                logging.error("pairs list is still empty.")

        # Generate the hough_points from the pairs
        self.ht.set_constraints(
            self.min_slope,
            self.max_slope,
            self.min_intercept,
            self.max_intercept,
        )

        if brute_force:
            self.ht.generate_hough_points_brute_force(
                self.pairs[:, 0], self.pairs[:, 1]
            )
        else:
            self.ht.generate_hough_points(
                self.pairs[:, 0], self.pairs[:, 1], num_slopes=self.num_slopes
            )

        self.ht.bin_hough_points(self.xbins, self.ybins)
        self.hough_points = self.ht.hough_points
        self.hough_lines = self.ht.hough_lines

    def save_hough_transform(
        self,
        filename="hough_transform",
        fileformat="npy",
        delimiter="+",
        to_disk=True,
    ):
        """
        Save the HoughTransform object to memory or to disk.

        Parameters
        ----------
        filename: str
            The filename of the output, not used if to_disk is False. It
            will be appended with the content type.
        format: str (default: 'npy')
            Choose from 'npy' and json'
        delimiter: str (default: '+')
            Delimiter for format and content types
        to_disk: boolean
            Set to True to save to disk, else return a numpy array object

        Returns
        -------
        hp_hough_points: numpy.ndarray
            only return if to_disk is False.

        """

        self.ht.save(
            filename=filename,
            fileformat=fileformat,
            delimiter=delimiter,
            to_disk=to_disk,
        )

    def load_hough_transform(self, filename="hough_transform", filetype="npy"):
        """
        Store the binned Hough space and/or the raw Hough pairs.

        Parameters
        ----------
        filename: str (default: 'hough_transform')
            The filename of the output, not used if to_disk is False. It
            will be appended with the content type.
        filetype: str (default: 'npy')
            The file type of the saved hough transform. Choose from 'npy'
            and 'json'.

        """

        self.ht.load(filename=filename, filetype=filetype)

    def set_known_pairs(self, pix=(), wave=()):
        """
        Provide manual pixel-wavelength pair(s), they will be appended to the
        list of pixel-wavelength pairs after the random sample being drawn from
        the RANSAC step, i.e. they are ALWAYS PRESENT in the fitting step. Use
        with caution because it can skew or bias the fit significantly, make
        sure the pixel value is accurate to at least 1/10 of a pixel. We do not
        recommend supplying more than a coupld of known pairs unless you are
        very confident with the solution and intend to skew with the known
        pairs.

        This can be used for example for low intensity lines at the edge of
        the spectrum. Or saturated lines where peaks cannot be well positioned.

        Parameters
        ----------
        pix: numeric value, list or numpy 1D array (N) (default: ())
            Any pixel value, can be outside the detector chip and
            serve purely as anchor points.
        wave: numeric value, list or numpy 1D array (N) (default: ())
            The matching wavelength for each of the pix.

        """

        pix = np.asarray(pix, dtype="float").reshape(-1)
        wave = np.asarray(wave, dtype="float").reshape(-1)

        assert pix.size == wave.size, ValueError(
            "Please check the length of the input arrays. pix has size {} "
            "and wave has size {}.".format(pix.size, wave.size)
        )

        if not all(
            isinstance(p, (float, int)) & (not np.isnan(p)) for p in pix
        ):

            raise ValueError("All pix elements have to be numeric.")

        if not all(
            isinstance(w, (float, int)) & (not np.isnan(w)) for w in wave
        ):

            raise ValueError("All wave elements have to be numeric.")

        self.pix_known = pix
        self.wave_known = wave

    def fit(
        self,
        max_tries=500,
        fit_deg=4,
        fit_coeff=None,
        fit_tolerance=5.0,
        fit_type="poly",
        candidate_tolerance=4.0,
        brute_force=False,
        progress=True,
        use_msac=True,
    ):
        """
        Solve for the wavelength calibration polynomial by getting the most
        likely solution with RANSAC.

        Parameters
        ----------
        max_tries: int (default: 5000)
            Maximum number of iteration.
        fit_deg: int (default: 4)
            The degree of the polynomial to be fitted.
        fit_coeff: list (default: None)
            Set the baseline of the least square fit. If no fits outform this
            set of polynomial coefficients, this will be used as the best fit.
        fit_tolerance: float (default: 5.0)
            Sets a tolerance on whether a fit found by RANSAC is considered
            acceptable
        fit_type: string (default: 'poly')
            One of 'poly', 'legendre' or 'chebyshev'
        candidate_tolerance: float (default: 2.0)
            toleranceold  (Angstroms) for considering a point to be an inlier
        brute_force: boolean (default: False)
            Set to True to try all possible combination in the given parameter
            space
        progress: boolean (default: True)
            True to show progress with tdqm. It is overrid if tdqm cannot be
            imported.
        use_msac: boolean
            Use M-SAC cost instead of inlier count

        Returns
        -------
        fit_coeff: list
            List of best fit polynomial fit_coefficient.
        matched_peaks: list
            Peaks used for final fit
        matched_atlas: list
            Atlas lines used for final fit
        rms: float
            The root-mean-squared of the residuals
        residual: float
            Residual from the best fit
        peak_utilisation: float
            Fraction of detected peaks (pixel) used for calibration [0-1].
        atlas_utilisation: float
            Fraction of supplied arc lines (wavelength) used for
            calibration [0-1].

        """

        self.max_tries = max_tries
        self.fit_deg = fit_deg
        self.fit_coeff = fit_coeff
        if fit_coeff is not None:

            self.fit_deg = len(fit_coeff) - 1

        self.fit_tolerance = fit_tolerance
        self.fit_type = fit_type
        self.brute_force = brute_force
        self.progress = progress
        self.use_msac = use_msac

        if self.fit_type == "poly":

            self.polyfit = np.polynomial.polynomial.polyfit
            self.polyval = np.polynomial.polynomial.polyval

        elif self.fit_type == "legendre":

            self.polyfit = np.polynomial.legendre.legfit
            self.polyval = np.polynomial.legendre.legval

        elif self.fit_type == "chebyshev":

            self.polyfit = np.polynomial.chebyshev.chebfit
            self.polyval = np.polynomial.chebyshev.chebval

        else:

            raise ValueError(
                "fit_type must be: (1) poly, (2) legendre or (3) chebyshev"
            )

        # Reduce sample_size if it is larger than the number of atlas available
        if self.sample_size > len(self.atlas):

            self.logger.warning(
                "Size of sample_size is larger than the size of atlas, "
                + "the sample_size is set to match the size of atlas = "
                + str(len(self.atlas))
                + "."
            )
            self.sample_size = len(self.atlas)

        if self.sample_size <= fit_deg:

            self.sample_size = fit_deg + 1

        if (self.hough_lines is None) or (self.hough_points is None):

            self.do_hough_transform()

        if self.minimum_matches > len(self.atlas):
            self.logger.warning(
                "Requested minimum matches is greater than the atlas size"
                "setting the minimum number of matches to equal the atlas"
                "size = " + str(len(self.atlas)) + "."
            )
            self.minimum_matches = len(self.atlas)

        if self.minimum_matches > len(self.peaks):
            self.logger.warning(
                "Requested minimum matches is greater than the number of "
                "peaks detected, which has a size of "
                "size = " + str(len(self.peaks)) + "."
            )
            self.minimum_matches = len(self.peaks)

        if self.linear:

            self._get_candidate_points_linear(candidate_tolerance)

        else:

            self._get_candidate_points_poly(candidate_tolerance)

        (
            self.candidate_peak,
            self.candidate_arc,
        ) = self._get_most_common_candidates(
            self.candidates,
            top_n_candidate=self.top_n_candidate,
            weighted=self.candidate_weighted,
        )

        # Note that there may be multiple matches for
        # each peak, that is len(x) > len(np.unique(x))
        x = np.array(self.candidate_peak)
        y = np.array(self.candidate_arc)

        self.success = False

        config = {
            "sample_size": self.sample_size,
            "filter_close": self.filter_close,
            "fit_tolerance": self.fit_tolerance,
            "hough_weight": self.hough_weight,
            "fit_type": self.fit_type,
            "max_tries": self.max_tries,
            "fit_deg": self.fit_deg,
            "use_msac": self.use_msac,
            "weight_samples": True,
            "progress": self.progress,
            "polyfit_fn": self.polyfit,
            "polyval_fn": self.polyval,
            "fit_valid_fn": self._fit_valid,
            "sampler": self.sampler,
            "hough": self.ht,
        }

        solver = RansacSolver(x, y, config)
        solver.solve()

        if solver.valid_solution:

            result = solver.best_result

            peak_utilisation = len(result.x) / len(self.peaks)
            atlas_utilisation = len(result.y) / len(self.atlas)
            self.matched_peaks = result.x
            self.matched_atlas = result.y

            if result.rms_residual > self.fit_tolerance:

                self.logger.warning(
                    "RMS too large {%s} > {%s}",
                    result.rms_residual,
                    self.fit_tolerance,
                )

            self.success = True

            self.fit_coeff = result.fit_coeffs
            self.rms = result.rms_residual
            self.residuals = result.residual
            self.peak_utilisation = peak_utilisation
            self.atlas_utilisation = atlas_utilisation

            self.res = {
                "fit_coeff": self.fit_coeff,
                "matched_peaks": self.matched_peaks,
                "matched_atlas": self.matched_atlas,
                "rms": self.rms,
                "residual": self.residuals,
                "peak_utilisation": self.peak_utilisation,
                "atlas_utilisation": self.atlas_utilisation,
                "success": self.success,
            }

        return self.res

    def _fit_valid(self, result):
        # reject lines outside the rms limit (ransac_tolerance)
        # TODO: should n_inliers be recalculated from the robust
        # fit?

        # Check the intercept.
        fit_intercept = result.fit_coeffs[0]
        if (fit_intercept < self.min_intercept) | (
            fit_intercept > self.max_intercept
        ):

            self.logger.debug(
                f"Intercept exceeds bounds, {fit_intercept} not within [{self.min_intercept}, {self.max_intercept}]."
            )
            return False

        # Check monotonicity.
        # Note, this could be pre-calculated
        pix_min = self.peaks[0] - np.ptp(self.peaks) * 0.2
        num_pix = self.peaks[-1] + np.ptp(self.peaks) * 0.2
        self.logger.debug((pix_min, num_pix))

        if not np.all(
            np.diff(
                self.polyval(
                    np.arange(result.x[0], num_pix, 1), result.fit_coeffs
                )
            )
            > 0
        ):
            self.logger.debug("Solution is not monotonically increasing.")
            return False

        # Check ends of fit:
        if self.min_wavelength is not None:

            min_wavelength_px = self.polyval(0, result.fit_coeffs)

            if min_wavelength_px < (
                self.min_wavelength - self.range_tolerance
            ) or min_wavelength_px > (
                self.min_wavelength + self.range_tolerance
            ):
                self.logger.debug(
                    "Lower wavelength of fit too small, "
                    "{:1.2f}.".format(min_wavelength_px)
                )

                return False

        if self.max_wavelength is not None:

            if self.spectrum is not None:
                fit_max_wavelength = len(self.spectrum)
            else:
                fit_max_wavelength = self.num_pix

            max_wavelength_px = self.polyval(
                fit_max_wavelength, result.fit_coeffs
            )

            if max_wavelength_px > (
                self.max_wavelength + self.range_tolerance
            ) or max_wavelength_px < (
                self.max_wavelength - self.range_tolerance
            ):
                self.logger.debug(
                    "Upper wavelength of fit too large, "
                    "{:1.2f}.".format(max_wavelength_px)
                )

                return False

        # Make sure that we don't accept fits with zero error
        if result.rms_residual < self.minimum_fit_error:

            self.logger.debug(
                "Fit error too small, " "{:1.2f}.".format(result.rms_residual)
            )

            return False

        # Check that we have enough inliers based on user specified
        # constraints
        n_inliers = len(result.x)
        if n_inliers < self.minimum_matches:

            self.logger.debug(
                "Not enough matched peaks for valid solution, "
                "user specified {}.".format(self.minimum_matches)
            )
            return False

        if n_inliers < self.minimum_peak_utilisation * len(self.peaks):

            self.logger.debug(
                "Not enough matched peaks for valid solution, "
                "user specified {:1.2f} %.".format(
                    100 * self.minimum_peak_utilisation
                )
            )
            return False

        return True

    def match_peaks(
        self,
        fit_coeff=None,
        n_delta=None,
        refine=False,
        tolerance=10.0,
        method="Nelder-Mead",
        convergence=1e-6,
        min_frac=0.5,
        robust_refit=True,
        best_err=1e9,
        fit_deg=None,
    ):
        """
        ** refine option is EXPERIMENTAL, use with caution **

        Refine the polynomial fit fit_coefficients. Recommended to use in it
        multiple calls to first refine the lowest order and gradually increase
        the order of fit_coefficients to be included for refinement. This is be
        achieved by providing delta in the length matching the number of the
        lowest degrees to be refined.

        Set refine to True to improve on the polynomial solution.

        Set robust_refit to True to fit all the detected peaks with the
        given polynomial solution for a fit using a robust polyfit, with
        the degree of polynomial = fit_deg.

        Set refine to False will return the list of
        arc lines are well fitted by the current solution within the
        tolerance limit provided.

        Parameters
        ----------
        fit_coeff: list (default: None)
            List of polynomial fit fit_coefficients.
        n_delta: int (default: None)
            The number of the lowest polynomial order to be adjusted
        refine: boolean (default: True)
            Set to True to refine solution.
        tolerance: float (default: 10.)
            Absolute difference between fit and model in the unit of nm.
        method: string (default: 'Nelder-Mead')
            scipy.optimize.minimize method.
        convergence: float (default: 1e-6)
            scipy.optimize.minimize tol.
        min_frac: float (default: 0.5)
            Minimum fractionof peaks to be refitted.
        robust_refit: boolean (default: True)
            Set to True to use a robust estimator instead of polyfit
        best_err:
            Provide the best current fit error (e.g. from calibrator.fit)
        fit_deg: int (default: length of the input fit_coefficients)
            Order of polynomial fit with all the detected peaks.

        Returns
        -------
        fit_coeff: list
            List of best fit polynomial fit_coefficient.
        peak_match: numpy 1D array
            Matched peaks
        atlas_match: numpy 1D array
            Corresponding atlas matches
        rms: float
            The root-mean-squared of the residuals
        residual: numpy 1D array
            The difference (NOT absolute) between the data and the best-fit
            solution. * EXPERIMENTAL *
        peak_utilisation: float
            Fraction of detected peaks (pixel) used for calibration [0-1].
        atlas_utilisation: float
            Fraction of supplied arc lines (wavelength) used for
            calibration [0-1].

        """

        if fit_coeff is None:

            fit_coeff = copy.deepcopy(self.fit_coeff)

        if fit_deg is None:

            fit_deg = len(fit_coeff) - 1

        if refine:

            fit_coeff_new = fit_coeff.copy()

            if n_delta is None:

                n_delta = len(fit_coeff_new) - 1

            # fit everything
            fitted_delta = minimize(
                self._adjust_polyfit,
                np.array(fit_coeff_new[: int(n_delta)]) * 1e-3,
                args=(fit_coeff, tolerance, min_frac),
                method=method,
                tol=convergence,
                options={"maxiter": 10000},
            ).x

            for i, d in enumerate(fitted_delta):

                fit_coeff_new[i] += d

            if np.any(np.isnan(fit_coeff_new)):

                self.logger.warning(
                    "_adjust_polyfit() returns None. "
                    "Input solution is returned."
                )
                self.res = {
                    "fit_coeff": self.fit_coeff,
                    "matched_peaks": self.matched_peaks,
                    "matched_atlas": self.matched_atlas,
                    "rms": self.rms,
                    "residual": self.residuals,
                    "peak_utilisation": self.peak_utilisation,
                    "atlas_utilisation": self.atlas_utilisation,
                    "success": self.success,
                }

                return self.res

            else:
                fit_coeff = fit_coeff_new

        matched_peaks = []
        matched_atlas = []
        residuals = []

        atlas_lines = self.atlas.get_lines()

        # Find all Atlas peaks within tolerance
        for p in self.peaks:

            x = self.polyval(p, fit_coeff)
            diff = atlas_lines - x
            diff_abs = np.abs(diff) < tolerance

            if diff_abs.any():

                matched_peaks.append(p)
                matched_atlas.append(list(np.asarray(atlas_lines)[diff_abs]))
                residuals.append(diff_abs)

        assert len(matched_peaks) == len(matched_atlas)

        matched_peaks = np.array(matched_peaks)

        candidates = _make_unique_permutation(_clean_matches(matched_atlas))

        if len(candidates) > 1:

            self.logger.info(
                "More than one match solution found, checking permutations."
            )

        if len(candidates) == 0:
            self.logger.warning("No candidates found.")

        for candidate in candidates:

            candidate_atlas = np.array(candidate)
            # element-wise None comparison
            valid_mask = candidate_atlas != None

            candidate_peaks = matched_peaks[valid_mask].astype(float)
            candidate_atlas = candidate_atlas[valid_mask].astype(float)

            if len(candidate_peaks) < fit_deg:
                self.logger.debug("Not enough candidate points for this fit.")
                continue

            if robust_refit:
                fit_coeff = models.robust_polyfit(
                    candidate_peaks,
                    candidate_atlas,
                    fit_deg,
                )
            else:
                fit_coeff = self.polyfit(
                    candidate_peaks, candidate_atlas, fit_deg
                )

            x = self.polyval(candidate_peaks, fit_coeff)
            residuals = np.abs(candidate_atlas - x)
            err = np.sum(residuals)

            if err < best_err:

                assert candidate_atlas is not None
                assert candidate_peaks is not None
                assert residuals is not None

                self.matched_atlas = candidate_atlas
                self.matched_peaks = candidate_peaks
                self.residuals = residuals
                self.fit_coeff = fit_coeff

                best_err = err

        self.rms = np.sqrt(
            np.nansum(self.residuals**2.0) / len(self.residuals)
        )

        self.peak_utilisation = len(self.matched_peaks) / len(self.peaks)
        self.atlas_utilisation = len(self.matched_atlas) / len(self.atlas)

        self.res = {
            "fit_coeff": self.fit_coeff,
            "matched_peaks": self.matched_peaks,
            "matched_atlas": self.matched_atlas,
            "rms": self.rms,
            "residual": self.residuals,
            "peak_utilisation": self.peak_utilisation,
            "atlas_utilisation": self.atlas_utilisation,
            "success": self.success,
        }

        return self.res

    def summary(self, return_string=False):
        """
        Return a summary of the fitted results of the Calibrator object.

        Parameters
        ----------
        return_string: bool
            Set to True to return the output string.

        """

        order_of_poly = len(self.fit_coeff)
        output = f"Order of polynomial fitted: {order_of_poly}{os.linesep}"

        for i in range(order_of_poly):

            if i == 0:

                ordinal = "st"

            elif i == 1:

                ordinal = "nd"

            elif i == 2:

                ordinal = "rd"

            else:

                ordinal = "th"

            output += "--> Coefficient of {}{} order: {}{}".format(
                i + 1, ordinal, self.fit_coeff[i], os.linesep
            )

        output += "RMS of the best fit solution: {}{}".format(
            self.rms, os.linesep
        )
        output += "Percentage of peaks unsed for fitting: {:.2f}%{}".format(
            self.peak_utilisation * 100.0, os.linesep
        )
        output += (
            "Percentage of atlas lines unsed for fitting: {:.2f}%{}".format(
                self.atlas_utilisation * 100.0, os.linesep
            )
        )

        output2 = ""
        output2_max_width = 0

        for p, a, r in zip(
            self.matched_peaks, self.matched_atlas, self.residuals
        ):

            output2_tmp = (
                "Peak {} (pix) is matched to wavelength {} A with a residual "
                "of {} A.{}".format(p, a, r, os.linesep)
            )

            if len(output2_tmp) - 2 > output2_max_width:

                output2_max_width = len(output2_tmp) - 2

            output2 += output2_tmp

        output += "+" * output2_max_width + os.linesep
        output += output2

        print(output)

        if return_string:

            return output

    def save_summary(self, filename=None):
        """
        Save the summary of the Calibrator object, see `summary` for more
        detail.

        Parameters
        ----------
        filename : str, optional
            The export destination path, None will return with filename
            "atlas_summary_YYMMDD_HHMMSS"  (Default: None)

        """

        if filename is None:

            filename = "rascal_fit_summary_{}.txt".format(
                time.strftime("%Y%m%d_%H%M%S", time.gmtime())
            )

        summary = self.summary(return_string=True)

        with open(filename, "w+") as f:
            f.write(summary)

        return filename

    def get_pix_wave_pairs(self):
        """
        Return the list of matched_peaks and matched_atlas with their
        position in the array.

        Return
        ------
        pw_pairs: list
            List of tuples each containing the array position, peak (pixel)
            and atlas (wavelength).

        """

        pw_pairs = []

        for i, (p, w) in enumerate(
            zip(self.matched_peaks, self.matched_atlas)
        ):

            pw_pairs.append((i, p, w))
            self.logger.info(
                "Position {}: pixel {} is matched to wavelength {}".format(
                    i, p, w
                )
            )

        return pw_pairs

    def add_pix_wave_pair(self, pix, wave):
        """
        Adding extra pixel-wavelength pair to the Calibrator for refitting.
        This DOES NOT work before the Calibrator having fit for a solution
        yet: use set_known_pairs() for that purpose.

        Parameters
        ----------
        pix: float
            pixel position
        wave: float
            wavelength

        """

        arg = np.argwhere(pix > self.matched_peaks)[0]

        # Only update the lists if both can be inserted
        matched_peaks = np.insert(self.matched_peaks, arg, pix)
        matched_atlas = np.insert(self.matched_atlas, arg, wave)

        self.matched_peaks = matched_peaks
        self.matched_atlas = matched_atlas

    def remove_pix_wave_pair(self, arg):
        """
        Remove fitted pixel-wavelength pair from the Calibrator for refitting.
        The positions can be found from get_pix_wave_pairs(). One at a time.

        Parameters
        ----------
        arg: int
            The position of the pairs in the arrays.

        """

        # Only update the lists if both can be deleted
        matched_peaks = np.delete(self.matched_peaks, arg)
        matched_atlas = np.delete(self.matched_atlas, arg)

        self.matched_peaks = matched_peaks
        self.matched_atlas = matched_atlas

    def manual_refit(
        self, matched_peaks=None, matched_atlas=None, degree=None, x0=None
    ):
        """
        Perform a refinement of the matched peaks and atlas lines.

        This function takes lists of matched peaks and atlases, along with
        user-specified lists of lines to add/remove from the lists.

        Any given peaks or atlas lines to remove are selected within a
        user-specified tolerance, by default 1 pixel and 5 atlas Angstrom.

        The final set of matching peaks/lines is then matched using a
        robust polyfit of the desired degree. Optionally, an initial
        fit x0 can be provided to condition the optimiser.

        The parameters are identical in the format in the fit() and
        match_peaks() functions, however, with manual changes to the lists of
        peaks and atlas, peak_utilisation and atlas_utilisation are
        meaningless so this function does not return in the same format.

        Parameters
        ----------
        matched_peaks: list
            List of matched peaks
        matched_atlas: list
            List of matched atlas lines
        degree: int
            Polynomial fit degree (Only used if x0 is None)
        x0: list
            Initial fit coefficients

        Returns
        -------
        fit_coeff: list
            List of best fit polynomial coefficients
        matched_peaks: list
            List of matched peaks
        matched_atlas: list
            List of matched atlas lines
        rms: float
            The root-mean-squared of the residuals
        residuals: numpy 1D array
            Residual match error per-peak

        """

        if matched_peaks is None:

            matched_peaks = self.matched_peaks

        if matched_atlas is None:

            matched_atlas = self.matched_atlas

        if (x0 is None) and (degree is None):

            x0 = self.fit_coeff
            degree = len(x0) - 1

        elif (x0 is not None) and (degree is None):

            assert isinstance(x0, list)
            degree = len(x0) - 1

        elif (x0 is None) and (degree is not None):

            assert isinstance(degree, int)

        else:

            assert isinstance(x0, list)
            assert isinstance(degree, int)
            assert len(x0) == degree + 1

        x = np.asarray(matched_peaks)
        y = np.asarray(matched_atlas)

        assert len(x) == len(y)
        assert len(x) > 0
        assert degree > 0
        assert degree <= len(x) - 1

        # Run robust fitting again
        fit_coeff_new = models.robust_polyfit(x, y, degree, x0)
        self.logger.info("Input fit_coeff is {}.".format(x0))
        self.logger.info("Refit fit_coeff is {}.".format(fit_coeff_new))

        self.fit_coeff = fit_coeff_new
        self.matched_peaks = copy.deepcopy(matched_peaks)
        self.matched_atlas = copy.deepcopy(matched_atlas)
        self.residuals = y - self.polyval(x, fit_coeff_new)
        self.rms = np.sqrt(
            np.nansum(self.residuals**2.0) / len(self.residuals)
        )

        self.peak_utilisation = len(self.matched_peaks) / len(self.peaks)
        self.atlas_utilisation = len(self.matched_atlas) / len(self.atlas)
        self.success = True

        self.res = {
            "fit_coeff": self.fit_coeff,
            "matched_peaks": self.matched_peaks,
            "matched_atlas": self.matched_atlas,
            "rms": self.rms,
            "residual": self.residuals,
            "peak_utilisation": self.peak_utilisation,
            "atlas_utilisation": self.atlas_utilisation,
            "success": self.success,
        }

        return self.res

    def save_matches(self, filename=None, format="csv"):
        """
        Export the matched peak-atlas pairs

        parameters
        ----------
        filename: str (Default: None)
            Export file name, if None, it will be saved as "matched_peaks"
        format: str (Default: csv)
            Export format, choose from csv and npy.

        """

        output = np.column_stack((self.matched_peaks, self.matched_atlas))

        if filename is None:

            filename = "matched_peaks"

        if format.lower() == "csv":

            np.savetxt(filename + ".csv", X=output, delimiter=",")

        elif format.lower() == "npy":

            np.save(file=filename + ".npy", arr=output)

    def plot_arc(
        self,
        effective_pixel=None,
        log_spectrum=False,
        save_fig=False,
        fig_type="png",
        filename=None,
        return_jsonstring=False,
        renderer="default",
        display=True,
    ):
        """
        Plots the 1D spectrum of the extracted arc.

        parameters
        ----------
        effective_pixel: array (default: None)
            pixel value of the of the spectrum, this is only needed if the
            spectrum spans multiple detector arrays.
        log_spectrum: boolean (default: False)
            Set to true to display the wavelength calibrated arc spectrum in
            logarithmic space.
        save_fig: boolean (default: False)
            Save an image if set to True. matplotlib uses the pyplot.save_fig()
            while the plotly uses the pio.write_html() or pio.write_image().
            The support format types should be provided in fig_type.
        fig_type: string (default: 'png')
            Image type to be saved, choose from:
            jpg, png, svg, pdf and iframe. Delimiter is '+'.
        filename: string (default: None)
            Provide a filename or full path. If the extension is not provided
            it is defaulted to png.
        return_jsonstring: boolean (default: False)
            Set to True to return json strings if using plotly as the plotting
            library.
        renderer: string (default: 'default')
            Indicate the Plotly renderer. Nothing gets displayed if json is
            set to True.
        display: boolean (Default: False)
            Set to True to display disgnostic plot.

        Returns
        -------
        Return json strings if using plotly as the plotting library and json
        is True.

        """

        return plotting.plot_arc(
            self,
            effective_pixel=effective_pixel,
            log_spectrum=log_spectrum,
            save_fig=save_fig,
            fig_type=fig_type,
            filename=filename,
            return_jsonstring=return_jsonstring,
            renderer=renderer,
            display=display,
        )

    def plot_search_space(
        self,
        fit_coeff=None,
        top_n_candidate=3,
        weighted=True,
        save_fig=False,
        fig_type="png",
        filename=None,
        return_jsonstring=False,
        renderer="default",
        display=True,
    ):
        """
        Plots the peak/arc line pairs that are considered as potential match
        candidates.

        If fit fit_coefficients are provided, the model solution will be
        overplotted.

        Parameters
        ----------
        fit_coeff: list (default: None)
            List of best polynomial fit_coefficients
        top_n_candidate: int (default: 3)
            Top ranked lines to be fitted.
        weighted: (default: True)
            Draw sample based on the distance from the matched known wavelength
            of the atlas.
        save_fig: boolean (default: False)
            Save an image if set to True. matplotlib uses the pyplot.save_fig()
            while the plotly uses the pio.write_html() or pio.write_image().
            The support format types should be provided in fig_type.
        fig_type: string (default: 'png')
            Image type to be saved, choose from:
            jpg, png, svg, pdf and iframe. Delimiter is '+'.
        filename: (default: None)
            The destination to save the image.
        return_jsonstring: (default: False)
            Set to True to save the plotly figure as json string. Ignored if
            matplotlib is used.
        renderer: (default: 'default')
            Set the rendered for the plotly display. Ignored if matplotlib is
            used.
        display: boolean (Default: False)
            Set to True to display disgnostic plot.

        Return
        ------
        json object if json is True.

        """

        return plotting.plot_search_space(
            self,
            fit_coeff=fit_coeff,
            top_n_candidate=top_n_candidate,
            weighted=weighted,
            save_fig=save_fig,
            fig_type=fig_type,
            filename=filename,
            return_jsonstring=return_jsonstring,
            renderer=renderer,
            display=display,
        )

    def plot_fit(
        self,
        fit_coeff=None,
        spectrum=None,
        plot_atlas=True,
        log_spectrum=False,
        save_fig=False,
        fig_type="png",
        filename=None,
        return_jsonstring=False,
        renderer="default",
        display=True,
    ):
        """
        Plots of the wavelength calibrated arc spectrum, the residual and the
        pixel-to-wavelength solution.

        Parameters
        ----------
        fit_coeff: 1D numpy array or list
            Best fit polynomial fit_coefficients
        spectrum: 1D numpy array (N)
            Array of length N pixels
        plot_atlas: boolean (default: True)
            Display all the relavent lines available in the atlas library.
        log_spectrum: boolean (default: False)
            Display the arc in log-space if set to True.
        save_fig: boolean (default: False)
            Save an image if set to True. matplotlib uses the pyplot.save_fig()
            while the plotly uses the pio.write_html() or pio.write_image().
            The support format types should be provided in fig_type.
        fig_type: string (default: 'png')
            Image type to be saved, choose from:
            jpg, png, svg, pdf and iframe. Delimiter is '+'.
        filename: string (default: None)
            Provide a filename or full path. If the extension is not provided
            it is defaulted to png.
        return_jsonstring: boolean (default: False)
            Set to True to return json strings if using plotly as the plotting
            library.
        renderer: string (default: 'default')
            Indicate the Plotly renderer. Nothing gets displayed if json is
            set to True.
        display: boolean (Default: False)
            Set to True to display disgnostic plot.

        Returns
        -------
        Return json strings if using plotly as the plotting library and json
        is True.

        """

        if fit_coeff is None:

            fit_coeff = self.fit_coeff

        return plotting.plot_fit(
            self,
            fit_coeff=fit_coeff,
            spectrum=spectrum,
            plot_atlas=plot_atlas,
            log_spectrum=log_spectrum,
            save_fig=save_fig,
            fig_type=fig_type,
            filename=filename,
            return_jsonstring=return_jsonstring,
            renderer=renderer,
            display=display,
        )
