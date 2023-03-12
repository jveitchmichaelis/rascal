#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is the core of rascal.

"""

import copy
import itertools
import logging
import os
import random
import time
from pathlib import Path
from typing import List, Union

import numpy as np
from omegaconf import OmegaConf
from scipy import interpolate as itp
from scipy.optimize import minimize
from scipy.spatial import Delaunay

from . import atlas, models, plotting
from .atlas import Atlas, AtlasCollection, AtlasLine
from .config import CalibratorConfig
from .houghtransform import HoughTransform
from .ransac import RansacSolver, SolveResult
from .util import _clean_matches, _make_unique_permutation, gauss


class Calibrator:
    """
    Josh will write something here.

    """

    def __init__(
        self,
        peaks: List[float],
        atlas_lines: List[AtlasLine],
        config: Union[str, list, dict] = None,
        spectrum: Union[list, np.ndarray] = None,
    ):
        """
        Initialise the calibrator object.

        Parameters
        ----------
        peaks: list
            List of identified arc line pixel values, required.
        atlas_lines: list
            List of atlas lines, required.
        config: str, dict, list
            Configuration override, optional.
        spectrum: list
            The spectral intensity as a function of pixel, optional.

        """

        # Default calibrator_properties
        self.config = OmegaConf.structured(CalibratorConfig)

        # Merge from user-supplied file, dict, etc.
        if config is not None:
            # Path to YAML
            if isinstance(config, str) or isinstance(config, Path):
                user_config = OmegaConf.load(config)
            elif isinstance(config, list) or isinstance(config, dict):
                user_config = OmegaConf.create(config)
            elif isinstance(config, CalibratorConfig):
                user_config = config
            else:
                raise NotImplementedError(
                    f"This config format {type(config)} is not supported yet."
                )

            self.config = OmegaConf.merge(self.config, user_config)

        # Finally overrides from CLI
        # cli_conf = OmegaConf.from_cli()
        # OmegaConf.merge(self.config, cli_conf)

        self.matplotlib_imported = False
        self.plotly_imported = False
        self.plot_with_matplotlib = False
        self.plot_with_plotly = False

        self.atlas_lines = atlas_lines
        self.contiguous_pixel = []

        self.pix_known = None
        self.wave_known = None
        self.hough_lines = None
        self.hough_points = None
        self.pairs = None
        self.hough_transformer = HoughTransform()

        # fitting properties
        self.progress = None
        self.polyfit = None
        self.polyval = None
        self.candidates = None
        self.candidate_peak = None
        self.candidate_arc = None

        # results
        self.matched_peaks = None
        self.matched_atlas = None
        self.fit_coeff = None
        self.rms = None
        self.residuals = None
        self.peak_utilisation = None
        self.atlas_utilisation = None
        self.success = False

        # Result dict
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

        self.set_logger(self.config.logger_name, self.config.log_level)

        # General calibrator setup
        self._set_calibrator_properties()

        # Spectral data input, modifies the config
        self.peaks = None
        self.spectrum = None
        self._set_data(peaks, spectrum)

        # Pixel setup
        self._set_num_pix()

        # Hough Transform setup
        self._set_hough_properties()

        self._generate_pairs()

        # RANSAC Properties
        self._check_ransac_properties()

        # Freeze the configuration for this experiment.
        OmegaConf.set_readonly(self.config, True)

    def _set_calibrator_properties(
        self,
    ):
        """
        Initialise general properties

        """

        if self.config.seed is not None:
            self.config.seed = random.randint(0, 1000000)

        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
        self.logger.info(f"Seeded RNG with: {self.config.seed}.")

        # check the choice of plotting library is available and used.
        if self.config.plotting_library == "matplotlib":
            self.use_matplotlib()
            self.logger.info("Plotting with matplotlib.")

        elif self.config.plotting_library == "plotly":
            self.use_plotly()
            self.logger.info("Plotting with plotly.")

        else:
            self.logger.warning(
                "Unknown plotting_library, please choose from "
                "matplotlib or plotly. Execute use_matplotlib() or "
                "use_plotly() to manually select the library."
            )

    def _set_data(
        self,
        peaks: list = None,
        spectrum: list = None,
    ):
        """
        Add the peaks to be solved for wavelength solution. The arc spectrum
        is optional but it can be useful for visualisation/diagnostics.

        peaks: list
            List of identified arc line pixel values.
        spectrum: list
            The spectral intensity as a function of pixel.

        """
        self.peaks = copy.deepcopy(peaks)

        if spectrum is not None:
            self.spectrum = copy.deepcopy(spectrum)

        # Check atlas
        for line in self.atlas_lines:
            if (
                line.wavelength < 0
                or line.wavelength < self.config.data.detector_min_wave
                or line.wavelength > self.config.data.detector_max_wave
            ):
                self.logger.warning(
                    f"The provided peak {line.wavelength} is outside the given range of the detector."
                )

    def _set_num_pix(self):
        """
        Josh will write something here.

        """

        # We're given a spectrum
        if self.spectrum is not None:
            if self.config.data.num_pix is None:
                self.config.data.num_pix = len(self.spectrum)
            else:
                # Assume that if number of pixels *and* spectrum
                # are provided, they should be the same length
                assert (
                    len(self.spectrum) == self.config.data.num_pix
                ), "The length of the provided spectrum should match the num_pix"

        # No spectrum provided and num_pix not provided
        elif self.config.data.num_pix is None:
            if len(self.peaks) > 0:
                self.config.data.num_pix = int(round(1.1 * max(self.peaks)))
                self.logger.warning(
                    "Neither num_pix nor spectrum is given, "
                    "it uses 1.1 times max(peaks) as the "
                    "maximum pixel value."
                )
            else:
                self.logger.warning(
                    "num_pix cannot be set, please provide a num_pix, "
                    "or the peaks, so that we can guess the num_pix."
                )
        # Only user-provided num pixels, so just check that it's
        # greater than the peak with the highest index
        else:
            if self.config.data.num_pix <= max(self.peaks):
                self.logger.error(
                    f"Maximum pixel {self.config.data.num_pix} is too low, max peak provided is {max(self.peaks)}"
                )
                raise ValueError

        self.logger.info(f"num_pix is set to {self.config.data.num_pix}.")

        # Default 1:1 mapping between pixel location and effective pixel location
        if self.config.data.contiguous_range is None:
            self.contiguous_pixel = list(range(self.config.data.num_pix))

        # Otherwise assert the effective pixel array is the same as the number
        # of pixels in the spectrum
        else:
            contiguous_ranges = np.array(
                self.config.data.contiguous_range
            ).flatten()
            n_ranges = int(contiguous_ranges.size / 2)
            assert n_ranges % 1 == 0

            for i in range(n_ranges):
                x0 = contiguous_ranges[i * 2]
                x1 = contiguous_ranges[i * 2 + 1] + 1
                self.contiguous_pixel.extend(list(np.arange(x0, x1)))

            assert (
                len(self.contiguous_pixel) == self.config.data.num_pix
            ), f"The length of the effective pixel array ({len(self.contiguous_pixel)}) should match num_pix ({self.config.data.num_pix})"

        self.pixel_mapping_itp = itp.interp1d(
            np.arange(self.config.data.num_pix), self.contiguous_pixel
        )
        self.peaks_effective = self.pixel_mapping_itp(np.array(self.peaks))

        self.logger.debug(
            f"contiguous_pixel is set to {self.contiguous_pixel}."
        )

    def set_logger(self, logger_name: str, log_level: str):
        """
        Josh will write something here.

        """

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
            for pair in itertools.product(
                self.peaks, [line.wavelength for line in self.atlas_lines]
            )
        ]

        if self.config.hough.constrain_poly:
            # Remove pairs outside polygon
            valid_area = Delaunay(
                [
                    (
                        0,
                        self.config.hough.max_intercept
                        + self.config.hough.linearity_tolerance,
                    ),
                    (
                        0,
                        self.config.hough.min_intercept
                        - self.config.hough.linearity_tolerance,
                    ),
                    (
                        self.contiguous_pixel.max(),
                        self.max_wavelength
                        - self.config.hough.range_tolerance
                        - self.config.hough.linearity_tolerance,
                    ),
                    (
                        self.contiguous_pixel.max(),
                        self.max_wavelength
                        + self.config.hough.range_tolerance
                        + self.config.hough.linearity_tolerance,
                    ),
                ]
            )

            mask = valid_area.find_simplex(pairs) >= 0
            self.pairs = np.array(pairs)[mask]

        else:
            self.pairs = np.array(pairs)

        self.logger.debug(f"Pairs: {self.pairs}")

    def _check_ransac_properties(
        self,
    ):
        """
        Check that RANSAC properties are within limits.
        """

        assert self.config.ransac.sample_size > self.config.ransac.degree
        assert self.config.ransac.top_n_candidate > 1
        assert self.config.ransac.inlier_tolerance > 0
        assert self.config.ransac.minimum_matches > 0
        assert self.config.ransac.hough_weight >= 0
        assert self.config.ransac.sampler in [
            "probabilistic",
            "weighted",
            "uniform",
        ]
        assert self.config.ransac.minimum_peak_utilisation >= 0
        assert self.config.ransac.minimum_peak_utilisation <= 1.0
        assert self.config.ransac.minimum_fit_error >= 0

    def _set_hough_properties(self):
        """
        Internal function which calculates Hough transform
        parameters based on the existing configuration.

        This includes the minimum and maximum intercepts, which
        are defined by the min wavelength and the range
        tolerances given.

        We then calculate the min and max slope, based on the
        intercepts, the maximum wavelength and the linearity
        tolerance.

        """

        # Start wavelength in the spectrum, +/- some tolerance
        self.config.hough.min_intercept = float(
            self.config.data.detector_min_wave
            - self.config.hough.range_tolerance
        )
        self.min_intercept = self.config.hough.min_intercept  # TODO fix this

        self.config.hough.max_intercept = float(
            self.config.data.detector_min_wave
            + self.config.hough.range_tolerance
        )
        self.max_intercept = self.config.hough.max_intercept  # TODO fix this

        if self.contiguous_pixel is not None:
            self.config.hough.min_slope = float(
                (
                    (
                        self.config.data.detector_max_wave
                        - self.config.hough.range_tolerance
                        - self.config.hough.linearity_tolerance
                    )
                    - (
                        self.config.hough.min_intercept
                        + self.config.hough.range_tolerance
                        + self.config.hough.linearity_tolerance
                    )
                )
                / np.ptp(self.contiguous_pixel)
            )

            self.config.hough.max_slope = float(
                (
                    (
                        self.config.data.detector_max_wave
                        + self.config.hough.range_tolerance
                        + self.config.hough.linearity_tolerance
                    )
                    - (
                        self.config.hough.min_intercept
                        - self.config.hough.range_tolerance
                        - self.config.hough.linearity_tolerance
                    )
                )
                / np.ptp(self.contiguous_pixel)
            )

        self.logger.debug("Hough parameters:")
        self.logger.debug(
            f"Minimum intercept: {self.config.hough.min_intercept}"
        )
        self.logger.debug(
            f"Maximum intercept: {self.config.hough.max_intercept}"
        )
        self.logger.debug(f"Minimum slope: {self.config.hough.min_slope}")
        self.logger.debug(f"Maximum slope: {self.config.hough.max_slope}")
        self.logger.debug(
            f"Minimum wavelength: {self.config.data.detector_min_wave}"
        )
        self.logger.debug(
            f"Maximum wavelength: {self.config.data.detector_max_wave}"
        )

    def _merge_candidates(self, candidates: Union[list, np.ndarray]):
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
        self, candidates: list, top_n_candidate: int, weighted: bool
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

                num = int(
                    min(top_n_candidate, len(np.unique(wavelengths_matched)))
                )

                unique_wavelengths = np.unique(wavelengths_matched)
                aggregated_count = np.zeros_like(unique_wavelengths)
                for j, wave in enumerate(unique_wavelengths):
                    idx_j = np.where(wavelengths_matched == wave)
                    aggregated_count[j] = np.sum(counts[idx_j])

                out_peaks.extend([peak] * num)
                out_wavelengths.extend(
                    wavelengths_matched[np.argsort(-aggregated_count)[:num]]
                )

        return out_peaks, out_wavelengths

    def _get_candidate_points_linear(self, candidate_tolerance: float):
        """
        Returns a list of peak/wavelengths pairs which agree with the fit

        (wavelength - gradient * x + intercept) < tolerance

        Note: depending on the tolerance set, one peak may match with
        multiple wavelengths.

        Parameters
        ----------
        candidate_tolerance: float
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

            if not np.any(mask):
                continue

            # Match the range_tolerance to 1.1775 s.d. to match the FWHM
            # Note that the pairs outside of the range_tolerance were already
            # removed in an earlier stage
            weight = gauss(
                actual[mask],
                1.0,
                predicted[mask],
                (
                    self.config.hough.range_tolerance
                    + self.config.hough.linearity_tolerance
                )
                * 1.1775,
            )

            self.candidates.append(
                (self.pairs[:, 0][mask], actual[mask], weight)
            )

    def _get_candidate_points_poly(self, candidate_tolerance: float):
        """
        **EXPERIMENTAL**

        Returns a list of peak/wavelengths pairs which agree with the fit

        (wavelength - gradient * x + intercept) < tolerance

        Note: depending on the candidate_tolerance, one peak may
        match with multiple wavelengths.

        Parameters
        ----------
        candidate_tolerance: float
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

        num = len(self.hough_lines)

        delta = (
            np.random.random(num) * self.range_tolerance * 2.0
            - self.range_tolerance
        )

        for _d in delta:
            # predicted wavelength
            predicted = self.polyval(self.peaks_effective, self.fit_coeff) + _d
            diff = np.abs(actual - predicted)
            mask = diff < candidate_tolerance

            if np.any(mask):
                weight = gauss(
                    actual[mask], 1.0, predicted[mask], self.range_tolerance
                )
                self.candidates.append(
                    [self.peaks_effective[mask], actual[mask], weight]
                )

    def _adjust_polyfit(
        self,
        delta: Union[list, np.ndarray],
        fit: Union[list, np.ndarray],
        tolerance: float,
        min_frac: float,
    ):
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

        atlas_lines = [line.wavelength for line in self.atlas_lines]

        for i, _d in enumerate(delta):
            fit_new[i] += _d

        for _p in self.peaks_effective:
            _x = self.polyval(_p, fit_new)
            diff = atlas_lines - _x
            diff_abs = np.abs(diff)
            idx = np.argmin(diff_abs)

            if diff_abs[idx] < tolerance:
                x_matched.append(_p)
                y_matched.append(atlas_lines[idx])

        x_matched = np.array(x_matched)
        y_matched = np.array(y_matched)

        dof = len(x_matched) - len(fit_new) - 1

        if dof < 1:
            return np.inf

        if len(x_matched) < len(self.peaks_effective) * min_frac:
            return np.inf

        if not np.all(
            np.diff(self.polyval(np.sort(self.contiguous_pixel), fit_new)) > 0
        ):
            self.logger.info("not monotonic")
            return np.inf

        lsq = (
            np.sum((y_matched - self.polyval(x_matched, fit_new)) ** 2.0) / dof
        )

        return lsq

    def save_config(self, filename: str):
        """
        Save the current configuration to a YAML file. Will create
        intermediate directory structure if necessary.

        Parameters
        ----------
        filename: str
            Output filename

        """

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, "w") as fp:
            OmegaConf.save(config=self.confiog, f=fp)

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

    def do_hough_transform(self, brute_force: bool = False):
        """
        Josh will write something here.

        """

        if self.pairs is not None and not len(self.pairs) > 0:
            logging.warning("pairs list is empty. Try generating now.")
            self._generate_pairs()

            if not len(self.pairs) > 0:
                logging.error("pairs list is still empty.")

        # Generate the hough_points from the pairs
        self.hough_transformer.set_constraints(
            self.config.hough.min_slope,
            self.config.hough.max_slope,
            self.config.hough.min_intercept,
            self.config.hough.max_intercept,
        )

        if brute_force:
            self.hough_transformer.generate_hough_points_brute_force(
                self.pairs[:, 0], self.pairs[:, 1]
            )
        else:
            self.hough_transformer.generate_hough_points(
                self.pairs[:, 0],
                self.pairs[:, 1],
                num_slopes=self.config.hough.num_slopes,
            )

        self.hough_transformer.bin_hough_points(
            self.config.hough.xbins, self.config.hough.ybins
        )
        self.hough_points = self.hough_transformer.hough_points
        self.hough_lines = self.hough_transformer.hough_lines

    def save_hough_transform(
        self,
        filename: str = "hough_transform",
        fileformat: str = "npy",
        delimiter: str = "+",
        to_disk: bool = True,
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

        self.hough_transformer.save(
            filename=filename,
            fileformat=fileformat,
            delimiter=delimiter,
            to_disk=to_disk,
        )

    def load_hough_transform(
        self, filename: str = "hough_transform", filetype: str = "npy"
    ):
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

        self.hough_transformer.load(filename=filename, filetype=filetype)

    def set_known_pairs(
        self,
        pix: Union[float, list, np.ndarray],
        wave: Union[float, list, np.ndarray],
    ):
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
            "Please check the length of the input arrays. pix has size "
            + f"{pix.size} and wave has size {wave.size}."
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
        max_tries: int = None,
        fit_deg: int = None,
        fit_coeff: Union[list, np.ndarray] = None,
        fit_tolerance: float = None,
        fit_type: str = None,
        candidate_tolerance: float = None,
        progress: bool = True,
        use_msac: bool = False,
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

        self.fit_deg = self.config.ransac.degree
        self.fit_coeff = fit_coeff

        if fit_coeff is not None:
            self.fit_deg = len(fit_coeff) - 1

        self.fit_tolerance = self.config.ransac.rms_tolerance
        self.progress = progress

        if self.config.ransac.type == "poly":
            self.polyfit = np.polynomial.polynomial.polyfit
            self.polyval = np.polynomial.polynomial.polyval

        elif self.config.ransac.type == "legendre":
            self.polyfit = np.polynomial.legendre.legfit
            self.polyval = np.polynomial.legendre.legval

        elif self.config.ransac.type == "chebyshev":
            self.polyfit = np.polynomial.chebyshev.chebfit
            self.polyval = np.polynomial.chebyshev.chebval

        else:
            raise ValueError(
                "fit_type must be: (1) poly, (2) legendre or (3) chebyshev"
            )

        n_lines = len(self.atlas_lines)

        # Reduce sample_size if it is larger than the number of atlas available
        """

        THIS IS NEVER USED

        if self.config.ransac.sample_size > n_lines:

            self.logger.warning(
                "Size of sample_size is larger than the size of atlas, "
                + "the sample_size is set to match the size of atlas = "
                + str(n_lines)
                + "."
            )
        
            self.sample_size = n_lines

        if self.sample_size <= fit_deg:

            self.sample_size = fit_deg + 1
        """

        if (self.hough_lines is None) or (self.hough_points is None):
            self.do_hough_transform()

        """

        This is now handled by RANSAC?

        if self.ransac.minimum_matches > n_lines:
            self.logger.warning(
                "Requested minimum matches is greater than the atlas size "
                + "setting the minimum number of matches to equal the atlas "
                + f"size = {n_lines}."
            )
            self.minimum_matches = n_lines

        if self.minimum_matches > len(self.peaks_effective):
            self.logger.warning(
                "Requested minimum matches is greater than the number of "
                + "peaks detected, which has a size of "
                + f"size = {len(self.peaks_effective)}."
            )
            self.minimum_matches = len(self.peaks_effective)

        """

        if self.config.ransac.linear:
            self.logger.debug("Using linear candidate points")
            self._get_candidate_points_linear(
                self.config.ransac.inlier_tolerance
            )

        else:
            self.logger.debug("Using polynomial candidate points")
            self._get_candidate_points_poly(
                self.config.ransac.inlier_tolerance
            )

        (
            self.candidate_peak,
            self.candidate_arc,
        ) = self._get_most_common_candidates(
            self.candidates,
            top_n_candidate=self.config.ransac.top_n_candidate,
            weighted=self.config.ransac.candidate_weighted,
        )

        # Note that there may be multiple matches for
        # each peak, that is len(x) > len(np.unique(x))
        _x = np.array(self.candidate_peak)
        _y = np.array(self.candidate_arc)

        self.success = False

        self.logger.debug(
            f"Candidates: {[xy for xy in list(zip(_x.round(2), _y.round(2)))]}"
        )

        self.logger.debug("Starting RANSAC")
        solver = RansacSolver(_x, _y, config=self.config.ransac)
        solver.solve()

        if solver.valid_solution:
            result = solver.best_result

            peak_utilisation = len(result.x) / len(self.peaks)
            atlas_utilisation = len(result.y) / n_lines
            self.matched_peaks = result.x
            self.matched_atlas = result.y

            if result.rms_residual > self.fit_tolerance:
                self.logger.warning(
                    f"RMS too large {result.rms_residual} > "
                    + f"{self.fit_tolerance}."
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
                "candidates": [
                    xy for xy in list(zip(_x.round(2), _y.round(2)))
                ],
            }

        return self.res

    def _fit_valid(self, result: SolveResult):
        """
        Josh will write something here.

        """

        # reject lines outside the rms limit (ransac_tolerance)
        # TODO: should n_inliers be recalculated from the robust
        # fit?

        assert self.min_intercept is not None and self.min_intercept > 0
        assert self.max_intercept is not None

        # Check the intercept.
        fit_intercept = result.fit_coeffs[0]
        if (fit_intercept < self.min_intercept) | (
            fit_intercept > self.max_intercept
        ):
            self.logger.debug(
                f"Intercept exceeds bounds, {fit_intercept} not within "
                + f"[{self.min_intercept}, {self.max_intercept}]."
            )
            return False

        # Check monotonicity.
        # Note, this could be pre-calculated
        pix_min = self.peaks_effective[0] - np.ptp(self.peaks_effective) * 0.2
        num_pix = self.peaks_effective[-1] + np.ptp(self.peaks_effective) * 0.2
        self.logger.debug(f"pix_min: {pix_min}, num_pix: {num_pix}")

        if not np.all(
            np.diff(
                self.polyval(
                    np.arange(result.inliers_x[0], num_pix, 1),
                    result.fit_coeffs,
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
                self.config.data.detector_min_wave
                - self.config.data.detector_edge_tolerance
            ) or min_wavelength_px > (
                self.config.data.detector_min_wave
                + self.config.data.detector_edge_tolerance
            ):
                self.logger.debug(
                    "Lower wavelength of fit too small, "
                    + f"{min_wavelength_px:1.2f}."
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
                self.config.data.detector_max_wave
                + self.detector_edge_tolerance
            ) or max_wavelength_px < (
                self.config.data.detector_max_wave
                - self.detector_edge_tolerance
            ):
                self.logger.debug(
                    "Upper wavelength of fit too large, "
                    + f"{max_wavelength_px:1.2f}."
                )

                return False

        # Make sure that we don't accept fits with zero error
        if result.rms_residual < self.config.ransac.minimum_fit_error:
            self.logger.debug(
                "Fit error too small, {result.rms_residual:1.2f}."
            )

            return False

        # Check that we have enough inliers based on user specified
        # constraints
        n_inliers = len(result.inliers_x)
        if n_inliers < self.minimum_matches:
            self.logger.debug(
                "Not enough matched peaks for valid solution, "
                + f"user specified {self.minimum_matches}."
            )
            return False

        if n_inliers < self.minimum_peak_utilisation * len(
            self.peaks_effective
        ):
            self.logger.debug(
                "Not enough matched peaks for valid solution, user specified "
                + f"{100 * self.minimum_peak_utilisation:1.2f} %."
            )
            return False

        return True

    def match_peaks(
        self,
        fit_coeff: Union[list, np.ndarray] = None,
        n_delta: int = None,
        refine: bool = False,
        tolerance: float = 10.0,
        method: str = "Nelder-Mead",
        convergence: float = 1e-6,
        min_frac: float = 0.5,
        robust_refit: bool = True,
        best_err: float = 1e9,
        fit_deg: int = None,
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

            for i, _d in enumerate(fitted_delta):
                fit_coeff_new[i] += _d

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

        atlas_lines = [line.wavelength for line in self.atlas_lines]

        # Find all Atlas peaks within tolerance
        for _p in self.peaks_effective:
            _x = self.polyval(_p, fit_coeff)
            diff = atlas_lines - _x
            diff_abs = np.abs(diff) < tolerance

            if diff_abs.any():
                matched_peaks.append(_p)
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

            _x = self.polyval(candidate_peaks, fit_coeff)
            residuals = np.abs(candidate_atlas - _x)
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
        self.atlas_utilisation = len(self.matched_atlas) / len(
            self.atlas_lines
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

    def summary(self, return_string: bool = False):
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

            output += (
                f"--> Coefficient of {i + 1}{ordinal} order: "
                + f"{self.fit_coeff[i]}{os.linesep}"
            )

        output += "RMS of the best fit solution: {self.rms}{os.linesep}"
        output += (
            "Percentage of peaks unsed for fitting: "
            + f"{self.peak_utilisation * 100.0:.2f}%{os.linesep}"
        )
        output += (
            "Percentage of atlas lines unsed for fitting: "
            + f"{self.atlas_utilisation * 100.0:.2f}%{os.linesep}"
        )

        output2 = ""
        output2_max_width = 0

        for _p, _a, _r in zip(
            self.matched_peaks, self.matched_atlas, self.residuals
        ):
            output2_tmp = (
                f"Peak {_p} (pix) is matched to wavelength {_a} A with a "
                + f"residual of {_r} A.{os.linesep}"
            )

            if len(output2_tmp) - 2 > output2_max_width:
                output2_max_width = len(output2_tmp) - 2

            output2 += output2_tmp

        output += "+" * output2_max_width + os.linesep
        output += output2

        self.logger.info(output)

        if return_string:
            return output

    def save_summary(self, filename: str = None):
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
            time_str = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
            filename = f"rascal_fit_summary_{time_str}.txt"

        summary = self.summary(return_string=True)

        with open(filename, "w+", encoding="ascii") as summary_file:
            summary_file.write(summary)

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

        for _i, (_p, _w) in enumerate(
            zip(self.matched_peaks, self.matched_atlas)
        ):
            pw_pairs.append((_i, _p, _w))
            self.logger.info(
                f"Position {_i}: pixel {_p} is matched to wavelength {_w}"
            )

        return pw_pairs

    def add_pix_wave_pair(self, pix: float, wave: float):
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

    def remove_pix_wave_pair(self, arg: int):
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
        self,
        matched_peaks: Union[list, np.ndarray] = None,
        matched_atlas: Union[list, np.ndarray] = None,
        degree: int = None,
        x0: Union[list, np.ndarray] = None,
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

        _x = np.asarray(matched_peaks)
        _y = np.asarray(matched_atlas)

        assert len(_x) == len(_y)
        assert len(_x) > 0
        assert degree > 0
        assert degree <= len(_x) - 1

        # Run robust fitting again
        fit_coeff_new = models.robust_polyfit(_x, _y, degree, x0)
        self.logger.info(f"Input fit_coeff is {x0}.")
        self.logger.info(f"Refit fit_coeff is {fit_coeff_new}.")

        self.fit_coeff = fit_coeff_new
        self.matched_peaks = copy.deepcopy(matched_peaks)
        self.matched_atlas = copy.deepcopy(matched_atlas)
        self.residuals = _y - self.polyval(_x, fit_coeff_new)
        self.rms = np.sqrt(
            np.nansum(self.residuals**2.0) / len(self.residuals)
        )

        self.peak_utilisation = len(self.matched_peaks) / len(self.peaks)
        self.atlas_utilisation = len(self.matched_atlas) / len(
            self.atlas_lines
        )
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

    def save_matches(self, filename: str = None, file_format: str = "csv"):
        """
        Export the matched peak-atlas pairs

        parameters
        ----------
        filename: str (Default: None)
            Export file name, if None, it will be saved as "matched_peaks"
        file_format: str (Default: csv)
            Export format, choose from csv and npy.

        """

        output = np.column_stack((self.matched_peaks, self.matched_atlas))

        if filename is None:
            filename = "matched_peaks"

        if file_format.lower() == "csv":
            np.savetxt(filename + ".csv", X=output, delimiter=",")

        elif file_format.lower() == "npy":
            np.save(file=filename + ".npy", arr=output)

    def plot_arc(
        self,
        log_spectrum: bool = False,
        save_fig: bool = False,
        fig_type: str = "png",
        filename: str = None,
        return_jsonstring: bool = False,
        renderer: str = "default",
        display: bool = True,
    ):
        """
        Plots the 1D spectrum of the extracted arc.

        parameters
        ----------
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
            contiguous_pixel=self.contiguous_pixel,
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
        fit_coeff: Union[list, np.ndarray] = None,
        top_n_candidate: int = 3,
        weighted: bool = True,
        save_fig: bool = False,
        fig_type: str = "png",
        filename: str = None,
        return_jsonstring: bool = False,
        renderer: str = "default",
        display: bool = True,
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
        fit_coeff: Union[list, np.ndarray] = None,
        spectrum: Union[list, np.ndarray] = None,
        plot_atlas: bool = True,
        log_spectrum: bool = False,
        save_fig: bool = False,
        fig_type: str = "png",
        filename: str = None,
        return_jsonstring: bool = False,
        renderer: str = "default",
        display: bool = True,
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
            fit_coeff = self.res["fit_coeff"]

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
