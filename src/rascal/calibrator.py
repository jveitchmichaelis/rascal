import copy
import itertools
import logging

import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
from scipy import interpolate
from tqdm.autonotebook import tqdm

from .util import _derivative
from .util import gauss

from . import plotting
from . import models
from .houghtransform import HoughTransform
from .atlas import Atlas


class Calibrator:
    def __init__(self, peaks, spectrum=None):
        """
        Initialise the calibrator object.

        Parameters
        ----------
        peaks: list
            List of identified arc line pixel values.
        spectrum: list
            The spectral intensity as a function of pixel.

        """

        self.logger = None
        self.log_level = None

        self.peaks = copy.deepcopy(peaks)
        self.spectrum = copy.deepcopy(spectrum)
        self.matplotlib_imported = False
        self.plotly_imported = False
        self.plot_with_matplotlib = False
        self.plot_with_plotly = False
        self.atlas = None
        self.pix_known = None
        self.wave_known = None
        self.hough_lines = None
        self.hough_points = None
        self.ht = HoughTransform()

        # calibrator_properties
        self.num_pix = None
        self.pixel_list = None
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
        self.fit_coeff = None

        self.set_calibrator_properties()
        self.set_hough_properties()
        self.set_ransac_properties()

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
                        self.pixel_list.max(),
                        self.max_wavelength
                        - self.range_tolerance
                        - self.candidate_tolerance,
                    ),
                    (
                        self.pixel_list.max(),
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

        Note: depending on the candidate_tolerance , one peak may match with
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

    def _match_bijective(self, candidates, peaks, fit_coeff):
        """

        Internal function used to return a list of inliers with a
        one-to-one relationship between peaks and wavelengths. This
        is critical as often we have several potential candidate lines
        for each peak. This function first iterates through each peak
        and selects the wavelength with the smallest error. It then
        iterates through this list and does the same for duplicate
        wavelengths.

        parameters
        ----------
        candidates: dict
            match candidates, internal to ransac

        peaks: list
            list of peaks [px]

        fit_coeff: list
            polynomial fit coefficients

        """

        err = []
        matched_x = []
        matched_y = []

        for peak in peaks:

            fit = self.polyval(peak, fit_coeff)

            # Get closest match for this peak
            errs = np.abs(fit - candidates[peak])
            idx = np.argmin(errs)

            err.append(errs[idx])
            matched_x.append(peak)
            matched_y.append(candidates[peak][idx])

        err = np.array(err)
        matched_x = np.array(matched_x)
        matched_y = np.array(matched_y)

        # Now we also need to resolve duplicate y's
        filtered_x = []
        filtered_y = []
        filtered_err = []

        for wavelength in np.unique(matched_y):

            mask = matched_y == wavelength
            filtered_y.append(wavelength)

            err_idx = np.argmin(err[mask])
            filtered_x.append(matched_x[mask][err_idx])
            filtered_err.append(err[mask][err_idx])

        # overwrite
        err = np.array(filtered_err)
        matched_x = np.array(filtered_x)
        matched_y = np.array(filtered_y)

        assert len(np.unique(matched_x)) == len(np.unique(matched_y))

        return err, matched_x, matched_y

    def _solve_candidate_ransac(
        self,
        fit_deg,
        fit_coeff,
        max_tries,
        candidate_tolerance,
        brute_force,
        progress,
    ):
        """
        Use RANSAC to sample the parameter space and give best guess

        Parameters
        ----------
        fit_deg: int
            The order of polynomial.
        fit_coeff: None or 1D numpy array
            Initial polynomial fit fit_coefficients.
        max_tries: int
            Number of trials of polynomial fitting.
        candidate_tolerance: float
            toleranceold  (Angstroms) for considering a point to be an inlier
            during candidate peak/line selection. This should be reasonable
            small as we want to search for candidate points which are
            *locally* linear.
        brute_force: boolean
            Solve all pixel-wavelength combinations with set to True.
        progress: boolean
            Show the progress bar with tdqm if set to True.

        Returns
        -------
        best_p: list
            A list of size fit_deg of the best fit polynomial
            fit_coefficient.
        best_err: float
            Arithmetic mean of the residuals.
        sum(best_inliers): int
            Number of lines fitted within the ransac_tolerance.
        valid_solution: boolean
            False if overfitted.

        """

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

        self.fit_deg = fit_deg

        valid_solution = False
        best_p = None
        best_cost = 1e50
        best_err = 1e50
        best_mask = [False]
        best_residual = None
        best_inliers = 0

        # Note that there may be multiple matches for
        # each peak, that is len(x) > len(np.unique(x))
        x = np.array(self.candidate_peak)
        y = np.array(self.candidate_arc)

        # Filter close wavelengths
        if self.filter_close:

            unique_y = np.unique(y)
            idx = np.argwhere(
                unique_y[1:] - unique_y[0:-1] < 3 * self.ransac_tolerance
            )
            separation_mask = np.argwhere((y == unique_y[idx]).sum(0) == 0)
            y = y[separation_mask].flatten()
            x = x[separation_mask].flatten()

        # If the number of lines is smaller than the number of degree of
        # polynomial fit, return failed fit.
        if len(np.unique(x)) <= self.fit_deg:

            return (best_p, best_err, sum(best_mask), 0, False)

        # Brute force check all combinations. If the request sample_size is
        # the same or larger than the available lines, it is essentially a
        # brute force.
        if brute_force or (self.sample_size >= len(np.unique(x))):

            idx = range(len(x))
            sampler = itertools.combinations(idx, self.sample_size)
            self.sample_size = len(np.unique(x))

        else:

            sampler = range(int(max_tries))

        if progress:

            sampler_list = tqdm(sampler)

        else:

            sampler_list = sampler

        peaks = np.sort(np.unique(x))
        idx = range(len(peaks))

        # Build a key(pixel)-value(wavelength) dictionary from the candidates
        candidates = {}

        for p in np.unique(x):

            candidates[p] = y[x == p]

        if self.ht.xedges is not None:

            xbin_size = (self.ht.xedges[1] - self.ht.xedges[0]) / 2.0
            ybin_size = (self.ht.yedges[1] - self.ht.yedges[0]) / 2.0

            if np.isfinite(self.hough_weight):

                twoditp = interpolate.RectBivariateSpline(
                    self.ht.xedges[1:] - xbin_size,
                    self.ht.yedges[1:] - ybin_size,
                    self.ht.hist,
                )

        else:

            twoditp = None

        # Calculate initial error given pre-existing fit
        if fit_coeff is not None:
            err, _, _ = self._match_bijective(candidates, peaks, fit_coeff)
            best_cost = sum(err)
            best_err = np.sqrt(np.mean(err**2.0))

        # The histogram is fixed, so pre-computed outside the loop
        if not brute_force:

            # weight the probability of choosing the sample by the inverse
            # line density
            h = np.histogram(peaks, bins=10)
            prob = 1.0 / h[0][np.digitize(peaks, h[1], right=True) - 1]
            prob = prob / np.sum(prob)

        for sample in sampler_list:

            keep_trying = True
            self.logger.debug(sample)

            while keep_trying:

                stop_n_candidateow = False

                if brute_force:

                    x_hat = x[[sample]]
                    y_hat = y[[sample]]

                else:

                    # Pick some random peaks
                    x_hat = np.random.choice(
                        peaks, self.sample_size, replace=False, p=prob
                    )
                    y_hat = []

                    # Pick a random wavelength for this x
                    for _x in x_hat:

                        y_choice = candidates[_x]

                        # Avoid picking a y that's already associated with
                        # another x
                        if not set(y_choice).issubset(set(y_hat)):

                            y_temp = np.random.choice(y_choice)

                            while y_temp in y_hat:

                                y_temp = np.random.choice(y_choice)

                            y_hat.append(y_temp)

                        else:

                            self.logger.debug(
                                "Not possible to draw a unique "
                                "set of atlas wavelengths."
                            )
                            stop_n_candidateow = True
                            break

                if stop_n_candidateow:

                    break

                # insert user given known pairs
                if self.pix_known is not None:

                    x_hat = np.concatenate((x_hat, self.pix_known))
                    y_hat = np.concatenate((y_hat, self.wave_known))

                # Try to fit the data.
                # This doesn't need to be robust, it's an exact fit.
                fit_coeffs = self.polyfit(x_hat, y_hat, self.fit_deg)

                # Check the intercept.
                if (fit_coeffs[0] < self.min_intercept) | (
                    fit_coeffs[0] > self.max_intercept
                ):

                    self.logger.debug("Intercept exceeds bounds.")
                    continue

                # Check monotonicity.
                pix_min = peaks[0] - np.ptp(peaks) * 0.2
                pix_max = peaks[-1] + np.ptp(peaks) * 0.2
                self.logger.debug((pix_min, pix_max))

                if not np.all(
                    np.diff(
                        self.polyval(
                            np.arange(pix_min, pix_max, 1), fit_coeffs
                        )
                    )
                    > 0
                ):

                    self.logger.debug(
                        "Solution is not monotonically increasing."
                    )
                    continue

                # Compute error and filter out many-to-one matches
                err, matched_x, matched_y = self._match_bijective(
                    candidates, peaks, fit_coeffs
                )

                if len(matched_x) == 0:
                    continue

                # M-SAC Estimator (Torr and Zisserman, 1996)
                err[err > self.ransac_tolerance] = self.ransac_tolerance

                # use the Hough space density as weights for the cost function
                wave = self.polyval(self.pixel_list, fit_coeffs)
                gradient = self.polyval(
                    self.pixel_list, _derivative(fit_coeffs)
                )
                intercept = wave - gradient * self.pixel_list

                # modified cost function weighted by the Hough space density
                if (self.hough_weight is not None) & (twoditp is not None):

                    weight = self.hough_weight * np.sum(
                        twoditp(intercept, gradient, grid=False)
                    )

                else:

                    weight = 1.0

                cost = (
                    sum(err)
                    / (len(err) - len(fit_coeffs) + 1)
                    / (weight + 1e-9)
                )

                # If this is potentially a new best fit, then handle that first
                if cost <= best_cost:

                    # reject lines outside the rms limit (ransac_tolerance)
                    # TODO: should n_inliers be recalculated from the robust
                    # fit?
                    mask = err < self.ransac_tolerance
                    n_inliers = sum(mask)
                    matched_peaks = matched_x[mask]
                    matched_atlas = matched_y[mask]

                    if len(matched_peaks) <= self.fit_deg:

                        self.logger.debug(
                            "Too few good candidates for fitting."
                        )
                        continue

                    # Now we do a robust fit
                    try:

                        coeffs = models.robust_polyfit(
                            matched_peaks, matched_atlas, self.fit_deg
                        )

                    except np.linalg.LinAlgError:

                        self.logger.warning(
                            "Linear algebra error in robust fit"
                        )
                        continue

                    # Get the residual of the fit
                    residual = (
                        self.polyval(matched_peaks, coeffs) - matched_atlas
                    )
                    residual[
                        np.abs(residual) > self.ransac_tolerance
                    ] = self.ransac_tolerance

                    rms_residual = np.sqrt(np.mean(residual**2))

                    # Make sure that we don't accept fits with zero error
                    if rms_residual < self.minimum_fit_error:

                        self.logger.debug(
                            "Fit error too small, " "{:1.2f}.".format(best_err)
                        )

                        continue

                    # Check that we have enough inliers based on user specified
                    # constraints

                    if n_inliers < self.minimum_matches:

                        self.logger.debug(
                            "Not enough matched peaks for valid solution, "
                            "user specified {}.".format(self.minimum_matches)
                        )
                        continue

                    if n_inliers < self.minimum_peak_utilisation * len(
                        self.peaks
                    ):

                        self.logger.debug(
                            "Not enough matched peaks for valid solution, "
                            "user specified {:1.2f} %.".format(
                                100 * self.minimum_matches
                            )
                        )
                        continue

                    # If the best fit is accepted, update the lists
                    best_cost = cost
                    best_inliers = n_inliers
                    best_p = coeffs
                    best_err = rms_residual
                    best_residual = residual
                    self.matched_peaks = list(copy.deepcopy(matched_peaks))
                    self.matched_atlas = list(copy.deepcopy(matched_atlas))

                    # Sanity check that matching peaks/atlas lines are 1:1
                    assert len(np.unique(self.matched_peaks)) == len(
                        self.matched_peaks
                    )
                    assert len(np.unique(self.matched_atlas)) == len(
                        self.matched_atlas
                    )
                    assert len(np.unique(self.matched_atlas)) == len(
                        np.unique(self.matched_peaks)
                    )

                    if progress:

                        sampler_list.set_description(
                            "Most inliers: {:d}, "
                            "best error: {:1.4f}".format(
                                best_inliers, best_err
                            )
                        )

                    # Break early if all peaks are matched
                    if best_inliers == len(peaks):
                        break

                # If we got this far, then we can continue to the next sample
                keep_trying = False

        # Overfit check
        if best_inliers <= self.fit_deg + 1:

            valid_solution = False

        else:

            valid_solution = True

        # If we totally failed then this can be empty
        assert best_inliers == len(self.matched_peaks)
        assert best_inliers == len(self.matched_atlas)

        assert len(self.matched_atlas) == len(set(self.matched_atlas))

        self.logger.info("Found: {}".format(best_inliers))

        return best_p, best_err, best_residual, best_inliers, valid_solution

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
            np.diff(self.polyval(np.sort(self.pixel_list), fit_new)) > 0
        ):

            self.logger.info("not monotonic")
            return np.inf

        lsq = (
            np.sum((y_matched - self.polyval(x_matched, fit_new)) ** 2.0) / dof
        )

        return lsq

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
        pixel_list=None,
        plotting_library=None,
        seed=None,
        logger_name="Calibrator",
        log_level="warning",
    ):
        """
        Initialise the calibrator object.

        Parameters
        ----------
        num_pix: int
            Number of pixels in the spectral axis.
        pixel_list: list
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

        # set the num_pix
        if num_pix is not None:

            self.num_pix = num_pix

        elif self.num_pix is None:

            try:

                self.num_pix = len(self.spectrum)

            except Exception as e:

                self.logger.warning(e)
                self.logger.warning(
                    "Neither num_pix nor spectrum is given, "
                    "it uses 1.1 times max(peaks) as the "
                    "maximum pixel value."
                )
                self.num_pix = 1.1 * max(self.peaks)

        else:

            pass

        self.logger.info("num_pix is set to {}.".format(num_pix))

        # set the pixel_list
        if pixel_list is not None:

            self.pixel_list = np.asarray(pixel_list)

        elif self.pixel_list is None:

            self.pixel_list = np.arange(self.num_pix)

        else:

            pass

        self.logger.info("pixel_list is set to {}.".format(pixel_list))

        # map the list position to the pixel value
        self.pix_to_rawpix = interpolate.interp1d(
            self.pixel_list,
            np.arange(len(self.pixel_list)),
            fill_value="extrapolate",
        )

        if seed is not None:
            np.random.seed(seed)

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
        num_slopes: int (default: 1000)
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
        ) / np.ptp(self.pixel_list)

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
        ) / np.ptp(self.pixel_list)

        if self.atlas is not None:

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
        ransac_tolerance: float (default: 1)
            The distance criteria  (Angstroms) to be considered an inlier to a
            fit. This should be close to the size of the expected residuals on
            the final fit (e.g. 1A is typical)
        candidate_weighted: boolean (default: True)
            Set to True to down-weight pairs that are far from the fit.
        hough_weight: float or None (default: 1.0)
            Set to use the hough space to weigh the fit. The theoretical
            optimal weighting is unclear. The larger the value, the heavily it
            relies on the overdensity in the hough space for a good fit.
        minimum_matches: int or None (default: 0)
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

            self.minimum_matches = 0

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

    def add_atlas(
        self,
        elements,
        min_atlas_wavelength=None,
        max_atlas_wavelength=None,
        min_intensity=10.0,
        min_distance=10.0,
        candidate_tolerance=10.0,
        constrain_poly=False,
        vacuum=False,
        pressure=101325.0,
        temperature=273.15,
        relative_humidity=0.0,
    ):

        self.logger.warning(
            "Using add_atlas is now deprecated. "
            "Please use the new Atlas class."
        )

        if min_atlas_wavelength is None:

            min_atlas_wavelength = self.min_wavelength - self.range_tolerance

        if max_atlas_wavelength is None:

            max_atlas_wavelength = self.max_wavelength + self.range_tolerance

        if self.atlas is None:

            new_atlas = Atlas(
                elements,
                min_atlas_wavelength=min_atlas_wavelength,
                max_atlas_wavelength=max_atlas_wavelength,
                min_intensity=min_intensity,
                min_distance=min_distance,
                range_tolerance=self.range_tolerance,
                vacuum=vacuum,
                pressure=pressure,
                temperature=temperature,
                relative_humidity=relative_humidity,
            )
            self.atlas = new_atlas

        else:

            self.atlas.add(
                elements,
                min_atlas_wavelength=min_atlas_wavelength,
                max_atlas_wavelength=max_atlas_wavelength,
                min_intensity=min_intensity,
                min_distance=min_distance,
                vacuum=vacuum,
                pressure=pressure,
                temperature=temperature,
                relative_humidity=relative_humidity,
            )

        self.candidate_tolerance = candidate_tolerance
        self.constrain_poly = constrain_poly

        self._generate_pairs()

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

    def add_user_atlas(
        self,
        elements,
        wavelengths,
        intensities=None,
        vacuum=False,
        pressure=101325.0,
        temperature=273.15,
        relative_humidity=0.0,
        candidate_tolerance=10,
        constrain_poly=False,
    ):

        self.logger.warning(
            "Using add_user_atlas is now deprecated. "
            "Please use the new Atlas class."
        )

        if self.atlas is None:

            self.atlas = Atlas()

        self.atlas.add_user_atlas(
            elements,
            wavelengths,
            intensities,
            vacuum,
            pressure,
            temperature,
            relative_humidity,
        )

        self.candidate_tolerance = candidate_tolerance
        self.constrain_poly = constrain_poly

        self._generate_pairs()

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
        self._generate_pairs()

    def do_hough_transform(self, brute_force=False):

        if self.pairs == []:

            logging.warning("pairs list is empty. Try generating now.")
            self._generate_pairs()

            if self.pairs == []:

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
        candidate_tolerance=2.0,
        brute_force=False,
        progress=True,
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

        # TODO also check whether minimum peak utilisation is greater than
        # minimum matches.

        (
            fit_coeff,
            rms,
            residual,
            n_inliers,
            valid,
        ) = self._solve_candidate_ransac(
            fit_deg=self.fit_deg,
            fit_coeff=self.fit_coeff,
            max_tries=self.max_tries,
            candidate_tolerance=candidate_tolerance,
            brute_force=self.brute_force,
            progress=self.progress,
        )

        peak_utilisation = n_inliers / len(self.peaks)
        atlas_utilisation = n_inliers / len(self.atlas)

        if not valid:

            self.logger.warning("Invalid fit")

        if rms > self.fit_tolerance:

            self.logger.warning(
                "RMS too large {} > {}".format(rms, self.fit_tolerance)
            )

        assert fit_coeff is not None, "Couldn't fit"

        self.fit_coeff = fit_coeff
        self.rms = rms
        self.residual = residual
        self.peak_utilisation = peak_utilisation
        self.atlas_utilisation = atlas_utilisation

        return (
            self.fit_coeff,
            self.matched_peaks,
            self.matched_atlas,
            self.rms,
            self.residual,
            self.peak_utilisation,
            self.atlas_utilisation,
        )

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
        given polynomial solution for a fit using maximal information, with
        the degree of polynomial = fit_deg.

        Set both refine and robust_refit to False will return the list of
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
            Set to True to fit all the detected peaks with the given polynomial
            solution.
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

            fit_coeff = self.fit_coeff.copy()

        if fit_deg is None:

            fit_deg = len(fit_coeff) - 1

        if refine:

            fit_coeff_new = fit_coeff.copy()

            if n_delta is None:

                n_delta = len(fit_coeff_new) - 1

            # fit everything
            fitted_delta = minimize(
                self._adjust_polyfit,
                fit_coeff_new[: int(n_delta)] * 1e-3,
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
                return fit_coeff, None, None, None, None, None, None

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

        # Create permutations:
        candidates = [[]]

        # match is a list
        for match in matched_atlas:

            if len(match) == 0:

                continue

            self.logger.info("matched: {}".format(match))

            new_candidates = []
            # i is an int
            # candidates is a list of list

            for i in range(len(candidates)):

                # c is a list
                c = candidates[i]

                if len(match) == 1:

                    c.extend(match)

                else:

                    # rep is a list of tuple
                    rep = ~np.in1d(match, c)

                    if rep.any():

                        for j in np.argwhere(rep):

                            new_c = c + [match[j]]
                            new_candidates.append(new_c)

                # Only add if new_candidates is not an empty list
                if new_candidates != []:

                    if candidates[0] == []:

                        candidates[0] = new_candidates

                    else:

                        candidates.append(new_candidates)

        if len(candidates) > 1:

            self.logger.info(
                "More than one match solution found, checking permutations."
            )

        self.matched_peaks = np.array(copy.deepcopy(matched_peaks))

        # Check all candidates
        best_err = 1e9
        self.matched_atlas = None
        self.residuals = None

        for candidate in candidates:

            matched_atlas = np.array(candidate)

            fit_coeff = self.polyfit(matched_peaks, matched_atlas, fit_deg)

            x = self.polyval(matched_peaks, fit_coeff)
            residuals = np.abs(matched_atlas - x)
            err = np.sum(residuals)

            if err < best_err:

                self.matched_atlas = matched_atlas
                self.residuals = residuals

                best_err = err

        assert self.matched_atlas is not None
        assert self.residuals is not None

        self.rms = np.sqrt(
            np.nansum(self.residuals**2.0) / len(self.residuals)
        )

        self.peak_utilisation = len(self.matched_peaks) / len(self.peaks)
        self.atlas_utilisation = len(self.matched_atlas) / len(self.atlas)

        if robust_refit:

            self.fit_coeff = models.robust_polyfit(
                np.asarray(self.matched_peaks),
                np.asarray(self.matched_atlas),
                fit_deg,
            )

            if np.any(np.isnan(self.fit_coeff)):

                self.logger.warning(
                    "robust_polyfit() returns None. "
                    "Input solution is returned."
                )
                return (
                    fit_coeff,
                    self.matched_peaks,
                    self.matched_atlas,
                    self.rms,
                    self.residuals,
                    self.peak_utilisation,
                    self.atlas_utilisation,
                )

            else:

                self.residuals = self.matched_atlas - self.polyval(
                    self.matched_peaks, self.fit_coeff
                )
                self.rms = np.sqrt(
                    np.nansum(self.residuals**2.0) / len(self.residuals)
                )

        else:

            self.fit_coeff = fit_coeff_new

        return (
            self.fit_coeff,
            self.matched_peaks,
            self.matched_atlas,
            self.rms,
            self.residuals,
            self.peak_utilisation,
            self.atlas_utilisation,
        )

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

        return (
            self.fit_coeff,
            self.matched_peaks,
            self.matched_atlas,
            self.rms,
            self.residuals,
        )

    def plot_arc(
        self,
        pixel_list=None,
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
        pixel_list: array (default: None)
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
            pixel_list=pixel_list,
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
        tolerance=5.0,
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
        tolerance: float (default: 5)
            Absolute difference between model and fitted wavelengths in unit
            of angstrom.
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
            tolerance=tolerance,
            plot_atlas=plot_atlas,
            log_spectrum=log_spectrum,
            save_fig=save_fig,
            fig_type=fig_type,
            filename=filename,
            return_jsonstring=return_jsonstring,
            renderer=renderer,
            display=display,
        )
