import itertools
import logging

import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
from scipy import interpolate
from tqdm.autonotebook import tqdm

from .util import load_calibration_lines
from .util import _derivative
from .util import gauss
from .util import vacuum_to_air_wavelength
from . import models
from .houghtransform import HoughTransform


class Calibrator:
    def __init__(self, peaks, spectrum=None):
        '''
        Initialise the calibrator object.

        Parameters
        ----------
        peaks: list
            List of identified arc line pixel values.
        spectrum: list
            The spectral intensity as a function of pixel.

        '''

        self.logger = None
        self.log_level = None

        self.peaks = peaks
        self.spectrum = spectrum
        self.matplotlib_imported = False
        self.plotly_imported = False
        self.plot_with_matplotlib = False
        self.plot_with_plotly = False
        self.atlas_elements = []
        self.atlas = []
        self.atlas_intensities = []
        self.pix_known = None
        self.wave_known = None
        self.hough_lines = None
        self.hough_points = None
        self.ht = HoughTransform()

        # calibrator_properties
        self.num_pix = None
        self.pixel_list = None
        self.plotting_library = None

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
        self.matched_peaks = None
        self.matched_atlas = None
        self.fit_coeff = None

        self.set_calibrator_properties()
        self.set_hough_properties()
        self.set_ransac_properties()

    def _import_matplotlib(self):
        '''
        Call to import matplotlib.

        '''

        try:

            global plt
            import matplotlib.pyplot as plt
            self.matplotlib_imported = True

        except ImportError:

            self.logger.error('matplotlib package not available.')

    def _import_plotly(self):
        '''
        Call to import plotly.

        '''

        try:

            global go
            global pio
            global psp
            import plotly.graph_objects as go
            import plotly.io as pio
            import plotly.subplots as psp

            self.plotly_imported = True
            pio.templates["CN"] = go.layout.Template(layout_colorway=[
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ])

            # setting Google color palette as default
            pio.templates.default = "CN"

        except ImportError:

            self.logger.error('plotly package not available.')

    def _generate_pairs(self, candidate_tolerance, constrain_poly):
        '''
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

        '''

        pairs = [pair for pair in itertools.product(self.peaks, self.atlas)]

        if constrain_poly:

            # Remove pairs outside polygon
            valid_area = Delaunay([
                (0, self.max_intercept + candidate_tolerance),
                (0, self.min_intercept - candidate_tolerance),
                (self.pixel_list.max(), self.max_wavelength -
                 self.range_tolerance - candidate_tolerance),
                (self.pixel_list.max(), self.max_wavelength +
                 self.range_tolerance + candidate_tolerance)
            ])

            mask = (valid_area.find_simplex(pairs) >= 0)
            self.pairs = np.array(pairs)[mask]

        else:

            self.pairs = np.array(pairs)

    def _merge_candidates(self, candidates):
        '''
        Merge two candidate lists.

        Parameters
        ----------
        candidates: list
            list containing pixel-wavelength pairs.

        '''

        merged = []

        for pairs in candidates:

            for pair in np.array(pairs).T:

                merged.append(pair)

        return np.sort(np.array(merged))

    def _get_most_common_candidates(self, candidates, top_n_candidate,
                                    weighted):
        '''
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

        '''

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
                    min(top_n_candidate, len(np.unique(wavelengths_matched))))

                unique_wavelengths = np.unique(wavelengths_matched)
                aggregated_count = np.zeros_like(unique_wavelengths)
                for j, w in enumerate(unique_wavelengths):

                    idx_j = np.where(wavelengths_matched == w)
                    aggregated_count[j] = np.sum(counts[idx_j])

                out_peaks.extend([peak] * n)
                out_wavelengths.extend(
                    wavelengths_matched[np.argsort(-aggregated_count)[:n]])

        return out_peaks, out_wavelengths

    def _get_candidate_points_linear(self, candidate_tolerance):
        '''
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

        '''

        # Locate candidate points for these lines fits
        self.candidates = []

        for line in self.hough_lines:

            gradient, intercept = line

            predicted = (gradient * self.pairs[:, 0] + intercept)
            actual = self.pairs[:, 1]
            diff = np.abs(predicted - actual)
            mask = (diff <= candidate_tolerance)

            # Match the range_tolerance to 1.1775 s.d. to match the FWHM
            # Note that the pairs outside of the range_tolerance were already
            # removed in an earlier stage
            weight = gauss(actual[mask], 1., predicted[mask],
                           (self.range_tolerance + self.linearity_tolerance) *
                           1.1775)

            self.candidates.append((self.pairs[:,
                                               0][mask], actual[mask], weight))

    def _get_candidate_points_poly(self, candidate_tolerance):
        '''
        **EXPERIMENTAL**

        Returns a list of peak/wavelengths pairs which agree with the fit

        (wavelength - gradient * x + intercept) < tolerance

        Note: depending on the toleranceold set, one peak may match with
        multiple wavelengths.

        Parameters
        ----------
        candidate_tolerance: float (default: 10)
            toleranceold  (Angstroms) for considering a point to be an inlier
            during candidate peak/line selection. This should be reasonable
            small as we want to search for candidate points which are
            *locally* linear.

        '''

        if self.fit_coeff is None:

            raise ValueError(
                'A guess solution for a polynomial fit has to '
                'be provided as fit_coeff in fit() in order to generate '
                'candidates for RANSAC sampling.')

        x_match = []
        y_match = []
        w_match = []
        self.candidates = []

        for p in self.peaks:

            x0 = self.polyval(p, self.fit_coeff)
            diff = np.abs(self.atlas - x0)

            x = np.array(self.atlas)[diff < candidate_tolerance]

            weight = gauss(x, 1., x0, self.range_tolerance)

            for y, w in zip(x, weight):

                x_match.append(p)
                y_match.append(y)
                w_match.append(w)

        x_match = np.array(x_match)
        y_match = np.array(y_match)
        w_match = np.array(w_match)

        self.candidates.append((x_match, y_match, w_match))

    def _match_bijective(self, candidates, peaks, fit_coeff):
        '''

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

        '''

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

        return err, matched_x, matched_y

    def _solve_candidate_ransac(self, fit_deg, fit_coeff, max_tries,
                                candidate_tolerance, brute_force, progress):
        '''
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

        '''

        if self.linear:

            self._get_candidate_points_linear(candidate_tolerance)

        else:

            self._get_candidate_points_poly(candidate_tolerance)

        self.candidate_peak, self.candidate_arc =\
            self._get_most_common_candidates(
               self.candidates,
               top_n_candidate=self.top_n_candidate,
               weighted=self.candidate_weighted)

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
                unique_y[1:] - unique_y[0:-1] < 3 * self.ransac_tolerance)
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

            xbin_size = (self.ht.xedges[1] - self.ht.xedges[0]) / 2.
            ybin_size = (self.ht.yedges[1] - self.ht.yedges[0]) / 2.

            if np.isfinite(self.hough_weight):

                twoditp = interpolate.RectBivariateSpline(
                    self.ht.xedges[1:] - xbin_size,
                    self.ht.yedges[1:] - ybin_size, self.ht.hist)

        else:

            twoditp = None

        # Calculate initial error given pre-existing fit
        if fit_coeff is not None:
            err, _, _ = self._match_bijective(candidates, peaks, fit_coeff)
            best_cost = sum(err)
            best_err = np.sqrt(np.mean(err**2.))

        # The histogram is fixed, so pre-computed outside the loop
        if not brute_force:

            # weight the probability of choosing the sample by the inverse
            # line density
            h = np.histogram(peaks, bins=10)
            prob = 1. / h[0][np.digitize(peaks, h[1], right=True) - 1]
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
                    x_hat = np.random.choice(peaks,
                                             self.sample_size,
                                             replace=False,
                                             p=prob)
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

                            self.logger.debug('Not possible to draw a unique '
                                              'set of atlas wavelengths.')
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
                if ((fit_coeffs[0] < self.min_intercept)
                        | (fit_coeffs[0] > self.max_intercept)):

                    self.logger.debug('Intercept exceeds bounds.')
                    continue

                # Check monotonicity.
                pix_min = peaks[0] - np.ptp(peaks) * 0.2
                pix_max = peaks[-1] + np.ptp(peaks) * 0.2
                self.logger.debug((pix_min, pix_max))

                if not np.all(
                        np.diff(
                            self.polyval(np.arange(pix_min, pix_max, 1),
                                         fit_coeffs)) > 0):

                    self.logger.debug(
                        'Solution is not monotonically increasing.')
                    continue

                # Compute error and filter out many-to-one matches
                err, matched_x, matched_y = self._match_bijective(
                    candidates, peaks, fit_coeffs)

                if len(matched_x) == 0:
                    continue

                # M-SAC Estimator (Torr and Zisserman, 1996)
                err[err > self.ransac_tolerance] = self.ransac_tolerance

                # use the Hough space density as weights for the cost function
                wave = self.polyval(self.pixel_list, fit_coeffs)
                gradient = self.polyval(self.pixel_list,
                                        _derivative(fit_coeffs))
                intercept = wave - gradient * self.pixel_list

                # modified cost function weighted by the Hough space density
                if (self.hough_weight is not None) & (twoditp is not None):

                    weight = self.hough_weight * np.sum(
                        twoditp(intercept, gradient, grid=False))

                else:

                    weight = 1.

                cost = sum(err) / (len(err) - len(fit_coeffs) + 1) / (weight +
                                                                      1e-9)

                # reject lines outside the rms limit (ransac_tolerance)
                best_mask = err < self.ransac_tolerance
                n_inliers = sum(best_mask)
                self.matched_peaks = matched_x[best_mask]
                self.matched_atlas = matched_y[best_mask]

                if len(self.matched_peaks) <= self.fit_deg:

                    self.logger.debug('Too few good candidates for fitting.')
                    continue

                # Want the most inliers with the lowest error
                if (cost <= best_cost):

                    # Now we do a robust fit
                    try:

                        best_p = models.robust_polyfit(self.matched_peaks,
                                                       self.matched_atlas,
                                                       self.fit_deg)

                    except np.linalg.LinAlgError:

                        self.logger.warning(
                            "Linear algebra error in robust fit")
                        continue

                    # Get the residual of the fit
                    err = self.polyval(self.matched_peaks,
                                       best_p) - self.matched_atlas
                    err[np.abs(err) >
                        self.ransac_tolerance] = self.ransac_tolerance

                    best_err = np.sqrt(np.mean(err**2))
                    best_residual = err
                    best_inliers = n_inliers

                    if best_inliers < self.minimum_matches:

                        self.logger.debug(
                            'Not enough matched peaks for valid solution, '
                            'user specified {}.'.format(self.minimum_matches))
                        continue

                    if best_inliers < self.minimum_peak_utilisation * len(
                            self.peaks):

                        self.logger.debug(
                            'Not enough matched peaks for valid solution, '
                            'user specified {:1.2f} %.'.format(
                                100 * self.minimum_matches))
                        continue

                    # Make sure that we don't accept fits with zero error
                    if best_err > self.minimum_fit_error:

                        best_cost = cost

                        if progress:

                            sampler_list.set_description(
                                'Most inliers: {:d}, '
                                'best error: {:1.4f}'.format(
                                    n_inliers, best_err))

                        if n_inliers == len(peaks):
                            """
                            all peaks matched
                            """
                            break

                keep_trying = False

        # Overfit check
        if best_inliers == self.fit_deg + 1:

            valid_solution = False

        else:

            valid_solution = True

        return best_p, best_err, best_residual, best_inliers, valid_solution

    def _adjust_polyfit(self, delta, fit, tolerance, min_frac):
        '''
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

        '''

        # x is wavelength
        # x_matched is pixel
        x_matched = []
        # y_matched is wavelength
        y_matched = []
        fit_new = fit.copy()

        for i, d in enumerate(delta):

            fit_new[i] += d

        for p in self.peaks:

            x = self.polyval(p, fit_new)
            diff = self.atlas - x
            diff_abs = np.abs(diff)
            idx = np.argmin(diff_abs)

            if diff_abs[idx] < tolerance:

                x_matched.append(p)
                y_matched.append(self.atlas[idx])

        x_matched = np.array(x_matched)
        y_matched = np.array(y_matched)

        dof = len(x_matched) - len(fit_new) - 1

        if dof < 1:

            return np.inf

        if len(x_matched) < len(self.peaks) * min_frac:

            return np.inf

        if not np.all(
                np.diff(self.polyval(np.sort(self.pixel_list), fit_new)) > 0):

            self.logger.info('not monotonic')
            return np.inf

        lsq = np.sum((y_matched - self.polyval(x_matched, fit_new))**2.) / dof

        return lsq

    def which_plotting_library(self):
        '''
        Call to show if the Calibrator is using matplotlib or plotly library
        (or neither).

        '''

        if self.plot_with_matplotlib:

            self.logger.info('Using matplotlib.')
            return 'matplotlib'

        elif self.plot_with_plotly:

            self.logger.info('Using plotly.')
            return 'plotly'

        else:

            self.logger.warning('Neither maplotlib nor plotly are imported.')
            return None

    def use_matplotlib(self):
        '''
        Call to switch to matplotlib.

        '''

        if not self.matplotlib_imported:

            self._import_matplotlib()

        self.plot_with_matplotlib = True
        self.plot_with_plotly = False

    def use_plotly(self):
        '''
        Call to switch to plotly.

        '''

        if not self.plotly_imported:

            self._import_plotly()

        self.plot_with_plotly = True
        self.plot_with_matplotlib = False

    def set_calibrator_properties(self,
                                  num_pix=None,
                                  pixel_list=None,
                                  plotting_library=None,
                                  seed=None,
                                  logger_name='Calibrator',
                                  log_level='warning'):
        '''
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

        '''

        # initialise the logger
        self.logger = logging.getLogger(logger_name)
        level = logging.getLevelName(log_level.upper())
        logging.basicConfig(level=level)
        self.log_level = level

        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] '
            '%(message)s',
            datefmt='%a, %d %b %Y %H:%M:%S')

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
                self.logger.warning('Neither num_pix nor spectrum is given, '
                                    'it uses 1.1 times max(peaks) as the '
                                    'maximum pixel value.')
                self.num_pix = 1.1 * max(self.peaks)

        else:

            pass

        self.logger.info('num_pix is set to {}.'.format(num_pix))

        # set the pixel_list
        if pixel_list is not None:

            self.pixel_list = np.asarray(pixel_list)

        elif self.pixel_list is None:

            self.pixel_list = np.arange(self.num_pix)

        else:

            pass

        self.logger.info('pixel_list is set to {}.'.format(pixel_list))

        # map the list position to the pixel value
        self.pix_to_rawpix = interpolate.interp1d(
            self.pixel_list, np.arange(len(self.pixel_list)))
        
        if seed is not None:
            np.random.seed(seed)

        # if the plotting library is supplied
        if plotting_library is not None:

            # set the plotting library
            self.plotting_library = plotting_library

        # if the plotting library is not supplied but the calibrator does not
        # know which library to use yet.
        elif self.plotting_library is None:

            self.plotting_library = 'matplotlib'

        # everything is good
        else:

            pass

        # check the choice of plotting library is available and used.
        if self.plotting_library == 'matplotlib':

            self.use_matplotlib()
            self.logger.info('Plotting with matplotlib.')

        elif self.plotting_library == 'plotly':

            self.use_plotly()
            self.logger.info('Plotting with plotly.')

        else:

            self.logger.warning(
                'Unknown plotting_library, please choose from '
                'matplotlib or plotly. Execute use_matplotlib() or '
                'use_plotly() to manually select the library.')

    def set_hough_properties(self,
                             num_slopes=None,
                             xbins=None,
                             ybins=None,
                             min_wavelength=None,
                             max_wavelength=None,
                             range_tolerance=None,
                             linearity_tolerance=None):
        '''
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

        '''

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

            self.min_wavelength = 3000

        else:

            pass

        # set the max_wavelength
        if max_wavelength is not None:

            self.max_wavelength = max_wavelength

        elif self.max_wavelength is None:

            self.max_wavelength = 9000

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

        self.min_slope = ((self.max_wavelength - self.range_tolerance -
                           self.linearity_tolerance) -
                          (self.min_intercept + self.range_tolerance +
                           self.linearity_tolerance)) / self.pixel_list.max()

        self.max_slope = ((self.max_wavelength + self.range_tolerance +
                           self.linearity_tolerance) -
                          (self.min_intercept - self.range_tolerance -
                           self.linearity_tolerance)) / self.pixel_list.max()

    def set_ransac_properties(self,
                              sample_size=None,
                              top_n_candidate=None,
                              linear=None,
                              filter_close=None,
                              ransac_tolerance=None,
                              candidate_weighted=None,
                              hough_weight=None,
                              minimum_matches=None,
                              minimum_peak_utilisation=None,
                              minimum_fit_error=None):
        '''
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

        '''

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

            assert (minimum_peak_utilisation >= 0
                    and minimum_peak_utilisation <= 1.0)
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

    def add_atlas(self,
                  elements,
                  min_atlas_wavelength=None,
                  max_atlas_wavelength=None,
                  min_intensity=10.,
                  min_distance=10.,
                  candidate_tolerance=10.,
                  constrain_poly=False,
                  vacuum=False,
                  pressure=101325.,
                  temperature=273.15,
                  relative_humidity=0.):
        '''
        Adds an atlas of arc lines to the calibrator, given an element.

        Arc lines are taken from a general list of NIST lines and can be
        filtered using the minimum relative intensity (note this may not be
        accurate due to instrumental effects such as detector response,
        dichroics, etc) and minimum line separation.

        Lines are filtered first by relative intensity, then by separation.
        This is to improve robustness in the case where there is a strong
        line very close to a weak line (which is within the separation limit).

        The vacuum to air wavelength conversion is deafult to False because
        observatories usually provide the line lists in the respective air
        wavelength, as the corrections from temperature and humidity are
        small. See https://emtoolbox.nist.gov/Wavelength/Documentation.asp

        Parameters
        ----------
        elements: string or list of strings
            Chemical symbol, case insensitive
        min_atlas_wavelength: float (default: None)
            Minimum wavelength of the arc lines.
        max_atlas_wavelength: float (default: None)
            Maximum wavelength of the arc lines.
        min_intensity: float (default: None)
            Minimum intensity of the arc lines. Refer to NIST for the
            intensity.
        min_distance: float (default: None)
            Minimum separation between neighbouring arc lines.
        candidate_tolerance: float (default: 10)
            toleranceold  (Angstroms) for considering a point to be an inlier
            during candidate peak/line selection. This should be reasonable
            small as we want to search for candidate points which are
            *locally* linear.
        constrain_poly: boolean
            Apply a polygonal constraint on possible peak/atlas pairs
        vacuum: boolean
            Set to True if the light path from the arc lamb to the detector
            plane is entirely in vacuum.
        pressure: float
            Pressure when the observation took place, in Pascal.
            If it is not known, we suggest you to assume 10% decrement per
            1000 meter altitude.
        temperature: float
            Temperature when the observation took place, in Kelvin.
        relative_humidity: float
            In percentage.

        '''

        if min_atlas_wavelength is None:

            min_atlas_wavelength = self.min_wavelength - self.range_tolerance

        if max_atlas_wavelength is None:

            max_atlas_wavelength = self.max_wavelength + self.range_tolerance

        if isinstance(elements, str):

            elements = [elements]

        for element in elements:

            atlas_elements_tmp, atlas_tmp, atlas_intensities_tmp =\
                load_calibration_lines(
                    element, min_atlas_wavelength, max_atlas_wavelength,
                    min_intensity, min_distance, vacuum, pressure, temperature,
                    relative_humidity)

            self.atlas_elements.extend(atlas_elements_tmp)
            self.atlas.extend(atlas_tmp)
            self.atlas_intensities.extend(atlas_intensities_tmp)

        # Create a list of all possible pairs of detected peaks and lines
        # from atlas
        self._generate_pairs(candidate_tolerance, constrain_poly)

    def add_user_atlas(self,
                       elements,
                       wavelengths,
                       intensities=None,
                       candidate_tolerance=10.,
                       constrain_poly=False,
                       vacuum=False,
                       pressure=101325.,
                       temperature=273.15,
                       relative_humidity=0.):
        '''
        Add a single or list of arc lines. Each arc line should have an
        element label associated with it. It is recommended that you use
        a standard periodic table abbreviation (e.g. 'Hg'), but it makes
        no difference to the fitting process.

        The vacuum to air wavelength conversion is deafult to False because
        observatories usually provide the line lists in the respective air
        wavelength, as the corrections from temperature and humidity are
        small. See https://emtoolbox.nist.gov/Wavelength/Documentation.asp

        Parameters
        ----------
        elements: list/str
            Elements (required). Preferably a standard (i.e. periodic table)
            name for convenience with built-in atlases
        wavelengths: list/float
            Wavelengths to add (Angstrom)
        intensities: list/float
            Relative line intensities (NIST value)
        candidate_tolerance: float (default: 15)
            toleranceold  (Angstroms) for considering a point to be an inlier
            during candidate peak/line selection. This should be reasonable
            small as we want to search for candidate points which are
            *locally* linear.
        constrain_poly: boolean
            Apply a polygonal constraint on possible peak/atlas pairs
        vacuum: boolean
            Set to true to convert the input wavelength to air-wavelengths
            based on the given pressure, temperature and humidity.
        pressure: float
            Pressure when the observation took place, in Pascal.
            If it is not known, assume 10% decrement per 1000 meter altitude
        temperature: float
            Temperature when the observation took place, in Kelvin.
        relative_humidity: float
            In percentage.

        '''

        if not isinstance(elements, list):

            elements = list(elements)

        if not isinstance(wavelengths, list):

            wavelengths = list(wavelengths)

        if intensities is None:

            intensities = [0] * len(wavelengths)

        else:

            if not isinstance(intensities, list):

                intensities = list(intensities)

        assert len(elements) == len(wavelengths), ValueError(
            'Input elements and wavelengths have different length.')
        assert len(elements) == len(intensities), ValueError(
            'Input elements and intensities have different length.')

        if vacuum:

            wavelengths = vacuum_to_air_wavelength(wavelengths, temperature,
                                                   pressure, relative_humidity)

        self.atlas_elements.extend(elements)
        self.atlas.extend(wavelengths)
        self.atlas_intensities.extend(intensities)

        # Create a list of all possible pairs of detected peaks and lines
        # from atlas
        self._generate_pairs(candidate_tolerance, constrain_poly)

    def remove_atlas_lines_range(self, wavelength, tolerance=10):
        '''
        Remove arc lines within a certain wavelength range.

        Parameters
        ----------
        wavelength: float
            Wavelength to remove (Angstrom)
        tolerance: float
            Tolerance around this wavelength where atlas lines will be removed

        '''

        for i, line in enumerate(self.atlas):

            if abs(line - wavelength) < tolerance:

                removed_element = self.atlas_elements.pop(i)
                removed_peak = self.atlas.pop(i)
                self.atlas_intensities.pop(i)

                self.logger.info('Removed {} line: {} A'.format(
                    removed_element, removed_peak))

    def list_atlas(self):
        '''
        List all the lines loaded to the Calibrator.

        '''

        for i in range(len(self.atlas)):

            print('Element ' + str(self.atlas_elements[i]) + ' at ' +
                  str(self.atlas[i]) + ' with intensity ' +
                  str(self.atlas_intensities[i]))

    def clear_atlas(self):
        '''
        Remove all the lines loaded to the Calibrator.

        '''

        self.atlas_elements = []
        self.atlas = []
        self.atlas_intensities = []

    def do_hough_transform(self, brute_force=False):

        # Generate the hough_points from the pairs
        self.ht.set_constraints(self.min_slope, self.max_slope,
                                self.min_intercept, self.max_intercept)
        if brute_force:
            self.ht.generate_hough_points_brute_force(self.pairs[:, 0],
                                                      self.pairs[:, 1])
        else:
            self.ht.generate_hough_points(self.pairs[:, 0],
                                          self.pairs[:, 1],
                                          num_slopes=self.num_slopes)

        self.ht.bin_hough_points(self.xbins, self.ybins)
        self.hough_points = self.ht.hough_points
        self.hough_lines = self.ht.hough_lines

    def save_hough_transform(self,
                             filename='hough_transform',
                             fileformat='npy',
                             delimiter='+',
                             to_disk=True):
        '''
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

        '''

        self.ht.save(filename=filename,
                     fileformat=fileformat,
                     delimiter=delimiter,
                     to_disk=to_disk)

    def load_hough_transform(self, filename='hough_transform', filetype='npy'):
        '''
        Store the binned Hough space and/or the raw Hough pairs.

        Parameters
        ----------
        filename: str (default: 'hough_transform')
            The filename of the output, not used if to_disk is False. It
            will be appended with the content type.
        filetype: str (default: 'npy')
            The file type of the saved hough transform. Choose from 'npy'
            and 'json'.

        '''

        self.ht.load(filename=filename, filetype=filetype)

    def set_known_pairs(self, pix=(), wave=()):
        '''
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

        '''

        pix = np.asarray(pix, dtype='float').reshape(-1)
        wave = np.asarray(wave, dtype='float').reshape(-1)

        assert pix.size == wave.size, ValueError(
            'Please check the length of the input arrays. pix has size {} '
            'and wave has size {}.'.format(pix.size, wave.size))

        if not all(
                isinstance(p, (float, int)) & (not np.isnan(p)) for p in pix):

            raise ValueError("All pix elements have to be numeric.")

        if not all(
                isinstance(w, (float, int)) & (not np.isnan(w)) for w in wave):

            raise ValueError("All wave elements have to be numeric.")

        self.pix_known = pix
        self.wave_known = wave

    def fit(self,
            max_tries=500,
            fit_deg=4,
            fit_coeff=None,
            fit_tolerance=5.,
            fit_type='poly',
            candidate_tolerance=2.,
            brute_force=False,
            progress=True):
        '''
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
        rms: float
            The root-mean-squared of the residuals
        residual: float
            Residual from the best fit
        peak_utilisation: float
            Fraction of detected peaks (pixel) used for calibration [0-1].
        atlas_utilisation: float
            Fraction of supplied arc lines (wavelength) used for
            calibration [0-1].

        '''

        self.max_tries = max_tries
        self.fit_deg = fit_deg
        self.fit_coeff = fit_coeff
        self.fit_tolerance = fit_tolerance
        self.fit_type = fit_type
        self.brute_force = brute_force
        self.progress = progress

        if self.fit_type == 'poly':

            self.polyfit = np.polynomial.polynomial.polyfit
            self.polyval = np.polynomial.polynomial.polyval

        elif self.fit_type == 'legendre':

            self.polyfit = np.polynomial.legendre.legfit
            self.polyval = np.polynomial.legendre.legval

        elif self.fit_type == 'chebyshev':

            self.polyfit = np.polynomial.chebyshev.chebfit
            self.polyval = np.polynomial.chebyshev.chebval

        else:

            raise ValueError(
                'fit_type must be: (1) poly, (2) legendre or (3) chebyshev')

        # Reduce sample_size if it is larger than the number of atlas available
        if self.sample_size > len(self.atlas):

            self.logger.warning(
                'Size of sample_size is larger than the size of atlas, ' +
                'the sample_size is set to match the size of atlas = ' +
                str(len(self.atlas)) + '.')
            self.sample_size = len(self.atlas)

        if self.sample_size <= fit_deg:

            self.sample_size = fit_deg + 1

        if (self.hough_lines is None) or (self.hough_points is None):

            self.do_hough_transform()

        if self.minimum_matches > len(self.atlas):
            self.logger.warning(
                'Requested minimum matches is greater than the atlas size' +
                'setting the minimum number of matches to equal the atlas' +
                'size = ' + str(len(self.atlas)) + '.')
            self.minimum_matches = len(self.atlas)

        # TODO also check whether minimum peak utilisation is greater than
        # minimum matches.

        fit_coeff, rms, residual, n_inliers, valid =\
            self._solve_candidate_ransac(
                fit_deg=self.fit_deg,
                fit_coeff=self.fit_coeff,
                max_tries=self.max_tries,
                candidate_tolerance=candidate_tolerance,
                brute_force=self.brute_force,
                progress=self.progress)

        peak_utilisation = n_inliers / len(self.peaks)
        atlas_utilisation = n_inliers / len(self.atlas)

        if not valid:

            self.logger.warning('Invalid fit')

        if rms > self.fit_tolerance:

            self.logger.warning('RMS too large {} > {}'.format(
                rms, self.fit_tolerance))

        assert (fit_coeff is not None), 'Couldn\'t fit'

        self.fit_coeff = fit_coeff
        self.rms = rms
        self.residual = residual
        self.peak_utilisation = peak_utilisation
        self.atlas_utilisation = atlas_utilisation

        return (self.fit_coeff, self.matched_peaks, self.matched_atlas,
                self.rms, self.residual, self.peak_utilisation,
                self.atlas_utilisation)

    def match_peaks(self,
                    fit_coeff=None,
                    n_delta=None,
                    refine=False,
                    tolerance=10.,
                    method='Nelder-Mead',
                    convergence=1e-6,
                    min_frac=0.5,
                    robust_refit=True,
                    fit_deg=None):
        '''
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

        '''

        if fit_coeff is None:

            fit_coeff = self.fit_coeff.copy

        if fit_deg is None:

            fit_deg = len(fit_coeff) - 1

        if refine:

            fit_coeff_new = fit_coeff.copy()

            if n_delta is None:

                n_delta = len(fit_coeff_new) - 1

            # fit everything
            fitted_delta = minimize(self._adjust_polyfit,
                                    fit_coeff_new[:int(n_delta)] * 1e-3,
                                    args=(fit_coeff, tolerance, min_frac),
                                    method=method,
                                    tol=convergence,
                                    options={
                                        'maxiter': 10000
                                    }).x

            for i, d in enumerate(fitted_delta):

                fit_coeff_new[i] += d

            if np.any(np.isnan(fit_coeff_new)):

                self.logger.warning('_adjust_polyfit() returns None. '
                                    'Input solution is returned.')
                return fit_coeff, None, None, None, None

        matched_peaks = []
        matched_atlas = []
        residuals = []

        for p in self.peaks:

            x = self.polyval(p, fit_coeff)
            diff = self.atlas - x
            diff_abs = np.abs(diff)
            idx = np.argmin(diff_abs)

            if diff_abs[idx] < tolerance:

                matched_peaks.append(p)
                matched_atlas.append(self.atlas[idx])
                residuals.append(diff[idx])

        matched_peaks = np.array(matched_peaks)
        matched_atlas = np.array(matched_atlas)
        self.residuals = np.array(residuals)
        self.rms = np.sqrt(np.nansum(self.residuals**2.) / len(self.residuals))

        self.peak_utilisation = len(matched_peaks) / len(self.peaks)
        self.atlas_utilisation = len(matched_peaks) / len(self.atlas)

        self.matched_peaks = matched_peaks
        self.matched_atlas = matched_atlas

        if robust_refit:

            self.fit_coeff = models.robust_polyfit(self.matched_peaks,
                                                   self.matched_atlas, fit_deg)

            if np.any(np.isnan(self.fit_coeff)):

                self.logger.warning('robust_polyfit() returns None. '
                                    'Input solution is returned.')
                return (fit_coeff, self.matched_peaks, self.matched_atlas,
                        self.rms, self.residuals, self.peak_utilisation,
                        self.atlas_utilisation)

            else:

                self.residuals = self.matched_atlas - self.polyval(
                    self.matched_peaks, self.fit_coeff)
                self.rms = np.sqrt(
                    np.nansum(self.residuals**2.) / len(self.residuals))

        else:

            self.fit_coeff = fit_coeff_new

        return (self.fit_coeff, self.matched_peaks, self.matched_atlas,
                self.rms, self.residuals, self.peak_utilisation,
                self.atlas_utilisation)

    def get_pix_wave_pairs(self):
        '''
        Return the list of matched_peaks and matched_atlas with their
        position in the array.

        Return
        ------
        pw_pairs: list
            List of tuples each containing the array position, peak (pixel)
            and atlas (wavelength).

        '''

        pw_pairs = []

        for i, (p, w) in enumerate(zip(self.matched_peaks,
                                       self.matched_atlas)):

            pw_pairs.append((i, p, w))
            self.logger.info(
                "Position {}: pixel {} is matched to wavelength {}".format(
                    i, p, w))

        return pw_pairs

    def add_pix_wave_pair(self, pix, wave):
        '''
        Adding extra pixel-wavelength pair to the Calibrator for refitting.
        This DOES NOT work before the Calibrator having fit for a solution
        yet: use set_known_pairs() for that purpose.

        Parameters
        ----------
        pix: float
            pixel position
        wave: float
            wavelength

        '''

        arg = np.argwhere(pix > self.matched_peaks)[0]

        # Only update the lists if both can be inserted
        matched_peaks = np.insert(self.matched_peaks, arg, pix)
        matched_atlas = np.insert(self.matched_atlas, arg, wave)

        self.matched_peaks = matched_peaks
        self.matched_atlas = matched_atlas

    def remove_pix_wave_pair(self, arg):
        '''
        Remove fitted pixel-wavelength pair from the Calibrator for refitting.
        The positions can be found from get_pix_wave_pairs(). One at a time.

        Parameters
        ----------
        arg: int
            The position of the pairs in the arrays.

        '''

        # Only update the lists if both can be deleted
        matched_peaks = np.delete(self.matched_peaks, arg)
        matched_atlas = np.delete(self.matched_atlas, arg)

        self.matched_peaks = matched_peaks
        self.matched_atlas = matched_atlas

    def manual_refit(self,
                     matched_peaks=None,
                     matched_atlas=None,
                     degree=None,
                     x0=None):
        '''
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

        '''

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
        self.matched_peaks = matched_peaks
        self.matched_atlas = matched_atlas
        self.residuals = y - self.polyval(x, fit_coeff_new)
        self.rms = np.sqrt(np.nansum(self.residuals**2.) / len(self.residuals))

        return (self.fit_coeff, self.matched_peaks, self.matched_atlas,
                self.rms, self.residuals)

    def plot_arc(self,
                 pixel_list=None,
                 log_spectrum=False,
                 save_fig=False,
                 fig_type='png',
                 filename=None,
                 return_jsonstring=False,
                 renderer='default',
                 display=True):
        '''
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

        '''

        if pixel_list is None:

            pixel_list = np.arange(len(self.spectrum))

        if self.plot_with_matplotlib:

            plt.figure(figsize=(18, 5))

            if self.spectrum is not None:
                if log_spectrum:
                    plt.plot(pixel_list,
                             np.log10(self.spectrum / self.spectrum.max()),
                             label='Arc Spectrum')
                    plt.vlines(self.peaks,
                               -2,
                               0,
                               label='Detected Peaks',
                               color='C1')
                    plt.ylabel("log(Normalised Count)")
                    plt.ylim(-2, 0)
                else:
                    plt.plot(pixel_list,
                             self.spectrum / self.spectrum.max(),
                             label='Arc Spectrum')
                    plt.ylabel("Normalised Count")
                    plt.vlines(self.peaks,
                               0,
                               1.05,
                               label='Detected Peaks',
                               color='C1')
                plt.title('Number of pixels: ' + str(self.spectrum.shape[0]))
                plt.xlim(0, self.spectrum.shape[0])
                plt.legend()

            else:

                plt.xlim(0, max(self.peaks))

            plt.xlabel("Pixel (Spectral Direction)")
            plt.grid()
            plt.tight_layout()

            if save_fig:

                fig_type = fig_type.split('+')

                if filename is None:

                    filename_output = 'rascal_arc'

                else:

                    filename_output = filename

                for t in fig_type:

                    if t in ['jpg', 'png', 'svg', 'pdf']:

                        plt.savefig(filename_output + '.' + t, format=t)

            if display:

                plt.show()

        if self.plot_with_plotly:

            fig = go.Figure()

            if log_spectrum:

                # Plot all-pairs
                fig.add_trace(
                    go.Scatter(
                        x=list(pixel_list),
                        y=list(np.log10(self.spectrum / self.spectrum.max())),
                        mode='lines',
                        name='Arc'))
                xmin = min(np.log10(self.spectrum / self.spectrum.max()))
                xmax = max(np.log10(self.spectrum / self.spectrum.max()))

            else:

                # Plot all-pairs
                fig.add_trace(
                    go.Scatter(x=list(pixel_list),
                               y=list(self.spectrum / self.spectrum.max()),
                               mode='lines',
                               name='Arc'))
                xmin = min(self.spectrum / self.spectrum.max())
                xmax = max(self.spectrum / self.spectrum.max())

            # Add vlines
            for i in self.peaks:
                fig.add_shape(type='line',
                              xref='x',
                              yref='y',
                              x0=i,
                              y0=0,
                              x1=i,
                              y1=1.05,
                              line=dict(
                                  color=pio.templates["CN"].layout.colorway[1],
                                  width=1))

            fig.update_layout(autosize=True,
                              yaxis=dict(title='Normalised Count',
                                         range=[xmin, xmax],
                                         showgrid=True),
                              xaxis=dict(
                                  title='Pixel',
                                  zeroline=False,
                                  range=[0., len(self.spectrum)],
                                  showgrid=True,
                              ),
                              hovermode='closest',
                              showlegend=True,
                              height=800,
                              width=1000)

            fig.update_xaxes(showline=True,
                             linewidth=1,
                             linecolor='black',
                             mirror=True)

            fig.update_yaxes(showline=True,
                             linewidth=1,
                             linecolor='black',
                             mirror=True)

            if save_fig:

                fig_type = fig_type.split('+')

                if filename is None:

                    filename_output = 'rascal_arc'

                else:

                    filename_output = filename

                for t in fig_type:

                    if t == 'iframe':

                        pio.write_html(fig, filename_output + '.' + t)

                    elif t in ['jpg', 'png', 'svg', 'pdf']:

                        pio.write_image(fig, filename_output + '.' + t)

            if display:

                if renderer == 'default':

                    fig.show()

                else:

                    fig.show(renderer)

            if return_jsonstring:

                return fig.to_json()

    def plot_search_space(self,
                          fit_coeff=None,
                          top_n_candidate=3,
                          weighted=True,
                          save_fig=False,
                          fig_type='png',
                          filename=None,
                          return_jsonstring=False,
                          renderer='default',
                          display=True):
        '''
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

        '''

        # Get top linear estimates and combine
        candidate_peak, candidate_arc = self._get_most_common_candidates(
            self.candidates,
            top_n_candidate=top_n_candidate,
            weighted=weighted)

        # Get the search space boundaries
        x = self.pixel_list

        m_1 = (self.max_wavelength -
               self.min_wavelength) / self.pixel_list.max()
        y_1 = m_1 * x + self.min_wavelength

        m_2 = (self.max_wavelength + self.range_tolerance -
               (self.min_wavelength +
                self.range_tolerance)) / self.pixel_list.max()
        y_2 = m_2 * x + self.min_wavelength + self.range_tolerance

        m_3 = (self.max_wavelength - self.range_tolerance -
               (self.min_wavelength -
                self.range_tolerance)) / self.pixel_list.max()
        y_3 = m_3 * x + (self.min_wavelength - self.range_tolerance)

        if self.plot_with_matplotlib:

            plt.figure(figsize=(10, 10))

            # Plot all-pairs
            plt.scatter(*self.pairs.T,
                        alpha=0.2,
                        color='C0',
                        label='All pairs')

            plt.scatter(self._merge_candidates(self.candidates)[:, 0],
                        self._merge_candidates(self.candidates)[:, 1],
                        alpha=0.2,
                        color='C1',
                        label='Candidate Pairs')

            # Tolerance region around the minimum wavelength
            plt.text(5, self.min_wavelength + 100,
                     'Min wavelength (user-supplied)')
            plt.hlines(self.min_wavelength,
                       0,
                       self.pixel_list.max(),
                       color='k')
            plt.hlines(self.min_wavelength + self.range_tolerance,
                       0,
                       self.pixel_list.max(),
                       linestyle='dashed',
                       alpha=0.5,
                       color='k')
            plt.hlines(self.min_wavelength - self.range_tolerance,
                       0,
                       self.pixel_list.max(),
                       linestyle='dashed',
                       alpha=0.5,
                       color='k')

            # Tolerance region around the maximum wavelength
            plt.text(5, self.max_wavelength + 100,
                     'Max wavelength (user-supplied)')
            plt.hlines(self.max_wavelength,
                       0,
                       self.pixel_list.max(),
                       color='k')
            plt.hlines(self.max_wavelength + self.range_tolerance,
                       0,
                       self.pixel_list.max(),
                       linestyle='dashed',
                       alpha=0.5,
                       color='k')
            plt.hlines(self.max_wavelength - self.range_tolerance,
                       0,
                       self.pixel_list.max(),
                       linestyle='dashed',
                       alpha=0.5,
                       color='k')

            # The line from (first pixel, minimum wavelength) to
            # (last pixel, maximum wavelength), and the two lines defining the
            # tolerance region.
            plt.plot(x, y_1, label='Linear Fit', color='C3')
            plt.plot(x,
                     y_2,
                     linestyle='dashed',
                     label='Tolerance Region',
                     color='C3')
            plt.plot(x, y_3, linestyle='dashed', color='C3')

            if fit_coeff is not None:

                plt.scatter(self.peaks,
                            self.polyval(self.peaks, fit_coeff),
                            color='C4',
                            label='Solution')

            plt.scatter(candidate_peak,
                        candidate_arc,
                        color='C2',
                        label='Best Candidate Pairs')

            plt.xlim(0, self.pixel_list.max())
            plt.ylim(self.min_wavelength - self.range_tolerance,
                     self.max_wavelength + self.range_tolerance)

            plt.xlabel('Pixel')
            plt.ylabel('Wavelength / A')
            plt.legend()
            plt.grid()
            plt.tight_layout()

            if save_fig:

                fig_type = fig_type.split('+')

                if filename is None:

                    filename_output = 'rascal_hough_search_space'

                else:

                    filename_output = filename

                for t in fig_type:

                    if t in ['jpg', 'png', 'svg', 'pdf']:

                        plt.savefig(filename_output + '.' + t, format=t)

            if display:

                plt.show()

        elif self.plot_with_plotly:

            fig = go.Figure()

            # Plot all-pairs
            fig.add_trace(
                go.Scatter(x=self.pairs[:, 0],
                           y=self.pairs[:, 1],
                           mode='markers',
                           name='All Pairs',
                           marker=dict(
                               color=pio.templates["CN"].layout.colorway[0],
                               opacity=0.2)))

            fig.add_trace(
                go.Scatter(x=self._merge_candidates(self.candidates)[:, 0],
                           y=self._merge_candidates(self.candidates)[:, 1],
                           mode='markers',
                           name='Candidate Pairs',
                           marker=dict(
                               color=pio.templates["CN"].layout.colorway[1],
                               opacity=0.2)))
            fig.add_trace(
                go.Scatter(
                    x=candidate_peak,
                    y=candidate_arc,
                    mode='markers',
                    name='Best Candidate Pairs',
                    marker=dict(color=pio.templates["CN"].layout.colorway[2])))

            # Tolerance region around the minimum wavelength
            fig.add_trace(
                go.Scatter(x=[0, self.pixel_list.max()],
                           y=[self.min_wavelength, self.min_wavelength],
                           name='Min/Maximum',
                           mode='lines',
                           line=dict(color='black')))
            fig.add_trace(
                go.Scatter(x=[0, self.pixel_list.max()],
                           y=[
                               self.min_wavelength + self.range_tolerance,
                               self.min_wavelength + self.range_tolerance
                           ],
                           name='Tolerance Range',
                           mode='lines',
                           line=dict(color='black', dash='dash')))
            fig.add_trace(
                go.Scatter(x=[0, self.pixel_list.max()],
                           y=[
                               self.min_wavelength - self.range_tolerance,
                               self.min_wavelength - self.range_tolerance
                           ],
                           showlegend=False,
                           mode='lines',
                           line=dict(color='black', dash='dash')))

            # Tolerance region around the minimum wavelength
            fig.add_trace(
                go.Scatter(x=[0, self.pixel_list.max()],
                           y=[self.max_wavelength, self.max_wavelength],
                           showlegend=False,
                           mode='lines',
                           line=dict(color='black')))
            fig.add_trace(
                go.Scatter(x=[0, self.pixel_list.max()],
                           y=[
                               self.max_wavelength + self.range_tolerance,
                               self.max_wavelength + self.range_tolerance
                           ],
                           showlegend=False,
                           mode='lines',
                           line=dict(color='black', dash='dash')))
            fig.add_trace(
                go.Scatter(x=[0, self.pixel_list.max()],
                           y=[
                               self.max_wavelength - self.range_tolerance,
                               self.max_wavelength - self.range_tolerance
                           ],
                           showlegend=False,
                           mode='lines',
                           line=dict(color='black', dash='dash')))

            # The line from (first pixel, minimum wavelength) to
            # (last pixel, maximum wavelength), and the two lines defining the
            # tolerance region.
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_1,
                    mode='lines',
                    name='Linear Fit',
                    line=dict(color=pio.templates["CN"].layout.colorway[3])))
            fig.add_trace(
                go.Scatter(x=x,
                           y=y_2,
                           mode='lines',
                           name='Tolerance Region',
                           line=dict(
                               color=pio.templates["CN"].layout.colorway[3],
                               dash='dashdot')))
            fig.add_trace(
                go.Scatter(x=x,
                           y=y_3,
                           showlegend=False,
                           mode='lines',
                           line=dict(
                               color=pio.templates["CN"].layout.colorway[3],
                               dash='dashdot')))

            if fit_coeff is not None:

                fig.add_trace(
                    go.Scatter(
                        x=self.peaks,
                        y=self.polyval(self.peaks, fit_coeff),
                        mode='markers',
                        name='Solution',
                        marker=dict(
                            color=pio.templates["CN"].layout.colorway[4])))

            # Layout, Title, Grid config
            fig.update_layout(
                autosize=True,
                yaxis=dict(
                    title='Wavelength / A',
                    range=[
                        self.min_wavelength - self.range_tolerance * 1.1,
                        self.max_wavelength + self.range_tolerance * 1.1
                    ],
                    showgrid=True),
                xaxis=dict(
                    title='Pixel',
                    zeroline=False,
                    range=[0., self.pixel_list.max()],
                    showgrid=True,
                ),
                hovermode='closest',
                showlegend=True,
                height=800,
                width=1000)

            fig.update_xaxes(showline=True,
                             linewidth=1,
                             linecolor='black',
                             mirror=True)
            fig.update_yaxes(showline=True,
                             linewidth=1,
                             linecolor='black',
                             mirror=True)

            if save_fig:

                fig_type = fig_type.split('+')

                if filename is None:

                    filename_output = 'rascal_hough_search_space'

                else:

                    filename_output = filename

                for t in fig_type:

                    if t == 'iframe':

                        pio.write_html(fig, filename_output + '.' + t)

                    elif t in ['jpg', 'png', 'svg', 'pdf']:

                        pio.write_image(fig, filename_output + '.' + t)

            if display:

                if renderer == 'default':

                    fig.show()

                else:

                    fig.show(renderer)

            if return_jsonstring:

                return fig.to_json()

    def plot_fit(self,
                 fit_coeff,
                 spectrum=None,
                 tolerance=5.,
                 plot_atlas=True,
                 log_spectrum=False,
                 save_fig=False,
                 fig_type='png',
                 filename=None,
                 return_jsonstring=False,
                 renderer='default',
                 display=True):
        '''
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

        '''

        if spectrum is None:

            try:

                spectrum = self.spectrum

            except Exception as e:

                self.logger.error(e)
                self.logger.error('Spectrum is not provided, it cannot be '
                                  'plotted.')

        if spectrum is not None:

            if log_spectrum:

                spectrum[spectrum < 0] = 1e-100
                spectrum = np.log10(spectrum)
                vline_max = np.nanmax(spectrum) * 2.0
                text_box_pos = 1.2 * max(spectrum)

            else:

                vline_max = np.nanmax(spectrum) * 1.2
                text_box_pos = 0.8 * max(spectrum)

        else:

            vline_max = 1.0
            text_box_pos = 0.5

        wave = self.polyval(self.pixel_list, fit_coeff)

        if self.plot_with_matplotlib:

            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3,
                                                sharex=True,
                                                gridspec_kw={'hspace': 0.},
                                                figsize=(15, 9))
            fig.tight_layout()

            # Plot fitted spectrum
            if spectrum is not None:

                ax1.plot(wave, spectrum, label='Arc Spectrum')
                ax1.vlines(self.polyval(self.peaks, fit_coeff),
                           np.array(spectrum)[self.pix_to_rawpix(
                               self.peaks).astype('int')],
                           vline_max,
                           linestyles='dashed',
                           colors='C1',
                           label='Detected Peaks')

            # Plot the atlas
            if plot_atlas:

                # spec = SyntheticSpectrum(
                #    fit, model_type='poly', degree=len(fit)-1)
                # x_locs = spec.get_pixels(self.atlas)
                ax1.vlines(self.atlas,
                           0,
                           vline_max,
                           colors='C2',
                           label='Given Lines')

            fitted_peaks = []
            fitted_diff = []
            all_diff = []

            first_one = True
            for p in self.peaks:

                x = self.polyval(p, fit_coeff)
                diff = self.atlas - x
                idx = np.argmin(np.abs(diff))
                all_diff.append(diff[idx])

                self.logger.info('Peak at: {} A'.format(x))

                if np.abs(diff[idx]) < tolerance:

                    fitted_peaks.append(p)
                    fitted_diff.append(diff[idx])
                    self.logger.info('- matched to {} A'.format(
                        self.atlas[idx]))

                    if spectrum is not None:

                        if first_one:
                            ax1.vlines(
                                self.polyval(p, fit_coeff),
                                spectrum[self.pix_to_rawpix(p).astype('int')],
                                vline_max,
                                colors='C1',
                                label='Fitted Peaks')
                            first_one = False

                        else:
                            ax1.vlines(
                                self.polyval(p, fit_coeff),
                                spectrum[self.pix_to_rawpix(p).astype('int')],
                                vline_max,
                                colors='C1')

                    ax1.text(x - 3,
                             text_box_pos,
                             s='{}:{:1.2f}'.format(self.atlas_elements[idx],
                                                   self.atlas[idx]),
                             rotation=90,
                             bbox=dict(facecolor='white', alpha=1))

            rms = np.sqrt(np.mean(np.array(fitted_diff)**2.))

            ax1.grid(linestyle=':')
            ax1.set_ylabel('Electron Count / e-')

            if spectrum is not None:

                if log_spectrum:

                    ax1.set_ylim(0, vline_max)

                else:

                    ax1.set_ylim(np.nanmin(spectrum), vline_max)

            ax1.legend(loc='center right')

            # Plot the residuals
            ax2.scatter(self.polyval(fitted_peaks, fit_coeff),
                        fitted_diff,
                        marker='+',
                        color='C1')
            ax2.hlines(0, wave.min(), wave.max(), linestyles='dashed')
            ax2.hlines(rms,
                       wave.min(),
                       wave.max(),
                       linestyles='dashed',
                       color='k',
                       label='RMS')
            ax2.hlines(-rms,
                       wave.min(),
                       wave.max(),
                       linestyles='dashed',
                       color='k')
            ax2.grid(linestyle=':')
            ax2.set_ylabel('Residual / A')
            ax2.legend()
            '''
            ax2.text(
                min(wave) + np.ptp(wave) * 0.05,
                max(spectrum),
                'RMS =' + str(rms)[:6]
                )
            '''

            # Plot the polynomial
            ax3.scatter(self.polyval(fitted_peaks, fit_coeff),
                        fitted_peaks,
                        marker='+',
                        color='C1',
                        label='Fitted Peaks')
            ax3.plot(wave, self.pixel_list, color='C2', label='Solution')
            ax3.grid(linestyle=':')
            ax3.set_xlabel('Wavelength / A')
            ax3.set_ylabel('Pixel')
            ax3.legend(loc='lower right')
            w_min = self.polyval(min(fitted_peaks), fit_coeff)
            w_max = self.polyval(max(fitted_peaks), fit_coeff)
            ax3.set_xlim(w_min * 0.95, w_max * 1.05)

            plt.tight_layout()

            if save_fig:

                fig_type = fig_type.split('+')

                if filename is None:

                    filename_output = 'rascal_solution'

                else:

                    filename_output = filename

                for t in fig_type:

                    if t in ['jpg', 'png', 'svg', 'pdf']:

                        plt.savefig(filename_output + '.' + t, format=t)

            if display:

                fig.show()

        elif self.plot_with_plotly:

            fig = psp.make_subplots(rows=3, cols=1, shared_xaxes=True)

            # Top plot - arc spectrum and matched peaks
            if spectrum is not None:
                fig.add_trace(go.Scatter(x=wave,
                                         y=spectrum,
                                         mode='lines',
                                         name='Arc Spectrum'),
                              row=3,
                              col=1)

                spec_max = np.nanmax(spectrum) * 1.05
            else:
                spec_max = 1.0

            fitted_peaks = []
            fitted_peaks_adu = []
            fitted_diff = []
            all_diff = []

            for p in self.peaks:

                x = self.polyval(p, fit_coeff)

                # Add vlines
                fig.add_shape(type='line',
                              row=3,
                              col=1,
                              x0=x,
                              y0=0,
                              x1=x,
                              y1=spec_max,
                              line=dict(
                                  color=pio.templates["CN"].layout.colorway[1],
                                  width=1))

                diff = self.atlas - x
                idx = np.argmin(np.abs(diff))
                all_diff.append(diff[idx])

                self.logger.info('Peak at: {} A'.format(x))

                if np.abs(diff[idx]) < tolerance:

                    fitted_peaks.append(p)
                    if spectrum is not None:
                        fitted_peaks_adu.append(spectrum[int(
                            self.pix_to_rawpix(p))])
                    fitted_diff.append(diff[idx])
                    self.logger.info('- matched to {} A'.format(
                        self.atlas[idx]))

            x_fitted = self.polyval(fitted_peaks, fit_coeff)

            fig.add_trace(go.Scatter(
                x=x_fitted,
                y=fitted_peaks_adu,
                mode='markers',
                marker=dict(color=pio.templates["CN"].layout.colorway[1]),
                showlegend=False),
                          row=3,
                          col=1)

            # Middle plot - Residual plot
            rms = np.sqrt(np.mean(np.array(fitted_diff)**2.))
            fig.add_trace(go.Scatter(
                x=x_fitted,
                y=fitted_diff,
                mode='markers',
                marker=dict(color=pio.templates["CN"].layout.colorway[1]),
                showlegend=False),
                          row=2,
                          col=1)
            fig.add_trace(go.Scatter(
                x=[
                    self.polyval(min(fitted_peaks), fit_coeff) * 0.95,
                    self.polyval(max(fitted_peaks), fit_coeff) * 1.05
                ],
                y=[0, 0],
                mode='lines',
                line=dict(color=pio.templates["CN"].layout.colorway[0],
                          dash='dash'),
                showlegend=False),
                          row=2,
                          col=1)
            fig.add_trace(go.Scatter(x=[
                self.polyval(min(fitted_peaks), fit_coeff) * 0.95,
                self.polyval(max(fitted_peaks), fit_coeff) * 1.05
            ],
                                     y=[rms, rms],
                                     mode='lines',
                                     line=dict(color='black', dash='dash'),
                                     showlegend=False),
                          row=2,
                          col=1)
            fig.add_trace(go.Scatter(x=[wave.min(), wave.max()],
                                     y=[-rms, -rms],
                                     mode='lines',
                                     line=dict(color='black', dash='dash'),
                                     name='RMS'),
                          row=2,
                          col=1)

            # Bottom plot - Polynomial fit for Pixel to Wavelength
            fig.add_trace(go.Scatter(
                x=x_fitted,
                y=fitted_peaks,
                mode='markers',
                marker=dict(color=pio.templates["CN"].layout.colorway[1]),
                name='Fitted Peaks'),
                          row=1,
                          col=1)
            fig.add_trace(go.Scatter(
                x=wave,
                y=self.pixel_list,
                mode='lines',
                line=dict(color=pio.templates["CN"].layout.colorway[2]),
                name='Solution'),
                          row=1,
                          col=1)

            # Layout, Title, Grid config
            if spectrum is not None:

                if log_spectrum:

                    fig.update_layout(
                        yaxis3=dict(title='Electron Count / e-',
                                    range=[
                                        np.log10(np.percentile(spectrum, 15)),
                                        np.log10(spec_max)
                                    ],
                                    domain=[0.666, 1.0],
                                    showgrid=True,
                                    type='log'))

                else:

                    fig.update_layout(yaxis3=dict(
                        title='Electron Count / e-',
                        range=[np.percentile(spectrum, 15), spec_max],
                        domain=[0.666, 1.0],
                        showgrid=True))

            fig.update_layout(autosize=True,
                              yaxis2=dict(
                                  title='Residual / A',
                                  range=[min(fitted_diff),
                                         max(fitted_diff)],
                                  domain=[0.333, 0.666],
                                  showgrid=True),
                              yaxis=dict(title='Pixel',
                                         range=[0., max(self.pixel_list)],
                                         domain=[0., 0.333],
                                         showgrid=True),
                              xaxis=dict(showticklabels=True),
                              xaxis2=dict(showticklabels=False),
                              xaxis3=dict(showticklabels=False),
                              hovermode='closest',
                              showlegend=True,
                              height=800,
                              width=1000)
            fig.update_yaxes(showline=True,
                             linewidth=1,
                             linecolor='black',
                             mirror=True)
            fig.update_xaxes(showline=True,
                             linewidth=1,
                             linecolor='black',
                             mirror=True)
            fig.update_xaxes(
                title='Wavelength / A',
                zeroline=False,
                range=[
                    self.polyval(min(fitted_peaks), fit_coeff) * 0.95,
                    self.polyval(max(fitted_peaks), fit_coeff) * 1.05
                ],
                showgrid=True,
                row=1,
                col=1)

            if save_fig:

                fig_type = fig_type.split('+')

                if filename is None:

                    filename_output = 'rascal_solution'

                else:

                    filename_output = filename

                for t in fig_type:

                    if t == 'iframe':

                        pio.write_html(fig, filename_output + '.' + t)

                    elif t in ['jpg', 'png', 'svg', 'pdf']:

                        pio.write_image(fig, filename_output + '.' + t)

            if display:

                if renderer == 'default':

                    fig.show()

                else:

                    fig.show(renderer)

            if return_jsonstring:

                return fig.to_json()

        else:

            assert (self.matplotlib_imported), (
                'matplotlib package not available. ' +
                'Plot cannot be generated.')
            assert (
                self.plotly_imported), ('plotly package is not available. ' +
                                        'Plot cannot be generated.')
