import warnings
import itertools
from collections import Counter
import logging
import astropy.units as u
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
from scipy import interpolate

from .util import load_calibration_lines
from .util import derivative
from .util import gauss
from .util import vacuum_to_air_wavelength
from .synthetic import SyntheticSpectrum
from . import models

try:
    from tqdm.autonotebook import tqdm
    tdqm_imported = True
except:
    warnings.warn(
        'tqdm package not available. Progress bar will not be shown.')
    tdqm_imported = False


class Calibrator:
    def __init__(self,
                 peaks,
                 num_pix,
                 pixel_list=None,
                 min_wavelength=3000,
                 max_wavelength=9000,
                 plotting_library='matplotlib',
                 log_level='info'):
        '''
        Initialise the calibrator object.

        Parameters
        ----------
        peaks: list
            List of identified arc line pixel values.
        num_pix: int
            Number of pixels in the spectral axis.
        min_wavelength: float (default: 3000)
            Minimum wavelength of the spectrum.
        max_wavelength: float (default: 9000)
            Maximum wavelength of the spectrum.
        plotting_library : string (default: 'matplotlib')
            Choose between matplotlib and plotly.
        log_level : string (default: 'info')
            Choose {critical, error, warning, info, debug, notset}.

        '''

        self.peaks = peaks
        self.num_pix = num_pix
        if pixel_list is None:
            self.pixel_list = np.arange(self.num_pix)
        else:
            self.pixel_list = np.asarray(pixel_list)
        self.pix_to_rawpix = interpolate.interp1d(
            self.pixel_list, np.arange(len(self.pixel_list)))
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength
        self.plotting_library = plotting_library
        self.matplotlib_imported = False
        self.plotly_imported = False
        self.plot_with_matplotlib = False
        self.plot_with_plotly = False

        self.logger = logging.getLogger(__name__)
        level = logging.getLevelName(log_level.upper())
        logging.basicConfig(level=level)

        self.atlas_elements = []
        self.atlas = []
        self.atlas_intensities = []

        # Configuring default fitting constraints
        self.set_fit_constraints()

        if self.plotting_library == 'matplotlib':
            self.use_matplotlib()
        elif self.plotting_library == 'plotly':
            self.use_plotly()
        elif self.plotting_library == 'none':
            pass
        else:
            warnings.warn('Unknown plotting_library, please choose from '
                          'matplotlib or plotly. Execute use_matplotlib() or '
                          'use_plotly() to manually select the library.')

    def _get_atlas(self, elements, min_atlas_wavelength, max_atlas_wavelength,
                   min_intensity, min_distance):
        '''
        Load lines.

        Parameters
        ----------
        elements: string or list of string
            Element name in form of chemical symbol. Case insensitive.
        min_atlas_wavelength: float
            Minimum wavelength of the arc lines.
        max_atlas_wavelength: float
            Maximum wavelength of the arc lines.
        min_intensity: float
            Minimum intensity of the lines.
        min_distance: float
            Minimum separation between neighbouring lines.

        '''

        self.atlas_elements, self.atlas, self.atlas_intensities = \
            load_calibration_lines(elements,
                                   min_atlas_wavelength,
                                   max_atlas_wavelength)

    def set_peaks(self, constrain_poly):
        '''
        Gather all the randomly matched and user-supplied pixel-wavelength pairs.

        Parameters
        ----------
        constrain_poly : boolean
            Apply a polygonal constraint on possible peak/atlas pairs

        '''

        # Create a list of all possible pairs of detected peaks and lines from atlas
        self._generate_pairs(constrain_poly)

        # Include user supplied pairs that are always fitted as given
        self.set_known_pairs()

    def _generate_pairs(self, constrain_poly):
        '''
        Generate pixel-wavelength pairs without the allowed regions set by the
        linearity limit. This assumes a relatively linear spectrograph.

        Parameters
        ----------
        constrain_poly : boolean
            Apply a polygonal constraint on possible peak/atlas pairs

        '''

        pairs = [pair for pair in itertools.product(self.peaks, self.atlas)]

        if constrain_poly:
            # Remove pairs outside polygon
            valid_area = Delaunay([
                (0, self.min_wavelength + self.range_tolerance +
                 self.candidate_thresh),
                (0, self.min_wavelength - self.range_tolerance -
                 self.candidate_thresh),
                (self.pixel_list.max(), self.max_wavelength -
                 self.range_tolerance - self.candidate_thresh),
                (self.pixel_list.max(), self.max_wavelength +
                 self.range_tolerance + self.candidate_thresh)
            ])

            mask = (valid_area.find_simplex(pairs) >= 0)
            self.pairs = np.array(pairs)[mask]
        else:
            self.pairs = np.array(pairs)

    def _hough_points(self, x, y, num_slopes):
        """
        Calculate the Hough transform for a set of input points and returns the
        2D Hough accumulator matrix.

        Parameters
        ----------
        x : 1D numpy array
            The x-axis represents slope.
        y : 1D numpy array
            The y-axis represents intercept. Vertical lines (infinite gradient)
            are not accommodated.
        num_slopes : int
            The number of slopes to be generated.

        Returns
        -------
        accumulator : 2D numpy array
            A 2D Hough accumulator array.

        """

        # Getting all the slopes
        slopes = np.linspace(self.min_slope, self.max_slope, num_slopes)

        # Computing all the intercepts and gradients
        intercepts = np.concatenate(y - np.outer(slopes, x))
        gradients = np.concatenate(np.column_stack([slopes] * len(x)))

        # Apply boundaries
        mask = ((self.min_intercept <= intercepts) &
                (intercepts <= self.max_intercept))
        intercepts = intercepts[mask]
        gradients = gradients[mask]

        # Create an array of Hough Points
        accumulator = np.column_stack((gradients, intercepts))

        return accumulator

    def _bin_accumulator(self, accumulator, xbins, ybins):
        '''
        Bin up data by using a 2D histogram method.

        Parameters
        ----------
        accumulator : 2D numpy array
            A 2D Hough accumulator array.
        xbins : int
            The number of bins in the pixel direction.
        ybins : int
            The number of bins in the wavelength direction.

        Returns
        -------
        hist : 2D numpy array (M*N)
            Height of the 2D histogram
        xedges : 1D numpy array (M)
            Centres of the y-bins
        yedges : 1D numpy array (N)
            Centres of the y-bins

        '''

        hist, xedges, yedges = np.histogram2d(accumulator[:, 0],
                                              accumulator[:, 1],
                                              bins=(xbins, ybins))

        return hist, xedges, yedges

    def _get_top_lines(self, accumulator, top_n, xbins, ybins):
        '''
        Get the most likely matched pairs in the binned Hough space.

        Parameters
        ----------
        accumulator : 2D numpy array
            A 2D Hough accumulator array.
        top_n : int
            Top ranked lines to be fitted.
        xbins : int
            The number of bins in the pixel direction.
        ybins : int
            The number of bins in the wavelength direction.

        Returns
        -------
        hist :
            €£$
        lines :
            €£$
        '''

        # Find the top bins
        hist, xedges, yedges = self._bin_accumulator(accumulator, xbins, ybins)

        xbin_width = (xedges[1] - xedges[0]) / 2
        ybin_width = (yedges[1] - yedges[0]) / 2

        top_bins = np.dstack(
            np.unravel_index(
                np.argsort(hist.ravel())[::-1][:top_n], hist.shape))[0]

        lines = []
        for b in top_bins:
            lines.append(
                (xedges[b[0]] + xbin_width, yedges[b[1]] + ybin_width))

        return hist, lines

    def _merge_candidates(self, candidates):
        '''
        Merge two candidate lists.

        Parameters
        ----------
        candidates : list
            €£$
        '''

        merged = []

        for pairs in candidates:
            for pair in np.array(pairs).T:
                merged.append(pair)

        return np.sort(np.array(merged))

    def _get_most_common_candidates(self, candidates, top_n, weighted=True):
        '''
        Takes a number of candidate pair sets and returns the most common pair for each wavelength

        Parameters
        ----------
        candidates: list of list(float, float)
            a list of list of peak/line pairs
        top_n : int
            Top ranked lines to be fitted.

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
        counts = np.ones(len(probabilities))

        out_peaks = []
        out_wavelengths = []

        for peak in np.unique(peaks):
            idx = np.where(peaks == peak)

            if len(idx) > 0:
                peak_matches = wavelengths[idx]
                if weighted:
                    counts = probabilities[idx]
                else:
                    counts = np.sum(len(idx))
                n = int(min(top_n, len(peak_matches)))
                if n == 1:
                    out_peaks.append(float(peak))
                    out_wavelengths.append(float(peak_matches))
                else:
                    out_peaks.extend([peak] * n)
                    out_wavelengths.extend(
                        peak_matches[np.argsort(-counts)[:n]])

        return out_peaks, out_wavelengths

    def _get_candidate_points_linear(self):
        '''
        Returns a list of peak/wavelengths pairs which agree with the fit

        (wavelength - dispersion * x + min_wavelength) < thresh

        Note: depending on the threshold set, one peak may match with multiple
        wavelengths.

        Parameters
        ----------
        dispersion: float
            In observational astronomy term, the R value
        min_wavelength: float
            Wavelength of the hough line at pixel 0.
        thresh: float
            €£$

        Returns
        -------
        peaks: np.array
            Filtered list of peaks which match this line
        atlas_line: np.array
            Atlas wavelength corresponding to these peaks (within tolerance)

        '''

        # Get the line coeffients from the promising bins in the accumulator
        _, self.hough_lines = self._get_top_lines(self.accumulator,
                                                  top_n=self.num_candidates,
                                                  xbins=self.xbins,
                                                  ybins=self.ybins)

        # Locate candidate points for these lines fits
        self.candidates = []

        for line in self.hough_lines:
            dispersion, min_wavelength = line

            predicted = (dispersion * self.pairs[:, 0] + min_wavelength)
            actual = self.pairs[:, 1]
            diff = np.abs(predicted - actual)
            mask = (diff <= self.candidate_thresh)

            # Match the range_tolerance to 1.1775 s.d. to match the FWHM
            # Note that the pairs outside of the range_tolerance were already
            # removed in an earlier stage
            weight = gauss(actual[mask], 1., predicted[mask],
                           (self.range_tolerance+self.linearity_thresh) * 1.1775)

            self.candidates.append((self.pairs[:,
                                               0][mask], actual[mask], weight))

    def _get_candidate_points_poly(self):
        '''
        Returns a list of peak/wavelengths pairs which agree with the fit

        (wavelength - dispersion * x + min_wavelength) < thresh

        Note: depending on the threshold set, one peak may match with multiple
        wavelengths.

        Parameters
        ----------
        fit: integer
            order of polynomial fit.
        tolerance: float
            €£$

        Returns
        -------
        x_match:
            €£$
        y_match:
            €£$

        '''

        if self.pfit is None:
            raise ValueError(
                'A guess solution for a polynomail fit has to '
                'be provided as coeff in fit() in order to generate '
                'candidates for RANSAC sampling.')

        x_match = []
        y_match = []
        w_match = []
        self.candidates = []

        for p in self.peaks:
            x = self.polyval(p, self.pfit)
            diff = np.abs(self.atlas - x)

            weight = gauss(self.atlas[diff < self.candidate_thresh], 1., x,
                           self.range_tolerance)

            for y, w in zip(self.atlas[diff < self.candidate_thresh], weight):
                x_match.append(p)
                y_match.append(y)
                w_match.append(weight)

        x_match = np.array(x_match)
        y_match = np.array(y_match)
        w_match = np.array(w_match)

        self.candidates.append((x_match, y_match, w_match))

    def _solve_candidate_ransac(self, polydeg, sample_size, max_tries, thresh,
                                brute_force, coeff, linear, weighted, progress,
                                filter_close):
        '''
        Use RANSAC to sample the parameter space and give best guess

        Parameters
        ----------
        x: 1D numpy array
            Array of pixels from peak detection.
        y: 1D numpy array
            Array of wavelengths from atlas.
        polydeg: int
            The order of polynomial (the polynomial type is definted in the
            set_fit_constraints).
        sample_size: int
            Number of lines to be fitted.
        max_tries: int
            Number of trials of polynomial fitting.
        thresh:
            Threshold for considering a point an inlier
        brute_force: boolean
            Solve all pixel-wavelength combinations with set to True.
        coeff: None or 1D numpy array
            Initial polynomial fit coefficients.
        progress: boolean
            Show the progress bar with tdqm if set to True.
        filter_close: boolean
            €£$

        Returns
        -------
        best_p : list
            A list of size polydeg of the best fit polynomial coefficient.
        best_err : float
            Arithmetic mean of the residuals.
        sum(best_inliers) : int
            Number of lines fitted within the thresh.
        valid_solution : boolean
            False if overfitted.

        '''

        if linear:
            self._get_candidate_points_linear()
        else:
            self._get_candidate_points_poly()

        self.candidate_peak, self.candidate_arc =\
            self._get_most_common_candidates(self.candidates, top_n=3, weighted=weighted)

        valid_solution = False
        best_p = None
        best_cost = 1e50
        best_err = 1e50
        best_mask = [False]
        best_residual = None
        best_inliers = 0

        if sample_size <= polydeg:
            sample_size = polydeg + 1

        x = np.array(self.candidate_peak)
        y = np.array(self.candidate_arc)

        # Filter close wavelengths
        if filter_close:
            unique_y = np.unique(y)
            idx = np.argwhere(unique_y[1:] - unique_y[0:-1] < 3 * thresh)
            separation_mask = np.argwhere((y == unique_y[idx]).sum(0) == 0)
            y = y[separation_mask].flatten()
            x = x[separation_mask].flatten()

        if coeff is not None:
            fit = self.polyval(x, coeff)
            err = np.abs(fit - y)
            best_cost = sum(err)
            best_err = np.sqrt(np.mean(err**.2))

        # If the number of lines is smaller than the number of degree of
        # polynomial fit, return failed fit.
        if len(np.unique(x)) <= polydeg:
            return (best_p, best_err, sum(best_mask), 0, False)

        idx = range(len(x))

        # if the request sample_size is the same or larger than the available
        # lines, it is essentially a brute force
        if brute_force or (sample_size >= len(np.unique(x))):
            sampler = itertools.combinations(idx, sample_size)
            sample_size = len(np.unique(x))
        else:
            sampler = range(int(max_tries))

        # Brute force check all combinations. N choose 4 is pretty fast.
        if tdqm_imported & progress:
            sampler_list = tqdm(sampler)
        else:
            sampler_list = sampler

        peaks = np.unique(x)
        idx = range(len(peaks))

        # Clean up:
        candidates = {}
        for p in np.unique(x):
            candidates[p] = y[x == p]

        hist, xedges, yedges = np.histogram2d(self.accumulator[:, 1],
                                              self.accumulator[:, 0],
                                              bins=(self.xbins, self.ybins),
                                              range=((self.min_intercept,
                                                      self.max_intercept),
                                                     (self.min_slope,
                                                      self.max_slope)))

        xbin_size = (xedges[1] - xedges[0]) / 2.
        ybin_size = (yedges[1] - yedges[0]) / 2.

        twoditp = interpolate.RectBivariateSpline(xedges[1:] - xbin_size,
                                                  yedges[1:] - ybin_size, hist)

        for sample in sampler_list:
            if brute_force:
                x_hat = x[[sample]]
                y_hat = y[[sample]]
            else:
                # weight the probability of choosing the sample by the inverse
                # line density
                h = np.histogram(peaks, bins=10)
                prob = 1. / h[0][np.digitize(peaks, h[1], right=True) - 1]
                prob = prob / np.sum(prob)

                # Pick some random peaks
                idxes = np.random.choice(idx,
                                         sample_size,
                                         replace=False,
                                         p=prob)
                x_hat = peaks[idxes]
                y_hat = []

                # Pick a random wavelength for this x
                for _x in x_hat:
                    y_hat.append(np.random.choice(candidates[_x]))

            if (len(y_hat) > len(np.unique(y_hat))):
                #print("dupe y - impossible?")
                continue

            # insert user given known pairs
            if self.pix_known is not None:
                x_hat = np.concatenate((x_hat, self.pix_known))
                y_hat = np.concatenate((y_hat, self.wave_known))

            # Try to fit the data.
            # This doesn't need to be robust, it's an exact fit.
            fit_coeffs = self.polyfit(x_hat, y_hat, polydeg)

            # Check monotonicity.
            if not np.all(
                    np.diff(self.polyval(self.pixel_list, fit_coeffs)) > 0):
                continue

            # Discard out-of-bounds fits
            if self.fittype == 'poly':
                if ((self.polyval(0, fit_coeffs) <
                     self.min_wavelength - self.range_tolerance) |
                    (self.polyval(0, fit_coeffs) >
                     self.min_wavelength + self.range_tolerance) |
                    (self.polyval(self.pixel_list.max(), fit_coeffs) >
                     self.max_wavelength + self.range_tolerance) |
                    (self.polyval(self.pixel_list.max(), fit_coeffs) <
                     self.max_wavelength - self.range_tolerance)):
                    continue
            elif self.fittype == 'chebyshev':
                pass
            elif self.fittype == 'legendre':
                pass
            else:
                warnings.warn('Unknown fittype: ' + str(self.fittype) +
                              ', boundary'
                              'conditions are not tested.')

            # TODO use point-in-polygon to check entire solution space
            # (not just tails)

            # M-SAC Estimator (Torr and Zisserman, 1996)
            fit = self.polyval(x, fit_coeffs)
            err = np.abs(fit - y)
            err[err > thresh] = thresh

            # compute the hough space density as weights for the cost function
            wave = self.polyval(self.pixel_list, fit_coeffs)
            gradient = self.polyval(self.pixel_list, derivative(fit_coeffs))
            intercept = wave - gradient * self.pixel_list

            weight = np.sum(twoditp(intercept, gradient, grid=False))
            cost = sum(err) / (len(err) - len(fit_coeffs) + 1) / weight

            # reject lines outside the rms limit (thresh)
            best_mask = err < thresh
            n_inliers = sum(best_mask)

            # Want the most inliers with the lowest error
            if cost <= best_cost:

                # Now we do a robust fit
                best_p = models.robust_polyfit(x[best_mask], y[best_mask],
                                               polydeg)

                best_cost = cost

                # Get the residual of the fit
                err = self.polyval(x[best_mask], best_p) - y[best_mask]
                err[np.abs(err) > thresh] = thresh
                #best_cost = sum(err)
                best_err = np.sqrt(np.mean(err**2))
                best_residual = err
                best_inliers = n_inliers

                if tdqm_imported & progress:
                    sampler_list.set_description(
                        "Most inliers: {:d}, best error: {:1.4f}".format(
                            n_inliers, best_err))

                # Perfect fit, break early
                if best_inliers == len(x):
                    break

        # Overfit check
        if best_inliers == polydeg + 1:
            valid_solution = False
        else:
            valid_solution = True

        return best_p, best_err, best_residual, best_inliers, valid_solution

    def _get_best_model(self, polydeg, sample_size, max_tries, thresh,
                        brute_force, coeff, linear, weighted, progress,
                        filter_close):
        '''
        Get the most likely solution with RANSAC

        Parameters
        ----------
        candidates :
            €£$
        polydeg : int
            The degree of the polynomial to be fitted.
        sample_size : int
            Number of lines to be fitted.
        max_tries : int
            Number of trials of polynomial fitting.
        thresh : float
            RANSAC tolerance
        brute_force : boolean
            Solve all pixel-wavelength combinations with set to True.
        coeff : None or 1D numpy array
            Intial guess of the set of polynomial fit co-efficients.
        progress : boolean
            Show the progress bar with tdqm if set to True.
        filter_close : boolean
            €£$

        Returns
        -------
        coeff: list
            List of best fit polynomial coefficient.
        rms: float
            Root mean square.
        residual: float
            Residual from the best fit.
        peak_utilisation: float
            Fraction of detected peaks used for calibration (if there are more
            peaks than the number of atlas lines, the fraction of atlas lines
            is returned instead).

        '''

        # Generate the accumulator from the pairs
        self.accumulator = self._hough_points(self.pairs[:, 0],
                                              self.pairs[:, 1],
                                              num_slopes=self.num_slopes)

        coeff, rms, residual, n_inliers, valid = self._solve_candidate_ransac(
            polydeg=polydeg,
            sample_size=sample_size,
            max_tries=max_tries,
            thresh=thresh,
            brute_force=brute_force,
            coeff=coeff,
            linear=linear,
            weighted=weighted,
            progress=progress,
            filter_close=filter_close)

        if len(self.peaks) < len(self.atlas):
            peak_utilisation = n_inliers / len(self.peaks)
        else:
            peak_utilisation = n_inliers / len(self.atlas)

        if not valid:
            self.logger.warn("Invalid fit")

        if rms > self.fit_tolerance:
            self.logger.warn("RMS too large {} > {}".format(
                rms, self.fit_tolerance))

        assert (coeff is not None), "Couldn't fit"

        return coeff, rms, residual, peak_utilisation

    def _import_matplotlib(self):
        '''
        Call to import plotly.

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
            import plotly.graph_objects as go
            import plotly.io as pio
            self.plotly_imported = True
        except ImportError:
            self.logger.error('plotly package not available.')

    def which_plotting_library(self):
        '''
        Call to show if the Calibrator is using matplotlib or plotly library
        (or neither).

        '''
        if self.plot_with_matplotlib:
            self.logger.info('Using matplotlib.')
        elif self.plot_with_plotly:
            self.logger.info('Using plotly.')
        else:
            self.logger.warn('Neither maplotlib nor plotly are imported.')

    def use_matplotlib(self):
        '''
        Call to switch to matplotlib.

        '''

        if not self.matplotlib_imported:
            self._import_matplotlib()
            self.plot_with_matplotlib = True
            self.plot_with_plotly = False
        else:
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
        else:
            self.plot_with_plotly = True
            self.plot_with_matplotlib = False

    def add_atlas(self,
                  elements,
                  min_atlas_wavelength=None,
                  max_atlas_wavelength=None,
                  min_intensity=10,
                  min_distance=10,
                  vacuum=False,
                  pressure=101325.,
                  temperature=273.15,
                  relative_humidity=0.,
                  constrain_poly=False):
        '''
        Adds an atlas of arc lines to the calibrator, given an element.

        Arc lines are taken from a general list of NIST lines and can be filtered
        using the minimum relative intensity (note this may not be accurate due to
        instrumental effects such as detector response, dichroics, etc) and
        minimum line separation.

        Lines are filtered first by relative intensity, then by separation. This
        is to improve robustness in the case where there is a strong line very
        close to a weak line (which is within the separation limit).

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
            Minimum intensity of the arc lines. Refer to NIST for the intensity.
        min_distance: float (default: None)
            Minimum separation between neighbouring arc lines.
        include_second_order: boolean (default: None)
            Set to True to include second order arc lines.
        pressure: float
            Pressure when the observation took place, in Pascal.
            If it is not known, assume 10% decrement per 1000 meter altitude
        temperature: float
            Temperature when the observation took place, in Kelvin.
        relative_humidity: float
            In percentage.
        constrain_poly : boolean
            Apply a polygonal constraint on possible peak/atlas pairs
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
                    min_intensity, min_distance, vacuum, pressure, temperature, relative_humidity)

            self.atlas_elements.extend(atlas_elements_tmp)
            self.atlas.extend(atlas_tmp)
            self.atlas_intensities.extend(atlas_intensities_tmp)

        self.set_peaks(constrain_poly)

    def list_atlas(self):
        '''
        List all the lines loaded to the Calibrator.

        '''

        for i in range(len(self.atlas)):
            print("Element " + str(self.atlas_elements[i]) + " at " +
                  str(self.atlas[i]) + " with intensity " +
                  str(self.atlas_intensities[i]))

    def clear_atlas(self):
        '''
        Remove all the lines loaded to the Calibrator.

        '''

        self.atlas_elements = []
        self.atlas = []
        self.atlas_intensities = []

    def add_user_atlas(self,
                       element,
                       atlas,
                       intensity=None,
                       vacuum=False,
                       pressure=101325.,
                       temperature=273.15,
                       relative_humidity=0.):
        '''
        Add a single or list of arc lines. Each arc line should have an
        element label associated with it. It is recommended that you use
        a standard periodic table abbreviation (e.g. "Hg"), but it makes
        no difference to the fitting process.

        The vacuum to air wavelength conversion is deafult to False because
        observatories usually provide the line lists in the respective air
        wavelength, as the corrections from temperature and humidity are
        small. See https://emtoolbox.nist.gov/Wavelength/Documentation.asp

        Parameters
        ----------
        elements : list/str
            Element (required). Preferably a standard (i.e. periodic table)
            name for convenience with built-in atlases
        atlas : list/float
            Wavelength to add (Angstrom)
        intensity : list/float
            Relative line intensity (NIST value)
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

        if not isinstance(element, list):
            element = list(element)

        if not isinstance(atlas, list):
            atlas = list(atlas)

        if intensity is None:
            intensity = [0] * len(atlas)
        else:
            if not isinstance(intensity, list):
                intensity = list(intensity)

        assert len(element) == len(atlas), ValueError(
                'Input element and atlas have different length.')
        assert len(element) == len(intensity), ValueError(
                'Input element and intensity have different length.')

        if vacuum:
            atlas = vacuum_to_air_wavelength(atlas, temperature, pressure,
                                             relative_humidity)

        self.atlas_elements.extend(element)
        self.atlas.extend(atlas)
        self.atlas_intensities.extend(intensity)

    def load_user_atlas(self,
                        elements,
                        wavelengths,
                        intensities=None,
                        constrain_poly=False,
                        vacuum=False,
                        pressure=101325.,
                        temperature=273.15,
                        relative_humidity=0.):
        '''
        *Remove* all the arc lines loaded to the Calibrator and then use the user
        supplied arc lines instead.

        The vacuum to air wavelength conversion is deafult to False because
        observatories usually provide the line lists in the respective air
        wavelength, as the corrections from temperature and humidity are
        small. See https://emtoolbox.nist.gov/Wavelength/Documentation.asp

        Parameters
        ----------
        elements : list
            Element (required). Preferably a standard (i.e. periodic table)
            name for convenience with built-in atlases
        wavelengths : list
            Wavelength to add (Angstrom)
        intensities : list
            Relative line intensities
        constrain_poly : boolean
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

        self.clear_atlas()

        if intensities is None:
            intensities = [0] * len(wavelengths)

        assert len(elements) == len(wavelengths), ValueError(
                'Input elements and wavelengths have different length.')
        assert len(elements) == len(intensities), ValueError(
                'Input elements and intensities have different length.')

        self.add_user_atlas(elements, wavelengths, intensities, vacuum, pressure, temperature,
                                             relative_humidity)
        self.set_peaks(constrain_poly)

    def remove_atlas_lines_range(self, wavelength, tolerance=10):
        """
        Remove arc lines within a certain wavelength range.

        Parameters
        ----------
        wavelength : float
            Wavelength to remove (Angstrom)
        tolerance : float
            Tolerance around this wavelength where atlas lines will be removed

        """

        for i, line in enumerate(self.atlas):
            if abs(line - wavelength) < tolerance:
                removed_element = self.atlas_elements.pop(i)
                removed_peak = self.atlas.pop(i)
                self.atlas_intensities.pop(i)

                self.logger.info("Removed {} line : {} A".format(
                    removed_element, removed_peak))

    def set_known_pairs(self, pix=(), wave=()):
        '''
        Provide manual pixel-wavelength pair(s), fixed values in the fit.
        Use with caution because it can completely skew or bias the fit.

        This can be used for example for low intensity lines at the edge of
        the spectrum.

        Parameters
        ----------
        pix : numeric value, list or numpy 1D array (N) (default: ())
            Any pixel value, can be outside the detector chip and
            serve purely as anchor points.
        wave : numeric value, list or numpy 1D array (N) (default: ())
            The matching wavelength for each of the pix.

        '''

        #assert len(pix) > 0 and len(wave) > 0, ValueError(
        #    'Please supply at least one pair')

        assert len(pix) == len(wave), ValueError(
            'Please check the length of the input lists.')

        self.pix_known = np.asarray(pix, dtype='float')
        self.wave_known = np.asarray(wave, dtype='float')

    def set_fit_constraints(self,
                            num_slopes=5000,
                            range_tolerance=500,
                            fit_tolerance=10.,
                            polydeg=4,
                            candidate_thresh=15.,
                            linearity_thresh=100,
                            ransac_thresh=3,
                            num_candidates=25,
                            xbins=100,
                            ybins=100,
                            brute_force=False,
                            fittype='poly'):
        '''
        Configure the Calibrator. This may require some manual twiddling before
        the calibrator can work efficiently. However, in theory, a large
        max_tries in fit() should provide a good solution in the expense of
        performance (minutes instead of seconds).

        Parameters
        ----------
        num_slopes : int (default: 1000)
            Number of slopes to consider during Hough transform
        range_tolerance : float (default: 500)
            Estimation of the error on the provided spectral range
            e.g. 3000-5000 with tolerance 500 will search for
            solutions that may satisfy 2500-5500
        fit_tolerance : float (default: 10)
            Sets a tolerance on whether a fit found by RANSAC is considered
            acceptable
        polydeg : int (default: 4)
            Degree of the polynomial fit.
        candidate_thresh : float (default: 15)
            Threshold  (Angstroms) for considering a point to be an inlier
            during candidate peak/line selection. This should be reasonable
            small as we want to search for candidate points which are
            *locally* linear.
        linearity_thresh : float (default: 100)
            A threshold (Ansgtroms) which defines some padding around the
            range tolerance to allow for non-linearity. This should be the
            maximum expected excursion from linearity.
        ransac_thresh : float (default: 1)
            The distance criteria  (Angstroms) to be considered an inlier to a
            fit. This should be close to the size of the expected residuals on
            the final fit (e.g. 1A is typical)
        num_candidates: int (default: 25)
            Number of best trial Hough pairs.
        xbins : int (default: 50)
            Number of bins for Hough accumulation
        ybins : int (default: 50)
            Number of bins for Hough accumulation
        brute_force : boolean (default: False)
            Set to True to try all possible combination in the given parameter
            space
        fittype : string (default: 'poly')
            One of 'poly', 'legendre' or 'chebyshev'

        '''

        self.num_slopes = num_slopes
        self.range_tolerance = range_tolerance
        self.linearity_thresh = linearity_thresh

        # Start wavelength in the spectrum, +/- some tolerance
        self.min_intercept = self.min_wavelength - self.range_tolerance
        self.max_intercept = self.min_wavelength + self.range_tolerance

        self.min_slope = ((self.max_wavelength - self.range_tolerance - self.linearity_thresh) -
                          (self.min_intercept +
                           self.range_tolerance + self.linearity_thresh)) / self.pixel_list.max()

        self.max_slope = ((self.max_wavelength + self.range_tolerance + self.linearity_thresh) -
                          (self.min_intercept -
                           self.range_tolerance - self.linearity_thresh)) / self.pixel_list.max()

        self.fit_tolerance = fit_tolerance
        self.polydeg = polydeg
        self.ransac_thresh = ransac_thresh
        self.candidate_thresh = candidate_thresh
        self.xbins = xbins
        self.ybins = ybins
        self.brute_force = brute_force
        self.fittype = fittype
        self.num_candidates = num_candidates

        if fittype == 'poly':
            self.polyfit = np.polynomial.polynomial.polyfit
            self.polyval = np.polynomial.polynomial.polyval
        elif fittype == 'legendre':
            self.polyfit = np.polynomial.legendre.legfit
            self.polyval = np.polynomial.legendre.legval
        elif fittype == 'chebyshev':
            self.polyfit = np.polynomial.chebyshev.chebfit
            self.polyval = np.polynomial.chebyshev.chebval
        else:
            raise ValueError(
                'fittype must be: (1) poly, (2) legendre or (3) chebyshev')

    def fit(self,
            sample_size=5,
            top_n=10,
            max_tries=5000,
            progress=True,
            polyfit_coeff=None,
            linear=True,
            weighted=True,
            filter_close=False):
        '''
        Solve for the wavelength calibration polynomial.

        Parameters
        ----------
        sample_size : int (default: 5)
            €£$
        top_n : int (default: 10)
            Top ranked lines to be fitted.
        max_tries : int (default: 5000)
            Maximum number of iteration.
        progress : boolean (default: True)
            True to show progress with tdqm. It is overrid if tdqm cannot be
            imported.
        polyfit_coeff : list (default: None)
            €£$ how is this used???
        filter_close : boolean (default: False)
            €£$

        Returns
        -------
        polyfit_coeff: list
            List of best fit polynomial coefficient.
        rms: float
            RMS
        residual: float
            Residual from the best fit
        peak_utilisation: float
            Fraction of detected peaks used for calibration [0-1].

        '''

        if sample_size > len(self.atlas):
            self.logger.warn(
                "Size of sample_size is larger than the size of atlas, " +
                "the sample_size is set to match the size of atlas = " +
                str(len(self.atlas)) + ".")
            sample_size = len(self.atlas)

        self.pfit = polyfit_coeff

        self.pfit, self.rms, self.residual, self.peak_utilisation = self._get_best_model(
            self.polydeg, sample_size, max_tries, self.ransac_thresh,
            self.brute_force, self.pfit, linear, weighted, progress,
            filter_close)

        return self.pfit, self.rms, self.residual, self.peak_utilisation

    def _adjust_polyfit(self, delta, fit, tolerance):

        # x is wavelength
        # x_matched is pixel
        # y_matched is wavelength
        x_match = []
        y_match = []
        fit_new = fit.copy()

        for i, d in enumerate(delta):
            fit_new[i] += d

        for p in self.peaks:
            x = self.polyval(p, fit_new)
            diff = self.atlas - x
            diff_abs = np.abs(diff)
            idx = np.argmin(diff_abs)

            if diff_abs[idx] < tolerance:
                x_match.append(p)
                y_match.append(self.atlas[idx])

        x_match = np.array(x_match)
        y_match = np.array(y_match)

        if not np.all(np.diff(self.polyval(self.pixel_list, fit_new)) > 0):
            return np.inf

        if len(x_match) < 5:
            return np.inf

        lsq = np.sum(
            (y_match - self.polyval(x_match, fit_new))**2.) / len(x_match)

        return lsq

    def match_peaks(self,
                    polyfit_coeff,
                    n_delta=None,
                    refine=True,
                    tolerance=10.,
                    method='Nelder-Mead',
                    convergence=1e-6,
                    robust_refit=True,
                    polydeg=None):
        '''
        Refine the polynomial fit coefficients. Recommended to use in it
        multiple calls to first refine the lowest order and gradually increase
        the order of coefficients to be included for refinement. This is be
        achieved by providing delta in the length matching the number of the
        lowest degrees to be refined.

        Set refine to True to improve on the polynomial solution.

        Set robust_refit to True to fit all the detected peaks with the
        given polynomial solution for a fit using maximal information, with
        the degree of polynomial = polydeg.

        Set both refine and robust_refit to False will return the list of
        arc lines are well fitted by the current solution within the
        tolerance limit provided.

        Parameters
        ----------
        polyfit_coeff : list
            List of polynomial fit coefficients.
        n_delta : int (default: None)
            The number of the highest polynomial order to be adjusted
        refine : boolean (default: True)
            Set to True to refine solution.
        tolerance : float (default: 10.)
            Absolute difference between fit and model in the unit of nm.
        method : string (default: 'Nelder-Mead')
            scipy.optimize.minimize method.
        convergence : float (default: 1e-6)
            scipy.optimize.minimize tol.
        robust_refit : boolean (default: True)
            Set to True to fit all the detected peaks with the given polynomial
            solution.
        polydeg : int (default: length of the input coefficients)
            Order of polynomial fit with all the detected peaks.

        Returns
        -------
        polyfit_coeff: list
            List of best fit polynomial coefficient.
        peak_match: numpy 1D array
            Matched peaks
        atlas_match: numpy 1D array
            Corresponding atlas matches
        residual: numpy 1D array
            The difference (NOT absolute) between the data and the best-fit
            solution.
        peak_utilisation: float
            Fraction of detected peaks used for calibration [0-1].

        '''

        polyfit_coeff_new = polyfit_coeff.copy()

        if polydeg is None:
            polydeg = len(polyfit_coeff) - 1

        if n_delta is None:
            n_delta = len(polyfit_coeff) - 1

        delta = polyfit_coeff_new[:int(n_delta)] * 0.001

        if refine:
            fit_delta = minimize(self._adjust_polyfit,
                                 delta,
                                 args=(polyfit_coeff, tolerance),
                                 method=method,
                                 tol=convergence,
                                 options={
                                     'maxiter': 10000
                                 }).x

            for i, d in enumerate(fit_delta):
                polyfit_coeff_new[i] += d

            if np.any(np.isnan(polyfit_coeff_new)):
                warnings.warn('_adjust_polyfit() returns None. '
                              'Input solution is returned.')
                return polyfit_coeff, None, None, None, None

        peak_match = []
        atlas_match = []
        residual = []

        for p in self.peaks:
            x = self.polyval(p, polyfit_coeff_new)
            diff = self.atlas - x
            diff_abs = np.abs(diff)
            idx = np.argmin(diff_abs)

            if diff_abs[idx] < tolerance:
                peak_match.append(p)
                atlas_match.append(self.atlas[idx])
                residual.append(diff[idx])

        peak_match = np.array(peak_match)
        atlas_match = np.array(atlas_match)
        residual = np.array(residual)

        if len(self.peaks) < len(self.atlas):
            peak_utilisation = len(peak_match) / len(self.peaks)
        else:
            peak_utilisation = len(peak_match) / len(self.atlas)

        if robust_refit:
            coeff = models.robust_polyfit(peak_match, atlas_match, polydeg)

            if np.any(np.isnan(coeff)):
                warnings.warn('robust_polyfit() returns None. '
                              'Input solution is returned.')

                return polyfit_coeff_new, peak_match, atlas_match, residual, peak_utilisation

            return coeff, peak_match, atlas_match, residual, peak_utilisation

        else:

            return polyfit_coeff_new, peak_match, atlas_match, residual, peak_utilisation

    def plot_search_space(self,
                          constrain_poly=False,
                          coeff=None,
                          top_n=3,
                          weighted=True,
                          savefig=False,
                          filename=None,
                          json=False,
                          renderer='default'):
        '''
        Plots the peak/arc line pairs that are considered as potential match
        candidates.

        If fit coefficients are provided, the model solution will be
        overplotted.

        Parameters
        ----------
        constrain_poly : boolean
            Apply a polygonal constraint on possible peak/atlas pairs
        coeff : list (default: None)
            List of best polynomial coefficients
        top_n : int (default: 3)
            Top ranked lines to be fitted.

        Return
        ------
        matplotlib.pyplot.gca() object

        '''

        # Generate Hough pairs and only accept those within the tolerance
        # region
        self._generate_pairs(constrain_poly=constrain_poly)

        self.accumulator = self._hough_points(self.pairs[:, 0],
                                              self.pairs[:, 1],
                                              num_slopes=self.num_slopes)

        # Get candidates
        self._get_candidate_points_linear()

        # ?
        self.candidate_peak, self.candidate_arc =\
            self._get_most_common_candidates(self.candidates, top_n=top_n, weighted=weighted)

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

            plt.figure(figsize=(16, 9))

            # Plot all-pairs
            plt.scatter(*self.pairs.T, alpha=0.2, c='red')
            #plt.scatter(*self._merge_candidates(self.candidates).T, alpha=0.2)

            # Tolerance region around the minimum wavelength
            plt.text(5, self.min_wavelength + 100,
                     "Min wavelength (user-supplied)")
            plt.hlines(self.min_wavelength, 0, self.pixel_list.max())
            plt.hlines(self.min_wavelength + self.range_tolerance,
                       0,
                       self.pixel_list.max(),
                       linestyle='dashed',
                       alpha=0.5)
            plt.hlines(self.min_wavelength - self.range_tolerance,
                       0,
                       self.pixel_list.max(),
                       linestyle='dashed',
                       alpha=0.5)

            # Tolerance region around the maximum wavelength
            plt.text(5, self.max_wavelength + 100,
                     "Max wavelength (user-supplied)")
            plt.hlines(self.max_wavelength, 0, self.pixel_list.max())
            plt.hlines(self.max_wavelength + self.range_tolerance,
                       0,
                       self.pixel_list.max(),
                       linestyle='dashed',
                       alpha=0.5)
            plt.hlines(self.max_wavelength - self.range_tolerance,
                       0,
                       self.pixel_list.max(),
                       linestyle='dashed',
                       alpha=0.5)

            # The line from (first pixel, minimum wavelength) to
            # (last pixel, maximum wavelength), and the two lines defining the
            # tolerance region.
            plt.plot(x, y_1, label="Nominal linear fit")
            plt.plot(x, y_2, c='black', linestyle='dashed')
            plt.plot(x, y_3, c='black', linestyle='dashed')

            if coeff is not None:
                plt.scatter(self.peaks,
                            self.polyval(self.peaks, coeff),
                            color='red')

            plt.scatter(self.candidate_peak,
                        self.candidate_arc,
                        s=20,
                        c='purple')

            plt.xlim(0, self.pixel_list.max())
            plt.ylim(self.min_wavelength - self.range_tolerance,
                     self.max_wavelength + self.range_tolerance)

            plt.xlabel('Pixel')
            plt.ylabel('Wavelength / A')

            return plt.gca()

        elif self.plot_with_plotly:

            fig = go.Figure()

            # Plot all-pairs
            fig.add_trace(
                go.Scatter(x=self.pairs[:, 0],
                           y=self.pairs[:, 1],
                           mode='markers',
                           name='All Pairs',
                           marker=dict(color='red', opacity=0.2)))
            fig.add_trace(
                go.Scatter(x=self._merge_candidates(self.candidates)[:, 0],
                           y=self._merge_candidates(self.candidates)[:, 1],
                           mode='markers',
                           name='Candidate Pairs',
                           marker=dict(color='royalblue', opacity=0.2)))
            fig.add_trace(
                go.Scatter(x=self.candidate_peak,
                           y=self.candidate_arc,
                           mode='markers',
                           name='Best Candidate Pairs',
                           marker=dict(color='purple')))

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
                go.Scatter(x=x,
                           y=y_1,
                           mode='lines',
                           name='Linear Fit',
                           line=dict(color='blue')))
            fig.add_trace(
                go.Scatter(x=x,
                           y=y_2,
                           mode='lines',
                           name='Tolerance Region',
                           line=dict(color='blue', dash='dashdot')))
            fig.add_trace(
                go.Scatter(x=x,
                           y=y_3,
                           showlegend=False,
                           mode='lines',
                           line=dict(color='blue', dash='dashdot')))

            if coeff is not None:
                fig.add_trace(
                    go.Scatter(x=self.peaks,
                               y=self.polyval(self.peaks, coeff),
                               mode='markers',
                               name='Solution',
                               marker=dict(color='red')))

            # Layout, Title, Grid config
            fig.update_layout(
                autosize=True,
                yaxis=dict(
                    title='Pixel',
                    range=[
                        self.min_wavelength - self.range_tolerance * 1.1,
                        self.max_wavelength + self.range_tolerance * 1.1
                    ],
                    showgrid=True),
                xaxis=dict(
                    title='Wavelength / A',
                    zeroline=False,
                    range=[0., self.pixel_list.max()],
                    showgrid=True,
                ),
                hovermode='closest',
                showlegend=True,
                height=height,
                width=width)

            if renderer == 'default':
                fig.show()
            else:
                fig.show(renderer)
            if json:
                return fig.to_json()

    def plot_fit(self,
                 spectrum,
                 fit,
                 tolerance=5.,
                 plot_atlas=True,
                 log_spectrum=False,
                 savefig=False,
                 filename=None,
                 json=False,
                 renderer='default'):
        '''
        Parameters
        ----------
        spectrum : 1D numpy array (N)
            Array of length N pixels
        fit : 1D numpy array or list
            Best fit polynomail coefficients
        tolerance : float (default: 5)
            Absolute difference between model and fitted wavelengths in unit
            of angstrom.
        plot_atlas : boolean (default: True)
            Display all the relavent lines available in the atlas library.
        log_spectrum : boolean (default: False)
            Display the arc in log-space if set to True.
        savefig : boolean (default: False)
            Save a png image if set to True. Other matplotlib.pyplot.savefig()
            support format type are possible through providing the extension
            in the filename.
        filename : string (default: None)
            Provide a filename or full path. If the extension is not provided
            it is defaulted to png.
        json : boolean (default: False)
            Return json strings if using plotly as the plotting library and
            this is set to True.
        renderer : string (default: 'default')
            Indicate the Plotly renderer. Nothing gets displayed if json is
            set to True.

        Returns
        -------
        Return json strings if using plotly as the plotting library and json
        is True.

        '''

        if log_spectrum:
            spectrum[spectrum < 0] = 1e-100
            spectrum = np.log10(spectrum)
            vline_max = np.nanmax(spectrum) * 2.0
            text_box_pos = 1.2 * max(spectrum)
        else:
            vline_max = np.nanmax(spectrum) * 1.05
            text_box_pos = 0.8 * max(spectrum)

        wave = self.polyval(self.pixel_list, fit)

        if self.plot_with_matplotlib:

            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3,
                                                sharex=True,
                                                gridspec_kw={'hspace': 0.},
                                                figsize=(15, 9))
            fig.tight_layout()

            # Plot fitted spectrum
            ax1.plot(wave, spectrum)
            ax1.vlines(self.polyval(self.peaks, fit),
                       spectrum[self.pix_to_rawpix(self.peaks).astype('int')],
                       vline_max,
                       linestyles='dashed',
                       colors='C1')

            # Plot the atlas
            if plot_atlas:
                #spec = SyntheticSpectrum(
                #    fit, model_type='poly', degree=len(fit)-1)
                #x_locs = spec.get_pixels(self.atlas)
                ax1.vlines(self.atlas, 0, vline_max, colors='C2')

            fitted_peaks = []
            fitted_diff = []
            all_diff = []
            for p in self.peaks:
                x = self.polyval(p, fit)
                diff = self.atlas - x
                idx = np.argmin(np.abs(diff))
                all_diff.append(diff[idx])

                self.logger.info("Peak at: {} A".format(x))

                if np.abs(diff[idx]) < tolerance:
                    fitted_peaks.append(p)
                    fitted_diff.append(diff[idx])
                    self.logger.info("- matched to {} A".format(
                        self.atlas[idx]))
                    ax1.vlines(self.polyval(p, fit),
                               spectrum[self.pix_to_rawpix(p).astype('int')],
                               vline_max,
                               colors='C1')

                    ax1.text(x - 3,
                             text_box_pos,
                             s="{}:{:1.2f}".format(self.atlas_elements[idx],
                                                   self.atlas[idx]),
                             rotation=90,
                             bbox=dict(facecolor='white', alpha=1))

            rms = np.sqrt(np.mean(np.array(fitted_diff)**2.))

            ax1.grid(linestyle=':')
            ax1.set_ylabel('ADU')
            if log_spectrum:
                ax1.set_ylim(0, vline_max)
            else:
                ax1.set_ylim(np.nanmin(spectrum), vline_max)

            # Plot the residuals
            ax2.scatter(self.polyval(fitted_peaks, fit),
                        fitted_diff,
                        marker='+',
                        color='C1')
            ax2.hlines(0, wave.min(), wave.max(), linestyles='dashed')
            ax2.grid(linestyle=':')
            ax2.set_ylabel('Residual / A')
            '''
            ax2.legend(loc='lower right')
            ax2.text(
                min(wave) + np.ptp(wave) * 0.05,
                max(spectrum),
                'RMS =' + str(rms)[:6]
                )
            '''

            # Plot the polynomial
            ax3.scatter(self.polyval(fitted_peaks, fit),
                        fitted_peaks,
                        marker='+',
                        color='C1',
                        label='Peaks used for fitting')
            ax3.plot(wave, self.pixel_list)
            ax3.grid(linestyle=':')
            ax3.set_xlabel('Wavelength / A')
            ax3.set_ylabel('Pixel')
            ax3.legend(loc='lower right')
            ax3.set_xlim(wave.min(), wave.max())

            plt.show()

            if savefig:
                if filename is not None:
                    fig.savefig(output)
                else:
                    fig.savefig()

        elif self.plot_with_plotly:

            fig = go.Figure()

            # Top plot - arc spectrum and matched peaks
            fig.add_trace(
                go.Scatter(x=wave,
                           y=spectrum,
                           mode='lines',
                           line=dict(color='royalblue'),
                           yaxis='y3'))

            spec_max = np.nanmax(spectrum) * 1.05

            p_x = []
            p_y = []
            for i, p in enumerate(self.peaks):
                p_x.append(self.polyval(p, fit))
                p_y.append(spectrum[int(self.pix_to_rawpix(p))])

            fig.add_trace(
                go.Scatter(x=p_x,
                           y=p_y,
                           mode='markers',
                           marker=dict(color='orange'),
                           yaxis='y3'))

            fitted_peaks = []
            fitted_peaks_adu = []
            fitted_diff = []
            all_diff = []

            for p in self.peaks:
                x = self.polyval(p, fit)
                diff = self.atlas - x
                idx = np.argmin(np.abs(diff))
                all_diff.append(diff[idx])

                self.logger.info("Peak at: {} A".format(x))

                if np.abs(diff[idx]) < tolerance:
                    fitted_peaks.append(p)
                    fitted_peaks_adu.append(spectrum[int(
                        self.pix_to_rawpix(p))])
                    fitted_diff.append(diff[idx])
                    self.logger.info("- matched to {} A".format(
                        self.atlas[idx]))

            x_fitted = self.polyval(fitted_peaks, fit)

            fig.add_trace(
                go.Scatter(x=x_fitted,
                           y=fitted_peaks_adu,
                           mode='markers',
                           marker=dict(color='firebrick'),
                           yaxis='y3'))

            # Middle plot - Residual plot
            rms = np.sqrt(np.mean(np.array(fitted_diff)**2.))
            fig.add_trace(
                go.Scatter(x=x_fitted,
                           y=fitted_diff,
                           mode='markers',
                           marker=dict(color='firebrick'),
                           yaxis='y2'))
            fig.add_trace(
                go.Scatter(x=[wave.min(), wave.max()],
                           y=[0, 0],
                           mode='lines',
                           line=dict(color='royalblue', dash='dash'),
                           yaxis='y2'))

            # Bottom plot - Polynomial fit for Pixel to Wavelength
            fig.add_trace(
                go.Scatter(x=x_fitted,
                           y=fitted_peaks,
                           mode='markers',
                           marker=dict(color='firebrick'),
                           yaxis='y1',
                           name='Peaks used for fitting'))
            fig.add_trace(
                go.Scatter(x=wave,
                           y=self.pixel_list,
                           mode='lines',
                           line=dict(color='royalblue'),
                           yaxis='y1'))

            # Layout, Title, Grid config
            fig.update_layout(
                autosize=True,
                yaxis3=dict(title='ADU',
                            range=[
                                np.log10(np.percentile(spectrum, 10)),
                                np.log10(spec_max)
                            ],
                            domain=[0.67, 1.0],
                            showgrid=True,
                            type='log'),
                yaxis2=dict(title='Residual / A',
                            range=[min(fitted_diff),
                                   max(fitted_diff)],
                            domain=[0.33, 0.66],
                            showgrid=True),
                yaxis=dict(title='Pixel',
                           range=[0., max(self.pixel_list)],
                           domain=[0., 0.32],
                           showgrid=True),
                xaxis=dict(
                    title='Wavelength / A',
                    zeroline=False,
                    range=[min(wave), max(wave)],
                    showgrid=True,
                ),
                hovermode='closest',
                showlegend=False,
                height=800,
                width=1000)

            if renderer == 'default':
                fig.show()
            else:
                fig.show(renderer)
            if json:
                return fig.to_json()

        else:

            assert (self.matplotlib_imported), (
                'matplotlib package not available. ' +
                'Plot cannot be generated.')
            assert (
                self.plotly_imported), ('plotly package is not available. ' +
                                        'Plot cannot be generated.')
