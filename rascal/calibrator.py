import warnings
import itertools
import logging

import json
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
from scipy import interpolate

from .util import load_calibration_lines
from .util import derivative
from .util import gauss
from .util import vacuum_to_air_wavelength
from . import models

try:
    from tqdm.autonotebook import tqdm
    tdqm_imported = True
except Exception as e:
    warnings.warn(e)
    warnings.warn(
        'tqdm package not available. Progress bar will not be shown.')
    tdqm_imported = False


class HoughTransform:
    '''
    This handles the hough transform operations on the pixel-wavelength space.

    '''
    def __init__(self):

        self.hough_points = None
        self.hough_lines = None
        self.hist = None
        self.xedges = None
        self.yedges = None
        self.min_slope = None
        self.max_slope = None
        self.min_intercept = None
        self.max_intercept = None

    def set_constraints(self, min_slope, max_slope, min_intercept,
                        max_intercept):
        '''
        Define the minimum and maximum of the intercepts (wavelength) and
        gradients (wavelength/pixel) that Hough pairs will be generated.

        Parameters
        ----------
        min_slope: int or float
            Minimum gradient for wavelength/pixel
        max_slope: int or float
            Maximum gradient for wavelength/pixel
        min_intercept: int/float
            Minimum interception point of the Hough line
        max_intercept: int/float
            Maximum interception point of the Hough line

        '''

        assert np.isfinite(
            min_slope), 'min_slope has to be finite, %s is given ' % min_slope
        assert np.isfinite(
            max_slope), 'max_slope has to be finite, %s is given ' % max_slope
        assert np.isfinite(
            min_intercept
        ), 'min_intercept has to be finite, %s is given ' % min_intercept
        assert np.isfinite(
            max_intercept
        ), 'max_intercept has to be finite, %s is given ' % max_intercept

        self.min_slope = min_slope
        self.max_slope = max_slope
        self.min_intercept = min_intercept
        self.max_intercept = max_intercept

    def generate_hough_points(self, x, y, num_slopes):
        '''
        Calculate the Hough transform for a set of input points and returns the
        2D Hough hough_points matrix.

        Parameters
        ----------
        x: 1D numpy array
            The x-axis represents slope.
        y: 1D numpy array
            The y-axis represents intercept. Vertical lines (infinite gradient)
            are not accommodated.
        num_slopes: int
            The number of slopes to be generated.

        '''

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
        self.hough_points = np.column_stack((gradients, intercepts))

    def add_hough_points(self, hp):
        '''
        Extending the Hough pairs with an externally supplied HoughTransform
        object. This can be useful if the arc lines are very concentrated in
        some wavelength ranges while nothing in available in another part.

        Parameters
        ----------
        hp: numpy.ndarray with 2 columns or HoughTransform object
            An externally supplied HoughTransform object that contains
            hough_points.

        '''

        if isinstance(hp, HoughTransform):

            points = hp.hough_points

        elif isinstance(hp, np.ndarray):

            points = hp

        else:

            raise TypeError('Unsupported type for extending hough points.')

        self.hough_points = np.vstack((self.hough_points, points))

    def bin_hough_points(self, xbins, ybins):
        '''
        Bin up data by using a 2D histogram method.

        Parameters
        ----------
        xbins: int
            The number of bins in the pixel direction.
        ybins: int
            The number of bins in the wavelength direction.

        '''

        assert self.hough_points is not None, 'Please load an hough_points or '
        'create an hough_points with hough_points() first.'

        self.hist, self.xedges, self.yedges = np.histogram2d(
            self.hough_points[:, 0],
            self.hough_points[:, 1],
            bins=(xbins, ybins))

        # Get the line fit_coeffients from the promising bins in the
        # histogram
        self.hist_sorted_arg = np.dstack(
            np.unravel_index(
                np.argsort(self.hist.ravel())[::-1], self.hist.shape))[0]

        xbin_width = (self.xedges[1] - self.xedges[0]) / 2
        ybin_width = (self.yedges[1] - self.yedges[0]) / 2

        lines = []

        for b in self.hist_sorted_arg:

            lines.append((self.xedges[b[0]] + xbin_width,
                          self.yedges[b[1]] + ybin_width))

        self.hough_lines = lines

    def save(self,
             filename='hough_transform',
             fileformat='npy',
             delimiter='+',
             to_disk=True):
        '''
        Store the binned Hough space and/or the raw Hough pairs.

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

        fileformat_split = fileformat.split(delimiter)

        if 'npy' in fileformat_split:

            output_npy = []

            output_npy.append(self.hough_points)
            output_npy.append(self.hist)
            output_npy.append(self.xedges)
            output_npy.append(self.yedges)
            output_npy.append([self.min_slope])
            output_npy.append([self.max_slope])
            output_npy.append([self.min_intercept])
            output_npy.append([self.max_intercept])

            if to_disk:

                np.save(filename + '.npy', output_npy)

        if 'json' in fileformat_split:

            output_json = {}

            output_json['hough_points'] = self.hough_points.tolist()
            output_json['hist'] = self.hist.tolist()
            output_json['xedges'] = self.xedges.tolist()
            output_json['yedges'] = self.yedges.tolist()
            output_json['min_slope'] = self.min_slope
            output_json['max_slope'] = self.max_slope
            output_json['min_intercept'] = self.min_intercept
            output_json['max_intercept'] = self.max_intercept

            if to_disk:

                with open(filename + '.json', 'w+') as f:

                    json.dump(output_json, f)

        if not to_disk:

            if ('npy' in fileformat_split) and ('json'
                                                not in fileformat_split):

                return output_npy

            elif ('npy' not in fileformat_split) and ('json'
                                                      in fileformat_split):

                return output_json

            elif ('npy' in fileformat_split) and ('json' in fileformat_split):

                return output_npy, output_json

            else:

                return None

    def load(self, filename='hough_transform', filetype='npy'):
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

        if filetype == 'npy':

            input_npy = np.load(filename, allow_pickle=True)

            self.hough_points = input_npy[0]
            self.hist = input_npy[1].astype('float')
            self.xedges = input_npy[2].astype('float')
            self.yedges = input_npy[3].astype('float')
            self.min_slope = float(input_npy[4][0])
            self.max_slope = float(input_npy[5][0])
            self.min_intercept = float(input_npy[6][0])
            self.max_intercept = float(input_npy[7][0])

        elif filetype == 'json':

            input_json = json.load(open(filename))

            self.hough_points = input_json['hough_points']
            self.hist = np.array(input_json['hist']).astype('float')
            self.xedges = np.array(input_json['xedges']).astype('float')
            self.yedges = np.array(input_json['yedges']).astype('float')
            self.min_slope = float(input_json['min_slope'])
            self.max_slope = float(input_json['max_slope'])
            self.min_intercept = float(input_json['min_intercept'])
            self.max_intercept = float(input_json['max_intercept'])

        else:

            raise ValueError('Unknown filetype %s, it has to be npy or json' %
                             filetype)


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
        self.set_calibrator_properties()
        self.set_hough_properties()
        self.set_ransac_properties()

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
                'A guess solution for a polynomail fit has to '
                'be provided as fit_coeff in fit() in order to generate '
                'candidates for RANSAC sampling.')

        x_match = []
        y_match = []
        w_match = []
        self.candidates = []

        for p in self.peaks:

            x = self.polyval(p, self.fit_coeff)
            diff = np.abs(self.atlas - x)

            weight = gauss(self.atlas[diff < candidate_tolerance], 1., x,
                           self.range_tolerance)

            for y, w in zip(self.atlas[diff < candidate_tolerance], weight):

                x_match.append(p)
                y_match.append(y)
                w_match.append(weight)

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
        candidate_tolerance: float (default: 10)
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

            self._get_candidate_points_poly()

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

        if self.sample_size <= self.fit_deg:

            self.sample_size = self.fit_deg + 1

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

        if tdqm_imported & progress:

            sampler_list = tqdm(sampler)

        else:

            sampler_list = sampler

        peaks = np.sort(np.unique(x))
        idx = range(len(peaks))

        # Build a key(pixel)-value(wavelength) dictionary from the candidates
        candidates = {}

        for p in np.unique(x):

            candidates[p] = y[x == p]

        xbin_size = (self.ht.xedges[1] - self.ht.xedges[0]) / 2.
        ybin_size = (self.ht.yedges[1] - self.ht.yedges[0]) / 2.

        if np.isfinite(self.hough_weight):

            twoditp = interpolate.RectBivariateSpline(
                self.ht.xedges[1:] - xbin_size, self.ht.yedges[1:] - ybin_size,
                self.ht.hist)

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

                    continue

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
                                        derivative(fit_coeffs))
                intercept = wave - gradient * self.pixel_list

                # modified cost function weighted by the Hough space density
                if np.isfinite(self.hough_weight):

                    weight = self.hough_weight * np.sum(
                        twoditp(intercept, gradient, grid=False))

                else:

                    weight = 1.

                cost = sum(err) / (len(err) - len(fit_coeffs) + 1) / (weight +
                                                                      1e-9)

                # reject lines outside the rms limit (ransac_tolerance)
                best_mask = err < self.ransac_tolerance
                n_inliers = sum(best_mask)

                if len(matched_x[best_mask]) <= self.fit_deg:

                    self.logger.debug('Too few good candidates for fitting.')
                    continue

                # Want the most inliers with the lowest error
                if cost <= best_cost:

                    # Now we do a robust fit
                    try:

                        best_p = models.robust_polyfit(matched_x[best_mask],
                                                       matched_y[best_mask],
                                                       self.fit_deg)

                    except np.linalg.LinAlgError:

                        self.logger.warning(
                            "Linear algebra error in robust fit")
                        continue

                    best_cost = cost

                    # Get the residual of the fit
                    err = self.polyval(matched_x[best_mask],
                                       best_p) - matched_y[best_mask]
                    err[np.abs(err) >
                        self.ransac_tolerance] = self.ransac_tolerance

                    best_err = np.sqrt(np.mean(err**2))
                    best_residual = err
                    best_inliers = n_inliers

                    if tdqm_imported & progress:

                        sampler_list.set_description(
                            'Most inliers: {:d}, best error: {:1.4f}'.format(
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

    def set_calibrator_properties(self,
                                  num_pix=None,
                                  pixel_list=None,
                                  plotting_library='matplotlib',
                                  log_level='info'):
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
        log_level: string (default: 'info')
            Choose {critical, error, warning, info, debug, notset}.

        '''

        # initialise the logger
        self.logger = logging.getLogger(__name__)
        level = logging.getLevelName(log_level.upper())
        logging.basicConfig(level=level)

        # set the num_pix
        if num_pix is None:

            try:

                self.num_pix = len(self.spectrum)

            except Exception as e:

                self.logger.warning(e)
                self.logger.warning('Neither num_pix nor spectrum is given, '
                                    'it uses 1.1 times max(peaks) as the '
                                    'maximum pixel value.')
                self.num_pix = 1.1 * max(self.peaks)

        else:

            self.num_pix = num_pix

        self.logger.info('num_pix is set to {}.'.format(num_pix))

        # set the pixel_list
        if pixel_list is None:

            self.pixel_list = np.arange(self.num_pix)

        else:

            self.pixel_list = np.asarray(pixel_list)

        self.logger.info('pixel_list is set to {}.'.format(pixel_list))

        # map the list position to the pixel value
        self.pix_to_rawpix = interpolate.interp1d(
            self.pixel_list, np.arange(len(self.pixel_list)))

        # set the plotting library
        self.plotting_library = plotting_library

        if self.plotting_library == 'matplotlib':

            self.use_matplotlib()
            self.logger.info('Plotting with matplotlib.')

        elif self.plotting_library == 'plotly':

            self.use_plotly()
            self.logger.info('Plotting with plotly.')

        elif self.plotting_library == 'none':

            self.logger.info('Plotting is disabled.')

        else:

            self.logger.warning(
                'Unknown plotting_library, please choose from '
                'matplotlib or plotly. Execute use_matplotlib() or '
                'use_plotly() to manually select the library.')

    def set_hough_properties(self,
                             num_slopes=5000,
                             xbins=500,
                             ybins=500,
                             min_wavelength=3000,
                             max_wavelength=9000,
                             range_tolerance=500,
                             linearity_tolerance=50):
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

        self.num_slopes = int(num_slopes)
        self.xbins = xbins
        self.ybins = ybins
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength
        self.range_tolerance = range_tolerance
        self.linearity_tolerance = linearity_tolerance

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
                              sample_size=5,
                              top_n_candidate=5,
                              linear=True,
                              filter_close=False,
                              ransac_tolerance=5,
                              candidate_weighted=True,
                              hough_weight=1.0):
        '''
        Configure the Calibrator. This may require some manual twiddling before
        the calibrator can work efficiently. However, in theory, a large
        max_tries in fit() should provide a good solution in the expense of
        performance (minutes instead of seconds).

        Parameters
        ----------
        sample_size: int (default: 5)
            €£$
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

        '''

        self.sample_size = sample_size

        if self.sample_size > len(self.atlas):

            self.logger.warning(
                'Size of sample_size is larger than the size of atlas, ' +
                'the sample_size is set to match the size of atlas = ' +
                str(len(self.atlas)) + '.')
            self.sample_size = len(self.atlas)

        self.top_n_candidate = top_n_candidate
        self.linear = linear
        self.filter_close = filter_close
        self.ransac_tolerance = ransac_tolerance
        self.candidate_weighted = candidate_weighted
        self.hough_weight = hough_weight

    def do_hough_transform(self):

        # Generate the hough_points from the pairs
        self.ht = HoughTransform()
        self.ht.set_constraints(self.min_slope, self.max_slope,
                                self.min_intercept, self.max_intercept)
        self.ht.generate_hough_points(self.pairs[:, 0],
                                      self.pairs[:, 1],
                                      num_slopes=self.num_slopes)
        self.ht.bin_hough_points(self.xbins, self.ybins)

        self.hough_points = self.ht.hough_points
        self.hough_lines = self.ht.hough_lines

    def save_hough_transform(self):
        # to be implemented
        pass

    def load_hough_transform(self):
        # to be implemented
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
            If it is not known, assume 10% decrement per 1000 meter altitude
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
                       element,
                       atlas,
                       intensity=None,
                       candidate_tolerance=10,
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
            Element (required). Preferably a standard (i.e. periodic table)
            name for convenience with built-in atlases
        atlas: list/float
            Wavelength to add (Angstrom)
        intensity: list/float
            Relative line intensity (NIST value)
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

    def set_known_pairs(self, pix=(), wave=()):
        '''
        Provide manual pixel-wavelength pair(s), they will be appended to the
        list of pixel-wavelength pairs after the random sample being drawn from
        the RANSAC step, i.e. they are ALWAYS PRESENT in the fitting step. Use
        with caution because it can skew or bias the fit significantly, make
        sure the pixel value is accurate to at least 1/10 of a pixel.

        This can be used for example for low intensity lines at the edge of
        the spectrum.

        Parameters
        ----------
        pix: numeric value, list or numpy 1D array (N) (default: ())
            Any pixel value, can be outside the detector chip and
            serve purely as anchor points.
        wave: numeric value, list or numpy 1D array (N) (default: ())
            The matching wavelength for each of the pix.

        '''

        pix = np.asarray(pix, dtype='float')
        wave = np.asarray(wave, dtype='float')

        assert pix.size == wave.size, ValueError(
            'Please check the length of the input arrays. pix has size {} '
            'and wave has size {}.'.format(pix.size, wave.size))

        if pix.size == 1:

            self.pix_known = np.array([pix])
            self.wave_known = np.array([wave])

        else:

            self.pix_known = pix
            self.wave_known = wave

    def fit(self,
            max_tries=500,
            fit_deg=4,
            fit_coeff=None,
            fit_tolerance=10.,
            fit_type='poly',
            candidate_tolerance=10.,
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
        fit_tolerance: float (default: 10)
            Sets a tolerance on whether a fit found by RANSAC is considered
            acceptable
        fit_type: string (default: 'poly')
            One of 'poly', 'legendre' or 'chebyshev'
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
            RMS
        residual: float
            Residual from the best fit
        peak_utilisation: float
            Fraction of detected peaks used for calibration (if there are more
            peaks than the number of atlas lines, the fraction of atlas lines
            is returned instead) [0-1].

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

        if (self.hough_lines is None) or (self.hough_points is None):

            self.do_hough_transform()

        fit_coeff, rms, residual, n_inliers, valid =\
            self._solve_candidate_ransac(
                fit_deg=self.fit_deg,
                fit_coeff=self.fit_coeff,
                max_tries=self.max_tries,
                candidate_tolerance=candidate_tolerance,
                brute_force=self.brute_force,
                progress=self.progress)

        if len(self.peaks) < len(self.atlas):

            peak_utilisation = n_inliers / len(self.peaks)

        else:

            peak_utilisation = n_inliers / len(self.atlas)

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

        return self.fit_coeff, self.rms, self.residual, self.peak_utilisation

    def match_peaks(self,
                    fit_coeff,
                    n_delta=None,
                    refine=True,
                    tolerance=10.,
                    method='Nelder-Mead',
                    convergence=1e-6,
                    min_frac=0.5,
                    robust_refit=True,
                    fit_deg=None):
        '''
        **EXPERIMENTAL**

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
        fit_coeff: list
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
        residual: numpy 1D array
            The difference (NOT absolute) between the data and the best-fit
            solution.
        peak_utilisation: float
            Fraction of detected peaks used for calibration [0-1].

        '''

        fit_coeff_new = fit_coeff.copy()

        if fit_deg is None:

            fit_deg = len(fit_coeff) - 1

        if refine:

            if n_delta is None:

                n_delta = len(fit_coeff) - 1

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

                warnings.warn('_adjust_polyfit() returns None. '
                              'Input solution is returned.')
                return fit_coeff, None, None, None, None

        peak_matched = []
        atlas_matched = []
        residual = []

        for p in self.peaks:

            x = self.polyval(p, fit_coeff_new)
            diff = self.atlas - x
            diff_abs = np.abs(diff)
            idx = np.argmin(diff_abs)

            if diff_abs[idx] < tolerance:

                peak_matched.append(p)
                atlas_matched.append(self.atlas[idx])
                residual.append(diff[idx])

        peak_matched = np.array(peak_matched)
        atlas_matched = np.array(atlas_matched)
        residual = np.array(residual)

        if len(self.peaks) < len(self.atlas):

            peak_utilisation = len(peak_matched) / len(self.peaks)

        else:

            peak_utilisation = len(peak_matched) / len(self.atlas)

        if robust_refit:

            fit_coeff = models.robust_polyfit(peak_matched, atlas_matched,
                                              fit_deg)

            if np.any(np.isnan(fit_coeff)):
                warnings.warn('robust_polyfit() returns None. '
                              'Input solution is returned.')

                return (fit_coeff_new, peak_matched, atlas_matched, residual,
                        peak_utilisation)

            return (fit_coeff, peak_matched, atlas_matched, residual,
                    peak_utilisation)

        else:

            return (fit_coeff_new, peak_matched, atlas_matched, residual,
                    peak_utilisation)

    def plot_arc(self,
                 log_spectrum=False,
                 savefig=False,
                 filename=None,
                 json=False,
                 renderer='default',
                 display=True):
        '''
        Plots the 1D spectrum of the extracted arc

        parameters
        ----------
        log_spectrum: boolean (default: False)
            Set to true to display the wavelength calibrated arc spectrum in
            logarithmic space.
        savefig: boolean (default: False)
            Save a png image if set to True. Other matplotlib.pyplot.savefig()
            support format type are possible through providing the extension
            in the filename.
        filename: string (default: None)
            Provide a filename or full path. If the extension is not provided
            it is defaulted to png.
        json: boolean (default: False)
            Set to True to return json strings if using plotly as the plotting
            library.
        renderer: string (default: 'default')
            Indicate the Plotly renderer. Nothing gets displayed if json is
            set to True.

        Returns
        -------
        Return json strings if using plotly as the plotting library and json
        is True.

        '''

        if self.plot_with_matplotlib:

            plt.figure(figsize=(18, 5))

            if self.spectrum is not None:
                if log_spectrum:
                    plt.plot(np.log10(self.spectrum / self.spectrum.max()),
                             label='Arc Spectrum')
                    plt.vlines(self.peaks,
                               -2,
                               0,
                               label='Detected Peaks',
                               color='C1')
                    plt.ylabel("log(Normalised Count)")
                    plt.ylim(-2, 0)
                else:
                    plt.plot(self.spectrum / self.spectrum.max(),
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

            if display:

                plt.show()

            if savefig:

                if filename is not None:

                    plt.savefig(filename)

                else:

                    plt.savefig()

        if self.plot_with_plotly:

            fig = go.Figure()

            if log_spectrum:

                # Plot all-pairs
                fig.add_trace(
                    go.Scatter(
                        x=list(np.arange(len(self.spectrum))),
                        y=list(np.log10(self.spectrum / self.spectrum.max())),
                        mode='lines',
                        name='Arc'))
                xmin = min(np.log10(self.spectrum / self.spectrum.max()))
                xmax = max(np.log10(self.spectrum / self.spectrum.max()))

            else:

                # Plot all-pairs
                fig.add_trace(
                    go.Scatter(x=list(np.arange(len(self.spectrum))),
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

            if display:

                if renderer == 'default':

                    fig.show()

                else:

                    fig.show(renderer)

            if json:

                return fig.to_json()

    def plot_search_space(self,
                          fit_coeff=None,
                          top_n_candidate=3,
                          weighted=True,
                          savefig=False,
                          filename=None,
                          json=False,
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
        savefig: (default: False)
            Set to True to save figure to the destination as provided in
            'filename'.
        filename: (default: None)
            The destination to save the image.
        json: (default: False)
            Set to True to save the plotly figure as json string. Ignored if
            matplotlib is used.
        renderer: (default: 'default')
            Set the rendered for the plotly display. Ignored if matplotlib is
            used.

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
                height=800,
                width=1000)

            if display:

                if renderer == 'default':

                    fig.show()

                else:

                    fig.show(renderer)

            if json:

                return fig.to_json()

    def plot_fit(self,
                 fit_coeff,
                 spectrum=None,
                 tolerance=5.,
                 plot_atlas=True,
                 log_spectrum=False,
                 savefig=False,
                 filename=None,
                 json=False,
                 renderer='default',
                 display=True):
        '''
        Plots of the wavelength calibrated arc spectrum, the residual and the
        pixel-to-wavelength solution.

        Parameters
        ----------
        fit_coeff: 1D numpy array or list
            Best fit polynomail fit_coefficients
        spectrum: 1D numpy array (N)
            Array of length N pixels
        tolerance: float (default: 5)
            Absolute difference between model and fitted wavelengths in unit
            of angstrom.
        plot_atlas: boolean (default: True)
            Display all the relavent lines available in the atlas library.
        log_spectrum: boolean (default: False)
            Display the arc in log-space if set to True.
        savefig: boolean (default: False)
            Save a png image if set to True. Other matplotlib.pyplot.savefig()
            support format type are possible through providing the extension
            in the filename.
        filename: string (default: None)
            Provide a filename or full path. If the extension is not provided
            it is defaulted to png.
        json: boolean (default: False)
            Set to True to return json strings if using plotly as the plotting
            library.
        renderer: string (default: 'default')
            Indicate the Plotly renderer. Nothing gets displayed if json is
            set to True.

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

        if log_spectrum:

            spectrum[spectrum < 0] = 1e-100
            spectrum = np.log10(spectrum)
            vline_max = np.nanmax(spectrum) * 2.0
            text_box_pos = 1.2 * max(spectrum)

        else:

            vline_max = np.nanmax(spectrum) * 1.2
            text_box_pos = 0.8 * max(spectrum)

        wave = self.polyval(self.pixel_list, fit_coeff)

        if self.plot_with_matplotlib:

            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3,
                                                sharex=True,
                                                gridspec_kw={'hspace': 0.},
                                                figsize=(15, 9))
            fig.tight_layout()

            # Plot fitted spectrum
            ax1.plot(wave, spectrum, label='Arc Spectrum')
            ax1.vlines(self.polyval(self.peaks, fit_coeff),
                       spectrum[self.pix_to_rawpix(self.peaks).astype('int')],
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

            if display:

                if renderer == 'default':

                    fig.show()

                else:

                    fig.show(renderer)

            if savefig:

                if filename is not None:

                    fig.savefig(filename)

                else:

                    fig.savefig()

        elif self.plot_with_plotly:

            fig = go.Figure()

            # Top plot - arc spectrum and matched peaks
            fig.add_trace(
                go.Scatter(x=wave,
                           y=spectrum,
                           mode='lines',
                           yaxis='y3',
                           name='Arc Spectrum'))

            spec_max = np.nanmax(spectrum) * 1.05

            fitted_peaks = []
            fitted_peaks_adu = []
            fitted_diff = []
            all_diff = []

            for p in self.peaks:

                x = self.polyval(p, fit_coeff)

                # Add vlines
                fig.add_shape(type='line',
                              xref='x',
                              yref='y3',
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
                    fitted_peaks_adu.append(spectrum[int(
                        self.pix_to_rawpix(p))])
                    fitted_diff.append(diff[idx])
                    self.logger.info('- matched to {} A'.format(
                        self.atlas[idx]))

            x_fitted = self.polyval(fitted_peaks, fit_coeff)

            fig.add_trace(
                go.Scatter(
                    x=x_fitted,
                    y=fitted_peaks_adu,
                    mode='markers',
                    marker=dict(color=pio.templates["CN"].layout.colorway[1]),
                    yaxis='y3',
                    showlegend=False))

            # Middle plot - Residual plot
            rms = np.sqrt(np.mean(np.array(fitted_diff)**2.))
            fig.add_trace(
                go.Scatter(
                    x=x_fitted,
                    y=fitted_diff,
                    mode='markers',
                    marker=dict(color=pio.templates["CN"].layout.colorway[1]),
                    yaxis='y2',
                    showlegend=False))
            fig.add_trace(
                go.Scatter(x=[wave.min(), wave.max()],
                           y=[0, 0],
                           mode='lines',
                           line=dict(
                               color=pio.templates["CN"].layout.colorway[0],
                               dash='dash'),
                           yaxis='y2',
                           showlegend=False))
            fig.add_trace(
                go.Scatter(x=[wave.min(), wave.max()],
                           y=[rms, rms],
                           mode='lines',
                           line=dict(color='black', dash='dash'),
                           yaxis='y2',
                           showlegend=False))
            fig.add_trace(
                go.Scatter(x=[wave.min(), wave.max()],
                           y=[-rms, -rms],
                           mode='lines',
                           line=dict(color='black', dash='dash'),
                           yaxis='y2',
                           name='RMS'))

            # Bottom plot - Polynomial fit for Pixel to Wavelength
            fig.add_trace(
                go.Scatter(
                    x=x_fitted,
                    y=fitted_peaks,
                    mode='markers',
                    marker=dict(color=pio.templates["CN"].layout.colorway[1]),
                    yaxis='y1',
                    name='Fitted Peaks'))
            fig.add_trace(
                go.Scatter(
                    x=wave,
                    y=self.pixel_list,
                    mode='lines',
                    line=dict(color=pio.templates["CN"].layout.colorway[2]),
                    yaxis='y1',
                    name='Solution'))

            # Layout, Title, Grid config
            fig.update_layout(
                autosize=True,
                yaxis3=dict(title='Electron Count / e-',
                            range=[
                                np.log10(np.percentile(spectrum, 15)),
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
                    range=[
                        self.polyval(min(fitted_peaks), fit_coeff) * 0.95,
                        self.polyval(max(fitted_peaks), fit_coeff) * 1.05
                    ],
                    showgrid=True,
                ),
                hovermode='closest',
                showlegend=True,
                height=800,
                width=1000)

            if display:

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
