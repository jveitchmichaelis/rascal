import warnings
import itertools
from collections import Counter

import astropy.units as u
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize

from .util import load_calibration_lines
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
                 min_wavelength=3000,
                 max_wavelength=9000,
                 silence=False,
                 plotting_library='matplotlib'):
        '''
        Initialise the calibrator object.

        Parameters
        ----------
        peaks: list
            List of identified arc line pixel values.
        num_pix: int
            Number of pixels in the spectral axis.
        min_wavelength: float (default: 3000)
            Minimum wavelength of the arc lines.
        max_wavelength: float (default: 9000)
            Maximum wavelength of the arc lines.
        silence : boolean (default: False)
            Suppress all verbose output if set to True.
        plotting_library : string (default: 'matplotlib')
            Choose between matplotlib and plotly.

        '''

        self.peaks = peaks
        self.num_pix = num_pix
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength
        self.silence = silence
        self.plotting_library = plotting_library
        self.matplotlib_imported = False
        self.plotly_imported = False
        self.plot_with_matplotlib = False
        self.plot_with_plotly = False

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

    def _get_atlas(self, elements, min_wavelength, max_wavelength,
                   min_intensity, min_distance):
        '''
        Load lines.

        Parameters
        ----------
        elements: string or list of string
            Element name in form of chemical symbol. Case insensitive.
        min_wavelength: float
            Minimum wavelength of the arc lines.
        max_wavelength: float
            Maximum wavelength of the arc lines.
        min_intensity: float
            Minimum intensity of the lines.
        min_distance: float
            Minimum separation between neighbouring lines.

        '''

        self.atlas_elements, self.atlas, self.atlas_intensities = \
            load_calibration_lines(elements,
                                   min_wavelength,
                                   max_wavelength)

    def _set_peaks(self, peaks, constrain_poly):
        '''
        Gather all the randomly matched and user-supplied pixel-wavelength pairs.

        Parameters
        ----------
        peaks :
            €£$
        constrain_poly : boolean
            €£$
        '''

        # Create a list of all possible pairs of detected peaks and lines from atlas
        self._generate_pairs(constrain_poly)

        # Include user supplied pairs that are always fitted to within a tolerance
        self.set_guess_pairs()

        # Include user supplied pairs that are always fitted as given
        self.set_known_pairs()

    def _generate_pairs(self, constrain_poly):
        '''
        Generate pixel-wavelength pairs without the allowed regions set by the
        linearity limit. This assumes a relatively linear spectrograph.

        Parameters
        ----------
        constrain_poly : boolean
            €£$
        '''

        pairs = [pair for pair in itertools.product(self.peaks, self.atlas)]

        if constrain_poly:
            # Remove pairs outside polygon
            valid_area = Delaunay([
                (0, self.min_wavelength + self.range_tolerance +
                 self.candidate_thresh),
                (0, self.min_wavelength - self.range_tolerance -
                 self.candidate_thresh),
                (self.num_pix, self.max_wavelength - self.range_tolerance -
                 self.candidate_thresh),
                (self.num_pix, self.max_wavelength + self.range_tolerance +
                 self.candidate_thresh)
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

    def _combine_linear_estimates(self, candidates, top_n):
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

        for candidate in candidates:
            peaks += list(candidate[0])
            wavelengths += list(candidate[1])

        peaks = np.array(peaks)
        wavelengths = np.array(wavelengths)

        out_peaks = []
        out_wavelengths = []

        for peak in np.unique(peaks):
            peak_matches = wavelengths[peaks == peak]

            if len(peak_matches) > 0:
                for match in Counter(peak_matches).most_common(
                        min(top_n, len(peak_matches))):
                    out_peaks.append(peak)
                    out_wavelengths.append(match[0])

        return out_peaks, out_wavelengths

    def _get_candidate_points_linear(self, dispersion, min_wavelength, thresh):
        '''
        Returns a list of peak/wavelengths pairs which agree with the fit

        (wavelength - dispersion*x + min_wavelength) < thresh

        Note: depending on the threshold set, one peak may match with multiple
        wavelengths.

        Parameters
        ----------
        dispersion: float
            In observational astronomy term, the R value
        min_wavelength: float
            Minimum wavelength of the arc lines.
        thresh: float

        Returns
        -------
        peaks: np.array
            Filtered list of peaks which match this line
        atlas_line: np.array
            Atlas wavelength corresponding to these peaks (within tolerance)

        '''

        predicted = (dispersion * self.pairs[:, 0] + min_wavelength)
        actual = self.pairs[:, 1]
        err = np.abs(predicted - actual)
        mask = (err <= thresh)

        return self.pairs[:, 0][mask], actual[mask]

    def _get_candidate_points_poly(self, fit, tolerance):
        '''
        Returns a list of peak/wavelengths pairs which agree with the fit

        (wavelength - dispersion*x + min_wavelength) < thresh

        Note: depending on the threshold set, one peak may match with multiple
        wavelengths.

        Parameters
        ----------
        fit: integer
            order of polynomial fit.
        tolerance: float

        Returns
        -------
        x_match:
            €£$
        y_match:
            €£$

        '''

        x_match = []
        y_match = []

        for p in self.peaks:
            x = self.polyval(p, fit)
            diff = np.abs(self.atlas - x)

            for y in self.atlas[diff < tolerance]:
                x_match.append(p)
                y_match.append(y)

        x_match = np.array(x_match)
        y_match = np.array(y_match)

        return x_match, y_match

    def _get_candidates(self, num_slope, top_n):
        '''
        Get the best trial pairs from the Hough space

        Parameters
        ----------
        num_slope : int
            €£$
        top_n : int
            Top ranked lines to be fitted.
        '''

        # Generate the accumulator from the pairs
        self.accumulator = self._hough_points(self.pairs[:, 0],
                                              self.pairs[:, 1],
                                              num_slopes=num_slope)

        # Get the line coeffients from the promising bins in the accumulator
        _, self.hough_lines = self._get_top_lines(self.accumulator,
                                                  top_n=top_n,
                                                  xbins=self.xbins,
                                                  ybins=self.ybins)

        # Locate candidate points for these lines fits
        self.candidates = []
        for line in self.hough_lines:
            m, c = line
            inliers_x, inliers_y = self._get_candidate_points_linear(
                m, c, self.candidate_thresh)
            self.candidates.append((inliers_x, inliers_y))

    def _solve_candidate_ransac(self, x, y, polydeg, sample_size, max_tries,
                                thresh, brute_force, coeff, progress,
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

        valid_solution = False
        best_p = None
        best_cost = 1e50
        best_err = 1e50
        best_mask = [False]
        best_residual = None
        best_inliers = 0

        if sample_size <= polydeg:
            sample_size = polydeg + 1

        x = np.array(x)
        y = np.array(y)

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
            return (best_p, best_err, sum(best_mask), False)

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

        for sample in sampler_list:
            if brute_force:
                x_hat = x[[sample]]
                y_hat = y[[sample]]
            else:
                # weight the probability of choosing the sample by the inverse
                # line density
                hist = np.histogram(peaks, bins=3)
                prob = 1. / hist[0][np.digitize(peaks, hist[1], right=True) -
                                    1]
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
            if self.pix is not None:
                x_hat = np.concatenate((x_hat, self.pix))
                y_hat = np.concatenate((y_hat, self.wave))

            # Try to fit the data.
            # This doesn't need to be robust, it's an exact fit.
            fit_coeffs = self.polyfit(x_hat, y_hat, polydeg)

            # Discard out-of-bounds fits
            if self.fittype == 'poly':
                if ((self.polyval(0, fit_coeffs) <
                     self.min_wavelength - self.range_tolerance) |
                    (self.polyval(0, fit_coeffs) >
                     self.min_wavelength + self.range_tolerance) |
                    (self.polyval(self.num_pix, fit_coeffs) >
                     self.max_wavelength + self.range_tolerance) |
                    (self.polyval(self.num_pix, fit_coeffs) <
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
            cost = sum(err)

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

    def _get_best_model(self, candidates, polydeg, sample_size, max_tries,
                        thresh, brute_force, coeff, progress, filter_close):
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
            €£$
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
            Fraction of detected peaks used for calibration [0-1].

        '''

        self.candidate_peak, self.candidate_arc =\
            self._combine_linear_estimates(candidates, top_n=3)

        coeff, rms, residual, _, valid = self._solve_candidate_ransac(
            self.candidate_peak,
            self.candidate_arc,
            polydeg=polydeg,
            sample_size=sample_size,
            max_tries=max_tries,
            thresh=thresh,
            brute_force=brute_force,
            coeff=coeff,
            progress=progress,
            filter_close=filter_close)

        peak_utilisation = len(residual) / len(self.peaks)

        if not self.silence:
            if not valid:
                warnings.warn("Invalid fit")

            if rms > self.fit_tolerance:
                warnings.warn("Error too large {} > {}".format(
                    err, self.fit_tolerance))
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
            print('matplotlib package not available.')

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
            print('plotly package not available.')

    def which_plotting_library(self):
        '''
        Call to show if the Calibrator is using matplotlib or plotly library
        (or neither).

        '''
        if self.plot_with_matplotlib:
            print('Using matplotlib.')
        elif self.plot_with_plotly:
            print('Using plotly.')
        else:
            print('Both maplotlib and plotly are not imported.')

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
                  min_wavelength=None,
                  max_wavelength=None,
                  min_intensity=None,
                  min_distance=None,
                  include_second_order=None,
                  constrain_poly=False):
        '''
        Provider the chemical symbol(s) to add arc lines to the Calibrator.

        Parameters
        ----------
        elements: string or list of strings
            Chemical symbol, case insensitive
        min_wavelength: float (default: None)
            Minimum wavelength of the arc lines.
        max_wavelength: float (default: None)
            Maximum wavelength of the arc lines.
        min_intensity: float (default: None)
            Minimum intensity of the arc lines. Refer to NIST for the intensity.
        min_distance: float (default: None)
            Minimum separation between neighbouring arc lines.
        include_second_order: boolean (default: None)
            Set to True to include second order arc lines.
        constrain_poly: boolean (default: False)
            €£$

        '''

        if min_wavelength is None:
            min_wavelength = self.min_wavelength - self.range_tolerance

        if max_wavelength is None:
            max_wavelength = self.max_wavelength + self.range_tolerance

        if isinstance(elements, str):
            elements = [elements]

        for element in elements:

            atlas_elements_tmp, atlas_tmp, atlas_intensities_tmp =\
                load_calibration_lines(
                    element, min_wavelength, max_wavelength,
                    include_second_order)

            self.atlas_elements.extend(atlas_elements_tmp)
            self.atlas.extend(atlas_tmp)
            self.atlas_intensities.extend(atlas_intensities_tmp)

        self._set_peaks(self.peaks, constrain_poly)

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

        self.atlas = None

    def append_atlas(self, elements, atlas, intensity):
        '''
        Add arc lines to the Calibrator.

        Parameters
        ----------
        elements : list
           Element name in form of chemical symbol. Case insensitive.
        atlas : list
           Wavelengths of the lines in Angstrom.
        intensity : list
           Intensities of the lines in Angstrom.

        '''

        assert (len(elements) == len(atlas) == len(intensity),
                ValueError('Please check the length of the input lists.'))

        self.atlas_elements.extend(elements)
        self.atlas.extend(atlas)
        self.atlas_intensities.extend(intensity)

    def load_user_atlas(self, elements, atlas, intensity):
        '''
        Remove all the arc lines loaded to the Calibrator and use the user
        provided arc lines instead.

        Parameters
        ----------
        elements : list
           Element name in form of chemical symbol. Case insensitive.
        atlas : list
           Wavelengths of the lines in Angstrom.
        intensity : list
           Intensities of the lines in Angstrom.

        '''

        assert (len(elements) == len(atlas) == len(intensity),
                ValueError('Please check the length of the input lists.'))

        self.atlas_elements = elements
        self.atlas = atlas
        self.atlas_intensities = intensity

    def set_guess_pairs(self, pix_guess=(), wave_guess=(), margin=5.):
        '''
        Provide manual pixel-wavelength pair(s), good guess values with a
        margin of error.

        Parameters
        ----------
        pix_guess : numeric value, list or numpy 1D array (N) (default: ())
            Any pixel value; can be outside the detector chip and
            serve purely as anchor points.
        wave_guess : numeric value, list or numpy 1D array (N) (default: ())
            The matching wavelength for each of the pix_guess.
        margin : float (default: 5)
            Tolerance in the wavelength value of the pixel-to-wavelength
            mappping.

        '''

        assert (len(pix_guess) == len(wave_guess),
                ValueError('Please check the length of the input lists.'))

        self.pix_guess = np.asarray(pix_guess, dtype='float')
        self.wave_guess = np.asarray(wave_guess, dtype='float')
        self.margin = margin

    def set_known_pairs(self, pix=(), wave=()):
        '''
        Provide manual pixel-wavelength pair(s), fixed values in the fit.
        Use with caution because it can completely skew or bias the fit.

        Parameters
        ----------
        pix : numeric value, list or numpy 1D array (N)
            Any pixel value, can be outside the detector chip and
            serve purely as anchor points.
        wave : numeric value, list or numpy 1D array (N)
            The matching wavelength for each of the pix.

        '''

        assert (len(pix) == len(wave),
                ValueError('Please check the length of the input lists.'))

        self.pix = np.asarray(pix, dtype='float')
        self.wave = np.asarray(wave, dtype='float')

    def set_fit_constraints(self,
                            num_slopes=5000,
                            range_tolerance=500,
                            fit_tolerance=10.,
                            polydeg=4,
                            candidate_thresh=15.,
                            linearity_thresh=1.5,
                            ransac_thresh=1,
                            num_candidates=25,
                            xbins=100,
                            ybins=100,
                            brute_force=False,
                            fittype='poly'):
        '''
        Configure the Calibrator. This requires some manual twiddling before
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
        linearity_thresh : float (default: 1.5)
            A threshold (Angstroms) that expresses how non-linear the solution
            can be. This mostly affects which atlas points are included and
            should be reasonably large, e.g. 500A.
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

        # TODO: This has problems if you reduce range tolerance.
        self.min_slope = (
            (self.max_wavelength - self.range_tolerance) -
            (self.min_intercept + self.range_tolerance)) / self.num_pix

        self.max_slope = (
            (self.max_wavelength + self.range_tolerance) -
            (self.min_intercept - self.range_tolerance)) / self.num_pix

        # This seems wrong.
        #self.min_slope /= self.linearity_thresh
        #self.max_slope *= self.linearity_thresh

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
            coeff=None,
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
        coeff : list (default: None)
            €£$ how is this used???
        filter_close : boolean (default: False)
            €£$

        Returns
        -------
        coeff: list
            List of best fit polynomial coefficient.
        rms: float
            RMS
        residual: float
            Residual from the best fit
        peak_utilisation: float
            Fraction of detected peaks used for calibration [0-1].

        '''

        if sample_size > len(self.atlas):
            print("Size of sample_size is larger than the size of atlas, " +
                  "the sample_size is set to match the size of atlas = " +
                  str(len(self.atlas)) + ".")
            sample_size = len(self.atlas)

        self._get_candidates(self.num_slopes, self.num_candidates)

        p, rms, residual, peak_utilisation = self._get_best_model(
            self.candidates, self.polydeg, sample_size, max_tries,
            self.ransac_thresh, self.brute_force, coeff, progress,
            filter_close)

        return p, rms, residual, peak_utilisation

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
            if x < 0:
                return np.inf
            diff = self.atlas - x
            diff_abs = np.abs(diff)
            idx = np.argmin(diff_abs)

            if diff_abs[idx] < tolerance:
                x_match.append(p)
                y_match.append(self.atlas[idx])

        x_match = np.array(x_match)
        y_match = np.array(y_match)

        if len(x_match) < 5:
            return np.inf

        lsq = np.sum((y_match - self.polyval(x_match, fit))**2.) / len(x_match)

        return lsq

    def refine_fit(self,
                   fit,
                   delta,
                   tolerance=10.,
                   method='Nelder-Mead',
                   convergence=1e-6,
                   robust_refit=True,
                   polydeg=4):
        '''
        Refine the polynomial fit coefficients. Recomended to use in it
        multiple calls to first refine the lowest order and gradually increase
        the order of coefficients to be included for refinement. This is be
        achieved by providing delta in the length matching the number of the
        lowest degrees to be refined.

        Setting robust_refit to True will fit all the detected peaks with the
        given polynomial solution for a fit using maximal information, with
        the degree of polynomial = polydeg.

        Parameters
        ----------
        fit : list
            List of polynomial fit coefficients.
        delta : list
            List of delta(fit) as a starting condition for refining the
            solution. The length has to be less than or equal to the length
            of fit.
        tolerance : float (default: 10.)
            Absolute difference between fit and model in the unit of nm.
        method : string (default: 'Nelder-Mead')
            scipy.optimize.minimize method.
        convergence : float (default: 1e-6)
            scipy.optimize.minimize tol.
        robust_refit : boolean (default: True)
            Set to True to fit all the detected peaks with the given polynomial
            solution.
        polydeg : int (default: 4)
            Order of polynomial fit with all the detected peaks.

        Returns
        -------
        fit_new/coeff: list
            List of best fit polynomial coefficient.
        x_match: numpy 1D array
            €£$
        y_match: numpy 1D array
            €£$
        residual: numpy 1D array
            The difference (NOT absolute) between the data and the best-fit
            solution.
        peak_utilisation: float
            Fraction of detected peaks used for calibration [0-1].

        '''

        fit_delta = minimize(self._adjust_polyfit,
                             delta,
                             args=(fit, tolerance),
                             method=method,
                             tol=convergence,
                             options={
                                 'maxiter': 10000
                             }).x
        fit_new = fit.copy()

        for i, d in enumerate(fit_delta):
            fit_new[i] += d

        if np.any(np.isnan(fit_new)):
            warnings.warn('_adjust_polyfit() returns None. '
                          'Input solution is returned.')
            return fit, None, None, None, None

        x_match = []
        y_match = []
        residual = []

        for p in self.peaks:
            x = self.polyval(p, fit_new)
            if x < 0:
                return np.inf
            diff = self.atlas - x
            diff_abs = np.abs(diff)
            idx = np.argmin(diff_abs)

            if diff_abs[idx] < tolerance:
                x_match.append(p)
                y_match.append(self.atlas[idx])
                residual.append(diff[idx])

        x_match = np.array(x_match)
        y_match = np.array(y_match)
        residual = np.array(residual)

        peak_utilisation = len(x_match) / len(self.peaks)

        if robust_refit:
            coeff = models.robust_polyfit(x_match, y_match, polydeg)

            if np.any(np.isnan(coeff)):
                warnings.warn('robust_polyfit() returns None. '
                              'Input solution is returned.')

                return fit_new, x_match, y_match, residual, peak_utilisation

            return coeff, x_match, y_match, residual, peak_utilisation

        else:

            return fit_new, x_match, y_match, residual, peak_utilisation

    def plot_search_space(self, constrain_poly=False, coeff=None, top_n=3):
        '''
        Plots the peak/arc line pairs that are considered as potential match
        candidates.

        If fit coefficients are provided, the model solution will be
        overplotted.

        Parameters
        ----------
        constrain_poly : boolean (default: False)
            €£$
        coeff : list (default: None)
            List of best polynomial coefficients
        top_n : int (default: 3)
            Top ranked lines to be fitted.

        Return
        ------
        matplotlib.pyplot.gca() object

        '''

        plt.figure(figsize=(16, 9))

        self._generate_pairs(constrain_poly=constrain_poly)

        # Plot all-pairs
        plt.scatter(*self.pairs.T, alpha=0.2, c='red')

        # Get candidates
        self._get_candidates(num_slope=self.num_slopes,
                             top_n=self.num_candidates)

        plt.scatter(*self._merge_candidates(self.candidates).T, alpha=0.2)
        """
        plt.hlines(self.min_intercept, 0, self.num_pix)
        plt.hlines(self.max_intercept,
                   0,
                   self.num_pix,
                   linestyle='dashed',
                   alpha=0.5)
        """

        plt.text(5, self.min_wavelength + 100,
                 "Min wavelength (user-supplied)")
        plt.hlines(self.min_wavelength, 0, self.num_pix)
        plt.hlines(self.min_wavelength + self.range_tolerance,
                   0,
                   self.num_pix,
                   linestyle='dashed',
                   alpha=0.5)

        plt.text(5, self.max_wavelength + 100,
                 "Max wavelength (user-supplied)")
        plt.hlines(self.max_wavelength, 0, self.num_pix)
        plt.hlines(self.max_wavelength - self.range_tolerance,
                   0,
                   self.num_pix,
                   linestyle='dashed',
                   alpha=0.5)

        x_1 = np.arange(0, self.num_pix)
        m_1 = (self.max_wavelength - self.min_wavelength) / self.num_pix
        y_1 = m_1 * x_1 + self.min_wavelength
        plt.plot(x_1, y_1, label="Nominal linear fit")

        m_1 = (self.max_wavelength + self.range_tolerance -
               (self.min_wavelength + self.range_tolerance)) / self.num_pix
        y_1 = m_1 * x_1 + self.min_wavelength + self.range_tolerance
        plt.plot(x_1, y_1, c='black', linestyle='dashed')

        m_1 = (self.max_wavelength - self.range_tolerance -
               (self.min_wavelength - self.range_tolerance)) / self.num_pix
        y_1 = m_1 * x_1 + (self.min_wavelength - self.range_tolerance)
        plt.plot(x_1, y_1, c='black', linestyle='dashed')

        if coeff is not None:
            plt.scatter(self.peaks,
                        self.polyval(self.peaks, coeff),
                        color='red')

        plt.xlim(0, self.num_pix)
        plt.ylim(self.min_wavelength - self.range_tolerance,
                 self.max_wavelength + self.range_tolerance)

        self.candidate_peak, self.candidate_arc =\
            self._combine_linear_estimates(self.candidates, top_n=top_n)

        plt.scatter(self.candidate_peak, self.candidate_arc, s=20, c='purple')

        return plt.gca()

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

        if self.plot_with_matplotlib:

            pix = np.arange(len(spectrum)).astype('float')
            wave = self.polyval(pix, fit)

            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3,
                                                sharex=True,
                                                gridspec_kw={'hspace': 0.},
                                                figsize=(15, 9))
            fig.tight_layout()

            # Plot fitted spectrum
            ax1.plot(wave, spectrum)
            ax1.vlines(self.polyval(self.peaks, fit),
                       spectrum[self.peaks.astype('int')],
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

                if not self.silence:
                    print("Peak at: {} A".format(x))

                if np.abs(diff[idx]) < tolerance:
                    fitted_peaks.append(p)
                    fitted_diff.append(diff[idx])
                    if not self.silence:
                        print("- matched to {} A".format(self.atlas[idx]))
                    ax1.vlines(self.polyval(p, fit),
                               spectrum[p.astype('int')],
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
            ax3.plot(wave, pix)
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

            pix = np.arange(len(spectrum)).astype('float')
            wave = self.polyval(pix, fit)

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
                p_y.append(spectrum[int(p)])

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

                if not self.silence:
                    print("Peak at: {} A".format(x))

                if np.abs(diff[idx]) < tolerance:
                    fitted_peaks.append(p)
                    fitted_peaks_adu.append(spectrum[int(p)])
                    fitted_diff.append(diff[idx])
                    if not self.silence:
                        print("- matched to {} A".format(self.atlas[idx]))

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
                           y=pix,
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
                           range=[0., max(pix)],
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

            if json:
                return fig.to_json()
            if renderer == 'default':
                fig.show()
            else:
                fig.show(renderer)

        else:

            assert (self.matplotlib_imported), (
                'matplotlib package not available. ' +
                'Plot cannot be generated.')
            assert (
                self.plotly_imported), ('plotly package is not available. ' +
                                        'Plot cannot be generated.')
