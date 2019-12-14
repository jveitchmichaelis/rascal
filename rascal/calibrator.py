import warnings
import itertools
import numpy as np
import astropy.units as u
from collections import Counter

from . util import load_calibration_lines
from . synthetic import SyntheticSpectrum
from . import models

plotly_imported = False
matplotlib_imported = False

try:
    import matplotlib.pyplot as plt
    matplotlib_imported = True
except:
    warnings.warn(
        'matplotlib package not available.')
    matplotlib_imported = False

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    plotly_imported = True
except ImportError:
    if matplotlib_imported:
        warnings.warn(
            'plotly is not present, only matplotlib can be used.')
    else:
        warnings.warn(
            'Plot cannot be generated.')


try:
    from tqdm.autonotebook import tqdm
    tdqm_imported = True
except:
    warnings.warn(
        'tqdm package not available. Progress bar will not be shown.')
    tdqm_imported = False


class Calibrator:
    def __init__(self, peaks=None, silence=False, elements=None,
                    min_wavelength=1000, 
                    max_wavelength=10000, 
                    range_tolerance=200, 
                    min_atlas=None, 
                    max_atlas=None, 
                    min_intensity=0,
                    min_distance=10):
        self.peaks = peaks
        self.silence = silence
        self.matplotlib_imported = matplotlib_imported
        self.plotly_imported = plotly_imported

        if elements is None:
            self.elements = ["Hg", "Ar", "Xe", "Kr", "Ne"]
        elif isinstance(elements, list) and len(elements) == 0:
            raise ValueError
        else:
            self.elements = elements
        
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength

        # Configuring default fitting constraints
        self.set_fit_constraints(range_tolerance=range_tolerance)

        if min_atlas is None:
            min_atlas = self.min_wavelength - self.range_tolerance
        
        if max_atlas is None:
            max_atlas = self.max_wavelength + self.range_tolerance

        assert max_atlas > min_atlas
        assert min_atlas >= 0

        self._get_atlas(self.elements, min_wavelength=min_atlas, max_wavelength=max_atlas, min_intensity=min_intensity, min_distance=min_distance)

        if peaks is not None:
            self._set_peaks(peaks)

    def _set_peaks(self, peaks):
        # Create a list of all possible pairs of detected peaks and lines from atlas 
        self._generate_pairs()

        # Include user supplied pairs that are always fitted to within a tolerance
        self.set_guess_pairs()

        # Include user supplied pairs that are always fitted as given
        self.set_known_pairs()

    def _generate_pairs(self):
        self.pairs = np.array(
            [pair for pair in itertools.product(self.peaks, self.atlas)])

    def _hough_points(self, x, y, num_slopes):
        """
        Calculate the Hough transform for a set of input points.
        
        Returns the 2D Hough accumulator matrix.

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
        return np.histogram2d(
            accumulator[:, 0], accumulator[:, 1], bins=(xbins, ybins))
        
    def _get_top_lines(self, accumulator, top_n, xbins, ybins):
        '''
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
        h : 

        lines : 

        '''

        # Find the top bins
        hist, xedges, yedges = self._bin_accumulator(accumulator, xbins, ybins)

        xbin_width = (xedges[1] - xedges[0]) / 2
        ybin_width = (yedges[1] - yedges[0]) / 2

        top_bins = np.dstack(
            np.unravel_index(np.argsort(hist.ravel())[::-1][:top_n], hist.shape))[0]

        lines = []
        for b in top_bins:
            lines.append((xedges[b[0]] + xbin_width,
                          yedges[b[1]] + ybin_width))

        return hist, lines

    def _combine_linear_estimates(self, candidates):
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
            out_peaks.append(peak)
            out_wavelengths.append(Counter(wavelengths[peaks == peak]).most_common(1)[0][0])

        return out_peaks, out_wavelengths

    def _get_candidate_points_linear(self, dispersion, min_wavelength, thresh):
        '''
        Returns a list of peak/wavelengths pairs which agree with the fit

        (wavelength - dispersion*x + min_wavelength) < thresh

        Note: depending on the threshold set, one peak may match with multiple
        wavelengths.
        '''

        predicted = (dispersion * self.pairs[:, 0] + min_wavelength)
        actual = self.pairs[:, 1]
        err = np.abs(predicted - actual)
        mask = (err < thresh)

        return self.pairs[:, 0][mask], actual[mask]

    def _get_candidate_points_poly(self, fit, tolerance=5):
        '''
        Returns a list of peak/wavelengths pairs which agree with the fit

        (wavelength - dispersion*x + min_wavelength) < thresh

        Note: depending on the threshold set, one peak may match with multiple
        wavelengths.
        '''

        x_match = []
        y_match = []

        for p in self.peaks:
            x = self.polyval(fit, p)
            diff = np.abs(self.atlas - x)

            for y in self.atlas[diff < tolerance]:
                x_match.append(p)
                y_match.append(y)

        x_match = np.array(x_match)
        y_match = np.array(y_match)

        return x_match, y_match

    def _solve_candidate_ransac(self, x, y,
                                polydeg = 3,
                                sample_size = 4,
                                max_tries = 1e4,
                                thresh = 15,
                                brute_force = False,
                                coeff = None,
                                progress = False,
                                filter_close = True):
        '''
        Parameters
        ----------
        x : 1D numpy array
            Array of pixels from peak detection.
        y : 1D numpy array
            Array of wavelengths from atlas.
        polydeg : int
            The order of polynomial (the polynomial type is definted in the 
            set_fit_constraints).
        sample_size : int
            Number of lines to be fitted.
        max_tries : int
            Number of trials of polynomial fitting.
        thresh :
            Threshold for considering a point an inlier
        brute_force : tuple
            Solve all pixel-wavelength combinations with set to True.
        coeff : None or 1D numpy array
            Initial polynomial fit coefficients.
        progress : tuple
            Show the progress bar with tdqm if set to True.

        Returns
        -------
        best_p : list
            A list of size polydeg of the best fit polynomial coefficient.
        best_err : float
            Arithmetic mean of the residuals.
        sum(best_inliers) : int
            Number of lines fitted within the thresh.
        valid_solution : tuple
            False if overfitted.

        '''

        valid_solution = False
        best_p = None
        best_cost = 1e50
        best_err = 1e50
        best_mask = [False]
        best_inliers = 0

        if sample_size <= polydeg:
            sample_size = polydeg + 1

        x = np.array(x)
        y = np.array(y)

        # Filter close wavelengths
        if filter_close:
            unique_y = np.unique(y)
            idx = np.argwhere(unique_y[1:] - unique_y[0:-1] < 3*thresh)
            separation_mask = np.argwhere((y == unique_y[idx]).sum(0) == 0)
            y = y[separation_mask].flatten()
            x = x[separation_mask].flatten()

        if coeff is not None:
            fit = self.polyval(coeff, x)
            err = np.abs(fit - y)
            best_cost = sum(err)
            best_err = np.sqrt(np.mean(err**.2))

        # If the number of lines is smaller than the number of degree of polynomial
        # fit, return failed fit.
        if len(np.unique(x)) <= polydeg:
            return (best_p, best_err, sum(best_mask), False)

        idx = range(len(x))

        # if the request sample_size is the same or larger than the available
        # lines, it is essentially a brute force
        if brute_force or (sample_size>=len(np.unique(x))):
            sampler = itertools.combinations(idx, sample_size)
            sample_size = len(np.unique(x))
        else:
            sampler = range(int(max_tries))

        # Brute force check all combinations. N choose 4 is pretty fast.
        if tdqm_imported & progress:
            sampler_list = tqdm(sampler)
        else:
            sampler_list = sampler
        for sample in sampler_list:

            if brute_force:
                x_hat = x[[sample]]
                y_hat = y[[sample]]
            else:
                # weight the probability of choosing the sample by the inverse line density
                hist = np.histogram(x, bins=3)
                prob = 1. / hist[0][np.digitize(x, hist[1], right=True)-1]
                prob = prob / np.sum(prob)

                idxes = np.random.choice(idx, sample_size, replace=False, p=prob)
                x_hat = x[idxes]
                y_hat = y[idxes]

            # Ignore samples with duplicate x/y coordinates
            if (len(x_hat) > len(np.unique(x_hat))):
                continue

            if (len(y_hat) > len(np.unique(y_hat))):
                continue

            # insert user given known pairs
            x_hat = np.concatenate((x_hat, self.pix))
            y_hat = np.concatenate((y_hat, self.wave))

            # Try to fit the data.
            # This doesn't need to be robust, it's an exact fit.
            fit_coeffs = self.polyfit(x_hat, y_hat, polydeg)

            # Discard out-of-bounds fits
            if ((fit_coeffs[-1] < self.min_intercept) |
                (fit_coeffs[-1] > self.max_intercept) |
                (fit_coeffs[-2] < self.min_slope) |
                (fit_coeffs[-2] > self.max_slope)):
                continue

            max_wavelength = self.polyval(fit_coeffs, max(x))
            if max_wavelength > self.max_wavelength:
                continue

            # M-SAC Estimator (Torr and Zisserman, 1996)
            fit = self.polyval(fit_coeffs, x)
            err = np.abs(fit - y)
            err[err > thresh] = thresh
            cost = sum(err)

            # reject lines outside the rms limit (thresh)
            best_mask = err < thresh
            n_inliers = sum(best_mask)

            # Want the most inliers with the lowest error
            if cost <= best_cost:

                # Now we do a robust fit
                best_p = models.robust_polyfit(x[best_mask], y[best_mask], polydeg)
                best_cost = cost

                # Get the residual of the fit
                err = np.abs(self.polyval(best_p, x[best_mask]) - y[best_mask])
                err[err > thresh] = thresh
                #best_cost = sum(err)
                best_err = np.sqrt(np.mean(err**2))
                best_inliers = n_inliers

                if tdqm_imported & progress:
                    sampler_list.set_description("Most inliers: {:d}, best error: {:1.2f}".format(n_inliers, best_err))

                # Perfect fit, break early
                if best_inliers == len(x):
                    break

        # Overfit check
        if best_inliers == polydeg + 1:
            valid_solution = False
        else:
            valid_solution = True

        return (best_p, best_err, best_inliers, valid_solution)

    def _get_best_model(self, candidates, polydeg, sample_size, max_tries,
                        thresh, brute_force, coeff, progress):
        '''
        Parameters
        ----------
        candidates :
        polydeg :
        sample_size : int
            Number of lines to be fitted.
        max_tries : int
            Number of trials of polynomial fitting.
        thresh :

        brute_force : tuple
            Solve all pixel-wavelength combinations with set to True.
        coeff : None or 1D numpy array
            Intial polynomial fit coefficient
        progress : tuple
            Show the progress bar with tdqm if set to True.
        Return
        ------
        coeff : list
            List of best fit polynomial coefficient.

        '''

        x, y = self._combine_linear_estimates(candidates)

        p, err, n_inliers, valid = self._solve_candidate_ransac(
                x,
                y,
                polydeg=polydeg,
                sample_size=sample_size,
                max_tries=max_tries,
                thresh=thresh,
                brute_force=brute_force,
                coeff=coeff,
                progress=progress)

        if not self.silence:
            if not valid:
                warnings.warn("Invalid fit")

            if err > self.fit_tolerance:
                warnings.warn("Error too large {} > {}".format(err, self.fit_tolerance))

        assert(p is not None), "Couldn't fit"

        return p

    def use_plotly(self):
        if plt:
            self.matplotlib_imported = False
            self.plotly_imported = True
        else:
            warnings.warn(
                'plotly package is not available.')

    def use_matplotlib(self):
        if plt:
            self.matplotlib_imported = True
            self.plotly_imported = False
        else:
            warnings.warn(
                'matplotlib package is not available.')

    def list_arc_library(self):
        print(self.elements)

    def _get_atlas(self, elements, min_wavelength, max_wavelength, min_intensity, min_distance):
        self.atlas_elements, self.atlas, self.atlas_intensities = load_calibration_lines(elements,
                                                    min_wavelength,
                                                    max_wavelength)

    def clear_calibration_lines(self):
        self.atlas = None

    def append_calibration_lines(self, lines):
        self.atlas = np.concatenate(self.atlas, lines)

    def load_user_calibration_lines(self, lines):
        self.atlas = lines

    def set_fit_constraints(self,
                            n_pix=1024,
                            num_slopes=1000,
                            range_tolerance=500,
                            fit_tolerance=20.,
                            polydeg=4,
                            candidate_thresh=15.,
                            ransac_thresh=10,
                            xbins=50,
                            ybins=50,
                            brute_force=False,
                            fittype='poly'):
        '''
        Parameters
        ----------
        range_tolerance: float
            Estimation of the error on the provided spectral range
            e.g. 3000-5000 with tolerance 500 will search for
            solutions that may satisfy 2500-5500
        
        fit_tolerance : float

        polydeg : int
            Degree of the polynomial fit.
        candidate_thresh : float
            Threshold for considering a point to be an inlier during candidate peak/line
            selection. Don't make this too small, it should allow for the error between
            a linear and non-linear fit.
        ransac_thresh: float
            The distance criteria to be considered an inlier to a fit. This should be close
            to the size of the expected residuals on the final fit (e.g. 10-20A maximum)
        xbins : int
        
        ybins : int

        brute_force : tuple
            Set to True to try all possible combination in the given parameter space
        fittype : string
            One of 'poly', 'legendre' or 'chebyshev'

        '''
        self.n_pix = n_pix
        self.num_slopes = num_slopes

        self.range_tolerance = range_tolerance
        self.min_intercept = self.min_wavelength - self.range_tolerance
        self.max_intercept = self.min_wavelength + self.range_tolerance

        self.min_slope = (self.max_wavelength - self.range_tolerance - self.max_intercept) / self.n_pix
        self.max_slope = (self.max_wavelength + self.range_tolerance - self.min_intercept) / self.n_pix
        
        self.fit_tolerance = fit_tolerance
        self.polydeg = polydeg
        self.ransac_thresh = ransac_thresh
        self.candidate_thresh = candidate_thresh
        self.xbins = xbins
        self.ybins = ybins
        self.brute_force = brute_force
        if fittype == 'poly':
            self.polyfit = np.polyfit
            self.polyval = np.polyval
        elif fittype == 'legendre':
            self.polyfit = np.polynomial.legendre.legfit
            self.polyval = np.polynomial.legendre.legval
        elif fittype == 'chebyshev':
            self.polyfit = np.polynomial.chebyshev.chebfit
            self.polyval = np.polynomial.chebyshev.chebval
        else:
            raise ValueError(
                'fittype must be: (1) poly, (2) legendre or (3) chebyshev')

    def set_guess_pairs(self, pix_guess=(), wave_guess=(), margin=5.):
        '''
        Provide manual pixel-to-wavelength mapping, good guess values with a margin
        of error.

        Parameters
        ----------
        pix_guess : numeric value, list or numpy 1D array (N)
            Any pixel value; can be outside the detector chip and
            serve purely as anchor points.
        wave_guess : numeric value, list or numpy 1D array (N)
            The matching wavelength for each of the pix_guess.
        margin : float
            Tolerance in the wavelength value of the pixel-to-wavelength mappping.

        '''

        self.pix_guess = np.asarray(pix_guess, dtype='float')
        self.wave_guess = np.asarray(wave_guess, dtype='float')
        self.margin = margin

    def set_known_pairs(self, pix=(), wave=()):
        '''
        Provide manual pixel-to-wavelength mapping, fixed values in the fit.
        Use with caution because it can completely skew or bias the fit.

        Parameters
        ----------
        pix : numeric value, list or numpy 1D array (N)
            Any pixel value, can be outside the detector chip and
            serve purely as anchor points.
        wave : numeric value, list or numpy 1D array (N)
            The matching wavelength for each of the pix.

        '''

        self.pix = np.asarray(pix, dtype='float')
        self.wave = np.asarray(wave, dtype='float')

    def fit(self, sample_size=5, max_tries=1000, top_n=20, n_slope=3000,
            mode='manual', progress=True, coeff=None):
        '''
        Parameters
        ----------
        sample_size : int (default: 5)

        max_tries : int (default: 1000)

        top_n : int (default: 10)

        mode : String
            Preset modes 'fast', 'normal', 'slow' and 'veryslow'. They override
            supplied sample_size, max_tries and top_n. 'slow' should be fairly
            reliable.
        progress : tuple
            True to show progress with tdqm. It is overrid if tdqm cannot be
            imported.

        '''

        # Presets for fast, normal, slow and veryslow
        if mode=='fast':
            sample_size = 3
            max_tries = 100
            top_n = 50
            n_slope = 500
        elif mode=='normal':
            sample_size = 5
            max_tries = 500
            top_n = 100
            n_slope = 1000
        elif mode=='slow':
            sample_size = 5
            max_tries = 1000
            top_n = 100
            n_slope = 5000
        elif mode=='manual':
            pass
        else:
            raise NameError('Unknown mode. Please choose from '
                '(1) fast,'
                '(2) normal,'
                '(3) slow, or'
                '(4) normal')

        if sample_size > len(self.atlas):
            sample_size = len(self.atlas)

        # Generate the accumulator from the pairs
        self.accumulator = self._hough_points(
            self.pairs[:, 0], self.pairs[:, 1], num_slopes=n_slope)

        # Get the line coeffients from the promising bins in the accumulator
        _, lines = self._get_top_lines(
            self.accumulator, top_n=top_n, xbins=self.xbins, ybins=self.ybins)

        # Locate candidate points for these lines fits
        self.candidates = []
        for line in lines:
            m, c = line
            inliers_x, inliers_y = self._get_candidate_points_linear(
                m, c, self.candidate_thresh)
            self.candidates.append((inliers_x, inliers_y))

        return self._get_best_model(self.candidates, self.polydeg, sample_size,
                                    max_tries, self.ransac_thresh, self.brute_force,
                                    coeff, progress)

    def match_peaks_to_atlas(self, fit, tolerance=1., polydeg=5):
        '''
        Fitting all the detected peaks with the given polynomail solution for
        a fit using maximal information.

        Parameters
        ----------
        fit : list
            List of polynomial fit coefficients
        tolerance : float (default: 0.5)
            Absolute difference between fit and model in the unit of nm.
        polydeg : int (default: 4)
            Order of polynomial fit with all the detected peaks

        Return
        ------
        coeff : list
            List of best fit polynomial coefficient.

        '''

        x_match = []
        y_match = []

        for p in self.peaks:
            x = self.polyval(fit, p)
            diff = np.abs(self.atlas - x)
            idx = np.argmin(diff)

            if diff[idx] < tolerance:
                x_match.append(p)
                y_match.append(self.atlas[idx])

        x_match = np.array(x_match)
        y_match = np.array(y_match)
        
        coeff = models.robust_polyfit(x_match, y_match, polydeg)
        return coeff, x_match, y_match


    def plot_fit(self, spectrum, fit, tolerance=5., plot_atlas=True, silence=True,
                 output_filename=None, verbose=False, renderer='default', log_spectrum=True):
        '''
        Parameters
        ----------
        spectrum : 1D numpy array (N)
            Array of length N pixels 
        fit : 
            Best fit polynomail coefficients
        tolerance : float (default: 0.5)
            Absolute difference between model and fitted wavelengths in unit of nm.

        '''

        if log_spectrum:
            spectrum = np.log(spectrum)

        if matplotlib_imported:

            pix = np.arange(len(spectrum)).astype('float')
            wave = self.polyval(fit, pix)

            fig, (ax1, ax2, ax3) = plt.subplots(
                nrows=3,
                sharex=True,
                gridspec_kw={'hspace': 0.},
                figsize=(15, 9))
            fig.tight_layout()

            # Plot fitted spectrum
            ax1.plot(wave, spectrum)
            ax1.vlines(
                self.polyval(fit, self.peaks),
                spectrum[self.peaks.astype('int')],
                spectrum.max() * 1.05,
                linestyles='dashed',
                colors='C1')
            
            # Plot the atlas
            if plot_atlas:
                #spec = SyntheticSpectrum(fit, model_type='poly', degree=len(fit)-1)
                #x_locs = spec.get_pixels(self.atlas)
                ax1.vlines(
                    self.atlas,
                    0,
                    spectrum.max() * 1.05,
                    colors='C2')

            fitted_peaks = []
            fitted_diff = []
            all_diff = []
            for p in self.peaks:
                x = self.polyval(fit, p)
                diff = self.atlas - x
                idx = np.argmin(np.abs(diff))
                all_diff.append(diff[idx])

                if not silence: 
                    print("Peak at: {} A".format(x))

                if np.abs(diff[idx]) < tolerance:
                    fitted_peaks.append(p)
                    fitted_diff.append(diff[idx])
                    if not silence:
                        print("- matched to {} A".format(self.atlas[idx]))
                    ax1.vlines(
                        self.polyval(fit, p),
                        spectrum[p.astype('int')],
                        spectrum.max() * 1.05,
                        colors='C1')

                    ax1.text(
                        x - 3,
                        0.8 * max(spectrum),
                        s="{:1.2f}".format(self.atlas[idx]),
                        rotation=90,
                        bbox=dict(facecolor='white', alpha=1))

            rms = np.sqrt(np.mean(np.array(fitted_diff)**2.))

            ax1.grid(linestyle=':')
            ax1.set_ylabel('ADU')
            if log_spectrum:
                ax1.set_ylim(0, spectrum.max() * 1.05)
            else:
                ax1.set_ylim(spectrum.min(), spectrum.max() * 1.05)

            # Plot the residuals
            ax2.scatter(
                self.polyval(fit, fitted_peaks),
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
            ax3.scatter(
                self.polyval(fit, fitted_peaks),
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

            if output_filename is not None:
                fig.savefig(output_filename)

        elif self.plotly_imported:

            pix = np.arange(len(spectrum)).astype('float')
            wave = self.polyval(fit, pix)

            fig = go.Figure()

            # Top plot - arc spectrum and matched peaks
            fig.add_trace(
                go.Scatter(x=wave,
                           y=spectrum,
                           mode='lines',
                           line=dict(color='royalblue'),
                           yaxis='y3'))

            spec_max = spectrum.max() * 1.05

            p_x = []
            p_y = []
            for i, p in enumerate(self.peaks):
                p_x.append(self.polyval(fit, p))
                p_y.append(spectrum[int(p)])

            fig.add_trace(
                go.Scatter(x=p_x,
                           y=p_y,
                           mode='markers',
                           marker=dict(color='firebrick'),
                           yaxis='y3'))

            fitted_peaks = []
            fitted_diff = []
            all_diff = []

            for p in self.peaks:
                x = self.polyval(fit, p)
                diff = self.atlas - x
                idx = np.argmin(np.abs(diff))
                all_diff.append(diff[idx])

                if not silence: 
                    print("Peak at: {} A".format(x))

                if np.abs(diff[idx]) < tolerance:
                    fitted_peaks.append(p)
                    fitted_diff.append(diff[idx])
                    if not silence:
                        print("- matched to {} A".format(self.atlas[idx]))

                    p_x_matched = []
                    p_y_matched = []
                    for i, p in enumerate(self.peaks):
                        p_x_matched.append(self.polyval(fit, p))
                        p_y_matched.append(spectrum[int(p)])
                    fig.add_trace(
                        go.Scatter(x=p_x_matched,
                                   y=p_y_matched,
                                   mode='markers',
                                   marker=dict(color='orange'),
                                   name='Matched peaks',
                                   yaxis='y3'
                                   )
                                )

            # Middle plot - Residual plot
            rms = np.sqrt(np.mean(np.array(fitted_diff)**2.))
            x_fitted = self.polyval(fit, fitted_peaks)
            fig.add_trace(
                go.Scatter(x=x_fitted,
                           y=fitted_diff,
                           mode='markers',
                           marker=dict(color='orange'),
                           yaxis='y2')
                )
            fig.add_trace(
                go.Scatter(x=[wave.min(), wave.max()],
                           y=[0, 0],
                           mode='lines',
                           line=dict(color='royalblue', dash='dash'),
                           yaxis='y2'
                           )
                )

            # Bottom plot - Polynomial fit for Pixel to Wavelength
            fig.add_trace(
                go.Scatter(x=x_fitted,
                           y=fitted_peaks,
                           mode='markers',
                           marker=dict(color='orange'),
                           yaxis='y1',
                           name='Peaks used for fitting')
                )
            fig.add_trace(
                go.Scatter(x=wave,
                           y=pix,
                           mode='lines',
                           line=dict(color='royalblue'),
                           yaxis='y1')
                )

            # Layout, Title, Grid config
            fig.update_layout(autosize=True,
                              yaxis3=dict(title='ADU',
                                         range=[np.log10(np.percentile(spectrum,10)), np.log10(spec_max)],
                                         domain=[0.67,1.0],
                                         showgrid=True,
                                         type='log'
                                        ),
                              yaxis2=dict(title='Residual / A',
                                          range=[min(fitted_diff), max(fitted_diff)],
                                          domain=[0.33,0.66],
                                          showgrid=True
                                         ),
                              yaxis=dict(title='Pixel',
                                          range=[0., max(pix)],
                                          domain=[0.,0.32],
                                          showgrid=True
                                         ),
                              xaxis=dict(title='Wavelength / A',
                                         zeroline=False,
                                         range=[min(wave), max(wave)],
                                         showgrid=True,
                                         ),
                              hovermode='closest',
                              showlegend=False,
                              height=800,
                              width=1000
                              )

            if verbose:
                return fig.to_json()
            if renderer == 'default':
                fig.show()
            else:
                fig.show(renderer)

        else:

            assert(self.matplotlib_imported), ('matplotlib package not available. ' +
                'Plot cannot be generated.')
            assert(self.plotly_imported), ('plotly package not available. ' +
                'Plot cannot be generated.')
