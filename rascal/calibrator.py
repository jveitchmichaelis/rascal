import warnings
import itertools
import numpy as np
import astropy.units as u
try:
    import matplotlib.pyplot as plt
    matplotlib_imported = True
except:
    warnings.warn(
        'matplotlib package not available. Plot cannot be generated.')
    matplotlib_imported = False
try:
    from tqdm.autonotebook import tqdm
    tdqm_imported = True
except:
    warnings.warn(
        'tqdm package not available. Progress bar will not be shown.')
    tdqm_imported = False


class Calibrator:
    def __init__(self, peaks, silence=False, **kwargs):
        self.peaks = peaks
        self.silence = silence
        self.elements = ["Hg", "Ar", "Xe", "CuNeAr", "Kr"]

        self.load_calibration_lines(**kwargs)

        # Create a list of all possible pairs of detected peaks and lines from atlas 
        self._generate_pairs()

        # Configuring fitting constraints
        self.set_fit_constraints()

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

        h, xedges, yedges = np.histogram2d(
            accumulator[:, 0], accumulator[:, 1], bins=(xbins, ybins))

        xbin_width = (xedges[1] - xedges[0]) / 2
        ybin_width = (yedges[1] - yedges[0]) / 2

        top_bins = np.dstack(
            np.unravel_index(np.argsort(h.ravel())[::-1][:top_n], h.shape))[0]

        lines = []
        for b in top_bins:
            lines.append((xedges[b[0]] + xbin_width,
                          yedges[b[1]] + ybin_width))

        return h, lines

    def _get_candidate_points(self, m, c, thresh):
        '''
        

        '''

        predicted = (m * self.pairs[:, 0] + c)
        actual = self.pairs[:, 1]
        err = np.abs(predicted - actual)
        mask = (err < thresh)

        return self.pairs[:, 0][mask], actual[mask]

    def _solve_candidate_ransac(self, x, y, polydeg, sample_size, max_tries,
                                thresh, brute_force, coeff, progress):
        '''
        Parameters
        ----------
        x : 1D numpy array
            Array of pixels from peak detection.
        y : 1D numpy array
            Array of wavlengths from atlas.
        polydeg : int
            The order of polynomial (the polynomial type is definted in the 
            set_fit_constraints).
        sample_size : int
            Number of lines to be fitted.
        max_tries : int
            Number of trials of polynomial fitting.
        thresh :

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
        n_best = sum(best_mask)

        if sample_size <= polydeg:
            sample_size = polydeg + 1

        x = np.array(x)
        y = np.array(y)

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

            # Try to fit the data
            fit_coeffs = self.polyfit(x_hat, y_hat, polydeg)

            '''
            # Discard out-of-bounds fits
            if ((fit_coeffs[-1] < self.min_intercept) |
                (fit_coeffs[-1] > self.max_intercept) |
                (fit_coeffs[-2] < self.min_slope) |
                (fit_coeffs[-2] > self.max_slope)):
                continue
            '''

            # M-SAC Estimator (Torr and Zisserman, 1996)
            fit = self.polyval(fit_coeffs, x)

            if max(fit) > self.max_wavelength:
                continue

            if min(fit) < self.min_wavelength:
                continue

            err = np.abs(fit - y)
            err[err > thresh] = thresh
            cost = sum(err)
            # reject lines outside the rms limit (thresh)
            best_mask = err < thresh
            n_best = sum(best_mask)

            # Want the most inliers with the lowest error
            if cost <= best_cost:

                # Get the best fit polynomial coefficient
                best_p = self.polyfit(x[best_mask], y[best_mask], polydeg)

                # Get the residual of the fit
                err = np.abs(self.polyval(best_p, x[best_mask]) - y[best_mask])
                best_cost = sum(err)
                best_err = np.sqrt(np.mean(err**.2))

                # Perfect fit, break early
                if n_best == len(x):
                    break

        # Overfit check
        if n_best == polydeg + 1:
            valid_solution = False
        else:
            valid_solution = True

        return (best_p, best_err, n_best, valid_solution)

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

        best_inliers = 0
        best_p = None
        best_err = 1e10
        if tdqm_imported & progress:
            candidate_list = tqdm(candidates)
        else:
            candidate_list = candidates

        for candidate in candidate_list:
            x, y = candidate
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

            if not silence:
                if not valid:
                    warnings.warn("Invalid fit")
                    continue

                if err > self.fit_tolerance:
                    warnings.warn("Error too large")
                    continue

            if n_inliers >= best_inliers:
                # Accept a better fit if we have equal numbers of inliers
                if n_inliers == best_inliers and err >= best_err:
                    continue
                else:
                    best_p = p
                    best_inliers = n_inliers
                    best_err = err

        assert(best_p is None), "Couldn't fit"

        return best_p

    def list_arc_library(self):
        print(self.elements)

    def load_calibration_lines(self,
                               elements=self.elements,
                               min_wavelength=1000.,
                               max_wavelength=10000.):
        '''
        https://apps.dtic.mil/dtic/tr/fulltext/u2/a105494.pdf
        '''
        cal_lines = np.loadtxt('calibration_lines.csv', delimiter=',', dtype='U', skiprows=1)
        wave = cal_lines[:, 0].astype('float')
        element = cal_lines[:, 1]
        # Get lines of the requested elements
        lines = wave[np.isin(element, elements)]
        # Get only lines within the requested wavelength
        mask = (lines > min_wavelength) * (lines < max_wavelength)
        self.atlas = lines[mask]
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength
        print(self.atlas)

    def clear_calibration_lines(self):
        self.atlas = None

    def append_calibration_lines(self, lines):
        self.atlas = np.concatenate(self.atlas, lines)

    def load_user_calibration_lines(self, lines):
        self.atlas = lines

    def set_fit_constraints(self,
                            n_pix=1024,
                            min_intercept=1000.,
                            max_intercept=6000.,
                            fit_tolerance=50.,
                            polydeg=5,
                            thresh=10.,
                            xbins=100,
                            ybins=100,
                            brute_force=False,
                            fittype='poly'):
        '''
        Parameters
        ----------
        min_intercept : float
            Minimum possible wavelength at pixel 0
        max_intercept : float
            Maximum possible wavelength at pixel 0
        
        fit_tolerance : float

        polydeg : int
            Degree of the polynomial fit.
        thresh : float
        
        xbins : int
        
        ybins : int

        brute_force : tuple
            Set to True to try all possible combination in the given parameter space
        fittype : string
            One of 'poly', 'legendre' or 'chebyshev'

        '''

        self.min_slope = (self.max_wavelength - max_intercept) / n_pix
        self.max_slope = (self.max_wavelength - min_intercept) / n_pix
        self.min_intercept = min_intercept
        self.max_intercept = max_intercept
        self.fit_tolerance = fit_tolerance
        self.polydeg = polydeg
        self.thresh = thresh
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

    def fit(self, sample_size=5, max_tries=1000, top_n=10, n_slope=3000,
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
            raise.NameError('Unknown mode. Please choose from '
                '(1) fast,'
                '(2) normal,'
                '(3) slow, or'
                '(4) normal')

        if sample_size > len(self.atlas):
            sample_size = len(self.atlas)

        # Generate the accumulator from the pairs
        self.accumulator = self._hough_points(
            self.pairs[:, 0], self.pairs[:, 1], num_slopes=n_slope)

        # ?
        h, lines = self._get_top_lines(
            self.accumulator, top_n=top_n, xbins=self.xbins, ybins=self.ybins)

        self.candidates = []
        for line in lines:
            m, c = line
            inliers_x, inliers_y = self._get_candidate_points(
                m, c, self.thresh)
            self.candidates.append((inliers_x, inliers_y))

        return self._get_best_model(self.candidates, self.polydeg, sample_size,
                                    max_tries, self.thresh, self.brute_force,
                                    coeff, progress)

    def match_peaks_to_atlas(self, fit, tolerance=5., polydeg=5.):
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

        coeff = self.polyfit(x_match, y_match, polydeg)
        return coeff, x_match, y_match

    def plot_fit(self, spectrum, fit, tolerance=5.):
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

            fitted_peaks = []
            fitted_diff = []
            all_diff = []
            for p in self.peaks:
                x = self.polyval(fit, p)
                diff = self.atlas - x
                idx = np.argmin(np.abs(diff))
                all_diff.append(diff[idx])

                print("Peak at: {} A".format(x))

                if np.abs(diff[idx]) < tolerance:
                    fitted_peaks.append(p)
                    fitted_diff.append(diff[idx])
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
        else:
            assert(matplotlib_imported), 'matplotlib package not available. Plot cannot be generated.'
