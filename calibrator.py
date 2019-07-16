import warnings
import itertools
import numpy as np
import astropy.units as u
try:
    import matplotlib.pyplot as plt
    matplotlib_imported = True
except:
    warnings.warn('matplotlib package not available. Plot cannot be generated.')
    matplotlib_imported = False
try:
    from tqdm.autonotebook import tqdm
    tdqm_imported = True
except:
    warnings.warn('tqdm package not available. Progress bar will not be shown.')
    tdqm_imported = False


class Calibrator:
    def __init__(self, peaks, atlas):
        self.peaks = peaks
        self.atlas = atlas

        self._generate_pairs()

        # It's unlikely we'll be out by more than half a nanometre.
        self.fit_tolerance = 0.5
        self.inlier_threshold = 1

        self.set_fit_constraints()

    def _generate_pairs(self):
        self.pairs = np.array(
            [pair for pair in itertools.product(self.peaks, self.atlas)])

    def _hough_points(self, x, y, num_slopes):
        """
        Calculate the Hough transform for a set of input points.
        
        Returns the 2D Hough accumulator matrix. The x-axis represents slope, the
        y-axis represents intercept. Vertical lines (infinite gradient)
        are not accommodated.
        """
        # Build up the accumulator
        accumulator = []

        for m in np.linspace(self.min_slope, self.max_slope, num_slopes):
            # y = mx+c -> (y-mx) = c
            intercepts = y - m * x

            for c in intercepts:
                # Initial wavelength unlikely to be below 150 nm
                if c < self.min_intercept:
                    continue

                if c > self.max_intercept:
                    continue

                accumulator.append((m, c))

        return np.array(accumulator)

    def _get_top_lines(self, accumulator, top_n, xbins, ybins):
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

        predicted = (m * self.pairs[:, 0] + c)
        actual = self.pairs[:, 1]
        err = np.abs(predicted - actual)

        return self.pairs[:, 0][err < thresh], self.pairs[:, 1][err < thresh]

    def _solve_candidate_ransac(self,
                                x,
                                y,
                                polydeg,
                                sample_size,
                                max_tries,
                                thresh,
                                brute_force):

        valid_solution = False
        best_inliers = [False]
        best_p = None
        best_cost = 1e50
        best_err = 1e50
        if sample_size <= polydeg:
            sample_size = polydeg + 1

        x = np.array(x)
        y = np.array(y)

        if len(np.unique(x)) <= polydeg:
            return (best_p, best_err, sum(best_inliers), False)

        idx = range(len(x))

        if brute_force:
            sampler = itertools.combinations(idx, sample_size)
        else:
            sampler = range(int(max_tries))

        # Brute force check all combinations. N choose 4 is pretty fast.
        if tdqm_imported:
            sampler_list = tqdm(sampler)
        else:
            sampler_list = sampler
        for sample in sampler_list:

            if brute_force:
                x_hat = x[[sample]]
                y_hat = y[[sample]]
            else:
                idxes = np.random.choice(idx, sample_size, replace=False)
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

            # Discard out-of-bounds fits
            if (fit_coeffs[-1] < self.min_intercept):
                continue

            if (fit_coeffs[-1] > self.max_intercept):
                continue

            if (fit_coeffs[-2] < self.min_slope):
                continue

            if (fit_coeffs[-2] > self.max_slope):
                continue

            # M-SAC Estimator (Torr and Zisserman, 1996)
            err = np.abs(self.polyval(fit_coeffs, x) - y)
            err[err > thresh] = thresh
            cost = sum(err)
            inliers = err < thresh

            # Want the most inliers with the lowest error
            if cost <= best_cost:
                best_inliers = inliers
                best_cost = cost
                
                best_p = self.polyfit(x[best_inliers], y[best_inliers], polydeg)
                err = np.abs(self.polyval(best_p, x[best_inliers]) - y[best_inliers])
                best_err = err.mean()

                # Perfect fit, break early
                if sum(best_inliers) == len(x):
                    break
            
        # Overfit check
        if sum(best_inliers) == polydeg + 1:
            valid_solution = False
        else:
            valid_solution = True

        print(best_p)

        return (best_p, best_err, sum(best_inliers), valid_solution)

    def _get_best_model(self, candidates, polydeg, sample_size, max_tries, thresh, brute_force):

        best_inliers = 0
        best_p = None
        best_err = 1e10
        if tdqm_imported:
            candidate_list = tqdm(candidates)
        else:
            candidate_list = candidates
        for candidate in candidate_list:
            x, y = candidate
            p, err, n_inliers, valid = self._solve_candidate_ransac(
                x, y, polydeg=polydeg, sample_size=sample_size, max_tries=max_tries, thresh=self.inlier_threshold, brute_force=brute_force)

            if valid == False:
                print("Invalid fit")
                continue

            if err > self.fit_tolerance:
                print("Error too large")
                continue

            if n_inliers >= best_inliers:
                # Accept a better fit if we have equal numbers of inliers
                if n_inliers == best_inliers and err >= best_err:
                    continue
                else:
                    best_p = p
                    best_inliers = n_inliers
                    best_err = err

        if best_p is None:
            print("Couldn't fit")

        return best_p

    def set_fit_constraints(self,
                            min_slope=0.1,
                            max_slope=2,
                            min_intercept=200,
                            max_intercept=500,
                            line_fit_thresh=5,
                            fit_tolerance=0.5,
                            inlier_tolerance=1,
                            polydeg=3,
                            thresh=10,
                            xbins=100,
                            ybins=100,
                            brute_force=False,
                            fittype='poly'):
        self.min_slope = min_slope
        self.max_slope = max_slope
        self.min_intercept = min_intercept
        self.max_intercept = max_intercept
        self.line_fit_thresh = line_fit_thresh
        self.fit_tolerance = fit_tolerance
        self.inlier_tolerance = inlier_tolerance
        self.polydeg = polydeg
        self.thresh = thresh
        self.xbins = xbins
        self.ybins = ybins
        self.brute_force = brute_force
        if fittype=='poly':
            self.polyfit = np.polyfit
            self.polyval = np.polyval
        elif fittype=='legendre':
            self.polyfit = np.polynomial.legendre.legfit
            self.polyval = np.polynomial.legendre.legval
        elif fittype=='chebyshev':
            self.polyfit = np.polynomial.chebyshev.chebfit
            self.polyval = np.polynomial.chebyshev.chebval
        else:
            raise NameError('fittype must be: (1) poly, (2) legendre or (3) chebyshev')

    def set_guess_pairs(self, pix=(), wave=(), margin=5):
        '''
        Provide manual pixel-to-wavelength mapping, good guess values with a margin
        of error.

        Not currently used.
        '''
        self.pix_guess = np.asarray(pix_guess, dtype='float')
        self.wave_guess = np.asarray(wave_guess, dtype='float')

    def set_known_pairs(self, pix=(), wave=()):
        '''
        Provide manual pixel-to-wavelength mapping, fixed values in the fit.
        '''
        self.pix = np.asarray(pix, dtype='float')
        self.wave = np.asarray(wave, dtype='float')

    def fit(self, sample_size=5, max_tries=1e5, top_n=5):

        self.accumulator = self._hough_points(self.pairs[:, 0],
                                              self.pairs[:, 1],
                                              num_slopes=3600)

        h, lines = self._get_top_lines(self.accumulator, top_n=top_n, xbins=self.xbins, ybins=self.ybins)

        self.candidates = []
        for line in lines:
            m, c = line
            inliers_x, inliers_y = self._get_candidate_points(m, c, self.thresh)
            self.candidates.append((inliers_x, inliers_y))

        return self._get_best_model(self.candidates, self.polydeg, sample_size, max_tries, self.thresh, self.brute_force)

    def match_peaks_to_atlas(self, fit, tolerance=0.5, polydeg=4):

        x_match = []
        y_match = []

        for p in self.peaks:
            x = self.polyval(fit, p)
            diff = np.abs(self.atlas - x)
            idx = np.argmin(diff)

            if diff[idx] < tolerance:
                x_match.append(p)
                y_match.append(self.atlas[idx])

        return self.polyfit(x_match, y_match, polydeg)

    def plot_fit(self, spectrum, fit, tolerance=0.5):
        if matplotlib_imported:

            pix = np.arange(len(spectrum)).astype('float')
            wave = self.polyval(fit, pix)

            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, gridspec_kw={'hspace': 0.}, figsize=(10,15))
            fig.tight_layout()
        
            # Plot fitted spectrum
            ax1.plot(
                wave, spectrum)
            ax1.vlines(
                self.polyval(fit, self.peaks),
                spectrum[self.peaks.astype('int')],
                spectrum.max(),
                colors='C1',
                alpha=0.5)

            fitted_peaks = []
            fitted_diff = []
            all_diff = []
            for p in self.peaks:
                x = self.polyval(fit, p)
                diff = self.atlas - x
                idx = np.argmin(np.abs(diff))
                all_diff.append(diff[idx])
                print("Peak at: {} nm".format(x))

                if np.abs(diff[idx]) < tolerance:
                    fitted_peaks.append(p)
                    fitted_diff.append(diff[idx])
                    print("- matched to {} nm".format(self.atlas[idx]))
                    ax1.text(
                        x - 3,
                        0.8 * max(spectrum),
                        s="{:1.2f}".format(self.atlas[idx]),
                        rotation=90,
                        bbox=dict(facecolor='white', alpha=1))
            ax1.grid(linestyle=':')
            ax1.set_ylabel('ADU')

            # Plot residuals
            ax2.scatter(self.polyval(fit, fitted_peaks), fitted_diff, marker='+', color='C1', label='Peaks used for fitting')
            ax2.grid(linestyle=':')
            ax2.set_xlabel('Wavelength / nm')
            ax2.set_ylabel('Residual / nm')
            ax2.legend(loc='lower right')

            # Plot polynomial
            ax3.scatter(self.polyval(fit, fitted_peaks), fitted_peaks, marker='+', color='C1', label='Peaks used for fitting')
            ax3.plot(wave, pix)
            ax3.grid(linestyle=':')
            ax3.set_ylabel('Pixel')
            ax3.legend(loc='lower right')

            plt.show()
