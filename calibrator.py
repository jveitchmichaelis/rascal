import itertools
import numpy as np
import astropy.units as u
import peakutils
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

def getPeaks(data, thres=0.1, min_dist=40, obs_range=(400,850)):
    
    x = np.arange(len(data))
    x_approx = np.linspace(obs_range[0], obs_range[1], len(x))
    y = data
    
    idx = peakutils.indexes(y, thres, min_dist)

    peaks_x = []
    peaks_approx = []
    for i in idx:
        try:
            peaks_x.append(peakutils.interpolate(x, y, ind=[i]))
            peaks_approx.append(peakutils.interpolate(x_approx, y, ind=[i]))
        except:
            pass
    
    return np.array(peaks_x).flatten(), np.array(peaks_approx).flatten()

def polyfit_value(x, p):
    return np.sum([p[n]*np.array(x).astype('float')**n for n in range(len(p))], axis=0)

class Calibrator:
    def __init__(self, peaks, atlas):
        self.peaks = peaks
        self.atlas = atlas

        self._generate_pairs()

        # It's unlikely we'll be out by more than half a nanometre.
        self.fit_tolerance = 0.5
        self.inlier_threshold = 1

        self.set_fit_constraints()

    def fit(self, top_n=5):

        self.accumulator = self._hough_points(self.pairs[:,0], self.pairs[:,1])
        
        h, lines = self._get_top_lines(self.accumulator, bins=100, top_n=top_n)

        self.candidates = []
        for line in lines:
            m, c = line
            inliers_x, inliers_y = self._get_candidate_points(m, c)
            self.candidates.append((inliers_x, inliers_y))

        return self._get_best_model(self.candidates)
        
    def set_fit_constraints(self, min_slope=0.1, max_slope=2, min_intercept=200, max_intercept=500, line_fit_thresh=5, fit_tolerance=0.5, inlier_tolerance=1):
        self.min_slope = min_slope
        self.max_slope = max_slope
        self.min_intercept = min_intercept
        self.max_intercept = max_intercept
        self.line_fit_thresh = line_fit_thresh
        self.fit_tolerance = fit_tolerance
        self.inlier_tolerance = inlier_tolerance
    
    def _generate_pairs(self):
        self.pairs = np.array([pair for pair in itertools.product(self.peaks,
                                                         self.atlas)])

    def _hough_points(self, x, y, num_slopes=3600):
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
            intercepts = y - m*x

            for c in intercepts:
                # Initial wavelength unlikely to be below 150 nm
                if c < self.min_intercept:
                    continue
                    
                if c > self.max_intercept:
                    continue

                accumulator.append((m, c))

        return np.array(accumulator)

    def _get_top_lines(self, accumulator, bins=30, top_n=5):
        h, xedges, yedges = np.histogram2d(accumulator[:,0], accumulator[:,1], bins=bins)

        xbin_width = (xedges[1]-xedges[0])/2
        ybin_width = (yedges[1]-yedges[0])/2

        top_bins = np.dstack(np.unravel_index(np.argsort(h.ravel())[::-1][:top_n], h.shape))[0]
        
        lines = []
        for b in top_bins:
            lines.append((xedges[b[0]]+xbin_width, yedges[b[1]]+ybin_width))
        
        return h, lines

    def _get_candidate_points(self, m, c, thresh=5):
    
        predicted = (m*self.pairs[:,0]+c)
        actual = self.pairs[:,1]
        err = np.abs(predicted - actual)
        
        return self.pairs[:,0][err < thresh], self.pairs[:,1][err < thresh]

    def _solve_candidate_ransac(self, x, y, polydeg=3, thresh=1, brute_force=False):

        valid_solution = False
        best_inliers = [False]
        best_p = None
        best_err = 1e50
        max_tries = 3e4
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
        for sample in tqdm(sampler):

            if brute_force:
                x_hat = x[[sample]]
                y_hat = y[[sample]]
            else:
                idxes = np.random.choice(idx, sample_size, replace=False)
                x_hat = x[idxes]
                y_hat = y[idxes]
            
            # Ignore samples with duplicate x/y coordinates
            if(len(x_hat) > len(np.unique(x_hat))):
                continue
            
            if(len(y_hat) > len(np.unique(y_hat))):
                continue
                
            # Try to fit the data
            fit_coeffs = np.polyfit(x_hat, y_hat, polydeg)
            
            # Discard out-of-bounds fits
            if(fit_coeffs[-1] < self.min_intercept):
                continue
                
            if(fit_coeffs[-1] > self.max_intercept):
                continue
            
            if(fit_coeffs[-2] < self.min_slope):
                continue
            
            if(fit_coeffs[-2] > self.max_slope):
                continue
            
            # Count inliers by absolute distance
            err = np.abs(polyfit_value(x, fit_coeffs[::-1]) - y)
            inliers = err < thresh
            
            # Want the most inliers with the lowest error
            if sum(inliers) >= sum(best_inliers):
                best_inliers = inliers

                best_p = np.polyfit(x[best_inliers], y[best_inliers], polydeg)
                err = np.abs(polyfit_value(x[best_inliers], fit_coeffs[::-1]) - y[best_inliers])
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
    
    def _get_best_model(self, candidates):

        best_inliers = 0
        best_p = None
        best_err = 1e10
        
        for candidate in tqdm(candidates):
            x, y = candidate
            p, err, n_inliers, valid = self._solve_candidate_ransac(x, y,
                                                    thresh=self.inlier_threshold)

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

    def match_peaks_to_atlas(self, fit, tolerance=0.5):

        x_match = []
        y_match = []

        for p in self.peaks:
            x = polyfit_value(p, fit[::-1])
            diff = np.abs(self.atlas - x)
            idx = np.argmin(diff)

            if diff[idx] < tolerance:
                x_match.append(p)
                y_match.append(self.atlas[idx])
        
        return np.polyfit(x_match, y_match, 4)


    def plot_fit(self, spectrum, fit, tolerance=0.5):
        plt.figure(figsize=(16,8))
        plt.plot(polyfit_value(np.arange(len(spectrum)).astype('float'), fit[::-1]), spectrum)
        plt.vlines(polyfit_value(self.peaks, fit[::-1]), spectrum[self.peaks.astype('int')], spectrum.max(), colors='red', alpha=0.5)

        for p in self.peaks:
            x = polyfit_value(p, fit[::-1])
            diff = np.abs(self.atlas - x)
            idx = np.argmin(diff)

            print("Peak at: {} nm".format(x))

            if diff[idx] < tolerance:
                print("- matched to {} nm".format(self.atlas[idx]))
                plt.text(x-3, 0.8*max(spectrum), s="{:1.2f}".format(self.atlas[idx]),rotation=90, bbox=dict(facecolor='white', alpha=1))
        
        plt.show()