Random sample consensus (RANSAC)
================================

RANSAC is an algorithm for robust fitting, popularised by the computer vision community `(Fischler and Bolles, 1981) <https://dl.acm.org/doi/10.1145/358669.358692>`_ .

Overview
--------

Suppose we have a dataset that contains good points (inliers) and spurious points (outliers). If we randomly sample our dataset, eventually we will pick a set of points which only contain inliers. In that case, a model fit to this sample should also fit the majority of the rest of our data. Conversely if we sample an outlier by mistake, the fit will not agree with the rest of the dataset.

This is conceptually very simple, but works extremely well in practice. Typically the sample size is the minimum required to fit the model, so for a linear model we would draw 2 random points. If you know the percentage of outliers in your data it is possible to calculate some statistical estimate of how many iterations is required before an inlier-only sample is drawn (to some degree of confidence). In practice we never know the inlier:outlier ratio and we just try "a lot" of samples (e.g. 500).

We also need to decide how to score a particular sample fit. In the original version of RANSAC, the number if inliers corresponding to a fit is used. More inliers equals a better fit. In RASCAL we use a slight modification called M-SAC `(Torr and Zisserman, 1996) <https://www.sciencedirect.com/science/article/abs/pii/S1077314299908329>`_ which also weights inliers by the fit error. This is useful because it acts as a tie-breaker between two fits with the same number of inliers.

For more information, you can read the `Wikipedia article <https://en.wikipedia.org/wiki/Random_sample_consensus>`_ .

User configurable options
-------------------------

If you find that you're not getting good fits and feel that more iterations would help, you can try that. Simply increase ``max_tries`` when you call ``Calibrator.fit()``. As a first pass, try increasing by an order of magnitude to see if things improve, if that works then you can lower the number of iterations until you reliably get a good fit every time.

You can also adjust the threshold that RANSAC uses to define inliers (``ransac_tolerance``, in Angstrom). This should normally be set quite small, because for most science-grade instruments we expect a very good model fit. If you think there is likely to be a lot of jitter in your peak finding, or your spectrometer is very low resolution, then you might want to relax this.

Other settings can be twiddled in ``Calibrator.set_ransac_properties()``, but for the most part you shouldn't need to adjust these and they are handled internally by the Calibrator.
