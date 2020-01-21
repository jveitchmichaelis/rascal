Hough Transform
===============

Hough transform is a technique used for feature extraction in image processing and analysis. The general and more in-depth explanation can be found on `Wikipedia <https://en.wikipedia.org/wiki/Hough_transform>`_. The following explains the specifics of its usage in RASCAL. We will examine the settings for Hough transform, also known as *hyperparameters* and how it is used to perform wavelength calibration.

A version of this explanation with inline codes is available with the `Jupyter Notebook <https://github.com/jveitchmichaelis/rascal/blob/master/examples/Hough%20Transform.ipynb>`_ at the GitHub repository.

Spectrum and Lines
------------------

Before jumping into the transformation, we need some data. The image below is a 2D image of arc lines dispersed in the x-direction.

.. figure:: hough_transform_figure/hough_transform_01.png

The first 110 pixels are trimmed because it is blank. Then, the median of along the y direction is taken to represent the arc spectrum and the locations of the peaks are identified for Hough transform.

.. figure:: hough_transform_figure/hough_transform_02.png

This is what the arc should look like (taken from the Liverpool Telescope `SPRAT <http://telescope.livjm.ac.uk/TelInst/Inst/SPRAT/>`_ instrument page). Note the first sizable peak on the left is at 4500A, and the three small but clear peaks to the far right are 7807A, 7887A and 7967A (partly truncated):

.. figure:: hough_transform_figure/hough_transform_03.png


Accumulator
-----------

Now we instantiate a ``calibrator``. We can see from the above plot, and recognising some key lines that our range is around 4000 to 8000A. To visualise the process, first, we plot a few 2D Hough ``accumulators`` matrices with different values of the ``num_slopes`` -- 10, 50, 100 and 500, to generate all the possible Hough pairs (angle-interception pair). The ``accumulators`` look like the following for the 4 values of the slope resolution (clockwise from the top left).

.. figure:: hough_transform_figure/hough_transform_04.png

Where do the slopes (each vertical line) come from?

We've specified a spectral range, and we've provided the length of the spectrum (1024). Rascal guesses sensible values of the intercept depending on how confident your range guess is, e.g. the default of range_tolerance = 500A / 50nm will give 30000A to 4000A.

From this we know that the maximum slope must be:

``(max_wavelength - range_tolerance - min_intercept) / n_pixels``

and the minimum:

``(max_wavelength + range_tolerance - max_intercept) / n_pixels``

i.e. ``(8000 + 500 - 3500) / 1024`` about ``3.3`` and ``(8000 - 500 - 4500) / 1024`` about ``5.5``.

Note that we have chosen to enumerate over dispersion - the algorithm works by choosing a range of dispersion values to check, and finds the intercept values that supports them (i.e. we solve ``y = m * x + c`` for ``c`` given ``m``). We could equivalently search over a range of initial wavelengths and calculate what dispersions would support that.

Binning
-------

.. figure:: hough_transform_figure/hough_transform_05.png

.. figure:: hough_transform_figure/hough_transform_06.png

.. figure:: hough_transform_figure/hough_transform_07.png

.. figure:: hough_transform_figure/hough_transform_08.png

Goodness-of-fit
---------------

.. figure:: hough_transform_figure/hough_transform_09.png

Threshold setting
-----------------

.. figure:: hough_transform_figure/hough_transform_10.png

.. figure:: hough_transform_figure/hough_transform_11.png
