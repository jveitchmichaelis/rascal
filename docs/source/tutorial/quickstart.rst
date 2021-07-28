Quickstart
==========

To demonstrate a more custom-built general reduction, we are extending on the bare minimum example code on the homepage. Please bear in mind this does not include all the example configurability of RASCAL.

To begin, we need to import all the necessary libraries and load the spectrum:

.. code-block:: python

    import numpy as np
    from scipy.signal import find_peaks
    from astropy.io import fits

    from rascal.calibrator import Calibrator
    from rascal.util import refine_peaks

    # Open the example file
    spectrum2D = fits.open("filename.fits")[0].data

    # Get the median along the spectral direction
    spectrum = np.median(spectrum2D, axis=0)

The emission lines from the arc lamp spectrum have to be idenified before wavelength calibration is possible. RASCAL does not contain a peaking finding function, in this example we are using the one available with SciPy which returns the peak accurate to the nearest pixel. For that reason RASCAL has a `refine_peaks` function to get the peak positions at subpixel level by fitting Gaussians over a few pixels on either side of the peaks.

.. code-block:: python

    # Get the spectral lines
    peaks, _ = find_peaks(spectrum)
    peaks_refined = refine_peaks(peaks)

To start the calibration, we need to initialise the calibrator by passing in the peak locations. The **calibrator properties** can be set altogether (it can be set later or modify by using `set_properties`). The `num_pix` or `pixel_list` are necessary for the calibration because the calibrator is abstracted from the properties of the detector array, however, for a detector plane with multiple detectors that are separated by a fixed number of pixels, the wavelength calibration will be completely unsable without taken into account of the step functions intoduced by the chip gaps. The `num_pix` is also important for checking the monotonicity of the entire range of the fitted pixel-to-wavelength function.

.. code-block:: python

    # Set up the Calibrator object
    c = Calibrator(peaks_refined,
                   spectrum)
    c.set_calibrator_properties(num_pix=len(spectrum),
                                plotting_library='matplotlib',
                                log_level='info') 

Once the Calibrator is provided with the peaks, and optionally the arc spectrum, a diagnostic plot for the arc spectrum can be plotted with

.. code-block:: python

    c.plot_arc()

To distinguish from the Hough transform and fitting from the calibrator, in manufacturing term, the calibrator is the factory, the Hough transform is the pre-processing before entering production line, while the fitter is the machine. Therefore, apart from setting the calibrator properties, we also need to set the **Hough transform properties**, and the **RANSAC properties**.

.. code-block:: python

    c.set_hough_properties(num_slopes=10000,
                           xbins=1000,
                           ybins=1000,
                           min_wavelength=3500.,
                           max_wavelength=9000.,
                           range_tolerance=500.,
                           linearity_tolerance=50)

    c.set_ransac_properties(sample_size=5,
                            top_n_candidate=8,
                            filter_close=True)

The calibration still does not know what it is calibrating against, so we have to provide the arc lines or use the built-in library by providing the Chemical symbols.

.. code-block:: python

    # Load the Lines from library
    c.add_atlas(["Xe"],
                min_intensity=10,
                min_distance=10,
                constrain_poly=True)

With everything set, we can perform the Hough transform on the pixel-wavelength pairs

.. code-block:: python

    c.do_hough_transform()

Finally, we can do the fitting, there are still a few more parameters that were not configured in the `set_ransac_properties`. The distinction is that, RANSAC properties concern the parameter space and the sampling of the fit, while the fitting function only concerns the properties of the polynomial.

.. code-block:: python

    # Solve for the wavelength calibration
    (best_polyfit_coefficient, matched_peaks, matched_atlas, rms, residual,
     peak_utilisation, atlas_utilisation) = c.fit(max_tries=1000, polydeg=7)

Show the wavelength calibrated spectrum.

.. code-block:: python

    # Produce the diagnostic plot
    c.plot_fit(best_polyfit_coefficient,
               spectrum,
               plot_atlas=True,
               log_spectrum=False,
               tolerance=5.)

Show the parameter space in which the solution searching was carried out.

.. code-block:: python

    c.plot_search_space()
