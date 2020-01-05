.. RASCAL documentation master file, created by
   sphinx-quickstart on Sun Jan  5 15:40:59 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

RASCAL Documentation
====================

The ``RASCAL`` package contains blah blah blah blah blah blah blah

Basic Usage
===========

The bare minimum example code to to get a wavelength calibration:

.. code-block:: python

    import numpy as np
    from scipy.signal import find_peaks
    from astropy.io import fits

    from rascal.calibrator import Calibrator
    from rascal.util import load_calibration_lines

    # Open the example file and get the median along the spectral direction
    spectrum = np.median(fits.open("./v_a_20190516_55_1_0_1.fits")[0].data)

    # Load the Lines from library
    atlas = load_calibration_lines(elements = ["Xe"])

    # Get the spectral lines
    peaks, _ = find_peaks(spectrum)

    # Set up the Calibrator object
    c = Calibrator(peaks, atlas)

    # Solve for the wavelength calibration
    best_p = c.fit()

    # Produce the diagnostic plot
    c.plot_fit(spectrum, best_p)


Some more complete examples are available in the :ref:`quickstart` tutorial.


How to Use This Guide
=====================

To start, you're probably going to need to follow the :ref:`installation` guide to
get RASCAL installed on your computer.
After you finish that, you can probably learn most of what you need from the
tutorials listed below (you might want to start with
:ref:`quickstart` and go from there).
If you need more details about specific functionality, the User Guide below
should have what you need.

We welcome bug reports, patches, feature requests, and other comments via `the GitHub
issue tracker <https://github.com/jveitchmichaelis/rascal/issues>`_.


User Guide
==========

.. toctree::
   :maxdepth: 2
   :caption: Installation

   installation

.. toctree::
   :maxdepth: 2
   :caption: Behind the Scene

   houghtransform
   ransac

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   quickstart
   lt-sprat
   lt-frodo
   keck-deimos
   ntt-efosc

.. toctree::
   :maxdepth: 2
   :caption: List of Modules

   calibrator
   models
   synthetic
   util

.. toctree::
   :maxdepth: 1
   :caption: API

   autoapi/index


License & Attribution
=====================

Copyright 2019-2020

If you make use of emcee in your work, please cite our paper
(`arXiv <https://arxiv.org/abs/1912.05883>`_,
`ADS <https://ui.adsabs.harvard.edu/abs/2019arXiv191205883V/abstract>`_,
`BibTeX <https://ui.adsabs.harvard.edu/abs/2019arXiv191205883V/exportcitation>`_).


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

