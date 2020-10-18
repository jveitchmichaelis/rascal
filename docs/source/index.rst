RASCAL Documentation
====================

Wavelength calibration is the proess of modelling a spectrometer such that every pixel can be mapped to a wavelength. This is normally perfomred by either amnuially matching spectral peaks to a line atlas, of using cross-correlation with a known lamp spectrum.

Manual calibration is tedious, particularly for arc lamps with many emission lines. Cross-correlation or template matching methods are often built for specific instruments and make assumptions about the long term stability of the calibration lamp.

**R**\ ANSAC- **A**\ ssisted **S**\ pectral **CAL**\ ibration aims to produce a fit model **automatically** from an arc lamp spectrum with only **minimal prior information**. RASCAL is inspired by the method of `Song (2018) <https://www.osapublishing.org/ao/abstract.cfm?uri=ao-57-24-6876>`_.

RASCAL is written in **Python 3** and has minimal dependencies. It has be tested for the `ASPIRED <https://aspired.readthedocs.io/en/latest/>`_ pipeline and with other scientific and commercial spectra.

.. note::
    **How fast is it?** RASCAL takes seconds to run.

    **How accurate do the initial conditions need to be?** We usually assume 10-20% uncertainty in the dispersion and spectral range.

    **What sources does it work with?** Anything. We have included the `NIST lines <https://physics.nist.gov/PhysRefData/ASD/lines_form.html>`_ by default.

    **What if I don't know the lamp?** Run RASCAL multiple times with different lammp options and inspect the outputs with the lowest errors.

Basic Usage
===========

The bare minimum example code to to get a wavelength calibration:

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

    # Get the spectral lines
    peaks, _ = find_peaks(spectrum)

    # Set up the Calibrator object
    c = Calibrator(peaks)

    # Load the Lines from library
    c.add_atlas(["Xe"])

    # Solve for the wavelength calibration
    best_polyfit_coefficient, rms, residual, peak_utilisation = c.fit()

    # Produce the diagnostic plot
    c.plot_fit(spectrum, best_polyfit_coefficient)


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

   installation/installation

.. toctree::
   :maxdepth: 2
   :caption: Behind the Scene

   background/summary
   background/houghtransform
   background/ransac

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorial/quickstart
   tutorial/lt-sprat
   tutorial/gemini-gmosls
   tutorial/wht-isis
   tutorial/keck-deimos
   tutorial/gtc-osiris

.. toctree::
   :maxdepth: 2
   :caption: List of Modules

   modules/calibrator
   modules/models
   modules/synthetic
   modules/util

License & Attribution
=====================

Copyright 2019-2020

If you make use of RASCAL in your work, please cite our paper
(`arXiv <https://arxiv.org/abs/1912.05883>`_,
`ADS <https://ui.adsabs.harvard.edu/abs/2019arXiv191205883V/abstract>`_,
`BibTeX <https://ui.adsabs.harvard.edu/abs/2019arXiv191205883V/exportcitation>`_).

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

